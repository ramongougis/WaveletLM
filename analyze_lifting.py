"""Compression analysis for trained lifting predict/update networks.

Loads a WaveletLM checkpoint and analyzes the per-level lifting MLP weights
for four structural patterns that justify parameter compression:

  1. Low-rank structure (SVD spectrum + effective rank at 95% / 99% / 99.9%
     spectral energy thresholds). Suggests Linear(C,C) → Linear(C,r)·Linear(r,C)
     replacement at meaningful r.
  2. Cross-level similarity (Frobenius / cosine distance matrices). Suggests
     weight sharing across "level groups" rather than per-level distinct nets.
  3. Diagonal-dominance (fraction of Frobenius energy on the diagonal).
     Suggests Diagonal + low-rank parameterization (W = D + U·V^T).
  4. Magnitude distribution percentiles. Suggests prunable sparsity threshold.

Output: tables to stdout with concrete compression recommendations and
estimated parameter savings. CPU-only, runs in 1–10 min depending on levels
count (each SVD on 2048×2048 takes ~5-10s on a modern CPU).

Usage:
    python analyze_lifting.py
    python analyze_lifting.py --checkpoint logs/wikitext-103_2026-05-03_02-13-07/best_model.pt
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import torch


# ============================================================================
# Checkpoint loading
# ============================================================================

def find_latest_checkpoint() -> str | None:
    candidates = []
    for root, _dirs, files in os.walk("logs"):
        if "best_model.pt" in files:
            candidates.append(os.path.join(root, "best_model.pt"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def load_state_dict(ckpt_path: str) -> dict:
    """Unwrap the model state dict (handles both `model_state` and `model`
    top-level key conventions, plus the bare-state-dict case)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")
    for key in ("model_state", "model"):
        inner = ckpt.get(key)
        if isinstance(inner, dict) and inner:
            return inner
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError(f"Could not unwrap state dict from {ckpt_path}")


# ============================================================================
# Weight collection
# ============================================================================

WEIGHT_PATTERN = re.compile(
    r"^(?P<prefix>.*?lifting_(?:wavelet|reconstruct\.decompose))"
    r"\.(?P<role>predict_nets|update_nets)"
    r"\.(?P<level>\d+)"
    r"\.(?P<lin>\d+)"
    r"\.weight$"
)


def collect_lifting_weights(state_dict: dict) -> dict:
    """{prefix: {role: {level: {linear_idx: tensor}}}}"""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for key, val in state_dict.items():
        m = WEIGHT_PATTERN.match(key)
        if not m:
            continue
        grouped[m.group("prefix")][m.group("role")][int(m.group("level"))][int(m.group("lin"))] = val
    return grouped


def detect_shared_tensors(grouped: dict) -> list[tuple[str, str]]:
    """Return list of (key1, key2) pairs where both refer to the same tensor
    storage — indicates PyTorch parameter sharing across model paths."""
    seen = {}  # data_ptr → first key seen
    pairs = []
    for prefix, roles in grouped.items():
        for role, levels in roles.items():
            for L, lins in levels.items():
                for lin, tensor in lins.items():
                    key = f"{prefix}.{role}.{L}.{lin}.weight"
                    ptr = tensor.data_ptr()
                    if ptr in seen:
                        pairs.append((seen[ptr], key))
                    else:
                        seen[ptr] = key
    return pairs


# ============================================================================
# Single-matrix analyses
# ============================================================================

def effective_rank(s: torch.Tensor, energy_threshold: float) -> int:
    sq = s.float() ** 2
    total = sq.sum().clamp_min(1e-30)
    cum_energy = sq.cumsum(0) / total
    return int((cum_energy < energy_threshold).sum().item()) + 1


def distance_from_identity(W: torch.Tensor) -> float | None:
    if W.shape[0] != W.shape[1]:
        return None
    eye = torch.eye(W.shape[0], dtype=torch.float32)
    return float((W.float() - eye).norm() / eye.norm())


def diagonal_energy_fraction(W: torch.Tensor) -> float | None:
    if W.shape[0] != W.shape[1]:
        return None
    Wf = W.float()
    diag_e = (Wf.diag() ** 2).sum().item()
    total_e = (Wf ** 2).sum().item()
    return diag_e / total_e if total_e > 0 else 0.0


def magnitude_percentiles(W: torch.Tensor) -> dict:
    abs_w = W.float().abs().flatten()
    return {
        "p50":   float(abs_w.quantile(0.50)),
        "p90":   float(abs_w.quantile(0.90)),
        "p99":   float(abs_w.quantile(0.99)),
        "p99.9": float(abs_w.quantile(0.999)),
        "max":   float(abs_w.max()),
    }


def analyze_matrix(W: torch.Tensor) -> dict:
    Wf = W.float()
    s = torch.linalg.svdvals(Wf)
    return {
        "shape": tuple(W.shape),
        "params": W.numel(),
        "rank_full": int(min(W.shape)),
        "rank_95":   effective_rank(s, 0.95),
        "rank_99":   effective_rank(s, 0.99),
        "rank_999":  effective_rank(s, 0.999),
        "frob_norm": float(Wf.norm()),
        "id_dist":   distance_from_identity(W),
        "diag_frac": diagonal_energy_fraction(W),
        "magnitude": magnitude_percentiles(W),
        "top_5_singular": [float(v) for v in s[:5].tolist()],
    }


# ============================================================================
# Cross-level similarity
# ============================================================================

def pairwise_similarity(matrices: dict) -> dict:
    levels = sorted(matrices.keys())
    n = len(levels)
    frob = torch.zeros(n, n)
    cos = torch.zeros(n, n)
    flat = {L: matrices[L].float().flatten() for L in levels}
    norms = {L: flat[L].norm().clamp_min(1e-30) for L in levels}
    for i, li in enumerate(levels):
        for j, lj in enumerate(levels):
            diff = flat[li] - flat[lj]
            frob[i, j] = diff.norm() / norms[li]
            cos[i, j] = torch.dot(flat[li], flat[lj]) / (norms[li] * norms[lj])
    return {"levels": levels, "frob_norm_distance": frob, "cosine_similarity": cos}


# ============================================================================
# Output
# ============================================================================

def print_per_matrix_table(prefix: str, role: str, lin_idx: int, results: dict):
    print(f"\n  {prefix}.{role}.[L].{lin_idx}.weight  ({len(results)} levels)")
    print(f"  {'L':>3}  {'shape':>11}  {'r95':>4}  {'r99':>4}  {'r99.9':>5}  "
          f"{'rfull':>5}  {'||W||F':>9}  {'id_dist':>8}  {'diag%':>7}  "
          f"{'p99 mag':>9}  top-5 σ")
    for L in sorted(results.keys()):
        r = results[L]
        sh = f"{r['shape'][0]}×{r['shape'][1]}"
        idd = f"{r['id_dist']:>8.4f}" if r['id_dist'] is not None else "     n/a"
        df = f"{r['diag_frac']*100:>6.2f}%" if r['diag_frac'] is not None else "    n/a"
        sigs = " ".join(f"{v:.3f}" for v in r["top_5_singular"])
        print(f"  {L:>3}  {sh:>11}  {r['rank_95']:>4}  {r['rank_99']:>4}  "
              f"{r['rank_999']:>5}  {r['rank_full']:>5}  {r['frob_norm']:>9.2f}  "
              f"{idd}  {df}  {r['magnitude']['p99']:>9.4f}  {sigs}")


def print_similarity(prefix: str, role: str, lin_idx: int, sim: dict):
    levels = sim["levels"]
    print(f"\n  {prefix}.{role}.[L].{lin_idx}.weight — cross-level similarity")
    print("  Cosine similarity (1.0=identical direction):")
    print("       " + " ".join(f"L={L:<2}  " for L in levels))
    for i, L in enumerate(levels):
        row = f"  L={L:<2} " + " ".join(f"{sim['cosine_similarity'][i, j].item():>6.3f} " for j in range(len(levels)))
        print(row)
    print("  Normalized Frobenius distance (0.0=identical):")
    print("       " + " ".join(f"L={L:<2}  " for L in levels))
    for i, L in enumerate(levels):
        row = f"  L={L:<2} " + " ".join(f"{sim['frob_norm_distance'][i, j].item():>6.3f} " for j in range(len(levels)))
        print(row)


def emit_recommendations(grouped: dict, all_results: dict, all_similarities: dict, total_params: int):
    print("\n" + "=" * 76)
    print("  COMPRESSION RECOMMENDATIONS")
    print("=" * 76)

    recommendations = []  # list of (priority, msg, est_params_saved)

    for (prefix, role, lin_idx), per_level in all_results.items():
        n_levels = len(per_level)
        sim = all_similarities.get((prefix, role, lin_idx))

        # Sum of params across levels for this (role, linear) cell
        cell_params = sum(r["params"] for r in per_level.values())

        # 1. Low-rank: if rank_99 << rank_full at most levels, low-rank decomp pays
        ranks_99 = [r["rank_99"] for r in per_level.values()]
        ranks_full = [r["rank_full"] for r in per_level.values()]
        avg_rank_99 = sum(ranks_99) / len(ranks_99)
        avg_rank_full = sum(ranks_full) / len(ranks_full)
        rank_compression = avg_rank_99 / avg_rank_full
        if rank_compression < 0.5:
            # Compute potential savings: Linear(C,C) → Linear(C,r)·Linear(r,C)
            C = per_level[next(iter(per_level))]["shape"][0]
            r = int(avg_rank_99)
            old_params = C * C
            new_params = 2 * C * r
            saved_per_matrix = old_params - new_params
            saved_total = saved_per_matrix * n_levels
            recommendations.append((
                1,
                f"[{prefix}.{role}.*.{lin_idx}] LOW-RANK (avg rank99 ≈ {avg_rank_99:.0f} of {avg_rank_full}, "
                f"{rank_compression:.0%} of full rank)\n"
                f"   → Replace each Linear({C},{C}) with Linear({C},{r}) → Linear({r},{C})\n"
                f"   → Per-matrix: {old_params/1e6:.2f}M → {new_params/1e6:.2f}M params "
                f"({100*new_params/old_params:.0f}% of original)\n"
                f"   → Total cell savings: ~{saved_total/1e6:.2f}M params ({n_levels} levels)",
                saved_total,
            ))

        # 2. Cross-level similarity
        if sim is not None and n_levels >= 2:
            cos_mat = sim["cosine_similarity"]
            n = len(sim["levels"])
            off_diag = [cos_mat[i, j].item() for i in range(n) for j in range(n) if i != j]
            if off_diag:
                avg_off = sum(off_diag) / len(off_diag)
                max_off = max(off_diag)
                min_off = min(off_diag)
                # If any pair is very similar, share-able
                if max_off > 0.95:
                    # Estimate saving as if 1 shared net replaces all levels
                    C = per_level[next(iter(per_level))]["shape"][0]
                    saved = (n - 1) * C * C
                    recommendations.append((
                        1,
                        f"[{prefix}.{role}.*.{lin_idx}] HIGH similarity (max cos {max_off:.3f}, "
                        f"avg off-diag {avg_off:.3f}, min {min_off:.3f})\n"
                        f"   → Some levels are nearly identical — try sharing weights across "
                        f"the most-similar level group\n"
                        f"   → Best-case savings: ~{saved/1e6:.2f}M params if all {n} levels share",
                        saved,
                    ))
                elif avg_off > 0.5:
                    C = per_level[next(iter(per_level))]["shape"][0]
                    saved = (n // 2) * C * C  # crude estimate: 2-3 groups
                    recommendations.append((
                        2,
                        f"[{prefix}.{role}.*.{lin_idx}] MODERATE similarity (avg cos {avg_off:.3f})\n"
                        f"   → Group-wise sharing (2-3 groups across {n} levels) likely viable\n"
                        f"   → Estimated savings: ~{saved/1e6:.2f}M params",
                        saved,
                    ))

        # 3. Diagonal-dominance
        diag_fracs = [r["diag_frac"] for r in per_level.values() if r["diag_frac"] is not None]
        if diag_fracs:
            avg_diag = sum(diag_fracs) / len(diag_fracs)
            # Random matrix has diag_frac ≈ 1/N; significant deviation suggests structure
            C = per_level[next(iter(per_level))]["shape"][0]
            random_baseline = 1.0 / C
            if avg_diag > 50 * random_baseline:  # 50× above random
                recommendations.append((
                    2,
                    f"[{prefix}.{role}.*.{lin_idx}] DIAGONAL-DOMINANT "
                    f"(avg diag energy {avg_diag*100:.2f}%, vs random baseline {random_baseline*100:.4f}%)\n"
                    f"   → Try W ≈ D + U·V^T parameterization (diagonal + low-rank correction)\n"
                    f"   → ~{(2*C + 4*C*16)/1e3:.0f}K params per matrix at r=16 vs {C*C/1e6:.2f}M for full",
                    0,  # rough estimate, depends on chosen r
                ))

        # 4. Pruning potential (high p50 to p99 ratio = many small weights)
        ratios = [r["magnitude"]["p99"] / max(r["magnitude"]["p50"], 1e-30) for r in per_level.values()]
        avg_ratio = sum(ratios) / len(ratios)
        if avg_ratio > 20:  # p99 magnitude is 20× the median → very long-tailed
            recommendations.append((
                3,
                f"[{prefix}.{role}.*.{lin_idx}] LONG-TAIL magnitude (p99/p50 ratio {avg_ratio:.1f}×)\n"
                f"   → Most weights are very small; magnitude pruning at the median "
                f"would zero ~50% of entries (sparse storage win, requires sparse-matmul kernel)",
                0,
            ))

    # Sort by priority (lower number = higher priority) then by est savings
    recommendations.sort(key=lambda x: (x[0], -x[2]))
    if not recommendations:
        print("\n  No strong compression patterns detected. The weights look generic.")
    else:
        for (priority, msg, _) in recommendations:
            tag = {1: "HIGH PRIORITY", 2: "MEDIUM PRIORITY", 3: "LOW PRIORITY"}[priority]
            print(f"\n  [{tag}] {msg}")

    print("\n" + "=" * 76)
    print(f"  Total lifting weight params analyzed: {total_params/1e6:.2f}M")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        sys.exit(f"No checkpoint at: {ckpt_path or '(none found under logs/)'}")

    print(f"=== Lifting weights compression analysis ===")
    print(f"  Checkpoint: {ckpt_path}")

    sd = load_state_dict(ckpt_path)
    grouped = collect_lifting_weights(sd)

    if not grouped:
        sys.exit("No lifting predict/update networks found in checkpoint.")

    shared_pairs = detect_shared_tensors(grouped)
    if shared_pairs:
        print(f"\n  Detected {len(shared_pairs)} pair(s) of tied tensors "
              f"(parameter sharing in PyTorch). Showing first 5:")
        for k1, k2 in shared_pairs[:5]:
            print(f"    {k1}\n      ≡ {k2}")
        if len(shared_pairs) > 5:
            print(f"    ... and {len(shared_pairs)-5} more")
        print("  Tied tensors are analyzed once per logical entry; counted toward")
        print("  total only at their first appearance.")

    all_results = {}        # (prefix, role, lin_idx) → {level: result}
    all_similarities = {}   # (prefix, role, lin_idx) → similarity dict
    total_params = 0
    seen_ptrs = set()

    for prefix in sorted(grouped.keys()):
        print(f"\n{'='*76}")
        print(f"  Path: {prefix}")
        print(f"{'='*76}")

        for role in ("predict_nets", "update_nets"):
            if role not in grouped[prefix]:
                continue
            level_dict = grouped[prefix][role]
            n_levels = len(level_dict)
            lin_indices = sorted(set(li for L in level_dict for li in level_dict[L]))
            print(f"\n  {role}: {n_levels} levels, Linear indices {lin_indices}")

            for lin_idx in lin_indices:
                per_level = {}
                matrices_for_pairwise = {}
                for L in sorted(level_dict.keys()):
                    if lin_idx in level_dict[L]:
                        W = level_dict[L][lin_idx]
                        per_level[L] = analyze_matrix(W)
                        matrices_for_pairwise[L] = W
                        if W.data_ptr() not in seen_ptrs:
                            total_params += W.numel()
                            seen_ptrs.add(W.data_ptr())

                all_results[(prefix, role, lin_idx)] = per_level
                print_per_matrix_table(prefix, role, lin_idx, per_level)

                if len(matrices_for_pairwise) > 1:
                    sim = pairwise_similarity(matrices_for_pairwise)
                    all_similarities[(prefix, role, lin_idx)] = sim
                    print_similarity(prefix, role, lin_idx, sim)

    emit_recommendations(grouped, all_results, all_similarities, total_params)


if __name__ == "__main__":
    main()
