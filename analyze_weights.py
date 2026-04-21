"""
Weight-storage-efficiency analysis of a WaveletLM checkpoint.

For each module/parameter group, reports:
  - Shape & param count
  - Distribution statistics (min, max, mean, std, percentiles)
  - Effective rank (for 2D+ weight matrices) as a fraction of full rank
  - Dead-weight fraction (|w| below a threshold)

Effective rank is computed from the SVD spectrum as the exponential of
Shannon entropy of the normalized singular values:
    r_eff = exp(H(σ²)) where σ² is normalized to sum to 1.
A low r_eff / rank_max ratio indicates the matrix is compressible.

Usage: python analyze_weights.py logs/wikitext-103_2026-04-19_13-16-24/best_model.pt
"""
import sys
import torch
from collections import defaultdict


def _safe_quantile(t, q, sample_cap=1_000_000):
    """pytorch's quantile has a ~16M-element limit. Subsample large tensors."""
    flat = t.reshape(-1)
    if flat.numel() > sample_cap:
        idx = torch.randint(0, flat.numel(), (sample_cap,))
        flat = flat[idx]
    return flat.quantile(q).item()


def analyze_tensor(name, w, dead_thresh=1e-4):
    """Return a dict of stats for a single tensor."""
    w = w.detach().float().cpu()
    numel = w.numel()
    absw = w.abs()
    stats = {
        'name': name,
        'shape': tuple(w.shape),
        'numel': numel,
        'mem_mib': numel * 4 / 1024 / 1024,  # at fp32
        'min': w.min().item(),
        'max': w.max().item(),
        'mean': w.mean().item(),
        'std': w.std().item(),
        'abs_median': _safe_quantile(absw, 0.5),
        'abs_p99': _safe_quantile(absw, 0.99),
        'abs_p999': _safe_quantile(absw, 0.999),
        'dead_frac': (absw < dead_thresh).float().mean().item(),
    }

    if w.dim() >= 2:
        # Flatten into 2D for SVD
        w2d = w.reshape(w.shape[0], -1)
        r_max = min(w2d.shape)
        if r_max > 1 and r_max <= 4096:
            try:
                s = torch.linalg.svdvals(w2d)
                s2 = (s ** 2)
                s2_norm = s2 / (s2.sum() + 1e-12)
                entropy = -(s2_norm * (s2_norm + 1e-20).log()).sum().item()
                r_eff = float(torch.exp(torch.tensor(entropy)).item())
                stats['r_eff'] = r_eff
                stats['r_max'] = r_max
                stats['rank_ratio'] = r_eff / r_max
            except Exception as e:
                stats['r_eff'] = None
                stats['r_max'] = r_max
                stats['rank_ratio'] = None
        else:
            stats['r_eff'] = None
            stats['r_max'] = r_max
            stats['rank_ratio'] = None
    else:
        stats['r_eff'] = None
        stats['r_max'] = None
        stats['rank_ratio'] = None

    return stats


def group_key(name):
    """Classify a parameter name into a module group."""
    lc = name.lower()
    if 'token_embedding' in lc:
        return 'embedding'
    if 'lm_head' in lc:
        return 'lm_head'
    if 'final_ln' in lc or 'layernorm' in lc or '.norm' in lc or '.ln' in lc:
        return 'layernorm'
    if 'lifting' in lc:
        if 'predict' in lc:
            return 'lifting.predict'
        if 'update' in lc:
            return 'lifting.update'
        return 'lifting.other'
    if 'scale_mixers' in lc or 'scale_routing' in lc:
        if 'proj_in' in lc or 'proj_out' in lc:
            return 'mixer.proj'
        if 'gate' in lc:
            return 'mixer.gate'
        if 'mixer' in lc and 'weight' in lc:
            return 'mixer.weight'
        return 'mixer.other'
    if 'ffwd' in lc or 'ffn' in lc:
        return 'mlp'
    if 'pkm' in lc and 'fwpkm' not in lc:
        if 'key' in lc:
            return 'pkm.keys'
        if 'value' in lc:
            return 'pkm.values'
        if 'query' in lc or 'proj' in lc:
            return 'pkm.query_proj'
        return 'pkm.other'
    if 'fwpkm' in lc:
        if 'key' in lc:
            return 'fwpkm.keys'
        if 'value' in lc:
            return 'fwpkm.values'
        if 'query' in lc or 'proj' in lc:
            return 'fwpkm.query_proj'
        return 'fwpkm.other'
    if 'cross_layer_mix' in lc:
        return 'decompose_bypass.cross_layer_mix'
    if 'history_gains' in lc:
        return 'decompose_bypass.history_gains'
    if 'scale_weights' in lc:
        return 'per_scale_weights'
    if 'residual_alpha' in lc or 'alpha_' in lc or lc.endswith('.alpha'):
        return 'residual_alphas'
    if 'wavelet_crawl' in lc or 'crawl' in lc:
        return 'wavelet_crawl'
    if 'ema_gate' in lc:
        return 'decompose_bypass.ema_gate'
    return 'other'


def main(ckpt_path):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # The checkpoint can be a plain state_dict or wrapped in {'model_state_dict': ...}
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
    else:
        sd = ckpt
    # Strip torch.compile prefix
    sd = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in sd.items()}

    print(f"Loaded {len(sd)} tensors\n")

    # Per-tensor analysis
    per_tensor = [analyze_tensor(k, v) for k, v in sd.items()]

    # Group by module type
    groups = defaultdict(list)
    for s in per_tensor:
        groups[group_key(s['name'])].append(s)

    # Print per-group summary
    print(f"{'='*110}")
    print(f"{'group':<34}  {'numel':>13}  {'MiB (fp32)':>10}  {'|w| p50':>10}  {'|w| p99':>10}  {'dead%':>6}  {'rank%':>6}")
    print(f"{'='*110}")
    group_totals = {}
    for g, items in sorted(groups.items(), key=lambda x: -sum(s['numel'] for s in x[1])):
        total_numel = sum(s['numel'] for s in items)
        total_mib = sum(s['mem_mib'] for s in items)
        all_abs_median = sum(s['abs_median'] * s['numel'] for s in items) / total_numel
        all_abs_p99 = sum(s['abs_p99'] * s['numel'] for s in items) / total_numel
        dead_frac = sum(s['dead_frac'] * s['numel'] for s in items) / total_numel
        ranked = [s for s in items if s.get('rank_ratio') is not None]
        if ranked:
            rank_ratio = sum(s['rank_ratio'] * s['numel'] for s in ranked) / sum(s['numel'] for s in ranked)
            rank_str = f"{rank_ratio*100:5.1f}%"
        else:
            rank_str = "  n/a"
        print(f"{g:<34}  {total_numel:>13,}  {total_mib:>10.2f}  {all_abs_median:>10.5f}  {all_abs_p99:>10.5f}  {dead_frac*100:>5.1f}%  {rank_str:>6}")
        group_totals[g] = (total_numel, total_mib, all_abs_median, all_abs_p99, dead_frac, rank_str)

    total_numel = sum(s['numel'] for s in per_tensor)
    total_mib = sum(s['mem_mib'] for s in per_tensor)
    print(f"{'-'*110}")
    print(f"{'TOTAL':<34}  {total_numel:>13,}  {total_mib:>10.2f}")
    print()

    # Per-group interpretation
    print(f"{'='*110}")
    print("COMPRESSION-OPPORTUNITY NOTES")
    print(f"{'='*110}")
    for g, (numel, mib, med, p99, dead, rank) in group_totals.items():
        notes = []
        if dead > 0.50:
            notes.append(f"{dead*100:.0f}% of values below 1e-4 — aggressive pruning candidate")
        elif dead > 0.20:
            notes.append(f"{dead*100:.0f}% dead — moderate pruning candidate")
        if 'rank%' not in rank and rank != "  n/a":
            rr = float(rank.replace('%','').strip())
            if rr < 40:
                notes.append(f"effective rank {rr:.0f}% of max — strong low-rank candidate")
            elif rr < 70:
                notes.append(f"effective rank {rr:.0f}% — some low-rank redundancy")
        if notes:
            print(f"  {g:<34}  {'; '.join(notes)}")
    print()

    # Per-tensor detail for the biggest 30 tensors
    print(f"{'='*110}")
    print("TOP 30 TENSORS BY PARAM COUNT")
    print(f"{'='*110}")
    top = sorted(per_tensor, key=lambda s: -s['numel'])[:30]
    for s in top:
        rank_str = f"{s['rank_ratio']*100:.1f}%" if s.get('rank_ratio') is not None else "n/a"
        print(f"  {s['name']:<60} {str(s['shape']):<20} p99|w|={s['abs_p99']:.4f}  rank_ratio={rank_str}  dead={s['dead_frac']*100:.1f}%")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/wikitext-103_2026-04-19_13-16-24/best_model.pt"
    main(path)
