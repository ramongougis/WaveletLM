"""salt_pilot.py — SALT rungs 1 & 2: build the store, then measure the ORACLE CEILING.

SALT (Scale-Addressed Lookup Table): for contexts where the frozen model got the next
token WRONG, store the CORRECT continuation, keyed by per-scale wavelet coefficients.
NOTE the direction: we store corrections, not errors — the error is the selection
criterion, the content is ground truth.

WHY THE ORACLE CEILING IS RUNG 2. It answers, cheaply and before any pipeline exists:
"if retrieval were perfect, how much is even available?" We look up each eval position
and ask whether the true next token is present ANYWHERE in the candidate set. If that
ceiling is low, the KEY is wrong and no gate tuning, interpolation weight, or index
cleverness can rescue it. Fail fast here (plans/salt.md, test ladder rung 2).

Design choices are measurement-driven (interpretability.md Findings 16-19):
  * rank on COARSE bands - fine bands encode current-token identity, which the
    last-token partition already matches exactly (F16/F17/F18)
  * cosine, PER-SCALE NORMALIZED - magnitude tracks surprise not content (F17), and
    the U-shaped gain profile would otherwise weight the metric 2-3x (F1)
  * partition by exact last token, then rank - Katz-style backoff logic; directly
    constrains the "similar context, different continuation" failure mode

Reported separately (they fail for different reasons, and conflating them hides which):
  coverage  - fraction of eval positions whose partition is non-empty
  oracle@k  - P(true next token in top-k candidates | partition non-empty)
  top1      - P(nearest neighbour's continuation is correct | partition non-empty)

Usage:
  python tools/interpretability/salt_pilot.py \
      --run_dir logs/wikitext-103_2026-07-15_10-53-46 --build_tokens 400000 --eval_tokens 20000
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from model import WaveletLM                      # noqa: E402
from train import load_and_encode_dataset        # noqa: E402
from tools.interpretability.scale_ablation import load_model   # noqa: E402


class _L:
    def log(self, m): print(m, flush=True)


def make_keys(C, scales, per_scale_norm=True):
    """C: (T,S,Cp) -> (T, len(scales)*Cp) unit-norm key. Per-scale normalize FIRST so
    the U-shaped gain profile does not silently weight the match (Finding 1)."""
    parts = []
    for s in scales:
        v = C[:, s, :]
        if per_scale_norm:
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        parts.append(v)
    K = torch.cat(parts, dim=-1)
    return K / K.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def sweep(model, data, n_tok, T, S, layer, scales, dev, label):
    """Forward over n_tok tokens; return keys, next-token, last-token, nll."""
    store, hooks = {}, []
    def mk(si):
        def h(_m, _i, out): store[si] = out.detach()[0]
        return h
    for si in range(S):
        hooks.append(model.layers[layer].decomp_norms[si].register_forward_hook(mk(si)))
    K, NXT, LST, NLL = [], [], [], []
    n_win = max(n_tok // T, 1)
    for w in range(n_win):
        s0 = w * T
        if s0 + T + 1 >= len(data):
            break
        x = data[s0:s0 + T].unsqueeze(0).to(dev).long()
        y = data[s0 + 1:s0 + T + 1].to(dev).long()
        store.clear()
        with torch.no_grad():
            logits, _ = model(x)
        C = torch.stack([store[s] for s in range(S)], dim=1)      # (T,S,Cp)
        lp = F.log_softmax(logits[0].float(), -1)
        nll = F.nll_loss(lp, y, reduction="none")
        K.append(make_keys(C.float(), scales).cpu())
        NXT.append(y.cpu()); LST.append(x[0].cpu()); NLL.append(nll.cpu())
        if (w + 1) % 200 == 0:
            print(f"  [{label}] {(w+1)*T:,}/{n_win*T:,} tokens", flush=True)
    for h in hooks: h.remove()
    return (torch.cat(K).numpy(), torch.cat(NXT).numpy(),
            torch.cat(LST).numpy(), torch.cat(NLL).numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--build_tokens", type=int, default=400000)
    p.add_argument("--eval_tokens", type=int, default=20000)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--scales", type=int, nargs="+", default=[0, 1], help="coarse bands (F16-18)")
    p.add_argument("--keep_pct", type=float, default=1.0, help="%% highest-loss kept")
    p.add_argument("--topk", type=int, nargs="+", default=[1, 5, 20])
    p.add_argument("--no_partition", action="store_true", help="ablate the last-token partition")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = torch.device(args.device)
    model, cfg = load_model(args.run_dir, dev); model.eval()
    T = cfg.get("block_size", 256); S = model.layers[0].scale_weights.numel()
    for blk in model.layers:
        if hasattr(blk, "decompose_bypass_cross_window"):
            blk.decompose_bypass_cross_window = False
    tr, va, _, _, _ = load_and_encode_dataset(
        {"dataset": cfg.get("dataset", "wikitext-103"), "tokenizer": "auto", "block_size": T}, _L())

    print(f"\n[SALT] layer {args.layer}, key scales {args.scales} of {S}, "
          f"keep top {args.keep_pct}% by loss")
    print(f"[build] sweeping {args.build_tokens:,} train tokens...")
    Kb, Nb, Lb, Eb = sweep(model, tr, args.build_tokens, T, S, args.layer, args.scales, dev, "build")

    thresh = np.percentile(Eb, 100 - args.keep_pct)
    sel = Eb >= thresh
    Ks, Ns, Ls = Kb[sel], Nb[sel], Lb[sel]
    kb = Ks.shape[1]
    print(f"[build] {len(Eb):,} positions -> kept {sel.sum():,} "
          f"({100*sel.mean():.2f}%, nll>={thresh:.2f}) | key dim {kb} | "
          f"store {Ks.nbytes/1e6:.1f} MB fp32 ({Ks.nbytes/2e6:.1f} MB fp16)")

    # partition index: last token -> row ids
    part = {}
    for i, t in enumerate(Ls):
        part.setdefault(int(t), []).append(i)
    part = {k: np.array(v) for k, v in part.items()}
    sizes = np.array([len(v) for v in part.values()])
    print(f"[build] partitions: {len(part):,} distinct last-tokens | "
          f"median {np.median(sizes):.0f}, p90 {np.percentile(sizes,90):.0f}, max {sizes.max()}")

    print(f"\n[eval] sweeping {args.eval_tokens:,} held-out val tokens...")
    Ke, Ne, Le, Ee = sweep(model, va, args.eval_tokens, T, S, args.layer, args.scales, dev, "eval")

    maxk = max(args.topk)
    hits = {k: 0 for k in args.topk}; top1 = 0; covered = 0
    for i in range(len(Ke)):
        cand = part.get(int(Le[i])) if not args.no_partition else np.arange(len(Ns))
        if cand is None or len(cand) == 0:
            continue
        covered += 1
        sims = Ks[cand] @ Ke[i]
        order = cand[np.argsort(-sims)[:maxk]]
        cont = Ns[order]
        if cont[0] == Ne[i]:
            top1 += 1
        for k in args.topk:
            if Ne[i] in cont[:k]:
                hits[k] += 1

    cov = covered / max(len(Ke), 1)
    print(f"\n=== RESULTS ({len(Ke):,} eval positions) ===")
    print(f"  coverage (partition non-empty) : {cov:6.2%}")
    if covered:
        print(f"  top1  | covered                 : {top1/covered:6.2%}   "
              f"(nearest neighbour's continuation is correct)")
        for k in args.topk:
            print(f"  oracle@{k:<3}| covered              : {hits[k]/covered:6.2%}   "
                  f"(true token anywhere in top-{k})")
        print(f"\n  ORACLE CEILING (unconditional) : {hits[maxk]/len(Ke):6.2%}")
        print(f"  -> this bounds ANY gate/interpolation/index built on this key.")
        print(f"  -> if low, the KEY is wrong; tuning downstream cannot rescue it.")
    if args.out:
        json.dump(dict(coverage=cov, covered=int(covered), n_eval=int(len(Ke)),
                       top1=top1 / max(covered, 1),
                       oracle={str(k): hits[k] / max(covered, 1) for k in args.topk},
                       store_rows=int(sel.sum()), key_dim=int(kb),
                       scales=args.scales, layer=args.layer),
                  open(args.out, "w"), indent=1)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
