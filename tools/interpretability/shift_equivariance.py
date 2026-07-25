"""shift_equivariance.py — is the learned decomposition shift-equivariant?

WHY THIS GATES EVERYTHING ELSE. Classical dyadic wavelet transforms are NOT
shift-invariant (the reason stationary / a-trous variants exist). If sliding the
window by one token changes the coefficients at positions whose causal history is
otherwise unchanged, then a coefficient is not a stable code for "this token in
this context" — it is partly an artifact of grid alignment. That would put a
caveat on the census, on Study 2's channel statistics, and on any claim of the
form "channel X fires at boundaries" (boundaries correlate with position, and
position correlates with alignment).

THE DISCRIMINATOR. A dyadic scheme is alignment-sensitive with period 2^level.
So compare the SAME absolute positions under:
  * power-of-two shifts (2, 4, 8, 16) — alignment PRESERVED at the coarser levels
  * odd shifts (1, 3, 5)              — alignment BROKEN at every level
If odd-shift deltas >> pow2-shift deltas, the transform is dyadically shift-variant.
If all deltas are comparably small, it behaves as an undecimated (shift-equivariant)
transform on this axis.

CONFOUND, controlled for: window `shift` drops the first `shift` tokens, so a
shifted window genuinely has less left-context. That is a real difference, not an
artifact. We therefore compare only positions at least --margin tokens from the
window start, where the dropped tokens are far outside the local receptive field,
and we report the pow2-vs-odd CONTRAST rather than the absolute delta — the
context loss is identical for equal shift sizes, so the contrast isolates alignment.

Usage:
  python tools/interpretability/shift_equivariance.py --run_dir logs/wikitext-103_2026-07-15_10-53-46
"""
import argparse, json, os, sys
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from model import WaveletLM                      # noqa: E402
from train import load_and_encode_dataset        # noqa: E402
from tools.interpretability.scale_ablation import load_model   # noqa: E402


class _L:
    def log(self, m): print(m, flush=True)


def capture(model, x, layers, S):
    """Run one window, return {layer: (T, S, Cp)} of post-decomp-norm coefficients."""
    store, hooks = {}, []
    def mk(li, si):
        def h(_m, _i, out):
            store.setdefault(li, {})[si] = out.detach()[0]      # (T, Cp), batch 0
        return h
    for li in layers:
        blk = model.layers[li]
        for si in range(S):
            hooks.append(blk.decomp_norms[si].register_forward_hook(mk(li, si)))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return {li: torch.stack([store[li][s] for s in range(S)], dim=1) for li in store}  # (T,S,Cp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--shifts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 16])
    p.add_argument("--margin", type=int, default=64,
                   help="ignore positions within this many tokens of the window start")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 4, 9])
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = torch.device(args.device)
    model, cfg = load_model(args.run_dir, dev); model.eval()
    T = cfg.get("block_size", 256)
    S = model.layers[0].scale_weights.numel()
    # cross-window bypass state would leak between the reference and shifted runs
    for blk in model.layers:
        if hasattr(blk, "decompose_bypass_cross_window"):
            blk.decompose_bypass_cross_window = False
    if hasattr(model, "_persistent_semantic_state"):
        model._persistent_semantic_state = None

    _, va, _, _, _ = load_and_encode_dataset(
        {"dataset": cfg.get("dataset", "wikitext-103"), "tokenizer": "auto", "block_size": T}, _L())
    maxs = max(args.shifts)
    g = torch.Generator().manual_seed(7)
    print(f"[setup] T={T} S={S} layers={args.layers} margin={args.margin} "
          f"trials={args.trials}\n")

    acc = {sh: {li: [] for li in args.layers} for sh in args.shifts}
    for t in range(args.trials):
        start = int(torch.randint(0, len(va) - T - maxs - 2, (1,), generator=g).item())
        xref = va[start:start + T].unsqueeze(0).to(dev).long()
        ref = capture(model, xref, args.layers, S)
        for sh in args.shifts:
            xs = va[start + sh:start + sh + T].unsqueeze(0).to(dev).long()
            cur = capture(model, xs, args.layers, S)
            for li in args.layers:
                # absolute position a  ->  ref index a, shifted index a-sh
                a0 = args.margin + maxs
                A = ref[li][a0:T]                      # (n, S, Cp)
                B = cur[li][a0 - sh:T - sh]
                num = (A - B).abs().mean(dim=(0, 2))   # per scale
                den = A.abs().mean(dim=(0, 2)).clamp_min(1e-9)
                acc[sh][li].append((num / den).cpu().numpy())

    rows = []
    print(f"{'shift':>6} {'kind':>6} {'layer':>6} " + " ".join(f"{'s'+str(s):>7}" for s in range(S)) + "   mean")
    for sh in args.shifts:
        kind = "pow2" if (sh & (sh - 1)) == 0 else ("odd" if sh % 2 else "even")
        for li in args.layers:
            v = np.mean(np.stack(acc[sh][li]), axis=0)
            print(f"{sh:>6} {kind:>6} {li:>6} " + " ".join(f"{x:7.3f}" for x in v)
                  + f"   {v.mean():.3f}")
            rows.append(dict(shift=sh, kind=kind, layer=li,
                             per_scale=[float(x) for x in v], mean=float(v.mean())))
        print()

    pw = [r["mean"] for r in rows if r["kind"] == "pow2"]
    od = [r["mean"] for r in rows if r["kind"] == "odd"]
    print(f"mean relative delta  |  pow2 shifts: {np.mean(pw):.3f}   odd shifts: {np.mean(od):.3f}")
    ratio = np.mean(od) / max(np.mean(pw), 1e-9)
    print(f"odd / pow2 contrast  : {ratio:.2f}x")
    print("  >1.5x => DYADICALLY SHIFT-VARIANT (alignment matters; coefficients are not a")
    print("           position-independent code, and position-correlated claims need a control)")
    print("  ~1.0x => behaves shift-EQUIVARIANT on this axis (undecimated-like); coefficients")
    print("           are a stable code and the flow/surprise study is well-posed")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1); print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
