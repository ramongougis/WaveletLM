"""group_ablation.py — Study 3b: is depth REDUNDANT or DISPENSABLE?

Study 3 (Finding 13) found that removing any single (layer, scale) band costs
almost nothing outside layer 0. That is evidence of REDUNDANCY. It is NOT evidence
that the deep stack is idle — a redundantly-coded ensemble is cheap to perturb and
expensive to delete. Single-point ablation cannot tell those apart; group ablation
can, and the distinction decides a real architecture question ("would one wide
layer do?").

Three modes:
  layer      zero ALL scales at one layer            -> is this layer's spectral
                                                        computation doing anything?
  scale      zero one scale across ALL layers        -> is this band needed anywhere?
  cumulative zero all scales for layers k..L-1       -> the decisive one: if removing
                                                        the deep stack is cheap, depth
                                                        really is dispensable; if damage
                                                        grows super-linearly, the layers
                                                        were redundant-but-necessary.

CAVEAT (stated because it bounds the claim): zeroing `scale_weights` removes a
layer's SPECTRAL contribution, but the residual stream, decompose-bypass and norms
still run. So "layer ablated" means "its wavelet path contributes nothing", i.e. the
layer degenerates toward a pass-through — a good proxy for "would fewer layers do?",
not a literal layer deletion.

Usage:
  python tools/interpretability/group_ablation.py --run_dir logs/wikitext-103_2026-07-15_10-53-46
"""
import argparse, json, os, sys
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from model import WaveletLM                      # noqa: E402
from train import load_and_encode_dataset        # noqa: E402
from tools.interpretability.scale_ablation import load_model, evaluate, NATS_PER_BPB  # noqa: E402


class _L:
    def log(self, m): print(m, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = torch.device(args.device)
    model, cfg = load_model(args.run_dir, dev); model.eval()
    T = cfg.get("block_size", 256)
    _, va, _, _, _ = load_and_encode_dataset(
        {"dataset": cfg.get("dataset", "wikitext-103"), "tokenizer": "auto", "block_size": T}, _L())
    g = torch.Generator().manual_seed(1234)
    n_win = max(args.tokens // T, args.batch)
    starts = torch.randint(0, len(va) - T - 1, (n_win,), generator=g)
    batches = []
    for i in range(0, n_win, args.batch):
        sel = starts[i:i + args.batch]
        if len(sel) < args.batch:
            break
        batches.append((torch.stack([va[s:s + T] for s in sel]).to(dev).long(),
                        torch.stack([va[s + 1:s + T + 1] for s in sel]).to(dev).long()))
    base = evaluate(model, batches)
    L = len(model.layers); S = model.layers[0].scale_weights.numel()
    print(f"[eval] {len(batches)*args.batch*T:,} tokens | [base] {base:.4f} nats | L={L} S={S}\n")

    orig = [blk.scale_weights.detach().clone() for blk in model.layers]

    def restore():
        with torch.no_grad():
            for blk, o in zip(model.layers, orig):
                blk.scale_weights.copy_(o)

    def zero(cells):
        with torch.no_grad():
            for (li, si) in cells:
                model.layers[li].scale_weights[si] = 0.0

    out = {"base_loss": base, "layer": [], "scale": [], "cumulative": []}

    print("=== MODE: whole-layer (zero all scales at layer L) ===")
    for li in range(L):
        restore(); zero([(li, s) for s in range(S)])
        d = (evaluate(model, batches) - base) / NATS_PER_BPB
        out["layer"].append(dict(layer=li, dBPB=d))
        print(f"  layer {li:>2} spectral path OFF -> dBPB {d:+.4f}")
    restore()

    print("\n=== MODE: cross-layer scale (zero scale s at ALL layers) ===")
    for si in range(S):
        restore(); zero([(li, si) for li in range(L)])
        d = (evaluate(model, batches) - base) / NATS_PER_BPB
        out["scale"].append(dict(scale=si, dBPB=d))
        print(f"  scale s{si} OFF everywhere    -> dBPB {d:+.4f}")
    restore()

    print("\n=== MODE: cumulative depth (zero layers k..%d) — THE DECISIVE ONE ===" % (L - 1))
    print("  if this stays cheap, depth is dispensable; if it explodes, the layers")
    print("  were redundant-but-necessary (single-ablation cheapness was misleading)")
    for k in range(L - 1, 0, -1):
        restore(); zero([(li, s) for li in range(k, L) for s in range(S)])
        d = (evaluate(model, batches) - base) / NATS_PER_BPB
        out["cumulative"].append(dict(keep_layers=k, removed=L - k, dBPB=d))
        print(f"  keep layers 0..{k-1:<2} (drop {L-k:>2}) -> dBPB {d:+.4f}")
    restore()

    # Redundancy test: SUM OF PARTS vs the WHOLE over the SAME layer set (1..L-1).
    # Comparing against the worst SINGLE ablation is wrong when that single is layer 0,
    # which is not in the group — an earlier version of this script made exactly that
    # error and printed "near-additive" for what is really an ~8x super-additive effect.
    parts = sum(r["dBPB"] for r in out["layer"] if r["layer"] >= 1)
    allbut0 = [r for r in out["cumulative"] if r["keep_layers"] == 1]
    if allbut0:
        whole = allbut0[0]["dBPB"]
        ratio = whole / max(parts, 1e-9)
        print(f"\n  layer 0 alone (single)             : {out['layer'][0]['dBPB']:+.4f} dBPB")
        print(f"  SUM of layers 1..{L-1} individually    : {parts:+.4f} dBPB")
        print(f"  WHOLE of layers 1..{L-1} together      : {whole:+.4f} dBPB")
        print(f"  super-additivity ratio             : {ratio:.1f}x")
        print("  >3x => REDUNDANT BUT NECESSARY: single-ablation cheapness is misleading,")
        print("         and depth cannot be traded for width on this evidence.")
        print("  ~1x => additive; the layers act independently.")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1); print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
