"""surprise_spectrum.py — the flow study: WHERE, and at WHAT TIMESCALE, is the
model's prediction being corrected?

MOTIVATION (Ramon, 2026-07-25). The lifting scheme is literally predict-then-update:
predict the odd samples from the even, keep the residual as the DETAIL coefficient,
then update. So detail coefficients are not "features" by analogy — they ARE
prediction errors, one per temporal resolution. Tracking them along the time axis
therefore tracks *where and at what timescale the model is being surprised*. No
transformer has an equivalent, because none has a scale axis to decompose surprise
across.

WELL-POSED BECAUSE OF FINDING 16: the decomposition measured shift-equivariant
(odd/pow2 contrast 0.87x), so a coefficient is a stable code for a position rather
than an artifact of grid alignment. Without that, per-position deltas would be
uninterpretable.

WHAT IT COMPUTES, per token position t:
  * surprise[s]  = mean |coefficient| at scale s, position t  (the per-scale
                   prediction-error magnitude at that instant)
  * delta[s]     = |coeff(t) - coeff(t-1)| at scale s         (the FLOW: how fast
                   the code is moving at that timescale)
  * the model's own next-token NLL at t, for alignment
Then it reports which scales spike where, and whether per-scale surprise actually
predicts token-level NLL — a falsifiable check that the "surprise" reading is more
than a name.

Usage:
  python tools/interpretability/surprise_spectrum.py \
      --run_dir logs/wikitext-103_2026-07-15_10-53-46 --layer 0 --tokens 512
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


def capture(model, x, layer, S):
    store, hooks = {}, []
    def mk(si):
        def h(_m, _i, out): store[si] = out.detach()[0]      # (T, Cp)
        return h
    blk = model.layers[layer]
    for si in range(S):
        hooks.append(blk.decomp_norms[si].register_forward_hook(mk(si)))
    with torch.no_grad():
        logits, _ = model(x)
    for h in hooks:
        h.remove()
    return torch.stack([store[s] for s in range(S)], dim=1), logits   # (T,S,Cp), (1,T,V)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--top", type=int, default=12, help="how many peak positions to show")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = torch.device(args.device)
    model, cfg = load_model(args.run_dir, dev); model.eval()
    T = cfg.get("block_size", 256)
    S = model.layers[0].scale_weights.numel()
    for blk in model.layers:
        if hasattr(blk, "decompose_bypass_cross_window"):
            blk.decompose_bypass_cross_window = False

    _, va, _, _, _ = load_and_encode_dataset(
        {"dataset": cfg.get("dataset", "wikitext-103"), "tokenizer": "auto", "block_size": T}, _L())
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    g = torch.Generator().manual_seed(11)
    n_win = max(args.tokens // T, 1)
    surprise, deltas, nlls, toks = [], [], [], []
    for w in range(n_win):
        start = int(torch.randint(0, len(va) - T - 2, (1,), generator=g).item())
        x = va[start:start + T].unsqueeze(0).to(dev).long()
        y = va[start + 1:start + T + 1].to(dev).long()
        C, logits = capture(model, x, args.layer, S)          # (T,S,Cp)
        sur = C.abs().mean(dim=2)                             # (T,S)
        dlt = (C[1:] - C[:-1]).abs().mean(dim=2)              # (T-1,S)
        nll = F.cross_entropy(logits[0].float(), y, reduction="none")   # (T,)
        surprise.append(sur.cpu().numpy()); deltas.append(dlt.cpu().numpy())
        nlls.append(nll.cpu().numpy()); toks.append(x[0].cpu().numpy())

    Sur = np.concatenate(surprise); Del = np.concatenate(deltas)
    NLL = np.concatenate([n[:-1] for n in nlls])              # align to Del
    print(f"[setup] layer {args.layer}, {len(Sur):,} positions, S={S}\n")

    print("=== per-scale SURPRISE (mean |coefficient|) and FLOW (mean |Δ| per step) ===")
    print(f"{'scale':>6} {'surprise':>10} {'flow |Δ|':>10} {'flow/surprise':>14}")
    for s in range(S):
        f_ = Del[:, s].mean(); u = Sur[:, s].mean()
        print(f"{'s'+str(s):>6} {u:10.4f} {f_:10.4f} {f_/max(u,1e-9):14.3f}")
    print("  flow/surprise high => that band's code turns over fast (local, volatile);")
    print("  low => it holds its value across tokens (integrative, slow).\n")

    print("=== does per-scale surprise PREDICT token NLL? (falsifiable check) ===")
    for s in range(S):
        r = np.corrcoef(Del[:, s], NLL)[0, 1]
        print(f"  corr(flow s{s}, NLL) = {r:+.3f}")
    tot = Del.mean(axis=1)
    print(f"  corr(total flow, NLL) = {np.corrcoef(tot, NLL)[0,1]:+.3f}"
          f"   <- if ~0, 'surprise' is a misnomer for these coefficients")

    print(f"\n=== peak-flow positions (where the code moves most) ===")
    flat = np.concatenate([d.mean(axis=1) for d in deltas])
    order = np.argsort(-flat)[:args.top]
    off = 0; bounds = []
    for d in deltas:
        bounds.append((off, off + len(d))); off += len(d)
    for idx in order:
        wi = next(i for i, (a, b) in enumerate(bounds) if a <= idx < b)
        pos = idx - bounds[wi][0] + 1
        dom = int(np.argmax(deltas[wi][idx - bounds[wi][0]]))
        ctx = enc.decode([int(t) for t in toks[wi][max(0, pos - 6):pos + 1]])
        nxt = enc.decode([int(toks[wi][pos + 1])]) if pos + 1 < T else ""
        print(f"  flow={flat[idx]:.3f} peak-scale=s{dom} nll={nlls[wi][pos]:5.2f} "
              f"| ...{ctx!r} -> {nxt!r}")

    if args.out:
        json.dump(dict(surprise=Sur.mean(0).tolist(), flow=Del.mean(0).tolist(),
                       corr_nll=[float(np.corrcoef(Del[:, s], NLL)[0, 1]) for s in range(S)]),
                  open(args.out, "w"), indent=1)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
