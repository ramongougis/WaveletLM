"""flow_vs_entropy.py — does per-scale coefficient FLOW tell us anything that
output ENTROPY doesn't?

WHY THIS GATES THE MEMORY DESIGN. A retrieval/memory mechanism needs a label-free
signal at inference time to decide "am I likely to be wrong here — should I consult
memory?". Finding 17 showed flow correlates with the model's own NLL (up to +0.39),
so flow is a candidate gate. BUT the incumbent is output-distribution entropy, which
is also label-free, is the standard difficulty proxy, and is probably a STRONGER NLL
predictor. If flow is subsumed by entropy, the mechanism should just use entropy and
stay simple.

So the question is not "does flow predict difficulty" but:
    does flow explain NLL variance that entropy does NOT?
Answered with partial correlation, r(flow, NLL | entropy).

Also reports per LAYER, since Finding 17 measured layer 0 only and the coupling may
strengthen with depth.

BUILT-IN NEGATIVE CONTROL: s7 measured r = -0.02 with NLL (Finding 17). Any
flow-based signal keyed on s7 should come out near zero here too; if it doesn't,
suspect the implementation rather than celebrate.

Usage:
  python tools/interpretability/flow_vs_entropy.py \
      --run_dir logs/wikitext-103_2026-07-15_10-53-46 --layers 0 4 9 --tokens 4096
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


def partial_corr(x, y, z):
    """r(x,y | z)"""
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    return (rxy - rxz * ryz) / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 4, 9])
    p.add_argument("--tokens", type=int, default=4096)
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

    store, hooks = {}, []
    def mk(li, si):
        def h(_m, _i, out): store.setdefault(li, {})[si] = out.detach()[0]
        return h
    for li in args.layers:
        for si in range(S):
            hooks.append(model.layers[li].decomp_norms[si].register_forward_hook(mk(li, si)))

    g = torch.Generator().manual_seed(11)
    FLOW = {li: [] for li in args.layers}
    ENT, NLL = [], []
    for _ in range(max(args.tokens // T, 1)):
        s0 = int(torch.randint(0, len(va) - T - 2, (1,), generator=g).item())
        x = va[s0:s0 + T].unsqueeze(0).to(dev).long()
        y = va[s0 + 1:s0 + T + 1].to(dev).long()
        store.clear()
        with torch.no_grad():
            logits, _ = model(x)
        lp = F.log_softmax(logits[0].float(), dim=-1)
        ent = -(lp.exp() * lp).sum(-1)                       # (T,)
        nll = F.nll_loss(lp, y, reduction="none")            # (T,)
        for li in args.layers:
            C = torch.stack([store[li][s] for s in range(S)], dim=1)   # (T,S,Cp)
            FLOW[li].append((C[1:] - C[:-1]).abs().mean(dim=2).cpu().numpy())
        ENT.append(ent[:-1].cpu().numpy()); NLL.append(nll[:-1].cpu().numpy())
    for h in hooks: h.remove()

    E = np.concatenate(ENT); N = np.concatenate(NLL)
    print(f"\n[setup] {len(N):,} positions, S={S}, layers={args.layers}")
    print(f"[incumbent] corr(entropy, NLL) = {np.corrcoef(E, N)[0,1]:+.3f}   <- the bar to beat\n")

    rows = []
    for li in args.layers:
        Fl = np.concatenate(FLOW[li])
        print(f"=== layer {li} ===")
        print(f"{'scale':>6} {'r(flow,NLL)':>12} {'r(flow,ent)':>12} {'PARTIAL r(flow,NLL|ent)':>25}")
        for s in range(S):
            f = Fl[:, s]
            r1 = np.corrcoef(f, N)[0, 1]; r2 = np.corrcoef(f, E)[0, 1]
            pr = partial_corr(f, N, E)
            print(f"{'s'+str(s):>6} {r1:+12.3f} {r2:+12.3f} {pr:+25.3f}")
            rows.append(dict(layer=li, scale=s, r_flow_nll=float(r1),
                             r_flow_ent=float(r2), partial=float(pr)))
        tot = Fl.mean(axis=1)
        print(f"{'ALL':>6} {np.corrcoef(tot,N)[0,1]:+12.3f} {np.corrcoef(tot,E)[0,1]:+12.3f} "
              f"{partial_corr(tot,N,E):+25.3f}\n")

    best = max(rows, key=lambda r: abs(r["partial"]))
    print(f"largest |partial| : L{best['layer']:02d}/s{best['scale']}  {best['partial']:+.3f}")
    print("  |partial| >~0.10 => flow carries difficulty signal ENTROPY MISSES -> a per-scale")
    print("                      gate is worth building (and it is per-scale, which entropy")
    print("                      has no analogue for)")
    print("  |partial| ~0     => flow is subsumed by entropy -> use entropy, keep it simple")
    if args.out:
        json.dump(dict(r_entropy_nll=float(np.corrcoef(E, N)[0, 1]), rows=rows),
                  open(args.out, "w"), indent=1); print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
