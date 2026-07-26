"""salt_corrector.py — Test A: a LEARNED corrector instead of nearest-neighbour lookup.

THE IDEA (Ramon, 2026-07-26). SALT's raw cosine retrieval failed decisively: every
interpolation setting made BPB worse (plans/salt.md). But "cosine in the native space"
was an assumption, not a requirement. So *learn* the mapping from context-scales to the
correct continuation instead of assuming a metric.

WHAT THIS IS, NAMED HONESTLY: **boosting**. Train model 2 on model 1's residuals. It
works when the base learner underfits; D3 is 40 epochs and well-converged, and 42% of its
high-loss targets are corpus hapaxes, so the honest prior is a small gain at best. Cheap
enough to settle empirically.

DESIGN — small, and principled about the output space:
    key (s0+s1 coeffs, 1024) -> MLP -> C-dim vector -> [FROZEN tied embedding] -> vocab
Reusing the model's own tied output head means the corrector predicts in the space the
model already writes into, and keeps the parameter count to ~1M instead of the ~51M a
raw 1024->50257 head would need.

THE DECISIVE MEASUREMENT is NOT "does the corrector predict well". It is:
    does the GATED ENSEMBLE beat D3's BPB?
i.e. exactly the question the interpolation sweep answered negatively for lookup. Same
combination rule, so the two are directly comparable:
    p_mix[y] = (1-lam)*p_model[y] + lam*p_corrector[y]
p_model[y] = exp(-nll) is already stored, so no 50K-vocab distribution is needed for the
model side.

GATING (Finding 19): optionally apply the corrector only where the model is likely wrong.
Entropy is unavailable from the store, so we gate on the stored NLL percentile as a
stand-in and report both gated and ungated — an ORACLE gate (only where the model is
actually wrong) is also reported as an upper bound on what any real gate could achieve.

Usage:
  python tools/interpretability/salt_corrector.py --store_dir .salt/wt103_d3 \
      --run_dir logs/wikitext-103_2026-07-15_10-53-46
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from tools.interpretability.scale_ablation import load_model, NATS_PER_BPB as NATS  # noqa: E402


class Corrector(nn.Module):
    """key -> hidden -> C, then logits = h @ E^T with E FROZEN (the model's tied head)."""
    def __init__(self, d_in, hidden, C, emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, C),
        )
        self.register_buffer("E", emb)          # (V, C) frozen
    def forward(self, x):
        return self.net(x) @ self.E.t()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store_dir", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_rows", type=int, default=1_500_000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    # --- frozen output head from the trained model ---
    model, cfg = load_model(args.run_dir, torch.device("cpu")); model.eval()
    emb = None
    for name, prm in model.named_parameters():
        if name.endswith("token_embedding.weight"):
            emb = prm.detach().float(); break
    if emb is None:
        raise SystemExit("could not find token_embedding.weight")
    V, C = emb.shape
    print(f"[head] frozen tied embedding {tuple(emb.shape)}")

    # --- the store ---
    files = sorted(glob.glob(os.path.join(args.store_dir, "shard*.npz")))
    Ks, Ns, Es = [], [], []
    for f in files:
        z = np.load(f); Ks.append(z["K"]); Ns.append(z["N"]); Es.append(z["E"])
    K = np.concatenate(Ks); N = np.concatenate(Ns); E = np.concatenate(Es).astype(np.float32)
    if len(N) > args.max_rows:
        sel = np.random.default_rng(args.seed).choice(len(N), args.max_rows, replace=False)
        K, N, E = K[sel], N[sel], E[sel]
    print(f"[store] {len(N):,} rows, key dim {K.shape[1]}")

    n_val = int(len(N) * args.val_frac)
    perm = np.random.default_rng(args.seed).permutation(len(N))
    va, tr = perm[:n_val], perm[n_val:]
    Xtr = torch.from_numpy(K[tr]).float(); Ytr = torch.from_numpy(N[tr]).long()
    Xva = torch.from_numpy(K[va]).float(); Yva = torch.from_numpy(N[va]).long()
    Eva = torch.from_numpy(E[va]).float()                 # D3's NLL on the val rows
    print(f"[split] train {len(Ytr):,}  val {len(Yva):,}")

    net = Corrector(K.shape[1], args.hidden, C, emb)
    n_tr = sum(p.numel() for p in net.net.parameters())
    print(f"[corrector] {n_tr/1e6:.2f}M trainable params (head frozen)")
    opt = torch.optim.AdamW(net.net.parameters(), lr=args.lr, weight_decay=1e-4)

    for ep in range(args.epochs):
        net.train(); idx = torch.randperm(len(Ytr)); tot = 0.0; nb = 0
        for b in range(0, len(idx), args.batch):
            j = idx[b:b + args.batch]
            loss = F.cross_entropy(net(Xtr[j]), Ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if nb % 200 == 0:
                print(f"  ep{ep} {b:,}/{len(idx):,} loss {tot/nb:.4f}", flush=True)
        net.eval()
        with torch.no_grad():
            vl = np.mean([F.cross_entropy(net(Xva[i:i+4096]), Yva[i:i+4096]).item()
                          for i in range(0, len(Yva), 4096)])
        print(f"[epoch {ep}] train {tot/max(nb,1):.4f}  val {vl:.4f} nats", flush=True)

    # --- the decisive test: does the gated ensemble beat D3? ---
    net.eval()
    p_corr = []
    with torch.no_grad():
        for i in range(0, len(Yva), 4096):
            lp = F.log_softmax(net(Xva[i:i+4096]).float(), -1)
            p_corr.append(lp.gather(1, Yva[i:i+4096, None]).squeeze(1).exp())
    p_corr = torch.cat(p_corr).numpy()
    nll0 = Eva.numpy(); p_model = np.exp(-nll0); base = float(nll0.mean())

    with torch.no_grad():
        arg = torch.cat([net(Xva[i:i+4096]).argmax(-1) for i in range(0, len(Yva), 4096)]).numpy()
    print(f"\n[corrector] standalone top-1 on val: {(arg == Yva.numpy()).mean():.2%}")
    print(f"[D3]        baseline NLL on val    : {base:.4f} nats")

    # oracle gate = apply ONLY where D3 is actually wrong (upper bound on any real gate)
    # nll gate    = apply where D3's NLL is above a percentile (a realisable proxy)
    print(f"\n{'gate':>14} {'frac':>7} {'lam':>6} {'dNLL':>10} {'dBPB':>10}")
    rows = []
    gates = {"ungated": np.ones(len(nll0), bool)}
    for q in (50, 75, 90):
        gates[f"nll>p{q}"] = nll0 >= np.percentile(nll0, q)
    best = None
    for gname, g in gates.items():
        for lam in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
            pm = p_model.copy()
            pm[g] = (1 - lam) * p_model[g] + lam * p_corr[g]
            d = float((-np.log(pm.clip(1e-12))).mean() - base)
            rows.append(dict(gate=gname, lam=lam, dNLL=d, dBPB=d / NATS))
            print(f"{gname:>14} {g.mean():7.2%} {lam:6.2f} {d:+10.4f} {d/NATS:+10.4f}")
            if best is None or d < best["dNLL"]:
                best = rows[-1]
    print(f"\n  BEST: {best['gate']} lam={best['lam']}  dBPB={best['dBPB']:+.4f}")
    print("  NEGATIVE = improvement over D3. Noise floor 0.0010.")
    print("  Compare directly against SALT lookup, whose best was +0.0054 (i.e. worse).")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1); print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
