"""ica_test.py — the decisive privileged-basis test (Study 2).

Finding 9 showed the naive random-direction control is confounded: dense random
directions average ~Cp channels, and averaging kills kurtosis by CLT regardless of
basis structure. The valid question is instead:

    Is there a ROTATION whose coordinates are far more non-Gaussian than the
    native channel axes?

If yes -> features are rotated relative to the channel basis -> the basis is NOT
privileged -> SAEs (or another un-mixing step) are indicated.
If no  -> the native axes are already near the most non-Gaussian directions
          available -> the basis IS privileged -> SAEs deprioritized.

Both sides are 1-D projections of the same data and excess kurtosis is
scale-invariant, so `kurtosis(X @ w_ica)` and `kurtosis(X[:, c])` are directly
comparable — no whitening asymmetry to argue about.

FastICA maximises non-Gaussianity over rotations (projection pursuit), so it is a
constructive search for the strongest counterexample to the thesis.

Usage:
  python tools/interpretability/ica_test.py --dump .interp/mini_d3 \
      --cells 0,2 0,0 3,3 1,7 9,0 --components 64
"""
import argparse, glob, json, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)


def load_cell(dump, layer, scale):
    sh = sorted(glob.glob(os.path.join(dump, f"shard*_L{layer:02d}_postdecomp.npy")))
    X = np.concatenate([np.load(p)[:, :, scale, :] for p in sh], axis=0)
    return X.reshape(-1, X.shape[-1]).astype(np.float64)


def exkurt(P):
    """Excess kurtosis per column of P (N, D)."""
    Pc = P - P.mean(0, keepdims=True)
    v = np.maximum((Pc ** 2).mean(0), 1e-12)
    return (Pc ** 4).mean(0) / v ** 2 - 3.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--cells", nargs="+", required=True, help="LAYER,SCALE pairs")
    p.add_argument("--components", type=int, default=64)
    p.add_argument("--subsample", type=int, default=20000, help="rows for ICA fit (speed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from sklearn.decomposition import FastICA

    print(f"[dump] {args.dump}   ICA components={args.components}   "
          f"fit rows={args.subsample}")
    print("Comparing the most non-Gaussian NATIVE channel against the most "
          "non-Gaussian ROTATED direction ICA can find.\n")
    print(f"{'cell':>8} {'nat max':>9} {'nat p95':>9} {'ICA max':>9} {'ICA p95':>9} "
          f"{'ICA/nat':>8}  verdict")
    rows = []
    rng = np.random.default_rng(args.seed)
    for spec in args.cells:
        L, s = (int(v) for v in spec.split(","))
        X = load_cell(args.dump, L, s)
        nat = exkurt(X)
        idx = rng.choice(X.shape[0], min(args.subsample, X.shape[0]), replace=False)
        try:
            ica = FastICA(n_components=args.components, random_state=args.seed,
                          whiten="unit-variance", max_iter=600, tol=1e-3)
            ica.fit(X[idx])
            W = ica.components_.T                      # (Cp, k) directions in ORIGINAL space
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            ic = exkurt(X @ W)
        except Exception as e:
            print(f"  L{L:02d}/s{s}: ICA failed ({type(e).__name__}: {str(e)[:50]})")
            continue
        ratio = float(np.abs(ic).max() / max(np.abs(nat).max(), 1e-9))
        verdict = ("NOT privileged" if ratio > 2.0 else
                   "privileged-ish" if ratio < 1.25 else "partial")
        print(f"  L{L:02d}/s{s} {np.abs(nat).max():9.2f} {np.percentile(np.abs(nat),95):9.2f} "
              f"{np.abs(ic).max():9.2f} {np.percentile(np.abs(ic),95):9.2f} {ratio:8.2f}  {verdict}")
        rows.append(dict(layer=L, scale=s, nat_max=float(np.abs(nat).max()),
                         nat_p95=float(np.percentile(np.abs(nat), 95)),
                         ica_max=float(np.abs(ic).max()),
                         ica_p95=float(np.percentile(np.abs(ic), 95)),
                         ratio=ratio, verdict=verdict))
        del X

    if rows:
        m = float(np.mean([r["ratio"] for r in rows]))
        print(f"\nMEAN ICA/native kurtosis ratio = {m:.2f}")
        print("  <1.25 => native axes are already near-optimal (PRIVILEGED BASIS)")
        print("  >2.0  => far more non-Gaussian rotations exist (NOT privileged; SAEs indicated)")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1); print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
