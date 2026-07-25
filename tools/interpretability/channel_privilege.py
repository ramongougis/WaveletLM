"""channel_privilege.py — Study 2 at full scale: is the native channel basis privileged?

Scores EVERY (layer x scale x channel) cell against a random-direction null run
through the identical pipeline — the control Finding 5 owes before its label can
harden (top-activating examples read as coherent even for random directions;
Bolukbasi et al. 2021, the interpretability illusion).

THE TEST. For a direction d (unit vector in channel space) the projection
p = X @ d has a "feature-like" profile if its mass concentrates in few positions.
Two scale-free statistics per direction:
  * excess kurtosis of p          (spikiness)
  * top-K |p| mass fraction       (tail concentration; K default 1000)
Native channels use d = e_c (i.e. p is just the column). The null uses the SAME
count of RANDOM unit directions in the same space. If the native axes are merely
an arbitrary rotation, the two distributions coincide. If channels are privileged,
native beats the null.

Reported per cell: the fraction of native channels above the null's 95th
percentile (`frac>p95`, chance = 0.05) and the tail-ownership of the single most
dominant channel (the Finding 5 metric).

Also: --track LAYER,SCALE,CHANNEL follows one channel across two dumps (feature
persistence, e.g. ch132 at L00/s2 from D2 -> D3).

Usage:
  python tools/interpretability/channel_privilege.py --dump .interp/mini_d3
  python tools/interpretability/channel_privilege.py --dump .interp/mini_d3 \
      --compare .interp/mini_d2 --track 0,2,132
"""
import argparse, glob, json, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)


def load_cell(dump, layer, scale):
    """Concatenate shards for one layer, return (N_positions, Cp) float32 for one scale."""
    shards = sorted(glob.glob(os.path.join(dump, f"shard*_L{layer:02d}_postdecomp.npy")))
    if not shards:
        return None
    parts = [np.load(s)[:, :, scale, :] for s in shards]      # (win, T, Cp)
    X = np.concatenate(parts, axis=0)
    return X.reshape(-1, X.shape[-1]).astype(np.float32)


def stats_for(P, topk):
    """P: (N, D) projections. Returns (kurtosis[D], topk_mass_frac[D])."""
    mu = P.mean(0, keepdims=True)
    Pc = P - mu
    var = (Pc ** 2).mean(0)
    var = np.maximum(var, 1e-12)
    kurt = (Pc ** 4).mean(0) / var ** 2 - 3.0
    A = np.abs(P)
    k = min(topk, A.shape[0])
    # top-k mass fraction per column, without a full sort
    part = np.partition(A, A.shape[0] - k, axis=0)[A.shape[0] - k:]
    tot = A.sum(0); tot = np.maximum(tot, 1e-12)
    return kurt, part.sum(0) / tot


def tail_ownership(X, topk):
    """Finding-5 metric: fraction of the global top-k |values| owned by the single
    most dominant channel, plus that channel's index."""
    A = np.abs(X).ravel()
    k = min(topk, A.size)
    idx = np.argpartition(A, A.size - k)[A.size - k:]
    ch = idx % X.shape[1]
    counts = np.bincount(ch, minlength=X.shape[1])
    top = int(counts.argmax())
    return top, counts[top] / k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, help="shard dir, e.g. .interp/mini_d3")
    p.add_argument("--compare", default=None, help="second dump for persistence check")
    p.add_argument("--track", default=None, help="LAYER,SCALE,CHANNEL to follow across dumps")
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write per-cell JSON here")
    args = p.parse_args()

    man = json.load(open(os.path.join(args.dump, "manifest.json")))
    layers, S, Cp = man["layers"], man["S"], man["Cp"]
    rng = np.random.default_rng(args.seed)
    print(f"[dump] {args.dump}  run={man['run_dir']}  layers={len(layers)} S={S} Cp={Cp}")
    print(f"[test] native channels vs {Cp} random unit directions, top-{args.topk} tail\n")
    print(f"{'layer':>5} {'scale':>5} {'frac>p95':>9} {'nat kurt med':>13} {'rand kurt med':>13} "
          f"{'top ch':>7} {'owns':>7}")

    rows = []
    for L in layers:
        for s in range(S):
            X = load_cell(args.dump, L, s)
            if X is None:
                continue
            nat_k, nat_m = stats_for(X, args.topk)
            D = rng.standard_normal((Cp, Cp)).astype(np.float32)
            D /= np.linalg.norm(D, axis=0, keepdims=True)      # random unit directions
            rnd_k, rnd_m = stats_for(X @ D, args.topk)
            p95 = np.percentile(rnd_k, 95)
            frac = float((nat_k > p95).mean())                  # chance = 0.05
            ch, owns = tail_ownership(X, args.topk)
            print(f"{L:>5} {s:>5} {frac:>9.3f} {np.median(nat_k):>13.2f} "
                  f"{np.median(rnd_k):>13.2f} {ch:>7} {owns:>7.3f}")
            rows.append(dict(layer=int(L), scale=int(s), frac_above_p95=frac,
                             nat_kurt_median=float(np.median(nat_k)),
                             rand_kurt_median=float(np.median(rnd_k)),
                             nat_kurt_max=float(nat_k.max()),
                             top_channel=int(ch), top_ownership=float(owns),
                             top_by_kurt=[int(i) for i in np.argsort(nat_k)[-5:][::-1]]))
            del X

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\n[wrote] {args.out}")

    overall = float(np.mean([r["frac_above_p95"] for r in rows])) if rows else float("nan")
    print(f"\nMEAN frac>p95 across cells = {overall:.3f}   (chance 0.05; "
          f"{'PRIVILEGED' if overall > 0.15 else 'NOT clearly privileged'})")

    if args.track:
        L, s, c = (int(v) for v in args.track.split(","))
        print(f"\n=== persistence of L{L:02d}/s{s}/ch{c} ===")
        for d in [args.dump] + ([args.compare] if args.compare else []):
            X = load_cell(d, L, s)
            if X is None:
                print(f"  {d}: cell missing"); continue
            k, m = stats_for(X, args.topk)
            ch, owns = tail_ownership(X, args.topk)
            rank = int((k > k[c]).sum())
            print(f"  {d:22s} ch{c}: kurt={k[c]:8.2f} rank={rank:>4}/{len(k)} | "
                  f"cell top-ch={ch} owns={owns:.3f} | sign(mean)={np.sign(X[:, c].mean()):+.0f}")


if __name__ == "__main__":
    main()
