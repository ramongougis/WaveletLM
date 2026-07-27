"""partition_ladder.py — how much of SALT's retrieval signal is WAVELET, and how much
is just n-gram?

THE QUESTION (Ramon, 2026-07-27). SALT's full-store run measured top-1 = 19.64% against a
random-rank-within-partition control of 11.43%. The decomposition was sobering: ~11.4 of
those points come from the LAST-TOKEN PARTITION ALONE — a bigram statistic — and only ~8.2
from scale-similarity ranking. Ramon then proposed holding 2, 3, or 4 tokens fixed to get
skip-n-gram dynamics. That sharpens the same question into one that must be answered before
any of it is built:

    fixing MORE tokens grows the term that is ALREADY DOMINANT.

A win at n=4 that is really a 5-gram model is not a result about wavelets. Per the
unit-discipline rule (feedback_unit_discipline_comparison_tables), the claim a number
supports is part of the number.

THE DESIGN THAT CANNOT BE FOOLED. At each n, score retrieval TWICE within the identical
candidate set:
    scale-ranked   -> the full method
    random-ranked  -> the pure n-gram contribution
    DIFFERENCE     -> the wavelet contribution, isolated, at that n
Both arms see the same partition, so partition size, Zipfian skew, and coverage cancel
exactly. Only the ranking differs.

n = 0 IS THE MOST INFORMATIVE CELL AND HAS NEVER BEEN RUN. No partition at all = pure scale
retrieval with the bigram effect removed. If the scale-vs-random gap is ~0 at n=0, then
scale retrieval never worked and the last-token partition was carrying it the whole time.

SKIP-GRAMS. A CONTIGUOUS suffix is what an n-gram model does. Non-adjacent lag sets
(--lags 1,3 or 1,2,5) are not, so they escape the "it's just an n-gram" critique and are the
variant genuinely worth testing. Same two-arm design; lag-set is a third axis.

SPARSITY IS THE OTHER FAILURE MODE, and it moves opposite to accuracy. At n=1 the store
already has median partition 7 / p90 65. Distinct contexts grow ~geometrically in n, so by
n=3-4 most partitions go singleton and there is nothing left to rank — the same collapse
that killed the top-1% pilot. So we report partition-size stats per n alongside accuracy;
a "win" on a singleton partition is not retrieval, it is memorisation.

READS. Also reports the ORACLE within each candidate set (is the answer present at all?),
because accuracy can fall with n either because ranking got harder or because the right
answer stopped being there — those are different diagnoses and must not be conflated.

Usage:
  python tools/interpretability/partition_ladder.py --store_dir .salt/wt103_d3 \
      --n 0 1 2 3 4 --max_queries 20000
  python tools/interpretability/partition_ladder.py --store_dir .salt/wt103_d3 \
      --lags 1,3 1,2,5 --max_queries 20000
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)


def load_store(store_dir, max_rows, seed, T, max_n):
    """Returns keys (L2-normalised), next-token, nll, and a derived context matrix.

    THE STORE ONLY KEEPS ONE CONTEXT TOKEN (`L` = x at the query position; salt_store.py
    line 151 saves K, N, L, E). n=0 and n=1 therefore work directly, but n>=2 would
    normally need a GPU rebuild.

    IT DOESN'T. salt_store.sweep writes rows in scan order over (B, T), so row r holds
    window b = r//T, position t = r%T, and the token at LAG k is simply L[r-k+1] — as long
    as t >= k-1, else we would cross a window boundary and splice in unrelated text. So we
    derive lags by SHIFTING L and masking the first (k-1) positions of every window.

    THIS IS ONLY VALID ON AN UNSHUFFLED, UNFILTERED STORE. Two guards, both hard:
      * len(rows) must be divisible by T (a keep_pct<100 build breaks contiguity, and the
        top-1% pilot store WILL fail this — correctly, since its rows are non-adjacent).
      * subsampling to max_rows happens AFTER derivation, never before.
    """
    files = sorted(glob.glob(os.path.join(store_dir, "shard*.npz")))
    if not files:
        raise SystemExit(f"no shards in {store_dir}")
    K, N, E, L = [], [], [], []
    for f in files:
        z = np.load(f)
        if len(z["N"]) % T:
            raise SystemExit(
                f"{os.path.basename(f)}: {len(z['N']):,} rows is not divisible by T={T}. "
                "This store is filtered or shuffled, so lag derivation is unsafe. "
                "Use --n 0 1 only, or rebuild with keep_pct=100.")
        K.append(z["K"]); N.append(z["N"]); E.append(z["E"]); L.append(z["L"])
    K = np.concatenate(K).astype(np.float32)
    N = np.concatenate(N).astype(np.int64)
    E = np.concatenate(E).astype(np.float32)
    L = np.concatenate(L).astype(np.int64)

    # derive lags 1..max_n by shifting; -1 marks "would cross a window boundary"
    pos = np.arange(len(L)) % T
    C = np.full((len(L), max(max_n, 1)), -1, dtype=np.int64)
    for k in range(1, max(max_n, 1) + 1):
        sh = np.roll(L, k - 1)
        C[:, k - 1] = np.where(pos >= k - 1, sh, -1)

    if len(N) > max_rows:
        sel = np.random.default_rng(seed).choice(len(N), max_rows, replace=False)
        K, N, E, C = K[sel], N[sel], E[sel], C[sel]
    K /= (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
    return K, N, E, C


def partition_key(C, N, n, lags):
    """Row -> hashable partition id.

    n=0 puts every row in ONE partition (no partitioning) — the pure-scale cell.
    n>=1 uses the last n context tokens. lags uses an arbitrary NON-ADJACENT set,
    counted back from the query position (lag 1 = the immediately preceding token).
    """
    if n == 0 and not lags:
        return np.zeros(len(N), dtype=np.int64)
    idx = [l - 1 for l in lags] if lags else list(range(n))
    if max(idx) >= C.shape[1]:
        raise SystemExit(f"derived {C.shape[1]} lags; need {max(idx)+1}")
    sub = C[:, idx]                                  # (rows, |idx|), lag 1 first
    # Rows whose lag reaches past the start of their window carry -1. They must NOT be
    # grouped together: "unknown token" is not a context they share. Give each its own
    # negative id so it forms a singleton and drops out, rather than silently merging.
    bad = (sub < 0).any(axis=1)
    out = np.zeros(len(N), dtype=np.int64)
    for j in range(sub.shape[1]):                    # positional mix, order matters
        out = out * 50257 + np.maximum(sub[:, j], 0)
    out[bad] = -(np.arange(int(bad.sum())) + 1)
    return out


def score(K, N, pid, queries, rng, max_cand, qbatch, label=""):
    """Two arms over an IDENTICAL candidate set: scale-ranked vs random-ranked.

    PERFORMANCE NOTE (bug fixed 2026-07-27, after a 15-hour hang). The obvious
    implementation gathers K[cand] per query. At n=0 the partition is the ENTIRE store, so
    that is a 6.1 GB fancy-index copy 20,000 times over — quadratic in the worst cell, and
    the worst cell is the one we most want. Three changes make it tractable:
      * group queries BY PARTITION so each candidate block is gathered at most once;
      * skip the gather entirely when the bucket IS the whole store (use K directly);
      * batch the query side into one matmul per (bucket, qbatch) instead of a matvec each.

    max_cand subsamples oversized partitions. This is safe for the GAP measurement because
    BOTH ARMS SEE THE IDENTICAL POOL — subsampling changes the absolute top-1 of each arm
    but not, in expectation, the difference between them, which is the only column we read.
    """
    order = np.argsort(pid, kind="stable")
    ps = pid[order]
    edges = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    buckets = {ps[edges[i]]: order[edges[i]:edges[i + 1]] for i in range(len(edges) - 1)}

    qs_by_p = defaultdict(list)
    for q in queries:
        qs_by_p[pid[q]].append(q)

    sizes, hit_s, hit_r, oracle, n_eval = [], 0, 0, 0, 0
    done = 0
    for p, qs in qs_by_p.items():
        cand = buckets[p]
        if len(cand) > max_cand:
            cand = np.sort(rng.choice(cand, max_cand, replace=False))
        if len(cand) <= 1:
            done += len(qs)
            continue
        Kc = K if len(cand) == len(K) else K[cand]
        Nc = N[cand]
        for b in range(0, len(qs), qbatch):
            qb = np.asarray(qs[b:b + qbatch])
            sims = Kc @ K[qb].T                              # (|cand|, |qb|)
            # locate each query inside its own candidate block so we never retrieve self
            loc = np.searchsorted(cand, qb)
            locc = np.minimum(loc, len(cand) - 1)
            is_self = cand[locc] == qb
            cols = np.arange(len(qb))
            sims[locc[is_self], cols[is_self]] = -np.inf
            n_c = len(cand) - is_self.astype(np.int64)       # usable candidates per query
            keep = n_c > 0
            if not keep.any():
                done += len(qb); continue

            best = sims.argmax(0)
            hit_s += int((Nc[best] == N[qb])[keep].sum())

            # random arm: uniform over the SAME pool, self excluded by construction
            r = rng.integers(0, np.maximum(n_c, 1))
            r = r + ((is_self) & (r >= locc)).astype(np.int64)
            r = np.minimum(r, len(cand) - 1)
            hit_r += int((Nc[r] == N[qb])[keep].sum())

            # oracle: is the answer present at all, ignoring the query's own row?
            present = (Nc[:, None] == N[qb][None, :])
            present[locc[is_self], cols[is_self]] = False
            oracle += int(present.any(0)[keep].sum())

            sizes.extend([int(v) for v in n_c[keep]])
            n_eval += int(keep.sum())
            done += len(qb)
        if label and done % 5000 < qbatch:
            print(f"    [{label}] {done:,}/{len(queries):,} queries", flush=True)
    if n_eval == 0:
        return None
    sizes = np.asarray(sizes)
    return dict(n_eval=n_eval, coverage=n_eval / len(queries),
                top1_scale=hit_s / n_eval, top1_random=hit_r / n_eval,
                oracle=oracle / n_eval,
                med=float(np.median(sizes)), p90=float(np.percentile(sizes, 90)),
                singleton=float((sizes == 1).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store_dir", required=True)
    p.add_argument("--n", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="contiguous suffix lengths; 0 = no partition (pure scale)")
    p.add_argument("--lags", nargs="+", default=[],
                   help="non-adjacent lag sets, e.g. 1,3 1,2,5 (true skip-grams)")
    p.add_argument("--T", type=int, default=256,
                   help="block_size the store was built with; lag derivation needs it")
    p.add_argument("--max_rows", type=int, default=1_500_000)
    p.add_argument("--max_queries", type=int, default=20_000)
    p.add_argument("--max_cand", type=int, default=100_000,
                   help="cap candidates per partition; identical pool for BOTH "
                        "arms, so the GAP column is unaffected in expectation")
    p.add_argument("--qbatch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    max_n = max(list(args.n) + [1] +
                [max(int(v) for v in x.split(",")) for x in args.lags])
    K, N, E, C = load_store(args.store_dir, args.max_rows, args.seed, args.T, max_n)
    print(f"[store] {len(N):,} rows, key dim {K.shape[1]}, lags derived: {C.shape[1]}")
    queries = rng.choice(len(N), min(args.max_queries, len(N)), replace=False)
    print(f"[queries] {len(queries):,}\n")

    cells = [("n=%d" % n, n, []) for n in args.n]
    for spec in args.lags:
        lg = [int(x) for x in spec.split(",")]
        cells.append(("lags{%s}" % spec, len(lg), lg))

    print(f"{'partition':>12} {'cover':>7} {'scale':>8} {'random':>8} {'GAP':>8} "
          f"{'lift':>6} {'oracle':>8} {'med':>6} {'p90':>6} {'1-only':>7}")
    rows = []
    for label, n, lags in cells:
        try:
            pid = partition_key(C, N, n, lags)
        except SystemExit as e:
            print(f"{label:>12}  SKIPPED — {e}")
            continue
        r = score(K, N, pid, queries, rng, args.max_cand, args.qbatch, label)
        if r is None:
            print(f"{label:>12}  no covered queries")
            continue
        gap = r["top1_scale"] - r["top1_random"]
        lift = r["top1_scale"] / r["top1_random"] if r["top1_random"] > 0 else float("nan")
        r.update(partition=label, gap=gap, lift=lift); rows.append(r)
        print(f"{label:>12} {r['coverage']:6.1%} {r['top1_scale']:7.2%} "
              f"{r['top1_random']:7.2%} {gap:+7.2%} {lift:5.2f}x {r['oracle']:7.2%} "
              f"{r['med']:6.0f} {r['p90']:6.0f} {r['singleton']:6.1%}")

    print("\nGAP = scale-ranked minus random-ranked WITHIN THE SAME CANDIDATE SET.")
    print("It is the ONLY column that isolates the wavelet contribution: partition size,")
    print("Zipfian skew and coverage cancel between the two arms.")
    print("  GAP ~ 0 at n=0        -> pure scale retrieval never worked; the last-token")
    print("                           partition was carrying the 19.64%, and the whole")
    print("                           direction closes (the corrector shares this key).")
    print("  GAP shrinking with n  -> added tokens are buying n-gram accuracy, not wavelet")
    print("                           signal; report any such win as an n-gram result.")
    print("  '1-only' high         -> partition collapsed to singletons; a 'win' there is")
    print("                           memorisation, not retrieval. Watch it rise with n.")
    print("  oracle falling with n -> the answer LEFT the candidate set (sparsity), which")
    print("                           is a different diagnosis from ranking getting harder.")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1); print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
