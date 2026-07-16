"""census.py — Study 1 instrument (plans/interpretability.md).

Reads a coeff_dump shard directory and prints the full layer x scale
statistics table. Pure numpy on disk shards; the model is never loaded.

Stats:
  absmean  — mean |coeff|: post-norm learned gain profile (the U-shape)
  std      — spread per scale
  kurtosis — excess kurtosis: sparsity/heavy-tailedness indicator (high
             kurtosis = few large coefficients carry the signal — the
             wavelet-sparsity signature, and Study 2 fuel)

Usage:
  python tools/interpretability/census.py --dump_dir .interp/mini_d2
  python tools/interpretability/census.py --dump_dir .interp/mini_d2 --stat kurtosis
"""
import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True)
    p.add_argument("--capture", default=None, help="default: first in manifest")
    p.add_argument("--stat", default="absmean",
                   choices=["absmean", "std", "kurtosis"])
    args = p.parse_args()

    with open(os.path.join(args.dump_dir, "manifest.json")) as f:
        man = json.load(f)
    pt = args.capture or man["capture"][0]
    layers, S = man["layers"], man["S"]
    n_shards = (man["n_windows"] + man["shard_windows"] - 1) // man["shard_windows"]

    print(f"[census] {args.dump_dir}: {man['dataset']}/{man['split']}, "
          f"{man['n_windows']} windows x {man['block_size']} tokens, "
          f"capture={pt}, stat={args.stat}")
    header = "layer | " + " ".join(f"{'s' + str(s):>7}" for s in range(S))
    print(header)
    print("-" * len(header))

    for li in layers:
        a_sum = np.zeros(S, dtype=np.float64)   # sum |x|
        s1 = np.zeros(S, dtype=np.float64)      # sum x
        s2 = np.zeros(S, dtype=np.float64)      # sum x^2
        s3 = np.zeros(S, dtype=np.float64)      # sum x^3
        s4 = np.zeros(S, dtype=np.float64)      # sum x^4
        n = 0
        for sh in range(n_shards):
            path = os.path.join(args.dump_dir, f"shard{sh:04d}_L{li:02d}_{pt}.npy")
            x = np.load(path).astype(np.float32)          # (W, T, S, Cp)
            axes = (0, 1, 3)
            a_sum += np.abs(x).sum(axis=axes, dtype=np.float64)
            s1 += x.sum(axis=axes, dtype=np.float64)
            xsq = (x.astype(np.float64)) ** 2
            s2 += xsq.sum(axis=axes)
            if args.stat == "kurtosis":
                s3 += (xsq * x).sum(axis=axes)
                s4 += (xsq * xsq).sum(axis=axes)
            n += x.shape[0] * x.shape[1] * x.shape[3]
        mu = s1 / n
        var = s2 / n - mu ** 2
        if args.stat == "absmean":
            vals = a_sum / n
        elif args.stat == "std":
            vals = np.sqrt(var)
        else:  # excess kurtosis from raw moments (central 4th / var^2 - 3)
            c4 = s4 / n - 4 * mu * (s3 / n) + 6 * mu ** 2 * (s2 / n) - 3 * mu ** 4
            vals = c4 / (var ** 2) - 3.0
        print(f"  L{li:02d} | " + " ".join(f"{v:7.3f}" for v in vals))

    print(f"[census] done ({n:,} values per scale per layer)")


if __name__ == "__main__":
    main()
