"""topk_contexts.py — Study 2 pilot instrument (plans/interpretability.md).

For one (layer, scale) cell of a coeff_dump, finds the top-k activations by
|value| across all (window, position, channel) and prints each with its
decoded text context, the hot token marked <<like so>>. Also reports channel
concentration in the extreme tail (top-1000): if a handful of channels own
the extremes, that's selectivity evidence; if the tail is spread thin, the
high kurtosis is variance-heterogeneity instead.

Scale indexing (confirmed, model.py ~2427: [approx] + details[::-1]):
s0 = approximation; s1..s7 = detail bands COARSE -> FINE
(s1 ~ 64-128 tokens ... s7 ~ 1-2 tokens at levels=7).

Usage:
  python tools/interpretability/topk_contexts.py --dump_dir .interp/mini_d2 \
      --layer 0 --scale 2 --k 12
"""
import argparse
import heapq
import json
import os
import sys
from collections import Counter

import numpy as np
import tiktoken


def main():
    # Windows consoles may run legacy codepages (cp950 etc.) that crash on
    # WikiText's Unicode (e.g. U+2212). Replace rather than die.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--scale", type=int, required=True)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--capture", default=None)
    p.add_argument("--context_before", type=int, default=40)
    p.add_argument("--context_after", type=int, default=12)
    p.add_argument("--tail", type=int, default=1000,
                   help="size of the extreme tail for channel-concentration stats")
    args = p.parse_args()

    with open(os.path.join(args.dump_dir, "manifest.json")) as f:
        man = json.load(f)
    pt = args.capture or man["capture"][0]
    W_shard = man["shard_windows"]
    n_shards = (man["n_windows"] + W_shard - 1) // W_shard

    heap = []  # min-heap of (|v|, v, gw, t, c), size <= max(k, tail)
    keep = max(args.k, args.tail)
    for sh in range(n_shards):
        path = os.path.join(args.dump_dir,
                            f"shard{sh:04d}_L{args.layer:02d}_{pt}.npy")
        x = np.load(path)[:, :, args.scale, :].astype(np.float32)  # (W, T, Cp)
        flat = np.abs(x).ravel()
        take = min(keep, flat.size)
        idx = np.argpartition(flat, -take)[-take:]
        Wl, T, Cp = x.shape
        for i in idx:
            w, rem = divmod(int(i), T * Cp)
            t, c = divmod(rem, Cp)
            v = float(x[w, t, c])
            item = (abs(v), v, sh * W_shard + w, t, c)
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    hits = sorted(heap, reverse=True)
    tail_channels = Counter(h[4] for h in hits[:args.tail])
    total_tail = sum(tail_channels.values())

    enc = tiktoken.get_encoding("gpt2")
    tok_cache = {}

    def window_tokens(gw):
        sh, w = divmod(gw, W_shard)
        if sh not in tok_cache:
            tok_cache[sh] = np.load(
                os.path.join(args.dump_dir, f"shard{sh:04d}_tokens.npy"))
        return tok_cache[sh][w]

    print(f"[topk] {args.dump_dir} L{args.layer:02d}/s{args.scale} ({pt}); "
          f"scale band: {'approximation' if args.scale == 0 else 'detail (coarse->fine idx)'}")
    print(f"[topk] channel concentration in top-{total_tail} tail: "
          f"{len(tail_channels)} distinct channels; top 8: "
          + ", ".join(f"ch{c}={n} ({100*n/total_tail:.0f}%)"
                      for c, n in tail_channels.most_common(8)))
    print("-" * 78)
    for rank, (_, v, gw, t, c) in enumerate(hits[:args.k], 1):
        toks = window_tokens(gw)
        lo = max(0, t - args.context_before)
        hi = min(len(toks), t + 1 + args.context_after)
        before = enc.decode(toks[lo:t].tolist())
        hot = enc.decode([int(toks[t])])
        after = enc.decode(toks[t + 1:hi].tolist())
        ctx = (before + "<<" + hot + ">>" + after).replace("\n", "\\n")
        print(f"#{rank:>2}  v={v:+8.2f}  ch={c:<4} win={gw:<3} pos={t:<3} | {ctx}")
    print("-" * 78)


if __name__ == "__main__":
    main()
