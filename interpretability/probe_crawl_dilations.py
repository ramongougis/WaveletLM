#!/usr/bin/env python
"""Probe what the wavelet crawl actually learned about relative-position lags.

The crawl (model.py: LiftingWaveletDecompose) gives each dyadic level a learned
softmax over K contiguous integer lags centred on 2^level. This script reads the
`dilation_logits` from a trained checkpoint and reports, per level (and per layer),
which lags the model actually put weight on — a direct test of the hypothesis that
the model wants *non-dyadic* (e.g. prime) relative positions.

It reconstructs the offset windows exactly as model.py does (half = K//2,
min_off = max(1, 2^level - half), offsets = range(min_off, min_off + K)), so it
needs no model import — just the checkpoint.

Usage:
    python interpretability/probe_crawl_dilations.py logs/<run>/best_model.pt
"""
import sys
import math
import torch
import torch.nn.functional as F

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
          71, 73, 79}


def find_state_dict(obj):
    """Checkpoints vary: raw state_dict, or wrapped under a known key."""
    if isinstance(obj, dict):
        for key in ("model_state", "model_state_dict", "model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # Maybe it's already a state_dict (keys map to tensors).
        if any(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise SystemExit("Could not locate a state_dict in the checkpoint.")


def level_offsets(level, K):
    base = 1 << level
    half = K // 2
    min_off = max(1, base - half)
    return base, list(range(min_off, min_off + K))


def main(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = find_state_dict(ckpt)
    keys = sorted(k for k in sd if k.endswith("dilation_logits"))
    if not keys:
        raise SystemExit("No `dilation_logits` in this checkpoint — crawl was off.")

    print(f"Checkpoint: {path}")
    print(f"Found {len(keys)} crawl tensor(s) (one per layer).\n")

    for key in keys:
        logits = sd[key].float()          # [levels, K]
        levels, K = logits.shape
        weights = F.softmax(logits, dim=1)
        print(f"=== {key}   (levels={levels}, K={K}) ===")
        print(f"{'lvl':>3} {'dyadic':>6} {'argmax':>6} {'CoM':>7} "
              f"{'w@dyadic':>9} {'w_offbase':>9} {'w@primes':>8}  top-3 lags (weight)")
        for lvl in range(levels):
            base, offs = level_offsets(lvl, K)
            w = weights[lvl]
            argmax_lag = offs[int(w.argmax())]
            com = float(sum(wk.item() * o for wk, o in zip(w, offs)))
            w_dyadic = float(w[offs.index(base)]) if base in offs else float("nan")
            w_offbase = 1.0 - w_dyadic if not math.isnan(w_dyadic) else float("nan")
            w_primes = float(sum(wk.item() for wk, o in zip(w, offs) if o in PRIMES))
            top = sorted(zip(w.tolist(), offs), reverse=True)[:3]
            top_str = ", ".join(f"{o}({wk:.2f})" for wk, o in top)
            print(f"{lvl:>3} {base:>6} {argmax_lag:>6} {com:>7.1f} "
                  f"{w_dyadic:>9.3f} {w_offbase:>9.3f} {w_primes:>8.3f}  {top_str}")
        # Headline: how far did this layer move off the dyadic init, overall?
        moved = float(torch.stack([
            1.0 - weights[lvl][level_offsets(lvl, K)[1].index(1 << lvl)]
            for lvl in range(levels)
            if (1 << lvl) in level_offsets(lvl, K)[1]
        ]).mean())
        print(f"  -> mean weight OFF the dyadic base across levels: {moved:.3f}")
        print(f"     (~0 = stayed dyadic; ~1 = fully relocated to other lags)\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: probe_crawl_dilations.py <checkpoint.pt>")
    main(sys.argv[1])
