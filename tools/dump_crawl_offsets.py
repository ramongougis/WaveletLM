"""Dump wavelet-crawl learned offset preferences from a checkpoint.

Reads the crawl `dilation_logits` (levels x K) straight from the checkpoint's
state dict — no model construction, no GPU, no dataset — softmaxes each level,
and prints which look-back offsets the model actually chose to weight.

This replaces a hypothetical K=2 ablation: instead of training runs to learn
"which neighbor of the base dilation does the work?", read the answer out of
any trained K>=3 checkpoint directly. It is also an interpretability artifact
in its own right: "level 4 moved 30% of its mass to offset 17" is a complete,
human-readable statement about what the model wants from its dilation lattice
(README 'Wavelet Crawl Dilation Window (K) Sweep').

Offset reconstruction mirrors model.py's crawl construction exactly:
    base = 2**level; half = K // 2
    min_off = max(1, base - half); offsets = [min_off .. min_off + K - 1]

Usage:
    python tools/dump_crawl_offsets.py --checkpoint logs/<run>/best_model.pt
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    keys = [k for k in state if k.endswith("dilation_logits")]
    if not keys:
        print("No 'dilation_logits' in checkpoint — was this run trained with wavelet_crawl=true?")
        return

    for key in keys:
        logits = state[key].float()
        levels, K = logits.shape
        half = K // 2
        print(f"\n{key}  (levels={levels}, K={K})")
        print(f"{'level':>5} {'base':>5} | offset:weight (learned softmax; * = base offset)")
        print("-" * 72)
        for level in range(levels):
            base = 1 << level
            min_off = max(1, base - half)
            offsets = list(range(min_off, min_off + K))
            w = torch.softmax(logits[level], dim=0)
            cells = []
            for k, off in enumerate(offsets):
                mark = "*" if off == base else ""
                cells.append(f"{off}{mark}:{w[k].item():.3f}")
            # how much mass left the base offset
            base_idx = offsets.index(base)
            moved = 1.0 - w[base_idx].item()
            print(f"{level:>5} {base:>5} | {'  '.join(cells)}   (moved off base: {moved:.1%})")


if __name__ == "__main__":
    main()
