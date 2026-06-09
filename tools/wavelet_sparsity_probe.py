"""Wavelet detail-coefficient sparsity probe (diagnostic, no training).

Measures the magnitude distribution of the lifting wavelet's DETAIL coefficients,
per scale (level), on a trained checkpoint over a held-out data slice. Classical
wavelet compression (JPEG 2000) exploits that natural-signal detail coefficients
are mostly near zero; this probe checks whether WaveletLM's *learned* wavelet has
the same property, and at which scales.

Why it matters (per the README "Wavelet Sparsity Probe & Wavelet Shrinkage"
section): if a large fraction of detail coefficients are ~0, it opens sparse
mixer compute (top-k per scale), low-bit detail quantization, sparse activation
storage for long context, and justifies trying wavelet shrinkage (soft-threshold
in forward). If they are NOT sparse, that is itself informative — the learned
wavelet does not concentrate energy the way fixed wavelets do.

Method (robust by construction): register a forward hook on each block's
`lifting_wavelet` module and capture whatever (approx, details) it returns at
runtime — no reconstruction of the decompose call, no assumptions about the
real-vs-complex tuple shape (the hook reads the actual output). Detail tensors
are the per-level high-frequency coefficients; we report their magnitude
distribution per scale.

Usage:
    python tools/wavelet_sparsity_probe.py --checkpoint logs/<run>/best_model.pt
        [--config logs/<run>/config.json]   # defaults to ./config.json
        [--batches 4] [--near-zero-frac 0.01]

Outputs a per-(layer, scale) table: mean |d|, median, p90, p99, max, and the
fraction of coefficients with |d| < near_zero_frac * max(|d|) (the "how sparse"
number). Also a summary across all scales.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# Import the model + dataset loader the same way generate.py / train.py do.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import WaveletLM, MultiNodeWaveletLM  # noqa: E402


def _load_config(args):
    cfg_path = args.config
    if cfg_path is None:
        # Prefer the run's own config.json next to the checkpoint, else ./config.json
        run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        candidate = os.path.join(run_dir, "config.json")
        cfg_path = candidate if os.path.exists(candidate) else "config.json"
    with open(cfg_path) as f:
        return json.load(f), cfg_path


def _load_checkpoint(model, ckpt_path):
    """Mirror generate.py's load: handle wrapped/bare formats + _orig_mod prefix."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    # strip torch.compile's _orig_mod. prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("fht.")]
    if real_missing:
        print(f"[WARNING] Missing keys (first 10): {real_missing[:10]}")
    return ckpt


def _iter_detail_tensors(captured):
    """Given a hook-captured wavelet output, yield (detail_index, tensor) for the
    detail coefficients, handling both the real path (approx, details) and the
    complex paths (approx_r, approx_i, details=[(dr,di),...])."""
    if len(captured) == 2:
        _approx, details = captured
        for i, d in enumerate(details):
            if isinstance(d, (tuple, list)):  # complex (dr, di) -> magnitude
                dr, di = d[0], d[1]
                yield i, torch.sqrt(dr.float() ** 2 + di.float() ** 2)
            else:
                yield i, d
    elif len(captured) == 3:  # invertible complex: (approx_r, approx_i, details)
        _ar, _ai, details = captured
        for i, d in enumerate(details):
            dr, di = d
            yield i, torch.sqrt(dr.float() ** 2 + di.float() ** 2)
    else:
        raise ValueError(f"Unexpected wavelet output arity: {len(captured)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--batches", type=int, default=4,
                    help="Number of held-out batches to probe (more = smoother stats).")
    ap.add_argument("--near-zero-frac", type=float, default=0.01,
                    help="A coeff counts as 'near zero' if |d| < this * per-tensor max|d|.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config, cfg_path = _load_config(args)
    print(f"[probe] config: {cfg_path}")
    print(f"[probe] checkpoint: {args.checkpoint}")
    device = torch.device(args.device)

    # Dataset: reuse the training loader for the exact same tokenization/cache.
    from train import load_and_encode_dataset

    class _NullLogger:
        def info(self, *a, **k): pass
        def __call__(self, *a, **k): pass
    try:
        _train, _val, test_data, enc, _bpt = load_and_encode_dataset(config, _NullLogger())
    except TypeError:
        # logger may be a plain callable in some versions
        _train, _val, test_data, enc, _bpt = load_and_encode_dataset(config, lambda *a, **k: None)
    vocab_size = enc.vocab_size

    if config.get("multinodal_enabled", False):
        model = MultiNodeWaveletLM(vocab_size, config, device=device)
    else:
        model = WaveletLM(vocab_size, config, device=device)
    model = model.to(device)
    _load_checkpoint(model, args.checkpoint)
    model.eval()

    # Find the per-layer lifting_wavelet modules and hook them.
    base = model.cells[0] if hasattr(model, "cells") else model
    layers = base.layers
    captured = {}

    def make_hook(idx):
        def hook(_module, _inp, out):
            captured[idx] = out
        return hook

    handles = []
    for li, layer in enumerate(layers):
        if hasattr(layer, "lifting_wavelet"):
            handles.append(layer.lifting_wavelet.register_forward_hook(make_hook(li)))
    if not handles:
        print("[probe] No lifting_wavelet modules found (wavelet_mode != 'lifting'?). Nothing to probe.")
        return

    block_size = config.get("block_size", 256)
    mbs = min(config.get("micro_batch_size", 8), 8)

    # Accumulate per-(layer, scale) magnitude stats over batches.
    # We collect raw |d| samples (subsampled) for quantiles, plus running counts.
    samples = {}   # (layer, scale) -> list of 1D fp32 tensors of |d|
    nearzero = {}  # (layer, scale) -> [near_zero_count, total_count]

    torch.manual_seed(0)
    n_tokens = test_data.numel()
    with torch.no_grad():
        for b in range(args.batches):
            # random contiguous windows from the test set
            starts = torch.randint(0, max(1, n_tokens - block_size - 1), (mbs,))
            xb = torch.stack([test_data[s:s + block_size] for s in starts]).to(device)
            captured.clear()
            model(xb)  # populates `captured` via hooks
            for li, out in captured.items():
                for si, dmag in _iter_detail_tensors(out):
                    dmag = dmag.detach().abs().flatten().float().cpu()
                    key = (li, si)
                    # subsample to bound memory (≤200k samples per key)
                    if dmag.numel() > 50000:
                        idx = torch.randint(0, dmag.numel(), (50000,))
                        dmag = dmag[idx]
                    samples.setdefault(key, []).append(dmag)
                    nz = nearzero.setdefault(key, [0, 0])
                    # near-zero computed per current tensor's max (relative threshold)
                    tmax = dmag.max().item() if dmag.numel() else 0.0
                    thr = args.near_zero_frac * tmax
                    nz[0] += int((dmag < thr).sum().item())
                    nz[1] += int(dmag.numel())

    for h in handles:
        h.remove()

    # Report.
    print()
    print(f"{'layer':>5} {'scale':>5} {'mean|d|':>10} {'median':>10} "
          f"{'p90':>10} {'p99':>10} {'max':>10} {'frac<' + str(args.near_zero_frac) + '·max':>14}")
    print("-" * 80)
    all_fracs = []
    for key in sorted(samples.keys()):
        li, si = key
        v = torch.cat(samples[key])
        q = torch.quantile(v, torch.tensor([0.5, 0.9, 0.99]))
        nz = nearzero[key]
        frac = nz[0] / max(1, nz[1])
        all_fracs.append(frac)
        print(f"{li:>5} {si:>5} {v.mean():>10.4f} {q[0]:>10.4f} "
              f"{q[1]:>10.4f} {q[2]:>10.4f} {v.max():>10.4f} {frac:>14.3f}")
    print("-" * 80)
    if all_fracs:
        mean_frac = sum(all_fracs) / len(all_fracs)
        print(f"[summary] mean near-zero fraction across all (layer,scale): {mean_frac:.3f}")
        print(f"[summary] interpretation: >~0.80 => strongly sparse (shrinkage / "
              f"low-bit detail / top-k mixing justified); <~0.50 => not sparse "
              f"(learned wavelet does not concentrate energy like fixed wavelets).")


if __name__ == "__main__":
    main()
