"""Locate the first module emitting a non-finite value during the
non-overlapping benchmark forward pass.

When a benchmark BPB returns `inf` despite training itself producing finite
losses, the cause is fp16 overflow somewhere in the eval-mode forward path
under specific input statistics. This script walks the model's forward
under exact-benchmark conditions, registering forward hooks that check each
module's output for non-finite values, and reports the FIRST module to fail.

Mirrors the conditions of `evaluate_full_validation` in train.py:
  - model.eval()
  - decompose_bypass_cross_window = False
  - fwpkm.inference_updates = False
  - AMP fp16 autocast
  - same per-window data slicing
  - no torch.compile (so module names in the trace are clean)

Usage:
    python find_inf_source.py --checkpoint logs/wikitext-103_2026-05-03_10-38-19/best_model.pt

Output: first module that emits a non-finite output, plus its input/output
statistics so the upstream context is visible.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

import train as train_mod
from model import WaveletLM


# ============================================================================
# Tensor utilities
# ============================================================================

def first_nonfinite_tensor(t):
    """Walk a possibly-nested structure, return first tensor with any
    non-finite values, else None."""
    if isinstance(t, torch.Tensor):
        if not torch.isfinite(t).all().item():
            return t
        return None
    if isinstance(t, (tuple, list)):
        for x in t:
            r = first_nonfinite_tensor(x)
            if r is not None:
                return r
    return None


def tensor_summary(t):
    """Return dict with shape, dtype, max|finite|, count of non-finite, total."""
    if not isinstance(t, torch.Tensor):
        return None
    finite_mask = torch.isfinite(t)
    if finite_mask.any():
        max_abs = float(t[finite_mask].abs().max().item())
    else:
        max_abs = float("inf")
    nonfin = int((~finite_mask).sum().item())
    total = t.numel()
    first_nonfin_idx = None
    if nonfin > 0:
        flat_nonfin = (~finite_mask).view(-1).nonzero()
        if len(flat_nonfin) > 0:
            first_nonfin_idx = int(flat_nonfin[0].item())
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "max_abs_finite": max_abs,
        "nonfinite_count": nonfin,
        "total_elem": total,
        "first_nonfinite_flat_idx": first_nonfin_idx,
    }


# ============================================================================
# Hook installer
# ============================================================================

class InfDetector:
    """Forward-hook detector that records the FIRST module (in execution
    order) to produce a non-finite output. Stops further reporting after
    the first failure."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.first_failure = None
        # Build qualified-name lookup once
        self.module_names = {id(m): n for n, m in model.named_modules()}

    def install(self):
        def make_hook(mod):
            name = self.module_names.get(id(mod), type(mod).__name__)
            klass = type(mod).__name__

            def hook(_module, inputs, output):
                if self.first_failure is not None:
                    return
                bad = first_nonfinite_tensor(output)
                if bad is None:
                    return
                # Capture input stats too — helps see whether bad input
                # was passed in or was generated INSIDE this op.
                in_summaries = []
                seq = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                for i, t in enumerate(seq):
                    s = tensor_summary(t) if isinstance(t, torch.Tensor) else None
                    if s is not None:
                        s["arg_idx"] = i
                        in_summaries.append(s)
                self.first_failure = {
                    "module_name": name,
                    "module_class": klass,
                    "output_summary": tensor_summary(bad),
                    "input_summaries": in_summaries,
                }
            return hook

        for mod in self.model.modules():
            self.handles.append(mod.register_forward_hook(make_hook(mod)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================================
# Checkpoint + model loading
# ============================================================================

def unwrap_state_dict(ckpt) -> dict:
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    for key in ("model_state", "model"):
        inner = ckpt.get(key)
        if isinstance(inner, dict) and inner:
            return inner
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError("Could not unwrap state dict from checkpoint")


def load_model_and_state(checkpoint_path: str, vocab_size: int, config: dict, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = unwrap_state_dict(ckpt)
    # Strip torch.compile's `_orig_mod.` prefix if present
    fixed = {}
    for k, v in sd.items():
        fixed[k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k] = v
    model = WaveletLM(vocab_size=vocab_size, config=config, device=device).to(device)
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing:
        print(f"  WARN: {len(missing)} missing key(s); first 3: {missing[:3]}")
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected key(s); first 3: {unexpected[:3]}")
    return model


# ============================================================================
# Match evaluate_full_validation's eval-mode setup
# ============================================================================

def configure_for_benchmark(model: nn.Module):
    model.eval()
    if hasattr(model, "reset_semantic_state"):
        model.reset_semantic_state()
    if hasattr(model, "decompose_bypass_cross_window"):
        model.decompose_bypass_cross_window = False
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "fwpkm") and getattr(layer.fwpkm, "inference_updates", False):
            layer.fwpkm.inference_updates = False


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best_model.pt to investigate")
    ap.add_argument("--config", default=None,
                    help="Path to config.json (default: sibling of checkpoint)")
    ap.add_argument("--max_windows", type=int, default=17,
                    help="Stop after this many windows (default: full WT-103 non-overlap = 17)")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu")
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    config_path = args.config or os.path.join(os.path.dirname(args.checkpoint), "config.json")
    if not os.path.isfile(config_path):
        sys.exit(f"No config.json at: {config_path}")
    config = json.load(open(config_path))

    print(f"=== find_inf_source ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config:     {config_path}")
    print(f"  Device:     {args.device}")

    # Tokenizer + dataset (reuse train.py's loader, but silent)
    class _SilentLogger:
        def log(self, _msg): pass
    train_data, val_data, test_data, enc, _bpb = train_mod.load_and_encode_dataset(
        config, _SilentLogger())
    vocab_size = enc.vocab_size

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type != "cuda" and args.device == "cuda":
        print("  WARN: CUDA requested but unavailable — falling back to CPU")

    model = load_model_and_state(args.checkpoint, vocab_size, config, device)
    configure_for_benchmark(model)

    detector = InfDetector(model)
    detector.install()

    # Match the benchmark's window setup
    T = config["block_size"]
    eval_len = len(test_data)
    num_windows = (eval_len - 1) // T
    use_amp = config.get("use_amp", True)
    amp_dtype = torch.float16 if config.get("amp_dtype", "fp16") == "fp16" else torch.bfloat16
    n = min(num_windows, args.max_windows)

    print(f"\n  Test set: {eval_len:,} tokens, {num_windows} non-overlap windows of length {T}")
    print(f"  AMP enabled: {use_amp}, dtype: {amp_dtype}")
    print(f"  Will check up to {n} windows.\n")

    failed_window = None
    with torch.no_grad():
        for w in range(n):
            offset = w * T
            X = test_data[offset:offset + T].unsqueeze(0).to(device)
            detector.first_failure = None
            ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
                   if device.type == "cuda" else torch.autocast(device_type="cpu",
                                                                 dtype=amp_dtype, enabled=False))
            with ctx:
                _logits, _ = model(X, targets=None)

            if detector.first_failure is not None:
                failed_window = w
                break
            print(f"  Window {w:>3}: all module outputs finite ✓")

    detector.remove()

    print()
    if failed_window is None:
        print(f"  No non-finite outputs across {n} windows.")
        print("  This is unexpected if the source benchmark produced inf — verify")
        print("  the checkpoint, config, and AMP dtype actually match the failing run.")
        return

    f = detector.first_failure
    out = f["output_summary"]
    print("=" * 76)
    print("  FIRST NON-FINITE OUTPUT")
    print("=" * 76)
    print(f"  Window:           {failed_window}  (test offset {failed_window*T:,}–{(failed_window+1)*T:,})")
    print(f"  Module path:      {f['module_name']}")
    print(f"  Module class:     {f['module_class']}")
    print(f"  Output shape:     {out['shape']}  dtype={out['dtype']}")
    print(f"  Output max |x| (finite-only): {out['max_abs_finite']:.4g}")
    print(f"  Output non-finite: {out['nonfinite_count']:,} / {out['total_elem']:,} "
          f"({100*out['nonfinite_count']/out['total_elem']:.2f}%)")
    if out["first_nonfinite_flat_idx"] is not None:
        flat_idx = out["first_nonfinite_flat_idx"]
        shape = out["shape"]
        if len(shape) >= 2:
            # Numpy-style unravel: walk shape right-to-left
            idx, rem = [], flat_idx
            for s in reversed(shape):
                idx.append(rem % s)
                rem //= s
            multi = tuple(reversed(idx))
            print(f"  Output first non-finite (multi-index): {multi}")
        else:
            print(f"  Output first non-finite flat idx: {flat_idx}")

    print(f"\n  Module inputs at the moment of failure:")
    if not f["input_summaries"]:
        print("    (no tensor inputs captured)")
    for s in f["input_summaries"]:
        print(f"    arg[{s['arg_idx']}]  shape={s['shape']}  dtype={s['dtype']}  "
              f"max|x|(finite)={s['max_abs_finite']:.4g}  "
              f"nonfin={s['nonfinite_count']:,}/{s['total_elem']:,}")
        if s["nonfinite_count"] > 0:
            print(f"        ↑ Input ALREADY had non-finite values — true source is upstream of this module.")
        else:
            print(f"        ↑ Input was finite — this module's computation produced the non-finite value.")


if __name__ == "__main__":
    main()
