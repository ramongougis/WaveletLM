"""Per-component runtime profiler for WaveletLM.

Identifies which architectural component dominates step time at varying
block_size. Loads the existing config.json, builds the model, and runs
warmup + measured steps wrapped in torch.profiler. Emits:

  1. A printed summary table (top kernels by self CUDA time, plus a
     module-attributed breakdown of LiftingWaveletDecompose,
     FastHadamardTransform, GatedSpectralMixer, MLP, FastWeightPKM, etc.).
  2. A Chrome trace per block_size at logs/profile_<bs>_<timestamp>.json
     openable in https://ui.perfetto.dev or chrome://tracing.

Usage:
    python profile_step.py
    python profile_step.py --block_sizes 256,1024,4096,16384
    python profile_step.py --no_compile      # easier-to-read trace, slower
    python profile_step.py --warmup 3 --steps 8

Designed to NOT touch model.py — uses forward hooks to wrap each named
submodule in a record_function region, plus the profiler's built-in
with_modules attribution.
"""
import argparse
import contextlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import train as train_mod
from model import WaveletLM


COMPONENTS_TO_LABEL = (
    "LiftingWaveletDecompose",
    "LiftingWaveletReconstruct",
    "FastHadamardTransform",
    "GatedSpectralMixer",
    "MLP",
    "PKM",
    "FastWeightPKM",
    "WaveletLMBlock",
)


def _component_label(module: torch.nn.Module) -> str | None:
    """Return a short label like 'mlp' / 'lifting_decompose' for components
    we care about, else None."""
    name = type(module).__name__
    if name not in COMPONENTS_TO_LABEL:
        return None
    return name


def install_record_function_hooks(model: torch.nn.Module):
    """Wrap forward() of each interesting submodule in a record_function
    region. Stack-based via per-module entry/exit hooks."""
    stack: list = []

    def make_pre_hook(label):
        def hook(_mod, _inputs):
            ctx = record_function(label)
            ctx.__enter__()
            stack.append(ctx)
        return hook

    def make_post_hook():
        def hook(_mod, _inputs, _outputs):
            if stack:
                stack.pop().__exit__(None, None, None)
        return hook

    handles = []
    for name, mod in model.named_modules():
        label = _component_label(mod)
        if label is None:
            continue
        marker = f"WLM::{label}"
        handles.append(mod.register_forward_pre_hook(make_pre_hook(marker)))
        handles.append(mod.register_forward_hook(make_post_hook()))
    return handles


def _self_device_time(evt) -> float:
    """Get self GPU time across PyTorch versions.
    Newer PyTorch (>=2.4) renamed `self_cuda_time_total` →
    `self_device_time_total`. Fall back to either."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


def summarize_self_time_by_label(prof: profile, prefix: str) -> dict[str, float]:
    """Aggregate self GPU time (us) by labels matching `prefix`, summed
    across all profiled steps."""
    totals: dict[str, float] = defaultdict(float)
    for evt in prof.key_averages():
        name = evt.key
        if not name.startswith(prefix):
            continue
        label = name[len(prefix):]
        totals[label] += _self_device_time(evt)
    return dict(totals)


def summarize_module_time(prof: profile, target_classes: tuple[str, ...]) -> dict[str, float]:
    """Aggregate self GPU time (us) by module class name. Falls back when
    record_function hooks get traced away by torch.compile — uses the
    profiler's built-in `with_modules=True` attribution which fires on
    nn.Module __call__ at the Python layer, surviving compilation as long
    as the module boundary itself isn't inlined.

    Looks for events named like 'nn.Module: <ClassName>_N' or just
    '<ClassName>'."""
    import re
    totals: dict[str, float] = defaultdict(float)
    pattern = re.compile(r"(?:nn\.Module:\s*)?(" + "|".join(re.escape(c) for c in target_classes) + r")(?:_\d+)?$")
    for evt in prof.key_averages():
        name = evt.key
        m = pattern.search(name)
        if m:
            totals[m.group(1)] += _self_device_time(evt)
    return dict(totals)


def run_profile(block_size: int, args, base_config: dict, train_data, val_data):
    config = dict(base_config)
    config["block_size"] = block_size
    # Force MBS=1 by default at very large block_size to avoid OOM in profiling;
    # caller can override via base_config / config.json.
    if block_size >= 4096 and "micro_batch_size" not in args.overrides:
        config["micro_batch_size"] = 1
        config["grad_accum"] = 1
    print(f"\n{'='*72}")
    print(f"  Profiling: block_size={block_size}, MBS={config['micro_batch_size']}, "
          f"GA={config['grad_accum']}, levels={config['levels']}, "
          f"compile={'no' if args.no_compile else 'yes'}")
    print(f"{'='*72}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required.")

    model = WaveletLM(vocab_size=args.vocab_size, config=config, device=device).to(device)
    model.train()

    if not args.no_compile and config.get("compile", True):
        compile_mode = config.get("compile_mode", "reduce-overhead")
        model = torch.compile(model, mode=compile_mode)

    install_record_function_hooks(model)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=config["lr"],
                                    eps=config.get("optimizer_eps", 1e-10))

    get_batch = train_mod.make_get_batch(train_data, val_data, config, device)
    use_amp = config.get("use_amp", True)
    amp_dtype = torch.float16 if config.get("amp_dtype", "fp16") == "fp16" else torch.bfloat16

    # Warmup
    for i in range(args.warmup):
        X, Y = get_batch("train")
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(X, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"  Warmup ({args.warmup} steps) complete. Profiling {args.steps} steps...")

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    trace_path = (f"logs/profile_bs{block_size}_"
                  f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")

    t0 = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        with_modules=True,
        with_flops=False,
    ) as prof:
        for i in range(args.steps):
            X, Y = get_batch("train")
            with record_function("WLM::step"):
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    with record_function("WLM::forward"):
                        _, loss = model(X, Y)
                with record_function("WLM::backward"):
                    loss.backward()
                with record_function("WLM::optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    per_step_ms = wall * 1000.0 / args.steps

    os.makedirs("logs", exist_ok=True)
    prof.export_chrome_trace(trace_path)
    print(f"\n  Wall time: {wall:.3f}s ({per_step_ms:.1f} ms/step over {args.steps} steps)")
    print(f"  Chrome trace: {trace_path}")
    print(f"  → open in https://ui.perfetto.dev or chrome://tracing\n")

    # Top-N kernels by self CUDA time
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=args.top_kernels,
        max_name_column_width=70,
    ))

    # Component-attributed summary. Two passes:
    #   1) Try explicit WLM:: record_function labels (works for un-compiled
    #      paths; torch.compile traces away the hooks and these come up empty).
    #   2) Fall back to nn.Module class-name attribution from
    #      with_modules=True, which fires at Python __call__ time and survives
    #      compilation as long as the module boundary isn't inlined away.
    comp_totals = summarize_self_time_by_label(prof, "WLM::")
    if not comp_totals:
        comp_totals = summarize_module_time(prof, COMPONENTS_TO_LABEL)
        attribution_note = ("(via nn.Module class-name attribution; "
                            "explicit record_function hooks were dropped by torch.compile)"
                            if not args.no_compile else
                            "(via nn.Module class-name attribution)")
    else:
        attribution_note = "(via WLM:: record_function regions)"

    if comp_totals:
        print(f"\n  --- WaveletLM component totals (self GPU time, "
              f"summed over {args.steps} steps) ---")
        print(f"  {attribution_note}")
        total_us = sum(comp_totals.values()) or 1.0
        wall_us = wall * 1e6
        print(f"  {'Component':<28} {'us / step':>12} {'% of comp total':>18} {'% of wall':>12}")
        for label, us in sorted(comp_totals.items(), key=lambda x: -x[1]):
            pct_comp = 100 * us / total_us
            pct_wall = 100 * us / wall_us
            print(f"  {label:<28} {us / args.steps:>12.0f} {pct_comp:>17.1f}% {pct_wall:>11.1f}%")
        print(f"  {'(sum of attributed)':<28} {total_us / args.steps:>12.0f} {'100.0':>17}% "
              f"{100 * total_us / wall_us:>11.1f}%")

    # VRAM headline
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    print(f"\n  Peak VRAM: allocated {peak_alloc:.0f} MiB, "
          f"reserved {peak_reserved:.0f} MiB")

    # Cleanup
    del model, optimizer, prof
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_sizes", default="256,1024,4096,16384",
                    help="Comma-separated list of block_size values to profile")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--no_compile", action="store_true",
                    help="Disable torch.compile for clearer kernel attribution")
    ap.add_argument("--top_kernels", type=int, default=25,
                    help="How many top kernels to print per block_size")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--overrides", default="",
                    help="Comma-separated 'key=value' overrides (parsed as JSON)")
    args = ap.parse_args()

    with open(args.config) as f:
        base_config = json.load(f)

    # Apply overrides
    args.overrides = dict(
        kv.split("=", 1) for kv in args.overrides.split(",") if "=" in kv
    )
    for k, v in args.overrides.items():
        try:
            base_config[k] = json.loads(v)
        except json.JSONDecodeError:
            base_config[k] = v

    # Load tokenizer + dataset (uses train.py's cached path if present)
    class _SilentLogger:
        def log(self, _msg): pass
    train_data, val_data, _test, enc, _bpb = train_mod.load_and_encode_dataset(
        base_config, _SilentLogger())
    args.vocab_size = enc.vocab_size

    block_sizes = [int(b) for b in args.block_sizes.split(",") if b.strip()]
    for bs in block_sizes:
        run_profile(bs, args, base_config, train_data, val_data)


if __name__ == "__main__":
    main()
