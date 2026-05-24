"""
Diagnostic: locate NaN/Inf in a WaveletLM checkpoint.

Usage:
    python analyze_nan_checkpoint.py <run_dir>

Example:
    python analyze_nan_checkpoint.py logs/wikitext-103_2026-05-24_07-57-45

Loads best_model.pt from the given run directory, reports:
  - Which named parameters contain NaN or Inf, with counts and percentages
  - Per-top-level-module weight statistics (param count, max |w|, NaN/Inf flags)
  - If any parameter is entirely NaN vs partially NaN (partial = instability in progress;
    total = component fully collapsed)

If best_model.pt has NO NaN weights, the forward-pass instability is numerical
(fp16 overflow on specific sequences) rather than a weight-level corruption.
"""

import sys
import torch
from pathlib import Path


def load_state(path: Path) -> dict:
    # weights_only=False is safe here; this is a trusted local checkpoint.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Checkpoint type : {type(ckpt).__name__}")
    if isinstance(ckpt, dict):
        print(f"  Top-level keys  : {list(ckpt.keys())[:10]}")
        for key in ("model_state", "model", "state_dict", "model_state_dict"):
            if key in ckpt:
                inner = ckpt[key]
                print(f"  Unwrapping key  : '{key}'  →  {type(inner).__name__}  ({len(inner)} entries)")
                return inner
        # No recognised wrapper key — assume it IS the state dict
        return ckpt
    # Bare model object saved with torch.save(model, path)
    if hasattr(ckpt, "state_dict"):
        print("  Detected bare model object — extracting state_dict()")
        return ckpt.state_dict()
    return ckpt


def analyze(state: dict, label: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    nan_entries = []
    inf_entries = []
    module_stats: dict[str, dict] = {}

    for name, param in state.items():
        if not isinstance(param, torch.Tensor) or not param.is_floating_point():
            continue

        p = param.float()
        n_nan = torch.isnan(p).sum().item()
        n_inf = torch.isinf(p).sum().item()
        total = p.numel()

        if n_nan:
            nan_entries.append((name, tuple(param.shape), n_nan, total))
        if n_inf:
            inf_entries.append((name, tuple(param.shape), n_inf, total))

        # Aggregate by top-level module name
        top = name.split(".")[0]
        s = module_stats.setdefault(top, {"params": 0, "nan": 0, "inf": 0, "max_abs": 0.0})
        s["params"] += total
        s["nan"] += n_nan
        s["inf"] += n_inf
        finite = p[torch.isfinite(p)]
        if finite.numel():
            s["max_abs"] = max(s["max_abs"], finite.abs().max().item())

    # NaN report
    if not nan_entries and not inf_entries:
        print("\n  CLEAN — no NaN or Inf in any parameter.")
        print("  Instability is forward-pass numerical (fp16 overflow on specific")
        print("  input sequences), not weight-level corruption.")
    else:
        if nan_entries:
            print(f"\n  NaN parameters ({len(nan_entries)} tensors):")
            for name, shape, n, total in sorted(nan_entries, key=lambda x: -x[2]):
                pct = 100.0 * n / total
                tag = "TOTAL" if n == total else "partial"
                print(f"    [{tag:7s}] {name}")
                print(f"             shape={shape}  NaN={n:,}/{total:,} ({pct:.1f}%)")
        if inf_entries:
            print(f"\n  Inf parameters ({len(inf_entries)} tensors):")
            for name, shape, n, total in sorted(inf_entries, key=lambda x: -x[2]):
                pct = 100.0 * n / total
                print(f"    {name}")
                print(f"      shape={shape}  Inf={n:,}/{total:,} ({pct:.1f}%)")

    # Per-module summary
    print(f"\n  Per-module summary:")
    print(f"  {'Module':<40} {'Params':>12}  {'max|w|':>10}  {'NaN':>8}  {'Inf':>8}")
    print(f"  {'-'*40} {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}")
    for mod, s in sorted(module_stats.items()):
        nan_str = str(s["nan"]) if s["nan"] else "—"
        inf_str = str(s["inf"]) if s["inf"] else "—"
        print(
            f"  {mod:<40} {s['params']:>12,}  {s['max_abs']:>10.4f}"
            f"  {nan_str:>8}  {inf_str:>8}"
        )

    # fp16 overflow risk — forward-pass NaN can occur even with clean weights
    # if activations or logits exceed fp16 max (~65504) during benchmark inference.
    # Large weight magnitudes are the most common trigger.
    FP16_MAX = 65504.0
    risk_rows = []
    for name, param in state.items():
        if not isinstance(param, torch.Tensor) or not param.is_floating_point():
            continue
        finite = param.float()
        finite = finite[torch.isfinite(finite)]
        if not finite.numel():
            continue
        max_abs = finite.abs().max().item()
        if max_abs > 100.0:
            n_over_fp16 = (finite.abs() > FP16_MAX).sum().item()
            n_over_1k   = (finite.abs() > 1000.0).sum().item()
            risk_rows.append((max_abs, name, param.shape, n_over_1k, n_over_fp16, param.numel()))

    if risk_rows:
        risk_rows.sort(reverse=True)
        print(f"\n  fp16 overflow risk (weights with max|w| > 100):")
        print(f"  {'Parameter':<60} {'max|w|':>10}  {'>1k':>8}  {'>fp16_max':>10}")
        print(f"  {'-'*60} {'-'*10}  {'-'*8}  {'-'*10}")
        for max_abs, name, shape, n1k, nfp16, total in risk_rows:
            tag = " *** FP16 OVERFLOW" if nfp16 > 0 else (" ** HIGH" if n1k > 0 else "")
            print(f"  {name:<60} {max_abs:>10.2f}  {n1k:>8}  {nfp16:>10}{tag}")
    else:
        print(f"\n  fp16 overflow risk: no parameters with max|w| > 100 — weights are well-scaled.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python analyze_nan_checkpoint.py <run_dir>")
        print("       run_dir should contain best_model.pt")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    best = run_dir / "best_model.pt"

    if not best.exists():
        print(f"Error: {best} not found.")
        sys.exit(1)

    print(f"Run directory: {run_dir}")
    print(f"Checkpoint:    {best}  ({best.stat().st_size / 1e6:.1f} MB)")

    state = load_state(best)
    analyze(state, "best_model.pt")

    total_params = sum(
        p.numel() for p in state.values()
        if isinstance(p, torch.Tensor) and p.is_floating_point()
    )
    total_nan = sum(
        torch.isnan(p.float()).sum().item() for p in state.values()
        if isinstance(p, torch.Tensor) and p.is_floating_point()
    )
    print(f"\n  Total floating-point params : {total_params:,}")
    if total_params > 0:
        print(f"  Total NaN                   : {total_nan:,}  ({100.0*total_nan/total_params:.4f}%)")
    else:
        print("  WARNING: no floating-point params found — checkpoint format not recognised.")
        print("  Re-run with TORCH_SHOW_CPP_STACKTRACES=1 or inspect ckpt keys manually.")
    print()


if __name__ == "__main__":
    main()
