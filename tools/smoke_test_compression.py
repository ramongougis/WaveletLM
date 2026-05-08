"""Pod-side smoke test for the compression-revert fix.

Run on the training pod (Linux, GPU) to verify the forward-multiply revert
of StructuredLinear / SparsePQEmbedding / MaskedTiedLinear works correctly
under realistic conditions: torch.compile=True, AMP fp16, and short
training-loop steps.

Tests cover the exact failure mode from the overnight runs:
  - sparse_pq_embedding_enabled=True with tied LM head
  - mlp_offdiag_structure=banded at the failing density (W=128)
  - run a handful of optimizer steps and check loss stays finite

Usage from repo root: `python tools/smoke_test_compression.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `from model import WaveletLM` to work whether this script is run
# from the repo root (`python tools/smoke_test_compression.py`) or from
# anywhere else. Inserts the repo root (parent of tools/) at the front
# of sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F


def _build_cfg(
    base_cfg_path: str = "config.json",
    *,
    sparse_pq: bool = False,
    sparse_pq_density: float = 0.10,
    sparse_pq_mode: str = "structural",
    mlp_struct: str = "none",
    mlp_band_width: int = 64,
    mlp_block_size: int = 64,
    mlp_pq_density: float = 0.10,
    compile: bool = True,
    use_amp: bool = True,
) -> dict:
    cfg = json.load(open(base_cfg_path))
    cfg.update({
        "C": 2048,
        "layers": 1,
        "epochs": 1,
        "levels": 7,
        "low_rank": 16,
        "mlp_expansion": 10,
        "mlp_layers": 2,
        "pkm_enabled": False,
        "fwpkm_enabled": True,
        "fwpkm_num_keys": 8281,
        "fwpkm_top_k": 32,
        "fwpkm_heads": 1,
        "block_size": 16384,
        "tie_embedding_to_lm_head": True,
        "micro_batch_size": 1,
        "grad_accum": 1,
        "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
        "wavelet_crawl": False,
        "lifting_diaglowrank": False,
        "lifting_offdiag_structure": "none",
        "compile": compile,
        "use_amp": use_amp,
        "amp_dtype": "fp16",
        "lr": 0.01,
        "min_lr": 0.0002,
        "warmup_fraction": 0.3,
        "grad_clip": 1.0,
        "multinodal_enabled": False,
        "sparse_pq_embedding_enabled": sparse_pq,
        "sparse_pq_embedding_density": sparse_pq_density,
        "sparse_pq_embedding_mode": sparse_pq_mode,
        "mlp_offdiag_structure": mlp_struct,
        "mlp_band_width": mlp_band_width,
        "mlp_block_size": mlp_block_size,
        "mlp_pq_density": mlp_pq_density,
        "mlp_pq_mode": "structural",
    })
    return cfg


def _step_loop(model, optimizer, scaler, use_amp: bool, n_steps: int = 12,
               vocab: int = 50257, T: int = 256, device: str = "cuda"):
    """Run a short optimizer loop. Returns list of losses (one per step)."""
    losses = []
    grad_finite_history = []
    nonmask_weight_max_history = []
    model.train()

    for step in range(n_steps):
        ids = torch.randint(0, vocab, (1, T), device=device)
        labels = torch.randint(0, vocab, (1, T), device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(ids, labels)
            loss = out[1] if isinstance(out, tuple) else out

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())

        # Verify gradient finite + non-mask weight positions stay at zero
        if hasattr(model, "token_embedding") and hasattr(model.token_embedding, "pq_mask"):
            mask = model.token_embedding.pq_mask
            w = model.token_embedding.weight
            nm_max = (w * (~mask)).abs().max().item()
            nonmask_weight_max_history.append(nm_max)

        all_finite = all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters() if p.grad is not None
        )
        grad_finite_history.append(all_finite)

    return losses, grad_finite_history, nonmask_weight_max_history


def run_test(label: str, **cfg_kwargs):
    """Build a WaveletLM, run a short loop, report PASS/FAIL."""
    print(f"\n[{label}]")
    cfg = _build_cfg(**cfg_kwargs)

    # Imports done inside so a failure to import doesn't kill the script
    from model import WaveletLM
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337)
    model = WaveletLM(50257, cfg).to(device)

    if cfg["compile"]:
        model = torch.compile(model, mode="default")

    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=cfg["lr"], eps=2e-13, weight_decay=1e-06
    )

    scaler = torch.cuda.amp.GradScaler() if cfg["use_amp"] else None

    losses, grad_finite, nm_max = _step_loop(
        model, optimizer, scaler, cfg["use_amp"],
        n_steps=12, vocab=50257, T=256, device=device,
    )

    finite_count = sum(1 for v in losses if v == v and v != float("inf") and v != float("-inf"))
    grad_pass_count = sum(1 for v in grad_finite if v)
    nm_max_overall = max(nm_max) if nm_max else 0.0

    print(f"  losses: {[f'{x:.3f}' for x in losses]}")
    print(f"  finite losses: {finite_count}/{len(losses)}")
    print(f"  finite grads: {grad_pass_count}/{len(grad_finite)}")
    if nm_max:
        print(f"  max |weight| at non-mask positions across run: {nm_max_overall:.3e}")

    pass_ = (
        finite_count == len(losses)
        and grad_pass_count == len(grad_finite)
        and nm_max_overall < 1e-3  # tolerate fp16 numerical noise
    )
    print(f"  RESULT: {'PASS' if pass_ else 'FAIL'}")
    return pass_


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARN: CUDA not available; running CPU-only without AMP/compile")

    has_cuda = torch.cuda.is_available()
    use_amp = has_cuda
    use_compile = has_cuda

    overall = []
    overall.append(run_test(
        "sparse embedding d=10% structural (was failing)",
        sparse_pq=True,
        sparse_pq_density=0.10,
        sparse_pq_mode="structural",
        compile=use_compile,
        use_amp=use_amp,
    ))
    overall.append(run_test(
        "MLP banded W=128 at 12.5% density (was failing)",
        mlp_struct="banded",
        mlp_band_width=128,
        compile=use_compile,
        use_amp=use_amp,
    ))
    overall.append(run_test(
        "MLP block_diagonal b=256 (was failing)",
        mlp_struct="block_diagonal",
        mlp_block_size=256,
        compile=use_compile,
        use_amp=use_amp,
    ))
    overall.append(run_test(
        "MLP pq_strided d=0.25 (was failing)",
        mlp_struct="pq_strided",
        mlp_pq_density=0.25,
        compile=use_compile,
        use_amp=use_amp,
    ))

    print(f"\n{'='*60}")
    print(f"OVERALL: {sum(overall)}/{len(overall)} configurations passed")
    print(f"{'='*60}")
    sys.exit(0 if all(overall) else 1)
