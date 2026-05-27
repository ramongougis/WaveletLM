# Copyright 2025-2026 Ramon Gougis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# WaveletLM
# model.py

import os
import random
import warnings
import math
import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=r".*Online softmax is disabled.*")
warnings.filterwarnings("ignore", message=r".*Not enough SMs to use max_autotune_gemm.*")

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as parametrize_spectral_norm
from typing import List, Tuple, Dict, Optional
from torch.utils.checkpoint import checkpoint

# Structural-prior infrastructure for the lifting predict/update Linears.
# See tools/lifting_constraints.py for the StructuredLinear / MonarchLinear
# classes and the make_structural_mask / build_structured_lifting_linear /
# init_structured_lifting_linear / effective_param_count helpers.
from tools.lifting_constraints import (
    StructuredLinear,
    MonarchLinear,
    build_structured_lifting_linear,
    build_structured_mlp_linear,
    init_structured_lifting_linear,
    effective_param_count,
)

# ==============================================================================
# 1. UTILITIES
# ==============================================================================

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def pad_features_to_pow2(x: torch.Tensor, C_pad: int):
    _, _, C = x.shape
    if C == C_pad:
        return x
    return F.pad(x, (0, C_pad - C))

# ==============================================================================
# 2. FREQUENCY SPACE - Fast Hadamard Transform
# ==============================================================================

def fwht_ortho_iterative(x: torch.Tensor) -> torch.Tensor:
    """Iterative 'Butterfly' FWHT. Memory efficient (N log N), but kernel-heavy.
    Best for very large dimensions (C >= 2048)."""
    y = x
    n = y.shape[-1]
    h = 1
    while h < n:
        y = y.reshape(y.shape[:-1] + (-1, 2, h))
        a, b = y.unbind(dim=-2)
        c1 = a + b
        c2 = a - b
        y = torch.stack([c1, c2], dim=-2)
        y = y.reshape(y.shape[:-3] + (-1,))
        h *= 2
    scale = torch.rsqrt(torch.tensor(n, device=x.device, dtype=x.dtype))
    return y * scale

class FastHadamardTransform(nn.Module):
    """Hybrid FWHT: matrix multiply for dim < 2048, iterative butterfly otherwise."""

    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.use_matrix = (dim < 2048)
        if self.use_matrix:
            H = self._make_hadamard(dim) / math.sqrt(dim)
            self.register_buffer("H", H.to(device=device, dtype=dtype), persistent=False)
        else:
            self.register_buffer("H", None, persistent=False)

    def _make_hadamard(self, n):
        if n == 1:
            return torch.tensor([[1.0]])
        h = self._make_hadamard(n // 2)
        return torch.cat([
            torch.cat([h, h], dim=1),
            torch.cat([h, -h], dim=1)
        ], dim=0)

    def forward(self, x):
        if self.use_matrix:
            return torch.matmul(x, self.H)
        else:
            return fwht_ortho_iterative(x)

# ==============================================================================
# 3. WAVELET TRANSFORMS
# ==============================================================================

def causal_haar_decompose(x: torch.Tensor, levels: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Fixed causal Haar wavelet decomposition."""
    details: List[torch.Tensor] = []
    cur = x
    for level in range(levels):
        dilation = 1 << level
        padded = F.pad(cur, (0, 0, dilation, 0))
        past = padded[:, :-dilation, :]
        approx = (cur + past) * 0.70710678118
        detail = (cur - past) * 0.70710678118
        details.append(detail)
        cur = approx
    return cur, details

def causal_haar_reconstruct(approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
    """Inverse causal Haar: x[t] = (approx[t] + detail[t]) / sqrt(2)."""
    inv_sqrt2 = 0.7071067811865476
    cur = approx
    for level in range(len(details) - 1, -1, -1):
        cur = (cur + details[level]) * inv_sqrt2
    return cur

class DiagonalLowRankLinear(nn.Module):
    """Compressed Linear with W = diag(D) + U @ V parameterization.

    For an effective C×C weight matrix:
        D ∈ R^C       diagonal vector
        U ∈ R^(C×r)   low-rank left factor
        V ∈ R^(r×C)   low-rank right factor

    Total params: C + 2·C·r vs C² for full Linear. At C=2048, r=16:
    67,584 vs 4,194,304 (98.4% reduction). At C=2048, r=64: 264,192
    (93.7% reduction). Forward: y = x·D + (x · V^T) · U^T + b.
    """
    def __init__(self, dim, rank, bias=True, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        f = {'device': device, 'dtype': dtype}
        self.D = nn.Parameter(torch.zeros(dim, **f))
        self.U = nn.Parameter(torch.empty(dim, rank, **f))
        self.V = nn.Parameter(torch.empty(rank, dim, **f))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim, **f))
        else:
            self.register_parameter('bias', None)
        nn.init.normal_(self.U, std=0.01)
        nn.init.normal_(self.V, std=0.01)

    @torch.no_grad()
    def init_identity(self, scale: float = 1.0):
        """Set W = scale · I (D=scale, U=0, V=0). Used for haar lifting init."""
        self.D.fill_(scale)
        self.U.zero_()
        self.V.zero_()
        if self.bias is not None:
            self.bias.zero_()

    @torch.no_grad()
    def init_zero(self):
        """Set W = 0 (D=0, U=0, V=0). Used for zero lifting init."""
        self.D.zero_()
        self.U.zero_()
        self.V.zero_()
        if self.bias is not None:
            self.bias.zero_()

    @torch.no_grad()
    def init_random(self, std: float = 0.01):
        """Small-std random init across D, U, V."""
        nn.init.normal_(self.D, std=std)
        nn.init.normal_(self.U, std=std)
        nn.init.normal_(self.V, std=std)
        if self.bias is not None:
            self.bias.zero_()

    @torch.no_grad()
    def scale_by(self, factor: float):
        """Multiply D by factor in-place (for stab_lifting_level_scaling).
        Affects only the diagonal component; U/V correction left as-is."""
        self.D.mul_(factor)

    def forward(self, x):
        y = x * self.D
        y = y + (x @ self.V.transpose(-2, -1)) @ self.U.transpose(-2, -1)
        if self.bias is not None:
            y = y + self.bias
        return y


def _load_lifting_reference_weights(
    checkpoint_path: str,
) -> Dict[Tuple[int, str, int], torch.Tensor]:
    """Load a trained checkpoint and extract the lifting predict/update weights
    keyed by (level, role, lin_idx).

    Used by lifting_offdiag_structure='magnitude_topk' to rank off-diagonal
    positions by magnitude in the loaded reference. Returns a dict like
    {(0, 'predict', 0): tensor(C, C), (0, 'predict', 3): tensor(C, C),
     (0, 'update', 0): tensor(C, C), (0, 'update', 3): tensor(C, C),
     ...}
    Handles both `model_state` and `model` top-level keys, plus bare state-dict
    checkpoints.
    """
    import re
    import sys

    def _friendly_exit(reason: str):
        """Print a compact, multi-line warning explaining the bootstrapping
        dependency, list architecturally-portable alternatives, then exit.
        Both the warning and the alternatives go to stdout, which train.py
        captures into log.txt via its standard tee."""
        print()
        print("[Lifting] !!! magnitude_topk: reference checkpoint missing or invalid !!!")
        print(f"[Lifting]   Path:   {checkpoint_path!r}")
        print(f"[Lifting]   Reason: {reason}")
        print("[Lifting]")
        print("[Lifting]   magnitude_topk requires a pre-trained reference at the SAME architecture")
        print("[Lifting]   (matching C, levels, low_rank) to compute per-matrix masks. New scales,")
        print("[Lifting]   new architectures, and first-time runs cannot use it without first")
        print("[Lifting]   training an uncompressed reference (2-stage training).")
        print("[Lifting]")
        print("[Lifting]   Architecturally portable alternatives -- set lifting_offdiag_structure to:")
        print("[Lifting]     random_topk          random mask at matched density (bootstrap-free)")
        print("[Lifting]     upper_triangular     50% mask")
        print("[Lifting]     lower_triangular     50% mask")
        print("[Lifting]     block_diagonal       set lifting_block_size (e.g. 64 or 256)")
        print("[Lifting]     banded               set lifting_band_width (e.g. 64 or 256)")
        print("[Lifting]     monarch              set lifting_monarch_blocks (e.g. 32 or 64)")
        print("[Lifting]")
        print("[Lifting]   Or train an uncompressed reference first and point")
        print("[Lifting]   `lifting_offdiag_mask_checkpoint` at its best_model.pt.")
        print("[Lifting]   Exiting.")
        print()
        sys.exit(1)

    if not checkpoint_path:
        _friendly_exit("lifting_offdiag_mask_checkpoint is empty")
    if not os.path.isfile(checkpoint_path):
        _friendly_exit("file does not exist")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        _friendly_exit(f"torch.load failed: {type(e).__name__}: {e}")
    # Unwrap state_dict
    state_dict = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state", "model"):
            inner = ckpt.get(key)
            if isinstance(inner, dict) and inner:
                state_dict = inner
                break
    if not isinstance(state_dict, dict) or not all(
        isinstance(v, torch.Tensor) for v in state_dict.values()
    ):
        _friendly_exit("could not unwrap state_dict from checkpoint")
    # Match keys like:
    #   layers.0.lifting_wavelet.predict_nets.<L>.<lin>.weight
    #   layers.0.lifting_reconstruct.decompose.predict_nets.<L>.<lin>.weight
    # Pull the first match per (level, role, lin_idx) — under shared_lifting_weights
    # both paths reference the same tensor, so deduplication via dict is safe.
    pattern = re.compile(
        r"^.*?\.(?:lifting_wavelet|lifting_reconstruct\.decompose)"
        r"\.(?P<role>predict|update)_nets"
        r"\.(?P<level>\d+)\.(?P<lin>\d+)\.weight$"
    )
    out: Dict[Tuple[int, str, int], torch.Tensor] = {}
    for key, tensor in state_dict.items():
        m = pattern.match(key)
        if m is None:
            continue
        triple = (int(m.group("level")), m.group("role"), int(m.group("lin")))
        if triple not in out:
            out[triple] = tensor.detach().to(torch.float32)
    if not out:
        _friendly_exit(
            f"no lifting predict/update weights found "
            f"(sample keys: {list(state_dict.keys())[:5]})"
        )
    return out


def _lifting_compression_group(level: int, num_levels: int) -> int:
    """3-group level-sharing assignment for the compressed lifting cascade.

    For levels=7, returns groups derived from analyze_lifting.py's cross-level
    cosine matrix on the L=1 / 5-epoch winner — adjacent levels share more
    direction (cos 0.74-0.82) and fall into a 3-block pattern. For other
    levels values, splits evenly into 3 groups by index. Re-running
    analyze_lifting.py at a different `levels` is recommended before relying
    on the heuristic split for that configuration."""
    if num_levels <= 3:
        return level  # one group per level — no meaningful sharing
    if num_levels == 7:
        return 0 if level <= 1 else (1 if level <= 4 else 2)
    third = num_levels / 3.0
    return 0 if level < third else (1 if level < 2 * third else 2)


class LiftingWaveletDecompose(nn.Module):
    """Learnable wavelet decomposition via parameterized lifting scheme.

    Split → Predict → Update at each level, with causal dilated access.
    Perfect reconstruction guaranteed by construction.
    """

    def __init__(
        self,
        levels: int,
        C: int,
        hidden_mult: int = 1,
        init_wavelet: str = 'haar',
        dropout: float = 0.0,
        linear_only: bool = False,
        device=None,
        dtype=None,
        stab_lifting_level_scaling: bool = False,
        wavelet_crawl: bool = False,
        wavelet_crawl_k: int = 3,
        lifting_diaglowrank: bool = False,
        lifting_level_sharing: bool = False,
        # Structural / top-k off-diagonal priors (Option B unified flag).
        # When set, replaces the unconstrained Linear(C, C) in the lifting
        # predict/update Sequentials with a constrained variant. See
        # tools/lifting_constraints.py for the supported `lifting_offdiag_structure`
        # values and their per-structure parameters.
        lifting_offdiag_structure: str = "none",
        lifting_block_size: int = 64,
        lifting_band_width: int = 64,
        lifting_monarch_blocks: int = 32,
        lifting_offdiag_density: float = 0.0,
        lifting_offdiag_mask_seed: int = 1337,
        lifting_reference_weights: Optional[Dict[Tuple[int, str, int], torch.Tensor]] = None,
    ):
        super().__init__()
        self.levels = levels
        self.C = C
        self.hidden_mult = hidden_mult
        self.init_wavelet = init_wavelet
        self.linear_only = linear_only
        self.wavelet_crawl = wavelet_crawl
        self.wavelet_crawl_k = wavelet_crawl_k
        self.lifting_diaglowrank = lifting_diaglowrank
        self.lifting_level_sharing = lifting_level_sharing
        self.lifting_offdiag_structure = lifting_offdiag_structure
        self.lifting_block_size = lifting_block_size
        self.lifting_band_width = lifting_band_width
        self.lifting_monarch_blocks = lifting_monarch_blocks
        self.lifting_offdiag_density = lifting_offdiag_density
        self.lifting_offdiag_mask_seed = lifting_offdiag_mask_seed

        # Mutual exclusion: structural priors are incompatible with the existing
        # diaglowrank / level_sharing / linear_only / hidden_mult>1 paths.
        if lifting_offdiag_structure not in (None, "none"):
            if lifting_diaglowrank or lifting_level_sharing:
                raise ValueError(
                    "lifting_offdiag_structure is mutually exclusive with "
                    "lifting_diaglowrank and lifting_level_sharing."
                )
            if linear_only or hidden_mult != 1:
                raise ValueError(
                    "lifting_offdiag_structure requires linear_only=False "
                    "and hidden_mult=1 (the analyzed lifting configuration)."
                )

        # Compression validity: only supports the standard hidden_mult=1, non-
        # linear-only path. Other configurations weren't covered by the
        # analyze_lifting.py study and would need their own analysis.
        if (lifting_diaglowrank or lifting_level_sharing) and (
            linear_only or hidden_mult != 1
        ):
            raise ValueError(
                "lifting_diaglowrank / lifting_level_sharing require "
                "linear_only=False and hidden_mult=1 (the analyzed config)."
            )

        # Per-matrix-type ranks from analyze_lifting.py: three diagonal-dominant
        # types (avg diag energy 71-76%, cross-level cos 0.74-0.77) compress at
        # r=16. The outlier update_nets[L].3 (48.7% diag energy, 0.54 cross-
        # level cos) needs r=64 and is never level-shared.
        PREDICT_RANK = 16
        UPDATE_0_RANK = 16
        UPDATE_3_RANK = 64
        self._diaglowrank_ranks = (PREDICT_RANK, PREDICT_RANK, UPDATE_0_RANK, UPDATE_3_RANK)
        if wavelet_crawl:
            if wavelet_crawl_k % 2 != 1:
                raise ValueError(f"wavelet_crawl_k must be odd, got {wavelet_crawl_k}")
            # Precompute K distinct positive-integer offsets per level. At level 0
            # base_dilation=1 and a symmetric ±half spread would underflow into 0,
            # so we shift the window upward to keep all K offsets distinct and >=1.
            # The init biases the softmax toward whichever offset == base_dilation.
            half = wavelet_crawl_k // 2
            self._crawl_offsets = []
            base_idx_per_level = []
            for level in range(levels):
                base = 1 << level
                min_off = max(1, base - half)
                offsets = list(range(min_off, min_off + wavelet_crawl_k))
                self._crawl_offsets.append(offsets)
                base_idx_per_level.append(offsets.index(base))
            self.dilation_logits = nn.Parameter(
                torch.zeros(levels, wavelet_crawl_k, device=device, dtype=dtype)
            )
            with torch.no_grad():
                for level, base_idx in enumerate(base_idx_per_level):
                    self.dilation_logits.data[level, base_idx] = 5.0

        hidden_dim = C * hidden_mult

        self.predict_nets = nn.ModuleList()
        self.update_nets = nn.ModuleList()

        # Compressed-lifting paths share three of four matrix types across level
        # groups (predict.0, predict.3, update.0). update.3 is per-level always.
        # Pre-build the shared module pool keyed by group when level_sharing.
        if lifting_diaglowrank and lifting_level_sharing:
            unique_groups = sorted({
                _lifting_compression_group(l, levels) for l in range(levels)
            })
            shared_lifting_modules = {
                g: {
                    'predict_0': DiagonalLowRankLinear(C, PREDICT_RANK, bias=True, device=device, dtype=dtype),
                    'predict_3': DiagonalLowRankLinear(C, PREDICT_RANK, bias=True, device=device, dtype=dtype),
                    'update_0':  DiagonalLowRankLinear(C, UPDATE_0_RANK, bias=True, device=device, dtype=dtype),
                }
                for g in unique_groups
            }
            # Init shared modules once (subsequent level reuses don't re-init)
            for g in unique_groups:
                pm = shared_lifting_modules[g]
                if init_wavelet == 'haar':
                    pm['predict_0'].init_identity(1.0)
                    pm['predict_3'].init_identity(1.0)
                    pm['update_0'].init_identity(1.0)
                elif init_wavelet == 'zero':
                    pm['predict_0'].init_zero()
                    pm['predict_3'].init_zero()
                    pm['update_0'].init_zero()
                elif init_wavelet == 'random':
                    pm['predict_0'].init_random()
                    pm['predict_3'].init_random()
                    pm['update_0'].init_random()
        else:
            shared_lifting_modules = None

        for level in range(levels):
            if linear_only:
                # Compression validity check above ensures linear_only is False
                # when compression is on, so this path is the original behavior.
                predict = nn.Linear(C, C, device=device, dtype=dtype)
                update = nn.Linear(C, C, device=device, dtype=dtype)

                if init_wavelet == 'haar':
                    nn.init.eye_(predict.weight)
                    nn.init.zeros_(predict.bias)
                    update.weight.data.copy_(0.5 * torch.eye(
                        C, device=device,
                        dtype=dtype if dtype else torch.float32))
                    nn.init.zeros_(update.bias)
                elif init_wavelet == 'zero':
                    nn.init.zeros_(predict.weight)
                    nn.init.zeros_(predict.bias)
                    nn.init.zeros_(update.weight)
                    nn.init.zeros_(update.bias)
                elif init_wavelet == 'random':
                    # Small-std random init for multi-basis diversity. Default Kaiming
                    # can cause NaN through the signal path in multi-basis mode; a tighter
                    # std with zero bias keeps the second wavelet's contribution bounded.
                    nn.init.normal_(predict.weight, std=0.01)
                    nn.init.zeros_(predict.bias)
                    nn.init.normal_(update.weight, std=0.01)
                    nn.init.zeros_(update.bias)
            elif lifting_diaglowrank:
                # Compressed Sequential: replace the two Linear(C, C) layers
                # at indices 0 and 3 with DiagonalLowRankLinear modules.
                if lifting_level_sharing:
                    g = _lifting_compression_group(level, levels)
                    predict_0 = shared_lifting_modules[g]['predict_0']
                    predict_3 = shared_lifting_modules[g]['predict_3']
                    update_0  = shared_lifting_modules[g]['update_0']
                else:
                    predict_0 = DiagonalLowRankLinear(C, PREDICT_RANK, bias=True, device=device, dtype=dtype)
                    predict_3 = DiagonalLowRankLinear(C, PREDICT_RANK, bias=True, device=device, dtype=dtype)
                    update_0  = DiagonalLowRankLinear(C, UPDATE_0_RANK, bias=True, device=device, dtype=dtype)
                # update[3] is the analyzer-flagged outlier — never shared, rank=64
                update_3 = DiagonalLowRankLinear(C, UPDATE_3_RANK, bias=True, device=device, dtype=dtype)

                predict = nn.Sequential(predict_0, nn.GELU(), nn.Dropout(dropout), predict_3)
                update  = nn.Sequential(update_0,  nn.GELU(), nn.Dropout(dropout), update_3)

                # Init for non-shared modules: shared modules were init'd above.
                if not lifting_level_sharing:
                    if init_wavelet == 'haar':
                        predict_0.init_identity(1.0)
                        predict_3.init_identity(1.0)
                        update_0.init_identity(1.0)
                    elif init_wavelet == 'zero':
                        predict_0.init_zero()
                        predict_3.init_zero()
                        update_0.init_zero()
                    elif init_wavelet == 'random':
                        predict_0.init_random()
                        predict_3.init_random()
                        update_0.init_random()
                # update_3 (per-level outlier) — init regardless of sharing
                if init_wavelet == 'haar':
                    update_3.init_identity(0.5)  # matches existing 0.5*I haar init for update[3]
                elif init_wavelet == 'zero':
                    update_3.init_zero()
                elif init_wavelet == 'random':
                    update_3.init_random()
            elif lifting_offdiag_structure not in (None, "none"):
                # Structural / top-k off-diagonal prior path. Replaces the four
                # Linear(C, C) in predict/update Sequentials with constrained
                # variants from tools/lifting_constraints.py. The validity
                # check at the top of __init__ ensures hidden_mult=1 (so
                # hidden_dim == C and these Linears are square).
                def _build_lifting_linear(role: str, lin_idx: int):
                    ref_w = None
                    if (
                        lifting_offdiag_structure == "magnitude_topk"
                        and lifting_reference_weights is not None
                    ):
                        ref_w = lifting_reference_weights.get((level, role, lin_idx))
                        if ref_w is None:
                            raise ValueError(
                                f"magnitude_topk: missing reference weight for "
                                f"(level={level}, role={role}, lin_idx={lin_idx}). "
                                f"Available keys: {sorted(lifting_reference_weights.keys())}"
                            )
                    return build_structured_lifting_linear(
                        C, C, lifting_offdiag_structure,
                        block_size=lifting_block_size,
                        band_width=lifting_band_width,
                        monarch_blocks=lifting_monarch_blocks,
                        reference_weight=ref_w,
                        density=lifting_offdiag_density,
                        seed=lifting_offdiag_mask_seed,
                        device=device, dtype=dtype,
                    )
                predict_0 = _build_lifting_linear("predict", 0)
                predict_3 = _build_lifting_linear("predict", 3)
                update_0  = _build_lifting_linear("update",  0)
                update_3  = _build_lifting_linear("update",  3)

                predict = nn.Sequential(predict_0, nn.GELU(), nn.Dropout(dropout), predict_3)
                update  = nn.Sequential(update_0,  nn.GELU(), nn.Dropout(dropout), update_3)

                # Haar-style init: predict_0/predict_3/update_0 → I, update_3 → 0.5*I.
                init_structured_lifting_linear(predict_0, init_wavelet, scale=1.0)
                init_structured_lifting_linear(predict_3, init_wavelet, scale=1.0)
                init_structured_lifting_linear(update_0,  init_wavelet, scale=1.0)
                init_structured_lifting_linear(
                    update_3, init_wavelet,
                    scale=0.5 if init_wavelet == "haar" else 1.0,
                )
            else:
                predict = nn.Sequential(
                    nn.Linear(C, hidden_dim, device=device, dtype=dtype),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, C, device=device, dtype=dtype),
                )
                update = nn.Sequential(
                    nn.Linear(C, hidden_dim, device=device, dtype=dtype),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, C, device=device, dtype=dtype),
                )

                if init_wavelet == 'haar':
                    nn.init.eye_(predict[0].weight[:C, :])
                    if hidden_mult > 1:
                        nn.init.normal_(predict[0].weight[C:, :], std=0.01)
                    nn.init.zeros_(predict[0].bias)
                    nn.init.eye_(predict[3].weight[:, :C])
                    if hidden_mult > 1:
                        nn.init.normal_(predict[3].weight[:, C:], std=0.01)
                    nn.init.zeros_(predict[3].bias)

                    nn.init.eye_(update[0].weight[:C, :])
                    if hidden_mult > 1:
                        nn.init.normal_(update[0].weight[C:, :], std=0.01)
                    nn.init.zeros_(update[0].bias)
                    nn.init.zeros_(update[3].weight)
                    update[3].weight.data[:, :C] = 0.5 * torch.eye(
                        C, device=device, dtype=dtype if dtype else torch.float32)
                    if hidden_mult > 1:
                        nn.init.normal_(update[3].weight[:, C:], std=0.01)
                    nn.init.zeros_(update[3].bias)

                elif init_wavelet == 'zero':
                    nn.init.zeros_(predict[3].weight)
                    nn.init.zeros_(predict[3].bias)
                    nn.init.zeros_(update[3].weight)
                    nn.init.zeros_(update[3].bias)
                elif init_wavelet == 'random':
                    # Small-std random init for multi-basis diversity; see comment above.
                    # Targets both Linear layers in the Sequential (predict[0] and predict[3]).
                    nn.init.normal_(predict[0].weight, std=0.01)
                    nn.init.zeros_(predict[0].bias)
                    nn.init.normal_(predict[3].weight, std=0.01)
                    nn.init.zeros_(predict[3].bias)
                    nn.init.normal_(update[0].weight, std=0.01)
                    nn.init.zeros_(update[0].bias)
                    nn.init.normal_(update[3].weight, std=0.01)
                    nn.init.zeros_(update[3].bias)

            if stab_lifting_level_scaling:
                # Damp higher-level (longer-range) interactions where signal-to-noise
                # is weaker. Equivalent to predict_scale = update_scale = 1/(1 + 0.1*l).
                lvl_scale = 1.0 / (1.0 + level * 0.1)
                with torch.no_grad():
                    if linear_only:
                        predict.weight.data.mul_(lvl_scale)
                        update.weight.data.mul_(lvl_scale)
                    elif lifting_diaglowrank:
                        # update[3] is per-level — safe to scale every iteration.
                        # predict[3] is shared across a level group when
                        # lifting_level_sharing — scaling it per-level would
                        # compound across levels in the same group, so only
                        # scale once per group (the first level).
                        update[3].scale_by(lvl_scale)
                        if (not lifting_level_sharing) or (
                            level == 0 or
                            _lifting_compression_group(level, levels) !=
                            _lifting_compression_group(level - 1, levels)
                        ):
                            predict[3].scale_by(lvl_scale)
                    elif lifting_offdiag_structure not in (None, "none"):
                        # StructuredLinear has .weight (mask is reapplied at forward);
                        # MonarchLinear has .scale_by() that scales the L factor.
                        for net in (predict, update):
                            mod = net[3]
                            if isinstance(mod, MonarchLinear):
                                mod.scale_by(lvl_scale)
                            else:
                                mod.weight.data.mul_(lvl_scale)
                    else:
                        predict[3].weight.data.mul_(lvl_scale)
                        update[3].weight.data.mul_(lvl_scale)

            self.predict_nets.append(predict)
            self.update_nets.append(update)

        self.inv_sqrt2 = 0.7071067811865476

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        details: List[torch.Tensor] = []
        cur = x

        for level in range(self.levels):
            base_dilation = 1 << level
            T = cur.shape[1]

            if self.wavelet_crawl:
                K = self.wavelet_crawl_k
                offsets = self._crawl_offsets[level]  # K distinct positive ints
                weights = F.softmax(self.dilation_logits[level], dim=0)
                max_d = offsets[-1]  # offsets are sorted ascending by construction
                padded = F.pad(cur, (0, 0, max_d, 0))
                # Sum_k w_k * cur shifted-back-by-offsets[k]
                odd = sum(
                    weights[k] * padded[:, max_d - offsets[k]:max_d - offsets[k] + T, :]
                    for k in range(K)
                )
            else:
                padded = F.pad(cur, (0, 0, base_dilation, 0))
                odd = padded[:, :-base_dilation, :]

            even = cur

            predicted = self.predict_nets[level](even)
            detail = (odd - predicted) * self.inv_sqrt2

            update = self.update_nets[level](detail)
            approx = (even + update) * self.inv_sqrt2

            details.append(detail)
            cur = approx

        return cur, details

class LiftingWaveletReconstruct(nn.Module):
    """Inverse lifting wavelet transform. Shares P/U networks with Decompose."""

    def __init__(self, decompose_module: LiftingWaveletDecompose):
        super().__init__()
        self.decompose = decompose_module
        self.inv_sqrt2 = 0.7071067811865476
        self.sqrt2 = 1.4142135623730951

    def forward(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        cur = approx
        num_levels = len(details)
        for level in range(num_levels - 1, -1, -1):
            detail = details[level]
            update = self.decompose.update_nets[level](detail)
            cur = cur * self.sqrt2 - update
        return cur

class MultiBasisLiftingWavelet(nn.Module):
    """K parallel learnable lifting wavelets with per-scale learned blending.

    Each constituent wavelet runs in parallel; outputs are blended scale-by-scale
    via softmax(basis_weights). Init biases the softmax toward wavelet 0 so
    initial behavior matches a single wavelet (other bases learn into use).
    """

    def __init__(self, levels: int, C: int, hidden_mult: int, inits: List[str],
                 dropout: float, linear_only: bool, device=None, dtype=None,
                 stab_lifting_level_scaling: bool = False,
                 wavelet_crawl: bool = False, wavelet_crawl_k: int = 3):
        super().__init__()
        self.K = len(inits)
        self.levels = levels
        self.wavelets = nn.ModuleList([
            LiftingWaveletDecompose(
                levels=levels, C=C, hidden_mult=hidden_mult,
                init_wavelet=init, dropout=dropout, linear_only=linear_only,
                device=device, dtype=dtype,
                stab_lifting_level_scaling=stab_lifting_level_scaling,
                wavelet_crawl=wavelet_crawl, wavelet_crawl_k=wavelet_crawl_k,
            ) for init in inits
        ])
        self.basis_weights = nn.Parameter(
            torch.zeros(self.K, levels + 1, device=device, dtype=dtype)
        )
        with torch.no_grad():
            self.basis_weights.data[0, :] = 10.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        outs = [w(x) for w in self.wavelets]
        weights = F.softmax(self.basis_weights, dim=0)  # (K, levels+1)
        approx = sum(weights[k, 0] * outs[k][0] for k in range(self.K))
        num_levels = len(outs[0][1])
        details = [
            sum(weights[k, lvl + 1] * outs[k][1][lvl] for k in range(self.K))
            for lvl in range(num_levels)
        ]
        return approx, details

class MultiBasisLiftingReconstruct(nn.Module):
    """K parallel reconstruction paths blended with the multi-basis softmax weights.

    Each constituent wavelet's reconstruct is applied to the (mixer-processed)
    coefficients, then outputs are blended with the same softmax over basis_weights
    as decomposition (using the approx-level row, since reconstruction starts from approx).
    """

    def __init__(self, multi_basis: MultiBasisLiftingWavelet):
        super().__init__()
        self.multi_basis = multi_basis  # weight source (no extra params)
        self.reconstructs = nn.ModuleList([
            LiftingWaveletReconstruct(w) for w in multi_basis.wavelets
        ])

    def forward(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        outs = [r(approx, details) for r in self.reconstructs]
        weights = F.softmax(self.multi_basis.basis_weights, dim=0)
        return sum(weights[k, 0] * outs[k] for k in range(self.multi_basis.K))

# ==============================================================================
# 4. SPECTRAL MIXER
# ==============================================================================

_MIXER_GATE_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}

class GatedSpectralMixer(nn.Module):
    def __init__(self, Cp: int, num_blocks: int, rank: int = 4, eps: float = 1e-3,
                 use_mixer_gate: bool = True, mixer_gate_activation: str = "sigmoid",
                 add_bias: bool = False, device=None, dtype=None,
                 stab_spectral_norm: bool = False,
                 stab_mixer_eps_scaling: bool = False):
        super().__init__()
        self.Cp = Cp
        self.use_mixer_gate = use_mixer_gate
        if stab_mixer_eps_scaling:
            eps = eps / math.sqrt(Cp)
        self.mixer = nn.Linear(Cp, Cp, bias=False, device=device, dtype=dtype)
        if use_mixer_gate:
            if mixer_gate_activation not in _MIXER_GATE_ACTIVATIONS:
                raise ValueError(f"Unknown mixer_gate_activation: {mixer_gate_activation!r}. "
                                 f"Options: {list(_MIXER_GATE_ACTIVATIONS.keys())}")
            self.gate_activation = _MIXER_GATE_ACTIVATIONS[mixer_gate_activation]
            self.gate = nn.Linear(Cp, Cp, bias=False, device=device, dtype=dtype)

        if rank > 0:
            self.U = nn.Parameter(torch.empty(Cp, rank, device=device, dtype=dtype))
            self.V = nn.Parameter(torch.empty(Cp, rank, device=device, dtype=dtype))
        else:
            self.U = None
            self.V = None

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(Cp, device=device, dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(eps)

        if stab_spectral_norm:
            # Constrain mixer to ||W||_2 = 1 (largest singular value); prevents
            # signal amplification through the mixer that drove NaN at depth/LR.
            self.mixer = parametrize_spectral_norm(self.mixer)

    def reset_parameters(self, eps=1e-3):
        with torch.no_grad():
            self.mixer.weight.data.copy_(torch.eye(self.Cp, device=self.mixer.weight.device))
            self.mixer.weight.data.add_(torch.randn_like(self.mixer.weight) * eps)
            if self.use_mixer_gate:
                nn.init.normal_(self.gate.weight, std=0.02)
            if self.U is not None:
                nn.init.normal_(self.U, std=0.01)
                nn.init.normal_(self.V, std=0.01)

    def forward(self, X_spec: torch.Tensor, gate_input: torch.Tensor = None,
                cached_gate: torch.Tensor = None, return_gate: bool = False):
        signal = self.mixer(X_spec)
        gate = None
        if self.use_mixer_gate:
            if cached_gate is not None:
                gate = cached_gate  # reuse a gate computed on an earlier step
            else:
                gi = gate_input if gate_input is not None else X_spec
                gate = self.gate_activation(self.gate(gi))
            out = signal * gate
        else:
            out = signal
        if self.U is not None:
            mid = torch.matmul(X_spec, self.V)
            out = out + torch.matmul(mid, self.U.t())
        if self.bias is not None:
            out = out + self.bias
        if return_gate:
            return out, gate
        return out

class PerScaleMixer(nn.Module):
    """Wraps GatedSpectralMixer with Cp <-> width projections for asymmetric
    per-scale capacity. When width == Cp, projections are skipped (no overhead)."""

    def __init__(self, Cp: int, width: int, num_blocks: int = 1, rank: int = 4,
                 eps: float = 1e-3, use_mixer_gate: bool = True,
                 mixer_gate_activation: str = "sigmoid", add_bias: bool = False,
                 device=None, dtype=None,
                 stab_spectral_norm: bool = False,
                 stab_mixer_eps_scaling: bool = False):
        super().__init__()
        self.Cp = Cp
        self.width = width
        if width != Cp:
            self.proj_in = nn.Linear(Cp, width, bias=False, device=device, dtype=dtype)
            self.proj_out = nn.Linear(width, Cp, bias=False, device=device, dtype=dtype)
            with torch.no_grad():
                nn.init.normal_(self.proj_in.weight, std=1.0 / math.sqrt(Cp))
                nn.init.normal_(self.proj_out.weight, std=1.0 / math.sqrt(width))
        else:
            self.proj_in = None
            self.proj_out = None
        self.mixer = GatedSpectralMixer(
            Cp=width, num_blocks=num_blocks, rank=rank, eps=eps,
            use_mixer_gate=use_mixer_gate, mixer_gate_activation=mixer_gate_activation,
            add_bias=add_bias, device=device, dtype=dtype,
            stab_spectral_norm=stab_spectral_norm,
            stab_mixer_eps_scaling=stab_mixer_eps_scaling,
        )

    def forward(self, X: torch.Tensor, gate_input: torch.Tensor = None):
        if self.proj_in is not None:
            X = self.proj_in(X)
            if gate_input is not None:
                gate_input = self.proj_in(gate_input)
        Y = self.mixer(X, gate_input=gate_input)
        if self.proj_out is not None:
            Y = self.proj_out(Y)
        return Y

# ==============================================================================
# 5. FEED-FORWARD (MLP)
# ==============================================================================

class FeedForward(nn.Module):
    def __init__(self, C, expansion=2, dropout_mlp=0.0, hidden_layers=2,
                 stab_ff_scaling: bool = False,
                 mlp_offdiag_structure: str = "none",
                 mlp_block_size: int = 64,
                 mlp_band_width: int = 64,
                 mlp_pq_density: float = 0.1,
                 mlp_pq_mode: str = "structural"):
        super().__init__()
        hidden_dim = C * expansion

        def _make_linear(in_f, out_f):
            if mlp_offdiag_structure in (None, "none"):
                return nn.Linear(in_f, out_f)
            return build_structured_mlp_linear(
                in_f, out_f, mlp_offdiag_structure,
                block_size=mlp_block_size,
                band_width=mlp_band_width,
                pq_density=mlp_pq_density,
                pq_mode=mlp_pq_mode,
                tile_C=C,
                outer_dim=hidden_dim,
            )

        layers = []
        layers.append(_make_linear(C, hidden_dim))
        for _ in range(hidden_layers - 2):
            layers.append(nn.GELU())
            layers.append(_make_linear(hidden_dim, hidden_dim))
        if hidden_layers >= 2:
            layers.append(nn.GELU())
        layers.append(_make_linear(hidden_dim, C))
        layers.append(nn.Dropout(dropout_mlp))
        self.net = nn.Sequential(*layers)
        final_linear = self.net[-2]
        with torch.no_grad():
            if stab_ff_scaling:
                # Xavier-like: keep variance constant regardless of hidden_dim.
                # Replaces fixed 0.02 which doesn't scale with MLP expansion.
                final_linear.weight.mul_(1.0 / math.sqrt(hidden_dim))
            else:
                final_linear.weight.mul_(0.02)
            final_linear.bias.zero_()
            if hasattr(final_linear, '_struct_mask'):
                final_linear.weight.mul_(final_linear._struct_mask)

    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 5b. PRODUCT KEY MEMORY (PKM)
# ==============================================================================

class ProductKeyMemory(nn.Module):
    """Product Key Memory (Lample et al. 2019).

    Splits the query into two halves, each matched against a codebook of
    sqrt(num_keys) sub-keys. The top-k product scores index into a value
    table of num_keys entries. This gives O(2*sqrt(num_keys)) lookup cost
    instead of O(num_keys).
    """

    def __init__(self, C: int, num_keys: int = 529, pkm_top_k: int = 32,
                 pkm_heads: int = 1, device=None, dtype=None):
        super().__init__()
        self.C = C
        self.num_keys = num_keys
        self.top_k = pkm_top_k
        self.heads = pkm_heads
        self.head_dim = C // pkm_heads

        # Sub-key size: sqrt(num_keys) entries per half
        self.sub_keys = int(math.sqrt(num_keys))
        assert self.sub_keys ** 2 == num_keys, \
            f"num_keys ({num_keys}) must be a perfect square"

        half_dim = self.head_dim // 2
        # Per-half top-k is clamped to sub_keys (can't retrieve more than exist)
        self.half_k = min(self.top_k, self.sub_keys)

        # Query projection: input -> two half-queries per head
        self.query_proj = nn.Linear(C, pkm_heads * self.head_dim, bias=False,
                                    device=device, dtype=dtype)

        # Two codebooks of sub-keys per head
        self.keys_a = nn.Parameter(
            torch.randn(pkm_heads, self.sub_keys, half_dim,
                        device=device, dtype=dtype) * 0.02)
        self.keys_b = nn.Parameter(
            torch.randn(pkm_heads, self.sub_keys, half_dim,
                        device=device, dtype=dtype) * 0.02)

        # Value table: num_keys entries per head, each maps to head_dim
        self.values = nn.EmbeddingBag(
            num_keys * pkm_heads, C, mode='sum',
            device=device, dtype=dtype)
        nn.init.normal_(self.values.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to queries: [B, T, heads, head_dim]
        q = self.query_proj(x).view(B, T, self.heads, self.head_dim)

        # Split each query into two halves
        half = self.head_dim // 2
        q_a = q[..., :half]  # [B, T, heads, half]
        q_b = q[..., half:]  # [B, T, heads, half]

        # Score against sub-key codebooks: [B, T, heads, sub_keys]
        scores_a = torch.einsum('bthd,hsd->bths', q_a, self.keys_a)
        scores_b = torch.einsum('bthd,hsd->bths', q_b, self.keys_b)

        # Top-k from each half (clamped to sub_keys)
        top_a_scores, top_a_idx = scores_a.topk(self.half_k, dim=-1)  # [B,T,H,hk]
        top_b_scores, top_b_idx = scores_b.topk(self.half_k, dim=-1)

        # Product scores: all hk*hk combinations -> top-k of those
        prod_scores = top_a_scores.unsqueeze(-1) + top_b_scores.unsqueeze(-2)
        prod_scores_flat = prod_scores.view(B, T, self.heads, -1)  # [B,T,H,hk*hk]

        # Combined indices: a_idx * sub_keys + b_idx
        prod_idx = (top_a_idx.unsqueeze(-1) * self.sub_keys +
                    top_b_idx.unsqueeze(-2))
        prod_idx_flat = prod_idx.view(B, T, self.heads, -1)

        # Select final top-k from hk*hk candidates
        final_top_k = min(self.top_k, prod_scores_flat.size(-1))
        final_scores, final_pos = prod_scores_flat.topk(final_top_k, dim=-1)
        final_idx = prod_idx_flat.gather(-1, final_pos)

        # Softmax weights over final top-k
        weights = F.softmax(final_scores.float(), dim=-1).to(x.dtype)

        # Offset indices per head for the shared EmbeddingBag
        head_offsets = torch.arange(self.heads, device=x.device) * self.num_keys
        final_idx = final_idx + head_offsets.view(1, 1, -1, 1)

        # Weighted lookup: flatten for EmbeddingBag
        flat_idx = final_idx.reshape(-1, final_top_k)
        flat_w = weights.reshape(-1, final_top_k)

        out = self.values(flat_idx, per_sample_weights=flat_w)  # [B*T*H, C]
        out = out.view(B, T, self.heads, C)

        # Sum across heads
        return out.sum(dim=2)  # [B, T, C]

# ==============================================================================
# 5b. FAST-WEIGHT PRODUCT KEY MEMORY (FwPKM)
# ==============================================================================

class FastWeightPKM(nn.Module):
    """
    Fast-Weight Product Key Memory.

    Structurally identical to ProductKeyMemory during training. At inference,
    an optional update_fast_weights() method updates value deltas per chunk,
    enabling episodic/contextual memory without retraining.
    """

    def __init__(self, C: int, num_keys: int = 529, top_k: int = 32,
                 heads: int = 1, device=None, dtype=None):
        super().__init__()
        self.C = C
        self.num_keys = num_keys
        self.top_k = top_k
        self.heads = heads
        self.head_dim = C // heads

        # Sub-key size: sqrt(num_keys) entries per half
        self.sub_keys = int(math.sqrt(num_keys))
        assert self.sub_keys ** 2 == num_keys, \
            f"num_keys ({num_keys}) must be a perfect square"

        half_dim = self.head_dim // 2
        self.half_k = min(self.top_k, self.sub_keys)

        # Query projection
        self.query_proj = nn.Linear(C, heads * self.head_dim, bias=False,
                                    device=device, dtype=dtype)

        # Two codebooks of sub-keys per head
        self.keys_a = nn.Parameter(
            torch.randn(heads, self.sub_keys, half_dim,
                        device=device, dtype=dtype) * 0.02)
        self.keys_b = nn.Parameter(
            torch.randn(heads, self.sub_keys, half_dim,
                        device=device, dtype=dtype) * 0.02)

        # Value table: num_keys entries per head, each maps to C
        self.values = nn.EmbeddingBag(
            num_keys * heads, C, mode='sum',
            device=device, dtype=dtype)
        nn.init.normal_(self.values.weight, std=0.02)

        # Fast-weight deltas: same shape as values, zeroed during training
        self.register_buffer(
            'value_deltas',
            torch.zeros(num_keys * heads, C, device=device,
                        dtype=dtype if dtype else torch.float32))

    def _pkm_lookup(self, x: torch.Tensor):
        """Core PKM lookup returning output and indices/weights for updates.

        Returns:
            out: [B, T, C] retrieved values
            flat_idx: [B*T*H, k] indices used
            flat_w: [B*T*H, k] softmax weights used
        """
        B, T, C = x.shape

        # Project to queries: [B, T, heads, head_dim]
        q = self.query_proj(x).view(B, T, self.heads, self.head_dim)

        # Split each query into two halves
        half = self.head_dim // 2
        q_a = q[..., :half]
        q_b = q[..., half:]

        # Score against sub-key codebooks
        scores_a = torch.einsum('bthd,hsd->bths', q_a, self.keys_a)
        scores_b = torch.einsum('bthd,hsd->bths', q_b, self.keys_b)

        # Top-k from each half (clamped to sub_keys)
        top_a_scores, top_a_idx = scores_a.topk(self.half_k, dim=-1)
        top_b_scores, top_b_idx = scores_b.topk(self.half_k, dim=-1)

        # Product scores: all hk*hk combinations -> top-k of those
        prod_scores = top_a_scores.unsqueeze(-1) + top_b_scores.unsqueeze(-2)
        prod_scores_flat = prod_scores.view(B, T, self.heads, -1)

        prod_idx = (top_a_idx.unsqueeze(-1) * self.sub_keys +
                    top_b_idx.unsqueeze(-2))
        prod_idx_flat = prod_idx.view(B, T, self.heads, -1)

        # Select final top-k from hk*hk candidates
        final_top_k = min(self.top_k, prod_scores_flat.size(-1))
        final_scores, final_pos = prod_scores_flat.topk(final_top_k, dim=-1)
        final_idx = prod_idx_flat.gather(-1, final_pos)

        # Softmax weights over final top-k
        weights = F.softmax(final_scores.float(), dim=-1).to(x.dtype)

        # Offset indices per head
        head_offsets = torch.arange(self.heads, device=x.device) * self.num_keys
        final_idx = final_idx + head_offsets.view(1, 1, -1, 1)

        # Flatten for lookup
        flat_idx = final_idx.reshape(-1, final_top_k)
        flat_w = weights.reshape(-1, final_top_k)

        # Weighted lookup from base values
        out = self.values(flat_idx, per_sample_weights=flat_w)

        # Add fast-weight delta contribution (always when grad needed, else only if nonzero)
        if self.value_deltas.requires_grad or self.value_deltas.any():
            delta_vals = self.value_deltas[flat_idx]  # [B*T*H, k, C]
            delta_out = (flat_w.unsqueeze(-1) * delta_vals).sum(dim=1)  # [B*T*H, C]
            out = out + delta_out

        out = out.view(B, T, self.heads, C)
        return out.sum(dim=2), flat_idx, flat_w  # [B, T, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = self._pkm_lookup(x)
        return out

    def update_fast_weights(self, queries: torch.Tensor,
                            targets: torch.Tensor, lr: float = 0.01):
        """Update value deltas using local MSE loss. Called between chunks
        during inference only.

        Args:
            queries: [B, chunk_size, C] - input representations from this chunk
            targets: [B, chunk_size, C] - lookahead targets (shifted by 1)
            lr: learning rate for fast-weight update
        """
        # Temporarily enable grad for value_deltas
        self.value_deltas.requires_grad_(True)

        # Retrieve current values
        retrieved, flat_idx, flat_w = self._pkm_lookup(queries)

        # MSE loss against targets
        loss = F.mse_loss(retrieved, targets)

        # Gradient w.r.t. value_deltas only
        grad = torch.autograd.grad(loss, self.value_deltas, retain_graph=False)[0]

        # Update deltas (gradient descent)
        with torch.no_grad():
            self.value_deltas -= lr * grad
            self.value_deltas.requires_grad_(False)

    def reset_fast_weights(self):
        """Reset deltas to zero. Call at start of new document/sequence."""
        self.value_deltas.zero_()

# ==============================================================================
# 6. DECOMPOSE BYPASS - Running Mean
# ==============================================================================

@torch.compiler.disable
def _compute_running_mean(x: torch.Tensor, prev_mean: torch.Tensor = None,
                          prev_count: int = 0) -> torch.Tensor:
    """Causal running mean along the time dimension.

    Cumsum is done in fp32 even when the input is fp16, mirroring how the
    sister `_compute_data_dependent_ema` casts its scan to fp32. Without
    the cast, fp16's max representable value (~65504) is exceeded once
    T grows past a few thousand positions for residual values of magnitude
    > ~4 — exactly the regime we hit at deeper `levels` and/or longer T.
    Symptom if it triggers during training: gate_bias_scales becomes inf,
    contaminating logits and corrupting weights at the next optimizer step
    (visible as a training NaN). This was a latent risk at our configs but
    has not been definitively confirmed as the cause of any specific NaN;
    fixing preventively because the fp32 cast is essentially free.

    Decorated with @torch.compiler.disable to avoid cumsum compilation issues."""
    T = x.size(1)
    orig_dtype = x.dtype
    x_f32 = x.float()
    if prev_mean is not None and prev_count > 0:
        prev_sum = prev_mean.float().unsqueeze(1) * prev_count
        history_sum = torch.cumsum(x_f32, dim=1) + prev_sum
        divisors = torch.arange(prev_count + 1, prev_count + T + 1,
                                device=x.device, dtype=torch.float32).view(1, -1, 1)
    else:
        history_sum = torch.cumsum(x_f32, dim=1)
        divisors = torch.arange(1, T + 1, device=x.device, dtype=torch.float32).view(1, -1, 1)
    return (history_sum / divisors).to(orig_dtype)

@torch.compiler.disable
def _compute_data_dependent_ema(x: torch.Tensor,
                                gate_weight: torch.Tensor,
                                gate_bias: torch.Tensor,
                                prev_ema: torch.Tensor = None) -> torch.Tensor:
    T = x.size(1)
    alphas = torch.sigmoid(F.linear(x, gate_weight, gate_bias))  # [B, T, C]

    # Cast scan state to fp32
    A = alphas.float()
    B_out = ((1.0 - alphas) * x).float()

    if prev_ema is not None:
        B_out = B_out.clone()
        B_out[:, 0] = B_out[:, 0] + alphas[:, 0].float() * prev_ema.float()

    # Slice-based scan
    step = 1
    while step < T:
        new_A = A.clone()
        new_B = B_out.clone()
        A_shifted = A[:, :T - step]
        B_shifted = B_out[:, :T - step]
        # Compose: outer = current (later position), inner = shifted (earlier)
        new_A[:, step:] = A[:, step:] * A_shifted
        new_B[:, step:] = A[:, step:] * B_shifted + B_out[:, step:]
        A = new_A
        B_out = new_B
        step *= 2

    return B_out.to(x.dtype)

# ==============================================================================
# 7. WaveletLM BLOCK
# ==============================================================================

class WaveletLMBlock(nn.Module):
    def __init__(
        self,
        C: int,
        levels: int = 3,
        low_rank: int = 4,
        mlp_expansion: int = 2,
        mlp_layers: int = 2,
        dropout_projection: float = 0.0,
        dropout_mixer: float = 0.0,
        dropout_mlp: float = 0.0,
        device=None,
        dtype=None,
        decompose_bypass: bool = True,
        decompose_bypass_ema: bool = False,
        wavelet_mode: str = "lifting",
        lifting_hidden_mult: int = 1,
        lifting_init: str = "haar",
        lifting_dropout: float = 0.0,
        lifting_linear_only: bool = False,
        skip_proj_out: bool = False,
        learned_residual: bool = False,
        shared_lifting_module: 'LiftingWaveletDecompose' = None,
        untied_reconstruction: bool = False,
        multi_basis_lifting: bool = False,
        multi_basis_inits: List[str] = None,
        cross_scale_gating: bool = False,
        per_scale_mixer_widths: List[float] = None,
        num_layers: int = 1,
        stab_spectral_norm: bool = False,
        stab_ff_scaling: bool = False,
        stab_proj_out_scaling: bool = False,
        stab_mixer_eps_scaling: bool = False,
        stab_lifting_level_scaling: bool = False,
        wavelet_crawl: bool = False,
        wavelet_crawl_k: int = 3,
        use_mixer_gate: bool = True,
        mixer_gate_activation: str = "silu",
        pkm_enabled: bool = False,
        pkm_num_keys: int = 529,
        pkm_top_k: int = 32,
        pkm_heads: int = 1,
        fwpkm_enabled: bool = False,
        fwpkm_num_keys: int = 529,
        fwpkm_top_k: int = 32,
        fwpkm_heads: int = 1,
        mixer_depth: int = 1,
        mixer_depth_stabilizers: bool = False,
        mixer_depth_residuals: bool = False,
        mixer_recurrence_steps: int = 1,
        mixer_recurrence_distinct_mixer_count: int = 1,
        mixer_recurrence_residuals: bool = True,
        mixer_recurrence_cache_gate: bool = False,
        per_layer_embedding: bool = False,
        fht_input_cap_enabled: bool = False,
        fht_input_cap_value: float = 1000.0,
        fht_thue_morse_signflips: bool = False,
        fht_thue_morse_increment: int = 21,
        lifting_diaglowrank: bool = False,
        lifting_level_sharing: bool = False,
        mlp_offdiag_structure: str = "none",
        mlp_block_size: int = 64,
        mlp_band_width: int = 64,
        mlp_pq_density: float = 0.1,
        mlp_pq_mode: str = "structural",
        lifting_offdiag_structure: str = "none",
        lifting_block_size: int = 64,
        lifting_band_width: int = 64,
        lifting_monarch_blocks: int = 32,
        lifting_offdiag_density: float = 0.0,
        lifting_offdiag_mask_seed: int = 1337,
        lifting_reference_weights: Optional[Dict[Tuple[int, str, int], torch.Tensor]] = None,
        wavelet_decomp_norm: bool = False,
        wavelet_recon_norm: bool = False,
    ):
        super().__init__()
        self.C = C
        self.levels = levels
        self.Cp = next_pow2(C)
        self.decompose_bypass = decompose_bypass
        self.wavelet_mode = wavelet_mode
        self.fht_input_cap_enabled = fht_input_cap_enabled
        self.fht_input_cap_value = fht_input_cap_value
        self.fht_thue_morse_signflips = fht_thue_morse_signflips
        self.fht = FastHadamardTransform(self.Cp, device=device, dtype=dtype)

        # Permuted Thue-Morse ±1 pattern for breaking spectral bias on FWHT
        # input (and symmetrically post-invFWHT). counter += odd_increment per
        # element, sign = +1 if popcount(counter) is even, -1 if odd. Per
        # O'Connor's "Fast Transforms for Neural Networks" — random/sub-random
        # sign flips at the boundary between structured signals (wavelet
        # coefficients) and the FWHT prevent residual stream's natural axes
        # from aligning with specific Walsh basis modes. Buffer is tiny (Cp
        # fp32 = 8 KiB at Cp=2048), no learnable parameters.
        if fht_thue_morse_signflips:
            counter = 0
            signs = []
            mask = 0xFFFFFFFF  # 32-bit wraparound
            for _ in range(self.Cp):
                signs.append(1.0 if (bin(counter).count("1") & 1) == 0 else -1.0)
                counter = (counter + fht_thue_morse_increment) & mask
            sign_tensor = torch.tensor(
                signs,
                dtype=dtype if dtype is not None else torch.float32,
                device=device,
            )
            self.register_buffer("fht_signs", sign_tensor, persistent=False)

        # Wavelet decomposition
        if wavelet_mode == "lifting":
            if multi_basis_lifting and shared_lifting_module is not None:
                raise ValueError("multi_basis_lifting is incompatible with shared_lifting_weights")

            if multi_basis_lifting:
                inits = multi_basis_inits if multi_basis_inits else ["haar", "random"]
                self.lifting_wavelet = MultiBasisLiftingWavelet(
                    levels=levels, C=self.Cp, hidden_mult=lifting_hidden_mult,
                    inits=inits, dropout=lifting_dropout,
                    linear_only=lifting_linear_only, device=device, dtype=dtype,
                    stab_lifting_level_scaling=stab_lifting_level_scaling,
                    wavelet_crawl=wavelet_crawl,
                    wavelet_crawl_k=wavelet_crawl_k,
                )
            elif shared_lifting_module is not None:
                self.lifting_wavelet = shared_lifting_module
            else:
                self.lifting_wavelet = LiftingWaveletDecompose(
                    levels=levels,
                    C=self.Cp,
                    hidden_mult=lifting_hidden_mult,
                    init_wavelet=lifting_init,
                    dropout=lifting_dropout,
                    linear_only=lifting_linear_only,
                    device=device,
                    dtype=dtype,
                    stab_lifting_level_scaling=stab_lifting_level_scaling,
                    wavelet_crawl=wavelet_crawl,
                    wavelet_crawl_k=wavelet_crawl_k,
                    lifting_diaglowrank=lifting_diaglowrank,
                    lifting_level_sharing=lifting_level_sharing,
                    lifting_offdiag_structure=lifting_offdiag_structure,
                    lifting_block_size=lifting_block_size,
                    lifting_band_width=lifting_band_width,
                    lifting_monarch_blocks=lifting_monarch_blocks,
                    lifting_offdiag_density=lifting_offdiag_density,
                    lifting_offdiag_mask_seed=lifting_offdiag_mask_seed,
                    lifting_reference_weights=lifting_reference_weights,
                )

            # Detect 2D wavelet "subband" mode via duck typing (avoids hard
            # import dep on tools/two_d_wavelets.py). In subband mode the
            # wavelet emits more details per joint level (3 sub-bands instead
            # of 1), so the per-scale mixer and decompose-bypass machinery
            # need to size to the larger S_effective. Other modes (off,
            # passthrough, internal) keep the standard 1D S = levels + 1.
            self.is_subband_mode = (
                getattr(self.lifting_wavelet, 'mode', None) == 'subband'
            )
            if self.is_subband_mode:
                self.s_effective = self.lifting_wavelet.subband_num_scales
            else:
                self.s_effective = self.levels + 1

            # Reconstruct path. In subband mode the 2D wavelet handles its own
            # reconstruct (via LiftingWavelet2D.reconstruct_subband) because it
            # needs to invert both the T-axis and B-axis lifting cascades with
            # the proper sub-band layout. The standard LiftingWaveletReconstruct
            # only handles 1D inversion. For all other modes (off, passthrough,
            # internal), the standard reconstruct path applies.
            if self.is_subband_mode:
                self.lifting_reconstruct = None  # handled by self.lifting_wavelet directly
            elif untied_reconstruction:
                # Asymmetric: reconstruction has its own predict/update networks,
                # breaking the strict-invertibility constraint of classical wavelets.
                if multi_basis_lifting:
                    inits = multi_basis_inits if multi_basis_inits else ["haar", "random"]
                    self.lifting_reconstruct_wavelet = MultiBasisLiftingWavelet(
                        levels=levels, C=self.Cp, hidden_mult=lifting_hidden_mult,
                        inits=inits, dropout=lifting_dropout,
                        linear_only=lifting_linear_only, device=device, dtype=dtype,
                    )
                    self.lifting_reconstruct = MultiBasisLiftingReconstruct(self.lifting_reconstruct_wavelet)
                else:
                    self.lifting_reconstruct_wavelet = LiftingWaveletDecompose(
                        levels=levels,
                        C=self.Cp,
                        hidden_mult=lifting_hidden_mult,
                        init_wavelet=lifting_init,
                        dropout=lifting_dropout,
                        linear_only=lifting_linear_only,
                        device=device,
                        dtype=dtype,
                        stab_lifting_level_scaling=stab_lifting_level_scaling,
                        lifting_diaglowrank=lifting_diaglowrank,
                        lifting_level_sharing=lifting_level_sharing,
                        lifting_offdiag_structure=lifting_offdiag_structure,
                        lifting_block_size=lifting_block_size,
                        lifting_band_width=lifting_band_width,
                        lifting_monarch_blocks=lifting_monarch_blocks,
                        lifting_offdiag_density=lifting_offdiag_density,
                        lifting_offdiag_mask_seed=lifting_offdiag_mask_seed,
                        lifting_reference_weights=lifting_reference_weights,
                    )
                    self.lifting_reconstruct = LiftingWaveletReconstruct(self.lifting_reconstruct_wavelet)
            else:
                if multi_basis_lifting:
                    self.lifting_reconstruct = MultiBasisLiftingReconstruct(self.lifting_wavelet)
                else:
                    self.lifting_reconstruct = LiftingWaveletReconstruct(self.lifting_wavelet)

        # Decompose bypass projections
        self.decompose_bypass_ema = decompose_bypass_ema and decompose_bypass
        if self.decompose_bypass:
            # Size to s_effective so subband mode's expanded scale count
            # (3*b_levels + (L - b_levels) + 1) gets its own per-scale gains.
            self.history_gains = nn.Parameter(
                torch.zeros(self.s_effective, self.C, device=device, dtype=dtype)
            )
            self.cross_layer_mix = nn.Linear(
                self.C, self.C, bias=False, device=device, dtype=dtype
            )
            with torch.no_grad():
                w = self.cross_layer_mix.weight
                w.zero_()
                eye_sz = min(self.C, self.C)
                eye = 0.5 * torch.eye(eye_sz, device=w.device, dtype=w.dtype)
                w[:eye_sz, :eye_sz] += eye

            if self.decompose_bypass_ema:
                # Data-dependent EMA gate: α_t = σ(W·x_t + b). Init so initial
                # behavior approximates "no forgetting" — bias=0 gives α=0.5
                # per-channel (50% retention), letting the gate learn from there.
                self.ema_gate = nn.Linear(
                    self.C, self.C, bias=True, device=device, dtype=dtype
                )
                with torch.no_grad():
                    self.ema_gate.weight.zero_()   # start with α = σ(b) = σ(0) = 0.5
                    self.ema_gate.bias.zero_()

        # Spectral mixers: one per scale, optionally stacked with mixer_depth.
        # In subband mode, S = 3*b_levels + (L - b_levels) + 1 (more sub-bands
        # per joint level); otherwise S = L + 1 (the standard 1D scale count).
        S = self.s_effective
        self.mixer_depth = mixer_depth

        # Mixer recurrence — apply the mixer stage multiple times between FWHT
        # and iFWHT, exploiting that Decompose/Reconstruct are inverses (so
        # identity cancels except for the mixer body). Two parameters control:
        #   N = mixer_recurrence_steps                — outer loop count
        #   K = mixer_recurrence_distinct_mixer_count — distinct per-scale
        #                                               mixer banks per cycle
        # Semantics: a single "cycle" applies bank 0, bank 1, ..., bank K-1
        # in sequence. The cycle is repeated N times. Total mixer applications
        # per block = N * K. K=1: one shared bank reused N times (no extra
        # params). K>1: K independent banks (K-1 extra mixer-only banks
        # allocated); the K-sequence repeats N times.
        #
        # mixer_recurrence_residuals: when True (default), each mixer
        # application inside the recurrence loop adds a residual connection
        # (Y_step = X_step + mixer(X_step)) to prevent representation collapse
        # over many recurrent steps (cf. ALBERT, Universal Transformers).
        # Only applied when N*K > 1 — preserves baseline behavior when no
        # recurrence is active (default N=K=1).
        self.mixer_recurrence_steps = int(mixer_recurrence_steps)
        self.mixer_recurrence_distinct_mixer_count = int(mixer_recurrence_distinct_mixer_count)
        self.mixer_recurrence_residuals = bool(mixer_recurrence_residuals)
        # Approximation: compute the (cross-scale) gate once on the first
        # recurrence cycle and reuse it for cycles 2..N, eliminating the
        # W_gate matmul + routing einsum on all but the first cycle. Halves
        # per-step matmul cost at K=1. Only meaningful when N*K > 1.
        self.mixer_recurrence_cache_gate = bool(mixer_recurrence_cache_gate)
        if self.mixer_recurrence_steps < 1:
            raise ValueError(
                f"mixer_recurrence_steps must be >= 1, got {self.mixer_recurrence_steps}"
            )
        if self.mixer_recurrence_distinct_mixer_count < 1:
            raise ValueError(
                "mixer_recurrence_distinct_mixer_count must be >= 1, "
                f"got {self.mixer_recurrence_distinct_mixer_count}"
            )
        # N and K are independent under the new nested-loop semantics.
        # Constraints below apply when recurrence is actually active.
        recurrence_active = (
            self.mixer_recurrence_steps > 1
            or self.mixer_recurrence_distinct_mixer_count > 1
        )
        if recurrence_active:
            if self.mixer_recurrence_distinct_mixer_count > 1 and mixer_depth != 1:
                raise ValueError(
                    "mixer_recurrence_distinct_mixer_count > 1 requires "
                    f"mixer_depth == 1 (got mixer_depth={mixer_depth}); K-distinct "
                    "banks are only allocated for the depth-1 mixer path."
                )
            if untied_reconstruction:
                raise ValueError(
                    "Mixer recurrence is mutually exclusive with "
                    "untied_reconstruction: recurrence relies on "
                    "Reconstruct ∘ Decompose = I, which untied breaks."
                )

        # Auto-expand per_scale_mixer_widths when user provided a 1D-length
        # list (L+1 entries) but subband mode requires the larger expanded
        # list. The expansion triples the widths at joint-level positions to
        # cover (LH, HL, HH) sub-bands with the same per-channel width as
        # the original 1D detail entry. See
        # tools/two_d_wavelets.py:expand_per_scale_widths_for_subband.
        if (
            self.is_subband_mode
            and per_scale_mixer_widths is not None
            and len(per_scale_mixer_widths) == levels + 1
        ):
            per_scale_mixer_widths = self.lifting_wavelet.expand_per_scale_widths_for_subband(
                list(per_scale_mixer_widths)
            )

        # Cross-scale gating (routing mode): learned (S, S) routing matrix that
        # mixes scales' inputs before each per-scale gate. Init to identity so
        # behavior matches today's per-scale gating at start.
        self.cross_scale_gating = cross_scale_gating
        if cross_scale_gating:
            self.scale_routing = nn.Parameter(
                torch.eye(S, device=device, dtype=dtype)
            )

        if mixer_depth == 1:
            if per_scale_mixer_widths is not None:
                if len(per_scale_mixer_widths) != S:
                    raise ValueError(
                        f"per_scale_mixer_widths must have length S={S}, "
                        f"got {len(per_scale_mixer_widths)}"
                    )
                widths = [max(1, int(self.Cp * w)) for w in per_scale_mixer_widths]
                self.scale_mixers = nn.ModuleList([
                    PerScaleMixer(
                        Cp=self.Cp, width=widths[s], num_blocks=1, rank=low_rank,
                        use_mixer_gate=use_mixer_gate,
                        mixer_gate_activation=mixer_gate_activation,
                        add_bias=False,
                        device=device, dtype=dtype,
                        stab_spectral_norm=stab_spectral_norm,
                        stab_mixer_eps_scaling=stab_mixer_eps_scaling,
                    )
                    for s in range(S)
                ])
            else:
                # Exactly today's code — single set of mixers, no LN, no bias
                self.scale_mixers = nn.ModuleList([
                    GatedSpectralMixer(
                        Cp=self.Cp, num_blocks=1, rank=low_rank,
                        use_mixer_gate=use_mixer_gate,
                        mixer_gate_activation=mixer_gate_activation,
                        add_bias=False,
                        device=device, dtype=dtype,
                        stab_spectral_norm=stab_spectral_norm,
                        stab_mixer_eps_scaling=stab_mixer_eps_scaling,
                    )
                    for _ in range(S)
                ])

            # Mixer recurrence with K > 1: allocate (K - 1) additional mixer
            # banks. Bank 0 is self.scale_mixers; banks 1..K-1 live in
            # self.scale_mixers_recurrent_extra[0..K-2]. The K-sequence (bank
            # 0, 1, ..., K-1) repeats N times in forward(). K=1 allocates
            # nothing — the same single bank is reused N times.
            if self.mixer_recurrence_distinct_mixer_count > 1:
                extra_banks = []
                for _ in range(self.mixer_recurrence_distinct_mixer_count - 1):
                    if per_scale_mixer_widths is not None:
                        widths_extra = [max(1, int(self.Cp * w)) for w in per_scale_mixer_widths]
                        bank = nn.ModuleList([
                            PerScaleMixer(
                                Cp=self.Cp, width=widths_extra[s], num_blocks=1, rank=low_rank,
                                use_mixer_gate=use_mixer_gate,
                                mixer_gate_activation=mixer_gate_activation,
                                add_bias=False,
                                device=device, dtype=dtype,
                                stab_spectral_norm=stab_spectral_norm,
                                stab_mixer_eps_scaling=stab_mixer_eps_scaling,
                            )
                            for s in range(S)
                        ])
                    else:
                        bank = nn.ModuleList([
                            GatedSpectralMixer(
                                Cp=self.Cp, num_blocks=1, rank=low_rank,
                                use_mixer_gate=use_mixer_gate,
                                mixer_gate_activation=mixer_gate_activation,
                                add_bias=False,
                                device=device, dtype=dtype,
                                stab_spectral_norm=stab_spectral_norm,
                                stab_mixer_eps_scaling=stab_mixer_eps_scaling,
                            )
                            for _ in range(S)
                        ])
                    extra_banks.append(bank)
                self.scale_mixers_recurrent_extra = nn.ModuleList(extra_banks)
        else:
            # Depth > 1: intermediate steps get LN + bias, final step gets neither
            self.mixer_depth_stabilizers = mixer_depth_stabilizers
            self.mixer_depth_residuals = mixer_depth_residuals
            self.mixer_depth_norms = nn.ModuleList([
                nn.ModuleList([nn.LayerNorm(self.Cp, device=device, dtype=dtype) for _ in range(S)])
                for _ in range(mixer_depth - 1)
            ])
            self.scale_mixers_by_depth = nn.ModuleList([
                nn.ModuleList([
                    GatedSpectralMixer(
                        Cp=self.Cp, num_blocks=1, rank=low_rank,
                        eps=1e-3 / (d + 1) if mixer_depth_stabilizers else 1e-3,
                        use_mixer_gate=use_mixer_gate,
                        mixer_gate_activation=mixer_gate_activation,
                        add_bias=(d < mixer_depth - 1),
                        stab_spectral_norm=stab_spectral_norm,
                        stab_mixer_eps_scaling=stab_mixer_eps_scaling,
                        device=device, dtype=dtype,
                    )
                    for _ in range(S)
                ])
                for d in range(mixer_depth)
            ])
            if mixer_depth_stabilizers:
                # Per-depth learnable scalars for stability (intermediate steps only)
                # beta_d: scales LN output before feeding to mixer (input magnitude)
                # alpha_d: scales mixer output (output magnitude)
                # Both initialized to 1/D so combined initial effect is 1/D² per step
                D = mixer_depth - 1  # number of intermediate steps
                self.mixer_depth_betas = nn.ParameterList([
                    nn.Parameter(torch.tensor(1.0 / mixer_depth, device=device, dtype=dtype))
                    for _ in range(D)
                ])
                self.mixer_depth_alphas = nn.ParameterList([
                    nn.Parameter(torch.tensor(1.0 / mixer_depth, device=device, dtype=dtype))
                    for _ in range(D)
                ])

        self.scale_weights = nn.Parameter(
            torch.full((S,), 1.0, device=device, dtype=dtype)
        )

        self.skip_proj_out = skip_proj_out and (self.Cp == self.C)
        if not self.skip_proj_out:
            self.proj_out = nn.Linear(self.Cp, self.C)
            with torch.no_grad():
                if stab_proj_out_scaling:
                    # GPT-2-style residual scaling: 1/sqrt(C * num_layers) keeps
                    # the residual stream from growing through L layers.
                    self.proj_out.weight.mul_(1.0 / math.sqrt(self.C * num_layers))
                else:
                    self.proj_out.weight.mul_(1e-3)
                self.proj_out.bias.zero_()

        self.learned_residual = learned_residual
        if learned_residual:
            self.residual_alpha_spectral = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
            self.residual_alpha_mlp = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))

        self.per_layer_embedding = per_layer_embedding
        if per_layer_embedding:
            self.embedding_residual_gamma = nn.Parameter(torch.zeros(C, device=device, dtype=dtype))

        self.ln1 = nn.LayerNorm(self.C)
        self.ln2 = nn.LayerNorm(self.C)

        self.wavelet_decomp_norm = wavelet_decomp_norm
        self.wavelet_recon_norm = wavelet_recon_norm
        S = self.s_effective
        if wavelet_decomp_norm:
            self.decomp_norms = nn.ModuleList([
                nn.LayerNorm(self.Cp, device=device, dtype=dtype) for _ in range(S)
            ])
        if wavelet_recon_norm:
            self.recon_norms = nn.ModuleList([
                nn.LayerNorm(self.Cp, device=device, dtype=dtype) for _ in range(S)
            ])

        # Per-step normalization *between* mixer applications during any
        # recurrence (N*K > 1). The recurrence runs inside the FWHT/iFWHT span,
        # where coefficients are ~sqrt(Cp) amplified; chaining residual-mixer
        # steps unnormalized lets magnitude grow until fp16 overflows. Two
        # failure modes observed, both fixed by this norm:
        #   - K > 1 (distinct banks): N=2 K=2 NaN'd by step 750.
        #   - high N (shared bank): N=5 K=1 overflowed at step 1 (loss=nan at
        #     lr=0, forward-pass overflow) and then diverged (loss climbing).
        # N=2 K=1 survived without it only because 2 steps don't accumulate
        # enough magnitude — not a guarantee. Scoped to N*K > 1 so it covers
        # both depth and bank-diversity; auto-enabled, no config flag.
        # Same per-scale LayerNorm(Cp) as the wavelet norms above; applied
        # between each pair of applications (N*K − 1 invocations), with the
        # final step left unnormalized so wavelet_recon_norm (after iFWHT)
        # remains the single boundary norm on the way out.
        total_mixer_apps = (self.mixer_recurrence_steps
                            * self.mixer_recurrence_distinct_mixer_count)
        if total_mixer_apps > 1:
            self.mixer_step_norms = nn.ModuleList([
                nn.LayerNorm(self.Cp, device=device, dtype=dtype) for _ in range(S)
            ])
        else:
            self.mixer_step_norms = None

        self.use_mlp = mlp_expansion > 0
        if self.use_mlp:
            self.ffwd = FeedForward(self.C, expansion=mlp_expansion,
                                    dropout_mlp=dropout_mlp, hidden_layers=mlp_layers,
                                    stab_ff_scaling=stab_ff_scaling,
                                    mlp_offdiag_structure=mlp_offdiag_structure,
                                    mlp_block_size=mlp_block_size,
                                    mlp_band_width=mlp_band_width,
                                    mlp_pq_density=mlp_pq_density,
                                    mlp_pq_mode=mlp_pq_mode)

        self.pkm_enabled = pkm_enabled
        if pkm_enabled:
            self.pkm = ProductKeyMemory(
                self.C, num_keys=pkm_num_keys, pkm_top_k=pkm_top_k,
                pkm_heads=pkm_heads, device=device, dtype=dtype)

        self.fwpkm_enabled = fwpkm_enabled
        if fwpkm_enabled:
            self.fwpkm = FastWeightPKM(
                self.C, num_keys=fwpkm_num_keys, top_k=fwpkm_top_k,
                heads=fwpkm_heads, device=device, dtype=dtype)

        # Learned scalars for memory module combination
        if self.use_mlp:
            self.alpha_mlp = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        if pkm_enabled:
            self.alpha_pkm = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        if fwpkm_enabled:
            self.alpha_fwpkm = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))

        self.dropout_proj = nn.Dropout(dropout_projection)
        self.dropout_mix = nn.Dropout(dropout_mixer)

        # FwPKM per-layer representation caching
        self._cache_h2 = False
        self._cached_h2 = None

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None, token_embeddings: torch.Tensor = None):
        _, _, C = x.shape
        current_running_mean = None
        gate_bias_scales = None

        if self.decompose_bypass:
            if self.decompose_bypass_ema:
                current_running_mean = _compute_data_dependent_ema(
                    x, self.ema_gate.weight, self.ema_gate.bias,
                )
            else:
                current_running_mean = _compute_running_mean(x)

            if prev_state is not None:
                mixed_context = current_running_mean + self.cross_layer_mix(prev_state)
            else:
                mixed_context = current_running_mean

            gate_bias_scales = []
            for s in range(self.s_effective):
                gb = mixed_context * self.history_gains[s].view(1, 1, self.C)
                gb = pad_features_to_pow2(gb, self.Cp)
                gate_bias_scales.append(gb)
            gate_bias_scales = torch.stack(gate_bias_scales, dim=2)  # [B,T,S,Cp]

        if self.per_layer_embedding and token_embeddings is not None:
            x = x + self.embedding_residual_gamma * token_embeddings

        h = self.ln1(x)
        h = pad_features_to_pow2(h, self.Cp)

        # Wavelet decomposition
        if self.wavelet_mode == "lifting":
            approx, details = self.lifting_wavelet(h)
        else:
            approx, details = causal_haar_decompose(h, self.levels)

        # Stack coefficients top-down: [approx, detail_coarsest, ..., detail_finest]
        coeffs_top_down = [approx] + details[::-1]
        stacked_coeffs = torch.stack(coeffs_top_down, dim=2)  # [B, T, S, Cp]
        S = self.s_effective

        if self.wavelet_decomp_norm:
            stacked_coeffs = torch.stack(
                [self.decomp_norms[s](stacked_coeffs[:, :, s, :]) for s in range(S)], dim=2
            )

        # Add decompose bypass bias
        if self.decompose_bypass and gate_bias_scales is not None:
            stacked_coeffs = stacked_coeffs + gate_bias_scales

        # Optional permuted Thue-Morse ±1 sign flips before forward FWHT.
        # Breaks alignment between residual stream axes and Walsh basis modes,
        # preventing energy concentration in specific Walsh components.
        if self.fht_thue_morse_signflips:
            stacked_coeffs = stacked_coeffs * self.fht_signs

        # Optional hard cap on FWHT input: clamp(x, -cap, cap). Bounds FWHT
        # output magnitude to √Cp · cap. At cap=1000, Cp=2048: bound ≈ 45000,
        # safely under fp16's 65504 ceiling. Trade-off: gradient is zero for
        # any element with |x| > cap (dead zone), so weights producing
        # consistently saturated activations get no learning signal.
        if self.fht_input_cap_enabled:
            stacked_coeffs = stacked_coeffs.clamp(-self.fht_input_cap_value, self.fht_input_cap_value)

        # FHT forward
        stacked_spec = self.fht(stacked_coeffs)

        # Per-scale spectral mixing (with optional depth stacking and optional
        # recurrence — total N*K applications of the mixer between FWHT and
        # iFWHT, see mixer_recurrence_steps / mixer_recurrence_distinct_mixer_count).
        if self.mixer_depth == 1:
            current_spec = stacked_spec
            N = self.mixer_recurrence_steps
            K = self.mixer_recurrence_distinct_mixer_count
            # Apply residual at every recurrent step (Y = X + mixer(X)) when
            # recurrence is active. Without it, chaining identical or repeated
            # layers can cause representation collapse over many steps.
            # Preserves baseline (N=K=1) behavior when no recurrence is active.
            apply_residual = self.mixer_recurrence_residuals and (N * K > 1)
            # Gate caching: compute each bank's gate once on the first cycle
            # (n_idx == 0) and reuse it for cycles 2..N, skipping the W_gate
            # matmul + routing einsum thereafter. Approximation — the gate no
            # longer tracks the evolving state past cycle 1. Only active when
            # there is more than one cycle to amortize over (N > 1).
            cache_gate = (self.mixer_recurrence_cache_gate
                          and self.use_mixer_gate and N > 1 and (N * K > 1))
            gate_cache = [[None] * S for _ in range(K)] if cache_gate else None
            # Outer loop: repeat the K-bank cycle N times.
            # Inner loop: apply each of the K distinct banks once per cycle.
            for n_idx in range(N):
                for bank_idx in range(K):
                    if bank_idx == 0:
                        step_mixers = self.scale_mixers
                    else:
                        step_mixers = self.scale_mixers_recurrent_extra[bank_idx - 1]
                    # Reuse cached gates on cycles after the first; otherwise
                    # recompute gate routing from the current spectral state.
                    reuse_cache = cache_gate and n_idx > 0
                    if self.cross_scale_gating and not reuse_cache:
                        routed_gate_input = torch.einsum(
                            'rs,btsd->btrd', self.scale_routing, current_spec)
                    else:
                        routed_gate_input = None
                    mixed_by_scale = []
                    for s in range(S):
                        Xs = current_spec[:, :, s, :]
                        if reuse_cache:
                            Ys = step_mixers[s](Xs, cached_gate=gate_cache[bank_idx][s])
                        elif cache_gate:
                            Gs = routed_gate_input[:, :, s, :] if routed_gate_input is not None else None
                            Ys, g = step_mixers[s](Xs, gate_input=Gs, return_gate=True)
                            gate_cache[bank_idx][s] = g
                        else:
                            Gs = routed_gate_input[:, :, s, :] if routed_gate_input is not None else None
                            Ys = step_mixers[s](Xs, gate_input=Gs)
                        mixed_by_scale.append(Xs + Ys if apply_residual else Ys)
                    current_spec = torch.stack(mixed_by_scale, dim=2)
                    # Per-step LayerNorm — K > 1 only, applied *between*
                    # mixer steps. Skipped on the final step so the canonical
                    # wavelet_recon_norm (after iFWHT) remains the boundary
                    # norm on the way out.
                    is_final_step = (n_idx == N - 1) and (bank_idx == K - 1)
                    if self.mixer_step_norms is not None and not is_final_step:
                        current_spec = torch.stack(
                            [self.mixer_step_norms[s](current_spec[:, :, s, :])
                             for s in range(S)], dim=2
                        )
            mixed_spec = current_spec
        else:
            # Depth > 1: the entire depth cascade is the unit that recurs.
            # K > 1 is disallowed at depth > 1 (no per-bank allocation), so
            # the only recurrence parameter that matters here is N.
            current_spec = stacked_spec
            for _ in range(self.mixer_recurrence_steps):
                mixed_spec = current_spec
                for d in range(self.mixer_depth):
                    depth_mixers = self.scale_mixers_by_depth[d]
                    mixed_by_scale = []
                    for s in range(S):
                        Xs = mixed_spec[:, :, s, :]
                        if d < self.mixer_depth - 1:
                            # Intermediate: LN + mixer(+bias)
                            Xs_normed = self.mixer_depth_norms[d][s](Xs)
                            if self.mixer_depth_stabilizers:
                                Xs_normed = self.mixer_depth_betas[d] * Xs_normed
                            Ys = depth_mixers[s](Xs_normed)
                            if self.mixer_depth_stabilizers:
                                Ys = self.mixer_depth_alphas[d] * Ys
                            mixed_by_scale.append(Xs + Ys if self.mixer_depth_residuals else Ys)
                        else:
                            # Final: raw mixer, no LN, no bias
                            Ys = depth_mixers[s](Xs)
                            mixed_by_scale.append(Xs + Ys if self.mixer_depth_residuals else Ys)
                    mixed_spec = torch.stack(mixed_by_scale, dim=2)
                # Feed the cascade output back as input for the next recurrence pass.
                current_spec = mixed_spec
            mixed_spec = current_spec

        # Apply same cap to inverse FWHT input (mixer output) so the inverse
        # FWHT output is bounded the same way the forward FWHT output is.
        if self.fht_input_cap_enabled:
            mixed_spec = mixed_spec.clamp(-self.fht_input_cap_value, self.fht_input_cap_value)

        # FHT inverse (self-inverse for orthogonal Hadamard)
        mixed_all = self.fht(mixed_spec)

        if self.wavelet_recon_norm:
            mixed_all = torch.stack(
                [self.recon_norms[s](mixed_all[:, :, s, :]) for s in range(S)], dim=2
            )

        # Apply same Thue-Morse sign flips after invFWHT. Since D² = I, this
        # restores the residual stream to canonical position-space coordinates
        # — the FWHT path's spectral basis is rotated, but the contribution
        # back to the residual stream is in the original axis system.
        if self.fht_thue_morse_signflips:
            mixed_all = mixed_all * self.fht_signs

        # Unstack and apply scale weights
        mixed_list = list(mixed_all.unbind(dim=2))
        processed_top_down = []
        for idx, mixed in enumerate(mixed_list):
            mixed = self.dropout_mix(mixed)
            scaled = mixed * self.scale_weights[idx]
            processed_top_down.append(scaled)

        # Wavelet reconstruct
        approx_proc = processed_top_down[0]
        details_proc = processed_top_down[1:][::-1]
        if self.wavelet_mode == "lifting":
            if self.is_subband_mode:
                # Subband mode: the 2D wavelet owns reconstruction since it
                # needs to invert both the T-axis and B-axis lifting cascades
                # with the proper (LH, HL, HH) sub-band layout per joint level.
                reconstructed_padded = self.lifting_wavelet.reconstruct_subband(
                    approx_proc, details_proc
                )
            else:
                reconstructed_padded = self.lifting_reconstruct(approx_proc, details_proc)
        else:
            reconstructed_padded = causal_haar_reconstruct(approx_proc, details_proc)

        # Projection and residual
        if self.skip_proj_out:
            projected = reconstructed_padded
        else:
            projected = self.proj_out(reconstructed_padded)

        if self.learned_residual:
            x = self.residual_alpha_spectral * x + self.dropout_proj(projected)
        else:
            x = x + self.dropout_proj(projected)

        # Memory modules (parallel, learned scalar combination)
        h2 = self.ln2(x)
        if self._cache_h2:
            self._cached_h2 = h2.detach()
        mem_out = torch.zeros_like(h2)
        if self.use_mlp:
            mem_out = mem_out + self.alpha_mlp * self.ffwd(h2)
        if self.pkm_enabled:
            mem_out = mem_out + self.alpha_pkm * self.pkm(h2)
        if self.fwpkm_enabled:
            mem_out = mem_out + self.alpha_fwpkm * self.fwpkm(h2)

        if self.learned_residual:
            x = self.residual_alpha_mlp * x + mem_out
        else:
            x = x + mem_out

        return x, current_running_mean


# ==============================================================================
# 8. WaveletLM LANGUAGE MODEL
# ==============================================================================

class WaveletLM(nn.Module):
    def __init__(self, vocab_size, config, device=None):
        super().__init__()

        self.C = config['C']
        C = self.C
        self.vocab_size = vocab_size

        # Embedding
        if config.get('sparse_encoder_decoder_embedding', False):
            from tools.encoder_decoder_embedding import EncoderDecoderEmbedding
            C_emb = int(config.get('sparse_encoder_decoder_embedding_C', C))
            self.token_embedding = EncoderDecoderEmbedding(vocab_size, C_emb, C)
            full_dense = vocab_size * C
            eff = self.token_embedding.effective_param_count()
            reduction = (1 - eff / full_dense) * 100 if C_emb < C else 0.0
            full_note = "(full-rank: C_emb == C)" if C_emb >= C else f"({reduction:.1f}% reduction vs dense V·C)"
            print(
                f"[Embedding] EncoderDecoderEmbedding: V={vocab_size}, C_emb={C_emb}, C={C} "
                f"-- {eff:,} of dense {full_dense:,} embedding params {full_note}"
            )
        elif config.get('sparse_pq_embedding_enabled', False):
            from tools.sparse_pq_embedding import SparsePQEmbedding
            density = config.get('sparse_pq_embedding_density', 0.1)
            mode = config.get('sparse_pq_embedding_mode', 'structural')
            epsilon = config.get('sparse_pq_embedding_epsilon', 0.0) or 0.0
            self.token_embedding = SparsePQEmbedding(
                vocab_size, C, density=density, mode=mode,
                init_std=1.0 / math.sqrt(C), epsilon=epsilon,
            )
            print(
                f"[Embedding] SparsePQEmbedding: p={self.token_embedding.p}, "
                f"q={self.token_embedding.q}, N'={self.token_embedding.N_prime}, "
                f"phantom_rows={self.token_embedding.phantom_rows}, "
                f"density={self.token_embedding.density_actual:.4f}, "
                f"epsilon={self.token_embedding.epsilon:.2e} "
                f"({self.token_embedding.effective_param_count():,} of "
                f"{self.token_embedding.weight.numel():,} params)"
            )
        else:
            self.token_embedding = nn.Embedding(vocab_size, C)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1.0 / math.sqrt(C))

        # Dropout
        self.dropout_emb = nn.Dropout(config.get('dropout_embedding', 0.0))
        self.dropout_lm = nn.Dropout(config.get('dropout_lm_head', 0.0))

        # Gradient checkpointing
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)

        # Wavelet config
        wavelet_mode = config.get("wavelet_mode", "lifting")
        lifting_hidden_mult = config.get("lifting_hidden_mult", 1)
        lifting_init = config.get("lifting_init", "haar")
        lifting_dropout = config.get("lifting_dropout", 0.0)
        lifting_linear_only = config.get("lifting_linear_only", True)
        skip_proj_out = config.get("skip_proj_out", True)
        learned_residual = config.get("learned_residual", True)

        # Off-diagonal structural / top-k prior config (Option B unified flag).
        lifting_offdiag_structure = config.get("lifting_offdiag_structure", "none")
        lifting_block_size = config.get("lifting_block_size", 64)
        lifting_band_width = config.get("lifting_band_width", 64)
        lifting_monarch_blocks = config.get("lifting_monarch_blocks", 32)
        lifting_offdiag_density = config.get("lifting_offdiag_density", 0.0)
        lifting_offdiag_mask_seed = config.get("lifting_offdiag_mask_seed", 1337)
        lifting_offdiag_mask_checkpoint = config.get("lifting_offdiag_mask_checkpoint", "")

        # For magnitude_topk, load the reference checkpoint and extract the trained
        # lifting weights as a dict keyed by (level, role, lin_idx). Other modes
        # don't need a reference, so this dict stays None and is ignored downstream.
        lifting_reference_weights = None
        if lifting_offdiag_structure == "magnitude_topk":
            if not lifting_offdiag_mask_checkpoint:
                raise ValueError(
                    "lifting_offdiag_structure='magnitude_topk' requires "
                    "lifting_offdiag_mask_checkpoint (path to a trained checkpoint "
                    "to rank off-diagonal positions by magnitude)."
                )
            lifting_reference_weights = _load_lifting_reference_weights(
                lifting_offdiag_mask_checkpoint
            )
            print(
                f"[Lifting] magnitude_topk: loaded {len(lifting_reference_weights)} "
                f"reference weights from {lifting_offdiag_mask_checkpoint}"
            )
        if lifting_offdiag_structure not in (None, "none"):
            print(
                f"[Lifting] lifting_offdiag_structure={lifting_offdiag_structure!r}, "
                f"density={lifting_offdiag_density}, "
                f"block_size={lifting_block_size}, band_width={lifting_band_width}, "
                f"monarch_blocks={lifting_monarch_blocks}, seed={lifting_offdiag_mask_seed}"
            )

        # Shared lifting wavelet module
        shared_lifting = None
        self.shared_lifting_weights = config.get("shared_lifting_weights", True)
        if wavelet_mode == "lifting" and self.shared_lifting_weights:
            Cp = next_pow2(C)
            shared_lifting = LiftingWaveletDecompose(
                levels=config['levels'],
                C=Cp,
                hidden_mult=lifting_hidden_mult,
                init_wavelet=lifting_init,
                dropout=lifting_dropout,
                linear_only=lifting_linear_only,
                device=device,
                stab_lifting_level_scaling=(
                    config.get("stable_parametrization", False)
                    or config.get("stab_lifting_level_scaling", False)
                ),
                wavelet_crawl=config.get("wavelet_crawl", False),
                wavelet_crawl_k=config.get("wavelet_crawl_k", 3),
                lifting_diaglowrank=config.get("lifting_diaglowrank", False),
                lifting_level_sharing=config.get("lifting_level_sharing", False),
                lifting_offdiag_structure=lifting_offdiag_structure,
                lifting_block_size=lifting_block_size,
                lifting_band_width=lifting_band_width,
                lifting_monarch_blocks=lifting_monarch_blocks,
                lifting_offdiag_density=lifting_offdiag_density,
                lifting_offdiag_mask_seed=lifting_offdiag_mask_seed,
                lifting_reference_weights=lifting_reference_weights,
            )
            # 2D wavelet over (batch, token): wrap the 1D T-axis lifting in
            # the 2D module when `wavelet_2d_mode != "off"`. Mode semantics
            # documented in tools/two_d_wavelets.py:
            #   - "passthrough": Phase 1 scaffold (no-op delegation to 1D)
            #   - "internal":    Phase 2.A (B-axis lift + per-sub-band scale +
            #                    B-axis inverse → same output shape as 1D)
            #   - "subband":     Phase 2.B (4 sub-bands exposed; NOT YET IMPL)
            wavelet_2d_mode = config.get("wavelet_2d_mode", "off")
            if wavelet_2d_mode != "off":
                from tools.two_d_wavelets import build_lifting_wavelet_2d
                shared_lifting = build_lifting_wavelet_2d(
                    t_wavelet=shared_lifting,
                    config=config,
                )
                print(
                    f"[Lifting] 2D wavelet enabled (mode={wavelet_2d_mode!r}). "
                    f"b_levels={shared_lifting.b_levels}, "
                    f"init_zero={config.get('wavelet_2d_init_zero', True)}, "
                    f"state_passing={config.get('wavelet_2d_state_passing', False)}"
                )
            lifting_params = sum(p.numel() for p in shared_lifting.parameters())
            print(f"[Lifting] Shared across all layers: {lifting_params/1e6:.2f}M params (dense)")

        if config.get("untied_reconstruction", False):
            print(f"[Lifting] Untied reconstruction: per-layer separate decompose/reconstruct weights")

        if config.get("cross_scale_gating", False):
            print(f"[Mixer] Cross-scale gating (routing): learned (S, S) routing matrix per layer, init to identity")

        if config.get("multi_basis_lifting", False):
            inits = config.get("multi_basis_inits", None) or ["haar", "random"]
            print(f"[Lifting] Multi-basis lifting: K={len(inits)} parallel wavelets per layer, inits={inits}")

        if config.get("per_scale_mixer_widths", None) is not None:
            widths = config["per_scale_mixer_widths"]
            print(f"[Mixer] Per-scale widths: multipliers={widths}")

        # Resolve stable parametrization flags (master OR individual)
        sp_master = config.get("stable_parametrization", False)
        stab_spectral_norm = sp_master or config.get("stab_spectral_norm", False)
        stab_ff_scaling = sp_master or config.get("stab_ff_scaling", False)
        stab_embed_scaling = sp_master or config.get("stab_embed_scaling", False)
        stab_proj_out_scaling = sp_master or config.get("stab_proj_out_scaling", False)
        stab_mixer_eps_scaling = sp_master or config.get("stab_mixer_eps_scaling", False)
        stab_lifting_level_scaling = sp_master or config.get("stab_lifting_level_scaling", False)
        self.stab_embed_scaling = stab_embed_scaling
        self._embed_scale = math.sqrt(C) if stab_embed_scaling else 1.0
        any_stab = any([
            stab_spectral_norm, stab_ff_scaling, stab_embed_scaling,
            stab_proj_out_scaling, stab_mixer_eps_scaling, stab_lifting_level_scaling,
        ])
        if config.get("wavelet_crawl", False):
            K = config.get("wavelet_crawl_k", 3)
            print(f"[Lifting] Wavelet crawl: learned dilation mixing K={K} per level")

        if any_stab:
            active = [
                name for name, on in [
                    ("spectral_norm", stab_spectral_norm),
                    ("ff_scaling", stab_ff_scaling),
                    ("embed_scaling", stab_embed_scaling),
                    ("proj_out_scaling", stab_proj_out_scaling),
                    ("mixer_eps_scaling", stab_mixer_eps_scaling),
                    ("lifting_level_scaling", stab_lifting_level_scaling),
                ] if on
            ]
            print(f"[StableParam] Active: {active}")

        Cp = next_pow2(C)
        if skip_proj_out and Cp == C:
            saved = config['layers'] * (Cp * C + C)
            print(f"[proj_out] Skipped (C={C} == Cp={Cp}): saves {saved/1e6:.2f}M params")

        if learned_residual:
            print(f"[Residual] Learned alpha (init=1.0, per-block spectral+MLP)")

        self.per_layer_embedding = config.get("per_layer_embedding", False)
        self.loop_iterations = config.get("loop_iterations", 1)

        # BBCE (Bisected Block Context Extension): preprocessor that bisects
        # the long-context input into mean-pooled compressed slots + raw
        # uncompressed tokens, returning a (B, block_size, C) embedding for
        # the standard wavelet pipeline. See tools/bbce.py.
        self.bbce_enabled = config.get("bbce_enabled", False)
        if self.bbce_enabled:
            from tools.bbce import build_bbce_preprocessor
            self._bbce_preprocessor = build_bbce_preprocessor(config)
            self.block_size = int(config["block_size"])
            print(
                f"[BBCE] Enabled. block_size={self.block_size}, "
                f"block_size_compressed={config['block_size_compressed']}, "
                f"num_slots={self._bbce_preprocessor.num_slots}, "
                f"g={self._bbce_preprocessor.g}, "
                f"dropped={self._bbce_preprocessor.num_dropped} boundary tokens"
            )

        # Feedback mechanisms
        self.looped_blocks = config.get("looped_blocks", False)
        self.looped_blocks_count = config.get("looped_blocks_count", 8)

        # Stochastic depth
        self.stochastic_depth_rate = config.get("stochastic_depth_rate", 0.0)
        L = config['layers']
        if self.stochastic_depth_rate > 0:
            self._drop_probs = [l / L * self.stochastic_depth_rate for l in range(L)]
            print(f"[StochasticDepth] rate={self.stochastic_depth_rate}, "
                  f"drop_probs=[{self._drop_probs[0]:.3f}..{self._drop_probs[-1]:.3f}]")
        else:
            self._drop_probs = [0.0] * L

        # Build layers (or single shared block for looped_blocks)
        effective_layer_count = (
            self.looped_blocks_count if self.looped_blocks else L
        )
        self.effective_layer_count = effective_layer_count
        if self.looped_blocks:
            print(f"[Looped] Single shared WaveletLMBlock applied {effective_layer_count} times")
        layer_build_count = 1 if self.looped_blocks else L
        self.layers = nn.ModuleList([
            WaveletLMBlock(
                C,
                levels=config['levels'],
                low_rank=config.get('low_rank', 0),
                mlp_expansion=config.get('mlp_expansion', 10),
                mlp_layers=config.get('mlp_layers', 2),
                dropout_projection=config.get('dropout_projection', 0.0),
                dropout_mixer=config.get('dropout_mixer', 0.0),
                dropout_mlp=config.get('dropout_mlp', 0.0),
                device=device,
                decompose_bypass=config.get("decompose_bypass", True),
                decompose_bypass_ema=config.get("decompose_bypass_ema", False),
                wavelet_mode=wavelet_mode,
                lifting_hidden_mult=lifting_hidden_mult,
                lifting_init=lifting_init,
                lifting_dropout=lifting_dropout,
                lifting_linear_only=lifting_linear_only,
                skip_proj_out=skip_proj_out,
                learned_residual=learned_residual,
                shared_lifting_module=shared_lifting,
                untied_reconstruction=config.get("untied_reconstruction", False),
                multi_basis_lifting=config.get("multi_basis_lifting", False),
                multi_basis_inits=config.get("multi_basis_inits", None),
                cross_scale_gating=config.get("cross_scale_gating", False),
                per_scale_mixer_widths=config.get("per_scale_mixer_widths", None),
                num_layers=L,
                stab_spectral_norm=stab_spectral_norm,
                stab_ff_scaling=stab_ff_scaling,
                stab_proj_out_scaling=stab_proj_out_scaling,
                stab_mixer_eps_scaling=stab_mixer_eps_scaling,
                stab_lifting_level_scaling=stab_lifting_level_scaling,
                wavelet_crawl=config.get("wavelet_crawl", False),
                wavelet_crawl_k=config.get("wavelet_crawl_k", 3),
                use_mixer_gate=config.get("use_mixer_gate", True),
                mixer_gate_activation=config.get("mixer_gate_activation", "silu"),
                pkm_enabled=config.get("pkm_enabled", False),
                pkm_num_keys=config.get("pkm_num_keys", 529),
                pkm_top_k=config.get("pkm_top_k", 32),
                pkm_heads=config.get("pkm_heads", 1),
                fwpkm_enabled=config.get("fwpkm_enabled", False),
                fwpkm_num_keys=config.get("fwpkm_num_keys", 529),
                fwpkm_top_k=config.get("fwpkm_top_k", 32),
                fwpkm_heads=config.get("fwpkm_heads", 1),
                mixer_depth=config.get("mixer_depth", 1),
                mixer_depth_stabilizers=config.get("mixer_depth_stabilizers", False),
                mixer_depth_residuals=config.get("mixer_depth_residuals", False),
                mixer_recurrence_steps=config.get("mixer_recurrence_steps", 1),
                mixer_recurrence_distinct_mixer_count=config.get(
                    "mixer_recurrence_distinct_mixer_count", 1
                ),
                mixer_recurrence_residuals=config.get(
                    "mixer_recurrence_residuals", True
                ),
                mixer_recurrence_cache_gate=config.get(
                    "mixer_recurrence_cache_gate", False
                ),
                per_layer_embedding=config.get("per_layer_embedding", False),
                fht_input_cap_enabled=config.get("fht_input_cap_enabled", False),
                fht_input_cap_value=config.get("fht_input_cap_value", 1000.0),
                fht_thue_morse_signflips=config.get("fht_thue_morse_signflips", False),
                fht_thue_morse_increment=config.get("fht_thue_morse_increment", 21),
                lifting_diaglowrank=config.get("lifting_diaglowrank", False),
                lifting_level_sharing=config.get("lifting_level_sharing", False),
                mlp_offdiag_structure=config.get("mlp_offdiag_structure", "none"),
                mlp_block_size=config.get("mlp_block_size", 64),
                mlp_band_width=config.get("mlp_band_width", 64),
                mlp_pq_density=config.get("mlp_pq_density", 0.1),
                mlp_pq_mode=config.get("mlp_pq_mode", "structural"),
                lifting_offdiag_structure=lifting_offdiag_structure,
                lifting_block_size=lifting_block_size,
                lifting_band_width=lifting_band_width,
                lifting_monarch_blocks=lifting_monarch_blocks,
                lifting_offdiag_density=lifting_offdiag_density,
                lifting_offdiag_mask_seed=lifting_offdiag_mask_seed,
                lifting_reference_weights=lifting_reference_weights,
                wavelet_decomp_norm=config.get("wavelet_decomp_norm", False),
                wavelet_recon_norm=config.get("wavelet_recon_norm", False),
            )
            for _ in range(layer_build_count)
        ])

        # Final LN and LM head
        self.final_ln = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size, bias=False)

        # Weight tying
        from tools.sparse_pq_embedding import SparsePQEmbedding, MaskedTiedLinear
        from tools.encoder_decoder_embedding import (
            EncoderDecoderEmbedding, EncoderDecoderTiedLMHead, EncoderDecoderUntiedLMHead,
        )
        if config.get("tie_embedding_to_lm_head", False):
            if isinstance(self.token_embedding, EncoderDecoderEmbedding):
                # Tied: hidden -> encoder -> input_embedding.weight^T -> logits.
                self.lm_head = EncoderDecoderTiedLMHead(self.token_embedding)
                print(f"[LM Head] Tied to EncoderDecoderEmbedding via EncoderDecoderTiedLMHead (V projection shares input embedding)")
            elif isinstance(self.token_embedding, SparsePQEmbedding):
                # Replace lm_head with a MaskedTiedLinear that shares the
                # embedding's parameter AND applies its mask in forward.
                # Required because torch.compile cannot rely on parameter
                # gradient hooks to keep non-mask positions at zero, so we
                # enforce the mask at every forward use site instead.
                self.lm_head = MaskedTiedLinear(self.token_embedding, bias=False)
                print(f"[LM Head] Tied to SparsePQEmbedding via MaskedTiedLinear (mask applied in forward)")
            else:
                self.lm_head.weight = self.token_embedding.weight
                print(f"[LM Head] Tied to token embedding")
        else:
            if isinstance(self.token_embedding, EncoderDecoderEmbedding):
                # Untied: hidden -> encoder (shared with ed_embedding) ->
                # output_embedding.weight^T -> logits. The encoder bridges
                # C -> C_emb on the output path; output_embedding is a
                # separate (untied) V × C_emb matrix.
                self.lm_head = EncoderDecoderUntiedLMHead(self.token_embedding)
                print(f"[LM Head] Untied EncoderDecoderUntiedLMHead (encoder shared, separate V × C_emb output_embedding)")
            # else: standard nn.Linear(C, V) lm_head from earlier in __init__ remains

        # Cross-window decompose bypass
        self.decompose_bypass_cross_window = config.get("decompose_bypass_cross_window", True)
        self._persistent_semantic_state = None
        self._persistent_token_count = 0

    def reset_semantic_state(self):
        """Reset cross-window semantic state (call at start of new document/sequence)."""
        self._persistent_semantic_state = None
        self._persistent_token_count = 0

    def reset_fast_weights(self):
        """Reset FwPKM deltas across all layers."""
        for layer in self.layers:
            if hasattr(layer, 'fwpkm') and layer.fwpkm_enabled:
                layer.fwpkm.reset_fast_weights()

    def update_fast_weights(self, chunk_ids: torch.Tensor, lr: float = 0.01):
        """Update FwPKM deltas using per-layer representations from a forward pass.

        Runs a no-grad forward pass on chunk_ids to populate each layer's
        cached h2, then updates each layer's FwPKM using its own representations.

        Args:
            chunk_ids: [B, chunk_size] - token IDs for the chunk
            lr: learning rate for fast-weight updates
        """
        # Enable h2 caching
        for layer in self.layers:
            if layer.fwpkm_enabled:
                layer._cache_h2 = True

        # Forward pass to populate per-layer cached h2
        with torch.no_grad():
            self.forward(chunk_ids)

        # Update each layer's FwPKM with its own representations
        for layer in self.layers:
            if layer.fwpkm_enabled:
                h2 = layer._cached_h2
                queries = h2[:, :-1, :]  # [B, T-1, C]
                targets = h2[:, 1:, :]   # [B, T-1, C]
                layer.fwpkm.update_fast_weights(queries, targets, lr=lr)
                layer._cache_h2 = False
                layer._cached_h2 = None

    def _forward_embed(self, idx):
        """Embedding lookup + dropout. Used by MultiNodeWaveletLM lockstep forward."""
        emb = self.token_embedding(idx)
        if self.stab_embed_scaling:
            emb = emb * self._embed_scale
        return self.dropout_emb(emb)

    def _forward_head(self, x, targets=None):
        """Final LN, LM head, and loss. Used by MultiNodeWaveletLM lockstep forward."""
        x = self.final_ln(x)
        x = self.dropout_lm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1).long())
        return logits, loss

    def forward(self, idx, targets=None):
        B, T_in = idx.shape

        # BBCE: chunked-embed + bisect + mean-pool → (B, block_size, C).
        # Memory-bounded by tools/bbce.py:BBCEPreprocessor's chunking. After
        # this branch, T is always self.block_size (the "supervised window").
        if self.bbce_enabled:
            tok_emb = self._bbce_preprocessor(idx, self.token_embedding)
            T = self.block_size
        else:
            tok_emb = self.token_embedding(idx)  # [B, T_in, C]
            T = T_in

        if self.stab_embed_scaling:
            tok_emb = tok_emb * self._embed_scale
        x = self.dropout_emb(tok_emb)
        ple = tok_emb if self.per_layer_embedding else None

        # Initialize from persistent state if cross-window bypass is enabled
        if self.decompose_bypass_cross_window and self._persistent_semantic_state is not None:
            current_state = self._persistent_semantic_state.unsqueeze(1).expand(-1, T, -1)
        else:
            current_state = None

        all_logits = [] if self.loop_iterations > 1 and targets is not None else None

        # Effective number of block applications per pass (looped vs stacked)
        eff_count = self.effective_layer_count
        get_layer = (lambda i: self.layers[0]) if self.looped_blocks else (lambda i: self.layers[i])

        for loop in range(self.loop_iterations):
            for layer_idx in range(eff_count):
                layer = get_layer(layer_idx)

                # Stochastic depth (only meaningful in stacked mode)
                if (not self.looped_blocks and self.training
                        and self.stochastic_depth_rate > 0):
                    if random.random() < self._drop_probs[layer_idx]:
                        continue

                if self.gradient_checkpointing and self.training:
                    def layer_wrapper(lx, _layer=layer, _state=current_state, _ple=ple):
                        return _layer(lx, _state, token_embeddings=_ple)
                    x, current_state = checkpoint(layer_wrapper, x, use_reentrant=False)
                else:
                    x, current_state = layer(x, current_state, token_embeddings=ple)

            # Produce logits at each loop iteration for multi-iteration loss
            if all_logits is not None:
                x_ln = self.final_ln(x)
                logits_t = self.lm_head(self.dropout_lm(x_ln))
                all_logits.append(logits_t)

        # Update persistent state for next window
        if self.decompose_bypass_cross_window and current_state is not None:
            self._persistent_semantic_state = current_state[:, -1, :].detach()
            self._persistent_token_count += T

        # Final logits (always from last iteration)
        x = self.final_ln(x)
        x = self.dropout_lm(x)
        logits = self.lm_head(x)

        # BBCE: only the last block_size/2 positions are loss-bearing (the
        # uncompressed half). Slice both logits and all_logits (if any)
        # before cross-entropy. The data loader already returns targets of
        # shape (B, block_size/2), so no slicing on targets.
        if self.bbce_enabled:
            num_uncompressed = self.block_size // 2
            logits_for_loss = logits[:, -num_uncompressed:, :]
            all_logits_for_loss = (
                [lg[:, -num_uncompressed:, :] for lg in all_logits]
                if all_logits is not None
                else None
            )
        else:
            logits_for_loss = logits
            all_logits_for_loss = all_logits

        loss = None
        if targets is not None:
            if self.loop_iterations == 1:
                loss = F.cross_entropy(
                    logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                    targets.view(-1).long(),
                )
            else:
                # Average loss across all iterations (uniform weighting)
                total_loss = sum(
                    F.cross_entropy(lg.reshape(-1, lg.size(-1)), targets.view(-1).long())
                    for lg in all_logits_for_loss
                )
                loss = total_loss / self.loop_iterations

        return logits, loss


# ==============================================================================
# 9. MULTINODAL — Cross-Cell Gate & Multi-Node WaveletLM
# ==============================================================================

class CrossCellGate(nn.Module):
    """Multiplicative cross-cell gate for multinodal WaveletLM.

    Each cell's hidden state is multiplied by (1 + delta), where
    delta = tanh(proj(mean_of_other_cells)). Zero-initialized so the gate
    starts as identity.
    """

    def __init__(self, num_cells, C, device=None, dtype=None):
        super().__init__()
        self.num_cells = num_cells
        self.proj = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.proj.weight)

    def forward(self, hiddens):
        sum_all = torch.stack(hiddens).sum(dim=0)
        result = []
        for h in hiddens:
            mean_others = (sum_all - h) / (self.num_cells - 1)
            delta = torch.tanh(self.proj(mean_others))
            result.append(h * (1.0 + delta))
        return result


class MultiNodeWaveletLM(nn.Module):
    """Multinodal WaveletLM: multiple independent cells with feature-bagged embeddings.

    Each cell is a complete WaveletLM instance operating on a different subset of
    the embedding dimensions. Logits are averaged (product of experts) for prediction.

    With learned embeddings, feature bagging simply zeros (sets to eps) random
    dimensions of the learned embedding per cell.
    """

    def __init__(self, vocab_size, config, device=None):
        super().__init__()
        self.num_cells = config.get('multinodal_num_cells', 2)
        self.cell_dim = config.get('multinodal_cell_dim', 512)
        self.combination = config.get('multinodal_combination', 'average')
        seeds = config.get('multinodal_seeds', [42, 137])

        if len(seeds) < self.num_cells:
            raise ValueError(
                f"multinodal_seeds has {len(seeds)} entries but "
                f"multinodal_num_cells={self.num_cells}")

        # Feature bagging config
        bagged_eps = config.get('multinodal_bagged_eps', 1e-6)
        features_per_cell = config.get('multinodal_features_per_cell', -1)
        if features_per_cell < 0:
            features_per_cell = self.cell_dim

        # Build cell configs
        cell_config = dict(config)
        cell_config['C'] = self.cell_dim

        # Create cells
        self.cells = nn.ModuleList()
        for i in range(self.num_cells):
            set_seed(seeds[i])
            cell = WaveletLM(
                vocab_size=vocab_size,
                config=cell_config,
                device=device,
            )

            # Feature bagging: zero out random embedding dims
            if features_per_cell < self.cell_dim:
                n_to_zero = self.cell_dim - features_per_cell
                rng = random.Random(seeds[i])
                all_dims = list(range(self.cell_dim))
                rng.shuffle(all_dims)
                dims_to_zero = set(all_dims[:n_to_zero])
                with torch.no_grad():
                    for d in dims_to_zero:
                        cell.token_embedding.weight[:, d] = bagged_eps
                print(f"[Multinodal] Cell {i} (seed={seeds[i]}): "
                      f"{features_per_cell}/{self.cell_dim} active dims, "
                      f"{n_to_zero} bagged (eps={bagged_eps})")

            self.cells.append(cell)

        # Gradient scaling: 1/sqrt(N) normalizes gradient variance across cells,
        # preventing fp16 overflow during warmup that caused persistent NaN instability.
        self.grad_scale = 1.0 / math.sqrt(self.num_cells)

        # Mirror attributes for training loop compatibility
        self.lm_head = self.cells[0].lm_head

        # Cross-cell gating
        self.cross_cell_gating = config.get('multinodal_cross_cell_gating', False)
        self.cross_cell_gate_interval = config.get('multinodal_cross_cell_gate_interval', 1)
        if self.cross_cell_gating:
            num_layers = config.get('layers', 20)
            num_gates = (num_layers + self.cross_cell_gate_interval - 1) // self.cross_cell_gate_interval
            _p = next(self.cells[0].parameters())
            self.cross_cell_gates = nn.ModuleList([
                CrossCellGate(self.num_cells, self.cell_dim, device=_p.device, dtype=_p.dtype)
                for _ in range(num_gates)
            ])
            print(f"[Multinodal] Cross-cell gating: {num_gates} gates, "
                  f"interval={self.cross_cell_gate_interval}")

    def reset_semantic_state(self):
        for cell in self.cells:
            cell.reset_semantic_state()

    def forward(self, idx, targets=None):
        if self.cross_cell_gating:
            # Lockstep forward: run all cells block-by-block
            xs = [cell._forward_embed(idx) for cell in self.cells]
            current_states = [None] * self.num_cells
            num_layers = len(self.cells[0].layers)
            gate_idx = 0
            for layer_idx in range(num_layers):
                for ci, cell in enumerate(self.cells):
                    if cell.training and cell.stochastic_depth_rate > 0:
                        if random.random() < cell._drop_probs[layer_idx]:
                            continue
                    layer = cell.layers[layer_idx]
                    state = current_states[ci]
                    if cell.gradient_checkpointing and cell.training:
                        def _wrapper(lx, _layer=layer, _state=state):
                            return _layer(lx, _state)
                        xs[ci], current_states[ci] = checkpoint(
                            _wrapper, xs[ci], use_reentrant=False)
                    else:
                        xs[ci], current_states[ci] = layer(xs[ci], state)
                if (layer_idx + 1) % self.cross_cell_gate_interval == 0:
                    if gate_idx < len(self.cross_cell_gates):
                        xs = self.cross_cell_gates[gate_idx](xs)
                        gate_idx += 1

            all_logits_losses = [cell._forward_head(x, targets)
                                 for cell, x in zip(self.cells, xs)]
            all_logits = [l for l, _ in all_logits_losses]
        else:
            # Sequential forward (no cross-cell gating)
            all_logits = []
            for cell in self.cells:
                logits, _ = cell(idx, targets)
                all_logits.append(logits)

        stacked = torch.stack(all_logits)
        combined = stacked.mean(dim=0) if self.combination == 'average' else stacked.sum(dim=0)

        combined_loss = None
        if targets is not None:
            combined_loss = F.cross_entropy(
                combined.view(-1, combined.size(-1)), targets.view(-1).long())
            combined_loss = combined_loss * self.grad_scale

        return combined, combined_loss


# ==============================================================================
# 10. UTILITIES — Logging & Parameter Breakdown
# ==============================================================================

class Logger:
    """Simple file + console logger."""

    def __init__(self, log_dir, filename="log.txt", append=False):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        mode = 'a' if append else 'w'
        self.file = open(self.log_path, mode, encoding='utf-8')

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        self.file.write(line + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


# ==============================================================================
# TOKENIZER WRAPPERS
# ==============================================================================
# Auto-selection by dataset: WikiText uses GPT-2 BPE (tiktoken) to match prior
# practice; PG-19 uses a 32K SentencePiece BPE trained on PG-19's train split,
# matching the tokenizer family used by Compressive Transformer / Transformer-XL
# in their PG-19 numbers. Override via config["tokenizer"] = "<name>".

DATASET_TOKENIZER = {
    "wikitext-103": "gpt2",
    "wikitext-2": "gpt2",
    "pg19": "sentencepiece-pg19-32k",
}
DEFAULT_TOKENIZER = "gpt2"
SP_PG19_MODEL_PATH = os.path.join(".cache", "pg19_sp32k.model")


class GPT2Tokenizer:
    name = "gpt2"

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._enc.n_vocab

    def encode(self, text, allow_special=False):
        if allow_special:
            return self._enc.encode(text)
        return self._enc.encode(text, allowed_special=set())

    def decode(self, ids):
        return self._enc.decode(ids)

    def display_name(self):
        return f"GPT-2 BPE (tiktoken), {self.vocab_size:,} vocab"


class SentencePieceTokenizer:
    name = "sentencepiece-pg19-32k"
    model_path = SP_PG19_MODEL_PATH

    def __init__(self):
        import sentencepiece as spm
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"SentencePiece model not found at {self.model_path}. "
                "Train it via train.py on the PG-19 train split (the first PG-19 "
                "run will train it automatically) or place a pre-trained .model file there."
            )
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(self.model_path)
        self.vocab_size = self._sp.get_piece_size()

    def encode(self, text, allow_special=False):
        return self._sp.encode(text, out_type=int)

    def decode(self, ids):
        return self._sp.decode(ids)

    def display_name(self):
        return f"SentencePiece BPE (PG-19, 32K), {self.vocab_size:,} vocab"


def resolve_tokenizer_name(config):
    """Resolve the tokenizer name from config, applying dataset auto-selection."""
    name = config.get("tokenizer", "auto")
    if name == "auto":
        dataset_name = config.get("dataset", "wikitext-103")
        name = DATASET_TOKENIZER.get(dataset_name, DEFAULT_TOKENIZER)
    return name


def get_tokenizer(config):
    """Construct the tokenizer wrapper for the active config."""
    name = resolve_tokenizer_name(config)
    if name == "gpt2":
        return GPT2Tokenizer()
    if name == "sentencepiece-pg19-32k":
        return SentencePieceTokenizer()
    raise ValueError(
        f"Unknown tokenizer: {name!r}. Known: 'gpt2', 'sentencepiece-pg19-32k', 'auto'."
    )


def parameter_breakdown(model, config, logger=None):
    """Print parameter breakdown for WaveletLM or MultiNodeWaveletLM.

    If a Logger is provided, routes output through logger.log so the breakdown
    also lands in log.txt (in addition to stdout). If logger is None, falls
    back to print() — preserves backward compatibility.
    """
    out = (lambda s: logger.log(s)) if logger is not None else print

    # Use effective_param_count so StructuredLinear masks are counted as their
    # nonzero entries (not full numel) and MonarchLinear factors are counted
    # Counts use sum(p.numel()) -- the dense / true storage count. Mask-based
    # structural priors (T-lower / BAND / BD / Monarch / PQ) zero out specific
    # weight positions but do NOT shrink the underlying tensor: the .pt file,
    # Adagrad accumulator, gradient buffer, and forward/backward FLOPs all
    # operate on the full dense shape. Reporting the dense count keeps the
    # breakdown honest about what the model actually weighs and costs to ship.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    W = 22  # alignment width for values

    out(f"\n{'='*60}")
    out(f"PARAMETER BREAKDOWN")
    out(f"{'='*60}")
    out(f"Total parameters:  {total:>{W},} ({total/1e6:.2f}M)")
    if trainable != total:
        out(f"Trainable parameters:{trainable:>{W},} ({trainable/1e6:.2f}M)")

    if isinstance(model, MultiNodeWaveletLM):
        for i, cell in enumerate(model.cells):
            cell_params = effective_param_count(cell)
            out(f"  Cell {i}:           {cell_params:>{W},} ({cell_params/1e6:.2f}M)")
        if model.cross_cell_gating:
            gate_params = sum(p.numel() for p in model.cross_cell_gates.parameters())
            out(f"  Cross-cell gates: {gate_params:>{W},} ({gate_params/1e6:.2f}M)")
    elif isinstance(model, WaveletLM):
        emb_params = model.token_embedding.weight.numel()
        lm_params = sum(p.numel() for p in model.lm_head.parameters())
        layer_params = sum(p.numel() for p in model.layers.parameters())
        ln_params = sum(p.numel() for p in model.final_ln.parameters())

        # When `shared_lifting_weights=True`, each layer holds a reference to
        # the same `LiftingWaveletDecompose` module, so traversing
        # `model.layers` visits the shared lifting once and rolls it into
        # `layer_params`. We also print it separately as `Wavelets`, so the
        # breakdown lines would visually double-count if we didn't subtract
        # it from `layer_params` here. Lifting / wavelet count is dense
        # storage (sum of numel) — masked entries (e.g. T-lower's 50%
        # zeros) are stored as real bytes and counted accordingly.
        lift_params = 0
        if model.shared_lifting_weights and hasattr(model.layers[0], 'lifting_wavelet'):
            lift_params = sum(p.numel() for p in model.layers[0].lifting_wavelet.parameters())
            out(f"  Wavelets:        {lift_params:>{W},} ({lift_params/1e6:.2f}M)")
            layer_params = max(0, layer_params - lift_params)

        out(f"  Token embedding: {emb_params:>{W},} ({emb_params/1e6:.2f}M)")
        out(f"  Layers (total):  {layer_params:>{W},} ({layer_params/1e6:.2f}M)")

        # Per-layer component breakdown
        block0 = model.layers[0]
        if block0.mixer_depth > 1:
            if hasattr(block0, 'scale_mixers_by_depth'):
                mixer_per = sum(p.numel() for p in block0.scale_mixers_by_depth.parameters())
                norm_per = sum(p.numel() for p in block0.mixer_depth_norms.parameters())
                out(f"    Mixer (depth={block0.mixer_depth}):{mixer_per + norm_per:>{W-2},} ({(mixer_per + norm_per)/1e6:.2f}M)")
        else:
            mixer_per = sum(p.numel() for p in block0.scale_mixers.parameters())
            out(f"    Mixer/layer:   {mixer_per:>{W},} ({mixer_per/1e6:.2f}M)")
        if block0.use_mlp:
            mlp_per = sum(p.numel() for p in block0.ffwd.parameters())
            out(f"    MLP/layer:     {mlp_per:>{W},} ({mlp_per/1e6:.2f}M)")
        if block0.pkm_enabled:
            pkm_per = sum(p.numel() for p in block0.pkm.parameters())
            out(f"    PKM/layer:     {pkm_per:>{W},} ({pkm_per/1e6:.2f}M)")
        if block0.fwpkm_enabled:
            fwpkm_per = sum(p.numel() for p in block0.fwpkm.parameters())
            out(f"    FwPKM/layer:   {fwpkm_per:>{W},} ({fwpkm_per/1e6:.2f}M)")

        # When the LM head is weight-tied to the token embedding, its tensor
        # is the same Parameter object as the embedding — counted once in
        # `total` (PyTorch's parameters() deduplicates shared modules) and
        # therefore zero ADDITIONAL params at the LM head line.
        lm_head_tied = (
            getattr(model.lm_head, 'weight', None) is model.token_embedding.weight
        )
        if lm_head_tied:
            out(f"  LM head (shared):{0:>{W},} additional")
        else:
            out(f"  LM head:         {lm_params:>{W},} ({lm_params/1e6:.2f}M)")
        out(f"  Final LayerNorm: {ln_params:>{W},} ({ln_params/1e6:.2f}M)")

    out(f"{'='*60}\n")
    return total, trainable


# ==============================================================================
# 11. POST-TRAINING QUANTIZATION (PTQ)
# ==============================================================================

def quantize_tensor(weight, bits, per_channel=True):
    """Symmetric quantization of a weight tensor to N-bit integer representation.

    Args:
        weight: float tensor (2D)
        bits: target bit-width (2, 4, 8, or 16). 16 = no-op.
        per_channel: if True, compute scale per output channel (dim=0)

    Returns:
        (quantized_int8, scales) where scales are float32
    """
    if bits >= 16:
        return weight, None

    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1

    if per_channel and weight.dim() >= 2:
        # Per output channel (dim=0)
        amax = weight.abs().amax(dim=list(range(1, weight.dim())), keepdim=True)
    else:
        amax = weight.abs().amax()

    amax = amax.clamp(min=1e-8)
    scale = amax / qmax

    quantized = (weight / scale).round().clamp(qmin, qmax).to(torch.int8)
    return quantized, scale.to(torch.float32)


def dequantize_tensor(quantized_int, scale, bits, dtype=torch.float16):
    """Dequantize: weight_approx = scale * quantized_int."""
    if bits >= 16:
        return quantized_int
    return (scale * quantized_int.float()).to(dtype)


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with quantized weights."""

    def __init__(self, original, bits):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bits = bits

        if bits >= 16:
            self.weight = original.weight
            self._quantized = False
        else:
            q_int, scale = quantize_tensor(original.weight.data, bits, per_channel=True)
            self.register_buffer('weight_int', q_int)
            self.register_buffer('weight_scale', scale)
            self._quantized = True

        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        if self._quantized:
            w = dequantize_tensor(self.weight_int, self.weight_scale, self.bits, dtype=x.dtype)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)


class QuantizedEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding with quantized weights."""

    def __init__(self, original, bits):
        super().__init__()
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.bits = bits
        self.padding_idx = original.padding_idx

        if bits >= 16:
            self.weight = original.weight
            self._quantized = False
        else:
            q_int, scale = quantize_tensor(original.weight.data, bits, per_channel=False)
            self.register_buffer('weight_int', q_int)
            self.register_buffer('weight_scale', scale)
            self._quantized = True

    def forward(self, x):
        if self._quantized:
            w = dequantize_tensor(self.weight_int, self.weight_scale, self.bits, dtype=torch.float16)
        else:
            w = self.weight
        return F.embedding(x, w, padding_idx=self.padding_idx)


def _get_mixer_bits(scale_idx, config):
    """Map wavelet scale index to bit-width tier."""
    if scale_idx <= 2:
        return config.get('quantize_mixer_coarse_bits', 8)
    elif scale_idx <= 5:
        return config.get('quantize_mixer_mid_bits', 4)
    else:
        return config.get('quantize_mixer_fine_bits', 2)


def _quantize_sequential(seq, bits):
    """Replace nn.Linear modules inside an nn.Sequential with QuantizedLinear."""
    for i, module in enumerate(seq):
        if isinstance(module, nn.Linear):
            seq[i] = QuantizedLinear(module, bits)


def _compute_module_mib(module):
    """Compute storage in MiB for a module's parameters and buffers."""
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total / (1024 ** 2)


def quantize_model(model, config):
    """Apply post-training quantization to a WaveletLM or MultiNodeWaveletLM.

    Replaces nn.Linear and nn.Embedding modules with quantized versions
    based on config settings. Small parameters (LayerNorm, scalars, scale_weights)
    are kept at full precision.

    Args:
        model: WaveletLM or MultiNodeWaveletLM in eval mode
        config: dict with quantize_* keys

    Returns:
        dict with {original_mib, quantized_mib, compression_ratio, per_component}
    """
    # Compute original size
    original_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    original_mib = original_bytes / (1024 ** 2)

    emb_bits = config.get('quantize_embedding_bits', 8)
    mlp_bits = config.get('quantize_mlp_bits', 4)
    lift_bits = config.get('quantize_lifting_bits', 16)

    # Collect all WaveletLM cells to quantize
    if isinstance(model, MultiNodeWaveletLM):
        cells = list(model.cells)
        # Quantize cross-cell gates
        if model.cross_cell_gating:
            for gate in model.cross_cell_gates:
                gate.proj = QuantizedLinear(gate.proj, mlp_bits)
    elif isinstance(model, WaveletLM):
        cells = [model]
    else:
        raise TypeError(f"quantize_model expects WaveletLM or MultiNodeWaveletLM, got {type(model)}")

    # Track shared lifting to avoid double-quantization
    quantized_lifting_ids = set()

    for cell in cells:
        # --- Embedding ---
        tied = (hasattr(cell, 'lm_head') and hasattr(cell, 'token_embedding') and
                cell.lm_head.weight.data_ptr() == cell.token_embedding.weight.data_ptr())

        cell.token_embedding = QuantizedEmbedding(cell.token_embedding, emb_bits)

        if tied:
            # Share the quantized weight with lm_head via a thin wrapper
            cell.lm_head = QuantizedLinear(cell.lm_head, emb_bits)
        else:
            cell.lm_head = QuantizedLinear(cell.lm_head, emb_bits)

        # --- Per-layer ---
        for layer in cell.layers:
            # Mixer (per-scale); unwrap PerScaleMixer to reach the inner GatedSpectralMixer
            for s, scale_mod in enumerate(layer.scale_mixers):
                bits = _get_mixer_bits(s, config)
                inner = scale_mod.mixer if isinstance(scale_mod, PerScaleMixer) else scale_mod
                inner.mixer = QuantizedLinear(inner.mixer, bits)
                if inner.use_mixer_gate:
                    inner.gate = QuantizedLinear(inner.gate, bits)
                if isinstance(scale_mod, PerScaleMixer) and scale_mod.proj_in is not None:
                    scale_mod.proj_in = QuantizedLinear(scale_mod.proj_in, bits)
                    scale_mod.proj_out = QuantizedLinear(scale_mod.proj_out, bits)

            # Lifting wavelet (decompose; possibly shared across layers; possibly multi-basis)
            lifting_roots = [layer.lifting_wavelet]
            if hasattr(layer, 'lifting_reconstruct_wavelet'):
                lifting_roots.append(layer.lifting_reconstruct_wavelet)
            # Flatten multi-basis wrappers into their constituent wavelets
            lifting_modules = []
            for root in lifting_roots:
                if isinstance(root, MultiBasisLiftingWavelet):
                    lifting_modules.extend(root.wavelets)
                else:
                    lifting_modules.append(root)
            for lifting in lifting_modules:
                if id(lifting) in quantized_lifting_ids:
                    continue
                quantized_lifting_ids.add(id(lifting))
                for i, net in enumerate(lifting.predict_nets):
                    if isinstance(net, nn.Linear):
                        lifting.predict_nets[i] = QuantizedLinear(net, lift_bits)
                    elif isinstance(net, nn.Sequential):
                        _quantize_sequential(net, lift_bits)
                for i, net in enumerate(lifting.update_nets):
                    if isinstance(net, nn.Linear):
                        lifting.update_nets[i] = QuantizedLinear(net, lift_bits)
                    elif isinstance(net, nn.Sequential):
                        _quantize_sequential(net, lift_bits)

            # proj_out
            if hasattr(layer, 'proj_out') and isinstance(layer.proj_out, nn.Linear):
                layer.proj_out = QuantizedLinear(layer.proj_out, mlp_bits)

            # MLP
            if hasattr(layer, 'ffwd'):
                _quantize_sequential(layer.ffwd.net, mlp_bits)

            # Decompose bypass cross_layer_mix
            if hasattr(layer, 'cross_layer_mix') and isinstance(layer.cross_layer_mix, nn.Linear):
                layer.cross_layer_mix = QuantizedLinear(layer.cross_layer_mix, mlp_bits)

            # PKM query_proj
            if layer.pkm_enabled and hasattr(layer.pkm, 'query_proj'):
                layer.pkm.query_proj = QuantizedLinear(layer.pkm.query_proj, mlp_bits)

            # FwPKM query_proj
            if layer.fwpkm_enabled and hasattr(layer.fwpkm, 'query_proj'):
                layer.fwpkm.query_proj = QuantizedLinear(layer.fwpkm.query_proj, mlp_bits)

    # Compute quantized size
    quantized_bytes = 0
    for p in model.parameters():
        quantized_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        quantized_bytes += b.numel() * b.element_size()
    quantized_mib = quantized_bytes / (1024 ** 2)

    ratio = original_mib / quantized_mib if quantized_mib > 0 else float('inf')

    stats = {
        'original_mib': original_mib,
        'quantized_mib': quantized_mib,
        'compression_ratio': ratio,
    }

    # Print report
    print(f"\n{'='*60}")
    print(f"[Quantization] Original: {original_mib:.1f} MiB -> "
          f"Quantized: {quantized_mib:.1f} MiB ({ratio:.2f}x compression)")
    print(f"  Mixer bits: coarse={config.get('quantize_mixer_coarse_bits', 8)}, "
          f"mid={config.get('quantize_mixer_mid_bits', 4)}, "
          f"fine={config.get('quantize_mixer_fine_bits', 2)}")
    print(f"  MLP bits: {mlp_bits}, Lifting bits: {lift_bits}, "
          f"Embedding bits: {emb_bits}")
    if mlp_bits < 8 or any(config.get(k, 8) < 8 for k in
                           ['quantize_mixer_coarse_bits', 'quantize_mixer_mid_bits',
                            'quantize_mixer_fine_bits']):
        print(f"  Note: sub-8-bit stored as int8 (no packing). "
              f"Deployment packing would further reduce size.")
    print(f"{'='*60}\n")

    return stats
