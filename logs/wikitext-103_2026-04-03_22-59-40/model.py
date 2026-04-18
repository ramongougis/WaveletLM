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

# WaveletLM - Exclusively Attentionless Reasoning with Causal Harmonics
# model.py

import os
import random
import warnings
import math
import json
import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=r".*Online softmax is disabled.*")
warnings.filterwarnings("ignore", message=r".*Not enough SMs to use max_autotune_gemm.*")

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch.utils.checkpoint import checkpoint


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
    B, T, C = x.shape
    if C == C_pad:
        return x
    return F.pad(x, (0, C_pad - C))


# ==============================================================================
# 2. FREQUENCY SPACE — Fast Hadamard Transform
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
    ):
        super().__init__()
        self.levels = levels
        self.C = C
        self.hidden_mult = hidden_mult
        self.init_wavelet = init_wavelet
        self.linear_only = linear_only

        hidden_dim = C * hidden_mult

        self.predict_nets = nn.ModuleList()
        self.update_nets = nn.ModuleList()

        for level in range(levels):
            if linear_only:
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
                        nn.init.zeros_(predict[0].weight[C:, :])
                    nn.init.zeros_(predict[0].bias)
                    nn.init.eye_(predict[3].weight[:, :C])
                    if hidden_mult > 1:
                        nn.init.zeros_(predict[3].weight[:, C:])
                    nn.init.zeros_(predict[3].bias)

                    nn.init.eye_(update[0].weight[:C, :])
                    if hidden_mult > 1:
                        nn.init.zeros_(update[0].weight[C:, :])
                    nn.init.zeros_(update[0].bias)
                    nn.init.zeros_(update[3].weight)
                    update[3].weight.data[:, :C] = 0.5 * torch.eye(
                        C, device=device, dtype=dtype if dtype else torch.float32)
                    nn.init.zeros_(update[3].bias)

                elif init_wavelet == 'zero':
                    nn.init.zeros_(predict[3].weight)
                    nn.init.zeros_(predict[3].bias)
                    nn.init.zeros_(update[3].weight)
                    nn.init.zeros_(update[3].bias)

            self.predict_nets.append(predict)
            self.update_nets.append(update)

        self.inv_sqrt2 = 0.7071067811865476

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        details: List[torch.Tensor] = []
        cur = x

        for level in range(self.levels):
            dilation = 1 << level
            padded = F.pad(cur, (0, 0, dilation, 0))

            even = cur
            odd = padded[:, :-dilation, :]

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
                 device=None, dtype=None):
        super().__init__()
        self.Cp = Cp
        self.use_mixer_gate = use_mixer_gate
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

        self.reset_parameters(eps)

    def reset_parameters(self, eps=1e-3):
        with torch.no_grad():
            self.mixer.weight.data.copy_(torch.eye(self.Cp, device=self.mixer.weight.device))
            self.mixer.weight.data.add_(torch.randn_like(self.mixer.weight) * eps)
            if self.use_mixer_gate:
                nn.init.normal_(self.gate.weight, std=0.02)
            if self.U is not None:
                nn.init.normal_(self.U, std=0.01)
                nn.init.normal_(self.V, std=0.01)

    def forward(self, X_spec: torch.Tensor):
        signal = self.mixer(X_spec)
        if self.use_mixer_gate:
            out = signal * self.gate_activation(self.gate(X_spec))
        else:
            out = signal
        if self.U is not None:
            mid = torch.matmul(X_spec, self.V)
            out = out + torch.matmul(mid, self.U.t())
        return out


# ==============================================================================
# 5. FEED-FORWARD (MLP)
# ==============================================================================

class FeedForward(nn.Module):
    def __init__(self, C, expansion=2, dropout_mlp=0.0, hidden_layers=2):
        super().__init__()
        hidden_dim = C * expansion
        layers = []
        layers.append(nn.Linear(C, hidden_dim))
        for _ in range(hidden_layers - 2):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        if hidden_layers >= 2:
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, C))
        layers.append(nn.Dropout(dropout_mlp))
        self.net = nn.Sequential(*layers)
        final_linear = self.net[-2]
        with torch.no_grad():
            final_linear.weight.mul_(0.02)
            final_linear.bias.zero_()

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
    """Fast-Weight Product Key Memory.

    Structurally identical to ProductKeyMemory during training. At inference,
    an optional update_fast_weights() method updates value deltas per chunk,
    enabling episodic/contextual memory without retraining.

    Sits on top of the MLP+PKM gated output (chained, not parallel).
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
# 6. DECOMPOSE BYPASS — Running Mean
# ==============================================================================

@torch.compiler.disable
def _compute_running_mean(x: torch.Tensor, prev_mean: torch.Tensor = None,
                          prev_count: int = 0) -> torch.Tensor:
    """Causal running mean along the time dimension.
    Decorated with @torch.compiler.disable to avoid cumsum compilation issues."""
    T = x.size(1)
    if prev_mean is not None and prev_count > 0:
        prev_sum = prev_mean.unsqueeze(1) * prev_count
        history_sum = torch.cumsum(x, dim=1) + prev_sum
        divisors = torch.arange(prev_count + 1, prev_count + T + 1, device=x.device).view(1, -1, 1)
    else:
        history_sum = torch.cumsum(x, dim=1)
        divisors = torch.arange(1, T + 1, device=x.device).view(1, -1, 1)
    return history_sum / divisors


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
        wavelet_mode: str = "lifting",
        lifting_hidden_mult: int = 1,
        lifting_init: str = "haar",
        lifting_dropout: float = 0.0,
        lifting_linear_only: bool = False,
        skip_proj_out: bool = False,
        learned_residual: bool = False,
        shared_lifting_module: 'LiftingWaveletDecompose' = None,
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
    ):
        super().__init__()
        self.C = C
        self.levels = levels
        self.Cp = next_pow2(C)
        self.decompose_bypass = decompose_bypass
        self.wavelet_mode = wavelet_mode
        self.fht = FastHadamardTransform(self.Cp, device=device, dtype=dtype)

        # Wavelet decomposition
        if wavelet_mode == "lifting":
            if shared_lifting_module is not None:
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
                )
            self.lifting_reconstruct = LiftingWaveletReconstruct(self.lifting_wavelet)

        # Decompose bypass projections
        if self.decompose_bypass:
            self.history_gains = nn.Parameter(
                torch.zeros(self.levels + 1, self.C, device=device, dtype=dtype)
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

        # Spectral mixers: one per scale
        S = levels + 1
        self.scale_mixers = nn.ModuleList([
            GatedSpectralMixer(
                Cp=self.Cp, num_blocks=1, rank=low_rank,
                use_mixer_gate=use_mixer_gate,
                mixer_gate_activation=mixer_gate_activation,
                device=device, dtype=dtype,
            )
            for _ in range(S)
        ])

        self.scale_weights = nn.Parameter(
            torch.full((S,), 1.0, device=device, dtype=dtype)
        )

        self.skip_proj_out = skip_proj_out and (self.Cp == self.C)
        if not self.skip_proj_out:
            self.proj_out = nn.Linear(self.Cp, self.C)
            with torch.no_grad():
                self.proj_out.weight.mul_(1e-3)
                self.proj_out.bias.zero_()

        self.learned_residual = learned_residual
        if learned_residual:
            self.residual_alpha_spectral = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
            self.residual_alpha_mlp = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))

        self.ln1 = nn.LayerNorm(self.C)
        self.ln2 = nn.LayerNorm(self.C)

        self.use_mlp = mlp_expansion > 0
        if self.use_mlp:
            self.ffwd = FeedForward(self.C, expansion=mlp_expansion,
                                    dropout_mlp=dropout_mlp, hidden_layers=mlp_layers)

        self.pkm_enabled = pkm_enabled
        if pkm_enabled:
            self.pkm = ProductKeyMemory(
                self.C, num_keys=pkm_num_keys, pkm_top_k=pkm_top_k,
                pkm_heads=pkm_heads, device=device, dtype=dtype)

        # Learned gate when both MLP and PKM are active
        if self.use_mlp and pkm_enabled:
            self.memory_gate = nn.Linear(self.C, 1, bias=True, device=device, dtype=dtype)
            nn.init.zeros_(self.memory_gate.weight)
            nn.init.constant_(self.memory_gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12, starts MLP-heavy

        # FwPKM: chains on top of MLP+PKM output (Stage 2 refinement)
        self.fwpkm_enabled = fwpkm_enabled
        if fwpkm_enabled:
            self.fwpkm = FastWeightPKM(
                self.C, num_keys=fwpkm_num_keys, top_k=fwpkm_top_k,
                heads=fwpkm_heads, device=device, dtype=dtype)

        self.dropout_proj = nn.Dropout(dropout_projection)
        self.dropout_mix = nn.Dropout(dropout_mixer)

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None):
        _, _, C = x.shape
        current_running_mean = None
        gate_bias_scales = None

        if self.decompose_bypass:
            current_running_mean = _compute_running_mean(x)

            if prev_state is not None:
                mixed_context = current_running_mean + self.cross_layer_mix(prev_state)
            else:
                mixed_context = current_running_mean

            gate_bias_scales = []
            for s in range(self.levels + 1):
                gb = mixed_context * self.history_gains[s].view(1, 1, self.C)
                gb = pad_features_to_pow2(gb, self.Cp)
                gate_bias_scales.append(gb)
            gate_bias_scales = torch.stack(gate_bias_scales, dim=2)  # [B,T,S,Cp]

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
        S = self.levels + 1

        # Add decompose bypass bias
        if self.decompose_bypass and gate_bias_scales is not None:
            stacked_coeffs = stacked_coeffs + gate_bias_scales

        # FHT forward
        stacked_spec = self.fht(stacked_coeffs)

        # Per-scale spectral mixing
        mixed_by_scale = []
        for s in range(S):
            Xs = stacked_spec[:, :, s, :]
            Ys = self.scale_mixers[s](Xs)
            mixed_by_scale.append(Ys)
        mixed_spec = torch.stack(mixed_by_scale, dim=2)

        # FHT inverse (self-inverse for orthogonal Hadamard)
        mixed_all = self.fht(mixed_spec)

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

        # Stage 1: MLP / PKM (base memory)
        h2 = self.ln2(x)
        if self.use_mlp and self.pkm_enabled:
            mlp_out = self.ffwd(h2)
            pkm_out = self.pkm(h2)
            g = torch.sigmoid(self.memory_gate(h2))  # [B, T, 1]
            base = (1 - g) * mlp_out + g * pkm_out
        elif self.use_mlp:
            base = self.ffwd(h2)
        elif self.pkm_enabled:
            base = self.pkm(h2)
        else:
            base = torch.zeros_like(h2)

        # Stage 2: FwPKM refinement (chains on base memory output)
        if self.fwpkm_enabled:
            mem_out = self.fwpkm(base)
        else:
            mem_out = base

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
            )
            lifting_params = sum(p.numel() for p in shared_lifting.parameters())
            print(f"[Lifting] Shared across all layers: {lifting_params/1e6:.2f}M params")

        Cp = next_pow2(C)
        if skip_proj_out and Cp == C:
            saved = config['layers'] * (Cp * C + C)
            print(f"[proj_out] Skipped (C={C} == Cp={Cp}): saves {saved/1e6:.2f}M params")

        if learned_residual:
            print(f"[Residual] Learned alpha (init=1.0, per-block spectral+MLP)")

        # Stochastic depth
        self.stochastic_depth_rate = config.get("stochastic_depth_rate", 0.0)
        L = config['layers']
        if self.stochastic_depth_rate > 0:
            self._drop_probs = [l / L * self.stochastic_depth_rate for l in range(L)]
            print(f"[StochasticDepth] rate={self.stochastic_depth_rate}, "
                  f"drop_probs=[{self._drop_probs[0]:.3f}..{self._drop_probs[-1]:.3f}]")
        else:
            self._drop_probs = [0.0] * L

        # Build layers
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
                wavelet_mode=wavelet_mode,
                lifting_hidden_mult=lifting_hidden_mult,
                lifting_init=lifting_init,
                lifting_dropout=lifting_dropout,
                lifting_linear_only=lifting_linear_only,
                skip_proj_out=skip_proj_out,
                learned_residual=learned_residual,
                shared_lifting_module=shared_lifting,
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
            )
            for _ in range(L)
        ])

        # Final LN and LM head
        self.final_ln = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size, bias=False)

        # Weight tying
        if config.get("tie_embedding_to_lm_head", False):
            self.lm_head.weight = self.token_embedding.weight
            print(f"[LM Head] Tied to token embedding")

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

    def update_fast_weights(self, queries: torch.Tensor,
                            targets: torch.Tensor, lr: float = 0.01):
        """Update FwPKM deltas across all layers using chunk representations.

        Args:
            queries: [B, chunk_size, C] - input to each layer's FwPKM
            targets: [B, chunk_size, C] - shifted-by-1 targets
            lr: learning rate for fast-weight updates
        """
        for layer in self.layers:
            if hasattr(layer, 'fwpkm') and layer.fwpkm_enabled:
                layer.fwpkm.update_fast_weights(queries, targets, lr=lr)

    def _forward_embed(self, idx):
        """Embedding lookup + dropout. Used by MultiNodeWaveletLM lockstep forward."""
        return self.dropout_emb(self.token_embedding(idx))

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
        B, T = idx.shape

        x = self.token_embedding(idx)  # [B, T, C]
        x = self.dropout_emb(x)

        # Initialize from persistent state if cross-window bypass is enabled
        if self.decompose_bypass_cross_window and self._persistent_semantic_state is not None:
            current_state = self._persistent_semantic_state.unsqueeze(1).expand(-1, T, -1)
        else:
            current_state = None

        for layer_idx, layer in enumerate(self.layers):
            # Stochastic depth
            if self.training and self.stochastic_depth_rate > 0:
                if random.random() < self._drop_probs[layer_idx]:
                    continue

            if self.gradient_checkpointing and self.training:
                def layer_wrapper(lx, _layer=layer, _state=current_state):
                    return _layer(lx, _state)
                x, current_state = checkpoint(layer_wrapper, x, use_reentrant=False)
            else:
                x, current_state = layer(x, current_state)

        # Update persistent state for next window
        if self.decompose_bypass_cross_window and current_state is not None:
            self._persistent_semantic_state = current_state[:, -1, :].detach()
            self._persistent_token_count += T

        x = self.final_ln(x)
        x = self.dropout_lm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1).long())

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


def parameter_breakdown(model, config):
    """Print parameter breakdown for WaveletLM or MultiNodeWaveletLM."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"PARAMETER BREAKDOWN")
    print(f"{'='*60}")
    print(f"Total parameters:     {total:>15,} ({total/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable:>15,} ({trainable/1e6:.2f}M)")

    if isinstance(model, MultiNodeWaveletLM):
        for i, cell in enumerate(model.cells):
            cell_params = sum(p.numel() for p in cell.parameters())
            print(f"  Cell {i}: {cell_params:>13,} ({cell_params/1e6:.2f}M)")
        if model.cross_cell_gating:
            gate_params = sum(p.numel() for p in model.cross_cell_gates.parameters())
            print(f"  Cross-cell gates: {gate_params:>8,} ({gate_params/1e6:.2f}M)")
    elif isinstance(model, WaveletLM):
        emb_params = model.token_embedding.weight.numel()
        lm_params = sum(p.numel() for p in model.lm_head.parameters())
        layer_params = sum(p.numel() for p in model.layers.parameters())
        ln_params = sum(p.numel() for p in model.final_ln.parameters())

        if model.shared_lifting_weights and hasattr(model.layers[0], 'lifting_wavelet'):
            lift_params = sum(p.numel() for p in model.layers[0].lifting_wavelet.parameters())
            print(f"  Shared lifting:   {lift_params:>13,} ({lift_params/1e6:.2f}M)")

        print(f"  Token embedding:  {emb_params:>13,} ({emb_params/1e6:.2f}M)")
        print(f"  Layers (total):   {layer_params:>13,} ({layer_params/1e6:.2f}M)")

        # Per-layer MLP vs PKM breakdown
        block0 = model.layers[0]
        if block0.use_mlp:
            mlp_per = sum(p.numel() for p in block0.ffwd.parameters())
            print(f"    MLP/layer:      {mlp_per:>13,} ({mlp_per/1e6:.2f}M)")
        if block0.pkm_enabled:
            pkm_per = sum(p.numel() for p in block0.pkm.parameters())
            print(f"    PKM/layer:      {pkm_per:>13,} ({pkm_per/1e6:.2f}M)")
        if block0.fwpkm_enabled:
            fwpkm_per = sum(p.numel() for p in block0.fwpkm.parameters())
            print(f"    FwPKM/layer:    {fwpkm_per:>13,} ({fwpkm_per/1e6:.2f}M)")

        print(f"  LM head:          {lm_params:>13,} ({lm_params/1e6:.2f}M)")
        print(f"  Final LayerNorm:  {ln_params:>13,} ({ln_params/1e6:.2f}M)")

    print(f"{'='*60}\n")
    return total, trainable
