"""Sparse (p, q) phantom-token embedding compression.

Implements the deterministic 1D-walk sparsity scheme described in
plans/new_compression_ideas.md. The embedding's (N, C) weight matrix
is masked along the flattened tensor by alternating step sizes p and q,
yielding density 2 / (p + q). Phantom tokens (vocab padded to N' >= N
to enable q | N') are a numerology device only — never allocated.

Sparsity is enforced via:
  1. Init-time mask application: weight = init * mask (non-mask positions = 0)
  2. Forward multiply: every forward pass uses `weight * mask`, so even if
     the optimizer would update non-mask positions, those updates are
     zeroed back out at the next forward use. The chain rule through
     `weight * mask` also naturally zeros gradient at non-mask positions
     during backward, so the optimizer doesn't accumulate Adagrad/Adam
     state on non-mask positions.

NOTE: Forward multiplication (not a parameter gradient hook) is required
because torch.compile does not reliably fire `register_hook` callbacks on
parameters. The forward multiply is a regular tensor op that compile
traces correctly.

For weight-tied LM heads: a separate MaskedTiedLinear class shares the
parameter AND applies the same mask in its forward. This is necessary
because the `nn.Linear.forward` of a vanilla tied LM head would NOT
multiply by mask, so non-mask drift in the parameter would still affect
the LM head's logits (even though the embedding lookup is masked).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _divisors(n: int) -> list[int]:
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def find_pq(C: int, N: int, density: float, mode: str = "structural",
            phantom_budget: int | None = None) -> tuple[int, int, int, int]:
    """Find a valid (p, q, N', phantom_rows) tuple at target density.

    Requirements:
      * q | C and q | N'        (q is a common divisor; gives macrocell tiling)
      * p does not divide C or N'   (avoids degenerate column-stripe patterns)
      * p, q > 1
      * p + q == round(2 / density) (sets density)

    mode:
      * "smallest_q" — minimal phantom rows; near-stride behavior at small q
      * "structural" — q closest to sqrt(C); square macrocells (default)
      * "budget"     — largest q with phantom_rows <= phantom_budget
    """
    s = round(2.0 / density)
    if s <= 2:
        raise ValueError(
            f"density {density} too high — need s = 2/density > 2 (got s={s})"
        )
    valid: list[tuple[int, int, int, int]] = []
    for q in _divisors(C):
        if q <= 1 or q >= s:
            continue
        p = s - q
        if p <= 1 or C % p == 0:
            continue
        N_prime = q * ((N + q - 1) // q)
        if N_prime % p == 0:
            continue
        valid.append((p, q, N_prime, N_prime - N))
    if not valid:
        raise ValueError(
            f"No valid (p, q, N') for C={C}, N={N}, density={density}. "
            f"Try a slightly different density."
        )
    if mode == "smallest_q":
        return valid[0]
    if mode == "structural":
        target = math.sqrt(C)
        return min(valid, key=lambda c: abs(c[1] - target))
    if mode == "budget":
        budget = phantom_budget if phantom_budget is not None else N // 100
        valid_b = [c for c in valid if c[3] <= budget]
        return max(valid_b or valid, key=lambda c: c[1])
    raise ValueError(f"Unknown mode: {mode!r}")


def make_pq_mask(N: int, C: int, p: int, q: int, N_prime: int) -> torch.Tensor:
    """Build the (N, C) bool mask from the (p, q) walk over (N', C),
    filtered to row < N. Phantom-row positions are dropped (never allocated)."""
    mask = torch.zeros(N, C, dtype=torch.bool)
    total = N_prime * C
    pos = 0
    step_q = False  # alternate p (False) then q (True) starting with p after pos=0
    while pos < total:
        row = pos // C
        if row < N:
            mask[row, pos - row * C] = True
        if step_q:
            pos += q
        else:
            pos += p
        step_q = not step_q
    return mask


class SparsePQEmbedding(nn.Embedding):
    """nn.Embedding with (p, q) phantom-token sparsity enforced via init-time
    mask application + per-forward multiplication.

    Forward semantics depend on `epsilon`:
      - epsilon == 0: forward uses `weight * mask` -> non-mask positions are
        exactly 0 in the materialized weight tensor.
      - epsilon > 0: forward uses `where(mask, weight, epsilon)` -> non-mask
        positions are a small constant `epsilon` instead of exact zero. Helps
        with fp16 kernel pathologies that exact-zero rows can trigger
        (LayerNorm variance underflow, sparse-matmul edge cases). Note that
        `epsilon` must be representable in the active dtype: under fp16 AMP,
        values below ~6e-8 underflow to 0 (smallest subnormal); 1e-6 is
        safely representable.

    Backward semantics are identical in both branches: the gradient at non-
    mask positions is exactly 0 (the chain rule through `where` gives 0 for
    the constant branch), so the optimizer never updates those positions and
    they stay at zero throughout training. Compatible with torch.compile.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 density: float, mode: str = "structural",
                 phantom_budget: int | None = None,
                 init_std: float | None = None,
                 epsilon: float = 0.0):
        super().__init__(num_embeddings, embedding_dim)
        p, q, N_prime, phantom_rows = find_pq(
            embedding_dim, num_embeddings, density, mode, phantom_budget
        )
        mask = make_pq_mask(num_embeddings, embedding_dim, p, q, N_prime)
        self.register_buffer("pq_mask", mask, persistent=True)
        self.p = p
        self.q = q
        self.N_prime = N_prime
        self.phantom_rows = phantom_rows
        self.density_target = density
        self.density_actual = mask.float().mean().item()
        self.epsilon = float(epsilon)

        # Re-init with the embedding's standard init, then mask once.
        if init_std is None:
            init_std = 1.0 / math.sqrt(embedding_dim)
        with torch.no_grad():
            nn.init.normal_(self.weight, mean=0.0, std=init_std)
            self.weight.mul_(mask)

    def _masked_weight(self) -> torch.Tensor:
        if self.epsilon > 0.0:
            return torch.where(self.pq_mask, self.weight, self.epsilon)
        return self.weight * self.pq_mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input, self._masked_weight(),
            self.padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse,
        )

    def effective_param_count(self) -> int:
        return int(self.pq_mask.sum().item())

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"p={self.p}, q={self.q}, N'={self.N_prime}, phantom_rows={self.phantom_rows}, "
            f"density_target={self.density_target:.4f}, density_actual={self.density_actual:.4f}, "
            f"epsilon={self.epsilon:.2e}"
        )


class MaskedTiedLinear(nn.Linear):
    """nn.Linear whose weight is tied to a SparsePQEmbedding's parameter
    AND applies that embedding's mask in forward.

    Required for weight-tied LM heads when the embedding is sparse: a
    vanilla `nn.Linear` with a tied weight would NOT multiply by mask in
    forward, so any drift of the parameter at non-mask positions (from
    optimizer updates that snuck through, init noise, or numerical
    perturbations) would leak into the LM head's logits even though the
    embedding lookup itself is masked.

    Inherits the embedding's `epsilon` semantics: if epsilon > 0 then the
    LM head's forward uses `where(mask, weight, epsilon)` for consistency
    with the embedding lookup. Otherwise uses `weight * mask`.

    The mask buffer is registered as a non-persistent reference to the
    embedding's buffer (same tensor object, so device moves stay in sync,
    and no extra checkpoint entry).
    """
    def __init__(self, sparse_embedding: SparsePQEmbedding, bias: bool = False):
        super().__init__(
            sparse_embedding.embedding_dim,
            sparse_embedding.num_embeddings,
            bias=bias,
        )
        # Tie weight to the embedding's parameter (same object).
        self.weight = sparse_embedding.weight
        # Share mask buffer (same tensor, no extra storage). non-persistent
        # so it isn't double-saved in the checkpoint.
        self.register_buffer("pq_mask", sparse_embedding.pq_mask, persistent=False)
        # Inherit epsilon from the embedding so both forward sites use the
        # same masked-weight semantics.
        self.epsilon = sparse_embedding.epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.epsilon > 0.0:
            w = torch.where(self.pq_mask, self.weight, self.epsilon)
        else:
            w = self.weight * self.pq_mask
        return F.linear(x, w, self.bias)
