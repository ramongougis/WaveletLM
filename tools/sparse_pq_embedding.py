"""Sparse (p, q) phantom-token embedding compression.

Implements the deterministic 1D-walk sparsity scheme described in
plans/new_compression_ideas.md. The embedding's (N, C) weight matrix
is masked along the flattened tensor by alternating step sizes p and q,
yielding density 2 / (p + q). Phantom tokens (vocab padded to N' >= N
to enable q | N') are a numerology device only — never allocated.

Sparsity is enforced via:
  1. Init-time mask application: weight = init * mask  (non-mask positions = 0)
  2. Gradient hook: gradients at non-mask positions are zeroed before the
     optimizer step, so non-mask weights stay at 0 throughout training.

Weight tying with the LM head works automatically because both modules
read the same underlying Parameter, which is held at zero on non-mask
positions.
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
    """nn.Embedding with (p, q) phantom-token sparsity enforced by gradient masking.

    The mask is applied once to the weight at construction (so non-mask positions
    are zero from init) and a gradient hook re-zeroes gradients at non-mask
    positions before each optimizer step, keeping the weight masked throughout
    training. No per-forward multiplication is needed; tied LM heads see masked
    weights automatically because both read the same Parameter.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 density: float, mode: str = "structural",
                 phantom_budget: int | None = None,
                 init_std: float | None = None):
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

        # Re-init with the embedding's standard init, then mask.
        if init_std is None:
            init_std = 1.0 / math.sqrt(embedding_dim)
        with torch.no_grad():
            nn.init.normal_(self.weight, mean=0.0, std=init_std)
            self.weight.mul_(mask)

        # Gradient hook: zero gradients at non-mask positions before optimizer step.
        # Captured via closure so the hook survives device moves of self.pq_mask
        # (the buffer reference inside self resolves at hook-call time).
        def _grad_hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * self.pq_mask.to(dtype=grad.dtype, device=grad.device)
        self.weight.register_hook(_grad_hook)

    def effective_param_count(self) -> int:
        return int(self.pq_mask.sum().item())

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"p={self.p}, q={self.q}, N'={self.N_prime}, phantom_rows={self.phantom_rows}, "
            f"density_target={self.density_target:.4f}, density_actual={self.density_actual:.4f}"
        )
