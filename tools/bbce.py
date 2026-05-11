"""Bisected Block Context Extension (BBCE).

Inspired by DeepSeek-V4's HCA. Bisects the input into:
  - Compressed half (first `block_size_compressed - block_size/2` tokens):
    embedded then mean-pooled in chunks of `g` consecutive tokens, producing
    `block_size/2` "compressed slots."
  - Uncompressed half (last `block_size/2` tokens): embedded normally.

The two halves concatenate to a (B, block_size, C) representation that is fed
into the standard WaveletLM pipeline (wavelet decompose → mixer → reconstruct
→ MLP → FwPKM → ...). The compressed half supplies long-range context; the
uncompressed half is where next-token prediction targets live.

The bisection point sits at `block_size/2`, which is a power of 2 because
`block_size` is a power of 2. This preserves the seam at every wavelet level —
each level's dyadic partition aligns with the seam, so compressed and
uncompressed regimes never mix within a single coarse coefficient. Only
O(log block_size) predict/update operations bridge the seam (the windows
that happen to straddle it at each scale).

See `plans/two_d_wavelet_sequential_training.md` (related but separate
feature) and the README BBCE section for the full architectural rationale.

==============================================================================
PHASE 1 (this file): Random-sampling BBCE. Each batch independently samples
a starting position and reads `block_size_compressed` consecutive tokens.
Chunked embedding lookup bounds peak VRAM so even large
`block_size_compressed` (1M, 4M) is feasible without caching.

PHASE 2 (planned): Sequential-ordering BBCE with cross-batch slot caching.
Most of the compressed history overlaps between consecutive batches under
sequential ordering, so we can cache the per-slot mean-pooled embeddings
and only recompute the slot that has changed. Cuts embedding compute by
~99% at large block_size_compressed. Requires sequential block ordering
(see Sequential Block Ordering section in README), which itself has a
known performance cost on Adagrad; that cost has been ~50% recoverable via
lr=0.015. Phase 2 is contingent on Phase 1 showing promise.
==============================================================================
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BBCEPreprocessor(nn.Module):
    """Bisect-and-mean-pool preprocessor for BBCE.

    Operates on token IDs (not embeddings) so it can manage the embedding
    lookup memory budget. Calls the parent model's `token_embedding` in
    chunks of `g` tokens (one compressed slot at a time, or batched to a
    target peak-VRAM budget), means each chunk into a single slot, and
    returns the concatenated (compressed_slots, uncompressed_embeddings)
    tensor of shape (B, block_size, C).

    Args:
        block_size: The "supervised" window size — total output positions
            after bisection (block_size/2 compressed + block_size/2
            uncompressed). Must be a power of 2 so the seam at block_size/2
            aligns with every wavelet level's dyadic partition.
        block_size_compressed: The "context" window size — total token IDs
            read from the corpus per batch element. Should be >= block_size.
        target_peak_chunk_bytes: Bound peak embedding-tensor VRAM during
            chunked lookup. Default 500MB; tune if VRAM-constrained.

    Forward signature:
        idx: (B, block_size_compressed) int64 token IDs
        token_embedding: nn.Embedding (or compatible) module from the parent
            model. Called as `token_embedding(idx_chunk)`.
        Returns: (B, block_size, C) float embeddings ready for the wavelet.

    Group size g: derived as ceil((block_size_compressed - block_size/2) /
    (block_size/2)). The first `num_slots * g` tokens of idx are partitioned
    into num_slots = block_size/2 groups of g; the last block_size/2 tokens
    are uncompressed. If `block_size_compressed - block_size/2` is not an
    exact multiple of block_size/2, the leftover tokens at the boundary are
    dropped (truncation toward the seam).
    """

    def __init__(
        self,
        block_size: int,
        block_size_compressed: int,
        target_peak_chunk_bytes: int = 500 * 1024 * 1024,
    ):
        super().__init__()
        if block_size & (block_size - 1) != 0:
            raise ValueError(
                f"BBCEPreprocessor requires block_size to be a power of 2 "
                f"(seam-preservation property); got {block_size}"
            )
        if block_size_compressed < block_size:
            raise ValueError(
                f"BBCEPreprocessor requires block_size_compressed >= block_size; "
                f"got bc={block_size_compressed}, b={block_size}"
            )

        self.block_size = block_size
        self.block_size_compressed = block_size_compressed
        self.target_peak_chunk_bytes = target_peak_chunk_bytes

        self.num_slots = block_size // 2  # = block_size_uncompressed = block_size/2
        # Compressed region length (after truncating to an exact multiple of num_slots).
        compressed_region = block_size_compressed - self.num_slots
        self.g = compressed_region // self.num_slots
        self.compressed_region_truncated = self.num_slots * self.g
        # Leftover tokens that don't divide evenly: drop them (just before the
        # uncompressed half). num_dropped = compressed_region - num_slots * g.
        self.num_dropped = compressed_region - self.compressed_region_truncated

    def _max_slots_per_chunk(self, B: int, C: int) -> int:
        """Number of compressed slots whose g-token embeddings fit within
        `target_peak_chunk_bytes` at fp16."""
        bytes_per_slot = B * self.g * C * 2  # fp16
        return max(1, self.target_peak_chunk_bytes // bytes_per_slot)

    def forward(
        self,
        idx: torch.Tensor,
        token_embedding: nn.Module,
    ) -> torch.Tensor:
        B, T_in = idx.shape
        if T_in != self.block_size_compressed:
            raise ValueError(
                f"BBCEPreprocessor expected idx of shape (B, {self.block_size_compressed}), "
                f"got (B, {T_in})"
            )

        # Compressed region: first `compressed_region_truncated` tokens.
        # Uncompressed region: last `num_slots` tokens.
        # Anything in between (num_dropped tokens) is discarded.
        compressed_idx = idx[:, : self.compressed_region_truncated]  # (B, num_slots * g)
        uncompressed_idx = idx[:, -self.num_slots:]  # (B, num_slots)

        # Reshape compressed_idx for slot-wise indexing: (B, num_slots, g)
        compressed_idx = compressed_idx.view(B, self.num_slots, self.g)

        # Need C to compute chunk size. Probe the embedding module.
        if hasattr(token_embedding, "embedding_dim"):
            C = token_embedding.embedding_dim
        elif hasattr(token_embedding, "weight") and hasattr(token_embedding.weight, "shape"):
            C = token_embedding.weight.shape[1]
        else:
            # Fallback: do a tiny probe.
            probe = token_embedding(idx[:1, :1])
            C = probe.shape[-1]

        slots_per_chunk = self._max_slots_per_chunk(B, C)

        # Chunked embed + mean-pool. Process `slots_per_chunk` slots at a time
        # to bound peak VRAM. Each chunk:
        #   1. Take (B, chunk_slots, g) indices, flatten to (B, chunk_slots*g)
        #   2. Embed → (B, chunk_slots*g, C)
        #   3. Reshape to (B, chunk_slots, g, C)
        #   4. Mean over dim=2 (the g axis) → (B, chunk_slots, C)
        #   5. Discard the full embedded tensor
        compressed_slots = []
        for chunk_start in range(0, self.num_slots, slots_per_chunk):
            chunk_end = min(chunk_start + slots_per_chunk, self.num_slots)
            chunk_size = chunk_end - chunk_start

            chunk_idx = compressed_idx[:, chunk_start:chunk_end, :]  # (B, chunk_size, g)
            chunk_idx_flat = chunk_idx.reshape(B, chunk_size * self.g)
            chunk_emb = token_embedding(chunk_idx_flat)  # (B, chunk_size*g, C)
            chunk_emb = chunk_emb.view(B, chunk_size, self.g, C)
            chunk_mean = chunk_emb.mean(dim=2)  # (B, chunk_size, C)
            compressed_slots.append(chunk_mean)
            del chunk_emb  # free intermediate

        compressed_slots = torch.cat(compressed_slots, dim=1)  # (B, num_slots, C)

        # Uncompressed half: normal embedding.
        uncompressed_emb = token_embedding(uncompressed_idx)  # (B, num_slots, C)

        # Concatenate: compressed_slots first (positions 0..num_slots-1), then
        # uncompressed (positions num_slots..block_size-1). The seam is at
        # position num_slots = block_size/2.
        return torch.cat([compressed_slots, uncompressed_emb], dim=1)  # (B, block_size, C)

    def extra_repr(self) -> str:
        return (
            f"block_size={self.block_size}, "
            f"block_size_compressed={self.block_size_compressed}, "
            f"num_slots={self.num_slots}, g={self.g}, "
            f"num_dropped={self.num_dropped} (boundary truncation), "
            f"target_peak_chunk_bytes={self.target_peak_chunk_bytes / 1024**2:.0f}MB"
        )


def build_bbce_preprocessor(config: dict) -> BBCEPreprocessor:
    """Construct a BBCEPreprocessor from the config dict."""
    return BBCEPreprocessor(
        block_size=int(config["block_size"]),
        block_size_compressed=int(config["block_size_compressed"]),
        target_peak_chunk_bytes=int(
            config.get("bbce_target_peak_chunk_bytes", 500 * 1024 * 1024)
        ),
    )


def pad_idx_for_bbce(
    idx: torch.Tensor,
    block_size_compressed: int,
    pad_id: int = 0,
) -> torch.Tensor:
    """Left-pad idx to (B, block_size_compressed) using pad_id.

    BBCEPreprocessor strictly requires the exact bs_compressed shape because
    its bisection and chunked-pooling math depend on it. At inference and
    benchmark time, the available context is often shorter (a 3-token prompt;
    a test-set window where bs_compressed > test_len). Left-padding fills the
    compressed half so the supervised half (rightmost block_size/2 positions)
    holds the real context closest to the prediction site.

    If idx is longer than bs_compressed, take only the trailing bs_compressed
    tokens (typical generation case where idx grows past the context window).
    """
    B, T = idx.shape
    if T == block_size_compressed:
        return idx
    if T > block_size_compressed:
        return idx[:, -block_size_compressed:]
    pad = torch.full(
        (B, block_size_compressed - T),
        pad_id,
        dtype=idx.dtype,
        device=idx.device,
    )
    return torch.cat([pad, idx], dim=1)
