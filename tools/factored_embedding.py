"""Factored embedding with separate learnable input decoder and output encoder.

Architecture:
  Input path:
    tokens → embedding(V, C_emb) → decoder(C_emb, C, bias=True) → C-dim
  Output path:
    C-dim hidden → encoder(C, C_emb, bias=True) → embedding.weight^T (tied) → V logits

The decoder and encoder are *separate* learnable matrices because the model's
nonlinearities (GELU in MLP, gating in mixer, lifting cascade) make the
output-side hidden representation a nonlinear transform of the input-side
embedding. The optimal C → C_emb compression on the output therefore differs
from the inverse of the input expansion, so each direction gets its own
learnable matrix. This is parameter-cheaper than ALBERT's full H × V output
projection while preserving the structural separation between input and
output projections.

Total parameters (with bias on both projections):
  V·C_emb              (embedding.weight, also tied as the V projection on output)
  + C·C_emb + C        (decoder.weight + decoder.bias)
  + C·C_emb + C_emb    (encoder.weight + encoder.bias)

Examples for V=50257, C=2048:
  C_emb=512:   25.73M + 1.05M + 1.05M ≈ 27.83M  (vs 102.93M dense → 73% reduction)
  C_emb=256:   12.87M + 0.53M + 0.53M ≈ 13.92M  (87% reduction)
  C_emb=2048:  102.93M + 4.20M + 4.20M ≈ 111.33M (no compression; learnable C×C
               refinement layers around the standard embedding)

Setting C_emb == C disables compression but keeps the encoder/decoder
machinery active — useful as a "factored, full-rank" reference / ablation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactoredEmbedding(nn.Module):
    """Embedding with learnable input expansion (decoder) and output compression
    (encoder). The vocab projection (V dim) is shared between input lookup and
    output logits via the embedding matrix.
    """
    def __init__(self, num_embeddings: int, C_emb: int, C: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.C_emb = C_emb
        self.C = C

        self.embedding = nn.Embedding(num_embeddings, C_emb)
        self.decoder = nn.Linear(C_emb, C, bias=True)
        self.encoder = nn.Linear(C, C_emb, bias=True)

        # Embedding-style normal init for the V × C_emb lookup table.
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(C_emb))
            # Decoder/encoder use nn.Linear's default Kaiming-uniform init; bias zeroed.
            nn.init.zeros_(self.decoder.bias)
            nn.init.zeros_(self.encoder.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility shim for code that checks `embedding.weight` directly
        (e.g., the `lm_head.weight is token_embedding.weight` tied-detection)."""
        return self.embedding.weight

    @property
    def embedding_dim(self) -> int:
        """Compatibility shim: nn.Embedding has this attribute."""
        return self.C

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # tokens → C_emb-dim embeddings → decoder → C-dim
        return self.decoder(self.embedding(ids))

    def output_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # C-dim hidden → encoder → C_emb-dim → embedding.weight^T → V logits
        compressed = self.encoder(hidden)
        return F.linear(compressed, self.embedding.weight)

    def effective_param_count(self) -> int:
        """Total learnable parameter count of the factored embedding.

        Includes embedding, decoder (weight + bias), and encoder (weight + bias).
        """
        return (
            self.embedding.weight.numel()
            + sum(p.numel() for p in self.decoder.parameters())
            + sum(p.numel() for p in self.encoder.parameters())
        )

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, C_emb={self.C_emb}, C={self.C}, "
            f"params={self.effective_param_count():,}"
        )


class FactoredTiedLMHead(nn.Module):
    """LM head that delegates to a FactoredEmbedding's output_logits.

    Stores the FactoredEmbedding via a non-submodule attribute (using
    `object.__setattr__` to bypass `nn.Module.__setattr__`) so the embedding's
    parameters aren't double-listed in state_dict / parameters() under both
    `token_embedding` and `lm_head` paths. The shared parameters are still
    visible to model.parameters() via the token_embedding path.
    """
    def __init__(self, factored_embedding: FactoredEmbedding):
        super().__init__()
        # Bypass nn.Module's submodule registration so this attribute does not
        # appear in self._modules. PyTorch's parameters() / state_dict() walks
        # _modules; without registration, factored_embedding's params are NOT
        # double-listed under lm_head. They remain accessible via the original
        # token_embedding path.
        object.__setattr__(self, "_factored_embedding", factored_embedding)

    @property
    def factored(self) -> FactoredEmbedding:
        return self._factored_embedding

    @property
    def weight(self) -> torch.Tensor:
        """For compatibility with the `lm_head.weight is token_embedding.weight`
        tied-detection check."""
        return self._factored_embedding.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._factored_embedding.output_logits(x)
