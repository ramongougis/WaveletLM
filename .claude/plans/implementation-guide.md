---
name: New repo implementation guide
description: Detailed instructions for building stripped-down EXARCH in a new repo — port from current model.py with GPT-2 tokenizer
type: project
---

## Overview
Create a new repo with a simplified EXARCH: wavelet-based attention-free LM with learned embedding and GPT-2 tokenizer. Priority is performance, not interpretability. Full plan is at `C:\Users\Tippy\.claude\plans\glistening-plotting-goose.md`.

## Source File Reference (EXARCH-research repo — the OLD repo)
All line numbers reference: `c:\Users\Tippy\OneDrive\Desktop\AI\EXARCH-research\model.py`
Generate.py references: `c:\Users\Tippy\OneDrive\Desktop\AI\EXARCH-research\generate.py`
New code goes into: `c:\Users\Tippy\OneDrive\Desktop\AI\EXARCH\` (the NEW repo)

## Implementation Order

### Step 1: model.py — Utilities & Core Components

**Port verbatim:**
- `set_seed()` (lines 63-73)
- `next_pow2()` (lines 75-76)
- `pad_features_to_pow2()` (lines 448-452)
- `fwht_ortho_iterative()` (lines 555-575)
- `FastHadamardTransform` class (lines 577-612) — hybrid matrix/iterative
- `causal_haar_decompose()` (lines 614-632) — used by wavelet coherence in generate.py
- `causal_haar_reconstruct()` (lines 634-650)
- `_compute_running_mean()` (lines 1412-1437) — MUST keep `@torch.compiler.disable`

**Port `LiftingWaveletDecompose` (lines 655-825):**
- Keep both linear_only and MLP paths
- Keep `lifting_init` (haar/zero/random), `lifting_dropout`, `lifting_hidden_mult`
- Constructor: `(levels, C, hidden_mult=1, init_wavelet='haar', dropout=0.0, linear_only=True)`
- Forward returns `(approx, [details...])`
- Algorithm: split even/odd with dilation=2^level, predict detail, update approx

**Port `LiftingWaveletReconstruct` (lines 828-874):**
- Shares P/U networks with Decompose
- Reverse lifting: undo update, reconstruct from approx+details

**Port `GatedSpectralMixer` (lines 1234-1274):**
- `mixer` Linear(Cp, Cp, bias=False) — init near-identity + noise
- `gate` Linear(Cp, Cp, bias=False) — SwiGLU gating
- Optional low_rank U/V (currently unused, rank=0, but keep the 3 lines)
- Forward: `signal * gate_activation(gate(X))` + optional low-rank residual

**Port `FeedForward` (lines 1382-1407):**
- C → C*expansion → [GELU → C*expansion →]* → C
- `mlp_layers` controls depth (default 2)
- Final layer init: weight *= 0.02, bias = 0

### Step 2: model.py — ExarchBlock

**Port from lines 1633-2264 with heavy simplification.**

Constructor params:
```
C, levels, low_rank, mlp_expansion, mlp_layers,
dropout_projection, dropout_mixer, dropout_mlp,
semantic_feedback, learned_residual, skip_proj_out,
shared_lifting_module, use_mixer_gate, mixer_gate_activation,
lifting_hidden_mult, lifting_init, lifting_dropout, lifting_linear_only
```

Key components to create in __init__:
- `ln1 = LayerNorm(C)`, `ln2 = LayerNorm(C)`
- Lifting decompose/reconstruct (shared or per-block)
- FHT instance
- `scale_mixers` = ModuleList of GatedSpectralMixer, one per level+1
- `scale_weights` = Parameter([levels+1])
- `proj_out` = Linear(Cp, C) — only if not (skip_proj_out and Cp==C)
- `ffwd` = FeedForward(C, expansion, dropout_mlp, mlp_layers)
- If semantic_feedback: `history_gains` Parameter([S, C]), `cross_layer_mix` Linear(C, C, bias=False)
- If learned_residual: `residual_alpha_spectral`, `residual_alpha_mlp` Parameters (init=1.0)
- `dropout_proj` = Dropout(dropout_projection)

Forward (`_forward_once(x, prev_state)`):
1. Compute running mean if semantic_feedback
2. Mix with prev_state via cross_layer_mix
3. Compute gate_bias_scales: per-scale `mixed * history_gains[s]`, padded to Cp
4. `h = ln1(x)`, pad to Cp
5. Wavelet decompose → stack `[B, T, S, Cp]`
6. Add gate_bias (if semantic_feedback)
7. FHT forward
8. Per-scale mixer (loop over S scales)
9. FHT inverse
10. Unstack, apply `scale_weights[s]`
11. Wavelet reconstruct
12. proj_out (if not skipped)
13. Residual: `x = alpha*x + dropout(projected)` or `x = x + dropout(projected)`
14. `h2 = ln2(x)`, MLP residual same pattern
15. Return `(x, current_running_mean)`

**DO NOT PORT:** logic gates, top-down gating, cross-scale bias, FDA, gradient suppression, Hebbian, wavelet aux loss, activation stats, dimensional suppression.

### Step 3: model.py — ExarchLM

**Port from lines 2635-3502 with heavy simplification.**

Constructor:
- `token_embedding = nn.Embedding(vocab_size, C_embed)`
- If `C_embed < C`: `expansion_factor = C // C_embed` (validate divisibility)
- Else: `expansion_factor = 1`
- `dropout_emb = Dropout(dropout_embedding)`
- `dropout_lm = Dropout(dropout_lm_head)`
- Build layers: ModuleList of ExarchBlock (optionally sharing lifting module)
- `final_ln = LayerNorm(C_embed)`
- `lm_head = Linear(C_embed, vocab_size, bias=False)`
- If `tie_embedding_to_lm_head`: `lm_head.weight = token_embedding.weight`
- Stochastic depth: `_drop_probs = [(l/L) * rate for l in range(L)]`
- Semantic feedback cross-window state

Forward:
```python
x = self.token_embedding(idx)  # [B, T, C_embed]
x = self.dropout_emb(x)
if self.expansion_factor > 1:
    x = x.repeat(1, 1, self.expansion_factor)  # [B, T, C]
current_state = self._persistent_semantic_state
for layer_idx, layer in enumerate(self.layers):
    # Stochastic depth
    if self.training and self.stochastic_depth_rate > 0:
        if random.random() < self._drop_probs[layer_idx]:
            continue
    # Gradient checkpointing
    if self.gradient_checkpointing and self.training:
        x, current_state = checkpoint(lambda lx, _l=layer, _s=current_state: _l(lx, _s), x, use_reentrant=False)
    else:
        x, current_state = layer(x, current_state)
# Contract if expanded
if self.expansion_factor > 1:
    x = x.view(B, T, self.expansion_factor, self.C_embed).mean(dim=2)
# Update persistent state
if self.semantic_feedback_cross_window and current_state is not None:
    self._persistent_semantic_state = current_state[:, -1, :].detach()
x = self.final_ln(x)
x = self.dropout_lm(x)
logits = self.lm_head(x)
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1)) if targets is not None else None
return logits, loss
```

### Step 4: model.py — CrossCellGate & MultiNodeExarchLM

**Port `CrossCellGate` (inserted before MultiNodeExarchLM in current code):**
- `proj = Linear(C, C, bias=False)` — zero-initialized
- Forward: for each cell, gate = `h * (1 + tanh(proj(mean_of_others)))`

**Port `MultiNodeExarchLM` (lines 3680+):**
- Creates multiple ExarchLM cells
- Feature bagging with `multinodal_bagged_eps` (default 1e-6, not 0.0!)
- Cross-cell gating: lockstep block-by-block with CrossCellGate
- Logit averaging (product of experts)
- Key config: num_cells, cell_dim, seeds, cross_cell_gating, gate_interval, features_per_cell

**IMPORTANT for multinodal with learned embedding:**
The current multinodal uses the frozen BCE embedding for feature assignment. With a learned embedding, feature bagging becomes simpler: just randomly zero (set to eps) some dimensions of the learned embedding per cell. No concept protection needed (no FDA dims to protect). The `build_feature_assignments` function can be greatly simplified.

### Step 5: train.py

**Dataset loading with tiktoken:**
```python
import tiktoken
from datasets import load_dataset
enc = tiktoken.get_encoding("gpt2")  # 50,257 tokens
ds = load_dataset("wikitext", "wikitext-103-raw-v1")
# Encode each split: "\n\n".join(split["text"]) → enc.encode() → torch.long tensor
```

**Training loop (simplified from model.py lines 5560-6903):**
- LR schedule: `get_lr()` — linear warmup + cosine decay
  - `warmup_steps = int(warmup_fraction * total_steps)`
  - Warmup: `lr * step / warmup_steps`
  - Cosine: `min_lr + 0.5*(lr - min_lr)*(1 + cos(pi * progress))`
- `get_batch()` — random positions in flat token tensor
- `estimate_loss()` — eval N batches on val set
- Gradient accumulation with AMP autocast
- GradScaler only for fp16 (not bf16)
- Checkpoint: save best model by val loss
- End-of-training: benchmark (sliding window BPB) + generate

**Benchmark functions (port from current evaluate_full_validation / evaluate_sliding_window):**
- Non-overlapping window: step through test set in block_size chunks
- Sliding window: stride = block_size // 2, only score non-overlapping tokens
- Report: perplexity, BPB, BPT

### Step 6: generate.py

**Checkpoint loading:**
- Load config.json from run directory
- Build model from config
- Load state dict (handle `_orig_mod.` prefix from torch.compile)
- tiktoken for encode/decode

**Core `generate_one()` function (port from current generate.py lines 362-610):**
- Temperature sampling
- Top-P nucleus filtering (lines 240-269)
- Repetition penalty (lines 211-237)
- Entropy-adaptive temperature (lines 508-517)
- Wavelet coherence monitoring (lines 416-434, 521-538)
- Lookahead reranking (lines 336-358, 545-577)
- Quality metrics (lines 272-313)

**Best-of-N (lines 796-835):** Generate N, select by highest mean log-prob.

**CLI:** `python generate.py --checkpoint path --prompt "..." --num_tokens 512`

### Step 7: config.json
See plan file for full config structure with __comment fields.

## Critical Implementation Notes

1. **`@torch.compiler.disable` on `_compute_running_mean`** — cumsum breaks torch.compile
2. **Persistent semantic state reset** — must reset `_persistent_semantic_state = None` at generation start (batch size changes from training)
3. **FHT is self-inverse** — same operation for forward and inverse (orthogonal)
4. **Scale ordering** — wavelet coefficients are stacked top-down: `[approx, detail_coarsest, ..., detail_finest]`
5. **Lifting wavelet causality** — uses `dilation = 2^level` for causal (no future) access
6. **proj_out guard** — `skip_proj_out = config_skip and (Cp == C)`
7. **expansion_factor validation** — `C % C_embed == 0` required
8. **GradScaler** — only create for fp16, not bf16
9. **Multinodal bagged_eps** — use config value (default 1e-6), NOT 0.0
