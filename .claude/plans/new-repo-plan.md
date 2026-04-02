# Plan: Stripped-Down EXARCH — New Repository

## Context

EXARCH has grown to ~7,300 lines in model.py with many experimental features (FDA, SOW/REAP, binary conceptual embeddings, etc.) that have either not proven beneficial or are being released separately. The goal is a clean, simple new repo that retains only the proven core architecture — the wavelet-based spectral mixer — with a standard GPT-2 tokenizer and learned embedding. Priority is **performance over interpretability**. Controllability can be achieved externally via standalone REAP/SOW tools. This positions EXARCH as a novel, accessible attention-free LM for the ML community.

---

## File Structure

```
new_repo/
├── model.py       # ~800-900 lines: all architecture components + ExarchLM + multinodal
├── train.py       # ~500 lines: dataset loading, training loop, benchmark
├── generate.py    # ~400 lines: generation with inference strategies
├── config.json    # config with __comment fields retained for now
├── data/          # dataset storage (auto-populated)
└── logs/          # training run logs
```

---

## 1. model.py — Architecture

### Utility functions (port verbatim)
- `set_seed()`
- `next_pow2()`
- `pad_features_to_pow2()`
- `fwht_ortho_iterative()`
- `causal_haar_decompose()` / `causal_haar_reconstruct()` — used by wavelet coherence in generate.py

### Classes to port

#### FastHadamardTransform
Port verbatim. Hybrid matrix/iterative approach. No config keys.

#### LiftingWaveletDecompose
Port with both paths:
- **linear_only=True (default):** Single Linear(C,C) for P and U — true linear filter bank.
- **linear_only=False:** 2-layer MLP (C → C*hidden_mult → C) with GELU — keep for future tuning.

Keep: `lifting_init` (haar/zero/random), `lifting_dropout`, `lifting_hidden_mult`, `shared_lifting_weights` (toggleable — may perform better unshared in multinodal mode).

#### LiftingWaveletReconstruct
Port verbatim. Shares P/U networks with Decompose.

#### GatedSpectralMixer
Port verbatim. Keep `low_rank` param (currently 0, just 3 lines of code if >0).
`mixer_repeats` hardcoded to 1 (no config key).

#### FeedForward (MLP)
Port verbatim. Configurable expansion and depth (mlp_expansion, mlp_layers).

#### `_compute_running_mean()`
Port verbatim with `@torch.compiler.disable` decorator.

#### ExarchBlock — simplified (~180 lines)
**Hardcoded defaults (no config key):**
- `mixer_repeats: 1` — single pass, no residual loop

**Toggleable via config:**
- `semantic_feedback` (bool) — cross-layer running mean state
- `learned_residual` (bool) — learnable alpha vs standard additive
- `skip_proj_out` (bool) — when True AND C==Cp, skips the proj_out Linear(Cp→C), saving ~5.2M params. When False or C!=Cp, proj_out is kept. Retained as config option because proj_out can serve as a learned per-block channel mixing layer, potentially useful for expansion/contraction experiments.

**Constructor params:**
```
C, levels, low_rank, mlp_expansion, mlp_layers,
dropout_projection, dropout_mixer, dropout_mlp,
semantic_feedback, learned_residual, skip_proj_out,
shared_lifting_module, use_mixer_gate, mixer_gate_activation,
lifting_hidden_mult, lifting_init, lifting_dropout, lifting_linear_only
```

**Simplified forward data flow:**
```
1. [Semantic feedback: compute gate bias per scale from cross-layer state]
2. LayerNorm → pad to Cp
3. Wavelet decompose (lifting or fixed Haar)
4. Stack coefficients [B, T, S, Cp]
5. Add semantic feedback bias (if enabled)
6. FHT forward
7. Per-scale GatedSpectralMixer
8. FHT inverse
9. Scale weights (learned [S] parameter)
10. Wavelet reconstruct
11. proj_out (Linear Cp→C) unless skip_proj_out and C==Cp
12. Residual (learned alpha or standard)
13. LayerNorm → MLP → Residual
14. Return (x, running_mean_state)
```

**Removed entirely:**
- Logic gates, top-down gating, cross-scale bias
- FDA transforms (input/output)
- Gradient suppression hooks
- Hebbian memory
- Wavelet aux loss caching
- Dimensional suppression/amplification buffers
- Activation stats collection

#### ExarchLM — simplified (~200 lines)

**Embedding:** `nn.Embedding(vocab_size, C_embed)` — fully learned, standard init.
- `C_embed` defaults to C (no expansion). Can be set < C to enable expansion.
- When `C_embed < C`: embedding is repeated `(C // C_embed)` times to expand from C_embed to C before the mixer. After all blocks, contracted back via `.mean()` over groups before the LM head.
- This expansion/contraction mechanism is optional and configurable, potentially useful as a performance lever.

**LM Head:** `Linear(C_embed, vocab_size, bias=False)`.
- `tie_embedding_to_lm_head` config key (default: **False**). When True, `lm_head.weight = token_embedding.weight`. Separate by default for maximum performance flexibility.

**Forward pass:**
```
1. x = embedding(idx) + dropout
2. If C_embed < C: x = x.repeat(1, 1, expansion_factor)
3. For each layer (with stochastic depth skip):
   x, state = block(x, prev_state)
4. If C_embed < C: x = x.view(B, T, exp_factor, C_embed).mean(dim=2)
5. x = LayerNorm(x) + dropout
6. logits = lm_head(x)
7. loss = cross_entropy if targets
```

**Removed:**
- Conceptual/binary/hybrid embedding paths
- Output bias, deterministic spacing
- WBM, MTP, interlayer LM heads
- Layer aggregation
- FDA, INLP, dimensional suppression
- All build_*.py script dependencies

#### MultiNodeExarchLM — retained (~100 lines)
Port the multinodal architecture (simplified):
- Multiple independent ExarchLM cells with averaged logits
- Cross-cell gating (`CrossCellGate` class)
- Feature bagging with `multinodal_bagged_eps`
- Config keys: `multinodal_enabled`, `multinodal_num_cells`, `multinodal_cell_dim`, `multinodal_seeds`, `multinodal_combination`, `multinodal_cross_cell_gating`, `multinodal_cross_cell_gate_interval`, `multinodal_features_per_cell`, `multinodal_bagged_eps`

**Note:** NaN instability in fp16 with feature bagging is a known issue. Future work: investigate gradient/parameter scaling proportional to number of nodes to stabilize training.

### Also in model.py
- `Logger` class (simplified — core logging only)
- `parameter_breakdown()` (simplified — no conceptual embedding branch)

---

## 2. train.py — Training

### Dataset loading
```python
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")  # 50,257 vocab
ds = load_dataset("wikitext", "wikitext-103-raw-v1")
# Encode each split to flat torch.long tensor
```

### Training loop (extracted from current model.py lines 5560-6903)
- Config loading
- Model creation + optional `torch.compile`
- Optimizer (configurable: AdamW, Adagrad)
- LR schedule: linear warmup + cosine decay to min_lr
- `get_batch()` — random positions in flat token tensor
- `estimate_loss()` — eval on val set
- Gradient accumulation, AMP, gradient clipping
- Checkpoint management (best model tracking)
- End-of-training: benchmark + generation

### Benchmark (extracted from current evaluate functions)
- Non-overlapping window evaluation
- Sliding window evaluation (configurable stride)
- Report: perplexity, BPB, BPT

### Removed from training:
- SOW/REAP, WBM, FDA, INLP, scheduled sampling
- Adaptive token finetuning, byte loss upweighting
- Sentence start aux loss, calibration
- SpaceInserter / deterministic spacing
- Multinodal support
- Gradient diagnostics

---

## 3. generate.py — Generation

### Core: `generate_one()` with inference strategies
All strategies **toggleable** via CLI args with sensible defaults:
- **Top-P nucleus sampling** — `--top_p 0.95`
- **Repetition penalty** — `--repetition_penalty 1.1`
- **Entropy-adaptive temperature** — `--entropy_adaptive`
- **Wavelet coherence monitoring** — `--wavelet_coherence`
- **Lookahead reranking** — `--lookahead_k 3 --lookahead_depth 5`
- **Best-of-N** — `--best_of_n 3`
- **Quality metrics** (mean log-prob, distinct-n, repetition rate)

### CLI interface:
```
python generate.py --checkpoint logs/run_dir/best_model.pt \
    --prompt "The history of" --num_tokens 512
```

### Encoding/decoding:
```python
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode(prompt)
text = enc.decode(generated_ids)
```

### Removed:
- Custom tokenizer loading / SpaceInserter
- Conceptual embedding coherence bias
- Dimensional min activation enforcement
- CLS, suppress mask logic
- Multinodal support

---

## 4. config.json

Config retains `__comment` fields for self-documentation (can be shortened/moved to README later).

```json
{
    "__comment_dataset": "Dataset to train on. Uses HuggingFace datasets + GPT-2 tokenizer (tiktoken).",
    "dataset": "wikitext-103",
    "out_dir": "logs",
    "compile": true,
    "seed": 1337,

    "__comment_training": "=== TRAINING ===",
    "epochs": 10,
    "micro_batch_size": 8,
    "grad_accum": 2,
    "block_size": 512,
    "eval_interval": 100,

    "__comment_architecture": "=== ARCHITECTURE ===",
    "C": 512,
    "__comment_C": "Mixer working width. Must be power of 2 for Hadamard (or will be padded to next power of 2).",
    "C_embed": 512,
    "__comment_C_embed": "Embedding dimension. When < C, embedding is repeated (C/C_embed) times before mixer, then contracted via mean before LM head. Set equal to C for no expansion.",
    "layers": 20,
    "levels": 9,
    "__comment_levels": "Wavelet decomposition levels. Should be ~log2(block_size).",
    "low_rank": 0,
    "mlp_expansion": 10,
    "mlp_layers": 2,

    "__comment_wavelet": "=== WAVELET ===",
    "wavelet_mode": "lifting",
    "__comment_wavelet_mode": "'lifting' (learnable, default) or 'fixed' (Haar only).",
    "shared_lifting_weights": true,
    "__comment_shared_lifting_weights": "When true, all layers share one lifting module. Saves ~180M params at 20 layers. May perform better unshared in multinodal mode.",
    "lifting_linear_only": true,
    "__comment_lifting_linear_only": "When true, P/U are single Linear(C,C). When false, 2-layer MLP with GELU.",
    "lifting_hidden_mult": 1,
    "lifting_init": "haar",

    "__comment_mixer": "=== SPECTRAL MIXER ===",
    "use_mixer_gate": true,
    "mixer_gate_activation": "silu",
    "__comment_mixer_gate_activation": "'silu' (SwiGLU), 'sigmoid' (vanilla GLU), 'gelu' (GeGLU), 'relu' (ReGLU).",

    "__comment_block": "=== BLOCK OPTIONS ===",
    "semantic_feedback": true,
    "semantic_feedback_cross_window": true,
    "learned_residual": true,
    "skip_proj_out": true,
    "__comment_skip_proj_out": "When true and C is power-of-2, skips proj_out Linear(Cp→C), saving ~5.2M params. Set false to keep proj_out as a learned channel mixing layer.",
    "stochastic_depth_rate": 0.0,

    "__comment_dropout": "=== DROPOUT ===",
    "dropout_embedding": 0.1,
    "dropout_projection": 0.05,
    "dropout_mixer": 0.05,
    "dropout_mlp": 0.05,
    "dropout_lm_head": 0.12,
    "lifting_dropout": 0.0,

    "__comment_optimizer": "=== OPTIMIZER ===",
    "optimizer": "Adagrad",
    "optimizer_eps": 2e-13,
    "lr": 2e-2,
    "min_lr": 2e-4,
    "warmup_fraction": 0.3,
    "__comment_warmup_fraction": "Fraction of total training steps for LR warmup. Auto-computes warmup = fraction * epochs * steps_per_epoch.",
    "grad_clip": 1.0,

    "__comment_embedding": "=== EMBEDDING / LM HEAD ===",
    "tie_embedding_to_lm_head": false,
    "__comment_tie_embedding_to_lm_head": "When true, LM head shares weights with token embedding. False allows separate learned head.",

    "__comment_hardware": "=== HARDWARE ===",
    "gradient_checkpointing": false,
    "use_amp": true,
    "amp_dtype": "fp16",
    "allow_tf32": true,

    "__comment_generation": "=== GENERATION ===",
    "generation_prompt": "The history of",
    "temperature": 1.0,
    "num_new_tokens": 512,
    "top_p": 0.95,
    "repetition_penalty": 1.1,

    "__comment_multinodal": "=== MULTINODAL (Product of Experts) ===",
    "multinodal_enabled": false,
    "multinodal_num_cells": 2,
    "multinodal_cell_dim": 512,
    "multinodal_seeds": [42, 137],
    "multinodal_combination": "average",
    "multinodal_cross_cell_gating": false,
    "multinodal_cross_cell_gate_interval": 1,
    "multinodal_features_per_cell": -1,
    "multinodal_bagged_eps": 1e-6
}
```

**Notes:**
- `warmup_fraction: 0.3` replaces absolute warmup steps — auto-computes from total steps
- `C_embed` enables optional expansion/contraction (set equal to C to disable)
- `tie_embedding_to_lm_head` defaults to False (performance over interpretability)
- `shared_lifting_weights` retained as config option (may help multinodal)
- `lifting_linear_only`, `lifting_hidden_mult`, `lifting_dropout` retained for future tuning
- `skip_proj_out` retained — proj_out can serve as additional learned channel mixing
- Dataset selection retained — GPT-2 tokenizer makes all datasets easy to handle
- Multinodal section retained for future scaling experiments

---

## 5. Key Design Decisions

### Tokenizer
GPT-2 BPE via tiktoken (50,257 tokens). Standard, widely used, many published baselines. No custom pipeline. Multiple datasets supported — the common tokenizer makes switching trivial.

### Embedding
Fully learned `nn.Embedding(vocab_size, C_embed)`. Not tied to LM head by default (configurable). Optional expansion from C_embed to C for mixer, contraction back before LM head.

### Warmup
`warmup_fraction: 0.3` — computes warmup steps automatically from total steps. Eliminates the recurring manual warmup calculation errors.

### Optimizer
Default Adagrad (proven best for EXARCH). Keep configurable for AdamW comparison runs.

### Performance Priority
All defaults optimize for performance, not interpretability. Interpretability features (weight tying, expansion/contraction) are available but off by default.

---

## 6. Potential Challenges

1. **BPE vs word-level tokenization**: GPT-2 BPE tokens represent different text amounts than EXARCH's word tokens. Hyperparameters (especially block_size, LR) may need retuning.
2. **levels vs C**: Must validate `2^levels <= Cp`. At C=512: levels=9 (2^9=512). Config should warn if mismatched.
3. **Cross-window semantic feedback**: Persistent state must reset at generation start (batch size mismatch from training). Port the reset logic.
4. **torch.compile + _compute_running_mean**: The `@torch.compiler.disable` decorator is critical — cumsum causes compilation issues.
5. **Multinodal NaN instability**: Feature bagging with fp16 causes NaN around LR 1.5e-02. Future work: investigate gradient/parameter scaling proportional to number of nodes. The `multinodal_bagged_eps` (default 1e-6) partially mitigates but doesn't fully solve.

---

## 7. Verification

1. **Model builds** for C=256, C=512 with all toggle combinations
2. **Forward pass shapes**: [B, T] → logits [B, T, 50257], scalar loss
3. **Wavelet perfect reconstruction**: decompose → reconstruct ≈ identity
4. **FHT self-inverse**: FHT(FHT(x)) ≈ x
5. **Weight tying**: `lm_head.weight is token_embedding.weight`
6. **Parameter count**: matches expectations (~256M for C=512/20L/exp=10)
7. **100-step smoke test**: loss decreases, checkpoint saves/loads, generation produces text
8. **Feature toggles**: semantic_feedback off, learned_residual off, stochastic_depth on — all work
9. **Full WikiText-103 training run**: compare BPB against current EXARCH with learned embedding

---

## 8. Implementation Order

1. **model.py**: utilities → FHT → wavelets → mixer → MLP → ExarchBlock → ExarchLM
2. **train.py**: dataset loading → training loop → benchmark
3. **generate.py**: checkpoint loading → generation → inference strategies
4. **config.json**: write minimal config
5. **Testing**: smoke test → full training run
