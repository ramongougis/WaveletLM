<p align="center">
  <img src="assets/waveletlm-header.svg" alt="WaveletLM" width="90%"/>
</p>

<br>

---

<br>

WaveletLM is a wavelet-based, attention-free language model that replaces attention with spectral mixing. Each block processes the input sequence using learned lifting wavelet decomposition, Fast Hadamard Transform, per-scale gated spectral mixing with SwiGLU activation, inverse FHT, and wavelet reconstruction. Combined with expanded MLPs and cross-layer decompose bypass, this produces a fully causal sequence language model with no attention mechanism, no quadratic scaling, and no key/value cache.

**Contents:**

 · [Installation](#installation)
 · [Training](#training)
 · [Generation](#generation)
 · [Architecture](#architecture)
 · [Multinodal](#multinodal)
 · [Results](#results)
 · [Post-Release Plans](#post-release-plans)
 · [License](#license)
 · [References](#references)

## Installation

Requires Python 3.10+, PyTorch 2.8+, and CUDA.

```bash
git clone https://github.com/ramongougis/WaveletLM.git
cd WaveletLM
pip install torch datasets tiktoken tqdm numpy
```

## Training

The configuration lives in `config.json`. Edit it to set model dimensions, dataset, optimizer, and hardware options, then run:

```bash
python train.py
```

The shipped defaults reproduce the headline ~880M-parameter WikiText-103 run (L=2, C=2048, MLP=20, PLE, PKM+FwPKM=16384, 5 epochs, 2.0× dropout). Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `C` | 2048 | Mixer working width (power of 2) |
| `layers` | 2 | Number of WaveletLM blocks |
| `mlp_expansion` | 20 | MLP hidden dim multiplier |
| `levels` | 5 | Wavelet decomposition levels (~log2(block_size)) |
| `epochs` | 5 | Training epochs |
| `block_size` | 256 | Context length |
| `dataset` | wikitext-103 | HuggingFace dataset name |
| `optimizer` | Adagrad | Adagrad or AdamW |
| `amp_dtype` | fp16 | fp16 or bf16 |

Training logs, checkpoints, and configs are saved to `logs/<dataset>_<timestamp>/`. Results from all runs are tracked in [`runs.md`](runs.md). The full default run takes ~14h on an RTX 5090; drop `epochs` to 1 for a quick smoke test.

## Generation

```bash
# Default generation example without inference strategies enabled
python generate.py --checkpoint logs/<run_dir>/best_model.pt
```

```bash
# Some additional options
python generate.py --checkpoint logs/<run_dir>/best_model.pt \
    --prompt "Put a prompt here." --num_tokens 1024 --seed 1337 --n 1 \
    --temperature 1.0
```

Optional inference strategies:

```bash
# Use all strategies (recommended)
python generate.py --checkpoint best_model.pt --strategies

# Entropy-adaptive temperature
python generate.py --checkpoint best_model.pt --entropy_adaptive

# Lookahead reranking
python generate.py --checkpoint best_model.pt --lookahead_k 3 --lookahead_depth 5

# Best-of-N sampling
python generate.py --checkpoint best_model.pt --best_of_n 5

# Clean WikiText-103 spacing artifacts (remove space before periods)
python generate.py --checkpoint best_model.pt --clean_spacing

# Wavelet coherence monitoring
python generate.py --checkpoint best_model.pt --wavelet_coherence
```

## Architecture

```
Input tokens
    |
    v
Learned Embedding (C)
    |
    v
+----------------------------+
| WaveletLM Block (x layers) |
|                            |
|  LayerNorm                 |
|  Lifting Wavelet Decompose |
|  Fast Hadamard Transform   |
|  Gated Spectral Mixer      |
|  Fast Hadamard Inverse     |
|  Learned Per-Scale Weights |
|  Wavelet Reconstruct       |
|  Learned Residual 1        |
|  LayerNorm                 |
|  Feedforward Layers (MLP)  |
|  Learned Residual 2        |
+----------------------------+
    |
    v
LayerNorm 
    |
    v
LM Head
    |
    v
Output tokens
```

### Key Components

- **Learnable lifting wavelet decomposition**: predict/update networks (initialized to Haar) decompose each sequence into multi-scale coefficients at every block, trained end-to-end. Causality is preserved through zero-padded dilation. Ablations show that sharing weights between the decompose and reconstruct paths is sufficient; untying yields no BPB improvement, suggesting the wavelet acts as a well-conditioned feature extractor whose inverse passes through cleanly while the mixer and MLP carry the learned transformation.

- **Fast Hadamard Transform (FHT)**: a fixed orthogonal O(C log C) cross-channel rotation that replaces attention's channel-mixing role. Cost is independent of sequence length: no quadratic blow-up regardless of context size, and no KV cache at inference.

- **Per-scale gated spectral mixer (SwiGLU)**: mixes each wavelet scale independently in Hadamard space through a gated linear layer. Captures interactions within each frequency band without forcing cross-band mixing, and runs in fixed O(S²) cost per layer for S wavelet scales regardless of context size - versus attention's O(N²) in sequence length.

- **Expanded MLP (expansion ≥ 20)**: the model's primary knowledge-storage mechanism, scaled well beyond the ~4× typical in Transformers. MLP expansion is a monotonic contributor to BPB in our ablations, taking on much of the memorization role that attention plays in Transformer architectures.

- **Decompose bypass**: a causal cumulative mean of pre-decompose hidden states, projected per-scale and added as bias to the post-decompose wavelet coefficients. Provides cross-layer global context at O(C) cost per token, with no attention required.

<details>
<summary><b>Additional key components</b> (always-on architectural pieces)</summary>

- Pre-norm LayerNorms on both paths of each block, plus one final LayerNorm before the LM head
- Two residual connections per block with learned scalar gating (`learned_residual` in config.json)
- Learned per-scale weights applied after the inverse FHT — one trainable scalar per wavelet scale
- Feature padding to the next power of 2, required for the Hadamard transform (`C` → `Cp = next_pow2(C)`)
- Causal zero-padded dilation in the lifting predict/update steps, preserving autoregressive causality at every level

</details>

### Optional features

- **Per-Layer Embedding**: adds a learned per-channel residual of the original token embedding at each block, letting deeper blocks reach back to the input representation when relevant.

- **Product Key Memory / Fast-Weight Product Key Memory**: sparse key-value memory modules that complement the dense MLP, providing parameter-efficient long-tail pattern storage with optional inference-time fast-weight updates.

- **Low-Rank Factorization**: adds a rank-r perturbation `U·V^T` to the spectral mixer, expanding mixing expressivity at trivial parameter cost (rank=4 yields a measurable BPB improvement).

- **Exponential Parametrization**: reparameterizes mixer weights through `exp()`, stabilizing training under high learning rates that would otherwise NaN.

- **Cross-scale gating (routing mode)**: a learned (S, S) routing matrix that mixes per-scale inputs before each gate, enabling conditional cross-scale interactions (e.g., "when scale 0 shows pattern X, modulate scale 4's processing"). Initialized to identity so it begins as a no-op and only contributes what it learns.

- **Per-scale mixer widths**: asymmetric per-scale mixer capacity. Coarse scales keep full mixer width while fine scales use reduced width via in/out projections. At widths `[1, 1, 1, 0.5, 0.5, 0.5]`, yields a small BPB improvement and a ~23% per-epoch speedup.

- **Wavelet crawl**: learned soft-mixed dilations per wavelet level. Instead of fixed `2^l`, each level sees a softmax-weighted mixture of K candidate offsets around the base dilation, letting the model discover slightly off-power-of-2 receptive fields. K=3 (±1 search radius) is the stable sweet spot.

- **Shared lifting weights**: a single lifting wavelet module shared across all blocks instead of per-block weights. Essentially free on BPB while cutting training VRAM by the weight of L−1 lifting modules (~5–10% at L=2).

- **Looped blocks (Universal Transformer-style)**: apply one shared block K times in place of stacking L distinct blocks. Achieves BPB reduction at fixed parameter count by trading parameters for compute, though the same compute is usually better spent on additional training epochs of the stacked model.

<details>
<summary><b>Additional optional features</b> (all configurable in <code>config.json</code>)</summary>

- Cross-layer decompose bypass state carry (`decompose_bypass_cross_window`)
- Stable-parametrization master flag (`stable_parametrization`)
- Spectral-norm constraint on mixer weights (`stab_spectral_norm`)
- MLP final-layer variance scaling (`stab_ff_scaling`)
- √C embedding output scaling (`stab_embed_scaling`)
- Projection-out residual-stream scaling (`stab_proj_out_scaling`)
- Mixer init-epsilon scaling (`stab_mixer_eps_scaling`)
- Per-level lifting init damping (`stab_lifting_level_scaling`)
- Multi-basis (K parallel) lifting wavelets (`multi_basis_lifting`, `multi_basis_inits`)
- Untied reconstruction weights (`untied_reconstruction`)
- Linear-only lifting networks — no GELU (`lifting_linear_only`)
- Stacked spectral mixer depth (`mixer_depth`, `mixer_depth_stabilizers`, `mixer_depth_residuals`)
- LoopLM mode — full-stack iterated inference (`loop_iterations`)
- Weight tying between embedding and LM head (`tie_embedding_to_lm_head`)
- Output-projection skip when C equals Cp (`skip_proj_out`)
- Gradient checkpointing (`gradient_checkpointing`)
- Stochastic depth (`stochastic_depth_rate`)
- Per-component dropouts (`dropout_embedding`, `dropout_projection`, `dropout_mixer`, `dropout_mlp`, `dropout_lm_head`)
- Lifting-network hidden-dim multiplier (`lifting_hidden_mult`)
- Lifting initialization choice — Haar / zero / random (`lifting_init`)
- Lifting dropout (`lifting_dropout`)
- Spectral mixer gate toggle and activation (`use_mixer_gate`, `mixer_gate_activation`)
- Non-learned fixed-Haar fallback for the wavelet (`wavelet_mode="haar"`)
- Multinodal feature bagging mode and its sub-flags (`multinodal_enabled`, `multinodal_num_cells`, `multinodal_cell_dim`, `multinodal_seeds`, `multinodal_combination`, `multinodal_cross_cell_gating`, `multinodal_features_per_cell`, `multinodal_bagged_eps`)

</details>

### Multinodal

WaveletLM supports a product-of-experts mode where multiple independent model cells process the input in parallel with different feature subsets (feature bagging), then combine logits via averaging. Enable with `multinodal_enabled: true` in config.

## Results

### WikiText-103 Perplexity Comparison

| Model | Type | Trained on | Params | PPL |
|-------|------|-----------|--------|-----|
| Transformer-XL* | Transformer + recurrence* | WikiText-103 (~0.5GB)* | 257M* | ~18[^2]* |
| GPT-2 XL | Transformer | WebText (~40GB) | 1.5B | ~18[^3] |
| GPT-2 Large | Transformer | WebText (~40GB) | 774M | ~19[^3] |
| S4* | SSM* | WikiText-103 (~0.5GB)* | 130M* | ~20[^4]* |
| GPT-2 Medium | Transformer | WebText (~40GB) | 355M | ~22[^3] |
| **WaveletLM** | **Wavelet mixer** | **WikiText-103 (~0.5GB)** | **1.18B** | **~24**[^1] |
| GPT-2 | Transformer | WebText (~40GB) | 124M | ~29[^3] |

\* Trained and evaluated on the same dataset (direct comparison to WaveletLM).

WaveletLM achieves this with only 2 layers (L=2, C=2048), no attention, and no KV cache. Validation loss was still improving at epoch 5, indicating further training will improve results. Comparison numbers are approximate and sourced from respective papers; see references below.

[^1]: L=2, C=2048, MLP=20, PLE, PKM+FwPKM-16384, 5 epochs, 2.0x dropout. BPB=1.0247. See [training log](logs/wikitext-103_2026-04-14_09-07-12/log.txt).

See [`runs.md`](runs.md) for a full log of training runs, configs, and benchmark results.

## Post-Release Plans

### Model comparisons

Side-by-side benchmarks against Transformer, Mamba, RWKV, and other modern architectures on WikiText-103 at matched compute and fully optimized. See [`runs.md`](runs.md#planned-model-comparisons-wikitext-103-matched-compute).

### Dataset comparisons

The best WaveletLM config trained on WikiText-103, PG-19, Pile-ArXiv, BookCorpusOpen, TinyStories, and OpenWebText separately to gauge performance. See [`runs.md`](runs.md#planned-dataset-comparisons-best-config-feasible-epochs).

### Scaled-up model (B200)

The current headline model (882.51M params: L=2, C=2048, MLP=20, PKM/FwPKM=16384) was trained on a single RTX 5090 due to budget constraints. A B200 (192 GB HBM3e) unlocks roughly an order of magnitude more parameter budget at training time and makes several scaling levers practical to stack simultaneously:

- **Width (C):** 2048 → 4096 (mixer working width)
- **Depth (L):** 2 → 4–8 (WaveletLM blocks)
- **MLP expansion:** 20 → 50–200 (primary knowledge-storage lever; monotonic BPB contributor in ablations)
- **PKM / FwPKM keys:** 16384 → 65536 (4× sparse memory capacity)

The target is a ~10–15B parameter configuration chosen after the 5090 sweep completes. Two training targets are planned:

- **WikiText-103 only**, for an apples-to-apples comparison against the 5090 headline run and against prior same-dataset baselines (Transformer-XL, S4).
- **Multi-dataset training** across a broader mix (PG-19, Pile-ArXiv, BookCorpusOpen, TinyStories, OpenWebText, and WikiText-103), establishing WaveletLM's behavior as a general-purpose language model across domains rather than a single-benchmark result.

fp16 inference should fit a single RTX 4090 (24 GB); Post-training quantization (PTQ with per-scale mixed precision, 8/4/2-bit) is expected to drop inference VRAM below 8 GB, enabling deployment on consumer-class GPUs. See [`runs.md`](runs.md#post-release-scaled-up-b200-configuration) for the pending run entry.

### Semantic embedding

An optional replacement of the learned token embedding with a **semantic embedding**, where each dimension is a plain-language description or condition, and each token (or n-gram) is expressed as a vector of values across those dimensions.

**Why WaveletLM is structurally well-suited to this:** the spectral mixer operates directly on human-readable features instead of learned token similarity in the style of attention. Each semantic concept's temporal signal is decomposed at multiple scales, letting interpretable concepts at the input be processed at different temporal granularities.

**Expected impact:** improved interpretability at the embedding and low-layer level at a small cost to single-token performance. Extending the scheme to n-gram tokens, and deliberately choosing a set of dimensions for the semantic embedding which maximizes performance versus other such sets while retaining generality across datasets, may allow the model to match or exceed baseline learned embedding performance while retaining the interpretability advantage.

See [plans/reincorporate_large_semantic_embedding.md](plans/reincorporate_large_semantic_embedding.md) for more information.

**Example plain-language features:**

- Is this token a noun?
- Is this token a person?
- Is this token associated with anger?
- Does this token have 3, 4, or 5 syllables?
- Does this token contain more than one word?
- What is the frequency with which this token is used in deceptive contexts?

Note that these are simply examples which may or may not be useful. The method by which per-token coefficients are assigned - one-hot/binary, LLM-scored, human-rated, or corpus-derived - is itself an open design choice, each with its own interpretability, quality, and monetary tradeoffs.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## References

[^2]: Dai et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." arXiv:1901.02860, 2019.
[^3]: Radford et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019.
[^4]: Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces." arXiv:2111.00396, 2021.
