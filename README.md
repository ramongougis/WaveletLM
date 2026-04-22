<p align="center">
  <img src="assets/waveletlm-header.svg" alt="WaveletLM" width="90%"/>
</p>

<br>

WaveletLM is a wavelet-based, attention-free language model that replaces attention with spectral mixing. Each block uses learned lifting wavelet decomposition, a Fast Walsh-Hadamard Transform, per-scale gated spectral mixing with SwiGLU activation, inverse FWHT, and wavelet reconstruction. Combined with expanded MLPs and a cross-layer decompose bypass, this produces a fully causal sequence language model with no attention mechanism and sub-quadratic scaling in sequence length.

<br>

<p align="center">
<a href="#installation">Installation</a><br>
<a href="#training">Training</a><br>
<a href="#generation">Generation</a><br>
<a href="#architecture">Architecture</a><br>
<a href="#results">Results</a><br>
<a href="#future-plans">Future Plans</a><br>
<a href="#license">License</a><br>
<a href="#references">References</a>
</p>

<br>

## Installation

Requires Python 3.10+, PyTorch 2.8+, and CUDA.

```bash
git clone https://github.com/ramongougis/WaveletLM.git
cd WaveletLM
pip install torch datasets tiktoken tqdm numpy
```

## Training

Run:

```bash
python train.py
```

`config.json` replicates the current best [880M parameter WikiText-103 run](logs\wikitext-103_2026-04-22_01-36-47\log.txt), which requires 18235 MiB to train.

Key config options:

| Option | Default | Description |
|--------|---------|-------------|
| `C` | 2048 | Mixer channel width (power of 2 recommended) |
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

Obtain weights from [HuggingFace](huggingface.co), then replace `best_model.pt` in the commands below with the path to the file. 

The [current best 880M parameter model](logs\wikitext-103_2026-04-22_01-36-47\log.txt) requires 4918 MiB for inference and generates at 28.8 tokens/s on a 5090.

```bash
# Recommended generation command
python generate.py --checkpoint best_model.pt --strategies \
    --prompt "Your prompt here"

# Default generation
python generate.py --checkpoint best_model.pt

# Additional options
python generate.py --checkpoint best_model.pt \
    --prompt "Your prompt goes here." --num_tokens 1024 --seed 1337 \
    --n 1 --temperature 1.0 --strategies --ptq8 --num_tokens 9000
```

### Inference Strategies:

Can run all strategies together with `--strategies` or individual ones. Use `--help` for a complete list.

```bash
# Use all inference strategies
python generate.py --checkpoint best_model.pt --strategies

# Some strategies options
python generate.py --checkpoint best_model.pt --entropy_adaptive \
    --lookahead_k 3 --lookahead_depth 5 --best_of_n 5 --clean_spacing \
    --wavelet_coherence
```

### Post-Training Quantization (PTQ; optional)

Near-lossless uniform 8-bit PTQ — recommended when VRAM or checkpoint size matters.

```bash
python generate.py --checkpoint best_model.pt --ptq8
```

PTQ effects:

- +0.0001 BPB hit (negligible performance impact)
- 10% less inference VRAM
- 50% less checkpoint file size
- 12% less tok/s currently. However, PTQ is expected to be 1.4-2.2x faster than the baseline 28.8 tok/s with bit-packed kernels. See [Future Plans → Bit-Packed PTQ Kernels](#bit-packed-ptq-kernels) and [`runs.md`](runs.md#post-release-bit-packed-ptq-kernels)).


## Architecture

```
Input tokens
    |
    v
Learned Embedding (C)
    |
    v
+---------------------------------+
|  WaveletLM Block (x N layers)   |
|                                 |
|  LayerNorm                      |
|  Lifting Wavelet Decompose      |
|  Fast Walsh-Hadamard Transform  |
|  Gated Spectral Mixer           |
|  Fast Walsh-Hadamard Inverse    |
|  Lifting Wavelet Reconstruct    |
|  Learned Residual 1             |
|  LayerNorm                      |
|  Feedforward Layers (MLP)       |
|  Learned Residual 2             |
+---------------------------------+
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

- **Fast Walsh-Hadamard Transform (FHT)**: a fixed orthogonal O(C log C) cross-channel rotation that replaces attention's channel-mixing role. Cost is independent of sequence length: no quadratic blow-up regardless of context size.

- **Per-scale gated spectral mixer (SwiGLU)**: mixes each wavelet scale independently in Walsh-Hadamard space through a gated linear layer. Captures interactions within each frequency band without forcing cross-band mixing, and runs in fixed O(S²) cost per layer for S wavelet scales regardless of context size - versus attention's O(N²) in sequence length.

- **Expanded MLP (expansion ≥ 20)**: the model's primary knowledge-storage mechanism, scaled well beyond the ~4× typical in Transformers. MLP expansion is a monotonic contributor to BPB in our ablations, taking on much of the memorization role that attention plays in Transformer architectures.

- **Decompose bypass**: a causal cumulative mean of pre-decompose hidden states, projected per-scale and added as bias to the post-decompose wavelet coefficients. Provides cross-layer global context at O(C) cost per token, with no attention required.

<details>
<summary><b>Additional key components</b> (always-on architectural pieces)</summary>

- LayerNorms near both ends of each block, plus one final LayerNorm before the LM head
- Two residual connections per block with learned scalar gating (`learned_residual` in config.json)
- Learned per-scale weights applied after the inverse FHT - one trainable scalar per wavelet scale
- Feature padding to the next power of 2, required for the Walsh-Hadamard transform (`C` → `Cp = next_pow2(C)`)
- Causal zero-padded dilation in the lifting predict/update steps, preserving autoregressive causality at every level

</details>

### Optional Features

- **Per-Layer Embedding**: adds a learned per-channel residual of the original token embedding at each block, letting deeper blocks reach back to the input representation when relevant.

- **Product Key Memory / Fast-Weight Product Key Memory**: sparse key-value memory modules that complement the dense MLP, providing parameter-efficient long-tail pattern storage with optional inference-time fast-weight updates.

- **Low-Rank Factorization**: adds a rank-r perturbation `U·V^T` to the spectral mixer, expanding mixing expressivity at trivial parameter cost (rank=4 yields a measurable BPB improvement).

- **Exponential Parametrization**: reparameterizes mixer weights through `exp()`, stabilizing training under high learning rates that would otherwise NaN.

- **Cross-scale gating (routing mode)**: a learned (S, S) routing matrix that mixes per-scale inputs before each gate, enabling conditional cross-scale interactions (e.g., "when scale 0 shows pattern X, modulate scale 4's processing"). Initialized to identity so it begins as a no-op and only contributes what it learns.

- **Per-scale mixer widths**: asymmetric per-scale mixer capacity. Coarse scales keep full mixer width while fine scales use reduced width via in/out projections. At widths `[1, 1, 1, 0.5, 0.5, 0.5]`, yields a small BPB improvement and a ~23% per-epoch speedup.

- **Wavelet crawl**: learned soft-mixed dilations per wavelet level. Instead of fixed `2^l`, each level sees a softmax-weighted mixture of K candidate offsets around the base dilation, letting the model discover slightly off-power-of-2 receptive fields. K=3 (±1 search radius) is the stable sweet spot.

- **Shared lifting weights**: a single lifting wavelet module shared across all blocks instead of per-block weights. Essentially free on BPB while cutting training VRAM by the weight of L−1 lifting modules (~5–10% at L=2).

- **Looped blocks (Universal Transformer-style)**: apply one shared block K times in place of stacking L distinct blocks. Achieves BPB reduction at fixed parameter count by trading parameters for compute, though the same compute is usually better spent on additional training epochs of the stacked model.

- **Data-dependent EMA bypass** (`decompose_bypass_ema`): replaces the cumulative-mean decompose-bypass with a σ-gated EMA: a leaky integrator whose leak rate is decided per-token by the current input. DSP-wise: an adaptive (Kalman/Wiener-flavor) 1st-order IIR filter whose cutoff frequency changes with input, letting the model selectively forget at topical/clausal boundaries. 

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
- Linear-only lifting networks - no GELU (`lifting_linear_only`)
- Stacked spectral mixer depth (`mixer_depth`, `mixer_depth_stabilizers`, `mixer_depth_residuals`)
- LoopLM mode - full-stack iterated inference (`loop_iterations`)
- Weight tying between embedding and LM head (`tie_embedding_to_lm_head`)
- Output-projection skip when C equals Cp (`skip_proj_out`)
- Gradient checkpointing (`gradient_checkpointing`)
- Stochastic depth (`stochastic_depth_rate`)
- Per-component dropouts (`dropout_embedding`, `dropout_projection`, `dropout_mixer`, `dropout_mlp`, `dropout_lm_head`)
- Lifting-network hidden-dim multiplier (`lifting_hidden_mult`)
- Lifting initialization choice - Haar / zero / random (`lifting_init`)
- Lifting dropout (`lifting_dropout`)
- Spectral mixer gate toggle and activation (`use_mixer_gate`, `mixer_gate_activation`)
- Non-learned fixed-Haar fallback for the wavelet (`wavelet_mode="haar"`)
- Multinodal feature bagging mode and its sub-flags (`multinodal_enabled`, `multinodal_num_cells`, `multinodal_cell_dim`, `multinodal_seeds`, `multinodal_combination`, `multinodal_cross_cell_gating`, `multinodal_features_per_cell`, `multinodal_bagged_eps`)

</details>

### Multinodal Mode

WaveletLM supports a product-of-experts mode where multiple independent model nodes process the input in parallel with different feature subsets (feature bagging), then combine logits via averaging. Enable with `multinodal_enabled: true` in the config. This mode may require additional stability improvements such as a lower learning rate and `stable_parametrization: true`, and acts as an as-yet underexplored capacity/scalability lever.


## Results

### WikiText-103 Test Set Perplexity Comparison

| Model | Type | Trained on | Params | PPL |
|-------|------|-----------|--------|-----|
| GPT-2 XL | Transformer | WebText (40GB) | 1.5B | 17.5[^3] |
| Transformer-XL Large* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 257M* | 18.3[^2]* |
| GPT-2 Large | Transformer | WebText (40GB) | 774M | 19.3[^3] |
| S4* | SSM* | WikiText-103 (0.5GB)* | 130M* | 20.9[^4]* |
| GPT-2 Medium | Transformer | WebText (40GB) | 355M | 22.1[^3] |
| **WaveletLM** | **Wavelet mixer** | **WikiText-103 (0.5GB)** | **883M** | **23.7** |
| Transformer-XL Standard* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 151M* | 24.0[^2]* |
| GPT-2 | Transformer | WebText (40GB) | 124M | 29.4[^3] |

\* Both trained and evaluated on WikiText-103 only (direct comparison to WaveletLM). GPT-2 BPE was used by WaveletLM for tokenization.

See [training log](logs/wikitext-103_2026-04-22_01-36-47/log.txt) and [benchmark.txt](logs/wikitext-103_2026-04-22_01-36-47/benchmark.txt) for the results (5 epochs, 2.0× dropout, weight decay 1e-6).

See [`runs.md`](runs.md) for a full log of training runs, configs, and benchmark results.

### PG-19 Test Set Perplexity Comparison

| Model | Type | Params | PPL |
|-------|------|--------|-----|
| Perceiver AR | Cross-attn + latents | 974M | 28.9[^5] |
| Block-Recurrent Transformer | Transformer + recurrence | ~200M | 29.0[^6] |
| Compressive Transformer | Transformer + compressive memory | 257M | 33.6[^7] |
| Transformer-XL | Transformer + recurrence | 257M | 36.3[^7] |
| **WaveletLM** | **Wavelet mixer** | **883M** | **TBD** (pending [pre-release run](runs.md#pg-19-pre-release-benchmark-best-seed-1-epoch)) |

All models in this table were trained and evaluated on PG-19 with its standard SentencePiece tokenization.

Comparison numbers for both datasets are sourced from their respective papers. See References below.

### Areas for Improvement

Longer training time, more regularization, and parameter compression are the surest ways to immediately improve the model's performance. We invite others to tackle each of these tasks in turn:

**More training time**: Validation loss was still improving at epoch 5, indicating further training will improve results. My budget is limited in the number of runs I can perform. More research and more resources are needed to uncover the effects of longer training.

**Regularization**: Greater than 0.8 train/val loss gaps at 5+ epochs show WaveletLM is vastly underregularized. Perplexity is expected to drop further with increased dropout and weight decay, but such parameter sweeps are also limited by budget. There are 5 dropout parameters to adjust alongside `weight_decay`: `dropout_embedding`, `dropout_projection`, `dropout_mixer`, `dropout_mlp`, and `dropout_lm_head`.

**Parameter compression**: The raw parameter count matters, but so does what those parameters are doing. Of WaveletLM's 883M total, around 55% (488M) live in two highly-compressible components: dense MLPs (335.6M) and product-key memory modules (PKM: 76M + FwPKM: 76M). The architecturally distinctive components like lifting wavelets (83.9M), the spectral mixer (88.2M), and decompose EMA bypass (8.4M), are small in comparison. Further work is needed to determine the degree of compressivity of high-parameter regions.


## Future Plans

### Dataset Comparisons

The best WaveletLM config trained on Pile-ArXiv, BookCorpusOpen, OpenWebText, and other datasets to gauge their performance.

### Model Comparisons

Side-by-side benchmarks against Transformer, Mamba, RWKV, and other modern architectures on WikiText-103 at matched compute and fully optimized.

### Scaled-Up Model (B200)

The 882M RTX 5090 headline run scales up naturally to a B200:

- `C`: 2048 → 4096 
- `layers`: 2 → 4–8
- `mlp_expansion`: 20 → 50–200
- `pkm_num_keys` & `fwpkm_num_keys`: 16384 → 65536 each
- fp16 → FP8 via Blackwell tensor cores (NYI)

Target is a 10–15B parameter configuration, trained individually on WikiText-103 and PG-19, and also on a multi-dataset mix of WikiText-103, PG-19, Pile-ArXiv, BookCorpusOpen, TinyStories, & OpenWebText. 

Inference would fit on a single RTX 4090 at fp16 and roughly half the VRAM with [uniform 8-bit PTQ](runs.md#ptq-sweep-summary). See [`runs.md`](runs.md#post-release-scaled-up-b200-configuration) for the pending run entry.

### Bit-Packed PTQ Kernels

The [current PTQ path](runs.md#ptq-sweep-summary) dequantizes int8 weights to fp16 inside `forward()` and runs a standard fp16 matmul, which pays the dequant cost every step with no bandwidth win - hence the 12% generation slowdown and the fact that sub-8-bit variants compress identically to 8-bit on disk. 

Swapping `QuantizedLinear` / `QuantizedEmbedding` for fused packed-weight kernels (Marlin W8A16 / W4A16, CUTLASS `i8gemm`, bitsandbytes, Triton for the embedding lookup) fixes both: storage scales with bit-width, and each matmul reads half or a quarter as many bytes. Expected generation at batch=1 (fp16 baseline 28.8 tok/s) is **~1.4–1.6× faster** for fused uniform 8-bit and **~1.8–2.2× faster** for fused mixed 8/4/2, with BPB unchanged. See [runs.md](runs.md#post-release-bit-packed-ptq-kernels) for the full plan.

### Semantic Embedding & Interpretability Work

An optional replacement for the learned token embedding is a **semantic embedding**, where each dimension is a plain-language feature (e.g. "is this token a noun?", "is this token associated with anger?", "corpus frequency in deceptive contexts") and each token or n-gram is a vector of values across those dimensions. 

WaveletLM is structurally well-suited for this: the spectral mixer can operate directly on human-readable features, and multi-scale decomposition lets the same concept be processed at different temporal granularities. Expected tradeoff is improved interpretability at a small performance cost, potentially recovered or even improved via n-gram tokens and careful feature selection for the dimensions. 

See [plans/reincorporate_large_semantic_embedding.md](plans/reincorporate_large_semantic_embedding.md) for the full design, including open questions on coefficient assignment methods: one-hot/binary, LLM-scored, human-rated, or corpus-derived.


## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## References

[^2]: Dai et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." arXiv:1901.02860, 2019.
[^3]: Radford et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019.
[^4]: Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces." arXiv:2111.00396, 2021.
[^5]: Hawthorne et al. "General-purpose, long-context autoregressive modeling with Perceiver AR." arXiv:2202.07765, 2022.
[^6]: Hutchins et al. "Block-Recurrent Transformers." arXiv:2203.07852, 2022.
[^7]: Rae et al. "Compressive Transformers for Long-Range Sequence Modelling." arXiv:1911.05507, 2019. (PG-19 dataset introduction; reports both Compressive Transformer and Transformer-XL on PG-19.)
