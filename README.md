<p align="center">
  <img src="assets/waveletlm-header.svg" alt="WaveletLM" width="90%"/>
</p>

<br>

WaveletLM is a wavelet-based, attention-free language model that replaces attention with spectral mixing. Each block processes the input sequence using learned lifting wavelet decomposition, a Fast Walsh-Hadamard Transform, per-scale gated spectral mixing with SwiGLU activation, inverse Fast Walsh-Hadamard Transform, and wavelet reconstruction. Combined with expanded MLPs and a cross-layer decompose bypass, this produces a fully causal sequence language model with no attention mechanism, no quadratic scaling, and no key/value cache.

<br>

<p align="center">
<a href="#installation">Installation</a><br>
<a href="#training">Training</a><br>
<a href="#generation">Generation</a><br>
<a href="#architecture">Architecture</a><br>
<a href="#results">Results</a><br>
<a href="#post-release-plans">Post-Release Plans</a><br>
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

Weights may be obtained [here](huggingface.co). Replace `best_model.pt` below with the appropriate path/name.

```bash
# Recommended generation command
python generate.py --checkpoint best_model.pt --strategies \
    --prompt "Your prompt here"

# Default generation
python generate.py --checkpoint best_model.pt

# Additional options
python generate.py --checkpoint best_model.pt \
    --prompt "Put a prompt here." --num_tokens 1024 --seed 1337 --n 1 \
    --temperature 1.0 --strategies --ptq8
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

### Post-training quantization (PTQ; optional)

Near-lossless uniform 8-bit PTQ — recommended when VRAM or checkpoint size matters.

```bash
python generate.py --checkpoint best_model.pt --ptq8
```

Effects:

+0.0001 BPB (negligible), -10% inference VRAM, -50% checkpoint size, & -12% tok/s until bit-packed kernels land (see [`runs.md`](runs.md#ptq-sweep-summary)).


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

- **Fast Walsh-Hadamard Transform (FHT)**: a fixed orthogonal O(C log C) cross-channel rotation that replaces attention's channel-mixing role. Cost is independent of sequence length: no quadratic blow-up regardless of context size, and no KV cache at inference.

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

### WikiText-103 Perplexity Comparison

| Model | Type | Trained on | Params | PPL |
|-------|------|-----------|--------|-----|
| GPT-2 XL | Transformer | WebText (40GB) | 1.5B | 17.5[^3] |
| Transformer-XL Large* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 257M* | 18.3[^2]* |
| GPT-2 Large | Transformer | WebText (40GB) | 774M | 19.3[^3] |
| S4* | SSM* | WikiText-103 (0.5GB)* | 130M* | 20.9[^4]* |
| GPT-2 Medium | Transformer | WebText (40GB) | 355M | 22.1[^3] |
| Transformer-XL Standard* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 151M* | 24.0[^2]* |
| **WaveletLM** | **Wavelet mixer** | **WikiText-103 (0.5GB)** | **883M** | **24.2**[^1] |
| GPT-2 | Transformer | WebText (40GB) | 124M | 29.4[^3] |

\* Trained and evaluated on the same dataset (direct comparison to WaveletLM).

WaveletLM achieves this with only 2 layers (L=2, C=2048), no attention and no KV cache. Comparison numbers are sourced from respective papers; see references below.

### Areas for improvement

Longer training time, more regularization, and parameter compression are the surest ways to immediately improve the model's performance. We invite others to tackle each of these tasks in turn:

**More training time**: Validation loss was still improving at epoch 5, indicating further training will improve results. My budget is limited in the number of runs I can perform. More research and more resources are needed to uncover the effects of longer training.

**Regularization**: Greater than 0.8 train/val loss gaps at 5+ epochs show WaveletLM is vastly underregularized. Perplexity is expected to drop further with increased dropout and well-chosen weight decay, but such parameter sweeps are also limited by budget. There are 5 dropout parameters to adjust: `dropout_embedding`, `dropout_projection`, `dropout_mixer`, `dropout_mlp`, and `dropout_lm_head`.

**Parameter compression**: The raw parameter count matters, but so does what those parameters are doing. Of WaveletLM's 883M total, around 55% (488M) live in two highly-compressible components: dense MLPs (335.6M) and product-key memory modules (PKM: 76M + FwPKM: 76M). The architecturally distinctive components like lifting wavelets (83.9M), the spectral mixer (88.2M), and decompose EMA bypass (8.4M), are small in comparison. Further work is needed to determine the degree of compressivity of high-parameter regions.

[^1]: See [training log](logs/wikitext-103_2026-04-19_13-16-24/log.txt) and [benchmark.txt](logs/wikitext-103_2026-04-19_13-16-24/benchmark.txt).

See [`runs.md`](runs.md) for a full log of training runs, configs, and benchmark results.


## Future Plans

### Dataset comparisons

The best WaveletLM config trained on PG-19, Pile-ArXiv, BookCorpusOpen, and/or OpenWebText to gauge performance on more data.

### Model comparisons

Side-by-side benchmarks against Transformer, Mamba, RWKV, and other modern architectures on WikiText-103 at matched compute and fully optimized.

### Scaled-up model (B200)

The current headline model (882.51M params: L=2, C=2048, MLP=20, PKM/FwPKM=16384) was trained on a single RTX 5090 due to budget constraints. A B200 (192 GB HBM3e) unlocks roughly an order of magnitude more parameter budget at training time and makes several scaling levers practical to stack simultaneously:

- **Width (C):** 2048 → 4096 (mixer working width)
- **Depth (L):** 2 → 4–8 (WaveletLM blocks)
- **MLP expansion:** 20 → 50–200 (primary knowledge-storage lever; monotonic BPB contributor in ablations)
- **PKM / FwPKM keys:** 16384 → 65536 (4× sparse memory capacity)
- **Training precision:** fp16 → **FP8** (E4M3/E5M2 via Blackwell native tensor cores), for ~2× training throughput and ~30–40% memory reduction. Requires Transformer Engine (or torchao) plus per-tensor dynamic scaling to handle FP8's narrow range; bf16 was previously tried as a wider-range alternative and regressed, so the stability recipe is non-trivial.

The target is a ~10–15B parameter configuration chosen after the 5090 sweep completes. Two training targets are planned:

- **WikiText-103 only**, for an apples-to-apples comparison against the 5090 headline run and against prior same-dataset baselines (Transformer-XL, S4).
- **Multi-dataset training** across a broader mix (PG-19, Pile-ArXiv, BookCorpusOpen, TinyStories, OpenWebText, and WikiText-103), establishing WaveletLM's behavior as a general-purpose language model across domains rather than a single-benchmark result.

fp16 inference should fit a single RTX 4090 (24 GB); Post-training quantization (PTQ with per-scale mixed precision, 8/4/2-bit) is expected to drop inference VRAM below 8 GB, enabling deployment on consumer-class GPUs. See [`runs.md`](runs.md#post-release-scaled-up-b200-configuration) for the pending run entry.

### Semantic embedding

An optional replacement of the learned token embedding to be developed soon is a **semantic embedding**, where each dimension is a plain-language description or condition, and each token (or n-gram) is expressed as a vector of values across those dimensions.

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
