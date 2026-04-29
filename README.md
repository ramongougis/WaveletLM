<p align="center">
  <img src="assets/waveletlm-header.svg" alt="WaveletLM" width="90%"/>
</p>

<p align="center">
  <img src="assets/divider.svg" alt="" width="85%" height="1"/>
</p>

<br>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://github.com/ramongougis/WaveletLM/actions/workflows/codeql.yml"><img src="https://github.com/ramongougis/WaveletLM/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"/></a>
  <a href="https://github.com/ramongougis/WaveletLM/actions/workflows/scorecard.yml"><img src="https://github.com/ramongougis/WaveletLM/actions/workflows/scorecard.yml/badge.svg" alt="Scorecard supply-chain security"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python"></a>
</p>

<br>
WaveletLM is a wavelet-based, attention-free language model that mixes tokens through learned lifting wavelet decomposition, a Fast Walsh-Hadamard Transform, per-scale gated spectral mixing with SwiGLU activation, an inverse FWHT, and wavelet reconstruction. Combined with expanded MLPs and sparse product-key memory, this yields an architecture with no attention and O(n log n) scaling in sequence length.

<br>

<p align="center">
<a href="#installation">Installation</a><br>
<a href="#training">Training</a><br>
<a href="#inference">Inference</a><br>
<a href="#sample-generations">Sample Generations</a></br>
<a href="#architecture">Architecture</a><br>
<a href="#results">Results</a><br>
<a href="#future-plans">Future Plans</a><br>
<a href="#license">License</a><br>
<a href="#references">References</a>
</p>


## Installation

Requires Python 3.10+, PyTorch 2.8+, and CUDA.

```bash
git clone https://github.com/ramongougis/WaveletLM.git
cd WaveletLM
pip install torch "datasets<3.0" tiktoken sentencepiece tqdm numpy
```

## Training

Run:

```bash
python train.py
```

`config.json` replicates the current best [883M parameter WikiText-103 run](logs/wikitext-103_2026-04-22_01-36-47/log.txt), which requires 18,235 MiB to train.

Key config options:

| Option | Default | Description |
|--------|---------|-------------|
| `C` | 2048 | Mixer channel width (power of 2 recommended) |
| `layers` | 2 | Number of WaveletLM blocks |
| `levels` | 5 | Wavelet decomposition levels (~log2(block_size)) |
| `mlp_expansion` | 20 | Hidden layer width multiplier |
| `block_size` | 256 | Context length |
| `epochs` | 5 | Training epochs |
| `micro_batch_size` | 8 | Per-step batch size (scale to fit VRAM) |
| `grad_accum` | 1 | Gradient accumulation; effective batch = `micro_batch_size × grad_accum` |
| `lr` | 0.01 | Peak learning rate (Adagrad-tuned; reduce substantially if switching to AdamW) |
| `weight_decay` | 1e-6 | L2 weight decay (1e-6 mildly beneficial; 1e-3 stalls training) |
| `warmup_fraction` | 0.3 | Fraction of total steps spent in LR warmup |
| `dropout_lm_head` | 0.24 | LM-head dropout (other heads: `dropout_mlp` / `mixer` / `projection` / `embedding`) |
| `dataset` | wikitext-103 | HuggingFace dataset ID |
| `seed` | 1337 | RNG seed (e.g. for variance studies) |
| `compile` | True | Enable `torch.compile`; disable when debugging |

Training logs, checkpoints, and configs are saved to `logs/<dataset>_<timestamp>/`. Results from all runs are tracked in [`runs.md`](runs.md). The full default run takes ~14h on an RTX 5090; drop `epochs` to 1 for a quick smoke test.


## Inference

Obtain weights from [HuggingFace](https://huggingface.co/anarmorarm/WaveletLM/tree/main), then use `best_model_pg-19.pt`, `best_model_wikitext-103.pt`, or whichever weights file is desired as the checkpoint in the commands below.

The [current best PG-19 model](logs\pg19_2026-04-25_13-34-46\benchmark.txt) requires 4,544 MiB for inference and generates at 28.3 tokens/s on a 5090.

The [current best WikiText-103 model](logs/wikitext-103_2026-04-22_01-36-47/log.txt) requires 4,918 MiB for inference and generates at 28.8 tokens/s on a 5090.

Recommended commands:

```bash
# PG-19
python generate.py --checkpoint best_model_pg-19.pt --strategies /
    --prompt "Your prompt here"

# WikiText-103
python generate.py --checkpoint best_model_wikitext-103.pt --strategies /
    --prompt "Your prompt here"
```

Default generation:

```bash
python generate.py --checkpoint best_model_pg-19.pt
```

Additional options:

```bash
python generate.py --checkpoint best_model_pg-19.pt /
    --prompt "Your prompt goes here." --num_tokens 1024 --seed 1337 /
    --n 1 --temperature 1.0 --strategies --ptq8 --num_tokens 9000
```

### Inference Strategies:

Can run all strategies together with `--strategies` or individual ones. Use `--help` for a complete list.

Use all inference strategies:

```bash
python generate.py --checkpoint best_model_pg-19.pt --strategies
```

Some strategies options:

```bash
python generate.py --checkpoint best_model_pg-19.pt --entropy_adaptive /
    --lookahead_k 3 --lookahead_depth 5 --best_of_n 5 --clean_spacing /
    --wavelet_coherence
```

The `--strategies` flag enables:

- Entropy-adaptive temperature (capped at 0.9)
- `top_p=0.85`
- `repetition_penalty=1.2`
- Metrics logging (`mean_log_prob`, `Distinct-n`, `Rep-4`)
- WikiText-103 spacing cleanup

### Post-Training Quantization (PTQ; optional)

Near-lossless uniform 8-bit PTQ:

```bash
python generate.py --checkpoint best_model_pg-19.pt --ptq8
```

PTQ effects:

- +0.0001 BPB hit (negligible performance impact)
- 10% less inference VRAM
- 50% less checkpoint file size
- 12% less tok/s currently. However, PTQ is expected to be 1.4-2.2x faster than the baseline 28.8 tok/s with bit-packed kernels. See [Future Plans → Bit-Packed PTQ Kernels](#bit-packed-ptq-kernels) and [`runs.md`](runs.md#post-release-bit-packed-ptq-kernels)).


## Sample Generations

### PG-19

Selections from the [PG-19 best run's generations log](https://github.com/ramongougis/WaveletLM/blob/main/logs/pg19_2026-04-25_13-34-46/generations.txt):

**Sample A** with prompt "The history of":

> The history of the world is a long and painful one. It has been so 
> much neglected, that it may be said to have become known through all 
> ages; but its influence is not confined solely to the individual or 
> class, nor even to those who are in any way connected with it. The 
> first thing which strikes us on entering into this subject is, that 
> there is no other mode by which we can bring ourselves to do justice 
> to our fellow-men; for what is more extraordinary than that they 
> should be able to find out the real character of those whom we are 
> about to see? If such were their intentions, as I have before stated, 
> then it would be very difficult indeed for them to keep themselves 
> alive at the present moment; because, if we take away from them 
> anything that makes life pleasant, or comfortable, or agreeable, they 
> must give up all thoughts of it.

<details>
<summary><h3 style="display:inline">More PG-19 samples</h3></summary>

**Sample B** with prompt "She turned to him and said":

> She turned to him and said, "I have been thinking a great deal about 
> you. I know that your father is a very good man; but he would not give 
> me his consent if I were to marry him. But it will be all the same with 
> me." He looked at her as though she had spoken to him out of a book 
> which was lying on the table. When he spoke again there came into her 
> face the expression of a most intense earnestness--a look of deep 
> emotion. It seemed to him that she felt for one moment as if she could 
> read his soul through this wonderful eyes: they were full of tenderness 
> and affection, and she knew that in them lay love's highest hope and 
> joy. Then she rose and walked slowly back to the house, leaving him 
> alone with the dead.
>
> CHAPTER XXXV. A CONFESSION OF ROMANCE.

**Sample C** with prompt "'Beware the High Elf mage,' exclaimed Antimour...":

> 'Beware the High Elf mage,' exclaimed Antimour. 'Although a novice, I 
> sense much arcane affinity within him.' Then he added: 'You have been 
> taught to read and write; but you are now in the seventh heaven of 
> bliss!' 'I am not so old as that, my lord,' replied the astrologer; 
> 'but there is no reason why I should be happy without you.' The 
> astrologer shook his head doubtfully. 'There is nothing for it, then,' 
> said he; 'for the present at least we will talk about the matter. 
> Meanwhile I must tell thee how thou didst come into this world. In what 
> way didst thou escape from thine enemies?' 'By means of some magic 
> wand which I possessed,' answered the magician.

Since PG-19 contains data from 1919 and earlier, this prompt is out of distribution. It uses post-Tolkien fantasy diction which the model routes to the closest pre-1919 register: Edwardian fairy-tale and Arabian Nights prose.

**Sample D** with prompt "The spaceship finally arrived in the Alpha Centauri system":

> The spaceship finally arrived in the Alpha Centauri system. It was 
> found that this would be very much like a double-acting motor, but it 
> was not to be compared with one of these machines which had been 
> invented by Professor Thomson and constructed for him by himself at 
> his own expense. In order to make use of such a machine he must have 
> an efficient engine capable of performing its work as soon as possible.
>
> [Illustration: FIGS. 1, 2, 3 and 4.--DIAGRAM OF ELECTRICAL DICTIONARY 
> INVENTION. A simple form of the principle shown is shown on the 
> accompanying diagram (Fig. 7), which represents a rectangular piece of 
> wire attached to the base of a cylinder.

A second out-of-distribution prompt. The model has no training data on spaceships or interstellar travel, so it routes these to the closest pre-1919 register: a Victorian mechanical-engineering treatise with a figure caption.

**Sample E** with prompt "The history of" — naive sampling (no `--strategies`, default `temp=1.0`, and `top_p=0.95`):

> The history of her interest in the modern world during this quarter, 
> has been given to the journal in which she appeared during her stay; 
> and I could mention no names more memorable than those connected with 
> the several sessions at this distinguished lady's house. In 1856, Mr. 
> Ramsey began the voyage along one of the many channels by which Boston 
> had heretofore eluded him, with very valuable additions to his papers.
>
> On January 30, 1833, a small packet arrived at Charleston from Jamaica, 
> designed for the benefit of George (afterwards Lord) Wilmot, an officer 
> in the American army--a man well qualified to fulfill all the duties 
> of a seaman as before stated. He remained at Point Cotterand until 
> October, 1827, when he died in his lodgings, without either pecuniary 
> credit or pecuniary obligations. At the close of 1805 Mrs. Fry visited 
> us. She had previously known her husband, who held an appointment 
> within a few miles of Camden Town.

Randomly selected naive (without `--strategies`) sample out of 20. Naive samples retain more of the model's raw distribution and tend to mix domains more loosely.

</details>

### WikiText-103

Selections from the [WikiText-103 best run's generations log](https://github.com/ramongougis/WaveletLM/blob/main/logs/wikitext-103_2026-04-22_01-36-47/generations.txt):

**Sample A** with prompt "The history of":

> The history of the city is reflected in its architecture, which includes 
> the historic Old Town and New Castle County Courthouse Square Historic 
> District. The building was designed by John H. Stevens, who also 
> designed the Albany-Fulton Celebration in 1906 and built a steel-hulled 
> shipyard on the lake shore.
>
> In 1909, Fort Wayne became home to the first naval base for military 
> aviation in the United States. The Navy Yard at the former Naval Air 
> Station Pensacola (now known as the Georgia Tech Research Institute), 
> located on the campus' eastern boundary, was constructed in 1925–1926 
> near the site of the original schoolhouse.

<details>
<summary><h3 style="display:inline">More WikiText-103 samples</h3></summary>

**Sample B** with prompt "The album was released":

> The album was released on August 25, 2007 by Sony Music Entertainment 
> and features several songs from the record including "Never Say Die", 
> "The Show", "Don't Cry for Me Argentina" and a cover of "I Can Only 
> Imagine (But You Are Not Alone)".
>
> In 2009, Tyler performed at the Rock in Rio festival held in Lisbon, 
> Portugal; a performance which included performances of his own 
> compositions by Queen Latifah and Ali Zardari. On July 1, 2010 he sang 
> "O Holy Night" during the closing ceremony of the 2012 Summer Olympics 
> opening ceremony at London Olympic Stadium as part of the celebrations 
> of the 75th anniversary of the United States Declaration of Independence.

**Sample C** with prompt "The species was first described":

> The species was first described by Swedish zoologist Carl Linnaeus 
> in 1758 as Agaricus adustus. The genus name is derived from the Latin 
> words perma "to tie", and pous ("like") means "with a large head". In 
> 1821, French mycologists Jean-Baptiste de Lacaille placed it in section 
> Cricetae of the order Carnivora. He later renamed it Spongiforma 
> punctata after the Greek kribensis.
>
> P. nigriceps was originally classified under its current binomial name 
> Sirmuellera speciosa; however, this has been rejected on the grounds 
> that Lactarius deterrimus may not be closely related to Boletus edulis 
> or other similar fungi in general. 

**Sample D** with prompt "Born in":

> Born in the city, and the people who have been living there. The town's 
> only synagogue is located on a hill overlooking the street; it was 
> built by John D. Rockefeller, Jr.
>
> A museum dedicated to the history of Jewish culture in Israel opened 
> in 2008 at the former Beth Elohim building. It features an original 
> painting from the Al-Fakhri Mosque that hangs in the Temple. This is 
> one of four mosques in the district named after Abraham Lincoln.
>
> Hebron Church (Hebrew: בית לחברג 'ה) is believed to be the oldest part 
> of Jerusalem's Old City. The name means "holy place" or "place of 
> prayer" in Islamic law and Arabic texts dating back to prehistory as 
> early as the 6th century BCE.

**Sample E** with prompt "The history of" and naive sampling (no `--strategies`, default `temp=1.0`, and `top_p=0.95`):

> The history of the tropical cyclone is unknown, but official records 
> suggest it formed on May 31. It developed into a tropical storm later 
> that day. After turning to the northeast, on September 20 Huron 
> strengthened into a hurricane while east-southeast of Bermuda. During 
> that time, Hurricane Humberto destroyed another ship in a prolonged 
> action. In late October, Ione crossed into the eastern Gulf of Mexico; 
> however, it reintensified slightly and later peaked with winds of 100 
> mph (160 km/h) as it drifted through western Cuba.

Note the typical failure mode with the naive generation: register-coherent meteorological prose, but the model freely interleaves the names of multiple unrelated real storms (Huron, Humberto, Ione) within a single passage. Without `--strategies`, the model is prevented from employing a more conservative sampling regime.

</details>


## Architecture

<p align="center">
  <img src="assets/waveletlm-architecture.svg" alt="WaveletLM architecture" width="80%"/>
</p>

### Key Components

- **Learnable lifting wavelet decomposition**: Haar-initialized predict/update networks decompose each sequence into multi-scale coefficients per block, trained end-to-end with causality preserved via zero-padded dilation. Decompose/reconstruct weights are shared. Untying them had negligible performance impact while saving parameters.

- **Fast Walsh-Hadamard Transform (FHT)**: a fixed orthogonal O(C log C) cross-channel rotation replacing attention's channel-mixing role. Cost is independent of sequence length.

- **Per-scale gated spectral mixer (SwiGLU)**: mixes each wavelet scale independently in Walsh-Hadamard space via a gated linear layer. Runs in fixed O(S²) per layer for S scales (S = levels + 1), versus attention's O(N²) in sequence length.

- **Expanded MLP (expansion ≥ 20)**: Hidden layer width multiplier for the MLP layers. Logarithmic relationship with BPB.

- **Decompose bypass**: a causal cumulative mean of pre-decompose hidden states, projected per-scale and added as bias to the post-decompose coefficients.

<details>
<summary><b>Additional key components</b> (always-on architectural pieces)</summary>

- LayerNorms near both ends of each block, and one before the LM head
- Two residual connections per block with learned scalar gating (`learned_residual` in config.json)
- Per-scale weights applied after the inverse FHT, one trainable scalar per wavelet scale
- Feature padding to the next power of 2, required for the Walsh-Hadamard transform (`C` → `Cp = next_pow2(C)`)
- Causal zero-padded dilation in the lifting predict/update steps, preserving autoregressive causality at every level

</details>

### Optional Features

- **Per-Layer Embedding**: a learned per-channel residual of the token embedding added at each block, letting deeper blocks reach back to the input representation.

- **Product Key Memory / Fast-Weight Product Key Memory**: sparse key-value memory modules complementing the dense MLP, with optional inference-time fast-weight updates.

- **Low-Rank Factorization**: a rank-r `U·V^T` perturbation added to the spectral mixer; rank=4 yields a measurable BPB improvement at trivial parameter cost.

- **Exponential Parametrization**: reparameterizes mixer weights through `exp()`, stabilizing training under high learning rates that would otherwise NaN.

- **Cross-scale gating (routing mode)**: a learned identity-initialized (S, S) routing matrix that mixes per-scale inputs before each gate, enabling conditional cross-scale interactions.

- **Per-scale mixer widths**: asymmetric per-scale mixer capacity (coarse scales full width, fine scales reduced). At `[1, 1, 1, 0.5, 0.5, 0.5]`: small BPB improvement + ~23% per-epoch speedup.

- **Wavelet crawl**: softmax-weighted mixture of K candidate dilations per level around the base `2^l`, letting the model discover off-power-of-2 receptive fields. K=3 (±1) is the stable sweet spot.

- **Shared lifting weights**: one lifting wavelet module shared across all blocks. Essentially free on BPB; cuts training VRAM by ~5–10% at L=2.

- **Looped blocks (Universal Transformer-style)**: one shared block applied K times in place of L stacked blocks. Reduces BPB at fixed parameter count; compute is usually better spent on more epochs of the stacked model.

<details>
<summary><b>Additional optional features</b> (all configurable in <code>config.json</code>)</summary>

- Data-dependent EMA decompose-bypass (`decompose_bypass_ema`): σ-gated adaptive IIR replacement for the cumulative running mean. Promising at 1 epoch (-0.30 nats val loss), regressed at 5 epochs (BPB 1.0226 vs 1.0201 baseline). Rejected for release; investigation plan in [plans/ema_post_release.md](plans/ema_post_release.md).
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


## Results

It is important to note that WaveletLM has **not** been fully optimized: 

- it is underregularized with a 0.8 train/val loss gap, 
- the 5 dropout parameters have not been swept, 
- weight decay needs further tuning, 
- longer training time is needed, and 
- parameter compression has not yet been applied.

My current run budget is limited. Other researchers are encouraged to train the model with these changes to more accurately gauge its potential performance.

See [Areas for Improvement](#areas-for-improvement) below for more info on optimization, and [Future Plans](#future-plans) for ways to push WaveletLM further post-release.

### PG-19 Test Set Perplexity Comparison

| Model | Type | Params | Context | Epochs | PPL |
|-------|------|--------|---------|--------|-----|
| Hyena ‡ | Long convolution and recurrence | 153M | 16,384 | 8 | 14.6[^8] |
| **WaveletLM (1 epoch)** | **Wavelet mixer** | **808M** | **256** | **1** | **27.40†** |
| Perceiver AR | Cross-attn + latents | 974M | 4,096 | ~210 | 28.9[^5] |
| Block-Recurrent Transformer | Transformer + recurrence | ~200M | 4,096 + recurrent | — | 29.0[^6] |
| Compressive Transformer | Transformer + compressive memory | 257M | 2,048 effective | ~50 | 33.6[^7] |
| Transformer-XL | Transformer + recurrence | 257M | 1,024 effective | ~50 | 36.3[^7] |

All models in this table were trained and evaluated on PG-19. Most use SentencePiece tokenization; Hyena uses GPT-2 BPE. WaveletLM was trained for one epoch only at the smallest context length of any entry.

Context and epoch derivations from source papers:

- **Compressive Transformer / Transformer-XL** (Rae et al. 2019): `~100B tokens / 2B PG-19 tokens ≈ 50 epochs`. Effective context: `512 (window) + 512 (memory) + 512 × 2 (compressed, C=2) = 2,048` (CT); `512 (window) + 512 (cache) = 1,024` (TX-L).
- **Block-Recurrent Transformer**: Context: `4,096-token segments + 512-vector recurrent state`, trained for 500k steps. Epoch count cannot be derived because the batch size was not reported.
- **Perceiver AR**: `~200k steps per batch × 2048 batches ≈ 420B tokens`; `420B / 2B PG-19 tokens ≈ 210 epochs` at `4,096-token context`.

† 27.40 sliding-window PPL. See the [PG-19 pre-release run](runs.md#pg-19-pre-release-benchmark-best-seed-1-epoch) for full details and the [run log](logs/pg19_2026-04-25_13-34-46/log.txt). Increased regularization and training time are in the [Future Plans](#future-plans) section.

‡ Hyena was trained with `block_size=16384` (64× WaveletLM's) and 8 epochs (8× WaveletLM's). It is also incredibly efficient parameter-wise with 153M vs. WaveletLM's 807M. Increasing both block size and epochs for WaveletLM while decreasing parameters are some of the [Future Plans](#future-plans).

Comparison numbers for both datasets are sourced from their respective papers. See References below.

### WikiText-103 Test Set Perplexity Comparison

| Model | Type | Trained on | Params | PPL |
|-------|------|-----------|--------|-----|
| GPT-2 XL | Transformer | WebText (40GB) | 1.5B | 17.5[^3] |
| Transformer-XL Large* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 257M* | 18.3[^2]* |
| GPT-2 Large | Transformer | WebText (40GB) | 774M | 19.3[^3] |
| S4* | SSM* | WikiText-103 (0.5GB)* | 130M* | 20.9[^4]* |
| GPT-2 Medium | Transformer | WebText (40GB) | 355M | 22.1[^3] |
| **WaveletLM** | **Wavelet mixer** | **WikiText-103 (0.5GB)†** | **883M** | **23.8†** |
| Transformer-XL Standard* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 151M* | 24.0[^2]* |
| GPT-2 | Transformer | WebText (40GB) | 124M | 29.4[^3] |

\* Both trained and evaluated on WikiText-103 only (direct comparison to WaveletLM). GPT-2 BPE was used by WaveletLM for tokenization.

† Best of 3 seeds PPL of 23.749 with mean PPL of 23.818. Significant parameter reduction is planned post-release in the [Future Plans](#future-plans) section.

- [3-seed variance study](runs.md#3-seed-variance-study-l2-c2048-20x-dropout-5-epochs) 
- [Best run's training log](logs/wikitext-103_2026-04-22_01-36-47/log.txt)

See [`runs.md`](runs.md) for a record of all training runs, logs, configs, and benchmark results with fully-reproducible point-in-time code snapshots.

### Areas for Improvement

Longer training time, more regularization, and parameter compression are the surest ways to immediately improve the model's performance.

**More training time**: More research and more resources are needed to uncover the effects of longer training.

**Regularization**: WaveletLM is vastly underregularized, with a 0.8 train/val loss gap at 5+ epochs. Dropout and weight decay parameter sweeps are limited by budget and involve tuning `weight_decay` `dropout_embedding`, `dropout_projection`, `dropout_mixer`, `dropout_mlp`, and `dropout_lm_head` in tandem.

**Parameter compression**: Of WaveletLM's 883M parameter total, around 55% (488M) live in two highly compressible components: dense MLPs (335.6M) and product-key memory modules (PKM: 76M + FwPKM: 76M). Further work is needed to determine the degree of compressivity of each during training, which makes it complementary to PTQ.


## Future Plans

### Single-Layer WaveletLM with Current Best Config

Re-test L=1 with the current best feature stack and the parameter-reduction bundle below. L=1 nearly halves runtime and parameter count, meaning it can train for twice the epochs as L=2 at fixed compute. 3 tests: L=1 with epochs=1, L=2 with epochs=1, and L=1 with epochs=5, each on WikiText-103. The existing L=2 with epochs=5 baseline completes the comparison matrix.

**Interpretability motivation**: single-hop information flow makes targeted interpretability techniques such as per-scale activation patching, embedding-to-coefficient mapping, and per-component ablation tractable. This compounds with the eventual semantic embedding reintroduction, where L=1 enables end-to-end conceptual traceability through human-readable dimensions.

See [plans/single_layer_waveletlm.md](plans/single_layer_waveletlm.md) for the full design.

### Combined Parameter Reduction and VRAM Reallocation

Combine four cheap reductions (`mlp_expansion: 10`, `pkm_enabled: false`, `fwpkm_num_keys: 8281`, and `tie_embedding_to_lm_head: true`) to bring the model from 882.5M parameters to ~497.8M (~44% reduction) at a projected +0.025 BPB additive cost. PKM is dropped (rather than FwPKM) since the two perform comparably on quality and FwPKM also retains the optional `fwpkm_inference_updates` capability. The freed VRAM can be reallocated to longer block size, larger effective batch size, more epochs, or dropout retuning at the same hardware budget. See [plans/other_post_release_plans.md §8](plans/other_post_release_plans.md#8-combined-parameter-reduction-and-vram-reallocation) for more.

### Cross-Layer Parameter Sharing

ALBERT-style weight tying across the two WaveletLM blocks. Test sequence: full-block tying first (~338M potential savings), then component-isolated tying with MLP only (~168M), then PKM/FwPKM only (~76M). Tests whether depth-via-iteration substitutes for depth-via-distinct-parameters in the wavelet architecture, as it does in transformers. See [plans/other_post_release_plans.md §9](plans/other_post_release_plans.md#9-cross-layer-parameter-sharing-albert-style) for the full design.

### Dropout Sweep

Re-tune the five dropout values (`dropout_lm_head`, `dropout_mlp`, `dropout_mixer`, `dropout_projection`, and `dropout_embedding`) once model parameters are reduced from above. A doubled-dropout ablation at the prior baseline gave -0.0221 BPB. This is larger than the projected BPB increase from parameter reduction. A true dropout sweep may surpass the gap.

### Weight Decay Sweep

Re-tune `weight_decay`. Current value (1e-6) was only tested alongside 1e-3. More values must to be attempted.

### Per-scale Mixer Transform Ablation

Test the contribution of the FWHT slot in the per-scale mixer versus having no transform, having a Hartley transform, using a DCT-II/III pair, or employing a butterfly-parametrized learned orthogonal mixer. Measures whether FWHT specifically is necessary, or whether any orthogonal mixer of similar structure (or none at all, with the learned embedding in place) achieves equivalent performance. See [plans/other_post_release_plans.md §10](plans/other_post_release_plans.md#10-per-scale-mixer-transform-ablation) for the full design and proposed test.

### 2D Wavelet over (Batch, Token) with Sequential Training

Generalize the lifting wavelet decomposition from 1D over the token axis to 2D over the joint (batch, token) axis pair. When training proceeds in document-sequential order, the batch axis carries the same multi-scale temporal structure as the token axis, and the same wavelet machinery applies to both. This requires reorganization of the current batch sampling method into a sequential batch processing method, since batches may not be IID with respect to each other. As one example, consider series of novels in PG-19 with temporal plot dependencies between books. Randomly sampling batches breaks this temporal relationship. Using 2D wavelets is a convenient way to enforce and encode temporal relationships at all levels within the model. See [plans/two_d_wavelet_sequential_training.md](plans/two_d_wavelet_sequential_training.md) for the full design.

### Longer PG-19 Training

The PG-19 run above was trained for a single epoch using the WikiText-optimized config. Published baselines for other models on the same dataset were likely trained for many more epochs or with much more effective compute. 

Once it is possible, the first post-release goal will be to train on PG-19 for 2 epochs, and loss permitting, 5 epochs, in order to better gauge language modeling on a large dataset at the current parameter size.

### Dataset Comparisons

The best WaveletLM config trained on Pile-ArXiv, BookCorpusOpen, OpenWebText, and other datasets to gauge their performance.

### Model Comparisons

Side-by-side benchmarks against Hyena, Transformer, Mamba, RWKV, and other modern architectures on WikiText-103 at matched compute and fully optimized.

### Scaled-Up Model (B200)

The 883M RTX 5090 headline run scales up naturally to a B200:

- `C`: 2048 → 4096 
- `layers`: 2 → 4–8
- `mlp_expansion`: 20 → 50–200
- `pkm_num_keys` & `fwpkm_num_keys`: 16384 → 65536 each
- fp16 → FP8 via Blackwell tensor cores (NYI)

The goal is a 10–15B parameter configuration, trained individually on WikiText-103 and PG-19, and also on a multi-dataset mix of WikiText-103, PG-19, Pile-ArXiv, BookCorpusOpen, TinyStories, & OpenWebText. 

Inference would fit on a single RTX 4090 at fp16 and roughly half the VRAM with [uniform 8-bit PTQ](runs.md#ptq-sweep-summary). See [`runs.md`](runs.md#post-release-scaled-up-b200-configuration) for the pending run entry.

### Optimizer Sweep (Adagrad / AdamW / Muon)

Adagrad (lr=0.01) is the validated optimizer for the released model but has not been directly compared against properly-tuned alternatives. WaveletLM is matrix-parameter-heavy (MLP at expansion=20 produces Linear(2048, 40960) weights, plus per-scale mixers and lifting matrices), so [Muon (Jordan et al., 2025)](https://arxiv.org/abs/2502.16982) - which orthogonalizes matrix gradient updates via Newton-Schulz iteration and reports 1.5–2× wall-clock speedups vs AdamW on small transformers - is a strong candidate. Plan: a 2-phase sweep (1-epoch LR screening + 5-epoch finalist validation) across Adagrad, AdamW, and Muon. Even a 30% wall-clock speedup compounds across every subsequent ablation and the B200 scale-up. See [plans/other_post_release_plans.md §6](plans/other_post_release_plans.md#6-optimizer-sweep-adagrad--adamw--muon).

### Bit-Packed PTQ Kernels

The [current PTQ path](runs.md#ptq-sweep-summary) dequantizes int8 weights to fp16 inside `forward()` and runs a standard fp16 matmul, which pays the dequant cost every step with no bandwidth win - hence the 12% generation slowdown and the fact that sub-8-bit variants compress identically to 8-bit on disk. 

Swapping `QuantizedLinear` / `QuantizedEmbedding` for fused packed-weight kernels (Marlin W8A16 / W4A16, CUTLASS `i8gemm`, bitsandbytes, Triton for the embedding lookup) fixes both: storage scales with bit-width, and each matmul reads half or a quarter as many bytes. Expected generation at batch=1 (fp16 baseline 28.8 tok/s) is **~1.4–1.6× faster** for fused uniform 8-bit and **~1.8–2.2× faster** for fused mixed 8/4/2, with BPB unchanged. See [runs.md](runs.md#post-release-bit-packed-ptq-kernels) for the full plan.

### Semantic Embedding & Interpretability Work

An optional replacement for the learned token embedding is a **semantic embedding**, where each dimension is a plain-language feature (e.g. "is this token a noun?", "is this token associated with anger?", "corpus frequency in deceptive contexts") and each token or n-gram is a vector of values across those dimensions. 

WaveletLM is structurally well-suited for this: the spectral mixer can operate directly on vectorized human-readable features, and multi-scale decomposition lets the same concept be processed at different temporal granularities. The expected tradeoff is improved interpretability at a small performance cost, potentially recovered or even improved via n-gram tokens and careful feature selection for the dimensions. 

See [plans/reincorporate_large_semantic_embedding.md](plans/reincorporate_large_semantic_embedding.md) for the full design, including open questions on coefficient assignment methods: one-hot/binary, LLM-scored, human-rated, or corpus-derived.

### Multinodal Mode

WaveletLM supports a product-of-experts mode where multiple independent nodes process the input in parallel with feature bagging and logit averaging. Enable with `multinodal_enabled: true` in the config. This mode may require stability adjustments such as a lower learning rate with `stable_parametrization` enabled, and acts as an as-yet underexplored capacity/scalability lever. Broader multi-expert training techniques (sparse MoE, mutual learning, weight averaging, Git Re-Basin, & ensemble distillation) surveyed in [plans/multinodal_training_techniques.md](plans/multinodal_training_techniques.md).

### Adaptive Decompose Bypass

Replacing the parameter-free cumulative running mean with a data-dependent EMA (`decompose_bypass_ema`) gained -0.30 nats at 1 epoch, but regressed at 5 epochs (BPB 1.0226 vs 1.0201). The inversion likely due to short-horizon forgetting and learned gate overfitting. Post-release plan: develop freeze-gate/bias correction probes and alternative formulations with a selective SSM bypass as fallback. See [plans/ema_post_release.md](plans/ema_post_release.md).

### Other Post-Release Plans

See [plans/other_post_release_plans.md](plans/other_post_release_plans.md) for info on each.

- Cross-scale phase gating (coarse-modulates-fine)
- Stable parametrization: validation and finishing gaps 
- Data-dependent lifting networks (Mamba-style)
- Wavelet Packet Decomposition (WPD)
- Top-K / hard thresholding in the Hadamard domain


## License

Apache License 2.0


## References

[^2]: Dai et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." arXiv:1901.02860, 2019.
[^3]: Radford et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019.
[^4]: Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces." arXiv:2111.00396, 2021.
[^5]: Hawthorne et al. "General-purpose, long-context autoregressive modeling with Perceiver AR." arXiv:2202.07765, 2022.
[^6]: Hutchins et al. "Block-Recurrent Transformers." arXiv:2203.07852, 2022.
[^7]: Rae et al. "Compressive Transformers for Long-Range Sequence Modelling." arXiv:1911.05507, 2019. (PG-19 dataset introduction; reports both Compressive Transformer and Transformer-XL on PG-19.)

[^8]: Poli et al. "Hyena Hierarchy: Towards Larger Convolutional Language Models." arXiv:2302.10866, 2023. PG-19 result on page 20: Hyena 153M reaches 14.6 test PPL with 16k context length, 8 epochs, GPT-2 BPE tokenization.
