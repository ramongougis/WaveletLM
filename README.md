<p align="center">
  <img src="assets/waveletlm-header.svg" alt="WaveletLM" width="90%"/>
</p>

<p align="center">
  <img src="assets/divider.svg" alt="" width="85%" height="1"/>
</p>

<br>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://github.com/ramongougis/WaveletLM/actions/workflows/codeql.yml"><img src="https://github.com/ramongougis/WaveletLM/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"/></a>
  <a href="https://github.com/ramongougis/WaveletLM/actions/workflows/scorecard.yml"><img src="https://github.com/ramongougis/WaveletLM/actions/workflows/scorecard.yml/badge.svg" alt="Scorecard supply-chain security"/></a>
  <a href="https://github.com/ramongougis/WaveletLM/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<br>

WaveletLM is a generative, attention-free language model based upon the architecture discovered by Andrew Kiruluta et al. (2025)[^1][^2] in the following papers:

- **Wavelet Logic Machines**: wavelet-based classification on fixed, pretrained embeddings ([arXiv:2507.19514](https://arxiv.org/abs/2507.19514))
- **Learnable Multi-Scale Wavelet Transformer**: wavelet-based machine translation ([arXiv:2504.08801](https://arxiv.org/abs/2504.08801))

WaveletLM adapts the Wavelet Logic Machine's approach to autoregressive language modeling with the components detailed in the [Architecture](#architecture) section below. Furthermore, a planned replacement of the current learned embedding with a fixed, human-readable semantic embedding would more than halve the trainable parameters achieved by our [benchmark results](#results) while extending the Wavelet Logic Machine's interpretability benefits to the generative setting. For details, see the [Future Plans](#future-plans) section.

It uses a learned embedding and mixes tokens using causal lifting wavelet decomposition, a Fast Walsh-Hadamard Transform, per-scale gated spectral mixer with SwiGLU activation, inverse FWHT, and wavelet reconstruction. Combined with a 2-layer, width-expanded MLP and Fast-weight Product Key Memory for inference-time updates, this yields an architecture with no attention and O(n log n) scaling in sequence length with the potential for limited-capacity continual learning.

Current [results](#results) show better performance on PG-19 than Perceiver AR, the Compressive Transformer, and Transformer-XL with a single epoch of training, and better performance on WikiText-103 than Transformer-XL and GPT-2. 

Furthermore, several improvements have been made since the headline model in the [Results](#results) section was trained, and are awaiting completion before the release of an updated version. For more information, see the [Future Plans](#future-plans) section, which tracks all work currently in progress.

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

Requires Python 3.10+, PyTorch 2.11+, and CUDA.

```bash
git clone https://github.com/ramongougis/WaveletLM.git
cd WaveletLM
pip install "torch>=2.11" "datasets<3.0" tiktoken sentencepiece tqdm numpy --extra-index-url https://download.pytorch.org/whl/cu128
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

The [current best PG-19 model](logs\pg19_2026-04-25_13-34-46\benchmark.txt) requires 5,266 MiB for inference and generates at 28.3 tokens/s on a 5090.

The [current best WikiText-103 model](logs/wikitext-103_2026-04-22_01-36-47/log.txt) requires 5,730 MiB for inference and generates at 28.8 tokens/s on a 5090.

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

**Sample E** with prompt "The soldiers marched" and naive sampling (no `--strategies`, default `temp=1.0`, and `top_p=0.95`):

> The soldiers marched in uniform. Ten minutes later, Mr. Farrell 
> approached Mr. Talbot and the captain of police. He had been told that 
> he must start immediately for New York, and sent for his son, Mr. 
> Waters, to meet him there. With this he departed on his return, leaving 
> his wife in such distress at the selfish desertion of her husband and 
> son as to despair of recovering the stolen property. But both Frank and 
> Harry were deeply anxious about her. They knew that their father would 
> not thank them for killing the lady with whom they had arrived; and 
> when he told them how it was done she resolutely turned away her head. 
> And

One-shot naive sample (without `--strategies`). Naive samples retain more of the model's raw distribution and tend to mix domains more loosely.

**Sample F** with prompt "The history of" and naive sampling (no `--strategies`, default `temp=1.0`, and `top_p=0.95`):

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

Randomly selected naive sample (without `--strategies`) out of 20.

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

**Sample E** with prompt "The soldiers marched" and naive sampling (no `--strategies`, default `temp=1.0`, and `top_p=0.95`):

> The soldiers marched through a number of hiding places and ambushes . 
> These were the first major engagements that took place during this 
> time — only one is known from sources , although most historians agree 
> the number differs in details . )
>
> The rebel army commenced its march from Livno on 29 October , followed 
> by their advance towards Palermo on 21 November . So – when there was 
> some initial resistance to that tactic – General Guy de Croix — 
> Enghien 's men started gathering at dawn on September 20 , but most 
> would not arrive until midday . Their combined forces divided into 
> two columns , with reinforcements arriving from all sides within 
> hours . The main

**Sample F** with prompt "The history of" and naive sampling (no `--strategies`, default `temp=1.0`, and `top_p=0.95`), but with the spacing post-processing fix now included in naive generations:

> The history of the tropical cyclone is unknown, but official records 
> suggest it formed on May 31. It developed into a tropical storm later 
> that day. After turning to the northeast, on September 20 Huron 
> strengthened into a hurricane while east-southeast of Bermuda. During 
> that time, Hurricane Humberto destroyed another ship in a prolonged 
> action. In late October, Ione crossed into the eastern Gulf of Mexico; 
> however, it reintensified slightly and later peaked with winds of 100 
> mph (160 km/h) as it drifted through western Cuba.

Note the typical failure mode with the naive generation: register-coherent meteorological prose, but the model freely interleaves the names of multiple unrelated real storms (Huron, Humberto, Ione) within a single passage. Without `--strategies`, the model does not employ a more conservative sampling regime.

</details>


## Architecture

<p align="center">
  <img src="assets/waveletlm-architecture.svg" alt="WaveletLM architecture" width="80%"/>
</p>

The high-level architectural premise, using learnable wavelets in place of self-attention as a sequence mixer, follows Kiruluta's' Wavelet Logic Machine[^1] and Kiruluta, Burity, and Williams's Learnable Multi-Scale Wavelet Transformer[^2]. WaveletLM extends this approach from sentiment classification to language modeling with several architectural additions and components, detailed below.

### Key Components

- **Learned lifting wavelets**: Haar-initialized MLPs decompose each block into multi-scale coefficients via lifting predict/update steps. Each wavelet scale processes either coarse summaries or fine details across tokens. Reconstruction reuses the same MLPs in reverse with a sign flip, so perfect inversion is structurally guaranteed regardless of what the weights learn. About 16.8M parameters per (predict, update) pair at C=2048, one pair per scale, and shared across all layers via `shared_lifting_weights`.

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
<summary><b>Additional features</b> (all configurable in <code>config.json</code>)</summary>

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
| Hyena ‡ | Long convolution and recurrence | 153M | 16,384 | 8 | 14.6[^10] |
| **WaveletLM (1 epoch)** | **Wavelet mixer** | **808M** | **256** | **1** | **27.40†** |
| Perceiver AR | Cross-attn + latents | 974M | 4,096 | ~210 | 28.9[^7] |
| Block-Recurrent Transformer | Transformer + recurrence | ~200M | 4,096 + recurrent | — | 29.0[^8] |
| Compressive Transformer | Transformer + compressive memory | 257M | 2,048 effective | ~50 | 33.6[^9] |
| Transformer-XL | Transformer + recurrence | 257M | 1,024 effective | ~50 | 36.3[^9] |

All models in this table were trained and evaluated on PG-19. Most use SentencePiece tokenization; Hyena uses GPT-2 BPE. WaveletLM was trained for one epoch only at the smallest context length of any entry.

Context and epoch derivations from source papers:

- **Compressive Transformer / Transformer-XL** (Rae et al. 2019): `~100B tokens / 2B PG-19 tokens ≈ 50 epochs`. Effective context: `512 (window) + 512 (memory) + 512 × 2 (compressed, C=2) = 2,048` (CT); `512 (window) + 512 (cache) = 1,024` (TX-L).
- **Block-Recurrent Transformer**: Context: `4,096-token segments + 512-vector recurrent state`, trained for 500k steps. Epoch count cannot be derived because the batch size was not reported.
- **Perceiver AR**: `~200k steps per batch × 2048 batches ≈ 420B tokens`; `420B / 2B PG-19 tokens ≈ 210 epochs` at `4,096-token context`.

† 27.40 sliding-window PPL. See the [PG-19 pre-release run](runs.md#pg-19-pre-release-benchmark-best-seed-1-epoch) for full details and the [run log](logs/pg19_2026-04-25_13-34-46/log.txt). Increased regularization and training time are in the [Future Plans](#future-plans) section.

‡ Hyena was trained with `block_size=16384` (64× WaveletLM's) and 8 epochs (8× WaveletLM's). It is also incredibly efficient parameter-wise with 153M vs. WaveletLM's 807M. Increasing both block size and epochs for WaveletLM while decreasing parameters are some of the [Future Plans](#future-plans).

Comparison numbers for both datasets are sourced from their respective papers. See References below.

### WikiText-103 Test Set Perplexity Comparison

| Model | Type | Trained on | Params | Context | PPL |
|-------|------|-----------|--------|---------|-----|
| GPT-2 XL | Transformer | WebText (40GB) | 1.5B | 1024 | 17.5[^5] |
| Transformer-XL Large* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 257M* | 1024 effective* | 18.3[^4]* |
| GPT-2 Large | Transformer | WebText (40GB) | 774M | 1024 | 19.3[^5] |
| S4* | SSM* | WikiText-103 (0.5GB)* | 130M* | 1024* | 20.9[^6]* |
| GPT-2 Medium | Transformer | WebText (40GB) | 355M | 1024 | 22.1[^5] |
| **WaveletLM** | **Wavelet mixer** | **WikiText-103 (0.5GB)†** | **883M** | **256†** | **23.8†** |
| Transformer-XL Standard* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 151M* | 1024 effective* | 24.0[^4]* |
| GPT-2 | Transformer | WebText (40GB) | 124M | 1024 | 29.4[^5] |

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

- [(Done) Single-Layer WaveletLM with Current Best Config](#done-single-layer-waveletlm-with-current-best-config)
- [(Done) Parameter Reduction](#done-parameter-reduction)
- [(Done) Larger Block Size](#done-larger-block-size)
- [(Done) Per-Scale Mixer Width Contraction and Expansion](#done-per-scale-mixer-width-contraction-and-expansion)
- [(Done) Mixer Low Rank](#done-mixer-low-rank)
- [(Done) T1 Baseline Without Wavelet Crawl](#done-t1-baseline-without-wavelet-crawl)
- [(Done) New Baseline T2 with 7 Levels, more Per-Scale Mixer Weights, and Wavelet Crawl](#new-baseline-t2-with-7-levels-more-per-scale-mixer-weights-and-wavelet-crawl)
- [(Post-Release) Optimizer Swap (Muon)](#post-release-optimizer-swap-muon)
- [(Done) Sequential Block Ordering](#done-sequential-block-ordering)
- [(Shelved on WikiText-103) 2D Wavelet over (Batch, Token) with Sequential Training](#shelved-on-WikiText-103-2d-wavelet-over-batch-token-with-sequential-training)
- [(Done) Bisected Block Context Extension](#done-bisected-block-context-extension)
- [(Done) Adagrad Learning Rate Tuning](#done-adagrad-learning-rate-tuning)
- [(Done) New T3 Baseline](#done-new-t3-baseline)
- [(Done) Recurrence with Adagrad (partial)](#done-recurrence-with-adagrad-partial)
- [(Done) Optimizer Swap (AdamW) and Wavelet Norms](#done-optimizer-swap-adamw-and-wavelet-norms)
- [(Done) Optimizer Tuning (Adagrad) with Wavelet Norms](#done-optimizer-tuning-adagrad-with-wavelet-norms)
- [(Done) Spectral Norm](#done-spectral-norm)
- [(Done) New T4 Baseline](#done-new-t4-baseline)
- [(Done) Recurrence with Adagrad (no residual)](#done-recurrence-with-adagrad-no-residual)
- [(Done) Recurrence with Adagrad (with residual)](#done-recurrence-with-adagrad-with-residual)
- [(Done) Recurrence Efficiency: Gate Caching](#done-recurrence-efficiency-gate-caching)
- [(Done) Long-Range Context: Multi-Pole SSM + Truncated BPTT](#done-long-range-context-multi-pole-ssm--truncated-bptt)
- [(Done) Dense Mixer Recurrence](#done-dense-mixer-recurrence)
- [(Done) Untied Wavelet Reconstruction](#done-untied-wavelet-reconstruction)
- [(Done) Dropout](#done-dropout)
- [(Done) Weight Decay](#done-weight-decay)
- [(Done) Complex Wavelets and Complex Mixer](#done-complex-wavelets-and-complex-mixer)
- [(Done) Wavelet Crawl Off](#done-wavelet-crawl-off)
- [(Done) Wavelet Sparsity Probe & Wavelet Shrinkage](#done-wavelet-sparsity-probe--wavelet-shrinkage)
- [(Done) Inference-Depth Flexibility with Mixer Recurrence (Train Deep, Infer Shallow)](#done-inference-depth-flexibility-with-mixer-recurrence-train-deep-infer-shallow)
- [(Done) Mixer Transform Ablation](#done-mixer-transform-ablation)
- [(Done) Wavelet Crawl Dilation Window (K) Sweep](#in-progress-wavelet-crawl-dilation-window-k-sweep)
- [Structure Factoring](#structure-factoring)
- [Step-Time Speedups](#step-time-speedups)
- [T5 Baseline](#t5-baseline)
- [More Layers](#more-layers)
- [More Epochs](#more-epochs)
- [More Width (C)](#more-width-c)
- [Cross-Layer Skip Connections](#cross-layer-skip-connections)
- [Longer PG-19 Training](#longer-pg-19-training)
- [Dataset Comparisons](#dataset-comparisons)
- [Model Comparisons](#model-comparisons)
- [Generation Decode Speedup (compile / CUDA graphs)](#generation-decode-speedup-compile--cuda-graphs)
- [Bit-Packed PTQ Kernels](#bit-packed-ptq-kernels)
- [Multi-Transform Parallelization](#multi-transform-parallelization)
- [Semantic Embedding & Interpretability Work](#semantic-embedding--interpretability-work)
- [Combined Multi-Transform + Semantic Embedding (Interpretability Compound)](#combined-multi-transform--semantic-embedding-interpretability-compound)
- [Adaptive Decompose Bypass](#adaptive-decompose-bypass)
- [Prime-Power Wavelet Filterbank](#prime-power-wavelet-filterbank)
- [Multinodal Mode (Product-of-Experts)](#multinodal-mode-product-of-experts)
- [Final Regularization Sweep](#final-regularization-sweep)
- [Scaled-Up Model (B200)](#scaled-up-model-b200)
- [Scaled-Up Model with PTQ and other Infernece Strategies](#scaled-up-model-with-ptq-and-other-infernece-strategies)
- [Downstream Transfer Fine-Tuning](#downstream-transfer-fine-tuning)
- [Instruction-Tuning Chat Demo](#instruction-tuning-chat-demo)
- [Other Post-Release Plans](#other-post-release-plans)

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Single-Layer WaveletLM with Current Best Config

**Result:** 

Tested `layers=1` and `layers=2`, each for 1 epoch and 5 epochs. Both runs were also trained at near-equal wall-clock by extending the 1-layer result to 8 epochs (`layers=1` and `epochs=8` took 15.86h, while `layers=2` and `epochs=5` took 16.25h). 

| Run | Layers | Epochs | BPB sliding | PPL sliding | Params | Train time | Links |
|-----|--------|--------|-------------|-------------|--------|------------|-------|
| A | 1 | 1 | 1.1648 | 38.04 | 586.15M | 2.09h | [link](logs/wikitext-103_2026-04-29_20-45-37/log.txt) |
| B | 2 | 1 | 1.1129 | 32.35 | 882.51M | 3.43h | [link](logs/wikitext-103_2026-04-29_22-52-28/log.txt) |
| C | 1 | 5 | 1.0809 | 29.28 | 586.15M | 9.74h | [link](logs/wikitext-103_2026-04-30_02-20-35/log.txt) |
| D | 2 | 5 (headline) | **1.0140** | **23.75** | 882.51M | 16.25h | [link](logs/wikitext-103_2026-04-22_01-36-47/log.txt) |
| E | 1 | 8 (compute-equalized) | 1.0715 | 28.43 | 586.15M | 15.86h | [link](logs/wikitext-103_2026-04-30_12-20-45/log.txt) |

**Conclusions:** 
- `layers=1` and `epochs=1` are set for future tests to hasten iteration time.
- Benchmark runs will use `layers=2` and `epochs=5`, or more of each.
- Training `layers=1` and `layers=2` for approximately equal wall-clock time does not compensate for the lack of layers.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Parameter Reduction

Starting with the best WikiText-103 config, we made the following changes to reduce parameters:

- `mlp_expansion: 20 → 10`
- `pkm_enabled: true → false`
- `fwpkm_num_keys: 16384 → 8281`
- `tie_embedding_to_lm_head: false → true`

**Results:** 
- 41% fewer parameters
- 21% less training time
- slight **positive** performance impact (−0.0013 BPB)
- 26% smaller train/val loss gap (**implicit regularization via less params yields better performance** at this depth & scale)

The "Test 1", aka **T1**, configuration in the table below incorporates these reductions. 

| Run | Recipe | Folder | BPB sliding | PPL sliding | Best val | Min train | Train/val gap | Params | Train time | Training VRAM |
|-----|--------|--------|-------------|-------------|----------|-----------|---------------|--------|------------|------|
| Best WikiText-103 run with layers=1 and epochs=5 | Best run with layers=1 and epochs=5 | [link](logs/wikitext-103_2026-04-30_02-20-35/log.txt) | 1.0809 | 29.28 | 3.3275 | 2.8292 | 0.498 | 586.15M | 9.74h | 11,537 MiB |
| **T1/Test 1** | Best run with layers=1, epochs=5, & parameter reductions | [link](logs/wikitext-103_2026-05-01_06-33-48/log.txt) | **1.0796** | **29.15** | 3.3341 | 2.9649 | **0.369** | **344.63M** | 7.69h | 6,867 MiB |
| Δ | — | — | −0.0013 | −0.13 | +0.007 | +0.136 | −26% | −41% | −21% | — |

**Conclusion:** Keep the parameter reductions listed here and use T1 as the baseline for tests (until it's changed to the [T2 baseline](#new-baseline-t2-with-7-levels-more-per-scale-mixer-weights-and-no-wavelet-crawl)).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Larger Block Size

In this section, we tested the new T1 baseline with `block_size=16384`, which required various adjustments to fit it into VRAM. With these changes, it was temporarily named the **R0** run and used as a new test baseline, but later deprecated as the baseline.

Changes required:

- All four [parameter reduction](#done-parameter-reduction) changes above from the T1 (reduced parameter) build
- `block_size: 256 → 16384` for more context and direct comparison with other models
- `levels: 5 → 7` to process the higher context
- `per_scale_mixer_widths` extended to 8 entries to match levels + 1 scales
- `micro_batch_size: 8 → 1` to accommodate the larger block size in VRAM
- `wavelet_crawl: True → False` to remove the only (very small) convolutional component, which also showed no performance benefit in earlier ablations
- `epochs: 5 → 1` for faster test time

**Results:**

| Run | Epochs | Folder | BPB (sliding) | Params | Train VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|-------|
| Test 1 | 1 | [link](logs/wikitext-103_2026-05-09_07-52-25/log.txt) | 1.1762 † | 344.63M | 6,867 MiB | Inference VRAM 2,876 MiB |
| **R0** (T1 with changes above) | 1 | [link](logs/wikitext-103_2026-05-02_21-43-22/log.txt) | 1.2361 | 392.91M | 23,411 MiB | 1-epoch reference for future tests |
| Test 1 | 5 | [link](logs/wikitext-103_2026-05-01_06-33-48/log.txt) | 1.0796 | 344.63M | 6,867 MiB | 5-epoch production result |
| **R0** (T1 with changes above) | 5 | [link](logs/wikitext-103_2026-05-03_02-13-07/log.txt) | 1.0974 | 392.91M | 23,411 MiB | 5-epoch comparison to Test 1 |

At 5 epochs, R0 is +0.0178 BPB higher than T1 (1.0974 vs 1.0796). 

**Decision:** 

Keep T1. 

We used  R0's larger block size as the baseline for some later tests in anticipation of higher performance which didn't materialize. This was later reverted back to T1, and those sections moved to the [runs.md](runs.md) doc's Deprecated section near the bottom. 

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Per-Scale Mixer Width Contraction and Expansion

**Results:** 

| Variant | per_scale_mixer_widths | Epochs | BPB sliding | ΔBPB vs R0 | Mixer params | Run Log |
|---|---|---|---|---|---|---|
| Baseline (R0, 1ep) | [1.0×4, 0.5×4] | 1 | 1.2361 | — | 59.11M | [link](logs/wikitext-103_2026-05-02_21-43-22/log.txt) |
| Baseline (R0, 5ep) | [1.0×4, 0.5×4] | 5 | 1.0974 | — | 59.11M | [link](logs/wikitext-103_2026-05-03_02-13-07/log.txt) |
| Mild contraction (1ep) | [0.5×4, 0.25×4] | 1 | 1.2437 | +0.0076 | 35.70M (−39%) | [link](logs/wikitext-103_2026-05-05_05-48-40/log.txt) |
| Mild contraction (5ep) | [0.5×4, 0.25×4] | 5 | pending | pending | 35.70M (−39%) | queued |
| Aggressive contraction | [0.1×4, 0.05×4] | 1 | NaN at step 1250 | — | — | [link](logs/wikitext-103_2026-05-05_04-37-47/log.txt) |
| Expansion | [1.5×4, 0.5×4] | 5 | 1.1037 | +0.0063 | 73.06M (+24%) | [link](logs/wikitext-103_2026-05-05_14-00-32/log.txt) |

Full details in [runs.md → Mixer width contractions](runs.md#mixer-width-contractions-post-combined-reduction-baseline-l1-levels7-epochs1).

**Decision:** Minor parameter savings for too much performance cost. Keeping T1's `per_scale_mixer_width=[1.0x4, 0.5x4]`.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Mixer Low Rank

**Result:** `low_rank=16` (LR16) achieves a 1-epoch sliding BPB of 1.2342 vs. the reference BPB of 1.2361. A longer [5-epoch test](logs/wikitext-103_2026-05-05_21-36-45/log.txt) yielded 1.0971 vs. the baseline 1.0974 = −0.0003, a negligible difference. Full table in [runs.md → Low-rank ablations](runs.md#low-rank-ablations-post-combined-reduction-baseline-l1-levels7-epochs1).

**Decision:** Keep T1 with `low_rank=4`.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) T1 Baseline Without Wavelet Crawl

Test the T1 baseline without wavelet crawl for 1 epoch. This removes the only (very small: only 15 parameters with levels=5) convolution operation in the model. Performance impact negligible.

| Variant | Params (dense) | BPB sliding | Best val | Train VRAM | Inference VRAM (strategies) | Run Log |
|---|---|---|---|---|---|---|
| T1 (1ep) | 344.63M | 1.1762 | 3.6393 | 6,867 MiB | 2,876 MiB | [link](logs/wikitext-103_2026-05-09_07-52-25/log.txt) |
| T1_NoWC (1ep) | 344.63M | 1.1845 | 3.6658 | 6,867 MiB | 2,954 MiB | [link](logs/wikitext-103_2026-05-10_00-02-14/log.txt) |

**Conclusion:** Keep wavelet crawl on until a 5+ epoch test validates or refutes this result. Removing wavelet crawl had a detrimental +0.0083 BPB impact on performance across 1 epoch.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) New Baseline T2 with 7 Levels, more Per-Scale Mixer Weights, and Wavelet Crawl

Test the T1 baseline with wavelet crawl, levels = 7, and per_scale_mixer_weights = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]. Keep whatever improves performance.

The new baseline shall be named **T2**.

| Variant | Levels | Wavelet Crawl | Epochs | Params (dense) | BPB sliding | PPL sliding | Best val | Train VRAM | Inference VRAM (strategies) | Run Log |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 5 | ✗ | 1 | 344.63M | 1.1845 | 40.4564 | 3.6658 | 6,867 MiB | 2,954 MiB | [link](logs/wikitext-103_2026-05-10_00-02-14/log.txt) |
| T1 | 5 | ✓ | 1 | 344.63M | 1.1762 | 39.4195 | 3.6393 | 6,867 MiB | 2,876 MiB | [link](logs/wikitext-103_2026-05-09_07-52-25/log.txt) |
| T1 | 5 | ✓ | 5 | 344.63M | 1.0796 | 29.1533 | 3.3341 | 6,867 MiB | 2,876 MiB | [link](logs/wikitext-103_2026-05-01_06-33-48/log.txt) |
| T2 | 7 | ✗ | 1 | 392.91M | 1.1616 | 37.6678 | 3.6094 | 7,788 MiB | 3,186 MiB | [link](logs/wikitext-103_2026-05-10_01-39-25/log.txt) |
| **T2** | **7** | **✓** | **1** | **392.91M** | **1.1541** | **36.7905** | **3.5881** | **7,788 MiB** | **3,258 MiB** | [link](logs/wikitext-103_2026-05-10_03-39-43/log.txt) |
| **T2** | **7** | **✓** | **5** | **392.91M** | **1.0485** | **26.4564** | **3.2630** | **7,788 MiB** | **3,238 MiB** | [link](logs/wikitext-103_2026-05-10_05-33-24/log.txt) |

**Conclusion:** T2 baseline now includes wavelet crawl, levels = 7, and per_scale_mixer_weights = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5].

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Post-Release) Optimizer Swap (Muon)

**Phase 1: Muon** ([Jordan et al., 2025](https://arxiv.org/abs/2502.16982); used in DeepSeek-V4) — Newton-Schulz orthogonalization bounds every update's spectral norm, applied to the matrix-heavy MLP / mixer / lifting `Linear(C, C)`. Hybrid split: 2D non-embedding hidden weights → Muon, biases / norms / embeddings / LM head → AdamW. **Phase 2: AdamW** as fallback baseline. All runs use T2 architecture (`levels=7`, T2 mixer widths, `wavelet_crawl=true`, `bs=256`, `MBS=8`).

| Optimizer | LR | Weight Decay | Momentum | Eps | NS Steps | NS Coefficients | Adjust LR Fn | Epochs | BPB sliding | PPL sliding | Best val | Train Time | Train VRAM | Inference VRAM (strategies) | Run Log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adagrad (T2 ref, 1ep) | 0.01 | 1e-6 | — | 2e-13 | — | — | — | 1 | 1.1541 | 36.7905 | 3.5881 | ~1.86h | 7,788 MiB | 3,258 MiB | [link](logs/wikitext-103_2026-05-10_03-39-43/log.txt) |
| Adagrad (T2 ref, 5ep) | 0.01 | 1e-6 | — | 2e-13 | — | — | — | 5 | 1.0485 | 26.4564 | 3.2630 | ~8.92h | 7,788 MiB | 3,238 MiB | [link](logs/wikitext-103_2026-05-10_05-33-24/log.txt) |
| Muon (defunct ※) | 0.001 | 0.1 | 0.95 | 1e-7 | 5 | (3.4445, -4.775, 2.0315) | original | 1 (cancelled 47%) | — | — | val=4.1243 vs T2 Adagrad's 3.9342 at matched step | partial | 7,788 MiB | — | [link](logs/wikitext-103_2026-05-10_15-49-10/log.txt) |
| Muon (over-aggressive ✗) | 0.01 | 0.1 | 0.95 | 1e-7 | 5 | (3.4445, -4.775, 2.0315) | original | 1 (cancelled 30%) | — | — | plateau in 4.72–4.77 band from step ~5000 | partial | 7,788 MiB | — | [link](logs/wikitext-103_2026-05-10_17-45-55/log.txt) |
| **Muon (queued)** | **0.003** | 0.1 | 0.95 | 1e-7 | 5 | (3.4445, -4.775, 2.0315) | original | 1 | queued | queued | queued | queued | queued | queued | queued |
| **Muon (queued)** | **0.005** | 0.1 | 0.95 | 1e-7 | 5 | (3.4445, -4.775, 2.0315) | original | 1 | queued | queued | queued | queued | queued | queued | queued |

※ **lr=0.001 under-scaled.** With `adjust_lr_fn="original"` (API default), `torch.optim.Muon` scales LR by `max(1, sqrt(A/B))` — for 2048×2048 that's 1.0, no amplification. The 0.001 default is calibrated for `match_rms_adamw` (~9–29× scaling for our matrices); effective LR was 10–50× below reference Muon recipes. Trains, but ~0.19 nats behind T2 at matched step.

✗ **lr=0.01 over-aggressive.** Strong early lead through step ~6000 (0.18–0.39 nats ahead), then plateau/oscillation in 4.72–4.77 val band from step ~5000 — classic too-high-LR signature. Cancelled at 30%; lr ∈ {0.05, 0.10, 0.20} skipped as guaranteed worse. The lr=0.003 / lr=0.005 sweep splits the band between under-scaled (0.001) and over-aggressive (0.01).

**Compute-justified-only criterion.** Muon's per-step cost is ~2× Adagrad's. To justify keeping Muon, it must clear T2 Adagrad's 1ep best val (3.5881) by enough that matched-compute favors it; otherwise Adagrad stays.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Sequential Block Ordering

Test whether visiting every token in corpus order (stride = `block_size`, no overlap, no shuffling, document boundaries ignored) helps as a standalone feature. Prerequisite for [2D Wavelet over (Batch, Token)](#2d-wavelet-over-batch-token-with-sequential-training); convenient (not strictly required) for efficient [BBCE](#bisected-block-context-extension) compression.

**Results:**

| Run | MBS | GA | LR | Epochs | BPB sliding | PPL sliding | Best val | Train Time | Train VRAM | Inference VRAM (strategies) | Run Log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T2 random sampling | 8 | 1 | 0.01 | 1 | 1.1541 | 36.79 | 3.5881 | ~1.86h | 7,788 MiB | 3,258 MiB | [link](logs/wikitext-103_2026-05-10_03-39-43/log.txt) |
| T2 random sampling | 8 | 1 | 0.01 | 5 | 1.0485 | 26.46 | 3.2630 | ~8.92h | 7,788 MiB | 3,238 MiB | [link](logs/wikitext-103_2026-05-10_05-33-24/log.txt) |
| T2 sequential | 8 | 1 | 0.01 | 1 | 1.1726 | 38.98 | 3.6601 | ~1.85h | 8,065 MiB | 3,258 MiB | [link](logs/wikitext-103_2026-05-10_19-36-28/log.txt) |
| T2 sequential | 8 | 1 | 0.01 | 2 | 1.1146 | 32.52 | 3.5072 | ~3.64h | 8,065 MiB | 3,258 MiB | [link](logs/wikitext-103_2026-05-10_21-29-38/log.txt) |
| T2 sequential | 1 | 8 | 0.01 | 1 | 1.1901 | 41.17 | 3.6503 | ~4.18h | 7,662 MiB | 3,166 MiB | [link](logs/wikitext-103_2026-05-11_01-09-59/log.txt) |
| **T2 sequential + lr=0.015** | **8** | **1** | **0.015** | **1** | **1.1580** | **37.25** | **3.6231** | **~1.91h** | **8,065 MiB** | **3,258 MiB** | [link](logs/wikitext-103_2026-05-11_10-14-25/log.txt) |
| T2 sequential + 2D internal | 8 | 1 | 0.01 | 1 | 1.1765 | 39.47 | 3.6691 | ~2.11h | 8,269 MiB | 3,398 MiB | [link](logs/wikitext-103_2026-05-11_08-05-10/log.txt) |
| T2 sequential + 2D subband | 8 | 1 | 0.01 | 1 | 1.1939 | 41.66 | 3.7199 | ~2.40h | 8,990 MiB | 3,514 MiB | [link](logs/wikitext-103_2026-05-11_13-00-11/log.txt) |

**Findings.** Sequential at lr=0.01 underperforms random (+0.0720 best val, +0.0185 BPB). **Sequential + lr=0.015 recovers ~50% of the gap** (best val 3.6231 vs 3.5881; remaining Δ +0.0350). The second-epoch gain (Δ −0.1529 between 1ep and 2ep) confirms sequential is a viable substrate for downstream features that require it (BBCE caching, longer-context, PG-19 2D revisit). **2D wavelet modes regressed**: "internal" +0.0090 vs sequential baseline (~6× noise, +14% wall-clock); "subband" +0.0598 (~40× noise, +16% params, +30% wall-clock). Both shelved — Wikipedia articles are largely independent so cross-batch temporal structure isn't present on WikiText-103; PG-19 (multi-book dependencies) is the natural revisit. Code preserved at [tools/two_d_wavelets.py](tools/two_d_wavelets.py); runs.sh entries commented out.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Shelved on WikiText-103) 2D Wavelet over (Batch, Token) with Sequential Training

Generalize the lifting wavelet from 1D (token axis) to 2D (joint batch-token axis), requiring sequential batch processing so cross-batch temporal structure is preserved (e.g., PG-19 multi-book plot dependencies). Two variants tested at 1ep on T2 + sequential WikiText-103: `"internal"` (+6% params, same output shape) and `"subband"` (+16% params, 4 sub-bands per joint level exposed to per-band mixers) — both regressed (3.6691 / 3.7199 vs 3.6601 best val) at +14% / +30% wall-clock. Wikipedia articles are largely independent at chunk level, so the cross-batch structure 2D decomposition needs isn't there. **PG-19** (long-form novels) is the natural revisit. Design: [plans/two_d_wavelet_sequential_training.md](plans/two_d_wavelet_sequential_training.md); code: [tools/two_d_wavelets.py](tools/two_d_wavelets.py).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Bisected Block Context Extension

Inspired by [DeepSeek-V4](https://huggingface.co/collections/deepseek-ai/deepseek-v4): take the most recent `block_size/2` slots as uncompressed input; use the other `block_size/2` slots to hold per-channel means of `g = (block_size_compressed − block_size/2) / (block_size/2)` consecutive corpus tokens. The half-block-size seam aligns with every wavelet scale's dyadic partitions, giving O(log block_size) seam-bridging predict/update ops. Sweep across `block_size × block_size_compressed ∈ {256, 512, 1024, 2048} × {65K, 131K, 250K}` was queued; partially completed before project closure on 2026-05-13.

**Findings (sweep partial).** Only **bs=256 / bc=65K (g=511)** measurably beat T2 baseline on Best Val (Δ −0.0156, ~10× the 0.0015-nat noise threshold); on test-set BPB it was statistically tied (Δ +0.0010). Increasing `bc` at fixed `bs ≥ 512` regressed in every measured case (bs=512: 3.5922 → 3.6364 → 3.6247 across bc ∈ {65K, 131K, 250K}); increasing `bs` at fixed `bc=65K` monotonically regressed (3.5725 → 3.5922 → 3.6541). Reproducibility was excellent (Δ 0.0011 between bs=2048/bc=65K reruns). The g-matched diagonal that would disambiguate `bs` vs `g` was not completed. Tentative read: BBCE may help in the *high-g, small-bs* regime where compressed slots carry topic-vector-like density; the "more bc monotonically helps" hypothesis was not supported.

Note: the results below do not include a sliding window BPB.

| Config | Source | Best val (BBCE) | BPB (`evaluate_bbce`) | PPL | BPT | Params | LR | Epochs | Train VRAM |
|---|---|---|---|---|---|---|---|---|---|
| bs=256, bc=65K, g=511 | [log](logs/wikitext-103_2026-05-11_17-25-07/log.txt) / [bench](logs/wikitext-103_2026-05-11_17-25-07/benchmark.txt) | 3.8018 | 1.2158 | 44.6146 | 5.4794 | 392.91M | 0.01 | 1 | 7,809 MiB |

**Methodological contributions preserved for future researchers** (architecture outcome aside):

1. **Active-stride epoch definition** — count `1 epoch` as one full pass over *supervised positions* (stride `block_size/2` under bisected-context, `block_size` otherwise). Makes epoch counts comparable across compressed-context schemes.
2. **g-matched comparison framework** — hold `g = tokens-per-compressed-slot` constant, not `bc`. The g axis determines per-slot information density; varying bc at fixed bs conflates two effects.
3. **HF-style sliding-window benchmark for bisected-context models** (`evaluate_bbce`) — stride = `block_size/2`, score supervised half only, explicit padded-window accounting. Directly comparable to standard sliding-window BPB in the literature.
4. **Explicit dataset-ceiling documentation** — for WikiText-103, the val cliff at bc=251K and padding-pressure ramp to bc=287K must be documented to distinguish "didn't help" from "couldn't measure cleanly." **PG-19** (~28M test tokens, multi-book structure) is the natural scale-up for bc > 256K.

Full sweep tables, the `bbce_compressed_grad` toggle, and the bc=1M OOM analysis are preserved in git history at the project-closure commit; the code path lives in `tools/bbce.py`.


<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Adagrad Learning Rate Tuning

T2's default `lr=0.01` was inherited from earlier baselines. Sequential block ordering surfaced that lr=0.015 recovered ~50% of the sequential-vs-random gap; the follow-up was whether this generalizes to T2 random.

**Isolation test (T2 random + lr=0.015, complete).** Same configuration as T2 reference, only LR bumped from 0.01 → 0.015 (with `min_lr` scaled to 0.0003). Run log: [logs/wikitext-103_2026-05-11_15-26-31/log.txt](logs/wikitext-103_2026-05-11_15-26-31/log.txt).

| Variant | Best val | BPB sliding | PPL sliding | Train time | Train VRAM | Inference VRAM |
|---|---|---|---|---|---|---|
| T2 baseline (lr=0.01) | 3.5881 | 1.1541 | 36.79 | 1.83h | 7,788 MiB | 3,258 MiB |
| T2 + lr=0.015 | **3.5345** | **1.1362** | **34.79** | 1.84h | 8,065 MiB | 3,258 MiB |
| Δ | **−0.0536** | **−0.0179** | **−2.00** | +0.6% | +3.6% | (matched) |

Δ best val of **−0.0536 nats is ~36× the noise threshold** — unambiguous win at near-zero compute cost. **T3 = T2 + lr=0.015** is the new baseline (or **T3 = T2 + lr=0.015 + BBCE** if BBCE shows a stack-on win). A lr=0.020 canary is queued to confirm 0.015 isn't undershooting; at sufficiently high LR Adagrad's accumulator dynamics may change qualitatively. The BBCE sweep stays at lr=0.01 for apples-to-apples comparison; winners get re-run at the locked LR as part of T3 consolidation.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) New T3 Baseline

Establishing a new baseline of T3 using lr = 0.015 (without the BBCE). Comparison table:

| Variant | Best val | BPB sliding | PPL sliding | Train time | Train VRAM | Inference VRAM |
|---|---|---|---|---|---|---|
| T2 baseline (lr=0.01) | 3.5881 | 1.1541 | 36.79 | 1.83h | 7,788 MiB | 3,258 MiB |
| T3 = T2 + lr=0.015 | **3.5345** | **1.1362** | **34.79** | 1.84h | 8,065 MiB | 3,258 MiB |
| Δ | **−0.0536** | **−0.0179** | **−2.00** | +0.6% | +3.6% | (matched) |

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Recurrence with Adagrad (partial)

Due to wavelet decomposition and reconstruction being inverses of each other, and FWHT being its own inverse, one form of recurrence in WaveletLM only requires repeating the mixer operation. In other words, N steps of recurrence would look like:

`x → Decompose → FWHT → Mixer₁ → Mixer₂ → ... → Mixer_N → iFWHT → Reconstruct → x'`

Three parameters control the sweep:

- **N = `mixer_recurrence_steps`** — outer-loop count.
- **K = `mixer_recurrence_distinct_mixer_count`** — distinct per-scale mixer banks per cycle.
- **`mixer_recurrence_residuals`** (bool, default `true`) — add a residual `X + mixer(X)` at every recurrent step to prevent representation collapse over many applications. Only applied when N×K > 1; at the default N=K=1 the residual is skipped to preserve baseline behavior.

**Semantics** (nested loops): one cycle applies bank 0, bank 1, …, bank K−1 in sequence, and the cycle repeats N times. Total mixer applications per block = **N × K**.

| (N, K) shape | Meaning | Param cost | Total applications |
|---|---|---|---|
| K = 1 | One shared bank reused N times | none | N |
| K > 1 | K independent banks; sequence repeats N times | (K − 1)× mixer-only params | N × K |

Mutually exclusive with `untied_reconstruction` (recurrence relies on `Reconstruct ∘ Decompose = I`). K > 1 additionally requires `mixer_depth == 1` (banks are only allocated for the depth-1 mixer path); N > 1 works at any depth — the depth cascade itself is wrapped in the N-loop.

**Wall-clock cost.** The mixer is roughly **~55%** of per-block forward+backward compute at T2, so total time ≈ `(1 + (N·K − 1) · 0.55) × baseline`. T3 baseline at 1 epoch is ~1.84h:

| N | K | Total apps | Cost factor | Wall-clock (1ep, 5090) |
|---|---|---|---|---|
| 2 | 1 | 2 | ~1.55× | ~2.8h |
| 2 | 2 | 4 | ~2.65× | ~4.9h |
| 5 | 1 | 5 | ~3.20× | ~5.9h |
| 5 | 2 | 10 | ~5.95× | ~10.9h |
| 10 | 1 | 10 | ~5.95× | ~10.9h |
| 20 | 1 | 20 | ~11.45× | ~21h |

(On A5000, multiply each by roughly 1.9× — see A5000 vs 5090 throughput note in the BBCE/optimizer-sweep context.)

**Sweep (1 epoch each, ordered cost-ascending; reference row = T3 baseline at N=K=1).**

| Run | N | K | Mode | Total apps | BPB sliding | PPL sliding | Best val | Δ vs T3 | Train Time† | Train VRAM | Run Log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T3 baseline (N=1, K=1) | 1 | 1 | — | 1 | 1.1362 | 34.79 | 3.5345 | (ref) | 1.84h (5090) | 8,065 MiB | [link](logs/wikitext-103_2026-05-11_15-26-31/log.txt) |
| T3 + recur N=2 K=1 | 2 | 1 | shared | 2 | 1.1357 | 34.74 | 3.5294 | −0.0051 | 5.08h (A5000) | 7,789 MiB | [link](logs/wikitext-103_2026-05-16_10-03-55/log.txt) |
| T3 + recur N=2 K=2 | 2 | 2 | distinct | 4 | 1.1310 | 34.23 | 3.5162 | −0.0183 | 6.30h (A5000) | 8,910 MiB | [link](logs/wikitext-103_2026-05-16_15-25-50/log.txt) |
| T3 + recur N=5 K=1 | 5 | 1 | shared | 5 | NaN (step 8000) | — | 4.6052‡ | — | 5.98h (A5000) | 7,789 MiB | [link](logs/wikitext-103_2026-05-16_21-47-17/log.txt) |
| T3 + recur N=5 K=2 | 5 | 2 | cyclic | 10 | NaN (step 250) | — | — | — | killed | 8,910 MiB | [link](logs/wikitext-103_2026-05-17_03-48-49/log.txt) |

† Train time was highly affected by GPU. Baseline was trained on a 5090, but the other runs were on an A5000. May redo the baseline on an A5000 later for a clean comparison.
‡ Best val achieved before NaN onset; value is from mid-warmup and not comparable to a full run. 

**Decision rule.** A recurrence variant must clear T3 best val (3.5345) by more than the 0.0015-nat noise threshold to be considered a win.

**What each row tests.** The N ∈ {2, 5, 10, 20}, K=1 column probes the *fixed-point dynamics* of repeatedly applying the same mixer — does it converge to a useful attractor, oscillate, or collapse? Per-step residuals (`mixer_recurrence_residuals=true`) prevent representation collapse over many steps, mirroring the design of Universal Transformers / ALBERT. The N=2, K=2 distinct cell tests whether two fully-decoupled banks (4 total apps) behave as a "deeper architecture" or just over-parameterize. The N=5, K=2 cyclic cell is the cheap diversification: one extra bank, two-state recurrent structure across 10 apps.

Other recurrence approaches likely exist, but this section will only test the mixer.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Optimizer Swap (AdamW) and Wavelet Norms

Adagrad NaN'd at recurrence N ≥ 5 (step 8,000 for K=1; immediate for K=2), prompting the switch to AdamW. The sweep also revealed the entire wavelet path (decompose → FWHT → mixer → iFWHT → reconstruct) was unnormalized between `ln1` and `ln2`, allowing the mixer to feed unconstrained magnitudes into reconstruction. Two per-scale `LayerNorm(Cp)` modules — `wavelet_decomp_norm` (after decompose) and `wavelet_recon_norm` (after iFWHT) — extend the stable LR range (A3/A4 complete cleanly where they previously NaN'd) and accelerate early convergence, but shift the effective LR landscape: normed runs trail the T3 baseline across all LRs tested so far. A1+norms and A2+norms sweep lower LRs to find the optimum; A3 was run at both clip values to isolate grad_clip. **All prior ablation deltas were measured on the unnormed baseline and will need re-sweeping** once the optimal normed LR is confirmed.

**Divergence visibility.** Without norms, instability was a near-instantaneous event — a single gradient spike permanently corrupting AdamW's `v_t`. With norms, the A5 (lr=0.01) run shows ~6,750 steps of visible stagnation (val loss plateaued at ~5.37 while LR was still ramping through warmup) before a spike at step 15,000 and NaN at step 15,250. Norms convert the failure mode from a sudden catastrophic event to an observable gradual divergence: warmup stagnation is now a legible early-warning signal that the LR is in an untenable regime, and NaN is a lagging indicator rather than the primary event. LR sensitivity remains tight — small LR differences produce large quality gaps — but the training regime is now diagnosable rather than unpredictably brittle.

**Config:** T3 base (C=2048, L=1, levels=7, T2 mixer widths, `wavelet_crawl=true`), AdamW defaults (`betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`), `min_lr = lr / 50`, bf16 (A2 fp16 diverged from overflow); LRs span ±2 √10-steps around 0.001.

**LR sweep (1 epoch each, T3 architecture base):**

| Run | lr | min_lr | betas | eps | weight_decay | amsgrad | amp_dtype | grad_clip | wavelet_norms | BPB sliding | PPL sliding | Best val | Δ vs T3 | Train time | Run log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T3 baseline (Adagrad ref) | 0.015 | 0.0003 | — | — | — | — | fp16 | 1.0 | ✗ | 1.1362 | 34.79 | 3.5345 | (ref) | 1.84h (5090) | [link](logs/wikitext-103_2026-05-11_15-26-31/log.txt) |
| AdamW A1 | 0.0001 | 2e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | fp16 | 1.0 | ✗ | 1.1539 | 36.77 | 3.5822 | +0.048 | 4.7h (A5000) | [link](logs/wikitext-103_2026-05-17_04-46-42/log.txt) |
| AdamW A2 (fp16)† | 0.00031623 | 6.32e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | fp16 | 1.0 | ✗ | NaN† | NaN† | 4.0270‡ | — | 4.7h (A5000) | [link](logs/wikitext-103_2026-05-17_09-57-10/log.txt) |
| AdamW A2 (bf16) | 0.00031623 | 6.32e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✗ | 1.1495 | 36.26 | 3.5725 | +0.038 | 4.6h (A5000) | [link](logs/wikitext-103_2026-05-17_13-17-31/log.txt) |
| AdamW A3 (no norms, clip=1.0)§ | 0.001 | 2e-5 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✗ | NaN§ | NaN§ | 4.4307§ | — | — | [link](logs/wikitext-103_2026-05-17_17-53-14/log.txt) |
| AdamW A3 (no norms, clip=0.5)¶ | 0.001 | 2e-5 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 0.5 | ✗ | NaN¶ | NaN¶ | — | — | — | [link](logs/wikitext-103_2026-05-17_23-20-31/log.txt) |
| AdamW A4 (no norms)‖ | 0.0031623 | 6.32e-5 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 0.5 | ✗ | NaN‖ | NaN‖ | — | — | — | [link](logs/wikitext-103_2026-05-18_03-38-58/log.txt) |
| AdamW A3 (norms, clip=0.5) | 0.001 | 2e-5 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 0.5 | ✓ | 1.1891 | 41.05 | 3.6888 | +0.154 | 4.75h (A5000) | [link](logs/wikitext-103_2026-05-18_06-05-19/log.txt) |
| AdamW A3 (norms, clip=1.0) | 0.001 | 2e-5 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1729 | 39.02 | 3.6374 | +0.103 | 4.7h (A5000) | [link](logs/wikitext-103_2026-05-18_10-53-23/log.txt) |
| AdamW A4 (norms) | 0.0031623 | 6.32e-5 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.2273 | 46.24 | 3.7932 | +0.259 | 4.7h (A5000) | [link](logs/wikitext-103_2026-05-18_15-35-44/log.txt) |
| AdamW A5 (norms)♦ | 0.01 | 2e-4 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | NaN♦ | NaN♦ | — | — | — | [link](logs/wikitext-103_2026-05-18_20-19-39/log.txt) |
| AdamW A1 (norms)★ | 0.0001 | 2e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1554 | 36.94 | 3.5887 | +0.054 | 4.7h (A5000) | [link](logs/wikitext-103_2026-05-18_21-50-54/log.txt) |
| AdamW A1.25 (norms)◆ | 0.00013335 | 2.6670e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1445 | 35.71 | 3.5593 | +0.025 | 4.71h (A5000) | [link](logs/wikitext-103_2026-05-19_14-04-06/log.txt) |
| AdamW A1.375 (norms)◆ | 0.00015399 | 3.0798e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1419 | 35.41 | 3.5488 | +0.014 | 4.66h (A5000) | [link](logs/wikitext-103_2026-05-19_23-42-39/log.txt) |
| AdamW A1.5 (norms)◆ | 0.00017783 | 3.5566e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1393 | 35.13 | 3.5429 | +0.008 | 4.75h (A5000) | [link](logs/wikitext-103_2026-05-19_08-50-57/log.txt) |
| AdamW A1.75 (norms)◆ | 0.00023714 | 4.7428e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1393 | 35.13 | 3.5435 | +0.009 | 4.64h (A5000) | [link](logs/wikitext-103_2026-05-20_04-24-59/log.txt) |
| AdamW A2 (norms)★ | 0.00031623 | 6.32e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1394 | 35.15 | 3.5396 | +0.005 | 4.75h (A5000) | [link](logs/wikitext-103_2026-05-19_02-37-49/log.txt) |
| AdamW A2.25 (norms)◆ | 0.00042170 | 8.4340e-6 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1412 | 35.34 | 3.5465 | +0.012 | 4.65h (A5000) | [link](logs/wikitext-103_2026-05-20_09-05-23/log.txt) |
| AdamW A2.5 (norms)◆ | 0.00056234 | 1.1247e-5 | (0.9, 0.999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1428 | 35.51 | 3.5528 | +0.018 | 4.67h (A5000) | [link](logs/wikitext-103_2026-05-19_18-49-14/log.txt) |
| AdamW β₂=0.98 (norms)◇ | 0.00023714 | 4.7428e-6 | (0.9, 0.98) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.2685 | 52.59 | 3.9309 | +0.396 | 4.68h (A5000) | [link](logs/wikitext-103_2026-05-20_14-10-07/log.txt) |
| AdamW β₂=0.99 (norms)◇ | 0.00023714 | 4.7428e-6 | (0.9, 0.99) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.2062 | 43.30 | 3.7399 | +0.205 | 4.73h (A5000) | [link](logs/wikitext-103_2026-05-20_18-53-21/log.txt) |
| AdamW β₂=0.99986 (norms)◇ | 0.00023714 | 4.7428e-6 | (0.9, 0.99986) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1366 | 34.83 | 3.5324 | −0.002 | 4.72h (A5000) | [link](logs/wikitext-103_2026-05-21_05-08-36/log.txt) |
| AdamW β₂=0.99988 (norms)◇ | 0.00023714 | 4.7428e-6 | (0.9, 0.99988) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1365 | 34.82 | 3.5323 | −0.002 | 4.70h (A5000) | [link](logs/wikitext-103_2026-05-21_09-52-27/log.txt) |
| AdamW β₂=0.9999 (norms)◇ | 0.00023714 | 4.7428e-6 | (0.9, 0.9999) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1370 | 34.88 | 3.5310 | −0.003 | 4.71h (A5000) | [link](logs/wikitext-103_2026-05-20_23-39-18/log.txt) |
| AdamW β₂=0.99992 (norms)◇ | 0.00023714 | 4.7428e-6 | (0.9, 0.99992) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1367 | 34.85 | 3.5344 | 0.000 | 4.71h (A5000) | [link](logs/wikitext-103_2026-05-21_14-35-13/log.txt) |
| AdamW β₂=0.99994 (norms)◇ | 0.00023714 | 4.7428e-6 | (0.9, 0.99994) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1382 | 35.01 | 3.5342 | 0.000 | 4.75h (A5000) | [link](logs/wikitext-103_2026-05-21_19-18-41/log.txt) |
| AdamW β₁=0.0 (norms)✦ | 0.00023714 | 4.7428e-6 | (0.0, 0.99988) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1396 | 35.17 | 3.5358 | +0.001 | 4.72h (A5000) | [link](logs/wikitext-103_2026-05-22_05-38-12/log.txt) |
| AdamW β₁=0.85 (norms)✦ | 0.00023714 | 4.7428e-6 | (0.85, 0.99988) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1374 | 34.92 | 3.5307 | −0.004 | 4.69h (A5000) | [link](logs/wikitext-103_2026-05-22_10-22-19/log.txt) |
| AdamW β₁=0.875 (norms)✦ | 0.00023714 | 4.7428e-6 | (0.875, 0.99988) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1367 | 34.85 | 3.5320 | −0.003 | 4.71h (A5000) | [link](logs/wikitext-103_2026-05-22_15-05-03/log.txt) |
| AdamW β₁=0.925 (norms)✦ | 0.00023714 | 4.7428e-6 | (0.925, 0.99988) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1372 | 34.90 | 3.5313 | −0.003 | 4.73h (A5000) | [link](logs/wikitext-103_2026-05-22_19-49-14/log.txt) |
| AdamW β₁=0.95 (norms)✦ | 0.00023714 | 4.7428e-6 | (0.95, 0.99988) | 1e-8 | 0.01 | False | bf16 | 1.0 | ✓ | 1.1371 | 34.89 | 3.5321 | −0.002 | 4.73h (A5000) | [link](logs/wikitext-103_2026-05-23_00-34-13/log.txt) |
| AdamW amsgrad=True (norms)✧ | 0.00023714 | 4.7428e-6 | (0.9, 0.99988) | 1e-8 | 0.01 | True | bf16 | 1.0 | ✓ | 1.1367 | 34.85 | 3.5342 | +0.005 | 4.82h (A5000) | [link](logs/wikitext-103_2026-05-23_07-35-55/log.txt) |

† Benchmark invalid — permanent NaN from step 27,250 due to fp16 overflow corrupting `v_t`.  
‡ Best val before divergence (step 27,000); not comparable to completed runs.  
§ NaN from step 19,250 (gradient spike at peak LR; best pre-divergence val not comparable).  
¶ NaN from step 25,250 (`grad_clip=0.5` delayed onset ~30% vs clip=1.0 but did not prevent it).  
‖ NaN from step 11,250 (3.16× higher peak LR overwhelmed clip=0.5; earlier onset than A3 despite tighter clip).  
★ LR recalibration runs: A3+norms trailed T3 baseline after warmup, indicating the wavelet norms shift the effective loss landscape. A1+norms and A2+norms cover the lower-LR end of the sweep under the normalized architecture (bf16, clip=1.0). Original A1/A2 rows above are retained for comparison. 
◆ Bisection runs: Full sweep complete. A flat plateau spans lr=0.00017783–0.00031623 (A1.5 through A2), with BPB pinned at 1.1393–1.1394 across all three interior points (A1.5, A1.75, A2). Lower cliff: A1.375 (0.00015399, BPB 1.1419) sits just below; upper cliff: A2.25 (0.00042170, BPB 1.1412) sits just above. **A1.75 (lr=0.00023714) is selected as the optimal LR**: geometric centre of the plateau, BPB tied with A1.5/A2 within noise, most robust to LR shift across future sweeps.  
♦ Divergence visible from step ~8,250 (val loss stagnant at ~5.37 while LR ramped); spike at step 15,000, NaN at step 15,250 (lr ~8.84e-3). Norms extend the stable range (A3/A4 complete cleanly) but do not eliminate instability at lr=0.01.

◇ β₂ sweep: all runs at lr=0.00023714 (A1.75). Baseline β₂=0.999 (A1.75 row above, BPB 1.1393, val 3.5435). Results trend monotonically: lower β₂ regresses sharply (0.98: +0.129 BPB; 0.99: +0.067 BPB), while higher β₂ improves. Fine bisection around 0.9999 (±0.00002, ±0.00004) reveals a flat BPB plateau from 0.99986–0.99992 (BPB 1.1365–1.1367); above 0.99992 performance regresses (0.99994: BPB 1.1382, val back to T3 level). Val-optimal is β₂=0.9999 (val 3.5310); BPB-optimal is β₂=0.99988 (BPB 1.1365). **β₂=0.99988 selected** as the locked value for subsequent sweeps (BPB-primary metric).

✦ β₁ sweep: all runs at lr=0.00023714, β₂=0.99988. Reference β₁=0.9 (β₂=0.99988 row above, BPB 1.1365, val 3.5323). Sweep complete. BPB is flat from 0.875–0.95 (1.1365–1.1372); β₁=0.0 regresses on both metrics (BPB 1.1396, val 3.5358). Val-optimal is β₁=0.85 (3.5307) but BPB regresses slightly (+0.0009). **β₁=0.9 (PyTorch default) confirmed as BPB-optimal**; locked for subsequent sweeps.

✧ AMSGrad probe: single run at the locked config (lr=0.00023714, β₁=0.9, β₂=0.99988, amsgrad=True). AMSGrad replaces the second-moment EMA with a running max (v̂_t = max(v̂_{t-1}, v_t)), providing stronger convergence guarantees at the cost of slower forgetting. **Result: BPB 1.1367 — indistinguishable from T4 baseline (1.1365).** No meaningful improvement; AMSGrad discarded from further sweeps.

After the amsgrad probe, `eps` and `weight_decay` sweeps follow.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Optimizer Tuning (Adagrad) with Wavelet Norms

The T3 Adagrad reference ran on the un-normed architecture. Ag0 (norms at the T3 LR of 0.015) yields BPB 1.1332 — better than T3 (1.1362) and T4 AdamW (1.1365) with no LR retuning, showing norms help Adagrad in place. Ag1 (10× lower, lr=0.0015) collapses to BPB 1.3340, ruling out the lower end. The 15× AdamW conversion hypothesis is retired. A log-symmetric grid centred on 0.015 (×10, ×√10, ×∛10, ÷∛10, ÷√10; or until divergence) locates the normed Adagrad optimum. All normed rows use fp16, `wavelet_decomp_norm=true`, `wavelet_recon_norm=true`, eps=2e-13, weight_decay=1e-6, grad_clip=1.0.

**LR tuning**

| Run | lr | min_lr | BPB sliding | PPL sliding | Best val | Δ vs T3 | Train time | Run log |
|---|---|---|---|---|---|---|---|---|
| T3 baseline (Adagrad, no norms, ref) | 0.015000 | 3e-4 | 1.1362 | 34.79 | 3.5345 | (ref) | 1.84h (5090) | [link](logs/wikitext-103_2026-05-11_15-26-31/log.txt) |
| Adagrad Ag1 + norms (lr=0.0015, ÷ 10) | 0.001500 | 3e-5 | 1.3340 | 64.52 | 4.1472 | +0.198 | 4.77h (A5000) | [link](logs/wikitext-103_2026-05-23_17-11-47/log.txt) |
| Adagrad Ag ÷√10 + norms (lr=0.004743) | 0.004743 | 9.486e-5 | 1.1953 | 41.85 | 3.7071 | +0.059 | 4.76h (A5000) | [link](logs/wikitext-103_2026-05-23_22-26-07/log.txt) |
| Adagrad Ag ÷∛10 + norms (lr=0.006963) | 0.006963 | 1.393e-4 | 1.1648 | 38.04 | 3.6168 | +0.029 | 4.72h (A5000) | [link](logs/wikitext-103_2026-05-24_03-12-28/log.txt) |
| Adagrad Ag0 + norms (lr=0.015, same as T3) | 0.015000 | 3e-4 | 1.1332 | 34.47 | 3.5273 | −0.003 | 4.75h (A5000) | [link](logs/wikitext-103_2026-05-23_12-25-37/log.txt) |
| Adagrad Ag ×1.25 + norms (lr=0.01875) | 0.01875 | 3.75e-4 | 1.1319 | 34.33 | 3.5194 | −0.004 | 4.77h (A5000) | [link](logs/wikitext-103_2026-05-24_14-35-21/log.txt) |
| Adagrad Ag ×1.50 + norms (lr=0.02250)★ | 0.02250 | 4.50e-4 | **1.1311** | **34.24** | **3.5157** | **−0.005** | 4.78h (A5000) | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| Adagrad Ag ×1.75 + norms (lr=0.02625) | 0.02625 | 5.25e-4 | 1.1328 | 34.42 | 3.5210 | −0.003 | 4.75h (A5000) | [link](logs/wikitext-103_2026-05-25_00-10-39/log.txt) |
| Adagrad Ag ×∛10 + norms (lr=0.032316)✶ | 0.032316 | 6.463e-4 | NaN✶ | NaN✶ | 4.0763✶ | — | 4.60h (A5000) | [link](logs/wikitext-103_2026-05-24_07-57-45/log.txt) |

✶ Late-training divergence: best_model.pt was saved cleanly (val 4.0763 before spike), but NaN contaminated the logits for some benchmark windows — BPB unmeasurable. Text generation remains functional from that checkpoint.

★ New LR optimum. All three fine-grained runs beat Ag0: ×1.25 (BPB 1.1319, −0.001), ×1.50 (BPB 1.1311, −0.002), ×1.75 (BPB 1.1328, 0.000). The curve peaks cleanly at ×1.50 (lr=0.02250) — symmetric regression on both sides — with the stability cliff confirmed between ×1.75 (stable, BPB 1.1328) and ×∛10 (diverges, NaN). ×√10 and ×10 remain cancelled.

**LR Tuning Findings:**

Below-baseline LRs regress monotonically (÷10: +0.198 BPB; ÷√10: +0.059; ÷∛10: +0.029). Above-baseline, all three fine-grained runs improve: the optimum is a clean single peak at **lr=0.02250 (Ag150, BPB 1.1311)**, with symmetric regression on both sides and a sharp stability cliff between lr=0.02625 and 0.032316. LR sweep complete.

**Other parameter tuning**

All runs use the locked Ag0 config as base (lr=0.015, min_lr=3e-4, fp16, wavelet norms, T3 architecture). One parameter is varied per run; all others held at Ag0 defaults (eps=2e-13, initial_accumulator_value=0, weight_decay=1e-6). Δ is BPB vs Ag0 (1.1332).

| Run | eps | initial_acc | weight_decay | BPB sliding | PPL sliding | Best val | Δ vs Ag0 | Train time | Run log |
|---|---|---|---|---|---|---|---|---|---|
| Ag0 (reference) | 2e-13 | 0 | 1e-6 | 1.1332 | 34.47 | 3.5273 | (ref) | 4.75h (A5000) | [link](logs/wikitext-103_2026-05-23_12-25-37/log.txt) |
| Adagrad initial_acc=0.1 | 2e-13 | 0.1 | 1e-6 | 1.7136 | 211.30 | 5.3393 | +0.5804 | 4.68h (A5000) | [link](logs/wikitext-103_2026-05-25_04-56-16/log.txt) |
| Adagrad initial_acc=1.0 | 2e-13 | 1.0 | 1e-6 | 1.9636 | 461 | 6.1071 | +0.8304 | 4.68h (A5000) | [link](logs/wikitext-103_2026-05-25_09-39-20/log.txt) |
| Adagrad eps=1e-10 (PyTorch default) | 1e-10 | 0 | 1e-6 | 1.1328 | 34.43 | 3.5277 | −0.0004 | 5.27h (A5000) | [link](logs/wikitext-103_2026-05-25_14-21-20/log.txt) |
| Adagrad eps=1e-8 | 1e-8 | 0 | 1e-6 | 1.1332 | 34.47 | 3.5271 | 0.0000 | 4.86h (A5000) | [link](logs/wikitext-103_2026-05-25_19-39-37/log.txt) |
| Adagrad weight_decay=0 | 2e-13 | 0 | 0 | 1.1400 | 35.20 | 3.5415 | +0.0068 | 4.62h (A5000) | [link](logs/wikitext-103_2026-05-26_00-33-17/log.txt) |
| Adagrad weight_decay=1e-4 | 2e-13 | 0 | 1e-4 | 1.3010 | 58.27 | 4.0522 | +0.1678 | 4.63h (A5000) | [link](logs/wikitext-103_2026-05-26_05-12-35/log.txt) |

**Other Tuning Findings:**

`initial_acc` degrades monotonically: 0.1 → +0.58 BPB, 1.0 → +0.83 BPB. With essentially-zero epsilon, Ag0's initial accumulator=0 allows an effectively unbounded first-step learning rate (bounded only by the warmup schedule), enabling rapid early adaptation. Any positive initial_acc pre-fills the denominator, capping the effective LR to `lr/√initial_acc` from the start and suppressing the adaptive advantage precisely where it matters most. Rule: **never use initial_acc > 0 with eps < 1e-8**.

`eps` is essentially inert across 5 orders of magnitude with `initial_acc=0`: eps ∈ {2e-13, 1e-10, 1e-8} all give BPB 1.1328–1.1332 (Δ ≤ 0.0004, within run-to-run noise). With initial_acc=0, the early-step accumulator is dominated by the gradient squared (which is >>eps for any reasonable eps), so the choice of eps doesn't bound the effective LR until much later in training when the accumulator has grown large enough that the additive eps becomes negligible regardless. The PyTorch default (1e-10) and the textbook-conservative 1e-8 are both safe; the locked Ag0 baseline at 2e-13 is overconservative but harmless. Sweep can be retired.

`weight_decay` has a real but narrow sweet spot at the Ag0 default of 1e-6. Both extremes regress: `wd=0` is +0.0068 BPB (mild — the model still trains, just overfits slightly more without the decay-driven regularization on the wavelet-norm-scaled weights), `wd=1e-4` is +0.1678 BPB (severe — 100× more aggressive decay underfits, val loss climbing to 4.05 vs Ag0's 3.53). The asymmetry (wd=0 mild, wd=1e-4 severe) suggests the optimum is closer to the current 1e-6 than to 0 — a follow-up sweep at {3e-7, 1e-6, 3e-6} could tighten this, but the lift is bounded by the wd=0 gap of 0.0068. Lower priority than other levers; sweep effectively closed at the existing baseline.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Spectral Norm

Adding for stability and testing its performance.

All runs use the locked Ag0 config as base (lr=0.015, min_lr=3e-4, fp16, wavelet norms, T3 architecture, Adagrad eps=2e-13, initial_acc=0, weight_decay=1e-6). `stab_spectral_norm` constrains the GatedSpectralMixer's dense routing matrix to σ₁(W) ≤ 1, preventing singular-value-driven amplitude growth. Δ is BPB vs Ag0 (1.1332).

| Run | lr | stab_spectral_norm | BPB sliding | PPL sliding | Best val | Δ vs Ag0 | Train time | Run log |
|---|---|---|---|---|---|---|---|---|
| Ag0 + norms (no SN, ref) | 0.015 | ✗ | 1.1332 | 34.47 | 3.5273 | (ref) | 4.75h (A5000) | [link](logs/wikitext-103_2026-05-23_12-25-37/log.txt) |
| SN1: Ag0 + SN (lr=0.015) | 0.015 | ✓ | 1.1329 | 34.43 | 3.5245 | −0.0003 | 4.74h (A5000) | [link](logs/wikitext-103_2026-05-26_09-53-04/log.txt) |
| SN2: Ag0 + SN (lr=0.032316, prev. NaN) | 0.032316 | ✓ | 1.1337 | 34.50 | 3.5230 | +0.0005 | 4.77h (A5000) | [link](logs/wikitext-103_2026-05-26_14-40-09/log.txt) |
| SN3✶: Ag0 + SN (lr=0.15, extreme) | 0.15 | ✓ | 13.1521✶ | —✶ | 5.4301✶ | +12.02✶ | 4.07h (A5000) | [link](logs/wikitext-103_2026-05-26_19-28-42/log.txt) |

**Spectral Norm Findings:**

`stab_spectral_norm` is a **stability lever, not a performance lever**. At the Ag0 baseline LR (0.015), SN1 reproduces Ag0 within noise (−0.0003 BPB) — the σ₁ ≤ 1 constraint on the GatedSpectralMixer's routing matrix doesn't materially restrict learning at this LR, but it also doesn't add anything. The real value shows at higher LRs: SN2 (lr=0.032316) successfully trains where the unconstrained Adagrad Ag ×∛10 + norms variant NaN'd (4.0763 val loss, never recovered). Spectral norm extends the stable-LR range upward by roughly one log-spaced step (×∛10 ≈ 2.15×).

But the achieved BPB at SN2 (1.1337) is *worse* than Ag0's baseline (1.1332). Pushing further to lr=0.15 (×10), SN3 still diverges — spectral norm has limits. The Pareto frontier of (stability, PPL) is shaped such that **no SN-enabled LR beats T4's lr=0.02250 + no SN at BPB 1.1311**. Spectral norm is best understood as an insurance mechanism for aggressive LR settings rather than a default-on configuration improvement.

✶ SN3 diverged early — best val settled at 5.4301 (vs Ag0's 3.5273) and never recovered; BPB / PPL are reported as-is for the record but reflect a broken model, not a meaningful comparison.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) New T4 Baseline

T4 = T3 architecture + wavelet norms (`wavelet_decomp_norm` + `wavelet_recon_norm`) + tuned hyperparameters for the best optimizer. LR sweep complete — optimum locked at **lr=0.02250** (Ag150, BPB 1.1311), a clean single peak confirmed by the fine-grained ×1.25/×1.50/×1.75 sweep. T4 updated accordingly.

| Variant | Best val | BPB sliding | PPL sliding | Train time | Train VRAM | Inference VRAM | Run log |
|---|---|---|---|---|---|---|---|
| T3 (Adagrad, lr=0.015, no wavelet norms, fp16) | 3.5345 | 1.1362 | 34.79 | 1.84h | 8,065 MiB | 3,258 MiB | [link](logs/wikitext-103_2026-05-11_15-26-31/log.txt) |
| T4 (Adagrad, lr=0.02250, wavelet norms, fp16) | **3.5157** | **1.1311** | **34.24** | 4.78h (A5000) | 7,790 MiB | ≈3,096 MiB ✶ | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| Δ vs T3 | **−0.019** | **−0.005** | **−0.55** | +2.60× | −275 MiB | ≈−162 MiB ✶ | — |

✶ T4 inference VRAM is the post-fix safety-net estimate (`torch.cuda.max_memory_allocated()` + ~750 MiB CUDA context). The pod's PID namespace prevents nvidia-smi from directly attributing memory to the generate.py process, so the precise number isn't measurable in this environment; the T3 row's 3,258 MiB is the real nvidia-smi reading from before the pod migration. All other post-2026-05-13 T3/T4-architecture runs (Adagrad parameter sweep, spectral norm sweep, recurrence) measure ≈3,096 MiB via the same estimator — the architecture is shared, so per-table Inference VRAM columns would be redundant and aren't added.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Recurrence with Adagrad (no residual)

> **What "no residual" means here.** These runs had `mixer_recurrence_residuals=true`, so each step applied the *step-local* residual `Xₜ = Xₜ₋₁ + m(Xₜ₋₁)`. But the residual was **incomplete**: it never re-injected the initial input X⁰, so the recurrence was a shared-weight (or weight-cycled) residual stack with no anchor to the input. Mathematically it had nowhere to converge — each step transforms the previous step's output, so more steps = more drift, which is why **depth past N=2 regresses** (see findings). The [with-residual section](#recurrence-with-adagrad-with-residual) corrects this by re-injecting X⁰ at every step (input-anchored iteration). The numbers below stand as the no-anchor baseline.

Full recurrence sweep using the locked T4 baseline (Adagrad, lr=0.02250, min_lr=4.50e-4, wavelet norms enabled, eps=2e-13, initial_acc=0, weight_decay=1e-6, fp16). T4 supersedes T3 as the reference here because the earlier [Recurrence (Adagrad, partial)](#recurrence-adagrad-partial) attempts NaN'd at N ≥ 5 — wavelet norms extend the stable range, and the tuned LR is materially better than T3 (Δ −0.005 BPB at N=K=1). Stability headroom is the prerequisite for pushing N deep; SN is available as an optional add-on if any specific cell diverges, but is **off** for the canonical sweep so results stay comparable to the T4 reference.

**Sweep (1 epoch each, ordered cost-ascending; reference row = T4 baseline at N=K=1). Decision rule: a recurrence variant must clear T4 best val (3.5157) by > 0.0015 (noise threshold) to be a win.**

| Run | N | K | Mode | Total apps | BPB sliding | PPL sliding | Best val | Δ vs T4 | Train Time | Train VRAM | Run Log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T4 baseline (N=1, K=1) | 1 | 1 | — | 1 | 1.1311 | 34.24 | 3.5157 | (ref) | 4.78h (A5000) | 7,790 MiB | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| T4 + recur N=1 K=2 | 1 | 2 | distinct | 2 | 1.1249 | 33.60 | 3.4973 | −0.0062 | 5.66h (A5000) | 8,912 MiB | [link](logs/wikitext-103_2026-05-27_20-39-29/log.txt) |
| T4 + recur N=2 K=1 | 2 | 1 | shared | 2 | 1.1279 | 33.91 | 3.5070 | −0.0032 | 5.46h (A5000) | 7,790 MiB | [link](logs/wikitext-103_2026-05-28_02-21-45/log.txt) |
| T4 + recur N=2 K=2 | 2 | 2 | distinct | 4 | **1.1227** | **33.37** | **3.4904** | **−0.0084** | 6.76h (A5000) | 9,028 MiB | [link](logs/wikitext-103_2026-05-28_07-52-14/log.txt) |
| T4 + recur N=5 K=1 | 5 | 1 | shared | 5 | 1.1291 | 34.04 | 3.5112 | −0.0020 | 7.31h (A5000) | 8,932 MiB | [link](logs/wikitext-103_2026-05-28_14-41-00/log.txt) |
| T4 + recur N=5 K=2 | 5 | 2 | cyclic | 10 | 1.1275 | 33.87 | 3.5086 | −0.0036 | 10.40h (A5000) | 11,814 MiB | [link](logs/wikitext-103_2026-05-28_22-02-41/log.txt) |'

**Inter-step stabilization (auto, N·K > 1).** A per-scale `LayerNorm(Cp)` is applied **between** mixer applications inside the recurrent loop (N·K − 1 invocations per forward; the final step is left unnormalized so `wavelet_recon_norm` after iFWHT stays the sole boundary norm). It prevents fp16 overflow / divergence in the recurrence — observed in two modes before the fix: distinct banks (N=2 K=2) NaN'd at step 750, and depth (N=5 K=1) overflowed at step 1 (loss=nan at lr=0) then diverged. Five residual-mixer steps on ~√Cp-amplified post-FWHT coefficients exceed fp16's 65504 ceiling without it. Originally scoped to K>1; broadened to **N·K > 1** after the N=5 K=1 divergence showed depth needs it too. Auto-instantiated, no config flag; the N=1 K=1 baseline is unaffected. Compute cost is negligible (memory-bound LayerNorm).

**Consistent norm regime.** All rows below were (re-)run with the same N·K>1 inter-step norm and seed 1337, so they are directly comparable. The factor-isolation design at fixed compute (2 mixer applications): **N=2 K=1** (shared bank, +0 params) vs **N=1 K=2** (two distinct banks, +58.85M params) isolates the pure value of parameter diversity; **N=2 K=2** adds depth×diversity. Decision rule: a variant must clear T4 best val (3.5157) by > 0.0015 (noise threshold) to count as a win — all five clear it.

**Findings:**

All five re-runs share seed 1337 and the N·K>1 inter-step norm, so the comparison is clean. Two results overturn the going-in hypothesis (which expected log-scaling improvement with depth N):

- **Diversity (K) beats depth (N), decisively.** Every K=2 run beats every K=1 run on BPB — the *worst* K=2 run (N=5 K=2, 1.1275) edges the *best* K=1 run (N=2 K=1, 1.1279). Parameter diversity across distinct mixer banks is the real lever; iterating a shared bank is the weaker one.
- **The fixed-compute isolation is conclusive.** At exactly 2 mixer applications: **N=1 K=2** (distinct banks, +58.85M params) hits 1.1249 / val 3.4973, vs **N=2 K=1** (shared, +0 params) at 1.1279 / val 3.5070. The diversity configuration wins by −0.0030 BPB / −0.0097 val at identical compute — the second bank's parameters do real work that shared-weight repetition does not replicate.
- **Depth past N=2 regresses.** Within both K values, going N=2 → N=5 *hurts*: K=1 worsens 1.1279 → 1.1291, K=2 worsens 1.1227 → 1.1275. And N=1 K=2 (1.1249) beats N=5 K=2 (1.1275). There is no log-scaling-up-with-depth regime here; recurrence peaks at low N and declines — likely harder optimization under a 1-epoch budget and/or representational drift the residual+norm only partly contain.
- **Best absolute: N=2 K=2** (1.1227, val 3.4904, Δ −0.0084 BPB / −0.0253 val vs T4) — but it costs +58.85M params *and* 4 apps (6.76h). **Best efficiency: N=1 K=2** (1.1249, 2 apps, 5.66h) captures ~75% of the BPB gain at the same param cost and far less compute.
- **Implication for the deep sweep:** N=10 K=1 and N=20 K=1 are very likely to regress further (K=1 already worsens from N=2 to N=5, and depth is the losing axis). **Recommend cancelling both** per the cost-ascending sweep's "cancel if depth plateaus/regresses" rule — they would cost ~28h and ~55h respectively to confirm a decline the N=5 K=1 point already establishes. Compute is better spent on the parameter-diversity axis (higher K at low N) or the still-open param-matched comparison below.

**Still open — is the diversity win worth its params vs. the primary lever?** N=1 K=2 buys −0.0062 BPB for +58.85M params. The established primary capacity lever is MLP expansion; the decisive comparison is a **param-matched single-pass T4** (`mlp_expansion` ~10→17, ≈+59M, no recurrence) against N=1 K=2. If the param-matched MLP matches or beats it, the params are better spent on width; if N=1 K=2 wins, the distinct-bank structure adds something raw capacity does not.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Recurrence with Adagrad (with residual)

Re-runs the recurrence sweep under the **corrected, input-anchored residual**. The fix (in `model.py`, folded into the already-`true` `mixer_recurrence_residuals` — no new flag) re-injects the initial post-FWHT spectrum X⁰ at the input of every mixer application *after the first*, in addition to the running-state residual:

$$X^{(t)} = \mathrm{LN}\!\left(\tilde{X}^{(t)} + m\big(\tilde{X}^{(t)}\big)\right), \qquad \tilde{X}^{(t)} = \begin{cases} X^{(0)} & t = 1 \\ X^{(t-1)} + X^{(0)} & t \geq 2 \end{cases}$$

The first step is the plain residual (the running state already equals X⁰ there, so re-adding it would double-count to 2·X⁰); injection starts at step 2. From there X⁰ feeds the mixer, the cross-scale gate routing, and the residual base at every step; it accumulates in the stream as a persistent **memory channel**, and the per-step LayerNorm rescales the magnitude. This converts the recurrence from a drifting shared-weight residual stack into an iteration *anchored* to the input — the structure that lets Universal Transformers / DEQs scale with depth. **Central hypothesis: input anchoring rescues depth**, which regressed past N=2 in the [no-residual section](#recurrence-with-adagrad-no-residual). N=5/10/20 are therefore back in scope.

**Sweep (1 epoch each; reference = T4 baseline at N=K=1, best val 3.5157 / BPB 1.1311). Each row pairs with its no-residual twin above; Δ resid = BPB change from adding input anchoring.**

| Run | N | K | Mode | Total apps | BPB sliding | PPL sliding | Best val | Δ vs T4 | no-resid BPB | Δ resid | Run Log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T4 baseline (N=1, K=1) | 1 | 1 | — | 1 | 1.1311 | 34.24 | 3.5157 | (ref) | 1.1311 | — | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| + resid N=1 K=2 | 1 | 2 | distinct | 2 | 1.1262 | 33.72 | 3.4966 | −0.0049 | 1.1249 | +0.0013 | [link](logs/wikitext-103_2026-05-29_09-36-18/log.txt) |
| + resid N=2 K=1 | 2 | 1 | shared | 2 | 1.1272 | 33.83 | 3.5094 | −0.0039 | 1.1279 | −0.0007 | [link](logs/wikitext-103_2026-05-29_16-13-40/log.txt) |
| + resid N=2 K=2 | 2 | 2 | distinct | 4 | 1.1219 | 33.28 | 3.4906 | −0.0092 | 1.1227 | −0.0008 | [link](logs/wikitext-103_2026-05-29_21-45-14/log.txt) |
| + resid N=5 K=1 | 5 | 1 | shared | 5 | 1.1240 | 33.51 | 3.4986 | −0.0071 | 1.1291 | −0.0051 | [link](logs/wikitext-103_2026-05-30_04-52-42/log.txt) |
| + resid N=5 K=2 | 5 | 2 | cyclic | 10 | **1.1215** | **33.23** | **3.4852** | **−0.0096** | 1.1275 | −0.0060 | [link](logs/wikitext-103_2026-05-30_12-37-15/log.txt) |
| + resid N=10 K=1 | 10 | 1 | shared | 10 | 1.1256 | 33.66 | 3.4979 | −0.0055 | N/A | N/A | [link](logs/wikitext-103_2026-05-30_23-16-36/log.txt) |
| + resid N=20 K=1 | 20 | 1 | shared | 20 | queued | queued | queued | — | — | queued | queued |

**What each comparison tests:**
- **N=5 K=1 (depth-rescue, the key test):** no-residual regressed to 1.1291 (worse than N=2 K=1's 1.1279). If the anchored version beats N=2 K=1 instead, input injection has flipped depth from harmful to helpful — the headline result. If it still regresses, depth genuinely doesn't help this architecture even when anchored.
- **N=10 / N=20 K=1:** conditional on N=5 improving. If anchoring rescues depth, these probe how far it scales (the regime the no-residual sweep ruled out). Cancel each if the prior depth step plateaus/regresses. → *Result: N=10 regressed vs N=5 (1.1256 vs 1.1240), so N=20 was cancelled per this rule.*
- **N=1/2 K=2, N=2 K=1:** re-establish the diversity-vs-depth picture under anchoring; checks whether the "K beats N" finding survives, and whether anchoring lifts the current best (N=2 K=2, 1.1227) further.

**Findings:**

- **Depth rescued — the headline result.** No-residual N=5 K=1 *regressed* to 1.1291 (worse than N=2 K=1's 1.1279); with input anchoring it improves to **1.1240 (Δ resid −0.0051)** and now *beats* anchored N=2 K=1 (1.1272). Anchoring flips depth from harmful to helpful: where the shared-weight stack drifted, the input-anchored iteration converges. Confirms the hypothesis — the no-residual depth regression was a missing-input-anchor problem, not a fundamental depth ceiling.
- **Partial parameter-efficiency win — depth closes most of the gap to diversity, but doesn't overtake it.** Anchored N=5 K=1 (1.1240, **+0 params** vs T4) lands within ~0.002 of the K=2 plateau — near the noise floor. So shared-bank depth is now *competitive* with expensive distinct-bank diversity, but the best results still carry the +58.85M K=2 bank. This is exactly the gap the [Dense Mixer Recurrence](#dense-mixer-recurrence) experiments target: can learned trajectory routing over the shared bank close that last ~0.002 for free?
- **K=2 saturates with depth; N=5 K=2 (1.1215) is the marginal new best.** Adding depth on top of diversity barely moves anything: N=2 K=2 (1.1219) → N=5 K=2 (1.1215) is only −0.0004 (within noise), for 2.5× the apps (10 vs 4, 10.6h vs 6.8h). The best recurrence results plateau at **~1.1215–1.1219** regardless of N once K=2 is present. Anchoring still helped N=5 K=2 (Δ resid −0.0060), consistent with the depth-dependent anchoring benefit — but depth and diversity don't compound. Diversity (K) sets the ceiling; depth (N) just reaches it faster or, once there, adds nothing.
- **Anchoring is neutral at low depth, decisive at higher depth.** Δ resid: N=1 K=2 +0.0013, N=2 K=1 −0.0007, N=2 K=2 −0.0008 (all within noise) → N=5 K=1 **−0.0051** (clearly real). At N·K ≤ 4 there's little iteration to stabilize; at N=5 the anchor earns its keep — the depth-dependent pattern the mechanism predicts.
- **Depth peaks at N=5, then regresses.** Anchored N=10 K=1 lands at **1.1256 (−0.0055)** — **+0.0016 worse than N=5 K=1 (1.1240)**, just past the ~0.0010 noise floor. So anchoring extends the useful-depth range from N=2 (no-residual) to N=5, but does not make depth scale indefinitely: past N=5 the shared-bank iteration gives ground back. This confirms the ~1.121–1.124 plateau and removes the rationale for **N=20 K=1** — depth has already turned over, so the deeper probe is not worth its ~15.7h. K=2 diversity (1.1215) remains the ceiling.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Recurrence Efficiency: Gate Caching

Recurrence multiplies the per-step mixer cost by N. Each step's dominant cost is two `C'×C'` matmuls — the signal projection `W_mix·X` and the gate projection `W_gate·(R·X)` (cross-scale routed). The **gate caching** approximation (`mixer_recurrence_cache_gate`) computes the gate `φ(W_gate·R·X)` once on the first recurrence cycle and reuses it for cycles 2..N, eliminating the `W_gate` matmul + routing einsum on all but the first cycle — roughly halving per-step matmul cost at K=1. The trade-off: the gate no longer tracks the evolving spectral state past cycle 1.

**Hypothesis:** the cross-scale gate stabilizes quickly across recurrence steps, so re-gating each step is wasted compute. If true, caching is a near-free runtime win; if the gate is load-bearing per-step, quality regresses and we keep recomputing.

Auto-conditions: active only when `N > 1` (needs cycles to amortize over) and `use_mixer_gate=True`. Exact-equivalent to the baseline at N=1 (no caching possible). Baseline = the queued N=5 K=1 run (cache off); the test is identical except `cache_gate=true`.

| Run | N | K | cache_gate | BPB sliding | PPL sliding | Best val | Δ val vs base | Train time | Δ time | Run log |
|---|---|---|---|---|---|---|---|---|---|---|
| N=5 K=1 (baseline) | 5 | 1 | ✗ | 1.1240 | 33.51 | 3.4986 | (ref) | — | — | [link](logs/wikitext-103_2026-05-30_04-52-42/log.txt) |
| N=5 K=1 + gate cache | 5 | 1 | ✓ | 1.1268 | 33.79 | 3.5041 | +0.0055 | 6.38h | — | [link](logs/wikitext-103_2026-05-31_13-38-53/log.txt) |

**Decision rule:** if Δ best val < 0.0015 (within noise) **and** wall-clock drops meaningfully, caching is a free win: enable for all deeper-N runs (N=10, N=20, N=50). If best val regresses past noise, the per-step re-gating is load-bearing and caching is rejected.

**Findings:**

**Gate caching rejected.** Δ best val = **+0.0055** (nearly 4× the 0.0015 noise threshold) — the per-step re-gating is load-bearing and the approximation degrades quality. BPB sliding improved by −0.0028 (1.1268 vs 1.1240), but this falls within the architecture's known val/BPB non-correlation band and does not override the val signal. The gate is not a redundant computation; it carries meaningful information about the evolving recurrent state at each step. Gate caching is **not enabled** for deeper-N runs.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Long-Range Context: Multi-Pole SSM + Truncated BPTT

Two attention-free upgrades targeting **cross-window** long-range dependency. Within a 256-token block, the lifting wavelet already couples tokens ~128 apart (multi-scale, O(n log n)); beyond the block, the only carrier is the decompose-bypass cross-window state, which today is (a) a first-moment causal mean, (b) `.detach()`ed so it's never trained across windows, (c) a single per-channel vector. Plan: [plans/long_range_ssm_bptt.md](plans/long_range_ssm_bptt.md). Both default off; `false` = byte-identical to T4.

**#1 — Multi-pole diagonal SSM (`decompose_bypass_ssm`).** Replaces the cumulative-mean context summary with a bank of P parallel EMAs at learned per-channel decay rates (S4D-style diagonal SSM) — the existing single-EMA path (`decompose_bypass_ema`) is the P=1 case. Decays `a = exp(-softplus(θ))`, init spanning timescales τ ∈ [1, 256]; output `o = Σ_p G·h_p`, init `G=1/P` so it starts ≈ the multi-scale mean it replaces. ~2·C·P ≈ 16K params at P=4. By default the scan is **within-window**.

**#1b — Cross-window SSM carry (`decompose_bypass_ssm_cross_window`).** Carries the final pole state `h_{T-1} ∈ [B,C,P]` across block boundaries so the long poles integrate beyond the 256-token window — the genuine cross-block long-range mechanism. (The within-window SSM is a *confounded* proxy for it: within a block the SSM competes with the wavelet's own multi-scale mixing, so a flat within-window result is ambiguous; cross-block, nothing else reaches.) Block-local carry, **forward-only in v1** — detached each window, so it propagates multi-timescale memory but isn't BPTT-trained through the pole state (that's a follow-up). Requires sequential batching + gradient checkpointing off.

**#2 — Truncated BPTT across windows (`decompose_bypass_bptt`).** Stops detaching the cross-window [B,C] state every window; retains the graph across the grad-accum span (consecutive windows) and backprops once, so the model is *trained* to write useful long-range info into the carried state. Sequential-mode only (random windows are unrelated → auto-disabled with a log line). Memory ~span× (holds the span's activations at once).

**Ablation (T4 base, 1 epoch, sequential, `grad_accum=2`, MBS=8, eff. batch 16; decision rule clear T4 best val 3.5157 by > 0.0015).** All six share the batch config so only the SSM/x-window/BPTT flags vary — `grad_accum=2` is required because BPTT's span = grad_accum, so the default GA=1 would make BPTT a single-window no-op. The reference is the sequential GA=2 baseline (LR0), not the random-batched T4.

| Variant | SSM | x-win | BPTT | BPB sliding | PPL sliding | Best val | Δ vs T4 | Run Log |
|---|---|---|---|---|---|---|---|---|
| T4 baseline (sequential) | ✗ | ✗ | ✗ | 1.1499 | 36.31 | 3.6043 | (ref) | [link](logs/wikitext-103_2026-05-31_09-51-52/log.txt) |
| + SSM (within-window) | ✓ | ✗ | ✗ | 1.1464 | 35.92 | 3.5511 | −0.0035 | [link](logs/wikitext-103_2026-05-31_20-08-12/log.txt) |
| + BPTT | ✗ | ✗ | ✓ | 1.1497 | 36.30 | 3.6179 | −0.0002 | [link](logs/wikitext-103_2026-06-01_00-49-00/log.txt) |
| + SSM + BPTT | ✓ | ✗ | ✓ | 1.1466 | 35.94 | 3.5633 | −0.0033 | [link](logs/wikitext-103_2026-06-01_04-25-04/log.txt) |
| + SSM cross-window | ✓ | ✓ | ✗ | 1.1454 | 35.81 | 3.5522 | −0.0045 | [link](logs/wikitext-103_2026-06-01_09-03-24/log.txt) |
| + SSM cross-window + BPTT | ✓ | ✓ | ✓ | 1.1467 | 35.96 | 3.5869 | −0.0032 | [link](logs/wikitext-103_2026-06-01_13-42-34/log.txt) |

**What each tests:** within-window SSM — does a multi-timescale summary beat the first moment (confounded by wavelet redundancy)? BPTT — does *training* the mean cross-window state help? cross-window SSM — does carrying multi-timescale memory across blocks help (the non-redundant long-range test)? The stacked rows probe whether the axes compound. If even the full stack is flat, cross-window dependency isn't where WT103 perplexity lives at this scale (a clean negative result). The reference is a **sequential** T4 (the cross-window state only does anything in sequential mode), not the random-batched T4 number.

**Findings:**

All variants improve on the *sequential* reference (best: +SSM cross-window, −0.0045), but the sequential reference itself (1.1499) sits ~+0.019 above random-batched T4 (1.1311) — the gain from the best SSM stack is several times smaller than the cost of the sequential mode it requires. Within-window SSM provides most of what little there is (−0.0035; the wavelet already covers in-block range), BPTT adds nothing (−0.0002 alone), and cross-window carry adds only −0.0010 over within-window. **Verdict: a clean negative for WT103 at 256-token context — cross-window dependency is not where perplexity lives at this scale, and nothing here joins the T5 baseline** (random batching stays; `decompose_bypass` + simple cross-window mean carry remain the only long-range components, as in all T4/T5-line runs).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Dense Mixer Recurrence

DenseNet-style depth-weighted averaging over the mixer recurrence steps ([DenseFormer](https://arxiv.org/abs/2402.02622), Pagliardini et al. 2024, applied to the recurrence depth axis). Instead of each step's input being just `latest + X⁰` (the input-anchored residual), it becomes a **learned weighted combination of all prior step outputs**: with M = N·K applications and states `[X⁰, X⁽¹⁾, …]`, step *t*'s input is `inpₜ = Σ_{k≤t} A[t,k]·stateₖ`, where `A` is an M×M lower-triangular learnable matrix. Plan: [plans/dense_recurrence.md](plans/dense_recurrence.md). Config: `mixer_recurrence_dense` (+ `mixer_recurrence_dense_normalize` for softmax/convex rows). Default off = identical to today.

`A` is **initialized to reproduce the input-anchored residual exactly** (`A[t,t]=1`, `A[t,0]=1` for t≥1) — verified bit-identical at init — so dense starts from the current best and learns away from it. Cost is ~M²/2 scalars (≈15 at N=5 K=1) and the intermediate states are already in the autograd graph, so memory is marginal.

**The thesis is parameter efficiency.** The recurrence sweep showed diversity (K>1, distinct banks) beats depth — but K>1 costs +58.85M params per bank. Dense routing over a *shared* bank (K=1) costs ~15 params. The headline test: can dense N=5 K=1 approach the quality of distinct-bank K=2? If so, routing substitutes for parameters. Secondary: dense routing may let depth scale where input-anchoring plateaus (each step sees the whole trajectory, not just t−1 + X⁰).

**Interpretability:** `A` is a legible depth-routing table — it shows whether the model uses its full trajectory or collapses back to "latest + X⁰", and how much it leans on the X⁰ anchor. The normalized variant gives a per-step distribution over depths. This *adds* inspectable structure rather than opaque mixing, aligning with the project's legibility thesis.

**Ablation (T4 base, 1 epoch; rank by BPB sliding — val loss understates context-exploiting configs; ~0.0010 BPB threshold). Report the learned `A` matrix per run.**

| Run | N | K | dense | BPB sliding | PPL sliding | Best val | Δ vs T4 | Run Log |
|---|---|---|---|---|---|---|---|---|
| input-anchored N=5 K=1 (ref) | 5 | 1 | ✗ | 1.1240 | 33.51 | 3.4986 | (ref) | [link](logs/wikitext-103_2026-05-30_04-52-42/log.txt) |
| dense N=5 K=1 (raw) | 5 | 1 | ✓ | 1.1249 | 33.59 | 3.5030 | −0.0009 (within noise) | [link](logs/wikitext-103_2026-06-01_18-23-09/log.txt) |
| dense N=5 K=1 (normalized) | 5 | 1 | ✓ softmax | 1.1257 | 33.67 | 3.4982 | −0.0017 | [link](logs/wikitext-103_2026-06-02_01-55-46/log.txt) |
| dense N=10 K=1 | 10 | 1 | ✓ | 1.1257 | 33.66 | 3.5011 | −0.0017 | [link](logs/wikitext-103_2026-06-02_09-23-30/log.txt) |

**What dense has to beat.** Input-anchored N=5 K=1 (+0 params) reached **1.1240** — competitive with distinct-bank K=2 (1.1219, +58.85M) but not beating it (gap 0.0021). Dense's job is to close that last gap *for free*: does learned trajectory routing extract more from the same shared-bank depth than the fixed "latest + X⁰" anchor? If dense N=5 K=1 reaches ~1.1219 or below, routing has bought diversity-grade quality at ~15 params instead of +58.85M — the parameter-efficiency result this section exists to find. Since dense's `A` init *is* the anchored loop (1.1240), it can only match-or-beat that unless training moves `A` somewhere worse — so the question is purely whether the off-anchor routing weights find anything. A flat result (A stays near init) would say "latest + X⁰ is all the trajectory routing the model wants."

**Findings:**

All variants improve on the *sequential* reference (best: +SSM cross-window, −0.0045), but the sequential reference itself (1.1499) sits ~+0.019 above random-batched T4 (1.1311) — the gain from the best SSM stack is several times smaller than the cost of the sequential mode it requires. Within-window SSM provides most of what little there is (−0.0035; the wavelet already covers in-block range), BPTT adds nothing (−0.0002 alone), and cross-window carry adds only −0.0010 over within-window. **Verdict: a clean negative for WT103 at 256-token context — cross-window dependency is not where perplexity lives at this scale, and nothing here joins the T5 baseline** (random batching stays; `decompose_bypass` + simple cross-window mean carry remain the only long-range components, as in all T4/T5-line runs).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Untied Wavelet Reconstruction

The current implementation **ties** the wavelet reconstruct path to the decompose path: they share the same `predict_nets` and `update_nets` (perfect mathematical invertibility — decompose followed by reconstruct is exactly identity when no processing happens in between). The flag `untied_reconstruction` (already in config.json, currently `false`) would give the reconstruct path its **own** predict/update networks — same architecture, separate weights.

**Trade-off:**
- **Tied** (current): invertibility preserved. The structural identity `Reconstruct ∘ Decompose = I` is what enables [Recurrence (Adagrad, partial)](#recurrence-adagrad-partial) to express N-step recurrence as just N chained mixers (since adjacent Decompose-Reconstruct cycles cancel). Lower parameter count.
- **Untied**: reconstruct can apply learned transformations that aren't constrained to invert decomposition. More expressive — reconstruction can "fix up" the mixer's spectral output in ways the strict inverse would not allow. But the structural identity breaks, so the model's `x → Decompose → mixers → Reconstruct → x'` is no longer reducible to "mixers in a wavelet basis." Adds ~83.93M params per layer at T2 (matching the existing wavelet param count) — roughly +21% over T2.

**Mutually exclusive with Recurrence (Mixer Only).** Untied reconstruction breaks the invariant that justifies "mixer only" recurrence. If both are pursued, the recurrence design has to be reformulated — either to fold the full `Decompose → ... → Reconstruct` cycle into the recurrent loop (multiplying compute by N), or to share recurrent updates only within the spectral basis with explicit care for the non-inverse reconstruct. Cleaner to commit to one direction first: test untied reconstruction as a standalone variant against T2 baseline (1-epoch at fixed compute), then decide whether to compose it with recurrence.

**Test plan:** single-flag flip (`untied_reconstruction: true`) on T4 + 1ep (random batching). Compare BPB sliding and best val to T4 reference.

| Variant | Params | BPB sliding | PPL sliding | Best val | Δ vs T4 | Run log |
|---|---|---|---|---|---|---|
| T4 baseline (tied, ref) | 393.01M | 1.1311 | 34.24 | 3.5157 | (ref) | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| + untied reconstruction | 510.48M | 1.1324 | 34.38 | 3.5197 | +0.0013 | [link](logs/wikitext-103_2026-06-03_01-48-38/log.txt) |

**Result — tying is load-bearing, not just cheaper.** Untied reconstruction spent **+117.5M params (+30%)** and came out *worse* on both metrics: BPB +0.0013 (just past the ~0.0010 noise floor) and best val +0.0040 (~2.7× noise). The headline isn't the size of the quality gap (small) — it's that a pure **expressivity increase** (reconstruct freed to apply non-inverse transforms) plus 30% more parameters still couldn't match the tied baseline. The `Reconstruct ∘ Decompose = I` symmetry is doing real work, not merely saving parameters: the model's function genuinely depends on the wavelet stage being a true invertible transform around the mixer, not an arbitrary learned encoder/decoder. **Decision: keep tied.** Untied is rejected; the symmetry constraint stays.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Dropout

Re-tune the five dropout values (`dropout_lm_head`, `dropout_mlp`, `dropout_mixer`, `dropout_projection`, and `dropout_embedding`) once model parameters are reduced from above. A doubled-dropout ablation at the prior baseline gave -0.0221 BPB. This is larger than the projected BPB increase from parameter reduction. A true dropout sweep may surpass the gap.

Sweep is to be conducted at L=1 first (faster iteration, more sensitive to regularization signal). The resulting optimal values will then be retroactively applied to L=2 (and any other higher-layer formulations) for performance measurement and benchmarking to verify whether L=2's (or higher level's) val loss also improves under the L=1-tuned regularization recipe. Headline numbers accordingly.

**Sweep — coordinate descent (1ep each).** One dropout type at a time, ±10% around its current value; the **winner of each pair (or the incumbent, if neither clears the ~0.0010 BPB noise floor) is carried forward** into the next type's runs. Each row's non-varied columns show the values *in force at that step*. The final coordinate's run *is* the optimized stack — no separate combine run needed. Bolded cell = the value being varied.

| Step | drop_emb | drop_proj | drop_mix | drop_mlp | drop_lm | BPB sliding | PPL sliding | Best val | Δ vs T4 | Run log |
|---|---|---|---|---|---|---|---|---|---|---|
| T4 baseline | 0.20 | 0.10 | 0.10 | 0.10 | 0.240 | 1.1311 | 34.24 | 3.5157 | (ref) | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| emb −10% | **0.18** | 0.10 | 0.10 | 0.10 | 0.240 | 1.1307 | 34.20 | 3.5161 | −0.0004 | [link](logs/wikitext-103_2026-06-03_06-43-53/log.txt) |
| emb +10% | 0.22 | 0.10 | 0.10 | 0.10 | 0.240 | 1.1309 | 34.22 | 3.5187 | −0.0002 | [link](logs/wikitext-103_2026-06-03_11-49-52/log.txt) |
| proj −10% | 0.18 | **0.09** | 0.10 | 0.10 | 0.240 | 1.1304 | 34.16 | 3.5164 | −0.0007 | [link](logs/wikitext-103_2026-06-03_16-46-23/log.txt) |
| proj +10% | 0.18 | 0.11 | 0.10 | 0.10 | 0.240 | 1.1316 | 34.30 | 3.5169 | +0.0005 | [link](logs/wikitext-103_2026-06-03_21-36-30/log.txt) |
| mix −10% | 0.18 | 0.09 | **0.09** | **0.10** | 0.240 | 1.1295 | 34.07 | 3.5143 | −0.0016 | [link](logs/wikitext-103_2026-06-04_05-24-45/log.txt) |
| mix +10% | 0.18 | 0.09 | 0.11 | 0.10 | 0.240 | 1.1311 | 34.24 | 3.5152 | +0.0000 | [link](logs/wikitext-103_2026-06-04_10-10-20/log.txt) |
| mlp −10% | 0.18 | 0.09 | 0.09 | 0.09 | 0.240 | 1.1305 | 34.17 | 3.5131 | −0.0006 | [link](logs/wikitext-103_2026-06-04_15-15-53/log.txt) |
| mlp +10% | 0.18 | 0.09 | 0.09 | 0.11 | 0.240 | 1.1309 | 34.22 | 3.5160 | −0.0002 | [link](logs/wikitext-103_2026-06-04_20-06-51/log.txt) |
| lm_head −10% | 0.18 | 0.09 | 0.09 | 0.10 | **0.216** | **1.1285** | **33.96** | **3.5127** | **−0.0026** | [link](logs/wikitext-103_2026-06-05_01-38-46/log.txt) |
| lm_head +10% | 0.18 | 0.09 | 0.09 | 0.10 | 0.264 | 1.1305 | 34.18 | 3.5158 | −0.0006 | [link](logs/wikitext-103_2026-06-05_06-27-30/log.txt) |
| **Final stack** | **0.18** | **0.09** | **0.09** | **0.10** | **0.216** | **1.1285** | **33.96** | **3.5127** | **−0.0026** | [link](logs/wikitext-103_2026-06-05_01-38-46/log.txt) |

**Coordinate-descent close.** Final optimized dropout stack: **emb 0.18 / proj 0.09 / mix 0.09 / mlp 0.10 / lm_head 0.216**, BPB sliding **1.1285** (−0.0026 vs T4 1.1311). The last coordinate's winning run *is* the stack — no separate combine run.

- **Cumulative, not per-step.** Every individual pair landed at-or-below the ~0.0010 noise floor; the −0.0026 stack total is the sum of small same-direction steps (SGD-trajectory effect), ~2.6× the single-seed floor.
- **Four of five coordinates wanted *less* dropout** (emb↓, proj↓, mix↓, lm_head↓; only mlp held at 0.10). Coherent signal that **L=1 is mildly over-regularized at the T4 defaults** — consistent with the project's "L=1 is regularization-bound" finding.
- **Not yet confirmed.** −0.0026 is single-seed and built by picking the lower-BPB value at five sub-noise steps, which is positively biased (min-of-noisy-pairs selection). True effect is likely between ~0 and −0.0026. **Requires a 2–3 seed check of the final stack vs T4 before being trusted as a real win.**

**Follow-ups for the higher-layer re-run** (where ±10% steps should clear noise and the surface isn't flat):
- **Edge-winners — continue the line search downward** (winner sat at the bottom of the tested range, monotonic): **proj** (probe 0.08, 0.07…), **mix** (probe 0.08…), **lm_head** (probe 0.20, 0.18…).
- **mlp — metric split.** BPB had an interior min at 0.10; best val was monotonic-down (0.09 best). Both variations were within seed-noise, so the disagreement may be two noise patterns over a flat surface. Probe **mlp ≤0.08, seed-checked**, to resolve.
- **emb — flat tie** (non-monotonic, 0.0004 spread); 0.18 is a weak prior only.
- **Convergence test:** a full second sweep with zero coordinate moves (one sweep can leave a jointly-suboptimal point since fixing one coordinate can shift another's optimum).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Weight Decay

Re-tune `weight_decay`. Current value (1e-6) was only tested alongside 1e-3. More values must to be attempted (likely slightly higher is best).

| weight_decay | BPB sliding | PPL sliding | Best val | Δ vs T4 | Run log |
|---|---|---|---|---|---|
| 1e-06 (T4 baseline) | 1.1311 | 34.24 | 3.5157 | (ref) | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| 5e-07 (lower) | 1.1330 | 34.45 | 3.5271 | +0.0019 | [link](logs/wikitext-103_2026-06-05_17-58-36/log.txt) |
| **2e-06 (higher)** | **1.1276** | **33.87** | **3.5107** | **−0.0035** | [link](logs/wikitext-103_2026-06-05_22-45-57/log.txt) |

**Finding — higher WD helps, and it's the cleanest single result of the cycle.** Monotonic with both metrics agreeing (5e-7 → 1e-6 → 2e-6: BPB 1.1330 → 1.1311 → 1.1276, val 3.5271 → 3.5157 → 3.5107). **2e-6 beats T4 by −0.0035 BPB (~3.5× the single-seed noise floor) on a single step** — the only individual ablation in this cycle to clear noise on its own (every dropout step was at/below floor; this is decisively above). Run at T4 dropout defaults, so directly comparable to T4.

**The dropout↔WD asymmetry is the interesting part.** Dropout wanted to go *down* (4 of 5 coordinates ↓), but weight decay wants to go *up*. Not a contradiction — they regularize different things: the model wanted *less* stochastic/activation regularization (dropout, targets co-adaptation) but *more* weight-norm regularization (WD). Suggests L=1's overfitting is more weight-magnitude growth than co-adaptation.

**Caveats / follow-ups:** (1) 2e-6 is the **range edge** — true optimum may be higher (5e-6, 1e-5); continue upward at T5/higher-L. (2) Single-seed; rides to the same seed-check as the dropout stack before locking. (3) Stacking with the dropout stack untested — both go into the scheduled **T5 Baseline** consolidation, not the current default.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Complex Wavelets and Complex Mixer

Replace the real-valued wavelet basis with a complex-valued one (e.g., dual-tree complex wavelet transform à la Kingsbury, or direct complex parameterization of the lifting predict/update networks). The motivation that matters is **not** "phase carries bonus information" (the vague version) but **shift-invariance** (the structural version).

**The real mechanism: shift-invariance.** A critically-sampled wavelet transform — including our learned lifting scheme — is shift-*variant*. The split into even/odd lattices is a fixed sampling operation, so the same token pattern starting at position 4 vs position 5 produces materially different coefficient patterns. The learned predict/update networks **cannot fully correct this**, because the variance comes from the sampling lattice, not the filter weights. The classic reason to go complex (the dual-tree CWT) is precisely that an analytic/complex representation yields *approximately shift-invariant magnitude*, with phase encoding where within the subband a feature sits.

For language this is a concrete inductive-bias gap, not an analogy: every syntactic constituent or n-gram that recurs at different offsets currently forces the mixer to learn multiple shifted coefficient signatures of the *same* structure. This is the same positional-binding / agreement-attraction weakness where WaveletLM is structurally weakest versus attention — the place a shift-invariant representation would most directly help.

**Why the added parameters could be load-bearing rather than confounding.** Widening a real lifting net adds capacity to *memorize* both shifted versions; it does **not** make the representation shift-invariant. Complex makes the magnitude shift-invariant *by construction*. So a matched-param win (complex > real-at-equal-params) would be a genuine architectural result — capability emerging *from the structure*, not from capacity. This is the cleanest such test in the current backlog. Secondary, more speculative upsides: the FWHT mixer currently never sees within-subband position (real coefficients collapse it), so phase could let cross-scale gating route on *relative position* — a capability the architecture structurally lacks today; and a complex representation is a richer substrate for *positional* mechanistic probing (you can read off where in the window a feature lives), a non-perplexity reason aligned with the interpretability direction.

**Cost (measured).** The real lifting stage is **117.50M params** (T4 breakdown, [log](logs/wikitext-103_2026-05-24_19-22-19/log.txt)), shared across layers, reconstruct tied (free). The invertible complex wavelet is **469.99M** — complex predict *and* update nets (each ≈2× a real net), reconstruct tied — measured via `tools/complex_wavelets.param_count()` at C=2048/levels=7. Total model **≈745.50M** vs T4's 393.01M. The spectral stack runs on both real and imaginary parts (the mixer sees the imaginary channel — that's how phase is mixed), so spectral-stage compute is ~2× as well; the real part only is taken at the block output.

**Overlap with `wavelet_crawl` (already harvesting part of this benefit).** `wavelet_crawl` (learned ±1 dilation per level, −0.0037 at T4) is a cheap existing mechanism that already buys *some* shift/scale-robustness. So complex wavelets are competing for a *smaller* marginal headroom than a from-scratch estimate would suggest — part of the shift-robustness gain is already taken. The matched-param control must run with `wavelet_crawl` in its current T4 state so the comparison isolates what complex adds *on top of* crawl.

**Construction — invertible, tied.** The wavelet decomposes to *full complex* coefficients, the spectral mixer processes them, and a **tied** reconstruct (reusing decompose's nets) inverts exactly — `Reconstruct∘Decompose = I` is preserved. This matters because the [untied reconstruction](#untied-wavelet-reconstruction) ablation showed that symmetry is load-bearing (+117.5M untied params *regressed*). So the complex wavelet keeps the symmetry *and* mixes phase, rather than collapsing phase away. Phase enters the mixer via `complex_mixer_activation`:
- **`split`** — the real spectral stack runs on the real and imaginary parts independently (split-complex; no re/im cross-coupling).
- **`modulus_phase`** — gate the magnitude `|z|` (the shift-invariant quantity) with a **non-negative** (softplus) gate and re-apply the preserved unit phase, so it scales magnitude without spurious sign-flips.

Round-trip identity verified to ~1e-6 (off-init weights), causality verified (no future leak), imaginary path verified to receive gradient at init.

**Test design (matched-param form).** A bare complex run beating T4 would be uninterpretable — params help monotonically here, so a win could be capacity, not phase. Each complex variant is paired with a **real-wavelet control widened (via `lifting_hidden_mult`) to the same param count**, and is validated only if it beats *that control*, not merely T4. The invertible wavelet is 469.99M (complex predict AND update, tied reconstruct → ~745M total); its matched real control is `hidden_mult=4` (469.91M wavelet, ratio 1.000 — near-exact).

| Variant | Params | BPB sliding | Best val | Δ vs T4 | Train VRAM | Inf VRAM | Run log |
|---|---|---|---|---|---|---|---|
| T4 baseline (real, tied) | 393.01M | 1.1311 | 3.5157 | (ref) | 7,790 MiB | n/m | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| complex invertible / split | 745.50M | 1.1500 | 3.5689 | +0.0189 | 10,229 MiB | 3,759 MiB | [link](logs/wikitext-103_2026-06-07_09-03-30/log.txt) |
| complex invertible / modulus_phase (softplus) | 745.50M | NaN | NaN | — | — | — | [link](logs/wikitext-103_2026-06-07_17-33-50/log.txt) |
| complex invertible / modulus_phase (tanh) | 745.50M | NaN | NaN | — | — | — | [link](logs/wikitext-103_2026-06-07_20-01-24/log.txt) |
| **real control (hidden_mult=4)** | 745.45M | **1.1362** | 3.5305 | −0.0011 (wins) | 14,511 MiB | 5,112 MiB | [link](logs/wikitext-103_2026-06-08_02-58-19/log.txt) |

**Verdict (complex wavelet): regresses, control wins.** split (1.1500) loses to its matched-param real control (1.1362) by **+0.0138** — the control converts the extra params to a small *gain* over T4 (−0.0011), while the complex machinery turns them into a loss. Both modulus_phase variants NaN'd at peak LR (~step 14k, lr≈0.019) regardless of gate (softplus or tanh) — the instability is in the magnitude/phasor path, not the gate; lr=0.01 would be the cheap fallback if ever revisited, but the split result already shows the basis doesn't help here.

The **tanh** gate variant (`complex_gate_activation="tanh"`) is bounded (|g|≤1, a stabilizer) and bipolar (learns discrete π phase flips — expressivity the strictly-positive softplus lacks), init-shifted to start at ln2 = softplus's init value so the two are a clean ablation.

Implementation: [plans/complex_wavelets.md](plans/complex_wavelets.md) and [tools/complex_wavelets.py](tools/complex_wavelets.py). Two real tensors (not native `complex64`), CGELU, tied complex reconstruct, ComplexLayerNorm (joint re/im whitening) for stability. Config: `wavelet_basis`, `complex_mixer_activation: "split"|"modulus_phase"`, `complex_gate_activation: "softplus"|"tanh"`. Mutually exclusive (hard-errored) with recurrence, mixer_depth>1, untied reconstruction, multi-basis, 2D wavelet, non-shared lifting, and wavelet_crawl.

**Option C — complex in the MIXER, real wavelet.** The complex wavelet regressed (split +0.0189 vs T4 at L=1), plausibly because the wavelet's invertibility/causality constraints fight the complex machinery. The mixer operates in spectral space where phase is native, so a separate experiment keeps the wavelet **real and exactly invertible** and makes the *mixer* complex via a per-scale learned real↔complex projection (`RealToComplexProjection`) around the FWHT — up-project real coeffs to (re, im), run the complex spectral pass, down-project to real, then the real reconstruct. This is the cleaner test of "does a complex spectral representation help" without the wavelet-side constraints, and it's a *distinct beast* from complex wavelets (mutually exclusive; `complex_mixer=true`). Measured 527.13M at T4 (+134.22M from the projections); matched real control = `hidden_mult=2` (510.38M, ~3% under — no integer hm matches exactly).

| Variant | Params | BPB sliding | Best val | Δ vs T4 | Train VRAM | Inf VRAM | Run log |
|---|---|---|---|---|---|---|---|
| complex mixer / split | 527.13M | 1.1535 | 3.5825 | +0.0224 | 10,288 MiB | 3,854 MiB | [link](logs/wikitext-103_2026-06-08_11-07-18/log.txt) |
| complex mixer / modulus_phase | 527.13M | 1.1642 | 3.6188 | +0.0331 | 10,288 MiB | 3,854 MiB | [link](logs/wikitext-103_2026-06-08_17-53-04/log.txt) |
| **real control (hidden_mult=2)** | 510.38M | **1.1434** | 3.5500 | +0.0123 | 10,030 MiB | 3,768 MiB | [link](logs/wikitext-103_2026-06-08_23-52-43/log.txt) |

**Verdict (complex mixer, Option C): also regresses, control also wins.** Stable (no NaNs, unlike the complex-wavelet modphase), but split (1.1535) and modphase (1.1642) both lose to their matched-param real control (1.1434) by **+0.0101 and +0.0208** respectively. modphase is worse than split here too, consistent with the wavelet results.

**Combined conclusion — complex representations don't help WaveletLM at L=1, regardless of location.** This was the question Option C was built to answer: are wavelet-complexity and mixer-complexity different beasts? **No** — both regress against their size-matched real controls, same direction, similar magnitude (wavelet split +0.0138; mixer split +0.0101). In every case a plain real widening converts the extra parameters to quality more effectively than the complex machinery does. The feature is a wash and is **set aside** (the per-block phase reset / cross-depth phase propagation, deferred items #1/#4, would be the only remaining avenue, and only worth revisiting at higher layer depth where shift-invariance could plausibly pay off — not at L=1). Config flags and code retained for that potential future depth-gated retest; not in any baseline.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Wavelet Crawl Off

**Chronological correction.** An earlier section (the deprecated bs=16384 / R0 / Test-1 line) recorded `wavelet_crawl` as removable "at no performance benefit," and that verdict is preserved there as an accurate snapshot of what was believed at that point. It was **regime-specific and later reversed.** At the *production* regime (T2: bs=256, levels=7 — the line that became T4), `wavelet_crawl` was re-ablated and is a real, repeatable win:

| Variant | Params | BPB sliding | Best val | Run log |
|---|---|---|---|---|
| T2 without wavelet_crawl (1ep) | 392.91M | 1.1616 | 3.6094 | [link](logs/wikitext-103_2026-05-10_01-39-25/log.txt) |
| **T2 with wavelet_crawl (1ep)** | 392.91M | **1.1541** | **3.5881** | [link](logs/wikitext-103_2026-05-10_03-39-43/log.txt) |
| Δ at T2 (crawl on − off) | — | **−0.0075** | **−0.0213** | ~5× the 0.0015 noise floor |
| T4 without wavelet_crawl (1ep) ‡ | 393.01M | 1.1492 | 3.5670 | [link](logs/wikitext-103_2026-06-09_22-59-25/log.txt) |
| **T4 with wavelet_crawl (1ep)** | 393.01M | **1.1311** | **3.5157** | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| **Δ at T4 (crawl on − off)** | — | **−0.0181** | **−0.0513** | **~18× the noise floor** |

‡ Measured incidentally as the fwht control of the [Mixer Transform Ablation](#done-mixer-transform-ablation) — identical config to T4 except `wavelet_crawl=false`.

So `wavelet_crawl=true` is a genuine part of the T4 production baseline (config.json default), *not* the no-op the deprecated section described. The two verdicts are both correct for their regimes: crawl is inert at bs=16384 (the coarsest scales span hundreds of tokens, where ±1 dilation is negligible) but helps at bs=256/levels=7 (the ±1 dilation offset is meaningful relative to the finer scales). **And the effect grows with the regime's tuning: at T4 (lr=0.0225) crawl is worth −0.0181 — 2.4× its T2 (lr=0.01) value, and the single largest component-level win measured on the T4 line.** The learned-dilation convolution axis is doing more work than the spectral-transform axis (see the transform ablation's identity result); pushing it further is the subject of the [Wavelet Crawl Dilation Window (K) Sweep](#in-progress-wavelet-crawl-dilation-window-k-sweep).

**Relevance to [Complex Wavelets and Complex Mixer](#done-complex-wavelets-and-complex-mixer).** The complex trees do not implement `wavelet_crawl` (and `model.py` hard-errors `wavelet_basis=complex` + `wavelet_crawl=true` rather than silently ignore it). So **all complex wavelet runsn have wavelet crawl turned off**, which is why their in-section reference is the matched real control (CW4, also crawl-off) and **not** the crawl-on T4 baseline — comparing a crawl-off complex run to crawl-on T4 would conflate the basis change with the loss of this −0.0075 crawl win.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Wavelet Sparsity Probe & Wavelet Shrinkage

A diagnostic probe ([tools/wavelet_sparsity_probe.py](tools/wavelet_sparsity_probe.py)) on the sparsity structure of the learned wavelet's detail coefficients — a property classical wavelet compression (JPEG 2000) heavily exploits in *fixed* wavelets on natural images, here measured for the first time on our *learned* wavelet on language. Forward hooks capture per-scale detail coefficients of the T4 checkpoint ([log](logs/wikitext-103_2026-05-24_19-22-19/log.txt), L=1, levels=7) over 8 held-out WikiText-103 batches; no training, ~30 s.

**Magnitude distribution per scale.** Detail-coefficient magnitudes decay smoothly from coarse (scale 0) to fine (scale 6), but the per-scale distributions are *spread*, not spiked: median sits at ~6% of max and `mean ≈ median` (the signature of a mildly right-skewed distribution, **not** the sharp spike-at-zero of a sparse wavelet). Only ~11% of coefficients fall below 1% of the per-scale max.

| Scale | mean\|d\| | median | p90 | p99 | max | frac < 0.01·max |
|---|---|---|---|---|---|---|
| 0 | 0.8236 | 0.6818 | 1.6832 | 2.9558 | 11.6351 | 0.082 |
| 1 | 0.4927 | 0.3994 | 1.0178 | 1.8188 | 8.4953 | 0.096 |
| 2 | 0.4191 | 0.3355 | 0.8704 | 1.6009 | 7.6758 | 0.104 |
| 3 | 0.3610 | 0.2838 | 0.7583 | 1.4573 | 7.1620 | 0.117 |
| 4 | 0.2657 | 0.2090 | 0.5619 | 1.0396 | 4.5080 | 0.099 |
| 5 | 0.1868 | 0.1472 | 0.3959 | 0.7119 | 6.8283 | 0.141 |
| 6 | 0.1240 | 0.0946 | 0.2693 | 0.4910 | 3.0489 | 0.125 |
| **mean** | | | | | | **0.109** |

**Energy-retention curve (the real compressibility metric).** Count-below-threshold depends on an arbitrary cutoff; what actually governs compressibility is *energy* (∑d²), since a few large coefficients dominate. For each scale we hard-threshold the smallest coefficients and ask: to retain X% of energy, what fraction of coefficients can be zeroed? The curve is remarkably **scale-uniform** (every scale within ±0.02 of the mean) — unlike fixed wavelets, where fine scales are far sparser than coarse.

| Energy retained | mean droppable fraction |
|---|---|
| 90% | 0.597 |
| 95% | 0.485 |
| 99% | 0.291 |
| 99.9% | 0.137 |
| 99.99% | 0.064 |
| 99.999% | 0.030 |
| 99.9999% | 0.014 |

**Verdict: semi-sparse, unstructured.** The learned wavelet is meaningfully compressible — drop ~29% of coefficients for 1% energy loss — but far from the ~95%-droppable of fixed wavelets on images, and well above the ~4% of a uniform distribution. The tail decays smoothly with no spike of machine-zero coefficients (even at 99.9999% energy only 1.4% are droppable), so there is **no free sparse tier to exploit**. This is itself the finding: **language, under a learned wavelet, does not concentrate into sparse detail coefficients the way smooth natural signals do** — the model spreads linguistic information densely and evenly across all scales (the cross-scale gating and per-scale widths appear to equalize the load).

**Wavelet shrinkage — not pursued.** Soft-thresholding detail coefficients during training was the planned follow-up *if* the representation proved sparse. It did not. With energy spread across the bulk (the smallest 29% of coefficients still carry 1% of energy, and dropping more costs real signal), shrinkage would either be a no-op (tiny λ) or actively harmful (meaningful λ). Separately, **dropping coefficients yields no inference-efficiency win regardless of sparsity**: detail coefficients are dense activations, so zeroing values neither reduces peak VRAM (a zero occupies the same fp16 slot) nor latency (the downstream FWHT/mixer are dense ops over every entry). The only efficiency lever the architecture supports here is low-bit *quantization* of the detail path (keep all coefficients, fewer bits), which is orthogonal to sparsity and tracked under the quantization config. The probe is therefore recorded as an **interpretability data point** about learned-wavelet behavior, not an optimization.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Inference-Depth Flexibility with Mixer Recurrence (Train Deep, Infer Shallow)

Recurrence multiplies inference latency, not just training cost: every token pays N× the mixer (N=5 K=2 generates at ~18 tok/s vs T4's ~34 tok/s). This section pursues **decoupling inference depth N′ from training depth N** — keeping the deep-trained quality while recovering speed. **Target checkpoint: the recurrence sweep settled on N=5 K=1** (BPB 1.1240, best val 3.4986, +0 params over T4 — the depth-ladder winner; N=10 regressed, so N=5 is the chosen depth). K=1 is also the clean case for Route 1's fixed-point argument (a single shared bank converges to one point; K>1 would converge to a K-cycle).

**Route 1 — N′ convergence sweep (no retraining, cheap, run first).** The input-anchored recurrence iterates toward a fixed point `X* = LN(X⁰ + m(X*+X⁰))`, so its output should *plateau* at some N′ ≤ N. On the N=5 K=1 checkpoint, cap the forward loop at N′ and measure BPB(N′). Requires a `mixer_recurrence_inference_steps` override that changes only the loop bound (the model is still *constructed* at the trained N=5 so the per-step norms match the checkpoint; inference reuses the first N′−1 norms). **Measured (see table below):** quality plateaus within noise at N′=4 (free), with a real but small cost at N′=3, and collapse below — matching the predicted shape (N′=1 worse than T4; the mixer is a refinement step toward the fixed point, not a one-shot transform). The deliverable is the quality/latency frontier, so report **BPB(N′) and tok/s(N′) together.**

Checkpoint: N=5 K=1 ([log](logs/wikitext-103_2026-05-30_04-52-42/log.txt)), trained BPB 1.1240. Each row caps the recurrence loop at N′ at inference only.

| N′ (infer steps) | BPB sliding | Δ vs full N=5 | Δ vs T4 (no recur) | tok/s | Speedup | Peak VRAM |
|---|---|---|---|---|---|---|
| 1 | 1.2290 | +0.1050 | +0.0979 | 36.2 | 1.36× | 3,096 MiB |
| 2 | 1.1417 | +0.0177 | +0.0106 | 30.1 | 1.13× | 3,096 MiB |
| 3 | 1.1285 | +0.0045 | −0.0026 | 29.4 | 1.11× | 3,096 MiB |
| 4 | 1.1249 | +0.0009 | −0.0062 | 26.5 | 1.00× | 3,096 MiB |
| 5 (= trained N) | 1.1240 | (ref) | −0.0071 | 26.6 | (ref) | 3,096 MiB |

(tok/s = mean of two warmed 512-token generations; speedup vs full N=5.)

**Findings — quality.** The N′=5 row reproduces the trained 1.1240 exactly, confirming the override changes only the loop bound (the full-depth path is untouched). Recovery is **monotonic and rapid** — validating the input-anchored residual: a plain (un-anchored) stack would not truncate gracefully. The plateau is at **N′=4, not N′=3**: N′=4 is within the ~0.0010 noise floor of full depth (+0.0009), while N′=3 costs a real +0.0045 (4.5× noise) though it still beats no-recurrence. The crossover vs the no-recurrence T4 baseline is **N′=3** — below that (N′≤2) truncation has eaten the entire recurrence benefit and then some, so you'd be better off not training recurrence at all. N′=1 is catastrophic (+0.098 over T4), confirming the mixer is a *refinement* step toward the fixed point, not a one-shot transform. As fixed-point dynamics, this is clean: the input-anchored iteration reaches its attractor by ~step 4.

**Findings — latency (the payoff does not land where quality is preserved).** VRAM is **flat at 3,096 MiB** across all N′ — recurrence intermediates are transient and weights dominate, so this is a compute knob, not a memory one (as predicted). On throughput, the catch: **N′=4, the free-quality point, gives essentially zero speedup** (1.00×, tied with N′=5 within timing noise). Meaningful speedup only appears at **N′≤3** (1.11× at N′=3), which costs quality, and the large gains (1.36× at N′=1) live in the quality-collapse zone. The reason is structural — at K=1, recurrence is 5 mixer applications out of a per-token forward dominated by lifting / MLP / PKM / FwPKM / head, so truncating one step is a small wall-clock fraction. **Net verdict:** "train deep, infer shallow" works as a *quality-graceful* knob (you can truncate to N′=3 without catastrophe), but it is **not a useful latency lever for this K=1 checkpoint** — the only operating point that preserves quality (N′=4) saves no time, and the only points that save time sacrifice quality. The standout deliverable is the **fixed-point convergence result**, not a practical speedup. (A heavier recurrence config — e.g. K=2, 10 applications — would give truncation more headroom to recover; not tested here.)

**Route 2 — per-step deep supervision (train-for-it, fallback).** To make low N′ (≤2) viable, train so that *every* intermediate is a valid prediction: apply the shared LM-head loss at each recurrence step (deep supervision / "anytime" inference), or randomize N during training (stochastic depth on N). Either makes inference depth a free knob — enabling aggressive early exit, even N′=1 — at the cost of extra training compute and a quality trade at full N. **Critical caveat (from Route 1's result):** flattening the curve for cheap low-N′ inference necessarily erodes the N=5 peak — the model can no longer specialize for exactly-5-step dynamics, so trained-for-flexibility N=5 BPB rises toward the truncated values. Net, randomized-N training trades peak quality for low-N′ robustness; it does not give both. And the latency results make the trade worse: the speedup Route 2 would unlock (N′≤2, up to 1.36×) is modest, while the quality-preserving zone already reachable for free (N′=3, 1.11×) captures most of the available throughput. Given that (a) the latency lever is structurally weak at K=1 (recurrence is a small forward fraction), and (b) recurrence's total benefit over T4 is only −0.0071, Route 2 is **not worth it** unless a deployment specifically needs the aggressive N′≤2 speedup *and* can absorb the quality loss.

**Decision logic:** Route 1 (free) is **done**. Its verdict: graceful quality truncation confirmed (a clean fixed-point result), but the latency payoff is weak — N′=4 preserves quality at zero speedup, N′=3 gives 1.11× at a small quality cost. Route 2 stays deferred/unlikely per the caveat above; only revisit if aggressive N′≤2 serving becomes a hard requirement and the quality trade is acceptable.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Mixer Transform Ablation

Tests the contribution of the FWHT slot in the per-scale mixer against alternative orthonormal transforms: no transform (identity), Hartley (DHT), DCT, and a **learned** orthogonal butterfly. All candidates are orthonormal, so they are amplitude-matched and **parameter-free** (learned_butterfly adds only log₂(Cp)·Cp/2 ≈ 11k angle params, ~0.003%) — there is no param confound, the only variable is the basis. Run at the T4 reference (L=1, levels=7, lr=0.0225, crawl off, wavelet norms on). See [plans/other_post_release_plans.md §10](plans/other_post_release_plans.md#10-per-scale-mixer-transform-ablation) for the full design.

**Why the basis can matter at all (it shouldn't, for the linear part).** The mixer's linear map is *basis-absorbable*: FWHT∘W∘FWHT⁻¹ is just another linear, so identity can replicate it with different weights. The only basis-*dependent* operation is the mixer's **element-wise gate** — gating Walsh-frequencies (FWHT) is a different nonlinearity than gating raw channels (identity) or other bases. So this ablation is really "does the gate care which basis it acts in?", and the **learned butterfly** is the definitive version: it lets gradient descent pick its own orthogonal gating basis (init = identity; rotation-only, so it spans a structured SO family rather than exactly containing FWHT's reflections).

| Transform | Learned? | BPB sliding | Best val | Δ vs T4† | Train VRAM | Inf VRAM | Run log |
|---|---|---|---|---|---|---|---|
| T4 baseline (fwht, crawl **on**) | no | 1.1311 | 3.5157 | (ref) | 7,790 MiB | 3,096 MiB | [link](logs/wikitext-103_2026-05-24_19-22-19/log.txt) |
| **learned_butterfly** | **yes** | **1.1455** | 3.5596 | +0.0144 | 10,090 MiB | 3,096 MiB | [link](logs/wikitext-103_2026-06-09_17-40-33/log.txt) |
| identity (no transform) | no | 1.1463 | 3.5596 | +0.0152 | 7,790 MiB | 3,096 MiB | [link](logs/wikitext-103_2026-06-09_13-12-26/log.txt) |
| fwht control (crawl **off**) | no | 1.1492 | 3.5670 | +0.0181 | 7,790 MiB | 3,096 MiB | [link](logs/wikitext-103_2026-06-09_22-59-25/log.txt) |
| dht (Hartley) | no | 1.1479 | 3.5683 | +0.0168 | 7,806 MiB | 3,112 MiB | [link](logs/wikitext-103_2026-06-10_03-44-52/log.txt) |
| dct | no | 1.1478 | 3.5654 | +0.0167 | 7,806 MiB | 3,112 MiB | [link](logs/wikitext-103_2026-06-10_08-16-53/log.txt) |
| crawl **on** + learned_butterfly | **yes** | 1.1297 | **3.5098** | −0.0014 | 10,211 MiB | 3,096 MiB | [link](logs/wikitext-103_2026-06-10_12-46-47/log.txt) |
| **crawl on + identity — new T4-class best** | no | **1.1287** | 3.5105 | **−0.0024** | 7,790 MiB | 3,096 MiB | [link](logs/wikitext-103_2026-06-10_18-06-13/log.txt) |

†Δ is against the T4 baseline (crawl **on**). The clean same-config reference for crawl-**off** rows is the fwht control (1.1492); for crawl-**on** rows it is T4 itself.

**Findings (sweep complete, including crawl×transform):**

1. **FWHT actively hurts at this config.** Against the clean crawl-off control, identity (*no transform at all*, 1.1463) **beats** FWHT (1.1492) by −0.0029 (~3× noise). The Walsh basis isn't merely non-integral — gating raw channels works *better* than gating Walsh-frequencies here. This confirms the absorbability reasoning (the mixer's linear part never needed a basis) and sharpens it: the hand-picked basis was a small net negative.
2. **The learned butterfly stays home.** Given free choice of orthogonal gating basis (init = identity), it converged to 1.1455 — only −0.0008 below identity (**within the ~0.0010 noise floor**) and identical best val (3.5596). Gradient descent, free to rotate into any basis in its family, found nothing meaningfully better than no rotation. Together with (1): **no specially-good gating basis appears to exist** at this config — the gate is close to basis-indifferent, with FWHT slightly on the wrong side of indifferent. Note the butterfly costs +2,300 MiB train VRAM (fp32 butterfly-layer activations) for its within-noise gain; inference VRAM is unchanged.
3. **DHT and DCT land between identity and FWHT, statistically tied with each other** (1.1479 and 1.1478: ~+0.0016 over identity, ~−0.0013 under FWHT), as the basis-indifference picture predicts. Final ranking: butterfly (1.1455) ≤ identity (1.1463) < dct ≈ dht (1.1478/1.1479) < fwht (1.1492). The full spread across all five variants is just **0.0037** — every fixed basis loses to no-basis, the two smooth-frequency bases are interchangeable to 4 decimal places, and the entire transform axis is worth less than a fifth of the crawl effect. **The fixed-basis question is closed**: there is no spectral basis worth hand-picking for the gate at this config.
4. **Incidental but important: the T4 crawl-off datapoint.** The fwht control is *exactly* T4-with-crawl-off, so this sweep incidentally measured the crawl contribution at T4: **−0.0181** (1.1311 vs 1.1492) — substantially larger than the −0.0075 measured at T2 (lr=0.01). Wavelet crawl matters *more* in the T4 LR regime, not less; it is doing more work than the transform slot is.

Implication for [Multi-Transform Parallelization](#multi-transform-parallelization): the leading indicator is unfavorable — if the gate barely distinguishes bases (and the learnable basis stays at identity), parallel fixed bases are likely redundant perspectives, and the compound would mostly add capacity that MLP width provides more cheaply. dht/dct complete the picture.

**Crawl×transform verdict — the FWHT is deleted from the forward path.** crawl+identity reaches **1.1287**, beating T4 (crawl+fwht, 1.1311) by **−0.0024** — removing the transform doesn't merely match, it *improves*, and at lower train VRAM than the butterfly alternative. The result is doubly validated by additivity: the FWHT penalty measured independently in both regimes is consistent (−0.0029 crawl-off, −0.0024 crawl-on), and the crawl benefit measured at both transforms is consistent (−0.0181 at fwht, −0.0176 at identity: 1.1463 → 1.1287). Two effects, four measurements, clean stacking, no interaction term. crawl+butterfly (1.1297) repeats the now-familiar pattern a fourth time — within ~noise of identity (+0.0010), not worth +2,400 MiB train VRAM (though it did post the best val loss of the sweep, 3.5098, the BPB rank metric says identity). **T5 proceeds transform-free**: `mixer_transform: "identity"`, crawl on, K from the [K sweep](#in-progress-wavelet-crawl-dilation-window-k-sweep) below. The new T4-class reference for all subsequent ablations is **crawl+identity at 1.1287**.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Wavelet Crawl Dilation Window (K) Sweep

Wavelet crawl replaced the fixed per-level dilation with a **learned K-tap causal look-back**: at level ℓ, instead of pairing each position with the single sample `2^ℓ` steps back, the "odd" stream is a softmax-weighted convex combination of K distinct look-back offsets in a window centered on `2^ℓ` (shifted upward at fine levels so all offsets stay ≥ 1). Initialization places nearly all softmax mass (logit 5.0) on the base offset, so K=anything starts ≈ the standard wavelet and learns to spread only if it helps. Parameters: `levels × K` logits — **21 params at K=3** — making this possibly the highest BPB-per-parameter feature in the model (−0.0181 at T4 for 21 params).

This is a learned, normalized, dilated causal convolution over time — the same structural family as Hyena's implicit long convolutions, in miniature. The T4 crawl result (−0.0181, 2.4× its T2 value, the largest component win on the T4 line) suggests the convolutional axis has more headroom, which this sweep probes by widening the window.

**What bounds K:** K must be odd (symmetric window), and it is *not* bounded by channels — the mixture is over time offsets, shared across all channels. The window construction is `min_off = max(1, 2^ℓ − K//2)`, `offsets = [min_off .. min_off+K−1]` — when the symmetric window would underflow past offset 1, it **clamps and shifts upward** rather than recentering. This clamping sets the real bounds (an earlier version of this section said K ≤ ~385 from the symmetric formula — wrong, the clamp binds first): the deepest level (base 64) stays symmetric only to K=127; at **K=129 every level's window has clamped to the identical [1..129]**, and the hard cap is **K=255** (window [1..255] at all levels — beyond that, taps reach past the block and read pure zero-padding). Important nuance: identical windows do *not* mean identical levels — each level operates on the previous level's approximation, so the recursive cascade preserves a scale hierarchy by composition even when the windows fully overlap. As K grows, the architecture therefore *interpolates from a dyadic wavelet toward a depth-7 stack of full-context learned causal convolutions* (a Hyena-like object) — the K sweep is measuring exactly where on that continuum the value lies. Secondary bounds: softmax dilution of the base-offset init is mild (~95% at K=9, ~82% at K=33, ~61% at K=255 — safe), and compute grows linearly in K but stays memory-bound-negligible vs the mixer matmuls.

**Sweep (1ep, T4 base + crawl on + `mixer_transform=identity`, geometric K spacing; extend to K=33 only if K=17 still improves).** Base/reference is the crawl+identity (K=3) run — confirmed as the new T4-class best (1.1287, beats crawl+fwht), so the sweep and the T5-bound config share a lineage and the fwht fallback is moot.

| K | Window at level 0 / level 6 | BPB sliding | Best val | Δ vs K=3 ref | Run log |
|---|---|---|---|---|---|
| 3 (ref) | [1..3] / [63..65] | 1.1287 | 3.5105 | (ref) | [link](logs/wikitext-103_2026-06-10_18-06-13/log.txt) |
| 5 | [1..5] / [62..66] | 1.1248 | 3.4933 | −0.0039 | [link](logs/wikitext-103_2026-06-10_22-24-24/log.txt) |
| 9 | [1..9] / [60..68] | 1.1194 | 3.4796 | −0.0093 | [link](logs/wikitext-103_2026-06-11_02-43-01/log.txt) |
| 17 | [1..17] / [56..72] | 1.1162 | 3.4709 | −0.0125 | [link](logs/wikitext-103_2026-06-11_07-04-35/log.txt) |
| **33 (knee)** | [1..33] / [48..80] | **1.1156** | **3.4679** | **−0.0131** | [link](logs/wikitext-103_2026-06-11_11-36-16/log.txt) |
| 65 | [1..65] / [32..96] | 1.1148 | 3.4687 | −0.0139 | [link](logs/wikitext-103_2026-06-11_16-09-53/log.txt) |
| 129 | [1..129] / [1..129] | 1.1163 | 3.4694 | −0.0124 | [link](logs/wikitext-103_2026-06-11_20-59-19/log.txt) |
| 255 (cap) | [1..255] / [1..255] | 1.1173 | 3.4791 | −0.0114 | [link](logs/wikitext-103_2026-06-12_02-22-23/log.txt) |

**Status: every registered prediction has graded correct, including the knee.** K=5/9/17 all improve monotonically — K=17 is the best T4-class result yet recorded (**1.1162**; −0.0149 below the original crawl+fwht T4, and below every recurrence variant at a few hundredths of the parameter cost). The sharpened forecast's first clause landed: K=9→17 gained −0.0032, smaller than the K=5→9 step (−0.0054), with per-tap efficiency falling from −0.0014 to −0.0004/tap — the deceleration profile of an approaching knee. The knee clause graded correct too: K=17→33 gained −0.0006 — smaller still, and **within the ~0.0010 noise floor**, i.e. the curve flattens in [17, 33] exactly where the half-width≈base/4 law put it (level 6 satisfied at ±16). Cumulative crawl-K arc: 1.1287 → **1.1156** (−0.0131), every step of it predicted from weight readouts before the runs landed. The final clause graded correct as well: K=33→65 is flat (−0.0008, within noise; 1.1148 the numerical minimum), and **K=129/255 regress** (+0.0015/+0.0025 vs 65, with best val degrading in step). The plateau is [17..65]; **T5 carries K=33** — tied with 65 on BPB (sub-noise), best val of the sweep, and the 'everyone satisfied' point of the half-width law with the most scale structure preserved (at K=65, six of seven windows have merged).

**The identity-experiment verdict — the wavelet wins its own vote.** K=129/255 are the configurations where every level's window is identical and the dyadic lattice survives only as initialization — the architecture's Hyena-limit (depth-7 full-context learned causal conv). It *loses*: monotone regression past the knee, with K=255 (1.1173) worse than K=17. Performance peaks exactly where scale-proportional structure is preserved and degrades as the structure dissolves — the dyadic wavelet prior is load-bearing, not scaffolding the convolution subsumes. (Honest caveat: at 1ep, part of the large-K regression may be optimization burden — 255-way softmaxes diluting the init — rather than purely structural inferiority. The discriminating readout is a dump on the K=255 checkpoint: if its learned kernels *re-derive* narrow-fine/wide-coarse structure despite full freedom, that strengthens the structural reading further — the model rebuilds the wavelet when nobody makes it.)

#### Mechanism readouts (weight-level evidence; chronological — every prediction below was registered before the run that tested it)

**Offset-preference readout (K=3, from the crawl+identity checkpoint via [tools/dump_crawl_offsets.py](tools/dump_crawl_offsets.py)).** The learned softmax weights over each level's 3-tap window, read directly from `dilation_logits` (the tied reconstruct exposes an identical copy, confirming weight tying):

| Level | Base | Learned weights (offset:weight) | Mass off base |
|---|---|---|---|
| 0 | 1 | **1:0.986** · 2:0.007 · 3:0.007 | 1.4% |
| 1 | 2 | 1:0.247 · **2:0.561** · 3:0.192 | 43.9% |
| 2 | 4 | 3:0.331 · **4:0.414** · 5:0.255 | 58.6% |
| 3 | 8 | **7:0.380** · 8:0.317 · 9:0.303 | 68.3% |
| 4 | 16 | **15:0.381** · 16:0.287 · 17:0.332 | 71.3% |
| 5 | 32 | **31:0.374** · 32:0.278 · 33:0.347 | 72.2% |
| 6 | 64 | **63:0.362** · 64:0.280 · 65:0.359 | 72.0% |

Three readable facts: (1) **monotone precision-to-diffusion gradient** — level 0 keeps 98.6% of its mass on exact adjacency (the bigram link is sacred), and off-base mass grows monotonically with scale, saturating at ~72%; (2) **coarse levels converge to near-uniform smoothing** — by level 3+ the weights approach uniform (⅓ each, slightly *under*-weighting the base), i.e. the model turns the single dilation tap into a 3-tap low-pass filter: at distance ~64 it wants a neighborhood average, not a precise sample; (3) **mild recency bias** — the shorter offset consistently edges the longer one at mid levels (e.g. 7:0.380 vs 9:0.303), answering what a K=2 ablation would have asked, without a run. The near-uniform saturation is the key signal: the model appears to have **maxed out the spread the K=3 window allows**, which is the signature of wanting a wider kernel. **Registered prediction (before the K-sweep results land): K=5/9 will improve BPB, with the gains concentrated at coarse levels, and the learned weights at higher K will again spread toward smooth/uniform at coarse scales.** This readout also mildly favors the *smearing* interpretation over the *new-lattice* interpretation for the [p-adic gate](#multinodal-mode-product-of-experts) — the model asks for width around dyadic points, not different lattice points per se.
**Mechanism check (K=9 offset readout): confirmed, and sharpened into a scale-proportional-width law.** Per-level learned weight shapes at K=9:

| Level (base) | Mass off base | Learned kernel shape |
|---|---|---|
| 0 (1) | 5.2% | point mass on offset 1 — the bigram link stays sacred even with 8 alternatives |
| 1 (2) | 38.4% | narrow ±1; **ignores 6 of its 9 taps** (and is *more* concentrated than at K=3: 38.4% vs 43.9%) |
| 2 (4) | 79.1% | asymmetric, mode at offset 3 (0.250) *below* base, dead tail past 7 — wants width ~±3, centered slightly short of dyadic |
| 3 (8) | 86.5% | ~uniform over [4..12], mild short-offset tilt |
| 4 (16) | 87.2% | ~uniform over [12..20] — content at this width |
| 5 (32) | 87.6% | **U-shaped: window edges carry the maxima** (28: 0.149, 36: 0.144 > base 32: 0.124) |
| 6 (64) | 87.4% | **U-shaped, strongest** (60: 0.151, 68: 0.162 > base 64: 0.126, dips beside base) |

Reading: each scale requests a kernel width roughly **proportional to its base offset** — width ~1 at base 1, ~3 at base 2, ~7 at base 4, ~9 fits bases 8–16, and bases 32/64 press against the window edges (U-shape = "give me more"). That is a **constant-Q filterbank** — constant *relative* bandwidth per scale — which is precisely the wavelet philosophy, spontaneously reconstructed by free optimization. **This argues directly against the convergence-to-Hyena worry**: a model drifting toward Hyena would spread *every* level toward full-context kernels, but the fine levels actively *refuse* width they were offered (level 1 uses 3 of 9 taps; level 0 uses 1). The hierarchy is being affirmed where it matters and widened only where scale-proportionality demands it. **Refined registered prediction (before K=17/33 land):** K=17's gain comes mostly from levels 5–6 (the only U-shapes) and is smaller than the K=5→9 step; the knee lies in K∈[17, 65]; at the knee, the dump shows no U-shapes (every level content). If large K keeps winning beyond that, the dump distinguishes the two endgames directly: *wavelet-re-derived* (kernels stay scale-structured: narrow-fine/wide-coarse) vs *Hyena-converged* (all levels wide) — the weights vote, and [tools/dump_crawl_offsets.py](tools/dump_crawl_offsets.py) reads the ballot. A **per-level K ∝ 2^ℓ ("constant-Q crawl")** variant is the natural post-knee refinement: it gives each level the width it measurably asks for without dead taps at fine levels.

**Three-aperture cross-check (K=5 readout added): the law goes quantitative.** Reading the same levels through three window widths (K=3/5/9) shows aperture-*invariant* preferences — level 0 keeps a point mass on offset 1 at every K (98.6%/97.0%/94.8%); level 1 keeps the identical narrow ±1 kernel at every K (off-base 43.9%/43.7%/38.4% — it ignores whatever extra taps it is offered); level 2's mode sits at offset 3, *below* its dyadic base, at both K=5 (0.282) and K=9 (0.250). The decisive observation is **level 4, which flips from starved to content as the aperture grows**: U-shaped/edge-pressed at K=5 (edges 0.252/0.212 over middle ~0.17) but relaxed at K=9 — exactly as the registered prediction required ("levels 3–4 should look starved at K=5"). The starvation boundary moves up one level per doubling of K, which pins the constant: **desired half-width ≈ base/4 for levels ≥ 3** (level 3 content at ±2, level 4 at ±4; level 5 should satisfy at ±8 = K 17, level 6 at ±16 = K 33). One honest partial: level 3 at K=5 shows a hump at offset 7 with only mild edge-lift, not a clean U. **Sharpened forecast: the knee is at K≈33** — K=17 improves (level 5 satisfied, level 6 partially), K=33 adds a smaller gain (level 6 satisfied), K=65+ flat-to-regressing. Fine levels (0–2) deviate from pure constant-Q in an interpretable direction: level 2 wants a *left-shifted* kernel (mass at offsets 1–3, dead tail past 7) — shorter-than-dyadic pairing at fine scales, consistent with the recency bias seen at K=3.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Structure Factoring

The organizing principle behind several of this project's wins, named and adopted as an explicit methodology (2026-06-11): **make the model opaque only about what genuinely requires learning.** Every bit of structure that can be prescribed explicitly — positional relationships, corpus statistics, memorization — is a bit the opaque weights no longer need to encode. Factor known structure into fixed or low-parameter inspectable components; let the network learn only the residual.

**The design rule, from our own evidence: factoring succeeds at interfaces and as additive parallel channels; it fails as mandatory mid-network re-encodings.** Wins: wavelet crawl (−0.0181 at T4 for 21 readable logits), PKM/FwPKM (explicit sparse memory, readable top-k addressing), the additive memory composition (per-factor contribution measurable by construction), and the planned semantic embedding / frozen-tied head (prescribing the input/output interfaces). The documented loss: the FWHT — a mandatory in-series re-encoding that [identity beat](#done-mixer-transform-ablation).

**It is a closed loop, not a bag of tricks**: (1) hypothesize — justifiably, from readouts or probes, never aesthetics — what the model encodes implicitly; (2) factor it out per the design rule; (3) measure acceptance with matched controls and *predictions registered before results land*; (4) **read back the factored structure's learned state** — because it is explicit, it is readable, and the readout generates the next hypothesis. The [crawl-K arc](#in-progress-wavelet-crawl-dilation-window-k-sweep) is the first completed cycle: 21 logits read → width hypothesis → prediction registered → confirmed by BPB (1.1287 → 1.1194) → readout sharpened into the constant-Q law → which proposed its own refinement. Each pass yields a capability gain *and* a mechanistic finding from the same artifact.

**What this buys interpretability — the precise claim.** Each factored component is interpretable *by construction*: its semantics are specified before training, and its learned state is readable by design (the crawl logits being the live proof). Every cycle therefore converts a slice of opaque capacity into transparent capacity — the interpretable fraction of the model ratchets up monotonically. The half to keep honest is the **residual**: it shrinks in *content* but is not automatically more *readable* — the remaining MLP weights stay as polysemantic as ever, merely responsible for less. So the claim is "each cycle grows the interpretable fraction and shrinks the uninterpreted remainder," not "the whole model becomes interpretable." Iterated to its empirical limit, the loop *locates the boundary between what was always just statistics and what is irreducibly learned computation* — everything on one side named, factored, and readable; everything on the other a genuine mystery, now minimal and precisely delimited, which is exactly where the SAE and semantic-embedding tooling should be aimed.

Next concrete test: the **frozen skip-gram logit prior** (count-based next-token tables at dyadic offsets, added at the head with learned per-offset scalars). Full recipe, measurements, and the standing failure-mode cautions (error-corrector residuals; superposition unaffected) in [plans/structure_factoring.md](plans/structure_factoring.md).

Also planned for [post-release work](#other-post-release-plans).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Step-Time Speedups

> **Status (2026-06-10): descoped to a pre-B200 checklist.** The original plan — profile step time across `bs ∈ {256 … 16384}` and attack whatever dominates at 16384 — targeted the deprecated bs=16384 line. Production is bs=256 (long context via [BBCE](#bisected-block-context-extension)/cross-window state, not raw block growth), so the multi-block-size scaling study answers a question the architecture no longer asks. One of the section's premises also didn't survive contact: the "fused SwiGLU" candidate doesn't apply (the MLP is plain GELU, which `torch.compile` already fuses well). The profiler itself exists at [`OLD/profile_step.py`](OLD/profile_step.py) (per-component `record_function` hooks + `with_modules` attribution, multi-bs, Chrome traces; built 2026-05-02, so it predates the complex/transform/recurrence additions and will need a small refresh against current `train.py`/`model.py` before use). Note the cost concern that motivated descoping the *study* was misplaced in one respect: speed profiling needs no trained models and no per-bs tuning — step time is weight-value-independent, so any profiling run costs minutes, not training runs. The study is dropped because its target regime is dead, not because it was expensive.

What survives is the part with real money attached — **cheap step-time wins applied just before the long headline runs** (3-seed × 5-epoch T5 + PG-19), where a 10–20% step-time saving compounds into days of GPU time. Checklist, in cost order:

1. **`compile_mode: "reduce-overhead"`** — already a config key; one short A/B timing run at the T5 config. CUDA-graph capture typically helps small-batch, kernel-launch-bound regimes like bs=256/MBS=8. Watch for recompile churn from the eval/benchmark interleave.
2. **`foreach`/`fused` Adagrad** — check the pod's torch version for fused Adagrad support; one-line optimizer change, one short run. ⚠️ Verify loss-curve equivalence over a few hundred steps before trusting it: the fp16 + `eps=2e-13` regime is numerically delicate and fused-kernel accumulation order differs.
3. **(Only if 1–2 disappoint) component attribution via the existing profiler** — refresh [`OLD/profile_step.py`](OLD/profile_step.py) against current `train.py`/`model.py` (likely minutes: it hooks submodules by class name, so the main drift risk is the dataset-loader/config API, not the model), then run it at the production config only (`--block_sizes 256`) to see whether lifting / mixer / MLP / FwPKM dominates before considering anything architectural (e.g. low-rank lifting predict/update). No multi-bs sweep.

Slot this alongside the final regularization sweep in the pre-B200 window — same "tune once, right before the expensive runs" logic.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### T5 Baseline

Time to establish a new baseline. Here, we'll incorporate the best features performance-wise so far and roll them all together.

**Regularization 2×2 (transfer + coupling test).** The [Dropout](#dropout) coordinate descent (final stack: emb 0.18 / proj 0.09 / mix 0.09 / mlp 0.10 / lm_head 0.216, −0.0026 at L=1) and [Weight Decay](#weight-decay) sweep (2e-6, −0.0035 at L=1) were both tuned at L=1/1-epoch and are single-seed. Before folding them into the production baseline they must (a) be confirmed to *transfer* to T5 scale — the dropout-down direction is the fragile one and may flip if a deeper/wider T5 wants more dropout, whereas WD-up is more likely scale-monotone — and (b) be checked for *coupling*, since dropout and WD are both regularizers and may trade off (a ridge) rather than stack additively. A 2×2 factorial answers both with 3 runs on top of the **T5 pre-baseline** (the old/old corner — T4 dropout defaults + WD=1e-6 on the locked recipe; conveniently already measured, since the K=33 sweep run is exactly this configuration). All four cells identical except the dropout/WD axes.

| Cell | Dropout | Weight decay | BPB sliding | PPL sliding | Best val | Δ vs T5 base | Run log |
|---|---|---|---|---|---|---|---|
| T5 pre-baseline (old / old) | T4 defaults | 1e-6 | 1.1156 | 32.62 | 3.4679 | (ref) | [link](logs/wikitext-103_2026-06-11_11-36-16/log.txt) |
| + new dropout only | descent stack | 1e-6 | 1.1121 | 32.27 | 3.4623 | −0.0035 | [link](logs/wikitext-103_2026-06-12_11-21-32/log.txt) |
| + new WD only | T4 defaults | 2e-6 | 1.1140 | 32.46 | 3.4654 | −0.0016 | [link](logs/wikitext-103_2026-06-12_15-53-50/log.txt) |
| **+ both** | descent stack | 2e-6 | **1.1118** | 32.23 | **3.4580** | **−0.0038** | [link](logs/wikitext-103_2026-06-12_20-25-51/log.txt) |

**Reading:** *new-dropout-only* vs base = does the L=1 dropout stack transfer to T5; *new-WD-only* vs base = does WD=2e-6 transfer; *both* vs (sum of the two single-axis Δs) = additive (independent → fold both in) or coupled (ridge → keep the better single axis, or tune jointly at the [final regularization sweep](#final-regularization-sweep)). Edge-winner directions to continue if confirmed: dropout proj/mix/lm_head ↓, WD ↑. Single-seed at T5 too — the chosen recipe still gets a seed-check before B200.

**Verdict (2×2 complete): adopt both — dropout carries it, WD rides along free.** Dropout-only transfers cleanly (−0.0035, 3.5× noise); WD-only is marginal (−0.0016, ~1.6× noise); *both* is **sub-additive** (observed −0.0038 vs −0.0051 if the singles stacked) — the two regularizers overlap, as expected. But *both* is the best cell on **both** metrics (BPB 1.1118, best val 3.4580 — the val edge over dropout-only's 3.4623 is above the val noise floor), and crucially WD=2e-6 does **not hurt** on top of the dropout stack (the user's adoption criterion). So the T5 baseline takes **the descent dropout stack + WD=2e-6**. The dropout-down direction held at T5 scale rather than flipping — the L=1 tuning transferred, contra the fragility worry.

**Recurrence stacking test (1ep, baseline candidate by election).** Input-anchored N=5 K=1 mixer recurrence (`mixer_recurrence_residuals=true`, no gate caching — the −0.0071 winner at 1.1240 on the *old* recipe, [log](logs/wikitext-103_2026-05-30_04-52-42/log.txt)) run on the new recipe (identity, K=33). The interaction is untested in either direction: recurrence iterates the *channel* mixer toward its fixed point while crawl-K widens *time* taps (different axes → additivity plausible), but both enrich temporal processing (→ sub-additivity also plausible). **Folds into the declared baseline iff it clears the noise floor vs the pre-baseline (1.1156).** Cost if adopted: ~+53% train time and ~1.3× inference latency — with the mitigation that the [inference-depth study](#inference-depth-flexibility-with-mixer-recurrence-train-deep-infer-shallow) showed N′=4 serving is quality-free and N′=3 cheap, so the adopted baseline would inherit an anytime-inference knob.

| Variant | BPB sliding | Best val | Δ vs pre-baseline | Run log |
|---|---|---|---|---|
| pre-baseline (no recurrence) | 1.1156 | 3.4679 | (ref) | [link](logs/wikitext-103_2026-06-11_11-36-16/log.txt) |
| + N=5 K=1 input-anchored | 1.1121 | 3.4619 | −0.0035 | [link](logs/wikitext-103_2026-06-13_00-58-58/log.txt) |

**Verdict: recurrence's benefit halved on the new recipe — now a borderline keep, leaning drop.** On the old recipe (fwht, K=3) N=5 K=1 was worth −0.0071; here it is **−0.0035** — exactly the sub-additivity predicted: crawl-K=33 already absorbed roughly half of what recurrence used to contribute (both enrich temporal processing). The remaining −0.0035 clears the noise floor but costs **+53% train time** (8,814 vs 7,790 MiB, and ~1.3× the wall-clock that propagates into every B200 arm). For reference it matches the *free* dropout stack's −0.0035. Recommendation: **defer it from the declared baseline** — fold in only if the confirmation/depth runs show headroom that justifies the train-time tax, since the N′=4-free inference knob recovers serving cost but not training cost. Flagged for the user's call (elected candidate).

**Capacity restoration (1ep, L=1, vs the shared pre-baseline).** The ablation line deliberately runs a reduced base for sweep cheapness (mlp_expansion 10, PKM off, FwPKM 8281, tied head), while the production headline carries full capacity (mlp 20, PKM+FwPKM 16384, untied head). Each component is restored individually on the **pre-baseline recipe** (identity, K=33, T4 dropouts, WD=1e-6 — the same 1.1156 reference the 2×2 and recurrence tests use, so every decision axis shares one anchor), then all together — measuring per-component value under the post-ablation architecture plus the A5000 resource/runtime numbers that calibrate the B200 plan. The combined row is the presumptive new-baseline capacity form; cross-axis interactions are caught by the declared baseline's own confirmation run.

| Restoration | Config delta | Params | BPB sliding | Best val | Δ vs T5 base | Train VRAM | Train time | Run log |
|---|---|---|---|---|---|---|---|---|
| T5 base (reduced) | — | 392.98M | 1.1156 | 3.4679 | (ref) | 7,790 MiB | ~4.5h | [link](logs/wikitext-103_2026-06-11_11-36-16/log.txt) |
| + MLP 20 | mlp_expansion 10→20 | 476.89M | 1.1081 | 3.4486 | −0.0075 | 9,390 MiB | ~5.1h | [link](logs/wikitext-103_2026-06-13_08-01-10/log.txt) |
| + PKM | pkm_enabled on, 16384 keys | 430.99M | 1.1151 | 3.4675 | −0.0005 | 8,515 MiB | ~4.7h | [link](logs/wikitext-103_2026-06-13_13-10-36/log.txt) |
| + FwPKM full | fwpkm_num_keys 8281→16384 | 409.65M | 1.1160 | 3.4702 | +0.0004 | 8,964 MiB | ~4.6h | [link](logs/wikitext-103_2026-06-13_17-55-32/log.txt) |
| + untied head | tie_embedding_to_lm_head off | 495.91M | 1.1190 | 3.4769 | +0.0034 | 9,753 MiB | ~4.7h | [link](logs/wikitext-103_2026-06-13_22-31-16/log.txt) |
| + all restored | all four | 634.49M | 1.1136 | 3.4624 | −0.0020 | 12,460 MiB | ~5.5h | [link](logs/wikitext-103_2026-06-14_03-14-48/log.txt) |

**Verdict: MLP-20 is the only capacity component that pays; stacking all four is *worse* than MLP alone.** Component deltas vs the pre-baseline (1.1156), each on the reduced base:

- **MLP 20 — clear win** (1.1081, −0.0075, +83.9M). Wider FFN, trains fast, the single largest capacity lever as the architecture history predicted. **Include.**
- **PKM @16384 — within noise** (1.1151, −0.0005, +38.0M). But this adds PKM *on top of the FwPKM the baseline already runs* (below), so it measures a **second** PKM-family memory — redundant with the first, not "memory is useless."
- **FwPKM widening — within noise / null** (1.1160, +0.0004, +16.7M). ⚠️ The baseline **already has `fwpkm_enabled=True` at 8281 keys**, so this row measures only the 8281→16384 *widening*, not FwPKM-vs-nothing. FwPKM-at-all is a *carried baseline component*; more keys just don't help. (Reconciles with runs.md — memory is still on.)
- **Untied head — actively harmful** (1.1190, +0.0034, +102.9M — the most expensive single component). Worst on both BPB and best val.
- **All four — 1.1136 (−0.0020), but +241.5M params and *+0.0055 worse than MLP alone* (1.1081).** Dragged down by the harmful untied head; the *extra* memory (2nd module + widening) is redundant, not additive.

So "restore all production capacity" was the wrong instinct *at this scale*. **All three implementations were read and verified — PKM, FwPKM, and the LM-head tying are textbook-correct, no bugs** (the user flagged the surprise; the cause is measurement-scope + training budget, not a regression). Two mechanisms: (1) the untied head is the active drag (undertraining, below); (2) the *extra* memory is redundant — the baseline already carries FwPKM@8281, so the PKM and FwPKM-widening runs stack a second/bigger memory onto an existing one, and **crawl-K=33 has absorbed much of the temporal work memory used to do**. Memory itself is not dead: one module at 8281 stays in the baseline, consistent with runs.md.

**Important undertraining caveat (why this is not the final word for the headline).** The three param-heavy components (untied head +103M, PKM +38M, FwPKM +17M) each add large matrices that need *training time* to pay off, and the old 1.0140 headline used all of them at **L=2 / 5 epochs**. A 1-epoch verdict is undertraining-confounded for exactly these — the untied head's −0.0034 is the classic signature of a big output matrix that hasn't converged. So they are **excluded from the 1ep-declared T5 baseline but re-tested at 5ep** in the [More Epochs](#more-epochs) arms before any final headline-capacity decision. MLP's win, by contrast, is the kind that transfers (it converges fast).

**Declared capacity form: MLP-20 + the baseline's FwPKM@8281** (PKM off, FwPKM not widened, head tied). i.e. identity + crawl-K=33 + reg(both) + mlp_expansion 20, carrying the one memory module the base already had. Every *added* component (MLP) has a measured benefit; the extras that didn't pay (second memory, wider memory, untied head) are left out. If it holds at 5ep, the new headline is *both better and smaller* than the old — which carried PKM@16384 **and** FwPKM@16384 **and** an untied head, the components this sweep showed add nothing (or harm) at this scale.

**T5 Baseline (declared).** Once the three decision blocks above resolve, the chosen combination **gets its own confirmation run** (interactions between axes are validated here, since each block measured against the shared pre-baseline) and is declared *the* T5 baseline — the single reference row that every subsequent section ([More Layers](#more-layers), [More Epochs](#more-epochs), PG-19, B200) measures against. Components already locked by the ablation arc, with provenance: `mixer_transform = identity` ([transform ablation](#done-mixer-transform-ablation)), `wavelet_crawl = true, K = 33` ([K sweep](#in-progress-wavelet-crawl-dilation-window-k-sweep)), levels = 7, per-scale widths [1.0×4, 0.5×4], wavelet norms on, lr = 0.0225 (T4 reference regime). All three decision blocks resolved: dropout stack + WD=2e-6 (2×2 — both, sub-additive but WD doesn't hurt); recurrence **off** (halved to −0.0035 on the new recipe, not worth +53% train time); capacity **none — MLP-20 only, all PKM-family memory removed** (PKM/FwPKM-widen/untied inert-or-harmful, and FwPKM-at-all tested tied-but-removable; all redundant with crawl-K+MLP at 1ep, re-tested data-rich by the 5ep ceiling arm). The combined confirmation run landed at 1.1082 / 3.4472 / 476.89M — tied with the MLP=20-only run (1.1081), confirming the regularization is net-neutral at 1ep (no overfitting to fix; it's insurance for the 5ep regime). The **paired FwPKM-off ablation then landed at 1.1073 / 3.4479 / 455.55M — statistically tied** (the −0.0009 BPB is sub-noise, val +0.0007) at **−21.3M params and −472 MiB**, so FwPKM is removed on parsimony. **The declared T5 baseline is therefore the no-memory config: 1.1073 / 455.55M** (−0.0238 vs the original T4). This completes the memory-redundancy finding — *every* PKM-family component (2nd module, widening, the last module, untied head) is removable at no measured cost on this recipe: **crawl-K + MLP fully substitute for associative memory at 1ep/WT103.** The model is now embedding → wavelet (identity transform, crawl K=33) → gated mixer → MLP → tied head. It beats the *old* headline's L=1/1ep equivalent (1.1648, 586.15M) by **−0.0575 at −131M params** — better and smaller. ⚠️ Data-starvation caveat stands: this is the 1ep/WT103 verdict (~0.2 tok/param); the [L=5/5ep full-capacity ceiling arm](#more-epochs) and the scale-up datasets re-test whether memory pays when data-rich.

| T5 Baseline | Transform | Crawl K | Dropout | WD | Recurrence | Capacity | Params | BPB sliding | PPL sliding | Best val | Train VRAM | Run log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **T5 baseline (declared)** | identity | 33 | descent stack | 2e-6 | off | MLP-20, **no memory** | 455.55M | **1.1073** | 31.78 | 3.4479 | 8,918 MiB | [link](logs/wikitext-103_2026-06-14_16-08-56/log.txt) |

**Other T5 baseline settings not shown in the table** (the full audited ledger — carried, pending, and tested-but-excluded):

*Carried — architecture:* levels=7; per_scale_mixer_widths=[1.0×4, 0.5×4]; `wavelet_decomp_norm`/`recon_norm` on; lifting: shared weights, dense (offdiag "none", no diaglowrank, no level-sharing), hidden_mult=1, haar init, lifting_dropout=0; low_rank=4 (the 5-ep revert decision in runs.md); cross_scale_gating on; mixer gate on (silu); mixer_depth=1; learned_residual on; **decompose_bypass on + cross_window mean carry on** (the long-range mechanism in every T4/T5-line run); per_layer_embedding on (inert at L=1; activates in [More Layers](#more-layers)); mlp_layers=2.

*Carried — training:* Adagrad (eps=2e-13, initial_accumulator=0); lr=0.0225 → min_lr=4.5e-4, warmup_fraction=0.3, grad_clip=1.0; block_size=256, micro_batch=8, grad_accum=1; **random batching**; fp16 AMP + TF32; compile (default mode); seed 1337 (recipe gets the 3-seed check before B200).

*Pending the two decision blocks above:* dropout stack and WD (2×2); capacity quartet (mlp_expansion 10→20, PKM on @16384, FwPKM 8281→16384, untied head).

*Audited and excluded — tested, with recorded reasons:* the **parameter-costly recurrence variants** (distinct banks K>1, dense recurrence, gate caching — best 1.1227 at +58.85M params vs crawl-K's 1.1156 at 231 logits); **N=5 K=1 input-anchored recurrence is the exception**: zero params, elected as a baseline candidate, pending its stacking test on the new recipe (see above — all prior recurrence numbers predate identity/K=33); the `--infer_n` truncation infra becomes a serving feature if it folds in; **sequential batching + SSM/BPTT bypass variants** (best gain −0.0045 on the sequential reference, several times smaller than sequential mode's ~+0.019 cost vs random batching); **decompose_bypass_ema** (1-ep win inverted at 5 ep); **BBCE** (context-extension tool, off at 256); **complex wavelet/mixer** (closed negatives vs matched controls); **FWHT/DHT/DCT/butterfly** (identity won; Thue-Morse signflips and the FWHT input cap are Hadamard-boundary tools, moot under identity); **untied reconstruction** (tied drastically better) and **multi-basis lifting** (NaN; dropped feature); the **stability bundle** (stable_parametrization / spectral norm / stab_* scalings — wavelet norms cover stability at this LR); **compression structures** (lifting/MLP offdiag masks, diaglowrank, sparse-PQ embedding — no real savings or BPB cost; opt-in tools only); stochastic_depth=0; multinodal / looped_blocks / 2D-wavelet modes off (separate tracks); fwpkm_inference_updates off.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### More Layers

Adding layers again after most of the tuning and architectural test ablations — the depth axis re-measured on the **post-ablation recipe** (identity transform, crawl on, K = the [K-sweep](#in-progress-wavelet-crawl-dilation-window-k-sweep) winner, T5 regularization verdicts folded in). The prior depth findings (deep >= wide at C=512; 30L regressing past 20L; the L=2 headline) all predate the FWHT deletion and the crawl-K widening, so the depth response curve needs re-establishing: crawl-K reshaped *where* temporal mixing happens, which plausibly changes what additional layers contribute.

1-epoch sweep at the T5 recipe (A5000-feasible; ~linear runtime in L). L=4 included if VRAM permits (expected to fit at the reduced ablation base — verify at launch). These 1ep runs double as the **pruning gate for [More Epochs](#more-epochs)**: only depths that hold up here graduate to the 5-epoch B200 arms.

| Layers | Capacity | learned_residual | Params | BPB sliding | Best val | Delta vs L=1 | Train VRAM | VM | Run log |
|---|---|---|---|---|---|---|---|---|---|
| 1 (T5 base) | no-memory | on | 455.55M | 1.1073 | 3.4479 | (ref) | 8,918 MiB | A5000 | [link](logs/wikitext-103_2026-06-14_16-08-56/log.txt) |
| 2 | no-memory | on | 690.66M | 1.1014 | 3.4370 | −0.0059 | 13,403 MiB | A5000 | [link](logs/wikitext-103_2026-06-14_22-31-13/log.txt) |
| 3 | no-memory | on | 925.78M | 1.0945 | 3.4098 | −0.0128 | 17,887 MiB | A5000 | [link](logs/wikitext-103_2026-06-15_06-58-47/log.txt) |
| 3 (learned-α off)§ | no-memory | **off** | 925.78M | 1.0933 | 3.4060 | −0.0140 | 17,887 MiB | 4090 | [link](logs/wikitext-103_2026-06-16_14-15-11/log.txt) |
| 3 (no residual)¶ | no-memory | n/a | 925.77M | diverged @~30k (reached val 4.90) | 4.90 | — | — | 5090 | [link](logs/wikitext-103_2026-06-17_18-16-28/log.txt) |
| 4 | no-memory | on | 1160.89M | 1.0890 | 3.4032 | −0.0183 | 22,372 MiB | 4090 | [link](logs/wikitext-103_2026-06-16_04-11-30/log.txt) |
| **4 (full capacity)** | **MLP-20 + PKM@16384 + FwPKM@16384 + untied** | on | 1567.91M | 1.0908 | 3.4010 | −0.0165 | 30,647 MiB† | 5090 | [link](logs/wikitext-103_2026-06-16_21-27-14/log.txt) |
| **5** | no-memory | on | 1396.01M | 1.0831 | 3.3887 | −0.0242 | 26,856 MiB | 5090 | [link](logs/wikitext-103_2026-06-17_10-06-29/log.txt) |
| **6+ (iterative)** | no-memory | on | — | as needed | as needed | — | — | 5090/B200‡ | — |

† L=4 full-capacity measured **30,647 MiB** on the 5090 (32 GB) — well above the earlier ~23.8 GB estimate, so it does **not** fit a 24 GB card, and even L=5 *full* would exceed 32 GB (lean L=5/L=6 still fit). Its in-process benchmark first OOM'd on the checkpoint reload (training state still resident) and silently reported a garbage BPB (PPL 61k vs val 3.40); recovered via a fresh `benchmark_only` pass (BPB 1.0908), and `train.py` now frees the optimizer/model before the reload so future large runs won't repeat it. ‡ L≥5 exceeds 24 GB (~+4.5 GB/layer → L=5 ~26.9 GB, L=6 ~31.4 GB): needs a 5090 (32 GB, ~L=5–6) or B200 (L=7+). Hardware ladder: **A5000 / 4090 (24 GB) ≈ L=4 ceiling**, 5090 ≈ L=5–6, B200 for deeper.

§ `learned_residual=false` — drops only the per-sublayer learned scalar α (init 1.0); the residual `x = x + f(x)` is unchanged. This is the α-*scaling* control (does the model want to rescale the residual?), **not** a residual ablation. ¶ `disable_residual=true` (+ `per_layer_embedding=false`) — the genuine no-residual ablation: `x = f(x)`, no cross-layer carry *and* no embedding re-injection. It also **drops `proj_out`'s 1e-3 "spectral epsilon" init** (a residual-regime "start-near-identity" trick) — without a residual that init starves the spectral signal to ~1e-3 and `ln2`'s backward amplifies the gradients → NaN (see *Result* below). Both rows are isolation controls, not headline candidates.

**Iterative deepening protocol (search for the max depth).** Depth's non-diminishing returns through L=4 (lean L=4 landed at 1.0890, −0.0055 on L=3 — the per-layer gain holding ~constant at ≈−0.006) mean the ceiling is unknown, so we find it greedily: run **L=5** with the memory setting that won the **L=4 lean-vs-full** comparison; if it beats the previous depth by **more than the ~0.0010 BPB noise floor**, bump to **L=6** (same setting), and repeat — L=7, L=8, … — stopping when either (a) a depth's gain falls within noise of the previous, or (b) budget/VRAM runs out. The deepest depth that still cleared noise is the **max**, which feeds the [More Epochs](#more-epochs) "Max from More Layers" row and the [More Width](#more-width-c) "max layers" cells. ⚠️ **VRAM ceiling per GPU** (the run grows ~+4.5 GB/layer at C=2048): **A5000 / 4090 (24 GB) ≈ L=4**, **5090 (32 GB) ≈ L=5–6**, **B200 for L=7+** — so "budget" includes VRAM, not just time. (Data-starvation caveat: this is a 1ep search; the winner gets its 5ep confirmation in More Epochs, where the optimum may shift.)

**Residual contribution at depth (corrected control).** The old depth verdict ("little gained past L=2") predates `learned_residual`, so it may have been confounded — L=3 with a weak cross-layer path, not L=3 inherently. Isolating that needs care, and the first control was mis-specified: **`learned_residual=false` does not remove the residual.** It drops only the per-sublayer learned scalar α (init 1.0); the connection `x = x + f(x)` is unchanged ([model.py](model.py) — the non-α branch is still additive). So the L=3 on-vs-off `learned_residual` pair tests *α-scaled vs plain (unscaled) residual* with the stream fully intact in **both** — which is exactly why they came out near-identical — now measured: α-off **1.0933** vs α-on **1.0945**, a −0.0012 gap right at the 0.0010 noise floor (off marginally ahead): α initializes at 1.0 = plain residual and the model keeps it there. The finding is that the **learned scaling is inert** (the residual wants no rescaling) — *not* that the residual is inert; the depth question is untouched.

The genuine test is the new **`disable_residual=true`** flag: it removes the carry entirely (`x = f(x)`, no `+ x`) in all three forward paths, so each layer *replaces* the stream instead of correcting it. One confound must be closed first — `per_layer_embedding` (**on** in config) re-injects the token embedding at every block (`x += γ·token_embeddings`), a learnable input→layer skip that can stand in for the residual; it is the cross-block analog of the input-anchoring that flipped recurrence depth from harmful to helpful ([Recurrence with residual](#done-recurrence-with-adagrad-with-residual)). So the clean arm also sets **`per_layer_embedding=false`**. Reading: if no-residual L=3 collapses toward (or below) L=1, the residual stream is the load-bearing depth mechanism — the expected outcome, consistent with the transformer residual-stream literature; if it holds up, depth survives without a residual here, the genuinely surprising result, and *only then* are the flat-structure / C-as-primary-axis implications on the table. All headline runs keep the production residual on; the α-off and no-residual rows are isolation controls.

**Result — residual is load-bearing; the failure runs deeper than the init.** Two runs. The **first** `disable_residual` run (07-28) stalled at val ~7.46 then NaN'd at step 5k — confounded by `proj_out`'s 1e-3 "spectral epsilon" init (a residual-regime "start-near-identity" trick: with no residual to carry it, the spectral output is starved to ~1e-3 and `ln2`'s backward amplifies gradients ~100–300×/layer). `model.py` now drops that 1e-3 init when `disable_residual=true`. The **corrected re-run** (`T5_L3_nores_v2`, 18-16-28 — fix confirmed in its `model.py` snapshot) then trained **far better, val 4.90** (vs the starved 7.46) — proving the spectral-epsilon *was* a real factor and that the residual-free model *can* optimize. **But it still diverged:** as the warmup LR climbed past ~0.017 toward the 0.0225 peak, the loss blew up (train/val → 60–370 by step ~30k). So removing the residual doesn't stop the model from learning — it removes the gradient highway and makes the model **intolerant of the residual-tuned LR**. **Verdict: the residual is load-bearing**, and a residual-free model is trainable only at a much lower LR. A low-LR re-run (expected to give the graceful-degradation number, and a proper LR-sensitivity-vs-depth study) is **deferred as a future-investigation direction** — there's clearly something deeper about residual removal worth a dedicated look — but the practical verdict (keep the residual) is settled, and all headline runs keep it on.

Caveat carried from the [final regularization sweep](#final-regularization-sweep): regularization needs likely grow with depth, so a depth winner here gets its dropout/WD re-checked before any headline claim.

**Full-capacity probe (L=4, last row):** a cheap 1ep preview of capacity-at-depth before the expensive [L=5/5ep ceiling arm](#more-epochs) — restores all the memory the declared baseline dropped (PKM@16384 + FwPKM@16384 + untied). Paired with the **lean L=4** row it isolates capacity at fixed depth: if full-L4 ≈ lean-L4, the memory-redundancy finding holds at depth too (and the ceiling arm is unlikely to surprise); if full-L4 pulls ahead, capacity-at-depth is real and the ceiling arm is well-motivated. (1ep/WT103 data-starved caveat still applies — the 5ep arm is the data-richer re-test.) **Result:** full-L4 **1.0908** vs lean-L4 **1.0890** — full is **0.0018 worse** (above noise) at +407M params / +8 GB VRAM, so the memory-redundancy finding **holds at depth**; L=5 runs **no-memory**. Val *diverges* (full 3.4010 < lean 3.4032 — ranked by BPB, lean wins; the val-vs-BPB non-correlation again). The 5ep full-capacity ceiling arm stays the fair data-richer re-test, but on this evidence it is unlikely to overturn the verdict.

**Capacity arms:** the sweep above runs at the reduced base for cost; depths that survive re-run with the **T5-winning capacity form** (per the [capacity-restoration table](#t5-baseline)) before graduating — giving the depth × capacity corner the B200 arms need, with A5000 VRAM/runtime recorded as scaling reference points.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### More Epochs

The 5-epoch confirmation arms for the depth sweep — **5090, sequential** (per the cost analysis: ~$0.99/hr, identical recipe so directly comparable; B200 reserved for the C scale-up). Layers 1-4 at 5 epochs on the lean (no PKM/FwPKM) T5 recipe, **gated on [More Layers](#more-layers)**: only depths that don't regress at 1ep run here. Plus a **max-layers arm** (last row).

| Layers | Epochs | Capacity | Params | BPB sliding | Best val | Delta vs L=1 | Train VRAM | VM | Run log |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | lean | 455.55M | queued | queued | (ref) | queued | 5090 | queued |
| 2 | 5 | lean | 690.66M | queued | queued | queued | queued | 5090 | queued |
| 3 | 5 | lean | 925.78M | queued | queued | queued | queued | 5090 | queued |
| 4 | 5 | lean | 1160.89M | queued | queued | queued | queued | 5090 | queued |
| **Max from More Layers section** | 5 | lean | queued | queued | queued | queued | queued | 5090 | queued |

**These are the headline-candidate runs.** The current production headline (L=2, 5ep, 3-seed best 1.0140) predates every win on the T4 line — the FWHT deletion (−0.0024), crawl-K widening (−0.0125 at K=17 and counting), and the regularization verdicts — so the winning cell here is expected to set the new headline for the Results section, with PG-19 following on the same winner ([Longer PG-19 Training](#longer-pg-19-training)). **Capacity:** settled lean — the [L=4 lean-vs-full probe](#more-layers) showed full capacity (PKM/FwPKM/untied) *hurt* at depth, so these run the no-memory recipe; beating the old headline at **substantially fewer parameters** is itself a headline result. Headline claims additionally require the 3-seed protocol.

**Data-starvation caveat.** No full-capacity arm runs here — the [L=4 lean-vs-full probe](#more-layers) already showed memory *hurts* at depth (full-L4 1.0908 vs lean 1.0890). The deeper reason it can't pay on WT103: at 1 epoch the model sees **~0.2 tokens/param, ~100× under Chinchilla-optimal** (~20 tok/param), so extra capacity can't show value and regularization is inert (more dropout/WD slightly *hurt* at 1ep — the underfitting signature, not overfitting). 5 epochs eases this but WT103 stays data-starved (even the old headline trained at ~0.6 tok/param), so the capacity question is most fairly settled on the **scale-up datasets** (PG-19 ~2.5 tok/param, the multi-dataset mix higher), not here.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### More Width (C)

Motivated by the [More Layers](#more-layers) result (depth pays cleanly and *non-diminishingly* through L=4 on the new recipe — overturning the old "little past L=2") and the open question: **is C (width) or layers (depth) the more parameter-efficient expansion axis?** The intuition "C always wins" comes from pre-new-recipe findings (C=1024/4L beat C=512/20L; the old 30L regression) — but those predate the learned residual that *made depth pay*, so they may not transfer.

**Structural asymmetry:** depth scales params *linearly* (~+235M/layer, ~−0.0061 BPB/layer non-diminishing through L=4); width scales *quadratically* (mixer + MLP ~C²) and lumpily (`Cp = next_pow2(C)`, so only powers of two — C ∈ {2048, 4096, 8192, 16384} — give clean Cp=C points). The C=2048 column already exists (the [More Layers](#more-layers) sweep).

**Scaling matrix.** Each C at three points: **L=1/1ep** (cheap width-response anchor), **max-layers/1ep** (width at the depth optimum), **max-layers/5ep** (headline-scale). "max" = the [More Layers](#more-layers) depth winner; all rows use the **no-memory** setting (the L=4 lean-vs-full probe winner). C=8192/16384 are opened for the [B200 scale-up](#scaled-up-model-b200) when budget permits.

| C | Layers | Epochs | Hardware | Params | BPB sliding | Best val | Delta vs C=2048 | Train VRAM | Run log |
|---|---|---|---|---|---|---|---|---|---|
| 4096 (lr 0.0225) | 1 | 1 | 6000 | 1615.73M | diverged (NaN @lr~0.016) | 4.30→NaN | — | — | [link](logs/wikitext-103_2026-06-17_10-07-16/log.txt) |
| 4096 (lr 0.014) | 1 | 1 | 6000 | 1615.73M | 1.0963 | 3.4142 | −0.0110 | 31,063 MiB | [link](logs/wikitext-103_2026-06-17_12-01-47/log.txt) |
| 4096 (lr 0.015) | 1 | 1 | 6000 | 1615.73M | queued | queued | queued | queued | queued |
| 4096 (E=10, lr 0.015) | 1 | 1 | 6000 | ~1280.2M | queued | queued | queued | queued | queued |
| 4096 (E=5, lr 0.015) | 1 | 1 | 6000 | ~1112.4M | queued | queued | queued | queued | queued |
| 4096 | max | 1 | B200 | queued | queued | queued | queued | queued | queued |
| 4096 | max | 5 | B200 | queued | queued | queued | queued | queued | queued |
| 8192 | 1 | 1 | B200 | open | open | open | open | open | open |
| 8192 | max | 1 | B200 | open | open | open | open | open | open |
| 8192 | max | 5 | B200 | open | open | open | open | open | open |
| 16384 | 1 | 1 | B200 | open | open | open | open | open | open |
| 16384 | max | 1 | B200 | open | open | open | open | open | open |
| 16384 | max | 5 | B200 | open | open | open | open | open | open |

**C=4096 LR sweep (L=1/1ep, on the 6000).** Tunes the LR for the wider model. **Results:** lr=0.0225 **diverged** (NaN @ lr≈0.016 — cliff **~0.0155**, clean at 0.0154 / spiked at 0.0157); **lr=0.014 = 1.0963** (−0.0110 vs C=2048) and *converged* (train loss flat at the tail). So the sweep goes **higher, not lower** — a lower LR under-converges in the 1ep budget (0.01125 dropped). Active point: **0.015**, the highest LR demonstrably below the cliff — the √width value 0.0159 sits *on* it and would NaN, and the ~6% gain isn't worth the gamble. `min_lr` tracks at lr/50. Expect a **small** 1ep gain at most (0.014 is likely near the data-limited floor); the durable win is a **transferable LR** for C=8192 (scaled the same way) and the data-rich runs.

**MLP capacity vs. width (E=10 / E=5 probe).** The MLP is `2·E·C²` params — *quadratic* in C — so at the fixed E=20 default, C=4096 already carries a **671M MLP, 4× C=2048's 168M**. Two results suggest that overshoots the 1ep data-supported optimum: MLP over-provisioning worsened BPB *monotonically* at C=1024 ([Less Width](#less-width) iso-param E-sweep, 20→171→724), and the winning ~455M config (C=2048/E=20) beat C=1024/E=171 *while carrying the smaller MLP* (168M vs 358M). Conjecture: **a wider C wants a leaner expansion** — the MLP's useful capacity is set by the data budget, which a larger C overshoots at fixed E. Probe: hold C=4096, drop E to **10** (336M MLP, ~1280M total) and **5** (168M MLP — *exactly* C=2048/E=20's MLP, the controlled "scale C at fixed MLP" point, ~1112M total); prediction is BPB holds-or-improves at **335–503M fewer params**. Only one expansion sweep exists (C=1024), so whether the MLP optimum is *absolute* (data-fixed) or *scales with C* is under-determined — this probe decides it. **Optimization-parity caveat (bears on the width-vs-depth verdict):** the C=4096 line is **not yet as well-tuned** as C=2048 — its LR was only just calibrated (0.014→0.015) and its MLP is un-swept — so the current depth-beats-width result (C=2048/L=5 = **1.0831** at −220M params vs C=4096/L=1 = **1.0963**) is **not yet at optimization parity**; a properly-tuned wide model (best-E + recalibrated LR) may close part of the gap. The decisive follow-up is an **L=5 C=4096 run at the best-found expansion**, compared to C=2048/L=5 with both lines similarly optimized.

**Delta vs C=2048** compares each cell to the matching (same L, same epochs) point in the C=2048 [More Layers](#more-layers) / [More Epochs](#more-epochs) sweeps — the width payoff at fixed depth and budget. The **max-layers/5ep** rows are **provisional headline candidates** — established with the *current* (not-yet-final) regularization, so if the [final regularization sweep](#final-regularization-sweep) revises the recipe, they are re-run. They are therefore the **pre-final-regularization** numbers, retained as the headline fallback if the current dropout/WD settings are not the ones shipped. The **iso-param depth-vs-width** efficiency question (which axis is more BPB-per-param at a matched budget) falls out of the L=1 rows vs the existing depth sweep, accounting for the Cp lumpiness.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Less Width (smaller C)

The flip side of [More Width](#more-width-c), motivated by the **compute-allocation** question its first result raised: C=2048→4096 buys only **−0.0110 BPB for 3.5× the params**, while depth buys **−0.006 BPB/layer for a linear +235M**. Width looks like a *poor* use of compute next to depth — so if a **leaner C** stays ~equivalent to C=2048, the saved compute (smaller width trains much faster, far less VRAM) is better spent going **deeper**. This section probes **smaller C** to find the **maximally effective width**: the smallest C that doesn't materially lose vs C=2048, as the base for depth scaling.

The LR scales the *opposite* way from More Width — smaller C tolerates and wants a **higher** LR. C=1024 starts at **lr 0.04** (≈ C=2048's 0.0225 scaled up ~1/width; if it NaNs, the small-C cliff is lower than expected → drop it).

| C | Layers | Epochs | Hardware | Params | BPB sliding | Best val | Delta vs C=2048 | Train VRAM | Run log |
|---|---|---|---|---|---|---|---|---|---|
| 1024 (lr 0.04) | 1 | 1 | 6000 | 139.69M | 1.1378 | 3.5423 | +0.0305 | 3,423 MiB | [link](logs/wikitext-103_2026-06-17_18-16-32/log.txt) |
| 1024 (lr 0.05) | 1 | 1 | 6000 | 139.69M | 1.1368 | 3.5302 | +0.0295 | 3,423 MiB | [link](logs/wikitext-103_2026-06-17_19-32-01/log.txt) |
| 512 | 1 | 1 | — | — | not pursued | — | — | — | — |

**Result.** C=1024 = **1.1378** — **+0.0305 *worse* than C=2048** (1.1073), ~30× the noise floor, so a quarter-width model is **not** equivalent; it loses real capacity. But the *shape* is the payoff — the width curve is **steeply convex**: C=1024→2048 buys **−0.0305** (a lot), C=2048→4096 only **−0.0110** (little). So **C=2048 sits right at the width knee** — the maximally effective width. Per added parameter it's even starker:

| step | ΔBPB | Δparams | BPB per M-param |
|---|---|---|---|
| width C=1024→2048 | −0.0305 | +316M | **−9.7e-5** (best) |
| depth L=1→L=2 | −0.0059 | +235M | −2.5e-5 |
| width C=2048→4096 | −0.0110 | +1160M | −0.95e-5 (worst) |

**Verdict:** don't shrink below 2048 (you *crater* — it's the most valuable single step in the table) and don't widen above it (you *stall*); from 2048, **depth is ~2.6× more param-efficient than more width**. So the compute-optimal path is **depth at C=2048** — exactly the current plan. (C=512 not pursued — the curve only steepens below 1024. LR was fine: 0.04 converged cleanly, no NaN, confirming the ~1/width ceiling estimate. 1ep/WT103 data-starvation caveat still applies, but C=1024 is *capacity*-limited, not data-limited, so its drop is a real width effect.)

**Iso-param — depth vs MLP-width vs model-width (lr 0.05).** The width-knee above is *fixed-depth*; the dual is **fixed *params*, traded three ways.** Hold C=1024 and reach the C=2048/L=1 (455M) and C=4096/L=1 (1616M) param counts by **stacking layers** (depth) *or* **fattening the MLP** (`mlp_expansion`), then compare both to the **model-width** reference (the actual C=2048/4096 runs). All at lr 0.05 (the residual stream stays C=1024, so the ~0.062 cliff holds even for the fat-MLP runs).

| iso target | depth (C=1024) | MLP-width (C=1024) | model-width (ref) |
|---|---|---|---|
| **~455M** | L=6 (~433M) → queued | L=1, E=171 (~456M) → queued | C=2048/L=1 = **1.1073** |
| **~1616M** | L=26 (~1609M) → queued | L=1, E=724 (~1615M) → queued | C=4096/L=1 = **1.0963** |

**Reading.** Three ways to spend the same parameters; the *ranking* is the prize. If **depth wins** (L=6/L=26 beat both other columns), narrow-deep is the compute-optimal shape — *and* the cheapest to train (C=1024 layers ~4× cheaper). If **MLP-width wins**, capacity wants the FFN, not depth or model-width — a cheap way to scale a shallow model. If **model-width (C) wins**, the 2048 width genuinely buys something the others can't replicate. The **~1616M row is the high-value one** — if depth keeps paying partway to L=26, narrow-very-deep could *clear* C=4096/L=1 by a wide margin and rewrite the scaling plan. (lr 0.05 throughout; the deep L=26 and fat E=724 runs are the NaN watches — drop the LR if either spikes. The **C=1024/L=1 @ 0.05** row above also confirms 0.04 wasn't under-tuned — expected ≈1.1378.)

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Cross-Layer Skip Connections

Richer cross-layer information flow, motivated by the [learned-residual depth result](#more-layers): if the residual stream is the *memory bus* that makes depth pay (the L=3 ± residual control measures this), then enriching that bus is the obvious next lever. Both designs below are **additive and init-to-identity** — they reproduce the plain sequential residual stream exactly at initialization and learn away from it only if it helps, so they are strict, safe generalizations under the [structure-factoring](#structure-factoring) design rule (cross-layer flow added in parallel, not an in-series re-encoding).

**1. Dense multi-hop skips (layer-axis dense recurrence).** Today the residual stream is *accumulative-sequential*: layer *t* reads the running sum of all prior sublayer outputs, but cannot weight layer 1's contribution differently from layer *t−1*'s. Dense skips make each layer's input a **learned lower-triangular weighted combination of all prior layer outputs** — `input_t = Σ_{j≤t} A[t,j] · x_j`. Implementation template already exists: the [dense-recurrence `recur_dense_A` matrix](#done-dense-mixer-recurrence) lifted from the mixer-step axis to the layer axis, with the same identity init (`A[t,t]=1`, else 0 → byte-identical to the plain stream). Cheap; effectively a config-gated generalization. Only meaningful at L≥3 (at L=2 it reduces to the existing residual).

**2. Wavelet U-Net skips.** Distinct from #1: wavelet-native, and operating on the *detail coefficients* rather than the residual stream `x`. Routes fine-scale (high-frequency) detail from an **early** layer's decomposition into a **later** layer's reconstruction — the encoder→decoder symmetry of a U-Net, which the decompose→reconstruct structure already mirrors *within* a block. Hypothesis: fine detail washes out through depth as coarse/semantic structure dominates; a detail skip preserves it. More plumbing than #1 (scale alignment, injection point, which detail levels to carry), init at skip-weight 0 → identity.

**3. Stacking.** #1 and #2 are orthogonal (generic x-mixing vs detail-specific routing) and can run jointly; the combined arm tests whether they compound.

**Timing — gated promotion (recommended: decide on the residual result, lean right-after-layers).** These are net-new implementations, so committing them *before* the headline is speculative scope. The disciplined trigger is the [More Layers](#more-layers) ± residual outcome: (a) if depth is **bottlenecked** even with the learned residual (L≥3 ≈ L=2), cross-layer skips become the mechanism that might unlock depth *before* the B200 spend — promote to right-after-layers; (b) if the residual is **strongly load-bearing** (L=3-on ≫ L=3-off), the memory-bus hypothesis is confirmed and richer buses are well-motivated — also promote; (c) if depth already pays cleanly on the plain stream, the headline proceeds as-is and these become post-release refinements toward the *next* headline. Cost-ascending within the branch: dense skips (#1) first, U-Net (#2) second, stacking (#3) last. Each is gated on its predecessor clearing the noise floor on the L=3 recipe.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Longer PG-19 Training

The PG-19 run above was trained for a single epoch using the WikiText-optimized config. Published baselines for other models on the same dataset were likely trained for many more epochs or with much more effective compute. 

Once it is possible, the first post-release goal will be to train on PG-19 for 2 epochs, and loss permitting, 5 epochs, in order to better gauge language modeling on a large dataset at the current parameter size.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Dataset Comparisons

The best WaveletLM config trained on Pile-ArXiv, BookCorpusOpen, OpenWebText, and other datasets to gauge their performance.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Model Comparisons

Side-by-side benchmarks against Hyena, Transformer, Mamba, RWKV, and other modern architectures on WikiText-103 at matched compute and fully optimized.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Generation Decode Speedup (compile / CUDA graphs)

The largest single-stream **inference** win available, and at **zero quality cost** — distinct from the [Step-Time Speedups](#step-time-speedups) above, which target *training*. Measured decode is ~27 tok/s single-stream, ~**30× below the bandwidth roofline**: the gap is per-token *overhead* (eager Python decode loop, kernel-launch dispatch — `generate.py` applies no `torch.compile`), not compute. The decode forward is **CUDA-graph-friendly**: after the context fills, every step runs a static `[1, context_len]` forward (no KV cache — attention-free, so it recomputes the window at a fixed shape). Tiered: (1) `torch.compile(mode="reduce-overhead")` on the decode forward (auto-CUDA-graphs, near-zero effort, likely most of the win); (2) hand-rolled graph capture with a fixed input buffer + sampling kept in eager; (3) *separately* — incremental/stateful decode (a KV-cache-equivalent caching causal wavelet/crawl/bypass state) to kill the full-window recompute per token, which is the win that matters for long-context generation. Zero BPB risk (same computation — validate with a compiled-vs-eager logit match). Full design, caveats (AMP+graphs, fixed buffers, prefill handling), and measurement protocol in [plans/generation_decode_speedup.md](plans/generation_decode_speedup.md).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Bit-Packed PTQ Kernels

The [current PTQ path](runs.md#ptq-sweep-summary) dequantizes int8 weights to fp16 inside `forward()` and runs a standard fp16 matmul, which pays the dequant cost every step with no bandwidth win - hence the 12% generation slowdown and the fact that sub-8-bit variants compress identically to 8-bit on disk. 

Swapping `QuantizedLinear` / `QuantizedEmbedding` for fused packed-weight kernels (Marlin W8A16 / W4A16, CUTLASS `i8gemm`, bitsandbytes, Triton for the embedding lookup) fixes both: storage scales with bit-width, and each matmul reads half or a quarter as many bytes. Expected generation at batch=1 (fp16 baseline 28.8 tok/s) is **~1.4–1.6× faster** for fused uniform 8-bit and **~1.8–2.2× faster** for fused mixed 8/4/2, with BPB unchanged. See [runs.md](runs.md#post-release-bit-packed-ptq-kernels) for the full plan.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Multi-Transform Parallelization

> **Status (2026-06-10): on hold pending the semantic-embedding transform test — likely reduces to [multinodal](#multinodal-mode-product-of-experts) otherwise.** The [Mixer Transform Ablation](#done-mixer-transform-ablation) undercut this section's premise *for the learned-embedding configuration*: the gate is near-basis-indifferent (full spread across identity/butterfly/DHT/FWHT just 0.0037, with **identity beating every fixed basis** and the learned butterfly staying at identity). If the bases are redundant perspectives, N parallel transform paths add capacity, not perspectives — which is the multinodal section's territory, done more directly there. The one condition that revives this section: the [semantic embedding](#semantic-embedding--interpretability-work)'s transform-reintroduction hypothesis. With a *frozen* embedding, upstream basis-adaptation is reduced and transforms may become genuinely non-redundant — and the [interpretability compound](#combined-multi-transform--semantic-embedding-interpretability-compound) below pairs multi-transform with exactly that frozen embedding. Decision point: the semantic-embedding ± transform test. If transforms stay irrelevant even there, this section folds into multinodal and the compound proceeds with identity/butterfly only.

A WaveletLM-native architecture worth exploring: the wavelet decomposition and reconstruction stay shared across nodes, but the FWHT slot in each per-scale mixer is replaced by N parallel orthogonal-transform paths (FWHT, DHT, DCT-II/III, learned butterfly orthogonals). Each node decomposes the wavelet coefficients through a different "prism" in terms of its own orthogonal basis, then a per-node mixer learns basis-specific gated interactions. Finally, node-specific inverse transforms bring outputs back to the shared wavelet coefficient space for recombination. The result is multi-spectral mixing that potentially captures structure that no single basis would, with shared scaffolding keeping per-step compute increase modest (~5-15% for N=4 since the mixer slot is small relative to MLP). This is architecturally distinct from the existing `multinodal_enabled` mode (which ensembles full-cell copies at the LM head) — the multi-transform split happens *inside* a single model.

<p align="center">
  <img src="assets/waveletlm-multi-transform.svg" alt="Multi-transform parallelization architecture" width="85%"/>
</p>

**Rationale (conjectural):** If multi-transform parallelization improves results, then the most plausible mechanism is that each orthogonal basis represents the channel-axis features in a different coordinate system simultaneously. A Walsh basis groups features by binary-symmetry pattern, a cosine basis groups them by smoothness, and a learned-orthogonal basis groups them by whatever residual structure gradient descent discovers. The same input is losslessly rotated through all bases in parallel, and the combiner weights them per-scale based on which "perspective" matters most for the signal. 

Standard transformer attention has no direct analog because (Q, K, V) projections conflate "the lens you use" with "the weights you compute" into a single learned operation. **With a semantic embedding in particular (using plain-language, human-readable feature dimensions), this may make interpretability more tractable and efficient:** a per-node, per-token-pair similarity score in the rotated basis answers "what does node K think these two tokens have in common?", making it possible to trace why two tokens are close or far depending on the conceptual lens/transform applied. 

The wavelet decomposition continues to handle sequence-axis multi-scale structure, and the multi-basis nodes add feature-axis multi-perspective structure, factorizing the two cleanly. We don't yet know whether this is the actual mechanism if it increases performance, but if it does, testing this hypothesis directly becomes the natural follow-up.

**Normalization note:** the FWHT is an isometry (orthonormal, norm-preserving), so homogeneous multi-FWHT nodes naturally produce comparable output magnitudes and a single `wavelet_recon_norm` after the Combine step is sufficient. For heterogeneous nodes (learned butterfly orthogonals, DCT, etc.), each transform has its own equilibrium magnitude. The diagram therefore includes a **per-node output LayerNorm** (placed at the top of each node, before recombination) to equalize contributions so no single transform basis dominates the Combine step by amplitude alone. This per-node LN is a planned normalization for the heterogeneous case; it is not needed for the current homogeneous-FWHT implementation and will be added when multi-transform is implemented.

**Learning-rate note:** the per-node output LN above equalizes *forward magnitudes*, but not *optimizer dynamics* — each orthogonal basis conditions the loss differently, so the optimal learning rate shifts per transform even at matched scale. Two consequences. (1) **Single-transform ablation** (the §10 prerequisite): re-probe LR per candidate around the FWHT optimum (a cheap 2–3-point sweep, not a full one) rather than forcing every transform to FWHT's tuned LR — otherwise a transform that merely needs a different LR is penalized as if its basis were worse. Orthonormalize the non-FWHT candidates first so the probe starts from a scale-matched baseline; the default FWHT reference is unchanged. (2) **Combined multi-transform model:** mixing bases changes the aggregate loss landscape, so the *whole model* enters a fresh LR regime and must be re-tuned as a unit — not inherited from any single-transform run. If one global LR can't satisfy all nodes simultaneously, per-transform optimizer **param-groups** are the fallback (orthonormalization fixes scale but not conditioning, so some residual per-basis LR sensitivity is expected). The default homogeneous-FWHT path is unaffected in both cases.

See [plans/multi_transform_parallelization.md](plans/multi_transform_parallelization.md) for the full design, the four-node reference lineup, and the prerequisite ablation (per-scale mixer transform ablation in [other_post_release_plans.md §10](plans/other_post_release_plans.md#10-per-scale-mixer-transform-ablation)).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Semantic Embedding & Interpretability Work

An optional replacement for the learned token embedding is a **semantic embedding**, where each dimension is a plain-language feature (e.g. "is this token a noun?", "is this token associated with anger?", "corpus frequency in deceptive contexts") and each token or n-gram is a vector of values across those dimensions. 

WaveletLM is structurally well-suited for this: the spectral mixer can operate directly on vectorized human-readable features, and multi-scale decomposition lets the same concept be processed at different temporal granularities. The expected tradeoff is improved interpretability at a small performance cost, potentially recovered or even improved via n-gram tokens and careful feature selection for the dimensions. 

**Transform-reintroduction hypothesis (from the [Mixer Transform Ablation](#done-mixer-transform-ablation)).** With the *learned* embedding, the spectral transform proved unnecessary (identity ≥ FWHT) — plausibly because learned upstream components absorb the basis choice: anything that can emit features in whatever coordinates the gate prefers makes a fixed mid-network rotation redundant gauge. Two candidate absorbers, with different predictions for the semantic embedding:
- *If the learned **embedding** is the absorber*: freezing it (semantic embedding) removes the gauge freedom, and a transform (FWHT, or the learned butterfly — within noise of best, simpler, and itself interpretable as a learned orthogonal basis) may become a win again. Historical support: earlier EXARCH semantic embeddings underperformed the learned embedding for unexplained reasons — a missing basis-adaptation mechanism is a candidate explanation, and reintroducing a (learned) transform is a candidate fix.
- *If the learned **lifting nets** are the absorber* (they sit immediately upstream of the mixer slot and mix channels freely per level): the transform stays unnecessary even with a frozen embedding, and the historical semantic gap needs a different explanation.

The discriminating test is cheap and should be part of the semantic-embedding bring-up: semantic embedding ± transform (identity vs fwht vs learned_butterfly) at 1ep. Whichever way it lands, it pins down *where* the architecture's basis-adaptation lives — itself an interpretability result.

**Two new candidate designs (2026-06-11; full recipes in the plan doc):**
- **Relational/positional construction** — embedding coordinates built from *dyadic-offset-bucketed PMI* (co-occurrence statistics at distances 1, 2–3, 4–7, …, mirroring the wavelet's scale structure), factorized V×C. Motivated by the Yoneda view (meaning = the totality of a token's relations) and by WaveletLM being a pure position-mixing machine: these are exactly the statistics the architecture natively consumes, so this construction should pay the smallest quality-for-naming cost. Pure corpus counting — no LLM labeling cost. (A simpler per-block absolute-position variant was considered and rejected: stationarity of language under arbitrary block cuts washes the statistic out, and position is an occurrence-level property that belongs at runtime, not in the type-level table.)
- **Runtime positional channel + frozen-tied head** — frozen semantic table **concatenated** (not added/convolved — those pollute the named basis) with a runtime positional channel at the input; output head tied to the *position-free* frozen table, forcing every forward pass to end in the pure named semantic frame. Gives feature-level output attribution for free (per-dimension logit contributions). Includes a mandatory ±PE ablation arm — the wavelet/crawl machinery already encodes relative position structurally, and "the PE channel is redundant" would itself be a mechanistic finding.

See [plans/reincorporate_large_semantic_embedding.md](plans/reincorporate_large_semantic_embedding.md) for the full design, including open questions on coefficient assignment methods: one-hot/binary, LLM-scored, human-rated, or corpus-derived.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Combined Multi-Transform + Semantic Embedding (Interpretability Compound)

**Standing commitment regardless of intermediate results:** once both multi-transform parallelization and the semantic embedding are independently validated, combine them. The combined configuration is the unique regime in which input dimensions are human-readable, each transform node represents those features in a distinct mathematically-grounded coordinate system (a different orthogonal basis), every transform is invertible, and sequence-axis (wavelet) and feature-axis (multi-transform) structures factorize cleanly. Even if multi-transform is marginally suboptimal vs single-transform variants (mathematically unlikely, since multi-transform strictly contains the single-transform case as N=1, so that the combiner gate would simply prefer the first transform in a multi-transform situation), the combined configuration uniquely enables per-node, per-token-pair similarity scores in named feature coordinates and direct probing of "what does node K think these tokens have in common?" This combined configuration's value is qualitatively different from either component alone, and is not to be deprioritized in favor of incremental BPB wins on simpler variants.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Adaptive Decompose Bypass

Replacing the parameter-free cumulative running mean with a data-dependent EMA (`decompose_bypass_ema`) gained -0.30 nats at 1 epoch, but regressed at 5 epochs (BPB 1.0226 vs 1.0201). The inversion likely due to short-horizon forgetting and learned gate overfitting. Post-release plan: develop freeze-gate/bias correction probes and alternative formulations with a selective SSM bypass as fallback. See [plans/ema_post_release.md](plans/ema_post_release.md).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Prime-Power Wavelet Filterbank

> **Status (2026-06-18): proposed, pre-research.** No code yet. To be undertaken only after thorough study of the M-band / à-trous / rational-dilation filterbank literature listed in the plan doc.

The decomposition currently uses dyadic (radix-2) dilations only — 1, 2, 4, 8, …. The motivating concern is **skip-bigrams** `a … b` (b = current token, `…` = a gap ≥ 1) whose gap distance is not a power of 2 (e.g. a dependency at gap 3, 5, or 6), and whether such a dependency is captured by any dyadic scale. The proposal builds parallel **undecimated à-trous filterbanks at prime-power radices** (2, 3, 5, 7, and if lightweight 11, 13) and feeds them into the per-scale mixer in a weighted-sum fashion alongside the dyadic scales. Because the decomposition is undecimated (every scale stays at full length `T`), the prime-radix banks are just dilation-`m` filtered copies that concatenate on the scale axis with no resampling.

**The premise, sharpened.** Three corrections turn this from a confounded capacity bump into an interpretable test:
- *Reachability is not the gap.* The undecimated lowpass `approx` at position `b` already integrates a causal window containing `a` regardless of whether `b−a` is a power of 2, and dyadic dilations **compose to any integer lag** (binary representation) given enough depth. Primes add a more **localized, directly-available** basis function at those lags — a sample-efficiency / inductive-bias benefit, not new coverage.
- *Arbitrary (content-determined) gaps are out of scope for any fixed dilation set.* "Any number of tokens ≥ 1" with a variable gap is an induction pattern — a job for cumulative/decaying memory or content matching, not a finite set of fixed lags. The matched lever already exists: `decompose_bypass_ssm`, a multi-pole EMA whose learned timescales cover a *continuum* of effective reaches. Raising its pole count is run as the **control arm**.
- *Falsifiable prediction:* any benefit should **shrink with depth** (a single shallow layer can only access the lags its filters provide; deep stacks compose the rest). WaveletLM runs at L=1–5 — exactly where it would show — so BPB is logged across depth to separate a real inductive-bias effect from a capacity artifact.

**Two fusion options, tested separately.** **(A)** concat on the scale axis — the existing `cross_scale_gating` `(S,S)` routing matrix *is* the weighted sum, blending radix-2 and prime-radix coefficients during mixing (max cross-basis interaction; conjectured better). **(B)** two parallel decompose→mix→reconstruct branches blended at the end, `out = Σ_m w_m ⊙ recon_m` with a learned gate (late fusion, cheaper routing). A-helps/B-doesn't isolates the routing interaction as the active ingredient; neither moving the maximal {2,3,5,7,11,13} set kills the hypothesis cheaply — a real possibility given the [Mixer Transform Ablation](#done-mixer-transform-ablation)'s finding that the model is near basis-indifferent.

**Parameter confound.** A full à-trous pyramid per radix with its own `Cp×Cp` mixer would add ~+75M params/layer (`S_total ≈ 28` vs ~10), making any gain unattributable. Mitigations: **single-dilation lag filters** (one filter per prime at dilation¹ — lags 3, 5, 7, 11, 13 — directly targeting the skip-bigrams, not full pyramids) and a **shared mixer** across the new scales, with all comparisons **capacity-matched** against a dyadic-only run at equal `S_total`/params.

**Normalization note.** Per-radix normalization is required *before* recombination or both A and B NaN (the risk grows with `S_total` summed in the identity reconstruction path). For A, the per-scale `decomp_norms` already cover this — keep `wavelet_decomp_norm` on, extend the `ModuleList`s to `S_total`, and **block-identity-init `scale_routing`** so disparate-energy radices are not violently mixed at step 0 (a `1/√S_total` recon factor and `fht_input_cap` are the backstops). For B, LayerNorm each branch's reconstruction before the weighted sum.

See [plans/prime_power_wavelets.md](plans/prime_power_wavelets.md) for the full design, the screening-first experimental arc (control arm → maximal-set kill-test → incremental build → depth-decay check), the capacity-matched comparison protocol, and the background-material reading list (à-trous, M-band, rational-dilation, tunable-Q, wavelet packets).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Multinodal Mode (Product-of-Experts)

WaveletLM supports a baseline product-of-experts mode where multiple independent full-cell copies process the input in parallel with feature bagging and logit averaging. Enable with `multinodal_enabled: true` in the config. This mode may require stability adjustments such as a lower learning rate with `stable_parametrization` enabled, and acts as an as-yet underexplored capacity/scalability lever — a capstone for pure scale-up once the rest of the architectural roadmap settles. Distinct from [Multi-Transform Parallelization](#multi-transform-parallelization) above (which parallelizes inside a single model at the FWHT slot); the PoE mode parallelizes whole models. This existing mode and broader multi-expert techniques (sparse MoE, mutual learning, weight averaging, Git Re-Basin, & ensemble distillation) are surveyed in [plans/multinodal_training_techniques.md](plans/multinodal_training_techniques.md).

**Heterogeneous p-adic cells (proposed 2026-06-11, gated on the [crawl-K sweep](#in-progress-wavelet-crawl-dilation-window-k-sweep)).** A multinodal variant where cells differ not by seed but by **wavelet dilation lattice**: one dyadic cell (1, 2, 4, …, 64), one triadic (1, 3, 9, 27, 81), optionally 5-adic (1, 5, 25), 7-adic, etc.: prime bases, because their lattices are multiplicatively independent (log p rationally independent ⇒ offsets never collide except at 1), so each cell contributes genuinely complementary timescale coverage. All cells keep the same `block_size` (the à trous lifting has no divisibility requirement; truncating blocks to fit a base's powers was considered and rejected — it *worsens* coarse-level pad-dominance, since the fraction of positions with full history at dilation D is (T−D)/T) with each base capped at dilations ≲ T/4, recombined by the existing PoE machinery on aligned positions. Rationale for why this survives where multi-transform fell: it diversifies the **time axis**, where the architecture demonstrably cares (crawl's −0.0181 is time-axis structure), not the channel axis the transform ablation showed to be gauge. Cost-ascending ladder before committing to full heterogeneous cells: (1) single trunk with a mixed dilation schedule (config-only, param-matched), (2) dual parallel lifting stacks (`multi_basis_lifting` pattern), (3) heterogeneous PoE cells. **Gate:** if the crawl-K sweep shows K=3 remains best (off-dyadic offsets carry no signal), this branch closes cheaply; if wide windows win, rung 1 is one 1ep run. Multiple recombination schemes (logit averaging, cross-cell gating, per-scale fusion) may merit comparison by the time this is reached.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Final Regularization Sweep

Do a final regularization sweep building on the results of the [Dropout](#dropout) and [Weight Decay](#weight-decay) sections above. With the increase in model layers and potentially width, higher regularization will likely be needed. Work in a coordinate descent fashion to discover the optimal hyperparameters here.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Scaled-Up Model (B200)

Conditional on the architectural research roadmap above (multi-transform parallelization, dropout sweep, semantic embedding, combined interpretability compound) producing meaningful gains, scale up the validated architecture to B200-class hardware. The 883M RTX 5090 headline run scales up naturally to:

- `C`: 2048 → 4096 
- `layers`: 2 → 4–8
- `mlp_expansion`: 20 → 50–200
- `pkm_num_keys` & `fwpkm_num_keys`: 16384 → 65536 each
- fp16 → FP8 via Blackwell tensor cores (NYI)

The goal is a 10–15B parameter (or however large it will be) configuration, trained individually on WikiText-103 and PG-19, and also on a multi-dataset mix of WikiText-103, PG-19, Pile-ArXiv, BookCorpusOpen, TinyStories, & OpenWebText. Other possibilities such as LAMBADA will also be considered post-release.

Inference would fit on a single RTX 4090 at fp16 and roughly half the VRAM with [uniform 8-bit PTQ](runs.md#ptq-sweep-summary). See [`runs.md`](runs.md#post-release-scaled-up-b200-configuration) for the pending run entry.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Scaled-Up Model with PTQ and other Infernece Strategies

Once trained, test the best model versions with PTQ to ascertain generation speeds and VRAM requirements on a variety of systems.

Insert a collection of tables here later for each dataset, configuration, GPU type, and their associated inference VRAM and generation speeds in tokens/s for public consumption.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Downstream Transfer Fine-Tuning

The first **post-training** step (everything above is *pretraining* — self-supervised next-token prediction producing a base model; BPB is the pretraining objective). Standard architecture-paper validation: show the learned representations transfer beyond perplexity. Take each **headline base model** (the new WT-103 and PG-19 versions, and the scaled-up variants) and fine-tune on a handful of standard *labeled* downstream tasks — GLUE subsets, a classification task, plus the already-planned LAMBADA — reporting transfer accuracy. Cheap (small labeled sets, 1–3 epochs), and conventional: it's the expected way to demonstrate "good representations, not just a good perplexity number," directly comparable to how transformer baselines report transfer. Uses separate labeled datasets (not the raw pretraining corpora — there are no task labels in raw WikiText/PG-19). Run **per headline model** so transfer is reported for each pretraining corpus.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Instruction-Tuning Chat Demo

The second post-training step, and a tangible **release artifact**: SFT the headline base on an open instruction set (Alpaca / Dolly / OASST, ~tens of thousands of examples, 1–3 epochs) to produce an **attention-free wavelet *assistant*** you can actually chat with — far more demoable than a perplexity table. Cheap *relative to pretraining*: ~40M instruction tokens vs. the 100M+ pretraining corpus, a few hours at the headline scale. The instruction data is separate and small (not the pretraining corpus). Note the data-starvation lesson applies in reverse: post-training pays off most on a *strong* base, so the **richest demo is on the scaled-up model**, with the WT-103 / PG-19 headline demos as valid pre-release artifacts on the smaller bases. Optional interpretability bonus: comparing the base vs. instruction-tuned model's *readable* structure (crawl logits, scale routing) is a clean "what does instruction-tuning change mechanistically?" study unique to this architecture.

**Multi-dataset note (deferred to post-release).** The ideal base is one pretrained on **as many concatenated datasets as possible at once** (WT-103 + PG-19 + Pile-ArXiv + BookCorpusOpen + OpenWebText + …) — the richest, least data-starved base, and the best foundation for both post-training steps above. That concatenated-corpus base is a **post-release** effort (it's a larger pretraining run). Pre-release, the two post-training steps are applied **separately to each of the WT-103 and PG-19 headline models** once those have their new headline versions — giving per-corpus transfer + demo now, with the unified multi-dataset base (and its stronger post-training) following after release.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Other Post-Release Plans

See [plans/other_post_release_plans.md](plans/other_post_release_plans.md) for info on each.

- Further [structure factoring](plans/structure_factoring.md) (see also the [Structure Factoring](#structure-factoring) section of this README)
- Cross-scale phase gating (coarse-modulates-fine)
- Stable parametrization: validation and finishing gaps 
- Data-dependent lifting networks (Mamba-style)
- Wavelet Packet Decomposition (WPD)
- Top-K / hard thresholding in the Hadamard domain
- Complete Muon sweep


## License

Apache License 2.0


## References

[^1]: Kiruluta. "Wavelet Logic Machines: Learning and Reasoning in the Spectral Domain Without Neural Networks." [arXiv:2507.19514](https://arxiv.org/abs/2507.19514), 2025. (classification-focused with frozen pretrained embeddings.)
[^2]: Kiruluta, Burity, and Williams. "Learnable Multi-Scale Wavelet Transformer: A Novel Alternative to Self-Attention." [arXiv:2504.08801](https://arxiv.org/abs/2504.08801), 2025.
[^3]: Kiruluta, Raju, and Burity. "Breaking Quadratic Barriers: A Non-Attention LLM for Ultra-Long Context Horizons." [arXiv:2506.01963](https://arxiv.org/abs/2506.01963), 2025. (Non-attention LLM on WikiText-103 / Enwik8 using SSM + multi-resolution convolution + RNN supervisor + retrieval — different primitives, same task as WaveletLM.)

[^4]: Dai et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." arXiv:1901.02860, 2019.
[^5]: Radford et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019.
[^6]: Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces." arXiv:2111.00396, 2021.
[^7]: Hawthorne et al. "General-purpose, long-context autoregressive modeling with Perceiver AR." arXiv:2202.07765, 2022.
[^8]: Hutchins et al. "Block-Recurrent Transformers." arXiv:2203.07852, 2022.
[^9]: Rae et al. "Compressive Transformers for Long-Range Sequence Modelling." arXiv:1911.05507, 2019. (PG-19 dataset introduction; reports both Compressive Transformer and Transformer-XL on PG-19.)

[^10]: Poli et al. "Hyena Hierarchy: Towards Larger Convolutional Language Models." arXiv:2302.10866, 2023. PG-19 result on page 20: Hyena 153M reaches 14.6 test PPL with 16k context length, 8 epochs, GPT-2 BPE tokenization.
