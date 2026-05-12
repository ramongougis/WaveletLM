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
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
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

| Model | Type | Trained on | Params | Context | PPL |
|-------|------|-----------|--------|---------|-----|
| GPT-2 XL | Transformer | WebText (40GB) | 1.5B | 1024 | 17.5[^3] |
| Transformer-XL Large* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 257M* | 1024 effective* | 18.3[^2]* |
| GPT-2 Large | Transformer | WebText (40GB) | 774M | 1024 | 19.3[^3] |
| S4* | SSM* | WikiText-103 (0.5GB)* | 130M* | 1024* | 20.9[^4]* |
| GPT-2 Medium | Transformer | WebText (40GB) | 355M | 1024 | 22.1[^3] |
| **WaveletLM** | **Wavelet mixer** | **WikiText-103 (0.5GB)†** | **883M** | **256†** | **23.8†** |
| Transformer-XL Standard* | Transformer + recurrence* | WikiText-103 (0.5GB)* | 151M* | 1024 effective* | 24.0[^2]* |
| GPT-2 | Transformer | WebText (40GB) | 124M | 1024 | 29.4[^3] |

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

1. [(Done) Single-Layer WaveletLM with Current Best Config](#done-single-layer-waveletlm-with-current-best-config)
2. [(Done) Parameter Reduction](#done-parameter-reduction)
3. [(Done) Larger Block Size](#done-larger-block-size)
4. [(Done) Per-Scale Mixer Width Contraction and Expansion](#done-per-scale-mixer-width-contraction-and-expansion)
5. [(Done) Mixer Low Rank](#done-mixer-low-rank)
6. [(Done) T1 Baseline Without Wavelet Crawl](#done-t1-baseline-without-wavelet-crawl)
7. [(Done) New Baseline T2 with 7 Levels, more Per-Scale Mixer Weights, and Wavelet Crawl](#new-baseline-t2-with-7-levels-more-per-scale-mixer-weights-and-wavelet-crawl)
8. [Optimizer Sweep (Muon → AdamW)](#optimizer-sweep-muon--adamw)
9. [(Done) Sequential Block Ordering](#done-sequential-block-ordering)
10. [(Shelved on WT-103) 2D Wavelet over (Batch, Token) with Sequential Training](#shelved-on-wt-103-2d-wavelet-over-batch-token-with-sequential-training)
11. [Bisected Block Context Extension](#bisected-block-context-extension)
12. [Adagrad Learning Rate Tuning](#adagrad-learning-rate-tuning)
13. [Wavelet Sparsity Probe & Wavelet Shrinkage](#wavelet-sparsity-probe--wavelet-shrinkage)
14. [Recurrence (Mixer Only)](#recurrence-mixer-only)
15. [Untied Wavelet Reconstruction](#untied-wavelet-reconstruction)
16. [Complex Wavelets](#complex-wavelets)
17. [Dropout](#dropout)
18. [Weight Decay](#weight-decay)
19. [Mixer Transform Ablation](#mixer-transform-ablation)
20. [Step-Time Speedups](#step-time-speedups)
21. [Longer PG-19 Training](#longer-pg-19-training)
22. [Dataset Comparisons](#dataset-comparisons)
23. [Model Comparisons](#model-comparisons)
24. [Bit-Packed PTQ Kernels](#bit-packed-ptq-kernels)
25. [Multi-Transform Parallelization](#multi-transform-parallelization)
26. [Semantic Embedding & Interpretability Work](#semantic-embedding--interpretability-work)
27. [Combined Multi-Transform + Semantic Embedding (Interpretability Compound)](#combined-multi-transform--semantic-embedding-interpretability-compound)
28. [Adaptive Decompose Bypass](#adaptive-decompose-bypass)
29. [Multinodal Mode (Product-of-Experts)](#multinodal-mode-product-of-experts)
30. [Scaled-Up Model (B200)](#scaled-up-model-b200)
31. [Other Post-Release Plans](#other-post-release-plans)

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
- minimal performance impact (−0.0013 BPB)
- 26% smaller train/val loss gap (implicit regularization via less params)

The "Test 1", aka **T1**, configuration in the table below incorporates these reductions. 

| Run | Recipe | Folder | BPB sliding | PPL sliding | Best val | Min train | Train/val gap | Params | Train time | Training VRAM |
|-----|--------|--------|-------------|-------------|----------|-----------|---------------|--------|------------|------|
| Best WT-103 run with layers=1 | Best run with layers=1 | [link](logs/wikitext-103_2026-04-30_02-20-35/log.txt) | 1.0809 | 29.28 | 3.3275 | 2.8292 | 0.498 | 586.15M | 9.74h | 11,537 MiB |
| **T1/Test 1** | Best run with layers=1 and parameter reductions | [link](logs/wikitext-103_2026-05-01_06-33-48/log.txt) | **1.0796** | **29.15** | 3.3341 | 2.9649 | **0.369** | **344.63M** | 7.69h | 6,867 MiB |
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

### Optimizer Sweep (Muon → AdamW)

**Phase 1: Muon** ([Jordan et al., 2025](https://arxiv.org/abs/2502.16982); used in DeepSeek-V4). Newton-Schulz orthogonalization bounds every update's spectral norm — structurally the same property mHC uses to scale residual depth, applied to our matrix-heavy MLP / mixer / lifting `Linear(C, C)`. Start from DeepSeek-V4's hybrid recipe (8 iterations at (3.4445, -4.7750, 2.0315) + 2 at (2, -1.5, 0.5)); embedding / LM head / RMSNorm stay on AdamW. **Phase 2: AdamW** as fallback baseline.

**Sweep table.** All runs use the T2 architecture (`levels=7`, `per_scale_mixer_widths=[1.0×4, 0.5×4]`, `wavelet_crawl=true`, `bs=256`, `MBS=8`). Muon hybrid splits params: 2D non-embedding hidden weights → Muon, biases / norms / embeddings / LM head → AdamW (per [torch.optim.Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) docs). For Adagrad the rows for momentum / NS steps / NS coefficients / adjust LR fn are not applicable.

| Optimizer | LR | Weight Decay | Momentum | Eps | NS Steps | NS Coefficients | Adjust LR Fn | Epochs | BPB sliding | PPL sliding | Best val | Train Time | Train VRAM | Inference VRAM (strategies) | Run Log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adagrad (T2 ref, 1ep) | 0.01 | 1e-6 | — | 2e-13 | — | — | — | 1 | 1.1541 | 36.7905 | 3.5881 | ~1.86h | 7,788 MiB | 3,258 MiB | [link](logs/wikitext-103_2026-05-10_03-39-43/log.txt) |
| Adagrad (T2 ref, 5ep) | 0.01 | 1e-6 | — | 2e-13 | — | — | — | 5 | 1.0485 | 26.4564 | 3.2630 | ~8.92h | 7,788 MiB | 3,238 MiB | [link](logs/wikitext-103_2026-05-10_05-33-24/log.txt) |
| Muon (defunct ※) | 0.001 | 0.1 | 0.95 | 1e-7 | 5 | (3.4445, -4.775, 2.0315) | original | 1 (cancelled at step 27,500 / 47%) | — | — | val=4.1243 vs T2 Adagrad's val=3.9342 at the same step (Δ = +0.1901) | partial (~1.62h to step 27,500, projected ~3.05h for 1ep) | 7,788 MiB | — | [link](logs/wikitext-103_2026-05-10_15-49-10/log.txt) |
| Muon (over-aggressive ✗) | 0.01 | 0.1 | 0.95 | 1e-7 | 5 | (3.4445, -4.775, 2.0315) | original | 1 (cancelled at ~step 17,537 / 30%) | — | — | best val 4.7274 at step 7,000; plateau/oscillation in 4.72–4.77 band from step ~5000; ahead of Adagrad through step ~6000 then crossed back behind — see footnote | partial | 7,788 MiB | — | [link](logs/wikitext-103_2026-05-10_17-45-55/log.txt) |
| ~~Muon (skipped — lr=0.01 already over-aggressive)~~ | ~~0.05~~ | — | — | — | — | — | — | ~~1~~ | skipped | skipped | skipped | skipped | skipped | skipped | — |
| ~~Muon (skipped)~~ | ~~0.10~~ | — | — | — | — | — | — | ~~1~~ | skipped | skipped | skipped | skipped | skipped | skipped | — |
| ~~Muon (skipped)~~ | ~~0.20~~ | — | — | — | — | — | — | ~~1~~ | skipped | skipped | skipped | skipped | skipped | skipped | — |
| **Muon (queued)** | **0.003** | 0.1 | 0.95 | 1e-7 | 5 | (3.4445, -4.775, 2.0315) | original | 1 | queued | queued | queued | queued | queued | queued | queued |
| **Muon (queued)** | **0.005** | 0.1 | 0.95 | 1e-7 | 5 | (3.4445, -4.775, 2.0315) | original | 1 | queued | queued | queued | queued | queued | queued | queued |

※ **Defunct — LR under-scaled.** With `adjust_lr_fn=None` (= "original" / Keller's scaling, the API default), `torch.optim.Muon` scales LR by `max(1, sqrt(A/B))` per matrix — for square 2048×2048 matrices that's 1.0, *no amplification*. The doc's `lr=0.001` default is calibrated for `match_rms_adamw` semantics, which would scale by ~9× for 2048×2048 and ~29× for our (2048, 20480) MLP weights. Under "original" scaling with lr=0.001, the effective LR is 10–50× below what Muon's reference implementations (Keller, DeepSeek-V4) intend. The partial run ([log](logs/wikitext-103_2026-05-10_15-49-10/log.txt)) confirms this: trains, but ~0.19 nats behind T2 (Adagrad) at matched step. The lr=0.002 / lr=0.0005 sweep variants (originally queued) are also in the under-scaled regime and were not run; subsequent runs jumped to lr ∈ {0.01, 0.05, 0.10, 0.20} to test Muon under "original" scaling at LRs in / above Keller's published 0.01–0.05 range.

✗ **Over-aggressive — too high LR for sustained training.** The lr=0.01 run ([log](logs/wikitext-103_2026-05-10_17-45-55/log.txt)) showed the opposite failure: a strong early-LR head start through ~step 6000 (ahead of Adagrad by 0.18–0.39 nats across steps 1000–6000), followed by **plateau/oscillation in a 4.72–4.77 val band from step ~5000 onward** while Adagrad continued its smooth descent past it. Best val 4.7274 at step 7,000; by step 8000 lr=0.01 had crossed back *behind* Adagrad (+0.05 nats) and showed no clear downward trend with continued training. Cancelled at ~step 17,537 (30%, just at peak LR) since further training would only deepen the oscillation. Classic "too high LR" signature: parameters bouncing around a local minimum rather than converging. The originally-queued lr=0.05 / lr=0.10 / lr=0.20 variants were guaranteed worse (higher LR, same failure mode amplified) and skipped. The Path B v2 sweep at lr=0.003 / lr=0.005 splits the band between under-scaled (0.001) and over-aggressive (0.01) to find Muon's native operating point on T2.

**Compute-justified-only criterion.** Muon's per-step cost is ~2× Adagrad's on this stack (5 NS iterations across 77 2D matrices). To justify keeping Muon over Adagrad, it must clear T2 (Adagrad)'s 1ep best val (3.5881) by enough margin that *matched-compute* favors it — i.e., reach equal best val in ≤ half the epochs, or strictly beat Adagrad's end-of-epoch numbers at matched epochs. Otherwise Adagrad stays as the production optimizer.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Done) Sequential Block Ordering

Currently, the model does random sampling: each block starts at a uniformly random corpus position with no relationship between consecutive batches.

This section tests sequential block ordering, where every token and block is visited exactly once per epoch in corpus order. We'll see if it works as a standalone feature. Otherwise, it'll be turned on only for later features which require it.

It is a prerequisite for:
- the [2D Wavelet over (Batch, Token)](#2d-wavelet-over-batch-token-with-sequential-training) 
- efficient [Bisected Block Context Extension] compression(#bisected-block-context-extension) (not strictly necessary).

Design choices:

- Stride = `block_size` (no overlap).
- Document/book boundaries are ignored. Matches the current random sampler behavior and the GPT-style "concat-and-chunk" pretraining default.
- No shuffling. The point of this experiment is to see whether maintaining corpus order matters.

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

**Findings:**

- Sequential ordering at lr=0.01 underperforms random sampling at matched epochs (Rainman 1ep best val 3.6601 vs T2 random 1ep best val 3.5881; Δ = +0.0720 best val, +0.0185 BPB).
- **Sequential + lr=0.015 (Adagrad uniform LR bump) recovers ~50% of the gap** — best val 3.6231 vs T2 random 3.5881 (Δ = +0.0350 remaining vs the +0.0720 starting gap; 51% gap recovery). BPB sliding gap closes from +0.0185 → +0.0039, within ~3× noise threshold. Hits exactly the predicted "~30–50% recovery" from the LR-bump analysis.
  - Caveat: not fully comparable yet — T2 random reference is at lr=0.01. A matched-LR comparison (T2 random at lr=0.015) is needed to know whether lr=0.015 is a sequential-specific fix or a general improvement. Worth a single follow-up run.
- **2D wavelet "internal" mode underperforms sequential baseline** (best val 3.6691 vs 3.6601; Δ = +0.0090, ~6× noise threshold). The per-sub-band scaling extracts no useful B-axis signal at WT-103 scale, plus a ~14% wall-clock cost. Internal mode shelved.
- **2D wavelet "subband" mode underperforms further** (best val 3.7199 vs 3.6601; Δ = +0.0598, ~40× noise threshold). +63M params (+16%), +30% wall-clock, +15% train VRAM — all costs with negative return. Subband mode shelved. Both 2D modes confirm that B-axis lifting carries no useful signal on WT-103, likely because Wikipedia articles are largely independent and the cross-batch temporal structure that would justify 2D decomposition isn't present. **2D wavelets may still work on PG-19** (long-form novels with multi-book dependencies) but that's a much bigger compute commitment and off the immediate roadmap. Code in `tools/two_d_wavelets.py` is preserved for future revisit; runs.sh entries are commented out.
- This feature still does not improve the model on its own, but lr=0.015 closes most of the regression and the second-epoch gain (Δ −0.1529 best val between Rainman 1ep and 2ep) confirms sequential is a viable substrate for downstream features that require it (BBCE caching, longer-context training, future PG-19 2D wavelet revisit).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### (Shelved on WT-103) 2D Wavelet over (Batch, Token) with Sequential Training

Generalize the lifting wavelet decomposition from 1D over the token axis to 2D over the joint (batch, token) axis pair. When training proceeds in document-sequential order, the batch axis carries the same multi-scale temporal structure as the token axis, and the same wavelet machinery applies to both. This requires reorganization of the current batch sampling method into a sequential batch processing method, since batches may not be IID with respect to each other. As one example, consider series of novels in PG-19 with temporal plot dependencies between books. Randomly sampling batches breaks this temporal relationship. Using 2D wavelets is a convenient way to enforce and encode temporal relationships at all levels within the model. See [plans/two_d_wavelet_sequential_training.md](plans/two_d_wavelet_sequential_training.md) for the full design.

**Status (2026-05-11):** Two architectural variants tested on T2 + sequential WT-103 at 1 epoch — `"internal"` mode (B-axis lift + per-sub-band scale + B-axis inverse; same output shape as 1D; +6% params) and `"subband"` mode (4 sub-bands per joint level exposed to per-band mixers; +16% params). Both **underperformed the sequential Rainman baseline** (3.6691 / 3.7199 vs 3.6601 best val) at 14% / 30% greater wall-clock respectively. Likely cause: Wikipedia articles are largely independent at the chunk level, so cross-batch temporal structure that would justify 2D decomposition isn't present in WT-103. **2D wavelets may still work on PG-19** (long-form novels with multi-book dependencies) where cross-batch structure is real. Code in [tools/two_d_wavelets.py](tools/two_d_wavelets.py) is preserved for future revisit; runs.sh entries commented out.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Bisected Block Context Extension

Inspired by [DeepSeek-V4 (DeepSeek-AI, 2026)](https://huggingface.co/collections/deepseek-ai/deepseek-v4). HCA-style summarized history at the input computing the mean of a certain number of continuous tokens across each channel. The number of tokens taken depends on the desired context (`block_size_compressed`), which is fully customizable, allowing for arbitrarily large context windows. 

Take the most recent `block_size/2` token slots as uncompressed input, and use the other `block_size/2` slots as compressed input, with each compressed token slot holding the per-channel means of `g = ceil((block_size_compressed − block_size/2) / (block_size/2))` consecutive corpus tokens. 

For `block_size=256`, if a 1M context window is desired, `block_size_compressed=1,000,000` → `g=3907`. If a 4M context window is desired, `block_size_compressed=4,000,000` → `g=15625`. For `block_size=512`, 1M context gives `g=1954`, and 4M context yields `g=7813`. 

The half-block-size point marks a common seam for every wavelet scale, as each has 2^k dyadic partitions. Thus the compressed and uncompressed regimes never lie in a single partition window, and so there are O(log block_size) total seam-bridging predict/update operations. 

This section will do 9 sweeps of `block_size` x `block_size_compressed` ∈ {256, 512, 2048} x {65K, 262K, 1M}, 1 epoch each, using the T2 baseline after optimizer tests. The best-performing version will then run for 5 epochs.

**Step Count Methodology.** "1 epoch" is defined as one full pass over the corpus's *supervised* positions — identical across BBCE and non-BBCE. The training schedule uses an **active stride** equal to the number of new corpus tokens consumed for supervision per micro-batch sample: `block_size` without BBCE (every position is supervised) and `block_size/2` with BBCE (only the uncompressed half is supervised; the compressed half is reused context, treated as an architectural add-on rather than additional epoch budget). Per-epoch supervised-token coverage is the full corpus in both cases. In practical step counts at MBS=8 over WikiText-103 (119.7M tokens):

| Config | Steps/epoch |
|---|---|
| bs=256 non-BBCE | 58,457 |
| bs=256 BBCE | 116,914 |
| bs=512 BBCE | 58,457 |
| bs=2048 BBCE | 14,650 |

**Comparability Note:** depending on the metric under consideration (equal wall-clock time or compute, equal block size overall, or double block size to have the same number of actual tokens seen), it may be essential to compare the performance of the model with BBCE and block size 512, as well as BBCE with block size 256; versus the model at block size 256 and without BBCE. An additional run without BBCE and with block size 512 would offer another perspective. That is, direct comparability with the baseline is difficult to gauge cleanly with this feature.

**Benchmark Methodology.** Test-set BPB follows the [HuggingFace sliding-window perplexity convention](https://huggingface.co/docs/transformers/perplexity): only non-context tokens contribute to the log-loss. For BBCE that means **only the supervised half** (the last `block_size/2` positions per window) is scored, and the compressed half is excluded as context. Stride is fixed at `block_size/2` so every test token is scored exactly once at its supervised position. This produces a single test-set BPB number that is directly comparable to the sliding-window BPB reported elsewhere in this README for non-BBCE runs, and to sliding-window perplexity numbers in the broader literature. The standard "non-overlapping" benchmark has no separate analogue for BBCE — BBCE has one natural test-set eval style (supervised-half scoring), not two.

**Sweep results (1 epoch).** Decision rule: a BBCE variant must clear the T2 baseline best val (3.5881) by more than the 0.0015-nat noise threshold to be considered a win.

| Run | Best Val | Δ vs T2 | BPB sliding | Train Time | Train VRAM | Inf VRAM |
|---|---|---|---|---|---|---|
| **T2 baseline** (bs=256, no BBCE) | 3.5881 | (ref) | 1.1541 | 1.83h | 7,788 MiB | 3,258 MiB |
| BBCE bs=256 / bc=65K | 3.5725 | −0.0156 | 1.1551 | 5.89h | 7,809 MiB | (pending) |
| BBCE bs=512 / bc=65K | 3.5922 | +0.0041 | **1.1530** | 3.88h | 8,177 MiB | (pending) |
| BBCE bs=512 / bc=131K | (pending) | — | — | — | — | — |
| BBCE bs=512 / bc=250K † | (pending) | — | — | — | — | — |
| BBCE bs=1024 / bc=65K | (pending) | — | — | — | — | — |
| BBCE bs=1024 / bc=131K | (pending) | — | — | — | — | — |
| BBCE bs=1024 / bc=250K † | (pending) | — | — | — | — | — |
| BBCE bs=2048 / bc=65K | (running) | — | — | — | — | — |
| BBCE bs=2048 / bc=131K | (pending) | — | — | — | — | — |
| BBCE bs=2048 / bc=250K † | (pending) | — | — | — | — | — |
| BBCE bs=512 / bc=65K + `compressed_grad=false` | (pending) | — | — | — | — | — |

**WT-103 test-set ceilings (honest limitations).** WikiText-103 caps the bc values we can measure cleanly:
- **Val cliff at bc=251K:** `val_data` is 251,048 tokens. When `bc ≥ 251K`, the BBCE val branch falls back to sampling from `train_data`, so "Best Val" stops measuring val-distribution loss and becomes a train-distribution proxy — not directly comparable to baseline.
- **Padding pressure gradual to bc=287K:** `test_data` is 287,644 tokens. As `bc` grows, fewer test windows have a full `bc`-token real context — the rest left-pad with zeros. At bc=131K, 54% of test windows are unpadded; at bc=250K, only 13% are. The BPB benchmark for high-bc cells therefore measures off-distribution performance (mostly-zero compressed slots, which the model never trained on). Cells marked † in the table fall in this regime — Best Val is the trustworthy metric there; BPB needs an asterisk.
- **OOM at bc=1M:** compressed-half saved activations for the backward pass exceed 32 GiB on a single 5090; either `bbce_compressed_grad: false` (drops gradient through compressed) or a streaming sum / chunked-reduction refactor would be needed to fit. Neither is implemented yet.

For bc > 256K, **PG-19** is the natural next test set: ~28M test tokens means bc=1M would be at 0.04× test_len vs WT-103's 3.5× — virtually no padding, and the long-form novel structure is exactly where BBCE's value proposition lives (cross-chapter / cross-book conditioning). Filed as the scale-up direction once the current WT-103 cells settle the bc-scaling question.

**Compensation observation (preliminary).** The `bs=256 / bc=65K` row lands at Δ +0.0010 BPB vs T2 baseline — within the 0.0015-nat noise threshold, statistically identical. That's the load-bearing surprise: BBCE supervises half as many tokens per batch as T2, yet test-set perplexity matches. The 65K of compressed context appears to exactly compensate for the halved per-batch supervision — no more, no less. The Best Val drift (Δ −0.0156, ~10× noise) suggests the compressed context is doing real work; it just doesn't show up on the test-set BPB until either `block_size` grows or, possibly, `block_size_compressed` does. The rest of the sweep distinguishes two scenarios:
- **(a) Longer bc yields strict gains beyond the compensation point**: `bs=256 / bc=1M` would beat T2 by a margin; `bs=512 / bc=1M` would beat its bc=65K counterpart. This is the desired outcome — compressed context is genuinely informative and scales.
- **(b) Per-batch supervised stride is the dominant factor**: any `bs=256` BBCE config plateaus near T2 regardless of bc, and gains only come from larger `bs`. Interesting if true (it would suggest block size and architecture are intrinsically coupled, independent of content), but a weaker claim for the feature.

**Compressed-half gradient toggle (`bbce_compressed_grad`).** When `true` (default), gradients flow back through the mean-pool into the embedding table for tokens appearing in the compressed half — every appearance contributes `(1/g) × dL/dslot` to its embedding row, so over a batch the embedding table learns to be a good mean-pool basis as well as a good direct-prediction basis. When `false`, the chunked embedding lookup runs under `torch.no_grad()` and the compressed slots are detached: the embedding table only updates from uncompressed-half appearances, and the wavelet/mixer/MLP still learn to USE compressed slots but can't push the embedding table toward better averaging behavior. The dilution-only argument that justifies `false` is unverified — at WT-103 scale (120M tokens, 1 epoch), incidental averaging quality emerging "for free" from next-token training is not guaranteed. Backward-pass speedup with `false` is expected ~1.3-1.5×. A single confirmatory A/B run is queued at the end of the sweep (bs=512 / bc=65K cell).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Adagrad Learning Rate Tuning

T2's default `lr=0.01` was inherited from earlier baselines and has not been re-tuned for the current architecture. The sequential block ordering experiments surfaced this: the sequential variant of T2 trailed T2 random by Δ best val +0.0720 at lr=0.01, but lr=0.015 (sequential) recovered ~50% of that gap (best val 3.6231 vs Rainman 3.6601). The natural follow-up question was whether lr=0.015 is a sequential-specific Adagrad fix or a general LR re-tune that helps T2 itself.

**Isolation test (T2 random + lr=0.015, complete).** Same configuration as T2 reference, only LR bumped from 0.01 → 0.015 (with `min_lr` scaled proportionally to 0.0003). Run log: [logs/wikitext-103_2026-05-11_15-26-31/log.txt](logs/wikitext-103_2026-05-11_15-26-31/log.txt).

| Metric | T2 baseline (lr=0.01) | T2 + lr=0.015 | Δ |
|---|---|---|---|
| Best val | 3.5881 | **3.5345** | **−0.0536** |
| BPB sliding | 1.1541 | **1.1362** | **−0.0179** |
| PPL sliding | 36.79 | **34.79** | **−2.00** |
| Train time | 1.83h | 1.84h | +0.6% |
| Train VRAM | 7,788 MiB | 8,065 MiB | +3.6% |
| Inference VRAM | 3,258 MiB | 3,258 MiB | (matched) |

Δ best val of **−0.0536 nats is ~36× the noise threshold**. Unambiguous win at near-zero compute cost. The mid-epoch ~0.044 nat lead held through the cosine decay tail and the final number is even larger.

**T3 candidate is now defined.** Two possible compositions depending on the BBCE sweep outcome and a possible further LR refinement (see below):
- **T3 = T2 + lr=0.015** (if BBCE shows no improvement). The LR retune alone becomes the new production stack.
- **T3 = T2 + lr=0.015 + BBCE** (if BBCE shows meaningful improvement on top of the LR retune). Both wins consolidate into one baseline.

The existing T2 baseline numbers (best val 3.5881 at 1ep, 3.2630 at 5ep) will be replaced by re-measured T3 numbers (1ep and 5ep) in the [Results](#results) section. The replacement is **deferred until after the lr=0.020 / further LR refinement sweep and the BBCE outcome are settled**, so the headline benchmark gets stacked all known improvements at once rather than incremented run-by-run.

**Why this matters strategically.** Recent architecture-level explorations (2D wavelets — both modes shelved; sequential ordering — needs LR fix to be viable; Muon — no clear win yet) have not surfaced improvements. The lr=0.015 retune is the only meaningful performance gain since the T2 baseline was set, and it costs zero extra compute.

**Follow-up: lr=0.020 sweep (queued).** Small extra step to confirm lr=0.015 is in the right neighborhood and we haven't undershot the optimum. Decision rule:
- lr=0.020 better than 0.015 by > 0.0015 → optimum is ≥ 0.020; consider another sweep at lr=0.025.
- lr=0.020 within ~0.002 of 0.015 → plateau; lock in 0.015.
- lr=0.020 worse than 0.015 → past the optimum; 0.015 wins.

Caveat: at sufficiently high LR, Adagrad's accumulator dynamics can change qualitatively (similar to what we saw with Muon at lr=0.01 — fast early descent then plateau/oscillation around step 5000). The 0.020 run is the canary.

**BBCE compatibility.** The currently-queued BBCE sweep deliberately uses lr=0.01 (T2's existing LR) for apples-to-apples comparison against the T2 baseline. If BBCE shows a winner, that winner will be re-run at the locked-in LR (0.015 or whatever the sweep settles on) as part of the T3 consolidation. Switching mid-queue would mix two variables and complicate attribution.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Wavelet Sparsity Probe & Wavelet Shrinkage

Two related ablations exploring the sparsity structure of the wavelet's detail coefficients — a property heavily exploited by classical wavelet compression (JPEG 2000) that we have not yet measured or used in our learned-wavelet pipeline.

**Sparsity probe (diagnostic, ~30 minutes).** Run a trained T2 checkpoint on a held-out batch slice; log the magnitude distribution of detail coefficients per scale (e.g., quantiles, fraction below 1% of max). If detail coefficients are heavily sparse (~80%+ near zero, as is typical for natural-signal wavelet decompositions), several optimization directions open up: sparse mixer compute (top-k mixing within each scale), low-bit detail quantization (run details at fp8/int8 while keeping approximation at fp16), sparse activation storage for long-context training, and structural intuition for [BBCE](#bisected-block-context-extension)'s compressed history. High information value per compute spent; this is the first thing to run.

**Wavelet shrinkage (training-time regularization).** Soft-threshold detail coefficients during the forward pass: `detail = sign(d) * max(|d| - λ * σ_scale, 0)` where `λ` is the shrinkage strength (start with 0.1) and `σ_scale` is the per-scale standard deviation (estimated as a running EMA or precomputed once). Forces the model to learn a sparse multi-scale representation, mirroring the noise-suppression behavior wavelet methods use in signal processing. One config flag (`wavelet_shrinkage_lambda`) and ~10 lines in the wavelet's forward. Two outcomes worth distinguishing:
- **Helps**: shrinkage acts as effective regularization; the model was using too many detail coefficients indiscriminately and dropping the smallest improves generalization.
- **Hurts**: detail coefficients are not redundant; suppressing them removes information the model was using. Tells us our learned wavelet doesn't have JPEG-style sparsity even after training, which is itself an informative finding.

Both ablations run cheaply on T2/Adagrad at 1 epoch. Combine: run the probe first, calibrate `λ` from the observed magnitude distribution (e.g., the 25th percentile per scale), then test shrinkage with that empirical-data-driven setting.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Recurrence (Mixer Only)

Due to wavelet decomposition and reconstruction being inverses of each other, and FWHT being its own inverse, one form of recurrence in WaveletLM only requires repeating the mixer operation. In other words, N steps of recurrence would look like:

`x → Decompose → FWHT → Mixer1 → Mixer2 → ... → MixerN → iFWHT → Reconstruct → x'`

This could be by either repeating the same mixer N times (most likely), or having N different mixers. The same mixer repeated N times could benefit from expansion of `per_scale_mixer_widths` per our [previous mixer width expansion results](#done-per-scale-mixer-width-contraction-and-expansion), depending on the dataset size. On the other hand, different mixers naturally adds more parameters. Training stability is dependent on the outcome of optimizer tests, degree of per-scale mixer width expansion, and the number of mixers used.

Other recurrence approaches likely exist, but this section will only test the mixer.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Untied Wavelet Reconstruction

The current implementation **ties** the wavelet reconstruct path to the decompose path: they share the same `predict_nets` and `update_nets` (perfect mathematical invertibility — decompose followed by reconstruct is exactly identity when no processing happens in between). The flag `untied_reconstruction` (already in config.json, currently `false`) would give the reconstruct path its **own** predict/update networks — same architecture, separate weights.

**Trade-off:**
- **Tied** (current): invertibility preserved. The structural identity `Reconstruct ∘ Decompose = I` is what enables [Recurrence (Mixer Only)](#recurrence-mixer-only) to express N-step recurrence as just N chained mixers (since adjacent Decompose-Reconstruct cycles cancel). Lower parameter count.
- **Untied**: reconstruct can apply learned transformations that aren't constrained to invert decomposition. More expressive — reconstruction can "fix up" the mixer's spectral output in ways the strict inverse would not allow. But the structural identity breaks, so the model's `x → Decompose → mixers → Reconstruct → x'` is no longer reducible to "mixers in a wavelet basis." Adds ~83.93M params per layer at T2 (matching the existing wavelet param count) — roughly +21% over T2.

**Mutually exclusive with Recurrence (Mixer Only).** Untied reconstruction breaks the invariant that justifies "mixer only" recurrence. If both are pursued, the recurrence design has to be reformulated — either to fold the full `Decompose → ... → Reconstruct` cycle into the recurrent loop (multiplying compute by N), or to share recurrent updates only within the spectral basis with explicit care for the non-inverse reconstruct. Cleaner to commit to one direction first: test untied reconstruction as a standalone variant against T2 baseline (1-epoch at fixed compute), then decide whether to compose it with recurrence.

**Test plan:** single-flag flip (`untied_reconstruction: true`) on T2 + 1ep + sequential? + random? — both sampling modes worth measuring since the wavelet's role differs between them. Compare BPB sliding and best val to T2 reference at matched compute.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Complex Wavelets

Replace the real-valued wavelet basis with a complex-valued one (e.g., dual-tree complex wavelet transform, or direct complex parameterization of the lifting predict/update networks). Real wavelets capture only magnitude; complex wavelets carry both magnitude and **phase**.

**Why phase might matter for language.** In signal processing, phase information is what distinguishes "an edge at this position" from "an edge at a slightly different position" — the magnitude is the same, but the phase differs. For text, the analog is positional / structural patterns that recur at different offsets: e.g., the same syntactic structure appearing 5 tokens later vs 20 tokens later. Real wavelets capture *that* such a structure is present at some scale; complex wavelets additionally capture *where within that scale's support window* it is positioned. Whether this distinction carries useful signal for next-token prediction is the empirical question.

**Cost.** Roughly 2× across the board:
- Wavelet predict/update networks need either complex-valued weights (`torch.complex64`) or real/imag interleaving (a 2C-wide real tensor representing C complex values).
- FWHT and mixer ops need to handle complex tensors, OR the real/imag interleaving lets the existing real-valued mixer operate on the 2C-wide tensor (simpler but doubles mixer compute).
- Total: roughly +25M params and +2× wavelet stage compute. The mixer stage stays roughly the same compute if real/imag interleaving is used (since mixer width is determined by the mixer's own config).

**Implementation surface.** Moderate. A new `LiftingWaveletComplex` class in `tools/complex_wavelets.py` mirroring the structure of `LiftingWavelet2D` (selectable via a `wavelet_basis: "real" | "complex"` config flag). Real/imag interleaving keeps `model.py` integration minimal. ~300-500 lines for the wavelet module plus minor model.py changes.

**Empirical question worth flagging upfront.** Most "phase matters for language" intuitions come from signal-processing analogies that may not transfer cleanly. Text isn't a sinusoidal signal; the wavelet basis we use is already learned (not fixed Haar/Daubechies). The learned real-valued predict/update networks may already implicitly capture phase-equivalent information via their shape. Test design needs to disambiguate: does complex outperform real *at matched parameter count* (so we know it's the phase, not the extra params)?

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Dropout

Re-tune the five dropout values (`dropout_lm_head`, `dropout_mlp`, `dropout_mixer`, `dropout_projection`, and `dropout_embedding`) once model parameters are reduced from above. A doubled-dropout ablation at the prior baseline gave -0.0221 BPB. This is larger than the projected BPB increase from parameter reduction. A true dropout sweep may surpass the gap.

Sweep is to be conducted at L=1 first (faster iteration, more sensitive to regularization signal). The resulting optimal values will then be retroactively applied to L=2 (and any other higher-layer formulations) for performance measurement and benchmarking to verify whether L=2's (or higher level's) val loss also improves under the L=1-tuned regularization recipe. Headline numbers accordingly.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Weight Decay

Re-tune `weight_decay`. Current value (1e-6) was only tested alongside 1e-3. More values must to be attempted (likely slightly higher is best).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Mixer Transform Ablation

Test the contribution of the FWHT slot in the per-scale mixer versus having no transform, having a Hartley transform, using a DCT-II/III pair, or employing a butterfly-parametrized learned orthogonal mixer. Measures whether FWHT specifically is necessary, or whether any orthogonal mixer of similar structure (or none at all, with the learned embedding in place) achieves equivalent performance. See [plans/other_post_release_plans.md §10](plans/other_post_release_plans.md#10-per-scale-mixer-transform-ablation) for the full design and proposed test.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Step-Time Speedups

Throughput per token of context flattens past `bs≈1024` despite linear-in-N theoretical scaling — a memory-bandwidth wall, not algorithmic. Use [`profile_step.py`](profile_step.py) to attribute step time across architectural components at `bs ∈ {256, 1024, 4096, 16384}`, then target whichever crosses 25% at `bs=16384`. Candidate quick wins: fused SwiGLU kernel (Liger / Unsloth / xformers — drop-in for the MLP block), `torch.compile(mode='reduce-overhead')` for CUDA Graphs capture, fused Adagrad, and (architectural) low-rank lifting predict/update networks. See [plans/other_post_release_plans.md §12](plans/other_post_release_plans.md#12-step-time-speedup-quick-wins-informed-by-profiler) for the full menu and decision rule.

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

### Bit-Packed PTQ Kernels

The [current PTQ path](runs.md#ptq-sweep-summary) dequantizes int8 weights to fp16 inside `forward()` and runs a standard fp16 matmul, which pays the dequant cost every step with no bandwidth win - hence the 12% generation slowdown and the fact that sub-8-bit variants compress identically to 8-bit on disk. 

Swapping `QuantizedLinear` / `QuantizedEmbedding` for fused packed-weight kernels (Marlin W8A16 / W4A16, CUTLASS `i8gemm`, bitsandbytes, Triton for the embedding lookup) fixes both: storage scales with bit-width, and each matmul reads half or a quarter as many bytes. Expected generation at batch=1 (fp16 baseline 28.8 tok/s) is **~1.4–1.6× faster** for fused uniform 8-bit and **~1.8–2.2× faster** for fused mixed 8/4/2, with BPB unchanged. See [runs.md](runs.md#post-release-bit-packed-ptq-kernels) for the full plan.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Multi-Transform Parallelization

A WaveletLM-native architecture worth exploring: the wavelet decomposition and reconstruction stay shared across nodes, but the FWHT slot in each per-scale mixer is replaced by N parallel orthogonal-transform paths (FWHT, DHT, DCT-II/III, learned butterfly orthogonals). Each node decomposes the wavelet coefficients through a different "prism" in terms of its own orthogonal basis, then a per-node mixer learns basis-specific gated interactions. Finally, node-specific inverse transforms bring outputs back to the shared wavelet coefficient space for recombination. The result is multi-spectral mixing that potentially captures structure that no single basis would, with shared scaffolding keeping per-step compute increase modest (~5-15% for N=4 since the mixer slot is small relative to MLP). This is architecturally distinct from the existing `multinodal_enabled` mode (which ensembles full-cell copies at the LM head) — the multi-transform split happens *inside* a single model.

<p align="center">
  <img src="assets/waveletlm-multi-transform.svg" alt="Multi-transform parallelization architecture" width="85%"/>
</p>

**Rationale (conjectural):** If multi-transform parallelization improves results, then the most plausible mechanism is that each orthogonal basis represents the channel-axis features in a different coordinate system simultaneously. A Walsh basis groups features by binary-symmetry pattern, a cosine basis groups them by smoothness, and a learned-orthogonal basis groups them by whatever residual structure gradient descent discovers. The same input is losslessly rotated through all bases in parallel, and the combiner weights them per-scale based on which "perspective" matters most for the signal. 

Standard transformer attention has no direct analog because (Q, K, V) projections conflate "the lens you use" with "the weights you compute" into a single learned operation. **With a semantic embedding in particular (using plain-language, human-readable feature dimensions), this may make interpretability more tractable and efficient:** a per-node, per-token-pair similarity score in the rotated basis answers "what does node K think these two tokens have in common?", making it possible to trace why two tokens are close or far depending on the conceptual lens/transform applied. 

The wavelet decomposition continues to handle sequence-axis multi-scale structure, and the multi-basis nodes add feature-axis multi-perspective structure, factorizing the two cleanly. We don't yet know whether this is the actual mechanism if it increases performance, but if it does, testing this hypothesis directly becomes the natural follow-up.

See [plans/multi_transform_parallelization.md](plans/multi_transform_parallelization.md) for the full design, the four-node reference lineup, and the prerequisite ablation (per-scale mixer transform ablation in [other_post_release_plans.md §10](plans/other_post_release_plans.md#10-per-scale-mixer-transform-ablation)).

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

### Semantic Embedding & Interpretability Work

An optional replacement for the learned token embedding is a **semantic embedding**, where each dimension is a plain-language feature (e.g. "is this token a noun?", "is this token associated with anger?", "corpus frequency in deceptive contexts") and each token or n-gram is a vector of values across those dimensions. 

WaveletLM is structurally well-suited for this: the spectral mixer can operate directly on vectorized human-readable features, and multi-scale decomposition lets the same concept be processed at different temporal granularities. The expected tradeoff is improved interpretability at a small performance cost, potentially recovered or even improved via n-gram tokens and careful feature selection for the dimensions. 

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

### Multinodal Mode (Product-of-Experts)

WaveletLM supports a baseline product-of-experts mode where multiple independent full-cell copies process the input in parallel with feature bagging and logit averaging. Enable with `multinodal_enabled: true` in the config. This mode may require stability adjustments such as a lower learning rate with `stable_parametrization` enabled, and acts as an as-yet underexplored capacity/scalability lever — a capstone for pure scale-up once the rest of the architectural roadmap settles. Distinct from [Multi-Transform Parallelization](#multi-transform-parallelization) above (which parallelizes inside a single model at the FWHT slot); the PoE mode parallelizes whole models. This existing mode and broader multi-expert techniques (sparse MoE, mutual learning, weight averaging, Git Re-Basin, & ensemble distillation) are surveyed in [plans/multinodal_training_techniques.md](plans/multinodal_training_techniques.md).

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

The goal is a 10–15B parameter configuration, trained individually on WikiText-103 and PG-19, and also on a multi-dataset mix of WikiText-103, PG-19, Pile-ArXiv, BookCorpusOpen, TinyStories, & OpenWebText. 

Inference would fit on a single RTX 4090 at fp16 and roughly half the VRAM with [uniform 8-bit PTQ](runs.md#ptq-sweep-summary). See [`runs.md`](runs.md#post-release-scaled-up-b200-configuration) for the pending run entry.

<p align="center">
  <img src="assets/divider.svg" alt="" width="50%" height="1"/>
</p>

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
