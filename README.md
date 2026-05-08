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

**Sample E** with prompt "The soldiers marched" - naive sampling (no `--strategies`, default `temp=1.0`, and `top_p=0.95`):

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

**Sample F** with prompt "The history of" - naive sampling (no `--strategies`, default `temp=1.0`, and `top_p=0.95`):

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

**Sample F** with prompt "The history of" and naive sampling (no `--strategies`, default `temp=1.0`, and `top_p=0.95`):

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

- **Learnable lifting wavelet decomposition**: Haar-initialized predict/update MLPs (`Linear → GELU → Dropout → Linear`, hidden_dim = C) decompose each sequence into multi-scale coefficients per block, trained end-to-end with causality preserved via zero-padded dilation. Constrained to act as predict/update steps within the lifting scheme rather than arbitrary functions, with mechanical inversion (same MLPs applied in reverse order with sign-flip) — perfect reconstruction is structurally guaranteed regardless of learned weights, leaving them free to learn deviations from classical Haar during training without compromising signal recovery. ~16.8M params per (predict, update) pair at C=2048, shared across layers via `shared_lifting_weights`. Decompose/reconstruct weights are also shared per layer; untying them had negligible performance impact while saving parameters.

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

1. [(Complete) Single-Layer WaveletLM with Current Best Config](#complete-single-layer-waveletlm-with-current-best-config)
2. [(Complete) Combined Parameter Reduction and VRAM Reallocation](#complete-combined-parameter-reduction-and-vram-reallocation)
3. [(Complete) Per-Scale Configuration at Longer Block Size](#complete-per-scale-configuration-at-longer-block-size)
4. [(Complete) Wavelet Compression](#complete-wavelet-compression)
5. [(Complete) Per-Scale Mixer Width Expansion](#complete-per-scale-mixer-width-expansion)
6. [(Complete) Low Rank Ablations](#complete-low-rank-ablations)
7. [(Complete) Decompose Bypass Disablement Ablation](#complete-decompose-bypass-disablement-ablation)
8. [Wavelet Off-Diagonal Masking with Top-K Percent](#wavelet-off-diagonal-masking-with-top-k-percent)
9. [Wavelet Off-Diagonal Masking with Structured Variants](#wavelet-off-diagonal-masking-with-structured-variants)
10. [New Testing Baseline with Mixer & Wavelet Parameter Reductions](#new-testing-baseline-with-mixer--wavelet-parameter-reductions)
11. [Sparse Embedding with (p, q) Striding](#sparse-embedding-with-p-q-striding)
12. [Encoder-Decoder Embedding](#encoder-decoder-embedding)
13. [MLP Structural Compression](#mlp-structural-compression)
14. [Gradient Checkpointing](#gradient-checkpointing)
15. [Levels = 9 and 11 Revisited (Conditional on M-Sweep Survivors)](#levels--9-and-11-revisited-conditional-on-m-sweep-survivors)
16. [Optimizer Sweep (Muon → AdamW)](#optimizer-sweep-muon--adamw)
17. [Bisected-Block Context Extension (DeepSeek-V4 HCA-Inspired)](#bisected-block-context-extension-deepseek-v4-hca-inspired)
18. [Dropout Sweep](#dropout-sweep)
19. [Weight Decay Sweep](#weight-decay-sweep)
20. [Per-scale Mixer Transform Ablation](#per-scale-mixer-transform-ablation)
21. [Step-Time Speedup Quick Wins](#step-time-speedup-quick-wins)
22. [2D Wavelet over (Batch, Token) with Sequential Training](#2d-wavelet-over-batch-token-with-sequential-training)
23. [Longer PG-19 Training](#longer-pg-19-training)
24. [Dataset Comparisons](#dataset-comparisons)
25. [Model Comparisons](#model-comparisons)
26. [Bit-Packed PTQ Kernels](#bit-packed-ptq-kernels)
27. [Multi-Transform Parallelization](#multi-transform-parallelization)
28. [Semantic Embedding & Interpretability Work](#semantic-embedding--interpretability-work)
29. [Combined Multi-Transform + Semantic Embedding (Interpretability Compound)](#combined-multi-transform--semantic-embedding-interpretability-compound)
30. [Adaptive Decompose Bypass](#adaptive-decompose-bypass)
31. [Multinodal Mode (Product-of-Experts)](#multinodal-mode-product-of-experts)
32. [Scaled-Up Model (B200)](#scaled-up-model-b200)
33. [Other Post-Release Plans](#other-post-release-plans)

### (Complete) Single-Layer WaveletLM with Current Best Config

**Result:** 

Both runs were trained at near-equal wall-clock (L=1 E=8: 15.86h; L=2 E=5: 16.25h). Comparing the minimum training and validation losses observed across the entire run:

| Run | Layers | Epochs | BPB sliding | PPL sliding | Params | Train time | Links |
|-----|--------|--------|-------------|-------------|--------|------------|-------|
| A | 1 | 1 | 1.1648 | 38.04 | 586.15M | ~1.5h | [link](logs/wikitext-103_2026-04-29_20-45-37/log.txt) |
| B | 2 | 1 | 1.1129 | 32.35 | 882.51M | ~3h | [link](logs/wikitext-103_2026-04-29_22-52-28/log.txt) |
| C | 1 | 5 | 1.0809 | 29.28 | 586.15M | 9.74h | [link](logs/wikitext-103_2026-04-30_02-20-35/log.txt) |
| D | 2 | 5 (baseline) | **1.0140** | **23.75** | 882.51M | 16.25h | [link](logs/wikitext-103_2026-04-22_01-36-47/log.txt) |
| E | 1 | 8 (compute-equalized) | 1.0715 | 28.43 | 586.15M | 15.86h | [link](logs/wikitext-103_2026-04-30_12-20-45/log.txt) |

**Decision:** L=1 becomes the iteration platform for upcoming ablations, capped at E=5 (val loss saturates there) due to iteration time. L=2 will be rebenchmarked once the L=1-tuned regularization recipe is applied retroactively.

### (Complete) Combined Parameter Reduction and VRAM Reallocation

**Result:** Parameters reduced by ~42% at minimal BPB cost. Made context longer with freed VRAM (bs=16384: stable training, -34% VRAM usage, modest val regression attributable to `levels=5` leaving 9 of 14 possible decomposition levels unused). See [plans/findings.md](plans/findings.md#combined-parameter-reduction-at-least-equivalent-bpb-at-l1-ebs-scaling-hurts), [plans/other_post_release_plans.md §8](plans/other_post_release_plans.md#8-combined-parameter-reduction-and-vram-reallocation), and [`runs.md`](runs.md) for the full analysis and per-test numbers.

**Decision:** Reduce parameters, use `block_size=16384`, and test increasing levels next.

### (Complete) Per-Scale Configuration at Longer Block Size

**Result:** [levels=7](logs/wikitext-103_2026-05-03_02-13-07/log.txt) wins the L=1 / bs=16384 / `wavelet_crawl=false` sweep at 5-epoch BPB sliding **1.0974** (392.91M params, 23.4 GiB train VRAM). `levels=11` and `levels=9` NaN in fp16 AMP under every stability fix attempted; `levels=13` OOMs without `gradient_checkpointing`. The FWHT is orthogonal, so the runaway is upstream: likely the lifting cascade or Adagrad's sparse-gradient amplification. The two unblockers are lower peak LR (rejected on performance) and a different optimizer (deferred to the [Optimizer Sweep](#optimizer-sweep-muon--adamw)).

**Decision:** ship `levels=7`; `levels ≥ 9` deferred until the optimizer sweep clears the cascade picture.

### (Complete) Wavelet Compression

**Result:** D + U·V^T compression at r=16 (lifting_diaglowrank, [A1](logs/wikitext-103_2026-05-04_16-22-02/log.txt)) reduced lifting from 117.44M → 3.33M (97%) and total params 29%, but landed at 1-epoch BPB sliding 1.2860 vs reference 1.2361. Cross-level group sharing (lifting_level_sharing) NaN'd at step 2250. [`tools/analyze_lifting.py`](tools/analyze_lifting.py) confirmed the structural picture (~75% diagonal energy, weak generic low-rank); the failure is the compression's expressive ceiling, not the analysis.

**Decision:** Deprecated in favor of other off-diagonal compression schemes in these sections: [Wavelet Off-Diagonal Masking with Top-K Percent](#wavelet-off-diagonal-masking-with-top-k-percent) and [Wavelet Off-Diagonal Masking with Structured Variants](#wavelet-off-diagonal-masking-with-structured-variants)

### (Complete) Per-Scale Mixer Width Expansion

**Result:** both directions tested at L=1 / levels=7 / bs=16384.
- *Contraction*: `per_scale_mixer_widths=[0.5×4, 0.25×4]` ([logs/wikitext-103_2026-05-05_05-48-40](logs/wikitext-103_2026-05-05_05-48-40/log.txt)) achieves a 1-epoch sliding BPB of **1.2437** vs the default's 1.2361 = +0.0076 with 39% fewer mixer parameters (35.70M vs 58.82M). Aggressive contraction at `[0.1×4, 0.05×4]` ([logs/wikitext-103_2026-05-05_04-37-47](logs/wikitext-103_2026-05-05_04-37-47/log.txt)) NaN'd at step 1250.
- *Expansion*: E5_5ep at `[1.5×4, 0.5×4]` ([logs/wikitext-103_2026-05-05_14-00-32](logs/wikitext-103_2026-05-05_14-00-32/log.txt)) lands at 5-epoch BPB **1.1037** vs [headline 1.0974](logs/wikitext-103_2026-05-03_02-13-07/log.txt) = +0.0063 at +24% mixer params (no improvement).

Full details in [runs.md → Mixer width contractions](runs.md#mixer-width-contractions-post-combined-reduction-baseline-l1-levels7-epochs1).

**Decision:** `per_scale_mixer_widths=[0.5×4, 0.25×4]` contraction is best. Follow-up tightenings (0.4/0.2, 0.3/0.15, 0.25/0.125) worth testing if the above's 5-epoch confirmation ships cleanly.

### (Complete) Low Rank Ablations

**Result:** `low_rank=16` (LR16) achieves a 1-epoch sliding BPB of 1.2342 (-0.0019 vs reference 1.2361). 5-epoch confirmation ([LR16_5ep](logs/wikitext-103_2026-05-05_21-36-45/log.txt)) yielded 1.0971 vs. the baseline 1.0974 = −0.0003, so essentially unchanged. The 1-epoch advantage didn't amplify with more training; the win was likely early-training expressivity that washes out at convergence. Full table in [runs.md → Low-rank ablations](runs.md#low-rank-ablations-post-combined-reduction-baseline-l1-levels7-epochs1).

**Decision:** keep `low_rank=4`. +1.92M params for a 5-epoch tie isn't worth it.

### (Complete) Decompose Bypass Disablement Ablation

**Result:** At L=1 / levels=7 / bs=16384 / `low_rank=16`, disabling both `decompose_bypass` and `decompose_bypass_cross_window` ([logs/wikitext-103_2026-05-05_12-47-12](logs/wikitext-103_2026-05-05_12-47-12/log.txt)) **NaN'd at step 1500, lr=6.84e-3**. Prior smaller-scale ablations projected free disablement (within ±0.0015 BPB at L=1 / E=1), but the projection does not survive once `levels=7` and `low_rank=16` are stacked.

**Decision:** Keep both flags `true`. Removal is off the table until the [Optimizer Sweep](#optimizer-sweep-muon--adamw) increases stability. If Muon clears the cascade-amplification issue structurally, disablement can be retested.

### Wavelet Off-Diagonal Masking with Top-K Percent

**Results:** Wavelet diagonal + top-k percent off-diagonal mask, ranked by magnitude on the [5-epoch reference checkpoint](logs/wikitext-103_2026-05-03_02-13-07/log.txt) using the [analyze_lifting.py script](tools/analyze_lifting.py). The top_k 10% run recovers 81.5% of the diagonal-only-vs-full gap at +0.0096 BPB with 89.9% lifting parameter reduction. **Unfortunately**, using top-k compression requires a reference checkpoint on the same architecture to compute the mask, so it's not portable to new architectures without redoing training. Structural variants below avoid this setback. Full results in [runs.md](runs.md#wavelet-off-diagonal-masking-with-top-k-percent-in-progress-l1-levels7-epochs1).

**Decision:** Not to be implemented due to duplicate training requirement above.

### Wavelet Off-Diagonal Masking with Structured Variants

Config options:
  `"lifting_offdiag_structure"`
  `"lifting_block_size"`
  `"lifting_band_width"`
  `"lifting_monarch_blocks"`
  `"lifting_offdiag_density"`
  `"lifting_offdiag_mask_seed"`
  `"lifting_offdiag_mask_checkpoint"`

**Results:**

| Variant | Lifting params | % of full params | Sliding BPB | % gap recovered |
|---------|---------------:|----------:|------------:|----------------:|
| Diagonal only (M0 / A1) | 3.33M | 2.83% | 1.2860 | 0% (floor) |
| T upper | 58.81M | 50.05% | 1.2381 | 92.5% |
| T lower | 58.81M | 50.05% | 1.2380 | 92.7% |
| BD 64 | 3.73M | 3.17% | 1.2613 | 47.7% |
| BD 128 | 7.40M | 6.30% | 1.2564 | 57.1% |
| BD 256 | 14.74M | 12.55% | 1.2509 | 67.8% |
| BD 512 | 29.42M | 25.04% | 1.2444 | 80.3% |
| BAND 32 | 3.76M | 3.20% | 1.2622 | 46.0% |
| BAND 64 | 7.34M | 6.25% | 1.2563 | 57.3% |
| BAND 128 | 14.33M | 12.20% | 1.2508 | 68.0% |
| BAND 256 | 27.63M | 23.51% | 1.2445 | 80.1% |
| MON 32 | 5.56M | 4.73% | 1.2614 | 47.5% |
| MON 64 | 5.56M | 4.73% | 1.2615 | 47.3% |
| MON 128 | 8.31M | 7.07% | 1.2584 | 53.3% |
| MON 256 | 15.20M | 12.94% | 1.2533 | 63.1% |
| Full wavelet (LR16) | 117.50M | 100.00% | 1.2342 | 100% (ceiling) |

**Decision:** BAND 128 achieves an 87.8% lifting wavelet parameter reduction at +0.0147 BPB cost vs. the uncompressed reference, striking a comfortable balance between efficiency and performance.

### New Testing Baseline with Mixer & Wavelet Parameter Reductions

Merges the banked wins (W2 per-scale mixer widths, `low_rank=4`, BAND 128 lifting) into a single combined-reductions baseline. All downstream compression sweeps (embedding, MLP, FwPKM) compare against this, not the old 1.2361 / 1.0974 references.

**Settings:** `layers=1`, `levels=7`, `block_size=16384`, `low_rank=4`, `per_scale_mixer_widths=[0.5×4, 0.25×4]`, `lifting_offdiag_structure="banded"`, `lifting_band_width=128`.

**Result vs the previous L=1 baseline ([log](logs/wikitext-103_2026-05-08_07-19-49/log.txt)):**

| | LR16 (former baseline) | CB (Compressed Baseline) | Δ |
|---|---|---|---|
| Total params | 393.21M | **266.63M** | **−32.2%** |
| Shared lifting | 117.50M | 14.33M | −87.8% |
| Mixer/layer | 59.11M | 35.70M | −39.6% |
| MLP/layer | 83.91M | 83.91M | unchanged |
| Token embedding | 102.93M | 102.93M | unchanged |
| Training peak VRAM | 23,416 MiB | 23,110 MiB | −1.3% |
| Inference peak VRAM | 3,260 MiB | **3,010 MiB** | −7.7% |
| BPB sliding | 1.2342 | **1.2586** | +0.0244 |

**Composability:** the +0.0244 BPB cost is **94% of the linear sum** of the individual costs (W2's +0.0095 + BAND 128's +0.0166 = +0.0261). Wins are essentially independent — no compounding gain, no negative interaction.

**Why training VRAM barely moved despite -32% params:** training VRAM is dominated by activation memory (forward intermediates saved for backward), not weights. Parameter compression saves the Adagrad accumulator and checkpoint storage but barely touches activation totals — the MLP hidden activation alone (`1 × 16384 × 20480` = 671 MB in fp16) plus per-scale wavelet intermediates dwarf the weights. The actual compression payoff lives in **inference VRAM** (no backward, no optimizer state, no saved activations) and **checkpoint size**.

**Inference VRAM also moved less than expected (−7.7% for −32% params)** because at inference the weights are only a minority of total process VRAM (786 MB out of 3,260 MiB at LR16, ~24%) — the rest is the CUDA context, cuDNN/cuBLAS workspaces, the PyTorch caching allocator's reserved headroom, the torch.compile cache, forward-pass activations, and cross-window decompose-bypass state. Most of that overhead is fixed-size and doesn't scale with parameter count, so compressing weights from 786 → 533 MB (−32%) only moves the weight-share of total VRAM. To see large inference VRAM wins we'd need to also reduce activation costs (smaller `block_size`, smaller `mlp_expansion`, smaller `C`) — see [runs.sh](runs.sh) for the inference-VRAM-measurement step that records the user-facing number into each run's `generations.txt`.

### Sparse Embedding with (p, q) Striding

The structural-variant sweep above compresses the lifting cascade. The token embedding (102.93M params, ~26% of the model — larger than MLP, mixer, or FwPKM individually) is the next natural compression target. Standard approaches exist (ALBERT-style factorization, vocab pruning, hash embeddings); the **(p, q) phantom-token striding scheme** is a number-theoretic alternative that preserves the (N × C) embedding shape exactly, masks the embedding via a deterministic 1D walk over the flattened tensor with alternating step sizes p and q (density = 2/(p+q)), and uses a "phantom token" trick to pad N to a value with useful divisibility properties without ever allocating the padded rows.

The scheme is content-blind, deterministic, has O(1) metadata cost, and ships with a **q ≈ √C structural-mode heuristic** that aligns with the Monarch / butterfly factorization philosophy already empirically validated in the lifting compression. A planned ablation compares (p=18, q=2) vs (p=12, q=8) at d = 10% to test whether macrocell structure matters empirically, and benchmarks both against a `random_topk` content-blind control at matched density.

Full scheme, requirements, selection algorithm, worked candidates for C=2048 at common densities, cognitive/linguistic framing, and the planned ablation are in [plans/new_compression_ideas.md](plans/new_compression_ideas.md).

### Encoder-Decoder Embedding

A second compression scheme for the token embedding, parallel to the (p, q) striding above but with different structural commitments. **Forward path:** tokens → `embedding(V × C_emb)` → learnable decoder `(C_emb, C, bias=True)` → C-dim model interior. **Output path:** C-dim hidden → learnable encoder `(C, C_emb, bias=True)` → tied vocab projection via `embedding.weight^T` → V logits.

The decoder and encoder are **separate learnable matrices** because the model's nonlinearities (GELU in MLP, gating in mixer, lifting cascade) make the output-side hidden a nonlinear transform of the input-side embedding — sharing the same matrix in transposed form would force a sub-optimal output compression. Total params: `V·C_emb + 2·C·C_emb + C + C_emb` (vs dense `V·C`). Implementation in [tools/encoder_decoder_embedding.py](tools/encoder_decoder_embedding.py).

| C_emb | Total params | % of dense (V·C) | Reduction |
|---|---|---|---|
| 128 | 6.95M | 6.75% | 93% |
| 256 | 13.92M | 13.53% | 87% |
| **512** | **27.83M** | **27.04%** | **73%** |
| 1024 | 55.66M | 54.07% | 46% |
| 2048 (= C) | 111.33M | 108.16% | none — full-rank refinement ablation |

**Why this might beat (p, q) on the failure modes we hit:** The sparse-embedding (p, q) NaN'd at peak LR partly because sparse output activations (90% zeros) stress downstream LayerNorms, with variance dominated by 10% of dims and ~3× amplification of active values. The encoder-decoder embedding produces **dense outputs** (the decoder mixes all `C_emb` dims into all `C` output dims), so this whole class of LayerNorm-amplification failure goes away.

**The C_emb=2048 case** is a no-compression ablation: when `C_emb == C`, the decoder and encoder become learnable C×C matrices acting as affine refinement layers around the standard embedding. Tests whether the encoder/decoder machinery itself helps (capacity-rich) even without the parameter savings. If this one wins on 1-epoch BPB cleanly over the combined-reductions baseline, the architectural addition is justified independently of compression.

**Sweep:** five 1-epoch ablations at C_emb ∈ {128, 256, 512, 1024, 2048}, layered on the new combined-reductions baseline. Locate the elbow on the recovery-vs-density curve.

**Initial results (5 of 5 tied runs landed; 2 larger probes planned):** The BPB gap closes monotonically as C_emb grows, hitting +0.0022 at C_emb = C and not crossing under. The encoder/decoder pair is **essentially free** at C_emb = C — neither a help nor a cost — which means the architectural-bonus hypothesis at C_emb = C is **not confirmed**. The C_emb > C extension probes whether forcing the embedding through a C-bottleneck unlocks additional BPB; pessimistically, ~−0.03 vs CB (compressed baseline) is the realistic ceiling.

| C_emb | Total params | Train VRAM | Inference VRAM | BPB sliding | ΔBPB vs the CB |
|---|---|---|---|---|---|
| 128 ([log](logs/wikitext-103_2026-05-08_09-12-05/log.txt)) | 170.66M | 22,012 MiB | 2,198 MiB | 1.4808 | +0.2222 |
| 256 ([log](logs/wikitext-103_2026-05-08_11-54-53/log.txt)) | 177.62M | 22,087 MiB | 2,452 MiB | 1.3916 | +0.1330 |
| 512 ([log](logs/wikitext-103_2026-05-08_13-37-26/log.txt)) | 191.53M | 22,235 MiB | 2,550 MiB | 1.3315 | +0.0729 |
| 1024 ([log](logs/wikitext-103_2026-05-08_14-41-41/log.txt)) | 219.36M | 22,533 MiB | 2,750 MiB | 1.2829 | +0.0243 |
| 2048 ([log](logs/wikitext-103_2026-05-08_15-47-44/log.txt)) | 275.02M | 23,128 MiB | 3,110 MiB | 1.2608 | +0.0022 |
| 4096 (planned) | 386.33M | ~24,315 MiB | ~3,870 MiB | pending | pessimistic ceiling ≈ −0.01 |
| 8192 (planned) | 608.96M | ~26,690 MiB | ~5,390 MiB | pending | pessimistic ceiling ≈ −0.03 |

**Decision: compression direction deprecated; expansion direction pursued.** The compression sweep (C_emb < C) is too BPB-costly to ship — even ED1024 (the closest to CB) is +0.024 BPB sliding, and ED512 / ED256 / ED128 give back substantial quality for inference-VRAM wins that are real but not worth the loss. The expansion direction (C_emb > C) becomes the actual opportunity here: with ED2048 landing essentially at CB, going past C_emb = C tests whether forcing the embedding through a C-dim bottleneck (many-to-few decoder) unlocks usable additional BPB. Pessimistic best-case is ~−0.03 vs CB at ED8192. The +8.39M extra params at ED2048 already learn close to identity, so any gain past C_emb = C is contingent on the bottleneck arrangement adding *new* expressiveness, not on the encoder/decoder machinery itself.

**Inference VRAM behavior under compression** (factual record, even though we're not shipping it). ED128 dropped inference VRAM by 810 MiB (−27%) for an embedding compression that saves only ~192 MB of weights at fp16. The compounding comes from the smaller embedding lookup intermediate (`(1, 16384, C_emb)` is 64 MB at C_emb=2048 vs 4 MB at C_emb=128 in fp16) plus downstream activation effects and allocator efficiency. The compression direction has the cleanest per-param inference-VRAM scaling we've measured; the BPB cost is what disqualifies it from production use, not the resource math.

**Untied LM head series:** five additional ablations at the same C_emb values but with `tie_embedding_to_lm_head=false`. The encoder is still allocated (it's required regardless of tying — it bridges C → C_emb on the output path); only the V projection matrix changes. The untied case uses a *separate* learnable `output_embedding(V × C_emb)` instead of reusing the input embedding for V projection. Adds V·C_emb params per run (e.g., +25.73M at C_emb=512). Isolates the cost of weight tying at each embedding-dim setting. Full table in [runs.md](runs.md#encoder-decoder-embedding-sweep-planned-l1-levels7-epochs1).

### MLP Structural Compression

The MLP is the second-largest single component after the token embedding (83.91M @ E=10, 167.82M @ E=20). Three structural variants apply to the MLP weight matrices W1 (C, E·C) and W2 (E·C, C):

- **Tiled banded** — view W1 as E concatenated `(C, C)` blocks left-to-right; in each block apply a bilateral band of width W. Per-block density `(2W+1)/C` matches BAND on the lifting matrices exactly.
- **Tiled block-diagonal** — same per-block view, but with block-of-blocks pattern of size b. Per-block density `b/C`. Each output "expansion group" sees only its own input group — an architecturally clean grouped-MLP / channel-grouped feedforward interpretation.
- **(p, q) striding** — single 1D walk over the flattened weight tensor, alternating step sizes p and q, with `q | C`. No phantom tokens needed since `gcd(C, E·C) = C`. Same `find_pq` algorithm as the embedding scheme; same `q ≈ √C` structural-mode default.

Lifting empirical priors (BAND 80.1% > BD 67.8% at matched density on the lifting cascade) suggest BAND likely wins on MLP too — but the MLP nonlinearity in the middle changes the calculus, BD has a cleaner architectural story (grouped MLP), and (p, q) brings a third connectivity pattern (global walk vs local band vs grouped block) into the comparison.

**Planned sweep:** four density points (25%, 12.5%, 6.25%, 3.125%) × three structures = 12 runs at 1-epoch, locating the recovery floor and identifying which structural prior wins on MLP. Stacked with the lifting + embedding compression, the production-default candidate stack lands in the 100-150M total-parameter range. Full table in [runs.md](runs.md#mlp-structural-compression-planned-l1-levels7-epochs1).

### Gradient Checkpointing

Activation memory dominates training VRAM (the `(1, 16384, 20480)` MLP hidden alone is 671 MB in fp16; per-scale wavelet intermediates contribute another ~1 GB combined). Parameter compression barely touches this, but gradient checkpointing does, by recomputing forward intermediates during backward instead of storing them.

**Expected effect** at the combined-reductions baseline:
- Training VRAM: ~23 GB → ~12 GB (−50%)
- Training compute: ~1.5–2× per epoch (extra forward per checkpointed segment)

**When to enable:** held off until a future ablation actually needs the VRAM — e.g., `block_size=32768`, `layers ≥ 2`, `mlp_expansion ≥ 20`, or `levels=11`. The `gradient_checkpointing` flag is already plumbed in `config.json`; toggle when needed. The compute cost is real (1.5–2× per ablation), so this is a reserve lever, not a default — currently the binding constraint is compute budget, not VRAM headroom.

### Levels = 9 and 11 Revisited (Conditional on M-Sweep Survivors)

The [(Complete) Per-Scale Configuration at Longer Block Size](#complete-per-scale-configuration-at-longer-block-size) sweep hit an unrecoverable NaN cliff at `levels=9` and `levels=11` under fp16 AMP — the lifting cascade was the suspected parameter-amplification source, and the unblock was deferred to the [Optimizer Sweep](#optimizer-sweep-muon--adamw). The M-series unlocks an **independent, complementary** path: if magnitude_topk at 5-10% density genuinely retains BPB performance (M3 at 64.9% gap recovered, M4 in progress), the lifting at deeper level counts becomes proportionally cheaper *and* less amplification-prone — fewer effective parameters per cascade level means less mass for fp16 to saturate.

**Plan.** Once the M-sweep and structured-variant queue finishes and the best surviving density is identified (likely M3 or M4), rerun `levels=9` and `levels=11` at L=1 / bs=16384 with magnitude_topk lifting compression at that density. Pass criteria, in order of decreasing strictness:
1. **Stability** — does the run complete without NaN? (Necessary; the original goal of the levels=9/11 deferral.)
2. **BPB sliding close to the levels=7 5-epoch headline 1.0974**, where deeper decomposition gains offset the compression cost.
3. **BPB sliding cleanly below 1.0974** — the strongest result, indicating compressed deeper cascades *outperform* uncompressed shorter ones at matched footprint.

**Approximate parameter math at M3-equivalent (94.9% lifting reduction):**

| Config | Lifting (full) | Lifting (M3-compressed) | Total model |
|--------|---------------|-------------------------|-------------|
| L=7 (current) | 117.5M | 5.98M | 393M / 282M (compressed) |
| L=9 | 151M | 7.7M | ~415M / ~283M |
| L=11 | 184M | 9.4M | ~437M / ~285M |

A compressed L=11 model lands at **~285M total params — smaller than the current uncompressed L=7 baseline (393M)** — while accessing four additional decomposition levels (effective context reach 2¹¹ = 2048× the fine token resolution, vs 2⁷ = 128×). If the cliff clears under compression alone, the M-sweep finding compounds from "lifting compression" into "depth unlock" — a structural regime change for the architecture.

**Complementary to the [Optimizer Sweep](#optimizer-sweep-muon--adamw)**, not redundant: Muon tests whether orthogonalized updates handle the cascade amplification *structurally*; lifting compression tests whether reducing the cascade's parameter mass *directly* is sufficient. If both clear independently, they may compose constructively (Muon + compressed lifting + deep levels = the deepest stable regime). If only one works, that path stands alone. If neither works individually, the combination is the natural last resort before declaring `levels ≥ 9` infeasible at fp16.

### Optimizer Sweep (Muon → AdamW)

Adagrad (lr=0.01, eps=2e-13) sits in the failure path for our two recurring NaN modes — the L=11 cascade explosion at bs=16384 and high-`low_rank` blowups (R1.5 / R2 / R3) — and is the highest-priority unblocker before regularization sweeps, which won't transfer cleanly across optimizers.

**Phase 1: Muon** ([Jordan et al., 2025](https://arxiv.org/abs/2502.16982); used in DeepSeek-V4). Newton-Schulz orthogonalization bounds every update's spectral norm — structurally the same property mHC uses to scale residual depth, applied to our matrix-heavy MLP / mixer / lifting `Linear(C, C)`. Start from DeepSeek-V4's hybrid recipe (8 iterations at (3.4445, -4.7750, 2.0315) + 2 at (2, -1.5, 0.5)); embedding / LM head / RMSNorm stay on AdamW. **Phase 2: AdamW** as fallback baseline.

**Procedure.** 1-epoch peak-LR screening at L=1 / levels=7 / bs=16384 against Adagrad reference 1.2361, 5-epoch confirmation against headline 1.0974, then retest at `levels=9 / 11` — orthogonalized updates may clear the deferred L=11 cliff. See [plans/other_post_release_plans.md §6](plans/other_post_release_plans.md#6-optimizer-sweep-adagrad--adamw--muon).

### Bisected-Block Context Extension (DeepSeek-V4 HCA-Inspired)

**Source: [DeepSeek-V4 (DeepSeek-AI, 2026)](https://huggingface.co/collections/deepseek-ai/deepseek-v4).** HCA-style summarized history as a data-loader transformation: bisect the input — recent `block_size/2` tokens uncompressed (the only loss-bearing positions), past `block_size/2` slots each holding the mean of `g = ceil((block_size_compressed − block_size/2) / (block_size/2))` consecutive corpus tokens. For `block_size=16384`, `block_size_compressed=1,000,000` → `g=122`, ~1.007M-token span; 4M context → `g=488`. Because the seam sits at a power of 2, the bisection is preserved at every wavelet level (seam moves inward, two regimes never mix within a single coarse coefficient), with O(log block_size) total seam-bridging predict/update operations. Sweep `block_size_compressed` ∈ {65K, 262K, 1M, 4M} at L=1 / levels=7 / bs=16384 against headline 1.0974, comparing BPB deltas directly; promote best to 5 epochs. Earlier tiered/recall variants archived in [plans/old_compression_ideas.md](plans/old_compression_ideas.md).

### Dropout Sweep

Re-tune the five dropout values (`dropout_lm_head`, `dropout_mlp`, `dropout_mixer`, `dropout_projection`, and `dropout_embedding`) once model parameters are reduced from above. A doubled-dropout ablation at the prior baseline gave -0.0221 BPB. This is larger than the projected BPB increase from parameter reduction. A true dropout sweep may surpass the gap.

Sweep is to be conducted at L=1 first (faster iteration, more sensitive to regularization signal). The resulting optimal values will then be retroactively applied to L=2 (and any other higher-layer formulations) for performance measurement and benchmarking to verify whether L=2's (or higher level's) val loss also improves under the L=1-tuned regularization recipe. Headline numbers accordingly.

### Weight Decay Sweep

Re-tune `weight_decay`. Current value (1e-6) was only tested alongside 1e-3. More values must to be attempted (likely slightly higher is best).

### Per-scale Mixer Transform Ablation

Test the contribution of the FWHT slot in the per-scale mixer versus having no transform, having a Hartley transform, using a DCT-II/III pair, or employing a butterfly-parametrized learned orthogonal mixer. Measures whether FWHT specifically is necessary, or whether any orthogonal mixer of similar structure (or none at all, with the learned embedding in place) achieves equivalent performance. See [plans/other_post_release_plans.md §10](plans/other_post_release_plans.md#10-per-scale-mixer-transform-ablation) for the full design and proposed test.

### Step-Time Speedup Quick Wins

Throughput per token of context flattens past `bs≈1024` despite linear-in-N theoretical scaling — a memory-bandwidth wall, not algorithmic. Use [`profile_step.py`](profile_step.py) to attribute step time across architectural components at `bs ∈ {256, 1024, 4096, 16384}`, then target whichever crosses 25% at `bs=16384`. Candidate quick wins: fused SwiGLU kernel (Liger / Unsloth / xformers — drop-in for the MLP block), `torch.compile(mode='reduce-overhead')` for CUDA Graphs capture, fused Adagrad, and (architectural) low-rank lifting predict/update networks. See [plans/other_post_release_plans.md §12](plans/other_post_release_plans.md#12-step-time-speedup-quick-wins-informed-by-profiler) for the full menu and decision rule.

### 2D Wavelet over (Batch, Token) with Sequential Training

Generalize the lifting wavelet decomposition from 1D over the token axis to 2D over the joint (batch, token) axis pair. When training proceeds in document-sequential order, the batch axis carries the same multi-scale temporal structure as the token axis, and the same wavelet machinery applies to both. This requires reorganization of the current batch sampling method into a sequential batch processing method, since batches may not be IID with respect to each other. As one example, consider series of novels in PG-19 with temporal plot dependencies between books. Randomly sampling batches breaks this temporal relationship. Using 2D wavelets is a convenient way to enforce and encode temporal relationships at all levels within the model. See [plans/two_d_wavelet_sequential_training.md](plans/two_d_wavelet_sequential_training.md) for the full design.

### Longer PG-19 Training

The PG-19 run above was trained for a single epoch using the WikiText-optimized config. Published baselines for other models on the same dataset were likely trained for many more epochs or with much more effective compute. 

Once it is possible, the first post-release goal will be to train on PG-19 for 2 epochs, and loss permitting, 5 epochs, in order to better gauge language modeling on a large dataset at the current parameter size.

### Dataset Comparisons

The best WaveletLM config trained on Pile-ArXiv, BookCorpusOpen, OpenWebText, and other datasets to gauge their performance.

### Model Comparisons

Side-by-side benchmarks against Hyena, Transformer, Mamba, RWKV, and other modern architectures on WikiText-103 at matched compute and fully optimized.

### Bit-Packed PTQ Kernels

The [current PTQ path](runs.md#ptq-sweep-summary) dequantizes int8 weights to fp16 inside `forward()` and runs a standard fp16 matmul, which pays the dequant cost every step with no bandwidth win - hence the 12% generation slowdown and the fact that sub-8-bit variants compress identically to 8-bit on disk. 

Swapping `QuantizedLinear` / `QuantizedEmbedding` for fused packed-weight kernels (Marlin W8A16 / W4A16, CUTLASS `i8gemm`, bitsandbytes, Triton for the embedding lookup) fixes both: storage scales with bit-width, and each matmul reads half or a quarter as many bytes. Expected generation at batch=1 (fp16 baseline 28.8 tok/s) is **~1.4–1.6× faster** for fused uniform 8-bit and **~1.8–2.2× faster** for fused mixed 8/4/2, with BPB unchanged. See [runs.md](runs.md#post-release-bit-packed-ptq-kernels) for the full plan.

### Multi-Transform Parallelization

A WaveletLM-native architecture worth exploring: the wavelet decomposition and reconstruction stay shared across nodes, but the FWHT slot in each per-scale mixer is replaced by N parallel orthogonal-transform paths (FWHT, DHT, DCT-II/III, learned butterfly orthogonals). Each node decomposes the wavelet coefficients through a different "prism" in terms of its own orthogonal basis, then a per-node mixer learns basis-specific gated interactions. Finally, node-specific inverse transforms bring outputs back to the shared wavelet coefficient space for recombination. The result is multi-spectral mixing that potentially captures structure that no single basis would, with shared scaffolding keeping per-step compute increase modest (~5-15% for N=4 since the mixer slot is small relative to MLP). This is architecturally distinct from the existing `multinodal_enabled` mode (which ensembles full-cell copies at the LM head) — the multi-transform split happens *inside* a single model.

<p align="center">
  <img src="assets/waveletlm-multi-transform.svg" alt="Multi-transform parallelization architecture" width="85%"/>
</p>

**Rationale (conjectural):** If multi-transform parallelization improves results, then the most plausible mechanism is that each orthogonal basis represents the channel-axis features in a different coordinate system simultaneously. A Walsh basis groups features by binary-symmetry pattern, a cosine basis groups them by smoothness, and a learned-orthogonal basis groups them by whatever residual structure gradient descent discovers. The same input is losslessly rotated through all bases in parallel, and the combiner weights them per-scale based on which "perspective" matters most for the signal. 

Standard transformer attention has no direct analog because (Q, K, V) projections conflate "the lens you use" with "the weights you compute" into a single learned operation. **With a semantic embedding in particular (using plain-language, human-readable feature dimensions), this may make interpretability more tractable and efficient:** a per-node, per-token-pair similarity score in the rotated basis answers "what does node K think these two tokens have in common?", making it possible to trace why two tokens are close or far depending on the conceptual lens/transform applied. 

The wavelet decomposition continues to handle sequence-axis multi-scale structure, and the multi-basis nodes add feature-axis multi-perspective structure, factorizing the two cleanly. We don't yet know whether this is the actual mechanism if it increases performance, but if it does, testing this hypothesis directly becomes the natural follow-up.

See [plans/multi_transform_parallelization.md](plans/multi_transform_parallelization.md) for the full design, the four-node reference lineup, and the prerequisite ablation (per-scale mixer transform ablation in [other_post_release_plans.md §10](plans/other_post_release_plans.md#10-per-scale-mixer-transform-ablation)).

### Semantic Embedding & Interpretability Work

An optional replacement for the learned token embedding is a **semantic embedding**, where each dimension is a plain-language feature (e.g. "is this token a noun?", "is this token associated with anger?", "corpus frequency in deceptive contexts") and each token or n-gram is a vector of values across those dimensions. 

WaveletLM is structurally well-suited for this: the spectral mixer can operate directly on vectorized human-readable features, and multi-scale decomposition lets the same concept be processed at different temporal granularities. The expected tradeoff is improved interpretability at a small performance cost, potentially recovered or even improved via n-gram tokens and careful feature selection for the dimensions. 

See [plans/reincorporate_large_semantic_embedding.md](plans/reincorporate_large_semantic_embedding.md) for the full design, including open questions on coefficient assignment methods: one-hot/binary, LLM-scored, human-rated, or corpus-derived.

### Combined Multi-Transform + Semantic Embedding (Interpretability Compound)

**Standing commitment regardless of intermediate results:** once both multi-transform parallelization and the semantic embedding are independently validated, combine them. The combined configuration is the unique regime in which input dimensions are human-readable, each transform node represents those features in a distinct mathematically-grounded coordinate system (a different orthogonal basis), every transform is invertible, and sequence-axis (wavelet) and feature-axis (multi-transform) structures factorize cleanly. Even if multi-transform is marginally suboptimal vs single-transform variants (mathematically unlikely, since multi-transform strictly contains the single-transform case as N=1, so that the combiner gate would simply prefer the first transform in a multi-transform situation), the combined configuration uniquely enables per-node, per-token-pair similarity scores in named feature coordinates and direct probing of "what does node K think these tokens have in common?" This combined configuration's value is qualitatively different from either component alone, and is not to be deprioritized in favor of incremental BPB wins on simpler variants.

### Adaptive Decompose Bypass

Replacing the parameter-free cumulative running mean with a data-dependent EMA (`decompose_bypass_ema`) gained -0.30 nats at 1 epoch, but regressed at 5 epochs (BPB 1.0226 vs 1.0201). The inversion likely due to short-horizon forgetting and learned gate overfitting. Post-release plan: develop freeze-gate/bias correction probes and alternative formulations with a selective SSM bypass as fallback. See [plans/ema_post_release.md](plans/ema_post_release.md).

### Multinodal Mode (Product-of-Experts)

WaveletLM supports a baseline product-of-experts mode where multiple independent full-cell copies process the input in parallel with feature bagging and logit averaging. Enable with `multinodal_enabled: true` in the config. This mode may require stability adjustments such as a lower learning rate with `stable_parametrization` enabled, and acts as an as-yet underexplored capacity/scalability lever — a capstone for pure scale-up once the rest of the architectural roadmap settles. Distinct from [Multi-Transform Parallelization](#multi-transform-parallelization) above (which parallelizes inside a single model at the FWHT slot); the PoE mode parallelizes whole models. This existing mode and broader multi-expert techniques (sparse MoE, mutual learning, weight averaging, Git Re-Basin, & ensemble distillation) are surveyed in [plans/multinodal_training_techniques.md](plans/multinodal_training_techniques.md).

### Scaled-Up Model (B200)

Conditional on the architectural research roadmap above (multi-transform parallelization, dropout sweep, semantic embedding, combined interpretability compound) producing meaningful gains, scale up the validated architecture to B200-class hardware. The 883M RTX 5090 headline run scales up naturally to:

- `C`: 2048 → 4096 
- `layers`: 2 → 4–8
- `mlp_expansion`: 20 → 50–200
- `pkm_num_keys` & `fwpkm_num_keys`: 16384 → 65536 each
- fp16 → FP8 via Blackwell tensor cores (NYI)

The goal is a 10–15B parameter configuration, trained individually on WikiText-103 and PG-19, and also on a multi-dataset mix of WikiText-103, PG-19, Pile-ArXiv, BookCorpusOpen, TinyStories, & OpenWebText. 

Inference would fit on a single RTX 4090 at fp16 and roughly half the VRAM with [uniform 8-bit PTQ](runs.md#ptq-sweep-summary). See [`runs.md`](runs.md#post-release-scaled-up-b200-configuration) for the pending run entry.

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
