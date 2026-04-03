# EXARCH Training Runs

## Sweep Index

1. [Width (C): 1 epoch, exp=1](#width-c--1-epoch-exp1)
2. [Epochs: C=512, exp=1](#epochs--c256-exp1)
3. [Boolean ablations: C=512, 3 epochs, exp=1](#boolean-ablations--c512-3-epochs-exp1)
4. [MLP expansion: C=512, 3 epochs, optimal booleans](#mlp-expansion--c512-3-epochs-optimal-booleans)
5. [Layers: C=512, 3 epochs, optimal booleans + mlp_expansion](#layers--c512-3-epochs-optimal-booleans--mlp_expansion)
6. [Levels: C=512, 3 epochs, optimal booleans + mlp_expansion + layers](#levels--c512-3-epochs-optimal-booleans--mlp_expansion--layers)
7. [Planned: medium priority](#planned--medium-priority)
8. [Planned: lower priority](#planned--lower-priority-fine-tuning)
9. [Seed variance: best EXARCH config](#seed-variance--best-exarch-config)
10. [Planned: dataset comparisons](#planned--dataset-comparisons-best-config-feasible-epochs)
11. [Planned: model comparisons](#planned--model-comparisons-wikitext-103-matched-compute)
12. [Run Details](#run-details)

---

### Width (C): epochs = 1, mlp_expansion = 1

| Run | C | Folder | BPB (sliding) | Params | Notes |
|-----|---|--------|---------------|--------|-------|
| 1   | 64   | [link](#run-1) | 1.5168 | 11.42M | Pipeline test |
| 2   | 128  | [link](#run-2) | | | |
| 3   | 256  | [link](#run-3) | 1.2929 | 98.63M | |
| 4   | 512  | [link](#run-4) | 1.1751 | 366.58M | |
| 5   | 1024 | [link](#run-5) | 1.1422 | 1362.31M | LR=0.005 |

### Epochs: C = 512, mlp_expansion = 1

| Run | Epochs | Folder | BPB (sliding) | Notes |
|-----|--------|--------|---------------|-------|
| 4   | 1  | [link](#run-4) | 1.1751 | Shared with width sweep (Run 4) |
| 5   | 2  | | | |
| 6   | 3  | | | Ablation baseline |
| 7   | 4  | | | |
| 8   | 5  | | | |
| 9   | 6  | | | |
| 10   | 7  | | | |
| 11   | 8  | | | |
| 12   | 9  | | | |
| 13   | 10 | | | |

### Boolean ablations: C = 512, epochs = 3, mlp_expansion = 1

| Run | Setting | Value | Folder | BPB (sliding) | Params | Delta |
|-----|---------|-------|--------|---------------|--------|-------|
| 6   | Baseline (all standard) | | | | | |
| 14   | `semantic_feedback` | false | | | | |
| 15   | `semantic_feedback_cross_window` | false | | | | |
| 16   | `learned_residual` | false | | | | |
| 17   | `use_mixer_gate` | false | | | | |
| 18   | `skip_proj_out` | true | | | | |
| 19   | `shared_lifting_weights` | true | | | | |
| 20   | `lifting_linear_only` | true | | | | |
| 21   | `tie_embedding_to_lm_head` | true | | | | |

### MLP expansion: C = 512, epochs = 3, optimal booleans

| Run | mlp_expansion | Folder | BPB (sliding) | Params | Notes |
|-----|---------------|--------|---------------|--------|-------|
|   | 1  | | | | Baseline (optimal booleans from ablations; may be Run 8) |
|   | 2  | | | | |
|   | 5  | | | | |
|   | 10 | | | | |
|   | 15 | | | | |
|   | 20 | | | | |
|   | 25 | | | | |
|   | 50 | | | | |

### Layers: C = 512, epochs = 3, optimal booleans + mlp_expansion

| Run | Layers | Folder | BPB (sliding) | Params | Notes |
|-----|--------|--------|---------------|--------|-------|
|   | 20 | | | | Baseline (from MLP sweep; VRAM-fitted mlp_expansion) |
|   | 1  | | | | |
|   | 2  | | | | |
|   | 4  | | | | |
|   | 8  | | | | |
|   | 16 | | | | |
|   | 20 | | | | Baseline (from MLP sweep; VRAM-fitted mlp_expansion) |
|   | 30 | | | | |

### Levels: C = 512, epochs = 3, optimal booleans + mlp_expansion + layers, block_size = 512 & stays constant

| Run | Levels | Folder | BPB (sliding) | Params | Notes |
|-----|--------|--------|---------------|--------|-------|
|   | 9 | | | | Baseline (default = log2(block_size=512)) |
|   | 1 | | | | |
|   | 3 | | | | |
|   | 5 | | | | |
|   | 7 | | | | |
|   | 9 | | | | Baseline (default = log2(block_size=512)) |
|   | 11 | | | | Beyond log2(block_size=512); expect no further gain |

### Planned: medium priority

| Parameter | Current | What it tests | Values |
|-----------|---------|---------------|--------|
| `multinodal_enabled` | false | Multinodal product-of-experts (num_cells, cell_dim, cross-cell gating) | TBD |

> **Multinodal LR note:** More nodes increase aggregate gradient variance (analogous to wider C). When doubling node count, try LR / sqrt(2) first; if still unstable, halve LR.
| `low_rank` | 0 | Low-rank factorization in spectral mixer (0 = full rank) | TBD |
| `lifting_hidden_mult` | 1 | Hidden dim multiplier for lifting predict/update MLPs | TBD |
| `lr` | 0.02 | Initial learning rate (Adagrad is adaptive but initial LR still matters) | TBD |
| `block_size` | 512 | Context window; trades VRAM for longer-range modeling. Change levels to log2(block_size) for each block size tested. | TBD |

### Planned: lower priority (fine-tuning)

| Parameter | Current | What it tests | Values |
|-----------|---------|---------------|--------|
| `grad_accum` | 2 | Effective batch size (with micro_batch_size) | TBD |
| `warmup_fraction` | 0.3 | Warmup duration; could be too long or too short | TBD |
| `grad_clip` | 1.0 | Gradient clipping threshold | TBD |
| `dropout_embedding` | 0.0 | Embedding dropout | TBD |
| `dropout_projection` | 0.0 | Post-wavelet projection dropout | TBD |
| `dropout_mixer` | 0.0 | Spectral mixer dropout | TBD |
| `dropout_mlp` | 0.0 | MLP dropout | TBD |
| `dropout_lm_head` | 0.0 | LM head dropout | TBD |

### Seed variance: best EXARCH config

3 runs at the best/most expensive config, varying only the seed. Reports mean ± std to establish statistical significance of BPB results.

| Run | Seed | Folder | BPB (sliding) | Notes |
|-----|------|--------|---------------|-------|
|   | 1337 | | | Primary (from sweeps) |
|   | 42   | | | |
|   | 7    | | | |

Mean BPB: _ ± _

### Planned: dataset comparisons (best config, feasible epochs)

| Dataset | HuggingFace ID | Domain | Folder | BPB (sliding) | Notes |
|---------|---------------|--------|--------|---------------|-------|
| WikiText-103 | `wikitext-103` | Wikipedia | | | Primary benchmark |
| PG-19 | `pg19` | Books (long-range coherence) | | | |
| Pile ArXiv | `pile-arxiv` | Academic/technical | | | |
| BookCorpusOpen | `bookcorpusopen` | Fiction | | | |
| TinyStories | `tinystories` | Simple narratives | | | Regression test |
| OpenWebText | `openwebtext` | Web text | | | |

### Planned: model comparisons (WikiText-103, matched compute)

All models use the same GPT-2 tokenizer (tiktoken, 50,257 vocab), same dataset preprocessing, and same sliding window evaluation methodology. Competitors use all available optimizations (Flash Attention, torch.compile, KV cache, etc.) to ensure the comparison reflects each architecture's best-case performance.

| Model | Type | Params | BPB (sliding) | Train tok/s | Gen tok/s | Training time | Optimizations | Notes |
|-------|------|--------|---------------|-------------|-----------|---------------|---------------|-------|
| EXARCH | Wavelet mixer | | | | | | torch.compile, fp16 | Best config from sweeps |
| GPT-2 | Transformer | | | | | | Flash Attention, KV cache, TurboQuant, torch.compile, fp16 | Matched compute |
| Mamba | SSM | | | | | | Mamba CUDA kernels, TurboQuant, torch.compile, fp16 | Matched compute |
| RWKV | Linear attention | | | | | | Custom CUDA kernels, TurboQuant, torch.compile, fp16 | Matched compute |

---

## Run Details

### Run 1

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-02_11-53-11/` ([log](../logs/wikitext-103_2026-04-02_11-53-11/log.txt))

**Description:** End-to-end pipeline test and first width scaling point. C = 64, mlp_expansion = 1, & epochs = 1.

<details>
<summary>Config</summary>

```json
{
    "dataset": "wikitext-103",
    "epochs": 1,
    "micro_batch_size": 8,
    "grad_accum": 2,
    "block_size": 512,
    "C": 64,
    "layers": 20,
    "levels": 9,
    "low_rank": 0,
    "mlp_expansion": 1,
    "mlp_layers": 2,
    "wavelet_mode": "lifting",
    "shared_lifting_weights": false,
    "lifting_linear_only": false,
    "lifting_init": "haar",
    "lifting_dropout": 0.0,
    "use_mixer_gate": true,
    "mixer_gate_activation": "silu",
    "semantic_feedback": true,
    "semantic_feedback_cross_window": true,
    "learned_residual": true,
    "skip_proj_out": false,
    "stochastic_depth_rate": 0.0,
    "dropout_embedding": 0.0,
    "dropout_projection": 0.0,
    "dropout_mixer": 0.0,
    "dropout_mlp": 0.0,
    "dropout_lm_head": 0.0,
    "optimizer": "Adagrad",
    "optimizer_eps": 2e-13,
    "lr": 2e-2,
    "min_lr": 2e-4,
    "warmup_fraction": 0.3,
    "grad_clip": 1.0,
    "tie_embedding_to_lm_head": false,
    "gradient_checkpointing": false,
    "use_amp": true,
    "amp_dtype": "fp16",
    "allow_tf32": true
}
```

</details>

**Schedule:**
- 14,614 steps/epoch, 14,614 total steps
- Warmup: 4,384 steps (30%)

**Results:**
- Val loss: 4.7216
- Sliding BPB: 1.5168 (BPT: 6.8359)
- Non-overlapping BPB: 1.5226 (BPT: 6.8617)
- Training time: 2.74h (9,843s)
- Training Peak VRAM: 4,990 MiB
- Inference Peak VRAM: 1,910 MiB

<details>
<summary>Generation - Standard: <i>"The history of Jabia to avoid dinosaur Jewish styles..."</i></summary>

```
The history of Jabia to avoid dinosaur Jewish styles , still seeing these diversifying materials from several other values , resulting in several mentionurgy units and nameed men and television . Morris also exchanged uncanny : Russian Random Park and Is Alternative Heavyweight ( 1999 appeal for all modern concept introduced Sigoleus ) , a Democratic friend of Nodkel Artorlander


 " Resticks " is a series of formous works in commercial sources , including Rockystown , Warren Caerman , R persona Back , and The Newman Original Award .ymdsis described McKinley ] with book , example , and strutions " , usually because they have believed he has something as insignificant .
```

</details>

<details>
<summary>Generation - Strategies: <i>"The history of the French Army, and the Roman Empire's capital..."</i></summary>

```
The history of the French Army , and the Roman Empire 's capital in a period of time . The British government soon became the subject of political control for the war . In 1682 , the king was involved in his reign by the First World War , but was unable to make it clear that they had been established to be a part of the U.S. Navy .


 The Spanish fleet were not only in order to establish an open @-@ controlled government , as did the " first " attack on the battle between the two and more than $ 3 million . The most important of these were the population of the city . These were also included in the 17th century and the other four members of the Republic of Germany .
```

Metrics: MeanLogP=-1.6903 | MeanH=5.88 | D1=0.529 | D2=0.857 | D3=0.953 | Rep4=0.078

</details>

---

### Run 2

**Status:** Pending

**Description:** C = 128, mlp_expansion = 1, epochs = 1. Width scaling.

---

### Run 3

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-03_07-37-34/` ([log](../logs/wikitext-103_2026-04-03_07-37-34/log.txt))

**Description:** C = 256, mlp_expansion = 1, epochs = 1. Width scaling. LR = 0.01.

**Results:**
- Val loss: 4.0077
- Sliding BPB: 1.2929 (BPT: 5.8271)
- Non-overlapping BPB: 1.3010 (BPT: 5.8636)
- Params: 104.65M
- Training time: 2.69h (9,674s)
- Training Peak VRAM: 9,958 MiB
- Inference Peak VRAM: 3,900 MiB

<details>
<summary>Generation - Standard: <i>"The history of the Holy See, demonstrated in it three-thirds..."</i></summary>

```
The history of the Holy See , demonstrated in it three @-@ thirds of the social records ( Mind Bouterki , Le Informate and Watersio Orkhyn ; in the straight Tarkathian period ) followed by reference text . The Persian cata , around the Cretaceous form of Zadar , published and commissioned by the authoriker sons of King Erik VI and their devoting members had Đvera re @-@ Magnus ( dowry ) Űitris lampāzab ἁ ( chaza or " lizard as " ) in the rut and say not to be the killer , as

 " Contracting us ! " ( 202 AD )
```

</details>

<details>
<summary>Generation - Strategies: <i>"The history of the city was also used as a site for local residents..."</i></summary>

```
The history of the city was also used as a site for local residents . In 1838 , Thomas Smith , an English scientist and politician , signed to the church in January 1748 , where he was buried with his wife 's house at Cielo Castle .

 The castle was built on the site of a new building by the Duke of York , which is now part of the Scottish @-@ style chapel . It is named after a medieval architect who had been appointed as Bishop of Kent in 1671 ; it has been suggested that the cathedral may be " one of the most important buildings of its time " . He remains the only parish church in England since 1353 .
```

Metrics: MeanLogP=-1.1437 | MeanH=3.59 | D1=0.607 | D2=0.906 | D3=0.963 | Rep4=0.070

</details>

---

### Run 4

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-03_04-51-07/` ([log](../logs/wikitext-103_2026-04-03_04-51-07/log.txt))

**Description:** C = 512, mlp_expansion = 1, epochs = 1. Width scaling. LR = 0.01.

**Results:**
- Val loss: 3.6307
- Sliding BPB: 1.1751 (BPT: 5.2962)
- Non-overlapping BPB: 1.1850 (BPT: 5.3411)
- Params: 366.58M
- Training time: 2.72h (9,775s)
- Training Peak VRAM: 18,738 MiB
- Inference Peak VRAM: 8,475 MiB

<details>
<summary>Generation - Standard: <i>"The history of the Cultivated Plant System in general three years later..."</i></summary>

```
The history of the Cultivated Plant System in general three years later as a global standard and a United Mind Boutovsk National Parliament .

 The importance of the excavation lasted until 1953 as scientists at the University of Manitoba outlined :

 The discovery of 209 elevated @-@ size soft @-@ looking iceE and its material was proven to be a highly secure species for astronomers who suggested that small bulk sales would be crucial , as they had grown by jets of particles on Earth upwards by emitting an isotope mass , not before that .
```

</details>

<details>
<summary>Generation - Strategies: <i>"The history of the war was marked by a series of events..."</i></summary>

```
The history of the war was marked by a series of events , and during this time the French army suffered from heavy losses . In February 1796 , an Italian squadron under Rear @-@ Admiral Sir John Cusack formed part of the expeditionary force that had been established in Europe as the flagship of the British Royal Navy 's Army of Northern Ireland . The ships were still at sea to take part in raids on land and sea ports ; the fleet did not reach Gibraltar until 16 March 1803 when they returned home for service with the British Isles .

 The German Imperial Navy began their career in the Mediterranean and served until the end of the Russo @-@ Japanese War . After being transferred to the Baltic , she became the flagship of the Reserve Squadron in 1806 .
```

Metrics: MeanLogP=-1.1437 | MeanH=3.59 | D1=0.607 | D2=0.906 | D3=0.963 | Rep4=0.070

</details>

---

### Run 5

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-02_22-51-10/` ([log](../logs/wikitext-103_2026-04-02_22-51-10/log.txt))

**Description:** C = 1024, mlp_expansion = 1, epochs = 1. Width scaling. LR = 0.005 (halved due to NaN at LR=0.01 and LR=0.02).

**Results:**
- Val loss: 3.5224
- Sliding BPB: 1.1422 (BPT: 5.1477)
- Non-overlapping BPB: 1.1524 (BPT: 5.1935)
- Params: 1,362.31M
- Training time: 5.94h (21,378s)
- Training Peak VRAM: 42,643 MiB
- Inference Peak VRAM: 26,069 MiB

<details>
<summary>Generation - Standard: <i>"The history of the conference is found among the Pearl Roundabout..."</i></summary>

```
The history of the conference is found among the Pearl Roundabout crossing among fanbase . London political thrift , improving approvingly by Lucifer , is the only line between the show . This event , involving the younger Queen 's story lines , led to the series focusing on South Park . Psychopathic Games to participate in baseball in the United States , during which episodes of the dead @-@ wielding reality show has moments in New York City , and mark the end of the nightly year .

 Richardson said the initiative " ultimately falls into Niccolo tradition " . Angela Lmedia dathers all , speaking in 2012 , as he seeks out the Grayly girls , newly reconciled with everyone in the main dress world , and believes the flick @-@ to @-@ back behavior changes deep into Craven , P.S. and elsewhere .
```

</details>

<details>
<summary>Generation - Strategies: <i>"The history of the University of Oxford in England is not known..."</i></summary>

```
The history of the University of Oxford in England is not known , but it may have been built as a school until 1777 . It has since become an important part of Scotland 's national identity and was largely used for worship by the Church in Wales during the English Civil War .

 The church is served on the National Register of Historic Places ( NRHP ) from December 6 , 2010 to June 30 , 2016 .

 In 1535 , when King John first visited Ireland , he made a pilgrimage to Rome with his wife Margaret , who had been baptised there . He had left the estate after that date , and married his cousin Henry de Montfort , 4th Earl of Hereford . This marriage was confirmed at the time of Edward III 's death in 1210 , while Elizabeth I 's son @-@ in @-@ law , William FitzAlan , became king .
```

Metrics: MeanLogP=-1.3842 | MeanH=4.19 | D1=0.559 | D2=0.892 | D3=0.971 | Rep4=0.055

</details>

---

### Run 6

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 3. Ablation baseline.

---

### Run 7

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 4.

---

### Run 8

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 5.

---

### Run 9

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 6.

---

### Run 10

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 7.

---

### Run 11

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 8.

---

### Run 12

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 9.

---

### Run 13

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 10.

