# EXARCH Training Runs

## Sweep Index

1. [Width (C): 1 epoch, exp=1](#width-c--1-epoch-exp1)
2. [Epochs: C=512, exp=1](#epochs--c256-exp1)
3. [Boolean ablations: C=512, 3 epochs, exp=1](#boolean-ablations--c512-3-epochs-exp1)
4. [MLP expansion: C=512, 3 epochs, optimal booleans](#mlp-expansion--c512-3-epochs-optimal-booleans)
5. [Memory (PKM/FwPKM): C=512, 3 epochs, optimal booleans + mlp_expansion](#memory--c512-3-epochs-optimal-booleans--mlp_expansion)
6. [Layers: C=512, 3 epochs, optimal booleans + mlp_expansion](#layers--c512-3-epochs-optimal-booleans--mlp_expansion)
7. [Levels: C=512, 3 epochs, optimal booleans + mlp_expansion + layers](#levels--c512-3-epochs-optimal-booleans--mlp_expansion--layers)
8. [Planned: medium priority](#planned--medium-priority)
9. [Planned: lower priority](#planned--lower-priority-fine-tuning)
10. [Seed variance: best EXARCH config](#seed-variance--best-exarch-config)
11. [Planned: dataset comparisons](#planned--dataset-comparisons-best-config-feasible-epochs)
12. [Planned: model comparisons](#planned--model-comparisons-wikitext-103-matched-compute)
13. [Run Details](#run-details)

---

### Width (C): epochs = 1, mlp_expansion = 1

| Run | C | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---|--------|---------------|--------|------------|----------------|-------|
| 1   | 64   | [link](#run-1) | 1.5168 | 11.42M | 5,110 MiB | 169 MiB | Pipeline test; LR=0.02 |
| 2   | 128  | [link](#run-2) | 1.4729 | 32.66M | 6,360 MiB | 285 MiB | LR=0.02 |
| 3   | 256  | [link](#run-3) | 1.2929 | 98.63M | 9,958 MiB | 689 MiB | LR=0.02 |
| 4   | 512  | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | LR=0.01 |
| 5   | 1024 | [link](#run-5) | 1.1422 | 1362.31M | 42,643 MiB | 7,890 MiB | LR=0.005 |

### Epochs: C = 512, mlp_expansion = 1

| Run | Epochs | Folder | BPB (sliding) | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|------------|----------------|-------|
| 4   | 1  | [link](#run-4) | 1.1751 | 18,738 MiB | 2,179 MiB | Shared with width sweep (Run 4) |
| 6   | 3  | [link](#run-6) | 1.1169 | 18,738 MiB | 2,179 MiB | Ablation baseline |
| 7   | 5  | | | | | |

### Boolean ablations: C = 512, epochs = 1, mlp_expansion = 1

| Run | Setting | Value | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta |
|-----|---------|-------|--------|---------------|--------|------------|----------------|-------|
| 4   | Baseline (all standard) | | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | |
| 8   | `semantic_feedback` | false | | | | | | |
| 9   | `semantic_feedback_cross_window` | false | | | | | | |
| 10  | `learned_residual` | false | | | | | | |
| 11  | `use_mixer_gate` | false | | | | | | |
| 12  | `skip_proj_out` | true | | | | | | |
| 13  | `shared_lifting_weights` | true | | | | | | |
| 14  | `lifting_linear_only` | true | | | | | | |
| 15  | `tie_embedding_to_lm_head` | true | | | | | | |

### MLP expansion: C = 512, epochs = 1, optimal booleans

| Run | mlp_expansion | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---------------|--------|---------------|--------|------------|----------------|-------|
|   | 1  | | | | | | Baseline (optimal booleans from ablations; may be Run 8) |
|   | 2  | | | | | | |
|   | 10 | | | | | | |
|   | 50 | | | | | | |

### Memory: C = 512, epochs = 1, optimal booleans + mlp_expansion

| Run | PKM | FwPKM | pkm_num_keys | fwpkm_num_keys | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|-----|-------|--------------|----------------|--------|---------------|--------|------------|----------------|-------|
|   | off | off | | | | | | | | Baseline (MLP only, from MLP sweep) |
|   | on  | off | 529 | | | | | | | PKM default (23x23 subkeys) |
|   | on  | off | 16384 | | | | | | | PKM large (128x128 sub-keys) |
|   | off | on  | | 529 | | | | | | FwPKM only (no static memory) |
|   | on  | on  | 529 | 529 | | | | | | PKM + FwPKM default |
|   | on  | on  | 16384 | 16384 | | | | | | PKM + FwPKM large |
|   | off | off | | | | | | | | MLP off, PKM off (wavelet pipeline only) |
|   | on  | off | 529 | | | | | | | MLP off, PKM only |

> **Note:** FwPKM trains statically (identical to PKM). Inference-time weight updates (`fwpkm_inference_update`) tested separately in generation quality, not BPB.

### Layers: C = 512, epochs = 1, optimal booleans + mlp_expansion

| Run | Layers | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
|   | 1  | | | | | | |
|   | 4  | | | | | | |
|   | 10  | | | | | | |
|   | 20 | | | | | | Baseline (from MLP sweep; VRAM-fitted mlp_expansion) |
|   | 30 | | | | | | |

### Levels: C = 512, epochs = 1, optimal booleans + mlp_expansion + layers, block_size = 512 & stays constant

| Run | Levels | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
|   | 1 | | | | | | |
|   | 5 | | | | | | |
|   | 9 | | | | | | Baseline (default = log2(block_size=512)) |
|   | 11 | | | | | | Beyond log2(block_size=512); expect no further gain |

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

### Best run: optimal config, 10 epochs, seed = 1337

| Run | Folder | BPB (sliding) | Params | Training time | VRAM (Train/Inf) | Notes |
|-----|--------|---------------|--------|---------------|------------------|-------|
|   | | | | | | All optimal settings from sweeps above |

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

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-03_10-22-17/` ([log](../logs/wikitext-103_2026-04-03_10-22-17/log.txt))

**Description:** C = 128, mlp_expansion = 1, epochs = 1. Width scaling. LR = 0.01.

**Results:**
- Val loss: 4.5788
- Sliding BPB: 1.4729 (BPT: 6.6377)
- Non-overlapping BPB: 1.4788 (BPT: 6.6642)
- Params: 32.66M
- Training time: 2.79h (10,045s)
- Training Peak VRAM: 6,360 MiB
- Inference Peak VRAM: 2,668 MiB

<details>
<summary>Generation - Standard: <i>"The history of Holy Evans. The climate three-style scene..."</i></summary>

```
The history of " Holy Evans " . The climate three @-@ style scene was also expressed by Mind Boutovsel as the Informate and Waters . Archaeologicalologist Murphy Fassemist Jerry T. Peter abamazand , of Annieyíben , is known as Christina Mackgenija . Lokis published and commissioned for Birmingham in 1940 . The.aly Clan features their own character members FC ğ Lie Soil , W. Magnus 's jets and Cyril " Toro Wil pregnant " The New Media ( chazarushotes ) as with Godalu ruté .
```

</details>

<details>
<summary>Generation - Strategies: <i>"The history of the war. The Court, however, was not a major role..."</i></summary>

```
The history of the war . The Court , however , was not a major role in the political and political life of the world .

 In 1852 , the United States became an important part of its government 's military law , and it was passed by the U.S. Army to be established in Europe in mid @-@ January 1774 , and the first city of North America ( now known as the University of New Zealand ) . It had been built at the age of 80 when a large number of American companies were destroyed by the British . The government also served on the National Register of Historic Places and the site of the National Commission for the Museum of Honor .
```

Metrics: MeanLogP=-1.4610 | MeanH=5.58 | D1=0.490 | D2=0.836 | D3=0.965 | Rep4=0.057

</details>

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

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-03_14-34-51/` ([log](../logs/wikitext-103_2026-04-03_14-34-51/log.txt))

**Description:** C = 512, mlp_expansion = 1, epochs = 3. Ablation baseline.

**Results:**
- Best val loss: 3.4292 (epoch 3)
- Sliding BPB: 1.1169 (BPT: 5.0337)
- Non-overlapping BPB: 1.1272 (BPT: 5.0801)
- Params: 366.58M
- Training time: 8.35h (30,076s), avg 2.84h/epoch
- Training Peak VRAM: 18,738 MiB
- Inference Peak VRAM: 8,475 MiB (use generate.py for accurate measurement)

<details>
<summary>Generation - Standard: <i>"The history of commercial commerce with inland sporting enterprises..."</i></summary>

```
The history of commercial commerce with inland sporting enterprises of an average age . The industries of British miners varied in frequency with good preservation . 



 = = Dance = = 



 The heart of a national theatre " brings our world around half the lead " . The temperature , sensation , movement , and Easter hall are diverse digital taste impedance , and teller swing of all the finer output . In the woven halls , almost all pavements in the Rock and roll clubs are typically open pericock . The field itself is predominantly Early English hall ( Funk Market ) . A Chicago musical theatre , called the Cub Theater , also incorporates a number of long ballets . The St Battery 's basic style style is classified with repertoire theme . Expositions with stops of a few hundred feet to form a dreamy lounge reflect social lives such as rosé : The Dveresien , the Gum accompany the widows and daughters in the convention ; football games and short piecebooks . In 1910 and 1913 the Gatehouse was the home of the main centre of music , design and storytelling . The rebuilt houses for the members and their patrons are enclosed in a single @-@ floor window , between which one in a scene is promised a gift from the ordinary , although this is concerned with the best conditions in earnest . 
```

</details>

<details>
<summary>Generation - Strategies: <i>"The history of the United States in which it is depicted..."</i></summary>

```
The history of the United States in which it is depicted . 

 The book 's title comes from its influence on " the most significant and important contribution to American literary literature " , as well as being a precursor to historical fiction . In the first volume of his biography , A.E. Scott , author of the Encyclopedia of African @-@ Americans , William Wordsworth wrote : " [ It ] contains nothing to do with the fact that the novel is about a woman who has been given access by her family to America or her old friend , but who he knows today can hardly be able to know what she wants to say ... She does not have to bear this idea that I will never write another poem for my own sake . " 

 = = Publication history = = 

 Poe was originally conceived as a publisher for an anthology called The Book of Mrs. Tiggyback ( 1868 ) ; it had been published by E. C. Wells before publication and later editions were published under the name My Life as a King . The following year , after reading the book , Crane sent a copy to George W. Bush to publish his memoirs . There was no formal suggestion , nor any agreement could be made to justify his work , so the story was changed into a single collection of letters to the public . He also published two other books — " The Great Gatsby " ( " The Longest Day " ) and " A Letter from the Last Judgment to the Revolt of the Colonies " — in each case titled " The Diary of a Black Bird " — which appeared in the February 12 , 1832 edition of the second part of the " True Story " issue , and reprinted several times over subsequent years . By June 17 , 1853 , Hemings had written six more volumes , and began publishing stories without their names until March 1915 . His last writing credit came for the third issue in May 1901 , when he suggested that he write one of the final chapters in the series , but at least three copies of the book are known . The book was published in August 1941 , and was serialized between November 1 and July 27 , 1922 . Two further short stories were collected in January 1910 : one in April 1916 , one in December 1917 , and the next four , including the first issue of the new magazine , " The Autobiography of Malcolm X " . This was followed by two short stories entitled " New Years " . 

 After some initial success , Johnson did not appear again until 1918 , though he continued writing
```

Metrics: MeanLogP=-1.3999 | MeanH=4.32 | D1=0.607 | D2=0.933 | D3=0.992 | Rep4=0.031

</details>

---

### Run 7

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 5.

---

### Run 8

**Status:** Pending

**Description:** C = 512, epochs = 1, `semantic_feedback` = false.

---

### Run 9

**Status:** Pending

**Description:** C = 512, epochs = 1, `semantic_feedback_cross_window` = false.

---

### Run 10

**Status:** Pending

**Description:** C = 512, epochs = 1, `learned_residual` = false.

---

### Run 11

**Status:** Pending

**Description:** C = 512, epochs = 1, `use_mixer_gate` = false.

---

### Run 12

**Status:** Pending

**Description:** C = 512, epochs = 1, `skip_proj_out` = true.

---

### Run 13

**Status:** Pending

**Description:** C = 512, epochs = 1, `shared_lifting_weights` = true.

---

### Run 14

**Status:** Pending

**Description:** C = 512, epochs = 1, `lifting_linear_only` = true.

---

### Run 15

**Status:** Pending

**Description:** C = 512, epochs = 1, `tie_embedding_to_lm_head` = true.

