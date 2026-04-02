# EXARCH Training Runs

## Sweep Index

1. [Width (C): 1 epoch, exp=1](#width-c--1-epoch-exp1)
2. [Epochs: C=256, exp=1](#epochs--c256-exp1)
3. [Boolean ablations: C=256, 3 epochs, exp=1](#boolean-ablations--c256-3-epochs-exp1)
4. [MLP expansion: C=256, 3 epochs, optimal booleans](#mlp-expansion--c256-3-epochs-optimal-booleans)
5. [Layers: C=256, 3 epochs, optimal booleans + mlp_expansion](#layers--c256-3-epochs-optimal-booleans--mlp_expansion)
6. [Levels: C=256, 3 epochs, optimal booleans + mlp_expansion + layers](#levels--c256-3-epochs-optimal-booleans--mlp_expansion--layers)
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
| 1   | 64   | [link](#run-1) | 6.8359 | 11.42M | Pipeline test |
| 2   | 128  | [link](#run-2) | | | |
| 3   | 256  | [link](#run-3) | | | |
| 4   | 512  | [link](#run-4) | | | |
| 5   | 1024 | [link](#run-5) | | | |

### Epochs: C = 256, mlp_expansion = 1

| Run | Epochs | Folder | BPB (sliding) | Notes |
|-----|--------|--------|---------------|-------|
| 6   | 1  | | | Shared with width sweep (Run 3) |
| 7   | 2  | | | |
| 8   | 3  | | | Ablation baseline |
| 9   | 4  | | | |
| 10   | 5  | | | |
| 11   | 6  | | | |
| 12   | 7  | | | |
| 13   | 8  | | | |
| 14   | 9  | | | |
| 15   | 10 | | | |

### Boolean ablations: C = 256, epochs = 3, mlp_expansion = 1

| Run | Setting | Value | Folder | BPB (sliding) | Params | Delta |
|-----|---------|-------|--------|---------------|--------|-------|
| 8   | Baseline (all standard) | | | | | |
| 16   | `semantic_feedback` | false | | | | |
| 17   | `semantic_feedback_cross_window` | false | | | | |
| 18   | `learned_residual` | false | | | | |
| 19   | `use_mixer_gate` | false | | | | |
| 20   | `skip_proj_out` | true | | | | |
| 21   | `shared_lifting_weights` | true | | | | |
| 22   | `lifting_linear_only` | true | | | | |
| 23   | `tie_embedding_to_lm_head` | true | | | | |

### MLP expansion: C = 256, epochs = 3, optimal booleans

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

### Layers: C = 256, epochs = 3, optimal booleans + mlp_expansion

| Run | Layers | Folder | BPB (sliding) | Params | Notes |
|-----|--------|--------|---------------|--------|-------|
|   | 20 | | | | Baseline (from MLP sweep; VRAM-fitted mlp_expansion) |
|   | 1  | | | | |
|   | 2  | | | | |
|   | 4  | | | | |
|   | 8  | | | | |
|   | 16 | | | | |
|   | 30 | | | | |

### Levels: C = 256, epochs = 3, optimal booleans + mlp_expansion + layers

| Run | Levels | Folder | BPB (sliding) | Params | Notes |
|-----|--------|--------|---------------|--------|-------|
|   | 9 | | | | Baseline (from layers sweep; default = log2(512)) |
|   | 1 | | | | Local only (dilation=1) |
|   | 3 | | | | Short range (dilation up to 4) |
|   | 5 | | | | Mid range (dilation up to 16) |
|   | 7 | | | | Long range (dilation up to 64) |
|   | 9 | | | | Full (dilation up to 256) |
|   | 11 | | | | Beyond sequence length (dilation up to 1024; expect no gain) |

### Planned: medium priority

| Parameter | Current | What it tests | Values |
|-----------|---------|---------------|--------|
| `multinodal_enabled` | false | Multinodal product-of-experts (num_cells, cell_dim, cross-cell gating) | TBD |
| `low_rank` | 0 | Low-rank factorization in spectral mixer (0 = full rank) | TBD |
| `lifting_hidden_mult` | 1 | Hidden dim multiplier for lifting predict/update MLPs | TBD |
| `lr` | 0.02 | Initial learning rate (Adagrad is adaptive but initial LR still matters) | TBD |
| `block_size` | 512 | Context window; trades VRAM for longer-range modeling | TBD |

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
- Sliding BPB: 6.8359
- Non-overlapping BPB: 6.8617
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

**Status:** Pending

**Description:** C = 256, mlp_expansion = 1, epochs = 1. Width scaling.

---

### Run 4

**Status:** Pending

**Description:** C = 512, mlp_expansion = 1, epochs = 1. Width scaling.

---

### Run 5

**Status:** Pending

**Description:** C = 1024, mlp_expansion = 1, epochs = 1. Width scaling.

