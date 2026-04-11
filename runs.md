# EXARCH Training Runs

## Sweep Index

1. [Width (C): epochs = 1, mlp_expansion = 1](#width-c-epochs--1-mlp_expansion--1)
2. [Epochs: C = 512, mlp_expansion = 1](#epochs-c--512-mlp_expansion--1)
3. [Boolean ablations part 1: C = 512, epochs = 1, mlp_expansion = 1](#boolean-ablations-part-1-c--512-epochs--1-mlp_expansion--1)
4. [Boolean ablations part 2: C = 512, epochs = 3, mlp_expansion = 1](#boolean-ablations-part-2-c--512-epochs--3-mlp_expansion--1)
5. [Best Boolean ablations combination: C=512, epochs = 3, mlp_expansion = 1](#best-boolean-ablations-combination-c512-epochs--3-mlp_expansion--1)
6. [MLP expansion: C = 512, epochs = 1, optimal booleans](#mlp-expansion-c--512-epochs--1-optimal-booleans)
7. [Memory: C = 512, epochs = 1, optimal booleans + mlp_expansion](#memory-c--512-epochs--1-optimal-booleans--mlp_expansion)
8. [Mixer depth: C = 512, epochs = 1, optimal booleans + mlp_expansion](#mixer-depth-c--512-epochs--1-optimal-booleans--mlp_expansion)
9. [MLP expansion = 50 + memory](#mlp-expansion--50--memory)
10. [Per-layer embedding (PLE): C = 512, epochs = 1, optimal booleans + mlp_expansion = 1](#per-layer-embedding-ple-c--512-epochs--1-optimal-booleans--mlp_expansion--1)
11. [Mixer depth + higher LR](#mixer-depth--higher-lr)
12. [Mixer depth stabilizers ablation](#mixer-depth-stabilizers-ablation-alpha_d-beta_d-init-1d-scaled-mixer-init)
13. [Layers: C = 512, epochs = 1, optimal booleans + mlp_expansion](#layers-c--512-epochs--1-optimal-booleans--mlp_expansion)
14. [Levels: C = 512, epochs = 1, optimal booleans + mlp_expansion + layers](#levels-c--512-epochs--1-optimal-booleans--mlp_expansion--layers-block_size--512)
15. [Low-rank factorization in spectral mixer](#low-rank-factorization-in-spectral-mixer)
16. [Lifting hidden multiplier](#lifting-hidden-multiplier)
17. [Learning rate](#learning-rate)
18. [Block size (context window)](#block-size-context-window)
19. [Dropout optimization](#dropout-optimization-c--512-5-epochs-optimal-booleans--mlp_expansion--layers--levels)
20. [Post-training quantization (PTQ)](#post-training-quantization-ptq-inference-only-applied-to-best-checkpoint)
21. [PTQ: Uniform quantization](#ptq-uniform-quantization-all-components-same-bits)
22. [PTQ: Per-scale mixed precision](#ptq-per-scale-mixed-precision-quantization)
23. [PTQ: Component isolation](#ptq-component-isolation-quantize-one-component-keep-the-rest-at-16)
24. [Best PTQ combination](#best-ptq-combination)
25. [Section](link)
26. [Section](link)
27. [Section](link)

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
| 7   | 5  | [link](#run-7) | 1.1237 | 18,738 MiB | 2,179 MiB | Overfit; best val at epoch 4, not 5. No dropout. |

### Boolean ablations part 1: C = 512, epochs = 1, mlp_expansion = 1

| Run | Setting | Value | Folder | BPB (sliding) | Params | Time | Train VRAM | Inference VRAM | Delta |
|-----|---------|-------|--------|---------------|--------|------|------------|----------------|-------|
| 4   | Baseline (all standard) | | [link](#run-4) | 1.1751 | 366.58M | 2.78h | 18,738 MiB | 2,179 MiB | |
| 8   | `semantic_feedback` | false | [link](#run-8) | 1.1737 | 361.23M | 2.83h | 17,764 MiB | 2,147 MiB | -0.0014 |
| 9   | `semantic_feedback_cross_window` | false | [link](#run-9) | 1.1748 | 366.58M | 2.62h | 18,809 MiB | 2,178 MiB | -0.0003 |
| 10  | `learned_residual` | false | [link](#run-10) | 1.1814 | 366.58M | 2.66h | 18,818 MiB | 2,178 MiB | +0.0063 |
| 11  | `use_mixer_gate` | false | [link](#run-11) | 1.2009 | 314.15M | 2.45h | 16,518 MiB | 1,878 MiB | +0.0258 |
| 12  | `skip_proj_out` | true | [link](#run-12) | 1.1835 | 361.33M | 2.60h | 18,668 MiB | 2,148 MiB | +0.0084 |
| 13  | `shared_lifting_weights` | true | [link](#run-13) | 1.1859 | 186.92M | 2.42h | 16,762 MiB | 1,150 MiB | +0.0108 |
| 14  | `lifting_linear_only` | true | [link](#run-14) | 1.1892 | 272.02M | 1.79h | 13,236 MiB | 1,637 MiB | +0.0141 |
| 15  | `tie_embedding_to_lm_head` | true | [link](#run-15) | 1.1815 | 340.85M | 2.77h | 18,523 MiB | 2,080 MiB | +0.0064 |

### Boolean ablations part 2: C = 512, epochs = 3, mlp_expansion = 1

Baseline: Run 6 (3 epochs, all defaults) = BPB 1.1169

| Run | Setting | Value | Folder | BPB (sliding) | Params | Time | Train VRAM | Inference VRAM | Delta (3ep) | Delta (1ep) | Notes |
|-----|---------|-------|--------|---------------|--------|------|------------|----------------|-------------|-------------|-------|
| 6   | Baseline (3ep) | | [link](#run-6) | 1.1169 | 366.58M | 8.34h | 18,738 MiB | 2,179 MiB | | | |
| 16  | `semantic_feedback` | false | [link](logs/wikitext-103_2026-04-05_16-08-09/log.txt) | 1.1179 | 361.23M | 7.60h | 17,764 MiB | 2,147 MiB | +0.0010 | -0.0014 | SF true is better. |
| 17  | `lifting_linear_only` | true | [link](logs/wikitext-103_2026-04-05_23-45-27/log.txt) | 1.1337 | 272.02M | 5.17h | 13,236 MiB | 1,637 MiB | +0.0168 | +0.0141 | LLO true performs worse with more epochs. |
| 18  | `shared_lifting_weights` | true | [link](logs/wikitext-103_2026-04-06_04-59-03/log.txt) | 1.1258 | 186.92M | 7.20h | 16,762 MiB | 1,150 MiB | +0.0089 | +0.0108 | SLW true performs better with more epochs. |

> **Notes:** With epochs >= 3 `semantic_feedback=true` is best, as is `lifting_linear_only=false`. It's likely that at much higher epochs, `shared_lifting_weights=true` is best; the parameters, run time, and extreme VRAM savings (~1/2 at 3 epochs!) it saves could be better used elsewhere (larger MLP, more epochs, larger micro batch size, etc.).

### Best Boolean ablations combination: C=512, epochs = 3, mlp_expansion = 1

All defaults are optimal with epochs = 3. No boolean change improved BPB. Note that shared lifting wavelets may contribute negligibly at much higher epochs, however. See previous note.

| Run | Folder | BPB (sliding) | Params | Training time | VRAM (Train/Inf) | Notes |
|-----|--------|---------------|--------|---------------|------------------|-------|
| 6   | [link](#run-6) | 1.1169 | 366.58M | 8.34h | 18,738 / 2,179 MiB | Baseline IS the best combo |

### MLP expansion: C = 512, epochs = 1, optimal booleans

| Run | mlp_expansion | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---------------|--------|---------------|--------|------------|----------------|-------|
| 4   | 1  | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | Baseline (Run 4) |
| 19  | 2  | [link](logs/wikitext-103_2026-04-06_15-09-30/log.txt) | 1.1678 | 377.08M | 19,118 MiB | 2,239 MiB | -0.0073 |
| 20  | 10 | [link](logs/wikitext-103_2026-04-06_17-51-28/log.txt) | 1.1532 | 461.04M | 21,519 MiB | 2,719 MiB | -0.0219 |
| 21  | 20 | [link](logs/wikitext-103_2026-04-06_20-38-06/log.txt) | 1.1487 | 566.00M | 24,520 MiB | 3,320 MiB | -0.0264 |
| 22  | 50 | [link](logs/wikitext-103_2026-04-06_23-32-36/log.txt) | 1.1409 | 880.88M | 33,524 MiB | 5,121 MiB | -0.0342 |

### Memory: C = 512, epochs = 1, optimal booleans + mlp_expansion

| Run | PKM | FwPKM | pkm_num_keys | fwpkm_num_keys | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|-----|-------|--------------|----------------|--------|---------------|--------|------------|----------------|-------|
| 4   | off | off | | | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | Baseline (Run 4, MLP only) |
| 23  | on  | off | 529 | | [link](logs/wikitext-103_2026-04-07_03-43-32/log.txt) | 1.1729 | 377.48M | 19,293 MiB | 2,230 MiB | PKM default; -0.0022 vs baseline |
| 24  | on  | off | 16384 | | [link](logs/wikitext-103_2026-04-07_06-37-05/log.txt) | 1.1625 | 540.91M | 21,177 MiB | 2,856 MiB | PKM large; -0.0126 vs baseline, +174M params |
| 25  | off | on  | | 529 | [link](logs/wikitext-103_2026-04-07_09-40-03/log.txt) | 1.1726 | 377.48M | 19,474 MiB | 2,251 MiB | FwPKM default; -0.0025 vs baseline |
| 26  | on  | on  | 529 | 529 | [link](logs/wikitext-103_2026-04-07_12-58-11/log.txt) | 1.1713 | 388.37M | 19,949 MiB | 2,303 MiB | PKM+FwPKM default; -0.0038 vs baseline |
| 27  | on  | on  | 16384 | 16384 | [link](logs/wikitext-103_2026-04-07_16-36-23/log.txt) | 1.1579 | 715.23M | 24,335 MiB | 4,173 MiB | PKM+FwPKM large; -0.0172 vs baseline |
| 28  | off | off | | | [link](logs/wikitext-103_2026-04-07_20-21-24/log.txt) | 1.2003 | 356.07M | 18,357 MiB | 2,118 MiB | MLP off; wavelet pipeline only; +0.0252 vs baseline |
| 29  | on  | off | 529 | | [link](logs/wikitext-103_2026-04-07_23-11-07/log.txt) | 1.1988 | 366.97M | 18,913 MiB | 2,170 MiB | MLP off, PKM only; +0.0237 vs baseline |
| 30  | on  | on  | 529 | 529 | [link](logs/wikitext-103_2026-04-08_02-19-39/log.txt) | 1.1960 | 377.86M | 19,569 MiB | 2,243 MiB | MLP off, PKM+FwPKM; +0.0209 vs baseline |
| 31  | off | on  | | 1681 | [link](logs/wikitext-103_2026-04-09_21-03-15/log.txt) | 1.1735 | 389.46M | 19,667 MiB | 2,343 MiB | FwPKM param-matched; -0.0016 vs baseline |

### Mixer depth: C = 512, epochs = 1, optimal booleans + mlp_expansion

Stacked spectral mixing within each block — adding depth to the per-scale gated transforms in Hadamard space without repeating wavelet/Hadamard passes. Each depth step: LN + gated linear + bias (no residual), final step omits LN/bias.

| Run | mixer_depth | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|--------|---------------|--------|------------|----------------|-------|-------|
| 4   | 1 | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | | Baseline (Run 4) |
| 32  | 2 | [link](logs/wikitext-103_2026-04-08_09-51-47/log.txt) | 1.1653 | 471.74M | 23,428 MiB | 2,780 MiB | -0.0098 | First depth increase |
| 33  | 3 | [link](logs/wikitext-103_2026-04-08_13-46-39/log.txt) | 1.1718 | 576.91M | 28,837 MiB | 3,381 MiB | -0.0033 | Diminishing vs depth=2 |
| 34  | 5 | [link](logs/wikitext-103_2026-04-08_18-26-31/log.txt) | NaN | 787.24M | 39,657 MiB | — | — | Diverged at step 3600 (LR=0.008); vanishing/exploding gradients without residuals |

### MLP expansion = 50 + memory

| Run | PKM | FwPKM | pkm_num_keys | fwpkm_num_keys | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|-----|-------|--------------|----------------|--------|---------------|--------|------------|----------------|-------|
| 22  | off | off | | | [link](logs/wikitext-103_2026-04-06_23-32-36/log.txt) | 1.1409 | 880.88M | 33,524 MiB | 5,121 MiB | MLP-50 baseline (from MLP sweep) |
| 35  | on  | off | 16384 | | [link](logs/wikitext-103_2026-04-09_00-16-49/log.txt) | NaN | 1055.21M | 35,882 MiB | — | Diverged at step 3600 (LR=0.008) |
| 36  | off | on  | | 16384 | [link](logs/wikitext-103_2026-04-09_03-57-03/log.txt) | 1.1411 | 1055.21M | 36,682 MiB | 6,439 MiB | FwPKM large; -0.0002 vs MLP-50 baseline; stable where PKM NaN'd |
| 37  | on  | on  | 16384 | 16384 | [link](logs/wikitext-103_2026-04-09_08-29-18/log.txt) | 1.1385 | 1229.54M | 39,041 MiB | 7,116 MiB | Both large; -0.0024 vs MLP-50; FwPKM stabilized PKM |
| 38  | on  | on  | 529 | 529 | [link](logs/wikitext-103_2026-04-09_13-22-27/log.txt) | 1.1398 | 902.67M | 34,655 MiB | 5,246 MiB | Both default; -0.0011 vs MLP-50 |

### Per-layer embedding (PLE): C = 512, epochs = 1, optimal booleans + mlp_expansion = 1

Reintroduces original token embedding as a learned per-channel residual at each block. Learned gamma (C,) per layer, zero-initialized. +0.01M params total.

| Run | per_layer_embedding | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|---------------------|--------|---------------|--------|------------|----------------|-------|-------|
| 4   | false | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | | Baseline (Run 4) |
| 39  | true  | [link](logs/wikitext-103_2026-04-09_18-06-43/log.txt) | 1.1742 | 366.59M | 18,898 MiB | 2,179 MiB | -0.0009 | +10,240 params; essentially free |

> **Note:** FwPKM trains statically (identical to PKM). Inference-time weight updates (`fwpkm_inference_update`) tested separately in generation quality, not BPB.

### Mixer depth + higher LR

| Run | mixer_depth | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|-----|--------|---------------|--------|------------|----------------|-------|-------|
| 32  | 2 | 0.01 | [link](logs/wikitext-103_2026-04-08_09-51-47/log.txt) | 1.1653 | 471.74M | 23,428 MiB | 2,780 MiB | -0.0098 | From depth sweep |
| 40  | 2 | 0.02 | [link](logs/wikitext-103_2026-04-10_00-23-31/log.txt) | NaN | 471.74M | — | — | — | Diverged step 2200 (LR=0.01); LN alone insufficient at 2x LR |

### Mixer depth stabilizers ablation: alpha_d, beta_d (init 1/D), scaled mixer init

| Run | mixer_depth | stabilizers | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|-------------|-----|--------|---------------|--------|------------|----------------|-------|-------|
| 32  | 2 | false | 0.01 | [link](logs/wikitext-103_2026-04-08_09-51-47/log.txt) | 1.1653 | 471.74M | 23,428 MiB | 2,780 MiB | -0.0098 | From depth sweep |
| 41  | 2 | true  | 0.01 | [link](logs/wikitext-103_2026-04-10_03-29-19/log.txt) | 1.1719 | 471.74M | 23,428 MiB | 2,781 MiB | -0.0032 | Stabilizers cost +0.0066 BPB vs unstabilized |
| 42  | 2 | true  | 0.02 | [link](logs/wikitext-103_2026-04-10_07-14-48/log.txt) | NaN | 471.74M | — | — | — | Diverged step 1800 (LR=0.008); stabilizers made it worse |

### Mixer depth + lower LR: can reduced LR stabilize deeper mixers?

NaN threshold is consistently at LR reaching ~0.008. Lower peak LR to stay below this boundary.

| Run | mixer_depth | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|-----|--------|---------------|--------|------------|----------------|-------|-------|
|   | 5 | 0.004 | [link](logs/wikitext-103_2026-04-10_07-59-36/log.txt) | 1.2897 | 787.24M | 39,657 MiB | 4,584 MiB | +0.1146 | Stable but severely undertrained; LR too low for 1 epoch |
|   | 10 | 0.001 | | | 1313.06M | | | | MBS=4, GA=4 to fit in VRAM; extreme depth stress test |

### Layers = 1: C = 512, epochs = 1, optimal booleans + mlp_expansion

| Run | Layers | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
|   | 1  | [link](logs/wikitext-103_2026-04-10_16-31-26/log.txt) | 1.3177 | 67.22M | 4,684 MiB | 448 MiB | 17min train; 46.6 tok/s gen |

> Note: The speed and results are so good that it warrants further testing various configurations of layers = 1 immediately.

### Layers = 1 ablations

L=1 baseline uses ~4.7 GB VRAM, leaving ~44 GB headroom. Each run takes ~17 min. Testing mixer depth, MLP width, and large batch sizes as substitutes for model layers.

| Run | mlp_expansion | mixer_depth | lr | MBS | GA | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---------------|-------------|-----|-----|-----|--------|---------------|--------|------------|----------------|-------|
|   | 1   | 1  | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_16-31-26/log.txt) | 1.3177 | 67.22M | 4,684 MiB | 448 MiB | L=1 baseline |
|   | 100 | 1  | 0.01 | 4  | 4 | [link](logs/wikitext-103_2026-04-10_17-14-18/log.txt) | 1.2469 | 119.18M | 4,324 MiB | 759 MiB | Massive MLP; -0.0708 vs L=1 baseline; 28min total |
|   | 1   | 10 | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_17-54-03/log.txt) | 1.4757 | 114.54M | 7,078 MiB | 719 MiB | MD=10 no residuals; +0.1580 vs L=1 baseline; WORSE |
|   | 1   | 10 | 0.02 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_18-42-34/log.txt) | NaN | 114.54M | — | — | — | Diverged step 4200 (LR=0.019) |
|   | 1   | 2  | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_19-29-57/log.txt) | 1.3035 | 72.48M | 4,915 MiB | 478 MiB | MD=2 no residuals; -0.0142 vs L=1 baseline |
|   | 1   | 2  | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_19-49-29/log.txt) | 1.2924 | 72.48M | 4,915 MiB | 478 MiB | MD=2 + residuals; -0.0253 vs L=1 baseline |
|   | 1   | 10 | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_20-10-24/log.txt) | 1.2936 | 114.54M | 7,078 MiB | 719 MiB | MD=10 + residuals; -0.0241 vs L=1 baseline |
|   | 100 | 10 | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_20-51-04/log.txt) | 1.5202 | 166.50M | 8,564 MiB | 1,041 MiB | MLP=100 + MD=10 no residuals; WORSE than either alone |
|   | 1   | 10 | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_21-37-45/log.txt) | 1.4391 | 114.54M | 7,082 MiB | 719 MiB | PLE=true; still worse than baseline (MD=10 no resid dominates) |
|   | 1   | 10 | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_22-16-17/log.txt) | NaN | 114.28M | 6,279 MiB | — | SF=false; NaN — SF provides critical stability at L=1 |
|   | 1   | 10 | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_22-48-20/log.txt) | NaN | 354.92M | — | — | — | C=1024, MD=10 no resid; NaN step 4500 (LR=0.01) |
|   | 1   | 2  | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-10_23-37-19/log.txt) | 1.2153 | 186.90M | 7,100 MiB | 1,113 MiB | C=1024, resid; -0.1024 vs L=1 baseline; -0.0771 vs MD=2+resid C=512 |
|   | 1   | 2  | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-11_00-15-26/log.txt) | **1.1660** | 541.57M | 13,453 MiB | 3,211 MiB | C=2048, resid; **-0.1517 vs L=1 baseline; approaches L=20!** |
|   | 1   | 2  | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-11_02-00-34/log.txt) | 1.3315 | 76.68M | 14,696 MiB | 505 MiB | block=2048, levels=11, C=512, resid; context alone doesn't help much |
|   | 100 | 2  | 0.01 | 8  | 2 | [link](logs/wikitext-103_2026-04-11_02-19-23/log.txt) | 1.2517 | 411.41M | 28,847 MiB | 2,526 MiB | Kitchen sink A: C=1024, block=2048, levels=11, PLE, resid, MLP=100 |
|   | 20  | 2  | 0.01 | 8  | 2 | | | ~2.2B | | | | Kitchen sink B: C=2048, block=2048, levels=11, PLE, resid, MLP=20 |

### Layers > 1: C = 512, epochs = 1, optimal booleans + mlp_expansion

| Run | Layers | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
|   | 1  | [link](logs/wikitext-103_2026-04-10_16-31-26/log.txt) | 1.3177 | 67.22M | 4,684 MiB | 448 MiB | 17min train; 46.6 tok/s gen |
|   | 4  | | | | | | |
|   | 10  | | | | | | |
|   | 15  | | | | | | |
|   | 18  | | | | | | |
|   | 20 | | | | | | Baseline (from MLP sweep; VRAM-fitted mlp_expansion) |
|   | 30 | | | | | | |

### Levels: C = 512, epochs = 1, optimal booleans + mlp_expansion + layers, block_size = 512

| Run | Levels | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
|   | 1 | | | | | | |
|   | 5 | | | | | | |
|   | 9 | | | | | | Baseline (default = log2(block_size=512)) |
|   | 11 | | | | | | Beyond log2(block_size=512); expect no further gain |

### Low-rank factorization in spectral mixer

| Run | low_rank | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|----------|--------|---------------|--------|------------|----------------|-------|-------|
|   | 0  | | | | | | | Baseline (full rank) |
|   | 4  | | | | | | | Adds U·V^T perturbation (~0.8M total) |
|   | 16 | | | | | | | Higher rank perturbation |

### Lifting hidden multiplier

| Run | lifting_hidden_mult | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|---------------------|--------|---------------|--------|------------|----------------|-------|-------|
|   | 1 | | | | | | | Baseline |
|   | 2 | | | | | | | Wider predict/update MLPs in wavelet lifting |
|   | 4 | | | | | | | |

### Learning rate

| Run | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-----|--------|---------------|--------|------------|----------------|-------|-------|
|   | 0.005 | | | | | | | Slower convergence, possibly more stable |
|   | 0.01  | | | | | | | Baseline |
|   | 0.02  | | | | | | | Faster; used for C=64/128/256 in width sweep |

### Block size (context window)

| Run | block_size | levels | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|------------|--------|--------|---------------|--------|------------|----------------|-------|-------|
|   | 256  | 8 | | | | | | | Half context; levels=log2(256) |
|   | 512  | 9 | | | | | | | Baseline |
|   | 1024 | 10 | | | | | | | Double context; ~2x VRAM |

### Dropout optimization: C = 512, 5 epochs, optimal booleans + mlp_expansion + layers + levels

Starting from EXARCH-research's tuned dropout values (jointly optimized at 10 epochs). Testing at 5 epochs, but comparing against better earlier 3-epoch baseline (which was better than 5-epochs earlier due to having no dropout).

| Run | Dropout values | Epochs | Folder | BPB (sliding) | Train VRAM | Inference VRAM | Train/val gap | Notes |
|-----|----------------|--------|--------|---------------|------------|----------------|---------------|-------|
|   | Baseline (all 0.0) | 3 | [link](#run-6) | 1.1169 | 18,738 MiB | 2,179 MiB | 0.43 | From epoch sweep |
|   | emb=0.1, proj=0.05, mixer=0.05, mlp=0.05, lm_head=0.12 | 5 | | | | | |  |
|   | 1.5×: emb=0.15, proj=0.075, mixer=0.075, mlp=0.075, lm_head=0.18 | 5 | | | | | | Only if still overfitting  |

### Post-training quantization (PTQ): inference-only, applied to best checkpoint

Per-scale mixed precision leveraging EXARCH's wavelet decomposition. Coarse scales (high-level semantics) get more bits; fine scales (local detail) tolerate aggressive quantization. All runs use the same trained checkpoint — no retraining needed.

**Baseline checkpoint:** best trained model from sweeps above (TBD)

#### PTQ: Uniform quantization (all components same bits)

| Run | Bits | Folder | BPB (sliding) | Model size (MiB) | Inference VRAM | Delta | Notes |
|-----|------|--------|---------------|------------------|----------------|-------|-------|
|   | 16 (baseline) | | | | | | No quantization |
|   | 8 | | | | | | Uniform INT8 |
|   | 4 | | | | | | Uniform INT4 — stress test |

#### PTQ: Per-scale mixed precision quantization

| Run | Mixer coarse | Mixer mid | Mixer fine | MLP | Lifting | Embedding | Folder | BPB (sliding) | Model size (MiB) | Inference VRAM | Delta | Notes |
|-----|-------------|-----------|------------|-----|---------|-----------|--------|---------------|------------------|----------------|-------|-------|
|   | 8 | 4 | 2 | 4 | 16 | 8 | | | | | | Default mixed config |
|   | 8 | 8 | 4 | 4 | 16 | 8 | | | | | | Conservative fine scales |
|   | 8 | 4 | 2 | 8 | 16 | 8 | | | | | | Higher MLP precision |
|   | 8 | 4 | 2 | 4 | 8 | 8 | | | | | | Quantize lifting too |
|   | 4 | 4 | 2 | 4 | 8 | 4 | | | | | | Aggressive — minimum viable |
|   | 8 | 4 | 2 | 4 | 16 | 4 | | | | | | Aggressive embedding |

#### PTQ: Component isolation (quantize one component; keep the rest at 16)

| Run | Component quantized | Bits | Folder | BPB (sliding) | Delta | Notes |
|-----|-------------------|------|--------|---------------|-------|-------|
|   | Mixer only (all scales) | 8 | | | | Mixer sensitivity |
|   | Mixer only (all scales) | 4 | | | | |
|   | MLP only | 8 | | | | MLP sensitivity |
|   | MLP only | 4 | | | | |
|   | Embedding only | 8 | | | | Embedding sensitivity |
|   | Embedding only | 4 | | | | |
|   | Lifting only | 8 | | | | Lifting sensitivity |

#### Best PTQ combination

| Run | Mixer coarse | Mixer mid | Mixer fine | MLP | Lifting | Embedding | Folder | BPB (sliding) | Model size (MiB) | Inference VRAM | Delta | Notes |
|-----|-------------|-----------|------------|-----|---------|-----------|--------|---------------|------------------|----------------|-------|-------|
|   | | | | | | | | | | | | Best combo from above; chosen to minimize size while keeping BPB delta < 0.01 |

### Planned: lower priority (fine-tuning)

| Parameter | Current | What it tests | Values |
|-----------|---------|---------------|--------|
| `grad_accum` | 2 | Effective batch size (with micro_batch_size) | TBD |
| `warmup_fraction` | 0.3 | Warmup duration; could be too long or too short | TBD |
| `grad_clip` | 1.0 | Gradient clipping threshold | TBD |

### Shared lifting weights at scale: 6 epochs

Test whether the SLW gap continues narrowing at full training length. Gap trend: +0.0108 (1ep) → +0.0089 (3ep) → ?

| Run | SLW | Folder | BPB (sliding) | Params | Training time | VRAM (Train/Inf) | Delta | Notes |
|-----|-----|--------|---------------|--------|---------------|------------------|-------|-------|
|   | false | | | | | | | Best config baseline (from best run below) |
|   | true  | | | | | | | If delta < 0.003, SLW becomes default for deployment |

### Best run: optimal config, 10 epochs, seed = 1337

Target config: C=1024, L=20, SLW=true, mlp_expansion=200, PKM=65536, FwPKM=65536, optimal booleans + dropout from sweeps. ~12B params. Requires B200 (192 GB HBM3e).

| Run | Folder | BPB (sliding) | Params | Training time | VRAM (Train/Inf) | Notes |
|-----|--------|---------------|--------|---------------|------------------|-------|
|   | | | ~12B | | ~140 GB / ~24 GB (fp16) | fp16 inference fits on a single 4090 |
|   | | | ~12B | | ~140 GB / ~6 GB (PTQ) | With PTQ; inference fits on RTX 4060 (8 GB) |

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

**Folder:** `logs/wikitext-103_2026-04-02_11-53-11/` ([log](logs/wikitext-103_2026-04-02_11-53-11/log.txt))

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

**Folder:** `logs/wikitext-103_2026-04-03_10-22-17/` ([log](logs/wikitext-103_2026-04-03_10-22-17/log.txt))

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

**Folder:** `logs/wikitext-103_2026-04-03_07-37-34/` ([log](logs/wikitext-103_2026-04-03_07-37-34/log.txt))

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

**Folder:** `logs/wikitext-103_2026-04-03_04-51-07/` ([log](logs/wikitext-103_2026-04-03_04-51-07/log.txt))

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

**Folder:** `logs/wikitext-103_2026-04-02_22-51-10/` ([log](logs/wikitext-103_2026-04-02_22-51-10/log.txt))

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

**Folder:** `logs/wikitext-103_2026-04-03_14-34-51/` ([log](logs/wikitext-103_2026-04-03_14-34-51/log.txt))

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

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-03_22-59-40/` ([log](logs/wikitext-103_2026-04-03_22-59-40/log.txt))

**Description:** C = 512, mlp_expansion = 1, epochs = 5. No dropout. Overfit — best val at epoch 4, BPB worse than 3-epoch run.

**Results:**
- Best val loss: 3.4428 (epoch 4)
- Sliding BPB: 1.1237 (BPT: 5.0645)
- Non-overlapping BPB: 1.1338 (BPT: 5.1100)
- Params: 366.58M
- Training time: 14.15h (50,929s), avg 2.82h/epoch
- Training Peak VRAM: 18,738 MiB
- Inference Peak VRAM: 2,178 MiB
- Train/val gap: 0.64 at epoch 5 (vs 0.43 at epoch 3) — clear overfitting

<details>
<summary>Generation - Standard: <i>"The history of Mureş, which Croatia held until 7 July..."</i></summary>

```
The history of Mureş , which Croatia held until 7 July . 

 Between 2 and 4 August , ZNG redeployed the 14th Infantry Brigade to southern Namia , establishing full surrender areas without being detected . Apart from RSHA logoplates , an artillery establishment unit put aside the 700 prisoners installed for the purpose of demarcation traffic and the establishment of barracks , guards , and positions . One battalion of the 366th Pan into 66th Division was assigned the task of capturing the Hungarian underworld from Maribor . The SADF opened the road attack , launched attempts to secure the barracks , electrification report , targeted capture of various documents with tactical faults and valise orders , and stores and blankets , preparing the wounded prisoners , weapons and equipment , and assisting those deployed by the regiment . 

 The planning of the ZNDOR Area Corps terminated on 31 August . During day night the brigade manoeuvred through much of the airspace of the 14th , deploying F / A @-@ 18s , two C @-@ 130Es , and five Bf 109s . At the same time , the Partisans started to aim them completely . The Kampfgruppe Carchi rouge was to be taken by western @-@ occupied ARSK troops to help rebuild the area . The 37th Guards Brigade was tasked with attacking at Neg pressed and Mazosi and the 28th Brigade halted . Two days later , 12th Infantry Brigade withdrew amidst bad weather throughout the town , now resumed by filling the border at Srenica . 
```

</details>

<details>
<summary>Generation - Strategies: <i>"The history of the United States and its expansion into southern Canada..."</i></summary>

```
The history of the United States and its expansion into southern Canada , which is now a part of North America . 

 The city has been an important centre for agriculture since the 19th century . In 1867 the population was estimated at more than 4 @,@ 000 people . By 1900 it had grown to become known as " The City of Portland 's Fairgrounds " . It later became the Union Bank Building ( formerly the headquarters building ) in 1918 , where it served until 1936 . 

 During World War II , Phoenix grew from 200 @-@ 9 % of its residents by age 25 , mainly due to poor health following the war , but still recovering much of its food . Today , the town 's economy needs significant economic growth , with large areas such as bars and restaurants being closed out . As of 2008 , there were about 400 active service personnel on the island , including the Army Air Corps and Royal Navy Service RAN aircraft . There are also many special operations facilities in the area that include military equipment such as radar and aerial surveillance . 

 = = Geography = = 

 Plymouth Harbour is located north @-@ west of Norfolk Island and lies along the eastern shore of Lake Ontario off the coast of Devonport Bay . A small tidal estuary extends across the River Thames at its mouth southwest of Glenrothes . The sea level rises in elevation approximately 1 mile ( 0 @.@ 6 km ) below sea level ; while the lower cliffs cover 2 miles ( 3 @.@ 2 km ) long , the harbour is very shallow and can reach up to 20 ft ( 7 @.@ 1 m ) above mean low water . 

 At the western edge of Bristol , the River Avon flows through the Atlantic Ocean near Port Moody towards the east , passing over the Channel Tunnel at its entrance into the Bay of Fundy just before the main landing . On the northern side of the river , the canal passes under the Great Western Railway 's railway line and crosses the Black River at the southern end of the lake . The river turns northeast again at this point , turning southeastwards toward Tiverton Lock and then heading southward past Blakeney Point to cross the River Clyde . After entering Bridgwater , the canal enters the heart of the River Lea at Paddington , leaving behind the Bridgewater Canal , which carries the A3 motorway . From here , it meets the South Wales Main Line on the River Trent , at the confluence of the River Mersey and the Stourhead Bridge
```

Metrics: MeanLogP=-1.1868 | MeanH=3.78 | D1=0.650 | D2=0.916 | D3=0.976 | Rep4=0.039

</details>

---

### Run 8

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-04_13-38-15/` ([log](logs/wikitext-103_2026-04-04_13-38-15/log.txt))

**Description:** C = 512, epochs = 1, `semantic_feedback` = false. Boolean ablation.

**Results:**
- Best val loss: 3.6294 (epoch 1)
- Sliding BPB: 1.1737 (BPT: 5.2899)
- Non-overlapping BPB: 1.1836 (BPT: 5.3344)
- Params: 361.23M
- Training time: 2.83h (10,213s)
- Training Peak VRAM: 17,764 MiB
- Delta: -0.0014 (negligible improvement)

---

### Run 9

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-04_16-31-53/` ([log](logs/wikitext-103_2026-04-04_16-31-53/log.txt))

**Description:** C = 512, epochs = 1, `semantic_feedback_cross_window` = false. Boolean ablation.

**Results:**
- Best val loss: 3.6499 (epoch 1)
- Sliding BPB: 1.1748 (BPT: 5.2949)
- Non-overlapping BPB: 1.1849 (BPT: 5.3404)
- Params: 366.58M
- Training Peak VRAM: 18,809 MiB
- Inference Peak VRAM: 2,178 MiB
- Generation metrics: MeanLogP=-1.1949 | MeanH=4.28 | D1=0.635 | D2=0.918 | D3=0.978 | Rep4=0.039
- Delta: -0.0003 (negligible)

---

### Run 10

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-04_19-12-31/` ([log](logs/wikitext-103_2026-04-04_19-12-31/log.txt))

**Description:** C = 512, epochs = 1, `learned_residual` = false. Boolean ablation.

**Results:**
- Best val loss: 3.6675 (epoch 1)
- Sliding BPB: 1.1814 (BPT: 5.3247)
- Non-overlapping BPB: 1.1916 (BPT: 5.3707)
- Params: 366.58M
- Training Peak VRAM: 18,818 MiB
- Inference Peak VRAM: 2,178 MiB
- Generation metrics: MeanLogP=-1.4711 | MeanH=4.73 | D1=0.602 | D2=0.896 | D3=0.963 | Rep4=0.078
- Delta: +0.0063 (slight degradation)

---

### Run 11

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-04_21-55-38/` ([log](logs/wikitext-103_2026-04-04_21-55-38/log.txt))

**Description:** C = 512, epochs = 1, `use_mixer_gate` = false. Boolean ablation.

**Results:**
- Best val loss: 3.7562 (epoch 1)
- Sliding BPB: 1.2009 (BPT: 5.4126)
- Non-overlapping BPB: 1.2115 (BPT: 5.4604)
- Params: 314.15M
- Training Peak VRAM: 16,518 MiB
- Inference Peak VRAM: 1,878 MiB
- Generation metrics: MeanLogP=-1.5123 | MeanH=4.70 | D1=0.533 | D2=0.851 | D3=0.957 | Rep4=0.090
- Delta: +0.0258 (significant degradation — mixer gate is critical)

---

### Run 12

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-05_00-26-06/` ([log](logs/wikitext-103_2026-04-05_00-26-06/log.txt))

**Description:** C = 512, epochs = 1, `skip_proj_out` = true. Boolean ablation.

**Results:**
- Best val loss: 3.6788 (epoch 1)
- Sliding BPB: 1.1835 (BPT: 5.3340)
- Non-overlapping BPB: 1.1935 (BPT: 5.3788)
- Params: 361.33M
- Training Peak VRAM: 18,668 MiB
- Inference Peak VRAM: 2,148 MiB
- Generation metrics: MeanLogP=-1.4088 | MeanH=4.57 | D1=0.600 | D2=0.892 | D3=0.965 | Rep4=0.076
- Delta: +0.0084 (slight degradation — proj_out contributes as channel mixing layer)

---

### Run 13

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-05_03-05-29/` ([log](logs/wikitext-103_2026-04-05_03-05-29/log.txt))

**Description:** C = 512, epochs = 1, `shared_lifting_weights` = true. Boolean ablation.

**Results:**
- Best val loss: 3.6848 (epoch 1)
- Sliding BPB: 1.1859 (BPT: 5.3448)
- Non-overlapping BPB: 1.1956 (BPT: 5.3885)
- Params: 186.92M (massive reduction — shared lifting saves ~180M params)
- Training Peak VRAM: 16,762 MiB
- Inference Peak VRAM: 1,150 MiB
- Generation metrics: MeanLogP=-1.1411 | MeanH=3.88 | D1=0.615 | D2=0.926 | D3=0.975 | Rep4=0.039
- Delta: +0.0108 (moderate degradation — per-layer lifting is worth the parameter cost)

---

### Run 14

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-05_05-33-33/` ([log](logs/wikitext-103_2026-04-05_05-33-33/log.txt))

**Description:** C = 512, epochs = 1, `lifting_linear_only` = true. Boolean ablation.

**Results:**
- Best val loss: 3.6762 (epoch 1)
- Sliding BPB: 1.1892 (BPT: 5.3593)
- Non-overlapping BPB: 1.1985 (BPT: 5.4016)
- Params: 272.02M (lighter — linear P/U vs 2-layer MLP)
- Training Peak VRAM: 13,236 MiB
- Inference Peak VRAM: 1,637 MiB
- Generation metrics: MeanLogP=-1.3912 | MeanH=4.54 | D1=0.604 | D2=0.890 | D3=0.976 | Rep4=0.039
- Delta: +0.0141 (moderate degradation — MLP-based lifting is worth the compute)

---

### Run 15

**Status:** Complete

**Folder:** `logs/wikitext-103_2026-04-05_07-23-27/` ([log](logs/wikitext-103_2026-04-05_07-23-27/log.txt))

**Description:** C = 512, epochs = 1, `tie_embedding_to_lm_head` = true. Boolean ablation.

**Results:**
- Best val loss: 3.6677 (epoch 1)
- Sliding BPB: 1.1815 (BPT: 5.3248)
- Non-overlapping BPB: 1.1912 (BPT: 5.3685)
- Params: 340.85M (saves ~26M params from shared embedding/head)
- Training Peak VRAM: 18,523 MiB
- Inference Peak VRAM: 2,080 MiB
- Generation metrics: MeanLogP=-1.4808 | MeanH=4.63 | D1=0.633 | D2=0.918 | D3=0.982 | Rep4=0.031
- Delta: +0.0064 (slight degradation — separate LM head helps)

