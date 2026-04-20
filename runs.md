# WaveletLM Training Runs

## Sweep Index

1. [Width (C): epochs = 1, mlp_expansion = 1](#width-c-epochs--1-mlp_expansion--1)
2. [Epochs: C = 512, mlp_expansion = 1](#epochs-c--512-mlp_expansion--1)
3. [Boolean ablations part 1: C = 512, epochs = 1, mlp_expansion = 1](#boolean-ablations-part-1-c--512-epochs--1-mlp_expansion--1)
4. [Boolean ablations part 2: C = 512, epochs = 3, mlp_expansion = 1](#boolean-ablations-part-2-c--512-epochs--3-mlp_expansion--1)
5. [Best Boolean ablations combination: C=512, epochs = 3, mlp_expansion = 1](#best-boolean-ablations-combination-c512-epochs--3-mlp_expansion--1)
6. [MLP expansion: C = 512, epochs = 1, optimal booleans](#mlp-expansion-c--512-epochs--1-optimal-booleans)
7. [Memory: C = 512, epochs = 1, optimal booleans + mlp_expansion](#memory-c--512-epochs--1-optimal-booleans--mlp_expansion)
8. [MLP expansion = 50 + memory](#mlp-expansion--50--memory)
9. [Per-layer embedding (PLE): C = 512, epochs = 1, optimal booleans + mlp_expansion = 1](#per-layer-embedding-ple-c--512-epochs--1-optimal-booleans--mlp_expansion--1)
10. [Mixer depth: C = 512, epochs = 1, optimal booleans + mlp_expansion](#mixer-depth-c--512-epochs--1-optimal-booleans--mlp_expansion)
11. [Mixer depth + higher LR](#mixer-depth--higher-lr)
12. [Mixer depth stabilizers ablation](#mixer-depth-stabilizers-ablation-alpha_d-beta_d-init-1d-scaled-mixer-init)
13. [Mixer depth + lower LR: can reduced LR stabilize deeper mixers?](#mixer-depth--lower-lr-can-reduced-lr-stabilize-deeper-mixers)
14. [Layers = 1: C = 512, epochs = 1, optimal booleans + mlp_expansion](#layers--1-c--512-epochs--1-optimal-booleans--mlp_expansion)
15. [Layers = 1 ablations, part 1 (C = 512)](#layers--1-ablations-part-1-c--512)
16. [Layers = 1 ablations, part 2 (bigger C)](#layers--1-ablations-part-2-bigger-c)
17. [Lifting hidden multiplier: L=1, C=2048, MLP=20](#lifting-hidden-multiplier-l1-c2048-mlp20)
18. [Loop iterations (LoopLM): L=1, C=2048, reuse same weights T times](#loop-iterations-looplm-l1-c2048-reuse-same-weights-t-times)
19. [Optimal low-layer config: L=2, C=2048, full recipe](#optimal-low-layer-config-l2-c2048-full-recipe)
20. [Grokking experiment: C=128, L=2, tiny core + massive memory](#grokking-experiment-c128-l2-tiny-core--massive-memory)
21. [Layers > 1: C = 512, epochs = 1, optimal booleans with mlp_expansion](#layers--1-c--512-epochs--1-optimal-booleans-with-mlp_expansion)
22. [Levels: C = 512, epochs = 1, optimal booleans + mlp_expansion + layers](#levels-c--512-epochs--1-optimal-booleans--mlp_expansion--layers-block_size--512)
23. [Low-rank factorization in spectral mixer](#low-rank-factorization-in-spectral-mixer)
24. [Lifting hidden multiplier](#lifting-hidden-multiplier)
25. [New Baseline](#new-baseline)
26. [New Baseline Boolean ablations part 1: C=2048, L=2, epochs=1 wide & shallow model](#new-baseline-boolean-ablations-part-1-c2048-l2-epochs1-wide--shallow-model)
27. [New Baseline Boolean ablations part 2: Stable parametrization](#new-baseline-boolean-ablations-part-2-stable-parametrization)
28. [New Baseline Boolean ablations part 3: Rescue tests (vs known NaN configs)](#new-baseline-boolean-ablations-part-3-rescue-tests-vs-known-nan-configs)
29. [Block size (context window) — at new baseline](#block-size-context-window--at-new-baseline)
30. [Grad accum — at new baseline](#grad-accum--at-new-baseline)
31. [Warmup fraction — at new baseline](#warmup-fraction--at-new-baseline)
32. [Grad clip — at new baseline](#grad-clip--at-new-baseline)
33. [Best run candidate: L=2, C=2048, lr=0.01, 2.0x dropout, 5 epochs](#best-run-candidate-l2-c2048-lr001-20x-dropout-5-epochs)
34. [3-seed 10-epoch variance study: L=2, C=2048, 2.5x dropout](#3-seed-10-epoch-variance-study-l2-c2048-25x-dropout)
35. [Post-training quantization (PTQ)](#post-training-quantization-ptq-inference-only-applied-to-best-checkpoint)
36. [PTQ: Uniform quantization](#ptq-uniform-quantization-all-components-same-bits)
37. [PTQ: Per-scale mixed precision](#ptq-per-scale-mixed-precision-quantization)
38. [PTQ: Component isolation](#ptq-component-isolation-quantize-one-component-keep-the-rest-at-fp16)
39. [Best PTQ combination](#best-ptq-combination)
40. [PTQ sweep summary](#ptq-sweep-summary)
41. [Planned: model comparisons (WikiText-103, matched compute)](#planned-model-comparisons-wikitext-103-matched-compute)
42. [Planned: dataset comparisons (B200, 20+ epochs, max EBS)](#planned-dataset-comparisons-b200-20-epochs-max-ebs)
43. [Post-release: scaled-up B200 configuration](#post-release-scaled-up-b200-configuration)
44. [Run Details](#run-details)

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
| 8   | `decompose_bypass` | false | [link](#run-8) | 1.1737 | 361.23M | 2.83h | 17,764 MiB | 2,147 MiB | -0.0014 |
| 9   | `decompose_bypass_cross_window` | false | [link](#run-9) | 1.1748 | 366.58M | 2.62h | 18,809 MiB | 2,178 MiB | -0.0003 |
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
| 16  | `decompose_bypass` | false | [link](logs/wikitext-103_2026-04-05_16-08-09/log.txt) | 1.1179 | 361.23M | 7.60h | 17,764 MiB | 2,147 MiB | +0.0010 | -0.0014 | DB true is better. |
| 17  | `lifting_linear_only` | true | [link](logs/wikitext-103_2026-04-05_23-45-27/log.txt) | 1.1337 | 272.02M | 5.17h | 13,236 MiB | 1,637 MiB | +0.0168 | +0.0141 | LLO true performs worse with more epochs. |
| 18  | `shared_lifting_weights` | true | [link](logs/wikitext-103_2026-04-06_04-59-03/log.txt) | 1.1258 | 186.92M | 7.20h | 16,762 MiB | 1,150 MiB | +0.0089 | +0.0108 | SLW true performs better with more epochs. |

> **Notes:** With epochs >= 3 `decompose_bypass=true` is best, as is `lifting_linear_only=false`. It's likely that at much higher epochs, `shared_lifting_weights=true` is best; the parameters, run time, and extreme VRAM savings (~1/2 at 3 epochs!) it saves could be better used elsewhere (larger MLP, more epochs, larger micro batch size, etc.).

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

| Run | PKM | FwPKM | PKM num keys | FwPKM num keys | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
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

### MLP expansion = 50 + memory

| Run | PKM | FwPKM | PKM num keys | FwPKM num keys | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|-----|-------|--------------|----------------|--------|---------------|--------|------------|----------------|-------|
| 22  | off | off | | | [link](logs/wikitext-103_2026-04-06_23-32-36/log.txt) | 1.1409 | 880.88M | 33,524 MiB | 5,121 MiB | MLP-50 baseline (from MLP sweep) |
| 32  | on  | off | 16384 | | [link](logs/wikitext-103_2026-04-09_00-16-49/log.txt) | NaN | 1055.21M | 35,882 MiB | — | Diverged at step 3600 (LR=0.008) |
| 33  | off | on  | | 16384 | [link](logs/wikitext-103_2026-04-09_03-57-03/log.txt) | 1.1411 | 1055.21M | 36,682 MiB | 6,439 MiB | FwPKM large; -0.0002 vs MLP-50 baseline; stable where PKM NaN'd |
| 34  | on  | on  | 16384 | 16384 | [link](logs/wikitext-103_2026-04-09_08-29-18/log.txt) | 1.1385 | 1229.54M | 39,041 MiB | 7,116 MiB | Both large; -0.0024 vs MLP-50; FwPKM stabilized PKM |
| 35  | on  | on  | 529 | 529 | [link](logs/wikitext-103_2026-04-09_13-22-27/log.txt) | 1.1398 | 902.67M | 34,655 MiB | 5,246 MiB | Both default; -0.0011 vs MLP-50 |

### Per-layer embedding (PLE): C = 512, epochs = 1, optimal booleans + mlp_expansion = 1

Reintroduces original token embedding as a learned per-channel residual at each block. Learned gamma (C,) per layer, zero-initialized. +0.01M params total.

| Run | per_layer_embedding | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|---------------------|--------|---------------|--------|------------|----------------|-------|-------|
| 4   | false | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | | Baseline (Run 4) |
| 36  | true  | [link](logs/wikitext-103_2026-04-09_18-06-43/log.txt) | 1.1742 | 366.59M | 18,898 MiB | 2,179 MiB | -0.0009 | +10,240 params; essentially free |

> **Note:** FwPKM trains statically (identical to PKM). Inference-time weight updates (`fwpkm_inference_update`) tested separately in generation quality, not BPB.

### Mixer depth: C = 512, epochs = 1, optimal booleans + mlp_expansion

Stacked spectral mixing within each block — adding depth to the per-scale gated transforms in Hadamard space without repeating wavelet/Hadamard passes. Each depth step: LN + gated linear + bias (no residual), final step omits LN/bias.

| Run | mixer_depth | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|--------|---------------|--------|------------|----------------|-------|-------|
| 4   | 1 | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | | Baseline (Run 4) |
| 37  | 2 | [link](logs/wikitext-103_2026-04-08_09-51-47/log.txt) | 1.1653 | 471.74M | 23,428 MiB | 2,780 MiB | -0.0098 | First depth increase |
| 38  | 3 | [link](logs/wikitext-103_2026-04-08_13-46-39/log.txt) | 1.1718 | 576.91M | 28,837 MiB | 3,381 MiB | -0.0033 | Diminishing vs depth=2 |
| 39  | 5 | [link](logs/wikitext-103_2026-04-08_18-26-31/log.txt) | NaN | 787.24M | 39,657 MiB | — | — | Diverged at step 3600 (LR=0.008); vanishing/exploding gradients without residuals |

### Mixer depth + higher LR

| Run | mixer_depth | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|-----|--------|---------------|--------|------------|----------------|-------|-------|
| 37  | 2 | 0.01 | [link](logs/wikitext-103_2026-04-08_09-51-47/log.txt) | 1.1653 | 471.74M | 23,428 MiB | 2,780 MiB | -0.0098 | From depth sweep |
| 40  | 2 | 0.02 | [link](logs/wikitext-103_2026-04-10_00-23-31/log.txt) | NaN | 471.74M | — | — | — | Diverged step 2200 (LR=0.01); LN alone insufficient at 2x LR |

### Mixer depth stabilizers ablation: alpha_d, beta_d (init 1/D), scaled mixer init

| Run | mixer_depth | stabilizers | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|-------------|-----|--------|---------------|--------|------------|----------------|-------|-------|
| 37  | 2 | false | 0.01 | [link](logs/wikitext-103_2026-04-08_09-51-47/log.txt) | 1.1653 | 471.74M | 23,428 MiB | 2,780 MiB | -0.0098 | From depth sweep |
| 41  | 2 | true  | 0.01 | [link](logs/wikitext-103_2026-04-10_03-29-19/log.txt) | 1.1719 | 471.74M | 23,428 MiB | 2,781 MiB | -0.0032 | Stabilizers cost +0.0066 BPB vs unstabilized |
| 42  | 2 | true  | 0.02 | [link](logs/wikitext-103_2026-04-10_07-14-48/log.txt) | NaN | 471.74M | — | — | — | Diverged step 1800 (LR=0.008); stabilizers made it worse |

### Mixer depth + lower LR: can reduced LR stabilize deeper mixers?

NaN threshold is consistently at LR reaching ~0.008. Lower peak LR to stay below this boundary.

| Run | mixer_depth | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|-----|--------|---------------|--------|------------|----------------|-------|-------|
| 43  | 5 | 0.004 | [link](logs/wikitext-103_2026-04-10_07-59-36/log.txt) | 1.2897 | 787.24M | 39,657 MiB | 4,584 MiB | +0.1146 | Stable but severely undertrained; LR too low for 1 epoch |
| 44  | 10 | 0.001 | | | 1313.06M | | | | MBS=4, GA=4 to fit in VRAM; extreme depth stress test |

### Layers = 1: C = 512, epochs = 1, optimal booleans + mlp_expansion

| Run | Layers | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
| 45  | 1  | [link](logs/wikitext-103_2026-04-10_16-31-26/log.txt) | 1.3177 | 67.22M | 4,684 MiB | 448 MiB | 17min train; 46.6 tok/s gen |

> Note: The speed and results are so good that it warrants further testing various configurations of layers = 1 immediately.

### Layers = 1 ablations, part 1 (C = 512)

L=1 baseline uses ~4.7 GB VRAM, leaving ~44 GB headroom. Each run takes ~17 min. Testing mixer depth, MLP width, and large batch sizes as substitutes for model layers.

All runs use MBS=8, GA=2 except run 46 (noted below).

| Run | mlp_expansion | mixer_depth | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---------------|-------------|-----|--------|---------------|--------|------------|----------------|-------|
| 45  | 1   | 1  | 0.01 | [link](logs/wikitext-103_2026-04-10_16-31-26/log.txt) | 1.3177 | 67.22M | 4,684 MiB | 448 MiB | L=1 baseline |
| 46  | 100 | 1  | 0.01 | [link](logs/wikitext-103_2026-04-10_17-14-18/log.txt) | 1.2469 | 119.18M | 4,324 MiB | 759 MiB | **MBS=4, GA=4.** Massive MLP; -0.0708 vs L=1 baseline; 28min total |
| 47  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_17-54-03/log.txt) | 1.4757 | 114.54M | 7,078 MiB | 719 MiB | MD=10 no residuals; +0.1580 vs L=1 baseline; WORSE |
| 48  | 1   | 10 | 0.02 | [link](logs/wikitext-103_2026-04-10_18-42-34/log.txt) | NaN | 114.54M | — | — | Diverged step 4200 (LR=0.019) |
| 49  | 1   | 2  | 0.01 | [link](logs/wikitext-103_2026-04-10_19-29-57/log.txt) | 1.3035 | 72.48M | 4,915 MiB | 478 MiB | MD=2 no residuals; -0.0142 vs L=1 baseline |
| 50  | 1   | 2  | 0.01 | [link](logs/wikitext-103_2026-04-10_19-49-29/log.txt) | 1.2924 | 72.48M | 4,915 MiB | 478 MiB | MD=2 + residuals; -0.0253 vs L=1 baseline |
| 51  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_20-10-24/log.txt) | 1.2936 | 114.54M | 7,078 MiB | 719 MiB | MD=10 + residuals; -0.0241 vs L=1 baseline |
| 52  | 100 | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_20-51-04/log.txt) | 1.5202 | 166.50M | 8,564 MiB | 1,041 MiB | MLP=100 + MD=10 no residuals; WORSE than either alone |
| 53  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_21-37-45/log.txt) | 1.4391 | 114.54M | 7,082 MiB | 719 MiB | PLE=true; still worse than baseline (MD=10 no resid dominates) |
| 54  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_22-16-17/log.txt) | NaN | 114.28M | 6,279 MiB | — | DB=false; NaN — DB provides critical stability at L=1 |
| 55  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_22-48-20/log.txt) | NaN | 354.92M | — | — | C=1024, MD=10 no resid; NaN step 4500 (LR=0.01) |

### Layers = 1 ablations, part 2 (bigger C)

| Run | C | MLP | MD | lr | block | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|------|-----|-----|------|-------|--------|---------------|--------|------------|----------------|-------|
| 56  | 1024 | 1   | 2  | 0.01 | 512  | [link](logs/wikitext-103_2026-04-10_23-37-19/log.txt) | 1.2153 | 186.90M | 7,100 MiB | 1,113 MiB | C=1024, resid; -0.0771 vs MD=2+resid C=512 |
| 57  | 2048 | 1   | 2  | 0.01 | 512  | [link](logs/wikitext-103_2026-04-11_00-15-26/log.txt) | **1.1660** | 541.57M | 13,453 MiB | 3,211 MiB | **C=2048 approaches L=20!** |
| 58  | 512  | 1   | 2  | 0.01 | 2048 | [link](logs/wikitext-103_2026-04-11_02-00-34/log.txt) | 1.3315 | 76.68M | 14,696 MiB | 505 MiB | Context alone doesn't help |
| 59  | 1024 | 100 | 2  | 0.01 | 2048 | [link](logs/wikitext-103_2026-04-11_02-19-23/log.txt) | 1.2517 | 411.41M | 28,847 MiB | 2,526 MiB | Kitchen sink A; PLE |
| 60  | 2048 | 20  | 2  | 0.01 | 2048 | [link](logs/wikitext-103_2026-04-11_03-24-37/log.txt) | NaN | 768.14M | — | — | — | Kitchen sink B; NaN step 300 |
| 61  | 2048 | 20  | 2  | 0.005| 1024 | [link](logs/wikitext-103_2026-04-11_05-30-57/log.txt) | 1.2024 | 734.56M | 23,346 MiB | 4,358 MiB | LR too low; worse than C=2048 MLP=1 |
| 62  | 2048 | 20  | 1  | 0.01 | 512  | [link](logs/wikitext-103_2026-04-11_08-13-12/log.txt) | **1.1431** | 617.05M | 14,109 MiB | 3,519 MiB | **New L=1 record! Beats L=20 C=512 baseline (1.1751)** |
| 63  | 2048 | 20  | 1  | 0.02 | 512  | [link](logs/wikitext-103_2026-04-11_10-04-14/log.txt) | NaN | 617.05M | — | — | — | NaN step 700 (LR=0.003); lr=0.02 too high for MLP=20 at C=2048 |
| 64  | 4096 | 20  | 1  | 0.01 | 512  | [link](logs/wikitext-103_2026-04-11_11-34-42/log.txt) | — | 2056.18M | ~33 GB | — | — | Early-stopped; diminishing returns vs C=2048 |
| 65  | 2048 | 20  | 2  | 0.01 | 512  | | | ~2.2B | | | | PLE, resid; full recipe at stable LR |
| 66  | 2048 | 20  | 2  | 0.01 | 512  | | | ~2.3B | | | | Same + PKM+FwPKM-16384 |

### Lifting hidden multiplier: L=1, C=2048, MLP=20

Wider predict/update MLPs in the lifting wavelet. Tests whether more expressive local token-to-token interaction improves BPB. Identity init for extra hidden dims (nn.init.eye_) unless noted.

| Run | lifting_hidden_mult | init | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---------------------|------|--------|---------------|--------|------------|----------------|-------|
| 62  | 1 | — | [link](logs/wikitext-103_2026-04-11_08-13-12/log.txt) | 1.1431 | 617.05M | 14,109 MiB | 3,519 MiB | Baseline |
| 67  | 2 | zeros | [link](logs/wikitext-103_2026-04-11_14-23-32/log.txt) | — | 768.08M | — | — | Early-stopped; zero init, no improvement |
| 68  | 2 | eye | [link](logs/wikitext-103_2026-04-11_15-39-24/log.txt) | NaN | 768.08M | — | — | — | Identity init; NaN step 1300 (LR=0.003); signal too strong |
| 69  | 2 | normal(0.01) | [link](logs/wikitext-103_2026-04-11_17-46-16/log.txt) | 1.1428 | 768.08M | 16,989 MiB | 4,383 MiB | Stable but identical to mult=1; local expressivity not the bottleneck |

### Loop iterations (LoopLM): L=1, C=2048, reuse same weights T times

Same layer stack applied T times sequentially. Loss averaged across all iterations. Zero additional parameters — adds compute, not capacity. Inspired by LoopLM (arxiv:2510.25741).

| Run | C | MLP | MD | lr | T | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|------|-----|-----|------|---|--------|---------------|--------|------------|----------------|-------|
| 62  | 2048 | 20  | 1  | 0.01 | 1 | [link](logs/wikitext-103_2026-04-11_08-13-12/log.txt) | 1.1431 | 617.05M | 14,109 MiB | 3,519 MiB | Baseline (no looping) |
| 70  | 2048 | 20  | 1  | 0.01 | 4 | [link](logs/wikitext-103_2026-04-11_13-19-59/log.txt) | — | 617.05M | 31,538 MiB | — | Early-stopped; ~0.04 val_loss gain for 3.5x compute; not worth it |

### Optimal low-layer config: L=2, C=2048, full recipe

L=2, C=2048, MLP=20, PLE, PKM+FwPKM-16384. ~1.18B params, ~21 GB estimated.

| Run | Layers | Epochs | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|--------|---------------|--------|------------|----------------|-------|
|   | 2 | 1 | [link](logs/wikitext-103_2026-04-11_21-09-05/log.txt) | **1.1133** | 1180.28M | 24,643 MiB | 6,733 MiB | **New overall best! Beats L=20 3-epoch baseline (1.1169)** |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-12_00-37-11/log.txt) | **1.0865** | 1180.28M | 24,643 MiB | 6,733 MiB | **New best! No dropout; best val at epoch 3; train/val gap 1.77 by epoch 5** |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-12_17-11-15/log.txt) | **1.0468** | 1180.28M | 24,883 MiB | 6,733 MiB | **1.0x dropout; new best! Val still improving at epoch 5; gap=1.00** |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-13_09-51-55/log.txt) | **1.0319** | 1180.28M | 24,883 MiB | 6,733 MiB | **1.5x dropout; new best! Gap=0.81; val still improving at epoch 5** |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-14_06-41-17/log.txt) | — | 1180.28M | — | — | — | 1.5x dropout + WD=1e-3; early-stopped; WD too aggressive for Adagrad, training stalled |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-14_09-07-12/log.txt) | **1.0247** | 1180.28M | 24,883 MiB | 6,733 MiB | **2.0x dropout; new best! Gap=0.69; val improving all 5 epochs** |

### Grokking experiment: C=128, L=2, tiny core + massive memory

~42M params total with L=2. Tiny mixer core with massive MLP and sparse memory. Two layers give double the wavelet decomposition depth at minimal cost. Testing if extreme epoch count can compensate for tiny width. If a ~42M model achieves comparable BPB to a 1B+ model, it demonstrates that WaveletLM's wavelet mixing can generalize language patterns at minimal scale given sufficient training.

| Run | C | Layers | Epochs | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|-----|--------|--------|--------|---------------|--------|------------|----------------|-------|
|   | 128 | 2 | 16/100 | [link](logs/wikitext-103_2026-04-14_02-31-17/log.txt) | — | ~42M | — | — | Early-stopped; train loss plateaued ~3.77 by epoch 11; gap only 0.10; insufficient capacity for memorization |

### Exponential parametrization: mixer only

Apply exp() reparameterization to GatedSpectralMixer weights only. Tests whether better mixer initialization/gradient dynamics improve BPB and fix NaN at previously-unstable configs.

| Run | Config | exp_param | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|-----------|--------|---------------|--------|------------|----------------|-------|
|   | L=1, C=2048, MLP=20, lr=0.01 | true | [link](logs/wikitext-103_2026-04-15_01-46-27/log.txt) | 1.1431 | 617.05M | 14,109 MiB | 3,519 MiB | Identical to baseline ([Run 62](logs/wikitext-103_2026-04-11_08-13-12/log.txt): 1.1431); no improvement at lr=0.01 |
|   | L=1, C=2048, MLP=20, lr=0.02 | true | [link](logs/wikitext-103_2026-04-15_03-35-54/log.txt) | **1.1347** | 617.05M | 14,109 MiB | 3,519 MiB | **Survived lr=0.02! Previously NaN'd; -0.0084 vs baseline** |
|   | L=20, C=512, MD=5, lr=0.01 | true | [link](logs/wikitext-103_2026-04-15_05-24-40/log.txt) | NaN | 787.24M | — | — | — | NaN at step 3600 again; exp param doesn't fix depth instability |

### Layers > 1: C = 512, epochs = 1, optimal booleans with mlp_expansion

| Run | Layers | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
| 45  | 1  | [link](logs/wikitext-103_2026-04-10_16-31-26/log.txt) | 1.3177 | 67.22M | 4,684 MiB | 448 MiB | 17min train; 46.6 tok/s gen |
|   | 4  | [link](logs/wikitext-103_2026-04-15_10-53-44/log.txt) | 1.2403 | 114.49M | 6,916 MiB | 722 MiB | |
|   | 10  | [link](logs/wikitext-103_2026-04-15_11-34-21/log.txt) | 1.1981 | 209.02M | 11,379 MiB | 1,269 MiB | |
|   | 15  | [link](logs/wikitext-103_2026-04-15_13-01-46/log.txt) | 1.1816 | 287.80M | 15,098 MiB | 1,724 MiB | |
|   | 18  | [link](logs/wikitext-103_2026-04-15_15-10-58/log.txt) | 1.1776 | 335.07M | 17,330 MiB | 1,998 MiB | |
|   | 20 | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | Baseline |
|   | 30 | [link](logs/wikitext-103_2026-04-15_17-46-01/log.txt) | 1.1642 | 524.14M | 26,257 MiB | 3,091 MiB | Slight improvement over L=20; diminishing returns |

### Levels: C = 512, epochs = 1, optimal booleans + mlp_expansion + layers, block_size = 512

| Run | Levels | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
|   | 1 | [link](logs/wikitext-103_2026-04-15_22-17-16/log.txt) | 1.2357 | 114.51M | 7,133 MiB | 738 MiB | 3.2x fewer params, only +0.061 BPB vs baseline |
|   | 2 | [link](logs/wikitext-103_2026-04-16_04-55-49/log.txt) | 1.1998 | 146.02M | 8,594 MiB | 918 MiB | |
|   | 3 | [link](logs/wikitext-103_2026-04-16_05-54-07/log.txt) | 1.1829 | 177.53M | 10,054 MiB | 1,098 MiB | |
|   | 4 | [link](logs/wikitext-103_2026-04-16_07-14-10/log.txt) | 1.1732 | 209.04M | 11,515 MiB | 1,279 MiB | |
|   | 5 | [link](logs/wikitext-103_2026-04-15_23-05-09/log.txt) | 1.1673 | 240.55M | 12,976 MiB | 1,459 MiB | Beats baseline with 34% fewer params! |
|   | 6 | [link](logs/wikitext-103_2026-04-16_08-50-21/log.txt) | 1.1677 | 272.05M | 14,436 MiB | 1,639 MiB | |
|   | 9 | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | Baseline (Run 4; default = log2(block_size=512)) |
|   | 11 | [link](logs/wikitext-103_2026-04-16_01-02-33/log.txt) | 1.1801 | 429.60M | 21,739 MiB | 2,541 MiB | Worse than levels=5; confirms diminishing returns past 5 |

### Low-rank factorization in spectral mixer

| Run | low_rank | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|----------|--------|---------------|--------|------------|----------------|-------|-------|
|   | 0  | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | | Baseline (Run 4; full rank) |
|   | 4  | [link](logs/wikitext-103_2026-04-16_10-43-48/log.txt) | 1.1706 | 367.40M | 18,835 MiB | 2,184 MiB | -0.0045 | Slight improvement! Worth keeping at small cost |
|   | 16 | [link](logs/wikitext-103_2026-04-16_14-02-48/log.txt) | 1.1694 | 369.86M | 18,887 MiB | 2,196 MiB | -0.0057 | Slightly better than rank=4; small marginal gain (-0.0012) |
|   | 64 | | | ~380M | | | | Higher rank; ~13M extra params, still cheap |
|   | 128 | | | ~393M | | | | Contingent on rank=64; only if 64 keeps improving |

### Lifting hidden multiplier

| Run | lifting_hidden_mult | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|---------------------|--------|---------------|--------|------------|----------------|-------|-------|
|   | 1 | [link](#run-4) | 1.1751 | 366.58M | 18,738 MiB | 2,179 MiB | | Baseline (Run 4) |
|   | 2 | [link](logs/wikitext-103_2026-04-16_17-23-50/log.txt) | NaN | 555.51M | — | — | — | NaN at step 2500 (LR=0.0057); wider lifting unstable at L=20. Future stability fixes (e.g., spectral norm on lifting predict/update, or scaled init for hidden dims) may make this viable. |
| N/A | 4 | — | — | — | — | — | — | Cancelled; mult=2 already NaN'd. Revisit with stability fixes. |

### New Baseline

The baseline used for all 1-epoch screening ablations. Combines proven wins (levels=5; low_rank=4) on top of the wide & shallow config. Halves per-epoch runtime vs the previous C=512/L=20 baseline. **PKM and FwPKM are intentionally OFF** during screening — they get re-introduced for the final 5-epoch best-run candidate (saves ~10-15% time and ~150M params per 1-epoch run).

**Config:** L=2, C=2048, MLP=20, levels=5, **PLE**, lr=0.01, low_rank=4

> **Note:** exp_param + lr=0.02 NaN'd at this config ([logs/.../2026-04-17_00-27-55](logs/wikitext-103_2026-04-17_00-27-55/log.txt), step 4000 LR=1.82e-02). Worked at L=1 previously but L=2 doubles residual signal accumulation. Re-test once stable_parametrization (spectral_norm in particular) is validated.

| Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Time | Notes |
|--------|---------------|--------|------------|----------------|------|-------|
| [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1173 | 827.03M | 18,016 MiB | 12,679 MiB | 2.40h | +0.004 BPB vs old baseline (1.1133) but 30% faster and 30% smaller. `per_scale_mixer_widths` was subsequently promoted into the baseline ([logs/.../2026-04-17_14-23-49](logs/wikitext-103_2026-04-17_14-23-49/log.txt), BPB 1.1168), which is the reference used for all Part 1 ablation deltas below. |

### New Baseline Boolean ablations part 1: C=2048, L=2, epochs=1 wide & shallow model

Screening wavelet/mixer and true-feedback augmentations from [`plans/wavelet_and_mixer_augmentations.md`](plans/wavelet_and_mixer_augmentations.md), [`plans/feedback_mechanisms.md`](plans/feedback_mechanisms.md), and [`plans/wavelet_crawl.md`](plans/wavelet_crawl.md) against the **new baseline** (C=2048/L=2/MLP=20/PLE + **levels=5, lr=0.01, low_rank=4**). Folds in proven wins (levels=5 from the L=20 finding; low_rank=4 = -0.0045 BPB at trivial cost). Halves runtime per epoch via fewer wavelet levels. PKM/FwPKM deferred to the final 5-epoch best run; exp_param + lr=0.02 deferred until stable_parametrization is validated. Each feature tested individually at 1 epoch; winners stack into the 5-epoch best-combo run.

| Run | Feature | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|---------|--------|---------------|--------|------------|----------------|-------|-------|
|   | New baseline probe | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1173 | 827.03M | 18,016 MiB | 12,679 MiB | **+0.0040** | 30% faster (2.40h vs 3.42h) and 30% smaller than old baseline for +0.004 BPB. lr=0.01 fallback after lr=0.02+exp_param NaN'd ([link](logs/wikitext-103_2026-04-17_00-27-55/log.txt)); defer until stab_spectral_norm is validated. |
|   | Untied reconstruction | [link](logs/wikitext-103_2026-04-17_06-20-21/log.txt) | 1.1173 | 994.88M | 19,777 MiB | 14,440 MiB | **+0.0000** | Exact tie at +168M (+20%) and +3% time. One wavelet copy suffices. **Dropped** from best-run. |
|   | Cross-scale gating (routing) | [link](logs/wikitext-103_2026-04-17_08-51-44/log.txt) | **1.1166** | 827.03M (+72) | 18,400 MiB | 12,679 MiB | **-0.0007** | Small consistent win; best val -0.0042. Essentially free (+72 params, +2.5% time). **Keep** for best-run. |
|   | Multi-basis lifting (haar+random, attempt 1) | [link](logs/wikitext-103_2026-04-17_11-21-47/log.txt) | NaN | 994.88M | — | — | — | Default Kaiming random init + basis_weights[0]=5.0. **NaN at step 1800, LR=4.10e-03** (well below peak 0.01). |
|   | Multi-basis lifting (haar+random, attempt 2) | [link](logs/wikitext-103_2026-04-17_13-25-19/log.txt) | NaN | 994.88M | — | — | — | Tightened: random init `N(0, 0.01²)` with zero bias, basis_weights[0]=10.0 (softmax ~0.99995 on haar at init). **NaN at step 3400, LR=7.75e-03** — fixes roughly doubled LR tolerance (step 1800→3400, LR 0.0041→0.0078) but didn't fully solve. Deferred until stable_parametrization validated (spectral_norm on mixer is the likely fix). |
|   | Per-scale mixer widths [1×3, 0.5×3] | [link](logs/wikitext-103_2026-04-17_14-23-49/log.txt) | **1.1168** | ~815M | 17,847 MiB | 12,509 MiB | **-0.0005** | Tiny BPB improvement AND **23% faster** (1.85h vs 2.40h), -1% VRAM. Starving fine scales produces less overfitting — best val +0.0054 but final BPB better. **Keep** for best-run. |
|   | Looped blocks (K=8 shared) | [link](logs/wikitext-103_2026-04-17_19-47-45/log.txt) | 1.1129 | ~360M | 29,367 MiB | 8,459 MiB | -0.0039 | **Inefficient** with 3× the training time (5.6h vs 1.85h baseline). Better to just train longer. |
|   | Wavelet crawl (K=3) | [link](logs/wikitext-103_2026-04-18_06-19-52/log.txt) | **1.1131** | ~840M | 18,171 MiB | 12,509 MiB | **-0.0037** | Learned ±1 dilation offsets per level. Third-biggest single-feature win after PSW and CSG. Essentially free (+1.6% time, +2% VRAM). **Keep** for best-run. ⚠️ **Stability caveat:** crawl shifts the predict/update networks' input distribution away from their Haar-init regime; K=5 NaN'd entirely. K=3 was stable in this 1-epoch run, but stacked with other features and/or longer training (5+ epochs, higher LR), there's a non-zero risk the softmax drift compounds. If stacked best-run NaNs, this is a primary suspect — try `stab_spectral_norm` first, then consider stronger init bias (10.0 → 15.0). |
|   | Wavelet crawl (K=5) | [link](logs/wikitext-103_2026-04-18_08-14-44/log.txt) | NaN | ~840M | — | — | — | ±2 dilation offsets. NaN'd step ~4300 (LR=9.81e-03). At higher levels (base ≥ 8), softmax spread of ±2 deviates too far from Haar init, destabilizes predict/update networks. Not rescued — K=3 captures the benefit. |
|   | Shared lifting weights (SLW) | [link](logs/wikitext-103_2026-04-18_09-46-25/log.txt) | 1.1165 | ~770M | 16,886 MiB | 11,388 MiB | **-0.0003** | One shared lifting wavelet instead of per-layer. Essentially tied on BPB; **-5% train VRAM, -9% inference VRAM**. Free memory savings. **Keep** for best-run. |
|   | Lifting linear-only (LLO) | [link](logs/wikitext-103_2026-04-18_11-37-14/log.txt) | 1.1261 | ~790M | 15,766 MiB | 11,388 MiB | **+0.0093** | Replaces predict/update Sequentials (Linear→GELU→Dropout→Linear) with single Linears. Significant BPB regression — the GELU nonlinearity matters. **Drop** despite -11% time / -12% train VRAM savings. |
|   | SLW + LLO combined | [link](logs/wikitext-103_2026-04-18_13-17-47/log.txt) | 1.1229 | ~720M | 15,286 MiB | — | **+0.0061** | LLO dominates the combo. SLW partially offsets but not enough. Fastest run yet (1.59h, -14%) and smallest VRAM, but BPB regression kills it. **Drop.** |

### New Baseline Boolean ablations part 2: Stable parametrization — SKIPPED

Six stability fixes from [`plans/stable_parametrization.md`](plans/stable_parametrization.md) were implemented (spectral norm on mixer, FF √(hidden_dim) scaling, embed √C scaling, proj_out √(C·L) scaling, mixer eps scaling, level-dependent lifting init) but **not evaluated individually.**

**Reason:** the master bundle performed *worse* than the unmodified configs in every rescue test attempted (see Part 3). Most notably, the lr=0.02+exp_param rescue NaN'd at step 3200 with stab, versus step 4000 without it — i.e. the stability features *accelerated* the failure rather than preventing it. This strongly suggests a bug in at least one of the implementations (likely `stab_proj_out_scaling`, whose `1/√(C·L)` formula produces a residual-stream contribution ~15× larger than the original `1e-3` at the new baseline).

With the final 5-epoch best run committed to `lr=0.01` (which is stable without any stab flags), diagnosing the specific bug is no longer release-critical. The code is retained in the repo for future work; the individual-flag ablations are skipped to conserve compute budget. Any researcher wanting to revisit can enable the flags individually via the config.

### New Baseline Boolean ablations part 3: Rescue tests (vs known NaN configs)

If the master flag rescues a previously-NaN config, follow up with per-feature ablation to identify the load-bearing fix.

| Run | NaN config under test | Stab flags | Folder | BPB (sliding) | Params | Notes |
|-----|----------------------|-----------|--------|---------------|--------|-------|
|   | mixer_depth=5 (was [Run 39](#run-39): NaN step 3600) | master | [link](logs/wikitext-103_2026-04-18_14-55-19/log.txt) | **crashed** | ~787M | **Silent crash** before step 100. Likely torch.compile + spectral_norm parametrization on 1000 wrapped mixers (L=20 × S=10 × MD=5) exceeded resources. **Not rescued.** |
|   | lifting_hidden_mult=2 (was [link](logs/wikitext-103_2026-04-16_17-23-50/log.txt): NaN step 2500) | master | [link](logs/wikitext-103_2026-04-18_14-58-10/log.txt) | **NaN step 200** | ~556M | **WORSE than unmodified config** (original NaN'd at step 2500; with stab bundle, NaN at step 200 with val 8.92 at step 100). Stab bundle *actively destabilized* this. Suspect `stab_proj_out_scaling` formula `1/√(C·L)` at L=20 gives ~10× stronger proj_out than original 1e-3, amplifying residual-stream signal. **Not rescued.** |
|   | C=2048, lr=0.02 (was [Run 63](#run-63): NaN step 700) | master | | | ~617M | Width × LR NaN; survived only with exp_param previously |
|   | new baseline + lr=0.02 + exp_param (NaN'd [link](logs/wikitext-103_2026-04-17_00-27-55/log.txt) step 4000, LR=1.82e-02) | master | | | 827.03M | If rescued, unlocks lr=0.02 at the new baseline — likely the biggest latent BPB win (L=1 previously showed -0.0084 at lr=0.02+exp_param). Primary target for `stab_spectral_norm`. |

### Block size (context window) — at new baseline

`levels=5` from the new baseline supports any `block_size >= 32`, so no `levels` adjustment is needed when changing context length.

| Run | block_size | Folder | BPB (sliding) | Params | Train VRAM | Time | Delta | Notes |
|-----|------------|--------|---------------|--------|------------|------|-------|-------|
| ~~64~~ | — | **Cancelled** | — | — | — | — | BS=128 came in worse than BS=256, so the floor is at 256. Not worth probing smaller. |
|   | 128  | [link](logs/wikitext-103_2026-04-19_07-00-01/log.txt) | 1.1075 | ~840M | 16,573 MiB | 2.74h | -0.0093 | Better than baseline (-0.0093) but **worse than BS=256** (+0.0047). The "more updates" trend plateaus between 256 and 128. BS=256 stays the winner. |
|   | **256**  | [link](logs/wikitext-103_2026-04-18_16-12-23/log.txt) | **1.1028** | ~840M | 16,680 MiB | 2.16h | **-0.0140** | **Biggest single-feature win so far.** ~2× gradient updates per epoch since dataset splits into more blocks. With levels=5 (max dilation 2^4=16), 256-token context is still ample. |
|   | **256 + grad_accum=1** | [link](logs/wikitext-103_2026-04-19_09-46-39/log.txt) | **1.0966** | ~840M | 16,680 MiB | 2.87h | **-0.0202** | **Stacks near-linearly!** Individual wins: BS=256 (-0.0140) + GA=1 (-0.0076) = -0.0216 linear prediction; actual -0.0202 (~94% of linear). First 1-epoch result below 1.10. Critical adoption decision for 5-epoch best-run (see "Best run candidate" section). |
|   | 512  | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1168 | ~840M | 18,016 MiB | 1.85h | | Baseline (new baseline probe) |
|   | 1024 | [link](logs/wikitext-103_2026-04-18_18-23-56/log.txt) | NaN (3.5220) | ~840M | 24,091 MiB | 1.53h | — | ❌ NaN'd. Effective batch reached 8192 tokens, crossed AMP/fp16 overflow threshold. Would need MBS reduction to recover. |

### Grad accum — at new baseline

| Run | grad_accum | Effective batch | Folder | BPB (sliding) | Params | Time | Delta | Notes |
|-----|-----------|----------------|--------|---------------|--------|------|-------|-------|
|   | **1** | **8**  | [link](logs/wikitext-103_2026-04-18_19-57-53/log.txt) | **1.1092** | ~840M | 2.39h | **-0.0076** | **Solid win.** Half the effective batch → 2× gradient updates per epoch at cost of +29% wall-clock time. |
|   | 2 | 16 | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1168 | ~840M | 1.85h | | Baseline (new baseline probe) |
|   | 4 | 32 | [link](logs/wikitext-103_2026-04-18_22-23-30/log.txt) | NaN (3.5221) | ~840M | 1.39h | — | ❌ NaN'd. Effective batch=32 too aggressive at lr=0.01. Same theme as block_size=1024 — larger-batch NaN. |

### Warmup fraction — at new baseline

| Run | warmup_fraction | Folder | BPB (sliding) | Params | Time | Delta | Notes |
|-----|-----------------|--------|---------------|--------|------|-------|-------|
|   | 0.1 | [link](logs/wikitext-103_2026-04-18_23-49-03/log.txt) | NaN (3.5220) | ~840M | 1.61h | — | ❌ NaN'd. LR ramped faster than the model could absorb. |
|   | 0.3 | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1168 | ~840M | 1.85h | | Baseline (new baseline probe) |
|   | 0.5 | [link](logs/wikitext-103_2026-04-19_01-27-15/log.txt) | 1.1210 | ~840M | 1.79h | +0.0042 | Slightly worse — over-cautious LR ramp means less effective training at peak. 0.3 is the sweet spot. |

### Grad clip — at new baseline

| Run | grad_clip | Folder | BPB (sliding) | Params | Time | Delta | Notes |
|-----|-----------|--------|---------------|--------|------|-------|-------|
|   | 0.5 | [link](logs/wikitext-103_2026-04-19_03-16-59/log.txt) | 1.1167 | ~840M | 1.82h | -0.0001 | Essentially tied with baseline. Tighter clipping neither helps nor hurts at lr=0.01. |
|   | 1.0 | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1168 | ~840M | 1.85h | | Baseline (new baseline probe) |
|   | 2.0 | [link](logs/wikitext-103_2026-04-19_05-08-30/log.txt) | 1.1244 | ~840M | 1.81h | **+0.0076** | Looser clipping actively hurts — larger allowed gradient norms let occasional spikes corrupt the optimizer state. Confirms `grad_clip=1.0` is optimal, not just conservative. |

### Best run candidate: L=2, C=2048, lr=0.01, 2.0x dropout, 5 epochs

Consolidated 5-epoch run stacking every proven win from the 1-epoch sweep. lr=0.02 + exp_param was dropped (NaN at L=2); the L=10 variant was dropped in favor of L=2, which produced the best 1-epoch BPB at this width. Baseline for comparison: L=2/C=2048/MLP=20/PLE/PKM+FwPKM-16384/lr=0.01, levels=9, 5ep, 2.0x dropout = BPB 1.0247 (see [training log](logs/wikitext-103_2026-04-14_09-07-12/log.txt)).

**Eval interval:** bumped from 100 → 250 from this run forward (~1,000 evals over 5 epochs, ~200 per epoch). Dense enough to resolve warmup / plateaus / tail without eval overhead from the ~50k steps/epoch at `block_size=256`. Drop back to 100 if curves come out noisy or early-stop signal is missed.

Proven wins stacked here (1-epoch deltas vs new baseline BPB 1.1168):
- `block_size=256` + `grad_accum=1`: -0.0202 (~94% linear stacking of the two individual wins)
- `wavelet_crawl` K=3: -0.0037
- `cross_scale_gating`: -0.0007
- `shared_lifting_weights`: -0.0003 + ~1/2 lifting VRAM
- `per_scale_mixer_widths` [1,1,1,0.5,0.5,0.5]: promoted into new baseline (-23% time, BPB tie)
- `low_rank=4`: baseline value
- `levels=5`: baseline value (was tested as a reduction from 9; stayed as baseline)

| Run | Config | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Time | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|------|-------|
|   | L=2, C=2048, MLP=20, PLE, PKM+FwPKM-16384, lr=0.01, block_size=256, grad_accum=1, levels=5, low_rank=4, **per_scale_mixer_widths**=[1,1,1,0.5,0.5,0.5], **cross_scale_gating**, **wavelet_crawl K=3**, **shared_lifting_weights**, 5ep, 2.0x dropout (emb=0.2, proj=0.1, mixer=0.1, mlp=0.1, lm=0.24) | [link](logs/wikitext-103_2026-04-19_13-16-24/log.txt) | **1.0219** | 882.51M | 18,235 MiB | 4,915 MiB | 15.79h | Non-overlapping BPB 1.0404. Only -0.0028 vs the levels=9 baseline (1.0247) — well short of the projected 0.99–1.02. |

**Why the projection overshot.** The 94%-linear-stacking estimate was built from 1-epoch deltas. Several of those wins (especially `block_size=256` at -0.0140) come from "more gradient updates per epoch" — but at 5 epochs, the levels=9 baseline has also had time to converge. Short-training advantages compress as training continues; additivity at 1 epoch ≠ additivity at 5.

**Overfit signal.** Final train loss 2.6492 / val loss 3.1925 → **~0.78 bit/token train-val gap**. Best val (3.1728) occurred mid-epoch-5; end-of-run val regressed slightly, indicating the model started drifting past its peak. 2.0× dropout is insufficient regularization for this capacity at 5 epochs, and the gap will widen further at 10 epochs if dropout isn't scaled.

**Dropped features (ablation record).** Untied reconstruction (1-epoch tie at +168M params, +3% time), multi-basis lifting (NaN even with tightened init), exp_param at lr=0.02 (NaN at L=2), iterative refinement, cross-time feedback, stability-parametrization bundle.

### 3-seed 10-epoch variance study: L=2, C=2048, 2.5x dropout

Next up. Same config as the 5-epoch run above, doubled to 10 epochs, with dropout scaled to 2.5× (emb=0.25, proj=0.125, mixer=0.125, mlp=0.125, lm=0.25) to counter the overfit signal observed at 2.0×. Three seeds (1337, 42, 7) for mean ± std reporting. Eval interval stays at 250. Estimated ~28h per seed, ~84h total on 5090. Peak training VRAM ~18 GB and peak generation VRAM ~5 GB from the 5-epoch run suggest PTQ work can proceed in tandem on the completed 5-epoch checkpoint.

| Run | Seed | Folder | BPB (sliding) | Train/Val loss | Params | Time | Notes |
|-----|------|--------|---------------|----------------|--------|------|-------|
|   | 1337 | | | | 882.51M | ~28h | Primary |
|   | 42   | | | | 882.51M | ~28h | |
|   | 7    | | | | 882.51M | ~28h | |

Mean BPB: _ ± _.

### Post-training quantization (PTQ): inference-only, applied to best checkpoint

Per-scale mixed precision leveraging WaveletLM's wavelet decomposition. Coarse scales (high-level semantics) get more bits; fine scales (local detail) tolerate aggressive quantization. All runs use the same trained checkpoint — no retraining needed.

**Baseline checkpoint:** the 5-epoch best run above (L=2, C=2048, MLP=20, PLE, PKM+FwPKM-16384, lr=0.01, block_size=256, grad_accum=1, levels=5, low_rank=4, per_scale_mixer_widths, cross_scale_gating, wavelet_crawl K=3, shared_lifting_weights, 2.0x dropout). Previous PTQ baseline was the levels=9 checkpoint at BPB 1.0247; switching to the consolidated best-run checkpoint once it completes.

> **Note on compression ratios.** The current impl stores sub-8-bit weights as int8 (one value per byte) — no bit-packing yet. So every "enabled" variant produces the same physical size per quantized component regardless of bit-width: uniform 8-bit and uniform 4-bit both give 1.95× compression, and mixed-precision only varies ratio by **which** components are kept at fp16. Proper bit-packing would multiply the compression wins proportionally (uniform 4-bit → ~4×, uniform 2-bit → ~8×).

### PTQ: Uniform quantization (all components same bits)

Baseline: 5-epoch best run, BPB 1.0219 (sliding). Peak inference VRAM at fp16: 4,918 MiB.

| Run | Bits | Folder | BPB (sliding) | Δ BPB | Size (MiB) | Inf. VRAM | tok/s | Notes |
|-----|------|--------|---------------|-------|------------|-----------|-------|-------|
|   | 16 (baseline) | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/01_baseline_fp16.log) | 1.0219 | — | 1,684* | 4,918 MiB | 27.1 | fp16 reference |
|   | 8 uniform | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/02_uniform_8bit.log) | **1.0220** | **+0.0001** | 1,726 | 4,408 MiB | 23.8 | Near-lossless drop-in. Lifting kept at 16. |
|   | 4 uniform | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/03_uniform_4bit.log) | 1.1948 | +0.1729 | 1,726 | 4,408 MiB | 23.1 | **Catastrophic** — coarse-mixer-at-4 is the cliff. |

\*fp16 reference size = 882.51M × 2 bytes ≈ 1,684 MiB; the quantized sizes above exclude the dequantize-to-fp16 working buffer.

### PTQ: Per-scale mixed precision quantization

Mixed precision by component, leveraging the wavelet-scale structure (coarse = semantics, fine = detail).

| Run | Mixer c/m/f | MLP | Lifting | Emb | Folder | BPB (sliding) | Δ BPB | Size (MiB) | Inf. VRAM | tok/s | Notes |
|-----|-------------|-----|---------|-----|--------|---------------|-------|------------|-----------|-------|-------|
|   | 8/4/2 | 4 | 16 | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/04_mixed_default.log) | 1.0302 | +0.0083 | 1,726 | 4,408 | 24.2 | Default mixed config |
|   | **8/8/4** | 4 | 16 | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/05_mixed_conservative.log) | **1.0246** | **+0.0027** | 1,726 | 4,408 | 23.9 | **Conservative fine scales — best quality-preserving mixed variant** |
|   | 8/4/2 | 8 | 16 | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/06_mixed_higher_mlp.log) | 1.0268 | +0.0049 | 1,726 | 4,408 | 24.4 | Higher MLP precision |
|   | 8/4/2 | 4 | 8 | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/07_mixed_quant_lifting.log) | 1.0308 | +0.0089 | 1,487 | 4,408 | 22.8 | Quantize lifting too; 2.26× ratio |
|   | 4/4/2 | 4 | 8 | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/08_mixed_aggressive.log) | 1.1958 | +0.1739 | 1,487 | 4,408 | 21.5 | **Catastrophic** — coarse mixer at 4-bit breaks it |
|   | 8/4/2 | 4 | 16 | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/09_mixed_aggressive_emb.log) | 1.0339 | +0.0120 | 1,726 | 4,408 | 24.5 | Aggressive embedding |

### PTQ: Component isolation (quantize one component; keep the rest at fp16)

Measures per-component sensitivity. Surfaces the key finding: **coarse mixer scales are the 4-bit cliff**; MLP, embedding, and lifting all tolerate 4-bit with negligible BPB loss.

| Run | Component | Bits | Folder | BPB (sliding) | Δ BPB | Size (MiB) | Inf. VRAM | tok/s | Notes |
|-----|-----------|------|--------|---------------|-------|------------|-----------|-------|-------|
|   | Mixer only | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/10_mixer_only_8.log) | 1.0220 | +0.0001 | 3,371 | 4,499 | 26.2 | Mixer at 8-bit: safe |
|   | Mixer only | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/11_mixer_only_4.log) | **1.1832** | **+0.1613** | 3,371 | 4,499 | 25.2 | **Mixer at 4-bit alone breaks BPB** |
|   | MLP only | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/12_mlp_only_8.log) | 1.0219 | 0.0000 | 2,567 | 4,251 | 25.5 | MLP at 8-bit: lossless |
|   | MLP only | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/13_mlp_only_4.log) | 1.0245 | +0.0026 | 2,567 | 4,251 | 25.6 | MLP tolerates 4-bit — biggest bit-packing opportunity (76% of layer params) |
|   | Embedding only | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/14_embedding_only_8.log) | 1.0219 | 0.0000 | 3,034 | 4,892 | 27.8 | Embedding at 8-bit: lossless |
|   | Embedding only | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/15_embedding_only_4.log) | 1.0233 | +0.0014 | 3,034 | 4,892 | 26.0 | Embedding tolerates 4-bit |
|   | Lifting only | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/16_lifting_only_8.log) | 1.0223 | +0.0004 | 3,383 | 4,519 | 26.3 | Lifting at 8-bit: near-lossless |

### Best PTQ combination

With the current (no-bit-packing) implementation, uniform 8-bit is the clear winner: near-lossless BPB and the maximum achievable compression. **Conservative mixed (8/8/4, MLP=4, lift=16, emb=8)** is the best candidate *for future bit-packing* — same compression today, but would drop below 0.5 GB with proper packing at +0.0027 BPB.

| Config | Mixer c/m/f | MLP | Lifting | Emb | BPB (sliding) | Δ BPB | Size (MiB) | Inf. VRAM | tok/s | Notes |
|--------|-------------|-----|---------|-----|---------------|-------|------------|-----------|-------|-------|
| Safest shippable | 8/8/8 | 8 | 16 | 8 | 1.0220 | +0.0001 | 1,726 | 4,408 | 23.8 | Uniform 8-bit — ship-ready today. |
| Bit-packing-ready | 8/8/4 | 4 | 16 | 8 | 1.0246 | +0.0027 | 1,726 | 4,408 | 23.9 | Same physical size today; projected ~0.5 GB with packing. |

### PTQ sweep summary

All 17 variants, full table. Logs folder: [`logs/wikitext-103_2026-04-19_13-16-24/ptq/`](logs/wikitext-103_2026-04-19_13-16-24/ptq/).

| Variant | BPB (sl) | BPB (nov) | Size (MiB) | Compress | tok/s | Peak VRAM |
|---------|----------|-----------|------------|----------|-------|-----------|
| 01_baseline_fp16 | 1.0219 | 1.0404 | — | fp16 | 27.1 | 4,918 |
| 02_uniform_8bit | 1.0220 | 1.0405 | 1,726.4 | 1.95× | 23.8 | 4,408 |
| 03_uniform_4bit | 1.1948 | 1.2047 | 1,726.4 | 1.95× | 23.1 | 4,408 |
| 04_mixed_default | 1.0302 | 1.0494 | 1,726.4 | 1.95× | 24.2 | 4,408 |
| 05_mixed_conservative | 1.0246 | 1.0435 | 1,726.4 | 1.95× | 23.9 | 4,408 |
| 06_mixed_higher_mlp | 1.0268 | 1.0456 | 1,726.4 | 1.95× | 24.4 | 4,408 |
| 07_mixed_quant_lifting | 1.0308 | 1.0501 | 1,486.5 | 2.26× | 22.8 | 4,408 |
| 08_mixed_aggressive | 1.1958 | 1.2056 | 1,486.5 | 2.26× | 21.5 | 4,408 |
| 09_mixed_aggressive_emb | 1.0339 | 1.0532 | 1,726.4 | 1.95× | 24.5 | 4,408 |
| 10_mixer_only_8 | 1.0220 | 1.0405 | 3,370.7 | 1.00× | 26.2 | 4,499 |
| 11_mixer_only_4 | 1.1832 | 1.1930 | 3,370.7 | 1.00× | 25.2 | 4,499 |
| 12_mlp_only_8 | 1.0219 | 1.0404 | 2,566.9 | 1.31× | 25.5 | 4,251 |
| 13_mlp_only_4 | 1.0245 | 1.0434 | 2,566.9 | 1.31× | 25.6 | 4,251 |
| 14_embedding_only_8 | 1.0219 | 1.0404 | 3,033.8 | 1.11× | 27.8 | 4,892 |
| 15_embedding_only_4 | 1.0233 | 1.0420 | 3,033.8 | 1.11× | 26.0 | 4,892 |
| 16_lifting_only_8 | 1.0223 | 1.0409 | 3,382.7 | 1.00× | 26.3 | 4,519 |

**Key findings:**
- **Coarse mixer scales (0-2) are the 4-bit cliff.** Every catastrophic variant (#03, #08, #11) has coarse mixer at 4-bit; every survivable variant keeps it ≥ 8-bit. Mid and fine scales tolerate 4-bit and 2-bit cleanly.
- **MLP, embedding, lifting are highly tolerant.** Each individually survives 4-bit with ≤0.003 BPB degradation. MLP is 76% of layer params — biggest bit-packing opportunity.
- **Generation is slightly slower under quantization** (21-25 vs 27 tok/s baseline) because we pay dequantize cost on every forward pass without a matching bandwidth win. Flips once packed kernels land.
- **VRAM savings are modest** (~10%) for the same reason — dequantize-to-fp16 on forward briefly holds the full fp16 weight.

### Planned: model comparisons (WikiText-103, matched compute)

All models use the same GPT-2 tokenizer (tiktoken, 50,257 vocab), same dataset preprocessing, and same sliding window evaluation methodology. Competitors use all available optimizations (Flash Attention, torch.compile, KV cache, etc.) to ensure the comparison reflects each architecture's best-case performance.

| Model | Type | Params | BPB (sliding) | Train tok/s | Gen tok/s | Training time | Optimizations | Notes |
|-------|------|--------|---------------|-------------|-----------|---------------|---------------|-------|
| WaveletLM | Wavelet mixer | | | | | | torch.compile, fp16 | Best config from sweeps |
| GPT-2 | Transformer | | | | | | Flash Attention, KV cache, TurboQuant, torch.compile, fp16 | Matched compute |
| Mamba | SSM | | | | | | Mamba CUDA kernels, TurboQuant, torch.compile, fp16 | Matched compute |
| RWKV | Linear attention | | | | | | Custom CUDA kernels, TurboQuant, torch.compile, fp16 | Matched compute |

### Planned: dataset comparisons (B200, 20+ epochs, max EBS)

Tests whether WaveletLM's wavelet-mixing inductive bias keeps pace with attention when data isn't the bottleneck — the 882M model on WikiText-103's ~0.5 GB is data-saturated (each token seen ~5×). Each run targets a ~10–50 GB dataset, training on a B200 (or whichever VM is most efficient) for 20+ epochs with effective batch size (MBS × GA) pushed as high as stability allows to fully utilize HBM3e.

**Stability notes:**
- Large EBS is NaN-prone at `lr=0.01` on 5090 runs (block_size=1024 NaN'd at EBS=8192 tokens; grad_accum=4 NaN'd). At B200 scale, expect to need careful LR-batch retuning and likely `stable_parametrization: true`.
- WikiText-103 stays included so the existing BPB 1.0247 baseline anchors comparisons.

| Dataset | HuggingFace ID | Approx size | Domain | Folder | BPB (sliding) | Notes |
|---------|---------------|-------------|--------|--------|---------------|-------|
| WikiText-103 | `wikitext-103` | ~0.5 GB | Wikipedia | | | Anchor benchmark — compare against 5090 headline |
| PG-19 | `pg19` | ~11 GB | Books (long-range coherence) | | | Tests long-document behavior |
| Pile ArXiv | `pile-arxiv` | ~60 GB | Academic/technical | | | Highly structured, formulaic |
| BookCorpusOpen | `bookcorpusopen` | ~6 GB | Fiction | | | Narrative prose |
| OpenWebText | `openwebtext` | ~38 GB | Web text | | | Broadest domain; closest to modern LM regime |

### Post-release: scaled-up B200 configuration

Budget-unconstrained follow-up to the 5090-bound headline run. Specific config chosen after the 5-epoch / 10-epoch 5090 sweeps complete and their BPB/VRAM curves identify the highest-leverage scaling axis. All four levers below are in play; exact numbers pending.

| Lever | 5090 headline | B200 target | Rationale |
|-------|--------------|-------------|-----------|
| `C` (mixer width) | 2048 | 4096 | Doubles mixer expressivity; FHT stays O(C log C) |
| `layers` | 2 | 4–8 | Depth past L=2 unexplored at C=2048 due to 5090 VRAM |
| `mlp_expansion` | 20 | 50–200 | Monotonic BPB contributor; primary knowledge-storage lever |
| `pkm_num_keys` / `fwpkm_num_keys` | 16384 | 65536 | 4× sparse memory capacity for long-tail patterns |
| `shared_lifting_weights` | true | true | Keeps lifting memory flat with layer scaling |
| Dropout | 2.0× (possibly 2.5×) | TBD | Likely scales further with added capacity |
| `amp_dtype` | fp16 | **fp8 (E4M3/E5M2)** | Blackwell native FP8 tensor cores: ~2× training throughput, ~30–40% memory savings. Requires Transformer Engine (or torchao) + per-tensor dynamic scaling to manage FP8's narrow range. Note: bf16 was previously tried as a wider-range alternative and regressed (slower + more NaNs than fp16), so the stability recipe is non-trivial. FP8 needs its own tuned policy, not a drop-in dtype swap. |

Two training targets are planned at this scale:

- **WikiText-103 only** — direct apples-to-apples comparison against the 5090 headline run and against prior same-dataset baselines (Transformer-XL, S4).
- **Multi-dataset** — training across PG-19, Pile-ArXiv, BookCorpusOpen, TinyStories, OpenWebText, and WikiText-103. Establishes WaveletLM's behavior as a general-purpose language model across domains rather than a single-benchmark result.

| Run | Training target | Config | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|----------------|--------|--------|---------------|--------|------------|----------------|-------|
|   | WikiText-103 only | TBD (target ~10–15B) | | | ~10–15B | ≤192 GB (B200) | ~24 GB (fp16, single 4090) | Headline scale-up, direct comparison to 5090 run |
|   | Multi-dataset mix | TBD (target ~10–15B) | | per-dataset | ~10–15B | ≤192 GB (B200) | ~24 GB (fp16, single 4090) | General-purpose LM behavior across domains |
|   | Either + PTQ (per-scale 8/4/2-bit) | inference-only | | matches source | matches source | — | <8 GB | Enables consumer-GPU inference on either checkpoint |

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
    "decompose_bypass": true,
    "decompose_bypass_cross_window": true,
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

**Description:** C = 512, epochs = 1, `decompose_bypass` = false. Boolean ablation.

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

**Description:** C = 512, epochs = 1, `decompose_bypass_cross_window` = false. Boolean ablation.

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

