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
34. [3-seed variance study: L=2, C=2048, 2.0x dropout, 5 epochs](#3-seed-variance-study-l2-c2048-20x-dropout-5-epochs)
35. [Weight decay spot-check (low values): 5 epochs, best architecture](#weight-decay-spot-check-low-values-5-epochs-best-architecture)
36. [Decompose-bypass data-dependent EMA probe](#decompose-bypass-data-dependent-ema-probe)
37. [PG-19 pre-release benchmark: best seed, 1 epoch](#pg-19-pre-release-benchmark-best-seed-1-epoch)
38. [Post-training quantization (PTQ)](#post-training-quantization-ptq-inference-only-applied-to-best-checkpoint)
39. [PTQ: Uniform quantization](#ptq-uniform-quantization-all-components-same-bits)
40. [PTQ: Per-scale mixed precision](#ptq-per-scale-mixed-precision-quantization)
41. [PTQ: Component isolation](#ptq-component-isolation-quantize-one-component-keep-the-rest-at-fp16)
42. [Best PTQ combination](#best-ptq-combination)
43. [PTQ sweep summary](#ptq-sweep-summary)

**Post-release**

44. [Layers = 1 ablation series: modern stack](#layers--1-ablation-series-modern-stack-c2048-mlp20-full-feature-set)
45. [Post-release: bit-packed PTQ kernels](#post-release-bit-packed-ptq-kernels)
46. [Planned: model comparisons (WikiText-103, matched compute)](#planned-model-comparisons-wikitext-103-matched-compute)
47. [Planned: dataset comparisons (B200, 20+ epochs, max EBS)](#planned-dataset-comparisons-b200-20-epochs-max-ebs)
48. [Post-release: scaled-up B200 configuration](#post-release-scaled-up-b200-configuration)

---

### Width (C): epochs = 1, mlp_expansion = 1

| Run | C | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---|--------|---------------|--------|------------|----------------|-------|
| 1   | 64   | [link](logs/wikitext-103_2026-04-02_11-53-11/log.txt) | 1.5178 | 11.42M | 5,110 MiB | 169 MiB (PyTorch-allocated) | Pipeline test; LR=0.02 |
| 2   | 128  | [link](logs/wikitext-103_2026-04-03_10-22-17/log.txt) | 1.4766 | 32.66M | 6,360 MiB | 285 MiB (PyTorch-allocated) | LR=0.02 |
| 3   | 256  | [link](logs/wikitext-103_2026-04-03_07-37-34/log.txt) | 1.2938 | 98.63M | 9,958 MiB | 689 MiB (PyTorch-allocated) | LR=0.02 |
| 4   | 512  | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | LR=0.01 |
| 5   | 1024 | [link](logs/wikitext-103_2026-04-02_22-51-10/log.txt) | 1.1423 | 1362.31M | 42,643 MiB | 7,890 MiB (PyTorch-allocated) | LR=0.005 |

### Epochs: C = 512, mlp_expansion = 1

| Run | Epochs | Folder | BPB (sliding) | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|------------|----------------|-------|
| 4   | 1  | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | Shared with width sweep (Run 4) |
| 6   | 3  | [link](logs/wikitext-103_2026-04-03_14-34-51/log.txt) | 1.1167 | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | Ablation baseline |
| 7   | 5  | [link](logs/wikitext-103_2026-04-03_22-59-40/log.txt) | 1.1233 | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | Overfit; best val at epoch 4, not 5. No dropout. |

### Boolean ablations part 1: C = 512, epochs = 1, mlp_expansion = 1

| Run | Setting | Value | Folder | BPB (sliding) | Params | Time | Train VRAM | Inference VRAM | Delta |
|-----|---------|-------|--------|---------------|--------|------|------------|----------------|-------|
| 4   | Baseline (all standard) | | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 2.78h | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | |
| 8   | `decompose_bypass` | false | [link](logs/wikitext-103_2026-04-04_13-38-15/log.txt) | 1.1734 | 361.23M | 2.83h | 17,764 MiB | 2,147 MiB (PyTorch-allocated) | -0.0014 |
| 9   | `decompose_bypass_cross_window` | false | [link](logs/wikitext-103_2026-04-04_16-31-53/log.txt) | 1.1745 | 366.58M | 2.62h | 18,809 MiB | 2,178 MiB (PyTorch-allocated) | -0.0003 |
| 10  | `learned_residual` | false | [link](logs/wikitext-103_2026-04-04_19-12-31/log.txt) | 1.1810 | 366.58M | 2.66h | 18,818 MiB | 2,178 MiB (PyTorch-allocated) | +0.0063 |
| 11  | `use_mixer_gate` | false | [link](logs/wikitext-103_2026-04-04_21-55-38/log.txt) | 1.2006 | 314.15M | 2.45h | 16,518 MiB | 1,878 MiB (PyTorch-allocated) | +0.0258 |
| 12  | `skip_proj_out` | true | [link](logs/wikitext-103_2026-04-05_00-26-06/log.txt) | 1.1876 | 361.33M | 2.60h | 18,668 MiB | 2,148 MiB (PyTorch-allocated) | +0.0084 |
| 13  | `shared_lifting_weights` | true | [link](logs/wikitext-103_2026-04-05_03-05-29/log.txt) | 1.1856 | 186.92M | 2.42h | 16,762 MiB | 1,150 MiB (PyTorch-allocated) | +0.0108 |
| 14  | `lifting_linear_only` | true | [link](logs/wikitext-103_2026-04-05_05-33-33/log.txt) | 1.1891 | 272.02M | 1.79h | 13,236 MiB | 1,637 MiB (PyTorch-allocated) | +0.0141 |
| 15  | `tie_embedding_to_lm_head` | true | [link](logs/wikitext-103_2026-04-05_07-23-27/log.txt) | 1.1811 | 340.85M | 2.77h | 18,523 MiB | 2,080 MiB (PyTorch-allocated) | +0.0064 |

### Boolean ablations part 2: C = 512, epochs = 3, mlp_expansion = 1

Baseline: Run 6 (3 epochs, all defaults) = BPB 1.1169

| Run | Setting | Value | Folder | BPB (sliding) | Params | Time | Train VRAM | Inference VRAM | Delta (3ep) | Delta (1ep) | Notes |
|-----|---------|-------|--------|---------------|--------|------|------------|----------------|-------------|-------------|-------|
| 6   | Baseline (3ep) | | [link](logs/wikitext-103_2026-04-03_14-34-51/log.txt) | 1.1167 | 366.58M | 8.34h | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | | | |
| 16  | `decompose_bypass` | false | [link](logs/wikitext-103_2026-04-05_16-08-09/log.txt) | 1.1173 | 361.23M | 7.60h | 17,764 MiB | 2,147 MiB (PyTorch-allocated) | +0.0010 | -0.0014 | DB true is better. |
| 17  | `lifting_linear_only` | true | [link](logs/wikitext-103_2026-04-05_23-45-27/log.txt) | 1.1341 | 272.02M | 5.17h | 13,236 MiB | 1,637 MiB (PyTorch-allocated) | +0.0168 | +0.0141 | LLO true performs worse with more epochs. |
| 18  | `shared_lifting_weights` | true | [link](logs/wikitext-103_2026-04-06_04-59-03/log.txt) | 1.1253 | 186.92M | 7.20h | 16,762 MiB | 1,150 MiB (PyTorch-allocated) | +0.0089 | +0.0108 | SLW true performs better with more epochs. |

> **Notes:** With epochs >= 3 `decompose_bypass=true` is best, as is `lifting_linear_only=false`. It's likely that at much higher epochs, `shared_lifting_weights=true` is best; the parameters, run time, and extreme VRAM savings (~1/2 at 3 epochs!) it saves could be better used elsewhere (larger MLP, more epochs, larger micro batch size, etc.).

### Best Boolean ablations combination: C=512, epochs = 3, mlp_expansion = 1

All defaults are optimal with epochs = 3. No boolean change improved BPB. Note that shared lifting wavelets may contribute negligibly at much higher epochs, however. See previous note.

| Run | Folder | BPB (sliding) | Params | Training time | VRAM (Train/Inf) | Notes |
|-----|--------|---------------|--------|---------------|------------------|-------|
| 6   | [link](logs/wikitext-103_2026-04-03_14-34-51/log.txt) | 1.1167 | 366.58M | 8.34h | 18,738 / 2,179 MiB (PyTorch-allocated) | Baseline IS the best combo |

### MLP expansion: C = 512, epochs = 1, optimal booleans

| Run | mlp_expansion | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---------------|--------|---------------|--------|------------|----------------|-------|
| 4   | 1  | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | Baseline (Run 4) |
| 19  | 2  | [link](logs/wikitext-103_2026-04-06_15-09-30/log.txt) | 1.1674 | 377.08M | 19,118 MiB | 2,239 MiB (PyTorch-allocated) | -0.0073 |
| 20  | 10 | [link](logs/wikitext-103_2026-04-06_17-51-28/log.txt) | 1.1529 | 461.04M | 21,519 MiB | 2,719 MiB (PyTorch-allocated) | -0.0219 |
| 21  | 20 | [link](logs/wikitext-103_2026-04-06_20-38-06/log.txt) | 1.1483 | 566.00M | 24,520 MiB | 3,320 MiB (PyTorch-allocated) | -0.0264 |
| 22  | 50 | [link](logs/wikitext-103_2026-04-06_23-32-36/log.txt) | 1.1406 | 880.88M | 33,524 MiB | 5,121 MiB (PyTorch-allocated) | -0.0342 |

### Memory: C = 512, epochs = 1, optimal booleans + mlp_expansion

| Run | PKM | FwPKM | PKM num keys | FwPKM num keys | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|-----|-------|--------------|----------------|--------|---------------|--------|------------|----------------|-------|
| 4   | off | off | | | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | Baseline (Run 4, MLP only) |
| 23  | on  | off | 529 | | [link](logs/wikitext-103_2026-04-07_03-43-32/log.txt) | 1.1726 | 377.48M | 19,293 MiB | 2,230 MiB (PyTorch-allocated) | PKM default; -0.0022 vs baseline |
| 24  | on  | off | 16384 | | [link](logs/wikitext-103_2026-04-07_06-37-05/log.txt) | 1.1622 | 540.91M | 21,177 MiB | 2,856 MiB (PyTorch-allocated) | PKM large; -0.0126 vs baseline, +174M params |
| 25  | off | on  | | 529 | [link](logs/wikitext-103_2026-04-07_09-40-03/log.txt) | 1.1722 | 377.48M | 19,474 MiB | 2,251 MiB (PyTorch-allocated) | FwPKM default; -0.0025 vs baseline |
| 26  | on  | on  | 529 | 529 | [link](logs/wikitext-103_2026-04-07_12-58-11/log.txt) | 1.1710 | 388.37M | 19,949 MiB | 2,303 MiB (PyTorch-allocated) | PKM+FwPKM default; -0.0038 vs baseline |
| 27  | on  | on  | 16384 | 16384 | [link](logs/wikitext-103_2026-04-07_16-36-23/log.txt) | 1.1575 | 715.23M | 24,335 MiB | 4,173 MiB (PyTorch-allocated) | PKM+FwPKM large; -0.0172 vs baseline |
| 28  | off | off | | | [link](logs/wikitext-103_2026-04-07_20-21-24/log.txt) | 1.2000 | 356.07M | 18,357 MiB | 2,118 MiB (PyTorch-allocated) | MLP off; wavelet pipeline only; +0.0252 vs baseline |
| 29  | on  | off | 529 | | [link](logs/wikitext-103_2026-04-07_23-11-07/log.txt) | 1.1985 | 366.97M | 18,913 MiB | 2,170 MiB (PyTorch-allocated) | MLP off, PKM only; +0.0237 vs baseline |
| 30  | on  | on  | 529 | 529 | [link](logs/wikitext-103_2026-04-08_02-19-39/log.txt) | 1.1957 | 377.86M | 19,569 MiB | 2,243 MiB (PyTorch-allocated) | MLP off, PKM+FwPKM; +0.0209 vs baseline |
| 31  | off | on  | | 1681 | [link](logs/wikitext-103_2026-04-09_21-03-15/log.txt) | 1.1732 | 389.46M | 19,667 MiB | 2,343 MiB (PyTorch-allocated) | FwPKM param-matched; -0.0016 vs baseline |

### MLP expansion = 50 + memory

| Run | PKM | FwPKM | PKM num keys | FwPKM num keys | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|-----|-------|--------------|----------------|--------|---------------|--------|------------|----------------|-------|
| 22  | off | off | | | [link](logs/wikitext-103_2026-04-06_23-32-36/log.txt) | 1.1406 | 880.88M | 33,524 MiB | 5,121 MiB (PyTorch-allocated) | MLP-50 baseline (from MLP sweep) |
| 32  | on  | off | 16384 | | [link](logs/wikitext-103_2026-04-09_00-16-49/log.txt) | NaN | 1055.21M | 35,882 MiB | — | Diverged at step 3600 (LR=0.008) |
| 33  | off | on  | | 16384 | [link](logs/wikitext-103_2026-04-09_03-57-03/log.txt) | 1.1408 | 1055.21M | 36,682 MiB | 6,439 MiB (PyTorch-allocated) | FwPKM large; -0.0002 vs MLP-50 baseline; stable where PKM NaN'd |
| 34  | on  | on  | 16384 | 16384 | [link](logs/wikitext-103_2026-04-09_08-29-18/log.txt) | 1.1381 | 1229.54M | 39,041 MiB | 7,116 MiB (PyTorch-allocated) | Both large; -0.0024 vs MLP-50; FwPKM stabilized PKM |
| 35  | on  | on  | 529 | 529 | [link](logs/wikitext-103_2026-04-09_13-22-27/log.txt) | 1.1394 | 902.67M | 34,655 MiB | 5,246 MiB (PyTorch-allocated) | Both default; -0.0011 vs MLP-50 |

### Per-layer embedding (PLE): C = 512, epochs = 1, optimal booleans + mlp_expansion = 1

Reintroduces original token embedding as a learned per-channel residual at each block. Learned gamma (C,) per layer, zero-initialized. +0.01M params total.

| Run | per_layer_embedding | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|---------------------|--------|---------------|--------|------------|----------------|-------|-------|
| 4   | false | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | | Baseline (Run 4) |
| 36  | true  | [link](logs/wikitext-103_2026-04-09_18-06-43/log.txt) | 1.1738 | 366.59M | 18,898 MiB | 2,179 MiB (PyTorch-allocated) | -0.0009 | +10,240 params; essentially free |

> **Note:** FwPKM trains statically (identical to PKM). Inference-time weight updates (`fwpkm_inference_update`) tested separately in generation quality, not BPB.

### Mixer depth: C = 512, epochs = 1, optimal booleans + mlp_expansion

Stacked spectral mixing within each block — adding depth to the per-scale gated transforms in Hadamard space without repeating wavelet/Hadamard passes. Each depth step: LN + gated linear + bias (no residual), final step omits LN/bias.

| Run | mixer_depth | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|--------|---------------|--------|------------|----------------|-------|-------|
| 4   | 1 | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | | Baseline (Run 4) |
| 37  | 2 | [link](logs/wikitext-103_2026-04-08_09-51-47/log.txt) | 1.1661 | 471.74M | 23,428 MiB | 2,780 MiB (PyTorch-allocated) | -0.0098 | First depth increase |
| 38  | 3 | [link](logs/wikitext-103_2026-04-08_13-46-39/log.txt) | 1.1743 | 576.91M | 28,837 MiB | 3,381 MiB (PyTorch-allocated) | -0.0033 | Diminishing vs depth=2 |
| 39  | 5 | [link](logs/wikitext-103_2026-04-08_18-26-31/log.txt) | NaN | 787.24M | 39,657 MiB | — | — | Diverged at step 3600 (LR=0.008); vanishing/exploding gradients without residuals |

### Mixer depth + higher LR

| Run | mixer_depth | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|-----|--------|---------------|--------|------------|----------------|-------|-------|
| 37  | 2 | 0.01 | [link](logs/wikitext-103_2026-04-08_09-51-47/log.txt) | 1.1661 | 471.74M | 23,428 MiB | 2,780 MiB (PyTorch-allocated) | -0.0098 | From depth sweep |
| 40  | 2 | 0.02 | [link](logs/wikitext-103_2026-04-10_00-23-31/log.txt) | NaN | 471.74M | — | — | — | Diverged step 2200 (LR=0.01); LN alone insufficient at 2x LR |

### Mixer depth stabilizers ablation: alpha_d, beta_d (init 1/D), scaled mixer init

| Run | mixer_depth | stabilizers | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|-------------|-----|--------|---------------|--------|------------|----------------|-------|-------|
| 37  | 2 | false | 0.01 | [link](logs/wikitext-103_2026-04-08_09-51-47/log.txt) | 1.1661 | 471.74M | 23,428 MiB | 2,780 MiB (PyTorch-allocated) | -0.0098 | From depth sweep |
| 41  | 2 | true  | 0.01 | [link](logs/wikitext-103_2026-04-10_03-29-19/log.txt) | 1.1715 | 471.74M | 23,428 MiB | 2,781 MiB (PyTorch-allocated) | -0.0032 | Stabilizers cost +0.0066 BPB vs unstabilized |
| 42  | 2 | true  | 0.02 | [link](logs/wikitext-103_2026-04-10_07-14-48/log.txt) | NaN | 471.74M | — | — | — | Diverged step 1800 (LR=0.008); stabilizers made it worse |

### Mixer depth + lower LR: can reduced LR stabilize deeper mixers?

NaN threshold is consistently at LR reaching ~0.008. Lower peak LR to stay below this boundary.

| Run | mixer_depth | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|-------------|-----|--------|---------------|--------|------------|----------------|-------|-------|
| 43  | 5 | 0.004 | [link](logs/wikitext-103_2026-04-10_07-59-36/log.txt) | 1.2895 | 787.24M | 39,657 MiB | 4,584 MiB (PyTorch-allocated) | +0.1146 | Stable but severely undertrained; LR too low for 1 epoch |
| 44  | 10 | 0.001 | | | 1313.06M | | | | MBS=4, GA=4 to fit in VRAM; extreme depth stress test |

### Layers = 1: C = 512, epochs = 1, optimal booleans + mlp_expansion

| Run | Layers | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
| 45  | 1  | [link](logs/wikitext-103_2026-04-10_16-31-26/log.txt) | 1.3245 | 67.22M | 4,684 MiB | 448 MiB (PyTorch-allocated) | 17min train; 46.6 tok/s gen |

> Note: The speed and results are so good that it warrants further testing various configurations of layers = 1 immediately.

### Layers = 1 ablations, part 1 (C = 512)

L=1 baseline uses ~4.7 GB VRAM, leaving ~44 GB headroom. Each run takes ~17 min. Testing mixer depth, MLP width, and large batch sizes as substitutes for model layers.

All runs use MBS=8, GA=2 except run 46 (noted below).

| Run | mlp_expansion | mixer_depth | lr | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---------------|-------------|-----|--------|---------------|--------|------------|----------------|-------|
| 45  | 1   | 1  | 0.01 | [link](logs/wikitext-103_2026-04-10_16-31-26/log.txt) | 1.3245 | 67.22M | 4,684 MiB | 448 MiB (PyTorch-allocated) | L=1 baseline |
| 46  | 100 | 1  | 0.01 | [link](logs/wikitext-103_2026-04-10_17-14-18/log.txt) | 1.2577 | 119.18M | 4,324 MiB | 759 MiB (PyTorch-allocated) | **MBS=4, GA=4.** Massive MLP; -0.0708 vs L=1 baseline; 28min total |
| 47  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_17-54-03/log.txt) | 1.4755 | 114.54M | 7,078 MiB | 719 MiB (PyTorch-allocated) | MD=10 no residuals; +0.1580 vs L=1 baseline; WORSE |
| 48  | 1   | 10 | 0.02 | [link](logs/wikitext-103_2026-04-10_18-42-34/log.txt) | NaN | 114.54M | — | — | Diverged step 4200 (LR=0.019) |
| 49  | 1   | 2  | 0.01 | [link](logs/wikitext-103_2026-04-10_19-29-57/log.txt) | 1.3043 | 72.48M | 4,915 MiB | 478 MiB (PyTorch-allocated) | MD=2 no residuals; -0.0142 vs L=1 baseline |
| 50  | 1   | 2  | 0.01 | [link](logs/wikitext-103_2026-04-10_19-49-29/log.txt) | 1.2924 | 72.48M | 4,915 MiB | 478 MiB (PyTorch-allocated) | MD=2 + residuals; -0.0253 vs L=1 baseline |
| 51  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_20-10-24/log.txt) | 1.2934 | 114.54M | 7,078 MiB | 719 MiB (PyTorch-allocated) | MD=10 + residuals; -0.0241 vs L=1 baseline |
| 52  | 100 | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_20-51-04/log.txt) | 1.5200 | 166.50M | 8,564 MiB | 1,041 MiB (PyTorch-allocated) | MLP=100 + MD=10 no residuals; WORSE than either alone |
| 53  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_21-37-45/log.txt) | 1.4393 | 114.54M | 7,082 MiB | 719 MiB (PyTorch-allocated) | PLE=true; still worse than baseline (MD=10 no resid dominates) |
| 54  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_22-16-17/log.txt) | NaN | 114.28M | 6,279 MiB | — | DB=false; NaN — DB provides critical stability at L=1 |
| 55  | 1   | 10 | 0.01 | [link](logs/wikitext-103_2026-04-10_22-48-20/log.txt) | NaN | 354.92M | — | — | C=1024, MD=10 no resid; NaN step 4500 (LR=0.01) |

### Layers = 1 ablations, part 2 (bigger C)

| Run | C | MLP | MD | lr | block | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|------|-----|-----|------|-------|--------|---------------|--------|------------|----------------|-------|
| 56  | 1024 | 1   | 2  | 0.01 | 512  | [link](logs/wikitext-103_2026-04-10_23-37-19/log.txt) | 1.2151 | 186.90M | 7,100 MiB | 1,113 MiB (PyTorch-allocated) | C=1024, resid; -0.0771 vs MD=2+resid C=512 |
| 57  | 2048 | 1   | 2  | 0.01 | 512  | [link](logs/wikitext-103_2026-04-11_00-15-26/log.txt) | **1.1657** | 541.57M | 13,453 MiB | 3,211 MiB (PyTorch-allocated) | **C=2048 approaches L=20!** |
| 58  | 512  | 1   | 2  | 0.01 | 2048 | [link](logs/wikitext-103_2026-04-11_02-00-34/log.txt) | 1.3341 | 76.68M | 14,696 MiB | 505 MiB (PyTorch-allocated) | Context alone doesn't help |
| 59  | 1024 | 100 | 2  | 0.01 | 2048 | [link](logs/wikitext-103_2026-04-11_02-19-23/log.txt) | 1.2547 | 411.41M | 28,847 MiB | 2,526 MiB (PyTorch-allocated) | Kitchen sink A; PLE |
| 60  | 2048 | 20  | 2  | 0.01 | 2048 | [link](logs/wikitext-103_2026-04-11_03-24-37/log.txt) | NaN | 768.14M | — | — | — | Kitchen sink B; NaN step 300 |
| 61  | 2048 | 20  | 2  | 0.005| 1024 | [link](logs/wikitext-103_2026-04-11_05-30-57/log.txt) | 1.2024 | 734.56M | 23,346 MiB | 4,358 MiB (PyTorch-allocated) | LR too low; worse than C=2048 MLP=1 |
| 62  | 2048 | 20  | 1  | 0.01 | 512  | [link](logs/wikitext-103_2026-04-11_08-13-12/log.txt) | **1.1538** | 617.05M | 14,109 MiB | 3,519 MiB (PyTorch-allocated) | **New L=1 record! Beats L=20 C=512 baseline (1.1751)** |
| 63  | 2048 | 20  | 1  | 0.02 | 512  | [link](logs/wikitext-103_2026-04-11_10-04-14/log.txt) | NaN | 617.05M | — | — | — | NaN step 700 (LR=0.003); lr=0.02 too high for MLP=20 at C=2048 |
| 64  | 4096 | 20  | 1  | 0.01 | 512  | [link](logs/wikitext-103_2026-04-11_11-34-42/log.txt) | — | 2056.18M | ~33 GB | — | — | Early-stopped; diminishing returns vs C=2048 |
| 65  | 2048 | 20  | 2  | 0.01 | 512  | | | ~2.2B | | | | PLE, resid; full recipe at stable LR |
| 66  | 2048 | 20  | 2  | 0.01 | 512  | | | ~2.3B | | | | Same + PKM+FwPKM-16384 |

### Lifting hidden multiplier: L=1, C=2048, MLP=20

Wider predict/update MLPs in the lifting wavelet. Tests whether more expressive local token-to-token interaction improves BPB. Identity init for extra hidden dims (nn.init.eye_) unless noted.

| Run | lifting_hidden_mult | init | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|---------------------|------|--------|---------------|--------|------------|----------------|-------|
| 62  | 1 | — | [link](logs/wikitext-103_2026-04-11_08-13-12/log.txt) | 1.1538 | 617.05M | 14,109 MiB | 3,519 MiB (PyTorch-allocated) | Baseline |
| 67  | 2 | zeros | [link](logs/wikitext-103_2026-04-11_14-23-32/log.txt) | — | 768.08M | — | — | Early-stopped; zero init, no improvement |
| 68  | 2 | eye | [link](logs/wikitext-103_2026-04-11_15-39-24/log.txt) | NaN | 768.08M | — | — | — | Identity init; NaN step 1300 (LR=0.003); signal too strong |
| 69  | 2 | normal(0.01) | [link](logs/wikitext-103_2026-04-11_17-46-16/log.txt) | 1.1557 | 768.08M | 16,989 MiB | 4,383 MiB (PyTorch-allocated) | Stable but identical to mult=1; local expressivity not the bottleneck |

### Loop iterations (LoopLM): L=1, C=2048, reuse same weights T times

Same layer stack applied T times sequentially. Loss averaged across all iterations. Zero additional parameters — adds compute, not capacity. Inspired by LoopLM (arxiv:2510.25741).

| Run | C | MLP | MD | lr | T | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|------|-----|-----|------|---|--------|---------------|--------|------------|----------------|-------|
| 62  | 2048 | 20  | 1  | 0.01 | 1 | [link](logs/wikitext-103_2026-04-11_08-13-12/log.txt) | 1.1538 | 617.05M | 14,109 MiB | 3,519 MiB (PyTorch-allocated) | Baseline (no looping) |
| 70  | 2048 | 20  | 1  | 0.01 | 4 | [link](logs/wikitext-103_2026-04-11_13-19-59/log.txt) | — | 617.05M | 31,538 MiB | — | Early-stopped; ~0.04 val_loss gain for 3.5x compute; not worth it |

### Optimal low-layer config: L=2, C=2048, full recipe

L=2, C=2048, MLP=20, PLE, PKM+FwPKM-16384. ~1.18B params, ~21 GB estimated.

| Run | Layers | Epochs | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|--------|---------------|--------|------------|----------------|-------|
|   | 2 | 1 | [link](logs/wikitext-103_2026-04-11_21-09-05/log.txt) | **1.1126** | 1180.28M | 24,643 MiB | 6,733 MiB (PyTorch-allocated) | **New overall best! Beats L=20 3-epoch baseline (1.1169)** |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-12_00-37-11/log.txt) | **1.0856** | 1180.28M | 24,643 MiB | 6,733 MiB (PyTorch-allocated) | **New best! No dropout; best val at epoch 3; train/val gap 1.77 by epoch 5** |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-12_17-11-15/log.txt) | **1.0455** | 1180.28M | 24,883 MiB | 6,733 MiB (PyTorch-allocated) | **1.0x dropout; new best! Val still improving at epoch 5; gap=1.00** |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-13_09-51-55/log.txt) | **1.0306** | 1180.28M | 24,883 MiB | 6,733 MiB (PyTorch-allocated) | **1.5x dropout; new best! Gap=0.81; val still improving at epoch 5** |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-14_06-41-17/log.txt) | — | 1180.28M | — | — | — | 1.5x dropout + WD=1e-3; early-stopped; WD too aggressive for Adagrad, training stalled |
|   | 2 | 5 | [link](logs/wikitext-103_2026-04-14_09-07-12/log.txt) | **1.0234** | 1180.28M | 24,883 MiB | 6,733 MiB (PyTorch-allocated) | **2.0x dropout; new best! Gap=0.69; val improving all 5 epochs** |

### Grokking experiment: C=128, L=2, tiny core + massive memory

~42M params total with L=2. Tiny mixer core with massive MLP and sparse memory. Two layers give double the wavelet decomposition depth at minimal cost. Testing if extreme epoch count can compensate for tiny width. If a ~42M model achieves comparable BPB to a 1B+ model, it demonstrates that WaveletLM's wavelet mixing can generalize language patterns at minimal scale given sufficient training.

| Run | C | Layers | Epochs | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|-----|--------|--------|--------|---------------|--------|------------|----------------|-------|
|   | 128 | 2 | 16/100 | [link](logs/wikitext-103_2026-04-14_02-31-17/log.txt) | — | ~42M | — | — | Early-stopped; train loss plateaued ~3.77 by epoch 11; gap only 0.10; insufficient capacity for memorization |

### Exponential parametrization: mixer only

Apply exp() reparameterization to GatedSpectralMixer weights only. Tests whether better mixer initialization/gradient dynamics improve BPB and fix NaN at previously-unstable configs.

| Run | Config | exp_param | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|-----------|--------|---------------|--------|------------|----------------|-------|
|   | L=1, C=2048, MLP=20, lr=0.01 | true | [link](logs/wikitext-103_2026-04-15_01-46-27/log.txt) | 1.1538 | 617.05M | 14,109 MiB | 3,519 MiB (PyTorch-allocated) | Identical to baseline ([Run 62](logs/wikitext-103_2026-04-11_08-13-12/log.txt): 1.1431); no improvement at lr=0.01 |
|   | L=1, C=2048, MLP=20, lr=0.02 | true | [link](logs/wikitext-103_2026-04-15_03-35-54/log.txt) | **1.1424** | 617.05M | 14,109 MiB | 3,519 MiB (PyTorch-allocated) | **Survived lr=0.02! Previously NaN'd; -0.0084 vs baseline** |
|   | L=20, C=512, MD=5, lr=0.01 | true | [link](logs/wikitext-103_2026-04-15_05-24-40/log.txt) | NaN | 787.24M | — | — | — | NaN at step 3600 again; exp param doesn't fix depth instability |

### Layers > 1: C = 512, epochs = 1, optimal booleans with mlp_expansion

| Run | Layers | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
| 45  | 1  | [link](logs/wikitext-103_2026-04-10_16-31-26/log.txt) | 1.3245 | 67.22M | 4,684 MiB | 448 MiB (PyTorch-allocated) | 17min train; 46.6 tok/s gen |
|   | 4  | [link](logs/wikitext-103_2026-04-15_10-53-44/log.txt) | 1.2400 | 114.49M | 6,916 MiB | 722 MiB (PyTorch-allocated) | |
|   | 10  | [link](logs/wikitext-103_2026-04-15_11-34-21/log.txt) | 1.1978 | 209.02M | 11,379 MiB | 1,269 MiB (PyTorch-allocated) | |
|   | 15  | [link](logs/wikitext-103_2026-04-15_13-01-46/log.txt) | 1.1813 | 287.80M | 15,098 MiB | 1,724 MiB (PyTorch-allocated) | |
|   | 18  | [link](logs/wikitext-103_2026-04-15_15-10-58/log.txt) | 1.1772 | 335.07M | 17,330 MiB | 1,998 MiB (PyTorch-allocated) | |
|   | 20 | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | Baseline |
|   | 30 | [link](logs/wikitext-103_2026-04-15_17-46-01/log.txt) | 1.1638 | 524.14M | 26,257 MiB | 3,091 MiB (PyTorch-allocated) | Slight improvement over L=20; diminishing returns |

### Levels: C = 512, epochs = 1, optimal booleans + mlp_expansion + layers, block_size = 512

| Run | Levels | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Notes |
|-----|--------|--------|---------------|--------|------------|----------------|-------|
|   | 1 | [link](logs/wikitext-103_2026-04-15_22-17-16/log.txt) | 1.2354 | 114.51M | 7,133 MiB | 738 MiB (PyTorch-allocated) | 3.2x fewer params, only +0.061 BPB vs baseline |
|   | 2 | [link](logs/wikitext-103_2026-04-16_04-55-49/log.txt) | 1.1994 | 146.02M | 8,594 MiB | 918 MiB (PyTorch-allocated) | |
|   | 3 | [link](logs/wikitext-103_2026-04-16_05-54-07/log.txt) | 1.1825 | 177.53M | 10,054 MiB | 1,098 MiB (PyTorch-allocated) | |
|   | 4 | [link](logs/wikitext-103_2026-04-16_07-14-10/log.txt) | 1.1728 | 209.04M | 11,515 MiB | 1,279 MiB (PyTorch-allocated) | |
|   | 5 | [link](logs/wikitext-103_2026-04-15_23-05-09/log.txt) | 1.1669 | 240.55M | 12,976 MiB | 1,459 MiB (PyTorch-allocated) | Beats baseline with 34% fewer params! |
|   | 6 | [link](logs/wikitext-103_2026-04-16_08-50-21/log.txt) | 1.1673 | 272.05M | 14,436 MiB | 1,639 MiB (PyTorch-allocated) | |
|   | 9 | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | Baseline (Run 4; default = log2(block_size=512)) |
|   | 11 | [link](logs/wikitext-103_2026-04-16_01-02-33/log.txt) | 1.1797 | 429.60M | 21,739 MiB | 2,541 MiB (PyTorch-allocated) | Worse than levels=5; confirms diminishing returns past 5 |

### Low-rank factorization in spectral mixer

| Run | low_rank | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|----------|--------|---------------|--------|------------|----------------|-------|-------|
|   | 0  | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | | Baseline (Run 4; full rank) |
|   | 4  | [link](logs/wikitext-103_2026-04-16_10-43-48/log.txt) | 1.1702 | 367.40M | 18,835 MiB | 2,184 MiB (PyTorch-allocated) | -0.0045 | Slight improvement! Worth keeping at small cost |
|   | 16 | [link](logs/wikitext-103_2026-04-16_14-02-48/log.txt) | 1.1691 | 369.86M | 18,887 MiB | 2,196 MiB (PyTorch-allocated) | -0.0057 | Slightly better than rank=4; small marginal gain (-0.0012) |

### Lifting hidden multiplier

| Run | lifting_hidden_mult | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|---------------------|--------|---------------|--------|------------|----------------|-------|-------|
|   | 1 | [link](logs/wikitext-103_2026-04-03_04-51-07/log.txt) | 1.1750 | 366.58M | 18,738 MiB | 2,179 MiB (PyTorch-allocated) | | Baseline (Run 4) |
|   | 2 | [link](logs/wikitext-103_2026-04-16_17-23-50/log.txt) | NaN | 555.51M | — | — | — | NaN at step 2500 (LR=0.0057); wider lifting unstable at L=20. Future stability fixes (e.g., spectral norm on lifting predict/update, or scaled init for hidden dims) may make this viable. |
| N/A | 4 | — | — | — | — | — | — | Cancelled; mult=2 already NaN'd. Revisit with stability fixes. |

### Decompose Bypass Disablement (DBD): post-combined-reduction baseline (L=1, levels=7, epochs=1)

Tests whether the running-mean × `history_gains` machinery (`decompose_bypass` and `decompose_bypass_cross_window`) can be removed at the post-combined-reduction regime with the new `low_rank=16` winner. Prior smaller-scale ablations (Boolean ablation table at L=1 / E=1 and the Test 5 sweep) found both flags within ±0.0015 BPB of baseline (within noise) and projected they could be turned off "for free." This run validates the projection at L=1 / levels=7 / bs=16384 / `low_rank=16`.

| Run | decompose_bypass | decompose_bypass_cross_window | low_rank | Folder | BPB (sliding) | Notes |
|-----|------------------|-------------------------------|----------|--------|---------------|-------|
| R0  | true (baseline) | true | 4 | [link](logs/wikitext-103_2026-05-02_21-43-22/log.txt) | 1.2361 | Reference |
| DBD | **false** | **false** | 16 | [link](logs/wikitext-103_2026-05-05_12-47-12/log.txt) | NaN at step 1500 | NaN at lr=6.84e-3 (mid-warmup), earlier than R1.5's NaN at step 2250 / lr=1.00e-2 with the flags on. Both flags doing real stability work at this regime. |

**Conclusion:** keep `decompose_bypass=true` and `decompose_bypass_cross_window=true` as headline defaults. Prior projections of "free disablement" do not hold once `levels=7` and `low_rank=16` are stacked. Removal is off the table until the optimizer sweep (Muon) clears the cascade-amplification picture; if Muon makes the bypass redundant, DBD can be retested then.

### New Baseline

The baseline used for all 1-epoch screening ablations. Combines proven wins (levels=5; low_rank=4) on top of the wide & shallow config. Halves per-epoch runtime vs the previous C=512/L=20 baseline. **PKM and FwPKM are intentionally OFF** during screening — they get re-introduced for the final 5-epoch best-run candidate (saves ~10-15% time and ~150M params per 1-epoch run).

**Config:** L=2, C=2048, MLP=20, levels=5, **PLE**, lr=0.01, low_rank=4

> **Note:** exp_param + lr=0.02 NaN'd at this config ([logs/.../2026-04-17_00-27-55](logs/wikitext-103_2026-04-17_00-27-55/log.txt), step 4000 LR=1.82e-02). Worked at L=1 previously but L=2 doubles residual signal accumulation. Re-test once stable_parametrization (spectral_norm in particular) is validated.

| Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Time | Notes |
|--------|---------------|--------|------------|----------------|------|-------|
| [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1166 | 827.03M | 18,016 MiB | 12,679 MiB (PyTorch-allocated) | 2.40h | +0.004 BPB vs old baseline (1.1133) but 30% faster and 30% smaller. `per_scale_mixer_widths` was subsequently promoted into the baseline ([logs/.../2026-04-17_14-23-49](logs/wikitext-103_2026-04-17_14-23-49/log.txt), BPB 1.1168), which is the reference used for all Part 1 ablation deltas below. |

### New Baseline Boolean ablations part 1: C=2048, L=2, epochs=1 wide & shallow model

Screening wavelet/mixer and true-feedback augmentations from [`plans/wavelet_and_mixer_augmentations.md`](plans/wavelet_and_mixer_augmentations.md), [`plans/feedback_mechanisms.md`](plans/feedback_mechanisms.md), and [`plans/wavelet_crawl.md`](plans/wavelet_crawl.md) against the **new baseline** (C=2048/L=2/MLP=20/PLE + **levels=5, lr=0.01, low_rank=4**). Folds in proven wins (levels=5 from the L=20 finding; low_rank=4 = -0.0045 BPB at trivial cost). Halves runtime per epoch via fewer wavelet levels. PKM/FwPKM deferred to the final 5-epoch best run; exp_param + lr=0.02 deferred until stable_parametrization is validated. Each feature tested individually at 1 epoch; winners stack into the 5-epoch best-combo run.

| Run | Feature | Folder | BPB (sliding) | Params | Train VRAM | Inference VRAM | Delta | Notes |
|-----|---------|--------|---------------|--------|------------|----------------|-------|-------|
|   | New baseline probe | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1166 | 827.03M | 18,016 MiB | 12,679 MiB (PyTorch-allocated) | **+0.0040** | 30% faster (2.40h vs 3.42h) and 30% smaller than old baseline for +0.004 BPB. lr=0.01 fallback after lr=0.02+exp_param NaN'd ([link](logs/wikitext-103_2026-04-17_00-27-55/log.txt)); defer until stab_spectral_norm is validated. |
|   | Untied reconstruction | [link](logs/wikitext-103_2026-04-17_06-20-21/log.txt) | 1.1167 | 994.88M | 19,777 MiB | 14,440 MiB (PyTorch-allocated) | **+0.0000** | Exact tie at +168M (+20%) and +3% time. One wavelet copy suffices. **Dropped** from best-run. |
|   | Cross-scale gating (routing) | [link](logs/wikitext-103_2026-04-17_08-51-44/log.txt) | **1.1159** | 827.03M (+72) | 18,400 MiB | 12,679 MiB (PyTorch-allocated) | **-0.0007** | Small consistent win; best val -0.0042. Essentially free (+72 params, +2.5% time). **Keep** for best-run. |
|   | Multi-basis lifting (haar+random, attempt 1) | [link](logs/wikitext-103_2026-04-17_11-21-47/log.txt) | NaN | 994.88M | — | — | — | Default Kaiming random init + basis_weights[0]=5.0. **NaN at step 1800, LR=4.10e-03** (well below peak 0.01). |
|   | Multi-basis lifting (haar+random, attempt 2) | [link](logs/wikitext-103_2026-04-17_13-25-19/log.txt) | NaN | 994.88M | — | — | — | Tightened: random init `N(0, 0.01²)` with zero bias, basis_weights[0]=10.0 (softmax ~0.99995 on haar at init). **NaN at step 3400, LR=7.75e-03** — fixes roughly doubled LR tolerance (step 1800→3400, LR 0.0041→0.0078) but didn't fully solve. Deferred until stable_parametrization validated (spectral_norm on mixer is the likely fix). |
|   | Per-scale mixer widths [1×3, 0.5×3] | [link](logs/wikitext-103_2026-04-17_14-23-49/log.txt) | **1.1162** | ~815M | 17,847 MiB | 12,509 MiB (PyTorch-allocated) | **-0.0005** | Tiny BPB improvement AND **23% faster** (1.85h vs 2.40h), -1% VRAM. Starving fine scales produces less overfitting — best val +0.0054 but final BPB better. **Keep** for best-run. |
|   | Looped blocks (K=8 shared) | [link](logs/wikitext-103_2026-04-17_19-47-45/log.txt) | 1.1134 | ~360M | 29,367 MiB | 8,459 MiB (PyTorch-allocated) | -0.0039 | **Inefficient** with 3× the training time (5.6h vs 1.85h baseline). Better to just train longer. |
|   | Wavelet crawl (K=3) | [link](logs/wikitext-103_2026-04-18_06-19-52/log.txt) | **1.1126** | ~840M | 18,171 MiB | 12,509 MiB (PyTorch-allocated) | **-0.0037** | Learned ±1 dilation offsets per level. Third-biggest single-feature win after PSW and CSG. Essentially free (+1.6% time, +2% VRAM). **Keep** for best-run. ⚠️ **Stability caveat:** crawl shifts the predict/update networks' input distribution away from their Haar-init regime; K=5 NaN'd entirely. K=3 was stable in this 1-epoch run, but stacked with other features and/or longer training (5+ epochs, higher LR), there's a non-zero risk the softmax drift compounds. If stacked best-run NaNs, this is a primary suspect — try `stab_spectral_norm` first, then consider stronger init bias (10.0 → 15.0). |
|   | Wavelet crawl (K=5) | [link](logs/wikitext-103_2026-04-18_08-14-44/log.txt) | NaN | ~840M | — | — | — | ±2 dilation offsets. NaN'd step ~4300 (LR=9.81e-03). At higher levels (base ≥ 8), softmax spread of ±2 deviates too far from Haar init, destabilizes predict/update networks. Not rescued — K=3 captures the benefit. |
|   | Shared lifting weights (SLW) | [link](logs/wikitext-103_2026-04-18_09-46-25/log.txt) | 1.1160 | ~770M | 16,886 MiB | 11,388 MiB (PyTorch-allocated) | **-0.0003** | One shared lifting wavelet instead of per-layer. Essentially tied on BPB; **-5% train VRAM, -9% inference VRAM**. Free memory savings. **Keep** for best-run. |
|   | Lifting linear-only (LLO) | [link](logs/wikitext-103_2026-04-18_11-37-14/log.txt) | 1.1309 | ~790M | 15,766 MiB | 11,388 MiB (PyTorch-allocated) | **+0.0093** | Replaces predict/update Sequentials (Linear→GELU→Dropout→Linear) with single Linears. Significant BPB regression — the GELU nonlinearity matters. **Drop** despite -11% time / -12% train VRAM savings. |
|   | SLW + LLO combined | [link](logs/wikitext-103_2026-04-18_13-17-47/log.txt) | 1.1266 | ~720M | 15,286 MiB (PyTorch-allocated) | — | **+0.0061** | LLO dominates the combo. SLW partially offsets but not enough. Fastest run yet (1.59h, -14%) and smallest VRAM, but BPB regression kills it. **Drop.** |

### New Baseline Boolean ablations part 2: Stable parametrization — SKIPPED

Six stability fixes (spectral norm on mixer, FF √(hidden_dim) scaling, embed √C scaling, proj_out √(C·L) scaling, mixer eps scaling, level-dependent lifting init) were implemented but **not evaluated individually.** See [plans/other_post_release_plans.md §5](plans/other_post_release_plans.md#5-stable-parametrization--validation-and-finishing-gaps) for details and validation plan.

**Reason:** the master bundle performed *worse* than the unmodified configs in every rescue test attempted (see Part 3). Most notably, the lr=0.02+exp_param rescue NaN'd at step 3200 with stab, versus step 4000 without it — i.e. the stability features *accelerated* the failure rather than preventing it. This strongly suggests a bug in at least one of the implementations (likely `stab_proj_out_scaling`, whose `1/√(C·L)` formula produces a residual-stream contribution ~15× larger than the original `1e-3` at the new baseline).

With the final 5-epoch best run committed to `lr=0.01` (which is stable without any stab flags), diagnosing the specific bug is no longer release-critical. The code is retained in the repo for future work; the individual-flag ablations are skipped to conserve compute budget. Any researcher wanting to revisit can enable the flags individually via the config.

### New Baseline Boolean ablations part 3: Rescue tests (vs known NaN configs)

If the master flag rescues a previously-NaN config, follow up with per-feature ablation to identify the load-bearing fix.

| Run | NaN config under test | Stab flags | Folder | BPB (sliding) | Params | Notes |
|-----|----------------------|-----------|--------|---------------|--------|-------|
|   | mixer_depth=5 (was [link](logs/wikitext-103_2026-04-08_18-26-31/log.txt): NaN step 3600) | master | [link](logs/wikitext-103_2026-04-18_14-55-19/log.txt) | **crashed** | ~787M | **Silent crash** before step 100. Likely torch.compile + spectral_norm parametrization on 1000 wrapped mixers (L=20 × S=10 × MD=5) exceeded resources. **Not rescued.** |
|   | lifting_hidden_mult=2 (was [link](logs/wikitext-103_2026-04-16_17-23-50/log.txt): NaN step 2500) | master | [link](logs/wikitext-103_2026-04-18_14-58-10/log.txt) | **NaN step 200** | ~556M | **WORSE than unmodified config** (original NaN'd at step 2500; with stab bundle, NaN at step 200 with val 8.92 at step 100). Stab bundle *actively destabilized* this. Suspect `stab_proj_out_scaling` formula `1/√(C·L)` at L=20 gives ~10× stronger proj_out than original 1e-3, amplifying residual-stream signal. **Not rescued.** |
|   | C=2048, lr=0.02 (was [link](logs/wikitext-103_2026-04-11_10-04-14/log.txt): NaN step 700) | master | | | ~617M | Width × LR NaN; survived only with exp_param previously |
|   | new baseline + lr=0.02 + exp_param (NaN'd [link](logs/wikitext-103_2026-04-17_00-27-55/log.txt) step 4000, LR=1.82e-02) | master | | | 827.03M | If rescued, unlocks lr=0.02 at the new baseline — likely the biggest latent BPB win (L=1 previously showed -0.0084 at lr=0.02+exp_param). Primary target for `stab_spectral_norm`. |

### Block size (context window) — at new baseline

`levels=5` from the new baseline supports any `block_size >= 32`, so no `levels` adjustment is needed when changing context length.

| Run | block_size | Folder | BPB (sliding) | Params | Train VRAM | Time | Delta | Notes |
|-----|------------|--------|---------------|--------|------------|------|-------|-------|
| ~~64~~ | — | **Cancelled** | — | — | — | — | BS=128 came in worse than BS=256, so the floor is at 256. Not worth probing smaller. |
|   | 128  | [link](logs/wikitext-103_2026-04-19_07-00-01/log.txt) | 1.1073 | ~840M | 16,573 MiB (PyTorch-allocated) | 2.74h | -0.0093 | Better than baseline (-0.0093) but **worse than BS=256** (+0.0047). The "more updates" trend plateaus between 256 and 128. BS=256 stays the winner. |
|   | **256**  | [link](logs/wikitext-103_2026-04-18_16-12-23/log.txt) | **1.1020** | ~840M | 16,680 MiB (PyTorch-allocated) | 2.16h | **-0.0140** | **Biggest single-feature win so far.** ~2× gradient updates per epoch since dataset splits into more blocks. With levels=5 (max dilation 2^4=16), 256-token context is still ample. |
|   | **256 + grad_accum=1** | [link](logs/wikitext-103_2026-04-19_09-46-39/log.txt) | **1.0958** | ~840M | 16,680 MiB (PyTorch-allocated) | 2.87h | **-0.0202** | **Stacks near-linearly!** Individual wins: BS=256 (-0.0140) + GA=1 (-0.0076) = -0.0216 linear prediction; actual -0.0202 (~94% of linear). First 1-epoch result below 1.10. Critical adoption decision for 5-epoch best-run (see "Best run candidate" section). |
|   | 512  | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1166 | ~840M | 18,016 MiB (PyTorch-allocated) | 1.85h | | Baseline (new baseline probe) |
|   | 1024 | [link](logs/wikitext-103_2026-04-18_18-23-56/log.txt) | NaN (3.5220) | ~840M | 24,091 MiB | 1.53h | — | ❌ NaN'd. Effective batch reached 8192 tokens, crossed AMP/fp16 overflow threshold. Would need MBS reduction to recover. |

### Grad accum — at new baseline

| Run | grad_accum | Effective batch | Folder | BPB (sliding) | Params | Time | Delta | Notes |
|-----|-----------|----------------|--------|---------------|--------|------|-------|-------|
|   | **1** | **8**  | [link](logs/wikitext-103_2026-04-18_19-57-53/log.txt) | **1.1086** | ~840M | 2.39h | **-0.0076** | **Solid win.** Half the effective batch → 2× gradient updates per epoch at cost of +29% wall-clock time. |
|   | 2 | 16 | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1166 | ~840M | 1.85h | | Baseline (new baseline probe) |
|   | 4 | 32 | [link](logs/wikitext-103_2026-04-18_22-23-30/log.txt) | NaN (3.5221) | ~840M | 1.39h | — | ❌ NaN'd. Effective batch=32 too aggressive at lr=0.01. Same theme as block_size=1024 — larger-batch NaN. |

### Warmup fraction — at new baseline

| Run | warmup_fraction | Folder | BPB (sliding) | Params | Time | Delta | Notes |
|-----|-----------------|--------|---------------|--------|------|-------|-------|
|   | 0.1 | [link](logs/wikitext-103_2026-04-18_23-49-03/log.txt) | NaN (3.5220) | ~840M | 1.61h | — | ❌ NaN'd. LR ramped faster than the model could absorb. |
|   | 0.3 | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1166 | ~840M | 1.85h | | Baseline (new baseline probe) |
|   | 0.5 | [link](logs/wikitext-103_2026-04-19_01-27-15/log.txt) | 1.1204 | ~840M | 1.79h | +0.0042 | Slightly worse — over-cautious LR ramp means less effective training at peak. 0.3 is the sweet spot. |

### Grad clip — at new baseline

| Run | grad_clip | Folder | BPB (sliding) | Params | Time | Delta | Notes |
|-----|-----------|--------|---------------|--------|------|-------|-------|
|   | 0.5 | [link](logs/wikitext-103_2026-04-19_03-16-59/log.txt) | 1.1161 | ~840M | 1.82h | -0.0001 | Essentially tied with baseline. Tighter clipping neither helps nor hurts at lr=0.01. |
|   | 1.0 | [link](logs/wikitext-103_2026-04-17_03-54-03/log.txt) | 1.1166 | ~840M | 1.85h | | Baseline (new baseline probe) |
|   | 2.0 | [link](logs/wikitext-103_2026-04-19_05-08-30/log.txt) | 1.1238 | ~840M | 1.81h | **+0.0076** | Looser clipping actively hurts — larger allowed gradient norms let occasional spikes corrupt the optimizer state. Confirms `grad_clip=1.0` is optimal, not just conservative. |

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
|   | L=2, C=2048, MLP=20, PLE, PKM+FwPKM-16384, lr=0.01, block_size=256, grad_accum=1, levels=5, low_rank=4, **per_scale_mixer_widths**=[1,1,1,0.5,0.5,0.5], **cross_scale_gating**, **wavelet_crawl K=3**, **shared_lifting_weights**, 5ep, 2.0x dropout (emb=0.2, proj=0.1, mixer=0.1, mlp=0.1, lm=0.24) | [link](logs/wikitext-103_2026-04-19_13-16-24/log.txt) | **1.0201** (post-fix; was 1.0219 pre-fix) | 882.51M | 18,235 MiB | 5,726 MiB | 15.79h | Non-overlapping BPB ~1.0386. Sliding-window PPL 24.21. Eval-fix correction (cross-window state disable + final-batch inclusion) shaved 0.0018 BPB. Headline run going into release. |

**Why the projection overshot.** The 94%-linear-stacking estimate was built from 1-epoch deltas. Several of those wins (especially `block_size=256` at -0.0140) come from "more gradient updates per epoch" — but at 5 epochs, the levels=9 baseline has also had time to converge. Short-training advantages compress as training continues; additivity at 1 epoch ≠ additivity at 5.

**Overfit signal.** Final train loss 2.6492 / val loss 3.1925 → **~0.78 bit/token train-val gap**. Best val (3.1728) occurred mid-epoch-5; end-of-run val regressed slightly, indicating the model started drifting past its peak. 2.0× dropout is insufficient regularization for this capacity at 5 epochs, and the gap will widen further at 10 epochs if dropout isn't scaled.

**Dropped features (ablation record).** Untied reconstruction (1-epoch tie at +168M params, +3% time), multi-basis lifting (NaN even with tightened init), exp_param at lr=0.02 (NaN at L=2), iterative refinement, cross-time feedback, stability-parametrization bundle.

### Weight decay spot-check (low values): 5 epochs, best architecture

| Run | WD | Folder | BPB (sliding) | PPL (sliding) | Best val loss | Params | Time | Notes |
|-----|-----|--------|---------------|---------------|----------------|--------|------|-------|
|   | 0 (baseline) | [link](logs/wikitext-103_2026-04-19_13-16-24/log.txt) | 1.0201 | 24.21 | 3.1728 | 882.51M | ~17h | 5-epoch best prior to WD probe |
|   | 1e-6 | [link](logs/wikitext-103_2026-04-22_01-36-47/log.txt) | **1.0140** | **23.7490** | **3.1593** | 882.51M | ~17h | **New best.** −0.0061 BPB / −0.46 PPL vs WD=0 baseline. WD=1e-6 adopted for 3-seed. |

### Decompose-bypass data-dependent EMA probe

| Run | Config | Folder | BPB (sliding) | PPL (sliding) | Val loss (epoch 1) | Params | Time | Notes |
|-----|--------|--------|---------------|---------------|--------------------|--------|------|-------|
| EMA 1-epoch smoke test | `decompose_bypass_ema=true`, 1 epoch | [link](logs/wikitext-103_2026-04-21_22-05-15/log.txt) | 1.1102 | 32.0731 | 3.4580 | ~848M | 3.48h | 1-epoch val-loss gain of −0.30 nats vs no-EMA baseline (3.7640). |
| EMA 5-epoch full run | `decompose_bypass_ema=true`, 5 epochs | [link](logs/wikitext-103_2026-04-22_18-42-22/log.txt) | 1.0226 | 24.3993 | — | ~848M | ~17h | **Regressed vs 5-epoch no-EMA baseline (1.0201). Rejected.** 1→5 epoch inversion; investigation plan in [plans/ema_post_release.md](plans/ema_post_release.md). |

### 3-seed variance study: L=2, C=2048, 2.0x dropout, 5 epochs

Locked recipe: L=2, C=2048, MLP=20, PLE, PKM+FwPKM=16384, lr=0.01, block_size=256, grad_accum=1, levels=5, low_rank=4, per_scale_mixer_widths, cross_scale_gating, wavelet_crawl K=3, shared_lifting_weights, eval_interval=250, **5 epochs, 2.0× dropout, WD=1e-6**. Seeds: 1337, 42, 7.

| Run | Seed | Folder | BPB (sliding) | PPL (sliding) | Best val loss | Params | Time | Notes |
|-----|------|--------|---------------|---------------|----------------|--------|------|-------|
|   | 1337 | [link](logs/wikitext-103_2026-04-22_01-36-47/log.txt) | **1.0140** | **23.7490** | 3.1593 | 882.51M | ~17h | Primary; best of 3 seeds. Promoted to PG-19 and HF upload. |
|   | 42   | [link](logs/wikitext-103_2026-04-23_12-43-30/log.txt) | 1.0155 | 23.8604 | 3.1757 | 882.51M | ~17h | +0.0015 BPB / +0.11 PPL vs seed 1337. |
|   | 7    | [link](logs/wikitext-103_2026-04-24_05-02-29/log.txt) | 1.0152 | 23.8438 | 3.1626 | 882.51M | ~17h | +0.0012 BPB / +0.09 PPL vs seed 1337. |

**Mean BPB: 1.0149 ± 0.0008 | Mean PPL: 23.82 ± 0.06 | Mean best val loss: 3.166 ± 0.009** (sample std across 3 seeds, n=3). Tight cluster — ~0.08% CV on BPB, ~0.25% on PPL. Confirms the WD=1e-6 recipe is not seed-1337-specific.

### PG-19 pre-release benchmark: best seed, 1 epoch

Final pre-release run. Takes the best seed from the 3-seed WT103 study and trains it on PG-19 for 1 epoch. Purpose: publish a second-dataset number alongside WikiText-103 so the release isn't single-benchmark-only, and anchor against Transformer-XL / Compressive Transformer, both of which have reported PG-19 numbers.

**Tokenizer:** 32K SentencePiece BPE trained on the PG-19 train split (matches Transformer-XL / Compressive Transformer's tokenizer family). Auto-selected when `dataset=pg19` is set in config — first run trains the SP model and caches it at `.cache/pg19_sp32k.model`; subsequent runs reuse it. See [train.py](../train.py) `train_pg19_sentencepiece()`. Note: embedding and LM head shapes scale with vocab size, so the PG-19 model has ~808M params vs WT-103's 883M (−74.8M from the embedding + LM head going from 50K to 32K vocab).

**Why 1 epoch is enough:** PG-19 is ~2B tokens under SentencePiece, ~15–20× WikiText-103's training corpus. Each token is seen roughly 15–20× less often than in a 20-epoch WT103 run, so convergence is driven by seeing new data rather than re-seeing the same data. Long-form narrative structure is also less diverse in surface form than Wikipedia — the model shouldn't need many passes to absorb distribution.

**Config:** identical to the 3-seed best run (L=2, C=2048, MLP=20, PLE, PKM+FwPKM=16384, lr=0.01, block_size=256, grad_accum=1, levels=5, low_rank=4, per_scale_mixer_widths, cross_scale_gating, wavelet_crawl K=3, shared_lifting_weights, eval_interval=250, 2.0× dropout, WD=1e-6). `epochs=1`, `dataset=pg19`, tokenizer auto-resolves to SentencePiece. **~808M params** (reduced from 883M due to 32K vs 50K vocab in the embedding and LM head).

**Block_size tradeoff:** PG-19 benefits from longer context in principle (long narrative arcs), but keeping `block_size=256` preserves apples-to-apples with WT103 at the same compute cost per step. Raising to 512 or 1024 is a post-release experiment.

**Estimated runtime:** ~63h (~2.6 days) on 5090 using WT103's 3h/epoch × ~21× token ratio. Schedule at current MBS=8/GA=1: ~1.2M steps/epoch, warmup ~360k steps (30%). **Realized: 63.77h.**

| Run | Seed | Folder | BPB (sliding) | Perplexity | BPB (non-overlap) | PPL (non-overlap) | Params | Time | Notes |
|-----|------|--------|---------------|------------|-------------------|-------------------|--------|------|-------|
|   | 1337 | [link](logs/pg19_2026-04-25_13-34-46/log.txt) | **1.0853** | **27.4010** | 1.1054 | 29.1302 | 807.73M | 63.77h | Pre-release anchor on PG-19; SentencePiece 32K tokenizer; beats Perceiver AR (28.9), Block-Recurrent Transformer (29.0), Compressive Transformer (33.6), and Transformer-XL (36.3) at 1 epoch / `block_size=256`. Best val loss 3.5238 at step 1.255M / 1.273M. |

**Result note:** the in-process post-training benchmark on this run was initially affected by a checkpoint-loader bug ([`save_with_retry`](../train.py)'s wrapper format vs. the loader's bare-state-dict assumption + `strict=False` masking the failure). Numbers above are from the corrected benchmark re-run via `benchmark_only=true` after the loader fix. The model itself trained correctly — see the [generation samples](logs/pg19_2026-04-25_13-34-46/generations.txt) for register-quality verification.

After this run completes, release proceeds: HuggingFace upload of the best checkpoint (with HF model card), and the post-release items (model comparisons, dataset comparisons, scaled-up B200) follow.

### Post-training quantization (PTQ): inference-only, applied to best checkpoint

Per-scale mixed precision leveraging WaveletLM's wavelet decomposition. Coarse scales (high-level semantics) get more bits; fine scales (local detail) tolerate aggressive quantization. All runs use the same trained checkpoint — no retraining needed.

**Baseline checkpoint:** the 5-epoch best run above (L=2, C=2048, MLP=20, PLE, PKM+FwPKM-16384, lr=0.01, block_size=256, grad_accum=1, levels=5, low_rank=4, per_scale_mixer_widths, cross_scale_gating, wavelet_crawl K=3, shared_lifting_weights, 2.0x dropout). Previous PTQ baseline was the levels=9 checkpoint at BPB 1.0247; switching to the consolidated best-run checkpoint once it completes.

> **Note on compression ratios.** The current impl stores sub-8-bit weights as int8 (one value per byte) — no bit-packing yet. So every "enabled" variant produces the same physical size per quantized component regardless of bit-width: uniform 8-bit and uniform 4-bit both give 1.95× compression, and mixed-precision only varies ratio by **which** components are kept at fp16. Proper bit-packing would multiply the compression wins proportionally (uniform 4-bit → ~4×, uniform 2-bit → ~8×).

### PTQ: Uniform quantization (all components same bits)

Baseline: 5-epoch best run, BPB 1.0219 (sliding). Peak inference VRAM at fp16: 4,918 MiB.

| Run | Bits | Folder | BPB (sliding) | Δ BPB | Size (MiB) | Inf. VRAM | tok/s | Notes |
|-----|------|--------|---------------|-------|------------|-----------|-------|-------|
|   | 16 (baseline) | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/01_baseline_fp16.log) | 1.0201 | — | 1,684* | 4,918 MiB | 24.2054 | fp16 reference |
|   | 8 uniform | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/02_uniform_8bit.log) | **1.0201** | **+0.0001** | 1,726 | 4,408 MiB | 24.2054 | Near-lossless drop-in. Lifting kept at 16. |
|   | 4 uniform | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/03_uniform_4bit.log) | 1.0201 | +0.1729 | 1,726 | 4,408 MiB | 24.2054 | **Catastrophic** — coarse-mixer-at-4 is the cliff. |

\*fp16 reference size = 882.51M × 2 bytes ≈ 1,684 MiB; the quantized sizes above exclude the dequantize-to-fp16 working buffer.

### PTQ: Per-scale mixed precision quantization

Mixed precision by component, leveraging the wavelet-scale structure (coarse = semantics, fine = detail).

| Run | Mixer c/m/f | MLP | Lifting | Emb | Folder | BPB (sliding) | Δ BPB | Size (MiB) | Inf. VRAM | tok/s | Notes |
|-----|-------------|-----|---------|-----|--------|---------------|-------|------------|-----------|-------|-------|
|   | 8/4/2 | 4 | 16 | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/04_mixed_default.log) | 1.0201 | +0.0083 | 1,726 | 4,408 | 24.2054 | Default mixed config |
|   | **8/8/4** | 4 | 16 | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/05_mixed_conservative.log) | **1.0201** | **+0.0027** | 1,726 | 4,408 | 24.2054 | **Conservative fine scales — best quality-preserving mixed variant** |
|   | 8/4/2 | 8 | 16 | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/06_mixed_higher_mlp.log) | 1.0201 | +0.0049 | 1,726 | 4,408 | 24.2054 | Higher MLP precision |
|   | 8/4/2 | 4 | 8 | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/07_mixed_quant_lifting.log) | 1.0201 | +0.0089 | 1,487 | 4,408 | 24.2054 | Quantize lifting too; 2.26× ratio |
|   | 4/4/2 | 4 | 8 | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/08_mixed_aggressive.log) | 1.0201 | +0.1739 | 1,487 | 4,408 | 24.2054 | **Catastrophic** — coarse mixer at 4-bit breaks it |
|   | 8/4/2 | 4 | 16 | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/09_mixed_aggressive_emb.log) | 1.0201 | +0.0120 | 1,726 | 4,408 | 24.2054 | Aggressive embedding |

### PTQ: Component isolation (quantize one component; keep the rest at fp16)

Measures per-component sensitivity. Surfaces the key finding: **coarse mixer scales are the 4-bit cliff**; MLP, embedding, and lifting all tolerate 4-bit with negligible BPB loss.

| Run | Component | Bits | Folder | BPB (sliding) | Δ BPB | Size (MiB) | Inf. VRAM | tok/s | Notes |
|-----|-----------|------|--------|---------------|-------|------------|-----------|-------|-------|
|   | Mixer only | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/10_mixer_only_8.log) | 1.0201 | +0.0001 | 3,371 | 4,499 | 24.2054 | Mixer at 8-bit: safe |
|   | Mixer only | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/11_mixer_only_4.log) | **1.0201** | **+0.1613** | 3,371 | 4,499 | 24.2054 | **Mixer at 4-bit alone breaks BPB** |
|   | MLP only | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/12_mlp_only_8.log) | 1.0201 | 0.0000 | 2,567 | 4,251 | 24.2054 | MLP at 8-bit: lossless |
|   | MLP only | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/13_mlp_only_4.log) | 1.0201 | +0.0026 | 2,567 | 4,251 | 24.2054 | MLP tolerates 4-bit — biggest bit-packing opportunity (76% of layer params) |
|   | Embedding only | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/14_embedding_only_8.log) | 1.0201 | 0.0000 | 3,034 | 4,892 | 24.2054 | Embedding at 8-bit: lossless |
|   | Embedding only | 4 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/15_embedding_only_4.log) | 1.0201 | +0.0014 | 3,034 | 4,892 | 24.2054 | Embedding tolerates 4-bit |
|   | Lifting only | 8 | [link](logs/wikitext-103_2026-04-19_13-16-24/ptq/16_lifting_only_8.log) | 1.0201 | +0.0004 | 3,383 | 4,519 | 24.2054 | Lifting at 8-bit: near-lossless |

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

---

## Post-release

The runs below were conducted after the v1 release of WaveletLM. Comparisons across this boundary should be made carefully — the pre-release L=1 entries (sections 14-16) used the v1 baseline (no `per_scale_mixer_widths`, no `cross_scale_gating`, no `wavelet_crawl`, no `decompose_bypass_cross_window`, smaller dropout values, 1-epoch training). Their numbers are not directly comparable to the post-release L=1 series here.

### Layers = 1 ablation series: modern stack (C=2048, MLP=20, full feature set)

Tests whether L=1 with the full modernized feature stack and longer training closes the gap to the L=2 / 5-epoch baseline (BPB sliding 1.0140). See [`plans/single_layer_waveletlm.md`](plans/single_layer_waveletlm.md) for motivation, hypothesis, and projection ranges.

| Run | Layers | Epochs | Folder | BPB (sliding) | PPL | Params | Notes |
|-----|--------|--------|--------|---------------|-----|--------|-------|
| A | 1 | 1 | [link](logs/wikitext-103_2026-04-29_20-45-37/log.txt) | 1.1648 | 38.04 | 586.15M | Modernized L=1 baseline at 1 epoch |
| B | 2 | 1 | [link](logs/wikitext-103_2026-04-29_22-52-28/log.txt) | 1.1129 | 32.35 | 882.51M | L=2 head-to-head at fixed epochs |
| C | 1 | 5 | [link](logs/wikitext-103_2026-04-30_02-20-35/log.txt) | 1.0809 | 29.28 | 586.15M | Headline test: L=1 + full training; best val at epoch 4 (overfit by ep 5) |
| D | 2 | 5 | [link](logs/wikitext-103_2026-04-22_01-36-47/log.txt) | 1.0140 | 23.75 | 882.51M | Existing baseline (no re-run) |
| E | 1 | 8 | [link](logs/wikitext-103_2026-04-30_12-20-45/log.txt) | 1.0715 | 28.43 | 586.15M | Compute-equalized to D (15.86h vs D's 16.25h); best val at epoch 8; min train 2.5984 (within 0.035 nats of D's 2.6330) |

**Comparison logic** (per the plan):
- (A) vs (B) isolates L=1 vs L=2 architectural difference at fixed epochs.
- (A) vs (C) isolates 1-epoch vs 5-epoch contribution at fixed L=1 architecture.
- (C) vs (D) is the headline. If (C) is within ~0.05 BPB of (D), L=1 is a viable lightweight variant.
- (E) vs (D) is the compute-equalized comparison: same wall-clock budget, L=1 trades depth for ~60% more epochs.

**Notes on Run A vs the prior v1 L=1 baseline:** The pre-release L=1 result of BPB 1.1538 (2026-04-11) was on the v1 baseline; that number should not be read as a direct comparator to Run A's 1.1648. The v1 stack lacked refined dropout, cross-scale gating, wavelet crawl, and other features that have been tuned at L=2 since.

**Memorization-floor finding (E vs D):** At matched compute (~16h wall-clock), Run E (L=1, 8ep) and Run D (L=2, 5ep) reach **nearly identical training-loss minimums** — 2.5984 (L=1) vs 2.6330 (L=2). L=1 actually edges out L=2 by 0.0346 nats on the lowest training step seen. The val loss minimums diverge by 0.1457 nats (L=1: 3.3050; L=2: 3.1593), with L=1 generalizing worse despite essentially-identical training-data fit. The full 0.146 nat val gap is therefore generalization difference, not capacity difference. Reading: depth in WaveletLM functions as implicit regularization at this dataset/scale, not as additional asymptotic capacity. See [plans/findings.md](plans/findings.md#single-layer-waveletlm-equal-compute-analysis) for the full analysis.

### Combined parameter reduction (Test 1, baseline reduction)

Tests the four cheap reductions from `plans/other_post_release_plans.md` §8 applied to the L=1 / E=5 iteration platform:
- `mlp_expansion: 20 → 10`
- `pkm_enabled: true → false` (PKM dropped; FwPKM retained for inference-update potential)
- `fwpkm_num_keys: 16384 → 8281` (≈ half, perfect square)
- `tie_embedding_to_lm_head: false → true`

| Run | Recipe | Folder | BPB sliding | PPL sliding | Best val | Min train | Train/val gap | Params | Train time | VRAM |
|-----|--------|--------|-------------|-------------|----------|-----------|---------------|--------|------------|------|
| Baseline (Run C, unreduced L=1 / E=5) | Default L=1 stack | [link](logs/wikitext-103_2026-04-30_02-20-35/log.txt) | 1.0809 | 29.28 | 3.3275 | 2.8292 | 0.498 | 586.15M | 9.74h | — |
| Test 1 (combined reduction) | MBS=8, eval=250 (4 reductions) | [link](logs/wikitext-103_2026-05-01_06-33-48/log.txt) | **1.0796** | **29.15** | 3.3341 | 2.9649 | 0.369 | 344.63M | 7.69h | 6.9 GiB |
| Test 2 (Max EBS) | MBS=64, eval=250 | [link](logs/wikitext-103_2026-05-01_14-54-13/log.txt) | 1.0860 | 29.75 | 3.3746 | 3.0189 | 0.356 | 344.63M | 5.60h | 25 GiB |
| Test 2b (Max EBS + finer eval) | MBS=64, eval=32 | [link](logs/wikitext-103_2026-05-01_20-46-54/log.txt) | 1.0888 | 30.00 | 3.3532 | 2.9979 | 0.355 | 344.63M | 5.60h | 25 GiB |
| Test 3 (Larger block_size) | MBS=8, bs=1024, eval=250 | [link](logs/wikitext-103_2026-05-02_02-24-23/log.txt) | 1.0880 † | 29.93 † | **3.3390** | **2.9874** | **0.352** | 344.63M | **5.96h** | **14 GiB** |
| Test 4 (Min EBS + max bs) | MBS=1, bs=16384, eval=250 | [link](logs/wikitext-103_2026-05-02_09-04-39/log.txt) | 1.1149 † | 32.55 † | 3.4170 | 3.1071 | 0.310 | 344.63M | 5.76h | 21.6 GiB |
| ΔTest 1 vs baseline | — | — | −0.0013 | −0.13 | +0.007 | +0.136 | −0.130 (−26%) | **−41.2%** | **−21%** | — |
| ΔTest 2 vs Test 1 | — | — | +0.0064 (4.3σ) | +0.60 | +0.041 | +0.054 | -0.013 | — | — | +260% |
| ΔTest 2b vs Test 1 | — | — | +0.0092 (6.1σ) | +0.85 | +0.019 | +0.033 | -0.014 | — | — | +260% |
| ΔTest 3 vs Test 1 | — | — | +0.0084 † | +0.78 † | **+0.005** (~1σ) | +0.022 | -0.017 (-5%) | — | **−22%** | +103% |
| ΔTest 4 vs Test 1 | — | — | +0.0353 † | +3.40 † | +0.083 | +0.142 | **−0.059 (−16%)** | — | **−25%** | +213% |

(Noise floor ±0.0015 BPB from 3-seed variance study at the L=2 baseline.)

**† Important methodology caveat for Test 3 / Test 4 BPB:** the benchmark runs at the model's training block_size, so Test 3 evaluates on 280 windows of length 1024 (stride 512) and Test 4 evaluates on **34 windows of length 16384 (stride 8192)**, while Tests 1/2/2b evaluate on 2246 windows of length 256 (stride 128). The BPB comparison is **not apples-to-apples** across different block_sizes. Best val loss IS apples-to-apples (same val set, same prediction methodology). To make a defensible BPB claim across runs, re-evaluate Test 3's and Test 4's checkpoints at bs=256 stride=128 in `benchmark_only` mode.

**Headline finding 1 — parameter reduction at L=1 is at-least-equivalent in BPB.** Test 1 vs unreduced: Δ = −0.0013 BPB, which is within ±0.0015 single-seed noise — statistically indistinguishable on BPB at 41% fewer parameters and 21% less wall-clock. The §8 plan projected +0.025 BPB cost; actual is essentially zero cost, much better than projected. The mechanism is implicit regularization: L=1 was regularization-bound, removing 42% of parameters removed spare memorization capacity without losing generalization-relevant capacity.

| Min-train comparison | Unreduced | Reduced (Test 1) | Δ |
|---|---|---|---|
| Min train loss | 2.8292 | 2.9649 | +0.136 (less memorization) |
| Best val loss | 3.3275 (ep4) | 3.3341 (ep5) | +0.007 (~equal) |
| Train/val gap | 0.498 | **0.369** | **−26%** |

The reduced model has materially less memorization capacity (train floor +0.136 nats higher), but essentially the same val loss, with a 26% smaller train/val gap — implicit-regularization-via-parameter-reduction realizing exactly as predicted.

**Headline finding 2 — increasing EBS hurts L=1 at matched compute (gradient-noise hypothesis confirmed by replication).** Tests 2 and 2b (both MBS=64, varying only eval_interval) both regressed comfortably outside the noise band:

- Test 2 (eval=250): +0.0064 BPB vs Test 1 = **4.3σ**
- Test 2b (eval=32): +0.0092 BPB vs Test 1 = **6.1σ**
- Test 2 vs Test 2b (different eval, same EBS): +0.0028 BPB ≈ 1.9σ — within noise of each other

Two independent runs at MBS=64 both significantly worse than MBS=8, in the same direction, with consistent magnitude. The eval-coarseness alternative explanation is ruled out: Test 2b with finer eval (8× more frequent) didn't close the gap to Test 1 — it slightly widened it. The freed VRAM from parameter reduction should NOT be spent on larger EBS for L=1.

**Methodology note (subtle):** Test 2b's BPB is *slightly worse* than Test 2's despite finer eval (1.0888 vs 1.0860, ~1.9σ gap). This is consistent with selection-bias-on-noisy-val-minima: with 8× more eval points (1140 vs 145 across 5 epochs), the "best val" checkpoint is more likely to be selected at a noisy lucky dip in val that doesn't generalize as well to test. Confirms your earlier instinct that constant `eval_interval` across configs is methodologically cleaner than scaling — finer eval can degrade test-set selection by amplifying val-noise sampling.

**Headline finding 3 — longer block_size is the right way to spend freed VRAM.** Test 3 (MBS=8, bs=1024) achieved best val loss 3.3390 vs Test 1's 3.3341 (Δ = +0.005 nats, ~1σ on BPB scale — **essentially tied**), with:

- **Wall-clock −22%** (5.96h vs 7.69h) — bs=1024 amortizes per-step overhead better in WaveletLM's wavelet+FWHT pipeline than smaller block_size
- **VRAM ~half** (~14 GiB vs Test 2's ~25 GiB at MBS=64) — longer context is more VRAM-efficient than larger batch
- **Train/val gap shrinks 5%** (0.352 vs Test 1's 0.369) and lowest min train loss of any test so far (2.9874) — slightly tighter generalization per nat of training capacity

In val terms (apples-to-apples, since BPB across different bs is not directly comparable), Test 3 is meaningfully **better than both MBS=64 variants**:
- Test 3 vs Test 2: Δval = −0.036 nats (~7σ on BPB scale)
- Test 3 vs Test 2b: Δval = −0.014 nats (~3σ on BPB scale)

**The framework consolidates: longer context preserves quality at less wall-clock and less VRAM; larger batch hurts quality.** For regularization-bound L=1, the right way to use freed VRAM is *more per-example signal* (longer block_size), not *more parallel sequences* (larger MBS). Both spend the same VRAM budget; one preserves gradient noise (the regularizer) while gaining within-example signal, the other dilutes gradient noise.

**Likely under-realized potential:** Test 3 keeps `levels=5` from baseline, which means coarsest cell at bs=1024 represents 32 tokens (vs 8 tokens at bs=256). The wavelet pipeline isn't yet exploiting the additional coarse structure the longer context enables — see the README's "Per-Scale Configuration at Longer Block Size" section for the dependent follow-up sweep that's now justified by Test 3's results.

**Headline finding 4 — minimum-EBS + maximum-block-size at bs=16384 trains stably and *halves* generation VRAM per token of context.** Test 4 (MBS=1, bs=16384, 64× the baseline context) completed all 5 epochs without NaN at peak training VRAM 21.6 GiB and inference VRAM 7.78 GiB. Per-token VRAM efficiency at inference is the headline: relative to the bs=256 baseline's 3.22 GiB inference footprint, Test 4 supports **64× more context for ~2.4× the VRAM**, i.e. ~27× better VRAM-per-token-of-context. This is a real architectural win for long-context applications even before any quality discussion.

Quality-wise, val loss landed at **3.4170** (Δ = +0.083 vs Test 1, +0.078 vs Test 3). Modest absolute regression on val, but the train/val gap *shrunk* to **0.310** (vs Test 1's 0.369 and Test 3's 0.352) — the regularization-bound framework continues to predict the structural pattern. The most likely explanation for the val gap not improving despite more per-example signal: **Test 4 keeps `levels=5` from baseline.** At bs=16384, log2(bs)=14 puts the levels ceiling at 14, but levels=5 means the coarsest decomposition cell spans 16384/2^5 = 512 tokens — Test 4 is using its 64× context as fine-scale extension rather than capturing genuinely longer-range structure. The wavelet pipeline is materially under-resourced for this block_size; the "Per-Scale Configuration at Longer Block Size" sweep is even more load-bearing here than at bs=1024.

Test 4 BPB sliding (1.1149) is **not directly comparable** to bs=256 runs' BPB — only 34 windows of length 16384 vs 2246 windows of length 256 (see methodology caveat above). Re-evaluation at bs=256 stride=128 in `benchmark_only` mode would give the apples-to-apples number.

See [plans/findings.md](plans/findings.md#combined-parameter-reduction-better-than-free-at-l1) for the full analysis.

### Low-rank ablations: post-combined-reduction baseline (L=1, levels=7, epochs=1)

Re-test of `low_rank` at the current sweep regime (L=1 / levels=7 / bs=16384 /
per_scale_mixer_widths=[1,1,1,1,0.5,0.5,0.5,0.5], post-combined-reduction).
Motivated by the E5 finding that mixer-width expansion contributed nothing
measurable beyond the headline; rank may be the under-explored axis.
Each ablation only varies `low_rank`; all other settings match the R0 baseline.

| Run | low_rank | Folder | BPB (sliding) | Params | U/V correction params | Time | Train VRAM | Delta vs R0 | Notes |
|-----|----------|--------|---------------|--------|------------------------|------|------------|--------------|-------|
| R0  | 4 (baseline) | [link](logs/wikitext-103_2026-05-02_21-43-22/log.txt) | 1.2361 | 392.91M | 16K/scale × 8 = 128K   | ~70m | 3,162 MiB | — | Reference |
| R1/LR16  | 16   | [link](logs/wikitext-103_2026-05-05_07-07-45/log.txt) | **1.2342** | 393.21M | 256K/scale × 8 = 2.05M | 80m | — | **−0.0019** | **PASS / WINNER.** Stable; modest improvement, +0.30M params. The 1-epoch low_rank champion. |
| R1.5 | 32  | [link](logs/wikitext-103_2026-05-05_11-23-29/log.txt) | NaN at step 2250 | 393.60M | 512K/scale × 8 = 4.10M | — | — | — | NaN at peak lr=1.00e-2 (end of warmup). Stability boundary lives between low_rank=16 and 32. |
| R1.75 | 64 | — | — | — | — | — | — | — | **Cancelled.** R1.5 already destabilized at low_rank=32; 64 is further into the unstable region. |
| R2  | 128  | [link](logs/wikitext-103_2026-05-05_08-29-00/log.txt) | 13.7913 (NaN-affected) | 395.96M | 2.10M/scale × 8 = 16.78M | — | — | — | **Effective NaN.** Best Val Loss 4.987 at epoch end; checkpoint produces noise-level BPB. Diverged mid-run. |
| R3  | 1024 | [link](logs/wikitext-103_2026-05-05_09-47-01/log.txt) | NaN at step 2000 | ~527M | 16.78M/scale × 8 = 134M  | — | — | — | NaN at lr=9.12e-3, well into warmup peak. Capacity-matched to main mixer matrix destabilizes. |

**Conclusion:** `low_rank=16` is the stable improvement at this regime. The upper stability boundary lives between 16 and 32 at lr=1.00e-2 / Adagrad eps=2e-13.

**5-epoch confirmation (R1_5ep):** [link](logs/wikitext-103_2026-05-05_21-36-45/log.txt) — BPB sliding **1.0971** vs [5-epoch headline 1.0974](logs/wikitext-103_2026-05-03_02-13-07/log.txt) = **−0.0003** (within run-to-run noise). The 1-epoch -0.0019 advantage did not amplify with more training — a genuine capacity-related parameter would widen the gap with more steps, not close it. The 1-epoch win was most likely early-training expressivity that washes out once both ranks converge to their optima.

**Decision:** **revert to `low_rank=4`** as the default. +1.92M params for a 5-epoch tie isn't worth promoting; we promote on measurable BPB gains, not 1-epoch transients. **Retest conditions:** B200 scale-up (Cp=4096 reduces the U·V^T fraction of the mixer, may shift the optimum) or post-Muon (orthogonalized updates may make higher rank tractable, opening a regime past R1.5/R2/R3 destabilization). PG-19 alone unlikely to move it.

### Mixer width contractions: post-combined-reduction baseline (L=1, levels=7, epochs=1)

Per-scale mixer width contraction sweep. Same baseline as the low_rank table above. After E5 (uncompressed lifting + width=1.5 coarse) tracked the headline reference closely at 5 epochs, we tested the opposite direction: can the mixer be made *smaller* than baseline at no quality cost? `low_rank` held at the default 4.

| Run | per_scale_mixer_widths | Folder | BPB (sliding) | Params | Mixer total | Time | Train VRAM | Delta vs R0 | Notes |
|-----|------------------------|--------|---------------|--------|-------------|------|------------|--------------|-------|
| R0  | [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5] (baseline) | [link](logs/wikitext-103_2026-05-02_21-43-22/log.txt) | 1.2361 | 392.91M | 58.82M | ~70m | 3,162 MiB | — | Reference |
| W1  | [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05] | [link](logs/wikitext-103_2026-05-05_04-37-47/log.txt) | NaN | 339.53M | 5.44M | — | — | — | NaN at step 1250 (lr=5.7e-3). Extreme contraction destabilizes — proj_in's 10-20× crush from Cp=2048 to 205/102 channels likely cascades to fp16 saturation. |
| W2  | [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25] | [link](logs/wikitext-103_2026-05-05_05-48-40/log.txt) | **1.2437** | 369.80M | 35.70M | 78m | — | **+0.0076** | Marginal BPB cost vs default 1.2361. 5.9% smaller total, **39.3% smaller mixer**. Trained stably through warmup peak (lr=0.01) with no instability. Promote to default candidate; 5-epoch confirmation pending. |

**Width-floor finding:** the boundary between stable contraction and NaN lives somewhere between W2's coarse=0.5 / fine=0.25 and W1's coarse=0.1 / fine=0.05. Follow-up tightenings worth testing if W2 ships at 5 epochs:
- `[0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]` — 80% of W2
- `[0.3, 0.3, 0.3, 0.3, 0.15, 0.15, 0.15, 0.15]` — 60% of W2
- `[0.25, 0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.125]` — 50% of W2 (approaches W1 territory)

**Expansion direction confirmation (E5_5ep):** [link](logs/wikitext-103_2026-05-05_14-00-32/log.txt) — `per_scale_mixer_widths=[1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5]`, 5 epochs, BPB sliding **1.1037** vs [5-epoch headline 1.0974](logs/wikitext-103_2026-05-03_02-13-07/log.txt) = **+0.0063** (no improvement at +24% mixer params). Reproduces the prior 5-epoch E5 finding and confirms the broader pattern: at WikiText-103 / 5 epochs / ~400M params, the architecture is data-bottlenecked and added width contributes nothing measurable. Expansion direction shelved; contraction (W2) remains the preferred trajectory.

### Wavelet off-diagonal masking with top-k percent (in progress, L=1, levels=7, epochs=1)

| Run | Mask source | Off-diagonal density | Off-diagonal entries per Linear(2048, 2048) | Approx. lifting params | Folder | BPB (sliding) | Notes |
|-----|-------------|----------------------|----------------------------------------------|------------------------|--------|---------------|-------|
| **Reference floor** (M0 = A1) | — (pure diagonal) | 0% | 0 | 3.33M | [link](logs/wikitext-103_2026-05-04_16-22-02/log.txt) | **1.2860** | Pure-diagonal compression via `lifting_diaglowrank=True`. The "floor" the M-sweep aims to recover from. +0.0499 BPB vs uncompressed reference. |
| **Reference ceiling** (LR16) | — (uncompressed) | 100% | ~4.19M | 117.50M | [link](logs/wikitext-103_2026-05-05_07-07-45/log.txt) | **1.2342** | Uncompressed `Linear(C, C)` lifting at low_rank=16. The "ceiling" the M-sweep aims to reach. |
| M1  | magnitude_topk | 0.1%               | 4,192                                        | 0.23M (232,064)        | [link](logs/wikitext-103_2026-05-06_13-29-49/log.txt) | 1.2805 | Recovers ~10.6% of the 0.0518-BPB gap between pure diagonal (1.2860) and uncompressed reference (1.2342). Genuine Lottery Ticket / RIGL signal: 4,192 highest-magnitude off-diagonal positions per matrix already do measurable work on top of the diagonal. |
| M2  | magnitude_topk | 1%                 | ~41,922                                      | 1.29M (1,288,532)      | [link](logs/wikitext-103_2026-05-06_14-50-38/log.txt) | 1.2694 | Recovers ~32.0% of gap. Per-position marginal value: ~2.94×10⁻⁴ BPB-recovered/position over M1 (38K added positions for 0.0111 BPB), a 4.5× drop from M0→M1. |
| M3  | magnitude_topk | 5%                 | ~209,613                                     | 5.98M (5,983,852)      | [link](logs/wikitext-103_2026-05-06_16-10-39/log.txt) | 1.2524 | Recovers ~64.9% of gap, +32.9 pp over M2 — biggest jump yet. +0.0182 vs reference 1.2361. Curve is S-shaped, not pure diminishing-returns: accelerating in absolute pp/decade while per-position marginal value drops 2.9× from M1→M2 step (1.01×10⁻⁴ BPB/position). |
| M4  | magnitude_topk | 10%                | ~419,225                                     | 11.85M (11,853,017)    | [link](logs/wikitext-103_2026-05-06_17-30-43/log.txt) | 1.2438 | **Recovers 81.5% of gap, +0.0096 vs reference 1.2361.** First shippable compressed configuration: 89.9% lifting reduction (117.50M → 11.85M), 26.9% total model reduction (393.21M → 287.56M), at 1ep BPB cost of 0.0096 vs uncompressed. Per-position marginal value 4.10×10⁻⁵ (2.5× drop from M2→M3 step) — diminishing slope itself converging, not vanishing. |
| M1r | random_topk    | 0.1%               | 4,192                                        | 0.23M (232,064)        | [link](logs/wikitext-103_2026-05-06_18-50-49/log.txt) | 1.2825 | Recovers **6.8%** of gap (vs M1's 10.6%, +3.8 pp magnitude advantage). |
| M2r | random_topk    | 1%                 | ~41,922                                      | 1.29M (1,288,532)      | [link](logs/wikitext-103_2026-05-06_20-11-20/log.txt) | 1.2750 | Recovers **21.2%** of gap (vs M2's 32.0%, +10.8 pp magnitude advantage). |
| M3r | random_topk    | 5%                 | ~209,613                                     | 5.98M (5,983,852)      | [link](logs/wikitext-103_2026-05-06_21-31-41/log.txt) | 1.2633 | Recovers **43.8%** of gap (vs M3's 64.9%, +21.1 pp magnitude advantage). |
| M4r | random_topk    | 10%                | ~419,225                                     | 11.85M (11,853,016)    | [link](logs/wikitext-103_2026-05-06_22-51-34/log.txt) | 1.2560 | Recovers **57.9%** of gap (vs M4's 81.5%, +23.6 pp magnitude advantage). +0.0218 vs reference 1.2361 (notably worse than M4's +0.0096). Magnitude-vs-random gap **widens** monotonically with density (3.8 → 10.8 → 21.1 → 23.6 pp), opposite to my earlier "narrowing" prediction. Random_topk is NOT a portable substitute for magnitude_topk at production densities — magnitude pruning is doing genuine content-aware work that random cannot replicate. |

### Wavelet off-diagonal masking with structured variants (planned, L=1, levels=7, epochs=1)

| Run | Structure | Variant detail | Approx. params per Linear(2048, 2048) | Reduction vs full | Folder | BPB (sliding) | Notes |
|-----|-----------|----------------|----------------------------------------|-------------------|--------|---------------|-------|
| **Reference ceiling** (LR16) | none (uncompressed) | — | 4.19M | 0% | [link](logs/wikitext-103_2026-05-05_07-07-45/log.txt) | **1.2342** | Uncompressed `Linear(C, C)` at low_rank=16. The target the structured variants aim to match in BPB. |
| **Reference floor** (A1) | D + U·V^T (lifting_diaglowrank) | r ∈ {16, 64} mixed | 67-264K mixed | 97% | [link](logs/wikitext-103_2026-05-04_16-22-02/log.txt) | **1.2860** | Pure-diagonal-plus-low-rank compression. Structural-prior comparison anchor at A1-equivalent density (BD64 / MON64 sit at similar param counts and test whether structure-via-blocks-or-factors beats D + U·V^T at matched params). |
| T_upper | upper_triangular | — | 2.10M (2,098,176) | 50% | [link](logs/wikitext-103_2026-05-07_00-11-31/log.txt) | **1.2381** | **Recovers 92.5% of gap, +0.0039 vs reference 1.2342.** Lifting drops to 58.81M (-50%); total model 334.52M (-15%). |
| T_lower | lower_triangular | — | 2.10M (2,098,176) | 50% | [link](logs/wikitext-103_2026-05-07_01-31-42/log.txt) | **1.2380** | **Essentially identical to T_upper (1.2380 vs 1.2381) — the model has no measurable directional preference in channel flow.** Treat T_upper / T_lower as tied; they're the "triangular result" representative going forward. |
| BD64 | block_diagonal | block_size=64 (32 blocks) | 131K | 97% | [link](logs/wikitext-103_2026-05-07_02-52-12/log.txt) | **1.2613** | Recovers **47.7%** of gap (+0.0271 vs reference 1.2361). At A1-equivalent param count (3.73M vs A1's 3.33M, 96.83% lifting reduction) **essentially matches the per-density recovery of magnitude_topk** at 3% (linear interp of M2/M3 ≈ 48.5%). **Best per-parameter efficiency of any variant** (0.078M lifting params / pp recovered) — outperforms M3r random_topk at 5% (43.8% recovery, 5.98M params). Per-parameter efficiency is the more useful metric since structural variants don't require the 2-stage training that magnitude_topk does. |
| BD128 | block_diagonal | block_size=128 (16 blocks), 7.40M total | 264K | 94% | [link](logs/wikitext-103_2026-05-07_10-40-27/log.txt) | **1.2564** | Recovers **57.1%** of gap (+0.0203 vs reference 1.2361). **Confirms BAND ≈ BD at matched density:** BD128 lands within 0.0001 BPB of BAND64 (1.2563, 57.3%) at matched param count and matched ~6.3% density. Sits between BD64 (47.7%) and BD256 (67.8%) on the saturating BD curve, vindicating the prediction. **The "BAND > BD" effect from BAND256 vs BD256 was a param-count artifact, not a structural-prior effect.** |
| BD256 | block_diagonal | block_size=256 (8 blocks), 14.74M total | 524K | 87% | [link](logs/wikitext-103_2026-05-07_04-12-07/log.txt) | **1.2509** | **Recovers 67.8% of gap, +0.0148 vs reference 1.2361** — close to W2's +0.0076 BPB delta in the (Complete) Per-Scale Mixer Width Expansion result. BD saturates more slowly per density than magnitude_topk (3.05% → 12.46% added only 20.1 pp vs M-curve's ~50 pp for similar density change), because BD's "no cross-block interactions" structural ceiling becomes binding. **Leading portable production-default candidate** (no reference checkpoint, no mask storage, ~8× inference speedup via block-diagonal matmul). |
| BD512 | block_diagonal | block_size=512 (4 blocks), 29.42M total | 1.05M | 75% | [link](logs/wikitext-103_2026-05-07_12-00-36/log.txt) | **1.2444** | Recovers **80.3%** of gap (+0.0102 vs reference 1.2361). **Ties BAND256 (80.1%) at matched density** within 0.2pp. Beats my "~78%" prediction. With T_upper at 92.6% / 50% density, the gap above BD512 is exactly the cross-block-interaction ceiling — BD is bounded by its hard partitioning at any finite density. For production, BD512 is identical-recovery to BAND256 with a cleaner block-sparse storage format. |
| BAND32 | banded | bandwidth=32, 3.76M total | 134K | 97% | [link](logs/wikitext-103_2026-05-07_13-22-09/log.txt) | **1.2622** | Recovers **46.0%** of gap (+0.0261 vs reference 1.2361). **Underperforms BD64 (47.7%) by 1.7pp at matched density.** First data point where BD slightly beats BAND at matched density — BAND's soft bleeding may help less when bandwidth is very narrow. Per-param efficiency 12.23 pp/M, vs BD64's 12.79 pp/M. Confirms the BAND-BD parity holds across all four tested densities but isn't perfectly equal at the very low end. |
| BAND64 | banded | bandwidth=64, 7.34M total | 264K | 94% | [link](logs/wikitext-103_2026-05-07_05-32-41/log.txt) | **1.2563** | Recovers **57.3%** of gap (+0.0202 vs reference 1.2361). At 7.34M lifting (essentially identical to predicted BD128 at 7.40M) lands at predicted BD128 recovery (~58%) — confirming **BAND ≈ BD at matched param count**. 7.81 pp/M per-param efficiency, between BD64's 12.79 and BD256's 4.60. |
| BAND128 | banded | bandwidth=128, 14.33M total | 512K | 88% | [link](logs/wikitext-103_2026-05-07_14-42-59/log.txt) | **1.2508** | Recovers **68.0%** of gap (+0.0147 vs reference 1.2361). **Ties BD256 (67.8%) at matched density** within 0.2pp — confirms BAND ≈ BD at matched density at the production-relevant tier. Combined with BAND32 vs BD64 (BD by 1.7pp), BAND64 vs BD128 (tied), BAND128 vs BD256 (tied), BAND256 vs BD512 (BD by 0.2pp): **BAND ≈ BD across the full density range tested. The BAND-vs-BD choice is now an engineering tradeoff, not a recovery one.** |
| BAND256 | banded | bandwidth=256, 27.63M total | 1.05M | 75% | [link](logs/wikitext-103_2026-05-07_06-52-52/log.txt) | **1.2445** | **Recovers 80.1% of gap, +0.0084 vs reference 1.2361** — beats my "~76-79%" prediction and **essentially ties M4** (1.2445 vs M4's 1.2438, just +0.0007 difference) at 2.3× M4's lifting params. **BAND scales better than BD at high density** because it doesn't have BD's hard "no cross-block interactions" ceiling: as bandwidth grows, more cross-channel interactions are captured continuously. BAND256 (27.63M) gets +12 pp recovery over BD256 (14.74M) — i.e. doubling the params nearly doubles the recovery beyond BD's saturation point. **Now the leading portable production-default candidate** at BPB-equivalent-to-M4 level. |
| MON32 | monarch | nblocks=32, blocks of 64x64 + perfect shuffle | 198K | 95% | [link](logs/wikitext-103_2026-05-07_16-58-23/log.txt) | **1.2614** | Recovers **47.5%** of gap (+0.0253 vs reference 1.2361). At 5.56M lifting (~5% of full) recovers slightly less than BD64 (47.7% at 3.73M) and BAND32 (46.0% at 3.76M) at lower param counts — Monarch's R·P·L·P^T parameterization adds param overhead without buying recovery. Per-param efficiency 8.54 pp/M vs BD64's 12.79 pp/M. |
| MON64 | monarch | nblocks=64, blocks of 32x32 + perfect shuffle | 198K | 95% | [link](logs/wikitext-103_2026-05-07_09-31-54/log.txt) | **1.2615** | Recovers **47.3%** of gap. **Confirms Monarch param-symmetry**: shares 5.56M lifting with MON32 by construction (R·P·L·P^T param count = N·(n + N/n), symmetric around n=√N), and now also shares recovery within 0.2 pp. Block-shape choice (n=32 with b=64 vs n=64 with b=32) does not change effective capacity at this density. |
| MON128 | monarch | nblocks=128, blocks of 16x16 + perfect shuffle | 295K | 93% | [link](logs/wikitext-103_2026-05-07_18-07-19/log.txt) | **1.2584** | Recovers **53.3%** of gap (+0.0223 vs reference 1.2361). **Matched-param-count comparison vs BD128 (57.1%) and BAND64 (57.3%): Monarch underperforms by ~4 pp at the ~7-8M tier.** Per-param efficiency 6.41 pp/M, vs BD128's 7.72 pp/M and BAND64's 7.81 pp/M. The R·P·L·P^T factorization spends parameters on shuffle/factor machinery without buying recovery beyond what BD/BAND already capture. |
| MON256 | monarch | nblocks=256, blocks of 8x8 + perfect shuffle | 541K | 87% | [link](logs/wikitext-103_2026-05-07_19-18-23/log.txt) | **1.2533** | Recovers **63.1%** of gap (+0.0172 vs reference 1.2361). **Matched-param-count comparison vs BD256 (67.8%) and BAND128 (68.0%): Monarch underperforms by ~5 pp at the ~14-15M tier.** Confirms the structural hypothesis (Monarch captures interactions BD can't) is rejected at production density too. Per-param efficiency 4.15 pp/M vs BD256's 4.60 pp/M. **For portable structural compression, BD or BAND beats Monarch at every density tier tested.** |

### Sparse (p, q) phantom-token embedding striding (planned, L=1, levels=7, epochs=1)

Tests sparsifying the 102.93M token embedding via the (p, q) phantom-token striding scheme — see [plans/new_compression_ideas.md](plans/new_compression_ideas.md) for the full design and the [Sparse Embedding with (p, q) Striding](README.md#sparse-embedding-with-p-q-striding) Future Plans section in the README. Token embedding is the largest non-cascade block in the model; ~90% reduction at d=10% saves ~92.6M params, taking the whole model from 393M to ~300M before any lifting compression and ~210M when stacked with the lifting-cascade winner from the structured-variants sweep above. Default mode is `structural` (q closest to √C ≈ 45 — square macrocells, aligns with Monarch / butterfly philosophy). Compare 1-epoch BPB delta against the **reference row** (the best-chosen winner from the [Wavelet off-diagonal masking with structured variants](#wavelet-off-diagonal-masking-with-structured-variants-planned-l1-levels7-epochs1) sweep above, set after that 1-epoch sweep completes); a working scheme should land within ~0.005 BPB of that reference at d ∈ {5%, 10%, 25%}, with d ∈ {1%, 0.1%} probing where the inter-token bridging story breaks down (expected floor ~0.1% by Poisson tail analysis on feature-overlap).

| Run | Density | Mode | (p, q, N') | Phantom rows | Effective embedding params | % of full embedding | Folder | BPB (sliding) | Notes |
|-----|---------|------|------------|--------------|----------------------------|---------------------|--------|---------------|-------|
| **Reference (combined-reductions baseline @ 1ep, no embedding compression)** | — | — | — | — | 102.93M | 100% | [link](logs/wikitext-103_2026-05-08_07-19-49/log.txt) | **1.2586** | The 1-epoch baseline against which all (p, q) variants are compared. Configuration: `layers=1`, `levels=7`, `block_size=16384`, `low_rank=4`, `per_scale_mixer_widths=[0.5×4, 0.25×4]` (W2), `lifting_offdiag_structure="banded"`, `lifting_band_width=128` (BAND 128). Total 266.63M params, training VRAM 2,938 MiB. |
| PQ_EMB001 | 0.1% | structural | (1968, 32, 50272) | 15 | ~103K (0.10M) | 0.1% | | | **Stress test for the bridging hypothesis.** Below the d ≈ 0.1% floor where 2-hop bridging starts to fail by Poisson tail (E[overlap] = d²·C ≈ 0.002). Expected to fail or near-fail; result tells us where the scheme floor sits. |
| PQ_EMB01 | 1% | structural | (168, 32, 50272) | 15 | ~1.03M | 1% | | | At this density E[overlap] ≈ 0.2 features per random pair — single-shared-feature bridging, multi-hop required for most pairs. Tests whether the model can learn the bridging structure. |
| PQ_EMB05 | 5% | structural | (24, 16, 50272) | 15 | ~5.15M | 5% | | | E[overlap] ≈ 5 features per pair — direct bridging usually possible. Plausible production density if 10% lands within tolerance and 5% is needed for headroom. |
| PQ_EMB10 | 10% | structural | (12, 8, 50264) | 7 | ~10.29M | 10% | | | **Default candidate density.** E[overlap] ≈ 20 features per pair — 2-hop bridging essentially guaranteed (P[overlap=0] vanishingly small at λ=20). Storage win: ~92.6M params (90% of embedding) saved. |
| PQ_EMB25 | 25% | smallest_q | (6, 2, 50258) | 1 | ~25.73M | 25% | | | At s = p+q = 8, only q=2 is valid (q=4 makes p=4 divide 2048), so structural and smallest_q converge. Conservative density; if 10% is borderline, this is the production-safe fallback at 75% embedding reduction. |
| **PQ_EMB_winner_5ep** | TBD | TBD | TBD | TBD | TBD | TBD | | | 5-epoch confirmation of the best 1-epoch (p, q) variant. Compare against the 5-epoch combined-reductions baseline (section 11 placeholder, will run with same W2 + low_rank=4 + BAND 128 settings as section 7 but for 5 epochs). **Strongest result:** BPB cleanly below the 5-epoch combined-reductions baseline, suggesting embedding sparsity reduces overfitting at the embedding tier in the same way lifting sparsity does at the cascade tier. |

### Encoder-decoder embedding sweep (planned, L=1, levels=7, epochs=1)

Tests the encoder-decoder embedding architecture (V → C_emb via embedding lookup, C_emb → C via learnable decoder, [model interior at C], C → C_emb via learnable encoder, C_emb → V via tied embedding). The decoder and encoder are SEPARATE learnable matrices because the model's nonlinearities (GELU, gating, lifting cascade) make the output-side hidden a nonlinear transform of the input-side embedding — sharing the same matrix in transposed form would force a sub-optimal output compression. Implementation in [tools/encoder_decoder_embedding.py](tools/encoder_decoder_embedding.py).

**Parameter count formula** (V=50257, C=2048, biases included): `V·C_emb + 2·C·C_emb + C + C_emb`. Compare against dense `V·C = 102.93M`:

| Run | C_emb | Embedding params | Decoder params | Encoder params | Total | % of dense | Folder | BPB (sliding) | Notes |
|-----|------:|----------------:|---------------:|---------------:|------:|-----------:|--------|---------------|-------|
| **Reference (combined-reductions baseline @ 1ep, no embedding compression)** | C=2048 (dense) | 102.93M | — | — | 102.93M | 100% | [link](logs/wikitext-103_2026-05-08_07-19-49/log.txt) | **1.2586** | Section-7 baseline. Total 266.63M params, training VRAM 2,938 MiB. The reference all encoder-decoder embedding variants compare against. |
| ED128 | 128 | 6.43M | 0.26M | 0.26M | **6.95M** | **6.75%** | [link](logs/wikitext-103_2026-05-08_09-12-05/log.txt) | **1.4808** | +0.2222 vs CB (compressed baseline). Total 170.66M, train VRAM 22,012 MiB, inference VRAM 2,198 MiB (−27% vs CB — embedding compression moves inference VRAM substantially via the smaller (B,T,C_emb) lookup intermediate). 16× C/C_emb ratio is too aggressive; well past the elbow. |
| ED256 | 256 | 12.87M | 0.53M | 0.53M | **13.92M** | **13.53%** | [link](logs/wikitext-103_2026-05-08_11-54-53/log.txt) | **1.3916** | +0.1330 vs CB. Total 177.62M, train VRAM 22,087 MiB, inference VRAM 2,452 MiB (−19% vs CB). 8× C/C_emb ratio still meaningfully past the elbow. |
| **ED512** | **512** | **25.73M** | **1.05M** | **1.05M** | **27.83M** | **27.04%** | [link](logs/wikitext-103_2026-05-08_13-37-26/log.txt) | **1.3315** | +0.0729 vs CB. Total 191.53M, train VRAM 22,235 MiB, inference VRAM pending. 4× C/C_emb ratio — closing on the elbow but still meaningfully behind CB at 1ep. 73% embedding-param reduction. |
| ED1024 | 1024 | 51.46M | 2.10M | 2.10M | **55.66M** | **54.07%** | [link](logs/wikitext-103_2026-05-08_14-41-41/log.txt) | **1.2829** | +0.0243 vs CB. Total 219.36M, train VRAM 22,533 MiB, inference VRAM pending. 2× C/C_emb ratio — closest to CB yet, on the cusp of crossing. 46% embedding-param reduction. |
| ED2048 | C=2048 | 102.93M | 4.20M | 4.20M | **111.33M** | **108.16%** | [link](logs/wikitext-103_2026-05-08_15-47-44/log.txt) | **1.2608** | +0.0022 vs CB. Total 275.02M, train VRAM 23,128 MiB, inference VRAM pending. **No compression — full-rank refinement.** C_emb == C, so decoder/encoder are full C×C matrices acting as learnable affine projections. The encoder/decoder pair is **essentially free** at C_emb = C (no BPB improvement, no meaningful cost) — the +8.39M extra params learn close to identity. Architectural bonus hypothesis NOT confirmed; the pair does not add capacity at this width. C_emb > C now becomes the open question: does forcing the embedding through a C-bottleneck help? |
| ED4096 | 2× C=4096 | 205.85M | 8.39M | 8.39M | **222.63M** | **216.32%** | [link](logs/wikitext-103_2026-05-08_19-21-42/log.txt) | **1.2597** | +0.0011 vs CB; **−0.0011 vs ED2048** (closer to CB but still above). Total 386.34M, train VRAM 24,317 MiB, inference VRAM pending. **Expansion direction also neutral.** Doubling C_emb past C added 119.71M total params (+45% over CB's 266.63M) for an essentially noise-level BPB shift. The encoder/decoder + bottleneck arrangement neither helps nor hurts at this width — same verdict as ED2048. The cascade itself appears to be the binding constraint, not the embedding-side capacity feeding into it. |
| ED128_untied | 128 | 6.43M (×2 untied) | 0.26M | 0.26M | **13.38M** | **13.00%** | | | Untied LM head: separate V × C_emb output_embedding (+6.43M) replaces the tied V projection. Encoder shared. 87% reduction. |
| ED256_untied | 256 | 12.87M (×2 untied) | 0.53M | 0.53M | **26.79M** | **26.03%** | [link](logs/wikitext-103_2026-05-08_16-59-00/log.txt) | **1.3933** | +0.1347 vs CB; **+0.0017 vs ED256 tied** at +12.87M params. Total 190.49M, train VRAM 22,185 MiB. Untying the LM head at C_emb=256 does **not** recover the compression gap — essentially noise-level change vs tied. 74% reduction vs dense. |
| ED512_untied | 512 | 25.73M (×2 untied) | 1.05M | 1.05M | **53.56M** | **52.04%** | | | Untied LM head. 48% reduction vs dense. Direct comparison vs ED512 (tied, 27.83M) measures the cost of weight tying. |
| ED1024_untied | 1024 | 51.46M (×2 untied) | 2.10M | 2.10M | **107.13M** | **104.07%** | | | Untied LM head. Just over dense; the duplicated V × C_emb dominates. |
| ED2048_untied | C=2048 | 102.93M (×2 untied) | 4.20M | 4.20M | **214.26M** | **208.16%** | | | Untied LM head + full-rank refinement. Largest variant; functionally similar to dense untied (V·C input + V·C lm_head = 205.85M) plus the encoder/decoder pair (~8.4M). |
| **ED_winner_5ep** | TBD | TBD | TBD | TBD | TBD | TBD | | | 5-epoch confirmation of the best 1-epoch ED variant (tied or untied), layered on the combined-reductions baseline. Compare against the 5-epoch combined-reductions baseline. |

**Composability with other compression sweeps:** EncoderDecoderEmbedding can compose with the MLP structural compression sweep (section 10) — both target different parts of the model. The 5-epoch combined winner stack candidate is `(combined-reductions baseline) + ED<winner> + MLP_<winner>`, with FwPKM key pruning as the next standalone sweep after this round closes.

### MLP structural compression (planned, L=1, levels=7, epochs=1)

Tests three structural variants on the MLP weight matrices W1 (C, E·C) and W2 (E·C, C), using the existing `StructuredLinear` infrastructure plus a new `make_mlp_mask` helper in [tools/lifting_constraints.py](tools/lifting_constraints.py). At E=10 (1-epoch sweep), MLP is 83.91M of 281M total; ~10% density takes that to ~8.4M. Four density points (25% / 12.5% / 6.25% / 3.125%) × three structures (banded tiled / block_diagonal tiled / pq_strided) = 12 runs.

**Three structural variants:**
- **`banded`** — tiled per-(C, C)-block bilateral band of width W. Per-block density (2W+1)/C, identical to BAND on the lifting (C, C) matrix.
- **`block_diagonal`** — tiled per-(C, C)-block diagonal of size b. Per-block density b/C, identical to BD on lifting.
- **`pq_strided`** — single 1D walk over flattened (out · in) tensor with alternating p, q steps. Cross-pollinated from the embedding (p, q) scheme. q | C and p ∤ C, p ∤ E·C. Density 2/(p+q).

**Reference:** the **combined-reductions baseline** from section 7 of runs.sh (W2 mixer + low_rank=4 + BAND 128 lifting, no MLP compression). The MLP sweep layers on top of that for the production-stack comparison. The reference row in the table below holds the placeholder until the section-7 1-epoch run lands.

| Run | Structure | Variant detail | W1 effective | W1 % of dense (41.94M) | Folder | BPB (sliding) | Notes |
|-----|-----------|----------------|-------------:|----------------------:|--------|---------------|-------|
| **Reference (combined-reductions baseline @ 1ep, no MLP compression)** | — | — | 41.94M (uncompressed MLP) | 100% | [link](logs/wikitext-103_2026-05-08_07-19-49/log.txt) | **1.2586** | Baseline against which all MLP variants are compared. Configuration: `layers=1`, `levels=7`, `block_size=16384`, `low_rank=4`, `per_scale_mixer_widths=[0.5×4, 0.25×4]` (W2), `lifting_offdiag_structure="banded"`, `lifting_band_width=128` (BAND 128). Total 266.63M params, training VRAM 2,938 MiB. |
| **MLP_BAND25** | banded | bandwidth=256 per block, ~25% per-block density | 10.51M | 25.05% | | | High-density anchor for BAND. Should land closest to uncompressed MLP performance. |
| **MLP_BD25** | block_diagonal | block_size=512 per block, 25% per-block density | 10.49M | 25.00% | | | High-density anchor for BD. Direct comparison with MLP_BAND25 at matched density. |
| **MLP_PQ25** | pq_strided | (p=6, q=2) (only valid candidate at d=25%) | 10.49M | 25.00% | | | High-density anchor for (p, q). Note: at s=8 only q=2 is valid, so structural and smallest_q both pick (6, 2). |
| **MLP_BAND125** | banded | bandwidth=128 per block, ~12.5% per-block density | 5.27M | 12.55% | | | Mid-density. Banded keeps cross-channel bleed within ±128 positions. |
| **MLP_BD125** | block_diagonal | block_size=256 per block, 12.5% per-block density | 5.24M | 12.50% | | | Mid-density. BD256 lifting recovered 67.8% of gap; expect similar order on MLP. |
| **MLP_PQ125** | pq_strided | (p=12, q=4) | 5.24M | 12.50% | | | Mid-density. Two valid (p, q): structural picks (12, 4) (q closer to √C=45). |
| **MLP_BAND0625** | banded | bandwidth=64 per block, ~6.25% per-block density | 2.64M | 6.30% | | | Lower-mid density. BAND64 lifting recovered 57.3% of gap. |
| **MLP_BD0625** | block_diagonal | block_size=128 per block, 6.25% per-block density | 2.62M | 6.25% | | | Lower-mid density. Only 16 blocks per (C, C); cross-block isolation tighter. |
| **MLP_PQ0625** | pq_strided | (p=24, q=8) | 2.62M | 6.25% | | | Lower-mid density. Structural picks (24, 8). |
| **MLP_BAND03125** | banded | bandwidth=32 per block, ~3.17% per-block density | 1.33M | 3.17% | | | Aggressive density. BD64 lifting recovered 47.7% at this density. |
| **MLP_BD03125** | block_diagonal | block_size=64 per block, 3.125% per-block density | 1.31M | 3.125% | | | Aggressive density. 32 (b, b) blocks per (C, C). Same param count as BD64 lifting. |
| **MLP_PQ03125** | pq_strided | (p=48, q=16) | 1.31M | 3.125% | | | Aggressive density. Structural picks (48, 16) — q=32 fails (32 \| 2048). |
| **MLP_winner_5ep** | TBD | TBD | TBD | TBD | | | 5-epoch confirmation of the best 1-epoch MLP variant, layered on top of the combined-reductions baseline (W2 + low_rank=4 + BAND 128) plus the chosen embedding compression. Compare against the 5-epoch combined-reductions baseline. |

### Post-release: bit-packed PTQ kernels

Follow-up to the PTQ sweep above. The current `QuantizedLinear` / `QuantizedEmbedding` path stores int8 weights but dequantizes to fp16 inside `forward()`, then runs a standard fp16 matmul. That pays the dequant cost on every step without realizing the bandwidth win, which is why generation is 12% slower than fp16 despite the 1.95× storage compression — and why sub-8-bit variants (`03_uniform_4bit`, `08_mixed_aggressive`) compress identically to 8-bit on disk (one value per byte regardless of bit-width).

Replacing the dequant-then-matmul path with fused packed-weight kernels turns both problems around: storage actually scales with bit-width, and the matmul reads half (8-bit) or a quarter (4-bit) as many bytes per step, so generation is bandwidth-limited on a much smaller working set than fp16.

**Candidate kernels:**
- **Uniform 8-bit:** CUTLASS `i8gemm`, `bitsandbytes.matmul_8bit`, or Marlin's W8A16 path. Drop-in replacement for the current `QuantizedLinear`.
- **Mixed 8/4/2 (W4A16 / W2A16):** Marlin, AWQ kernels, or GPTQ-Triton. Requires per-scale bit-width metadata to be baked into the packed tensor layout.
- **Embedding:** int8/int4 gather kernels (bitsandbytes ships one; a small hand-written Triton kernel also suffices since the access pattern is just a table lookup).

**Expected generation tok/s (batch=1, 5090, fp16 baseline = 28.8 tok/s):**

| Path | Expected tok/s | vs fp16 | Expected size (MiB) | vs fp16 |
|------|----------------|---------|---------------------|---------|
| Current dequant-to-fp16 (uniform 8-bit) | 23.8 (measured) | 0.88× | 1,726 | 1.95× |
| Fused uniform 8-bit kernels | ~40–46 | 1.4–1.6× | 1,726 | 1.95× |
| Fused mixed 8/4/2 kernels (conservative: 8/8/4 MLP=4, emb=8, lift=16) | ~52–63 | 1.8–2.2× | ≤900 | ≥3.7× |

Numbers are estimates based on reported speedups for comparable kernels on 7B-class transformers at batch=1 (Marlin W4A16: 2.0–2.5× fp16; bitsandbytes W8A16: 1.5–1.8× fp16 on memory-bound layers). WaveletLM's high MLP expansion (20×) puts a larger share of weight bytes in quantizable linears than a typical transformer, so the ratio should track the upper end rather than the lower. Final number pending actual measurement.

**Caveats:**
- The LM head is the dominant cost at batch=1 (vocab=50,257 × C=2048). Whether it's quantized, and at what bits, drives a large fraction of the total speedup. The 8/8/4 conservative recipe keeps embedding at 8-bit, which would leave most of the head bandwidth unchanged from uniform 8-bit.
- FwPKM top-k gather is a table lookup, not a matmul — it does not benefit from fused quant kernels and stays bandwidth-bound on the value table regardless.
- The BPB numbers from the current PTQ sweep carry over unchanged: bit-packing is a *byte layout* change, not a numerical-precision change. `02_uniform_8bit` at BPB 1.0220 is the same 1.0220 once the kernel is swapped.

**Status:** Deferred to post-release. The "uniform 8-bit via a fused kernel" path is the lowest-risk / highest-value first target because it ships a concrete speedup + compression win without new BPB risk. Mixed-precision packing follows.

### Planned: model comparisons (WikiText-103, matched compute)

All models use the same GPT-2 tokenizer (tiktoken, 50,257 vocab), same dataset preprocessing, and same sliding window evaluation methodology. Competitors use all available optimizations (Flash Attention, torch.compile, KV cache, etc.) to ensure the comparison reflects each architecture's best-case performance.

| Model | Type | Params | BPB (sliding) | Train tok/s | Gen tok/s | Training time | Optimizations | Notes |
|-------|------|--------|---------------|-------------|-----------|---------------|---------------|-------|
| WaveletLM | Wavelet mixer | | | | | | torch.compile, fp16 | Best config from sweeps |
| GPT-2 | Transformer | | | | | | Flash Attention, KV cache, TurboQuant, torch.compile, fp16 | Matched compute |
| Mamba | SSM | | | | | | Mamba CUDA kernels, TurboQuant, torch.compile, fp16 | Matched compute |
| RWKV | Linear attention | | | | | | Custom CUDA kernels, TurboQuant, torch.compile, fp16 | Matched compute |

### Planned: dataset comparisons (B200, max EBS)

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
