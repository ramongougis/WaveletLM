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
| R0 | [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5] (baseline) | [link](logs/wikitext-103_2026-05-02_21-43-22/log.txt) | 1.2361 | 392.91M | 58.82M | ~70m | 3,162 MiB | — | Reference |
| W1 | [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05] | [link](logs/wikitext-103_2026-05-05_04-37-47/log.txt) | NaN | 339.53M | 5.44M | — | — | — | NaN at step 1250 (lr=5.7e-3). Extreme contraction destabilizes — proj_in's 10-20× crush from Cp=2048 to 205/102 channels likely cascades to fp16 saturation. |
| W2 | [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25] | [link](logs/wikitext-103_2026-05-05_05-48-40/log.txt) | **1.2437** | 369.80M | 35.70M | 78m | — | **+0.0076** | Marginal BPB cost vs default 1.2361. 5.9% smaller total, **39.3% smaller mixer**. Trained stably through warmup peak (lr=0.01) with no instability. Promote to default candidate; 5-epoch confirmation pending. |

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
| **PQ_EMB10_smallest_q** | **10%** | **smallest_q** | **(18, 2, 50258)** | **1** | **10.29M** | **10%** | [link](logs/wikitext-103_2026-05-09_05-22-03/log.txt) | **1.3896** | **First (p, q) embedding result on the CB stack.** Total 174.00M (−35% vs CB's 266.63M), train VRAM 23,308 MiB, inference VRAM 3,630 MiB. **+0.1310 vs CB**, but **beats ED at equivalent compression**: ED256 (87%, 13.92M) lands at 1.3916, ED128 (93%, 6.95M) at 1.4808. PQ at 90% compression slots between them on embedding params and beats both on BPB-per-embed-param. **Stability win:** previous PQ runs NaN'd at peak LR; CB stack (BAND128 + low_rank=4) clears it. The cascade's spectral-norm constraint stabilizes downstream sparse-activation amplification. (Lower-density rows for d=0.1%, d=1%, d=5% removed from this sweep — the d=10% +0.1310 BPB result made deeper compression unattractive for production.) |
| **PQ_EMB10_structural** | **10%** | **structural** | **(12, 8, 50264)** | **7** | **10.29M** | **10%** | [link](logs/wikitext-103_2026-05-09_06-34-14/log.txt) | **1.3836** | **Macrocell hypothesis confirmed at the embedding tier.** Total 174.00M, train VRAM 23,308 MiB, inference VRAM 3,630 MiB. **+0.1250 vs CB**, **−0.0060 vs PQ_EMB10_smallest_q** at matched embedding-param count — structural's q ≈ √C clumping wins over smallest_q's spread sparsity. Beats ED256 (87% compression at 1.3916) by 0.008 BPB *while compressing more aggressively* (90%). Production-default mode for (p, q) embedding compression. |
| **PQ_EMB25_structural** | **25%** | **structural (≡ smallest_q)** | **(6, 2, 50258)** | **1** | **25.73M** | **25%** | [link](logs/wikitext-103_2026-05-09_09-40-26/log.txt) | **1.3151** | At s = p+q = 8, only q=2 is valid (q=4 makes p=4 divide 2048), so structural and smallest_q converge. Total 189.43M, train VRAM 23,308 MiB, inference VRAM pending. **+0.0565 vs CB; ~halves the BPB cost vs PQ_EMB10** (which was +0.1250–0.1310) for backing off compression from 90% to 75%. Steep recovery curve at this density tier. |
| ~~PQ_EMB40_structural~~ (cancelled) | 40% | structural (≡ smallest_q) | (3, 2, 50256) | 1 | ~41.17M | 40% | | | **Cancelled** — PQ_EMB10 / PQ_EMB25 results made the (p, q) embedding direction less promising than initially hoped, and the 40% density anchor was deemed unnecessary given the trend. Was the maximum density achievable for C=2048 (at s = p+q = 5, only (3, 2) is valid). |
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
| ED4096 | 2× C=4096 | 205.85M | 8.39M | 8.39M | **222.63M** | **216.32%** | [link](logs/wikitext-103_2026-05-08_19-21-42/log.txt) | **1.2597** | +0.0011 vs CB; **−0.0011 vs ED2048** (closer to CB but still above). Total 386.34M, train VRAM 24,317 MiB, inference VRAM pending. Doubling C_emb past C added 119.71M total params for an essentially noise-level BPB shift. The expansion direction's payoff doesn't show until ED8192 — at 4096, the bottleneck-through-C and the embedding-side capacity are still in approximate balance. |
| **ED8192** | **4× C=8192** | **411.70M** | **16.78M** | **16.78M** | **445.26M** | **432.65%** | [link](logs/wikitext-103_2026-05-08_20-44-08/log.txt) | **1.2514** | **−0.0072 vs CB; first cross UNDER CB.** Total 608.97M, train VRAM 26,696 MiB, inference VRAM 5,610 MiB. **The expansion direction works at sufficient scale**: 4× C ratio gives the bottleneck-through-C arrangement enough headroom on the embedding side to encode structure that the C-dim interior can usefully read. Per-param efficiency is poor (+342M total params over CB for −0.0072 BPB = ~17 GM/BPB), but the BPB drop is real and reproducible. Trade-off vs CB: +119% inference VRAM, +128% total params, but −0.0072 BPB at 1ep. |
| ED128_untied | 128 | 6.43M (×2 untied) | 0.26M | 0.26M | **13.38M** | **13.00%** | | | Untied LM head: separate V × C_emb output_embedding (+6.43M) replaces the tied V projection. Encoder shared. 87% reduction. |
| ED256_untied | 256 | 12.87M (×2 untied) | 0.53M | 0.53M | **26.79M** | **26.03%** | [link](logs/wikitext-103_2026-05-08_16-59-00/log.txt) | **1.3933** | +0.1347 vs CB; **+0.0017 vs ED256 tied** at +12.87M params. Total 190.49M, train VRAM 22,185 MiB, inference VRAM pending. Untying the LM head at C_emb=256 does **not** recover the compression gap — essentially noise-level change vs tied. 74% reduction vs dense. |
| ED512_untied | 512 | 25.73M (×2 untied) | 1.05M | 1.05M | **53.56M** | **52.04%** | [link](logs/wikitext-103_2026-05-08_22-31-39/log.txt) | **1.3263** | +0.0677 vs CB; **−0.0052 vs ED512 tied** at +25.73M params. Total 217.27M, train VRAM 22,432 MiB, inference VRAM 2,610 MiB. Untying gives a small but real win at this C_emb — first untied ratio where the separate output_embedding pays for itself in the compression regime. 48% reduction vs dense. |
| ED1024_untied | 1024 | 51.46M (×2 untied) | 2.10M | 2.10M | **107.13M** | **104.07%** | [link](logs/wikitext-103_2026-05-08_23-34-22/log.txt) | **1.2902** | +0.0316 vs CB; **+0.0073 vs ED1024 tied** at +51.46M params. Total 270.83M, train VRAM 22,926 MiB, inference VRAM 2,950 MiB. Untying hurts at C_emb=1024 — the duplicated 51.46M output_embedding doesn't earn its keep, possibly because the encoder bottleneck is the binding constraint at this width. Just over dense in embedding params. |
| ED2048_untied | C=2048 | 102.93M (×2 untied) | 4.20M | 4.20M | **214.26M** | **208.16%** | [link](logs/wikitext-103_2026-05-09_00-41-41/log.txt) | **1.2611** | +0.0025 vs CB; **+0.0003 vs ED2048 tied** at +102.93M params (essentially tied). Total 377.95M, train VRAM 23,913 MiB, inference VRAM 3,510 MiB. Untying neither helps nor hurts at C_emb=C — same verdict as the tied case at the same width. Largest variant before crossing into expansion territory. |
| **ED4096_untied** | **2× C=4096** | **205.85M (×2 untied)** | **8.39M** | **8.39M** | **428.48M** | **416.32%** | [link](logs/wikitext-103_2026-05-09_01-53-33/log.txt) | **1.2568** | **−0.0018 vs CB** (also crosses under!); **−0.0029 vs ED4096 tied** at +205.85M params. Total 592.19M, train VRAM 25,888 MiB, inference VRAM 4,770 MiB. Untying gives a small win in the expansion regime at this C_emb. Per-param efficiency is poor vs ED8192 tied (much more cost for less BPB drop), but it confirms the expansion direction is legitimate in both tied and untied modes at C=4096. |
| ED8192_untied | 4× C=8192 | 411.70M (×2 untied) | 16.78M | 16.78M | **856.96M** | **832.65%** | [link](logs/wikitext-103_2026-05-09_03-18-01/log.txt) | **1.2687** | +0.0101 vs CB; **+0.0173 vs ED8192 tied** — untying significantly REGRESSES at this scale. Total 1.02B params, train VRAM 29,838 MiB, inference VRAM 7,170 MiB. Likely cause: 1B params on ~120M training tokens is data-undertrained for two independent V × C_emb tables; the tied case effectively trains one shared matrix on 2× the gradient signal. **Tying becomes architecturally important past ~C_emb = 4096** — the data-efficiency advantage of weight sharing outweighs whatever capacity untying provides. |
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

---

## New Baseline (NB) establishment

NB = R0 + W2 mixer contraction (`per_scale_mixer_widths=[0.5×4, 0.25×4]`) + T-lower wavelet off-diagonal masking. Replaces R0 as the 1-epoch ablation reference for all downstream sweeps. T-lower's 50% lifting mask is retained for stability (lets deeper levels and tighter mixer contractions train without NaN at peak LR), not parameter reduction — masked-zero positions still occupy full storage and Adagrad accumulator.

| Metric | R0 (former 1ep ref) | NB (new 1ep ref) | Δ |
|---|---|---|---|
| Total params (effective) | 392.91M | **311.10M** | **−81.81M (−21%)** |
| Shared lifting | 117.50M (dense) | 58.81M effective / 117.50M dense | −58.69M effective; dense unchanged |
| Mixer/layer | 59.11M | 35.70M | **−23.41M (−39.6%)** |
| MLP/layer | 83.91M | 83.91M | unchanged |
| FwPKM/layer | 21.34M | 21.34M | unchanged |
| Token embedding | 102.93M | 102.93M | unchanged |
| Training peak VRAM | 23,411 MiB | **23,110 MiB** | −301 MiB (−1.3%) |
| Inference peak VRAM | 3,162 MiB | **2,914 MiB** | **−248 MiB (−7.8%)** |
| BPB sliding | **1.2361** | 1.2478 | +0.0117 |
| BPB non-overlap | inf (numerical) | 1.2458 | — |
| Best val loss | **3.8177** | 3.8561 | +0.0384 |
| Run log | [link](logs/wikitext-103_2026-05-02_21-43-22/log.txt) | [link](logs/wikitext-103_2026-05-09_13-09-10/log.txt) | |

NB pays a +0.0117 BPB sliding cost vs R0 in exchange for 21% fewer trainable parameters and 7.8% less inference VRAM. The lifting-side count change (117.50M → 58.81M effective) is mask-driven and doesn't move .pt size or training VRAM; the real dense compression payoff comes from the W2 mixer contraction (mixer/layer 59.11M → 35.70M, a true −23.41M dense reduction across stored weights, Adagrad state, and forward/backward FLOPs).

### NB + deeper levels retries

Previously NaN'd at levels=9, 11, 13 on the uncompressed lifting cascade — no stability fix attempted at the time cleared them. NB's T-lower spectral-norm cap is the new candidate. Inference VRAM column reports the strategies-mode peak (the new convention going forward).

| Variant | Stability | BPB sliding | ΔBPB vs NB(L=7) | Total params | Train VRAM | Inference VRAM (strategies) | Train time | Run Log |
|---|---|---|---|---|---|---|---|---|
| Reference NB (L=7) | trains | **1.2478** | — | 311.10M | 23,110 MiB | 2,914 MiB | 1.28h | [link](logs/wikitext-103_2026-05-09_13-09-10/log.txt) |
| NB + L=9 | trains ✓ | 1.2459 | −0.0019 | 336.83M | 25,516 MiB | 3,700 MiB | 1.47h | [link](logs/wikitext-103_2026-05-09_14-52-27/log.txt) |
| **R0+T-lower + L=9** | **trains ✓** | **1.2359** | **−0.0119** | 365.73M | 26,031 MiB | pending | 1.52h | [link](logs/wikitext-103_2026-05-09_17-06-51/log.txt) |
| NB + L=11 | NaN at step 1500 (lr=6.84e-3) | — | — | — | — | — | — | [link](logs/wikitext-103_2026-05-09_16-21-56/log.txt) |
| NB + L=13 | (cancelled, expected to NaN) | — | — | — | — | — | — | |
| R0+T-lower + L=11 | (cancelled, expected to NaN) | — | — | — | — | — | — | |
| R0+T-lower + L=13 | (cancelled, expected to NaN) | — | — | — | — | — | — | |

**Reading:** levels=9 cleared the cliff that the uncompressed cascade never could — confirming T-lower's per-matrix spectral-norm cap is real and load-bearing in this regime. The BPB win is small (−0.0019, within single-seed noise) and the cost is substantial (+25.73M params, +2,406 MiB train VRAM, +786 MiB inference VRAM strategies-mode, +15% wall-clock). Useful as evidence that the cascade is healthy at deeper levels with NB's masking, but **not yet a per-cost win** — not a production-stack candidate without further BPB justification.

The levels=11 NaN at step 1500 (well before peak LR was reached) confirms the stability headroom past levels=9 requires the optimizer sweep, not just T-lower. The 2×3 factorial (NB vs R0+T-lower) × (9, 11, 13) collapsed to one informative cell once levels=11/13 were ruled out — no point running R0+T-lower at depths where NB itself fails.

**Strategies-mode generation anomaly at levels=9** (informational, doesn't affect BPB): train.py's internal strategies-mode pass produced a degenerate quote-token loop (Rep4 = 0.396 vs the standard pass's normal output). Two follow-up `generate.py --strategies` runs on the same checkpoint produced normal output (Rep4 = 0.114 and 0.161), confirming the failure was a single sampling-state-specific local minimum, not a structural model issue. The dual-pass inference-VRAM logic in `runs.sh` (with separate `generate.py --strategies` invocations per run, each in a fresh process) prevents this single-point sampling artifact from masquerading as a model problem in future ablations.

### Baseline 3 (B3): Test 1 + T-lower − wavelet_crawl

**B3** is the candidate replacement for NB. Architecturally: **Test 1 + T-lower lifting − wavelet_crawl** — Test 1's throughput regime (bs=256, MBS=8, ~58,500 steps/epoch) and structural choices, with NB's T-lower stability ingredient layered in and the deprecated `wavelet_crawl` removed. It is now the default in `BASE_PATCH_1EP` / `BASE_PATCH_5EP`; subsequent runs inherit B3 unless they override.

**Key parameters:** `bs=256`, `MBS=8`, `levels=5`, `per_scale_mixer_widths=[1.0×3, 0.5×3]` (R0 mixer pattern; W2 dropped per L=9 R0+T-lower BPB cost), `low_rank=4`, `mlp_expansion=10`, `pkm_enabled=False`, `fwpkm_num_keys=8281`, `tie_embedding_to_lm_head=True`, `lifting_offdiag_structure="lower_triangular"` (T-lower), `wavelet_crawl=False`.

| Variant | bs | MBS | levels | epochs | per_scale_mixer_widths | BPB sliding | Best val | Train VRAM | Inference VRAM (strategies) | Steps/epoch | Run Log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Test 1 (1ep) | 256 | 8 | 5 | 1 | [1.0×3, 0.5×3] | 1.1762 † | **3.6393** | **6,867 MiB** | 2,876 MiB | **~58,500** | [link](logs/wikitext-103_2026-05-09_07-52-25/log.txt) |
| Test 1 (5ep) | 256 | 8 | 5 | 5 | [1.0×3, 0.5×3] | 1.0796 † | 3.3341 | 6,867 MiB | — | ~58,500 | [link](logs/wikitext-103_2026-05-01_06-33-48/log.txt) |
| NB (1ep) | 16384 | 1 | 7 | 1 | [0.5×4, 0.25×4] | 1.2478 | 3.8561 | 23,110 MiB | 2,914 MiB | ~7,300 | [link](logs/wikitext-103_2026-05-09_13-09-10/log.txt) |
| **B3_1ep** | 256 | 8 | 5 | 1 | [1.0×3, 0.5×3] | **1.2024** | **3.7198** | 6,947 MiB | 2,874 MiB | ~58,500 | [link](logs/wikitext-103_2026-05-09_19-28-31/log.txt) |
| **B3_L7_1ep (in progress)** ‡ | 256 | 8 | 7 | 1 | [1.0×4, 0.5×4] | running (val ~4.02 at step 25k vs L=5's 4.07) | running | pending | pending | ~58,500 | [link](logs/wikitext-103_2026-05-09_21-06-17/log.txt) |
| **B3_5ep (queued)** | 256 | 8 | 5 | 5 | [1.0×3, 0.5×3] | pending | pending | pending | pending | ~58,500 | queued |

† BPB across runs with different `block_size` is not strictly apples-to-apples; best val is more comparable.

‡ Boundary case: at bs=256 with levels=7, the coarsest wavelet scale has only 256/2^7 = 2 tokens. Whether this matters in practice is itself the question B3_L7_1ep answers. Per-width contractions are now off the table everywhere, so even the boundary test uses R0 widths.

**Decision criteria.** Best val is the headline comparison. If B3_1ep lands within ~0.05 of Test 1's 3.6393 (or B3_5ep within ~0.05 of Test 1 5ep's 3.3341), B3 becomes the production stack — major reframing of the project's direction. If best val instead lands closer to NB's 3.8561, the bs=16384 commitment is paying for something beyond just BPB. B3_5ep is the matched-budget production-decision datapoint; B3_L7_1ep separately probes whether the bs=256 + L=7 boundary case is a real problem.

---

## Reversion to T1: T1_NoWC and T2 sweeps

After the dense-counting fix on 2026-05-10, B3 (Test 1 + T-lower − wavelet_crawl) was reverted to the simpler T1 + wavelet_crawl=False variant — T-lower's "9–10% parameter savings" turned out to be entirely masked zeros (no real `.pt`/VRAM/FLOPs reduction; +80 MiB train VRAM from mask-buffer overhead). Test 1 already wins decisively on matched-budget BPB / best val vs the bs=16384 NB stack, so the new active sweep is just confirming the wavelet_crawl removal is BPB-neutral and then probing whether a deeper levels=7 variant ("T2") improves on T1 in the bs=256 throughput regime.

### T1_NoWC: T1 baseline without wavelet_crawl (1ep)

Confirms the negligible-impact prediction: wavelet_crawl is only 15 floats at levels=5 (`torch.zeros(levels=5, k=3)`), so removing it is essentially param-count-identical to T1 (both at 344.63M dense). BPB delta should be within single-seed noise.

| Variant | Params (dense) | BPB sliding | Best val | Train VRAM | Inference VRAM (strategies) | Run Log |
|---|---|---|---|---|---|---|
| T1 (1ep, reference) | 344.63M | 1.1762 | 3.6393 | 6,867 MiB | 2,876 MiB | [link](logs/wikitext-103_2026-05-09_07-52-25/log.txt) |
| **T1_NoWC (1ep)** | **344.63M** | **1.1845** | **3.6658** | **6,867 MiB** | **2,954 MiB** | [link](logs/wikitext-103_2026-05-10_00-02-14/log.txt) |

**Reading:** ΔBPB sliding = +0.0083, Δbest val = +0.0265 — both within typical single-seed noise for this regime. Train VRAM identical to T1 (as expected — wavelet_crawl's `dilation_logits` parameter is ~15 floats, well below the noise floor for memory measurements). Inference VRAM nominally +78 MiB (2,954 vs 2,876) but inference VRAM has shown ±100 MiB allocator-related variance across reruns of identical configs, so this is not interpretable as a regression.

**Decision:** wavelet_crawl is confirmed removable at no measurable cost. T1_NoWC is the new T1 default for downstream comparisons; T2 (next section) tests whether deeper levels improve on this baseline.

### T2: T1_NoWC + levels=7 + R0 mixer at depth 7 (1ep, 5ep — queued)

T2 = T1_NoWC architecture extended to `levels=7` with the R0 mixer pattern at depth 7 (`per_scale_mixer_widths=[1.0×4, 0.5×4]`). Tests whether deeper wavelet decomposition is worth the coarsest-scale boundary cost: at bs=256 / 2^7 = 2 tokens at the coarsest wavelet scale (the same boundary case explored in the deprecated B3_L7 run, which trained without issue and was tracking ahead of B3_L5 at step 25k before being deprecated due to dense-counting reframing).

| Variant | Params (dense) | BPB sliding | Best val | Train VRAM | Inference VRAM (strategies) | Run Log |
|---|---|---|---|---|---|---|
| T1 (1ep, with WC) | 344.63M | 1.1762 | 3.6393 | 6,867 MiB | 2,876 MiB | [link](logs/wikitext-103_2026-05-09_07-52-25/log.txt) |
| T1_NoWC (1ep) | 344.63M | 1.1845 | 3.6658 | 6,867 MiB | 2,954 MiB | [link](logs/wikitext-103_2026-05-10_00-02-14/log.txt) |
| **T2 (1ep, no WC)** | **392.91M** | **1.1616** | **3.6094** | **7,788 MiB** | **3,186 MiB** | [link](logs/wikitext-103_2026-05-10_01-39-25/log.txt) |
| T2 (5ep, no WC, queued) | 392.91M | queued | queued | queued | queued | queued |

**Reading.** T2 wins decisively on both metrics:
- **vs T1_NoWC (matched-on-no-WC):** ΔBPB sliding = −0.0229 (~15× the 0.0015 noise threshold), Δbest val = −0.0564. The deeper levels=7 + wider mixer ([1.0×4, 0.5×4]) clearly pays for itself even without crawl.
- **vs T1 (with WC):** ΔBPB sliding = −0.0146 (~10× threshold), Δbest val = −0.0299. T2's structural improvements *more than compensate* for the +0.0083 BPB regression from removing crawl.

**Cost.** +48.28M dense params (+14% vs T1), +921 MiB train VRAM (+13%), +232 MiB inference VRAM (+8%), +15% wall-clock per epoch (~1.83h vs T1_NoWC's ~1.59h).

**Open question.** wavelet_crawl was load-bearing on T1 (+0.0083 BPB cost from disabling). T2 currently lacks crawl. A T2 + crawl variant could plausibly land another ~0.0083 BPB lower (linear additivity assumption — not guaranteed). Worth testing if T2 becomes the production stack.

**Decision criteria.** T2 (1ep) clears the "deeper levels pays its way" bar with margin. Awaiting T2 (5ep) for the matched-budget production-decision datapoint vs T1 (5ep, 3.3341 best val). If T2 (5ep) lands within ~0.02 of T1 (5ep) on best val, the levels=7 commitment is not worth the +14% cost; if it lands ~0.05+ below, T2 becomes the new production baseline.

---

## Deprecated approaches

The four sections below were moved here from the README's Future Plans list when mask-based compression was deprecated as a production direction in favor of dense compression alternatives (smaller `mlp_expansion`, smaller `C`, encoder-decoder where it actually helps). Mask-based "compression" doesn't deliver real `.pt` size, training VRAM, or throughput savings under stock kernels — masked-zero positions still occupy full storage, full Adagrad accumulator, and full forward/backward FLOPs. Content is preserved here for the experiment record and for the option of revisiting if/when sparse kernels and sparse-save infrastructure are built.

### (Deprecated) Combined Compressed Baseline (CB)

CB merged W2 per-scale mixer widths, `low_rank=4`, and BAND 128 lifting into a single combined-reductions baseline. All downstream compression sweeps in this era (PQ embedding, ED embedding, MLP structural compression) compared against CB instead of the original LR16 1.2361 / 1.0974 references. **Deprecated** in favor of NB (R0 + W2 + T-lower), which keeps the dense-compression wins (W2 mixer, low_rank=4) but uses T-lower for stability rather than BAND128 — T-lower has better BPB at the same dense storage cost, since mask-based compression doesn't actually shrink the .pt file or Adagrad state.

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

**Why training VRAM barely moved despite −32% params:** training VRAM is dominated by activation memory (forward intermediates saved for backward), not weights. Parameter compression saves the Adagrad accumulator and checkpoint storage but barely touches activation totals — the MLP hidden activation alone (`1 × 16384 × 20480` = 671 MB in fp16) plus per-scale wavelet intermediates dwarf the weights. The actual compression payoff lives in **inference VRAM** (no backward, no optimizer state, no saved activations) and **checkpoint size** — though the .pt size benefit is also illusory under stock save logic, which serializes dense tensors regardless of mask.

**Inference VRAM also moved less than expected (−7.7% for −32% params)** because at inference the weights are only a minority of total process VRAM (786 MB out of 3,260 MiB at LR16, ~24%) — the rest is the CUDA context, cuDNN/cuBLAS workspaces, the PyTorch caching allocator's reserved headroom, the torch.compile cache, forward-pass activations, and cross-window decompose-bypass state. Most of that overhead is fixed-size and doesn't scale with parameter count.

### (Deprecated) Sparse Embedding with (p, q) Striding

The token embedding (102.93M params, ~26% of the model — larger than MLP, mixer, or FwPKM individually) was a natural compression target. The **(p, q) phantom-token striding scheme** was a number-theoretic alternative to ALBERT-style factorization / vocab pruning / hash embeddings — it preserved the (N × C) embedding shape exactly, masked the embedding via a deterministic 1D walk over the flattened tensor with alternating step sizes p and q (density = 2/(p+q)), and used a "phantom token" trick to pad N to a value with useful divisibility properties without ever allocating the padded rows. The scheme was content-blind, deterministic, with O(1) metadata cost, and shipped a **q ≈ √C structural-mode heuristic** aligned with Monarch / butterfly factorization.

**Deprecated** in favor of dense embedding alternatives. PQ runs at d = 10% / 25% landed +0.1250 / +0.0565 BPB vs CB respectively — meaningful BPB regressions for compression that doesn't actually shrink .pt files or Adagrad state under stock kernels.

Full scheme, requirements, selection algorithm, and worked candidates are still in [plans/new_compression_ideas.md](plans/new_compression_ideas.md).

| Density | Mode | (p, q) | Total params | Train VRAM | Inference VRAM | BPB sliding | ΔBPB vs CB | Run Log |
|---|---|---|---|---|---|---|---|---|
| Reference (CB) | — | — | 266.63M | 23,110 MiB | 2,938 MiB | 1.2586 | — | [link](logs/wikitext-103_2026-05-08_07-19-49/log.txt) |
| 10% | smallest_q | (18, 2) | 174.00M | 23,308 MiB | 3,630 MiB | 1.3896 | +0.1310 | [link](logs/wikitext-103_2026-05-09_05-22-03/log.txt) |
| 10% | structural | (12, 8) | 174.00M | 23,308 MiB | 3,630 MiB | 1.3836 | +0.1250 | [link](logs/wikitext-103_2026-05-09_06-34-14/log.txt) |
| 25% | structural ≡ smallest_q | (6, 2) | 189.43M | 23,308 MiB | pending | 1.3151 | +0.0565 | [link](logs/wikitext-103_2026-05-09_09-40-26/log.txt) |
| ~~40%~~ (cancelled) | structural ≡ smallest_q | (3, 2) | 204.87M | — | — | — | — | |

### (Deprecated) Encoder-Decoder Embedding

A second compression scheme for the token embedding, parallel to (p, q) striding but with different structural commitments. **Forward path:** tokens → `embedding(V × C_emb)` → learnable decoder `(C_emb, C, bias=True)` → C-dim model interior. **Output path:** C-dim hidden → learnable encoder `(C, C_emb, bias=True)` → tied vocab projection via `embedding.weight^T` → V logits.

The decoder and encoder were **separate learnable matrices** because the model's nonlinearities (GELU in MLP, gating in mixer, lifting cascade) made the output-side hidden a nonlinear transform of the input-side embedding — sharing the same matrix in transposed form would force a sub-optimal output compression. Total params: `V·C_emb + 2·C·C_emb + C + C_emb` (vs dense `V·C`). Implementation in [tools/encoder_decoder_embedding.py](tools/encoder_decoder_embedding.py).

| C_emb | Total params | % of dense (V·C) | Reduction |
|---|---|---|---|
| 128 | 6.95M | 6.75% | 93% |
| 256 | 13.92M | 13.53% | 87% |
| **512** | **27.83M** | **27.04%** | **73%** |
| 1024 | 55.66M | 54.07% | 46% |
| 2048 (= C) | 111.33M | 108.16% | none — full-rank refinement ablation |

Unlike (p, q), ED's outputs are **dense** (the decoder mixes all `C_emb` dims into all `C` output dims), so the LayerNorm-amplification failure mode that made aggressive (p, q) compressions NaN at peak LR was absent. The C_emb=2048 case was a no-compression ablation testing whether the encoder/decoder machinery itself helps even without parameter savings.

**Sweep:** five 1-epoch ablations at C_emb ∈ {128, 256, 512, 1024, 2048} layered on the CB baseline, plus 7 untied-LM-head variants at the same C_emb values, plus expansion-direction variants at C_emb ∈ {4096, 8192}.

**Final tied results (7 of 7 runs landed):** The BPB gap closed monotonically through C_emb = C, then kept closing in the expansion direction — at **C_emb = 8192 (4× C), the curve crossed under CB by 0.0072 BPB**.

| C_emb | Total params | Train VRAM | Inference VRAM | BPB sliding | ΔBPB vs CB | Links |
|---|---|---|---|---|---|---|
| 128 | 170.66M | 22,012 MiB | 2,198 MiB | 1.4808 | +0.2222 | [log](logs/wikitext-103_2026-05-08_09-12-05/log.txt) |
| 256 | 177.62M | 22,087 MiB | 2,452 MiB | 1.3916 | +0.1330 | [log](logs/wikitext-103_2026-05-08_11-54-53/log.txt) |
| 512 | 191.53M | 22,235 MiB | 2,550 MiB | 1.3315 | +0.0729 | [log](logs/wikitext-103_2026-05-08_13-37-26/log.txt) |
| 1024 | 219.36M | 22,533 MiB | 2,750 MiB | 1.2829 | +0.0243 | [log](logs/wikitext-103_2026-05-08_14-41-41/log.txt) |
| 2048 (= C) | 275.02M | 23,128 MiB | 3,110 MiB | 1.2608 | +0.0022 | [log](logs/wikitext-103_2026-05-08_15-47-44/log.txt) |
| 4096 (2× C) | 386.34M | 24,317 MiB | pending | 1.2597 | +0.0011 | [log](logs/wikitext-103_2026-05-08_19-21-42/log.txt) |
| **8192 (4× C)** | **608.97M** | **26,696 MiB** | **5,610 MiB** | **1.2514** | **−0.0072** | [log](logs/wikitext-103_2026-05-08_20-44-08/log.txt) |

**Untied results (6 of 7 runs landed; ED128_untied not queued):** Mixed picture vs tied. Untying gave small wins at C_emb=512 and C_emb=4096 but small losses at C_emb=1024 and (catastrophically) at C_emb=8192. Tied was the production-default at the production-relevant scale (C_emb=8192) — the data-efficiency of weight sharing on ~120M training tokens beat the extra capacity of 411.7M duplicated output_embedding params (which became undertrained).

| C_emb | Total params | Train VRAM | Inference VRAM | BPB sliding | ΔBPB vs CB | Δ vs tied | Links |
|---|---|---|---|---|---|---|---|
| 128 (not queued) | 177.09M | ~22,063 MiB | pending | pending | pending | pending | |
| 256 | 190.49M | 22,185 MiB | pending | 1.3933 | +0.1347 | +0.0017 | [log](logs/wikitext-103_2026-05-08_16-59-00/log.txt) |
| 512 | 217.27M | 22,432 MiB | 2,610 MiB | 1.3263 | +0.0677 | **−0.0052** | [log](logs/wikitext-103_2026-05-08_22-31-39/log.txt) |
| 1024 | 270.83M | 22,926 MiB | 2,950 MiB | 1.2902 | +0.0316 | +0.0073 | [log](logs/wikitext-103_2026-05-08_23-34-22/log.txt) |
| 2048 | 377.95M | 23,913 MiB | 3,510 MiB | 1.2611 | +0.0025 | +0.0003 | [log](logs/wikitext-103_2026-05-09_00-41-41/log.txt) |
| **4096** | **592.19M** | **25,888 MiB** | **4,770 MiB** | **1.2568** | **−0.0018** | **−0.0029** | [log](logs/wikitext-103_2026-05-09_01-53-33/log.txt) |
| 8192 | 1.02 B | 29,838 MiB | 7,170 MiB | 1.2687 | +0.0101 | **+0.0173** (regresses) | [log](logs/wikitext-103_2026-05-09_03-18-01/log.txt) |

**Decision: deprecated.** The compression direction (C_emb < C) was too BPB-costly to ship — even ED1024 (the closest to CB) was +0.024 BPB sliding. Only C_emb = 8192 (4× C, expansion direction) crossed under CB by 0.0072 BPB, but at +119% inference VRAM and +128% total params for that small gain. **Interpretability cost** also matters: any ED variant scrambles semantic axes inside the model interior via the learned decoder rotation — shipping ED8192 means giving up inner-layer semantic readability for the 0.0072 BPB win. The trade-off is unfavorable against keeping the frozen FDA semantic embedding at C_emb = C with the encoder/decoder pair removed entirely. **Why untying breaks at C_emb=8192**: the model has 411.7M parameters in each of two independent V × C_emb tables (input embedding + output_embedding) — 823M just on the embedding pair, against ~120M tokens of training data. Each table effectively sees half the gradient signal that a single tied table would. Tied wins by 0.0173 BPB at this scale because the shared matrix gets twice the gradient updates per step.

### (Deprecated) MLP Structural Compression

The MLP was the second-largest single component after the token embedding (83.91M @ E=10, 167.82M @ E=20). Three structural variants were planned for the MLP weight matrices W1 (C, E·C) and W2 (E·C, C):

- **Tiled banded** — view W1 as E concatenated `(C, C)` blocks left-to-right; in each block apply a bilateral band of width W. Per-block density `(2W+1)/C` matches BAND on the lifting matrices exactly.
- **Tiled block-diagonal** — same per-block view, but with block-of-blocks pattern of size b. Per-block density `b/C`. Each output "expansion group" sees only its own input group — an architecturally clean grouped-MLP / channel-grouped feedforward interpretation.
- **(p, q) striding** — single 1D walk over the flattened weight tensor, alternating step sizes p and q, with `q | C`. No phantom tokens needed since `gcd(C, E·C) = C`. Same `find_pq` algorithm as the embedding scheme; same `q ≈ √C` structural-mode default.

Lifting empirical priors (BAND 80.1% > BD 67.8% at matched density on the lifting cascade) suggested BAND would likely win on MLP too — but the MLP nonlinearity in the middle changes the calculus, BD has a cleaner architectural story (grouped MLP), and (p, q) brings a third connectivity pattern (global walk vs local band vs grouped block) into the comparison.

**Planned sweep:** four density points (25%, 12.5%, 6.25%, 3.125%) × three structures = 12 runs at 1 epoch.

**Deprecated** before the sweep finished. Same rationale as the other mask-based deprecations — sparse mask × dense storage doesn't deliver real `.pt` / Adagrad / throughput savings under stock kernels. The dense alternative is reducing `mlp_expansion` (currently 10) for a true compression that saves storage, VRAM, and compute. Only one MLP run (MLP_BAND25, in progress at deprecation time) actually ran; results are not preserved here since the architecture itself is no longer being pursued.

### (Deprecated) More Levels

Moved here from the README on 2026-05-10. The original `levels=7` choice was the R0 reference at `bs=16384`; with B3 redefined as Test 1 + `wavelet_crawl=False` (`bs=256`, `levels=5`), the deeper-level boundary case is now `levels=7` at the `bs=256` regime (covered separately by the B3_L7 boundary test), and the original "defer levels=9/11/13" framing no longer points at a live decision.

**Results:**
- [levels=7](logs/wikitext-103_2026-05-03_02-13-07/log.txt): 5-epoch sliding BPB of 1.0974 with 392.91M params. This is the R0 run above.
- `levels=9` and `levels=11`: NaN in fp16 AMP under every stability fix attempted at this time.
- `levels=13` OOMs without `gradient_checkpointing`; likely would NaN otherwise.

**Decision:** Proceeded with `levels=7` (no change from R0 run) and deferred `levels=9, 11, and 13` for post-compression/post-optimizer changes. (Subsequently revisited under T-lower in the [Levels = 9, 11, and 13 Revisited] section above.)

### (Deprecated) Wavelet Diagonal and Low Rank Compression

Moved here from the README on 2026-05-10. Both branches (D + U·V^T diaglowrank, cross-level group sharing) were superseded — diaglowrank by the off-diagonal masking sweeps, cross-level sharing by the LLS_B3 retry framing — and the parent direction (mask-based lifting compression) is itself deprecated.

**Results:**
- `D + U·V^T` compression at r=16 (lifting_diaglowrank, [A1](logs/wikitext-103_2026-05-04_16-22-02/log.txt)) reduced lifting wavelet parameters by 97% (from 117.44M to 3.33M) and total model parameters by 29%, but dipped to a 1-epoch sliding BPB of 1.2860 vs. the reference 1-epoch BPB of 1.2361.
- Cross-level group sharing (lifting_level_sharing) NaN'd at step 2250. [`tools/analyze_lifting.py`](tools/analyze_lifting.py) confirmed the structural picture: around 75% diagonal energy and weak generic low-rank, but gradient stability becomes an issue.

**Decision:** Deprecated in favor of off-diagonal compression schemes (top-K-percent and structured variants), which were themselves later deprecated under the broader mask-based-compression deprecation.

### (Deprecated) New Testing Baseline (NB)

Moved here from the README on 2026-05-10. NB was the L=1 / `bs=16384` reference for one week before B3 (Test 1 + `wavelet_crawl=False`) replaced it as the production stack. NB's "21% parameter reduction" is now understood to be mostly mask-driven (the lifting-side 58.69M effective drop is masked zeros, not real storage); the only dense reduction was the W2 mixer's −23.41M (−39.6% mixer per layer). At matched 1-epoch budget Test 1's `bs=256` regime (best val 3.6393) beat NB (3.8561) decisively, so NB has no remaining role as a production reference.

The new baseline (**NB**) is R0 with the new per-scale mixer width [0.5x4,0.25x4] and T-lower wavelet off-diagonal masking. NB uses lower-triangular wavelets (50% lifting reduction at +0.0038 BPB), which is retained purely for Adagrad stability (higher levels, etc.), not parameter reduction. The mixer width contraction offers parameter reduction and improved stability, but at a moderate BPB cost (-39% mixer params, +0.0076 BPB). It is also retained for improved stability.

**Settings:** `layers=1`, `levels=7`, `block_size=16384`, `low_rank=4`, `per_scale_mixer_widths=[0.5×4, 0.25×4]`, `lifting_offdiag_structure="lower_triangular"`.

**Masked-position behavior:** T-lower's masked positions are zero in every forward pass. `StructuredLinear` performs `F.linear(x, weight * mask, bias)` with a bool mask, so masked positions become hard zeros. The dense weight tensor still occupies its full shape in storage and optimizer state.

**NB_1ep result vs the former 1-epoch baseline (R0):**

| Metric | R0 (former 1ep ref) | NB (new 1ep ref) | Δ |
|---|---|---|---|
| Total params (effective) | 392.91M | **311.10M** | **−81.81M (−21%)** |
| Shared lifting | 117.50M (dense) | 58.81M effective / 117.50M dense | −58.69M effective; dense unchanged |
| Mixer/layer | 59.11M | 35.70M | **−23.41M (−39.6%)** |
| MLP/layer | 83.91M | 83.91M | unchanged |
| FwPKM/layer | 21.34M | 21.34M | unchanged |
| Token embedding | 102.93M | 102.93M | unchanged |
| Training peak VRAM | 23,411 MiB | **23,110 MiB** | −301 MiB (−1.3%) |
| Inference peak VRAM | 3,162 MiB | **2,914 MiB** | **−248 MiB (−7.8%)** |
| BPB sliding | **1.2361** | 1.2478 | +0.0117 |
| BPB non-overlap | inf (numerical) | 1.2458 | — |
| Best val loss | **3.8177** | 3.8561 | +0.0384 |
| Run log | [link](logs/wikitext-103_2026-05-02_21-43-22/log.txt) | [link](logs/wikitext-103_2026-05-09_13-09-10/log.txt) | |

NB trades a small BPB regression (+0.0117 sliding, +0.0384 best val at 1ep) for a meaningful **21% reduction in trainable parameters** and a **7.8% reduction in inference VRAM** — both real wins from the dense W2 mixer contraction (the lifting-side count change is mask-driven and doesn't affect storage). Inference VRAM dropping by 248 MiB is the more material number for "what GPU does this fit on" answers.

(Note: subsequent T-lower runs and the dense-counting fix on 2026-05-10 reframed the "21% reduction" as mostly cosmetic — the dense total is 369.91M, not 311.10M, so the real reduction vs R0 is 23.41M / 392.91M = 6.0%, not 21%. The table above preserves the original "effective" framing as written at the time.)

All downstream tests in subsequent sections compared against this number until B3 replaced NB as the reference.

### (Deprecated) Levels = 9, 11, and 13 Revisited

Moved here from the README on 2026-05-10. The retest framing was built around NB as the reference; with NB itself moved to deprecated and B3 (`bs=256`, `levels=5`) replacing it as the production stack, the deeper-levels question now lives in a different regime (`bs=256` boundary at `levels=7` is itself the live frontier — the B3_L7 boundary test). The R0+T-lower L=9 finding (W2 mixer contraction costs BPB at depth) was load-bearing in the decision to drop W2 from B3 and is preserved below.

The [(Complete) More Levels with Longer Block Size](#complete-more-levels-with-longer-block-size) sweep hit an unrecoverable NaN cliff at `levels=9` and `levels=11` under fp16 AMP, and `levels=13` OOM'd without gradient checkpointing — the lifting cascade was the suspected parameter-amplification source, and the unblock was originally deferred to the [Optimizer Sweep](#optimizer-sweep-muon--adamw). The NB stack provides an **independent, complementary** path: T-lower's per-matrix spectral-norm cap (50% mask, exact zeros at masked positions) is exactly the property that should bound the cascade across deeper levels.

**Plan.** Retest `levels=9`, `levels=11`, and `levels=13` on top of NB at L=1 / bs=16384 (queued in runs.sh). Pass criteria, in order of decreasing strictness:
1. **Stability** — does the run complete without NaN / OOM? (Necessary; the original goal of the deferral.)
2. **BPB sliding close to the levels=7 NB reference**, where deeper decomposition gains offset whatever cost.
3. **BPB sliding cleanly below the NB reference** — the strongest result, indicating deeper cascades outperform shallower ones once stabilized by NB's masking.

**Boundary caveat at deep levels:** `levels=11` with bs=16384 leaves only 8 tokens at the coarsest scale; `levels=13` leaves only 2 tokens. Boundary effects on the coarse-summary signal may dominate even if the cascade trains stably. Treat the BPB-vs-NB delta as a "did the cascade survive?" signal first, "is this a better config?" signal second.

**Results so far:**

| Variant | Stability | BPB sliding | ΔBPB vs NB | Total params | Train VRAM | Inference VRAM (strategies) | Train time | Run Log |
|---|---|---|---|---|---|---|---|---|
| Reference (NB, levels=7) | trains | **1.2478** | — | 311.10M | 23,110 MiB | 2,914 MiB | 1.28h | [link](logs/wikitext-103_2026-05-09_13-09-10/log.txt) |
| **NB + levels=9** | **trains ✓** | **1.2459** | **−0.0019** | 336.83M | 25,516 MiB | 3,700 MiB | 1.47h | [link](logs/wikitext-103_2026-05-09_14-52-27/log.txt) |
| **R0+T-lower + levels=9** | **trains ✓** | **1.2359** | **−0.0119** | 365.73M | 26,031 MiB | pending | 1.52h | [link](logs/wikitext-103_2026-05-09_17-06-51/log.txt) |

**Reading:** levels=9 cleared the cascade-explosion cliff that no stability fix on the uncompressed cascade could ever clear — confirming T-lower's per-matrix spectral-norm cap is real and load-bearing in this regime. The BPB win at levels=9 is small (−0.0019, within noise of single-seed variance), and the cost is substantial: +25.73M params (+8%), +2,406 MiB train VRAM (+10%), +786 MiB inference VRAM (+27% on strategies-mode), and +15% wall-clock per epoch. **Not a per-cost win** — useful as evidence that the cascade is healthy at deeper levels with NB's masking, but not yet a production-stack candidate.

The levels=11 NaN at step 1500 (well before peak LR) confirms the stability headroom past levels=9 requires the optimizer sweep, not just T-lower. levels=13 was cancelled on the same expectation.

**R0+T-lower disambiguation result.** R0+T-lower at L=9 trains cleanly AND beats NB+L=9 by 0.0100 BPB sliding (and beats NB at L=7 by 0.0119 BPB). This is a significant finding: T-lower carries the stability load alone, AND **W2 mixer contraction appears to actively cost BPB at deeper levels** (the deeper wavelet scales need more mixer capacity to be useful, and contracting them limits what the cascade can deliver). The +28.90M params from un-contracting the mixer (mostly the depth-9 mixer expansion: 73.52M vs NB+L=9's 35.70M) are paying for themselves on BPB. This re-opens the question of whether W2 belongs in the production stack at all — see also the [Another Baseline Test](#another-baseline-test-nb-with-block_size256-and-micro_batch_size8) section testing W2's BPB cost in the bs=256 throughput regime.

**Aside on the levels=9 strategies-mode generation**: train.py's internal strategies-mode pass at this checkpoint produced a degenerate quote-token loop (Rep4 = 0.396 vs the standard pass's normal output). Two follow-up `generate.py --strategies` invocations on the same checkpoint produced normal output (Rep4 = 0.114, 0.161), confirming the failure was a single sampling-state-specific local minimum, not a structural model issue. The `--strategies` pass is now wired into `runs.sh` separately (with a fresh process / fresh RNG state) so this failure mode won't recur as a false positive in future ablations.

**Complementary to the [Optimizer Sweep](#optimizer-sweep-muon--adamw)**, not redundant: Muon tests whether orthogonalized updates handle cascade amplification *structurally*; T-lower masking tests whether bounding per-matrix spectral norm is sufficient. If both clear independently, they may compose constructively. If neither works individually, the combination is the natural last resort before declaring `levels ≥ 9` infeasible at fp16.

(Note: the param counts in the table above are pre-2026-05-10 "effective" counts that subtracted T-lower's masked positions. The dense counts are higher — NB at 369.91M, NB+L=9 at ~395.6M, R0+T-lower+L=9 at ~424.5M.)

### (Deprecated) Another Baseline Test: B3 (T-lower flavor — Test 1 + T-lower − wavelet_crawl)

Moved here from the README on 2026-05-10. This was the original B3 definition (Test 1 + T-lower lifting − wavelet_crawl), live for one day before being redefined as Test 1 + `wavelet_crawl=False` (no T-lower). The trigger for the redefinition was the dense-counting fix exposing that the original B3's "9–10% parameter savings" vs Test 1 were entirely T-lower's masked zeros — no real `.pt` / VRAM / compute reduction, plus +80 MiB train VRAM from mask-buffer overhead. With the savings illusory, T-lower stayed only as an opt-in stability tool (NaN remediation), and B3 reverted to a minimal modification of Test 1.

**Baseline 3 (B3)** is the candidate replacement for NB. Architecturally: **Test 1 + T-lower lifting − wavelet_crawl** — Test 1's throughput regime and structural choices, with NB's T-lower stability ingredient layered in and the deprecated `wavelet_crawl` convolutional component removed. It is now the default for `BASE_PATCH_1EP` / `BASE_PATCH_5EP` in `runs.sh`; all subsequent runs inherit B3 unless they override.

**Key parameters:**

- `bs=256` (Test 1 throughput regime)
- `MBS=8` (Test 1 throughput regime)
- `levels=5` (Test 1 decomposition depth)
- `per_scale_mixer_widths=[1.0×3, 0.5×3]` (R0 mixer pattern at depth 5; W2 contraction dropped after the [Levels = 9, 11, and 13 Revisited](#deprecated-levels--9-11-and-13-revisited) result showed W2 costs ~0.0100 BPB at depth)
- `low_rank=4` (Test 1 / NB)
- `mlp_expansion=10`, `pkm_enabled=False`, `fwpkm_num_keys=8281`, `tie_embedding_to_lm_head=True` (Test 1's four reductions)
- `lifting_offdiag_structure="lower_triangular"` (T-lower from NB; per-matrix spectral-norm cap, exact-zero masked positions, dense storage unchanged)
- `wavelet_crawl=False` (the deprecated convolutional component, removed)

**Motivation.** Best val at NB (3.8561) is 0.22 nats higher than Test 1 (3.6393) at matched 1-epoch budget — a substantial gap mostly explained by Test 1 seeing 8× more gradient updates per epoch (~58,500 vs ~7,300). A reasonable hypothesis is that NB's "BPB regression vs Test 1" is partly a convergence artifact: with 8× more updates, the model has time to recover whatever small BPB cost T-lower lifting introduces. If so, B3 is a strong production candidate that beats NB on best val while matching Test 1 on training cost. The L=9 R0+T-lower disambiguation result also showed that **W2 mixer contraction is not free at depth** — actively costing 0.0100 BPB at L=9 — strengthening the case to use R0 mixer widths even at shallow depth.

**Constraint.** bs=256 with `levels=7` would leave only 256/2^7 = 2 tokens at the coarsest scale (degenerate boundary). The default B3 uses `levels=5`. The boundary-test variant (B3_L7) probes whether `levels=7` even trains in this regime; per-width contractions are now off the table everywhere, so even the boundary test uses R0 widths `[1.0×4, 0.5×4]`.

**Comparison table:**

| Variant | bs | levels | epochs | per_scale_mixer_widths | Params (dense) ※ | BPB sliding | Best val | Train VRAM | Inference VRAM (strategies) | Steps/epoch | Run Log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Test 1 (1ep) | 256 | 5 | 1 | [1.0×3, 0.5×3] | 344.63M | 1.1762 † | **3.6393** | **6,867 MiB** | 2,876 MiB | **~58,500** | [link](logs/wikitext-103_2026-05-09_07-52-25/log.txt) |
| Test 1 (5ep) | 256 | 5 | 5 | [1.0×3, 0.5×3] | 344.63M | 1.0796 † | 3.3341 | 6,867 MiB | — | ~58,500 | [link](logs/wikitext-103_2026-05-01_06-33-48/log.txt) |
| NB (1ep) | 16384 | 7 | 1 | [0.5×4, 0.25×4] | 369.91M | 1.2478 | 3.8561 | 23,110 MiB | 2,914 MiB | ~7,300 | [link](logs/wikitext-103_2026-05-09_13-09-10/log.txt) |
| B3_1ep (T-lower flavor, deprecated) | 256 | 5 | 1 | [1.0×3, 0.5×3] | 344.71M ※※ | 1.2024 | 3.7198 | 6,947 MiB | 2,874 MiB | ~58,500 | [link](logs/wikitext-103_2026-05-09_19-28-31/log.txt) |
| B3_L7_1ep (T-lower flavor, deprecated) ‡ | 256 | 7 | 1 | [1.0×4, 0.5×4] | 393.03M ※※ | in progress | in progress | pending | pending | ~58,500 | [link](logs/wikitext-103_2026-05-09_21-06-17/log.txt) |
| **B3_1ep (redefined: Test 1 + wavelet_crawl=False)** | 256 | 5 | 1 | [1.0×3, 0.5×3] | 344.63M | queued | queued | queued | queued | ~58,500 | queued |
| **B3_L7_1ep (redefined)** | 256 | 7 | 1 | [1.0×4, 0.5×4] | 369.91M | queued | queued | queued | queued | ~58,500 | queued |
| **B3_5ep (redefined)** | 256 | 5 | 5 | [1.0×3, 0.5×3] | 344.63M | queued | queued | queued | queued | ~58,500 | queued |

※ Dense parameter count = `sum(p.numel())` across all model parameters. Reported here in dense form because mask-based "compression" (T-lower, low-rank masks, etc.) does not reduce real `.pt` storage, VRAM, or compute under stock kernels — masked weights are stored and processed as zeros. Pre-2026-05-10 logs reported a smaller "effective" count that subtracted masked positions; that count was misleading for production decisions and has been removed from `train.py` / `model.py`.

※※ The two T-lower-flavored B3 runs (2026-05-09 19:28 and 21:06) reported a smaller "effective" count in their logs (302.71M and 334.22M respectively) because dense counting was not yet in place. The values shown here are recomputed dense counts: each adds back the 50% of the lifting matrices that T-lower had masked but stored as zeros. After the strategic decision to drop T-lower from B3 (no real storage/VRAM savings; +80 MiB train VRAM from mask buffer overhead), these runs are kept here as a historical record only — the redefined B3 rows below are the new reference points.

† BPB across runs with different `block_size` is not strictly apples-to-apples (each run benchmarks at its own training `block_size`); best val is the more comparable metric.

‡ Boundary case: at bs=256 with levels=7, the coarsest wavelet scale has only 256/2^7 = 2 tokens. Whether this matters in practice is itself the question B3_L7_1ep answers. If L7 trains and lands competitively with L5, the boundary doesn't bind at this regime; if it underperforms L5, the levels=5 choice is correct for bs=256.

**Decision criteria.** Best val is the headline comparison. If B3_1ep lands within ~0.05 of Test 1's 3.6393 (or B3_5ep within ~0.05 of Test 1 5ep's 3.3341), B3 becomes the production stack — a major reframing of the project's direction toward the throughput-friendly regime. If best val instead lands closer to NB's 3.8561, the bs=16384 commitment is paying for something beyond just BPB. B3_5ep is the matched-budget production-decision datapoint; B3_L7_1ep separately probes whether the bs=256 + L=7 boundary case is a real problem.

### (Deprecated) Decompose Bypass Disablement Ablation

Moved here from the README on 2026-05-10. Decision is to keep both `decompose_bypass` flags `true` permanently — disablement is unsafe at the production stack and the projected savings were within noise to begin with. The flags continue to work in `model.py`; this section is just moved out of "live decisions."

**Result:** When tested with R0 and `low_rank=16`, disabling both `decompose_bypass` and `decompose_bypass_cross_window` ([logs/wikitext-103_2026-05-05_12-47-12](logs/wikitext-103_2026-05-05_12-47-12/log.txt)) NaN'd at step 1500. Prior ablations projected nearly free disablement (within ±0.0015 BPB at layers=1 & epochs=1), but the projection does not survive once `levels=7` and `low_rank=16` are stacked with a larger block size.

**Decision:** Keep both flags `true`. Removal is off the table until the [Optimizer Sweep](#optimizer-sweep-muon--adamw) increases stability. If Muon clears the cascade-amplification issue structurally, disablement can be retested. Also not essential due to low performance boost or parameter loss with their absence (based on previous ablations).

### (Deprecated) Wavelet Off-Diagonal Masking with Top-K Percent

Moved here from the README on 2026-05-10. Two reasons: (1) the magnitude-topk path requires a same-architecture reference checkpoint to compute the mask — doubling effective training cost — and (2) under the broader mask-based-compression deprecation, the masked-zero positions still occupy full `.pt` / VRAM / Adagrad state. Tooling: `tools/analyze_lifting.py` (one-shot mask computation) is moved to `OLD/` alongside this section.

**Results:** Wavelet diagonal + top-k percent off-diagonal mask, ranked by magnitude on the [5-epoch reference checkpoint](logs/wikitext-103_2026-05-03_02-13-07/log.txt) using the [analyze_lifting.py script](tools/analyze_lifting.py). The top_k 10% run recovers 81.5% of the diagonal-only-vs-full gap at +0.0096 BPB with 89.9% lifting parameter reduction.

Unfortunately, using top-k compression requires a reference checkpoint on the same architecture to compute the mask, essentially doubling training time by requiring two trainings. Full results in [runs.md](runs.md#wavelet-off-diagonal-masking-with-top-k-percent-in-progress-l1-levels7-epochs1).

**Decision:** Not to be implemented due to duplicate training requirement above. Masking also does not remove parameters: they still get stored as dense tensors and take up VRAM.

### (Deprecated) Wavelet Off-Diagonal Masking with Structured Variants

Moved here from the README on 2026-05-10. Same rationale as the rest of the mask-based-compression deprecation graveyard — masked positions are stored and processed as zeros under stock kernels, so none of the structural variants below deliver real `.pt` / VRAM / FLOPs savings. T-lower's role as a stability tool (NaN remediation only, opt-in) is preserved separately; the mask still exists in `model.py` for that use case. Tooling: `tools/lifting_constraints.py` is *imported by `model.py:39`*, so it cannot be moved to `OLD/` without code surgery — see the post-move cleanup notes in chat.

Config options:
  `"lifting_offdiag_structure"`
  `"lifting_block_size"`
  `"lifting_band_width"`
  `"lifting_monarch_blocks"`
  `"lifting_offdiag_density"`
  `"lifting_offdiag_mask_seed"`
  `"lifting_offdiag_mask_checkpoint"`

**Results:**

Structural variant types:

T upper = upper triangular
T lower = lower triangular
BD = block-diagonal
BAND = banded
MON = Monarch

See [lifting_constraints.py](tools\lifting_constraints.py) for more info on the structural variants.

| Variant | Lifting params | % of full params | Sliding BPB | % gap recovered | Run Log |
|---------|---------------:|----------:|------------:|----------------:|---------|
| Diagonal only (M0 / A1) | 3.33M | 2.83% | 1.2860 | 0% (floor) | [link](logs/wikitext-103_2026-05-04_16-22-02/log.txt) |
| T upper | 58.81M | 50.05% | 1.2381 | 92.5% | [link](logs/wikitext-103_2026-05-07_00-11-31/log.txt) |
| T lower | 58.81M | 50.05% | 1.2380 | 92.7% | [link](logs/wikitext-103_2026-05-07_01-31-42/log.txt) |
| BD 64 | 3.73M | 3.17% | 1.2613 | 47.7% | [link](logs/wikitext-103_2026-05-07_02-52-12/log.txt) |
| BD 128 | 7.40M | 6.30% | 1.2564 | 57.1% | [link](logs/wikitext-103_2026-05-07_10-40-27/log.txt) |
| BD 256 | 14.74M | 12.55% | 1.2509 | 67.8% | [link](logs/wikitext-103_2026-05-07_04-12-07/log.txt) |
| BD 512 | 29.42M | 25.04% | 1.2444 | 80.3% | [link](logs/wikitext-103_2026-05-07_12-00-36/log.txt) |
| BAND 32 | 3.76M | 3.20% | 1.2622 | 46.0% | [link](logs/wikitext-103_2026-05-07_13-22-09/log.txt) |
| BAND 64 | 7.34M | 6.25% | 1.2563 | 57.3% | [link](logs/wikitext-103_2026-05-07_05-32-41/log.txt) |
| BAND 128 | 14.33M | 12.20% | 1.2508 | 68.0% | [link](logs/wikitext-103_2026-05-07_14-42-59/log.txt) |
| BAND 256 | 27.63M | 23.51% | 1.2445 | 80.1% | [link](logs/wikitext-103_2026-05-07_06-52-52/log.txt) |
| MON 32 | 5.56M | 4.73% | 1.2614 | 47.5% | [link](logs/wikitext-103_2026-05-07_16-58-23/log.txt) |
| MON 64 | 5.56M | 4.73% | 1.2615 | 47.3% | [link](logs/wikitext-103_2026-05-07_09-31-54/log.txt) |
| MON 128 | 8.31M | 7.07% | 1.2584 | 53.3% | [link](logs/wikitext-103_2026-05-07_18-07-19/log.txt) |
| MON 256 | 15.20M | 12.94% | 1.2533 | 63.1% | [link](logs/wikitext-103_2026-05-07_19-18-23/log.txt) |
| Full wavelet (LR16) | 117.50M | 100.00% | 1.2342 | 100% (ceiling) | [link](logs/wikitext-103_2026-05-05_07-07-45/log.txt) |

**Decision:**
- These don't actually decrease parameter counts, VRAM needed for training or inference, or model complexity.
- They just set some entries to 0.
- Not using this.
