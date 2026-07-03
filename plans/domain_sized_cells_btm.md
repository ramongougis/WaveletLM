# Domain-Sized Cells: Branch-Train-Merge with Chinchilla-Matched Widths

## Status

**Proposed — post-release research (2026-07-02).** Independent WaveletLM "cells," each trained on a
different corpus (or subset) and each **sized to its own data via free-C**, blended at the logit level
for the overall model's predictions. Builds on the multinodal PoE machinery + the free-C unlock.

## Core idea

Per-domain expert LMs, trained fully independently (possibly *sequentially on one cheap GPU*), each with a
width matched to its corpus (Chinchilla-style ÷20 heuristic, or better, the measured knee from the free-C
sweep): e.g. a C≈300 WT-103 cell, a C≈750 PG-19 cell, a C≈? code cell. Predictions blend at the LM-head
logit level (shared vocab), via uniform averaging → learned gates → a context router.

Taxonomy: **MoM** (mixture-of-mixers) routes mixers *inside a block*; **multinodal PoE** ensembles identical
whole models; **domain cells** specialize whole models by *data and size*.

## Precedents (this is the BTM family)

- **Branch-Train-Merge** (Li et al. 2022) — independent expert LMs on domain shards, ensembled/averaged.
- **c-BTM** (Gururangan et al. 2023) — unsupervised cluster shards, nearest-cluster weighting at inference.
- **DEMix layers** (Gururangan et al. 2021) — domain-expert FFNs inside one model.
- **Branch-Train-MiX** (Sukhbaatar et al. 2024) — branch-train, then merge experts back into a single MoE.

**The novel wrinkle here:** the BTM line uses *uniform* expert sizes. **Per-domain Chinchilla-matched widths**
(possible only with free-C) is unexplored — "every domain gets the width its data can feed."

## Design constraints (the two hard ones)

1. **Shared tokenizer, logit-level blending only.** Heterogeneous-C cells cannot share hidden states
   (dimension mismatch; projections would reintroduce the machinery being avoided). Logits over a **shared
   vocab** blend trivially — but PG-19 currently auto-trains its own 32K SentencePiece vs WT-103's GPT-2
   BPE. Domain cells force one vocabulary for all cells (GPT-2 BPE everywhere, or one union-trained SP).
2. **The baseline to beat is the blend-trained monolith** (already on the roadmap): a single model at
   matched *total* params and tokens trained on the mixed corpus. BTM's honest record: experts win on
   their domains; the ensemble can lag the monolith on general text. Report the cell-sum as the param count.

## Why WaveletLM-friendly

- **Budget-parallel in time:** each cell is small and trains solo on a 5090, sequentially. No big-GPU rung.
- **Extensible:** a new domain = one new cell; existing cells untouched (the "branch" property).
- **Shared frozen lifting** (ties to the frozen-wavelet transfer test): seed every cell with the same
  trained lifting → common temporal basis across cells, cheaper training, and a clean story
  ("same wavelet ears, different domain brains"). No BTM analog exists.
- **Interpretability:** domain attribution at inference is free — read the gate/router weights.
- **Width sizing input:** the free-C knee sweep directly supplies each cell's C.

## Blending ladder (cheap → rich)

1. Uniform logit average (existing multinodal PoE machinery; needs `multinodal_cell_dim` scalar → per-cell list).
2. Learned scalar gates per cell.
3. Context router: c-BTM-style cluster similarity, or perplexity-based weighting (each cell scores the
   prefix; weight by recent per-cell loss).

## Minimal test ladder

1. **Two cells:** WT-103 cell + PG-19-subset cell (both GPT-2 BPE, sized per the knee sweep), uniform
   logit average. Eval on both domains + a neutral held-out set **vs the matched blend-trained monolith**.
2. Learned gates; measure the gate's domain attribution against ground truth.
3. **Extensibility demo:** add a third domain cell (e.g. code or ArXiv) with zero retraining of cells 1–2.

## Relationship

- Machinery: [multinodal PoE](../README.md#multinodal-mode-product-of-experts) (needs per-cell dims).
- Orthogonal/composable with **MoM** (a cell could itself use MoM internally).
- Inputs: free-C unlock; the C-knee sweep (per-domain widths); frozen-wavelet transfer (shared basis).
- Competitor baseline: the [data blend](pretraining_data_blend.md) monolith runs.
