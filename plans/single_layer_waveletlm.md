# Single-layer WaveletLM (L=1)

Test whether WaveletLM can match or approach its current L=2 quality at a single block, when paired with the full modern feature stack (5-epoch training, 2.0× dropout, `per_scale_mixer_widths`, `cross_scale_gating`, `wavelet_crawl`, `decompose_bypass_cross_window`, refined `weight_decay`, tied embeddings) and the parameter-reduction bundle from Section 8 of `other_post_release_plans.md`. Most recent L=1 result is from 2026-04-11 (BPB 1.1538 at 1 epoch); predates roughly six meaningful improvements that contributed to the current L=2 baseline of BPB 1.0140 (PPL 23.75).

## Why a single layer matters

### Architectural simplicity (the engineering case)

At L=1, every Layers-bucket parameter halves: Mixer/layer, MLP/layer, PKM/layer, FwPKM/layer, plus the unlisted per-layer items. Combined with the parameter-reduction bundle from Section 8 of `other_post_release_plans.md`, an L=1 reduced configuration could plausibly land near **250–300M parameters** — putting WaveletLM in direct parameter-count competition with Transformer-XL Standard (151M) at moderate quality cost.

Wall-clock is approximately halved as well: per-step compute drops by ~50% (one block instead of two), so a 5-epoch L=1 run finishes in roughly the time of a 2.5-epoch L=2 run. This means **for the same compute budget, L=1 can be trained for ~2× as many epochs as L=2**, which may close (or invert) part of the architectural quality gap. Training-time savings compound with the parameter reduction, since smaller MLPs further reduce per-step cost.

### Interpretability (the research case)

A single-layer WaveletLM has an information-flow story uniquely amenable to interpretation:

```
input tokens → embedding → wavelet decomposition (5 scales) →
  per-scale FWHT → gated SwiGLU mixer → inverse FWHT →
  wavelet reconstruction → MLP → LM head → output logits
```

Every output logit traces back to the input through a single composition of named operations rather than through L iterated stages. This makes several interpretability techniques substantially more tractable:

- **Per-scale activation patching** — knocking out a specific wavelet scale (e.g., zeroing scale 3's coefficients) and observing the change in generation register. At L=2, the same intervention reverberates through a second block whose contribution is hard to disentangle.
- **Single-hop circuit analysis** — every neuron in the LM head reads from a single wavelet-reconstruction output. Tracing what each scale contributes to specific token predictions is a one-hop attribution problem.
- **Embedding-to-coefficient mapping** — with a single-layer model, the relationship between input embeddings and per-scale wavelet activations is direct and measurable. With a semantic embedding (see below), this mapping becomes the basis for a controllability study.
- **Per-component ablation** — turning off the FWHT slot, the gated SwiGLU, the lifting decomposition, etc., and measuring the change in output register has clean attribution at L=1. At L=2, the second block can sometimes compensate for impaired first-block components, blurring the result.

The interpretability benefits are non-trivial even without a semantic embedding: a single learned-embedding L=1 model is already easier to study than the L=2 baseline. **With** a semantic embedding reintroduced, L=1 becomes the natural variant for investigating concept-to-output controllability, since both the input space (semantic-anchored) and the architectural depth (single-hop) are simultaneously interpretable.

This isn't a guaranteed unlock — interpretability gains compound with simplicity but don't directly translate to capability. The argument is structural: **whatever interpretability work follows is dramatically cheaper at L=1.** If the quality cost is modest (say, +0.05 BPB), the trade may be obviously worthwhile for any research direction that requires understanding the model's internal computation.

## Why this hasn't been done

Most recent L=1 result was 2026-04-11. Improvements added since then that have **not** been retested at L=1:

| Feature | Tested at L=1 historically? | Notes |
|---|---|---|
| 5-epoch training | No | Single biggest contributor to current best run |
| 2.0× dropout sweep result | No | -0.0221 BPB at the prior baseline; never re-tuned at L=1 |
| `per_scale_mixer_widths: [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]` | No | Per-epoch speedup + small BPB win |
| `cross_scale_gating: True` | No | Routing matrix between scales |
| `wavelet_crawl: True` (K=3) | No | Near-noise free win |
| `decompose_bypass_cross_window: True` | No | Cross-window state carry |
| Refined five-dropout config | No | The dropout sweep that produced `dropout_lm_head=0.24` etc. didn't visit L=1 |
| Refined `weight_decay` | No | Sweep at L=1 not done |

`shared_lifting_weights` is structurally irrelevant at L=1 (only one block). `tie_embedding_to_lm_head` is independently available and would compose cleanly.

## Proposed test matrix

Four runs covering the L=1 vs L=2 question and the 1-epoch vs 5-epoch question, with the current best feature stack across all four:

| Run | Layers | Epochs | Notes |
|---|---|---|---|
| **A** | 1 | 1 | Modernized L=1 baseline at 1 epoch. Direct successor to the 2026-04-11 L=1 record. Measures how much of the prior L=1 result was held back by missing features. |
| **B** | 2 | 1 | L=2 at 1 epoch with the full modern stack. Already roughly known but worth re-running for clean head-to-head against (A). |
| **C** | 1 | 5 | The interesting cell. Measures whether L=1 with full training time closes the gap to the L=2 baseline. |
| **D** | 2 | 5 | Already exists as the current best run (BPB 1.0140). No re-run needed; included in the comparison set as the baseline. |

### Comparison logic

- **(A) vs (B)** isolates the L=1 vs L=2 architectural difference at fixed (limited) epochs. Tells us how much of L=2's quality comes from architectural depth before training matters.
- **(A) vs (C)** isolates the 1-epoch vs 5-epoch contribution at fixed L=1 architecture.
- **(B) vs (D)** confirms that the existing 5-epoch L=2 number reproduces. (Optional sanity check.)
- **(C) vs (D)** is the headline comparison. **If (C) is within ~0.05 BPB of (D), L=1 is a viable lightweight variant with strong interpretability properties.**

### Compute cost

At the current 882.5M-parameter scale, per-epoch time on a 5090 is ~2.5-3h. L=1 halves per-step compute, so:

- Run A (L=1, 1 epoch): ~1.25-1.5h
- Run B (L=2, 1 epoch): ~2.5-3h (same as one epoch of the current best run)
- Run C (L=1, 5 epochs): ~6.25-7.5h
- Run D: 0h (already exists)

**Total new compute: ~10-12h on a 5090.** Single weekend's worth.

If the parameter-reduction bundle (Section 8) is applied first, all four runs scale down further. Reduced-parameter L=1 at 5 epochs is plausibly ~3-5h.

## Variations worth considering

These are not part of the headline test matrix above but should be tested if the headline matrix shows L=1 is competitive:

### Compute-equalized comparison

Run an L=1 model for **2× the epochs of the L=2 baseline** within the same wall-clock budget. If L=2 5-epoch is the baseline, run L=1 for 10 epochs. This measures whether L=1 trained longer at fixed compute beats L=2 trained shorter.

### Smaller effective batch size

Counterintuitive but plausible: reducing MBS at L=1 may help via two mechanisms simultaneously:

1. **Stability**: smaller batches admit higher learning rates without NaN'ing. The 2026-04-17 exp_param result at L=1 with lr=0.02 showed this works at the prior scale.
2. **Gradient noise**: smaller batches inject more gradient noise, which can act as implicit regularization and may help avoid the train/val gap that limits the current L=2 baseline.

Concretely: try MBS=4 with grad_accum tuned to maintain effective batch size, then separately try MBS=4 with reduced effective batch (let the gradient noise increase). LR sweep around 0.01 - 0.02 in conjunction.

### Extended context

L=1 frees substantial VRAM relative to L=2. Reallocating that to longer block size (1024 or 2048) tests whether the freed memory is better spent on context length than on architectural depth. Particularly relevant for PG-19, where long-form narrative makes context-length headroom load-bearing. May interact with the 2D wavelet plan (`plans/two_d_wavelet_sequential_training.md`).

### With and without semantic embedding (interpretability arc)

The L=1 architecture is the natural starting point for the eventual semantic embedding reintroduction work (`plans/reincorporate_large_semantic_embedding.md`). Two parallel runs are worth doing:

- L=1 with current learned embedding (this plan's headline test).
- L=1 with semantic embedding (post-reintroduction). Compare BPB delta to learned-embedding L=1, and run interpretability ablations at this configuration to study the embedding-to-output mapping directly.

## Composability

This plan is naturally composable with several other post-release items in `other_post_release_plans.md`:

- **Section 8 (Parameter reduction)**: at L=1, the parameter-reduction bundle has even larger relative effect since fewer copies of MLP/PKM/FwPKM exist. The combined "L=1 + reduced parameters" config is the most compact viable variant of WaveletLM.
- **Section 9 (Cross-layer parameter sharing)**: largely irrelevant at L=1 (no cross-layer to share with), but the analysis is informative — if L=2 with full sharing approaches L=1 with no sharing, sharing is essentially "L=1 with twice the iterated processing."
- **Section 10 (Per-scale mixer transform ablation)**: cleaner attribution at L=1. The four-condition ablation has only one block to interfere; results transfer more directly to architectural understanding.
- **2D wavelet over (Batch, Token)** (`two_d_wavelet_sequential_training.md`): the cross-batch wavelet decomposition machinery is independent of layer count. L=1 makes the 2D wavelet's contribution easier to attribute.
- **Semantic embedding reintroduction** (`reincorporate_large_semantic_embedding.md`): direct dependency. L=1 is the natural architectural target for that work; running this plan's headline test first establishes the BPB cost of L=1 before the semantic embedding adds its own variable.

## Hypothesis

L=1 with the modern feature stack will land **substantially better than the 2026-04-11 result of BPB 1.1538**. The honest projection range is wide because none of the post-April-11 features have been measured at L=1:

- **Conservative**: BPB ~1.10 at 5 epochs (close to but below the L=2 baseline)
- **Mid**: BPB ~1.07 at 5 epochs (within ~0.05 BPB of L=2 baseline; clearly viable for interpretability work)
- **Optimistic**: BPB ~1.04 at 5 epochs (near or matching the L=2 baseline; L=1 becomes the new release candidate)

If the optimistic range realizes, the architectural framing of WaveletLM shifts: depth becomes optional, and the *single-block multi-scale mixer + wavelet decomposition + MLP* is the load-bearing primitive. That is the strongest possible outcome for both interpretability and engineering.

## Why this matters

Three independent reasons to prioritize this:

1. **Engineering**: L=1 + parameter reduction is the path to a lightweight WaveletLM at ~250-300M params. Lower deployment cost, broader hardware reach, faster inference.
2. **Interpretability**: every interpretability-research direction is dramatically cheaper at L=1. Most relevant to the eventual semantic embedding reintroduction work.
3. **Architectural understanding**: clarifies what depth is actually doing in WaveletLM. If L=1 with longer training matches L=2, the architecture's apparent depth-dependence is partially an artifact of compute allocation rather than a true depth requirement.

The compute cost is small (~10-12h total), the experimental design is clean (four runs, three of them new), and the result is informative regardless of which way it lands. **This is the single highest-ROI experiment available before the parameter-reduction bundle even runs**, and it should be elevated to the top of the post-release roadmap.
