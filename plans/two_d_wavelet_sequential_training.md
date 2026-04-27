# 2D wavelet decomposition over (batch, token) with sequential training

Extend WaveletLM's lifting wavelet decomposition from a 1D operator over the token axis (`T`) to a 2D operator over the joint axis pair (`B`, `T`), where the batch dimension `B` carries the same multi-scale temporal structure as `T` when training proceeds in document-sequential order. Both axes are time-related — `T` over tokens within a chunk, `B` over chunks across the document — and the same wavelet machinery applies to both. This adds two new sub-bands per level on top of the existing within-chunk decomposition, providing a multi-scale architectural mechanism for long-range structure that current `block_size=256` training cannot capture.

## The architectural insight

The current WaveletLM lifting wavelet operates only along T (token positions within a chunk). When training is randomly shuffled, the batch axis `B` carries no temporal structure — different samples are independent. But **when training proceeds in document-sequential order** (chunks of the same document loaded in source order, with chunk N+1 immediately following chunk N), `B` acquires the same three properties that justify wavelet decomposition along T:

1. **Locality**: chunk i+1 is more related to chunk i than to chunk i+5.
2. **Shift-relevance**: book-level patterns recur across chunks (chapter structure, character introductions).
3. **Multi-scale structure**: short-range narrative continuity, medium-range scene structure, long-range plot arcs.

A 2D wavelet decomposition over (B, T) produces four sub-bands per level instead of two:

- **LL** (low-low): coarse approximation in both axes.
- **LH** (low-high): rapid variation within chunks, stable across chunks (local-detail content).
- **HL** (high-low): slow within-chunk, rapid across-chunk (chapter transitions, scene shifts).
- **HH** (high-high): joint high-frequency variation in both axes.

Implemented separably (T-wavelet then B-wavelet), this is a **strict generalization of the current 1D T-wavelet**: with the B-axis lifting initialized as a near-no-op (zero predict and update networks), the architecture recovers exactly current behavior. Training learns whether to make the B-axis lifting active.

## Why this is the cleanest cross-batch architecture

The 2D wavelet leverages the existing lifting infrastructure already in WaveletLM and adds an explicit multi-resolution prior on cross-batch structure — coarse approximations capture document-region summaries, fine details capture chunk-to-chunk transitions, and the per-band mixers can specialize. Other potential cross-batch mechanisms (recurrent state, hidden-state caches, explicit compression) would introduce a parallel mechanism alongside the existing wavelet rather than generalizing it.

The 2D version is also the *strict* extension of the existing architecture: it can be made to recover current behavior at initialization, then training learns whether to expand into the new B-axis capacity. The cross-batch mechanism is therefore architecturally homogeneous with the within-chunk mechanism — same operator, generalized to a second axis — rather than a separate machinery bolted on.

## Sequential training as a hard requirement

The 2D wavelet's value collapses to zero if `B` is randomly shuffled.

A subtlety worth acknowledging: the standard "i.i.d. samples" assumption underlying random-shuffled SGD is an *architectural idealization*, not a property of real corpora. Natural language data carries dependence structure at multiple granularities — within-document (chapters of a novel), within-series (sequels and connected works), within-author (themes recurring across a bibliography), and cross-author (intertextual references). Random shuffling does not eliminate these dependencies; it merely hides them from the architecture, where they degrade to noise in gradient estimation. Sequential training preserves and exposes the dependence structure to the architecture, which the 2D wavelet then provides a mechanism to exploit.

Concrete example: PG-19 contains series of novels with cross-book plot dependencies (one book's events depend on a prior book's events). Random sampling pairs unrelated chunks together and discards this dependence; sequential training preserves it. The same logic applies more broadly to authored intertextual universes (e.g., Star Wars Expanded Universe, Forgotten Realms) where works are explicitly designed to depend on other works.

The architecture therefore commits to:

- **Within-batch chunk ordering**: chunks at indices `B=0, 1, 2, ..., N-1` must be consecutive segments of the same document (or parallel sequential streams), in source order.
- **Cross-batch state passing**: at the end of each forward pass, save the B-axis approximation coefficients at the deepest scale; use them as the initial state for the next training step.
- **Document boundary handling**: when a document ends and a new one begins, reset the B-axis state (or signal a discontinuity to the model). Mid-batch document transitions need explicit handling.
- **Optional richer ordering at higher granularities**: the data loader can in principle preserve order at multiple levels of corpus structure (within-document chunks, within-series documents, within-author bibliography, chronological corpus order). Each additional level preserved is additional dependence structure the 2D wavelet can exploit. The minimum useful commitment is within-document; higher granularities are optional research extensions.
- **Detached gradients across batches**: do not backpropagate through the cross-batch state. The model learns to *use* historical context but does not learn to *shape* what gets passed forward. Required for memory-bounded training; full backprop across batches would multiply activation memory by the number of batches in the BPTT window.
- **Same ordering at evaluation time**: random-access generation breaks the architectural assumption.

**Practical batching strategy**: rather than B=8 sequential chunks of one document, use B=8 *parallel streams* from 8 different documents, each advancing in lockstep. Each B-index has its own document and its own sequential history. This recovers some i.i.d. property at the stream level while preserving sequential order within each stream.

## Implementation considerations

### Depth asymmetry between B and T

With `B=8` and `T=256`, B supports `log2(8)=3` levels of decomposition and T supports `log2(256)=8`. Two clean handlings:

- **Independent max-depth (preferred)**: do 3 levels of joint (B, T) decomposition, then continue 5 more levels of T-only decomposition on the LL band. Preserves all current T-multi-scale capacity while adding the 2D structure where B has scale to support it.
- **Capped at min(log2(B), log2(T))**: only do `min` levels of joint decomposition, leaving deeper T scales un-decomposed. Loses current T-expressivity.

### Causality along both axes

T-causality is preserved by front-padding T (current implementation). B-causality requires the same trick along B: chunk i+1 can see chunks ≤ i but not chunks > i. The dilated-shift access pattern becomes 2D, with both shifts being one-directional (front-pad only).

### Per-band capacity allocation

Generalize `per_scale_mixer_widths` from a 1D list to a 2D table indexed by `(B-scale, T-scale)`. Bias capacity toward the LL corner (long-range structure) and away from the HH corner (high-frequency local-and-cross-chunk variation). Reasonable initial allocation:

| B-scale ↓ \ T-scale → | coarsest | mid | finest |
|---|---|---|---|
| coarsest | 1.0 | 0.75 | 0.5 |
| mid | 0.75 | 0.5 | 0.25 |
| finest | 0.5 | 0.25 | 0.125 |

### Compute and memory cost

Approximately 2-4× the current wavelet-stage cost: more sub-bands per level, more mixers per level. Some of this is offset by the smaller per-band sizes (each sub-band is `B/2 × T/2` after one level vs. the current `T/2`). Net effect: real but bounded overhead, well within consumer-hardware budgets when paired with the parameter-reduction bundle (Section 8 of `other_post_release_plans.md`).

### Catastrophic forgetting risk

Sequential training violates the i.i.d. assumption underlying SGD. Practical mitigations from continual-learning literature:

- Document-level shuffling above chunk-level ordering (process documents in a randomized order, but each document's chunks in source order).
- Replay buffers for older content (probabilistic re-presentation of earlier documents).
- Regularization terms (e.g., elastic weight consolidation) for parameters that contribute heavily to earlier-document loss.

These can be added incrementally as needed; initial runs should test whether they're actually required.

## Hypothesis and proposed study

**Hypothesis**: 2D wavelet decomposition over (B, T) with sequential training outperforms random-shuffled WaveletLM on long-dependency corpora like PG-19, where cross-batch structure carries meaningful information.

**Phase 1 — sequential training baseline**:
- Run WaveletLM with sequential training and the *current 1D T-wavelet* (no architectural change), with cross-batch state passing of the deepest-scale hidden activations.
- Establishes whether sequential training alone (without the 2D extension) provides benefit. Isolates the data-ordering variable from the architectural variable.

**Phase 2 — 2D wavelet extension**:
- Add the 2D wavelet decomposition over (B, T) on top of the sequential training infrastructure.
- Compare to Phase 1 baseline at matched compute. Isolates the architectural variable.

**Datasets**: PG-19 (primary — long-dependency corpus where the architectural commitment matters most). WT-103 as secondary (sanity check; does the 2D wavelet hurt on a corpus without strong cross-batch structure?).

**Compute estimate**: ~1-2 weeks of 5090 time when paired with the parameter-reduction bundle.

## Composability with the rest of the roadmap

This plan is naturally composable with several other post-release items:

- **Parameter reduction (Section 8)**: reduces the per-block VRAM cost, freeing memory for the 2D wavelet's increased mixer count and for longer effective context.
- **Optimizer sweep (Section 6)**: sequential training may interact differently with Adagrad vs AdamW vs Muon than random-shuffled training does. Worth re-validating optimizer choice once sequential training is in place.
- **Stable parametrization (Section 5)**: more cascade depth (2D wavelet has more sequential operations per block) increases gradient-depth pressure. Stable parametrization may become more important.
- **Long-context Hyena head-to-head**: the 2D wavelet provides one cross-batch mechanism; Hyena's 16k-context training extends within-context capacity differently. The eventual Hyena comparison is more meaningful when both architectures have access to long context via their respective mechanisms.

Suggested ordering: parameter reduction → optimizer sweep → sequential-training baseline (Phase 1) → 2D wavelet extension (Phase 2) → long-context model comparisons.

## Caveats

- **Engineering complexity**: 2D wavelet code is meaningfully more involved than 1D. Cross-batch state machinery, document-aware data loading, document-boundary handling, and 2D causality all add implementation effort.
- **Data loader requirements**: the existing data loader needs significant changes to support document-sequential parallel-stream batching with proper document-boundary signaling.
- **Compute cost is real**: phases 1-2 collectively are weeks-of-5090 work, not days. This is post-parameter-reduction territory.
- **Catastrophic forgetting may emerge as a practical issue**: standard continual-learning mitigations may be needed; budget time for that.

## Why this matters

WaveletLM's central architectural commitment is "multi-scale decomposition is the right inductive bias for sequence modeling." The 1D wavelet over the token axis validates this within a single context window. The 2D extension generalizes the same commitment across the batch axis when sequential training is available, providing an architecturally homogeneous pathway to long-context modeling — same operator, generalized to a second axis — rather than introducing a separate cross-batch mechanism.

If the hypothesis holds, this is a real architectural contribution: **multi-scale wavelet decomposition as a unified mechanism for both within-chunk and cross-batch information flow, generalizing the central architectural primitive across both time-related axes with a single principled operator.** That's the strongest possible thesis statement for the eventual paper.

If the hypothesis doesn't hold, the negative result is also informative: it would suggest that the multi-scale prior is genuinely token-axis-specific and doesn't generalize naturally to the cross-batch axis. That's useful information about what wavelet inductive bias actually buys.

Either outcome is publishable. The experimental design produces information regardless of which direction the result lands.
