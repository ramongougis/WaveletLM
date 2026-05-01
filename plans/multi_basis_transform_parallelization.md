# Multi-basis transform parallelization

A WaveletLM-native architecture where the FWHT slot in each per-scale mixer is replaced by N parallel orthogonal-transform paths. Wavelet decomposition and reconstruction stay shared across nodes; only the transform → mixer → inverse-transform middle of the pipeline is duplicated. Conceptually, each node decomposes the wavelet coefficients through a different "prism" (a different orthogonal basis), the mixer learns gated interactions in that basis, and the inverse transform brings each node's output back to the shared wavelet coefficient space for combination.

## Distinction from existing multinodal mode

This is **not** the same as the existing `multinodal_enabled` mode in the codebase. That mode runs N independent full-cell copies of WaveletLM in parallel and combines them at the LM head via logit averaging (a model-level product-of-experts ensemble). Multi-basis transform parallelization operates *inside* a single WaveletLM model — only the FWHT slot is parallelized, with shared wavelet scaffolding above and below — and combines per-node outputs in the shared wavelet coefficient space, before MLP and LM head.

| | Existing `multinodal_enabled` (PoE) | Multi-basis transform parallelization |
|---|---|---|
| Granularity | Whole models in parallel | Internal mixer paths in parallel |
| What's duplicated | Everything (full WaveletLM cells × N) | Just the FWHT slot |
| Where combination happens | At the LM head (logit averaging) | Inside each per-scale mixer (wavelet-coefficient sum/gating) |
| Mechanism | Product-of-experts ensembling | Multi-perspective feature decomposition |
| Compute overhead | ~N× (each cell is full model) | ~5-15% (only mixer slot duplicated) |

The two architectures are complementary, not competing. The existing multinodal mode is surveyed alongside other multi-expert techniques in [`multinodal_training_techniques.md`](multinodal_training_techniques.md). This document is exclusively about the multi-basis architecture.

## Architecture

```
Input → Shared Wavelet Decomposition → wavelet coefficients
                                              │
            ┌─────────────────────────────────┼────────────────────────┐
            ↓                                  ↓                        ↓
        Node 1: FWHT                  Node 2: DHT (Hartley)     Node N: learned-orth
            ↓                                  ↓                        ↓
        Node 1: gated mixer           Node 2: gated mixer       Node N: gated mixer
            ↓                                  ↓                        ↓
        Node 1: inverse FWHT          Node 2: inverse DHT       Node N: Wᵀ
            └─────────────────────────────────┼────────────────────────┘
                                              ↓
                                    Combine (sum or learned per-node-per-scale gate)
                                              ↓
                                Shared Wavelet Reconstruction
                                              ↓
                                    MLP → LM Head → Output
```

See [`../assets/waveletlm-multi-basis.svg`](../assets/waveletlm-multi-basis.svg) for the rendered diagram embedded in the README.

## Reference node lineup (4-node configuration)

| Node | Forward transform | Inverse | Notes |
|---|---|---|---|
| 1 | FWHT | FWHT (involutive) | Current default; the existing per-scale mixer slot |
| 2 | DHT (Hartley) | DHT (involutive) | Real-valued analog of DFT; involutive up to scale |
| 3 | DCT-II | DCT-III | True forward/inverse pair; well-understood basis |
| 4 | Learned butterfly orthogonal | Wᵀ (parameter-tied) | Discovers any structure the fixed bases miss |

Extensible — adding more learned-orthogonal nodes increases per-step compute linearly in N (only the mixer slot is duplicated), but doesn't increase the dominant MLP cost.

## Invertibility requirement

The per-node inverse transform must produce outputs in the shared wavelet coefficient space, so all transforms must be invertible. FWHT and DHT are involutive (forward = inverse up to scale); DCT-II inverts via DCT-III; learned transforms must be orthogonality-constrained (butterfly parametrization is standard). General unconstrained learned linear layers would *not* be invertible and would break the architecture.

## Why split at this point in the architecture

Three properties make the post-decomposition / pre-FWHT split the architecturally clean choice:

1. **The orthogonal transforms are the "prisms."** Each transform decomposes the wavelet coefficients into a different generalized-frequency view. Combining N such views is analogous to integrating multiple spectral readouts of the same input.
2. **Shared wavelet pipeline preserves invertibility for free.** The space before the split (wavelet coefficients) and after the per-node inverse (also wavelet coefficients) are identical, so combination is just summing/gating in that shared space — no bridging machinery.
3. **The mixer is the actual learning.** Each per-node mixer reads transform-domain coefficients and learns gated SwiGLU interactions in that basis. Different bases = different learned interactions = real specialization across nodes.

## Combination variants

- **Equal-weight sum across nodes** — simplest. May over-mix if some nodes are learning irrelevant structure.
- **Learned per-node gating** — single learned weight per node per scale (negligible parameter cost). Lets the model down-weight or ignore basis paths that don't contribute.
- **Learned per-node-per-scale-per-coefficient gating** — most expressive, highest parameter cost, highest mode-collapse risk.

## Mode-collapse mitigation for learned-orthogonal nodes

If multiple nodes use learned bases without constraint diversity, gradient descent will likely converge them all to the same useful basis (e.g., effectively all-FWHT). To prevent this, anchor at least 2 nodes to known fixed bases (FWHT + DHT minimum) so the learned nodes are forced to specialize in residual structure. Optional: add an orthogonality penalty between pairs of learned transforms (`||W_i · W_jᵀ - I||`) to push them apart in basis space.

## Prerequisite

The per-scale mixer transform ablation (§10 of [`other_post_release_plans.md`](other_post_release_plans.md#10-per-scale-mixer-transform-ablation)) is a prerequisite. It tests single-basis variants individually (FWHT vs DHT vs DCT vs learned vs identity) at L=1 baseline and tells us which transforms are individually competitive. The multi-basis variant should use the strongest individual performers as anchor nodes.

## Compute cost

The wavelet decomp + reconstruction (the expensive parts) are computed once, not N times. Only the FWHT-equivalents and per-node mixers are duplicated. For N=4, expect ~5-15% per-step compute increase since the mixer slot is a small fraction of total per-step compute (MLP dominates). For L=1 with the parameter-reduction bundle (mlp_expansion 20→10), the mixer share is larger and the percentage compute increase rises to ~10-20%.

## Compute to test

~5h for a first 5-epoch validation run with N=2 (FWHT + DHT) at L=1 baseline, after §10 establishes the strongest individual bases. Then ~5h for N=4 (FWHT + DHT + DCT + 1 learned) to measure scaling.

## Rationale (conjectural)

If multi-basis transform parallelization improves results, the most plausible mechanism is that each orthogonal basis represents the channel-axis features in a different coordinate system simultaneously. A Walsh basis groups features by binary-symmetry pattern, a cosine basis groups them by smoothness, and a learned-orthogonal basis groups them by whatever residual structure gradient descent discovers. The same input is losslessly rotated through all bases in parallel, and the combiner weights them per-scale based on which "perspective" matters most for the signal.

Standard transformer attention has no direct analog because (Q, K, V) projections conflate "the lens you use" with "the weights you compute" into a single learned operation. With a semantic embedding in particular (using plain-language, human-readable feature dimensions), this may make interpretability more tractable: a per-node, per-token-pair similarity score in the rotated basis answers "what does node K think these two tokens have in common?", making it possible to trace why two tokens are close or far depending on the conceptual lens/transform applied.

The wavelet decomposition continues to handle sequence-axis multi-scale structure, and the multi-basis nodes add feature-axis multi-perspective structure, factorizing the two cleanly. We don't yet know whether this is the actual mechanism if it increases performance, but if it does, testing this hypothesis directly becomes the natural follow-up.

## Open questions

1. Does N=2 (FWHT + DHT) already capture most of the value, or does N=4 (adding DCT and a learned basis) provide a meaningful additional lift?
2. Does mode collapse occur with all-learned bases, and does the FWHT+DHT anchor pair fully prevent it?
3. How does this interact with the parameter-reduction direction? At smaller per-node mixer sizes, the multi-basis duplication is a larger relative compute increase; if the smaller mixer has less spare capacity per basis, the per-basis specialization may be sharper.
4. Does the combined "multi-basis + semantic embedding" configuration unlock interpretability tooling that no single-basis configuration provides? This is the headline downstream research direction (see `reincorporate_large_semantic_embedding.md` and the README's "Combined Multi-Basis + Semantic Embedding (Interpretability Compound)" section).
