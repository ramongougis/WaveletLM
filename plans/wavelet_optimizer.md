# Wavelet-Domain Optimizer for WaveletLM (self-consistent learned-basis GWT)

## Status 

**Proposed — candidate for a pre-release test on the C=1024 baseline (2026-06-27).** A memory-efficient
optimizer that compresses gradients in the wavelet domain (GWT), with a novel twist: use WaveletLM's **own
learned shared lifting wavelet** as the gradient-compression basis (self-consistent). Builds on Wen et al.,
"Breaking Memory Limits: Gradient Wavelet Transform" (arXiv:2501.07237, Jan 2025).

## Core idea

**GWT (Wen et al. 2025)** applies a wavelet transform to the *gradients* to compress the *optimizer state*:
`[A,D] = G·H` (Haar/DB2 matrix `H`), update the Adam moments on the **approximation `A` only** (half/quarter
the size), scale the detail `D` by the same second moment, reconstruct `G̃ = [Ã,D̃]·H̃`, step. Matches or beats
full-rank Adam at **65–75% less optimizer memory + ~2× throughput**; the "beats full-rank" is implicit
regularization (detail-coefficient noise suppression).

**The twist (the head-turner):** instead of a fixed Haar/DB2 basis, use **WaveletLM's learned, shared lifting
wavelet** as `H` — the *same* basis the model learned for its forward pass now compresses its own gradients in
the backward pass. A frequency-domain model trained by a frequency-domain optimizer, in a **single,
self-consistent, learned** basis. No prior work does this.

## Why WaveletLM is the natural host

- Already a frequency-domain model (the mixer operates per-scale) → GWT *completes* the picture, not grafts on.
- The lifting is **shared across all layers** → one basis for every gradient; clean.
- No MLP / no attention → the GWT surface is the **mixer** (the bulk now), `proj_out`, and lifting — **not** the tied embedding/LM head (see *Where to apply GWT* below).

## Where to apply GWT — leave the tied head full-rank

The tied **W** is now the **biggest single parameter** (51M of 250M ≈ 20% with no MLP) and it learns **hardest
and earliest** (it's the last linear before the loss). Its gradient is structurally weird — **output-side
dense** (every vocab row gets a gradient through the logits), **input-side sparse** (only the batch's tokens).
That doesn't fit GWT's "compress a dense, smooth weight-matrix gradient" assumption. **Recommendation: leave
the tied head full-rank — do NOT GWT it** (matching the paper, which only GWTs attention/MLP). Apply GWT to the
**mixer** (the bulk of the rest) and possibly **`proj_out` / lifting**. Compressing the most-critical,
earliest-learning parameter's gradient is exactly where you'd lose the most.

## Two tiers (risk / effort spectrum)

- **Tier 1 — vanilla GWT (fixed Haar): ~2–4 days.** Adapt the reference impl
  (github.com/zqOuO/GWT) to WaveletLM's optimizer. Proven, low-risk; delivers the memory/throughput win + the
  "wavelet model + wavelet optimizer" unification story. **The floor.**
- **Tier 2 — learned-basis GWT (the novelty): +1–2 weeks + research risk.** Use the model's learned lifting as
  `H`. The head-turner story, but with real hurdles (below).

## The hard part (Tier 2)

1. **Orthogonality vs perfect reconstruction.** GWT's moment-scaling (its Eq. 6: divide `D` by `√(V+ε)`) leans
   on `H` being **orthogonal**. The lifting is **invertible** (perfect reconstruction) but not necessarily
   **orthonormal** → the correction becomes approximate and may need rework (orthogonalize the lifting via QR
   for the gradient transform, or re-derive the scaling for the non-orthogonal case). **Main research risk.**
2. **Is a *language* basis a good *gradient* basis?** The lifting was learned for token-axis language
   structure; gradients are a different statistical object. The paper's own open problem (b) asks "can a
   specialized wavelet for gradients be developed?" — your learned lifting *might* be that, or might not.
   Empirical bet.
3. **Frozen vs co-evolving basis.** The lifting changes during training; a moving compression basis could be
   unstable. Use a **frozen snapshot** (from the trained C=1024 model) for the first test — ties to the
   [[frozen-wavelet transfer]] idea.
4. **Axis repurposing.** The lifting is a 1D conv over the *token* axis; here it's applied over the gradient's
   *parameter* (column) axis — mechanically fine (1D filters apply to any axis), but a different axis than it
   was trained on.

## Implementation effort (rough)

| Piece | Effort | Risk |
|---|---|---|
| Tier 1 vanilla GWT on Adagrad (adapt reference + param selection + AMP/fp32 care + numerical test) | **~2–4 days** | low |
| Tier 2 learned-basis (orthogonality handling, frozen snapshot, moment-correction rework, stability) | **+1–2 weeks** | moderate–high |
| Testing on C=1024 (memory: instant; convergence A/B @1ep: ~½ day each; full 5ep BPB A/B: ~2–4 days) | **~1 week** | — |
| **Full head-turner story** | **~3–4 weeks** | — |
| **Floor (vanilla win + unification story)** | **~1 week** | low |

## Adagrad vs Adam caveat

WaveletLM uses **Adagrad (1 state)**; GWT's headline 75% is on **Adam (2 states)**. On Adagrad you compress one
accumulator → real but smaller savings. The biggest memory win comes if you *also* move to a heavier optimizer
(Adam/Muon) — and a GWT result could **justify** that move (GWT makes Adam affordable). Worth testing GWT+Adam
alongside GWT+Adagrad.

## Test protocol (C=1024 baseline, pre-release)

1. **Memory check (instant):** does the optimizer state shrink as predicted? (one step → read VRAM).
2. **Convergence A/B (1ep):** GWT (Haar first, then learned-basis) vs baseline — does it match/beat the curve?
3. **Full BPB A/B (5ep):** the 0.9884 baseline vs GWT — BPB within noise (or better)?
4. **Throughput:** tokens/s vs baseline.

Gate each tier on the previous (vanilla works → try learned-basis).

## Honest framing (the PlanarLM rule)

The pitch — *"we compressed gradients with our own learned, shared lifting wavelet basis"* — is a genuine
head-turner **if it works**. But:
- **Vanilla GWT is the floor** (proven win, low risk) — land it first; it's a real result regardless.
- **The learned basis is an untested bet** with a concrete hurdle (orthogonality + is-language-lifting-good-
  for-gradients). **Do not write the abstract before the C=1024 result is in.** If it works → headline. If it
  ties vanilla → still a clean "self-consistent wavelet optimizer" result. If it fails → vanilla GWT + an
  honest negative on the learned basis.
- It rides the current memory-efficient-optimizer wave (GaLore / Fira / APOLLO / GWT) — timely, *and* reviewers
  know the area well, so the novelty must be precisely the **learned, self-consistent basis**, not "we used a
  wavelet" (that's GWT).

## Precedents

- **GWT** (Wen et al. 2025, arXiv:2501.07237) — the direct basis; reference impl github.com/zqOuO/GWT.
- **GaLore** (Zhao et al. 2024), **Fira** (Chen et al. 2024), **APOLLO** (Zhu et al. 2024) — the low-rank /
  SVD-free memory-efficient optimizer line GWT positions against.

## Relationship

- Trains WaveletLM → lives in **WaveletLM/plans** (not the WLM diffusion repo).
- Ties to [[frozen-wavelet transfer]] (a frozen lifting snapshot is the natural basis source) and the no-MLP
  result (a leaner training surface — mixer + lifting, no MLP).
