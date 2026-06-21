# Post-release: Long-context WaveletLM — decimation, scaling & length generalization

> The **retrieval mechanism** is split into the sibling [long_context_retrieval.md](long_context_retrieval.md) (wavelet-keyed kNN-LM / FSRR). This doc keeps the *engineering* side — the decimated transform, the memory ladder, and length generalization.

## Status

**Proposed — post-release.** Prompted by SubQ-1.1-Small / Subquadratic Sparse Attention (SSA),
which trains at 1–2M tokens and generalizes single-needle retrieval to 12M. WaveletLM is *natively*
subquadratic, so the scaling axis is largely solved by construction; this doc is about the two things
that actually gate a long-context WaveletLM: **content-dependent retrieval** (the likely weakness) and
the **undecimated memory blowup** (the engineering wall past a few million tokens). Read alongside the
[block-size / length-generalization experiment](#block-size-experiment) below, which is the cheap
prototype that decides whether any of this is worth pursuing.

## Competitive framing (vs SSA)

SSA's pitch has two prongs, and WaveletLM lands on opposite sides of them:

1. **Efficient scaling — WaveletLM wins by construction.** SSA's main critique of DeepSeek is that the
   Lightning Indexer's *selection* stage is quadratic, reintroducing O(n²) after providing only scalar
   savings. WaveletLM has **no selection stage at all** — the wavelet mixer is ≈ O(T·log T) end-to-end,
   no learned indexer, no sparse-attention machinery. On SSA's own axis of criticism, WaveletLM is a
   *cleaner* subquadratic story.
2. **Content-dependent retrieval — WaveletLM is in the SSM camp (the likely weakness).** SSA's
   Background buries Mamba/RWKV/SSMs with: *"a state that compresses the past cannot also preserve it
   verbatim, and long-context retrieval requires the latter."* WaveletLM is a **fixed-basis *position*
   mixer** — the crawl learns *lags*, the per-scale mixer routes by *relative position*, the
   decompose-bypass is *recency*. None of it is content-addressed. So WaveletLM sits in exactly the
   family SSA argues fails RULER/NIAH. The only content-addressed component is the PKM/FwPKM
   associative memory (key→value), and like an SSM state it is fixed-size / lossy.

**Implication:** chasing raw context length races on the axis WaveletLM already wins and ignores the one
it would lose. The gate is **retrieval at length**, not length itself.

## The retrieval gate (do this first)

Before any multi-million-token work, answer: *can WaveletLM do needle-in-a-haystack at modest long
context (64K–256K) at all?* A position-mixer may fundamentally cap out on single-needle retrieval the
way pure SSMs do. Two outcomes, both useful:

- **It can't.** Then WaveletLM is a long-context *aggregation/reasoning* model (strong on RULER
  frequency/multi-hop tasks, weak on exact single-needle) — a legitimate but different niche than SSA's
  retrieval pitch, and the honest one to claim. A content-addressed mechanism (scaled PKM/FwPKM, or a
  new component) would be the path to close the gap — and PKM/FwPKM are **data-hungry** (associative
  memory only earns its parameters when not data-starved; see the iso-param / MLP-capacity findings),
  so they get their fair test only in the big-combined-dataset regime, not on WT-103. The concrete
  proposal for that content-addressed mechanism — a **wavelet-keyed kNN-LM / fine-scale retrieval
  (FSRR)** — is the sibling [long_context_retrieval.md](long_context_retrieval.md).
- **It can** (even partially). Then the length-generalization story (train short, eval long) becomes
  worth the engineering, and the rest of this plan applies.

## Wavelet decimation: what we have vs. what long context wants

WaveletLM currently uses the **undecimated** (à-trous / "stationary") transform: filter at each level
but never downsample — *dilate* the filter instead (à trous = "with holes," which is literally the
crawl's dilated taps). Every scale stays **full length T**. For block 256 / 7 levels: 8 scales × 256 =
**2048** coefficient-positions per channel — overcomplete (≈ S× redundant) and **shift-invariant**.

The **decimated** (critically-sampled / Mallat) DWT downsamples by 2 at each level, so each scale is
half the previous: 128, 64, 32, …, ≈ **256 total** coefficient-positions — same count as the input, no
redundancy, but **shift-variant**.

| | Undecimated (now) | Decimated (DWT) |
|---|---|---|
| Downsample / level | No (dilate filter) | Yes (÷2) |
| Coefficients | **T·S** (≈ S× redundant) | **T** (critically sampled) |
| Scale lengths | T, T, T, … | T/2, T/4, T/8, … |
| Memory / compute | O(T·log T) | O(T) |
| Shift-invariance | Yes | No |

Decimation removes the `S = log T` factor from **both memory and compute** (~24× at 16M). The costs are
the architectural changes it forces:

- **Variable-length scales** — the per-scale mixer must operate on scales of different lengths.
- **Cross-scale routing alignment** — the `(S,S)` routing matrix currently mixes scales position-by-
  position because they are all aligned at length T. With decimation, scales are at different
  resolutions and must be upsampled/aligned before mixing (the trickiest change).
- **Crawl semantics** — the per-position dilated-lag interpretation shifts at downsampled scales.
- **Reconstruction** — inverse DWT (upsample + filter) replaces the current iFHT/reconstruct.
- **Decompose-bypass** — the coarse-scale contributions become downsampled summaries.

It is the textbook transform, so it is well-understood, but it is a real redesign of the spectral block,
not a config flag.

## The coarse-decimation insight (from the crawl probe)

The [Crawl Dilation Probe](../README.md#crawl-dilation-probe-prime-power-wavelets-measured) found that
**coarse scales are smoothers, fine scales are precise lag detectors.** Decimation costs resolution
exactly where it is cheapest — the coarse scales, which are *already averaging* — and is most harmful at
the fine scales, where shift-invariant precision is used. So the principled form is **not** "decimate
everything." It is:

> **Decimate the coarse scales, keep the fine scales undecimated.**

This saves the most memory (the coarse-scale × full-T copies are the worst T·S offenders) while
preserving short-lag precision where the data says it matters. The interpretability result directly
designs the long-context transform — and it gives a clean memory/quality knob (how many fine scales to
keep full-resolution).

## Memory math

`stacked_coeffs` is `[B, T, S, Cp]`; per-sequence (B=1, fp16) cost drives the wall.

| Regime | C=2048 undecimated | C=2048 decimated | On 8× B200 (1,536 GB) |
|---|---|---|---|
| 1M (S≈20) | ~82 GB | ~4 GB | undecimated fits, sharded |
| 2M (S≈21) | ~172 GB | ~8 GB | undecimated fits, sharded |
| 16M (S≈24) | ~1,650 GB | ~69 GB | **decimated only** |

Two conclusions: (1) **at SSA's training scale (1–2M), undecimated C=2048 already fits 8 B200s** with
sequence parallelism + gradient checkpointing + offload — no C drop, no decimation needed; (2) at
many-M, **decimation (not a smaller C) is what makes C=2048 survive.** Keep the width at the knee
throughout; the earlier "shrink C to afford context" idea was only ever a workaround for the
undecimated blowup that decimation fixes at the source.

## Hyperparameters at long context

- **LR is width-bound (~1/C) and context-invariant.** The per-block LayerNorms + residual normalize
  gradient magnitudes regardless of T or S (same reason depth doesn't move the LR). Set the LR by C
  (C=1024 ≈ 0.05, C=2048 ≈ 0.0225) and hold it across context lengths; decrease only empirically if a
  NaN appears, not as a planned function of T.
- **Levels = log2(block_size); per-scale widths = levels + 1 entries.** +1 level and +1 width entry per
  context doubling; new coarse scales get 0.5 (or lower — they are smoothers).

## Block-size experiment

The cheap prototype (see the README [Block-Size Extension](../README.md#block-size-extension--length-generalization)
section): C=1024 / L=5 on the 5090, **MBS=1 + grad-accum** (the binding knob — frees ~8× activation
budget vs MBS=8, so the max block jumps from ~512 to ~2048–4096), levels = log2(block), widths extended,
LR 0.05. Measure two things:

1. **Block-size robustness** — does BPB hold (or degrade tolerably) as the training block grows?
2. **Length generalization** — train at block 256/512, *evaluate* at 1024/2048/4096 (and a small NIAH
   probe). The *train-short/eval-long* curve is the SSA-relevant property: it decides whether the
   2M-train → 12M-eval story is even open for WaveletLM.

This runs on the *undecimated* transform — fine at these scales — so it is independent of the decimation
work and a strictly cheaper first step.

## Ordering

1. **Retrieval probe** (64K–256K NIAH) — the gate. If it fails, this is an aggregation model, not a
   retrieval one; re-scope honestly.
2. **Block-size / length-generalization prototype** (undecimated, 5090) — does the architecture extend
   context gracefully and generalize past the training window?
3. **Undecimated 1–2M training** (8 B200s, the memory ladder) — only if 1–2 pass.
4. **Decimated transform** (coarse-decimation hybrid) — the many-M unlock, only when memory at the
   target context actually demands it.

C is held at the width knee throughout; nothing here requires shrinking it.

## Open questions

1. Does a fixed-basis position mixer have a hard ceiling on single-needle retrieval, or can PKM/FwPKM
   (data-rich) lift it?
2. Does WaveletLM length-generalize at all — and does the wavelet's relative-position structure help or
   hurt vs absolute-position attention?
3. For the coarse-decimation hybrid: how many fine scales must stay undecimated before retrieval/quality
   degrades, and how does the cross-resolution routing alignment behave in practice?
4. Does shift-variance (introduced by decimation) measurably hurt an LM, or is it irrelevant once the
   per-scale mixer is learned?
