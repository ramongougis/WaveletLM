# Post-release: Prime-Power Wavelet Filterbank (mixed-radix dilations)

## Status

**Proposed — pre-research.** No code yet. The motivating concern (skip-bigram gap
coverage) is partly addressed by mechanisms already in the model, so the first job
is to sharpen what this feature would actually buy over the dyadic baseline, then
run a cheap kill-test. This doc is to be read alongside thorough background study of
the filterbank literature in [Background to study](#background-to-study) before any
implementation — the design choices below (causality, M-band vs à-trous, frame
redundancy, normalization) all hinge on that material.

## Motivation (the user's concern)

Current decomposition uses dyadic (radix-2) dilations only: 1, 2, 4, 8, …. The worry
is **skip-bigrams** `a … b` (b = current token, `…` = any gap ≥ 1) whose gap distance
is not a power of 2 — e.g. a syntactic dependency at gap 3, 5, or 6 — and whether such
a dependency is "reflected" by any dyadic wavelet. The proposal: build parallel
filterbanks at **prime-power radices** (2, 3, 5, 7, and if lightweight 11, 13) and feed
them into the per-scale mixer in a weighted-sum fashion alongside the dyadic scales.
Start with a large set (up to 11/13) as a screening run; if even the maximal set does
not move BPB, the hypothesis dies cheaply.

## The premise, sharpened (read before committing compute)

The intuition is geometrically appealing but needs three corrections, all of which
shape the experiment into something interpretable rather than a confounded capacity bump.

**1. Reachability is not the problem; directness/efficiency is.**
- The decomposition is **undecimated** — every scale stays at full length `T`
  ([model.py:2512](../model.py#L2512)). Scales are à-trous (dilated) filtered copies, not
  downsampled subbands. The lowpass `approx` at position `b` integrates a causal window
  that already includes `a` regardless of whether `b−a` is a power of 2. The skip-bigram
  information is **present**, carried in the lowpass and propagated by the residual stream.
- Dyadic dilations **compose to any integer lag** (binary representation: 3 = 1+2,
  5 = 1+4, …) given enough levels/depth. So primes do **not** add new reachable gaps.
- What primes add is a **more localized, directly-available basis function** at those
  lags — a sample-efficiency / inductive-bias benefit (the gap-3 pattern becomes a
  one-step feature instead of something composed across scales/layers), not a coverage one.

**2. Genuinely arbitrary (content-determined) gaps are out of scope for any fixed
dilation set.** "Any number of tokens ≥ 1" with a *variable* gap is an
induction/skip-trigram pattern. No finite set {2,3,5,7,11,13} covers it. That is a job
for either (a) cumulative/decaying memory that integrates over all gaps, or (b)
content-addressed matching (attention). WaveletLM is attention-free by design, so the
matched lever is **(a)** — which already exists as `decompose_bypass_ssm`
([model.py:2458-2478](../model.py#L2458-L2478)), a multi-pole EMA whose learned timescales
cover a *continuum* of effective gap-reaches. This makes the SSM-poles arm the natural
control (see below).

**3. Falsifiable prediction: the benefit (if any) shrinks with depth.** A single shallow
layer can only access the lags its filters provide directly; a deep stack composes the
missing lags across layers and the lowpass fills them in. WaveletLM runs at L=1–5, which
is exactly where prime dilations would help most if they help at all. **Log BPB across
depth** so the depth-decay prediction can be checked — a clean way to tell a real
inductive-bias effect from a capacity artifact.

## Design

Add a causal, **undecimated à-trous filterbank at dilation factor `m`** for each prime
radix `m ∈ {3, 5, 7, 11, 13}` alongside the existing dyadic (`m=2`) bank. Each radix
produces `[B, T, S_m, Cp]` coefficients at full temporal length `T`; concatenating on the
scale axis is trivial because no resampling is needed (the undecimated structure keeps all
scales at `T`). The new code is a **causal** dilation-`m` decompose/reconstruct — the
predict/update lifting steps must look strictly backward (same constraint as the current
causal lifting at [model.py:2505-2508](../model.py#L2505-L2508)).

Two fusion options, to be tested **separately** (they answer different questions):

### Option A — concat on the scale axis (early/mid fusion)

Stack all radices' scales so `S_total = Σ_m S_m`. The existing `cross_scale_gating`
`(S,S)` `scale_routing` einsum ([model.py:2632-2634](../model.py#L2632-L2634)) *is* the
weighted sum: it linearly blends radix-2 and prime-radix coefficients to form each scale's
gate, and the per-scale mixers + reconstruction do the rest. Maximum cross-basis
interaction. Conjectured better (the routing lets the bases interact during mixing, not
only at the end).

### Option B — two-branch weighted sum (late fusion)

Decompose → mix → reconstruct each radix branch independently, then blend the outputs:
`out = Σ_m w_m ⊙ recon_m`, with `w_m` a learned per-channel or input-dependent (sigmoid)
gate. Cheaper routing (no cross-bank `S²`), but the bases never interact during the
spectral stage — only at the end.

**What each outcome teaches:**
- A helps, B doesn't → the cross-basis **routing interaction** is the active ingredient.
- B helps, A doesn't → early mixing of incommensurate bases *interferes* (overcompleteness
  confuses the mixer); late fusion is the safe form.
- Neither moves the maximal {2,3,5,7,11,13} set → hypothesis dead cheaply. Assign real
  prior mass here: the [Mixer Transform Ablation](../README.md#done-mixer-transform-ablation)
  showed the model is near basis-indifferent (identity ≥ FWHT), which argues the spectral
  *basis* is a weak lever relative to per-scale mixing capacity.

## The parameter confound (this will bite the screening run)

Giving each new radix a full à-trous pyramid with its own `Cp×Cp` per-scale mixer explodes
both `S_total` and parameters. Rough count at block 512, Cp=2048: radix-2 ~9 levels, +3 ~6,
+5 ~4, +7 ~4, +11 ~3, +13 ~3 → `S_total ≈ 28` vs ~10 today, each scale a ~4.2M mixer ⇒
**+75M params/layer**. Any BPB move is then unattributable — depth/MLP buy parameters more
efficiently per the width/depth curves. Two fixes (both also serve the concern better):

- **Single-dilation lag filters, not full pyramids.** The concern is *small specific gaps*
  (3, 5, 7). Add one detail filter per prime at dilation¹ only (lags 3, 5, 7, 11, 13) rather
  than whole pyramids. Directly targets the skip-bigrams; far cheaper.
- **Share the mixer across the new scales** (one reused bank, or a single shared `Cp×Cp`),
  so adding primes adds scales but not many params, isolating the *basis* effect.

**Comparison must be capacity-matched.** Compare `{radix-2 only, S_total scales}` (push more
dyadic levels / duplicate to match scale count and params) **vs** `{2+3+5+7+11+13, same
S_total}`. Not dyadic-baseline vs the larger model. If primes win at equal capacity, the
basis diversity is real; if they tie, you bought parameters.

## Normalization & numerical stability

The user's conjecture is correct: both A and B NaN without per-radix normalization applied
**before recombination**, and the risk grows with `S_total` (more scales summed in the
identity reconstruction path → closer to fp16's 65504 ceiling). Where it goes, and what
already exists:

- **Option A:** the per-scale `decomp_norms[s]` ([model.py:2515-2518](../model.py#L2515-L2518))
  already normalize each scale before the routing sum — ensure `wavelet_decomp_norm` is
  **on** and the `ModuleList`s (`decomp_norms`, `recon_norms`, `scale_mixers`,
  `history_gains`) are extended to `S_total`. **Block-identity-init the `scale_routing`
  matrix** (identity within each radix, ≈0 across) so high- and low-energy radices are not
  violently mixed at step 0. Consider a `1/√S_total` factor on reconstruction; `fht_input_cap`
  remains the backstop.
- **Option B:** LayerNorm each branch's reconstruction *before* `Σ_m w_m ⊙ recon_m`, else the
  higher-energy branch dominates and the gate cannot recover.

## Control arm (run alongside, possibly the better lever)

Before or beside the prime filterbank, **raise `decompose_bypass_ssm_poles`** (default 4).
More multi-pole EMAs at learned timescales is the *continuous*, parameter-cheap version of
"cover many gap distances," and it is the mechanism actually suited to arbitrary-gap
dependencies. If a handful of extra poles captures the skip-bigram signal, the prime
filterbank is solving a problem the SSM bypass already handles better.

## Experimental plan (cheapest → most involved)

All runs 1-epoch WT103, L≤2 unless probing the depth-decay prediction.

1. **Control arm:** `decompose_bypass_ssm_poles` ∈ {4 (baseline), 8, 16}. ~3 runs. If extra
   poles already close the gap the prime bank is chasing, deprioritize the bank.
2. **Option A screening (kill-test):** maximal {2,3,5,7,11,13}, single-dilation lag filters,
   **shared** mixer, per-scale norm, block-identity routing init. Compare against a
   capacity-matched dyadic-only run. If no movement at the maximal set → stop.
3. **Option B screening:** same radix set, two-branch late fusion, per-branch norm. Capacity-
   matched. A-vs-B comparison per the decision table above.
4. **Incremental build (only if 2 or 3 moved):** {2,3} → {2,3,5} → {2,3,5,7} → … to find where
   the gain saturates and *which* primes carry it (disentangles the contributors).
5. **Depth-decay check (only if a winner survives):** rerun the winner at L∈{1,2,3} and confirm
   the BPB gain shrinks with depth (the inductive-bias signature) vs. staying flat (a capacity
   artifact).

## Definition of done

- Either: a mixed-radix configuration that beats the capacity-matched dyadic baseline by
  ≥0.005 BPB at 1-epoch WT103, reproducibly, with the gain attributable to the basis (not
  parameters) per the matched comparison and the depth-decay signature. Promote to a 5-epoch
  run and the release roadmap.
- Or: a written "we tried radices up to 13 under A and B; no movement at matched capacity"
  result, which is itself a useful negative (corroborates the basis-indifference finding) and
  redirects gap-coverage effort to the SSM-poles arm.

## Background to study (prerequisites — read before implementing)

The design questions (causal M-band vs à-trous, frame redundancy, per-radix normalization,
where the benefit comes from) are settled topics in the filterbank literature. Study these
before writing code:

- **À-trous / stationary (undecimated) wavelet transform** — why WaveletLM's scales are full-
  length and how dilation-`m` filtering generalizes. Shensa, "The Discrete Wavelet Transform:
  Wedding the À Trous and Mallat Algorithms," *IEEE TSP* 40(10), 1992.
- **M-band (radix-M) wavelets** — the proper theory of factor-`m` (m>2) decompositions, the
  one-lowpass-plus-(m−1)-highpass filter structure, and regularity conditions. Steffen, Heller,
  Gopinath, Burrus, "Theory of Regular M-band Wavelet Bases," *IEEE TSP* 41(12), 1993.
- **Overcomplete / rational-dilation transforms** — what happens to a frame when you union
  incommensurate dilations (redundancy, non-orthogonality, stability). Bayram & Selesnick,
  "Frequency-Domain Design of Overcomplete Rational-Dilation Wavelet Transforms," *IEEE TSP*
  57(8), 2009.
- **Tunable-Q / multi-Q filterbanks** — the precedent that denser, non-octave frequency
  sampling helps some tasks, and the cost it carries. Selesnick, "Wavelet Transform With
  Tunable Q-Factor," *IEEE TSP* 59(8), 2011; Andén & Mallat, "Deep Scattering Spectrum,"
  *IEEE TSP* 62(16), 2014 (Q>1 / multiple wavelets per octave).
- **Wavelet packets** — the adjacent way to enrich the basis (sibling of this proposal; already
  listed under Other Post-Release Plans). Coifman & Wickerhauser, "Entropy-Based Algorithms for
  Best Basis Selection," *IEEE Trans. Information Theory*, 1992.

## Open questions

1. Is a single-dilation lag filter at gap `m` actually distinguishable, after the per-scale
   `Cp×Cp` mixer, from a linear readout the dyadic bank can already form? (If the dyadic
   per-position feature vector already spans lag-`m`, primes add nothing even at L=1.)
2. Does the screening result reproduce on a tiny sanity config (C=512, L=1) so iteration is
   ~1h rather than full-scale?
3. For M-band done *properly* (decimated, m>2) rather than à-trous single-dilation, do the
   (m−1) highpass subbands per level add anything the single-dilation lag filter doesn't? Worth
   a sketch only if the cheap à-trous form shows signal.
4. If the SSM-poles control arm closes the gap, is there any residual case for the prime bank at
   all, or does it fold entirely into "tune the bypass memory"?
