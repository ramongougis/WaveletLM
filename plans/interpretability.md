# WaveletLM Interpretability — Phase 1: The Privileged-Basis Studies

> **Status (2026-07-13): plan of record for the interpretability on-ramp.** Designed to run
> **during the compute pause** — every study below is CPU-viable on a workstation or laptop
> against existing checkpoints (WaveletLM Mini, 73M). Compute cost ≈ $0. This is the deep-stream
> counterpart to the on-rails run queue (D-ladder → K5 → M2), and the seed of paper 2.

## Why now

Three things converged:
1. **The checkpoints exist.** The D-ladder produced strong small models (D1: 1.0103 sliding BPB
   at 72.89M, `logs/wikitext-103_2026-07-12_14-20-46/`) whose full inference footprint is
   ~292MB of fp32 weights — CPU-runnable, no VRAM exposure on fragile hardware.
2. **The budget pause** (~1 month) idles the deep stream unless it has a $0 workstream.
3. **The thesis is finally testable** (below) — and it's the claim that motivates the entire
   interpretability program, so it should be tested *first*, before any SAE machinery is bought.

## The thesis

**WaveletLM's activations should be easier to interpret than a transformer's, by construction.**

The SAE program exists because a transformer's residual stream is an *unprivileged, superposed
basis* — features are rotated into arbitrary directions and must be un-mixed with a learned
dictionary, which always carries reconstruction error. WaveletLM's block activations arrive
**pre-factorized**: coefficients organized as **scale × channel × position**, with per-scale
LayerNorms and channel-wise gates that make channels a privileged basis, and — critically —
**perfect reconstruction**: the coefficients are a *complete, invertible* description of the
layer's computation. Ablate a coefficient, resynthesize exactly, measure the causal effect.
No reconstruction-error caveat, ever.

If the thesis holds, WaveletLM skips the most expensive and most caveated stage of the
transformer interpretability pipeline. If it fails (channels turn out polysemantic), that
tells us precisely what the second GPU is for (SAE training) — either outcome sets the
program's direction cheaply.

**Sharpened form (2026-07-13):** the **scale axis is privileged by construction** (the
decomposition is hard-wired); the **channel axis is privileged-ish** — channel-wise gates and
per-scale norms create basis-alignment pressure (the same mechanism that makes transformer MLP
neurons more interpretable than residual-stream directions), but nothing prevents Cp channels
from superposing more than Cp features. Study 6 (the SAE null test) measures that residual
doubt quantitatively, per scale and per layer.

## Phase 0 — Instrument: `interpretability/coeff_dump.py`

One script, one job: run the model forward over N tokens of WT-103 (val/test), hook the
per-layer post-decompose coefficient tensors, and write them to disk as compressed shards.
All downstream studies are numpy/sklearn/matplotlib on the shards — the model is loaded once.

- **Verify shapes in model.py first** (house rule: no guessing). The decompose path stacks
  coefficients as `stacked_coeffs[:, :, s, :]` with per-scale `decomp_norms[s] = LayerNorm(Cp)`
  (model.py ~2104–2111, ~2230), i.e. expected shape `(B, T', S, Cp)` — confirm T' semantics
  (per-scale coefficient counts under the dyadic grid) before writing the dump.
- Capture points: post-decompose (pre-mixer) and post-mixer (pre-reconstruct), per layer.
  Also dump the per-scale gate/routing activations if cheap.
- Sharding: one `.npz` per (layer, window-batch); target ≤ ~200MB per shard; a manifest JSON
  with config hash + checkpoint path so shards are self-describing.
- Budget check *(estimate)*: Mini at C=512, L=10, S=8 scales → dumping ~100K tokens of
  coefficients at two capture points is a few GB compressed — fits any disk; CPU forward for
  ~400 windows of 256 tokens is minutes, not hours.
- **Crash-proofing (laptop rule):** CPU inference only (`--device cpu`), windows processed
  in a resumable loop (skip shards that already exist). Never hold more than one window's
  activations in memory.

Checkpoints to dump, in priority order:
1. **Mini** (best D-ladder rung available — D2 if it lands well, else D1) — the workhorse.
2. **K0 (C=100)** — the starved-width case, and methodologically the **positive control**: at
   C=100 superposition pressure is maximal (far fewer channels than plausible features), so if
   the Study-2/6 instruments can't detect superposition *there*, a clean result at C=512 is
   evidence of instrument insensitivity, not of a privileged basis. The methods must light up
   on K0 before a null at Mini counts.
3. **SP1 Small (C=1024, 239M)** — flagship cross-check (956MB fp32; still CPU-fine, slower).
   Mini→SP1→K0 also gives the **width trend** of every metric (does monosemanticity grow with
   channel headroom?), turning each study into three points instead of one.

## The studies

### Study 1 — Coefficient census *(descriptive; zero theory prerequisite)*
Per (layer × scale × channel): magnitude distributions, sparsity/kurtosis, dead-channel
counts, cross-layer trends. Deliverable: the "map" figures every later study navigates by.
Effort: a day of numpy/matplotlib once shards exist.

### Study 2 — Monosemanticity of per-scale channels *(the headline question)*
For each (layer, scale, channel): collect the **top-k max-activating contexts** from the
dump; judge coherence (human read first; LLM judge later if scaled); score each channel's
consistency. Compare qualitatively against the literature's transformer-neuron baseline
(known to be heavily polysemantic).
- **Control (mandatory, per the "interpretability illusion" caution — Bolukbasi et al.):**
  score **random-direction baselines** through the identical top-k pipeline. Top-activating
  examples can look coherent even for random directions; a channel only counts as
  monosemantic if it beats the random-direction distribution, not just if it "reads well."
- **Decision rule:** a substantial interpretable fraction (vs. control) → privileged-basis
  thesis holds → paper-2 headline, SAEs deprioritized. Mostly polysemantic → SAEs needed →
  sizes the second-GPU budget with evidence instead of assumption.
- **Prediction on record** *(estimate)*: partial monosemanticity, with **coarser scales more
  interpretable than finer ones** (coarse coefficients aggregate longer spans → more
  concept-like; finest scales → local orthography/syntax, likely cleaner but lower-level).

### Study 3 — Exact causal ablation *(the invertibility advantage)*
Zero (or scale) individual coefficients / channels / whole scales → resynthesize → measure
Δnext-token distribution and ΔBPB on targeted contexts. Attribution is *exact* because
reconstruction is exact — the differentiator vs. SAE-based causal claims.
- Scale-level sweep first (S=8 ablations × L=10 layers — tiny), then channel-level on
  Study-2's most/least monosemantic channels.
- **Metric discipline (per Zhang & Nanda):** report both a probability-space metric (ΔBPB /
  Δp(correct)) and logit-diff on targeted contexts — the two disagree in known ways and
  reviewers will ask.
- **Prediction on record** *(estimate)*: coarsest-scale ablation degrades long-range
  coherence/topic maintenance disproportionately; finest-scale ablation hits local grammar
  and spelling; mid scales are where the interesting semantics live.

### Study 4 — Scale-role dissection via linear probes
Logistic-regression probes on cached per-scale coefficient streams for cheap linguistic
labels (POS, sentence position, quote/list/heading context, topic id). Pure sklearn on
shards. Deliverable: a **scale × feature heat map** — "which temporal scale encodes what."
This story has no transformer analogue (they have no scale axis); it is uniquely ours.
- **Control (mandatory, per Hewitt & Liang):** every probe ships with a control task
  (shuffled labels / random feature assignment); report **selectivity** (real minus control
  accuracy), never raw accuracy — otherwise the probe may be learning the task, not reading
  the model.

### Study 5 — What did the lifting learn?
Compare the learned shared lifting against its Haar init: per-level filter taps, frequency
responses, the crawl's dilation-softmax weights per level (extends the crawl probe), and the
shrinkage λ-map's "protect-ends / squeeze-middle" finding (README §Coefficient Shrinkage).
This is the study that **meets the Strang & Nguyen reading in the middle** — frequency
responses of learned filter banks are exactly its material, making the theory pass
observation-driven rather than prerequisite-driven.
Also weights-only and free: the learned **(S, S) cross-scale routing matrices** per layer
(identity-init) — how far each has moved from identity, and which scales read which. A
scale-interaction graph per layer, straight from the checkpoint, no dump required.

### Study 6 — The SAE null test *(SAE as ruler, not microscope)*
Train **small per-scale SAEs** (input dim Cp=512, dictionary 2–8×, on cached shards — pilot
scale is CPU/modest-GPU feasible, NOT second-GPU work) and measure **alignment of the learned
dictionary to the native channel axes**: max cosine per feature, permutation-matched, reported
per scale × layer.
- **Logic:** if the channel basis is privileged, the SAE largely rediscovers the axes it was
  given ("lights up the same activations") → the thesis gets a *number* instead of an
  impression. If the SAE finds heavily rotated, more-interpretable features → superposition
  lives within the channel dim, and we've measured exactly where SAE machinery is needed.
  Either outcome is a paper-2 result; convergent validation with Studies 2–4 either way.
- **Prediction on record** *(estimate)*: moderate-to-high alignment overall, higher at coarse
  scales than fine, consistent with Study 2's prediction.

## Reading map (one instrument at a time, consumed as its study begins)

| paper | for | takeaway needed |
|---|---|---|
| Toy Models of Superposition (Elhage et al. 2022) | thesis; read first | privileged vs unprivileged bases — the framework this program tests on a new architecture |
| Towards Monosemanticity (Bricken et al. 2023) | Studies 2, 6 | how feature interpretability is judged in practice; the SAE recipe the null test borrows |
| An Interpretability Illusion for BERT (Bolukbasi et al. 2021) | Study 2 | top-k inspection fools you; random-direction controls are mandatory |
| Towards Best Practices of Activation Patching (Zhang & Nanda 2023) | Study 3 | ablation metric choices (logit-diff vs prob; noising vs denoising) and their failure modes |
| Linear probes (Alain & Bengio 2016) + Control Tasks (Hewitt & Liang 2019) | Study 4 | the probe method + selectivity: is the probe reading the model or learning the task? |
| SAEs Find Highly Interpretable Features (Cunningham et al. 2023) | Study 6 | compact SAE-on-LM reference to pair with Bricken |

*Titles from memory as a reading map; fetch and verify each primary source before any claim
from them enters the README (house rule).*

## Practical notes

- **Order:** Phase 0 → Study 1 → 2 → 3, with 4 and 5 interleaved as energy dictates. Studies
  2 and 3 together are the thesis test; everything else is context for them.
- **The paper-2 methodological spine** the studies assemble into: *claim (thesis) → causal
  test (Studies 2–3) → correlational cross-check (Study 4) → adversarial instrument turned on
  ourselves (Study 6)*. An SAE trying and failing to improve on the native basis is the
  strongest form of the claim — the transformer world's own microscope finding nothing to fix.
- **Phase-2 candidate (noted, not scoped):** the decompose-bypass cross-window state — the
  model's *only* channel across window boundaries, i.e. its entire working memory. Reading
  what survives the boundary is a uniquely-WaveletLM study; it needs its own instrumentation,
  so it waits for Phase 2.
- **Phase-2 candidates from the embedding↔wavelet co-adaptation thread (2026-07-20; both run
  on existing dumps):** (a) **token spectral fingerprints** — per token id, mean per-scale
  coefficient signature over occurrences; correlate fingerprint similarity with embedding
  cosine; test whether BPE-related pairs cluster (Ramon's routing-similarity claim,
  quantified); (b) **identity/context variance split** — per scale, decompose coefficient
  variance into between-token vs within-token components (ANOVA over shards). *Prediction on
  record (est.): fine bands token-identity-dominated, coarse bands context-dominated.* Note
  also: the scale-role map is tokenizer-dependent (vocab granularity sets the semantic
  wavelength of every band), so the planned semantic-embedding variant retunes the whole
  instrument, and a fixed human-readable embedding would pre-label channels — compounding
  with the privileged-basis thesis.
- **Streams discipline:** this plan is the *deep stream*. The run queue (D2/D3 → K5 → M2 → M4)
  stays on rails and is recorded as usual; nothing here blocks or waits on it.
- **Recording convention:** results accumulate in this file first (tables + shard/notebook
  pointers), promoted to a README interpretability section only when a claim survives its
  own decision rule. Predictions above are on record to be scored, per house style.
- **Naming:** the 73M C=512 tier is referred to as **WaveletLM Mini** going forward;
  **Nano is reserved** for the future ultra-light picked at the C-knee (C≈200–300).
- **Non-goals for Phase 1:** *large-scale* SAE training (the tiny per-scale null-test SAEs of
  Study 6 are in scope — they're laptop-scale), cross-model dictionary comparisons, GPU-scale
  probing, SOW/REAP integration — all deferred until the thesis test says they're needed.

## Results log

**2026-07-15 — Phase 0 shipped; first census (Study 1 opens).** `tools/interpretability/
coeff_dump.py` operational; 49,152 WT-103-val tokens × 10 layers of post-decompose
coefficients dumped from Mini/D2 (`.interp/mini_d2`, shards fp16, manifest'd).

- **Finding 1 — the U-shaped per-scale gain profile.** Per-scale mean |post-norm coeff|,
  layer 0: `s0=0.622, s1=0.346, s2=0.344, s3=0.338, s4=0.275, s5=0.243, s6=0.359, s7=0.512`;
  layer 9: `s0=0.436, s1=0.205, s2=0.227, s3=0.254, s4=0.232, s5=0.213, s6=0.218, s7=0.284`.
  Approximation band and finest detail run loud, middle scales quiet — and since these are
  post-LayerNorm magnitudes (≈0.8 expected under identity affine), the deviations are the
  **learned per-scale gains**: the model's own volume knobs. **Converges with the shrinkage
  λ-map's independent "protect-ends / squeeze-middle" finding** (README §Coefficient
  Shrinkage) — two unrelated instruments, same geometry. Depth trend: layer 9 gains
  uniformly ~30–40% below layer 0 (the stack quiets with depth).
- **Finding 2 — census statistics converge absurdly fast.** An 8-window (2K-token) sample
  reproduces the 192-window census to ±0.005 per scale. Census-class probes are reliable at
  trivial sample sizes. *(Correction for the record: the apparent "cross-dataset" agreement
  of the first two censuses was nothing of the kind — wikitext-2 and wikitext-103 share
  their val/test sets (Merity et al.); it was the same text at two sample sizes.)*
- **Prediction on record (Pile cross-domain census)** *(estimate)*: post-norm per-scale
  gains are largely weight-borne (LayerNorm standardizes per position; the affine is fixed),
  so the Pile census should match WT-103 within ~±0.02 per scale. If confirmed → the gain
  profile is readable **from the checkpoint alone**, and Study 1 gains a weights-only
  companion probe (census of `decomp_norms[s]` γ/β directly — no forward passes at all).
  If it *deviates*, the deviation localizes where input-distribution shape (not scale)
  differs per channel — itself informative. Command: `--dataset pile
  --dataset_max_tokens 300000` (offline via the smoke cache).
- **Prediction scored (same day): CONFIRMED.** Pile census (16,384 tokens via the offline
  smoke cache) matches WT-103 within ±0.015 on all 16 layer×scale cells, most within ±0.005
  → the gain profile is **weight-borne**; the zero-inference γ/β companion probe is
  legitimate. The one systematic deviation concentrates in **s0** (approximation band,
  −0.015/−0.010 on Pile) → hypothesis for Study 4: *domain identity lives at coarse scales;
  fine-scale statistics are domain-universal.*
- **Finding 3 — the depth × scale gain surface is structured, not a fade** (census.py, all
  10 layers, 25.2M values/cell): the U-shape exists full-strength **only at L00** (entry
  band-shaping); L01–L04 drop sharply into a flat quiet trunk; then **s0 climbs
  monotonically L03→L09** (0.330→0.431), s7 ticking up late — the U gently re-forms toward
  the head. Hypothesis: the network funnels toward coarse/contextual content approaching
  prediction. *(Scored: the earlier "uniform quieting with depth" read from the 2-layer
  preview was wrong in shape — full-depth data corrected it.)*
- **Scale ordering confirmed in code** (model.py ~2427, `[approx] + details[::-1]`):
  s0 = approximation; s1..s7 = detail bands **coarse → fine** (s1 ≈ 64–128 tokens,
  s2 ≈ 32–64, … s7 ≈ 1–2 at levels=7). Manifest note updated from "unconfirmed".
- **Finding 4 — the kurtosis (sparsity) map is structured and depth-decaying.** Excess
  kurtosis per layer×scale: L00 nearly Gaussian at the band ends (s0=1.1, s7=1.3) with one
  screaming anomaly at **s2 (32–64-token band): 21.6**; L01–L03 turn heavy-tailed at BOTH
  ends (s0 up to 17.7, s7 up to 22.6 — rare large events at the most-local and most-global
  scales); then near-monotone decline everywhere to L09 (2–6) — representations densify
  toward the head. s1 (coarsest detail) is consistently the lightest-tailed band.
  *Interpretation caution on record:* pooled kurtosis conflates per-channel selectivity with
  cross-channel variance heterogeneity; discriminate per cell (see Finding 5's method).
- **Finding 5 — first candidate monosemantic channel (Study 2 pilot, L00/s2).**
  `topk_contexts.py`: **ch132 owns 100% of the top-1000 extreme tail** of the L00/s2 band
  (512 channels available), sign-consistent (all ≈ −9), firing at **completions of long
  information-dense spans** (sentence-final periods, clause boundaries capping long
  enumerations) across unrelated topics (weather/demographics/anime/mycology/NASCAR) — a
  *structural* long-span boundary detector at the entry layer, in exactly the band whose
  wavelength (32–64 tokens) matches the feature. For this cell the kurtosis WAS selectivity
  (a variance mixture cannot concentrate a tail into one channel). **Owed before the label
  hardens:** random-direction control (interpretability illusion) + Study-3 causal ablation.
  Next probe: tail concentration at late layers (diffuse tails would support the
  early-selective / late-distributed hypothesis).
- **Finding 6 — the wavelet autopsy (Study 5 opens): what the lifting learned**
  (`wavelet_autopsy.py`, impulse-response probing, Mini/D2 vs seed-matched Haar-init;
  per-channel taps saved `.interp/autopsy_.../taps.npz`):
  1. **The lifting absorbed the removed projections' job.** Cross-channel response energy:
     ~0.00 at init → **0.26 / 0.60 / 0.74–0.80** (levels 0→6) trained. In the fully spectral
     architecture (no proj_out, no MLP), the P/U MLPs became the network's channel-mixers —
     channel rotation relocated *into* the wavelet, multi-scale and PR-constrained, rather
     than disappearing. (Explains how skip_proj_out could be a win: the mixing moved, it
     didn't die.)
  2. **One shared mother kernel on the diagonal** (top-1 SVD participation 0.96–1.00 at every
     level): channel-wise, the model learned essentially a single new wavelet shape, not a
     bank of 512.
  3. **The shape: Haar-pair → "leaky causal differencer."** Mid-level taps became a single
     negative present-tap against a smooth broad positive causal-average ramp (e.g. L2:
     −0.098 at n=0, then +0.05→+0.015 decaying over n=2–9) — present-minus-local-average,
     the structure of biorthogonal spline / average-interpolating wavelets (causal variant),
     NOT Daubechies-like. Level 0 nearly halved its subtractive tap (−0.36 → −0.16 vs
     +0.63 pass-through of the previous token): the finest "detail" is closer to a delayed
     copy than a difference — consistent with s7's loud gain in the census.
  4. **The crawl left the dyadic ladder**: L0 sharpened onto d1 (0.89); L1–L3 flattened into
     broad 1–8-token windows (L2 near-uniform over d1–d8); **L4 reaches for d1–d3 despite
     base dilation 16** (fine reach at a coarse level); L5–L6 diffuse around base.
  5. **Moderate nonlinearity (~9–15% at operating amplitude) — and training made the fine
     levels MORE linear than their init** (L0: 0.35 → 0.15).
  6. **Instrument caveat on record:** small-ε probing sits in GELU's 0.5-slope regime, so
     absolute m0/vanishing-moment values are biased (init "Haar" itself shows m0≈0.33 at ε);
     init-vs-trained *differences* are valid; a v2 pass extracting taps at operating
     amplitude (eps_big) is the cheap fix before any moment claims harden.
  *Family verdict: not Daubechies; rhymes with causal average-interpolating/spline
  biorthogonal in shape — but the whole object (vector-valued, mildly nonlinear, learned
  non-dyadic reach, PR guaranteed by lifting) has no classical name. It is a new,
  characterizable wavelet system.*
- Instruments to date: `tools/interpretability/coeff_dump.py` (Phase 0),
  `tools/interpretability/census.py` (Study 1: absmean/std/kurtosis over shards),
  `tools/interpretability/topk_contexts.py` (Study 2: top-k contexts + tail concentration),
  `tools/interpretability/wavelet_autopsy.py` (Study 5: effective filters, invariants,
  crawl distributions, trained-vs-init).
- Ops note: bypass cross-window state is batch-shaped and carries across forward calls →
  constant batch size enforced in the tool; zero-state reset at dump start (deterministic).

## What this unblocks

- **Paper 2's spine**, with its central claim tested before a dollar of GPU is spent on it.
- An evidence-based **second-GPU budget** (SAE-scale only if Study 2 demands it).
- The interpretability suite promised in the Release Pipeline ("developed & processed fully
  here" on Small) inherits working instruments instead of starting cold.
