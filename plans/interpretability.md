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
- **Study 2b — coaxing the basis** *(the promoted follow-up)*: add an explicit **L1 penalty on coefficients** (L1 in a fixed basis is the axis-alignment pressure ICA says is missing) and sweep its weight; hypothesis on record — the ICA/native ratio falls monotonically while BPB rises, giving a **monosemanticity-vs-capability frontier**. A basis induced on demand at a known price beats one merely hoped for

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

### Study 7 — Concept control: REAP + SOW *(the control half of the program)*

Studies 1–6 ask *what can we read*. This one asks *what can we change* — and it is the half with
the most direct alignment relevance. Both methods were built and measured in the research fork
(`EXARCH-research/interpretability/{reap,sow}.py`); this study ports them and runs the
demonstration WaveletLM's invertibility makes uniquely clean.

**The two methods** (both act on the *training data*, not the trained weights — so what the model
never sees, it never learns; no post-hoc suppression to be jailbroken around):
- **REAP — REplacing Ablated Passages.** Merges FDA-labelled concept spans into passages, sends
  each to an LLM for rewriting that removes the concept while preserving factual content, names,
  dates, and approximate length; the replacement JSON is loaded at training time.
- **SOW — Substitution Of Words.** Token-level. For each token carrying the target concept, finds
  the k nearest neighbours in binary conceptual-embedding space *with the concept dimensions
  masked out*, so the replacement is maximally similar on everything except the concept.
  **Invariant tiers** constrain replacements to share grammatical/ontological properties (POS,
  entity type), which is what keeps the substitution from wrecking syntax.
- **Layerable:** SOW is near-zero-cost insurance over a REAP'd corpus, catching concept tokens
  that survive a passage rewrite.

**Prerequisites differ, and that sets the order.** REAP needs only FDA labels (~$200–250/concept
of LLM labelling) → demonstrable now. SOW's neighbour search and invariant tiers are defined on
the native 256-dim BCE features → **gated on reintroducing the semantic embedding**. Run REAP
first; SOW follows the embedding's return, or via a concept-list variant that drops the
embedding dependency (Ramon's generalization: replace words failing *any* stated constraint,
not only BCE-dimension constraints — untested, and the invariant tiers would need a
non-BCE source).

**The question worth publishing is the cost, not the removal.** That a concept disappears from
generation is the easy half and is already observed. The under-reported half — and the one
reviewers will actually press on — is **what else broke**: a measured *capability-cost curve* of
concept removal (ΔBPB, and targeted evals on neighbouring-but-innocent concepts, vs. removal
strength / concept breadth). WaveletLM is an unusually honest place to measure it, because
per-scale ablation is exact: the same invertibility that powers Study 3 lets us ask *which
scales* lose information when a concept is stripped from the data.

**Open risk, stated plainly:** concept removal may damage information integrity in ways
proportional to how load-bearing the concept is (removing "violence" from a corpus also removes
history, medicine, and law). That is an empirical question, not a reason to avoid the method —
but the result must be reported whichever way it lands, including if the cost is prohibitive.

**Method-simplicity note:** both methods are conceptually simple, which is a deployment advantage
rather than a weakness — the activation-steering literature is likewise simple in concept and
consequential in practice. The contribution here is the *measurement*, not the mechanism.

### Study 8 — Concept directions via per-scale FDA *(activation-space control)*

Where Study 7 edits the *data*, this edits the *representation* — and unlike SOW it is **not gated
on the semantic embedding**: Fisher Discriminant Analysis needs activations plus labels, nothing
else. Same prerequisite as REAP (the FDA labelling pipeline, ~$200–250/concept), so it is
unblocked today.

**Why revisit it — the old result was mediocre for a diagnosable reason.** In the previous
architecture, FDA input-only violence suppression measured **zero BPB cost but weak suppression**;
the FDA-*output* variant (Jacobian trace) cost ~2.3× step time for no quality gain and was
shelved. "Zero cost + weak effect" is the signature of a direction that wasn't carrying much of
the concept — one Fisher direction searched in the *whole residual stream* gets diluted across
everything else living there.

**What the fully-spectral architecture changes.** Coefficients are factorized as scale × channel
× position, so FDA can be fit **per scale band** (and per layer) instead of once on the whole
stream:
- **Separability map** — Fisher ratio per (layer × scale) for a labelled concept. Asks a question
  the old architecture could not pose: *at which scales is this concept linearly separable?*
  Coarse-only separability ⇒ the concept is discourse-level; fine-only ⇒ lexical/orthographic.
  This is a finding about the concept's structure, independent of whether we then suppress it.
- **Exact ablation** — project the direction out in coefficient space, resynthesize *exactly*,
  measure ΔBPB and targeted-eval damage. The invertibility advantage of Study 3, applied to a
  concept direction; no reconstruction-error caveat, which is the standing weakness of
  SAE-based concept-removal claims.
- **Minimal-removal anchor** — a Fisher direction is the least-damaging *linear* removal, so it
  anchors the cheap end of Study 7's capability-cost curve, with REAP/SOW (data-side, blunter but
  more thorough) at the other end. Same curve, two mechanisms, directly comparable because the
  cost metric is identical.
- **Cross-check against Study 2** — does the concept direction align with any channel already
  flagged monosemantic? Alignment is convergent validation from two independent instruments;
  non-alignment says concepts live in channel *mixtures* even in a privileged basis, which is a
  sharper (and more publishable) statement about where superposition survives.

**Predictions on record** *(estimates)*: (a) per-scale Fisher ratios will exceed the
whole-stream ratio for at least some bands — that is the whole bet, and if it fails, FDA is
genuinely a weak tool here and should be dropped rather than tuned; (b) mid-to-coarse scales
carry more concept separability than the finest bands.

**With vs. without the CE/SE.** Without: everything above works — directions are discovered,
unnamed. With: the discovered direction can be compared against *named* BCE dimensions, turning
"a direction that separates violence" into "a direction that aligns with `is_weapon` +
`relates_to_war`" — interpretive value, not a prerequisite.

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
- **Finding 8 — Finding 5's control FAILED: ch132 is the LOUDEST channel, not a selective one**
  (`channel_privilege.py` on Mini/D3 `.interp/mini_d3`, 2026-07-25). ch132's dominance of the
  L00/s2 top-1000 tail (99.8% here, 100% on D2) is a **variance artifact**: its std is **1.716 vs
  a 0.367 cell median — 4.7x, and rank 0/512 (the single loudest channel)** — while its **excess
  kurtosis is -0.65, rank 511/512** (platykurtic, i.e. *less* heavy-tailed than a Gaussian). A
  channel 4.7x louder than its neighbours dominates any global |value| ranking whether or not it
  is selective, so Finding 5's stated defence — *"a variance mixture cannot concentrate a tail
  into one channel"* — **is incorrect**; a single dominant-variance channel does exactly that.
  This is the caution Finding 4 already recorded, now measured. **Finding 5's "first candidate
  monosemantic channel" label does not survive its control and is withdrawn**; the coherent
  top-contexts observation stands as an observation, but tail-ownership is not evidence for it.
  *Persistence (Study 2 pilot, D2 -> D3, 20ep -> 40ep):* the phenomenon is perfectly stable —
  ownership 1.000 -> 0.998, sign consistent, kurtosis rank 511/512 in both — i.e. a **stable
  loud channel**, not a feature that sharpened with training.
- **Finding 9 — the naive random-direction control is CONFOUNDED by mixing depth (CLT); the
  privileged-basis question remains OPEN.** A first pass scored native channels against 512
  dense random unit directions and found 93.5% of native channels above the null's 95th
  percentile (chance 5%) across all 80 layer x scale cells — apparently overwhelming support for
  the thesis. **It is not.** A dense random direction averages 512 channels, and averaging drives
  kurtosis toward 0 by the central limit theorem *regardless of any privileged structure*.
  Measured decay on L00/s2 (median excess kurtosis by mixing depth): **1 channel 0.772 -> 2:
  0.476 -> 4: 0.372 -> 8: 0.229 -> 64: 0.017 -> 512: -0.053.** The null sits at the far end of
  that curve, so "coordinate beats 512-way mixture" is largely a statement about mixing depth,
  not basis alignment. **Do not cite the 0.935 figure as evidence for the thesis.**
  *What a valid test looks like:* search for the rotation that MAXIMISES kurtosis (ICA / FastICA,
  or a direct max-kurtosis direction search) and compare it against the best native channel at
  equal mixing depth. If no rotation beats the native axes materially, the basis is privileged;
  if ICA finds far more kurtotic directions, features are rotated and SAEs are indicated. That
  is the decisive Study 2 experiment and it is now the next one to run.
  *Side observation worth keeping:* native per-channel excess kurtosis runs only **~0.2-1.8**
  across cells. A crisply sparse monosemantic feature would sit in the tens. On this metric the
  channels do not look strongly sparse — a caution for the thesis, independent of the control.
- **Finding 10 — THE CHANNEL BASIS IS NOT PRIVILEGED (Study 2 answered).** `ica_test.py` on
  Mini/D3, 5 cells, two search budgets. FastICA searches rotations for maximum non-Gaussianity;
  both it and a native channel are 1-D projections of the same data and excess kurtosis is
  scale-invariant, so the comparison is direct. **ICA finds far more non-Gaussian directions than
  any native channel, in every cell**, and the advantage GROWS with search budget — the signature
  of features living in rotated directions:

  | cell | native p95 | ICA p95 (256) | ratio @64 | ratio @256 |
  |---|---|---|---|---|
  | L00/s0 | 1.06 | 78.31 | 11.46 | **17.34** |
  | L00/s2 | 2.95 | 50.16 | 2.37 | **4.04** |
  | L01/s7 | 23.96 | 82.52 | 1.02 | **2.57** |
  | L09/s0 | 8.54 | 57.94 | 1.04 | **2.26** |
  | L03/s3 | 16.36 | 37.60 | 0.52 | **1.57** |

  *Prediction scored:* the @64 run produced three "privileged-ish" verdicts including an
  impossible ratio of 0.52 (ICA cannot do worse than the native axes, which are themselves
  candidate directions). That was diagnosed on the spot as **under-search** at 64 components of
  512 dims; raising to 256 flipped **all three**, 0.52 -> 1.57 included. Mean ratio 3.28 -> 5.56.
  **Read ratios <=1 as instrument under-search, never as native superiority.**
- **Consequences (the decision rule fires).** (a) **SAEs return to the critical path** for the
  channel axis; Study 6 stops being a formality and becomes the second-GPU sizing exercise, now
  evidence-backed. (b) **The SCALE axis is untouched** — it is privileged *by construction*
  (hard-wired decomposition, not learned), so per-scale attribution, exact ablation and Studies
  3/4/8 are unaffected; only the stronger channel-level claim falls. (c) The thesis statement in
  this document must be **rewritten from "channels are privileged-ish" to "channels are NOT
  privileged; the scale factorization is"** before any of it reaches the README.
- **Finding 11 — the coaxing hypothesis (next experiment).** `CoefficientShrinkage` PERMITS
  sparsity but does not INCENTIVISE it: `lam_raw` inits at -10 (softplus ~ 4.5e-5, an
  effectively-zero threshold) and is trained only against BPB, so the model raises lambda only
  where thresholding aids prediction. The six 2026-07 shrinkage runs are therefore *not* tests of
  coaxed monosemanticity (and carry no checkpoints, and use C=100/C=1024 rather than the C=512
  interp baseline — three reasons they cannot answer this). **The mechanism that would coax it is
  an explicit L1 penalty on the coefficients (weight beta): L1 in a FIXED basis is exactly the
  pressure that produces axis-aligned sparsity — the property ICA says is absent.**
  *Hypothesis on record:* the ICA/native ratio falls monotonically in beta while BPB rises,
  yielding a **monosemanticity-vs-capability frontier**. `ica_test.py` is the gauge. A coaxed
  privileged basis would be a stronger result than a found one.
- **Finding 12 — the shrinkage lambda moved, but almost certainly NOT by learning (prediction
  MISSED, and the miss is informative).** Read from the C=1024/1ep `shrinkage=pre` checkpoint
  (`logs/wikitext-103_2026-07-06_03-54-06`, 81,920 lambda cells over 10 layers). **Prediction on
  record was "lambda stays near its 4.54e-5 init". It did not: lambda -> ~6.83e-2, a ~1,500x
  increase, in 100.00% of cells.** So the mechanism is not inert in the trivial sense.
  **But the distribution is the tell:** median 6.816e-2 to 6.842e-2 across every layer, max
  7.009e-2 model-wide — a **~0.4% spread across 81,920 independently-parameterized values**.
  Learned per-(scale,channel) sparsity would be *heterogeneous* (different channels need
  different thresholds); a single near-constant value is what a **uniform force** produces.
  *Hypothesis (strong, not yet proven): this is weight-decay drift, not learning.* `lam_raw`
  inits at **-10**, far from zero, and carries `weight_decay=2e-6`; decay pulls it toward 0,
  which RAISES softplus(lam_raw). The companion parameters move exactly as that predicts:
  **gamma 1.0 -> 0.2733 median (toward 0), monotonically with depth (L0 0.559 -> L9 0.022)**.
  (`theta` stayed at exactly 0.0000/sd 0.0000, but that is *not* independent evidence — 0 is a
  stationary point of cos, so its gradient vanishes there by construction.)
  **Decisive test (cheap):** re-run one shrinkage arm with **weight decay excluded from the
  shrinkage parameters**. If lambda stays near init, the entire 2026-07 shrinkage screen was
  measuring optimizer drift rather than a learned sparsity prior — which would also explain why
  it cost BPB (+0.0019 pre / +0.0033 post vs the matched control 1.1101) instead of helping.
  **Design lesson for Study 2b (the L1-penalty sweep):** (a) put **no weight decay** on any
  sparsity parameter, (b) do **not** initialize it far from zero, and (c) make the sparsity
  pressure an **explicit loss term** whose weight we set, so the dial is ours rather than an
  emergent tug-of-war with the optimizer. The gauge stays `ica_test.py`'s ICA/native ratio.
  *Also note:* gamma is **gauge-degenerate with the downstream mixer weights** (the mixer can
  rescale freely), so gamma's absolute magnitude is not independently interpretable — only
  lambda, which acts on |z| before that scaling, carries threshold meaning.
- **Finding 13 — Study 3 (exact scale ablation): LAYER 0 CARRIES THE BAND STRUCTURE; the rest of
  the stack is highly redundant.** `scale_ablation.py` on Mini/D3, all 80 (layer x scale) cells.
  The ablation is exact and hook-free: `scaled = mixed * scale_weights[idx]`, so zeroing
  `scale_weights[s]` removes scale s's whole contribution, and reconstruction is exact. Eval set
  is FIXED across ablations, so differences are deterministic (the uncertainty is sampling, not
  measurement). Base loss 3.1641 nats on 8,192 val tokens.
  **The top SIX most damaging ablations are all in layer 0**, led by **L00/s0 +0.0182 dBPB** and
  **L00/s7 +0.0172** — 4-9x anything at depth. Every other (layer, scale) costs ~0.0004-0.0042
  dBPB; the median cell is ~0.002. **Removing an entire frequency band from any single layer
  1-9 is nearly free**, which is a strong statement about depth redundancy.
  *Convergence:* this independently reproduces **Finding 3** ("the U-shape exists full-strength
  only at L00 — entry band-shaping"), now with causal rather than descriptive evidence. Two
  unrelated instruments, same conclusion.
  *Damage is NOT just the scale weight:* L00/s0 has the SMALLEST weight in its layer (w=+0.457)
  and the LARGEST damage, while L00/s6 has w=+1.860 and 3x less damage — so this measures
  function, not magnitude.
  *Prediction partially scored:* "coarsest-scale ablation degrades most" CONFIRMED (s0 leads,
  summed +0.0460); "finest hits local grammar" consistent (s7 second at L00); but **"mid scales
  are where the interesting semantics live" is NOT supported** — s4/s5/s6 are the LEAST damaging
  bands (+0.0185/+0.0203/+0.0176 summed). Caveat: dBPB measures aggregate prediction damage, so
  a redundantly-encoded semantic scale could still read as cheap to ablate.
- **Finding 14 — the zero-inference gain probe is VALIDATED (r = 0.997), and buys the trends
  free.** `gain_probe.py` predicts per-scale mean|coeff| from `decomp_norms[s].weight` alone via
  `0.7979 * mean|gamma|` (E|z| for standardized Gaussian z). Against the measured D2 layer-0
  census: **mean |err| 0.0165, max 0.0359, shape correlation r = 0.997.** The census is
  therefore readable **from any checkpoint in seconds, with no forward passes, no dataset, and
  no dump** — including checkpoints never dumped. *Systematic bias, explained:* predictions run
  high in 8/8 cells because 0.7979 assumes Gaussianity while Finding 4 measured heavy tails
  (heavy tails lower E|z|/sigma); a calibrated constant ~0.77 gives near-exact agreement.
  **Trends obtained for free:**
  (a) *Training duration* (D2 20ep -> D3 40ep, same width): gains fall ~8% **uniformly**, shape
  fully preserved — the stack quiets with training without re-allocating across scales.
  (b) *Width* (D3 C=512 -> SP1 C=1024): the wider model specializes far **harder at depth** —
  at L09 the s0:s1 contrast is **3.5x in SP1 (0.445 vs 0.128) versus 2.1x in D3 (0.434 vs
  0.205)**, with mid-scales s4/s5 collapsing to ~0.12 in SP1 against ~0.21-0.24 in D3. More
  channel headroom is spent sharpening the band structure, not spreading it.
- **Finding 15 — Study 3b: depth is REDUNDANT BUT NECESSARY (8.3x super-additive); layer 0 is
  load-bearing.** `group_ablation.py` on Mini/D3, 8,192 val tokens, base 3.1641 nats. Prompted by
  a sharp question from Ramon: if single-band ablations are ~free outside layer 0 (Finding 13),
  would one wide layer do? **No — and the group test says so decisively.**

  | ablation | dBPB |
  |---|---|
  | layer 0 spectral path OFF (alone) | **+1.4289** |
  | each of layers 1-9 OFF (alone) | +0.0176 to +0.0251 |
  | **SUM of layers 1-9 individually** | **+0.1824** |
  | **WHOLE of layers 1-9 together** | **+1.5063** |
  | **super-additivity ratio** | **8.3x** |

  Cheap to perturb, catastrophic to delete — the textbook signature of **distributed, redundant
  computation that is collectively essential**. Single-point ablation cannot distinguish
  "redundant" from "idle"; group ablation can, and it lands firmly on redundant.
  *Marginal cost RISES with each layer removed* (deltas 0.064, 0.110, 0.134, 0.166, 0.165, 0.168,
  0.247, 0.427 dBPB) — each layer refines on top of the last rather than duplicating it.
  *Layer 0 is separately critical*: killing its spectral path alone costs +1.4289 dBPB, ~57x any
  single deep layer, confirming Findings 3 and 13's "entry band-shaping" reading causally.
  **Architecture consequence:** this evidence does NOT support trading depth for width. It also
  aligns with the existing depth sweep (depth pays through ~L=10, 30L worse than 20L at C=512).
  *Cross-layer scale ablation (same run):* killing a whole band everywhere costs s0 +0.1045,
  s3 +0.0711, s7 +0.0644, s1 +0.0565, s2 +0.0473, s5 +0.0338, s6 +0.0329, s4 +0.0302 — the same
  ordering as Finding 13's per-layer sums (convergent), and notably **the model is far more
  robust to losing an entire frequency band (0.03-0.10) than to losing depth (1.51)**. s0 is
  super-additive at only 2.3x (0.0460 summed -> 0.1045 whole), versus depth's 8.3x.
  *Instrument bug caught and fixed:* the script's first auto-verdict printed "near-additive"
  because it compared the group against the worst SINGLE ablation — which was layer 0, not a
  member of the group. Corrected to sum-of-parts vs whole. Recorded because the wrong verdict
  was the exact opposite of the truth.
- **Finding 16 — the decomposition IS shift-equivariant; the position-artifact worry is
  RULED OUT (and the test doubles as a receptive-field probe).** `shift_equivariance.py` on
  Mini/D3. Classical dyadic wavelets are shift-VARIANT, which would have meant coefficients are
  partly an artifact of grid alignment — a caveat on the census, on Study 2's channel statistics,
  and specifically on "ch132 fires at clause boundaries" (boundaries correlate with position,
  position correlates with alignment). Discriminator: alignment sensitivity is parity-dependent
  (odd shifts break dyadic alignment, powers of two preserve it), while context loss is
  magnitude-dependent. **Measured: odd/pow2 contrast = 0.87x (i.e. ~1.0, no parity effect).**
  Deltas instead grow monotonically with shift SIZE (1 -> 16) and with DEPTH (L0 ~0.001,
  L4 ~0.015, L9 ~0.028) — exactly the context-loss signature. **Coefficients are a stable,
  position-independent code; every position-correlated finding to date survives.**
  *Bonus — per-scale effective receptive field, free:* at layer 0, scales **s2-s7 are EXACTLY
  0.000** under every shift tested (strictly local, no left-context dependence beyond the
  margin), while s0/s1 move (0.001-0.026) — the coarse bands integrate context, the fine bands
  do not. This is a direct measurement of receptive field per scale, and it independently
  supports the scale-semantics reading that Study 4 is designed to test.
  *Control applied:* `decompose_bypass_cross_window` forced off and persistent state cleared, so
  no state leaks between the reference and shifted runs.
- **Finding 17 — the surprise spectrum: prediction error is carried at COARSE-TO-MID scales,
  and the finest band carries none of it.** `surprise_spectrum.py` on Mini/D3 layer 0, 2,048
  positions. Motivated by Ramon's flow conjecture: lifting is literally predict-then-update, so
  detail coefficients ARE prediction errors per timescale, and their movement along the time axis
  should trace where the model is being corrected. Well-posed only because of Finding 16
  (shift-equivariance).
  **(a) Falsifiable check, PASSED for most bands.** Correlation between per-scale coefficient
  flow |Δ| and the model's own next-token NLL: **s2 +0.389, s1 +0.387, s0 +0.363, s4 +0.348,
  s3 +0.276, s5 +0.260, s6 +0.139, and s7 −0.022** (total +0.339). So "surprise" is an earned
  name for coarse-to-mid bands — but **the finest band is uncorrelated with predictive
  difficulty**, i.e. s7 tracks something else entirely (local orthography / sub-word identity),
  which moves whether or not the model is struggling.
  **(b) Per-band timescale, measured.** flow/surprise ratios: s1 1.089 (lowest) ... s7 1.364
  (highest). Under a Gaussian assumption the ratio is √(2(1−ρ)) for lag-1 autocorrelation ρ, so
  **ρ ≈ 0.41 for s1 (coarsest detail, persistent) down to ρ ≈ 0.07 for s7 (finest, essentially
  memoryless)** — a direct quantitative confirmation of multi-resolution semantics: coarse bands
  integrate across tokens, fine bands do not. Independent of, and consistent with, Finding 16's
  receptive-field reading (L0 s2–s7 exactly shift-insensitive).
  **(c) Where the code moves most: BOUNDARIES.** All 12 peak-flow positions peak in **s0
  (approximation)**, and the contexts are overwhelmingly paragraph/section breaks and
  sentence-final punctuation — `'

 Meridian' -> ' was'`, `' unrestored . 


 Temple'`,
  `' existence " . 


 Churchill'`. Topic/section transitions move the coarse topical code
  hardest. **This corroborates the structural half of the withdrawn Finding 5** (ch132 fired at
  sentence-final periods and clause boundaries) from a completely independent instrument:
  boundaries really are salient in the coarse bands, even though tail-ownership was the wrong
  evidence for it.
  **(d) Flow is NOT the same as NLL.** Several peak-flow positions have near-zero NLL (e.g.
  `' father , Yax Nuun Ay' -> 'i'`, nll 0.00) — sub-word continuations where the code moves a
  lot but the prediction is trivial. So flow measures representational change, and correlates
  with surprise without being identical to it.
- **Finding 18 — Study 4 v1 is SATURATED and its labels were the wrong ones (my design error);
  but the depth x scale GRADIENT inside it is real and informative.** `scale_probes.py` on
  Mini/D3, labels {heading, boundary, numeric, capitalized, subword}, layers {0,4,9}, with the
  mandatory shuffled-label selectivity control (Hewitt & Liang).
  **The error:** every label chosen is a *deterministic function of the current token*
  ("is this token capitalized/numeric/punctuation"). Any representation that encodes token
  identity — which all of them must — decodes such labels trivially. Result: **accuracy is
  0.996-1.000 in EVERY (layer, scale) cell**, controls sit at the majority-class rate, and
  "selectivity" merely reproduces the class balance (heading 1.37% positive -> selectivity 0.016;
  capitalized 15.67% -> 0.154). Cross-label comparison is therefore meaningless, and the absolute
  numbers say nothing about scale specialization. *The control did its job: saturation is exactly
  what it is designed to expose.*
  **The fix for v2:** labels must require CONTEXT, not just the current token — e.g. POS of a
  syntactically ambiguous word, "inside a quotation/heading SPAN" (not the marker token itself),
  distance since the last sentence boundary, article/topic identity, or NEXT-token properties.
  Also switch the metric to balanced accuracy or AUC, which is invariant to class imbalance.
  **What IS informative — the within-label depth x scale gradient.** At layer 0 all scales
  decode token identity about equally, but by layer 9 it has concentrated in the FINEST band:
  capitalized **s7 0.136 vs s1 0.078 (1.7x)**, subword **s7 0.095 vs s1 0.046 (2.1x)**. So the
  coarse bands progressively DISCARD current-token identity with depth while the fine bands
  retain it. This converges with Finding 17 (s7 uncorrelated with predictive difficulty, lag-1
  autocorrelation ~0.07 = memoryless/local) and Finding 16 (L0 s2-s7 exactly shift-insensitive =
  strictly local). Three instruments, one consistent picture: **fine bands carry local token
  identity, coarse bands carry integrated context.**
- **Finding 19 — coefficient FLOW carries difficulty signal that output ENTROPY misses, and it
  is available after ONE layer.** `flow_vs_entropy.py` on Mini/D3, 8,160 positions, layers
  {0,4,9}. Gates the memory-mechanism design: a retrieval gate needs a LABEL-FREE difficulty
  signal at inference, and entropy is the incumbent (label-free, standard, and measured here at
  **r(entropy, NLL) = +0.613**, well above flow's +0.40). The question is therefore not whether
  flow predicts difficulty but whether it adds anything entropy does not.
  **It does.** Partial correlation r(flow, NLL | entropy) at layer 0: **s0 +0.293, s1 +0.293,
  s2 +0.250**, decaying to s7 +0.090.
  **Three structural facts that shape the design:**
  (a) *Signal concentrates in EARLY layers* — partials L00 ~0.29, L04 ~0.16, L09 ~0.10. Entropy
  is computed from final logits, so late-layer activity is largely redundant with it while
  early-layer activity is independent.
  (b) *Coarse bands win* — same s0/s1 > ... > s7 ordering as Findings 17 and 18.
  (c) **Early-exit gating is possible.** Entropy needs the FULL forward pass (final logits);
  layer-0 flow needs **one layer of ten**. A coarse-band L0 gate can trigger retrieval before
  the forward pass completes — a capability entropy structurally cannot provide.
  **Negative control PASSED:** s7 measured r(flow, NLL) = +0.025 at L0, matching Finding 17's
  −0.02, and carries the lowest partial. The built-in falsifier behaved as predicted.
  *Honest calibration:* partial r 0.29 is ~8.5% of residual variance — real but modest. The gate
  should COMBINE flow with entropy, not replace it. One checkpoint, 73M, layer-0 measurement.
- **Finding 20 — Study 4 v2 SUCCEEDS: the representation linearly encodes the SEMANTIC
  CATEGORY of the UPCOMING token, and decodability rises monotonically with category
  coherence.** `relatedness_ladder.py` on Mini/D3, 12 sets/rung, AUC metric.
  **Two design fixes were required, both from failures:**
  (a) *v1's saturation (Finding 18)* — labels must be **PREDICTIVE**, not descriptive:
  `y[t] = token[t+1] in set X`. Probing position t for a property of t+1 cannot be solved
  by reading off the current token, which is what made every v1 label trivial.
  (b) *the first v2 run's null was not null* — naive random sets scored AUC **0.87-0.96**
  at k=0, because a random set routinely contains a COMMON token and "a common token is
  coming" is a highly learnable linguistic task. Positive rates swung 0.0011-0.0034 across
  rungs, so set composition swamped coherence. Fixed by **frequency-matching** each filler
  token to a related token's frequency neighbourhood, making coherence the only difference.
  **The ladder (Ramon's dose-response design), after the fix — monotonic in ALL EIGHT cells:**

  | cell | k=0 (null) | k=8 | AUC-vs-k |
  |---|---|---|---|
  | L00/s0 | 0.677 | 0.892 | **+0.992** |
  | L00/s7 | 0.673 | 0.850 | **+0.986** |
  | L09/s0 | 0.801 | 0.951 | +0.971 |
  | L09/s1 | 0.773 | 0.930 | +0.948 |
  | L09/s7 | 0.742 | 0.958 | +0.930 |
  | L00/s1 | 0.640 | 0.876 | +0.926 |
  | L00/s4 | 0.729 | 0.889 | +0.901 |
  | L09/s4 | 0.840 | 0.947 | +0.895 |

  Eight independent monotone ladders (r = +0.90 to +0.99) is very hard to produce by
  artifact — which is exactly why the escalating design beats a binary control.
  *Depth trend:* **L09 > L00 in every rung** (null 0.80 vs 0.68) — category information
  concentrates toward the prediction head, consistent with Findings 17/18 showing coarse/late
  representations carry integrated rather than local content.
  **Honest scope:** the null still sits at 0.64-0.84, not 0.5, so frequency matching improved
  it greatly but did not fully neutralise it — **the SLOPE is the result, not the level**. And
  this cannot separate "the model learned semantics" from "language makes coherent categories
  predictable"; the model must encode it either way to predict well. Do not overclaim.
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
- **Finding 7 — no in-context copying (associative recall / induction): the capability gap**
  (`induction_probe.py`, 2026-07-21). *Why it matters:* associative recall — having seen
  "KEY VALUE", predict VALUE when KEY reappears — is the capability that gates attention-free
  architectures as modern-LLM components. It is why the field converged on hybrids
  (Jamba 1 attention : 7 Mamba; Nemotron-H ~8% attention layers; IBM Granite 4.0 at 9:1), and
  the [Mamba paper](https://arxiv.org/pdf/2312.00752) reports that linear time-invariant SSMs
  *"cannot [solve selective copying] even when combined with more powerful architectures."*
  WaveletLM's mixing is structurally LTI-like: fixed dilation ladder, position-uniform
  predict/update nets, content-dependence entering only via cross-scale gating and the bypass.
  *Protocol:* `[filler] KEY VALUE [filler] KEY -> ?` vs a token-identical control in which the
  demonstrated pair is replaced; the difference in log P(VALUE) at the final position isolates
  the in-context copy. Filler is real WT-103 text, shared across all models; KEY/VALUE are
  random mid-vocabulary ids; 256 trials each.

  | model | params | induction lift | median rank of VALUE | argmax | top-10 |
  |---|---|---|---|---|---|
  | **GPT-2 small (control)** | 124M | **+10.414 nats** | **5** | **40.2%** | **55.5%** |
  | Mini D3 (WT-103, 40ep) | 73M | +0.145 | 25,222 | 0.0% | 0.0% |
  | Mini D1 (WT-103, 10ep) | 73M | +0.114 | 22,675 | 0.0% | 0.0% |
  | Mini D3, gap=8 | 73M | +0.512 | 21,842 | 0.0% | 0.4% |
  | F1 (Pile, 1ep) | 73M | +0.494 | 19,220 | 0.0% | 0.0% |
  | **SP1 Small (WT-103, 5ep)** | **239M** | **+0.134** | **22,752** | **0.0%** | **0.0%** |

  **Width does not rescue it.** SP1 at 239M — 3.3× Mini and *nearly 2× the control's 124M* —
  is indistinguishable from the 73M models. Scaling C from 512 to 1024 moved the induction
  lift by −0.011 nats. (M1 at 893M was started and stopped: 3.57 GB of weights on a 6 GB card
  ran at 96% VRAM and 86 °C with the answer already settled; re-run pod-side if ever wanted,
  though the 73M→239M flatline makes a qualitative change at 893M implausible.)

  **The GPT-2 control validates the instrument** (~70× the lift, rank 5 vs rank ~25,000, on a
  model of comparable size sharing our exact tokenizer) — so the null is a property of the
  models, not the probe. Two directional whispers, both tiny: Pile-trained > WikiText-trained
  (diverse data nudges toward copying) and short gap > long gap (mild locality).
  **Finding 7b — the scope of 7, measured (`context_usage.py`): this is a DISSOCIATION, not an
  absence of in-context memory.** Position-wise loss on natural text (128 sequences × 256
  tokens, no synthetic insertions): D3 improves from **5.188 nats at the contextless first
  token to 3.079 at the last** — a **2.11-nat gain from context**, and *better than GPT-2's
  3.390 at full context* (D3 is in-domain; GPT-2 is zero-shot). GPT-2's gain is larger
  (−3.94 nats; ICL score 1.917 vs D3's 0.544), so it extracts ~3.5× more *marginal* value per
  unit of added context. Conclusion: WaveletLM has strong **contextual/statistical** memory
  (history sharpens the distribution — what multi-scale mixing does well, and what BPB
  measures) and lacks **retrieval** memory (exact lookup/copy of an arbitrary bound symbol).
  Degree vs kind: the ICL gap is a difference of degree; the induction result is a difference
  of kind. *(Recorded because an earlier phrasing of Finding 7 — "at zero regardless of width"
  — was scoped to induction but read as a claim about context use generally, which the data
  above refutes.)*
  *Honest scope:* Finding 7 shows WT-103/Pile pretraining did **not** produce induction; it does
  not prove the architecture **cannot** learn it — that requires the synthetic-task training arm
  (MQAR / selective copying, ~$6 at Mini scale), now the highest-value experiment on the
  modern-LLM axis. *Strategic consequence:* if it holds at larger widths (SP1 239M / M1 893M
  pending), the wavelet+attention hybrid (separate repo, house rule 9) stops being a hedge and
  becomes the evidenced path — with a clear division of labour: legible wavelet layers for
  mixing, a thin attention component for recall.
- Instruments to date: `tools/interpretability/coeff_dump.py` (Phase 0),
  `tools/interpretability/census.py` (Study 1: absmean/std/kurtosis over shards),
  `tools/interpretability/topk_contexts.py` (Study 2: top-k contexts + tail concentration),
  `tools/interpretability/wavelet_autopsy.py` (Study 5: effective filters, invariants,
  crawl distributions, trained-vs-init),
  `tools/interpretability/induction_probe.py` (Finding 7: in-context copying, with
  `--hf_model` for transformer positive controls).
- Ops note: bypass cross-window state is batch-shaped and carries across forward calls →
  constant batch size enforced in the tool; zero-state reset at dump start (deterministic).

## What this unblocks

- **Paper 2's spine**, with its central claim tested before a dollar of GPU is spent on it.
- An evidence-based **second-GPU budget** (SAE-scale only if Study 2 demands it).
- The interpretability suite promised in the Release Pipeline ("developed & processed fully
  here" on Small) inherits working instruments instead of starting cold.
