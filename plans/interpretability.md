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
2. **K0 (C=100)** — the starved-width contrast case.
3. **SP1 Small (C=1024, 239M)** — flagship cross-check (956MB fp32; still CPU-fine, slower).

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
- **Prediction on record** *(estimate)*: coarsest-scale ablation degrades long-range
  coherence/topic maintenance disproportionately; finest-scale ablation hits local grammar
  and spelling; mid scales are where the interesting semantics live.

### Study 4 — Scale-role dissection via linear probes
Logistic-regression probes on cached per-scale coefficient streams for cheap linguistic
labels (POS, sentence position, quote/list/heading context, topic id). Pure sklearn on
shards. Deliverable: a **scale × feature heat map** — "which temporal scale encodes what."
This story has no transformer analogue (they have no scale axis); it is uniquely ours.

### Study 5 — What did the lifting learn?
Compare the learned shared lifting against its Haar init: per-level filter taps, frequency
responses, the crawl's dilation-softmax weights per level (extends the crawl probe), and the
shrinkage λ-map's "protect-ends / squeeze-middle" finding (README §Coefficient Shrinkage).
This is the study that **meets the Strang & Nguyen reading in the middle** — frequency
responses of learned filter banks are exactly its material, making the theory pass
observation-driven rather than prerequisite-driven.

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

## What this unblocks

- **Paper 2's spine**, with its central claim tested before a dollar of GPU is spent on it.
- An evidence-based **second-GPU budget** (SAE-scale only if Study 2 demands it).
- The interpretability suite promised in the Release Pipeline ("developed & processed fully
  here" on Small) inherits working instruments instead of starting cold.
