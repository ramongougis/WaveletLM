# Post-release architectural extensions

Deferred items from Gemini's adversarial architectural audit (2026-04-21). The pre-release plan tests only one of these (data-dependent EMA for `decompose_bypass` — see `runs.md#decompose-bypass-data-dependent-ema-probe-1-epoch`). The rest below are significant changes that require dedicated exploration, not release-gating.

For each: a short "what it is," a fair impact estimate, and the main implementation hurdles.

---

## 1. Data-dependent lifting networks (Mamba-style)

**What:** The Predict and Update networks inside `LiftingWaveletDecompose` are currently static MLPs (initialized to Haar). Gemini's suggestion: make them input-conditioned, so at each token the lifting scheme adaptively chooses its basis. DSP equivalent: an adaptive filter (Kalman/Wiener) that changes shape based on local signal predictability.

**Expected impact:** Potentially large (−0.01 to −0.05 BPB) but highly uncertain. This is the most architecturally invasive of Gemini's suggestions and the one with the highest ceiling.

**Implementation hurdles:**
- Breaks `shared_lifting_weights` (each token's lifting is now different).
- Requires careful init — data-dependent predict/update networks initialized randomly would destabilize the Haar-seed invariant that makes our current lifting trainable. Need low-rank perturbations around the static baseline, probably with near-zero init on the data-conditioning path so the block starts identical to today's behavior.
- Parameter count grows non-trivially (per-token gating projection adds ~C² params per lifting level).

**Recommended framing:** Paper-scale extension, not a patch. A "WaveletLM-Mamba" or "Adaptive WaveletLM" follow-up.

---

## 2. Wavelet Packet Decomposition (WPD)

**What:** The current decomposition iteratively splits only the low-pass (Approximation) band — standard DWT. WPD applies the lifting scheme to both the Approximation *and* the Details at each level, yielding a full binary tree of frequency sub-bands instead of a lopsided cascade.

**Rationale:** Language isn't audio — high-frequency semantic shifts ("NOT" negating a sentence, topic transitions) matter as much as low-frequency themes. The per-scale mixer widths ablation already showed fine-scale detail carries real signal; WPD extends that intuition systematically.

**Expected impact:** Moderate (−0.005 to −0.02 BPB). Roughly doubles the number of sub-bands (from `levels+1` to `2^levels`), proportionally increasing mixer parameters at the same per-scale width.

**Implementation hurdles:**
- Lifting reconstruction logic needs to handle a full binary tree, not a single cascade. `LiftingWaveletReconstruct` rewrite.
- Mixer geometry changes — `S = levels + 1` → `S = 2^levels`. `per_scale_mixer_widths`, `cross_scale_gating`, `wavelet_crawl`, and the PTQ per-scale bit tiers all need reinterpretation for a packet tree.
- Expect a new sweep over levels × widths to re-tune; existing per-scale_mixer_widths values don't transfer directly.

**Recommended framing:** Separate research direction; probably a dedicated 1-2 week effort with its own ablation set.

---

## 3. Cross-scale phase gating (coarse-modulates-fine)

**What:** Extend `cross_scale_gating` beyond the current learned (S, S) routing matrix with a specific structural form: coarse-scale outputs sigmoid-gate fine-scale processing.

```
Detail_level_k = Detail_level_k * σ(Linear(Approximation_level_0))
```

**Rationale:** In classical DSP, the alignment between coarse and fine coefficients carries phase information about where edges occur. If the coarse scale signals "we're inside a predictable word," suppress the fine-detail computation; if it signals "boundary/transition," amplify detail processing. More structured than today's free (S, S) routing.

**Expected impact:** Small to moderate (−0.001 to −0.005 BPB). Marginal over the existing `cross_scale_gating=true` since that already allows this pattern to emerge from the identity-initialized routing matrix if the model chooses.

**Implementation hurdles:**
- Low. Additional parameters are small. Mostly a question of whether the explicit structural form beats the current data-driven (S, S) matrix in practice.

**Recommended framing:** Easy v1.1 candidate after release. Good candidate for a short weekend ablation.

---

## 4. Top-K / hard thresholding in the Hadamard domain

**What:** After `FastHadamardTransform` concentrates channel energy, apply a hard threshold that zeros out low-magnitude coefficients before the spectral mixer:

```
X_spec = X_spec * (|X_spec| > threshold)
```

Acts as a non-linear denoiser, forcing the mixer to process only the principal Hadamard components.

**Expected impact:** Small for BPB (training-time gradient flow through a hard threshold is fragile; soft L1 penalty is more stable but gains are small). Larger for deployment — pairs naturally with PTQ's per-scale sensitivity: fine-scale mixer scales already tolerate 2-bit, and top-K thresholding further reduces the work done at those scales.

**Implementation hurdles:**
- Hard thresholds break gradient flow; use soft alternatives (L1, straight-through estimator, or learned soft-thresholding) for training. Hard threshold only at inference.
- Threshold tuning per-scale — same calibration effort as the PTQ bit-width tiers.

**Recommended framing:** Deployment optimization rather than accuracy story. Reaches consumer-GPU inference latency better; doesn't unlock new BPB territory. Pairs with bit-packing as the "inference speedup bundle."

---

## 5. Stable parametrization — validation and finishing gaps

**What:** The six stability fixes described in [`plans/stable_parametrization.md`](stable_parametrization.md) (spectral norm on mixer, FF √(hidden_dim) scaling, embedding √C scaling, proj_out √(C·layers) scaling, mixer eps scaling with C, level-dependent lifting init) are all implemented in [model.py](../model.py) and wired through the master flag `stable_parametrization` plus individual `stab_*` toggles. But per [runs.md:361](../runs.md#L361), they were **never evaluated individually**, and the master-flag attempt on the mixer_depth=5 @ L=20 config silently crashed before step 100 (likely torch.compile + ~1000 spectral-norm-wrapped mixers exceeding a compilation resource limit). Consequently the flags are unused latent infrastructure in the release configuration.

**Why this matters:** Several latent-win runs are blocked on this validation. The most valuable is `lr=0.02 + exp_param` ([runs.md:346](../runs.md#L346)), which previously showed −0.0084 BPB at L=1 but NaN'd at L=2 step 4000 — spectral norm on the mixer is the hypothesized fix. Multi-basis lifting and high-depth mixer configs are also deferred behind this validation. Stable parametrization will likely also be load-bearing for the B200 scale-up (larger C, deeper L, longer block_size) where existing lr=0.01 recipes start hitting stability walls.

**Outstanding work:**

- **Small-scale validation sweep.** Reproduce the three failing configs from the original testing plan (C=2048 lr=0.02, C=2048 block_size=2048, mixer_depth=5 at L=20) at reduced scale (e.g. C=512) so each probe costs 1–3h instead of 17h. Run each of the six `stab_*` sub-flags individually against each failing config. Identifies which fixes are load-bearing and in which failure modes.
- **Re-attempt mixer_depth=5 without torch.compile**, or with on-the-fly weight normalization replacing spectral norm in the >3 mixer_depth regime. The failure was a compilation resource issue, not a stability issue, and needs to be diagnosed separately from the numerical question.
- **Gate.weight scaling gap.** [model.py:471](../model.py#L471) currently uses fixed `nn.init.normal_(self.gate.weight, std=0.02)` regardless of C. The original plan (section 2, lines 55–62) proposed `N(0, 0.02 / sqrt(C / 512))` but didn't include it in the priority list; no `stab_gate_scaling` flag exists. Low priority unless validation sweeps show the gate contributes to instability at high C, in which case add the flag to match.
- **Update [plans/stable_parametrization.md](stable_parametrization.md)** to reflect actual status: mark the six as implemented, document the mixer_depth=5 crash with its likely compile-limit cause, list the gate.weight gap, and convert "Testing plan" into an outstanding validation backlog with pointers to the runs.md entries.

**Recommended framing:** Run before the B200 scale-up work. Skip if the B200 recipe happens to train cleanly without stabilizers — but if any NaN appears in the scale-up regime, this is the first place to look, and the ablation work is needed to know which subset to enable rather than flipping the master flag blindly.

---

## Prioritization order for post-release

1. **Data-dependent EMA** (further investigation; see [plans/ema_post_release.md](ema_post_release.md) — superseded by post-release work since the 1→5 epoch inversion rejected it pre-release).
2. **Cross-scale phase gating (3)**: cheapest to test, complements existing CSG.
3. **Stable parametrization validation (5)**: gates multiple latent-win runs and the B200 scale-up; small-scale sweep is cheap.
4. **Data-dependent lifting (1)**: largest uncertainty, largest potential payoff, biggest code lift. Start with single-block experiment at small C to calibrate before full sweep.
5. **Wavelet Packet Decomposition (2)**: dedicated research project; don't do simultaneously with (1) or the attribution becomes impossible.
6. **Top-K Hadamard thresholding (4)**: pair with bit-packing as a deployment-optimization bundle.

## Provenance

All five suggestions originated from a Gemini "adversarial reviewer" audit (2026-04-21, user-requested). The reviewer adopted an S4/Mamba/DSP-focused persona. My filtering for pre-release vs post-release was primarily based on (a) code-change surface area, (b) expected BPB impact, and (c) whether the change could be A/B-tested cleanly in the ~20h pre-release window. Only data-dependent EMA cleared all three bars.
