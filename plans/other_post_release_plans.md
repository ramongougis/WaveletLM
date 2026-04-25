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

**What:** A six-fix stabilization suite inspired by the Linear Recurrent Unit paper (Orvieto et al., arXiv:2303.06349), GPT-2's residual-scaling convention, and the Transformer paper's √d_model embedding scaling. Targets four historically-observed NaN failure modes — mixer_depth ≥ 3 at L=20 with lr=0.01, C=2048 with block_size=2048, C=2048 with lr=0.02, and mixer_depth ≥ 5 at any LR without residuals — whose root cause is uncontrolled signal magnitude growth through composed transforms.

All six fixes are implemented in [model.py](../model.py) and wired through the master flag `stable_parametrization` plus individual `stab_*` toggles:

1. **Spectral norm on mixer** (`stab_spectral_norm`, [model.py:461](../model.py#L461)) — constrains largest singular value of `mixer.weight` to 1.0, preventing any single direction from amplifying signal. Directly addresses the NaN-at-depth/LR failure mode. Cost: one SVD-like computation per forward pass.
2. **FF final-layer √(hidden_dim) scaling** (`stab_ff_scaling`, [model.py:553](../model.py#L553)) — replaces the fixed `.mul_(0.02)` with `1/√hidden_dim` Xavier-style scaling. At MLP=20 and C=2048, hidden_dim=40,960, so the old 0.02 scaling left each output dim receiving a sum of ~819 weighted inputs with uncontrolled variance.
3. **Embedding √C scaling** (`stab_embed_scaling`, [model.py:1409](../model.py#L1409)) — multiplies embedding output by √C at runtime, keeping magnitude constant regardless of C. Follows Vaswani et al. (2017).
4. **proj_out √(C·layers) scaling** (`stab_proj_out_scaling`, [model.py:1143](../model.py#L1143)) — replaces `.mul_(1e-3)` with `1/√(C·num_layers)`. Follows GPT-2's residual-stream scaling convention — each layer contributes proportionally less as depth grows, preventing signal growth through the residual path.
5. **Mixer eps scaling with C** (`stab_mixer_eps_scaling`, [model.py:437](../model.py#L437)) — replaces the fixed `eps=1e-3` mixer init noise with `eps/√C`, keeping noise-to-signal ratio consistent across C.
6. **Level-dependent lifting init** (`stab_lifting_level_scaling`, [model.py:275](../model.py#L275)) — gentler predict/update init at higher wavelet levels (where dilated tokens are weakly correlated): `predict_scale = 1/(1 + 0.1·level)`, `update_scale = 0.5/(1 + 0.1·level)`. Speculative; test last.

These fixes address the root cause (unbounded weight growth) rather than the symptom (signal magnitude). If the sweep below validates them, the earlier `mixer_depth_stabilizers` learnable-scalar approach becomes unnecessary.

**Why this matters:** Per [runs.md:361](../runs.md#L361), the six fixes were **never evaluated individually**, and the master-flag attempt on the mixer_depth=5 @ L=20 config silently crashed before step 100 — likely torch.compile + ~1000 spectral-norm-wrapped mixers (L=20 × S=10 × MD=5) exceeded a compilation resource limit. Consequently the flags are unused latent infrastructure in the release configuration. Several latent-win runs are blocked on validation. The most valuable is `lr=0.02 + exp_param` ([runs.md:346](../runs.md#L346)), which previously showed −0.0084 BPB at L=1 but NaN'd at L=2 step 4000 — spectral norm on the mixer is the hypothesized fix. Multi-basis lifting and high-depth mixer configs are also deferred behind this validation. Stable parametrization will likely also be load-bearing for the B200 scale-up (larger C, deeper L, longer block_size) where existing lr=0.01 recipes start hitting stability walls.

**Outstanding work:**

- **Small-scale validation sweep.** Reproduce the three failing configs (C=2048 lr=0.02, C=2048 block_size=2048, mixer_depth=5 at L=20) at reduced scale (e.g. C=512) so each probe costs 1–3h instead of 17h. Run each of the six `stab_*` sub-flags individually against each failing config. Expected behavior: spectral norm most impactful; FF and proj_out scaling provide variance control; eps scaling and level-dependent lifting are refinements.
- **Re-attempt mixer_depth=5 without torch.compile**, or with on-the-fly weight normalization replacing spectral norm in the >3 mixer_depth regime. The crash was a compilation resource issue, not a numerical stability issue, and needs to be diagnosed separately.
- **Gate.weight scaling gap.** [model.py:471](../model.py#L471) currently uses fixed `nn.init.normal_(self.gate.weight, std=0.02)` regardless of C. The original design proposed `N(0, 0.02/√(C/512))` but didn't include it in the priority list; no `stab_gate_scaling` flag exists. At C=2048, the gate output σ(Gx) has 2048 dimensions each with std ~0.02, and the collective effect is larger at higher C. Low priority unless validation sweeps show the gate contributes to instability at high C.

**Alternative direction (not implemented):** Exponential parametrization for the mixer — store raw parameters θ, compute effective weights as `eye(Cp) + diag(exp(θ))·noise_directions`, eigenvalues bounded by `exp(θ)` and controllable via θ. This is already implemented as a separate feature (the `use_mixer_gate` / Exponential Parametrization path in model.py), independent of the stabilization flags. If the spectral-norm fix proves numerically expensive in the B200 regime, the exponential parametrization path is a potential substitute.

**Recommended framing:** Run before the B200 scale-up work. Skip if the B200 recipe happens to train cleanly without stabilizers — but if any NaN appears in the scale-up regime, this is the first place to look, and the ablation work is needed to know which subset to enable rather than flipping the master flag blindly.

---

## 6. Optimizer sweep (Adagrad / AdamW / Muon)

**What:** A controlled comparison of three optimizers on the locked WaveletLM recipe to determine whether the current Adagrad choice is genuinely best or simply the first-validated one. Specifically:

- **Adagrad** (current best, lr=0.01) — already tuned; serves as the baseline. No additional sweep needed.
- **AdamW** (Loshchilov & Hutter, 2017, [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)) — the modern transformer default; previously tried and rejected for WaveletLM but without a dedicated LR sweep. Standard LR ranges typically fall in 1e-5 to 1e-3.
- **Muon** (Jordan et al., 2025, [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)) — orthogonalizes matrix-parameter gradient updates via Newton-Schulz iteration. Reported 1.5–2× wall-clock speedups vs AdamW on small transformers. Untested on wavelet/spectral architectures. Standard LR ranges typically fall in 3e-3 to 3e-2.

**Why this matters:** WaveletLM is matrix-parameter-heavy (MLP at expansion=20 produces Linear(2048, 40960) weights; per-scale mixers, lifting, cross_layer_mix, proj_out are all matrices), which is exactly where Muon's orthogonalization should pay off. If even a 30% wall-clock speedup transfers, every subsequent ablation and the B200 scale-up benefits compoundedly. The current Adagrad result (BPB 1.0149 ± 0.0008 across 3 seeds) was a defensible choice but never directly compared to a properly-tuned Muon or AdamW, so we don't actually know whether it's optimal.

**Experimental protocol (tiered, ~45–65h total):**

- **Phase 1 — 1-epoch LR screening.** ~10–30h.
  - AdamW: 3 LRs (1e-4, 3e-4, 1e-3), one 1-epoch run each.
  - Muon: 3 LRs (3e-3, 1e-2, 3e-2), one 1-epoch run each.
  - Identifies catastrophic failures (NaN), establishes viable LR range per optimizer, gives rough convergence-rate ranking.
  - Adagrad's 1-epoch number is already known from prior runs — no sweep needed.

- **Phase 2 — 5-epoch validation of finalists.** ~17–34h.
  - Take the best LR per optimizer from Phase 1.
  - Run a full 5-epoch training using the locked WaveletLM recipe (L=2, C=2048, MLP=20, PLE, PKM+FwPKM=16384, block_size=256, WD=1e-6, 2.0× dropout). Only the optimizer + its LR change.
  - Compare against the existing Adagrad@5ep BPB 1.0149 ± 0.0008 baseline.

**Per-optimizer parameter-group handling:** Muon should be applied only to "regular" matrix parameters. Embeddings, LM head, PKM/FwPKM value tables, lifting predict/update networks (Haar-initialized, near-orthogonal already), and all scalar/vector parameters (LayerNorm gains, residual α, history_gains, per-scale weights) should stay on AdamW. This is a parameter-group split, ~30–50 lines in [train.py](../train.py)'s optimizer construction.

**Expected outcomes:**

1. **Muon hits BPB ≤ 1.0140 in 3 epochs** → adopt Muon as default; ~40% training-time savings on every future experiment.
2. **Muon matches Adagrad at 5 epochs** → no swap; useful negative result confirming Adagrad's appropriateness.
3. **AdamW also competitive when properly tuned** → revisit the original Adagrad-vs-AdamW decision.
4. **Muon NaNs or destabilizes** (most likely failure mode given the lifting Haar-init interaction) → either fix with parameter-group exclusions or rule it out.

**Why 1-epoch alone is insufficient:** the EMA decompose-bypass probe showed a +0.30 nat val-loss improvement at 1 epoch but regressed at 5 epochs (see [plans/ema_post_release.md](ema_post_release.md)). 1→N epoch inversions are a real risk — Adagrad's accumulator effect is progressive, Muon's orthogonalization compounds across the LR schedule, and warmup/decay durations differ between 1-epoch and 5-epoch runs. 1-epoch screening is good for catching catastrophic failures and ranking by margin; final claims need full-length validation.

**Recommended framing:** This is the highest-impact-per-compute post-release experiment. A successful Muon adoption produces a multiplicative speedup that compounds across the entire post-release roadmap (B200 scaling, MoE-on-MLP investigation, semantic embedding work, etc.). Worth scheduling early in the post-release workstream, before any major architectural sweeps that would benefit from faster training cadence.

---

## Prioritization order for post-release

1. **Optimizer sweep (6)**: highest impact-per-compute; potential ~1.5–2× wall-clock speedup compounds across all subsequent ablations and the B200 scale-up. Run first.
2. **Data-dependent EMA** (further investigation; see [plans/ema_post_release.md](ema_post_release.md) — superseded by post-release work since the 1→5 epoch inversion rejected it pre-release).
3. **Cross-scale phase gating (3)**: cheapest to test, complements existing CSG.
4. **Stable parametrization validation (5)**: gates multiple latent-win runs and the B200 scale-up; small-scale sweep is cheap.
5. **Data-dependent lifting (1)**: largest uncertainty, largest potential payoff, biggest code lift. Start with single-block experiment at small C to calibrate before full sweep.
6. **Wavelet Packet Decomposition (2)**: dedicated research project; don't do simultaneously with (1) or the attribution becomes impossible.
7. **Top-K Hadamard thresholding (4)**: pair with bit-packing as a deployment-optimization bundle.

## Provenance

All five suggestions originated from a Gemini "adversarial reviewer" audit (2026-04-21, user-requested). The reviewer adopted an S4/Mamba/DSP-focused persona. My filtering for pre-release vs post-release was primarily based on (a) code-change surface area, (b) expected BPB impact, and (c) whether the change could be A/B-tested cleanly in the ~20h pre-release window. Only data-dependent EMA cleared all three bars.
