# Post-release architectural extensions

The pre-release plan tests only one of these (data-dependent EMA for `decompose_bypass`: see `runs.md#decompose-bypass-data-dependent-ema-probe-1-epoch`). The rest below are significant changes that require dedicated exploration post-release.

---

## 1. Data-dependent lifting networks (Mamba-style)

**Explanation:** The Predict and Update networks inside `LiftingWaveletDecompose` are currently static MLPs initialized to Haar. Suggestion: make them input-conditioned, so at each token the lifting scheme adaptively chooses its basis. DSP equivalent: an adaptive filter (Kalman/Wiener) that changes shape based on local signal predictability.

**Expected impact:** Potentially large (−0.01 to −0.05 BPB) but highly uncertain. This is the most architecturally invasive suggestion and the one with the highest ceiling for improvement.

**Implementation hurdles:**
- Breaks `shared_lifting_weights` (each token's lifting is now different).
- Requires careful init: data-dependent predict/update networks initialized randomly would destabilize the Haar-seed invariant that makes our current lifting trainable. Need low-rank perturbations around the static baseline, probably with near-zero init on the data-conditioning path so the block starts identical to today's behavior.
- Parameter count grows non-trivially (per-token gating projection adds ~C² params per lifting level).

**Recommended framing:** Paper-scale extension, not a patch. A "WaveletLM-Mamba" or "Adaptive WaveletLM" follow-up.

---

## 2. Wavelet Packet Decomposition (WPD)

**Explanation:** The current decomposition iteratively splits only the low-pass (Approximation) band as a standard DWT. WPD applies the lifting scheme to both the Approximation *and* the Details at each level, yielding a full binary tree of frequency sub-bands instead of a lopsided cascade.

**Rationale:** Language isn't audio. High-frequency semantic shifts ("NOT" negating a sentence, topic transitions) matter as much as low-frequency themes. The per-scale mixer widths ablation already showed fine-scale detail carries real signal; WPD extends that intuition systematically.

**Expected impact:** Moderate (−0.005 to −0.02 BPB). Roughly doubles the number of sub-bands (from `levels+1` to `2^levels`), proportionally increasing mixer parameters at the same per-scale width.

**Implementation hurdles:**
- Lifting reconstruction logic needs to handle a full binary tree, not a single cascade. `LiftingWaveletReconstruct` rewrite.
- Mixer geometry changes: `S = levels + 1` → `S = 2^levels`. `per_scale_mixer_widths`, `cross_scale_gating`, `wavelet_crawl`, and the PTQ per-scale bit tiers all need reinterpretation for a packet tree.
- Expect a new sweep over levels × widths to re-tune; existing per-scale_mixer_widths values don't transfer directly.

**Recommended framing:** Separate research direction; probably a dedicated 1-2 week effort with its own ablation set.

---

## 3. Cross-scale phase gating (coarse-modulates-fine)

**Explanation:** Extend `cross_scale_gating` beyond the current learned (S, S) routing matrix with a specific structural form: coarse-scale outputs sigmoid-gate fine-scale processing.

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

**Explanation:** After `FastHadamardTransform` concentrates channel energy, apply a hard threshold that zeros out low-magnitude coefficients before the spectral mixer:

```
X_spec = X_spec * (|X_spec| > threshold)
```

Acts as a non-linear denoiser, forcing the mixer to process only the principal Hadamard components.

**Expected impact:** Small for BPB (training-time gradient flow through a hard threshold is fragile; soft L1 penalty is more stable but gains are small). Larger for deployment. Pairs naturally with PTQ's per-scale sensitivity: fine-scale mixer scales already tolerate 2-bit, and top-K thresholding further reduces the work done at those scales.

**Implementation hurdles:**
- Hard thresholds break gradient flow; use soft alternatives (L1, straight-through estimator, or learned soft-thresholding) for training. Hard threshold only at inference.
- Threshold tuning per-scale. Same calibration effort as the PTQ bit-width tiers.

**Recommended framing:** Deployment optimization rather than accuracy story. Reaches consumer-GPU inference latency better; doesn't unlock new BPB territory. Pairs with bit-packing as the "inference speedup bundle."

---

## 5. Stable parametrization: validation and finishing gaps

**Explanation:** A six-fix stabilization suite inspired by the Linear Recurrent Unit paper (Orvieto et al., arXiv:2303.06349), GPT-2's residual-scaling convention, and the Transformer paper's √d_model embedding scaling. Targets four historically-observed NaN failure modes: mixer_depth ≥ 3 at L=20 with lr=0.01, C=2048 with block_size=2048, C=2048 with lr=0.02, and mixer_depth ≥ 5 at any LR without residuals, whose root cause is uncontrolled signal magnitude growth through composed transforms.

All six fixes are implemented in [model.py](../model.py) and wired through the master flag `stable_parametrization` plus individual `stab_*` toggles:

1. **Spectral norm on mixer** (`stab_spectral_norm`, [model.py:461](../model.py#L461)): constrains largest singular value of `mixer.weight` to 1.0, preventing any single direction from amplifying signal. Directly addresses the NaN-at-depth/LR failure mode. Cost: one SVD-like computation per forward pass.
2. **FF final-layer √(hidden_dim) scaling** (`stab_ff_scaling`, [model.py:553](../model.py#L553)): replaces the fixed `.mul_(0.02)` with `1/√hidden_dim` Xavier-style scaling. At MLP=20 and C=2048, hidden_dim=40,960, so the old 0.02 scaling left each output dim receiving a sum of ~819 weighted inputs with uncontrolled variance.
3. **Embedding √C scaling** (`stab_embed_scaling`, [model.py:1409](../model.py#L1409)): multiplies embedding output by √C at runtime, keeping magnitude constant regardless of C. Follows Vaswani et al. (2017).
4. **proj_out √(C·layers) scaling** (`stab_proj_out_scaling`, [model.py:1143](../model.py#L1143)): replaces `.mul_(1e-3)` with `1/√(C·num_layers)`. Follows GPT-2's residual-stream scaling convention, where each layer contributes proportionally less as depth grows, preventing signal growth through the residual path.
5. **Mixer eps scaling with C** (`stab_mixer_eps_scaling`, [model.py:437](../model.py#L437)): replaces the fixed `eps=1e-3` mixer init noise with `eps/√C`, keeping noise-to-signal ratio consistent across C.
6. **Level-dependent lifting init** (`stab_lifting_level_scaling`, [model.py:275](../model.py#L275)): gentler predict/update init at higher wavelet levels (where dilated tokens are weakly correlated): `predict_scale = 1/(1 + 0.1·level)`, `update_scale = 0.5/(1 + 0.1·level)`. Speculative; test last.

These fixes address the root cause (unbounded weight growth) rather than the symptom (signal magnitude). If the sweep below validates them, the earlier `mixer_depth_stabilizers` learnable-scalar approach becomes unnecessary.

**Why this matters:** Per [runs.md:361](../runs.md#L361), the six fixes were **never evaluated individually**, and the master-flag attempt on the mixer_depth=5 @ L=20 config silently crashed before step 100 - likely torch.compile + ~1000 spectral-norm-wrapped mixers (L=20 × S=10 × MD=5) exceeded a compilation resource limit. Consequently the flags are unused latent infrastructure in the release configuration. Several latent-win runs are blocked on validation. The most valuable is `lr=0.02 + exp_param` ([runs.md:346](../runs.md#L346)), which previously showed −0.0084 BPB at L=1 but NaN'd at L=2 step 4000. A spectral norm on the mixer is the hypothesized fix. Multi-basis lifting and high-depth mixer configs are also deferred behind this validation. Stable parametrization will likely also be load-bearing for the B200 scale-up (larger C, deeper L, longer block_size) where existing lr=0.01 recipes start hitting stability walls.

**Outstanding work:**

- **Small-scale validation sweep.** Reproduce the three failing configs (C=2048 lr=0.02, C=2048 block_size=2048, mixer_depth=5 at L=20) at reduced scale (e.g. C=512) so each probe costs 1–3h instead of 17h. Run each of the six `stab_*` sub-flags individually against each failing config. Expected behavior: spectral norm most impactful; FF and proj_out scaling provide variance control; eps scaling and level-dependent lifting are refinements.
- **Re-attempt mixer_depth=5 without torch.compile**, or with on-the-fly weight normalization replacing spectral norm in the >3 mixer_depth regime. The crash was a compilation resource issue, not a numerical stability issue, and needs to be diagnosed separately.
- **Gate.weight scaling gap.** [model.py:471](../model.py#L471) currently uses fixed `nn.init.normal_(self.gate.weight, std=0.02)` regardless of C. The original design proposed `N(0, 0.02/√(C/512))` but didn't include it in the priority list; no `stab_gate_scaling` flag exists. At C=2048, the gate output σ(Gx) has 2048 dimensions each with std ~0.02, and the collective effect is larger at higher C. Low priority unless validation sweeps show the gate contributes to instability at high C.

**Alternative direction (not implemented):** Exponential parametrization for the mixer: store raw parameters θ, compute effective weights as `eye(Cp) + diag(exp(θ))·noise_directions`, eigenvalues bounded by `exp(θ)` and controllable via θ. This is already implemented as a separate feature (the `use_mixer_gate` / Exponential Parametrization path in model.py), independent of the stabilization flags. If the spectral-norm fix proves numerically expensive in the B200 regime, the exponential parametrization path is a potential substitute.

**Recommended framing:** Run before the B200 scale-up work. Skip if the B200 recipe happens to train cleanly without stabilizers, but if any NaN appears in the scale-up regime, this is the first place to look, and the ablation work is needed to know which subset to enable rather than flipping the master flag blindly.

---

## 6. Optimizer sweep (Adagrad / AdamW / Muon)

**Explanation:** A controlled comparison of three optimizers on the locked WaveletLM recipe to determine whether the current Adagrad choice is genuinely best or simply the first-validated one. Specifically:

- **Adagrad** (current best, lr=0.01): already tuned; serves as the baseline. No additional sweep needed.
- **AdamW** (Loshchilov & Hutter, 2017, [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)): the modern transformer default; previously tried and rejected for WaveletLM but without a dedicated LR sweep. Standard LR ranges typically fall in 1e-5 to 1e-3.
- **Muon** (Jordan et al., 2025, [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)): orthogonalizes matrix-parameter gradient updates via Newton-Schulz iteration. Reported 1.5–2× wall-clock speedups vs AdamW on small transformers. Untested on wavelet/spectral architectures. Standard LR ranges typically fall in 3e-3 to 3e-2.

**Why this matters:** WaveletLM is matrix-parameter-heavy (MLP at expansion=20 produces Linear(2048, 40960) weights; per-scale mixers, lifting, cross_layer_mix, proj_out are all matrices), which is exactly where Muon's orthogonalization should pay off. If even a 30% wall-clock speedup transfers, every subsequent ablation and the B200 scale-up benefits compoundedly. The current Adagrad result (BPB 1.0149 ± 0.0008 across 3 seeds) was a defensible choice but never directly compared to a properly-tuned Muon or AdamW, so we don't actually know whether it's optimal.

**Experimental protocol (tiered, ~45–65h total):**

- **Phase 1: 1-epoch LR screening.** ~10–30h.
  - AdamW: 3 LRs (1e-4, 3e-4, 1e-3), one 1-epoch run each.
  - Muon: 3 LRs (3e-3, 1e-2, 3e-2), one 1-epoch run each.
  - Identifies catastrophic failures (NaN), establishes viable LR range per optimizer, gives rough convergence-rate ranking.
  - Adagrad's 1-epoch number is already known from prior runs, so no sweep needed.

- **Phase 2: 5-epoch validation of finalists.** ~17–34h.
  - Take the best LR per optimizer from Phase 1.
  - Run a full 5-epoch training using the locked WaveletLM recipe (L=2, C=2048, MLP=20, PLE, PKM+FwPKM=16384, block_size=256, WD=1e-6, 2.0× dropout). Only the optimizer + its LR change.
  - Compare against the existing Adagrad@5ep BPB 1.0149 ± 0.0008 baseline.

**Per-optimizer parameter-group handling:** Muon should be applied only to "regular" matrix parameters. Embeddings, LM head, PKM/FwPKM value tables, lifting predict/update networks (Haar-initialized, near-orthogonal already), and all scalar/vector parameters (LayerNorm gains, residual α, history_gains, per-scale weights) should stay on AdamW. This is a parameter-group split, ~30–50 lines in [train.py](../train.py)'s optimizer construction.

**Expected outcomes:**

1. **Muon hits BPB ≤ 1.0140 in 3 epochs** → adopt Muon as default; ~40% training-time savings on every future experiment.
2. **Muon matches Adagrad at 5 epochs** → no swap; useful negative result confirming Adagrad's appropriateness.
3. **AdamW also competitive when properly tuned** → revisit the original Adagrad-vs-AdamW decision.
4. **Muon NaNs or destabilizes** (most likely failure mode given the lifting Haar-init interaction) → either fix with parameter-group exclusions or rule it out.

**Why 1-epoch alone is insufficient:** the EMA decompose-bypass probe showed a +0.30 nat val-loss improvement at 1 epoch but regressed at 5 epochs (see [plans/ema_post_release.md](ema_post_release.md)). 1→N epoch inversions are a real risk. Adagrad's accumulator effect is progressive, Muon's orthogonalization compounds across the LR schedule, and warmup/decay durations differ between 1-epoch and 5-epoch runs. 1-epoch screening is good for catching catastrophic failures and ranking by margin; final claims need full-length validation.

**Recommended framing:** This is the highest-impact-per-compute post-release experiment. A successful Muon adoption produces a multiplicative speedup that compounds across the entire post-release roadmap (B200 scaling, MoE-on-MLP investigation, semantic embedding work, etc.). Worth scheduling early in the post-release workstream, before any major architectural sweeps that would benefit from faster training cadence.

---

## 7. Inference strategies ablations

The `--strategies` bundle currently enables four real decoding interventions (entropy-adaptive temperature with cap 0.9, `top_p=0.85`, `repetition_penalty=1.2`, plus metrics + spacing cleanup). Three heavier strategies — `best_of_n`, multi-token `lookahead`, and `wavelet_coherence` decoding bias — were tested pre-WaveletLM and excluded from the bundle on a cost-vs-benefit basis (too slow, too VRAM-hungry, or quality-equivalent to lighter alternatives). Per-strategy contribution within the active bundle is unmeasured. The pre-WaveletLM rejection of the heavier strategies has not been re-validated against the current architecture.

### Open questions

1. Within the active `--strategies` bundle, which of the four interventions contributes the most to subjective sample quality? Plausible top contender: `repetition_penalty=1.2` (suppresses the model's natural tendency to loop, which would otherwise be visible).
2. Is the entropy-adaptive temperature with cap 0.9 doing useful work, or is it indistinguishable in practice from a static `temperature=0.8`?
3. Does `top_p=0.85` strictly dominate `top_p=0.95` for this model, or is the difference within sample noise?
4. **Re-validation question:** were the heavier strategies (`best_of_n`, `lookahead`, `wavelet_coherence`) actually worse-or-equivalent for *WaveletLM*, or were they tested on an earlier architecture where the cost/benefit shifted? The wavelet_coherence component in particular is wavelet-architecture-specific and may behave differently here than in pre-WaveletLM testing.

### Proposed study

N=20 samples per condition, single prompt ("The history of"), single seed, on the WT-103 best checkpoint.

**Phase 1 — within-bundle ablation:**

- Naive (`temp=1.0, top_p=0.95`, no other strategies)
- `--strategies` (full bundle — current default)
- `--strategies` minus `repetition_penalty` (set to 1.0)
- `--strategies` minus `entropy_adaptive` (replace with static `temperature=0.8`)
- `--strategies` with `top_p=0.95` instead of `0.85`

**Phase 2 — re-validate the excluded strategies:**

- `--strategies` plus `--best_of_n 5`
- `--strategies` plus `--lookahead_k 5 --lookahead_depth 5`
- `--strategies` plus `--wavelet_coherence`
- `--strategies` plus all three above (the maximum-strategies regime)

Metrics: mean log-probability (already logged), Distinct-1/2/3 (already logged), Rep-4 (already logged), per-token wall-clock time, peak VRAM, and human qualitative ranking on a 5-point scale across blinded conditions.

Compute cost: ~3-4 hours total on a 5090 (single checkpoint, no retraining; Phase 2 conditions are individually slower than Phase 1).

### Why this matters

Phase 1 attributes the visible quality gap between Sample D (naive) and Samples A–C (strategies-on) in the README to specific decoding interventions, which closes a residual interpretation gap for adversarial readers. Phase 2 either re-confirms the pre-WaveletLM rejection of the heavier strategies (in which case the bundle is well-chosen) or surfaces a case for re-bundling them (in which case `--strategies` should be expanded). The wavelet-coherence component in particular has no other validation in the literature — its inclusion in or exclusion from the bundle should rest on a measurement, not on an inherited heuristic.

---

## 8. Combined parameter reduction and VRAM reallocation

The current best WikiText-103 run uses 882.5M parameters chosen to maximize quality within budget. Several individual ablations from `runs.md` suggest substantial parameter savings are available with modest quality cost — and the freed VRAM can be reallocated to longer block size, larger effective batch size, or more training steps per second to recover (and potentially exceed) the lost quality.

### Parameter reductions in the proposed bundle

Combining the four cheapest reductions:

| Change | Param savings | Projected BPB cost (vs prior baseline) | Notes |
|---|---|---|---|
| `mlp_expansion: 10` (from 20) | 167.8M | +0.0045 BPB | Per [`runs.md` line 109](../runs.md). Largest activation-memory saving as well — halves MLP intermediate at any context length. |
| `pkm_enabled: false` | 76.0M | small (~0.0022 BPB) | Per [`runs.md` line 119, runs 23 vs 25](../runs.md): FwPKM-only (-0.0025 BPB vs baseline) and PKM-only (-0.0022 BPB) perform comparably, with FwPKM slightly better. Drop PKM rather than FwPKM to preserve the optional `fwpkm_inference_updates` feature path. |
| `fwpkm_num_keys: 8281` (from 16384) | 38.0M | ~0.009 BPB (interpolated) | 8281 = 91² (FwPKM requires perfect-square num_keys, same as PKM). Halves FwPKM value table; key-table savings minor. |
| `tie_embedding_to_lm_head: true` | 102.9M | +0.0064 BPB | Per [`runs.md` line 85](../runs.md). Standard practice in modern LMs; sometimes helps via regularization. |
| **Bundle total** | **~384.7M (-43.6%)** | **~+0.022 BPB additive estimate** | Additivity is a leap — true cost could be 0.5×–2× the predicted value. |

Resulting model size: **~497.8M parameters** (from 882.5M).

### Projected quality cost (worst-case additive)

Additive estimate of +0.025 BPB at the prior baseline scale translates to roughly +1.7 PPL on WT-103 (using the conversion `ΔPPL ≈ PPL × bytes_per_token × ln(2) × ΔBPB`). This would put a reduced model at ~25.5 PPL on WT-103 if naively applied to the current best run. **At that level, WaveletLM would beat only vanilla GPT-2 (29.4) in the comparison table — losing to Transformer-XL Standard (24.0) and everything above it.**

This worst-case framing assumes the bundle costs add linearly *and* that the prior-baseline deltas hold at the current best-run scale. Both are unverified.

### Why this might not be the worst case

Several factors plausibly reduce the actual cost:

- **Additivity is rarely exact.** Stacked changes often have sub-additive interactions (one change's loss is partly absorbed by another's slack). Could shrink the realized cost meaningfully.
- **Better baselines lose less per ablation.** The prior-baseline deltas were measured at higher BPB. At a lower-PPL regime, the same architectural change typically costs less.
- **Freed VRAM can reallocate to recovery.** Reducing parameters frees substantial memory and compute. That budget can be spent on:
  - Larger `micro_batch_size` (better gradient statistics per step)
  - Longer `block_size` (genuine architectural improvement; most beneficial on long-dependency corpora like PG-19)
  - More epochs at fixed wall-clock budget (more training)
  - Re-tuned dropout (the prior dropout-doubling ablation gave -0.0221 BPB at the prior baseline — almost exactly cancels the projected reduction cost)

The reallocation potential is the main reason this bundle is worth running: parameter reduction is not just an efficiency story, it's an *enabler* for context-length and training-time experiments that aren't currently feasible at the 882.5M scale on consumer hardware.

### Proposed test

**Single combined run**, on WT-103, current best-run config except for the four bundled changes above. 5 epochs, single seed, on a 5090. Goal: measure realized BPB vs the additive projection.

After that run completes, the natural follow-ups are:

1. **Same reduced config + extended block size**: at ~497.8M, the freed activation memory enables `block_size: 1024` or higher on a 5090 (currently bounded near 256–512 at the 882.5M config). Run on WT-103 first to verify the architecture is stable at longer context, then port to PG-19 where long-range dependencies actually matter.
2. **Same reduced config + larger micro_batch_size**: VRAM headroom probably allows MBS=16–32 vs current MBS=8. Larger effective batch may help with gradient noise.
3. **Same reduced config + dropout sweep**: re-tune the five dropout values (`dropout_lm_head`, `dropout_mlp`, `dropout_mixer`, `dropout_projection`, `dropout_embedding`) at the new scale. Most likely place to recover the projected BPB cost.
4. **Same reduced config + more epochs**: at ~43% fewer parameters, per-step wall-clock drops meaningfully. The same compute budget buys more epochs.

The strategic frame: **the parameter reduction is a means to enable longer context, larger batch, and more training within the same hardware budget — not an end in itself.** The realized quality should be evaluated against "current best at fixed compute," not against "current best at fixed config."

### Open questions

1. Does the additive BPB cost estimate hold at the current best-run scale, or do the cost terms compound super-additively when stacked?
2. Does the freed VRAM, reallocated to longer context (e.g., `block_size: 1024`), recover or exceed the parameter-reduction quality cost on WT-103? On PG-19?
3. Does the freed wall-clock per step let more epochs of training compensate within the same total compute budget?
4. Is the FwPKM removal's loss of the `fwpkm_inference_updates` feature worth the parameter savings, or should that feature be tested before the FwPKM module is permanently removed?
5. Where is the Pareto frontier? At what reduced parameter count does quality degradation become non-recoverable by the reallocation strategies above?

### Why this matters

Parameter efficiency was the second of the two substantive critiques received in the public comment thread on the post-release announcement. The current 882.5M parameter count is a "maximize quality within budget" choice, not a Pareto-optimal point — that's fine as a research configuration but reads as parameter-inefficient relative to baselines like Transformer-XL Standard (151M) that achieve comparable quality with 5.8× fewer parameters. A measured Pareto-trade study would either (a) demonstrate that the architecture *can* be parameter-efficient when configured for it, closing the critique vector, or (b) confirm the current architecture is genuinely parameter-inefficient at its quality regime, which is itself useful information for future development direction.

The bundle is also the prerequisite for the long-context experiments needed to validate WaveletLM's central O(n log n) architectural claim (currently theoretical, since `block_size: 256` is below the regime where the asymptotic advantage matters). Running it serves both the efficiency story and the architectural-validation story.

---

## 9. Cross-layer parameter sharing (ALBERT-style)

With `layers: 2` in the current best run, every component (Mixer, MLP, PKM, FwPKM, plus the unlisted per-layer items) has two independent copies. ALBERT (Lan et al. 2019) demonstrated that for transformers, sharing weights across layers can reduce parameter count substantially while preserving most quality, on the hypothesis that depth-via-iteration substitutes for depth-via-distinct-parameters. This plan tests whether the same holds for WaveletLM's wavelet-based architecture.

Note that lifting weights are already shared across layers via `shared_lifting_weights: True`. This plan tests sharing the *remaining* per-layer components on top of that.

### Components and potential savings (with `layers: 2`)

Per the parameter breakdown, per-layer component costs are:

- Mixer: 44.11M → tying saves 44.11M
- MLP: 167.82M → tying saves 167.82M (largest single lever)
- PKM: 38.01M → tying saves 38.01M (mostly value table)
- FwPKM: 38.01M → tying saves 38.01M (mostly value table)
- Per-layer unlisted (proj_out, decompose_bypass, cross_scale_gating, etc.): ~50M → tying saves ~50M

**Maximum possible savings via full layer-tying: ~338M (~38% of current 882.5M).** Combined with the parameter-reduction bundle (Section 8), this could bring the model to roughly 300M parameters at the same architectural depth.

### Proposed test sequence

The order is designed for ablation-style attribution: full tying first to establish the ceiling, then component-isolated tying to identify which sharing carries the cost (or savings) most cleanly.

**Phase 1 — Full layer-block tying.**
Share every per-layer component across both layers. The two layers become structurally identical. Compares against the parameter-reduced baseline from Section 8. If quality holds (or approaches), this is the strongest possible parameter-efficiency result.

**Phase 2 — MLP-only tying.**
Un-tie everything except the MLP. The MLP is 167.82M of per-layer cost — the single largest component — so MLP-only tying is the most likely to carry meaningful savings. If Phase 1 hurts but Phase 2 holds, the MLP is the safe component to share and other components must remain distinct.

**Phase 3 — PKM/FwPKM-only tying.**
Un-tie everything except the sparse memory modules. PKM and FwPKM together carry per-layer cost ~76M, mostly in the value tables. Tests whether the sparse-memory access patterns are tolerant of sharing (different layers querying the same memory bank) or whether layer-distinct value tables matter.

### Implementation considerations

- **Tying is structural, not just parameter-equality**: requires constructing the model with a single ModuleList for the shared component and referencing it twice in the layer stack. Not a config change — a model-construction change.
- **Gradient flow**: shared parameters receive gradient contributions from both layers. Effective learning rate on shared parameters is ~2× higher than on un-shared ones; may need adjustment.
- **Initialization**: shared parameters need a single initialization that works for both layers' use cases, rather than per-layer init.
- **Inference behavior**: identical. Tied parameters add no inference cost overhead — they just save VRAM on the parameter count side.

### Open questions

1. Does full layer-tying preserve quality, or does WaveletLM (with only 2 layers) lack the depth to make iteration-based capacity work the way ALBERT does (which had 12+ layers)?
2. Is MLP-tying alone enough to capture most of the parameter savings without the quality hit of full tying?
3. Does PKM/FwPKM tying interact pathologically with the sparse-key access pattern?
4. At only 2 layers, is "tying" effectively the same as "use 1 layer with twice the depth-equivalent processing"? If so, that suggests the test is really "is 1 unique layer enough?" — which is itself an interesting result.

### Why this matters

Combined with the parameter-reduction bundle, full layer-tying would bring WaveletLM from 882.5M to potentially under 300M parameters — putting it in direct parameter-count competition with Transformer-XL Standard (151M) at hopefully comparable quality. The combination addresses the parameter-inefficiency critique decisively if it works, and provides architecturally interesting negative results if it doesn't.

---

## 10. Per-scale mixer transform ablation

The FWHT slot in the per-scale mixer between the wavelet decomposition and gated SwiGLU is currently fixed. Whether FWHT specifically is necessary for the architecture's quality, or whether any orthogonal mixer of similar structure (or none at all, in the current learned-embedding context) achieves equivalent performance, has not been measured directly.

This is testable on WaveletLM via four ablation conditions:

1. **Identity (no transform)**: Baseline. Tests whether the FWHT slot is architecturally inert. If BPB matches the FWHT-baseline, the slot can be removed for compute savings. Previously catastrophic with the binary conceptual embedding, but potentially survivable now with the learned embedding.
2. **Hartley (DHT)**: Drop-in replacement; same self-inverse and real-valued properties as FWHT, but with sinusoidal (continuous) basis rather than Walsh (discontinuous). Tests whether the discontinuous nature of the Walsh basis matters.
3. **DCT-II / DCT-III pair**: `dct2 → mixer → dct3` round-trip. Continuous cosine basis with strong energy-compaction properties (provably optimal for Markov-1 signals). Slightly higher constant-factor compute than FWHT (~2-4×) but preserves O(n log n).
4. **Butterfly-parametrized learned orthogonal mixer**: Same butterfly cascade structure as FWHT, but each ±1 weight becomes a learned scalar with orthogonality constraints. Adds O(n log n) parameters and compute, and tests whether *any* learned orthogonal mixer matches or exceeds the fixed Walsh basis.

### Setup

1-epoch WT-103 runs at the current best-run config, single seed each. Around 2.5-3h on a 5090 per test, and 10-12h total across all four. Document each result in `runs.md` with BPB and quick generation samples for quality verification.

### Why this matters

WaveletLM names FWHT as one of its central features. Validating whether FWHT is optimal, or whether any orthogonal mixer of similar structure works equivalently, sharpens the architectural contribution:

- If condition 1 collapses, "wavelets + structured orthogonal channel rotation + gated SwiGLU" becomes the load-bearing claim. The orthogonal mixer is genuinely necessary.
- If condition 1 survives, "wavelets + per-position SwiGLU mixing - transformation" is enough with a learned embedding, and the architecture simplifies.
- If conditions 2-3 match condition 0 (the current FWHT), the orthogonal-mixing role is what matters, making the specific Walsh basis a convenient default, but not unique.
- If condition 4 substantially exceeds condition 0, fixed FWHT is leaving capability on the table, and a learned mixer is the upgrade path.

---

## Prioritization order for post-release

1. **Optimizer sweep (6)**: highest impact-per-compute; potential ~1.5–2× wall-clock speedup compounds across all subsequent ablations and the B200 scale-up. Run first.
2. **Combined parameter reduction (8)**: addresses the strongest public critique, enables longer-context experiments, single combined run is cheap. Run early.
3. **Cross-layer parameter sharing (9)**: builds directly on the parameter reduction; full-tying → MLP-only → PKM/FwPKM-only ablation series. Largest remaining parameter-efficiency lever.
4. **Cross-scale phase gating (3)**: cheapest to test, complements existing CSG.
5. **Stable parametrization validation (5)**: gates multiple latent-win runs and the B200 scale-up; small-scale sweep is cheap.
6. **Data-dependent lifting (1)**: largest uncertainty, largest potential payoff, biggest code lift. Start with single-block experiment at small C to calibrate before full sweep.
7. **Wavelet Packet Decomposition (2)**: dedicated research project; don't do simultaneously with (1) or the attribution becomes impossible.
8. **Per-scale mixer transform ablation (10)**: validates whether FWHT specifically is necessary. Cheap (50h total of 5090 time) and decisive — sharpens the architectural contribution regardless of outcome.
9. **Top-K Hadamard thresholding (4)**: pair with bit-packing as a deployment-optimization bundle.
10. **Inference strategies ablations**: self-explanatory.
