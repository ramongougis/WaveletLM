# Post-release: data-dependent EMA decompose-bypass

## Status

**Rejected for release.** Config flag preserved (`decompose_bypass_ema`), implementation preserved at [model.py:839](../model.py#L839) (`_compute_data_dependent_ema`) and [model.py:1038](../model.py#L1038) (block wiring). Not adopted because of a 1-epoch → 5-epoch performance inversion.

## Observed behavior

| Run | Config | BPB (sliding) | PPL (sliding) | Val loss (epoch 1) | Folder |
|-----|--------|---------------|---------------|--------------------|--------|
| 1-epoch smoke test | `decompose_bypass_ema=true`, 1 epoch | 1.1102 | 32.07 | 3.4580 | `logs/wikitext-103_2026-04-21_22-05-15/` |
| 5-epoch full run | `decompose_bypass_ema=true`, 5 epochs | 1.0226 | 24.40 | - | `logs/wikitext-103_2026-04-22_18-42-22/` |
| 5-epoch baseline (no EMA) | cumulative running mean | **1.0201** | **24.21** | - | `logs/wikitext-103_2026-04-19_13-16-24/` |
| 5-epoch current best (no EMA, WD=1e-6) | cumulative running mean | **1.0140** | **23.75** | 3.1593 | `logs/wikitext-103_2026-04-22_01-36-47/` |

**The inversion.** At 1 epoch, EMA beat the no-EMA reference by 0.30 nats of val loss. At 5 epochs it regressed by +0.0025 BPB / +0.19 PPL against that same architecture's 5-epoch baseline, and is now +0.0086 BPB / +0.65 PPL behind the current best (WD=1e-6). Whatever mechanism helped at 1 epoch either inverted or was dominated by a competing regression during the remaining 4 epochs.

## Why we think it regressed (design gaps, not bugs)

The Hillis-Steele scan is computationally correct. The four design issues below all get *worse* as training proceeds, which is the shape of failure mode we'd expect given the 1→5 inversion.

**1. No bias correction on the EMA startup transient.**
The recurrence is `μ_t = α_t · μ_{t-1} + (1-α_t) · x_t` with implicit `μ_{-1} = 0`. At init (α=0.5 uniform), position 0 outputs `0.5·x_0` vs the cumulative mean's `x_0`. Downstream `history_gains` absorb the scale at init, but as the gate learns higher α (selective retention), the position-0 output shrinks toward 0, and prior gain calibration no longer matches. Adam-style bias correction (`μ_t / (1 - ∏α)`) would remove this, but isn't applied.

**2. The learnable gate is ~8.4M additional parameters on a data-saturated setting.**
At L=2, C=2048, `ema_gate = nn.Linear(C, C, bias=True)` adds ~4.2M params per block (~8.4M total). At 1 epoch, these are nearly frozen at init; the observed benefit is almost entirely structural (fixed α=0.5 → exponentially-weighted history vs equal-weighted). At 5 epochs, the gate IS training, but (a) backprop through a log-depth scan produces noisy gradients, (b) WikiText-103 at this model size is data-saturated (~5× passes through 120M tokens), (c) the gate has no direct supervision signal. It's optimized purely through the downstream loss gradient. Plausible outcome: the learned gate settles on something *worse* than the fixed α=0.5 prior.

**3. Cross-window state not seeded into the EMA scan.**
The `_compute_data_dependent_ema` function signature accepts an optional `prev_ema` argument ([model.py:842](../model.py#L842)), but the callsite at [model.py:1203](../model.py#L1203) never passes it. Every new window starts the EMA scan from `μ = 0`. Cross-window context only enters through the additive `mixed_context = current_running_mean + self.cross_layer_mix(prev_state)` term, which bypasses the EMA's own recurrent state. The cumulative running mean has the same structural issue, but its impact is bounded (position 0 is always `x_0` regardless). EMA's position 0 varies with learned α and gets worse as α saturates.

**4. fp16 underflow in `(1 - alphas)` at saturated α.**
At [model.py:871](../model.py#L871), `(1.0 - alphas) * x` is computed in fp16 before being cast to fp32. For logits above ~15, σ pins to 1.0 in fp16, making `1 - α` exactly 0. The scan then contributes nothing from tokens at saturated α. Minor effect, easy fix: promote before the subtraction.

## Why a more fundamental redesign may be needed

The four gaps above are tactical: each has a concrete fix, and stacked together they might recover the 1-epoch signal at 5 epochs. But they may not be sufficient, and there are structural reasons to think the current formulation is architecturally off:

**A. The bypass path has no clear role to play.**
The cumulative running mean's job is to give each token a cheap summary of "all prior content." It's parameter-free, monotonic, and the downstream per-scale `history_gains` learn how to use it. Replacing it with a *learnable* adaptive filter conflates "what summary to carry forward" with "what to forget" into a single 8.4M-parameter module. These might want to be separate modules, or the bypass path might want to stay parameter-free and any adaptive forgetting should happen elsewhere (e.g. inside the per-scale mixer's spectral coefficients).

**B. The EMA is local, but the bypass is supposed to be global.**
EMA exponentially decays older context with a timescale on the order of `1/(1-α)` tokens. For α=0.9 that's a 10-token memory; for α=0.99, 100 tokens. The cumulative running mean, by contrast, weights all prior tokens equally. "Selective forgetting at topical/clausal boundaries" is arguably the job of *attention*, or a state-space model, not a decompose-bypass. The bypass may want to stay a long-horizon summary statistic, and short-horizon adaptive behavior may belong to a different architectural slot.

**C. The scan gets harder to optimize at larger T.**
Backprop through a length-T scan with data-dependent gates is notoriously fragile (see Mamba's selective SSM stability work). At T=256 it's tractable; at T=1024+ (the B200-scale roadmap) it may degrade further. The cumulative mean's `torch.cumsum` has trivial gradients; the scan does not.

**D. Alternative formulations to consider.**
- **Fixed-α EMA with no learnable gate.** Preserves the 1-epoch structural benefit, removes the optimization problem. Compare vs α ∈ {0.5, 0.9, 0.99}.
- **Decaying running sum instead of mean.** `s_t = α · s_{t-1} + x_t`, divided by a normalizer at readout. Separates state from magnitude.
- **Two-term bypass: global mean + local recency.** Keep the cumulative mean AND add a second bypass path (fixed-α EMA or short convolution) that carries recency. Downstream mixer decides weighting.
- **Keep the cumulative mean; introduce adaptive forgetting elsewhere.** E.g. a forget gate on the `cross_layer_mix` term, not on the bypass itself. Smaller surface area, less optimization risk.
- **Mamba-style selective state-space block as a bypass.** Heavier, but proven to train. Probably overkill for a "bypass" but worth a sketch.

## Investigation plan (post-release)

Ordered cheapest → most involved.

**Phase 1: diagnose the 1→5 inversion.** ~40h compute. Answers: is it the learned gate, the bias transient, or both?

1. **Freeze-gate 5-epoch run.** `ema_gate.weight = 0, bias = 0` frozen. Uses fixed-α=0.5 EMA with no learned adaptation. ~17h. If this recovers the 1-epoch benefit at 5 epochs → hypothesis #2 (gate overfitting) is the primary cause.
2. **Bias-corrected EMA 5-epoch run.** Same learned gate, but output divided by `1 - ∏α` (or equivalent). ~17h. If this recovers the 1-epoch benefit → hypothesis #1 (startup transient) is the primary cause.
3. **Both at once.** Only if neither #1 nor #2 alone is sufficient. ~17h.

**Phase 2: alternative formulations.** ~50–70h compute, only if Phase 1 identifies a promising direction.

4. **Fixed-α sweep at α ∈ {0.5, 0.9, 0.99} if freeze-gate worked.** Three 5-epoch runs to characterize the benefit curve.
5. **Two-term bypass: cumulative mean + fixed-α EMA.** 5-epoch run. Tests whether the benefit is additive vs replacing.
6. **Adaptive forgetting on `cross_layer_mix` instead of the bypass.** 5-epoch run. Tests hypothesis (D.4), moving the selection from bypass to cross-window carry.

**Phase 3: larger-scope redesign.** Only if Phases 1+2 don't yield a release-quality configuration.

7. **Selective-SSM bypass à la Mamba.** Probably a separate research project, not a bypass variant. Scoped independently.

## Definition of done

- Either: a bypass variant that beats BPB 1.0140 by ≥0.005 on 5-epoch WT103 at the same compute cost, reproducibly across 3 seeds. Adopt and release as v2.
- Or: a defensible write-up of "we tried X, Y, Z; fundamentally different regime is needed," which feeds into the scaled-up B200 roadmap. Don't re-try EMA at larger scale without first understanding why the 1→5 inversion happened.

## Related fixes (apply regardless of Phase outcomes)

- **fp16 underflow in `(1 - alphas)`.** Promote to fp32 before subtraction: `B_out = (1.0 - alphas.float()) * x.float()`. Trivial.
- **Pass `prev_ema` at the callsite** if cross-window EMA is ever revisited. Pipe `prev_state` or a dedicated persistent EMA state into `_compute_data_dependent_ema`.
- **Document the 1→5 inversion in the `decompose_bypass_ema` docstring.** Anyone encountering the flag should see the result and this plan linked, so they don't burn compute rediscovering the same failure.

## Open questions

1. Is the 1-epoch benefit a genuine structural advantage of exponentially-weighted history, or an artifact of how far along training is at epoch 1 (e.g. the network hasn't yet calibrated the `history_gains` to the cumulative mean's magnitude, so *any* alternative summary with different magnitude looks better transiently)?
2. Does the regression reproduce on a smaller-scale sanity config (e.g. C=512, L=2, 1-epoch vs 5-epoch) so iteration cost is 1–2 hours instead of 17? Worth trying before burning full-scale compute.
3. Is the right comparison actually EMA vs *no bypass at all*? If EMA hurts AND the cumulative mean helps, that's informative. If EMA hurts but removing the bypass entirely hurts more, that's also informative.
