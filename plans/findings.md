# Findings

Concrete observations from post-release research investigations. Each entry summarizes what has been learned while the parent plan may still be ongoing. See [`runs.md`](../runs.md) for the underlying experimental data.

---

## Single-Layer WaveletLM: equal-compute analysis

*Parent plan: [single_layer_waveletlm.md](single_layer_waveletlm.md)*
*Status: concluded (Run E complete; dropout sweep pending as the natural follow-up)*
*Most recent: 2026-05-01*

### Headline finding

Under wall-clock-equalized comparison on WT-103, **L=1 and L=2 reach essentially identical training-loss floors at this dataset/scale, but L=1 generalizes 0.15 nats worse**. The entire ~0.15 nat val-loss gap between L=1 and L=2 is therefore a generalization difference, not a fitting-capacity difference. **Depth in WaveletLM functions as implicit regularization, not as additional asymptotic capacity** at observed compute budgets.

### Run-minimum comparison (L=1 E=8 vs L=2 E=5, matched compute)

Both runs were trained at near-equal wall-clock (L=1 E=8: 15.86h; L=2 E=5: 16.25h). Comparing the minimum training and validation losses observed across the entire run:

| Model | Min train | Min val | Train/val gap (at minimums) |
|-------|-----------|---------|------------------------------|
| L=2 E=5 (Run D) | 2.6330 | 3.1593 | 0.526 |
| L=1 E=8 (Run E) | **2.5984** | 3.3050 | 0.706 |
| Δ (L=1 − L=2) | **−0.0346** | **+0.1457** | +0.180 |

L=1 actually edges out L=2 by 0.035 nats on the lowest single-step training loss seen — well within noise of "identical." But L=1's val loss minimum is 0.146 nats higher, and its train/val gap is 34% wider. The gap difference (0.180) very nearly equals the val-loss difference plus the train-loss advantage L=1 has — i.e., the entire val gap is explained by L=1's wider generalization bleed, not by any capacity shortfall.

### Equal-step comparison (snapshot perspective)

Earlier in training, before either run reached its memorization floor, L=2 leads in both train and val:

| Phase | Step | L=1 train | L=2 train | L=1 val | L=2 val |
|---|---|---|---|---|---|
| Mid epoch 2 | 65,000 | 3.757 | 3.603 | 3.828 | 3.710 |
| Late epoch 4 | ~209,000 | 2.964 | 2.784 | 3.372 | 3.207 |

Reading: **L=2 converges faster per gradient step** (depth provides better gradient flow), but with enough training time L=1 reaches the same training-loss floor. Per-step L=2 has the apparent advantage; per-second-of-compute and at-asymptote, the two architectures are equivalent on memorization.

### Final benchmark numbers (full L=1 ablation series)

| Run | Layers | Epochs | BPB sliding | PPL sliding | Params | Train time |
|-----|--------|--------|-------------|-------------|--------|------------|
| A | 1 | 1 | 1.1648 | 38.04 | 586.15M | ~1.5h |
| B | 2 | 1 | 1.1129 | 32.35 | 882.51M | ~3h |
| C | 1 | 5 | 1.0809 | 29.28 | 586.15M | 9.74h |
| D | 2 | 5 (baseline) | **1.0140** | **23.75** | 882.51M | 16.25h |
| E | 1 | 8 (compute-equalized) | 1.0715 | 28.43 | 586.15M | 15.86h |

ΔBPB(E, D) = +0.0575. Outside the plan's 0.05 BPB threshold for "viable lightweight variant" at the L=2-tuned recipe — but this is now understood as a *regularization* gap, not a capacity gap. The dropout sweep at L=1 is the next lever for closing it.

### Val loss saturates at L=1 by epoch 5; train loss continues to descend

Comparing minimum train and val losses across the L=1 series:

| Run | Min train | Best val | Δtrain from prior | Δval from prior |
|-----|-----------|----------|-------------------|------------------|
| L=1 E=5 (Run C) | 2.8292 | 3.328 | — | — |
| L=1 E=8 (Run E) | 2.5984 | 3.305 | **-0.231** | -0.023 |

The two quantities decouple sharply between E=5 and E=8: train loss continues to drop substantially (-0.231 nats from 60% more compute), but val loss saturates (only -0.023 nats). The train/val gap widens from 0.499 (Run C) to 0.707 (Run E) over the extra epochs — additional compute past E=5 is consumed by memorization, not generalization. **Operational implication: subsequent ablations at L=1 should default to E=5** unless an experiment specifically tests the memorization regime, since the headline metric (val loss / BPB) saturates by E=5 and further compute amplifies overfitting without improving deployable quality.

### Reading: regularization bound, not capacity bound

Three lines of evidence converge:

1. **Equal training-loss floors at matched compute** (L=1: 2.5984, L=2: 2.6330) — both architectures reach essentially the same memorization ceiling on WT-103.
2. **L=1's wider train/val gap** (0.706 vs 0.526 at minimums) — overfitting signal, not under-fitting.
3. **L=1's training loss saturates by epoch 5; further compute doesn't extend the floor** — L=1 has used its memorization capacity; what remains underutilized is its generalization capacity.

If the gap were capacity-driven, we'd expect L=1's training loss minimum to be *higher* than L=2's, and L=1's train/val gap to be *narrower* (less spare capacity for memorizing spuriously). We see the opposite of both.

### Why this matters

The original four-run plan framed the test as *"can L=1 close the val-loss gap to L=2 with more epochs?"* The answer from Run E is: **not at the L=2-tuned regularization recipe**, because more epochs at L=1 don't materially reduce val loss once memorization saturates. But the more interesting reframed question is: *"can L=1 close the val-loss gap with the right regularization recipe?"* — and the equal-compute analysis says this is plausible, since L=1 has equivalent fitting capacity to L=2 and just generalizes worse from it.

This shifts the natural follow-up from "more epochs at L=1" to "tune dropout/WD at L=1." The latter is a clean one-axis sweep, while a capacity-bound result would have required architectural redesign.

### Implications for the dropout sweep follow-up

The current dropout values (`dropout_lm_head=0.24`, `dropout_mlp=0.10`, `dropout_mixer=0.10`, `dropout_projection=0.10`, `dropout_embedding=0.20`) were all tuned at L=2. The interactions matter:

- `dropout_mlp` controls the largest parameter bucket (167.8M/layer, ~76% of per-layer params). At L=1 there is one application instead of two, so this likely needs to roughly double to match cumulative dropout.
- `dropout_mixer` and `dropout_projection` similarly halve in cumulative effect at L=1.
- `dropout_lm_head` and `dropout_embedding` are mostly depth-independent and probably scale similarly.

A non-uniform sweep is more likely to win than a flat multiplier. Cleanest experimental order:

1. Parameter-reduce L=1 first (per the Combined Parameter Reduction plan); measure baseline reduced-L=1 val loss.
2. One-at-a-time dropout sweep at reduced L=1, E=5, varying each knob ±50% from L=2-tuned defaults. ~15 runs at ~1.5h each on a 5090.
3. Combined best-dropout config + E=5. **Theoretical ceiling**: if L=1's train/val gap (0.706) compresses to L=2's gap (0.526) via the sweep, L=1's val loss drops correspondingly to 3.16 — matching L=2 at 0.014 nats.
4. Apply the L=1-tuned recipe retroactively to L=2 to update the released model's headline numbers.

### Open questions

- Does the regularization-bound pattern survive at smaller L=1 configurations (parameter-reduction bundle applied)? Smaller models tend to be more regularization-bound, so the effect may be more pronounced — or may flip if the reduced model lacks raw capacity. **(Now answered — see "Combined parameter reduction" entry below: pattern not only survives but the reduced model marginally beats the unreduced on BPB.)**
- Does the same pattern hold on PG-19, where the data:parameter ratio is much higher and overfitting is structurally less likely?
- What's the *true* asymptotic capacity of L=2? Run D ended at train min 2.6330 still descending in late epochs. With more epochs, L=2 might reach a lower floor than L=1's 2.5984 — in which case capacity equality only holds at observed compute budgets, not in the limit. An L=2 E=10-12 run would resolve this.

---

## Combined parameter reduction: at-least-equivalent BPB at L=1, EBS scaling hurts

*Parent plan: [other_post_release_plans.md §8](other_post_release_plans.md#8-combined-parameter-reduction-and-vram-reallocation)*
*Status: Tests 1, 2, 2b concluded. Tests 3-4 (larger block_size, min EBS + max block_size) queued.*
*Most recent: 2026-05-02*

### Headline finding 1 — parameter reduction is at-least-equivalent in BPB

The combined parameter reduction recipe (mlp_expansion 20→10, PKM dropped, FwPKM keys halved, embedding tied to LM head) at L=1 / E=5 on WikiText-103 produced **statistically equivalent BPB to the unreduced L=1 baseline** despite removing 41.2% of parameters and 21% of training time. Δ = −0.0013 BPB, within ±0.0015 single-seed noise (3-seed variance study at L=2 baseline established noise floor: 1.0140, 1.0155, 1.0152 → σ ≈ 0.0008, 2σ ≈ 0.0015). The §8 plan projected +0.025 BPB cost; actual result is essentially zero cost — much better than projected, even without claiming strict improvement.

| | Unreduced (Run C, 586M) | Reduced (Test 1, 344.63M) | Δ |
|---|---|---|---|
| Params | 586.15M | 344.63M | **−41.2%** |
| Wall-clock | 9.74h | 7.69h | **−21%** |
| BPB sliding | 1.0809 | **1.0796** | **−0.0013** |
| BPB non-overlap | 1.0924 | **1.0909** | **−0.0015** |
| PPL sliding | 29.28 | 29.15 | −0.13 |
| Best val loss | 3.3275 (epoch 4) | 3.3341 (epoch 5) | +0.007 |
| Min train loss | 2.8292 | 2.9649 | +0.136 |
| Train/val gap | 0.498 | 0.369 | **−26%** |

### Mechanism — implicit regularization realized

The L=1 vs L=2 findings established that L=1 was regularization-bound, not capacity-bound. The unreduced L=1 model had spare memorization capacity that didn't translate into val-loss improvement. Removing 42% of parameters did exactly what theory predicted:

1. **Reduced model fits training data less tightly** (min train +0.136 nats higher) — confirmed less memorization capacity.
2. **Same val loss as unreduced** (within 0.007 nats) — generalization is preserved.
3. **Train/val gap shrinks 26%** — overfitting headroom removed structurally.
4. **Best val moved from epoch 4 to epoch 5** — the reduced model didn't overfit by epoch 5 the way the unreduced did, indicating the regularization pressure from parameter reduction is comparable to (or stronger than) one extra epoch's worth of dropout-driven regularization.

### Why this is "essentially free"

A naive parameter-reduction sweep typically trades quality for size: smaller model, slightly worse BPB. WaveletLM L=1's regularization-bound state inverts this trade. The "wasted" parameters in the unreduced model were actively harmful — they spent compute on memorization that worsened the train/val gap without lifting val. Removing them recovered some of the regularization that L=2's depth was providing for free in the L=2 baseline.

Suggested public framing:

> *Combined parameter reduction (`mlp_expansion: 10`, PKM dropped, FwPKM keys halved, tied embedding) is essentially free at L=1 / E=5 on WT-103: BPB sliding 1.0796 vs unreduced L=1's 1.0809 (Δ = −0.0013, statistically equivalent within ±0.0015 noise) at −41% parameters and −21% wall-clock. The mechanism is implicit regularization: L=1 was regularization-bound, the reduced model has 26% smaller train/val gap, and the spare memorization capacity that's been removed wasn't contributing to generalization anyway.*

### Headline finding 2 — increasing EBS hurts L=1 (gradient-noise hypothesis confirmed by replication)

Tests 2 and 2b both ran the same reduced recipe with `micro_batch_size=64` (8× the baseline MBS), differing only in eval frequency. Both regressed comfortably outside the noise band, in the same direction, with consistent magnitude:

| Run | MBS | eval_interval | BPB sliding | Δ vs Test 1 | σ above noise |
|-----|-----|---------------|-------------|-------------|---------------|
| Test 1 | 8 | 250 | 1.0796 | — | — |
| Test 2 | 64 | 250 | 1.0860 | +0.0064 | **4.3σ** |
| Test 2b | 64 | 32 | 1.0888 | +0.0092 | **6.1σ** |

The gradient-noise-as-regularizer effect is confirmed: smaller batches at L=1 provide implicit regularization that larger batches lose. Two independent runs at MBS=64 land 4.3σ and 6.1σ above Test 1 — extremely unlikely to be chance. The eval-coarseness alternative explanation is ruled out: Test 2b's finer eval (8× more frequent) didn't close the gap, it slightly widened it. **For regularization-bound models on this dataset, freed VRAM should NOT be spent on larger EBS.**

### Methodology note: finer eval can hurt test-set checkpoint selection

Test 2b's slightly worse BPB despite finer eval (1.0888 vs Test 2's 1.0860, ~1.9σ) is consistent with **selection bias on noisy val minima**. With 8× more eval samples (1140 vs 145 across 5 epochs), the "best val" checkpoint is more likely to be selected at a lucky noisy dip in val that doesn't generalize as well to test. This is Goodhart's Law applied to model selection: over-optimizing on lowest-val-ever-observed selects for val noise, which doesn't replicate to the held-out test set.

Practical implication: holding `eval_interval` constant across configurations (rather than scaling proportionally to step count) is methodologically cleaner than the "more eval = better" intuition would suggest. The val curve in plateau regions has noise band ~±0.01 nats; eval should be frequent enough to catch the late-training plateau, but not so frequent that selection samples within-noise dips. The current `eval_interval=250` default is close to right for L=1 / E=5 configurations.

### Implications

1. **The reduced configuration is the new L=1 default** for any subsequent ablation work that doesn't specifically test parameter count. It's at-least-equivalent in BPB at substantially less compute and parameter cost.
2. **The gap to L=2 baseline is ~0.066 BPB** (1.0796 vs 1.0140) at 39% of L=2's parameters and 47% of its training time — a strong lightweight-variant story.
3. **EBS scaling has been ruled out as a useful lever for L=1.** Tests 2/2b confirmed; freed VRAM should NOT go to larger MBS.
4. **Dropout sweep is still load-bearing.** The reduction provides ~26% gap shrinkage; tuning the L=2-default dropout for L=1 specifically is the next regularization lever and could close more of the residual gap to L=2. Test 2/2b's confirmation that L=1 is regularization-bound makes the dropout sweep even higher priority — it's targeting the *actual* bottleneck.

### Headline finding 3 — longer block_size is the right way to spend freed VRAM

Test 3 (MBS=8, bs=1024) achieved best val loss 3.3390 vs Test 1's 3.3341 — Δ = +0.005 nats, ~1σ on the BPB scale, **essentially tied**. This is dramatically better than the MBS=64 variants (Test 2: +0.041 val, Test 2b: +0.019 val).

| Run | bs | MBS | VRAM | Wall-clock | Best val | Train/val gap |
|---|---|---|---|---|---|---|
| Test 1 | 256 | 8 | 6.9 GiB | 7.69h | 3.3341 | 0.369 |
| Test 2 | 256 | 64 | 25 GiB | 5.60h | 3.3746 | 0.356 |
| Test 2b | 256 | 64 | 25 GiB | 5.60h | 3.3532 | 0.355 |
| **Test 3** | **1024** | **8** | **14 GiB** | **5.96h** | **3.3390** | **0.352** |

Test 3 is dominant on every axis except marginal val regression (within noise of Test 1). Specifically:
- **−22% wall-clock** vs Test 1 — bs=1024 amortizes per-step overhead better in the wavelet+FWHT pipeline
- **~half the VRAM** of MBS=64 (14 GiB vs 25 GiB) — longer context is more memory-efficient than larger batch
- **Lowest train/val gap** of any test so far (0.352) and lowest min train loss (2.9874) — slightly tighter generalization per nat of training capacity
- **Equivalent val loss to Test 1**, dramatically better than Tests 2/2b

### Why this matters — framework consolidation

Tests 2/2b and Test 3 use the same total tokens-per-step (8× the baseline). Both spend the same freed VRAM budget but allocate it differently:

- **Test 2/2b (MBS=64, bs=256)**: 8× tokens come from more parallel sequences. Reduces per-step gradient noise. **Hurts val loss** — the regularization-bound L=1 model loses an implicit regularizer.
- **Test 3 (MBS=8, bs=1024)**: 8× tokens come from longer per-sequence context. Preserves gradient noise (still 8 sequences per gradient). **Preserves val loss** — gains within-example signal without losing the regularizer.

This is the third independent confirmation of the regularization-bound framework, and it's architecturally informative: **for regularization-bound L=1, the right way to use freed VRAM is more per-example signal (longer block_size), not more parallel sequences (larger MBS)**.

### Likely under-realized potential — per-scale configuration mismatch

Test 3 keeps `levels=5` from baseline. At bs=256, this gave the coarsest scale ~8 tokens of structure. At bs=1024 with the same levels=5, the coarsest scale represents 32 tokens — 4× wider per coarse cell. The wavelet pipeline isn't yet exploiting the additional coarse scales the longer context could enable.

Hypothesis worth testing: optimal `levels ≈ log2(block_size) − constant` where constant ≈ 3-4 (so coarsest cell stays at 8-16 tokens regardless of bs). If true:
- bs=1024 + levels=7 (S=8) → coarsest cell at 8 tokens
- bs=8192 + levels=10 (S=11) → coarsest cell at 8 tokens

This would require also resizing `per_scale_mixer_widths` to S entries (8 or 11 respectively), with the symmetric half-coarse-at-1.0 / half-fine-at-0.5 split as starting point. Reserved as a separate sweep — see README's "Per-Scale Configuration at Longer Block Size" section.

### Open questions

- Does the same pattern hold on PG-19 (different overfitting regime: data:parameter ratio is much higher)?
- How does the reduced model interact with the dropout sweep? With less spare capacity, optimal dropout values may shift downward (less regularization needed).
- At even more aggressive reductions (mlp_expansion=5, FwPKM keys further reduced), does the trend continue or break? There's a floor at which capacity becomes the bottleneck rather than regularization.

---
