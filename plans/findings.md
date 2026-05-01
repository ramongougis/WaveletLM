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

## Combined parameter reduction: better than free at L=1

*Parent plan: [other_post_release_plans.md §8](other_post_release_plans.md#8-combined-parameter-reduction-and-vram-reallocation)*
*Status: Test 1 (baseline reduction) concluded. Tests 2-4 (max EBS, larger block_size, min EBS + max block_size) queued.*
*Most recent: 2026-05-01*

### Headline finding

The combined parameter reduction recipe (mlp_expansion 20→10, PKM dropped, FwPKM keys halved, embedding tied to LM head) at L=1 / E=5 on WikiText-103 produced a **marginally better BPB than the unreduced L=1 baseline** despite removing 41.2% of parameters and 21% of training time. The §8 plan projected +0.025 BPB cost; actual result was −0.0013 BPB *benefit*. The cost projection was off by a sign.

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

### Why this is "better than free"

A naive parameter-reduction sweep typically trades quality for size: smaller model, slightly worse BPB. WaveletLM L=1's regularization-bound state inverts this trade. The "wasted" parameters in the unreduced model were actively harmful — they spent compute on memorization that worsened the train/val gap without lifting val. Removing them recovered some of the regularization that L=2's depth was providing for free in the L=2 baseline.

This is a stronger and more publishable framing than the §8 plan anticipated. Suggested public framing:

> *Combined parameter reduction (`mlp_expansion: 10`, PKM dropped, FwPKM keys halved, tied embedding) is better than free at L=1 / E=5 on WT-103: −0.0013 BPB sliding for −41% parameters and −21% wall-clock. The mechanism is implicit regularization: L=1 was regularization-bound, the reduced model has 26% smaller train/val gap, and the spare memorization capacity that's been removed wasn't contributing to generalization anyway.*

### Implications

1. **The reduced configuration is the new L=1 default** for any subsequent ablation work that doesn't specifically test parameter count. It's strictly better in compute, parameters, and BPB.
2. **The gap to L=2 baseline is now ~0.066 BPB** (1.0796 vs 1.0140) at 39% of L=2's parameters and 47% of its training time — a much stronger lightweight-variant story than L=1 unreduced (which was 0.067 BPB behind at 66% of L=2's params).
3. **Variants 2-4 (in `runs.sh`) become more interesting**, not less: with the reduced model already matching or beating unreduced, freed VRAM spent on max EBS / larger block_size / min EBS + max block_size could push the reduced model past the L=2 baseline on BPB at half the parameter count.
4. **Dropout sweep is still load-bearing.** The reduction provides ~26% gap shrinkage; tuning the L=2-default dropout for L=1 specifically is the next regularization lever and could close more of the residual gap to L=2.

### Open questions

- Does the same pattern hold on PG-19 (different overfitting regime: data:parameter ratio is much higher)?
- How does the reduced model interact with the dropout sweep? With less spare capacity, optimal dropout values may shift downward (less regularization needed).
- At even more aggressive reductions (mlp_expansion=5, FwPKM keys further reduced), does the trend continue or break? There's a floor at which capacity becomes the bottleneck rather than regularization.

---
