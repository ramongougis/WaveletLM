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

- Does the regularization-bound pattern survive at smaller L=1 configurations (parameter-reduction bundle applied)? Smaller models tend to be more regularization-bound, so the effect may be more pronounced — or may flip if the reduced model lacks raw capacity.
- Does the same pattern hold on PG-19, where the data:parameter ratio is much higher and overfitting is structurally less likely?
- What's the *true* asymptotic capacity of L=2? Run D ended at train min 2.6330 still descending in late epochs. With more epochs, L=2 might reach a lower floor than L=1's 2.5984 — in which case capacity equality only holds at observed compute budgets, not in the limit. An L=2 E=10-12 run would resolve this.

---
