# Findings

Concrete observations from post-release research investigations. Each entry summarizes what has been learned while the parent plan may still be ongoing. See [`runs.md`](../runs.md) for the underlying experimental data.

---

## Single-Layer WaveletLM: equal-compute analysis

*Parent plan: [single_layer_waveletlm.md](single_layer_waveletlm.md)*
*Status: in progress (Run E in progress; dropout sweep pending)*
*Most recent: 2026-04-30*

### Headline finding

Under wall-clock-equalized comparison, **L=1 is regularization-bound, not capacity-bound**. At equal compute (L=1's full 5-epoch run vs L=2's matched-time checkpoint), L=1 fits the training data *more tightly* than L=2 but generalizes *worse* — pointing to under-regularization at the L=2-tuned recipe rather than insufficient architectural depth.

### Equal-step vs equal-compute comparison

The L=1 vs L=2 comparison gives different stories depending on the alignment axis:

**Equal step** (~step 65k, mid-epoch 2 of 5 for both runs):

| Model | Train | Val | Train/val gap |
|-------|-------|-----|---------------|
| L=1 (Run C) | 3.757 | 3.828 | 0.070 |
| L=2 (Run D) | 3.603 | 3.710 | 0.107 |
| Δ (L=1 − L=2) | **+0.154** | **+0.117** | -0.037 |

L=2 leads in both train and val; L=1's within-run gap is smaller. Reading: L=2 has more capacity per gradient step.

**Equal wall-clock** (~9.74h: end of L=1 E=5 vs L=2 at step 175k):

| Model | Step | Train | Val | Train/val gap |
|-------|------|-------|-----|---------------|
| L=1 final (Run C) | 292,000 | 2.837 | 3.344 (best 3.328) | **0.507** |
| L=2 at matched compute | 175,000 | 2.931 | 3.275 | 0.344 |
| Δ (L=1 − L=2) | — | **−0.094** | **+0.069** | +0.163 |

The story flips. L=1 fits training data *0.094 nats better* than L=2 at equal compute, but loses 0.069 nats in val loss — its train/val gap is **47% wider** than L=2's. Reading: L=1 has more *fitting capacity* per second of compute, but lacks enough regularization to translate that into generalization.

### Late-training delta widening

Both train-loss and val-loss deltas between the two runs widened over the course of training:

| Phase | Step (L=1 / L=2) | Δtrain | Δval |
|---|---|---|---|
| Mid epoch 2 | 65,000 / 65,000 | 0.154 | 0.117 |
| Late epoch 4 | 206,750 / 210,750 | 0.181 | 0.165 |

If L=1 had equal *asymptotic* capacity, the train-loss delta should narrow with more training (L=1 catching up as it has more steps). Instead, the delta grew — L=2 is fitting data faster per step *and* converting that fit into generalization more efficiently. This is consistent with the equal-compute reading: L=1 can fit data faster per second but cannot convert that fitting into val gains as cleanly as L=2 can.

### Final benchmark numbers (Run C)

| Run | Layers | Epochs | BPB sliding | PPL sliding | Params | Train time |
|-----|--------|--------|-------------|-------------|--------|------------|
| A | 1 | 1 | 1.1648 | 38.04 | 586.15M | ~1.5h |
| B | 2 | 1 | 1.1129 | 32.35 | 882.51M | ~3h |
| C | 1 | 5 | **1.0809** | **29.28** | 586.15M | 9.74h |
| D | 2 | 5 (baseline) | 1.0140 | 23.75 | 882.51M | 16.25h |

ΔBPB(C, D) = +0.0669 — outside the plan's 0.05 BPB threshold for "viable lightweight variant," but well within the conservative-mid projection range. Run C's best val was at epoch 4; epoch 5 was already overfitting.

### Reading: regularization bound, not capacity bound

Three lines of evidence converge on the same conclusion:

1. **Wider train/val gap at L=1 under equal compute** (0.507 vs 0.344) — overfitting signal, not under-fitting.
2. **L=1's training loss at equal compute is *lower* than L=2's** (2.837 vs 2.931) — fitting capacity is present.
3. **Run C's best val came at epoch 4, not epoch 5** — L=1 reaches a generalization floor and starts memorizing while L=2 keeps improving.

If the gap were capacity-driven, we'd expect L=1's training loss to be *higher* than L=2's at equal compute (insufficient expressive ceiling), and L=1's train/val gap to be *narrower* (no spare capacity to overfit with). We see the opposite of both.

### Why this matters

The original four-run plan framed the test as *"can L=1 close the val-loss gap to L=2 with more epochs?"* The answer from Run C is: **not at the L=2-tuned regularization recipe**, because more epochs at L=1 amplify overfitting rather than improving generalization. But the more interesting reframed question is: *"can L=1 close the val-loss gap with the right regularization recipe?"* — and the equal-compute analysis says this is plausible, since L=1 has spare fitting capacity that's currently being wasted on memorization.

This shifts the natural follow-up from "more epochs at L=1" to "tune dropout/WD at L=1." The latter is a clean one-axis sweep, while a capacity-bound result would have required architectural redesign.

### Implications for Run E (L=1, 8 epochs, in progress)

Run E is the empirical control for the regularization-bound hypothesis. Expected outcome:

- Train loss continues to drop (L=1 has fitting capacity left)
- Val loss plateaus or rises after epoch 4-5 (overfitting wall)
- Train/val gap widens further (>0.51)
- Final BPB sliding likely 1.07-1.09 — probably *worse* or roughly equal to Run C

If Run E confirms this pattern, the regularization-bound reading is empirically locked in.

### Implications for the dropout sweep follow-up

The current dropout values (`dropout_lm_head=0.24`, `dropout_mlp=0.10`, `dropout_mixer=0.10`, `dropout_projection=0.10`, `dropout_embedding=0.20`) were all tuned at L=2. The interactions matter:

- `dropout_mlp` controls the largest parameter bucket (167.8M/layer, ~76% of per-layer params). At L=1 there is one application instead of two, so this likely needs to roughly double to match cumulative dropout.
- `dropout_mixer` and `dropout_projection` similarly halve in cumulative effect at L=1.
- `dropout_lm_head` and `dropout_embedding` are mostly depth-independent and probably scale similarly.

A non-uniform sweep is more likely to win than a flat multiplier. Cleanest experimental order:

1. Wait for Run E to confirm or refute the regularization-bound reading.
2. One-at-a-time dropout sweep at L=1, E=5, varying each knob ±50% from L=2-tuned defaults. ~15 runs at ~10 min each at the eval intervals; ~2.5h total.
3. Combined best-dropout config + E=5. If this lands within 0.02 BPB of L=2 baseline, the lightweight-variant story is resurrected.
4. (Optional) E=8 at the tuned recipe — only if step 3 succeeds.

### Open questions

- Does the regularization-bound pattern survive at smaller L=1 configurations (parameter-reduction bundle applied)? Smaller models tend to be more regularization-bound, so the effect may be more pronounced — or may flip if the reduced model lacks raw capacity.
- Does the same pattern hold on PG-19, where the data:parameter ratio is much higher and overfitting is structurally less likely?
- What's the asymptotic capacity gap? The memorization probe (train both on a 1% subset until train loss flattens) would test this directly at low cost.

---
