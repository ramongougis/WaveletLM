# DenseNet-Style Recurrence (Depth-Weighted Averaging over Mixer Steps)

Generalize the mixer recurrence so each step's input is a **learned weighted
combination of all prior step outputs** (plus the original input), rather than
just `previous + input`. This is DenseFormer's depth-weighted averaging
(Pagliardini et al., 2024) applied to the recurrence depth axis — itself the
DenseNet (Huang et al., 2017) idea on a sequence model's depth.

Status: **plan only** — implement when higher-effort mode is enabled.

---

## Why (and the parameter-efficiency thesis)

The recurrence sweep showed **diversity (K>1, distinct banks) beats depth
(N>1, shared bank)** — but K>1 costs +58.85M params per extra bank. The
no-residual sweep also showed shared-bank depth *regresses* past N=2; the
input-anchored residual is testing whether anchoring rescues it.

Dense recurrence is the natural third axis: **richer routing over a shared
bank's trajectory, at ~zero parameter cost** (the routing weights are an
N·K × N·K lower-triangular matrix — tens of scalars). The headline question:

> Can dense routing over a *shared* bank (K=1, ~15 extra params at N=5)
> approach the quality of *distinct* banks (K=2, +58.85M params)?

If yes, that's a large parameter-efficiency win — diversity-like benefit from
routing instead of new weights — which is exactly the "greater parameter
efficiency and interpretability without hurting performance" goal.

Secondary thesis: dense routing may let **depth scale further** than the
input-anchored residual does (each step sees the whole trajectory, not just
step t−1 + X⁰), reopening N=10/20 if input-anchoring alone plateaus.

---

## Background: current recurrence

In `WaveletLMBlock.forward` ([model.py](../model.py), the `mixer_depth == 1`
branch), the loop keeps only the latest state `current_spec` and the initial
input `input_spec = X⁰`. Each of the M = N·K applications does (input-anchored,
post-fix):

```
step_spec = current_spec + X0          # (skip on the very first step)
Y         = mixer_{bank(t)}(step_spec)
current_spec = LN(step_spec + Y)       # LN skipped on the final step
```

So the current input combination is the restricted case "latest + initial".
Dense recurrence generalizes the `step_spec` line to a learned sum over **all**
stored states.

---

## Design

Index the M = N·K applications t = 0 … M−1. Maintain a list of states starting
with the input: `states = [X⁰]`; application t appends `X^(t+1)`.

**Per application t** (states currently holds X⁰ … X^(t), i.e. t+1 entries):

```
inp_t      = sum_{k=0}^{t} A[t, k] * states[k]      # dense weighted combination
Y_t        = mixer_{bank(t)}(inp_t)                 # gate routing also from inp_t
X^(t+1)    = inp_t + Y_t                            # residual (preserve current semantics)
X^(t+1)    = LN_step(X^(t+1))   if t < M-1          # existing inter-step norm, final step exempt
states.append(X^(t+1))
```

Output: `mixed_spec = states[-1]` (v1). Optional follow-up: a final dense
readout `mixed_spec = sum_k B[k] * states[k]`.

`A` is an **M × M lower-triangular** learnable matrix (row t has t+1 active
entries indexing states 0…t). Param count ≈ M(M+1)/2 — e.g. 15 at N=5 K=1,
210 at N=20. Negligible vs the 58.85M of one extra bank.

### Initialization — recover the current best as a special case

Init `A` to reproduce the input-anchored residual exactly, so dense starts from
known-good behavior and learns away from it:

```
A[t, t] = 1            for all t          # the latest state (= old current_spec)
A[t, 0] = 1            for t >= 1         # the initial input X0 (= old input_spec)
A[t, k] = 0            otherwise
```

At init this gives `inp_t = states[t] + states[0] = current_spec + X⁰` for t≥1
and `inp_0 = X⁰` — byte-identical to the post-fix input-anchored loop. Training
then opens up the other entries.

### Two weighting modes (config)

- **Raw (default):** unconstrained learned `A`, init as above. Strict
  generalization of the current recurrence; safest.
- **Normalized:** softmax each row of `A` (convex combination over states) —
  more interpretable (a distribution over depths) and naturally magnitude-bounded,
  but does *not* recover the additive input-anchored init exactly. Offer as a
  variant for the interpretability-max case.

---

## Config flags (all default off → identical to today)

```
mixer_recurrence_dense           : false   # master toggle
mixer_recurrence_dense_normalize : false   # softmax rows (convex) vs raw learned weights
```

When `mixer_recurrence_dense=false`, the loop is unchanged. When true, it
replaces the `step_spec` computation with the dense combination and stores the
state list. Composes with K>1 (dense forms the input; bank k(t) processes it),
with gate caching (gate computed from `inp_t`), and with the inter-step norm
(unchanged).

---

## Implementation sites

1. **`WaveletLMBlock.__init__`** — when `mixer_recurrence_dense`, allocate
   `self.recur_dense_A` as an M×M lower-triangular `nn.Parameter`, init per
   above (M = `mixer_recurrence_steps * mixer_recurrence_distinct_mixer_count`).
   Store `self.mixer_recurrence_dense` / `_normalize` flags. Guard M>1.
2. **`WaveletLMBlock.forward`** (recurrence loop) — replace the single
   `current_spec` carry with a `states` list; compute `inp_t` as the dense sum
   (apply softmax to the row first if normalized); feed `inp_t` to the mixer and
   gate routing; append the post-norm output. `mixed_spec = states[-1]`.
3. **Config plumbing** — pass both flags from `config.get(...)` where the other
   `mixer_recurrence_*` args are threaded into the block constructor.
4. **config.json** — add the two keys (default false).
5. **Interpretability readout** — log the (softmaxed, if normalized) `A` matrix
   at end of training (or expose a small accessor), so the depth-routing is
   inspectable: row t shows how much step t draws on X⁰ vs each prior step.

---

## Memory / compute

- The M intermediate states are **already in the autograd graph** during
  recurrence (needed for backprop through the chain), so keeping references for
  the dense sum adds little beyond the weighted-sum compute itself. Marginal.
- The dense sum is M small `[B,T,S,Cp]` axpy operations per step → O(M²)
  weighted adds total; trivial next to the mixer matmuls.
- No new large tensors; param add is ~M²/2 scalars.

---

## Ablation plan (after input-anchored sweep lands)

Rank by **BPB sliding** (per the val-vs-BPB metric note — val loss understates
context-exploiting configs); decision threshold ~0.0010 BPB above noise. Report
the learned `A` matrix for each.

| Run | N | K | dense | Tests |
|---|---|---|---|---|
| input-anchored N=5 K=1 (ref) | 5 | 1 | ✗ | the current best shared-bank depth |
| dense N=5 K=1 (raw) | 5 | 1 | ✓ | does trajectory routing beat latest+input at equal params? |
| dense N=5 K=1 (normalized) | 5 | 1 | ✓ softmax | interpretability-max variant; cost vs raw |
| dense N=10 K=1 | 10 | 1 | ✓ | does dense routing let depth scale where anchoring plateaus? |
| **dense N=5 K=1 vs no-dense K=2** | — | — | — | **param-efficiency headline**: ~15 params vs +58.85M |

The last comparison is the thesis test: if dense K=1 approaches K=2's BPB,
routing substitutes for parameters.

---

## Interpretability angle

Unlike additive residuals (which blend states irreversibly), the dense `A`
matrix is a **legible routing over recurrence depth** — directly readable:
- Does the model use the full trajectory, or collapse to "latest + input"
  (i.e. does `A` stay near its init)?
- How much weight on X⁰ (the anchor) vs intermediate refinements?
- Normalized mode gives a per-step distribution over depths — a clean
  interpretability artifact aligning with WaveletLM's legibility thesis.

This is the "interpretability retained or improved" property the project
favors: the mechanism adds a small, inspectable routing table rather than
opaque mixing.

---

## Risk / rollback

- Self-contained in the recurrence loop + 2 config flags; default off = byte-
  identical. Low–moderate risk (touches the hot recurrence path; verify the
  init reproduces input-anchored exactly with a smoke test).
- Composes with K>1, gate caching, inter-step norm — verify each in a tiny
  forward test (shapes + init-equivalence) before launching runs.
- Orthogonal to the decompose-bypass SSM/BPTT long-range work; no interaction.
