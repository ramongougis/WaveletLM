# True Feedback Mechanisms: Performance-First Round

## Summary

EXARCH currently has no true feedback. The mechanism formerly called "semantic
feedback" (now **Cumulative Mean Bias / CMB**) is forward-propagated state, not
feedback — layer N's running mean biases layer N+1, never the other way around.

This plan introduces three optional feedback mechanisms, each implementable as a
config flag for clean Boolean ablations. They are intended to run **after** the
[wavelet & mixer augmentations](wavelet_and_mixer_augmentations.md) round, since
those are cheaper changes with more predictable wins, and the best CMB+mixer
combo will define the baseline these feedback mechanisms test against.

## Status: Implementation candidate (post-wavelet/mixer round)

All features designed to be:
- **Optional** (config flag, false = today's behavior)
- **Independently togglable** (can be ablated one at a time)
- **Stackable in principle** (compatible at config level; some may interact)
- **Backward-compatible** (existing checkpoints load unchanged when flag=false)

---

## Feature 1: Cross-time recurrence (causal top-down)

### What

At position t, layer N reads layer N+1's output at position t−1. This preserves
causality (only past timesteps inform the present) and creates a genuine
top-down information flow without breaking autoregressive structure.

```
        position t-1        position t
         ↓                   ↓
layer N: ──────► h_{N,t-1}  ──────► h_{N,t}
         ↑                   ↑
         │                   │ (read h_{N+1,t-1})
         │                   │
layer N+1: ───► h_{N+1,t-1} ──────► h_{N+1,t}
```

### Cost

Storage: one hidden state per layer per position (already kept during training).
Compute: a per-layer Linear projection of the shifted-by-one upper-layer state,
added as a bias before the mixer. ≈ C² per layer = ~16M params at L=2/C=2048.

### Training implementation

Two-pass training is the cleanest correct implementation:
1. **Pass 1:** standard forward, collect `h_{N+1}` for all layers and positions.
2. **Pass 2:** forward again, but each layer N receives `shift_right(h_{N+1})`
   from pass 1 as an additive bias.
3. Backprop only through pass 2 (pass 1 acts as a fixed feedback context).

Doubles forward cost during training. Inference is single-pass: when
generating token t autoregressively, all layers at positions <t have been fully
computed and cached, so the cross-time read is free.

Cheap alternative: use **previous training step's** stored layer outputs (stale
feedback). Skips the second forward pass at the cost of slightly stale signal.

### Config

```json
"cross_time_feedback": false,
"cross_time_feedback_mode": "two_pass"  // or "stale"
```

### Why this might help

This is the only one of the three that introduces feedback **within the same
forward pass** (in the inference sense) without changing depth or compute
structure. It mirrors how cortical feedback connections operate: layer N+1's
slightly-delayed state shapes layer N's current processing.

---

## Feature 2: Iterative refinement (predictive coding)

### What

Run the full network once to get a final hidden state `h_final`. Then run it
again, with `h_final` injected as a priming bias at layer 0 (and optionally at
each layer). The second pass "knows" what the first pass concluded and can
revise its understanding accordingly.

```
Pass 1: x → block_1 → block_2 → ... → block_L → h_final
Pass 2: x + α·proj(h_final) → block_1 → block_2 → ... → block_L → h_final_2
```

Loss can be computed on either the final pass output (cheaper) or as a sum
across all passes (predictive-coding-style refinement objective).

### Cost

K-pass: K× forward compute. With K=2: 2× training time, 2× inference time.
Param overhead: a single Linear `proj` per layer (or shared) = ~4M at L=2/C=2048.

### Config

```json
"iterative_refinement": false,
"iterative_refinement_passes": 2,
"iterative_refinement_loss": "final"  // or "all_passes"
```

### Implementation

Wrap the existing forward pass in a loop:

```python
priming = None
for k in range(passes):
    h = embed(x)
    if priming is not None:
        h = h + self.priming_proj(priming)
    for block in self.blocks:
        h, _ = block(h, ...)
    priming = h.detach() if k < passes - 1 else h
logits = self.lm_head(self.final_ln(h))
```

`detach()` between passes keeps gradient cost manageable; only the final pass
contributes to backprop in the "final" loss mode.

### Why this might help

The first pass produces a coarse global summary of the input. The second pass
can use that summary to disambiguate locally — exactly the function predictive
coding plays in cortex. EXARCH's depth-vs-width experiments showed depth past
~20 layers hurts; iterative refinement adds *effective depth* without adding
parameters, by reusing the same blocks.

---

## Feature 3: Looped blocks (Universal Transformer style)

### What

Apply the same ExarchBlock K times instead of stacking K distinct blocks. State
flows through the loop with optional per-iteration positional/depth signals.
Different from Feature 2 in that the loop is **per-block**, not whole-network.

```python
# Standard:
for block in self.blocks:  # L distinct blocks
    h = block(h)

# Looped:
for _ in range(K):
    h = self.shared_block(h)  # 1 block, K iterations
```

Could also be a hybrid: some layers looped, some not.

### Cost

Param savings: L=K=8 looped from one block uses 1/8 the params of 8 distinct
blocks. Compute is unchanged (still K block evaluations).

### Config

```json
"looped_blocks": false,
"looped_blocks_count": 8,
"looped_blocks_share_lifting": true,
"looped_blocks_share_mixer": true,
"looped_blocks_share_mlp": true
```

Granular sharing flags let us ablate "share everything" vs. "share only the
expensive parts" (e.g., share mixer but keep per-iteration MLP).

### Implementation

Replace the `nn.ModuleList` of blocks with a single block + loop:

```python
if looped_blocks:
    self.shared_block = ExarchBlock(...)
    self.loop_count = looped_blocks_count
else:
    self.blocks = nn.ModuleList([ExarchBlock(...) for _ in range(layers)])

# Forward:
if looped_blocks:
    prev_state = None
    for k in range(self.loop_count):
        h, prev_state = self.shared_block(h, prev_state, ...)
else:
    prev_state = None
    for block in self.blocks:
        h, prev_state = block(h, prev_state, ...)
```

CMB carries naturally between iterations via `prev_state`.

### Why this might help

The 30L vs 20L result (BPB 1.0207 vs 1.0136 — depth hurts past 20) suggests
that adding *parameter count* via depth has diminishing returns, but the
**computation** of repeated block application might still help. Looped blocks
let us test "more compute, same parameters" cleanly. If looped blocks at K=8
match or beat 8 distinct blocks, that's a major capacity-efficiency win.

---

## Suggested ablation order

1. **Looped blocks** — safest first test (no extra compute over baseline at
   matched K, just shares params). If it matches a stacked equivalent, confirms
   that depth-as-compute matters more than depth-as-parameters.
2. **Iterative refinement, K=2, final-pass loss** — moderate cost, clean test
   of "does a second informed pass help."
3. **Cross-time feedback, stale mode** — cheap, confirms whether top-down
   signal helps at all before committing to two-pass training.
4. **Cross-time feedback, two-pass mode** — only if stale mode showed signal.
5. **Iterative refinement, K=3 or all-passes loss** — only if K=2 helped.
6. **Combinations** — e.g., looped blocks + cross-time feedback could be the
   "EXARCH-Recurrent" variant.

## Implementation effort estimate

- Looped blocks: ~2 hours (refactor block storage, wire up sharing flags)
- Iterative refinement: ~3 hours (forward-pass loop, priming projection, loss modes)
- Cross-time feedback (stale mode): ~3 hours (per-layer state buffer, projection)
- Cross-time feedback (two-pass mode): ~5 hours (training-loop changes, gradient handling)

Total ~half a day for stale modes, ~1.5 days including two-pass cross-time.

## Open questions

- **Does CMB become redundant under any of these?** Iterative refinement and
  looped blocks both already propagate context across passes; CMB might be
  subsumed. Worth ablating CMB-off when these are on.
- **Inference cost vs. quality:** iterative refinement at K=2 doubles inference
  time. Acceptable for benchmark/quality runs, painful for serving. Worth a
  separate "inference-mode K=1, training-mode K=2" experiment.
- **Convergence under looped blocks:** Universal Transformers needed adaptive
  computation time to converge well. We may need a similar mechanism if K is
  large.
