# Long-Range Context: Multi-Pole SSM + Truncated BPTT

Two attention-free upgrades targeting **cross-window** long-range dependency —
the architecture's thin spot. Within a 256-token block the lifting wavelet
already couples tokens ~128 apart (multi-scale, O(n log n)); beyond the block,
the only carrier is `_persistent_semantic_state`, which today is (a) a
first-moment causal mean, (b) `.detach()`ed (untrained across windows), (c) a
single per-channel vector. Both upgrades attack exactly those weaknesses.

Tested right after the recurrence section. Ablation order: **#1 vs T4**, then
**#2 vs T4**, then **#1+#2 together**.

---

## Background: where long-range flows today

- `WaveletLMBlock.forward` ([model.py](../model.py)) computes a context summary
  in the `decompose_bypass` path: either `_compute_running_mean(x)` (causal
  cumulative mean) or, if `decompose_bypass_ema`, `_compute_data_dependent_ema`
  (a **single-pole** data-dependent linear recurrence via a Blelloch
  associative scan). That summary is injected as a per-scale gate bias before
  the FWHT mixer.
- Across windows, `decompose_bypass_cross_window` carries
  `current_state[:, -1, :].detach()` into the next forward as the initial
  context. **The detach means no gradient ever crosses a window boundary** —
  the model is never trained to write useful long-range info into that state.
- Only meaningful with `sequential_blocks=true` (consecutive corpus order).
  Under random batching, adjacent windows are unrelated, so cross-window state
  (and BPTT through it) is noise.

---

## Feature #1 — Multi-pole diagonal SSM (`decompose_bypass_ssm`)

Generalize the single EMA into a **bank of P diagonal poles** with learned
per-channel decay rates — i.e. P parallel EMAs spanning short→long timescales,
combined per channel. This is the S4D/S5-style diagonal state-space primitive;
the existing `_compute_data_dependent_ema` is the P=1, data-dependent special
case.

### Math

Per channel `c`, pole `p` (P poles total), over block input `x_t ∈ R^C`:

```
a_{c,p}   = exp(-softplus(theta_{c,p}))           # learned decay in (0,1), time-invariant
h_{t,c,p} = a_{c,p} * h_{t-1,c,p} + (1 - a_{c,p}) * x_{t,c}   # normalized EMA (unit DC gain)
o_{t,c}   = sum_p G_{c,p} * h_{t,c,p}             # learned per-channel pole mix
```

- `o ∈ R^{B,T,C}` replaces `current_running_mean` in the bypass path.
- Parameters: `theta ∈ R^{C×P}` (decays), `G ∈ R^{C×P}` (output mix). Tiny:
  2·C·P ≈ 16K at C=2048, P=4.
- **Time-invariant decays** (not data-dependent) for the first version: cheaper,
  lower-risk, and already a strict generalization of the running mean. (At
  `a→1, G=1/P` it *is* a slow EMA ≈ running mean — so init can recover the
  baseline.) Data-dependent (Mamba-style selective) decays are a noted
  follow-up, not in this pass.

### Initialization (recover-baseline + span timescales)

- `theta` init so the P decays span e.g. `{0.5, 0.9, 0.99, 0.999}` for P=4
  (invert: `theta = softplus^{-1}(-log a)`). Short pole ≈ local, long pole ≈
  whole-context memory.
- `G` init to `1/P` (uniform) so the initial output is the average of the pole
  states — close to the existing multi-scale-mean behavior; the model learns to
  re-weight from there.

### Scan

Reuse the log-depth associative scan structure from
`_compute_data_dependent_ema`, batched over the pole axis P. Decays are
time-invariant here, so the scan composition is `(a,b)` with constant `a` per
(c,p) — the existing machinery handles time-varying `a` as the general case, so
this is a simplification. Keep fp32 accumulation (same overflow rationale as the
existing scan). Tensor shape during scan: `[B,T,C,P]` fp32 ≈ 67 MB at
B=8,T=256,C=2048,P=4 — fine.

### Cross-window carry

Two separately-ablatable modes (config `decompose_bypass_ssm_cross_window`):

**Within-window (default, x-window off):** the SSM runs over the block's T
positions seeded from zero; cross-window memory rides the existing `prev_state`
/ `cross_layer_mix` → `_persistent_semantic_state` [B,C] path. A drop-in upgrade
of the within-window summary; the [B,C] carry is what #2 (BPTT) trains.

**Cross-window (x-window on, implemented):** the block carries its final pole
state `h_{T-1} ∈ [B,C,P]` across block boundaries via a **block-local**
attribute `_ssm_carry` (no layer-return-signature change → low blast radius),
seeding the next window's scan so the long poles integrate beyond 256 tokens.
**Forward-only in v1**: the carry is detached each window, so it propagates
multi-timescale memory but is not BPTT-trained through the pole state. The
shared θ/G are trained within-window; the carry transmits their states forward.
Reset via `reset_semantic_state` (clears every block's `_ssm_carry`).
Constraints: requires sequential batching to be meaningful, and gradient
checkpointing OFF (the in-forward carry mutation isn't checkpoint-recompute
safe; T4 has it off).

**Follow-up:** BPTT *through* the pole-state carry (retain its graph across the
span like the [B,C] state), and/or threading the carry through the layer return
signature for full checkpoint safety. Only if forward-only cross-window shows
signal.

### Config

```
decompose_bypass_ssm        : false   # master toggle (overrides _ema when true)
decompose_bypass_ssm_poles  : 4       # P
```

Mutually exclusive with `decompose_bypass_ema` (SSM is the strict
generalization). When `decompose_bypass_ssm=false`, behavior is byte-identical
to today.

---

## Feature #2 — Truncated BPTT across windows (`decompose_bypass_bptt`)

Stop detaching the cross-window state every window; instead retain the graph for
`span` consecutive windows, backprop through the chain once, then truncate. This
gives the model an actual gradient incentive to encode useful long-range
information into the carried state (whatever its form — mean, EMA, or #1's pole
state). **Independent of #1** but most powerful combined with it: a richer state
that is also trained to be useful.

### Mechanism

- Today: `self._persistent_semantic_state = current_state[:, -1, :].detach()`
  ([model.py:2556](../model.py)). Add an instance flag `self._semantic_keep_graph`;
  when set, skip the detach (and likewise for `_persistent_ssm_state` under #1).
- Training loop (`train.py`, sequential path only): process `span` consecutive
  windows accumulating loss **without** detaching state between them, call
  `backward()` once over the span, then detach/reset the carried state at the
  span boundary. Truncation length = `span` windows = `span × block_size`
  tokens of gradient reach.
- Cleanest fit: tie `span` to the existing sequential grad-accumulation cycle —
  the `grad_accum` consecutive micro-batches in pure-sequential mode are already
  consecutive windows. Keep graph across the accum cycle, single `backward` at
  the optimizer step, detach at step boundary. So `span = grad_accum` by default
  (configurable cap).

### Guards

- **Requires `sequential_blocks=true`.** Hard error (or silent no-op with a
  warning) otherwise — BPTT through unrelated random windows is meaningless.
- Memory: retaining `span` windows' graphs raises activation memory ~`span×`.
  At span=grad_accum=4 and the current 7.8 GB train footprint this is the main
  cost to watch; `bptt_span` config caps it.

### Config

```
decompose_bypass_bptt       : false   # toggle truncated BPTT across windows
decompose_bypass_bptt_span  : 0       # windows per truncation; 0 = use grad_accum
```

---

## Ablation plan (README section after recurrence)

All on T4 base (Adagrad lr=0.02250, wavelet norms, fp16, 1 epoch). Decision rule:
clear T4 best val 3.5157 by > 0.0015. **Sequential mode required for the
cross-window effects to be real** — note whether the T4 reference for this
section is run sequentially or random (the cross-window state only does
something in sequential mode, so the honest baseline is a sequential T4).

| Variant | #1 SSM | #2 BPTT | Tests |
|---|---|---|---|
| T4 baseline (sequential) | ✗ | ✗ | reference for this section |
| + SSM | ✓ | ✗ | does a multi-timescale state beat the first-moment mean? |
| + BPTT | ✗ | ✓ | does training the (mean) cross-window state help at all? |
| + SSM + BPTT | ✓ | ✓ | richer state **and** trained to be useful — the full bet |

Expectation: #1 alone gives a modest lift (better state, but still untrained
across windows); #2 alone gives a modest lift (trained, but only a first moment
to write into); **#1+#2 is the one with a real ceiling** — a multi-timescale
recurrent memory that gradient actually shapes. If even #1+#2 is flat, it says
cross-window dependency isn't where WT103 perplexity lives at this scale (itself
a clean finding).

---

## Risk / rollback

- #1 is self-contained in `model.py` + 2 config keys; `false` = identical to
  today. Low risk.
- #2 touches the `train.py` sequential loop and the detach site; higher risk
  (memory, graph-retention correctness). Gated, sequential-only, `false` =
  identical to today.
- Both default off. Existing runs and the recurrence sweep are unaffected.
