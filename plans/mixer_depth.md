# Mixer Depth: Stacked Spectral Mixing

## Summary

Add configurable depth to the per-scale spectral mixing step within each ExarchBlock.
Currently, each scale gets a single GatedSpectralMixer pass (depth 1). This feature stacks
multiple passes with independent weights, LayerNorm, bias, and residual connections — creating
a mini deep network in Hadamard/frequency space without repeating wavelet decompose/reconstruct
or Hadamard transforms.

## Motivation

- The current mixer is a depth-1 gated linear transform per scale — equivalent to a
  single-layer network in frequency space
- Adding depth here increases spectral expressivity without adding full ExarchBlock layers
  (which redo wavelets, Hadamard, MLP, PKM, etc.)
- EXARCH-research had `mixer_repeats` but it used raw residuals with no normalization,
  causing NaN at step 7300 during warmup (LR ~0.015). This is a stabilized redesign.

## Mathematical formulation

```
mixer_depth=1 (today's behavior, unchanged):
  Y = W_mixer * X . sigma(W_gate * X)

mixer_depth=D (D >= 2):
  Y_0 = X   (input in Hadamard space, per scale)

  For depth d = 1, ..., D-1:      (intermediate steps: LN + bias + residual)
    Z_d = LN_d(Y_{d-1})
    Y_d = Y_{d-1} + [W_mixer_d * Z_d . sigma(W_gate_d * Z_d) + B_d]

  Final step d = D:                (no LN, no bias — mirrors depth=1 output)
    Y_D = Y_{D-1} + [W_mixer_D * Y_{D-1} . sigma(W_gate_D * Y_{D-1})]

  Output = Y_D
```

Where:
- LN_d = LayerNorm(Cp) — pre-norm stabilization (intermediate steps only)
- W_mixer_d = Linear(Cp, Cp, bias=False) — mixer path
- W_gate_d = Linear(Cp, Cp, bias=False) — gating path, sigma = SiLU
- B_d = Parameter(Cp) — learned bias after gating (intermediate steps only)
- Residual connection from Y_{d-1} to Y_d

At mixer_depth=1, no LN or bias is added — behavior is identical to today's code.
At mixer_depth>1, the final step also omits LN/bias so its output boundary matches
the current single-pass behavior (no magnitude change before downstream path).

## Config

```json
"mixer_depth": 1,
"__comment_mixer_depth": "Depth of per-scale spectral mixing within each block. 1 = single gated mixer (near-baseline). Each additional depth step adds LN + gated linear + bias + residual in Hadamard space. Adds ~105M params per +1 depth at C=512/L=20/S=10."
```

## Parameter cost

Per +1 depth (C=512, S=10 scales, L=20 layers):

| Component        | Per scale | x10 scales | x20 layers |
|------------------|-----------|------------|------------|
| W_mixer (Cp^2)   | 262,144   | 2.62M      | 52.4M      |
| W_gate (Cp^2)    | 262,144   | 2.62M      | 52.4M      |
| B (Cp)           | 512       | 5,120      | 102K       |
| LN (2 x Cp)      | 1,024     | 10,240     | 205K       |
| **Total**        | **525,824** | **5.26M** | **~105.2M** |

Note: depth=1 is exactly today's code — no LN or bias added, zero overhead.
Depth=2 adds one set of LN+bias (intermediate step) plus one set of W_mixer+W_gate
(final step). The intermediate steps (d=1..D-1) have LN+bias; the final step (d=D)
has only W_mixer+W_gate.

## Implementation

### Files to modify

1. **config.json** — add `mixer_depth` key (default 1)
2. **model.py** — GatedSpectralMixer and ExarchBlock changes

### model.py changes

#### GatedSpectralMixer (lines ~286-329)

Add `bias` parameter to `__init__`:

```python
def __init__(self, Cp, num_blocks, rank, ..., add_bias=False, ...):
    ...
    if add_bias:
        self.bias = nn.Parameter(torch.zeros(Cp, device=device, dtype=dtype))
    else:
        self.bias = None
```

In `forward`:
```python
def forward(self, X_spec):
    signal = self.mixer(X_spec)
    if self.use_mixer_gate:
        out = signal * self.gate_activation(self.gate(X_spec))
    else:
        out = signal
    if self.U is not None:
        mid = torch.matmul(X_spec, self.V)
        out = out + torch.matmul(mid, self.U.t())
    if self.bias is not None:
        out = out + self.bias
    return out
```

#### ExarchBlock.__init__ (lines ~636+)

Replace single `scale_mixers` list with nested structure:

```python
mixer_depth = config.get("mixer_depth", 1)
self.mixer_depth = mixer_depth

if mixer_depth == 1:
    # Exactly today's code — single set of mixers, no LN, no bias
    self.scale_mixers = nn.ModuleList([
        GatedSpectralMixer(
            Cp=Cp, num_blocks=1, rank=low_rank,
            use_mixer_gate=use_mixer_gate,
            mixer_gate_activation=mixer_gate_activation,
            add_bias=False,
            device=device, dtype=dtype,
        )
        for _ in range(S)
    ])
else:
    # Depth > 1: intermediate steps (LN + bias), final step (no LN, no bias)
    # Intermediate steps: d=0..D-2
    self.mixer_depth_norms = nn.ModuleList([
        nn.ModuleList([nn.LayerNorm(Cp, device=device, dtype=dtype) for _ in range(S)])
        for _ in range(mixer_depth - 1)
    ])
    self.scale_mixers_by_depth = nn.ModuleList([
        nn.ModuleList([
            GatedSpectralMixer(
                Cp=Cp, num_blocks=1, rank=low_rank,
                use_mixer_gate=use_mixer_gate,
                mixer_gate_activation=mixer_gate_activation,
                add_bias=(d < mixer_depth - 1),  # bias on intermediate, not final
                device=device, dtype=dtype,
            )
            for _ in range(S)
        ])
        for d in range(mixer_depth)
    ])
```

#### ExarchBlock.forward (lines ~813-819)

Replace single mixer pass with depth loop:

```python
if self.mixer_depth == 1:
    # Exactly today's code path
    mixed_by_scale = []
    for s in range(S):
        Xs = stacked_spec[:, :, s, :]
        Ys = self.scale_mixers[s](Xs)
        mixed_by_scale.append(Ys)
    mixed_spec = torch.stack(mixed_by_scale, dim=2)
else:
    # Stacked spectral mixing with depth
    mixed_spec = stacked_spec
    for d in range(self.mixer_depth):
        depth_mixers = self.scale_mixers_by_depth[d]
        mixed_by_scale = []
        for s in range(S):
            Xs = mixed_spec[:, :, s, :]
            if d < self.mixer_depth - 1:
                # Intermediate: LN + mixer(+bias) + residual
                Xs_normed = self.mixer_depth_norms[d][s](Xs)
                Ys = depth_mixers[s](Xs_normed)
                mixed_by_scale.append(Xs + Ys)
            else:
                # Final: raw mixer, no LN, no bias, + residual
                Ys = depth_mixers[s](Xs)
                mixed_by_scale.append(Xs + Ys)
        mixed_spec = torch.stack(mixed_by_scale, dim=2)
```

### Checkpoint compatibility

- mixer_depth=1 is identical to today's code — `scale_mixers` key unchanged,
  full checkpoint compatibility with no migration needed
- mixer_depth>1 uses `scale_mixers_by_depth` — new checkpoints only, no need
  to load old checkpoints at depth>1 since this is a new architecture variant

### generate.py / train.py

No changes needed beyond config passthrough (already reads all config keys).

## Ablation plan

Add to runs.md after memory sweep, before layers sweep:

| Run | mixer_depth | Params     | Notes                          |
|-----|-------------|------------|--------------------------------|
|     | 1           | ~366.9M    | Baseline + LN/bias overhead    |
|     | 2           | ~472.1M    | First depth increase           |
|     | 3           | ~577.3M    | Diminishing returns expected   |
|     | 5           | ~787.7M    | Stress test                    |

All at C=512, L=20, mlp_expansion=1, epochs=1. Compare against MLP expansion
at matched param counts to determine which is a better use of parameters.

## Key comparison: mixer_depth vs mlp_expansion

Both add ~105M params per step. The question is whether spectral depth
(richer frequency transforms) buys more than MLP width (more semantic memory).
If mixer_depth=2 beats mlp_expansion=10 (~94M params, -0.0219 BPB), spectral
depth is the more efficient lever. If not, MLP expansion remains primary and
mixer_depth is secondary.

## Stability notes (lessons from EXARCH-research)

- Old `mixer_repeats=2` hit NaN at step 7300 (LR=0.015, still in warmup)
- Root cause: no normalization between repeats, gradient magnitudes compounded
- Fix: pre-norm LayerNorm before each depth step (this plan)
- Additional safety: bias initialized to zero, mixer W initialized near-identity
- If NaN still occurs: reduce LR, or add dropout between depth steps
