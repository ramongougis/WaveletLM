# Stable Parametrization for EXARCH

## Summary

Reparameterize key weight matrices to prevent signal explosion during training,
enabling stable training at higher depth, larger C, bigger block sizes, and
higher learning rates. Inspired by the Linear Recurrent Unit paper (arxiv:2303.06349)
and its stable exponential parametrization approach.

## Status: Priority — addresses multiple NaN failure modes

## Problem

EXARCH experiences NaN during training in several configurations:
- Mixer depth >= 3 at L=20 with lr=0.01
- C=2048 with block_size=2048
- C=2048 with lr=0.02
- Mixer depth >= 5 at any LR (without residuals)

Root cause: signal magnitudes grow uncontrolled through composed transforms.

## Affected components and proposed changes

### 1. GatedSpectralMixer — mixer.weight

**Current init:**
```python
mixer.weight = eye(Cp) + N(0, 1e-3)  # fixed eps regardless of C
```

**Problems:**
- eps=1e-3 is constant across C — at C=2048, the noise-to-signal ratio is
  identical to C=128, but gradient accumulation differs
- No runtime constraint — weights can grow unbounded during training
- The gated output (Wx * σ(Gx)) can amplify signals quadratically

**Proposed changes:**
- Scale eps with C: `eps = 1e-3 / sqrt(C)` (or `1/C` for more aggressive scaling)
- Apply spectral normalization to mixer.weight:
  ```python
  self.mixer = nn.utils.spectral_norm(nn.Linear(Cp, Cp, bias=False))
  ```
  This constrains the largest singular value to 1.0, preventing any single
  direction from amplifying signal. Cost: one SVD-like computation per forward
  pass (cheap for the matrix sizes involved).

**Alternative — exponential parametrization:**
Store raw parameters θ, compute effective weights as:
```python
# For diagonal case (most efficient):
effective_weight = eye(Cp) + diag(exp(theta)) @ noise_directions
# Eigenvalues bounded by exp(theta), controllable via θ
```

### 2. GatedSpectralMixer — gate.weight

**Current init:** `N(0, 0.02)` — no scaling with C.

**Problem:** At C=2048, the gate output σ(Gx) has 2048 dimensions, each with
std ~0.02. The collective effect is larger at high C.

**Proposed:** `N(0, 0.02 / sqrt(C / 512))` — scale relative to baseline C=512.

### 3. FeedForward — final layer scaling

**Current:** `.weight.mul_(0.02)` — fixed regardless of C, layers, or MLP expansion.

**Problem:** At MLP=20 with C=2048, the hidden dimension is 40,960. The final
projection from 40,960 → 2048 with 0.02 scaling means each output dimension
receives a sum of ~40,960 * 0.02 ≈ 819 weighted inputs at init. The variance
is not controlled.

**Proposed:** Scale as `1 / sqrt(hidden_dim)`:
```python
final_linear.weight.mul_(1.0 / math.sqrt(hidden_dim))
```
This is standard Xavier-like scaling that keeps variance constant regardless
of hidden dimension.

### 4. proj_out scaling

**Current:** `.weight.mul_(1e-3)` — very small, fixed.

**Problem:** At large C, the proj_out maps Cp → C. The 1e-3 scaling makes the
wavelet mixer's contribution negligibly small at init, forcing the model to
learn through the residual path first. This is conservative but may slow
the mixer's ability to contribute early in training.

**Proposed:** Scale as `1 / sqrt(C * layers)`:
```python
proj_out.weight.mul_(1.0 / math.sqrt(C * num_layers))
```
This follows the GPT-2 convention of scaling residual contributions by
the number of residual additions. Each layer contributes proportionally
less, preventing signal growth through the residual stream.

### 5. Embedding scaling

**Current:** `N(0, 1/sqrt(C))` — good scaling for init.

**Problem:** No runtime scaling. The original Transformer paper multiplies
embeddings by sqrt(C) to match the scale of positional encodings. EXARCH
doesn't do this, meaning embedding magnitudes decrease as C increases.

**Proposed:** Multiply embedding output by sqrt(C):
```python
x = self.token_embedding(idx) * math.sqrt(self.C)
```
This keeps the embedding signal magnitude constant regardless of C.

### 6. Lifting predict/update — dilated interaction scaling

**Current:** Identity init with 0.5 scaling on update step.

**Problem:** At higher wavelet levels (larger dilation), the predict/update
steps compare tokens that are far apart. The signal-to-noise ratio degrades
at higher levels because the relationship between distant tokens is weaker.

**Proposed:** Scale the init by level:
```python
# Level-dependent scaling for predict step
predict_scale = 1.0 / (1 + level * 0.1)  # gentler at higher levels
update_scale = 0.5 / (1 + level * 0.1)
```

## Implementation priority

1. **Spectral norm on mixer** — most impactful, directly addresses NaN in depth/LR
2. **FF final layer scaling with sqrt(hidden_dim)** — proper variance control
3. **Embedding sqrt(C) scaling** — standard practice, easy to add
4. **proj_out scaling with sqrt(C * layers)** — prevents residual stream growth
5. **eps scaling with C** — refinement, less critical
6. **Level-dependent lifting scaling** — speculative, test last

## Config

```json
"stable_parametrization": false
```

false = today's behavior (backward compatible). true = enable all stabilization
changes. Individual components could also be toggled if needed.

## Testing plan

Test each change individually at the previously-failing configurations:
1. C=2048, lr=0.02 (NaN'd at step 700)
2. C=2048, block_size=2048 (NaN'd at step 300)
3. mixer_depth=5 at L=20 (NaN'd at step 3600)

If any single change fixes the NaN, it identifies the bottleneck.
If multiple are needed, enable all together.

## Relationship to mixer_depth_stabilizers

The mixer_depth_stabilizers (alpha/beta scaling) were an earlier attempt at this
problem. They added learnable scalars but didn't constrain weight magnitudes.
Stable parametrization addresses the root cause (unbounded weight growth) rather
than the symptom (signal magnitude). If stable parametrization works, stabilizers
become unnecessary.

## References

- Orvieto et al. (2023). "Resurrecting Recurrent Neural Networks for Long Sequences."
  arXiv:2303.06349. Key insight: exponential parametrization of recurrence matrices
  ensures stable forward pass for arbitrary sequence lengths.
- Radford et al. (2019). GPT-2: residual scaling by 1/sqrt(N) for depth.
- Vaswani et al. (2017). Transformer: embedding scaling by sqrt(d_model).
