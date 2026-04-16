# Wavelet & Mixer Augmentations: Performance-First Round

## Summary

A set of modular augmentations to the wavelet decomposition and spectral mixer,
each implementable as an optional config flag for clean Boolean ablations.
Targets performance gains in the current learned-embedding EXARCH (interpretability
considerations deferred to post-semantic-embedding reintegration).

## Status: Implementation candidate

All features designed to be:
- **Optional** (config flag, false = today's behavior)
- **Independently togglable** (can be ablated one at a time)
- **Stackable** (compatible with each other)
- **Backward-compatible** (existing checkpoints load unchanged when flag=false)

## Feature 1: Untied reconstruction weights (cheapest)

### What

Currently `LiftingWaveletReconstruct` shares weights with `LiftingWaveletDecompose`:
```python
self.lifting_reconstruct = LiftingWaveletReconstruct(self.lifting_wavelet)
```

This enforces strict invertibility. Untying gives reconstruction its own predict/update
networks, allowing asymmetric "decompose this way → reconstruct that way" behavior.

### Cost

Roughly 2x lifting params per layer. At C=512: lifting is ~10M params per layer,
so untying adds ~10M per layer. At L=20: +200M total. At L=2 (current best): +20M.

### Config

```json
"untied_reconstruction": false
```

### Implementation

Replace shared module with a separate `LiftingWaveletDecompose` for reconstruction:
```python
if untied_reconstruction:
    self.reconstruct_wavelet = LiftingWaveletDecompose(...)  # separate weights
    self.lifting_reconstruct = LiftingWaveletReconstruct(self.reconstruct_wavelet)
else:
    self.lifting_reconstruct = LiftingWaveletReconstruct(self.lifting_wavelet)
```

The reconstruction logic stays the same — only the source of predict/update weights changes.

### Why this might help

EXARCH uses wavelets for *prediction*, not signal compression. The strict invertibility
constraint that classical wavelets need isn't necessary. The decompose path can specialize
on "what to extract from the input," and the reconstruct path on "how to combine back into
predictions." These are different objectives.

---

## Feature 2: Multi-basis lifting ensemble

### What

Run K parallel learnable lifting wavelets per layer, each initialized to a different
classical wavelet family (Haar, Daubechies-4, Symlet-4). Combine their decompositions
with per-scale learned weights (analogous to old EXARCH-research's Haar+db8 blend).

### Cost

K × lifting params per layer (K=2: doubles, K=3: triples). At L=2/C=2048 with K=2:
adds ~80M params. At K=3: ~160M.

### Config

```json
"multi_basis_lifting": false,
"multi_basis_inits": ["haar", "db4"]
```

The list controls which initializations to use. Length determines K.

### Implementation

Need to add `db4`, `sym4`, etc. initialization options to `LiftingWaveletDecompose`.
The lifting predict/update networks would be initialized to approximate these wavelets'
filter coefficients. Then per-layer:

```python
if multi_basis_lifting:
    self.lifting_wavelets = nn.ModuleList([
        LiftingWaveletDecompose(init_wavelet=name, ...)
        for name in multi_basis_inits
    ])
    # Learned blending weights per scale
    self.basis_weights = nn.Parameter(torch.zeros(K, levels + 1))
```

Forward pass: each wavelet produces its own (approx, details), then blend per scale:
```python
combined_approx = sum(softmax(basis_weights[:, 0])[k] * outs[k][0] for k in range(K))
combined_detail[level] = sum(softmax(basis_weights[:, level+1])[k] * outs[k][1][level] for k in range(K))
```

### Initialization details

This is the trickiest part. To init lifting predict/update to approximate db4, we need
to derive the equivalent lifting steps from db4's filter coefficients. This is well-known
in classical signal processing (Sweldens 1998 lifting scheme paper) but requires implementing
the conversion. Could start with just Haar+random (random gives diversity even without
proper init) as a quick test.

---

## Feature 3: Cross-scale gating in mixer

### What

Currently the SwiGLU gate at scale s only sees scale s's data. Make the gate input
depend on all scales:

```python
# Currently per-scale:
gate_s = σ(W_gate_s · X_s)

# Cross-scale (lightweight version):
# Mix scales via small learned routing matrix before per-scale gate
mixed_input = scale_routing @ stacked_X  # (S, S) routing matrix
gate_s = σ(W_gate_s · mixed_input[s])

# Cross-scale (full version):
# Concatenate all scales, gate sees everything
gate_s = σ(W_gate_s · concat(X_0, X_1, ..., X_{S-1}))
```

### Cost

**Lightweight (routing matrix):** S² extra params per layer, essentially free.
At S=10, L=2: 200 params total.

**Full (concat input):** Each gate's input grows from Cp to S·Cp.
W_gate_s grows from (Cp, Cp) to (S·Cp, Cp). At S=10, Cp=2048, L=2:
each gate is 5x bigger × 10 scales × 2 layers = ~83M extra params per gate set.

### Config

```json
"cross_scale_gating": false,
"cross_scale_gating_mode": "routing"  // or "full"
```

Two modes for two cost/expressivity tradeoffs.

### Implementation

In `GatedSpectralMixer` (or a new wrapper), accept all scales as input instead of one.
The "routing" mode is much smaller change — just multiply stacked spec by a learned
scale-routing matrix before extracting per-scale slices.

### Why this might help

The Hadamard transform is a *fixed* cross-scale rotation. The mixer then processes
each scale independently. Cross-scale gating lets the model learn *conditional*
cross-scale interactions: "when scale 0 (coarse) shows pattern X, modulate scale 4
(fine) processing." This kind of multiplicative interaction is fundamentally absent
from the current architecture.

---

## Feature 4: Asymmetric per-scale mixer width

### What

Currently all scales use the same Cp×Cp mixer. But coarse scales (high-level semantics)
likely benefit from more capacity than fine scales (local detail). Use a width
multiplier per scale:

```python
# Coarsest scale: full Cp width
# Mid scales: Cp width
# Finest scales: Cp/2 width (or shared mixer)
```

### Cost

Saves params if implemented as decreasing width. For example, halving the finest 3 scales:
saves `3 * (Cp²/2) * 2 (mixer + gate) * L` params. At L=2/C=2048: saves ~62M.

### Config

```json
"per_scale_mixer_widths": null  // or [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
```

Optional list of width multipliers (one per scale). null = all 1.0 (today's behavior).

### Implementation

In `ExarchBlock.__init__`, build mixers with different widths:

```python
if per_scale_mixer_widths is not None:
    widths = [int(Cp * w) for w in per_scale_mixer_widths]
    # Each scale's mixer has its own width
    # Need projections to/from common Cp dimension at scale boundaries
else:
    widths = [Cp] * S
```

Slight wrinkle: scales operate in a common Hadamard space, so scale s with width w_s
needs `Cp → w_s` projection in, mixer at w_s, `w_s → Cp` projection out. Adds a small
overhead per scale.

### Why this might help

The levels=5 finding showed the finest 4 levels (in a 9-level decomposition) added
params without value. Per-scale widths formalize this: keep all levels but starve the
fine ones of capacity, freeing budget for coarser scales.

---

## Feature 5: Wavelet packet decomposition

### What

Standard pyramid wavelet only decomposes the approximation at each step:
```
input → (approx_1, detail_1)
  approx_1 → (approx_2, detail_2)
    approx_2 → (approx_3, detail_3)
      ...
```

Wavelet packet decomposes BOTH approximation and detail at each step:
```
input → (approx_1, detail_1)
  approx_1 → (aa_2, ad_2)
  detail_1 → (da_2, dd_2)
    aa_2 → (aaa_3, aad_3)
    ad_2 → (ada_3, add_3)
    da_2 → (daa_3, dad_3)
    dd_2 → (dda_3, ddd_3)
```

This gives a binary tree of subbands instead of a pyramid. For levels=L, you get 2^L
subbands instead of L+1. Each subband captures a specific frequency range.

### Cost

At levels=5: pyramid has 6 subbands, packet has 32. The mixer would need to handle 32
scales instead of 6. Mixer cost grows from `6 * Cp² * 2` to `32 * Cp² * 2` per layer.
Significant — about 5x mixer params.

To make this feasible, would likely combine with Feature 4 (per-scale widths) — keep
only the most useful subbands at full width, others at reduced width or pruned.

### Config

```json
"wavelet_packet": false,
"wavelet_packet_max_subbands": 16  // optional pruning
```

### Implementation

Would significantly restructure `LiftingWaveletDecompose`. The lifting steps would
recursively apply to both approx and detail at each level. Reconstruction follows
the same tree in reverse.

### Why this might help

The pyramid only captures fixed-bandwidth subbands at each scale. Packets give a
richer subband structure that can match the actual frequency content of language.
Some language patterns (e.g., periodic syntactic structures) might live in subbands
that the pyramid lumps together.

---

## Suggested ablation order

1. **Untied reconstruction** — cheapest, most likely to help
2. **Cross-scale gating (routing mode)** — essentially free, plausible win
3. **Multi-basis lifting (Haar + random init)** — moderate cost, novel
4. **Per-scale mixer widths** — likely a win given levels=5 finding
5. **Cross-scale gating (full mode)** — if routing helped, try the bigger version
6. **Wavelet packet decomposition** — biggest change, save for last

After individual ablations, the best combinations form the next "best run candidate."

## Implementation effort estimate

- Untied reconstruction: ~30 min (trivial)
- Cross-scale gating routing: ~1 hour
- Per-scale mixer widths: ~2 hours
- Multi-basis lifting (Haar + random only): ~1 hour
- Cross-scale gating full: ~2 hours
- Multi-basis lifting (proper db4/sym4 inits): ~4-8 hours (filter coefficient derivation)
- Wavelet packet decomposition: ~1-2 days (significant restructure)

The first 4 features could all be implemented in a single afternoon and tested in
sequence — making this a high-value, low-effort batch of improvements.
