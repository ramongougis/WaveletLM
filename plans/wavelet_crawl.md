# Wavelet Crawl: Learned Dilation Patterns for Lifting Wavelets

## Summary

Replace the fixed power-of-2 dilation pattern in WaveletLM's lifting wavelet decomposition
with learnable dilation offsets. Each wavelet level discovers which token spacings are
most informative rather than being locked to 2^k strides. The dilations "crawl" toward
optimal positions via gradient descent.

## Motivation

- WaveletLM's lifting predict/update steps compare positions at fixed dilations: 1, 2, 4, 8, ...
- The model cannot learn that level 3 should compare tokens 7 apart instead of 8
- Wider lifting MLPs (hidden_mult) don't help because they add local expressivity,
  not range — the bottleneck is which positions are compared, not how the comparison
  is computed
- Layers help precisely because each layer re-runs the decomposition, allowing
  information to cascade through multiple rounds of fixed-dilation comparisons
- Wavelet crawl could achieve similar benefits within a single layer by letting
  each level find its optimal receptive field

## Approach: Soft dilation mixing

For each level, instead of a single hardcoded dilation, learn a weighted mixture
of K candidate dilations centered around the default 2^k:

```python
# Current (fixed):
dilation = 1 << level
odd = padded[:, :-dilation, :]

# Wavelet crawl (learned):
base_dilation = 1 << level
candidates = [base_dilation - 1, base_dilation, base_dilation + 1]  # or wider range
weights = softmax(self.dilation_logits[level])  # learned, (K,)
odd = sum(w * get_offset(padded, d) for w, d in zip(weights, candidates))
```

### Parameter cost

- Per level: K logit values (e.g., K=3 gives 3 params per level)
- Total: levels * K = 9 * 3 = 27 parameters per layer
- Negligible — this is a pure routing decision, not a capacity addition

### Compute cost

- K padded lookups per level instead of 1
- At K=3: 3x the pad+index operations, but these are memory ops, not matmuls
- Net wall time increase: < 5% estimated

## Design decisions to explore

1. **Candidate range**: K=3 (base-1, base, base+1) is minimal.
   K=5 (base-2 to base+2) gives more freedom.
   Asymmetric ranges could also work (looking further forward vs backward).

2. **Integer vs soft dilation**: The soft mixture above blends contributions from
   multiple offsets. An alternative is Gumbel-softmax to learn discrete dilation
   choices, converging to a single offset per level after training.

3. **Per-channel dilations**: Instead of one dilation per level (shared across all
   C channels), learn per-channel or per-group dilations. Different embedding
   dimensions might benefit from different receptive fields. Cost: K * C params
   per level instead of K.

4. **Dilation schedule across training**: Start with fixed 2^k dilations (warm start),
   then gradually enable crawl. This prevents the model from destabilizing early
   when the dilation pattern is still random.

## Interaction with other features

- **Mixer depth**: Wavelet crawl + MD=2 would give each mixing pass access to
  a learned receptive field. The depth steps could benefit more from non-standard
  dilation patterns.
- **Layers**: With wavelet crawl, fewer layers might be needed — each layer's
  wavelet decomposition is already optimized rather than relying on cascaded
  fixed decompositions across layers.
- **L=1 models**: Most impactful here, since L=1 has only one chance to decompose.
  The fixed dilation pattern is the biggest limitation at L=1.

## When to implement

After the current sweep (lifting hidden mult, L=1 ablations) confirms that the
bottleneck is receptive field pattern, not local expressivity. If hidden_mult=4
shows negligible improvement over hidden_mult=1, that directly motivates wavelet
crawl as the next architectural change.

## Potential for publication

Learned dilation lifting wavelets are novel. Fixed-dilation lifting is well-studied
in signal processing (Sweldens 1996). Applying learned dilations to language modeling
via the lifting scheme has not been explored. If wavelet crawl improves BPB at L=1,
it's a standalone contribution independent of the rest of WaveletLM.
