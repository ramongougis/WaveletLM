# Strategies ablations

The `--strategies` bundle currently enables four real decoding interventions (entropy-adaptive temperature with cap 0.9, `top_p=0.85`, `repetition_penalty=1.2`, plus metrics + spacing cleanup). Three heavier strategies — `best_of_n`, multi-token `lookahead`, and `wavelet_coherence` decoding bias — were tested pre-WaveletLM and excluded from the bundle on a cost-vs-benefit basis (too slow, too VRAM-hungry, or quality-equivalent to lighter alternatives). Per-strategy contribution within the active bundle is unmeasured. The pre-WaveletLM rejection of the heavier strategies has not been re-validated against the current architecture.

## Open questions

1. Within the active `--strategies` bundle, which of the four interventions contributes the most to subjective sample quality? Plausible top contender: `repetition_penalty=1.2` (suppresses the model's natural tendency to loop, which would otherwise be visible).
2. Is the entropy-adaptive temperature with cap 0.9 doing useful work, or is it indistinguishable in practice from a static `temperature=0.8`?
3. Does `top_p=0.85` strictly dominate `top_p=0.95` for this model, or is the difference within sample noise?
4. **Re-validation question:** were the heavier strategies (`best_of_n`, `lookahead`, `wavelet_coherence`) actually worse-or-equivalent for *WaveletLM*, or were they tested on an earlier architecture where the cost/benefit shifted? The wavelet_coherence component in particular is wavelet-architecture-specific and may behave differently here than in pre-WaveletLM testing.

## Proposed study

N=20 samples per condition, single prompt ("The history of"), single seed, on the WT-103 best checkpoint.

**Phase 1 — within-bundle ablation:**

- Naive (`temp=1.0, top_p=0.95`, no other strategies)
- `--strategies` (full bundle — current default)
- `--strategies` minus `repetition_penalty` (set to 1.0)
- `--strategies` minus `entropy_adaptive` (replace with static `temperature=0.8`)
- `--strategies` with `top_p=0.95` instead of `0.85`

**Phase 2 — re-validate the excluded strategies:**

- `--strategies` plus `--best_of_n 5`
- `--strategies` plus `--lookahead_k 5 --lookahead_depth 5`
- `--strategies` plus `--wavelet_coherence`
- `--strategies` plus all three above (the maximum-strategies regime)

Metrics: mean log-probability (already logged), Distinct-1/2/3 (already logged), Rep-4 (already logged), per-token wall-clock time, peak VRAM, and human qualitative ranking on a 5-point scale across blinded conditions.

Compute cost: ~3-4 hours total on a 5090 (single checkpoint, no retraining; Phase 2 conditions are individually slower than Phase 1).

## Why this matters

Phase 1 attributes the visible quality gap between Sample D (naive) and Samples A–C (strategies-on) in the README to specific decoding interventions, which closes a residual interpretation gap for adversarial readers. Phase 2 either re-confirms the pre-WaveletLM rejection of the heavier strategies (in which case the bundle is well-chosen) or surfaces a case for re-bundling them (in which case `--strategies` should be expanded). The wavelet-coherence component in particular has no other validation in the literature — its inclusion in or exclusion from the bundle should rest on a measurement, not on an inherited heuristic.
