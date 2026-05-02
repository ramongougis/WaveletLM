# Multinodal / multi-expert training techniques — research survey

Tentative catalog of techniques for combining multiple nodes/experts/branches that could extend WaveletLM's current multinodal mode beyond naive logit averaging + feature bagging. No commitment to any specific path; this is a reference for future-us to pick from when the release dust settles. Budget-limited: most entries include a cost estimate.

Current multinodal mode (baseline to compare against):
- N independent cells, different seeds, feature bagging per cell
- Combination: averaged logits (`multinodal_combination: average`)
- Optional `multinodal_cross_cell_gating` — underexplored

## Families

### 1. Sparse Mixture of Experts (sparse MoE)

Tokens routed to a subset of experts per step; unused experts don't consume compute. Biggest parameter-per-compute efficiency lever in modern LMs.

- **Switch Transformer** — Fedus et al., 2021 ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961)). Top-1 routing, load balancing loss. Simplest variant.
- **Mixtral 8x7B** — Jiang et al., 2024 ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088)). Top-2 routing; dense-quality at sparse compute.
- **GLaM** — Du et al., 2022 ([arXiv:2112.06905](https://arxiv.org/abs/2112.06905)). 64 experts × top-2.
- **Expert Choice routing** — Zhou et al., 2022 ([arXiv:2202.09368](https://arxiv.org/abs/2202.09368)). Experts pick tokens; perfect load balancing by construction, works at smaller batches.
- **Hash Layers** — Roller et al., 2021 ([arXiv:2106.04426](https://arxiv.org/abs/2106.04426)). No learned routing; tokens hashed to experts.
- **SMEAR** — Muqeeth et al., 2023 ([arXiv:2306.03745](https://arxiv.org/abs/2306.03745)). Soft expert merging; fixes MoE's discrete-routing training fragility.

**Fit for WaveletLM:** High. MLP is the primary capacity lever and 60%+ of param budget at MLP=20. Replacing dense MLP with sparse MoE would give ~4–8× effective MLP capacity at ~2× MLP compute. Wavelet/FWHT pipeline unaffected — MoE lives in the position-wise slot.

**Compute to test:** ~17h for a first 5-epoch run at C=2048 with 4 experts × top-2 on the MLP. One architecture change, one training run.

---

### 2. Dense ensembles / weight averaging

Multiple full models combined at output (averaged predictions) or weights (averaged parameters).

- **Deep Ensembles** — Lakshminarayanan et al., 2017 ([arXiv:1612.01474](https://arxiv.org/abs/1612.01474)). Independent random inits, averaged predictions.
- **Snapshot Ensembles** — Huang et al., 2017 ([arXiv:1704.00109](https://arxiv.org/abs/1704.00109)). Cyclic LR schedule producing multiple checkpoints from one run.
- **Stochastic Weight Averaging (SWA)** — Izmailov et al., 2018 ([arXiv:1803.05407](https://arxiv.org/abs/1803.05407)). Averaging checkpoints across late-training steps of a *single* run. Works reliably.
- **Model Soups** — Wortsman et al., 2022 ([arXiv:2203.05482](https://arxiv.org/abs/2203.05482)). Averaging weights of fine-tuned models *sharing a common init*. Does not generalize to independent-init runs.
- **Linear Mode Connectivity** — Frankle et al., 2020 ([arXiv:1912.05671](https://arxiv.org/abs/1912.05671)). Foundational theory for when weight averaging works.
- **Git Re-Basin** — Ainsworth et al., 2022 ([arXiv:2209.04836](https://arxiv.org/abs/2209.04836)). Permutation-aligned weight averaging across independent inits. Research-grade; works cleanly for small networks, less clear for large transformers.
- **Fisher-Weighted Averaging** — Matena & Raffel, 2021 ([arXiv:2111.09832](https://arxiv.org/abs/2111.09832)). Fisher-information-weighted merging for fine-tuned models.
- **Neyshabur et al., 2020** — "What is being transferred in transfer learning?" ([arXiv:2008.11687](https://arxiv.org/abs/2008.11687)). Confirms same-init descendants share basin.
- **Born-Again Networks** — Furlanello et al., 2018 ([arXiv:1805.04770](https://arxiv.org/abs/1805.04770)). Distill ensemble into single model.

**Fit for WaveletLM:**
- **SWA within a single training run** — Cheap; average the last N checkpoints of any one of the 3 seed runs. Well-validated technique, low risk.
- **Model Soups of 3-seed runs** — Low expected value. The 3 seeds are independent-init; averaging weights typically degrades, not improves. *Correction to my earlier suggestion: this is not a free expected-positive ablation.*
- **Git Re-Basin on the 3 seeds** — Research-grade. Would actually test the "independent inits can be aligned" hypothesis on a wavelet architecture, which hasn't been done elsewhere. Could produce a novel finding (positive or negative) for a follow-up paper.
- **Fisher-weighted averaging** — Only applicable if we ever do fine-tuning from a shared pretrained WaveletLM.

**Open research question raised by the user:** if independently-trained WaveletLMs can be weight-averaged successfully (after permutation alignment or directly), that would say something fundamental about how the wavelet-mixer inductive bias organizes representations. Weight-distribution correlation analysis across seeds could test this. **Semantic embedding reintroduction plausibly strengthens this** — shared, anchored input/output coordinate systems reduce permutation freedom in the interior.

**Compute cost:** SWA within a run ≈ 0 extra compute. Git Re-Basin on 3 seeds ≈ CPU-hours + 3-epoch re-eval.

---

### 3. Multi-branch / multi-view architectures

Multiple processing paths within one network, jointly trained.

- **Inception** — Szegedy et al., 2015 ([arXiv:1409.4842](https://arxiv.org/abs/1409.4842)). Parallel CNN branches with different receptive fields.
- **BranchyNet / Early Exit** — Teerapittayanon et al., 2016 ([arXiv:1709.01686](https://arxiv.org/abs/1709.01686)). Auxiliary heads at different depths.
- **Dual Path Networks** — Chen et al., 2017 ([arXiv:1707.01629](https://arxiv.org/abs/1707.01629)). Two parallel paths combined per layer.
- **Cross-stitch Networks** — Misra et al., 2016 ([arXiv:1604.03539](https://arxiv.org/abs/1604.03539)). Learned per-task combination weights.

**Fit for WaveletLM:**
- **Multi-basis lifting** is already in the codebase (`multi_basis_lifting` flag, Haar + random parallel wavelets). It's a multi-branch architecture within the lifting cascade. Tested previously; stability-sensitive (see [runs.md](../runs.md), Multi-basis lifting rows). Validation deferred to stable_parametrization completion.
- **LM head at multiple depths (BranchyNet-style)** — Useful if pushing to L=8+ at B200. Provides auxiliary gradient paths; known to stabilize deep nets.
- **Inception-style parallel wavelet branches** — Wavelet packet decomposition (see [other_post_release_plans.md §2](other_post_release_plans.md)) is effectively this idea for frequency sub-bands.

**Compute cost:** Varies; architectural surgery required.

---

### 4. Mutual learning and cross-model knowledge transfer

Multiple models trained *with each other* as implicit teachers.

- **Deep Mutual Learning** — Zhang et al., 2018 ([arXiv:1706.00384](https://arxiv.org/abs/1706.00384)). N networks in parallel, each distilled from the others' soft predictions via KL. All networks improve vs solo training.
- **Co-training** — Blum & Mitchell, 1998. Classical; two models on different data views. Less applicable to LM.
- **FitNet** — Romero et al., 2014 ([arXiv:1412.6550](https://arxiv.org/abs/1412.6550)). Hint-based distillation from hidden states.
- **Knowledge Distillation** — Hinton et al., 2015 ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)). Canonical teacher-student setup.

**Fit for WaveletLM:**
- **Deep Mutual Learning across multinodal cells** — Excellent fit. Current multinodal mode trains N cells independently; adding KL-divergence loss between pairs of cells' logit distributions during training would let cells teach each other. Typical reported gain: +0.5–1% accuracy over plain ensemble averaging. Small code change, no architectural change.
- **Ensemble distillation into single release model** — Train N-cell multinodal model, distill predictions into a single-cell model for release. Gets ensemble quality at single-cell inference cost.

**Compute cost:** Deep Mutual Learning adds N² KL terms per forward pass; marginal compute overhead. Ensemble distillation needs a second training pass on the distilled target.

---

### 5. Routing and specialization losses

Auxiliary losses that shape expert behavior regardless of the architecture choice above.

- **Load balancing loss** — Ubiquitous in sparse MoE. Penalizes expert utilization variance.
- **Router z-loss** — Stabilizes router logits against explosion. Used in Switch, Mixtral.
- **Orthogonality / diversity loss** — `||W_i · W_jᵀ - I||` or cosine-similarity penalties to force expert specialization. Used in disentanglement / VAE literature.
- **Disagreement loss** — Reverse of mutual-learning KL; explicitly encourages cells to produce different predictions (for ensemble diversity).
- **Expert dropout** — Randomly drop experts during training to prevent gating overfit.

**Fit for WaveletLM:** Any MoE direction requires load balancing + z-loss. Orthogonality loss could strengthen current multinodal mode without any architecture change — add `λ·||W_i · W_jᵀ - I||` penalty across cells to push from purely-additive (averaging) toward multiplicative (PoE-like) combination.

---

### 6. WaveletLM-native full-node combination strategies

Combination methods that exploit WaveletLM's specific structure (multi-scale decomposition, per-scale gated mixing) and don't cleanly map to generic MoE or ensemble literature. Distinct from logit averaging or weight averaging because the combination happens *inside* the wavelet pipeline rather than after the final logits.

> **Note:** The WaveletLM-native architecture-internal direction — **multi-transform parallelization**, where N parallel orthogonal-transform paths replace the single FWHT slot inside each per-scale mixer with shared wavelet scaffolding — was previously described in this section. It has been promoted to its own dedicated plan: see [`multi_transform_parallelization.md`](multi_transform_parallelization.md). That direction is architectural (operates inside a single model, ~5-15% compute overhead). The §6 entries below are full-node directions (operate at the model-ensemble level, building on the existing `multinodal_enabled` mode).

**Wavelet-domain combination of full nodes (primary §6 proposal):** Each node produces its full multi-scale output (S=6 scales from the standard `levels=5` setup). Instead of averaging final logits across nodes (current `multinodal_combination: average`), each node's output is decomposed into wavelet coefficients, and matching coefficients are combined across nodes *per-scale* before inverse transform and reconstruction. Three variants:

- **Equal-weight per-scale averaging** — identical to logit averaging in expectation, but operates on wavelet coefficients before the final reconstruction. Differs primarily in gradient flow (per-scale gradients reach each node directly rather than mediated by the LM head).
- **Learned per-scale gating across nodes** — each scale has a learned (S × N) gate matrix that weights node contributions per scale. Lets nodes naturally specialize: node A's coarse coefficients dominate at scales 0-2, node B's fine coefficients dominate at scales 3-5. Differentiable; no hard routing needed. Adds N×S learned scalars per layer (negligible parameter cost).
- **Hard scale-specialization** — assign nodes to scales statically (node A handles scales 0-2, node B handles scales 3-5). Combine by zero-padding non-assigned scales and summing. Reduces per-node compute since each node only needs to compute a subset of scales — potential 2× speedup when paired correctly. Forces specialization but loses adaptability.

**Spatial topology between nodes:** Add adjacency relationships ("brain patches" metaphor — neighboring nodes share more representational structure than distant ones). Implementations: shared lifting weights between neighboring nodes only (not all-shared), or a learned (N × N) coupling matrix that lets one node's per-scale outputs propagate into a neighbor's mixer input as side-channel information. Most useful when nodes are not functionally identical (e.g., when paired with hard scale-specialization or capacity-aware routing).

**Capacity-aware routing:** Most MoE routes by content affinity (token-to-expert similarity). For WaveletLM, *load* signals are available without additional machinery: FwPKM key utilization (how saturated is each node's sparse memory), per-scale mixer activation magnitudes (which scales is each node "full" in), gradient magnitudes (which nodes are still actively learning vs. plateaued). A small router — a single linear layer over a concatenation of these signals plus the input embedding — could route to the least-loaded node, balancing memorization across nodes without requiring an explicit load-balancing auxiliary loss.

**Fit for WaveletLM:** Native — all three strategies use machinery WaveletLM already has (per-scale wavelet outputs, FwPKM utilization, multi-scale decomposition). No transformer-derived machinery (attention, position encodings, key-value caches) is required for any of them. The wavelet-domain combination in particular is essentially "do what's already happening at the per-cell level, but stop averaging at the wrong layer of abstraction."

**Compute to test:** Wavelet-domain combination is the cheapest first step. Code change is in the multinodal combiner only (existing infrastructure already provides per-node outputs; the combiner just needs to switch from averaging final logits to combining per-scale wavelet coefficients before reconstruction). No new training pipeline. ~5h for a first 5-epoch validation run on top of the existing 4-cell multinodal config.

**Open questions:**

1. Does wavelet-domain combination produce meaningfully different fits than logit averaging at equal compute? Same N-cell run with both combiners as a clean A/B test.
2. Does learned per-scale gating across nodes naturally produce the specialization the gating architecture allows, or do nodes converge to uniform contributions? (Mode-collapse risk.)
3. Does hard scale-specialization recover the per-node compute savings without quality loss? (If node A can skip computing scales 3-5 entirely, that's a real wall-clock win.)
4. Does spatial topology between nodes provide any benefit over independent nodes, given WaveletLM's wavelet pipeline already provides natural multi-scale locality at the within-node level?
5. How does this interact with the L=1 + parameter reduction direction? The reduced model has less per-node capacity, which may make scale-specialization more attractive (each node has a tighter functional ceiling and might benefit from focusing on fewer scales).

---

## Tentative priority order

Ranked by expected value per compute-hour spent, given budget constraints:

> **Note:** Multi-transform parallelization (the architectural-internal "prism" direction, previously listed at #1 of this priority list) has been moved to its own plan: [`multi_transform_parallelization.md`](multi_transform_parallelization.md). It remains the highest-priority WaveletLM-native multi-perspective direction, but is architecturally distinct from the full-node multinodal techniques tracked here.

1. **Wavelet-domain combination of full nodes (§6)** — Combiner-only code change applied to the existing multinodal infrastructure; tests whether per-scale combination outperforms current logit averaging. Cheap to test.
2. **Deep Mutual Learning on multinodal (§4)** — Small code change; adds KL loss between cells. Cheap to test, known-positive technique.
3. **Sparse MoE on MLP (§1)** — Largest architectural lever available. One 17h run validates the direction.
4. **SWA within a single seed run (§2)** — Free; reuse late-training checkpoints from the 3-seed study.
5. **Orthogonality loss on multinodal cells (§5)** — Tiny code change; strengthens existing multinodal mode.
6. **Learned per-scale gating across nodes (§6)** — Follow-up to #1. Adds N×S learned scalars; tests whether nodes naturally specialize across scales.
7. **Git Re-Basin cross-seed weight merging (§2)** — Research-grade; potential novel finding about wavelet inductive bias and basin structure. Low compute; high intellectual yield.
8. **Ensemble distillation of multinodal → single-cell (§4)** — Gets ensemble quality at release-time single-model cost. Needs second training pass.
9. **Hard scale-specialization (§6)** — Follow-up to #6. Tests whether per-node compute can drop by skipping non-assigned scales.
10. **Capacity-aware routing (§6)** — Most ambitious of the §6 entries. Requires a small router and load signal aggregation; only worth pursuing if §6 #1 shows wavelet-domain combination is a real lever.
11. **Multi-basis lifting (§3)** — Blocked behind stable_parametrization validation.
12. **BranchyNet LM heads (§3)** — Only worth doing if we push to L=8+ at B200.

## Open questions

1. Does the wavelet inductive bias lead to more or less structure in the weight space across independent seeds? (Testable via Git Re-Basin + weight-correlation analysis.)
2. Would semantic embedding reintroduction (see [reincorporate_large_semantic_embedding.md](reincorporate_large_semantic_embedding.md)) reduce permutation freedom enough that naive cross-seed averaging starts working?
3. Is MoE-on-MLP or depth-scaling the better capacity lever at B200 scale? No ablation exists to answer this.
4. How does sparse MoE interact with WaveletLM's spectral mixer? Mostly independent (MLP is position-wise, mixer is sequence-wise), but worth verifying there are no surprising interactions.

## Not pursued

- **Naive model soup across 3-seed runs**: overclaimed earlier; for independent-init from-scratch runs, expected outcome is degradation, not improvement. Listed here to prevent future us from re-trying it.
- **Classical Product of Experts (Hinton 2002)**: intractable normalization for LMs; modern sparse MoE is the practical descendant.
- **Co-training**: no natural "different views" of language data.
