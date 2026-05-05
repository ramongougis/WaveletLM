# Old Compression / Stabilization Ideas (Archived)

These two designs were drafted from the [DeepSeek-V4 paper](https://huggingface.co/collections/deepseek-ai/deepseek-v4) and lived briefly in the README's Future Plans before being superseded.

The first (Adaptive N-Gram Recall + Tiered Mean-Pool Compression) was replaced by the simpler **Bisected-Block Context Extension** — the new design uses a flat half-and-half block layout (recent half uncompressed + past half uniformly mean-pooled) that maps cleanly onto the wavelet's natural binary decomposition. The tiered design's complexity (n-gram inverted index, fallback chain, multiple group sizes, transition tokens, no-overlap constraint) wasn't earning its keep against a uniform-compression alternative that the wavelet's level-1 split mirrors exactly.

The second (Sinkhorn-Knopp Parameter Stabilization) is parked here because Muon's orthogonalized update — already prioritized as #9 in the Future Plans — addresses the same spectral-norm-amplification failure mode that motivated the Sinkhorn-Knopp constraint. If Muon clears the L=11 NaN cliff, this section is moot. If Muon doesn't clear it, this is the next thing to try.

Both are kept here verbatim so the design rationale isn't lost.

---

## Adaptive N-Gram Recall + Tiered Mean-Pool Compression (DeepSeek-V4 CSA/HCA-Inspired)

**Source: [DeepSeek-V4 (DeepSeek-AI, 2026)](https://huggingface.co/collections/deepseek-ai/deepseek-v4).** DeepSeek-V4 introduces a hybrid attention scheme: Compressed Sparse Attention (CSA) compresses KV entries every `m=4` tokens, then a lightning indexer scores each query against compressed blocks and selects top-`k` for actual attention; Heavily Compressed Attention (HCA) skips the sparse selection and instead consolidates KV every `m'=128` tokens via weighted mean. Together they reduce 1M-context KV cache to ~10% of DeepSeek-V3.2 and inference FLOPs to ~27%. WaveletLM has no attention or KV cache, so neither mechanism transfers literally. What does transfer is the *structural* idea behind each: CSA is "look up immediately past relevant context," and HCA is "carry summarized history at coarser and coarser granularity." Recasting both for a flat, channel-simultaneous attention-free architecture gives the two-stage design below.

**Stage 1 — Adaptive n-gram recall (CSA-flavored).** Mandate `block_size = 256` as the canonical fine-scale block size (matching the prior headline benchmark regime). The first 1024 slots of the full input block are populated by **deterministic, training-free, hard top-k retrieval** of past 256-token windows surrounding occurrences of the current cursor's last n-gram suffix:

- Slots 1-256: the most recent 256 tokens, always (the immediate-context anchor).
- Slots 257-512: search the rest of the corpus *outside* the recent 256-block for the **most recent 3-gram match** of the current cursor's last 3 tokens. If found, copy the 256-token window centered on that match (positions 127 before through 128 after, or vice versa — block alignment doesn't matter, only that the matched n-gram sits centrally). If no 3-gram match, fall back to the most recent 2-gram match. If still no match, fall back to the most recent 1-gram (single-token) match. If even the 1-gram has never occurred before, fall back to the **next 256 most recent tokens** (slots -512 to -257 in chronological terms).
- Slots 513-768: same procedure, longest-suffix-first, but the search excludes any 256-block already included (no overlap with slots 1-512). 3-gram → 2-gram → 1-gram → most-recent-unincluded fallback.
- Slots 769-1024: same procedure once more.

The fallback chain is the recall scoring function, ordered by specificity: longer n-gram matches retrieve more topically-similar context, so we try them first; the "most recent unincluded" tail fills the budget when matches are sparse. Subword-token noise is partially mitigated by the n-gram fallback (a fragmented BPE word will rarely have a coincident 3-gram repeat that *isn't* a real recurrence of the surrounding word/phrase), and is further mitigable by preferring matches whose n-gram boundaries align with whitespace in the decoded stream (cheap to precompute at tokenization). A v2 of recall, post-headline, can replace literal n-gram match with **frozen-embedding cosine similarity** for semantic lookup at zero training cost — flagged as a follow-up.

This is the analog of CSA's lightning indexer + top-k selector, but with the scoring function fixed to "shares a recent suffix" rather than learned. That choice is deliberate: it removes a learned indexer's failure modes (the indexer head being undertrained early in training, divergence between indexer and main attention) and makes the recall purely a **data-loading transformation** that requires no model changes.

**Stage 2 — Tiered mean-pool compression (HCA-flavored).** Slots 1025-16384 carry summarized history at progressively coarser temporal granularity, each tier producing 256 pseudo-tokens via component-wise mean of `g`-token groups. Pseudo-tokens are **context-only inputs and are never label targets** — the loss is computed only over the genuine fine-resolution tokens in slots 1-1024, never over compressed slots. This is what makes the compression cost-free: we are not asking the model to reconstruct smoothed segments, only to *use* them as side information.

| Slots in input block | # pseudo-tokens | Group size `g` (per pseudo-token) | Tokens of past corpus condensed |
|--|--|--|--|
| 1025-1280     | 256 | 2  | 512    |
| 1281-1536     | 256 | 3  | 768    |
| 1537-1792     | 256 | 4  | 1024   |
| 1793-2048     | 256 | 5  | 1280   |
| 2049-2304 … 4096 | 8 × 256 | 6, 7, 8, 9, 10, 11, 12, 13 | 256·(6+…+13) = 19,456 |
| 4097-4352 … 8192 | 16 × 256 | 14, 15, …, 29 | 256·(14+…+29) = 88,064 |
| 8193-8448 … 16384 | 32 × 256 | 30, 31, …, 61 | 256·(30+…+61) = 372,736 |

**Total: 16,384 input slots covering 484,096 tokens of past corpus** (1024 fine + 3,584 + 19,456 + 88,064 + 372,736), a ~30× context extension over the bs=16384 headline at the same model footprint and step time. The +1-per-tier ramp keeps information density high — doubling group sizes (2, 4, 8, 16, …) was considered but allocates pseudo-tokens to extreme distances where the signal-to-noise of "this is what was happening 4B tokens ago" is dominated by corpus-distribution noise that the embedding already captures via learned token frequencies. Linear-step gives a denser, more usable curve. The ramp rate is exposed as a single config knob `recall_compression_ramp` (default `1.0`, allowing fractional values that repeat the same group size across multiple tiers) so the schedule can ablate without code changes.

**Why we trust the wavelet to handle this naturally.** The lifting wavelet is invariant to what kind of vector occupies each position — it just decomposes whatever sequence it's handed into multiscale coefficients. Compressed pseudo-tokens contribute to coarse-scale wavelet coefficients exactly like any other token, and the lifting predict/update weights at the coarsest scales should learn to weight them appropriately, since they carry summarized history relative to the fine-scale recent slots. The architecture's flat, channel-simultaneous mixing is what makes this work without further surgery — no per-position type embedding, no separate processing pathway. The only subtle concern is **boundary handling at tier seams** (slot 1024→1025, 2048→2049, …), where the lifting wavelet's reflection or zero-padding may smear unphysically across regimes. Mitigation: insert a learnable "transition token" at each tier boundary (5 transition tokens total: between fine recall and tier 1, and between each subsequent tier), at trivial parameter cost. Ablate with and without.

**No-overlap constraint (Stage 1 only).** Past-occurrence searches in slots 257-1024 must exclude any 256-token range already included in the current input block. Stage 2 pseudo-tokens are not subject to this constraint — they are *intended* to overlap chronologically with Stage 1 recall windows, since they contribute different (smoothed) information about overlapping spans of corpus history.

**Implementation.** New config keys:
- `recall_enabled` (bool)
- `recall_block_size` (int, must equal `block_size`, default 256, asserted at config-load time)
- `recall_n_fine_blocks` (int, default 4 → 1024 fine slots)
- `recall_max_ngram` (int, default 3)
- `recall_compression_tier_size` (int, default 256 pseudo-tokens per tier)
- `recall_compression_ramp` (float, default 1.0)
- `recall_transition_tokens` (bool, default True)

The recall lookup runs once per input sequence in the data loader, not in the model. It uses a precomputed inverted n-gram index over the training corpus (`{n-gram: sorted list of corpus positions}`), which costs ~O(corpus_size) memory but enables O(log corpus) lookup per recall slot. WikiText-103 has ~120M tokens, so a 3-gram inverted index fits in <8 GiB host RAM. Pseudo-token computation is a single `F.avg_pool1d` per tier on the embedded-and-pooled corpus past, also in the data loader — the model itself sees a normal `[batch, 16384, C]` tensor and is unmodified.

**Sweep.** 1-epoch screening at L=1 / levels=7 / `block_size=256` (the mandated value for this regime — distinct from the bs=16384 headline) with `recall_enabled=true`, against the same configuration with `recall_enabled=false` (which collapses to a 16384-token uncompressed window = current headline regime). Primary metric: BPB sliding on the held-out validation set. The recall-on configuration pays a one-time cost to build the inverted index but should not regress per-step throughput meaningfully. Pass criterion: BPB sliding within ±0.018 of the bs=16384 headline 1.0974, since the comparison is "same model footprint, same step time, 30× effective context." If recall ships within tolerance at 1 epoch, take to 5 epochs; if it improves on 1.0974, promote to default. The mandatory `block_size=256` constraint should also cause coarse-scale wavelet levels (the deepest 2-3 of `levels=7`) to allocate more weight to compressed-tier pseudo-tokens, which is verifiable post-hoc with [`tools/analyze_lifting.py`](../tools/analyze_lifting.py).

---

## Sinkhorn-Knopp Parameter Stabilization (DeepSeek-V4 mHC-Inspired)

**Source: [DeepSeek-V4 (DeepSeek-AI, 2026)](https://huggingface.co/collections/deepseek-ai/deepseek-v4) and the dedicated mHC paper ([Xie et al., 2026](https://arxiv.org/abs/2512.24880)).** DeepSeek-V4's Manifold-Constrained Hyper-Connections (mHC) constrain the residual mapping matrix `B_l` to the manifold of doubly stochastic matrices (the Birkhoff polytope) via the Sinkhorn-Knopp algorithm with `t_max = 20` iterations of alternating row/column normalization. The constraint guarantees `‖B_l‖₂ ≤ 1`, making the residual transformation **non-expansive** — directly preventing the gradient explosion that limits naive hyper-connection scaling, and ensuring stability under deep mHC stacks because the doubly-stochastic manifold is closed under multiplication.

WaveletLM has no residual stream to apply this to (every signal path is a wavelet decomposition + reconstruction, not an additive residual), but the same failure mode — repeated multiplication amplifying spectral norm across depth — bites us in two known places: the lifting predict/update cascade across `levels` (the L=11 NaN cliff that no stability fix has cleared) and the high-`low_rank` regime in the mixer (R1.5 / R2 / R3, where larger U·V^T corrections destabilize at peak LR). The lifting predict/update `Linear(C, C)` matrices are square — the same shape mHC constrains — and they cascade across levels exactly the way mHC's `B_l` cascades across residual depth.

**Target.** The `predict_nets[L].0/.3` and `update_nets[L].0/.3` `Linear(C, C)` matrices in `LiftingWaveletDecompose` (28 matrices total at `levels=7`, halved to 14 with `shared_lifting_weights=True`). Bounding each matrix's spectral norm to 1 via Sinkhorn-Knopp makes the entire cascade non-expansive in spectral norm, which is the structural property that lets mHC scale residual depth.

**Parameterization.** Following mHC's design, treat the constraint as a *correction* rather than the full weight: `W = α·I + β·B`, where `B` is constrained to be doubly stochastic via Sinkhorn-Knopp at every optimizer step (`t_max = 20`, matching DeepSeek-V4 exactly), and `α, β` are learnable scalars per matrix. This sidesteps the expressivity ceiling of pure doubly stochastic — every doubly-stochastic matrix is a convex combination of permutations (Birkhoff's theorem), which is too restrictive for a free-form `Linear(C, C)`. The `α·I + β·B` form lets the matrix be approximately the identity (lifting's prior under `lifting_init="haar"`) plus a non-expansive correction; with `α = 1, β = 0` at init, the constraint is inactive and the matrix is exactly the Haar identity, then `β` grows during training as the correction direction is learned. This mirrors mHC's own use of the constraint as an additive correction to a residual identity.

**Implementation.** New flag `lifting_sinkhorn_knopp_stab` (bool), default `False`. When enabled, after every optimizer step, the raw weight `W̃` of each lifting `Linear(C, C)` is decomposed as `α = mean(diag(W̃))`, `B = Sinkhorn-Knopp(exp(W̃ - α·I))` (column/row normalization, 20 iterations, exactly mHC's recipe), and the matrix is reconstructed as `α·I + β·B` with `β` set so the Frobenius norm of the correction matches `‖W̃ - α·I‖_F`. This is a strict generalization of the existing `stab_spectral_norm` flag (which applies a global spectral-norm bound but isn't activated anywhere) — Sinkhorn-Knopp is the same family of constraint, but per-matrix and tighter.

**Sweep.** 1-epoch screening at L=1 / levels=7 / bs=16384 with the flag toggled vs the current Adagrad reference 1.2361. Primary success criterion is **stability at `levels=11`** (which currently NaNs under every fix attempted including spectral norm, FF scaling, embed scaling, etc.) — if Sinkhorn-Knopp clears the L=11 cliff at acceptable BPB, that's the unblocker we have spent multiple sweeps trying to find. Secondary criterion: BPB at `levels=7` within ±0.018 of reference. The interaction with Muon (also non-expansive, but on updates rather than weights) is the natural follow-up — they may compose constructively (Sinkhorn-Knopp on weights + Muon on updates = doubly bounded cascade) or one may make the other redundant, and that test runs after both individual sweeps land.
