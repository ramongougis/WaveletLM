# New Compression Ideas

Forward-looking compression directions that haven't yet been screened. Companion file to [old_compression_ideas.md](old_compression_ideas.md), which archives superseded designs.

---

## Sparse Embedding with (p, q) Striding (Phantom-Token Number-Theoretic Compression)

**Source: developed in-conversation 2026-05-07** as a follow-up to the M-sweep / structural-variants compression work, after the lifting-cascade compression results suggested we should ask which other modules in WaveletLM admit similar sparsification. The 102.93M token-embedding block is the largest single non-cascade target (above MLP at 83.91M and mixer at 59.11M). Standard approaches — ALBERT-style factorization, vocab pruning, hash embeddings — exist; this is a number-theoretic alternative that preserves the (N × C) embedding shape exactly while sparsifying it via a deterministic 1D walk over the flattened tensor.

**Scheme.** Flatten the (N × C) embedding to a 1D tensor of length N·C and walk it with alternating step sizes p and q: `0, p, p+q, 2p+q, 2p+2q, 3p+2q, 3p+3q, ...`. Density = 2/(p+q). Selected positions are kept; everything else masked to zero (or, equivalently, never allocated). Decoding to (row, col) at access time is `idx → (idx // C, idx % C)`. The scheme is content-blind, deterministic, and has O(1) metadata cost — store (p, q, N, C) and the mask is reconstructable.

**Phantom-token trick.** Vocab N = 50,257 (GPT-2 BPE) is coprime to C = 2048 = 2¹¹, so the only common divisor is 1 — eliminating most of the (p, q) design space, since q must divide both. Fix this by padding N to N' > N where N' shares a useful divisor with C. The padded rows ("phantom tokens") never appear in input (no embedding lookup hits them) and are masked to -∞ in output logits (no gradient, no sampling).

Crucially, phantom tokens are a **numerology device only** — at load time, compute the (p, q) walk over the conceptual (N' × C) grid, filter to indices where `row < N`, and only allocate those positions. Phantom rows are never materialized in storage / VRAM / optimizer state. Zero runtime cost, zero memory cost. The trick is just standard vocab-padding for tensor-core alignment (NVIDIA recommends padding to multiples of 64–128 anyway), repurposed for number-theoretic convenience.

**Requirements.**
1. `q | C` and `q | N'` — q is a common divisor of C and N'. Gives macrocell tiling and 2D regularity. (Not strictly necessary for the 1D walk to work, but provides a clean (N'/q × C/q) macrocell grid where each cell is q × q. At q = √C, macrocells are square — minimum perimeter for given area, balanced 2D coverage, aligned with Monarch / butterfly factorization philosophy.)
2. `p ∤ C` and `p ∤ N'` — p does not divide C or N'. Avoids degenerate column-stripe patterns where every row selects the same columns.
3. `p, q > 1` — both steps must be non-trivial.

**Selection algorithm.** Trivially fast — O(τ(C)) where τ(2048) = 12. Iterate divisors of C; for each candidate q, compute p = round(2/density) - q and N' = q·⌈N/q⌉; check both non-divisibility conditions. Three modes for choosing among valid candidates:

```python
def find_pq(C, N, density, mode="structural"):
    s = round(2 / density)
    divisors = sorted(d for d in divisors_of(C) if 1 < d < s)
    valid = []
    for q in divisors:
        p = s - q
        if p <= 1 or C % p == 0: continue
        N_prime = q * ((N + q - 1) // q)
        if N_prime % p == 0: continue
        valid.append((p, q, N_prime, N_prime - N))
    if not valid: return None
    if mode == "smallest_q":  return valid[0]                                   # storage-optimal
    if mode == "structural":  return min(valid, key=lambda c: abs(c[1] - C**0.5))  # sqrt(C)
    if mode == "budget":      return max((c for c in valid if c[3] <= budget), key=lambda c: c[1])
```

1. `smallest_q` (storage-optimal): minimal phantom rows; behaves close to pure stride-p when q is small.
2. `structural` (recommended default, **q ≈ √C**): square macrocells, aligns with Monarch / butterfly philosophy (already empirically validated in the lifting compression), tensor-core-friendly block size, balanced 2D coverage.
3. `budget` (largest q under phantom-row budget): maximizes 2D structure subject to phantom overhead cap.

At high densities (d ≥ 10%), `s = 2/d ≤ 20` constrains q tightly, so structural and smallest-q modes converge to similar small-q picks. The √C heuristic only meaningfully diverges from smallest-q at d ≤ 1%, where q can roam up to ~64 freely.

**Worked picks for C = 2048, N = 50,257.**

| Target d | s = p+q | smallest-q (p, q, N', phantom rows) | structural √C (p, q, N', phantom rows) |
|---|---|---|---|
| 50% | 4 | (2, 2, 50258, 1) | (2, 2, 50258, 1) — only valid candidate |
| 20% | 10 | (8, 2, 50258, 1) | (2, 8, 50264, 7) |
| 10% | 20 | (18, 2, 50258, 1) | (12, 8, 50264, 7) |
| 5% | 40 | (38, 2, 50258, 1) | (24, 16, 50272, 15) |
| 1% | 200 | (198, 2, 50258, 1) | (168, 32, 50272, 15) |
| 0.1% | 2000 | (1998, 2, 50258, 1) | (1968, 32, 50272, 15) |

For d = 10%: q = 4 fails (p = 16 divides 2048); q = 16 fails (p = 4 divides 2048). The valid candidates are q ∈ {2, 8}.

**Cognitive / linguistic frame.** The scheme has a clean analogy to relational / distributional theories of meaning: each token has a partial set of features (not all C dims), and inter-token semantics emerge from the *graph of feature-overlap relations* rather than from any single dense feature vector. When two tokens share zero features directly, the model bridges through intermediates — a "hidden conceptual layer" of intermediate tokens — analogous to multi-hop reasoning in knowledge graphs (TransE / RotatE style).

At d = 10% with C = 2048, expected feature overlap between random pairs is d²·C ≈ 20 features → 2-hop bridging essentially guaranteed by Poisson tail (P[overlap = 0] is vanishingly small at λ = 20). Density floor below which the scheme breaks down semantically: roughly d ≈ 0.1% (E[overlap] drops to 0.002, 2-hop bridging starts to fail). Non-uniform per-row capacity (some tokens get slightly fewer or more selected entries depending on (p, q) alignment near phantom rows) is acceptable because the hypothesis is that the model handles non-uniformity through its inter-token graph — already evinced empirically by Zipfian rare-token learning in standard LMs.

**Planned ablation.** Compare (p=18, q=2) vs (p=12, q=8) at d = 10% — test whether macrocell structure (q > 2) provides any BPB benefit over near-stride behavior (q = 2). If no significant difference, default to smallest_q for simplicity. If structural q wins, validate at d = 5% and d = 1% with √C-mode picks. Sweep also versus a `random_topk` content-blind control at matched density to measure what the deterministic structure actually buys (if anything) over uniform random sparsity.

**Storage win.** Mask reconstruction at load time costs ~10 lines of code and (N·C) / 8 bytes of bit-packed mask if materialized — but with the phantom-token "never allocate" approach, only the selected real-token positions are stored as a flat array of shape (real_selected_count,) plus the (p, q, N) tuple. For C = 2048, N = 50,257, d = 10%: ~10.3M parameters stored vs the full 102.93M of an uncompressed embedding — a 90% reduction at the embedding tier, on top of whatever compression the lifting / MLP / mixer tiers achieve via the M-sweep / structural-variant work.

---

## MLP Structural Compression (Three-Way Connectivity Comparison)

**Source: developed in-conversation 2026-05-07** as the natural follow-up to the embedding (p, q) work. The MLP is the second-largest single component (83.91M @ E=10, 167.82M @ E=20). Three structural variants are tested at five density points (25%, 12.5%, 6.25%, 3.125%, 1.5625%) for a clean three-way connectivity comparison.

**Three variants** applied to the MLP weight matrices W1 (C, E·C) and W2 (E·C, C):

1. **Tiled banded** — view W1 as E concatenated `(C, C)` blocks left-to-right; in each block apply a bilateral band of width W around the block's main diagonal. Each output expansion neuron sees its ±W-neighborhood of input dims. Per-block density `(2W+1)/C`, identical to BAND on the lifting matrices. The user's "diagonals are included" definition: the (c, c+k·C) entries for k ∈ {0, ..., E-1} are the main diagonals of the E tiled blocks, automatically captured by the per-block bilateral band.

2. **Tiled block-diagonal** — same per-block view, but with block-of-blocks pattern of size b. Per-block density `b/C`. Each output "expansion group" sees only its own input group — an architecturally clean grouped-MLP / channel-grouped feedforward interpretation, well-studied in the convolutional-network literature (grouped convolutions, depthwise-separable convs).

3. **(p, q) striding** — single 1D walk over the flattened weight tensor with alternating step sizes p and q. **No phantom-token padding needed** because `gcd(C, E·C) = C` (since `C | E·C`), so any divisor of C is automatically a common divisor. Same `find_pq` algorithm as the embedding scheme, with the substitution `N → E·C` and `C → C`. The same (p, q) tuple is valid for both W1 and W2 (with transpose for W2's (C, E·C) shape).

**Worked candidates for C=2048, E=10 or E=20 (q | C and p ∤ C, p ∤ E·C):**

| Target d | s = p+q | Structural (p, q) | Density |
|---|---|---|---|
| 25% | 8 | (6, 2) — only valid | 25.0% |
| 12.5% | 16 | (12, 4) | 12.5% |
| 6.25% | 32 | (24, 8) | 6.25% |
| 3.125% | 64 | (48, 16) | 3.125% |
| 1.5625% | 128 | (96, 32) | 1.5625% |

**Per-block density equivalence across structures.** At density d, the three structures are tuned to produce the same total parameter count: BAND with bandwidth `W ≈ d·C/2`, BD with block_size `b = d·C`, (p, q) at density `d`. Total W1 + W2 effective params ≈ `2 · d · E · C²`. At C=2048, E=10, d=12.5%: ~10.5M params per matrix, ~21M for the MLP block (vs uncompressed 84M).

**Hypothesis priors** from the lifting empirical data (BAND > BD by ~12 pp at matched density):
- BAND likely wins on MLP at most densities, but the MLP nonlinearity in the middle changes the calculus — BAND's "soft" cross-channel bleeding may matter more or less.
- BD has a cleaner architectural story (grouped MLP / depthwise-conv-like) and may win at production densities.
- (p, q) is the wild card — global walk vs local band vs grouped block. Matches lifting's `random_topk` family as a content-blind structural baseline.

**Implementation.** Single new helper `make_mlp_mask(in_features, out_features, structure, ...)` in `tools/lifting_constraints.py`, wrapping the existing `make_structural_mask` (for tiled banded / block_diagonal) and the embedding scheme's `find_pq` / `make_pq_mask` (for pq_strided). Plumbed through `model.py` at the `FeedForward` construction site via five new config flags: `mlp_offdiag_structure`, `mlp_block_size`, `mlp_band_width`, `mlp_pq_density`, `mlp_pq_mode`. 15-run ablation block in `runs.sh` section 8.
