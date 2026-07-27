# Multi-Resolution MoE — experts over compressed context lengths

**Status:** proposed 2026-07-27 (Ramon). The **scale-ladder proxy is IMPLEMENTED and queued
as MLR1** (`block_moe_scale_ladder`, coarse-only nested ladder, CPU smoke-tested). The full
compressed-context version is still blocked on the decimating compressor — to be built during
the pod suspension; see Dependencies and the decimation resolution below.

**Decimation resolution (2026-07-27).** Finding 16 measured the decomposition to be
shift-EQUIVARIANT (odd/pow2 contrast 0.87x, no parity effect), and rules out a position
artifact for the channel census, Study 2 and Finding 17. Classical decimated wavelets are
shift-VARIANT, so decimating the block's INTERNAL transform would retract a property three
findings rest on. Resolution: decimate only in an **input-side compressor** feeding the
long-context experts, leaving the main path a-trous. The compressor may be close to free —
DWT coefficients are the SWT sampled every 2^k-th position, so it is plausibly a strided read
of the approximation band we already compute. CAVEAT: that equivalence is exact for a pure
dilated lifting, and the crawl (learned K=33 mixture over dilations) breaks the clean
correspondence — verify on CPU before relying on it. Separately noted: decimating the main
path would cut mixer compute from ~8T to ~2T positions, which makes it tempting; that is a
distinct experiment needing its own argument, not something to smuggle in via this work.

## The idea

Each expert in a mixture handles a *different compressed context length*: expert 0 sees the
last `block_size` tokens at full resolution, expert 1 the last `2·block_size` compressed 2:1,
expert 2 the last `4·block_size` compressed 4:1, and so on — every expert's input still fits
in `block_size` slots. Combine through the usual MoE router.

**Why it is worth taking seriously**, in order of weight:

1. **It attacks the model's real, acknowledged limitation.** The README states the context
   caveat in the same breath as every headline: 256 vs the 1024+ that comparison models use.
   This is the first proposal in the queue that addresses it *without attention*, and without
   paying full attention-like cost for the extra span.
2. **Attention-free by construction** — it is a data-routing scheme over the existing spectral
   block, so hard rule 9 is not in tension.
3. **It reuses machinery that already exists**: `block_moe_*` (implemented, smoke-tested,
   MOEA queued) and `decompose_bypass_cross_window` (already carries state across windows,
   which is exactly the cache a compressed long context needs).
4. **Ramon's own instinct on the compression operator is right.** Mean-pooling "smears" — it
   is an unweighted box filter, the worst anti-aliasing kernel available. The alternative he
   proposed ("start with its scale 1 as scale 0, 2 as 1, ...") is the correct one: the
   **wavelet approximation band already IS the compressed signal**, produced by a *learned*
   low-pass rather than a box filter. The compression operator this design needs is the one
   the model already computes.

## Dependencies and honest obstacles

**1. Decimation is not implemented, and this design needs it.** The current lifting is
*undecimated* (a-trous): every scale band is length `T`, which is precisely why C can be any
integer and why the crawl operates on dilations. So there is no free 2:1 downsample today.
Decimation is already planned post-release (and feeds the hybrid repo) — this proposal should
be sequenced *behind* it, not used to justify pulling it forward.

**2. Compression is not free at build time.** Seeing 2048 tokens requires touching 2048
tokens. The saving is in what the *expert* processes, not in what the *compressor* reads.
The escape is caching compressed representations across windows, which is what
`decompose_bypass_cross_window` already does — but that makes windows order-dependent, which
the SALT tooling had to disable for exactly this reason. Expect that tension to recur.

**3. Router collapse is the likely failure mode.** The full-resolution expert holds the most
precise information about the immediate past, which is what next-token prediction mostly
needs. A per-token router will be tempted to select it always. MOEA is already instrumented
for router collapse (aux magnitude, balanced = 2.0), so we will have direct evidence on
whether this router family collapses here *before* building this.

**4. The prior evidence on long context is discouraging — and confounded.** Two block=1024
runs exist at C=512/L=10/5ep, no-MLP:
| run | params | BPB |
|---|---|---|
| `logs/wikitext-103_2026-07-22_08-57-30` | 72.89M | 1.0654 |
| `logs/wikitext-103_2026-07-23_06-49-44` | 84.28M | 1.0481 |
Both are **worse than D0's 1.0436 at block=256**. The confound is real though: block 1024
forced MBS from 48 down to 12, moving the batch/LR operating point, so this is not a clean
context test. One encouraging detail inside it: at fixed block=1024, adding scales
(72.89M -> 84.28M) bought 1.0654 -> 1.0481, i.e. **more resolutions did help once the context
was long** — which is the closest thing to direct support this proposal has.

**5. Three independent signals say WT-103 is a poor place to test this.** The AMB work
concluded WT-103 is recall-light; SALT's retrieval failed and then its gate closed; and the
block-1024 runs above did not win. **PG-19 is the right dataset for long-context machinery** —
book-length prose where distant context actually pays, versus WT-103's encyclopedia articles.
P2 (PG-19 fully-spectral redo) is already queued and would give the baseline this needs.

## The cheap intermediate that needs NO decimation

Ramon's alternative framing is testable today and isolates the routing hypothesis from the
compression hypothesis:

> **expert k sees only scales >= k.**

No decimation, no longer context, no new data path — just a per-expert scale mask over the
bands the block already computes. It answers the load-bearing question underneath the whole
proposal: *does routing over resolution buy anything at all?* If experts specialising by
scale do not beat a single expert seeing all scales, the compressed-long-context version
inherits that failure and should not be built. If it does, the decimation work has a
measured reason to happen.

This is a small change to the block-MoE expert construction and costs one 5ep C=512 run
(~$7-9), and it should be judged on the same width-law-at-own-params rule as everything else,
not against D0 directly.

## Sequencing

1. **MOEA result** — does per-token block routing work here at all, or collapse?
2. **Scale-masked experts** (above) — does routing over *resolution* specifically pay?
3. **Decimation** lands (already planned).
4. **Then** this proposal becomes cheap to build, and should be run on **PG-19**, not WT-103.

Do not build 4 before 1-3 return. Each earlier rung can falsify the later ones at a fraction
of the cost.
