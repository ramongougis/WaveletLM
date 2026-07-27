# SALT — Scale-Addressed Lookup Table

> **Status: DESIGN (2026-07-26).** A post-hoc, frozen-weights correction store for WaveletLM.
> Sibling to [long_context_retrieval.md](long_context_retrieval.md) (wavelet-keyed kNN-LM /
> FSRR) — that plan targets *long-range factual reach*; SALT targets *targeted error repair*.
> They share machinery (frozen keys, FAISS index, interpolation) and should be built once.
> Grounded in Findings 16–19 of [interpretability.md](interpretability.md) — the design choices
> below are consequences of measurements, not preferences.

## What it is (and what it is NOT)

SALT stores, for contexts where the trained model **got the next token wrong**, the
**correct continuation** — keyed by the per-scale wavelet coefficients of that context.

**It does not store errors.** The error is the *selection criterion*; the *content* is ground
truth. (An earlier naming pass — "Error Retrieval", "Error Datastore" — got this backwards and
was rejected for exactly that reason. Keep the distinction in any write-up: we retrieve
corrections, indexed by where the model failed.)

At inference the model computes its own difficulty signal, and where that signal is high, it
consults SALT and interpolates the retrieved continuation with its parametric prediction.

## Why post-hoc construction dissolves the "erode vs. lose" tension

Ramon's framing: *during* training we can see which examples are hard but the representation
drifts under weight updates, eroding stored similarities; *after* training the representation is
stable but we've lost the reference to the original examples. "You either know the context
precisely or the example precisely, but not both."

**The tension is real for ONLINE memory** (AMB's within-sequence state; FwPKM's runtime updates)
**and dissolves for a post-hoc store:**

1. Train to convergence. **Freeze.**
2. One forward pass over the corpus. Frozen weights ⇒ every key lives in one stable
   representation space. **No erosion.**
3. The original examples are present — we are re-running them.
4. Labels are available (it is training data) ⇒ **exact NLL** ⇒ error-driven selection is exact,
   not proxied.

All three properties hold simultaneously. **No training is required for SALT** — it is entirely a
post-training procedure, which means it can be added to a released checkpoint, tested on the
checkpoints already on disk, and iterated in hours without a GPU. That fast-iteration property is
the single biggest practical advantage over AMB, and it should be protected in every design
decision below.

*Where the tension returns:* continual learning. Fine-tune the model and the store goes stale and
must be rebuilt. For a fixed release model, a non-issue.

## Storage: pruning is MANDATORY, not optional

Arithmetic for WT-103 (119.7M tokens), keying on s0 coefficients at C=512, fp16 = 1KB/key:

| regime | size | note |
|---|---|---|
| every token | **~122 GB** | ≈225× the 540MB corpus — not "3–10×" |
| top 10% by loss | ~12 GB | |
| **top 1% by loss** | **~1.2 GB** | fits in RAM; rebuilds in minutes |
| Pile (5B tokens), all | ~5 TB | |
| Pile, top 1% | ~50 GB | |

**Error-driven selection is what makes this tractable at all.** int8 key quantization buys another
4×; product quantization more.

**Disk is not the binding constraint — index build time and lookup latency are.** A 122GB index
must be built, held partly in RAM, and searched per token. The 1%-regime fits in RAM entirely and
rebuilds in minutes, which is precisely what preserves the fast-iteration advantage. Keeping
everything would forfeit the property that makes SALT worth building.

## Lookup design

### Partition first, rank second (Ramon, 2026-07-26)

1. **Partition by the exact last token** of the context.
2. **Within the partition, rank by scale-similarity.**

This is Katz-style backoff logic — condition on the most specific context with evidence — fused
with neural similarity, and it directly constrains the standard retrieval failure mode (*similar
context, different continuation*) because the previous token is the strongest single predictor of
the next. It also prunes the candidate set before any vector math.

### The coverage problem, and backoff

Top-1% of WT-103 ≈ 1.2M entries over a 50,257 vocab ≈ 24/token *on average*, but Zipfian in
practice. **Errors concentrate where prediction is hard, and prediction is hard after unusual
context — so the partitions we most need are the sparsest.** Mitigation is standard n-gram
backoff: exact last token → last-token *class* (POS or embedding cluster) → unconstrained.
Implement as a hierarchy with a per-level minimum-candidate threshold.

### Which scales to rank on — answered by measurement

**Rank on COARSE bands (s0/s1); the fine bands are redundant here.** Once the partition matches
the last token exactly, s7 adds nothing, because Findings 16–18 show s7 encodes *current-token
identity* (lag-1 ρ≈0.07, strictly shift-local, r≈0 with difficulty) — precisely the information
the partition already guarantees. The coarse bands carry integrated context (ρ≈0.41) and the
entropy-independent difficulty signal (Finding 19).
*Falsifiable:* if adding fine bands to the metric helps, this reasoning is wrong.

### Similarity metric

**Cosine, per-scale normalized.** Two reasons, both measured:
- Finding 17 showed coefficient *magnitude* tracks **surprise** — a *state* property, not a
  *content* property. Dot product would conflate "similar context" with "similarly difficult".
  (Magnitude-matching is a coherent *alternative* retrieval goal, worth testing separately.)
- **Normalize per scale BEFORE combining.** The U-shaped gain profile (Finding 1) means s0 and s7
  run loud (0.57, 0.47 mean |coeff|) while s4/s5 run quiet (0.24, 0.21). A naive concatenated
  cosine silently weights the match by that profile, 2–3×. This is the kind of detail that
  produces a mysterious null.

### Index structure — use IVF, not a scalar sort

A nested-quartile index over a combined scalar metric **will not work**: collapsing ~512
dimensions to one scalar destroys the geometry (points adjacent in the scalar are arbitrarily far
in the space). The correct form of that instinct — partition the space into cells, keep a table
of which cell to search — is an **inverted file index (IVF)**, implemented at billion-scale in
FAISS (IVF-PQ), which kNN-LM already uses. **Off-the-shelf; do not build.**
What stays genuinely ours is the **key definition** (per-scale coefficients, coarse-band, per-scale
normalized) and the **difficulty gate** — FAISS will index whatever vector we hand it.

## The gate: when to consult SALT

Retrieval is expensive; always-retrieve wastes most of it. **Finding 19** establishes the gate:

- `r(entropy, NLL) = +0.613` — entropy is the strong incumbent, and is label-free.
- **`partial r(flow, NLL | entropy) = +0.293` at layer 0, s0/s1** — coefficient flow carries
  difficulty signal entropy *misses*.
- **Early-exit capability:** entropy needs the FULL forward pass (final logits); **layer-0 flow
  needs one layer of ten**. A coarse-band L0 gate can trigger retrieval *before the forward pass
  completes* — something entropy structurally cannot do.
- **Combine flow with entropy, don't replace it.** Partial r 0.29 is ~8.5% of residual variance:
  real but modest, and noisy alone.
- **Built-in negative control:** s7 measured r≈+0.02 with NLL. Any gate keyed on s7 should fail.
  If it works, suspect the implementation.

## Honest risks

1. **Similar context ≠ same continuation.** The standard failure of retrieval-augmented LMs. Two
   near-identical representations routinely take different next tokens — that is most of what
   makes language hard. kNN-LM works not because retrieval is reliable but because it
   **interpolates softly**. *The interpolation weight is a safety mechanism, not a cosmetic dial.*
2. **Surface-form retrieval.** Nearest neighbours may match rhymes rather than reasons — exactly
   the risk if the key leans on fine bands (Finding 18). Mitigated by coarse-band keying; verify
   empirically.
3. **Representational capacity.** At 73M / 256-token context the coarse bands may not discriminate
   contexts needing different answers. A null here may be a property of the model, not the method
   — re-test at M2/M3 (C=2048) before concluding.
4. **The first key definition will probably fail.** Build every choice (partition strictness,
   backoff depth, scale weighting, cosine vs dot, gate threshold, interpolation weight) as a
   **flag**, not a commitment. The whole value of a post-hoc frozen design is cheap sweeps.

## Naming (settled 2026-07-26)

**SALT — Scale-Addressed Lookup Table.** "Addressed" because the scale coefficients *are* the
address (content-addressable memory is the established term for this structure); "Lookup Table"
because it is honest about the structure without asserting anything false about the contents.
Salt is also the classic agrarian *preservation* technology — how the harvest is kept usable
across seasons — added sparingly to a finished dish, which matches the interpolation weight.
Slots into the family: **SOW** the corpus → **REAP** the passages → **SALT** what is worth keeping.
*Rejected en route:* SEER/SEED with "Error Retrieval/Datastore" expansions (imply we store the
errors themselves — false); WEFT (hard-codes the wavelet assumption, does not survive a key
change); nested-quartile indexing (see IVF above).

## RESULT — rung 2 FAILED, and it invalidates the selection criterion (2026-07-26)

`salt_pilot.py` on Mini/D3: 299,776 build positions, top-1% kept (2,998 entries, nll>=11.25),
key = coarse bands s0+s1 at layer 0, per-scale normalized cosine, last-token partition.

| metric | value |
|---|---|
| coverage (partition non-empty) | 51.47% |
| top-1 \| covered | **0.00%** |
| oracle@20 \| covered | **0.08%** |
| **ORACLE CEILING (unconditional)** | **0.04%** |

**This does NOT falsify scale-similarity — the pilot could not test it.** Two structural faults:

**1. Median partition size = 1.** (1,111 distinct last-tokens; median 1, p90 2.) For a typical
covered position there is exactly ONE candidate, so "rank by scale-similarity" selects among a
single item. The similarity metric was never exercised.

**2. THE REAL FLAW — top-N%-by-loss selects for IRREDUCIBLE tokens, not learnable ones.**
Diagnostic on the same checkpoint: **42.2% of top-1% targets appear exactly ONCE in the corpus**;
median corpus frequency of high-loss targets is **2** versus **26** for all targets. Samples:
`' Employ'`(1), `' simulation'`(1), `' falsely'`(1), `' plausible'`(1), `'urg'`(1), `'ila'`(1).
A token appearing once **can never be retrieved usefully** — the context that produced it will
not recur. Selecting the highest-loss positions selects, almost by definition, the positions
where the correct answer is *unpredictable in principle*. We built a table of questions that
will never be asked again.

**Consequence for the design.** "Error-driven selection" was adopted as an elegant compression
(and it does solve the storage arithmetic). It is now clear it may be **incompatible with why
retrieval works at all**: kNN-LM stores *everything* and derives its gain from the aggregate
distribution over many neighbours, not from finding one exactly-correct match. Storing 1% and
doing exact top-k lookup replaced a soft, robust mechanism with a brittle one.

**Revised candidates for selection (all cheap to test on the same harness):**
- **Frequency-floored errors** — high loss AND target token frequency >= N. Directly excludes
  hapaxes; keeps errors on tokens that can recur.
- **Mid-band loss** — e.g. 60th-90th percentile rather than the extreme tail: excludes both the
  already-easy and the irreducible.
- **Recurrence-selected** — keep (context, next-token) pairs whose *pattern* appears more than
  once in the corpus; the most direct expression of "retrievable".
- **Store everything, compress differently** — abandon error-selection; get tractability from
  int8/PQ quantization instead, matching kNN-LM's actual mechanism. Most faithful, most storage.

**Rung-2 protocol note:** run the oracle ceiling BEFORE building any downstream machinery, and
report coverage, top-1, and oracle@k *separately* — the conflated number would have looked like
"the key is wrong" when the true cause was store construction. This is the ladder working.

## RESULT — the mechanism test, and the verdict (2026-07-26)

Rebuilt with the fixes from the failed pilot: **store everything** (`keep_pct=100`,
1,999,872 rows from 2M tokens, 4.10 GB fp16) instead of top-1%-by-loss. That fixed both
earlier faults — coverage 51% -> **99.15%**, median partition 1 -> **7**.

### The retrieval numbers, with the control

| metric | similarity | RANDOM-rank control | lift |
|---|---|---|---|
| top-1 \| covered | **19.64%** | 11.43% | **1.72x** |
| oracle@100 \| covered | 56.04% | 50.16% | 1.12x |
| repair (top-1 \| model wrong) | 3.58% | 2.53% | 1.41x |
| oracle@100 \| model wrong | 34.26% | 28.96% | 1.18x |

**Scale-similarity IS doing real work** — 1.72x over chance-within-partition. But the
decomposition is sobering: of the 19.64%, ~11.4 points come from the **last-token partition
alone** (a bigram statistic) and only ~8.2 from the similarity ranking.
**oracle@100 is confirmed as mostly bigram**: with p90 = 65 entries, ~90% of partitions are
smaller than 100, so "top-100" is just "the whole partition" and ranking never operates. Its
1.12x lift comes entirely from the ~10-13% of queries with large partitions.
**Reference point that reframes everything: D3's own top-1 is 45.50%** on the same positions,
versus retrieval's 19.64%. Retrieval is **2.3x worse than the model it was meant to help**, so
substitution was never viable; only selective consultation could work.

### The decisive test: kNN-LM interpolation — ALL COMBINATIONS WORSE

`--interp` sweep, `p_mix = (1-lam)*p_model + lam*p_retrieval`, weights `softmax(sim/tau)`,
baseline NLL 2.9448 nats on covered positions:

| | best | worst |
|---|---|---|
| dBPB | **+0.0054** (tau 0.05, lam 0.05) | +0.1303 (tau 1.0, lam 0.5) |
| dBPB \| model wrong | **+0.0018** | +0.1363 |

**Not one of the 25 (tau, lam) cells improved anything.** Three properties make this decisive
rather than a tuning failure:
1. **Monotonic in lam** (+0.0054 -> +0.0140 -> +0.0350 -> +0.1260 as lam goes 0.05 -> 0.50).
   More retrieval, more harm — the signature of the retrieved distribution being **pure noise**
   relative to the model. Any real signal would make small lam helpful.
2. **tau barely matters** (0.02 to 1.0 within ~0.0005), so it is not a sharpness problem.
3. **`dBPB | wrong` is also positive**, so even restricted to the model's errors — the only
   slice where SALT could ever help — retrieval degrades. That kills the selective-consultation
   fallback too.

### Verdict

**A 34% oracle cannot be converted into any gain. SALT is DONE on this key** (layer-0 coarse
bands s0+s1, cosine, per-scale normalized, last-token partition), per the pre-registered
decision rule. *(This verdict stands and is left visible: it is about the COSINE key. The
learned-metric escape hatch below was subsequently run and flipped the sign — see "Test A
RESULT" — but on a different key, so it does not overturn anything stated here.)* The oracle-ceiling rung did its job: the whole direction was falsified in about
ten minutes of eval, before any index, gate, or pipeline was built on top of it.

**What remains open (do not read this as "retrieval cannot work here"):**
- **Scale.** This is 1.7% of WT-103. Retrieval methods scale strongly with datastore size, and
  the failure could be neighbour sparsity rather than a wrong key. The full-corpus build is
  queued to eliminate that explanation rather than assume it.
- **Learned metric** (Ramon, 2026-07-26). If raw cosine in the native space fails, *learn* the
  mapping — a small secondary model from context-scales to the correct continuation. This is
  **boosting** (train model 2 on model 1's residuals), which works when the base learner
  underfits; D3 at 40 epochs is well-converged and its errors are 42% hapaxes, so expected
  gains are modest — but it is cheap, CPU-viable, and the bypass problem already has an answer
  in Finding 19's difficulty gate. Judge it the same way: does the GATED ENSEMBLE beat D3's BPB.
- *Rejected variant:* replacing the model wholesale with a small net over layer-0 scales.
  Layer-0 coefficients are an **invertible re-encoding of the input**, so predicting from them
  is exactly as hard as predicting from tokens; and Finding 15 puts the work in the depth
  (removing layers 1-9 costs +1.5063 dBPB, 8.3x super-additive). That proposal reduces to
  "train a smaller LM". The legitimate form is **distillation** against the big model's output
  distribution.

## Test A RESULT (2026-07-27): the learned corrector FLIPS THE SIGN

`salt_corrector.py` on the same D3 store (1,500,000 rows, key dim 1024; 1,350,000 train /
150,000 val). Corrector = 1.57M trainable params, `key -> MLP -> C -> [FROZEN tied embedding]
-> vocab`. Three epochs: val 5.3117 -> 5.0216 -> **4.8760 nats**, standalone top-1 **23.65%**
(vs lookup's 19.64% and D3's 45.50%). D3 baseline on these rows: **2.6946 nats**.

| gate | frac | lam | dBPB |
|---|---|---|---|
| ungated | 100% | 0.05 | +0.0057 |
| ungated | 100% | 0.80 | +0.2623 |
| nll>p50 | 50.0% | 0.05 | +0.0002 |
| nll>p75 | 25.0% | 0.10 | −0.0021 |
| **nll>p90** | **10.0%** | **0.20** | **−0.0038** |
| nll>p90 | 10.0% | 0.80 | +0.0050 |

**Lookup's best was +0.0054; the corrector's best is −0.0038** — a swing of 0.0092 against a
0.0010 noise floor. Two structural properties argue this is signal, not a lucky cell:
1. **Monotone in gate tightness**: ungated all-positive -> p50 ~neutral -> p75 negative -> p90
   most negative. Exactly Finding 19's prediction — value concentrates where the model is weak.
   Note this is the *opposite* of the lookup sweep, where even `dBPB | wrong` was positive.
2. **Smooth interior optimum in lam** at p90 (−0.0022, −0.0030, **−0.0038**, −0.0037, −0.0021,
   +0.0050 across lam 0.05 -> 0.80). A unimodal curve over six lam values *and* a monotone
   trend over four gates is not the shape selection noise takes.

**Mechanism — as hypothesized, not as designed.** The gain comes from *a distribution over
neighbours, not from one exactly-correct match*: the corrector's top-1 (23.65%) is still far
below D3's (45.50%), so it cannot be right more often — its diffuse-but-correctly-shaped
distribution is what helps when mixed at lam=0.2. The learned metric beat cosine, as predicted.

**THREE CAVEATS, AND THE FIRST IS LOAD-BEARING:**
1. **The gate peeks at the answer.** `gates` is built from `nll0 = Eva`, D3's NLL *on the true
   next token*, so `nll>p90` means "apply the corrector where the model turned out to be wrong."
   That is NOT computable at inference. The docstring calls it a stand-in for entropy, but
   entropy and NLL diverge precisely on confidently-wrong positions. **−0.0038 is an upper
   bound, not a deployable number.**
2. **(gate, lam) were selected on the same 150,000 rows they are scored on.** Best-of-24 against
   a 0.0010 floor is optimistic; needs a held-out third split for selection.
3. **Store-val subset, not the benchmark.** D3 scores 2.6946 nats here vs 3.0606 on the
   sliding-window eval — an easier slice, so dBPB need not transfer.

**What was actually bought:** the corrector's key is the position's OWN coarse bands, so this is
boosting on a decorrelated weak learner — an **ensembling** gain, not new information, and not
evidence about long-range retrieval. Real, but it caps the ceiling.

**Next rung — BUILT 2026-07-27: `tools/interpretability/salt_gate.py`** (CPU-only, no budget).
Kills all three caveats at once:
- **Realizable gates.** Entropy was not recoverable from the store — `sweep` computed the
  model's argmax but `savez` wrote only `K, N, L, E`. Fixed at source: `sweep` now also
  returns predictive entropy, and build mode persists `A` (argmax) and `H` (entropy), so no
  future run needs a re-sweep. `salt_gate.py` works with OLD stores too (it reads only `K`/`N`
  and re-sweeps for the eval set). Nine realizable gates — `entropy>pQ`, `corrconf>pQ`,
  `disagree`, `combo` — versus three `ORACLE nll>pQ` reported only as the ceiling.
- **SELECT/TEST split.** The (gate, lam) sweep runs on SELECT; the single winning config is
  scored ONCE on TEST. The oracle ceiling is scored on TEST too, so "how much the realizable
  gate gives up" is a measured number rather than an argument.
- **Fresh independent eval set.** Rather than reuse store rows (which the corrector trained
  on), it re-sweeps a held-out token range and applies `--min_context 128`, matching the
  sliding-window protocol — so dBPB is comparable to the BPB everything else is ranked by.

**PRE-REGISTERED DECISION RULE** (in the script, printed with the result; noise floor 0.0010):
| TEST dBPB on a REALIZABLE gate | verdict |
|---|---|
| <= −0.0020 | **SHIP** — SALT is a viable post-hoc release feature |
| −0.0020 to −0.0010 | real but marginal; NOT worth the release complexity |
| >= −0.0010 | the Test A gain lived in the ORACLE gate; SALT closes, record as negative |

Scoring core is self-tested (`--selftest`, passing): empty gate and lam=0 are exactly 0.0; a
perfect corrector scores −0.5790 and a worthless one +0.0714 (opposite signs, so the metric
discriminates); a difficulty-targeted gate beats a random gate at equal fraction (−0.2961 vs
−0.0607). Those had to pass before the metric was trusted with a release decision.

## The n-token partition variant (Ramon, 2026-07-27) — skip-n-gram dynamics

Proposal: hold **2, 3, or 4** tokens fixed instead of one, letting scale similarity match the
rest. **This cuts directly at the central diagnosis above**, and the arithmetic runs both ways:
- It WILL raise match accuracy — but the decomposition already says ~11.4 of the 19.64 points
  are the last-token partition (a bigram statistic) and only ~8.2 are the scale ranking. Fixing
  more tokens grows the term that is already dominant, so the method converges toward **being an
  n-gram table with a wavelet garnish**. A win at n=4 that is really a 5-gram model is not a
  result about wavelets, and per the unit-discipline rule it must not be reported as one.
- **Sparsity moves the opposite way.** At n=1 the store already has p90 = 65 entries/partition
  and median 7. Distinct-context count grows ~geometrically in n, so at n=3-4 most partitions go
  singleton or empty and there is nothing left to rank — the same failure that killed the
  top-1% pilot (median partition 1).

**The design that cannot be fooled** — measure both axes at once, n = 0, 1, 2, 3, 4:
| | what it isolates |
|---|---|
| top-1, scale-ranked within partition | the full method |
| top-1, RANDOM-ranked within partition | the pure n-gram contribution |
| **difference** | **the wavelet contribution, at each n** |
plus partition-size distribution (median, p90, % singleton) per n to find where sparsity bites.
n=0 (no partition at all) is the single most informative cell and is missing from every run so
far: it measures pure scale retrieval with the bigram effect removed. **If the scale-vs-random
gap is ~0 at every n, scale retrieval is decorative and the whole direction closes** — for the
corrector too, since it shares the key.

**True skip-grams are the more interesting variant.** Fixing a CONTIGUOUS suffix is what an
n-gram model does. Fixing NON-adjacent lags (e.g. {1,3} or {1,2,5}) is not, so it escapes the
"it's just an n-gram" critique and is the version genuinely worth testing. Same measurement
design; add lag-set as a third axis.

## Test ladder (all frozen, forward-only, no GPU)

1. Build the store on Mini/D3 at top-1%; verify size and build time match the arithmetic above.
2. Oracle ceiling: retrieve with the *true* next token available. If the ceiling is low, the key
   is wrong and no gate tuning will save it — fail fast here.
3. Real retrieval, coarse-band cosine, last-token partition + backoff. Measure ΔBPB vs D3.
4. Gate sweep: entropy alone vs flow alone vs combined; measure retrieval calls saved at equal ΔBPB.
5. Ablate the design choices one at a time (fine bands in/out, cosine vs dot, partition on/off).
