# Substitution LM — Working Notes

Design reference, decision log, and result tracker for the substitution-based language
model described in `substitution_lm_design.pdf`.

---

## Overview

A language model grounded in the logical principle of substitution: predict what n-gram
can legally follow or replace another, given mutual semantic property compatibility.
Rather than a dense end-to-end neural model, the system builds an explicit weighted DAG
over linguistically-motivated n-gram nodes, with property tables attached to each node
for logical consistency checking.

Corpus: WikiText-103. Runs alongside or between existing WaveletLM ablations on the A5000.

---

## Pipeline (MVP)

```
Raw corpus (WikiText-103)
    │
    ▼
1.  [SpaCy parse]  en_core_web_lg — full dependency parse + POS + NP chunks
    │
    ▼
2.  [Collocation extraction]  PMI over bigrams/trigrams → atomic phrase nodes
    │
    ▼
3.  [NP n-gram extraction]  nested noun phrases up to length 5, freq ≥ 5
    │
    ▼
4.  [DAG construction]  co-occurrence edges within clause window → weighted adjacency
    │
    ▼
5.  [Property lookup]  ConceptNet assertions → binary property vectors per node
    │
    ▼
6.  [Consistency filter]  prune / down-weight edges where property tables conflict
    │
    ▼
7.  [Prediction]  three mechanisms evaluated side-by-side (see below)
    │
    ▼
8.  [Evaluation]  PPL, BPB, SVDR on WikiText-103 test set
```

---

## Design Decisions

### Decisions Made

| Decision | Choice | Rationale | Alternatives considered |
|---|---|---|---|
| Parser | SpaCy `en_core_web_lg` | Fast (batched), good NP chunking, Python-native | Stanford CoreNLP (slower, more accurate); skip parsing (loses NP structure) |
| Max n-gram length | 5 | Covers most useful combinations; 10 causes memory explosion at 103M tokens | 3 (too short for NPs), 10 (300–500M unique nodes, 10–16 GB) |
| Frequency threshold | ≥ 5 occurrences | Prunes ~75% of hapax; keeps node count to ~30–50M | ≥ 2 (more coverage, more noise); ≥ 10 (aggressive, misses rare but valid NPs) |
| DAG backend | Plain Python dicts | Simplest, no heavy dependency, fast for prototype | NetworkX (richer API, slower at scale); scipy sparse (efficient but less readable) |
| Property source | ConceptNet lookup | ~300K English concepts, direct binary assertions, no training needed | MLP over GloVe (better coverage but needs labeled data); WordNet (less rich) |
| Word sense disambiguation | Deferred (Phase 2) | Adds 1.5–3h preprocessing; homonym ambiguity acceptable for MVP | DistilBERT-based WSD; coarse 3–5 sense inventory via WordNet synsets |
| MLP property classifier | Deferred (Phase 2) | ConceptNet coverage sufficient for MVP; MLP adds training complexity | 2–3 layer MLP on mean-pooled GloVe → binary property vector |
| Prediction mechanism | All 3 implemented | Each mechanism is cheap to add once DAG is built; empirical comparison beats a priori choice | See Prediction Mechanisms section below |

### Deferred to Phase 2

- **Word sense disambiguation** — surface token IDs only in MVP; distinct sense IDs planned
  once the pipeline is validated. WSD will use clausal context already parsed in step 1.
- **MLP property classifier** — replaces ConceptNet lookup for n-grams not in ConceptNet.
  Input: mean-pooled GloVe (300-dim); output: soft binary property vector; 2–3 layers,
  256–512 hidden units; trained on ConceptNet + crowd annotation.
- **Exception / instance tables** — per-node property overrides for idioms, metaphors,
  domain-specific usages. Candidates surfaced automatically where MLP prediction conflicts
  with observed corpus context.
- **Optimisation beyond counting** — iterative rule induction, default logic chaining.

---

## Prediction Mechanisms

Three mechanisms run in parallel for every evaluation. A single DAG and property table
are shared; only the scoring function differs.

### A — Weighted Edge Walk

Given the current context as a set of active n-gram nodes, retrieve all outgoing edges,
filter to those whose target node passes property compatibility with all active context
nodes, normalise remaining edge weights into a probability distribution, then sample or
rank.

```
P(next | context) ∝ Σ_{n ∈ context} w(n → next) × compat(n, next)
```

- Fastest at inference; most similar to classical n-gram LM
- Expected to dominate at small dataset sizes where edge weights are reliable

### B — Aggregated Vote

All context nodes that share any edge toward a candidate next node contribute a vote
weighted by both edge weight and their own activation strength. Property conflicts between
any context node and the candidate impose a multiplicative penalty rather than hard
pruning.

```
score(next | context) = Σ_{n ∈ context} activation(n) × w(n → next) × Π compat(n, next)
```

- More expressive; captures multi-node context agreement
- Expected to improve relative to A with larger corpus / more repeated observations

### C — Hybrid Re-rank

Generate top-K candidates from Mechanism A (fast baseline), then re-rank by summing
property compatibility scores from all context nodes. Combines A's speed with richer
context signal.

- Easiest to tune (K is a single hyperparameter)
- Useful diagnostic: if re-ranking doesn't help, property tables aren't adding signal

---

## Evaluation Metrics

All metrics computed on the WikiText-103 **test set** (245K tokens).

| Metric | Symbol | Definition | Notes |
|---|---|---|---|
| Perplexity | PPL | exp(mean cross-entropy in nats) | Standard LM metric; lower is better |
| Bits per byte | BPB | CE_nats × log₂(e) / avg_bytes_per_token | Comparable across tokenisations |
| Selectional violation detection | SVDR | F1 on anomalous vs. normal sentence pairs | Tests whether property tables add signal beyond co-occurrence |

**SVDR test set construction:** 200 normal sentences from WT103 test set paired with
hand-crafted or automatically perturbed anomalous versions (subject–verb selectional
violation, e.g. "The computer swam"). Consistency checker scores each; threshold tuned
on 50-pair dev set. Reports precision, recall, F1.

---

## File Structure

```
tangent/substitution_lm/
├── notes.md                  ← this file
├── substitution_lm_design.pdf
├── pipeline/
│   ├── 01_parse.py           parse corpus → SpaCy docs (streamed, not stored)
│   ├── 02_extract_ngrams.py  collocation + NP n-gram extraction → node table
│   ├── 03_build_dag.py       clause co-occurrence → edge table
│   ├── 04_property_lookup.py ConceptNet → per-node property vectors
│   └── 05_build_model.py     assemble DAG + properties → model object
├── predict/
│   ├── mechanism_a.py        weighted edge walk
│   ├── mechanism_b.py        aggregated vote
│   └── mechanism_c.py        hybrid re-rank
├── eval/
│   ├── ppl_bpb.py            PPL + BPB on WT103 test set
│   └── svdr.py               selectional violation detection
├── data/                     ← gitignored (large generated artefacts)
└── cache/                    ← gitignored (SpaCy parse cache, intermediate tables)
```

---

## Results

Fill as runs complete. All three mechanisms share the same DAG and property tables;
only the prediction function differs.

### Build Stats

| Stat | Value |
|---|---|
| Corpus word-tokens (train, step 2) | 84,244,939 |
| Collocations (PMI ≥ 3.0, freq ≥ 10) | 235,584 |
| Unique nodes (freq ≥ 5 filter) | 1,325,573 |
| Unique edges (after min-count=2, fan-out cap=500) | 25,647,341 |
| ConceptNet coverage | 0.0% (properties.pkl built before CSV arrived; re-run step 4) |
| Parse time (step 1, CPU) | 10,827s ≈ 3h 0min |
| N-gram extraction (step 2) | 844s ≈ 14min |
| DAG build (step 3) | 522s ≈ 8.7min |
| Model assembly (step 5) | 41s |
| Total pipeline wall time | ~3h 24min |
| Test set size | 235,821 words (1.7% UNK) |

### Evaluation Results (Phase 1 — no ConceptNet)

| Mechanism | PPL ↓ | BPB ↓ | SVDR F1 ↑ | Notes |
|---|---|---|---|---|
| A — Weighted edge walk | 4337.53 | 2.7768 | — | No ConceptNet → compat always 1.0 |
| B — Aggregated vote | 4337.53 | 2.7768 | — | Identical to A; contradicted set never populated |
| C — Hybrid re-rank (K=50) | 3939.28 | **2.7449** | — | Top-50 re-normalisation concentrates mass; best mechanism |
| KenLM 5-gram (reference) | — | — | — | queued |

**Findings:**
- **A = B** with 0% ConceptNet coverage — expected. B's multiplicative contradiction penalty never fires when all property dicts are empty. The two mechanisms are provably identical in this regime.
- **C beats A/B by Δ −0.032 BPB** even without property signal. Restricting to top-50 candidates from A and re-normalising concentrates probability mass on likely successors; the long tail of unlikely co-occurrence edges is discarded.
- **1.7% UNK** — excellent vocabulary coverage from the 1.3M-node table. The remaining 1.7% are rare proper nouns, numbers, and punctuation variants not seen ≥5× in training.
- **Next step:** delete `data/properties.pkl` and `data/model.pkl`, re-run steps 4–5 with the ConceptNet CSV now present, re-evaluate. Property-based filtering should help mechanisms B and C most.

### Evaluation Results (Phase 2 — with ConceptNet, 9.1% node coverage)

Original Phase 2 results below were collected with three implementation bugs
present (see "Phase 2a — eval fixes" below). Numbers re-collected after the
fix supersede these.

| Mechanism | PPL ↓ | BPB ↓ | SVDR R | Δ BPB vs Phase 1 | Notes |
|---|---|---|---|---|---|
| A — Weighted edge walk | 4340.75 | 2.7771 | 0.620 | +0.0003 | pre-fix |
| B — Aggregated vote | 4340.54 | 2.7771 | 0.620 | +0.0003 | pre-fix |
| C — Hybrid re-rank (K=50) | 4039.03 | 2.7532 | 0.540 | +0.0083 | pre-fix |

**Bugs identified (2026-05-25, after Opus 4.7 review):**
1. **`compatibility()` was structurally wrong.** It intersected `ctx.NotCapableOf` with `cand.CapableOf` (both as lists of ConceptNet concept words). For a contradiction to fire, the *same concept word* needed to appear in both lists — e.g. `rocks.NotCapableOf=["swim"]` and `swim.CapableOf=["swim"]`. The second never happens (swim's CapableOf is "move through water", etc.), so contradictions almost never fired. ConceptNet's 9.1% coverage was effectively inert. Fixed: the function now takes `(ctx_props, cand_word: str)` and checks whether `cand_word` literally appears in `ctx.NotCapableOf` / `CapableOf` / `HasProperty` / `RelatedTo`. This is the correct polarity for "is this candidate something the context permits?"
2. **Lemma vs surface-form mismatch.** Node table was keyed by spaCy lemma (`02_extract_ngrams`), but eval read raw surface forms — so "swam"/"is"/"fishes" never matched their nodes "swim"/"be"/"fish" and silently became UNK. Fixed: eval now lemmatises via the same spaCy pipeline (`en_core_web_lg`, parser/ner off) before lookup.
3. **SVDR P=1.0 is structural, not informative.** Paired eval cannot produce false positives; the meaningful metric is recall = pair-classification accuracy. Output now leads with `accuracy=` and labels P/R/F1 as derived.

### Evaluation Results (Phase 2a — with eval fixes)

| Mechanism | PPL ↓ | BPB ↓ | SVDR acc | Δ BPB vs Phase 2 | Notes |
|---|---|---|---|---|---|
| A — Weighted edge walk | 1414.98 | **2.1403** | **0.730** | **−0.6368** | Major lift from lemmatisation |
| B — Aggregated vote | 1414.97 | **2.1403** | **0.730** | **−0.6368** | A ≈ B even with fixed `compatibility()` |
| C — Hybrid re-rank (K=50) | 1423.57 | 2.1421 | 0.430 | −0.6111 | SVDR *regressed* (−0.11) — over-rerank? |
| KenLM 5-gram (reference) | — | — | — | — | queued |

### Evaluation Results (Phase 2b — extended compatibility: 8 positive relations)

`compatibility()` now consults all 8 positive ConceptNet relations (was 3): added `IsA`, `UsedFor`, `Causes`, `ReceivesAction`, `AtLocation` alongside `CapableOf`, `HasProperty`, `RelatedTo`. Boost magnitudes 1.2–1.5, multiplicative across matched relations.

| Mechanism | PPL ↓ | BPB ↓ | SVDR acc | Δ BPB vs 2a | Notes |
|---|---|---|---|---|---|
| A — Weighted edge walk | 1414.25 | 2.1402 | 0.730 | −0.0001 | Diagnostic null result |
| B — Aggregated vote | 1414.24 | 2.1402 | 0.730 | −0.0001 | A still ≈ B; contradictions unchanged |
| C — Hybrid re-rank (K=50) | 1422.67 | 2.1419 | 0.430 | −0.0002 | Same pattern |

**Diagnostic outcome:** the 5 added positive relations contribute essentially zero aggregate lift. This confirms coverage (9.1% of nodes) is the binding constraint, not relation richness. The new relations require the *literal next word* to appear in the context's `IsA` / `UsedFor` / `AtLocation` / etc. lists, which almost never happens in flowing text — e.g. `fish.IsA = ["animal", "vertebrate"]` rarely matches whatever word actually follows "fish" in a sentence. Validates moving to MLP property imputation as the next step rather than further hand-curating relations.

### Evaluation Results (Phase 2c — CONTEXT_SIZE=10, HYBRID_K=500)

| Mechanism | PPL ↓ | BPB ↓ | SVDR acc | Δ BPB vs 2b | Notes |
|---|---|---|---|---|---|
| A — Weighted edge walk | 1401.08 | 2.1374 | 0.730 | −0.0028 | Mild lift from wider context |
| B — Aggregated vote | 1401.07 | 2.1374 | 0.730 | −0.0028 | A still ≈ B to 0.01 PPL |
| C — Hybrid re-rank (K=500) | **1337.43** | **2.1237** | 0.710 | **−0.0182** | Reversal: now best on PPL/BPB |

**Key finding:** Mechanism C went from worst-PPL / worst-SVDR (0.43) to best-PPL / near-best-SVDR (0.71) from a single config change. Top-50 was throwing real candidates off the cliff into `UNK_LOGPROB = −15`; expanding to top-500 reclaims them. 2× runtime cost per mechanism (7:25 vs 3:45 per chunk) is the price of 10× more rerank work — acceptable for the quality jump.

A vs B still in lockstep (1401.08 vs 1401.07) — contradictions essentially never fire with ConceptNet-only properties at 9.1% node coverage. This is precisely where MLP imputation is expected to break the tie: B's `contradicted` set should populate non-trivially once `NotCapableOf` predictions exist for the 90.9% gap.

Tokens: 200,381 (down from 235,821 — spaCy's POS filter removes 15% more than the old `=-<>` heuristic). UNK: 1.1% (down from 1.7%).

**Findings:**
- **Lemmatisation was almost the entire win.** A −0.64 BPB drop is bigger than the entire remaining gap to KenLM (~0.84). Inflected verbs and irregulars previously falling to UNK=−15 log₂ now score against the real lemma node. Total per-token CE dropped from ~12.5 bits to ~9.6 bits.
- **A ≈ B to 4 decimals.** Even with the corrected compatibility function, hard contradictions (`cand_word ∈ ctx.NotCapableOf`) fire so rarely that B's `contradicted` set barely populates. The 0.01 PPL gap between A and B suggests contradiction fires for <0.01% of (ctx, cand) pairs across the whole test set. ConceptNet's `NotCapableOf` is simply too sparse for this scoring approach to bite at WT103 scale.
- **C regressed on SVDR (0.540 → 0.430).** With stronger compat boosts (1.5x for CapableOf hits, 1.3x for HasProperty), C's re-ranking now actively shifts the top-50 ordering — but in directions that don't align with anomaly detection. A candidate that matches one context node's CapableOf gets a ~10% avg-compat boost, which can override A's base ordering. For SVDR pairs where the anomalous subject happens to match some loose CapableOf entry, C now mis-ranks. C's structural weakness (top-50 restriction) compounds with this.
- **SVDR accuracy up 0.62 → 0.73 (A/B).** Part of this is lemmatisation alone (more accurate word-level scoring for inflected forms in the pairs). Some is likely ConceptNet contribution on common verbs (eat, fly, swim, etc.) where CapableOf/NotCapableOf entries now correctly modulate scores. Hard to isolate without an ablation.
- **Headroom toward KenLM (~1.3 BPB) is 0.84 BPB.** Most of that gap is fundamental — KN smoothing and continuous interpolation aren't reachable with hard property filtering. Realistic Phase 2 ceiling is probably 1.8–2.0 BPB with MLP + WSD + exception tables.

### Phase 2 Additions (planned)

| Variant | Change vs. MVP | PPL ↓ | BPB ↓ | SVDR acc | Notes |
|---|---|---|---|---|---|
| + WSD (coarse) | Sense-disambiguated token IDs | — | — | — | |
| + WSD (fine) | WordNet synset IDs | — | — | — | |
| + MLP properties | Imputed ConceptNet via linear probe on spaCy vectors | — | — | — | implemented; awaiting run |
| + Exception tables | Manual + auto-detected overrides | — | — | — | |

**MLP property classifier (step 6)** — Linear probe (`Linear(300, K)`) over
spaCy `en_core_web_lg` word vectors, BCE loss with per-class `pos_weight`
capped at 100. Per-relation top-N% target word cutoff plus inference-time
confidence threshold (both configurable). ConceptNet entries override MLP
imputations where present (Reiter exception-override). To enable:
```
# config.py
MLP_PROPERTIES_ENABLED = True
# defaults: COVERAGE_PCT=80, CONFIDENCE=0.7, LAYERS=0 (linear probe)
```
```
# pod
rm tangent/substitution_lm/data/mlp_properties.pkl tangent/substitution_lm/data/model.pkl
python tangent/substitution_lm/run_pipeline.py --steps 6,5
python tangent/substitution_lm/eval/ppl_bpb.py --mechanism all --workers 4
```

---

## Phase 3 — Skip-N-Gram Hybridisation (EXARCH augmentation)

The substitution structure handles local consistency well but has no mechanism
for long-range topical / referential signal — `CONTEXT_SIZE` is bounded by the
recency-weighted window, and pushing it much past 10 trades signal for noise.
The natural complement is the skip-N-gram primitive from the EXARCH design
(see [private/explicit_lm_design.md](../../private/explicit_lm_design.md)):
a distant anchor token `A` modulates the local bigram-conditioned prediction
`P(D | B,C)`, capturing exactly what a one-layer attention-only transformer's
skip-trigram circuit computes (per Elhage et al.). The two architectures aren't
competitors — the anchor lookup augments substitution without replacing any
component. **N=3 (`A...BCD`) is the practical instantiation**; higher-order
skip-N-grams subsume lower orders via marginalisation but the local bigram
is already enough disambiguation for our use case.

The substitution LM's existing iterative-subtraction pipeline (collocations
→ clausal parse → nested NPs → residual) gives us the lattice for both
integration paths below "for free" — the node table already separates
content-bearing multi-token units (NPs, collocations) from the residual
function-word / discourse-marker stream.

### Reading A — Anchor pool (complementary, secondary)

Reuse the NP-head and named-entity nodes from the node table as the *anchor*
pool. Each prediction looks up skip-3-gram contributions from every anchor
present in the recent context window, in addition to the local substitution
prediction. Conceptually:

```
P(D | context) ∝ P_substitution(D | local) × Σ_{A ∈ anchors} P_skip(D | A, BC)
```

- Anchors are already extracted (NP nodes in `node_table.pkl`, ~50–100K)
- Integrates as a Mechanism-C-style re-rank: substitution gives top-K, skip
  lookup re-weights by anchor agreement
- Modulates *all* predictions, not just specific token classes
- More expensive — every prediction pays the anchor-lookup cost

### Reading B — Specialist routing (primary, larger expected lift)

Skip-3-grams predict *only* for the residual class — function words and
discourse markers that the substitution machinery handles poorly. Content
words inside NPs continue to use substitution + properties; only the
"connective tissue" gets routed to skip-N-grams.

Why this is the cleaner division of labour:

- **Function words carry discourse state, not local semantics.** `"the"` vs
  `"a"` depends on whether a referent has been introduced; `"however"` vs
  `"therefore"` depends on the rhetorical stance of preceding clauses.
  Local bigram context is uninformative; distant anchor is exactly the
  right signal.
- **Content words carry local syntactic / semantic signal.** Co-occurrence
  + property compatibility already get strong predictions for the head of
  an NP; routing them to skip-N-grams would dilute, not improve.
- **Mirrors human language structure.** Content words are locally predictable
  from semantic priming; function words encode broader discourse state. The
  split isn't arbitrary — it tracks a real cleavage in how language works.

Gated routing rather than score fusion: at predict time, classify the
target-token slot (function-word / discourse-marker vs. content); route to
the appropriate mechanism; no need for the two distributions to be combined
arithmetically.

### Feasibility / sizing (Reading B specifically)

| Set | Approximate size | Source |
|---|---|---|
| Anchor pool `A` (NP heads, named entities) | ~50K–100K | existing node table |
| Residual prediction targets `D` (function words + discourse markers) | ~5K | POS-filter the lemma vocab |
| Local bigrams `BC` (top-frequency, observed) | ~50K–100K | derivable from existing bigram cache |
| Sparse observed `(A, BC, D)` triples in WT103 | ~10–50M | empirical, to measure |
| Storage (int32 sparse) | ~0.5–2 GB | comparable to existing DAG |

Tractable. The dense `V^3` worst case never materialises because we're
sparse and the prediction set is narrow.

### Open implementation questions

- **Residual identification:** POS-based (closed-class tags: `DET`, `ADP`,
  `CCONJ`, `SCONJ`, `PART`, `AUX`, common adverbs) or frequency-based
  (top-K most common lemmas in the corpus)? POS is more principled; the
  spaCy parse cache already has the tags.
- **Multi-anchor handling:** when CONTEXT_SIZE=10 holds ~5–10 anchors, how
  to aggregate their skip-3-gram contributions — sum (independent
  evidence), max (most-confident anchor wins), or recency-weighted? EXARCH
  defaults to binary anchor-presence; recency-weighted is a soft alternative.
- **Backoff:** what happens when `(A, BC)` has no observed `D` distribution
  in the table? Fall through to substitution, or to a marginalised
  skip-bigram?
- **Anchor decay window:** does `A` count if it appeared in the last 5
  tokens, the last 20, the whole clause, the whole paragraph? Different
  windows likely matter for different anchor types (entities = long;
  sentiment = short).
- **Pipeline integration:** is skip-N-gram extraction a new pipeline step 7
  (built once from parse cache) or computed on-the-fly from the existing
  DAG? The latter is simpler but slower per prediction.

Position in roadmap: **after Phase 2 results land** (MLP + WSD + exception
tables). Phase 3 builds on a stable substitution baseline rather than
chasing improvements in parallel.

---

## Open Questions

- Does property-based edge filtering improve PPL, or is raw co-occurrence already
  capturing the same signal? (SVDR will tell us if properties add anything independent.)
- At what corpus size does Mechanism B overtake A? (Run on WT103 subsets: 10%, 50%, 100%.)
- What frequency threshold is optimal? (≥5 chosen conservatively; may revisit after
  seeing node/edge counts.)
- How to handle n-grams not in ConceptNet in MVP (before MLP)? Options: (a) assign neutral
  property vector (all 0.5), (b) skip property check for unknown nodes, (c) inherit from
  most frequent sub-n-gram that has coverage.
