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
| Corpus tokens | 103,227,021 |
| Parse time | — |
| Unique nodes (after freq filter) | — |
| Unique edges | — |
| ConceptNet coverage (nodes with ≥1 property) | — |
| Build peak RAM | — |
| Build wall time | — |

### Evaluation Results

| Mechanism | PPL ↓ | BPB ↓ | SVDR F1 ↑ | Inference time / token | Notes |
|---|---|---|---|---|---|
| A — Weighted edge walk | — | — | — | — | |
| B — Aggregated vote | — | — | — | — | |
| C — Hybrid re-rank (K=?) | — | — | — | — | |
| KenLM 5-gram (baseline) | — | — | — | — | reference n-gram LM |

### Phase 2 Additions (planned)

| Variant | Change vs. MVP | PPL ↓ | BPB ↓ | SVDR F1 ↑ | Notes |
|---|---|---|---|---|---|
| + WSD (coarse) | Sense-disambiguated token IDs | — | — | — | |
| + WSD (fine) | WordNet synset IDs | — | — | — | |
| + MLP properties | Replace ConceptNet lookup | — | — | — | |
| + Exception tables | Manual + auto-detected overrides | — | — | — | |

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
