# Canonical Lattice Encoding (fixed-slot conlang input)

**Status:** design / pre-experimental. Post-release track; not a release gate.
**Depends on:** [`reincorporate_large_semantic_embedding.md`](reincorporate_large_semantic_embedding.md) — the lattice supplies the *positional* named frame; the prescribed embedding supplies the *dimensional* named frame; E5 is where they combine. Also inherits that plan's word-level-tokenization assumption (see [Tokenizer dependency](#tokenizer-dependency)). Sequencing per the feedback recorded in that doc stands: the dyadic-bucket PMI variance-explained curve runs before any embedding training; only E0/E1 here (CPU-only) are cheap enough to interleave with the release pipeline.

## Summary

Re-encode the input so that **position becomes semantically meaningful**. Sentences are deterministically transformed into fixed-width clause lattices — one slot per grammatical role, in a fixed order — connected by an inter-clause relation stream (AND, IF/THEN, NOT, coreference entity tokens). WaveletLM is a pure position-mixing machine: every token interaction is a function of relative offset (dyadic dilations + crawl lags). Under the lattice, relative offsets *are* grammatical relations — SUBJ→VERB is the same offset in every clause — so fixed-offset mixing performs alignment work that surface text demands content-based mechanisms for.

Invertibility to surface text is achieved by **bookkeeping, not linguistics**: the encoder emits a per-clause receipt (surface permutation, deleted function words, morphological features, coref realizations) sufficient for exact reconstruction. The model never sees the receipt.

## Motivation

1. **The recall localization finding** ([Localizing the Recall Break](../README.md#localizing-the-recall-break-the-write-exists-the-read-is-missing)): on D3, KEY→VALUE binding forms at the source (SRC Δ ~+0.045) but never reaches the query (QRY Δ ~+0.002, at floor) — retrieval breaks at the *read*, because static position mixing cannot supply content-dependent routing. The lattice attacks the same gap from the data side: if roles sit at known offsets, retrieval-by-role becomes a *positional* operation the existing machinery already performs. Whether that substitutes for content-addressed retrieval on recall probes is E4's question.
2. **The crawl learns lags, not content** ([Long-Context Retrieval](../README.md#long-context-retrieval-wavelet-keyed-knn-lm)). Under the lattice, a lag *is* a role relation — the learned crawl weights become interpretable role selectors rather than opaque distance preferences, and the wavelet-autopsy's finding that fine levels sharpen onto small lags acquires a grammatical reading.
3. **Stationarity of syntactic statistics.** In surface text, subject–verb agreement occurs at every possible offset and the model marginalizes over all of them. In the lattice it is one fixed-offset statistic per scale.
4. **Interpretability by construction — the Privileged-Basis Program's other axis.** Study 2 answered the channel-basis question negatively: the native channel basis is not privileged, and Study 2b's response is to *induce* alignment with an explicit penalty rather than hope for it. The lattice is the same philosophy applied to the position axis, one step stronger: the positional basis is not induced but **prescribed** — slot semantics are fixed before training, so every mixing weight is a statement about named grammatical role pairs with no probing required. Jointly with the prescribed embedding, activations become a table indexed by (named feature, named role).
5. **Deterministic encode path and legible failures.** Frozen parser + written routing table + exact receipts: model inputs are reproducible bit-for-bit from the spec, and errors localize in role-space (wrong OBJ; broken SUBJ–VERB agreement; mishandled NOT scope) rather than requiring post-hoc reconstruction.

## Non-goals / scoping

- **Not** a general-purpose interlingua. Coverage target is the phenomena present in the experimental corpora, extended only as measurement demands.
- **Not** semantic normalization (no paraphrase collapsing, no entailment canonicalization). Only syntactic/role regularization with exact round-trip.
- The interpretability claim covers the **model**, not the system. The parser and coref resolver are opaque learned components; the interpretability boundary is drawn at the canonical stream. Any writeup states this explicitly.

## Design

### Clause frame, sized against the current decomposition

Fixed-width, power-of-two slot frame. Working draft (width 8, minimal; the real frame likely needs OBL/IOBJ and width 16 — E0 decides):

```
[DET_S] [ADJ_S] [SUBJ] [ADV] [VERB] [DET_O] [ADJ_O] [OBJ]
```

Empty slots hold `PAD`. Alignment with the current Micro/Mini recipe (`levels=7`, `block_size=256`, `wavelet_crawl_k=33`):

- **Within-clause** role relations at width 8 live at offsets 1–7 → scales 1–3.
- **Inter-clause** relations (adjacent and nearby clauses) live at offsets 8–128 → scales 4–7. Clause structure lands cleanly on the existing 7-level ladder with no depth change; a width-16 frame shifts the within/between boundary up one scale.
- A 256-token block holds 32 width-8 clauses (16 at width 16); the crawl window (k=33) spans ~4 adjacent clauses at width 8.

Slot inventory, ordering, and width are **design constants** recorded in the spec; changing them invalidates cross-run comparisons.

### Tokenizer dependency

The lattice assumes **one token per slot**, which the current GPT-2 BPE (`tokenizer: auto`) violates: a multi-subtoken word in a slot breaks fixed width and destroys offset stationarity — the entire mechanism. The lattice therefore requires **word-level tokenization with subword fallback confined to a designated overflow region**, i.e. the same tokenizer regime the semantic-embedding plan already assumes (word-level rows in the frozen table, frequency-capped vocab). This is a shared prerequisite, not a new one, and it hard-couples the two plans at the tokenizer: the lattice cannot run a meaningful experiment under plain BPE. E0 reports the out-of-vocab rate per slot as one of its go/no-go numbers.

### Routing

A written table from UD dependency labels to slots. Content words are routed by their label relative to the clause head (`nsubj`→SUBJ, `amod(nsubj)`→ADJ_S, `advmod`→ADV, `obj`→OBJ, …). Verbs stored as **lemma**; tense/aspect/polarity/do-support decompose into morph features in the receipt.

### Overflow

When multiple tokens compete for one slot, the head-closest token stays in-slot; the remainder append to a variable-length overflow tail after the clause, each tagged with its home slot:

```
the friendly dog PAD barked PAD PAD PAD | OVF ADJ_S:big ADJ_S:old
```

Keeps the main lattice fixed-width and dense; rare pile-ups pay a tail instead of forcing worst-case width everywhere. The overflow region is also where subword-fallback tokens live. In-slot selection rule (head-closest vs. frequency-ranked) is a recorded convention.

### Relation stream (inter-clause operators)

The recursive structure that killed flat-template formalisms (conjunction, conditionals, negation scope) lives **between** clauses, not inside them:

- `AND(C1, C2)`, `OR(...)` — from `conj` + connective.
- `IF(C1) THEN(C2)` — from `advcl` with `if` marker; other subordinators get their own relation tokens as coverage demands.
- `NOT(Ci)` — clauses are stored affirmative; polarity is a scoped operator. **Rejected variant (recorded so it stays rejected):** negation as a probabilistic inverse / feature-flip of the clause's components. Negation composes with scope ("not bad", "no one left"); vector negation fails on exactly this, per the distributional-semantics literature.
- Entity tokens `E1, E2, …` from coreference resolution occupy nominal slots. Pronouns and elided arguments resolve to explicit entity tokens — the canonical stream is *more* regular than the surface text.

### Receipt (invertibility by bookkeeping)

Per clause, the encoder records: (a) the permutation mapping filled slots back to surface indices, (b) deleted function words, (c) morph features for re-inflection, (d) surface realization of each entity mention ("it", elided, full NP). Reconstruction is deterministic: read filled slots, invert the permutation, re-inflect, reinsert deletions. Exact round-trip is guaranteed **by construction**; a mismatch is a bug in the bookkeeping code, not a modeling result. The receipt is metadata only — never in the model's input stream, so no leakage surface.

### Worked example

"The big old dog didn't chase the cat, and it slept quietly."

```
Relations:  AND(C1, C2)   NOT(C1)
Entities:   E1 = "the dog"    E2 = "the cat"

C1:  the  old  E1  PAD      chase  the  PAD  E2   |  OVF ADJ_S:big
C2:  PAD  PAD  E1  quietly  sleep  PAD  PAD  PAD

Receipt C1: perm=[…], morph={chase:(Past, Neg, do-support)}
Receipt C2: coref={E1→"it"}, morph={sleep:(Past)}
```

## Metrics: the accounting rule

**Raw BPB on the canonical token stream is meaningless** — a majority-PAD stream is trivially cheap per token, and the canonical stream's token count differs from the surface's. The fair metric is the one the repo already ranks by: **bits per byte of the original surface text** — total bits spent predicting the canonical encoding (lattice + relation stream + overflow) ÷ surface byte count. The denominator is fixed by the source regardless of encoding, so canonical and surface arms are directly comparable, tokenizer-agnostically, on the existing sliding-window protocol with the existing ~0.0010 noise floor. Format overhead (relation tokens, OVF tags, entity tokens) is a tax paid inside the numerator before any modeling advantage counts.

The honest headline claim, if the approach works, is system-level: *WaveletLM-on-canonical beats WaveletLM-on-surface at matched surface-BPB, encoder held fixed* — and the frozen pipeline (parsing, coref, re-inflection) is doing predictive work the model gets to skip, so the external comparison against a transformer-on-surface at matched compute is the one that decides whether the combination is competitive rather than just internally improved.

## Known risks

1. **PAD/DC pathology (top risk).** Dense PAD runs concentrate energy at coarse scales — the same DC-domination failure family as the Haar normalization bug (pre-`wavelet_decomp_norm`/`recon_norm`) and the feature-map DC floor that stalled selectivity in the retrieval screens, reintroduced through the data layout. E0 gates all training on this measurement.
2. **Canonicalization deletes predictively useful redundancy.** Agreement, word-order convention, do-support are the low-entropy scaffolding that makes surface text partially self-predicting. The canonical stream removes the easy bits and keeps the hard ones (content-word choice); surface-BPB may *rise* even as legibility improves. Pre-registered prior: coin flip, slight lean negative.
3. **Encoder quality floor.** Parser/coref errors are invisible downstream: a wrong coref chain yields a perfectly legible lattice of the wrong proposition. Mitigation: start on TinyStories, where sentence structure keeps parse/coref error near-floor; treat encoder version as a frozen experimental constant.
4. **Corpus format shock.** WT-103's idiosyncratic markup (` @-@ `, spaced punctuation — the same artifact implicated in F1's away-penalty) will degrade any off-the-shelf parser. Extending past TinyStories requires a markup-normalization pass, itself receipt-tracked to preserve round-trip.
5. **Coverage gaps.** Questions, raising/control, comparatives, quotation are unhandled by the draft frame. Policy: pass-through as a `RAW` clause type rather than ad-hoc extension; extend the spec only when E0's measured corpus fraction demands it.

## Experiments

Protocol conventions follow the release pipeline: Micro tier (C=256, L=10, levels=7, fully spectral, MBS=48, `lr=48/C`) screens; **asymmetric promotion** — a pass advances to Mini (D0 protocol), a failure is recorded as "failed at Micro" and never kills the direction; paired arms judged against each other, not against unrelated baselines; predictions pre-registered.

| ID | Experiment | Cost | Gates |
|----|-----------|------|-------|
| E0 | **Encoder census.** Encode TinyStories; report per-slot fill rate, median PAD fraction, overflow frequency, relation-token overhead, `RAW`-clause fraction, per-slot OOV rate under the word-level vocab. Wavelet-decompose sample canonical streams at levels=7; inspect per-scale variance for DC domination. Decides frame width (8 vs 16) and slot inventory. | CPU only | Gates all training |
| E1 | **Round-trip validation.** Exact string match over the full corpus. Unit test of the bookkeeping, run before any training. | CPU only | Gates E2–E4 |
| E2 | **Synthetic regularity curve.** Synthetic language with a controllable canonicalization knob; sub-Micro WaveletLM + matched small transformer at each regularity level; measure how the gap moves with regularity. Sidesteps the encoder pipeline entirely; isolates representation misfit from capacity. | sub-Micro GPU | Informs go/no-go on E3 |
| E3 | **Canonical vs. surface, paired.** Micro-on-canonical vs. Micro-on-surface at matched compute, scored in surface-BPB (sliding protocol). The two arms are each other's control. Promotion to Mini on a pass per the asymmetric rule. | Micro (~5h) → Mini | Headline prediction result |
| E4 | **Recall-through-position probe.** MQAR-style synthetic recall ([tools/interpretability/mqar.py](../tools/interpretability/mqar.py)) and the recall-localization diagnostics (`recall_diagnostics.py`) on the lattice model: does fixed-slot encoding move the QRY-side read off its floor — i.e., does retrieval-by-role through position supply what content-addressed reads were needed for on surface text? | Micro-scale | Standalone finding either way |
| E5 | **Named-frame interpretability readout.** With the prescribed embedding (dependency doc) under the frozen-tied head: mixing-weight and crawl-weight maps over (role-pair, scale); verify role-offset statistics (e.g. agreement at the SUBJ→VERB offset) are load-bearing via the existing FDA / dimensional-suppression / SOW rails (Study 8 machinery). | analysis | Joint artifact with embedding doc |

Results, including any parallels to previously parked retrieval mechanisms, go in a Results section here and in the final paper once testing completes.

## Interpretability artifacts (what this buys even if E3's canonical arm loses)

- **Positional named frame:** every mixing and crawl weight labeled by grammatical role pair and scale, fixed before training — the position-axis complement to the prescribed embedding's named dimensions, and jointly a (feature × role) activation table. Prescribed rather than induced: the Study-2b philosophy without the penalty sweep.
- **Deterministic encode path:** reproducible model inputs from a written spec; no tokenizer-artifact disputes.
- **Legible failure modes:** errors localized in role-space by construction.

A lattice model that loses modestly on surface-BPB but whose every weight has a grammatical name remains a publishable object on the Privileged-Basis track. A lattice model that fails to train from PAD pathology is not — hence E0 first.

## Tooling

- **UD parsing + morphology:** spaCy `en_core_web_trf` (throughput) or Stanza (closest tracking of UD label documentation — relevant since routing rules are written against those labels). Trankit as fallback. Lemma + morph features come from the same pass.
- **Clause segmentation:** own code (~50 lines) — traversal of the dependency tree; boundaries at `conj`, `advcl`, `ccomp`, `xcomp` subtrees; relation tokens from connectives. Deliberately in-repo: routing conventions are the design content.
- **Coreference:** `fastcoref` (spaCy pipeline component, corpus-scale); `coreferee` as lighter fallback.
- **Round-trip test:** exact string match (deterministic by construction).
- **Reference, not dependency:** `amrlib` — parse the same sentences to AMR to spot phenomena the frame can't express yet.

## Open questions

- Frame width and slot inventory after E0? (Draft width-8 frame is known-insufficient: no OBL/IOBJ.)
- Word-level vocab construction: shared with the semantic-embedding plan's frequency-capped vocab, or lattice-specific? (They must agree if E5 is to run on one model.)
- Does PAD need its own embedding treatment (zero-vector, excluded from whitening statistics) to avoid polluting the frozen embedding's normalization?
- Relation tokens: prefix stream, interleaved between clauses, or a separate concat channel (mirroring the PE-channel design in the dependency doc)?
- How much does the frozen encoder's error rate on harder corpora (post-TinyStories, markup-normalized WT-103) erode both the prediction and interpretability claims?
- Does the lattice change the crawl's learned posture (lag weights snapping to role offsets — multiples of the frame width) — a free check via the existing wavelet-autopsy instrument?

## Relationship to the embedding reincorporation plan

The two plans are the same bet made on orthogonal axes: prescribe the *representation* (named dimensions) and prescribe the *layout* (named positions), and check whether an architecture whose mixing is native to both prescriptions closes the gap to learned/surface baselines at acceptable cost, with a categorically stronger interpretability artifact as the payoff. They share the word-level tokenizer prerequisite and meet at E5. Sequencing stands as recorded in the dependency doc: PMI variance-explained curve first; E0/E1 here are CPU-only and can interleave with the release pipeline at any point.
