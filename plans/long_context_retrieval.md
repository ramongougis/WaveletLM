# Long-Context Retrieval for WaveletLM (wavelet-keyed kNN-LM / FSRR)

## Status

**Proposed — post-release.** Captures the retrieval deliberations of 2026-06-21. Sibling to
[long_context_decimation.md](long_context_decimation.md) (decimation + length-generalization). That file
is about making the *model* cheaper at length; **this file is about content-addressed retrieval** to get
past the **~512-token useful-context ceiling** the length-gen sweep measured.

## Why retrieval, not longer context

The length-gen result (see README → Block-Size Extension) is two-sided:
- WaveletLM **ingests** arbitrary length cheaply and robustly (linear memory, rising throughput, no
  positional-embedding breakdown), but it only **exploits** ~512 tokens for *quality*.
- It is a fixed-basis **position mixer** (the crawl learns lags, not content) — the SSM-camp weakness on
  NIAH/RULER. There is no native content addressing.

So the path to long-range **factual / retrieval** ability is a **content-addressed retrieval layer**
(augmentation on a frozen model), not extending the model's intrinsic reach. This is exactly the regime
where retrieval-augmented LMs help most.

## The core: wavelet-keyed kNN-LM

This is **kNN-LM** (Khandelwal et al. 2020), independently re-derived, with a wavelet twist:
- **After** pre-training, sweep the corpus once with the **frozen** model and store
  `(key = a context representation, value = the actual next token)`.
- At inference, query with the current context, pull *k* nearest neighbors → `p_kNN`, and **interpolate**:
  `p = λ·p_kNN + (1−λ)·p_LM`.
- kNN-LM helps **where the parametric model is weakest** (rare / factual / long-range / exact copy) —
  *precisely* WaveletLM's ~512 ceiling. Unusually clean synergy: the augmentation covers the exact deficit.

The novel parts are the **wavelet-coefficient key** and using the **wavelet scales as a native
coarse-to-fine index** — not the retrieve-and-interpolate framework, which is kNN-LM.

## FSRR — Fine-Scale Resolution Retrieval (the refined pipeline)

A **retrieve-and-rerank** stack (the modern IR/RAG standard), with the wavelet as the dense reranker:

1. **Sparse first-stage retrieval on high-IDF anchors** (rare nouns / named entities / years — the
   "sufficiently rare" tokens; "sufficiently rare" = an IDF threshold). Inverted index / BM25. This
   **sidesteps the coarse-ordering problem entirely** — you match an *exact* rare token, not a fuzzy
   whole-passage shape. Three reasons it fits:
   - rare content-bearing terms are the most *discriminative* (the IDF principle);
   - **NIAH needles *are* anchors** (the entity/number/name you query) — tailor-made;
   - **self-gating**: a prompt with no rare anchor → nothing to search → no-op fallback to the model, so
     retrieval fires *only* on the rare/factual cases where the model is weak.

2. **Dense reranking** of the candidate windows via **fine-scale wavelet** similarity. Fine beats coarse
   for **order-robustness**: fine scales encode *local collocation/syntax* (constrained permutations — a
   noun stays glued to its modifiers), coarse scales encode *global document structure* (free to
   reorganize). So fine-around-anchor is the better-conditioned similarity.

3. **Interpolate with the model** (kNN-LM); the learned version is adaptive-kNN-LM / RETRO.

## Coarse-to-fine + pooling (for the dense leg, when used as a passage index)

- A passage's **pooled** coarse coefficients (mean/max over the passage) form a fixed-dimension key,
  comparable across **any granularity** (article / chapter / paragraph / sentence). The undecimated
  (à-trous) transform is **shift-invariant**, so the keys are position-canonical.
- **Pooling is order-invariant**, which resolves the "intro → content → summary" reorganization fragility
  for the *unpooled* coarse sequence. You want order-sensitivity only at the **fine** stage (localizing
  *where* the needle sits inside the matched passage).
- Residual limit: even pooled, the coarse key is a **structural/spectral fingerprint** — order-invariant
  but *genre*-sensitive, weak on semantic content. That is why the **anchor (sparse/lexical) leg** is
  needed for content discrimination → **hybrid dense-sparse retrieval** (the modern standard).
- Assembled: coarse-pooled + anchors = **"which passage"** (order-invariant); fine wavelet = **"where in
  it"** (localization → copy the exact continuation).

## The contribution worth chasing: a compressed datastore

kNN-LM's practical wall is **storage** — one C-dim vector per corpus token (~400 GB for WT-103, far more
at scale). The **strided/coarse wavelet keys are a *compressed* datastore** — fewer, smaller, shift-
invariant entries, with fine detail computed on-demand only for the handful of coarse/anchor hits. This
attacks kNN-LM's main limitation directly, and is the most publishable angle.

## Honest caveats

- **Augmentation, not intrinsic reach.** Find-then-vote on a frozen model, working *around* the ~512
  ceiling — legitimate (that's what kNN-LM is), but it must not be reported as "the model now uses 100K
  context natively."
- **Content vs structure.** Verbatim NIAH (needle = anchor) is the strong case; a **paraphrased semantic
  needle with no distinctive anchor** is the weak case — may need a *learned* retrieval projection of the
  coefficients, or it fails. (Same content-vs-position issue that gates WaveletLM's retrieval generally.)
- **Continuation, not QA.** FSRR/kNN-LM vote for the **next token** by reading the token that *follows the
  matched context*; that is clean when the corpus phrases things like the prompt. A question-shaped prompt
  needs the corpus to contain a similar continuation, or an answer-extraction step on top.
- **The wavelet key is an intermediate, not the most predictive representation.** The model's final
  pre-LM-head hidden state is the *known-good* kNN-LM key; wavelet coeffs may retrieve worse neighbors.
  Hedge with the **hybrid**: wavelet-coarse coeffs for cheap candidate *search*, the hidden state for the
  kNN *scoring*.

## Learned-enhancement spectrum (cheapest first)

1. **Fixed λ** interpolation (vanilla kNN-LM).
2. **Learned gate** `λ(context)` — adaptive kNN-LM; the gate can *be* a function of anchor-match confidence
   + wavelet-rerank score. Frozen model, tiny head — the pragmatic sweet spot.
3. **Learned reranker head** — train the fine-wavelet-similarity → relevance map.
4. **Joint training** — RETRO-style chunked cross-attention (most powerful, most expensive).

## Test ladder (all frozen, forward-only)

1. **recall@k per scale** — plant a synthetic needle; per scale, measure retrieval recall. The best scale
   *is* the search scale, found with zero training. **Start with verbatim needles** (the likely-to-work
   case) before semantic.
2. **The decisive 4-way rerank ablation** — on held-out continuations that contain a rare anchor, measure
   PPL for: **anchor-only**, **anchor + BM25-rerank**, **anchor + hidden-state-rerank**,
   **anchor + wavelet-fine-rerank** (each with kNN-LM interpolation). If the wavelet column wins — or ties
   at lower cost — FSRR is real; if not, lexical + model suffices. Either answer is publishable.
3. **Compressed-datastore comparison** — storage vs held-out PPL for the wavelet-coarse vs hidden-state
   datastores.

## Precedents (cite honestly)

- **kNN-LM** (Khandelwal et al. 2020, "Generalization through Memorization"); **RETRO** (Borgeaud et al.
  2022); **adaptive kNN-LM** (He et al. 2021).
- **Hybrid dense-sparse retrieval** (BM25 + dense; SPLADE, ColBERT-style late interaction).
- **Multiresolution / wavelet image retrieval** (coarse-to-fine matching); **FAISS IVF**
  (coarse-quantize → fine rerank) — the systems analog of the coarse-to-fine wavelet index.

## Relationship to existing components

- The **`decompose_bypass` running mean** is already a cumulative coarse summary the model computes — a
  candidate ready-made coarse key, no new machinery.
- **PKM / FwPKM** is the other content-addressed-memory path the project already floats; the wavelet
  retriever is a sibling — arguably cleaner, since it reuses the wavelet basis.
