# Semantic Embedding Reincorporation from EXARCH

## Summary

Scale the semantic embedding approach from EXARCH to C=2048+ by replacing human-interpretable concept labels with structural, statistical, and syntactic features derived from corpus analysis. Instead of "violence" or "happiness," dimensions would encode n-gram co-occurrence patterns, positional statistics, and token-level structural properties that more closely mirror what learned embeddings likely discover on their own.

## Motivation

- EXARCH ("Explicit Attentionless Reasoning with Causal Harmonics", the previous version of WaveletLM) achieved BPB 1.0112 at C=512 with 256 human-defined concepts. Note this was before the v2 benchmark test fixes, so its BPB numbers are underestimates of true performance.
- Learned embeddings at C=2048 achieve BPB 1.1431 at 1 epoch (and improving).
- Human-interpretable concepts previously hit a ceiling at 512 orthogonal dimensions. Increasing C beyond that number slightly degraded performance. However, the expressivity and numerical representation of the dimensions as well-chosen coefficients matters. Previously, only one-hot coefficients were used in a "yes/no" format.
- However, structural/statistical features can scale indefinitely: there are far more n-gram patterns, positional encodings, and co-occurrence statistics than there are single-word human concepts.
- Question: can a hand-crafted structural embedding match or beat learned embeddings at high C while retaining the potential interpretability benefits of prescribed features?

## Proposed feature categories

### 1. N-gram identity features (scalable)
- Common 3-grams from the training corpus
- Each dimension indicates whether the current token participates in a specific 3-gram
- With vocab=50257 and n=3, there are billions of possible 3-grams, but the top 10,000-50,000 cover most of the distribution
- Dimensionality: as many as needed (easily 1000+)

### 2. Positional/structural features
- "Token follows punctuation" (comma, period, colon, etc.)
- "Token is clause-initial" (after period + space)
- "Token is within quotation marks"
- "Token position mod 2/4/8/16" (periodic positional encoding)
- "Distance to nearest sentence boundary"
- Dimensionality: ~50-100

### 3. Co-occurrence statistics
- "This token has high PMI with tokens N positions ahead/behind"
- "This token typically appears in sequences of length > K"
- "This token's forward entropy (how predictable the next token is)"
- "This token's backward entropy"
- Dimensionality: ~100-200 (varies with granularity)

### 4. Frequency and distribution features
- Token unigram frequency band (log-scaled buckets)
- Token burstiness (how clustered its occurrences are)
- Token domain specificity (appears mainly in certain topic clusters)
- Part-of-speech probability distribution (if POS tagger available)
- Dimensionality: ~50-100

### 5. Traditional semantic features (retained from EXARCH)
- High-level concept labels from FDA or similar
- Limited to ~256-500 dimensions where orthogonal concepts exist
- These provide the interpretability layer

## Total feature budget at C=2048

| Category | Dimensions | Source |
|----------|-----------|--------|
| N-gram identity | ~1000 | Corpus statistics |
| Positional/structural | ~100 | Rule-based |
| Co-occurrence | ~200 | Corpus statistics |
| Frequency/distribution | ~100 | Corpus statistics |
| Traditional semantic | ~256 | FDA/LLM labeling |
| Headroom / learned residual | ~392 | Learnable (hybrid) |
| **Total** | **2048** | |

The "headroom" dimensions could be left as learnable (zero-initialized), creating a hybrid embedding where ~80% is prescribed and ~20% is learned. This tests whether the prescribed features cover the important dimensions and the model only needs to learn the remainder.

## Coefficient assignment: how feature values are determined

For each (token, feature) pair, a numerical coefficient must be assigned. This is a design choice orthogonal to feature selection, with meaningful tradeoffs across interpretability, quality, cost, and scalability.

### 1. One-hot / binary (0 or 1)

Each feature is a hard yes/no.

- **Pros:** cheapest, trivially interpretable, deterministic, no labeling pipeline for rule-based features
- **Cons:** information-poor (ignores degree); sparse at high dimensionality
- **Best for:** rule-based and n-gram features where the answer is genuinely binary

### 2. Corpus-statistical (continuous, automatic)

Coefficients derived from corpus statistics (frequency, PMI, positional distributions, etc.).

- **Pros:** cheap ($0), scales to large feature sets, deterministic given a fixed corpus
- **Cons:** inherits corpus biases; captures distributional rather than semantic content; shallow for abstract concepts
- **Best for:** frequency, co-occurrence, and positional features

### 3. LLM-scored (continuous, typically 0-1)

An LLM scores each (token, feature) pair against a prompt describing the feature.

- **Pros:** captures nuance; handles abstract/interpretive concepts; scales well once prompts are defined
- **Cons:** cost grows with vocab × features (~$200+/concept at Gemini 2.5 Flash pricing, more for higher-quality models); bakes in LLM biases; non-deterministic without careful seeding
- **Best for:** abstract or interpretive features ("associated with anger", "primarily technical")

### 4. Human-rated (continuous or ordinal)

Human annotators score (token, feature) pairs, typically with inter-annotator agreement metrics.

- **Pros:** highest quality on subjective features; ground truth for validating other methods
- **Cons:** expensive; doesn't scale to large feature sets or vocabs; subject to rater variance
- **Best for:** small high-stakes feature sets; validation benchmarks for other methods

### 5. Hybrid pipelines

Combinations: LLM-scored with human validation on the tail; corpus-statistical seeded with human-curated anchors; active-learning routing (cheap method first, escalate to expensive method only on disagreement); one-hot/binary with any of the others.

- **Pros:** often the best quality/cost ratio in practice
- **Cons:** pipeline complexity; requires careful validation methodology

### Recommended default for a first large-scale experiment

- Rule-based/structural features → **one-hot**
- Frequency, co-occurrence, n-gram identity → **corpus-statistical**
- Semantic / abstract / interpretive features → **LLM-scored** at a cheap-but-competent model (e.g., Gemini 2.5 Flash, Claude Haiku)
- **Spot-validate with human raters** wherever the downstream model's behavior shows unexpected patterns at a given dimension

## Homonym separation (sense-disambiguated tokens)

A tokenization-level design variant orthogonal to feature selection and coefficient assignment. Each (word, sense) pair becomes a distinct token with its own feature vector, rather than a single polysemous token whose embedding must encode multiple unrelated meanings.

### Motivation

A prescribed embedding where every dimension is a plain-language feature only makes sense if each token has internally consistent semantics. A polysemous word like *bat* carries contradictory feature values in a single row:

- Is this token a flying mammal? (mammal sense: high; sport sense: low)
- Is this token a piece of sporting equipment? (mammal sense: low; sport sense: high)
- Is this token associated with caves? (mammal sense: high; sport sense: low)

Under a single shared row, these contradictions must be averaged, blunting every dimension that touches them. Sense separation eliminates the averaging: each sense gets a clean, one-hot-leaning feature row that pushes the embedding toward the disentangled ideal the prescribed-embedding approach is aiming for in the first place.

### Approach

- **Preprocessing:** run a word-sense disambiguator (WSD classifier - could be WordNet-sense-tagged, or an LLM-based sense labeler) over the training corpus. Each occurrence of a polysemous token is mapped to a sense-specific variant: `bat` → `bat_mammal`, `bat_sport`, `bat_blink`, etc.
- **Feature assignment:** each sense-variant token receives its own feature vector using any of the coefficient-assignment methods above. LLM-scoring is particularly clean here since the sense label itself disambiguates the prompt.
- **Tokenizer:** becomes stateful and context-aware at the encoding step. Decode is straightforward (sense-variants emit their parent word's surface form; the sense label is metadata, not orthographic).
- **Vocab cap:** rank sense-disambiguated tokens by corpus frequency, keep the top N (e.g., N=50,000 matching the current GPT-2 vocab), and fall back to character-level or BPE subwords for everything else. This is standard practice for any vocabulary-limited LM and does not introduce new prediction-surface complexity.

### Tradeoffs

- **Pro - cleaner feature space:** each row is internally consistent; dimensions that would have averaged across senses now carry signal.
- **Pro - interpretability at inference:** the model's chosen sense is explicit at generation time, rather than emergent from context. This is a direct win for the interpretability motivation.
- **Pro - easier prescription:** LLM-scoring a sense label (`bat_mammal`) is less ambiguous than scoring a bare word - scorers don't have to hedge across meanings.
- **Con - preprocessing cost:** adds a WSD classifier to the pipeline. If LLM-based, non-trivial per-corpus cost; if rule/dictionary-based (WordNet), much cheaper but lower accuracy on noisy corpora.
- **Con - vocabulary bloat before capping:** a ~50k GPT-2 vocab might need 70–100k sense-variants to cover the same surface-form coverage, then re-capped. Careful frequency accounting is required to avoid losing rare-but-important sense-variants.
- **Con - training-inference consistency:** the same WSD pipeline must run on any input text at inference to assign sense tokens consistently. Inference cost is modest but nonzero.

### Prior art and where this slots in

Sense-separated tokenization was explored in sense2vec (Trask et al., 2015), AdaGram (Bartunov et al., 2016), and related work, but was largely abandoned in favor of contextual models (BERT-style) that disambiguate implicitly through self-attention. A prescribed-embedding architecture is a natural home to revive the idea: per-token feature rows are already the representation, so sense-splits are a near-free extension of the existing embedding table rather than a wholesale architecture change. This is a meaningful point of differentiation from attention-based models, not a detour.

### Open questions specific to sense separation

- How many senses per word is the right default? WordNet has long tails; top-K-by-frequency per lemma is a reasonable cap.
- Does the WSD classifier's error rate dominate the interpretability win? If 15% of sense assignments are wrong, does the noise wipe out the cleaner feature rows?
- Can sense-separated tokens share some feature dimensions with their lemma-parent (e.g., "is a noun") via parameter tying, without collapsing the separation? Hybrid lemma+sense representation is worth probing.

## Relational/positional construction (dyadic-bucket PMI factorization)

A third coefficient-construction route (2026-06-11), motivated by the Yoneda view of meaning — a token is characterized by the totality of its relations to other tokens — and by the architecture itself: WaveletLM is a pure position-mixing machine (no content-based attention; every token interaction is a fixed-or-learned function of relative offset via wavelet dilations and crawl windows), so an embedding whose named coordinates are *offset-bucketed relational statistics* encodes exactly the statistics the model natively consumes.

**Recipe:**
1. One pass over the corpus: for each token pair (v, w) and each **dyadic offset bucket** d ∈ {1, 2–3, 4–7, 8–15, …, 128–255}, accumulate co-occurrence counts. Buckets deliberately mirror the wavelet's scale structure, so embedding coordinates align with decomposition levels ("dimension k = co-occurs-with-cluster-j at distance 8–15" pairs naturally with the scale-3 gate).
2. Convert to PMI (or shifted PPMI) per bucket: the V×V×D relational tensor.
3. Factorize to V×C (per-slice SVD or joint tensor factorization); name each dimension by its dominant (cluster, bucket) loading. All names are verifiable from corpus statistics — less plain-language-readable than "is-a-noun", but auditable.
4. Measure variance-explained vs C **before training anything** — the graceful-degradation curve of the factorization is a free preview of how much relational structure survives compression.

**Cost:** pure counting + factorization; no LLM labeling (vs ~$200–250/concept for FDA-style features).

**Rejected simpler variant (recorded so it stays rejected):** per-block *absolute* position statistics — e.g. the mean of (t_mid − t_b) per vocab word per corpus chunk, compressed into V×C. Block boundaries are arbitrary 256-token cuts with no alignment to sentences/documents, so by (approximate) stationarity of language every word's within-block position distribution is ~uniform and its mean converges to ~0: the statistic is washed out. More fundamentally it is a *unary* statistic, while the Yoneda framing itself locates meaning in *pairwise* relations — position is a property of an occurrence, not of a type. Type-level statistics should be relational (the recipe above); occurrence-level position belongs at runtime (next section).

**Caveats:** a frozen Yoneda snapshot — corpus-global, no context sensitivity, so polysemy lands on the embedding (see homonym separation above). The learned embedding will likely still win raw BPB; the bet is that this construction's quality gap is smaller than plain-language features' because it matches the architecture's native statistics, at comparable traceability.

## Runtime positional channel + frozen-tied-head architecture

Companion architecture design (2026-06-11) for *any* of the frozen constructions: keep type-level semantics frozen in the embedding, inject occurrence-level position at runtime, and tie the output head to the *position-free* frozen embedding.

- **Input:** `x_t = E[token_t] ⊕ PE(t)` — frozen semantic table E (V×C_sem) **concatenated** with a positional channel (C_pos dims), reusing the existing `concat` hybrid-embedding mechanism. Concatenation, NOT addition or convolution: additive PE pollutes every named dimension and convolution scrambles them — concat keeps semantic dims pure/labeled and position dims separately labeled.
- **Output:** logits = h·Eᵀ with the **frozen** E (no position). The model must end every forward pass in the pure semantic frame — position is used by the trunk and discarded by the head ("the MLP handles the unencoding"). Minimal relaxation if the frozen-tied head costs too much BPB: a learned per-vocab scalar gain (+ optional bias) on the logits — one parameter per token, fixes softmax geometry without rotating the named frame.
- **Free interpretability artifact:** the pre-head representation is the model's prediction expressed as a *named semantic profile of the next token*, and per-dimension logit contributions `h_i·E[v]_i` give feature-level output attribution with no SAE.
- **Preprocessing requirement:** whiten/normalize E before freezing. Frequency-derived vectors are Zipf-dominated (first principal component ≈ log-frequency) and raw norm disparities wreck softmax geometry under a frozen head.
- **Mandatory ablation arm — ±PE:** WaveletLM's mixing machinery already encodes relative position structurally, so the PE channel may be redundant. If frozen-E-no-PE matches frozen-E⊕PE, that is itself a mechanistic finding (position lives in the mixing structure, not the representation) and simplifies the architecture.
- **First runs:** frozen-E⊕PE + identity transform + scalar-gain head vs frozen-E-no-PE, 1ep, then branch. Combine with the transform-reintroduction test (± fwht/butterfly) from the README's Semantic Embedding section — the same runs localize where basis-adaptation lives.

## Honesty / deception features + causal validation (safety-interpretability axis)

A small, named feature set on the safety axis (2026-06-19), motivated by Bengio's "safe superintelligence" direction — a probe-able internal truth/honesty structure makes a model *auditable* — but scoped honestly. Two framing corrections up front: (1) Bengio's strongest move is **non-agency** (a goal-less predictor has no instrumental incentive to deceive); truth-labeling is a *piece*, not the whole. (2) A truth *representation* does **not** by itself guarantee honest *output* — a capable agent can represent "what's true" and "what to say" separately. So the portable, achievable version here is **not** corpus-wide factuality (unlabelable at scale — we reject it, as does the project lead) but **honesty/deception as named embedding dimensions**: style/context markers, the natural extension of the "corpus frequency in deceptive contexts" example dimension above. There is real grounding that such structure is linearly decodable in LLMs (Marks & Tegmark, *The Geometry of Truth*; Azaria & Mitchell, *The Internal State of an LLM Knows When It's Lying*; Zou et al., *Representation Engineering*).

**Feature set (small, named):**
- `deception / dishonesty` — association with deceptive contexts (scams, propaganda, unreliable-narrator fiction, mislead-by-hedging).
- `honesty / forthrightness` — association with candid, sincere, disclosure-heavy contexts.
- *(optional)* `evasiveness / manipulation`, `sincerity`, `confidence-vs-hedging`.

Coefficient assignment per the methods above: **corpus-statistical** (deceptive-context co-occurrence frequency, $0) as the base, optionally **LLM-scored** for the abstract end. Explicitly **not** factuality/truth-value.

**The differentiator — name the feature, then *prove causality* (don't stop at correlation).** A corpus-derived "dishonesty" dimension is by default only a *correlate*: it fires for deceptive content but may not be what the model's deceptive *generation* routes through. WaveletLM's existing **FDA / dimensional-suppression / SOW** machinery does exactly the needed intervention — ablate or amplify the dishonesty dimension(s) and measure whether deceptive output **causally** changes. Two outcomes, both reportable:
- *Causal* → a load-bearing honesty handle (probe **and** steer) — a genuine MI/safety result.
- *Not causal* → the dimension is a surface correlate; say so honestly.

This intervention loop — name → ablate/amplify → measure causal effect on generation — is what separates a real result from a comfortable story, and WaveletLM is unusually well-equipped to run it (named features + dimensional-intervention tools on the same rails).

**Honest caveats (so the claim stays scoped):**
- **Monitor, not control.** A correlational feature is a smoke detector, not a suppressant — it flags deceptive *content*, doesn't prevent deceptive *generation*, and a capable model could deceive without lighting it up.
- **Use/mention contamination.** The dim fires for honest *discussion of* lying ("the scammer lied") as much as for lying — corpus co-occurrence can't separate use from mention. The [homonym separation](#homonym-separation-sense-disambiguated-tokens) machinery above may help (sense-split `lie_falsehood` vs `lie_recline`, etc.).
- **Token-level vs intention-level.** Deception is a sequence/output-level intention; a lie can be built from honest-seeming tokens. Per-token features catch loaded markers, not the deeper "is this statement a lie" structure.
- **Goodhart.** If the feature ever becomes a training *penalty*, the model optimizes to avoid the feature, not to be honest. Keep it a passive monitor or a causally-grounded steer — never a trainable proxy.

**Data dependence.** Like every capacity lever in this project, the correlations between deceptive behaviour and the feature only form robustly with *volume* — at WT-103/1ep this is a weak, noisy signal. The fair test is the [big-combined-dataset](../README.md#release-pipeline) regime; treat WT-103 results as a smoke test of the plumbing, not a verdict on the feature.

**Open questions:**
- Does the causal intervention move *generation-level* deception (held-out deceptive-task probes), or only token-level markers?
- Does sense-splitting deception-related tokens sharpen the dim (use/mention separation)?
- Is a single "honesty axis" enough, or does deception decompose into separable sub-features (evasion vs fabrication vs manipulation) that each need their own dim?
- Does the handle transfer across datasets, or is it corpus-specific?

## Key differences from EXARCH's current approach

1. **Feature construction**: Statistical/structural extraction from corpus, not LLM-based concept labeling. Much cheaper ($0 vs $200+/concept for FDA).
2. **Scalability**: N-gram features scale to any dimensionality needed.
3. **Alignment with wavelet processing**: Structural features (positional, periodic, co-occurrence at specific distances) directly correspond to what the wavelet decomposition captures at different scales.
4. **Hybrid approach**: Mix prescribed and learned dimensions rather than all-or-nothing.

## Implementation sketch

1. Run corpus analysis on WikiText-103: extract top-K 3-grams, compute token-level statistics (PMI, frequency, positional distributions)
2. Build a feature extraction function: `token_id -> feature_vector (C=2048)`
3. Store as a frozen embedding table (same as EXARCH's conceptual_embedding)
4. Optional: leave last N dimensions learnable (hybrid mode)
5. Train with same pipeline as EXARCH

## Evaluation plan

| Experiment | C | Prescribed dims | Learned dims | Comparison |
|-----------|------|----------------|-------------|------------|
| Full prescribed | 2048 | 2048 | 0 | vs learned C=2048 |
| Hybrid 80/20 | 2048 | 1636 | 412 | vs both above |
| Matched C=512 | 512 | 512 | 0 | vs EXARCH's 256-concept embedding |

## Open questions

- Does prescribing features that the model would learn anyway help (faster convergence) or hurt (suboptimal feature space)?
- Are n-gram features redundant with what the wavelet decomposition already captures?
- What's the optimal prescribed/learned ratio in hybrid mode?
- Can structural features transfer across datasets better than learned embeddings?

---

# NOTE: Claude Opus 5's Feedback (7/25/2026; doc created on 6/19/2026)

See below for feedback from Opus 5 on the preceding document.

-

Read it. There's a lot of good thinking in here, and one section I'd cut outright. Taking the substitution question first, then the AMB one, since the answer to the second depends on which construction you pick.

## The motivating comparison doesn't support the plan yet

Line 9 vs. line 10: EXARCH 1.0112 at C=512 with 256 concepts, learned 1.1431 at C=2048/1ep. Those differ on benchmark version (you flag this), C, and training budget simultaneously. You can't read "prescribed beats learned" off them — you can't even read the sign. Before building a 2048-dim feature pipeline, the cheap thing is to re-run the EXARCH 256-concept embedding under the v2 benchmark at matched C and matched budget. If the gap inverts, the plan's premise changes and you've spent a day instead of a month.

## The scalability claim is data-bound, not feature-bound

Line 12 says structural features "can scale indefinitely: there are far more n-gram patterns... than single-word human concepts." That's dimension-counting, not information-counting. At WT-103 (~100M tokens, 50k vocab) the V×V co-occurrence matrix has ~2.5e9 cells against 1e8 observations — most cells are zero or singleton, and PMI on a count of 1 is noise. The usable *rank* of the relational structure is bounded by the corpus, not by how many features you can enumerate.

Which means the 512 ceiling you attribute to "human concepts run out" may not be about human concepts at all. It may be the corpus's relational rank at this scale. And you already have one independent measurement pointing the same way: the SVD showing the POS mixers using ~38% of their dimensional capacity.

**This makes line 162 the most valuable line in the document.** Build the dyadic-bucket PMI tensor, factorize, and plot variance-explained vs. C *before training anything*. It costs a day of counting, it requires no LLM labeling, and it tells you whether C=2048 is reachable or whether the structure is exhausted at 400. If that curve dies where the mixer SVD died, you have two independent measurements of the same ceiling and the whole "scale to 2048" framing needs rewriting. I'd run this before anything else in the plan.

## Cut the n-gram identity features

They're 1000 of your 2048 dims (49%) and I think the category is malformed. The embedding is a V×C table — a *type-level* lookup. "This token participates in a specific 3-gram" is an *occurrence-level* property that depends on context the table cannot see. Your own rejection note at line 166 makes exactly this argument against absolute-position statistics ("position is a property of an occurrence, not of a type") — n-gram participation falls under the identical objection and slipped past it.

You can only put it in a frozen table by reducing it to "token v appears in 3-gram g with probability p", which is a unary marginal — i.e. it collapses into category 3, co-occurrence, and isn't a distinct feature family. Make it a runtime channel instead and you break the frozen-tied head (line 175), since the head would then have to predict n-gram membership; and forward-looking 3-grams leak the next token.

Two more strikes: backward-looking 3-grams are close to what scale-1 Haar coefficients already compute densely (your open question 2 — I think the answer is yes), and the top-1000 3-grams in WT-103 are almost entirely function-word sequences, which is a familiar failure surface for you given the RARE-category misclassification from EXARCH. Reallocate those 1000 dims to the PMI factorization, which produces them from the same corpus statistics with a principled ordering.

## The frozen-tied head is the real risk

Whitening handles the Zipf/norm problem you identify. It doesn't handle the harder one: with logits = h·Eᵀ and E frozen, two tokens whose feature rows are near-collinear become permanently inseparable in the output, no matter what the trunk learns. Prescribed embeddings produce near-duplicate rows constantly in the rare-token tail. Your per-vocab scalar gain fixes *norm* but not *direction*, so it can't pull collinear rows apart — that's a hard BPB floor, and it's the mechanism I'd bet on if the frozen-head arm underperforms.

Cheapest escape that preserves the interpretability artifact: a learned low-rank residual on the head, logits = h·(E + UVᵀ)ᵀ at rank 32–64. The named-frame attribution `h_i·E[v]_i` survives, r=0 is a clean ablation, and ‖UVᵀ‖ becomes a reported number — literally "how much the prescribed frame had to be rotated to work."

Also, run **frozen-E-no-PE first**, not second. Your EXARCH result that positional encoding provided no benefit (Haar handles it) is a strong prior that the PE channel is redundant, and the no-PE arm is the simpler architecture.

## On the hybrid ratio

The doc proposes one 80/20 point. The informative object is the *curve* — marginal BPB cost per prescribed dimension — because it separates the cheap prescriptions (structural, frequency, positional: things the model would learn anyway) from the expensive ones (the abstract semantic end).

And rather than partitioning dims into prescribed vs. learned blocks, consider the cheapest hybrid that keeps names intact: **frozen E with one learned scalar gain per dimension**. Dimension identity is preserved exactly, it's 2048 parameters, and the learned gains are a direct readout of which prescribed features the model actually wants. That's a free feature-importance artifact and a much better use of a first run than picking a ratio blind.

## The AMB question: no to random, and the subset isn't the right axis

CEv2 was never the point — the properties were. The AMB wants keys that are **sparse, near-orthogonal, and named**, because that's what simultaneously reduces interference (capacity) and makes S a legible binding matrix (interpretability). Any embedding with those properties qualifies.

A random subset fails on the first two, and specifically fails in the way we just diagnosed. Under your budget table, a random 64 of 2048 draws ~49% n-gram indicator dims — near-constant zero across the vocab — plus a few Zipf-dominated frequency bands. Near-constant key coordinates are exactly the DC floor that made `softplus_l2` unselective. You'd be reintroducing the same pathology through the input side after fixing it in the feature map.

Better, in rough order of how much I'd trust them:

1. **Split the roles.** Take **q and k from the frozen embedding** (type-level, so matching is on token identity — which is what MQAR-style recall actually needs) and keep **v from the contextual residual** (or you can only ever retrieve a neighbor's embedding, never the bound value). This is a classic content-addressable memory keyed by identity, and it should beat the all-contextual version on recall probes while making every write attributable to a named token.
2. **Select dims by pivoted QR, not randomly.** Column-pivoted QR on Eᵀ gives you d near-orthogonal, high-variance, *still-named* columns for free. That's the selection rule that actually buys what random sampling was hoping to approximate.
3. **If you adopt the dyadic-bucket construction, select from the long-offset buckets** (64–127, 128–255). The AMB exists to supply the long-range content-addressed retrieval the local wavelet scales miss, so key it on the coordinates that encode long-range relational structure. This is the most architecturally motivated rule available and it only exists if you go the PMI route.

Keep frozen-vs-learned keys as an ablation arm rather than an assumption, though — frozen keys mean the AMB can't learn *what* to match on, which is a real capability cost. The middle option (learned W applied to the frozen named frame) probably wins: you retain partial attribution through W's row structure without giving up adaptivity.

On the repo: worth knowing that I won't be picking this thread up on my own initiative — when the extension access lands I'll be reading things fresh, so pointing me at the relevant plan files and the current config at the start of that session will save you a round trip.