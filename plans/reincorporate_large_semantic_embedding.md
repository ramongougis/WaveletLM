# Structural Semantic Embedding: Beyond Concept Labels

## Summary

Scale the semantic embedding approach to C=2048+ by replacing human-interpretable
concept labels with structural, statistical, and syntactic features derived from
corpus analysis. Instead of "violence" or "happiness," dimensions would encode
n-gram co-occurrence patterns, positional statistics, and token-level structural
properties that more closely mirror what learned embeddings discover on their own.

## Motivation

- EXARCH-semantic (the previous version of WaveletLM) achieved BPB 1.0112 at C=512 with 256 human-defined concepts
- Learned embeddings at C=2048 achieve BPB 1.1431 at 1 epoch (and improving)
- Human-interpretable concepts hit a ceiling around ~500 orthogonal dimensions -
  language doesn't have 2048 independent semantic categories
- But structural/statistical features can scale indefinitely: there are far more
  n-gram patterns, positional encodings, and co-occurrence statistics than there
  are human concepts
- Question: can a hand-crafted structural embedding match or beat learned embeddings
  at high C, while retaining the interpretability benefits of prescribed features?

## Proposed feature categories

### 1. N-gram identity features (scalable)
- Common 3-grams from the training corpus
- Each dimension indicates whether the current token participates in a specific 3-gram
- With vocab=50257 and n=3, there are billions of possible 3-grams, but the top
  10,000-50,000 cover most of the distribution
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

### 5. Traditional semantic features (retained from EXARCH-semantic)
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

The "headroom" dimensions could be left as learnable (zero-initialized), creating
a hybrid embedding where ~80% is prescribed and ~20% is learned. This tests whether
the prescribed features cover the important dimensions and the model only needs to
learn the remainder.

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

## Key differences from EXARCH-semantic's current approach

1. **Feature construction**: Statistical/structural extraction from corpus, not
   LLM-based concept labeling. Much cheaper ($0 vs $200+/concept for FDA).
2. **Scalability**: N-gram features scale to any dimensionality needed.
3. **Alignment with wavelet processing**: Structural features (positional, periodic,
   co-occurrence at specific distances) directly correspond to what the wavelet
   decomposition captures at different scales.
4. **Hybrid approach**: Mix prescribed and learned dimensions rather than all-or-nothing.

## Implementation sketch

1. Run corpus analysis on WikiText-103: extract top-K 3-grams, compute token-level
   statistics (PMI, frequency, positional distributions)
2. Build a feature extraction function: `token_id -> feature_vector (C=2048)`
3. Store as a frozen embedding table (same as EXARCH-semantic's conceptual_embedding)
4. Optional: leave last N dimensions learnable (hybrid mode)
5. Train with same pipeline as EXARCH-semantic

## Evaluation plan

| Experiment | C | Prescribed dims | Learned dims | Comparison |
|-----------|------|----------------|-------------|------------|
| Full prescribed | 2048 | 2048 | 0 | vs learned C=2048 |
| Hybrid 80/20 | 2048 | 1636 | 412 | vs both above |
| Matched C=512 | 512 | 512 | 0 | vs EXARCH-semantic's 256-concept embedding |

## Open questions

- Does prescribing features that the model would learn anyway help (faster convergence)
  or hurt (suboptimal feature space)?
- Are n-gram features redundant with what the wavelet decomposition already captures?
- What's the optimal prescribed/learned ratio in hybrid mode?
- Can structural features transfer across datasets better than learned embeddings?
