# Structural Semantic Embedding: Beyond Concept Labels

> This plan is duplicated in EXARCH-semantic (formerly EXARCH-research) in ../EXARCH-research/plans/reincorporate_large_semantic_embedding.md

## Status: Post-release exploration

## Approach: Fork current WaveletLM project post-release and selectively reincorporate
semantic embedding functionality. This keeps the main WaveletLM repo clean (learned
embedding only) while the fork adds optional prescribed embedding support as a
separate, well-structured codebase suitable for its own paper.

## Summary

Scale the semantic embedding approach to C=2048+ by replacing human-interpretable
concept labels with structural, statistical, and syntactic features derived from
corpus analysis. Instead of "violence" or "happiness," dimensions would encode
n-gram co-occurrence patterns, positional statistics, and token-level structural
properties that more closely mirror what learned embeddings discover on their own.

## Motivation

- EXARCH-semantic achieved BPB 1.0112 at C=512 with 256 human-defined concepts
- Learned embeddings at C=2048 achieve BPB 1.1431 at 1 epoch (and improving)
- Human-interpretable concepts hit a ceiling around ~500 orthogonal dimensions —
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
