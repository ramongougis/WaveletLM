"""
Shared configuration for the substitution LM pipeline.

WikiText-103 is loaded via HuggingFace datasets (same cache WaveletLM uses).
No raw text files needed on disk.
"""
import os
from pathlib import Path

# ── Force CPU: this project must not consume VRAM alongside WaveletLM ──────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"    # gitignored — large generated artefacts
CACHE_DIR = ROOT / "cache"   # gitignored — intermediate parse/count caches

# ── WikiText-103 via HuggingFace datasets ─────────────────────────────────
# train.py already downloaded this dataset; we reuse the same HF cache.
# load_dataset("wikitext", "wikitext-103-raw-v1") is called in 01_parse.py.
# Each example has a "text" field; articles are separated by "= Title =" lines.
HF_DATASET  = "wikitext"
HF_CONFIG   = "wikitext-103-raw-v1"

# ── Intermediate cache files ───────────────────────────────────────────────
PARSE_CACHE_TRAIN = CACHE_DIR / "parse_train.jsonl"
PARSE_CACHE_TEST  = CACHE_DIR / "parse_test.jsonl"
UNIGRAM_CACHE     = CACHE_DIR / "unigrams.pkl"
BIGRAM_CACHE      = CACHE_DIR / "bigrams.pkl"

# ── Final data files ───────────────────────────────────────────────────────
NODE_TABLE_PATH   = DATA_DIR / "node_table.pkl"
DAG_PATH          = DATA_DIR / "dag.pkl"
PROPERTIES_PATH   = DATA_DIR / "properties.pkl"
MODEL_PATH        = DATA_DIR / "model.pkl"

# ── Extraction parameters ──────────────────────────────────────────────────
MAX_NGRAM    = 5    # maximum n-gram length (tokens); keep ≤5 to control memory
MIN_FREQ     = 5    # minimum corpus occurrences to admit a node
PMI_MIN      = 3.0  # minimum PMI to treat a phrase as an atomic collocation
COLLOC_FREQ  = 10   # minimum raw frequency for collocation candidates

# ── SpaCy ──────────────────────────────────────────────────────────────────
SPACY_MODEL     = "en_core_web_lg"
SPACY_BATCH     = 64    # documents per SpaCy pipe call (memory vs. speed)
SPACY_N_PROCESS = 1     # parallel workers; set >1 carefully (memory per worker)
PARSE_CHUNK     = 2000  # docs to accumulate before flushing JSONL cache

# ── DAG ────────────────────────────────────────────────────────────────────
CO_OCC_WINDOW = 10  # max token offset between two nodes within a clause

# ── ConceptNet ─────────────────────────────────────────────────────────────
# Place conceptnet-assertions-5.7.0.csv in this directory (gitignored).
# Download: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
CONCEPTNET_CSV  = ROOT / "conceptnet-assertions-5.7.0.csv"
CONCEPTNET_RELS = frozenset({
    "HasProperty", "NotHasProperty",
    "CapableOf",   "NotCapableOf",
    "IsA",         "RelatedTo",
    "UsedFor",     "AtLocation",
    "Causes",      "ReceivesAction",
})

# ── Prediction ─────────────────────────────────────────────────────────────
CONTEXT_SIZE      = 10    # number of prior tokens used as context window
HYBRID_K          = 500   # Mechanism C: candidates from A before re-ranking
MAX_EDGES_PER_SRC = 500  # max outgoing edges stored per source node (caps predict cost)

# ── MLP Property Classifier (Phase 2) ──────────────────────────────────────
# When enabled, fills gaps in ConceptNet coverage by imputing properties from
# word embeddings. ConceptNet entries always override MLP predictions
# (Reiter-style exceptions override the default classifier).
#
# Two noise-control levers (see design notes — both can be combined):
#   MLP_PROPERTY_COVERAGE_PCT: train/predict only the top-N% of (relation,
#       target_word) pairs by frequency in ConceptNet. Rare targets without
#       enough training labels are dropped entirely; MLP capacity concentrates
#       on well-supported features. Lower N → less noise, more sparsity.
#   MLP_PROPERTY_CONFIDENCE:   at inference, only treat MLP sigmoid outputs
#       above this threshold as positive predictions (and below 1−threshold
#       as negative). Predictions in between are "unknown" and contribute
#       nothing to compatibility(). Per-prediction noise control.
MLP_PROPERTIES_ENABLED      = True                           # set True after training
MLP_PROPERTY_COVERAGE_PCT   = 80                              # top-N% (per relation) of target
                                                              # words to predict, by frequency
MLP_PROPERTY_CONFIDENCE     = 0.7                             # sigmoid threshold at inference
MLP_PROPERTY_PATH           = DATA_DIR / "mlp_properties.pkl" # imputed properties dict
MLP_PROPERTY_MODEL_PATH     = DATA_DIR / "mlp_property_classifier.pt"
MLP_PROPERTY_EMBED_DIM      = 300                             # GloVe-300 input dim
MLP_PROPERTY_HIDDEN_DIM     = 256                             # only used if LAYERS > 0
MLP_PROPERTY_LAYERS         = 0                               # hidden layers; 0 = linear probe
                                                              # (fully interpretable per-property
                                                              # weight vector). Bump to 1 only
                                                              # if linear demonstrably underfits.
MLP_PROPERTY_EPOCHS         = 5                               # 5 is usually enough for a linear
                                                              # probe on ~100K samples; loss
                                                              # plateaus by epoch 3–5
MLP_PROPERTY_BATCH_SIZE     = 1024                            # larger = fewer iterations, less
                                                              # per-iter Python/BLAS overhead;
                                                              # ~313 MB per-batch dense Y at K=76K

# ── Evaluation ─────────────────────────────────────────────────────────────
UNK_LOGPROB = -15.0  # log₂ fallback for tokens not reachable from context
