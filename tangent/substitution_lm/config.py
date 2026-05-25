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
CONTEXT_SIZE = 5   # number of prior tokens used as context window
HYBRID_K     = 50  # Mechanism C: candidates from A before re-ranking

# ── Evaluation ─────────────────────────────────────────────────────────────
UNK_LOGPROB = -15.0  # log₂ fallback for tokens not reachable from context
