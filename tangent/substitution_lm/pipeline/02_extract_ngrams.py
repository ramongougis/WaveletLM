"""
Step 2 — Extract n-gram nodes from the parse cache.

Two passes over the JSONL cache:

Pass A — count unigrams and bigrams to compute PMI for collocation detection.
          Common collocations (PMI ≥ PMI_MIN, freq ≥ COLLOC_FREQ) are promoted
          to atomic phrase nodes.

Pass B — extract candidate n-gram nodes. Two modes (config.EXTRACT_SENTENCE_NGRAMS):
          True  — every n-gram of length 2..MAX_NGRAM from POS-filtered
                  sentence lemma streams. Subsumes NP-chunk extraction; also
                  captures cross-boundary patterns (function-word + content-word,
                  VP / PP / subordinate-clause spans) which dominate the
                  unaccounted n-gram distribution.
          False — collocation bigrams + NP-chunk sub-n-grams only (original
                  behaviour).

Writes: NODE_TABLE_PATH — dict with:
    {
      "id2ngram":  dict[int, str],
      "ngram2id":  dict[str, int],
      "freq":      dict[int, int],
      "collocations": set[str],   # atomic phrase strings
    }

VRAM: none.
RAM:
    EXTRACT_SENTENCE_NGRAMS=False: O(unique n-grams × ~200 bytes) ≈ 2–4 GB
                                   unfiltered; < 500 MB after MIN_FREQ pruning.
    EXTRACT_SENTENCE_NGRAMS=True:  10–15 GB peak (bounded by periodic
                                   singleton pruning when Counter exceeds
                                   ~40M entries); 1–4 GB after MIN_FREQ.
"""

import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C


# ── Pass A: count unigrams and bigrams ────────────────────────────────────

def _count_pass(cache_path: Path):
    unigrams: Counter = Counter()
    bigrams:  Counter = Counter()

    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            for sent in doc["sentences"]:
                lemmas = [t["lemma"] for t in sent["tokens"] if t["pos"] not in ("PUNCT", "SPACE", "SYM", "NUM")]
                for lem in lemmas:
                    unigrams[lem] += 1
                for a, b in zip(lemmas, lemmas[1:]):
                    bigrams[(a, b)] += 1

    return unigrams, bigrams


def _compute_collocations(unigrams: Counter, bigrams: Counter, total: int) -> set:
    collocations = set()
    for (a, b), cnt in bigrams.items():
        if cnt < C.COLLOC_FREQ:
            continue
        pa = unigrams[a] / total
        pb = unigrams[b] / total
        pab = cnt / total
        if pa > 0 and pb > 0:
            pmi = math.log2(pab / (pa * pb))
            if pmi >= C.PMI_MIN:
                collocations.add(f"{a} {b}")
    return collocations


# ── Pass B: extract n-gram nodes ──────────────────────────────────────────

def _ngrams_from_chunk(chunk_text: str, collocations: set) -> list:
    """Return all sub-n-grams of a noun-phrase chunk up to MAX_NGRAM tokens."""
    tokens = chunk_text.split()
    n = len(tokens)
    results = []
    for start in range(n):
        for end in range(start + 1, min(start + C.MAX_NGRAM, n) + 1):
            results.append(" ".join(tokens[start:end]))
    return results


def _extract_pass(cache_path: Path, collocations: set) -> Counter:
    """Count all candidate n-gram nodes; dispatches by EXTRACT_SENTENCE_NGRAMS."""
    if C.EXTRACT_SENTENCE_NGRAMS:
        return _extract_pass_sentence_stream(cache_path)
    return _extract_pass_np_chunks(cache_path, collocations)


def _extract_pass_np_chunks(cache_path: Path, collocations: set) -> Counter:
    """Original behaviour: unigrams + collocation bigrams + NP-chunk sub-n-grams."""
    node_freq: Counter = Counter()
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            for sent in doc["sentences"]:
                lemmas = [t["lemma"] for t in sent["tokens"]
                          if t["pos"] not in ("PUNCT", "SPACE", "SYM", "NUM")]
                for a, b in zip(lemmas, lemmas[1:]):
                    phrase = f"{a} {b}"
                    if phrase in collocations:
                        node_freq[phrase] += 1
                for lem in lemmas:
                    node_freq[lem] += 1
            for chunk in doc["chunks"]:
                for ngram in _ngrams_from_chunk(chunk["text"], collocations):
                    node_freq[ngram] += 1
    return node_freq


def _extract_pass_sentence_stream(cache_path: Path) -> Counter:
    """Every n-gram of length 1..MAX_NGRAM from POS-filtered sentence streams.

    Memory-bounded by periodic singleton pruning: when the Counter exceeds
    ~40M entries, drop everything with count==1. Bounded loss for very-rare
    n-grams at MIN_FREQ=2 (a true-count-2 n-gram could be lost if its two
    occurrences straddle a prune event; this is rare).
    """
    PRUNE_EVERY   = 500_000      # docs between memory checks
    PRUNE_AT_SIZE = 40_000_000   # Counter entries threshold
    max_n = C.MAX_NGRAM

    node_freq: Counter = Counter()
    n_docs = 0

    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            for sent in doc["sentences"]:
                lemmas = [t["lemma"] for t in sent["tokens"]
                          if t["pos"] not in ("PUNCT", "SPACE", "SYM", "NUM")]
                L = len(lemmas)
                if not L:
                    continue
                # Unigrams
                for lem in lemmas:
                    node_freq[lem] += 1
                # All sentence-stream n-grams 2..MAX_NGRAM
                for n in range(2, max_n + 1):
                    if L < n:
                        break
                    for i in range(L - n + 1):
                        node_freq[" ".join(lemmas[i:i + n])] += 1

            n_docs += 1
            if n_docs % PRUNE_EVERY == 0 and len(node_freq) > PRUNE_AT_SIZE:
                pre = len(node_freq)
                node_freq = Counter({k: v for k, v in node_freq.items() if v >= 2})
                print(f"  [{n_docs:,} docs] singleton prune: "
                      f"{pre:,} → {len(node_freq):,}", flush=True)

    return node_freq


# ── Main ──────────────────────────────────────────────────────────────────

def run() -> None:
    if C.NODE_TABLE_PATH.exists():
        print(f"[02_extract] Node table already exists: {C.NODE_TABLE_PATH}  (delete to re-extract)")
        return

    if not C.PARSE_CACHE_TRAIN.exists():
        print("[02_extract] Parse cache missing — run 01_parse.py first.")
        sys.exit(1)

    C.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pass A ────────────────────────────────────────────────────────────
    print("[02_extract] Pass A: counting unigrams and bigrams …")
    t0 = time.time()
    unigrams, bigrams = _count_pass(C.PARSE_CACHE_TRAIN)
    total = sum(unigrams.values())
    print(f"  Unigrams: {len(unigrams):,}  Bigrams: {len(bigrams):,}  Tokens: {total:,}  ({time.time()-t0:.0f}s)")

    collocations = _compute_collocations(unigrams, bigrams, total)
    print(f"  Collocations (PMI≥{C.PMI_MIN}, freq≥{C.COLLOC_FREQ}): {len(collocations):,}")

    # Cache raw counts for DAG step
    with open(C.UNIGRAM_CACHE, "wb") as f:
        pickle.dump(unigrams, f)
    with open(C.BIGRAM_CACHE, "wb") as f:
        pickle.dump(bigrams, f)

    # ── Pass B ────────────────────────────────────────────────────────────
    mode = "sentence-stream (all n-grams)" if C.EXTRACT_SENTENCE_NGRAMS else "NP-chunks only"
    print(f"[02_extract] Pass B: extracting n-gram nodes — mode: {mode} …")
    t0 = time.time()
    node_freq = _extract_pass(C.PARSE_CACHE_TRAIN, collocations)
    print(f"  Raw candidates: {len(node_freq):,}  ({time.time()-t0:.0f}s)")

    # Prune by frequency threshold
    node_freq = Counter({k: v for k, v in node_freq.items() if v >= C.MIN_FREQ})
    print(f"  After freq≥{C.MIN_FREQ} filter: {len(node_freq):,}")

    # Assign integer IDs (sorted by frequency desc for stable ordering)
    id2ngram = {i: ng for i, (ng, _) in enumerate(node_freq.most_common())}
    ngram2id = {ng: i for i, ng in id2ngram.items()}
    freq      = {ngram2id[ng]: cnt for ng, cnt in node_freq.items()}

    table = {
        "id2ngram":     id2ngram,
        "ngram2id":     ngram2id,
        "freq":         freq,
        "collocations": collocations,
    }

    with open(C.NODE_TABLE_PATH, "wb") as f:
        pickle.dump(table, f)

    print(f"[02_extract] Saved {len(id2ngram):,} nodes → {C.NODE_TABLE_PATH}")


if __name__ == "__main__":
    run()
