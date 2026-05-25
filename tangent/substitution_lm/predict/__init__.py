"""
Shared utilities for all prediction mechanisms.

Exports:
    load_model()              — load model.pkl once and cache in memory
    tokenize(text, model)     — text → list of node IDs (longest-match greedy)
    compatibility(pa, pb)     — ConceptNet-based compatibility score in [0, ∞)
    compat_score(a, b, props) — convenience wrapper using node IDs
"""

import math
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C


_MODEL_CACHE = None

def load_model() -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        with open(C.MODEL_PATH, "rb") as f:
            _MODEL_CACHE = pickle.load(f)
    return _MODEL_CACHE


def tokenize(text: str, model: dict) -> list[int]:
    """Convert whitespace-tokenised text to node IDs via greedy longest-match."""
    ngram2id = model["ngram2id"]
    words = text.lower().split()
    n = len(words)
    ids = []
    i = 0
    while i < n:
        best_id, best_len = None, 0
        for length in range(min(C.MAX_NGRAM, n - i), 0, -1):
            phrase = " ".join(words[i: i + length])
            nid = ngram2id.get(phrase)
            if nid is not None:
                best_id, best_len = nid, length
                break
        if best_id is not None:
            ids.append(best_id)
            i += best_len
        else:
            i += 1
    return ids


def compatibility(props_a: dict, props_b: dict) -> float:
    """
    Compatibility score between two property dicts.
        0.0  — hard contradiction (NotCapableOf / NotHasProperty clash)
        1.0  — neutral (no signal)
        >1.0 — positive association (shared RelatedTo concepts)
    """
    if not props_a or not props_b:
        return 1.0

    a_cant    = set(props_a.get("NotCapableOf",   []))
    b_can     = set(props_b.get("CapableOf",       []))
    b_cant    = set(props_b.get("NotCapableOf",   []))
    a_can     = set(props_a.get("CapableOf",       []))
    a_not_has = set(props_a.get("NotHasProperty", []))
    b_has     = set(props_b.get("HasProperty",     []))
    b_not_has = set(props_b.get("NotHasProperty", []))
    a_has     = set(props_a.get("HasProperty",     []))

    if (a_cant & b_can) or (b_cant & a_can):
        return 0.0
    if (a_not_has & b_has) or (b_not_has & a_has):
        return 0.0

    shared = len(set(props_a.get("RelatedTo", [])) & set(props_b.get("RelatedTo", [])))
    return 1.0 + 0.1 * shared


def compat_score(nid_a: int, nid_b: int, properties: dict) -> float:
    return compatibility(properties.get(nid_a, {}), properties.get(nid_b, {}))
