"""
Shared utilities for all prediction mechanisms.

Exports:
    load_model()                  — load model.pkl once and cache in memory
    tokenize(text, model)         — text → list of node IDs (longest-match greedy)
    compatibility(ctx_props, w)   — does the candidate word fit the context's
                                    ConceptNet selectional constraints?
    compat_score(cid, nid, model) — convenience wrapper using node IDs
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


def compatibility(ctx_props: dict, cand_word: str) -> float:
    """
    Does the candidate word satisfy the context node's selectional constraints?

    Lookup is directional: we check whether the surface form `cand_word`
    appears in the context's ConceptNet relation lists. This is the right
    polarity for substitution / next-token scoring — "is the candidate
    something this context can/can't do or have?"

        0.0  — hard contradiction (cand_word ∈ ctx.{NotCapableOf, NotHasProperty})
        >1.0 — positive association; multiplicative across matched relations
        1.0  — neutral / no signal

    Relation → boost mapping reflects approximate signal strength:
        CapableOf       1.5   — strong selectional fit (verb after agent)
        HasProperty     1.3   — attribute after subject
        IsA             1.3   — taxonomic link
        UsedFor         1.3   — instrument → action / object → purpose
        Causes          1.2   — causal consequence
        ReceivesAction  1.2   — passive-role link (food ReceivesAction eat)
        AtLocation      1.2   — locational co-occurrence
        RelatedTo       1.1   — generic broad association (weakest)
    """
    if not ctx_props or not cand_word:
        return 1.0

    if cand_word in ctx_props.get("NotCapableOf", ()):
        return 0.0
    if cand_word in ctx_props.get("NotHasProperty", ()):
        return 0.0

    boost = 1.0
    if cand_word in ctx_props.get("CapableOf", ()):
        boost *= 1.5
    if cand_word in ctx_props.get("HasProperty", ()):
        boost *= 1.3
    if cand_word in ctx_props.get("IsA", ()):
        boost *= 1.3
    if cand_word in ctx_props.get("UsedFor", ()):
        boost *= 1.3
    if cand_word in ctx_props.get("Causes", ()):
        boost *= 1.2
    if cand_word in ctx_props.get("ReceivesAction", ()):
        boost *= 1.2
    if cand_word in ctx_props.get("AtLocation", ()):
        boost *= 1.2
    if cand_word in ctx_props.get("RelatedTo", ()):
        boost *= 1.1
    return boost


def compat_score(ctx_id: int, cand_id: int, model: dict) -> float:
    """Convenience wrapper using node IDs."""
    return compatibility(
        model["properties"].get(ctx_id, {}),
        model["id2ngram"].get(cand_id, ""),
    )
