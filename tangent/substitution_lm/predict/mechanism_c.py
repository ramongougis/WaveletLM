"""
Mechanism C — Hybrid Re-rank.

Generate top-HYBRID_K candidates using Mechanism A (fast baseline), then
re-rank by summing compatibility scores from all context nodes.  Combines A's
efficient candidate generation with a richer multi-context compatibility signal.

K is a tunable hyperparameter (config.HYBRID_K).  If re-ranking produces no
change over A, that is a diagnostic signal that property tables are not adding
information beyond co-occurrence.
"""

import math
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C
from predict import compat_score, compatibility
import predict.mechanism_a as mech_a


def predict_logprobs(context_ids: list[int], model: dict) -> dict[int, float]:
    logprob  = model["logprob"]
    props    = model["properties"]
    id2ngram = model["id2ngram"]
    uni_lp   = model["unigram_logprob"]

    # Step 1: get top-K candidates from Mechanism A
    a_lps = mech_a.predict_logprobs(context_ids, model)
    if not a_lps:
        return {nid: lp for nid, lp in uni_lp.items()}

    top_k_ids = sorted(a_lps, key=lambda x: a_lps[x], reverse=True)[: C.HYBRID_K]

    # Step 2: re-rank by summing compatibility with all context nodes
    window = context_ids[-C.CONTEXT_SIZE:]
    rerank_scores: dict[int, float] = {}
    for cand_id in top_k_ids:
        base_prob = math.exp2(a_lps[cand_id])
        cand_word = id2ngram.get(cand_id, "")
        compat_sum = 0.0
        for ctx_id in window:
            ctx_props = props.get(ctx_id)
            if ctx_props is not None:
                compat_sum += compatibility(ctx_props, cand_word)
            else:
                compat_sum += 1.0
        avg_compat = compat_sum / max(len(window), 1)
        rerank_scores[cand_id] = base_prob * avg_compat

    # Zero-compat candidates are already penalised through avg_compat → 0
    total = sum(rerank_scores.values())
    if total == 0:
        return {nid: lp for nid, lp in uni_lp.items()}

    return {nid: math.log2(s / total) for nid, s in rerank_scores.items()}


def top_k(context_ids: list[int], model: dict, k: int = 10) -> list[tuple[str, float]]:
    id2ngram = model["id2ngram"]
    lps = predict_logprobs(context_ids, model)
    ranked = sorted(lps.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(id2ngram.get(nid, f"<{nid}>"), lp) for nid, lp in ranked]
