"""
Mechanism A — Weighted Edge Walk.

Given a context window of node IDs, retrieve all outgoing edges from each
context node, multiply edge log-prob by the compatibility score between the
context node and the candidate, sum across context nodes, then normalise to
get log P(next | context).

If a candidate has no compatible context edges, falls back to the unigram
log-probability (with a penalty of UNK_LOGPROB if truly absent).

Returns: dict[candidate_node_id, log2_probability]
"""

import math
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C
from predict import compat_score, compatibility


def predict_logprobs(context_ids: list[int], model: dict) -> dict[int, float]:
    """
    Args:
        context_ids: ordered list of recent node IDs (most recent last)
        model:       loaded model dict from predict.load_model()

    Returns:
        dict mapping candidate_node_id → log2 probability (unnormalised scores
        are converted to a proper distribution via log-sum-exp normalisation).
    """
    logprob   = model["logprob"]
    props     = model["properties"]
    uni_lp    = model["unigram_logprob"]

    # Accumulate raw scores in linear space (sum of weighted edge probs)
    scores: dict[int, float] = defaultdict(float)

    # Use the most recent CONTEXT_SIZE tokens, weighted by recency
    window = context_ids[-C.CONTEXT_SIZE:]
    for rank, ctx_id in enumerate(window):
        recency_weight = 1.0 + 0.2 * rank
        ctx_props = props.get(ctx_id)  # None if not in ConceptNet
        for cand_id, lp in logprob.get(ctx_id, {}).items():
            if ctx_props is not None:
                cand_props = props.get(cand_id)
                if cand_props is not None:
                    compat = compatibility(ctx_props, cand_props)
                    if compat == 0.0:
                        continue
                    scores[cand_id] += math.exp2(lp) * compat * recency_weight
                    continue
            scores[cand_id] += math.exp2(lp) * recency_weight

    if not scores:
        # Full backoff to unigram distribution
        return {nid: lp for nid, lp in uni_lp.items()}

    # Normalise to a probability distribution, return as log2
    total = sum(scores.values())
    return {nid: math.log2(s / total) for nid, s in scores.items()}


def top_k(context_ids: list[int], model: dict, k: int = 10) -> list[tuple[str, float]]:
    """Return top-k (ngram_string, log2_prob) pairs for display."""
    id2ngram = model["id2ngram"]
    lps = predict_logprobs(context_ids, model)
    ranked = sorted(lps.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(id2ngram.get(nid, f"<{nid}>"), lp) for nid, lp in ranked]
