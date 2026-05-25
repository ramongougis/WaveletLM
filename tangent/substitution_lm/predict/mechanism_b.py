"""
Mechanism B — Aggregated Vote.

Every context node votes for every candidate it has an edge to.  Each vote is
weighted by the context node's own activation (recency-decayed) × edge weight
× compatibility penalty (multiplicative, not hard filter).  Negative votes
(compat = 0) set the candidate score to zero regardless of other votes.

Expected to improve over A at larger corpus sizes where multiple context nodes
independently confirm a candidate.
"""

import math
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C
from predict import compat_score, compatibility


def predict_logprobs(context_ids: list[int], model: dict) -> dict[int, float]:
    logprob  = model["logprob"]
    props    = model["properties"]
    id2ngram = model["id2ngram"]
    uni_lp   = model["unigram_logprob"]

    scores:       dict[int, float] = defaultdict(float)
    contradicted: set[int]         = set()

    window = context_ids[-C.CONTEXT_SIZE:]
    for rank, ctx_id in enumerate(window):
        activation = 1.0 + 0.2 * rank
        ctx_props = props.get(ctx_id)
        for cand_id, lp in logprob.get(ctx_id, {}).items():
            if cand_id in contradicted:
                continue
            if ctx_props is not None:
                compat = compatibility(ctx_props, id2ngram.get(cand_id, ""))
                if compat == 0.0:
                    contradicted.add(cand_id)
                    scores.pop(cand_id, None)
                    continue
                scores[cand_id] += math.exp2(lp) * compat * activation
            else:
                scores[cand_id] += math.exp2(lp) * activation

    if not scores:
        return {nid: lp for nid, lp in uni_lp.items()}

    total = sum(scores.values())
    return {nid: math.log2(s / total) for nid, s in scores.items()}


def top_k(context_ids: list[int], model: dict, k: int = 10) -> list[tuple[str, float]]:
    id2ngram = model["id2ngram"]
    lps = predict_logprobs(context_ids, model)
    ranked = sorted(lps.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(id2ngram.get(nid, f"<{nid}>"), lp) for nid, lp in ranked]
