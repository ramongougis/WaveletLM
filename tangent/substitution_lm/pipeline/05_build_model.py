"""
Step 5 — Assemble and save the final model object.

Loads the node table, DAG, and property table produced by steps 2–4,
normalises edge weights to log-probabilities, and writes a single MODEL_PATH
pickle that the prediction and evaluation modules load.

Model dict structure:
    {
      "id2ngram":    dict[int, str],
      "ngram2id":    dict[str, int],
      "freq":        dict[int, int],          # raw corpus frequency per node
      "collocations": set[str],
      "logprob":     dict[int, dict[int, float]], # log2 P(dst | src)
      "properties":  dict[int, dict[str, list]],  # ConceptNet properties
      "unigram_logprob": dict[int, float],         # fallback unigram log2 probs
    }

VRAM: none.  RAM: dominated by logprob dict.
"""

import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C


def _to_logprob(dag_raw: dict, min_count: int = 2) -> dict:
    """Convert raw count DAG to log2-probability DAG (row-normalised).

    Edges with count < min_count are pruned (noise). High-fan-out source
    nodes are capped at MAX_EDGES_PER_SRC by keeping the most frequent edges,
    bounding per-token prediction cost in pure Python.
    """
    logprob = {}
    max_edges = C.MAX_EDGES_PER_SRC
    for src, dsts in dag_raw.items():
        filtered = {dst: cnt for dst, cnt in dsts.items() if cnt >= min_count}
        if not filtered:
            continue
        if len(filtered) > max_edges:
            filtered = dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:max_edges])
        total = sum(filtered.values())
        logprob[src] = {dst: math.log2(cnt / total) for dst, cnt in filtered.items()}
    return logprob


def run() -> None:
    if C.MODEL_PATH.exists():
        print(f"[05_build_model] Model already exists: {C.MODEL_PATH}  (delete to rebuild)")
        return

    for dep in (C.NODE_TABLE_PATH, C.DAG_PATH, C.PROPERTIES_PATH):
        if not dep.exists():
            print(f"[05_build_model] Missing: {dep} — run earlier steps first.")
            sys.exit(1)

    t0 = time.time()
    print("[05_build_model] Loading components …")

    with open(C.NODE_TABLE_PATH, "rb") as f:
        table = pickle.load(f)
    with open(C.DAG_PATH, "rb") as f:
        dag_raw = pickle.load(f)
    with open(C.PROPERTIES_PATH, "rb") as f:
        properties = pickle.load(f)

    print("  Computing log-probability DAG …")
    logprob = _to_logprob(dag_raw)
    del dag_raw  # free raw counts — not stored in model, not used by predictors

    # Unigram fallback: log2(freq / total_freq)
    total_freq = sum(table["freq"].values())
    unigram_logprob = {
        nid: math.log2(cnt / total_freq)
        for nid, cnt in table["freq"].items()
    }

    model = {
        "id2ngram":         table["id2ngram"],
        "ngram2id":         table["ngram2id"],
        "freq":             table["freq"],
        "collocations":     table["collocations"],
        "logprob":          logprob,
        "properties":       properties,
        "unigram_logprob":  unigram_logprob,
    }

    with open(C.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    n_edges = sum(len(v) for v in logprob.values())
    coverage = 100.0 * len(properties) / max(len(table["id2ngram"]), 1)
    print(
        f"[05_build_model] Model saved → {C.MODEL_PATH}\n"
        f"  Nodes    : {len(table['id2ngram']):,}\n"
        f"  Edges    : {n_edges:,}\n"
        f"  Properties coverage: {coverage:.1f}%\n"
        f"  Time     : {time.time()-t0:.0f}s"
    )


if __name__ == "__main__":
    run()
