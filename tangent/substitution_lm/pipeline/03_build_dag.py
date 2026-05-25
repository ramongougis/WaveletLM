"""
Step 3 — Build the co-occurrence DAG.

For each sentence in the parse cache, identifies which node IDs are present
(matching the longest n-gram span at each position), then emits directed
co-occurrence edges between every pair of nodes within CO_OCC_WINDOW tokens.

The DAG is a dict[int, dict[int, int]]:  dag[src_id][dst_id] = count.
Edges where dst comes after src in the sentence are the primary signal;
reverse edges are also counted (for Mechanism B aggregated vote).

Writes: DAG_PATH — the adjacency dict (pickle).

VRAM: none.  RAM: O(unique_edges × 24 bytes) ≈ 1–3 GB depending on corpus.
"""

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C


def _match_nodes(lemmas: list[str], ngram2id: dict) -> list[tuple[int, int]]:
    """
    Return [(position, node_id), ...] by greedy longest-match left-to-right.
    Each token position is matched to at most one node (the longest n-gram
    starting there that exists in the node table).
    """
    n = len(lemmas)
    matched = []
    i = 0
    while i < n:
        best_id, best_len = None, 0
        for length in range(min(C.MAX_NGRAM, n - i), 0, -1):
            phrase = " ".join(lemmas[i: i + length])
            nid = ngram2id.get(phrase)
            if nid is not None:
                best_id, best_len = nid, length
                break
        if best_id is not None:
            matched.append((i, best_id))
            i += best_len
        else:
            i += 1
    return matched


def run() -> None:
    if C.DAG_PATH.exists():
        print(f"[03_build_dag] DAG already exists: {C.DAG_PATH}  (delete to rebuild)")
        return

    for dep in (C.PARSE_CACHE_TRAIN, C.NODE_TABLE_PATH):
        if not dep.exists():
            print(f"[03_build_dag] Missing dependency: {dep}")
            sys.exit(1)

    print("[03_build_dag] Loading node table …")
    with open(C.NODE_TABLE_PATH, "rb") as f:
        table = pickle.load(f)
    ngram2id = table["ngram2id"]

    dag: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    print("[03_build_dag] Building DAG …")
    t0 = time.time()
    docs_done = edges_added = 0

    with open(C.PARSE_CACHE_TRAIN, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            for sent in doc["sentences"]:
                lemmas = [
                    t["lemma"] for t in sent["tokens"]
                    if t["pos"] not in ("PUNCT", "SPACE", "SYM", "NUM")
                ]
                matched = _match_nodes(lemmas, ngram2id)

                # Emit edges for all pairs within CO_OCC_WINDOW positions
                for i, (pos_i, nid_i) in enumerate(matched):
                    for j in range(i + 1, len(matched)):
                        pos_j, nid_j = matched[j]
                        if pos_j - pos_i > C.CO_OCC_WINDOW:
                            break
                        dag[nid_i][nid_j] += 1  # forward (i precedes j)
                        dag[nid_j][nid_i] += 1  # backward (for aggregated vote)
                        edges_added += 1

            docs_done += 1
            if docs_done % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {docs_done:,} docs  {edges_added:,} edge-increments  ({elapsed:.0f}s)", flush=True)

    # Convert nested defaultdicts to plain dicts for pickling
    dag = {src: dict(dsts) for src, dsts in dag.items()}

    with open(C.DAG_PATH, "wb") as f:
        pickle.dump(dag, f)

    elapsed = time.time() - t0
    print(
        f"[03_build_dag] Done — {len(dag):,} source nodes  "
        f"{edges_added:,} edge-increments  →  {C.DAG_PATH}  ({elapsed:.0f}s)"
    )


if __name__ == "__main__":
    run()
