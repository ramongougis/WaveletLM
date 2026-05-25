"""
Step 4 — Build property tables from ConceptNet.

Reads conceptnet-assertions-5.7.0.csv (English-only relations), filters to
relations in CONCEPTNET_RELS, and builds a per-node property dict:

    properties[node_id] = {
        "HasProperty":    ["hard", "fast", ...],
        "NotCapableOf":   ["swim", "fly", ...],
        "CapableOf":      ["compute", ...],
        "IsA":            ["machine", "device", ...],
        ...
    }

Only nodes present in the node table receive entries; unknown nodes get an
empty dict (handled as "no contradiction found" by the consistency checker).

ConceptNet download (1.5 GB compressed):
    wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
    gunzip conceptnet-assertions-5.7.0.csv.gz
Place the resulting .csv in the substitution_lm/ directory (gitignored).

VRAM: none.  RAM: < 300 MB for the filtered relations.
"""

import csv
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C


def _rel_name(uri: str) -> str:
    """Extract relation name from a ConceptNet URI, e.g. /r/HasProperty → HasProperty."""
    return uri.split("/")[-1]


def _concept_word(uri: str) -> str | None:
    """
    Extract the word from an English concept URI.
    /c/en/computer/n → 'computer'
    /c/fr/chat      → None (not English)
    """
    parts = uri.split("/")
    if len(parts) < 4 or parts[2] != "en":
        return None
    return parts[3].replace("_", " ").lower()


def run() -> None:
    if C.PROPERTIES_PATH.exists():
        print(f"[04_property] Property table already exists: {C.PROPERTIES_PATH}  (delete to rebuild)")
        return

    if not C.NODE_TABLE_PATH.exists():
        print("[04_property] Node table missing — run 02_extract_ngrams.py first.")
        sys.exit(1)

    if not C.CONCEPTNET_CSV.exists():
        print(
            f"[04_property] ConceptNet CSV not found: {C.CONCEPTNET_CSV}\n"
            "Download:\n"
            "  wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
            "conceptnet-assertions-5.7.0.csv.gz\n"
            "  gunzip conceptnet-assertions-5.7.0.csv.gz\n"
            f"  mv conceptnet-assertions-5.7.0.csv {C.ROOT}/"
        )
        print("[04_property] Skipping — no ConceptNet file. Properties will be empty.")
        # Write empty property table so downstream steps can proceed
        with open(C.PROPERTIES_PATH, "wb") as f:
            pickle.dump({}, f)
        return

    print("[04_property] Loading node table …")
    with open(C.NODE_TABLE_PATH, "rb") as f:
        table = pickle.load(f)
    ngram2id = table["ngram2id"]

    # Raw mapping: concept_word → {relation: [target_words]}
    word_props: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    print(f"[04_property] Reading ConceptNet from {C.CONCEPTNET_CSV} …")
    t0 = time.time()
    rows_read = rows_kept = 0

    with open(C.CONCEPTNET_CSV, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            # Columns: uri, relation_uri, subject_uri, object_uri, metadata_json
            rel   = _rel_name(row[1])
            if rel not in C.CONCEPTNET_RELS:
                rows_read += 1
                continue
            subj = _concept_word(row[2])
            obj  = _concept_word(row[3])
            rows_read += 1
            if subj and obj:
                word_props[subj][rel].append(obj)
                rows_kept += 1

            if rows_read % 2_000_000 == 0:
                print(f"  {rows_read:,} rows read  {rows_kept:,} kept  ({time.time()-t0:.0f}s)", flush=True)

    print(f"  Done: {rows_read:,} rows  {rows_kept:,} kept  concepts: {len(word_props):,}")

    # Map to node IDs
    properties: dict[int, dict[str, list]] = {}
    matched = 0
    for word, props in word_props.items():
        nid = ngram2id.get(word)
        if nid is not None:
            properties[nid] = dict(props)
            matched += 1

    with open(C.PROPERTIES_PATH, "wb") as f:
        pickle.dump(properties, f)

    elapsed = time.time() - t0
    coverage = 100.0 * matched / max(len(ngram2id), 1)
    print(
        f"[04_property] {matched:,}/{len(ngram2id):,} nodes with ConceptNet data "
        f"({coverage:.1f}%)  →  {C.PROPERTIES_PATH}  ({elapsed:.0f}s)"
    )


if __name__ == "__main__":
    run()
