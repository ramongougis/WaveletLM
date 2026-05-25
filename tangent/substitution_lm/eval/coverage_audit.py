"""
Corpus n-gram coverage audit — implements the design's "iterative subtraction"
diagnostic. Scans the training parse cache and counts which n-grams (bigrams
and trigrams by default) appear as nodes in the table (collocation or NP
n-gram) versus which slip through unaccounted by the existing extraction
methods.

For each n-gram length, reports:
  - Total / accounted / unaccounted instance counts
  - Top-100 tokens by frequency of appearance in unaccounted n-grams
    (tokens most often involved in patterns the current pipeline doesn't
    extract as units)

Use this to decide what specialised handling deserves to be added next:
function-word patterns slipping through suggest skip-N-gram routing;
content-word collocations slipping through suggest tightening PMI/freq
thresholds in step 2; etc.

Usage:
    python eval/coverage_audit.py [--max_n 3] [--top 100]
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C


def _iter_lemma_sequences(cache_path: Path):
    """Yield per-sentence lemma lists from the parse cache, matching
    02_extract_ngrams' POS filter (skip PUNCT/SPACE/SYM/NUM)."""
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            for sent in doc["sentences"]:
                lemmas = [
                    t["lemma"]
                    for t in sent["tokens"]
                    if t["pos"] not in ("PUNCT", "SPACE", "SYM", "NUM")
                ]
                if lemmas:
                    yield lemmas


def _audit(ngram2id: dict, max_n: int):
    """Single pass over the parse cache; counts per n-gram length."""
    counts = {n: [0, 0] for n in range(2, max_n + 1)}  # [total, accounted]
    per_token_unacc = {n: Counter() for n in range(2, max_n + 1)}

    bar = tqdm(
        _iter_lemma_sequences(C.PARSE_CACHE_TRAIN),
        desc="  Scanning sentences",
        unit="sent",
        dynamic_ncols=True,
        smoothing=0.05,
    )

    for lemmas in bar:
        L = len(lemmas)
        for n in range(2, max_n + 1):
            if L < n:
                continue
            for i in range(L - n + 1):
                ngram = " ".join(lemmas[i:i + n])
                counts[n][0] += 1
                if ngram in ngram2id:
                    counts[n][1] += 1
                else:
                    for tok in lemmas[i:i + n]:
                        per_token_unacc[n][tok] += 1

    return counts, per_token_unacc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_n", type=int, default=3,
                        help="Maximum n-gram length to audit (default 3)")
    parser.add_argument("--top", type=int, default=100,
                        help="Top-N tokens per length to report (default 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Corpus N-Gram Coverage Audit (iterative subtraction)")
    print("=" * 60)

    t0 = time.time()
    print("\n[audit] Loading node table …")
    with open(C.NODE_TABLE_PATH, "rb") as f:
        table = pickle.load(f)
    ngram2id = table["ngram2id"]
    print(f"  Nodes in table: {len(ngram2id):,}")

    print(f"\n[audit] Scanning {C.PARSE_CACHE_TRAIN.name} for n-grams up to length {args.max_n} …")
    counts, per_token = _audit(ngram2id, args.max_n)

    print(f"\n{'=' * 60}")
    print("  Coverage Summary")
    print(f"{'=' * 60}")
    for n in range(2, args.max_n + 1):
        total, accounted = counts[n]
        unaccounted = total - accounted
        pct_acc = 100.0 * accounted / max(total, 1)
        pct_unacc = 100.0 - pct_acc
        print(f"\n  {n}-grams:")
        print(f"    Total       : {total:>14,}")
        print(f"    Accounted   : {accounted:>14,}  ({pct_acc:6.2f}%)")
        print(f"    Unaccounted : {unaccounted:>14,}  ({pct_unacc:6.2f}%)")

    for n in range(2, args.max_n + 1):
        print(f"\n{'=' * 60}")
        print(f"  Top-{args.top} tokens by appearance in unaccounted {n}-grams")
        print(f"{'=' * 60}")
        print(f"  {'rank':>4}  {'token_id':>10}  {'lemma':<28s}  {'count':>14}  {'% of unacc':>10}")
        print("  " + "-" * 74)
        total_unacc_tokens = sum(per_token[n].values())
        for rank, (tok, cnt) in enumerate(per_token[n].most_common(args.top), start=1):
            nid = ngram2id.get(tok, -1)
            pct = 100.0 * cnt / max(total_unacc_tokens, 1)
            print(f"  {rank:>4}  {nid:>10}  {tok:<28s}  {cnt:>14,}  {pct:>9.2f}%")

    print(f"\n[audit] Done — {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
