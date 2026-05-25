"""
Evaluation — PPL, BPB, and Selectional Violation Detection Rate (SVDR).

PPL / BPB:
    Iterates the WT103 test split token by token. At each position t, builds
    a context window from the preceding CONTEXT_SIZE tokens, calls each
    mechanism's predict_logprobs(), looks up log2 P(w_t | context), and
    accumulates the total log-likelihood. Reports:
        PPL  = exp(mean CE in nats)
        BPB  = mean CE in bits / mean bytes per word

    Unseen tokens (not reachable from any context edge) receive UNK_LOGPROB.

SVDR:
    Loads a hand-crafted sentence pair file (normal vs. anomalous), scores
    each sentence's mean per-token log-prob, and classifies anomalous sentences
    as those with lower score than their normal pair.  Reports P / R / F1.

Usage:
    python eval/ppl_bpb.py [--mechanism a|b|c|all] [--workers N]
"""

import argparse
import math
import multiprocessing
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C
from predict import load_model
import predict.mechanism_a as mech_a
import predict.mechanism_b as mech_b
import predict.mechanism_c as mech_c

MECHANISMS = {"a": mech_a, "b": mech_b, "c": mech_c}

# Globals set before forking; workers inherit via CoW on Linux/macOS
_SHARED_MODEL: dict = {}
_SHARED_WORDS: list = []  # list of (wid_or_None, byte_len)
_MECH_MAP = {"A": mech_a, "B": mech_b, "C": mech_c}


def _iter_test_words():
    """Yield lowercased words from the WT103 test split, skipping headings."""
    from datasets import load_dataset
    ds = load_dataset(C.HF_DATASET, C.HF_CONFIG, split="test")
    for example in ds:
        line = example["text"].strip()
        if not line or line.startswith("="):
            continue
        for word in line.split():
            if word and not all(c in "=-<>" for c in word):
                yield word.lower()


def _preprocess_words(model: dict) -> list:
    """Pre-tokenize the full test set into (wid_or_None, byte_len) pairs."""
    ngram2id = model["ngram2id"]
    result = []
    for word in tqdm(_iter_test_words(), desc="  Pre-tokenizing", unit="tok",
                     dynamic_ncols=True, smoothing=0.05):
        result.append((ngram2id.get(word), len(word.encode("utf-8"))))
    return result


def _chunk_worker(args):
    """
    Process one slice of _SHARED_WORDS sequentially.
    args = (start, end, ctx_prefix, mech_label)
    Returns (total_log2, total_bytes, n_tokens, n_unk).
    """
    start, end, ctx_prefix, mech_label = args
    mech_module = _MECH_MAP[mech_label]
    model   = _SHARED_MODEL
    uni_lp  = model["unigram_logprob"]

    context_ids = list(ctx_prefix)
    total_log2  = 0.0
    total_bytes = 0
    n_tokens    = 0
    n_unk       = 0

    for wid, byte_len in _SHARED_WORDS[start:end]:
        lps = (mech_module.predict_logprobs(context_ids, model)
               if context_ids else {nid: lp for nid, lp in uni_lp.items()})

        if wid is not None and wid in lps:
            log2p = lps[wid]
        elif wid is not None and wid in uni_lp:
            log2p = uni_lp[wid]
        else:
            log2p = C.UNK_LOGPROB
            n_unk += 1

        total_log2  += log2p
        total_bytes += byte_len
        n_tokens    += 1

        if wid is not None:
            context_ids.append(wid)
        if len(context_ids) > C.CONTEXT_SIZE:
            context_ids.pop(0)

    return total_log2, total_bytes, n_tokens, n_unk


def eval_ppl_bpb(mech_module, model: dict, label: str, n_workers: int = 1) -> dict:
    """Run PPL/BPB evaluation; dispatches to sequential or parallel path."""
    if n_workers > 1:
        if sys.platform == "win32":
            print(f"  [{label}] Parallel eval requires fork (Linux/macOS) — running sequentially")
            n_workers = 1
        else:
            cpu_n = multiprocessing.cpu_count()
            if n_workers > cpu_n // 2:
                print(f"  [{label}] Warning: {n_workers} workers > {cpu_n // 2} "
                      f"(half of {cpu_n} CPUs) — may compete with WaveletLM training")

    return (_eval_sequential(mech_module, model, label)
            if n_workers <= 1
            else _eval_parallel(model, label, n_workers))


def _eval_sequential(mech_module, model: dict, label: str) -> dict:
    ngram2id = model["ngram2id"]
    uni_lp   = model["unigram_logprob"]

    context_ids: list[int] = []
    total_log2 = 0.0
    total_bytes = 0
    n_tokens = 0
    n_unk = 0

    bar = tqdm(
        _iter_test_words(),
        desc=f"  Mech {label}",
        unit="tok",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    for word in bar:
        wid = ngram2id.get(word)

        if context_ids:
            lps = mech_module.predict_logprobs(context_ids, model)
        else:
            lps = {nid: lp for nid, lp in uni_lp.items()}

        if wid is not None and wid in lps:
            log2p = lps[wid]
        elif wid is not None and wid in uni_lp:
            log2p = uni_lp[wid]
        else:
            log2p = C.UNK_LOGPROB
            n_unk += 1

        total_log2 += log2p
        total_bytes += len(word.encode("utf-8"))
        n_tokens += 1

        if wid is not None:
            context_ids.append(wid)
        if len(context_ids) > C.CONTEXT_SIZE:
            context_ids.pop(0)

        if n_tokens % 500 == 0:
            bar.set_postfix(
                BPB=f"{-total_log2 / max(total_bytes, 1):.4f}",
                UNK=f"{100 * n_unk / n_tokens:.1f}%",
            )

    mean_ce_bits = -total_log2 / n_tokens
    mean_ce_nats = mean_ce_bits * math.log(2)
    bpb          = -total_log2 / total_bytes
    ppl          = math.exp(mean_ce_nats)

    return {
        "mechanism": label,
        "ppl":       ppl,
        "bpb":       bpb,
        "n_tokens":  n_tokens,
        "n_unk":     n_unk,
        "unk_pct":   100.0 * n_unk / n_tokens,
    }


def _eval_parallel(model: dict, label: str, n_workers: int) -> dict:
    """Fork-based parallel evaluation. Pre-tokenizes once and reuses across mechanisms."""
    global _SHARED_MODEL, _SHARED_WORDS

    _SHARED_MODEL = model
    if not _SHARED_WORDS:
        _SHARED_WORDS = _preprocess_words(model)

    n = len(_SHARED_WORDS)
    chunk_size = math.ceil(n / n_workers)

    chunks = []
    for i in range(n_workers):
        start = i * chunk_size
        end   = min(start + chunk_size, n)
        if start >= n:
            break
        # Context prefix: last CONTEXT_SIZE IDs preceding this chunk
        prior = _SHARED_WORDS[max(0, start - C.CONTEXT_SIZE * 5): start]
        ctx_prefix = [wid for wid, _ in prior if wid is not None][-C.CONTEXT_SIZE:]
        chunks.append((start, end, ctx_prefix, label))

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_chunk_worker, chunks),
            total=len(chunks),
            desc=f"  Mech {label}",
            unit="chunk",
            dynamic_ncols=True,
        ))

    total_log2  = sum(r[0] for r in results)
    total_bytes = sum(r[1] for r in results)
    n_tokens    = sum(r[2] for r in results)
    n_unk       = sum(r[3] for r in results)

    mean_ce_bits = -total_log2 / n_tokens
    mean_ce_nats = mean_ce_bits * math.log(2)
    bpb          = -total_log2 / total_bytes
    ppl          = math.exp(mean_ce_nats)

    return {
        "mechanism": label,
        "ppl":       ppl,
        "bpb":       bpb,
        "n_tokens":  n_tokens,
        "n_unk":     n_unk,
        "unk_pct":   100.0 * n_unk / n_tokens,
    }


def _sentence_score(sentence: str, mech_module, model: dict) -> float:
    """Mean log2 per-token probability for a sentence (used by SVDR)."""
    words = sentence.lower().split()
    ngram2id = model["ngram2id"]
    uni_lp   = model["unigram_logprob"]
    ctx: list[int] = []
    total = 0.0
    for word in words:
        wid = ngram2id.get(word)
        lps = mech_module.predict_logprobs(ctx, model) if ctx else uni_lp
        if wid is not None and wid in lps:
            total += lps[wid]
        else:
            total += C.UNK_LOGPROB
        if wid is not None:
            ctx.append(wid)
        if len(ctx) > C.CONTEXT_SIZE:
            ctx.pop(0)
    return total / max(len(words), 1)


def eval_svdr(mech_module, model: dict, label: str, pairs_file: Path) -> dict | None:
    """
    SVDR evaluation from a tab-separated file: normal_sentence\tanomalous_sentence
    Returns precision, recall, F1, or None if file not found.
    """
    if not pairs_file.exists():
        print(f"  [{label}] SVDR pairs file not found: {pairs_file} — skipping")
        return None

    tp = fp = fn = tn = 0
    with open(pairs_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            normal, anomalous = parts
            s_norm = _sentence_score(normal,    mech_module, model)
            s_anom = _sentence_score(anomalous, mech_module, model)
            if s_norm > s_anom:
                tp += 1
            else:
                fn += 1

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"mechanism": label, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", default="all", choices=["a", "b", "c", "all"])
    parser.add_argument("--svdr_pairs", default=str(C.ROOT / "eval" / "svdr_pairs.tsv"))
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for PPL/BPB (>1 requires Linux/macOS fork). "
             "Keep ≤ half of CPU count to leave cores for WaveletLM training.",
    )
    args = parser.parse_args()

    model = load_model()
    mechs = MECHANISMS if args.mechanism == "all" else {args.mechanism: MECHANISMS[args.mechanism]}
    pairs_file = Path(args.svdr_pairs)

    print(f"\n{'='*60}")
    print("  PPL / BPB / SVDR Evaluation")
    print(f"  Test split: {C.HF_DATASET} / {C.HF_CONFIG} (test)")
    if args.workers > 1:
        print(f"  Workers   : {args.workers}")
    print(f"{'='*60}\n")

    for key, mod in mechs.items():
        print(f"── Mechanism {key.upper()} ──────────────────────────────────")
        res = eval_ppl_bpb(mod, model, key.upper(), n_workers=args.workers)
        print(
            f"  PPL   : {res['ppl']:.2f}\n"
            f"  BPB   : {res['bpb']:.4f}\n"
            f"  Tokens: {res['n_tokens']:,}  UNK: {res['unk_pct']:.1f}%\n"
        )
        svdr = eval_svdr(mod, model, key.upper(), pairs_file)
        if svdr:
            print(
                f"  SVDR  P={svdr['precision']:.3f}  R={svdr['recall']:.3f}  F1={svdr['f1']:.3f}\n"
            )


if __name__ == "__main__":
    main()
