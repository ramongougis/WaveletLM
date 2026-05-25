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
_NLP = None  # lazily-loaded spaCy pipeline


def _get_nlp():
    """Load spaCy with the same model the pipeline trained on, parser/ner off."""
    global _NLP
    if _NLP is None:
        import spacy
        spacy.require_cpu()
        _NLP = spacy.load(C.SPACY_MODEL, disable=["parser", "ner"])
    return _NLP


def _lemmatise(text: str) -> list[tuple[str, int]]:
    """Lemmatise + POS-filter text, matching 02_extract_ngrams.

    Returns list of (lemma_lowercased, surface_byte_len) — the surface byte
    length is what BPB normalises against, independent of any inflection.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    out = []
    for t in doc:
        if t.pos_ in ("PUNCT", "SPACE", "SYM", "NUM"):
            continue
        out.append((t.lemma_.lower(), len(t.text.encode("utf-8"))))
    return out


def _preprocess_words(model: dict) -> list:
    """
    Lemmatise the WT103 test split with spaCy (matching the training pipeline),
    look up each lemma in the node table, and return:
        list of (wid_or_None, surface_byte_len)
    """
    from datasets import load_dataset

    ngram2id = model["ngram2id"]
    nlp = _get_nlp()

    ds = load_dataset(C.HF_DATASET, C.HF_CONFIG, split="test")
    texts = [ex["text"] for ex in ds if ex["text"].strip()]

    result = []
    bar = tqdm(desc="  Lemmatising", unit="tok",
               dynamic_ncols=True, smoothing=0.05)
    for doc in nlp.pipe(texts, batch_size=128):
        for t in doc:
            if t.pos_ in ("PUNCT", "SPACE", "SYM", "NUM"):
                continue
            lemma = t.lemma_.lower()
            byte_len = len(t.text.encode("utf-8"))
            result.append((ngram2id.get(lemma), byte_len))
            bar.update(1)
    bar.close()
    return result


def _chunk_worker(args):
    """
    Process one slice of _SHARED_WORDS sequentially with its own tqdm bar.
    args = (start, end, ctx_prefix, mech_label, chunk_idx)
    Returns (total_log2, total_bytes, n_tokens, n_unk).
    """
    start, end, ctx_prefix, mech_label, chunk_idx = args
    mech_module = _MECH_MAP[mech_label]
    model   = _SHARED_MODEL
    uni_lp  = model["unigram_logprob"]

    context_ids = list(ctx_prefix)
    total_log2  = 0.0
    total_bytes = 0
    n_tokens    = 0
    n_unk       = 0

    bar = tqdm(
        total=end - start,
        desc=f"  Chunk {chunk_idx + 1}",
        position=chunk_idx,
        unit="tok",
        leave=True,
        dynamic_ncols=True,
        smoothing=0.05,
    )
    pending = 0

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
        pending     += 1

        if wid is not None:
            context_ids.append(wid)
        if len(context_ids) > C.CONTEXT_SIZE:
            context_ids.pop(0)

        if pending >= 500:
            bar.update(500)
            pending -= 500

    if pending:
        bar.update(pending)
    bar.close()

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
    global _SHARED_WORDS
    if not _SHARED_WORDS:
        _SHARED_WORDS = _preprocess_words(model)

    uni_lp = model["unigram_logprob"]

    context_ids: list[int] = []
    total_log2 = 0.0
    total_bytes = 0
    n_tokens = 0
    n_unk = 0

    bar = tqdm(
        _SHARED_WORDS,
        desc=f"  Mech {label}",
        unit="tok",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    for wid, byte_len in bar:
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
        total_bytes += byte_len
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
        chunks.append((start, end, ctx_prefix, label, i))

    ctx = multiprocessing.get_context("fork")
    lock = ctx.RLock()
    tqdm.set_lock(lock)

    with ctx.Pool(n_workers, initializer=tqdm.set_lock, initargs=(lock,)) as pool:
        async_results = [pool.apply_async(_chunk_worker, (chunk,)) for chunk in chunks]
        for r in async_results:
            r.wait()
        results = [r.get() for r in async_results]

    sys.stderr.write("\n" * len(chunks))
    sys.stderr.flush()

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
    """Mean log2 per-token probability for a sentence (used by SVDR).

    Lemmatises through spaCy to match the lemma-keyed node table.
    """
    tokens = _lemmatise(sentence)
    if not tokens:
        return 0.0

    ngram2id = model["ngram2id"]
    uni_lp   = model["unigram_logprob"]
    ctx: list[int] = []
    total = 0.0
    for lemma, _ in tokens:
        wid = ngram2id.get(lemma)
        lps = mech_module.predict_logprobs(ctx, model) if ctx else uni_lp
        if wid is not None and wid in lps:
            total += lps[wid]
        else:
            total += C.UNK_LOGPROB
        if wid is not None:
            ctx.append(wid)
        if len(ctx) > C.CONTEXT_SIZE:
            ctx.pop(0)
    return total / max(len(tokens), 1)


def eval_svdr(mech_module, model: dict, label: str, pairs_file: Path) -> dict | None:
    """
    SVDR evaluation from a tab-separated file: normal_sentence\tanomalous_sentence

    Paired evaluation: every pair contributes either a true positive (model
    scores the normal sentence higher than the anomalous) or a false negative
    (model fails to). There is no false-positive case by construction, so
    **precision is always 1.0** — the meaningful metric is recall, which equals
    pair-classification accuracy. F1 = 2R/(1+R) is then a deterministic
    function of recall.

    Returns precision, recall, F1, or None if file not found.
    """
    if not pairs_file.exists():
        print(f"  [{label}] SVDR pairs file not found: {pairs_file} — skipping")
        return None

    tp = fn = 0
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

    precision = 1.0  # by construction of paired evaluation (no FP case)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "mechanism": label,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  recall,  # alias — same number, clearer interpretation
    }


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
                f"  SVDR  accuracy={svdr['accuracy']:.3f}  "
                f"(P={svdr['precision']:.3f} R={svdr['recall']:.3f} F1={svdr['f1']:.3f})\n"
            )


if __name__ == "__main__":
    main()
