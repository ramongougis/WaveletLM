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
    python eval/ppl_bpb.py [--mechanism a|b|c|all]
"""

import argparse
import math
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C
from predict import load_model, tokenize
import predict.mechanism_a as mech_a
import predict.mechanism_b as mech_b
import predict.mechanism_c as mech_c

MECHANISMS = {"a": mech_a, "b": mech_b, "c": mech_c}


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


def eval_ppl_bpb(mech_module, model: dict, label: str) -> dict:
    """Run PPL/BPB evaluation for one mechanism."""
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

        # Get distribution over next token
        if context_ids:
            lps = mech_module.predict_logprobs(context_ids, model)
        else:
            lps = {nid: lp for nid, lp in uni_lp.items()}

        if wid is not None and wid in lps:
            log2p = lps[wid]
        elif wid is not None and wid in uni_lp:
            log2p = uni_lp[wid]  # backoff to unigram
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

    mean_ce_bits = -total_log2 / n_tokens          # bits per token
    mean_ce_nats = mean_ce_bits * math.log(2)      # nats per token
    bpb          = -total_log2 / total_bytes       # bits per byte
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
            if s_norm > s_anom:   # correctly detects anomaly
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
    args = parser.parse_args()

    model = load_model()
    mechs = MECHANISMS if args.mechanism == "all" else {args.mechanism: MECHANISMS[args.mechanism]}
    pairs_file = Path(args.svdr_pairs)

    print(f"\n{'='*60}")
    print("  PPL / BPB / SVDR Evaluation")
    print(f"  Test split: {C.HF_DATASET} / {C.HF_CONFIG} (test)")
    print(f"{'='*60}\n")

    for key, mod in mechs.items():
        print(f"── Mechanism {key.upper()} ──────────────────────────────────")
        res = eval_ppl_bpb(mod, model, key.upper())
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
