"""cross_eval.py — evaluate any WaveletLM checkpoint on any dataset's test split.

The twin-experiment endpoint (F1-on-WT103 vs D3-on-WT103) and, generally, the
cross-domain evaluation affordance benchmark_only lacks. IMPORTS train.py's
exact benchmark functions (evaluate_full_validation, evaluate_sliding_window)
— no reimplementation, so evaluation semantics are identical to every number
in the README by construction. Validation gate: running the D3 checkpoint on
wikitext-103 must reproduce its recorded 0.9797 sliding BPB / 21.34 PPL
(pod benchmark ran fp16 CUDA; expect exact-to-noise agreement on fp16 CUDA,
~1e-3-class drift on fp32 CPU).

Tokenizer guard: the eval dataset's tokenizer must equal the checkpoint's
(GPT-2 BPE <-> GPT-2 BPE fine; PG-19's SentencePiece cross-evals are invalid
and refused).

Usage:
  python tools/cross_eval.py --run_dir logs/pile_2026-07-17_20-02-08 \
      --dataset wikitext-103 --device cuda
"""
import argparse
import json
import math
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from model import WaveletLM, get_tokenizer, resolve_tokenizer_name  # noqa: E402
from train import (load_and_encode_dataset, evaluate_full_validation,  # noqa: E402
                   evaluate_sliding_window, CANONICAL_TEST_WORDS)


class _PrintLogger:
    def log(self, msg):
        print(msg, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Run folder providing best_model.pt + config.json")
    p.add_argument("--dataset", required=True,
                   help="Dataset whose TEST split to evaluate on")
    p.add_argument("--dataset_max_tokens", type=int, default=None,
                   help="pile only: subset cap selecting the cache")
    p.add_argument("--pile_holdout_stride", type=int, default=None,
                   help="pile only: holdout stride selecting the cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable fp16 autocast on CUDA (pod benchmarks used fp16)")
    args = p.parse_args()

    log = _PrintLogger()
    with open(os.path.join(args.run_dir, "config.json")) as f:
        run_cfg = json.load(f)

    eval_cfg = dict(run_cfg)
    eval_cfg["dataset"] = args.dataset
    eval_cfg["tokenizer"] = "auto"
    if args.dataset_max_tokens is not None:
        eval_cfg["dataset_max_tokens"] = args.dataset_max_tokens
    if args.pile_holdout_stride is not None:
        eval_cfg["pile_holdout_stride"] = args.pile_holdout_stride
    if args.dataset == "pile" and args.dataset_max_tokens is None:
        raise SystemExit("[cross_eval] Refusing: --dataset pile without "
                         "--dataset_max_tokens (would stream the 5B default).")

    tok_run = resolve_tokenizer_name(run_cfg)
    tok_eval = resolve_tokenizer_name(eval_cfg)
    if tok_run != tok_eval:
        raise SystemExit(f"[cross_eval] Tokenizer mismatch: checkpoint={tok_run!r} "
                         f"vs eval dataset={tok_eval!r} — cross-eval invalid.")

    enc = get_tokenizer(run_cfg)
    device = args.device
    log.log(f"[cross_eval] checkpoint: {args.run_dir} (trained on "
            f"{run_cfg.get('dataset')}) -> eval on {args.dataset} test split; "
            f"device={device}")

    model = WaveletLM(enc.vocab_size, run_cfg, device=device).to(device)
    ckpt = torch.load(os.path.join(args.run_dir, "best_model.pt"),
                      map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    state = {(k[10:] if k.startswith("_orig_mod.") else k): v
             for k, v in ckpt.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    # Pile fast-path: cross-eval only needs the TEST split, but the training
    # loader returns all three splits as int64 (the 4.79B-token train tensor
    # alone is 38 GB — laptop-fatal). If the cap-namespaced cache exists, load
    # it memory-mapped and materialize only the tiny test split.
    pile_cache = None
    if eval_cfg["dataset"] == "pile":
        cap = int(eval_cfg["dataset_max_tokens"])
        stride = int(eval_cfg.get("pile_holdout_stride", 6000))
        pile_cache = os.path.join(
            ".cache", f"pile-{cap}tok-s{stride}_{tok_eval}.pt")
    if pile_cache and os.path.exists(pile_cache):
        log.log(f"[cross_eval] mmap-loading test split only from {pile_cache}")
        cached = torch.load(pile_cache, map_location="cpu",
                            weights_only=False, mmap=True)
        test_data = cached["test"].long()
        bytes_per_token = cached["test_bytes"] / max(1, len(test_data))
        log.log(f"[cross_eval] pile test: {len(test_data):,} tokens, "
                f"{bytes_per_token:.4f} bytes/token")
        del cached
    else:
        _, _, test_data, _, bytes_per_token = load_and_encode_dataset(eval_cfg, log)

    use_amp = (device == "cuda") and not args.no_amp
    amp_dtype = torch.float16
    log.log(f"[cross_eval] AMP: {use_amp} (fp16) — pod benchmarks used fp16 CUDA")

    res_full = evaluate_full_validation(
        model, test_data, run_cfg, log, device, use_amp, amp_dtype)
    res_slide = evaluate_sliding_window(
        model, test_data, run_cfg, log, device, use_amp, amp_dtype)

    ln2 = math.log(2)
    for name, res in (("Non-overlapping", res_full), ("Sliding Window", res_slide)):
        if res is None:
            log.log(f"[cross_eval] {name}: no result (test too small?)")
            continue
        bpb = res["avg_loss"] / (ln2 * bytes_per_token)
        log.log(f"\n[CROSS-EVAL - {name}] "
                f"{os.path.basename(os.path.normpath(args.run_dir))} on {args.dataset}")
        log.log(f"  Perplexity: {res['perplexity']:.4f}")
        log.log(f"  BPT: {res['bits_per_token']:.4f}")
        log.log(f"  BPB: {bpb:.4f}")
        log.log(f"  Avg Loss: {res['avg_loss']:.4f}")
    log.log(f"  ({bytes_per_token:.4f} bytes/token on this test split)")

    canon = CANONICAL_TEST_WORDS.get(args.dataset)
    if canon:
        ratio = len(test_data) / canon
        log.log(f"\n[CROSS-EVAL - Word-level normalization] canonical words: "
                f"{canon:,}, tokens/word: {ratio:.4f}")
        for name, res in (("Non-overlapping", res_full), ("Sliding", res_slide)):
            if res:
                log.log(f"  {name} word-level PPL: "
                        f"{math.exp(res['avg_loss'] * ratio):.2f}")


if __name__ == "__main__":
    main()
