"""context_usage.py — how much does IN-CONTEXT information help?

Finding 7 (induction_probe.py) showed WaveletLM does not copy arbitrary
bound symbols from context. That is a narrow claim about RETRIEVAL memory
and says nothing about whether the model uses context at all. This probe
measures the broader capability directly, on natural text, with no synthetic
insertions: mean next-token loss as a function of position within the
window.

A model that ignores context has a flat curve. A model that uses context has
a decreasing curve — later positions are cheaper because more history is
available. Olsson et al. call the early-vs-late difference the in-context
learning score; it is the standard non-synthetic measure of context use.

Reported: mean loss per position bucket, and the ICL score
(mean loss over an early bucket minus mean loss over a late bucket, in nats;
larger = context is worth more).

Usage:
  python tools/interpretability/context_usage.py \
      --run_dir logs/wikitext-103_2026-07-15_10-53-46 --sequences 128
  python tools/interpretability/context_usage.py \
      --run_dir logs/wikitext-103_2026-07-15_10-53-46 --hf_model gpt2
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from model import WaveletLM, get_tokenizer  # noqa: E402
from train import load_and_encode_dataset    # noqa: E402


class _L:
    def log(self, m):
        print(m, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--hf_model", default=None, help="transformer control, e.g. 'gpt2'")
    p.add_argument("--sequences", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--filler_dataset", default="wikitext-103")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    enc = get_tokenizer(cfg)
    V, T = enc.vocab_size, int(cfg["block_size"])

    if args.hf_model:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.hf_model).to(device).eval()
        label = f"{args.hf_model} (HF control)"

        def _fwd(xb):
            return model(xb).logits
    else:
        model = WaveletLM(V, cfg, device=device).to(device)
        ck = torch.load(os.path.join(args.run_dir, "best_model.pt"), map_location=device)
        if isinstance(ck, dict) and "model_state" in ck:
            ck = ck["model_state"]
        model.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k): v
                               for k, v in ck.items()}, strict=True)
        model.eval()
        base = getattr(model, "_orig_mod", model)
        if hasattr(base, "reset_semantic_state"):
            base.reset_semantic_state()
        base.decompose_bypass_cross_window = False   # each window stands alone
        label = args.run_dir

        def _fwd(xb):
            return model(xb, targets=None)[0]

    data_cfg = dict(cfg)
    data_cfg["dataset"] = args.filler_dataset
    data_cfg["tokenizer"] = "auto"
    _, val_data, test_data, _, _ = load_and_encode_dataset(data_cfg, _L())
    src = test_data if len(test_data) > len(val_data) else val_data

    starts = torch.randint(0, len(src) - T - 2, (args.sequences,))
    X = torch.stack([src[s:s + T].clone().to(torch.long) for s in starts])
    Y = torch.stack([src[s + 1:s + T + 1].clone().to(torch.long) for s in starts])

    pos_loss = torch.zeros(T - 1, dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            xb, yb = X[i:i + args.batch].to(device), Y[i:i + args.batch].to(device)
            logits = _fwd(xb)
            l = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                yb.reshape(-1), reduction="none")
            pos_loss += l.view(xb.size(0), T)[:, :T - 1].double().sum(0).cpu()
    pos_loss /= len(X)

    print(f"\n[context-usage] {label}  ({args.sequences} sequences x {T} tokens, "
          f"{args.filler_dataset})")
    nb = 8
    w = (T - 1) // nb
    print("  position bucket | mean loss (nats)")
    for b in range(nb):
        seg = pos_loss[b * w:(b + 1) * w]
        print(f"   {b*w:>4}-{(b+1)*w-1:<4}      | {seg.mean():.4f}")
    early = pos_loss[1:17].mean()      # positions 1-16 (skip the contextless first)
    late = pos_loss[-32:].mean()       # last 32 positions
    print(f"  ICL score (early[1:17] - late[-32:]) : {early - late:+.4f} nats")
    print(f"  first token (no context) : {pos_loss[0]:.4f}   "
          f"last token (max context) : {pos_loss[-1]:.4f}")


if __name__ == "__main__":
    main()
