"""mqar.py — Multi-Query Associative Recall: the go/no-go for the memory bypass.

Zoology's synthetic recall task (Arora et al., arXiv:2312.04927). A sequence of
key->value pairs, then queries that repeat keys; the model must predict the value
bound to each queried key. Vanilla WaveletLM should FAIL (no content-addressable
retrieval + bounded wavelet reach); the associative-memory bypass — a global
causal linear-attention state — should SOLVE it. That contrast validates the
mechanism BEFORE any WT-103 spend.

    pairs   : k0 v0 k1 v1 ... k_{D-1} v_{D-1}       (D distinct keys)
    queries : kq0 vq0 kq1 vq1 ...                    (loss at each query-key
              position, predicting the bound value)

By default trains BOTH (vanilla, then bypass) on identical data and prints the
comparison. ~minutes on a laptop. Usage:
  python tools/interpretability/mqar.py                       # both, defaults
  python tools/interpretability/mqar.py --pairs 24 --queries 24 --steps 2000
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from model import WaveletLM  # noqa: E402


def make_batch(bs, V, D, Q, device, gen):
    """One MQAR batch. Returns (x, loss_mask). loss_mask marks query-KEY
    positions p, whose next token x[p+1] is the bound value to predict."""
    seq_len = 2 * (D + Q)
    x = torch.zeros(bs, seq_len, dtype=torch.long)
    mask = torch.zeros(bs, seq_len, dtype=torch.bool)
    for b in range(bs):
        keys = torch.randperm(V, generator=gen)[:D]            # distinct keys
        vals = torch.randint(0, V, (D,), generator=gen)
        for i in range(D):
            x[b, 2 * i] = keys[i]; x[b, 2 * i + 1] = vals[i]
        qsel = torch.randint(0, D, (Q,), generator=gen)
        for j in range(Q):
            pos = 2 * D + 2 * j
            x[b, pos] = keys[qsel[j]]                           # query key
            x[b, pos + 1] = vals[qsel[j]]                       # bound value (target)
            mask[b, pos] = True                                 # predict value here
    return x.to(device), mask.to(device)


def build_model(V, cfg_over, device):
    base = json.load(open("logs/wikitext-103_2026-07-12_08-04-38/config.json"))
    cfg = dict(base)
    cfg.update({
        "compile": False, "per_scale_mixer_widths": None,
        "wavelet_crawl": True, "mlp_expansion": 0, "skip_proj_out": True,
        "mixer_transform": "identity", "wavelet_basis": "real",
        "pkm_enabled": False, "fwpkm_enabled": False,
        "dropout_embedding": 0.0, "dropout_projection": 0.0,
        "dropout_mixer": 0.0, "dropout_mlp": 0.0, "dropout_lm_head": 0.0,
    })
    cfg.update(cfg_over)
    return WaveletLM(V, cfg, device=device).to(device)


def train(label, assoc, args, device):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 1)
    seq_len = 2 * (args.pairs + args.queries)
    model = build_model(args.vocab, {
        "C": args.dim, "layers": args.layers, "levels": args.levels,
        "block_size": seq_len, "wavelet_crawl_k": args.crawl_k,
        "associative_bypass_enabled": assoc, "associative_bypass_dim": args.assoc_dim,
    }, device)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    chance = 1.0 / args.vocab
    print(f"\n--- {label} (assoc={assoc}, {n/1e6:.2f}M params, seq_len={seq_len}) ---")
    model.train()
    final = 0.0
    for step in range(1, args.steps + 1):
        x, mask = make_batch(args.batch, args.vocab, args.pairs, args.queries, device, gen)
        logits, _ = model(x, targets=None)
        lg = logits[:, :-1, :]; tgt = x[:, 1:]; lm = mask[:, :-1]
        loss = F.cross_entropy(lg[lm], tgt[lm])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.report == 0 or step == args.steps:
            with torch.no_grad():
                acc = (lg[lm].argmax(-1) == tgt[lm]).float().mean().item()
            final = acc
            print(f"  step {step:>5} | loss {loss.item():.3f} | query-value acc {acc:6.1%} "
                  f"(chance {chance:.1%})")
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--pairs", type=int, default=16)      # D
    p.add_argument("--queries", type=int, default=16)    # Q  -> seq_len = 2(D+Q) = 64
    p.add_argument("--dim", type=int, default=128)       # C
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--crawl_k", type=int, default=5)
    p.add_argument("--assoc_dim", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--report", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--only", choices=["vanilla", "assoc"], default=None,
                   help="run just one arm (default: both)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    print(f"[MQAR] vocab={args.vocab} pairs={args.pairs} queries={args.queries} "
          f"seq_len={2*(args.pairs+args.queries)} | device={device}")
    va = ba = None
    if args.only in (None, "vanilla"):
        va = train("VANILLA WaveletLM", False, args, device)
    if args.only in (None, "assoc"):
        ba = train("WaveletLM + associative-memory bypass", True, args, device)
    if va is not None and ba is not None:
        print(f"\n=== VERDICT ===")
        print(f"  vanilla query-value acc : {va:6.1%}")
        print(f"  +bypass query-value acc : {ba:6.1%}   (chance {1/args.vocab:.1%})")
        if ba > 0.8 and ba - va > 0.3:
            print("  BYPASS SOLVES MQAR where vanilla fails -> mechanism validated.")
        elif ba - va > 0.1:
            print("  bypass helps but doesn't solve -> tune (assoc_dim, layers, delta rule).")
        else:
            print("  bypass did NOT clearly help -> investigate before any WT-103 spend.")


if __name__ == "__main__":
    main()
