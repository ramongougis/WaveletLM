"""mqar_mixed.py — the "does AMB earn its place in a language model" test.

WT-103 alone is recall-light: training the AMB on it installs no recall (the
frozen adapter AMBA v1 came back flat, beta-collapsed, QRY-Δ unchanged). The AMB
only learns recall when the OBJECTIVE rewards it (MQAR 96%). So this trains one
model on a MIXED objective — real WT-103 language-modeling batches interleaved
with generated MQAR recall batches (shared GPT-2 vocab) — so the loss has BOTH a
language signal and a recall signal. Runs vanilla vs +AMB.

Success (for +AMB): MQAR accuracy RISES (recall installed) while WT-103 val stays
~vanilla (language not degraded). That is: an attention-free LM that also does
in-context recall, because we finally gave it a reason to. (Then confirm with
recall_diagnostics on the checkpoint — QRY-Δ should leave ~0, unlike AMBA v1.)

Both tasks share the GPT-2 vocab so one model does both; MQAR keys/values are
random mid-vocab GPT-2 ids, causally masked (targets at pos+1, never in context).

Usage (laptop concept-check, then scale on the pod):
  python tools/interpretability/mqar_mixed.py --steps 3000
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
from train import load_and_encode_dataset  # noqa: E402


class _L:
    def log(self, m):
        print(m, flush=True)


def make_mqar_batch(bs, V, D, Q, T, device, gen, key_lo=1000, key_hi=40000):
    """GPT-2-vocab MQAR in a T-token block (pairs, then queries, zero-padded).
    Returns (x, loss_mask); mask marks query-KEY positions whose next token is
    the bound value to predict."""
    x = torch.zeros(bs, T, dtype=torch.long)
    mask = torch.zeros(bs, T, dtype=torch.bool)
    for b in range(bs):
        keys = torch.randperm(key_hi - key_lo, generator=gen)[:D] + key_lo
        vals = torch.randint(key_lo, key_hi, (D,), generator=gen)
        for i in range(D):
            x[b, 2 * i] = keys[i]; x[b, 2 * i + 1] = vals[i]
        qsel = torch.randint(0, D, (Q,), generator=gen)
        for j in range(Q):
            pos = 2 * D + 2 * j
            x[b, pos] = keys[qsel[j]]
            x[b, pos + 1] = vals[qsel[j]]
            mask[b, pos] = True
    return x.to(device), mask.to(device)


def make_wt_batch(bs, data, T, device, gen):
    """Random WT-103 windows: x (T), y = next-token (T)."""
    starts = torch.randint(0, len(data) - T - 1, (bs,), generator=gen)
    x = torch.stack([data[s:s + T] for s in starts]).to(device)
    y = torch.stack([data[s + 1:s + T + 1] for s in starts]).to(device)
    return x.long(), y.long()


def build_model(V, over, device):
    base = json.load(open("logs/wikitext-103_2026-07-12_08-04-38/config.json"))
    cfg = dict(base)
    cfg.update({"compile": False, "per_scale_mixer_widths": None, "wavelet_crawl": True,
                "mlp_expansion": 0, "skip_proj_out": True, "mixer_transform": "identity",
                "wavelet_basis": "real", "pkm_enabled": False, "fwpkm_enabled": False,
                # persistent bypass state leaks across (unrelated) random batches and
                # breaks when train/eval batch sizes differ — off for this probe.
                "decompose_bypass_cross_window": False,
                "dropout_embedding": 0.0, "dropout_projection": 0.0, "dropout_mixer": 0.0,
                "dropout_mlp": 0.0, "dropout_lm_head": 0.0})
    cfg.update(over)
    return WaveletLM(V, cfg, device=device).to(device)


def run(label, assoc, args, wt_train, wt_val, device):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 1)
    V = 50257
    model = build_model(V, {"C": args.dim, "layers": args.layers, "levels": args.levels,
                            "block_size": args.block, "wavelet_crawl_k": args.crawl_k,
                            "associative_bypass_enabled": assoc,
                            "associative_bypass_dim": args.assoc_dim,
                            "associative_bypass_feature_map": args.feature_map,
                            "associative_bypass_per_scale": args.per_scale}, device)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\n--- {label} (assoc={assoc}, {n/1e6:.1f}M params) ---")
    model.train()
    fin_wt = fin_mqar = 0.0
    for step in range(1, args.steps + 1):
        # coin-flip mix: an MQAR recall step or a WT-103 LM step
        if torch.rand((), generator=gen).item() < args.mqar_prob:
            x, m = make_mqar_batch(args.batch, V, args.pairs, args.queries, args.block, device, gen)
            logits, _ = model(x, targets=None)
            lg = logits[:, :-1, :]; tgt = x[:, 1:]; lm = m[:, :-1]
            loss = F.cross_entropy(lg[lm], tgt[lm])
        else:
            x, y = make_wt_batch(args.batch, wt_train, args.block, device, gen)
            _, loss = model(x, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.report == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                # WT-103 val loss (held-out windows)
                xv, yv = make_wt_batch(64, wt_val, args.block, device, gen)
                _, wl = model(xv, targets=yv)
                # MQAR held-out accuracy (fresh pairs)
                xm, mm = make_mqar_batch(64, V, args.pairs, args.queries, args.block, device, gen)
                lgm, _ = model(xm, targets=None)
                lg = lgm[:, :-1, :]; tg = xm[:, 1:]; ma = mm[:, :-1]
                acc = (lg[ma].argmax(-1) == tg[ma]).float().mean().item()
            model.train()
            fin_wt, fin_mqar = wl.item(), acc
            print(f"  step {step:>5} | WT-103 val {wl.item():.3f} | MQAR acc {acc:6.1%} "
                  f"(chance {1/(args.batch):.1%}-ish)")
    return fin_wt, fin_mqar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=256)      # C (modest for laptop; scale on pod)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--levels", type=int, default=6)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--crawl_k", type=int, default=33)
    p.add_argument("--assoc_dim", type=int, default=64)
    p.add_argument("--feature_map", default="softplus_l2",
                   choices=["elu1","relu2","relu_l2","softplus_l2","relu2_l2","softplus_s"])
    p.add_argument("--per_scale", action="store_true",
                   help="inject the AMB read in coefficient space, one gain per scale")
    p.add_argument("--pairs", type=int, default=32)     # D  (2(D+Q)<=block)
    p.add_argument("--queries", type=int, default=32)   # Q
    p.add_argument("--mqar_prob", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--report", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--only", choices=["vanilla", "assoc"], default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    assert 2 * (args.pairs + args.queries) <= args.block, "MQAR structure must fit the block"
    device = torch.device(args.device)

    cfg = {"dataset": "wikitext-103", "tokenizer": "auto", "block_size": args.block}
    wt_train, wt_val, _, _, _ = load_and_encode_dataset(cfg, _L())
    print(f"[MQAR-mixed] mixing WT-103 LM + MQAR (D={args.pairs},Q={args.queries}) "
          f"@ p_mqar={args.mqar_prob} | device={device}")

    vw = vm = aw = am = None
    if args.only in (None, "vanilla"):
        vw, vm = run("VANILLA (WT-103 + MQAR)", False, args, wt_train, wt_val, device)
    if args.only in (None, "assoc"):
        aw, am = run("+AMB (WT-103 + MQAR)", True, args, wt_train, wt_val, device)
    if vw is not None and aw is not None:
        print("\n=== VERDICT ===")
        print(f"  vanilla: WT-103 val {vw:.3f} | MQAR acc {vm:5.1%}")
        print(f"  +AMB   : WT-103 val {aw:.3f} | MQAR acc {am:5.1%}")
        recall_win = am - vm > 0.2
        lm_ok = aw <= vw + 0.05
        if recall_win and lm_ok:
            print("  +AMB LEARNS RECALL (MQAR up) while WT-103 stays healthy -> earns its place.")
        elif recall_win:
            print("  +AMB learns recall but WT-103 val regressed -> task interference; tune mqar_prob/lr.")
        else:
            print("  +AMB did NOT learn recall even with the mixed signal -> wiring problem, not objective.")


if __name__ == "__main__":
    main()
