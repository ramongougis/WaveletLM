"""scale_ablation.py — Study 3: exact causal ablation of the scale axis.

Study 2 falsified the channel-level privileged-basis claim (Finding 10). The SCALE
axis is the half that survived — it is privileged *by construction*, not by hope —
so this is now the load-bearing experiment.

The ablation is EXACT and needs no hooks. In the block,
    scaled = mixed * self.scale_weights[idx]
so setting `scale_weights[s] = 0` removes scale s's entire contribution before
reconstruction, and reconstruction is exact. That is the invertibility advantage
over SAE-based causal claims: no reconstruction-error caveat, ever.

Reports, per (layer x scale), both metrics the literature asks for (Zhang & Nanda:
probability-space and logit-space disagree in known ways and reviewers will ask):
  * dBPB     — probability-space, via the project's dBPB = dnats / 3.124
  * dloss    — raw nats
and, for orientation, the fraction of total damage each scale accounts for.

Usage:
  python tools/interpretability/scale_ablation.py --run_dir logs/wikitext-103_2026-07-15_10-53-46
"""
import argparse, json, os, sys
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from model import WaveletLM                      # noqa: E402
from train import load_and_encode_dataset        # noqa: E402

NATS_PER_BPB = 3.124   # project convention: dBPB = dval-loss(nats) / 3.124


class _L:
    def log(self, m): print(m, flush=True)


def load_model(run_dir, device):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    cfg["compile"] = False
    m = WaveletLM(cfg.get("vocab_size", 50257), cfg, device=device).to(device)
    ck = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(ck):
        raise SystemExit(f"no best_model.pt in {run_dir}")
    obj = torch.load(ck, map_location=device, weights_only=False)
    sd = obj
    if isinstance(obj, dict):
        for k in ("model_state", "model_state_dict", "model", "state_dict"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]; break
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items() if torch.is_tensor(v)}
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[load] {ck}  (missing {len(miss)}, unexpected {len(unexp)})")
    return m, cfg


@torch.no_grad()
def evaluate(model, batches):
    tot, n = 0.0, 0
    for x, y in batches:
        _, loss = model(x, targets=y)
        tot += loss.item(); n += 1
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = torch.device(args.device)
    model, cfg = load_model(args.run_dir, dev); model.eval()
    T = cfg.get("block_size", 256)
    _, va, _, _, _ = load_and_encode_dataset(
        {"dataset": cfg.get("dataset", "wikitext-103"), "tokenizer": "auto", "block_size": T}, _L())

    # fixed evaluation set — identical for every ablation
    g = torch.Generator().manual_seed(1234)
    n_win = max(args.tokens // T, args.batch)
    starts = torch.randint(0, len(va) - T - 1, (n_win,), generator=g)
    batches = []
    for i in range(0, n_win, args.batch):
        sel = starts[i:i + args.batch]
        if len(sel) < args.batch:
            break
        x = torch.stack([va[s:s + T] for s in sel]).to(dev).long()
        y = torch.stack([va[s + 1:s + T + 1] for s in sel]).to(dev).long()
        batches.append((x, y))
    print(f"[eval] {len(batches)*args.batch} windows x {T} tokens "
          f"= {len(batches)*args.batch*T:,} tokens, fixed across ablations")

    base = evaluate(model, batches)
    print(f"[base] loss {base:.4f} nats\n")

    S = model.layers[0].scale_weights.numel()
    L = len(model.layers)
    print(f"{'layer':>5} " + " ".join(f"{'s'+str(s):>8}" for s in range(S)) + "   (dBPB)")
    rows, grid = [], []
    for li, blk in enumerate(model.layers):
        line = []
        for s in range(S):
            orig = blk.scale_weights[s].item()
            with torch.no_grad():
                blk.scale_weights[s] = 0.0
            abl = evaluate(model, batches)
            with torch.no_grad():
                blk.scale_weights[s] = orig
            dn = abl - base
            dbpb = dn / NATS_PER_BPB
            line.append(dbpb)
            rows.append(dict(layer=li, scale=s, dloss_nats=dn, dBPB=dbpb,
                             abl_loss=abl, orig_weight=orig))
        grid.append(line)
        print(f"{li:>5} " + " ".join(f"{v:8.4f}" for v in line), flush=True)

    flat = [r["dBPB"] for r in rows]
    tot = sum(abs(v) for v in flat) or 1.0
    print(f"\n[base] {base:.4f} nats   |   ablations: {len(rows)}")
    top = sorted(rows, key=lambda r: -r["dBPB"])[:8]
    print("\nMost damaging ablations:")
    for r in top:
        print(f"  L{r['layer']:02d}/s{r['scale']}  dBPB {r['dBPB']:+.4f}  "
              f"({100*abs(r['dBPB'])/tot:4.1f}% of total damage)  w={r['orig_weight']:+.3f}")
    per_scale = [sum(g[s] for g in grid) for s in range(S)]
    print("\nDamage summed over layers, per scale (s0 = approximation, s1..sN coarse->fine):")
    for s, v in enumerate(per_scale):
        print(f"  s{s}: {v:+.4f} dBPB")

    if args.out:
        json.dump(dict(base_loss=base, rows=rows), open(args.out, "w"), indent=1)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
