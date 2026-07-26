"""salt_store.py — SALT store build + oracle ceiling, sharded for a many-core box.

Supersedes the single-process `salt_pilot.py` for anything past a toy scale.

WHY THIS EXISTS. The first pilot (300K tokens, top-1% by loss) returned an oracle
ceiling of 0.04% — but it could not have tested the key. Two faults, both fixed here:
  1. median partition size was 1, so "rank by scale-similarity" chose among ONE
     candidate. Fix: `--keep_pct 100` (store everything) -> dense partitions.
  2. top-N%-by-loss selects hapaxes: 42.2% of its targets appeared exactly ONCE in
     the corpus, so their context can never recur and they can never be retrieved.
     Fix: no loss-based selection at all, matching kNN-LM's actual mechanism (its
     gain comes from the aggregate over many neighbours, not one exact match).

PARALLELISM. PyTorch CPU inference scales poorly past ~8 threads on one model, so
speedup comes from N PROCESSES over corpus slices, not from more threads each.
Use --threads 6 and run 8 shards => 48 cores, near-linear.

  build:  python salt_store.py --mode build --shard i --n_shards 8 --threads 6 ...
  eval :  python salt_store.py --mode eval  --store_dir .salt/wt103 ...
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from train import load_and_encode_dataset        # noqa: E402
from tools.interpretability.scale_ablation import load_model   # noqa: E402


class _L:
    def log(self, m): print(m, flush=True)


def make_keys(C, scales):
    """(T,S,Cp) -> (T, len(scales)*Cp) unit key. Per-scale normalize FIRST so the
    U-shaped gain profile (Finding 1) does not silently weight the match 2-3x."""
    parts = []
    for s in scales:
        v = C[:, s, :]
        parts.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    K = torch.cat(parts, dim=-1)
    return K / K.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def sweep(model, data, lo, hi, T, S, layer, scales, dev, tag):
    store, hooks = {}, []
    def mk(si):
        def h(_m, _i, out): store[si] = out.detach()[0]
        return h
    for si in range(S):
        hooks.append(model.layers[layer].decomp_norms[si].register_forward_hook(mk(si)))
    K, NXT, LST, NLL = [], [], [], []
    n_win = (hi - lo) // T
    for w in range(n_win):
        s0 = lo + w * T
        if s0 + T + 1 >= len(data):
            break
        x = data[s0:s0 + T].unsqueeze(0).to(dev).long()
        y = data[s0 + 1:s0 + T + 1].to(dev).long()
        store.clear()
        with torch.no_grad():
            logits, _ = model(x)
        C = torch.stack([store[s] for s in range(S)], dim=1)
        lp = F.log_softmax(logits[0].float(), -1)
        K.append(make_keys(C.float(), scales).half().cpu())      # fp16 keys: 2x smaller
        NXT.append(y.cpu().to(torch.int32)); LST.append(x[0].cpu().to(torch.int32))
        NLL.append(F.nll_loss(lp, y, reduction="none").cpu().half())
        if (w + 1) % 250 == 0:
            print(f"  [{tag}] {(w+1)*T:,}/{n_win*T:,}", flush=True)
    for h in hooks: h.remove()
    return (torch.cat(K).numpy(), torch.cat(NXT).numpy(),
            torch.cat(LST).numpy(), torch.cat(NLL).numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["build", "eval"], required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--store_dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--build_tokens", type=int, default=10_000_000, help="total across shards")
    p.add_argument("--eval_tokens", type=int, default=50_000)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--scales", type=int, nargs="+", default=[0, 1])
    p.add_argument("--keep_pct", type=float, default=100.0,
                   help="%% highest-loss kept; 100 = store everything (recommended)")
    p.add_argument("--topk", type=int, nargs="+", default=[1, 5, 20, 100])
    p.add_argument("--no_partition", action="store_true")
    p.add_argument("--min_freq", type=int, default=0,
                   help="drop entries whose target token is rarer than this (anti-hapax)")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    dev = torch.device(args.device)
    model, cfg = load_model(args.run_dir, dev); model.eval()
    T = cfg.get("block_size", 256); S = model.layers[0].scale_weights.numel()
    for blk in model.layers:
        if hasattr(blk, "decompose_bypass_cross_window"):
            blk.decompose_bypass_cross_window = False
    tr, va, _, _, _ = load_and_encode_dataset(
        {"dataset": cfg.get("dataset", "wikitext-103"), "tokenizer": "auto", "block_size": T}, _L())
    os.makedirs(args.store_dir, exist_ok=True)

    if args.mode == "build":
        per = args.build_tokens // args.n_shards
        lo = args.shard * per
        hi = lo + per
        print(f"[build] shard {args.shard}/{args.n_shards}: tokens [{lo:,},{hi:,}) "
              f"threads={args.threads} scales={args.scales} layer={args.layer}")
        K, N, L, E = sweep(model, tr, lo, hi, T, S, args.layer, args.scales, dev,
                           f"s{args.shard}")
        if args.keep_pct < 100:
            thr = np.percentile(E.astype(np.float32), 100 - args.keep_pct)
            m = E.astype(np.float32) >= thr
            K, N, L, E = K[m], N[m], L[m], E[m]
            print(f"[build] kept {m.sum():,} ({100*m.mean():.2f}%) at nll>={thr:.2f}")
        out = os.path.join(args.store_dir, f"shard{args.shard:03d}.npz")
        np.savez(out, K=K, N=N, L=L, E=E)
        print(f"[build] wrote {out}  rows={len(N):,}  keys={K.nbytes/1e9:.2f} GB")
        return

    # ---- eval ----
    files = sorted(glob.glob(os.path.join(args.store_dir, "shard*.npz")))
    if not files:
        raise SystemExit(f"no shards in {args.store_dir}")
    Ks, Ns, Ls = [], [], []
    for f in files:
        z = np.load(f); Ks.append(z["K"]); Ns.append(z["N"]); Ls.append(z["L"])
    Ks = np.concatenate(Ks); Ns = np.concatenate(Ns); Ls = np.concatenate(Ls)
    print(f"[eval] store: {len(files)} shards, {len(Ns):,} rows, "
          f"{Ks.nbytes/1e9:.2f} GB keys (fp16)")

    if args.min_freq > 0:
        cnt = np.bincount(Ns.astype(np.int64), minlength=50257)
        keep = cnt[Ns.astype(np.int64)] >= args.min_freq
        Ks, Ns, Ls = Ks[keep], Ns[keep], Ls[keep]
        print(f"[eval] anti-hapax filter (min_freq={args.min_freq}): "
              f"{keep.sum():,} rows kept ({100*keep.mean():.1f}%)")

    part = {}
    for i, t in enumerate(Ls):
        part.setdefault(int(t), []).append(i)
    part = {k: np.array(v) for k, v in part.items()}
    sz = np.array([len(v) for v in part.values()])
    print(f"[eval] partitions: {len(part):,} | median {np.median(sz):.0f} "
          f"p90 {np.percentile(sz,90):.0f} max {sz.max():,}")

    Ke, Ne, Le, _ = sweep(model, va, 0, args.eval_tokens, T, S, args.layer,
                          args.scales, dev, "eval")
    Ksf = Ks.astype(np.float32)
    maxk = max(args.topk)
    hits = {k: 0 for k in args.topk}; top1 = 0; covered = 0
    for i in range(len(Ke)):
        cand = np.arange(len(Ns)) if args.no_partition else part.get(int(Le[i]))
        if cand is None or len(cand) == 0:
            continue
        covered += 1
        sims = Ksf[cand] @ Ke[i].astype(np.float32)
        order = cand[np.argsort(-sims)[:maxk]]
        cont = Ns[order]
        if cont[0] == Ne[i]: top1 += 1
        for k in args.topk:
            if Ne[i] in cont[:k]: hits[k] += 1

    cov = covered / max(len(Ke), 1)
    print(f"\n=== RESULTS ({len(Ke):,} eval positions, partition="
          f"{'OFF' if args.no_partition else 'ON'}) ===")
    print(f"  coverage                    : {cov:7.2%}")
    if covered:
        print(f"  top1  | covered             : {top1/covered:7.2%}")
        for k in args.topk:
            print(f"  oracle@{k:<4}| covered         : {hits[k]/covered:7.2%}")
        print(f"\n  ORACLE CEILING (uncond.)    : {hits[maxk]/len(Ke):7.2%}")
        print("  -> bounds ANY gate/interpolation built on this key.")


if __name__ == "__main__":
    main()
