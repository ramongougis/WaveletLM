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
from tools.interpretability.scale_ablation import load_model, NATS_PER_BPB as NATS   # noqa: E402


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


def sweep(model, data, lo, hi, T, S, layer, scales, dev, tag, batch=32):
    """BATCHED sweep. At batch=1 a 73M model barely occupies BLAS — batching is
    worth ~an order of magnitude on CPU and far more on GPU. Safe because
    decompose_bypass_cross_window is disabled by the caller, so windows are
    independent and batch composition cannot leak state between them."""
    store, hooks = {}, []
    def mk(si):
        def h(_m, _i, out): store[si] = out.detach()      # (B,T,Cp)
        return h
    for si in range(S):
        hooks.append(model.layers[layer].decomp_norms[si].register_forward_hook(mk(si)))
    K, NXT, LST, NLL, ARG, ENT = [], [], [], [], [], []
    starts = [lo + w * T for w in range((hi - lo) // T)]
    starts = [s0 for s0 in starts if s0 + T + 1 < len(data)]
    done = 0
    for b0 in range(0, len(starts), batch):
        grp = starts[b0:b0 + batch]
        x = torch.stack([data[s0:s0 + T] for s0 in grp]).to(dev).long()          # (B,T)
        y = torch.stack([data[s0 + 1:s0 + T + 1] for s0 in grp]).to(dev).long()
        store.clear()
        with torch.no_grad():
            logits, _ = model(x)
        C = torch.stack([store[s] for s in range(S)], dim=2)                      # (B,T,S,Cp)
        B_, T_, S_, Cp_ = C.shape
        lp = F.log_softmax(logits.float().reshape(-1, logits.shape[-1]), -1)
        K.append(make_keys(C.float().reshape(B_ * T_, S_, Cp_), scales).half().cpu())
        NXT.append(y.reshape(-1).cpu().to(torch.int32))
        LST.append(x.reshape(-1).cpu().to(torch.int32))
        NLL.append(F.nll_loss(lp, y.reshape(-1), reduction="none").cpu().half())
        ARG.append(lp.argmax(-1).cpu().to(torch.int32))   # model's own top-1
        # Predictive entropy. The ONLY difficulty signal here that is computable at
        # inference: NLL needs the true next token, so an NLL gate silently peeks at the
        # answer and can only ever be an upper bound (see plans/salt.md Test A caveat 1).
        ENT.append((-(lp.exp() * lp).sum(-1)).cpu().half())
        done += len(grp)
        if done % 256 < batch:
            print(f"  [{tag}] {done*T:,}/{len(starts)*T:,}", flush=True)
    for h in hooks: h.remove()
    return (torch.cat(K).numpy(), torch.cat(NXT).numpy(),
            torch.cat(LST).numpy(), torch.cat(NLL).numpy(),
            torch.cat(ARG).numpy(), torch.cat(ENT).numpy())


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
    p.add_argument("--interp", action="store_true",
                   help="kNN-LM interpolation test: mix the similarity-weighted retrieved "
                        "distribution with the model and report dBPB. This is the mechanism the "
                        "method actually uses — top-1 accuracy understates it, since interpolation "
                        "only needs MASS on the right token, not rank 1.")
    p.add_argument("--random_rank", action="store_true",
                   help="CONTROL: rank candidates RANDOMLY instead of by similarity. "
                        "With median partition ~7, oracle@100 is largely a bigram statistic; "
                        "only top1-vs-random isolates what scale-similarity contributes.")
    p.add_argument("--max_cand", type=int, default=50000,
                   help="cap candidates scored per query (random subsample above this)")
    p.add_argument("--min_freq", type=int, default=0,
                   help="drop entries whose target token is rarer than this (anti-hapax)")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--batch", type=int, default=32,
                   help="windows per forward; 1 wastes BLAS, 32+ is much faster")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    dev = torch.device(args.device)
    model, cfg = load_model(args.run_dir, dev); model.eval()
    T = cfg.get("block_size", 256); S = model.layers[0].scale_weights.numel()
    # Disable cross-window bypass state at BOTH levels. The persistent state lives on
    # the MODEL (model.py ~3873), not the blocks, and it is batch-shaped: carrying it
    # across forwards crashes the moment a batch has a different size — which is exactly
    # what the final partial batch does. Setting only the per-block flags is insufficient.
    model.decompose_bypass_cross_window = False
    model._persistent_semantic_state = None
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
        K, N, L, E, A, H = sweep(model, tr, lo, hi, T, S, args.layer, args.scales, dev,
                           f"s{args.shard}", batch=args.batch)
        if args.keep_pct < 100:
            thr = np.percentile(E.astype(np.float32), 100 - args.keep_pct)
            m = E.astype(np.float32) >= thr
            K, N, L, E, A, H = K[m], N[m], L[m], E[m], A[m], H[m]
            print(f"[build] kept {m.sum():,} ({100*m.mean():.2f}%) at nll>={thr:.2f}")
        out = os.path.join(args.store_dir, f"shard{args.shard:03d}.npz")
        # A (model argmax) and H (predictive entropy) are saved so gate selection never
        # needs a re-sweep. Earlier builds lack them; salt_gate.py re-sweeps in that case.
        np.savez(out, K=K, N=N, L=L, E=E, A=A, H=H)
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

    # SORT the store by last-token so each partition is a CONTIGUOUS SLICE.
    # Critical: fancy indexing (Ks[cand]) COPIES — with store-everything a common
    # token's partition can be ~500K rows x 1024 dims = ~2GB copied PER QUERY.
    # Sorting turns partition access into a view, which costs nothing.
    ordr = np.argsort(Ls, kind="stable")
    Ks, Ns, Ls = Ks[ordr], Ns[ordr], Ls[ordr]
    uniq, starts = np.unique(Ls, return_index=True)
    ends = np.append(starts[1:], len(Ls))
    span = {int(t): (int(a), int(b)) for t, a, b in zip(uniq, starts, ends)}
    sz = ends - starts
    print(f"[eval] partitions: {len(span):,} | median {np.median(sz):.0f} "
          f"p90 {np.percentile(sz,90):.0f} max {sz.max():,}")

    Ke, Ne, Le, Ee, Ae, He = sweep(model, va, 0, args.eval_tokens, T, S, args.layer,
                          args.scales, dev, "eval", batch=args.batch)
    Ksf = np.ascontiguousarray(Ks.astype(np.float32))
    Kef = Ke.astype(np.float32)
    maxk = max(args.topk)
    rng = np.random.default_rng(0)
    hits = {k: 0 for k in args.topk}; top1 = 0; covered = 0; capped = 0
    mw = 0; mw_top1 = 0; mw_oracle = 0; model_ok = 0
    I_sims, I_match, I_nll, I_wrong = [], [], [], []
    for i in range(len(Ke)):
        if args.no_partition:
            lo, hi = 0, len(Ns)
        else:
            sp = span.get(int(Le[i]))
            if sp is None:
                continue
            lo, hi = sp
        n = hi - lo
        if n == 0:
            continue
        covered += 1
        if n > args.max_cand:                 # bound per-query work
            idx = lo + rng.choice(n, args.max_cand, replace=False)
            sims = (rng.standard_normal(len(idx)).astype(np.float32)
                    if args.random_rank else Ksf[idx] @ Kef[i])
            capped += 1
        else:
            idx = np.arange(lo, hi)
            sims = (rng.standard_normal(n).astype(np.float32)
                    if args.random_rank else Ksf[lo:hi] @ Kef[i])   # VIEW, no copy
        kk = min(maxk, len(sims))
        take = np.argpartition(-sims, kk - 1)[:kk] if kk < len(sims) else np.arange(len(sims))
        take = take[np.argsort(-sims[take])]
        cont = Ns[idx[take]]
        if cont[0] == Ne[i]: top1 += 1
        for k in args.topk:
            if Ne[i] in cont[:k]: hits[k] += 1
        if args.interp:
            ss = sims[take].astype(np.float32)
            I_sims.append(np.pad(ss, (0, maxk - len(ss)), constant_values=-1e9))
            I_match.append(np.pad((cont == Ne[i]).astype(np.float32), (0, maxk - len(cont))))
            I_nll.append(float(Ee[i])); I_wrong.append(bool(Ae[i] != Ne[i]))
        # THE decisive slice: SALT can only help where the model is already wrong.
        if Ae[i] == Ne[i]:
            model_ok += 1
        else:
            mw += 1
            if cont[0] == Ne[i]: mw_top1 += 1
            if Ne[i] in cont[:maxk]: mw_oracle += 1

    cov = covered / max(len(Ke), 1)
    print(f"\n=== RESULTS ({len(Ke):,} eval positions, partition="
          f"{'OFF' if args.no_partition else 'ON'}"
          f"{', RANDOM-RANK CONTROL' if args.random_rank else ''}) ===")
    print(f"  coverage                    : {cov:7.2%}")
    if covered:
        print(f"  top1  | covered             : {top1/covered:7.2%}")
        for k in args.topk:
            print(f"  oracle@{k:<4}| covered         : {hits[k]/covered:7.2%}")
        if capped:
            print(f"  (partition capped at {args.max_cand:,} for {capped:,}/{covered:,} queries)")
        print(f"\n  --- the slice that decides usefulness ---")
        print(f"  model top-1 (same positions): {model_ok/covered:7.2%}")
        print(f"  positions model got WRONG   : {mw:,} ({mw/covered:.1%})")
        if mw:
            print(f"  retrieval top1 | model WRONG: {mw_top1/mw:7.2%}   <- direct repair rate")
            print(f"  oracle@{maxk} | model WRONG   : {mw_oracle/mw:7.2%}   <- ceiling on repair")
        print(f"\n  ORACLE CEILING (uncond.)    : {hits[maxk]/len(Ke):7.2%}")
        print("  -> bounds ANY gate/interpolation built on this key.")

    if args.interp and I_sims:
        # THE MECHANISM TEST. kNN-LM does not substitute the top-1 neighbour; it mixes a
        # similarity-weighted distribution over the top-k with the model's own. So it only
        # needs MASS on the true token, not rank 1 — which is why a 34% oracle with a 3.6%
        # top-1 is exactly the profile where interpolation can win and exact retrieval loses.
        # p_model[y] = exp(-nll) is already known, so no 50K-vocab distribution is needed.
        Sm = np.stack(I_sims); Mm = np.stack(I_match)
        nll0 = np.array(I_nll, dtype=np.float32); wrong = np.array(I_wrong)
        p_model = np.exp(-nll0); base = float(nll0.mean())
        print(f"\n=== kNN-LM INTERPOLATION (the mechanism the method actually uses) ===")
        print(f"  baseline model NLL (covered positions): {base:.4f} nats")
        print(f"  p_mix = (1-lam)*p_model + lam*p_retrieval,  weights = softmax(sim/tau)")
        print(f"{'tau':>6} {'lam':>6} {'dNLL':>10} {'dBPB':>10} {'dBPB|wrong':>12}")
        best = None
        for tau in (0.02, 0.05, 0.1, 0.3, 1.0):
            W = np.exp((Sm - Sm.max(1, keepdims=True)) / tau)
            W /= W.sum(1, keepdims=True).clip(1e-9)
            p_ret = (W * Mm).sum(1)                 # retrieval mass on the TRUE token
            for lam in (0.05, 0.1, 0.2, 0.3, 0.5):
                nll1 = -np.log(((1 - lam) * p_model + lam * p_ret).clip(1e-12))
                d = float(nll1.mean() - base)
                dw = float((nll1[wrong] - nll0[wrong]).mean()) if wrong.any() else float("nan")
                print(f"{tau:6.2f} {lam:6.2f} {d:+10.4f} {d/NATS:+10.4f} {dw/NATS:+12.4f}")
                if best is None or d < best[0]:
                    best = (d, tau, lam)
        print(f"\n  BEST: tau={best[1]} lam={best[2]}  dNLL={best[0]:+.4f}  dBPB={best[0]/NATS:+.4f}")
        print("  NEGATIVE dBPB = improvement. Noise floor 0.0010.")
        print("  If nothing is negative, a 34% oracle cannot be converted into a gain and")
        print("  the retrieved neighbours are too noisy -> SALT is done on this key.")


if __name__ == "__main__":
    main()
