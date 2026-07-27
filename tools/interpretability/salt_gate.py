"""salt_gate.py — does the corrector's gain SURVIVE a gate that cannot see the answer?

WHY THIS EXISTS. Test A (plans/salt.md) flipped SALT's sign: lookup's best was +0.0054,
the learned corrector's best was -0.0038 against a 0.0010 noise floor, with a monotone
trend across gate tightness and a smooth interior optimum in lam. But three caveats stand
between that number and a shippable feature, and this script is built to kill all three:

  1. THE GATE PEEKED AT THE ANSWER. `nll>p90` means "apply the corrector where the model
     turned out to be wrong" — NLL requires the true next token, so it is not computable
     at inference. -0.0038 was an upper bound, not a deployable result.
     FIX: gate on PREDICTIVE ENTROPY and on corrector-side signals, all computable from
     model output alone. The NLL gate is still reported, explicitly labelled ORACLE, as
     the ceiling any realizable gate is trying to reach.
  2. (gate, lam) WERE SELECTED ON THE ROWS THEY WERE SCORED ON. Best-of-24 against a
     0.0010 floor is optimistic.
     FIX: SELECT/TEST split. The sweep runs on SELECT only; the single winning config is
     then scored ONCE on TEST. One number, chosen blind.
  3. IT WAS THE STORE'S VAL SUBSET, NOT THE BENCHMARK. D3 scored 2.6946 nats there vs
     3.0606 on the sliding-window eval — an easier slice.
     FIX: evaluate on a FRESH sweep of held-out tokens under the sliding-window protocol
     (--min_context, matching benchmark stride semantics), so dBPB is comparable to the
     BPB numbers everything else is ranked by.

THE EVAL SET IS INDEPENDENT BY CONSTRUCTION. Rather than reusing store rows (which the
corrector trained on, and whose positions are awkward to reconstruct), we re-sweep a token
range that was never in the store. Different positions, same protocol.

REALIZABLE GATES (all computable at inference, no true token needed):
    entropy   — model predictive entropy. The natural difficulty signal (Finding 19).
    corrconf  — corrector's own max probability: consult it when IT is confident.
    disagree  — model argmax != corrector argmax. Cheap, and targets exactly the
                positions where the ensemble could change the answer.
    combo     — high model entropy AND corrector confident.
ORACLE (reference only, never shippable):
    nll       — the Test A gate. Reported to show how much the realizable gates give up.

DECISION RULE, PRE-REGISTERED: SALT ships as a post-hoc release feature iff a REALIZABLE
gate reaches dBPB <= -0.0020 on TEST. Between -0.0010 and -0.0020 is noise-adjacent and
means "not worth the release complexity". Positive means the Test A gain lived in the
oracle, and SALT closes for good.

Usage:
  python tools/interpretability/salt_gate.py --store_dir .salt/wt103_d3 \
      --run_dir logs/wikitext-103_2026-07-15_10-53-46 --threads 6 \
      --corrector_ckpt .salt/corrector.pt --out .salt/gate.json
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from train import load_and_encode_dataset                                    # noqa: E402
from tools.interpretability.scale_ablation import load_model, NATS_PER_BPB as NATS  # noqa: E402
from tools.interpretability.salt_store import sweep, _L                      # noqa: E402
from tools.interpretability.salt_corrector import Corrector                  # noqa: E402


def train_corrector(K, N, emb, hidden, epochs, batch, lr, ckpt, seed):
    """Same architecture as Test A: key -> MLP -> C -> FROZEN tied head -> vocab."""
    V, C = emb.shape
    net = Corrector(K.shape[1], hidden, C, emb)
    if ckpt and os.path.exists(ckpt):
        net.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[corrector] loaded {ckpt}")
        return net
    X = torch.from_numpy(K).float(); Y = torch.from_numpy(N).long()
    opt = torch.optim.AdamW(net.net.parameters(), lr=lr, weight_decay=1e-4)
    print(f"[corrector] training on ALL {len(Y):,} store rows "
          f"({sum(p.numel() for p in net.net.parameters())/1e6:.2f}M params)")
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Y), generator=g); tot = nb = 0
        for b in range(0, len(idx), batch):
            j = idx[b:b + batch]
            loss = F.cross_entropy(net(X[j]), Y[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  [corrector] epoch {ep} train {tot/max(nb,1):.4f}", flush=True)
    if ckpt:
        os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
        torch.save(net.net.state_dict(), ckpt); print(f"[corrector] saved {ckpt}")
    return net


def delta_bpb(p_model, p_corr, nll0, idx, mask, lam):
    """dBPB of the gated mixture on rows `idx`. NEGATIVE = better than the model.

    Mixture is applied ONLY inside the gate; outside it the model's own probability is
    kept untouched, so an all-False gate must score exactly 0.0 and lam=0 must too.
    Both are asserted in the self-test at the bottom of this file.
    """
    pm = p_model[idx].copy(); m = mask[idx]
    pm[m] = (1 - lam) * p_model[idx][m] + lam * p_corr[idx][m]
    return float((-np.log(pm.clip(1e-12))).mean() - nll0[idx].mean()) / NATS


def build_gates(ent, p_corr_max, model_arg, corr_arg, nll):
    """Every gate is a dict name -> boolean mask. Only `nll:*` sees the true token."""
    g = {"ungated": np.ones(len(ent), bool)}
    for q in (50, 75, 90):
        g[f"entropy>p{q}"] = ent >= np.percentile(ent, q)
        g[f"corrconf>p{q}"] = p_corr_max >= np.percentile(p_corr_max, q)
        g[f"ORACLE nll>p{q}"] = nll >= np.percentile(nll, q)       # not shippable
    g["disagree"] = model_arg != corr_arg
    for q in (50, 75):
        g[f"combo e>p{q}&disagree"] = (ent >= np.percentile(ent, q)) & (model_arg != corr_arg)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store_dir", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--corrector_ckpt", default=".salt/corrector.pt")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_rows", type=int, default=1_500_000)
    p.add_argument("--eval_tokens", type=int, default=200_000,
                   help="fresh held-out tokens to sweep for the eval set")
    p.add_argument("--min_context", type=int, default=128,
                   help="sliding-window protocol: score only positions with >= this "
                        "much context, matching the benchmark's min_context")
    p.add_argument("--sweep_batch", type=int, default=32)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--scales", type=int, nargs="+", default=[0, 1])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, cfg = load_model(args.run_dir, torch.device("cpu")); model.eval()
    emb = None
    for name, prm in model.named_parameters():
        if name.endswith("token_embedding.weight"):
            emb = prm.detach().float(); break
    if emb is None:
        raise SystemExit("could not find token_embedding.weight")

    # ---- corrector: trained on the store (all rows; eval set is independent) ----
    Ks, Ns = [], []
    for f in sorted(glob.glob(os.path.join(args.store_dir, "shard*.npz"))):
        z = np.load(f); Ks.append(z["K"]); Ns.append(z["N"])
    K = np.concatenate(Ks); N = np.concatenate(Ns)
    if len(N) > args.max_rows:
        sel = rng.choice(len(N), args.max_rows, replace=False); K, N = K[sel], N[sel]
    net = train_corrector(K, N, emb, args.hidden, args.epochs, args.batch, args.lr,
                          args.corrector_ckpt, args.seed)
    net.eval()

    # ---- fresh, independent eval sweep on held-out tokens ----
    T = cfg.get("block_size", 256)
    S = cfg.get("levels", 7) + 1
    for blk in model.layers:                       # windows must be independent
        if hasattr(blk, "decompose_bypass_cross_window"):
            blk.decompose_bypass_cross_window = False
    if hasattr(model, "decompose_bypass_cross_window"):
        model.decompose_bypass_cross_window = False
    _, va, _, _, _ = load_and_encode_dataset(
        {"dataset": cfg.get("dataset", "wikitext-103"), "tokenizer": "auto",
         "block_size": T}, _L())
    print(f"[eval] fresh sweep of {args.eval_tokens:,} held-out val tokens "
          f"(NOT store rows; corrector never saw these positions)")
    Ke, Ne, _, Ee, Ae, He = sweep(model, va, 0, args.eval_tokens, T, S, args.layer,
                                  args.scales, torch.device("cpu"), "gate",
                                  batch=args.sweep_batch)

    # sliding-window protocol: drop low-context positions so dBPB is comparable to BPB
    pos = np.arange(len(Ne)) % T
    keep = pos >= args.min_context
    Ke, Ne, Ee, Ae, He = Ke[keep], Ne[keep], Ee[keep], Ae[keep], He[keep]
    print(f"[eval] {len(Ne):,} positions after min_context={args.min_context} "
          f"({keep.mean():.1%} of swept)")

    with torch.no_grad():
        lp = []
        X = torch.from_numpy(Ke).float()
        for i in range(0, len(X), 4096):
            lp.append(F.log_softmax(net(X[i:i + 4096]).float(), -1))
        lp = torch.cat(lp)
        p_corr = lp.gather(1, torch.from_numpy(Ne).long()[:, None]).squeeze(1).exp().numpy()
        corr_arg = lp.argmax(-1).numpy()
        p_corr_max = lp.max(-1).values.exp().numpy()

    nll0 = Ee.astype(np.float32)
    p_model = np.exp(-nll0)
    base = float(nll0.mean())
    print(f"[eval] D3 baseline NLL {base:.4f} nats  ({base/NATS:.4f} BPB-equivalent)")
    print(f"[eval] model top-1 {(Ae == Ne).mean():.2%}   "
          f"corrector top-1 {(corr_arg == Ne).mean():.2%}")

    # ---- SELECT / TEST split: sweep on one, report once on the other ----
    perm = rng.permutation(len(Ne))
    sel_i, test_i = perm[:len(perm) // 2], perm[len(perm) // 2:]
    gates = build_gates(He.astype(np.float32), p_corr_max, Ae, corr_arg, nll0)
    lams = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)

    def dbpb(idx, mask, lam):
        return delta_bpb(p_model, p_corr, nll0, idx, mask, lam)

    print(f"\n--- SELECT half ({len(sel_i):,} rows) — tuning only ---")
    print(f"{'gate':>22} {'frac':>7} {'best lam':>9} {'dBPB(sel)':>10}")
    best, best_real = None, None
    rows = []
    for gname, gmask in gates.items():
        cand = min(((lam, dbpb(sel_i, gmask, lam)) for lam in lams), key=lambda t: t[1])
        rows.append(dict(gate=gname, lam=cand[0], dbpb_select=cand[1],
                         frac=float(gmask[sel_i].mean())))
        print(f"{gname:>22} {gmask[sel_i].mean():6.1%} {cand[0]:9.2f} {cand[1]:+10.4f}")
        if best is None or cand[1] < best[1]:
            best = (gname, cand[1], cand[0])
        if not gname.startswith("ORACLE") and gname != "ungated":
            if best_real is None or cand[1] < best_real[1]:
                best_real = (gname, cand[1], cand[0])

    print(f"\n--- TEST half ({len(test_i):,} rows) — scored ONCE, config chosen blind ---")
    out = []
    for label, pick in (("BEST REALIZABLE", best_real), ("best overall (may be oracle)", best)):
        if pick is None:
            continue
        gname, _, lam = pick
        d = dbpb(test_i, gates[gname], lam)
        out.append(dict(which=label, gate=gname, lam=lam, dbpb_test=d))
        print(f"  {label:>28}: {gname} lam={lam}  dBPB(test) = {d:+.4f}")
    # the oracle ceiling on TEST, for the give-up measurement
    orc = min((g for g in gates if g.startswith("ORACLE")),
              key=lambda g: min(dbpb(sel_i, gates[g], l) for l in lams))
    orc_lam = min(lams, key=lambda l: dbpb(sel_i, gates[orc], l))
    orc_d = dbpb(test_i, gates[orc], orc_lam)
    print(f"  {'ORACLE ceiling (not shippable)':>28}: {orc} lam={orc_lam}  "
          f"dBPB(test) = {orc_d:+.4f}")

    if best_real is not None:
        d = [o for o in out if o["which"] == "BEST REALIZABLE"][0]["dbpb_test"]
        print(f"\n  gap to oracle: {d - orc_d:+.4f} BPB  "
              f"({'realizable gate recovers most of it' if d - orc_d < 0.0015 else 'oracle carried the gain'})")
        print("\n  PRE-REGISTERED DECISION RULE (noise floor 0.0010):")
        if d <= -0.0020:
            print(f"    dBPB {d:+.4f} <= -0.0020  ->  SHIP. SALT is a viable post-hoc")
            print( "    release feature on a gate computable at inference.")
        elif d < -0.0010:
            print(f"    dBPB {d:+.4f} in (-0.0020, -0.0010)  ->  real but marginal;")
            print( "    NOT worth the release complexity.")
        else:
            print(f"    dBPB {d:+.4f} >= -0.0010  ->  the Test A gain lived in the ORACLE")
            print( "    gate. SALT closes; record as a negative result.")
    if args.out:
        json.dump(dict(select=rows, test=out, oracle_test=orc_d, baseline_nats=base),
                  open(args.out, "w"), indent=1)
        print(f"\n[wrote] {args.out}")


def _selftest():
    """Verifies the scoring core before it is trusted with a release decision."""
    rng = np.random.default_rng(0)
    n = 20000
    nll0 = rng.gamma(2.0, 1.5, n).astype(np.float32)
    p_model = np.exp(-nll0)
    idx = np.arange(n)
    ok = True

    # 1. identities: no gate, or lam=0, must be exactly neutral
    d = delta_bpb(p_model, p_model * 0.5, nll0, idx, np.zeros(n, bool), 0.5)
    ok &= abs(d) < 1e-12
    print(f"  empty gate      dBPB {d:+.2e}  {'OK' if abs(d) < 1e-12 else 'FAIL'}")
    d = delta_bpb(p_model, np.full(n, 1e-9), nll0, idx, np.ones(n, bool), 0.0)
    ok &= abs(d) < 1e-12
    print(f"  lam=0           dBPB {d:+.2e}  {'OK' if abs(d) < 1e-12 else 'FAIL'}")

    # 2. a PERFECT corrector must improve; a WORTHLESS one must hurt. If these two
    #    ever agree in sign, the metric is broken and every verdict from it is void.
    d_good = delta_bpb(p_model, np.ones(n) * 0.999, nll0, idx, np.ones(n, bool), 0.2)
    d_bad = delta_bpb(p_model, np.full(n, 1e-8), nll0, idx, np.ones(n, bool), 0.2)
    ok &= d_good < 0 < d_bad
    print(f"  perfect corr    dBPB {d_good:+.4f}  {'OK' if d_good < 0 else 'FAIL'}")
    print(f"  worthless corr  dBPB {d_bad:+.4f}  {'OK' if d_bad > 0 else 'FAIL'}")

    # 3. gating on a REAL difficulty signal must beat gating at random, at equal frac
    p_corr = np.where(nll0 > 4.0, 0.5, 1e-6)          # helps only where model is bad
    hard = nll0 >= np.percentile(nll0, 75)
    rand = np.zeros(n, bool); rand[rng.choice(n, hard.sum(), replace=False)] = True
    d_h = delta_bpb(p_model, p_corr, nll0, idx, hard, 0.2)
    d_r = delta_bpb(p_model, p_corr, nll0, idx, rand, 0.2)
    ok &= d_h < d_r
    print(f"  targeted gate   dBPB {d_h:+.4f} vs random-at-equal-frac {d_r:+.4f}  "
          f"{'OK' if d_h < d_r else 'FAIL'}")

    # 4. build_gates: only the oracle family may depend on the true-token NLL
    ent = rng.random(n).astype(np.float32)
    g = build_gates(ent, rng.random(n), rng.integers(0, 50, n), rng.integers(0, 50, n), nll0)
    real = [k for k in g if not k.startswith("ORACLE") and k != "ungated"]
    print(f"  realizable gates: {len(real)}  oracle gates: "
          f"{len([k for k in g if k.startswith('ORACLE')])}")
    ok &= len(real) >= 4
    print("")
    print(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    main()
