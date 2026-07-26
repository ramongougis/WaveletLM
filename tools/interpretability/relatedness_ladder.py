"""relatedness_ladder.py — Study 4 v2: does the representation encode SEMANTIC
CATEGORY of the upcoming token, and at which scale?

WHY v1 FAILED (Finding 18). Every label was a deterministic function of the CURRENT
token (capitalized, numeric, punctuation...). Any representation that encodes token
identity — which all of them must, to predict the next token — decodes those
trivially. Result: 0.996-1.000 accuracy in every (layer, scale) cell, and
"selectivity" merely restating the class balance. The map was uniform and meaningless.

THE FIX: make the label PREDICTIVE, not descriptive.
    y[t] = 1  iff  token[t+1] is a member of set X
Probing position t for a property of position t+1 cannot be solved by reading off
the current token. It requires the representation to encode *what kind of word is
coming*, which is the semantic structure we want to measure.

THE LADDER (Ramon's design, 2026-07-26). Instead of one control, use a DOSE-RESPONSE.
Build sets of n tokens containing k semantically related members and (n-k) random
ones, for k = 0, 2, 3, ... A monotonic rise in decodability with k is far harder to
produce by artifact than any single significant contrast, and the k=0 rung IS the
null — so the control comes free with the treatment.

NON-CIRCULARITY. Related sets are drawn from HAND-CURATED categories, never from the
model's own embedding space. Using embedding neighbours would guarantee the result
(the probe would rediscover the geometry we used to build the labels).

METRIC: AUC, which is invariant to class imbalance — v1's raw accuracy was
uninterpretable because positive rates ranged 1.4%-15.7% across labels.

HONEST SCOPE: a rise with k shows the representation linearly encodes upcoming-token
category. It does NOT separate "the model learned semantics" from "language makes
coherent categories predictable" — the model must encode it either way to predict
well. State that; don't claim more.

Usage:
  python tools/interpretability/relatedness_ladder.py --dump .interp/mini_d3
"""
import argparse, glob, json, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)

# Hand-curated categories (never from the model's embedding — see NON-CIRCULARITY).
CATEGORIES = {
    "months":    [" January"," February"," March"," April"," May"," June"," July",
                  " August"," September"," October"," November"," December"],
    "days":      [" Monday"," Tuesday"," Wednesday"," Thursday"," Friday"," Saturday"," Sunday"],
    "numbers":   [" one"," two"," three"," four"," five"," six"," seven"," eight"," nine"," ten"],
    "colors":    [" red"," blue"," green"," black"," white"," yellow"," brown"," grey"," gray"],
    "countries": [" England"," France"," Germany"," Japan"," China"," Russia"," Italy",
                  " Spain"," India"," Canada"," Australia"," Brazil"],
    "family":    [" father"," mother"," brother"," sister"," son"," daughter"," wife"," husband"],
    "time":      [" year"," month"," week"," day"," hour"," century"," decade"," season"],
    "military":  [" army"," navy"," troops"," soldiers"," battle"," war"," regiment"," artillery"],
}


def build_sets(enc, rng, k, n, n_sets, freq, cat_ids):
    """n_sets sets of n token-ids: k members from ONE category + (n-k) FREQUENCY-MATCHED
    random tokens.

    FREQUENCY MATCHING IS ESSENTIAL (learned the hard way, 2026-07-26). A naively random
    set routinely contains a COMMON token, and "a common token is coming" is a highly
    learnable linguistic task with nothing to do with semantics. Unmatched, the k=0 null
    scored AUC 0.87-0.96 instead of ~0.5, and set composition swamped the relatedness
    signal entirely (positive rates swung 0.0011-0.0034 across rungs). Matching each
    filler to a related token's frequency decile makes coherence the ONLY difference
    between rungs."""
    order = np.argsort(freq)                     # ascending frequency
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    out = []
    cats = list(cat_ids)
    for _ in range(n_sets):
        cat = cats[rng.integers(len(cats))]
        pool_ids = cat_ids[cat]
        take = rng.choice(len(pool_ids), min(max(k, 1), len(pool_ids)), replace=False)
        anchors = [pool_ids[i] for i in take]     # defines the frequency profile
        ids = list(anchors[:k])                   # k=0 keeps none, but profile still used
        # fillers: match each remaining slot to an anchor's frequency neighbourhood
        need = n - len(ids)
        for j in range(need):
            a = anchors[j % len(anchors)]
            r = rank[a]
            lo, hi = max(0, r - 500), min(len(order), r + 500)
            for _try in range(20):
                cand = int(order[rng.integers(lo, hi)])
                if cand not in ids and freq[cand] > 0:
                    ids.append(cand); break
            else:
                ids.append(int(order[rng.integers(lo, hi)]))
        out.append(set(ids[:n]))
    return out


def load_cell(dump, layer, scale):
    sh = sorted(glob.glob(os.path.join(dump, f"shard*_L{layer:02d}_postdecomp.npy")))
    X = np.concatenate([np.load(p)[:, :, scale, :] for p in sh], axis=0)   # (win,T,Cp)
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 9])
    p.add_argument("--scales", type=int, nargs="+", default=[0, 1, 4, 7])
    p.add_argument("--levels", type=int, nargs="+", default=[0, 2, 3, 4, 6, 8],
                   help="k = number of RELATED members per set; k=0 is the null rung")
    p.add_argument("--set_size", type=int, default=8)
    p.add_argument("--n_sets", type=int, default=8, help="sets per rung (averaged)")
    p.add_argument("--max_rows", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    man = json.load(open(os.path.join(args.dump, "manifest.json")))
    tok = np.concatenate([np.load(f) for f in
                          sorted(glob.glob(os.path.join(args.dump, "shard*_tokens.npy")))], axis=0)
    n_win, T = tok.shape
    # PREDICTIVE label: position t is labelled by token[t+1]. Drop each window's last position.
    nxt = tok[:, 1:].reshape(-1)                 # (n_win*(T-1),)
    keep_pos = np.arange(n_win * T).reshape(n_win, T)[:, :-1].reshape(-1)
    rng = np.random.default_rng(args.seed)
    freq = np.bincount(tok.reshape(-1), minlength=50257).astype(np.float64)
    cat_ids = {}
    for c, words in CATEGORIES.items():
        ids = [enc.encode(w)[0] for w in words if len(enc.encode(w)) == 1 and freq[enc.encode(w)[0]] > 0]
        if len(ids) >= 4:
            cat_ids[c] = ids
    print(f"[categories] usable: {', '.join(f'{c}({len(v)})' for c, v in cat_ids.items())}")

    print(f"[dump] {args.dump}  layers={args.layers} scales={args.scales}")
    print(f"[design] predictive label y[t] = token[t+1] in set; set size {args.set_size}, "
          f"{args.n_sets} sets/rung, AUC metric")
    print(f"[ladder] k (related members) = {args.levels};  k=0 is the null\n")

    # cache cells once
    cells = {}
    for L in args.layers:
        Xl = load_cell(args.dump, L, args.scales[0])   # shape probe
        for s in args.scales:
            X = load_cell(args.dump, L, s)
            cells[(L, s)] = X.reshape(-1, X.shape[-1])[keep_pos].astype(np.float32)
        del Xl

    rows = []
    print(f"{'k':>3} {'pos-rate':>9} " + " ".join(f"L{L:02d}/s{s}" for L in args.layers
                                                  for s in args.scales))
    for k in args.levels:
        sets = build_sets(enc, rng, k, args.set_size, args.n_sets, freq, cat_ids)
        aucs = {(L, s): [] for L in args.layers for s in args.scales}
        prate = []
        for st in sets:
            y = np.isin(nxt, list(st))
            if y.sum() < 40:
                continue
            prate.append(y.mean())
            idx = rng.choice(len(y), min(args.max_rows, len(y)), replace=False)
            ys = y[idx]
            if ys.sum() < 20 or (~ys).sum() < 20:
                continue
            for (L, s), Xf in cells.items():
                Xtr, Xte, ytr, yte = train_test_split(Xf[idx], ys, test_size=0.3,
                                                      random_state=args.seed, stratify=ys)
                clf = LogisticRegression(max_iter=200, C=0.05)
                clf.fit(Xtr, ytr)
                aucs[(L, s)].append(roc_auc_score(yte, clf.decision_function(Xte)))
        line = []
        for L in args.layers:
            for s in args.scales:
                v = float(np.mean(aucs[(L, s)])) if aucs[(L, s)] else float("nan")
                line.append(v)
                rows.append(dict(k=k, layer=L, scale=s, auc=v))
        print(f"{k:>3} {np.mean(prate) if prate else float('nan'):9.4f} "
              + " ".join(f"{v:7.3f}" for v in line), flush=True)

    print("\n0.500 = chance. A MONOTONIC rise with k => the representation encodes")
    print("upcoming-token semantic category; k=0 is the built-in null.")
    if rows:
        for L in args.layers:
            for s in args.scales:
                v = [r["auc"] for r in rows if r["layer"] == L and r["scale"] == s]
                if len(v) > 2 and not any(np.isnan(v)):
                    kk = np.array(args.levels[:len(v)], dtype=float)
                    r = np.corrcoef(kk, v)[0, 1]
                    print(f"  L{L:02d}/s{s}: AUC-vs-k correlation {r:+.3f}"
                          f"   (k=0 {v[0]:.3f} -> k={args.levels[len(v)-1]} {v[-1]:.3f})")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1); print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
