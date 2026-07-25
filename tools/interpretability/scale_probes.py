"""scale_probes.py — Study 4: which temporal SCALE encodes what?

The study with no transformer analogue: they have no scale axis to map onto. For
each (layer x scale) cell we fit a cheap linear probe for a linguistic/structural
label and report accuracy, producing a scale x feature heat map.

LABELS ARE FREE. WT-103's own markup and the token stream give us everything
without a single LLM call:
  heading      ' = = Section = = ' markup lines
  boundary     next token begins a new sentence (after . ! ?)
  newline      paragraph break at this position
  numeric      token is/starts a number
  capitalized  token starts with a capital (proper-noun proxy)
  punct        token is punctuation
  rare         token is in the low-frequency tail (corpus-derived)
  subword      token continues a word (no leading space)

CONTROL (mandatory, per Hewitt & Liang's selectivity critique): a probe can score
well simply because the *probe* is expressive, not because the representation
encodes the label. So every probe is paired with a CONTROL probe fit on the same
features against SHUFFLED labels, and we report SELECTIVITY = acc - control_acc.
Only selectivity is interpretable; raw accuracy is not.

Usage:
  python tools/interpretability/scale_probes.py --dump .interp/mini_d3
  python tools/interpretability/scale_probes.py --dump .interp/mini_d3 --labels heading numeric
"""
import argparse, glob, json, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)

LABELS = ["heading", "boundary", "newline", "numeric", "capitalized", "punct", "rare", "subword"]


def build_labels(tokens, enc, which):
    """tokens: (n_windows, T) int -> {name: (n_windows*T,) bool}"""
    flat = tokens.reshape(-1)
    txt = [enc.decode([int(t)]) for t in range(50257)]      # id -> string, once
    out = {}
    dec = np.array([txt[int(t)] for t in flat], dtype=object)
    if "heading" in which:
        out["heading"] = np.array([("=" in d) for d in dec])
    if "newline" in which:
        out["newline"] = np.array([("\n" in d) for d in dec])
    if "numeric" in which:
        out["numeric"] = np.array([d.strip()[:1].isdigit() if d.strip() else False for d in dec])
    if "capitalized" in which:
        out["capitalized"] = np.array([d.strip()[:1].isupper() if d.strip() else False for d in dec])
    if "punct" in which:
        P = set(".,;:!?\"'()[]—-")
        out["punct"] = np.array([bool(d.strip()) and all(c in P for c in d.strip()) for d in dec])
    if "subword" in which:
        out["subword"] = np.array([not d.startswith(" ") and d.strip() != "" for d in dec])
    if "boundary" in which:                                  # PREVIOUS token ended a sentence
        ends = np.array([d.strip() in (".", "!", "?") for d in dec])
        b = np.zeros_like(ends); b[1:] = ends[:-1]
        out["boundary"] = b
    if "rare" in which:
        cnt = np.bincount(flat, minlength=50257)
        rare_ids = set(np.where((cnt > 0) & (cnt <= 2))[0].tolist())
        out["rare"] = np.array([int(t) in rare_ids for t in flat])
    return out


def load_cell(dump, layer, scale):
    sh = sorted(glob.glob(os.path.join(dump, f"shard*_L{layer:02d}_postdecomp.npy")))
    X = np.concatenate([np.load(p)[:, :, scale, :] for p in sh], axis=0)
    return X.reshape(-1, X.shape[-1]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--labels", nargs="+", default=LABELS)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 4, 9])
    p.add_argument("--max_rows", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    man = json.load(open(os.path.join(args.dump, "manifest.json")))
    S = man["S"]
    tok = np.concatenate([np.load(f) for f in
                          sorted(glob.glob(os.path.join(args.dump, "shard*_tokens.npy")))], axis=0)
    Y = build_labels(tok, enc, set(args.labels))
    print(f"[dump] {args.dump}  layers={args.layers}  S={S}")
    for k in args.labels:
        print(f"  label {k:12s} positive rate {Y[k].mean()*100:5.2f}%")
    print("\nreporting SELECTIVITY = acc(real labels) - acc(shuffled labels).")
    print("raw accuracy is uninterpretable on its own (Hewitt & Liang).\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    for name in args.labels:
        y = Y[name]
        if y.sum() < 50 or (~y).sum() < 50:
            print(f"[skip] {name}: too few examples ({int(y.sum())})"); continue
        print(f"=== {name} ===")
        print(f"{'layer':>5} " + " ".join(f"{'s'+str(s):>7}" for s in range(S)))
        for L in args.layers:
            line = []
            for s in range(S):
                X = load_cell(args.dump, L, s)
                n = min(args.max_rows, X.shape[0])
                idx = rng.choice(X.shape[0], n, replace=False)
                Xs, ys = X[idx], y[idx]
                Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.3,
                                                      random_state=args.seed, stratify=ys)
                clf = LogisticRegression(max_iter=300, C=0.1)
                clf.fit(Xtr, ytr); acc = clf.score(Xte, yte)
                yctl = rng.permutation(ytr)          # selectivity control
                ctl = LogisticRegression(max_iter=300, C=0.1)
                ctl.fit(Xtr, yctl); cacc = ctl.score(Xte, yte)
                sel = acc - cacc
                line.append(sel)
                rows.append(dict(label=name, layer=L, scale=s, acc=float(acc),
                                 control=float(cacc), selectivity=float(sel)))
                del X
            print(f"{L:>5} " + " ".join(f"{v:7.3f}" for v in line))
        print()

    if rows:
        print("=== best (layer, scale) per label, by selectivity ===")
        for name in args.labels:
            rs = [r for r in rows if r["label"] == name]
            if not rs: continue
            b = max(rs, key=lambda r: r["selectivity"])
            print(f"  {name:12s} -> L{b['layer']:02d}/s{b['scale']}  "
                  f"selectivity {b['selectivity']:+.3f}  (acc {b['acc']:.3f} vs ctl {b['control']:.3f})")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1); print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
