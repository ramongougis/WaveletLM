"""gain_probe.py — Study 1's zero-inference companion probe.

The Pile-census prediction (scored CONFIRMED) was that the per-scale gain profile is
**weight-borne**: LayerNorm standardizes each position, so the surviving per-scale
"volume" lives in the affine gamma of `decomp_norms[s]`, not in the data. If true,
the census is readable from the CHECKPOINT ALONE — no forward passes, no dataset,
no dump.

This probe tests that quantitatively and then exploits it. Post-LayerNorm, a
standardized coefficient z has E|z| = sqrt(2/pi) ~ 0.7979 under a Gaussian, so
    predicted mean|coeff| for (layer, scale)  ~  0.7979 * mean|gamma|
Comparing that against a measured census is a falsifiable check on the whole
shortcut. Once validated, ANY checkpoint can be censused in seconds — including
ones never dumped (SP1, the shrinkage arms, future M2/M3).

Usage:
  python tools/interpretability/gain_probe.py --runs logs/wikitext-103_2026-07-15_10-53-46
  python tools/interpretability/gain_probe.py --runs <d2> <d3> <sp1> --compare-layer 0
"""
import argparse, json, math, os, re, sys
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)

E_ABS_GAUSS = math.sqrt(2.0 / math.pi)   # E|z| for standardized z ~ N(0,1)

# Measured census (Finding 1, Mini/D2 post-decomp-norm mean|coeff|) for validation.
CENSUS_D2 = {
    0: [0.622, 0.346, 0.344, 0.338, 0.275, 0.243, 0.359, 0.512],
    9: [0.436, 0.205, 0.227, 0.254, 0.232, 0.213, 0.218, 0.284],
}


def load_sd(run_dir):
    for name in ("best_model.pt", "last_checkpoint.pt"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            obj = torch.load(p, map_location="cpu", weights_only=False)
            sd = obj
            if isinstance(obj, dict):
                for k in ("model_state", "model_state_dict", "model", "state_dict"):
                    if k in obj and isinstance(obj[k], dict):
                        sd = obj[k]; break
            return {k.replace("_orig_mod.", ""): v for k, v in sd.items()
                    if torch.is_tensor(v)}, p
    return None, None


def gains(sd):
    """-> {(layer, scale): (mean|gamma|, std gamma, mean beta)}"""
    pat = re.compile(r"layers\.(\d+)\.decomp_norms\.(\d+)\.(weight|bias)$")
    g, b = {}, {}
    for k, v in sd.items():
        m = pat.match(k)
        if not m:
            continue
        li, si, kind = int(m.group(1)), int(m.group(2)), m.group(3)
        (g if kind == "weight" else b)[(li, si)] = v.float()
    return {key: (val.abs().mean().item(), val.std().item(),
                  b.get(key, torch.zeros(1)).mean().item())
            for key, val in g.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--compare-layer", type=int, default=None,
                   help="validate predicted vs measured census for this layer (D2 only)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    allrows = {}
    for rd in args.runs:
        sd, path = load_sd(rd)
        if sd is None:
            print(f"[skip] no checkpoint in {rd}"); continue
        G = gains(sd)
        if not G:
            print(f"[skip] no decomp_norms in {rd} (wavelet_decomp_norm off?)"); continue
        L = max(k[0] for k in G) + 1
        S = max(k[1] for k in G) + 1
        print(f"\n=== {rd}   ({os.path.basename(path)}, L={L} S={S}) ===")
        print("predicted mean|coeff| = 0.7979 * mean|gamma|   (zero forward passes)")
        print(f"{'layer':>5} " + " ".join(f"{'s'+str(s):>7}" for s in range(S)))
        rows = []
        for li in range(L):
            vals = [E_ABS_GAUSS * G[(li, s)][0] if (li, s) in G else float('nan')
                    for s in range(S)]
            rows.append(vals)
            print(f"{li:>5} " + " ".join(f"{v:7.3f}" for v in vals))
        allrows[rd] = rows

        if args.compare_layer is not None and args.compare_layer in CENSUS_D2:
            li = args.compare_layer
            meas = CENSUS_D2[li]
            pred = rows[li][:len(meas)]
            errs = [abs(p_ - m_) for p_, m_ in zip(pred, meas)]
            print(f"\n  validation vs measured census (D2 layer {li}):")
            print("   scale " + " ".join(f"{'s'+str(s):>7}" for s in range(len(meas))))
            print("   meas  " + " ".join(f"{m_:7.3f}" for m_ in meas))
            print("   pred  " + " ".join(f"{p_:7.3f}" for p_ in pred))
            print("   |err| " + " ".join(f"{e:7.3f}" for e in errs))
            print(f"   mean |err| = {sum(errs)/len(errs):.4f}   max = {max(errs):.4f}")
            # shape agreement matters more than absolute scale
            import statistics as st
            if len(meas) > 2:
                mp, mm = st.mean(pred), st.mean(meas)
                num = sum((a-mp)*(b-mm) for a, b in zip(pred, meas))
                den = (sum((a-mp)**2 for a in pred) * sum((b-mm)**2 for b in meas)) ** 0.5
                print(f"   shape correlation r = {num/den if den else float('nan'):.3f}"
                      f"   (1.0 = the profile is fully weight-borne)")

    if args.out:
        json.dump({k: v for k, v in allrows.items()}, open(args.out, "w"), indent=1)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
