"""wavelet_autopsy.py — Study 5 instrument (plans/interpretability.md).

What did the lifting learn? Extracts the EFFECTIVE analysis filters of the
learned causal lifting transform by impulse-response probing of the actual
module (all nonlinearity, crawl dilations, and cascade effects included),
and reports the classical wavelet invariants, trained vs Haar-init:

  taps      — effective per-level high-pass filter (channel-mean diagonal)
  support   — tap positions with |h| > 1% of max (crawl dilations folded in)
  moments   — normalized m0, m1, m2 of the taps; "vanishing" if |m|/SUM|h|
              < 0.02. Haar kills only m0 (1 vanishing moment); dbN kills N —
              this is the family axis.
  symmetry  — corr(h, reversed h) within support (symlets ~1, Haar-pair ~-1)
  cross-ch  — fraction of response energy OFF the input channel (how far
              beyond scalar filtering the transform went)
  diversity — SVD participation of the (C x W) per-channel tap matrix:
              top-1 energy ~1 => ONE learned mother wavelet broadcast;
              low => a bank of channel-specific wavelets
  nonlin    — ||resp(eps_big) - resp(eps)|| / ||resp(eps)|| on the diagonal:
              how nonlinear the transform is at operating amplitude
  crawl     — learned dilation softmax, top-3 offsets per level

Perfect reconstruction is guaranteed by the lifting structure regardless of
learned P/U (the update step is inverted exactly at reconstruct), so the
learned system is a legitimate (nonlinear, vector-valued) biorthogonal
wavelet family by construction — the question here is only WHICH one.

Probing detail: responses are measured against the zero-input baseline
(P/U MLPs have biases, so f(0) != 0); causal filters respond at t >= t0, so
taps are read forward from the impulse. eps=0.01 probes the linear regime;
eps_big=1.0 probes embedding-scale operating amplitude.

Usage:
  python tools/interpretability/wavelet_autopsy.py \
      --run_dir logs/wikitext-103_2026-07-14_09-11-32
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from model import WaveletLM, get_tokenizer, set_seed  # noqa: E402


def build_model(run_dir, device, load_weights):
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    enc = get_tokenizer(config)
    set_seed(config.get("seed", 1337))  # reproducible init for the baseline
    model = WaveletLM(enc.vocab_size, config, device=device).to(device)
    if load_weights:
        ckpt = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        state = {(k[10:] if k.startswith("_orig_mod.") else k): v
                 for k, v in ckpt.items()}
        model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


@torch.no_grad()
def impulse_responses(lw, C, T, t0, eps, device):
    """Returns per-level (C_in, T, C_out) responses to unit impulses."""
    zero = torch.zeros(1, T, C, device=device)
    _, d0 = lw(zero)
    base = [d[0] for d in d0]                     # (T, C)
    x = torch.zeros(C, T, C, device=device)
    idx = torch.arange(C)
    x[idx, t0, idx] = eps
    _, d = lw(x)
    return [(dl - b.unsqueeze(0)) / eps for dl, b in zip(d, base)]


def analyze(resp_small, resp_big, lw, t0, W, eps_ratio):
    C = resp_small[0].shape[0]
    idx = torch.arange(C)
    rows = []
    for lvl, (rs, rb) in enumerate(zip(resp_small, resp_big)):
        r = rs[:, t0:t0 + W, :]                   # (C_in, W, C_out)
        diag = r[idx, :, idx].cpu().numpy()       # (C, W) per-channel taps
        taps = diag.mean(axis=0)
        total_e = float((r ** 2).sum())
        diag_e = float((diag ** 2).sum())
        cross = 1.0 - diag_e / max(total_e, 1e-30)

        a = np.abs(taps)
        thresh = 0.01 * a.max()
        sup_idx = np.nonzero(a > thresh)[0]
        support = (int(sup_idx[0]), int(sup_idx[-1])) if len(sup_idx) else (0, 0)
        n = np.arange(len(taps), dtype=np.float64)
        l1 = np.abs(taps).sum() + 1e-30
        m = [float((taps * n ** k).sum() / (l1 * max(1.0, (support[1] or 1) ** k)))
             for k in range(3)]
        vanish = sum(1 for k in range(3) if abs(m[k]) < 0.02)

        s0, s1 = support
        seg = taps[s0:s1 + 1]
        sym = float(np.corrcoef(seg, seg[::-1])[0, 1]) if len(seg) > 2 else 1.0

        sv = np.linalg.svd(diag, compute_uv=False)
        part = float(sv[0] ** 2 / (sv ** 2).sum())

        db = rb[idx, t0:t0 + W, idx].cpu().numpy()
        nonlin = float(np.linalg.norm(db - diag) / (np.linalg.norm(diag) + 1e-30))

        if getattr(lw, "wavelet_crawl", False):
            w = torch.softmax(lw.dilation_logits[lvl], dim=0).detach().cpu().numpy()
            offs = lw._crawl_offsets[lvl]
            top = np.argsort(w)[::-1][:3]
            crawl = " ".join(f"d{offs[i]}:{w[i]:.2f}" for i in top)
        else:
            crawl = f"d{1 << lvl}"

        rows.append(dict(level=lvl, taps=taps, diag=diag, support=support,
                         moments=m, vanish=vanish, sym=sym, cross=cross,
                         part=part, nonlin=nonlin, crawl=crawl))
    return rows


def report(tag, rows):
    print(f"\n=== {tag} ===")
    print("lvl | support  | m0     m1     m2    | van | sym   | cross | top1-ch | nonlin | crawl top-3")
    for r in rows:
        print(f" {r['level']}  | {r['support'][0]:>3}-{r['support'][1]:<4} "
              f"| {r['moments'][0]:+.3f} {r['moments'][1]:+.3f} {r['moments'][2]:+.3f} "
              f"| {r['vanish']}   | {r['sym']:+.2f} | {r['cross']:.2f}  "
              f"| {r['part']:.2f}    | {r['nonlin']:.2f}   | {r['crawl']}")
    print("--- channel-mean taps (first 10 significant, as n:h) ---")
    for r in rows:
        a = np.abs(r["taps"])
        sig = np.argsort(a)[::-1][:10]
        sig = sorted(int(i) for i in sig if a[i] > 0.01 * a.max())
        print(f" lvl {r['level']}: " +
              " ".join(f"{i}:{r['taps'][i]:+.3f}" for i in sig))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--t0", type=int, default=48)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--eps_big", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    W = args.T - args.t0 - 8

    results = {}
    for tag, load in (("HAAR-INIT (untrained reference)", False),
                      ("TRAINED", True)):
        model, config = build_model(args.run_dir, device, load_weights=load)
        lw = model.layers[0].lifting_wavelet
        C = config["C"]
        rs = impulse_responses(lw, C, args.T, args.t0, args.eps, device)
        rb = impulse_responses(lw, C, args.T, args.t0, args.eps_big, device)
        rows = analyze(rs, rb, lw, args.t0, W, args.eps_big / args.eps)
        report(tag, rows)
        results[tag] = rows
        del model

    out = args.out or os.path.join(
        ".interp", "autopsy_" + os.path.basename(os.path.normpath(args.run_dir)))
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(
        os.path.join(out, "taps.npz"),
        **{f"{'init' if 'INIT' in tag else 'trained'}_L{r['level']}_diag": r["diag"]
           for tag, rows in results.items() for r in rows})
    print(f"\n[autopsy] per-channel taps saved -> {out}/taps.npz")


if __name__ == "__main__":
    main()
