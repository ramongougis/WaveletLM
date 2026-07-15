"""coeff_dump.py — Phase 0 instrument of plans/interpretability.md.

Runs a WaveletLM checkpoint forward over N tokens and dumps the per-layer,
per-scale coefficient tensors to disk shards. Every downstream study
(census, monosemanticity, ablation prep, probes) is numpy on these shards —
the model is loaded once, here, and never again.

Shape facts (verified against model.py 2026-07-15, per the plan's no-guessing
rule): the lifting decomposition is UNDECIMATED — every scale keeps all T
positions. Coefficients flow as (B, T, S, Cp) with S = levels+1 scales;
per-scale `decomp_norms[s]` (LayerNorm over Cp; model.py ~2576) sees
(B, T, Cp) — its output is the "post-decompose" capture point; per-scale
`scale_mixers[s]` output (same shape) is the "post-mixer" point. Both fire
exactly once per block forward in production configs (mixer_depth=1, no
recurrence). Scale index 0 is the approximation band per the shrinkage-probe
convention; detail-band ordering is confirmed empirically in Study 1.

Storage: fp16 shards, ~82 KB per token per capture point at C=512/L=10
(→ ~4 GB per 50K tokens). Default captures post-decompose only; add
--capture postmixer for the second point. Token ids are saved per window so
Study 2 can recover the text context of any activation.

Crash-proof by design (laptop rule): windows are processed in shard-sized
batches; a shard whose files all exist is skipped, so any crash resumes
where it left off. CPU by default; --device cuda fits Mini in ~1.3 GB.

Windows are non-overlapping block_size chunks with NO cross-window bypass
carry — matching training's random-window statistics. (A sequential-carry
variant is a Phase-2 instrument.)

Usage (first real dump, from repo root or anywhere):
  python tools/interpretability/coeff_dump.py \
      --run_dir logs/wikitext-103_2026-07-14_09-11-32 \
      --tokens 50000 --out .interp/mini_d2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)  # load_and_encode_dataset uses the repo-relative .cache/

from model import WaveletLM, get_tokenizer  # noqa: E402
from train import load_and_encode_dataset   # noqa: E402


class _PrintLogger:
    def log(self, msg):
        print(msg, flush=True)


def load_model(run_dir, device):
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        config = json.load(f)
    enc = get_tokenizer(config)
    model = WaveletLM(enc.vocab_size, config, device=device).to(device)
    ckpt = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    state = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ckpt.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, enc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Run folder containing best_model.pt + config.json")
    p.add_argument("--out", required=True, help="Output shard directory")
    p.add_argument("--dataset", default=None,
                   help="Override dataset for the dump text (default: the run's own)")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="Which split to dump from (default val)")
    p.add_argument("--tokens", type=int, default=50_000)
    p.add_argument("--capture", nargs="+", default=["postdecomp"],
                   choices=["postdecomp", "postmixer"])
    p.add_argument("--layers", default="all",
                   help="'all' or comma list of layer indices, e.g. 0,4,9")
    p.add_argument("--batch_windows", type=int, default=8)
    p.add_argument("--windows_per_shard", type=int, default=32)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    t0 = time.time()
    print(f"[coeff_dump] Loading model from {args.run_dir} on {device} ...")
    model, config, enc = load_model(args.run_dir, device)

    layers = model.layers
    L = len(layers)
    layer_ids = list(range(L)) if args.layers == "all" else \
        sorted(int(x) for x in args.layers.split(","))
    T = int(config["block_size"])
    S = layers[0].s_effective
    Cp = layers[0].Cp
    if not getattr(layers[0], "wavelet_decomp_norm", False) and "postdecomp" in args.capture:
        raise ValueError("postdecomp capture hooks decomp_norms, which this config disables")

    # Data: reuse the training loader (cached tensors; downloads once if absent).
    data_cfg = dict(config)
    if args.dataset:
        data_cfg["dataset"] = args.dataset
        data_cfg["tokenizer"] = "auto"
    _, val_data, test_data, _, _ = load_and_encode_dataset(data_cfg, _PrintLogger())
    tokens = val_data if args.split == "val" else test_data
    n_windows = min(args.tokens // T, len(tokens) // T)
    if n_windows == 0:
        raise ValueError(f"Not enough tokens for one {T}-token window")
    tokens = tokens[: n_windows * T].view(n_windows, T)
    print(f"[coeff_dump] {n_windows} windows x {T} tokens "
          f"({n_windows * T:,} total), layers {layer_ids}, capture {args.capture}")

    # Hooks: buffers[(point, layer)][scale] = (B, T, Cp) fp32 -> stacked per batch.
    os.makedirs(args.out, exist_ok=True)
    buffers = {}
    hooks = []

    def make_hook(point, li, si):
        def hook(_mod, _inp, out):
            buffers[(point, li)][si] = out.detach().to(torch.float16).cpu()
        return hook

    for li in layer_ids:
        blk = layers[li]
        if "postdecomp" in args.capture:
            for si in range(S):
                hooks.append(blk.decomp_norms[si].register_forward_hook(
                    make_hook("postdecomp", li, si)))
        if "postmixer" in args.capture:
            for si in range(S):
                hooks.append(blk.scale_mixers[si].register_forward_hook(
                    make_hook("postmixer", li, si)))

    manifest = {
        "run_dir": args.run_dir,
        "dataset": data_cfg.get("dataset"),
        "split": args.split,
        "n_windows": n_windows,
        "block_size": T,
        "S": S,
        "Cp": Cp,
        "layers": layer_ids,
        "capture": args.capture,
        "dtype": "float16",
        "shard_windows": args.windows_per_shard,
        "scale_note": "scale 0 = approximation band (shrinkage-probe convention); "
                      "detail ordering confirmed empirically in Study 1",
        "bypass_note": "no cross-window bypass carry (matches random-window training stats)",
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    n_shards = (n_windows + args.windows_per_shard - 1) // args.windows_per_shard
    with torch.no_grad():
        for shard in range(n_shards):
            w0 = shard * args.windows_per_shard
            w1 = min(w0 + args.windows_per_shard, n_windows)
            paths = {(pt, li): os.path.join(
                args.out, f"shard{shard:04d}_L{li:02d}_{pt}.npy")
                for pt in args.capture for li in layer_ids}
            tok_path = os.path.join(args.out, f"shard{shard:04d}_tokens.npy")
            if all(os.path.exists(p) for p in paths.values()) and os.path.exists(tok_path):
                print(f"[coeff_dump] shard {shard}: exists, skipping (resume)")
                continue

            shard_out = {k: [] for k in paths}
            for b0 in range(w0, w1, args.batch_windows):
                b1 = min(b0 + args.batch_windows, w1)
                x = tokens[b0:b1].to(device)
                for key in shard_out:
                    buffers[key] = [None] * S
                model(x, targets=None)
                for (pt, li), scales in list(buffers.items()):
                    if any(s is None for s in scales):
                        raise RuntimeError(
                            f"capture {pt} layer {li}: hook did not fire for all scales")
                    shard_out[(pt, li)].append(torch.stack(scales, dim=2))  # (B,T,S,Cp)

            np.save(tok_path, tokens[w0:w1].numpy().astype(np.int32))
            for key, chunks in shard_out.items():
                np.save(paths[key], torch.cat(chunks, dim=0).numpy())
            done = sum(1 for s in range(n_shards) if os.path.exists(
                os.path.join(args.out, f"shard{s:04d}_tokens.npy")))
            print(f"[coeff_dump] shard {shard + 1}/{n_shards} written "
                  f"({done}/{n_shards} on disk, {time.time() - t0:.0f}s elapsed)")

    for h in hooks:
        h.remove()

    # Mini-census: instant sanity + the first real look at the coefficients.
    pt = args.capture[0]
    for li in (layer_ids[0], layer_ids[-1]):
        arr = np.load(os.path.join(args.out, f"shard0000_L{li:02d}_{pt}.npy"))
        mags = np.abs(arr.astype(np.float32)).mean(axis=(0, 1, 3))  # per-scale mean |x|
        print(f"[census] layer {li} {pt} per-scale mean|coeff|: "
              + " ".join(f"s{s}={m:.3f}" for s, m in enumerate(mags))
              + f"  (finite: {np.isfinite(arr).all()})")
    print(f"[coeff_dump] DONE in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
