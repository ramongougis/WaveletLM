#!/bin/bash
#
# runs_6000.sh — launcher for the RTX PRO 6000 (96 GB), run IN TANDEM with the
# 5090's runs.sh queue. The 6000's queue: (1) C=4096/L=1 width anchor, then
# (2) L=6 depth, then the staged deeper rungs (L=10, L=15, …) for the skip-layer
# deep dive — VRAM-heavy runs the 96 GB swallows, while the 5090 keeps L=5 + the
# 5ep arms. Self-contained (own helper copies) so it never triggers runs.sh.
#
# Tandem-operation notes:
#   - Distinct run-dir timestamps mean the S3 sync is ADDITIVE — the 6000 and the
#     5090 upload their own run folders and never clobber each other.
#   - Both pods push to the same git branch; git_commit_push uses
#     `git pull --no-rebase -X theirs` so concurrent pushes merge loosely (each
#     run dir is distinct, so no real conflicts — just occasional merge commits).
#   - config.json is NEVER modified; the recipe is applied as an override onto a
#     temp config, exactly like runs.sh.
#
# VRAM: C=4096/L=1 (~1.6B params) peaks ~26-29 GB — roomy on the 96 GB 6000
# (it was the "tight" config on the 32 GB 5090). MBS stays 8; no fallback needed.

set -uo pipefail

# Temp config file used for the train.py --config invocation. Auto-deleted on any
# exit path. Canonical config.json is NEVER modified by this script.
TMP_CFG=$(mktemp -t exarch_run_XXXXXX.json)
trap 'rm -f "$TMP_CFG"' EXIT

build_run_config() {
    python -c "
import json, sys
cfg = json.load(open('config.json'))
for patch in sys.argv[1:]:
    cfg.update(json.loads(patch))
json.dump(cfg, open('$TMP_CFG', 'w'), indent=4)
" "$@"
}

git_commit_push() {
    local MSG="$1"
    if ! git diff --quiet config.json 2>/dev/null; then
        echo "[runs_6000.sh] WARNING: config.json was unexpectedly modified."
        echo "[runs_6000.sh] Reverting config.json before commit (canonical preserved)."
        git checkout -- config.json
    fi
    # Stage ONLY run outputs (logs/) — NOT the scripts or README. A blanket
    # `git add .` makes the pod commit runs.sh/runs_6000.sh/README, and with
    # `-X theirs` on pull the pod's copy clobbers your workstation edits (the
    # "local changes would be overwritten" friction). Scripts/README flow one-way:
    # workstation -> GitHub -> pod.
    git add logs/ || true
    git commit --no-edit -m "${MSG}" || true
    git pull --no-rebase --no-edit -X theirs || true
    git push || true

    # Mirror the working tree to S3 after the push so a wiped volume disk loses
    # nothing (logs + best_model.pt checkpoints + .cache tokenized tensors). sync is
    # incremental; hf_cache excluded (large, re-downloadable). Runs even if the push
    # above failed — S3 is the backup of record. NEVER add --delete: the local tree
    # may hold only this run, so --delete would erase every other run from S3.
    aws s3 sync /workspace/EXARCH s3://exarch-ai-model/EXARCH --exclude "hf_cache/*" \
        || echo "[runs_6000.sh] WARNING: aws s3 sync failed — S3 backup NOT updated this run"
}

run_inference_vram_latest() {
    # Fresh-process generate.py passes (standard + --strategies) on the just-trained
    # checkpoint, for inference VRAM + tok/s. Select by NAME (timestamped dir names
    # sort chronologically) so the just-finished run is always picked, regardless of
    # mtime churn from S3 pulls.
    # The EXACT run dir is passed in by run_ablation ($1). Don't guess it — name-sort
    # misfires in tandem (a git pull brings the other pod's newer run into logs/, so
    # generate.py targets the wrong one). $1 is authoritative; name-sort is a fallback.
    local LATEST_RUN="$1"
    if [ -z "$LATEST_RUN" ]; then
        LATEST_RUN=$(ls -d logs/wikitext-103_*/ 2>/dev/null | sort | tail -1)
    fi
    if [ -z "$LATEST_RUN" ]; then
        echo "[runs_6000.sh] Skipping inference VRAM measurement (no log dir found)"
        return
    fi
    LATEST_RUN="${LATEST_RUN%/}"
    if [ ! -f "$LATEST_RUN/best_model.pt" ]; then
        echo "[runs_6000.sh] Skipping inference VRAM measurement (no best_model.pt at $LATEST_RUN)"
        return
    fi
    echo ""
    echo "=== Measuring inference VRAM (fresh process, standard) for ${LATEST_RUN}"
    python generate.py --checkpoint "$LATEST_RUN/best_model.pt" || \
        echo "[runs_6000.sh] generate.py (standard) exited non-zero; continuing"
    echo ""
    echo "=== Measuring inference VRAM (fresh process, --strategies) for ${LATEST_RUN}"
    python generate.py --checkpoint "$LATEST_RUN/best_model.pt" --strategies || \
        echo "[runs_6000.sh] generate.py --strategies exited non-zero; continuing"
}

run_ablation() {
    local LABEL="$1"
    local BASE_JSON="$2"
    local OVERRIDE_JSON="$3"
    local COMMIT_MSG="$4"

    echo ""
    echo "============================================================"
    echo "=== Ablation: ${LABEL}"
    echo "============================================================"

    build_run_config "$BASE_JSON" "$OVERRIDE_JSON"
    # Snapshot run dirs BEFORE train.py so we hand run_inference_vram_latest the EXACT
    # dir it creates (set difference), not a guessed "latest" (misfires in tandem).
    local DIRS_BEFORE; DIRS_BEFORE=$(ls -d logs/wikitext-103_*/ 2>/dev/null)
    python train.py --config "$TMP_CFG"
    local TRAIN_EXIT=$?
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "[runs_6000.sh] train.py exited with code $TRAIN_EXIT; continuing"
    fi
    local RUN_DIR
    if [ -z "$DIRS_BEFORE" ]; then
        RUN_DIR=$(ls -d logs/wikitext-103_*/ 2>/dev/null | tail -1)
    else
        RUN_DIR=$(ls -d logs/wikitext-103_*/ 2>/dev/null | grep -vxF "$DIRS_BEFORE" | tail -1)
    fi
    run_inference_vram_latest "${RUN_DIR%/}"
    git_commit_push "${COMMIT_MSG}"
}

# 1-epoch base — Baseline 3 (B3) defaults; the C=4096 override below replaces the
# recipe-relevant keys (levels, widths, mlp_expansion, lr, crawl, C, ...).
BASE_PATCH_1EP='{
    "dataset": "wikitext-103",
    "layers": 1,
    "epochs": 1,
    "mlp_expansion": 10,
    "pkm_enabled": false,
    "fwpkm_num_keys": 8281,
    "tie_embedding_to_lm_head": true,
    "micro_batch_size": 8,
    "grad_accum": 1,
    "block_size": 256,
    "levels": 5,
    "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
    "wavelet_crawl": false,
    "lifting_diaglowrank": false,
    "lifting_level_sharing": false,
    "low_rank": 4,
    "lifting_offdiag_structure": "none",
    "lr": 0.01,
    "min_lr": 0.0002,
    "eval_interval": 250
}'

# ==============================================================================
# More Width (C) — C=4096, L=1 width-response anchor (1ep) — on the 6000
# First width scale-up point: ~1.6B params, lean no-memory T5 recipe, learned
# embedding (per_layer_embedding on, from config.json). LR inherited at 0.0225
# (no width-LR retune yet — this anchor measures the width response at the
# depth-tuned LR; a width-LR sweep is a separate step). Pairs with the C=2048
# column from More Layers to give the first depth-vs-width efficiency datapoint.
# ==============================================================================

# run_ablation "T5_C4096_L1_1ep More Width — C=4096 L=1 (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1, "C": 4096}'     "T5_C4096_L1_1ep: More Width anchor — C=4096, L=1, no-memory T5 recipe; ~1.6B params; run on RTX PRO 6000"

# ==============================================================================
# Less Width (smaller C) — C=1024 L=1 lr=0.04 (1ep, 6000)
# Compute-allocation probe. Width pays only ~-0.011 BPB for 3.5x params (C=2048->
# 4096); depth pays ~-0.006/layer for +235M (linear). So a LEANER C that stays
# ~equivalent to C=2048 frees compute to spend on DEPTH (which scales cleanly
# through L=5). C=1024 = ~quarter the C=2048 params (~140M), much faster. LR scaled
# UP for the smaller width (0.04 ~ 0.0225 x ~1/width; smaller C tolerates higher LR).
# If 0.04 NaNs, the C=1024 cliff is lower than expected -> drop it. Goal: find the
# "maximally effective C" -- smallest width without material loss vs C=2048 -- then
# stack depth on it. Next rung if promising: C=512 @ lr~0.08.
# ==============================================================================
# run_ablation "T5_C1024_L1_1ep Less Width — C=1024 L=1 lr=0.04 (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.04, "min_lr": 0.0008, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1, "C": 1024}'     "T5_C1024_L1_1ep: Less Width probe — C=1024, L=1, lr=0.04 (LR scaled UP for smaller width); is a leaner C ~equivalent to C=2048? if so, spend the saved compute on depth"

# ---- C=1024: LR=0.05 confirm + ISO-PARAM 3-way (depth vs MLP-width vs model-width)
# C=1024 @ 0.04 DONE = 1.1378 (logs/...18-16-32). Now confirm the LR at 0.05, then ask
# how to best spend a FIXED param budget at C=1024: via DEPTH (more layers) or via
# MLP-WIDTH (bigger mlp_expansion) — each matched to the C=2048/L=1 (455M) and
# C=4096/L=1 (1616M) "model-width" reference points. C=1024 = 97.55M non-MLP +
# 2.096M/expansion + 58.78M/layer → depth L=6~433M / L=26~1609M; MLP E=171~456M /
# E=724~1615M. All at lr=0.05 (the residual stream stays C=1024, so the ~0.062 cliff
# holds even for the fat-MLP runs). Watch the deep L=26 + fat E=724 for NaN.
# Runtime: L=1 ~45min, L=6 / E=171 ~2.3h, L=26 / E=724 ~6-9h; VRAM all fit 96 GB.
run_ablation "T5_C1024_L1_lr05_1ep Less Width — C=1024 L=1 lr=0.05 (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1, "C": 1024}'     "T5_C1024_L1_lr05_1ep: LR confirm — C=1024 L=1 lr=0.05 (vs 0.04=1.1378; expect ~same, 0.04 converged)"

run_ablation "T5_C1024_L6_iso2048d_1ep Less Width — C=1024 L=6 iso-C2048 via DEPTH (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 6, "C": 1024}'     "T5_C1024_L6_iso2048d_1ep: iso-param via DEPTH — C=1024 L=6 (~433M) vs C=2048/L=1 (1.1073)"

run_ablation "T5_C1024_E171_iso2048m_1ep Less Width — C=1024 mlp_exp=171 iso-C2048 via MLP (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 171, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1, "C": 1024}'     "T5_C1024_E171_iso2048m_1ep: iso-param via MLP-width — C=1024 L=1 mlp_exp=171 (~456M) vs C=2048/L=1 (1.1073) and the L=6 depth version"

run_ablation "T5_C1024_L26_iso4096d_1ep Less Width — C=1024 L=26 iso-C4096 via DEPTH (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 26, "C": 1024}'     "T5_C1024_L26_iso4096d_1ep: iso-param via DEPTH — C=1024 L=26 (~1609M) vs C=4096/L=1 (1.0963)"

run_ablation "T5_C1024_E724_iso4096m_1ep Less Width — C=1024 mlp_exp=724 iso-C4096 via MLP (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 724, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1, "C": 1024}'     "T5_C1024_E724_iso4096m_1ep: iso-param via MLP-width — C=1024 L=1 mlp_exp=724 (~1615M) vs C=4096/L=1 (1.0963) and the L=26 depth version"


# ---- C=4096 LR sweep (L=1/1ep) — width-scaled LR tuning ----------------------
# RESULTS: lr=0.0225 DIVERGED (NaN @ lr~0.016; cliff ~0.0155). lr=0.014 = 1.0963
# (-0.0110 vs C=2048), and it CONVERGED — so go HIGHER, not lower (a lower LR would
# under-converge in the 1ep budget). Active point: 0.015, the highest LR demonstrably
# below the ~0.0155 cliff (0.0159 sits ON it and would NaN; the 6% gain isn't worth
# the gamble). 0.014 and 0.01125 commented (done / dropped). min_lr=lr/50. Winner
# transfers to C=8192 (scaled the same way) + the 5ep/PG-19 runs.
# run_ablation "T5_C4096_lr014_1ep More Width — C=4096 L=1 lr=0.014 (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.014, "min_lr": 0.00028, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1, "C": 4096}'     "T5_C4096_lr014_1ep: C=4096 LR sweep — lr=0.014 (just under the ~0.0155 NaN ceiling); 6000"

# run_ablation "T5_C4096_lr01125_1ep More Width — C=4096 L=1 lr=0.01125 (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.01125, "min_lr": 0.000225, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1, "C": 4096}'     "T5_C4096_lr01125_1ep: C=4096 LR sweep — lr=0.01125 (=0.0225/2, 1/width rule); 6000"

run_ablation "T5_C4096_lr015_1ep More Width — C=4096 L=1 lr=0.015 (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.015, "min_lr": 0.0003, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1, "C": 4096}'     "T5_C4096_lr015_1ep: C=4096 LR sweep — lr=0.015 (highest LR below the ~0.0155 NaN cliff; 0.014 converged at 1.0963, this probes a small higher-LR gain); 6000"


# ==============================================================================
# Iterative depth on the 6000 — L=6, then the skip-layer deep dive (L=10, L=15…).
# Runs after C=4096. VRAM at C=2048 (~+4.5 GB/layer): L=6 ~31.3 GB, L=10 ~49 GB,
# L=15 ~72 GB — all comfortable on 96 GB (depth ceiling ~L=19-20). Runs in PARALLEL
# with L=5 on the 5090 (runs.sh), so L=6 isn't gated on L=5 in real time. Uncomment
# each deeper rung as the previous clears the ~0.0010 BPB noise floor.
# ==============================================================================

run_ablation "T5_L6_1ep More Layers — L=6 (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 6}'     "T5_L6_1ep: iterative depth — L=6, no-memory T5 recipe; 6000 (tandem with L=5 on the 5090); ~31.3 GB"

# Staged deeper rungs (skip-layer dive) — uncomment as each clears noise vs the prior:
# run_ablation "T5_L10_1ep More Layers — L=10 (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 10}'     "T5_L10_1ep: iterative depth — L=10, no-memory; 6000; ~49 GB"
# run_ablation "T5_L15_1ep More Layers — L=15 (1ep, 6000)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 15}'     "T5_L15_1ep: iterative depth — L=15, no-memory; 6000; ~72 GB"
