#!/bin/bash
#
# runs2.sh — second-VM launcher, identical plumbing to runs.sh (temp-config
# overrides, git+S3 sync, exact run-dir capture, fresh-process inference) but
# carrying ONLY the block-size extension ablations. Run on a SEPARATE 5090 in
# tandem with runs.sh: distinct run-dir timestamps make the S3 sync additive and
# the git pull (-X theirs) merge loosely, so the two VMs never clobber each other
# (same tandem pattern as runs.sh + runs_6000.sh). config.json is NEVER modified.

set -uo pipefail

# Temp config file used for all train.py --config invocations. Auto-deleted
# on any exit path. Canonical config.json is NEVER modified by this script.
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
        echo "[runs2.sh] WARNING: config.json was unexpectedly modified."
        echo "[runs2.sh] Reverting config.json before commit (canonical preserved)."
        git checkout -- config.json
    fi
    # Stage ONLY run outputs (logs/) — NOT the scripts or README. A blanket
    # `git add .` makes the pod commit runs.sh/runs2.sh/README, and with `-X theirs`
    # on pull the pod's copy clobbers your workstation edits. Scripts/README flow
    # one-way: workstation -> GitHub -> pod.
    git add logs/ || true
    git commit --no-edit -m "${MSG}" || true
    git pull --no-rebase --no-edit -X theirs || true
    git push || true

    # Mirror the working tree to S3 after every push so a wiped volume disk loses
    # nothing. sync is incremental; hf_cache excluded. NEVER add --delete: on a fresh
    # volume-disk pod the local tree may hold ONLY the current run, so --delete would
    # erase every other run from the S3 archive.
    aws s3 sync /workspace/EXARCH s3://exarch-ai-model/EXARCH --exclude "hf_cache/*" \
        || echo "[runs2.sh] WARNING: aws s3 sync failed — S3 backup NOT updated this run"
}

# BASE_PATCH_* are the redefined-B3 defaults, byte-identical to runs.sh (see
# runs.sh for the full B3 rationale). block_size=256 here; the BS=4096 runs below
# override it.
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

BASE_PATCH_5EP='{
    "dataset": "wikitext-103",
    "layers": 1,
    "epochs": 5,
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

run_inference_vram_latest() {
    # Fresh-process generate.py passes (standard + --strategies) on the just-trained
    # checkpoint, for inference VRAM + tok/s. The EXACT run dir is passed in by
    # run_ablation ($1); don't guess it (name-sort/mtime misfire in tandem).
    local LATEST_RUN="$1"
    if [ -z "$LATEST_RUN" ]; then
        LATEST_RUN=$(ls -d logs/wikitext-103_*/ 2>/dev/null | sort | tail -1)
    fi
    if [ -z "$LATEST_RUN" ]; then
        echo "[runs2.sh] Skipping inference VRAM measurement (no log dir found)"
        return
    fi
    LATEST_RUN="${LATEST_RUN%/}"
    if [ ! -f "$LATEST_RUN/best_model.pt" ]; then
        echo "[runs2.sh] Skipping inference VRAM measurement (no best_model.pt at $LATEST_RUN)"
        return
    fi
    echo ""
    echo "=== Measuring inference VRAM (fresh process, standard) for ${LATEST_RUN}"
    python generate.py --checkpoint "$LATEST_RUN/best_model.pt" || \
        echo "[runs2.sh] generate.py (standard) exited non-zero; continuing"
    echo ""
    echo "=== Measuring inference VRAM (fresh process, --strategies) for ${LATEST_RUN}"
    python generate.py --checkpoint "$LATEST_RUN/best_model.pt" --strategies || \
        echo "[runs2.sh] generate.py --strategies exited non-zero; continuing"
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
        echo "[runs2.sh] train.py exited with code $TRAIN_EXIT; continuing to next ablation"
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

# ==============================================================================
# Block-Size Extension & Length Generalization (C=1024 / L=5, 5090) — runs2.sh ONLY
# See README "Block-Size Extension & Length Generalization". block_size 256 vs 4096
# only (budget triage). Per-epoch wall-clock is ~flat in block size (fixed WT-103
# token count → fewer, bigger steps), so BS=4096 costs ~the same as BS=256 per epoch.
#   levels = log2(block) - 1 ; per_scale_mixer_widths length = levels + 1.
#   LR = 0.05 (C=1024's 1/C ceiling; context-invariant — NOT lowered for block size).
#   MBS=8 throughout: at BS=4096 the activations are still small (~17 GB est on the
#   5090) — verify at launch, drop MBS only if it OOMs. (MBS=1 is a multi-MILLION-token
#   concern, not a few-thousand-token one.)
# Runs (~34-35h total on a 5090): BS256/5ep (~13.5h) + BS4096/1ep (~3.4h) + BS4096/5ep (~17h).
# ==============================================================================

# (1) BS=256, 5ep — the width-proxy baseline: A/B vs C=2048/L=5/5ep (runs.sh More Epochs Max row).
run_ablation "T5_C1024_L5_bs256_5ep Block-Size — C=1024 L=5 block=256 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 5, "C": 1024}'     "T5_C1024_L5_bs256_5ep: block-size baseline + width proxy — C=1024 L=5 block=256 5ep; A/B vs C=2048/L=5/5ep; ~13.5h on 5090"

# (2) BS=4096, 1ep — context extension, fast signal. levels 11, widths [1x6, 0.5x6] (S=12).
run_ablation "T5_C1024_L5_bs4096_1ep Block-Size — C=1024 L=5 block=4096 (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 11, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "block_size": 4096, "layers": 5, "C": 1024}'     "T5_C1024_L5_bs4096_1ep: block-size extension — C=1024 L=5 block=4096 (levels 11, widths [1x6,0.5x6]) 1ep; MBS=8 ~17GB est, verify at launch; ~3.4h"

# (3) BS=4096, 5ep — context extension at headline scale.
run_ablation "T5_C1024_L5_bs4096_5ep Block-Size — C=1024 L=5 block=4096 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 11, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "block_size": 4096, "layers": 5, "C": 1024}'     "T5_C1024_L5_bs4096_5ep: block-size extension — C=1024 L=5 block=4096 5ep; ~17h on 5090"
