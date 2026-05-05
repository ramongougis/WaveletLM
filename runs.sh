#!/bin/bash
# low_rank intermediate ablations: 32, 64.
#
# Prior 1-epoch sweep results at L=1 / levels=7 / bs=16384 baseline (BPB 1.2361):
#   R1  low_rank=16:   1.2342   PASS (-0.0019 vs baseline)
#   R2  low_rank=128:  13.79    NaN-affected mid-run
#   R3  low_rank=1024: NaN at step 2000 (lr=9.12e-3)
#
# This script tests the unmapped region between R1 and R2 to find the
# upper stability boundary. If R1.5 (32) or R1.75 (64) lands within
# the ±0.018 BPB tolerance and beats R1's 1.2342, take it to a 5-epoch
# confirmation; otherwise R1 (low_rank=16) is the winner.

set -euo pipefail

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
        echo "[runs.sh] WARNING: config.json was unexpectedly modified."
        echo "[runs.sh] Reverting config.json before commit (canonical preserved)."
        git checkout -- config.json
    fi
    git add . || true
    git commit --no-edit -m "${MSG}" || true
    git pull --no-edit || true
    git push || true
}

BASE_PATCH='{
    "dataset": "wikitext-103",
    "layers": 1,
    "epochs": 1,
    "mlp_expansion": 10,
    "pkm_enabled": false,
    "fwpkm_num_keys": 8281,
    "tie_embedding_to_lm_head": true,
    "micro_batch_size": 1,
    "grad_accum": 1,
    "block_size": 16384,
    "levels": 7,
    "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
    "wavelet_crawl": false,
    "lifting_diaglowrank": false,
    "lifting_level_sharing": false,
    "lr": 0.01,
    "min_lr": 0.0002,
    "eval_interval": 250
}'

run_ablation() {
    local LABEL="$1"
    local OVERRIDE_JSON="$2"

    echo ""
    echo "============================================================"
    echo "=== Ablation: ${LABEL}"
    echo "============================================================"

    build_run_config "$BASE_PATCH" "$OVERRIDE_JSON"
    python train.py --config "$TMP_CFG"
    git_commit_push "low_rank intermediate ablation: ${LABEL} (1 epoch, L=7)"
}

# R1.5: midpoint between R1 (16, stable) and R2 (128, diverged)
run_ablation "R1.5 low_rank=32" '{"low_rank": 32}'

# R1.75: closer to R2's instability boundary
run_ablation "R1.75 low_rank=64" '{"low_rank": 64}'

echo ""
echo "============================================================"
echo "=== low_rank intermediate ablations complete."
echo "===   Reference: R1 low_rank=16 → BPB sliding 1.2342"
echo "===   If R1.5 or R1.75 beats 1.2342 cleanly: take winner to 5 epochs."
echo "===   If both diverge / fail to beat R1: R1 (low_rank=16) wins."
echo "============================================================"
