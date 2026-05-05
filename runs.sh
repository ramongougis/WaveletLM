#!/bin/bash
# Mixer ablations: width contractions (W1, W2) + low_rank sweep (R1, R2, R3).
#
# IMPORTANT: this script does NOT mutate config.json. Each ablation builds
# a temporary config file by merging the canonical config.json with patch
# JSON (BASE_PATCH + per-ablation override) and passes it to train.py via
# --config. The temp file is auto-deleted on EXIT (including Ctrl-C, NaN
# kill, OOM crash). Canonical config.json stays untouched, eliminating
# the cross-machine config-pollution bug from earlier sweeps.
#
# Reference: 1-epoch L=7 / layers=1 / post-combined-reduction baseline:
#   logs/wikitext-103_2026-05-02_21-43-22 → BPB sliding 1.2361, 392.91M params
#
# Width contractions (per_scale_mixer_widths):
#   W1: [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]   (extreme)
#   W2: [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]   (half of current)
#
# low_rank sweep (U/V parallel correction inside GatedSpectralMixer):
#   R0 (baseline): low_rank=4    → +16K per scale, +128K total       — already at 1.2361
#   R1:            low_rank=16   → +256K per scale, +2.05M total
#   R2:            low_rank=128  → +2.10M per scale, +16.78M total
#   R3:            low_rank=1024 → +16.78M per scale, +134M total
#                                  (~equals main mixer matrix capacity)
#
# Decision rule: if any pass +0.018 BPB tolerance vs 1.2361, take the
# lowest-BPB to 5 epochs. If R3 wins, also test low_rank=2048 to verify
# saturating-at-full-rank behavior.

set -euo pipefail

# Temp file used for all train.py --config invocations. Auto-deleted on
# any exit path (success, error, Ctrl-C, signal). Canonical config.json
# is NEVER modified by this script.
TMP_CFG=$(mktemp -t exarch_run_XXXXXX.json)
trap 'rm -f "$TMP_CFG"' EXIT

build_run_config() {
    # Merge canonical config.json with all provided JSON patches in order
    # into $TMP_CFG. Args: each arg is a JSON string. Later patches override
    # earlier ones (standard dict.update semantics).
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
    # Defensive guard: refuse to commit changes to config.json. Canonical
    # config.json should never be touched during a sweep — if some future
    # script mutates it accidentally, this revert keeps the canonical
    # version intact in git.
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
    git_commit_push "Mixer ablation: ${LABEL} (1 epoch, L=7, ref BPB 1.2361)"
}

# W1: extreme width contraction — 0.1 coarse / 0.05 fine
run_ablation "W1 widths=[0.1,0.1,0.1,0.1,.05,.05,.05,.05]" \
    '{"per_scale_mixer_widths": [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]}'

# W2: half of current default — 0.5 coarse / 0.25 fine
run_ablation "W2 widths=[0.5,0.5,0.5,0.5,.25,.25,.25,.25]" \
    '{"per_scale_mixer_widths": [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]}'

# R1: low_rank modest bump
run_ablation "R1 low_rank=16" '{"low_rank": 16}'

# R2: low_rank substantial mid-point
run_ablation "R2 low_rank=128" '{"low_rank": 128}'

# R3: low_rank matched to main mixer matrix capacity
run_ablation "R3 low_rank=1024" '{"low_rank": 1024}'

echo ""
echo "============================================================"
echo "=== Ablations complete."
echo "===   Reference: 1-epoch L=7 BPB sliding 1.2361"
echo "===   Tolerance: ±0.018 → pass range [1.2181, 1.2541]"
echo "===   Take lowest-BPB winner (if any pass) to a 5-epoch confirmation."
echo "============================================================"
