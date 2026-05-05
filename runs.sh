#!/bin/bash
# Active queue (post-low_rank-sweep). All runs use the L=1 / levels=7 / bs=16384
# baseline; canonical config.json is NEVER modified by this script.
#
# Settled wins so far:
#   - low_rank=16 (R1):                BPB 1.2342, -0.0019 vs reference 1.2361
#   - per_scale_mixer_widths=[0.5×4, 0.25×4] (W2):
#                                      BPB 1.2437, +0.0076 (within ±0.018), -39% mixer
#   - lifting_diaglowrank=False (A1 shelved at +0.0499)
#
# Cancelled:
#   - low_rank=32 (R1.5):  NaN at step 2250, peak lr=1.00e-2
#   - low_rank=64 (R1.75): cancelled (further into unstable region)
#
# Queue:
#   1. DBD                            1ep   decompose_bypass=false + low_rank=16
#   2. E5_5ep                         5ep   per_scale_mixer_widths=[1.5×4, 0.5×4]   (uncompressed coarse expansion confirmation)
#   3. R1_5ep                         5ep   low_rank=16                              (winner confirmation)
#   4. M1..M4                         1ep   off-diagonal magnitude-pruned masking sweep (0.1/1/5/10%)
#   5. M1r..M4r                       1ep   off-diagonal random masking (full controls at matched densities)
#
# After everything completes (likely tomorrow morning), a new combined 5-epoch
# run will fold the surviving winners into a new post-parameter-reduction
# relative baseline.

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

# 1-epoch sweep base (L=1 / levels=7 / bs=16384, post-combined-reduction).
BASE_PATCH_1EP='{
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

# 5-epoch confirmation base (same as 1-epoch except epochs=5).
BASE_PATCH_5EP='{
    "dataset": "wikitext-103",
    "layers": 1,
    "epochs": 5,
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
    local BASE_JSON="$2"
    local OVERRIDE_JSON="$3"
    local COMMIT_MSG="$4"

    echo ""
    echo "============================================================"
    echo "=== Ablation: ${LABEL}"
    echo "============================================================"

    build_run_config "$BASE_JSON" "$OVERRIDE_JSON"
    python train.py --config "$TMP_CFG"
    git_commit_push "${COMMIT_MSG}"
}

# ---- 1. DBD: Decompose Bypass Disablement at 1ep with low_rank=16 -----------
# Validates Future Plans #7 while the 1-epoch baseline is fresh. Boolean
# ablation at L=1/E=1 previously found both flags within ±0.0015 BPB of
# baseline (within noise); this run confirms at L=7 / bs=16384 / 1ep with the
# new low_rank=16 winner. Pass: BPB sliding within ±0.018 of R1's 1.2342.
run_ablation "DBD decompose_bypass=false + low_rank=16" \
    "$BASE_PATCH_1EP" \
    '{"low_rank": 16, "decompose_bypass": false, "decompose_bypass_cross_window": false}' \
    "DBD: decompose_bypass=false + low_rank=16 (1ep, L=7)"

# ---- 2. E5 confirmation: per_scale_mixer_widths=[1.5×4, 0.5×4] @ 5ep ---------
# 5-epoch confirmation of the coarse-expansion config (E5 at 1ep was
# +0.0063 BPB at +24% params — within tolerance but the 5-epoch behavior is
# what determines whether expansion survives). Headline reference at 5ep is
# 1.0974; pass criterion ±0.018 BPB = [1.0794, 1.1154].
run_ablation "E5_5ep per_scale_mixer_widths=[1.5x4,0.5x4] (5 epochs)" \
    "$BASE_PATCH_5EP" \
    '{"per_scale_mixer_widths": [1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5]}' \
    "E5_5ep: per_scale_mixer_widths=[1.5x4,0.5x4] (5ep, L=7)"

# ---- 3. R1 confirmation: low_rank=16 @ 5ep ----------------------------------
# 5-epoch confirmation of the 1-epoch R1 winner. Compared against the headline
# 5-epoch reference of 1.0974; if it beats or ties, low_rank=16 becomes the
# new default. Pass: BPB sliding within ±0.018 of 1.0974 = [1.0794, 1.1154];
# winner is whatever lands cleanly below 1.0974.
run_ablation "R1_5ep low_rank=16 (5 epochs)" \
    "$BASE_PATCH_5EP" \
    '{"low_rank": 16}' \
    "R1_5ep: low_rank=16 (5ep, L=7)"

# ---- 4. Wavelet off-diagonal magnitude-pruned masking sweep (1ep each) ------
# Always uses lifting_diaglowrank=true (mandatory diagonal). Mask is computed
# once at training start from the L=1/levels=7/5-epoch winner's lifting weight
# magnitudes (logs/wikitext-103_2026-05-03_02-13-07/best_model.pt) and frozen
# for both training and inference. low_rank=16 carried forward as the winner.
# Pass criterion vs reference 1.2361: ±0.018 BPB = [1.2181, 1.2541].
OFFDIAG_BASE='{"low_rank": 16, "lifting_diaglowrank": true, "lifting_offdiag_mask": true, "lifting_offdiag_mask_source": "magnitude", "lifting_offdiag_mask_checkpoint": "logs/wikitext-103_2026-05-03_02-13-07/best_model.pt"}'

run_ablation "M1 off-diagonal magnitude 0.1%" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$OFFDIAG_BASE'''); b['lifting_offdiag_density']=0.001; print(json.dumps(b))")" \
    "M1: off-diagonal magnitude-pruned 0.1% (1ep, L=7)"

run_ablation "M2 off-diagonal magnitude 1%" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$OFFDIAG_BASE'''); b['lifting_offdiag_density']=0.01; print(json.dumps(b))")" \
    "M2: off-diagonal magnitude-pruned 1% (1ep, L=7)"

run_ablation "M3 off-diagonal magnitude 5%" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$OFFDIAG_BASE'''); b['lifting_offdiag_density']=0.05; print(json.dumps(b))")" \
    "M3: off-diagonal magnitude-pruned 5% (1ep, L=7)"

run_ablation "M4 off-diagonal magnitude 10%" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$OFFDIAG_BASE'''); b['lifting_offdiag_density']=0.10; print(json.dumps(b))")" \
    "M4: off-diagonal magnitude-pruned 10% (1ep, L=7)"

# ---- 5. Random off-diagonal masking controls (1ep each) ---------------------
# Same densities as M1/M2/M3/M4 with a random mask seed instead of the
# magnitude-pruned ranking. Isolates the deliberate-vs-random contribution at
# each density.
OFFDIAG_RANDOM_BASE='{"low_rank": 16, "lifting_diaglowrank": true, "lifting_offdiag_mask": true, "lifting_offdiag_mask_source": "random", "lifting_offdiag_mask_seed": 1337}'

run_ablation "M1r off-diagonal random 0.1%" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$OFFDIAG_RANDOM_BASE'''); b['lifting_offdiag_density']=0.001; print(json.dumps(b))")" \
    "M1r: off-diagonal random 0.1% (1ep, L=7)"

run_ablation "M2r off-diagonal random 1%" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$OFFDIAG_RANDOM_BASE'''); b['lifting_offdiag_density']=0.01; print(json.dumps(b))")" \
    "M2r: off-diagonal random 1% (1ep, L=7)"

run_ablation "M3r off-diagonal random 5%" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$OFFDIAG_RANDOM_BASE'''); b['lifting_offdiag_density']=0.05; print(json.dumps(b))")" \
    "M3r: off-diagonal random 5% (1ep, L=7)"

run_ablation "M4r off-diagonal random 10%" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$OFFDIAG_RANDOM_BASE'''); b['lifting_offdiag_density']=0.10; print(json.dumps(b))")" \
    "M4r: off-diagonal random 10% (1ep, L=7)"

echo ""
echo "============================================================"
echo "=== Queue complete."
echo "===   1) DBD (1ep)              — pass within ±0.018 of 1.2342"
echo "===   2) E5_5ep widths=[1.5x4]  — pass within ±0.018 of 1.0974"
echo "===   3) R1_5ep low_rank=16     — winner if <= 1.0974"
echo "===   4) M1..M4 magnitude-pruned off-diagonal — pass within ±0.018 of 1.2361"
echo "===   5) M1r/M3r/M4r random controls at matched densities"
echo "==="
echo "=== Next: combine surviving winners into a new 5-epoch baseline."
echo "============================================================"
