#!/bin/bash

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
        echo "[runs.sh] WARNING: config.json was unexpectedly modified."
        echo "[runs.sh] Reverting config.json before commit (canonical preserved)."
        git checkout -- config.json
    fi
    git add . || true
    git commit --no-edit -m "${MSG}" || true
    git pull --no-edit || true
    git push || true
}

# 1-epoch sweep base — Baseline 3 (B3) defaults.
# B3 = Test 1 + wavelet_crawl=False. Architecturally minimal change from
# Test 1: drop the deprecated wavelet_crawl convolutional component, keep
# everything else. T-lower is intentionally NOT included — it provided no
# real storage/VRAM savings (mask buffer overhead actually slightly INCREASED
# train VRAM, +80 MiB), and Test 1's matched-budget BPB / best val won
# decisively over the bs=16384 NB stack. Mask-based "compression" is no
# longer a production direction; T-lower remains an opt-in stability tool
# (NaN remediation only).
#
# Architecturally:
#   - Test 1 throughput regime: bs=256, MBS=8 (~58,500 steps/epoch vs the
#     bs=16384 stack's ~7,300 at matched epoch budget)
#   - Test 1 structural choices: levels=5, low_rank=4, R0 mixer pattern
#     [1.0×3, 0.5×3] (W2 mixer contraction dropped — per L=9 R0+T-lower
#     finding that W2 costs ~0.0100 BPB at depth)
#   - Test 1 reductions: mlp_expansion=10, pkm_enabled=False,
#     fwpkm_num_keys=8281, tie_embedding_to_lm_head=True
#   - Lifting: lifting_offdiag_structure="none" (dense, unmasked — Test 1
#     default)
#   - Cleanup: wavelet_crawl=False (deprecated convolutional component)
# Historical pre-B3 runs (DBD, M1-M4, BAND/BD/MON sweeps, prior CB/NB-stack
# runs, NB at bs=16384, the earlier T-lower-flavored B3 runs from
# 2026-05-09 19:28 / 21:06) overrode these with their own per-run patches;
# new runs inherit the redefined B3 by default.
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

# 5-epoch confirmation base (same as 1-epoch except epochs=5).
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
    # Locate the most recent run folder by mtime and invoke generate.py on its
    # best checkpoint to get fresh-process inference VRAM measurements. Two
    # passes are run: one standard (naive sampling), one with --strategies
    # enabled. Both append to the run's generations.txt with their own
    # Peak GPU memory line. The strategies-mode pass is also a useful canary
    # for diagnosing strategies-only generation issues (e.g. the levels=9
    # strategies-mode anomaly observed on the NB stack).
    local LATEST_RUN
    LATEST_RUN=$(ls -td logs/wikitext-103_*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "[runs.sh] Skipping inference VRAM measurement (no log dir found)"
        return
    fi
    LATEST_RUN="${LATEST_RUN%/}"
    if [ ! -f "$LATEST_RUN/best_model.pt" ]; then
        echo "[runs.sh] Skipping inference VRAM measurement (no best_model.pt at $LATEST_RUN)"
        return
    fi
    echo ""
    echo "=== Measuring inference VRAM (fresh process, standard) for ${LATEST_RUN}"
    python generate.py --checkpoint "$LATEST_RUN/best_model.pt" || \
        echo "[runs.sh] generate.py (standard) exited non-zero; continuing"
    echo ""
    echo "=== Measuring inference VRAM (fresh process, --strategies) for ${LATEST_RUN}"
    python generate.py --checkpoint "$LATEST_RUN/best_model.pt" --strategies || \
        echo "[runs.sh] generate.py --strategies exited non-zero; continuing"
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
    # Diagnostic: capture train.py's actual exit code so we can SEE what's
    # happening when the queue halts unexpectedly. With `set -e` removed at
    # script top, a non-zero exit no longer halts the queue, but we still
    # want it visible in the log.
    python train.py --config "$TMP_CFG"
    local TRAIN_EXIT=$?
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "[runs.sh] train.py exited with code $TRAIN_EXIT; continuing to next ablation"
    fi
    run_inference_vram_latest
    git_commit_push "${COMMIT_MSG}"
}


# # ---- B3 Baseline (deprecated) ----
# run_ablation "T2_1ep Baseline 3 verification (1ep, redefined: Test 1 + wavelet_crawl=False)" \
#     "$BASE_PATCH_1EP" \
#     '{}' \
#     "B3_1ep: Baseline 3 redefined = Test 1 + wavelet_crawl=False (1ep, bs=256, MBS=8, levels=5, R0 mixer widths; no T-lower)"

# # B3_L7_1ep — same as B3_1ep but at levels=7 with R0 mixer pattern at depth 7
# # ([1.0×4, 0.5×4]). Tests the boundary regime explicitly: bs=256 / 2^7 = 2
# # tokens at the coarsest wavelet scale. If it trains and lands competitively
# # vs B3_1ep (levels=5), the boundary doesn't bind at this regime. If it
# # underperforms, levels=5 is the right choice for the bs=256 throughput
# # regime. Per-width contractions (W2) deprecated — uses R0 mixer widths.
# run_ablation "B3_L7_1ep Baseline 3 at levels=7 (boundary test: 2 tokens at coarsest scale)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]}' \
#     "B3_L7_1ep: Baseline 3 at levels=7 (1ep, bs=256, MBS=8, R0 mixer widths [1.0x4, 0.5x4], no T-lower; boundary case — coarsest scale has only 2 tokens at bs=256)"

# # 5-epoch confirmation of B3 — uses BASE_PATCH_5EP unchanged (B3 is the
# # default). The matched-budget production-decision datapoint vs Test 1 5ep
# # (3.3341 best val).
# run_ablation "B3_5ep Baseline 3 at 5 epochs" \
#     "$BASE_PATCH_5EP" \
#     '{}' \
#     "B3_5ep: Baseline 3 at 5 epochs (bs=256, MBS=8, levels=5, R0 mixer widths, no T-lower; production-decision datapoint vs Test 1 5ep)"


# ---- T1 baseline without wavelet_crawl (1 epoch) ----------------------------
# Confirms the negligible-impact hypothesis: wavelet_crawl is only 15 floats
# at levels=5, so removing it should land within noise of T1. Inherits T1's
# config from BASE_PATCH_1EP (which already has wavelet_crawl=False).
# run_ablation "T1_NoWC_1ep T1 baseline without wavelet_crawl (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{}' \
#     "T1_NoWC_1ep: T1 baseline without wavelet_crawl (1ep, bs=256, MBS=8, levels=5, [1.0x3, 0.5x3] mixer widths)"

# # ---- T2: T1 + levels=7 + 8-entry R0 mixer + no wavelet_crawl ----------------
# # T2 = T1 architecture extended to levels=7 with R0 mixer pattern at depth 7
# # ([1.0x4, 0.5x4]). Tests whether deeper wavelet decomposition is worth the
# # coarsest-scale boundary cost: bs=256 / 2^7 = 2 tokens at the coarsest
# # scale (the same boundary case explored in the deprecated B3_L7).
# run_ablation "T2_1ep New baseline T2 (levels=7, [1.0x4, 0.5x4] mixer widths, no wavelet_crawl) at 1 epoch" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]}' \
#     "T2_1ep: T1 + levels=7 + R0 mixer widths [1.0x4, 0.5x4] + no wavelet_crawl (1ep, bs=256, MBS=8)"

# ---- T2_WC_1ep: T2 + wavelet_crawl=True (1ep) -------------------------------
# T1_NoWC_1ep showed wavelet_crawl removal is load-bearing on T1's leaner
# stack (+0.0083 BPB vs T1, ~5.5x the 0.0015 noise threshold from the 3-seed
# variance study). T2 was already a clear win without crawl (-0.0146 BPB vs
# T1, -0.0229 vs T1_NoWC). This run tests whether stacking crawl on top of T2
# is roughly additive — could land another ~0.0083 BPB lower if so.
run_ablation "T2_WC_1ep T2 with wavelet_crawl=True (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true}' \
    "T2_WC_1ep: T1 + levels=7 + R0 mixer widths [1.0x4, 0.5x4] + wavelet_crawl=True (1ep, bs=256, MBS=8)"

run_ablation "T2_WC_5ep New baseline T2 at 5 epochs (with wavelet_crawl)" \
    "$BASE_PATCH_5EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true}' \
    "T2_WC_5ep: T1 + levels=7 + R0 mixer widths [1.0x4, 0.5x4] + wavelet_crawl=True (5ep, bs=256, MBS=8)"


echo ""
echo "============================================================"
echo "=== Queue complete."
echo "===   1) T2_WC_1ep — T2 + wavelet_crawl=True (1 epoch) — additivity check"
echo "===   2) T2_WC_5ep — T2 + wavelet_crawl=True (5 epochs) — production-decision datapoint"
echo "============================================================"
