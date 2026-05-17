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
    git pull --no-rebase --no-edit -X theirs || true
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

benchmark_only_run() {
    # Replay the post-training steps (test benchmark + two generation passes)
    # against an existing run directory's best_model.pt. No training, no model
    # state mutation, no new run dir.
    #
    # train.py is invoked in benchmark_only mode: it reads $TARGET_DIR/config.json
    # as the source of truth for every architectural key, only re-pinning
    # benchmark_only=true and benchmark_run_dir from our temp config. The root
    # config.json is never touched (build_run_config writes only to $TMP_CFG;
    # git_commit_push reverts any unexpected changes), and the run dir's saved
    # config.json is only READ. Downstream consumers of the checkpoint
    # (HF release, future benchmark replays) continue to see the original
    # training config as authoritative.
    local LABEL="$1"
    local TARGET_DIR="$2"
    local COMMIT_MSG="$3"

    echo ""
    echo "============================================================"
    echo "=== Benchmark-only: ${LABEL}"
    echo "===   Target run dir: ${TARGET_DIR}"
    echo "============================================================"

    if [ ! -d "$TARGET_DIR" ]; then
        echo "[runs.sh] ERROR: ${TARGET_DIR} does not exist; skipping benchmark-only run"
        return
    fi
    if [ ! -f "$TARGET_DIR/best_model.pt" ]; then
        echo "[runs.sh] ERROR: ${TARGET_DIR}/best_model.pt missing; skipping"
        return
    fi

    build_run_config "{\"benchmark_only\": true, \"benchmark_run_dir\": \"${TARGET_DIR}\"}"
    python train.py --config "$TMP_CFG"
    local TRAIN_EXIT=$?
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "[runs.sh] train.py (benchmark_only) exited with code $TRAIN_EXIT; continuing"
    fi

    echo ""
    echo "=== Measuring inference VRAM (fresh process, standard) for ${TARGET_DIR}"
    python generate.py --checkpoint "$TARGET_DIR/best_model.pt" || \
        echo "[runs.sh] generate.py (standard) exited non-zero; continuing"
    echo ""
    echo "=== Measuring inference VRAM (fresh process, --strategies) for ${TARGET_DIR}"
    python generate.py --checkpoint "$TARGET_DIR/best_model.pt" --strategies || \
        echo "[runs.sh] generate.py --strategies exited non-zero; continuing"

    git_commit_push "${COMMIT_MSG}"
}


# ---- Mixer recurrence sweep (T3 + N x K) ------------------------------------
# Tests the "mixer-only recurrence" architectural feature: apply the per-scale
# mixer stack N times in a row inside one wavelet+FWHT pipeline before
# reconstruction. Wavelet decompose/reconstruct and FWHT/iFWHT are inverses,
# so the only thing that repeats is the mixer body.

# Three parameters (semantics: nested loops, total mixer applications = N*K):
#   N = mixer_recurrence_steps                — outer loop count
#   K = mixer_recurrence_distinct_mixer_count — distinct per-scale banks per
#                                               cycle. One cycle applies banks
#                                               0..K-1 in sequence; cycle
#                                               repeats N times.
#   mixer_recurrence_residuals (default true) — add X + mixer(X) residual at
#                                               every recurrent step to prevent
#                                               representation collapse. Only
#                                               applied when N*K > 1; baseline
#                                               (N=K=1) behavior unchanged.
# K = 1: one shared bank reused N times (no extra params; pure compute).
# K > 1: K independent banks, sequence repeats N times (K-1 extra mixer
#        banks allocated). Requires mixer_depth == 1.

# All runs build on T3 (= T2 + lr=0.015 + min_lr=0.0003). Reference row in
# the README "Recurrence (Mixer Only)" section is the T3 baseline at N=K=1
# (best val 3.5345). Decision rule: a recurrence variant must clear T3 best
# val (3.5345) by > 0.0015 (noise threshold) to be considered a win.

# Wall-clock estimate (mixer is ~55% of per-block forward+backward at T2):
# total_factor ≈ 1 + (N*K - 1) * 0.55, multiplied by T3 baseline ~1.84h.
#   N=2  K=1  (2 apps):  ~1.55x  = ~2.8h
#   N=2  K=2  (4 apps):  ~2.65x  = ~4.9h
#   N=5  K=1  (5 apps):  ~3.20x  = ~5.9h
#   N=5  K=2  (10 apps): ~5.95x  = ~10.9h
#   N=5  K=5  (25 apps): ~14.2x  = ~26h
#   N=10 K=1  (10 apps): ~5.95x  = ~10.9h
#   N=20 K=1  (20 apps): ~11.45x = ~21h
# K multiplies the mixer-only param subset by K (other params unchanged).
# T2 mixer-only is ~40-80M params, so K=5 adds ~160-320M.

# On A5000 multiply each wall-clock by roughly 1.9x.

# Sweep ordered cost-ascending so the schedule can be cancelled midstream if
# early canaries (N=2) already diverge or plateau.

# # Run R1: N=2, K=1 (shared) — cheapest canary.
# run_ablation "T3_recur_N2_K1_1ep T3 + mixer recurrence N=2 K=1 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 2, "mixer_recurrence_distinct_mixer_count": 1}' \
#     "T3_recur_N2_K1_1ep: T3 + mixer recurrence (N=2 steps, K=1 shared bank; cheapest canary at ~1.55x wall-clock)"

# # Run R2: N=2, K=2 (full distinct at N=2).
# run_ablation "T3_recur_N2_K2_1ep T3 + mixer recurrence N=2 K=2 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 2, "mixer_recurrence_distinct_mixer_count": 2}' \
#     "T3_recur_N2_K2_1ep: T3 + mixer recurrence (N=2 outer x K=2 banks = 4 apps, +1x mixer params vs T3)"

# # Run R3: N=5, K=1 (shared).
# run_ablation "T3_recur_N5_K1_1ep T3 + mixer recurrence N=5 K=1 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 5, "mixer_recurrence_distinct_mixer_count": 1}' \
#     "T3_recur_N5_K1_1ep: T3 + mixer recurrence (N=5 steps, K=1 shared bank; ~3.2x wall-clock)"

# # Run R4: N=5, K=2 (cyclic — 2 banks rotating across 5 steps).
# run_ablation "T3_recur_N5_K2_1ep T3 + mixer recurrence N=5 K=2 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 5, "mixer_recurrence_distinct_mixer_count": 2}' \
#     "T3_recur_N5_K2_1ep: T3 + mixer recurrence (N=5 outer x K=2 banks = 10 apps, cyclic m0,m1 repeated 5x; +1x mixer params)"

# # Run R6: N=10, K=1 (shared, substantive depth).
# run_ablation "T3_recur_N10_K1_1ep T3 + mixer recurrence N=10 K=1 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 10, "mixer_recurrence_distinct_mixer_count": 1}' \
#     "T3_recur_N10_K1_1ep: T3 + mixer recurrence (N=10 steps, K=1 shared bank; ~5.95x wall-clock)"

# # Run R7: N=20, K=1 (shared, deep). Conditional canary for "does it keep
# # climbing?"; if N=10 already plateaus/regresses, skip.
# run_ablation "T3_recur_N20_K1_1ep T3 + mixer recurrence N=20 K=1 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 20, "mixer_recurrence_distinct_mixer_count": 1}' \
#     "T3_recur_N20_K1_1ep: T3 + mixer recurrence (N=20 steps, K=1 shared bank; ~11.45x wall-clock, ~21h)"


# echo ""
# echo "============================================================"
# echo "=== MIXER RECURRENCE QUEUE (T3 + N x K sweep, 1ep each)"
# echo "=== (ALL RUNS COMMENTED OUT — Adagrad unstable at N>=5; pivoting to AdamW)"
# echo "==="
# echo "=== All runs build on T3 (= T2 + lr=0.015). Reference: T3 baseline"
# echo "===   best val 3.5345 (BPB 1.1362) — see New T3 Baseline section in"
# echo "===   README. Decision rule: > 0.0015 nat improvement on best val."
# echo "==="
# echo "=== Queue (cost-ascending on 5090; multiply by ~1.9x for A5000):"
# echo "===   1) T3_recur_N2_K1_1ep    — N=2  K=1  (2 apps,  shared)   ~2.8h"
# echo "===   2) T3_recur_N2_K2_1ep    — N=2  K=2  (4 apps,  distinct) ~4.9h, +1x mixer params"
# echo "===   3) T3_recur_N5_K1_1ep    — N=5  K=1  (5 apps,  shared)   ~5.9h"
# echo "===   4) T3_recur_N5_K2_1ep    — N=5  K=2  (10 apps, cyclic)   ~10.9h, +1x mixer params"
# echo "===   5) T3_recur_N10_K1_1ep   — N=10 K=1  (10 apps, shared)   ~10.9h"
# echo "===   6) T3_recur_N20_K1_1ep   — N=20 K=1  (20 apps, shared)   ~21h (cancel if N=10 plateaus)"
# echo "==="
# echo "=== Total budget if all run on 5090: ~66h (~2.8 days). On A5000: ~125h (~5.2 days)."
# echo "============================================================"


# ---- AdamW LR sweep (T3 base, 1ep each) -------------------------------------
# Adagrad became unstable at N >= 5 recurrence (NaN at step ~8000 for N=5 K=1,
# immediate NaN at step 250 for N=5 K=2). Switching to AdamW for the full
# recurrence sweep. This section finds the best LR before running recurrence.
#
# All other AdamW defaults held constant: betas=(0.9, 0.999), eps=1e-8,
# weight_decay=0.01, amsgrad=False. T3 architecture patch applied to all runs
# (levels=7, T2 mixer widths, wavelet_crawl=true). min_lr = lr/50 throughout.
#
# LRs sample a geometric sequence centred on the AdamW default (0.001) at
# ±1 and ±2 steps of sqrt(10) spacing:
#   A1: lr=0.0001      min_lr=2e-6      (10x below default)
#   A2: lr=0.00031623  min_lr=6.3246e-6 (sqrt(10)x below default)
#   A3: lr=0.001       min_lr=2e-5      (PyTorch AdamW default)
#   A4: lr=0.0031623   min_lr=6.3246e-5 (sqrt(10)x above default)
#   A5: lr=0.01        min_lr=2e-4      (T3/Adagrad LR, 10x above default)

# Run A1: lr=0.0001 (lowest; 10x below AdamW default)
# run_ablation "AdamW_LR0.0001_1ep AdamW LR=0.0001 (T3 base)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.0001, "min_lr": 2e-6}' \
#     "AdamW_LR0.0001_1ep: AdamW LR sweep (lr=0.0001, min_lr=2e-6, T3 base)"

# Run A2: lr=0.00031623 (≈ sqrt(10)/10000; one step above A1)
# run_ablation "AdamW_LR0.00031623_1ep AdamW LR=0.00031623 (T3 base)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00031623, "min_lr": 6.3246e-6, "amp_dtype": "bf16"}' \
#     "AdamW_LR0.00031623_1ep: AdamW LR sweep (lr=0.00031623, min_lr=6.3246e-6, T3 base)"

# Run A3: lr=0.001 (PyTorch AdamW default) — bf16 to avoid fp16 overflow seen in A2
# run_ablation "AdamW_LR0.001_1ep AdamW LR=0.001 (T3 base, bf16)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.001, "min_lr": 2e-5, "amp_dtype": "bf16"}' \
#     "AdamW_LR0.001_1ep: AdamW LR sweep (lr=0.001, min_lr=2e-5, T3 base, bf16)"

# Run A4: lr=0.0031623 (≈ sqrt(10)/1000; one step above default) — bf16, tighter clip
run_ablation "AdamW_LR0.0031623_1ep AdamW LR=0.0031623 (T3 base, bf16, clip=0.5)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.0031623, "min_lr": 6.3246e-5, "amp_dtype": "bf16", "grad_clip": 0.5}' \
    "AdamW_LR0.0031623_1ep: AdamW LR sweep (lr=0.0031623, min_lr=6.3246e-5, T3 base, bf16, grad_clip=0.5)"

# Run A5: lr=0.01 (matches Adagrad T3 LR; upper-bound canary) — bf16, tighter clip
run_ablation "AdamW_LR0.01_1ep AdamW LR=0.01 (T3 base, bf16, clip=0.5)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.01, "min_lr": 2e-4, "amp_dtype": "bf16", "grad_clip": 0.5}' \
    "AdamW_LR0.01_1ep: AdamW LR sweep (lr=0.01, min_lr=2e-4, T3 base, bf16, grad_clip=0.5)"
