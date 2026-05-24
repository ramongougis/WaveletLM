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

# Run A3: lr=0.001 (PyTorch AdamW default) — bf16, wavelet norms
# run_ablation "AdamW_LR0.001_1ep AdamW LR=0.001 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.001, "min_lr": 2e-5, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.001_1ep: AdamW LR sweep (lr=0.001, min_lr=2e-5, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# # Run A4: lr=0.0031623 (≈ sqrt(10)/1000; one step above default) — bf16, wavelet norms
# run_ablation "AdamW_LR0.0031623_1ep AdamW LR=0.0031623 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.0031623, "min_lr": 6.3246e-5, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.0031623_1ep: AdamW LR sweep (lr=0.0031623, min_lr=6.3246e-5, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# # Run A5: lr=0.01 (matches Adagrad T3 LR; upper-bound canary) — bf16, wavelet norms
# run_ablation "AdamW_LR0.01_1ep AdamW LR=0.01 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.01, "min_lr": 2e-4, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.01_1ep: AdamW LR sweep (lr=0.01, min_lr=2e-4, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- LR recalibration: A1+norms and A2+norms -------------------------------------
# A3+norms trailed the Adagrad T3 baseline after warmup despite strong early
# convergence, suggesting the wavelet norms shift the effective loss landscape
# enough to require LR recalibration. The norms constrain activation magnitude
# into the spectral mixer, which changes gradient scale throughout the block
# relative to the unnormalized runs (A1/A2 without norms). Re-running A1 and A2
# with norms enabled covers the lower end of the LR sweep under the new
# normalized architecture. Original A1/A2 rows preserved in README for comparison.

# Run A1+norms: lr=0.0001 — lower-end LR recalibration, normed architecture
# run_ablation "AdamW_LR0.0001_norms_1ep AdamW LR=0.0001 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.0001, "min_lr": 2e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.0001_norms_1ep: AdamW LR sweep (lr=0.0001, min_lr=2e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# Run A2+norms: lr=0.00031623 — sqrt(10)x above A1+norms
# run_ablation "AdamW_LR0.00031623_norms_1ep AdamW LR=0.00031623 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00031623, "min_lr": 6.3246e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.00031623_norms_1ep: AdamW LR sweep (lr=0.00031623, min_lr=6.3246e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- LR bisection: geometric midpoints between best normed runs ------------------
# Results so far (BPB sliding, 1ep, bf16, wavelet norms, clip=1.0):
#   A1    lr=0.00010000: BPB 1.1554  val 3.5887  Δ+0.054
#   A1.25 lr=0.00013335: BPB 1.1445  val 3.5593  Δ+0.025  — transition (below plateau)
#   A1.5  lr=0.00017783: BPB 1.1393  val 3.5429  Δ+0.008  \  flat plateau
#   A2    lr=0.00031623: BPB 1.1394  val 3.5396  Δ+0.005  /  (essentially identical)
#   A2.5  lr=0.00056234: BPB 1.1428  val 3.5528  Δ+0.018  — transition (above plateau)
#   A3    lr=0.00100000: BPB 1.1729  val 3.7932  Δ+0.259
# Lower cliff: between A1.25 (0.00013335) and A1.5 (0.00017783).
# Upper cliff: between A2 (0.00031623) and A2.5 (0.00056234).
# Round 3: bisect each cliff edge + sample plateau interior.

# DONE — Run A1.5+norms: lr=0.00017783 — bisection midpoint between A1 and A2 normed
# run_ablation "AdamW_LR0.00017783_norms_1ep AdamW LR=0.00017783 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00017783, "min_lr": 3.5566e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.00017783_norms_1ep: AdamW LR bisection (lr=0.00017783, min_lr=3.5566e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# Run A1.25+norms: lr=0.00013335 — bisection midpoint between A1 and A1.5 (below plateau)
# run_ablation "AdamW_LR0.00013335_norms_1ep AdamW LR=0.00013335 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00013335, "min_lr": 2.6670e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.00013335_norms_1ep: AdamW LR bisection (lr=0.00013335, min_lr=2.6670e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# DONE — Run A2.5+norms: lr=0.00056234 — bisection midpoint between A2 and A3 (above plateau)
# run_ablation "AdamW_LR0.00056234_norms_1ep AdamW LR=0.00056234 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00056234, "min_lr": 1.1247e-5, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.00056234_norms_1ep: AdamW LR bisection (lr=0.00056234, min_lr=1.1247e-5, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# Run A1.375+norms: lr=0.00015399 — lower cliff edge (midpoint A1.25↔A1.5, 10^-3.8125)
# run_ablation "AdamW_LR0.00015399_norms_1ep AdamW LR=0.00015399 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00015399, "min_lr": 3.0798e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.00015399_norms_1ep: AdamW LR bisection (lr=0.00015399, min_lr=3.0798e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# # Run A1.75+norms: lr=0.00023714 — plateau interior (midpoint A1.5↔A2, 10^-3.625)
# run_ablation "AdamW_LR0.00023714_norms_1ep AdamW LR=0.00023714 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.00023714_norms_1ep: AdamW LR bisection (lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# DONE — Run A2.25+norms: lr=0.00042170 — upper cliff edge (midpoint A2↔A2.5, 10^-3.375)
# run_ablation "AdamW_LR0.00042170_norms_1ep AdamW LR=0.00042170 (T3 base, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00042170, "min_lr": 8.4340e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "AdamW_LR0.00042170_norms_1ep: AdamW LR bisection (lr=0.00042170, min_lr=8.4340e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- β₂ sweep: optimal lr=0.00023714 (A1.75), baseline β₂=0.999 -----------------
# Baseline already measured: A1.75 row (BPB 1.1393, best val 3.5435).

# Run β₂=0.98: faster second-moment adaptation
# run_ablation "AdamW_B2_0.98_norms_1ep AdamW beta2=0.98 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.9, 0.98]}' \
#     "AdamW_B2_0.98_norms_1ep: AdamW beta2 sweep (beta2=0.98, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# Run β₂=0.99: intermediate second-moment window
# run_ablation "AdamW_B2_0.99_norms_1ep AdamW beta2=0.99 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.9, 0.99]}' \
#     "AdamW_B2_0.99_norms_1ep: AdamW beta2 sweep (beta2=0.99, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# DONE — Run β₂=0.9999: very slow second-moment EMA
# run_ablation "AdamW_B2_0.9999_norms_1ep AdamW beta2=0.9999 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.9, 0.9999]}' \
#     "AdamW_B2_0.9999_norms_1ep: AdamW beta2 sweep (beta2=0.9999, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- β₂ fine sweep: symmetric probes around 0.9999 (±0.00002, ±0.00004) --------
# β₂=0.9999 (BPB 1.1370, val 3.5310) beats β₂=0.999 baseline and Adagrad T3 ref.
# Four probes stepping symmetrically on both sides: 0.99986, 0.99988, 0.99992, 0.99994.

# run_ablation "AdamW_B2_0.99986_norms_1ep AdamW beta2=0.99986 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.9, 0.99986]}' \
#     "AdamW_B2_0.99986_norms_1ep: AdamW beta2 sweep (beta2=0.99986, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "AdamW_B2_0.99988_norms_1ep AdamW beta2=0.99988 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.9, 0.99988]}' \
#     "AdamW_B2_0.99988_norms_1ep: AdamW beta2 sweep (beta2=0.99988, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "AdamW_B2_0.99992_norms_1ep AdamW beta2=0.99992 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.9, 0.99992]}' \
#     "AdamW_B2_0.99992_norms_1ep: AdamW beta2 sweep (beta2=0.99992, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "AdamW_B2_0.99994_norms_1ep AdamW beta2=0.99994 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.9, 0.99994]}' \
#     "AdamW_B2_0.99994_norms_1ep: AdamW beta2 sweep (beta2=0.99994, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- β₁ sweep: β₂ locked at 0.99988 (BPB-optimal), lr=0.00023714 ----------------
# Reference: (0.9, 0.99988) → BPB 1.1365, val 3.5323.
# Probes: β₁=0.0 (no momentum / RMSProp-like), 0.85, 0.875, 0.925, 0.95.

# run_ablation "AdamW_B1_0.0_norms_1ep AdamW beta1=0.0 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.0, 0.99988]}' \
#     "AdamW_B1_0.0_norms_1ep: AdamW beta1 sweep (beta1=0.0, beta2=0.99988, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "AdamW_B1_0.85_norms_1ep AdamW beta1=0.85 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.85, 0.99988]}' \
#     "AdamW_B1_0.85_norms_1ep: AdamW beta1 sweep (beta1=0.85, beta2=0.99988, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "AdamW_B1_0.875_norms_1ep AdamW beta1=0.875 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.875, 0.99988]}' \
#     "AdamW_B1_0.875_norms_1ep: AdamW beta1 sweep (beta1=0.875, beta2=0.99988, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "AdamW_B1_0.925_norms_1ep AdamW beta1=0.925 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.925, 0.99988]}' \
#     "AdamW_B1_0.925_norms_1ep: AdamW beta1 sweep (beta1=0.925, beta2=0.99988, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "AdamW_B1_0.95_norms_1ep AdamW beta1=0.95 (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.95, 0.99988]}' \
#     "AdamW_B1_0.95_norms_1ep: AdamW beta1 sweep (beta1=0.95, beta2=0.99988, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- AMSGrad probe: locked config (lr=0.00023714, β₁=0.9, β₂=0.99988) ----------
# AMSGrad replaces second-moment EMA with running max (v̂_t = max(v̂_{t-1}, v_t)).

# run_ablation "AdamW_amsgrad_norms_1ep AdamW amsgrad=True (T3 base, lr=0.00023714, bf16, wavelet_norms)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "AdamW", "weight_decay": 0.01, "optimizer_eps": 1e-8, "lr": 0.00023714, "min_lr": 4.7428e-6, "amp_dtype": "bf16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "optimizer_betas": [0.9, 0.99988], "optimizer_amsgrad": true}' \
#     "AdamW_amsgrad_norms_1ep: AdamW amsgrad sweep (amsgrad=True, beta1=0.9, beta2=0.99988, lr=0.00023714, min_lr=4.7428e-6, T3 base, bf16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- Adagrad + wavelet norms LR sweep -------------------------------------------
# Mirror of the AdamW normed LR sweep with each LR × 15 (Adagrad/AdamW scale factor).
# Architecture: T3 base (levels=7, wavelet_crawl=true, per_scale_mixer_widths as below).
# All normed rows: fp16, eps=2e-13, weight_decay=1e-6, grad_clip=1.0.
# Ag0 = T3 LR (0.015) with norms enabled — BPB 1.1332, beats T3 (1.1362) and T4 (1.1365). DONE.
# Ag1 = 10× lower (0.0015) — collapses to BPB 1.3340. DONE.
# 15× AdamW conversion hypothesis retired. Now sweeping a log-symmetric grid around 0.015:
# ÷√10 (0.004743), ÷∛10 (0.006963), ×∛10 (0.032316), ×√10 (0.047434), ×10 (0.15).

# run_ablation "Adagrad_Ag0_norms_1ep Adagrad lr=0.015 + wavelet_norms (T3 LR, norms ablation)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.015, "min_lr": 0.0003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "Adagrad_Ag0_norms_1ep: Adagrad LR sweep (lr=0.015, min_lr=0.0003, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "Adagrad_Ag1_norms_1ep Adagrad lr=0.0015 + wavelet_norms (= A1 AdamW × 15)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.0015, "min_lr": 3e-5, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "Adagrad_Ag1_norms_1ep: Adagrad LR sweep (lr=0.0015, min_lr=3e-5, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "Adagrad_AgDivSqrt10_norms_1ep Adagrad lr=0.004743 + wavelet_norms (÷√10 below baseline)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.004743, "min_lr": 9.486e-5, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "Adagrad_AgDivSqrt10_norms_1ep: Adagrad LR sweep (lr=0.004743, min_lr=9.486e-5, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "Adagrad_AgDivCbrt10_norms_1ep Adagrad lr=0.006963 + wavelet_norms (÷∛10 below baseline)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.006963, "min_lr": 1.393e-4, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "Adagrad_AgDivCbrt10_norms_1ep: Adagrad LR sweep (lr=0.006963, min_lr=1.393e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "Adagrad_AgCbrt10_norms_1ep Adagrad lr=0.032316 + wavelet_norms (×∛10 above baseline)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.032316, "min_lr": 6.463e-4, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "Adagrad_AgCbrt10_norms_1ep: Adagrad LR sweep (lr=0.032316, min_lr=6.463e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "Adagrad_AgSqrt10_norms_1ep Adagrad lr=0.047434 + wavelet_norms (×√10 above baseline)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.047434, "min_lr": 9.487e-4, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "Adagrad_AgSqrt10_norms_1ep: Adagrad LR sweep (lr=0.047434, min_lr=9.487e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

# run_ablation "Adagrad_AgX10_norms_1ep Adagrad lr=0.15 + wavelet_norms (×10 above baseline)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.15, "min_lr": 0.003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
#     "Adagrad_AgX10_norms_1ep: Adagrad LR sweep (lr=0.15, min_lr=0.003, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- Fine-grained LR sweep above Ag0 (×1.25, ×1.5, ×1.75) --------------------------------
# Ag0 (lr=0.015) is the current optimum; AgCbrt10 (lr=0.032316, ×∛10) NaN'd.
# Three arithmetic steps probe the gap to bracket where instability begins.

run_ablation "Adagrad_Ag125_norms_1ep Adagrad lr=0.01875 (+25% above Ag0)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.01875, "min_lr": 3.75e-4, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
    "Adagrad_Ag125_norms_1ep: Adagrad fine LR sweep (lr=0.01875, min_lr=3.75e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

run_ablation "Adagrad_Ag150_norms_1ep Adagrad lr=0.02250 (+50% above Ag0)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.02250, "min_lr": 4.50e-4, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
    "Adagrad_Ag150_norms_1ep: Adagrad fine LR sweep (lr=0.02250, min_lr=4.50e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

run_ablation "Adagrad_Ag175_norms_1ep Adagrad lr=0.02625 (+75% above Ag0)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.02625, "min_lr": 5.25e-4, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
    "Adagrad_Ag175_norms_1ep: Adagrad fine LR sweep (lr=0.02625, min_lr=5.25e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- Adagrad parameter sweep (eps, initial_accumulator_value, weight_decay) ---------------
# Base: locked Ag0 config (lr=0.015, min_lr=3e-4, fp16, wavelet norms, T3 arch).
# One parameter varied per run; all others held at Ag0 defaults.
# Order: initial_accumulator_value first (highest variance / memory-motivated), then eps, then weight_decay.

run_ablation "Adagrad_Av0.1_norms_1ep Adagrad initial_accumulator_value=0.1 (Ag0 base)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "optimizer_initial_accumulator_value": 0.1, "lr": 0.015, "min_lr": 0.0003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
    "Adagrad_Av0.1_norms_1ep: Adagrad param sweep (initial_accumulator_value=0.1, lr=0.015, min_lr=3e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

run_ablation "Adagrad_Av1.0_norms_1ep Adagrad initial_accumulator_value=1.0 (Ag0 base)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "optimizer_initial_accumulator_value": 1.0, "lr": 0.015, "min_lr": 0.0003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
    "Adagrad_Av1.0_norms_1ep: Adagrad param sweep (initial_accumulator_value=1.0, lr=0.015, min_lr=3e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

run_ablation "Adagrad_eps1e10_norms_1ep Adagrad eps=1e-10 PyTorch default (Ag0 base)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 1e-10, "lr": 0.015, "min_lr": 0.0003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
    "Adagrad_eps1e10_norms_1ep: Adagrad param sweep (eps=1e-10, lr=0.015, min_lr=3e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

run_ablation "Adagrad_eps1e8_norms_1ep Adagrad eps=1e-8 (Ag0 base)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 1e-8, "lr": 0.015, "min_lr": 0.0003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
    "Adagrad_eps1e8_norms_1ep: Adagrad param sweep (eps=1e-8, lr=0.015, min_lr=3e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

run_ablation "Adagrad_wd0_norms_1ep Adagrad weight_decay=0 (Ag0 base)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 0, "optimizer_eps": 2e-13, "lr": 0.015, "min_lr": 0.0003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
    "Adagrad_wd0_norms_1ep: Adagrad param sweep (weight_decay=0, lr=0.015, min_lr=3e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

run_ablation "Adagrad_wd1e4_norms_1ep Adagrad weight_decay=1e-4 (Ag0 base)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-4, "optimizer_eps": 2e-13, "lr": 0.015, "min_lr": 0.0003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true}' \
    "Adagrad_wd1e4_norms_1ep: Adagrad param sweep (weight_decay=1e-4, lr=0.015, min_lr=3e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm)"

# ---- Spectral Norm ablation (stab_spectral_norm on GatedSpectralMixer) ---------------------
# Base: locked Ag0 config. Parametrizations API spectral_norm constrains mixer σ₁(W) ≤ 1.
# SN1: same LR as Ag0 — measures pure effect of spectral constraint. 
# SN2: LR that NaN'd without SN (×∛10) — tests whether SN rescues it.
# SN3: extreme LR (×10) — stress-test upper stability limit with SN enabled.

run_ablation "Adagrad_SN1_norms_1ep Spectral norm + Ag0 LR (lr=0.015)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.015, "min_lr": 0.0003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "stab_spectral_norm": true}' \
    "Adagrad_SN1_norms_1ep: Spectral norm ablation (lr=0.015, min_lr=3e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm, stab_spectral_norm)"

run_ablation "Adagrad_SN2_norms_1ep Spectral norm + AgCbrt10 LR (lr=0.032316, prev. NaN)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.032316, "min_lr": 6.463e-4, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "stab_spectral_norm": true}' \
    "Adagrad_SN2_norms_1ep: Spectral norm ablation (lr=0.032316, min_lr=6.463e-4, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm, stab_spectral_norm)"

run_ablation "Adagrad_SN3_norms_1ep Spectral norm + extreme LR (lr=0.15)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Adagrad", "weight_decay": 1e-6, "optimizer_eps": 2e-13, "lr": 0.15, "min_lr": 0.003, "amp_dtype": "fp16", "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "stab_spectral_norm": true}' \
    "Adagrad_SN3_norms_1ep: Spectral norm ablation (lr=0.15, min_lr=0.003, T3 base, fp16, wavelet_decomp_norm, wavelet_recon_norm, stab_spectral_norm)"
