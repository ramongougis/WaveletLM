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


# ---- Mixer recurrence sweep (T4 + N x K) ------------------------------------
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

# All runs build on T4 (= T3 + wavelet_decomp_norm + wavelet_recon_norm
# + lr=0.02250 + min_lr=4.50e-4, Adagrad, eps=2e-13, weight_decay=1e-6).
# T4 supersedes T3 here: the earlier Adagrad recurrence (without wavelet
# norms) NaN'd at N >= 5; wavelet norms restored stability and the tuned LR
# gives a better baseline (T4: BPB 1.1311 vs T3: 1.1362). Reference row in
# the README "Recurrence with Best Optimizer (Adagrad)" section is the T4
# baseline at N=K=1 (best val 3.5157). Decision rule: a recurrence variant
# must clear T4 best val (3.5157) by > 0.0015 (noise threshold) to win.

# Wall-clock estimate on A5000 (mixer is ~55% of per-block forward+backward at
# T4 architecture): total_factor ~= 1 + (N*K - 1) * 0.55, multiplied by T4
# baseline of ~4.78h on A5000.
#   N=2  K=1  (2 apps):  ~1.55x  = ~7.4h
#   N=2  K=2  (4 apps):  ~2.65x  = ~12.7h
#   N=5  K=1  (5 apps):  ~3.20x  = ~15.3h
#   N=5  K=2  (10 apps): ~5.95x  = ~28.4h
#   N=10 K=1  (10 apps): ~5.95x  = ~28.4h
#   N=20 K=1  (20 apps): ~11.45x = ~54.7h
# K multiplies the mixer-only param subset by K (other params unchanged).
# T4 mixer-only is ~58.8M params, so K=2 adds ~58.8M.

# Sweep ordered cost-ascending so the schedule can be cancelled midstream if
# early canaries (N=2) already diverge or plateau.

# Common T4 overrides (added to every per-run patch below):
#   levels=7, per_scale_mixer_widths=T4 layout, wavelet_crawl=true,
#   wavelet_decomp_norm=true, wavelet_recon_norm=true,
#   lr=0.02250, min_lr=4.50e-4

# === WITH-RESIDUAL (input-anchored) sweep ===================================
# These re-run the recurrence configs under the corrected residual: the
# initial post-FWHT spectrum X0 is re-injected at every mixer step's input
# (model.py — folded into mixer_recurrence_residuals, which is already true),
#     X^t = LN( (X^{t-1} + X0) + m(X^{t-1} + X0) )
# anchoring the iteration to the input instead of letting a shared-weight
# residual stack drift. The earlier (no-residual) sweep showed depth N>2
# REGRESSES; the hypothesis here is that input anchoring rescues depth, so
# N=5/10/20 are back in scope (not cancelled). Renamed _resid so logs don't
# conflate with the historical no-injection runs. mixer_recurrence_residuals
# set explicitly (= true) to mark intent though it is the config default.

# Run F0: N=1, K=2 (diversity at depth 1) — re-run with input injection.
# run_ablation "T4_recur_resid_N1_K2_1ep T4 recurrence N=1 K=2 + input-anchored residual (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 1, "mixer_recurrence_distinct_mixer_count": 2, "mixer_recurrence_residuals": true}' \
#     "T4_recur_resid_N1_K2_1ep: N=1 K=2 with input-anchored residual; vs no-residual N=1 K=2 (1.1249) (~5.7h)"

# Run F1: N=2, K=1 (shared) — re-run with input injection.
# run_ablation "T4_recur_resid_N2_K1_1ep T4 recurrence N=2 K=1 + input-anchored residual (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 2, "mixer_recurrence_distinct_mixer_count": 1, "mixer_recurrence_residuals": true}' \
#     "T4_recur_resid_N2_K1_1ep: N=2 K=1 with input-anchored residual; vs no-residual N=2 K=1 (1.1279) (~5.5h)"

# # Run F2: N=2, K=2 — re-run with input injection.
# run_ablation "T4_recur_resid_N2_K2_1ep T4 recurrence N=2 K=2 + input-anchored residual (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 2, "mixer_recurrence_distinct_mixer_count": 2, "mixer_recurrence_residuals": true}' \
#     "T4_recur_resid_N2_K2_1ep: N=2 K=2 with input-anchored residual; vs no-residual N=2 K=2 (1.1227, current best) (~6.8h)"

# Run F3: N=5, K=1 — depth-rescue test. No-residual N=5 K=1 regressed to 1.1291
# (worse than N=2 K=1); does input anchoring let depth help instead of hurt?
# run_ablation "T4_recur_resid_N5_K1_1ep T4 recurrence N=5 K=1 + input-anchored residual (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 5, "mixer_recurrence_distinct_mixer_count": 1, "mixer_recurrence_residuals": true}' \
#     "T4_recur_resid_N5_K1_1ep: N=5 K=1 with input-anchored residual; depth-rescue test vs no-residual N=5 K=1 (1.1291) (~7.3h)"

# Run F4: N=5, K=2 (cyclic) — re-run with input injection.
# run_ablation "T4_recur_resid_N5_K2_1ep T4 recurrence N=5 K=2 + input-anchored residual (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 5, "mixer_recurrence_distinct_mixer_count": 2, "mixer_recurrence_residuals": true}' \
#     "T4_recur_resid_N5_K2_1ep: N=5 K=2 with input-anchored residual; vs no-residual N=5 K=2 (1.1275) (~10.4h)"

# # Run F5: N=10, K=1 — deep-rescue probe. Only worth it if F3 (N=5) shows depth
# # now helps under anchoring; otherwise cancel.
# run_ablation "T4_recur_resid_N10_K1_1ep T4 recurrence N=10 K=1 + input-anchored residual (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 10, "mixer_recurrence_distinct_mixer_count": 1, "mixer_recurrence_residuals": true}' \
#     "T4_recur_resid_N10_K1_1ep: N=10 K=1 with input-anchored residual; deep-rescue probe (~9.9h, cancel if N=5 anchored still regresses)"

# # Run F6: N=20, K=1 — deepest probe. Only if F5 keeps improving.
# run_ablation "T4_recur_resid_N20_K1_1ep T4 recurrence N=20 K=1 + input-anchored residual (1ep)" \
#     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 20, "mixer_recurrence_distinct_mixer_count": 1, "mixer_recurrence_residuals": true}' \
# #     "T4_recur_resid_N20_K1_1ep: N=20 K=1 with input-anchored residual; deepest probe (~15.7h, cancel if N=10 anchored plateaus)"


# echo ""
# echo "============================================================"
# echo "=== MIXER RECURRENCE QUEUE — WITH INPUT-ANCHORED RESIDUAL (1ep each)"
# echo "==="
# echo "=== Re-runs the N x K sweep under the corrected residual that"
# echo "===   re-injects X0 at every mixer step. Tests whether input"
# echo "===   anchoring rescues depth (no-residual sweep regressed past N=2)."
# echo "===   Reference: T4 baseline best val 3.5157 (BPB 1.1311)."
# echo "===   Compare each to its no-residual twin in the README."
# echo "==="
# echo "=== Queue (cost-ascending on A5000):"
# echo "===   1) T4_recur_resid_N1_K2_1ep   — N=1  K=2  (2 apps)   ~5.7h"
# echo "===   2) T4_recur_resid_N2_K1_1ep   — N=2  K=1  (2 apps)   ~5.5h"
# echo "===   3) T4_recur_resid_N2_K2_1ep   — N=2  K=2  (4 apps)   ~6.8h"
# echo "===   4) T4_recur_resid_N5_K1_1ep   — N=5  K=1  (5 apps)   ~7.3h  [depth-rescue test]"
# echo "===   5) T4_recur_resid_N5_K2_1ep   — N=5  K=2  (10 apps)  ~10.4h"
# echo "===   6) T4_recur_resid_N10_K1_1ep  — N=10 K=1  (10 apps)  ~9.9h  [cancel if N=5 regresses]"
# echo "===   7) T4_recur_resid_N20_K1_1ep  — N=20 K=1  (20 apps)  ~15.7h [cancel if N=10 plateaus]"
# echo "============================================================"


# # ---- Recurrence efficiency: gate caching ------------------------------------
# # Approximation that computes the cross-scale gate once on the first recurrence
# # cycle and reuses it for cycles 2..N (mixer_recurrence_cache_gate=true),
# # eliminating the W_gate matmul + routing einsum on all but the first cycle.
# # Roughly halves per-step matmul cost at K=1.
# #
# # Baseline for comparison is the queued T4_recur_N5_K1_1ep (cache off). This
# # run is identical except cache_gate=true — isolates the approximation's
# # quality cost and runtime saving. Decision: if best val stays within 0.0015
# # of the no-cache N=5 K=1 run AND wall-clock drops meaningfully, caching is a
# # free win and should be enabled for deeper-N runs.

# # run_ablation "T4_recur_N5_K1_cachegate_1ep T4 + recurrence N=5 K=1 + gate cache (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 5, "mixer_recurrence_distinct_mixer_count": 1, "mixer_recurrence_cache_gate": true}' \
# #     "T4_recur_N5_K1_cachegate_1ep: gate cached after cycle 1, reused for cycles 2-5; vs T4_recur_N5_K1_1ep baseline"

# echo ""
# echo "============================================================"
# echo "=== RECURRENCE GATE-CACHE TEST (1ep)"
# echo "==="
# echo "=== Compares against T4_recur_N5_K1_1ep (cache off). Same config,"
# echo "===   mixer_recurrence_cache_gate=true. Expect lower wall-clock;"
# echo "===   watch best val stays within 0.0015 of the no-cache run."
# echo "============================================================"


# # ---- Long-range context: multi-pole SSM + truncated BPTT --------------------
# # Two attention-free upgrades to the cross-window decompose-bypass context.
# # All four runs use sequential_blocks=true — the cross-window state (and BPTT
# # through it) is only meaningful in corpus order. The reference is therefore a
# # SEQUENTIAL T4, NOT the random-batched T4 from the recurrence sections.
# #   #1 decompose_bypass_ssm   : multi-pole diagonal SSM summary (P EMAs)
# #   #2 decompose_bypass_bptt  : retain cross-window graph across grad_accum span
# # Ablation order: SSM vs baseline, BPTT vs baseline, then both. Decision rule:
# # clear T4 best val 3.5157 by > 0.0015.

# # All four use grad_accum=2 (so BPTT span = 2 windows = one trained
# # cross-window link per step; with the BASE grad_accum=1 the span would be a
# # single window and BPTT a no-op). All four share this batch config so only the
# # SSM/BPTT flags vary. Effective batch = MBS(8) x GA(2) = 16.

# # LR0: sequential T4 baseline — the honest reference for this section.
# # run_ablation "T4_seq_baseline_1ep T4 sequential baseline for long-range (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "grad_accum": 2, "sequential_blocks": true}' \
# #     "T4_seq_baseline_1ep: T4 sequential, GA=2 (no SSM, no BPTT); reference for the long-range ablations"

# # LR1: + multi-pole SSM summary (P=4).
# # run_ablation "T4_seq_ssm_1ep T4 sequential + multi-pole SSM (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "grad_accum": 2, "sequential_blocks": true, "decompose_bypass_ssm": true, "decompose_bypass_ssm_poles": 4}' \
# #     "T4_seq_ssm_1ep: + multi-pole diagonal SSM (P=4) replacing the cumulative-mean context summary"

# # # LR2: + truncated BPTT across windows (span = grad_accum = 2).
# # run_ablation "T4_seq_bptt_1ep T4 sequential + truncated BPTT (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "grad_accum": 2, "sequential_blocks": true, "decompose_bypass_bptt": true}' \
# #     "T4_seq_bptt_1ep: + truncated BPTT (span=2 windows; trains the mean cross-window state)"

# # LR3: + both SSM and BPTT — richer state, trained to be useful.
# # run_ablation "T4_seq_ssm_bptt_1ep T4 sequential + SSM + BPTT (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "grad_accum": 2, "sequential_blocks": true, "decompose_bypass_ssm": true, "decompose_bypass_ssm_poles": 4, "decompose_bypass_bptt": true}' \
# #     "T4_seq_ssm_bptt_1ep: + multi-pole SSM AND truncated BPTT (span=2; the full long-range bet)"

# # LR4: + cross-window SSM — carry the pole state across block boundaries so the
# # long poles integrate beyond 256 tokens (forward-only carry in v1). Tests the
# # genuine cross-block long-range mechanism that the within-window SSM (LR1) is
# # only a confounded proxy for (within-window competes with the wavelet).
# # run_ablation "T4_seq_ssm_xwin_1ep T4 sequential + cross-window SSM (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "grad_accum": 2, "sequential_blocks": true, "decompose_bypass_ssm": true, "decompose_bypass_ssm_poles": 4, "decompose_bypass_ssm_cross_window": true}' \
# #     "T4_seq_ssm_xwin_1ep: + multi-pole SSM with cross-window pole-state carry (multi-timescale memory across blocks)"

# # # LR5: + cross-window SSM AND BPTT — the full long-range stack.
# # run_ablation "T4_seq_ssm_xwin_bptt_1ep T4 sequential + cross-window SSM + BPTT (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "grad_accum": 2, "sequential_blocks": true, "decompose_bypass_ssm": true, "decompose_bypass_ssm_poles": 4, "decompose_bypass_ssm_cross_window": true, "decompose_bypass_bptt": true}' \
# #     "T4_seq_ssm_xwin_bptt_1ep: + cross-window SSM AND truncated BPTT (full long-range stack)"

# echo ""
# echo "============================================================"
# echo "=== LONG-RANGE CONTEXT QUEUE (SSM + BPTT, 1ep each, sequential)"
# echo "==="
# echo "=== Reference: T4_seq_baseline_1ep (sequential T4, no SSM/BPTT)."
# echo "===   1) T4_seq_baseline_1ep      — sequential T4 reference"
# echo "===   2) T4_seq_ssm_1ep           — + within-window multi-pole SSM (P=4)"
# echo "===   3) T4_seq_bptt_1ep          — + truncated BPTT"
# echo "===   4) T4_seq_ssm_bptt_1ep      — + within-window SSM + BPTT"
# echo "===   5) T4_seq_ssm_xwin_1ep      — + cross-window SSM (pole-state carry)"
# echo "===   6) T4_seq_ssm_xwin_bptt_1ep — + cross-window SSM + BPTT (full stack)"
# echo "=== Decision rule: > 0.0015 nat improvement on best val vs sequential T4."
# echo "============================================================"


# # ---- Dense recurrence (DenseNet-style depth-weighted averaging) -------------
# # Each recurrence step's input is a learned weighted combination of ALL prior
# # step outputs (mixer_recurrence_dense=true), not just latest+X0. The M x M
# # lower-triangular weight matrix A is init to reproduce the input-anchored
# # residual exactly (verified bit-identical at init), so these are strict
# # generalizations of the with-residual recurrence runs. ~M^2/2 extra params
# # (~15 at N=5) — negligible. Thesis: routing over a SHARED bank (K=1) may
# # approach DISTINCT-bank (K=2, +58.85M params) quality → parameter efficiency.
# # Rank by BPB sliding (val understates context configs); compare each to its
# # input-anchored twin in the recurrence with-residual table.

# # DR1: dense N=5 K=1 (raw weights). Headline param-efficiency test vs K=2.
# # run_ablation "T4_recur_dense_N5_K1_1ep T4 dense recurrence N=5 K=1 (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 5, "mixer_recurrence_distinct_mixer_count": 1, "mixer_recurrence_residuals": true, "mixer_recurrence_dense": true}' \
# #     "T4_recur_dense_N5_K1_1ep: dense depth-weighted averaging over 5 steps (shared bank); vs input-anchored N=5 K=1 and vs K=2 (param efficiency)"

# # # DR2: dense N=5 K=1, normalized (softmax/convex rows) — interpretability-max.
# # run_ablation "T4_recur_dense_norm_N5_K1_1ep T4 dense recurrence N=5 K=1 normalized (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 5, "mixer_recurrence_distinct_mixer_count": 1, "mixer_recurrence_residuals": true, "mixer_recurrence_dense": true, "mixer_recurrence_dense_normalize": true}' \
# #     "T4_recur_dense_norm_N5_K1_1ep: softmax rows (per-step distribution over depths); interpretability variant vs raw DR1"

# # # DR3: dense N=10 K=1 — does dense routing let depth scale past where
# # # input-anchoring plateaus? Run only if DR1 shows promise.
# # run_ablation "T4_recur_dense_N10_K1_1ep T4 dense recurrence N=10 K=1 (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "mixer_recurrence_steps": 10, "mixer_recurrence_distinct_mixer_count": 1, "mixer_recurrence_residuals": true, "mixer_recurrence_dense": true}' \
# #     "T4_recur_dense_N10_K1_1ep: dense routing at depth 10; tests whether dense scales depth past the anchored plateau"

# echo ""
# echo "============================================================"
# echo "=== DENSE RECURRENCE QUEUE (depth-weighted averaging, 1ep each)"
# echo "==="
# echo "=== A init reproduces input-anchored exactly; these generalize it."
# echo "===   1) T4_recur_dense_N5_K1_1ep       — dense N=5 K=1 (raw)"
# echo "===   2) T4_recur_dense_norm_N5_K1_1ep  — dense N=5 K=1 (softmax rows)"
# echo "===   3) T4_recur_dense_N10_K1_1ep      — dense N=10 K=1 (depth-scale test)"
# echo "=== Rank by BPB sliding (~0.0010 threshold). Headline: dense N=5 K=1"
# echo "===   (~15 params) vs K=2 (+58.85M) — does routing replace parameters?"
# echo "=== Report the learned A matrix per run for interpretability."
# echo "============================================================"

# ---- Untied Wavelet Reconstruction ------------------------------------------
# Tests whether giving the reconstruct path its own predict/update networks
# (separate weights from decompose) improves expressivity enough to justify
# +~84M params. Mutually exclusive with mixer-only recurrence (breaks the
# Reconstruct∘Decompose=I invariant). Tested standalone at T4, random batching.

echo ""
echo "============================================================"
echo "=== Ablation: T4_untied_recon_1ep T4 + untied reconstruction (1ep)"
echo "============================================================"

run_ablation "T4_untied_recon_1ep T4 + untied wavelet reconstruction (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "untied_reconstruction": true}' \
    "T4_untied_recon_1ep: untied reconstruct path (+~84M params); standalone expressivity test vs T4 baseline"

# ---- Dropout sweep -----------------------------------------------------------
# Vary each of the 5 dropout quantities individually ±10% from T4 defaults,
# all others held at baseline. Final combined run uses the better value from
# each pair (or baseline if neither improved). Reference: T4 BPB 1.1311.
# T4 defaults: emb=0.20, proj=0.10, mix=0.10, mlp=0.10, lm_head=0.24.

# D1: dropout_embedding low (0.18)
run_ablation "T4_dropout_emb_low_1ep T4 dropout_embedding=0.18 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_embedding": 0.18}' \
    "T4_dropout_emb_low_1ep: dropout_embedding 0.20 -> 0.18 (-10%); all others at T4 defaults"

# D2: dropout_embedding high (0.22)
run_ablation "T4_dropout_emb_high_1ep T4 dropout_embedding=0.22 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_embedding": 0.22}' \
    "T4_dropout_emb_high_1ep: dropout_embedding 0.20 -> 0.22 (+10%); all others at T4 defaults"

# D3: dropout_projection low (0.09)
run_ablation "T4_dropout_proj_low_1ep T4 dropout_projection=0.09 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_projection": 0.09}' \
    "T4_dropout_proj_low_1ep: dropout_projection 0.10 -> 0.09 (-10%); all others at T4 defaults"

# D4: dropout_projection high (0.11)
run_ablation "T4_dropout_proj_high_1ep T4 dropout_projection=0.11 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_projection": 0.11}' \
    "T4_dropout_proj_high_1ep: dropout_projection 0.10 -> 0.11 (+10%); all others at T4 defaults"

# D5: dropout_mixer low (0.09)
run_ablation "T4_dropout_mix_low_1ep T4 dropout_mixer=0.09 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_mixer": 0.09}' \
    "T4_dropout_mix_low_1ep: dropout_mixer 0.10 -> 0.09 (-10%); all others at T4 defaults"

# D6: dropout_mixer high (0.11)
run_ablation "T4_dropout_mix_high_1ep T4 dropout_mixer=0.11 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_mixer": 0.11}' \
    "T4_dropout_mix_high_1ep: dropout_mixer 0.10 -> 0.11 (+10%); all others at T4 defaults"

# D7: dropout_mlp low (0.09)
run_ablation "T4_dropout_mlp_low_1ep T4 dropout_mlp=0.09 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_mlp": 0.09}' \
    "T4_dropout_mlp_low_1ep: dropout_mlp 0.10 -> 0.09 (-10%); all others at T4 defaults"

# D8: dropout_mlp high (0.11)
run_ablation "T4_dropout_mlp_high_1ep T4 dropout_mlp=0.11 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_mlp": 0.11}' \
    "T4_dropout_mlp_high_1ep: dropout_mlp 0.10 -> 0.11 (+10%); all others at T4 defaults"

# D9: dropout_lm_head low (0.216)
run_ablation "T4_dropout_lmh_low_1ep T4 dropout_lm_head=0.216 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_lm_head": 0.216}' \
    "T4_dropout_lmh_low_1ep: dropout_lm_head 0.24 -> 0.216 (-10%); all others at T4 defaults"

# D10: dropout_lm_head high (0.264)
run_ablation "T4_dropout_lmh_high_1ep T4 dropout_lm_head=0.264 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_lm_head": 0.264}' \
    "T4_dropout_lmh_high_1ep: dropout_lm_head 0.24 -> 0.264 (+10%); all others at T4 defaults"

# D11: optimal combined dropout — fill in best value from each pair above.
# Values below are placeholders; replace each TBD with the better of the two
# values tested for that quantity (or the T4 default if neither improved).
# run_ablation "T4_dropout_optimal_1ep T4 optimal combined dropout (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "dropout_embedding": TBD, "dropout_projection": TBD, "dropout_mixer": TBD, "dropout_mlp": TBD, "dropout_lm_head": TBD}' \
#     "T4_dropout_optimal_1ep: best dropout value per type from individual sweeps; combined optimum"

# ---- Weight decay sweep ------------------------------------------------------
# Two flanking values around the T4 default (1e-6). Reference: T4 BPB 1.1311.

# WD1: lower weight decay (5e-7)
run_ablation "T4_wd_5e7_1ep T4 weight_decay=5e-7 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "weight_decay": 5e-7}' \
    "T4_wd_5e7_1ep: weight_decay 1e-6 -> 5e-7 (halved); vs T4 baseline"

# WD2: higher weight decay (2e-6)
run_ablation "T4_wd_2e6_1ep T4 weight_decay=2e-6 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "weight_decay": 2e-6}' \
    "T4_wd_2e6_1ep: weight_decay 1e-6 -> 2e-6 (doubled); vs T4 baseline"

