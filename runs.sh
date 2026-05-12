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
# run_ablation "T2_WC_1ep T2 with wavelet_crawl=True (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true}' \
#     "T2_WC_1ep: T1 + levels=7 + R0 mixer widths [1.0x4, 0.5x4] + wavelet_crawl=True (1ep, bs=256, MBS=8)"

# run_ablation "T2_WC_5ep New baseline T2 at 5 epochs (with wavelet_crawl)" \
#     "$BASE_PATCH_5EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true}' \
#     "T2_WC_5ep: T1 + levels=7 + R0 mixer widths [1.0x4, 0.5x4] + wavelet_crawl=True (5ep, bs=256, MBS=8)"


# ---- Muon optimizer sweep on T2 (Path B: lr in Keller's regime) ------------
# Hybrid Muon + AdamW: Muon on 2D non-embedding hidden weights (Linear matrices
# in lifting/mixer/MLP), AdamW on biases/norms/embeddings/LM head. Per
# torch.optim.Muon docs, Muon is for 2D parameters of hidden layers only.
#
# IMPORTANT — LR scaling rationale:
# torch.optim.Muon's default `adjust_lr_fn=None` (= "original" / Keller's
# scaling) scales LR by max(1, sqrt(A/B)) per matrix. For square matrices
# (2048x2048 mixer/lifting), this is 1.0 — no LR amplification. The doc's
# default lr=0.001 is calibrated for "match_rms_adamw" semantics (which would
# scale by ~9-29x for our matrices); under "original" scaling with lr=0.001
# you're effectively running at 10-50x lower LR than Muon's reference recipes
# (Keller, DeepSeek-V4) intend. Empirically confirmed by the partial run at
# logs/wikitext-103_2026-05-10_15-49-10/log.txt — trains, but slowly.
#
# Path B (this sweep): keep `adjust_lr_fn=None` (the API default) and lift
# LR into Keller's published range (0.01-0.05) and beyond, to find Muon's
# native operating point on our T2 stack.
#
# Defaults retained: weight_decay=0.1, momentum=0.95, nesterov=True,
# ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-7, ns_steps=5. AdamW group
# inherits Muon's lr and weight_decay by default. min_lr scaled at the same
# 1/50 floor as the existing Adagrad config.
#
# Decision criterion: Muon must clear T2 (Adagrad)'s end-of-epoch best val
# (3.5881 at 1ep) by enough margin to compensate its ~2x wall-clock cost —
# i.e., at least half the epochs for matched performance, or strictly better
# end-of-epoch numbers at matched epochs. Otherwise Adagrad stays.
#
# All four runs use the T2 architecture (levels=7, per_scale_mixer_widths
# =[1.0x4, 0.5x4], wavelet_crawl=True).
#
# DEFUNCT / suboptimal — historical record of the Path A LR misfire:
# # Run 1 (PARTIAL — cancelled): Muon defaults (lr=0.001)
# # logs/wikitext-103_2026-05-10_15-49-10/log.txt — under-LR'd per the analysis above.
# run_ablation "T2_Muon_default_1ep T2 + Muon defaults (lr=0.001, 1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.001, "min_lr": 0.00002, "weight_decay": 0.1}' \
#     "T2_Muon_default_1ep: T2 + Muon defaults (1ep, lr=0.001, wd=0.1, momentum=0.95, eps=1e-7, ns_steps=5)"
#
# # Run 2: Muon at lr=0.002 — also under-LR'd, never queued.
# run_ablation "T2_Muon_lr2e-3_1ep T2 + Muon lr=0.002 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.002, "min_lr": 0.00004, "weight_decay": 0.1}' \
#     "T2_Muon_lr2e-3_1ep: T2 + Muon lr=0.002 (1ep, otherwise default)"
#
# # Run 3: Muon at lr=0.0005 — also under-LR'd, never queued.
# run_ablation "T2_Muon_lr5e-4_1ep T2 + Muon lr=0.0005 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.0005, "min_lr": 0.00001, "weight_decay": 0.1}' \
#     "T2_Muon_lr5e-4_1ep: T2 + Muon lr=0.0005 (1ep, otherwise default)"

# DEFUNCT — Path B v1: lr=0.01 already showed too-aggressive symptoms
# (oscillation/plateau in a 4.72-4.77 val band from step ~5000 onward, while
# Adagrad continued smoothly descending past it). lr=0.05 / 0.10 / 0.20 were
# guaranteed-worse and skipped. Pivoting to lr=0.003 / lr=0.005 to find the
# true sweet spot between under-scaled (0.001) and over-aggressive (0.01).
# # Run B1: Muon at lr=0.01 (low end of Keller's published range — partial run).
# # logs/wikitext-103_2026-05-10_17-45-55/log.txt — cancelled at ~step 17,537 (~30%)
# run_ablation "T2_Muon_lr1e-2_1ep T2 + Muon lr=0.01 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.01, "min_lr": 0.0002, "weight_decay": 0.1}' \
#     "T2_Muon_lr1e-2_1ep: T2 + Muon lr=0.01 (1ep, adjust_lr_fn=None, wd=0.1, momentum=0.95, eps=1e-7, ns_steps=5)"
#
# # Run B2: Muon at lr=0.05 — skipped (lr=0.01 already over-aggressive).
# run_ablation "T2_Muon_lr5e-2_1ep T2 + Muon lr=0.05 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.05, "min_lr": 0.001, "weight_decay": 0.1}' \
#     "T2_Muon_lr5e-2_1ep: T2 + Muon lr=0.05 (1ep, otherwise as B1)"
#
# # Run B3: Muon at lr=0.1 — skipped.
# run_ablation "T2_Muon_lr1e-1_1ep T2 + Muon lr=0.1 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.1, "min_lr": 0.002, "weight_decay": 0.1}' \
#     "T2_Muon_lr1e-1_1ep: T2 + Muon lr=0.1 (1ep, otherwise as B1)"
#
# # Run B4: Muon at lr=0.2 — skipped.
# run_ablation "T2_Muon_lr2e-1_1ep T2 + Muon lr=0.2 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.2, "min_lr": 0.004, "weight_decay": 0.1}' \
#     "T2_Muon_lr2e-1_1ep: T2 + Muon lr=0.2 (1ep, stress test — well above Keller's published range)"


# ---- Sequential block ordering sweep on T2 (4 runs, 2x2 cross) -------------
# Tests deterministic sequential block ordering vs. the current random sampler.
# Two MBS regimes (perfect-within-batch via MBS=1/GA=8 vs. stream-batched via
# MBS=8/GA=1), each at 1 and 2 epochs. The 2-epoch runs probe the one-shot-
# learner hypothesis: with sequential ordering, every token is seen exactly
# once per epoch, so a second epoch is purely "second pass over already-seen
# data." If WaveletLM is a strong one-shot learner, T2_seq_2ep should plateau
# in late epoch 2 *despite still-elevated post-peak LR* (peak is at step
# 35,074 = 60% through epoch 1; cosine decay through end of epoch 2 keeps LR
# above min throughout). The plateau-while-LR-nontrivial is the load-bearing
# evidence; LR-decay-driven convergence is excluded as a confound.
#
# All four runs use T2 architecture (levels=7, [1.0x4, 0.5x4] mixer,
# wavelet_crawl=True, Adagrad lr=0.01 per BASE_PATCH defaults). Sequential
# mode is gated by `sequential_blocks=true` in the override. The closure in
# make_get_batch automatically picks Rainman vs pure based on MBS.

# Run S1: Rainman 1ep (MBS=8 / GA=1, 8 parallel streams advancing in lockstep).
# run_ablation "T2_seq_M8_1ep T2 sequential blocks, MBS=8/GA=1 (Rainman, 1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "sequential_blocks": true, "micro_batch_size": 8, "grad_accum": 1}' \
#     "T2_seq_M8_1ep: sequential blocks, MBS=8/GA=1 (Rainman; 1ep, T2 stack, Adagrad lr=0.01)"

# # Run S2: Rainman 2ep — one-shot-learner hypothesis test (Rainman variant).
# run_ablation "T2_seq_M8_2ep T2 sequential blocks, MBS=8/GA=1 (Rainman, 2ep)" \
#     "$BASE_PATCH_5EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "sequential_blocks": true, "micro_batch_size": 8, "grad_accum": 1, "epochs": 2}' \
#     "T2_seq_M8_2ep: sequential blocks, MBS=8/GA=1 (Rainman; 2ep one-shot test, T2 stack, Adagrad lr=0.01)"

# # Run S3: Pure sequential 1ep (MBS=1 / GA=8, single stream).
# run_ablation "T2_seq_M1_1ep T2 sequential blocks, MBS=1/GA=8 (pure sequential, 1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "sequential_blocks": true, "micro_batch_size": 1, "grad_accum": 8}' \
#     "T2_seq_M1_1ep: sequential blocks, MBS=1/GA=8 (pure sequential; 1ep, T2 stack, Adagrad lr=0.01)"

# # Run S4: Pure sequential 2ep — one-shot-learner hypothesis test (pure variant).
# run_ablation "T2_seq_M1_2ep T2 sequential blocks, MBS=1/GA=8 (pure sequential, 2ep)" \
#     "$BASE_PATCH_5EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "sequential_blocks": true, "micro_batch_size": 1, "grad_accum": 8, "epochs": 2}' \
#     "T2_seq_M1_2ep: sequential blocks, MBS=1/GA=8 (pure sequential; 2ep one-shot test, T2 stack, Adagrad lr=0.01)"


# ---- 2D wavelet over (batch, token) — PRIORITY: tested first ----------------
# Phase 2.A of the 2D wavelet rollout: "internal" mode. Adds separable B-axis
# lifting at each level (up to log2(B)=3 levels for B=8), with per-sub-band
# scaling, then B-axis inverse lift to reassemble into the standard (approx,
# details) output shape. Cross-batch information enters via the per-sub-band
# processing; no changes to model.py mixer or reconstruct paths required.
#
# Phase 1 scaffold (mode="passthrough") was validated on 2026-05-11
# (logs/wikitext-103_2026-05-11_05-26-15 et al.): trajectory matches the
# Rainman baseline within noise (~0.001 nats), confirming the integration
# surface (config flag, lazy import, forwarded properties, autograd) is
# clean. We now enable mode="internal" to test whether the B-axis lifting
# itself provides cross-batch benefit.
#
# Init semantics: wavelet_2d_init_zero=true means B-axis predict/update nets
# start as zero (no contribution) AND per-sub-band scales start as 1.0
# (identity). Net effect: step-0 behavior is exactly the 1D wavelet. Training
# learns whether to activate the B-axis path.
#
# Expected outcome:
#   * If "internal" mode helps: best val < 3.66 (Rainman baseline) by a
#     meaningful margin (> 0.0015 noise threshold).
#   * If it hurts: best val > 3.66. Probably won't be catastrophic since
#     init=identity, but the optimizer may push it in a bad direction.
#   * If it ties: best val ≈ 3.66. Suggests B-axis lifting via "internal"
#     mode doesn't carry useful gradient signal, and we should try "subband"
#     (Phase 2.B) which exposes sub-bands to per-band mixers.
#
# Param overhead: small. 2 * b_levels Linear(C, C) for b_predict_nets +
# b_update_nets = 2 * 3 * 2048^2 ≈ 25M new params. Per-sub-band scales:
# 3 * 4 * 2048 = 24K params. Total ≈ +25M (~6% increase over T2's 393M).

# Run W1: T2 Rainman + wavelet_2d_mode="internal" (Phase 2.A test).
# run_ablation "T2_seq_M8_1ep_2d_internal T2 Rainman + 2D wavelet 'internal' mode (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "sequential_blocks": true, "micro_batch_size": 8, "grad_accum": 1, "wavelet_2d_mode": "internal"}' \
#     "T2_seq_M8_1ep_2d_internal: Rainman + 2D wavelet 'internal' mode (Phase 2.A; identity-init B-axis lift + per-sub-band scale + B-axis inverse; same output shape as 1D)"

# Run W2: T2 Rainman + wavelet_2d_mode="subband" (Phase 2.B test).
# Per the architectural distinction (see tools/two_d_wavelets.py docstring):
# subband mode exposes 3 detail sub-bands per joint level (LH, HL, HH) to the
# downstream per-scale mixer instead of collapsing them back into 1 detail.
# Per-sub-band mixer specialization captures band-specific structure that
# "internal" mode's reassembly throws away. Mixer count auto-extends from 8
# (1D: L+1) to 14 (subband: 3*b_levels + (L-b_levels) + 1 for T2's b_levels=3).
# per_scale_mixer_widths auto-expands from 8 entries to 14 (joint-level
# entries get tripled). Reconstruction uses LiftingWavelet2D.reconstruct_subband.
#
# Init semantics: same as internal mode — B-axis nets zero-init, so step-0
# behavior is exactly the 1D wavelet (the extra mixer slots get identity-ish
# init via the existing per-scale mixer code). Training learns whether to
# activate the B-axis path AND the per-sub-band mixer specialization.
#
# Param overhead vs internal mode: same B-axis nets (~25M) PLUS the extra 6
# mixer slots (3 per joint level × 3 joint levels = 9 extra mixer instances
# vs internal's 8). Each PerScaleMixer is ~5.9M params at width=0.5*Cp=1024,
# so 6 extra mixers ≈ 35M more params. Total: T2 + ~60M (~15% over T2's 393M).
# SHELVED (2026-05-11): both 2D wavelet modes underperformed Rainman baseline
# on WT-103 sequential. "internal" mode: best val 3.6691 (Δ +0.0090, +14%
# wall-clock — logs/wikitext-103_2026-05-11_08-05-10/). "subband" mode: best
# val 3.7199 (Δ +0.0598, +30% wall-clock — logs/wikitext-103_2026-05-11_13-00-11/).
# Likely cause: Wikipedia articles are largely independent at the chunk level,
# so cross-batch temporal structure that would justify 2D decomposition isn't
# present in WT-103. 2D wavelets may still work on PG-19 (long-form novels with
# multi-book dependencies); revisit there if compute allows. Code preserved in
# tools/two_d_wavelets.py for future revisit.
# run_ablation "T2_seq_M8_1ep_2d_subband T2 Rainman + 2D wavelet 'subband' mode (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "sequential_blocks": true, "micro_batch_size": 8, "grad_accum": 1, "wavelet_2d_mode": "subband"}' \
#     "T2_seq_M8_1ep_2d_subband: Rainman + 2D wavelet 'subband' mode (Phase 2.B; 4 sub-bands per joint level exposed to per-band mixers; mixer count 14 vs 1D's 8)"


# ---- T2 random sampling + lr=0.015 (Adagrad comparison) ---------------------
# Fair-comparison companion to the sequential lr=0.015 result
# (logs/wikitext-103_2026-05-11_10-14-25: best val 3.6231). That sequential
# run beat the lr=0.01 Rainman baseline (3.6601) by Δ=-0.0370, recovering
# ~50% of the random→sequential gap. But T2 random was still at lr=0.01,
# so we don't yet know whether lr=0.015 is a sequential-specific fix or a
# general improvement. This run isolates the LR variable on the random-
# sampling stack — if random+lr=0.015 also outperforms random+lr=0.01
# (3.5881), then lr=0.015 is partly a general-purpose Adagrad tune for T2.
# If it ties or regresses, the lr=0.015 benefit is sequential-specific.
# run_ablation "T2_rand_1ep_lr15 T2 random sampling + lr=0.015 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003}' \
#     "T2_rand_1ep_lr15: T2 random sampling + lr=0.015 (1ep; fair-comparison companion to T2_seq_M8_1ep_lr15; isolates LR vs sequential variable)"


# ---- BBCE sweep (Bisected Block Context Extension) --------------------------
# Each batch reads `block_size_compressed` tokens; the first
# (block_size_compressed - block_size/2) tokens are chunk-averaged into
# block_size/2 compressed slots; the last block_size/2 tokens are
# uncompressed. Concatenated, the (B, block_size, C) result feeds the
# standard wavelet pipeline. Loss is computed only on the last block_size/2
# positions (the supervised "uncompressed half").
#
# Random sampling is used throughout (Phase 1). Sequential ordering with
# slot caching is a Phase 2 follow-up if any of these runs shows promise.
#
# Sweep grid: block_size ∈ {256, 512} × block_size_compressed ∈ {65K, 262K,
# 1M, 4M}. Ordered shortest→longest wall-clock to surface signal quickly.
# The very long variants (4M) may need to be cancelled if too slow; results
# from smaller variants should already settle the "does long context help?"
# question.
#
# Expected wall-clock per run (random sampling, MBS=8, 5090). BBCE epoch
# defined as one full pass over supervised positions = step count doubles
# vs the prior bs-only formula (see Step Count Methodology in README BBCE
# section).
#   - bc=65K:  ~4-5h
#   - bc=262K: ~6h
#   - bc=1M:   ~16-20h
#   - bc=4M:   ~60-80h  (long; may want to cancel early or run separately)
# block_size=512 variants are roughly 10-20% slower than bs=256 counterparts.
#
# Comparison reference: T2 random 1ep best val 3.5881 (the standard baseline).
# Decision: any BBCE variant must clear 3.5881 by > 0.0015 (noise threshold)
# to be considered a win. Tied or worse → BBCE adds context that the model
# can't usefully exploit at this scale, and we drop the direction.

# BBCE base patch — block_size=256, levels=7, T2 mixer widths.
BBCE_BASE_256='{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "bbce_enabled": true, "block_size": 256}'
# BBCE base patch — block_size=512, levels=7, same mixer widths (8 entries =
# L+1 regardless of block_size).
BBCE_BASE_512='{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "bbce_enabled": true, "block_size": 512}'

# BBCE-1: bc=65K (smallest, fastest signal)
# run_ablation "T2_BBCE_b256_bc65K_1ep block_size=256, block_size_compressed=65,536 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 256, \"block_size_compressed\": 65536}" \
#     "T2_BBCE_b256_bc65K_1ep: BBCE bs=256/bc=65K (1ep, random sampling, T2 stack)"

# ---- One-off: replay benchmark + generations for T2_BBCE_b256_bc65K_1ep -----
# The 2026-05-11_17-25-07 run trained successfully (best val 3.8018) but the
# original post-training steps failed: the old code skipped the test benchmark
# for any BBCE run (block_size mismatch with test slices) and generation
# crashed because the BBCE preprocessor strictly requires (B, bs_compressed)
# input while generate.py was passing the raw 3-token prompt. Both fixed in
# train.py (evaluate_bbce) and generate.py (pad_idx_for_bbce), so we replay
# the post-training steps against the existing best_model.pt before resuming
# the rest of the sweep. Neither config.json is mutated — see
# benchmark_only_run docstring.
# benchmark_only_run "T2_BBCE_b256_bc65K_1ep (benchmark + generations replay)" \
#     "logs/wikitext-103_2026-05-11_17-25-07" \
#     "T2_BBCE_b256_bc65K_1ep: replay benchmark + generations against existing checkpoint (post BBCE benchmark/generation fixes)"

# run_ablation "T2_BBCE_b512_bc65K_1ep block_size=512, block_size_compressed=65,536 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 512, \"block_size_compressed\": 65536}" \
#     "T2_BBCE_b512_bc65K_1ep: BBCE bs=512/bc=65K (1ep, random sampling, T2 stack)"

# # BBCE-2: bc=262K
# run_ablation "T2_BBCE_b256_bc262K_1ep block_size=256, block_size_compressed=262,144 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 256, \"block_size_compressed\": 262144}" \
#     "T2_BBCE_b256_bc262K_1ep: BBCE bs=256/bc=262K (1ep, random sampling, T2 stack)"

run_ablation "T2_BBCE_b512_bc262K_1ep block_size=512, block_size_compressed=262,144 (1ep)" \
    "$BASE_PATCH_1EP" \
    "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 512, \"block_size_compressed\": 262144}" \
    "T2_BBCE_b512_bc262K_1ep: BBCE bs=512/bc=262K (1ep, random sampling, T2 stack)"

# # BBCE-3: bc=1M
# run_ablation "T2_BBCE_b256_bc1M_1ep block_size=256, block_size_compressed=1,048,576 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 256, \"block_size_compressed\": 1048576}" \
#     "T2_BBCE_b256_bc1M_1ep: BBCE bs=256/bc=1M (1ep, random sampling, T2 stack)"

run_ablation "T2_BBCE_b512_bc1M_1ep block_size=512, block_size_compressed=1,048,576 (1ep)" \
    "$BASE_PATCH_1EP" \
    "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 512, \"block_size_compressed\": 1048576}" \
    "T2_BBCE_b512_bc1M_1ep: BBCE bs=512/bc=1M (1ep, random sampling, T2 stack)"

# BBCE-large-bs: block_size=2048 × bc ∈ {65K, 262K, 1M}. Key compute insight:
# steps_per_epoch = corpus_size / (block_size * effective_batch). At bs=2048,
# steps_per_epoch is 8× smaller than bs=256 (≈7,325 vs ≈58,500). Per-step
# compute is ~8× higher (wavelet/mixer/MLP all scale with T), so total
# compute per epoch is roughly equivalent. BUT BBCE's embedding overhead is
# per-step, so bs=2048 sees 8× less embedding work per epoch — meaningful
# savings at large bc.
#
# VRAM caveat: at bs=2048/MBS=8 the activation memory would OOM the 5090
# (~32GB activations alone). Setting MBS=4/GA=2 maintains effective batch=8
# while fitting VRAM (~16-20GB activations).

run_ablation "T2_BBCE_b2048_bc65K_1ep block_size=2048, block_size_compressed=65,536 (1ep)" \
    "$BASE_PATCH_1EP" \
    "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 2048, \"block_size_compressed\": 65536, \"micro_batch_size\": 4, \"grad_accum\": 2}" \
    "T2_BBCE_b2048_bc65K_1ep: BBCE bs=2048/bc=65K (1ep, random sampling, MBS=4/GA=2 for VRAM)"

run_ablation "T2_BBCE_b2048_bc262K_1ep block_size=2048, block_size_compressed=262,144 (1ep)" \
    "$BASE_PATCH_1EP" \
    "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 2048, \"block_size_compressed\": 262144, \"micro_batch_size\": 4, \"grad_accum\": 2}" \
    "T2_BBCE_b2048_bc262K_1ep: BBCE bs=2048/bc=262K (1ep, random sampling, MBS=4/GA=2)"

run_ablation "T2_BBCE_b2048_bc1M_1ep block_size=2048, block_size_compressed=1,048,576 (1ep)" \
    "$BASE_PATCH_1EP" \
    "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 2048, \"block_size_compressed\": 1048576, \"micro_batch_size\": 4, \"grad_accum\": 2}" \
    "T2_BBCE_b2048_bc1M_1ep: BBCE bs=2048/bc=1M (1ep, random sampling, MBS=4/GA=2)"


# ---- T2 random sampling + lr=0.020 (upper bracket of the LR sweep) ----------
# Confirmation companion to T2_rand_1ep_lr15. The lr=0.015 run is tracking
# ~0.044 nats ahead of T2 baseline (lr=0.01) at matched step — clear win.
# This run checks whether lr=0.020 is even better or whether we've already
# passed the optimum. Decision rule:
#   - lr=0.020 best val < lr=0.015 best val → optimum is >=0.02; consider
#     another sweep at lr=0.025 to find the peak.
#   - lr=0.020 best val ≈ lr=0.015 best val (within noise) → plateau; lock
#     in 0.015 as the new default.
#   - lr=0.020 best val > lr=0.015 best val → past the optimum; 0.015 wins.
# Watch for late-epoch oscillation symptomatic of too-high LR — saw this
# with Muon at lr=0.01 (plateau/oscillation after step 5000). If lr=0.020
# shows similar oscillation, it's likely too high for Adagrad on T2.
run_ablation "T2_rand_1ep_lr20 T2 random sampling + lr=0.020 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.020, "min_lr": 0.0004}' \
    "T2_rand_1ep_lr20: T2 random sampling + lr=0.020 (1ep; upper-bracket of LR sweep, follows the lr=0.015 win)"


# BBCE-4: bc=4M REMOVED — wall-clock impractical (~30-45h per run on a 5090).
# If the bc=65K/262K/1M sweep shows BBCE is worth pursuing, the 4M variant
# may be worth revisiting (perhaps with sequential+caching, since random
# sampling re-embeds the full 4M window every step).
# run_ablation "T2_BBCE_b256_bc4M_1ep block_size=256, block_size_compressed=4,194,304 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 256, \"block_size_compressed\": 4194304}" \
#     "T2_BBCE_b256_bc4M_1ep: BBCE bs=256/bc=4M (1ep, random sampling, T2 stack)"
#
# run_ablation "T2_BBCE_b512_bc4M_1ep block_size=512, block_size_compressed=4,194,304 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 512, \"block_size_compressed\": 4194304}" \
#     "T2_BBCE_b512_bc4M_1ep: BBCE bs=512/bc=4M (1ep, random sampling, T2 stack)"


# ---- Adagrad-fix Rainman tests on sequential ordering (deprioritized) -------
# Rainman 1ep (logs/wikitext-103_2026-05-10_19-36-28) trailed T2 random 1ep by
# +0.0720 best val (3.6601 vs 3.5881). The IAV=0.1 probe was attempted but
# failed catastrophically (val ~7.6 at step 2000 vs Rainman baseline 5.77) —
# see Sequential Block Ordering section in README for the post-mortem.
# IAV support has been removed from train.py / config.json as failed-experiment
# bloat. The lr=0.015 probe remains as a mechanism-agnostic uniform LR bump.

# Run A1: Rainman 1ep + lr=0.015 (uniform LR bump; partial fix expected to
# recover ~30-50% of the gap by lifting the floor for accumulator-collapsed
# parameters).
# run_ablation "T2_seq_M8_1ep_lr15 T2 Rainman + lr=0.015 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "sequential_blocks": true, "micro_batch_size": 8, "grad_accum": 1, "lr": 0.015, "min_lr": 0.0003}' \
#     "T2_seq_M8_1ep_lr15: Rainman + lr=0.015 (1ep, otherwise as T2_seq_M8_1ep baseline)"


# ---- Path B v2: tighter LR band between 0.001 (smooth/slow) and 0.01 (oscillates) ----
# The lr=0.001 run (15:49) trained smoothly but ~10x slower than needed. The
# lr=0.01 run (17:45) leapt ahead of Adagrad through ~step 6000 but plateaued
# in a 4.72-4.77 oscillation band from ~step 5000. The optimal Muon LR for
# T2 is likely between these — values that give the early-LR head start
# without the post-warmup oscillation.

# Run B5: Muon at lr=0.003 (3x the under-scaled 0.001).
run_ablation "T2_Muon_lr3e-3_1ep T2 + Muon lr=0.003 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.003, "min_lr": 0.00006, "weight_decay": 0.1}' \
    "T2_Muon_lr3e-3_1ep: T2 + Muon lr=0.003 (1ep, between under-scaled 0.001 and over-aggressive 0.01)"

# Run B6: Muon at lr=0.005 (half the over-aggressive 0.01).
run_ablation "T2_Muon_lr5e-3_1ep T2 + Muon lr=0.005 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.005, "min_lr": 0.0001, "weight_decay": 0.1}' \
    "T2_Muon_lr5e-3_1ep: T2 + Muon lr=0.005 (1ep, otherwise as B5)"


# ---- BBCE compressed-half gradient ablation --------------------------------
# Single confirmatory test of `bbce_compressed_grad=false`: torch.no_grad
# wraps the chunked embedding lookup + mean-pool for the compressed half,
# halting backprop into the embedding table for tokens that appear there.
# The embedding table still learns from uncompressed-half appearances; the
# wavelet/mixer/MLP still learn to USE compressed slots — they just don't
# get to tell the embedding table to be a better mean-pool basis.
#
# Expected speedup: ~1.3-1.5x on backward (per Gemini's analysis, unverified).
# Expected quality: marked decrease vs the bbce_compressed_grad=true variant
# of the same cell (bs=512 / bc=65K), but still likely better than the T2
# baseline (the compressed half being treated as cheap "additional context"
# is the working theory). Decision logic:
#   - Within noise of the gradient-on variant: bbce_compressed_grad=false
#     becomes a viable speed/quality tradeoff for compute-bound regimes.
#   - Notable regression (Δ best val > +0.01 vs gradient-on): the embedding
#     table's role as a learned mean-pool basis matters; keep gradients on
#     by default.
#   - Catastrophic (Δ best val > +0.05): bbce_compressed_grad=false is not
#     usable; the embedding table being trained for mean-pool quality is
#     load-bearing.
#
# Picked bs=512 / bc=65K as the test cell because it's the cheapest of the
# headline (per-batch-supervision-matched) BBCE configs.
run_ablation "T2_BBCE_b512_bc65K_1ep_ngrad bbce_compressed_grad=false (1ep)" \
    "$BASE_PATCH_1EP" \
    "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 512, \"block_size_compressed\": 65536, \"bbce_compressed_grad\": false}" \
    "T2_BBCE_b512_bc65K_1ep_ngrad: BBCE bs=512/bc=65K + bbce_compressed_grad=false (1ep, single confirmatory A/B vs the gradient-on counterpart from the main sweep)"


echo ""
echo "============================================================"
echo "=== Queue complete (trimmed BBCE sweep + LR=0.020 + Muon LR + compressed_grad A/B)."
echo "===   1) T2_BBCE_b512_bc262K_1ep      — BBCE bs=512/bc=262K  (~6-8h)"
echo "===   2) T2_BBCE_b512_bc1M_1ep        — BBCE bs=512/bc=1M    (~18-24h)"
echo "===   3) T2_BBCE_b2048_bc65K_1ep      — BBCE bs=2048/bc=65K  (~4-6h)"
echo "===   4) T2_BBCE_b2048_bc262K_1ep     — BBCE bs=2048/bc=262K (~5-7h)"
echo "===   5) T2_BBCE_b2048_bc1M_1ep       — BBCE bs=2048/bc=1M   (~6-8h)"
echo "===   6) T2_rand_1ep_lr20             — T2 random + lr=0.020 (upper-bracket of LR sweep, ~1.85h)"
echo "===   7) T2_Muon_lr3e-3_1ep           — Muon lr=0.003 (Path B v2)"
echo "===   8) T2_Muon_lr5e-3_1ep           — Muon lr=0.005 (Path B v2)"
echo "===   9) T2_BBCE_b512_bc65K_1ep_ngrad — bbce_compressed_grad=false A/B (~2.5h)"
echo "==="
echo "=== Completed (commented out above):"
echo "===   - T2_rand_1ep_lr15             — T2 + lr=0.015, Δ best val −0.0536 vs T2 baseline (win)"
echo "===   - T2_BBCE_b256_bc65K_1ep       — completed: best val 3.5725, BPB 1.1551 (Δ +0.0010 vs T2, tied within noise)"
echo "===   - T2_BBCE_b512_bc65K_1ep       — in progress / completed (the bs=512 headline bc=65K cell)"
echo "==="
echo "=== Dropped (cost vs information not worth it; bs=512 / bs=2048 columns"
echo "===   already cover the bc-scaling story at more interesting block sizes):"
echo "===   - T2_BBCE_b256_bc262K_1ep      — ~6h saved"
echo "===   - T2_BBCE_b256_bc1M_1ep        — ~16-20h saved"
echo "==="
echo "=== BBCE step count: full-corpus supervised coverage per epoch under the"
echo "===   new active-stride formula (block_size/2 for BBCE, block_size"
echo "===   otherwise). Per-epoch step count is ~2x the prior bs-only formula."
echo "==="
echo "=== bs=2048 runs: MBS=4/GA=2 (effective batch unchanged at 8) to fit VRAM"
echo "===   at T=2048. steps_per_epoch=14,650 (4x fewer than bs=256 BBCE's"
echo "===   116,914) means BBCE embedding overhead is 4x less per epoch —"
echo "===   meaningful at large bc. Also a useful axis for studying"
echo "===   block-size effect on BBCE quality."
echo "==="
echo "=== Removed (wall-clock impractical at ~30-45h per run):"
echo "===   - T2_BBCE_b256_bc4M_1ep / T2_BBCE_b512_bc4M_1ep"
echo "===   May revisit with sequential+caching if smaller variants show promise."
echo "==="
echo "=== BBCE decision rule: any variant must clear T2 random 1ep best val"
echo "===   (3.5881) by > 0.0015 (noise threshold) to be considered a win."
echo "===   Tied or worse → BBCE adds context that the model can't usefully"
echo "===   exploit at this scale, and we drop the direction."
echo "==="
echo "=== Shelved (commented out, code preserved):"
echo "===   - T2_seq_M8_1ep_2d_internal  — Δ +0.0090 vs Rainman, +14% wall-clock"
echo "===   - T2_seq_M8_1ep_2d_subband   — Δ +0.0598 vs Rainman, +30% wall-clock"
echo "===   - T2_seq_M8_1ep_lr15         — completed: best val 3.6231 (~50% gap recovery)"
echo "==="
echo "=== Two architectural shots at finding cross-batch lift benefit:"
echo "===   'internal' — B-axis lift internal to wavelet (sub-band scaling +"
echo "===                inverse lift); same (approx, details) output shape."
echo "===   'subband' — 4 sub-bands per joint level exposed to per-band mixers"
echo "===                (mixer count 14 vs 1D's 8); per-band specialization."
echo "==="
echo "=== Watch vs Rainman baseline (3.66 best val). If lower by > 0.0015 nats,"
echo "===   that mode's architectural premise carries useful signal."
echo "=== See tools/two_d_wavelets.py and plans/two_d_wavelet_sequential_training.md."
echo "============================================================"
