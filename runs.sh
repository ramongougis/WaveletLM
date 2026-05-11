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
run_ablation "T2_seq_M8_1ep_2d_internal T2 Rainman + 2D wavelet 'internal' mode (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "sequential_blocks": true, "micro_batch_size": 8, "grad_accum": 1, "wavelet_2d_mode": "internal"}' \
    "T2_seq_M8_1ep_2d_internal: Rainman + 2D wavelet 'internal' mode (Phase 2.A; identity-init B-axis lift + per-sub-band scale + B-axis inverse; same output shape as 1D)"


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
run_ablation "T2_seq_M8_1ep_lr15 T2 Rainman + lr=0.015 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "sequential_blocks": true, "micro_batch_size": 8, "grad_accum": 1, "lr": 0.015, "min_lr": 0.0003}' \
    "T2_seq_M8_1ep_lr15: Rainman + lr=0.015 (1ep, otherwise as T2_seq_M8_1ep baseline)"


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


echo ""
echo "============================================================"
echo "=== Queue complete (2D wavelet 'internal' + Adagrad-fix + Muon LR sweep)."
echo "===   1) T2_seq_M8_1ep_2d_internal — 2D wavelet 'internal' mode (Phase 2.A, PRIORITY)"
echo "===   2) T2_seq_M8_1ep_lr15        — Rainman + lr=0.015 (1ep)"
echo "===   3) T2_Muon_lr3e-3_1ep        — Muon lr=0.003 (Path B v2)"
echo "===   4) T2_Muon_lr5e-3_1ep        — Muon lr=0.005 (Path B v2)"
echo "==="
echo "=== 2D wavelet 'internal' mode test: B-axis lift + per-sub-band scale +"
echo "===   B-axis inverse, init as identity (recovers 1D at step 0). Watch"
echo "===   trajectory vs Rainman baseline (3.66 best val). If lower by"
echo "===   > 0.0015 nats, B-axis lifting carries useful signal. If higher,"
echo "===   training is finding the B-axis path counterproductive. See"
echo "===   tools/two_d_wavelets.py and plans/two_d_wavelet_sequential_training.md."
echo "============================================================"
