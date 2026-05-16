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




# # ---- BBCE sweep (Bisected Block Context Extension) --------------------------
# # Each batch reads `block_size_compressed` tokens; the first
# # (block_size_compressed - block_size/2) tokens are chunk-averaged into
# # block_size/2 compressed slots; the last block_size/2 tokens are
# # uncompressed. Concatenated, the (B, block_size, C) result feeds the
# # standard wavelet pipeline. Loss is computed only on the last block_size/2
# # positions (the supervised "uncompressed half").
# #
# # Random sampling is used throughout (Phase 1). Sequential ordering with
# # slot caching is a Phase 2 follow-up if any of these runs shows promise.
# #
# # Sweep grid: block_size ∈ {256, 512} × block_size_compressed ∈ {65K, 262K,
# # 1M, 4M}. Ordered shortest→longest wall-clock to surface signal quickly.
# # The very long variants (4M) may need to be cancelled if too slow; results
# # from smaller variants should already settle the "does long context help?"
# # question.
# #
# # Expected wall-clock per run (random sampling, MBS=8, 5090). BBCE epoch
# # defined as one full pass over supervised positions = step count doubles
# # vs the prior bs-only formula (see Step Count Methodology in README BBCE
# # section).
# #   - bc=65K:  ~4-5h
# #   - bc=262K: ~6h
# #   - bc=1M:   ~16-20h
# #   - bc=4M:   ~60-80h  (long; may want to cancel early or run separately)
# # block_size=512 variants are roughly 10-20% slower than bs=256 counterparts.
# #
# # Comparison reference: T2 random 1ep best val 3.5881 (the standard baseline).
# # Decision: any BBCE variant must clear 3.5881 by > 0.0015 (noise threshold)
# # to be considered a win. Tied or worse → BBCE adds context that the model
# # can't usefully exploit at this scale, and we drop the direction.

# # BBCE base patch — block_size=256, levels=7, T2 mixer widths.
# BBCE_BASE_256='{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "bbce_enabled": true, "block_size": 256}'
# # BBCE base patch — block_size=512, levels=7, same mixer widths (8 entries =
# # L+1 regardless of block_size).
# BBCE_BASE_512='{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "bbce_enabled": true, "block_size": 512}'

# # BBCE-1: bc=65K (smallest, fastest signal)

# # ---- One-off: replay benchmark + generations for T2_BBCE_b256_bc65K_1ep -----
# # The 2026-05-11_17-25-07 run trained successfully (best val 3.8018) but the
# # original post-training steps failed: the old code skipped the test benchmark
# # for any BBCE run (block_size mismatch with test slices) and generation
# # crashed because the BBCE preprocessor strictly requires (B, bs_compressed)
# # input while generate.py was passing the raw 3-token prompt. Both fixed in
# # train.py (evaluate_bbce) and generate.py (pad_idx_for_bbce), so we replay
# # the post-training steps against the existing best_model.pt before resuming
# # the rest of the sweep. Neither config.json is mutated — see
# # benchmark_only_run docstring.
# # benchmark_only_run "T2_BBCE_b256_bc65K_1ep (benchmark + generations replay)" \
# #     "logs/wikitext-103_2026-05-11_17-25-07" \
# #     "T2_BBCE_b256_bc65K_1ep: replay benchmark + generations against existing checkpoint (post BBCE benchmark/generation fixes)"

# # BBCE-2: bc=131K (~125K) — rebalanced from bc=262K. The 262K variant was
# # completed (logs/wikitext-103_2026-05-12_09-08-21) but hit two measurement
# # ceilings on WT-103: (1) bc >= val_data size (251K) forces the BBCE val
# # branch to fall back to sampling from train_data, so "Best Val" stops
# # measuring val-distribution loss; (2) bc near or above test_data size
# # (287K) means most test windows require left-padding and the BPB benchmark
# # measures off-distribution (mostly-zero compressed context). bc=131K is
# # well below both ceilings (~120K val starts, 54% unpadded test windows)
# # and probes the bc-scaling story one step beyond bc=65K without methodological
# # compromise. Power-of-2 keeps g math clean (g=511 for bs=512).
# # run_ablation "T2_BBCE_b512_bc262K_1ep block_size=512, block_size_compressed=262,144 (1ep)" \
# #     "$BASE_PATCH_1EP" \
# #     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 512, \"block_size_compressed\": 262144}" \
# #     "T2_BBCE_b512_bc262K_1ep: BBCE bs=512/bc=262K (1ep, random sampling, T2 stack)"

# # BBCE-3: bc=250K — rebalanced from bc=1M. The 1M variant OOM'd at step 0
# # (compressed-half saved activations for backward exceed 32 GiB at bc=1M);
# # would require either bbce_compressed_grad=false or a streaming sum / chunked-
# # reduction refactor to fit. bc=250K is right under the val ceiling (251K)
# # so val still uses val_data (~1K starts — narrow but reasonable coverage at
# # eval_interval=250 × MBS=8 = 2000 samples per eval). BPB still 87% padded
# # at bc=250K (test_data is only 287K), which is the remaining honest limitation
# # — Best Val is the trustworthy metric for this cell; BPB needs an asterisk.
# # Bigger bc would need PG-19 or another long-form test set; flagged as the
# # natural scale-up direction once bc-scaling is confirmed at bc<=250K.

# run_ablation "T2_BBCE_b512_bc250K_1ep block_size=512, block_size_compressed=250,000 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 512, \"block_size_compressed\": 250000}" \
#     "T2_BBCE_b512_bc250K_1ep: BBCE bs=512/bc=250K (1ep, random sampling, T2 stack; rebalanced from bc=1M — right under WT-103 val ceiling; BPB 87% padded so Best Val is the trustworthy metric)"

# # BBCE-large-bs: block_size=2048 × bc ∈ {65K, 262K, 1M}. Key compute insight:
# # steps_per_epoch = corpus_size / (block_size * effective_batch). At bs=2048,
# # steps_per_epoch is 8× smaller than bs=256 (≈7,325 vs ≈58,500). Per-step
# # compute is ~8× higher (wavelet/mixer/MLP all scale with T), so total
# # compute per epoch is roughly equivalent. BUT BBCE's embedding overhead is
# # per-step, so bs=2048 sees 8× less embedding work per epoch — meaningful
# # savings at large bc.
# #
# # VRAM caveat: at bs=2048/MBS=8 the activation memory would OOM the 5090
# # (~32GB activations alone). Setting MBS=4/GA=2 maintains effective batch=8
# # while fitting VRAM (~16-20GB activations).

# run_ablation "T2_BBCE_b2048_bc65K_1ep block_size=2048, block_size_compressed=65,536 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 2048, \"block_size_compressed\": 65536, \"micro_batch_size\": 4, \"grad_accum\": 2}" \
#     "T2_BBCE_b2048_bc65K_1ep: BBCE bs=2048/bc=65K (1ep, random sampling, MBS=4/GA=2 for VRAM)"

# # Rebalanced from bc=262K and bc=1M (see comments above in the bs=512 block
# # for the WT-103 measurement-ceiling rationale). bs=2048 bcs are matched to
# # bs=512 for direct comparison along the bc axis.

# run_ablation "T2_BBCE_b2048_bc131K_1ep block_size=2048, block_size_compressed=131,072 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 2048, \"block_size_compressed\": 131072, \"micro_batch_size\": 4, \"grad_accum\": 2}" \
#     "T2_BBCE_b2048_bc131K_1ep: BBCE bs=2048/bc=131K (1ep, random sampling, MBS=4/GA=2; rebalanced from bc=262K — within WT-103 val ceiling, 54% unpadded test windows)"

# run_ablation "T2_BBCE_b2048_bc250K_1ep block_size=2048, block_size_compressed=250,000 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 2048, \"block_size_compressed\": 250000, \"micro_batch_size\": 4, \"grad_accum\": 2}" \
#     "T2_BBCE_b2048_bc250K_1ep: BBCE bs=2048/bc=250K (1ep, random sampling, MBS=4/GA=2; rebalanced from bc=1M — right under WT-103 val ceiling, BPB 87% padded so Best Val is the trustworthy metric)"


# # BBCE-middle-bs: bs=1024 × bc ∈ {65K, 131K, 250K} — interpolation row.
# # Fills in the bs axis between bs=512 and bs=2048 so we have 3 data points
# # per block_size and a reliable bc-vs-bs interpolation surface (3 bs × 3 bc
# # = 9 cells total at the headline level). MBS=8/GA=1 matches the bs=512
# # column (the unified-MBS regime); empirically activation memory scales
# # linearly with bs and bs=512/MBS=8 used ~8 GiB, so bs=1024/MBS=8 lands
# # around ~16 GiB at bc=65K and ~18-22 GiB at bc=250K — comfortably within
# # the 32 GiB 5090. If VRAM pressure surfaces, fall back to MBS=4/GA=2 to
# # match the bs=2048 column.
# run_ablation "T2_BBCE_b1024_bc65K_1ep block_size=1024, block_size_compressed=65,536 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 1024, \"block_size_compressed\": 65536}" \
#     "T2_BBCE_b1024_bc65K_1ep: BBCE bs=1024/bc=65K (1ep, random sampling, T2 stack; interpolation row between bs=512 and bs=2048)"

# run_ablation "T2_BBCE_b1024_bc131K_1ep block_size=1024, block_size_compressed=131,072 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 1024, \"block_size_compressed\": 131072}" \
#     "T2_BBCE_b1024_bc131K_1ep: BBCE bs=1024/bc=131K (1ep, random sampling, T2 stack; interpolation row, within WT-103 val ceiling, 54% unpadded test windows)"

# run_ablation "T2_BBCE_b1024_bc250K_1ep block_size=1024, block_size_compressed=250,000 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 1024, \"block_size_compressed\": 250000}" \
#     "T2_BBCE_b1024_bc250K_1ep: BBCE bs=1024/bc=250K (1ep, random sampling, T2 stack; interpolation row, right under WT-103 val ceiling, BPB 87% padded so Best Val is the trustworthy metric)"


# # ---- Non-BBCE controls at bs=1024 and bs=2048 -------------------------------
# # Without these the BBCE sweep can't separate "long compressed context helps"
# # from "larger block_size at random sampling helps". Both runs use the T2
# # architecture (levels=7, T2 mixer widths, wavelet_crawl=true, Adagrad lr=0.01),
# # random sampling, 1 epoch — only block_size varies vs T2 baseline.
# #
# # CAVEAT: under the active-stride formula, larger block_size at non-BBCE means
# # fewer steps/epoch (T2 has 58,457; bs=1024 has 14,610; bs=2048 has 7,304). All
# # three see the full corpus's worth of supervised tokens per epoch (the active-
# # stride normalization), but Adagrad gets fewer optimizer updates at large bs.
# # If these controls underperform T2 dramatically, the cause may be update-count
# # starvation rather than block_size itself; a 2-4 epoch follow-up at large bs
# # (matched step count to T2 1ep) is the disambiguation if needed.
# #
# # bs=2048 uses MBS=4/GA=2 to match the bs=2048 BBCE run's VRAM strategy.
# run_ablation "T2_nobbce_b1024_1ep block_size=1024, no BBCE (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "block_size": 1024}' \
#     "T2_nobbce_b1024_1ep: T2 stack at block_size=1024 without BBCE (1ep, control for separating long-context value from larger-window value in the BBCE sweep)"

# run_ablation "T2_nobbce_b2048_1ep block_size=2048, no BBCE (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "block_size": 2048, "micro_batch_size": 4, "grad_accum": 2}' \
#     "T2_nobbce_b2048_1ep: T2 stack at block_size=2048 without BBCE (1ep, MBS=4/GA=2 for VRAM; control for the BBCE sweep at the largest queued bs)"


# # ---- T2 random sampling + lr=0.020 (upper bracket of the LR sweep) ----------
# # Confirmation companion to T2_rand_1ep_lr15. The lr=0.015 run is tracking
# # ~0.044 nats ahead of T2 baseline (lr=0.01) at matched step — clear win.
# # This run checks whether lr=0.020 is even better or whether we've already
# # passed the optimum. Decision rule:
# #   - lr=0.020 best val < lr=0.015 best val → optimum is >=0.02; consider
# #     another sweep at lr=0.025 to find the peak.
# #   - lr=0.020 best val ≈ lr=0.015 best val (within noise) → plateau; lock
# #     in 0.015 as the new default.
# #   - lr=0.020 best val > lr=0.015 best val → past the optimum; 0.015 wins.
# # Watch for late-epoch oscillation symptomatic of too-high LR — saw this
# # with Muon at lr=0.01 (plateau/oscillation after step 5000). If lr=0.020
# # shows similar oscillation, it's likely too high for Adagrad on T2.
# run_ablation "T2_rand_1ep_lr20 T2 random sampling + lr=0.020 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.020, "min_lr": 0.0004}' \
#     "T2_rand_1ep_lr20: T2 random sampling + lr=0.020 (1ep; upper-bracket of LR sweep, follows the lr=0.015 win)"


# # BBCE-4: bc=4M REMOVED — wall-clock impractical (~30-45h per run on a 5090).
# # If the bc=65K/262K/1M sweep shows BBCE is worth pursuing, the 4M variant
# # may be worth revisiting (perhaps with sequential+caching, since random
# # sampling re-embeds the full 4M window every step).


# # ---- Adagrad-fix Rainman tests on sequential ordering (deprioritized) -------
# # Rainman 1ep (logs/wikitext-103_2026-05-10_19-36-28) trailed T2 random 1ep by
# # +0.0720 best val (3.6601 vs 3.5881). The IAV=0.1 probe was attempted but
# # failed catastrophically (val ~7.6 at step 2000 vs Rainman baseline 5.77) —
# # see Sequential Block Ordering section in README for the post-mortem.
# # IAV support has been removed from train.py / config.json as failed-experiment
# # bloat. The lr=0.015 probe remains as a mechanism-agnostic uniform LR bump.

# # Run A1: Rainman 1ep + lr=0.015 (uniform LR bump; partial fix expected to
# # recover ~30-50% of the gap by lifting the floor for accumulator-collapsed
# # parameters).

# # ---- Path B v2: tighter LR band between 0.001 (smooth/slow) and 0.01 (oscillates) ----
# # The lr=0.001 run (15:49) trained smoothly but ~10x slower than needed. The
# # lr=0.01 run (17:45) leapt ahead of Adagrad through ~step 6000 but plateaued
# # in a 4.72-4.77 oscillation band from ~step 5000. The optimal Muon LR for
# # T2 is likely between these — values that give the early-LR head start
# # without the post-warmup oscillation.

# # Run B5: Muon at lr=0.003 (3x the under-scaled 0.001).
# run_ablation "T2_Muon_lr3e-3_1ep T2 + Muon lr=0.003 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.003, "min_lr": 0.00006, "weight_decay": 0.1}' \
#     "T2_Muon_lr3e-3_1ep: T2 + Muon lr=0.003 (1ep, between under-scaled 0.001 and over-aggressive 0.01)"

# # Run B6: Muon at lr=0.005 (half the over-aggressive 0.01).
# run_ablation "T2_Muon_lr5e-3_1ep T2 + Muon lr=0.005 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "optimizer": "Muon", "lr": 0.005, "min_lr": 0.0001, "weight_decay": 0.1}' \
#     "T2_Muon_lr5e-3_1ep: T2 + Muon lr=0.005 (1ep, otherwise as B5)"


# # ---- BBCE compressed-half gradient ablation --------------------------------
# # Single confirmatory test of `bbce_compressed_grad=false`: torch.no_grad
# # wraps the chunked embedding lookup + mean-pool for the compressed half,
# # halting backprop into the embedding table for tokens that appear there.
# # The embedding table still learns from uncompressed-half appearances; the
# # wavelet/mixer/MLP still learn to USE compressed slots — they just don't
# # get to tell the embedding table to be a better mean-pool basis.
# #
# # Expected speedup: ~1.3-1.5x on backward (per Gemini's analysis, unverified).
# # Expected quality: marked decrease vs the bbce_compressed_grad=true variant
# # of the same cell (bs=512 / bc=65K), but still likely better than the T2
# # baseline (the compressed half being treated as cheap "additional context"
# # is the working theory). Decision logic:
# #   - Within noise of the gradient-on variant: bbce_compressed_grad=false
# #     becomes a viable speed/quality tradeoff for compute-bound regimes.
# #   - Notable regression (Δ best val > +0.01 vs gradient-on): the embedding
# #     table's role as a learned mean-pool basis matters; keep gradients on
# #     by default.
# #   - Catastrophic (Δ best val > +0.05): bbce_compressed_grad=false is not
# #     usable; the embedding table being trained for mean-pool quality is
# #     load-bearing.
# #
# # Picked bs=512 / bc=65K as the test cell because it's the cheapest of the
# # headline (per-batch-supervision-matched) BBCE configs.
# run_ablation "T2_BBCE_b512_bc65K_1ep_ngrad bbce_compressed_grad=false (1ep)" \
#     "$BASE_PATCH_1EP" \
#     "{\"levels\": 7, \"per_scale_mixer_widths\": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], \"wavelet_crawl\": true, \"bbce_enabled\": true, \"block_size\": 512, \"block_size_compressed\": 65536, \"bbce_compressed_grad\": false}" \
#     "T2_BBCE_b512_bc65K_1ep_ngrad: BBCE bs=512/bc=65K + bbce_compressed_grad=false (1ep, single confirmatory A/B vs the gradient-on counterpart from the main sweep)"


# echo ""
# echo "============================================================"
# echo "=== PROJECT CLOSED 2026-05-13. The queue below was the planned sweep at"
# echo "===   shutdown; not all entries ran. Final results table and findings"
# echo "===   summary are in the README BBCE section. Headline outcomes:"
# echo "==="
# echo "=== Completed BBCE cells (best val / BPB sliding):"
# echo "===   - bs=256 / bc=65K  (g=511): 3.5725 / 1.1551    Δ best val −0.0156 (win, 10x noise)"
# echo "===   - bs=512 / bc=65K  (g=255): 3.5922 / 1.1530    Δ best val +0.0041 (within noise)"
# echo "===   - bs=512 / bc=131K (g=511): 3.6364 / 1.1567    Δ best val +0.0483 (regression)"
# echo "===   - bs=512 / bc=250K (g=975): 3.6247 / 1.1593 †  Δ best val +0.0366 (regression; 87% padded BPB)"
# echo "===   - bs=2048 / bc=65K (g=63):  3.6541 / 1.1706    Δ best val +0.0660 (clear regression)"
# echo "===   - bs=2048 / bc=65K rerun:   3.6552 / 1.1713    reproducibility check, < noise"
# echo "==="
# echo "=== Did not complete before shutdown:"
# echo "===   - bs=2048 / bc=131K (in progress at ~14% trajectory)"
# echo "===   - bs=1024 column (3 cells, all not run)"
# echo "===   - bs=2048 / bc=250K (not run)"
# echo "===   - bs=1024 and bs=2048 non-BBCE controls (not run)"
# echo "===   - T2_rand_1ep_lr20, Muon LR sweep, compressed_grad=false A/B (not run)"
# echo "==="
# echo "=== Queue (planned, partial completion):"
# echo "===   1)  T2_BBCE_b512_bc131K_1ep     — BBCE bs=512/bc=131K  COMPLETED"
# echo "===   2)  T2_BBCE_b512_bc250K_1ep     — BBCE bs=512/bc=250K  COMPLETED"
# echo "===   3)  T2_BBCE_b2048_bc65K_1ep     — BBCE bs=2048/bc=65K  COMPLETED (x2)"
# echo "===   4)  T2_BBCE_b2048_bc131K_1ep    — BBCE bs=2048/bc=131K IN PROGRESS at shutdown"
# echo "===   5)  T2_BBCE_b2048_bc250K_1ep    — BBCE bs=2048/bc=250K NOT RUN"
# echo "===   6)  T2_BBCE_b1024_bc65K_1ep     — BBCE bs=1024/bc=65K  NOT RUN"
# echo "===   7)  T2_BBCE_b1024_bc131K_1ep    — BBCE bs=1024/bc=131K NOT RUN (matched-g diagonal cell)"
# echo "===   8)  T2_BBCE_b1024_bc250K_1ep    — BBCE bs=1024/bc=250K NOT RUN"
# echo "===   9)  T2_nobbce_b1024_1ep         — bs=1024 non-BBCE control NOT RUN"
# echo "===   10) T2_nobbce_b2048_1ep         — bs=2048 non-BBCE control NOT RUN"
# echo "===   11) T2_rand_1ep_lr20            — T2 random + lr=0.020 NOT RUN"
# echo "===   12) T2_Muon_lr3e-3_1ep          — Muon lr=0.003 (Path B v2) NOT RUN"
# echo "===   13) T2_Muon_lr5e-3_1ep          — Muon lr=0.005 (Path B v2) NOT RUN"
# echo "===   14) T2_BBCE_b512_bc65K_1ep_ngrad — bbce_compressed_grad=false A/B NOT RUN"
# echo "==="
# echo "=== BBCE headline grid (block_size × bc, 3 × 3 = 9 cells, completed +"
# echo "===   queued). The 9 cells let us interpolate the bc-scaling surface"
# echo "===   along the bs axis with three data points per block_size."
# echo "===   bc=65K cells: completed (bs=256, bs=512) or running (bs=2048)."
# echo "===   bs=512 / bs=2048 columns have bc=131K and bc=250K queued; bs=1024"
# echo "===   column is the new interpolation row."
# echo "==="
# echo "=== Completed (commented out above):"
# echo "===   - T2_rand_1ep_lr15             — T2 + lr=0.015, Δ best val −0.0536 vs T2 baseline (win)"
# echo "===   - T2_BBCE_b256_bc65K_1ep       — completed: best val 3.5725, BPB 1.1551 (Δ +0.0010 vs T2, tied within noise)"
# echo "===   - T2_BBCE_b512_bc65K_1ep       — in progress / completed (the bs=512 headline bc=65K cell)"
# echo "==="
# echo "=== Dropped (cost vs information not worth it; bs=512 / bs=2048 columns"
# echo "===   already cover the bc-scaling story at more interesting block sizes):"
# echo "===   - T2_BBCE_b256_bc262K_1ep      — ~6h saved"
# echo "===   - T2_BBCE_b256_bc1M_1ep        — ~16-20h saved"
# echo "==="
# echo "=== Rebalanced (WT-103 measurement-ceiling: val_data=251K caps bc <= 251K"
# echo "===   before val falls back to train_data; test_data=287K makes BPB"
# echo "===   heavily padded above bc=200K-ish; bc=1M also OOMs at step 0):"
# echo "===   - T2_BBCE_b512_bc262K_1ep      — completed methodologically compromised; replaced by bc=131K"
# echo "===   - T2_BBCE_b512_bc1M_1ep        — OOM at step 0; replaced by bc=250K"
# echo "===   - T2_BBCE_b2048_bc262K_1ep     — preemptively replaced by bc=131K"
# echo "===   - T2_BBCE_b2048_bc1M_1ep       — preemptively replaced by bc=250K"
# echo "===   For bc > 256K, PG-19 (~28M test tokens, long-form) is the natural"
# echo "===   next test set — virtually zero padding and the long-form structure"
# echo "===   matches BBCE's value proposition. Filed as future direction."
# echo "==="
# echo "=== BBCE step count: full-corpus supervised coverage per epoch under the"
# echo "===   new active-stride formula (block_size/2 for BBCE, block_size"
# echo "===   otherwise). Per-epoch step count is ~2x the prior bs-only formula."
# echo "==="
# echo "=== bs=2048 runs: MBS=4/GA=2 (effective batch unchanged at 8) to fit VRAM"
# echo "===   at T=2048. steps_per_epoch=14,650 (4x fewer than bs=256 BBCE's"
# echo "===   116,914) means BBCE embedding overhead is 4x less per epoch —"
# echo "===   meaningful at large bc. Also a useful axis for studying"
# echo "===   block-size effect on BBCE quality."
# echo "==="
# echo "=== Removed (wall-clock impractical at ~30-45h per run):"
# echo "===   - T2_BBCE_b256_bc4M_1ep / T2_BBCE_b512_bc4M_1ep"
# echo "===   May revisit with sequential+caching if smaller variants show promise."
# echo "==="
# echo "=== BBCE decision rule: any variant must clear T2 random 1ep best val"
# echo "===   (3.5881) by > 0.0015 (noise threshold) to be considered a win."
# echo "===   Tied or worse → BBCE adds context that the model can't usefully"
# echo "===   exploit at this scale, and we drop the direction."
# echo "==="
# echo "=== Shelved (commented out, code preserved):"
# echo "===   - T2_seq_M8_1ep_2d_internal  — Δ +0.0090 vs Rainman, +14% wall-clock"
# echo "===   - T2_seq_M8_1ep_2d_subband   — Δ +0.0598 vs Rainman, +30% wall-clock"
# echo "===   - T2_seq_M8_1ep_lr15         — completed: best val 3.6231 (~50% gap recovery)"
# echo "==="
# echo "=== Two architectural shots at finding cross-batch lift benefit:"
# echo "===   'internal' — B-axis lift internal to wavelet (sub-band scaling +"
# echo "===                inverse lift); same (approx, details) output shape."
# echo "===   'subband' — 4 sub-bands per joint level exposed to per-band mixers"
# echo "===                (mixer count 14 vs 1D's 8); per-band specialization."
# echo "==="
# echo "=== Watch vs Rainman baseline (3.66 best val). If lower by > 0.0015 nats,"
# echo "===   that mode's architectural premise carries useful signal."
# echo "=== See tools/two_d_wavelets.py and plans/two_d_wavelet_sequential_training.md."
# echo "============================================================"


# ---- Mixer recurrence sweep (T3 + N x K) ------------------------------------
# Tests the "mixer-only recurrence" architectural feature: apply the per-scale
# mixer stack N times in a row inside one wavelet+FWHT pipeline before
# reconstruction. Wavelet decompose/reconstruct and FWHT/iFWHT are inverses,
# so the only thing that repeats is the mixer body.
#
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
#
# All runs build on T3 (= T2 + lr=0.015 + min_lr=0.0003). Reference row in
# the README "Recurrence (Mixer Only)" section is the T3 baseline at N=K=1
# (best val 3.5345). Decision rule: a recurrence variant must clear T3 best
# val (3.5345) by > 0.0015 (noise threshold) to be considered a win.
#
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
#
# On A5000 multiply each wall-clock by roughly 1.9x.
#
# Sweep ordered cost-ascending so the schedule can be cancelled midstream if
# early canaries (N=2) already diverge or plateau.

# Run R1: N=2, K=1 (shared) — cheapest canary.
run_ablation "T3_recur_N2_K1_1ep T3 + mixer recurrence N=2 K=1 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 2, "mixer_recurrence_distinct_mixer_count": 1}' \
    "T3_recur_N2_K1_1ep: T3 + mixer recurrence (N=2 steps, K=1 shared bank; cheapest canary at ~1.55x wall-clock)"

# Run R2: N=2, K=2 (full distinct at N=2).
run_ablation "T3_recur_N2_K2_1ep T3 + mixer recurrence N=2 K=2 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 2, "mixer_recurrence_distinct_mixer_count": 2}' \
    "T3_recur_N2_K2_1ep: T3 + mixer recurrence (N=2 outer x K=2 banks = 4 apps, +1x mixer params vs T3)"

# Run R3: N=5, K=1 (shared).
run_ablation "T3_recur_N5_K1_1ep T3 + mixer recurrence N=5 K=1 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 5, "mixer_recurrence_distinct_mixer_count": 1}' \
    "T3_recur_N5_K1_1ep: T3 + mixer recurrence (N=5 steps, K=1 shared bank; ~3.2x wall-clock)"

# Run R4: N=5, K=2 (cyclic — 2 banks rotating across 5 steps).
run_ablation "T3_recur_N5_K2_1ep T3 + mixer recurrence N=5 K=2 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 5, "mixer_recurrence_distinct_mixer_count": 2}' \
    "T3_recur_N5_K2_1ep: T3 + mixer recurrence (N=5 outer x K=2 banks = 10 apps, cyclic m0,m1 repeated 5x; +1x mixer params)"

# Run R6: N=10, K=1 (shared, substantive depth).
run_ablation "T3_recur_N10_K1_1ep T3 + mixer recurrence N=10 K=1 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 10, "mixer_recurrence_distinct_mixer_count": 1}' \
    "T3_recur_N10_K1_1ep: T3 + mixer recurrence (N=10 steps, K=1 shared bank; ~5.95x wall-clock)"

# Run R7: N=20, K=1 (shared, deep). Conditional canary for "does it keep
# climbing?"; if N=10 already plateaus/regresses, skip.
run_ablation "T3_recur_N20_K1_1ep T3 + mixer recurrence N=20 K=1 (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lr": 0.015, "min_lr": 0.0003, "mixer_recurrence_steps": 20, "mixer_recurrence_distinct_mixer_count": 1}' \
    "T3_recur_N20_K1_1ep: T3 + mixer recurrence (N=20 steps, K=1 shared bank; ~11.45x wall-clock, ~21h)"


echo ""
echo "============================================================"
echo "=== MIXER RECURRENCE QUEUE (T3 + N x K sweep, 1ep each)"
echo "==="
echo "=== All runs build on T3 (= T2 + lr=0.015). Reference: T3 baseline"
echo "===   best val 3.5345 (BPB 1.1362) — see New T3 Baseline section in"
echo "===   README. Decision rule: > 0.0015 nat improvement on best val."
echo "==="
echo "=== Queue (cost-ascending on 5090; multiply by ~1.9x for A5000):"
echo "===   1) T3_recur_N2_K1_1ep    — N=2  K=1  (2 apps,  shared)   ~2.8h"
echo "===   2) T3_recur_N2_K2_1ep    — N=2  K=2  (4 apps,  distinct) ~4.9h, +1x mixer params"
echo "===   3) T3_recur_N5_K1_1ep    — N=5  K=1  (5 apps,  shared)   ~5.9h"
echo "===   4) T3_recur_N5_K2_1ep    — N=5  K=2  (10 apps, cyclic)   ~10.9h, +1x mixer params"
echo "===   5) T3_recur_N10_K1_1ep   — N=10 K=1  (10 apps, shared)   ~10.9h"
echo "===   6) T3_recur_N20_K1_1ep   — N=20 K=1  (20 apps, shared)   ~21h (cancel if N=10 plateaus)"
echo "==="
echo "=== Total budget if all run on 5090: ~66h (~2.8 days). On A5000: ~125h (~5.2 days)."
echo "============================================================"
