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
    # Stage ONLY run outputs (logs/) — NOT the scripts or README. A blanket
    # `git add .` makes the pod commit runs.sh/runs_6000.sh/README, and with
    # `-X theirs` on pull the pod's copy clobbers your workstation edits (the
    # "local changes would be overwritten" friction). Scripts/README flow one-way:
    # workstation -> GitHub -> pod.
    git add logs/ || true
    git commit --no-edit -m "${MSG}" || true
    git pull --no-rebase --no-edit -X theirs || true
    git push || true

    # Mirror the working tree to S3 after every push so a wiped volume disk loses
    # nothing (logs + best_model.pt checkpoints + .cache tokenized tensors). sync is
    # incremental (only changed/new files upload); hf_cache excluded (large, re-
    # downloadable). Runs even if the push above failed — S3 is the backup of record.
    # NEVER add --delete: on a fresh volume-disk pod the local tree holds ONLY the
    # current run, so --delete would erase every other run from the S3 archive.
    aws s3 sync /workspace/EXARCH s3://exarch-ai-model/EXARCH --exclude "hf_cache/*" \
        || echo "[runs.sh] WARNING: aws s3 sync failed — S3 backup NOT updated this run"
}

# Resume a crashed/preempted run in place (RunPod machine failures, SSH/tmux
# deaths, OOM kills). Usage:  resume_run logs/wikitext-103_2026-07-13_...
# Everything is read from the run's own config.json (source of truth — same
# rule as benchmark_only); the temp config only carries the pointer key.
# Requires last_checkpoint.pt in the run dir: train.py writes it at every
# epoch end automatically, plus every checkpoint_interval_steps global steps
# when that key is set (recommended for runs with hours-long epochs — the
# interval bounds crash loss; epoch-end alone bounds it to one epoch).
# Resume is exact: model + optimizer accumulators + scaler + RNG streams are
# restored, so the continued run is step-for-step the run that never died.
resume_run() {
    local RUN_DIR="$1"
    local TITLE="${2:-resume $(basename "$RUN_DIR")}"
    echo "============================================================"
    echo "=== Resume: ${TITLE}"
    echo "============================================================"
    build_run_config "{\"resume_run_dir\": \"${RUN_DIR}\"}"
    python train.py --config "$TMP_CFG"
    git_commit_push "resume: ${TITLE}"
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
    # The EXACT run dir is passed in by run_ablation (the dir train.py just created).
    # Don't GUESS it — name-sort/mtime both misfire in TANDEM: a git pull brings the
    # other pod's newer run into logs/ and generate.py targets the wrong one (observed
    # 2026-06-16 with mtime, then 2026-06-17 with name-sort — the 5090's L=5 missed
    # its own generations). $1 is authoritative; name-sort below is a fallback only.
    local LATEST_RUN="$1"
    if [ -z "$LATEST_RUN" ]; then
        LATEST_RUN=$(ls -d logs/wikitext-103_*/ 2>/dev/null | sort | tail -1)
    fi
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
    # Snapshot run dirs BEFORE train.py so we can hand run_inference_vram_latest the
    # EXACT dir it creates (set difference), instead of guessing "latest" — which
    # misfires in tandem when a git pull brings the other pod's newer run into logs/.
    local DIRS_BEFORE; DIRS_BEFORE=$(ls -d logs/wikitext-103_*/ 2>/dev/null)
    python train.py --config "$TMP_CFG"
    local TRAIN_EXIT=$?
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "[runs.sh] train.py exited with code $TRAIN_EXIT; continuing to next ablation"
    fi
    local RUN_DIR
    if [ -z "$DIRS_BEFORE" ]; then
        RUN_DIR=$(ls -d logs/wikitext-103_*/ 2>/dev/null | tail -1)
    else
        RUN_DIR=$(ls -d logs/wikitext-103_*/ 2>/dev/null | grep -vxF "$DIRS_BEFORE" | tail -1)
    fi
    run_inference_vram_latest "${RUN_DIR%/}"
    git_commit_push "${COMMIT_MSG}"
}

pause_for_decision() {
    # Coordinate-descent gate: after a dropout pair has run AND been pushed,
    # halt so the operator can read the two results and set the winning value
    # for that dropout type. The value is read from the terminal and assigned
    # to the named carried-forward variable IN MEMORY (editing runs.sh on disk
    # would NOT work — bash already executed the DO_* assignment lines and won't
    # re-read them). Empty input keeps the current value.
    #   $1 = variable name to set (e.g. DO_PROJ); empty = final pause, no var.
    #   $2 = message.
    local VARNAME="$1"
    local MSG="$2"
    echo ""
    echo "############################################################"
    echo "### COORDINATE-DESCENT PAUSE"
    echo "### ${MSG}"
    if [ -n "$VARNAME" ]; then
        echo "### Current ${VARNAME}=${!VARNAME}"
    fi
    echo "### Both results above are pushed (read them, then decide)."
    echo "############################################################"
    if [ "${DROPOUT_AUTO:-0}" = "1" ]; then
        echo "[runs.sh] DROPOUT_AUTO=1 — skipping pause, keeping ${VARNAME:-N/A}=${!VARNAME:-}."
        return
    fi
    if [ -z "$VARNAME" ]; then
        echo "### Press Enter to finish (or Ctrl-C)."
        read -r _ </dev/tty || true
        return
    fi
    # Accept any of: "0.2" | "DO_EMB=0.2" | "dropout_emb=0.2" |
    # "dropout_embedding=0.2". Strip everything up to and including the last
    # '=', trim whitespace, and require the result to be a bare number — so a
    # mistyped key never gets assigned as a string and corrupts the JSON.
    while true; do
        local RAW=""
        printf "### Winning value for %s (e.g. 0.2, or DO_EMB=0.2; blank = keep %s): " \
            "$VARNAME" "${!VARNAME}" > /dev/tty
        read -r RAW </dev/tty || { echo "[runs.sh] no input; keeping ${VARNAME}=${!VARNAME}."; break; }
        # blank -> keep current
        if [ -z "${RAW//[[:space:]]/}" ]; then
            echo "[runs.sh] ${VARNAME} kept at ${!VARNAME}."
            break
        fi
        local VAL="${RAW##*=}"          # drop any "key=" prefix (last '=' wins)
        VAL="${VAL//[[:space:]]/}"      # strip all whitespace
        # Validate: integer or decimal (optionally signed), e.g. 0.2 / .09 / 1
        if [[ "$VAL" =~ ^-?([0-9]+(\.[0-9]*)?|\.[0-9]+)$ ]]; then
            printf -v "$VARNAME" '%s' "$VAL"
            echo "[runs.sh] ${VARNAME} set to ${!VARNAME}."
            break
        fi
        echo "[runs.sh] '${RAW}' is not a valid number — try again (or blank to keep)." > /dev/tty
    done
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

# CARRIED-FORWARD CHOSEN VALUES — edit each at its pause before continuing:
DO_EMB=0.18    # emb pair done: 0.18 BPB 1.1307 (best of 3 on BPB) vs 0.20 1.1311, 0.22 1.1309 — all within noise; 0.18 chosen (lowest BPB)
DO_PROJ=0.09   # proj pair done: 0.09 BPB 1.1304 < 0.10 1.1307 < 0.11 1.1316 (monotonic; 0.09-vs-0.10 within noise but consistent). 0.09 chosen; edge of range — probe lower at higher L
DO_MIX=0.09    # mix pair done: 0.09 BPB 1.1295 < 0.10 1.1304 < 0.11 1.1311 (monotonic; 0.09-vs-0.10 = 0.0009, ~noise floor — strongest pair yet). 0.09 chosen; edge of range — probe lower at higher L
DO_MLP=0.10    # mlp pair done: incumbent 0.10 BPB 1.1295 < 0.09 1.1305 & 0.11 1.1309 (interior min on BPB — kept). NOTE val disagrees (0.09 best on val, monotonic) — metric split; revisit at higher L. NOT an edge-winner
DO_LM=0.216    # lm_head pair done: 0.216 BPB 1.1285 < 0.240 1.1295 < 0.264 1.1305 (monotonic, both metrics agree; 0.216-vs-0.240 = 0.0010 ~floor). 0.216 chosen; edge-winner ↓. FINAL STACK best point: 1.1285 (-0.0026 vs T4)
DO_COMMON='"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450'

#############
###########
########### RUNS ##

# ---- FwPKM, PROPERLY (2026-07-28) -----------------------------------------
# The module was rewritten from scratch against arXiv:2601.00671 sections 3.2-3.5.
# What was there before implemented roughly a third of the paper and none of the
# part the paper is about: retrieval only, in the MLP slot AFTER the token mixer
# (the paper places it BEFORE), with the fast-weight path living in generate.py
# behind a default-off flag. So all 277 historical fwpkm_enabled runs, plus
# FWPKM_Micro, measured a PKM. "FwPKM never contributed positively" was never
# evidence about memory.
#
# ADDED, all previously absent: value path v_t, sigmoid gate g_t and the gated
# residual o = g*vhat + (1-g)*v, output projection, four RMSNorms, IDW scoring
# (-log(eps + ||q-K||^2) instead of dot product), chunk-wise apply-then-update,
# memorization loss with read-count gradient aggregation, addressing loss on
# chunk-marginal usage entropy, lookahead targets (q_t paired with v_{t+1}),
# and z-scored targets.
#
# THREE DEVIATIONS, stated rather than buried: updates are DETACHED (the paper
# calls them local objectives; second-order TTT is out of budget); V_init/K_init
# are LEARNABLE with V_init zero (the paper is silent on initial fast weights,
# LaCT makes them learnable, and zero V_init gives identity-at-init WITH a live
# gradient - the LoRA pattern that fixes the alpha=0 deadlock); state is per
# batch lane and persists across steps under sequential_blocks.
#
# EVAL PROTOCOL DELIBERATELY UNCHANGED (Ramon's call): fast weights reset per
# window at benchmark, exactly as decompose_bypass_cross_window does. So the
# memory accumulates across the whole corpus during training but must prove
# itself WITHIN a 256-token window at eval. That asymmetry is the experiment:
# can gradients alone select keys/values general enough to survive the reset.
# run_ablation "SEQ0_Micro_C256_5ep Sequential-order control - C=256, no memory"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "sequential_blocks": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216}'     "SEQ0 Micro: THE CONTROL FWPKM2 IS JUDGED AGAINST, and it exists because sequential ordering is not free - MEMORY.md records a ~0.14 nat sequential-vs-random penalty on the old T2 config with two proposed Adagrad fixes never run, and it has NEVER been measured in the current architecture. Comparing FwPKM-sequential against the random-order Micro baseline (1.1330) would conflate two changes and the ordering penalty would swamp any memory effect. Also the first same-architecture measurement of what sequential ordering costs"
run_ablation "FWPKM2_Micro_C256_5ep FwPKM Micro v2 - paper-faithful, pre-mixer, online updates"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": true, "gradient_checkpointing": false, "layers": 10, "C": 256, "sequential_blocks": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "fwpkm_num_keys": 5625, "fwpkm_top_k": 32, "fwpkm_heads": 1, "fwpkm_chunk_size": 64, "fwpkm_persist_state": false, "fwpkm_second_order": true, "fwpkm_compress_value": "zero_mean", "fwpkm_lookahead": 1, "fwpkm_lr": 1.0, "fwpkm_addr_loss_weight": 1.0, "fwpkm_idw_eps": 0.001, "fwpkm_layers": [2, 6]}'     "FWPKM2 Micro: the FIRST run that actually tests FwPKM. Every prior FwPKM number in this repo measured a PKM - retrieval only, sited after the mixer, fast-weight path off. Now paper-faithful: pre-mixer sublayer, gated residual, IDW scoring, chunked apply-then-update, memorization loss with read-count aggregation, addressing loss on chunk marginals, lookahead targets, z-scored targets. 41,353,941 params (+66.9%), 2.76 GB fast state at MBS=48. JUDGE VS SEQ0, not vs the random-order baseline; width-law bar 1.0905. Causality smoke-verified exact (perturbing token t moves nothing before t). If it underperforms, note 73% of value rows are rewritten per window - store SIZE is the first lever, and num_keys is nearly free since the state is not parameters"
#
# RETENTION ARM. The governing ratio is not tokens-per-parameter but
#     retention (windows) ~ num_keys / (T * top_k)
# because writes per window = T * top_k is fixed by GEOMETRY, independent of store size.
# At the paper defaults that is 5,625 / (256*32) = 0.69 — WE RETAIN LESS THAN ONE WINDOW,
# so the store is fully overwritten before it can carry anything across a boundary. That
# is the mechanism behind the 73% row churn measured on a single window.
#
# NOTE the tokens-per-parameter framing does NOT apply and points the wrong way: a slot is
# a row of C=256 values, so 5,625 keys is 1.44M parameters per layer and 1.73 tokens per
# parameter — already ~11x MORE parameterised than Chinchilla's 20:1. But Chinchilla governs
# how much data it takes to TRAIN parameters by gradient descent over a corpus; this store
# is REWRITTEN online by a local loss. Working-set question, not a training-efficiency one.
#
# WHY NOT SIMPLY 20-25x THE STORE: fast state is PER BATCH LANE, scaling as
# num_keys*C*batch*layers, so MBS=48 multiplies it 48-fold. 25x (375^2 = 140,625) needs
# 69 GB fp32 / 34.6 GB fp16 on a 32 GB card — impossible at any precision. 137^2 = 18,769
# is the largest store that fits (50.4M params, 4.6 GB fp16 state).
#
# TOP_K IS THE FREE HALF OF THE LEVER. It divides write volume exactly as num_keys
# multiplies capacity, at zero memory cost: 18,769 keys at top_k=8 buys the SAME retention
# as 81,796 keys at top_k=32, for 4.6 GB instead of 20.1 GB. The cost is retrieval
# richness — 8 blended rows per query instead of 32 — which is a real tradeoff, not a
# free lunch, and is the thing to suspect if this arm loses despite more retention.
#
# fwpkm_state_fp16 defaults true and is verified equivalent: |dV|max 9.4727e-02 vs
# 9.4711e-02 fp32, identical loss to 4 dp. Updates are unit-step rewrites of magnitude
# ~1e-2, three orders above fp16 resolution. Set false to rule precision out if a
# large-store run behaves strangely.
#
# IF MORE STORE HELPS, the follow-up is LOW-RANK v_init (LaCT's trick: v_init = L*R with
# rank 32), which decouples store size from parameter count entirely - 18,769 keys would
# cost 0.61M/layer instead of 4.8M - and makes the width law winnable. Do not build that
# until this arm says store size matters.
run_ablation "FWPKM3_Micro_C256_retention_5ep FwPKM Micro v3 - 13x retention"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": true, "gradient_checkpointing": false, "layers": 10, "C": 256, "sequential_blocks": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "fwpkm_num_keys": 18769, "fwpkm_top_k": 8, "fwpkm_heads": 1, "fwpkm_chunk_size": 64, "fwpkm_persist_state": false, "fwpkm_second_order": true, "fwpkm_compress_value": "zero_mean", "fwpkm_lookahead": 1, "fwpkm_lr": 1.0, "fwpkm_addr_loss_weight": 1.0, "fwpkm_idw_eps": 0.001, "fwpkm_layers": [2, 6]}'     "FWPKM3 Micro: num_keys 5,625->18,769 AND top_k 32->8, together raising retention 0.69 -> 9.16 windows (~176 -> ~2,346 tokens), a 13x increase. Read FWPKM3-MINUS-FWPKM2; the width law at 75.16M demands 1.0410 and no store-scaling arm can clear it while v_init is dense. Deliberately confounds two knobs to maximise effect size on a thin budget - the disambiguation arm (5,625 keys at top_k=8, retention 2.75, ZERO extra memory) is free to run afterwards if this wins"

run_ablation "MOEA_Micro_C256_5ep Block-MoE Micro - E=4 full-block experts, top-2"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "shared_lifting_weights": false, "block_moe_enabled": true, "block_moe_experts": 4, "block_moe_topk": 2, "block_moe_aux_weight": 0.01, "block_moe_layers": [6, 8]}'     "MOEA Micro: vs Micro-untied = does per-token routing help at all. NOTE MoE is much cheaper at Micro - the tied LM head is 71 pct of compute at C=256 vs 55 pct at C=512, so E=4 costs 1.87x the baseline here against 2.35x at Mini"
#
# COST FIX 2026-07-29. MOEA at E=4 across ALL layers measured 4.36 s/it against SEQ0's
# 0.284 — 15x, i.e. ~59h and ~$59 for one screening arm. Block-MoE replicates the whole
# block per expert AND runs every expert (dense by design), so all-layers placement
# multiplies cost by E across the entire stack: 40 block-forwards per step instead of 10.
# Restricting to block_moe_layers [6, 8] — recommended practice is to set to later layers for higher expert differentiation, but NOT the last layer. This gives 16
# forwards/step, and dropping gradient_checkpointing (only 2 layers are expanded now, so
# activation pressure is far lower than the Mini config it was set for) takes it to ~18h.
# E=4 is KEPT deliberately: at E=2 a top-2 router selects both experts every time, which is
# a weighted blend, not routing. If this OOMs, set gradient_checkpointing true (+33%).
run_ablation "MLR1_Micro_C256_5ep Multiresolution ladder Micro - E=4 coarse-only experts"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "shared_lifting_weights": false, "block_moe_enabled": true, "block_moe_experts": 4, "block_moe_topk": 2, "block_moe_aux_weight": 0.01, "block_moe_scale_ladder": true, "block_moe_layers": [6, 8]}'     "MLR1 Micro: the ONLY valid A/B is vs MOEA_Micro (identical params and compute, the ladder is the sole difference). The MoE capacity effect - the thing most likely to inflate at small C - appears in BOTH arms and cancels. What does NOT cancel is the ladder information cost, which should bite HARDER at C=256 where there is less redundancy, so a WIN here is strong evidence while a LOSS stays weak"
run_ablation "PP1_Micro_C256_5ep Prime-power Micro - max=11, cap=128, crawl ON"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "prime_power_wavelet_basis_max": 11, "prime_power_dilation_cap": 128}'     "PP1 Micro: 18-level prime-power union ladder. Judge PP1-minus-PP2, not PP1 vs baseline - the subtraction answers the crawl-redundancy question and is far more robust to width than absolute BPB"
run_ablation "PP2_Micro_C256_5ep Prime-power Micro - max=11, cap=128, crawl OFF"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "prime_power_wavelet_basis_max": 11, "prime_power_dilation_cap": 128}'     "PP2 Micro: crawl-OFF twin. PP1-minus-PP2 = crawl redundancy of the prime rungs, answered directly"
run_ablation "MOMA_Micro_C256_5ep Mixture-of-Mixers Micro - E=4, top-2"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": null, "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "mixer_mom_enabled": true, "mixer_mom_experts": 4, "mixer_mom_topk": 2, "mixer_mom_aux_weight": 0.01}'     "MOMA Micro: the one arm that SAVES params (at Mini it is 57.08M vs D0 72.89M), so a tie is a win. Watch expert-usage collapse via aux magnitude"
run_ablation "PROJ_Micro_C256_5ep Micro CALIBRATION - skip_proj_out OFF"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": false, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216}'     "PROJ Micro (Ramon 2026-07-27): the run that measures HOW MUCH MICRO LIES. skip_proj_out scored +0.0115 at C=100 (a failure) and -0.0079 at C=1024 (the fully-spectral headline) - a MEASURED SIGN FLIP. Re-running it at C=256 locates where that flip happens and calibrates every other Micro verdict here. If skip_proj_out is already winning at C=256, Micro is a trustworthy prioritiser; if it is still losing, Micro systematically punishes this whole class of simplification and Micro negatives mean very little"

#
# ============================================================================
# MICRO BASELINE RESULT (2026-07-28, logs/wikitext-103_2026-07-27_21-37-29)
# BPB 1.1330 / PPL 34.4445 / sliding avg loss 3.5393 / best val 3.5394 (ep 5)
# @ 24,777,931 params, 4.90h. lr=0.15 SURVIVED all 5 epochs — the highest
# absolute LR this project has run; the 0.075 fallback is not needed anywhere.
# Timing prediction landed: ~5.0h projected, 4.90h measured; intervals
# 178/180/186s vs ~185s projected. The eval fix delivered its 2x.
#
# *** THE WIDTH LAW IS NOW MEASURED AT MBS=48, AND IT IS STEEPER ***
# Predicted 1.1137 from the -0.065/e-fold slope; measured 1.1330 (miss +0.0193).
# That slope was fitted ENTIRELY on MBS=8 runs. With two same-batch anchors
# (D0 72.89M/1.0436 and Micro 24.78M/1.1330) the MBS=48 slope is -0.0829,
# i.e. 27% STEEPER. Every width-law kill rule written before today used -0.065
# and was therefore TOO LENIENT for arms larger than D0. Corrected bars:
#     FWP1      118.37M   1.0121 -> 1.0034
#     PP1/PP2   134.86M   1.0036 -> 0.9926
#     MOMA       57.08M   1.0595 -> 1.0639   (smaller than D0, bar moves the other way)
#     MOEA/MLR1 479.00M   0.9212 -> 0.8876
#     SB4       139.08M   1.0016 -> 0.9901   (measured 1.0198, so it MISSES by 0.0298,
#                                             not 0.0182 — verdict direction unchanged)
# Micro-tier bars, from the Micro anchor at -0.0829:
#     Micro-untied 41.36M 1.0905 | FWPKM_Micro 40.03M 1.0933
#     MOEA/MLR1_Micro 126.85M 0.9977 | MOMA_Micro 20.81M 1.1475
#     PROJ_Micro 25.44M 1.1308 | SB1_Micro 21.90M 1.1432
#     SB3_Micro 20.65M 1.1481 | SB0/SB2_Micro 24.78M 1.1330
#
# CAVEAT, AND IT IS NOT SMALL: this is a TWO-POINT slope spanning C=256->512
# only. The true law is curved in log-params, so a local slope measured going
# DOWN from D0 overstates the gain available going UP from it — which makes the
# corrected bars for the LARGE arms (MOEA/MLR1 at 0.8876) pessimistic by an
# unknown amount. K5 (C=768) would give the third MBS=48 anchor that fixes this.
#
# AND A COMPETING EXPLANATION FOR THE MISS. Against the MBS=8 curve interpolated
# at its own params, D0 sits ON the curve (-0.0020) but Micro sits BELOW it
# (+0.0097). MBS=8 runs take 292,290 optimizer steps; MBS=48 takes 48,710 — 6x
# fewer updates. So part of the "steeper slope" may be an UPDATE-COUNT deficit
# that hits small models hardest, not width at all. If so the slope is not a
# clean width law and the Micro tier is mildly handicapped across the board —
# which does NOT affect within-Micro rankings (all arms share the batch and step
# count) but DOES mean Micro absolute BPB should never be placed on the C-knee
# ladder next to MBS=8 points without this footnote.
# ============================================================================
#
# ---- SCALE-BUDGET + TRANSFER ARMS AT MICRO (added 2026-07-27) --------------
# Ramon asked for Micro twins of the SB/FT block. Checking the logs first changed
# what this group IS, in a way worth stating plainly:
#
# THREE OF THESE ALREADY HAVE MINI ANSWERS. SB0 1.0625, SB1 1.0608, SB2 1.0593,
# alongside D0 1.0436 and SB4 1.0198. So running them at Micro is NOT redundant
# and NOT new science either - it is a CALIBRATION SUITE with known ground truth.
# The Mini ranking is:
#       SB4 1.0198  <  D0 1.0436  <  SB2 1.0593  <  SB1 1.0608  <  SB0 1.0625
# and the D0-to-SB0 spread is 0.0189 BPB, ~19x the measured 0.0010 noise floor,
# so that ordering is well resolved and a real test rather than a coin flip.
# Micro already contains twins of the first two (Micro baseline = D0,
# Micro-untied = SB4), so adding SB0/SB1/SB2 gives FIVE paired points. If Micro
# reproduces the ordering, Micro is a trustworthy prioritiser - a far stronger
# statement than PROJ_Micro alone can make, and it comes with a number
# (Spearman rho over 5 pairs, plus per-arm gap-vs-gap).
# NOTE this cuts BOTH ways and that is the point: if the ordering scrambles, the
# whole Micro tier is discredited by its own control group and we fold back to
# Mini having spent ~20 dollars to find out.
#
# SB4_Micro IS DELIBERATELY ABSENT - it would duplicate Micro-untied above
# (untied lifting, C=256, lr 0.075). Its calibration pair is already in hand.
#
# ONLY SB3 IS A GENUINELY NEW ANSWER: Mini SB3 died ~1 minute in and never
# produced a benchmark, so the SSM-vs-dyadic question is still open.
#
# LR: 0.15 (shared lifting, the 48/C rule at C=256) for every arm here.
# COST: ~5 dollars each at the post-eval-fix rate (~185s per 500 steps measured
# on the Micro baseline), so ~30 dollars for all six.

run_ablation "SB0_Micro_C256_5ep Scale-budget control Micro - dyadic ladder, crawl OFF"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216}'     "SB0 Micro | CALIBRATION, Mini ground truth 1.0625 (logs/wikitext-103_2026-07-21_08-39-52). Also the crawl-OFF control for every schedule arm at this width. Micro-minus-baseline should reproduce Mini's +0.0189 gap if width transfers"
run_ablation "SB1_Micro_C256_5ep Scale-budget A Micro - prune to levels {1,2,4,8,16}, crawl ON"     "$BASE_PATCH_5EP"     '{"levels": 5, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216}'     "SB1 Micro | CALIBRATION, Mini ground truth 1.0608 (logs/wikitext-103_2026-07-21_19-13-36). Coarse-prune: fewer params than baseline, so a tie is a WIN. Mini put it between SB2 and SB0 - does Micro agree"
run_ablation "SB2_Micro_C256_5ep Scale-budget B Micro - schedule {1,2,3,4,8,16,32}, crawl OFF"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "wavelet_dilation_schedule": [1, 2, 3, 4, 8, 16, 32]}'     "SB2 Micro | CALIBRATION, Mini ground truth 1.0593 (logs/wikitext-103_2026-07-26_00-31-54). Best of the three crawl-OFF/prune arms at Mini and iso-param vs SB0 - the tightest ranking test in the calibration set"
run_ablation "SB3_Micro_C256_5ep Scale-budget C Micro - levels {1,2,4,8} + SSM bypass, crawl ON"     "$BASE_PATCH_5EP"     '{"levels": 4, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "decompose_bypass_ssm": true}'     "SB3 Micro | THE ONLY NEW ANSWER in this group - Mini SB3 died ~1 min in (logs/wikitext-103_2026-07-26_11-48-32, no benchmark) so its question is still open: do coarse dyadic levels reduce to SSM poles. Fewer scales than baseline so it should fit at C=256; if it OOMs again set gradient_checkpointing true"
run_ablation "FT1_Micro_C256_5ep Frozen-wavelet transfer Micro - Micro lifting imported + FROZEN"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "lifting_import_checkpoint": "logs/wikitext-103_2026-07-27_21-37-29/best_model.pt", "lifting_freeze": true}'     "FT1 Micro: donor is the MICRO BASELINE's own checkpoint, NOT D2 - a C=512 donor cannot load into a C=256 model (train.py:1402 load_state_dict strict=True raises on shape mismatch). This makes it a same-config transfer test (is a converged lifting reusable across runs) rather than Mini's cross-config one. It also SIDESTEPS the blocker that killed Mini FT1/FT2: the donor is generated pod-side, so no S3 upload is needed. Compare matched-step val at 2K/4K/8K vs the Micro baseline"
run_ablation "FT2_Micro_C256_5ep Frozen-wavelet transfer Micro - Micro lifting imported, UNFROZEN"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "lifting_import_checkpoint": "logs/wikitext-103_2026-07-27_21-37-29/best_model.pt", "lifting_freeze": false}'     "FT2 Micro: warm-start upper bound. FT1-minus-FT2 = what freezing costs at this width. Same pod-side donor as FT1"

# CONDITIONAL — DO NOT RUN UNTIL THE MICRO FwPKM ARMS REPORT. Left commented on purpose.
# This line predates the 2026-07-28 rewrite and its config is now stale in two ways:
#   * no sequential_blocks, so fast weights cannot persist across windows at all;
#   * num_keys=8281 at top_k=32 gives retention 8281/(256*32) = 1.01 windows, i.e. the
#     store is overwritten almost exactly once per window — the same starved regime the
#     Micro arms exist to escape.
# It would therefore spend ~$8-10 at Mini re-testing the configuration we already expect
# to be null. Rebuild it from whichever Micro arm wins (retention setting, sequential
# ordering, and the num_keys/top_k pair), then promote.
# run_ablation "FWP1_C512_fwpkm8281_5ep Fast-weight PKM retry — rewired, D0 config + memory"     "$BASE_PATCH_5EP"     '{"fwpkm_enabled": true, "fwpkm_num_keys": 8281, "fwpkm_top_k": 32, "fwpkm_heads": 1, "fwpkm_aux_weight": 0.01, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "FWP1: first FwPKM trial in the fully-spectral architecture and the first with identity-init, a selective softmax, and a usage penalty; 118.37M, so the bar is the width law (~1.012), not D0 (1.0436)"

#
# TWO BUGS FOUND 2026-07-27 while smoke-testing the scale ladder. Both matter here:
#
# (1) BLOCK-MoE COULD NOT RUN AT ALL. The top-level loop passes assoc_prev=... to every
#     layer UNCONDITIONALLY (model.py ~3993), and MoE layers never take the checkpointing
#     branch that omits it (the guard is `not is_moe_layer`), so WaveletLMBlockMoE.forward
#     died on "unexpected keyword argument 'assoc_prev'" at step 0. MOEA WOULD HAVE
#     CRASHED IMMEDIATELY. Fixed: the MoE forward now accepts assoc_prev, and REFUSES
#     (NotImplementedError) if it is non-None, because cross-layer AMB state has no single
#     owner under top-k routing. AMB is parked, so shipping configs pass None.
#
# (2) MOE0 WAS NOT A 1-EXPERT MoE. Its override never set block_moe_enabled, so it ran as
#     a plain shared-lifting-OFF model — byte-identical param count to SB4 (139,077,098,
#     a router would have added 512). It therefore did NOT isolate routing from lifting-
#     untying — BUT DO NOT RE-QUEUE IT. A 1-expert block-MoE is provably identical to its
#     bare expert: with num_experts=1 the router softmax is 1.0 by construction, and a CPU
#     check gives max|moe(x) - expert(x)| = 0.000e+00 exactly. The control is an IDENTITY,
#     discharged for free and permanently; five epochs of GPU could never have shown more.
#     The right lesson is that this arm was the wrong instrument, not that it was run
#     wrong. What it accidentally produced is more
#     useful than it sounds — and it COMPLETED, so the numbers are final:
#
#     *** EMPIRICAL NOISE FLOOR, MEASURED (logs/wikitext-103_2026-07-27_01-45-36) ***
#     SB4 and MOE0 differ in ZERO of 154 config keys, same seed 1337, same 139,077,098
#     params. Two runs of a bit-identical configuration:
#         metric            SB4        MOE0       delta
#         sliding BPB       1.0198     1.0201     +0.0003
#         PPL              24.1871    24.2074     +0.0203
#         sliding avg loss  3.1858     3.1867     +0.0009
#         best val          3.2198     3.2199     +0.0001
#     So end-to-end run-to-run drift is ~0.0003 BPB — about a THIRD of the 0.0010 threshold
#     everything in this repo is ranked by. That threshold was a stipulation until now; it
#     is now measured, and it is conservative. Residual drift is fp16 atomics + cuDNN algo
#     selection + torch.compile kernel choice; bitwise reproducibility was never available.
#     Robustness note: the two runs took 13.7h and 15.15h (~10% apart, so real contention
#     or thermal variation) and the result still barely moved.
#     LIMIT, STATED HONESTLY: n=2 is ONE PAIRED DIFFERENCE, not a distribution. It shows
#     drift CAN be 0.0003 and that 0.0010 is not optimistic; it does NOT give a sigma.
#     Do not buy more replicates at ~$15 each just to tighten this.
#     BONUS: SB4's untied-lifting result is now n=2 (1.0198, 1.0201; mean 1.01995) rather
#     than a single seed — worth having for a number heading into the paper.
#
#     The matched-step
#     val agreement is 0.0009 nats max (steps 38500/41000/43000/45000/45500/46000:
#     3.2583/3.2576, 3.2438/3.2447, 3.2373/3.2376, 3.2300/3.2299, 3.2266/3.2268,
#     3.2344/3.2347). That is ~0.0003 BPB of pure run-to-run nondeterminism — an EMPIRICAL
#     noise floor that independently corroborates the 0.0010 BPB threshold we rank by.
#
# HOW TO JUDGE BLOCK-MoE (the width law does not apply cleanly). At E=4 these are ~479M,
# where the width law demands 0.9212 — below the C=1024 headline (0.9805). No block-MoE
# arm can clear that, because block-MoE replicates the WHOLE block per expert and (per its
# own docstring) runs every expert in training, so it buys neither param- nor compute-
# efficiency. Judge MOEA against MOE0/SB4 (1.0198) for "does routing help at all", and
# judge MLR1 against MOEA ONLY — same E, same params, same compute, differing solely in
# the scale ladder. That difference is the one clean measurement in the family.
#
# MLR1 — MULTIRESOLUTION LADDER (plans/multiresolution_moe.md; Ramon 2026-07-27).
# Expert k reconstructs from only the COARSEST scales: e0<-s0..s7, e1<-s0..s5, e2<-s0..s3,
# e3<-s0..s1 (smoke-verified). s0 is coarsest and coarse bands integrate (lag-1 rho ~0.41)
# while fine bands are memoryless (~0.07, Finding 17), so higher k = longer effective span,
# less detail. Nested, so no expert is starved.
# HONEST SCOPE — THIS IS THE RESOLUTION HALF ONLY. Without the decimating compressor the
# window is still block_size=256, so experts get LESS DETAIL WITHOUT MORE SPAN. Therefore:
#   MLR1 < MOEA  -> weak evidence against; dropping fine bands costs information by itself,
#                   so a loss is confounded and does NOT falsify the compressed version.
#   MLR1 >= MOEA -> the INFORMATIVE outcome: resolution specialisation pays even when the
#                   span benefit is absent, which is the strongest possible motivation for
#                   building the compressor.
# Watch aux (balanced = 2.0 for top-2 of 4; smoke-measured 2.0003 at init) for collapse
# onto e0, which is the failure mode to expect since e0 alone sees every scale.
# COST: ~4x compute like MOEA; gradient_checkpointing=true (MOEA needed it). ~$15-20.
#
# ORDERING (Ramon, 2026-07-27): MLR1 runs BEFORE MOEA at his request. The pair is what
# matters, not the order — MLR1-minus-MOEA is the same measurement either way. The ONE
# risk of this order: if budget stops after MLR1, its number is UNINTERPRETABLE on its own
# (479M against a 0.9212 width-law bar it cannot clear, and no matched control). MOEA is
# the arm that stands alone, against SB4/MOE0 1.0198. So if only one of the two can run,
# run MOEA.
run_ablation "MLR1_C512_moe4_scaleladder_5ep Multiresolution ladder — E=4 coarse-only experts"     "$BASE_PATCH_5EP"     '{"block_moe_enabled": true, "block_moe_experts": 4, "block_moe_topk": 2, "block_moe_aux_weight": 0.01, "block_moe_scale_ladder": true, "shared_lifting_weights": false, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.0375, "min_lr": 0.00075, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 512}'     "MLR1: coarse-only resolution ladder over E=4 experts; the ONLY valid A/B is vs MOEA (same params/compute, ladder is the sole difference). Resolution half only - no compressor yet, so a LOSS is confounded and a WIN is the informative result"
run_ablation "MOEA_C512_moe4top2_5ep Block-MoE A — E=4 full-block experts, top-2, per-token router"     "$BASE_PATCH_5EP"     '{"block_moe_enabled": true, "block_moe_experts": 4, "block_moe_topk": 2, "block_moe_aux_weight": 0.01, "shared_lifting_weights": false, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.0375, "min_lr": 0.00075, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 512}'     "MOEA: block-MoE E=4 (479M, ~4x compute); vs MOE0 = does per-token routing beat one big expert; watch aux (balanced=2.0) for router collapse + inspect learned specialization"
# ==============================================================================
# OOM RETRIES (2026-07-27). Six runs died in the 01:38-01:44 batch. They failed for THREE
# different reasons, and only three were OOM — the logs distinguish them by how far they got:
#   * DIED AT FIRST STEP (reached "=== EPOCH 1/5 ===", no Step 500) = genuine OOM:
#       PP1, PP2 (S=19 scales x MBS=48 activations) and MOMA. Remedied below with
#       gradient_checkpointing=true, which preserves MBS=48 — the PP/MOM comparisons are
#       against D0 at MBS=48, so dropping the batch instead would break them.
#   * DIED BEFORE COMPILE (stopped right after the param breakdown) = NOT OOM at all:
#       FT1, FT2, MOMB. All three import
#       logs/wikitext-103_2026-07-14_09-11-32/best_model.pt, and *.pt is gitignored
#       (.gitignore:23) — the file has never been on the pod. They never reached the GPU.
#       PREREQUISITE, NOT A CODE FIX: upload that checkpoint to S3 and fetch it pod-side
#       (.gitignore:53 already names this route: "fetch from S3 alongside checkpoints").
#       Left commented until the file is confirmed present; re-queueing now just re-fails.
#       MOMB additionally needs gradient_checkpointing=true, since MOMA OOM'd.
# COST: ~$7-9 each at 5ep/C=512; checkpointing costs ~20-30% wall-clock.
run_ablation "PP1_C512_primepow11_crawlON_5ep Prime-power hedge — max=11, cap=128, crawl ON (OOM retry)"     "$BASE_PATCH_5EP"     '{"prime_power_wavelet_basis_max": 11, "prime_power_dilation_cap": 128, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 512}'     "PP1 retry: 18-level prime-power union ladder (134.86M MEASURED from the failed run, not the ~133M estimate) crawl ON; kill rule: must beat 1.0036 — the width law at 134.86M vs D0 72.89M/1.0436, using the 4-point C-knee slope -0.065/e-fold. NOTE the commented PP1 line above says ~1.015 from a shallower slope; 1.0036 supersedes it"
run_ablation "PP2_C512_primepow11_crawlOFF_5ep Prime-power hedge — max=11, cap=128, crawl OFF (OOM retry)"     "$BASE_PATCH_5EP"     '{"prime_power_wavelet_basis_max": 11, "prime_power_dilation_cap": 128, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 512}'     "PP2 retry: crawl-OFF twin; PP1-minus-PP2 = crawl redundancy of the prime rungs — the dead-weight question, answered directly"
run_ablation "MOMA_C512_mom4top2_5ep Mixture-of-Mixers A — E=4, top-2, from scratch (OOM retry)"     "$BASE_PATCH_5EP"     '{"mixer_mom_enabled": true, "mixer_mom_experts": 4, "mixer_mom_topk": 2, "mixer_mom_aux_weight": 0.01, "per_scale_mixer_widths": null, "levels": 7, "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 512}'     "MOMA retry: MoM existence test — 57.08M MEASURED (the ~12M-saved estimate was low; it saves 15.81M vs D0 72.89M). This is the one arm SMALLER than D0, so the width law works FOR it: 57.08M is expected at 1.0595, and anything under that is a win on params while under 1.0436 beats D0 outright. Watch expert-usage collapse via aux magnitude"
# ==============================================================================
# BLOCK-LEVEL MIXTURE OF EXPERTS (continued).
# ==============================================================================

run_ablation "K5_C768_L10_noMLP_5ep C-knee sweep 5/5 — C=768 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.06, "min_lr": 0.0012, "checkpoint_interval_steps": 5000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 768}'     "K5_C768_L10_noMLP_5ep: C-knee 5/5, C=768 (~150.0M), lr=0.06; completes the 7-point law with the C=1024/2048 anchors"
# ==============================================================================


# ==============================================================================
# FULLY-SPECTRAL BASELINE REDOS + EPOCH LADDER (2026-07-05, ~$300-400 envelope approved).
# SP1's verdict: removing proj_out WINS at C=1024 (0.9805 vs 0.9884 sliding BPB, -0.0079 = ~8x noise,
# at -10.5M params; val 3.0749 vs 3.0942) while COSTING at C=100 (+0.0115) -> sign flip with width.
# skip_proj_out=true is now the default recipe; all remaining baselines get redone fully spectral.
# Order: P2 (PG-19 Small redo) -> M2 (Medium WT-103 5ep redo) -> M4 (Medium 1ep, cheap epoch-curve
# point + fail-fast for M3) -> M3 (Medium 10ep, the TXL/GPT-2-XL chase). Est costs at ~$1/hr 5090:
# P2 ~$100 (4d) + M2 ~$50 (2d) + M4 ~$10 (10h) + M3 ~$100 (4d) ~= $260 + K-sweep remainder.
#
# P2 — PG-19 SMALL REDO, fully spectral. P1's recipe + skip_proj_out (~220.4M vs P1's 230.89M).
# A/B vs P1 (27.72 sliding PPL / 1.0892 BPB / best val 3.5023) AND vs the old 808M headline (27.40).
# If SP1's -0.0079 BPB carries over, sliding PPL lands ~27.0-27.1 (est) -> would TAKE the PG-19
# headline outright at 3.7x fewer params than the 808M. Same 32K SP tokenizer/cache as P1.
run_ablation "P2_C1024_L10_noMLP_noproj_pg19_1ep PG-19 Small redo — fully spectral C=1024 L=10 (1ep)"     "$BASE_PATCH_1EP"     '{"dataset": "pg19", "checkpoint_interval_steps": 10000, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "P2_C1024_L10_noMLP_noproj_pg19_1ep: PG-19 Small redo fully spectral (~220.4M); A/B vs P1 27.72/1.0892/val 3.5023 and vs old 808M 27.40; if the SP1 gain carries, takes the PG-19 headline"
#
# M2 — MEDIUM WT-103 5EP REDO, fully spectral. M1's recipe + skip_proj_out (~851.5M vs M1's 893.44M).
# A/B vs M1 (0.9597 sliding BPB / 20.04 sliding PPL). Becomes the C=2048 anchor for the C-knee fit.
# If the projection gain holds or grows with width, sliding PPL ~19.5-19.8 (est).
run_ablation "M2_C2048_L10_noMLP_noproj_5ep Medium redo — fully spectral C=2048 L=10 (5ep, WT-103)"     "$BASE_PATCH_5EP"     '{"checkpoint_interval_steps": 20000, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.0225, "min_lr": 0.00045, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 2048}'     "M2_C2048_L10_noMLP_noproj_5ep: Medium WT-103 redo fully spectral (~851.5M); A/B vs M1 0.9597/20.04; new C=2048 anchor for the C-knee law"
#
# M4 — MEDIUM 1EP (epoch-curve point E=1 + fail-fast for M3). Same recipe as M2, 1 epoch (~10h).
# Completes the 1/5/10-epoch ladder cheaply and early regardless of M3's fate.
run_ablation "M4_C2048_L10_noMLP_noproj_1ep Medium epoch ladder — fully spectral C=2048 (1ep, WT-103)"     "$BASE_PATCH_1EP"     '{"checkpoint_interval_steps": 20000, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.0225, "min_lr": 0.00045, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 2048}'     "M4_C2048_L10_noMLP_noproj_1ep: Medium 1ep epoch-ladder point (E=1 of 1/5/10); cheap fail-fast ahead of M3"
#
# M3 — MEDIUM 10EP (the record chase). M2's recipe + epochs=10 (~4 days). Gauges the epoch axis at
# C=2048 and pushes toward TXL-Large (18.3 @ ~1,900 epochs) / GPT-2 XL (17.5 zero-shot) territory.
# Honest expectation: ~18.7-19.3 sliding PPL (est) — approaching, not necessarily passing, 18.3;
# warmup_fraction=0.3 auto-scales to the doubled step count. WATCH the train/val gap: 10 epochs on
# 0.5GB doubles memorization pressure and the recipe's dropout was tuned at 5ep (kept unchanged for
# comparability; if the gap balloons by mid-run, that is itself the finding).
run_ablation "M3_C2048_L10_noMLP_noproj_10ep Medium epoch ladder — fully spectral C=2048 (10ep, WT-103)"     "$BASE_PATCH_5EP"     '{"epochs": 10, "checkpoint_interval_steps": 20000, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.0225, "min_lr": 0.00045, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 2048}'     "M3_C2048_L10_noMLP_noproj_10ep: Medium 10ep record chase (E=10 of 1/5/10); toward TXL-L 18.3 territory; watch train/val gap (dropout tuned at 5ep)"
# ==============================================================================

