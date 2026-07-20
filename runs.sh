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

# ==============================================================================
# LENGTH-GENERALIZATION EVAL (eval-only — NO training) — RUNS FIRST (curious about it)
# Evaluate the best 256-trained checkpoint at growing eval windows for the
# train-short / eval-long BPB curve. Uses the opt-in --eval_block_size flag (the
# benchmark path is unchanged at the trained size). Forward-only, so it does NOT hit
# the 62GB training wall. Architecture stays as trained (levels=7), so this measures
# whether a longer window helps WITHIN the trained wavelet reach (~2^7≈128-256 tokens)
# plus the cross-window decompose-bypass recurrence — NOT full 2048-token dependency
# use (that needs more levels = retraining).
#   block 256 is the CONTROL: --eval_block_size 256 == trained → no-op branch, so it
#   must reproduce the run's original sliding BPB (~0.9748), proving the default path
#   is untouched.
#   PREREQUISITE — pull the checkpoint from S3 first (config.json + best_model.pt):
#     aws s3 sync s3://exarch-ai-model/EXARCH/logs/wikitext-103_2026-06-18_19-18-42/ \
#                 /workspace/EXARCH/logs/wikitext-103_2026-06-18_19-18-42/
#   If best_model.pt is absent the sweep SKIPS (guarded) and training proceeds.
# ==============================================================================
# LENGTHGEN_CKPT="logs/wikitext-103_2026-06-18_19-18-42"   # C=2048/L=5/5ep, BPB 0.9748 (trained block 256)
# if [ -f "$LENGTHGEN_CKPT/best_model.pt" ]; then
#     # Full degradation map. NOTE: the ceiling is the WT-103 TEST SET (287K tokens), NOT
#     # VRAM — batch=1 is forced for benchmarks (train.py:605) and levels stay at 7, so eval
#     # memory stays low even at 128K. Blocks >=2^16 are statistically THIN (<=8 windows),
#     # >=2^17 very thin (2-4). Any block that OOMs or can't form a full window is skipped.
#     for EVAL_BS in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144; do
#         echo ""
#         echo "=== [length-gen] eval $LENGTHGEN_CKPT at block_size=$EVAL_BS ==="
#         build_run_config "{\"benchmark_only\": true, \"benchmark_run_dir\": \"$LENGTHGEN_CKPT\"}"
#         python train.py --config "$TMP_CFG" --eval_block_size "$EVAL_BS" \
#             || echo "[runs.sh] length-gen eval at block_size=$EVAL_BS exited non-zero; continuing"
#         # benchmark.txt is overwritten on each benchmark_only run — snapshot it per
#         # block size so the whole sweep survives.
#         [ -f "$LENGTHGEN_CKPT/benchmark.txt" ] && \
#             cp "$LENGTHGEN_CKPT/benchmark.txt" "$LENGTHGEN_CKPT/benchmark_lengthgen_bs${EVAL_BS}.txt"
#     done
#     git_commit_push "length-gen eval sweep: $LENGTHGEN_CKPT at block 256..262144 (degradation map)"
# else
#     echo "[runs.sh] length-gen sweep SKIPPED — $LENGTHGEN_CKPT/best_model.pt not found."
#     echo "[runs.sh]   Pull it first: aws s3 sync s3://exarch-ai-model/EXARCH/$LENGTHGEN_CKPT/ /workspace/EXARCH/$LENGTHGEN_CKPT/"
# fi

# # --- generate.py inference efficiency profile (decode VRAM + tok/s vs generation length) ---
# # Profiles AUTOREGRESSIVE DECODE (generate N new tokens) — a DIFFERENT axis from the eval
# # sweep above (which profiled long-INPUT prefill/scoring). Both characterize the efficiency.
# if [ -f "$LENGTHGEN_CKPT/best_model.pt" ]; then
#     for NT in 256 1024 4096; do
#         echo ""
#         echo "=== [gen-profile] $LENGTHGEN_CKPT --num_tokens $NT --metrics ==="
#         python generate.py --checkpoint "$LENGTHGEN_CKPT/best_model.pt" --num_tokens "$NT" --metrics \
#             || echo "[runs.sh] gen-profile at num_tokens=$NT exited non-zero; continuing"
#     done
#     git_commit_push "gen-profile: $LENGTHGEN_CKPT decode VRAM + tok/s at num_tokens 256/1024/4096"
# fi

# run_ablation "T6_L5_xskip_1ep Cross-Layer Skip — C=2048 L=5 dense skips (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "cross_layer_dense_skips": true, "layers": 5}'     "T6_L5_xskip_1ep: cross-layer dense skips (init-to-identity) at L=5/1ep; A/B vs shared no-skip L=5 (1.0831); smoke-test first eval == baseline then diverge"

# run_ablation "T6_L5_xskip_5ep Cross-Layer Skip — C=2048 L=5 dense skips (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "cross_layer_dense_skips": true, "layers": 5}'     "T6_L5_xskip_5ep: cross-layer dense skips at L=5/5ep; A/B vs shared no-skip L=5 (5ep TBD)"

# ---- Untied lifting RETRY at the LR-cliff-corrected rate (last in queue) ----------
# The first untied ep=1 run NaN'd at lr=0.0225, but the cliff was measured MILD: NaN
# onset at lr=0.0205 (step 16000), only ~11% below shared's 0.0225 — not the steep
# LR/L drop one might assume (LR/L=0.0045, LR/sqrt(L)=0.010 are both far too low). So
# this retries at lr=0.018 (just under the 0.020 cliff, with margin). MBS 8->4 + GA
# 1->2 holds the effective batch while shedding activations to fit untied's ~35.8 GB
# under the 5090's 32 GB — VERIFY at launch; if it OOMs, drop to MBS=2/GA=4. If it now
# trains AND beats shared L=5 (1.0831), untied re-enters the conversation; else it
# stays a post-release item.
# run_ablation "T5_L5_untied_lr018_1ep Untied Lifting retry — C=2048 L=5 shared_lifting_weights=false lr=0.018 (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.018, "min_lr": 0.00036, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "shared_lifting_weights": false, "layers": 5}'     "T5_L5_untied_lr018_1ep: untied retry at lr=0.018 (just below the measured 0.020 NaN cliff; onset 0.0205 at shared-tuned 0.0225); (set MBS=4/GA=2 if OOM); A/B vs shared L=5 (1.0831)"


# ==============================================================================
# Block-Size Extension & Length Generalization (C=1024 / L=5) — runs.sh
# See README "Block-Size Extension & Length Generalization". block_size 256 vs 2048 —
# the saner INCREMENTAL step: block 4096 OOM'd at every MBS on the 5090 (MBS-independent
# T·S·C floor) and needs MBS<8 even on a 48GB card, so we expand to 2048 first (8x context,
# still fits MBS=8). Per-epoch wall-clock is ~flat in block size (fixed WT-103 tokens).
#   levels = log2(block) - 1 ; per_scale_mixer_widths length = levels + 1.
#   LR = 0.05 (C=1024's 1/C ceiling; context-invariant — NOT lowered for block size).
#   MBS=8/GA=1 throughout: BS=2048 fits at ~28-29 GB est (comfortable on a 48GB 6000;
#   tight on a 32GB 5090 — verify at launch given the block-4096 estimate ran low 3x).
#   Widths = block-4096's [1x6, 0.5x6] minus the coarsest scale -> [1x6, 0.5x5] (S=11).
# Runs: BS256/5ep (DONE = 1.0002) + BS2048/1ep + BS2048/5ep.
# ==============================================================================

# (1) BS=256, 5ep — the width-proxy baseline: A/B vs C=2048/L=5/5ep (runs.sh More Epochs Max row).
# run_ablation "T5_C1024_L5_bs256_5ep Block-Size — C=1024 L=5 block=256 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 5, "C": 1024}'     "T5_C1024_L5_bs256_5ep: block-size baseline + width proxy — C=1024 L=5 block=256 5ep; A/B vs C=2048/L=5/5ep; ~13.5h on 5090"

# (1b) BS=256, 1ep — the MISSING C=1024/L=5/1ep baseline (fills the depth×epoch grid:
#      we have L=1/1ep=1.1368 and L=5/5ep=1.0002, but no L=5/1ep). Cheap, block-256 speed.
# run_ablation "T5_C1024_L5_bs256_1ep Baseline — C=1024 L=5 block=256 (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 5, "C": 1024}'     "T5_C1024_L5_bs256_1ep: missing baseline — C=1024 L=5 block=256 1ep; A/B vs L=1/1ep (1.1368) and L=5/5ep (1.0002)"

# (2) CANCELLED — BS=2048 TRAINING dropped (eval-only pivot). Reasons: ~62 GB on a 96 GB
#     card (undecimated T·S·C wall), NaN cliff drops to ~0.011, and the steps/LR confound
#     makes 1ep uninterpretable. We now do length-GENERALIZATION evals on the 256 model and
#     defer real long context to decimation. Commented runs kept for the record.
# run_ablation "T5_C1024_L5_bs2048_1ep Block-Size — C=1024 L=5 block=2048 (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 10, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.00625, "min_lr": 0.000125, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "block_size": 2048, "layers": 5, "C": 1024}'     "T5_C1024_L5_bs2048_1ep: block-size extension — C=1024 L=5 block=2048 1ep; lr 0.00625 (NaN cliff ~0.011 at this block per log 16-31-05 — block 2048 LOWERS the LR ceiling vs block 256's ~0.05); min_lr=lr/50; MBS=8 used ~62GB on the 96GB Blackwell 6000"

# (3) CANCELLED — BS=2048/5ep training dropped (see (2) above).
# run_ablation "T5_C1024_L5_bs2048_5ep Block-Size — C=1024 L=5 block=2048 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 10, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.00625, "min_lr": 0.000125, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "block_size": 2048, "layers": 5, "C": 1024}'     "T5_C1024_L5_bs2048_5ep: block-size extension — C=1024 L=5 block=2048 5ep; lr 0.00625 (cliff ~0.011 — see 1ep), min_lr=lr/50; MBS=8 ~62GB on the 96GB Blackwell 6000"



# ==============================================================================
# C=1024 ITERATIVE PIPELINE (the new fast track) — runs after the C=2048 xskip/untied
# tests above. C=1024/L=5/5ep already hit 1.0002 BPB / 22.75 PPL at 375M (beats the old
# 883M headline) and the C=1024-vs-C=2048 gap SHRANK with depth+epochs (0.0295 -> 0.0254
# BPB), so C=1024 is the validated cheap proxy; C=2048/C=4096 are reserved for final
# headlines. All at LR 0.05 (C=1024's 1/C ceiling), block 256, A/B vs C=1024/L=5/5ep = 1.0002.
# ==============================================================================

# --- C=1024 cross-layer skip (does dense skipping help the C=1024 case too?) ---
# run_ablation "T6_C1024_L5_xskip_1ep C=1024 Cross-Layer Skip — L=5 (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "cross_layer_dense_skips": true, "layers": 5, "C": 1024}'     "T6_C1024_L5_xskip_1ep: does cross-layer skip help C=1024 too? init-to-identity; A/B vs C=1024/L=5 plain"

# run_ablation "T6_C1024_L5_xskip_5ep C=1024 Cross-Layer Skip — L=5 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "cross_layer_dense_skips": true, "layers": 5, "C": 1024}'     "T6_C1024_L5_xskip_5ep: cross-layer skip at C=1024 L=5/5ep; A/B vs C=1024/L=5/5ep = 1.0002"

# # --- C=1024 untied lifting (theory: MORE stable here — C=1024's cliff is ~0.062, we run
# # 0.05, and the measured ~11% untie haircut lands ~0.055, still above 0.05). If it NaNs,
# # drop lr 0.05 -> 0.04.
# run_ablation "T5_C1024_L5_untied_1ep C=1024 Untied Lifting — L=5 shared_lifting_weights=false (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "shared_lifting_weights": false, "layers": 5, "C": 1024}'     "T5_C1024_L5_untied_1ep: untied at C=1024 (more LR headroom than C=2048; cliff ~0.062 vs lr 0.05); re-probe per-layer dilation_logits; if NaN drop lr->0.04"

# --- Deep C=1024 iterative-deepening (1ep CEILING-FINDER). gradient_checkpointing is OFF
# now: the RTX 6000 has the memory (L=20 ≈ ~50 GB without it — verify at launch; re-enable
# grad-ckpt only if it OOMs). Run L=10, check it clears the ~0.0010 noise
# floor vs L=5; only then L=15; only then L=20. STOP bumping when a depth fails to clear
# noise. ⚠ PRIOR DEPTH-CEILING SIGNAL: the OLD-recipe 30L/C=512 run REGRESSED vs 20L
# (depth hurt past ~20 layers); the new learned-residual recipe may push the ceiling
# higher, but L=20 is plausibly near it. The 5ep HEADLINE run goes on the depth WINNER
# ONLY (1ep is the cheap finder; L=20/5ep would be ~50h).
# run_ablation "T5_C1024_L10_1ep Deep C=1024 — L=10 (1ep, grad-ckpt)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "T5_C1024_L10_1ep: deep C=1024 ceiling-finder — L=10, grad-ckpt; clears noise vs L=5?"

# run_ablation "T5_C1024_L15_1ep Deep C=1024 — L=15 (1ep, grad-ckpt)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 15, "C": 1024}'     "T5_C1024_L15_1ep: deep C=1024 — L=15, grad-ckpt; only run if L=10 cleared noise"

# CANCELLED — L=15 plateaued (1.1099, within noise of L=10's 1.1113); depth ceiling ~L=10:
# run_ablation "T5_C1024_L20_1ep Deep C=1024 — L=20 (1ep, grad-ckpt)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 20, "C": 1024}'     "T5_C1024_L20_1ep: deep C=1024 — L=20, grad-ckpt; near the prior ~20L ceiling, only if L=15 cleared noise"


# ==============================================================================
# C=1024 L=10 HEADLINE — the Small final, 5ep on WT-103 (the depth WINNER's headline run).
# L=10 cleared the noise floor vs L=5 at 1ep (-0.0093 BPB); L=15 plateaued (depth ceiling ~L=10),
# so the 5ep headline goes on L=10 ONLY. Identical to the L=10/1ep finder above, just epochs=5
# (BASE_PATCH_5EP) -> grad-ckpt OFF, MBS=8, GA=1, block 256 all unchanged. Measured peak training
# VRAM was 16838 MiB at L=10 -> FITS the A5000 (24 GB) with NO grad-ckpt / MBS / GA changes.
# Expect ~1.5 days on the A5000 (~64K steps/epoch x measured s/it; clock the first ~50 steps).
# A/B vs C=1024/L=5/5ep = 1.0002 (logs/wikitext-103_2026-06-19_13-21-20). PG-19 5ep is ~20x the
# tokens (~5-6 days/epoch here) -> run PG-19 at 1ep on a more economical box. UNCOMMENT to launch.
# run_ablation "S1_C1024_L10_5ep Small Headline — C=1024 L=10 (5ep, WT-103)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "S1_C1024_L10_5ep: C=1024 L=10 5ep headline on WT-103; depth winner (L=10 cleared noise, L=15 plateaued); 16838 MiB peak -> fits A5000 ckpt-off; A/B vs C=1024/L=5/5ep=1.0002"


# ==============================================================================
# NO-MLP ABLATION (motivated by Kiruluta's MLP-free Wavelet Logic Machines, his 2026-06-24 review).
# The MLP is 419.6M of the 669.24M S1 headline (62.7%); mlp_expansion=0 removes it cleanly
# (model.py:2124, use_mlp = mlp_expansion > 0). LR 0.05 still applies (width-bound ~1/C, not
# MLP-bound; if it NaNs, drop to 0.04). All A/B vs S1 headline = 0.9894. UNCOMMENT the one(s) to run.
#
# (A) all-else-equal: L=10, mlp_expansion=0 -> ~249.6M. The RAW cost of the MLP (-63% params).
# run_ablation "S2a_C1024_L10_noMLP_5ep No-MLP all-equal — C=1024 L=10 mlp_exp=0 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "S2a_C1024_L10_noMLP_5ep: MLP removed (mlp_expansion=0), all else equal -> ~249.6M; raw cost of the MLP; A/B vs S1 0.9894"
#
# (B) iso-param via DEPTH: L=35, mlp_expansion=0 -> ~671.5M. ISO-PARAM and ISO-COMPUTE (35 no-MLP
# layers ~= 10 MLP layers in FLOPs -> ~same ~2.4-day wall-clock). Tests if depth can replace the MLP.
# CAVEAT: L=35 is ~2x past the depth plateau (L=15) -> EXPECT a regression; that itself answers "can
# # depth substitute for the MLP's channel-mixing" (likely not fully). 35 layers of activations may
# # need grad-ckpt on the A5000 -> if it OOMs, set "gradient_checkpointing": true.
# run_ablation "S2b_C1024_L35_noMLP_5ep No-MLP iso-param via depth — C=1024 L=35 mlp_exp=0 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 35, "C": 1024}'     "S2b_C1024_L35_noMLP_5ep: no-MLP iso-param via depth (L=35, ~671.5M, iso-compute); can depth replace the MLP? expect regression past depth ceiling; A/B vs S1 0.9894"
# #
# # (C) FOLLOW-UP — iso-param via WIDER MIXER (more spectral capacity/layer, closer to the paper's
# # coefficient-domain emphasis; likely the no-MLP variant that gets NEAREST 0.9894). mixer_depth~4 at
# # L=10 is the rough target (mixer is 14.73M/layer at depth 1; need ~56.7M to match the MLP) -- VERIFY
# # the printed PARAMETER BREAKDOWN at launch and tune mixer_depth to ~669M; may need "mixer_depth_stabilizers": true.
# run_ablation "S2c_C1024_L10_noMLP_widemixer_5ep No-MLP iso-param via wide mixer — C=1024 L=10 mixer_depth~4 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "mixer_depth": 4, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "S2c_C1024_L10_noMLP_widemixer_5ep: no-MLP iso-param via wider mixer (mixer_depth~4, tune to ~669M); spectral substitute for the MLP; A/B vs S1 0.9894"
# #
# # (D) iso-param via MIXTURE — mixer_depth=2 + increased depth (the hybrid: if mixer+depth matches or
# # beats no-MLP, it's a strong case to drop the MLP PERMANENTLY). mixer_depth=2 ~doubles the mixer
# # (~29.5M/layer), so iso-param lands near L=18 (~650M) / L=19 (~681M); L=18 also sits in the known-good
# # depth range (~15-20), unlike S2b's L=35 -> the best-conditioned no-MLP iso-param variant. VERIFY the
# # printed PARAMETER BREAKDOWN and tune layers +/-1 to ~669M; if the depth-2 mixer is unstable, set
# # "mixer_depth_residuals": true.
# run_ablation "S2d_C1024_L18_noMLP_mixer2_5ep No-MLP iso-param via mixer_depth=2 + depth — C=1024 L=18 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "mixer_depth": 2, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 18, "C": 1024}'     "S2d_C1024_L18_noMLP_mixer2_5ep: no-MLP iso-param via mixer_depth=2 + L=18 (~650M, tune to ~669M); the mixer+depth hybrid; if it matches/beats S1 0.9894, strong case to drop the MLP for good"
# ==============================================================================


# ==============================================================================
# C=2048 NO-MLP HEADLINE (Medium, 5ep WT-103). The no-MLP win (S2a: 0.9884 BPB at 249.59M, a TIE with
# the 669M MLP version) makes C=2048 CHEAP: no-MLP C=2048/L=10 ~= 850M-1B (vs 2.6B with MLP) -> fits a
# 5090 (32GB), ~1 day, ~$25-30. mlp_expansion=0; LR follows the C=2048 width ceiling (0.0225, ~half the
# C=1024 0.05); min_lr=lr/50. VERIFY the printed PARAMETER BREAKDOWN + VRAM at launch; if it OOMs set
# gradient_checkpointing: true. A/B vs the C=1024 no-MLP Small headline 0.9884.
# NOTE: comment out the finished/cancelled S2 runs above (S2a done; S2b/c/d cancelled) so a fresh 5090
# pod runs ONLY this.
# run_ablation "M1_C2048_L10_noMLP_5ep Medium Headline — C=2048 L=10 no-MLP (5ep, WT-103)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.0225, "min_lr": 0.00045, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 2048}'     "M1_C2048_L10_noMLP_5ep: Medium headline C=2048 L=10 no-MLP 5ep WT-103; ~850M-1B (no-MLP makes C=2048 cheap, fits 5090); lr=0.0225 (C=2048 ceiling); A/B vs C=1024 no-MLP 0.9884"
# ==============================================================================


# ==============================================================================
# FREE-C TEST (C=100, non-power-of-2). Validates the pow2 unlock end-to-end: with
# mixer_transform=identity the FWHT is off, so Cp=C=100 -- NO padding (before the
# gate this width up-padded to Cp=128). Same recipe as the Small headline, just
# C=100 + levels=6 (7 scales, widths [1,1,1,1,.5,.5,.5]) so the per-scale widths
# track the tiny model. MBS=64 (8x the base batch): the C=100 model idles the GPU
# (~27% util, 8% VRAM) on tiny matmuls, so a big batch amortizes kernel-launch
# overhead -- memory is NOT the limit, update count is, so we stop at 64 (not
# max-RAM, which would starve the optimizer on WT-103). lr=0.3 = ~sqrt(8)x the 0.1
# base for the 8x batch (linear 0.8 would top the C=100 width ceiling; the bigger
# batch raises it anyway). eval_interval=500 (steps/epoch drop ~8x). mlp_expansion=0. Tiny
# (the V x 100 embedding dominates; mixer/lifting are ~C^2, negligible at C=100)
# -> a few hours on the 5090 between M1 and PG-19. BPB WILL be poor (100-dim
# model) -- the POINT is that a non-pow2 C trains at all.
# run_ablation "F1_C100_L10_noMLP_freeC_5ep Free-C Test — C=100 (non-pow2) L=10 no-MLP (5ep, WT-103)"     "$BASE_PATCH_5EP"     '{"levels": 6, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.3, "min_lr": 0.006, "micro_batch_size": 64, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 100}'     "F1_C100_L10_noMLP_freeC_5ep: Free-C validation — C=100 (non-pow2; Cp=C=100, identity/FWHT off, no padding) L=10 no-MLP 5ep WT-103; embedding-dominated; proves non-pow2 widths train; BPB expected poor (tiny model)"
# # ==============================================================================


# # ==============================================================================
# # PG-19 SMALL (C=1024 L=10 no-MLP, 1 epoch). Queued AFTER M1 + F1 -> runs back-to-back on the same 5090.
# # dataset="pg19" auto-selects PG-19's own 32K SentencePiece tokenizer (model.py:3665); the FIRST run
# # TRAINS that SP model + tokenizes PG-19 (~a few hours + ~5-10GB .cache, BEFORE the train clock; syncs
# # to S3, reusable). NOTE: 32K vocab (not GPT-2 50K) -> smaller embedding, AND BPB is on PG-19's OWN
# # scale -> NOT comparable to the 0.9884 WT-103 number. ~2.5-3B+ tokens, 1ep -> ~2.3-2.7 days on a 5090.
# # lr=0.05 (C=1024 width ceiling). mlp_expansion=0.
# run_ablation "P1_C1024_L10_noMLP_pg19_1ep PG-19 Small — C=1024 L=10 no-MLP (1ep)"     "$BASE_PATCH_1EP"     '{"dataset": "pg19", "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "P1_C1024_L10_noMLP_pg19_1ep: PG-19 Small C=1024 L=10 no-MLP 1ep (~250M, 32K SP vocab); first run trains PG-19 SentencePiece + tokenizes; runs after M1 on the 5090; BPB not comparable to WT-103"
# ==============================================================================


# ==============================================================================
# SKIP-PROJ-OUT QUICK CHECK (C=100 free-C config, 5ep WT-103). Runs BEFORE SP1 — ~2.3 h on a 5090
# (F1 measured 8189s at MBS=64), so the projection-ablation result is in hand fast (wanted for the
# Kiruluta reply). Identical to F1 (logs/wikitext-103_2026-06-29_19-49-04: C=100, levels=6, MBS=64,
# lr=0.3) plus "skip_proj_out": true — Cp==C=100 under identity, so the flag engages. Removes
# 10 x (100*100+100) = 0.10M of 6.80M (-1.5%; proj share grows with C -> 4.2% at C=1024). A/B vs
# F1 sliding BPB 1.2781 / PPL 54.20 / val 4.0100.
# run_ablation "SP0_C100_L10_noMLP_skipproj_freeC_5ep Skip proj_out quick check — C=100 (non-pow2) L=10 no-MLP (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 6, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.3, "min_lr": 0.006, "micro_batch_size": 64, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 100}'     "SP0_C100_L10_noMLP_skipproj_freeC_5ep: C=100 free-C config + skip_proj_out=true (-0.10M, fully spectral core at C=100); quick ~2.3h A/B vs F1 1.2781 BPB before committing SP1 at C=1024; also demonstrates fine-grained C iteration speed"
# ==============================================================================


# ==============================================================================
# SKIP-PROJ-OUT ABLATION (C=1024 Small, 5ep WT-103). Queued AFTER SP0 (the C=100 quick check) on the 5090. Kiruluta's review
# flagged the MLP + "projection-style components" as the non-spectral parts; the MLP is gone (S2a),
# and proj_out is the biggest projection left. With mixer_transform=identity, Cp == C, so the existing
# skip_proj_out flag engages (model.py: skip_proj_out and Cp==C) -> deletes the per-layer Linear(C,C)
# after reconstruct: -10.50M params across L=10 (249.59M -> ~239.1M), fully spectral block core.
# CAUTION: proj_out normally carries the near-zero "spectral epsilon" init (the residual-regime trick);
# skipping it means the spectral branch enters the learned residual at O(1) from step 0 -- recon_norm
# + scale_weights should absorb it, but if it NaNs in warmup, drop lr 0.05 -> 0.04. All else = S2a.
# A/B vs the no-MLP Small headline 0.9884.
# run_ablation "SP1_C1024_L10_noMLP_skipproj_5ep Skip proj_out — C=1024 L=10 no-MLP, no proj_out (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "SP1_C1024_L10_noMLP_skipproj_5ep: skip_proj_out=true removes the last per-layer projection (-10.50M, fully spectral core); Cp==C under identity so the flag engages; A/B vs S2a 0.9884; if warmup NaNs, retry lr=0.04"
# ==============================================================================


# ==============================================================================
# KIRULUTA COEFFICIENT-SHRINKAGE SCREEN (lambda/gamma/theta — his 3rd architecture suggestion; the
# first two, no-MLP and no-projection, both WON). phi(z) = gamma*sign(z)*relu(|z|-lambda)*cos(theta)
# on the stacked coefficients, per-scale/per-channel, identity-init (lambda~0, gamma=1, theta=0;
# smoke-verified: pre/post match the no-shrink loss to <1e-4 at init). Modes: pre (denoise before
# mixer), post (after mixer), replace (phi IS the computation; mixers/routing not allocated — the
# WLM-purist point). Soft-thresholding = the prox of an l1 coefficient prior (Donoho-Johnstone),
# i.e. a learned sparsity REGULARIZER. Runs BEFORE the K-sweep so a winner folds into the sweep and
# the P2/M2/M3/M4 redos before the config locks.
# DECISION RULE (budget-aware — no 3x5ep): regularizers underperform in the underfit 1ep regime, so
# at C=1024/1ep KILL only if clearly worse than SH4 control beyond noise; a tie or small win sends
# the SINGLE better variant of {pre, post} to one 5ep confirm vs SP1 0.9805 (~$29). 'replace' is
# C=100-only (near-certain large regression at 1024; the purist question is answered cheaply).
# Tier 1 (C=100, 5ep, MBS=64; control = SP0 1.2896; ~2.7h/$3 each): smoke + mechanism probe (read
# the learned per-scale lambdas: do fine scales threshold harder?) — NOT decision-grade (the
# projection SIGN-FLIPPED between C=100 and C=1024). Tier 2 (C=1024, 1ep, MBS=8; ~5.8h each): the
# actual gate; SH4 doubles as the missing fully-spectral 1ep C=1024 baseline (useful forever).
# theta note: on real coefficients cos(theta) is a bounded gain redundant with gamma; kept for WLM
# fidelity  (true phase belongs to the complex-mixer variant, where shrinkage is not wired).
# run_ablation "SH1_C100_shrink_pre_5ep Shrinkage screen — C=100 phi-pre (5ep, MBS=64)"     "$BASE_PATCH_5EP"     '{"levels": 6, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.3, "min_lr": 0.006, "micro_batch_size": 64, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "coefficient_shrinkage": "pre", "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 100}'     "SH1_C100_shrink_pre_5ep: phi before the mixer at C=100 (SP0 recipe); A/B vs SP0 1.2896; mechanism probe, not decision-grade"
# run_ablation "SH2_C100_shrink_post_5ep Shrinkage screen — C=100 phi-post (5ep, MBS=64)"     "$BASE_PATCH_5EP"     '{"levels": 6, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.3, "min_lr": 0.006, "micro_batch_size": 64, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "coefficient_shrinkage": "post", "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 100}'     "SH2_C100_shrink_post_5ep: phi after the mixer at C=100; A/B vs SP0 1.2896 and SH1"
# run_ablation "SH3_C100_shrink_replace_5ep Shrinkage screen — C=100 phi-REPLACE (5ep, MBS=64)"     "$BASE_PATCH_5EP"     '{"levels": 6, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.3, "min_lr": 0.006, "micro_batch_size": 64, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "coefficient_shrinkage": "replace", "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 100}'     "SH3_C100_shrink_replace_5ep: phi REPLACES the mixer (WLM-purist, ~5.4M params); measures what the mixer buys; expect regression"
# run_ablation "SH4_C1024_noshrink_1ep Shrinkage gate — C=1024 control (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "SH4_C1024_noshrink_1ep: fully spectral C=1024 1ep CONTROL (also the missing 1ep no-proj baseline, useful beyond this screen)"
# run_ablation "SH5_C1024_shrink_pre_1ep Shrinkage gate — C=1024 phi-pre (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "coefficient_shrinkage": "pre", "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "SH5_C1024_shrink_pre_1ep: phi-pre at production width; A/B vs SH4; kill only if clearly worse (regularizers underperform at 1ep)"
# run_ablation "SH6_C1024_shrink_post_1ep Shrinkage gate — C=1024 phi-post (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.05, "min_lr": 0.001, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "coefficient_shrinkage": "post", "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 1024}'     "SH6_C1024_shrink_post_1ep: phi-post at production width; A/B vs SH4 and SH5; better of pre/post graduates to ONE 5ep confirm vs SP1 0.9805"
# ==============================================================================


# ==============================================================================
# D-LADDER (the DATA axis of L(N,D) — the missing half of the scaling law; runs BEFORE K5 so the
# MBS fit is probed immediately). Fixed C=512 / L=10, fully spectral, at MBS=48 (64 OOM'd) — bumped from
# C=400 on 2026-07-12: the Chinchilla ratio puts the 10-20ep rungs' optimal N at ~62-200M (C=512's
# 73M is in-band, C=400's 49M is below it), and K4's fresh benchmark (1.0365) came in BETTER than
# the width law predicted (1.041) — C=512 punches above the curve.
# VRAM (MEASURED 2026-07-12): MBS=64 OOM'd at the cross-entropy step — 29.6GB allocated + a 1.54GB
# CE/logits spike (the (B,256,50257) fp16 logits + CE upcast are the peak, not the mixers; count
# them in future MBS math: ~1.6-3GB per 64-batch at V=50K). MBS=48 fits (31.3GB/32.6GB, 96% util).
# LR (MEASURED 2026-07-12 — corrects the sqrt-batch assumption): the first MBS=48 attempt at lr=0.19
# NaN'd during warmup — last stable lr=0.0845 (step 6500, val 3.98 descending), NaN by lr=0.091.
# K4 sustained lr=0.09 at MBS=8, so the 6x batch DID NOT raise the NaN ceiling (~0.09 both) — the
# sqrt-batch LR rule FAILS for this Adagrad(acc=0)+fp16 setup; the ceiling is width-bound and ~batch-
# invariant. (Likely cause: bigger batch grows Adagrad's accumulator slower, prolonging the large-
# effective-step warmup phase.) NOTE: "halve on NaN" under-fixes here (0.19/2=0.095 > cliff). Fixed
# to lr=0.075 (~11% under the proven-stable 0.0845), NO batch inflation. min_lr=lr/50. Future lever
# to reclaim a higher ceiling: initial_accumulator_value=0.1 (see sequential-adagrad memory).
# D0-vs-K4 is now a batch A/B at near-matched LR (0.075 MBS48 vs 0.09 MBS8). eval_interval=500.
# ALL RUNGS SAME MBS+LR; D0-vs-K4 (MBS-8, 1.0365) doubles as the batch-effect A/B at C=512
# (cf. SP0-vs-K0 at C=100, where the big batch WON).
# WHAT IT MEASURES: the data exponent + repeated-token decay + overfit onset (dropout was tuned at
# 5ep — if the train/val gap balloons at D2/D3, that IS the finding; also the natural home for the
# deferred phi-pre shrinkage confirm later). PREDICTION ON RECORD: D2 (20ep) lands ~0.97-1.00
# sliding BPB — a genuine coin-flip against SP1's 0.9805; ties/beats = a 73M model matches the
# 239M flagship (3.3x smaller) and the deployment story rewrites. Est: D0 ~6-7h, D1 ~13h, D2 ~26h,
# D3 ~52h (cancellable).
# run_ablation "D0_C512_L10_fs_MBS48_5ep D-ladder 0/3 — C=512 fully spectral, MBS=48 control (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "D0_C512_L10_fs_MBS48_5ep: D-ladder control at MBS=48 (64 OOM'd at the CE/logits spike, measured); A/B vs K4 MBS-8 1.0365 (batch effect at C=512); if 48 also OOMs fall back 32 (lr 0.16) + restart ladder"
# run_ablation "D1_C512_L10_fs_MBS48_10ep D-ladder 1/3 — C=512 fully spectral (10ep)"     "$BASE_PATCH_5EP"     '{"epochs": 10, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "D1_C512_L10_fs_MBS48_10ep: D-ladder E=10 (1.31B nominal tokens); watch train/val gap vs D0"
# run_ablation "D2_C512_L10_fs_MBS48_20ep D-ladder 2/3 — C=512 fully spectral (20ep)"     "$BASE_PATCH_5EP"     '{"epochs": 20, "checkpoint_interval_steps": 2000, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "D2_C512_L10_fs_MBS48_20ep: D-ladder E=20 (2.62B nominal); THE prediction-scoring rung — ~0.97-1.00 est, coin-flip vs SP1 0.9805; ties/beats = 73M matches the 239M flagship"
# run_ablation "D3_C512_L10_fs_MBS48_40ep D-ladder 3/3 — C=512 fully spectral (40ep)"     "$BASE_PATCH_5EP"     '{"epochs": 40, "checkpoint_interval_steps": 2000, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "D3_C512_L10_fs_MBS48_40ep: D-ladder E=40 (5.24B nominal — the repeats-decay/overfit stress rung); cancellable if D2 shows the curve has flattened"
#
# F1 — THE FRESH-TOKEN TWIN OF D3 (queue after D3 + the law anchors K5/M2, per 2026-07-15 plan).
# Identical Mini recipe, but 1 epoch of EXACTLY D3's repeated-token budget in FRESH Pile tokens:
# 40 x 119,721,489 = 4,788,859,560. The matched-budget A/B that isolates repeats-decay (the
# Muennighoff comparison on our own architecture) AND the first contested-track (Pile) point.
# Dropout/LR/MBS frozen to the D-ladder recipe ON PURPOSE — changing them would confound the twin;
# a low-dropout data-rich variant is the follow-up lever, not this run. LR transfers (width-bound,
# batch held). Two evals when it lands: (1) own Pile-slice test BPB (NOTE: Pile ~3.70 bytes/token
# vs WT-103's 4.5069 — BPBs are different currencies, do not cross-compare); (2) benchmark the F1
# checkpoint on the WT-103 test (same GPT-2 BPE — needs the small cross-eval override in
# benchmark_only) for the apples-to-apples vs D3: does fresh+diverse beat repeated+in-domain ON
# the in-domain test? PREDICTION ON RECORD (est): D3 wins the WT-103 eval (home advantage at only
# 40 repeats); F1 is the stronger general model (Pile test + transfer). One-time prep: streaming
# tokenize+cache ~1-3h (wide bars) then S3-cached; loader smoke-tested 2026-07-15 (ALL PASS).
# Est: ~389.7K steps, ~29h, ~$29. Fallback: if HF streaming stalls on the pod, set pile_hf_id to a
# mirror; if prep OOMs, halve dataset_max_tokens (the twin logic degrades gracefully to half-budget).
# RESTART 2026-07-17 (~2.5h in): the original stream-head val carving measured ~1.2 nats easier than
# the training mixture (non-iid — whatever component leads shard 0). Switched to INTERLEAVED holdout
# (pile_holdout_stride=6000: doc residue 1 -> val, 2-3 -> test, spread over the whole ~3M-doc
# stream). Cache renamed (-s6000) so the stale head-carved cache can't collide; rm the old
# .cache/pile-4788859560tok_gpt2.pt on the pod to reclaim ~19GB.
# run_ablation "F1_C512_L10_fs_pile4.8B_1ep Fresh-token twin of D3 — Mini on 4.79B Pile tokens (1ep)"     "$BASE_PATCH_1EP"     '{"dataset": "pile", "dataset_max_tokens": 4788859560, "pile_val_docs": 500, "pile_test_docs": 1000, "pile_holdout_stride": 6000, "checkpoint_interval_steps": 2000, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "F1_C512_L10_fs_pile4.8B_1ep: fresh-token twin of D3 (4,788,859,560 fresh Pile tokens = D3s exact repeated budget); Muennighoff A/B + first contested-track point; cross-eval on WT-103 test when it lands"
# ==============================================================================


# ==============================================================================
# FREE-C SCALING-LAW SWEEP (the C-knee; K0-K5). Queued AFTER SP1. Five 5ep points at C in
# {200, 300, 400, 512, 768}, EXACT headline protocol (levels=7, widths [1x4,.5x4], MBS=8, GA=1,
# block 256) so the C=1024 (0.9884) and C=2048 (0.9597) headlines join the curve as free anchors ->
# a 7-point BPB-vs-C law + WaveletLM's own tokens/param ratio (replaces the borrowed Chinchilla 20:1)
# + the ultra-light deployment knee. DELIBERATELY NOT MBS=64 (the F1 speedup): mixing batch sizes
# forks the curve off the anchors. Cost: small-C points are launch-overhead-bound at MBS=8 ->
# ~10-15h each on a 5090, ~2.5-3 days total. LR rule lr ~= 48/C (measured: 0.05@1024, 0.0225@2048;
# k in [46, 51]), min_lr = lr/50. C=200 is the largest extrapolation of the rule -> it runs FIRST
# (fails fast); if any K-run NaNs in warmup, halve its lr and relaunch that point only.
# Est params (fit: V*C + 198.13M*(C/1024)^2): K1 ~17.6M, K2 ~32.1M, K3 ~50.3M, K4 ~75.3M, K5 ~150.0M.
# UPDATE 2026-07-05: ALL K-runs now set skip_proj_out=true — SP1 showed removing the projection WINS
# at C=1024 (0.9805 vs 0.9884 sliding BPB at -10.5M params), so the law must be fit on the shipping
# (fully spectral) architecture. The C=1024 anchor is now SP1 ITSELF (0.9805 @ 239.09M — protocol-
# matched: BASE_5EP, levels 7, MBS 8, no-proj); the C=2048 anchor becomes M2 below when it lands
# (M1's proj-on 0.9597 is the interim stand-in, with a small proj-share caveat). Param estimates
# above shift down by the proj share: read exact values from the launch PARAMETER BREAKDOWN prints.
#
# K0 — C=100 MBS-8 RERUN (dual purpose, runs first). NOT sweep protocol: it is the EXACT original F1
# recipe at the base batch (levels=6, widths x7, lr=0.1 = 0.3/sqrt(8), eval 250) so that K0 vs F1
# (logs/wikitext-103_2026-06-29_19-49-04: MBS=64, lr=0.3, 1.2781 BPB) cleanly isolates the MBS
# effect at matched everything-else — "did the 8x batch + sqrt(8) LR shortcut cost quality?".
# Joins the sweep table only as a FLAGGED point (levels=6, lr below the 48/C rule) — bonus, not fit.
# ~6-9h at MBS=8 (F1's pre-bump ETA); 6.80M params (same arch as F1).
# UPDATE 2026-07-05: skip_proj_out added here too (new default) -> params ~6.70M and the MBS A/B
# repoints to SP0 (no-proj, MBS=64, lr=0.3, 1.2896 BPB) — same clean batch-size isolation, now on
# the shipping architecture. The F1 comparison above is retired.
# run_ablation "K0_C100_L10_noMLP_MBS8_5ep C-knee sweep 0/5 — C=100 MBS=8 rerun (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 6, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.1, "min_lr": 0.002, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 100}'     "K0_C100_L10_noMLP_MBS8_5ep: C=100 at MBS=8/lr=0.1 (exact original F1 recipe pre-MBS-bump); A/B vs SP0 MBS=64/lr=0.3 1.2896 (both no-proj) isolates the batch-size effect on the new default arch; flagged (levels=6, off-rule lr) bonus point for the C-knee fit"
# run_ablation "K1_C200_L10_noMLP_5ep C-knee sweep 1/5 — C=200 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.24, "min_lr": 0.0048, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 200}'     "K1_C200_L10_noMLP_5ep: C-knee 1/5, C=200 (~17.6M), lr=0.24 (48/C rule, largest extrapolation -> runs first, halve on NaN); headline protocol so anchors join the fit"
# # run_ablation "K2_C300_L10_noMLP_5ep C-knee sweep 2/5 — C=300 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.16, "min_lr": 0.0032, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 300}'     "K2_C300_L10_noMLP_5ep: C-knee 2/5, C=300 (~32.1M, the prior for the 5ep knee), lr=0.16"
# run_ablation "K3_C400_L10_noMLP_5ep C-knee sweep 3/5 — C=400 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.12, "min_lr": 0.0024, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 400}'     "K3_C400_L10_noMLP_5ep: C-knee 3/5, C=400 (~50.3M), lr=0.12"
# run_ablation "K4_C512_L10_noMLP_5ep C-knee sweep 4/5 — C=512 (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.09, "min_lr": 0.0018, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "K4_C512_L10_noMLP_5ep: C-knee 4/5, C=512 (~75.3M), lr=0.09"
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

