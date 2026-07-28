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

#
# CONTEXT-LENGTH / COMPARABILITY (2026-07-22). The WT-103 baselines (TXL, S4) evaluate
# with far longer effective context than our block-256 recipe. Now that Mini (C=512) is
# strong at high epochs, block 1024 is feasible for the first time (stability + VRAM had
# blocked it at larger C). Dual value: (a) does 4x context help Mini's BPB? (b) closes the
# context axis of the comparison table. NaN CEILING CAVEAT: lr ceiling is width-bound
# (~48/C ~= 0.094 at C=512) BUT block-size LOWERS it (measured ~0.011 at block 2048; block
# 256 safe at 0.075). Block 1024 is untested -> lr=0.03 is a CONSERVATIVE estimate.
# FALLBACKS: warmup NaN -> halve lr to 0.015; OOM at MBS=12 -> gradient_checkpointing:true
# FIRST (protocol-clean, keeps MBS), only then drop MBS. Batch 48->12 does NOT itself need
# an LR cut (NaN ceiling is batch-invariant per the sqrt-batch-fails finding); the cut is
# for the block increase. warmup_fraction=0.3 auto-computes from the new step count.
# Est ~$10-12/arm (similar token throughput, 4x longer sequences at 1/4 the batch).
# run_ablation "CTX1024_C512_L10_5ep Context-1024 Mini — 4x context, block 256->1024"     "$BASE_PATCH_5EP"     '{"block_size": 1024, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.03, "min_lr": 0.0006, "micro_batch_size": 12, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "CTX1024: block 1024 vs D3 block 256 (0.9797) — larger context + comparison-table context match; lr=0.03 conservative (block-1024 NaN ceiling untested)"
#
# CTX1024_L9 — RESULT of CTX1024 (7-level): 1.0654 sliding BPB, +0.0218 WORSE than D0 (1.0436).
# Diagnosis: at levels=7 the coarsest dyadic scale is dilation 64 = only 6% of the 1024 window
# (vs 64/256 = 25% at block 256), so the wavelet receptive field never spanned the context.
# FIX: levels=9 (dilations 1..256) restores coarsest/context = 256/1024 = 25%, matching D0's ratio.
# Widths follow the config's own "coarsest HALF of scales get full width" rule (7-level was 4/8 =
# 50/50); faithful scale-up to 10 scales = coarsest 5 full -> [1,1,1,1,1,0.5,0.5,0.5,0.5,0.5]
# (dilation-16 shifts full->half as the 50/50 boundary moves). 84.28M (+11.4M vs CTX1024).
# A/B TARGET: compare to the 7-level CTX1024 (1.0654), NOT D0 — held at identical lr/MBS/block,
# the only change is levels+widths, so it isolates the receptive-field hypothesis. If it beats
# 1.0654, reach was the bottleneck. (Both still carry the lr=0.03 confound vs D0 — separate probe.)
# gradient_checkpointing:true keeps MBS=12 (clean A/B) while fitting the +2 scales/levels in VRAM
# (7-level peaked at 28.7GB; result-neutral for non-MoE blocks). lr held at 0.03 (width+block bound,
# both unchanged). Est ~$12-15/arm (checkpointing adds ~30% compute).
# run_ablation "CTX1024_L9_C512_5ep Context-1024 Mini, 9 levels — receptive field matched to context"     "$BASE_PATCH_5EP"     '{"block_size": 1024, "levels": 9, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.03, "min_lr": 0.0006, "micro_batch_size": 12, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 512}'     "CTX1024_L9: 9 levels (coarsest dilation 256 = 25% of window) vs 7-level CTX1024 1.0654 — tests whether matching receptive field to context recovers the loss; A/B is vs 1.0654, not D0"
# ==============================================================================
#
# ASSOCIATIVE-MEMORY BYPASS ADAPTER (plans/associative_memory_bypass.md; model.py + train.py,
# 2026-07-23). MQAR VALIDATED the mechanism: additive AMB solves synthetic recall at D=8 (91%)
# AND D=16 (93%) where vanilla plateaus (56% / 40%) — additive is sufficient, delta not needed
# at this scale. This installs it into the frozen D3 Mini: only the +1.311M AMB params train
# (freeze_core), core stays at D3's converged 0.9797 weights, AMB starts at exact identity.
# CONSERVATIVE test (frozen core can't reorganize to make room -> a LOWER bound on AMB's benefit).
# HONEST EXPECTATION: WT-103 is recall-LIGHT (Finding 7 — the model does fine on it without
# recall), so BPB may be modest/flat even though the mechanism works; a drop below 0.9797 means
# WT-103 rewards recall, flat means AMB's value is on recall-heavy tasks (long-context/retrieval).
# EVAL AFTER: recall_diagnostics.py + induction_probe.py on the checkpoint (does QRY-delta rise /
# induction lift leave 0?) AND the sliding BPB vs D3 0.9797. lr=0.03 (only the fresh AMB trains,
# core frozen -> safe). If frozen-adapter is flat, the FULL fine-tune (freeze_core:false) and the
# NATIVE from-scratch AMB Mini are the follow-ups. ~2h/1ep on a 5090.
# run_ablation "AMBA_adapter_D3_frozen_1ep AMB adapter on D3 — install recall, frozen core, train only AMB"     "$BASE_PATCH_1EP"     '{"associative_bypass_enabled": true, "associative_bypass_dim": 64, "associative_bypass_adapter_checkpoint": "logs/wikitext-103_2026-07-15_10-53-46/best_model.pt", "associative_bypass_adapter_freeze_core": true, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.03, "min_lr": 0.0006, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "AMBA: install recall into frozen D3 via the +1.31M AMB adapter; eval induction lift + sliding BPB vs D3 0.9797 (recall-light WT-103 caveat applies)"
#
# AMBA_v2 — CANCELLED 2026-07-24. v1 + its recall_diagnostics already answered v2's question:
# the frozen AMB LEARNED (|out_proj|~1.1) but the model DAMPED it (beta 1.0->0.10) and installed
# NO recall (QRY-delta unchanged from D3, induction lift still +0.14), with BPB flat (0.9798 vs
# 0.9797). On recall-light WT-103 the frozen core has no gradient signal to install recall, so
# more AMB training strength (v2's lr=0.1/3ep) just trains a bigger module the model still turns
# off -> redundant. Skipping straight to the native run below. (Recipe kept for the record.)
# run_ablation "AMBA_v2_D3_frozen_3ep_lr01 AMB adapter on D3 — STRONG: lr=0.1, 3ep, 3% warmup"     "$BASE_PATCH_1EP"     '{"associative_bypass_enabled": true, "associative_bypass_dim": 64, "associative_bypass_adapter_checkpoint": "logs/wikitext-103_2026-07-15_10-53-46/best_model.pt", "associative_bypass_adapter_freeze_core": true, "epochs": 3, "warmup_fraction": 0.03, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.1, "min_lr": 0.002, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "AMBA_v2 (cancelled): strong frozen-adapter — redundant given v1 + recall_diagnostics"
# ==============================================================================
#
# AMB FEATURE-MAP SCREEN (2026-07-24). Native AMB stalled at epoch 2 with elu1 (DC-offset ->
# unselective running mean, +0.17 val vs D0 and widening) and NaN'd on the POD with relu2 (step
# 635, lr=3.25e-3 -- TINY, so unbounded squared features, not LR, caused it; relu2 then stayed
# FINITE in a local repro at C=512 but batch=4 vs the pod's 48 under-samples the extreme-
# activation batches that trigger it -- "finite locally" != "safe on the pod", hence this screen
# runs for real on the pod instead of trusting a batch-mismatched local test further). Three
# L2-normalized (structurally BOUNDED: <q,k> in [0,1] regardless of input scale/batch/depth)
# candidates, each CPU-smoked (compile clean, identity-at-init exact-0, finite fwd/bwd) and MQAR-
# sanity-checked locally (all finite, no NaN in short runs):
#   relu_l2     = normalize(relu(x))       -- MQAR 77%@500, ceiling ~80% (elu1/relu2 reach ~93%)
#   softplus_l2 = normalize(softplus(x))   -- MQAR 34%@500 (slow start), ~80%@1000
#   relu2_l2    = normalize(relu(x)^2)     -- MQAR 80%@500, ~81%@1000 (squaring-before-normalize
#                                              only partially recovers sharpness -- the L2-norm
#                                              itself, not the pre-nonlinearity, likely dominates
#                                              the ceiling loss: T-1 mismatches' background dot-
#                                              product mass rivals one bounded [0,1] true match)
# NOTE: a uniform scalar "temperature" on q or k was considered and REJECTED -- this read is a
# ratio y=(sum score*v)/(sum score), so any uniform positive rescale of every score cancels
# exactly; it cannot sharpen anything here (would need a per-score nonlinearity, which is what
# relu2_l2's squaring actually does, unlike a temperature).
# 1-EPOCH each via BASE_PATCH_1EP (auto-recomputed warmup ~2923 steps of ~9742 total -> reaches
# peak lr FASTER than the 5-epoch schedule and holds it ~6800 steps -> MORE exposure to the
# failure regime than epochs 1-2 of the full run gave, for ~1/5 the cost).
#
# RESULTS (2026-07-24, all three run):
#   relu_l2 (logs/wikitext-103_2026-07-24_17-53-20): FAILED -- stopped silently after step 500
#     (val 5.7860, not itself pathological); no NaN logged, likely a crash/exception the custom
#     logger doesn't capture (stderr, not checked).
#   softplus_l2 (logs/wikitext-103_2026-07-24_18-08-02): WINNER -- completed the FULL epoch
#     cleanly, monotonic descent through the entire peak-LR-hold AND cosine-decay window that
#     killed elu1 (step 3000 lr=7.50e-2 val=4.2451 -> step 9500 lr=1.73e-3 val=3.6514, zero
#     plateau). Epoch: train 3.6107 / val 3.6459. Sliding BPB 1.1563 (1-epoch; not comparable to
#     D0's 5-epoch 1.0436).
#   relu2_l2 (logs/wikitext-103_2026-07-24_20-57-27): FAILED, visibly -- val 5.6045 (s500) ->
#     14.6188 (s1000) -> 104.7077 (s1500) -> NaN (s2000). An exponential-blowup RAMP, not a sudden
#     spike -- different failure shape from relu_l2's quiet stop.
# HYPOTHESIS (why relu-based L2-norm failed but softplus_l2 didn't): plain relu(x) can output an
# EXACT all-zero d=64 vector (every channel negative for a token); F.normalize's eps guards the
# forward div-by-zero there, but its GRADIENT (~1/||v||) is large in the band just ABOVE the eps
# threshold, which a near-zero (not exactly-zero) relu output sits in often as training perturbs
# weights. softplus(x)=log(1+e^x) is NEVER exactly zero for finite x, so softplus_l2 never enters
# that band. relu2_l2 squares first, shrinking already-small values further toward the danger zone
# (relu(x)=0.01 -> squared=0.0001) -- consistent with it being the more severely/visibly broken of
# the two relu-based variants. Not fully proven (would need ||v|| distributions logged to confirm)
# but consistent with all observed evidence.
# DECISION: softplus_l2 promoted to AMBN_native_C512_5ep and AMBN_xlayer_C512_5ep below.
# run_ablation "AMBN_fmap_relu_l2_1ep AMB feature-map screen 1/3 — relu_l2 (bounded, ~80% MQAR ceiling)"     "$BASE_PATCH_1EP"     '{"associative_bypass_enabled": true, "associative_bypass_dim": 64, "associative_bypass_feature_map": "relu_l2", "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "AMBN_fmap screen 1/3: relu_l2 — bounded+selective, MQAR ceiling ~80%; safest local evidence so far"
# run_ablation "AMBN_fmap_softplus_l2_1ep AMB feature-map screen 2/3 — softplus_l2"     "$BASE_PATCH_1EP"     '{"associative_bypass_enabled": true, "associative_bypass_dim": 64, "associative_bypass_feature_map": "softplus_l2", "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "AMBN_fmap screen 2/3: softplus_l2 — bounded, smoother than relu; slower MQAR start (34%@500), similar ~80% ceiling"
# run_ablation "AMBN_fmap_relu2_l2_1ep AMB feature-map screen 3/3 — relu2_l2 (squared-then-normalized)"     "$BASE_PATCH_1EP"     '{"associative_bypass_enabled": true, "associative_bypass_dim": 64, "associative_bypass_feature_map": "relu2_l2", "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "AMBN_fmap screen 3/3: relu2_l2 — bounded + squaring-based sharpening, best local MQAR (80%@500, 81%@1000) of the three bounded options"
# ==============================================================================
#
# NATIVE AMB MINI (2026-07-24). The CLEAN architecture test: a fresh Mini trained WITH the AMB
# from step 0 — no frozen core, no over-converged donor, no adapter LR tension (base + AMB both
# fresh -> single lr=0.075, D0's recipe). ISO-everything vs D0 (1.0436, crawl-ON 5ep) except the
# +1.31M AMB: does integrating recall from the start help WaveletLM's WT-103 BPB? HONEST PRIOR:
# AMBA v1 came in FLAT (0.9798 vs D3 0.9797) with beta collapsing 1.0->0.17 (the model trained the
# memory then damped it) -> WT-103 looks recall-light, so native likely lands ~flat on D0 too. But
# this is the confound-free confirmation the paper needs. If it DOES beat D0 (>0.001), recall helps
# the from-scratch architecture after all. EVAL: sliding BPB vs D0 1.0436 + recall_diagnostics QRY-delta.
# STALL + NaN HISTORY (2026-07-24): run 1 (elu1 read, logs/wikitext-103_2026-07-24_09-43-03)
# trained clean in epoch 1 then STALLED at val~3.83 through epoch 2 while D0 descended to 3.66
# (+0.17 and widening) — elu(x)+1 carries a ~1.0 DC floor/dim -> q·k dominated by a ~d=64 constant
# offset -> the read collapses to an UNSELECTIVE running mean (redundant with decompose_bypass),
# fighting the spectral stack as its gain grows with LR. Tried relu2 (relu(x)^2, no DC floor) next
# — MQAR-faster than elu1 (78% vs 30%@500) but UNBOUNDED -> NaN'd on the pod at step 635, lr=3.25e-3
# (tiny -> not an LR problem, a magnitude problem). 3-way screen (relu_l2 / softplus_l2 / relu2_l2)
# above: relu_l2 and relu2_l2 BOTH failed (silent stop; visible NaN ramp, respectively — likely
# relu(x)'s exact-zero vectors hitting F.normalize's high-gradient band just above its eps clamp).
# softplus_l2 WON: completed the full 1-epoch screen cleanly through the entire peak-LR/decay
# window (sliding BPB 1.1563 @ 1ep). Promoted here. lr stays 0.075 (iso vs D0) — the fix targets
# read magnitude, not LR.
# run_ablation "AMBN_native_C512_5ep Native AMB Mini — fresh base + AMB from step 0, iso vs D0"     "$BASE_PATCH_5EP"     '{"associative_bypass_enabled": true, "associative_bypass_dim": 64, "associative_bypass_feature_map": "softplus_l2", "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "AMBN: native AMB Mini (softplus_l2 read, winner of the 3-way feature-map screen, fresh base+AMB, iso vs D0 1.0436) — elu1 STALLED epoch 2, relu2/relu_l2/relu2_l2 all NaN'd or crashed; softplus_l2 completed a clean 1-epoch screen (sliding BPB 1.1563)"
# ==============================================================================
#
# NATIVE AMB — softplus_s read (2026-07-25). The 3-way bounded screen showed the elu1 stall was a
# SELECTIVITY problem, and an init sim + our reproduction showed softplus_l2 (the screen "winner")
# is a near-running-mean: unrelated <q,k> cosine 0.86 (elu1 0.83!) -> a perfect match outweighs a
# random token only ~1.16x -> installs ~no recall -> tracks D0 flat (it just didn't STALL, because
# L2-bounded means its inert mean can't grow to fight the backbone the way elu1's unbounded one did).
# softplus_s = normalize(softplus(s*Wx)) with a LEARNABLE per-layer sharpness s=exp(log_s), init s=4:
# s does NOT cancel under L2-norm (softplus non-homogeneous), and softplus(s*z)/s -> relu(z) as
# s->inf, so s interpolates running-mean -> selective while staying STRICTLY POSITIVE (relu_l2/
# relu2_l2 crashed on exact-zero relu vectors hitting normalize's high-gradient band; softplus is
# never exactly zero). Local: identity-at-init exact-0, log_s live once the AMB engages, MQAR
# 76.6%@500 (vs softplus_l2's 34%) / ceiling ~85% (vs ~80%; still below the unstable relu-family's
# ~92%). s=exp(log_s) per layer reads out "how content-addressed is this layer" -> probe the
# checkpoint with tools/interpretability/amb_selectivity.py. iso vs D0 1.0436, lr 0.075. EXPECT
# flat-to-slightly-worse (WT-103 recall-light); the value is a stable+selective read for the later
# mixed-objective test, not a WT-103 BPB win. FOLLOW-UPS parked (do NOT bundle into this run, each
# confounds it): per-scale beta via coefficient-space write (per-scale AMB attribution for the
# paper); delta-rule/decay-gate v2 (erase-before-write, needs a chunkwise recurrent scan + recall-
# heavy eval); cross-layer memory (below, held — it changes phi's INPUT, can't fix in-phi selectivity).
# run_ablation "AMBN_softplus_s_C512_5ep Native AMB — learnable-sharpness softplus_s read, iso vs D0"     "$BASE_PATCH_5EP"     '{"associative_bypass_enabled": true, "associative_bypass_dim": 64, "associative_bypass_feature_map": "softplus_s", "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "AMBN_softplus_s: native AMB, learnable-sharpness read normalize(softplus(s*Wx)) (init s=4); MQAR 76.6%@500/~85% ceiling, stable+selective; iso vs D0 1.0436; probe learned s per layer post-run"
# ==============================================================================
#
# NATIVE AMB + CROSS-LAYER MEMORY (2026-07-24; idea: Ramon). Queued immediately after AMBN.
# Same fresh Mini + AMB from step 0, ISO to AMBN in every way EXCEPT
# associative_bypass_cross_layer=true: each block's AMB now reads
#   assoc_ln(x + gamma * previous_block_AMB_output)   (per-channel learned gamma)
# so the associative read COMPOUNDS across depth (a recall highway) instead of only seeing
# the prior read after the residual has diluted it. Adds only ~5.1K params total (C=512 gamma
# x 10 layers) -> the A/B is the MECHANISM, not capacity. Identity-at-init preserved (prev AMB
# output = 0 at init via zero-init out_proj; CPU-smoked 2026-07-24: identity exact-0, gamma
# grad live in layers 1+, guard fires vs checkpointing/multinodal, Dynamo trace clean).
# A/B: vs AMBN (isolates the cross-layer term) AND vs D0 1.0436. >0.001 over AMBN => depth-wise
# recall compounding helps; flat => WT-103 still just doesn't reward recall (same read as AMBN).
# COST ~6.1h (~$6). FALLBACK: if inductor ever chokes on the varying-arity AMB return, set
# compile:false (explicit-return threading is eager-safe). Requires the fp32 retrieval-scan fix
# (2026-07-24) that both AMBN and this inherit — without it the native AMB NaN's ~step 79-500.
# run_ablation "AMBN_xlayer_C512_5ep Native AMB + cross-layer memory — iso vs AMBN"     "$BASE_PATCH_5EP"     '{"associative_bypass_enabled": true, "associative_bypass_dim": 64, "associative_bypass_cross_layer": true, "associative_bypass_feature_map": "softplus_l2", "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "AMBN_xlayer: native AMB + cross-layer memory (assoc_ln(x+gamma*prev_AMB_out)); iso vs AMBN, isolates depth-wise recall compounding; +5.1K params only"
# ==============================================================================
#
# CTX1024_L9 RESUME (postponed AGAIN 2026-07-24 for the native AMB run; first postponed
# 2026-07-23 for AMBA — the MQAR validation made the associative-memory bypass the immediate
# priority). The fresh L9 arm above already trained into epoch 4 of run dir
# logs/wikitext-103_2026-07-23_06-49-44 before being
# stopped; resume_run continues it EXACTLY (model + Adagrad accumulators + scaler +
# RNG streams) from its last_checkpoint.pt (written every 2000 steps + at epoch end),
# so the finished A/B vs the 7-level CTX1024 1.0654 is step-for-step the run that never
# paused. Runs immediately after AMBA, before the scale-budget screen.
# resume_run "logs/wikitext-103_2026-07-23_06-49-44" "CTX1024_L9 resume (postponed for AMBA)"
# ==============================================================================

# ==============================================================================
# AMB CLOSE-OUT (2026-07-25). The AMB thread is PARKED after five WT-103 attempts
# (elu1 stalled epoch 2; relu2 NaN'd pod step 635; relu_l2 crashed step 500; relu2_l2
# diverged to NaN by step 2000; softplus_s NaN'd step ~10.1K after one clean epoch).
# Root cause of the *scientific* dead end is NOT numerical: WT-103 does not reward
# recall, so the AMB installs none. Measured on the softplus_l2 checkpoint with
# tools/interpretability/amb_selectivity.py: unrelated <q,k> 0.863-0.866 across all
# 10 layers (= its value at INIT -> never sharpened), read-weight entropy 1.000
# (perfectly uniform = a running mean), and ||beta*amb||/||x|| = 0.0000 (the model
# turned the module off). A BPB A/B on WT-103 therefore cannot answer "does AMB work",
# only "does it harm" — which it doesn't. These two runs close the thread on the ONE
# venue where the read is actually exercised, then AMB rests until post-release.
# Both are the small mixed-objective harness (WT-103 LM batches interleaved with
# generated MQAR recall batches, shared GPT-2 vocab), NOT train.py ablations — ~7M
# params, minutes not hours, ~$1 total for the pair.
#   1) Does the AMB install recall when the objective REWARDS it? (vanilla vs +AMB)
#   2) WHICH SCALES does that recall write into? (per-scale coefficient-space write,
#      one learned gain per scale — the attribution the full-width residual write
#      cannot give. CPU-smoked 2026-07-25: identity-at-init exact-0, all S gains
#      receive independent gradients, output verified != full-width write.)
# Read (1) as: MQAR acc rises AND WT-103 val stays ~vanilla => AMB earns its place.
# Read (2) as: the beta_s profile — coarse-heavy would mean recall is a long-span
# phenomenon; fine-heavy would mean it rides local detail. Either is a real result.
# mkdir -p logs
# python tools/interpretability/mqar_mixed.py --steps 3000 --feature_map softplus_l2 \
#     2>&1 | tee "logs/amb_mixed_$(date +%Y-%m-%d_%H-%M-%S).txt"
# python tools/interpretability/mqar_mixed.py --steps 3000 --feature_map softplus_l2 --per_scale --only assoc \
#     2>&1 | tee "logs/amb_mixed_perscale_$(date +%Y-%m-%d_%H-%M-%S).txt"
# git_commit_push "AMB close-out: mixed-objective recall test + per-scale write attribution"
# ==============================================================================

# ==============================================================================
# SCALE-BUDGET REALLOCATION + FROZEN-WAVELET TRANSFER + PRIME-POWER SCREEN
# (2026-07-20; runs BEFORE the remaining queue by request). Mini = the iteration
# machine: C=512/L=10 fully spectral, 5ep, MBS=48, lr=0.075 — every arm ~6.1h
# (~$6) except PP (~$10-12, S=19). Baselines: D0 (dyadic, crawl ON) = 1.0436
# sliding BPB; D2 (20ep) = 0.9906 for the winner's confirm. Noise floor 0.0010.
# EVIDENCE BASE: June crawl probe + July wavelet autopsy (both closed prime
# subbands as originally posed; both show coarse levels flatten to averagers
# and L4 reaching for lags 1-3) -> the live hypotheses are ladder REALLOCATION
# and cheap prime-hedge falsification. Schedule arms run crawl-OFF pairs where
# ladder identity must be clean (K=33 crawl windows cover all lags <=33 and
# blur schedule differences). New features (smoke-tested 2026-07-20):
# wavelet_dilation_schedule / prime_power_wavelet_basis_max(+_cap),
# lifting_import_checkpoint / lifting_freeze.
# DECISION RULES: SB2 vs SB0 = clean ladder effect (crawl off both). SB arms vs
# D0 = practical effect. PP kill rule: PP is 134.86M params (MEASURED 2026-07-27
# from the OOM'd run's breakdown; the ~133M estimate was close), so it must BEAT
# the width-law expectation for its own param class — 1.0036, not the ~1.015
# estimated here before the C-knee slope was fitted — else primes are dead
# weight bought with parameters.
#
# ---- SB4 RESULT (2026-07-27, logs/wikitext-103_2026-07-26_11-49-57/log.txt) ----
# BPB 1.0198 / PPL 24.1871 / sliding avg loss 3.1858 / best val 3.2198 (epoch 5)
# @ 139.08M, 5 epochs, 13.7h. (log.txt:412-415, best val at :392)
# PRIMARY QUESTION ANSWERED: untied lifting is STABLE at lr/2 (0.0375). June's NaN
# at the shared LR was a learning-rate failure, not an architectural one — per-layer
# lifting bases train fine, which also de-risks MOE0/MOEA (both use shared OFF).
# BUT judged on the same width-law rule applied to every other arm: at 139.08M the
# bar is 1.0016, so SB4 beats D0 (1.0436) by 0.0238 while MISSING its own param
# class by 0.0182. Verdict: untied lifting WORKS but is NOT worth its parameters —
# a negative result on the allocation question, a positive one on stability.
# Do NOT promote to default. Autopsy still owed: per-layer lag specialisation
# (do the untied bases diverge from each other, and along which dilations?) — the
# checkpoint exists, and that is the finding worth keeping from this run. FT1 tests "lifting = transferable
# router": log matched-step val vs D0 at steps 2K/4K/8K (convergence speedup)
# plus final BPB gap; FT2 (unfrozen warm-start) is the upper bound.
# run_ablation "SB0_C512_dyadic_crawlOFF_5ep Scale-budget control — dyadic ladder, crawl OFF"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "SB0: crawl-OFF dyadic control for all schedule arms; also measures what crawl itself is worth vs D0 1.0436"
# run_ablation "SB1_C512_coarseprune_5ep Scale-budget A — prune to levels {1,2,4,8,16}, crawl ON"     "$BASE_PATCH_5EP"     '{"levels": 5, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "SB1: coarse-prune (autopsy: coarse levels are redundant averagers; bypass covers global mean); fewer params than D0 — a tie is a WIN"
# run_ablation "SB2_C512_finedense_crawlOFF_5ep Scale-budget B — schedule {1,2,3,4,8,16,32}, crawl OFF"     "$BASE_PATCH_5EP"     '{"wavelet_dilation_schedule": [1, 2, 3, 4, 8, 16, 32], "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "SB2: fine-densified ladder (lag 3 promoted to a dedicated level, one coarse dropped), ISO-PARAM vs SB0; the cleanest ladder test in the screen"
# run_ablation "SB3_C512_ssmswap_5ep Scale-budget C — levels {1,2,4,8} + SSM bypass, crawl ON"     "$BASE_PATCH_5EP"     '{"levels": 4, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5], "decompose_bypass_ssm": true, "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "SB3: do coarse dyadic levels reduce to SSM poles? (the June probe's own question, run literally)"
# run_ablation "SB4_C512_untie_5ep Scale-budget D — untied lifting retry at halved LR"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "shared_lifting_weights": false, "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.0375, "min_lr": 0.00075, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "SB4: per-layer lifting bases (NaNd at shared LR in June; retried at lr/2 with resume armor); if stable, autopsy per-layer for lag specialisation"
# run_ablation "FT1_C512_frozenlifting_5ep Frozen-wavelet transfer — D2 lifting imported + FROZEN"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "lifting_import_checkpoint": "logs/wikitext-103_2026-07-14_09-11-32/best_model.pt", "lifting_freeze": true, "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "FT1: Release Pipeline frozen-wavelet transfer test — near-lossless vs D0 validates lifting-as-transferable-router + enables warm starts; compare matched-step val at 2K/4K/8K"
# run_ablation "FT2_C512_warmlifting_5ep Frozen-wavelet transfer — D2 lifting imported, UNFROZEN"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "lifting_import_checkpoint": "logs/wikitext-103_2026-07-14_09-11-32/best_model.pt", "lifting_freeze": false, "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "FT2: warm-start upper bound (imported, trainable); FT1-vs-FT2 gap = what freezing costs"
# run_ablation "PP1_C512_primepow11_crawlON_5ep Prime-power hedge — max=11, cap=128, crawl ON"     "$BASE_PATCH_5EP"     '{"prime_power_wavelet_basis_max": 11, "prime_power_dilation_cap": 128, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "PP1: 18-level prime-power union ladder (~133M est) crawl ON; kill rule: must beat ~1.015 (width-law at own params), not just D0"
# run_ablation "PP2_C512_primepow11_crawlOFF_5ep Prime-power hedge — max=11, cap=128, crawl OFF"     "$BASE_PATCH_5EP"     '{"prime_power_wavelet_basis_max": 11, "prime_power_dilation_cap": 128, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "PP2: crawl-OFF twin (clean ladder identity vs SB0); PP1-minus-PP2 = crawl redundancy of the prime rungs — the dead-weight question, answered directly"
#=============================================================================
#
# MIXTURE-OF-MIXERS (plans/mixture_of_mixers.md; implemented + smoke-tested 2026-07-21).
# Shared per-layer pool of E=4 full-width experts, learned STATIC per-scale top-2 router,
# usage-entropy aux loss (w=0.01; zero-init aux = ln(1/4) verified). Pool at E=4 SAVES
# ~12M params vs dedicated mixers (~60.6M total, est.) at ~20-35% extra mixer FLOPs (top-2).
# MOMA vs D0 (1.0436): existence test — a tie is a WIN (fewer params). MOMB = the frozen-
# lifting synergy arm (Ramon's conjecture: experts behind a fixed decomposition converge
# faster): judge on matched-step val at 2K/4K/8K vs MOMA AND FT1, not just final BPB.
# E-sweep {2,8,16} (the third-axis question) is CONDITIONAL on MOMA >= tie. NOTE: top-k
# routing may cause torch.compile graph breaks (python-int expert indices) — watch arm-1
# it/s; fallback "compile": false (~30-40% slower, still ~$9/arm).
# run_ablation "MOMA_C512_mom4top2_5ep Mixture-of-Mixers A — E=4, top-2, from scratch"     "$BASE_PATCH_5EP"     '{"mixer_mom_enabled": true, "mixer_mom_experts": 4, "mixer_mom_topk": 2, "mixer_mom_aux_weight": 0.01, "per_scale_mixer_widths": null, "levels": 7, "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "MOMA: MoM existence test vs D0 1.0436 — saves ~12M params, so a tie is a win; watch expert-usage collapse via aux magnitude"
# run_ablation "MOMB_C512_mom4_frozenlift_5ep Mixture-of-Mixers B — E=4 + frozen D2 lifting"     "$BASE_PATCH_5EP"     '{"mixer_mom_enabled": true, "mixer_mom_experts": 4, "mixer_mom_topk": 2, "mixer_mom_aux_weight": 0.01, "per_scale_mixer_widths": null, "lifting_import_checkpoint": "logs/wikitext-103_2026-07-14_09-11-32/best_model.pt", "lifting_freeze": true, "levels": 7, "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "MOMB: the synergy arm — experts trained behind a FROZEN imported decomposition; judged on convergence speed (matched-step val 2K/4K/8K) vs MOMA and FT1"
# ==============================================================================
#
# BLOCK-LEVEL MIXTURE OF EXPERTS (plans/mixture_of_mixers.md block-MoE section;
# implemented + smoke-tested 2026-07-22). E INDEPENDENT full WaveletLMBlocks per layer
# (own lifting + mixers + bypass), per-token LEARNED router, top-2, Switch load-balance
# aux (w=0.01; balanced value = 2.0 verified). DISTINCT from MoM (static per-scale router,
# shares only mixers). Release-pipeline "next step after the current screen."
# REALITY CHECK (measured, not iso-param): D0=72.89M; single-expert shared-OFF=139.08M;
# E=4=479.13M. Dense eval => ~E x block FLOPs (routing a SEQUENCE transform can't be
# sparsified). So E=4 = 6.6x D0 params AND ~4x compute: a capacity+routing test, NOT an
# efficiency one. LR=0.0375 (shared_lifting OFF is SB4's untied regime, which NaN'd at
# 0.075). MOE0 is the essential control: 1 expert, shared-OFF — isolates "routing" from
# "sharing off" (which alone adds +66M). Judge MOEA vs MOE0 (does routing beat one big
# expert?) AND vs what 479M-of-width would give — NOT vs D0. E=2 (252M) is the cheaper
# existence test if E=4 is too costly. VRAM: gradient_checkpointing ON for MOEA (protocol-
# clean); if still OOM, drop MBS (breaks the MBS=48 protocol -> BPB carries the small-batch
# handicap, NOTE it in the log read). Est: MOE0 ~$10, MOEA ~$30-50.
# run_ablation "MOE0_C512_1expert_sharedOFF_5ep Block-MoE control — 1 expert, shared lifting OFF"     "$BASE_PATCH_5EP"     '{"shared_lifting_weights": false, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.0375, "min_lr": 0.00075, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "MOE0: 1-expert shared-OFF control (139M); isolates routing from lifting-untying; also a second untied-lifting stability datapoint vs SB4"
# ==============================================================================
#
# FAST-WEIGHT PKM RETRY (rewired 2026-07-26; model.py FastWeightPKM).
# FwPKM has been in the codebase since April and NEVER contributed positively. An audit
# found the reason is not "memory doesn't help" but four defects — three in the wiring,
# one in every evaluation it ever received:
#   1. NO IDENTITY AT INIT. alpha_fwpkm defaulted to 1.0, so a random value-mixture entered
#      the residual at FULL GAIN from step 0 — the only module in the block that did not
#      start as identity. Smoke-measured perturbation at init: max|dlogit| = 5.63. FIXED:
#      alpha_fwpkm inits to 0.0 (gradient still reaches it immediately; no deadlock).
#   2. UNSELECTIVE SOFTMAX. Keys init at std 0.02 over half_dim=256 gave dot products of
#      std ~0.16, so the top-k softmax was EXACTLY uniform at init (measured entropy 3.4657
#      vs ln(32) = 3.4657) and the lookup returned the MEAN of 32 random rows. Same DC-floor
#      pathology that stalled the AMB's elu1 map. NOTE a plain 1/sqrt(d) scale is the WRONG
#      SIGN here (it shrinks already-tiny scores) — tried, measured, rejected. FIXED: cosine
#      scores x LEARNED temperature (log_tau, init ln 10) -> max weight 0.1145 vs 1/32 =
#      0.0312, i.e. 3.7x peaked at init, and sharpness is now trained not accidental.
#   3. NO USAGE REGULARISATION. top-32 of 8281 leaves rarely-selected keys at random init
#      forever (dead memory). MoM and block-MoE both carry a usage penalty; FwPKM had none.
#      FIXED: fwpkm_aux_weight (default 0.01) on negative selection entropy, TRAIN ONLY —
#      _usage_aux is None at eval, so val loss and BPB stay uncontaminated.
#   4. IT WAS ALWAYS TESTED IN THE SHADOW OF AN MLP. Of 277 historical fwpkm_enabled runs,
#      276 had mlp_expansion >= 1 (246 at expansion 10) and 277 of 277 had projections on.
#      At C=2048 an expansion-10 MLP is ~84M dense params/layer writing into the SAME
#      mem_out sum — a 4M sparse memory was strictly redundant. FwPKM has NEVER run in the
#      current fully-spectral (mlp_expansion=0, skip_proj_out=true) architecture, where
#      nothing else occupies that slot. The old null does not transfer.
# A/B TARGET: D0 = 1.0436 BPB / val 3.2600 @ 72.89M (logs/wikitext-103_2026-07-12_08-04-38,
# log.txt:412). FWP1 is D0's config with ONLY the fwpkm keys changed.
# DECISION RULE — judge against the WIDTH LAW at its own params, not against D0 (same
# convention as PP1). FwPKM adds 4,548,610 params/layer x 10 = 45.49M, so 72.89M -> 118.37M.
# Fitting BPB vs ln(params) over the four C-knee points (C=200 1.1562 / 300 1.1014 /
# 400 1.0674 / 1024 0.9805) gives -0.065 BPB per e-fold; ln(118.37/72.89) = 0.485, so width
# would buy ~0.031 BPB. Bar is therefore ~1.012 (range 1.009-1.017 across slope estimates).
#   BPB >= 1.0426  -> no effect (within noise 0.0010 of D0); the fixes did not help. CLOSE
#                     FwPKM out permanently and delete it (no dead code).
#   1.012 - 1.0426 -> memory works but WIDTH IS A BETTER BUY at the same params. Report as a
#                     negative result; do not ship. Optional follow-up at num_keys=1024
#                     (+8.0M only), where the param bar drops to ~1.037.
#   BPB < 1.012    -> sparse memory beats width iso-param. Real finding; then run the
#                     num_keys sweep {1024, 8281, 16384} and inspect key-usage histograms.
# WHY THIS SLOT (ahead of MoM/MoE): FwPKM (sparse memory, top-32 of 8281) and block-MoE
# (sparse compute, top-2 of 4) are the same bet on conditional computation. Knowing whether
# fine-grained sparsity pays here is worth having BEFORE spending 4x compute on MOEA.
# COST: D0 ran 6h11m (08:04:41 -> 14:15:26). Expect ~7-8h with the lookup, ~$7-8 on a 5090.
# VRAM: +45M params plus DENSE EmbeddingBag grads (8281x512 fp32 = 17MB/layer, 170MB total)
# — comfortable at MBS=48 on 32GB, but if it OOMs set "gradient_checkpointing": true rather
# than dropping MBS (preserves iso-batch comparability with D0).
# FALLBACK: if it NaNs in warmup, halve lr to 0.0375. If the aux dominates (train loss drops
# by ~0.09 with no val movement — the bound is ln(8281)*0.01 = 0.090), set fwpkm_aux_weight
# to 0.001. The OPPOSITE also needs watching: in the CPU smoke test (single repeated batch,
# so a memorisation setting where collapse is expected) usage entropy fell 6.20 -> 3.54 nats
# in 40 steps at weight 0.01 — effective keys ~495 -> ~34. The task gradient dominates the
# penalty, so 0.01 is a nudge, not a constraint; if the autopsy shows severe collapse, raise
# to 0.05. Also note tau self-adjusted 10 -> 7.6 immediately, i.e. the model prefers a softer
# softmax than the init — expected, and exactly why sharpness was made learnable.
# AUTOPSY REGARDLESS OF OUTCOME: histogram key-selection frequency per layer and
# read alpha_fwpkm — if alpha stays ~0 the model declined the memory, which is itself the
# answer to "why did it never contribute".

# ==============================================================================
# ===== WaveletLM MICRO (C=256) - PRIORITISATION TIER, NOT A SCREEN ============
# ==============================================================================
# Ramon 2026-07-27. Purpose: run the cheap ablations at C=256 first to PRIORITISE
# which features deserve Mini (C=512) compute, then test combinations of whatever
# works. The Mini arms below are NOT removed and NOT replaced.
#
# THE RULE IS ASYMMETRIC, AND THAT IS THE WHOLE POINT:
#   PASS at Micro -> promote to Mini for confirmation.
#   FAIL at Micro -> record as "failed at Micro". Do NOT kill the feature.
# This project has the canonical counterexample. Removing projections scored
# +0.0115 BPB at C=100 (a clear failure) and -0.0079 at C=1024 (a win). That
# feature is now skip_proj_out=true, the default, and the "fully spectral"
# headline. A symmetric Micro screen would have killed the biggest architectural
# win in the repo. Treat Micro negatives as weak evidence unless there is a
# mechanistic reason the sign cannot depend on width. PROJ_Micro at the end of
# this block measures that bias directly.
#
# WHAT CHANGES FROM MINI, AND WHAT DELIBERATELY DOES NOT:
#   C 512 -> 256, and LR doubles per the 48/C width rule (shared 0.075 -> 0.15,
#   untied 0.0375 -> 0.075, min_lr = lr/50). NOTHING ELSE MOVES.
#   * levels stay 7: levels are a TIME-axis property (dilations over the
#     sequence), independent of C - they track block_size, not width.
#   * per_scale_mixer_widths stay: they are C-multipliers, so the 0.5 scales are
#     128 channels at C=256, still reasonable.
#   * MBS stays 48: the NaN ceiling is width-bound and batch-INVARIANT
#     (project_sqrt_batch_lr_rule_fails), so LR transfers; and a different batch
#     would forfeit comparability with Mini, which is what promotion depends on.
#     A bigger batch also buys little here - the LM head is 71 pct of compute at
#     C=256 and its matmul already saturates.
#   * dropout stays: tuned at C=512 and probably too strong for a 24.78M model on
#     655M tokens, but it is CONSTANT ACROSS EVERY MICRO ARM, so the internal
#     comparisons - the only ones these runs exist for - remain valid.
#
# COST: ~45 dollars for all nine, vs ~144 for the Mini MoE pair alone.
#
# PERF POSTSCRIPT (2026-07-28, tools/perf_probe.py on the current pod — READ BEFORE
# TRUSTING ANY $/RUN ESTIMATE IN THIS BLOCK):
#   * GPU healthy: 219 TFLOPS sustained fp16. Launch rate 9.8us. NOT compute-bound:
#     eager C=256 step is 596ms where compute-bound would be ~37ms.
#   * THE MODEL ISSUES ~41,726 CUDA KERNELS PER STEP (eager). At 9.8us/launch that is
#     ~410ms of pure dispatch = 69% of the step — and kernel count is C-INDEPENDENT,
#     which is why C=256 ran no faster than C=512 overall. Chief suspect: the crawl
#     (7 levels x 33 offsets x 10 layers, fwd+bwd). This is the architecture's real
#     performance bug and the quantitative case for dispatch work (crawl batching,
#     CUDA graphs, decimation) during the suspension.
#   * OVERSUBSCRIPTION HYPOTHESIS DEAD: OMP_NUM_THREADS=4 changed launch rate 9.8->9.5us
#     (nothing) and did not help the model step. Do NOT add thread env vars to runs.sh.
#   * The earlier "C=256 == C=512 at 1.30 it/s" comparison was TRAIN+EVAL wall rate; the
#     eval half (1000 dispatch-bound forwards) is itself C-insensitive and flattened the
#     comparison. With eval_batches=64 (fixed 2026-07-27) the flattening term is ~gone.
#   * DECISION RULE FOR THIS TIER: read the FIRST 500-step interval of the Micro
#     baseline under the eval fix. <=~210s -> Micro is ~2x cheaper than Mini and the
#     tier proceeds as written. >=~350s -> dispatch floor erases width entirely; fold
#     the Micro arms back into Mini (same price, no sign-flip risk) except PROJ_Micro,
#     which stays (the calibration question stands regardless of cost).
# gradient_checkpointing is left ON for arms that already OOM'd at Mini rather
# than re-risk a wasted slot, and OFF for the baselines, which never have.
# NOTE FWP1 (Mini, C=512) is deliberately still running and NOT converted: at
# C=256 FwPKM would be +90 pct of params instead of +62 pct unless num_keys is
# retuned, which FWPKM_Micro above does.
# run_ablation "Micro_C256_L10_noMLP_5ep Micro baseline - C=256 shared lifting"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216}'     "Micro baseline (24.78M): the reference every other Micro arm is judged against, AND the missing C=256 rung of the C-knee ladder. Critically it is the FIRST C-knee point at MBS=48 besides D0 - the -0.065/e-fold slope was fitted entirely on MBS=8 points, and every width-law kill rule in this file depends on it. Two same-batch anchors let us check that slope directly for ~2 dollars"
run_ablation "Micro-untied_C256_L10_5ep Micro-untied - C=256 per-layer lifting bases"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "shared_lifting_weights": false}'     "Micro-untied: the control the MoE arms need, since block_moe HARD-REQUIRES shared_lifting_weights=false (model.py:3773). Also the Micro echo of SB4 (1.0198/1.0201 at C=512), so it doubles as a width-transfer check on the untied-lifting verdict"
run_ablation "FWPKM_Micro_C256_5ep FwPKM Micro - rewired memory at matched param share"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": true, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "fwpkm_num_keys": 5625, "fwpkm_top_k": 32, "fwpkm_heads": 1, "fwpkm_aux_weight": 0.01}'     "FWPKM Micro: num_keys=5625 (75^2) chosen so the memory is +61.5% of base params, matching Mini FWP1 at +62.4%. WITHOUT retuning, 8281 keys would be +90% here, because the value table scales as num_keys*C (linear) while blocks scale as C^2 - memory gets relatively MORE expensive as the model shrinks. Verified by building it: 40,025,311 params. sub_keys=75 >= top_k=32 so half_k does not clamp"
run_ablation "MOEA_Micro_C256_5ep Block-MoE Micro - E=4 full-block experts, top-2"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "shared_lifting_weights": false, "block_moe_enabled": true, "block_moe_experts": 4, "block_moe_topk": 2, "block_moe_aux_weight": 0.01}'     "MOEA Micro: vs Micro-untied = does per-token routing help at all. NOTE MoE is much cheaper at Micro - the tied LM head is 71 pct of compute at C=256 vs 55 pct at C=512, so E=4 costs 1.87x the baseline here against 2.35x at Mini"
run_ablation "MLR1_Micro_C256_5ep Multiresolution ladder Micro - E=4 coarse-only experts"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "shared_lifting_weights": false, "block_moe_enabled": true, "block_moe_experts": 4, "block_moe_topk": 2, "block_moe_aux_weight": 0.01, "block_moe_scale_ladder": true}'     "MLR1 Micro: the ONLY valid A/B is vs MOEA_Micro (identical params and compute, the ladder is the sole difference). The MoE capacity effect - the thing most likely to inflate at small C - appears in BOTH arms and cancels. What does NOT cancel is the ladder information cost, which should bite HARDER at C=256 where there is less redundancy, so a WIN here is strong evidence while a LOSS stays weak"
run_ablation "PP1_Micro_C256_5ep Prime-power Micro - max=11, cap=128, crawl ON"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "prime_power_wavelet_basis_max": 11, "prime_power_dilation_cap": 128}'     "PP1 Micro: 18-level prime-power union ladder. Judge PP1-minus-PP2, not PP1 vs baseline - the subtraction answers the crawl-redundancy question and is far more robust to width than absolute BPB"
run_ablation "PP2_Micro_C256_5ep Prime-power Micro - max=11, cap=128, crawl OFF"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "prime_power_wavelet_basis_max": 11, "prime_power_dilation_cap": 128}'     "PP2 Micro: crawl-OFF twin. PP1-minus-PP2 = crawl redundancy of the prime rungs, answered directly"
run_ablation "MOMA_Micro_C256_5ep Mixture-of-Mixers Micro - E=4, top-2"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": null, "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": true, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "mixer_mom_enabled": true, "mixer_mom_experts": 4, "mixer_mom_topk": 2, "mixer_mom_aux_weight": 0.01}'     "MOMA Micro: the one arm that SAVES params (at Mini it is 57.08M vs D0 72.89M), so a tie is a win. Watch expert-usage collapse via aux magnitude"
run_ablation "PROJ_Micro_C256_5ep Micro CALIBRATION - skip_proj_out OFF"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": false, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216}'     "PROJ Micro (Ramon 2026-07-27): the run that measures HOW MUCH MICRO LIES. skip_proj_out scored +0.0115 at C=100 (a failure) and -0.0079 at C=1024 (the fully-spectral headline) - a MEASURED SIGN FLIP. Re-running it at C=256 locates where that flip happens and calibrates every other Micro verdict here. If skip_proj_out is already winning at C=256, Micro is a trustworthy prioritiser; if it is still losing, Micro systematically punishes this whole class of simplification and Micro negatives mean very little"

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
run_ablation "FT1_Micro_C256_5ep Frozen-wavelet transfer Micro - Micro lifting imported + FROZEN"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "lifting_import_checkpoint": "logs/wikitext-103_2026-07-27_20-38-43/best_model.pt", "lifting_freeze": true}'     "FT1 Micro: donor is the MICRO BASELINE's own checkpoint, NOT D2 - a C=512 donor cannot load into a C=256 model (train.py:1402 load_state_dict strict=True raises on shape mismatch). This makes it a same-config transfer test (is a converged lifting reusable across runs) rather than Mini's cross-config one. It also SIDESTEPS the blocker that killed Mini FT1/FT2: the donor is generated pod-side, so no S3 upload is needed. Compare matched-step val at 2K/4K/8K vs the Micro baseline"
run_ablation "FT2_Micro_C256_5ep Frozen-wavelet transfer Micro - Micro lifting imported, UNFROZEN"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.15, "min_lr": 0.003, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "weight_decay": 2e-06, "fwpkm_enabled": false, "gradient_checkpointing": false, "layers": 10, "C": 256, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.1, "dropout_lm_head": 0.216, "lifting_import_checkpoint": "logs/wikitext-103_2026-07-27_20-38-43/best_model.pt", "lifting_freeze": false}'     "FT2 Micro: warm-start upper bound. FT1-minus-FT2 = what freezing costs at this width. Same pod-side donor as FT1"

run_ablation "FWP1_C512_fwpkm8281_5ep Fast-weight PKM retry — rewired, D0 config + memory"     "$BASE_PATCH_5EP"     '{"fwpkm_enabled": true, "fwpkm_num_keys": 8281, "fwpkm_top_k": 32, "fwpkm_heads": 1, "fwpkm_aux_weight": 0.01, "levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.075, "min_lr": 0.0015, "micro_batch_size": 48, "eval_interval": 500, "checkpoint_interval_steps": 2000, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 0, "skip_proj_out": true, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "gradient_checkpointing": false, "layers": 10, "C": 512}'     "FWP1: first FwPKM trial in the fully-spectral architecture and the first with identity-init, a selective softmax, and a usage penalty; 118.37M, so the bar is the width law (~1.012), not D0 (1.0436)"

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

