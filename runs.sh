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

# # ---- Capacity restoration (vs shared pre-baseline 1.1156) ---------------------
# # Each production-capacity component restored individually on the pre-baseline
# # recipe (identity, K=33, T4 dropouts, WD=1e-6, no recurrence), then all four.
# # All five compare against the same 1.1156 reference as the 2x2 and the
# # recurrence test — every decision axis shares one anchor; the chosen combined
# # baseline then gets its own confirmation run (the declared T5 row).
# # Also records A5000 VRAM/runtime per component to calibrate the B200 plan.

# run_ablation "T5_cap_mlp20_1ep T5 capacity — mlp_expansion 20 (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20}'     "T5_cap_mlp20_1ep: capacity restoration — MLP expansion 10->20 vs pre-baseline (1.1156)"

# run_ablation "T5_cap_pkm_1ep T5 capacity — PKM on, 16384 keys (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "pkm_enabled": true, "pkm_num_keys": 16384}'     "T5_cap_pkm_1ep: capacity restoration — PKM enabled @16384 vs pre-baseline (1.1156)"

# run_ablation "T5_cap_fwpkm_1ep T5 capacity — FwPKM 16384 keys (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "fwpkm_num_keys": 16384}'     "T5_cap_fwpkm_1ep: capacity restoration — FwPKM 8281->16384 vs pre-baseline (1.1156)"

# run_ablation "T5_cap_untied_1ep T5 capacity — untied LM head (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "tie_embedding_to_lm_head": false}'     "T5_cap_untied_1ep: capacity restoration — untied embedding/head vs pre-baseline (1.1156)"

# run_ablation "T5_cap_all_1ep T5 capacity — all four restored (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "pkm_enabled": true, "pkm_num_keys": 16384, "fwpkm_num_keys": 16384, "tie_embedding_to_lm_head": false}'     "T5_cap_all_1ep: capacity restoration — all four components; presumptive new-baseline capacity form"

# ---- T5 declared baseline: combined confirmation run --------------------------
# Resolved verdicts: identity transform, crawl K=33, dropout descent stack +
# WD=2e-6, recurrence OFF, capacity = MLP-20 ONLY (PKM/FwPKM/untied were
# inert-or-harmful at 1ep, deferred to a 5ep re-test). This run confirms the
# wins stack and becomes the declared T5 baseline. Expectation: reg(-0.0038) +
# MLP(-0.0075) on 1.1156, sub-additive -> ~1.105-1.107.
# run_ablation "T5_baseline_combined_1ep T5 declared baseline — identity+K33+reg+MLP20 (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6}'     "T5_baseline_combined_1ep: declared T5 baseline confirmation — identity + crawl K=33 + dropout stack + WD=2e-6 + MLP-20; capacity MLP-only"

# # Paired FwPKM ablation: the combined baseline MINUS FwPKM (fwpkm_enabled=false),
# # else identical. First test of FwPKM-vs-nothing on the new recipe (prior runs
# # only tested widening / a 2nd memory). If within noise of the combined run,
# # the declared baseline drops FwPKM for a smaller config; if it regresses, the
# # one carried memory module is confirmed to earn its slot. 1ep verdict is
# # provisional for memory (pending 5ep re-test).
# run_ablation "T5_baseline_nofwpkm_1ep T5 combined baseline MINUS FwPKM (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false}'     "T5_baseline_nofwpkm_1ep: combined baseline minus FwPKM — FwPKM-vs-nothing on the new recipe; drop it if within noise of the combined run"


# ==============================================================================
# More Layers (1ep depth gate on the locked no-memory T5 recipe)
# Declared T5 baseline = L=1 (1.1073 / 455.55M, logs/...06-14_16-08-56). These
# add L=2/3/4 (lean, no memory) + an L=3 learned_residual=off control + an L=4
# FULL-CAPACITY probe (restores PKM@16384 + FwPKM@16384 + untied) as a cheap 1ep
# preview of capacity-at-depth before the 5ep ceiling arm. lean-L4 vs full-L4
# isolates capacity at L=4. L=2 first (overnight timing scope on the A5000).
# ==============================================================================

# ---- L=3 TRUE no-residual — RE-TEST (spectral-epsilon init removed) — RUNS FIRST
# The first disable_residual run (07-28, NaN @5k) was confounded: proj_out's 1e-3
# "spectral epsilon" init (a residual-regime "start-near-identity" trick) starved
# the spectral output to ~1e-3 with no residual to carry it, so ln2's backward
# (1/std) amplified gradients ~100-300x/layer -> NaN. model.py now drops that 1e-3
# init when disable_residual=true (proj_out keeps its O(1) default). FAIR test now:
# does depth collapse without the residual, or degrade gracefully? REQUIRES the
# updated model.py on the pod (git pull). LR 0.0225 matched to the residual runs;
# if it still NaNs near peak, lower the LR (the dominant amplifier is now gone).
# run_ablation "T5_L3_nores_v2_1ep More Layers — L=3 no-residual, spectral-eps removed (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 3, "disable_residual": true, "per_layer_embedding": false}'     "T5_L3_nores_v2_1ep: corrected no-residual control — disable_residual + per_layer_embedding=false, proj_out 1e-3 spectral-epsilon dropped via model.py guard (prior NaN source); fair test of whether depth needs the residual. vs L=3 baseline 1.0945."

# run_ablation "T5_L2_1ep More Layers — L=2 lean (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 2}'     "T5_L2_1ep: More Layers depth gate — L=2, no-memory T5 recipe; vs declared L=1 baseline (1.1073)"

# run_ablation "T5_L3_1ep More Layers — L=3 lean (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 3}'     "T5_L3_1ep: More Layers depth gate — L=3, no-memory T5 recipe"

# run_ablation "T5_L4_1ep More Layers — L=4 lean (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 4}'     "T5_L4_1ep: More Layers depth gate — L=4, no-memory T5 recipe"

# run_ablation "T5_L3_nores_1ep More Layers — L=3 learned_residual OFF (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 3, "learned_residual": false}'     "T5_L3_nores_1ep: learned-residual contribution control at depth — L=3 with residual off vs L=3 on"

# ---- L=3 TRUE no-residual control (disable_residual) — RUNS FIRST -----------
# The CORRECTED residual test. learned_residual=false (commented run above) only
# drops the per-sublayer learned alpha (init 1.0); the residual x = x + f(x)
# STAYS, which is why on/off came out near-identical. disable_residual=true
# removes the carry entirely (x <- f(x)) in every forward path. per_layer_embedding
# is also turned OFF so the per-block token-embedding re-injection can't stand in
# for the residual (the cross-block analog of input anchoring). If L=3 no-residual
# collapses toward L=1 (1.1073), the residual stream is the load-bearing depth
# mechanism. ~17.9 GB (fits 4090/5090).
# REQUIRES the updated model.py (disable_residual flag) on the pod — a stale
# model.py silently IGNORES the key (config.get default False) and runs WITH the
# residual. git pull before launching.
# run_ablation "T5_L3_nores_true_1ep More Layers — L=3 TRUE no-residual (disable_residual, 1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 3, "disable_residual": true, "per_layer_embedding": false}'     "T5_L3_nores_true_1ep: TRUE no-residual control at depth — disable_residual=true + per_layer_embedding=false (x<-f(x), no carry, no embedding skip); vs L=3 baseline 1.0945. Tests whether the residual stream is the load-bearing depth mechanism."

# run_ablation "T5_L4_fullcap_1ep More Layers — L=4 FULL capacity (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 4, "pkm_enabled": true, "pkm_num_keys": 16384, "fwpkm_enabled": true, "fwpkm_num_keys": 16384, "tie_embedding_to_lm_head": false}'     "T5_L4_fullcap_1ep: capacity-at-depth probe — L=4 + MLP20 + PKM@16384 + FwPKM@16384 + untied; vs lean L=4 isolates capacity; 1ep preview of the 5ep ceiling arm"

# ---- Iterative deepening L=5+ (STAGED, COMMENTED) ----------------------------
# VRAM: ~+4.5 GB/layer at C=2048, so L=5 ~26.9 GB and L=6 ~31.4 GB — both OOM the
# 24 GB 4090 (and A5000). DO NOT uncomment on a 24 GB card; needs a 5090 (32 GB,
# ~L=5-6) or B200 (L=7+). Memory setting defaults to no-memory (predicted L=4
# winner per the redundancy finding); if T5_L4_fullcap beats T5_L4 by > noise,
# switch these to full capacity (pkm/fwpkm @16384 + untied). L=6 is gated on L=5
# clearing the ~0.0010 noise floor vs L=4; L=7+ added iteratively as results come.

# run_ablation "T5_L5_1ep More Layers — L=5 (1ep, >=32GB GPU)"      "$BASE_PATCH_1EP"      '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 5}'      "T5_L5_1ep: iterative depth — L=5, no-memory (switch to full if L=4-full won); needs >=32GB"

# L=6 → moved to runs_6000.sh (6000), right after C=4096 — its ~31.3 GB has
# headroom on the 96 GB 6000 (borderline on the 5090), and the 6000 continues the
# deep dive from there (L=10, L=15, …). Runs in PARALLEL with L=5 above.


# ==============================================================================
# More Epochs (5ep confirmation arms — the headline candidates)
# 5-epoch re-runs of the lean depth sweep, L=1..4, all of which cleared the
# ~0.0010 noise floor at 1ep. Same lean no-memory T5 recipe; only epochs differ
# (BASE_PATCH_5EP, epochs=5). warmup_fraction=0.3 auto-scales the warmup to the
# longer schedule. The "Max from More Layers" 5ep row is deferred until the depth
# ceiling is known (L=5+ still pending). 5090, sequential — runtime ~5x the 1ep
# arms (L=4 ~50h), so this is the big compute block; comment out any arm to defer
# it. (Optional: bump eval_interval 250->1000 to save ~2h/arm.)
# ==============================================================================

# run_ablation "T5_L1_5ep More Epochs — L=1 lean (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1}'     "T5_L1_5ep: More Epochs confirmation — L=1, no-memory T5 recipe, 5 epochs"

# run_ablation "T5_L2_5ep More Epochs — L=2 lean (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 2}'     "T5_L2_5ep: More Epochs confirmation — L=2, no-memory T5 recipe, 5 epochs"

run_ablation "T5_L3_5ep More Epochs — L=3 lean (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 3}'     "T5_L3_5ep: More Epochs confirmation — L=3, no-memory T5 recipe, 5 epochs"

run_ablation "T5_L4_5ep More Epochs — L=4 lean (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 4}'     "T5_L4_5ep: More Epochs confirmation — L=4, no-memory T5 recipe, 5 epochs"


# ==============================================================================
# More Width (C) — C=4096, L=1 width-response anchor (1ep)
# First width scale-up point. Est. ~1.6B params (embedding ~2x linear; layers +
# wavelets ~4x quadratic in Cp), ~26-29 GB peak on the 5090 — SHOULD fit 32 GB
# but it's TIGHT; VERIFY AT LAUNCH. If it OOMs, drop micro_batch_size 8->4 (set
# grad_accum 1->2 to hold the effective batch) — sheds ~3-4 GB of activations.
# LR inherited at 0.0225 (no width-LR retune yet — this anchor measures the width
# response at the depth-tuned LR; a width-LR sweep is a separate step). C=8192/
# 16384 and the max-depth/5ep width cells need the B200 (deferred per the 32 GB
# ceiling).
# ==============================================================================

# run_ablation "T5_C4096_L1_1ep More Width — C=4096 L=1 (1ep, 5090 ~tight)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "layers": 1, "C": 4096}'     "T5_C4096_L1_1ep: More Width anchor — C=4096, L=1, no-memory T5 recipe; ~1.6B params; verify VRAM fits 32GB at launch (MBS 8->4 fallback if OOM)"


# ==============================================================================
# T6-candidate ablations on C=2048/L=5 (5090) — dual-cadence (ep=1 AND ep=5).
# Two init-to-identity cross-layer-flow enrichments, each measured against the
# shared / no-skip L=5 baseline (ep=1 = 1.0831, logs/wikitext-103_2026-06-17_10-06-29;
# ep=5 TBD). Whichever clears the ~0.0010 noise floor IN ISOLATION graduates into
# the T6 baseline; if neither helps alone, the combined T6 test is skipped.
#   (1) UNTIED LIFTING (shared_lifting_weights=false): per-layer temporal basis.
#       ep=1 already runs on the 6000; this adds the ep=5 confirmation.
#   (2) CROSS-LAYER DENSE SKIPS (cross_layer_dense_skips=true): each layer reads a
#       learned lower-triangular combo of all prior layer outputs (model.py layer
#       loop), identity-init so step 0 == the plain stream. Meaningful at L>=3.
# C=2048/L=5 = ~26.9 GB, fits the 5090 (32 GB) at both epoch counts.
# SMOKE-TEST the skips: identity init => the FIRST eval must match the no-skip
# run's loss, then diverge as layer_dense_A learns. If they differ at step 0, stop.
# ==============================================================================

run_ablation "T5_L5_untied_5ep Untied Lifting — C=2048 L=5 shared_lifting_weights=false (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "shared_lifting_weights": false, "layers": 5}'     "T5_L5_untied_5ep: untied per-layer lifting at 5ep — pairs with the 6000 ep=1 run; A/B vs shared L=5; re-probe per-layer dilation_logits"

run_ablation "T6_L5_xskip_1ep Cross-Layer Skip — C=2048 L=5 dense skips (1ep)"     "$BASE_PATCH_1EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "cross_layer_dense_skips": true, "layers": 5}'     "T6_L5_xskip_1ep: cross-layer dense skips (init-to-identity) at L=5/1ep; A/B vs shared no-skip L=5 (1.0831); smoke-test first eval == baseline then diverge"

run_ablation "T6_L5_xskip_5ep Cross-Layer Skip — C=2048 L=5 dense skips (5ep)"     "$BASE_PATCH_5EP"     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_crawl_k": 33, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity", "mlp_expansion": 20, "dropout_embedding": 0.18, "dropout_projection": 0.09, "dropout_mixer": 0.09, "dropout_mlp": 0.10, "dropout_lm_head": 0.216, "weight_decay": 2e-6, "fwpkm_enabled": false, "cross_layer_dense_skips": true, "layers": 5}'     "T6_L5_xskip_5ep: cross-layer dense skips at L=5/5ep; A/B vs shared no-skip L=5 (5ep TBD)"
