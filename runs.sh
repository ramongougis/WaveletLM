#!/bin/bash
# Tests 2-4: Combined parameter reduction + VRAM reallocation variants.
# All tests apply the four reductions (mlp_expansion=10, pkm_enabled=false,
# fwpkm_num_keys=8281, tie_embedding_to_lm_head=true) at L=1 / E=5, varying
# how the freed VRAM is used.
#
#   Test 2: Max EBS — MBS=64, GA=1, bs=256, levels=5 (uses ~25 GiB / 32 GiB)
#   Test 3: Larger block_size — MBS=8, GA=1, bs=1024, levels=5
#   Test 4: Min EBS + max block_size — MBS=1, GA=1, bs starts at 16384 and
#           halves on NaN detection (16384→8192→4096→2048→1024→512→256) until
#           a stable run completes; levels=5 throughout. bs=16384 puts
#           WaveletLM in directly comparable territory with Hyena's 16k context
#           and uses ~27 GiB / 32 GiB VRAM (vs MBS=1, bs=8192 which would
#           leave 18 GiB idle).
#
# levels and per_scale_mixer_widths remain at baseline values across all tests.
# Tuning them to match the longer block_size in Tests 3 and 4 (so the wavelet
# decomposition captures additional coarse scales) is a follow-up worth doing
# if Tests 3/4 show promise — but doing it here would force a per_scale_mixer_widths
# resize that's a separate variable from the test's primary axis.
#
# Test 4 uses an automated NaN watcher that polls each run's log every 30s
# and terminates training (returning code 99) if "NaN" appears, then retries
# at the next-smaller block_size. Stable completion (exit 0) ends the loop.
#
# After all tests complete, the config is reset to the L=2 release default
# (matches README Training section).

set_keys() {
    python -c "
import json
cfg = json.load(open('config.json'))
patch = json.loads('''$1''')
cfg.update(patch)
json.dump(cfg, open('config.json', 'w'), indent=4)
"
}

run_generation_if_ckpt() {
    local LATEST_CKPT
    LATEST_CKPT=$(ls -dt logs/wikitext-103_*/best_model.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        python generate.py --checkpoint "$LATEST_CKPT" || true
        python generate.py --checkpoint "$LATEST_CKPT" --strategies || true
    else
        echo "[runs.sh] No best_model.pt found; skipping generation."
    fi
}

# Commit and push results after each run so a later failure doesn't lose
# earlier completed results. Takes one argument: the commit message.
git_commit_push() {
    local MSG="$1"
    git add . || true
    git commit --no-edit -m "${MSG}" || true
    git pull --no-edit || true
    git push || true
}

run_one() {
    local LABEL="$1"
    local PATCH="$2"

    echo ""
    echo "============================================================"
    echo "=== ${LABEL}"
    echo "============================================================"
    set_keys "${PATCH}"
    python train.py
    local TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "[runs.sh] train.py failed in ${LABEL} with exit code $TRAIN_EXIT."
        echo "[runs.sh] Restoring L=2 release default and aborting remaining tests."
        set_keys '{"dataset": "wikitext-103", "layers": 2, "epochs": 5, "mlp_expansion": 20, "pkm_enabled": true, "fwpkm_num_keys": 16384, "tie_embedding_to_lm_head": false, "micro_batch_size": 8, "grad_accum": 1, "block_size": 256, "levels": 5, "eval_interval": 250}'
        exit $TRAIN_EXIT
    fi
    run_generation_if_ckpt
}

# nan_safe_run: launches train.py in the background, watches its log file
# for NaN every 30s, and terminates training if found.
# Returns 99 on NaN, otherwise train.py's exit code.
nan_safe_run() {
    local LABEL="$1"
    local PATCH="$2"

    echo ""
    echo "============================================================"
    echo "=== ${LABEL}"
    echo "============================================================"
    set_keys "${PATCH}"

    # Snapshot existing log dirs so we can identify the new one created by train.py
    local PRE_LOG_COUNT
    PRE_LOG_COUNT=$(ls -d logs/wikitext-103_*/ 2>/dev/null | wc -l)

    python train.py &
    local TRAIN_PID=$!

    # Wait up to 2 min for train.py to create its run folder
    local LOG_FILE=""
    for i in $(seq 1 60); do
        sleep 2
        local NEW_LOG_COUNT
        NEW_LOG_COUNT=$(ls -d logs/wikitext-103_*/ 2>/dev/null | wc -l)
        if [ "$NEW_LOG_COUNT" -gt "$PRE_LOG_COUNT" ]; then
            LOG_FILE="$(ls -dt logs/wikitext-103_*/ | head -1)log.txt"
            echo "[runs.sh] Monitoring $LOG_FILE for NaN."
            break
        fi
    done

    if [ -z "$LOG_FILE" ]; then
        echo "[runs.sh] No new log folder after 2 min; falling through to wait."
        wait $TRAIN_PID
        return $?
    fi

    # Poll log for NaN every 30s while training runs
    while kill -0 $TRAIN_PID 2>/dev/null; do
        if grep -q "NaN" "$LOG_FILE" 2>/dev/null; then
            echo "[runs.sh] NaN detected in $LOG_FILE; terminating PID $TRAIN_PID."
            kill -TERM $TRAIN_PID 2>/dev/null
            sleep 5
            kill -KILL $TRAIN_PID 2>/dev/null
            wait $TRAIN_PID 2>/dev/null
            sleep 10  # give CUDA a moment to release VRAM
            return 99
        fi
        sleep 30
    done

    wait $TRAIN_PID
    local TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -eq 0 ]; then
        run_generation_if_ckpt
    fi
    return $TRAIN_EXIT
}

# ============================================================
# Test 2b: Max EBS variant (MBS=64, GA=1, bs=256, levels=5) with proportional
# eval_interval. Same as Test 2 (which already completed at BPB sliding 1.0860,
# vs Test 1's 1.0796), but with eval_interval=32 instead of 250 to match
# baseline's eval frequency in evals-per-epoch (~228 vs ~29).
#
# Hypothesis under test: was Test 2's regression due to (a) the gradient-noise-
# as-regularizer effect, (b) eval coarseness causing a sub-optimal best-
# checkpoint to be saved, or (c) just within-noise variation? At MBS=64 with
# eval_interval=250, the run had only ~29 evals/epoch vs baseline's ~234.
# Test 2b removes the eval-coarseness confound while leaving everything else
# the same as Test 2.
#
# Noise context: 3-seed variance study established noise floor at ~±0.0015 BPB
# (3-seed BPBs: 1.0140, 1.0155, 1.0152; sample σ ≈ 0.0008). Test 2's 0.0064
# regression vs Test 1 is ~4× the noise threshold — comfortably significant
# for a single-seed comparison.
#
# Decision rule (with corrected noise band):
#   - Test 2b BPB ≤ ~1.0811 (Test 1 + 0.0015): eval coarseness was likely the
#     dominant factor; gradient-noise hypothesis weakens
#   - Test 2b BPB ≥ ~1.0845 (Test 2 - 0.0015): replication of Test 2's
#     direction with finer eval — gradient-noise hypothesis strengthens
#     to the point of two single-seed confirmations
#   - Test 2b BPB in [1.0811, 1.0845]: eval coarseness explains some but not
#     all of Test 2's regression; gradient-noise effect still real but
#     somewhat smaller than Test 2 alone suggested
# ============================================================
# run_one "Test 2b: Max EBS + proportional eval_interval — MBS=64, GA=1, bs=256, eval_interval=32" \
#     '{"dataset": "wikitext-103", "layers": 1, "epochs": 5, "mlp_expansion": 10, "pkm_enabled": false, "fwpkm_num_keys": 8281, "tie_embedding_to_lm_head": true, "micro_batch_size": 64, "grad_accum": 1, "block_size": 256, "levels": 5, "eval_interval": 32}'
# git_commit_push "Test 2b (combined reduction + Max EBS, MBS=64, eval_interval=32): completed run"

# # ============================================================
# # Test 3: Larger block_size variant (MBS=8, GA=1, bs=1024, levels=5)
# # Uses freed VRAM for longer context. levels and per_scale_mixer_widths
# # remain at baseline values (5 / 6-entry array) — adjusting them to match
# # the longer context is left for a follow-up sweep.
# # ============================================================
# run_one "Test 3: Larger block_size — MBS=8, GA=1, bs=1024, levels=5 (combined reduction recipe)" \
#     '{"dataset": "wikitext-103", "layers": 1, "epochs": 5, "mlp_expansion": 10, "pkm_enabled": false, "fwpkm_num_keys": 8281, "tie_embedding_to_lm_head": true, "micro_batch_size": 8, "grad_accum": 1, "block_size": 1024, "levels": 5, "eval_interval": 250}'
# git_commit_push "Test 3 (combined reduction + larger block_size, bs=1024): completed run"

# ============================================================
# Test 4 (COMPLETED): Min EBS + max block_size variant (MBS=1, GA=1).
# Trained stably at bs=16384, levels=5, val 3.4170, BPB sliding 1.1149,
# inference VRAM 7.78 GiB. See logs/wikitext-103_2026-05-02_09-04-39/.
# Block kept commented for reference; Test 5 (below) replaces it as the
# active test, sweeping `levels` to fix Test 4's per-scale undersizing.
# ============================================================
# TEST_4_DONE=0
# for BLOCK_SIZE in 16384 8192 4096 2048 1024 512 256; do
#     PATCH="{\"dataset\": \"wikitext-103\", \"layers\": 1, \"epochs\": 5, \"mlp_expansion\": 10, \"pkm_enabled\": false, \"fwpkm_num_keys\": 8281, \"tie_embedding_to_lm_head\": true, \"micro_batch_size\": 1, \"grad_accum\": 1, \"block_size\": $BLOCK_SIZE, \"levels\": 5, \"eval_interval\": 250}"
#     nan_safe_run "Test 4: MBS=1, GA=1, bs=$BLOCK_SIZE, levels=5 (combined reduction recipe)" "$PATCH"
#     EXIT_CODE=$?
#     if [ $EXIT_CODE -eq 0 ]; then
#         echo "[runs.sh] Test 4 completed stably at block_size=$BLOCK_SIZE."
#         git_commit_push "Test 4 (combined reduction + min EBS + max block_size): completed at bs=$BLOCK_SIZE"
#         TEST_4_DONE=1
#         break
#     elif [ $EXIT_CODE -eq 99 ]; then
#         echo "[runs.sh] NaN at block_size=$BLOCK_SIZE; halving and retrying."
#     else
#         echo "[runs.sh] Non-NaN failure (exit $EXIT_CODE) at block_size=$BLOCK_SIZE; treating as instability and halving."
#         sleep 10  # give CUDA a moment to release VRAM after OOM/crash
#     fi
# done
# if [ $TEST_4_DONE -eq 0 ]; then
#     echo "[runs.sh] Test 4 failed at all block_sizes including 256. Investigate manually."
# fi

# ============================================================
# Test 5 Phase 2: retry levels=9 and 11 at lr=0.01 (original LR) now that
# the FWHT fp16-overflow has been mitigated by mean-centering before the
# forward FWHT (model.py 2026-05-04).
#
# Each iteration runs 1 epoch at lr=0.01. After both finish, the iteration
# with the lowest sliding-window BPB is auto-launched for a 5-epoch follow-up.
# A NaN or non-zero exit on any iteration disqualifies it from the winner pick.
#
# Levels=13 skipped (OOMs at bs=16384 / MBS=1 without gradient_checkpointing).
# Levels=5 / 7 at lr=0.01 already completed in earlier sweeps — no re-run here;
# their winning configuration is compared against this sweep's winner manually.
# ============================================================
TEST5_BLOCK_SIZE=16384

build_psmw() {
    # Builds a per_scale_mixer_widths JSON array for a given levels value.
    # levels=L → S=L+1 scales → first S/2 entries 1.0, second S/2 entries 0.5.
    python -c "
S = $1 + 1
h = S // 2
print('[' + ', '.join(['1.0']*h + ['0.5']*h) + ']')
"
}

get_stable_lr() {
    # get_stable_lr <levels> → echoes "<peak_lr> <min_lr>"
    # Per-level peak LR is set to half the last finite step's LR observed in
    # the prior K=False sweep — fp16 cliff failures need real margin, and the
    # observed NaN-onset between training-step granularity hides the actual
    # cliff position. min_lr scaled to keep the cosine schedule's 50× peak/min
    # ratio so the late-training optimization shape stays consistent.
    #   levels=5/7  : trained stably at lr=0.01 with K=False → keep
    #   levels=9    : last finite step at lr=6.84e-3 → peak=3.42e-3
    #   levels=11   : last finite step at lr=2.28e-3 → peak=1.14e-3
    case "$1" in
        5|7)   echo "0.01 0.0002" ;;
        9)     echo "0.00342 0.0000684" ;;
        11)    echo "0.00114 0.0000228" ;;
        13)    echo "0.001 0.00002" ;;  # OOMs at this config; LR is placeholder
        *)     echo "0.01 0.0002" ;;
    esac
}

build_test5_patch() {
    # build_test5_patch <levels> <epochs> [lr] [min_lr]
    # When lr/min_lr are omitted, looked up from get_stable_lr() per the
    # heterogeneous-LR sweep design (Position C — see README Per-Scale section
    # for rationale).
    #
    # NOTE: wavelet_crawl is set to false for the levels sweep. With K=3, every
    # level evaluates the lifting MLP cascade three times and softmax-mixes
    # them, multiplying activation compounding by 3× per level. At levels=7
    # this hit fp16 instability during warmup (see iter 2,
    # logs/wikitext-103_2026-05-02_19-02-53/log.txt — NaN'd at step 1750,
    # lr~8e-3). Disabling wavelet_crawl for the sweep keeps higher-levels
    # configurations stable and is doubly justified by the §11 finding that
    # at least 2 of 5 levels at L=5 already collapse to wasted K=3 distributions.
    local LEVELS=$1
    local EPOCHS=$2
    local LR=${3:-}
    local MIN_LR=${4:-}
    if [ -z "$LR" ]; then
        read LR MIN_LR <<< "$(get_stable_lr $LEVELS)"
    fi
    local PSMW
    PSMW=$(build_psmw $LEVELS)
    python -c "
import json
print(json.dumps({
    'dataset': 'wikitext-103',
    'layers': 1,
    'epochs': $EPOCHS,
    'mlp_expansion': 10,
    'pkm_enabled': False,
    'fwpkm_num_keys': 8281,
    'tie_embedding_to_lm_head': True,
    'micro_batch_size': 1,
    'grad_accum': 1,
    'block_size': $TEST5_BLOCK_SIZE,
    'levels': $LEVELS,
    'per_scale_mixer_widths': $PSMW,
    'wavelet_crawl': False,
    'lr': $LR,
    'min_lr': $MIN_LR,
    'eval_interval': 250,
}))
"
}

# Read sliding-window BPB from a run directory. Prefers benchmark.txt
# (post-fix corrected output, written by benchmark_only mode) and falls back
# to log.txt. Returns empty if neither file or the BPB line is missing.
extract_bpb_sliding() {
    local DIR="$1"
    local F=""
    if [ -f "${DIR}/benchmark.txt" ]; then
        F="${DIR}/benchmark.txt"
    elif [ -f "${DIR}/log.txt" ]; then
        F="${DIR}/log.txt"
    fi
    [ -n "$F" ] || return 1
    # First "BPB:" line that follows the "[BENCHMARK - Sliding Window]" header.
    awk '/\[BENCHMARK - Sliding Window\]/{flag=1; next} flag && /BPB:/{print $2; exit}' "$F"
}

# Run one Phase-2 tier: builds the levels-N base patch, optionally overlays
# a tier-specific override (e.g. {"mlp_expansion": 9} or {"compile": false}),
# launches train.py, and judges success by whether the run produced a
# parseable sliding-window BPB. Captures the new log dir into PHASE2_LOGDIR
# and the resulting BPB into PHASE2_BPB. Returns 0 on success (BPB present),
# non-zero otherwise.
# Launch `python train.py` in the background, identify the new log dir it
# creates, and poll log.txt every 30s for a NaN train- or val-loss entry
# (matches lines like "Step 750: train loss nan, val loss nan ..."). On NaN,
# terminates the train PID and returns 99. Otherwise returns train.py's exit
# code. Sets RUN_LOGDIR to the captured log dir (empty if none was created).
RUN_LOGDIR=""
run_train_with_nan_watch() {
    RUN_LOGDIR=""
    local PRE_LATEST
    PRE_LATEST=$(ls -dt logs/wikitext-103_*/ 2>/dev/null | head -1)

    python train.py &
    local TRAIN_PID=$!

    # Wait up to 2 min for train.py to create its run folder
    local LOG_FILE=""
    for i in $(seq 1 60); do
        sleep 2
        local POST_LATEST
        POST_LATEST=$(ls -dt logs/wikitext-103_*/ 2>/dev/null | head -1)
        if [ -n "$POST_LATEST" ] && [ "$POST_LATEST" != "$PRE_LATEST" ]; then
            RUN_LOGDIR="$POST_LATEST"
            LOG_FILE="${POST_LATEST}log.txt"
            echo "[runs.sh] Monitoring $LOG_FILE for NaN."
            break
        fi
    done

    if [ -z "$LOG_FILE" ]; then
        echo "[runs.sh] No new log folder after 2 min; falling through to wait."
        wait $TRAIN_PID
        return $?
    fi

    # Poll log for "loss nan" (catches both train and val loss) every 30s.
    # `loss nan` is specific enough to avoid false positives on unrelated tokens.
    while kill -0 $TRAIN_PID 2>/dev/null; do
        if grep -qi "loss nan" "$LOG_FILE" 2>/dev/null; then
            echo "[runs.sh] NaN loss detected in $LOG_FILE; terminating PID $TRAIN_PID."
            kill -TERM $TRAIN_PID 2>/dev/null
            sleep 5
            kill -KILL $TRAIN_PID 2>/dev/null
            wait $TRAIN_PID 2>/dev/null
            sleep 10  # give CUDA a moment to release VRAM
            return 99
        fi
        sleep 30
    done

    wait $TRAIN_PID
    return $?
}

# Run one Phase-2 tier: builds the levels-N base patch, optionally overlays
# a tier-specific override (e.g. {"mlp_expansion": 9} or {"compile": false}),
# launches train.py under NaN-watch early stopping, and judges success by
# whether the run produced a parseable sliding-window BPB. Captures the new
# log dir into PHASE2_LOGDIR and the resulting BPB into PHASE2_BPB. Returns
# 0 on success (BPB present), non-zero otherwise (NaN-killed or other crash).
PHASE2_LOGDIR=""
PHASE2_BPB=""
run_test5_phase2_tier() {
    local LEVELS=$1
    local TIER_LABEL="$2"
    local OVERRIDE_JSON="$3"
    local SCALES=$((LEVELS + 1))
    local LABEL="Test 5 Phase 2: levels=$LEVELS (S=$SCALES) @ lr=0.01, bs=$TEST5_BLOCK_SIZE, tier=[$TIER_LABEL]"

    echo ""
    echo "============================================================"
    echo "=== ${LABEL}"
    echo "============================================================"

    set_keys "$(build_test5_patch $LEVELS 1 0.01 0.0002)"
    if [ -n "$OVERRIDE_JSON" ]; then
        set_keys "$OVERRIDE_JSON"
    fi
    run_train_with_nan_watch
    local EXIT=$?
    PHASE2_LOGDIR="$RUN_LOGDIR"

    PHASE2_BPB=""
    if [ -n "$PHASE2_LOGDIR" ]; then
        PHASE2_BPB=$(extract_bpb_sliding "$PHASE2_LOGDIR")
    fi

    if [ -n "$PHASE2_BPB" ]; then
        git_commit_push "Test 5 Phase 2: L=$LEVELS [$TIER_LABEL] completed (BPB sliding=$PHASE2_BPB)"
        return 0
    else
        if [ "$EXIT" -eq 99 ]; then
            echo "[runs.sh] L=$LEVELS [$TIER_LABEL] killed by NaN-watch."
            git_commit_push "Test 5 Phase 2: L=$LEVELS [$TIER_LABEL] FAILED (NaN early-stop)"
        else
            echo "[runs.sh] L=$LEVELS [$TIER_LABEL] failed (exit $EXIT, no BPB)."
            git_commit_push "Test 5 Phase 2: L=$LEVELS [$TIER_LABEL] FAILED exit $EXIT (no BPB)"
        fi
        return 1
    fi
}

# L=11 progression. Stop at the first tier that produces a parseable BPB.
# Each tier loosens the failure mode the previous tier could not cure:
#   stock     → mlp_expansion=10, compile=True (matches production default)
#   mlp9      → mlp_expansion=9 (test whether the lowered MLP expansion in the
#               diagnostic was the real fix vs. mean centering)
#   no-compile → mlp_expansion=9 + compile=False (rule out a torch.compile
#               numerical interaction)
# If all three fail (NaN at step ~431 even with no-compile), the inconsistency
# between diagnostic and training is deeper than the tested axes — investigate
# manually before further sweeps.
PHASE2_WIN_LEVEL=""
PHASE2_WIN_BPB=""
PHASE2_WIN_OVERRIDE=""
PHASE2_WIN_TIER=""
for TIER in "stock|" "mlp9|{\"mlp_expansion\": 9}" "no-compile|{\"mlp_expansion\": 9, \"compile\": false}"; do
    TIER_LABEL="${TIER%%|*}"
    OVERRIDE="${TIER#*|}"
    if run_test5_phase2_tier 11 "$TIER_LABEL" "$OVERRIDE"; then
        PHASE2_WIN_LEVEL=11
        PHASE2_WIN_BPB="$PHASE2_BPB"
        PHASE2_WIN_OVERRIDE="$OVERRIDE"
        PHASE2_WIN_TIER="$TIER_LABEL"
        break
    fi
done

# 5-epoch follow-up: at the L=11 winning tier if any survived, otherwise at
# the prior best stable level (L=7, BPB sliding 1.0974 from the earlier sweep).
echo ""
echo "============================================================"
echo "=== Test 5 Phase 2 follow-up pick"
echo "============================================================"
if [ -n "$PHASE2_WIN_LEVEL" ]; then
    FOLLOWUP_LEVEL=$PHASE2_WIN_LEVEL
    FOLLOWUP_OVERRIDE="$PHASE2_WIN_OVERRIDE"
    FOLLOWUP_NOTE="L=$PHASE2_WIN_LEVEL [$PHASE2_WIN_TIER] (1-epoch BPB sliding=$PHASE2_WIN_BPB)"
else
    FOLLOWUP_LEVEL=7
    FOLLOWUP_OVERRIDE=""
    FOLLOWUP_NOTE="fallback to L=7 (all L=11 tiers failed)"
fi
echo "  → $FOLLOWUP_NOTE"

echo ""
echo "============================================================"
echo "=== Test 5 Phase 2 follow-up: 5-epoch run at L=$FOLLOWUP_LEVEL"
echo "============================================================"
set_keys "$(build_test5_patch $FOLLOWUP_LEVEL 5 0.01 0.0002)"
[ -n "$FOLLOWUP_OVERRIDE" ] && set_keys "$FOLLOWUP_OVERRIDE"
run_train_with_nan_watch
FOLLOWUP_EXIT=$?
if [ $FOLLOWUP_EXIT -eq 0 ]; then
    run_generation_if_ckpt
    git_commit_push "Test 5 Phase 2 follow-up: 5-epoch L=$FOLLOWUP_LEVEL @ lr=0.01 completed ($FOLLOWUP_NOTE)"
elif [ $FOLLOWUP_EXIT -eq 99 ]; then
    echo "[runs.sh] 5-epoch follow-up at L=$FOLLOWUP_LEVEL killed by NaN-watch."
    git_commit_push "Test 5 Phase 2 follow-up: 5-epoch L=$FOLLOWUP_LEVEL FAILED (NaN early-stop)"
else
    echo "[runs.sh] 5-epoch follow-up at L=$FOLLOWUP_LEVEL failed exit $FOLLOWUP_EXIT."
    git_commit_push "Test 5 Phase 2 follow-up: 5-epoch L=$FOLLOWUP_LEVEL FAILED exit $FOLLOWUP_EXIT"
fi

# ============================================================
# Reset config to L=2 release default (matches README Training section)
# ============================================================
set_keys '{"dataset": "wikitext-103", "layers": 2, "epochs": 5, "mlp_expansion": 20, "pkm_enabled": true, "fwpkm_num_keys": 16384, "tie_embedding_to_lm_head": false, "micro_batch_size": 8, "grad_accum": 1, "block_size": 256, "levels": 5, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "eval_interval": 250}'

git_commit_push "Reset config.json to L=2 release default after Test 5 Phase 2"

echo ""
echo "============================================================"
echo "=== Test 5 Phase 2 complete."
echo "===   Pull BPB sliding-window numbers and update:"
echo "===     - plans/other_post_release_plans.md §8 (results section)"
echo "===     - plans/findings.md (Combined Parameter Reduction entry)"
echo "===     - runs.md (results table)"
echo "===     - README.md (Combined Parameter Reduction + Per-Scale Config subsections)"
echo "============================================================"
