#!/bin/bash
# Tests 2-4: Combined parameter reduction + VRAM reallocation variants.
# All tests apply the four reductions (mlp_expansion=10, pkm_enabled=false,
# fwpkm_num_keys=8281, tie_embedding_to_lm_head=true) at L=1 / E=5, varying
# how the freed VRAM is used.
#
#   Test 2: Max EBS — MBS=64, GA=1, bs=256, levels=5 (uses ~25 GiB / 32 GiB)
#   Test 3: Larger block_size — MBS=8, GA=1, bs=1024, levels=5
#   Test 4: Min EBS + max block_size — MBS=1, GA=1, bs starts at 8192 and
#           halves on NaN detection (8192→4096→2048→1024→512→256) until a
#           stable run completes; levels=5 throughout
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
# Noise context: 3-seed variance study established noise floor at ~±0.004 BPB.
# Test 2's 0.0064 regression vs Test 1 is only ~1.6σ — directionally suggestive
# but not conclusive against noise.
#
# Decision rule (with noise band acknowledged):
#   - Test 2b BPB ≤ ~1.082: eval coarseness was likely the dominant factor;
#     gradient-noise hypothesis weakens
#   - Test 2b BPB ≥ ~1.084: gradient-noise hypothesis strengthens (independent
#     replication of Test 2's direction with finer eval)
#   - Test 2b BPB in [1.082, 1.084]: ambiguous; both effects probably present
#     OR all single-run gaps so far are within noise. Would need 3-seed runs at
#     each MBS to resolve cleanly.
# ============================================================
run_one "Test 2b: Max EBS + proportional eval_interval — MBS=64, GA=1, bs=256, eval_interval=32" \
    '{"dataset": "wikitext-103", "layers": 1, "epochs": 5, "mlp_expansion": 10, "pkm_enabled": false, "fwpkm_num_keys": 8281, "tie_embedding_to_lm_head": true, "micro_batch_size": 64, "grad_accum": 1, "block_size": 256, "levels": 5, "eval_interval": 32}'
git_commit_push "Test 2b (combined reduction + Max EBS, MBS=64, eval_interval=32): completed run"

# ============================================================
# Test 3: Larger block_size variant (MBS=8, GA=1, bs=1024, levels=5)
# Uses freed VRAM for longer context. levels and per_scale_mixer_widths
# remain at baseline values (5 / 6-entry array) — adjusting them to match
# the longer context is left for a follow-up sweep.
# ============================================================
run_one "Test 3: Larger block_size — MBS=8, GA=1, bs=1024, levels=5 (combined reduction recipe)" \
    '{"dataset": "wikitext-103", "layers": 1, "epochs": 5, "mlp_expansion": 10, "pkm_enabled": false, "fwpkm_num_keys": 8281, "tie_embedding_to_lm_head": true, "micro_batch_size": 8, "grad_accum": 1, "block_size": 1024, "levels": 5, "eval_interval": 250}'
git_commit_push "Test 3 (combined reduction + larger block_size, bs=1024): completed run"

# ============================================================
# Test 4: Min EBS + max block_size variant (MBS=1, GA=1)
# Iterates block_size from 8192 down by halving on NaN detection until stable.
# Hypothesizes the best-of-both-worlds combination for a regularization-bound
# model: maximum gradient noise from single-sequence updates plus rich
# per-example signal from very long context for the wavelet pipeline.
# ============================================================
TEST_4_DONE=0
for BLOCK_SIZE in 8192 4096 2048 1024 512 256; do
    PATCH="{\"dataset\": \"wikitext-103\", \"layers\": 1, \"epochs\": 5, \"mlp_expansion\": 10, \"pkm_enabled\": false, \"fwpkm_num_keys\": 8281, \"tie_embedding_to_lm_head\": true, \"micro_batch_size\": 1, \"grad_accum\": 1, \"block_size\": $BLOCK_SIZE, \"levels\": 8, \"eval_interval\": 250}"
    nan_safe_run "Test 4: MBS=1, GA=1, bs=$BLOCK_SIZE, levels=5 (combined reduction recipe)" "$PATCH"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[runs.sh] Test 4 completed stably at block_size=$BLOCK_SIZE."
        git_commit_push "Test 4 (combined reduction + min EBS + max block_size): completed at bs=$BLOCK_SIZE"
        TEST_4_DONE=1
        break
    elif [ $EXIT_CODE -eq 99 ]; then
        echo "[runs.sh] NaN at block_size=$BLOCK_SIZE; halving and retrying."
    else
        echo "[runs.sh] Non-NaN failure (exit $EXIT_CODE) at block_size=$BLOCK_SIZE; treating as instability and halving."
        sleep 10  # give CUDA a moment to release VRAM after OOM/crash
    fi
done
if [ $TEST_4_DONE -eq 0 ]; then
    echo "[runs.sh] Test 4 failed at all block_sizes including 256. Investigate manually."
fi

# ============================================================
# Reset config to L=2 release default (matches README Training section)
# ============================================================
set_keys '{"dataset": "wikitext-103", "layers": 2, "epochs": 5, "mlp_expansion": 20, "pkm_enabled": true, "fwpkm_num_keys": 16384, "tie_embedding_to_lm_head": false, "micro_batch_size": 8, "grad_accum": 1, "block_size": 256, "levels": 5, "eval_interval": 250}'

git_commit_push "Reset config.json to L=2 release default after Tests 2-4 variants completed"

echo ""
echo "============================================================"
echo "=== Tests 2-4 (combined reduction variants) complete."
echo "===   Pull BPB sliding-window numbers and update:"
echo "===     - plans/other_post_release_plans.md §8 (results section)"
echo "===     - plans/findings.md (Combined Parameter Reduction entry)"
echo "===     - runs.md (results table)"
echo "===     - README.md (Combined Parameter Reduction subsection)"
echo "============================================================"
