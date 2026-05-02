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
# Test 5: Per-scale configuration sweep at bs=16384.
# Hypothesis: optimal levels ≈ log2(block_size) − 3. At bs=256 with
# levels=5, log2(256) − 5 = 3 was optimal. At bs=16384 (log2=14),
# levels=11 (14 − 3) is the prediction. Sweep levels = 5, 7, 9, 11, 13;
# matching per_scale_mixer_widths = (levels+1)/2 entries of 1.0 then
# (levels+1)/2 entries of 0.5 (symmetric half-coarse / half-fine split).
#
# Each sweep run is 1 epoch (~1.15h on a 5090) → ~6h total for the sweep.
# Auto-picks the lowest-BPB-sliding level and runs 5 epochs at that level
# (~5.76h). Total wall-clock ~12h.
#
# All five 1-epoch sweep runs share block_size=16384, so their BPB sliding
# numbers ARE directly comparable (same window count, same stride, same
# eval set). The 5-epoch winner run also uses bs=16384, so its BPB sliding
# is comparable to Test 4's 1.1149 baseline at bs=16384 with levels=5.
# ============================================================
TEST5_BLOCK_SIZE=16384
TEST5_RESULTS_FILE="logs/test5_levels_sweep_results.txt"
> "$TEST5_RESULTS_FILE"

build_psmw() {
    # Builds a per_scale_mixer_widths JSON array for a given levels value.
    # levels=L → S=L+1 scales → first S/2 entries 1.0, second S/2 entries 0.5.
    python -c "
S = $1 + 1
h = S // 2
print('[' + ', '.join(['1.0']*h + ['0.5']*h) + ']')
"
}

build_test5_patch() {
    # build_test5_patch <levels> <epochs>
    # NOTE: wavelet_crawl is set to false for the levels sweep. With K=3, every
    # level evaluates the lifting MLP cascade three times and softmax-mixes
    # them, multiplying activation compounding by 3× per level. At levels=7
    # this hit fp16 instability during warmup (see Test 5 sweep iter 2,
    # logs/wikitext-103_2026-05-02_19-02-53/log.txt — NaN'd at step 1750,
    # lr~8e-3). Disabling wavelet_crawl for the sweep keeps higher-levels
    # configurations stable and is doubly justified by the §11 finding that
    # at least 2 of 5 levels at L=5 already collapse to wasted K=3 distributions.
    local LEVELS=$1
    local EPOCHS=$2
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
    'eval_interval': 250,
}))
"
}

run_test5_sweep_one() {
    # Runs train.py with the patched config at the given levels for 1 epoch.
    # Captures sliding-window BPB from the resulting log and appends to
    # TEST5_RESULTS_FILE. Does NOT abort the script on failure (so OOM at
    # high levels just records N/A and the sweep continues).
    local LEVELS=$1
    local SCALES=$((LEVELS + 1))
    local LABEL="Test 5 sweep (1-ep): levels=$LEVELS (S=$SCALES), bs=$TEST5_BLOCK_SIZE"

    echo ""
    echo "============================================================"
    echo "=== ${LABEL}"
    echo "============================================================"

    local PATCH
    PATCH=$(build_test5_patch $LEVELS 1)
    set_keys "$PATCH"
    python train.py
    local TRAIN_EXIT=$?

    local LATEST_LOG=""
    LATEST_LOG=$(ls -dt logs/wikitext-103_*/log.txt 2>/dev/null | head -1)

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "[runs.sh] levels=$LEVELS sweep run failed (exit $TRAIN_EXIT). Recording N/A."
        printf "%s\t%s\t%s\texit=%s\n" "$LEVELS" "N/A" "$LATEST_LOG" "$TRAIN_EXIT" >> "$TEST5_RESULTS_FILE"
        sleep 10  # give CUDA a moment to release VRAM after OOM/crash
        git_commit_push "Test 5 sweep: levels=$LEVELS @ bs=$TEST5_BLOCK_SIZE → FAILED (exit $TRAIN_EXIT)"
        return
    fi

    # Parse sliding-window BPB from the log
    local BPB
    BPB=$(python -c "
import re
log = open('$LATEST_LOG', encoding='utf-8', errors='replace').read()
m = re.search(r'\[BENCHMARK - Sliding Window\].*?BPB:\s*([\d.]+)', log, re.DOTALL)
print(m.group(1) if m else 'N/A')
")
    echo "[runs.sh] levels=$LEVELS sliding BPB = $BPB ($LATEST_LOG)"
    printf "%s\t%s\t%s\tok\n" "$LEVELS" "$BPB" "$LATEST_LOG" >> "$TEST5_RESULTS_FILE"
    git_commit_push "Test 5 sweep: levels=$LEVELS @ bs=$TEST5_BLOCK_SIZE → BPB sliding $BPB"
}

for L in 5 7 9 11 13; do
    run_test5_sweep_one $L
done

echo ""
echo "============================================================"
echo "=== Test 5 sweep results ($TEST5_RESULTS_FILE):"
echo "============================================================"
cat "$TEST5_RESULTS_FILE"

# Auto-pick winner (lowest BPB among successful runs) and run 5-epoch follow-up
WINNER_LEVELS=$(python -c "
results = []
for line in open('$TEST5_RESULTS_FILE'):
    parts = line.rstrip('\n').split('\t')
    if len(parts) >= 4 and parts[1] != 'N/A' and parts[3] == 'ok':
        try:
            results.append((int(parts[0]), float(parts[1])))
        except ValueError:
            pass
print(min(results, key=lambda x: x[1])[0] if results else 'NONE')
")

if [ "$WINNER_LEVELS" = "NONE" ]; then
    echo "[runs.sh] Test 5 sweep: no successful runs. Skipping 5-epoch follow-up."
else
    echo ""
    echo "============================================================"
    echo "=== Test 5 winner: levels=$WINNER_LEVELS — launching 5-epoch follow-up"
    echo "============================================================"
    WINNER_PATCH=$(build_test5_patch $WINNER_LEVELS 5)
    run_one "Test 5 final (5-ep): levels=$WINNER_LEVELS @ bs=$TEST5_BLOCK_SIZE (sweep winner)" "$WINNER_PATCH"
    git_commit_push "Test 5 final: levels=$WINNER_LEVELS @ bs=$TEST5_BLOCK_SIZE — 5-epoch run at sweep winner"
fi

# ============================================================
# Reset config to L=2 release default (matches README Training section)
# ============================================================
set_keys '{"dataset": "wikitext-103", "layers": 2, "epochs": 5, "mlp_expansion": 20, "pkm_enabled": true, "fwpkm_num_keys": 16384, "tie_embedding_to_lm_head": false, "micro_batch_size": 8, "grad_accum": 1, "block_size": 256, "levels": 5, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "eval_interval": 250}'

git_commit_push "Reset config.json to L=2 release default after Tests 2-5 variants completed"

echo ""
echo "============================================================"
echo "=== Tests 2-5 (combined reduction + per-scale configuration) complete."
echo "===   Pull BPB sliding-window numbers and update:"
echo "===     - plans/other_post_release_plans.md §8 (results section)"
echo "===     - plans/findings.md (Combined Parameter Reduction entry)"
echo "===     - runs.md (results table)"
echo "===     - README.md (Combined Parameter Reduction + Per-Scale Config subsections)"
echo "============================================================"
