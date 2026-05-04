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
# PHASE 2: retry levels=9 and 11 at lr=0.01 (the original LR) now that the
# fp16-cumsum overflow in _compute_running_mean is fixed (model.py 2026-05-03).
# The earlier K=3-attributed NaNs at deeper cascades were partly the cumsum
# overflow we just fixed; this phase tests whether levels=9/11 are now
# trainable at the original LR, eliminating the LR-confound from Phase 1's
# heterogeneous-LR design.
#
# Sweep iterations to RUN this phase: {9, 11} at lr=0.01 with NaN detection
# (~2.3h total if both succeed; less if either NaNs early).
# Pre-populated (no re-run): levels=5 / 7 at lr=0.01 from Phase 1's results
# (their BPB is read fresh from benchmark.txt — corrected post-rebench —
# falling back to log.txt if benchmark.txt isn't present).
# Levels=13 skipped (OOMs at this config without gradient_checkpointing).
#
# After the sweep, auto-picks the lowest-BPB level across the 4 candidates
# (5/7/9/11) and runs 5 epochs at lr=0.01 with that level + matching
# per_scale_mixer_widths. If levels=7 wins, reuses the existing 5-epoch run
# at logs/wikitext-103_2026-05-03_02-13-07 instead of re-training.
#
# Total wall-clock: ~2.3h sweep + 0-8.7h follow-up = 2.3-11h depending on
# whether 9/11 NaN at lr=0.01 and which level wins.
#
# All sweep runs share block_size=16384 + lr=0.01 + crawl=False, so their
# BPB sliding numbers ARE directly comparable (same window count, same
# stride, same eval set, same LR — fixed-LR apples-to-apples).
# ============================================================
TEST5_BLOCK_SIZE=16384
# Per-iteration results accumulate in a multi-line bash variable instead
# of a logs/ file (in-memory only — no commit pollution, no merge conflicts).
# Format per line, tab-separated: levels  bpb  log_path  phase  status
TEST5_RESULTS_LINES=""

build_psmw() {
    # Builds a per_scale_mixer_widths JSON array for a given levels value.
    # levels=L → S=L+1 scales → first S/2 entries 1.0, second S/2 entries 0.5.
    python -c "
S = $1 + 1
h = S // 2
print('[' + ', '.join(['1.0']*h + ['0.5']*h) + ']')
"
}

build_psmw_uniform() {
    # All-1.0 per_scale_mixer_widths for a given levels value. Wider fine-detail
    # scales test O'Connor's "1 & 2 require width to be increased by 4, 8, or
    # 16 to allow proper information flow" guidance — narrow mixers may
    # bottleneck the FWHT path's spectral signal at high levels.
    python -c "
S = $1 + 1
print('[' + ', '.join(['1.0']*S) + ']')
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
    # Uniform widths (all 1.0): testing O'Connor's "wider needed for proper
    # information flow" hypothesis at high levels. Switch to build_psmw $LEVELS
    # for the prior 1.0/0.5 split if uniform doesn't help.
    PSMW=$(build_psmw_uniform $LEVELS)
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
    'fht_input_cap_enabled': False,
    'fht_input_cap_value': 10000.0,
    'fht_thue_morse_signflips': False,
    'fht_thue_morse_increment': 21,
    'lr': $LR,
    'min_lr': $MIN_LR,
    'eval_interval': 250,
}))
"
}

# Read sliding-window BPB from a run directory. Prefers benchmark.txt
# (post-fix corrected output, written by benchmark_only mode) and falls back
# to log.txt (original training-end benchmark output, may contain inf for
# pre-fix runs). Returns the LAST match in either file so rebenched results
# take precedence over original inf entries that remain above them.
parse_sliding_bpb_from_dir() {
    local LOG_DIR="$1"
    # Strip any trailing slash
    LOG_DIR="${LOG_DIR%/}"
    local FILE="$LOG_DIR/log.txt"
    if [ -f "$LOG_DIR/benchmark.txt" ]; then
        FILE="$LOG_DIR/benchmark.txt"
    fi
    python -c "
import re
log = open('$FILE', encoding='utf-8', errors='replace').read()
m = list(re.finditer(r'\[BENCHMARK - Sliding Window\].*?BPB:\s*([\d.]+|inf)', log, re.DOTALL))
print(m[-1].group(1) if m else 'N/A')
"
}

# Run a single Phase-2 levels iteration at lr=0.01 with NaN detection. On
# NaN, the run is killed mid-flight and recorded as N/A. On any non-zero
# exit, recorded as failure. On success, parses sliding BPB from the
# resulting log dir and appends an "ok" row. Commits + pushes the result
# (success OR failure) after each run.
run_test5_phase2_iter() {
    local LEVELS=$1
    local SCALES=$((LEVELS + 1))
    local LABEL="Test 5 Phase 2: levels=$LEVELS (S=$SCALES) @ lr=0.01, bs=$TEST5_BLOCK_SIZE, crawl=False (cumsum fp32-fix in place)"
    # Force lr=0.01 / min_lr=0.0002 for the Phase 2 retry (overrides
    # get_stable_lr's heterogeneous-LR values from Phase 1).
    local PATCH
    PATCH=$(build_test5_patch $LEVELS 1 0.01 0.0002)

    nan_safe_run "$LABEL" "$PATCH"
    local EXIT=$?

    local LATEST_LOG_DIR
    LATEST_LOG_DIR=$(ls -dt logs/wikitext-103_*/ 2>/dev/null | head -1 | sed 's:/$::')

    if [ $EXIT -eq 99 ]; then
        echo "[runs.sh] levels=$LEVELS NaN'd at lr=0.01 ($LATEST_LOG_DIR). Recording N/A."
        TEST5_RESULTS_LINES+=$(printf "%s\t%s\t%s/log.txt\tphase2\tnan\n" "$LEVELS" "N/A" "$LATEST_LOG_DIR")$'\n'
        git_commit_push "Test 5 Phase 2: levels=$LEVELS @ lr=0.01 NaN'd"
    elif [ $EXIT -ne 0 ]; then
        echo "[runs.sh] levels=$LEVELS failed exit $EXIT ($LATEST_LOG_DIR). Recording N/A."
        TEST5_RESULTS_LINES+=$(printf "%s\t%s\t%s/log.txt\tphase2\texit=%s\n" "$LEVELS" "N/A" "$LATEST_LOG_DIR" "$EXIT")$'\n'
        sleep 10
        git_commit_push "Test 5 Phase 2: levels=$LEVELS @ lr=0.01 FAILED exit $EXIT"
    else
        local BPB
        BPB=$(parse_sliding_bpb_from_dir "$LATEST_LOG_DIR")
        echo "[runs.sh] levels=$LEVELS @ lr=0.01 sliding BPB = $BPB ($LATEST_LOG_DIR)"
        TEST5_RESULTS_LINES+=$(printf "%s\t%s\t%s/log.txt\tphase2\tok\n" "$LEVELS" "$BPB" "$LATEST_LOG_DIR")$'\n'
        git_commit_push "Test 5 Phase 2: levels=$LEVELS @ lr=0.01 → BPB sliding $BPB"
    fi
}

# ============================================================
# Test 5 Phase 2: retry levels=9 and 11 at lr=0.01 with the cumsum fp32 fix
# in place. The earlier K=3-attributed NaNs at deeper levels may have been
# (partly) the fp16-overflow class we just fixed in _compute_running_mean.
# This phase tests whether the original lr=0.01 is now trainable at deeper
# cascades, which would give us a clean apples-to-apples per-level
# comparison without LR-confound.
# ============================================================

# Refresh pre-populated rows by reading current BPB from benchmark.txt (post
# rebench.sh) or log.txt for the existing levels=5/7 lr=0.01 runs. Reading
# at script start ensures we use the latest corrected numbers, not stale
# hardcoded values.
PREPOP_TARGETS=(
    "5 logs/wikitext-103_2026-05-02_20-32-04"
    "7 logs/wikitext-103_2026-05-02_21-43-22"
)
for entry in "${PREPOP_TARGETS[@]}"; do
    L=$(echo $entry | awk '{print $1}')
    DIR=$(echo $entry | awk '{print $2}')
    BPB=$(parse_sliding_bpb_from_dir "$DIR")
    TEST5_RESULTS_LINES+=$(printf "%s\t%s\t%s/log.txt\tphase2\tok\n" "$L" "$BPB" "$DIR")$'\n'
    echo "[runs.sh] Pre-pop levels=$L → sliding BPB $BPB (from $DIR)"
done

# Run the Phase 2 retries at lr=0.01. Order: 11 first (NaN'd fastest in
# prior sweeps so it's the quickest signal on whether the RMS-rescale fix
# holds), then 9 for completeness so both levels have apples-to-apples
# results at the new fix. build_psmw constructs per_scale_mixer_widths
# matching each levels value automatically.
for L in 11 9; do
    run_test5_phase2_iter $L
done

echo ""
echo "============================================================"
echo "=== Test 5 sweep results (in-memory):"
echo "============================================================"
printf '%s' "$TEST5_RESULTS_LINES"

# Auto-pick winner (lowest BPB among ok rows). All Phase 2 rows use
# lr=0.01 so the comparison is apples-to-apples.
WINNER_LEVELS=$(printf '%s' "$TEST5_RESULTS_LINES" | python -c "
import sys
results = []
for line in sys.stdin:
    parts = line.rstrip('\n').split('\t')
    # Format per row: levels, bpb, log, kind, status
    if len(parts) >= 5 and parts[1] not in ('N/A', 'inf') and parts[4] == 'ok':
        try:
            results.append((int(parts[0]), float(parts[1])))
        except ValueError:
            pass
print(min(results, key=lambda x: x[1])[0] if results else 'NONE')
")

# Helper: find the most recent log matching levels=$1, epochs=$2.
# If $3 == "complete" require benchmark completion; if "inprogress" require absence.
find_l_e_log() {
    local target_L=$1 target_E=$2 status=$3
    local f L E
    for f in $(ls -dt logs/wikitext-103_*/log.txt 2>/dev/null | head -30); do
        L=$(grep -m1 -E '^\[2026.*\][[:space:]]+levels[[:space:]]*:' "$f" 2>/dev/null | awk '{print $NF}')
        E=$(grep -m1 -E '^\[2026.*\][[:space:]]+epochs[[:space:]]*:' "$f" 2>/dev/null | awk '{print $NF}')
        [ "$L" = "$target_L" ] || continue
        [ "$E" = "$target_E" ] || continue
        if [ "$status" = "complete" ]; then
            grep -q '=== TRAINING COMPLETE ===' "$f" 2>/dev/null && { echo "$f"; return; }
        elif [ "$status" = "inprogress" ]; then
            grep -q '=== TRAINING COMPLETE ===' "$f" 2>/dev/null || { echo "$f"; return; }
        fi
    done
}

if [ "$WINNER_LEVELS" = "NONE" ]; then
    echo "[runs.sh] Test 5 sweep: no successful runs. Skipping 5-epoch follow-up."
elif [ "$WINNER_LEVELS" = "7" ]; then
    # levels=7 in this sweep uses lr=0.01, matching the 5-epoch levels=7 run
    # that was already auto-launched by the prior sweep iteration. Reuse it
    # if available rather than burning ~5.7h on a redundant run.
    EXISTING_LOG=$(find_l_e_log 7 5 complete)
    INPROGRESS_LOG=$(find_l_e_log 7 5 inprogress)
    if [ -n "$EXISTING_LOG" ]; then
        echo ""
        echo "============================================================"
        echo "=== Test 5 winner: levels=7 — REUSING existing complete 5-epoch run"
        echo "===   $EXISTING_LOG"
        echo "============================================================"
        echo "[runs.sh] Same lr=0.01 / wavelet_crawl=False config; no need to re-train."
    elif [ -n "$INPROGRESS_LOG" ]; then
        echo ""
        echo "============================================================"
        echo "=== Test 5 winner: levels=7 — 5-epoch run is IN PROGRESS at:"
        echo "===   $INPROGRESS_LOG"
        echo "============================================================"
        echo "[runs.sh] Skipping fresh launch to avoid GPU contention. The existing"
        echo "         in-progress run will be the Test 5 final result once it finishes."
    else
        echo ""
        echo "============================================================"
        echo "=== Test 5 winner: levels=7 — no prior 5-epoch run found, launching fresh"
        echo "============================================================"
        WINNER_PATCH=$(build_test5_patch $WINNER_LEVELS 5 0.01 0.0002)
        nan_safe_run "Test 5 Phase 2 final (5-ep): levels=$WINNER_LEVELS @ lr=0.01" "$WINNER_PATCH"
        FINAL_EXIT=$?
        FINAL_LOG_DIR=$(ls -dt logs/wikitext-103_*/ 2>/dev/null | head -1 | sed 's:/$::')
        if [ $FINAL_EXIT -eq 99 ]; then
            git_commit_push "Test 5 Phase 2 final: levels=$WINNER_LEVELS @ lr=0.01 5-ep NaN'd"
        elif [ $FINAL_EXIT -ne 0 ]; then
            git_commit_push "Test 5 Phase 2 final: levels=$WINNER_LEVELS FAILED exit $FINAL_EXIT"
        else
            FINAL_BPB=$(parse_sliding_bpb_from_dir "$FINAL_LOG_DIR")
            git_commit_push "Test 5 Phase 2 final: levels=$WINNER_LEVELS @ lr=0.01 5-ep → BPB sliding $FINAL_BPB"
        fi
    fi
else
    # Winner is levels ∈ {5, 9, 11}: launch fresh 5-epoch at lr=0.01 with
    # NaN detection. Phase 2 forces lr=0.01 across all levels (heterogeneous
    # LR was Phase 1's design; Phase 2 tests fixed-LR feasibility now that
    # the cumsum overflow is fixed).
    echo ""
    echo "============================================================"
    echo "=== Test 5 Phase 2 winner: levels=$WINNER_LEVELS — launching 5-epoch follow-up"
    echo "============================================================"
    WINNER_PATCH=$(build_test5_patch $WINNER_LEVELS 5 0.01 0.0002)
    nan_safe_run "Test 5 Phase 2 final (5-ep): levels=$WINNER_LEVELS @ lr=0.01, bs=$TEST5_BLOCK_SIZE" "$WINNER_PATCH"
    FINAL_EXIT=$?
    FINAL_LOG_DIR=$(ls -dt logs/wikitext-103_*/ 2>/dev/null | head -1 | sed 's:/$::')
    if [ $FINAL_EXIT -eq 99 ]; then
        echo "[runs.sh] 5-epoch run NaN'd at levels=$WINNER_LEVELS / lr=0.01."
        git_commit_push "Test 5 Phase 2 final: levels=$WINNER_LEVELS @ lr=0.01 5-ep NaN'd"
    elif [ $FINAL_EXIT -ne 0 ]; then
        echo "[runs.sh] 5-epoch run failed exit $FINAL_EXIT."
        git_commit_push "Test 5 Phase 2 final: levels=$WINNER_LEVELS FAILED exit $FINAL_EXIT"
    else
        FINAL_BPB=$(parse_sliding_bpb_from_dir "$FINAL_LOG_DIR")
        echo "[runs.sh] 5-epoch BPB sliding: $FINAL_BPB"
        git_commit_push "Test 5 Phase 2 final: levels=$WINNER_LEVELS @ lr=0.01 5-ep → BPB sliding $FINAL_BPB"
    fi
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