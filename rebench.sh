#!/bin/bash
# rebench.sh — re-run the benchmark phase against each existing checkpoint
# whose original benchmark produced inf BPB due to the fp16-overflow bug in
# the cross-entropy accumulator. Train.py was patched to cast logits to fp32
# before cross_entropy + .sum(), so the fix is purely on the benchmark path
# and prior model weights / training are unaffected.
#
# How this works:
# 1. For each target run, set the ROOT config.json to {benchmark_only: true,
#    benchmark_run_dir: <target>}. train.py then reads the run's own
#    config.json + best_model.pt and runs ONLY the benchmark phase against
#    that checkpoint, appending the new (now-finite) BPB lines to the run's
#    log.txt.
# 2. Parse the LAST [BENCHMARK - ...] block in the run's log.txt to extract
#    the corrected BPB. Earlier inf lines remain in the log for traceability.
# 3. git commit + push the updated log so the rebenched values propagate.
# 4. After all targets, restore the root config.json to its pre-script state.
#
# Targets list = May 2026 sweep iterations whose training was NaN-free but
# whose benchmark produced inf. April 2026 inf runs were experimental; their
# checkpoints are not preserved on the pod and they're not load-bearing for
# any current finding.

# Targets — directories with stored best_model.pt + config.json
TARGETS=(
    "logs/wikitext-103_2026-05-02_17-52-31"  # levels=5, crawl=true, 1-epoch
    "logs/wikitext-103_2026-05-02_20-32-04"  # levels=5, crawl=false, 1-epoch
    "logs/wikitext-103_2026-05-02_21-43-22"  # levels=7, crawl=false, 1-epoch
    "logs/wikitext-103_2026-05-03_09-04-00"  # levels=9, crawl=false, lr=3.42e-3, 1-epoch
    "logs/wikitext-103_2026-05-03_10-38-19"  # levels=11, crawl=false, lr=1.14e-3, 1-epoch
)

# Backup root config.json before mutating it
cp config.json config.json.rebench-backup
echo "[rebench.sh] Backed up root config.json → config.json.rebench-backup"

set_keys() {
    python -c "
import json
cfg = json.load(open('config.json'))
patch = json.loads('''$1''')
cfg.update(patch)
json.dump(cfg, open('config.json', 'w'), indent=4)
"
}

git_commit_push() {
    local MSG="$1"
    git add . 2>/dev/null || true
    git commit --no-edit -m "${MSG}" 2>/dev/null || true
    git pull --no-edit 2>/dev/null || true
    git push 2>/dev/null || true
}

# Parse the LAST [BENCHMARK - <kind>] block for its BPB. Handles both finite
# numbers (e.g. "1.2540") and "inf". Returns "N/A" if no match.
parse_last_bpb() {
    local LOG_FILE="$1"
    local KIND="$2"  # "Non-overlapping" or "Sliding Window"
    python -c "
import re
log = open('$LOG_FILE', encoding='utf-8', errors='replace').read()
matches = list(re.finditer(
    r'\[BENCHMARK - $KIND\].*?BPB:\s*([\d.]+|inf)', log, re.DOTALL))
print(matches[-1].group(1) if matches else 'N/A')
"
}

# Iterate
echo ""
echo "[rebench.sh] ${#TARGETS[@]} target(s) to re-benchmark."

for TARGET in "${TARGETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "=== Re-benchmark: $TARGET"
    echo "============================================================"

    if [ ! -f "$TARGET/best_model.pt" ]; then
        echo "[rebench.sh] SKIP: $TARGET — no best_model.pt found"
        continue
    fi
    if [ ! -f "$TARGET/config.json" ]; then
        echo "[rebench.sh] SKIP: $TARGET — no config.json found"
        continue
    fi

    # Patch root config to point train.py at this target in benchmark_only mode
    set_keys "{\"benchmark_only\": true, \"benchmark_run_dir\": \"$TARGET\"}"

    python train.py
    EXIT=$?

    if [ $EXIT -ne 0 ]; then
        echo "[rebench.sh] FAILED exit $EXIT for $TARGET"
        git_commit_push "Rebench: $TARGET FAILED (exit $EXIT)"
        continue
    fi

    BPB_NONOVR=$(parse_last_bpb "$TARGET/log.txt" "Non-overlapping")
    BPB_SLIDE=$(parse_last_bpb "$TARGET/log.txt" "Sliding Window")
    echo "[rebench.sh] $TARGET → non-overlap BPB=$BPB_NONOVR, sliding BPB=$BPB_SLIDE"
    git_commit_push "Rebench (fp32 CE accumulator fix): $TARGET → non-overlap=$BPB_NONOVR, sliding=$BPB_SLIDE"
done

# Restore root config.json
mv config.json.rebench-backup config.json
echo ""
echo "============================================================"
echo "[rebench.sh] All re-benchmarks complete. Restored root config.json."
echo "============================================================"
echo ""
echo "Summary of corrected BPBs:"
echo ""
printf "  %-50s  %-12s  %-12s\n" "Target" "Non-overlap" "Sliding"
printf "  %-50s  %-12s  %-12s\n" "----" "----" "----"
for TARGET in "${TARGETS[@]}"; do
    if [ -f "$TARGET/log.txt" ]; then
        BPB_NONOVR=$(parse_last_bpb "$TARGET/log.txt" "Non-overlapping")
        BPB_SLIDE=$(parse_last_bpb "$TARGET/log.txt" "Sliding Window")
        printf "  %-50s  %-12s  %-12s\n" "$(basename $TARGET)" "$BPB_NONOVR" "$BPB_SLIDE"
    fi
done
