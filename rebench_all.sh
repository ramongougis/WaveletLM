#!/bin/bash
# Re-benchmark every prior run folder with the fixed eval code.
# Usage:   ./rebench_all.sh
# Outputs: rebench_summary.csv with old vs new BPB/PPL per folder.
#
# Resilience: each folder is wrapped so a single failure (architecture
# drift, missing checkpoint, OOM, etc.) doesn't halt the sweep. Failed
# folders are logged in rebench_failures.log.
#
# Historical preservation: on first contact with a folder, the existing
# benchmark.txt is snapshotted to benchmark_v1_pre_fix.txt. OLD numbers
# are read from this snapshot so re-runs of this script don't lose the
# genuine historical figures.
#
# Per-folder full log: train.py output is captured to
# $folder/rebench_run.log for debugging — only the tail is shown live.
#
# Note: GNU grep's -oP is used. On macOS BSD grep, swap to -oE alternatives.

OUT="rebench_summary.csv"
FAIL_LOG="rebench_failures.log"
echo "folder,old_bpb_sl,new_bpb_sl,old_ppl_sl,new_ppl_sl,old_bpb_nov,new_bpb_nov,old_ppl_nov,new_ppl_nov,status" > "$OUT"
: > "$FAIL_LOG"

cleanup() {
    # Restore root config to a non-benchmark state on any exit
    python - <<'EOF'
import json
p = 'config.json'
c = json.load(open(p))
c['benchmark_only'] = False
c['benchmark_run_dir'] = ''
c['quantize_enabled'] = False
json.dump(c, open(p, 'w'), indent=4)
EOF
}
trap cleanup EXIT

# Flexible parser used for both OLD (from snapshot) and NEW values.
# benchmark.txt has two sections, non-overlapping then sliding window,
# each with Perplexity / BPT / BPB. head -1 grabs non-overlapping;
# tail -1 grabs sliding window.
extract_bpb_sl()  { grep -oP 'BPB:\s+\K[0-9.]+' "$1" 2>/dev/null | tail -1; }
extract_bpb_nov() { grep -oP 'BPB:\s+\K[0-9.]+' "$1" 2>/dev/null | head -1; }
extract_ppl_sl()  { grep -oP 'Perplexity:\s+\K[0-9.]+' "$1" 2>/dev/null | tail -1; }
extract_ppl_nov() { grep -oP 'Perplexity:\s+\K[0-9.]+' "$1" 2>/dev/null | head -1; }

for folder in logs/wikitext-103_* ; do
    if [ ! -f "$folder/best_model.pt" ]; then
        continue
    fi

    # Snapshot original benchmark.txt on first contact (preserves true
    # historical OLD numbers across multiple rebench_all.sh invocations).
    OLD_BENCH="$folder/benchmark.txt"
    SNAPSHOT="$folder/benchmark_v1_pre_fix.txt"
    if [ -f "$OLD_BENCH" ] && [ ! -f "$SNAPSHOT" ]; then
        cp "$OLD_BENCH" "$SNAPSHOT"
    fi

    # OLD numbers come from the snapshot if it exists, else the live file.
    OLD_SOURCE="$SNAPSHOT"
    [ ! -f "$OLD_SOURCE" ] && OLD_SOURCE="$OLD_BENCH"

    OLD_BPB_SL=""
    OLD_PPL_SL=""
    OLD_BPB_NOV=""
    OLD_PPL_NOV=""
    if [ -f "$OLD_SOURCE" ]; then
        OLD_BPB_SL=$(extract_bpb_sl "$OLD_SOURCE")
        OLD_PPL_SL=$(extract_ppl_sl "$OLD_SOURCE")
        OLD_BPB_NOV=$(extract_bpb_nov "$OLD_SOURCE")
        OLD_PPL_NOV=$(extract_ppl_nov "$OLD_SOURCE")
    fi

    # Patch root config to benchmark_only pointing at this folder
    python - "$folder" <<'EOF'
import json, sys
p = 'config.json'
c = json.load(open(p))
c['benchmark_only'] = True
c['benchmark_run_dir'] = sys.argv[1]
c['quantize_enabled'] = False
json.dump(c, open(p, 'w'), indent=4)
EOF

    echo ""
    echo "============================================================"
    echo "=== Rebenching: $folder"
    echo "============================================================"

    # Per-folder full log capture; live tail to stdout.
    PER_LOG="$folder/rebench_run.log"
    set +e
    python train.py > "$PER_LOG" 2>&1
    RC=$?
    set -e
    tail -80 "$PER_LOG"

    NEW_BENCH="$folder/benchmark.txt"
    if [ "$RC" -ne 0 ] || [ ! -f "$NEW_BENCH" ]; then
        STATUS="FAILED"
        NEW_BPB_SL=""
        NEW_PPL_SL=""
        NEW_BPB_NOV=""
        NEW_PPL_NOV=""
        echo "[FAIL] $folder (exit code $RC) — see $PER_LOG" | tee -a "$FAIL_LOG"
    else
        STATUS="ok"
        NEW_BPB_SL=$(extract_bpb_sl "$NEW_BENCH")
        NEW_PPL_SL=$(extract_ppl_sl "$NEW_BENCH")
        NEW_BPB_NOV=$(extract_bpb_nov "$NEW_BENCH")
        NEW_PPL_NOV=$(extract_ppl_nov "$NEW_BENCH")
    fi

    echo "$folder,${OLD_BPB_SL:--},${NEW_BPB_SL:--},${OLD_PPL_SL:--},${NEW_PPL_SL:--},${OLD_BPB_NOV:--},${NEW_BPB_NOV:--},${OLD_PPL_NOV:--},${NEW_PPL_NOV:--},$STATUS" >> "$OUT"
done

echo ""
echo "=== Rebench complete ==="
echo "Summary: $OUT"
if [ -s "$FAIL_LOG" ]; then
    echo "Failures: $FAIL_LOG"
    echo "  $(wc -l < $FAIL_LOG) folders failed; per-folder logs in <folder>/rebench_run.log"
fi
cat "$OUT"
