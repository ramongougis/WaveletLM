#!/bin/bash
# Re-benchmark every prior run folder with the fixed eval code.
# Usage:   ./rebench_all.sh
# Outputs: rebench_summary.csv with old vs new BPB/PPL per folder.
#
# Preconditions:
#   1. Step 0 verification has been run (verify_canonical_bytes.py) and
#      encode_split in train.py uses the correct text_for_bytes join.
#   2. .cache/ has been cleared so the fixed encoder re-runs:
#        rm -rf .cache/
#   3. The train.py fixes (flag disable+restore in both eval fns, continue
#      removals, asymmetric denominator) are in place.

set -e

OUT="rebench_summary.csv"
echo "folder,old_bpb_sl,new_bpb_sl,old_ppl_sl,new_ppl_sl,old_bpb_nov,new_bpb_nov,old_ppl_nov,new_ppl_nov" > "$OUT"

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

for folder in logs/wikitext-103_* ; do
    if [ ! -f "$folder/best_model.pt" ]; then
        continue
    fi

    # Harvest previously reported numbers from the run's benchmark.txt (if present)
    # Note: GNU grep's -oP is used here; use -oE alternatives on macOS.
    OLD_BENCH="$folder/benchmark.txt"
    OLD_BPB_SL="—"
    OLD_PPL_SL="—"
    OLD_BPB_NOV="—"
    OLD_PPL_NOV="—"
    if [ -f "$OLD_BENCH" ]; then
        OLD_BPB_SL=$(grep -oP '(?<=BPB \(sliding\):\s{5})[0-9.]+' "$OLD_BENCH" 2>/dev/null | tail -1 || echo "—")
        # Fallback pattern variants (different versions of the logger used slightly different formats)
        [ -z "$OLD_BPB_SL" ] && OLD_BPB_SL=$(grep -A1 'SLIDING WINDOW' "$OLD_BENCH" | grep -oP 'BPB.*?[0-9.]+' | tail -1 || echo "—")
        OLD_PPL_SL=$(grep -oP '(?<=Perplexity:\s{9})[0-9.]+' "$OLD_BENCH" 2>/dev/null | tail -1 || echo "—")
        OLD_BPB_NOV=$(grep -oP '(?<=BPB:\s{15})[0-9.]+' "$OLD_BENCH" 2>/dev/null | head -1 || echo "—")
        OLD_PPL_NOV=$(grep -oP '(?<=Perplexity:\s{9})[0-9.]+' "$OLD_BENCH" 2>/dev/null | head -1 || echo "—")
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

    echo "============================================================"
    echo "=== Rebenching: $folder"
    echo "============================================================"
    python train.py 2>&1 | tail -80

    # Extract new numbers from the freshly-written benchmark.txt
    NEW_BENCH="$folder/benchmark.txt"
    NEW_BPB_SL=$(grep -oP 'BPB:\s+\K[0-9.]+' "$NEW_BENCH" | tail -1 || echo "—")
    NEW_PPL_SL=$(grep -oP 'Perplexity:\s+\K[0-9.]+' "$NEW_BENCH" | tail -1 || echo "—")
    NEW_BPB_NOV=$(grep -oP 'BPB:\s+\K[0-9.]+' "$NEW_BENCH" | head -1 || echo "—")
    NEW_PPL_NOV=$(grep -oP 'Perplexity:\s+\K[0-9.]+' "$NEW_BENCH" | head -1 || echo "—")

    echo "$folder,${OLD_BPB_SL:--},${NEW_BPB_SL:--},${OLD_PPL_SL:--},${NEW_PPL_SL:--},${OLD_BPB_NOV:--},${NEW_BPB_NOV:--},${OLD_PPL_NOV:--},${NEW_PPL_NOV:--}" >> "$OUT"
done

echo ""
echo "=== Rebench complete ==="
echo "Summary: $OUT"
cat "$OUT"
