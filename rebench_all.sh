#!/bin/bash
# Re-benchmark every prior run folder with the fixed eval code.
# Usage:   ./rebench_all.sh
#
# Single-responsibility design:
#   1. Pre-rename: every existing benchmark.txt → benchmark_pre_v2.txt
#      (avoids any chance of overwriting historical numbers).
#   2. Set benchmark_only=True in root config.json.
#   3. For each run folder with a checkpoint, set benchmark_run_dir to it
#      and invoke `python train.py`. train.py reads architecture from the
#      run's own config.json (with feature-off defaults for any missing
#      arch keys) and writes a fresh benchmark.txt to that folder.
#   4. Restore root config (benchmark_only=False, benchmark_run_dir='').
#
# Per-folder failures don't halt the sweep — failures are logged to
# rebench_failures.log. Post-processing (collecting new BPB/PPL into a
# CSV, comparing against benchmark_pre_v2.txt) is a separate concern.

FAIL_LOG="rebench_failures.log"
: > "$FAIL_LOG"

cleanup() {
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

# ----- Step 1: pre-rename existing benchmark.txt files -----
echo "=== Pre-renaming existing benchmark.txt → benchmark_pre_v2.txt ==="
for folder in logs/wikitext-103_* ; do
    if [ -f "$folder/benchmark.txt" ] && [ ! -f "$folder/benchmark_pre_v2.txt" ]; then
        mv "$folder/benchmark.txt" "$folder/benchmark_pre_v2.txt"
        echo "  $folder"
    fi
done

# ----- Step 2: enable benchmark_only globally -----
python - <<'EOF'
import json
p = 'config.json'
c = json.load(open(p))
c['benchmark_only'] = True
c['quantize_enabled'] = False
json.dump(c, open(p, 'w'), indent=4)
EOF

# ----- Step 3: loop folders, set benchmark_run_dir, run train.py -----
for folder in logs/wikitext-103_* ; do
    if [ ! -f "$folder/best_model.pt" ]; then
        continue
    fi

    python - "$folder" <<'EOF'
import json, sys
p = 'config.json'
c = json.load(open(p))
c['benchmark_run_dir'] = sys.argv[1]
json.dump(c, open(p, 'w'), indent=4)
EOF

    echo ""
    echo "============================================================"
    echo "=== Rebenching: $folder"
    echo "============================================================"

    set +e
    python train.py
    RC=$?
    set -e

    if [ "$RC" -ne 0 ] || [ ! -f "$folder/benchmark.txt" ]; then
        echo "[FAIL] $folder (exit code $RC)" | tee -a "$FAIL_LOG"
    fi
done

echo ""
echo "=== Rebench complete ==="
if [ -s "$FAIL_LOG" ]; then
    echo "Failures: $(wc -l < $FAIL_LOG) folders failed (see $FAIL_LOG)"
fi
echo "Each successful folder now has a fresh benchmark.txt and a preserved benchmark_pre_v2.txt."
echo "Run the post-processing/CSV step separately when you want a comparison summary."
