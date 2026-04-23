#!/bin/bash
# 3-seed variance study (seeds 42, 7). Seed 1337 already completed at
# logs/wikitext-103_2026-04-22_01-36-47 (BPB 1.0140, PPL 23.7490). Each run
# below edits config.json's seed, trains, runs generate.py (standard +
# strategies), commits and pushes.

set_seed() {
    python -c "
import json
cfg = json.load(open('config.json'))
cfg['seed'] = $1
json.dump(cfg, open('config.json', 'w'), indent=4)
"
}

run_seed() {
    local SEED="$1"

    echo ""
    echo "============================================================"
    echo "=== 3-seed run: seed=$SEED"
    echo "============================================================"
    set_seed "$SEED"
    python train.py
    LATEST_CKPT=$(ls -dt logs/wikitext-103_*/best_model.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        python generate.py --checkpoint "$LATEST_CKPT"
        python generate.py --checkpoint "$LATEST_CKPT" --strategies
    fi
    git add .
    git commit --no-edit -m "3-seed variance study: seed=$SEED"
    git pull --no-edit
    git push
}

run_seed 42
run_seed 7

# Reset seed to the primary.
set_seed 1337

echo ""
echo "============================================================"
echo "=== 3-seed sweep complete"
echo "===   Update runs.md with BPB/PPL for seeds 42 and 7,"
echo "===   compute mean BPB ± std across all three seeds."
echo "============================================================"
