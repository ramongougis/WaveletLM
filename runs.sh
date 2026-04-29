#!/bin/bash
# Single-Layer WaveletLM (L=1) ablation series.
# Three runs covering L=1 vs L=2 at 1 vs 5 epochs on WikiText-103.
# The existing 5-epoch L=2 baseline (logs/wikitext-103_2026-04-22_01-36-47/)
# is the fourth comparison point and is not re-run here.
#
# Plan source: plans/single_layer_waveletlm.md
# Total estimated wall-clock on an RTX 5090: ~11 hours.
#
# Each run trains, runs the post-training benchmark (sliding window + non-
# overlapping), writes results to its own logs/wikitext-103_<timestamp>/,
# and runs two short generation samples (default + --strategies) for register
# verification. After all three runs complete, config.json is reset to the
# L=2 / 5-epoch baseline so the repo's default always reflects the released
# headline recipe.

set_keys() {
    python -c "
import json
cfg = json.load(open('config.json'))
patch = json.loads('''$1''')
cfg.update(patch)
json.dump(cfg, open('config.json', 'w'), indent=4)
"
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
        echo ""
        echo "[runs.sh] train.py failed in ${LABEL} with exit code $TRAIN_EXIT."
        echo "[runs.sh] Restoring baseline config and aborting remaining runs."
        set_keys '{"dataset": "wikitext-103", "layers": 2, "epochs": 5, "eval_interval": 250}'
        exit $TRAIN_EXIT
    fi

    local LATEST_CKPT
    LATEST_CKPT=$(ls -dt logs/wikitext-103_*/best_model.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        python generate.py --checkpoint "$LATEST_CKPT" || true
        python generate.py --checkpoint "$LATEST_CKPT" --strategies || true
    else
        echo "[runs.sh] No best_model.pt found for ${LABEL}. Skipping generation."
    fi
}

# ============================================================
# Run A — L=1, 1 epoch (modernized L=1 baseline)
# ============================================================
run_one "Run A: layers=1, epochs=1" \
    '{"dataset": "wikitext-103", "layers": 1, "epochs": 1, "eval_interval": 250}'

# ============================================================
# Run B — L=2, 1 epoch (head-to-head against Run A at fixed epochs)
# ============================================================
run_one "Run B: layers=2, epochs=1" \
    '{"dataset": "wikitext-103", "layers": 2, "epochs": 1, "eval_interval": 250}'

# ============================================================
# Run C — L=1, 5 epochs (the headline test: L=1 with full training time)
# ============================================================
run_one "Run C: layers=1, epochs=5" \
    '{"dataset": "wikitext-103", "layers": 1, "epochs": 5, "eval_interval": 250}'

# ============================================================
# Reset config to baseline and commit
# ============================================================
set_keys '{"dataset": "wikitext-103", "layers": 2, "epochs": 5, "eval_interval": 250}'

git add .
git commit --no-edit -m "L=1 ablation series: A (L=1, 1ep), B (L=2, 1ep), C (L=1, 5ep)"
git pull --no-edit
git push

echo ""
echo "============================================================"
echo "=== L=1 ablation series complete."
echo "===   Three new logs/wikitext-103_*/log.txt files now exist."
echo "===   Pull BPB sliding-window numbers and update:"
echo "===     - plans/single_layer_waveletlm.md (results section)"
echo "===     - runs.md (new L=1 ablation section)"
echo "===     - README.md (Future Plans -> Single-Layer subsection)"
echo "============================================================"
