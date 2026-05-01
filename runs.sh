#!/bin/bash
# Combined parameter reduction test (L=1, E=5).
# Tests the four cheap reductions described in plans/other_post_release_plans.md §8:
#   - mlp_expansion: 20 → 10
#   - pkm_enabled: true → false (PKM dropped; FwPKM retained for inference-update potential)
#   - fwpkm_num_keys: 16384 → 8281 (= 91², closest perfect square to half of 16384)
#   - tie_embedding_to_lm_head: false → true
# Projected: ~586M (current L=1) → ~340M params (~42% reduction).
#
# After completion, the config is reset to the L=1 E=5 iteration platform default.
# Estimated wall-clock on RTX 5090: ~5-7h (smaller model trains faster than baseline L=1).

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
        echo "[runs.sh] Restoring L=1 E=5 iteration default and aborting remaining runs."
        set_keys '{"dataset": "wikitext-103", "layers": 1, "epochs": 5, "mlp_expansion": 20, "pkm_enabled": true, "fwpkm_num_keys": 16384, "tie_embedding_to_lm_head": false, "eval_interval": 250}'
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
# Combined parameter reduction baseline
# ============================================================
run_one "Combined param reduction: L=1, E=5, MLP=10, PKM off, FwPKM=8281, tied emb" \
    '{"dataset": "wikitext-103", "layers": 1, "epochs": 5, "mlp_expansion": 10, "pkm_enabled": false, "fwpkm_num_keys": 8281, "tie_embedding_to_lm_head": true, "eval_interval": 250}'

# ============================================================
# Reset config to L=1 E=5 iteration platform default
# ============================================================
set_keys '{"dataset": "wikitext-103", "layers": 1, "epochs": 5, "mlp_expansion": 20, "pkm_enabled": true, "fwpkm_num_keys": 16384, "tie_embedding_to_lm_head": false, "eval_interval": 250}'

git add .
git commit --no-edit -m "Combined parameter reduction test (L=1, E=5)"
git pull --no-edit
git push

echo ""
echo "============================================================"
echo "=== Combined parameter reduction test complete."
echo "===   Pull BPB sliding-window numbers and update:"
echo "===     - plans/other_post_release_plans.md §8 (results section)"
echo "===     - plans/findings.md (new entry under Combined Parameter Reduction)"
echo "===     - runs.md (results table)"
echo "===     - README.md (Combined Parameter Reduction subsection)"
echo "===   Then plan the two follow-up variants (max EBS, larger block size)."
echo "============================================================"
