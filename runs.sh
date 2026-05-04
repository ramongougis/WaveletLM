#!/bin/bash
# Wavelet compression ablations (1-epoch each).
#
# Reference: L=1 / levels=7 / bs=16384 / wavelet_crawl=False / 1-epoch run at
#   logs/wikitext-103_2026-05-02_21-43-22 → BPB sliding 1.2361, 392.91M params
# Pass criterion: each ablation must land within ±0.018 BPB of 1.2361, i.e.
# in [1.2181, 1.2541]. Survivors get combined for a 5-epoch confirmation
# (manual decision after these 3 runs finish).
#
# Three ablations:
#   A1: lifting_diaglowrank=true,  lifting_level_sharing=false  (D+UV^T only)
#   A2: lifting_diaglowrank=false, lifting_level_sharing=true   (group sharing only)
#   A3: lifting_diaglowrank=true,  lifting_level_sharing=true   (combined)
#
# Each run takes ~70 min on a 5090 (matches the original 1-epoch L=7 wall clock).

set -euo pipefail

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
    git add . || true
    git commit --no-edit -m "${MSG}" || true
    git pull --no-edit || true
    git push || true
}

# Common patch matching the 1-epoch L=7 reference run's configuration.
# (Same as the build_test5_patch baseline minus per-ablation overrides.)
BASE_PATCH='{
    "dataset": "wikitext-103",
    "layers": 1,
    "epochs": 1,
    "mlp_expansion": 10,
    "pkm_enabled": false,
    "fwpkm_num_keys": 8281,
    "tie_embedding_to_lm_head": true,
    "micro_batch_size": 1,
    "grad_accum": 1,
    "block_size": 16384,
    "levels": 7,
    "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
    "wavelet_crawl": false,
    "lr": 0.01,
    "min_lr": 0.0002,
    "eval_interval": 250
}'

run_ablation() {
    local LABEL="$1"
    local OVERRIDE_JSON="$2"

    echo ""
    echo "============================================================"
    echo "=== Wavelet compression ablation: ${LABEL}"
    echo "============================================================"

    set_keys "$BASE_PATCH"
    set_keys "$OVERRIDE_JSON"
    python train.py
    git_commit_push "Wavelet compression ablation: ${LABEL} (1 epoch, L=7, ref BPB 1.2361)"
}

# A1: D+UV^T only
run_ablation "A1 diaglowrank-only" \
    '{"lifting_diaglowrank": true,  "lifting_level_sharing": false}'

# A2: level sharing only
run_ablation "A2 level-sharing-only" \
    '{"lifting_diaglowrank": false, "lifting_level_sharing": true}'

# A3: both combined
run_ablation "A3 diaglowrank+level-sharing" \
    '{"lifting_diaglowrank": true,  "lifting_level_sharing": true}'

echo ""
echo "============================================================"
echo "=== All 3 ablations complete."
echo "===   Pass criterion: BPB sliding within [1.2181, 1.2541]"
echo "===   (1-epoch L=7 reference: 1.2361 ±0.018)"
echo "===   Inspect each run's benchmark.txt or log.txt."
echo "===   Take survivor(s) to a 5-epoch confirmation manually."
echo "============================================================"
