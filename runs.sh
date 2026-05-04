#!/bin/bash
# Mixer expansion ablations (1-epoch each), all with lifting_diaglowrank=true.
#
# Reference points:
#   - 1-epoch L=7 reference (uncompressed lifting, current widths):
#       logs/wikitext-103_2026-05-02_21-43-22 → BPB sliding 1.2361
#   - 1-epoch A1 (lifting_diaglowrank=true, current widths [1,1,1,1,.5,.5,.5,.5]):
#       logs/wikitext-103_2026-05-04_16-22-02 → BPB sliding 1.2860
#
# A1 freed ~114M params from the lifting cascade; these three ablations
# reinvest that freed capacity into the per-scale mixer at constant C.
# The math says ~1.5x uniform multiplier exactly matches the freed budget;
# 2x and 3x overshoot but test whether richer mixing helps even past parity.
#
# E1: per_scale_mixer_widths = [2,2,2,2, 0.5,0.5,0.5,0.5]    (~226M mixer)
# E2: per_scale_mixer_widths = [2,2,2,2, 1,1,1,1]             (~235M mixer)
# E4: per_scale_mixer_widths = [2.5,2.5,2.5,2.5, 0.5,0.5,0.5,0.5] (~319M mixer)
# E3: per_scale_mixer_widths = [3,3,3,3, 0.5,0.5,0.5,0.5]    (~428M mixer, tight on VRAM)
#
# Order: E1, E2, E4, E3 — ascending capacity. E4 placed before E3 so
# we still get a data point at 2.5× even if E3 OOMs and aborts the script
# (set -e exits on the first non-zero return, so E4 must run beforehand).
#
# Compare 1-epoch BPB sliding against A1's 1.2860; the best-performing
# config goes to a 5-epoch confirmation manually.

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
# Each ablation overrides only per_scale_mixer_widths and the lifting flag.
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
    "wavelet_crawl": false,
    "lifting_diaglowrank": true,
    "lifting_level_sharing": false,
    "lr": 0.01,
    "min_lr": 0.0002,
    "eval_interval": 250
}'

run_ablation() {
    local LABEL="$1"
    local OVERRIDE_JSON="$2"

    echo ""
    echo "============================================================"
    echo "=== Mixer expansion ablation: ${LABEL}"
    echo "============================================================"

    set_keys "$BASE_PATCH"
    set_keys "$OVERRIDE_JSON"
    python train.py
    git_commit_push "Mixer expansion ablation: ${LABEL} (1 epoch, L=7, lifting_diaglowrank=true)"
}

# E1: expand coarse-only to 2.0×, leave fine at 0.5
run_ablation "E1 widths=[2,2,2,2,.5,.5,.5,.5]" \
    '{"per_scale_mixer_widths": [2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5]}'

# E2: proportionally double current widths
run_ablation "E2 widths=[2,2,2,2,1,1,1,1]" \
    '{"per_scale_mixer_widths": [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]}'

# E4: midpoint between E2 and E3 in coarse-scale capacity. Runs BEFORE E3
# so we capture this data point even if E3 OOMs (set -e aborts on failure).
run_ablation "E4 widths=[2.5,2.5,2.5,2.5,.5,.5,.5,.5]" \
    '{"per_scale_mixer_widths": [2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5]}'

# E3: expand coarse-only to 3.0×, leave fine at 0.5 (largest, tight on VRAM)
run_ablation "E3 widths=[3,3,3,3,.5,.5,.5,.5]" \
    '{"per_scale_mixer_widths": [3.0, 3.0, 3.0, 3.0, 0.5, 0.5, 0.5, 0.5]}'

echo ""
echo "============================================================"
echo "=== Mixer expansion ablations complete."
echo "===   Comparison point: A1 1-epoch BPB sliding = 1.2860"
echo "===   (lifting_diaglowrank=true, default widths)"
echo "===   Best config wins → 5-epoch confirmation."
echo "===   Inspect each run's benchmark.txt or log.txt."
echo "============================================================"
