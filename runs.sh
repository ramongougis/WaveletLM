#!/bin/bash
# Pre-3-seed probes. Each probe edits only the relevant config keys, runs
# train.py + generate.py (standard + strategies), commits/pushes, and resets
# the changed keys before the next probe. Decision criteria for which to
# adopt are in runs.md (EMA, WD, 2.0x dropout sections).
#
# Probe order (cheapest signal first):
#   1. EMA probe — 1 epoch, decompose_bypass_ema=true
#   2. WD 1e-6 — 5 epochs
#   3. WD 1e-7 — 5 epochs
#
# After all three complete, manually inspect the BPB/PPL deltas vs the 5-epoch
# best (1.0201 BPB / 24.21 PPL) and update runs.md before kicking off the
# 3-seed runs (which live in a separate driver, to be added once the recipe
# is locked).

set_keys() {
    # Usage: set_keys '{"epochs": 1, "weight_decay": 0.0, ...}'
    python -c "
import json
cfg = json.load(open('config.json'))
patch = json.loads('''$1''')
cfg.update(patch)
json.dump(cfg, open('config.json', 'w'), indent=4)
"
}

run_probe() {
    local NAME="$1"
    local PATCH="$2"
    local RESET_PATCH="$3"

    echo ""
    echo "============================================================"
    echo "=== Probe: $NAME"
    echo "===   patch: $PATCH"
    echo "============================================================"
    set_keys "$PATCH"
    python train.py
    LATEST_CKPT=$(ls -dt logs/wikitext-103_*/best_model.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        python generate.py --checkpoint "$LATEST_CKPT"
        python generate.py --checkpoint "$LATEST_CKPT" --strategies
    fi
    git add .
    git commit --no-edit -m "Probe: $NAME"
    git pull --no-edit
    git push

    # Reset the keys this probe changed so the next probe starts from the
    # post-reset baseline (6 epochs, 2.0x dropout, no EMA, no WD).
    set_keys "$RESET_PATCH"
}

# ====================================================================
# Probe 1 — EMA probe (~5h)
# Tests data-dependent EMA replacement of cumulative running mean in
# decompose_bypass. 1 epoch, single forward pass, structural test.
# ====================================================================
# run_probe \
#     "EMA probe (1 epoch, decompose_bypass_ema=true)" \
#     '{"epochs": 1, "decompose_bypass_ema": true}' \
#     '{"epochs": 6, "decompose_bypass_ema": false}'

# ====================================================================
# Probe 2 — Weight decay 1e-6 (~14h)
# Low-WD probe. Original WD=1e-3 stalled training; 1e-6 should be small
# enough to not interact with Adagrad accumulators yet still detectable
# if WD has any beneficial effect at all.
# ====================================================================
# run_probe \
#     "WD 1e-6 probe (5 epochs)" \
#     '{"epochs": 5, "weight_decay": 1e-6}' \
#     '{"epochs": 6, "weight_decay": 0.0}'

# ====================================================================
# Probe 3 — 5-epoch EMA probe (~17h)
# The 1-epoch EMA smoke test showed a 0.30-nat val-loss improvement
# (3.4580 vs 3.7640 baseline). Extending to 5 epochs to see whether
# the gain compounds through training or asymptotes. Replaces what
# was originally the WD 1e-7 probe — further regularization sweeps
# (including WD grid and dropout grid) are deferred to post-release,
# since the 1e-6 probe already confirms WD has negligible effect at
# this scale and compute budget is better spent characterizing EMA.
# ====================================================================
run_probe \
    "EMA 5-epoch run (decompose_bypass_ema=true)" \
    '{"epochs": 5, "decompose_bypass_ema": true}' \
    '{"epochs": 6, "decompose_bypass_ema": false}'

echo ""
echo "============================================================"
echo "=== All probes complete ==="
echo "===   Compare each probe folder's benchmark.txt to:"
echo "===     5-epoch best: logs/wikitext-103_2026-04-19_13-16-24"
echo "===     (BPB 1.0201, PPL 24.21, post-fix eval)"
echo "===   If EMA 5-epoch beats 1.0201, adopt it for the 3-seed"
echo "===   variance study. Update runs.md then update runs.sh"
echo "===   for the 3-seed runs at the winning recipe."
echo "============================================================"
