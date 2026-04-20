#!/bin/bash
# 3-seed 10-epoch variance study at the best-run config.
# config.json is already set to L=2, C=2048, MLP=20, PLE, PKM+FwPKM=16384,
# lr=0.01, block_size=256, grad_accum=1, levels=5, low_rank=4, per_scale_mixer_widths,
# cross_scale_gating, wavelet_crawl K=3, shared_lifting_weights, 10 epochs, 2.5x dropout.
# Each run only overrides `seed` before training.

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
    echo "=== 10-epoch run, seed=$SEED ==="
    set_seed "$SEED"
    python train.py
    LATEST_CKPT=$(ls -dt logs/wikitext-103_*/best_model.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        python generate.py --checkpoint "$LATEST_CKPT"
        python generate.py --checkpoint "$LATEST_CKPT" --strategies
    fi
    git add .
    git commit --no-edit -m "10-epoch 3-seed variance: seed=$SEED"
    git pull --no-edit
    git push
}

run_seed 1337
run_seed 42
run_seed 7

# Reset seed to 1337 for cleanliness
set_seed 1337
git add config.json
git commit --no-edit -m "Reset seed to 1337 after 3-seed variance study"
git pull --no-edit
git push

echo "=== All 3 seeds complete ==="
