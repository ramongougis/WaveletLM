#!/bin/bash
# 5-epoch confirmation of E1 mixer expansion (lifting_diaglowrank=true,
# per_scale_mixer_widths=[2,2,2,2,0.5,0.5,0.5,0.5]).
#
# Reference points:
#   - 5-epoch L=7 reference (uncompressed lifting, current widths):
#       logs/wikitext-103_2026-05-03_02-13-07 → BPB sliding 1.0974
#   - 1-epoch A1 (lifting_diaglowrank=true, current widths):
#       logs/wikitext-103_2026-05-04_16-22-02 → BPB sliding 1.2860
#
# E1 spec at L=7 / bs=16384 / Cp=2048:
#   - Mixer: ~226M (4 coarse @ width=4096, 4 fine @ width=1024)
#   - Total: ~446M (vs 392.91M headline; vs 278.74M A1 default-widths)
#   - VRAM: ~31 GiB peak (5% headroom on a 5090's 32.6 GiB)
#
# E2/E4/E3 dropped from this round: E1 already maxes VRAM, so larger
# expansions OOM. E2 (~similar memory to E1) optionally queued after if
# E1 ships, as a final sanity ablation on whether widening fine scales
# from 0.5 → 1.0 helps.
#
# Pass criterion (5-epoch BPB sliding):
#   - Excellent  : ≤ 1.0974  (matches or beats uncompressed reference)
#   - Pragmatic  : ≤ 1.10    (29% smaller lifting + richer mixer wins)
#   - Regression : > 1.10    (mixer expansion didn't recover the lifting compression cost)

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

# E1 5-epoch, single run.
set_keys '{
    "dataset": "wikitext-103",
    "layers": 1,
    "epochs": 5,
    "mlp_expansion": 10,
    "pkm_enabled": false,
    "fwpkm_num_keys": 8281,
    "tie_embedding_to_lm_head": true,
    "micro_batch_size": 1,
    "grad_accum": 1,
    "block_size": 16384,
    "levels": 7,
    "per_scale_mixer_widths": [2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5],
    "wavelet_crawl": false,
    "lifting_diaglowrank": true,
    "lifting_level_sharing": false,
    "lr": 0.01,
    "min_lr": 0.0002,
    "eval_interval": 250
}'

echo ""
echo "============================================================"
echo "=== E1 mixer expansion + lifting compression — 5-epoch confirm"
echo "===   widths=[2,2,2,2,.5,.5,.5,.5], lifting_diaglowrank=true"
echo "===   ~446M total, 226M mixer, ~31 GiB VRAM peak"
echo "===   5-epoch reference: 1.0974 (uncompressed)"
echo "===   Targets: ≤1.0974 excellent, ≤1.10 pragmatic, >1.10 regression"
echo "============================================================"

python train.py
git_commit_push "E1 mixer expansion + lifting_diaglowrank, 5-epoch L=7"
