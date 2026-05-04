#!/bin/bash
# E5: uncompressed lifting + moderate mixer expansion (5-epoch confirm).
#
# Hypothesis: compressed lifting bottlenecks the expanded mixer's value.
# E1 5-epoch (compressed lifting + width=2.0 coarse) showed a widening
# val-loss gap to the uncompressed reference, projecting BPB ≈ 1.14
# (regression past pragmatic 1.10 threshold). E5 tests whether full
# lifting expressivity + moderate mixer expansion at constant C beats
# the headline reference.
#
# Reference points:
#   - 5-epoch L=7 reference (uncompressed, default widths):
#       logs/wikitext-103_2026-05-03_02-13-07 → BPB sliding 1.0974, 392.91M
#   - E1 5-epoch (compressed + width=2.0 coarse, killed mid-run):
#       logs/wikitext-103_2026-05-04_19-35-25 → projected BPB ~1.14, 446M
#
# E5 spec at L=7 / bs=16384 / Cp=2048:
#   - lifting_diaglowrank=false (full 117.44M lifting, uncompressed)
#   - per_scale_mixer_widths=[1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5]
#       Coarse @ width=3072 → 31.46M per scale → 125.8M total
#       Fine @ width=1024 → 6.29M per scale → 25.2M total
#       Mixer total: ~151M (vs default 58.8M, vs E1 226M)
#   - Total: ~488M (vs 393M headline, vs E1 446M)
#   - VRAM: ~32 GiB peak — tight on a 32.6 GiB card. If first 250 steps
#     log shows >32 GiB, pull widths back to [1.25, 1.25, 1.25, 1.25,
#     0.5, 0.5, 0.5, 0.5] or fine to 0.25.
#
# Pass criteria (5-epoch BPB sliding):
#   - Excellent  : ≤ 1.0974  (matches or beats uncompressed reference)
#   - Strong     : ≤ 1.0959  (better than reference by ≥ 0.0015 noise floor)
#   - Pragmatic  : ≤ 1.10    (24% larger model with marginal cost worth shipping)
#   - Regression : > 1.10    (mixer expansion alone doesn't recover ≤ 0.0015 BPB)

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
    "per_scale_mixer_widths": [1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5],
    "wavelet_crawl": false,
    "lifting_diaglowrank": false,
    "lifting_level_sharing": false,
    "lr": 0.01,
    "min_lr": 0.0002,
    "eval_interval": 250
}'

echo ""
echo "============================================================"
echo "=== E5: uncompressed lifting + moderate mixer expansion (5 epochs)"
echo "===   widths=[1.5,1.5,1.5,1.5,.5,.5,.5,.5], lifting full"
echo "===   ~488M total, 117M lifting, 151M mixer, ~32 GiB VRAM peak"
echo "===   5-epoch reference: 1.0974"
echo "===   Targets: ≤1.0959 strong, ≤1.0974 excellent, ≤1.10 pragmatic, >1.10 regression"
echo "============================================================"

python train.py
git_commit_push "E5 uncompressed lifting + width=1.5 coarse mixer, 5-epoch L=7"
