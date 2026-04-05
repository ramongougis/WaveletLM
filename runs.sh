# 3-epoch ablation retests: verify if 1-epoch results hold at longer training
# Each run resets to baseline, sets epochs=3, then toggles one setting

ABLATIONS=(
    "semantic_feedback:semantic_feedback:False"
    "lifting_linear_only:lifting_linear_only:True"
    "shared_lifting_weights:shared_lifting_weights:True"
)

for entry in "${ABLATIONS[@]}"; do
    IFS=':' read -r NAME KEY VALUE <<< "$entry"
    echo "=== 3-epoch ablation: $KEY=$VALUE ==="
    python -c "
import json
cfg = json.load(open('config.json'))
# Reset all to baseline
cfg['C'] = 512
cfg['lr'] = 0.01
cfg['mlp_expansion'] = 1
cfg['epochs'] = 3
cfg['semantic_feedback'] = True
cfg['semantic_feedback_cross_window'] = True
cfg['learned_residual'] = True
cfg['use_mixer_gate'] = True
cfg['skip_proj_out'] = False
cfg['shared_lifting_weights'] = False
cfg['lifting_linear_only'] = False
cfg['tie_embedding_to_lm_head'] = False
# Apply this ablation
cfg['$KEY'] = $VALUE
json.dump(cfg, open('config.json', 'w'), indent=4)
"
    python train.py
    git add .
    git commit --no-edit -m "3-epoch ablation: $KEY=$VALUE"
    git pull --no-edit
    git push
done
