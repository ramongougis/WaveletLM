# Boolean ablations: C=512, epochs=1, mlp_expansion=1
# Each run toggles one setting from baseline

# Setting name, config key, value
ABLATIONS=(
    "semantic_feedback:semantic_feedback:False"
    "semantic_feedback_cross_window:semantic_feedback_cross_window:False"
    "learned_residual:learned_residual:False"
    "use_mixer_gate:use_mixer_gate:False"
    "skip_proj_out:skip_proj_out:True"
    "shared_lifting_weights:shared_lifting_weights:True"
    "lifting_linear_only:lifting_linear_only:True"
    "tie_embedding_to_lm_head:tie_embedding_to_lm_head:True"
)

for entry in "${ABLATIONS[@]}"; do
    IFS=':' read -r NAME KEY VALUE <<< "$entry"
    echo "=== Ablation: $KEY=$VALUE ==="
    python -c "
import json
cfg = json.load(open('config.json'))
# Reset all to baseline
cfg['C'] = 512
cfg['lr'] = 0.01
cfg['mlp_expansion'] = 1
cfg['epochs'] = 1
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
    git commit --no-edit -m "boolean ablation: $KEY=$VALUE"
    git pull --no-edit
    git push
done
