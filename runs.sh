# Mixer depth sweep + MLP exp=50 + Memory sweep
# Each run resets config.json to the 1-epoch baseline, then applies its changes.
# Baseline sourced from logs/wikitext-103_2026-04-03_04-51-07/config.json

BASELINE='
import json

baseline = {
    "dataset": "wikitext-103",
    "out_dir": "logs",
    "compile": True,
    "seed": 1337,
    "epochs": 1,
    "micro_batch_size": 8,
    "grad_accum": 2,
    "block_size": 512,
    "eval_interval": 100,
    "skip_warmup_saves": True,
    "C": 512,
    "layers": 20,
    "levels": 9,
    "low_rank": 0,
    "mlp_expansion": 1,
    "mlp_layers": 2,
    "pkm_enabled": False,
    "pkm_num_keys": 529,
    "pkm_top_k": 32,
    "pkm_heads": 1,
    "fwpkm_enabled": False,
    "fwpkm_num_keys": 529,
    "fwpkm_top_k": 32,
    "fwpkm_heads": 1,
    "fwpkm_inference_updates": False,
    "fwpkm_update_lr": 0.01,
    "fwpkm_chunk_size": 64,
    "wavelet_mode": "lifting",
    "shared_lifting_weights": False,
    "lifting_linear_only": False,
    "lifting_hidden_mult": 1,
    "lifting_init": "haar",
    "lifting_dropout": 0.0,
    "use_mixer_gate": True,
    "mixer_gate_activation": "silu",
    "mixer_depth": 1,
    "mixer_depth_stabilizers": False,
    "semantic_feedback": True,
    "semantic_feedback_cross_window": True,
    "learned_residual": True,
    "skip_proj_out": False,
    "stochastic_depth_rate": 0.0,
    "dropout_embedding": 0.0,
    "dropout_projection": 0.0,
    "dropout_mixer": 0.0,
    "dropout_mlp": 0.0,
    "dropout_lm_head": 0.0,
    "optimizer": "Adagrad",
    "optimizer_eps": 2e-13,
    "lr": 0.01,
    "min_lr": 0.0002,
    "warmup_fraction": 0.3,
    "grad_clip": 1.0,
    "tie_embedding_to_lm_head": False,
    "gradient_checkpointing": False,
    "use_amp": True,
    "amp_dtype": "fp16",
    "allow_tf32": True,
    "generation_prompt": "The history of",
    "temperature": 1.0,
    "num_new_tokens": 512,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "benchmark_only": False,
    "benchmark_run_dir": "",
    "multinodal_enabled": False,
    "multinodal_num_cells": 2,
    "multinodal_cell_dim": 512,
    "multinodal_seeds": [42, 137],
    "multinodal_combination": "average",
    "multinodal_cross_cell_gating": False,
    "multinodal_cross_cell_gate_interval": 1,
    "multinodal_features_per_cell": -1,
    "multinodal_bagged_eps": 1e-06,
    "quantize_enabled": False,
    "quantize_mixer_coarse_bits": 8,
    "quantize_mixer_mid_bits": 4,
    "quantize_mixer_fine_bits": 2,
    "quantize_mlp_bits": 4,
    "quantize_lifting_bits": 16,
    "quantize_embedding_bits": 8,
    "per_layer_embedding": False
}
'

reset_config() {
    python -c "
$BASELINE
import json
json.dump(baseline, open('config.json', 'w'), indent=4)
"
}

run_with() {
    local NAME="$1"
    local OVERRIDES="$2"
    echo "=== $NAME ==="
    python -c "
$BASELINE
cfg = dict(baseline)
$OVERRIDES
import json
json.dump(cfg, open('config.json', 'w'), indent=4)
"
    python train.py
    LATEST_CKPT=$(ls -dt logs/wikitext-103_*/best_model.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        python generate.py --checkpoint "$LATEST_CKPT"
    fi
    git add .
    git commit --no-edit -m "$NAME"
    git pull --no-edit
    git push
}

# =====================================================================
# MIXER DEPTH + LOWER LR (peak LR below 0.008 NaN threshold)
# =====================================================================

run_with "Mixer depth: 5, lr=0.004" "cfg['mixer_depth'] = 5; cfg['lr'] = 0.004"
run_with "Mixer depth: 10, lr=0.001" "cfg['mixer_depth'] = 10; cfg['lr'] = 0.001; cfg['micro_batch_size'] = 4; cfg['grad_accum'] = 4"

# =====================================================================
# LAYERS SWEEP (epochs=1, mlp_expansion=1, C=512, levels=9)
# =====================================================================

# layers=20 is baseline (Run 4), no need to rerun
run_with "Layers: 1" "cfg['layers'] = 1"
run_with "Layers: 4" "cfg['layers'] = 4"
run_with "Layers: 10" "cfg['layers'] = 10"
run_with "Layers: 15" "cfg['layers'] = 15"
run_with "Layers: 18" "cfg['layers'] = 18"
run_with "Layers: 30" "cfg['layers'] = 30"

# =====================================================================
# LEVELS SWEEP (epochs=1, mlp_expansion=1, C=512, layers=20, block_size=512)
# =====================================================================

# levels=9 is baseline (default = log2(block_size=512)), no need to rerun
run_with "Levels: 1" "cfg['levels'] = 1"
run_with "Levels: 5" "cfg['levels'] = 5"
run_with "Levels: 11" "cfg['levels'] = 11"

# =====================================================================
# MEDIUM PRIORITY SWEEPS
# =====================================================================

# Low-rank factorization (0 = baseline/full rank)
run_with "Low-rank: 4" "cfg['low_rank'] = 4"
run_with "Low-rank: 16" "cfg['low_rank'] = 16"

# Lifting hidden multiplier
run_with "Lifting hidden mult: 2" "cfg['lifting_hidden_mult'] = 2"
run_with "Lifting hidden mult: 4" "cfg['lifting_hidden_mult'] = 4"

# Learning rate
run_with "LR: 0.005" "cfg['lr'] = 0.005"
run_with "LR: 0.02" "cfg['lr'] = 0.02"

# Block size (adjust levels to match)
run_with "Block size: 256, levels=8" "cfg['block_size'] = 256; cfg['levels'] = 8"
run_with "Block size: 1024, levels=10" "cfg['block_size'] = 1024; cfg['levels'] = 10"

# =====================================================================
# RESET to baseline after all runs
# =====================================================================

echo "=== Resetting config to baseline ==="
reset_config
git add .
git commit --no-edit -m "Reset config to baseline after sweep"
git pull --no-edit
git push

echo "=== All runs complete ==="
