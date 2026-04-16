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
    "mixer_depth_residuals": False,
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
    "per_layer_embedding": False,
    "loop_iterations": 1
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
        python generate.py --checkpoint "$LATEST_CKPT" --strategies
    fi
    git add .
    git commit --no-edit -m "$NAME"
    git pull --no-edit
    git push
}

# =====================================================================
# LEVELS SWEEP (epochs=1, mlp_expansion=1, C=512, layers=20, block_size=512)
# =====================================================================

# levels=9 is baseline (default = log2(block_size=512)), no need to rerun
run_with "Levels: 2" "cfg['levels'] = 2"
run_with "Levels: 3" "cfg['levels'] = 3"
run_with "Levels: 4" "cfg['levels'] = 4"
run_with "Levels: 6" "cfg['levels'] = 6"

# =====================================================================
# LOW-RANK FACTORIZATION
# =====================================================================

# Low-rank factorization (0 = baseline/full rank)
run_with "Low-rank: 4" "cfg['low_rank'] = 4"
run_with "Low-rank: 16" "cfg['low_rank'] = 16"

# =====================================================================
# LIFTING HIDDEN MULT
# =====================================================================

run_with "Lifting hidden mult: 2" "cfg['lifting_hidden_mult'] = 2"
run_with "Lifting hidden mult: 4" "cfg['lifting_hidden_mult'] = 4"

# =====================================================================
# BLOCK SIZE
# =====================================================================

# Block size (adjust levels to match)
run_with "Block size: 256, levels=8" "cfg['block_size'] = 256; cfg['levels'] = 8"
run_with "Block size: 1024, levels=10" "cfg['block_size'] = 1024; cfg['levels'] = 10"

# =====================================================================
# GRAD ACCUM
# =====================================================================

run_with "Grad accum: 1 (batch=8)" "cfg['grad_accum'] = 1"
run_with "Grad accum: 4 (batch=32)" "cfg['grad_accum'] = 4"

# =====================================================================
# WARMUP FRACTION
# =====================================================================

run_with "Warmup fraction: 0.1" "cfg['warmup_fraction'] = 0.1"
run_with "Warmup fraction: 0.5" "cfg['warmup_fraction'] = 0.5"

# =====================================================================
# GRAD CLIP
# =====================================================================

run_with "Grad clip: 0.5" "cfg['grad_clip'] = 0.5"
run_with "Grad clip: 2.0" "cfg['grad_clip'] = 2.0"

# =====================================================================
# C=4096 WIDTH SCALING
# =====================================================================

run_with "L=1, C=4096, MLP=10, lr=0.01" "cfg['layers'] = 1; cfg['C'] = 4096; cfg['mlp_expansion'] = 10; cfg['per_layer_embedding'] = True; cfg['pkm_enabled'] = True; cfg['pkm_num_keys'] = 16384; cfg['fwpkm_enabled'] = True; cfg['fwpkm_num_keys'] = 16384"
run_with "L=2, C=4096, MLP=10, lr=0.01, MBS=4/GA=4" "cfg['layers'] = 2; cfg['C'] = 4096; cfg['mlp_expansion'] = 10; cfg['per_layer_embedding'] = True; cfg['pkm_enabled'] = True; cfg['pkm_num_keys'] = 16384; cfg['fwpkm_enabled'] = True; cfg['fwpkm_num_keys'] = 16384; cfg['micro_batch_size'] = 4; cfg['grad_accum'] = 4"
run_with "L=1, C=4096, MLP=10, lr=0.02, exp_param" "cfg['layers'] = 1; cfg['C'] = 4096; cfg['mlp_expansion'] = 10; cfg['lr'] = 0.02; cfg['exp_parametrization'] = True; cfg['per_layer_embedding'] = True; cfg['pkm_enabled'] = True; cfg['pkm_num_keys'] = 16384; cfg['fwpkm_enabled'] = True; cfg['fwpkm_num_keys'] = 16384"
run_with "L=2, C=4096, MLP=10, lr=0.02, exp_param, MBS=4/GA=4" "cfg['layers'] = 2; cfg['C'] = 4096; cfg['mlp_expansion'] = 10; cfg['lr'] = 0.02; cfg['exp_parametrization'] = True; cfg['per_layer_embedding'] = True; cfg['pkm_enabled'] = True; cfg['pkm_num_keys'] = 16384; cfg['fwpkm_enabled'] = True; cfg['fwpkm_num_keys'] = 16384; cfg['micro_batch_size'] = 4; cfg['grad_accum'] = 4"


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
