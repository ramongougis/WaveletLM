# MLP expansion sweep + Memory (PKM/FwPKM) sweep
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
    "quantize_embedding_bits": 8
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
    git add .
    git commit --no-edit -m "$NAME"
    git pull --no-edit
    git push
}

# =====================================================================
# MLP EXPANSION SWEEP (epochs=1, all defaults)
# =====================================================================

run_with "MLP expansion=2" "cfg['mlp_expansion'] = 2"
run_with "MLP expansion=10" "cfg['mlp_expansion'] = 10"
run_with "MLP expansion=20" "cfg['mlp_expansion'] = 20"
run_with "MLP expansion=50" "cfg['mlp_expansion'] = 50"

# =====================================================================
# MEMORY SWEEP (epochs=1, mlp_expansion=1, all defaults)
# =====================================================================

run_with "Memory: PKM only (529 keys)" "cfg['pkm_enabled'] = True"
run_with "Memory: PKM large (16384 keys)" "cfg['pkm_enabled'] = True; cfg['pkm_num_keys'] = 16384"
run_with "Memory: FwPKM only (529 keys)" "cfg['fwpkm_enabled'] = True"
run_with "Memory: PKM+FwPKM (529 keys)" "cfg['pkm_enabled'] = True; cfg['fwpkm_enabled'] = True"
run_with "Memory: PKM+FwPKM large (16384 keys)" "cfg['pkm_enabled'] = True; cfg['fwpkm_enabled'] = True; cfg['pkm_num_keys'] = 16384; cfg['fwpkm_num_keys'] = 16384"
run_with "Memory: MLP off, wavelet only" "cfg['mlp_expansion'] = 0"
run_with "Memory: MLP off, PKM only (529 keys)" "cfg['mlp_expansion'] = 0; cfg['pkm_enabled'] = True"

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
