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
    "untied_reconstruction": False,
    "multi_basis_lifting": False,
    "multi_basis_inits": ["haar", "random"],
    "cross_scale_gating": False,
    "per_scale_mixer_widths": None,
    "stable_parametrization": False,
    "stab_spectral_norm": False,
    "stab_ff_scaling": False,
    "stab_embed_scaling": False,
    "stab_proj_out_scaling": False,
    "stab_mixer_eps_scaling": False,
    "stab_lifting_level_scaling": False,
    "looped_blocks": False,
    "looped_blocks_count": 8,
    "wavelet_crawl": False,
    "wavelet_crawl_k": 3,
    "lifting_linear_only": False,
    "lifting_hidden_mult": 1,
    "lifting_init": "haar",
    "lifting_dropout": 0.0,
    "use_mixer_gate": True,
    "mixer_gate_activation": "silu",
    "mixer_depth": 1,
    "mixer_depth_stabilizers": False,
    "mixer_depth_residuals": False,
    "decompose_bypass": True,
    "decompose_bypass_cross_window": True,
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
# NEW BASELINE PROBE: levels=5, exp_param, lr=0.02, low_rank=4
# Tests whether this trio (proven wins) folded into the W&S config beats
# the previous 1-epoch baseline (BPB 1.1133, levels=9, lr=0.01, no exp_param,
# low_rank=0). Halves runtime per epoch via fewer wavelet levels.
# Probe runs first; if BPB beats 1.1133, this becomes the comparison
# point for all Part 3 ablations below.
# =====================================================================

NEW_BASELINE="cfg['layers'] = 2; cfg['C'] = 2048; cfg['mlp_expansion'] = 20; cfg['per_layer_embedding'] = True; cfg['levels'] = 5; cfg['lr'] = 0.01; cfg['low_rank'] = 4; cfg['per_scale_mixer_widths'] = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]"
# Note: per_scale_mixer_widths promoted into NEW_BASELINE after screening result
# (logs/wikitext-103_2026-04-17_14-23-49, BPB 1.1168, -23% epoch time). All
# subsequent Part 3/4 runs now include PSW. Reference BPB for comparisons
# shifts from 1.1173 → 1.1168.
# Note: PKM and FwPKM are intentionally OFF here. They get re-introduced for
# the final 5-epoch best-run candidate (with 2.0x+ dropout). Keeping them off
# during 1-epoch screening saves ~10-15% training time and ~150M params.
# Note: exp_parametrization + lr=0.02 NaN'd at L=2 (logs/.../2026-04-17_00-27-55,
# step 4000 LR=1.82e-02). Reverted to lr=0.01 without exp_param. Re-test the
# lr=0.02 + exp_param combo later when stable_parametrization features are
# proven, since spectral_norm should provide the additional signal damping needed.

# run_with "New baseline probe (levels=5, lr=0.01, low_rank=4)" "$NEW_BASELINE"

# =====================================================================
# BOOLEAN ABLATIONS PART 3: C=2048, L=2, EP=1 WIDE & SHALLOW MODEL
# Each ablation = NEW_BASELINE + one feature flag. Compares against the
# baseline probe above.
# =====================================================================

# run_with "W&S: untied reconstruction" "$NEW_BASELINE; cfg['untied_reconstruction'] = True"
# run_with "W&S: cross-scale gating (routing)" "$NEW_BASELINE; cfg['cross_scale_gating'] = True"
# run_with "W&S: multi-basis lifting (haar+random)" "$NEW_BASELINE; cfg['multi_basis_lifting'] = True; cfg['multi_basis_inits'] = ['haar', 'random']"
# run_with "W&S: per-scale mixer widths (1,1,1,.5,.5,.5)" "$NEW_BASELINE; cfg['per_scale_mixer_widths'] = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]"
# ^ Now part of NEW_BASELINE; completed 2026-04-17_14-23-49 (BPB 1.1168, -23% time).

# --- Feedback mechanisms (plans/feedback_mechanisms.md) ---
# run_with "W&S: looped blocks (K=8 shared)" "$NEW_BASELINE; cfg['looped_blocks'] = True; cfg['looped_blocks_count'] = 8"
# All three feedback mechanisms removed:
#   - Looped blocks: worked (-0.0039 BPB) but 3x training time; better to train more epochs.
#   - Iterative refinement: NaN'd twice, weaker variant of looped blocks.
#   - Cross-time feedback: random-batch coupling = noise, train/inference asymmetry.

# --- Wavelet crawl (plans/wavelet_crawl.md) ---
# run_with "W&S: wavelet crawl (K=3)" "$NEW_BASELINE; cfg['wavelet_crawl'] = True; cfg['wavelet_crawl_k'] = 3"
# run_with "W&S: wavelet crawl (K=5)" "$NEW_BASELINE; cfg['wavelet_crawl'] = True; cfg['wavelet_crawl_k'] = 5"

# --- Lifting efficiency probes (param/compute savings; quality impact unknown) ---
run_with "W&S: shared_lifting_weights" "$NEW_BASELINE; cfg['shared_lifting_weights'] = True"
run_with "W&S: lifting_linear_only" "$NEW_BASELINE; cfg['lifting_linear_only'] = True"
run_with "W&S: shared_lifting + linear_only" "$NEW_BASELINE; cfg['shared_lifting_weights'] = True; cfg['lifting_linear_only'] = True"

# =====================================================================
# BOOLEAN ABLATIONS PART 4: STABLE PARAMETRIZATION (vs KNOWN NaN CONFIGS)
# Tests whether stable_parametrization rescues configs that previously NaN'd.
# Each row pairs a known-unstable config with stable_parametrization=True.
# Reference NaN runs:
#   - mixer_depth=5 at L=20:           Run 39, NaN step 3600 (LR=0.008)
#   - lifting_hidden_mult=2 at L=20:   logs/...2026-04-16_17-23-50, NaN step 2500
#   - C=2048, lr=0.02 at L=20, MLP=1:  Run 63, NaN step 700 (LR=0.003)
# =====================================================================

run_with "Stab: vs mixer_depth=5 NaN (was Run 39)" "cfg['mixer_depth'] = 5; cfg['stable_parametrization'] = True"
run_with "Stab: vs lifting_hidden_mult=2 NaN" "cfg['lifting_hidden_mult'] = 2; cfg['stable_parametrization'] = True"
run_with "Stab: vs C=2048 lr=0.02 NaN (was Run 63)" "cfg['C'] = 2048; cfg['lr'] = 0.02; cfg['stable_parametrization'] = True"
run_with "Stab: vs new-baseline+lr=0.02+exp_param NaN" "$NEW_BASELINE; cfg['lr'] = 0.02; cfg['exp_parametrization'] = True; cfg['stable_parametrization'] = True"

# --- Compatibility tests: stab sub-features at the stable W&S baseline ---
# Verifies each sub-feature runs without breakage at a known-stable config and
# measures its standalone effect on BPB (positive delta = useful, ~0 = neutral,
# negative = harmful at this config but may still rescue unstable ones).

run_with "W&S+Stab: master (all 6)" "$NEW_BASELINE; cfg['stable_parametrization'] = True"
run_with "W&S+Stab: spectral_norm" "$NEW_BASELINE; cfg['stab_spectral_norm'] = True"
run_with "W&S+Stab: ff_scaling" "$NEW_BASELINE; cfg['stab_ff_scaling'] = True"
run_with "W&S+Stab: embed_scaling" "$NEW_BASELINE; cfg['stab_embed_scaling'] = True"
run_with "W&S+Stab: proj_out_scaling" "$NEW_BASELINE; cfg['stab_proj_out_scaling'] = True"
run_with "W&S+Stab: mixer_eps_scaling" "$NEW_BASELINE; cfg['stab_mixer_eps_scaling'] = True"
run_with "W&S+Stab: lifting_level_scaling" "$NEW_BASELINE; cfg['stab_lifting_level_scaling'] = True"

# =====================================================================
# BLOCK SIZE — at NEW_BASELINE (levels=5 supports any block_size >= 32)
# =====================================================================

run_with "Block size: 256" "$NEW_BASELINE; cfg['block_size'] = 256"
run_with "Block size: 1024" "$NEW_BASELINE; cfg['block_size'] = 1024"

# =====================================================================
# GRAD ACCUM — at NEW_BASELINE
# =====================================================================

run_with "Grad accum: 1 (batch=8)" "$NEW_BASELINE; cfg['grad_accum'] = 1"
run_with "Grad accum: 4 (batch=32)" "$NEW_BASELINE; cfg['grad_accum'] = 4"

# =====================================================================
# WARMUP FRACTION — at NEW_BASELINE
# =====================================================================

run_with "Warmup fraction: 0.1" "$NEW_BASELINE; cfg['warmup_fraction'] = 0.1"
run_with "Warmup fraction: 0.5" "$NEW_BASELINE; cfg['warmup_fraction'] = 0.5"

# =====================================================================
# GRAD CLIP — at NEW_BASELINE
# =====================================================================

run_with "Grad clip: 0.5" "$NEW_BASELINE; cfg['grad_clip'] = 0.5"
run_with "Grad clip: 2.0" "$NEW_BASELINE; cfg['grad_clip'] = 2.0"

# =====================================================================
# C=4096 WIDTH SCALING — at NEW_BASELINE (lr=0.02 + exp_param baked in)
# Drops the prior lr=0.01 controls since lr=0.02+exp_param is now standard.
# =====================================================================

run_with "C=4096: L=1, MLP=10" "$NEW_BASELINE; cfg['layers'] = 1; cfg['C'] = 4096; cfg['mlp_expansion'] = 10"
run_with "C=4096: L=2, MLP=10, MBS=4/GA=4" "$NEW_BASELINE; cfg['layers'] = 2; cfg['C'] = 4096; cfg['mlp_expansion'] = 10; cfg['micro_batch_size'] = 4; cfg['grad_accum'] = 4"

# =====================================================================
# REDUCED LEVELS AT SCALE — at NEW_BASELINE, 5 epochs, 2.0x dropout
# Tests whether levels<5 still wins at L=2/C=2048 now that NEW_BASELINE
# already has levels=5 (from the L=20 finding). The levels=5 entry is
# the canonical 5-epoch best-run candidate.
# =====================================================================

DROPOUT_2X="cfg['dropout_embedding'] = 0.2; cfg['dropout_projection'] = 0.1; cfg['dropout_mixer'] = 0.1; cfg['dropout_mlp'] = 0.1; cfg['dropout_lm_head'] = 0.24"

run_with "5ep+2.0x dropout: levels=1" "$NEW_BASELINE; cfg['epochs'] = 5; cfg['levels'] = 1; cfg['per_scale_mixer_widths'] = None; $DROPOUT_2X"
run_with "5ep+2.0x dropout: levels=2" "$NEW_BASELINE; cfg['epochs'] = 5; cfg['levels'] = 2; cfg['per_scale_mixer_widths'] = None; $DROPOUT_2X"
run_with "5ep+2.0x dropout: levels=5 (5-epoch best candidate)" "$NEW_BASELINE; cfg['epochs'] = 5; $DROPOUT_2X"

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
