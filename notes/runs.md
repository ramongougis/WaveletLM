# EXARCH Training Runs

## Summary

| Run | Sweep | C | Epochs | Folder | BPB (sliding) | Params | Notes |
|-----|-------|---|--------|--------|---------------|--------|-------|
| 1   | C with 1 epoch | 64 | 1 | [link](#run-1) | — | — | End-to-end pipeline test |

---

## Run Details

### Run 1

**Status:** Running

**Description:** End-to-end pipeline test. C=64, 1 epoch. Validates benchmarks, VRAM reporting, and generation with strategies.

<details>
<summary>Config</summary>

```json
{
    "dataset": "wikitext-103",
    "epochs": 1,
    "micro_batch_size": 8,
    "grad_accum": 2,
    "block_size": 512,
    "C": 64,
    "layers": 20,
    "levels": 9,
    "low_rank": 0,
    "mlp_expansion": 20,
    "mlp_layers": 2,
    "wavelet_mode": "lifting",
    "shared_lifting_weights": true,
    "lifting_linear_only": true,
    "lifting_init": "haar",
    "lifting_dropout": 0.0,
    "use_mixer_gate": true,
    "mixer_gate_activation": "silu",
    "semantic_feedback": true,
    "semantic_feedback_cross_window": true,
    "learned_residual": true,
    "skip_proj_out": true,
    "stochastic_depth_rate": 0.0,
    "dropout_embedding": 0.1,
    "dropout_projection": 0.05,
    "dropout_mixer": 0.05,
    "dropout_mlp": 0.05,
    "dropout_lm_head": 0.12,
    "optimizer": "Adagrad",
    "optimizer_eps": 2e-13,
    "lr": 2e-2,
    "min_lr": 2e-4,
    "warmup_fraction": 0.3,
    "grad_clip": 1.0,
    "tie_embedding_to_lm_head": false,
    "gradient_checkpointing": false,
    "use_amp": true,
    "amp_dtype": "fp16",
    "allow_tf32": true
}
```

</details>

**Schedule:**
- Steps/epoch: ~16,005 (MBS=8, GA=2)
- Warmup: ~4,802 steps (30%)

**Results:**
- Val loss: —
- Sliding BPB: —
- Non-overlapping BPB: —
- Training time: —
- Training Peak VRAM: —
- Inference Peak VRAM: —
