# EXARCH Training Runs

## Summary

| Run | Sweep | C | Epochs | Folder | BPB (sliding) | Params | Notes |
|-----|-------|---|--------|--------|---------------|--------|-------|
| 1   | C with 1 epoch (exp=1) | 64  | 1 | [link](#run-1) | — | 11.42M | Pipeline test, mlp_exp=1 |
| 2   | C with 1 epoch (exp=1) | 128 | 1 | [link](#run-2) | — | — | Width scaling |
| 3   | C with 1 epoch (exp=1) | 256 | 1 | [link](#run-3) | — | — | Width scaling |
| 4   | C with 1 epoch (exp=1) | 512 | 1 | [link](#run-4) | — | — | Width scaling |
| 5   | C with 1 epoch (exp=1) | 1024 | 1 | [link](#run-5) | — | — | Width scaling |

---

## Run Details

### Run 1

**Status:** Running

**Folder:** `logs/wikitext-103_2026-04-02_11-53-11/` ([log](../logs/wikitext-103_2026-04-02_11-53-11/log.txt))

**Description:** End-to-end pipeline test + first width scaling point. C=64, mlp_expansion=1, 1 epoch.

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
    "mlp_expansion": 1,
    "mlp_layers": 2,
    "wavelet_mode": "lifting",
    "shared_lifting_weights": false,
    "lifting_linear_only": false,
    "lifting_init": "haar",
    "lifting_dropout": 0.0,
    "use_mixer_gate": true,
    "mixer_gate_activation": "silu",
    "semantic_feedback": true,
    "semantic_feedback_cross_window": true,
    "learned_residual": true,
    "skip_proj_out": false,
    "stochastic_depth_rate": 0.0,
    "dropout_embedding": 0.0,
    "dropout_projection": 0.0,
    "dropout_mixer": 0.0,
    "dropout_mlp": 0.0,
    "dropout_lm_head": 0.0,
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
- 14,614 steps/epoch, 14,614 total steps
- Warmup: 4,384 steps (30%)

**Results:**
- Val loss: —
- Sliding BPB: —
- Non-overlapping BPB: —
- Training time: —
- Training Peak VRAM: —
- Inference Peak VRAM: —

---

### Run 2

**Status:** Pending

**Description:** C=128, mlp_expansion=1, 1 epoch. Width scaling.

---

### Run 3

**Status:** Pending

**Description:** C=256, mlp_expansion=1, 1 epoch. Width scaling.

---

### Run 4

**Status:** Pending

**Description:** C=512, mlp_expansion=1, 1 epoch. Width scaling.

---

### Run 5

**Status:** Pending

**Description:** C=1024, mlp_expansion=1, 1 epoch. Width scaling.
