# EXARCH Training Runs

## Summary

| Run | Sweep | C | Epochs | Folder | BPB (sliding) | Params | Notes |
|-----|-------|---|--------|--------|---------------|--------|-------|
| 1   | C with 1 epoch | 64  | 1 | [link](#run-1) | — | 11.55M | End-to-end pipeline test |
| 2   | C with 1 epoch | 128 | 1 | [link](#run-2) | — | — | Width scaling |
| 3   | C with 1 epoch | 256 | 1 | [link](#run-3) | — | — | Width scaling |
| 4   | C with 1 epoch | 512 | 1 | [link](#run-4) | — | — | Width scaling |
| 5   | C with 1 epoch | 1024 | 1 | [link](#run-5) | — | — | Width scaling |

---

## Run Details

### Run 1

**Status:** Running

**Folder:** `logs/wikitext-103_2026-04-02_11-32-52/` ([log](../logs/wikitext-103_2026-04-02_11-32-52/log.txt))

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

**Description:** C=128, 1 epoch. Width scaling ablation.

---

### Run 3

**Status:** Pending

**Description:** C=256, 1 epoch. Width scaling ablation.

---

### Run 4

**Status:** Pending

**Description:** C=512, 1 epoch. Width scaling ablation.

---

### Run 5

**Status:** Pending

**Description:** C=1024, 1 epoch. Width scaling ablation.
