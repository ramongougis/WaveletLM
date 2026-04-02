# EXARCH Training Runs

## Summary

| Run | Folder | BPB (sliding) | Params | Notes |
|-----|--------|---------------|--------|-------|
| 1   | [wikitext-103_YYYY-MM-DD_HH-MM-SS](#run-1) | — | — | Baseline: default config, C=512/20L, Adagrad |

---

## Run Details

### Run 1

**Status:** Pending

**Description:** Baseline run with default config. Learned embedding, GPT-2 tokenizer, WikiText-103.

<details>
<summary>Config</summary>

```json
{
    "dataset": "wikitext-103",
    "epochs": 10,
    "micro_batch_size": 8,
    "grad_accum": 2,
    "block_size": 512,
    "C": 512,
    "C_embed": 512,
    "layers": 20,
    "levels": 9,
    "low_rank": 0,
    "mlp_expansion": 10,
    "mlp_layers": 2,
    "wavelet_mode": "lifting",
    "shared_lifting_weights": true,
    "lifting_linear_only": true,
    "lifting_init": "haar",
    "use_mixer_gate": true,
    "mixer_gate_activation": "silu",
    "semantic_feedback": true,
    "learned_residual": true,
    "skip_proj_out": true,
    "stochastic_depth_rate": 0.0,
    "optimizer": "Adagrad",
    "lr": 2e-2,
    "min_lr": 2e-4,
    "warmup_fraction": 0.3,
    "use_amp": true,
    "amp_dtype": "fp16"
}
```

</details>

**Results:**
- Val loss: —
- Sliding BPB: —
- Non-overlapping BPB: —
- Training time: —
- Peak VRAM: —

**Observations:**

—
