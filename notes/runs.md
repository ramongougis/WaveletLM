# EXARCH Training Runs

## Summary

| Run | Folder | BPB (sliding) | Params | Notes |
|-----|--------|---------------|--------|-------|
| 1   | [wikitext-103_2026-04-02_10-23-13](#run-1) | — | 1,401.80M | Baseline: C=1024/20L, mlp_exp=20, learned emb, Adagrad, fp16 |

---

## Run Details

### Run 1

**Status:** Running

**Description:** First run in new repo. Learned embedding, GPT-2 tokenizer, WikiText-103. C=1024 with mlp_expansion=20 on 4090 (49GB).

<details>
<summary>Config</summary>

```json
{
    "dataset": "wikitext-103",
    "epochs": 10,
    "micro_batch_size": 8,
    "grad_accum": 2,
    "block_size": 512,
    "C": 1024,
    "C_embed": 1024,
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
- 14,614 steps/epoch, 146,140 total steps
- Warmup: 43,842 steps (30%)
- Est. training time: ~46h at 1.14s/it
- VRAM: 43,820 / 49,140 MiB

**Results:**
- Val loss: —
- Sliding BPB: —
- Non-overlapping BPB: —
- Training time: —
- Peak VRAM: 43,820 MiB

**Observations:**

- Loss dropping normally from 9.95 at step 22
- fp16 stable so far
