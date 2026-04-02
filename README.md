<p align="center">
  <img src="assets/exarch-header.svg" alt="EXARCH" width="90%"/>
</p>

<h3 align="center"><b>Explicit Attentionless Reasoning with Causal Harmonics</b></h3>

<br>

---

<br>

EXARCH is an attention-free language model that replaces attention with a wavelet-based spectral mixing architecture. Each block decomposes the input sequence via a learnable lifting wavelet scheme, applies a Fast Hadamard Transform followed by per-scale gated spectral mixing (SwiGLU), then reconstructs via the inverse wavelet transform. Stacked with standard MLPs and optional cross-layer semantic feedback, this produces a fully causal sequence model with no attention mechanism, no quadratic scaling, and no key/value cache. 

## Installation

Requires Python 3.10+ and PyTorch 2.8+.

```bash
pip install torch datasets tiktoken tqdm
```

Clone the repo:

```bash
git clone https://github.com/ramongougis/EXARCH.git
cd EXARCH
```

## Usage

### Training

The configuration lives in `config.json`. Edit it to set model dimensions, dataset, optimizer, and hardware options, then run:

```bash
python train.py
```

Key config options:

| Option | Default | Description |
|--------|---------|-------------|
| `C` | 1024 | Mixer working width (power of 2) |
| `layers` | 20 | Number of EXARCH blocks |
| `mlp_expansion` | 20 | MLP hidden dim multiplier |
| `levels` | 9 | Wavelet decomposition levels (~log2(block_size)) |
| `dataset` | wikitext-103 | HuggingFace dataset name |
| `optimizer` | Adagrad | Adagrad or AdamW |
| `amp_dtype` | fp16 | fp16 or bf16 |

Training logs, checkpoints, and configs are saved to `logs/<dataset>_<timestamp>/`. Results from all runs are tracked in [`notes/runs.md`](notes/runs.md).

### Generation

```bash
# Default generation without inference strategies enabled
python generate.py --checkpoint logs/<run_dir>/best_model.pt \
    --prompt "The history of" --num_tokens 512
```

Optional inference strategies:

```bash
# Use all strategies (recommended)
python generate.py --checkpoint best_model.pt --strategies

# Entropy-adaptive temperature
python generate.py --checkpoint best_model.pt --entropy_adaptive

# Lookahead reranking
python generate.py --checkpoint best_model.pt --lookahead_k 3 --lookahead_depth 5

# Best-of-N sampling
python generate.py --checkpoint best_model.pt --best_of_n 5

# Wavelet coherence monitoring
python generate.py --checkpoint best_model.pt --wavelet_coherence
```

## Architecture

```
Input tokens
    |
    v
Learned Embedding (C)
    |
    v
+----------------------------+
| ExarchBlock (x layers)     |
|                            |
|  LayerNorm                 |
|  Lifting Wavelet Decompose |
|  Fast Hadamard Transform   |
|  Gated Spectral Mixer      |  <-- per-scale SwiGLU mixing
|  Fast Hadamard (inverse)   |
|  Learned Scale Weights     |
|  Wavelet Reconstruct       |
|  Residual Connection       |
|  LayerNorm --> MLP          |
|  Residual Connection       |
+----------------------------+
    |
    v
LayerNorm --> LM Head --> logits
```

**Wavelet decomposition** uses a learnable lifting scheme (predict/update networks initialized to Haar wavelets) that decomposes the sequence into multi-scale coefficients, capturing both coarse structure and fine detail at each position.

**Spectral mixing** applies a Fast Hadamard Transform across channels, then mixes each wavelet scale independently through gated linear layers (SwiGLU by default), before inverting the Hadamard.

**Semantic feedback** optionally passes a causal running mean of hidden states between layers, providing cross-layer context without attention.

## Multinodal

EXARCH supports a product-of-experts mode where multiple independent model cells process the input in parallel with different feature subsets (feature bagging), then combine logits via averaging. Enable with `multinodal_enabled: true` in config.

## Results

See [`notes/runs.md`](notes/runs.md) for a full log of training runs, configs, and benchmark results.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
