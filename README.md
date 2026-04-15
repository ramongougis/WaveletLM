<p align="center">
  <img src="assets/exarch-header.svg" alt="EXARCH" width="90%"/>
</p>

<h3 align="center"><b>Exclusively Attentionless Reasoning with Causal Harmonics</b></h3>

<br>

---

<br>

EXARCH is an attention-free language model that replaces attention with a wavelet-based spectral mixing architecture. Each block processes the input sequence using a learned lifting wavelet scheme, Fast Hadamard Transform, per-scale gated spectral mixing (SwiGLU), and inverse wavelet transform. Combined with expanded MLPs and cross-layer semantic feedback, this produces a fully causal sequence language model with no attention mechanism, no quadratic scaling, and no key/value cache. 

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

Training logs, checkpoints, and configs are saved to `logs/<dataset>_<timestamp>/`. Results from all runs are tracked in [`runs.md`](runs.md).

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

# Clean WikiText-103 spacing artifacts (remove space before periods)
python generate.py --checkpoint best_model.pt --clean_spacing

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
|  LayerNorm --> MLP         |
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

### WikiText-103 Perplexity Comparison

| Model | Type | Trained on | Params | PPL |
|-------|------|-----------|--------|-----|
| Mamba-2 | SSM | The Pile (~800GB) | 2.7B | ~13[^2] |
| Mamba | SSM | The Pile (~800GB) | 1.4B | ~17[^3] |
| Transformer-XL* | Transformer + recurrence | WikiText-103 | 257M | ~18[^4] |
| GPT-2 XL | Transformer | WebText (~40GB) | 1.5B | ~18[^5] |
| RWKV-6 | Linear RNN | The Pile (~800GB) | 1.5B | ~18[^6] |
| xLSTM | Extended LSTM | SlimPajama (~627GB) | 1.3B | ~18[^7] |
| Hyena | Long convolution | The Pile (~800GB) | 1.4B | ~18[^8] |
| GPT-2 Large | Transformer | WebText (~40GB) | 774M | ~19[^5] |
| S4* | SSM | WikiText-103 | 130M | ~20[^9] |
| GPT-2 Medium | Transformer | WebText (~40GB) | 355M | ~22[^5] |
| RWKV-4 | Linear RNN | The Pile (~800GB) | 430M | ~22[^10] |
| **EXARCH**\* | **Wavelet mixer** | **WikiText-103 (~0.5GB)** | **1.18B** | **~24**[^1] |
| GPT-2 | Transformer | WebText (~40GB) | 124M | ~29[^5] |

\* Trained and evaluated on the same dataset (direct comparison to EXARCH).

EXARCH achieves this with only 2 layers (L=2, C=2048), no attention, no recurrence, and no KV cache. Validation loss was still improving at epoch 5, indicating further training will improve results. Comparison numbers are approximate and sourced from respective papers; see references below.

[^1]: L=2, C=2048, MLP=20, PLE, PKM+FwPKM-16384, 5 epochs, 2.0x dropout. BPB=1.0247. See [training log](logs/wikitext-103_2026-04-14_09-07-12/log.txt).

See [`runs.md`](runs.md) for a full log of training runs, configs, and benchmark results.

## References

[^2]: Dao & Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." arXiv:2405.21060, 2024.
[^3]: Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.
[^4]: Dai et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." arXiv:1901.02860, 2019.
[^5]: Radford et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019.
[^6]: Peng et al. "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence." arXiv:2404.05892, 2024.
[^7]: Beck et al. "xLSTM: Extended Long Short-Term Memory." arXiv:2405.04517, 2024.
[^8]: Poli et al. "Hyena Hierarchy: Towards Larger Convolutional Language Models." arXiv:2302.10866, 2023.
[^9]: Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces." arXiv:2111.00396, 2021.
[^10]: Peng et al. "RWKV: Reinventing RNNs for the Transformer Era." arXiv:2305.13048, 2023.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
