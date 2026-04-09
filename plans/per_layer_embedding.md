# Per-Layer Embedding Residual (PLE)

## Summary

Reintroduce the original token embedding as a learned per-channel residual at each
ExarchBlock, before LayerNorm and wavelet decomposition. Each layer gets a learned
gamma vector (C,) that controls which embedding dimensions to reintroduce, allowing
deeper layers to access token-specific information that may have been transformed
away in the residual stream.

Inspired by Gemma 4's Per-Layer Embeddings, but simplified: no separate low-dim
embedding table, no per-layer projection matrices. Just the original embedding
modulated by a learned per-channel gate.

## Motivation

In the current architecture, each token gets a single embedding at input. By layer 20,
the residual stream has been transformed through 20 rounds of wavelet mixing and MLP.
The original token identity signal is heavily diluted. Some layers may need to
"re-read" specific aspects of the token identity (semantic, syntactic, positional)
that earlier layers have already consumed and transformed.

PLE gives each layer a direct, cheap channel back to the original embedding.

## Mathematical formulation

```
For each ExarchBlock at layer l:
  h_l' = h_l + gamma_l . embed[token_ids]
  (then proceed to ln1, wavelet decompose, etc.)
```

Where:
- gamma_l = Parameter(C,), initialized to zeros
- embed[token_ids] = original token embedding lookup (shared reference, not copied)
- . = elementwise multiply (per-channel gating)

At init (gamma=0): behavior is identical to today. No embedding residual.
During training: each layer learns which embedding channels to reintroduce.

## Config

```json
"per_layer_embedding": false
```

false = today's behavior (no change). true = add gamma per block.

## Parameter cost

Per layer: C parameters (one gamma vector)
Total: L * C = 20 * 512 = 10,240 parameters (~0.01M)

Negligible. Less than a single bias vector.

## Implementation

### Files to modify

1. **config.json** - add `per_layer_embedding` key (default false)
2. **model.py** - ExarchBlock and ExarchLM changes

### ExarchBlock changes

Add to `__init__`:
```python
self.per_layer_embedding = per_layer_embedding
if per_layer_embedding:
    self.embedding_residual_gamma = nn.Parameter(torch.zeros(C, device=device, dtype=dtype))
```

Add to `forward`, before `h = self.ln1(x)`:
```python
if self.per_layer_embedding:
    x = x + self.embedding_residual_gamma * token_embeddings
```

Where `token_embeddings` is passed into the forward call.

### ExarchLM changes

The token embedding lookup happens in ExarchLM.forward(). Currently the embedding
result flows into the layer stack and is never referenced again. With PLE, each
block needs access to it.

In ExarchLM.forward():
```python
tok_emb = self.token_embedding(idx)
# ... existing code ...
for layer in self.layers:
    x, semantic_state = layer(x, semantic_state, token_embeddings=tok_emb if self.per_layer_embedding else None)
```

ExarchBlock.forward() signature adds `token_embeddings=None`.

### Checkpoint compatibility

- per_layer_embedding=false: identical to today, no new params, full compat
- per_layer_embedding=true: adds gamma params that won't exist in old checkpoints;
  strict=False handles this, and zero init means the model starts as if PLE is off

### What gamma learns

Each gamma channel controls one embedding dimension at one layer:
- gamma_l[i] > 0: layer l reintroduces embedding dim i (amplifies)
- gamma_l[i] < 0: layer l subtracts embedding dim i (suppresses)
- gamma_l[i] ~ 0: layer l ignores embedding dim i (default)

Interesting diagnostic: after training, visualize gamma across layers to see
which embedding dimensions each layer "reaches back" for. This could reveal
what information the wavelet mixer consumes vs. what it needs refreshed.

## Ablation plan

| Run | per_layer_embedding | Folder | BPB (sliding) | Params | Delta | Notes |
|-----|---------------------|--------|---------------|--------|-------|-------|
|     | false               |        |               |        |       | Baseline |
|     | true                |        |               |        |       | +0.01M params |

Single ablation at C=512, L=20, epochs=1, mlp_expansion=1. If positive,
test interaction with mixer_depth=2.
