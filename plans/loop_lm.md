# LoopLM: Iterative Block Reuse for EXARCH

## Summary

Apply the same stack of ExarchBlock layers T times sequentially, producing logits at
each iteration. Training loss is a weighted sum across all iterations. No additional
parameters are added — the same weights are reused, adding computation (depth) without
increasing model size.

Inspired by LoopLM (arxiv:2510.25741). Adapted for EXARCH's wavelet-based architecture.

## Motivation

- C=2048 L=1 achieves BPB 1.1431 (617M params) — approaching L=20 at C=512 (1.1751)
- Adding more layers adds params linearly; looping adds compute without params
- Each loop re-applies wavelet decompose → spectral mix → reconstruct → MLP
- This gives iterative multi-scale refinement: each pass re-decomposes the mixed signal,
  capturing higher-order scale interactions that a single pass misses
- Semantic feedback naturally carries state between loop iterations

## Architecture

```
For input x:
  h = embedding(x)
  
  For t = 1, ..., T:
    h, semantic_state = layer_stack(h, semantic_state)
    logits_t = lm_head(final_ln(h))       # produce logits at each iteration
  
  Final prediction: logits_T (last iteration)
  Training loss: weighted sum of losses at all iterations
```

Where `layer_stack` is the same L ExarchBlock layers with shared weights.

## Key design decisions

### 1. Loss at each iteration

Each iteration t produces independent logits and a cross-entropy loss:
```
L_total = sum(w_t * L_t for t in 1..T)
```

Simplest weighting: uniform `w_t = 1/T`. This encourages every iteration to produce
useful predictions, not just the last one. The model learns to progressively refine.

Alternative: increasing weights `w_t = t/sum(1..T)`, emphasizing later iterations.

### 2. No exit gates (simplified version)

LoopLM uses learned exit gates for adaptive compute. For the initial EXARCH implementation,
use fixed T iterations — simpler, deterministic, and sufficient to test whether looping
helps. Exit gates can be added later if looping proves beneficial.

### 3. Semantic feedback across iterations

EXARCH's semantic feedback already passes state between layers. With looping, this
naturally extends: iteration t's semantic state feeds into iteration t+1's first layer.
This is free — no code changes needed, the existing `prev_state` mechanism handles it.

### 4. LayerNorm between iterations

Apply the existing `final_ln` before producing logits at each iteration, but pass the
pre-LN hidden state `h` (not the post-LN version) to the next iteration. This prevents
the LN from collapsing the representation between iterations.

## Config

```json
"loop_iterations": 1
```

1 = today's behavior (no looping). 2+ = apply the layer stack T times.

## Parameter cost

Zero. Same weights reused. VRAM increases only from activation memory for
gradient computation through T iterations.

At T=4 with gradient checkpointing, activation memory increase is modest since
each iteration's intermediate activations can be recomputed.

## VRAM estimate

Current L=1, C=2048, MLP=20: 14,109 MiB at T=1.
- T=2: ~16,000 MiB (extra activation storage for backprop through 2 passes)
- T=4: ~20,000 MiB (with gradient checkpointing: ~17,000 MiB)
All fit comfortably in 49 GB.

## Implementation

### Files to modify

1. **config.json** — add `loop_iterations` key (default 1)
2. **model.py** — ExarchLM.forward() changes
3. **train.py** — loss computation changes (sum over iterations)

### model.py changes

In ExarchLM.forward():

```python
def forward(self, idx, targets=None):
    B, T_seq = idx.shape
    loop_iterations = self.config.get('loop_iterations', 1)
    
    tok_emb = self.token_embedding(idx)
    x = self.dropout_emb(tok_emb)
    ple = tok_emb if self.per_layer_embedding else None
    
    # Initialize semantic state
    if self.semantic_feedback_cross_window and self._persistent_semantic_state is not None:
        current_state = self._persistent_semantic_state.unsqueeze(1).expand(-1, T_seq, -1)
    else:
        current_state = None
    
    all_logits = []
    
    for loop in range(loop_iterations):
        # Apply the same layer stack
        for layer_idx, layer in enumerate(self.layers):
            if self.training and self.stochastic_depth_rate > 0:
                if random.random() < self._drop_probs[layer_idx]:
                    continue
            x, current_state = layer(x, current_state, token_embeddings=ple)
        
        # Produce logits at this iteration
        x_ln = self.final_ln(x)
        logits_t = self.lm_head(self.dropout_lm(x_ln))
        all_logits.append(logits_t)
        # Note: pass pre-LN x to next iteration, not x_ln
    
    # Update persistent state
    if self.semantic_feedback_cross_window and current_state is not None:
        self._persistent_semantic_state = current_state[:, -1, :].detach()
        self._persistent_token_count += T_seq
    
    # Final logits = last iteration
    logits = all_logits[-1]
    
    loss = None
    if targets is not None:
        if loop_iterations == 1:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1).long())
        else:
            # Weighted sum of losses across iterations
            total_loss = 0
            for t, lg in enumerate(all_logits):
                t_loss = F.cross_entropy(lg.view(-1, lg.size(-1)), targets.view(-1).long())
                total_loss += t_loss  # uniform weighting
            loss = total_loss / loop_iterations
    
    return logits, loss
```

### train.py changes

None needed if loss is computed inside model.forward(). The training loop already
uses `model(x, targets)` and gets back `(logits, loss)`.

### generate.py changes

During generation, use only the final iteration's logits (already the case since
`logits = all_logits[-1]`). Optionally allow configuring inference iterations
separately from training iterations.

## Ablation plan

All at L=1, C=2048, MLP=20, MD=1 (best L=1 config at 1.1431 BPB):

| Run | loop_iterations | Params | Est VRAM | Notes |
|-----|-----------------|--------|----------|-------|
|     | 1               | 617M   | 14 GB    | Baseline (no looping) |
|     | 2               | 617M   | ~16 GB   | First loop test |
|     | 4               | 617M   | ~20 GB   | LoopLM default |
|     | 8               | 617M   | ~28 GB   | Stress test |

If looping helps, also test:
- L=1, C=2048, MLP=20, T=4 with gradient_checkpointing=true
- L=2, C=2048, MLP=20, T=2 (loops + layers)
- L=1, C=2048, MLP=1, T=4 (can looping substitute for MLP width?)

## Why this could be transformative for EXARCH

1. **Wavelet refinement**: Each loop re-decomposes the signal. Pass 1 captures
   coarse structure, pass 2 captures fine details the first pass missed, pass 3
   captures interactions between the corrections, etc.

2. **No KV cache advantage multiplied**: EXARCH has no KV cache, so looping adds
   pure compute without the memory scaling that attention-based loops would need.

3. **Deployment story**: A 617M parameter model that computes like a 2.5B model
   (4 loops) but stores/transfers as 617M. PTQ to INT4 = ~150 MB on disk.

4. **Training efficiency**: Each training step does T forward+backward passes
   through the same weights, effectively giving T gradient updates per step.
   This is similar to accumulating gradients from T different "views" of the
   same input at different processing depths.
