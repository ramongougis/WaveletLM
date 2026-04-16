# Experimental: Bidirectional Training for EXARCH

## Summary

Train the model on both forward (left-to-right) and backward (right-to-left) sequences,
doubling the effective training signal per epoch while maintaining directional awareness.

## Status: Post-release exploration

## Core idea

- With 50% probability, reverse the input sequence before feeding to the model
- The causal prediction task stays the same: predict the "next" token
- On reversed sequences, "next" is actually the previous token in original order
- The model sees both directions across an epoch, learning bidirectional patterns

## EXARCH-specific advantage

The lifting wavelet predict step compares even[i] with odd[i - dilation], always
looking backward in the sequence. On a reversed sequence, this effectively looks
forward in the original order. Bidirectional training gives EXARCH both causal
directions through the wavelet decomposition without architectural changes.

## Key challenge: directional awareness

The model must know which direction it's predicting in. Without this, forward and
backward gradients would conflict. Options:

1. **Direction embedding**: a learned vector added to all positions indicating
   forward vs backward. Cheapest approach — one parameter vector.
2. **Direction token**: prepend a special token indicating direction.
3. **Implicit**: rely on decompose bypass / cross-window state to carry
   directional information. Risky — may not be sufficient.

## Implementation sketch

In train.py, during batch preparation:
```python
if config.get('bidirectional_training', False):
    if random.random() < 0.5:
        x = x.flip(1)       # reverse sequence
        y = y.flip(1)       # reverse targets
        # Add direction signal if using direction embedding
```

## Questions to resolve before implementation

- Does the model need an explicit direction signal, or can it learn direction implicitly?
- Should the proportion be 50/50, or should forward be weighted more heavily?
- How does this interact with decompose_bypass_cross_window? Cross-window state from
  a forward pass would be incorrect context for a backward pass.
- Should generation always be forward-only, or can backward generation be useful?
- Does this help more at L=1 (limited receptive field) vs L=20 (already good context)?
- Would this be better framed as a data augmentation or as a separate training objective?

## Expected impact

- Effectively 2x training data per epoch (each sequence seen in both directions)
- May improve representation quality by forcing the model to capture bidirectional
  patterns in the wavelet decomposition
- Could particularly help with symmetric linguistic structures (e.g., "A is to B as
  B is to A") that unidirectional training underrepresents
