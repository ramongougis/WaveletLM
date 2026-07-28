# Large-Chunk Test-Time Training (LaCT) — parked for post-release

**Status:** banked 2026-07-28, no implementation intended now. Reference only.
**Paper:** Zhang et al., *Test-Time Training Done Right*, arXiv:2505.23884.

## The one idea

Existing TTT updates fast weights every 16–64 tokens, which runs below 5% FLOPs
utilisation on modern GPUs. LaCT inverts this: **update once per 2K–1M-token chunk**.
Larger chunks raise utilisation by orders of magnitude (up to ~70% on A100) in plain
PyTorch, which in turn makes it affordable to scale the *state* to ~40% of model
parameter size — an order of magnitude past the 0.1–5% typical of prior TTT work.
Bigger state is where the measured gains come from (their Fig. 7a).

## Why it is parked, not rejected

- **It is dispatch-friendly, which is our measured bottleneck.** We issue ~41,726 CUDA
  kernels/step, ~69% pure dispatch. LaCT's whole argument is about doing fewer, larger
  operations. That argument applies to us more than to most.
- **But it targets 32K contexts and we train at 256.** Their language-model runs use
  chunk = window = 2048+ tokens, i.e. eight of our entire windows per chunk. There is no
  meaningful "large chunk" at our context length.
- **It is attention-hybrid by design.** Language modelling needs sliding-window attention
  alongside the TTT layer (window >= chunk) to recover per-token causality, since a chunk
  is treated as an unordered set. That collides with hard rule 9 — a wavelet+attention
  hybrid belongs in the separate repo, not here. Whether the wavelet mixer could play the
  window-attention role is an open and interesting question.

## Points worth stealing even if LaCT never lands

- **Shifted block-wise causal ordering** (their Fig. 3c): *apply* the fast weight to a chunk
  BEFORE *updating* on it, so predictions never see inside their own chunk. This is the
  correct causal pattern for any chunked fast-weight scheme, FwPKM included.
- **Update-rule menu:** gradient descent -> momentum -> Muon, with L2 weight normalisation
  after each update. Muon measurably beat GD and momentum (Fig. 7b); their normalisation
  replaces weight decay and is analogous to post-LN on the residual path.
- **Per-token learning rates** predicted from the input (`softplus(Linear(x) + bias)`),
  init'd so `softplus(bias) ~ 0.001–0.01`.
- **Nonlinear state beats linear state at equal or smaller size** (Fig. 8a) — their fast
  weight is a SwiGLU MLP, not a matrix.

## Prerequisites before this is worth revisiting

Longer context (decimation), and a decision on the attention-hybrid question. Revisit only
after FwPKM has produced a real result, since FwPKM answers the cheaper version of the same
question: does online-updated memory help this architecture at all.
