# Decoding Efficiency — Batch, Streaming, Speculative

> **Status (2026-07-14): plan of record.** Three stacked inference accelerations, all
> retraining-free, all quantitatively benchmarked with binary correctness gates. Development
> is CPU-viable against the Mini checkpoint (D1, `logs/wikitext-103_2026-07-12_14-20-46/`,
> 72.89M / 1.0103 sliding BPB); GPU rental only for the measurement sessions. Runs in the
> deep stream while the D-ladder occupies the training pod.

## Baseline fact (verified in code)

`generate_one` recomputes the full forward over `idx[:, -context_len:]` for **every token**
(generate.py ~L405–413) — O(window) work per token, sliding semantics (the window shifts by
one each step, re-aligning the dyadic grid). Training uses **block** semantics
(non-overlapping 256-windows + decompose-bypass carry). Measured baseline to beat: ~13.5
tok/s single-stream on the 5090 at C=512 (K4 generations).

## Stage 1 — Batch generation *(small; do first — it also builds the bench harness)*

Batched sampling loop in generate.py: N sequences decoded in parallel (leading batch dim),
per-sequence top-p / repetition-penalty / stopping. No model changes.
- **Correctness gate:** batch=1 output token-identical to today's path at fixed seed.
- **Benchmark:** aggregate tok/s and VRAM vs batch ∈ {1, 8, 32, 64, 128, 256} — find Mini's
  GPU saturation point. Deliverable: one table + curve.
- **Prediction on record** *(estimate)*: near-linear aggregate scaling to batch ~64–128 on a
  5090 (Mini leaves the GPU nearly idle at batch 1); >30× aggregate tok/s at the knee.

## Stage 2 — Streaming / stateful incremental decoding *(the big one)*

Cache per-level coefficient streams + crawl FIR histories + bypass state in ring buffers;
each new token computes only the ~2 amortized new coefficients per level instead of the full
window. Streaming-safety of components verified 2026-07-13: decomp/recon norms are
per-position `LayerNorm(Cp)` (model.py ~2104–2111); crawl offsets are strictly causal with
bounded support (~L526–546); mixers/gates are per-position; bypass is already stateful.

**Design: the `hop` knob.** Recompute the window from scratch every `hop` tokens; stream
incrementally between recomputes. `hop=1` ≡ today's sliding baseline; `hop=block_size` ≡
training-faithful block streaming (bypass carried at the boundary, as in training); values
between trade grid-freshness vs speed. This makes the semantics question an A/B, not a
debate.

- **Correctness gates (binary, the resume-test pattern):**
  1. While a window fills (no boundary crossed): streaming logits must equal the
     full-recompute forward **exactly** (max |Δlogit| ≤ fp32 tolerance ~1e-5; run the
     equality harness in fp32 on CPU).
  2. At hop boundaries: must equal a reference full-recompute implementation of the same
     hop semantics, exactly.
  3. Sliding BPB of the model function is untouched (streaming changes schedule, not math).
- **Benchmarks:**
  - **Per-token latency vs context position** — the headline plot: baseline is linear in
    position, streaming must be ~flat. This curve IS the O(window)→O(log) claim.
  - Single-stream tok/s at context 256: baseline ~13.5 tok/s (5090) vs streaming.
  - Quality A/B of hop semantics: hop ∈ {1, 64, 256} on fixed prompts — MeanLogP / Rep4 /
    D1-D3 metrics + eyeball. (hop=256 is training-faithful, so if anything it should
    *improve* on hop=1.)
- **Prediction on record** *(estimate)*: compute drops 50–100×; wall-clock single-stream
  gain **5–20×** (launch/Python overhead becomes the floor on GPU; CPU laptop gains may
  exceed GPU gains relatively). First version targets correctness + a ≥5× GPU win;
  kernel-level tuning is a later pass.
- **Engineering notes:** static ring buffers so torch.compile doesn't recompile per step
  (or compile disabled for the streaming path in v1); state object per layer {coeff streams
  per level, crawl history per level, bypass state}; `reset()` at hop boundaries except
  bypass (carried, per training).

## Stage 3 — Speculative decoding *(after 1+2; multiplies with both)*

Classic draft/verify (Leviathan/Chen rejection rule): **Mini (D1/D2) drafts γ tokens, SP1
(C=1024, 0.9805) verifies in one parallel forward** — WaveletLM verification is native
(training-style forward gives all-position logits). Same GPT-2 BPE tokenizer across the
family. Lossless by construction.
- **Correctness gate:** greedy spec-dec output token-identical to target-alone greedy.
- **Benchmarks:** accepted length τ per round (the DSpark headline metric) at γ ∈ {4, 6, 8};
  end-to-end tok/s vs SP1-alone; τ on WT-103 prompts vs off-domain prompts.
- **Prediction on record** *(estimate)*: high acceptance — Mini and SP1 are unusually close
  in quality (Δ≈0.03 BPB), so τ ≈ 3–5 at γ=6 on-domain; end-to-end speedup bounded by the
  ~2.5–3× per-token cost ratio (C² scaling), landing **1.5–2.5×** over SP1-alone. The
  interesting scientific byproduct: τ measures *model-family agreement* — a scaling-law
  cousin.
- Stage-2 streaming accelerates the drafter's sequential loop → spec-dec inherits it.

## Cost & workflow discipline

- **Develop + correctness-gate on CPU/workstation ($0)** — every gate above is exact-match
  and fp32-CPU-runnable against the Mini checkpoint. House rule anyway.
- **Rent the second 5090 only for measurement sessions** (throughput curves, latency plots,
  spec-dec timing): ~2–4h per stage ≈ **$2–4 per stage**, not a dev-long rental. Exception:
  if iterating on GPU-specific overheads (compile/CUDA-graph behavior) in Stage 2, a short
  dev rental is justified — cap it and bank the latency plot first.
- Results recorded here first (tables + harness pointers), promoted to README when a claim
  survives its gate — same convention as plans/interpretability.md.

## Order & status

1. [ ] Stage 1: batch generation + bench harness (harness serves all three stages)
2. [ ] Stage 2: streaming (correctness first, CPU; then the latency plot on GPU)
3. [ ] Stage 3: speculative decoding (needs 1's harness; benefits from 2)

Non-goals for v1: continuous batching/serving loop, CUDA graphs, tree/multi-draft
speculation, DSpark-style confidence heads (revisit post-release if serving ever matters).
