# Generation Decode Speedup (compile / CUDA graphs)

> The single largest single-stream inference win available, at **zero quality cost** (same computation, just lower overhead). Measured baseline: ~27 tok/s single-stream — roughly **30× below the bandwidth roofline** (~800 tok/s for the ~1 GB C=2048 model). That gap is per-token *overhead* (Python decode loop, kernel-launch dispatch, no generation-mode compile), not compute or bandwidth — so removing overhead, not changing the architecture, is the lever. `generate.py` currently applies **no `torch.compile`** to the model; decode runs fully eager.

## Why it's graph-able

The decode loop (`generate.py` `generate_one`, ~L405) is:
```
for i in range(num_tokens):
    idx_cond = idx[:, -context_len:]      # last context_len tokens
    logits, _ = model(idx_cond)            # full forward (no KV cache — attention-free)
    next = sample(logits[:, -1, :])        # data-dependent sampling
    idx = torch.cat((idx, next), dim=1)
```
- **Steady state (after the first `context_len` tokens): `idx_cond` is a fixed `[1, context_len]` shape every step** — static, the precondition for CUDA-graph capture.
- **Prefill / early phase**: shape grows `1 → context_len` (dynamic) — handle separately (eager, or a dynamic-shape compile; it's a one-time cost, not the steady-state hot path).
- No KV cache means each token recomputes the whole window (O(context_len·log) per token). Overhead-removal (this plan) does **not** change that; reducing the recompute is the separate Tier 3 below.

## Tiered approach (cost-ascending)

**Tier 1 — `torch.compile(model, mode="reduce-overhead")` for decode.** `reduce-overhead` auto-applies CUDA graphs under the hood; likely captures most of the win for near-zero effort. Compile the steady-state forward; run prefill eager (or `dynamic=True`). One config/flag in `generate.py`. Expected: a large multiple over 27 tok/s.

**Tier 2 — hand-rolled CUDA-graph capture** (only if Tier 1 leaves residual overhead). Capture the static `[1, context_len] → logits` forward against a **fixed input buffer** (copy the current window in, replay, read static output). **Sample outside the graph** — temperature/top-p/repetition-penalty are data-dependent and stay in eager on the single-position logits; the token append + window slice are cheap host ops. More control, removes dispatch overhead the compiler leaves.

**Tier 3 (separate, bigger, architectural) — incremental/stateful decode.** Avoid the full-window recompute per token by caching causal wavelet / crawl / decompose-bypass state, turning per-token cost from O(n·log n) to ~O(log n). This is the KV-cache-equivalent for an attention-free model, and it is the win that matters for **long-context generation** (where full recompute per token dominates — see the wide-C / long-block_size discussion). Architecturally involved (stateful causal lifting + crawl windows); plan separately if/when long-context generation becomes a target.

## Correctness & measurement

- **Quality cost: zero** — it's the same computation. Validate with a compiled-vs-eager logit match (allclose within fp16 tol) on a fixed prompt; no BPB re-run needed.
- **Measure** tok/s vs the 27 tok/s eager baseline (the inference-depth study's `--n 2` warmed-sample protocol). Report on the target serving GPU (5090 / B200), since overhead removal interacts with the hardware's launch latency.

## Caveats

- **AMP + CUDA graphs**: autocast must be inside capture; the fp16 path is the one to graph.
- **Fixed buffers**: graph replay requires stable memory addresses — static input/output tensors copied in/out each step.
- **Prefill**: the dynamic-shape early phase is a one-time cost; don't over-engineer it — eager prefill then graphed steady-state is fine.
- **Compile cache**: first call pays compilation; warm up before timing (and before serving).
- This is single-stream latency. Aggregate serving throughput is a separate axis (batch hard — no KV cache enables it); the two compose.
