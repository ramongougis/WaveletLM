# Associative Memory & Activation/Weight Engineering for WaveletLM

> **Status: PLAN (2026-07-23).** Motivated by [interpretability.md](interpretability.md) Finding 7:
> WaveletLM does no in-context *retrieval* (induction lift ~0; VALUE at rank ~25k) while its
> *contextual/statistical* memory is strong (−2.11-nat position ramp). This plan pursues two
> complementary goals: (A/B) **understand and steer** the recall gap via activation/weight
> engineering, and (C) **install** dynamic recall via an architectural memory that is neither
> attention nor an MLP.
>
> **Hard constraint (Ramon):** no self-attention, no new MLPs. Everything here respects that —
> the memory in Track C is a *linear-attention-family state* (outer-product / delta-rule), which
> is O(T), attention-free, and MLP-free.

---

## 0. The organizing hypothesis

**Latent-capability hypothesis:** the information needed for recall is *present* in the model's
representations, but not *routed* to where the readout can use it. If true, a small, structured
edit to early/mid computation should unlock it; if false, no edit generalizes and we must add a
mechanism (Track C). This is exactly the premise of **representation engineering** — reading and
writing internal representations to control behavior ([Zou et al. 2023, arXiv:2310.01405](https://arxiv.org/abs/2310.01405)).
Ramon's framing — *"steer the model to a more correct region of semantic space, not directly to
one correct logit"* — is the right one, and it has a precise operational meaning: an **early/mid
edit that changes routing generalizes; a last-layer/logit edit memorizes one example.** The
generalization test below is what distinguishes the two.

Alignment relevance is real, not decorative: activation/representation engineering is an active
alignment agenda (steering honesty, refusal, sycophancy), and a minimal-edit + generalization
methodology on a *fully legible* architecture is a stronger transparency story than on a
transformer.

---

## Track A — Diagnostics first (cheap, existing checkpoints, no training)

These answer *is recall latent, and where does it live* before we spend effort steering it. Both
extend `tools/interpretability/induction_probe.py` (the induction testbed already built).

**A1 — Read vs. write bottleneck (representation surgery / activation patching).**
Inject a synthetic binding into KEY's representation at layer ℓ (patch it toward VALUE, or add a
constructed k⊗v), then check whether the query position retrieves VALUE.
- Reads work / writes don't → the readout is fine; nothing *writes* the binding → Track C is the
  fix and it *will* be readable.
- Even injected bindings can't be read → deeper readout problem; Track C must also address addressing.
Sweep ℓ to find *where* a binding must live to be retrievable → tells Track C where to wire.

**A2 — Do generalizing recall directions exist? (difference-of-means / contrastive steering.)**
Per example, find a constrained activation intervention that raises logP(VALUE); collect across
examples; extract the shared low-rank component (the candidate "recall direction"). This is the
established **activation-addition / contrastive-activation-addition** method
([Turner et al., ActAdd, arXiv:2308.10248](https://arxiv.org/abs/2308.10248);
[Rimsky et al., CAA, arXiv:2312.06681](https://arxiv.org/abs/2312.06681) — note CAA's key lesson:
*hundreds of contrast pairs* reduce steering-vector noise; do not derive a direction from one pair).
**The control that makes it non-vacuous:** the intervention must be (a) *constrained* — low-rank,
injected early/mid, NOT at the logits (else the trivial solution is "add VALUE's unembedding");
and (b) *generalization-tested* — fit on one set of KEY/VALUE pairs, measure recall on held-out
pairs at held-out positions. Generalizes → capability is **latent** (Track C will work); doesn't →
recall is genuinely absent and needs the mechanism.

**Expected outcome (honest, given Finding 7):** A1 shows reads work / writes don't; A2 finds a
weak, partially-generalizing direction. That combination = *"representable but neither learned nor
writable without a memory mechanism"* → a clean, publishable motivation for Track C. A *cleanly*
generalizing direction would be the more exciting surprise.

---

## Track B — Minimal structured weight editing (the "engineering" track)

This is Ramon's proposal. The **instinct is good and matches a real literature; one method choice
needs swapping.** Straight talk, then the reframe.

### What's right (keep it)
- **Minimality** — *"nudge the weights as little as possible; minimize the magnitude of the
  adjustment."* This is exactly the **ROME** principle: a *rank-one*, minimal, localized edit that
  changes one association while preserving everything else ([Meng et al. 2022, arXiv:2202.05262](https://arxiv.org/abs/2202.05262)).
- **Weight-difference candidates, select the best** — this is **task arithmetic**: a task vector is
  a weight-diff (finetuned − base) that can be scaled, combined, negated
  ([Ilharco et al. 2022, arXiv:2212.04089](https://arxiv.org/abs/2212.04089)).
- **Early/mid-layer targeting** — ROME found *middle-layer* modules are the causal mediators of
  factual recall; edit where information is *routed*, not where it's *read out*. Track A localizes
  the right site for WaveletLM specifically.
- **Batch of previously-incorrect examples; push toward "majority corrected"** — correct instinct;
  see the forgetting caveat below for why it must be batch, not sequential.
- **Subset BPB for fast comparison** — good and standard (minibatch loss estimation).

### The one wrong tool: a random walk over 73M parameters
A gradient-free random walk in 73-million-dimensional weight space is **catastrophically
sample-inefficient** — the fraction of random directions that improve a high-dim objective shrinks
toward zero, and each BPB probe costs real time. More fundamentally: **"find weights that lower
BPB" is exactly what training already does, optimally, using the gradient.** A random walk throws
the gradient away and re-solves a solved problem worse. So the full-parameter random walk is not
the method.

### The reconciliation (this rescues the instinct)
The random-walk intuition becomes tractable the moment the edit is **parameterized
low-dimensionally** — which is *also* the minimality Ramon wanted. Two moves:
1. **Constrain the edit to be small and structured** — a rank-1/low-rank delta on one chosen
   weight (ROME-style), or an additive steering direction (repE-style). Now the search space is
   ~1–1000 dims, not 73M.
2. **Optimize it with the gradient** (it's differentiable: `∂ logP(correct) / ∂ Δ`), with an
   explicit **magnitude penalty** (`+λ‖Δ‖`) to enforce minimality. If the parameterization is tiny
   (a few coefficients on a fixed direction), a random/grid search over *those* is even fine — the
   point is low dimension, not zeroth-order per se.

So: **gradient-based, magnitude-penalized, low-rank, early/mid edit** that maximizes correctness on
a *batch* of held-out-incorrect examples, evaluated by subset BPB, and **generalization-tested**.
That achieves everything the proposal wanted (minimal edit, correct the examples, select by BPB)
without the intractable part.

### Correct the framing of the *objective*
Reducing BPB by editing weights, taken to its limit, *is* training — so raw BPB reduction is not
the scientific prize. The prize is: **how minimal/low-rank/localized is the edit that induces
recall, and does it generalize?** Report the *edit's rank and norm* and its *held-out
generalization*, with BPB as a guardrail (the edit must not raise global BPB — specificity), not as
the thing to minimize. This is precisely ROME's dual metric: **efficacy + specificity/generalization.**

### Catastrophic forgetting: batch, not sequential
The proposed loop (correct example 1, then 2, then 3…) hits a documented failure: sequential edits
**gradually then catastrophically forget** earlier ones ([Gupta et al. 2024, arXiv:2401.07453](https://arxiv.org/abs/2401.07453)).
Correcting example 2 un-corrects example 1. The fix is the **mass-editing** approach — solve for
*one shared edit* over the whole batch at once — which scales to thousands of edits in practice
([MEMIT, Meng et al., arXiv:2210.07229](https://arxiv.org/abs/2210.07229)). Ramon's "majority
corrected → BPB drops" is right; reach it with a shared batched edit, not a greedy chain.

### WaveletLM has no FFN — so where do we edit?
ROME/MEMIT edit *transformer FFN* weights; we have none by design. But the methodology ports to
non-transformer architectures: [**"Locating and Editing Factual Associations in Mamba"**
(arXiv:2404.03646)](https://arxiv.org/abs/2404.03646) adapts ROME to an SSM by editing the
analogous state/projection matrices. For WaveletLM the candidate edit sites are the **lifting
weights, the per-scale mixers, and the decompose-bypass** — and **Track A's read/write map tells us
which one is causal.** Doing ROME-style localized editing on a *wavelet* model is genuinely novel
and a paper-worthy angle in its own right.

### Practical notes (correcting two cost estimates)
- **VRAM for 3 weight copies is a non-issue:** 73M params × 3 copies × 4 B ≈ **0.9 GB**. Hold
  original + best + candidate in memory freely. (The real cost is eval time, correctly identified.)
- **Subset BPB — refine the sampling:** for comparing two candidate edits, evaluate both on the
  **same** random subset (paired comparison → low variance, the deciding signal isn't drowned by
  which samples were drawn); **reshuffle the subset across iterations** (so no edit overfits a fixed
  subset). This is standard stochastic-optimization practice (and how CMA-ES-style evaluation is
  done). Size the subset for ~seconds/eval vs the full 1m17s.
- **The full sliding-window BPB is the *final* specificity gate**, run only on the surviving
  edit(s), not inside the inner loop.

---

## Track C — The architectural fix: associative-memory bypass

The mechanism that *dynamically* installs recall, informed by A (where) and B (is it latent).
Detail and literature in the earlier discussion; summary:

- **Upgrade the decompose-bypass from a vector running-mean to a matrix key–value state.** Today
  `_compute_running_mean(x)` is `S = Σ xᵢ` — the degenerate, uniform-key associative memory.
  Add k/v/q projections and carry `S = Σ kᵢ⊗vᵢ`, retrieve `≈ S·q`. This is the fast-weight /
  linear-attention associative memory — **O(T), attention-free, MLP-free** (respects the constraint).
- **Write rule, staged:** additive first (easy), then the **delta rule** `S += (v − S·k)⊗k`, which
  is error-correcting and the measured recall winner ([DeltaNet, arXiv:2406.06484](https://arxiv.org/abs/2406.06484);
  [Gated DeltaNet, arXiv:2412.06464](https://arxiv.org/abs/2412.06464)). Optional input-dependent
  gate = selectivity.
- **Recall is state-bounded** ([Zoology, arXiv:2312.04927](https://arxiv.org/abs/2312.04927);
  [Based, arXiv:2402.18668](https://arxiv.org/abs/2402.18668)) — won't match attention's unbounded
  KV cache, but should move induction lift from ~0 to real.
- **Eval:** the induction probe + Zoology's **MQAR** synthetic task ([code](https://github.com/HazyResearch/zoology)) + BPB.

---

## Staging & decision rules

1. **Track A** (days, laptop, no training). Read/write verdict + generalizing-direction test.
2. Branch on A:
   - **Latent + writes-are-the-gap** → **Track C** (build the bypass memory); **Track B** becomes
     the *interpretability probe* that shows *what* the memory needs to store.
   - **Not latent** → recall is genuinely absent; Track C is mandatory, Track B is diagnostic only.
3. **Track B** runs in parallel as the mechanism-discovery / interpretability instrument (minimal
   edit rank+norm+generalization), never as a BPB optimizer.
4. Gate everything on the **induction probe + MQAR**, with BPB as the specificity guardrail.

## Key references
- Model editing: [ROME 2202.05262](https://arxiv.org/abs/2202.05262) ·
  [MEMIT 2210.07229](https://arxiv.org/abs/2210.07229) ·
  [ROME-on-Mamba 2404.03646](https://arxiv.org/abs/2404.03646) ·
  [Task Arithmetic 2212.04089](https://arxiv.org/abs/2212.04089) ·
  [Editing-at-scale forgetting 2401.07453](https://arxiv.org/abs/2401.07453)
- Activation/representation engineering: [RepE 2310.01405](https://arxiv.org/abs/2310.01405) ·
  [ActAdd 2308.10248](https://arxiv.org/abs/2308.10248) ·
  [CAA 2312.06681](https://arxiv.org/abs/2312.06681)
- Subquadratic recall: [Zoology 2312.04927](https://arxiv.org/abs/2312.04927) ·
  [Based 2402.18668](https://arxiv.org/abs/2402.18668) ·
  [DeltaNet 2406.06484](https://arxiv.org/abs/2406.06484) ·
  [Gated DeltaNet 2412.06464](https://arxiv.org/abs/2412.06464)
