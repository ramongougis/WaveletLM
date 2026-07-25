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

---

## Results & findings (2026-07-23 → 24)

**Additive AMB (unnormalized read) — MECHANISM VALIDATED on MQAR.** `tools/interpretability/mqar.py`:
vanilla WaveletLM plateaus (D=8 **56%**, D=16 **40%**); +AMB solves it (D=8 **91%**, D=16 **96%**,
still climbing), chance 1.6%. The bypass supplies the content-addressable retrieval vanilla can only
heuristically approximate. Near-ceiling oscillation (~96–98%) is a fixed-LR + per-batch-eval
artifact; the ceiling (<100%) is additive interference (crosstalk) — the delta rule's target.

**Denominator bug found + fixed.** First read was normalized (`y = Σ(q·k)v / Σ(q·k)`). With
L2-normalized (SIGNED) q,k the denominator is negative ~51% / near-zero ~12% of the time → `y`
exploded (absmax ~779 vs ~3) and corrupted the base — the initial MQAR failure. **Fix B:** drop the
denominator entirely, unnormalized read `y = q·S`. `eps` removed (dead).

**WT-103 is RECALL-LIGHT — the AMB installs nothing there.** Frozen-core adapter on D3 (AMBA v1,
`logs/wikitext-103_2026-07-23_19-03-41`): sliding BPB **0.9798 vs D3's 0.9797** — flat, inside the
0.0010 noise floor. The AMB *learned* (`|out_proj|`~1.1) but the model *damped it* (β 1.0→~0.10), and
`recall_diagnostics` on it is **identical to plain D3** (QRY-Δ +0.0036 absent, induction lift +0.142)
— **no recall installed.** Interpretation: **AMB recall is task-driven.** MQAR's loss rewards recall
→ it learns; WT-103's loss doesn't (Finding 7) → it never learns, even with the mechanism present and
the substrate (SRC-Δ +0.05) available. Not a wiring failure — an *objective* mismatch. v2 (stronger
frozen adapter) CANCELLED as redundant.

**Native (AMBN) prediction:** removes the frozen-core confound but NOT the objective one → expected
~flat + no-recall; the confound-free confirmation that WT-103 doesn't teach recall even to a
co-adapting model.

**The real test — a recall-DEMANDING objective.** Mixed WT-103 + generated-MQAR training
(`tools/interpretability/mqar_mixed.py`): the language half keeps it an LM, the MQAR half *rewards*
recall so the AMB gets a gradient to install it. Success = MQAR acc rises AND WT-103 stays healthy AND
`recall_diagnostics` QRY-Δ finally rises on the checkpoint. This is the "does AMB earn its place in a
language model" demonstration; natural recall-heavy data (long-context / code / Pile long deps) is the
scaled version.

**DeltaNet (v2) — for capacity, on recall-heavy data ONLY.** Read unchanged (`y = q·S`); write becomes
error-correcting `S_t = S_{t-1} + (v_t − S_{t-1}k_t)⊗k_t` (overwrites stale bindings → less
interference). Carries forward: unnormalized read, L2-normed keys, identity-at-init, ablatable. Needs
a chunkwise-parallel scan (the delta term is a recurrence, not a cumsum). **Does NOT fix the objective**
— flat on plain WT-103 for the same reason; test only on the mixed / recall-heavy objective.

## Native-run NaN: ROOT CAUSE + fix (2026-07-24, CONFIRMED by reproduction)

The native AMB (`associative_bypass_enabled` from step 0, fp16 AMP, C=512/L=10) NaN'd repeatedly
(~step 79–500). **Root cause is NOT residual feedback and NOT the injection magnitude** (both earlier
hypotheses were wrong). It is a **sporadic fp16 OVERFLOW inside the retrieval einsum** `num = q·S`:
`S = cumsum(k⊗v)` over T, so the sum-over-d reaches ~1e5 on an occasional batch — past the fp16 max
(65504) → `inf` → `y=inf` → NaN. Reproduced faithfully (fp16 + Adagrad + GradScaler + clip) at
C=256/L=6: NaN at the **deepest layer**, step 93, with residual healthy (~400, = AMB-off control),
`out_proj`=0.478, `beta`=1.008, injection=0.5 — i.e. **all weights tiny, purely fp16 range**, not a
growth/feedback blowup. An AMB-off control at the same scale was finite through 120.

**Fix:** run the whole scan in **true fp32** — wrap it in `torch.autocast(device_type=…, enabled=False)`.
`.float()` alone is insufficient (autocast re-casts einsum/matmul operands back to fp16). Confirmed:
finite through 400 steps where it was NaN@93. The pre-norm `assoc_ln` is kept (bounds `|v|`, reads
direction) but was never the fix — it slowed MQAR (84%@500 → 93%@1500, ceiling intact) and did not
stop the overflow. MQAR harness runs fp32 so the fix is a no-op there. Both AMBN and AMBN_xlayer
inherit this fix.

## Cross-layer AMB memory (2026-07-24, idea: Ramon) — queued as AMBN_xlayer

Depth-wise recall highway: each block's AMB reads `assoc_ln(x + γ · previous_block_AMB_output)`
(per-channel learned `γ`, init 0.1), so the associative read **compounds across depth** instead of
only seeing the prior read after the residual dilutes it. Config: `associative_bypass_cross_layer`
(default off). Threaded via explicit block return (`amb_out` as a 3rd value; robust `len(_ret)>2`
unpack) — compile-safe (Dynamo trace clean), **guarded off** with gradient_checkpointing (closure
recompute would drop the cross-layer gradient) and multinodal. Identity-at-init preserved (prev output
= 0 via zero-init out_proj → exact-0 for any γ; CPU-smoked). Adds ~5.1K params (10·C). A/B iso vs AMBN
isolates the cross-layer term; >0.001 over AMBN ⇒ compounding helps, flat ⇒ WT-103 still doesn't reward
recall (same read as AMBN). A capacity axis distinct from DeltaNet (which compounds *within* a
sequence); both are post-stability ablations.

## PARKED 2026-07-25 — close-out record

**Verdict: the mechanism works; WT-103 cannot score it.** Parked after five WT-103 attempts,
resumes post-release on recall-demanding data (README "Immediate post-release").

**What the instrument showed** (`tools/interpretability/amb_selectivity.py` on the softplus_l2
1-epoch checkpoint `logs/wikitext-103_2026-07-24_18-08-02`, all 10 layers):

| metric | measured | meaning |
|---|---|---|
| unrelated `⟨q,k⟩` | 0.863–0.866 | **identical to its init value** — the read never sharpened |
| read-weight entropy / log(t) | 1.000 | perfectly uniform ⇒ a running mean, not a content-addressed read |
| ‖β·amb‖/‖x‖ | 0.0000 | the model switched the module off |
| effrank(S) of d=64 | 3.5–8.5 | state heavily superposed |

So the flat BPB was never a tuning failure — the AMB contributed *exactly nothing*, which is the
predicted consequence of Finding 7 (WT-103 is recall-light). **A BPB A/B on WT-103 can only show
"does it harm" (it doesn't), never "does it work."**

**Feature-map screen (all five, for the record).** The DC floor and numerical stability turned out
to be coupled: elu1's ~1.0/dim floor makes `q·k` ≈ d-dominated → unselective → stalls epoch 2
(+0.17 val vs D0 and widening); removing the floor buys selectivity but every unbounded or
normalize-based variant diverged under Adagrad(acc=0)+fp16 — relu2 NaN'd pod step 635 at lr=3.25e-3
(*tiny* — a magnitude problem, not an LR one), relu_l2 died step 500, relu2_l2 ramped 5.6→14.6→104.7
→NaN by step 2000, softplus_s ran one clean epoch then NaN'd ~step 10.1K. softplus_l2 is the only
stable one and is stable *because* it is inert. A denominator-collapse hypothesis was tested and
**refuted** (at init `min den` ≥ 0.08 and `|y|` ≤ max|v| for every map incl. softplus_s at s=100),
so the divergence lives in trained-weight drift, not a static property. Not further diagnosed: the
one checkpoint that could have shown it was overwritten by 10K post-NaN steps (see below).

**Infrastructure fix that came out of this (`train.py`).** A diverged run previously trained on for
hours and *overwrote* `last_checkpoint.pt` with all-NaN weights — the softplus_s run reached
global_step 20000 with 2034/2054 tensors non-finite incl. `token_embedding`, destroying the
step-10000 snapshot taken right before the blowup. `train.py` now aborts on non-finite eval loss
(`abort_on_nan`, default true) and never checkpoints non-finite weights, so the last good snapshot
survives and `run_ablation` continues the queue. Applies to every future run, not just AMB.

**Built and validated, waiting for a venue:**
- `associative_bypass_per_scale` — injects the read in **coefficient space, one learned gain per
  scale** (after `scale_weights`, so `beta_s` is a clean per-scale write statistic) instead of the
  full-width post-reconstruction write, which is the one block component with no scale index.
  CPU-smoked: identity-at-init exact-0, all S gains receive independent gradients, output verified
  ≠ full-width write. Attribution caveat: the *injection* is exactly per-scale, but lifting
  reconstruct is non-linear, so "AMB share of the output at scale s" is not exactly decomposable.
- `mqar_mixed.py --feature_map/--per_scale` — the mixed WT-103+MQAR objective, the one venue where
  the read is exercised. Queued after the CTX1024_L9 resume.
- **Delta rule (v2)** — still the indicated capacity/interference upgrade; needs a chunkwise-
  parallel scan (the delta term is a recurrence, not a cumsum) and recall-heavy eval.

**Rejected on inspection (recorded so it isn't re-proposed):** a uniform scalar "temperature" on
q or k cannot sharpen this read — it is a *ratio* `y = Σw·v / Σw`, so any uniform positive rescale
of every score cancels exactly. Sharpening requires a per-score nonlinearity (what `relu2_l2`'s
squaring does), not a temperature.
