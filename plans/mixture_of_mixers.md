# Mixture-of-Mixers (MoM) — v1 Design

> **Status (2026-07-21): IMPLEMENTED and smoke-tested** (forward/backward finite, router
> gradient flowing, zero-init aux = ln(1/4) exactly, parameter_breakdown handled, guards
> for depth>1 / shrinkage-replace / topk>E all fire, default path regression-checked).
> Arms MOMA/MOMB queued in runs.sh after PP2, before K5. Supersedes the one-line "Sparse
> mixture-of-mixers first test" bullet in the README Release Pipeline.
> **Design amendment vs the original spec:** the router is *static per-scale* (a
> parameter, not data-dependent), so Switch-style dispatch-fraction load balancing does
> not apply; implemented instead as an **expert-usage entropy penalty** (usage = mean
> over scales of the softmax router; aux = Σ u·log u, minimized → uniform usage).

## Motivation

Three stacked hypotheses, cheapest first:
1. **Capacity sharing beats dedication.** Today every scale owns a dedicated gated-SwiGLU
   mixer at a hand-set width ([1.0×4, 0.5×4]). A shared pool lets hot scales borrow
   capacity from quiet ones — and the census/kurtosis maps show scale activity is far
   from uniform (loud ends, quiet middle).
2. **Synergy with the frozen/imported lifting (FT1/FT2 results feed this).** If the
   lifting is a transferable router, a MoM model warm-started on a frozen donor lifting
   trains only "experts behind a fixed decomposition" — plausibly faster-converging than
   end-to-end from scratch (Ramon's conjecture; testable as a direct arm).
3. **A possible third scaling axis.** With top-k routing fixed at 2, expert count E grows
   parameters ~linearly while step-time stays ~flat (2 expert applications per scale
   regardless of E). If BPB follows a power law in E at fixed C and epochs, that is a
   params-without-wall-clock axis distinct from both C and E_epochs — checkable with a
   single E ∈ {2, 4, 8, 16} sweep at Mini prices (*pre-registered as a hypothesis, not an
   expectation: MoE literature says gains need scale and good balancing*).

## v1 design

- **Pool:** per layer, E full-width (Cp→Cp) gated SwiGLU mixers, replacing the per-scale
  dedicated ModuleList. (`mixer_mom_share_across_layers` exists but defaults false in v1.)
- **Router:** learned (S × E) logits per layer; per scale, softmax over experts →
  **top-2** dispatch, outputs combined by the renormalized gate weights. Cross-scale
  gating (the (S,S) routing) is upstream and unchanged.
- **Load balancing:** Switch-style auxiliary loss (fraction-dispatched × mean-gate dot
  product), weight `mixer_mom_aux_weight` = 0.01; logged per eval so collapse is visible.
- **Init:** router logits zero (uniform) + experts standard-init; identity-at-init is NOT
  preserved (unlike most WaveletLM features) — the smoke test therefore checks
  finiteness + gradient flow + aux-loss magnitude, and the first arm watches early-step
  stability at lr 0.075 (fallback: halve).
- **Config:** `mixer_mom_enabled`, `mixer_mom_experts` (E), `mixer_mom_topk` (default 2),
  `mixer_mom_aux_weight`, `mixer_mom_share_across_layers`.

## Cost accounting (Mini, C=512)

Dedicated mixers today: ~3.69M/layer (width-sum 6.0). MoM pool at E=4 full-width:
~2.46M/layer → **saves ~12M params model-wide** while giving every scale access to two
full-width experts. Compute: top-2 ≈ 16 mixer applications per block vs 8 → mixer FLOPs
×2; expected step-time hit ~20–35% *(est.)* — measure at arm 1.

## Arms (Mini, 5ep, MBS=48, ~$6–8 each)

1. **MoM-A:** E=4, top-2, vs D0 (1.0436) — the existence test.
2. **MoM-B:** E=4 + `lifting_import_checkpoint` (D2) + `lifting_freeze` — the synergy arm;
   compare matched-step val vs MoM-A AND vs FT1 (isolates MoM×frozen interaction).
3. **MoM-C (conditional on A ≥ tie):** E sweep {2, 8, 16} for the third-axis question;
   fit BPB vs E if monotone.

Decision rules: A must be within noise of D0 to justify C (it already saves params — a
tie is a win, same logic as the coarse-prune arm); B is judged on convergence speed at
steps 2K/4K/8K, not just final BPB.

## Risks

Router collapse (mitigated by aux loss + logging), init sensitivity at the house LR, and
torch.compile graph churn from top-k dispatch (fallback: gather-based dense dispatch of
2 experts — compute both, weight, no dynamic control flow).

---

## Block-level MoE (the sibling design) — IMPLEMENTED 2026-07-22

Where MoM shares only the **mixers** behind one decomposition via a *static per-scale*
router, block-MoE routes to **E independent full WaveletLMBlocks per layer** — each with its
own lifting wavelets, mixers, bypass, and reconstruction ("own wavelets and everything in
between") — via a **per-token, data-dependent** router (`nn.Linear(C, E)` on the token
state). Class `WaveletLMBlockMoE` (model.py); config keys `block_moe_enabled`,
`block_moe_experts`, `block_moe_topk`, `block_moe_aux_weight`.

Key design points (and how they differ from MoM):
- **Switch-style load-balance aux is correct here** (`E·⟨f_e, P_e⟩`, balanced value = 2.0 at
  E=4/top-2, verified) *because* the router is genuinely data-dependent — the exact opposite
  of MoM, where a static router made Switch balancing meaningless and we used usage-entropy.
- **Dense compute, ~E× a single block.** Routing individual tokens through a *sequence*
  transform can't be sparsified without breaking the transform, so every expert runs on the
  full sequence and outputs (and the linearly-combinable running-mean bypass state) are
  blended per token by the renormalized top-k weight. top-k controls blending, not FLOPs.
  All experts run in training (stable aux gradient + graph); inference skips unselected ones.
- **Experts must own their wavelets** ⇒ requires `shared_lifting_weights=false` (guarded).
  This is SB4's untied-lifting regime, so it inherits the LR halving (0.0375, not 0.075).
- **Not iso-param** (measured at Mini): D0 shared-on 72.89M; 1-expert shared-off 139.08M;
  E=4 479.13M. So the honest baseline is the **1-expert shared-off control (MOE0)**, which
  isolates routing from the +66M that turning off lifting-sharing alone costs — *not* D0.

Guards: incompatible with `looped_blocks`, `mixer_mom_enabled`, and (via shared-lifting)
the dilation-schedule / prime-power / complex-basis features. parameter_breakdown detects
the wrapper and reports the pool + a representative expert (avoids the SH3-class crash).

Smoke-tested (CPU): forward/backward finite, aux finite = 2.0 at balanced init, router +
**all** experts receive gradient, router routes (distinct per-token top-k), breakdown runs,
guards fire, default path unaffected. Arms MOE0/MOEA in runs.sh. Strategic note: its
content-dependent routing is WaveletLM's first move toward the input-conditioned mixing that
the induction probe (plans/interpretability.md, Finding 7) showed the LTI-like core lacks.
