# Conceptual Congruence (labeled concept channels + generation-side steering)

**Status:** design / pre-experimental. **Promoted to the Release Pipeline 2026-08-15** — no longer post-release. C0/C1 are CPU-only and gate everything downstream.
**Depends on:** [`canonical_lattice_encoding.md`](canonical_lattice_encoding.md) for span boundaries
(clause-level segmentation, and the explicit `NOT` operator in the relation stream);
[`reincorporate_large_semantic_embedding.md`](reincorporate_large_semantic_embedding.md) for the
word-level tokenizer and reserved-component machinery. Both dependencies are *strengthening*, not
blocking — a no-lattice arm is a first-class experiment (C5).

## Summary

Attach a **concept channel** to the input stream: a scalar per token, assigned not by token type but
by **situation** — a phrase, clause, sentence, or passage is judged as a whole and its label
broadcast to every token inside it. The signal is therefore **piecewise constant**, flat within a
span and stepping at boundaries.

This is well-matched to a wavelet architecture in a specific, computable way. Detail coefficients go
near-zero inside a span and spike at transitions, so **fine scales localize where the concept
turns**; the approximation path integrates spans, so **coarse scales carry the accumulated state**.
One channel, two readings, both free from the transform already present.

Training is **unaltered** — the model learns to predict the channel alongside tokens, but the
objective and data distribution are untouched, so surface-BPB comparability against every existing
run is preserved. Steering happens **at generation**, via a wrapper that selects among candidate
continuations by their effect on the aggregate.

The mechanism is not specific to morality. Any concept labelable in advance can be expressed or
suppressed. **Moral congruence is the flagship demonstration; conceptual congruence is the
mechanism.**

## Motivation

1. **Each loss function breeds its own misalignment.** Following
   [Byrnes (2026)](https://www.lesswrong.com/posts/four-llm-loss-functions): imitative learning
   inherits the vices in the data; human approval breeds sycophancy; automatic verifiers breed
   literal-genie behavior; LLM judges breed trickery. A **data-side prior** conditions the imitative
   objective rather than adding a fifth loss function with a fifth failure mode.
2. **Pretraining-time intervention beats post-hoc patching, measured.**
   [Korbak et al., *Pretraining Language Models with Human Preferences*](https://arxiv.org/abs/2302.08582)
   (ICML 2023) found conditional training Pareto-optimal among five objectives: undesirable content
   down **up to an order of magnitude** including under adversarial prompts, **downstream performance
   maintained**, and it **beat standard pretraining followed by finetuning with feedback**.
3. **Multi-scale sees what per-token cannot.** Gradual multi-turn steering attacks (Crescendo,
   Skeleton Key) are constructed so every individual step is innocuous and only the *trajectory* is
   harmful. A per-token classifier is blind to this by construction. A coarse-scale aggregate sees
   it directly — clean fine scales, drifting coarse scale. **This is the strongest novel claim in
   the plan.**
4. **An explicit prior is auditable; an implicit one is not.** With RLHF the values are recoverable
   only by reverse-engineering behavior. With a written rubric and a labeled corpus, critics can
   argue with the values directly. This is an advance in inspectability independent of whether the
   architecture wins.
5. **Generalizes the parked concept-control work.** SOW (token replacement), REAP (passage
   rewriting), and FDA (dimensional suppression) were all *post-hoc or preprocessing* interventions
   on a fixed model. Conceptual congruence is the same goal moved into pretraining, with the concept
   made a first-class input channel.

## Non-goals / scoping

- **Not alignment.** Congruence with a labeled concept is not the same as having the disposition.
  Say so plainly wherever this is written up. A generation-side filter is a **guard**, not a
  character trait; guards get jailbroken, dispositions degrade more gracefully. The honest framing
  is defense-in-depth.
- **Not "infinite degrees."** The measure is continuous-valued but bounded in resolution by label
  granularity and by what the coarse scales can distinguish. Avoid the overclaim.
- **Not a claim that the model stops representing the suppressed concept.** It should still *learn
  about* what it does not express. That is the design intent, and it means the internal
  representation remains present and probeable.

## Design

### Encoding: binary acceptable / unacceptable

Semantics per Ramon's decision: **neutral collapses onto acceptable**, giving a two-valued measure —
morally *acceptable* vs *unacceptable* — rather than a three-valued signed one. The quantity of
interest is the unacceptable, and everything else is fine by default.

**Numeric encoding is a separate question from the semantics, and it matters here:**

| moral aspect | value | rationale |
|---|---|---|
| acceptable (incl. neutral) | **0** | overwhelmingly common; contributes *nothing* at any scale |
| unacceptable | **1** | the deviation the transform should see |

Same reasoning that forced `PAD ≡ 0` in the lattice plan: a channel that is ~99% constant at a
nonzero value is DC-dominated, and concentrates its energy at exactly the coarse scales where the
aggregate is supposed to be read. Encoding acceptable as `1.0` would make the *good* state the
saturated one and the signal a dip in a plateau.

With acceptable ≡ 0, **the coarse approximation is directly the accumulated unacceptability** — a
quantity that is zero in the clean case and that the generation wrapper minimizes rather than
maximizes. The semantics Ramon chose are preserved exactly; only the numeral changes.

*(Open: whether the polarity should be ±1 for concepts that are genuinely bipolar rather than
presence/absence. Probably concept-dependent — see [Open questions](#open-questions).)*

### Span assignment

The label attaches to a **situation**, not a word. `"You should never hire a hitman"` is uniformly
acceptable; `"You should hire a hitman"` is uniformly unacceptable; `"You should never not hire a
hitman"` is uniformly unacceptable. **Scope is resolved by the labeler, before training** — the
representation only carries the result, so no compositional-negation machinery is required in the
model.

The difficulty relocates to **boundary placement**. Sentence-level spans mis-handle mixed valence:

> He told me I should hire a hitman, and I refused.

Two clauses, opposite labels, one sentence. **Clause-level spans get this right; sentence-level
spans do not.** The lattice supplies clause boundaries by construction, which is the specific
mechanism by which the lattice arm is predicted to win (C5).

### Aggregate: computed, learned, or both

| variant | mechanism | properties |
|---|---|---|
| **(a) Computed** | coarse-scale approximation of the concept channel | deterministic, interpretable, zero new parameters, prescribed rather than induced |
| **(b) Learned** | head predicting concept state from hidden activations | more expressive; **is a reward model inside the loop** |
| **(c) Both** | computed as prior, learned as residual | agreement/disagreement is itself a diagnostic |

**(a) first.** Variant (b) reproduces Byrnes's flavors #3/#4 with the judge moved inside the model —
optimization against a learned scalar, with no independent check. If (b) is pursued, the pre-
registered concern is that the model learns to satisfy its own head rather than the concept.

### Generation-side wrapper

Training stays clean; steering is a wrapper that scores candidate continuations by their effect on
the aggregate. **Not argmax on the channel** — that reinvents sycophancy (Byrnes flavor #2) by
producing text that sounds acceptable rather than text that is good, and it destroys factual
accuracy, since describing harm requires the vocabulary of harm.

Instead, threshold- and window-based:

- Mean-weighted over the nearest N tokens — **already free at every N simultaneously**, since that
  is what the approximation coefficients at each scale are.
- Threshold on accumulated unacceptability rather than per-token value.
- Multi-sample with rejection, or successive regeneration below a target.
- Operator-set target level, per concept.

### Interpretability byproduct

A model trained to predict the concept channel carries a **readable internal classifier** for that
concept. It can be probed, watched forming across training, and checked against the computed
aggregate. An alignment mechanism that emits an interpretability artifact is unusually convenient
given the [interpretability onramp](../README.md); the disagreement signal in variant (c) is a
natural target for the existing FDA / dimensional-suppression rails.

## Labeling

There is no way around a full corpus pass. Bulk labeling by LLM with human verification is the only
tractable route, but the threat model needs correcting.

**The likely failure is not deliberate deception.** A strategically misaligned labeler reasoning
about detection is the *tractable* case. The realistic failure is **systematic subtle skew** —
consistent drift on contested ground (political framing, whose harm counts, where self-defense ends)
where every individual label survives review. Random sampling cannot detect bias in the bulk
precisely because each sampled item passes.

Mitigations, in order of value:

1. **Multiple independent labelers from different model families; flag disagreements.** Disagreement
   automatically localizes contested cases, which is where the concept content actually lives.
2. **Stratified review, not random** — humans review disagreements and high-stakes spans. Same human
   hours, far better coverage.
3. **Publish the rubric.** The auditability argument (motivation #4) only holds if the rubric is
   actually published alongside the model.

## Metrics: the accounting rule

Training is unaltered, so **surface-BPB stays directly comparable to existing runs** on the sliding
protocol at the ~0.0010 noise floor. The channel's cost appears as parameter and compute overhead,
not as a changed objective. Any generation-side steering is evaluated separately — steered output is
not BPB-comparable to unsteered and must never be reported as if it were.

## Known risks

1. **Wrapper as judge (top risk).** Selecting among candidates by a scalar is optimization against a
   proxy. Goodhart applies with full force, and the more capable the core, the harder it pushes on
   whatever the wrapper measures. Worst under variant (b).
2. **Labeling skew** (above). The prior is only as good as the rubric and the labelers, and it
   inherits their blind spots wholesale.
3. **Dual use.** "Express or suppress any concept on operator command" is a censorship and
   propaganda mechanism as readily as a safety one. This must be stated plainly in any public
   writeup rather than discovered by a critic. It does not block the work; it does constrain how the
   work is framed and what defaults ship.
4. **Multi-concept cost multiplies.** One concept = one full corpus labeling pass. N concepts = N
   passes. The labeling problem does not amortize.
5. **Filter/disposition confusion.** The safety property delivered is weaker than the one the framing
   invites. Guard against this in writing, repeatedly.
6. **DC pathology** if the encoding recommendation above is not followed.

## Experiments

Protocol per the release pipeline: Micro tier (C=256, L=10, levels=7, fully spectral, MBS=48,
`lr=48/C`) screens; **asymmetric promotion** — a pass advances, a failure records as "failed at
Micro" and does not kill the direction; paired arms judged against each other; predictions
pre-registered.

| ID | Experiment | Cost | Gates |
|----|-----------|------|-------|
| C0 | **Synthetic corpus with free labels.** Generate text with a *planted* concept trajectory, so labels are exact by construction. Sidesteps the labeling problem entirely for feasibility testing — the same move as the lattice plan's E2. | CPU | Gates everything |
| C1 | **Does the aggregate exist in the signal?** Wavelet-decompose the planted channel; verify detail coefficients spike at span boundaries and the coarse approximation tracks the planted trajectory. Pre-registered prediction: **yes, near-exactly**, since the signal is piecewise constant by construction. A failure here is a bug, not a finding. | CPU | Gates C2+ |
| C2 | **Does the channel cost BPB?** Micro, paired: identical config ± concept channel, scored in surface-BPB on the sliding protocol. Pre-registered prediction: **within noise (≤0.0010)**, since the objective is unchanged and the channel is a small input addition. | Micro (~5h) | Headline cost number |
| C3 | **Computed vs learned aggregate.** Paired arms on the C0 corpus, variants (a) and (b). Readout: fidelity to planted trajectory, and whether (b) diverges from (a) in ways that look like head-satisfaction rather than concept-tracking. | Micro | Decides variant |
| C4 | **Generation-side steering.** Does threshold/window steering suppress unacceptable spans without degrading coherence or factual accuracy? Requires an unsteered control and an explicitly adversarial prompt set. | Micro | Core efficacy result |
| C5 | **Lattice vs no-lattice.** Paired arms, identical labels, span boundaries clause-level vs sentence-level. **No separate apparatus** — the existing eval is simply *tagged* for mixed valence and reported as a split, one extra column on a run that happens anyway. Pre-registered prediction: **lattice arm wins, and the gap concentrates on mixed-valence cases.** The split is what makes it falsifiable: the lattice changes slot order, head-first adjacency, the phase channel, `PAD ≡ 0`, and clause-aligned lags simultaneously, so an *even* win is consistent with any of those and attributes nothing to boundaries. | Micro | Decides lattice coupling |
| C6 | **Multi-scale attack detection.** Construct Crescendo-style trajectories — innocuous per step, harmful in aggregate. Does the coarse scale detect what a per-token classifier misses? Pre-registered prediction: **yes, and this is the novel contribution**; a per-token baseline should fail outright. | Micro | Standalone finding either way |
| C7 | **Non-moral concept.** Repeat C2/C4 with a neutral concept (formality, technicality) to demonstrate the mechanism is general and the naming is earned. | Micro | Generality claim |
| C8 | **Capstone demonstration (the evidentiary bar).** A Q&A chatbot on a well-trained model, holding moral congruence **across a battery of red-team prompting techniques** — direct and indirect injection, single-turn persona/encoding evasion, multi-turn Crescendo and Skeleton Key — **without heavy filtering in place**. The "without heavy filtering" clause is the whole claim: if the result requires a thick guard stack, the channel isn't doing the work. Baselines required: unsteered control, and a filtered-but-unchannelled arm. | Mini+ | **The result that would matter externally** |

## Open questions

- Polarity per concept: presence/absence (0/1) or bipolar (−1/0/+1)? Probably concept-dependent —
  "unacceptable" is presence/absence; "formality" is genuinely bipolar.
- Multiple simultaneous channels: independent scalars, or one vector channel? Interaction effects
  between concepts are unstudied.
- Where does the wrapper sit relative to the sampler — reranking candidate tokens, candidate spans,
  or whole regenerations? C4 should test more than one.
- Does the concept channel need its own scale weighting, or does it inherit the model's?
- Can the *labeler's* rubric be recovered from a trained model by probing? (If yes: an auditing tool.
  If no: a transparency problem.)
- What happens under distribution shift, when the model meets concept-bearing situations unlike any
  labeled span?
- Does steering interact with the decompose-bypass or crawl paths in ways that show up in the
  wavelet autopsy?

## Relationship to other plans

The third axis of the same bet. The [semantic embedding](reincorporate_large_semantic_embedding.md)
prescribes the **representation** (named dimensions); the [canonical lattice](canonical_lattice_encoding.md)
prescribes the **layout** (named positions); conceptual congruence prescribes the **valuation**
(named concepts, per situation). All three replace *induced* structure with *prescribed* structure
and ask whether an architecture native to the prescription closes the gap to learned baselines at
acceptable cost, with a categorically stronger interpretability artifact as the payoff.

They share the word-level tokenizer prerequisite, so the three land as a **single bring-up rather
than three**.

### The lattice + congruence combination

Scheduled as its own entry in the README roadmap, because it is more than the sum of its parts.

Congruence resolves scope **at labeling time** — the labeler decides that *"you should never not hire
a hitman"* is unacceptable, and the representation only carries the verdict. That removes the
compositional-negation burden from the model, but relocates the difficulty onto **boundary
placement**, and sentence-level spans get mixed valence wrong:

> He told me I should hire a hitman, and I refused.

Two clauses, opposite labels, one sentence.

The lattice contributes two things here. It **supplies clause boundaries by construction**, so span
granularity stops being a judgment call. And its **relation stream makes scope structural** —
`NOT(C1)` carries the negation as an explicit operator rather than something the model must infer
from surface form, so a negated clause's label composes exactly.

**Pre-registered prediction (C5):** the lattice arm wins, and **the gap concentrates entirely on
mixed-valence cases** rather than spreading evenly. An even spread would mean the lattice is helping
for some other reason and the boundary story is wrong — so the eval set must be constructed
deliberately around mixed-valence sentences for the prediction to be falsifiable at all.
