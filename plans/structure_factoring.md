# Structure Factoring

> Organizing principle (2026-06-11): **make the model opaque only about what genuinely requires learning.** Every bit of structure that can be prescribed explicitly — corpus statistics, positional relationships, memorization — is a bit the opaque weights no longer need to encode. Factor known structure into fixed, inspectable components; let the network learn only the residual. The bet: the residual is smaller in content and simpler in mechanism, and every factored component is a falsifiable interpretability artifact (ablate it, measure its contribution).

This unifies several existing components and planned directions (crawl, PKM/FwPKM, the semantic embedding, the frozen-tied head) under one statement, and gives a design rule for future factorings.

## The design rule (from our own evidence)

**Factoring succeeds at interfaces and as additive parallel channels; it fails as mandatory mid-network re-encodings.**

| Instance | Location | Outcome | Evidence |
|---|---|---|---|
| Wavelet crawl | augments input pairing, additive over offsets | **Win** | −0.0181 BPB at T4 for 21 readable softmax logits — best interpretability-per-parameter component in the model |
| PKM / FwPKM | parallel additive channels | **Win** | explicit sparse key-value memory, readable top-k addressing, established components |
| Additive memory composition (`alpha_mlp/pkm/fwpkm`) | composition rule | **Win** | per-factor contribution measurable by construction |
| Semantic embedding + frozen-tied head | input/output interfaces | planned | prescribes the edges; trunk unconstrained (see `reincorporate_large_semantic_embedding.md`) |
| FWHT in the mixer slot | **mandatory in-series mid-network re-encoding** | **Loss** | identity beat it (+0.0029); learned butterfly declined to leave identity (README Mixer Transform Ablation) |

Prescribe at the edges and in parallel; never in series through the middle. A mid-network prescription forces every signal through structure the learnable components on both sides must accommodate; an interface or parallel prescription only *offers* structure, with a learned scalar deciding how much to use.

## Next concrete test: frozen skip-gram logit prior

Factor low-order positional co-occurrence statistics out of the weights and into explicit count tables at the LM head.

**Recipe:**
1. One corpus pass: count-based next-token tables conditioned on the token at distance d, for dyadic offsets d ∈ {1, 2, 4, 8} (matching wavelet scales). Sparse: top-K contexts per table (heavy-tailed statistics — capture the head explicitly, leave the tail to weights).
2. Freeze tables. Add at the head with learned per-offset scalars:
   `logits = f_θ(x) + Σ_d α_d · SkipGram_d(x_{t−d})`
3. Train at T4 1ep vs T4 baseline.

**Measurements:**
- BPB delta and the learned α_d values (how much explicit structure the model accepts).
- Ablation: zero the prior at eval — how much quality is attributable to the explicit factor.
- **Residual shrinkage** (the deeper question): can `mlp_expansion` drop at matched quality once n-gram statistics are offloaded? If yes, the factoring genuinely relieved the weights.
- **Error-corrector check** (see cautions): correlation between the residual network's output and the prior's errors.

## Cautions (standing failure modes)

1. **The residual is not guaranteed to be simpler.** A network trained alongside a frozen prior can learn *anti-correlated corrections* to the prior's errors, and an error-corrector can be weirder than the original. Measurable (anticorrelation check above); mitigable (anneal the prior in, or gate it). Record this check with every factoring experiment.
2. **Factoring reduces what the weights encode, not how readably they encode it.** The residual MLP remains polysemantic; superposition is unaffected by a lighter workload. Structure factoring shrinks the haystack — the feature-level tools (SAEs, semantic interface, probing) still search it. The two programs compose; neither replaces the other.

## Relation to post-hoc parameter decomposition

The weight-side version of this idea exists as a young interpretability thread (attribution-based parameter decomposition, transcoders): split *trained* weights into mechanism-aligned components after the fact. Structure factoring is the stronger position when available — factor *before and during* training by architectural prescription, rather than fighting superposition after it has formed. Post-hoc methods remain the fallback for structure we failed to anticipate.

## The iteration loop (2026-06-11)

Structure factoring is not a bag of one-off tricks but a **closed improvement loop**:

1. **Hypothesize** (justifiably — from readouts, probes, or measured behavior, not aesthetics) what the model currently encodes implicitly in opaque weights.
2. **Factor** it out as explicit structure, per the design rule (interfaces / parallel channels).
3. **Measure** acceptance: learned mixing scalars, ablation deltas, matched controls, *registered predictions before results land*.
4. **Read back** the factored structure's learned parameters — because it is explicit, it is readable, and the readout generates the *next* hypothesis.

Step 4 is what makes the loop self-propelling, and the crawl arc is the first completed cycle: crawl factored positional structure into 21 readable logits → the K=3 readout showed coarse levels maxing out their windows (hypothesis: wants width) → prediction registered → K=5/9 confirmed (1.1287 → 1.1194) → the K=9 readout sharpened the hypothesis into a **scale-proportional-width (constant-Q) law** → which proposes its own refinement (per-level K ∝ 2^ℓ). Each pass through the loop produced both a capability gain and a mechanistic finding from the same artifact — the dual yield is the point.

**Grounding cautions for the trajectory:** (a) the "justifiably" in step 1 is load-bearing — the FWHT was also once a justified-seeming prior, and the protection against aesthetic factorings is registered predictions + matched controls, never conviction; (b) expect diminishing territory — each cycle peels the most legible remaining layer of structure, and the deep semantic content of the MLPs may admit no clean factorization; the loop's endpoint is precisely the boundary between what is statistics and what is genuinely learned computation, and *finding that boundary is itself the interpretability result*.

## Candidate future factorings (beyond skip-gram)

- Trigram / skip-n-gram positional tables at additional offsets (same recipe, more tables).
- The relational/positional embedding construction (dyadic-bucket PMI) — factoring type-level relational statistics into the input interface (see `reincorporate_large_semantic_embedding.md`).
- Unigram log-frequency bias at the head (classic, trivial, worth including for completeness in any sweep).
- Limited/key-relationships-only variants: deliberately partial factorings that capture only the highest-mass statistics — the goal is not completeness but maximal structure per prescribed parameter.
