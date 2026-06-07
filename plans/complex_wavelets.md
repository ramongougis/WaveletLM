# Complex Wavelets: Shift-Invariance via Analytic Representation

A complex-valued lifting wavelet variant whose motivation is **shift-invariance**,
not "phase carries bonus information." The README "Complex Wavelets" section
states the rationale and the required two-run test; this doc is the
implementation plan.

Tested immediately after the minimal dropout/weight-decay finetune (per user
direction — not deferred to post-release), with the explicit caveat that the
finetune is shallow (L=1) and lacks layer depth, so a flat result here does not
rule out complex wavelets at higher L.

---

## Background: why the real lifting wavelet is shift-variant

`LiftingWaveletDecompose` ([model.py:323](../model.py)) splits each level into
even/odd lattices (Split → Predict → Update). The split is a **fixed sampling
operation**: the same token pattern beginning at position 4 vs position 5 lands
on opposite lattices and produces materially different coefficient patterns.
The learned predict/update networks **cannot fully correct this** — the variance
comes from the sampling lattice, not the filter weights. `wavelet_crawl`
(learned ±1 dilation per level) buys *partial* shift-robustness, which is why it
helped (−0.0037 at T4), but it does not make the representation shift-invariant.

The classic fix (Kingsbury's dual-tree complex wavelet transform) is an
analytic/complex representation with **approximately shift-invariant magnitude**;
phase encodes *where within the subband* a feature sits. For language, the gap
this closes is concrete: every constituent / n-gram recurring at different
offsets currently forces the mixer to learn multiple shifted signatures of the
same structure — the positional-binding / agreement-attraction weakness where
WaveletLM is structurally weakest vs attention.

---

## Design (as built — invertible, tied)

1. **Two real tensors (re, im), not `torch.complex64`.** Complex values are
   carried as two real C-wide tensors; all complex arithmetic
   (`(Wᵣ+iWᵢ)(xᵣ+ixᵢ)`) is explicit on the halves. Rationale: `torch.compile` +
   fp16 handle `complex64` poorly, and bf16 already regressed (`feedback_no_bf16`).

2. **Invertible, tied.** Decompose returns FULL complex `(approx_r, approx_i,
   details)`; reconstruct (`InvertibleComplexLiftingReconstruct`) **reuses
   decompose's update nets**, so `Reconstruct∘Decompose = I` holds exactly. This
   is the load-bearing property: the [untied reconstruction](../README.md)
   ablation showed breaking that symmetry (+117.5M params) *regressed*. So the
   complex wavelet keeps the symmetry and mixes phase, rather than collapsing it.

3. **Complex nonlinearity = CGELU** (GELU on re and im independently) — standard,
   compile/fp16-stable.

4. **Phase mixing in the spectral stack** (`complex_mixer_activation`, applied in
   model.py `_forward_complex_invertible`):
   - **`split`** — the existing real spectral stack runs on re and im
     independently (split-complex; no re/im cross-coupling). Cheapest.
   - **`modulus_phase`** — gate the magnitude `|z|` (the shift-invariant
     quantity) with a **non-negative (softplus)** gate, re-applying the preserved
     unit phase. Softplus is required: without it the real spectral stack can emit
     a negative gate, flipping sign on both parts = an unintended π phase rotation.

5. **Block forward** (`_forward_complex_invertible`, model.py): isolated branch,
   early-dispatched — the real forward is untouched. Carries (re, im) through the
   spectral stack, takes the **real part only at the block output**. mixer_depth=1
   required (single-application spectral path).

6. **Matched-param control.** A complex win over T4 is uninterpretable (params
   help monotonically), so each complex run is paired with a real-wavelet control
   widened via `lifting_hidden_mult` to the same param count, and is validated
   only if it beats *that control*. The invertible wavelet is 469.99M
   (complex predict AND update, tied reconstruct → ~745M total); matched real
   control = `hidden_mult=4` (469.91M wavelet, ratio 1.000 — near-exact). All arms
   run `wavelet_crawl=false` (hard-errored otherwise), so the only cross-arm
   variable is the basis; the in-section reference is the control, not crawl-on T4.

Config: `wavelet_basis: "real"|"complex"`, `complex_mixer_activation:
"split"|"modulus_phase"`. Mutually exclusive (hard-errored) with recurrence,
mixer_depth>1, untied reconstruction, multi-basis, 2D wavelet, non-shared
lifting, and wavelet_crawl.

## Verification (smoke tests in tools/complex_wavelets.py `__main__`)

- **Round-trip identity** `Reconstruct∘Decompose=I` to ~1e-6 with off-init
  weights (not just at init) — the symmetry holds for trained weights.
- **Causality** — perturbing a strictly-future token moves no earlier output
  position (autoregressive safety; a future-reading split would leak the label).
- **modulus_phase non-negativity** — a negative gate does not flip coefficient
  sign (softplus guard), and the gate is finite + has finite gradient at |z|=0.
- **Imaginary path liveness** — imag weights receive nonzero gradient at init
  (small nonzero imag init; a zero init silently freezes the phase path).
- End-to-end (model.py): real-basis regression-safe, both activations
  build/fwd/bwd with finite grads, removed constructions error cleanly.

## NaN fix: coupled ComplexLayerNorm + phasor eps (2026-06-07)

First invertible runs NaN'd (split: intermittent from step ~25.7k/lr≈0.02, fought
through; modulus_phase: persistent from ~15.5k, dominated). Two distinct causes,
both fixed:

1. **Joint-magnitude drift (both activations).** The spectral pass applied the
   per-part real `LayerNorm` to re and im *independently*, so each looked
   normalized while their JOINT magnitude — the quantity the tied √2-per-level
   reconstruct amplifies — drifted until fp16 overflow. Fix: `ComplexLayerNorm`
   (`tools/complex_wavelets.py`) whitens the 2×2 (re,im) covariance jointly
   (Trabelsi-style complex LN), in fp32 with an eps FLOOR on the determinant
   (1/√det explodes for near-degenerate covariance). It reduces *exactly* to real
   LayerNorm when im=0 (verified ~7e-7), so it is a safe drop-in. Wired only into
   a complex-only `_spectral_stack_complex` (and the modulus_phase coeff pre-norm)
   via per-scale `decomp_norms_complex`/`recon_norms_complex`, built only when
   complex is active — the real path's norms and `_spectral_stack` are untouched
   (verified: real loss identical pre/post).
2. **Phasor division (modulus_phase only).** `1/(|z|+ε)` with ε=1e-6 (and ε² inside
   the sqrt) underflows at fp16 for small |z|. Fix: ε=1e-3, so ε² inside the sqrt
   floors |z| at 1e-3 (the magnitude squares the guard); divisor is the floored
   mag, no double `+ε`. fp32 throughout.

Stress-checked: 60 Adagrad steps at lr=0.0225 (the LR that NaN'd) finite for both
activations. Note this is NOT param-count parity with the real model — the
invertible complex wavelet is ~4× the real wavelet (complex predict AND update),
so it is a larger/higher-variance model; if NaNs recur at scale, lowering LR
(toward 0.01) remains the cheap fallback before further structural change.

## Open questions (for a fuller implementation — not yet built)

- **Per-block phase reset (#1 from the impl review).** The block takes real x and
  returns real x, so the imaginary channel reseeds at zero each block — phase does
  not propagate *across* depth. A deep complex hierarchy would require carrying
  (re, im) through the residual stream / embedding / head. Large redesign; open.
- **Real-only spectral mixers (#4).** `split` runs two independent real streams
  with no re/im cross-coupling; `modulus_phase` couples only through magnitude. A
  genuinely complex mixer (complex linear with cross terms) would enable learned
  phase rotations. Open; gauge against whether the cheap versions show any signal.
- **Shallow-finetune caveat.** L=1/1-epoch surface is flat; a null result here
  does not rule out complex wavelets at the final layer count.

## tanh gate for modulus_phase (2026-06-07)

`modulus_phase_gate` takes a `gate` arg: `softplus` (default, strictly positive,
magnitude-only) or `tanh`. tanh is BOUNDED (|g|≤1 — a stabilizer; cannot drive the
magnitude explosion behind the complex NaNs) and BIPOLAR (g<0 = learnable discrete
π phase flip — expressivity softplus lacks). Amplification >1 is supplied
elsewhere (scale_weights, proj_out), so the bound costs little. Init-shifted by
`atanh(ln2)≈0.854` so g starts at ln2 (= softplus's init value): begins as
standard positive scaling, learns into flips — no chaotic π rotations on the first
backward pass. Config `complex_gate_activation`. Run: T4_cwav_inv_modphase_tanh.

## Option C: complex in the MIXER, real wavelets (2026-06-07)

The wavelet was the wrong home for complex — its invertibility/causality fought
the complex machinery and the cheap version regressed (+0.0189 vs T4 for
split at L=1). The mixer operates in spectral space where phase is natural, so
move complex THERE and keep the wavelet real. Chosen form: **learnable
real→complex projection (Gemini's Option C), per-scale, with a tied/learned
inverse** — preferred over a fixed transform (Option A, custom CHT kernel) or a
causal Hilbert filter (Option B, reintroduces the causal-Hilbert ambiguity that
sank the dual-tree) because it (a) reuses the existing real FWHT untouched,
(b) sidesteps every constraint that sank the complex wavelet (no invertibility
coupling, no causality issue — it's a per-token channel projection), and (c) is
the honest test of "does a complex spectral representation help" without imposing
a possibly-wrong fixed mapping.

Flow (per layer, real wavelet unchanged):
1. Real wavelet decompose → real coeffs `[B,T,S,Cp]`.
2. **Per-scale up-projection** `Linear(Cp→2Cp)` → (re, im); init scaled by 1/√2
   so magnitude into the FWHT matches the real baseline. ComplexLayerNorm after.
3. Real FWHT on re and im separately → complex spectrum.
4. Complex gated spectral mixer (re/im), `complex_mixer_activation` reused.
5. Inverse FWHT on each part.
6. **Per-scale down-projection** `Linear(2Cp→Cp)` → real coeffs.
7. Real wavelet reconstruct (unchanged, still exactly invertible — the real
   wavelet's `Reconstruct∘Decompose=I` is untouched; the complex machinery lives
   entirely between decompose and reconstruct, on the coefficients).

Decisions (locked): per-scale projections (not one shared across scales — scales
carry different structure); ComplexLayerNorm reused (not wavelet-specific); the
real wavelet stays the default real path, so this is gated on a config flag.
Matched-param control: a real model widened to Option C's param count (the
projections add ~2·Cp² per scale per layer). Same crawl-off / vs-control framing.
Config: `complex_mixer_complex` (or reuse `wavelet_basis` semantics — TBD at
implementation). Mutually exclusive with the complex *wavelet* basis (they're two
different homes for the same idea; don't stack).
**Status: DESIGNED, not built** — implement after the current split/modphase/
control runs report, so the wavelet-vs-mixer comparison has both numbers.
