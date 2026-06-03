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

## Design decisions (locked)

1. **Two real tensors (re, im), NOT native `torch.complex64`.** Complex values
   are carried as two real `C`-wide tensors. `torch.compile` + fp16 handle
   `complex64` poorly; the project's fp16 path is load-bearing and bf16 already
   regressed (see `feedback_no_bf16`). All complex arithmetic — the
   `(Wᵣ+iWᵢ)(xᵣ+ixᵢ)` multiplies — is implemented explicitly on the real/imag
   halves inside `tools/complex_wavelets.py`, contained to that module.

2. **CRITICAL integration correction (verified, supersedes the README's
   "mixer consumes 2C unchanged").** Wavelet coefficients are width `C`. They
   feed `decomp_norms[s]` (LayerNorm sized `C`), the FWHT (needs power-of-2
   width), and each per-scale mixer (sized `C`) — see [model.py:1999-2033](../model.py).
   Emitting `2C` would force the **entire spectral stack** (norms + FWHT +
   every mixer) to double, which is invasive and doubles mixer compute. So the
   complex cascade **must collapse back to real width `C` before returning**
   `(approx, details)`. Complex math lives strictly inside the lifting cascade;
   downstream (FWHT, mixer, reconstruct) is untouched and mixer cost stays flat.

3. **Complex nonlinearity = CGELU.** The predict/update nets are
   `Linear→GELU→Dropout→Linear`. The complex variant applies the complex-linear
   rule to the Linear layers and **GELU to real and imaginary parts
   independently** (CGELU) — the standard, compile-safe, fp16-stable convention.
   (Alternatives — modGELU on magnitude, etc. — are lossier or less stable;
   noted but not used.)

4. **Matched-param control widens the real predict/update hidden dim
   (`hidden_mult`)** — the same knob complex effectively doubles, NOT `low_rank`
   or width elsewhere. `param_count()` sizes the control to the complex variant
   exactly.

## Design matrix (2 constructions × 2 collapse modes, each vs its own control)

Config flags: `wavelet_basis: "real"|"complex"`, `complex_construction:
"direct"|"dualtree"`, `complex_collapse: "per_level"|"end"`.

- **Construction "direct":** one lifting tree; predict/update are complex
  (CGELU). Imaginary input path zero-initialized so the variant approximately
  reduces toward the real wavelet at init. Tests "can the model *use* phase
  capacity." Cheaper (~+40–60M, tunable). **Increment 1 (this build).**
- **Construction "dualtree":** two real lifting trees approximating a Hilbert
  pair → true approximately-shift-invariant magnitude. Faithful to the Kingsbury
  rationale, strongest structural claim. ~+117M. The Hilbert-pair coupling is
  the hard part. **Increment 2 (next build).**
- **Collapse "per_level":** REAL-PART collapse; complex part is transient within
  a level. Starts as a small perturbation of the real Haar wavelet — the
  clean-init ablation baseline.
- **Collapse "end":** re+im carried through the whole cascade, then MAGNITUDE
  collapse `|z|=sqrt(re²+im²)` at return. Strongest tie to the shift-invariance
  theory (|z| is the approximately shift-invariant quantity). Magnitude computed
  in fp32 with an inside-sqrt eps (1e-12) for finite gradients near zero.

per_level = real-part (clean init), end = magnitude (theory-max). No learned
2C→C projection in either — that would confound phase with capacity, which the
matched-param control exists to isolate.

**Reconstruction caveat.** The real lifting wavelet has *perfect*
reconstruction (`Reconstruct∘Decompose = I`), which is what the mixer-recurrence
mechanism relies on. Collapsing complex→real is **not exactly invertible**, so
the complex variant loses perfect reconstruction. This is acceptable because
(a) complex wavelets are tested **standalone**, not composed with recurrence
(same mutual-exclusion as untied reconstruction), and (b) the forward pass
`decompose → FWHT → mixer → iFWHT → reconstruct → x'` still trains as a learned
encoder/decoder.

---

## Implementation surface

### `tools/complex_wavelets.py` (~300–400 lines, new)

Mirrors the `tools/two_d_wavelets.py` precedent (factory + `nn.Module`, lazy
import from `model.py` to avoid a circular import):

- `class LiftingWaveletComplex(nn.Module)` — parallels
  `LiftingWaveletDecompose`/`LiftingWaveletReconstruct`. Predict/update nets
  operate on the interleaved `2C` tensor; complex multiply implemented on the
  real/imag halves. Returns the same `(approx, details)` interface (each now
  `2C`-wide) so downstream is transparent.
- `def build_complex_wavelet(config, ...)` — factory matching the
  `build_lifting_wavelet_2d(...)` signature.
- `def param_count(config) -> int` — **prerequisite helper.** Returns the exact
  param count of the complex variant at a given config so the matched-param real
  control can be sized to it. This is what resolves the README's TBD cost cells
  with a *measured* number rather than the inconsistent "+25M vs 2×" estimates
  currently flagged there. **Run this and record the number before launching
  either training run.**

### `model.py` (~15–20 lines)

One dispatch flag at construction, paralleling the existing `wavelet_2d_mode`
swap ([model.py / two_d_wavelets.py:62-64](../tools/two_d_wavelets.py)):

```python
if config.get('wavelet_basis', 'real') == 'complex':
    from tools.complex_wavelets import build_complex_wavelet
    shared_lifting = build_complex_wavelet(config=config, ...)
```

The interleaved `2C` tensor flows into the existing FWHT + per-scale mixer
unchanged. Reconstruct path mirrors decompose (tied, consistent with current
default; untied-reconstruction is a separate, mutually-exclusive axis).

### `config.json`

Add `"wavelet_basis": "real"` (default preserves current behavior exactly).

---

## Cost (to be VERIFIED, not asserted)

The lifting stage is **117.50M params** (verified, T4 breakdown
[log](../logs/wikitext-103_2026-05-24_19-22-19/log.txt)). Full complex ≈ 2× the
lifting nets ≈ +117M; interleaving with shared structure is cheaper. The README
section flags that the prior "+25M" and "2× across the board" estimates are
mutually inconsistent. **`param_count()` resolves this before any run.** Compute
is ~2× on the wavelet stage regardless; mixer stage ~constant under interleaving.

---

## Test protocol (required two-run form)

A bare complex run beating T4 is **uninterpretable** — params help monotonically
in this project, so a win could be capacity, not phase. Validation requires:

1. **Complex wavelet** at T4 (1 epoch, `wavelet_basis="complex"`), `wavelet_crawl`
   left in its current T4 state so the test isolates what complex adds *on top of*
   crawl's partial shift-robustness.
2. **Real control**, param-matched to (1) via `hidden_mult`, `wavelet_crawl` on.

**Decision rule:** complex is validated only if it beats the **matched-param real
control** (not merely T4) by > 0.0015 nats best val / ~0.0010 BPB sliding. If it
only beats T4 but ties the control, the gain was capacity, not shift-invariance —
reject. Rank by BPB sliding per project convention (val understates
context-exploiting configs).

Tables for both runs are stubbed in the README "Complex Wavelets" section.

---

## Implementation status

- **Increment 1 — DONE.** `tools/complex_wavelets.py`, standalone (no model.py
  dependency). Direct construction, both collapse modes, `ComplexLinear`, CGELU,
  factory, `param_count()`, `__main__` smoke tests (shapes, finite forward+grad,
  imaginary-path gradient flow, causality).
- **Increment 2 — DONE.** Dual-tree construction
  (`complex_construction="dualtree"`): two real lifting trees (A=real, B=imag)
  with a distinct causal level-0 phase offset, returning shift-invariant
  magnitude `|z|=sqrt(A²+B²)`. Magnitude collapse only — `dualtree`+`per_level`
  raises ValueError (real-part collapse would discard tree B → degenerate to a
  single real wavelet). Self-contained `_RealLiftingTree` (no model.py import).
  Direct and dualtree have equal param counts (both ~2× a single real wavelet);
  the "+117M / 2×" figure is vs the single real wavelet.
- **Increment 3 — DONE.** Wired into model.py: model-level dispatch on
  `wavelet_basis` (mirrors the 2D-wavelet swap), complex decompose + separate
  shared reconstruct passed to blocks via `shared_lifting_reconstruct`; block
  duck-types the complex path and skips real reconstruct derivation. Config keys
  added (`wavelet_basis`/`complex_construction`/`complex_collapse`, defaults
  preserve real behavior). Hard-error mutual exclusion with recurrence, untied
  reconstruction, multi-basis, 2D wavelet, non-shared lifting, off-diag
  structure, and `wavelet_crawl` (complex trees don't implement crawl — erroring
  prevents silently breaking the matched comparison). Measured params at T4
  scale (C=2048/levels=7): complex wavelet 293.74M (dec+rec) vs real 117.50M →
  total 569.25M vs T4 393.01M. Matched real control = `lifting_hidden_mult=3`
  (~628M, slightly over — conservative). All arms run `wavelet_crawl=false` so
  the only cross-arm variable is the basis; in-section reference is the real
  control, not the crawl-on T4. runs.sh: CW1–CW4. Verified: real-basis
  regression-safe, all complex arms build/fwd/bwd, all guards fire.

## Open questions / risks

- **Interleaved complex-multiply cost.** The explicit `(a+bi)(c+di)` is 4 real
  matmuls vs 1; if this dominates, consider the Gauss 3-multiply trick. Profile
  before optimizing.
- **Shallow-finetune caveat.** L=1 may not surface a shift-invariance benefit
  that only pays off with depth; record this so a flat result isn't over-read.
