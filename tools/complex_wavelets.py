"""Complex-valued lifting wavelet variants.

Motivation is **shift-invariance**, not "phase carries bonus information." A
critically-sampled real lifting wavelet (`LiftingWaveletDecompose` in
`model.py`) is shift-variant: the even/odd split is a fixed sampling lattice, so
the same token pattern at offset 4 vs 5 yields different coefficients, and the
learned predict/update nets cannot fully correct lattice-induced variance. The
classic remedy (Kingsbury dual-tree CWT) uses an analytic/complex representation
with approximately shift-invariant magnitude. See `plans/complex_wavelets.md`.

==============================================================================
DESIGN (see plan doc for full rationale)

- Complex values carried as TWO real C-wide tensors (re, im), never
  torch.complex64 (compile + fp16 friendliness; bf16 already regressed).
- Complex math contained to this module. Downstream FWHT / mixer / reconstruct
  in model.py are UNTOUCHED because the cascade COLLAPSES back to real width C
  before returning (approx, details). Emitting 2C would force the entire
  spectral stack to double — verified against model.py:1999-2033.
- Complex nonlinearity = CGELU (GELU on re and im independently).

CONSTRUCTIONS (config: complex_construction)
  "direct":   one lifting tree, complex predict/update. Imag input path
              zero-init so the variant approximately reduces to the real
              wavelet at init. Tests "can the model use phase capacity."
              [implemented here — increment 1]
  "dualtree": two real trees approximating a Hilbert pair → true approx
              shift-invariant magnitude. [increment 2 — NotImplementedError]

COLLAPSE MODES (config: complex_collapse) — how complex returns to real C:
  "per_level": each level emits a real coefficient; complex part is transient
               within that level's predict/update.
  "end":       re+im carried through the whole cascade; collapse to real C only
               at the final approx + each detail before return.

Collapse is NOT exactly invertible, so the complex variant loses perfect
reconstruction (acceptable: tested standalone, not composed with recurrence —
same mutual-exclusion as untied reconstruction). The reconstruct path therefore
gets its OWN (untied) update nets rather than reusing decompose's, since the
strict-inverse identity no longer holds.

Integration surface in model.py (increment 3, not yet wired):
    if config.get('wavelet_basis', 'real') == 'complex':
        from tools.complex_wavelets import build_complex_wavelet
        shared_lifting = build_complex_wavelet(config=config, ...)
==============================================================================
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_INV_SQRT2 = 0.7071067811865476
_SQRT2 = 1.4142135623730951


class ComplexLinear(nn.Module):
    """Complex-valued linear layer on split (re, im) real tensors.

    Computes (W_r + i W_i)(x_r + i x_i) + (b_r + i b_i):
        out_r = W_r x_r - W_i x_i + b_r
        out_i = W_r x_i + W_i x_r + b_i

    Two real Linear(in, out) submodules hold W_r/b_r and W_i/b_i. Param count
    is exactly 2x a real Linear(in, out) — this is the source of the complex
    variant's extra parameters, which the matched-param real control matches via
    hidden_mult.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.lin_r = nn.Linear(in_features, out_features, bias=bias,
                               device=device, dtype=dtype)
        self.lin_i = nn.Linear(in_features, out_features, bias=bias,
                               device=device, dtype=dtype)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_r = self.lin_r(x_r) - self.lin_i(x_i)
        out_i = self.lin_r(x_i) + self.lin_i(x_r)
        return out_r, out_i

    def init_haar_real_identity(self, scale: float = 1.0, imag_std: float = 0.01):
        """Real part -> scale*I; imag part -> small N(0, imag_std).

        The imag weights MUST be nonzero, not zero: with zero imag weights the
        imaginary path receives exactly zero gradient at init (every route
        input->imag->output passes through a zero imag weight), so it stays
        frozen at zero forever and the phase hypothesis is never tested
        (verified empirically, 2026-06-03). A small nonzero imag init unblocks
        gradient flow (verified). Trade-off: the layer no longer acts as the
        real Haar layer *exactly* at init — it's a small perturbation of it.
        Set imag_std=0.0 only if an exact-real-Haar init is explicitly wanted
        (and accept the dead imaginary path that comes with it)."""
        with torch.no_grad():
            nn.init.eye_(self.lin_r.weight)
            self.lin_r.weight.mul_(scale)
            if self.lin_r.bias is not None:
                nn.init.zeros_(self.lin_r.bias)
            if imag_std > 0:
                nn.init.normal_(self.lin_i.weight, std=imag_std)
            else:
                nn.init.zeros_(self.lin_i.weight)
            if self.lin_i.bias is not None:
                nn.init.zeros_(self.lin_i.bias)


def _cgelu(x_r: torch.Tensor, x_i: torch.Tensor
           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """CGELU: GELU applied to real and imaginary parts independently."""
    return F.gelu(x_r), F.gelu(x_i)


class _ComplexPredictUpdate(nn.Module):
    """Complex analog of the real lifting Sequential
    (Linear -> GELU -> Dropout -> Linear), operating on (re, im).

    At init, real parts = identity*scale and imag parts = small N(0, imag_std)
    (NOT zero — a zero imag init leaves the imaginary path with zero gradient
    and it never trains; see ComplexLinear.init_haar_real_identity). So the
    block starts as a small perturbation of the real Haar net, with a live
    imaginary path.
    """

    def __init__(self, C: int, hidden_mult: int = 1, dropout: float = 0.0,
                 haar_scale: float = 1.0, imag_std: float = 0.01,
                 device=None, dtype=None):
        super().__init__()
        hidden = C * hidden_mult
        self.fc1 = ComplexLinear(C, hidden, bias=True, device=device, dtype=dtype)
        self.fc2 = ComplexLinear(hidden, C, bias=True, device=device, dtype=dtype)
        self.drop = nn.Dropout(dropout)
        # Haar-style real init + small nonzero imag init (live imaginary path).
        self.fc1.init_haar_real_identity(1.0, imag_std=imag_std)
        self.fc2.init_haar_real_identity(haar_scale, imag_std=imag_std)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_r, x_i = self.fc1(x_r, x_i)
        x_r, x_i = _cgelu(x_r, x_i)
        x_r, x_i = self.drop(x_r), self.drop(x_i)
        x_r, x_i = self.fc2(x_r, x_i)
        return x_r, x_i


class ComplexLiftingWaveletDirect(nn.Module):
    """Direct complex lifting wavelet (one tree, complex predict/update).

    Mirrors LiftingWaveletDecompose's Split->Predict->Update cascade, but the
    predict/update nets are complex (ComplexLinear + CGELU). The real input
    embedding seeds the real channel; the imaginary channel starts at zero and
    is populated only by the complex predict/update math (zero-init imag weights
    mean it starts inert, so the variant approximately reduces to the real
    wavelet at init).

    Returns real (approx, details) — collapse back to width C happens per the
    `collapse` mode. Downstream FWHT/mixer never see the imaginary channel.
    """

    def __init__(self, levels: int, C: int, hidden_mult: int = 1,
                 dropout: float = 0.0, collapse: str = "per_level",
                 device=None, dtype=None):
        super().__init__()
        if collapse not in ("per_level", "end"):
            raise ValueError(f"collapse must be 'per_level'|'end', got {collapse!r}")
        self.levels = levels
        self.C = C
        self.collapse = collapse
        self.predict_nets = nn.ModuleList([
            _ComplexPredictUpdate(C, hidden_mult, dropout, haar_scale=1.0,
                                  device=device, dtype=dtype)
            for _ in range(levels)
        ])
        self.update_nets = nn.ModuleList([
            _ComplexPredictUpdate(C, hidden_mult, dropout, haar_scale=0.5,
                                  device=device, dtype=dtype)
            for _ in range(levels)
        ])
        # Collapse semantics differ by mode (deliberate contrast — see plan):
        #   "per_level": REAL-PART collapse. Imag is transient within a level;
        #     at init (imag=0) this == the real Haar wavelet EXACTLY, giving a
        #     clean-init ablation baseline.
        #   "end": MAGNITUDE collapse |z|=sqrt(re^2+im^2). This is the strongest
        #     tie to the shift-invariance theory (|z| is the approximately
        #     shift-invariant quantity), at the cost of NOT recovering the real
        #     wavelet at init: with imag=0 at init, |z|=|re| folds the sign of
        #     every coefficient. Expected and documented, not a bug.
        # No learned 2C->C projection in either mode (that would confound phase
        # with capacity — the matched-param control isolates capacity).
        self.inv_sqrt2 = _INV_SQRT2
        self._mag_eps = 1e-12  # inside-sqrt guard; magnitude computed in fp32

    def _collapse_real(self, z_r: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        return z_r

    def _collapse_mag(self, z_r: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        # fp32 square avoids fp16 underflow; eps inside sqrt bounds the gradient
        # near zero. Cast back to the input dtype for the downstream stack.
        mag = torch.sqrt(z_r.float() ** 2 + z_i.float() ** 2 + self._mag_eps)
        return mag.to(z_r.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        details: List[torch.Tensor] = []
        cur_r = x
        cur_i = torch.zeros_like(x)

        for level in range(self.levels):
            base_dilation = 1 << level
            T = cur_r.shape[1]

            def _odd(t):
                padded = F.pad(t, (0, 0, base_dilation, 0))
                return padded[:, :-base_dilation, :]

            odd_r, odd_i = _odd(cur_r), _odd(cur_i)
            even_r, even_i = cur_r, cur_i

            pred_r, pred_i = self.predict_nets[level](even_r, even_i)
            detail_r = (odd_r - pred_r) * self.inv_sqrt2
            detail_i = (odd_i - pred_i) * self.inv_sqrt2

            upd_r, upd_i = self.update_nets[level](detail_r, detail_i)
            approx_r = (even_r + upd_r) * self.inv_sqrt2
            approx_i = (even_i + upd_i) * self.inv_sqrt2

            if self.collapse == "per_level":
                # Real-part collapse now; drop imag from the carried state so the
                # next level starts real (imag is transient within the level).
                details.append(self._collapse_real(detail_r, detail_i))
                cur_r, cur_i = approx_r, torch.zeros_like(approx_r)
            else:  # "end": carry complex through; magnitude-collapse at return
                details.append((detail_r, detail_i))
                cur_r, cur_i = approx_r, approx_i

        if self.collapse == "end":
            details = [self._collapse_mag(dr, di) for (dr, di) in details]
            approx = self._collapse_mag(cur_r, cur_i)
        else:
            approx = cur_r
        return approx, details


class ComplexLiftingWaveletReconstruct(nn.Module):
    """Inverse path for the complex lifting wavelet.

    The collapse-to-real step is not exactly invertible, so this does NOT reuse
    decompose's nets (no strict-inverse identity). It owns its own real update
    nets and learns an approximate inverse. Operates entirely in real width C
    (the collapsed coefficients are real).
    """

    def __init__(self, levels: int, C: int, hidden_mult: int = 1,
                 dropout: float = 0.0, device=None, dtype=None):
        super().__init__()
        self.levels = levels
        self.sqrt2 = _SQRT2
        hidden = C * hidden_mult
        self.update_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C, hidden, device=device, dtype=dtype),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, C, device=device, dtype=dtype),
            )
            for _ in range(levels)
        ])
        # Haar-style init: update real-part 0.5*I so reconstruct ~ inverts the
        # real Haar update at init.
        for net in self.update_nets:
            nn.init.eye_(net[0].weight[:C, :])
            nn.init.zeros_(net[0].bias)
            nn.init.zeros_(net[3].weight)
            net[3].weight.data[:, :C] = 0.5 * torch.eye(
                C, device=net[3].weight.device, dtype=net[3].weight.dtype)
            nn.init.zeros_(net[3].bias)

    def forward(self, approx: torch.Tensor, details: List[torch.Tensor]
                ) -> torch.Tensor:
        cur = approx
        num_levels = len(details)
        for level in range(num_levels - 1, -1, -1):
            update = self.update_nets[level](details[level])
            cur = cur * self.sqrt2 - update
        return cur


class _RealLiftingTree(nn.Module):
    """Minimal real lifting tree (Split->Predict->Update), standalone analog of
    model.py's LiftingWaveletDecompose base path (no crawl/compression/structure
    — those aren't needed for the dual-tree complex construction). Kept in this
    module so dual-tree is testable without importing model.py.

    `phase` selects the level-0 lattice, both CAUSAL (never read the future —
    this is an autoregressive LM; a future-reading split leaks the label):
      "even": standard split, odd lattice = 1 step behind (odd[t]=cur[t-1]).
      "odd":  deeper-past split, odd lattice = 2 steps behind (odd[t]=cur[t-2]).
    The differing past offsets give the two trees distinct phase responses (the
    discrete, causal stand-in for the half-sample delay between the two trees of
    a dual-tree CWT), keeping im != re without ever looking ahead. Deeper levels
    use the standard split in both trees.
    """

    def __init__(self, levels: int, C: int, hidden_mult: int = 1,
                 dropout: float = 0.0, phase: str = "even",
                 device=None, dtype=None):
        super().__init__()
        if phase not in ("even", "odd"):
            raise ValueError(f"phase must be 'even'|'odd', got {phase!r}")
        self.levels = levels
        self.C = C
        self.phase = phase
        self.inv_sqrt2 = _INV_SQRT2
        hidden = C * hidden_mult

        def _net(haar_scale):
            net = nn.Sequential(
                nn.Linear(C, hidden, device=device, dtype=dtype), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, C, device=device, dtype=dtype))
            nn.init.eye_(net[0].weight[:C, :]); nn.init.zeros_(net[0].bias)
            nn.init.zeros_(net[3].weight)
            net[3].weight.data[:, :C] = haar_scale * torch.eye(
                C, device=net[3].weight.device, dtype=net[3].weight.dtype)
            nn.init.zeros_(net[3].bias)
            return net

        self.predict_nets = nn.ModuleList([_net(1.0) for _ in range(levels)])
        self.update_nets = nn.ModuleList([_net(0.5) for _ in range(levels)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        details: List[torch.Tensor] = []
        cur = x
        for level in range(self.levels):
            base_dilation = 1 << level
            T = cur.shape[1]
            if level == 0 and self.phase == "odd":
                # Causal phase offset: look TWO steps back instead of one
                # (odd[t]=cur[t-2]). Distinct phase response from the even tree
                # WITHOUT reading the future. Pad-back-2, slice — same causal
                # structure as the standard split, just a deeper offset.
                shift = 2 * base_dilation
                padded = F.pad(cur, (0, 0, shift, 0))
                odd = padded[:, :-shift, :]
            else:
                padded = F.pad(cur, (0, 0, base_dilation, 0))
                odd = padded[:, :-base_dilation, :]
            even = cur
            predicted = self.predict_nets[level](even)
            detail = (odd - predicted) * self.inv_sqrt2
            update = self.update_nets[level](detail)
            approx = (even + update) * self.inv_sqrt2
            details.append(detail)
            cur = approx
        return cur, details


class DualTreeComplexLiftingWavelet(nn.Module):
    """Dual-tree complex lifting wavelet: two real trees (A=real, B=imag)
    approximating a Hilbert pair via opposite level-0 phase. Returns the
    shift-invariant MAGNITUDE |z|=sqrt(A^2+B^2) per coefficient (real width C).

    Magnitude collapse is the ONLY mode: real-part collapse would discard tree B
    entirely, degenerating to a single real wavelet (enforced in the factory).
    ~2x the real wavelet's params (the +117M high-end in the plan).
    """

    def __init__(self, levels: int, C: int, hidden_mult: int = 1,
                 dropout: float = 0.0, device=None, dtype=None):
        super().__init__()
        self.levels = levels
        self.C = C
        self.tree_re = _RealLiftingTree(levels, C, hidden_mult, dropout,
                                        phase="even", device=device, dtype=dtype)
        self.tree_im = _RealLiftingTree(levels, C, hidden_mult, dropout,
                                        phase="odd", device=device, dtype=dtype)
        self._mag_eps = 1e-12

    def _mag(self, a, b):
        return torch.sqrt(a.float() ** 2 + b.float() ** 2 + self._mag_eps).to(a.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        approx_r, details_r = self.tree_re(x)
        approx_i, details_i = self.tree_im(x)
        approx = self._mag(approx_r, approx_i)
        details = [self._mag(dr, di) for dr, di in zip(details_r, details_i)]
        return approx, details


# =============================================================================
# Invertible construction (4th): tied complex decompose/reconstruct. Phase is
# mixed INSIDE the symmetry (by the spectral mixer, wired in model.py), not
# collapsed away. Unlike the collapse variants, this keeps
# Reconstruct∘Decompose = I — the symmetry the untied-reconstruction ablation
# showed is load-bearing. Decompose returns FULL complex coefficients; the
# block carries (re, im) through the mixer and calls the tied reconstruct.
# =============================================================================

class InvertibleComplexLiftingDecompose(nn.Module):
    """Complex lifting decompose that returns full complex (re, im) coefficients
    — NO collapse. Pairs with InvertibleComplexLiftingReconstruct (tied: reuses
    these same predict/update nets), giving exact Reconstruct∘Decompose = I.

    Returns (approx_r, approx_i, details) where details is a list of
    (detail_r, detail_i) complex pairs, coarsest-last (same order as the real
    LiftingWaveletDecompose).
    """

    def __init__(self, levels: int, C: int, hidden_mult: int = 1,
                 dropout: float = 0.0, imag_std: float = 0.01,
                 device=None, dtype=None):
        super().__init__()
        self.levels = levels
        self.C = C
        self.inv_sqrt2 = _INV_SQRT2
        self.predict_nets = nn.ModuleList([
            _ComplexPredictUpdate(C, hidden_mult, dropout, haar_scale=1.0,
                                  imag_std=imag_std, device=device, dtype=dtype)
            for _ in range(levels)
        ])
        self.update_nets = nn.ModuleList([
            _ComplexPredictUpdate(C, hidden_mult, dropout, haar_scale=0.5,
                                  imag_std=imag_std, device=device, dtype=dtype)
            for _ in range(levels)
        ])

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor = None):
        if x_i is None:
            x_i = torch.zeros_like(x_r)
        details = []
        cur_r, cur_i = x_r, x_i
        for level in range(self.levels):
            base = 1 << level
            def _odd(t):
                padded = F.pad(t, (0, 0, base, 0))
                return padded[:, :-base, :]
            odd_r, odd_i = _odd(cur_r), _odd(cur_i)
            even_r, even_i = cur_r, cur_i
            pr, pi = self.predict_nets[level](even_r, even_i)
            det_r = (odd_r - pr) * self.inv_sqrt2
            det_i = (odd_i - pi) * self.inv_sqrt2
            ur, ui = self.update_nets[level](det_r, det_i)
            cur_r = (even_r + ur) * self.inv_sqrt2
            cur_i = (even_i + ui) * self.inv_sqrt2
            details.append((det_r, det_i))
        return cur_r, cur_i, details


class InvertibleComplexLiftingReconstruct(nn.Module):
    """Tied inverse of InvertibleComplexLiftingDecompose. Reuses the decompose
    module's update_nets (no own params) — inverting only the update step
    recovers `even` exactly, mirroring the real LiftingWaveletReconstruct.

    Exact inverse holds because the lifting steps are triangular:
        approx = (even + U(detail))*inv_sqrt2  ->  even = approx*sqrt2 - U(detail)
    U is recomputed from the (unchanged) detail, so it cancels exactly. (The
    predict step is not inverted: `even` fully determines the recombination.)
    """

    def __init__(self, decompose: InvertibleComplexLiftingDecompose):
        super().__init__()
        self.decompose = decompose
        self.sqrt2 = _SQRT2

    def forward(self, approx_r, approx_i, details):
        cur_r, cur_i = approx_r, approx_i
        for level in range(len(details) - 1, -1, -1):
            det_r, det_i = details[level]
            ur, ui = self.decompose.update_nets[level](det_r, det_i)
            cur_r = cur_r * self.sqrt2 - ur
            cur_i = cur_i * self.sqrt2 - ui
            # Note: full lifting inverse would also undo the predict/split to
            # re-interleave even/odd. Here (as in the real reconstruct) we invert
            # the update to recover `even`; the model's forward is
            # decompose->mixer->reconstruct where the mixer acts on coefficients,
            # so this update-inverse is the exact partner of the decompose above
            # when details are unchanged (verified by the round-trip smoke test).
        return cur_r, cur_i


def mix_phase_split(z_r, z_i):
    """'split' mixer-activation helper — identity here; the actual mixing is the
    existing real mixer applied to the interleaved [re, im] 2C tensor in
    model.py. Provided for symmetry / explicitness. Split-complex == applying a
    real nonlinearity to re and im independently."""
    return z_r, z_i


def modulus_phase_gate(z_r, z_i, gate_mag, eps: float = 1e-6):
    """'modulus_phase' mixer-activation helper: scale the complex value's
    MAGNITUDE by a (real, nonnegative) gate while preserving its phase angle.
        z' = gate_mag * (z / (|z| + eps))   [direction preserved, magnitude set]
    gate_mag is the real mixer's output interpreted as the new magnitude. The
    eps-guarded unit phasor keeps gradients finite near |z|=0 (computed in fp32).
    """
    mag = torch.sqrt(z_r.float() ** 2 + z_i.float() ** 2 + eps * eps)
    inv = (1.0 / (mag + eps)).to(z_r.dtype)
    return gate_mag * z_r * inv, gate_mag * z_i * inv


def complex_magnitude(z_r, z_i, eps: float = 1e-12):
    """|z| in fp32 with inside-sqrt eps (for the modulus_phase mixer input)."""
    return torch.sqrt(z_r.float() ** 2 + z_i.float() ** 2 + eps).to(z_r.dtype)


def build_complex_wavelet(config: dict, levels: int, C: int,
                          device=None, dtype=None):
    """Factory mirroring tools/two_d_wavelets.build_lifting_wavelet_2d.

    Returns (decompose, reconstruct). Construction/collapse selected from
    config keys complex_construction / complex_collapse.
    """
    construction = config.get("complex_construction", "direct")
    collapse = config.get("complex_collapse", "per_level")
    hidden_mult = config.get("lifting_hidden_mult", 1)
    dropout = config.get("lifting_dropout", 0.0)

    if construction == "direct":
        dec = ComplexLiftingWaveletDirect(
            levels=levels, C=C, hidden_mult=hidden_mult, dropout=dropout,
            collapse=collapse, device=device, dtype=dtype)
        rec = ComplexLiftingWaveletReconstruct(
            levels=levels, C=C, hidden_mult=hidden_mult, dropout=dropout,
            device=device, dtype=dtype)
        return dec, rec
    elif construction == "dualtree":
        # Dual-tree is magnitude-collapse only — real-part collapse would discard
        # the imaginary tree, degenerating to a single real wavelet.
        if collapse not in ("end", "magnitude"):
            raise ValueError(
                f"dualtree requires complex_collapse='end' (magnitude); "
                f"got {collapse!r}. Real-part collapse degenerates dualtree to a "
                f"single real tree.")
        dec = DualTreeComplexLiftingWavelet(
            levels=levels, C=C, hidden_mult=hidden_mult, dropout=dropout,
            device=device, dtype=dtype)
        rec = ComplexLiftingWaveletReconstruct(
            levels=levels, C=C, hidden_mult=hidden_mult, dropout=dropout,
            device=device, dtype=dtype)
        return dec, rec
    elif construction == "invertible":
        # Tied complex decompose/reconstruct, full complex coefficients (no
        # collapse). Reconstruct reuses decompose's update_nets -> exact inverse.
        imag_std = config.get("complex_imag_std", 0.01)
        dec = InvertibleComplexLiftingDecompose(
            levels=levels, C=C, hidden_mult=hidden_mult, dropout=dropout,
            imag_std=imag_std, device=device, dtype=dtype)
        rec = InvertibleComplexLiftingReconstruct(dec)
        return dec, rec
    else:
        raise ValueError(
            f"complex_construction must be 'direct'|'dualtree'|'invertible', "
            f"got {construction!r}")


def param_count(config: dict, levels: int, C: int) -> int:
    """Exact trainable-param count of the complex variant at this config, so the
    matched-param real control can be sized via hidden_mult. Builds the module
    on the default device and counts — measured, never estimated (see plan)."""
    dec, rec = build_complex_wavelet(config, levels=levels, C=C)
    n = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    n += sum(p.numel() for p in rec.parameters() if p.requires_grad)
    return n


if __name__ == "__main__":
    # Smoke tests — run before any model.py wiring (increment 1 gate).
    torch.manual_seed(0)
    B, T, C, levels = 2, 64, 32, 4

    print("=== Complex wavelet smoke tests (direct construction) ===")
    for collapse in ("per_level", "end"):
        cfg = {"complex_construction": "direct", "complex_collapse": collapse,
               "lifting_hidden_mult": 1, "lifting_dropout": 0.0}
        dec, rec = build_complex_wavelet(cfg, levels=levels, C=C)
        dec.eval(); rec.eval()
        x = torch.randn(B, T, C)

        # 1) Shapes: approx and every detail must be real width C.
        approx, details = dec(x)
        assert approx.shape == (B, T, C), f"approx {approx.shape}"
        assert len(details) == levels, f"#details {len(details)}"
        for d in details:
            assert d.shape == (B, T, C), f"detail {d.shape}"
            assert d.dtype == x.dtype and not torch.is_complex(d)

        # 2) Reconstruction runs and returns real width C. Imag is now SMALL but
        #    NONZERO at init (live imaginary path), so neither mode is exactly
        #    identity at init — per_level is a small perturbation of real-Haar,
        #    end folds sign via magnitude. Assert finiteness/shape; report resid.
        xr = rec(approx, details)
        assert xr.shape == (B, T, C), f"recon {xr.shape}"
        assert torch.isfinite(xr).all(), "non-finite reconstruction"
        resid = (xr - x).abs().mean().item()
        if collapse == "per_level":
            # Small (imag_std=0.01 perturbation), not exact — bound loosely.
            assert resid < 0.2, f"per_level init residual unexpectedly large: {resid}"

        # 3) Forward is finite, AND gradients are finite (the magnitude collapse
        #    eps-guard exists precisely to keep grad finite near |z|=0 — verify
        #    rather than assume). Build a fresh train-mode pair, backprop a sum.
        dg, rg = build_complex_wavelet(cfg, levels=levels, C=C)
        xg = torch.randn(B, T, C, requires_grad=True)
        ag, dts = dg(xg)
        loss = rg(ag, dts).sum()
        loss.backward()
        assert xg.grad is not None and torch.isfinite(xg.grad).all(), \
            f"non-finite input grad ({collapse})"
        gp = [p.grad for p in list(dg.parameters()) + list(rg.parameters())
              if p.grad is not None]
        assert gp and all(torch.isfinite(g).all() for g in gp), \
            f"non-finite param grad ({collapse})"

        # 3b) REGRESSION GUARD: the imaginary path must receive NONZERO gradient
        #     at init. A zero imag init silently freezes the imaginary path and
        #     nulls the entire phase experiment (caught 2026-06-03). This asserts
        #     the small-imag-init fix stays in place.
        imag_grad = 0.0
        for net in list(dg.predict_nets) + list(dg.update_nets):
            for w in (net.fc1.lin_i.weight, net.fc2.lin_i.weight):
                if w.grad is not None:
                    imag_grad += w.grad.abs().sum().item()
        assert imag_grad > 0, \
            f"imaginary path got ZERO gradient at init ({collapse}) — dead phase path"

        n = param_count(cfg, levels=levels, C=C)
        note = "small perturbation of real-Haar" if collapse == "per_level" \
            else "magnitude (sign-fold) at init"
        print(f"  collapse={collapse:9s}  OK  | init recon residual {resid:.2e} "
              f"[{note}] | imag |grad|={imag_grad:.1f} | params(C={C},L={levels})={n:,}")

    # 4) Dual-tree construction (increment 2): magnitude-collapse only.
    print("=== Dual-tree construction (increment 2) ===")
    cfg_dt = {"complex_construction": "dualtree", "complex_collapse": "end",
              "lifting_hidden_mult": 1, "lifting_dropout": 0.0}
    dec, rec = build_complex_wavelet(cfg_dt, levels=levels, C=C)
    dec.eval(); rec.eval()
    x = torch.randn(B, T, C)
    approx, details = dec(x)
    assert approx.shape == (B, T, C) and len(details) == levels
    for d in details:
        assert d.shape == (B, T, C) and not torch.is_complex(d)
    assert torch.isfinite(approx).all() and all(torch.isfinite(d).all() for d in details)
    # Magnitude is nonnegative by construction — verify.
    assert (approx >= 0).all() and all((d >= 0).all() for d in details), \
        "magnitude collapse must yield nonnegative coefficients"
    xr = rec(approx, details)
    assert xr.shape == (B, T, C) and torch.isfinite(xr).all()

    # Two trees must differ (distinct phase) — else im is a copy of re and the
    # construction is pointless. Check tree outputs aren't identical.
    ar, _ = dec.tree_re(x); ai, _ = dec.tree_im(x)
    assert (ar - ai).abs().mean().item() > 1e-4, "trees identical — phase split inert"

    # CAUSALITY GUARD: NO position's output may depend on a FUTURE input. This
    # is an autoregressive LM — a future-reading split leaks the label and the
    # model trains to a deceptively low loss it can't reproduce at generation.
    # The odd-phase tree had exactly this bug (forward shift); caught 2026-06-03.
    # Test every construction/phase: perturb a strictly-future token, assert no
    # earlier output position moves.
    print("=== Causality guard (all trees must be strictly causal) ===")
    for label, builder in [
        ("real-even", lambda: _RealLiftingTree(levels, C, phase="even")),
        ("real-odd",  lambda: _RealLiftingTree(levels, C, phase="odd")),
        ("direct",    lambda: build_complex_wavelet(
            {"complex_construction": "direct", "complex_collapse": "end"}, levels, C)[0]),
        ("dualtree",  lambda: build_complex_wavelet(
            {"complex_construction": "dualtree", "complex_collapse": "end"}, levels, C)[0]),
    ]:
        m = builder(); m.eval()
        xc = torch.randn(1, T, C)
        _, dd = m(xc)
        t_probe = T // 2
        base = dd[0][0, t_probe].clone()
        worst = 0.0
        for dt in range(1, T - t_probe):  # every strictly-future position
            x2 = xc.clone(); x2[0, t_probe + dt] += 100.0
            _, dd2 = m(x2)
            worst = max(worst, (dd2[0][0, t_probe] - base).abs().max().item())
        assert worst < 1e-6, f"{label}: FUTURE LEAK — detail[t] moved {worst:.4f} on future perturb"
        print(f"  {label:9s} causal (max future-perturb leak {worst:.2e})")

    # Gradient finiteness through the dual-tree magnitude.
    dg, rg = build_complex_wavelet(cfg_dt, levels=levels, C=C)
    xg = torch.randn(B, T, C, requires_grad=True)
    ag, dts = dg(xg)
    rg(ag, dts).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all(), "non-finite dualtree grad"
    n_dt = param_count(cfg_dt, levels=levels, C=C)
    print(f"  dualtree   OK  | nonneg magnitude, trees differ, grads finite | "
          f"params(C={C},L={levels})={n_dt:,}")

    # 5) Degenerate config rejection: dualtree + per_level must raise.
    try:
        build_complex_wavelet(
            {"complex_construction": "dualtree", "complex_collapse": "per_level"},
            levels=levels, C=C)
        raise AssertionError("dualtree+per_level should have raised ValueError")
    except ValueError:
        print("  dualtree+per_level -> ValueError (correct; degenerate)")

    # 6) Invertible construction (increment 4): tied complex decompose/recon.
    print("=== Invertible construction (increment 4) ===")
    cfg_inv = {"complex_construction": "invertible", "lifting_hidden_mult": 1,
               "lifting_dropout": 0.0}
    dec, rec = build_complex_wavelet(cfg_inv, levels=levels, C=C)
    dec.eval(); rec.eval()
    x = torch.randn(B, T, C)

    # 6a) ROUND-TRIP IDENTITY: Reconstruct(Decompose(x)) == x with coefficients
    #     unchanged. This is the whole point — the symmetry the untied ablation
    #     showed is load-bearing. Must hold to high precision for ALL weights
    #     (not just at init), so perturb weights first, then check.
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ar, ai, dts = dec(x)
    xr, xi = rec(ar, ai, dts)
    rt_resid = (xr - x).abs().max().item()
    assert rt_resid < 1e-3, f"invertible round-trip FAILED: max resid {rt_resid}"
    # imaginary part of the reconstruction should also return to ~0 (x was real)
    xi_resid = xi.abs().max().item()
    assert xi_resid < 1e-3, f"invertible round-trip imag not ~0: {xi_resid}"

    # 6b) Causality (3-tuple return — own loop). Perturb future, check detail[t].
    dC = build_complex_wavelet(cfg_inv, levels=levels, C=C)[0]; dC.eval()
    xc = torch.randn(1, T, C)
    _, _, dd = dC(xc)
    t_probe = T // 2
    base = dd[0][0][0, t_probe].clone()  # detail[0] real part
    worst = 0.0
    for dt in range(1, T - t_probe):
        x2 = xc.clone(); x2[0, t_probe + dt] += 100.0
        _, _, dd2 = dC(x2)
        worst = max(worst, (dd2[0][0][0, t_probe] - base).abs().max().item())
    assert worst < 1e-6, f"invertible: FUTURE LEAK {worst:.4f}"

    # 6c) modulus_phase gate: finite forward + grad near |z|=0 (eps-guard works).
    zr = torch.zeros(2, 4, C, requires_grad=True)
    zi = torch.zeros(2, 4, C, requires_grad=True)
    gate = torch.rand(2, 4, C)
    gr, gi = modulus_phase_gate(zr, zi, gate)
    (gr.sum() + gi.sum()).backward()
    assert torch.isfinite(gr).all() and torch.isfinite(gi).all(), "modulus_phase non-finite at |z|=0"
    assert torch.isfinite(zr.grad).all() and torch.isfinite(zi.grad).all(), "modulus_phase grad non-finite at |z|=0"

    # 6d) gradient flow through the full invertible decompose+reconstruct.
    dg, rg = build_complex_wavelet(cfg_inv, levels=levels, C=C)
    xg = torch.randn(B, T, C, requires_grad=True)
    ar, ai, dts = dg(xg)
    (rg(ar, ai, dts)[0].sum()).backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all(), "invertible grad non-finite"
    n_inv = param_count(cfg_inv, levels=levels, C=C)
    print(f"  invertible OK  | round-trip resid {rt_resid:.2e} (imag {xi_resid:.2e}), "
          f"causal (leak {worst:.0e}), modulus_phase grad finite | "
          f"params(C={C},L={levels})={n_inv:,}")

    print("All increment-1 + 2 + 4 smoke tests passed.")
