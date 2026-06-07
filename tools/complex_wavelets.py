"""Complex-valued lifting wavelet (invertible construction).

Motivation is **shift-invariance**, not "phase carries bonus information." A
critically-sampled real lifting wavelet (`LiftingWaveletDecompose` in
`model.py`) is shift-variant: the even/odd split is a fixed sampling lattice, so
the same token pattern at offset 4 vs 5 yields different coefficients, and the
learned predict/update nets cannot fully correct lattice-induced variance.

DESIGN
- Complex values carried as TWO real C-wide tensors (re, im), never
  torch.complex64 (compile + fp16 friendliness; bf16 already regressed).
- Complex math contained to this module. Complex nonlinearity = CGELU (GELU on
  re and im independently).
- **Invertible, tied:** decompose returns FULL complex (re, im) coefficients;
  reconstruct reuses decompose's update nets, so Reconstruct∘Decompose = I is
  preserved exactly (the symmetry the untied-reconstruction ablation showed is
  load-bearing). Phase is carried through the spectral mixer and reconstructed,
  NOT collapsed to magnitude. The block's forward keeps (re, im) through the
  spectral stack and takes the real part only at the final block output.

Phase mixing in the spectral stack (config: complex_mixer_activation, applied
in model.py `_forward_complex_invertible`):
  "split":         the real spectral stack runs on re and im independently
                   (split-complex). Cheapest; no re/im cross-coupling.
  "modulus_phase": gate the magnitude |z| (the shift-invariant quantity) and
                   re-apply the preserved unit phase. The gate is forced
                   non-negative (softplus) so it scales magnitude without
                   flipping sign / injecting spurious π phase rotations.

Integration surface in model.py:
    if config.get('wavelet_basis', 'real') == 'complex':
        from tools.complex_wavelets import build_complex_wavelet
        decompose, reconstruct = build_complex_wavelet(config, levels, C)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_INV_SQRT2 = 0.7071067811865476
_SQRT2 = 1.4142135623730951


class ComplexLayerNorm(nn.Module):
    """LayerNorm for a complex value carried as two real tensors (x_r, x_i).

    Normalizes re and im **jointly** as a coupled 2D quantity via covariance
    whitening (Trabelsi et al. 2018, "Deep Complex Networks", complex BN/LN):
    center, then multiply by the inverse square root of the 2x2 (rr, ri; ri, ii)
    covariance so the output has identity covariance, then a complex affine
    (gamma, beta). This is the coupling the per-part real LayerNorm lacks: it
    bounds the *joint* magnitude, not re and im independently — the magnitude
    that the tied reconstruct amplifies over levels.

    Why this and not the obvious complex64 version:
    - Operates on REAL (x_r, x_i) tensors — NEVER torch.complex64 (the whole
      module avoids it: torch.compile + fp16 handle complex dtypes poorly).
    - Whitening is computed in fp32 with an eps FLOOR ON THE DETERMINANT, because
      1/sqrt(det) explodes for near-degenerate covariance (tiny/early variance,
      fp16) — the same failure class as the modulus_phase phasor.

    Reduces to standard real LayerNorm when x_i == 0: then Vii→eps, Vri→0, the
    whitening collapses to (x_r - mean_r)/sqrt(Vrr) on the real part. So it is a
    safe drop-in for the real LayerNorm at points where both parts are in hand.

    normalized_shape: the feature dim C (last axis), matching nn.LayerNorm(C).
    """

    def __init__(self, normalized_shape, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        # Complex affine as separate real params (gamma_r, gamma_i, beta_r,
        # beta_i). gamma init: real=1, imag=0 (identity scale at init).
        self.gamma_rr = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        self.gamma_ii = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        self.gamma_ri = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype))
        self.beta_r = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype))
        self.beta_i = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype))

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor):
        ndims = len(self.normalized_shape)
        dim = tuple(range(-ndims, 0))
        in_dtype = x_r.dtype
        xr = x_r.float()
        xi = x_i.float()
        eps = self.eps

        mr = xr.mean(dim=dim, keepdim=True)
        mi = xi.mean(dim=dim, keepdim=True)
        xr = xr - mr
        xi = xi - mi

        Vrr = xr.pow(2).mean(dim=dim, keepdim=True) + eps
        Vii = xi.pow(2).mean(dim=dim, keepdim=True) + eps
        Vri = (xr * xi).mean(dim=dim, keepdim=True)

        # Inverse sqrt of the 2x2 covariance via the closed form, with the
        # determinant FLOORED so 1/(s*t) cannot explode (fp16-degenerate cov).
        det = (Vrr * Vii - Vri.pow(2)).clamp_min(eps * eps)
        s = torch.sqrt(det)
        t = torch.sqrt(Vrr + Vii + 2.0 * s).clamp_min(eps)
        inv_st = 1.0 / (s * t)
        Wrr = (Vii + s) * inv_st
        Wii = (Vrr + s) * inv_st
        Wri = -Vri * inv_st

        nr = Wrr * xr + Wri * xi
        ni = Wri * xr + Wii * xi

        # Complex affine: (gamma_rr + i gamma_ri)(nr + i ni) ... using a 2x2
        # gamma (rr, ri; ri, ii) for a full learnable complex scale, + beta.
        gr = self.gamma_rr.float() * nr + self.gamma_ri.float() * ni + self.beta_r.float()
        gi = self.gamma_ri.float() * nr + self.gamma_ii.float() * ni + self.beta_i.float()
        return gr.to(in_dtype), gi.to(in_dtype)


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


# =============================================================================
# Invertible complex lifting: tied complex decompose/reconstruct. Phase is
# mixed INSIDE the symmetry (by the spectral mixer, wired in model.py), not
# collapsed away — Reconstruct∘Decompose = I is preserved. Decompose returns
# FULL complex coefficients; the block carries (re, im) through the mixer and
# calls the tied reconstruct, taking the real part only at the block output.
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


def modulus_phase_gate(z_r, z_i, gate_mag, eps: float = 1e-3):
    """'modulus_phase' mixer-activation helper: scale the complex value's
    MAGNITUDE by a gate while preserving its phase angle.
        z' = softplus(gate_mag) * (z / (|z| + eps))   with |z| floored at eps
    The gate is forced NON-NEGATIVE via softplus before scaling — without it a
    negative scale flips both components' sign (an unintended π phase rotation).

    fp16 NaN guard: the unit phasor 1/|z| explodes for small |z|, and at fp16
    a small coefficient underflows toward 0. The magnitude is therefore computed
    in fp32 with eps² INSIDE the sqrt, which floors |z| at `eps` itself (since
    sqrt(0 + eps²) = eps) — note the magnitude squares the guard, so flooring
    |z| at 1e-3 needs eps²=1e-6 inside, exactly what eps=1e-3 gives. The divisor
    is then `mag` (already floored); no second `+eps` (that would double-count).
    Division done in fp32, cast back to the input dtype.
    """
    gate_pos = F.softplus(gate_mag)
    mag = torch.sqrt(z_r.float() ** 2 + z_i.float() ** 2 + eps * eps)
    inv = (1.0 / mag).to(z_r.dtype)
    return gate_pos * z_r * inv, gate_pos * z_i * inv


def complex_magnitude(z_r, z_i, eps: float = 1e-3):
    """|z| in fp32 with eps² inside the sqrt → floors |z| at `eps` (fp16-safe;
    matches modulus_phase_gate's guard). Feeds the modulus_phase mixer input."""
    return torch.sqrt(z_r.float() ** 2 + z_i.float() ** 2 + eps * eps).to(z_r.dtype)


def build_complex_wavelet(config: dict, levels: int, C: int,
                          device=None, dtype=None):
    """Factory mirroring tools/two_d_wavelets.build_lifting_wavelet_2d.

    Returns (decompose, reconstruct) for the invertible complex wavelet (tied,
    full-complex coefficients). `complex_mixer_activation` ('split' |
    'modulus_phase') is consumed in model.py's forward, not here.
    """
    construction = config.get("complex_construction", "invertible")
    if construction != "invertible":
        raise ValueError(
            f"complex_construction must be 'invertible', got {construction!r}.")
    hidden_mult = config.get("lifting_hidden_mult", 1)
    dropout = config.get("lifting_dropout", 0.0)
    imag_std = config.get("complex_imag_std", 0.01)
    dec = InvertibleComplexLiftingDecompose(
        levels=levels, C=C, hidden_mult=hidden_mult, dropout=dropout,
        imag_std=imag_std, device=device, dtype=dtype)
    rec = InvertibleComplexLiftingReconstruct(dec)
    return dec, rec


def param_count(config: dict, levels: int, C: int) -> int:
    """Exact trainable-param count of the complex variant at this config, so the
    matched-param real control can be sized via hidden_mult. Builds the module
    on the default device and counts — measured, never estimated (see plan)."""
    dec, rec = build_complex_wavelet(config, levels=levels, C=C)
    n = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    n += sum(p.numel() for p in rec.parameters() if p.requires_grad)
    return n


if __name__ == "__main__":
    # Smoke tests for the invertible complex wavelet (the only construction).
    torch.manual_seed(0)
    B, T, C, levels = 2, 64, 32, 4
    cfg = {"complex_construction": "invertible", "lifting_hidden_mult": 1,
           "lifting_dropout": 0.0}

    # --- ComplexLayerNorm checks ---
    print("=== ComplexLayerNorm ===")
    cln = ComplexLayerNorm(C); cln.eval()
    # (a) reduces to real LayerNorm when imag == 0
    xr = torch.randn(B, T, C)
    zi = torch.zeros(B, T, C)
    gr, gi = cln(xr, zi)
    ref = nn.functional.layer_norm(xr, (C,))  # standard LN, unit gamma/zero beta
    red_err = (gr - ref).abs().max().item()
    assert red_err < 1e-4, f"ComplexLayerNorm != real LN at imag=0: {red_err}"
    assert gi.abs().max().item() < 1e-4, "imag output nonzero when imag input is zero"
    # (b) finite forward + grad on generic complex input
    ar = torch.randn(B, T, C, requires_grad=True)
    ai = torch.randn(B, T, C, requires_grad=True)
    or_, oi_ = cln(ar, ai)
    (or_.pow(2).sum() + oi_.pow(2).sum()).backward()
    assert torch.isfinite(or_).all() and torch.isfinite(oi_).all(), "CLN non-finite fwd"
    assert torch.isfinite(ar.grad).all() and torch.isfinite(ai.grad).all(), "CLN non-finite grad"
    # (c) degenerate input (zero variance) stays finite (det-floor works)
    zr0 = torch.zeros(B, T, C, requires_grad=True)
    zi0 = torch.zeros(B, T, C, requires_grad=True)
    dr, di = cln(zr0, zi0)
    (dr.sum() + di.sum()).backward()
    assert torch.isfinite(dr).all() and torch.isfinite(di).all(), "CLN non-finite on zero input"
    assert torch.isfinite(zr0.grad).all() and torch.isfinite(zi0.grad).all(), "CLN non-finite grad on zero input"
    print(f"  OK | reduces-to-real-LN err {red_err:.1e} | finite fwd/grad incl. zero-variance input")

    print("=== Invertible complex wavelet smoke tests ===")
    dec, rec = build_complex_wavelet(cfg, levels=levels, C=C)
    dec.eval(); rec.eval()
    x = torch.randn(B, T, C)

    # 1) ROUND-TRIP IDENTITY: Reconstruct(Decompose(x)) == x, coefficients
    #    unchanged. The load-bearing symmetry. Must hold for ALL weights (not
    #    only at init), so perturb weights first.
    with torch.no_grad():
        for p_ in dec.parameters():
            p_.add_(0.05 * torch.randn_like(p_))
    ar, ai, dts = dec(x)
    xr, xi = rec(ar, ai, dts)
    rt_resid = (xr - x).abs().max().item()
    xi_resid = xi.abs().max().item()
    assert rt_resid < 1e-3, f"round-trip FAILED: max resid {rt_resid}"
    assert xi_resid < 1e-3, f"round-trip imag not ~0: {xi_resid}"

    # 2) Decompose returns full complex (re, im) coefficients of real width C.
    ar, ai, dts = dec(x)
    assert ar.shape == (B, T, C) and ai.shape == (B, T, C)
    assert len(dts) == levels
    for dr, di in dts:
        assert dr.shape == (B, T, C) and di.shape == (B, T, C)
        assert not torch.is_complex(dr) and not torch.is_complex(di)

    # 3) CAUSALITY GUARD: no output position may depend on a FUTURE input.
    dC = build_complex_wavelet(cfg, levels=levels, C=C)[0]; dC.eval()
    xc = torch.randn(1, T, C)
    _, _, dd = dC(xc)
    t_probe = T // 2
    base = dd[0][0][0, t_probe].clone()
    worst = 0.0
    for dt in range(1, T - t_probe):
        x2 = xc.clone(); x2[0, t_probe + dt] += 100.0
        _, _, dd2 = dC(x2)
        worst = max(worst, (dd2[0][0][0, t_probe] - base).abs().max().item())
    assert worst < 1e-6, f"FUTURE LEAK {worst:.4f}"

    # 4) modulus_phase gate: (a) finite forward+grad near |z|=0 (eps guard),
    #    (b) gate is NON-NEGATIVE (softplus) so it never sign-flips / phase-rotates.
    zr = torch.zeros(2, 4, C, requires_grad=True)
    zi = torch.zeros(2, 4, C, requires_grad=True)
    gate = torch.randn(2, 4, C)  # includes NEGATIVE values on purpose
    gr, gi = modulus_phase_gate(zr, zi, gate)
    (gr.sum() + gi.sum()).backward()
    assert torch.isfinite(gr).all() and torch.isfinite(gi).all(), "modulus_phase non-finite at |z|=0"
    assert torch.isfinite(zr.grad).all() and torch.isfinite(zi.grad).all(), "modulus_phase grad non-finite at |z|=0"
    # Non-negativity: with a nonzero phasor, sign(z') must match sign(z) (no flip).
    zr2 = torch.randn(2, 4, C); zi2 = torch.randn(2, 4, C)
    neg_gate = -torch.abs(torch.randn(2, 4, C)) - 5.0  # strongly negative
    gr2, gi2 = modulus_phase_gate(zr2, zi2, neg_gate)
    # softplus(neg_gate) > 0, so scaled vector keeps the unit-phasor direction:
    assert (torch.sign(gr2) == torch.sign(zr2)).all(), "modulus_phase sign-flip on negative gate (softplus missing)"

    # 5) Imaginary path receives NONZERO gradient at init (no dead phase path).
    dg, rg = build_complex_wavelet(cfg, levels=levels, C=C)
    xg = torch.randn(B, T, C, requires_grad=True)
    ar, ai, dts = dg(xg)
    (rg(ar, ai, dts)[0].sum()).backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all(), "input grad non-finite"
    imag_grad = 0.0
    for net in list(dg.predict_nets) + list(dg.update_nets):
        for w in (net.fc1.lin_i.weight, net.fc2.lin_i.weight):
            if w.grad is not None:
                imag_grad += w.grad.abs().sum().item()
    assert imag_grad > 0, "imaginary path got ZERO gradient at init — dead phase path"

    n_inv = param_count(cfg, levels=levels, C=C)
    print(f"  OK | round-trip {rt_resid:.2e} (imag {xi_resid:.2e}) | "
          f"causal (leak {worst:.0e}) | modulus_phase non-neg + grad finite | "
          f"imag |grad|={imag_grad:.2e} | params(C={C},L={levels})={n_inv:,}")
    print("All invertible complex-wavelet smoke tests passed.")
