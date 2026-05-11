"""2D wavelet decomposition over (batch, token) axes.

Strict generalization of `LiftingWaveletDecompose` (in `model.py`) to operate
over both the token axis (T) and the batch axis (B). When training proceeds in
document-sequential order with parallel-stream batching, B acquires the same
multi-scale temporal structure that justifies wavelet decomposition along T.

See `plans/two_d_wavelet_sequential_training.md` for the full architectural
rationale, hypothesis, and study design.

==============================================================================
MODES (selectable via `wavelet_2d_mode` config flag)

"off":         No 2D code path. This file isn't even imported. Identical to
               the 1D T-axis wavelet (current production T2 behavior).

"passthrough": Phase 1 scaffold. The class instantiates the existing 1D
               T-axis lifting wavelet internally and forwards calls to it.
               No B-axis decomposition. Used as a smoke test to validate the
               integration surface (config flag, lazy import, forwarded
               properties, autograd, param counting). Validated on
               2026-05-11 (logs/wikitext-103_2026-05-11_05-26-15 et al.):
               trajectory matches Rainman baseline within noise (~0.001 nats).

"internal":    Phase 2.A. At each lifting level, do separable T-axis then
               B-axis lifting → 4 sub-bands (LL, LH, HL, HH). Apply per-
               sub-band processing (a per-channel scalar gain, init to
               identity), then B-axis inverse lifting → reassembled into the
               standard (approx_T, detail_T) pair, same shape as 1D. Returns
               the existing (approx, details) interface. Cross-batch
               information enters via the per-sub-band processing; nothing
               downstream of the wavelet needs to change.

"subband":     Phase 2.B (NOT YET IMPLEMENTED). Same separable T-then-B
               lifting at each level, but the 4 sub-bands are kept separate
               and returned to the caller. Per-scale mixer is generalized to
               a 2D table indexed by (B-scale, T-scale), with each sub-band
               getting its own mixer. Reconstruction mirrored. This is the
               architecturally pure version from the plan but touches
               model.py and the mixer code. Stubbed with NotImplementedError;
               will be implemented after "internal" is tested.

The "internal" and "subband" modes are deliberately offered as alternatives
so that we get two architectural shots at finding cross-batch lift benefit.
"internal" might work because B-axis mixing routes useful temporal info into
the wavelet cascade without disturbing downstream architecture. "subband"
might work because per-sub-band mixer specialization captures band-specific
structure that "internal"'s reassembly throws away. They could compose in a
future "both" mode if both independently show signal.

PHASE 3 (planned, orthogonal to mode): Cross-batch state passing. LL
approximation at the deepest B-axis scale gets saved at the end of each
forward pass and used as initial state for the next training step
(detached gradients — no BPTT across batches). Document-boundary handling
resets the state.

==============================================================================

Integration surface in `model.py` is intentionally minimal — one conditional
that swaps in the 2D wrapper when `wavelet_2d_mode != "off"`:

    if config.get('wavelet_2d_mode', 'off') != 'off':
        from tools.two_d_wavelets import build_lifting_wavelet_2d
        shared_lifting = build_lifting_wavelet_2d(t_wavelet=shared_lifting, config=config)

The 2D wavelet is a `nn.Module` so its parameters are tracked by autograd,
included in `model.parameters()`, and checkpointed normally. `predict_nets`
and `update_nets` are exposed as forwarded properties so the existing
`LiftingWaveletReconstruct` works against the 2D wrapper transparently in
the "passthrough" and "internal" modes (which return the same (approx,
details) output shape as the 1D wavelet).
==============================================================================
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Sub-band indexing constants for sb_scales tensor.
SB_LL = 0  # approx-T, approx-B (smoothest)
SB_LH = 1  # approx-T, detail-B (smooth in T, rapid in B)
SB_HL = 2  # detail-T, approx-B (rapid in T, smooth in B)
SB_HH = 3  # detail-T, detail-B (rapid in both — fastest)


class LiftingWavelet2D(nn.Module):
    """2D lifting wavelet over (batch, token).

    Wraps an existing 1D `LiftingWaveletDecompose` instance and adds B-axis
    lifting machinery on top. The mode parameter selects how the B-axis
    decomposition is exposed (or not) to downstream model components.

    Args:
        t_wavelet: An instance of `LiftingWaveletDecompose` from `model.py`,
            constructed by the caller with the standard 1D lifting config.
            Passed in (rather than constructed here) to avoid a circular
            import between `model.py` and `tools/two_d_wavelets.py`.
        C: Channel dimension. Needed to size B-axis predict/update nets and
            per-sub-band scale parameters.
        mode: One of "passthrough" | "internal" | "subband". See module
            docstring for semantics.
        batch_size: B (= MBS). Used to auto-cap b_levels at log2(B) when
            b_levels=-1.
        b_levels: Number of levels at which to apply B-axis lifting. -1 (the
            default) auto-derives from min(log2(B), t_wavelet.levels).
            Levels beyond b_levels run T-only (just like the 1D wavelet).
        init_zero: If True (default), B-axis predict/update networks are
            zero-initialized AND per-sub-band scale parameters initialize to
            1.0 (identity). With these settings, the architecture exactly
            recovers the 1D wavelet at step 0. Training learns whether to
            activate the B-axis path.
        state_passing: Phase 3 placeholder. Not functional yet.

    Forward signature (for "passthrough" and "internal" modes):
        x: Tensor of shape (B, T, C)
        Returns: (approx, details) — same interface as LiftingWaveletDecompose.

    For "subband" mode (when implemented), the return type will be different
    (4 sub-bands per level instead of 2). Phase 2.B will define that interface.
    """

    def __init__(
        self,
        t_wavelet: nn.Module,
        C: int,
        mode: str = "passthrough",
        batch_size: int = 8,
        b_levels: int = -1,
        init_zero: bool = True,
        state_passing: bool = False,
    ):
        super().__init__()
        self._t_wavelet = t_wavelet
        self.C = C
        self.mode = mode
        self.batch_size = batch_size
        self.init_zero = init_zero
        self.state_passing = state_passing

        # Derive b_levels: capped at min(log2(B), t_wavelet.levels).
        max_b_levels_from_batch = int(math.floor(math.log2(batch_size))) if batch_size > 1 else 0
        max_b_levels_from_t = t_wavelet.levels
        if b_levels == -1:
            self.b_levels = min(max_b_levels_from_batch, max_b_levels_from_t)
        else:
            self.b_levels = min(b_levels, max_b_levels_from_batch, max_b_levels_from_t)

        self.inv_sqrt2 = 0.7071067811865476
        self.sqrt2 = 1.4142135623730951

        # B-axis predict/update networks and per-sub-band scales — only
        # constructed for modes that actually use them.
        if mode in ("internal", "subband") and self.b_levels > 0:
            # b_predict_nets[level] and b_update_nets[level]: Linear(C, C).
            # Shared between the (approx_T → LL/LH) and (detail_T → HL/HH)
            # branches at each level. Could be split into separate nets per
            # branch as a future experiment; sharing is the simpler design.
            self.b_predict_nets = nn.ModuleList([
                nn.Linear(C, C) for _ in range(self.b_levels)
            ])
            self.b_update_nets = nn.ModuleList([
                nn.Linear(C, C) for _ in range(self.b_levels)
            ])

            # Per-sub-band scales: shape (b_levels, 4, C). 4 sub-bands per
            # level (LL, LH, HL, HH); each gets a per-channel scalar gain.
            # Cheap (only ~24K params total for b_levels=3, C=2048) and
            # cleanly identity-initializable.
            self.sb_scales = nn.Parameter(torch.ones(self.b_levels, 4, C))

            if init_zero:
                # Zero-init the B-axis predict/update networks so that
                # b_detail = (odd_B - 0) * inv_sqrt2 = odd_B * inv_sqrt2 and
                # b_approx = (even_B + 0) * inv_sqrt2 = even_B * inv_sqrt2.
                # Combined with sb_scales=1.0, the B-axis inverse lift
                # exactly recovers the input (approx_T, detail_T) pair.
                for lin in self.b_predict_nets:
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)
                for lin in self.b_update_nets:
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)
                # sb_scales already initialized to ones above.
        else:
            self.b_predict_nets = None
            self.b_update_nets = None
            self.sb_scales = None

        # Phase 3: cross-batch state buffer (LL approximation from previous
        # step, detached, with document-boundary reset). Not functional yet.

    # ----- Forwarded attributes for LiftingWaveletReconstruct compatibility ---
    # LiftingWaveletReconstruct accesses `update_nets[level]` directly off its
    # decompose_module argument. Forward those attributes from the inner 1D
    # wavelet so the existing reconstruct path works against a 2D wrapper.
    # (Reconstruct only inverts the T-axis lift; the B-axis lift-then-inverse
    # cycle inside the 2D forward leaves no residual to invert externally.)

    @property
    def predict_nets(self) -> nn.ModuleList:
        return self._t_wavelet.predict_nets

    @property
    def update_nets(self) -> nn.ModuleList:
        return self._t_wavelet.update_nets

    @property
    def levels(self) -> int:
        return self._t_wavelet.levels

    def reset_cross_batch_state(self) -> None:
        """Phase 3 document-boundary signal — no-op until Phase 3 lands."""
        pass

    # ----- Forward dispatch ---------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.mode == "passthrough":
            return self._forward_passthrough(x)
        elif self.mode == "internal":
            return self._forward_internal(x)
        elif self.mode == "subband":
            raise NotImplementedError(
                "wavelet_2d_mode='subband' is Phase 2.B and not yet implemented. "
                "Implementing it requires changes to model.py (2D per-scale mixer "
                "table) and the reconstruct path (4 sub-bands per level instead "
                "of 2). Use 'internal' mode for now, which is self-contained in "
                "tools/two_d_wavelets.py."
            )
        else:
            raise ValueError(
                f"Unknown wavelet_2d_mode: {self.mode!r}. "
                f"Expected one of: 'passthrough', 'internal', 'subband'."
            )

    def _forward_passthrough(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Phase 1 mode: pure pass-through to 1D T-axis wavelet."""
        return self._t_wavelet(x)

    def _forward_internal(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Phase 2.A mode: separable T-then-B lifting at each level with
        per-sub-band scaling, then B-axis inverse lift, reassembled into the
        standard (approx_T, detail_T) pair. Same output shape as 1D wavelet.

        Per level k (for k < b_levels):
            1. T-axis lift: (B, T, C) cur → approx_T, detail_T (each (B, T, C)).
            2. B-axis lift on approx_T → (LL, LH); on detail_T → (HL, HH).
               (All four sub-bands are (B, T, C) — undecimated transform.)
            3. Per-sub-band scaling: multiply each sub-band element-wise by
               its learnable per-channel scale (init = 1.0 → identity).
            4. B-axis inverse lift: combine (LL, LH) → approx_T'; combine
               (HL, HH) → detail_T'. Same (B, T, C) shape as before.
            5. Append detail_T' to details; recurse on approx_T'.

        For levels k >= b_levels, fall back to pure 1D T-axis lifting.
        """
        t_wavelet = self._t_wavelet
        details: List[torch.Tensor] = []
        cur = x
        B = cur.shape[0]

        for level in range(t_wavelet.levels):
            # ---- Step 1: T-axis lift (mirrors LiftingWaveletDecompose.forward).
            base_dilation_T = 1 << level
            T = cur.shape[1]

            if t_wavelet.wavelet_crawl:
                K = t_wavelet.wavelet_crawl_k
                offsets = t_wavelet._crawl_offsets[level]
                weights = F.softmax(t_wavelet.dilation_logits[level], dim=0)
                max_d = offsets[-1]
                padded = F.pad(cur, (0, 0, max_d, 0))
                odd_T = sum(
                    weights[k] * padded[:, max_d - offsets[k]:max_d - offsets[k] + T, :]
                    for k in range(K)
                )
            else:
                padded = F.pad(cur, (0, 0, base_dilation_T, 0))
                odd_T = padded[:, :-base_dilation_T, :]

            even_T = cur
            predicted_T = t_wavelet.predict_nets[level](even_T)
            detail_T = (odd_T - predicted_T) * self.inv_sqrt2
            update_T = t_wavelet.update_nets[level](detail_T)
            approx_T = (even_T + update_T) * self.inv_sqrt2

            # ---- Step 2-5: B-axis lift + sub-band scale + B-axis inverse,
            # applied at levels with B-axis capacity (k < b_levels).
            if level < self.b_levels:
                base_dilation_B = 1 << level

                # B-axis lift on approx_T → (LL, LH).
                # Pad B axis at start so odd_B reaches back into history.
                # F.pad signature for (B, T, C): pad given last-dim-first;
                # (0, 0, 0, 0, base_dilation_B, 0) pads dim 0 (B) at start.
                padded_B = F.pad(approx_T, (0, 0, 0, 0, base_dilation_B, 0))
                odd_B_a = padded_B[:-base_dilation_B, :, :]
                even_B_a = approx_T
                pred_B_a = self.b_predict_nets[level](even_B_a)
                LH = (odd_B_a - pred_B_a) * self.inv_sqrt2
                upd_B_a = self.b_update_nets[level](LH)
                LL = (even_B_a + upd_B_a) * self.inv_sqrt2

                # B-axis lift on detail_T → (HL, HH).
                padded_B = F.pad(detail_T, (0, 0, 0, 0, base_dilation_B, 0))
                odd_B_d = padded_B[:-base_dilation_B, :, :]
                even_B_d = detail_T
                pred_B_d = self.b_predict_nets[level](even_B_d)
                HH = (odd_B_d - pred_B_d) * self.inv_sqrt2
                upd_B_d = self.b_update_nets[level](HH)
                HL = (even_B_d + upd_B_d) * self.inv_sqrt2

                # Per-sub-band scaling (init = 1.0 → identity).
                # sb_scales shape: (b_levels, 4, C). Index by [level, sb_idx]
                # to get a (C,) vector that broadcasts across (B, T).
                LL = LL * self.sb_scales[level, SB_LL]
                LH = LH * self.sb_scales[level, SB_LH]
                HL = HL * self.sb_scales[level, SB_HL]
                HH = HH * self.sb_scales[level, SB_HH]

                # B-axis inverse lift: reconstruct approx_T and detail_T from
                # their respective sub-band pairs. The inverse formula
                # mirrors LiftingWaveletReconstruct: given (approx, detail),
                # recover the original via `cur = approx * sqrt2 - update_net(detail)`.
                upd_B_a_inv = self.b_update_nets[level](LH)
                approx_T = LL * self.sqrt2 - upd_B_a_inv

                upd_B_d_inv = self.b_update_nets[level](HH)
                detail_T = HL * self.sqrt2 - upd_B_d_inv

            details.append(detail_T)
            cur = approx_T

        return cur, details

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, b_levels={self.b_levels}, "
            f"init_zero={self.init_zero}, state_passing={self.state_passing}"
        )


# =============================================================================
# Construction helper — exposed so model.py can build the 2D variant without
# replicating its constructor parameter list.
# =============================================================================


def build_lifting_wavelet_2d(t_wavelet: nn.Module, config: dict) -> LiftingWavelet2D:
    """Construct a `LiftingWavelet2D` instance from a config dict.

    Args:
        t_wavelet: pre-built `LiftingWaveletDecompose` instance (the inner
            1D T-axis lifting wavelet).
        config: the model config dict; reads the `wavelet_2d_*` knobs and `C`.

    Returns:
        A `LiftingWavelet2D` ready to be assigned to `self.shared_lifting`.
    """
    mode = config.get("wavelet_2d_mode", "passthrough")
    return LiftingWavelet2D(
        t_wavelet=t_wavelet,
        C=int(config["C"]),
        mode=mode,
        batch_size=int(config["micro_batch_size"]),
        b_levels=config.get("wavelet_2d_b_levels", -1),
        init_zero=config.get("wavelet_2d_init_zero", True),
        state_passing=config.get("wavelet_2d_state_passing", False),
    )
