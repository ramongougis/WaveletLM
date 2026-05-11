"""2D wavelet decomposition over (batch, token) axes.

Strict generalization of `LiftingWaveletDecompose` (in `model.py`) to operate
over both the token axis (T) and the batch axis (B). When training proceeds in
document-sequential order with parallel-stream batching, B acquires the same
multi-scale temporal structure that justifies wavelet decomposition along T.

See `plans/two_d_wavelet_sequential_training.md` for the full architectural
rationale, hypothesis, and study design.

==============================================================================
PHASE 1 (this file): Scaffold + pass-through. The class instantiates the
existing 1D T-axis lifting wavelet internally and forwards calls to it. No
B-axis decomposition yet. Initialization preserves T2 baseline behavior
exactly; enabling `wavelet_2d_enabled=true` in config is a no-op while the
B-axis logic remains unimplemented.

PHASE 2 (planned): Separable B-axis lifting after each T-axis level, producing
4 sub-bands per level (LL, LH, HL, HH). Initialized as near-no-op via zero
predict/update networks on B-axis so the architecture recovers the 1D wavelet
exactly at step 0; training learns whether to make B-axis active.

PHASE 3 (planned): Cross-batch state passing. LL approximation at the deepest
B-axis scale gets saved at the end of each forward pass and used as initial
state for the next training step (detached gradients — no BPTT across
batches). Document-boundary handling resets the state.

The integration surface in `model.py` is intentionally minimal:
    if config.get('wavelet_2d_enabled', False):
        from tools.two_d_wavelets import LiftingWavelet2D
        shared_lifting = LiftingWavelet2D(t_wavelet=inner_t_wavelet, ...)
    else:
        shared_lifting = LiftingWaveletDecompose(...)  # current 1D path

The 2D wavelet is a `nn.Module` so its parameters are tracked by autograd,
included in `model.parameters()`, and checkpointed normally. `predict_nets`
and `update_nets` are exposed as forwarded attributes so the existing
`LiftingWaveletReconstruct` works against the 2D wrapper transparently.
==============================================================================
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class LiftingWavelet2D(nn.Module):
    """2D lifting wavelet over (batch, token) — Phase 1 scaffold.

    Wraps an existing 1D `LiftingWaveletDecompose` instance and delegates the
    forward pass to it. The B-axis decomposition is **not yet implemented**
    in this scaffold; the class exists to validate the integration surface in
    `model.py` and `train.py` without changing T2's behavior.

    Args:
        t_wavelet: An instance of `LiftingWaveletDecompose` from `model.py`,
            constructed by the caller with the standard 1D lifting config.
            This is passed in (rather than constructed here) to avoid a
            circular import between `model.py` and `tools/two_d_wavelets.py`.
        b_levels: Maximum number of B-axis decomposition levels. -1 (default)
            auto-derives from `log2(batch_size)` at first forward call.
            Capped at `min(b_levels, t_levels)`; deeper T-only levels continue
            on the LL band beyond the B-axis depth limit.
        init_zero: If True (default), B-axis predict/update networks are
            initialized to zero so the architecture exactly recovers the 1D
            wavelet behavior at step 0. Training learns whether to activate
            the B-axis path. If False, B-axis networks use standard init.
        state_passing: If True, the deepest-scale LL approximation is saved
            across forward calls as cross-batch state (Phase 3 feature; not
            functional in Phase 1).
        cross_batch_state_dim: Channel dim for cross-batch state buffer. Must
            match `C` of the surrounding model when state_passing is on.

    Forward signature matches `LiftingWaveletDecompose`:
        x: Tensor of shape (B, T, C)
        Returns: (approx, details) where approx is (B, T_coarsest, C) and
            details is a list of length `levels`, each (B, T_at_level, C).
    """

    def __init__(
        self,
        t_wavelet: nn.Module,
        b_levels: int = -1,
        init_zero: bool = True,
        state_passing: bool = False,
        cross_batch_state_dim: int = None,
    ):
        super().__init__()
        self._t_wavelet = t_wavelet
        self.b_levels = b_levels
        self.init_zero = init_zero
        self.state_passing = state_passing
        self.cross_batch_state_dim = cross_batch_state_dim

        # Phase 1: B-axis lifting not yet built. Phase 2 will populate:
        #   self.b_predict_nets = nn.ModuleList(...)
        #   self.b_update_nets = nn.ModuleList(...)
        # initialized as zero networks when self.init_zero, standard init otherwise.

        # Phase 3: cross-batch state buffer (LL approximation from previous step,
        # detached, with document-boundary reset). Registered as a non-persistent
        # buffer so it's not checkpointed:
        #   if state_passing:
        #       self.register_buffer('cross_batch_state', None, persistent=False)

    # ----- Forwarded attributes for LiftingWaveletReconstruct compatibility ---
    # LiftingWaveletReconstruct accesses `update_nets[level]` directly off its
    # decompose_module argument. Forward those attributes from the inner 1D
    # wavelet so the existing reconstruct path works against a 2D wrapper.

    @property
    def predict_nets(self) -> nn.ModuleList:
        return self._t_wavelet.predict_nets

    @property
    def update_nets(self) -> nn.ModuleList:
        return self._t_wavelet.update_nets

    @property
    def levels(self) -> int:
        return self._t_wavelet.levels

    @property
    def inv_sqrt2(self) -> float:
        return self._t_wavelet.inv_sqrt2

    @property
    def sqrt2(self) -> float:
        return self._t_wavelet.sqrt2

    def reset_cross_batch_state(self) -> None:
        """Document-boundary signal. Phase 3 will use this to clear the
        cross-batch state buffer when a new document begins."""
        # Phase 3 placeholder — no-op in Phase 1.
        pass

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # ---- Phase 1: pure pass-through to 1D T-axis wavelet ----------------
        # This branch exists to validate the integration surface without
        # changing T2 behavior. Enabling `wavelet_2d_enabled` in config is a
        # no-op until Phase 2 lands.
        approx, details = self._t_wavelet(x)
        return approx, details

        # ---- Phase 2 sketch (NOT YET IMPLEMENTED) ---------------------------
        # B, T, C = x.shape
        # if self.b_levels == -1:
        #     b_levels = int(math.log2(B)) if B > 1 else 0
        # else:
        #     b_levels = min(self.b_levels, int(math.log2(B)))
        # joint_levels = min(b_levels, self._t_wavelet.levels)
        #
        # cur = x
        # details: List[torch.Tensor] = []
        #
        # # Joint (B, T) decomposition for first joint_levels levels.
        # for level in range(joint_levels):
        #     # Step 1: T-axis lifting (existing 1D logic, factored).
        #     t_approx, t_detail = self._t_axis_lift_step(cur, level)
        #     # Step 2: B-axis lifting on both T-approx and T-detail.
        #     # → produces 4 sub-bands: (LL, LH) from T-approx; (HL, HH) from T-detail.
        #     ll, lh = self._b_axis_lift_step(t_approx, level)
        #     hl, hh = self._b_axis_lift_step(t_detail, level)
        #     details.append({
        #         'LH': lh, 'HL': hl, 'HH': hh
        #     })  # interface change — also needs mixer-side adaptation
        #     cur = ll  # recurse on the LL band only
        #
        # # T-only continuation for remaining levels (depth asymmetry).
        # for level in range(joint_levels, self._t_wavelet.levels):
        #     cur, detail = self._t_axis_lift_step(cur, level)
        #     details.append(detail)  # back to 1D detail (no B sub-bands)
        #
        # # Phase 3: stash deepest-scale LL approx as cross-batch state.
        # # if self.state_passing:
        # #     self.cross_batch_state = cur.detach().clone()
        #
        # return cur, details

    def extra_repr(self) -> str:
        return (
            f"phase=1 (pass-through), "
            f"b_levels={self.b_levels}, init_zero={self.init_zero}, "
            f"state_passing={self.state_passing}"
        )


# =============================================================================
# Construction helpers — exposed so model.py can build the 2D variant without
# replicating its constructor parameter list.
# =============================================================================


def build_lifting_wavelet_2d(t_wavelet: nn.Module, config: dict) -> LiftingWavelet2D:
    """Construct a `LiftingWavelet2D` instance from a config dict.

    Args:
        t_wavelet: pre-built `LiftingWaveletDecompose` instance (the inner
            1D T-axis lifting wavelet).
        config: the model config dict; reads the `wavelet_2d_*` knobs.

    Returns:
        A `LiftingWavelet2D` ready to be assigned to `self.shared_lifting`.
    """
    return LiftingWavelet2D(
        t_wavelet=t_wavelet,
        b_levels=config.get("wavelet_2d_b_levels", -1),
        init_zero=config.get("wavelet_2d_init_zero", True),
        state_passing=config.get("wavelet_2d_state_passing", False),
        cross_batch_state_dim=config.get("C", None),
    )
