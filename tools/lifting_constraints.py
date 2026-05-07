"""Structural constraints for the lifting predict/update Linear matrices.

Tests structured-sparsity alternatives to the magnitude-pruned masking from
the M-series sweep (M1-M4 in runs.md). Each constraint replaces the
unconstrained Linear(C, C) in the lifting predict/update Sequentials with a
parameterization that limits which (i, j) positions can be nonzero, forcing
gradient descent to find a constrained solution rather than selecting from
an unconstrained reference.

Three mask-based variants share a single StructuredLinear class (which is
nn.Linear with a fixed binary mask applied to the weight at every forward
pass via autograd-friendly multiplication). Monarch is a separate
re-parameterization (product of block-diagonal factors with a perfect-shuffle
permutation between them) and is implemented as MonarchLinear.

See README -> Wavelet Off-Diagonal Masking with Structured Variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_structural_mask(
    kind: str,
    C: int,
    *,
    block_size: int = 64,
    band_width: int = 64,
    reference_weight: torch.Tensor = None,
    density: float = 0.0,
    seed: int = 1337,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Build a fixed (C, C) binary mask for a structural or top-k constraint.

    Returns a (C, C) tensor with values in {0, 1}. Caller registers it as a
    buffer and applies it to the Linear weight before forward.

    Structural kinds (no reference_weight needed):
        upper_triangular -- W[i, j] kept iff j >= i (50% incl. diagonal)
        lower_triangular -- W[i, j] kept iff j <= i (50% incl. diagonal)
        block_diagonal   -- W[i, j] kept iff i // b == j // b
        banded           -- W[i, j] kept iff |i - j| <= band_width

    Top-k kinds (require `density`; magnitude_topk also requires `reference_weight`):
        magnitude_topk -- diagonal always kept, plus top `density` fraction of
                          off-diagonal positions ranked by |reference_weight|.
        random_topk    -- diagonal always kept, plus a random `density` fraction
                          of off-diagonal positions, seeded by `seed`.
    """
    dtype = dtype or torch.float32
    if kind == "upper_triangular":
        return torch.triu(torch.ones(C, C, device=device, dtype=dtype))
    elif kind == "lower_triangular":
        return torch.tril(torch.ones(C, C, device=device, dtype=dtype))
    elif kind == "block_diagonal":
        if C % block_size != 0:
            raise ValueError(
                f"C={C} must be divisible by block_size={block_size} "
                f"for block_diagonal structure"
            )
        nblocks = C // block_size
        return torch.block_diag(
            *[torch.ones(block_size, block_size, device=device, dtype=dtype)
              for _ in range(nblocks)]
        )
    elif kind == "banded":
        idx = torch.arange(C, device=device)
        return (torch.abs(idx[:, None] - idx[None, :]) <= band_width).to(dtype)
    elif kind in ("magnitude_topk", "random_topk"):
        if not (0.0 <= density <= 1.0):
            raise ValueError(f"density must be in [0, 1], got {density}")
        # Diagonal always kept (structural backbone)
        mask = torch.eye(C, device=device, dtype=dtype)
        n_offdiag = C * C - C
        n_keep = int(round(density * n_offdiag))
        if n_keep == 0:
            return mask
        if kind == "magnitude_topk":
            if reference_weight is None:
                raise ValueError(
                    "magnitude_topk requires reference_weight (the trained "
                    "weight matrix to rank off-diagonal positions by magnitude)"
                )
            if reference_weight.shape != (C, C):
                raise ValueError(
                    f"reference_weight shape {tuple(reference_weight.shape)} "
                    f"must equal (C, C) = ({C}, {C})"
                )
            scores = reference_weight.detach().abs().to(
                device=device, dtype=torch.float32
            )
        else:  # random_topk
            gen_device = device if device is not None else "cpu"
            g = torch.Generator(device=gen_device)
            g.manual_seed(int(seed))
            scores = torch.rand(C, C, device=gen_device, generator=g)
        # Zero diagonal in scores so it doesn't compete in the top-k
        scores = scores - torch.diag(torch.diag(scores))
        flat_scores = scores.flatten()
        threshold = torch.topk(flat_scores, n_keep, largest=True).values[-1]
        offdiag_mask = (scores >= threshold).to(dtype)
        mask = mask + offdiag_mask
        # Clamp to {0, 1} in case ties or numerical edge cases produced 2s
        return (mask > 0).to(dtype)
    else:
        raise ValueError(f"Unknown structural mask kind: {kind!r}")


class StructuredLinear(nn.Linear):
    """nn.Linear with a fixed (out_features, in_features) binary mask enforced
    via init-time multiplication + a gradient hook.

    Forward computes plain y = W @ x + bias. Sparsity is preserved because:
      1. The mask is multiplied into the weight at __init__ -> non-mask
         positions start at zero.
      2. A gradient hook on `self.weight` zeroes the gradient at non-mask
         positions before each optimizer step -> the optimizer never updates
         those positions -> they stay at zero throughout training.
      3. The mask is also re-applied via init_structured_lifting_linear after
         the structural Haar / random / zero re-init, so any caller that
         re-initializes the weight ends up masked.

    The mask is stored as a `bool` buffer (4x smaller than fp32). Multiplying
    a float gradient by a bool tensor uses PyTorch's implicit dtype
    promotion in the multiply kernel, so no separate fp32 copy of the mask
    is materialized at backward time.

    Effective parameter count is reported by `effective_numel()`, which counts
    nonzero positions in the mask plus bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        if mask.shape != (out_features, in_features):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != "
                f"(out_features, in_features) = ({out_features}, {in_features})"
            )
        # Persistent so loaded checkpoints retain the mask exactly. Stored as
        # bool to keep the buffer 4x smaller than the parameter; PyTorch's
        # element-wise multiply promotes bool to grad's dtype implicitly.
        self.register_buffer(
            "_struct_mask",
            mask.to(device=self.weight.device, dtype=torch.bool),
            persistent=True,
        )
        # Apply mask to init weights so masked positions start at zero.
        with torch.no_grad():
            self.weight.mul_(self._struct_mask)

        # Gradient hook: zero gradient at non-mask positions before optimizer
        # step. Captured via closure so the hook resolves self._struct_mask at
        # call time (after any device move).
        def _grad_hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * self._struct_mask
        self.weight.register_hook(_grad_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def effective_numel(self) -> int:
        n = int(self._struct_mask.sum().item())
        if self.bias is not None:
            n += self.bias.numel()
        return n


class MonarchLinear(nn.Module):
    """2-factor Monarch matrix: M = R . P . L applied to a square N x N transform.

    For nblocks = n, block_size = b with n * b = N:
      L: (n, b, b) -- n parallel b x b blocks applied to (..., n, b) view
      P: perfect-shuffle permutation, reshape (n, b) -> (b, n)
      R: (b, n, n) -- b parallel n x n blocks applied to (..., b, n) view

    Total params: n * b^2 + b * n^2 = n * b * (n + b).

    For C = 2048:
      nblocks=32, block_size=64: 32 * 64 * 96 = 196,608 (~197K)
      nblocks=64, block_size=32: 64 * 32 * 96 = 196,608 (~197K)
    Same param count by symmetry; the variants differ in inductive bias (which
    grouping the larger blocks come first), not in capacity.

    Init: both factors set to block-diagonal identity, so M ~= I at start
    (matches the Haar lifting prior).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nblocks: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if in_features != out_features:
            raise ValueError(
                f"MonarchLinear requires square shape, got "
                f"in={in_features}, out={out_features}"
            )
        N = in_features
        if N % nblocks != 0:
            raise ValueError(
                f"in_features={N} must be divisible by nblocks={nblocks}"
            )
        block_size = N // nblocks
        self.in_features = N
        self.out_features = N
        self.nblocks = nblocks
        self.block_size = block_size
        kw = dict(device=device, dtype=dtype)
        # L: n blocks of size b x b, applied to the (..., n, b) view
        self.L = nn.Parameter(torch.empty(nblocks, block_size, block_size, **kw))
        # R: b blocks of size n x n, applied to the (..., b, n) post-shuffle view
        self.R = nn.Parameter(torch.empty(block_size, nblocks, nblocks, **kw))
        if bias:
            self.bias = nn.Parameter(torch.zeros(N, **kw))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Init L, R to block-diagonal identity so M ~= I at start.
        Matches the Haar lifting init prior."""
        with torch.no_grad():
            n, b = self.nblocks, self.block_size
            eye_b = torch.eye(b, device=self.L.device, dtype=self.L.dtype)
            eye_n = torch.eye(n, device=self.R.device, dtype=self.R.dtype)
            for i in range(n):
                self.L[i].copy_(eye_b)
            for i in range(b):
                self.R[i].copy_(eye_n)
            if self.bias is not None:
                self.bias.zero_()

    def init_zero(self):
        with torch.no_grad():
            self.L.zero_()
            self.R.zero_()
            if self.bias is not None:
                self.bias.zero_()

    def init_random(self, std: float = 0.01):
        with torch.no_grad():
            nn.init.normal_(self.L, std=std)
            nn.init.normal_(self.R, std=std)
            if self.bias is not None:
                self.bias.zero_()

    def scale_by(self, factor: float):
        """Scale the effective matrix by `factor` by scaling one of the factors."""
        with torch.no_grad():
            self.L.mul_(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Canonical Monarch: M = R . P . L . P^T (two perfect shuffles).
        # The leading P^T (input view as (b, n) then transpose to (n, b))
        # cancels the inner P when L = I, allowing M = I at Haar init when
        # R = I too -- a property required by the lifting cascade's
        # near-identity prior. A single-shuffle construction (M = R . P . L)
        # cannot represent the identity with block-diagonal factors.
        N, n, b = self.in_features, self.nblocks, self.block_size
        leading = x.shape[:-1]
        if x.shape[-1] != N:
            raise ValueError(f"Expected last dim {N}, got {x.shape[-1]}")
        x = x.reshape(*leading, b, n)                          # (..., b, n)
        x = x.transpose(-1, -2).contiguous()                   # (..., n, b)  -- P^T applied
        x = torch.einsum("nij,...nj->...ni", self.L, x)        # (..., n, b)  -- L applied
        x = x.transpose(-1, -2).contiguous()                   # (..., b, n)  -- P applied
        x = torch.einsum("bij,...bj->...bi", self.R, x)        # (..., b, n)  -- R applied
        x = x.reshape(*leading, N)                             # (..., N)
        if self.bias is not None:
            x = x + self.bias
        return x

    def effective_numel(self) -> int:
        n = self.L.numel() + self.R.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n


def build_structured_lifting_linear(
    in_features: int,
    out_features: int,
    structure: str,
    *,
    block_size: int = 64,
    band_width: int = 64,
    monarch_blocks: int = 32,
    reference_weight: torch.Tensor = None,
    density: float = 0.0,
    seed: int = 1337,
    bias: bool = True,
    device=None,
    dtype=None,
) -> nn.Module:
    """Construct the appropriate structured Linear-replacement for the lifting
    predict/update path. Returns either a StructuredLinear (mask-based) or a
    MonarchLinear (re-parameterization).

    Supported `structure` values:
        "upper_triangular" / "lower_triangular" / "block_diagonal" / "banded"
            -- structural masks; no reference_weight needed
        "magnitude_topk" / "random_topk"
            -- top-k masks; magnitude_topk requires reference_weight
        "monarch"
            -- 2-factor block-diagonal product (no mask)

    Asserts square shape (in == out); the lifting analyzed config has
    hidden_mult=1 and structural priors are defined for square matrices.
    """
    if in_features != out_features:
        raise ValueError(
            f"Structured lifting Linears require square shape "
            f"(in == out), got in={in_features}, out={out_features}. "
            f"Structural priors are defined for the hidden_mult=1 lifting "
            f"configuration only."
        )
    C = in_features
    if structure == "monarch":
        return MonarchLinear(
            C, C, nblocks=monarch_blocks, bias=bias, device=device, dtype=dtype
        )
    elif structure in (
        "upper_triangular", "lower_triangular", "block_diagonal", "banded",
        "magnitude_topk", "random_topk",
    ):
        mask = make_structural_mask(
            structure, C,
            block_size=block_size, band_width=band_width,
            reference_weight=reference_weight, density=density, seed=seed,
            device=device, dtype=dtype,
        )
        return StructuredLinear(
            C, C, mask, bias=bias, device=device, dtype=dtype
        )
    else:
        raise ValueError(
            f"Unknown lifting_offdiag_structure: {structure!r}. "
            f"Expected one of: upper_triangular, lower_triangular, "
            f"block_diagonal, banded, magnitude_topk, random_topk, monarch."
        )


def init_structured_lifting_linear(
    linear: nn.Module,
    init_mode: str,
    scale: float = 1.0,
):
    """Apply lifting_init mode (haar / zero / random) to a structured Linear.

    For Haar (the default), the matrix is set to `scale * I` (identity scaled
    by `scale`). All four mask-based structures include the diagonal, so this
    init survives the mask. For Monarch, both factors are set so M ~= scale * I.
    """
    if isinstance(linear, MonarchLinear):
        if init_mode == "haar":
            linear.reset_parameters()  # already block-diag identity
            if scale != 1.0:
                linear.scale_by(scale)
        elif init_mode == "zero":
            linear.init_zero()
        elif init_mode == "random":
            linear.init_random(std=0.01)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode!r}")
    elif isinstance(linear, StructuredLinear):
        with torch.no_grad():
            if init_mode == "haar":
                nn.init.eye_(linear.weight)
                if scale != 1.0:
                    linear.weight.mul_(scale)
                # Re-apply mask (no-op for diagonal-preserving masks, but defensive)
                linear.weight.mul_(linear._struct_mask)
                if linear.bias is not None:
                    linear.bias.zero_()
            elif init_mode == "zero":
                linear.weight.zero_()
                if linear.bias is not None:
                    linear.bias.zero_()
            elif init_mode == "random":
                nn.init.normal_(linear.weight, std=0.01)
                linear.weight.mul_(linear._struct_mask)
                if linear.bias is not None:
                    linear.bias.zero_()
            else:
                raise ValueError(f"Unknown init_mode: {init_mode!r}")
    else:
        raise TypeError(
            f"Expected StructuredLinear or MonarchLinear, got {type(linear)}"
        )


def make_mlp_mask(
    in_features: int,
    out_features: int,
    structure: str,
    *,
    block_size: int = 64,
    band_width: int = 64,
    pq_density: float = 0.1,
    pq_mode: str = "structural",
    tile_C: int | None = None,
    outer_dim: int | None = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Build a (out_features, in_features) binary mask for an MLP weight matrix.

    Supports three matrix shapes that arise in a standard MLP block at C / E·C:
      - W1 (out=E·C, in=C):  the expansion matrix
      - W2 (out=C, in=E·C):  the contraction matrix
      - middle (E·C, E·C):   appears when hidden_layers > 2

    Tiling is parametrized by `tile_C`, the per-block tile size (architectural
    C of the model). Both `in_features` and `out_features` must be multiples of
    `tile_C`. Defaults to `min(in_features, out_features)` for backward compat
    when only outer linears are involved (tile_C = C in that case).

    `outer_dim` (the architectural E·C) is used by `pq_strided` to find a single
    (p, q) tuple valid across all three matrix shapes. Defaults to
    `max(in_features, out_features)`, which is correct for any individual MLP
    matrix; pass explicitly if you want to share the same (p, q) across W1 / W2 /
    middle linears (see `FeedForward._make_linear`).

    - 'banded' / 'block_diagonal': matrix is tiled into a (E_out × E_in) grid
      of (tile_C, tile_C) blocks, each with the standard square-matrix
      structural pattern from `make_structural_mask`. Per-block density matches
      the lifting BAND/BD curves at the same (block_size / band_width).
    - 'pq_strided': 1D walk (alternating p, q steps) directly over the
      (out, in) tensor. q must divide tile_C (which guarantees q divides every
      dim of every MLP matrix that's a multiple of tile_C). p must not divide
      tile_C or outer_dim, which find_pq's two-pronged check enforces.
      Density = 2/(p+q).
    """
    dtype = dtype or torch.float32
    if tile_C is None:
        tile_C = min(in_features, out_features)
    if in_features % tile_C != 0 or out_features % tile_C != 0:
        raise ValueError(
            f"Both in_features ({in_features}) and out_features ({out_features}) "
            f"must be multiples of tile_C ({tile_C})."
        )
    if outer_dim is None:
        outer_dim = max(in_features, out_features)

    if structure in ("banded", "block_diagonal"):
        E_in = in_features // tile_C
        E_out = out_features // tile_C
        block_mask = make_structural_mask(
            structure, tile_C,
            block_size=block_size, band_width=band_width,
            device=device, dtype=dtype,
        )
        return block_mask.repeat(E_out, E_in)

    elif structure == "pq_strided":
        from tools.sparse_pq_embedding import find_pq, make_pq_mask
        # find_pq with C=tile_C, N=outer_dim ensures the resulting (p, q) is
        # valid for *every* MLP matrix that has both dims as multiples of
        # tile_C and outer_dim as the largest dim (so q | tile_C => q | dim,
        # and the p ∤ outer_dim check covers the largest dim's row-stride).
        p, q, _N_prime, _phantom = find_pq(
            tile_C, outer_dim, pq_density, mode=pq_mode
        )
        # Direct walk over (out_features, in_features); no phantom rows needed
        # since out_features is already a multiple of q (q | tile_C | out_features).
        mask = make_pq_mask(out_features, in_features, p, q, N_prime=out_features)
        return mask.to(device=device, dtype=dtype)

    else:
        raise ValueError(
            f"Unknown MLP structure: {structure!r}. Expected one of: "
            f"banded, block_diagonal, pq_strided."
        )


def build_structured_mlp_linear(
    in_features: int,
    out_features: int,
    structure: str,
    *,
    block_size: int = 64,
    band_width: int = 64,
    pq_density: float = 0.1,
    pq_mode: str = "structural",
    tile_C: int | None = None,
    outer_dim: int | None = None,
    bias: bool = True,
    device=None,
    dtype=None,
) -> "StructuredLinear":
    """Construct a StructuredLinear for an MLP layer using the given structure.

    `tile_C` controls per-block tiling; `outer_dim` controls (p, q) selection so
    the same tuple is valid across W1, W2, and middle linears. See `make_mlp_mask`.
    """
    mask = make_mlp_mask(
        in_features, out_features, structure,
        block_size=block_size, band_width=band_width,
        pq_density=pq_density, pq_mode=pq_mode,
        tile_C=tile_C, outer_dim=outer_dim,
        device=device, dtype=dtype,
    )
    return StructuredLinear(
        in_features, out_features, mask, bias=bias, device=device, dtype=dtype,
    )


def effective_param_count(
    module: nn.Module, *, requires_grad_only: bool = False
) -> int:
    """Sum the effective parameter count across `module`, accounting for
    StructuredLinear masks (count nonzeros) and MonarchLinear (already
    correctly sized parameters).

    For all other module types, sums direct (non-recursive) parameter numels.

    If `requires_grad_only=True`, frozen parameters (requires_grad=False) are
    excluded -- useful for distinguishing the trainable subset of params from
    the full effective count.
    """
    total = 0
    seen_param_ids = set()
    for m in module.modules():
        if isinstance(m, StructuredLinear):
            for p in m.parameters(recurse=False):
                if id(p) in seen_param_ids:
                    continue
                seen_param_ids.add(id(p))
                if requires_grad_only and not p.requires_grad:
                    continue
                if p is m.weight:
                    # Count nonzero positions in the mask instead of full numel
                    total += int(m._struct_mask.sum().item())
                else:
                    total += p.numel()
        else:
            for p in m.parameters(recurse=False):
                if id(p) in seen_param_ids:
                    continue
                seen_param_ids.add(id(p))
                if requires_grad_only and not p.requires_grad:
                    continue
                total += p.numel()
    return total
