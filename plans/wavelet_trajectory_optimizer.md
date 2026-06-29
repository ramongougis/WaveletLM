# Wavelet Trajectory Optimizer (multi-scale momentum + curvature over training steps)

## Status

**Proposed — exploratory (2026-06-30).** A sibling to [the GWT gradient compressor](wavelet_optimizer.md), but a *different optimizer*, not a compression scheme. GWT runs the wavelet down the **parameter axis** (compress one gradient). This runs it down the **time / training-step axis** (decompose the *sequence* of gradients), which turns into a unified momentum + curvature method.

## Core idea

The gradients across steps form a time series `g₁, g₂, g₃, …`. Wavelet-decompose that trajectory (per parameter, or per block) into scales:

- **Coarse (lowpass) coefficients** = a smoothed gradient history → **momentum**, and multi-scale → multi-timescale momentum (the consistent descent direction, "where am I heading").
- **Detail (highpass = discrete-derivative) coefficients** = `g_t − g_{t-1}`-type differences → to first order `H·(θ_t − θ_{t-1})`, a **Hessian-vector product**. This is the **secant equation** that powers quasi-Newton. So the detail branch *is* curvature / second-order information.

The wavelet detail being a discrete derivative (vanishing moments → finite difference; lifting's `d = odd − P(even)` is a prediction residual) is the hook: **one transform yields momentum (coarse) and curvature (detail) together.** In WaveletLM you'd run it through the model's **learned** lifting, whose predict-residual is a *learned* finite difference → a learned, multi-scale curvature estimate in the model's own basis (the same self-consistency story as GWT).

## Honest caveats (this is the bigger, riskier bet)

- **The coarse half mostly reinvents known things.** Momentum ≡ a low-pass filter on the gradient stream (well known); Aggregated / multi-timescale momentum is already a crude multi-scale lowpass. So the **novelty must live in the detail-as-curvature branch.**
- **That branch is where the risk is.** Finite-difference curvature is **noisy** — exactly why quasi-Newton/L-BFGS, more powerful in principle, lost to Adam at deep-learning scale. The wavelet's multi-scale denoising is the optimistic angle, but "beats Adam" is a high bar most second-order-flavored methods miss.
- **Memory cost is the opposite of GWT.** You need a rolling window of past gradients to transform over time (like L-BFGS history) → it *adds* a history buffer rather than saving state. Different point on the resource axis.

## Minimal first test (don't build L-BFGS-in-wavelets)

- A **few lifting levels over a short gradient-history window** (e.g., 8–16 past gradients), not a full trajectory.
- Coarse scales → multi-timescale momentum; **one** detail level → a cheap curvature correction term. Keep it small.
- Use the model's **learned lifting** (frozen snapshot, as in GWT) for the self-consistent basis; compare against a fixed Haar control.
- A/B on the C=1024 baseline vs Adam/Adagrad: convergence curve, final BPB, wall-clock, and the added memory.

## Relationship

- **Orthogonal to [GWT](wavelet_optimizer.md):** GWT compresses the parameter-axis gradient (saves state); this preconditions along the time axis (adds curvature). They could even compose (GWT-compress the trajectory buffer).
- **Distinct from the frequency-space target-propagation idea** (2026-06-30): that changes the *loss / where error is defined*; this changes the *optimizer mechanics*. Three separate levers on "optimize in frequency space."
