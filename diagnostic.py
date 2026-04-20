"""
diagnostic.py – Diagnostic tools for Gaussian Random Field analysis.

Functions
---------
empirical_variogram     – estimate the empirical variogram from a field realization
theoretical_variogram   – compute the exact theoretical variogram from a correlation model

These utilities are intentionally separated from the core generation module (grf.py)
so that grf.py remains a pure field-generation library with no diagnostic overhead.

Typical usage
-------------
>>> import numpy as np
>>> import grf
>>> import diagnostic
>>> corr = grf.GaussianCorrelation(L=100.0, N_freq=512, phi=5.0, dim=1)
>>> x = np.linspace(0, 100.0, 512)
>>> field = grf.generate_grf(x, corr, seed=42)
>>> h_emp, g_emp = diagnostic.empirical_variogram(x, field)
>>> h_th, g_th   = diagnostic.theoretical_variogram(corr, h_max=50.0)
"""

from __future__ import annotations

import numpy as np

# Type alias so that diagnostic.py does not force grf to be imported
# when only the function signatures are needed.
_CorrModel = object  # GaussianCorrelation | ExponentialCorrelation


# =============================================================================
# EMPIRICAL VARIOGRAM
# =============================================================================

def empirical_variogram(
    x: np.ndarray,
    field: np.ndarray,
    n_bins: int = 30,
    n_sample: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the empirical variogram from a single GRF realization.

    The variogram (or semi-variogram) is defined as:

        gamma(h) = 0.5 * E[(f(x) - f(x + h))^2]

    For a stationary isotropic field this depends only on the lag distance
    ``h = |x - x'|``.  The estimator groups all point pairs by lag distance
    into ``n_bins`` equal-width bins and computes the mean squared difference
    within each bin.

    Supports 1D, 2D, and 3D point sets.  A random subsample of at most
    ``n_sample`` points is used to keep the O(M^2) pairwise cost tractable.

    Parameters
    ----------
    x : np.ndarray
        Point coordinates.

        * 1D: shape ``(M,)``
        * 2D: shape ``(M, 2)``
        * 3D: shape ``(M, 3)``

    field : np.ndarray, shape ``(M,)``
        Field values at the corresponding points in ``x``.
    n_bins : int
        Number of lag bins.  Default 30.
    n_sample : int
        Maximum number of randomly selected points used to form pairs.
        Larger values give a better estimate but increase compute time
        quadratically.  Default 300.

    Returns
    -------
    h_vals : np.ndarray, shape ``(K,)``  where  K ≤ n_bins
        Lag bin centres [same units as ``x``].
        Bins containing fewer than 5 pairs are silently dropped.
    g_vals : np.ndarray, shape ``(K,)``
        Estimated variogram values ``gamma(h)`` at the bin centres.

    Notes
    -----
    The estimator is the classical Matheron estimator:

        gamma_hat(h) = (1 / 2*|N(h)|) * Σ_{(i,j) in N(h)} (f_i - f_j)^2

    where N(h) is the set of pairs whose lag distance falls in the h-bin.
    """
    rng = np.random.default_rng(0)
    x = np.asarray(x)
    field = np.asarray(field)

    idx = rng.choice(len(field), size=min(len(field), n_sample), replace=False)
    xi, fi = x[idx], field[idx]

    # Pairwise distances
    if xi.ndim == 1:
        dists = np.abs(xi[:, None] - xi[None, :])
    else:
        diff = xi[:, None, :] - xi[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))

    i_u, j_u = np.triu_indices(len(xi), k=1)
    pairs_h  = dists[i_u, j_u]
    pairs_sq = (fi[i_u] - fi[j_u]) ** 2

    bins = np.linspace(0.0, pairs_h.max(), n_bins + 1)
    h_vals, g_vals = [], []
    for k in range(n_bins):
        mask = (pairs_h >= bins[k]) & (pairs_h < bins[k + 1])
        if mask.sum() > 5:
            h_vals.append(0.5 * (bins[k] + bins[k + 1]))
            g_vals.append(0.5 * pairs_sq[mask].mean())

    return np.array(h_vals), np.array(g_vals)


# =============================================================================
# THEORETICAL VARIOGRAM
# =============================================================================

def theoretical_variogram(
    corr: _CorrModel,
    h_max: float | None = None,
    n_points: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact theoretical variogram for a given correlation model.

    The theoretical variogram is:

        gamma(h) = C(0) - C(h) = sigma^2 - C(h)

    where C(h) is the correlation function evaluated at lag ``h``.

    Parameters
    ----------
    corr : GaussianCorrelation | ExponentialCorrelation
        Correlation model.  Must expose ``sigma``, ``phi``, and ``L``
        attributes (standard interface from ``grf.py``).
    h_max : float or None
        Maximum lag distance to evaluate [same units as ``corr.L``].
        Defaults to ``corr.L / 2``.
    n_points : int
        Number of lag values at which to evaluate the variogram.  Default 300.

    Returns
    -------
    h : np.ndarray, shape ``(n_points,)``
        Lag distances from 0 to ``h_max``.
    gamma : np.ndarray, shape ``(n_points,)``
        Theoretical variogram values.

    Notes
    -----
    Import the concrete correlation classes from ``grf`` to avoid a circular
    dependency:

    >>> import grf, diagnostic
    >>> corr = grf.GaussianCorrelation(L=100.0, N_freq=512, phi=5.0)
    >>> h, g = diagnostic.theoretical_variogram(corr)
    """
    import grf as _grf   # local import avoids circular dependency at module level

    if h_max is None:
        h_max = corr.L / 2.0

    h = np.linspace(0.0, h_max, n_points)
    sigma2 = corr.sigma ** 2

    if isinstance(corr, _grf.GaussianCorrelation):
        C_h = sigma2 * np.exp(-0.5 * (h / corr.phi) ** 2)
    elif isinstance(corr, _grf.ExponentialCorrelation):
        C_h = sigma2 * np.exp(-h / corr.phi)
    else:
        raise TypeError(
            f"theoretical_variogram: unsupported correlation model {type(corr).__name__}"
        )

    return h, sigma2 - C_h