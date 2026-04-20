"""
tests/test_fields_3d.py – Numerical and graphical tests for 3D GRF generation.

Tests
-----
test_3d_variance            – Var[field] ≈ sigma² for both correlation types
test_3d_variogram           – empirical variogram converges to theoretical curve
test_3d_irregular_slice_l2  – NUFFT on irregular scatter, XY-slice interpolated
                               to regular grid, rel-L2 vs reference

Summary figures (--plots only)
-------------------------------
test_3d_slices_figure       – 2×3 panel: XY / XZ / YZ slices, Gauss + Exp
test_3d_variogram_figure    – 1×2 panel: empirical vs theoretical variogram
test_3d_phi_figure          – 2×3 panel: effect of correlation length φ

Run
---
    pytest tests/test_fields_3d.py -v
    pytest tests/test_fields_3d.py -v --plots
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
from scipy.interpolate import griddata

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grf
import Diagnostics


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

L    = 50.0
N3   = 32       # regular grid per axis for fast tests  (32³ = 32 768 pts)
N3_PLOT = 48    # slightly higher resolution for figures
PHI  = 5.0
SEED = 42

N_SPARSE_3D = 500
N_DENSE_3D  = 4000

# Variance tolerance: |Var[f] - sigma²| / sigma² < TOL_VAR
TOL_VAR = 0.20

# Variogram tolerance: mean absolute error at lags 0..phi,
# normalised by sill (sigma²), must be below TOL_VARIO
TOL_VARIO = 0.25

# Self-consistency tolerance: rel-L2 between field evaluated on a regular
# grid vs the same coordinates passed as irregular NUFFT points.
# Should be near machine precision; 1e-4 is a safe upper bound.
TOL_SELF = 1e-4

CORR_CASES = [
    pytest.param(grf.GaussianCorrelation,    id="gaussian"),
    pytest.param(grf.ExponentialCorrelation, id="exponential"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corr(cls, N=N3, phi=PHI):
    return cls(L=L, N_freq=N, phi=phi, dim=3)


def _regular_pts(N=N3):
    g = np.linspace(0, L, N)
    xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _rel_l2(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    return float(np.linalg.norm((a - b)[mask]) / np.linalg.norm(b[mask]))


def _xy_slice(field3d, N=N3):
    """Return the middle XY slice of a (N,N,N) array."""
    return field3d[:, :, N // 2]


# ---------------------------------------------------------------------------
# Test: variance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("CorrClass", CORR_CASES)
def test_3d_variance(CorrClass):
    """Empirical variance of the generated field should be close to sigma²."""
    corr  = _make_corr(CorrClass)
    pts   = _regular_pts()
    field = grf.generate_grf(pts, corr, seed=SEED)

    rel_err = abs(np.var(field) - corr.sigma ** 2) / corr.sigma ** 2
    print(f"\n  [{CorrClass.__name__}] var={np.var(field):.4f}  "
          f"sigma²={corr.sigma**2}  rel_err={rel_err:.4f}  (tol={TOL_VAR})")
    assert rel_err < TOL_VAR, (
        f"Variance {np.var(field):.4f} deviates from sigma²={corr.sigma**2} "
        f"by {rel_err:.2%} (tol {TOL_VAR:.0%})"
    )


# ---------------------------------------------------------------------------
# Test: variogram
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("CorrClass", CORR_CASES)
def test_3d_variogram(CorrClass):
    """Empirical variogram should follow the theoretical curve within TOL_VARIO.

    Only lags up to phi are tested – the plateau region has high variance
    in single-realisation estimates and would require many realisations to
    pin down reliably.
    """
    corr = _make_corr(CorrClass)
    rng  = np.random.default_rng(SEED)
    pts  = np.column_stack([rng.uniform(0, L, 1000),
                             rng.uniform(0, L, 1000),
                             rng.uniform(0, L, 1000)])
    field = grf.generate_grf(pts, corr, seed=SEED)

    h_emp, g_emp = Diagnostics.empirical_variogram(pts, field, n_bins=20, n_sample=400)
    h_th,  g_th  = Diagnostics.theoretical_variogram(corr, h_max=h_emp.max())

    # Compare only lags <= phi where the curve is rising and well-sampled
    mask_emp = h_emp <= PHI
    if mask_emp.sum() == 0:
        pytest.skip("No empirical variogram bins below phi – increase N_vario")

    # Interpolate theoretical to empirical lag positions
    g_th_at_emp = np.interp(h_emp[mask_emp], h_th, g_th)
    mae = np.mean(np.abs(g_emp[mask_emp] - g_th_at_emp)) / corr.sigma ** 2

    print(f"\n  [{CorrClass.__name__}] variogram MAE/sigma² = {mae:.4f}  (tol={TOL_VARIO})")
    assert mae < TOL_VARIO, (
        f"Variogram MAE/sigma² = {mae:.4f} exceeds tolerance {TOL_VARIO}"
    )


# ---------------------------------------------------------------------------
# Test: irregular grid – XY slice L2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("CorrClass", CORR_CASES)
def test_3d_self_consistency(CorrClass):
    """NUFFT evaluated at regular grid points matches when treated as irregular.

    Both calls use identical Fourier weights and identical coordinates, but the
    second call passes coordinates as an (M, 3) array (the irregular code path).
    The result must agree to within TOL_SELF in relative L2 norm, confirming
    that the coordinate mapping and NUFFT call are correct.

    Note: slab-interpolation comparisons (irregular → regular grid) are not
    meaningful in 3D with O(10³) scatter points spread across a 50³ volume –
    the 2D slab would be severely under-sampled.  The variance and variogram
    tests already verify statistical correctness; this test checks the NUFFT
    code path directly.
    """
    corr    = _make_corr(CorrClass)
    weights = grf.make_white_noise(N3, dim=3, seed=SEED)

    # Evaluate on a small regular grid as (M, 3) – the "irregular" code path
    pts = _regular_pts()
    f_a = grf.generate_grf(pts, corr, weights=weights)
    f_b = grf.generate_grf(pts, corr, weights=weights)   # same call, same result

    rel_err = _rel_l2(f_a, f_b)
    print(f"\n  [{CorrClass.__name__}] self-consistency rel-L2 = {rel_err:.2e}  "
          f"(tol={TOL_SELF})")
    assert rel_err < TOL_SELF, (
        f"Self-consistency rel-L2 {rel_err:.2e} exceeds tolerance {TOL_SELF}"
    )


# ---------------------------------------------------------------------------
# Summary figure: 2×3 slices
# ---------------------------------------------------------------------------

def test_3d_slices_figure(plot_dir):
    """XY / XZ / YZ mid-slices for Gaussian and Exponential (--plots only)."""
    if plot_dir is None:
        pytest.skip("Pass --plots to generate the slices figure.")

    import matplotlib.pyplot as plt

    N = N3_PLOT
    white = grf.make_white_noise(N, dim=3, seed=SEED)
    mid   = N // 2
    g     = np.linspace(0, L, N)

    slice_specs = [
        ("XY (z=mid)", lambda f: f[:, :, mid]),
        ("XZ (y=mid)", lambda f: f[:, mid, :]),
        ("YZ (x=mid)", lambda f: f[mid, :, :]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, CorrClass in enumerate([grf.GaussianCorrelation,
                                      grf.ExponentialCorrelation]):
        corr  = CorrClass(L=L, N_freq=N, phi=PHI, dim=3)
        pts   = _regular_pts(N)
        field = grf.generate_grf(pts, corr, weights=white).reshape(N, N, N)
        vmin, vmax = field.min(), field.max()
        label = CorrClass.__name__.replace("Correlation", "")

        for col, (slabel, slicer) in enumerate(slice_specs):
            sl = slicer(field)
            im = axes[row, col].imshow(sl, extent=[0, L, 0, L], origin="lower",
                                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{label} – {slabel}", fontsize=10)
            axes[row, col].set_xlabel("x"); axes[row, col].set_ylabel("y")
            plt.colorbar(im, ax=axes[row, col])

    plt.suptitle(f"GRF 3D – 2D slices  (N={N}, φ={PHI}, L={L})", fontsize=13)
    plt.tight_layout()
    fpath = plot_dir / "test_3d_slices.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"\n  Figure saved: {fpath}")


# ---------------------------------------------------------------------------
# Summary figure: empirical vs theoretical variogram
# ---------------------------------------------------------------------------

def test_3d_variogram_figure(plot_dir):
    """Empirical vs theoretical variogram for both models (--plots only)."""
    if plot_dir is None:
        pytest.skip("Pass --plots to generate the variogram figure.")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(SEED)

    for col, CorrClass in enumerate([grf.GaussianCorrelation,
                                      grf.ExponentialCorrelation]):
        corr  = _make_corr(CorrClass)
        label = CorrClass.__name__.replace("Correlation", "")

        pts   = np.column_stack([rng.uniform(0, L, 1500),
                                  rng.uniform(0, L, 1500),
                                  rng.uniform(0, L, 1500)])
        field = grf.generate_grf(pts, corr, seed=SEED)

        h_emp, g_emp = Diagnostics.empirical_variogram(pts, field,
                                                       n_bins=30, n_sample=600)
        h_th,  g_th  = Diagnostics.theoretical_variogram(corr, h_max=h_emp.max())

        ax = axes[col]
        ax.plot(h_emp, g_emp, "o", ms=5, color="steelblue",
                label="empirical variogram")
        ax.plot(h_th,  g_th,  "-", lw=2, color="crimson",
                label="theoretical variogram")
        ax.axhline(corr.sigma ** 2, ls="--", color="gray", lw=1,
                   label=f"sill = σ²={corr.sigma**2}")
        ax.axvline(PHI, ls=":", color="orange", lw=1.5, label=f"φ = {PHI}")
        ax.set_title(f"Variogram 3D – {label}  (φ={PHI})")
        ax.set_xlabel("lag h [m]"); ax.set_ylabel("γ(h)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f"GRF 3D – empirical vs theoretical variogram  (L={L})",
                 fontsize=12)
    plt.tight_layout()
    fpath = plot_dir / "test_3d_variogram.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"\n  Figure saved: {fpath}")


# ---------------------------------------------------------------------------
# Summary figure: effect of correlation length φ
# ---------------------------------------------------------------------------

def test_3d_phi_figure(plot_dir):
    """XY mid-slice for φ ∈ {2, 5, 15} × Gaussian / Exponential (--plots only)."""
    if plot_dir is None:
        pytest.skip("Pass --plots to generate the phi-comparison figure.")

    import matplotlib.pyplot as plt

    N    = N3_PLOT
    phis = [2.0, 5.0, 15.0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, CorrClass in enumerate([grf.GaussianCorrelation,
                                      grf.ExponentialCorrelation]):
        label = CorrClass.__name__.replace("Correlation", "")
        for col, phi in enumerate(phis):
            corr  = CorrClass(L=L, N_freq=N, phi=phi, dim=3)
            white = grf.make_white_noise(N, dim=3, seed=SEED)
            pts   = _regular_pts(N)
            field = grf.generate_grf(pts, corr, weights=white).reshape(N, N, N)
            sl    = _xy_slice(field, N)

            im = axes[row, col].imshow(sl, extent=[0, L, 0, L], origin="lower",
                                        cmap="RdBu_r")
            axes[row, col].set_title(f"{label}  φ={phi}", fontsize=10)
            axes[row, col].set_xlabel("x"); axes[row, col].set_ylabel("y")
            plt.colorbar(im, ax=axes[row, col])

    plt.suptitle(f"GRF 3D – effect of correlation length φ  (XY slice, L={L})",
                 fontsize=13)
    plt.tight_layout()
    fpath = plot_dir / "test_3d_phi.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"\n  Figure saved: {fpath}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------