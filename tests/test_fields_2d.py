"""
tests/test_fields_2d.py – Numerical and graphical tests for 2D GRF generation.

Tests
-----
For each (GaussianCorrelation, ExponentialCorrelation) ×
         (sparse irregular, dense irregular):

  1. Reference field on a regular N×N grid (same weights).
  2. NUFFT field on an irregular grid (same weights).
  3. Interpolate (scipy.interpolate.griddata) to the regular grid.
  4. Assert relative L2 error < tolerance.

     rel_err = ||f_nufft_interp - f_ref||_2 / ||f_ref||_2

Run
---
    pytest tests/test_fields_2d.py -v
    pytest tests/test_fields_2d.py -v --plots
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import griddata

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grf


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

L    = 100.0
N2   = 64        # reference regular grid: N2 × N2
PHI  = 5.0
SEED = 42

# Irregular grid sizes
N_SPARSE_2D = 20 ** 2   #  400 pts  (≈ N2² / 10 )
N_DENSE_2D  = 64 ** 2   # 4096 pts  (= N2²)

# Relative L2 tolerances – calibrated per correlation type and grid density.
#
# Exponential fields are rougher than Gaussian (non-differentiable), so
# linear griddata interpolation introduces much larger errors, especially
# for sparse scatter. This is an expected and reportable result.
#
#                          Gaussian   Exponential
#   sparse (400 pts):      ~0.47      ~0.68
#   dense  (4096 pts):     ~0.04      ~0.27
#
TOL_SPARSE_2D = {"gaussian": 0.60, "exponential": 0.85}
TOL_DENSE_2D  = {"gaussian": 0.08, "exponential": 0.35}

def _tol(tol_dict, CorrClass):
    key = "gaussian" if "Gaussian" in CorrClass.__name__ else "exponential"
    return tol_dict[key]

CORR_CASES = [
    pytest.param(grf.GaussianCorrelation,    id="gaussian"),
    pytest.param(grf.ExponentialCorrelation, id="exponential"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corr(cls):
    return cls(L=L, N_freq=N2, phi=PHI, dim=2)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    # NaN pixels from griddata extrapolation outside convex hull → ignored
    mask = ~(np.isnan(a) | np.isnan(b))
    return float(np.linalg.norm((a - b)[mask]) / np.linalg.norm(b[mask]))


def _regular_grid():
    g = np.linspace(0, L, N2)
    xx, yy = np.meshgrid(g, g)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


# ---------------------------------------------------------------------------
# Tests – sparse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("CorrClass", CORR_CASES)
def test_2d_sparse_relative_l2(CorrClass, plot_dir):
    """NUFFT on sparse 2D scatter interpolated to regular grid (rel-L2 < TOL)."""
    corr    = _make_corr(CorrClass)
    weights = grf.make_white_noise(N2, dim=2, seed=SEED)
    xx, yy, pts_reg = _regular_grid()

    f_ref = grf.generate_grf(pts_reg, corr, weights=weights).reshape(N2, N2)

    rng = np.random.default_rng(SEED)
    pts_sp = np.column_stack([rng.uniform(0, L, N_SPARSE_2D),
                               rng.uniform(0, L, N_SPARSE_2D)])
    f_sp   = grf.generate_grf(pts_sp, corr, weights=weights)
    f_int  = griddata(pts_sp, f_sp, (xx, yy), method="linear")

    tol = _tol(TOL_SPARSE_2D, CorrClass)
    rel_err = _rel_l2(f_int, f_ref)
    print(f"\n  [{CorrClass.__name__}] 2D sparse rel-L2 = {rel_err:.4f}  (tol={tol})")
    assert rel_err < tol, (
        f"Sparse 2D relative L2 error {rel_err:.4f} exceeds tolerance {tol}"
    )

    if plot_dir is not None:
        _plot_2d_comparison(
            xx, yy, f_ref, pts_sp, f_sp, f_int, rel_err,
            title=f"2D sparse – {CorrClass.__name__}  φ={PHI}",
            fname=plot_dir / f"test_2d_sparse_{CorrClass.__name__.lower()[:4]}.png",
        )


# ---------------------------------------------------------------------------
# Tests – dense
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("CorrClass", CORR_CASES)
def test_2d_dense_relative_l2(CorrClass, plot_dir):
    """NUFFT on dense 2D scatter interpolated to regular grid (rel-L2 < TOL)."""
    corr    = _make_corr(CorrClass)
    weights = grf.make_white_noise(N2, dim=2, seed=SEED)
    xx, yy, pts_reg = _regular_grid()

    f_ref = grf.generate_grf(pts_reg, corr, weights=weights).reshape(N2, N2)

    rng = np.random.default_rng(SEED)
    pts_dn = np.column_stack([rng.uniform(0, L, N_DENSE_2D),
                               rng.uniform(0, L, N_DENSE_2D)])
    f_dn  = grf.generate_grf(pts_dn, corr, weights=weights)
    f_int = griddata(pts_dn, f_dn, (xx, yy), method="linear")

    tol = _tol(TOL_DENSE_2D, CorrClass)
    rel_err = _rel_l2(f_int, f_ref)
    print(f"\n  [{CorrClass.__name__}] 2D dense  rel-L2 = {rel_err:.4f}  (tol={tol})")
    assert rel_err < tol, (
        f"Dense 2D relative L2 error {rel_err:.4f} exceeds tolerance {tol}"
    )

    if plot_dir is not None:
        _plot_2d_comparison(
            xx, yy, f_ref, pts_dn, f_dn, f_int, rel_err,
            title=f"2D dense – {CorrClass.__name__}  φ={PHI}",
            fname=plot_dir / f"test_2d_dense_{CorrClass.__name__.lower()[:4]}.png",
        )


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def test_2d_summary_figure(plot_dir):
    """4-panel summary figure per correlation type (runs only with --plots)."""
    if plot_dir is None:
        pytest.skip("Pass --plots to generate the summary figure.")

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    xx, yy, pts_reg = _regular_grid()

    for CorrClass in [grf.GaussianCorrelation, grf.ExponentialCorrelation]:
        corr    = _make_corr(CorrClass)
        weights = grf.make_white_noise(N2, dim=2, seed=SEED)
        label   = CorrClass.__name__.replace("Correlation", "")

        f_ref = grf.generate_grf(pts_reg, corr, weights=weights).reshape(N2, N2)
        vmin, vmax = f_ref.min(), f_ref.max()

        pts_sp = np.column_stack([rng.uniform(0, L, N_SPARSE_2D),
                                   rng.uniform(0, L, N_SPARSE_2D)])
        f_sp   = grf.generate_grf(pts_sp, corr, weights=weights)
        f_si   = griddata(pts_sp, f_sp, (xx, yy), method="linear")

        pts_dn = np.column_stack([rng.uniform(0, L, N_DENSE_2D),
                                   rng.uniform(0, L, N_DENSE_2D)])
        f_dn  = grf.generate_grf(pts_dn, corr, weights=weights)
        f_di  = griddata(pts_dn, f_dn, (xx, yy), method="linear")

        rel_s = _rel_l2(f_si, f_ref)
        rel_d = _rel_l2(f_di, f_ref)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        kw = dict(extent=[0, L, 0, L], origin="lower", cmap="viridis",
                  vmin=vmin, vmax=vmax)

        im0 = axes[0].imshow(f_ref, **kw)
        axes[0].set_title(f"Reference ({N2}×{N2})")
        plt.colorbar(im0, ax=axes[0])

        sc = axes[1].scatter(pts_sp[:, 0], pts_sp[:, 1],
                             c=f_sp, cmap="viridis", s=18, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Irregular scatter ({N_SPARSE_2D} pts)")
        axes[1].set_xlim(0, L); axes[1].set_ylim(0, L)
        axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.2)
        plt.colorbar(sc, ax=axes[1])

        im2 = axes[2].imshow(f_si, **kw)
        axes[2].plot(pts_sp[:, 0], pts_sp[:, 1], ".", color="white",
                     ms=2.5, alpha=0.5)
        axes[2].set_title(f"Sparse + griddata  (rel-L2={rel_s:.3f})")
        plt.colorbar(im2, ax=axes[2])

        im3 = axes[3].imshow(f_di, **kw)
        axes[3].plot(pts_dn[:, 0], pts_dn[:, 1], ".", color="white",
                     ms=1.5, alpha=0.3)
        axes[3].set_title(f"Dense + griddata  (rel-L2={rel_d:.3f})")
        plt.colorbar(im3, ax=axes[3])

        plt.suptitle(f"GRF 2D – {label}  φ={PHI}", fontsize=13)
        plt.tight_layout()
        fpath = plot_dir / f"test_2d_summary_{label.lower()[:4]}.png"
        plt.savefig(fpath, dpi=150)
        plt.close()
        print(f"\n  Figure saved: {fpath}")


# ---------------------------------------------------------------------------
# Internal plotting helper
# ---------------------------------------------------------------------------

def _plot_2d_comparison(xx, yy, f_ref, pts_irr, f_irr, f_int, rel_err,
                         title: str, fname: Path) -> None:
    import matplotlib.pyplot as plt

    vmin, vmax = f_ref.min(), f_ref.max()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    kw = dict(extent=[0, L, 0, L], origin="lower", cmap="viridis",
              vmin=vmin, vmax=vmax)

    im0 = axes[0].imshow(f_ref, **kw)
    axes[0].set_title("Reference (regular grid)")
    plt.colorbar(im0, ax=axes[0])

    sc = axes[1].scatter(pts_irr[:, 0], pts_irr[:, 1],
                         c=f_irr, cmap="viridis", s=12, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"NUFFT ({len(pts_irr)} pts)")
    axes[1].set_xlim(0, L); axes[1].set_ylim(0, L); axes[1].set_aspect("equal")
    plt.colorbar(sc, ax=axes[1])

    diff = f_int - f_ref
    lim = np.nanpercentile(np.abs(diff), 97)
    im2 = axes[2].imshow(diff, extent=[0, L, 0, L], origin="lower",
                          cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title(f"f_interp − f_ref  (rel-L2={rel_err:.3f})")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"\n  Figure saved: {fname}")