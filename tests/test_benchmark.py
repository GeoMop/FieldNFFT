"""
tests/test_benchmark.py – Timing benchmark: our NUFFT vs GSTools SRF.

Measures wall-clock time for field generation across dimensions (2D, 3D),
grid sizes (N per axis), and correlation lengths (phi).

Metrics
-------
* median wall time over REPEATS runs  [ms]
* speedup = t_gstools / t_nufft

Test structure
--------------
test_benchmark_2d   – parametrised over N_VALUES_2D × PHI_VALUES
test_benchmark_3d   – parametrised over N_VALUES_3D × PHI_VALUES
test_benchmark_summary_figure  – log-log time vs N figure (--plots only)
test_benchmark_table           – prints a formatted table to stdout (always)

Run
---
    pytest tests/test_benchmark.py -v -s            # table printed to stdout
    pytest tests/test_benchmark.py -v -s --plots    # + saves figure
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grf

try:
    import gstools as gs
    HAS_GSTOOLS = True
except ImportError:
    HAS_GSTOOLS = False

pytestmark = pytest.mark.skipif(not HAS_GSTOOLS,
                                 reason="gstools not installed")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

L       = 100.0
SEED    = 42
REPEATS = 5   # median over this many runs per configuration

N_VALUES_2D = [32, 64, 128, 256]
N_VALUES_3D = [16, 24, 32, 48, 64]
PHI_VALUES  = [2.0, 5.0, 15.0]

# GSTools parametrisation: C(r) = exp(-pi*r² / 4*ls²)
# Our parametrisation:     C(r) = exp(-r²  / 2*phi²)
# Matching condition:      ls = phi * sqrt(pi/2)
_LS_FACTOR = np.sqrt(np.pi / 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_nufft(dim: int, N: int, phi: float) -> float:
    """Return median wall time [s] for generate_grf over REPEATS runs."""
    g = np.linspace(0, L, N)
    if dim == 2:
        xx, yy = np.meshgrid(g, g)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    corr    = grf.GaussianCorrelation(L=L, N_freq=N, phi=phi, dim=dim)
    weights = grf.make_white_noise(N, dim=dim, seed=SEED)

    # WARM-UP (odstraní overhead první inicializace C-vláken finufft)
    grf.generate_grf(pts, corr, weights=weights)

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        grf.generate_grf(pts, corr, weights=weights)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _time_gstools(dim: int, N: int, phi: float) -> float:
    """Return median wall time [s] for gs.SRF over REPEATS runs."""
    g = np.linspace(0, L, N)
    model = gs.Gaussian(dim=dim, var=1.0, len_scale=phi * _LS_FACTOR)
    
    # PŘESUNUTO MIMO CYKLUS: Inicializace pole je nákladná, měříme jen generování
    srf = gs.SRF(model, seed=SEED)
    
    # WARM-UP (odstraní overhead první alokace uvnitř knihovny)
    srf.structured([g] * dim)

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        srf.structured([g] * dim)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Parametrised tests – these also collect timing data into a module-level
# registry so that the summary figure and table can reuse it without re-running
# ---------------------------------------------------------------------------

_RESULTS: list[dict] = []   # [{dim, N, phi, t_nufft, t_gs, speedup}, ...]


@pytest.mark.parametrize("phi", PHI_VALUES)
@pytest.mark.parametrize("N",   N_VALUES_2D)
def test_benchmark_2d(N, phi):
    """Time 2D field generation; assert NUFFT is not slower than GSTools."""
    t_n  = _time_nufft(dim=2, N=N, phi=phi)
    t_gs = _time_gstools(dim=2, N=N, phi=phi)
    speedup = t_gs / t_n

    _RESULTS.append(dict(dim=2, N=N, phi=phi,
                         t_nufft=t_n, t_gs=t_gs, speedup=speedup))

    print(f"\n  2D N={N:3d} φ={phi:4.1f}  "
          f"NUFFT={t_n*1e3:7.1f} ms  "
          f"GSTools={t_gs*1e3:7.1f} ms  "
          f"speedup={speedup:.1f}×")

    assert speedup >= 1.0, (
        f"NUFFT ({t_n*1e3:.1f} ms) slower than GSTools ({t_gs*1e3:.1f} ms) "
        f"for 2D N={N} φ={phi}"
    )


@pytest.mark.parametrize("phi", PHI_VALUES)
@pytest.mark.parametrize("N",   N_VALUES_3D)
def test_benchmark_3d(N, phi):
    """Time 3D field generation; assert NUFFT is not slower than GSTools."""
    t_n  = _time_nufft(dim=3, N=N, phi=phi)
    t_gs = _time_gstools(dim=3, N=N, phi=phi)
    speedup = t_gs / t_n

    _RESULTS.append(dict(dim=3, N=N, phi=phi,
                         t_nufft=t_n, t_gs=t_gs, speedup=speedup))

    print(f"\n  3D N={N:3d} φ={phi:4.1f}  "
          f"NUFFT={t_n*1e3:7.1f} ms  "
          f"GSTools={t_gs*1e3:7.1f} ms  "
          f"speedup={speedup:.1f}×")

    assert speedup >= 1.0, (
        f"NUFFT ({t_n*1e3:.1f} ms) slower than GSTools ({t_gs*1e3:.1f} ms) "
        f"for 3D N={N} φ={phi}"
    )


# ---------------------------------------------------------------------------
# Summary table  (always printed, no --plots needed)
# ---------------------------------------------------------------------------

def test_benchmark_table():
    """Print a formatted summary table of all collected timing results.

    This test always passes; it just formats whatever _RESULTS contains.
    Run after the parametrised tests (-v -s) to see the full table.
    """
    if not _RESULTS:
        pytest.skip("No timing results yet – run test_benchmark_2d/3d first.")

    header = f"{'dim':>4}  {'N':>4}  {'phi':>5}  "
    header += f"{'NUFFT [ms]':>12}  {'GSTools [ms]':>13}  {'speedup':>9}"
    sep = "-" * len(header)

    rows_2d = [r for r in _RESULTS if r["dim"] == 2]
    rows_3d = [r for r in _RESULTS if r["dim"] == 3]

    lines = ["\n", sep, header, sep]
    for rows, label in [(rows_2d, "2D"), (rows_3d, "3D")]:
        if not rows:
            continue
        lines.append(f"  {label}")
        # Sort by phi then N
        for r in sorted(rows, key=lambda x: (x["phi"], x["N"])):
            lines.append(
                f"  {r['dim']:>3}  {r['N']:>4}  {r['phi']:>5.1f}  "
                f"{r['t_nufft']*1e3:>12.1f}  "
                f"{r['t_gs']*1e3:>13.1f}  "
                f"{r['speedup']:>8.1f}×"
            )
    lines.append(sep)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Summary figure  (--plots only)
# ---------------------------------------------------------------------------

def test_benchmark_summary_figure(plot_dir):
    """Log-log timing vs N figure for report (--plots only)."""
    if plot_dir is None:
        pytest.skip("Pass --plots to generate the benchmark figure.")
    if not _RESULTS:
        pytest.skip("No timing results – run parametrised tests first.")

    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    colors = {2.0: "#1f77b4", 5.0: "#ff7f0e", 15.0: "#2ca02c"}
    markers_nufft = {"o", "s", "^"}
    markers = list(markers_nufft)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, dim in enumerate([2, 3]):
        ax   = axes[ax_idx]
        rows = [r for r in _RESULTS if r["dim"] == dim]
        if not rows:
            continue

        for phi_idx, phi in enumerate(PHI_VALUES):
            sub = sorted([r for r in rows if r["phi"] == phi], key=lambda x: x["N"])
            if not sub:
                continue
            Ns      = [r["N"] for r in sub]
            pts     = [N ** dim for N in Ns]
            t_n     = [r["t_nufft"] * 1e3 for r in sub]
            t_gs    = [r["t_gs"]    * 1e3 for r in sub]
            mk      = markers[phi_idx % len(markers)]
            col     = colors[phi]

            ax.loglog(pts, t_n,  mk + "-",  color=col, lw=1.8, ms=7,
                      label=f"NUFFT φ={phi}")
            ax.loglog(pts, t_gs, mk + "--", color=col, lw=1.8, ms=7, alpha=0.6,
                      label=f"GSTools φ={phi}")

        ax.set_xlabel("Number of grid points  (N^dim)", fontsize=11)
        ax.set_ylabel("Median wall time [ms]", fontsize=11)
        ax.set_title(f"{dim}D field generation: NUFFT vs GSTools", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)

        # Add speedup annotations at the largest N per phi
        for phi in PHI_VALUES:
            sub = sorted([r for r in rows if r["phi"] == phi], key=lambda x: x["N"])
            if not sub:
                continue
            r   = sub[-1]
            pts_last = r["N"] ** dim
            ax.annotate(
                f"{r['speedup']:.1f}×",
                xy=(pts_last, r["t_nufft"] * 1e3),
                xytext=(pts_last * 1.05, r["t_nufft"] * 1e3 * 1.3),
                fontsize=8, color=colors[phi],
                arrowprops=dict(arrowstyle="-", color=colors[phi], lw=0.8),
            )

    # Unified legend: one entry per phi, solid=NUFFT, dashed=GSTools
    legend_handles = []
    for phi in PHI_VALUES:
        col = colors[phi]
        legend_handles.append(
            mlines.Line2D([], [], color=col, lw=2, label=f"φ={phi}")
        )
    legend_handles += [
        mlines.Line2D([], [], color="k", lw=2, ls="-",  label="NUFFT (solid)"),
        mlines.Line2D([], [], color="k", lw=2, ls="--", label="GSTools (dashed)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        f"NUFFT vs GSTools – wall time  (median of {REPEATS} runs, L={L})",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fpath = plot_dir / "test_benchmark.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {fpath}")