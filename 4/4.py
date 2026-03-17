"""
Porovnání našeho FFT/NUFFT přístupu s GSTools.

GSTools používá RandMeth (Randomization Method):
  u(x) = sqrt(sigma^2 / N) * sum_j [ Z1_j * cos(<k_j, x>) + Z2_j * sin(<k_j, x>) ]
  kde k_j jsou náhodné vzorky ze spektrální hustoty S(omega)
  a Z1, Z2 jsou nezávislé N(0,1) vzorky

Náš přístup (FFT/NUFFT):
  u(x) = IFFT[ white * sqrt(S(k)) ]
  kde k jsou PRAVIDELNÉ frekvence a white je komplexní N(0,1)

-> algoritmy jsou různé, srovnáváme proto statisticky přes variogram
"""

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs
import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "3" / "3.2d.py"
spec = importlib.util.spec_from_file_location("grf_2d_module", module_path)
grf_2d_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grf_2d_module)

GaussianCorrelation = grf_2d_module.GaussianCorrelation
make_white_noise = grf_2d_module.make_white_noise
generate_grf_nufft = grf_2d_module.generate_grf_nufft


L, N, phi, sigma = 100.0, 512, 5.0, 1.0
seed = 42

# =============================================================================
# 1. NAŠE METODA – 1D pole
# =============================================================================
x = np.linspace(0, L, N)
corr = GaussianCorrelation(L, N, phi, sigma=sigma, dim=1)
white = make_white_noise(N, dim=1, seed=seed)
field_ours = generate_grf_nufft(x, corr, weights=white)

# =============================================================================
# 2. GSTOOLS – 1D pole
# =============================================================================
model = gs.Gaussian(dim=1, var=sigma**2, len_scale=phi)
srf = gs.SRF(model, seed=seed)
field_gs = np.asarray(srf(x)).ravel()

# --- Var 1: reimplementace RandMeth pomocí interních atributů GSTools ---
# po zavolání srf(x) jsou dostupné:
#   srf.generator._z_1, _z_2  : N(0,1) vzorky (shape: mode_no,)
#   srf.generator._cov_sample  : náhodné frekvence k_j (shape: dim, mode_no)
z1   = srf.generator._z_1          # N(0,1)
z2   = srf.generator._z_2          # N(0,1)
k_j  = srf.generator._cov_sample   # shape (1, mode_no) pro 1D
mode_no = len(z1)

# u(x) = sqrt(sigma^2 / N) * sum_j [ Z1_j*cos(k_j*x) + Z2_j*sin(k_j*x) ]
# k_j * x  ->  (mode_no, N) matice skalárních součinů
kx = k_j[0, :, np.newaxis] * x[np.newaxis, :]   # (mode_no, N)
field_randmeth = np.sqrt(sigma**2 / mode_no) * np.sum(
    z1[:, np.newaxis] * np.cos(kx) + z2[:, np.newaxis] * np.sin(kx), axis=0
)

# =============================================================================
# 3. EMPIRICKÝ VARIOGRAM  gamma(h) = 0.5 * E[(f(x) - f(x+h))^2]
# =============================================================================

def empirical_variogram(x, field, n_bins=30):
    """Odhadne variogram z jedné realizace pomocí všech párů bodů."""
    h_vals, gamma_vals = [], []
    N = len(x)
    # bereme podvzorek párů aby to nebylo O(N^2) příliš pomalé
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(N, 300), replace=False)
    xi, fi = x[idx], field[idx]

    pairs_h, pairs_sq = [], []
    for i in range(len(xi)):
        for j in range(i + 1, len(xi)):
            pairs_h.append(abs(xi[i] - xi[j]))
            pairs_sq.append((fi[i] - fi[j])**2)

    pairs_h = np.array(pairs_h)
    pairs_sq = np.array(pairs_sq)

    bins = np.linspace(0, pairs_h.max(), n_bins + 1)
    for k in range(n_bins):
        mask = (pairs_h >= bins[k]) & (pairs_h < bins[k+1])
        if mask.sum() > 5:
            h_vals.append(0.5 * (bins[k] + bins[k+1]))
            gamma_vals.append(0.5 * pairs_sq[mask].mean())

    return np.array(h_vals), np.array(gamma_vals)

# teoretický variogram: gamma(h) = sigma^2 * (1 - C(h)/sigma^2)
#   Gaussovský: C(h) = sigma^2 * exp(-h^2 / 2*phi^2)
h_theory = np.linspace(0, L/2, 200)
gamma_theory = sigma**2 * (1 - np.exp(-0.5 * (h_theory / phi)**2))

h_ours, g_ours = empirical_variogram(x, field_ours)
h_gs,   g_gs   = empirical_variogram(x, field_gs)
h_rm,   g_rm   = empirical_variogram(x, field_randmeth)

# =============================================================================
# GRAFY
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# pole: GSTools vs naše reimplementace RandMeth (měly by být identické)
axes[0].plot(x, field_gs,       lw=1.5, label="GSTools")
axes[0].plot(x, field_randmeth, lw=1,   label="naše RandMeth", linestyle='--', alpha=0.8)
axes[0].plot(x, field_ours,     lw=1,   label="naše NUFFT",    alpha=0.6)
axes[0].set_title(f"1D realizace (φ={phi})")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# variogram – naše vs GSTools
axes[1].plot(h_ours, g_ours, 'o-', ms=4, label="empirický – naše NUFFT")
axes[1].plot(h_gs,   g_gs,   's-', ms=4, label="empirický – GSTools")
axes[1].plot(h_rm,   g_rm,   '^-', ms=4, label="empirický – naše RandMeth")
axes[1].plot(h_theory, gamma_theory, 'k--', lw=2, label="teoretický")
axes[1].set_title("Empirický variogram")
axes[1].set_xlabel("h [m]"); axes[1].set_ylabel("γ(h)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# variogram pouze GSTools (gs má vestavěný)
bin_edges = np.linspace(0, L/2, 31)
bin_centers, vario_gs = gs.vario_estimate([x], field_gs, bin_edges=bin_edges)
axes[2].scatter(bin_centers, vario_gs,
                s=20, label="gs.vario_estimate")
axes[2].plot(h_theory, gamma_theory, 'k--', lw=2, label="teoretický")
axes[2].set_title("GSTools vario_estimate")
axes[2].set_xlabel("h [m]"); axes[2].set_ylabel("γ(h)")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.suptitle(f"Srovnání naše NUFFT vs GSTools  (Gaussovská korelace, φ={phi})")
plt.tight_layout()
plt.show()