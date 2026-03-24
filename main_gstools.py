"""
main_gstools.py – Bod 5: Porovnání naší NUFFT metody s GSTools

Srovnání přes empirický variogram – obě metody by měly konvergovat
k teoretické křivce gamma(h) = sigma^2 * (1 - exp(-h^2 / 2*phi^2))
"""

import importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gstools as gs
from scipy.interpolate import griddata

spec = importlib.util.spec_from_file_location("grf", Path(__file__).parent / "grf.py")
grf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grf)

L, N, phi, seed = 100.0, 1024, 5.0, 42

# =============================================================================
# 1D – jedna realizace
# =============================================================================

x = np.linspace(0, L, N)
corr  = grf.GaussianCorrelation(L, N, phi, sigma=1.0, dim=1)
model = gs.Gaussian(dim=1, var=1.0, len_scale=phi)

white        = grf.make_white_noise(N, dim=1, seed=seed)
field_nufft  = grf.generate_grf(x, corr, weights=white)
field_gs     = np.asarray(gs.SRF(model, seed=seed)(x)).ravel()

# =============================================================================
# VARIOGRAM – průměr přes N_real realizací
# =============================================================================

N_real    = 100
n_bins    = 40
bin_edges = np.linspace(0, L / 2, n_bins + 1)
bin_c     = 0.5 * (bin_edges[:-1] + bin_edges[1:])

g_nufft  = np.zeros(n_bins)
g_gs     = np.zeros(n_bins)
cnt_nufft = np.zeros(n_bins, dtype=int)  # počet realizací které přispěly
cnt_gs    = np.zeros(n_bins, dtype=int)

for i in range(N_real):
    w   = grf.make_white_noise(N, dim=1, seed=i)
    f_n = grf.generate_grf(x, corr, weights=w)
    _, g = grf.empirical_variogram(x, f_n, n_bins=n_bins)
    if len(g) == n_bins:
        g_nufft  += g
        cnt_nufft += 1

    f_g = np.asarray(gs.SRF(model, seed=i)(x)).ravel()
    _, g = grf.empirical_variogram(x, f_g, n_bins=n_bins)
    if len(g) == n_bins:
        g_gs  += g
        cnt_gs += 1

# dělíme skutečným počtem přispívajících realizací (ne vždy N_real)
g_nufft = np.divide(g_nufft, cnt_nufft, where=cnt_nufft > 0)
g_gs    = np.divide(g_gs,    cnt_gs,    where=cnt_gs    > 0)

h_th     = np.linspace(0, L / 2, 300)
gamma_th = 1 - np.exp(-0.5 * (h_th / phi)**2)

# =============================================================================
# 2D
# =============================================================================

N2      = 128
g2      = np.linspace(0, L, N2)
xx, yy  = np.meshgrid(g2, g2)
rng_np  = np.random.default_rng(seed)
pts_irr = np.column_stack([rng_np.uniform(0, L, N2**2),
                            rng_np.uniform(0, L, N2**2)])

corr2d        = grf.GaussianCorrelation(L, N2, phi, sigma=1.0, dim=2)
white2d       = grf.make_white_noise(N2, dim=2, seed=seed)
f_irr2d       = grf.generate_grf(pts_irr, corr2d, weights=white2d)
field2d_nufft = griddata(pts_irr, f_irr2d, (xx, yy), method='linear')

model2d    = gs.Gaussian(dim=2, var=1.0, len_scale=phi)
field2d_gs = gs.SRF(model2d, seed=seed).structured([g2, g2])

# =============================================================================
# GRAFY
# =============================================================================

fig = plt.figure(figsize=(16, 9))
layout = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

# 1D realizace
ax0 = fig.add_subplot(layout[0, 0])
ax0.plot(x, field_gs,    lw=1.2, label="GSTools")
ax0.plot(x, field_nufft, lw=1,   label="naše NUFFT", alpha=0.8)
ax0.set_title(f"1D realizace (φ={phi})")
ax0.set_xlabel("x [m]")
ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

# variogram průměr
ax1 = fig.add_subplot(layout[0, 1])
ax1.plot(bin_c, g_nufft, 'o-', ms=3, label=f"naše NUFFT (avg {N_real})")
ax1.plot(bin_c, g_gs,    's-', ms=3, label=f"GSTools (avg {N_real})")
ax1.plot(h_th,  gamma_th, 'k--', lw=2, label="teoretický")
ax1.set_title(f"Variogram – průměr přes {N_real} realizací")
ax1.set_xlabel("h [m]"); ax1.set_ylabel("γ(h)")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# gs.vario_estimate
bin_c_gs, vario_gs_builtin = gs.vario_estimate([x], field_gs, bin_edges=bin_edges)
ax2 = fig.add_subplot(layout[0, 2])
ax2.scatter(bin_c_gs, vario_gs_builtin, s=15, label="gs.vario_estimate")
ax2.plot(h_th, gamma_th, 'k--', lw=2, label="teoretický")
ax2.set_title("GSTools vario_estimate (1 realizace)")
ax2.set_xlabel("h [m]"); ax2.set_ylabel("γ(h)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# 2D NUFFT
ax3 = fig.add_subplot(layout[1, 0])
im3 = ax3.imshow(field2d_nufft, extent=[0,L,0,L], origin='lower', cmap='viridis')
plt.colorbar(im3, ax=ax3)
ax3.set_title(f"2D naše NUFFT (φ={phi})")

# 2D GSTools
ax4 = fig.add_subplot(layout[1, 1])
im4 = ax4.imshow(field2d_gs, extent=[0,L,0,L], origin='lower', cmap='viridis')
plt.colorbar(im4, ax=ax4)
ax4.set_title(f"2D GSTools (φ={phi})")

# Histogram hodnot obou 2D polí – ukazuje stejné statistické vlastnosti
ax5 = fig.add_subplot(layout[1, 2])
vals_nufft = field2d_nufft[~np.isnan(field2d_nufft)].ravel()
vals_gs    = field2d_gs.ravel()
bins_hist  = np.linspace(min(vals_nufft.min(), vals_gs.min()),
                          max(vals_nufft.max(), vals_gs.max()), 40)
ax5.hist(vals_nufft, bins=bins_hist, alpha=0.5, density=True, label="naše NUFFT")
ax5.hist(vals_gs,    bins=bins_hist, alpha=0.5, density=True, label="GSTools")
# teoretické N(0,1)
from scipy.stats import norm as sp_norm
xn = np.linspace(bins_hist[0], bins_hist[-1], 200)
ax5.plot(xn, sp_norm.pdf(xn, 0, 1), 'k--', lw=2, label="N(0,1)")
ax5.set_title("Histogram hodnot 2D polí")
ax5.set_xlabel("hodnota pole"); ax5.set_ylabel("hustota")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

plt.suptitle(f"Bod 5: Srovnání s GSTools – Gaussovská korelace φ={phi}", fontsize=13)
plt.tight_layout()
plt.show()