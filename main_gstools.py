"""
main_gstools.py – Bod 5: Porovnání naší NUFFT metody s GSTools

Srovnání přes empirický variogram – obě metody by měly konvergovat
k teoretické křivce gamma(h) = sigma^2 * (1 - exp(-h^2 / 2*phi^2))

POZOR na parametrizaci GSTools Gaussian:
  GSTools: C(r) = exp(-pi * r^2 / (4 * len_scale^2))
  Naše:    C(r) = exp(-r^2 / (2 * phi^2))
  Převod:  len_scale_gs = phi * sqrt(pi/2)  ≈  phi * 1.2533

POZOR na spektrální rozlišení:
  Frekvenční krok Δω = 2π/L – při velkém phi (phi > L/8) spadne
  do spektrální hustoty jen pár frekvenčních bodů a naše NUFFT
  metoda nebude správně odhadovat rozptyl. Řešení: zvýšit L.
"""

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs
from scipy.interpolate import griddata
from scipy.stats import norm as sp_norm
import grf


L, N, phi, seed = 100.0, 1024, 11.0, 42

# Převod phi -> len_scale pro GSTools Gaussian
# GSTools: C(r) = exp(-pi*r^2 / (4*ls^2)), naše: C(r) = exp(-r^2 / (2*phi^2))
# => ls = phi * sqrt(pi/2)
len_scale_gs = phi * np.sqrt(np.pi / 2)

# Varování při špatném poměru phi/L
delta_omega = 2 * np.pi / L
spectral_width = 1.0 / phi
n_spectral_pts = spectral_width / delta_omega
if n_spectral_pts < 3:
    print(f"VAROVÁNÍ: phi={phi}, L={L} -> do spektra spadnou jen "
          f"~{n_spectral_pts:.1f} frekvenční body. "
          f"Zvyš L (doporučeno L > {int(8*phi)+1}) nebo sniž phi.")

# =============================================================================
# 1D – jedna realizace
# =============================================================================

x     = np.linspace(0, L, N)
corr  = grf.GaussianCorrelation(L, N, phi, sigma=1.0, dim=1)
model = gs.Gaussian(dim=1, var=1.0, len_scale=len_scale_gs)

white       = grf.make_white_noise(N, dim=1, seed=seed)
field_nufft = grf.generate_grf(x, corr, weights=white)
field_gs    = np.asarray(gs.SRF(model, seed=seed)(x)).ravel()

# =============================================================================
# VARIOGRAM 1D – průměr přes N_real realizací
# =============================================================================

N_real    = 100
n_bins    = 40
# Oříznutí na L/3 – při větších lagy je málo párů a odhad je nespolehlivý
h_max     = L / 3
bin_edges = np.linspace(0, h_max, n_bins + 1)
bin_c     = 0.5 * (bin_edges[:-1] + bin_edges[1:])

g_nufft   = np.zeros(n_bins)
g_gs      = np.zeros(n_bins)
cnt_nufft = np.zeros(n_bins, dtype=int)
cnt_gs    = np.zeros(n_bins, dtype=int)

for i in range(N_real):
    w   = grf.make_white_noise(N, dim=1, seed=i)
    f_n = grf.generate_grf(x, corr, weights=w)
    h_v, g = grf.empirical_variogram(x, f_n, n_bins=n_bins)
    # empirical_variogram vrací variabilní délku – mapujeme do fixních binů
    for k, hk in enumerate(h_v):
        bi = np.searchsorted(bin_edges[1:], hk)
        if 0 <= bi < n_bins:
            g_nufft[bi]   += g[k]
            cnt_nufft[bi] += 1

    f_g = np.asarray(gs.SRF(model, seed=i)(x)).ravel()
    h_v, g = grf.empirical_variogram(x, f_g, n_bins=n_bins)
    for k, hk in enumerate(h_v):
        bi = np.searchsorted(bin_edges[1:], hk)
        if 0 <= bi < n_bins:
            g_gs[bi]  += g[k]
            cnt_gs[bi] += 1

g_nufft = np.divide(g_nufft, cnt_nufft, where=cnt_nufft > 0)
g_gs    = np.divide(g_gs,    cnt_gs,    where=cnt_gs    > 0)

# Maska – zobraz jen biny kde máme data
mask_n = cnt_nufft > 0
mask_g = cnt_gs    > 0

# Teoretický variogram: C(r) = exp(-r^2 / 2*phi^2)  =>  gamma(h) = 1 - C(h)
h_th     = np.linspace(0, h_max, 300)
gamma_th = 1 - np.exp(-0.5 * (h_th / phi)**2)

# =============================================================================
# 2D – generování a variogram přímo z nepravidelných bodů
# =============================================================================

N2      = 128
g2      = np.linspace(0, L, N2)
xx, yy  = np.meshgrid(g2, g2)
rng_np  = np.random.default_rng(seed)
pts_irr = np.column_stack([rng_np.uniform(0, L, N2**2),
                            rng_np.uniform(0, L, N2**2)])

corr2d        = grf.GaussianCorrelation(L, N2, phi, sigma=1.0, dim=2)
white2d       = grf.make_white_noise(N2, dim=2, seed=seed)

# Generujeme pole v nepravidelných bodech
f_irr2d = grf.generate_grf(pts_irr, corr2d, weights=white2d)

# ✅ Interpolace jen pro vizualizaci – variogram počítáme z původních bodů
field2d_nufft = griddata(pts_irr, f_irr2d, (xx, yy), method='linear')

model2d    = gs.Gaussian(dim=2, var=1.0, len_scale=len_scale_gs)
field2d_gs = gs.SRF(model2d, seed=seed).structured([g2, g2])

# ✅ Variogram 2D – počítáme z nepravidelných bodů, ne z interpolované mřížky
h_max_2d  = L / 3
n_bins_2d = 30

# NUFFT: přímo z pts_irr a f_irr2d
h_2d_nufft, g_2d_nufft = grf.empirical_variogram(pts_irr, f_irr2d,
                                                   n_bins=n_bins_2d)

# GSTools: z pravidelné mřížky (jako (M,2) pole bodů)
pts_reg  = np.column_stack([xx.ravel(), yy.ravel()])
f_gs_reg = field2d_gs.ravel()
h_2d_gs, g_2d_gs = grf.empirical_variogram(pts_reg, f_gs_reg,
                                             n_bins=n_bins_2d)

# Oříznutí na L/3
mask_2d_n = h_2d_nufft <= h_max_2d
mask_2d_g = h_2d_gs    <= h_max_2d

h_th_2d     = np.linspace(0, h_max_2d, 300)
gamma_th_2d = 1 - np.exp(-0.5 * (h_th_2d / phi)**2)

# =============================================================================
# GRAFY
# =============================================================================

fig = plt.figure(figsize=(16, 9))
layout = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

# --- 1D realizace ---
ax0 = fig.add_subplot(layout[0, 0])
ax0.plot(x, field_gs,    lw=1.2, label="GSTools")
ax0.plot(x, field_nufft, lw=1,   label="naše NUFFT", alpha=0.8)
ax0.set_title(f"1D realizace (φ={phi})")
ax0.set_xlabel("x [m]")
ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

# --- Variogram 1D průměr ---
ax1 = fig.add_subplot(layout[0, 1])
ax1.plot(bin_c[mask_n], g_nufft[mask_n], 'o-', ms=3,
         label=f"naše NUFFT (avg {N_real})")
ax1.plot(bin_c[mask_g], g_gs[mask_g],    's-', ms=3,
         label=f"GSTools (avg {N_real})")
ax1.plot(h_th, gamma_th, 'k--', lw=2, label="teoretický")
ax1.set_title(f"Variogram 1D – průměr přes {N_real} realizací\n"
              f"(zobrazeno do h = L/3 = {h_max:.0f} m)")
ax1.set_xlabel("h [m]"); ax1.set_ylabel("γ(h)")
ax1.set_xlim(0, h_max)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# --- GSTools vario_estimate (1 realizace) ---
bin_edges_gs = np.linspace(0, h_max, n_bins + 1)
bin_c_gs, vario_gs_builtin = gs.vario_estimate([x], field_gs,
                                                bin_edges=bin_edges_gs)
ax2 = fig.add_subplot(layout[0, 2])
ax2.scatter(bin_c_gs, vario_gs_builtin, s=15, label="gs.vario_estimate")
ax2.plot(h_th, gamma_th, 'k--', lw=2, label="teoretický")
ax2.set_title(f"GSTools vario_estimate (1 realizace)\n"
              f"(zobrazeno do h = L/3 = {h_max:.0f} m)")
ax2.set_xlabel("h [m]"); ax2.set_ylabel("γ(h)")
ax2.set_xlim(0, h_max)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# --- 2D NUFFT (interpolovaná vizualizace) ---
ax3 = fig.add_subplot(layout[1, 0])
im3 = ax3.imshow(field2d_nufft, extent=[0,L,0,L], origin='lower', cmap='viridis')
plt.colorbar(im3, ax=ax3)
ax3.set_title(f"2D naše NUFFT (φ={phi})")

# --- 2D GSTools ---
ax4 = fig.add_subplot(layout[1, 1])
im4 = ax4.imshow(field2d_gs, extent=[0,L,0,L], origin='lower', cmap='viridis')
plt.colorbar(im4, ax=ax4)
ax4.set_title(f"2D GSTools (φ={phi})")

# --- Variogram 2D – z nepravidelných bodů ---
ax5 = fig.add_subplot(layout[1, 2])
ax5.plot(h_2d_nufft[mask_2d_n], g_2d_nufft[mask_2d_n], 'o-', ms=3,
         label="naše NUFFT (z irr. bodů)")
ax5.plot(h_2d_gs[mask_2d_g],    g_2d_gs[mask_2d_g],    's-', ms=3,
         label="GSTools (z reg. mřížky)")
ax5.plot(h_th_2d, gamma_th_2d, 'k--', lw=2, label="teoretický")
ax5.set_title(f"Variogram 2D (do L/3 = {h_max_2d:.0f} m)\n"
              f"✅ počítáno z původních bodů, ne z interpolace")
ax5.set_xlabel("h [m]"); ax5.set_ylabel("γ(h)")
ax5.set_xlim(0, h_max_2d)
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

plt.suptitle(f"Bod 5: Srovnání s GSTools – Gaussovská korelace φ={phi}", fontsize=13)
plt.tight_layout()
plt.show()