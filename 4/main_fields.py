"""
main_fields.py – Bod 3 + 4

Bod 3: GRF 1D – pravidelná a nepravidelná mřížka, body vyznačeny na ose
Bod 4: GRF 2D – pravidelná a nepravidelná mřížka
"""

import numpy as np
import matplotlib.pyplot as plt
import finufft
from scipy.interpolate import griddata

import grf

L, N, phi, seed = 100.0, 512, 5.0, 42
N_sparse = 50     # počet bodů řídké NUFFT mřížky
N_dense  = 500    # počet bodů husté NUFFT mřížky
rng = np.random.default_rng(seed)


# =============================================================================
# BOD 3: GRF 1D
# =============================================================================

x_reg = np.linspace(0, L, N)
x_irr = np.sort(rng.uniform(0, L, N // 2))
white = grf.make_white_noise(N, dim=1, seed=seed)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for row, (label, CorrClass) in enumerate([("Gaussovská", grf.GaussianCorrelation),
                                           ("Exponenciální", grf.ExponentialCorrelation)]):
    corr = CorrClass(L, N, phi, dim=1)

    f_r = grf.generate_grf(x_reg, corr, weights=white)
    ax = axes[row][0]
    ax.plot(x_reg, f_r, lw=1)
    ax.plot(x_reg, np.full(N, f_r.min() - 0.3),
            '|', color='steelblue', ms=6, label="body mřížky")
    ax.set_title(f"FFT pravidelná – {label} φ={phi}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # reference: hustá pravidelná mřížka
    x_ref   = np.linspace(0, L, N)
    f_ref   = grf.generate_grf(x_ref, corr, weights=white)

    # řídká nepravidelná
    x_sparse = np.sort(rng.uniform(0, L, N_sparse))
    f_sparse = grf.generate_grf(x_sparse, corr, weights=white)
    # interpolace řídkých bodů na jemnou mřížku
    from scipy.interpolate import interp1d
    f_sparse_interp = interp1d(x_sparse, f_sparse, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')(x_ref)

    ax = axes[row][1]
    ax.plot(x_ref, f_sparse_interp, '-', color='darkorange', lw=1.5, alpha=0.4,
            label=f"interp z {N_sparse} bodů")
    ax.plot(x_ref, f_ref, 'k-', lw=1, label="reference")
    ax.scatter(x_sparse, f_sparse, s=25, color='darkorange', zorder=3)
    ax.plot(x_sparse, np.full(len(x_sparse), f_ref.min() - 0.3),
            '|', color='darkorange', ms=8)
    ax.set_title(f"NUFFT řídká ({N_sparse} bodů) – {label}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # hustá nepravidelná
    x_dense = np.sort(rng.uniform(0, L, N_dense))
    f_dense = grf.generate_grf(x_dense, corr, weights=white)
    f_dense_interp = interp1d(x_dense, f_dense, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')(x_ref)

    ax = axes[row][2]
    ax.plot(x_ref, f_dense_interp, '-', color='steelblue', lw=1.5, alpha=0.4,
            label=f"interp z {N_dense} bodů")
    ax.plot(x_ref, f_ref, 'k-', lw=1, label="reference")
    ax.scatter(x_dense, f_dense, s=4, color='steelblue', zorder=3)
    ax.plot(x_dense, np.full(len(x_dense), f_ref.min() - 0.3),
            '|', color='steelblue', ms=4)
    ax.set_title(f"NUFFT hustá ({N_dense} bodů) – {label}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle("Bod 3: GRF 1D")
plt.tight_layout(); plt.show()


# =============================================================================
# BOD 4: GRF 2D
# =============================================================================

N2      = 64
N2_sparse = 20    # řídká mřížka – počet bodů na osu (N2_sparse^2 celkem)
N2_dense  = 64    # hustá mřížka – počet bodů na osu (N2_dense^2 celkem)

white2d = grf.make_white_noise(N2, dim=2, seed=seed)
g = np.linspace(0, L, N2)
xx, yy = np.meshgrid(g, g)

n_sparse = N2_sparse**2
n_dense  = N2_dense**2
pts_sparse = np.column_stack([rng.uniform(0, L, n_sparse), rng.uniform(0, L, n_sparse)])
pts_dense  = np.column_stack([rng.uniform(0, L, n_dense),  rng.uniform(0, L, n_dense)])

for label, CorrClass in [("Gaussovská", grf.GaussianCorrelation),
                          ("Exponenciální", grf.ExponentialCorrelation)]:
    corr2d = CorrClass(L, N2, phi, dim=2)

    # 1. Pravidelná referenční mřížka (stejná)
    pts_reg = np.column_stack([xx.ravel(), yy.ravel()])
    field_ref = grf.generate_grf(pts_reg, corr2d, weights=white2d).reshape(N2, N2)

    # 2. Generování z řídkých a hustých bodů (stejné)
    f_sparse = grf.generate_grf(pts_sparse, corr2d, weights=white2d)
    field_sparse_interp = griddata(pts_sparse, f_sparse, (xx, yy), method='linear')

    f_dense = grf.generate_grf(pts_dense, corr2d, weights=white2d)
    field_dense_interp = griddata(pts_dense, f_dense, (xx, yy), method='linear')


    # --- VIZUALIZACE (4 subploty, tečky všude, kam patří) ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5)) # Širší plátno
    vmin, vmax = field_ref.min(), field_ref.max()
    
    # Nastavení pro scatter-bar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # 1. Referenční pole
    im0 = axes[0].imshow(field_ref, extent=[0,L,0,L], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Referenční pravidelná ({N2}×{N2})")
    plt.colorbar(im0, ax=axes[0])

    # 2. NOVÝ: Samotná nepravidelná mřížka (Barevné tečky)
    sc = axes[1].scatter(pts_sparse[:, 0], pts_sparse[:, 1], c=f_sparse, cmap='viridis', s=18, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Nepravidelná mřížka\n({n_sparse} vzorků)")
    axes[1].set_xlim(0, L); axes[1].set_ylim(0, L)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.2) # Jemná mřížka pro orientaci
    plt.colorbar(sc, ax=axes[1])

    # 3. PŮVODNÍ (s tečkami): Interpolace z řídké mřížky
    im2 = axes[2].imshow(field_sparse_interp, extent=[0,L,0,L], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    # Tady jsou ty bílé tečky navrch
    axes[2].plot(pts_sparse[:, 0], pts_sparse[:, 1], '.', color='white', ms=2.5, alpha=0.5, label='vzorky')
    axes[2].set_title(f"NUFFT řídká ({n_sparse} bodů) + interp")
    plt.colorbar(im2, ax=axes[2])

    # 4. PŮVODNÍ (s tečkami): Interpolace z husté mřížky
    im3 = axes[3].imshow(field_dense_interp, extent=[0,L,0,L], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    # Tady jsou ty bílé tečky navrch (menší ms= a alpha=)
    axes[3].plot(pts_dense[:, 0], pts_dense[:, 1], '.', color='white', ms=1.8, alpha=0.4)
    axes[3].set_title(f"NUFFT hustá ({n_dense} bodů) + interp")
    plt.colorbar(im3, ax=axes[3])

    plt.suptitle(f"Bod 4: GRF 2D – {label} φ={phi}")
    plt.tight_layout()
    plt.show()