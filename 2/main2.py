import matplotlib.pyplot as plt
import numpy as np
import fft1d as rf

L, N, phi = 100.0, 1024, 3.0

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# --- Gaussovská ---
x, f = rf.generate_grf_fft(L, N, phi, seed=42)
axes[0][0].plot(x, f, lw=1)
axes[0][0].set_title(f"FFT Gaussovská (φ={phi})")

x_nu, f_nu = rf.generate_grf_nufft(L, 256, 500, phi, seed=42)
idx = np.argsort(x_nu)
axes[0][1].plot(x_nu[idx], f_nu[idx], '-', alpha=0.4, color='gray', lw=0.8)
axes[0][1].scatter(x_nu, f_nu, s=12, c=f_nu, cmap='viridis', zorder=3)
axes[0][1].set_title(f"NUFFT Gaussovská (φ={phi})")

# --- Exponenciální ---
x, f = rf.generate_grf_fft(L, N, phi, seed=42, spectral_density_fn=rf.spectral_density_exponential)
axes[1][0].plot(x, f, lw=1)
axes[1][0].set_title(f"FFT Exponenciální (φ={phi})")

x_nu, f_nu = rf.generate_grf_nufft(L, 256, 500, phi, seed=42, spectral_density_fn=rf.spectral_density_exponential)
idx = np.argsort(x_nu)
axes[1][1].plot(x_nu[idx], f_nu[idx], '-', alpha=0.4, color='gray', lw=0.8)
axes[1][1].scatter(x_nu, f_nu, s=12, c=f_nu, cmap='viridis', zorder=3)
axes[1][1].set_title(f"NUFFT Exponenciální (φ={phi})")

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()