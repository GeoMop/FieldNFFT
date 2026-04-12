"""
gstools_nufft3.py – Rekonstrukce GSTools pole pomocí NUFFT typ 3

GSTools (RandMeth) počítá:
    f(x) = sqrt(1/N) * sum_k [ z1_k*cos(w_k*x) + z2_k*sin(w_k*x) ]
         = Re[ sum_k  c_k * exp(i*w_k*x) ]   kde c_k = z1_k - i*z2_k

finufft.nufft1d3: neuniformní frekvence s_k -> hodnoty na libovolných bodech x_j
"""

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs
import finufft

L, phi, seed = 100.0, 5.0, 42
N = 512
x = np.linspace(0, L, N)

# --- 1. Vyhodnotíme GSTools normálně ---
model = gs.Gaussian(dim=1, var=1.0, len_scale=phi)
srf   = gs.SRF(model, seed=seed)
field_gs = np.asarray(srf(x)).ravel()

# --- 2. Vytáhneme interní stav generátoru ---
gen = srf.generator
print("Atributy generátoru:", [a for a in dir(gen) if not a.startswith('__')])

z1    = gen._z_1          # shape (N_modes,)
z2    = gen._z_2          # shape (N_modes,)
modes = gen._cov_sample   # shape (1, N_modes) pro 1D

omega = modes[0]          # frekvence w_k (1D)

print(f"Počet módů: {len(omega)}")
print(f"Rozsah frekvencí: [{omega.min():.4f}, {omega.max():.4f}]")

# --- 3. Rekonstrukce přes NUFFT typ 3 ---
# f(x_j) = Re[ sum_k c_k * exp(i * w_k * x_j) ]
# c_k = (z1_k - i*z2_k) / sqrt(N_modes)
N_modes = len(omega)
c = (z1 - 1j * z2) / np.sqrt(N_modes)

# nufft1d3: vstupy jsou neuniformní frekvence a neuniformní body
# pozor: finufft očekává x v libovolných reálných hodnotách (ne nutně [-pi,pi])
field_nufft3 = finufft.nufft1d3(
    omega.astype(np.float64),   # neuniformní frekvence s_k
    c.astype(np.complex128),    # komplexní váhy c_k
    x.astype(np.float64),       # body kde chceme hodnoty
    eps=1e-9
).real

# --- 4. Porovnání ---
max_diff = np.max(np.abs(field_gs - field_nufft3))
print(f"\nMax |GSTools - NUFFT3|: {max_diff:.2e}")

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].plot(x, field_gs,     lw=1.5, label="GSTools přímý výpočet")
axes[0].plot(x, field_nufft3, lw=1,   label="NUFFT typ 3 rekonstrukce", alpha=0.8, ls='--')
axes[0].set_title(f"1D pole – GSTools vs NUFFT3 rekonstrukce (φ={phi})")
axes[0].set_xlabel("x [m]")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(x, field_gs - field_nufft3, lw=1, color='red')
axes[1].set_title(f"Rozdíl: max = {max_diff:.2e}")
axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("GSTools − NUFFT3")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()