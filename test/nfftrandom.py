import numpy as np
import finufft
import matplotlib.pyplot as plt

def generate_grf_nufft(N_freq, M_points, phi, L):
    """
    N_freq: počet frekvenčních módů (pravidelná mřížka ve frekvenci)
    M_points: počet nepravidelných bodů, které chceme vygenerovat
    phi: korelační délka (vliv na hladkost)
    L: celková délka domény
    """
    
    # --- 1. PŘÍPRAVA FREKVENCÍ ---
    # Indexy frekvencí k (centrované kolem nuly pro NUFFT)
    k = np.arange(-N_freq // 2, N_freq // 2)
    # Převod indexů na úhlovou frekvenci omega
    # (předpokládáme doménu 2*pi pro jednoduchost NUFFT)
    omega = k * (2 * np.pi / L)
    
    # --- 2. SPEKTRÁLNÍ HUSTOTA S(omega) ---
    # Gaussovská korelační funkce -> Gaussovské spektrum
    S_omega = np.sqrt(2 * np.pi) * phi * np.exp(-(omega**2 * phi**2) / 2)
    
    # --- 3. GENEROVÁNÍ NÁHODNÝCH KOEFICIENTŮ ---
    # Standardní normální rozdělení pro Re a Im část
    white_noise = (np.random.normal(0, 1, N_freq) + 1j * np.random.normal(0, 1, N_freq)) / np.sqrt(2)
    
    # Modulace šumu filtrem (odmocnina ze spektrální hustoty)
    f_hat = white_noise * np.sqrt(S_omega)

    # --- 4. TECHNICKÉ ZAJIŠTĚNÍ REÁLNOSTI (Bod 1) ---
    # Pro NUFFT s frekvencemi centrovanými kolem 0:
    # f_hat(-k) musí být conj(f_hat(k))
    # Najdeme index nuly
    zero_idx = N_freq // 2
    f_hat[zero_idx] = f_hat[zero_idx].real # DC složka reálná
    
    for i in range(1, N_freq // 2):
        f_hat[zero_idx - i] = np.conj(f_hat[zero_idx + i])

    # --- 5. NEPRAVIDELNÁ MŘÍŽKA (Bod 3) ---
    # Vygenerujeme náhodné pozice v rozsahu [0, L]
    # finufft nufft1d2 standardně očekává souřadnice v [-pi, pi]
    x_irregular = np.random.uniform(-np.pi, np.pi, M_points)
    
    # Výpočet NUFFT typu 2 (z pravidelných frekvencí do nepravidelných bodů)
    # nufft1d2(souřadnice, koeficienty)
    field = finufft.nufft1d2(x_irregular, f_hat.astype(np.complex128))
    
    # Vrátíme reálnou část (imaginární je díky symetrii prakticky nulová)
    # Musíme přeškálovat x zpět na naši délku L, pokud chceme
    x_final = (x_irregular + np.pi) * (L / (2 * np.pi))
    
    return x_final, field.real

# --- TESTOVACÍ SPUŠTĚNÍ ---
N = 256    # Počet frekvencí (přesnost spektra)
M = 500   # Počet bodů v prostoru (nepravidelných)
L = 100.0  # Délka území
phi = 3  # Zkus měnit (0.5 pro zubaté, 5.0 pro hladké)

x_coords, y_values = generate_grf_nufft(N, M, phi, L)

# Seřazení pro hezký graf (protože body jsou náhodně rozházené)
sort_idx = np.argsort(x_coords)
x_plot = x_coords[sort_idx]
y_plot = y_values[sort_idx]

plt.figure(figsize=(12, 5))
plt.plot(x_plot, y_plot, '-', alpha=0.5, color='gray')
plt.scatter(x_coords, y_values, s=10, c=y_values, cmap='viridis')
plt.title(f"1D Náhodné pole (Gaussovská korelace $\\phi={phi}$)\nNepravidelná mřížka (NUFFT)")
plt.xlabel("Pozice [m]")
plt.ylabel("Hodnota (např. výška)")
plt.colorbar(label="Amplituda")
plt.grid(True)
plt.show()