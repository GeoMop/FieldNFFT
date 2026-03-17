import numpy as np
N = 1000          # Počet bodů mřížky
L = 100.0         # Fyzická délka (např. 100 metrů)
phi = 2.0         # Korelační délka (v tvém zadání)

# Pravidelná mřížka v prostoru
x = np.linspace(0, L, N, endpoint=False)
dx = L / N

# Úhlové frekvence omega
# fftfreq vrací frekvence v cyklech, proto musíme násobit 2*pi
freqs = np.fft.fftfreq(N, d=dx) * 2 * np.pi

# S(omega) - Gaussovský recept
S_omega = np.sqrt(2 * np.pi) * phi * np.exp(-(freqs**2 * phi**2) / 2)
amplitude_filter = np.sqrt(S_omega * N / dx)

# Náhodná komplexní čísla (Standardní Normální rozdělení)
# Generujeme real a imag část zvlášť
white_noise = (np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N)) / np.sqrt(2)

# Aplikujeme náš Gaussovský recept na náhodný šum
F_k = white_noise * amplitude_filter
# Tady použiješ tu svoji funkci, co zajistí zrcadlení (Hermitovskou symetrii)
F_k_final = fn.force_hermitian_symmetry(F_k) # To je ta naše funkce z minula
# Inverzní FFT nám vytvoří to náhodné pole v 1D
random_field = np.fft.ifft(F_k_final).real