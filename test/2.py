import numpy as np
import matplotlib.pyplot as plt

def generate_1d_grf(N, L, correlation_length):
    # 1. Mřížka a frekvence
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    freqs = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Úhlová frekvence omega
    
    # 2. Spektrální hustota S(omega) pro Gaussovu korelaci
    # S(w) = sqrt(2*pi)*l * exp(-w^2 * l^2 / 2)
    l = correlation_length
    psd = np.sqrt(2 * np.pi) * l * np.exp(-(freqs**2 * l**2) / 2)
    
    # 3. Generování náhodných komplexních koeficientů
    # Standardní normální rozdělení (průměr 0, rozptyl 1)
    white_noise = (np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N)) / np.sqrt(2)
    
    # 4. Modulace šumu spektrální hustotou
    # Odmocnina z PSD, protože amplituda = sqrt(výkonu)
    coeffs_f = white_noise * np.sqrt(psd)
    print("white_noise", white_noise)
    print("x", x)
    print("Frekvence", freqs)
    print("PSD", psd)
    print("Náhodné koeficienty před úpravou", coeffs_f)

    # 5. TECHNICKÁ ČÁST: Správné komplexní sdružení pro REÁLNÝ výstup
    # Aby ifft vrátilo reálná čísla, musí platit f(k) = conj(f(-k))
    # Numpy to má uspořádané: [0, pos_freqs, nyquist, neg_freqs]
    
    coeffs_f[0] = coeffs_f[0].real * np.sqrt(2) # DC složka musí být reálná
    half = N // 2
    for i in range(1, half):
        coeffs_f[N - i] = np.conj(coeffs_f[i])
    
    if N % 2 == 0:
        coeffs_f[half] = coeffs_f[half].real * np.sqrt(2) # Nyquistova frekv. reálná

    # 6. Inverzní FFT
    # Použijeme normalizaci 'ortho' nebo musíme násobit sqrt(N) podle definice
    field = np.fft.ifft(coeffs_f * np.sqrt(N / dx)).real
    
    return x, field

# Spuštění
N = 1000
L = 100
phi = 6.0 # korelační délka
x, y = generate_1d_grf(N, L, phi)

plt.figure(figsize=(10, 4))
plt.plot(x, y)
plt.title(f"Náhodné pole (Gaussovská korelace, $\ell={phi}$)")
plt.xlabel("x")
plt.ylabel("Hodnota")
plt.grid(True)
plt.show()