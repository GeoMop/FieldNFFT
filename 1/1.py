import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
p = True
N = 100 
x, dx = np.linspace(0, 2*np.pi, N, endpoint=False, retstep=True)
f_x = 1 + 2*np.sin(x) + 10*np.sin(5*x) + 3*np.cos(5*x)
F_k = fft(f_x)
if p:
    print("dx:", dx)
    print("x:", x)
    print("f(x):", f_x)
    print("F_k:", F_k)

f_x_reconstructed = ifft(F_k)

# --- VYKRESLENÍ VÝSLEDKŮ PRO KONTROLU ---
plt.figure(figsize=(12, 8))

# Graf 1: Původní a složená funkce (měly by se překrývat)
plt.subplot(2, 1, 1)
plt.plot(x, f_x, label="Původní f(x)", linewidth=3)
# ifft vrací komplexní čísla, pro graf vezmeme reálnou část (.real)
plt.plot(x, f_x_reconstructed.real, '--', label="Složená (iFFT)", color='red')
plt.title("Původní signál vs. Rekonstruovaný signál (Úspěšný test!)")
plt.legend()
plt.grid()

# Graf 2: Zobrazení frekvenčního spektra (co vlastně FFT našla)
plt.subplot(2, 1, 2)
# Spočítáme frekvence pro osu X (zajímají nás jen kladné frekvence, proto úprava do N//2)
xf = fftfreq(N, dx)[:N//2]
print("xf:", xf)
# Amplituda (síla) signálu - musíme normalizovat dělením (2/N)
amplitudy = (2.0/N) * np.abs(F_k[:N//2])
print("Amplitudy:", amplitudy)
# Ruční úprava nulté frekvence (DC složky - to je ta "1" na začátku rovnice)
amplitudy[0] = amplitudy[0] / 2 

# Vykreslíme to jako sloupečky (stem plot)
plt.stem(xf, amplitudy)
plt.xlim(-0.1, 2) # Přiblížíme začátek grafu, aby byly vidět špičky
plt.title("Spektrum (FFT) - Všimni si špiček na frekvencích, které odpovídají rovnici")
plt.xlabel("Frekvence (Hz)")
plt.ylabel("Amplituda")
plt.grid()

plt.tight_layout()
plt.show()