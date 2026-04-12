import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
p = False

# ==========================================
# 1. Definice mřížky (Grid) a signálu
# ==========================================
N = 100 
x, dx = np.linspace(0, 2*np.pi, N, endpoint=False, retstep=True)

# Rozložení signálu na jednotlivé komponenty (abychom viděli, co se z čeho skládá)
f1 = 1 * np.ones_like(x)         # DC složka (konstanta 1)
f2 = 2 * np.sin(x)               # Frekvence 1
f3 = 10 * np.sin(5*x)            # Frekvence 5 (sinusovka)
f4 = 3 * np.cos(5*x)             # Frekvence 5 (kosinusovka)

# Výsledná složená funkce ze zadání
f_x = f1 + f2 + f3 + f4

if p:
    print("dx:", dx)
    print("x:", x)
    print("f(x):", f_x)

# ==========================================
# 2. Výpočet FFT a zpětné transformace
# ==========================================
F_k = fft(f_x)
f_x_reconstructed = ifft(F_k)

# Výpočet frekvencí pro osu X
xf_vsechny = fftfreq(N, dx)

# Pro srozumitelné spektrum nás většinou zajímá jen kladná část spektra
xf_kladne = xf_vsechny[:N//2]
F_k_kladne = F_k[:N//2]

# Amplitudy (síla signálu) - normalizace pomocí (2/N)
amplitudy = (2.0/N) * np.abs(F_k_kladne)
amplitudy[0] = amplitudy[0] / 2  # DC složka se nedělí dvěma, vracíme ji na správnou hodnotu

# Fáze (posun signálu v radiánech)
faze = np.angle(F_k_kladne)
# Odfiltrování šumu z fáze: tam, kde je nulová nebo minimální amplituda, nemá smysl počítat fázi
faze[amplitudy < 0.1] = 0

# Výkonové spektrum (Power Spectrum - energie na dané frekvenci)
vykon = np.abs(F_k_kladne)**2


# ==========================================
# 3. VYKRESLENÍ - OBRÁZEK 1: Časová oblast
# ==========================================
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(x, f_x, label="Výsledný signál f(x)", color='black', linewidth=2)
plt.title("1. Celkový složený signál f(x) v čase")
plt.ylabel("Amplituda")
plt.grid(True); plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, f1, label="DC (1)", linestyle='--')
plt.plot(x, f2, label="2*sin(x)", linestyle='--')
plt.plot(x, f3, label="10*sin(5x)", linestyle='--')
plt.plot(x, f4, label="3*cos(5x)", linestyle='--')
plt.title("2. Jednotlivé " + "skryté" + " složky, ze kterých se signál skládá")
plt.ylabel("Amplituda")
plt.grid(True); plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, f_x, label="Původní f(x)", linewidth=4, alpha=0.5)
plt.plot(x, f_x_reconstructed.real, '--', label="Složená (iFFT)", color='red')
plt.title("3. Původní signál vs. Rekonstruovaný signál (Úspěšný test!)")
plt.xlabel("x (Čas / Prostor)")
plt.ylabel("Amplituda")
plt.grid(True); plt.legend()
plt.tight_layout()


# ==========================================
# VYKRESLENÍ - OBRÁZEK 2: Surová komplexní čísla z FFT
# ==========================================
plt.figure(figsize=(12, 8))
# Reálná část reprezentuje podíly KOSINUSOVÝCH funkcí
plt.subplot(2, 1, 1)
plt.stem(range(N), F_k.real, basefmt=" ")
plt.title("4. Surový výstup FFT: Reálná část (Odpovídá zastoupení KOSINUSŮ ve frekvencích)")
plt.ylabel("Reálná hodnota")
plt.grid(True)

# Imaginární část reprezentuje podíly SINUSOVÝCH funkcí
plt.subplot(2, 1, 2)
plt.stem(range(N), F_k.imag, basefmt=" ")
plt.title("5. Surový výstup FFT: Imaginární část (Odpovídá zastoupení SINUSŮ ve frekvencích)")
plt.xlabel("Index pole (frekvenční koš)")
plt.ylabel("Imaginární hodnota")
plt.grid(True)
plt.tight_layout()


# ==========================================
# VYKRESLENÍ - OBRÁZEK 3: Polární forma (Amplituda a Fáze)
# ==========================================
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.stem(xf_kladne, amplitudy, basefmt=" ")
plt.xlim(-0.1, 1.2) # Ořízneme graf jen na zajímavou část, ať vidíme špičky
plt.title("6. Amplitudové spektrum - To nejdůležitější (Síla jednotlivých frekvencí)")
plt.ylabel("Amplituda")
plt.grid(True)

# Přidání textu přímo nad špičky v grafu pro přehlednost
for i, amp in enumerate(amplitudy):
    if amp > 0.5:
        plt.text(xf_kladne[i], amp + 0.5, f"{amp:.1f}", ha='center', fontweight='bold')

plt.subplot(2, 1, 2)
plt.stem(xf_kladne, faze, basefmt=" ")
plt.xlim(-0.1, 1.2)
plt.title("7. Fázové spektrum - Fázový posun odhalených vln (v radiánech)")
plt.xlabel("Frekvence")
plt.ylabel("Fáze [rad]")
plt.grid(True)
plt.tight_layout()


# ==========================================
# VYKRESLENÍ - OBRÁZEK 4: Výkonové spektrum (Power Spectrum)
# ==========================================
plt.figure(figsize=(12, 4))
plt.stem(xf_kladne, vykon, basefmt=" ", markerfmt='ro', linefmt='r-')
plt.xlim(-0.1, 1.2)
plt.title("8. Výkonové spektrum (Power Spectral Density) - Kde je ukryta většina energie signálu")
plt.xlabel("Frekvence")
plt.ylabel("Výkon (Absolutní hodnota na druhou)")
plt.grid(True)
plt.tight_layout()

plt.show()