"""
Generování náhodných polí pomocí spektrálních metod (FFT / NUFFT).
"""
import numpy as np
from scipy.fft import fftfreq
import finufft
# =============================================================================
# CORRELATION TŘÍDY
# =============================================================================
class GaussianCorrelation:
    """
    C(r) = sigma^2 * exp(-r^2 / 2*phi^2)
    S(w) = sigma^2 * sqrt(2pi) * phi * exp(-w^2*phi^2 / 2)
    """
    def __init__(self, L, N_freq, phi, sigma=1.0):
        self.phi = phi
        self.sigma = sigma
        # frekvenční body v centrovaném pořadí: k = -N/2, ..., N/2-1
        k = np.arange(-N_freq // 2, N_freq // 2)
        self.f_points = k * (2 * np.pi / L)    # úhlové frekvence [rad/m]

    def spectral_density(self, omega):
        # S(w) = sigma^2 * sqrt(2pi) * phi * exp(-w^2*phi^2/2)
        return self.sigma**2 * np.sqrt(2 * np.pi) * self.phi * np.exp(-0.5 * (omega * self.phi)**2)

class ExponentialCorrelation:
    """
    C(r) = sigma^2 * exp(-|r|/phi)
    S(w) = sigma^2 * 2*phi / (1 + (w*phi)^2)
    """
    def __init__(self, L, N_freq, phi, sigma=1.0):
        self.phi = phi
        self.sigma = sigma
        k = np.arange(-N_freq // 2, N_freq // 2)
        self.f_points = k * (2 * np.pi / L)

    def spectral_density(self, omega):
        # S(w) = sigma^2 * 2*phi / (1 + (w*phi)^2)
        return self.sigma**2 * 2 * self.phi / (1 + (omega * self.phi)**2)

# =============================================================================
# HERMITOVSKÁ SYMETRIE – zaručí reálný výstup IFFT
# =============================================================================
def make_hermitian(f_hat, centered=False):
    """
    Vyrobí hermitovsky symetrické koeficienty: f_hat[-k] = conj(f_hat[k])

    centered=False  – FFT pořadí:        DC na indexu 0
    centered=True   – centrované pořadí: DC uprostřed na indexu N//2
    """
    N = len(f_hat)
    f = f_hat.copy()
    if centered:
        zi = N // 2                           # index DC
        f[zi] = f[zi].real
        for i in range(1, N // 2):
            f[zi - i] = np.conj(f[zi + i])   # f[-k] = conj(f[k])
    else:
        f[0] = f[0].real                      # DC
        if N % 2 == 0:
            f[N // 2] = f[N // 2].real        # Nyquistova frekvence
        for k in range(1, N // 2):
            f[-k] = np.conj(f[k])
    return f


def make_white_noise(N, seed=None):
    """Komplexní bílý šum ze standardního normálního rozdělení."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)


# =============================================================================
# FFT – PRAVIDELNÁ MŘÍŽKA
# =============================================================================

def generate_grf_fft(x_points, corr, weights=None, seed=None):
    """
    Generuje 1D náhodné pole na pravidelné mřížce pomocí FFT.

    x_points : ndarray (N,)  – pravidelná prostorová mřížka
    corr     : GaussianCorrelation | ExponentialCorrelation
    weights  : komplexní koeficienty šumu (N_freq,); pokud None, vygenerují se nové
    """
    N_freq = len(corr.f_points)
    if weights is None:
        weights = make_white_noise(N_freq, seed=seed)

    S = corr.spectral_density(corr.f_points)
    f_hat = make_hermitian(weights * np.sqrt(S), centered=True)

    # stejná code path jako NUFFT -> identické výsledky na pravidelné mřížce
    return generate_grf_nufft(x_points, corr, weights=weights)


# =============================================================================
# NUFFT – NEPRAVIDELNÁ MŘÍŽKA
# =============================================================================

def generate_grf_nufft(x_points, corr, weights=None, seed=None):
    """
    Generuje 1D náhodné pole na nepravidelné mřížce pomocí NUFFT typu 2.

    x_points : ndarray (M,)  – libovolné prostorové souřadnice v [0, L]
    corr     : GaussianCorrelation | ExponentialCorrelation
    weights  : komplexní koeficienty šumu (N_freq,); pokud None, vygenerují se nové
               -> stejné weights jako u FFT = srovnatelné realizace
    """
    N_freq = len(corr.f_points)
    if weights is None:
        weights = make_white_noise(N_freq, seed=seed)

    S = corr.spectral_density(corr.f_points)
    f_hat = make_hermitian(weights * np.sqrt(S), centered=True)

    # finufft očekává souřadnice v [-pi, pi]
    L = x_points[-1] - x_points[0]
    x_nufft = (x_points / L) * 2 * np.pi - np.pi           # [0,L] -> [-pi,pi]

    field = finufft.nufft1d2(x_nufft, f_hat.astype(np.complex128))
    return field.real


# =============================================================================
# demo.py
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L, N, phi = 100.0, 1024, 3.0
    N_freq = N  # stejný pro FFT i NUFFT -> srovnatelné realizace

    x_regular   = np.linspace(0, L, N)
    x_irregular = np.sort(np.random.uniform(0, L, 500))

    corr_g = GaussianCorrelation(L, N_freq, phi)
    corr_e = ExponentialCorrelation(L, N_freq, phi)

    white = make_white_noise(N_freq)  # stejný šum pro obě metody

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Gaussovská
    f_fft = generate_grf_fft(x_regular, corr_g, weights=white)
    axes[0][0].plot(x_regular, f_fft, lw=1)
    axes[0][0].set_title(f"FFT Gaussovská (φ={phi})")

    f_nu = generate_grf_nufft(x_irregular, corr_g, weights=white)
    axes[0][1].plot(x_irregular, f_nu, "-", alpha=0.4, color="gray", lw=0.8)
    axes[0][1].scatter(x_irregular, f_nu, s=12, c=f_nu, cmap="viridis", zorder=3)
    axes[0][1].set_title(f"NUFFT Gaussovská (φ={phi})")

    # Exponenciální
    f_fft = generate_grf_fft(x_regular, corr_e, weights=white)
    axes[1][0].plot(x_regular, f_fft, lw=1)
    axes[1][0].set_title(f"FFT Exponenciální (φ={phi})")

    f_nu = generate_grf_nufft(x_irregular, corr_e, weights=white)
    axes[1][1].plot(x_irregular, f_nu, "-", alpha=0.4, color="gray", lw=0.8)
    axes[1][1].scatter(x_irregular, f_nu, s=12, c=f_nu, cmap="viridis", zorder=3)
    axes[1][1].set_title(f"NUFFT Exponenciální (φ={phi})")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()