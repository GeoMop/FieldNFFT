"""
Upravit zadání

"""

import random
import numpy as np
import finufft
# =============================================================================
# CORRELATION TŘÍDY
# =============================================================================

class GaussianCorrelation:
    """
    C(r) = sigma^2 * exp(-|r|^2 / 2*phi^2)
    S_nD(omega) = sigma^2 * (sqrt(2pi)*phi)^dim * exp(-|omega|^2*phi^2/2)
    """
    def __init__(self, L, N_freq, phi, sigma=1.0, dim=1):
        self.phi = phi
        self.sigma = sigma
        self.dim = dim
        self.N_freq = N_freq
        self.L = L
        k = np.arange(-N_freq // 2, N_freq // 2)
        self.f_points = k * (2 * np.pi / L)    # 1D osa frekvencí [rad/m]

    def spectral_density(self, omega_mag):
        # omega_mag = |omega| (skalár nebo pole)
        return (self.sigma**2
                * (np.sqrt(2 * np.pi) * self.phi)**self.dim
                * np.exp(-0.5 * (omega_mag * self.phi)**2))


class ExponentialCorrelation:
    """
    C(r) = sigma^2 * exp(-|r|/phi)
    1D: S(w) = sigma^2 * 2*phi / (1 + (w*phi)^2)
    2D: S(w) = sigma^2 * 2*pi*phi^2 / (1 + (|w|*phi)^2)^(3/2)
    """
    def __init__(self, L, N_freq, phi, sigma=1.0, dim=1):
        self.phi = phi
        self.sigma = sigma
        self.dim = dim
        self.N_freq = N_freq
        self.L = L
        k = np.arange(-N_freq // 2, N_freq // 2)
        self.f_points = k * (2 * np.pi / L)

    def spectral_density(self, omega_mag):
        if self.dim == 1:
            return self.sigma**2 * 2 * self.phi / (1 + (omega_mag * self.phi)**2)
        elif self.dim == 2:
            return self.sigma**2 * 2 * np.pi * self.phi**2 / (1 + (omega_mag * self.phi)**2)**1.5
        else:
            raise NotImplementedError("Exponential: dim > 2 není implementováno")


# =============================================================================
# HERMITOVSKÁ SYMETRIE – zaručí reálný výstup IFFT
# centrované pořadí: DC uprostřed na indexu N//2 (v každé dimenzi)
# podmínka: f_hat[-k] = conj(f_hat[k])  pro všechny k
# =============================================================================

def make_hermitian_nd(f_hat):
    """
    nD hermitovská symetrie v centrovaném pořadí.
    f_hat shape: (N,) pro 1D, (N, N) pro 2D, (N, N, N) pro 3D.
    """
    N = f_hat.shape[0]
    zi = N // 2   # index DC v každé dimenzi
    f = f_hat.copy()

    # DC (všechny indexy == zi) musí být reálná
    dc_idx = tuple([zi] * f.ndim)
    f[dc_idx] = f[dc_idx].real

    # pro každý multi-index k: f[-k] = conj(f[k])
    # iterujeme přes všechny indexy, záporné nastavíme jako sdružené kladných
    for idx in np.ndindex(*f.shape):
        # přeskočit DC a body s nulovým indexem (ty jsou sdružené sami sobě)
        shifted = tuple(i - zi for i in idx)
        if all(s <= 0 for s in shifted):
            continue
        neg_idx = tuple((zi - s) % N for s in shifted)
        f[neg_idx] = np.conj(f[idx])

    return f


def make_white_noise(N_freq, dim=1, seed=None):
    """Komplexní bílý šum tvaru (N_freq,)*dim."""
    rng = np.random.default_rng(seed)
    shape = (N_freq,) * dim
    size = N_freq ** dim
    flat = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)
    return flat.reshape(shape) 


# =============================================================================
# GENERATE GRF
# =============================================================================

def generate_grf_nufft(x_points, corr, weights=None, seed=None):
    """
    Generuje náhodné pole v dim dimenzích pomocí NUFFT typu 2.

    x_points : (M,)    pro 1D
               (M, 2)  pro 2D
               (M, 3)  pro 3D
    corr     : GaussianCorrelation | ExponentialCorrelation  (s dim nastaveným)
    weights  : bílý šum tvaru (N_freq,)*dim; pokud None, vygeneruje se nový
    """
    dim = corr.dim
    N_freq = corr.N_freq

    if weights is None:
        weights = make_white_noise(N_freq, dim=dim, seed=seed)

    # frekvenční mřížka v nD: |omega| pro každý bod
    if dim == 1:
        omega_mag = np.abs(corr.f_points)
    else:
        grids = np.meshgrid(*([corr.f_points] * dim), indexing='ij')
        omega_mag = np.sqrt(sum(g**2 for g in grids))  # shape (N_freq,)*dim

    S = corr.spectral_density(omega_mag)
    f_hat = make_hermitian_nd(weights * np.sqrt(S))

    # souřadnice do [-pi, pi]
    x_points = np.asarray(x_points)
    L = corr.L

    if dim == 1:
        x_nu = (x_points / L) * 2 * np.pi - np.pi
        field = finufft.nufft1d2(x_nu, f_hat.astype(np.complex128))

    elif dim == 2:
        x_nu = (x_points[:, 0] / L) * 2 * np.pi - np.pi
        y_nu = (x_points[:, 1] / L) * 2 * np.pi - np.pi
        field = finufft.nufft2d2(x_nu, y_nu, f_hat.astype(np.complex128))

    elif dim == 3:
        x_nu = (x_points[:, 0] / L) * 2 * np.pi - np.pi
        y_nu = (x_points[:, 1] / L) * 2 * np.pi - np.pi
        z_nu = (x_points[:, 2] / L) * 2 * np.pi - np.pi
        field = finufft.nufft3d2(x_nu, y_nu, z_nu, f_hat.astype(np.complex128))

    else:
        raise NotImplementedError("dim > 3 není podporováno finufft")

    return field.real


def generate_grf_fft(x_points, corr, weights=None, seed=None):
    """Alias pro pravidelnou mřížku – volá generate_grf_nufft."""
    return generate_grf_nufft(x_points, corr, weights=weights, seed=seed)


# =============================================================================
# demo
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L, N, phi = 100.0, 256, 3.0
    seed = random.randint(3, 9)
    # --- 1D ---
    x = np.linspace(0, L, N)
    x_irr = np.sort(np.random.uniform(0, L, 300))
    white1d = make_white_noise(N, dim=1, seed=seed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (label, CorrClass) in enumerate([("Gaussovská", GaussianCorrelation),
                                               ("Exponenciální", ExponentialCorrelation)]):
        corr = CorrClass(L, N, phi, dim=1)
        axes[row][0].plot(x, generate_grf_fft(x, corr, weights=white1d), lw=1)
        axes[row][0].set_title(f"1D FFT {label} (φ={phi})")
        f_nu = generate_grf_nufft(x_irr, corr, weights=white1d)
        axes[row][1].plot(x_irr, f_nu, '-', alpha=0.4, color='gray', lw=0.8)
        axes[row][1].scatter(x_irr, f_nu, s=12, c=f_nu, cmap='viridis', zorder=3)
        axes[row][1].set_title(f"1D NUFFT {label} (φ={phi})")
    for ax in axes.flat: ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # --- 2D ---
    N2 = 64  # menší kvůli N^2 bodům
    white2d = make_white_noise(N2, dim=2, seed=seed)
    g = np.linspace(0, L, N2)
    xx, yy = np.meshgrid(g, g)
    pts2d = np.column_stack([xx.ravel(), yy.ravel()])

    # nepravidelné body pro NUFFT
    pts2d_irr = np.column_stack([np.random.uniform(0, L, N2**2),
                                  np.random.uniform(0, L, N2**2)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, (label, CorrClass) in enumerate([("Gaussovská", GaussianCorrelation),
                                               ("Exponenciální", ExponentialCorrelation)]):
        corr2d = CorrClass(L, N2, phi, dim=2)

        # pravidelná mřížka (FFT)
        field2d = generate_grf_nufft(pts2d, corr2d, weights=white2d).reshape(N2, N2)
        im = axes[0][col].imshow(field2d, extent=[0, L, 0, L], origin='lower', cmap='viridis')
        plt.colorbar(im, ax=axes[0][col])
        axes[0][col].set_title(f"2D FFT {label} (φ={phi})")

        # nepravidelná mřížka (NUFFT) – interpolováno na pravidelnou mřížku pro vizualizaci
        f_irr = generate_grf_nufft(pts2d_irr, corr2d, weights=white2d)
        from scipy.interpolate import griddata
        field_interp = griddata(pts2d_irr, f_irr, (xx, yy), method='linear')
        im2 = axes[1][col].imshow(field_interp, extent=[0, L, 0, L], origin='lower', cmap='viridis')
        plt.colorbar(im2, ax=axes[1][col])
        axes[1][col].set_title(f"2D NUFFT {label} (φ={phi}, interpolováno)")

    plt.tight_layout(); plt.show()