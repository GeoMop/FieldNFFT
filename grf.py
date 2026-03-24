"""
grf.py – knihovna pro generování náhodných polí (GRF)
"""

import numpy as np
import finufft
from scipy.fft import fft, ifft, fftfreq, fftshift


# =============================================================================
# KORELAČNÍ TŘÍDY
# =============================================================================

class GaussianCorrelation:
    """
    C(r) = sigma^2 * exp(-|r|^2 / 2*phi^2)
    S(w) = sigma^2 * (sqrt(2pi)*phi)^dim * exp(-|w|^2*phi^2/2)
    """
    def __init__(self, L, N_freq, phi, sigma=1.0, dim=1):
        self.L, self.N_freq, self.phi, self.sigma, self.dim = L, N_freq, phi, sigma, dim
        k = np.arange(-N_freq // 2, N_freq // 2)
        self.f_points = k * (2 * np.pi / L)   # centrované úhlové frekvence

    def spectral_density(self, omega_mag):
        return (self.sigma**2
                * (np.sqrt(2 * np.pi) * self.phi) ** self.dim
                * np.exp(-0.5 * (omega_mag * self.phi)**2))


class ExponentialCorrelation:
    """
    C(r) = sigma^2 * exp(-|r|/phi)
    1D: S(w) = sigma^2 * 2*phi / (1 + (w*phi)^2)
    2D: S(w) = sigma^2 * 2*pi*phi^2 / (1 + (|w|*phi)^2)^(3/2)
    """
    def __init__(self, L, N_freq, phi, sigma=1.0, dim=1):
        self.L, self.N_freq, self.phi, self.sigma, self.dim = L, N_freq, phi, sigma, dim
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
# HERMITOVSKÁ SYMETRIE  f_hat[-k] = conj(f_hat[k])  → reálný IFFT výstup
# =============================================================================

def make_hermitian_nd(f_hat):
    """nD hermitovská symetrie v centrovaném pořadí (DC uprostřed na indexu N//2)."""
    N = f_hat.shape[0]
    zi = N // 2
    f = f_hat.copy()
    f[tuple([zi] * f.ndim)] = f[tuple([zi] * f.ndim)].real   # DC reálná
    for idx in np.ndindex(*f.shape):
        shifted = tuple(i - zi for i in idx)
        if all(s <= 0 for s in shifted):
            continue
        f[tuple((zi - s) % N for s in shifted)] = np.conj(f[idx])
    return f


def make_white_noise(N_freq, dim=1, seed=None, use_gstools_rng=False):
    """
    Komplexní bílý šum tvaru (N_freq,)*dim ze standardního normálního rozdělení.

    use_gstools_rng=True  – použije gstools.random.RNG (stejný generátor jako GSTools interně)
                            self._rng = RNG(seed)
                            z_1 = self._rng.random.normal(size=N)
                            z_2 = self._rng.random.normal(size=N)
    use_gstools_rng=False – numpy default_rng (výchozí)
    """
    size = N_freq ** dim
    if use_gstools_rng:
        from gstools.random import RNG
        gs_rng = RNG(seed)
        rs = gs_rng.random   # numpy RandomState stream
        flat = (rs.normal(size=size) + 1j * rs.normal(size=size)) / np.sqrt(2)
    else:
        rng = np.random.default_rng(seed)
        flat = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)
    return flat.reshape((N_freq,) * dim)


# =============================================================================
# GENEROVÁNÍ GRF
# =============================================================================

def generate_grf(x_points, corr, weights=None, seed=None):
    """
    Generuje náhodné pole pomocí NUFFT typu 2.

    x_points : (M,)   pro 1D,  (M, 2) pro 2D
    corr     : GaussianCorrelation | ExponentialCorrelation
    weights  : bílý šum tvaru (N_freq,)*dim; None = vygeneruje se nový

    Vrací field.real tvaru (M,)
    """
    dim = corr.dim
    N_freq = corr.N_freq
    L = corr.L

    if weights is None:
        weights = make_white_noise(N_freq, dim=dim, seed=seed)

    # |omega| pro každý frekvenční bod
    if dim == 1:
        omega_mag = np.abs(corr.f_points)
    else:
        grids = np.meshgrid(*([corr.f_points] * dim), indexing='ij')
        omega_mag = np.sqrt(sum(g**2 for g in grids))

    S = corr.spectral_density(omega_mag)
    f_hat = make_hermitian_nd(weights * np.sqrt(S))

    x_points = np.asarray(x_points)

    if dim == 1:
        x_nu = (x_points / L) * 2 * np.pi - np.pi             # [0,L] -> [-pi,pi]
        field = finufft.nufft1d2(x_nu, f_hat.astype(np.complex128))
    elif dim == 2:
        x_nu = (x_points[:, 0] / L) * 2 * np.pi - np.pi
        y_nu = (x_points[:, 1] / L) * 2 * np.pi - np.pi
        field = finufft.nufft2d2(x_nu, y_nu, f_hat.astype(np.complex128))
    else:
        raise NotImplementedError("dim > 2")

    # normalizace: C(0) = sigma² = (1/2π)*∫S(ω)dω ≈ (Δω/2π)*ΣS(ωk) = (1/L)*ΣS(ωk)
    # -> faktor (1/L)^(dim/2)
    norm = (1.0 / L) ** (dim / 2)
    return (field * norm).real

# =============================================================================
# EMPIRICKÝ VARIOGRAM  gamma(h) = 0.5 * E[(f(x)-f(x+h))^2]
# =============================================================================

def empirical_variogram(x, field, n_bins=30, n_sample=300):
    """
    Odhadne variogram z realizace pole.
    Funguje pro 1D i 2D:
      x: (M,)   pro 1D
      x: (M, 2) pro 2D
    Bere podvzorek n_sample bodů kvůli rychlosti.
    """
    rng = np.random.default_rng(0)
    x = np.asarray(x)
    idx = rng.choice(len(field), size=min(len(field), n_sample), replace=False)
    xi, fi = x[idx], field[idx]

    # vzdálenost – funguje pro 1D i 2D
    if xi.ndim == 1:
        dists = np.abs(xi[:, None] - xi[None, :])          # (n, n)
    else:
        diff  = xi[:, None, :] - xi[None, :, :]            # (n, n, 2)
        dists = np.sqrt((diff ** 2).sum(axis=-1))           # (n, n)

    # jen horní trojúhelník (každý pár jednou)
    i_upper, j_upper = np.triu_indices(len(xi), k=1)
    pairs_h  = dists[i_upper, j_upper]
    pairs_sq = (fi[i_upper] - fi[j_upper]) ** 2

    bins = np.linspace(0, pairs_h.max(), n_bins + 1)
    h_vals, g_vals = [], []
    for k in range(n_bins):
        mask = (pairs_h >= bins[k]) & (pairs_h < bins[k + 1])
        if mask.sum() > 5:
            h_vals.append(0.5 * (bins[k] + bins[k + 1]))
            g_vals.append(0.5 * pairs_sq[mask].mean())

    return np.array(h_vals), np.array(g_vals)