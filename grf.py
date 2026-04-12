"""
grf.py – knihovna pro generování náhodných polí (GRF)
"""

import numpy as np
from dataclasses import dataclass, field
import finufft
from scipy.fft import fft, ifft, fftfreq, fftshift


# =============================================================================
# KORELAČNÍ TŘÍDY
# =============================================================================
@dataclass
class GaussianCorrelation:
    """Gaussian correlation model.
 
    Correlation function:
        C(r) = sigma^2 * exp(-|r|^2 / (2 * phi^2))
 
    Spectral density:
        S(omega) = sigma^2 * (sqrt(2*pi) * phi)^dim * exp(-|omega|^2 * phi^2 / 2)
 
    Parameters
    ----------
    L : float
        Domain size [m]. Assumes a cubic domain [0, L]^dim.
    N_freq : int
        Number of frequency points per axis. Total N_freq^dim nodes in Fourier space.
    phi : float
        Correlation length [m]. Controls the spatial range of dependence.
    sigma : float
        Standard deviation of the field (square root of variance). Default 1.0.
    dim : int
        Spatial dimension (1 or 2). Default 1.
 
    Attributes
    ----------
    f_points : np.ndarray, shape (N_freq,)
        Centered angular frequencies [rad/m]:
        omega_k = k * (2*pi / L)  for k = -N_freq//2, ..., N_freq//2 - 1
    """
    L: float
    N_freq: int
    phi: float
    sigma: float = 1.0
    dim: int = 1
    f_points: np.ndarray = field(init=False, repr=False)
 
    def __post_init__(self) -> None:
        k = np.arange(-self.N_freq // 2, self.N_freq // 2)
        self.f_points = k * (2 * np.pi / self.L)
 
    def spectral_density(self, omega_mag: np.ndarray) -> np.ndarray:
        """Evaluate spectral density S(|omega|).
 
        Parameters
        ----------
        omega_mag : np.ndarray, arbitrary shape
            Magnitudes of angular frequencies [rad/m].
 
        Returns
        -------
        S : np.ndarray, same shape as omega_mag
        """
        return (
            self.sigma ** 2
            * (np.sqrt(2 * np.pi) * self.phi) ** self.dim
            * np.exp(-0.5 * (omega_mag * self.phi) ** 2)
        )
 
 
@dataclass
class ExponentialCorrelation:
    """Exponential (Matérn-1/2) correlation model.
 
    Correlation function:
        C(r) = sigma^2 * exp(-|r| / phi)
 
    Spectral density:
        1D: S(omega) = sigma^2 * 2*phi / (1 + (omega*phi)^2)
        2D: S(omega) = sigma^2 * 2*pi*phi^2 / (1 + (|omega|*phi)^2)^(3/2)
 
    Parameters
    ----------
    L : float
        Domain size [m]. Assumes a cubic domain [0, L]^dim.
    N_freq : int
        Number of frequency points per axis. Total N_freq^dim nodes in Fourier space.
    phi : float
        Correlation length [m]. Controls the spatial range of dependence.
    sigma : float
        Standard deviation of the field (square root of variance). Default 1.0.
    dim : int
        Spatial dimension (1 or 2). Default 1.
 
    Attributes
    ----------
    f_points : np.ndarray, shape (N_freq,)
        Centered angular frequencies [rad/m]:
        omega_k = k * (2*pi / L)  for k = -N_freq//2, ..., N_freq//2 - 1
    """
 
    L: float
    N_freq: int
    phi: float
    sigma: float = 1.0
    dim: int = 1
    f_points: np.ndarray = field(init=False, repr=False)
 
    def __post_init__(self) -> None:
        k = np.arange(-self.N_freq // 2, self.N_freq // 2)
        self.f_points = k * (2 * np.pi / self.L)
 
    def spectral_density(self, omega_mag: np.ndarray) -> np.ndarray:
        """Evaluate spectral density S(|omega|).
 
        Parameters
        ----------
        omega_mag : np.ndarray, arbitrary shape
            Magnitudes of angular frequencies [rad/m].
 
        Returns
        -------
        S : np.ndarray, same shape as omega_mag
 
        Raises
        ------
        NotImplementedError
            For dim > 2.
        """
        if self.dim == 1:
            return self.sigma ** 2 * 2 * self.phi / (1 + (omega_mag * self.phi) ** 2)
        elif self.dim == 2:
            return (
                self.sigma ** 2
                * 2 * np.pi * self.phi ** 2
                / (1 + (omega_mag * self.phi) ** 2) ** 1.5
            )
        else:
            raise NotImplementedError(
                f"ExponentialCorrelation: dim={self.dim} not implemented (max dim=2)"
            )
 
 


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


def make_white_noise(
    N_freq: int,
    dim: int = 1,
    seed: int | None = None,
    use_gstools_rng: bool = False,
) -> np.ndarray:
    """Generate complex white noise in Fourier space.
 
    Each component is an independent complex Gaussian variable with zero mean
    and unit variance: z ~ CN(0, 1).
 
    Parameters
    ----------
    N_freq : int
        Number of frequency points per axis.
    dim : int
        Spatial dimension. Default 1.
    seed : int or None
        Random seed for reproducibility. None = random seed.
    use_gstools_rng : bool
        True  - use gstools.random.RNG for direct comparison with GSTools.
        False - use numpy.random.default_rng (default, recommended).
 
    Returns
    -------
    noise : np.ndarray, shape (N_freq,) * dim, dtype complex128
    """
    size = N_freq ** dim
    if use_gstools_rng:
        from gstools.random import RNG
        gs_rng = RNG(seed)
        rs = gs_rng.random
        flat = (rs.normal(size=size) + 1j * rs.normal(size=size)) / np.sqrt(2)
    else:
        rng = np.random.default_rng(seed)
        flat = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)
    return flat.reshape((N_freq,) * dim)


# =============================================================================
# GRF GENERATION
# =============================================================================

def generate_grf(
    x_points: np.ndarray,
    corr: GaussianCorrelation | ExponentialCorrelation,
    weights: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a GRF realization at arbitrary points via NUFFT type 2.

    x_points : np.ndarray, shape (M,) for 1D or (M, 2) for 2D
    corr     : GaussianCorrelation | ExponentialCorrelation
    weights  : white noise, shape (N_freq,)^dim; None = generate from seed

    Returns field.real, shape (M,)
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
# EMPIRICKÝ V`ARIOGRAM  gamma(h) = 0.5 * E[(f(x)-f(x+h))^2]
# =============================================================================

def empirical_variogram(x, field, n_bins=30, n_sample=300):
    """
    Estimate the empirical variogram from a single field realization.
    Works for both 1D and 2D. Subsamples n_sample points to reduce O(M^2) cost.

    x     : shape (M,) for 1D, (M, 2) for 2D
    field : shape (M,)
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