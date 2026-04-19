"""
grf.py – Gaussian Random Field (GRF) generation via NUFFT type 2.

Public API
----------
GaussianCorrelation    – Gaussian correlation / spectral density model
ExponentialCorrelation – Exponential (Matérn-1/2) correlation model
make_white_noise       – complex white noise in Fourier space
make_hermitian_nd      – enforce Hermitian symmetry → real IFFT output
generate_grf           – generate a GRF realization at arbitrary points (1D, 2D, 3D)

Typical usage
-------------
>>> import numpy as np
>>> import grf
>>> L, N, phi = 100.0, 512, 5.0
>>> corr = grf.GaussianCorrelation(L=L, N_freq=N, phi=phi, dim=1)
>>> x = np.linspace(0, L, N)
>>> field = grf.generate_grf(x, corr, seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import finufft


# =============================================================================
# CORRELATION / SPECTRAL DENSITY MODELS
# =============================================================================

@dataclass
class GaussianCorrelation:
    """Gaussian correlation model.

    Correlation function (isotropic, stationary):

        C(r) = sigma^2 * exp(-|r|^2 / (2 * phi^2))

    Fourier transform (spectral density, exact):

        S(omega) = sigma^2 * (sqrt(2*pi) * phi)^dim
                   * exp(-|omega|^2 * phi^2 / 2)

    Parameters
    ----------
    L : float
        Domain size [m].  The domain is the cube [0, L]^dim.
    N_freq : int
        Number of frequency grid points per axis.
        Total nodes in Fourier space: N_freq^dim.
    phi : float
        Correlation length [m].  Controls the spatial range of dependence.
        Larger phi → smoother fields.
    sigma : float
        Standard deviation of the field (sqrt of variance).  Default 1.0.
    dim : int
        Spatial dimension: 1, 2, or 3.  Default 1.

    Attributes
    ----------
    f_points : np.ndarray, shape (N_freq,)
        Centered angular frequencies [rad/m]:

            omega_k = k * (2*pi / L),   k = -N_freq//2, ..., N_freq//2 - 1

        The same frequency axis is used for every spatial dimension.

    Notes
    -----
    Spectral resolution: the frequency step is Δω = 2π/L and the highest
    represented frequency is ω_max = (N_freq//2) * 2π/L.  For reliable
    variance reproduction, the condition phi * Δω << 1 should hold, i.e.,
    L >> 2π * phi  (at least L > 8 * phi is recommended).
    """

    L: float
    N_freq: int
    phi: float
    sigma: float = 1.0
    dim: int = 1
    f_points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        k = np.arange(-self.N_freq // 2, self.N_freq // 2)
        self.f_points = k * (2.0 * np.pi / self.L)

    def spectral_density(self, omega_mag: np.ndarray) -> np.ndarray:
        """Evaluate the spectral density S(|omega|).

        Parameters
        ----------
        omega_mag : np.ndarray, arbitrary shape
            Magnitudes of angular frequency vectors [rad/m].
            For a 1D model this is shape (N_freq,);
            for nD it is the element-wise norm over an (N_freq,)*dim grid.

        Returns
        -------
        S : np.ndarray, same shape as omega_mag
            Non-negative spectral density values.
        """
        return (
            self.sigma ** 2
            * (np.sqrt(2.0 * np.pi) * self.phi) ** self.dim
            * np.exp(-0.5 * (omega_mag * self.phi) ** 2)
        )


@dataclass
class ExponentialCorrelation:
    """Exponential (Matérn-1/2) correlation model.

    Correlation function (isotropic, stationary):

        C(r) = sigma^2 * exp(-|r| / phi)

    Spectral density (exact, dimension-dependent):

        1D: S(omega) = sigma^2 * 2*phi / (1 + (omega*phi)^2)
        2D: S(omega) = sigma^2 * 2*pi*phi^2 / (1 + (|omega|*phi)^2)^(3/2)
        3D: S(omega) = sigma^2 * 8*pi*phi^3 / (1 + (|omega|*phi)^2)^2

    Parameters
    ----------
    L : float
        Domain size [m].  The domain is the cube [0, L]^dim.
    N_freq : int
        Number of frequency grid points per axis.
        Total nodes in Fourier space: N_freq^dim.
    phi : float
        Correlation length [m].  Controls the spatial range of dependence.
    sigma : float
        Standard deviation of the field (sqrt of variance).  Default 1.0.
    dim : int
        Spatial dimension: 1, 2, or 3.  Default 1.

    Attributes
    ----------
    f_points : np.ndarray, shape (N_freq,)
        Centered angular frequencies [rad/m]:

            omega_k = k * (2*pi / L),   k = -N_freq//2, ..., N_freq//2 - 1

    Notes
    -----
    The exponential model produces fields that are only mean-square continuous
    (not differentiable), yielding rougher realizations than the Gaussian model.
    """

    L: float
    N_freq: int
    phi: float
    sigma: float = 1.0
    dim: int = 1
    f_points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        k = np.arange(-self.N_freq // 2, self.N_freq // 2)
        self.f_points = k * (2.0 * np.pi / self.L)

    def spectral_density(self, omega_mag: np.ndarray) -> np.ndarray:
        """Evaluate the spectral density S(|omega|).

        Parameters
        ----------
        omega_mag : np.ndarray, arbitrary shape
            Magnitudes of angular frequency vectors [rad/m].

        Returns
        -------
        S : np.ndarray, same shape as omega_mag
            Non-negative spectral density values.

        Raises
        ------
        NotImplementedError
            If ``self.dim`` is not 1, 2, or 3.
        """
        p = (omega_mag * self.phi) ** 2
        if self.dim == 1:
            return self.sigma ** 2 * 2.0 * self.phi / (1.0 + p)
        elif self.dim == 2:
            return self.sigma ** 2 * 2.0 * np.pi * self.phi ** 2 / (1.0 + p) ** 1.5
        elif self.dim == 3:
            return self.sigma ** 2 * 8.0 * np.pi * self.phi ** 3 / (1.0 + p) ** 2
        else:
            raise NotImplementedError(
                f"ExponentialCorrelation: dim={self.dim} not supported (max dim=3)"
            )


# =============================================================================
# HERMITIAN SYMMETRY  →  real-valued IFFT output
# =============================================================================

def make_hermitian_nd(f_hat: np.ndarray) -> np.ndarray:
    """Enforce Hermitian symmetry so that the IFFT of the output is real-valued.

    The symmetry condition is  f_hat[-k] = conj(f_hat[k])  for every
    multi-index k.  The function operates in *centered* layout where the
    DC component sits at index ``zi = N//2`` along every axis (consistent
    with ``np.fft.fftshift``).  All axes must have the same even length N.

    Parameters
    ----------
    f_hat : np.ndarray, shape (N,)^dim
        Complex Fourier coefficients in centered order.
        Supported shapes: (N,), (N, N), (N, N, N).

    Returns
    -------
    f : np.ndarray, same shape and dtype as f_hat
        Copy of the input with Hermitian symmetry enforced:

        * DC coefficient (index ``zi`` in every axis) → set to real part.
        * For every multi-index with at least one positive shifted component,
          the conjugate-mirror index is overwritten:
          ``f[zi - shift] = conj(f[zi + shift])``.

    Notes
    -----
    After applying this function, ``np.fft.ifftn(np.fft.ifftshift(f)).real``
    will have negligible imaginary part (machine precision).
    """
    N = f_hat.shape[0]
    zi = N // 2
    f = f_hat.copy()
    # DC component must be real
    f[tuple([zi] * f.ndim)] = f[tuple([zi] * f.ndim)].real
    for idx in np.ndindex(*f.shape):
        shifted = tuple(i - zi for i in idx)
        # Only process indices with at least one positive shifted component
        # to avoid processing each pair twice
        if all(s <= 0 for s in shifted):
            continue
        f[tuple((zi - s) % N for s in shifted)] = np.conj(f[idx])
    return f


# =============================================================================
# WHITE NOISE IN FOURIER SPACE
# =============================================================================

def make_white_noise(
    N_freq: int,
    dim: int = 1,
    seed: int | None = None,
    use_gstools_rng: bool = False,
) -> np.ndarray:
    """Generate complex white noise in Fourier space.

    Each component is an independent complex Gaussian variable with zero mean
    and unit variance:  z ~ CN(0, 1),  i.e.  Re(z), Im(z) ~ N(0, 1/2).

    Parameters
    ----------
    N_freq : int
        Number of frequency points per axis.
    dim : int
        Spatial dimension (1, 2, or 3).  Default 1.
    seed : int or None
        Random seed for reproducibility.  ``None`` draws a random seed.
    use_gstools_rng : bool
        If ``True``, use ``gstools.random.RNG`` (the same generator as
        GSTools uses internally) instead of ``numpy.random.default_rng``.
        Useful for exact reproducibility comparisons with GSTools.
        Default ``False``.

    Returns
    -------
    noise : np.ndarray, shape (N_freq,) * dim, dtype complex128
        Complex white noise array.
    """
    size = N_freq ** dim
    if use_gstools_rng:
        from gstools.random import RNG
        gs_rng = RNG(seed)
        rs = gs_rng.random  # underlying numpy RandomState
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
    """Generate a Gaussian Random Field realization at arbitrary points.

    Uses NUFFT type 2 (uniform Fourier coefficients → non-uniform spatial points).

    Algorithm
    ---------
    1. Draw (or accept) complex white noise  w_k ~ CN(0, 1)  in Fourier space.
    2. Color by spectrum:  ``f_hat_k = w_k * sqrt(S(|omega_k|))``.
    3. Enforce Hermitian symmetry → guarantees a real-valued output.
    4. NUFFT type 2:
       ``f(x_j) = sum_k  f_hat_k * exp(i * omega_k · x_j)``
    5. Normalize by  ``(1 / L)^(dim/2)``  so that  ``Var[f] ≈ sigma^2``.

    Normalization derivation:

        C(0) = sigma^2 = (1 / (2*pi)^dim) * ∫ S(omega) d^dim omega
                      ≈ (Δω / 2π)^dim * Σ_k S(omega_k)
                      = (1 / L)^dim   * Σ_k S(omega_k)

        The NUFFT computes  Σ_k f_hat_k * exp(...)  without the (1/L)^dim
        prefactor, so we multiply the output by  (1/L)^(dim/2)  (since
        f_hat already carries one sqrt(S) factor).

    Parameters
    ----------
    x_points : np.ndarray
        Spatial coordinates of the output points.

        * 1D: shape ``(M,)``
        * 2D: shape ``(M, 2)``  – columns are x and y.
        * 3D: shape ``(M, 3)``  – columns are x, y, and z.

        Coordinates must lie in ``[0, L]`` (the domain defined by ``corr.L``).
    corr : GaussianCorrelation | ExponentialCorrelation
        Correlation model providing ``L``, ``N_freq``, ``dim``, ``f_points``,
        and ``spectral_density()``.
    weights : np.ndarray or None
        Pre-generated white noise, shape ``(N_freq,)^dim``, dtype complex128.
        When provided, ``seed`` is ignored.  Pass the same ``weights`` to
        different point sets to obtain the same underlying field realization
        on different grids.
    seed : int or None
        Seed for internal noise generation.  Ignored when ``weights`` is given.

    Returns
    -------
    field : np.ndarray, shape ``(M,)``
        Real-valued GRF evaluated at ``x_points``.

    Raises
    ------
    NotImplementedError
        If ``corr.dim`` is not 1, 2, or 3.

    Examples
    --------
    >>> import numpy as np, grf
    >>> corr = grf.GaussianCorrelation(L=100.0, N_freq=256, phi=5.0, dim=2)
    >>> pts = np.random.default_rng(0).uniform(0, 100, (500, 2))
    >>> f = grf.generate_grf(pts, corr, seed=0)
    >>> f.shape
    (500,)
    """
    dim = corr.dim
    N_freq = corr.N_freq
    L = corr.L

    if weights is None:
        weights = make_white_noise(N_freq, dim=dim, seed=seed)

    # Build |omega| array over the full frequency grid
    if dim == 1:
        omega_mag = np.abs(corr.f_points)
    else:
        grids = np.meshgrid(*([corr.f_points] * dim), indexing='ij')
        omega_mag = np.sqrt(sum(g ** 2 for g in grids))

    # Color noise and enforce Hermitian symmetry
    S = corr.spectral_density(omega_mag)
    f_hat = make_hermitian_nd(weights * np.sqrt(S))

    x_points = np.asarray(x_points)

    # Map [0, L] -> [-pi, pi] (finufft convention for type-2 transform)
    if dim == 1:
        x_nu = (x_points / L) * 2.0 * np.pi - np.pi
        field = finufft.nufft1d2(x_nu, f_hat.astype(np.complex128))
    elif dim == 2:
        x_nu = (x_points[:, 0] / L) * 2.0 * np.pi - np.pi
        y_nu = (x_points[:, 1] / L) * 2.0 * np.pi - np.pi
        field = finufft.nufft2d2(x_nu, y_nu, f_hat.astype(np.complex128))
    elif dim == 3:
        x_nu = (x_points[:, 0] / L) * 2.0 * np.pi - np.pi
        y_nu = (x_points[:, 1] / L) * 2.0 * np.pi - np.pi
        z_nu = (x_points[:, 2] / L) * 2.0 * np.pi - np.pi
        field = finufft.nufft3d2(x_nu, y_nu, z_nu, f_hat.astype(np.complex128))
    else:
        raise NotImplementedError(
            f"generate_grf: dim={dim} not supported (max dim=3)"
        )

    norm = (1.0 / L) ** (dim / 2)
    return (field * norm).real