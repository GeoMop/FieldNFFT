"""
Generování náhodných polí pomocí spektrálních metod (FFT / NUFFT).
Úkoly 3.3:
  1) make_hermitian    – reálná funkce z náhodných komplexních koeficientů
  2) generate_grf_fft  – GRF na pravidelné mřížce (Gaussovská korelace)
  3) generate_grf_nufft – GRF na nepravidelné mřížce (finufft)

    literatura

"""

import numpy as np
from scipy.fft import fftfreq
import finufft


# SPEKTRÁLNÍ HUSTOTA

def spectral_density_gaussian(omega, phi, sigma=1.0):
    # C(r) = sigma^2 * exp(-r^2 / 2phi^2)  =>  S(w) = sigma^2 * sqrt(2pi)*phi * exp(-w^2*phi^2/2)
    return sigma**2 * np.sqrt(2 * np.pi) * phi * np.exp(-0.5 * (omega * phi)**2)

def spectral_density_exponential(omega, phi, sigma=1.0):
    # C(r) = sigma^2 * exp(-|r|/phi)  =>  S(w) = sigma^2 * 2phi / (1 + (w*phi)^2)
    return sigma**2 * 2 * phi / (1 + (omega * phi)**2)


# HERMITOVSKÁ SYMETRIE – zaručí reálný výstup IFFT

def make_hermitian(f_hat, centered=False):
    """
    Vyrobí hermitovsky symetrické koeficienty: f_hat[-k] = conj(f_hat[k])

    centered=False  – FFT pořadí:      DC na indexu 0, záporné frekvence na konci
    centered=True   – centrované pořadí: DC uprostřed na indexu N//2
    """
    N = len(f_hat)
    f = f_hat.copy()

    if centered:
        zi = N // 2                           # index DC (nulové frekvence)
        f[zi] = f[zi].real
        for i in range(1, N // 2):
            f[zi - i] = np.conj(f[zi + i])   # záporné frekvence = sdružené kladných
    else:
        f[0] = f[0].real                      # DC
        if N % 2 == 0:
            f[N // 2] = f[N // 2].real        # Nyquistova frekvence
        for k in range(1, N // 2):
            f[-k] = np.conj(f[k])

    return f


# FFT – PRAVIDELNÁ MŘÍŽKA

def generate_grf_fft(L, N, phi, sigma=1.0, seed=None,
                     spectral_density_fn=spectral_density_gaussian):
    """
    Generuje 1D náhodné pole na pravidelné mřížce.
      - náhodné koeficienty z N(0,1) komplexního rozdělení
      - modulace filtrem sqrt(S(omega))
      - hermitovská symetrie -> reálný výstup přes IFFT
    spectral_density_fn: spectral_density_gaussian | spectral_density_exponential
    """
    rng = np.random.default_rng(seed)
    dx = L / N
    x = np.arange(N) * dx

    omega = 2 * np.pi * fftfreq(N, d=dx)  # FFT pořadí
    S_k = spectral_density_fn(omega, phi, sigma)

    white = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    f_hat = make_hermitian(white * np.sqrt(S_k), centered=False)

    field = np.fft.ifft(f_hat).real * N / np.sqrt(L)
    return x, field


# NUFFT – NEPRAVIDELNÁ MŘÍŽKA

def generate_grf_nufft(L, N_freq, M_points, phi, sigma=1.0, seed=None, x_points=None,
                       spectral_density_fn=spectral_density_gaussian):
    """
    Generuje 1D náhodné pole na nepravidelné mřížce pomocí NUFFT typu 2
    (pravidelné frekvence -> nepravidelné prostorové body).
    finufft očekává souřadnice v [-pi, pi].
    """
    rng = np.random.default_rng(seed)

    # centrované pořadí: k = -N/2, ..., -1, 0, +1, ..., N/2-1
    k = np.arange(-N_freq // 2, N_freq // 2)
    omega = k * (2 * np.pi / L)
    S_omega = spectral_density_fn(omega, phi, sigma)

    white = (rng.standard_normal(N_freq) + 1j * rng.standard_normal(N_freq)) / np.sqrt(2)
    f_hat = make_hermitian(white * np.sqrt(S_omega), centered=True)

    if x_points is None:
        x_nufft = rng.uniform(-np.pi, np.pi, M_points)
    else:
        x_nufft = (np.asarray(x_points) / L) * 2 * np.pi - np.pi  # [0,L] -> [-pi,pi]

    field = finufft.nufft1d2(x_nufft, f_hat.astype(np.complex128))
    x_final = (x_nufft + np.pi) * (L / (2 * np.pi))               # [-pi,pi] -> [0,L]
    return x_final, field.real


