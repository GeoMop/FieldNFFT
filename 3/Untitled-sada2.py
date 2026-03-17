"""
Živá vizualizace 2D GRF – spektrální metoda (FFT)
Každou sekundu nový white noise; přepínání Gaussovská ↔ Exponenciální.

Postaveno na původním kódu (GaussianCorrelation, ExponentialCorrelation,
make_hermitian_nd, make_white_noise) — generování přes numpy.fft místo finufft,
výsledek je identický pro pravidelnou mřížku.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, RadioButtons
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    "figure.facecolor":  "#07080f",
    "axes.facecolor":    "#07080f",
    "text.color":        "#9ac8e8",
    "axes.edgecolor":    "#1a3a5a",
    "axes.labelcolor":   "#4a7a9a",
    "xtick.color":       "#2a5a7a",
    "ytick.color":       "#2a5a7a",
    "font.family":       "monospace",
    "font.size":         9,
})


# =============================================================================
# CORRELATION TŘÍDY  (beze změny logiky z původního souboru)
# =============================================================================

class GaussianCorrelation:
    """C(r) = σ² exp(−|r|²/2φ²)  →  S_2D(ω) = σ²(√2π φ)² exp(−|ω|²φ²/2)"""
    def __init__(self, L, N_freq, phi, sigma=1.0, dim=2):
        self.phi, self.sigma, self.dim = phi, sigma, dim
        self.N_freq, self.L = N_freq, L
        k = np.arange(-N_freq // 2, N_freq // 2)
        self.f_points = k * (2 * np.pi / L)

    def spectral_density(self, omega_mag):
        return (self.sigma ** 2
                * (np.sqrt(2 * np.pi) * self.phi) ** self.dim
                * np.exp(-0.5 * (omega_mag * self.phi) ** 2))


class ExponentialCorrelation:
    """C(r) = σ² exp(−|r|/φ)  →  S_2D(ω) = σ² 2π φ² / (1+|ω|²φ²)^(3/2)"""
    def __init__(self, L, N_freq, phi, sigma=1.0, dim=2):
        self.phi, self.sigma, self.dim = phi, sigma, dim
        self.N_freq, self.L = N_freq, L
        k = np.arange(-N_freq // 2, N_freq // 2)
        self.f_points = k * (2 * np.pi / L)

    def spectral_density(self, omega_mag):
        if self.dim == 2:
            return (self.sigma ** 2 * 2 * np.pi * self.phi ** 2
                    / (1 + (omega_mag * self.phi) ** 2) ** 1.5)
        raise NotImplementedError("dim != 2")


# =============================================================================
# HERMITOVSKÁ SYMETRIE  (původní make_hermitian_nd – zrychlená pro 2D)
# =============================================================================

def make_hermitian_2d(f_hat):
    """
    Hermitovská symetrie v centrovaném pořadí pro 2D pole.
    Použito np.roll místo explicitní iterace → ~100× rychlejší.
    """
    N = f_hat.shape[0]
    f = f_hat.copy()
    # f[-kx, -ky] = conj(f[kx, ky])
    # v centrovaném pořadí: index zi odpovídá k=0
    zi = N // 2
    # flip přes obě osy a otočit o jeden prvek, aby DC zůstalo na místě
    conj_flip = np.conj(np.roll(np.roll(f[::-1, ::-1], 1, axis=0), 1, axis=1))
    # DC musí být reálný
    mask = np.ones((N, N), dtype=bool)
    mask[zi, zi] = False
    # průměrujeme: f_herm = (f + conj_flip) / sqrt(2) zachovává rozptyl
    f = (f + conj_flip) / np.sqrt(2)
    f[zi, zi] = f[zi, zi].real
    return f


def make_white_noise(N_freq, dim=2, seed=None):
    """Komplexní bílý šum tvaru (N_freq,)*dim."""
    rng = np.random.default_rng(seed)
    shape = (N_freq,) * dim
    size = N_freq ** dim
    flat = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)
    return flat.reshape(shape)


# =============================================================================
# GENEROVÁNÍ GRF  (pravidelná 2D mřížka – numpy.fft místo finufft)
# =============================================================================

def generate_grf_fft_2d(N, corr, weights=None, seed=None):
    """
    Generuje 2D GRF na pravidelné mřížce N×N.

    Ekvivalentní generate_grf_nufft pro pts2d na pravidelné mřížce,
    ale bez závislosti na finufft.
    """
    if weights is None:
        weights = make_white_noise(N, dim=2, seed=seed)

    # |ω| pro každý frekvenční bod
    gx, gy = np.meshgrid(corr.f_points, corr.f_points, indexing='ij')
    omega_mag = np.sqrt(gx ** 2 + gy ** 2)

    S = corr.spectral_density(omega_mag)                   # (N, N)
    f_hat_centered = make_hermitian_2d(weights * np.sqrt(S))

    # centrované → standardní FFT pořadí
    f_hat = np.fft.ifftshift(f_hat_centered)

    # IFFT → reálná část
    field = np.fft.ifft2(f_hat).real                       # (N, N)
    return field


# =============================================================================
# GUI
# =============================================================================

L   = 100.0
N   = 128
PHI = 8.0

CMAP = "turbo"

CORR_CLASSES = {
    "Gaussovská":     GaussianCorrelation,
    "Exponenciální":  ExponentialCorrelation,
}


class LiveGRF:
    def __init__(self):
        self.corr_name = "Gaussovská"
        self.phi       = PHI
        self.paused    = False
        self.frame     = 0
        self._build_corr()

        # ── layout ──────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(9, 8), facecolor="#07080f")
        self.fig.canvas.manager.set_window_title("2D GRF – živá vizualizace")

        gs = gridspec.GridSpec(
            4, 3,
            figure=self.fig,
            left=0.06, right=0.96,
            top=0.93,  bottom=0.04,
            hspace=0.55, wspace=0.35,
            height_ratios=[14, 1.2, 1.2, 1.2],
        )

        # hlavní obraz
        self.ax_img = self.fig.add_subplot(gs[0, :])
        self.ax_img.set_aspect("equal")
        self.ax_img.tick_params(labelsize=7)
        self.ax_img.set_xlabel("x  [m]", fontsize=8)
        self.ax_img.set_ylabel("y  [m]", fontsize=8)

        # slider φ
        ax_sl = self.fig.add_subplot(gs[1, :])
        ax_sl.set_facecolor("#07080f")
        self.slider = Slider(
            ax_sl, "φ  [korelační délka]",
            valmin=1, valmax=30, valinit=self.phi, valstep=0.5,
            color="#1a4a7a", track_color="#0d1a2a",
        )
        self.slider.label.set_color("#4a9adf")
        self.slider.valtext.set_color("#7ecfff")
        self.slider.on_changed(self._on_phi)

        # radio tlačítka
        ax_radio = self.fig.add_subplot(gs[2, 0])
        ax_radio.set_facecolor("#07080f")
        self.radio = RadioButtons(
            ax_radio,
            labels=list(CORR_CLASSES.keys()),
            active=0,
            activecolor="#4a9adf",
        )
        for lbl in self.radio.labels:
            lbl.set_fontsize(9)
            lbl.set_color("#9ac8e8")
        self.radio.on_clicked(self._on_corr)

        # pause / step tlačítka
        ax_btn_pause = self.fig.add_subplot(gs[2, 1])
        ax_btn_step  = self.fig.add_subplot(gs[2, 2])
        for ax in (ax_btn_pause, ax_btn_step):
            ax.set_facecolor("#07080f")

        self.btn_pause = Button(ax_btn_pause, "⏸  Pauza",
                                color="#0d1520", hovercolor="#162030")
        self.btn_step  = Button(ax_btn_step,  "↻  Generovat",
                                color="#0d1520", hovercolor="#162030")
        for btn in (self.btn_pause, self.btn_step):
            btn.label.set_color("#7ecfff")
            btn.label.set_fontsize(9)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_step.on_clicked(self._on_step)

        # info text ve spodku
        self.ax_info = self.fig.add_subplot(gs[3, :])
        self.ax_info.axis("off")
        self.info_txt = self.ax_info.text(
            0.5, 0.5, "", ha="center", va="center",
            transform=self.ax_info.transAxes,
            fontsize=8, color="#2a6a9a",
        )

        # colorbar
        self.cbar = None

        # první snímek
        field = self._gen()
        self.im = self.ax_img.imshow(
            field, extent=[0, L, 0, L],
            origin="lower", cmap=CMAP,
            interpolation="bilinear",
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_img,
                                       fraction=0.03, pad=0.02)
        self.cbar.ax.tick_params(colors="#4a7a9a", labelsize=7)
        self._set_title()
        self._update_info(field)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _build_corr(self):
        cls = CORR_CLASSES[self.corr_name]
        self.corr = cls(L, N, self.phi, dim=2)

    def _gen(self):
        self.frame += 1
        return generate_grf_fft_2d(N, self.corr)

    def _set_title(self):
        self.ax_img.set_title(
            f"2D GRF  ·  {self.corr_name} korelace  ·  "
            f"φ = {self.phi:.1f} m  ·  frame #{self.frame:04d}",
            fontsize=10, color="#7ecfff", pad=6,
        )

    def _update_info(self, field):
        mn, mx, sd = field.min(), field.max(), field.std()
        self.info_txt.set_text(
            f"min = {mn:+.4f}   max = {mx:+.4f}   σ = {sd:.4f}   "
            f"N = {N}×{N}   L = {L} m"
        )

    # ── callbacks ────────────────────────────────────────────────────────────

    def _on_phi(self, val):
        self.phi = val
        self._build_corr()
        self._refresh()

    def _on_corr(self, label):
        self.corr_name = label
        self._build_corr()
        self._refresh()

    def _on_pause(self, _):
        self.paused = not self.paused
        self.btn_pause.label.set_text("▶  Spustit" if self.paused else "⏸  Pauza")
        self.fig.canvas.draw_idle()

    def _on_step(self, _):
        self._refresh()

    def _refresh(self):
        field = self._gen()
        self.im.set_data(field)
        self.im.set_clim(field.min(), field.max())
        self._set_title()
        self._update_info(field)
        self.fig.canvas.draw_idle()

    # ── animace ──────────────────────────────────────────────────────────────

    def animate(self, _i):
        if not self.paused:
            self._refresh()

    def run(self):
        self._anim = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=1000,   # každou sekundu
            cache_frame_data=False,
        )
        plt.show()


# =============================================================================

if __name__ == "__main__":
    app = LiveGRF()
    app.run()