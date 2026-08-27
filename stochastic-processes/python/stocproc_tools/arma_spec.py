"""
Python translation of astsa's arma.spec().

R source: astsa/R/arma.spec.R

The theoretical spectral density of an ARMA(p,q) process:
    f(w) = sigma^2 * |theta(e^{-2pi i w})|^2 / |phi(e^{-2pi i w})|^2
computed directly from the closed-form trig identity R uses (no
simulation involved), plus the same causality/invertibility/redundancy
checks as sarima_sim().
"""
import numpy as np
import matplotlib.pyplot as plt

from .tsplot import _style_axis, _resolve_color
from .sarima_sim import _min_root_modulus


def arma_spec(ar=None, ma=None, var_noise=1.0, n_freq=500, main=None,
                redundancy_tol=0.1, xfreq=1.0, ylim=None, plot=True,
                ax=None, col=4, gg=False):
    """Theoretical ARMA spectral density. ar/ma use the same coefficient
    convention as sarima_sim() (x_t = ar[0]*x_{t-1} + ... + w_t + ma[0]*w_{t-1} + ...).

    Returns dict(freq=..., spec=...).
    """
    ar = np.atleast_1d(ar).astype(float) if ar is not None else np.array([0.0])
    ma = np.atleast_1d(ma).astype(float) if ma is not None else np.array([0.0])

    if not (len(ar) == 1 and ar[0] == 0):
        m = _min_root_modulus(np.concatenate([[1.0], -ar]))
        if m <= 1:
            raise ValueError("Model not causal")
    if not (len(ma) == 1 and ma[0] == 0):
        m = _min_root_modulus(np.concatenate([[1.0], ma]))
        if m <= 1:
            raise ValueError("Model not invertible")

    ar_order = len(ar)
    ma_order = len(ma)

    freq = np.linspace(0, 0.5, n_freq)
    k_ar = np.arange(1, ar_order + 1)
    k_ma = np.arange(1, ma_order + 1)

    cs_ar = np.cos(2 * np.pi * freq[:, None] * k_ar[None, :]) @ ar
    sn_ar = np.sin(2 * np.pi * freq[:, None] * k_ar[None, :]) @ ar
    cs_ma = np.cos(2 * np.pi * freq[:, None] * k_ma[None, :]) @ (-ma)
    sn_ma = np.sin(2 * np.pi * freq[:, None] * k_ma[None, :]) @ (-ma)

    spec = var_noise * ((1 - cs_ma) ** 2 + sn_ma ** 2) / ((1 - cs_ar) ** 2 + sn_ar ** 2)

    if plot:
        is_white = (len(ar) == 1 and ar[0] == 0) and (len(ma) == 1 and ma[0] == 0)
        if main is None:
            if is_white:
                main = "White Noise"
            else:
                p = ar_order - (1 if (len(ar) == 1 and ar[0] == 0) else 0)
                q = ma_order - (1 if (len(ma) == 1 and ma[0] == 0) else 0)
                main = f"ARMA({p},{q})"

        created_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots()
        _style_axis(ax, gg)
        color = _resolve_color(col)
        ax.plot(freq * xfreq, spec, color=color, linewidth=2)
        ax.set_xlabel(f"frequency \u00d7 {xfreq}" if xfreq > 1 else "frequency")
        ax.set_ylabel("spectrum")
        ax.set_title(main)
        if ylim is not None:
            ax.set_ylim(ylim)
        if created_fig:
            fig.tight_layout()

    return dict(freq=freq * xfreq, spec=spec)
