"""
Python translation of astsa's acf1() and acf2().

R source: astsa/R/acf1.R, astsa/R/acf2.R

Both are thin, styled wrappers around R's acf()/pacf() that:
  - pick a sensible default max.lag based on series length
  - plot with tsplot()'s look, with the usual +-2/sqrt(n) bounds

acf2() is acf1()'s two-panel version showing ACF and PACF together
(this is the one used most often in Ch.5 of the book).
"""
import numpy as np
import matplotlib.pyplot as plt
from .tsplot import tsplot, _style_axis


def _default_max_lag(num, xfreq=1):
    if num > 59:
        return max(int(np.ceil(10 + np.sqrt(num))), 4 * xfreq)
    return int(np.floor(6 * np.log10(num)))


def acf(series, nlags):
    """Sample autocorrelation at lags 0..nlags (matches R's acf()$acf)."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    return np.array([1.0 if k == 0 else np.dot(x[:n - k], x[k:]) / n / c0
                      for k in range(nlags + 1)])


def pacf(series, nlags):
    """Sample partial autocorrelation at lags 0..nlags via Durbin-Levinson
    recursion (matches R's pacf(), up to the usual sign/estimator
    conventions -- both use the Yule-Walker/innovations approach)."""
    r = acf(series, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    pacf_vals = np.zeros(nlags + 1)
    phi[1, 1] = r[1]
    pacf_vals[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf_vals[k] = phi[k, k]
    return pacf_vals


def acf1(series, max_lag=None, plot=True, main=None, ylim=None, pacf_=False,
          ylab=None, xlab=None, xfreq=1):
    """Plot (or just return) the sample ACF (or PACF if pacf_=True).

    `series` : array-like or pandas Series.
    `xfreq`  : the sampling frequency (e.g. 12 for monthly data), used only
               to label the lag axis in the same units R's frequency()
               would (R infers this from a ts object automatically).
    """
    series = np.asarray(series, dtype=float)
    num = len(series)
    if num < 3:
        raise ValueError("More than 2 observations are needed")

    if max_lag is None:
        max_lag = _default_max_lag(num, xfreq)
    max_lag = min(max_lag, num - 1)

    if pacf_:
        ACF = pacf(series, max_lag)[1:]
    else:
        ACF = acf(series, max_lag)[1:]

    if not plot:
        return ACF

    U = -1 / num + 2 / np.sqrt(num)
    L = -1 / num - 2 / np.sqrt(num)
    if ylim is None:
        minu = min(ACF.min(), L) - 0.01
        maxu = min(ACF.max() + 0.2, 1)
        ylim = (minu, maxu)

    Xlab = xlab or (f"LAG \u00f7 {xfreq}" if xfreq > 1 else "LAG")
    Ylab = ylab or ("PACF" if pacf_ else "ACF")
    LAG = np.arange(1, max_lag + 1) / xfreq

    fig, ax = plt.subplots()
    ax.vlines(LAG, 0, ACF, color=4 if False else "#1874cd")  # type='h' spikes
    _style_axis(ax, gg=False)
    ax.axhline(0, color="0.5", linewidth=1)
    ax.axhline(L, color="#1874cd", linestyle="--", linewidth=1)
    ax.axhline(U, color="#1874cd", linestyle="--", linewidth=1)
    ax.set_ylim(ylim)
    ax.set_xlabel(Xlab)
    ax.set_ylabel(Ylab)
    ax.set_title(main or "")
    fig.tight_layout()
    return np.round(ACF, 2)


def acf2(series, max_lag=None, plot=True, main=None, ylim=None, xfreq=1):
    """Two-panel ACF + PACF plot, matching R's acf2().

    Returns a (2, max_lag) array stacked as [ACF, PACF] (same as R's
    rbind(ACF, PACF)) when plot=True, or a (max_lag, 2) array (columns
    ACF, PACF, matching R's cbind) when plot=False.
    """
    series = np.asarray(series, dtype=float)
    num = len(series)
    if num < 3:
        raise ValueError("More than 2 observations are needed")

    if max_lag is None:
        max_lag = _default_max_lag(num, xfreq)
    max_lag = min(max_lag, num - 1)

    ACF = acf(series, max_lag)[1:]
    PACF = pacf(series, max_lag)[1:]

    if not plot:
        return np.column_stack([ACF, PACF])

    U = -1 / num + 2 / np.sqrt(num)
    L = -1 / num - 2 / np.sqrt(num)
    if ylim is None:
        minu = min(ACF.min(), PACF.min(), L) - 0.01
        maxu = min(max(ACF.max() + 0.1, PACF.max() + 0.1), 1)
        ylim = (minu, maxu)

    Xlab = f"LAG \u00f7 {xfreq}" if xfreq > 1 else "LAG"
    LAG = np.arange(1, max_lag + 1) / xfreq

    fig, axes = plt.subplots(2, 1, sharex=True)
    for ax, vals, ylab, title in zip(axes, [ACF, PACF], ["ACF", "PACF"],
                                       [main, None]):
        ax.vlines(LAG, 0, vals, color="#1874cd")
        _style_axis(ax, gg=False)
        ax.axhline(0, color="0.5", linewidth=1)
        ax.axhline(L, color="#1874cd", linestyle="--", linewidth=1)
        ax.axhline(U, color="#1874cd", linestyle="--", linewidth=1)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylab)
        if title:
            ax.set_title(title)
    axes[-1].set_xlabel(Xlab)
    fig.tight_layout()

    return np.vstack([np.round(ACF, 2), np.round(PACF, 2)])
