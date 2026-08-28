"""
Python translation of astsa's mvspec() -- univariate case only.

R source: astsa/R/mvspec.R

astsa's mvspec() is genuinely multivariate (coherency/phase panels for 3+
series), but both calls in this book are univariate:
    mvspec(rnorm(2^10), ...)
    mvspec(sunspots, spans=c(3,3,3), ...)
so only that path is translated here. If a later chapter needs the
coherency/phase panels for multiple series, that's a separate (larger)
piece of work -- flag it and it can be added.
"""
import numpy as np
import matplotlib.pyplot as plt

from .tsplot import _style_axis, _resolve_color


def _detrend_series(x, order=1):
    """R astsa's detrend(): residuals from regressing x on a polynomial
    in the time index (order=1 is what mvspec() uses by default)."""
    t = np.arange(len(x), dtype=float)
    coefs = np.polyfit(t, x, order)
    fitted = np.polyval(coefs, t)
    return x - fitted


def _spec_taper(x, p):
    """Split-cosine-bell taper on fraction p at each end (matches R's
    stats::spec.taper)."""
    n = len(x)
    if p <= 0:
        return x.copy()
    m = int(np.floor(n * p))
    if m == 0:
        return x.copy()
    w = 0.5 * (1 - np.cos(np.pi * (np.arange(1, m + 1) - 0.5) / m))
    taper_vec = np.ones(n)
    taper_vec[:m] = w
    taper_vec[-m:] = w[::-1]
    return x * taper_vec


def _modified_daniell_kernel(spans):
    """Composite modified-Daniell smoothing kernel from a list of spans
    (matches R's kernel("modified.daniell", spans %/% 2) then convolving
    each span together, e.g. spans=[3,3,3])."""
    k = np.array([1.0])
    for span in spans:
        m = span // 2
        elem = np.full(2 * m + 1, 1.0 / (2 * m))
        elem[0] = elem[-1] = 1.0 / (4 * m)
        k = np.convolve(k, elem)
    return k / k.sum()


def _kernapply_circular(x, kernel):
    """Circular convolution smoothing, matching R's kernapply(..., circular=TRUE)."""
    n = len(x)
    m = (len(kernel) - 1) // 2
    out = np.zeros(n)
    for t in range(n):
        for j, w in enumerate(kernel):
            lag = j - m
            out[t] += w * x[(t + lag) % n]
    return out


def mvspec(x, spans=None, taper=0.0, detrend=True, demean=False,
            pad=0, fast=True, xfreq=1.0, log="n", gg=False, col=4,
            main=None, ylab="periodogram", xlab=None, xlim=None,
            plot=True, ax=None, report=True):
    """Univariate periodogram (raw or smoothed), R-astsa mvspec() style.

    Returns a dict: freq, spec, df, bandwidth, Lh, kernel (the smoothing
    weights, or None), method (string).
    """
    x = np.asarray(x, dtype=float)
    N0 = len(x)

    if demean:
        x = x - x.mean()
    elif detrend:
        x = _detrend_series(x, order=1)

    x = _spec_taper(x, taper)
    u2 = 1 - (5 / 8) * taper * 2
    u4 = 1 - (93 / 128) * taper * 2

    if pad > 0:
        x = np.concatenate([x, np.zeros(N0 * pad)])
    N = len(x)

    if fast:
        # next highly-composite length >= N (matches R's nextn() default,
        # which favors factors of 2,3,5) -- good enough: next power of 2
        # covers our two use cases (2^10 already a power of 2; sunspots
        # will get padded to the next power of 2, which nextn() would
        # also frequently land on or near)
        NewN = 1 << (N - 1).bit_length()
    else:
        NewN = N
    if NewN > N:
        x = np.concatenate([x, np.zeros(NewN - N)])
    N = len(x)

    Nspec = N // 2
    freq = np.arange(1, Nspec + 1) * xfreq / N

    xfft = np.fft.fft(x)
    pgram = (xfft * np.conj(xfft)).real / (N0 * xfreq)
    pgram[0] = 0.5 * (pgram[1] + pgram[-1])  # matches R's pgram[1,] wraparound fix

    kernel = None
    if spans is not None:
        kernel = _modified_daniell_kernel(list(np.atleast_1d(spans)))
        pgram = _kernapply_circular(pgram, kernel)
        Lh = 1 / np.sum(kernel ** 2)
    else:
        Lh = 1.0

    df = 2 * Lh
    df = df / (u4 / u2 ** 2)
    df = df * (N0 / N)
    bandwidth = Lh * xfreq / N

    spec = pgram[1:Nspec + 1] / u2

    if plot:
        created_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots()
        _style_axis(ax, gg)
        yvals = np.log(spec) if log == "y" else spec
        ax.plot(freq, yvals, color=_resolve_color(col), linewidth=1)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xlabel(xlab or ("frequency" if xfreq == 1 else f"frequency \u00d7 {xfreq}"))
        ax.set_ylabel(ylab)
        if main is not None:
            ax.set_title(main)
        if kernel is not None and report:
            print(f"Bandwidth: {bandwidth:.3f} | Degrees of Freedom: {df:.2f} | "
                   f"split taper: {100 * taper:.0f}%")
        if created_fig:
            fig.tight_layout()

    return dict(freq=freq, spec=spec, df=df, bandwidth=bandwidth, Lh=Lh,
                 kernel=kernel, method="Smoothed Periodogram" if kernel is not None
                 else "Raw Periodogram")
