"""
Python translation of astsa's QQnorm().

R source: astsa/R/QQnorm.R

A normal Q-Q plot with a fitted line (through the 1st/3rd quartiles, same
as R's qqline default) and an optional pointwise confidence band computed
from the standard order-statistic SE formula:
    SE(z) = (b / dnorm(z)) * sqrt(p*(1-p)/n)
"""
import numpy as np
from scipy.stats import norm


def _ppoints(n):
    """Matches R's ppoints(n): (i - a)/(n + 1 - 2a), a=3/8 if n<=10 else 1/2."""
    a = 3 / 8 if n <= 10 else 0.5
    i = np.arange(1, n + 1)
    return (i - a) / (n + 1 - 2 * a)


def qq_norm(x, ax, col=("#1874cd", "#cd1874"), ylab="Sample Quantiles",
            xlab="Theoretical Quantiles", main="Normal Q-Q Plot",
            ylim=None, ci=True, qqlwd=1):
    """Draw a normal Q-Q plot of `x` onto matplotlib axes `ax`.

    `ci` mirrors R's flexible convention: True -> 99.99% band (R's
    default), a fraction in (0,1) -> that confidence level, a number in
    [1,100) -> that percentage, False/0 -> no band.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    xord = np.sort(x)
    PP = _ppoints(n)
    z = norm.ppf(PP)

    # quartiles via R's type=7 (linear interpolation) -- numpy's default
    # 'linear' method matches type 7
    y = np.quantile(xord, [0.25, 0.75], method="linear")
    xq = norm.ppf([0.25, 0.75])
    b = np.diff(y)[0] / np.diff(xq)[0]
    a = y[0] - b * xq[0]
    qqfit = a + b * z

    ax.scatter(z, xord, color=col[0], s=18, zorder=3)
    if ylim is not None:
        ax.set_ylim(ylim)

    if ci:
        # replicate R's ci-width bookkeeping exactly
        ci_val = 99.99 if ci is True else (100 * ci if ci < 1 else ci)
        if ci_val >= 100:
            ci_val = 99.99
        half_pct = (100 - ci_val) / 2 if ci_val > 50 else ci_val / 2
        wde = norm.ppf(1 - half_pct / 100)

        SE = (b / norm.pdf(z)) * np.sqrt(PP * (1 - PP) / n)
        U = qqfit + wde * SE
        L = qqfit - wde * SE
        if ylim is None:
            ax.set_ylim(min(xord[0], L[0]), max(xord[-1], U[-1]))
        zz = z.copy()
        zz[0] -= 0.1
        zz[-1] += 0.1
        ax.fill_between(zz, L, U, color="0.5", alpha=0.2, linewidth=0, zorder=1)

    zz_line = np.array([z.min() - 0.5, z.max() + 0.5])
    ax.plot(zz_line, a + b * zz_line, color=col[1], linewidth=qqlwd, zorder=2)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(main)
    return ax
