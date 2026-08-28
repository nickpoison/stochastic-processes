"""
Python translation of astsa's ts.diag().

R source: astsa/R/ts.diag.R

The standard 4-panel residual diagnostic plot used throughout the book
(and inside sarima()'s details=TRUE block, which calls this same logic):
  1. standardized residuals over time
  2. ACF of the residuals
  3. Normal Q-Q plot of the standardized residuals
  4. p-values of the adjusted (Kan & Wang) Box-Pierce Q-statistic by lag

This is deliberately split out from sarima() (see astsa's sarima.R,
which has this same code duplicated inline) so it can be reused on
residuals from *any* model -- not just ones fit with astsa's own
sarima() -- e.g. statsmodels SARIMAX residuals.
"""
import numpy as np
import matplotlib.pyplot as plt

from .tsplot import tsplot
from .acf import acf1
from .qqnorm import qq_norm
from .adj_qstat import adj_qstat


def ts_diag(resids, col=4, nlag=20, qstat=True, fitdf=0, model_label=None):
    """Plot the 4-panel residual diagnostics.

    Parameters
    ----------
    resids : array-like
        Residuals from a fitted model (e.g. `statsmodels` SARIMAXResults.resid).
    nlag : int
        Max lag for the Q-statistic panel (bumped up to fitdf+8 if too small,
        matching R).
    qstat : bool
        If False, drop the p-value panel and use a 3-panel layout instead.
    fitdf : int
        Number of fitted parameters to subtract from Q-stat degrees of
        freedom (R's `p+q+P+Q` for a sarima() fit; pass 0 for raw residuals).
    model_label : str, optional
        Small annotation on the standardized-residuals panel, e.g. "(1,0,1)".
    """
    resids = np.asarray(resids, dtype=float)
    if nlag < fitdf + 8:
        nlag = fitdf + 8

    rs = resids - np.nanmean(resids)
    stdres = rs / np.nanstd(rs, ddof=0)

    if qstat:
        fig = plt.figure(figsize=(9, 7))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
    else:
        fig = plt.figure(figsize=(9, 5))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)

    # [1] Standardized residuals
    plot_type = "o" if np.any(np.isnan(resids)) else "l"
    tsplot(stdres, main="Standardized Residuals", ylab="", col=col,
           type=plot_type, ax=ax1)
    if model_label:
        ax1.text(0.01, 0.98, f"Model: {model_label}", transform=ax1.transAxes,
                  fontsize=8.5, ha="left", va="top",
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0))

    # [2] ACF of residuals
    acf1_vals = acf1(rs, plot=False)
    max_lag = len(acf1_vals)
    n = len(rs)
    U = -1 / n + 2 / np.sqrt(n)
    L = -1 / n - 2 / np.sqrt(n)
    LAG = np.arange(1, max_lag + 1)
    ax2.vlines(LAG, 0, acf1_vals, color="#1874cd")
    ax2.axhline(0, color="0.5", linewidth=1)
    ax2.axhline(L, color="#1874cd", linestyle="--", linewidth=1)
    ax2.axhline(U, color="#1874cd", linestyle="--", linewidth=1)
    ax2.set_title("ACF of Residuals")
    ax2.set_xlabel("LAG")
    ax2.set_ylabel("ACF")
    ax2.grid(True, color="0.9")

    # [3] Normal Q-Q plot
    qq_norm(stdres, ax3, main="Normal Q-Q Plot of Std Residuals")

    if qstat:
        # [4] adjusted Q-statistic p-values
        lags = np.arange(fitdf + 1, nlag + 1)
        pvals = np.array([adj_qstat(rs, lag=int(lag), fitdf=fitdf)["p_value"]
                            for lag in lags])
        ax4.scatter(lags, pvals, color="#1874cd", s=20)
        ax4.axhline(0.05, color="#1874cd", linestyle="--")
        ax4.set_ylim(-0.14, 1)
        ax4.set_xlabel("LAG (H)")
        ax4.set_ylabel("p value")
        ax4.set_title("p values for Adjusted Q-Statistic")
        ax4.grid(True, color="0.9")

    fig.tight_layout()
    return fig
