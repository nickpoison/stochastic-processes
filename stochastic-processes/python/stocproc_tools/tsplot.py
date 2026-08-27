"""
Python translation of astsa's tsplot().

R source: astsa/R/tsplot.R

This covers the two call patterns that actually appear in the book:

  1. A single series:      tsplot(x, col=4, gg=True, ylab=..., main=...)
  2. Overlaid ("spaghetti") series:
                            tsplot(cbind(x1, x2), spag=True, col=[...], addLegend=True)

R's tsplot() also supports laying multiple series out in separate side-by-side
panels (nser>1, spaghetti=False) via ncolm/byrow — that path isn't used
anywhere in this book, so it's not translated here. If you need it, use
plt.subplots() directly; matplotlib makes that easy without a helper.

Differences from R on purpose (idiomatic Python, not a 1:1 port):
  - x is any array-like or a pandas Series/DataFrame (its index is used as
    the time axis if present, matching what timex()/dates give you in R).
  - "gg" style approximates R's ggplot2-like panel (light gray background,
    white gridlines) using matplotlib, not an exact pixel match.
  - Grid()'s exact minor-tick math is not reproduced; we use matplotlib's
    AutoMinorLocator, which looks equivalent but the tick positions can
    differ slightly from R's.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

_R_COL = {1: "black", 2: "#F6483C", 3: "#00BA38", 4: "#1874cd",
          5: "#0D9AC0", 6: "#cd1874", 7: "#CD7118", 8: "0.62"}


def _resolve_color(c):
    """Map R-style integer color codes (1=black, 4=blue, ...) to hex, but
    pass through anything that's already a real color spec."""
    if isinstance(c, (int, np.integer)):
        return _R_COL.get(int(c) % 8 or 8, "black")
    return c


def _as_xy(x, y):
    """Mirror R's tsplot(x) vs tsplot(x, y) — if y is None, x supplies both
    the values and (if it's a pandas object) the time axis."""
    if y is not None:
        return np.asarray(x), np.asarray(y)
    if hasattr(x, "index"):  # pandas Series
        return np.asarray(x.index), np.asarray(x.values)
    x = np.asarray(x)
    return np.arange(len(x)), x


def _style_axis(ax, gg):
    if gg:
        ax.set_facecolor("0.92")
        ax.grid(True, which="major", color="white", linewidth=1)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="minor", color="white", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(2)
        ax.tick_params(colors="0.3")
    else:
        ax.grid(True, color="0.9", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("gray")


def tsplot(x, y=None, main=None, ylab=None, xlab="Time", type="l",
           col=4, gg=False, spaghetti=False, pch=None, lty="-", lwd=1,
           addLegend=False, legend=None, location="upper right",
           ylim=None, xlim=None, ax=None, **kwargs):
    """Plot one series, or several overlaid ("spaghetti") series.

    Parameters mirror the R astsa::tsplot() names where practical:
      x, y      : data (see _as_xy)-- for spaghetti, x should be 2D
                  (columns = series) or a DataFrame with multiple columns
      col       : R-style int (1-8) or any matplotlib color; list for spaghetti
      gg        : bool, R's `gg=TRUE` light-panel style
      spaghetti : bool, overlay multiple series on one axis (R's `spag=`)
      type      : 'l' (line), 'o' (line+points), 'p' (points), 's' (steps)
      addLegend, legend, location : legend controls (location uses
                  matplotlib names like 'upper right', not R's 'topright')
    """
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots()

    _style_axis(ax, gg)

    type_map = {"l": "-", "o": "-", "p": "None", "s": "-"}
    marker_map = {"o": "o", "p": "o"}

    if spaghetti:
        # x expected to have multiple columns (DataFrame, 2D array, or
        # dict-like of series) -- overlay each as its own line
        if hasattr(x, "columns"):  # DataFrame
            names = list(x.columns)
            series = [x[c] for c in names]
            xaxis = x.index
        else:
            arr = np.asarray(x)
            names = [f"Series {i+1}" for i in range(arr.shape[1])]
            series = [arr[:, i] for i in range(arr.shape[1])]
            xaxis = np.arange(arr.shape[0])

        cols = col if isinstance(col, (list, tuple, np.ndarray)) else [col] * len(series)
        cols = [_resolve_color(c) for c in cols]

        for s, c, name in zip(series, cols, names):
            drawstyle = "steps-post" if type == "s" else "default"
            ax.plot(xaxis, s, color=c, linestyle=type_map.get(type, "-"),
                     marker=marker_map.get(type), drawstyle=drawstyle,
                     linewidth=lwd, label=name, **kwargs)

        if addLegend:
            ax.legend(labels=legend or names, loc=location, frameon=True,
                      framealpha=0.8)

    else:
        xv, yv = _as_xy(x, y)
        c = _resolve_color(col)
        drawstyle = "steps-post" if type == "s" else "default"
        ax.plot(xv, yv, color=c, linestyle=type_map.get(type, "-"),
                 marker=marker_map.get(type), drawstyle=drawstyle,
                 linewidth=lwd, **kwargs)

    if main:
        ax.set_title(main, fontsize=12, fontweight="bold" if not gg else "normal")
    if ylab is not None:
        ax.set_ylabel(ylab)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    if created_fig:
        fig.tight_layout()
    return ax
