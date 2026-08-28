"""
Python translation of astsa's scatter.hist().

R source: astsa/R/scatter.hist.r

A scatterplot of x vs y with marginal histograms on the top and right
edges -- a classic "joint + marginals" layout.
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter_hist(x, y, xlab=None, ylab=None, title=None, pt_size=20,
                   hist_col="0.82", pt_col=(0.1, 0.1, 0.1, 0.25),
                   figsize=(6, 6)):
    """Scatterplot of x vs y with marginal histograms (top: x, right: y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                            wspace=0.05, hspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(x, y, s=pt_size, color=pt_col)
    ax_main.set_xlabel(xlab or "x")
    ax_main.set_ylabel(ylab or "y")

    ax_top.hist(x, color=hist_col, edgecolor="white")
    ax_top.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax_top.spines[:].set_visible(False)

    ax_right.hist(y, color=hist_col, edgecolor="white", orientation="horizontal")
    ax_right.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax_right.spines[:].set_visible(False)

    if title:
        fig.suptitle(title)

    return fig, (ax_main, ax_top, ax_right)
