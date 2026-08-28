"""
Python translation of astsa's astsa.col().

R source: astsa/R/astsa.col.r

    astsa.col <- function(col=1, alpha=1, wheel=FALSE, pie=FALSE, num,
                           sat=NULL, val=NULL, ...)

Only the two usages that appear in the book are supported:
  - astsa.col(4, .7)                       -> simple palette color + alpha
  - astsa.col(4, wheel=TRUE, num=17, v=.7) -> a wheel of `num` hues

Note: R's astsa.col recycles col+alpha over a vector of color codes
(e.g. astsa.col(2*2:3, .7) for two colors at once); this Python version
does the same when `col` is a list/array.
"""
import colorsys
import numpy as np
import matplotlib.colors as mcolors

# same 8-color palette as the R version (index 0 unused: R uses 1-based
# mod-8 indexing, col=1 -> "black", col=2 -> "#F6483C", etc.)
_PALETTE = ["black", "#F6483C", "#00BA38", "#1874cd",
            "#0D9AC0", "#cd1874", "#CD7118", "0.62"]


def _with_alpha(color, alpha):
    r, g, b = mcolors.to_rgb(color)
    return (r, g, b, alpha)


def astsa_col(col=1, alpha=1.0, wheel=False, num=None, sat=None, val=None):
    """Return one color (or a list of colors) with astsa's palette + alpha.

    Parameters mirror the R function's `col`, `alpha`, `wheel`, `num`,
    `sat`, `val` (R's `pie=` plotting side-effect is dropped here — just
    call matplotlib's plt.pie yourself with the returned colors if needed).
    """
    if not wheel:
        cols = np.atleast_1d(col)
        picked = [_PALETTE[(int(c) - 1) % 8] for c in cols]
        out = [_with_alpha(c, alpha) for c in picked]
        return out[0] if np.isscalar(col) else out

    # wheel=TRUE: spin `num` hues starting from the hue of `col`
    if num is None:
        raise ValueError("num= is required when wheel=True")
    base_rgb = mcolors.to_rgb(_PALETTE[(int(col) - 1) % 8] if np.isscalar(col) else col)
    h, s, v = colorsys.rgb_to_hsv(*base_rgb)
    s = sat if sat is not None else s
    v = val if val is not None else v
    hues = [(h + i / num) % 1.0 for i in range(num)]
    return [colorsys.hsv_to_rgb(hh, s, v) + (alpha,) for hh in hues]
