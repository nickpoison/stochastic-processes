"""
Chapter 5 - fitting an AR(16) to the sunspots data.

R:
  sarima(sunspots, p=16)

Needs statsmodels (pip install statsmodels) -- this is the only function
in the package that does.

NOTE: this script wasn't run end-to-end against real R output when
built -- see the note at the top of stocproc_tools/sarima.py. Please
compare your first run's coefficient table / AIC-AICc-BIC line against
the book's printed output, and let us know if anything looks off.

Run: python ch5/ch5_sarima_sunspots.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

try:
    import statsmodels  # noqa: F401
except ImportError:
    print("This example needs statsmodels: pip install statsmodels")
    sys.exit(1)

from stocproc_tools.sarima import sarima

# The book uses R's built-in `sunspots` dataset (monthly sunspot numbers,
# 1749-1983). That's a base-R dataset, not part of astsa, so it isn't
# bundled in data/ here -- get it from wherever you'd normally source
# built-in R datasets, or substitute another series to try this out, e.g.:
#   from stocproc_tools import load
#   x = load("EQcount").values
#
# x = ...  # your sunspots series as a 1D array

# out = sarima(x, p=16)
# out["fit"].summary()  # full statsmodels summary, if you want more detail

print(__doc__)
