"""
Chapter 5 - spectral analysis examples: mvspec() and arma.spec().

R:
  P = mvspec(rnorm(2^10), col=8, main=NA, ylab='periodogram', gg=TRUE)
  mvspec(sunspots, spans=c(3,3,3), col=5, gg=TRUE, taper=.1, xlim=c(0,.5))
  arma.spec(ar = c(1,-.9), main=NA, gg=TRUE, col=6, lwd=2)
  arma.spec(ma=-.9, main=NA, gg=TRUE, col=4, lwd=2)

Run: python ch5/ch5_spectral.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from stocproc_tools.mvspec import mvspec
from stocproc_tools.arma_spec import arma_spec

rng = np.random.default_rng(3)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

mvspec(rng.standard_normal(2**10), col=8, ylab="periodogram", gg=True, ax=axes[0, 0])
axes[0, 0].set_title("White Noise Periodogram")

# sunspots not bundled here (base-R dataset, see ch5_sarima_sunspots.py note);
# demonstrate the smoothed-periodogram path on a stand-in AR(2) series instead
from stocproc_tools.sarima_sim import sarima_sim
x = sarima_sim(ar=[1, -0.9], n=1000, rng=np.random.default_rng(5))
mvspec(x, spans=[3, 3, 3], col=5, gg=True, taper=0.1, xlim=(0, 0.5), ax=axes[0, 1])
axes[0, 1].set_title("Smoothed Periodogram (AR(2) stand-in for sunspots)")

arma_spec(ar=[1, -0.9], gg=True, col=6, ax=axes[1, 0])
arma_spec(ma=-0.9, gg=True, col=4, ax=axes[1, 1])

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "ch5_spectral_examples.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
