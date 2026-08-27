"""
Chapter 5 - Second Order Processes: sarima_sim() examples.

R:
  tsplot(sarima.sim(ma=-.9, n=200), col=4, gg=TRUE, main='Moving Average')
  tsplot(sarima.sim(ar = c(1,-.9), n=200), col=6, gg=TRUE, main='Autoregression')

Run: python ch5/ch5_sarima_sim.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from stocproc_tools.sarima_sim import sarima_sim
from stocproc_tools.tsplot import tsplot

rng_ma = np.random.default_rng(42)
rng_ar = np.random.default_rng(43)

ma_series = sarima_sim(ma=-0.9, n=200, rng=rng_ma)
ar_series = sarima_sim(ar=[1, -0.9], n=200, rng=rng_ar)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
tsplot(ma_series, col=4, gg=True, main="Moving Average", ax=axes[0])
tsplot(ar_series, col=6, gg=True, main="Autoregression", ax=axes[1])
fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "ch5_ma_ar_examples.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
