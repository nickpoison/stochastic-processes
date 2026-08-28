"""
Chapter 1 - Preliminaries: Python translations of the R examples.

Mirrors the R code in book Section 1.6 as closely as practical.
Run directly to reproduce the plots: `python ch1/ch1_examples.py`
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from stocproc_tools.tsplot import tsplot, _style_axis

# ---- Gambler's Ruin -------------------------------------------------
# R:
#   set.seed(111)
#   u = sample(c(-1,1), 20, replace=TRUE)
#   x = ts(c(10, 10+cumsum(u)), start=0)
#   tsplot(x, type='o', xlab='n', gg=TRUE, pch=19, ylab=bquote(X[n]), col=4)
rng = np.random.default_rng(111)
u = rng.choice([-1, 1], size=20)
x = np.concatenate([[10], 10 + np.cumsum(u)])

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

tsplot(x, type="o", xlab="n", gg=True, ylab="X_n", col=4, ax=axes[0, 0])
axes[0, 0].set_title("Gambler's Ruin")

# ---- Counting Process (Poisson-ish arrivals) ------------------------
# R:
#   t = c(0, sort(runif(50, 0, 20)))  (schematic -- book uses a specific
#   arrival-time construction; here we just illustrate the counting step)
rng2 = np.random.default_rng(2)
arrivals = np.sort(rng2.uniform(0, 20, 15))
t_grid = np.concatenate([[0], arrivals])
N = np.arange(len(t_grid))
tsplot(t_grid, N, type="s", gg=True, ylab="N_t", pch=19, col=4, ax=axes[0, 1])
axes[0, 1].set_title("Counting Process")

# ---- Time series-ish sample path ------------------------------------
rng3 = np.random.default_rng(3)
w = rng3.standard_normal(200)
tsplot(np.cumsum(w), col=4, gg=True, ylab="X_n", ax=axes[1, 0])
axes[1, 0].set_title("Random Walk")

# ---- 2D Brownian-motion-ish path -------------------------------------
rng4 = np.random.default_rng(4)
steps = rng4.standard_normal((500, 2)) * 0.1
path = np.cumsum(steps, axis=0)
_style_axis(axes[1, 1], gg=True)
axes[1, 1].plot(path[:, 0], path[:, 1], color="#1874cd", linewidth=1)
axes[1, 1].set_title("2D Brownian Motion")
axes[1, 1].set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "ch1_four_examples.png"), dpi=120)
print("saved ch1/ch1_four_examples.png")
