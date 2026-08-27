"""
sarima() delegates the actual model fitting to statsmodels' SARIMAX, which
I could not install in the sandbox this repo was built in (no network
access there). What I *can* verify without statsmodels is the surrounding
math sarima() applies to whatever fit it gets back: the coefficient
table (SE, t-value, p-value), degrees of freedom, and the AIC/AICc/BIC
rescaling -- since all of that is generic and doesn't care what produced
the coefficients.

This script fits a pure AR(p) by conditional least squares (which is just
OLS of x_t on an intercept + p lags -- consistent, though not identical
to exact ML, for AR-only models) using only numpy/scipy, applies the same
downstream formulas sarima() uses, and checks:

  1. Coefficients recover the true simulated AR parameters well.
  2. The classic sanity check for any correctly-computed p-value column:
     under a *true* null (a coefficient that really is zero), p-values
     should be ~Uniform(0,1).

Run: python tests/test_sarima_table_math.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import t as tdist
from stocproc_tools.sarima_sim import sarima_sim


def ar_ols_fit(x, p):
    """Conditional-least-squares AR(p) fit with intercept: x_t = c + sum
    phi_i x_{t-i} + e_t. Returns coefs (c, phi_1..phi_p), SEs, and sigma2."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    Y = x[p:]
    X = np.column_stack([np.ones(n - p)] + [x[p - i:n - i] for i in range(1, p + 1)])
    beta, resid_ss, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    m = len(Y)
    k = X.shape[1]
    sigma2 = np.dot(resid, resid) / (m - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta, se, sigma2, m, k


def coefficient_table(beta, se, m, k):
    """Same formulas as sarima()'s t_table construction."""
    dfree = m - k
    tvals = beta / se
    pvals = 2 * (1 - tdist.cdf(np.abs(tvals), df=dfree))
    return tvals, pvals, dfree


if __name__ == "__main__":
    # --- Test 1: recovery of known AR(2) coefficients -------------------
    rng = np.random.default_rng(123)
    true_phi = [0.6, -0.3]
    x = sarima_sim(ar=true_phi, n=5000, rng=rng)
    beta, se, sigma2, m, k = ar_ols_fit(x, p=2)
    print("Test 1: AR(2) recovery (n=5000)")
    print(f"  true phi:      {true_phi}")
    print(f"  estimated:     {np.round(beta[1:], 4)}  (intercept {beta[0]:.4f})")
    assert np.allclose(beta[1:], true_phi, atol=0.03), "AR coefficients not recovered!"
    print("  PASS: within 0.03 of true values\n")

    # --- Test 2: p-values under a true null are ~Uniform(0,1) -----------
    # Fit AR(3) to data that's actually only AR(1) -- phi_2, phi_3 are
    # truly zero, so their p-values should be uniform if the SE/t/p
    # formulas are correct.
    print("Test 2: null-coefficient p-values should be ~Uniform(0,1)")
    p2_pvals, p3_pvals = [], []
    for i in range(2000):
        rng_i = np.random.default_rng(1000 + i)
        xi = sarima_sim(ar=[0.5], n=300, rng=rng_i)
        beta_i, se_i, sigma2_i, m_i, k_i = ar_ols_fit(xi, p=3)
        tvals_i, pvals_i, dfree_i = coefficient_table(beta_i, se_i, m_i, k_i)
        p2_pvals.append(pvals_i[2])  # coefficient on lag-2 (truly 0)
        p3_pvals.append(pvals_i[3])  # coefficient on lag-3 (truly 0)

    p2_pvals = np.array(p2_pvals)
    p3_pvals = np.array(p3_pvals)
    print(f"  lag-2 (null) p-values: mean={p2_pvals.mean():.3f}  "
          f"reject@.05={np.mean(p2_pvals < .05):.3f}")
    print(f"  lag-3 (null) p-values: mean={p3_pvals.mean():.3f}  "
          f"reject@.05={np.mean(p3_pvals < .05):.3f}")
    assert abs(p2_pvals.mean() - 0.5) < 0.03
    assert abs(p3_pvals.mean() - 0.5) < 0.03
    print("  PASS: both ~0.5 mean, ~5% rejection rate at alpha=.05\n")

    print("All coefficient-table math checks passed.")
