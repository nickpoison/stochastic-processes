"""
Python translation of astsa's sarima() -- fitting (not simulation).

R source: astsa/R/sarima.R

Requires `statsmodels` (not needed by anything else in this repo). Fits a
(seasonal) ARIMA model via statsmodels' state-space SARIMAX -- the same
Kalman-filter MLE approach R's arima() uses -- then reproduces the
coefficient table / AIC-AICc-BIC printout astsa's sarima() gives you.
Residual diagnostics are handled separately by ts_diag() (see ts_diag.py)
rather than duplicated here, since diagnostics apply to any fitted model.

NOTE ON VERIFICATION: I don't have network access in the sandbox this was
built in, so I could not install statsmodels there to test this function
end-to-end against real output. The surrounding math (coefficient table,
degrees of freedom, the AICc formula) was validated separately using a
self-contained least-squares AR(p) fit -- see
tests/test_sarima_table_math.py -- which exercises the exact same
downstream formulas with a fit I *could* run and check against known
simulated coefficients. Please sanity-check the first real run of this
against R's sarima(sunspots, p=16) output if that matters to you.

Trend/mean handling mirrors R's arima()-via-sarima() logic exactly:
  - d==0 and D==0            -> include a mean (unless no_constant),
                                 reported as "xmean" (matches R's sarima())
  - exactly one of d,D == 1  -> include a linear drift term (unless
                                 no_constant), reported as "constant"
                                 (matches R's sarima())
  - otherwise                -> no automatic mean/drift term
  - if xreg is supplied, no automatic mean/drift is added (matches R)

Note on "xmean" vs statsmodels' native "intercept": statsmodels reports
the raw regression-style intercept c (where mean = c/(1 - sum(ar coefs))),
not the process mean itself. This module converts that to the actual
mean (with a proper delta-method SE using the full parameter covariance,
not just the intercept's own SE) and labels it "xmean" to match R's
sarima(), for the no-differencing case. For the drift case (trend="t"),
statsmodels' "drift" parameter already matches R's convention directly,
so it's just relabeled "constant" -- same names R's own sarima() uses,
which vary between "xmean" and "constant" depending on the branch.
"""
import numpy as np
import pandas as pd
from scipy.stats import t as tdist


def sarima(xdata, p=0, d=0, q=0, P=0, D=0, Q=0, S=None,
           xreg=None, no_constant=False, details=True, col=4, qlag=None,
           model_label=True):
    """Fit a (seasonal) ARIMA model, R-astsa style.

    Returns a dict: fit (the statsmodels results object), sigma2,
    degrees_of_freedom, t_table (pandas DataFrame), ICs (dict of AIC/
    AICc/BIC, each divided by n -- matches R's convention, not
    statsmodels' raw .aic/.bic).

    If details=True, also prints the coefficient table + IC line and
    draws the ts_diag() 4-panel residual plot.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        raise ImportError(
            "sarima() needs statsmodels: pip install statsmodels"
        ) from e

    xdata = np.asarray(xdata, dtype=float)
    n = len(xdata)

    seasonal_order = (P, D, Q, S) if S else (0, 0, 0, 0)

    if xreg is None:
        if d == 0 and D == 0:
            trend = "n" if no_constant else "c"
        elif (d == 1) != (D == 1):  # xor
            trend = "n" if no_constant else "t"
        else:
            trend = "n"
    else:
        trend = "n"  # matches R: supplying xreg skips the automatic mean/drift

    mod = SARIMAX(xdata, order=(p, d, q), seasonal_order=seasonal_order,
                   trend=trend, exog=xreg, enforce_stationarity=False,
                   enforce_invertibility=False)
    res = mod.fit(disp=False)

    param_names = list(res.param_names)
    has_sigma2 = "sigma2" in param_names
    coef_idx = [i for i, nm in enumerate(param_names) if nm != "sigma2"]

    coefs = np.asarray(res.params)[coef_idx].copy()
    ses = np.asarray(res.bse)[coef_idx].copy()
    names = [param_names[i] for i in coef_idx]

    # --- convert statsmodels' regression-style intercept into R's "xmean" ---
    # R's arima() reports the actual process mean when a constant is
    # included and there's no differencing (labeled "xmean"); statsmodels'
    # SARIMAX instead reports the raw intercept of a "regression with ARMA
    # errors" (y_t = intercept + sum(ar) * y_{t-lag} + ...), where
    # mean = intercept / (1 - sum(ar_coefs)). Without this conversion the
    # printed "intercept" can look very different from the sample mean --
    # same underlying model, different reporting convention. Only applies
    # when trend == "c" (mean case); the trend == "t" (drift, from an
    # explicit xreg time-index column) case matches R's convention already
    # without any adjustment.
    if trend == "c" and "intercept" in names:
        i_idx = names.index("intercept")
        ar_idx = [j for j, nm in enumerate(names) if nm.startswith("ar.")]
        if ar_idx:
            ar_sum = coefs[ar_idx].sum()
            denom = 1 - ar_sum
            mean_est = coefs[i_idx] / denom

            full_cov = np.asarray(res.cov_params())
            full_idx = [coef_idx[i_idx]] + [coef_idx[j] for j in ar_idx]
            sub_cov = full_cov[np.ix_(full_idx, full_idx)]

            grad = np.zeros(len(full_idx))
            grad[0] = 1 / denom                                # d(mean)/d(intercept)
            grad[1:] = coefs[i_idx] / denom**2                  # d(mean)/d(ar_j)

            var_mean = grad @ sub_cov @ grad
            se_mean = np.sqrt(var_mean) if var_mean > 0 else np.nan

            coefs[i_idx] = mean_est
            ses[i_idx] = se_mean
            names[i_idx] = "xmean"

    # --- trend == "t" (drift) case: R's sarima() calls this "constant"
    # (it's a literal xreg regression coefficient there, same convention
    # statsmodels already uses -- no math conversion needed, just the label
    # to match R's naming). statsmodels names this parameter "drift".
    if trend == "t" and "drift" in names:
        names[names.index("drift")] = "constant"

    k = len(coefs)
    dfree = n - k
    tvals = coefs / ses
    pvals = 2 * (1 - tdist.cdf(np.abs(tvals), df=dfree))

    t_table = pd.DataFrame(
        {"Estimate": coefs, "SE": ses, "t.value": tvals, "p.value": pvals},
        index=names,
    ).round(4)

    sigma2 = res.params[param_names.index("sigma2")] if has_sigma2 else np.var(res.resid)

    AIC = res.aic / n
    BIC = res.bic / n
    AICc = (n * AIC + (2 * k**2 + 2 * k) / (n - k - 1)) / n

    if details:
        print("<><><><><><><><><><><><><><>\n")
        if k > 0:
            print("Coefficients:")
            print(t_table)
            print()
        print(f"sigma^2 estimated as {sigma2:.6g} on {dfree} degrees of freedom\n")
        print(f"AIC = {AIC:.4f}   AICc = {AICc:.4f}   BIC = {BIC:.4f}\n")

        from .ts_diag import ts_diag
        model_str = (f"({p},{d},{q})" if not S
                      else f"({p},{d},{q}) x ({P},{D},{Q})[{S}]")
        default_nlag = 20 if (S or 0) < 7 else min(3 * (S or 1), 52)
        nlag = qlag if qlag is not None else default_nlag
        ts_diag(res.resid, fitdf=p + q + P + Q, col=col, nlag=nlag,
                model_label=model_str if model_label else None)

    return {
        "fit": res,
        "sigma2": sigma2,
        "degrees_of_freedom": dfree,
        "t_table": t_table,
        "ICs": {"AIC": AIC, "AICc": AICc, "BIC": BIC},
    }
