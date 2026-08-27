"""
Python translation of astsa's adjQstat().

R source: astsa/R/adjQstat.R

Implements the "adjusted Box-Pierce" statistic from Kan & Wang (2010),
which corrects the classic Ljung-Box/Box-Pierce Q-statistic using the
*exact* finite-sample moments of the sample autocorrelations rho_hat(k),
rather than their large-n asymptotic approximation. This matters most
at small n / large fitdf, which is exactly the regime residual
diagnostics get used in.

This is a direct, careful transcription of the R formulas (Kan & Wang
2010, eqs. 39/41/45) -- there's no way to "derive" these from first
principles in a docstring, so correctness here rests on transcription
being exact plus the empirical check in tests/test_diagnostics.py
(simulating white noise and checking p-values come out ~Uniform(0,1)).
"""
import numpy as np
from scipy.stats import chi2


def _pp(a):
    """Positive-part operator a+ = max(a, 0)."""
    return np.maximum(a, 0)


def _Erho2(k, n):
    """Exact E[rho_hat(k)^2], Kan & Wang eq. (39)."""
    return ((n - k) * (n**2 + n - 3 * k) / (n**2 * (n**2 - 1))
             - 2 * _pp(n - 2 * k) / (n * (n**2 - 1)))


def _Erho4(k, n):
    """Exact E[rho_hat(k)^4], Kan & Wang eq. (41)."""
    term1 = (3 * (n - k) * (n**3 * (n + 1) * (n + 3)
                             - n**2 * (n**2 + 8 * n + 21) * k
                             + 3 * n * (2 * n + 15) * k**2
                             - 35 * k**3)
             / (n**4 * (n**2 - 1) * (n + 3) * (n + 5)))

    term2 = (-12 * (2 * n**2 - n * (n + 6) * k + 15 * k**2) * _pp(n - 2 * k)
              / (n**3 * (n**2 - 1) * (n + 3) * (n + 5)))

    term3 = (24 * (_pp(n - 3 * k) ** 2 - n * _pp(n - 4 * k))
              / (n**2 * (n**2 - 1) * (n + 3) * (n + 5)))

    return term1 + term2 + term3


def _Erho2rho2(j, k, n):
    """Exact E[rho_hat(j)^2 * rho_hat(k)^2] for j < k, Kan & Wang eq. (45)."""
    assert j < k
    d2jk = 1.0 if 2 * j == k else 0.0

    A = (-(n - k) * (105 * j**2 * k
                      - (75 * j**2 + 60 * j * k) * n
                      + (36 * j - 3 * j**2 + 15 * k - 3 * j * k) * n**2
                      + (5 * j + 3 * k - 5) * n**3
                      + (j - 6) * n**4
                      - n**5)
         / (n**4 * (n**2 - 1) * (n + 3) * (n + 5)))

    B = (-2 * ((15 * j**2 + 9 * n**2 - j * n**2 + n**3) * _pp(n - 2 * k)
                + (15 * k**2 - 24 * k * n + 9 * n**2 - k * n**2 + n**3) * _pp(n - 2 * j)
                + (60 * j * k - 24 * j * n - 4 * n**3) * _pp(n - j - k))
         / (n**3 * (n**2 - 1) * (n + 3) * (n + 5)))

    C = (2 * (6 * (n - 3 * k) * _pp(n - 2 * j - k)
               + 6 * (n - 3 * j) * _pp(n - j - 2 * k)
               + 2 * (n - 3 * j) * _pp(n + j - 2 * k)
               - 12 * n * _pp(n - 2 * j - 2 * k)
               + 4 * (2 * n - 3 * k) * _pp(n - max(2 * j, k))
               - 2 * n * _pp(n + 2 * j - 2 * max(2 * j, k))
               - 2 * n * (n - k) ** 2 * d2jk)
         / (n**2 * (n**2 - 1) * (n + 3) * (n + 5)))

    return A + B + C


def _QBP_moments(n, m):
    """Exact E[Q_BP] and Var[Q_BP] for series length n, m lags."""
    n = float(n)
    ks = np.arange(1, m + 1)
    S2 = sum(_Erho2(k, n) for k in ks)
    S4 = sum(_Erho4(k, n) for k in ks)

    Scross = 0.0
    for j in range(1, m):
        for k in range(j + 1, m + 1):
            Scross += _Erho2rho2(j, k, n)

    EQ = n * S2
    EQ2 = n**2 * (S4 + 2 * Scross)
    VarQ = EQ2 - EQ**2
    return EQ, VarQ


def acf_raw(x, lag_max):
    """R-style acf()$acf: lags 0..lag_max, mean-centered, normalized by
    lag-0 sample covariance (population, i.e. divide by n not n-1)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    return np.array([1.0 if k == 0 else np.dot(x[:n - k], x[k:]) / n / c0
                      for k in range(lag_max + 1)])


def adj_qstat(x, lag, fitdf=0):
    """Adjusted Box-Pierce statistic (Kan & Wang, 2010, exact moments).

    Returns dict with statistic, df, p_value, QBP, E_QBP, Var_QBP --
    mirrors R's adjQstat() list output (renamed to snake_case).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    m = lag
    if m <= fitdf:
        raise ValueError("lag must be greater than fitdf")

    rho = acf_raw(x, m)[1:]
    QBP = n * np.sum(rho**2)

    EQ, VarQ = _QBP_moments(n, m)

    df = m - fitdf
    Qadj = df + np.sqrt(2 * df / VarQ) * (QBP - EQ)
    pval = 1 - chi2.cdf(Qadj, df=df)

    return {
        "statistic": Qadj, "df": df, "p_value": pval,
        "method": "Adjusted Box-Pierce (Kan & Wang, 2010, exact moments)",
        "QBP": QBP, "E_QBP": EQ, "Var_QBP": VarQ,
    }
