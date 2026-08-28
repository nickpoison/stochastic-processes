"""
Python translation of astsa's sarima.sim().

R source: astsa/R/sarima.sim.R (plus its helpers .arima_sim, .red_check)

Simulates from a (S)ARIMA(p,d,q)x(P,D,Q)_S model:
    phi(B) Phi(B^S) (1-B)^d (1-B^S)^D x_t = theta(B) Theta(B^S) w_t

The book only ever calls this with plain, non-seasonal AR or MA (no d, no
seasonal terms) -- e.g. sarima.sim(ma=-.9, n=200) and
sarima.sim(ar=c(1,-.9), n=200) -- so that path is the one most carefully
tested here. The seasonal/differencing path is translated for
completeness (mirroring the R source function-for-function) but is
lightly tested since it's not exercised by any example in this text.

Convention (same as R's arima.sim / astsa's sarima.sim):
    ar = [phi_1, phi_2, ...]  in   x_t = phi_1 x_{t-1} + ... + w_t + ma terms
    ma = [theta_1, theta_2, ...] in  ... + w_t + theta_1 w_{t-1} + ...
i.e. these are the model coefficients as usually written in a stats text,
NOT the polynomial-root convention some other software uses.
"""
import warnings as _warnings
import numpy as np


def _as_coef_list(v):
    if v is None:
        return []
    v = np.atleast_1d(v).astype(float)
    if len(v) == 1 and v[0] == 0:
        return []
    return list(v)


def _min_root_modulus(coeffs_increasing):
    """coeffs_increasing[k] = coefficient of z^k. Returns min |root|
    (matches R's min(Mod(polyroot(...))))."""
    c = np.asarray(coeffs_increasing, dtype=float)
    c = np.trim_zeros(c[::-1], "f")  # numpy.roots wants highest-degree first
    if len(c) <= 1:
        return np.inf
    return np.min(np.abs(np.roots(c)))


def _ma_filter(innov, ma):
    """One-sided MA filter: y[t] = innov[t] + ma[0]*innov[t-1] + ...
    First len(ma) entries are 0 (matches R's filter(...,sides=1) + the
    explicit x[seq_along(ma)] <- 0 cleanup in .arima_sim)."""
    q = len(ma)
    x = np.asarray(innov, dtype=float)
    if q == 0:
        return x.copy()
    filt = np.array([1.0] + list(ma))
    n = len(x)
    y = np.zeros(n)
    for t in range(q, n):
        y[t] = np.dot(filt, x[t - np.arange(q + 1)])
    return y


def _ar_recursive(x, ar):
    """Recursive AR filter: y[t] = x[t] + ar[0]*y[t-1] + ar[1]*y[t-2] + ...
    with y[t]=0 for t<0 (matches R's filter(x, ar, method='recursive'))."""
    p = len(ar)
    x = np.asarray(x, dtype=float)
    if p == 0:
        return x.copy()
    n = len(x)
    y = np.zeros(n)
    for t in range(n):
        s = x[t]
        for j in range(p):
            if t - j - 1 >= 0:
                s += ar[j] * y[t - j - 1]
        y[t] = s
    return y


def _diffinv_once(x, lag=1):
    """Matches R's diffinv(x, lag=lag, differences=1): integrates a
    (seasonally) differenced series back up, prepending `lag` zeros."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros(n + lag)
    for i in range(n):
        y[i + lag] = y[i] + x[i]
    return y


def _diffinv(x, d, lag=1):
    for _ in range(int(d)):
        x = _diffinv_once(x, lag)
    return x


def _red_check(ar, ma, sar_poly, sma_poly, red_tol, warn):
    """Warn about (near-)redundant AR/MA or seasonal AR/MA roots, matching
    R's .red_check()."""
    if len(ar) > 0 and len(ma) > 0:
        ar_roots = 1 / np.roots(([1] + [-a for a in ar])[::-1])
        ma_roots = 1 / np.roots(([1] + list(ma))[::-1])
        diffs = np.abs(ar_roots[:, None] - ma_roots[None, :])
        if np.any(diffs < red_tol) and warn:
            _warnings.warn("(Possible) Parameter Redundancy")
    if sar_poly is not None and sma_poly is not None:
        sar_roots = 1 / np.roots(np.trim_zeros(sar_poly[::-1], "f"))
        sma_roots = 1 / np.roots(np.trim_zeros(sma_poly[::-1], "f"))
        diffs = np.abs(sar_roots[:, None] - sma_roots[None, :])
        if np.any(diffs < red_tol) and warn:
            _warnings.warn("(Possible) Seasonal Parameter Redundancy")


def sarima_sim(ar=None, d=0, ma=None, sar=None, D=0, sma=None, S=None,
                n=500, burnin=None, t0=0, red_tol=0.1, warn=True, rng=None):
    """Simulate a (seasonal) ARIMA series. See module docstring for the
    ar/ma sign convention. Returns a plain numpy array of length n (R's
    version returns a ts object with start=t0, frequency=S or 1; wrap the
    result yourself in a pandas Series with whatever index you want).

    Parameters mirror astsa::sarima.sim()'s ar, d, ma, sar, D, sma, S, n,
    burnin, t0, red_tol.
    """
    if rng is None:
        rng = np.random.default_rng()

    ar = _as_coef_list(ar)
    ma = _as_coef_list(ma)
    po, qo = len(ar), len(ma)

    if po > 0:
        m = _min_root_modulus([1.0] + [-a for a in ar])
        if m <= 1 and warn:
            _warnings.warn("AR not causal")
    if qo > 0:
        m = _min_root_modulus([1.0] + ma)
        if m <= 1 and warn:
            _warnings.warn("MA not invertible")

    sar_poly = sma_poly = None

    if S is None:
        if _as_coef_list(sar) or _as_coef_list(sma):
            raise ValueError("the seasonal period 'S' is not specified")
        burnin_ = burnin if burnin is not None else 50 + po + qo + d
        num = n + burnin_
        innov = rng.standard_normal(num)
        x = _ar_recursive(_ma_filter(innov, ma), ar)
        if d > 0:
            x = _diffinv(x, d, lag=1)

    else:
        sar = _as_coef_list(sar)
        sma = _as_coef_list(sma)
        if S <= po:
            raise ValueError("AR order should be less than seasonal order 'S'")
        if S <= qo:
            raise ValueError("MA order should be less than seasonal order 'S'")
        Po, Qo = len(sar), len(sma)

        if Po > 0:
            sar_poly = np.zeros(Po * S + 1)
            sar_poly[0] = 1.0
            for i in range(1, Po + 1):
                sar_poly[i * S] = -sar[i - 1]
            m = _min_root_modulus(sar_poly)
            if m <= 1 and warn:
                _warnings.warn("SAR not causal")
            base_ar_poly = np.array([1.0] + [-a for a in ar]) if po > 0 else np.array([1.0])
            combined = np.convolve(base_ar_poly, sar_poly)
            arnew = list(-combined[1:])
        else:
            arnew = ar

        if Qo > 0:
            sma_poly = np.zeros(Qo * S + 1)
            sma_poly[0] = 1.0
            for i in range(1, Qo + 1):
                sma_poly[i * S] = sma[i - 1]
            m = _min_root_modulus(sma_poly)
            if m <= 1 and warn:
                _warnings.warn("SMA not invertible")
            base_ma_poly = np.array([1.0] + ma) if qo > 0 else np.array([1.0])
            combined = np.convolve(base_ma_poly, sma_poly)
            manew = list(combined[1:])
        else:
            manew = ma

        burnin_default = 50 + (D + Po + Qo) * S + d + po + qo
        burnin_ = burnin if burnin is not None else burnin_default
        num = n + burnin_
        innov = rng.standard_normal(num)
        x = _ar_recursive(_ma_filter(innov, manew), arnew)
        if d > 0:
            x = _diffinv(x, d, lag=1)
        if D > 0:
            x = _diffinv(x, D, lag=S)

    if red_tol > 0:
        _red_check(ar, ma, sar_poly, sma_poly, red_tol, warn)

    return np.asarray(x)[-n:]
