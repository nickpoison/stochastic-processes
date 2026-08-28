"""
Python translation of astsa's HmmFit().

R source: astsa/R/HmmFit.R

Note on scope: Kfilter(), Ksmooth(), EM() (the linear-Gaussian state-space
versions), and Viterbi() are discussed conceptually in the book's Ch.5.6
but never actually *called* anywhere in the text -- the only real calls
are three HmmFit() examples (Poisson HMM on EQcount, Normal HMM on a
synthetic series, both with m=2 states). So this module focuses on
HmmFit() plus a small Viterbi decoder (useful, and genuinely tiny, even
though it's not directly called in the text).

Not ported (used a `se="none"` fallback or simplification instead):
  - parametric bootstrap SEs (se="boot") -- the default is se="hessian",
    which IS ported below.
"""
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm as _norm, poisson as _poisson


# ------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------
def _stat_dist(Gamma):
    """Stationary distribution of a transition matrix (left eigenvector
    for eigenvalue 1)."""
    vals, vecs = np.linalg.eig(Gamma.T)
    idx = np.argmin(np.abs(vals - 1))
    v = np.real(vecs[:, idx])
    return v / v.sum()


def _gamma_logit_pack(Gamma):
    m = Gamma.shape[0]
    out = []
    for i in range(m):
        p = np.maximum(Gamma[i, :], 1e-10)
        out.extend(np.log(p[:-1] / p[-1]))
    return np.array(out)


def _gamma_logit_unpack(logits, m):
    Gamma = np.zeros((m, m))
    idx = 0
    for i in range(m):
        lg = np.concatenate([logits[idx:idx + m - 1], [0.0]])
        p = np.exp(lg - lg.max())
        p = p / p.sum()
        Gamma[i, :] = p
        idx += m - 1
    return Gamma


def _perturb_gamma(Gamma0, m, rng, strength=None):
    if strength is None:
        strength = rng.uniform(0.1, 0.7)
    Gnew = np.zeros((m, m))
    for i in range(m):
        rand_row = rng.gamma(shape=1.0, size=m)
        rand_row = rand_row / rand_row.sum()
        Gnew[i, :] = (1 - strength) * Gamma0[i, :] + strength * rand_row
        Gnew[i, :] = Gnew[i, :] / Gnew[i, :].sum()
    return Gnew


def _central_diff_hessian(f, x, h=1e-4):
    """Numerical Hessian of scalar function f at x via central differences
    (matches R's nlme::fdHess Hessian output closely enough for our SE use)."""
    n = len(x)
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
            H[j, i] = H[i, j]
    return H


def _central_diff_jacobian(f, x, h=1e-4):
    fx = np.asarray(f(x))
    n = len(x)
    J = np.zeros((len(fx), n))
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        J[:, i] = (np.asarray(f(xp)) - np.asarray(f(xm))) / (2 * h)
    return J


def _kmeans_1d(x, m, rng_seed=0):
    """1-D k-means matching R's kmeans(x, centers=m, nstart=25) closely
    enough: sklearn's n_init=25 does the same "many random starts, keep
    best inertia" thing R's nstart does."""
    km = KMeans(n_clusters=m, n_init=25, random_state=rng_seed).fit(x.reshape(-1, 1))
    return km.labels_, km.cluster_centers_.ravel()


def _empirical_transition(map_, m):
    n = len(map_)
    Gamma0 = np.zeros((m, m))
    for t in range(n - 1):
        Gamma0[map_[t], map_[t + 1]] += 1
    Gamma0 += 0.5
    Gamma0 = Gamma0 / Gamma0.sum(axis=1, keepdims=True)
    return Gamma0


def _empirical_delta(map_, m):
    counts = np.array([(map_ == j).sum() for j in range(m)], dtype=float)
    return counts / counts.sum()


# ------------------------------------------------------------------
# Poisson HMM
# ------------------------------------------------------------------
def _pois_hmm_em(x, m, lambda0=None, Gamma0=None, delta0=None,
                  maxiter=500, tol=1e-8):
    n = len(x)
    lam = (np.quantile(x, np.linspace(0, 1, m + 2))[1:-1]
            if lambda0 is None else np.asarray(lambda0, dtype=float))
    if Gamma0 is None:
        Gamma = np.full((m, m), 0.1 / (m - 1))
        np.fill_diagonal(Gamma, 0.9)
    else:
        Gamma = np.asarray(Gamma0, dtype=float).copy()
    delta = np.full(m, 1 / m) if delta0 is None else np.asarray(delta0, dtype=float).copy()

    oldloglik = -np.inf
    loglik = -np.inf
    for it in range(1, maxiter + 1):
        B = np.column_stack([_poisson.pmf(x, lam[j]) for j in range(m)])

        alpha = np.zeros((n, m))
        cvec = np.zeros(n)
        a = delta * B[0, :]
        cvec[0] = a.sum()
        alpha[0, :] = a / cvec[0]
        for t in range(1, n):
            a = (alpha[t - 1, :] @ Gamma) * B[t, :]
            cvec[t] = a.sum()
            alpha[t, :] = a / cvec[t]
        loglik = np.sum(np.log(cvec))

        beta = np.zeros((n, m))
        beta[-1, :] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t, :] = (Gamma @ (B[t + 1, :] * beta[t + 1, :])) / cvec[t + 1]

        gam = alpha * beta
        gam = gam / gam.sum(axis=1, keepdims=True)

        xi = np.zeros((m, m))
        for t in range(n - 1):
            num = np.outer(alpha[t, :], B[t + 1, :] * beta[t + 1, :]) * Gamma
            xi += num / cvec[t + 1]

        Gamma_new = xi / xi.sum(axis=1, keepdims=True)
        lam_new = (gam * x[:, None]).sum(axis=0) / gam.sum(axis=0)
        delta_new = gam[0, :]

        diff = abs(loglik - oldloglik)
        Gamma, lam, delta = Gamma_new, lam_new, delta_new
        if diff < tol:
            break
        oldloglik = loglik

    pis = _stat_dist(Gamma)
    return dict(lam=lam, Gamma=Gamma, delta=delta, pis=pis, loglik=loglik,
                 posterior=gam, niter=it)


def _auto_init_pois_hmm(x, m):
    n = len(x)
    if n >= 5 * m:
        map_, centers = _kmeans_1d(x, m)
        order = np.argsort(centers)
        relabel = np.zeros(m, dtype=int)
        relabel[order] = np.arange(m)
        map_ = relabel[map_]
        lam0 = np.sort(centers)
    else:
        ranks = np.argsort(np.argsort(x)) + 1
        map_ = np.clip(np.ceil(ranks / n * m).astype(int) - 1, 0, m - 1)
        lam0 = np.array([x[map_ == j].mean() for j in range(m)])
    Gamma0 = _empirical_transition(map_, m)
    delta0 = _empirical_delta(map_, m)
    return dict(lambda0=lam0, Gamma0=Gamma0, delta0=delta0, cluster=map_)


def _hmm_fit_pois(x, m=2, n_perturb=5, start=None, se="hessian", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, dtype=float)

    if start is not None:
        init = dict(lambda0=np.asarray(start["lambda0"], dtype=float),
                     Gamma0=np.asarray(start["Gamma0"], dtype=float),
                     delta0=np.asarray(start.get("delta0", np.full(m, 1 / m))))
    else:
        init = _auto_init_pois_hmm(x, m)

    best = _pois_hmm_em(x, m, lambda0=init["lambda0"], Gamma0=init["Gamma0"],
                          delta0=init["delta0"])

    for _ in range(n_perturb):
        lam_p = np.maximum(init["lambda0"] * rng.uniform(0.7, 1.3, m), 0.1)
        Gamma_p = _perturb_gamma(init["Gamma0"], m, rng)
        try:
            fit = _pois_hmm_em(x, m, lambda0=lam_p, Gamma0=Gamma_p, delta0=init["delta0"])
            if fit["loglik"] > best["loglik"]:
                best = fit
        except Exception:
            pass

    order = np.argsort(best["lam"])
    best["lam"] = best["lam"][order]
    best["delta"] = best["delta"][order]
    best["pis"] = best["pis"][order]
    best["Gamma"] = best["Gamma"][np.ix_(order, order)]
    best["posterior"] = best["posterior"][:, order]

    n = len(x)
    npar = m + m * (m - 1) + (m - 1)
    best["npar"] = npar
    best["AIC"] = -2 * best["loglik"] + 2 * npar
    best["BIC"] = -2 * best["loglik"] + np.log(n) * npar
    best["x"] = x
    best["family"] = "pois"
    return best


# ------------------------------------------------------------------
# Normal HMM
# ------------------------------------------------------------------
def _norm_hmm_em(x, m, mu0=None, sigma0=None, Gamma0=None, delta0=None,
                  maxiter=500, tol=1e-8):
    n = len(x)
    mu = (np.quantile(x, np.linspace(0, 1, m + 2))[1:-1]
           if mu0 is None else np.asarray(mu0, dtype=float))
    sigma = (np.full(m, x.std(ddof=1)) if sigma0 is None
              else np.asarray(sigma0, dtype=float).copy())
    if Gamma0 is None:
        Gamma = np.full((m, m), 0.1 / (m - 1))
        np.fill_diagonal(Gamma, 0.9)
    else:
        Gamma = np.asarray(Gamma0, dtype=float).copy()
    delta = np.full(m, 1 / m) if delta0 is None else np.asarray(delta0, dtype=float).copy()

    sigma_floor = max(1e-6, 1e-4 * x.std(ddof=1))
    oldloglik = -np.inf
    loglik = -np.inf

    for it in range(1, maxiter + 1):
        B = np.column_stack([_norm.pdf(x, loc=mu[j], scale=sigma[j]) for j in range(m)])

        alpha = np.zeros((n, m))
        cvec = np.zeros(n)
        a = delta * B[0, :]
        cvec[0] = a.sum()
        alpha[0, :] = a / cvec[0]
        for t in range(1, n):
            a = (alpha[t - 1, :] @ Gamma) * B[t, :]
            cvec[t] = a.sum()
            alpha[t, :] = a / cvec[t]
        loglik = np.sum(np.log(cvec))

        beta = np.zeros((n, m))
        beta[-1, :] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t, :] = (Gamma @ (B[t + 1, :] * beta[t + 1, :])) / cvec[t + 1]

        gam = alpha * beta
        gam = gam / gam.sum(axis=1, keepdims=True)

        xi = np.zeros((m, m))
        for t in range(n - 1):
            num = np.outer(alpha[t, :], B[t + 1, :] * beta[t + 1, :]) * Gamma
            xi += num / cvec[t + 1]

        Gamma_new = xi / xi.sum(axis=1, keepdims=True)
        mu_new = (gam * x[:, None]).sum(axis=0) / gam.sum(axis=0)
        sigma_new = np.sqrt((gam * (x[:, None] - mu_new[None, :]) ** 2).sum(axis=0)
                              / gam.sum(axis=0))
        sigma_new = np.maximum(sigma_new, sigma_floor)
        delta_new = gam[0, :]

        diff = abs(loglik - oldloglik)
        Gamma, mu, sigma, delta = Gamma_new, mu_new, sigma_new, delta_new
        if diff < tol:
            break
        oldloglik = loglik

    pis = _stat_dist(Gamma)
    return dict(mu=mu, sigma=sigma, Gamma=Gamma, delta=delta, pis=pis,
                 loglik=loglik, posterior=gam, niter=it)


def _auto_init_norm_hmm(x, m):
    n = len(x)
    if n >= 5 * m:
        map_, centers = _kmeans_1d(x, m)
        order = np.argsort(centers)
        relabel = np.zeros(m, dtype=int)
        relabel[order] = np.arange(m)
        map_ = relabel[map_]
        mu0 = np.sort(centers)
    else:
        ranks = np.argsort(np.argsort(x)) + 1
        map_ = np.clip(np.ceil(ranks / n * m).astype(int) - 1, 0, m - 1)
        mu0 = np.array([x[map_ == j].mean() for j in range(m)])
    sigma0 = np.array([
        x[map_ == j].std(ddof=1) if (map_ == j).sum() > 1 else x.std(ddof=1)
        for j in range(m)
    ])
    sigma0 = np.where((sigma0 <= 0) | np.isnan(sigma0), x.std(ddof=1), sigma0)
    Gamma0 = _empirical_transition(map_, m)
    delta0 = _empirical_delta(map_, m)
    return dict(mu0=mu0, sigma0=sigma0, Gamma0=Gamma0, delta0=delta0, cluster=map_)


def _auto_init_norm_hmm_vol(x, m, w=None):
    n = len(x)
    if w is None:
        w = max(3, min(10, n // (2 * m)))
    dev2 = (x - x.mean()) ** 2
    roll = np.array([dev2[max(0, t - w + 1):t + 1].mean() for t in range(n)])
    logroll = np.log(roll + 1e-8 * x.var(ddof=1))
    map_, centers = _kmeans_1d(logroll, m)
    order = np.argsort(centers)
    relabel = np.zeros(m, dtype=int)
    relabel[order] = np.arange(m)
    map_ = relabel[map_]
    mu0 = np.array([x[map_ == j].mean() for j in range(m)])
    sigma0 = np.array([
        x[map_ == j].std(ddof=1) if (map_ == j).sum() > 1 else x.std(ddof=1)
        for j in range(m)
    ])
    sigma0 = np.where((sigma0 <= 0) | np.isnan(sigma0), x.std(ddof=1), sigma0)
    Gamma0 = _empirical_transition(map_, m)
    delta0 = _empirical_delta(map_, m)
    return dict(mu0=mu0, sigma0=sigma0, Gamma0=Gamma0, delta0=delta0, cluster=map_)


def _norm_hmm_hessian_se(fit, x, m):
    """Delta-method SEs for mu/sigma/Gamma, via a numerical Hessian of the
    negative log-likelihood in a reparameterized (unconstrained) space:
    mu free, log(sigma) for positivity, logit-per-row for Gamma."""
    n = len(x)
    delta = fit["delta"]

    def unpack(theta):
        mu = theta[:m]
        sigma = np.exp(theta[m:2 * m])
        Gamma = _gamma_logit_unpack(theta[2 * m:2 * m + m * (m - 1)], m)
        return mu, sigma, Gamma

    def loglik_fn(mu, sigma, Gamma):
        B = np.column_stack([_norm.pdf(x, loc=mu[j], scale=sigma[j]) for j in range(m)])
        a = delta * B[0, :]
        c1 = a.sum()
        ll = np.log(c1)
        a = a / c1
        for t in range(1, n):
            a = (a @ Gamma) * B[t, :]
            ct = a.sum()
            ll += np.log(ct)
            a = a / ct
        return ll

    def negloglik(theta):
        mu, sigma, Gamma = unpack(theta)
        return -loglik_fn(mu, sigma, Gamma)

    def natural(theta):
        mu, sigma, Gamma = unpack(theta)
        return np.concatenate([mu, sigma, Gamma.ravel()])

    theta_hat = np.concatenate([fit["mu"], np.log(fit["sigma"]),
                                  _gamma_logit_pack(fit["Gamma"])])
    H = _central_diff_hessian(negloglik, theta_hat)
    try:
        Vt = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Vt = np.full((len(theta_hat), len(theta_hat)), np.nan)

    J = _central_diff_jacobian(natural, theta_hat)
    Vnat = J @ Vt @ J.T
    with np.errstate(invalid="ignore"):
        se_natural = np.sqrt(np.diag(Vnat))

    se_mu = se_natural[:m]
    se_sigma = se_natural[m:2 * m]
    se_Gamma = se_natural[2 * m:2 * m + m * m].reshape(m, m)

    return dict(
        state=np.arange(1, m + 1), mu=fit["mu"], se_mu=se_mu,
        mu_lower95=fit["mu"] - 1.96 * se_mu, mu_upper95=fit["mu"] + 1.96 * se_mu,
        sigma=fit["sigma"], se_sigma=se_sigma,
        sigma_lower95=fit["sigma"] - 1.96 * se_sigma,
        sigma_upper95=fit["sigma"] + 1.96 * se_sigma,
        Gamma_se=se_Gamma,
    )


def _hmm_fit_norm(x, m=2, n_perturb=5, start=None, se="hessian",
                    order_by="mean", init_method="auto", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, dtype=float)
    n = len(x)

    init_loc = init_vol = None
    best = None
    best_init = None

    if start is not None:
        init = dict(mu0=np.asarray(start["mu0"], dtype=float),
                     sigma0=np.asarray(start["sigma0"], dtype=float),
                     Gamma0=np.asarray(start["Gamma0"], dtype=float),
                     delta0=np.asarray(start.get("delta0", np.full(m, 1 / m))))
        best = _norm_hmm_em(x, m, **init)
        best_init = init
    else:
        if init_method != "volatility":
            init_loc = _auto_init_norm_hmm(x, m)
            try:
                fit_loc = _norm_hmm_em(x, m, mu0=init_loc["mu0"], sigma0=init_loc["sigma0"],
                                         Gamma0=init_loc["Gamma0"], delta0=init_loc["delta0"])
                best, best_init = fit_loc, init_loc
            except Exception:
                pass

        if init_method != "location" and n >= 5 * m:
            try:
                init_vol = _auto_init_norm_hmm_vol(x, m)
                fit_vol = _norm_hmm_em(x, m, mu0=init_vol["mu0"], sigma0=init_vol["sigma0"],
                                         Gamma0=init_vol["Gamma0"], delta0=init_vol["delta0"])
                if np.isfinite(fit_vol["loglik"]) and (best is None or fit_vol["loglik"] > best["loglik"]):
                    best, best_init = fit_vol, init_vol
            except Exception:
                pass

        if best is None:
            init_loc = _auto_init_norm_hmm(x, m)
            best = _norm_hmm_em(x, m, mu0=init_loc["mu0"], sigma0=init_loc["sigma0"],
                                  Gamma0=init_loc["Gamma0"], delta0=init_loc["delta0"])
            best_init = init_loc

    jitter_scale = 0.3 * x.std(ddof=1)
    for _ in range(n_perturb):
        mu_p = best_init["mu0"] + rng.normal(0, jitter_scale, m)
        sigma_p = np.maximum(best_init["sigma0"] * rng.uniform(0.7, 1.3, m), 1e-3)
        Gamma_p = _perturb_gamma(best_init["Gamma0"], m, rng)
        try:
            fit = _norm_hmm_em(x, m, mu0=mu_p, sigma0=sigma_p, Gamma0=Gamma_p,
                                 delta0=best_init["delta0"])
            if np.isfinite(fit["loglik"]) and fit["loglik"] > best["loglik"]:
                best = fit
        except Exception:
            pass

    order = np.argsort(best["sigma"]) if order_by == "sd" else np.argsort(best["mu"])
    best["mu"] = best["mu"][order]
    best["sigma"] = best["sigma"][order]
    best["delta"] = best["delta"][order]
    best["pis"] = best["pis"][order]
    best["Gamma"] = best["Gamma"][np.ix_(order, order)]
    best["posterior"] = best["posterior"][:, order]

    npar = 2 * m + m * (m - 1) + (m - 1)
    best["npar"] = npar
    best["AIC"] = -2 * best["loglik"] + 2 * npar
    best["BIC"] = -2 * best["loglik"] + np.log(n) * npar

    if se == "hessian":
        best["se"] = _norm_hmm_hessian_se(best, x, m)

    best["x"] = x
    best["family"] = "norm"
    return best


# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------
def hmm_fit(x, m=2, family="pois", n_perturb=5, start=None, se="hessian",
             order_by="mean", init_method="auto", seed=None):
    """Fit an m-state Hidden Markov Model to x, R-astsa HmmFit() style.

    family: "pois" (default) or "norm".
    Returns a dict with keys matching R's astsa::HmmFit() output
    (lam/Pmatrix/delta for pois; mu/sigma/Pmatrix/delta for norm), plus
    loglik, posterior, AIC, BIC, and se (if se="hessian" and family="norm").
    (Pmatrix here is called "Gamma" internally but returned as "Pmatrix"
    for parity with R's fm$Pmatrix.)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)

    if family == "pois":
        fit = _hmm_fit_pois(x, m=m, n_perturb=n_perturb, start=start, se=se, rng=rng)
        fit["Pmatrix"] = fit.pop("Gamma")
        fit["lambda"] = fit.pop("lam")
    elif family == "norm":
        fit = _hmm_fit_norm(x, m=m, n_perturb=n_perturb, start=start, se=se,
                              order_by=order_by, init_method=init_method, rng=rng)
        fit["Pmatrix"] = fit.pop("Gamma")
    else:
        raise ValueError("family must be 'pois' or 'norm'")
    return fit


def viterbi(fit):
    """Most likely state sequence given a fitted HMM (from hmm_fit()).
    Standard Viterbi in log-space -- not called anywhere in this book's
    text, but small enough to include since it's a natural pairing with
    hmm_fit()'s posterior-probability decoding."""
    x = fit["x"]
    Gamma = fit["Pmatrix"]
    delta = fit["delta"]
    m = len(delta)
    n = len(x)

    if "lambda" in fit:
        logB = np.column_stack([_poisson.logpmf(x, fit["lambda"][j]) for j in range(m)])
    else:
        logB = np.column_stack([_norm.logpdf(x, loc=fit["mu"][j], scale=fit["sigma"][j])
                                  for j in range(m)])

    logGamma = np.log(np.maximum(Gamma, 1e-300))
    logdelta = np.log(np.maximum(delta, 1e-300))

    logV = np.zeros((n, m))
    back = np.zeros((n, m), dtype=int)
    logV[0, :] = logdelta + logB[0, :]
    for t in range(1, n):
        for j in range(m):
            scores = logV[t - 1, :] + logGamma[:, j]
            back[t, j] = np.argmax(scores)
            logV[t, j] = scores[back[t, j]] + logB[t, j]

    states = np.zeros(n, dtype=int)
    states[-1] = np.argmax(logV[-1, :])
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]
    return states + 1  # 1-indexed, matching R
