# Stochastic Processes and Its Applications — Python Companion

Python translations of the R code from David Stoffer's
*Stochastic Processes and Its Applications, With R Examples* (Springer).

This is a companion, not a replacement — the book is written in R, and the
`astsa` R package it uses is not fully ported here. Instead, this repo:

- gives Python equivalents for the R code that actually appears in the text
  and problems, chapter by chapter
- ships the handful of real (non-simulated) datasets the book actually
  uses, as plain CSVs, so you don't need the R `astsa` package installed
- everything else in the book (gambler's ruin, Markov chain simulations,
  MCMC, Poisson processes, etc.) is simulated directly in code, so no
  data files are needed for those examples at all

## Datasets

Only five real datasets are referenced in this book (everything else is
simulated in-code): `EQcount`, `gnp`, `polio`, `sp500w`, `djia`, plus
`GGBsuicide` (used in the Poisson-process/exponential-waiting-time
section of Chapter 5).

```python
from stocproc_tools import load

eq = load("EQcount")       # earthquake counts per year, 1900-2006
gnp = load("gnp")           # US GNP, quarterly
polio = load("polio")       # US polio cases, monthly
suicide = load("GGBsuicide")  # waiting times (days) between events
djia = load("djia")         # Dow Jones OHLCV, daily
sp500 = load("sp500w")      # S&P 500 weekly returns
```

These CSVs were generated from the R `astsa` package's `.rda` files using
`tools/rda_to_csv.py`, which is a small pure-Python reader for R's
serialization format (no R installation or `rpy2`/`pyreadr` required — see
that file if you want to pull in additional `astsa` datasets later).

## Translated so far

- `tsplot()` — single-series and overlaid ("spaghetti") plots, `gg=` styling
- `acf1()`, `acf2()` — ACF / PACF plots with the usual confidence bands
  (implemented directly with numpy/Durbin-Levinson, no statsmodels needed)
- `astsa_col()` — the palette + alpha/wheel color helper
- `sarima_sim()` — simulate from a (seasonal) ARIMA model; validated against
  theoretical ACF (Yule-Walker) for both the plain and seasonal AR cases,
  not just "runs without crashing"
- `ts_diag()` — the 4-panel residual diagnostics plot (standardized
  residuals, ACF, Normal Q-Q, adjusted Q-statistic p-values), split out
  from `sarima()` so it works on residuals from *any* fitted model,
  including `statsmodels` SARIMAX. Includes `adj_qstat()` (Kan & Wang
  2010 exact-moments adjusted Box-Pierce test) and `qq_norm()`.
  `adj_qstat()`'s p-values were validated by simulation: under white
  noise they come out ~Uniform(0,1) (mean 0.51, 4.8% rejected at α=.05
  across several n/fitdf/lag combinations).
- `sarima()` — ARIMA/SARIMA fitting on top of `statsmodels.tsa.statespace.SARIMAX`
  (needs `statsmodels`; nothing else in this repo does). Prints the same
  coefficient table / AIC-AICc-BIC line R's version does, and calls
  `ts_diag()` for the residual plot instead of duplicating that logic.
  **The surrounding math (coefficient SE/t/p, AICc formula) was
  validated** with a self-contained AR(p) least-squares fit in
  `tests/test_sarima_table_math.py` (recovers known simulated
  coefficients; p-values on truly-zero coefficients come out
  ~Uniform(0,1)). The `statsmodels`-fitting path itself could **not** be
  run end-to-end in the sandbox this was built in (no network access to
  install statsmodels there) — please sanity-check your first real run
  against the book's `sarima(sunspots, p=16)` output.
- `ch1/ch1_examples.py` — a runnable translation of the Ch.1 R examples
- `hmm_fit()`, `viterbi()` — Poisson/Normal Hidden Markov Model fitting via
  Baum-Welch EM (k-means auto-init, perturbation restarts, Hessian-based
  SEs for the Normal case), plus Viterbi state decoding.
  **This is the most rigorously validated piece in the repo**: fit
  directly on the real `EQcount` data and compared against the book's
  actual printed output --- `lambda [15.4207, 26.0181]` matched exactly,
  `Pmatrix` matched to 4+ decimal places. The Normal-HMM Hessian SEs were
  checked against empirical variability across 60 independent
  simulated-and-refit replicates (average Hessian SE [0.082, 0.128] vs.
  empirical SD [0.082, 0.132] — excellent agreement). `viterbi()` recovered
  100% of true hidden states on a simulated test case.

- `mvspec()` — periodogram (raw or smoothed via modified-Daniell kernel),
  univariate case only (the two book usages are both univariate; the
  coherency/phase multivariate panels weren't ported -- flag if a later
  chapter needs them). **Validated three ways**: the periodogram's average
  level matches the true white-noise variance almost exactly (3.98 vs.
  true 4.0); the smoothed kernel weights for `spans=[3,3,3]` come out as
  the exact binomial weights `[1,6,15,20,15,6,1]/64` documented for this
  textbook's own example; and the smoothed periodogram's peak frequency
  matches the theoretical spectral density's peak for the same process.
- `arma_spec()` — theoretical ARMA spectral density. **Validated exactly**
  against the closed-form AR(1) formula (difference at floating-point
  precision) and against the variance-integral identity
  (`integral of f(w) dw = process variance`, matched to 4 decimals).
- `timex()` — datetime-to-decimal-year conversion. Validated against
  hand-computed values, including R's specific leap-year day-count
  convention.
- `scatter_hist()` — scatterplot with marginal histograms.

This closes out everything actually called in this book's text. See
`ch5/ch5_spectral.py` for a runnable demo of the last two.

Still not ported (found to be unused by this book when checked): `trend()`
(only the English word "trend" appears in the text, never the function),
the multivariate coherency/phase panels of `mvspec()`, and the broader
`astsa` surface not touched by this text (`Kfilter`/`Ksmooth`/`EM`
state-space versions, `specenv`/`autoSpec`/`autoParm`, `SV.mcmc`/`ar.mcmc`,
etc.) -- see the note earlier in this README about the difference between
"what this book needs" and "a full astsa port."

## Structure

```
data/              the 6 CSVs described above
stocproc_tools/    load_data.py - pandas loaders for the datasets above
tools/             r_unserialize.py - pure-Python R .rda reader
                   rda_to_csv.py    - converts astsa .rda files to CSV
ch1/ ... ch5/      per-chapter Python translations of the book's R code
```

## Requirements

See `requirements.txt`. Core: `numpy`, `pandas`, `scipy`, `matplotlib`,
`statsmodels`.

## Status / scope

This intentionally does **not** attempt a full port of the `astsa` R
package (it's ~50 functions and ~75 datasets, many of them research-grade
code with no direct Python equivalent — e.g. `SV.mcmc`, `autoSpec`,
`specenv`). It covers only what this specific book uses. If you need the
full `astsa` surface, see the (early-stage) community port at
https://github.com/shishitao/astsa.
