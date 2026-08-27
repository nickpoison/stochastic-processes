# Testing this code with no local Python install

The easiest way to try any of this without installing anything on your
own machine is **Google Colab** (colab.research.google.com) — it's free,
runs in a browser, and already has numpy/pandas/scipy/matplotlib/
statsmodels preinstalled.

## 1. Open a new notebook

Go to https://colab.research.google.com -> "New notebook".

## 2. Upload the repo

In the left sidebar, click the folder icon ("Files"), then the upload
icon, and upload `stocproc-python.zip`.

Then in the first code cell, run:

```python
!unzip -q stocproc-python.zip -d stocproc-python
%cd stocproc-python
```

(If Colab already unzipped it into a folder for you, adjust the `%cd`
path to match what you see in the file browser on the left.)

## 3. Quick smoke test

Run this in a cell to confirm everything imports cleanly:

```python
from stocproc_tools import load, tsplot, acf1, acf2, astsa_col, sarima_sim, ts_diag
print("core imports OK")

from stocproc_tools import sarima
from stocproc_tools import load as _load
try:
    sarima(_load("EQcount").values, p=1)
    print("sarima() + statsmodels both work")
except ImportError:
    print("sarima() needs statsmodels -- run: !pip install statsmodels")
```

`sarima` itself always imports fine (the rest of the package doesn't
need `statsmodels`), but *calling* it does need `statsmodels` -- that's
what the try/except above actually checks. Colab usually has it
preinstalled already.

## 4. Run the chapter scripts

These save a PNG instead of popping up a window (that's how they're meant
to be run from a plain terminal), so in Colab, display the image
afterward:

```python
!python ch1/ch1_examples.py
from IPython.display import Image
Image("ch1/ch1_four_examples.png")
```

```python
!python ch5/ch5_sarima_sim.py
Image("ch5/ch5_ma_ar_examples.png")
```

Compare these against the book's Figure 1.1/1.2 (gambler's ruin /
counting process panel) and the Moving Average / Autoregression example
in Ch.5 -- they should look like the same kind of series (not
pixel-identical, since the random seeds differ, but same shape/character).

## 5. Try things interactively

You don't have to run whole scripts -- individual functions work fine in
notebook cells, and plots display automatically:

```python
from stocproc_tools import load, tsplot, acf2

eq = load("EQcount")
tsplot(eq, type="o", col=4, gg=True, main="EQcount")
```

```python
acf2(eq.values, main="EQcount")
```

## 6. Testing `sarima()` (needs a real series)

The book's one fitting example is `sarima(sunspots, p=16)`, using R's
built-in `sunspots` dataset, which isn't bundled in `data/` here (it's a
base-R dataset, not part of astsa, so it wasn't in the package you sent).

Easiest way to check the mechanics work, without the exact series:

```python
from stocproc_tools import load, sarima
eq = load("EQcount").values
out = sarima(eq, p=2)   # arbitrary small example, just to see it run
```

This won't match the book's sunspots output (different series), but it
will tell you whether `sarima()` runs at all and produces a sensible-
looking coefficient table + 4-panel diagnostic plot.

If you want the exact book comparison, the easiest way to get the real
`sunspots` series over is from R itself:

```r
write.csv(data.frame(sunspots=as.numeric(sunspots)), "sunspots.csv", row.names=FALSE)
```

then upload `sunspots.csv` to Colab and:

```python
import pandas as pd
x = pd.read_csv("sunspots.csv")["sunspots"].values
out = sarima(x, p=16)
```

and compare the printed table against the book's Example output.

## 7. What "looks right" vs "looks wrong"

- **`ts_diag()` / `sarima()` diagnostics on a good fit**: ACF-of-residuals
  panel should have no bars poking past the dashed blue lines; Q-Q plot
  points should hug the diagonal line; the p-value panel (bottom right)
  should have most points *above* the dashed 0.05 line.
- **A bad/misspecified fit** looks the opposite: ACF spikes early on,
  curved-away Q-Q tails, p-values pinned near 0.
- If `sarima()`'s printed coefficient table has wildly different
  `Estimate` values than you'd expect from eyeballing the series, or the
  `SE`/`p.value` columns look off (e.g. all p-values near 0 or all near
  1 regardless of the coefficient), that's the thing most likely to need
  a fix — flag it and we'll dig in.
