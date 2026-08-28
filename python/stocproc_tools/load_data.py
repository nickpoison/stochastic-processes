"""
Loaders for the small set of real (non-simulated) astsa datasets used in
*Stochastic Processes and Its Applications, With R Examples* (Stoffer).

These were converted from the R astsa package's .rda files with
tools/rda_to_csv.py (a pure-Python R-serialization reader — no R or rpy2
needed). See that script if you want to add more datasets later.

Usage:
    from stocproc_tools import load
    eq = load("EQcount")   # pandas Series, indexed by year
    dj = load("djia")      # pandas DataFrame, indexed by date
"""
import os
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# datasets that are a single named series -> return pd.Series
_SERIES = {"EQcount", "gnp", "polio"}
# datasets that are a plain index (no real date/time axis) -> return pd.Series
_PLAIN = {"GGBsuicide"}
# datasets that are multi-column with a date index -> return pd.DataFrame
_FRAMES = {"djia", "sp500w"}

_ALL = _SERIES | _PLAIN | _FRAMES


def load(name):
    """Load one of the book's real datasets by name.

    Available: EQcount, gnp, polio, GGBsuicide, djia, sp500w
    """
    if name not in _ALL:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(_ALL)}")

    path = os.path.join(_DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path)

    if name in _SERIES:
        s = df.set_index("date")[name]
        return s

    if name in _PLAIN:
        return df.set_index("index")[name]

    if name in _FRAMES:
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    raise AssertionError("unreachable")
