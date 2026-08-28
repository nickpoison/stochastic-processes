"""
Convert astsa R package .rda datasets to CSV, without needing R or rpy2.

Usage:
    1. Download/extract the astsa R package source (e.g. from CRAN or
       https://github.com/nickpoison/astsa), so you have a folder like
       astsa/data/*.rda
    2. python tools/rda_to_csv.py /path/to/astsa/data name1 name2 ...
       (or edit `targets` below and run with no args)

Only a handful of datasets are wired up with nice date-index export logic
(ts objects, xts objects, plain numeric vectors). For other datasets you
may need to extend export_* below — use r_unserialize.load_rda() directly
to inspect structure first.
"""
import csv
import datetime
import os
import sys
from r_unserialize import load_rda

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def ts_dates(tsp, n):
    """Reconstruct date labels for an R 'ts' object from its tsp=[start,end,freq]."""
    start, end, freq = tsp
    freq = round(freq)
    start_period = round(start * freq)  # integer period index
    labels = []
    for i in range(n):
        period = start_period + i
        year = period // freq
        sub = period % freq
        if freq == 1:
            labels.append(f"{year}")
        elif freq == 4:
            labels.append(f"{year}-Q{sub+1}")
        elif freq == 12:
            labels.append(f"{year}-{sub+1:02d}")
        else:
            labels.append(f"{year}+{sub}/{freq}")
    return labels


def unix_to_date(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d")


def export_ts(name, var):
    tsp = var.attrs["tsp"].value
    n = len(var.value)
    dates = ts_dates(tsp, n)
    path = f"{OUT_DIR}/{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", name])
        for d, v in zip(dates, var.value):
            w.writerow([d, v])
    print(f"wrote {path} ({n} rows)")


def export_xts(name, var):
    dim = var.attrs["dim"].value
    nrow, ncol = dim
    colnames = var.attrs["dimnames"].value[1].value if var.attrs.get("dimnames") else [f"V{i+1}" for i in range(ncol)]
    index = var.attrs["index"].value
    dates = [unix_to_date(t) for t in index]
    # R matrices are stored column-major
    vals = var.value
    path = f"{OUT_DIR}/{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + list(colnames))
        for r in range(nrow):
            row = [dates[r]]
            for c in range(ncol):
                row.append(vals[c * nrow + r])
            w.writerow(row)
    print(f"wrote {path} ({nrow} rows x {ncol} cols)")


def export_plain(name, var):
    path = f"{OUT_DIR}/{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", name])
        for i, v in enumerate(var.value, start=1):
            w.writerow([i, v])
    print(f"wrote {path} ({len(var.value)} rows)")


DEFAULT_TARGETS = ["djia", "EQcount", "gnp", "polio", "sp500w", "GGBsuicide"]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        targets = sys.argv[2:] or DEFAULT_TARGETS
    else:
        data_dir = "astsa_pkg/astsa/data"  # adjust to your local extraction path
        targets = DEFAULT_TARGETS

    for name in targets:
        d = load_rda(f"{data_dir}/{name}.rda")
        var = d[name]
        classes = var.attrs.get("class").value if var.attrs.get("class") else []
        if "xts" in classes:
            export_xts(name, var)
        elif "ts" in classes:
            export_ts(name, var)
        else:
            export_plain(name, var)
