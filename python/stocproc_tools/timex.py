"""
Python translation of astsa's timex().

R source: astsa/R/timex.R

Converts a datetime index into decimal-year time (e.g. March 15, 2020 ->
~2020.20), matching R's implementation exactly including its leap-year
day-count handling. Used in this book to get a numeric x-axis for the
xts-based djia/sp500w datasets.
"""
import numpy as np
import pandas as pd


def timex(dates):
    """dates: a pandas DatetimeIndex, or anything pd.to_datetime() accepts
    (e.g. the index of a DataFrame loaded via stocproc_tools.load()).

    Returns a numpy array of decimal-year values.
    """
    dates = pd.to_datetime(dates)
    year = dates.year.values
    day_of_year = dates.dayofyear.values
    leap = ((year % 100 == 0) & (year % 400 == 0)) | ((year % 100 != 0) & (year % 4 == 0))
    tot = np.where(leap, 366, 365)
    return year + day_of_year / tot
