import numpy as np


def preprocess_data(series):
    if len(series.shape) != 1 and series.shape[-1] != 1:
        series = series[:, 0]
    series = series.astype(np.float64)
    return series
