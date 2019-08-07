import numpy as np


def preprocess_data(series, pattern):
    if len(series.shape) != 1 and series.shape[-1] != 1:
        series = series[:, 0]
    series = series.astype(np.float64)
    mean, std = pattern.mean(), pattern.std()
    series -= mean
    series /= std
    pattern -= mean
    pattern /= std
    return series, pattern
