from numpy import correlate
import numpy as np

cross_correlation_threshold = 0.2
supposed_feature_duration = 10_000


def find_features_cross_corr(series, pattern, threshold=cross_correlation_threshold):
    correlation = correlate(series, pattern)
    max_corr = correlate(pattern, pattern)[0]
    correlation /= max_corr
    features = []
    features_indices = []
    index = 0
    while index < len(correlation):
        if correlation[index] > threshold and feature_has_enough_length(correlation, index):
            features.append(normalize_segment(np.copy(series[index: index + supposed_feature_duration])))
            features_indices.append(index)
            index += supposed_feature_duration
        else:
            index += 1
    return features, features_indices


def find_features_cross_corr_slow(series, pattern, threshold=cross_correlation_threshold):
    pattern = np.abs(normalize_segment(pattern))
    correlation_array_len = len(series) - supposed_feature_duration
    correlation = np.zeros(correlation_array_len)
    for index in range(correlation_array_len):
        series_segment = np.abs(normalize_segment(np.copy(series[index: index + supposed_feature_duration])))
        correlation[index] = (series_segment * pattern).sum()
    max_corr = correlate(pattern, pattern)[0]
    correlation /= max_corr
    features = []
    features_indices = []
    index = 0
    while index < len(correlation):
        if correlation[index] > threshold:
            series_segment = normalize_segment(np.copy(series[index: index + supposed_feature_duration]))
            features.append(series_segment)
            features_indices.append(index)
            index += supposed_feature_duration
        else:
            index += 1
    return features, features_indices


def feature_has_enough_length(correlation, index):
    return len(correlation) - index >= supposed_feature_duration


def normalize_segment(series_segment):
    segment_mean = series_segment.mean()
    segment_abs_max = np.abs(series_segment).max()
    if segment_abs_max != 0:
        series_segment -= segment_mean
        series_segment /= segment_abs_max
    return series_segment
