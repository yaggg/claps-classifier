import numpy as np

data_base_path = 'data/'
pattern_filename = 'new-pattern.npy'
supposed_feature_duration = 500
feature_count = 50


def get_pattern():
    return np.load(data_base_path + pattern_filename).astype(np.float64)


def preprocess_data(series):
    if len(series.shape) != 1 and series.shape[-1] != 1:
        series = series[:, 0]
    series = series.astype(np.float64)
    return series


def find_features_cross_corr(series, pattern):
    pattern = normalize_segment(pattern)
    correlation = evaluate_correlation_array(series, pattern)
    features = []
    features_indices = []
    for index, value in sort_correlations_by_relevance(correlation):
        if not is_too_close_to_any_feature(index, features_indices):
            segment = series[index: index + supposed_feature_duration]
            segment = normalize_segment(segment)
            features.append(segment)
            features_indices.append(index)
            if len(features) == feature_count:
                break
    return features, features_indices


def sort_correlations_by_relevance(correlation):
    return sorted(enumerate(correlation), key=lambda x: x[1], reverse=True)


def is_too_close_to_any_feature(index, features_indices):
    list(filter(lambda x: 2 * abs(x - index) < supposed_feature_duration, features_indices))


def evaluate_correlation_array(series, pattern):
    correlation_array_len = len(series) - supposed_feature_duration
    correlation = np.zeros(correlation_array_len)
    for index in range(correlation_array_len):
        series_segment = series[index: index + supposed_feature_duration]
        series_segment = normalize_segment(series_segment)
        correlation[index] = (series_segment * pattern).sum()
    return correlation


def normalize_segment(series_segment):
    series_segment = np.copy(series_segment)
    segment_mean = series_segment.mean()
    segment_abs_max = np.abs(series_segment).max()
    if segment_abs_max != 0:
        series_segment -= segment_mean
        series_segment /= segment_abs_max
    return series_segment
