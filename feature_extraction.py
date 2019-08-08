from numpy import correlate

cross_correlation_threshold = 0.2
supposed_feature_duration = 10_000


def find_features_cross_corr(series, pattern):
    correlation = correlate(series, pattern)
    max_corr = correlate(pattern, pattern)[0]
    correlation /= max_corr
    features = []
    features_indices = []
    index = 0
    while index < len(correlation):
        if correlation[index] > cross_correlation_threshold and feature_has_enough_length(correlation, index):
            features.append(series[index: index + supposed_feature_duration])
            features_indices.append(index)
            index += supposed_feature_duration
        else:
            index += 1
    return features, features_indices


def feature_has_enough_length(correlation, index):
    return len(correlation) - index >= supposed_feature_duration
