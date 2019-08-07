from numpy import correlate

cross_correlation_threshold = 0.2
supposed_feature_duration = 10_000


def find_features_cross_corr(series, pattern):
    correlation = correlate(series, pattern)
    max_corr = correlate(pattern, pattern)[0]
    correlation /= max_corr
    features = []
    index = 0
    while index < len(correlation):
        if correlation[index] > cross_correlation_threshold:
            features.append(index)
            index += supposed_feature_duration
    return features
