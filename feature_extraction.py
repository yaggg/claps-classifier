import numpy as np
from scipy.io import wavfile as wf

data_base_path = 'data/'


class FeatureExtractor:
    def __init__(self, pattern_file=data_base_path + 'pattern.npy', feature_count=20, distance_factor=5):
        self.pattern_filename = pattern_file
        self.pattern = np.load(self.pattern_filename).astype(np.float64)
        self.pattern = self._preprocess_data(self.pattern)
        self.feature_duration = len(self.pattern)
        self.feature_count = feature_count
        self.distance_factor = distance_factor

    def extract_features_from_file(self, filename):
        series = self.read_series(filename)
        series = self._preprocess_data(series)
        return self._find_features_cross_corr(series)

    @staticmethod
    def read_series(filename):
        _, series = wf.read(filename)
        if len(series.shape) != 1 and series.shape[-1] != 1:
            series = series[:, 0]
        return series

    def _preprocess_data(self, series):
        series = series.astype(np.float64)
        series /= np.abs(self.pattern.max())
        series -= self.pattern.mean()
        return series

    def _find_features_cross_corr(self, series):
        correlation = np.correlate(series, self.pattern)
        features = []
        features_indices = []
        for index, value in self._sort_correlations_by_relevance(correlation):
            if not self._is_too_close_to_any_feature(index, features_indices):
                segment = series[index: index + self.feature_duration]
                segment = self._normalize_segment(segment)
                features.append(segment)
                features_indices.append(index)
                if len(features) == self.feature_count:
                    break
        return features, features_indices

    def _sort_correlations_by_relevance(self, correlation):
        return np.sort(np.asarray(list(enumerate(correlation)), dtype=[('index', int), ('value', np.float64)]),
                       order=['value'])

    def _is_too_close_to_any_feature(self, index, features_indices):
        return list(filter(lambda x: abs(x - index) < self.distance_factor * self.feature_duration, features_indices))

    def _normalize_segment(self, series_segment):
        series_segment = np.copy(series_segment)
        abs_max = np.abs(series_segment.max())
        series_segment /= abs_max
        mean = series_segment.mean()
        series_segment -= mean
        return series_segment
