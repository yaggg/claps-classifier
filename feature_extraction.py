import numpy as np
from librosa.feature import mfcc
from scipy.io import wavfile as wf

data_base_path = 'data/'


class FeatureExtractor:
    def __init__(self, pattern_files=None, feature_count=20, distance_factor=0.5, segment_duration=None,
                 feature_size=256):
        if pattern_files is None:
            pattern_files = [data_base_path + 'pattern.npy']

        self.pattern_filenames = pattern_files
        self.patterns = []
        for file in pattern_files:
            pattern = np.load(file).astype(np.float64)
            pattern = self._preprocess_data(pattern, pattern)
            self.patterns.append(pattern)

        if segment_duration:
            self.feature_duration = segment_duration
        else:
            self.feature_duration = len(self.patterns[0])
        self.feature_count = feature_count
        self.feature_size = feature_size
        self.distance_factor = distance_factor

    def extract_features_from_file(self, filename):
        series, sr = self.read_series(filename)
        return self._find_features_cross_corr(series, sr)

    @staticmethod
    def read_series(filename):
        sr, series = wf.read(filename)
        if len(series.shape) != 1 and series.shape[-1] != 1:
            series = series[:, 0]
        series = series.astype(np.float64)
        return series, sr

    def _preprocess_data(self, series, pattern):
        series = np.copy(series)
        series -= pattern.mean()
        series /= np.abs(pattern.max())
        return series

    def _find_features_cross_corr(self, series, sr):
        correlation = np.zeros((len(self.patterns), len(series)), dtype=np.float64)
        for index, pattern in enumerate(self.patterns):
            tmp_corr = np.correlate(self._preprocess_data(series, pattern), pattern)
            correlation[index, :len(tmp_corr)] = tmp_corr
        correlation = correlation.T.mean(axis=-1)
        features = []
        features_indices = []
        for index, value in self._sort_correlations_by_relevance(correlation):
            if not self._is_too_close_to_any_feature(index, features_indices):
                segment = series[index: index + self.feature_duration]
                segment = segment * np.hanning(len(segment))
                segment = self._normalize_segment(segment)
                feature = mfcc(segment, n_mfcc=self.feature_size, sr=sr)
                feature = feature.mean(axis=1)
                features.append(feature)
                features_indices.append(index)
                if len(features) == self.feature_count:
                    break
        return features, features_indices

    def _sort_correlations_by_relevance(self, correlation):
        return np.sort(np.asarray(list(enumerate(filter(lambda x: x != 0, abs(correlation)))),
                                  dtype=[('index', int), ('value', np.float64)]),
                       order=['value'])[::-1]

    def _is_too_close_to_any_feature(self, index, features_indices):
        return list(filter(lambda x: abs(x - index) < self.distance_factor * self.feature_duration, features_indices))

    def _normalize_segment(self, series_segment):
        series_segment = np.copy(series_segment)
        mean = series_segment.mean()
        series_segment -= mean
        abs_max = np.abs(series_segment.max())
        series_segment /= abs_max
        return series_segment
