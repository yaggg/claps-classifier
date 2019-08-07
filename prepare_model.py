import numpy as np
from os import listdir
from scipy.io import wavfile as wf
from series_preprocessing import preprocess_data
from feature_extraction import find_features_cross_corr

data_base_path = 'data/'
claps_dir = data_base_path + 'claps/'
non_claps_dir = data_base_path + 'non-claps/'
pattern_filename = 'pattern.npy'

pattern = np.load(data_base_path + pattern_filename)


def process_data_directory(directory):
    features = []
    labels = []
    for filename in listdir(directory):
        _, series = wf.read(filename)
        series = preprocess_data(series, pattern)
        one_file_features = find_features_cross_corr(series, pattern)
        features = features + one_file_features
        label = 0
        if directory == claps_dir:
            label = 1
        labels = labels + [label] * len(one_file_features)
    return features, labels


clap_features, clap_labels = process_data_directory(claps_dir)
non_clap_features, non_clap_labels = process_data_directory(non_claps_dir)
all_features, all_labels = clap_features + non_clap_features, clap_labels + non_clap_labels
