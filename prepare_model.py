import datetime
from os import listdir
from os import path

import numpy as np
from sklearn.model_selection import train_test_split

from feature_extraction import FeatureExtractor, data_base_path
from keras_model import KerasModelWrapper

claps_dir = data_base_path + 'claps/'
non_claps_dir = data_base_path + 'non-claps/'
all_features_file = data_base_path + 'all_features.npy'
all_labels_file = data_base_path + 'all_labels.npy'
count_of_files_to_extract_features = 1000


def get_features_and_labels():
    if path.isfile(all_features_file):
        all_features = np.load(all_features_file)
        all_labels = np.load(all_labels_file)
    else:
        all_features, all_labels = extract_features_from_file()
        np.save(all_features_file, all_features)
        np.save(all_labels_file, all_labels)
    return all_features, all_labels


def process_data_directory(directory, extractor):
    features = []
    labels = []
    for filename in listdir(directory)[:count_of_files_to_extract_features]:
        one_file_features, _ = extractor.extract_features_from_file(directory + filename)
        features = features + one_file_features
        label = 0
        if directory == claps_dir:
            label = 1
        labels = labels + [label] * len(one_file_features)
        print('{}: Processed file {} from directory {}, get {} features'
              .format(datetime.datetime.now(), filename, directory, len(features)))
    return features, labels


def extract_features_from_file():
    clap_features, clap_labels = process_data_directory(claps_dir, claps_extractor)
    non_clap_features, non_clap_labels = process_data_directory(non_claps_dir, non_claps_extractor)
    one_class_sample_count = min(len(clap_features), len(non_clap_features))
    return np.array(clap_features[:one_class_sample_count] + non_clap_features[:one_class_sample_count]), \
           np.array(clap_labels[:one_class_sample_count] + non_clap_labels[:one_class_sample_count])


pattern_filenames = ['data/pattern-1.npy', 'data/pattern-2.npy']
claps_extractor = FeatureExtractor(feature_count=5, distance_factor=10, pattern_files=pattern_filenames,
                                   segment_duration=1024, feature_size=128)
non_claps_extractor = FeatureExtractor(feature_count=1, distance_factor=10, pattern_files=pattern_filenames,
                                       segment_duration=1024, feature_size=128)
wrapper = KerasModelWrapper(input_length=claps_extractor.feature_size)
all_features, all_labels = get_features_and_labels()
x_train, x_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3)
history = wrapper.fit_model(x_train, x_test, y_train, y_test, claps_extractor.feature_size, epochs=30)
wrapper.save_model()
wrapper.show_learning_curve(history)
