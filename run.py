from feature_extraction import *
from keras_model import KerasModelWrapper
import numpy as np
import matplotlib.pyplot as plt

model_json_file = data_base_path + "model2019-08-19 22:59:39.933736.json"
model_weights_file = data_base_path + "model2019-08-19 22:59:39.933736.h5"
series_filename = "PUT FILENAME HERE"
classification_threshold = 0.2


def convert_features_to_keras_model_input(features):
    to_predict = np.zeros((len(features), extractor.feature_size))
    for index, feature in enumerate(features):
        to_predict[index, :] = feature
    return to_predict.reshape((len(features), extractor.feature_size, 1))


def extract_claps_from_predictions(indices, prediction):
    claps = np.zeros(len(series))
    for index, prediction in zip(indices, prediction):
        if prediction > classification_threshold:
            claps[index: index + extractor.feature_duration] = 1
    return claps


def show_predictions(series, claps):
    scaling_factor = 10_000
    plt.plot(series)
    plt.plot(claps * scaling_factor, 'y')
    plt.show()


pattern_filenames = ['data/pattern-1.npy', 'data/pattern-2.npy']
extractor = FeatureExtractor(feature_count=20, distance_factor=10, pattern_files=pattern_filenames,
                             segment_duration=1024, feature_size=128)
series, _ = extractor.read_series(series_filename)
features, feature_positions = extractor.extract_features_from_file(series_filename)
prepared_features = convert_features_to_keras_model_input(features)
model = KerasModelWrapper(input_length=extractor.feature_size, model_json_filename=model_json_file,
                          model_weights_filename=model_weights_file).model
prediction = model.predict(prepared_features)
claps = extract_claps_from_predictions(feature_positions, prediction)
show_predictions(series, claps)
