from keras.models import model_from_json
from scipy.io import wavfile as wf
from feature_extraction import *
import numpy as np
import matplotlib.pyplot as plt

test_files = [
    '/home/yakov/sound-search/ESC-50-master/audio/1-115920-A-22.wav',  # low frequency shit
    '/home/yakov/Documents/sum.wav',
    '/home/yakov/Audio/test_sum.wav',
    '/home/yakov/Audio/2019-08-04-20:18:28.wav',
    '/home/yakov/Audio/2019-08-07-20:59:27.wav',
    '/home/yakov/Audio/2019-08-09-08:36:07.wav'
]
series_filename = test_files[-1]
model_json_filename = data_base_path + "model2019-08-13 21:00:23.212254.json"
model_weights_filename = data_base_path + "model2019-08-13 21:00:23.212254.h5"
classification_threshold = 0.1


def read_series_from_file(filename):
    _, series = wf.read(filename)
    series = preprocess_data(series)
    return series


def extract_features_from_file(series):
    pattern = get_pattern()
    return find_features_cross_corr(series, pattern)


def convert_features_to_keras_model_input(features):
    to_predict = np.zeros((len(features), supposed_feature_duration))
    for index, feature in enumerate(features):
        to_predict[index, :] = feature
    return to_predict.reshape((len(features), supposed_feature_duration, 1))


def get_keras_model():
    model_file = open(model_json_filename)
    loaded_model = model_file.read()
    model = model_from_json(loaded_model)
    model.load_weights(model_weights_filename)
    return model


def extract_claps_from_predictions(indices, prediction):
    claps = np.zeros(len(series))
    for index, prediction in zip(indices, prediction):
        if prediction > classification_threshold:
            claps[index: index + supposed_feature_duration] = 1
    return claps


def show_predictions(series, claps):
    plt.plot(series)
    plt.plot(claps * 10_000)
    plt.show()


series = read_series_from_file(series_filename)
features, feature_positions = extract_features_from_file(series)
prepared_features = convert_features_to_keras_model_input(features)
model = get_keras_model()
prediction = model.predict(prepared_features)
claps = extract_claps_from_predictions(feature_positions, prediction)
show_predictions(series, claps)
