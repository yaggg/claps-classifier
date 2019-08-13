import datetime
from os import listdir
from os import path

import matplotlib.pyplot as plt
from scipy.io import wavfile as wf
from sklearn.model_selection import train_test_split

from feature_extraction import *
from keras_model import create_model

claps_dir = data_base_path + 'claps/'
non_claps_dir = data_base_path + 'non-claps/'
all_features_file = data_base_path + 'all_features.npy'
all_labels_file = data_base_path + 'all_labels.npy'
count_of_files_to_extract_features = 1


def get_features_and_labels():
    if path.isfile(all_features_file):
        all_features = np.load(all_features_file)
        all_labels = np.load(all_labels_file)
    else:
        all_features, all_labels = extract_features_from_file()
        np.save(all_features_file, all_features)
        np.save(all_labels_file, all_labels)
    return all_features, all_labels


def process_data_directory(directory, pattern):
    features = []
    labels = []
    for filename in listdir(directory)[:count_of_files_to_extract_features]:
        _, series = wf.read(directory + filename)
        series = preprocess_data(series)
        one_file_features, _ = find_features_cross_corr(series, pattern)
        features = features + one_file_features
        label = 0
        if directory == claps_dir:
            label = 1
        labels = labels + [label] * len(one_file_features)
    return features, labels


def extract_features_from_file():
    pattern = get_pattern()
    clap_features, clap_labels = process_data_directory(claps_dir, pattern)
    non_clap_features, non_clap_labels = process_data_directory(non_claps_dir, pattern)
    one_class_sample_count = min(len(clap_features), len(non_clap_features))
    return np.array(clap_features[:one_class_sample_count] + non_clap_features[:one_class_sample_count]), \
           np.array(clap_labels[:one_class_sample_count] + non_clap_labels[:one_class_sample_count])


def fit_model(x_train, x_test, y_train, y_test):
    model = create_model(input_length=supposed_feature_duration)
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(
        x_train.reshape(len(x_train), supposed_feature_duration, 1), y_train,
        validation_data=(x_test.reshape(len(x_test), supposed_feature_duration, 1), y_test),
        epochs=30, steps_per_epoch=20, validation_steps=20, shuffle=True
    )
    return model, history


def save_model(model):
    model_json = model.to_json()
    unique_string = str(datetime.datetime.now())
    json_file = open(data_base_path + 'model' + unique_string + '.json', 'w')
    json_file.write(model_json)
    model.save(data_base_path + 'model' + unique_string + '.h5')


def show_learning_curve(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(acc, 'g', val_acc, 'b')
    plt.show()


all_features, all_labels = get_features_and_labels()
model, history = fit_model(*train_test_split(all_features, all_labels, test_size=0.3))
save_model(model)
show_learning_curve(history)
