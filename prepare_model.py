import numpy as np
from os import listdir
from scipy.io import wavfile as wf
from series_preprocessing import preprocess_data
from feature_extraction import find_features_cross_corr, supposed_feature_duration
from keras_model import create_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_base_path = 'data/'
claps_dir = data_base_path + 'claps/'
non_claps_dir = data_base_path + 'non-claps/'
pattern_filename = 'pattern.npy'


def process_data_directory(directory, pattern):
    features = []
    labels = []
    for filename in listdir(directory)[:10]:
        _, series = wf.read(directory + filename)
        series, pattern = preprocess_data(series, pattern)
        one_file_features, _ = find_features_cross_corr(series, pattern)
        features = features + one_file_features
        label = 0
        if directory == claps_dir:
            label = 1
        labels = labels + [label] * len(one_file_features)
    return features, labels


clap_features, clap_labels = process_data_directory(claps_dir, pattern=np.load(data_base_path + pattern_filename))
non_clap_features, non_clap_labels = process_data_directory(
    non_claps_dir,
    pattern=np.load(data_base_path + pattern_filename)
)
all_features, all_labels = np.array(clap_features + non_clap_features), np.array(clap_labels + non_clap_labels)

model = create_model(input_length=supposed_feature_duration)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

x_train, x_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3)
history = model.fit(
    x_train.reshape(1009, 10000, 1),
    y_train.reshape(1009, 1),
    validation_data=(x_test.reshape(433, 10000, 1), y_test.reshape(433, 1)),
    epochs=20,
    steps_per_epoch=10,
    validation_steps=10
)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")

acc = history.history["acc"]
plt.plot(acc)
plt.show()
