import numpy as np
from os import listdir
from scipy.io import wavfile as wf
from series_preprocessing import preprocess_data
from feature_extraction import find_features_cross_corr, supposed_feature_duration, cross_correlation_threshold
from keras_model import create_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_base_path = 'data/'
claps_dir = data_base_path + 'claps/'
non_claps_dir = data_base_path + 'non-claps/'
pattern_filename = 'pattern.npy'


def process_data_directory(directory, pattern, threshold=cross_correlation_threshold):
    features = []
    labels = []
    # pattern_mean = pattern.mean()
    # pattern_abs_max = np.abs(pattern).max()
    # pattern -= pattern_mean
    # pattern /= pattern_abs_max
    for filename in listdir(directory)[:20]:
        _, series = wf.read(directory + filename)
        series = preprocess_data(series)
        one_file_features, _ = find_features_cross_corr(series, pattern, threshold)
        features = features + one_file_features
        label = 0
        if directory == claps_dir:
            label = 1
        labels = labels + [label] * len(one_file_features)
    return features, labels


clap_features, clap_labels = process_data_directory(
    claps_dir,
    pattern=np.load(data_base_path + pattern_filename),
    threshold=0.1
)
non_clap_features, non_clap_labels = process_data_directory(
    non_claps_dir,
    pattern=np.load(data_base_path + pattern_filename),
    threshold=0.05
)
one_class_sample_count = min(len(clap_features), len(non_clap_features))
all_features, all_labels = np.array(
    clap_features[:one_class_sample_count] + non_clap_features[:one_class_sample_count]), \
                           np.array(clap_labels[:one_class_sample_count] + non_clap_labels[:one_class_sample_count])
x_train, x_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3)

model = create_model(input_length=supposed_feature_duration)
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(
    x_train.reshape(len(x_train), supposed_feature_duration, 1), y_train,
    validation_data=(x_test.reshape(len(x_test), supposed_feature_duration, 1), y_test),
    epochs=20, steps_per_epoch=10, validation_steps=10, shuffle=True
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")

acc = history.history["acc"]
val_acc = history.history["val_acc"]
plt.plot(acc, val_acc)
plt.show()
