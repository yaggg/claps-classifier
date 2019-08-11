from keras.models import model_from_json
from scipy.io import wavfile as wf
from prepare_model import preprocess_data, pattern_filename, data_base_path
from feature_extraction import find_features_cross_corr_slow
import numpy as np
import matplotlib.pyplot as plt


with open("model.json") as model_file:
    loaded_model = model_file.read()
model = model_from_json(loaded_model)
model.load_weights("model.h5")
filename = '/home/yakov/sound-search/ESC-50-master/audio/1-115920-A-22.wav'
# filename = '/home/yakov/Audio/2019-08-04-20:18:28.wav'
# filename = '/home/yakov/Documents/sum.wav'
# filename = '/home/yakov/Audio/2019-08-07-20:59:27.wav'
# filename = '/home/yakov/Audio/2019-08-09-08:36:07.wav'
_, series = wf.read(filename)
series = preprocess_data(series)
one_file_features, indices = find_features_cross_corr_slow(series, np.load(data_base_path + pattern_filename), 0.2)
prediction = model.predict(np.array(one_file_features).reshape((len(one_file_features), 10000, 1)))

plt.plot(series)

claps = np.zeros(len(series))
for index, prediction in zip(indices, prediction):
    if prediction > 0.1:
        claps[index: index + 10000] = 1000

plt.plot(claps * 20)
plt.show()

