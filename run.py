from keras.models import model_from_json
from scipy.io import wavfile as wf
from prepare_model import preprocess_data, pattern_filename
from feature_extraction import find_features_cross_corr
import numpy as np


with open('/home/yakov/PycharmProjects/seld-net-master/model.json') as model_file:
    loaded_model = model_file.read()
model = model_from_json(loaded_model)
model.load_weights("model.h5")
filename = '/home/yakov/sound-search/ESC-50-master/audio/1-115920-A-22.wav'
_, series = wf.read(filename)
series, pattern = preprocess_data(series, np.load(pattern_filename))
one_file_features, indices = find_features_cross_corr(series, pattern)
prediction = model.predict(one_file_features)
