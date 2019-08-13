from keras.models import Sequential
from keras import layers


def create_model(input_length):
    model = Sequential()
    model.add(layers.Conv1D(32, 7, activation='relu', input_shape=(input_length, 1)))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
