import datetime
from keras.models import Sequential
from keras import layers
from keras.models import model_from_json
import matplotlib.pyplot as plt

from feature_extraction import data_base_path


class KerasModelWrapper:
    def __init__(self, input_length=None, model_json_filename=None, model_weights_filename=None):
        self.input_length = input_length
        self.model_json_file = model_json_filename
        self.model_weights_file = model_weights_filename
        if self.model_json_file:
            self.model = self._load_model_from_file()
        else:
            self.model = self._create_model()

    def fit_model(self, x_train, x_test, y_train, y_test, feature_size, epochs=20, steps=30, val_steps=30):
        self.model.summary()
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = self.model.fit(
            x_train.reshape(len(x_train), feature_size, 1), y_train,
            validation_data=(x_test.reshape(len(x_test), feature_size, 1), y_test),
            epochs=epochs, steps_per_epoch=steps, validation_steps=val_steps, shuffle=True
        )
        return history

    @staticmethod
    def show_learning_curve(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.plot(acc, 'g', val_acc, 'b')
        plt.legend(['acc', 'val_acc'], loc='upper left')
        plt.show()

    def _create_model(self):
        model = Sequential()
        model.add(layers.Conv1D(32, 7, activation='relu', input_shape=(self.input_length, 1)))
        model.add(layers.MaxPooling1D(5))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv1D(32, 5, activation='relu'))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def _load_model_from_file(self):
        model_file = open(self.model_json_file)
        loaded_model = model_file.read()
        model = model_from_json(loaded_model)
        model.load_weights(self.model_weights_file)
        return model

    def save_model(self):
        model_json = self.model.to_json()
        unique_string = str(datetime.datetime.now())
        json_file = open(data_base_path + 'model' + unique_string + '.json', 'w')
        json_file.write(model_json)
        self.model.save(data_base_path + 'model' + unique_string + '.h5')
