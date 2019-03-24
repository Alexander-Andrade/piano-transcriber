import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from shared import *
from constants import *
import math
import time


def pad(arr, length):
    padded_arr = np.zeros(shape=(length, arr.shape[1]))
    padded_arr[:arr.shape[0]] = arr
    return padded_arr


def data_generator(filenames, timesteps_per_portion):
    for name in filenames:
        features = np.load("datasets/features_{0}.npy".format(name))
        labels = np.load("datasets/labels_{0}.npy".format(name))

        n_timesteps = features.shape[0]
        n_portions = int(math.ceil(n_timesteps / float(timesteps_per_portion)))

        for ind in range(n_portions):
            if ind == n_portions - 1:
                features_portion = pad(features[ind*timesteps_per_portion:], timesteps_per_portion)
                labels_portion = pad(labels[ind*timesteps_per_portion:], timesteps_per_portion)
                yield (features_portion, labels_portion, True)
            else:
                features_portion = features[ind*timesteps_per_portion:(ind+1)*timesteps_per_portion]
                labels_portion = labels[ind*timesteps_per_portion:(ind+1)*timesteps_per_portion]
                yield (features_portion, labels_portion, False)


def build_model(batch_size, n_timesteps, n_features):
    model = Sequential()
    model.add(LSTM(n_lstm_neurons,
                   batch_input_shape=(batch_size, n_timesteps, n_features),
                   return_sequences=True,
                   stateful=True))
    model.add(TimeDistributed(Dense(N_NOTES)))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


n_epoch = 1
batch_size = 1
n_lstm_neurons = 2048
n_timesteps = 1024
n_features = 84

if __name__ == "__main__":
    gen = data_generator(samples_names()[:2], n_timesteps)
    model = build_model(batch_size=1,
                        n_timesteps=n_timesteps,
                        n_features=n_features)

    for x_train, y_train, need_reset in gen:
        model.fit(x_train.reshape(batch_size, n_timesteps, n_features),
                  y_train.reshape(batch_size, n_timesteps, N_NOTES),
                  epochs=1,
                  batch_size=1,
                  shuffle=False)
        if need_reset:
            model.reset_states()
