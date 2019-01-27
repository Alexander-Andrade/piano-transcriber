import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from shared import *
from constants import *


features = []
labels = []

file_names = samples_names()[:7]

for name in file_names:
    features.append(np.load("datasets/features_{0}.npy".format(name)))
    features.append(np.load("datasets/labels_{0}.npy".format(name)))


features = np.concatenate(features)
labels = np.concatenate(labels)

n_timesteps = features.shape[0]
n_features = features.shape[1]

x_train = features.reshape(1, n_timesteps, n_features)
y_train = labels.reshape(1, n_timesteps, N_NOTES)


n_epoch = 1
batch_size = 1024
n_lstm_neurons = 1024


model = Sequential()
model.add(LSTM(n_lstm_neurons, input_shape=(n_timesteps, n_features), return_sequences=True))
model.add(TimeDistributed(Dense(N_NOTES)))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=n_epoch, batch_size=batch_size)


model_json = model.to_json()
with open("trained_models/simple_lstm.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("trained_models/simple_lstm.h5")
