import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.callbacks import TensorBoard
from shared import *
from constants import *
from data_generator import DataGenerator
import yaml
import time


n_epoch = 1
batch_size = 1
n_frames = 512
n_lstm_neurons = 300
n_features = 84

file = open("data_info.yaml", 'r')
train_info = yaml.load(file)
gen = DataGenerator(info=train_info, n_frames=n_frames, batch_size=batch_size)

model = Sequential()
model.add(LSTM(n_lstm_neurons,
               input_shape=(n_frames, n_features),
               return_sequences=True))
model.add(TimeDistributed(Dense(N_NOTES)))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

tb_callback = TensorBoard(log_dir='D:/tensorboard_logs/{}'.format(time.time()),
                          batch_size=batch_size,
                          update_freq='batch'
                          )

model.fit_generator(generator=gen,
                    callbacks=[tb_callback])

# for x, y in gen:
#     x_train = x.reshape(batch_size, n_frames, x.shape[1])
#     y_train = y.reshape(batch_size, n_frames, N_NOTES)
#     ##print(model.train_on_batch(x_train, y_train))
#     model.fit(x_train, y_train,
#               epochs=n_epoch,
#               batch_size=batch_size,
#               callbacks=[tb_callback])

