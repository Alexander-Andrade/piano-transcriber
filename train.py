import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.callbacks import TensorBoard
from shared import *
from constants import *
from data_generator import DataGenerator
import time
from keras.models import model_from_json


n_epoch = 1
batch_size = 1
n_frames = 512
n_lstm_neurons = 300
n_features = 84


def save_model(model, name):
    model_json = model.to_json()
    with open("trained_models/{0}.json".format(name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("trained_models/{0}.h5".format(name))

def rebuild_model(filename):
    # load json and create model
    file = open('trained_models/{0}.json'.format(filename), 'r')
    model_json = file.read()
    file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("trained_models/{0}.h5".format(filename))

    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

train_generator = DataGenerator.from_file("train.yaml", n_frames=n_frames, batch_size=batch_size)
validation_generator = DataGenerator.from_file("validation.yaml", n_frames=n_frames, batch_size=batch_size)

model = Sequential()
model.add(LSTM(300,
               dropout=0.2,
               recurrent_dropout=0.2,
               input_shape=(n_frames, n_features),
               return_sequences=True))
model.add(LSTM(88,
               dropout=0.2,
               recurrent_dropout=0.2,
               return_sequences=True))
model.add(TimeDistributed(Dense(N_NOTES, activation='sigmoid')))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

tb_callback = TensorBoard(log_dir='D:/tensorboard_logs/{}'.format(time.time()),
                          batch_size=batch_size,
                          update_freq='batch'
                          )

model.fit_generator(generator=train_generator,
                    validation_data=validation_generator,
                    callbacks=[tb_callback],
                    epochs=1)

save_model(model, 'multi_layer_lstm')
