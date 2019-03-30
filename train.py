import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.callbacks import TensorBoard
from shared import *
from constants import *
from data_generator import DataGenerator
import time
from keras.models import model_from_json
import tensorflow as tf


n_epoch = 2
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


# Transform train_on_batch return value
# to dict expected by on_batch_end callback
def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result


train_generator = DataGenerator.from_file("train.yaml", n_frames=n_frames, batch_size=batch_size)
# validation_generator = DataGenerator.from_file("validation.yaml", n_frames=n_frames, batch_size=batch_size)

model = Sequential()
model.add(LSTM(N_NOTES,
               dropout=0.1,
               recurrent_dropout=0.1,
               batch_input_shape=(batch_size, n_frames, n_features),
               return_sequences=True, stateful=True))
model.add(LSTM(N_NOTES,
               dropout=0.1,
               recurrent_dropout=0.1,
               return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(N_NOTES)))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

tb_callback = TensorBoard(log_dir='D:/tensorboard_logs/',
                          histogram_freq=0,
                          batch_size=batch_size,
                          write_graph=True,
                          write_grads=True,

                          )

tb_callback.set_model(model)

for i in range(n_epoch):
    for batch_id, (x, y) in enumerate(train_generator):
        logs = model.train_on_batch(x, y)
        print(logs)
        tb_callback.on_epoch_end(batch_id, named_logs(model, logs))
        if train_generator.sequence_reseted():
            model.reset_states()
tb_callback.on_train_end(None)
# save_model(model, 'multi_layer_lstm')
