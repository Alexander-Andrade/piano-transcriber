import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.callbacks import TensorBoard, Callback, TerminateOnNaN, ReduceLROnPlateau
from shared import *
from constants import *
from data_generator import DataGenerator
from keras.models import model_from_json
import time


n_epoch = 1
batch_size = 1
n_frames = 512
n_lstm_neurons = 300
n_features = 84


class SaveCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        model_json = self.model.to_json()
        name = "{0}_{1}".format('rmsprop_sigmoid_full_dataset', logs.get('loss'))
        with open("trained_models/{0}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("trained_models/{0}.h5".format(name))


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
                  optimizer='rmsprop')

    return model


def save_model(model, name):
    model_json = model.to_json()
    with open("trained_models/{0}.json".format(name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("trained_models/{0}.h5".format(name))


train_generator = DataGenerator.from_file("train.yaml", n_frames=n_frames, batch_size=batch_size)
validation_generator = DataGenerator.from_file("validation.yaml", n_frames=n_frames, batch_size=batch_size)

# model = rebuild_model("adam_sigmoid_full_dataset_0.06400026078128519")

model = Sequential()
model.add(LSTM(200,
               dropout=0.5,
               recurrent_dropout=0.5,
               input_shape=(n_frames, n_features),
               return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(N_NOTES, activation='sigmoid')))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

tb_callback = TensorBoard(log_dir='D:/tensorboard_logs/{}'.format(time.time()),
                          batch_size=batch_size,
                          update_freq='batch'
                          )

terminateOnNaN = TerminateOnNaN()

save_cb = SaveCallback()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1)

model.fit_generator(generator=train_generator,
                    validation_data=validation_generator,
                    callbacks=[tb_callback, terminateOnNaN, save_cb, reduce_lr],
                    # steps_per_epoch=10,
                    epochs=10)

