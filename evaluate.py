import numpy as np
from keras.models import model_from_json
from constants import *

features = np.load('datasets/features_balakirew_islamei.npy')
labels = np.load('datasets/labels_balakirew_islamei.npy')

n_timesteps = features.shape[0]
n_features = features.shape[1]

x_train = features.reshape(1, n_timesteps, n_features)
y_train = labels.reshape(1, n_timesteps, N_NOTES)

# load json and create model
file = open('trained_models/simple_lstm.json', 'r')
model_json = file.read()
file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("trained_models/simple_lstm.h5")

# evaluate loaded model on test data
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())
print(x_train.shape)
print(y_train.shape)
# score = model.evaluate(x_train, y_train, batch_size=1)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

predictions = model.predict(x_train, batch_size=1)
print(predictions.shape)
