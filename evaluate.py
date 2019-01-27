import numpy as np
from keras.models import model_from_json


spectrum = np.load('spectrum.npy')
labels = np.load('train.npy')

n_timesteps = spectrum.shape[0]
n_features = spectrum.shape[1]

n_classes = 88

n_epoch = 1
batch_size = 1024
n_lstm_neurons = 1024

x_train = spectrum.reshape(1, n_timesteps, n_features)
y_train = labels.reshape(1, n_timesteps, n_classes)

# load json and create model
file = open('model.json', 'r')
model_json = file.read()
file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("model.h5")

# evaluate loaded model on test data
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
score = model.evaluate(x_train, y_train)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
