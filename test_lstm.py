import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
import time

shape = (1, 2048, 4)
features = np.random.rand(*shape)
labels = np.where(features > 0.5, 1., 0.)


model = Sequential()
model.add(LSTM(512,
               batch_input_shape=(512, shape[1] // 2, shape[2]),
               return_sequences=True,
               stateful=True))
model.add(TimeDistributed(Dense(shape[2])))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(features[:, :1024, :], labels[:, :1024, :], epochs=1, batch_size=512, shuffle=False)
model.fit(features[:, 1024:, :], labels[:, 1024:, :], epochs=1, batch_size=512, shuffle=False)

score = model.evaluate(features[:, :1024, :], labels[:, :1024, :], batch_size=512)

# model.train_on_batch(features[:, 1024:, :], labels[:, 1024:, :])
# model.train_on_batch(features[:, :1024, :], labels[:, :1024, :])
#
# score = model.test_on_batch(features[:, 1024:, :], labels[:, 1024:, :])

print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

