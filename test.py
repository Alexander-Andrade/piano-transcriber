import numpy as np
from keras.models import model_from_json
from constants import *
from data_generator import DataGenerator
import matplotlib.pyplot as plt


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
                  optimizer='adam')

    return model


test_generator = DataGenerator.from_file("test.yaml", n_frames=512, batch_size=1)
model = rebuild_model("adam_sigmoid")


for x, y in test_generator:
    prediction = model.predict(x)

    x = x.reshape(x.shape[1], x.shape[2])
    y = y.reshape(y.shape[1], y.shape[2])
    prediction = prediction.reshape(prediction.shape[1], prediction.shape[2])

    # # TODO remove after test
    # x = x.reshape(x.shape[0], x.shape[2])
    # y = y.reshape(y.shape[0], y.shape[2])
    # prediction = prediction.reshape(prediction.shape[0], prediction.shape[2])

    grid = plt.GridSpec(2, 6, bottom=0.04, top=0.98, left=0.02, right=0.98)

    axes = plt.subplot(grid[:, 0])
    axes.imshow(x)

    axes = plt.subplot(grid[:, 1])
    axes.imshow(y)

    axes = plt.subplot(grid[:, 2])
    axes.imshow(prediction)

    axes = plt.subplot(grid[:, 3])
    prediction = np.where(prediction > -0.3, 1., 0.)
    axes.imshow(prediction)

    print(prediction.shape)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()
