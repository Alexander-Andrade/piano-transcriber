import numpy as np
from keras.models import model_from_json
from constants import *
from data_generator import DataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize


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
model = rebuild_model("adam_sigmoid_full_dataset_0.06281912852946044")


note = 30

hits_sum = 0
total_sum = 0

for x, y in test_generator:
    prediction = model.predict(x)

    x = x.reshape(x.shape[1], x.shape[2])
    y = y.reshape(y.shape[1], y.shape[2])
    prediction = prediction.reshape(prediction.shape[1], prediction.shape[2])
    prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
    prediction = np.where(prediction > 0.5, 1, 0)
    hits = 0
    total = 0
    for i in range(512):
        if y[:, note][i] == prediction[:, note][i]:
            hits += 1
    total += 512
    print(hits*100/total)
    hits_sum += hits
    total_sum += total


print("total: {0} %".format(hits_sum/total_sum))
