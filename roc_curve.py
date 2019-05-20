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


x, y = next(test_generator)
prediction = model.predict(x)

x = x.reshape(x.shape[1], x.shape[2])
y = y.reshape(y.shape[1], y.shape[2])
prediction = prediction.reshape(prediction.shape[1], prediction.shape[2])


note_numbers = [30, 34, 37, 46, 47, 48, 49]

for note_num in note_numbers:
    note_pred = prediction[:, note_num]
    note_y = y[:, note_num]
    # note_pred = (prediction[:, 30] - np.min(prediction))/(np.max(prediction) - np.min(prediction))

    auc = roc_auc_score(note_y, note_pred)
    fpr, tpr, thresholds = roc_curve(note_y, note_pred)
    print(note_num)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# grid = plt.GridSpec(2, 6, bottom=0.04, top=0.98, left=0.02, right=0.98)
#
# axes = plt.subplot(grid[:, 0])
# axes.imshow(x)
#
# axes = plt.subplot(grid[:, 1])
# axes.imshow(y)
#
# axes = plt.subplot(grid[:, 2])
# axes.imshow(prediction)
#
# print(prediction.shape)
#
# mng = plt.get_current_fig_manager()
# mng.window.state('zoomed')
#
# plt.show()