import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from shared import *

import time
from scipy.signal import argrelextrema
from scipy.ndimage.filters import median_filter


def train_slice(features, labels, start_frame, slice_length, corr_interval):
    shifts = np.arange(corr_interval[0], corr_interval[1])
    corr = np.zeros(shape=corr_interval[1] - corr_interval[0])

    features_slice = features[start_frame: start_frame + slice_length]
    features_sum = normalize(features_slice.sum(axis=1))

    for i, shift in enumerate(shifts):
        labels_slice = shifted_slice(labels, start_frame+shift, start_frame+slice_length+shift)
        labels_sum = median_filter(labels_slice.sum(axis=1), size=3)

        if labels_sum.min() != labels_sum.max():
            labels_sum = normalize(labels_sum)

        corr[i] = np.correlate(features_sum, labels_sum)[0]

    best_ind = corr.argmax()
    best_shift = shifts[best_ind]
    best_correlation = corr[best_ind]

    best_labels_slice = shifted_slice(labels, start_frame+best_shift, start_frame+slice_length+best_shift)
    return features_slice, best_labels_slice, best_shift, best_correlation, corr


if __name__ == "__main__":
    name = 'beethoven_opus22_1'
    spectrum = np.load("../datasets/features_{0}.npy".format(name))
    roll = np.load("../datasets/labels_{0}.npy".format(name))

    print("frames: {0}".format(spectrum.shape[0]))

    start_frame = 19000
    slice_length = 512
    corr_interval = (-100, 30)

    features_slice, labels_slice, shift, corr_best, corr = train_slice(spectrum, roll,
                                                            start_frame=start_frame,
                                                            slice_length=slice_length,
                                                            corr_interval=corr_interval)
    print("shift: {0}, corr: {1}".format(shift, corr_best))

    best_shift = -4
    labels_slice = shifted_slice(roll, start_frame + best_shift, start_frame + slice_length + best_shift)

    shifts = np.arange(corr_interval[0], corr_interval[1])
    extrema = argrelextrema(corr, np.greater)
    print(list(zip(np.take(shifts, extrema[0]).tolist(), np.take(corr, extrema[0]).tolist())))

    l_sum = (median_filter(labels_slice.sum(axis=1), size=3))
    if l_sum.max() != l_sum.min():
        l_sum = normalize(l_sum)


    grid = plt.GridSpec(2, 6, bottom=0.04, top=0.98, left=0.02, right=0.98)

    axes = plt.subplot(grid[:, 0])
    axes.imshow(features_slice)

    axes = plt.subplot(grid[:, 1])
    axes.imshow(labels_slice)

    axes = plt.subplot(grid[0, 2:])
    axes.plot(normalize(features_slice.sum(axis=1)))
    axes.plot(l_sum)

    axes = plt.subplot(grid[1, 2:])
    axes.plot(shifts, corr)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()
