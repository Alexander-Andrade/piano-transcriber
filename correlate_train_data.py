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


def stat(name, slice_length=512, corr_interval=(-30, 30)):
    spectrum = np.load("../datasets/features_{0}.npy".format(name))
    roll = np.load("../datasets/labels_{0}.npy".format(name))
    hits = {}

    for i in range(0, spectrum.shape[0], slice_length):
        features_slice, labels_slice, shift, corr_best, corr = train_slice(
            spectrum,
            roll,
            start_frame=i,
            slice_length=slice_length,
            corr_interval=corr_interval)

        shifts = np.arange(corr_interval[0], corr_interval[1])
        extrema = argrelextrema(corr, np.greater)

        ext_shifts = np.take(shifts, extrema[0]).tolist()
        ext_values = np.take(corr, extrema[0]).tolist()

        print("f: {0} , extrema: {1}".format(i, list(zip(ext_shifts, ext_values))))

        for ext_shift in ext_shifts:
            if ext_shift in hits:
                hits[ext_shift].append(i)
            else:
                hits[ext_shift] = [i]

    print('\nhits')
    sorted_hits_keys = sorted(hits.keys(), key=lambda k: len(hits[k]), reverse=True)
    for key in sorted_hits_keys:
        print("key: {0}, len: {1},frames: {2}".format(key, len(hits[key]), hits[key]))

    for key in sorted_hits_keys[:3]:
        print("key: {0}".format(key))
        for frame in hits[key]:
            print("- [{0}, {1}]".format(frame, key))


def manual_look():
    name = 'beethoven_hammerklavier_3'
    spectrum = np.load("../datasets/features_{0}.npy".format(name))
    roll = np.load("../datasets/labels_{0}.npy".format(name))

    print("frames: {0}".format(spectrum.shape[0]))

    start_frame = 12000
    slice_length = 512
    corr_interval = (-50, 30)

    features_slice, labels_slice, shift, corr_best, corr = train_slice(spectrum, roll,
                                                            start_frame=start_frame,
                                                            slice_length=slice_length,
                                                            corr_interval=corr_interval)
    print("shift: {0}, corr: {1}".format(shift, corr_best))

    best_shift = -3
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
if __name__ == "__main__":
    stat('beethoven_hammerklavier_3')
    #manual_look()
