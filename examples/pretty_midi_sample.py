import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from shared import *
import librosa
import time


def piano_roll(midi_filename, frames_total):
    midi = pretty_midi.PrettyMIDI(midi_filename)
    frames_timestamps = librosa.core.frames_to_time(np.arange(frames_total), SR)
    roll = np.transpose(midi.get_piano_roll(times=frames_timestamps)[FIRST_NOTE_MIDI_NUM:LAST_NOTE_MIDI_NUM])
    return np.where(roll > 0, 1., 0.)


def shifted_slice(v, start, end):
    v_length = v.shape[0]
    length = end - start

    if start < 0:
        res = np.zeros(shape=(length, N_NOTES))
        res[-start:length] = v[0:length+start]
        return res
    elif end > v_length - 1:
        res = np.zeros(shape=(length, N_NOTES))
        res[0:v_length-start] = v[start:v_length]
        return res
    else:
        return v[start:end]


def train_slice(features, labels, start_frame, slice_length, corr_interval):
    shifts = np.arange(corr_interval[0], corr_interval[1])
    corr = np.zeros(shape=corr_interval[1] - corr_interval[0])

    features_slice = features[start_frame: start_frame + slice_length]
    features_sum = normalize(features_slice.sum(axis=1))

    for i, shift in enumerate(shifts):
        labels_slice = shifted_slice(labels, start_frame+shift, start_frame+slice_length+shift)
        labels_sum = normalize(labels_slice.sum(axis=1))

        corr[i] = np.correlate(features_sum, labels_sum)[0]

    best_ind = corr.argmax()
    best_shift = shifts[best_ind]
    best_correlation = corr[best_ind]

    best_labels_slice = shifted_slice(labels, start_frame+best_shift, start_frame+slice_length+best_shift)
    return features_slice, best_labels_slice, best_shift, best_correlation, corr


if __name__ == "__main__":

    # name = samples_names()[0]
    spectrum = np.load("../datasets/features_brahms_opus1_1.npy")
    roll = piano_roll('../samples/brahms_opus1_1.mid', spectrum.shape[0])

    start_frame = 0
    slice_length = 512
    corr_interval = (-150, 50)

    features_slice, labels_slice, shift, corr_best, corr = train_slice(spectrum, roll,
                                                            start_frame=start_frame,
                                                            slice_length=slice_length,
                                                            corr_interval=corr_interval)
    print("shift: {0}, corr: {1}".format(shift, corr_best))
    print("metric: {0}".format(corr_best / features_slice.sum() * labels_slice.sum()))

    f, axes = plt.subplots(2, 2)
    #
    axes[0][0].imshow(features_slice)
    axes[0][1].imshow(labels_slice)
    axes[1][0].plot(normalize(features_slice.sum(axis=1)))
    axes[1][0].plot(normalize(labels_slice.sum(axis=1)))
    axes[1][1].plot(corr)
    plt.show()
