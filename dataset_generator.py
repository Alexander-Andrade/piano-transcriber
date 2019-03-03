import numpy as np
import librosa
from midi_notes import MidiNotes
from constants import *
from shared import *
import matplotlib.pyplot as plt
import time


CORR_INTERVAL = (-30, 30)
SLICE_LENGTH = 512


def cqt_features(audio_filename):
    y, sr = librosa.load(audio_filename, SR)
    y_mono = librosa.to_mono(y)
    spectrum = np.abs(librosa.cqt(y_mono, sr=sr, fmin=librosa.note_to_hz('C0')))
    spectrum = np.transpose(spectrum)
    return normalize(spectrum)


def piano_roll(midi_filename, frames_total):
    midi = MidiNotes(midi_filename)
    frames_timestamps = librosa.core.frames_to_time(np.arange(frames_total), SR)
    # sec to ms
    frames_timestamps *= 1000
    roll = np.zeros((frames_total, N_NOTES), dtype=np.float)

    for i, timestamp in enumerate(frames_timestamps):
        notes = np.array(midi.notes_at(timestamp), dtype=np.dtype('u4'))
        notes_indexes = notes - FIRST_NOTE_MIDI_NUM
        np.put(roll[i], notes_indexes, 1.)

    return roll


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
    # for name in samples_names():
    #     cqt_features(features_filename="samples/{0}.mp3".format(name))
    #     print("sample {0} added to dataset".format(name))

    name = samples_names()[2]
    spectrum = np.load("datasets/features_{0}.npy".format(name))
    roll = np.load("datasets/labels_{0}.npy".format(name))

    start_frame = 43000
    slice_length = 512
    corr_interval = (-150, 50)

    features_slice, labels_slice, shift, corr_best, corr = train_slice(spectrum, roll,
                                                            start_frame=start_frame,
                                                            slice_length=slice_length,
                                                            corr_interval=corr_interval)
    print("shift: {0}, corr: {1}".format(shift, corr_best))
    print("metric: {0}".format(corr_best / features_slice.sum() * labels_slice.sum()))
    # best_shift = -78
    # labels_slice = shifted_slice(roll, start_frame + best_shift, start_frame + slice_length + best_shift)
    f, axes = plt.subplots(2, 2)
    #
    axes[0][0].imshow(features_slice)
    axes[0][1].imshow(labels_slice)
    axes[1][0].plot(normalize(features_slice.sum(axis=1)))
    axes[1][0].plot(normalize(labels_slice.sum(axis=1)))
    axes[1][1].plot(corr)
    plt.show()
