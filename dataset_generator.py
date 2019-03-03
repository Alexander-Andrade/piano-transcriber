import numpy as np
import librosa
from midi_notes import MidiNotes
from constants import *
from shared import *
import os
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy import signal
import time

def extract_features(features_filename):
    y, sr = librosa.load(features_filename, SR)
    y_mono = librosa.to_mono(y)
    spectrum = np.abs(librosa.cqt(y_mono, sr=sr, fmin=librosa.note_to_hz('C0')))
    spectrum = np.transpose(spectrum)

    filename = os.path.splitext(features_filename)[0]
    filename = filename.split('/')[-1]
    np.save("datasets/features_{0}.npy".format(filename), normalize(spectrum))


def train_set_slice(features, midi_notes, start_frame, n_frames, shift=0., scale=1.):
    frames_timestamps = librosa.core.frames_to_time(np.arange(start_frame, start_frame+n_frames), SR)
    # sec to ms
    frames_timestamps *= 1000

    labels = np.zeros((n_frames, N_NOTES), dtype=np.float)

    for i, timestamp in enumerate(frames_timestamps):
        t = timestamp*scale + shift
        notes = np.array(midi_notes.notes_at(t), dtype=np.dtype('u4'))
        notes_indexes = notes - 21
        np.put(labels[i], notes_indexes, 1.)

    features_slice = features[start_frame:start_frame+n_frames]

    return features_slice, labels


if __name__ == "__main__":
    # for name in samples_names():
    #     extract_features(features_filename="samples/{0}.mp3".format(name))
    #     print("sample {0} added to dataset".format(name))

    name = samples_names()[2]
    features = np.load("datasets/features_{0}.npy".format(name))
    midi_notes = MidiNotes("samples/{0}.mid".format(name))

    start = 0
    length = 512
    size = 1400
    corr = np.zeros(shape=size)
    t1 = time.time()
    for i in range(-size, 0):
        features_slice, labels_slice = train_set_slice(features, midi_notes, start, length, shift=i, scale=1)
        features_sum = normalize(features_slice.sum(axis=1))
        labels_sum = normalize(labels_slice.sum(axis=1))

        corr[size+i] = np.correlate(features_sum, labels_sum)[0]

    shift = corr.argmax() - size
    print(shift)
    features_slice, labels_slice = train_set_slice(features, midi_notes, start, length, shift=shift, scale=1)
    print("time: {0}".format(time.time()-t1))
    f, axes = plt.subplots(2, 2)

    axes[0][0].imshow(features_slice)
    axes[0][1].imshow(labels_slice)
    axes[1][0].plot(features_slice.sum(axis=1))
    axes[1][0].plot(labels_slice.sum(axis=1))
    axes[1][1].plot(corr)
    plt.show()
