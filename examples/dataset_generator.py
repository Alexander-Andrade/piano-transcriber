import pretty_midi
import librosa
from examples.midi_notes import MidiNotes
from shared import *
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.ndimage.filters import median_filter


CORR_INTERVAL = (-30, 30)
SLICE_LENGTH = 512


def cqt_features(audio_filename):
    y, sr = librosa.load(audio_filename, SR)
    y_mono = librosa.to_mono(y)
    spectrum = np.abs(librosa.cqt(y_mono, sr=sr))
    spectrum = np.transpose(spectrum)
    return normalize(spectrum)


def piano_roll_pretty(midi_filename, frames_total):
    midi = pretty_midi.PrettyMIDI(midi_filename)
    frames_timestamps = librosa.core.frames_to_time(np.arange(frames_total), SR)
    roll = np.transpose(midi.get_piano_roll(times=frames_timestamps)[FIRST_NOTE_MIDI_NUM:LAST_NOTE_MIDI_NUM])
    return np.where(roll > 0, 1., 0.)


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
    #     spectrum = cqt_features("samples/{0}.mp3".format(name))
    #     roll = piano_roll_pretty("samples/{0}.mid".format(name), spectrum.shape[0])
    #
    #     np.save("datasets/features_{0}.npy".format(name), spectrum)
    #     np.save("datasets/labels_{0}.npy".format(name), roll)
    #
    #     print("sample {0} added to dataset".format(name))

    name = 'mz_311_1'
    spectrum = np.load("datasets/features_{0}.npy".format(name))
    # roll = np.load("datasets/labels_{0}.npy".format(name))
    roll = piano_roll('samples/mz_311_1.mid', spectrum.shape[0])
    start_frame = 19000
    slice_length = 512
    corr_interval = (-50, 50)
    #
    features_slice, labels_slice, shift, corr_best, corr = train_slice(spectrum, roll,
                                                            start_frame=start_frame,
                                                            slice_length=slice_length,
                                                            corr_interval=corr_interval)
    print("shift: {0}, corr: {1}".format(shift, corr_best))

    # best_shift = -3
    # labels_slice = shifted_slice(roll, start_frame + best_shift, start_frame + slice_length + best_shift)

    shifts = np.arange(corr_interval[0], corr_interval[1])
    extrema = argrelextrema(corr, np.greater)
    print(list(zip(np.take(shifts, extrema[0]).tolist(), np.take(corr, extrema[0]).tolist())))

    l_sum = normalize(median_filter(labels_slice.sum(axis=1), size=3))

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
