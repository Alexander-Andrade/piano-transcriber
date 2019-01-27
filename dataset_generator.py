import numpy as np
import librosa
from midi_notes import MidiNotes
from constants import *
from shared import *
import os


class DatasetGenerator:

    def __init__(self, features_filename, labels_filename):
        self.features_filename = features_filename
        self.labels_filename = labels_filename
        self.n_frames = 0

    def x_data(self):
        y, sr = librosa.load(self.features_filename, SR)
        y_mono = librosa.to_mono(y)
        spectrum = np.abs(librosa.cqt(y_mono, sr=sr, fmin=librosa.note_to_hz('C0')))
        self.n_frames = spectrum.shape[1]

        return np.transpose(spectrum)
    
    def y_data(self):
        midi_notes = MidiNotes(self.labels_filename)

        frames_timestamps = librosa.core.frames_to_time(np.arange(self.n_frames), SR)
        # sec to ms
        frames_timestamps *= 1000

        labels = np.zeros((self.n_frames, N_NOTES), dtype=np.float)

        for i, timestamp in enumerate(frames_timestamps):
            notes = np.array(midi_notes.notes_at(timestamp), dtype=np.dtype('u4'))
            notes_indexes = notes - 21
            np.put(labels[i], notes_indexes, 1.)

        return labels

    def generate(self):
        filename = os.path.splitext(self.features_filename)[0]
        filename = filename.split('/')[-1]

        np.save("datasets/features_{0}.npy".format(filename), normalize(self.x_data()))
        np.save("datasets/labels_{0}.npy".format(filename), self.y_data())


if __name__ == "__main__":
    for name in samples_names():
        DatasetGenerator(features_filename="samples/{0}.mp3".format(name),
                         labels_filename="samples/{0}.mid".format(name)).generate()
