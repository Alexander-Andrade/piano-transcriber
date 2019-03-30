import librosa
from shared import *
import pretty_midi
import os


def cqt_features(audio_filename):
    y, sr = librosa.load(audio_filename, SR)
    y_mono = librosa.to_mono(y)
    spectrum = np.abs(librosa.cqt(y_mono, sr=sr))
    spectrum = np.transpose(spectrum)
    return normalize(spectrum)


def piano_roll(midi_filename, frames_total):
    midi = pretty_midi.PrettyMIDI(midi_filename)
    frames_timestamps = librosa.core.frames_to_time(np.arange(frames_total), SR)
    roll = np.transpose(midi.get_piano_roll(times=frames_timestamps)[FIRST_NOTE_MIDI_NUM:LAST_NOTE_MIDI_NUM])
    return np.where(roll > 0, 1., 0.)


if __name__ == "__main__":
    for name, ext in samples_names():
        if not os.path.isfile("datasets/features_{0}.npy".format(name)):
            spectrum = cqt_features("samples/{0}".format(name+ext))
            roll = piano_roll('samples/{0}.mid'.format(name), spectrum.shape[0])

            np.save("datasets/features_{0}.npy".format(name), spectrum)
            np.save("datasets/labels_{0}.npy".format(name), roll)

            print("{0} was extracted".format(name))
