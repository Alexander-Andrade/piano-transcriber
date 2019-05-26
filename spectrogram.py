import numpy as np
import matplotlib.pyplot as plt
import librosa
from shared import *


def cqt_features(audio_filename, sr=44100, n_bins=84, bins_per_octave=12):
    y, _ = librosa.load(audio_filename, sr)
    y_mono = librosa.to_mono(y)
    spectrum = np.abs(librosa.cqt(y_mono, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave))
    spectrum = np.transpose(spectrum)
    return normalize(spectrum)


if __name__ == "__main__":
    # features = np.load('datasets/features_alb_esp4.npy')
    # labels = np.load('datasets/labels_alb_esp4.npy')

    features = cqt_features('samples/alb_esp4.ogg')
    features1 = cqt_features('samples/alb_esp4.ogg', n_bins=168, bins_per_octave=24)

    start = 0
    end = 512

    grid = plt.GridSpec(2, 6, bottom=0.04, top=0.98, left=0.02, right=0.98)

    axes = plt.subplot(grid[:, 0])
    axes.imshow(features[start:end])

    axes = plt.subplot(grid[:, 1])
    axes.imshow(features1[start:end])

    plt.show()
