import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from shared import *
import librosa
import time


def correlate(features, midi, start, end, window):
    corr = np.zeros(shape=window)
    x = features[start:end]

    for i in range(0, window):
        frames_timestamps = librosa.core.frames_to_time(np.arange(start, end), 44100)
        frames_timestamps = frames_timestamps + i * 0.001
        roll = np.transpose(midi.get_piano_roll(times=frames_timestamps))
        y = normalize(roll)

        x_sum = x.sum(axis=1)
        y_sum = y.sum(axis=1)

        corr[i] = np.correlate(x_sum, y_sum)[0]

    return corr


def roll_slice(midi, start, end, shift=0.):
    frames_timestamps = librosa.core.frames_to_time(np.arange(start, end), 44100)
    frames_timestamps = frames_timestamps - shift * 0.001
    roll = np.transpose(midi.get_piano_roll(times=frames_timestamps))
    return np.where(roll > 0, 1., 0.)

start = 0
end = 512

midi = pretty_midi.PrettyMIDI('../samples/brahms_opus1_1.mid')

# name = samples_names()[0]
features = np.load("../datasets/features_brahms_opus1_1.npy")

t1 = time.time()
# corr = correlate(features, midi, start, end, 1400)
# shift = corr.argmax()

shift = 0
print(shift)

roll = normalize(roll_slice(midi, start, end, shift))
features_slice = normalize(features[start:end])

print("time: {0}".format(time.time()-t1))

f, axes = plt.subplots(1, 2)
# axes[0][0].imshow(roll)
# axes[0][1].imshow(features_slice)
# axes[1][0].plot(roll.sum(axis=1))
# axes[1][0].plot(features_slice.sum(axis=1))
# axes[1][1].plot(corr)
axes[0].imshow(features_slice)
axes[1].imshow(roll)
plt.show()
