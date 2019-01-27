import numpy as np
import os


def normalize(v):
    min_v = np.min(v)
    max_v = np.max(v)
    return (v - min_v)/(max_v - min_v)


def samples_names():
    return [os.path.splitext(name)[0] for name in os.listdir('samples') if name.endswith('.mp3')]
