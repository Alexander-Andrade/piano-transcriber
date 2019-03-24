import numpy as np
import os
from constants import *


def normalize(v):
    min_v = np.min(v)
    max_v = np.max(v)
    return (v - min_v)/(max_v - min_v)


def samples_names():
    return [os.path.splitext(name) for name in os.listdir('samples') if name.endswith('.mp3') or name.endswith('.ogg')]


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
