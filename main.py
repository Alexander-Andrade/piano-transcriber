import numpy as np
import librosa
import matplotlib.pyplot as plt
import time
from models.midi_model import MidiNotes

y, sr = librosa.load("samples/original.mp3", 44100)
y_mono = librosa.to_mono(y)
spectrum = librosa.core.stft(y_mono)

midi_notes = MidiNotes("samples/midi.mid")
#%%

start = time.time()

train_set_len = spectrum.shape[1]
frames_timestamps = librosa.core.frames_to_time(np.arange(train_set_len), 44100)
frames_timestamps *= 1000


#notes_items = midi_notes.notes.items()
#midi_notes_dict = midi_notes.__dict__
#labels = [ midi_notes.notes_at(timestamp)  for timestamp in frames_timestamps]

train = np.zeros((train_set_len, 88), dtype=np.bool)

for i, timestamp in enumerate(frames_timestamps):
    notes = np.array(midi_notes.notes_at(timestamp), dtype=np.dtype('u4'))
    notes_indexes = notes - 21
    np.put(train[i], notes_indexes, True)
    
end = time.time()
elapced = end - start