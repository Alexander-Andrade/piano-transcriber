import numpy as np
import librosa
import matplotlib.pyplot as plt
import time
from models.midi_model import MidiNotes

y, sr = librosa.load("samples/  original.mp3", 44100)
y_mono = librosa.to_mono(y)
spectrum = librosa.core.stft(y_mono)

midi_notes = MidiNotes("samples/midi.mid")

start = time.time()

frames_timestamps = librosa.core.frames_to_time(range(0, spectrum.shape[1]), 44100)
frames_timestamps = [frames_timestamp * 1000 for frames_timestamp in frames_timestamps]
labels = [ midi_notes.notes_at(timestamp)  for timestamp in frames_timestamps]

end = time.time()
elapced = end - start