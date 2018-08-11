class TrainSetBuilder:
    
    def __init__(self, train_pair, sampling_rate=44100):
        self.train_pair = train_pair
        self.sampling_rate = sampling_rate
        
        
    def __x(self):
        audio, sr = librosa.load(self.train_pair[0], self.sampling_rate)
        audio_mono = librosa.to_mono(audio)
        audio_spectrum = librosa.core.stft(audio_mono)
        return audio_spectrum
    
    def __y(self, x_size):
        midi_notes = MidiNotes(self.train_pair[1])
        frames_timestamps = librosa.core.frames_to_time(np.arange(x_size), self.sampling_rate)
        # sec to ms
        frames_timestamps *= 1000
        y = np.zeros((train_set_len, 88), dtype=np.bool)
        for i, timestamp in enumerate(frames_timestamps):
            notes = np.array(midi_notes.notes_at(timestamp), dtype=np.dtype('u4'))
            notes_indexes = notes - 21
            np.put(train[i], notes_indexes, True)
        return y
    
    def generate(self):
        x = self.__x()
        x_size = x.shape[1]
        y = self.__y(x_size)
        return (x, y)
        
        