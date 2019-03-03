import midi
import midi.timeresolver as tres
from bisect import bisect_left
from shared import *

class MidiNotes:

    def __init__(self, filename):
        self.filename = filename
        self.pattern = midi.read_midifile(filename)
        self.pattern.make_ticks_abs()
        self.time_resolver = tres.TimeResolver(self.pattern)
        self.__parse()

    def __parse(self):
        self.notes = {}
        for event in self.pattern[0]:
            event_type = type(event)
            if event_type != midi.events.NoteOnEvent and event_type != midi.events.NoteOffEvent:
                continue
            milliseconds = self.time_resolver.tick2ms(event.tick)
            self.notes.setdefault(event.pitch, []).append(milliseconds)

    def notes_at(self, ms):
        hits = []
        for pitch, milliseconds in self.notes.items():
            pos = bisect_left(milliseconds, ms)
            if (pos & 1) == 1 and ms >= milliseconds[pos - 1]:
                hits.append(pitch)
        return hits


if __name__ == "__main__":
    midi = MidiNotes("samples/{0}.mid".format(samples_names()[0]))
    print(midi.notes_at(30*1000))
