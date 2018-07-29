import midi
import numpy as np
from bisect import bisect_left


class MidiNotes:

    def __init__(self, filename):
        self.filename = filename
        self.pattern = midi.read_midifile(filename)
        self.pattern.make_ticks_abs()
        self.__parse()

    def __parse(self):
        self.notes = {}
        for event in self.pattern[0]:
            event_type = type(event)
            if event_type != midi.events.NoteOnEvent and event_type != midi.events.NoteOffEvent:
                continue
            self.notes.setdefault(event.pitch, []).append(event.tick)

    def notes_at(self, ms):
        hits = []
        for pitch, ticks in self.notes.items():
            pos = bisect_left(ticks, ms)
            if (pos & 1) == 1 and ms >= ticks[pos - 1]:
                hits.append(pitch)
        return hits
