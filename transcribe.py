from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import fluidsynth
import mido

class Note(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    AS = "A#"
    CS = "C#"
    DS = "D#"
    FS = "F#"
    GS = "G#"

@dataclass
class Key: # <- THIS IS THE INTERFACE
    note: Note
    octave: int

_NOTE_OFFSETS: dict[Note, int] = {
    Note.C: 0, Note.CS: 1, Note.D: 2, Note.DS: 3,
    Note.E: 4, Note.F: 5, Note.FS: 6, Note.G: 7,
    Note.GS: 8, Note.A: 9, Note.AS: 10, Note.B: 11,
}

def _key_to_midi(key: Key) -> int:
    return 12 * (key.octave + 1) + _NOTE_OFFSETS[key.note]

@dataclass
class _NoteEvent:
    note: Note
    octave: int
    start_time: float  # seconds from first update call
    duration: float    # seconds

_SOUNDFONT = Path(__file__).parent / "sound_fonts" / "FluidR3_GM_GS.sf2"

class Transcriber:
    def __init__(self, fps: float = 30, audio_driver: str = "coreaudio"):
        self.fps = fps
        self._frame_duration = 1.0 / fps
        self._current_time = 0.0

        # (note, octave) -> time when note_on was fired
        self._active: dict[tuple[Note, int], float] = {}
        self._events: list[_NoteEvent] = []

        self._fs = fluidsynth.Synth()
        self._fs.start(driver=audio_driver)
        sfid = self._fs.sfload(str(_SOUNDFONT))
        self._fs.program_select(0, sfid, 0, 0)

    def update(self, keys: list[Key]) -> None:
        """Record and play keys pressed in this frame. Call once per frame."""
        current = {(k.note, k.octave) for k in keys}
        previous = set(self._active)

        for key_id in previous - current:
            start = self._active.pop(key_id)
            note, octave = key_id
            self._events.append(_NoteEvent(note, octave, start, self._current_time - start))
            self._fs.noteoff(0, _key_to_midi(Key(note, octave)))

        for key_id in current - previous:
            note, octave = key_id
            self._active[key_id] = self._current_time
            self._fs.noteon(0, _key_to_midi(Key(note, octave)), 100)

        self._current_time += self._frame_duration

    def save_midi(self, path: str) -> None:
        """Write a MIDI file from all recorded note events to the given path."""
        ticks_per_beat = 480
        tempo = 500_000  # microseconds per beat = 120 BPM

        def secs_to_ticks(secs: float) -> int:
            return int(secs / (tempo / 1_000_000) * ticks_per_beat)

        # Flush any notes still active at save time
        pending = [
            _NoteEvent(note, octave, start, self._current_time - start)
            for (note, octave), start in self._active.items()
        ]
        all_events = self._events + pending

        # Build (abs_tick, Message) pairs then sort and convert to delta times
        messages: list[tuple[int, mido.Message]] = []
        for ev in all_events:
            midi_num = _key_to_midi(Key(ev.note, ev.octave))
            messages.append((secs_to_ticks(ev.start_time),
                             mido.Message('note_on', note=midi_num, velocity=100, time=0)))
            messages.append((secs_to_ticks(ev.start_time + ev.duration),
                             mido.Message('note_off', note=midi_num, velocity=0, time=0)))

        messages.sort(key=lambda x: x[0])

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        prev_tick = 0
        for abs_tick, msg in messages:
            track.append(msg.copy(time=abs_tick - prev_tick))
            prev_tick = abs_tick

        mid.save(path)

    def __del__(self) -> None:
        try:
            for note, octave in list(self._active):
                self._fs.noteoff(0, _key_to_midi(Key(note, octave)))
            self._fs.delete()
        except Exception:
            pass
