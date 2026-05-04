from dataclasses import dataclass
from enum import StrEnum

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

