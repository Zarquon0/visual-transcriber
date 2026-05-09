#!/usr/bin/env python3
"""Live piano via keyboard. Q–] = chromatic octave. Esc = stop and save MIDI."""

import time
from datetime import datetime
from pathlib import Path

from pynput import keyboard as pynput_kb

from transcribe import Key, Note, Transcriber

import ctypes, ctypes.util                                                                                                                                    

FPS = 30
OCTAVE = 4

# Q W E R T Y U I O P [  ]
# C C# D D# E F F# G G# A A# B
KEY_MAP: dict[str, Key] = {
    'q': Key(Note.C,  OCTAVE),
    'w': Key(Note.CS, OCTAVE),
    'e': Key(Note.D,  OCTAVE),
    'r': Key(Note.DS, OCTAVE),
    't': Key(Note.E,  OCTAVE),
    'y': Key(Note.F,  OCTAVE),
    'u': Key(Note.FS, OCTAVE),
    'i': Key(Note.G,  OCTAVE),
    'o': Key(Note.GS, OCTAVE),
    'p': Key(Note.A,  OCTAVE),
    '[': Key(Note.AS, OCTAVE),
    ']': Key(Note.B,  OCTAVE),
}

_LABEL = {
    'q': 'C', 'w': 'C#', 'e': 'D', 'r': 'D#', 't': 'E',
    'y': 'F', 'u': 'F#', 'i': 'G', 'o': 'G#', 'p': 'A',
    '[': 'A#', ']': 'B',
}

def main() -> None:
    pressed: set[str] = set()
    running = True

    def on_press(key: pynput_kb.Key) -> None:
        nonlocal running
        try:
            char = key.char.lower()  # type: ignore[union-attr]
            if char in KEY_MAP:
                pressed.add(char)
        except AttributeError:
            if key == pynput_kb.Key.esc:
                running = False

    def on_release(key: pynput_kb.Key) -> None:
        try:
            char = key.char.lower()  # type: ignore[union-attr]
            pressed.discard(char)
        except AttributeError:
            pass

    print("=" * 52)
    print("  Live Piano — keyboard map (all octave 4)")
    print("  Q  W  E  R  T  Y  U  I  O  P  [  ]")
    print("  C  C# D  D# E  F  F# G  G# A  A# B")
    print("  Press Esc to stop and export MIDI.")
    print("=" * 52)

    transcriber = Transcriber(fps=FPS)
    listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    frame_duration = 1.0 / FPS
    while running:
        t0 = time.perf_counter()
        current_keys = [KEY_MAP[k] for k in pressed]
        transcriber.update(current_keys)
        if current_keys:
            names = " ".join(_LABEL[k] for k in sorted(pressed))
            print(f"\r  Playing: {names:<30}", end="", flush=True)
        else:
            print(f"\r  {'(silent)':<40}", end="", flush=True)
        sleep = frame_duration - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)

    listener.stop()
    print()

    Path("midi_outs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"midi_outs/{timestamp}.mid"
    transcriber.save_midi(out_path)
    print(f"  Saved MIDI → {out_path}")

if __name__ == "__main__":
    _app_svc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('ApplicationServices'))                                                                           
    if not _app_svc.AXIsProcessTrusted():
        print("WARNING: This process is NOT trusted for Accessibility. Keyboard events will be silently ignored.")
    else: 
        print("Trusted!")
    main()
