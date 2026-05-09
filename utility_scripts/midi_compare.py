"""midi_compare.py — score a MIDI transcription against a ground truth MIDI.

Pipeline
--------
1. Parse both files into (start, end, midi_note) note events.
2. Trim transcription: if any gap between consecutive notes is ≥
   pause_threshold (default 5 s), discard every note before the first
   such gap so spurious early presses don't pollute the comparison.
3. Normalize: shift each file so its first note starts at t=0.
4. Octave correction: if the first notes share the same pitch class but
   differ in octave, shift ALL transcription pitches by that octave
   delta so the systematic offset doesn't penalize accuracy.
5. One-to-one greedy matching: for each (GT note, transcription note) pair
   with the same pitch and |midpoint_gt − midpoint_trans| ≤ THRESHOLD,
   greedily assign the closest pair first.
6. Accuracy = matched GT notes / total GT notes.
7. Visualize: piano-roll with GT bars (blue), accurate trans bars (green),
   inaccurate trans bars (red), and unmatched GT bars (light blue outline).

Usage
-----
    uv run python utility_scripts/midi_compare.py ground_truth.mid transcription.mid
    uv run python utility_scripts/midi_compare.py gt.mid trans.mid --threshold 0.5
    uv run python utility_scripts/midi_compare.py gt.mid trans.mid --pause-threshold 3.0
    uv run python utility_scripts/midi_compare.py gt.mid trans.mid --no-plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mido
import numpy as np


MIDI_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class Note:
    start: float   # seconds
    end: float     # seconds
    pitch: int     # MIDI note number

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return self.end - self.start


def midi_note_name(pitch: int) -> str:
    return f"{MIDI_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def parse_midi(path: str | Path) -> list[Note]:
    """Return a list of Note events sorted by start time."""
    mid = mido.MidiFile(str(path))
    active: dict[int, float] = {}  # pitch → start_time
    notes: list[Note] = []
    t = 0.0
    for msg in mid:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = t
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                notes.append(Note(active.pop(msg.note), t, msg.note))
    # Flush any notes that never received a note_off (treat as ending at last event)
    for pitch, start in active.items():
        notes.append(Note(start, t, pitch))
    notes.sort(key=lambda n: n.start)
    return notes


def normalize_start(notes: list[Note]) -> list[Note]:
    """Shift all notes so the first note starts at t=0."""
    if not notes:
        return notes
    offset = notes[0].start
    return [Note(n.start - offset, n.end - offset, n.pitch) for n in notes]


def trim_before_long_pause(notes: list[Note], pause_threshold: float) -> list[Note]:
    """Keep only notes after the last inter-note gap >= pause_threshold seconds.

    The gap is measured from the end of one note to the start of the next, so
    overlapping / legato notes (negative gap) are never treated as a pause.
    Returns the original list unchanged if no qualifying gap is found.
    """
    last_pause_idx = None
    for i in range(1, len(notes)):
        gap = notes[i].start - notes[i - 1].end
        if gap >= pause_threshold:
            last_pause_idx = i

    if last_pause_idx is None:
        return notes

    gap = notes[last_pause_idx].start - notes[last_pause_idx - 1].end
    print(
        f"last long pause ({gap:.1f}s) before note {last_pause_idx} at "
        f"t={notes[last_pause_idx].start:.2f}s — trimming {last_pause_idx} leading note(s)"
    )
    return notes[last_pause_idx:]


def scale_to_gt_duration(gt: list[Note], trans: list[Note]) -> tuple[list[Note], float]:
    """Linearly stretch or compress trans so its total duration matches gt's.

    Duration is the end time of the last-ending note in each file.  Both
    lists must already be normalized to t=0.  Returns (scaled_trans, scale_factor).
    """
    if not gt or not trans:
        return trans, 1.0
    gt_end    = max(n.end for n in gt)
    trans_end = max(n.end for n in trans)
    if trans_end == 0:
        return trans, 1.0
    scale = gt_end / trans_end
    print(f"time scaling: trans {trans_end:.2f}s → {gt_end:.2f}s (×{scale:.4f})")
    return [Note(n.start * scale, n.end * scale, n.pitch) for n in trans], scale


def apply_octave_correction(gt: list[Note], trans: list[Note]) -> tuple[list[Note], int]:
    """If the first notes share a pitch class but differ in octave, shift all
    transcription pitches to close that gap. Returns (corrected_trans, semitone_shift)."""
    if not gt or not trans:
        return trans, 0
    gt_pitch = gt[0].pitch
    tr_pitch = trans[0].pitch
    if gt_pitch % 12 != tr_pitch % 12:
        return trans, 0   # different note type — no correction
    shift = gt_pitch - tr_pitch   # always a multiple of 12
    if shift == 0:
        return trans, 0
    print(f"octave correction: shifting transcription pitches by {shift:+d} semitones "
          f"({midi_note_name(tr_pitch)} → {midi_note_name(gt_pitch)})")
    corrected = [Note(n.start, n.end, n.pitch + shift) for n in trans]
    return corrected, shift


def match_notes(
    gt: list[Note],
    trans: list[Note],
    threshold: float,
) -> tuple[list[bool], list[bool]]:
    """One-to-one greedy matching by midpoint proximity.

    Returns:
        gt_matched   — bool list, True if gt[i] was hit by a trans note
        trans_matched — bool list, True if trans[j] matched a gt note
    """
    # Collect all candidate pairs (distance, gt_idx, trans_idx)
    candidates: list[tuple[float, int, int]] = []
    for gi, gn in enumerate(gt):
        for ti, tn in enumerate(trans):
            if tn.pitch != gn.pitch:
                continue
            dist = abs(gn.midpoint - tn.midpoint)
            if dist <= threshold:
                candidates.append((dist, gi, ti))

    candidates.sort()
    gt_matched = [False] * len(gt)
    trans_matched = [False] * len(trans)
    for dist, gi, ti in candidates:
        if gt_matched[gi] or trans_matched[ti]:
            continue
        gt_matched[gi] = True
        trans_matched[ti] = True

    return gt_matched, trans_matched


def score(gt_matched: list[bool]) -> float:
    if not gt_matched:
        return 0.0
    return sum(gt_matched) / len(gt_matched) * 100.0


def plot_comparison(
    gt: list[Note],
    trans: list[Note],
    gt_matched: list[bool],
    trans_matched: list[bool],
    threshold: float,
    gt_path: str,
    trans_path: str,
    accuracy: float,
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    all_pitches = [n.pitch for n in gt] + [n.pitch for n in trans]
    if not all_pitches:
        plt.close()
        return
    pitch_lo = min(all_pitches) - 1
    pitch_hi = max(all_pitches) + 1

    # ── Ground truth ────────────────────────────────────────────────────
    ax_gt = axes[0]
    ax_gt.set_title(f"Ground truth: {Path(gt_path).name}", fontsize=10)
    for note, matched in zip(gt, gt_matched):
        color = "#4a90d9" if matched else "#b0c8e8"
        rect = mpatches.FancyBboxPatch(
            (note.start, note.pitch - 0.4), note.duration, 0.8,
            boxstyle="round,pad=0.02",
            linewidth=0.8,
            edgecolor="#2060a0" if matched else "#8090b0",
            facecolor=color,
            alpha=0.85,
        )
        ax_gt.add_patch(rect)
        ax_gt.text(
            note.midpoint, note.pitch, midi_note_name(note.pitch),
            ha="center", va="center", fontsize=6, color="white",
        )

    # ── Transcription ───────────────────────────────────────────────────
    ax_tr = axes[1]
    ax_tr.set_title(
        f"Transcription: {Path(trans_path).name}  "
        f"(accuracy {accuracy:.1f}%)",
        fontsize=10,
    )
    for note, matched in zip(trans, trans_matched):
        color = "#2ecc71" if matched else "#e74c3c"
        edge  = "#1a8a4a" if matched else "#9b1e1e"
        rect = mpatches.FancyBboxPatch(
            (note.start, note.pitch - 0.4), note.duration, 0.8,
            boxstyle="round,pad=0.02",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=color,
            alpha=0.85,
        )
        ax_tr.add_patch(rect)
        ax_tr.text(
            note.midpoint, note.pitch, midi_note_name(note.pitch),
            ha="center", va="center", fontsize=6, color="white",
        )

    # ── Shared formatting ────────────────────────────────────────────────
    time_end = max(
        (max((n.end for n in gt), default=0)),
        (max((n.end for n in trans), default=0)),
    )
    for ax in axes:
        ax.set_xlim(-0.2, time_end + 0.5)
        ax.set_ylim(pitch_lo - 0.5, pitch_hi + 0.5)
        ax.set_ylabel("MIDI note")
        tick_pitches = sorted({n.pitch for n in gt} | {n.pitch for n in trans})
        ax.set_yticks(tick_pitches)
        ax.set_yticklabels([midi_note_name(p) for p in tick_pitches], fontsize=7)
        ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    axes[1].set_xlabel("Time (s, normalized)")

    # ── Legend ───────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor="#4a90d9", label="GT matched"),
        mpatches.Patch(facecolor="#b0c8e8", label="GT unmatched"),
        mpatches.Patch(facecolor="#2ecc71", label="Trans accurate"),
        mpatches.Patch(facecolor="#e74c3c", label="Trans inaccurate"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(
        f"MIDI comparison  |  threshold ±{threshold}  |  accuracy {accuracy:.1f}%  "
        f"({sum(gt_matched)}/{len(gt_matched)} GT notes matched)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Score a MIDI transcription against a ground truth MIDI."
    )
    ap.add_argument("ground_truth", help="ground truth .mid file")
    ap.add_argument("transcription", help="transcription .mid file to evaluate")
    ap.add_argument(
        "--threshold", type=float, default=0.3,
        help="midpoint match window in seconds (default 0.3)",
    )
    ap.add_argument(
        "--pause-threshold", type=float, default=2.0,
        help="gap (seconds) between consecutive transcription notes that "
             "triggers trimming of all preceding notes (default 5.0)",
    )
    ap.add_argument(
        "--no-plot", action="store_true",
        help="skip the matplotlib visualization",
    )
    args = ap.parse_args()

    gt_raw    = parse_midi(args.ground_truth)
    trans_raw = parse_midi(args.transcription)

    if not gt_raw:
        print("ground truth MIDI contains no notes", file=sys.stderr)
        sys.exit(1)
    if not trans_raw:
        print("transcription MIDI contains no notes", file=sys.stderr)
        sys.exit(1)

    trans_raw = trim_before_long_pause(trans_raw, args.pause_threshold)
    if not trans_raw:
        print("transcription is empty after pause-trimming", file=sys.stderr)
        sys.exit(1)

    gt    = normalize_start(gt_raw)
    trans = normalize_start(trans_raw)

    trans, time_scale = scale_to_gt_duration(gt, trans)
    trans, octave_shift = apply_octave_correction(gt, trans)

    gt_matched, trans_matched = match_notes(gt, trans, args.threshold)
    acc = score(gt_matched)

    n_gt    = len(gt)
    n_trans = len(trans)
    n_hit   = sum(gt_matched)
    n_acc   = sum(trans_matched)

    print()
    print(f"Ground truth notes : {n_gt}")
    print(f"Transcription notes: {n_trans}")
    print(f"Threshold          : ±{args.threshold}s")
    if abs(time_scale - 1.0) > 0.001:
        print(f"Time scale         : ×{time_scale:.4f}")
    if octave_shift:
        print(f"Octave shift       : {octave_shift:+d} semitones applied to transcription")
    print()
    print(f"GT notes matched   : {n_hit} / {n_gt}")
    print(f"Trans notes matched: {n_acc} / {n_trans}")
    print(f"Accuracy           : {acc:.1f}%")
    print()

    if not args.no_plot:
        plot_comparison(
            gt, trans, gt_matched, trans_matched,
            threshold=args.threshold,
            gt_path=args.ground_truth,
            trans_path=args.transcription,
            accuracy=acc,
        )


if __name__ == "__main__":
    main()
