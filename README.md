# visual-transcriber

A two-camera visual piano transcriber. The system watches a piano from
two cameras, detects key presses from per-frame back-edge motion, and
emits MIDI in real time. Same pipeline runs on a 25-key C4–C6 mini and
on a 61-key C2–C7 keyboard.

CSCI 1430 final project, Spring 2026.

---

## Quick start

**First-time setup is required before this works** — see [Setup](#setup)
below. You need: `uv sync`, one or two cameras connected, fluidsynth
installed via brew, and a soundfont in `sound_fonts/`. The
`Transcriber` is constructed unconditionally at startup and will fail
fast if fluidsynth or the soundfont is missing. The MediaPipe Hand
Landmarker model is optional but recommended (without it the
diagnostic hand-mask panel renders blank; the active press detector
runs either way).

```bash
uv sync
uv run python main.py
```

`main.py` opens both Canon streams, previews them side-by-side, and
waits for hotkeys:

| Key   | Action                                                |
| ----- | ----------------------------------------------------- |
| SPACE | Calibrate both cams from the current frame, begin detection |
| `c`   | Recalibrate (e.g. after the keyboard or camera moves) |
| `b`   | Recapture the rest baseline (~2 s, hands off keys)    |
| ESC   | Save MIDI to `midi_outs/` and exit                    |

All knobs (camera indices, `KEYBOARD_LAYOUT`, press threshold, fusion
window) are constants at the top of `main.py`. To run on a 25-key
mini, set `KEYBOARD_LAYOUT = LAYOUT_25KEY`.

---

## Setup

### Cameras (required)

We use two Canon R50s in webcam-streaming mode connected over USB-C.
On each camera: Menu → wrench tab → 4 → "Video Streaming". Plug each
camera into a separate USB-C port (avoid hubs — they introduce
desync). Run `uv run python stream_webcams.py` to confirm both feeds
are live.

Resolution and FPS are read from `config.yaml`. Single-camera mode also
works: set `DUAL_CAM = False` in `main.py` and adjust `CAM_INDEX`.

### Audio (required — Transcriber is constructed at startup)

The `Transcriber` routes detected notes through fluidsynth to a
soundfont so you hear what was detected as you play, and also writes
the MIDI file on exit. It is constructed unconditionally when
`main.py` starts, so the system fluidsynth binary and a soundfont
file must both be present or the process will fail at launch.

```bash
brew install fluidsynth
```

Drop a `.sf2` file into `sound_fonts/`. We use
[FluidR3_GM_GS.sf2](https://musical-artifacts.com/artifacts/1229/FluidR3_GM_GS.sf2).
Sound-font filename can be set explicitly in `config.yaml` (`soundfont:`);
if left blank, the first `.sf2` in `sound_fonts/` is picked up.

Set the macOS audio output device **before** launching — fluidsynth
captures the default at startup and won't reroute mid-session.

### MediaPipe model (optional, for the diagnostic hand-mask viz)

The `mediapipe` Python package is installed by `uv sync` and is needed
for the project to import. The Hand Landmarker *model file* is a
separate ~7.5 MB download placed at the repo root (gitignored):

```bash
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

The active press detector does not consume the hand mask (see "What's
*not* on the active path" below), so this download is genuinely
optional. If you skip it you'll see one warning line at startup —
`[Detector] mediapipe requested but unavailable: installed=True
model_exists=False` — and the diagnostic hand-mask panel will be
blank. Everything else, including detection and MIDI output, runs
identically.

---

## Active pipeline

```
main.py
└── pipeline.py                    # orchestration: preview, calibrate, fuse
    ├── stream_webcams.py          # CanonStream / DualCanonStream
    ├── auto_calibrate.py          # calibrate_frame: warp + segment + label
    │   ├── seg_to_keys.py         # warp_to_piano: corners via LSD/RANSAC
    │   └── calibration.py         # build_calibration_data: persists keys.json
    │       └── key_labeler.py     # black-key detection, canonical alignment
    ├── detection.py               # Detector: per-frame diff press detector
    │   ├── analyze.py             # build_overlays: polygons → key_id_map
    │   └── press_diff.py          # detect_press_regions: top-band diff
    ├── record.py                  # overlay drawing, transcribe LUT
    ├── playback.py                # _label_panel: viz cells
    └── transcribe.py              # press set → MIDI + fluidsynth
```

### Pipeline overview

1. **Calibration (one-time per session, on SPACE).** Each camera
   warps its frame to a flat top-down strip via 4 detected corners
   (`seg_to_keys.warp_to_piano`), then segments black-key polygons by
   2D Otsu on the upper band, splits merged blobs at U-valleys with
   per-blob far-side template projection, aligns the detected pattern
   to the canonical narrow/wide gap pattern derived from the layout
   (`KeyboardLayout(n_octaves, start_octave)`), assigns note labels,
   and computes white-key seams analytically. Result is saved to
   `<photo>_keys.json` (polygon, type, note per key, plus warp corners
   and `far_side`).

2. **Press detection (per frame).** Each frame is differenced against
   a rest baseline, restricted to the top 15% of the warp (where the
   key tilts and the back edge shifts), and connected components are
   computed. Each blob is attributed to a single key by a **camera-side
   geometric rule**: a left-mounted cam credits the leftmost-touched
   key, a right-mounted cam the rightmost. This corrects for the
   perspective bias that makes a press cue land in the seam between
   the pressed key and its neighbour on the camera-far side.

3. **Dual-camera fusion.** Per-frame press sets from the two cameras
   are merged by **union with a short per-key temporal hold** — if
   either cam fires key K its TTL is refreshed, and K stays "pressed"
   in the fused set until the TTL decays. No per-key voting or
   visibility weighting in the active path.

4. **MIDI emission.** The fused press set drives `transcribe.update`,
   which converts to MIDI note-on / note-off events and (optionally)
   plays them through fluidsynth in real time. ESC saves to
   `midi_outs/YYYYMMDD_HHMMSS.mid`.

### What's *not* on the active path

The Detector class still contains code for MOG2 background
subtraction, MediaPipe-based hand-mask AND-NOT gating, and per-key
brightness / slope / tempdiff scoring channels. These were earlier
exploratory approaches; we found that restricting the rest-baseline
diff to the top 15% of the warp captured the press cue cleanly enough
that the hand mask was eating real signal and the multi-channel
scoring was redundant. The legacy code is retained for diagnostic
visualisation and documented in `detection.py`'s module docstring.

`hand_gate.py` (MediaPipe fingertip-to-key candidate filter) is built
only on the single-camera path. Dual-cam mode forces it off because
running two MediaPipe instances per frame at 30 fps blows the
per-frame budget.

---

## Standalone scripts

These are not part of the live pipeline — run them on their own.

| Script                      | Purpose                                                                     |
| --------------------------- | --------------------------------------------------------------------------- |
| `auto_calibrate.py`         | One-shot calibration on a saved photo. Pops a popup; writes `_keys.json`.   |
| `manual_calibrate.py`       | 4-click manual calibration fallback when auto-detection fails.              |
| `validate_calibration.py`   | Spot-check a saved `_keys.json` (sanity-check note labels, baselines).      |
| `archive_recording.py`      | Bundle a `recordings/<folder>` with its calibration into a self-describing archive. |
| `seg_to_keys.py`            | Run as `__main__` to view the warp pipeline live on a Canon stream.         |
| `key_labeler.py`            | Run as `__main__` on a single image to see the segmentation overlay.        |
| `stream_webcams.py`         | Run as `__main__` to confirm both Canon feeds are live.                     |
| `utility_scripts/cam_probe.py`     | Open camera indices 0..4 and print which ones connect — sanity-check before launching `main.py`. |
| `utility_scripts/cam_identify.py`  | Open a single camera index and display its feed — confirm which physical camera is which index. |
| `utility_scripts/midi_compare.py`  | Score a transcribed MIDI against a ground-truth MIDI (note-level F1).        |

Example:

```bash
uv run python auto_calibrate.py piano_photos/MINI_left_cam.jpeg \
  --layout 25 --far-side right --top-crop 10
```

`--layout {25,61}`, `--far-side {left,right}`, `--top-crop N`.

---

## Outdated / legacy

`outdated/` contains earlier iterations kept for historical context.
Nothing in `outdated/` is imported by the active pipeline. Folder
contents:

| File                       | Status / why retired                                                  |
| -------------------------- | --------------------------------------------------------------------- |
| `archive.py`               | Earlier archive helper — superseded by `archive_recording.py`.        |
| `auto_calibrate.py`        | Earlier auto-calibration draft — superseded by the top-level version. |
| `key_extractor2.py`        | Earlier key-extraction module.                                        |
| `old_key_labeler.py`       | Pre-canonical-alignment labeler.                                      |
| `test_threshold.py`        | One-off threshold experiment.                                         |
| `batch_test.py`            | Photo-grid batch tester (depended on `key_extractor2`, broken).       |
| `key_detection.py`         | Older `KeyDetector` style — read dead JSON fields.                    |
| `live_labeler.py`          | Predecessor to `key_labeler.draw_labels_tight_crop`.                  |
| `keyboard_play.py`         | Laptop-keyboard-as-piano test, unrelated to transcription.            |
| `live_test.py`             | Early "grab a frame and display" smoke test. Its single useful helper (`find_specific_camera_index`) lives in `stream_webcams.py` now. |

---

## Layouts supported

| Layout         | Span     | Black keys | White keys |
| -------------- | -------- | ---------- | ---------- |
| `LAYOUT_61KEY` | C2–C7    | 25         | 36         |
| `LAYOUT_25KEY` | C4–C6    | 10         | 15         |

Other C-to-C spans work without code changes — pass a
`KeyboardLayout(n_octaves, start_octave)` to the calibration entry
points. The layout drives expected key counts, the canonical
narrow/wide black-key gap pattern, and note-label assignment.

---

## Output

- `midi_outs/YYYYMMDD_HHMMSS.mid` — MIDI saved on ESC.
- `piano_photos/<stem>_keys.json` — calibration record (polygons,
  notes, warp corners, `far_side`, output size).
- `piano_photos/<stem>_calib.json` — corner-only calibration.
- `piano_photos/<stem>_warped.png` / `_labeled.png` — calibration
  visualisations.
