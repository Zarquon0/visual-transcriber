## Set Up
Clone, then:
```bash
uv sync
```
To run scripts:
```
uv run python <script>
```
### Additional Setup for Transcriber Usage
The `Transcriber` object in `transcribe.py` enables live playing through an external synthesizer. To get live playing to work:
```zsh
# On Mac
brew install fluidsynth
```
Additionally, you'll need at least one sound font file in `sound_fonts/`. One good option is [here](https://musical-artifacts.com/artifacts/1229/FluidR3_GM_GS.sf2). Feel free to add multiple sound fonts to this folder, you may optionally specify which one to use in `config.yaml`.

### Additional Setup for MediaPipe Hand Mask
`detection.py`'s `--mediapipe` path needs the MediaPipe Hand Landmarker model dropped at the repo root. It's gitignored (7.5 MB binary). Download once:
```bash
curl -L -o hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Webcam Streaming
`stream_webcams.py` creates cv2 connections to attached Canon R50 cameras. To properly connect cameras:

1. Set all Canon R50 cameras into webcam streaming mode. To do so, power on each camera and hit Menu (button) > yellow wrench (tab) > 4 (tab) > select "Video Streaming". The feed on the camera's display will change slightly in indication of this.
2. Connect cameras to your Mac (only Macs are supported currently) via USB-c cord. NOTE: it is recommended that you connect each camera to a separate USB-c port on your computer (as opposed to connecting multiple to an adapter) to limit delays or freezes.
3. Run `uv run python stream_webcams.py`. You should see all connected cameras' streams (you may need to move a video window to see occluded ones).

If one or more video streams are missing, you can try using `cam_probe.py` and `cam_identify.py` to debug. `stream_webcams.py` should automatically find and detect connected Canon cameras, but this functionality has not been thoroughly tested.
 
To change the image resolution or FPS (maybe - not sure FPS control works), alter the values in the `config.yaml` file.

### Synced Dual Canon Camera Streaming
For synced dual camera streaming, use the DualCanonStream object in `stream_webcams.py`. The `read()` method in the object allows you to read the most recent synced frames from the two cameras. Look to the `if __name__=="__main__":` in `stream_webcams.py` for a usage example.

### iPhone Continuity Camera

`open_canon_streams()` filter rejects all non Canon camera streams by default, but iPhones can be accepted if the `allow_iphone` flag is passed as `True` to the function.

## Crop/Warp to Piano
`seg_to_keys.py` contains functionality for cropping/warping the view to fit the piano keys. To give it a test, simply run connect a Canon camera and run the script. You should see a live stream that looks something like the following:
<img width="1367" height="551" alt="Screenshot 2026-04-25 at 1 40 43 PM" src="https://github.com/user-attachments/assets/ed681501-d025-4cbc-ba7e-69251cf2a138" />
<img width="1374" height="510" alt="Screenshot 2026-04-25 at 1 39 42 PM" src="https://github.com/user-attachments/assets/682c2714-a4d9-457d-9461-2219474c3e88" />
<img width="957" height="535" alt="Screenshot 2026-04-28 at 4 20 09 PM" src="https://github.com/user-attachments/assets/5f64c6ac-2ab4-4697-94d5-ec7b1f7d9f19" />

The warping algorithm is now rotationally invariant, so in normal lighting and without an excess of bright objects in frame, it should work well!

## Note Labeling

### Pipeline overview
1. Use `warp_to_piano` from `seg_to_keys.py` to obtain clean image of keys.
2. Find `y_black_bottom` (the red line) via Sobel-y horizontal edge.
3. **Black-key detection** (`_detect_blacks_2d`): 2D Otsu connected-component on the upper band. Each merged blob (multi-key region) is split via **U-valley analysis** on the bottom-y profile of the blob mask, then each inner piece gets the **camera-far outer piece's actual contour** projected onto it as a local template. Z-order clipping resolves overlap between adjacent projected pieces. Falls back to 1D column-projection when 2D fails on top-down shots.
4. **SWSSW projection** (`_project_to_25`) aligns detected blacks to the canonical 25-key pattern and fills any still-missing positions with translated template polygons.
5. **Geometric edge guard** trims any polygon that over-extends past the keyboard's playable area (`0.5 * white_key_w` from the left edge, `1.5 * white_key_w` from the right edge — the 61-key C-to-C layout's natural buffer for C2 and B6+C7).
6. **White-key seams**: Sobel-x peak detection on the white band, with local-median gap-fill (between detected peaks) and edge-extrapolation (past the first/last detected peak). Each seam draws through every row where no black-key polygon covers its column — one unified rule for partial vs full-height.
7. **Note labels**: hardcoded canonical SWSSW pattern assigns letters; final labels are C#2..A#6 for blacks, C2..C7 for whites. Auto-scaled font size + 2-row stagger keeps labels readable on narrow side-view warps.

The `far_side` parameter on `draw_labels_tight_crop` / `_detect_blacks_2d` selects the camera-far direction (``"right"`` or ``"left"``). The 4-corner detection, geometric edge guard, and seam pipeline are all **camera-agnostic**; only the per-blob template-projection step is camera-side dependent. In a dual-cam rig each camera sets its own `far_side`.

### Scripts
- **`key_labeler.py`** — Image labeling. Can be run standalone on a single photo: `uv run python key_labeler.py path/to/photo.jpg` → writes `<photo>_labeled.png`.
- **`live_labeler.py`** - Stream labeling. Labels a canon stream live: `uv run python3 live_labeler.py`. NOTE: To enable iPhone streaming, make sure to pass in the `allow_iphone` flag as `True` to `open_canon_streams`.
- **`manual_calibrate.py`** — fallback 4-click calibration for shots auto fails on. Click TL, TR, BR, BL. Saves `<photo>_warped.png`, `<photo>_labeled.png`, `<photo>_calib.json` next to the input.

### Labeler output
On the tight warped image:
- **Red horizontal line**: detected black/white boundary (`y_black_bottom`).
- **Blue polygons**: black-key outlines. Either the actual `cv2.findContours` contour for unmerged blobs, or the camera-far outer piece's contour translated to inner pieces of merged blobs (with overlap resolved by Z-order: closer key wins).
- **Yellow vertical lines**: white-key seams. Drawn through every row of each seam's column where no black-key polygon covers that row — single unified rule, so seams go full-height in E|F / B|C gaps and clip above any black-key body otherwise.
- **Note labels**: C#2–A#6 on blacks, C2–C7 on whites (assumes 61-key board with leftmost black = C#2; override `start_octave` in `_label_notes_61key` if different). Labels stagger across two y-rows so adjacent labels don't overlap on narrow warps.

### Per-key region storage (handoff for next dev)

Each calibration run writes a `<photo>_keys.json` next to the input image, containing per-key polygon, label, source/confidence tag, baseline intensity, and a "safe" subregion bbox (pre-shrunk to drop edge pixels and leave a fingertip-occlusion buffer). Schema is documented in `calibration.py`'s module docstring.

**Generate:**
```
uv run python auto_calibrate.py path/to/calibration_frame.jpg
# → writes path/to/calibration_frame_keys.json
```

**Load at runtime** (the next dev's starting point — see `calibration.py`):
```python
from calibration import Calibration
rt = Calibration.load("path/to/calibration_frame_keys.json")
warped = rt.warp(frame_bgr)              # one cv2.warpPerspective
gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
for key in rt.keys:
    pixels = gray[key.safe_mask]         # pre-rasterized bool mask
    # … press-detection logic goes here, using:
    #     key.note                 ("C#3", "F4", …)
    #     key.type                 ("black" / "white")
    #     key.source               ("detected" / "template_projected" /
    #                               "inferred" / "geometric")
    #     key.confidence           (0.55..0.95)
    #     key.baseline_intensity   (mean gray inside safe_bbox at calib)
```

Each `Calibration` pre-rasterizes every key's safe-region polygon to a `np.bool_` mask **once at load time**, so per-frame work for the next dev is just a warp + mean-by-mask per key (~61 keys, a few ms total).

`calibration.py` deliberately stops at storage / segmentation. **Press detection itself isn't implemented here** — that's the next dev's task.

**Validator:** `validate_calibration.py` produces a side-by-side image showing the raw warp vs. each stored key's polygon outline (color-coded by source/confidence) with note labels overlaid, plus a summary footer flagging duplicate / missing labels and baseline-intensity sanity violations. Run it after `auto_calibrate.py` on any `_keys.json` to confirm the storage matches the visible keys:

```
uv run python validate_calibration.py piano_photos/<photo>_keys.json
# → writes <photo>_validate.png alongside the input
```

### Jumping-off point for the next dev (press detection)

**Files to read first:**
- `calibration.py` — JSON schema in docstring; `Calibration.load` + `RuntimeKey` (note, type, source, confidence, baseline_intensity, safe_mask, polygon).
- `key_labeler.py` — `draw_labels_tight_crop` (whole detection pipeline), `_detect_blacks_2d` (per-blob template projection that gives each key its polygon).
- `_calib.json` files in `piano_photos/` (warp corners) and `_keys.json` files (per-key regions + baselines) — sample data to develop against.

**Possible directions for change/press detection:**

- **Threshold on intensity delta** in each `safe_mask` vs. `baseline_intensity`. Weight the threshold by `1 / key.confidence` — high-confidence keys can flag presses on smaller deltas. Auto-tune the threshold per camera / lighting from a short calibration video (no-press baseline + every-key-pressed sequence).
- **Hand / skin masking** before the mean: HSV skin filter (broad first pass: `H ∈ [0, 20], S ∈ [20, 150], V ∈ [70, 255]`) excludes hand pixels per frame. If <50% of a key's safe-mask survives the filter, treat as occluded (reuse last frame's value).
- **Shadow rejection**: a hand hovering over the keyboard casts a shadow that drops gray-mean across multiple adjacent keys uniformly. Detect by comparing per-key delta to the *local median* of nearby keys' deltas — only flag a press where one key's delta diverges from its neighbours'.
- **Color over intensity**: black-key press-down doesn't change brightness much; saturation/hue around the front of the key shifts more. Consider per-key color histograms instead of single-channel mean.
- **Temporal filter**: require press to persist for ≥ N consecutive frames before committing a note-on event. Filters shadow flicker and skin-mask noise.
- **Per-cam fusion** (dual-cam setup): each camera produces its own `Calibration`. Run press detection on each, then fuse — a note is "pressed" if either cam reports it, weighted by which cam covers it on its near half (where polygons are most accurate).

### Remaining bugs / improvements in the labelling pipeline itself

To clean up before the press-detection work above is reliable:

- **Far-most blob still occasionally has a too-wide outline** when the splitter's per-blob template is itself rectangular (the foreshortened far-most piece has no curve to extract). Fix idea: when the rightmost piece of a merged blob is overly rectangular (rect-fill > some threshold), use the previous blob's far-template as a fallback. Earlier attempts at this misclassified valid keys; needs a *provenance flag* (template-projected vs. own-contour) rather than a rect-fill heuristic.
- **U-valley detection on flat-bottomed blobs** can miss real key boundaries when adjacent columns share the same `bot_sm` value. Currently relaxed to `<=` on one side of the local-min check (catches plateau left-edges). Could go further — explicit plateau detection — if cases persist.
- **Edge polygons at the keyboard's left/right** sometimes drift past the playable area. Mitigated by the geometric edge guard (`0.5 / 1.5 × white_key_w`), but very tight or very loose warp calibrations can still slip past it.
- **`_label_notes_61key` assumes exactly 25 black keys** detected; if `_project_to_25` returns fewer, no labels appear. Should label whatever subset of canonical positions is present.
- **Polygons over-extending into white-key area** at the bottom of some keys — the blob mask sometimes includes shadow under a key. Per-column dark→light gradient detection (Sobel-y, per column) could clip each polygon's bottom precisely; tried and reverted because it interacted badly with the union-with-blob-mask step.
- **Note-label cluster on extreme side-views** is mostly fixed by 2-row staggering + auto-scaling, but very narrow warps (< 600px) still cram. Vertical (rotated 90°) labels would fully solve this.

### Wider directions (not committed plans)

- **Dual-camera capture**: two upper-side-angle cams, one each with `far_side="right"` and `far_side="left"`. Each sees the full keyboard; fusion at the press-event level (above) gives near-side-of-each-cam priority.
- **Hand occlusion**: the current detection tries to union keyboard-shaped pieces, which helps with small splits. Per-cam redundancy in the fusion step compensates further.
- **Auto-calibration recovery**: today `_calib.json` files are committed per camera mount. A periodic recalibration step (re-detect corners every few minutes if the camera moves) could remove the manual-clicking step long-term.


## Live PoC Pipeline (May 2026)

A live recording + offline playback workflow on top of the existing calibration pipeline. Lets you capture clips, iterate calibration without leaving the live view, and replay-with-detection offline.

> ### ⚠️ Before tuning thresholds, know what's actually wrong
>
> The detection pipeline has multiple noise sources that **dominate over** any threshold tuning. Tuning per-key margins helps at the margin (no pun) but doesn't fix the underlying issues:
>
> 1. **Hand mask is unreliable.** Current motion+HSV+persistence mask still bleeds onto pressed-key edges and sometimes misses still hands. **MediaPipe-based `core/hand_gate.py` on the `visual_detection` branch** is the right fix.
> 2. **Shadows can fire detection.** A hand-shadow over a key looks like a brightness drop. Mitigated partially by the LSD gradient floor (drops weak-gradient segments), but soft shadows still cause flicker.
> 3. **White-key seams have low contrast.** LSD doesn't reliably fire on white-on-white edges. White-key press signal lives mostly in brightness + tempdiff channels, not lines.
> 4. **General LSD frame-to-frame jitter.** LSD is non-deterministic across small pixel changes. Smoothing window helps but doesn't eliminate flicker.
>
> Threshold tuning (`+/-`, `1/2/3/4` hotkeys) is a fine-grained knob, useful AFTER the structural issues are addressed. Don't expect it to compensate for a broken hand mask.

### Quick start — copy-paste commands

The current primary detection path is **`--diff`** (William's `press_diff.detect_press_regions` for the activation mask, MediaPipe for the hand mask, per-key absolute pixel-count scoring with boundary erosion). Use `--mediapipe` alongside `--diff` so hands are excluded properly. `--transcribe` plugs the press events into a soundfont synth + MIDI recorder.

**Two equivalent live entry points:**

- **`main.py`** — simplified single-camera live pipeline. SPACE to calibrate, `c` to recalibrate live, `b` to recapture baseline, ESC to save MIDI. All hyperparameters as constants at the top of the file. **Recommended for normal play.**
  ```bash
  uv run python main.py
  ```

- **`record.py`** — dev-mode equivalent with full CLI surface, recording capability, and live tuning hotkeys (`1`/`2` press_pix, `3`/`4` min_blob, `5`/`6` boundary, `m` motion supp). Used during pipeline tuning.
  ```bash
  uv run python record.py --no-iphone --cam-index 0 \
      --keys recordings/_snapshots/calib_<timestamp>_cam0_keys.json \
      --mediapipe --diff \
      --diff-press-pixels 5 --diff-min-blob-area 1 --diff-boundary-margin 1 \
      --smooth-window 1 --transcribe
  ```

> **Pre-flight:** `brew install fluidsynth` (one-time), `uv sync`, drop a `.sf2` in `sound_fonts/` (we use `FluidR3_GM_GS.sf2`), set the macOS audio output device (System Settings → Sound or menu-bar volume icon) **before** launching — fluidsynth captures whichever device is default at startup and won't reroute mid-session.

> **macOS Reactions warning:** macOS Sonoma+ has system-level **Camera Reactions** that overlay balloons / hearts / fireworks etc. on the camera feed when it sees gestures like thumbs-up or peace-signs. These are applied upstream of OpenCV — they corrupt the frames our pipeline receives, breaking warp / segmentation / hand mask / press detection. **Disable via Control Center → Video Effects → Reactions OFF while the camera is in use.**

Then in the recorder window:
1. Press **`c`** — auto-calibrate to current camera position. HUD shows `61 keys (b:25 w:36, labeled:61)` on success. The transcribe note-LUT is rebuilt from the calibration's note labels.
2. Press **`d`** — enable detection.
3. **Hands away from keyboard**, press **`b`** — captures REST baseline (60 frames ≈ 2 s in `--diff` mode; CHAOS phase is auto-skipped). HUD shows `REST CAPTURE — keep hands away (60 left)`. After it completes you'll see `DETECTING — pressed: …` on panel 1.
4. Play. Pressed keys flash red on the source view + per-polygon overlay; fluidsynth plays each note via the loaded soundfont.
5. **`ESC`** / **`q`** to quit. MIDI dumped to `midi_outs/YYYYMMDD_HHMMSS.mid`.

**Offline replay of a recorded clip — same pipeline against bundled frames:**

```bash
uv run python playback.py recordings/1777663914_press \
    --mediapipe --diff \
    --diff-press-pixels 5 --diff-min-blob-area 1 --diff-boundary-margin 1 \
    --smooth-window 2 --debounce 1 \
    --transcribe
```

**main.py hotkeys** (focus the `piano` window):
- `SPACE` calibrate (preview phase) → starts detection
- `c` recalibrate from current frame (during detection)
- `b` recapture REST baseline over `BASELINE_FRAMES` (60) quiet frames — hands AWAY
- `ESC` save MIDI to `midi_outs/<timestamp>.mid` and quit

**The seven `warp_lines` panels** (top → bottom — same in main.py, record.py's `d` mode, and playback.py):

1. **RAW WARP + PRESSES** — pressed keys outlined red on the warped strip.
2. **SEGMENTATION** — colored key polygons over the warp (calibration sanity check).
3. **HAND MASK (MP)** — orange skin fill + magenta contour over warp (what MP + persistence excluded).
4. **MP EXTENDED WARP** — broader rectified view MP runs on, with the keyboard region in red and the 21-landmark hand skeleton overlaid. Confirms that MediaPipe is actually detecting hands.
5. **PRESS DIFF (raw mask)** — `detect_press_regions` output **before** hand exclusion. Should light up white wherever the live frame differs by ≥75 from the REST baseline.
6. **COUNTED (-hand -boundary)** — panel 5 minus the hand mask AND minus the polygon-boundary ribbon. **This is the exact pixel signal that gets `bincount`-summed into per-key totals.** If a press shows up here, it counts.
7. **PER-KEY ACTIVATION + SCORES** — per-polygon overlay with red activation fill + top-3 `kN: count/threshold` annotations on the most-active keys.

**Live hotkeys** (focus warp_lines or playback window):
- `SPACE` pause/resume (playback only) · `[`/`]` ±1 frame · `,`/`.` ±30 frames · `<`/`>` ±150 frames · `0`/`9` jump start/end (playback only)
- `c` recalibrate / `d` toggle detect / `b` capture baseline (record.py only)
- `1` / `2` press-pixel threshold − / + 5 (lower = more sensitive). Default 20; we run 5.
- `3` / `4` min-blob area − / + 5 (CC area floor on post-hand mask). Default 15; we run 1 (= disabled).
- `5` / `6` boundary margin − / + 1 px (ribbon ignored at every seam). Default 1; we run 1.
- `m` toggle motion-supplement hand mask. Default OFF — turning it on adds frame-to-frame Δgray>80 pixels to the persistence map; useful against fast sweeps but can mask press signal.
- `s` save current frame · `q` / `ESC` quit (saves MIDI if `--transcribe`).

**What each diff knob does:**
| Flag | Default in code | What we run | Effect |
|---|---|---|---|
| `--diff-press-pixels` | 20 | **5** | Per-key activated-pixel count needed to fire. Lower = more sensitive. |
| `--diff-min-blob-area` | 15 (5 in our run) | **1** | CC area floor on post-hand activation mask. 1 = no filter. Higher rejects small specks. |
| `--diff-boundary-margin` | 1 | **1** | Pixels of erosion per polygon → 2× px gap between adjacent keys. Stops boundary blobs cross-firing. 0 = no gap. |
| `--smooth-window` | 5 (rec) / 5 (pb) | **1** (rec) / **3** (pb) | Rolling-mean window over per-key counts. Lower = snappier onset. |
| `--debounce` | 1 (rec) / 3 (pb) | **1** (rec) / **1** (pb) | Frames a key must stay above threshold. 1 = instant. |

These are all intentionally aggressive; press_diff's strict threshold-75 + top-frame blob filter means surviving activation is rarely noise.

**Why ``--mediapipe``:** the hand mask. With it on, MP runs on every frame's extended warp, finds 21-landmark hand skeletons + convex hull, projects them into warped coords, and AND-NOT's them out of the activation mask before per-key counting.

**Why ``--transcribe``:** wires `press_set` → `Key(note, octave)` → `Transcriber.update()` → fluidsynth note-on/off + MIDI accumulation. Soundfont is loaded from `sound_fonts/<config.yaml soundfont>`; if that field is empty the first `.sf2` in the folder is used.

**Legacy 4-channel mode** (line / brightness / slope / tempdiff fusion, kept for comparison) — drop `--diff` and use the older flags:
```bash
uv run python playback.py recordings/1777663914_press \
    --thresholds recordings/_analysis/1777665677/summary.csv \
    --margin 1.0 --margin-black 1.5 --margin-white 0.6 \
    --bright-margin 1.5 --tempdiff-sigma 0.5 \
    --smooth-window 3 --debounce 1 --mediapipe
```

### `detection.py` — unified per-frame `Detector` class

The single source of truth for press detection. Used identically by both `record.py` (live `d` mode) and `playback.py` (offline replay). Encapsulates:

- **Hand mask** — motion + HSV color (adaptive within static-skin envelope) + persistence + connected-component blob filter on tight-mask shape + hole fill.
- **4 channels** — anomalous-LSD-line length, brightness delta vs rest baseline (with global illumination shift correction), weighted-mean line angle z-score, temporal-difference σ vs chaos floor.
- **Shadow rejection** — LSD gradient-magnitude floor (`GRAD_MIN=30`) drops weak-gradient segments that characterize shadow boundaries.
- **Per-type fusion** — blacks: `lines OR slope OR tempdiff`; whites: `brightness OR slope OR tempdiff`.
- **Rolling-mean smoothing** + **temporal debounce** before press fires.

Public API:
```python
det = Detector(keys_dict, smooth_window=5, debounce=1, ...)
det.set_rest_mean_frame(mean_gray)            # for brightness, tempdiff, hand-mask motion side
det.set_brightness_baseline(per_key_array)
det.set_brightness_thresholds(per_key_array)
det.set_slope_baseline(mean_array, std_array)
det.set_tempdiff_chaos_stats(mean_array, std_array)
det.set_line_thresholds(per_key_array)
# per frame:
pressed_set, line_viz_image = det.process(warped_bgr)
```

If a baseline isn't set, that channel is silently disabled (`process()` still works on the channels it does have).

### `record.py` — live recorder + iterative calibration

```
uv run python record.py --cam-index 0 --keys recordings/_snapshots/<latest>_keys.json
```

Pass `--no-iphone` to exclude iPhone Continuity Camera, or `--cam-index N` to override auto-selection. `--top-crop N` sets the initial top-crop for the warp (default 10 — trims the case-top band that `warp_to_piano` includes by design).

**Hotkeys** (recorder window must be focused):

| Key | Action |
|---|---|
| `c` | Calibrate from the current frame in-place. Pops the `warp_cam0` inspector window with the colored polygon overlay and key counts. |
| `/` `,` `0` | Top-crop +5 / -5 / reset. Each press auto-recalibrates and refreshes the inspector. |
| `r` / SPACE | Start/stop recording. Frames dump to `recordings/<ts>/cam0/000xxx.png`. On stop, auto-archives (warp + segmented + overlay snapshots + manifest). |
| `s` | Snapshot a single frame to `recordings/_snapshots/snap_<ts>_cam0.png`. |
| `o` | Toggle the live-overlay rendering. |
| `k` | Save the current in-memory calibration to `recordings/_snapshots/calib_<ts>_cam0_keys.json`. |
| `d` | Toggle live press detection. Pressed keys flash red on the source view, plus a `warp_lines_cam0` window opens showing color-coded LSD segments. |
| `b` | **Two-phase baseline capture for live detection.** First 30 frames REST (hands AWAY) → builds rest mean / brightness / slope baselines. Next 60 frames CHAOS (hover, NO presses) → builds line thresholds + tempdiff noise floor. HUD overlay tells you which phase + how many frames left. After both phases the live detector is fully populated. |
| `+` / `-` | Raise/lower the per-type margins (both blacks and whites). |
| ESC / q | Quit. |

### `auto_calibrate.py` — corner detection + keys.json

In-process and CLI; called from `record.py` on `c` press, also runnable standalone:

```
uv run python auto_calibrate.py path/to/snap.png [--top-crop N]
# → writes <stem>_calib.json, <stem>_keys.json, <stem>_warped.png, <stem>_labeled.png
```

### `playback.py` — replay a recorded clip with detection overlay

```
uv run python playback.py recordings/<press_folder> \
    --thresholds recordings/_analysis/<ts>/summary.csv \
    --margin 1.0 --margin-black 1.5 --margin-white 0.6 \
    --bright-margin 1.5 --tempdiff-sigma 0.5 \
    --smooth-window 3 --debounce 1
```

Reads the recording's bundled `cam0_keys.json` for segmentation and runs the same 4-channel detection from `analyze.py` on each frame, with rolling-mean temporal smoothing. Press-detected keys flash red on the source view; the second window (`warp_lines`) shows every LSD segment color-coded by classification (green = anomalous, yellow = polygon-edge, red = on-skin, blue = outside-polygon) plus a translucent **orange skin-mask fill + magenta contour outline** so you can see exactly which pixels the hand mask covers each frame.

**Hotkeys:**

| Key | Action |
|---|---|
| SPACE | Pause/resume |
| `[` `]` | ±1 frame (auto-pauses) |
| `,` `.` | ±30 frames (~1 sec) |
| `<` `>` | ±150 frames (~5 sec) |
| `0` `9` | Jump to start / end |
| `+` / `-` | Global threshold margin (multiplies all per-key thresholds) |
| `1` `2` | Black-key margin -/+ (lower = more sensitive) |
| `3` `4` | White-key margin -/+ |
| `s` | Save current frame with overlay |
| ESC / q | Quit |

### `analyze.py` — offline characterization (optional)

`analyze.py` is **not required for live detection** — `record.py` captures its own baselines in-session via the `b` hotkey. It's a development / validation tool:

```
uv run python analyze.py recordings/<rest_folder> recordings/<chaos_folder> recordings/<press_folder>
# → writes recordings/_analysis/<ts>/summary.csv + summary.png + timeseries.png
```

**What it does:** runs the 4 detection channels on each frame of all three clips and outputs per-key SNR (`(press_max − chaos_mean) / chaos_std`) and a CSV of all channels' max values per phase. Useful for:

- **Validating signal exists**: confirm a press actually produces a measurable channel response (e.g., key 26 white hit 17σ on lines in our reference data).
- **Generating thresholds for offline replay**: `playback.py --thresholds <CSV>` consumes the line-channel chaos values from this CSV.
- **Comparing channels**: see which channel works for which key type (drove the per-type fusion design).

**You don't need to run analyze.py:**
- For LIVE — `record.py` captures everything in-session.
- For replaying the committed reference clips — the committed `recordings/_analysis/1777665677/summary.csv` is already there.
- Only needed for fresh recordings where you want offline-replay thresholds.

The 4 channels analyze.py computes (also used live by `Detector`):
- **Brightness delta** — mean intensity inside polygon (skin-masked), global-illumination corrected. Best for whites where lines fail.
- **Anomalous-LSD-line length** — segments inside polygon AND outside its boundary band. Best for blacks.
- **Slope (weighted-mean angle)** — average angle of LSD segments per polygon. Tracks tilt of pre-existing lines.
- **Temporal difference** — `|current − rest_mean|` mean per polygon, σ vs chaos noise floor. Simplest and currently most robust.

### `archive_recording.py` — bundle a recording

```
uv run python archive_recording.py recordings/<folder>
# → writes manifest.md, cam0_warp.png, cam0_segmented.png, cam0_overlay.png
```

Auto-runs after each `r`-stop in `record.py`.

### Recordings format

Each `recordings/<ts>_<phase>/` folder contains:
- `cam0/*.png` — raw frames at native resolution (typically 1.4–2 GB; **gitignored**)
- `cam0.mp4` — H.264 near-lossless (CRF 14) encoding (~25–35 MB; **committed**)
- `cam0_keys.json` — the calibration applied to this recording
- `cam0_warp.png`, `cam0_segmented.png`, `cam0_overlay.png` — visualization snapshots
- `manifest.md` — metadata, frame count, key counts

`playback.py` reads from either source automatically: if the `cam0/*.png` folder is missing or empty, it falls back to `cam0.mp4` via OpenCV's `VideoCapture`. So fresh clones of the repo can run the full pipeline without needing the raw PNGs.

For teammates wanting raw PNG access (e.g., for detection development on lossless frames): tar the `cam0/` folder and attach to a GitHub Release. MP4 is fine for visual inspection but compression artifacts shift LSD line counts by ~10–30% — bit-exact analysis needs PNGs.

### Three committed reference clips

Three labeled phases, all using the same calibration mount:

| Folder | Phase | Duration | What's in it |
|---|---|---|---|
| `recordings/1777663774_rest/` | Rest | ~5 s | Empty keyboard, no hands. Used as canonical "rest baseline" for brightness / temp-diff / slope channels. |
| `recordings/1777663818_chaos/` | Chaos | ~40 s | Hands hovering above keys, casting shadows, fingers near keys but **no presses**. Measures noise floor under realistic playing conditions. |
| `recordings/1777663914_press/` | Press | ~56 s | Deliberate single-key presses, ~1 sec held each, varied positions across the keyboard. The actual test signal. |

### Press detection — current state

A single `Detector` class in `detection.py` is the per-frame engine for **both live (`record.py`'s `d` mode) and offline (`playback.py`)**. Identical algorithm in both — improvements propagate automatically.

Per-type channel routing:

- **Black keys** → `line OR slope OR tempdiff` (brightness omitted; black-key brightness is too noisy under shadows)
- **White keys** → `brightness OR slope OR tempdiff` (line omitted; white-on-white seams don't reliably fire LSD)

Plus 5-frame rolling-mean smoothing per channel before threshold comparison + temporal debounce on the fused flag.

**Threshold sources:**

- **Live** (`record.py`): captured in-session via the `b` hotkey's two-phase REST + CHAOS baseline. The `Detector` is incrementally populated — line thresholds from chaos, brightness/slope baselines from rest, tempdiff stats from chaos. Once both phases complete the detector runs at full capability. No dependency on offline analysis.
- **Offline** (`playback.py`): loaded from `analyze.py`'s `summary.csv` (per-key `chaos_max × margin × type_margin`) for the line channel; rest baselines computed from the sibling `_rest` clip; tempdiff stats from the `_chaos` clip. Useful for replaying a specific recording with detection overlay.

### Threshold tuning controls — what they actually mean

Each per-key score is compared to a **threshold**. If the smoothed score exceeds threshold for `--debounce` frames, that key fires red.

For the LINE channel, the per-key threshold is:
```
threshold[k] = chaos_max[k] × global_margin × (margin_black if k is black else margin_white)
```

| Knob | What it does |
|---|---|
| `--margin 1.0` (or `+`/`-` runtime) | Global multiplier on ALL per-key thresholds. Higher → less sensitive overall. |
| `--margin-black 1.5` (or `2`/`1` runtime) | Multiplier on BLACK-key thresholds only. Black keys have rich natural line activity, usually need higher margin. |
| `--margin-white 0.6` (or `4`/`3` runtime) | Multiplier on WHITE-key thresholds only. Whites need to be more sensitive (low chaos_max, real press signal can be small). |
| `--debounce 1` | Frames the score must STAY above threshold before firing. Higher kills brief flickers, slower to detect real presses. |
| `--smooth-window 3` | Rolling-mean window over per-channel scores. Larger smooths frame-to-frame jitter. |
| `--bright-margin 1.5` | Threshold multiplier specifically for the brightness channel. |
| `--tempdiff-sigma 0.5` | σ-floor for the tempdiff channel. Lower fires on smaller pixel deltas. |

**Where threshold tuning helps:** distinguishing borderline presses from borderline noise. **Where it doesn't:** any of the structural issues above (hand mask, shadows, white-on-white contrast). If the underlying signal is buried in those, no threshold combination recovers it.

### **Known limitation: skin masking is not yet reliable**

The current motion-based hand mask in `playback.py` was a successive set of attempts:

1. Static YCrCb (`analyze.py:skin_mask`) — broken on warm-lit white keys (matches whites as skin, leaves hands unmasked).
2. Motion-only (`|current - rest|` AND `|current - prev|`) — fails when hands stop moving (e.g., on a held press).
3. Motion + HSV color fusion + adaptive color sampling + 12-frame persistence + connected-component blob filter on tight pixels.

The current state (#3) is an improvement but still bleeds at hand edges and occasionally onto pressed-key boundaries. **Bill's `core/hand_gate.py` on the `visual_detection` branch** uses MediaPipe Hand Landmarker, which sidesteps the color/motion heuristics entirely and is likely the right path forward — port that to main when ready.

### Open issues

- **Hand masking** — see above. Most critical bottleneck for current detection accuracy.
- **White-key press signal** — LSD doesn't reliably fire on white-on-white seams at this resolution. Front-lip Y-tracking (per-key bottom edge position over time) would be more direct.
- **Per-key thresholds** from one-off chaos analysis don't transfer perfectly between sessions / lighting; would benefit from per-session chaos baseline (`b` hotkey in `record.py` does this for live mode).
