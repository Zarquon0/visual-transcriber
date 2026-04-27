## Set Up
Clone, then:
```bash
uv sync
```
To run scripts:
```
uv run python <script>
```
## Webcam Streaming
`stream_webcams.py` creates cv2 connections to attached Canon R50 cameras. To properly connect cameras:

1. Set all Canon R50 cameras into webcam streaming mode. To do so, power on each camera and hit Menu (button) > yellow wrench (tab) > 4 (tab) > select "Video Streaming". The feed on the camera's display will change slightly in indication of this.
2. Connect cameras to your Mac (only Macs are supported currently) via USB-c cord. NOTE: it is recommended that you connect each camera to a separate USB-c port on your computer (as opposed to connecting multiple to an adapter) to limit delays or freezes.
3. Run `uv run python stream_webcams.py`. You should see all connected cameras' streams (you may need to move a video window to see occluded ones).

If one or more video streams are missing, you can try using `cam_probe.py` and `cam_identify.py` to debug. `stream_webcams.py` should automatically find and detect connected Canon cameras, but this functionality has not been thoroughly tested.
 
To change the image resolution or FPS (maybe - not sure FPS control works), alter the values in the `camera_config.yaml` file.

## Crop/Warp to Piano
`seg_to_keys.py` contains functionality for cropping/warping the view to fit the piano keys. To give it a test, simply run connect a Canon camera and run the script. You should see a live stream that looks something like the following:
<img width="1367" height="551" alt="Screenshot 2026-04-25 at 1 40 43 PM" src="https://github.com/user-attachments/assets/ed681501-d025-4cbc-ba7e-69251cf2a138" />
<img width="1374" height="510" alt="Screenshot 2026-04-25 at 1 39 42 PM" src="https://github.com/user-attachments/assets/682c2714-a4d9-457d-9461-2219474c3e88" />

For best results, be sure to align the bottom and top rails of the keyboard parallel with the camera display. In generally normal lighting and without an excess of bright objects in frame, it should work well!

## Crop/Warp to Piano
`key_extractor2.py` contains functionality attempting to segment/detect the keys on the keyboard. The code uses edge detection and sobel filtering as part of the functionality to find the white key and black key edges for piano. Still pending significant improvement. Line 242 is where you can adjust the threshold of how sensitive the edge detector is. Can be run in 2 views: live uses functionalities from what was originally test_hough (now seg_to_keys.py). 

to run:
static image (more stable): uv run key_extractor2.py --mode image --image test_piano.avif
live: uv run key_extractor2.py --mode live


## Auto Calibration + Tight-Crop Labeler + Note Labeling (alternate pipeline)

A static-image keyboard-detection + labeling pipeline lives in `auto_calibrate.py` + `key_labeler.py`. Separate from the live-stream pipeline in `seg_to_keys.py` / `key_extractor2.py`, and tuned for hand-held photos where that one struggles (the Hough rail finder picks up shelves/tiles as "rails", and the labeler assumes padding above/below the keys that doesn't exist in a tight crop).

### iPhone Continuity Camera

`stream_webcams.py`'s device filter accepts iPhones, so Continuity Camera works for `key_extractor2.py --mode live` without Canon hardware plugged in. The alt pipeline scripts below are all static-image.

### Pipeline overview

1. **Detect 4 loose corners** (`find_corners_auto`) from the white-key blob (top-down shots): isolate whites → horizontal dilate → pick largest wide-aspect contour → RANSAC-fit top/bottom rails → RANSAC-fit left/right rails on the pure-white lower region. For **side-view shots** where this fails, a manual `<photo>_calib.json` next to the image overrides auto-detection (these calib files are committed per camera mount).
2. **Tighten the corners to key tops** (`tighten_corners_to_tops`) using the physical piano geometry: black keys are ~0.7× as tall as white keys on the Oxygen 61. Warp with the loose corners, detect where black keys end, compute where white key tops should end via the ratio, inverse-project back to original coords for tight corners.
3. **Produce a tight warp** of just the key tops (= the keyboard's playable surface, no body / floor / front-face). This is what the labeler annotates AND what gets stored as the live runtime warp in `<photo>_keys.json` — so the next dev's press-detection runs against this same warp on every live frame. (The pipeline also computes a "loose" warp that includes the key front faces; this was an earlier idea for press-via-front-face-motion but is currently unused — the per-key safe-region masks generated from the tight warp serve press detection directly.)
4. **Label the tight warp** (`draw_labels_tight_crop`):
   - Find `y_black_bottom` (the red line) via Sobel-y horizontal edge.
   - **Black-key detection** (`_detect_blacks_2d`): 2D Otsu connected-component on the upper band. Each merged blob (multi-key region) is split via **U-valley analysis** on the bottom-y profile of the blob mask, then each inner piece gets the **camera-far outer piece's actual contour** projected onto it as a local template. Z-order clipping resolves overlap between adjacent projected pieces. Falls back to 1D column-projection when 2D fails on top-down shots.
   - **SWSSW projection** (`_project_to_25`) aligns detected blacks to the canonical 25-key pattern and fills any still-missing positions with translated template polygons.
   - **Geometric edge guard** trims any polygon that over-extends past the keyboard's playable area (`0.5 * white_key_w` from the left edge, `1.5 * white_key_w` from the right edge — the 61-key C-to-C layout's natural buffer for C2 and B6+C7).
   - **White-key seams**: Sobel-x peak detection on the white band, with local-median gap-fill (between detected peaks) and edge-extrapolation (past the first/last detected peak). Each seam draws through every row where no black-key polygon covers its column — one unified rule for partial vs full-height.
   - **Note labels**: hardcoded canonical SWSSW pattern assigns letters; final labels are C#2..A#6 for blacks, C2..C7 for whites. Auto-scaled font size + 2-row stagger keeps labels readable on narrow side-view warps.

The `far_side` parameter on `draw_labels_tight_crop` / `_detect_blacks_2d` selects the camera-far direction (``"right"`` or ``"left"``). The 4-corner detection, geometric edge guard, and seam pipeline are all **camera-agnostic**; only the per-blob template-projection step is camera-side dependent. In a dual-cam rig each camera sets its own `far_side`.

### Scripts
- **`key_labeler.py`** — core detection + labeling primitives. Can be run standalone on a single photo (uses `find_keyboard_bbox` for a simple axis-aligned crop): `uv run python key_labeler.py path/to/photo.jpg` → writes `<photo>_labeled.png`.
- **`auto_calibrate.py`** — full 4-corner + tightening + labeling pipeline: `uv run python auto_calibrate.py piano_photos/IMG_9064.jpg piano_photos/IMG_9066.jpg` → writes `auto_calib_result.png` (grid: `corners | warped loose | warped tight + labels` per input).
- **`manual_calibrate.py`** — fallback 4-click calibration for shots auto fails on. Click TL, TR, BR, BL. Saves `<photo>_warped.png`, `<photo>_labeled.png`, `<photo>_calib.json` next to the input.

### Labeler output
On the tight warped image:
- **Red horizontal line**: detected black/white boundary (`y_black_bottom`).
- **Blue polygons**: black-key outlines. Either the actual `cv2.findContours` contour for unmerged blobs, or the camera-far outer piece's contour translated to inner pieces of merged blobs (with overlap resolved by Z-order: closer key wins).
- **Yellow vertical lines**: white-key seams. Drawn through every row of each seam's column where no black-key polygon covers that row — single unified rule, so seams go full-height in E|F / B|C gaps and clip above any black-key body otherwise.
- **Note labels**: C#2–A#6 on blacks, C2–C7 on whites (assumes 61-key board with leftmost black = C#2; override `start_octave` in `_label_notes_61key` if different). Labels stagger across two y-rows so adjacent labels don't overlap on narrow warps.

### Status

**Top-down / front-on shots** (`IMG_9064`, `IMG_9065`, `IMG_9066`, `IMG_9072`, `IMG_9073`): auto corner detection + labeling work end-to-end.

**Side-view shots** (`live_1776971374`, `live_1776972098`, plus the older `IMG_9067/8/70/71`): auto corner detection currently fails on these, so they use a manual `<photo>_calib.json` next to the image (4-click corners via `manual_calibrate.py`, then committed). The detection pipeline downstream of corners works on the manually-calibrated warps — extreme foreshortening on the camera-far end still produces some imperfect outlines (see "Remaining bugs" below).

`IMG_9069` is shot from under the keyboard (only the stand visible) and isn't usable.

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
