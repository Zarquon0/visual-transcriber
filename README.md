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
<img width="955" height="387" alt="Screenshot 2026-04-20 at 4 04 47 PM" src="https://github.com/user-attachments/assets/ef477d09-a225-4af2-aba7-3ccd34a96c52" />
The algorithm is pretty brittle currently - it's somewhat unstable even in the best conditions. For best results, be sure to position the camera such that the rails of the piano keys are roughly horizontal in the camera's output stream and such that the camera has a roughly top down view of the keys. If...

- ...the "threshed" view is missing some of the keys, lower the BRIGHTNESS_THRESHOLD parameter until all keys are clearly in view.
- ...the picked out "key" lines are not of the proper length or don't properly bound the keys, move the camera around, making sure all the keys of the piano are visible while moving closer to the piano (to cut out background pixels).
- ...all else looks fine but the warped view seems unstable, that's just because it kind of is. It shouldn't be *that* unstable though if everything else looks fine.

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

1. **Detect 4 loose corners** (`find_corners_auto`) from the white-key blob: isolate whites → horizontal dilate → pick largest wide-aspect contour → RANSAC-fit top/bottom rails → RANSAC-fit left/right rails on the pure-white lower region. Produces a trapezoid that follows perspective on all 4 sides.
2. **Tighten the corners to key tops** (`tighten_corners_to_tops`) using the physical piano geometry: black keys are ~0.7× as tall as white keys on the Oxygen 61. Warp with the loose corners, detect where black keys end, compute where white key tops should end via the ratio, inverse-project back to original coords for tight corners.
3. **Produce two warps** — **loose** (includes key front faces; for future press-detection via front-face motion) and **tight** (just the key tops; for labeling).
4. **Label the tight warp** (`draw_labels_tight_crop`): dynamically detects the black/white boundary (red line), finds black keys via column-brightness valleys with an **adaptive local threshold** (robust to lighting variation across the warp), draws per-key contours, derives yellow seam positions from detected black-key centers + wide-gap midpoints, and overlays note names (C#2..A#6 for blacks, C2..C7 for whites) assuming a 61-key keyboard with leftmost black = C#2.

### Scripts
- **`key_labeler.py`** — core detection + labeling primitives. Can be run standalone on a single photo (uses `find_keyboard_bbox` for a simple axis-aligned crop): `uv run python key_labeler.py path/to/photo.jpg` → writes `<photo>_labeled.png`.
- **`auto_calibrate.py`** — full 4-corner + tightening + labeling pipeline: `uv run python auto_calibrate.py piano_photos/IMG_9064.jpg piano_photos/IMG_9066.jpg` → writes `auto_calib_result.png` (grid: `corners | warped loose | warped tight + labels` per input).
- **`manual_calibrate.py`** — fallback 4-click calibration for shots auto fails on. Click TL, TR, BR, BL. Saves `<photo>_warped.png`, `<photo>_labeled.png`, `<photo>_calib.json` next to the input.

### Labeler output
On the tight warped image:
- **Red horizontal line**: detected black/white boundary (`y_black_bottom`).
- **Blue polygons**: actual pixel contours of black keys (follow the real key shape including chamfered fronts, not axis-aligned rectangles).
- **Yellow vertical lines**: white-key seams. Drawn partial-height (below the red line only) through each detected black key's center; drawn full-height at wide-gap midpoints (E|F and B|C positions where no black interrupts).
- **Note labels**: C#2–A#6 on blacks, C2–C7 on whites (assumes 61-key board with leftmost black = C#2; override `start_octave` in `_label_notes_61key` if different).

### Status

**Works end-to-end** on top-down / front-on shots with the whole keyboard roughly horizontal in frame: `IMG_9064`, `IMG_9065`, `IMG_9066`, `IMG_9072`, `IMG_9073`.

**Fails corner detection** on heavily-angled / side-view shots where the keyboard isn't an axis-aligned wide blob: `IMG_9067`, `IMG_9068`, `IMG_9070`, `IMG_9071`. `IMG_9069` is shot from under the keyboard (only stand visible) and isn't usable.

### What's next (being worked on)

- **Side-view detector**: a separate detection path for angled/tilted keyboards (9067/8/70/71). Direction I'm trying: strict-whiteness filter to drop floor/background + union of keyboard-shaped blobs (so a keyboard split by hand occlusion still recovers) → combined point cloud → `cv2.minAreaRect` for a rotated 4-corner box. Everything downstream stays the same.

### Ideas under consideration (not committed plans)

These are directions being explored, not decided architecture — listed so the context is in one place:

- **Dual-camera capture**: two upper-side-angle cams, each seeing the full keyboard, possibly fused at runtime. Whether to split the board into halves vs have each cam attempt the whole thing (with some form of confidence weighting) is still open. Fusion mechanism — simple max-of-confidence, weighted average, or something else — hasn't been designed.
- **Change detection for note events**: the eventual transcription step would compare per-frame key regions against a baseline to flag presses. Calibration here is intended as the one-time input to that runtime step, but the runtime pipeline itself hasn't been written.
- **Hand occlusion**: the current detection tries to union keyboard-shaped pieces, which helps with small splits. Whether this is enough on its own or needs camera redundancy (point above) is TBD.
- **Persisted calibration file** (corners + per-key regions + labels + whatever confidence info): a reasonable next step once there's agreement on format.

Nothing here is a commitment — marking these as exploration, not a plan.
