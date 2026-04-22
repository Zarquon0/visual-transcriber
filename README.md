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


## Auto Calibration + Tight-Crop Labeler (alternate pipeline)

An alternative detection pipeline lives in `auto_calibrate.py` + `key_labeler.py`, operating on **static images** rather than live streams. The Hough-based warp in `seg_to_keys.py` picks background horizontals (shelves, floor tiles) as keyboard rails on real hand-held photos, and the labeler in `key_extractor2.py` assumes the warp leaves padding above/below the keys. This alt pipeline works on *tight* warps where the whole warped image is the keyboard.

### iPhone Continuity Camera

`stream_webcams.py`'s device filter accepts iPhones now, so Continuity Camera works for `key_extractor2.py --mode live` without needing Canon hardware. The alt pipeline scripts below are all static-image; they don't use streams.

### Scripts
- **`key_labeler.py`** — core detection + labeling. Uses a simple axis-aligned bbox crop (via `find_keyboard_bbox`). Run on a single photo:
  ```
  uv run python key_labeler.py path/to/photo.jpg
  ```
  Writes `<photo>_labeled.png` next to the input.
- **`auto_calibrate.py`** — automatic 4-corner detection via blob geometry + RANSAC rail fit on all 4 sides (proper trapezoidal perspective warp, not axis-aligned). Then the labeler. Run on one or more photos:
  ```
  uv run python auto_calibrate.py piano_photos/IMG_9064.jpg piano_photos/IMG_9066.jpg
  ```
  Writes `auto_calib_result.png` — a grid with `corners | warped | warped+labels` per input.
- **`manual_calibrate.py`** — fallback 4-click calibration for shots where auto fails. Click TL, TR, BR, BL. Saves three artifacts next to the input photo: `<photo>_warped.png`, `<photo>_labeled.png`, `<photo>_calib.json` (corners in original-image coords).

### Labeler output
On the warped keyboard image:
- Red horizontal line = detected black/white boundary (`y_black_bottom`)
- Blue polygons = actual pixel contours of black keys (not axis-aligned rectangles, so perspective slant and chamfered fronts are preserved)
- Yellow vertical lines = white-key seams. Full-height at E-F / B-C gaps (no black key interrupting); below the red line elsewhere.

### Known issues
- `IMG_9073` (front-on, keyboard is a narrow horizontal strip in the frame) still has drift in yellow-seam placement even though the warp itself is decent. Top-down-ish shots like 9064 and 9066 work well end-to-end.
- Auto calibration's left/right rail fit is pre-filtered to the outermost 40% of points (see the percentile pre-filter in `find_corners_auto`) so shadow-edge rows don't drag the rail inward. If it still misses rightmost keys on your photos, relaxing that percentile is the first knob to try; falling back to `manual_calibrate.py` is the second.
