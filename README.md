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


