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
