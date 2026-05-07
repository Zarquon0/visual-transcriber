The file called main.py should exhibit the following behavior:
When run, it should first:
1. Open all Canon camera streams and select the first one
2. Continuously run the stream through the warp to piano and key labeling functions and display the resulting image of the warped-to keyboard and its key labels

Then, when the user presses the space bar, it should:
1. Create a Calibration object based off of the frame that the user pressed the space bar on
2. Run each following frame through the warp the Calibration prescribes, mask any detected hands (via media pipe), and run difference-based keypress detection
3. Display a view of the segmented keyboard with keys with detected keypresses highlighted red. Should a DEBUG hyperparameter be set, the thresholded and blobbed per-frame difference image should also be displayed
4. Create a Transcriber object and update it with the detected keypresses every frame

When the user presses escape, the program should save to MIDI what the transcriber recorded and exit.