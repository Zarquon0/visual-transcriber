import argparse
import cv2
import numpy as np

from core.calibration import Calibration
from core.seg_to_keys import warp_to_piano


def fit(img, h=420):
    ih, iw = img.shape[:2]
    s = h / max(1, ih)
    return cv2.resize(img, (max(1, int(iw * s)), h))


p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--calib", required=True)
args = p.parse_args()

frame = cv2.imread(args.image)
if frame is None:
    raise SystemExit(f"could not read {args.image}")

rt = Calibration.load(args.calib)

result = warp_to_piano(frame)
warp_live = result[0] if isinstance(result, tuple) else result

warp_saved = rt.warp(frame)

print("warp_to_piano:", None if warp_live is None else warp_live.shape)
print("rt.warp:", None if warp_saved is None else warp_saved.shape)
print("calib warp_size:", rt.warp_size)

display = np.hstack([
    fit(frame),
    fit(warp_live),
    fit(warp_saved),
])

cv2.imwrite("debug_compare_warps.jpg", display)
print("wrote debug_compare_warps.jpg")

cv2.imshow("raw | warp_to_piano | rt.warp", display)
cv2.waitKey(0)