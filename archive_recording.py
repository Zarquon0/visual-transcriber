"""archive_recording.py — bundle a recording folder with its calibration
artifacts (warp, colored segmentation, manifest) so the data is
self-describing and reproducible.

STATUS (current pipeline):
    Standalone CLI utility. Not imported by the live transcription
    pipeline; run post-hoc to package raw frame folders together with
    their calibration into a self-describing archive.

Usage:
    uv run python archive_recording.py recordings/<folder>

For each cam<i> in the folder, expects cam<i>_keys.json to already be
present. Reads frame 0 of cam<i>/, computes:
    cam<i>_warp.png       — rectified strip from the calibration warp
    cam<i>_segmented.png  — same warp with colored per-key polygons + indices
    cam<i>_overlay.png    — frame 0 with polygons drawn on the source view

Writes manifest.md describing the clip (frame count, duration at 30fps,
calibration source, key counts).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


def warp_with_keys(frame: np.ndarray, keys_dict: dict) -> np.ndarray:
    src = np.array(keys_dict["warp"]["corners_tl_tr_br_bl"], dtype=np.float32)
    W, H = keys_dict["warp"]["out_size"]
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (W, H))


def archive(folder: Path):
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")

    cams = sorted([p for p in folder.iterdir() if p.is_dir() and p.name.startswith("cam")])
    if not cams:
        raise SystemExit(f"no cam<i>/ subfolders in {folder}")

    # Lazy-import the visualizer from record.py so we share the palette logic.
    from record import draw_warp_colored, draw_overlay, overlay_from_dict

    manifest_lines = [
        f"# Recording: {folder.name}",
        "",
        f"Archived: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for cam in cams:
        ci = cam.name  # e.g. "cam0"
        keys_path = folder / f"{ci}_keys.json"
        if not keys_path.exists():
            print(f"  {ci}: no keys.json bundled, skipping segmentation artifacts")
            manifest_lines += [f"## {ci}", f"- frames: {len(list(cam.glob('*.png')))}", ""]
            continue

        keys_dict = json.loads(keys_path.read_text())
        frames = sorted(cam.glob("*.png"))
        if not frames:
            print(f"  {ci}: no frames")
            continue
        first = cv2.imread(str(frames[0]))
        if first is None:
            print(f"  {ci}: could not read first frame")
            continue

        warped = warp_with_keys(first, keys_dict)
        cv2.imwrite(str(folder / f"{ci}_warp.png"), warped)
        segmented = draw_warp_colored(warped, keys_dict)
        cv2.imwrite(str(folder / f"{ci}_segmented.png"), segmented)

        polys, sbbs, types = overlay_from_dict(keys_dict)
        overlay = draw_overlay(first, polys, sbbs, types)
        cv2.imwrite(str(folder / f"{ci}_overlay.png"), overlay)

        n_keys = len(keys_dict["keys"])
        n_blacks = sum(1 for k in keys_dict["keys"] if k["type"] == "black")
        n_whites = sum(1 for k in keys_dict["keys"] if k["type"] == "white")
        n_labeled = sum(1 for k in keys_dict["keys"] if k["note"] != "?")
        nf = len(frames)
        manifest_lines += [
            f"## {ci}",
            f"- frames: {nf} (~{nf / 30:.1f}s at 30fps)",
            f"- calibration: {keys_path.name}",
            f"- keys: {n_keys} (blacks: {n_blacks}, whites: {n_whites}, labeled: {n_labeled})",
            f"- warp size: {keys_dict['warp']['out_size']}",
            f"- y_black_bottom: {keys_dict['y_black_bottom']}",
            "",
            f"![warp]({ci}_warp.png)",
            f"![segmented]({ci}_segmented.png)",
            f"![overlay]({ci}_overlay.png)",
            "",
        ]
        print(f"  {ci}: warp + segmented + overlay written")

    (folder / "manifest.md").write_text("\n".join(manifest_lines))
    print(f"  wrote {folder}/manifest.md")


def main():
    if len(sys.argv) < 2:
        print("usage: uv run python archive_recording.py recordings/<folder> [more...]")
        raise SystemExit(2)
    for arg in sys.argv[1:]:
        p = Path(arg)
        print(f"archiving {p}")
        archive(p)


if __name__ == "__main__":
    main()
