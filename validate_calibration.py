"""Visual + sanity validation for ``<photo>_keys.json`` calibration files.

Run::

    uv run python validate_calibration.py piano_photos/live_1776972098_keys.json
    # or pass multiple
    uv run python validate_calibration.py piano_photos/*_keys.json

For each input it produces ``<photo>_validate.png`` showing:
- the warped image with every key's SAFE region filled (source-colored)
- per-key note label drawn at region center
- baseline intensity sanity: keys whose baseline is "wrong for type"
  (e.g. a black key reading >100, a white key reading <140) get
  outlined in RED so they jump out
- bottom-corner summary text: total keys, sources breakdown, # of
  baseline-violations

Source colors:
- detected             → bright green
- template_projected   → cyan
- inferred             → orange
- geometric (whites)   → soft yellow
- baseline-violation   → red outline overlay
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from key_labeler import load_image
from calibration import Calibration


SOURCE_COLORS_BGR = {
    "detected":           (40, 220, 40),    # bright green
    "template_projected": (220, 200, 0),    # cyan/teal
    "inferred":           (0, 140, 255),    # orange
    "geometric":          (40, 220, 220),   # soft yellow
}
DEFAULT_COLOR = (200, 200, 200)
VIOLATION_COLOR = (0, 0, 255)


# Per-type sanity bounds on baseline_intensity. Generous: a misaligned
# region typically reads near the opposite type's range, not the
# borderline.
BLACK_BASELINE_MAX = 100   # black-key safe region should be darker than this
WHITE_BASELINE_MIN = 140   # white-key safe region should be brighter than this


def _find_source_image(json_path: Path) -> Path | None:
    """Resolve the original photo for a ``<photo>_keys.json`` file.
    Strips the ``_keys.json`` suffix and tries common image extensions.
    """
    base = json_path.with_suffix("").as_posix()
    if base.endswith("_keys"):
        base = base[: -len("_keys")]
    for ext in (".jpg", ".jpeg", ".JPG", ".png", ".PNG", ".heic", ".HEIC"):
        candidate = Path(base + ext)
        if candidate.exists():
            return candidate
    return None


def _draw_validation(json_path: Path) -> Path | None:
    src_path = _find_source_image(json_path)
    if src_path is None:
        print(f"{json_path.name}: source image not found")
        return None

    rt = Calibration.load(json_path)
    img = load_image(str(src_path))
    warped = rt.warp(img)
    # Upscale so per-key outlines are individually visible at typical
    # warp widths (~700px). 2x = clear details without huge files.
    UP = 2
    warped = cv2.resize(warped, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    h, w = warped.shape[:2]
    out = warped.copy()
    raw_panel = warped.copy()

    # Sort keys left-to-right so adjacent-key alternating colors actually
    # alternate visually.
    keys_sorted = sorted(rt.keys, key=lambda k: k.bbox[0] + k.bbox[2] / 2)

    n_violations = 0
    src_counts: dict[str, int] = {}
    for i, k in enumerate(keys_sorted):
        src_counts[k.source] = src_counts.get(k.source, 0) + 1
        violates = (
            (k.type == "black" and k.baseline_intensity > BLACK_BASELINE_MAX)
            or (k.type == "white" and k.baseline_intensity < WHITE_BASELINE_MIN)
        )
        if violates:
            n_violations += 1
            color = VIOLATION_COLOR
            thickness = 3
        else:
            base_color = SOURCE_COLORS_BGR.get(k.source, DEFAULT_COLOR)
            # Alternate brightness on adjacent keys so neighbours visually
            # separate even when their polygons share an edge.
            if i % 2 == 0:
                color = base_color
            else:
                color = tuple(int(c * 0.55) for c in base_color)
            thickness = 2
        # JUST the polygon outline. No fill. The underlying warp stays
        # fully visible so you can verify each polygon hugs its key.
        # Upscale polygon coords to match the upscaled image.
        scaled_poly = (k.polygon * UP).astype(np.int32)
        cv2.drawContours(out, [scaled_poly], -1, color, thickness=thickness)

    # Labels in a SECOND pass on top of the outlines so they don't get
    # obscured. Position labels at the polygon's bbox center (upscaled).
    for k in keys_sorted:
        bx, by, bw, bh = k.bbox
        cx, cy = (bx + bw // 2) * UP, (by + bh // 2) * UP
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1
        (tw, th), _ = cv2.getTextSize(k.note, font, scale, thick)
        pos = (cx - tw // 2, cy + th // 2)
        cv2.putText(out, k.note, pos, font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(out, k.note, pos, font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    # Bottom legend / summary text.
    legend_h = 60
    legend = np.full((legend_h, w, 3), 30, dtype=np.uint8)
    summary = (
        f"keys={len(rt.keys)}  "
        + "  ".join(f"{k}={v}" for k, v in src_counts.items())
        + f"  baseline_violations={n_violations}"
    )
    cv2.putText(legend, summary, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (220, 220, 220), 1, cv2.LINE_AA)
    # Source-color legend
    x_cursor = 10
    for src, color in SOURCE_COLORS_BGR.items():
        cv2.rectangle(legend, (x_cursor, 35), (x_cursor + 18, 50), color, -1)
        cv2.putText(legend, src, (x_cursor + 22, 47), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220, 220, 220), 1, cv2.LINE_AA)
        x_cursor += 22 + len(src) * 7 + 14
    cv2.rectangle(legend, (x_cursor, 35), (x_cursor + 18, 50), VIOLATION_COLOR, -1)
    cv2.putText(legend, "violation", (x_cursor + 22, 47), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (220, 220, 220), 1, cv2.LINE_AA)

    # Stack: raw warp on top, polygon overlay below, then legend.
    # Lets you directly compare each key's visible body against the
    # stored region directly below it.
    sep = np.full((4, w, 3), 60, dtype=np.uint8)
    final = np.vstack([raw_panel, sep, out, legend])

    out_path = json_path.with_suffix("").as_posix()
    if out_path.endswith("_keys"):
        out_path = out_path[: -len("_keys")]
    out_path = Path(out_path + "_validate.png")
    cv2.imwrite(str(out_path), final)
    print(f"{json_path.name}: keys={len(rt.keys)}  "
          f"violations={n_violations}  →  {out_path.name}")
    return out_path


def main(paths: list[str]) -> None:
    if not paths:
        print("usage: uv run python validate_calibration.py <photo>_keys.json [...]")
        return
    written: list[Path] = []
    for p in paths:
        result = _draw_validation(Path(p))
        if result is not None:
            written.append(result)
    if written and sys.platform == "darwin":
        import subprocess
        subprocess.Popen(["open", *map(str, written)])


if __name__ == "__main__":
    main(sys.argv[1:])
