"""analyze.py — offline characterization of press-detection channels.

Two channels evaluated per key per frame:
  C1 (brightness): mean intensity inside the full polygon (skin-masked),
      delta from rest baseline, with global-illumination correction
      (median white-key shift). This is the Bill-style intensity signal,
      moved off the safe_bbox onto the full polygon as discussed.
  C2 (anomalous lines): LSD on the skin-masked warp; for each segment,
      check whether its midpoint lies inside a polygon AND outside that
      polygon's boundary-exclusion band. Interior segments accumulate
      length per key.

Usage:
    uv run python analyze.py \\
        recordings/<ts>_rest \\
        recordings/<ts>_chaos \\
        recordings/<ts>_press

Outputs (recordings/_analysis/<ts>/):
    summary.csv     — per-key {max,snr} for each channel × phase
    summary.png     — scatter: brightness SNR vs line SNR per key
    timeseries.png  — sample of per-key time series across phases
"""
from __future__ import annotations
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ─── parameters ────────────────────────────────────────────────────────────
EPS_BOUNDARY = 3            # px; segments within this of a polygon edge ignored
SKIN_DILATE = 5             # px; widen skin mask to also kill hand-edge gradients
STRIDE = 1                  # frame stride (1 = every frame)
SAMPLE_KEY_COUNT = 6        # how many representative keys to plot timeseries
# YCrCb skin bounds. Default 133..173 / 77..127 misfires on warm-lit
# piano keys whose Cr lands ~134. Tightened to require stronger redness.
SKIN_LO = (40, 140, 85)
SKIN_HI = (240, 175, 130)


def build_overlays(keys_dict: dict):
    """Pre-rasterize per-frame work into static masks (warped-strip space)."""
    W, H = keys_dict["warp"]["out_size"]
    keys = keys_dict["keys"]
    polys = [np.array(k["polygon"], dtype=np.int32).reshape(-1, 1, 2)
             for k in keys]
    types = [k["type"] for k in keys]

    # key_id_map[y, x] = which key index contains this pixel (-1 if none).
    key_id_map = np.full((H, W), -1, dtype=np.int32)
    for ki, poly in enumerate(polys):
        cv2.fillPoly(key_id_map, [poly], ki)
    # Per-key boolean mask + area. The 2-px erosion that used to live
    # here was originally a cross-fire guard (a press blob spanning
    # two polygons no longer "doubled fires"). It became redundant
    # once the diff path adopted the cam-side blob-to-key rule
    # (detection._process_diff), which deterministically attributes a
    # whole blob to a single key by perspective-bias direction. With
    # the rule in place, eroding the polygons just throws away real
    # press signal at the seams.
    ERODE_KEY_PX = 0
    erode_kernel = np.ones(
        (ERODE_KEY_PX * 2 + 1, ERODE_KEY_PX * 2 + 1), dtype=np.uint8
    )
    per_key_mask = [np.zeros((H, W), dtype=np.uint8) for _ in polys]
    per_key_area = np.zeros(len(polys), dtype=np.float32)
    for ki, poly in enumerate(polys):
        cv2.fillPoly(per_key_mask[ki], [poly], 1)
        if ERODE_KEY_PX > 0:
            per_key_mask[ki] = cv2.erode(per_key_mask[ki], erode_kernel)
        per_key_area[ki] = float(per_key_mask[ki].sum())
    # boundary band: dilated polygon edge for ALL polygons, used as the
    # "expected lines exist here" exclusion mask for C2.
    boundary_band = np.zeros((H, W), dtype=np.uint8)
    for poly in polys:
        cv2.polylines(boundary_band, [poly], True, 1, 1, cv2.LINE_AA)
    boundary_band = cv2.dilate(
        boundary_band, np.ones((EPS_BOUNDARY * 2 + 1, EPS_BOUNDARY * 2 + 1), np.uint8)
    )
    # perspective warp matrix
    src = np.array(keys_dict["warp"]["corners_tl_tr_br_bl"], dtype=np.float32)
    dst = np.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(src, dst)

    return {
        "W": W, "H": H, "M": M, "types": types,
        "per_key_mask": per_key_mask,
        "per_key_area": per_key_area,
        "key_id_map": key_id_map,
        "boundary_band": boundary_band,
    }


def skin_mask(bgr_warped: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr_warped, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycrcb, SKIN_LO, SKIN_HI)
    if SKIN_DILATE > 0:
        m = cv2.dilate(m, np.ones((SKIN_DILATE * 2 + 1, SKIN_DILATE * 2 + 1), np.uint8))
    return m  # 0/255


def channel_brightness(gray: np.ndarray, skin: np.ndarray, ov: dict) -> np.ndarray:
    """Per-key mean intensity within polygon, ignoring skin pixels.
    Returns array length n_keys (NaN where polygon is fully skin-occluded).
    """
    n = len(ov["per_key_mask"])
    out = np.full(n, np.nan, dtype=np.float32)
    skin_b = skin > 0
    for ki, mask in enumerate(ov["per_key_mask"]):
        m = mask.astype(bool) & ~skin_b
        if m.sum() < 0.1 * ov["per_key_area"][ki]:
            continue  # > 90% skin-occluded
        out[ki] = float(gray[m].mean())
    return out


_LSD = cv2.createLineSegmentDetector()


def channel_temp_diff(gray: np.ndarray, skin: np.ndarray, ov: dict,
                      rest_gray: np.ndarray) -> np.ndarray:
    """Per-key mean absolute pixel-difference vs the rest mean frame
    (skin-masked). The dumbest possible motion signal: 'did anything
    change in this region?' Robust to LSD's frame-to-frame jitter."""
    n = len(ov["per_key_mask"])
    out = np.full(n, np.nan, dtype=np.float32)
    diff = cv2.absdiff(gray, rest_gray)
    skin_b = skin > 0
    for ki, mask in enumerate(ov["per_key_mask"]):
        m = mask.astype(bool) & ~skin_b
        if m.sum() < 0.1 * ov["per_key_area"][ki]:
            continue
        out[ki] = float(diff[m].mean())
    return out


def channel_slope(gray: np.ndarray, skin: np.ndarray, ov: dict) -> np.ndarray:
    """Per-key weighted-mean angle (degrees) of LSD segments whose
    midpoints fall inside the polygon. Returns NaN where no segments
    found (so baselines and deltas can skip those frames).

    Press tilts a key, which shifts the angles of its existing detected
    lines. Subtracting a rest baseline gives a slope-change signal that
    works on keys with rich existing lines (blacks especially).
    """
    n = len(ov["per_key_mask"])
    out = np.full(n, np.nan, dtype=np.float32)
    g = gray.copy()
    g[skin > 0] = 128
    res = _LSD.detect(g)
    if res is None or res[0] is None:
        return out
    lines = res[0].reshape(-1, 4)
    if lines.size == 0:
        return out
    mx = ((lines[:, 0] + lines[:, 2]) * 0.5).astype(np.int32)
    my = ((lines[:, 1] + lines[:, 3]) * 0.5).astype(np.int32)
    H, W = ov["H"], ov["W"]
    valid = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
    mx, my = mx[valid], my[valid]
    dx = (lines[:, 2] - lines[:, 0])[valid]
    dy = (lines[:, 3] - lines[:, 1])[valid]
    lengths = np.hypot(dx, dy)
    # Angle in degrees, mod 180 (a line and its reverse are the same).
    angles = (np.degrees(np.arctan2(dy, dx)) % 180.0)
    key_ids = ov["key_id_map"][my, mx]
    on_skin = skin[my, mx] > 0
    keep = (key_ids >= 0) & ~on_skin
    if not np.any(keep):
        return out
    sums = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    for ki, a, w in zip(key_ids[keep], angles[keep], lengths[keep]):
        ki = int(ki)
        sums[ki] += a * w
        weights[ki] += w
    mask = weights > 0
    out[mask] = (sums[mask] / weights[mask]).astype(np.float32)
    return out


def channel_lines(gray: np.ndarray, skin: np.ndarray, ov: dict) -> np.ndarray:
    """Per-key total length of LSD segments whose midpoints are interior
    to the polygon (not in boundary_band) and not on skin pixels.
    """
    n = len(ov["per_key_mask"])
    out = np.zeros(n, dtype=np.float32)
    # Suppress skin edges by setting skin pixels to a flat mid-gray.
    g = gray.copy()
    g[skin > 0] = 128
    res = _LSD.detect(g)
    if res is None or res[0] is None:
        return out
    lines = res[0].reshape(-1, 4)  # (x1, y1, x2, y2)
    if lines.size == 0:
        return out
    mx = ((lines[:, 0] + lines[:, 2]) * 0.5).astype(np.int32)
    my = ((lines[:, 1] + lines[:, 3]) * 0.5).astype(np.int32)
    H, W = ov["H"], ov["W"]
    valid = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
    mx, my = mx[valid], my[valid]
    lengths = np.linalg.norm(
        lines[:, 2:4] - lines[:, 0:2], axis=1
    )[valid]
    key_ids = ov["key_id_map"][my, mx]
    on_boundary = ov["boundary_band"][my, mx] > 0
    on_skin = skin[my, mx] > 0
    keep = (key_ids >= 0) & ~on_boundary & ~on_skin
    for ki, L in zip(key_ids[keep], lengths[keep]):
        out[int(ki)] += float(L)
    return out


def process_folder(folder: Path, ov: dict, stride: int = STRIDE):
    """Returns (T, n_keys) arrays for brightness_raw and lines, plus the
    per-frame global-shift values (median delta across white keys)."""
    cam = folder / "cam0"
    frames = sorted(cam.glob("*.png"))[::stride]
    n_keys = len(ov["per_key_mask"])
    bright = np.full((len(frames), n_keys), np.nan, dtype=np.float32)
    lines = np.zeros((len(frames), n_keys), dtype=np.float32)
    for fi, fp in enumerate(frames):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        warped = cv2.warpPerspective(bgr, ov["M"], (ov["W"], ov["H"]))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sk = skin_mask(warped)
        bright[fi] = channel_brightness(gray, sk, ov)
        lines[fi] = channel_lines(gray, sk, ov)
    return bright, lines, frames


def apply_global_shift(bright: np.ndarray, baseline: np.ndarray, types: list[str]):
    """Subtract per-frame median (delta over white keys) from every key's
    delta. Cancels room lighting drift / camera gain. Returns delta array."""
    delta = bright - baseline[None, :]
    white_idx = np.array([i for i, t in enumerate(types) if t == "white"])
    if len(white_idx) == 0:
        return delta
    # nan-aware median across white keys per frame
    white_deltas = delta[:, white_idx]
    shifts = np.nanmedian(white_deltas, axis=1)
    return delta - shifts[:, None]


def per_key_summary(bright_delta: np.ndarray, lines: np.ndarray):
    """Per-key max(|delta|) for brightness; max(length) for lines.
    Returns (bright_max, line_max), each shape (n_keys,)."""
    b = np.nanmax(np.abs(bright_delta), axis=0) if bright_delta.shape[0] else np.zeros(bright_delta.shape[1])
    L = lines.max(axis=0) if lines.shape[0] else np.zeros(lines.shape[1])
    return b, L


def main():
    if len(sys.argv) < 4:
        print("usage: uv run python analyze.py <rest_dir> <chaos_dir> <press_dir>")
        raise SystemExit(2)
    rest_dir = Path(sys.argv[1])
    chaos_dir = Path(sys.argv[2])
    press_dir = Path(sys.argv[3])

    keys_path = rest_dir / "cam0_keys.json"
    if not keys_path.exists():
        raise SystemExit(f"missing {keys_path}")
    keys_dict = json.loads(keys_path.read_text())
    types = [k["type"] for k in keys_dict["keys"]]
    n_keys = len(types)
    print(f"calibration: {n_keys} keys ({sum(t=='black' for t in types)} blacks, "
          f"{sum(t=='white' for t in types)} whites)")

    ov = build_overlays(keys_dict)

    print("processing REST...");   t = time.perf_counter()
    rest_b, rest_L, _ = process_folder(rest_dir, ov)
    print(f"  {rest_b.shape[0]} frames in {time.perf_counter()-t:.1f}s")
    print("processing CHAOS...");  t = time.perf_counter()
    chaos_b, chaos_L, _ = process_folder(chaos_dir, ov)
    print(f"  {chaos_b.shape[0]} frames in {time.perf_counter()-t:.1f}s")
    print("processing PRESS...");  t = time.perf_counter()
    press_b, press_L, _ = process_folder(press_dir, ov)
    print(f"  {press_b.shape[0]} frames in {time.perf_counter()-t:.1f}s")

    # Brightness baseline = mean over rest frames per key (nan-safe).
    baseline = np.nanmean(rest_b, axis=0)
    rest_delta = apply_global_shift(rest_b, baseline, types)
    chaos_delta = apply_global_shift(chaos_b, baseline, types)
    press_delta = apply_global_shift(press_b, baseline, types)

    # Per-key summary stats
    rest_bm, rest_Lm = per_key_summary(rest_delta, rest_L)
    chaos_bm, chaos_Lm = per_key_summary(chaos_delta, chaos_L)
    press_bm, press_Lm = per_key_summary(press_delta, press_L)

    # SNR = press peak above chaos mean, in chaos-std units (proper z-score).
    # Floor on std avoids divide-by-zero when chaos is silent on that key.
    chaos_b_mean = np.nanmean(chaos_delta, axis=0) if chaos_delta.size else np.zeros(n_keys)
    chaos_b_std = np.nanstd(chaos_delta, axis=0) if chaos_delta.size else np.ones(n_keys)
    chaos_L_mean = chaos_L.mean(axis=0) if chaos_L.size else np.zeros(n_keys)
    chaos_L_std = chaos_L.std(axis=0) if chaos_L.size else np.ones(n_keys)
    bright_snr = (press_bm - chaos_b_mean) / np.maximum(chaos_b_std, 1.0)
    line_snr = (press_Lm - chaos_L_mean) / np.maximum(chaos_L_std, 1.0)

    out_dir = Path(f"recordings/_analysis/{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "key_idx", "type",
            "B_rest_max", "B_chaos_max", "B_press_max", "B_SNR",
            "L_rest_max", "L_chaos_max", "L_press_max", "L_SNR",
        ])
        for i in range(n_keys):
            w.writerow([
                i, types[i],
                f"{rest_bm[i]:.2f}", f"{chaos_bm[i]:.2f}", f"{press_bm[i]:.2f}", f"{bright_snr[i]:.2f}",
                f"{rest_Lm[i]:.1f}", f"{chaos_Lm[i]:.1f}", f"{press_Lm[i]:.1f}", f"{line_snr[i]:.2f}",
            ])
    print(f"wrote {csv_path}")

    # Scatter: brightness SNR vs line SNR per key.
    fig, ax = plt.subplots(figsize=(7, 6))
    is_white = np.array([t == "white" for t in types])
    ax.scatter(bright_snr[is_white], line_snr[is_white], c="goldenrod", label="white", s=40, alpha=0.7)
    ax.scatter(bright_snr[~is_white], line_snr[~is_white], c="purple", label="black", s=40, alpha=0.7)
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.axvline(1.0, color="gray", lw=0.5)
    ax.set_xlabel("Brightness SNR (press_max / chaos_max)")
    ax.set_ylabel("Line SNR (press_max / chaos_max)")
    ax.set_title("Per-key signal-to-noise: brightness vs anomalous-line")
    ax.legend()
    ax.set_xscale("log"); ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir/'summary.png'}")

    # Per-key time series — pick a representative sample.
    sample_count = min(SAMPLE_KEY_COUNT, n_keys)
    # Sort by line_snr descending; take the top keys (most informative).
    sample_idx = np.argsort(-line_snr)[:sample_count].tolist()
    fig, axes = plt.subplots(sample_count, 2, figsize=(14, 2.5 * sample_count), sharex="col")
    if sample_count == 1:
        axes = axes[None, :]
    for row, ki in enumerate(sample_idx):
        # Concatenate phases with separators for visual reference.
        b_series = np.concatenate([rest_delta[:, ki], chaos_delta[:, ki], press_delta[:, ki]])
        L_series = np.concatenate([rest_L[:, ki], chaos_L[:, ki], press_L[:, ki]])
        n_r, n_c = rest_delta.shape[0], chaos_delta.shape[0]
        for ax, series, title in zip(axes[row], [b_series, L_series], ["brightness Δ (skin-masked)", "anomalous-line length"]):
            ax.plot(series, lw=0.7)
            ax.axvline(n_r, color="gray", lw=0.5, ls="--")
            ax.axvline(n_r + n_c, color="gray", lw=0.5, ls="--")
            ax.set_title(f"key {ki} ({types[ki]}) — {title}")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "timeseries.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir/'timeseries.png'}")

    # Quick stdout digest.
    print("\nTop-10 keys by line SNR:")
    order = np.argsort(-line_snr)[:10]
    for i in order:
        print(f"  key {i:2d} ({types[i]:5}): B_SNR={bright_snr[i]:5.2f}  L_SNR={line_snr[i]:5.2f}")
    print("\nBottom-5 keys by line SNR:")
    order_b = np.argsort(line_snr)[:5]
    for i in order_b:
        print(f"  key {i:2d} ({types[i]:5}): B_SNR={bright_snr[i]:5.2f}  L_SNR={line_snr[i]:5.2f}")
    print(f"\nSee {out_dir} for full plots + CSV.")


if __name__ == "__main__":
    main()
