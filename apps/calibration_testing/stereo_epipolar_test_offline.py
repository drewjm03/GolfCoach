"""
Stereo epipolar geometry sanity check (offline).

- Uses stereo_offline_calibration_*.json from stereo_cam_calibrator_offline.py
- Loads one stereo keyframe (images + detections) from a keyframes directory
- Draws epipolar lines on cam1 for points chosen in cam0
"""

import os
import sys
import glob
import json
import argparse
import time
import math

import numpy as np
import cv2

# Reuse helpers for loading calib + keyframes from the pose script
try:
    from .stereo_pose_plot_offline import (
        _find_latest_calib_json,
        _load_calib,
        _load_keyframe_pair,
    )
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps.calibration_testing.stereo_pose_plot_offline import (  # type: ignore
        _find_latest_calib_json,
        _load_calib,
        _load_keyframe_pair,
    )


def _draw_epipolar_line(img, line, color, thickness=1):
    """Draw an epipolar line ax + by + c = 0 on image."""
    a, b, c = line
    h, w = img.shape[:2]

    pts = []
    # Try intersections with y=0 and y=h-1
    if abs(a) > 1e-6:
        x0 = -c / a
        x1 = -(b * (h - 1) + c) / a
        pts.append((x0, 0))
        pts.append((x1, h - 1))
    # If that was degenerate, try x=0 and x=w-1
    if len(pts) < 2 and abs(b) > 1e-6:
        y0 = -c / b
        y1 = -(a * (w - 1) + c) / b
        pts.append((0, y0))
        pts.append((w - 1, y1))

    if len(pts) < 2:
        return

    p1 = (int(round(pts[0][0])), int(round(pts[0][1])))
    p2 = (int(round(pts[1][0])), int(round(pts[1][1])))

    # Optionally clip to image bounds for safety
    cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Stereo epipolar geometry sanity check (offline).")
    parser.add_argument("--calib-json", type=str, default=None,
                        help="Path to stereo_offline_calibration_*.json. If omitted, use latest in data/.")
    parser.add_argument("--keyframes-dir", type=str, required=True,
                        help="Directory containing stereo keyframes (frame_XXX_cam*.png + frame_XXX.json).")
    parser.add_argument("--frame-index", type=int, default=None,
                        help="Optional frame index (XXX). If omitted, use last usable stereo frame.")
    parser.add_argument("--max-points", type=int, default=40,
                        help="Maximum number of tag centers to use for epipolar lines.")

    args = parser.parse_args()

    # Calibration JSON
    calib_json_path = _find_latest_calib_json(args.calib_json)
    (W, H), K0, D0, K1, D1, R, T, board_source = _load_calib(calib_json_path)
    image_size = (W, H)
    print(f"[CALIB] image_size={image_size}")

    # Fundamental matrix from calibration JSON
    with open(calib_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "F" not in data:
        raise RuntimeError("[CALIB] Fundamental matrix F not found in calibration JSON.")
    F = np.asarray(data["F"], dtype=np.float64)
    print("[CALIB] Loaded F from calibration JSON.")

    # Keyframe
    (frame_idx,
     frame0_bgr,
     frame1_bgr,
     corners0,
     ids0,
     corners1,
     ids1) = _load_keyframe_pair(args.keyframes_dir, args.frame_index)

    h1, w1 = frame1_bgr.shape[:2]
    assert (w1, h1) == image_size, f"Image size mismatch: calib={image_size}, frame1={w1,h1}"

    # Build id -> corners maps so we can match between cams
    ids0_flat = ids0.reshape(-1)
    ids1_flat = ids1.reshape(-1)

    map0 = {}
    for i, tid in enumerate(ids0_flat):
        if i < len(corners0):
            map0[int(tid)] = corners0[i].reshape(4, 2)

    map1 = {}
    for i, tid in enumerate(ids1_flat):
        if i < len(corners1):
            map1[int(tid)] = corners1[i].reshape(4, 2)

    common_ids = sorted(set(map0.keys()) & set(map1.keys()))
    if not common_ids:
        raise RuntimeError("[EPI] No common tag ids between cam0 and cam1 for this keyframe.")

    print(f"[EPI] Common tag ids: {len(common_ids)}")

    # Build matched centers
    pts0 = []
    pts1 = []
    for tid in common_ids:
        c0 = map0[tid]
        c1 = map1[tid]
        center0 = c0.mean(axis=0)
        center1 = c1.mean(axis=0)
        pts0.append(center0)
        pts1.append(center1)
        if len(pts0) >= args.max_points:
            break

    pts0 = np.asarray(pts0, dtype=np.float64)  # Nx2
    pts1 = np.asarray(pts1, dtype=np.float64)  # Nx2
    N = pts0.shape[0]
    print(f"[EPI] Using {N} matched tag centers for epipolar lines.")

    vis0 = frame0_bgr.copy()
    vis1 = frame1_bgr.copy()

    # Colors for visualization
    colors = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 128, 0),
    ]

    # Draw points + epipolar lines
    for i in range(N):
        u0, v0 = pts0[i]
        u1, v1 = pts1[i]

        x0_h = np.array([u0, v0, 1.0], dtype=np.float64)
        l1 = F @ x0_h  # (3,)
        a, b, c = l1
        denom = math.sqrt(a * a + b * b) if (a * a + b * b) > 0 else 1.0
        dist = abs(a * u1 + b * v1 + c) / denom

        col = colors[i % len(colors)]

        # Draw centers
        cv2.circle(vis0, (int(round(u0)), int(round(v0))), 4, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis1, (int(round(u1)), int(round(v1))), 4, col, -1, lineType=cv2.LINE_AA)

        # Draw epipolar line on cam1
        _draw_epipolar_line(vis1, l1, col, thickness=1)

        # Annotate distance on cam1
        cv2.putText(
            vis1,
            f"{dist:.2f}px",
            (int(round(u1)) + 5, int(round(v1)) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            col,
            1,
            lineType=cv2.LINE_AA,
        )

    # Compose side-by-side visualization
    canvas = np.hstack([vis0, vis1])

    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(data_dir, f"stereo_epipolar_test_{stamp}_frame{frame_idx:03d}.png")
    cv2.imwrite(out_png, canvas)
    print(f"[EPI] Wrote epipolar sanity image -> {out_png}")


if __name__ == "__main__":
    main()


