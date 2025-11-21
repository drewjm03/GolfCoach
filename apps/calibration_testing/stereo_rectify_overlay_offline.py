"""
Stereo rectification + horizontal scanline overlay (offline).

- Uses stereo_offline_calibration_*.json from stereo_cam_calibrator_offline.py
- Loads one stereo keyframe (images + detections) from a keyframes directory
- Rectifies both views and draws horizontal scanlines to visualize alignment
"""

import os
import sys
import argparse
import time

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


def main():
    parser = argparse.ArgumentParser(description="Stereo rectification + scanline overlay (offline).")
    parser.add_argument("--calib-json", type=str, default=None,
                        help="Path to stereo_offline_calibration_*.json. If omitted, use latest in data/.")
    parser.add_argument("--keyframes-dir", type=str, required=True,
                        help="Directory containing stereo keyframes (frame_XXX_cam*.png + frame_XXX.json).")
    parser.add_argument("--frame-index", type=int, default=None,
                        help="Optional frame index (XXX). If omitted, use last usable stereo frame.")
    parser.add_argument("--line-step", type=int, default=40,
                        help="Vertical spacing between horizontal scanlines (pixels).")

    args = parser.parse_args()

    # Calibration JSON
    calib_json_path = _find_latest_calib_json(args.calib_json)
    (W, H), K0, D0, K1, D1, R, T, board_source = _load_calib(calib_json_path)
    image_size = (W, H)
    print(f"[CALIB] image_size={image_size}")

    # Keyframe
    (frame_idx,
     frame0_bgr,
     frame1_bgr,
     corners0,
     ids0,
     corners1,
     ids1) = _load_keyframe_pair(args.keyframes_dir, args.frame_index)

    # Stereo rectification
    print("[RECT] Running stereoRectify…")
    R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
        K0, D0, K1, D1,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1,
    )

    # Rectification maps
    print("[RECT] Building undistort/rectify maps…")
    map0_x, map0_y = cv2.initUndistortRectifyMap(
        K0, D0, R0, P0, image_size, cv2.CV_32FC1
    )
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1
    )

    # Apply rectification
    rect0 = cv2.remap(frame0_bgr, map0_x, map0_y, interpolation=cv2.INTER_LINEAR)
    rect1 = cv2.remap(frame1_bgr, map1_x, map1_y, interpolation=cv2.INTER_LINEAR)

    h, w = rect0.shape[:2]
    # Draw horizontal scanlines on both rectified images
    step = max(1, int(args.line_step))
    color = (0, 255, 255)
    for y in range(step // 2, h, step):
        cv2.line(rect0, (0, y), (w - 1, y), color, 1, lineType=cv2.LINE_AA)
        cv2.line(rect1, (0, y), (w - 1, y), color, 1, lineType=cv2.LINE_AA)

    # Side-by-side canvas
    canvas = np.hstack([rect0, rect1])

    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(data_dir, f"stereo_rectify_overlay_{stamp}_frame{frame_idx:03d}.png")
    cv2.imwrite(out_png, canvas)
    print(f"[RECT] Wrote rectified overlay image -> {out_png}")


if __name__ == "__main__":
    main()


