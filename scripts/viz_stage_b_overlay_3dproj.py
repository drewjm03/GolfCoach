from __future__ import annotations

"""
Stage B visualizer: project triangulated 3D joints back into a chosen camera.

This script:
  - Loads kpts3d_tri.npy (T, J, 3) from Stage B
  - Optionally loads frames.npy (T,) and uses video_start_frame to align rows to
    absolute video frame indices
  - Projects 3D points into either cam0 or cam1 using the stereo rig JSON
  - Overlays the projected 2D joints onto the original video

This lets you visually inspect how well the triangulated skeleton reprojects
into each camera, using the same intrinsics/extrinsics and distortion model
used during Stage B reprojection error computation.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from golfcoach.io.rig_config import load_rig_config


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Project Stage B triangulated 3D joints back into a camera and overlay on the video."
    )
    ap.add_argument("--video", required=True, help="Input video path (full/original video).")
    ap.add_argument(
        "--kpts3d_npy",
        required=True,
        help="Path to kpts3d_tri.npy from Stage B (T, J, 3) in cam0 frame.",
    )
    ap.add_argument(
        "--frames_npy",
        required=True,
        help="Path to frames.npy from Stage B (T,) giving frame indices for each row.",
    )
    ap.add_argument("--rig_json", required=True, help="Stereo rig calibration JSON (same as used in Stage B).")
    ap.add_argument("--out", required=True, help="Output video path.")
    ap.add_argument(
        "--video_start_frame",
        type=int,
        default=0,
        help="Start frame index in the original video where Stage A/PHALP began (e.g. 10).",
    )
    ap.add_argument(
        "--camera_idx",
        type=int,
        choices=[0, 1],
        default=0,
        help="Camera index to project into: 0 (left / cam0) or 1 (right / cam1).",
    )
    ap.add_argument("--radius", type=int, default=3, help="Circle radius for joints.")
    ap.add_argument("--thickness", type=int, default=-1, help="Circle thickness (-1 for filled).")
    ap.add_argument(
        "--draw_indices",
        action="store_true",
        help="If set, draw the joint index next to each projected point.",
    )
    args = ap.parse_args()

    # Load 3D joints
    kpts3d = np.load(args.kpts3d_npy)  # (T, J, 3)
    if kpts3d.ndim != 3 or kpts3d.shape[2] != 3:
        raise ValueError(f"Expected kpts3d shape (T, J, 3), got {kpts3d.shape}")

    T, J, _ = kpts3d.shape

    # Load frame indices from Stage B
    frames = np.load(args.frames_npy).astype(np.int64)
    if frames.ndim != 1 or frames.shape[0] != T:
        raise ValueError(f"frames.npy must be shape (T,), got {frames.shape} vs T={T}")

    # Build absolute video frame indices
    abs_frames = args.video_start_frame + (frames - frames[0])

    # Load rig calibration
    image_size, K0, D0, K1, D1, R, T_vec = load_rig_config(args.rig_json)

    if args.camera_idx == 0:
        K = K0
        D = D0
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
    else:
        K = K1
        D = D1
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        tvec = T_vec.astype(np.float64).reshape(3, 1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    current_video_idx = 0

    for ti in range(T):
        target_idx = int(abs_frames[ti])

        # Advance the capture to the desired frame index
        while current_video_idx < target_idx:
            ret = cap.grab()
            if not ret:
                cap.release()
                writer.release()
                raise RuntimeError(
                    f"Video ended while seeking to frame {target_idx}. "
                    f"Reached {current_video_idx}."
                )
            current_video_idx += 1

        # Decode the target frame
        ret, frame = cap.read()
        if not ret:
            break
        current_video_idx += 1

        pts3d = kpts3d[ti].astype(np.float64)  # (J, 3)

        # Only project finite 3D points
        valid_mask = np.isfinite(pts3d).all(axis=1)
        if not np.any(valid_mask):
            writer.write(frame)
            continue

        obj_pts = pts3d[valid_mask].reshape(-1, 1, 3)

        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        proj_pts = proj_pts.reshape(-1, 2)

        # Scatter projected points back into full J-length array
        proj_full = np.full((J, 2), np.nan, dtype=np.float32)
        proj_full[valid_mask] = proj_pts.astype(np.float32)

        for j, (x, y) in enumerate(proj_full):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            cv2.circle(
                frame,
                (int(round(x)), int(round(y))),
                args.radius,
                (0, 0, 255),
                args.thickness,
                lineType=cv2.LINE_AA,
            )
            if args.draw_indices:
                cv2.putText(
                    frame,
                    str(j),
                    (int(round(x)) + 4, int(round(y)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote 3D reprojection overlay video to: {args.out}")


if __name__ == "__main__":
    main()








