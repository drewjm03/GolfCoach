from __future__ import annotations

"""
Generic 2D keypoint overlay for a video, without any triangulation.

Inputs:
  - A video file (full/original clip)
  - A NumPy array of 2D joints, shape (T, J, 2)
  - Optional frames.npy to map each row to an absolute video frame index
  - Optional video_start_frame indicating where processing started

This is similar to the Stage B visualizer but does not assume anything
about how the 2D keypoints were produced.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay generic 2D joints (T, J, 2) onto a video, with optional frame alignment."
    )
    ap.add_argument("--video", required=True, help="Input video path (full/original video).")
    ap.add_argument("--kpts2d_npy", required=True, help="Path to kpts2d.npy (T, J, 2).")
    ap.add_argument("--out", required=True, help="Output video path.")
    ap.add_argument(
        "--frames_npy",
        default=None,
        help="Optional frames.npy (T,) giving frame indices used when generating the keypoints.",
    )
    ap.add_argument(
        "--video_start_frame",
        type=int,
        default=0,
        help="Start frame index in the original video where processing began (e.g. 10).",
    )
    ap.add_argument(
        "--normalized",
        action="store_true",
        help="Treat kpts2d as normalized [0,1] coords; otherwise auto-detect pixels vs normalized.",
    )
    ap.add_argument("--radius", type=int, default=3, help="Circle radius for joints.")
    ap.add_argument("--thickness", type=int, default=-1, help="Circle thickness (-1 for filled).")
    args = ap.parse_args()

    kpts2d = np.load(args.kpts2d_npy)  # (T, J, 2)
    if kpts2d.ndim != 3 or kpts2d.shape[2] != 2:
        raise ValueError(f"Expected kpts2d shape (T, J, 2), got {kpts2d.shape}")

    T = kpts2d.shape[0]

    # Build absolute video frame indices for each row of kpts2d
    if args.frames_npy is not None:
        frames = np.load(args.frames_npy).astype(np.int64)
        if frames.ndim != 1 or frames.shape[0] != T:
            raise ValueError(f"frames.npy must be shape (T,), got {frames.shape} vs T={T}")
        abs_frames = args.video_start_frame + (frames - frames[0])
    else:
        abs_frames = args.video_start_frame + np.arange(T, dtype=np.int64)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Convert keypoints to pixel coordinates if needed
    kpts = kpts2d.astype(np.float32)
    if args.normalized:
        kpts_px = kpts.copy()
        kpts_px[..., 0] *= float(width)
        kpts_px[..., 1] *= float(height)
    else:
        max_val = float(np.nanmax(kpts))
        if max_val <= 2.0:
            # Looks like normalized [0,1]
            kpts_px = kpts.copy()
            kpts_px[..., 0] *= float(width)
            kpts_px[..., 1] *= float(height)
        else:
            # Assume already in pixels
            kpts_px = kpts

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

        pts = kpts_px[ti]  # (J, 2)
        for (x, y) in pts:
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            cv2.circle(
                frame,
                (int(round(x)), int(round(y))),
                args.radius,
                (0, 255, 0),
                args.thickness,
                lineType=cv2.LINE_AA,
            )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote 2D overlay video to: {args.out}")


if __name__ == "__main__":
    main()




