from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


def get_video_info(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap.release()
    return VideoInfo(video_path, fps, frame_count, width, height)


def iter_video_frames(
    video_path: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """
    Yields (frame_idx, t_sec, frame_bgr).
    t_sec is computed from fps and frame index (sufficient for Stage A).
    """
    info = get_video_info(video_path)
    if info.fps <= 0:
        raise RuntimeError(f"Could not read FPS from video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    out_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride == 0:
            t_sec = frame_idx / info.fps
            yield frame_idx, t_sec, frame
            out_count += 1
            if max_frames is not None and out_count >= max_frames:
                break

        frame_idx += 1

    cap.release()






