from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from golfcoach.io.video import get_video_info, iter_video_frames
from golfcoach.io.npz_io import load_npz


@dataclass
class OverlayStyle:
    radius: int = 4
    thickness: int = 2
    conf_thresh: float = 0.3
    draw_bbox: bool = True
    draw_labels: bool = False
    label_scale: float = 0.5


def _to_int_bbox(b: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b.astype(float)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _default_edges_from_names(names: List[str]) -> List[Tuple[int, int]]:
    """
    Best-effort skeleton edges based on common joint naming.
    If names are unknown (kpt_0...), returns empty list.
    """
    name_to_i = {n: i for i, n in enumerate(names)}

    def has(a: str, b: str) -> bool:
        return a in name_to_i and b in name_to_i

    edges = []
    # Try a COCO-ish / human-style skeleton if names exist
    pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("neck", "head"),
        ("thorax", "neck"),
        ("pelvis", "thorax"),
    ]
    for a, b in pairs:
        if has(a, b):
            edges.append((name_to_i[a], name_to_i[b]))

    return edges


def render_pose2d_overlay(
    video_path: str,
    pose2d_npz_path: str,
    out_video_path: str,
    style: OverlayStyle = OverlayStyle(),
    edges: Optional[List[Tuple[int, int]]] = None,
    max_frames: Optional[int] = None,
    stride: int = 1,
    show_preview: bool = False,
) -> None:
    data = load_npz(pose2d_npz_path)
    kpts = data["kpts"]          # (T,J,2)
    conf = data["conf"]          # (T,J)
    bbox = data["bbox"]          # (T,4)
    joint_names = [str(x) for x in data["joint_names"].tolist()]
    T, J, _ = kpts.shape

    if edges is None:
        edges = _default_edges_from_names(joint_names)

    info = get_video_info(video_path)

    # Video writer
    out_path = Path(out_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, info.fps / stride, (info.width, info.height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_video_path}")

    # iterate video frames in same stride; align by pose frame index list if present
    # We saved pose frame_idx in the NPZ; use it to map video frame->pose row
    pose_frame_idx = data.get("frame_idx", None)
    pose_frame_idx = pose_frame_idx.astype(int) if pose_frame_idx is not None else None
    idx_to_row: Dict[int, int] = {}
    if pose_frame_idx is not None:
        for row, fi in enumerate(pose_frame_idx.tolist()):
            idx_to_row[int(fi)] = row

    n_written = 0
    for frame_idx, t_sec, frame in iter_video_frames(video_path, stride=stride, max_frames=max_frames):
        # choose pose row
        if pose_frame_idx is None:
            row = min(n_written, T - 1)
        else:
            if frame_idx not in idx_to_row:
                # if missing (shouldn't happen), skip drawing
                row = None
            else:
                row = idx_to_row[frame_idx]

        if row is not None:
            # bbox
            if style.draw_bbox:
                x1, y1, x2, y2 = _to_int_bbox(bbox[row])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # skeleton edges
            for i, j in edges:
                if i < J and j < J and conf[row, i] >= style.conf_thresh and conf[row, j] >= style.conf_thresh:
                    p1 = tuple(np.round(kpts[row, i]).astype(int).tolist())
                    p2 = tuple(np.round(kpts[row, j]).astype(int).tolist())
                    cv2.line(frame, p1, p2, (255, 0, 0), style.thickness)

            # keypoints
            for j in range(J):
                if conf[row, j] < style.conf_thresh:
                    continue
                x, y = np.round(kpts[row, j]).astype(int)
                cv2.circle(frame, (x, y), style.radius, (0, 255, 0), -1)

                if style.draw_labels:
                    name = joint_names[j] if j < len(joint_names) else str(j)
                    cv2.putText(frame, name, (x + 4, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, style.label_scale,
                                (255, 255, 255), 1, cv2.LINE_AA)

            # overlay info text
            cv2.putText(frame, f"frame={frame_idx} t={t_sec:.3f}s",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        n_written += 1

        if show_preview:
            cv2.imshow("Pose2D Overlay", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    writer.release()
    if show_preview:
        cv2.destroyAllWindows()



