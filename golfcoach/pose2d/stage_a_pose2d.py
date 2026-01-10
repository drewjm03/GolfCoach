from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from golfcoach.io.video import get_video_info, iter_video_frames
from golfcoach.io.npz_io import save_npz_compressed
from golfcoach.pose2d.datatypes import Pose2DSequence
from golfcoach.pose2d.providers.golfpose_provider import DetectorConfig, GolfPose2DProvider


def run_pose2d_on_video(
    video_path: str,
    out_npz_path: str,
    pose_config: str,
    pose_ckpt: str,
    device: str = "cpu",
    stride: int = 1,
    max_frames: Optional[int] = None,
	detector_cfg: Optional[DetectorConfig] = None,
	force_center_bbox: bool = False,
	force_bbox_frac: float = 2.0 / 3.0,
) -> Pose2DSequence:
    info = get_video_info(video_path)
    provider = GolfPose2DProvider(
        pose_config=pose_config,
        pose_checkpoint=pose_ckpt,
        device=device,
		detector=detector_cfg,
		force_center_bbox=bool(force_center_bbox),
		force_bbox_frac=float(force_bbox_frac),
    )

	if force_center_bbox:
		print(f"[POSE2D] Using forced center bbox: width_frac={float(force_bbox_frac):.4f} (central ~{float(force_bbox_frac)*100:.0f}%), full height; detector disabled")

    T_est = (info.frame_count + stride - 1) // stride if info.frame_count > 0 else 0

    t_list = []
    idx_list = []
    kpts_list = []
    conf_list = []
    bbox_list = []

    for frame_idx, t_sec, frame_bgr in iter_video_frames(video_path, stride=stride, max_frames=max_frames):
        kpts, conf, bbox = provider.infer_frame(frame_bgr, bbox=None)
        t_list.append(t_sec)
        idx_list.append(frame_idx)
        kpts_list.append(kpts)
        conf_list.append(conf)
        bbox_list.append(bbox)
        # Lightweight progress print every 20 processed frames.
        if len(idx_list) % 20 == 0:
            total = info.frame_count if info.frame_count > 0 else "?"
            print(
                f"[StageA] {Path(video_path).name}: "
                f"processed {len(idx_list)} frames "
                f"(last source frame_idx={frame_idx}, total={total})"
            )

    t = np.array(t_list, dtype=np.float32)
    frame_idx_arr = np.array(idx_list, dtype=np.int32)
    kpts_arr = np.stack(kpts_list, axis=0).astype(np.float32)     # (T,22,2)
    conf_arr = np.stack(conf_list, axis=0).astype(np.float32)     # (T,22)
    bbox_arr = np.stack(bbox_list, axis=0).astype(np.float32)     # (T,4)

    seq = Pose2DSequence(
        t=t,
        frame_idx=frame_idx_arr,
        kpts=kpts_arr,
        conf=conf_arr,
        bbox=bbox_arr,
        image_size=(info.width, info.height),
        joint_names=provider.joint_names,
    )

    save_npz_compressed(
        out_npz_path,
        t=seq.t,
        frame_idx=seq.frame_idx,
        kpts=seq.kpts,
        conf=seq.conf,
        bbox=seq.bbox,
        image_size=np.array(seq.image_size, dtype=np.int32),
        joint_names=np.array(seq.joint_names, dtype=object),
        video_path=np.array([video_path], dtype=object),
        fps=np.array([info.fps], dtype=np.float32),
    )

    return seq






