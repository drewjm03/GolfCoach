from __future__ import annotations

import argparse
from pathlib import Path

from golfcoach.pose2d.stage_a_pose2d import run_pose2d_on_video
from golfcoach.pose2d.providers.golfpose_provider import DetectorConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="Left video path")
    ap.add_argument("--right", required=True, help="Right video path")
    ap.add_argument("--rig_json", required=True, help="Rig calibration JSON (stored for later)")

    ap.add_argument("--pose_config", required=True, help="GolfPose mmpose config (.py)")
    ap.add_argument("--pose_ckpt", required=True, help="GolfPose 2D checkpoint (.pth)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")

    ap.add_argument("--out_dir", required=True, help="Output run folder (e.g. runs/test1)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)

    # Optional detector (recommended for top-down stability)
    ap.add_argument("--det_config", default=None)
    ap.add_argument("--det_ckpt", default=None)
    ap.add_argument("--det_device", default="cpu")
    ap.add_argument("--det_score_thr", type=float, default=0.3)
    ap.add_argument("--det_class_id", type=int, default=0)
    ap.add_argument("--det_ema", type=float, default=0.8)
    ap.add_argument("--det_pad_scale", type=float, default=1.25)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector_cfg = None
    if args.det_config and args.det_ckpt:
        detector_cfg = DetectorConfig(
            config=args.det_config,
            checkpoint=args.det_ckpt,
            device=args.det_device,
            score_thr=args.det_score_thr,
            class_id=args.det_class_id,
            ema=args.det_ema,
            pad_scale=args.det_pad_scale,
        )

    left_out = out_dir / "pose2d_left.npz"
    right_out = out_dir / "pose2d_right.npz"

    run_pose2d_on_video(
        video_path=args.left,
        out_npz_path=str(left_out),
        pose_config=args.pose_config,
        pose_ckpt=args.pose_ckpt,
        device=args.device,
        stride=args.stride,
        max_frames=args.max_frames,
        detector_cfg=detector_cfg,
    )

    run_pose2d_on_video(
        video_path=args.right,
        out_npz_path=str(right_out),
        pose_config=args.pose_config,
        pose_ckpt=args.pose_ckpt,
        device=args.device,
        stride=args.stride,
        max_frames=args.max_frames,
        detector_cfg=detector_cfg,
    )

    print(f"Wrote:\n  {left_out}\n  {right_out}")


if __name__ == "__main__":
    main()





