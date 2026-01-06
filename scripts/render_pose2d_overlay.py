from __future__ import annotations

import argparse

from golfcoach.viz.pose2d_overlay import OverlayStyle, render_pose2d_overlay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pose_npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--labels", action="store_true")
    args = ap.parse_args()

    style = OverlayStyle(
        conf_thresh=args.conf,
        draw_labels=args.labels,
        draw_bbox=True,
    )

    render_pose2d_overlay(
        video_path=args.video,
        pose2d_npz_path=args.pose_npz,
        out_video_path=args.out,
        style=style,
        stride=args.stride,
        max_frames=args.max_frames,
        show_preview=args.preview,
    )

    print(f"Wrote overlay video to: {args.out}")


if __name__ == "__main__":
    main()






