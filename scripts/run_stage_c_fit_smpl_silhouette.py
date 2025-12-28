from __future__ import annotations

import argparse

from golfcoach.pose3d.stage_c_fit_smpl_silhouette import (
    fit_smpl_silhouette_stereo,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage C: fit SMPL(-X) in cam0 using stereo silhouettes from PHALP / 4DHumans."
    )
    ap.add_argument(
        "--pkl_left",
        required=True,
        help="PHALP/4DHumans pickle for the left camera.",
    )
    ap.add_argument(
        "--pkl_right",
        required=True,
        help="PHALP/4DHumans pickle for the right camera.",
    )
    ap.add_argument(
        "--rig_json",
        required=True,
        help="Stereo rig calibration JSON (same one used in Stages A/B).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for Stage C results (e.g. runs/my_swing_stageC).",
    )
    ap.add_argument(
        "--person_i_left",
        type=int,
        default=0,
        help="Person index in left PHALP tracks (default: 0).",
    )
    ap.add_argument(
        "--person_i_right",
        type=int,
        default=0,
        help="Person index in right PHALP tracks (default: 0).",
    )
    ap.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional limit on number of common frames to fit (default: all).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Torch device to use, e.g. "cuda" or "cpu" (default: "cuda").',
    )

    args = ap.parse_args()

    fit_smpl_silhouette_stereo(
        pkl_left=args.pkl_left,
        pkl_right=args.pkl_right,
        rig_json=args.rig_json,
        out_dir=args.out_dir,
        person_i_left=args.person_i_left,
        person_i_right=args.person_i_right,
        max_frames=args.max_frames,
        device=args.device,
    )

    print(f"Stage C silhouette-based SMPL fitting complete. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()




