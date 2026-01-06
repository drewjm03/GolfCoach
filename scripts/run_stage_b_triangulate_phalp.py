from __future__ import annotations

import argparse

from golfcoach.pose3d.stage_b_triangulate_phalp import triangulate_phalp_pkls


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B: triangulate PHALP 2D joints from stereo cameras.")
    ap.add_argument("--pkl_left", required=True, help="PHALP/4DHumans pickle for left camera.")
    ap.add_argument("--pkl_right", required=True, help="PHALP/4DHumans pickle for right camera.")
    ap.add_argument("--rig_json", required=True, help="Stereo rig calibration JSON.")
    ap.add_argument("--out_dir", required=True, help="Output directory for numpy arrays.")
    ap.add_argument("--person_i_left", type=int, default=0, help="Person index in left PHALP tracks (default: 0).")
    ap.add_argument("--person_i_right", type=int, default=0, help="Person index in right PHALP tracks (default: 0).")
    ap.add_argument(
        "--reproj_thresh_px",
        type=float,
        default=25.0,
        help="Reprojection error threshold (pixels) for invalidating 3D joints.",
    )

    args = ap.parse_args()

    triangulate_phalp_pkls(
        pkl_left=args.pkl_left,
        pkl_right=args.pkl_right,
        rig_json=args.rig_json,
        out_dir=args.out_dir,
        person_i_left=args.person_i_left,
        person_i_right=args.person_i_right,
        reproj_thresh_px=args.reproj_thresh_px,
    )

    print(f"Stage B triangulation complete. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()






