"""
Live stereo keyframe recorder.

- Uses the same detection + gating pipeline as `stereo_calib_plot` to decide when a
  stereo sample is good enough to keep.
- Saves keyframes in the *same format* as `stereo_calib_plot`:
  - frame_XXX_cam0.png, frame_XXX_cam1.png
  - frame_XXX.json with size, ids0/1, corners0/1
  - meta.json describing board/recording config
- Displays live annotated frames while collecting, similar to single_cam_calibrator7.
"""

import os
import sys
import time
import json
import argparse
import queue

import numpy as np
import cv2

# ---- local imports ----
try:
    from .. import config
    from ..capture import CamReader
    from ..detect import CalibrationAccumulator
    from ..stereo_calib_plot import load_board
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config  # type: ignore
    from apps.capture import CamReader  # type: ignore
    from apps.detect import CalibrationAccumulator  # type: ignore
    from apps.stereo_calib_plot import load_board  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Live stereo keyframe recorder (grid/Harvard).")
    parser.add_argument("--board-source", type=str, choices=["harvard", "grid8x5"], default="harvard")
    parser.add_argument("--april-pickle", type=str, default=None,
                        help="Path to local AprilBoards.pickle (for Harvard board).")
    parser.add_argument("--harvard-tag-size-m", type=float, default=None,
                        help="Tag side length in meters for Harvard board.")
    parser.add_argument("--harvard-tag-spacing-m", type=float, default=None,
                        help="Tag spacing (meters) for Harvard board (informational).")
    parser.add_argument("--target-keyframes", type=int, default=50,
                        help="Number of stereo keyframes to record before exiting.")
    parser.add_argument("--accept-period", type=float, default=0.5,
                        help="Minimum seconds between accepted keyframes.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Optional output base directory (default: repo_root/data).")
    parser.add_argument("--cam0", type=int, default=0, help="Camera index for left/first camera.")
    parser.add_argument("--cam1", type=int, default=1, help="Camera index for right/second camera.")
    parser.add_argument("--corner-order", type=str, default=None,
                        help="Manual corner order override as four comma-separated indices, e.g. '0,1,2,3'. "
                             "If provided, auto corner reordering is disabled.")
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help=(
            "Accept/save keyframes when each camera independently has good detections "
            "(min tags per view), without requiring any common tag IDs between cameras."
        ),
    )

    args, _ = parser.parse_known_args()

    # Open two cameras
    print(f"[MAIN] Opening two cameras… cam0={args.cam0} cam1={args.cam1}")
    cams = [CamReader(int(args.cam0)), CamReader(int(args.cam1))]
    ts0, f0 = cams[0].latest()
    _, f1 = cams[1].latest()
    H, W = f0.shape[:2]
    image_size = (W, H)

    # Build board and accumulator (same as stereo_calib_plot)
    board = load_board(board_source=args.board_source,
                       april_pickle=args.april_pickle,
                       harvard_tag_size_m=args.harvard_tag_size_m,
                       harvard_tag_spacing_m=args.harvard_tag_spacing_m)

    corner_order_override = None
    disable_autoreorder = False
    if args.corner_order:
        try:
            parts = [int(x.strip()) for x in args.corner_order.split(",")]
            if len(parts) == 4 and sorted(parts) == [0, 1, 2, 3]:
                corner_order_override = parts
                disable_autoreorder = True
                print(f"[APRIL] Using manual corner order: {corner_order_override}")
            else:
                print(f"[WARN] Ignoring invalid --corner-order '{args.corner_order}'. Expected 4 comma-separated indices 0..3.")
        except Exception as e:
            print(f"[WARN] Failed to parse --corner-order '{args.corner_order}': {e}. Ignoring.")
    else:
        # For Harvard, mirror mono default
        if str(args.board_source).lower().strip() == "harvard":
            corner_order_override = [3, 0, 1, 2]
            disable_autoreorder = True
            print("[APRIL] Using default per-tag corner order 3,0,1,2 for Harvard board")

    acc = CalibrationAccumulator(
        board,
        image_size,
        corner_order_override=corner_order_override,
        disable_corner_autoreorder=disable_autoreorder,
    )
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())

    # Recording folder and meta (same format as stereo_calib_plot)
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    base_out = os.path.join(repo_root, "data") if not args.out_dir else os.path.abspath(args.out_dir)
    os.makedirs(base_out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    record_dir = os.path.join(base_out, f"stereo_keyframes_{stamp}")
    os.makedirs(record_dir, exist_ok=True)

    meta = {
        "image_size": [int(W), int(H)],
        "APRIL_DICT": int(getattr(config, "APRIL_DICT", 0)),
        "board_source": str(args.board_source),
        "april_pickle": args.april_pickle or "",
        "harvard_tag_size_m": float(args.harvard_tag_size_m) if args.harvard_tag_size_m is not None else None,
        "harvard_tag_spacing_m": float(args.harvard_tag_spacing_m) if args.harvard_tag_spacing_m is not None else None,
        "corner_order": args.corner_order or "",
        "target_keyframes": int(args.target_keyframes),
        "accept_period": float(args.accept_period),
        "no_overlap": bool(getattr(args, "no_overlap", False)),
    }
    with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[REC] Recording stereo keyframes to {record_dir}")

    # UI
    win = "Stereo Keyframe Recorder"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    keyframes = 0
    last_accept = 0.0

    try:
        while keyframes < args.target_keyframes:
            try:
                ts0, f0 = cams[0].latest()
                ts1, f1 = cams[1].latest()
            except queue.Empty:
                time.sleep(0.01)
                continue

            g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

            # Detect tags on both cameras
            try:
                corners0, ids0 = acc.detect(g0)
            except Exception:
                corners0, ids0 = [], None
            try:
                corners1, ids1 = acc.detect(g1)
            except Exception:
                corners1, ids1 = [], None

            n0 = 0 if ids0 is None else len(ids0)
            n1 = 0 if ids1 is None else len(ids1)

            # Build common ID set for info / JSON
            common_ids = []
            if ids0 is not None and ids1 is not None:
                set0 = {int(iv[0]) for iv in ids0}
                set1 = {int(iv[0]) for iv in ids1}
                common_ids = sorted(set0 & set1)

            now = time.perf_counter()
            added = False
            if (now - last_accept) >= float(args.accept_period):
                # Require enough tags on each cam; overlap requirement is controlled by --no-overlap
                ok0 = (corners0 is not None and ids0 is not None and len(corners0) >= config.MIN_MARKERS_PER_VIEW)
                ok1 = (corners1 is not None and ids1 is not None and len(corners1) >= config.MIN_MARKERS_PER_VIEW)
                if ok0 and ok1:
                    # Save per-cam views into accumulator (for future calibrations if desired)
                    acc._accumulate_single(0, corners0, ids0)
                    acc._accumulate_single(1, corners1, ids1)

                    if getattr(args, "no_overlap", False):
                        # In no-overlap mode, accept this keyframe as long as each
                        # camera independently has enough tags; do not require any
                        # common tag IDs at this stage.
                        added = True
                    else:
                        # Default behavior: require at least MIN_MARKERS_PER_VIEW common
                        # tags for a stereo sample, using the same sample building logic
                        # as the original implementation.
                        from apps.calib import StereoSample  # safe import; small helper
                        # Build maps and sample (same as _match_stereo)
                        map0 = {int(i[0]): c.reshape(-1, 2) for c, i in zip(corners0, ids0)}
                        map1 = {int(i[0]): c.reshape(-1, 2) for c, i in zip(corners1, ids1)}
                        common = sorted(set(map0.keys()) & set(map1.keys()))
                        obj_pts = []
                        img0 = []
                        img1 = []
                        for tag_id in common:
                            if tag_id not in acc.id_to_obj:
                                continue
                            obj = acc.id_to_obj[tag_id]
                            obj_pts.append(obj)
                            img0.append(map0[tag_id])
                            img1.append(map1[tag_id])
                        if obj_pts:
                            obj_pts = np.concatenate(obj_pts, axis=0).astype(np.float32)
                            img0 = np.concatenate(img0, axis=0).astype(np.float32)
                            img1 = np.concatenate(img1, axis=0).astype(np.float32)
                            sample = StereoSample(obj_pts, img0, img1)
                            if sample.obj_pts.shape[0] >= config.MIN_MARKERS_PER_VIEW * 4:
                                acc.stereo_samples.append(sample)
                                added = True

                if added:
                    last_accept = now
                    # Save keyframe artifacts in same format as stereo_calib_plot
                    idx = keyframes
                    out_img0 = os.path.join(record_dir, f"frame_{idx:03d}_cam0.png")
                    out_img1 = os.path.join(record_dir, f"frame_{idx:03d}_cam1.png")
                    cv2.imwrite(out_img0, f0)
                    cv2.imwrite(out_img1, f1)

                    out_js = os.path.join(record_dir, f"frame_{idx:03d}.json")
                    ids0_list = [] if ids0 is None else [int(iv[0]) for iv in ids0]
                    ids1_list = [] if ids1 is None else [int(iv[0]) for iv in ids1]
                    corners0_list = [] if not corners0 else [c.reshape(4, 2).tolist() for c in corners0]
                    corners1_list = [] if not corners1 else [c.reshape(4, 2).tolist() for c in corners1]
                    with open(out_js, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "size": [int(W), int(H)],
                                "ids0": ids0_list,
                                "ids1": ids1_list,
                                "corners0": corners0_list,
                                "corners1": corners1_list,
                                # Optional diagnostics / bookkeeping
                                "tags0_count": int(n0),
                                "tags1_count": int(n1),
                                "common_ids_count": int(len(common_ids)),
                                "ts0": float(ts0),
                                "ts1": float(ts1),
                            },
                            f,
                            indent=2,
                        )
                    keyframes += 1
                    print(f"[REC] Keyframes: {keyframes}  (tags0={n0} tags1={n1} common={len(common_ids)})")

            # Annotate and display
            vis0 = f0.copy()
            vis1 = f1.copy()
            try:
                if corners0 is not None and ids0 is not None and len(corners0) > 0:
                    cv2.aruco.drawDetectedMarkers(vis0, corners0, ids0)
                if corners1 is not None and ids1 is not None and len(corners1) > 0:
                    cv2.aruco.drawDetectedMarkers(vis1, corners1, ids1)
            except Exception:
                pass

            canvas = np.hstack([vis0, vis1])
            status_line = f"KF={keyframes}/{args.target_keyframes} tags0={n0} tags1={n1} common={len(common_ids)}"
            cv2.putText(
                canvas,
                status_line,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.imshow(win, canvas)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                print("[KEY] Quit.")
                break

    finally:
        for cam in cams:
            cam.release()
        cv2.destroyAllWindows()
        print("[MAIN] Closed.")


if __name__ == "__main__":
    main()


