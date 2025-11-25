import os
import sys
import time
import argparse
import queue

import cv2
import numpy as np

try:
    from . import config
    from .capture import CamReader
    from .detect import CalibrationAccumulator
    from .stereo_calib_plot import load_board
    from .ui import draw_ids
except Exception:
    # Fallback when executed as a script
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from apps import config  # type: ignore
    from apps.capture import CamReader  # type: ignore
    from apps.detect import CalibrationAccumulator  # type: ignore
    from apps.stereo_calib_plot import load_board  # type: ignore
    from apps.ui import draw_ids  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Live AprilTag viewer (no calibration, just detections).")
    parser.add_argument("--board-source", type=str, choices=["harvard", "grid8x5"], default="harvard",
                        help="Board type to gate IDs against (harvard or 8x5 grid).")
    parser.add_argument("--april-pickle", type=str, default=None,
                        help="Path to Harvard AprilBoards.pickle (for harvard board).")
    parser.add_argument("--harvard-tag-size-m", type=float, default=None,
                        help="Tag side length in meters for Harvard board (optional).")
    parser.add_argument("--harvard-tag-spacing-m", type=float, default=None,
                        help="Tag spacing in meters for Harvard board (optional; only matters if using centers).")
    parser.add_argument("--corner-order", type=str, default=None,
                        help="Corner order override 'i0,i1,i2,i3'. If given, disables auto corner reordering.")
    parser.add_argument("--cam0", type=int, default=0, help="Camera index for first camera.")
    parser.add_argument("--cam1", type=int, default=None,
                        help="Optional second camera index. If omitted, only cam0 is used.")

    args, _ = parser.parse_known_args()

    # Open cameras
    cam_indices = [int(args.cam0)]
    if args.cam1 is not None:
        cam_indices.append(int(args.cam1))

    print(f"[VIEW] Opening cameras: {cam_indices}")
    cams = []
    for idx in cam_indices:
        try:
            cams.append(CamReader(idx))
        except Exception as e:
            print(f"[VIEW][ERR] Failed to open camera index {idx}: {e}")
    if not cams:
        print("[VIEW][ERR] No cameras opened; exiting.")
        return

    # Peek first frame for size
    try:
        ts0, f0 = cams[0].latest()
    except queue.Empty:
        print("[VIEW][ERR] Failed to grab first frame from cam0; exiting.")
        for c in cams:
            c.release()
        return
    H, W = f0.shape[:2]
    image_size = (W, H)
    print(f"[VIEW] Image size: {W}x{H}")

    # Build board / accumulator so detection uses the same gating + corner-order logic as calibration tools
    board = load_board(
        board_source=args.board_source,
        april_pickle=args.april_pickle,
        harvard_tag_size_m=args.harvard_tag_size_m,
        harvard_tag_spacing_m=args.harvard_tag_spacing_m,
    )

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
                print(f"[APRIL][WARN] Ignoring invalid --corner-order '{args.corner_order}'")
        except Exception as e:
            print(f"[APRIL][WARN] Failed to parse --corner-order: {e}")
    else:
        # Mirror other tools: for Harvard, default to 3,0,1,2 and disable autoreorder
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

    win = "Tag Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = []
            grays = []
            for c in cams:
                try:
                    ts, f = c.latest()
                except queue.Empty:
                    continue
                frames.append(f)
                grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))

            if not frames:
                time.sleep(0.01)
                continue

            annotated = []
            tag_counts = []
            tag_ids_strs = []

            for idx, (frame, gray) in enumerate(zip(frames, grays)):
                try:
                    corners, ids = acc.detect(gray)
                except Exception:
                    corners, ids = [], None

                vis = frame.copy()
                n_tags = 0 if ids is None else len(ids)
                tag_counts.append(n_tags)
                if ids is not None:
                    tag_ids_strs.append(", ".join(str(int(iv[0])) for iv in ids))
                else:
                    tag_ids_strs.append("")

                try:
                    if corners is not None and ids is not None and len(corners) > 0:
                        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                        draw_ids(vis, corners, ids, color=(0, 255, 255))
                except Exception:
                    pass

                label = f"cam{cam_indices[idx]} tags={n_tags}"
                cv2.putText(vis, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                if tag_ids_strs[-1]:
                    cv2.putText(vis, f"ids: {tag_ids_strs[-1]}", (16, 64),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                annotated.append(vis)

            # Compose output canvas (stack horizontally or vertically depending on number of cams)
            if len(annotated) == 1:
                canvas = annotated[0]
            else:
                # Match heights
                h_max = max(img.shape[0] for img in annotated)
                resized = []
                for img in annotated:
                    h, w = img.shape[:2]
                    if h != h_max:
                        img = cv2.resize(img, (int(round(w * h_max / float(h))), h_max))
                    resized.append(img)
                canvas = cv2.hconcat(resized)

            cv2.imshow(win, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                print("[VIEW] Quit.")
                break
    finally:
        for c in cams:
            c.release()
        cv2.destroyAllWindows()
        print("[VIEW] Closed.")


if __name__ == "__main__":
    main()


