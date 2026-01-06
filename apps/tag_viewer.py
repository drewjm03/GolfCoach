import os
import sys
import time
import argparse
import queue
import json
import threading

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

    # Local per-camera FPS history, measured at this viewer loop
    from collections import deque
    t_hist = [deque(maxlen=60) for _ in cams]

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

    # Directory for saving recorded video streams
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames_dir = os.path.join(project_root, "data", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    print("[VIEW] frames_dir:", frames_dir)
    recording = False
    start_gate = None
    stop_gate = None

    win = "Tag Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Display throttling: at high FPS, only draw every Nth frame to reduce UI cost.
    display_stride = 1
    try:
        target_fps = float(getattr(config, "CAPTURE_FPS", 30))
        if target_fps >= 60:
            display_stride = 4
            print(f"[VIEW] Display stride: {display_stride}")
    except Exception:
        display_stride = 1
    frame_counter = 0
    # Precompute camera-index -> array-index mapping to avoid repeated index lookups.
    # This index (0,1,...) is also used as the *logical* camera ID (cam0, cam1)
    # so that cam0/cam1 are defined purely by CLI order, not OS enumeration.
    idx_of = {cam_idx: i for i, cam_idx in enumerate(cam_indices)}

    try:
        while True:
            # Latest frames per camera index
            frame_by_cam: dict[int, np.ndarray] = {}
            for i, (cam_idx, cam) in enumerate(zip(cam_indices, cams)):
                try:
                    ts, f = cam.latest()
                except queue.Empty:
                    continue
                # Track local dequeue timestamps per camera for measured FPS
                try:
                    t_hist[i].append(time.perf_counter())
                except Exception:
                    pass
                frame_by_cam[cam_idx] = f

            if not frame_by_cam:
                time.sleep(0.01)
                continue

            # Throttle display to every Nth frame to reduce resize/hconcat/imshow overhead
            frame_counter += 1
            do_display = (frame_counter % display_stride == 0)
            if not do_display:
                continue

            annotated = []
            tag_counts = []
            tag_ids_strs = []

            for cam_idx in cam_indices:
                frame = frame_by_cam.get(cam_idx)
                if frame is None:
                    continue
                # Temporarily disable AprilTag detection; just display raw frames.
                # When re-enabling detection, compute grayscale on-demand:
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # corners, ids = acc.detect(gray)
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

                # Logical camera ID (0,1,...) based on CLI order, not OS index.
                cam_logical = idx_of.get(cam_idx, 0)

                # Show live capture FPS from CamReader plus local measured FPS
                fps_cam = 0.0
                try:
                    cam = cams[cam_logical]
                    fps_cam = float(getattr(cam, "fps", 0.0))
                except Exception:
                    fps_cam = 0.0

                fps_meas = 0.0
                try:
                    th = t_hist[cam_logical]
                    if len(th) >= 2 and (th[-1] - th[0]) > 0:
                        fps_meas = (len(th) - 1) / (th[-1] - th[0])
                except Exception:
                    fps_meas = 0.0

                # cam{cam_logical} guarantees that cam0 is always the first
                # CLI camera (typically the left view), independent of OS index.
                label = f"cam{cam_logical} {fps_cam:4.1f}/{fps_meas:4.1f} fps tags={n_tags}"
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

            if k == ord("r"):
                # Toggle synchronized recording of all cameras using shared
                # start/stop Events so both streams start and stop together.
                if not recording:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    fps = getattr(config, "CAPTURE_FPS", 30)
                    start_gate = threading.Event()
                    stop_gate = threading.Event()
                    any_armed = False
                    for logical_id, (cam_idx, cam) in enumerate(zip(cam_indices, cams)):
                        # Use logical_id (0,1,...) for filenames so that
                        # record_*_cam0.mp4 always corresponds to --cam0,
                        # record_*_cam1.mp4 to --cam1, regardless of OS index.
                        fname = f"record_{timestamp}_cam{logical_id}.mp4"
                        out_path = os.path.join(frames_dir, fname)
                        ok = False
                        try:
                            # Provide the shared start gate via the API, and
                            # attach the shared stop gate on the CamReader so
                            # its enqueue/write loops can halt in sync.
                            cam.record_stop_gate = stop_gate
                            ok = cam.arm_recording_mp4(out_path, fps, start_gate, fourcc="mp4v")
                        except Exception as e:
                            print(f"[VIEW][WARN] Failed to arm recording for cam{cam_idx}: {e}")
                        if ok:
                            any_armed = True
                            print(f"[VIEW] Armed recording cam{cam_idx} -> {out_path}")
                    if any_armed:
                        t0 = time.perf_counter()
                        start_gate.set()
                        recording = True
                        print(f"[VIEW] Recording GO at t={t0:.6f}")
                    else:
                        print("[VIEW][WARN] No camera recordings armed; recording disabled.")
                else:
                    # Trip the shared stop gate first so that all cameras halt
                    # at (approximately) the same instant, then let each
                    # CamReader flush and write stats/JSON.
                    if stop_gate is not None:
                        stop_gate.set()
                    # Stop per-camera recording; CamReader handles stats + JSON
                    for cam_idx, cam in zip(cam_indices, cams):
                        try:
                            ret = cam.stop_recording()
                            print(f"[VIEW] Stopped recording cam{cam_idx}, stop_recording() returned: {ret}")
                        except Exception as e:
                            print(f"[VIEW][ERR] stop_recording failed for cam{cam_idx}: {type(e).__name__}: {e}")
                    recording = False

            if k in (27, ord("q")):
                print("[VIEW] Quit.")
                break
    finally:
        # If we quit while recording, trip the shared stop gate (if any) so
        # writers halt in sync, then flush recordings + timestamp JSON.
        try:
            if stop_gate is not None:
                stop_gate.set()
        except Exception:
            pass
        for cam_idx, cam in zip(cam_indices, cams):
            try:
                ret = cam.stop_recording()
                print(f"[VIEW] Stopped recording cam{cam_idx} (shutdown), stop_recording() returned: {ret}")
            except Exception as e:
                print(f"[VIEW][ERR] stop_recording failed for cam{cam_idx} (shutdown): {type(e).__name__}: {e}")

        # Clean up cameras and windows
        for c in cams:
            c.release()
        cv2.destroyAllWindows()
        print("[VIEW] Closed.")


if __name__ == "__main__":
    main()


