"""
Stereo pose visualization (offline, Plotly).

- Consumes the latest stereo_offline_calibration_*.json from
  stereo_cam_calibrator_offline.py
- Loads one stereo keyframe (images + detections) from a keyframes directory
- Uses the *exact same* board construction path as the stereo offline calibrator
- Produces a Plotly HTML showing both camera poses relative to the board
"""

import os
import sys
import glob
import json
import argparse
import time

import numpy as np
import cv2


# ---- local imports / fallbacks ----
try:
    from .. import config
    from ..detect import CalibrationAccumulator
    from ..stereo_calib_plot import plot_3d
    from . import stereo_cam_calibrator_offline as stereo_cal
except Exception:
    # Fallback when run as a top-level script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config  # type: ignore
    from apps.detect import CalibrationAccumulator  # type: ignore
    from apps.stereo_calib_plot import plot_3d  # type: ignore
    import apps.calibration_testing.stereo_cam_calibrator_offline as stereo_cal  # type: ignore


def _find_latest_calib_json(calib_json_arg: str | None) -> str:
    """Return path to calibration JSON (explicit or latest stereo_offline_calibration_*.json)."""
    if calib_json_arg:
        path = os.path.abspath(calib_json_arg)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[CALIB] Calibration JSON not found: {path}")
        return path

    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    data_dir = os.path.join(repo_root, "data")
    pattern = os.path.join(data_dir, "stereo_offline_calibration_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"[CALIB] No stereo_offline_calibration_*.json found under {data_dir}")
    # Files are timestamped in the name, so lexical sort is fine; also sort by mtime for robustness
    paths.sort(key=lambda p: os.path.getmtime(p))
    latest = paths[-1]
    print(f"[CALIB] Using latest calibration JSON: {latest}")
    return latest


def _load_calib(calib_json_path: str):
    with open(calib_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    W, H = data["image_size"]
    K0 = np.asarray(data["K0"], dtype=np.float64)
    D0 = np.asarray(data["D0"], dtype=np.float64)
    K1 = np.asarray(data["K1"], dtype=np.float64)
    D1 = np.asarray(data["D1"], dtype=np.float64)
    R = np.asarray(data["R"], dtype=np.float64)
    T = np.asarray(data["T"], dtype=np.float64)
    board_source = str(data.get("board_source", "harvard"))
    return (W, H), K0, D0, K1, D1, R, T, board_source


def _load_meta_from_keyframes(keyframes_dir: str) -> dict:
    meta_path = os.path.join(keyframes_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[META] Loaded {meta_path}")
        return meta
    except Exception as e:
        print(f"[META] Failed to load meta.json: {e}")
        return {}


def _resolve_board_and_accumulator(image_size, args, board_source_from_calib: str, keyframes_dir: str):
    """
    Use the *exact same* board creation path as stereo_cam_calibrator_offline.py by
    importing its module, setting the same globals, and calling its wrapper.
    """
    (W, H) = image_size

    meta = _load_meta_from_keyframes(keyframes_dir)

    # Resolve board configuration in priority:
    # 1) CLI args
    # 2) keyframes meta.json
    # 3) calibration JSON's board_source (for board_source only)
    board_source = (
        args.board_source
        or meta.get("board_source")
        or board_source_from_calib
        or "harvard"
    )

    april_pickle = args.april_pickle or meta.get("april_pickle") or None
    harvard_tag_size_m = (
        args.harvard_tag_size_m
        if args.harvard_tag_size_m is not None
        else meta.get("harvard_tag_size_m")
    )
    harvard_tag_spacing_m = (
        args.harvard_tag_spacing_m
        if args.harvard_tag_spacing_m is not None
        else meta.get("harvard_tag_spacing_m")
    )

    # Corner order override: same logic as calibrator
    corner_order_str = args.corner_order or meta.get("corner_order") or ""
    corner_order_override = None
    disable_autoreorder = False
    if corner_order_str:
        try:
            parts = [int(x.strip()) for x in corner_order_str.split(",")]
            if len(parts) == 4 and sorted(parts) == [0, 1, 2, 3]:
                corner_order_override = parts
                disable_autoreorder = True
            else:
                print(f"[WARN] Ignoring invalid corner_order '{corner_order_str}' from CLI/meta.")
        except Exception as e:
            print(f"[WARN] Failed to parse corner_order '{corner_order_str}': {e}")

    # Mirror calibrator globals *in its module* and call its board factory
    stereo_cal._APRIL_LOCAL_PICKLE = april_pickle
    stereo_cal._BOARD_SOURCE = str(board_source)
    stereo_cal._HARVARD_TAG_SIZE_M = harvard_tag_size_m
    stereo_cal._HARVARD_TAG_SPACING_M = harvard_tag_spacing_m

    try:
        board = stereo_cal._make_board_wrapper()
    except Exception as e:
        raise RuntimeError(f"[BOARD] Failed to construct board via calibrator module: {e}")

    image_size_cv = (int(W), int(H))
    acc = CalibrationAccumulator(
        board,
        image_size_cv,
        corner_order_override=corner_order_override,
        disable_corner_autoreorder=disable_autoreorder,
    )
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())
    return board, acc


def _load_keyframe_pair(keyframes_dir: str, frame_index: int | None):
    """Load one stereo keyframe (images + detections) from keyframes_dir."""
    keyframes_dir = os.path.abspath(keyframes_dir)
    if not os.path.isdir(keyframes_dir):
        raise FileNotFoundError(f"[KEYFRAMES] Directory not found: {keyframes_dir}")

    # Enumerate available JSONs
    json_paths = sorted(glob.glob(os.path.join(keyframes_dir, "frame_*.json")))
    if not json_paths:
        raise FileNotFoundError(f"[KEYFRAMES] No frame_*.json files found under {keyframes_dir}")

    def _idx_from_path(p: str) -> int | None:
        base = os.path.basename(p)
        try:
            return int(base.split("_")[1].split(".")[0])
        except Exception:
            return None

    # If user specified a frame index, try that first; else search from highest index down
    candidate_indices: list[int] = []
    if frame_index is not None:
        candidate_indices = [int(frame_index)]
    else:
        idxs = [i for i in (_idx_from_path(p) for p in json_paths) if i is not None]
        if not idxs:
            raise RuntimeError("[KEYFRAMES] Could not parse frame indices from JSON filenames.")
        candidate_indices = sorted(set(idxs), reverse=True)

    for idx in candidate_indices:
        json_path = os.path.join(keyframes_dir, f"frame_{idx:03d}.json")
        img0_path = os.path.join(keyframes_dir, f"frame_{idx:03d}_cam0.png")
        img1_path = os.path.join(keyframes_dir, f"frame_{idx:03d}_cam1.png")
        if not (os.path.isfile(json_path) and os.path.isfile(img0_path) and os.path.isfile(img1_path)):
            print(f"[KEYFRAMES] Missing files for frame {idx:03d}; skipping.")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[KEYFRAMES] Failed to read {json_path}: {e}; skipping.")
            continue

        ids0 = data.get("ids0", [])
        ids1 = data.get("ids1", [])
        corners0 = data.get("corners0", [])
        corners1 = data.get("corners1", [])

        if not ids0 or not ids1 or not corners0 or not corners1:
            print(f"[KEYFRAMES] Frame {idx:03d} has empty ids/corners; skipping.")
            continue

        # Convert to numpy forms expected by plot_3d
        ids0_arr = np.asarray(ids0, dtype=np.int32).reshape(-1, 1)
        ids1_arr = np.asarray(ids1, dtype=np.int32).reshape(-1, 1)

        corners0_list = []
        for c in corners0:
            if len(c) != 4:
                continue
            corners0_list.append(np.asarray(c, dtype=np.float32).reshape(1, 4, 2))

        corners1_list = []
        for c in corners1:
            if len(c) != 4:
                continue
            corners1_list.append(np.asarray(c, dtype=np.float32).reshape(1, 4, 2))

        if not corners0_list or not corners1_list:
            print(f"[KEYFRAMES] Frame {idx:03d} had invalid corner shapes; skipping.")
            continue

        frame0_bgr = cv2.imread(img0_path, cv2.IMREAD_COLOR)
        frame1_bgr = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        if frame0_bgr is None or frame1_bgr is None:
            print(f"[KEYFRAMES] Failed to load images for frame {idx:03d}; skipping.")
            continue

        print(f"[KEYFRAMES] Using frame {idx:03d} for pose plot.")
        return idx, frame0_bgr, frame1_bgr, corners0_list, ids0_arr, corners1_list, ids1_arr

    raise RuntimeError("[KEYFRAMES] Could not find any usable stereo keyframe with ids/corners on both cameras.")


def main():
    parser = argparse.ArgumentParser(description="Stereo pose Plotly visualization from offline calibration/keyframes.")
    parser.add_argument("--calib-json", type=str, default=None,
                        help="Path to stereo_offline_calibration_*.json. If omitted, use latest in data/.")
    parser.add_argument("--keyframes-dir", type=str, required=True,
                        help="Directory containing stereo keyframes (frame_XXX_cam*.png + frame_XXX.json).")
    parser.add_argument("--frame-index", type=int, default=None,
                        help="Optional frame index (XXX). If omitted, use last usable stereo frame.")

    # Board-related args (mirror calibrator; all optional, fall back to keyframes meta + calib JSON)
    parser.add_argument("--board-source", type=str, choices=["harvard", "grid8x5"], default=None,
                        help="Board source override; if omitted, use keyframes meta or calib JSON board_source.")
    parser.add_argument("--april-pickle", type=str, default=None,
                        help="Path to local AprilBoards.pickle (overrides meta).")
    parser.add_argument("--harvard-tag-size-m", type=float, default=None,
                        help="Tag side length in meters for Harvard board (overrides meta).")
    parser.add_argument("--harvard-tag-spacing-m", type=float, default=None,
                        help="Tag spacing (meters) for Harvard board (overrides meta).")
    parser.add_argument("--corner-order", type=str, default=None,
                        help="Manual corner order override as four comma-separated indices, e.g. '3,0,1,2'.")

    args = parser.parse_args()

    # Calibration JSON
    calib_json_path = _find_latest_calib_json(args.calib_json)
    (W, H), K0, D0, K1, D1, R, T, board_source_from_calib = _load_calib(calib_json_path)
    image_size = (W, H)
    print(f"[CALIB] image_size={image_size}")

    # Board + accumulator using *exact* calibrator board creation path
    board, acc = _resolve_board_and_accumulator(image_size, args, board_source_from_calib, args.keyframes_dir)

    # Keyframe
    (frame_idx,
     frame0_bgr,
     frame1_bgr,
     corners0,
     ids0,
     corners1,
     ids1) = _load_keyframe_pair(args.keyframes_dir, args.frame_index)

    # Output HTML path
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_html = os.path.join(data_dir, f"stereo_pose_plot_{stamp}_frame{frame_idx:03d}.html")

    # Plot 3D board + camera poses in board frame, using known stereo extrinsics
    print(f"[PLOT] Writing Plotly HTML to {out_html}")
    plot_3d(
        board,
        K0, D0,
        K1, D1,
        corners0, ids0,
        corners1, ids1,
        acc,
        frame0_bgr, frame1_bgr,
        out_html,
        R_01=R,
        T_01=T,
    )
    print("[PLOT] Done.")


if __name__ == "__main__":
    main()


