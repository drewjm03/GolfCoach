"""
Sweep tag corner orders for mono calibration on saved stereo keyframes (offline).

Given a stereo keyframe folder (frame_XXX_cam0.png / frame_XXX_cam1.png),
this script:

- Re-detects tags from images (does NOT use the saved JSON detections).
- For each of the 24 permutations of tag corner ordering [0,1,2,3], it:
  - Runs mono calibration for cam0 and cam1 separately using the same
    board, detector, gating, and pinhole+rational model as the stereo
    offline calibrator.
  - Prints the resulting RMS for each camera and corner order.

This is useful to diagnose corner-order issues on new boards (e.g. 8x5 grid).
"""

import os
import sys
import glob
import json
import argparse
import itertools
import time

import numpy as np
import cv2

try:
    from .. import config
    from ..detect import CalibrationAccumulator
    from . import stereo_cam_calibrator_offline as stereo_cal
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config  # type: ignore
    from apps.detect import CalibrationAccumulator  # type: ignore
    import apps.calibration_testing.stereo_cam_calibrator_offline as stereo_cal  # type: ignore


def _load_meta(keyframes_dir: str) -> dict:
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


def _resolve_board(image_size, args, keyframes_dir: str):
    """Use the same board creation path as stereo_cam_calibrator_offline.py."""
    (W, H) = image_size
    meta = _load_meta(keyframes_dir)

    board_source = (
        args.board_source
        or meta.get("board_source")
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

    # Mirror calibrator globals and call its board factory
    stereo_cal._APRIL_LOCAL_PICKLE = april_pickle
    stereo_cal._BOARD_SOURCE = str(board_source)
    stereo_cal._HARVARD_TAG_SIZE_M = harvard_tag_size_m
    stereo_cal._HARVARD_TAG_SPACING_M = harvard_tag_spacing_m

    try:
        board = stereo_cal._make_board_wrapper()
    except Exception as e:
        raise RuntimeError(f"[BOARD] Failed to construct board via calibrator module: {e}")

    image_size_cv = (int(W), int(H))
    return board, image_size_cv


def _collect_views_for_cam(cam_idx: int, keyframes_dir: str, acc: CalibrationAccumulator,
                           min_tags: int, min_span: float):
    """
    Detect tags on all frames for a given cam index and accumulate views
    using the same gating as the mono offline flow.
    """
    pattern = os.path.join(os.path.abspath(keyframes_dir), f"frame_*_cam{cam_idx}.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[CAM{cam_idx}] No images found for pattern: {pattern}")
        return []

    # Get image size from first valid frame
    H = W = None
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is not None:
            H, W = im.shape[:2]
            break
    if H is None:
        print(f"[CAM{cam_idx}] Failed to read any images.")
        return []

    kept_frames = []
    for p in paths:
        frame = cv2.imread(p, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            corners, ids = acc.detect(gray)
        except Exception:
            corners, ids = [], None
        n = 0 if ids is None else len(ids)
        if n >= min_tags and corners and stereo_cal._has_coverage(corners, W, H, min_span):
            if acc._accumulate_single(cam_idx, corners, ids):
                kept_frames.append(frame.copy())

    print(f"[CAM{cam_idx}] Collected {len(kept_frames)} candidate views.")
    return kept_frames


def _build_obj_img_lists_for_cam(cam_idx: int, acc: CalibrationAccumulator,
                                 image_size_cv, min_tags: int, min_span: float):
    """Rebuild obj_list/img_list from accumulated corners/ids using calibrator's logic."""
    W, H = image_size_cv
    if cam_idx == 0:
        corners_all, ids_all = acc.corners0, acc.ids0
    else:
        corners_all, ids_all = acc.corners1, acc.ids1

    obj_list = []
    img_list = []
    kept_idx = []
    drop_no_ids = drop_few_tags = drop_coverage = drop_no_map = 0

    for vi, (corners_img, ids_img) in enumerate(zip(corners_all, ids_all)):
        if ids_img is None:
            drop_no_ids += 1
            continue
        if len(ids_img) < min_tags:
            drop_few_tags += 1
            continue
        if not stereo_cal._has_coverage(corners_img, W, H, min_span):
            drop_coverage += 1
            continue
        O, I = [], []
        for c, iv in zip(corners_img, ids_img):
            tid = int(iv[0])
            if tid not in acc.id_to_obj:
                continue
            O.append(acc.id_to_obj[tid])
            I.append(c.reshape(-1, 2))
        if O:
            obj_cat = np.concatenate(O, 0).astype(np.float64).reshape(-1, 1, 3)
            img_cat = np.concatenate(I, 0).astype(np.float64).reshape(-1, 1, 2)
            obj_list.append(obj_cat)
            img_list.append(img_cat)
            kept_idx.append(vi)
        else:
            drop_no_map += 1

    if (drop_no_ids + drop_few_tags + drop_coverage + drop_no_map) > 0:
        print(
            f"[CAM{cam_idx}] View filter: kept={len(kept_idx)}  "
            f"no_ids={drop_no_ids}  few_tags={drop_few_tags}  "
            f"coverage={drop_coverage}  no_map={drop_no_map}"
        )

    return obj_list, img_list, kept_idx


def _mono_calibrate_for_order(cam_idx: int, keyframes_dir: str, board, image_size_cv,
                              corner_order, min_tags: int, min_span: float,
                              max_view_rms: float):
    """Run mono calibration for a given camera and corner order; return RMS or None."""
    W, H = image_size_cv

    # New accumulator per order
    acc = CalibrationAccumulator(
        board,
        image_size_cv,
        corner_order_override=list(corner_order),
        disable_corner_autoreorder=True,
    )
    print(f"[CAM{cam_idx}] Corner order={corner_order}")

    kept_frames = _collect_views_for_cam(cam_idx, keyframes_dir, acc, min_tags, min_span)
    if not kept_frames:
        print(f"[CAM{cam_idx}] No kept frames for order {corner_order}.")
        return None

    obj_list, img_list, kept = _build_obj_img_lists_for_cam(
        cam_idx, acc, image_size_cv, min_tags, min_span
    )
    if not obj_list:
        print(f"[CAM{cam_idx}] No valid obj/img lists for order {corner_order}.")
        return None

    K_seed = stereo_cal.seed_K_pinhole(W, H, f_scale=1.0)
    t0 = time.perf_counter()
    rms, K, D, rvecs, tvecs = stereo_cal.calibrate_pinhole_full(
        obj_list, img_list, image_size_cv, K_seed
    )
    dt = time.perf_counter() - t0
    print(f"[CAM{cam_idx}] RMS={rms:.3f} (D_len={D.size if D is not None else 0}) in {dt:.2f}s")

    # Optional view-level RMS pruning for better comparability
    if max_view_rms > 0:
        keep_mask = []
        for vi, (O, I) in enumerate(zip(obj_list, img_list)):
            rv, tv = rvecs[vi], tvecs[vi]
            view_rms = stereo_cal._view_rms_pinhole(O, I, K, D, rv, tv)
            keep_mask.append(view_rms <= max_view_rms)
        if any(not m for m in keep_mask) and sum(keep_mask) >= 5:
            obj_list2 = [o for o, m in zip(obj_list, keep_mask) if m]
            img_list2 = [i for i, m in zip(img_list, keep_mask) if m]
            print(
                f"[CAM{cam_idx}] Dropped {len(obj_list) - len(obj_list2)} "
                f"high-RMS views (> {max_view_rms}px) for order {corner_order}."
            )
            rms, K, D, rvecs, tvecs = stereo_cal.calibrate_pinhole_full(
                obj_list2, img_list2, image_size_cv, K
            )
            print(f"[CAM{cam_idx}] RMS (after prune, order={corner_order}) = {rms:.3f}")

    return rms


def main():
    parser = argparse.ArgumentParser(
        description="Sweep tag corner orders for mono calibration on saved stereo keyframes."
    )
    parser.add_argument(
        "--keyframes-dir",
        type=str,
        required=True,
        help="Stereo keyframes folder (frame_XXX_cam0.png / frame_XXX_cam1.png + meta.json).",
    )
    parser.add_argument(
        "--board-source",
        type=str,
        choices=["harvard", "grid8x5"],
        default=None,
        help="Board source override; if omitted, use keyframes meta or 'harvard'.",
    )
    parser.add_argument(
        "--april-pickle",
        type=str,
        default=None,
        help="Path to local AprilBoards.pickle (for Harvard board).",
    )
    parser.add_argument(
        "--harvard-tag-size-m",
        type=float,
        default=None,
        help="Tag side length in meters for Harvard board.",
    )
    parser.add_argument(
        "--harvard-tag-spacing-m",
        type=float,
        default=None,
        help="Tag spacing (meters) for Harvard board (informational).",
    )
    parser.add_argument(
        "--max-view-rms",
        type=float,
        default=stereo_cal.MAX_VIEW_RMS_PX,
        help="Optional per-view RMS pruning threshold (pixels); <=0 disables prune.",
    )

    args, _ = parser.parse_known_args()

    keyframes_dir = os.path.abspath(args.keyframes_dir)
    if not os.path.isdir(keyframes_dir):
        raise FileNotFoundError(f"[MAIN] keyframes dir not found: {keyframes_dir}")

    # Determine image size from meta.json or first cam0 frame
    meta = _load_meta(keyframes_dir)
    if "image_size" in meta:
        W, H = meta["image_size"]
    else:
        # Fallback: read first cam0
        pattern0 = os.path.join(keyframes_dir, "frame_*_cam0.png")
        paths0 = sorted(glob.glob(pattern0))
        if not paths0:
            raise RuntimeError(f"[MAIN] No frame_*_cam0.png in {keyframes_dir}")
        im0 = cv2.imread(paths0[0], cv2.IMREAD_COLOR)
        if im0 is None:
            raise RuntimeError(f"[MAIN] Failed to read {paths0[0]}")
        H, W = im0.shape[:2]
    image_size = (W, H)
    print(f"[MAIN] image_size={image_size}")

    # Board geometry (shared across orders)
    board, image_size_cv = _resolve_board(image_size, args, keyframes_dir)

    min_tags = int(getattr(config, "MIN_MARKERS_PER_VIEW", stereo_cal.MIN_TAGS_PER_VIEW))
    min_span = float(stereo_cal.MIN_SPAN)

    print(f"[MAIN] MIN_MARKERS_PER_VIEW={min_tags} MIN_SPAN={min_span} MAX_VIEW_RMS={args.max_view_rms}")

    all_orders = list(itertools.permutations([0, 1, 2, 3], 4))
    print(f"[MAIN] Testing {len(all_orders)} corner-order permutations for cam0 and cam1.")

    results = {0: [], 1: []}  # cam_idx -> list of (order, rms or None)

    for cam_idx in (0, 1):
        print(f"\n===== CAM{cam_idx} sweep =====")
        for order in all_orders:
            rms = _mono_calibrate_for_order(
                cam_idx,
                keyframes_dir,
                board,
                image_size_cv,
                order,
                min_tags,
                min_span,
                args.max_view_rms,
            )
            results[cam_idx].append((order, rms))

    # Summary
    print("\n===== SUMMARY (RMS by corner order) =====")
    for cam_idx in (0, 1):
        print(f"\nCAM{cam_idx}:")
        # Sort by RMS (None at end)
        sorted_res = sorted(
            results[cam_idx],
            key=lambda t: float("inf") if t[1] is None else t[1],
        )
        for order, rms in sorted_res:
            if rms is None:
                print(f"  order={order} -> RMS=NA (insufficient views)")
            else:
                print(f"  order={order} -> RMS={rms:.3f}")


if __name__ == "__main__":
    main()


