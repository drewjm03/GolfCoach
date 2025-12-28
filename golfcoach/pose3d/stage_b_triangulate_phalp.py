from __future__ import annotations

"""
Stage B: triangulate 3D joints from PHALP / 4DHumans outputs using our stereo rig.

This variant:
  - Loads left/right PHALP pickle tracks
  - Aligns frames by intersection of frame indices
  - Extracts per-frame 3D joints for a chosen person index from each view
  - Projects those 3D joints into each camera using our calibrated intrinsics
  - Uses the resulting 2D joints (in our camera model) to triangulate to 3D
    in cam0 frame with StereoTriangulator
  - Computes per-joint reprojection error in pixels and NaNs out bad joints
  - Saves frames, 2D (pixels), 3D, and reprojection error arrays to disk

In effect, we ignore PHALP / HMR's internal camera prediction and re-use its
3D joints only as shape estimates, enforcing stereo consistency under our
known rig calibration.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from apps.triangulation import StereoTriangulator
from golfcoach.io.phalp_pkl import load_phalp_tracks, extract_3d_joints
from golfcoach.io.rig_config import load_rig_config


def ensure_normalized(points: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Ensure 2D points are in normalized [0,1] coordinates.

    If points look like pixel coords, convert to normalized [0,1]:
        x /= w, y /= h

    Heuristic:
        if points.max() > 2.0: treat as pixels
        else: treat as already normalized.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts

    max_val = float(np.nanmax(pts))

    # Already normalized or very small range
    if max_val <= 2.0:
        return pts.copy()

    # Convert from pixels to normalized
    w, h = image_size
    norm = pts.copy()
    norm[:, 0] /= float(w)
    norm[:, 1] /= float(h)
    return norm


def _to_pixel_coords(points: np.ndarray, image_size: Tuple[int, int], assume_pixels: bool) -> np.ndarray:
    """
    Helper to ensure we have pixel coordinates for saving / error computation.

    If `assume_pixels` is True, we just return a copy of `points`.
    Otherwise we assume `points` is normalized [0,1] and convert to pixels.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts

    if assume_pixels:
        return pts.copy()

    w, h = image_size
    out = pts.copy()
    out[:, 0] *= float(w)
    out[:, 1] *= float(h)
    return out


def triangulate_phalp_pkls(
    pkl_left: str,
    pkl_right: str,
    rig_json: str,
    out_dir: str,
    person_i_left: int = 0,
    person_i_right: int = 0,
    reproj_thresh_px: float = 25.0,
) -> None:
    """
    Triangulate 3D joints from left/right PHALP pickle outputs, using our rig.

    Steps
    -----
    1) Load both PKLs -> dict frame_idx -> frame_data
    2) Align frames by intersection of frame_idx sets, sorted
    3) For each frame:
       - extract per-view 3D joints -> (J, 3)
       - project these 3D joints into each camera using our K, D (ignoring
         PHALP's internal camera prediction)
       - convert these projected 2D joints to normalized [0,1]
       - triangulator.triangulate(norm_left, norm_right) -> (J,3) in cam0 frame
       - compute reprojection error and NaN out bad joints
    4) Save:
       - frames.npy (T,)
       - kpts2d_left.npy, kpts2d_right.npy (T,J,2) in PIXELS (from our projection)
       - kpts3d_tri.npy (T,J,3)
       - reproj_err.npy (T,J) in pixels
    """
    # Load rig + calibration
    image_size, K0, D0, K1, D1, R, T = load_rig_config(rig_json)

    # Load PHALP tracks for left/right
    tracks_left = load_phalp_tracks(pkl_left)
    tracks_right = load_phalp_tracks(pkl_right)

    # Align frames by intersection of indices
    frame_idxs_left = set(tracks_left.keys())
    frame_idxs_right = set(tracks_right.keys())
    common_frames = sorted(frame_idxs_left & frame_idxs_right)

    if not common_frames:
        raise RuntimeError("No overlapping frame indices between left and right PKLs.")

    # Determine number of joints J from the first common frame, using 3D joints
    first_idx = common_frames[0]
    left_first_3d = extract_3d_joints(tracks_left[first_idx], person_i_left)
    right_first_3d = extract_3d_joints(tracks_right[first_idx], person_i_right)
    J = min(left_first_3d.shape[0], right_first_3d.shape[0])

    if J == 0:
        raise RuntimeError("No 3D joints found in the first common frame.")

    Tlen = len(common_frames)

    frames_arr = np.asarray(common_frames, dtype=np.int32)
    kpts2d_left_px = np.full((Tlen, J, 2), np.nan, dtype=np.float32)
    kpts2d_right_px = np.full((Tlen, J, 2), np.nan, dtype=np.float32)
    kpts3d_tri = np.full((Tlen, J, 3), np.nan, dtype=np.float32)
    reproj_err = np.full((Tlen, J), np.nan, dtype=np.float32)

    # Triangulator expects normalized [0,1]
    triangulator = StereoTriangulator(K0, D0, K1, D1, R, T, image_size=image_size)

    # Prepare projection parameters for reprojection / 2D generation
    rvec0 = np.zeros(3, dtype=np.float64)
    tvec0 = np.zeros(3, dtype=np.float64)
    rvec1, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec1 = T.astype(np.float64).reshape(3, 1)

    for ti, frame_idx in enumerate(common_frames):
        frame_left = tracks_left[frame_idx]
        frame_right = tracks_right[frame_idx]

        # Extract per-frame 3D joints from each view
        j3d_left = extract_3d_joints(frame_left, person_i_left)[:J]
        j3d_right = extract_3d_joints(frame_right, person_i_right)[:J]

        # Project 3D joints into each camera using our rig intrinsics/distortion.
        # We treat each per-view 3D estimate as already expressed in its own
        # camera's frame; extrinsics are handled by StereoTriangulator later.
        # Left view -> cam0
        obj_l = j3d_left.astype(np.float64).reshape(-1, 1, 3)
        proj_l, _ = cv2.projectPoints(obj_l, rvec0, tvec0, K0, D0)
        pts2d_l_px = proj_l.reshape(-1, 2).astype(np.float32)

        # Right view -> cam1
        obj_r = j3d_right.astype(np.float64).reshape(-1, 1, 3)
        proj_r, _ = cv2.projectPoints(obj_r, rvec0, tvec0, K1, D1)
        pts2d_r_px = proj_r.reshape(-1, 2).astype(np.float32)

        # Normalize for triangulation
        pts2d_l_norm = ensure_normalized(pts2d_l_px, image_size)
        pts2d_r_norm = ensure_normalized(pts2d_r_px, image_size)

        kpts2d_left_px[ti] = pts2d_l_px
        kpts2d_right_px[ti] = pts2d_r_px

        # Triangulate in cam0 frame
        pts3d = triangulator.triangulate(pts2d_l_norm, pts2d_r_norm)
        if pts3d is None or pts3d.shape[0] == 0:
            continue

        pts3d = pts3d.astype(np.float64, copy=False)

        # Reproject into both cameras using full distortion model
        obj_pts = pts3d.reshape(-1, 1, 3)

        proj0, _ = cv2.projectPoints(obj_pts, rvec0, tvec0, K0, D0)
        proj1, _ = cv2.projectPoints(obj_pts, rvec1, tvec1, K1, D1)

        proj0 = proj0.reshape(-1, 2)
        proj1 = proj1.reshape(-1, 2)

        # Compute per-joint reprojection error in pixels
        err0 = np.linalg.norm(proj0 - pts2d_l_px.astype(np.float64), axis=1)
        err1 = np.linalg.norm(proj1 - pts2d_r_px.astype(np.float64), axis=1)
        err_mean = 0.5 * (err0 + err1)

        # Mark invalid 3D points as NaN and reflect in error
        invalid_3d = ~np.isfinite(pts3d).all(axis=1)
        err_mean[invalid_3d] = np.nan

        # Apply reprojection threshold
        bad = err_mean > reproj_thresh_px
        pts3d[bad] = np.nan

        kpts3d_tri[ti] = pts3d.astype(np.float32)
        reproj_err[ti] = err_mean.astype(np.float32)

    # Save outputs
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "frames.npy", frames_arr)
    np.save(out_path / "kpts2d_left.npy", kpts2d_left_px)
    np.save(out_path / "kpts2d_right.npy", kpts2d_right_px)
    np.save(out_path / "kpts3d_tri.npy", kpts3d_tri)
    np.save(out_path / "reproj_err.npy", reproj_err)



