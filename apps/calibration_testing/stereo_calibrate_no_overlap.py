# STEREO offline calibration (pinhole/rational model)
# - Uses the exact same mono calibration pipeline from single_cam_calibrator7.py
# - Processes offline stereo keyframes from folder structure
# - Generates diagnostic overlays for both cameras

import os, sys, time, json, glob, re, argparse
import numpy as np
import cv2

try:
    import requests, pickle
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# ---- app imports / fallback ----
try:
    from .. import config
    from ..detect import CalibrationAccumulator
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config
    from apps.detect import CalibrationAccumulator

# Import helper functions from mono script (these don't depend on globals)
# We'll handle board construction separately since it uses globals
try:
    from .single_cam_calibrator7 import (
        _has_coverage, _print_intrinsics, seed_K_pinhole, calibrate_pinhole_full,
        _save_diag_pinhole, _view_rms_pinhole, _best_kept_index
    )
except Exception:
    # Fallback: define minimal inline versions
    def _has_coverage(corners, W, H, min_span=0.35):
        xs = [p[0] for ci in corners for p in ci.reshape(4,2)]
        ys = [p[1] for ci in corners for p in ci.reshape(4,2)]
        if not xs: return False
        span_x = (max(xs)-min(xs))/float(max(1.0,W))
        span_y = (max(ys)-min(ys))/float(max(1.0,H))
        return min(span_x, span_y) >= min_span
    
    def _print_intrinsics(K, D):
        try:
            fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
        except Exception:
            fx = fy = cx = cy = float("nan")
        dlen = (int(D.size) if D is not None else 0)
        print(f"[INTR] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  D_len={dlen}")
    
    def seed_K_pinhole(W, H, f_scale=1.0):
        f = float(max(W, H)) * float(f_scale)
        K = np.eye(3, dtype=np.float64)
        K[0,0], K[1,1] = f, f
        K[0,2], K[1,2] = W*0.5, H*0.5
        return K
    
    def calibrate_pinhole_full(obj_list, img_list, image_size, K_seed=None):
        obj_std = [o.reshape(-1,3).astype(np.float32, copy=False) for o in obj_list]
        img_std = [i.reshape(-1,2).astype(np.float32, copy=False) for i in img_list]
        if K_seed is not None:
            K = K_seed.copy().astype(np.float64)
            D = np.zeros((8,1), dtype=np.float64)
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL)
        else:
            K = None
            D = None
            flags = cv2.CALIB_RATIONAL_MODEL
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_std, img_std, image_size, K, D, flags=flags, criteria=crit
        )
        return float(rms), K, D, rvecs, tvecs
    
    def _save_diag_pinhole(path_png, frame_bgr, corners, ids, acc, K, D, rvec, tvec,
                           r_cyan=8, r_mag=3, draw_corners=True, draw_proj_corners=True):
        img = frame_bgr.copy()
        obj_all = []
        img_all = []
        for c, iv in zip(corners, ids):
            tid = int(iv[0])
            if tid not in acc.id_to_obj: continue
            obj4 = acc.id_to_obj[tid].astype(np.float64).reshape(-1,1,3)
            proj,_ = cv2.projectPoints(obj4, rvec, tvec, K, D)
            det_corners = c.reshape(4,2).astype(np.float32)
            proj_corners = proj.reshape(-1,2)
            det_center = det_corners.mean(axis=0)
            proj_center = proj_corners.mean(axis=0)
            cv2.circle(img, tuple(np.int32(det_center)), int(r_cyan), (255,255,0), -1)
            cv2.circle(img, tuple(np.int32(proj_center)), max(3, int(r_mag+1)), (0,0,0), -1)
            if draw_corners:
                p_ctr = tuple(np.int32(proj_center))
                for k in range(4):
                    p_det = tuple(np.int32(det_corners[k]))
                    cv2.circle(img, p_det, max(3, int(r_cyan*0.70)), (255,255,0), -1)
                    if draw_proj_corners:
                        p_prj = tuple(np.int32(proj_corners[k]))
                        cv2.line(img, p_ctr, p_prj, (0,200,255), 2)
                        cv2.circle(img, p_prj, max(4, int(r_mag+2)), (255,0,255), -1)
            obj_all.append(acc.id_to_obj[tid].reshape(-1,3))
            img_all.append(det_corners.reshape(-1,2))
        try:
            if obj_all and img_all:
                O = np.concatenate(obj_all, axis=0).astype(np.float32).reshape(-1,1,3)
                I = np.concatenate(img_all, axis=0).astype(np.float32).reshape(-1,1,2)
                proj,_ = cv2.projectPoints(O, rvec, tvec, K, D)
                err = I.reshape(-1,2) - proj.reshape(-1,2)
                rms_px = float(np.sqrt(np.mean(np.sum(err*err, axis=1))))
                cv2.putText(img, f"Per-view RMS: {rms_px:.3f} px", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        except Exception:
            pass
        legend = "Cyan=detected corners/center, Black=projected center"
        if draw_proj_corners:
            legend += ", Magenta=projected corners"
        cv2.putText(img, legend, (16, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
        cv2.imwrite(path_png, img)
    
    def _view_rms_pinhole(obj_pts, img_pts, K, D, rvec, tvec):
        proj,_ = cv2.projectPoints(obj_pts.reshape(-1,1,3).astype(np.float32), rvec, tvec, K, D)
        proj = proj.reshape(-1,2)
        err = img_pts.reshape(-1,2).astype(np.float32) - proj
        return float(np.sqrt(np.mean(np.sum(err*err, axis=1))))
    
    def _best_kept_index(acc, kept_idxs, cam_idx=0):
        """Find best kept index. cam_idx: 0 for cam0, 1 for cam1."""
        if not kept_idxs: return 0
        ids_list = acc.ids0 if cam_idx == 0 else acc.ids1
        counts = [0 if ids_list[i] is None else len(ids_list[i]) for i in kept_idxs]
        return int(np.argmax(counts))

# ====== TUNABLES (matching mono script) ======
MIN_TAGS_PER_VIEW = 6
MIN_SPAN = 0.1
MAX_VIEW_RMS_PX = 5.0

# Wrapper for _best_kept_index to support both cameras
def _best_kept_index_stereo(acc, kept_idxs, cam_idx=0):
    """Find best kept index for stereo (supports both cameras)."""
    try:
        # Try to use imported function if it supports cam_idx
        return _best_kept_index(acc, kept_idxs, cam_idx)
    except TypeError:
        # Fallback: imported function doesn't support cam_idx (mono script version)
        # Only works for cam0, so manually check cam1
        if not kept_idxs:
            return 0
        ids_list = acc.ids0 if cam_idx == 0 else acc.ids1
        counts = [0 if ids_list[i] is None else len(ids_list[i]) for i in kept_idxs]
        return int(np.argmax(counts))

# ---------- helpers ----------
_APRIL_LOCAL_PICKLE = None
_BOARD_SOURCE = "harvard"
_HARVARD_TAG_SIZE_M = None
_HARVARD_TAG_SPACING_M = None

def _make_board_wrapper():
    """Wrapper to create board using same logic as mono script."""
    return _make_board_inline()

def _make_board_inline():
    """Inline board construction matching mono script."""
    dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
    if _BOARD_SOURCE == "grid8x5":
        TAGS_X = 8
        TAGS_Y = 5
        TAG_SIZE_M = 0.075
        TAG_SEP_M = 0.01875
        ids_grid = np.arange(TAGS_X * TAGS_Y, dtype=np.int32).reshape(TAGS_Y, TAGS_X)
        ids_grid = np.flipud(ids_grid)
        ids = ids_grid.reshape(-1, 1).astype(np.int32)
        print(f"[BOARD] Using GridBoard 8x5 (0 bottom-left -> 39 top-right), tag={TAG_SIZE_M}m sep={TAG_SEP_M}m")
        return cv2.aruco.GridBoard((TAGS_X, TAGS_Y), TAG_SIZE_M, TAG_SEP_M, dictionary, ids)
    # Harvard board
    board = _load_harvard_board_inline(dictionary)
    if board is None:
        raise RuntimeError("[BOARD] Failed to load Harvard board")
    return board

def _load_harvard_board_inline(dictionary):
    """Load Harvard board inline."""
    local_override = _APRIL_LOCAL_PICKLE or os.environ.get("APRIL_BOARDS_PICKLE", "").strip()
    if local_override:
        try:
            with open(local_override, "rb") as f:
                data = pickle.loads(f.read())
            tag_size_m = _harvard_tag_size_inline(data)
            board = _parse_board_pickle_inline(dictionary, data, tag_size_m=tag_size_m)
            if board:
                print(f"[BOARD] Loaded Harvard board from: {local_override}")
                return board
        except Exception as e:
            print(f"[BOARD] Failed to read local pickle: {e}")
        return None
    if not HAVE_REQUESTS:
        return None
    urls = [
        "https://github.com/Harvard-CS283/pset-data/raw/f1a90573ae88cd530a3df3cd0cea71aa2363b1b3/april/AprilBoards.pickle",
        "https://raw.githubusercontent.com/Harvard-CS283/pset-data/f1a90573ae88cd530a3df3cd0cea71aa2363b1b3/april/AprilBoards.pickle",
    ]
    for url in urls:
        for attempt in range(2):
            try:
                resp = requests.get(url, timeout=15, allow_redirects=True, headers={"User-Agent": "GolfCoach/1.0"})
                if resp.status_code != 200:
                    continue
                data = pickle.loads(resp.content)
                tag_size_m = _harvard_tag_size_inline(data)
                board = _parse_board_pickle_inline(dictionary, data, tag_size_m=tag_size_m)
                if board:
                    print(f"[BOARD] Loaded Harvard board from: {url}")
                    return board
            except Exception as e:
                print(f"[BOARD] Fetch attempt {attempt+1} failed: {e}")
                time.sleep(0.5)
    return None

def _harvard_tag_size_inline(data):
    """Extract tag size from data."""
    if _HARVARD_TAG_SIZE_M is not None:
        return float(_HARVARD_TAG_SIZE_M)
    env = os.environ.get("HARVARD_TAG_SIZE_M", "").strip()
    if env:
        return float(env)
    if isinstance(data, dict):
        for k in ("tag_size", "tag_side", "tag_width", "april_tag_side", "tag_length", "marker_length"):
            v = data.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        for pk in ("params", "metadata", "board_params"):
            sub = data.get(pk)
            if isinstance(sub, dict):
                for k in ("tag_size", "tag_side", "tag_width", "april_tag_side", "tag_length", "marker_length"):
                    v = sub.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
    return None

def _parse_board_pickle_inline(dictionary, data, tag_size_m=None):
    """Parse board pickle inline (simplified)."""
    try:
        if hasattr(data, "getObjPoints"):
            return data
        candidate_keys = ["at_board_d", "at_coarseboard", "board", "at_board"]
        board_data = None
        for k in candidate_keys:
            if k in getattr(data, "keys", lambda: [])():
                board_data = data.get(k)
                break
        if board_data is None and isinstance(data, dict):
            for k, v in data.items():
                if hasattr(v, "getObjPoints"):
                    board_data = v
                    break
        if board_data is None:
            return None
        if hasattr(board_data, "getObjPoints"):
            return board_data
        obj_points = []
        ids_list = []
        if isinstance(board_data, dict) and 'ids' in board_data:
            ids_raw = list(board_data.get('ids', []))
            corners_raw = board_data.get('objPoints', board_data.get('corners', []))
            for tid, pts in zip(ids_raw, corners_raw):
                P = np.array(pts, dtype=np.float32).reshape(-1, 3 if np.array(pts).shape[-1] == 3 else 2)
                if P.shape[1] == 2:
                    P = np.hstack([P, np.zeros((P.shape[0], 1), dtype=np.float32)])
                if P.shape[0] == 4:
                    obj_points.append(P.reshape(4, 3))
                    ids_list.append(int(tid))
        elif isinstance(board_data, (list, tuple)):
            centers_units = []
            for item in board_data:
                if isinstance(item, dict) and 'tag_id' in item and 'center' in item:
                    c = np.array(item['center'], dtype=np.float32).reshape(-1)
                    centers_units.append(c[:2].astype(np.float32))
            center_scale = None
            if centers_units and (tag_size_m is not None) and (_HARVARD_TAG_SPACING_M is not None):
                desired_cc_m = float(tag_size_m) + float(_HARVARD_TAG_SPACING_M)
                if desired_cc_m > 0:
                    C = np.vstack(centers_units)
                    dists = []
                    for i in range(C.shape[0]):
                        di = np.hypot(C[i,0]-C[:,0], C[i,1]-C[:,1])
                        di = di[di > 1e-6]
                        if di.size > 0:
                            dists.append(np.min(di))
                    if dists:
                        nn_units = float(np.median(np.array(dists, dtype=np.float32)))
                        if nn_units > 0:
                            center_scale = desired_cc_m / nn_units
            for item in board_data:
                if not isinstance(item, dict):
                    continue
                if 'id' in item and ('corners' in item or 'objPoints' in item):
                    tid = int(item['id'])
                    pts = item.get('objPoints', item.get('corners'))
                    P = np.array(pts, dtype=np.float32).reshape(-1, 3 if np.array(pts).shape[-1] == 3 else 2)
                    if P.shape[1] == 2:
                        P = np.hstack([P, np.zeros((P.shape[0], 1), dtype=np.float32)])
                    if P.shape[0] == 4:
                        obj_points.append(P.reshape(4, 3))
                        ids_list.append(tid)
                elif 'tag_id' in item and 'center' in item:
                    if tag_size_m is None:
                        raise RuntimeError("[BOARD] Harvard centers found but tag size unknown.")
                    tid = int(item['tag_id'])
                    c = np.array(item['center'], dtype=np.float32).reshape(-1)
                    if center_scale is not None:
                        c = c * float(center_scale)
                    if c.size == 2:
                        cx, cy, cz = float(c[0]), float(c[1]), 0.0
                    else:
                        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
                    half = float(tag_size_m) * 0.5
                    pts3 = np.array([
                        [cx - half, cy - half, cz],
                        [cx + half, cy - half, cz],
                        [cx + half, cy + half, cz],
                        [cx - half, cy + half, cz],
                    ], dtype=np.float32)
                    obj_points.append(pts3.reshape(4, 3))
                    ids_list.append(tid)
        if not obj_points or not ids_list:
            return None
        ids_arr = np.array(ids_list, dtype=np.int32).reshape(-1, 1)
        try:
            return cv2.aruco.Board(obj_points, dictionary, ids_arr)
        except Exception:
            try:
                return cv2.aruco.Board_create(obj_points, dictionary, ids_arr)
            except Exception:
                return None
    except Exception:
        return None

def load_keyframes(keyframes_dir):
    """Load stereo keyframes from folder structure.
    
    Returns:
        List of tuples: (frame_num, img0_path, img1_path, json_path, frame0_bgr, frame1_bgr, json_data)
    """
    keyframes_dir = os.path.abspath(keyframes_dir)
    if not os.path.isdir(keyframes_dir):
        raise RuntimeError(f"[KEYFRAMES] Directory not found: {keyframes_dir}")
    
    # Find all frame_XXX_cam0.png files
    pattern = os.path.join(keyframes_dir, "frame_*_cam0.png")
    cam0_files = sorted(glob.glob(pattern))
    
    if not cam0_files:
        raise RuntimeError(f"[KEYFRAMES] No frame_*_cam0.png files found in {keyframes_dir}")
    
    keyframes = []
    for cam0_path in cam0_files:
        # Extract frame number
        match = re.search(r'frame_(\d+)_cam0\.png', cam0_path)
        if not match:
            continue
        frame_num = int(match.group(1))
        
        # Construct paths
        cam1_path = cam0_path.replace("_cam0.png", "_cam1.png")
        json_path = os.path.join(keyframes_dir, f"frame_{frame_num:03d}.json")
        
        if not os.path.exists(cam1_path):
            print(f"[KEYFRAMES] Warning: {cam1_path} not found, skipping frame {frame_num}")
            continue
        
        # Load images
        frame0_bgr = cv2.imread(cam0_path, cv2.IMREAD_COLOR)
        frame1_bgr = cv2.imread(cam1_path, cv2.IMREAD_COLOR)
        
        if frame0_bgr is None or frame1_bgr is None:
            print(f"[KEYFRAMES] Warning: Failed to load images for frame {frame_num}")
            continue
        
        # Load JSON if exists
        json_data = None
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"[KEYFRAMES] Warning: Failed to load JSON for frame {frame_num}: {e}")
        
        keyframes.append((frame_num, cam0_path, cam1_path, json_path, frame0_bgr, frame1_bgr, json_data))
    
    print(f"[KEYFRAMES] Loaded {len(keyframes)} stereo keyframes from {keyframes_dir}")
    return keyframes

def load_detections_from_json(json_data, cam_idx):
    """Extract corners and ids from JSON data for one camera.
    
    Returns:
        corners (list of 1x4x2 arrays), ids (Nx1 array)
    """
    if json_data is None:
        return None, None
    
    ids_key = f"ids{cam_idx}"
    corners_key = f"corners{cam_idx}"
    
    if ids_key not in json_data or corners_key not in json_data:
        return None, None
    
    ids_list = json_data[ids_key]
    corners_list = json_data[corners_key]
    
    if not ids_list or not corners_list or len(ids_list) != len(corners_list):
        return None, None
    
    # Convert to numpy arrays
    ids = np.array(ids_list, dtype=np.int32).reshape(-1, 1)
    corners = []
    for corner_data in corners_list:
        if len(corner_data) != 4:
            return None, None
        corner_array = np.array(corner_data, dtype=np.float32).reshape(1, 4, 2)
        corners.append(corner_array)
    
    return corners, ids


def build_obj_img_points(acc, corners_img, ids_img):
    """
    Build concatenated 3D-2D correspondences for a single camera view.

    Args:
        acc: CalibrationAccumulator with id_to_obj mapping (tag_id -> 4x3)
        corners_img: list of (1, 4, 2) arrays of detected corners
        ids_img: (N, 1) array of tag IDs

    Returns:
        obj_pts: (M, 1, 3) float32 array of object points (M >= 4) or None
        img_pts: (M, 1, 2) float32 array of image points (M >= 4) or None
    """
    if ids_img is None or corners_img is None:
        return None, None

    obj_list = []
    img_list = []
    for c, iv in zip(corners_img, ids_img):
        tid = int(iv[0])
        if tid not in acc.id_to_obj:
            continue
        obj = acc.id_to_obj[tid]
        # Expect 4x3 per tag
        assert obj.shape[0] == 4 and obj.shape[1] == 3, "acc.id_to_obj entries must be 4x3"
        obj_list.append(obj.reshape(-1, 1, 3).astype(np.float32))
        img_list.append(c.reshape(-1, 1, 2).astype(np.float32))

    if not obj_list or not img_list:
        return None, None

    obj_pts = np.concatenate(obj_list, axis=0)
    img_pts = np.concatenate(img_list, axis=0)

    # Minimal sanity check
    assert obj_pts.shape[0] == img_pts.shape[0] and obj_pts.shape[0] >= 4

    return obj_pts, img_pts


def solve_pnp_pose(obj_pts, img_pts, K, D):
    """
    Solve PnP for a single view using known intrinsics.

    Prefers IPPE_SQUARE when exactly 4 points (single tag), otherwise ITERATIVE.

    Returns:
        ok: bool
        R: (3, 3) rotation matrix
        t: (3, 1) translation vector
        rms_px: per-view reprojection RMS in pixels (float)
    """
    obj_pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float32).reshape(-1, 1, 2)

    assert obj_pts.shape[0] == img_pts.shape[0] and obj_pts.shape[0] >= 4

    n = obj_pts.shape[0]
    try:
        if n == 4:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts.reshape(-1, 3),
                img_pts.reshape(-1, 2),
                K,
                D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts.reshape(-1, 3),
                    img_pts.reshape(-1, 2),
                    K,
                    D,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts.reshape(-1, 3),
                img_pts.reshape(-1, 2),
                K,
                D,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
    except cv2.error:
        return False, None, None, float("inf")

    if not ok:
        return False, None, None, float("inf")

    R, _ = cv2.Rodrigues(rvec)
    rms_px = _view_rms_pinhole(obj_pts, img_pts, K, D, rvec, tvec)

    # Shape checks
    assert R.shape == (3, 3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    assert tvec.shape == (3, 1)

    return True, R.astype(np.float64), tvec, float(rms_px)


def invert_T(R, t):
    """
    Invert a rigid transform T_ab: X_a = R * X_b + t.

    Returns T_ba with X_b = R_inv * X_a + t_inv.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)

    R_inv = R.T
    t_inv = -R_inv @ t

    assert R_inv.shape == (3, 3)
    assert t_inv.shape == (3, 1)

    return R_inv, t_inv


def compose_T(R_ab, t_ab, R_bc, t_bc):
    """
    Compose two transforms T_ab and T_bc to get T_ac.

    T_ab: X_a = R_ab * X_b + t_ab
    T_bc: X_b = R_bc * X_c + t_bc
    T_ac: X_a = R_ac * X_c + t_ac
    """
    R_ab = np.asarray(R_ab, dtype=np.float64).reshape(3, 3)
    t_ab = np.asarray(t_ab, dtype=np.float64).reshape(3, 1)
    R_bc = np.asarray(R_bc, dtype=np.float64).reshape(3, 3)
    t_bc = np.asarray(t_bc, dtype=np.float64).reshape(3, 1)

    R_ac = R_ab @ R_bc
    t_ac = R_ab @ t_bc + t_ab

    assert R_ac.shape == (3, 3)
    assert t_ac.shape == (3, 1)

    return R_ac, t_ac


def rodrigues_avg(R_list):
    """
    Average a list of rotation matrices in Rodrigues (axis-angle) space.

    Args:
        R_list: list of (3, 3) rotation matrices

    Returns:
        R_mean: (3, 3) rotation matrix
    """
    if not R_list:
        raise ValueError("rodrigues_avg requires at least one rotation matrix")

    rvecs = []
    for R in R_list:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(R)
        rvecs.append(rvec.reshape(1, 3))

    rvecs = np.vstack(rvecs)
    assert rvecs.shape[1] == 3

    r_mean = np.mean(rvecs, axis=0).reshape(3, 1)
    R_mean, _ = cv2.Rodrigues(r_mean)
    assert R_mean.shape == (3, 3)

    return R_mean

def main():
    parser = argparse.ArgumentParser(description="Stereo offline calibration using keyframes")
    parser.add_argument("--keyframes-dir", type=str, default="data/stereo_keyframes_20251119_014056",
                        help="Directory containing stereo keyframes")
    parser.add_argument("--april-pickle", type=str, default=None,
                        help="Path to local AprilBoards.pickle (overrides network)")
    parser.add_argument("--save-all-diag", action="store_true",
                        help="Save diagnostic overlays for all kept frames")
    parser.add_argument("--board-source", type=str, choices=["harvard", "grid8x5"], default="harvard",
                        help="Which board to use: Harvard pickle or original 8x5 grid")
    parser.add_argument("--harvard-tag-size-m", type=float, default=None,
                        help="Tag side length in meters for Harvard board (overrides pickle/env)")
    parser.add_argument("--harvard-tag-spacing-m", type=float, default=None,
                        help="Tag spacing (meters) for Harvard board (informational)")
    parser.add_argument("--corner-order", type=str, default=None,
                        help="Manual corner order override as four comma-separated indices, e.g. '0,1,2,3'")
    args = parser.parse_known_args()[0]
    
    # Set globals for board construction
    global _APRIL_LOCAL_PICKLE, _BOARD_SOURCE, _HARVARD_TAG_SIZE_M, _HARVARD_TAG_SPACING_M
    _APRIL_LOCAL_PICKLE = args.april_pickle
    _BOARD_SOURCE = args.board_source
    _HARVARD_TAG_SIZE_M = args.harvard_tag_size_m
    _HARVARD_TAG_SPACING_M = args.harvard_tag_spacing_m
    
    # Load keyframes
    print(f"[MAIN] Loading keyframes from: {args.keyframes_dir}")
    keyframes = load_keyframes(args.keyframes_dir)
    
    if not keyframes:
        print("[MAIN] No valid keyframes found; aborting.")
        return
    
    # Get image size from first frame
    _, _, _, _, frame0_bgr, _, _ = keyframes[0]
    H, W = frame0_bgr.shape[:2]
    image_size = (W, H)
    print(f"[MAIN] Image size: {W}x{H}")
    
    # Load board
    try:
        board = _make_board_wrapper()
    except Exception as e:
        print(f"[BOARD] Failed to load board: {e}")
        return
    
    # Parse corner order override
    corner_order_override = None
    disable_autoreorder = False
    if args.corner_order:
        try:
            parts = [int(x.strip()) for x in args.corner_order.split(",")]
            if len(parts) == 4 and sorted(parts) == [0, 1, 2, 3]:
                corner_order_override = parts
                disable_autoreorder = True
            else:
                print(f"[WARN] Ignoring invalid --corner-order '{args.corner_order}'")
        except Exception as e:
            print(f"[WARN] Failed to parse --corner-order: {e}")
    
    # Initialize accumulator
    acc = CalibrationAccumulator(board, image_size,
                                 corner_order_override=corner_order_override,
                                 disable_corner_autoreorder=disable_autoreorder)
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())
    
    # Process keyframes
    # Maintain per-view frame lists aligned with acc.corners0 / acc.corners1
    frames0_by_view = []
    frames1_by_view = []
    kept_frames = []  # stereo samples (for debugging / potential future use)
    for frame_num, cam0_path, cam1_path, json_path, frame0_bgr, frame1_bgr, json_data in keyframes:
        # Try to load detections from JSON
        corners0, ids0 = load_detections_from_json(json_data, 0)
        corners1, ids1 = load_detections_from_json(json_data, 1)
        
        # If JSON missing or invalid, detect from images
        if corners0 is None or ids0 is None:
            gray0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
            try:
                corners0, ids0 = acc.detect(gray0)
            except Exception:
                corners0, ids0 = [], None
        
        if corners1 is None or ids1 is None:
            gray1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)
            try:
                corners1, ids1 = acc.detect(gray1)
            except Exception:
                corners1, ids1 = [], None
        
        # Filter views using same criteria as mono script
        n0 = 0 if ids0 is None else len(ids0)
        n1 = 0 if ids1 is None else len(ids1)
        
        ok0 = (n0 >= MIN_TAGS_PER_VIEW and corners0 and _has_coverage(corners0, W, H, MIN_SPAN))
        ok1 = (n1 >= MIN_TAGS_PER_VIEW and corners1 and _has_coverage(corners1, W, H, MIN_SPAN))

        if ok0:
            if acc._accumulate_single(0, corners0, ids0):
                # Keep frame aligned with this mono view index
                frames0_by_view.append(frame0_bgr.copy())
        
        if ok1:
            if acc._accumulate_single(1, corners1, ids1):
                # Keep frame aligned with this mono view index
                frames1_by_view.append(frame1_bgr.copy())
        
        # Create stereo sample if both cameras have valid, well-distributed detections
        if ok0 and ok1 and ids0 is not None and ids1 is not None:
            stereo_sample = acc._match_stereo(corners0, ids0, corners1, ids1)
            if stereo_sample is not None and stereo_sample.obj_pts.shape[0] >= MIN_TAGS_PER_VIEW * 4:
                acc.stereo_samples.append(stereo_sample)
                kept_frames.append((frame0_bgr.copy(), frame1_bgr.copy()))
    
    print(f"[CAL] Collected views: cam0={len(acc.corners0)}, cam1={len(acc.corners1)}, stereo={len(acc.stereo_samples)}")
    
    # Check if we have enough samples
    if len(acc.corners0) == 0 or len(acc.corners1) == 0:
        print("[CAL] Not enough valid samples; aborting.")
        return
    
    # Mono calibration for cam0 (using same logic as mono script)
    print("[CAL] Calibrating mono intrinsics for cam0...")
    K_seed = seed_K_pinhole(W, H, f_scale=1.0)
    obj0_list, img0_list, kept0 = [], [], []
    drop_no_ids = drop_few_tags = drop_coverage = drop_no_map = 0
    
    for vi, (corners_img, ids_img) in enumerate(zip(acc.corners0, acc.ids0)):
        if ids_img is None:
            drop_no_ids += 1
            continue
        if len(ids_img) < MIN_TAGS_PER_VIEW:
            drop_few_tags += 1
            continue
        if not _has_coverage(corners_img, W, H, MIN_SPAN):
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
            obj0_list.append(obj_cat)
            img0_list.append(img_cat)
            kept0.append(vi)
        else:
            drop_no_map += 1
    
    if (drop_no_ids + drop_few_tags + drop_coverage + drop_no_map) > 0:
        print(f"[CAL] Cam0 filter: kept={len(kept0)}  no_ids={drop_no_ids}  few_tags={drop_few_tags}  coverage={drop_coverage}  no_map={drop_no_map}")
    
    if not obj0_list:
        print("[CAL] Cam0: Not enough valid samples after filtering; aborting.")
        return
    
    rms0, K0, D0, rvecs0, tvecs0 = calibrate_pinhole_full(obj0_list, img0_list, image_size, K_seed)
    print(f"[CAL] Cam0 RMS: {rms0:.3f} (D has {D0.size if D0 is not None else 0} coeffs)")
    _print_intrinsics(K0, D0)
    
    # Optional view-level RMS pruning for cam0
    if MAX_VIEW_RMS_PX > 0:
        # Debug: log per-view RMS before pruning
        print("[CAL] Cam0 per-view RMS:")
        for vi, (O, I) in enumerate(zip(obj0_list, img0_list)):
            rv, tv = rvecs0[vi], tvecs0[vi]
            view_rms = _view_rms_pinhole(O, I, K0, D0, rv, tv)
            print(f"  view {vi:03d} (global index {kept0[vi]}): {view_rms:.2f} px")

        keep_mask0 = []
        for vi, (O, I) in enumerate(zip(obj0_list, img0_list)):
            rv, tv = rvecs0[vi], tvecs0[vi]
            view_rms = _view_rms_pinhole(O, I, K0, D0, rv, tv)
            keep_mask0.append(view_rms <= MAX_VIEW_RMS_PX)
        if any(not m for m in keep_mask0) and sum(keep_mask0) >= 5:
            obj0_list2 = [o for o, m in zip(obj0_list, keep_mask0) if m]
            img0_list2 = [i for i, m in zip(img0_list, keep_mask0) if m]
            print(f"[CAL] Cam0: Dropped {len(obj0_list)-len(obj0_list2)} high-RMS views (> {MAX_VIEW_RMS_PX}px)")
            rms0, K0, D0, rvecs0, tvecs0 = calibrate_pinhole_full(obj0_list2, img0_list2, image_size, K0)
            print(f"[CAL] Cam0 RMS (after prune): {rms0:.3f} (D has {D0.size if D0 is not None else 0} coeffs)")
            kept0 = [k for k, m in zip(kept0, keep_mask0) if m]
            obj0_list, img0_list = obj0_list2, img0_list2
    
    # Mono calibration for cam1 (same logic)
    print("[CAL] Calibrating mono intrinsics for cam1...")
    obj1_list, img1_list, kept1 = [], [], []
    drop_no_ids = drop_few_tags = drop_coverage = drop_no_map = 0
    
    for vi, (corners_img, ids_img) in enumerate(zip(acc.corners1, acc.ids1)):
        if ids_img is None:
            drop_no_ids += 1
            continue
        if len(ids_img) < MIN_TAGS_PER_VIEW:
            drop_few_tags += 1
            continue
        if not _has_coverage(corners_img, W, H, MIN_SPAN):
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
            obj1_list.append(obj_cat)
            img1_list.append(img_cat)
            kept1.append(vi)
        else:
            drop_no_map += 1
    
    if (drop_no_ids + drop_few_tags + drop_coverage + drop_no_map) > 0:
        print(f"[CAL] Cam1 filter: kept={len(kept1)}  no_ids={drop_no_ids}  few_tags={drop_few_tags}  coverage={drop_coverage}  no_map={drop_no_map}")
    
    if not obj1_list:
        print("[CAL] Cam1: Not enough valid samples after filtering; aborting.")
        return
    
    rms1, K1, D1, rvecs1, tvecs1 = calibrate_pinhole_full(obj1_list, img1_list, image_size, K_seed)
    print(f"[CAL] Cam1 RMS: {rms1:.3f} (D has {D1.size if D1 is not None else 0} coeffs)")
    _print_intrinsics(K1, D1)
    
    # Optional view-level RMS pruning for cam1
    if MAX_VIEW_RMS_PX > 0:
        # Debug: log per-view RMS before pruning
        print("[CAL] Cam1 per-view RMS:")
        for vi, (O, I) in enumerate(zip(obj1_list, img1_list)):
            rv, tv = rvecs1[vi], tvecs1[vi]
            view_rms = _view_rms_pinhole(O, I, K1, D1, rv, tv)
            print(f"  view {vi:03d} (global index {kept1[vi]}): {view_rms:.2f} px")

        keep_mask1 = []
        for vi, (O, I) in enumerate(zip(obj1_list, img1_list)):
            rv, tv = rvecs1[vi], tvecs1[vi]
            view_rms = _view_rms_pinhole(O, I, K1, D1, rv, tv)
            keep_mask1.append(view_rms <= MAX_VIEW_RMS_PX)
        if any(not m for m in keep_mask1) and sum(keep_mask1) >= 2:
            obj1_list2 = [o for o, m in zip(obj1_list, keep_mask1) if m]
            img1_list2 = [i for i, m in zip(img1_list, keep_mask1) if m]
            print(f"[CAL] Cam1: Dropped {len(obj1_list)-len(obj1_list2)} high-RMS views (> {MAX_VIEW_RMS_PX}px)")
            rms1, K1, D1, rvecs1, tvecs1 = calibrate_pinhole_full(obj1_list2, img1_list2, image_size, K1)
            print(f"[CAL] Cam1 RMS (after prune): {rms1:.3f} (D has {D1.size if D1 is not None else 0} coeffs)")
            kept1 = [k for k, m in zip(kept1, keep_mask1) if m]
            obj1_list, img1_list = obj1_list2, img1_list2
    
    # ----- Save mono diagnostic overlays (always, even if stereo fails later) -----
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    out_dir = os.path.join(repo_root, "data")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    diag_dir = os.path.join(out_dir, f"stereo_offline_diag_{stamp}")
    os.makedirs(diag_dir, exist_ok=True)

    # Best view indices per cam (by number of points)
    best0_idx = _best_kept_index_stereo(acc, kept0, cam_idx=0) if kept0 else 0
    best1_idx = _best_kept_index_stereo(acc, kept1, cam_idx=1) if kept1 else 0
    view0_idx = kept0[best0_idx] if kept0 and best0_idx < len(kept0) else (kept0[0] if kept0 else 0)
    view1_idx = kept1[best1_idx] if kept1 and best1_idx < len(kept1) else (kept1[0] if kept1 else 0)

    # Find corresponding frames for the chosen best views
    if frames0_by_view and 0 <= view0_idx < len(frames0_by_view):
        frame0_bgr_diag = frames0_by_view[view0_idx]
    else:
        frame0_bgr_diag = keyframes[0][4]

    if frames1_by_view and 0 <= view1_idx < len(frames1_by_view):
        frame1_bgr_diag = frames1_by_view[view1_idx]
    else:
        frame1_bgr_diag = keyframes[0][5]

    # Get best corners and ids
    best_corners0 = acc.corners0[view0_idx] if view0_idx < len(acc.corners0) else acc.corners0[0]
    best_ids0 = acc.ids0[view0_idx] if view0_idx < len(acc.ids0) else acc.ids0[0]
    best_corners1 = acc.corners1[view1_idx] if view1_idx < len(acc.corners1) else acc.corners1[0]
    best_ids1 = acc.ids1[view1_idx] if view1_idx < len(acc.ids1) else acc.ids1[0]

    # Get corresponding rvec/tvec
    rvec0_best = rvecs0[best0_idx] if best0_idx < len(rvecs0) else rvecs0[0]
    tvec0_best = tvecs0[best0_idx] if best0_idx < len(tvecs0) else tvecs0[0]
    rvec1_best = rvecs1[best1_idx] if best1_idx < len(rvecs1) else rvecs1[0]
    tvec1_best = tvecs1[best1_idx] if best1_idx < len(tvecs1) else tvecs1[0]

    # Save single best-view diagnostics
    out_png0 = os.path.join(diag_dir, f"stereo_offline_cam0_diag_{stamp}.png")
    _save_diag_pinhole(out_png0, frame0_bgr_diag, best_corners0, best_ids0, acc, K0, D0,
                       rvec0_best, tvec0_best, r_cyan=8, r_mag=3)
    print(f"[SAVE] Cam0 diagnostic -> {out_png0}")

    out_png1 = os.path.join(diag_dir, f"stereo_offline_cam1_diag_{stamp}.png")
    _save_diag_pinhole(out_png1, frame1_bgr_diag, best_corners1, best_ids1, acc, K1, D1,
                       rvec1_best, tvec1_best, r_cyan=8, r_mag=3)
    print(f"[SAVE] Cam1 diagnostic -> {out_png1}")

    # Save overlays for all kept frames if requested
    if args.save_all_diag:
        for idx, (vi0, O0, I0) in enumerate(zip(kept0, obj0_list, img0_list)):
            view_idx0 = vi0
            if view_idx0 < len(acc.corners0):
                corners0_here = acc.corners0[view_idx0]
                ids0_here = acc.ids0[view_idx0]
                if frames0_by_view and 0 <= view_idx0 < len(frames0_by_view):
                    frame0_here = frames0_by_view[view_idx0]
                else:
                    frame0_here = frame0_bgr_diag
                out_png_i = os.path.join(diag_dir, f"stereo_offline_cam0_diag_{stamp}_view{idx:03d}.png")
                _save_diag_pinhole(out_png_i, frame0_here, corners0_here, ids0_here, acc, K0, D0,
                                   rvecs0[idx], tvecs0[idx], r_cyan=8, r_mag=3)

        for idx, (vi1, O1, I1) in enumerate(zip(kept1, obj1_list, img1_list)):
            view_idx1 = vi1
            if view_idx1 < len(acc.corners1):
                corners1_here = acc.corners1[view_idx1]
                ids1_here = acc.ids1[view_idx1]
                if frames1_by_view and 0 <= view_idx1 < len(frames1_by_view):
                    frame1_here = frames1_by_view[view_idx1]
                else:
                    frame1_here = frame1_bgr_diag
                out_png_i = os.path.join(diag_dir, f"stereo_offline_cam1_diag_{stamp}_view{idx:03d}.png")
                _save_diag_pinhole(out_png_i, frame1_here, corners1_here, ids1_here, acc, K1, D1,
                                   rvecs1[idx], tvecs1[idx], r_cyan=8, r_mag=3)
        print(f"[SAVE] Saved diagnostic overlays for all views to {diag_dir}")

    # Stereo calibration (no tag-ID overlap required)
    print("[CAL] Estimating stereo extrinsics from per-keyframe PnP (no tag-ID overlap required)...")

    stereo_rms_thresh = MAX_VIEW_RMS_PX if MAX_VIEW_RMS_PX > 0 else 5.0
    extr_R_list = []
    extr_t_list = []
    per_view_rms = []
    baseline_norms = []

    # Iterate over original keyframes so cam0/cam1 are synchronized
    for frame_num, cam0_path, cam1_path, json_path, frame0_bgr, frame1_bgr, json_data in keyframes:
        # Reload or re-detect corners/ids for both cameras
        corners0, ids0 = load_detections_from_json(json_data, 0)
        corners1, ids1 = load_detections_from_json(json_data, 1)

        if corners0 is None or ids0 is None:
            gray0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
            try:
                corners0, ids0 = acc.detect(gray0)
            except Exception:
                corners0, ids0 = [], None

        if corners1 is None or ids1 is None:
            gray1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)
            try:
                corners1, ids1 = acc.detect(gray1)
            except Exception:
                corners1, ids1 = [], None

        n0 = 0 if ids0 is None else len(ids0)
        n1 = 0 if ids1 is None else len(ids1)

        ok0 = (n0 >= MIN_TAGS_PER_VIEW and corners0 and _has_coverage(corners0, W, H, MIN_SPAN))
        ok1 = (n1 >= MIN_TAGS_PER_VIEW and corners1 and _has_coverage(corners1, W, H, MIN_SPAN))
        if not (ok0 and ok1):
            continue

        obj0, img0 = build_obj_img_points(acc, corners0, ids0)
        obj1, img1 = build_obj_img_points(acc, corners1, ids1)
        if obj0 is None or obj1 is None:
            continue

        ok_pnp0, R0, t0, rms0_view = solve_pnp_pose(obj0, img0, K0, D0)
        ok_pnp1, R1, t1, rms1_view = solve_pnp_pose(obj1, img1, K1, D1)
        if not (ok_pnp0 and ok_pnp1):
            continue

        view_rms = max(rms0_view, rms1_view)
        if view_rms > stereo_rms_thresh:
            continue

        # Compose T_c1_c0 = T_c1_b @ inv(T_c0_b)
        R_b_c0, t_b_c0 = invert_T(R0, t0)
        R10, t10 = compose_T(R1, t1, R_b_c0, t_b_c0)

        extr_R_list.append(R10)
        extr_t_list.append(t10)
        per_view_rms.append(view_rms)
        baseline_norms.append(float(np.linalg.norm(t10.reshape(-1))))

    num_extr_samples = len(extr_R_list)
    if num_extr_samples == 0:
        print("[CAL] No valid stereo extrinsic samples; aborting stereo calibration.")
        return

    # Baseline-norm statistics and outlier rejection (median ± 25%)
    norms = np.asarray(baseline_norms, dtype=np.float64)
    median_norm = float(np.median(norms))
    std_norm = float(np.std(norms)) if norms.size > 1 else 0.0
    print(f"[BASE] baseline |t10| raw median={median_norm:.4f} std={std_norm:.4f} over {len(norms)} samples")

    if median_norm > 0:
        band = 0.25 * median_norm
        keep_mask = np.abs(norms - median_norm) <= band
    else:
        keep_mask = np.ones_like(norms, dtype=bool)

    if keep_mask.sum() == 0:
        print("[CAL] All stereo samples rejected by baseline filter; aborting stereo calibration.")
        return

    # Apply baseline filter
    extr_R_kept = [R for R, k in zip(extr_R_list, keep_mask) if k]
    extr_t_kept = [t for t, k in zip(extr_t_list, keep_mask) if k]
    per_view_rms_kept = [r for r, k in zip(per_view_rms, keep_mask) if k]
    norms_kept = norms[keep_mask]

    num_extr_kept = len(extr_R_kept)
    if num_extr_kept == 0:
        print("[CAL] No stereo samples left after baseline filtering; aborting stereo calibration.")
        return

    baseline_norm_median = float(np.median(norms_kept))
    baseline_norm_std = float(np.std(norms_kept)) if norms_kept.size > 1 else 0.0
    print(f"[BASE] baseline |t10| median={baseline_norm_median:.4f} std={baseline_norm_std:.4f} over {num_extr_kept} samples")

    # Aggregate rotation and translation
    R = rodrigues_avg(extr_R_kept)
    t_stack = np.hstack([t.reshape(3, 1) for t in extr_t_kept])
    assert t_stack.shape[0] == 3
    T = np.median(t_stack, axis=1, keepdims=True)  # (3,1)

    rms_st = float(np.mean(per_view_rms_kept))
    print(f"[CAL] Stereo RMS (per-view mean over kept samples): {rms_st:.3f}")
    print(f"[CAL] R (cam1<-cam0):\n{R}")
    print(f"[CAL] T (cam1<-cam0): {T.flatten()}")

    # Save calibration JSON (no essential matrix / fundamental matrix here)
    out_json = os.path.join(out_dir, f"stereo_offline_calibration_no_overlap_{stamp}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image_size": [W, H],
            "K0": K0.tolist(),
            "D0": D0.tolist(),
            "K1": K1.tolist(),
            "D1": D1.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "E": None,
            "F": None,
            "rms0": float(rms0),
            "rms1": float(rms1),
            "rms_stereo": float(rms_st),
            "num_keyframes": len(keyframes),
            "num_stereo_samples": int(num_extr_samples),
            "num_extr_samples": int(num_extr_samples),
            "num_extr_kept": int(num_extr_kept),
            "baseline_norm_median": float(baseline_norm_median),
            "baseline_norm_std": float(baseline_norm_std),
            "model": "pinhole_rational",
            "board_source": args.board_source
        }, f, indent=2)
    print(f"[SAVE] Wrote {out_json}")

    print("[MAIN] Calibration complete!")

if __name__ == "__main__":
    main()

