# MONO calibration (pinhole/rational model)
# - Uses cv2.calibrateCamera with CALIB_RATIONAL_MODEL for full Brown-Conrady
# - Adds basic per-view quality checks and optional outlier pruning
# - Keeps board/object definitions identical to the original code

import os, sys, time, json, queue, cv2, argparse, re
import glob
import numpy as np
try:
    import requests, pickle
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# ---- app imports / fallback ----
try:
    from .. import config
    from ..capture import CamReader
    from ..detect import CalibrationAccumulator
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config
    from apps.capture import CamReader
    from apps.detect import CalibrationAccumulator

# ====== TUNABLES ======
MIN_TAGS_PER_VIEW = 6
MIN_SPAN = 0.35  # min coverage fraction along either axis
WINDOW_SECONDS = 60.0
TARGET_KEYFRAMES = 70
ACCEPT_PERIOD_S = WINDOW_SECONDS / float(TARGET_KEYFRAMES)

# Optional outlier-rejection (view level)
MAX_VIEW_RMS_PX = 5.0  # if > 0, drop views above this RMS and recalibrate once

# ---------- helpers ----------
_APRIL_LOCAL_PICKLE = None  # optional local override set in main() or via env
_BOARD_SOURCE = "harvard"   # 'harvard' or 'grid8x5' (set in main())
_HARVARD_TAG_SIZE_M = None  # optional tag size override (meters)
_HARVARD_TAG_SPACING_M = None  # optional tag spacing (meters; informational)

def _make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
    # Board selection
    if _BOARD_SOURCE == "grid8x5":
        # Construct 8x5 GridBoard, 0 at bottom-left, 39 at top-right, tag size 0.075m, sep 0.01875m
        TAGS_X = 8
        TAGS_Y = 5
        TAG_SIZE_M = 0.075
        TAG_SEP_M = 0.01875
        ids_grid = np.arange(TAGS_X * TAGS_Y, dtype=np.int32).reshape(TAGS_Y, TAGS_X)
        ids_grid = np.flipud(ids_grid)  # 0 at bottom row, increasing upwards
        ids = ids_grid.reshape(-1, 1).astype(np.int32)
        print(f"[BOARD] Using GridBoard 8x5 (0 bottom-left -> 39 top-right), tag={TAG_SIZE_M}m sep={TAG_SEP_M}m")
        return cv2.aruco.GridBoard((TAGS_X, TAGS_Y), TAG_SIZE_M, TAG_SEP_M, dictionary, ids)
    # Load Harvard CS283 board (required if selected)
    board = _load_harvard_coarse_board(dictionary)
    if board is None:
        if not HAVE_REQUESTS:
            raise RuntimeError("[BOARD] Harvard CS283 board required but 'requests' is not available. Please install requests or enable internet.")
        raise RuntimeError("[BOARD] Failed to load Harvard CS283 at_board_d from GitHub. Check your internet connection.")
    print("[BOARD] Using Harvard CS283 at_board_d (required)")
    return board

def _load_harvard_coarse_board(dictionary):
    # 1) Local override via CLI/env
    local_override = _APRIL_LOCAL_PICKLE or os.environ.get("APRIL_BOARDS_PICKLE", "").strip()
    if local_override:
        try:
            path = os.path.abspath(local_override)
            with open(path, "rb") as f:
                data = pickle.loads(f.read())
            tag_size_m = _harvard_tag_size_from_data_or_env(data)
            board = _parse_board_pickle(dictionary, data, tag_size_m=tag_size_m)
            if board is not None:
                print(f"[BOARD] Loaded Harvard board from local file: {path}")
                return board
            else:
                keys = list(data.keys()) if isinstance(data, dict) else "N/A"
                print(f"[BOARD] Local pickle parsed but no usable board found. Available keys: {keys}")
        except Exception as e:
            print(f"[BOARD] Failed to read local pickle '{local_override}': {e}")
        # If a local override was provided, do NOT attempt network fallback
        return None
    # 2) Remote fetch (with retries and alternate raw URL)
    if not HAVE_REQUESTS:
        return None
    urls = [
        "https://github.com/Harvard-CS283/pset-data/raw/f1a90573ae88cd530a3df3cd0cea71aa2363b1b3/april/AprilBoards.pickle",
        "https://raw.githubusercontent.com/Harvard-CS283/pset-data/f1a90573ae88cd530a3df3cd0cea71aa2363b1b3/april/AprilBoards.pickle",
    ]
    headers = {"User-Agent": "GolfCoach/1.0 (+requests)"},
    for url in urls:
        for attempt in range(2):
            try:
                resp = requests.get(url, timeout=15, allow_redirects=True, headers={"User-Agent": "GolfCoach/1.0"})
                if resp.status_code != 200:
                    print(f"[BOARD] HTTP {resp.status_code} fetching board from {url}")
                    continue
                data = pickle.loads(resp.content)
                tag_size_m = _harvard_tag_size_from_data_or_env(data)
                board = _parse_board_pickle(dictionary, data, tag_size_m=tag_size_m)
                if board is not None:
                    print(f"[BOARD] Loaded Harvard board from: {url}")
                    return board
                else:
                    print(f"[BOARD] Pickle at {url} missing/invalid 'at_board_d'")
            except Exception as e:
                print(f"[BOARD] Fetch attempt {attempt+1} failed for {url}: {e}")
                time.sleep(0.5)
    return None

def _harvard_tag_size_from_data_or_env(data):
    """Return tag size (meters) from CLI/env or from pickle metadata if present."""
    # CLI/global override
    if _HARVARD_TAG_SIZE_M is not None:
        try:
            return float(_HARVARD_TAG_SIZE_M)
        except Exception:
            pass
    # ENV override
    env = os.environ.get("HARVARD_TAG_SIZE_M", "").strip()
    if env:
        try:
            return float(env)
        except Exception:
            pass
    # Try from pickle metadata
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

def _parse_board_pickle(dictionary, data, tag_size_m=None):
    try:
        # If data itself is already a cv2.aruco.Board-like object, return as-is
        if hasattr(data, "getObjPoints") and (hasattr(data, "ids") or hasattr(data, "getIds")):
            return data
        # Prefer known keys; fall back to first value that looks like a board
        candidate_keys = ["at_board_d", "at_coarseboard", "board", "at_board"]
        board_data = None
        for k in candidate_keys:
            if k in getattr(data, "keys", lambda: [])():
                board_data = data.get(k)
                break
        if board_data is None:
            # try scan for first item that looks like a board or dict with ids/corners
            if isinstance(data, dict):
                for k, v in data.items():
                    if hasattr(v, "getObjPoints") and (hasattr(v, "ids") or hasattr(v, "getIds")):
                        board_data = v; break
                if board_data is None and any(isinstance(v, (dict, list, tuple)) for v in data.values()):
                    # choose first nested dict/list that seems plausible
                    for k, v in data.items():
                        if isinstance(v, dict) and ("ids" in v or "objPoints" in v or "corners" in v):
                            board_data = v; break
                        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict):
                            board_data = v; break
        if board_data is None:
            print(f"[BOARD] Available pickle keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            return None
        # If board_data is already a Board, return it
        if hasattr(board_data, "getObjPoints") and (hasattr(board_data, "ids") or hasattr(board_data, "getIds")):
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
        elif isinstance(board_data, (list, tuple)) and len(board_data) > 0:
            # Pre-collect centers if present to optionally infer unit scale using tag size + spacing
            centers_units = []
            ids_units = []
            for item in board_data:
                if isinstance(item, dict) and 'tag_id' in item and 'center' in item:
                    c = np.array(item['center'], dtype=np.float32).reshape(-1)
                    centers_units.append(c[:2].astype(np.float32))
                    ids_units.append(int(item['tag_id']))
            center_scale = None
            # Only infer scale when BOTH tag size and spacing are provided
            if centers_units and (tag_size_m is not None) and (_HARVARD_TAG_SPACING_M is not None):
                desired_cc_m = float(tag_size_m) + float(_HARVARD_TAG_SPACING_M)
                if desired_cc_m > 0:
                    # Robust nearest neighbor distance in units
                    C = np.vstack(centers_units)  # N x 2
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
                            print(f"[BOARD] Harvard center scale inferred: {center_scale:.6f} m/unit (nn={nn_units:.6f} units, cc={desired_cc_m:.6f} m)")
            for item in board_data:
                if isinstance(item, dict):
                    # Case A: dict with 'id' and 'corners'/'objPoints'
                    if 'id' in item and ('corners' in item or 'objPoints' in item):
                        tid = int(item['id'])
                        pts = item.get('objPoints', item.get('corners'))
                        P = np.array(pts, dtype=np.float32).reshape(-1, 3 if np.array(pts).shape[-1] == 3 else 2)
                        if P.shape[1] == 2:
                            P = np.hstack([P, np.zeros((P.shape[0], 1), dtype=np.float32)])
                        if P.shape[0] == 4:
                            obj_points.append(P.reshape(4, 3))
                            ids_list.append(tid)
                        continue
                    # Case B: dict with 'tag_id' and 'center' (Harvard format)
                    if 'tag_id' in item and 'center' in item:
                        if tag_size_m is None:
                            raise RuntimeError("[BOARD] Harvard centers found but tag size unknown. Provide --harvard-tag-size-m or HARVARD_TAG_SIZE_M env.")
                        tid = int(item['tag_id'])
                        c = np.array(item['center'], dtype=np.float32).reshape(-1)
                        if center_scale is not None:
                            c = c * float(center_scale)
                        # center can be (x,y) or (x,y,z); assume z=0 if 2D
                        if c.size == 2:
                            cx, cy = float(c[0]), float(c[1]); cz = 0.0
                        else:
                            cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
                        half = float(tag_size_m) * 0.5
                        # Construct square corners around center in board plane
                        pts3 = np.array([
                            [cx - half, cy - half, cz],
                            [cx + half, cy - half, cz],
                            [cx + half, cy + half, cz],
                            [cx - half, cy + half, cz],
                        ], dtype=np.float32)
                        obj_points.append(pts3.reshape(4, 3))
                        ids_list.append(tid)
                        continue
                # Unsupported element
        else:
            return None
        if not obj_points or not ids_list:
            return None
        ids_arr = np.array(ids_list, dtype=np.int32).reshape(-1, 1)
        try:
            board = cv2.aruco.Board(obj_points, dictionary, ids_arr)
        except Exception:
            try:
                board = cv2.aruco.Board_create(obj_points, dictionary, ids_arr)
            except Exception:
                board = None
        return board
    except Exception:
        try:
            print(f"[BOARD] Failed to parse board; available keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        except Exception:
            pass
        return None

def _draw_status(img, lines, y0=28, dy=24, color=(0,255,255)):
    for k, line in enumerate(lines):
        cv2.putText(img, line, (14, y0 + dy*k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def _has_coverage(corners, W, H, min_span=MIN_SPAN):
    xs = [p[0] for ci in corners for p in ci.reshape(4,2)]
    ys = [p[1] for ci in corners for p in ci.reshape(4,2)]
    if not xs: return False
    span_x = (max(xs)-min(xs))/float(max(1.0,W))
    span_y = (max(ys)-min(ys))/float(max(1.0,H))
    # Require both axes coverage (avoid thin hulls)
    return min(span_x, span_y) >= min_span

def _radial_spread_ok(img_pts, K_like, W, H,
                      min_inner=0.25, min_outer=0.85,
                      min_frac_inner=0.10, min_frac_outer=0.10):
    P = img_pts.reshape(-1,2).astype(np.float32)
    cx, cy = float(K_like[0,2]), float(K_like[1,2])
    r  = np.hypot(P[:,0]-cx, P[:,1]-cy)
    rmax = np.hypot(max(cx, W-cx), max(cy, H-cy))
    if rmax <= 0:
        return False
    frac_inner = float(np.mean(r <= min_inner*rmax))
    frac_outer = float(np.mean(r >= min_outer*rmax))
    return (frac_inner >= min_frac_inner) and (frac_outer >= min_frac_outer)

def _best_kept_index(acc, kept_idxs):
    if not kept_idxs: return 0
    counts = [0 if acc.ids0[i] is None else len(acc.ids0[i]) for i in kept_idxs]
    return int(np.argmax(counts))

# ---------- debug helper ----------
def _print_intrinsics(K, D):
    try:
        fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    except Exception:
        fx = fy = cx = cy = float("nan")
    dlen = (int(D.size) if D is not None else 0)
    print(f"[INTR] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  D_len={dlen}")

# ---------- simple K seed for pinhole ----------
def seed_K_pinhole(W, H, f_scale=1.0):
    """Seed intrinsics roughly at focal ~ f_scale * max(W,H), principal at center."""
    f = float(max(W, H)) * float(f_scale)
    K = np.eye(3, dtype=np.float64)
    K[0,0], K[1,1] = f, f
    K[0,2], K[1,2] = W*0.5, H*0.5
    return K

# ---------- pinhole (rational) calibrate ----------
def calibrate_pinhole_full(obj_list, img_list, image_size, K_seed=None):
    # OpenCV expects lists of (N,1,3)/(N,1,2) with multi-channel points
    # For calibrateCamera(), use Point3f/Point2f equivalents: float32 with shapes (N,3) and (N,2)
    obj_std = [o.reshape(-1,3).astype(np.float32, copy=False) for o in obj_list]
    img_std = [i.reshape(-1,2).astype(np.float32, copy=False) for i in img_list]
    if K_seed is not None:
        K = K_seed.copy().astype(np.float64)
        # Allocate up to 8 coeffs for rational model; OpenCV will resize as needed
        D = np.zeros((8,1), dtype=np.float64)
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
                 cv2.CALIB_RATIONAL_MODEL)
    else:
        K = None
        D = None
        flags = cv2.CALIB_RATIONAL_MODEL
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_std, img_std, image_size, K, D, flags=flags, criteria=crit
    )
    return float(rms), K, D, rvecs, tvecs

# ---------- diagnostics ----------
def _save_diag_pinhole(path_png, frame_bgr, corners, ids, acc, K, D, rvec, tvec,
                       r_cyan=8, r_mag=3, draw_corners=True, draw_proj_corners=True):
    img = frame_bgr.copy()
    # Optional: accumulate for per-view RMS
    obj_all = []
    img_all = []
    for c, iv in zip(corners, ids):
        tid = int(iv[0])
        if tid not in acc.id_to_obj: continue
        obj4 = acc.id_to_obj[tid].astype(np.float64).reshape(-1,1,3)  # 4x1x3
        proj,_ = cv2.projectPoints(obj4, rvec, tvec, K, D)            # 4x1x2
        det_corners = c.reshape(4,2).astype(np.float32)               # 4x2
        proj_corners = proj.reshape(-1,2)                             # 4x2
        det_center = det_corners.mean(axis=0)
        proj_center = proj_corners.mean(axis=0)
        cv2.circle(img, tuple(np.int32(det_center)), int(r_cyan), (255,255,0), -1)  # cyan (detected center)
        # projected center in black for contrast
        cv2.circle(img, tuple(np.int32(proj_center)), max(3, int(r_mag+1)), (0,0,0), -1)  # black (projected center)
        if draw_corners:
            p_ctr = tuple(np.int32(proj_center))
            for k in range(4):
                p_det = tuple(np.int32(det_corners[k]))
                # detected corner (cyan)
                cv2.circle(img, p_det, max(3, int(r_cyan*0.70)), (255,255,0), -1)
                if draw_proj_corners:
                    p_prj = tuple(np.int32(proj_corners[k]))
                    # connector line from projected center to projected corner
                    cv2.line(img, p_ctr, p_prj, (0,200,255), 2)
                    # projected corner (magenta)
                    cv2.circle(img, p_prj, max(4, int(r_mag+2)), (255,0,255), -1)
        # Accumulate for RMS
        obj_all.append(acc.id_to_obj[tid].reshape(-1,3))
        img_all.append(det_corners.reshape(-1,2))
    # Per-view RMS annotation (in pixels)
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
    cv2.putText(img, legend,
                (16, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.imwrite(path_png, img)

# ---------- view-level RMS ----------
def _view_rms_pinhole(obj_pts, img_pts, K, D, rvec, tvec):
    proj,_ = cv2.projectPoints(obj_pts.reshape(-1,1,3).astype(np.float32), rvec, tvec, K, D)
    proj = proj.reshape(-1,2)
    err = img_pts.reshape(-1,2).astype(np.float32) - proj
    return float(np.sqrt(np.mean(np.sum(err*err, axis=1))))

# ---------- main ----------
def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-index","-i", type=int, default=None, help="Camera index for live capture")
    parser.add_argument("--images","-d", type=str, default=None, help="Directory of images for offline calibration")
    parser.add_argument("--pattern","-p", type=str, default="*.png", help="Glob pattern (e.g., *.png, *.jpg)")
    parser.add_argument("--april-pickle", type=str, default=None, help="Path to local AprilBoards.pickle (overrides network)")
    parser.add_argument("--save-all-diag", action="store_true", help="Save diagnostic overlays for all kept frames")
    parser.add_argument("--board-source", type=str, choices=["harvard","grid8x5"], default="harvard",
                        help="Which board to use: Harvard pickle or original 8x5 grid")
    parser.add_argument("--harvard-tag-size-m", type=float, default=None, help="Tag side length in meters for Harvard board (overrides pickle/env)")
    parser.add_argument("--harvard-tag-spacing-m", type=float, default=None, help="Tag spacing (meters) for Harvard board (informational)")
    parser.add_argument("--corner-order", type=str, default=None,
                        help="Manual corner order override as four comma-separated indices, e.g. '0,1,2,3'. Applies to all detections. If provided, auto corner reordering is disabled.")
    args,_ = parser.parse_known_args()

    # camera selection helper
    def _resolve_camera_index():
        if args.cam_index is not None: return int(args.cam_index)
        env = os.environ.get("CAM_INDEX_ORDER","\n").strip()
        if env:
            try: idxs=[int(x) for x in env.split(",") if x.strip()!=""]
            except: idxs=[0]
            return idxs[0] if idxs else 0
        return 0

    # SDK will be loaded lazily only in live mode (see below)

    # Optional local pickle override from CLI
    global _APRIL_LOCAL_PICKLE
    if args.april_pickle:
        _APRIL_LOCAL_PICKLE = args.april_pickle
    # Optional Harvard tag size override
    global _HARVARD_TAG_SIZE_M
    if args.harvard_tag_size_m is not None:
        _HARVARD_TAG_SIZE_M = float(args.harvard_tag_size_m)
    # Optional Harvard tag spacing (informational)
    global _HARVARD_TAG_SPACING_M
    if args.harvard_tag_spacing_m is not None:
        _HARVARD_TAG_SPACING_M = float(args.harvard_tag_spacing_m)
    # Board source selection
    global _BOARD_SOURCE
    _BOARD_SOURCE = str(args.board_source or "harvard").lower().strip()

    # Offline images mode
    if args.images:
        img_dir = os.path.abspath(args.images)
        pattern = args.pattern or "*.png"
        glob_path = os.path.join(img_dir, pattern)
        recursive = ("**" in pattern)
        paths = sorted(glob.glob(glob_path, recursive=recursive))
        if not paths:
            print(f"[MAIN] No images matched: {img_dir} with pattern {args.pattern}")
            return
        print(f"[MAIN] Offline mode: found {len(paths)} images (pattern={pattern}, recursive={recursive})")
        print("[MAIN] Camera/SDK not required in offline mode.")
        # read first to get size
        frame0 = None
        for p in paths:
            frame0 = cv2.imread(p, cv2.IMREAD_COLOR)
            if frame0 is not None:
                break
        if frame0 is None:
            print("[MAIN] Failed to read any image")
            return
        H, W = frame0.shape[:2]; image_size=(W,H)
        try:
            board = _make_board()
        except Exception as e:
            print(str(e))
            return
        # Parse optional corner order override
        corner_order_override = None
        disable_autoreorder = False
        if args.corner_order:
            try:
                parts = [int(x.strip()) for x in args.corner_order.split(",")]
                if len(parts) == 4 and sorted(parts) == [0,1,2,3]:
                    corner_order = parts
                    corner_order_override = corner_order
                    disable_autoreorder = True
                else:
                    print(f"[WARN] Ignoring invalid --corner-order '{args.corner_order}'. Expected 4 comma-separated indices 0..3.")
            except Exception as _e:
                print(f"[WARN] Failed to parse --corner-order '{args.corner_order}': {_e}. Ignoring.")
        acc = CalibrationAccumulator(board, image_size,
                                     corner_order_override=corner_order_override,
                                     disable_corner_autoreorder=disable_autoreorder)
        print("[APRIL] Backend:", acc.get_backend_name())
        print("[APRIL] Families:", acc._apriltag_family_string())

        kept_frames = []
        for p in paths:
            frame = cv2.imread(p, cv2.IMREAD_COLOR)
            if frame is None: continue
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try: corners, ids = acc.detect(gray)
            except Exception: corners, ids = [], None
            n = 0 if ids is None else len(ids)
            if n >= MIN_TAGS_PER_VIEW and corners and _has_coverage(corners, W,H, MIN_SPAN):
                if acc._accumulate_single(0, corners, ids):
                    kept_frames.append(frame.copy())
        print(f"[CAL] Collected views: {len(acc.corners0)}")

        # Run calibration using same logic as live flow
        if len(acc.corners0) == 0:
            print("[CAL] Not enough valid samples; aborting.")
            return

        K_seed = seed_K_pinhole(W, H, f_scale=1.0)
        obj_list,img_list,kept=[],[],[]
        drop_no_ids=0; drop_few_tags=0; drop_coverage=0; drop_no_map=0
        for vi,(corners_img,ids_img) in enumerate(zip(acc.corners0, acc.ids0)):
            if ids_img is None:
                drop_no_ids += 1
                continue
            if len(ids_img) < MIN_TAGS_PER_VIEW:
                drop_few_tags += 1
                continue
            if not _has_coverage(corners_img, W,H, MIN_SPAN):
                drop_coverage += 1
                continue
            O,I=[],[]
            for c,iv in zip(corners_img, ids_img):
                tid=int(iv[0])
                if tid not in acc.id_to_obj: continue
                O.append(acc.id_to_obj[tid])
                I.append(c.reshape(-1,2))
            if O:
                obj_cat = np.concatenate(O,0).astype(np.float64).reshape(-1,1,3)
                img_cat = np.concatenate(I,0).astype(np.float64).reshape(-1,1,2)
                obj_list.append(obj_cat)
                img_list.append(img_cat)
                kept.append(vi)
            else:
                drop_no_map += 1

        if (drop_no_ids+drop_few_tags+drop_coverage+drop_no_map) > 0:
            print(f"[CAL] View filter summary: kept={len(kept)}  no_ids={drop_no_ids}  few_tags={drop_few_tags}  coverage_fail={drop_coverage}  no_map={drop_no_map}")

        if not obj_list:
            print("[CAL] Not enough valid samples after filtering; aborting calibration.")
            return

        rms_val, K, D, rvecs, tvecs = calibrate_pinhole_full(obj_list, img_list, image_size, K_seed)
        print(f"[CAL] Pinhole RMS: {rms_val:.3f} (D has {D.size if D is not None else 0} coeffs)")
        _print_intrinsics(K, D)

        # diagnostics save
        best_local = _best_kept_index(acc, kept)
        kf_idx = kept[best_local]
        best_corners = acc.corners0[kf_idx]
        best_ids     = acc.ids0[kf_idx]
        best_bgr     = kept_frames[kf_idx] if kf_idx < len(kept_frames) else frame0

        repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
        out_dir = os.path.join(repo_root, "data"); os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_png = os.path.join(out_dir, f"mono_calib_diag_{stamp}.png")
        _save_diag_pinhole(out_png, best_bgr, best_corners, best_ids, acc, K, D, rvecs[best_local], tvecs[best_local], r_cyan=8, r_mag=3)
        print(f"[SAVE] Diagnostic image -> {out_png}")

        # Optionally save overlays for all kept frames
        if args.save_all_diag:
            for idx, (vi, O, I) in enumerate(zip(kept, obj_list, img_list)):
                kf_idx = vi
                corners = acc.corners0[kf_idx]
                ids_here = acc.ids0[kf_idx]
                bgr = kept_frames[kf_idx] if kf_idx < len(kept_frames) else frame0
                out_png_i = os.path.join(out_dir, f"mono_calib_diag_{stamp}_view{idx:03d}.png")
                _save_diag_pinhole(out_png_i, bgr, corners, ids_here, acc, K, D, rvecs[idx], tvecs[idx], r_cyan=8, r_mag=3)
            print(f"[SAVE] Saved {len(kept)} diagnostic overlays to {out_dir}")

        out_json = os.path.join(repo_root, "data", "mono_calibration_latest.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "image_size": [W,H],
                "K": K.tolist(),
                "D": D.tolist(),
                "rms": float(rms_val),
                "num_keyframes": int(len(acc.corners0)),
                "model": "pinhole_rational"
            }, f, indent=2)
        print(f"[SAVE] Wrote {out_json}")
        return

    # Live camera mode
    print("[MAIN] Opening one camera…")
    # Lazy import SDK only for live mode to avoid DLL logs in offline mode
    _try_load_dll = None
    _sdk_config = None
    try:
        try:
            from ..sdk import try_load_dll as _try_load_dll, sdk_config as _sdk_config
        except Exception:
            try:
                from apps.sdk import try_load_dll as _try_load_dll, sdk_config as _sdk_config
            except Exception:
                _try_load_dll = None
                _sdk_config = None
        if _try_load_dll is not None and _sdk_config is not None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dll, has_af, cam_ids = _try_load_dll(base_dir)
            if cam_ids:
                for iid in cam_ids[:1]:
                    _sdk_config(dll, has_af, iid, fps=config.CAPTURE_FPS, lock_autos=True, anti_flicker_60hz=True,
                                exposure_us=(8000 if config.USE_SDK_EXPOSURE else None))
                time.sleep(0.3)
    except Exception:
        pass
    cam = CamReader(_resolve_camera_index())

    # frame size
    _, frame0 = cam.latest()
    H, W = frame0.shape[:2]; image_size=(W,H)

    # detector
    try:
        board = _make_board()
    except Exception as e:
        print(str(e))
        cam.release()
        cv2.destroyAllWindows()
        return
    # Parse optional corner order override
    corner_order_override = None
    disable_autoreorder = False
    if args.corner_order:
        try:
            parts = [int(x.strip()) for x in args.corner_order.split(",")]
            if len(parts) == 4 and sorted(parts) == [0,1,2,3]:
                corner_order = parts
                corner_order_override = corner_order
                disable_autoreorder = True
            else:
                print(f"[WARN] Ignoring invalid --corner-order '{args.corner_order}'. Expected 4 comma-separated indices 0..3.")
        except Exception as _e:
            print(f"[WARN] Failed to parse --corner-order '{args.corner_order}': {_e}. Ignoring.")
    acc = CalibrationAccumulator(board, image_size,
                                 corner_order_override=corner_order_override,
                                 disable_corner_autoreorder=disable_autoreorder)
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())

    # UI/state
    calibrating=False; deadline=0.0; last_sample=0.0; last_accept=0.0; keyframes=[]
    record_dir = None
    win="Single Cam Calibrator (PINHOLE)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            try:
                ts, frame = cam.latest()
            except queue.Empty:
                time.sleep(0.01); continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            annotated = frame.copy()

            # run calibration when window ends
            if calibrating and time.perf_counter() >= deadline:
                calibrating=False
                print(f"[CAL] Window complete. Collected {len(keyframes)} keyframes.")

                # build per-view lists & filter weak views
                # Seed K for pinhole (optional)
                K_seed = seed_K_pinhole(W, H, f_scale=1.0)
                obj_list,img_list,kept=[],[],[]
                # Diagnostics: filter reasons
                drop_no_ids=0; drop_few_tags=0; drop_coverage=0; drop_no_map=0
                for vi,(corners_img,ids_img) in enumerate(zip(acc.corners0, acc.ids0)):
                    if ids_img is None:
                        drop_no_ids += 1
                        continue
                    if len(ids_img) < MIN_TAGS_PER_VIEW:
                        drop_few_tags += 1
                        continue
                    if not _has_coverage(corners_img, W,H, MIN_SPAN):
                        drop_coverage += 1
                        continue
                    O,I=[],[]
                    for c,iv in zip(corners_img, ids_img):
                        tid=int(iv[0])
                        if tid not in acc.id_to_obj: continue
                        O.append(acc.id_to_obj[tid])               # 4x3 object points
                        I.append(c.reshape(-1,2))                   # 4x2 image points (SAME ORDER)
                    if O:
                        obj_cat = np.concatenate(O,0).astype(np.float64).reshape(-1,1,3)
                        img_cat = np.concatenate(I,0).astype(np.float64).reshape(-1,1,2)
                        obj_list.append(obj_cat)
                        img_list.append(img_cat)
                        kept.append(vi)
                    else:
                        drop_no_map += 1

                if (drop_no_ids+drop_few_tags+drop_coverage+drop_no_map) > 0:
                    print(f"[CAL] View filter summary: kept={len(kept)}  no_ids={drop_no_ids}  few_tags={drop_few_tags}  coverage_fail={drop_coverage}  no_map={drop_no_map}")

                if not obj_list:
                    print("[CAL] Not enough valid samples after filtering; aborting calibration.")
                    keyframes.clear(); acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()
                    continue

                # Calibrate (pinhole + rational)
                rms_val, K, D, rvecs, tvecs = calibrate_pinhole_full(obj_list, img_list, image_size, K_seed)
                print(f"[CAL] Pinhole RMS: {rms_val:.3f} (D has {D.size if D is not None else 0} coeffs)")
                _print_intrinsics(K, D)

                # Optional one-shot view pruning and recalibration
                if MAX_VIEW_RMS_PX > 0:
                    keep_mask = []
                    for vi,(O,I) in enumerate(zip(obj_list, img_list)):
                        rv, tv = rvecs[vi], tvecs[vi]
                        view_rms = _view_rms_pinhole(O, I, K, D, rv, tv)
                        keep_mask.append(view_rms <= MAX_VIEW_RMS_PX)
                    if any(not m for m in keep_mask) and sum(keep_mask) >= 5:
                        obj_list2 = [o for o,m in zip(obj_list, keep_mask) if m]
                        img_list2 = [i for i,m in zip(img_list, keep_mask) if m]
                        print(f"[CAL] Dropped {len(obj_list)-len(obj_list2)} high-RMS views (> {MAX_VIEW_RMS_PX}px). Recalibrating…")
                        rms_val, K, D, rvecs, tvecs = calibrate_pinhole_full(obj_list2, img_list2, image_size, K)
                        print(f"[CAL] Pinhole RMS (after prune): {rms_val:.3f}")
                        _print_intrinsics(K, D)
                        # Map back to original kept indices for diagnostics
                        kept = [k for k,m in zip(kept, keep_mask) if m]
                        obj_list, img_list = obj_list2, img_list2

                # Diagnostic + save
                if K is not None:
                    best_local = _best_kept_index(acc, kept)
                    kf_idx = kept[best_local]
                    best_corners = acc.corners0[kf_idx]
                    best_ids     = acc.ids0[kf_idx]
                    _, _, _, best_bgr = keyframes[kf_idx]

                    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
                    out_dir = os.path.join(repo_root, "data"); os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    out_png = os.path.join(out_dir, f"mono_calib_diag_{stamp}.png")
                    _save_diag_pinhole(out_png, best_bgr, best_corners, best_ids, acc, K, D, rvecs[best_local], tvecs[best_local], r_cyan=8, r_mag=3)
                    print(f"[SAVE] Diagnostic image -> {out_png}")

                    # Optionally save overlays for all kept frames
                    if args.save_all_diag:
                        for idx, vi in enumerate(kept):
                            kf_idx = vi
                            corners = acc.corners0[kf_idx]
                            ids_here = acc.ids0[kf_idx]
                            _, _, _, bgr = keyframes[kf_idx]
                            out_png_i = os.path.join(out_dir, f"mono_calib_diag_{stamp}_view{idx:03d}.png")
                            _save_diag_pinhole(out_png_i, bgr, corners, ids_here, acc, K, D, rvecs[idx], tvecs[idx], r_cyan=8, r_mag=3)
                        print(f"[SAVE] Saved {len(kept)} diagnostic overlays to {out_dir}")

                    out_json = os.path.join(repo_root, "data", "mono_calibration_latest.json")
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump({
                            "image_size": [W,H],
                            "K": K.tolist(),
                            "D": D.tolist(),   # k1..k6 (rational adds k4..k6), p1, p2 if present
                            "rms": float(rms_val),
                            "num_keyframes": int(len(keyframes)),
                            "model": "pinhole_rational"
                        }, f, indent=2)
                    print(f"[SAVE] Wrote {out_json}")

                # reset buffers
                keyframes.clear(); acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()

            # sampling gate
            if calibrating and (time.perf_counter()-last_sample) >= 0.5:
                now = time.perf_counter()
                last_sample = now
                try: corners, ids = acc.detect(gray)
                except Exception: corners, ids = [], None
                n = 0 if ids is None else len(ids)
                if (n >= MIN_TAGS_PER_VIEW and corners and _has_coverage(corners, W,H, MIN_SPAN)
                        and (now - last_accept) >= ACCEPT_PERIOD_S and len(keyframes) < TARGET_KEYFRAMES):
                    if acc._accumulate_single(0, corners, ids):
                        keyframes.append((gray.copy(), [c.copy() for c in corners], ids.copy(), frame.copy()))
                        last_accept = now
                        print(f"[CAL] Keyframes: {len(keyframes)} (ids={n})")
                        # Save recording artifacts if enabled
                        if record_dir:
                            idx = len(keyframes) - 1
                            try:
                                out_img = os.path.join(record_dir, f"frame_{idx:03d}.png")
                                out_js  = os.path.join(record_dir, f"frame_{idx:03d}.json")
                                cv2.imwrite(out_img, frame)
                                ids_list = [int(iv[0]) for iv in ids]
                                corners_list = [c.reshape(4,2).tolist() for c in corners]
                                with open(out_js, "w", encoding="utf-8") as f:
                                    json.dump({
                                        "ids": ids_list,
                                        "corners": corners_list,
                                        "size": [W, H]
                                    }, f, indent=2)
                            except Exception as e:
                                print("[REC] Failed to write keyframe:", e)
                try: cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
                except Exception: pass

            # UI
            status=[]
            if calibrating:
                status.append(f"Calibrating… keyframes={len(keyframes)}  remain={int(max(0.0,deadline-time.perf_counter()))}s")
            status.append(f"Detector: {acc.get_backend_name()}")
            _draw_status(annotated, status, y0=28)
            cv2.imshow(win, annotated)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')): break
            if k == ord('c'):
                calibrating = not calibrating
                print(f"[KEY] Calibrating -> {calibrating}")
                if calibrating:
                    deadline = time.perf_counter() + WINDOW_SECONDS
                    keyframes.clear(); acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()
                    last_sample = 0.0; last_accept = 0.0
                    # Start a recording folder under data/
                    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
                    data_dir = os.path.join(repo_root, "data")
                    os.makedirs(data_dir, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    record_dir = os.path.join(data_dir, f"calibration_{stamp}")
                    os.makedirs(record_dir, exist_ok=True)
                    # Save session meta
                    try:
                        meta = {
                            "image_size": [W, H],
                            "APRIL_DICT": int(getattr(config, "APRIL_DICT", 0)),
                            "TAGS_X": int(getattr(config, "TAGS_X", 0)),
                            "TAGS_Y": int(getattr(config, "TAGS_Y", 0)),
                            "TAG_SIZE_M": float(getattr(config, "TAG_SIZE_M", 0.0)),
                            "TAG_SEP_M": float(getattr(config, "TAG_SEP_M", 0.0)),
                            "model": "pinhole_rational",
                        }
                        with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)
                        print(f"[REC] Recording keyframes to {record_dir}")
                    except Exception as e:
                        print("[REC] Failed to write meta.json:", e)

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[MAIN] Closed")

if __name__ == "__main__":
    main()
