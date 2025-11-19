import os, sys, time, json, argparse, queue
import numpy as np
import cv2

# ---- local imports ----
try:
    from . import config
    from .capture import CamReader
    from .detect import CalibrationAccumulator, board_ids_safe
except Exception:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from apps import config
    from apps.capture import CamReader
    from apps.detect import CalibrationAccumulator, board_ids_safe

try:
    import requests, pickle  # optional (only if using Harvard board via URL)
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

from .boards import harvard_tag_size_from_data_or_env, parse_board_pickle


def load_board(board_source="harvard", april_pickle=None, harvard_tag_size_m=None, harvard_tag_spacing_m=None):
    """Return an OpenCV aruco Board based on selection: 'harvard' or 'grid8x5'."""
    dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
    if str(board_source).lower().strip() == "grid8x5":
        tags_x, tags_y = 8, 5
        tag_size_m = 0.075
        tag_sep_m = 0.01875
        ids_grid = np.arange(tags_x * tags_y, dtype=np.int32).reshape(tags_y, tags_x)
        ids_grid = np.flipud(ids_grid)  # 0 bottom row -> 39 top row
        ids = ids_grid.reshape(-1, 1).astype(np.int32)
        print(f"[BOARD] GridBoard 8x5, tag={tag_size_m}m sep={tag_sep_m}m")
        return cv2.aruco.GridBoard((tags_x, tags_y), tag_size_m, tag_sep_m, dictionary, ids)

    # Harvard board: prefer local pickle, else env override, else attempt URL (use shared parser)
    data = None
    if april_pickle:
        path = os.path.abspath(april_pickle)
        with open(path, "rb") as f:
            data = pickle.loads(f.read())
    elif os.environ.get("APRIL_BOARDS_PICKLE", "").strip():
        path = os.path.abspath(os.environ["APRIL_BOARDS_PICKLE"])
        with open(path, "rb") as f:
            data = pickle.loads(f.read())
    else:
        if not HAVE_REQUESTS:
            raise RuntimeError("[BOARD] Harvard board requires local pickle or requests installed.")
        urls = [
            "https://github.com/Harvard-CS283/pset-data/raw/f1a90573ae88cd530a3df3cd0cea71aa2363b1b3/april/AprilBoards.pickle",
            "https://raw.githubusercontent.com/Harvard-CS283/pset-data/f1a90573ae88cd530a3df3cd0cea71aa2363b1b3/april/AprilBoards.pickle",
        ]
        last_err = None
        for url in urls:
            try:
                r = requests.get(url, timeout=15, headers={"User-Agent": "GolfCoach/1.0"})
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    continue
                data = pickle.loads(r.content)
                break
            except Exception as e:
                last_err = str(e)
        if data is None:
            raise RuntimeError(f"[BOARD] Failed to fetch Harvard board: {last_err}")

    # Use shared parser for identical behavior with mono script
    size_m = harvard_tag_size_from_data_or_env(data, cli_size=harvard_tag_size_m)
    board = parse_board_pickle(dictionary, data, tag_size_m=size_m, harvard_tag_spacing_m=harvard_tag_spacing_m)
    if board is None:
        raise RuntimeError("[BOARD] Harvard board found but format was not recognized.")
    print("[BOARD] Harvard board loaded")
    return board


def solve_pnp_for_view(K, D, obj_pts_4x3_list, img_pts_4x2_list):
    """Estimate pose (rvec,tvec) for a single view from matched tag corners (lists)."""
    if not obj_pts_4x3_list or not img_pts_4x2_list:
        return None, None
    O = np.concatenate(obj_pts_4x3_list, axis=0).astype(np.float32).reshape(-1, 3)
    I = np.concatenate(img_pts_4x2_list, axis=0).astype(np.float32).reshape(-1, 2)
    # Use IPPE_SQUARE only when we have exactly one square (4 points)
    try:
        if O.shape[0] == 4:
            ok, rvec, tvec = cv2.solvePnP(O, I, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        else:
            ok, rvec, tvec = cv2.solvePnP(O, I, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        ok = False; rvec = None; tvec = None
    if not ok:
        try:
            ok, rvec, tvec = cv2.solvePnP(O, I, K, D, flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            ok = False; rvec = None; tvec = None
    return (rvec, tvec) if ok else (None, None)


def camera_pose_in_board(rvec, tvec, axis_len=0.2):
    """Return camera origin and axes in board frame from rvec,tvec (object->camera).
    axis_len is the drawn length of each camera axis (meters).
    """
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    # Transform from camera to board: X_b = R^T (X_c - t)
    origin = (-R.T @ t).reshape(3)
    x_axis = (R.T @ np.array([[axis_len, 0, 0]]).T).reshape(3) + origin
    y_axis = (R.T @ np.array([[0, axis_len, 0]]).T).reshape(3) + origin
    z_axis = (R.T @ np.array([[0, 0, axis_len]]).T).reshape(3) + origin
    return origin, x_axis, y_axis, z_axis


# ---------- debug helpers (copied from mono script) ----------
def _print_intrinsics(K, D):
    try:
        fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    except Exception:
        fx = fy = cx = cy = float("nan")
    dlen = (int(D.size) if D is not None else 0)
    print(f"[INTR] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  D_len={dlen}")

def calibrate_pinhole_full(obj_list, img_list, image_size, K_seed=None):
    # obj_list: list of (N,3), img_list: list of (N,2)
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
    # Accumulate for per-view RMS
    obj_all = []
    img_all = []
    for c, iv in zip(corners, ids):
        tid = int(iv[0])
        if tid not in acc.id_to_obj: 
            continue
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
    cv2.putText(img, legend, (16, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.imwrite(path_png, img)

def _view_rms_pinhole(obj_pts, img_pts, K, D, rvec, tvec):
    proj,_ = cv2.projectPoints(obj_pts.reshape(-1,1,3).astype(np.float32), rvec, tvec, K, D)
    proj = proj.reshape(-1,2)
    err = img_pts.reshape(-1,2).astype(np.float32) - proj
    return float(np.sqrt(np.mean(np.sum(err*err, axis=1))))

# ---- selection criteria to match mono script ----
MIN_TAGS_PER_VIEW = 6
MIN_SPAN = 0.35
MAX_VIEW_RMS_PX = 5.0

def _has_coverage(corners, W, H, min_span=MIN_SPAN):
    xs = [p[0] for ci in corners for p in ci.reshape(4,2)]
    ys = [p[1] for ci in corners for p in ci.reshape(4,2)]
    if not xs:
        return False
    span_x = (max(xs)-min(xs))/float(max(1.0,W))
    span_y = (max(ys)-min(ys))/float(max(1.0,H))
    return min(span_x, span_y) >= min_span


def plot_3d(board, K0, D0, K1, D1, corners0, ids0, corners1, ids1, acc, frame0_bgr, frame1_bgr, out_html, R_01=None, T_01=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print("[PLOT] Plotly not available:", e)
        return

    # Build per-view correspondences for PnP
    map0 = {int(i[0]): c.reshape(-1, 2) for c, i in zip(corners0, ids0)}
    map1 = {int(i[0]): c.reshape(-1, 2) for c, i in zip(corners1, ids1)}
    common = sorted(set(map0.keys()) & set(map1.keys()))
    obj0, img0, obj1, img1 = [], [], [], []
    for tid in common:
        if tid not in acc.id_to_obj:
            continue
        obj = acc.id_to_obj[tid].reshape(4, 3)
        obj0.append(obj); img0.append(map0[tid].reshape(4, 2))
        obj1.append(obj); img1.append(map1[tid].reshape(4, 2))

    r0, t0 = solve_pnp_for_view(K0, D0, obj0, img0)
    if r0 is None:
        print("[PLOT] solvePnP failed for cam0; skipping plot.")
        return
    # If stereo extrinsics are provided, derive cam1 from cam0+baseline for stability
    if R_01 is not None and T_01 is not None:
        R0, _ = cv2.Rodrigues(r0)
        R1 = R_01 @ R0
        t1 = (R_01 @ t0.reshape(3, 1) + T_01.reshape(3, 1)).reshape(3, 1)
        r1, _ = cv2.Rodrigues(R1)
    else:
        r1, t1 = solve_pnp_for_view(K1, D1, obj1, img1)
        if r1 is None:
            print("[PLOT] solvePnP failed for cam1 and no stereo R,T; skipping plot.")
            return

    # Make axes longer so direction is obvious
    o0, x0, y0, z0 = camera_pose_in_board(r0, t0, axis_len=0.25)
    o1, x1, y1, z1 = camera_pose_in_board(r1, t1, axis_len=0.25)

    # Board points for visualization
    obj_points = board.getObjPoints()
    centers = []
    for pts4 in obj_points:
        P = np.array(pts4, dtype=np.float32).reshape(4, 3)
        centers.append(P.mean(axis=0))
    centers = np.array(centers, dtype=np.float32)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Stereo poses in board frame", "Keyframe (cam0)", "Keyframe (cam1)")
    )

    # Board centers
    fig.add_trace(go.Scatter3d(
        x=centers[:,0], y=centers[:,1], z=centers[:,2],
        mode="markers", marker=dict(size=3, color="gray"),
        name="Board markers"
    ), row=1, col=1)

    # Camera 0 origin marker + label
    fig.add_trace(go.Scatter3d(
        x=[o0[0]], y=[o0[1]], z=[o0[2]], mode="markers+text",
        marker=dict(size=8, color="orange", symbol="circle"),
        text=["cam0"], textposition="top center", name="cam0 origin"
    ), row=1, col=1)
    # Camera 0 axes
    fig.add_trace(go.Scatter3d(x=[o0[0], x0[0]], y=[o0[1], x0[1]], z=[o0[2], x0[2]], mode="lines",
                                line=dict(color="red", width=10), name="cam0 X"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[o0[0], y0[0]], y=[o0[1], y0[1]], z=[o0[2], y0[2]], mode="lines",
                                line=dict(color="green", width=10), name="cam0 Y"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[o0[0], z0[0]], y=[o0[1], z0[1]], z=[o0[2], z0[2]], mode="lines",
                                line=dict(color="blue", width=10), name="cam0 Z"), row=1, col=1)

    # Camera 1 origin marker + label
    fig.add_trace(go.Scatter3d(
        x=[o1[0]], y=[o1[1]], z=[o1[2]], mode="markers+text",
        marker=dict(size=8, color="purple", symbol="diamond"),
        text=["cam1"], textposition="top center", name="cam1 origin"
    ), row=1, col=1)
    # Camera 1 axes (dashed)
    fig.add_trace(go.Scatter3d(x=[o1[0], x1[0]], y=[o1[1], x1[1]], z=[o1[2], x1[2]], mode="lines",
                                line=dict(color="red", width=10, dash="dash"), name="cam1 X"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[o1[0], y1[0]], y=[o1[1], y1[1]], z=[o1[2], y1[2]], mode="lines",
                                line=dict(color="green", width=10, dash="dash"), name="cam1 Y"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[o1[0], z1[0]], y=[o1[1], z1[1]], z=[o1[2], z1[2]], mode="lines",
                                line=dict(color="blue", width=10, dash="dash"), name="cam1 Z"), row=1, col=1)

    # Add images for cam0 and cam1
    h, w = frame0_bgr.shape[:2]
    frame0_rgb = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)
    fig.add_trace(go.Image(z=frame0_rgb), row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2).update_yaxes(showticklabels=False, row=1, col=2)
    if frame1_bgr is not None:
        frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
        fig.add_trace(go.Image(z=frame1_rgb), row=1, col=3)
        fig.update_xaxes(showticklabels=False, row=1, col=3).update_yaxes(showticklabels=False, row=1, col=3)

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            xaxis=dict(showgrid=True, zeroline=True),
            yaxis=dict(showgrid=True, zeroline=True),
            zaxis=dict(showgrid=True, zeroline=True),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0)
    )
    fig.write_html(out_html)
    print(f"[PLOT] Wrote {out_html}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-source", type=str, choices=["harvard", "grid8x5"], default="harvard")
    parser.add_argument("--april-pickle", type=str, default=None)
    parser.add_argument("--harvard-tag-size-m", type=float, default=None, help="Tag side length in meters for Harvard board")
    parser.add_argument("--harvard-tag-spacing-m", type=float, default=None, help="Tag spacing (meters) for Harvard board (informational)")
    parser.add_argument("--target-keyframes", type=int, default=50)
    parser.add_argument("--accept-period", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--cam0", type=int, default=0, help="Camera index for left/first camera")
    parser.add_argument("--cam1", type=int, default=1, help="Camera index for right/second camera")
    parser.add_argument("--mono-max-views", type=int, default=30, help="Max views used for each mono calibration")
    parser.add_argument("--mono-max-iters", type=int, default=60, help="Max iterations for mono calibration solver")
    parser.add_argument("--corner-order", type=str, default=None,
                        help="Manual corner order override as four comma-separated indices, e.g. '0,1,2,3'. "
                             "If provided, auto corner reordering is disabled.")
    parser.add_argument("--save-keyframes", action="store_true",
                        help="If set, save accepted keyframes from both cameras and detections to data/ for later calibration.")
    args, _ = parser.parse_known_args()

    # Open two cameras
    print(f"[MAIN] Opening two cameras… cam0={args.cam0} cam1={args.cam1}")
    cams = [CamReader(int(args.cam0)), CamReader(int(args.cam1))]
    ts0, f0 = cams[0].latest()
    _, f1 = cams[1].latest()
    H, W = f0.shape[:2]
    image_size = (W, H)

    # Build board and accumulator
    board = load_board(board_source=args.board_source, april_pickle=args.april_pickle,
                       harvard_tag_size_m=args.harvard_tag_size_m, harvard_tag_spacing_m=args.harvard_tag_spacing_m)

    # Parse optional corner order override (same behavior as mono script)
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
        # Match mono calibrator default: for Harvard board, prefer 3,0,1,2 if not specified
        if str(args.board_source).lower().strip() == "harvard":
            corner_order_override = [3,0,1,2]
            disable_autoreorder = True
            print("[APRIL] Using default per-tag corner order 3,0,1,2 for Harvard board")

    acc = CalibrationAccumulator(board, image_size,
                                 corner_order_override=corner_order_override,
                                 disable_corner_autoreorder=disable_autoreorder)
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())
    print("[DBG] backend=", acc.get_backend_name(),
          "corner_order_override=", getattr(acc, "corner_order_override", None),
          "disable_corner_autoreorder=", getattr(acc, "disable_corner_autoreorder", None))

    # Collect keyframes
    keyframes = []
    last_accept = 0.0
    print(f"[MAIN] Target keyframes: {args.target_keyframes}")

    # Optional keyframe recorder
    record_dir = None
    if args.save_keyframes:
        try:
            repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
            base_out = os.path.join(repo_root, "data") if not args.out_dir else os.path.abspath(args.out_dir)
            os.makedirs(base_out, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            record_dir = os.path.join(base_out, f"stereo_keyframes_{stamp}")
            os.makedirs(record_dir, exist_ok=True)
            # Write session meta
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
            }
            with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[REC] Saving stereo keyframes to {record_dir}")
        except Exception as e:
            print("[REC] Failed to initialize keyframe recorder:", e)
            record_dir = None
    try:
        while len(keyframes) < args.target_keyframes:
            try:
                ts0, f0 = cams[0].latest()
                ts1, f1 = cams[1].latest()
            except queue.Empty:
                time.sleep(0.01); continue
            g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

            now = time.perf_counter()
            if (now - last_accept) < float(args.accept_period):
                continue

            # Try accumulate stereo sample
            added = acc.accumulate_pair(g0, g1)
            if added:
                keyframes.append((f0.copy(), f1.copy()))
                last_accept = now
                print(f"[CAL] Keyframes: {len(keyframes)}  stereo_samples: {len(acc.stereo_samples)}")
                # Save images and detections if recording
                if record_dir:
                    try:
                        idx = len(keyframes) - 1
                        out_img0 = os.path.join(record_dir, f"frame_{idx:03d}_cam0.png")
                        out_img1 = os.path.join(record_dir, f"frame_{idx:03d}_cam1.png")
                        cv2.imwrite(out_img0, f0)
                        cv2.imwrite(out_img1, f1)
                        # Use last appended detections
                        ids0 = acc.ids0[-1] if len(acc.ids0) > 0 else None
                        ids1 = acc.ids1[-1] if len(acc.ids1) > 0 else None
                        corners0 = acc.corners0[-1] if len(acc.corners0) > 0 else None
                        corners1 = acc.corners1[-1] if len(acc.corners1) > 0 else None
                        out_js = os.path.join(record_dir, f"frame_{idx:03d}.json")
                        with open(out_js, "w", encoding="utf-8") as f:
                            json.dump({
                                "size": [int(W), int(H)],
                                "ids0": ([] if ids0 is None else [int(iv[0]) for iv in ids0]),
                                "ids1": ([] if ids1 is None else [int(iv[0]) for iv in ids1]),
                                "corners0": ([] if corners0 is None else [c.reshape(4,2).tolist() for c in corners0]),
                                "corners1": ([] if corners1 is None else [c.reshape(4,2).tolist() for c in corners1]),
                            }, f, indent=2)
                    except Exception as e:
                        print("[REC] Failed to write keyframe artifacts:", e)
    finally:
        for c in cams:
            c.release()
        print("[MAIN] Capture closed for calibration processing")

    if len(acc.stereo_samples) < 5:
        print("[CAL] Not enough stereo samples collected; aborting.")
        return

    # Mono intrinsics (rational model)
    def seed_K_pinhole(width, height, f_scale=1.0):
        f = float(max(width, height)) * float(f_scale)
        K = np.eye(3, dtype=np.float64)
        K[0,0] = f; K[1,1] = f
        K[0,2] = width * 0.5; K[1,2] = height * 0.5
        return K

    def mono_calibrate(which):
        corners = acc.corners0 if which == 0 else acc.corners1
        ids = acc.ids0 if which == 0 else acc.ids1
        # Debug at start
        try:
            print(f"[DBG] Mono{which}: raw views = {len(corners)}")
            for vi, (corners_img, ids_img) in enumerate(zip(corners, ids)):
                n_ids = 0 if ids_img is None else len(ids_img)
                print(f"[DBG] Mono{which}: view {vi}: ids={n_ids}")
                if ids_img is not None and len(ids_img) > 0:
                    sample = [int(iv[0]) for iv in ids_img[:8]]
                    print(f"[DBG] Mono{which}: view {vi} sample ids: {sample}")
                    try:
                        missing = [tid for tid in sample if tid not in acc.id_to_obj]
                    except Exception:
                        missing = []
                    print(f"[DBG] Mono{which}: sample missing in id_to_obj: {missing}")
                    if which == 0 and vi == 0:
                        try:
                            print("[DBG] id_to_obj keys sample:", list(acc.id_to_obj.keys())[:20])
                        except Exception:
                            pass
        except Exception:
            pass

        obj_list = []
        img_list = []
        for vi, (corners_img, ids_img) in enumerate(zip(corners, ids)):
            if ids_img is None or len(ids_img) == 0:
                continue
            O, I = [], []
            for c, iv in zip(corners_img, ids_img):
                tid = int(iv[0])
                if tid not in acc.id_to_obj:
                    continue
                O.append(acc.id_to_obj[tid])
                I.append(c.reshape(-1, 2))
            if not O:
                continue
            obj_cat = np.concatenate(O, axis=0).astype(np.float32)
            img_cat = np.concatenate(I, axis=0).astype(np.float32)
            obj_list.append(obj_cat)
            img_list.append(img_cat)

        total_pts = sum(len(o) for o in obj_list)
        print(f"[CAL] Mono{which}: views={len(obj_list)} points={total_pts}")
        if not obj_list:
            raise RuntimeError(f"[CAL] Mono{which}: no valid views after ID mapping!")

        K_seed = seed_K_pinhole(W, H, f_scale=1.0)
        t0 = time.perf_counter()
        rms, K, D, rvecs, tvecs = calibrate_pinhole_full(obj_list, img_list, image_size, K_seed)
        dt = time.perf_counter() - t0
        print(f"[CAL] Mono{which} done: RMS={rms:.3f} in {dt:.2f}s")
        _print_intrinsics(K, D)

        # Save one diagnostic view (best coverage by number of points)
        try:
            best_vi = max(range(len(obj_list)), key=lambda idx: len(obj_list[idx]))
            diag_index = best_vi
            best_corners = corners[diag_index]
            best_ids     = ids[diag_index]
            frame_bgr    = keyframes[diag_index][which]  # 0 or 1
            repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
            out_dir = os.path.join(repo_root, "data"); os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_png = os.path.join(out_dir, f"stereo_mono{which}_diag_{stamp}.png")
            _save_diag_pinhole(out_png, frame_bgr, best_corners, best_ids, acc, K, D, rvecs[best_vi], tvecs[best_vi], r_cyan=8, r_mag=3)
            print(f"[SAVE] Mono{which} diag -> {out_png}")
        except Exception as e:
            print(f"[WARN] Failed to save Mono{which} diag: {e}")

        return rms, K, D, obj_list, img_list, rvecs, tvecs

    print("[CAL] Calibrating mono intrinsics…")
    rms0, K0, D0, obj0_list, img0_list, rvecs0, tvecs0 = mono_calibrate(0)
    rms1, K1, D1, obj1_list, img1_list, rvecs1, tvecs1 = mono_calibrate(1)
    print(f"[CAL] RMS0={rms0:.3f}  RMS1={rms1:.3f}")

    # Save per-camera diagnostic overlays for best-coverage views
    try:
        best0 = int(np.argmax([len(o) for o in obj0_list])) if obj0_list else -1
        best1 = int(np.argmax([len(o) for o in obj1_list])) if obj1_list else -1
        if best0 >= 0 or best1 >= 0:
            repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
            out_dir = os.path.join(repo_root, "data") if not args.out_dir else os.path.abspath(args.out_dir)
            os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            if best0 >= 0:
                frame0_bgr = keyframes[min(best0, len(keyframes)-1)][0]
                diag0_png = os.path.join(out_dir, f"stereo_mono0_diag_{stamp}.png")
                _save_diag_pinhole(diag0_png, frame0_bgr, acc.corners0[best0], acc.ids0[best0], acc, K0, D0, rvecs0[best0], tvecs0[best0])
                print(f"[SAVE] Mono0 diag -> {diag0_png}")
            if best1 >= 0:
                frame1_bgr = keyframes[min(best1, len(keyframes)-1)][1]
                diag1_png = os.path.join(out_dir, f"stereo_mono1_diag_{stamp}.png")
                _save_diag_pinhole(diag1_png, frame1_bgr, acc.corners1[best1], acc.ids1[best1], acc, K1, D1, rvecs1[best1], tvecs1[best1])
                print(f"[SAVE] Mono1 diag -> {diag1_png}")
    except Exception as e:
        print("[SAVE] Failed to write mono diagnostics:", e)

    # Stereo extrinsics
    obj_list = [s.obj_pts for s in acc.stereo_samples]
    img0_list = [s.img_pts0 for s in acc.stereo_samples]
    img1_list = [s.img_pts1 for s in acc.stereo_samples]
    flags = cv2.CALIB_FIX_INTRINSIC
    print("[CAL] Calibrating stereo extrinsics…")
    rms_st, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
        obj_list, img0_list, img1_list, K0, D0, K1, D1, image_size,
        flags=flags, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    )
    print(f"[CAL] Stereo RMS={rms_st:.3f}")

    # Choose a keyframe: the one with most common IDs in latest sample
    # Use the last accumulated pair for plotting
    corners0 = acc.corners0[-1]; ids0 = acc.ids0[-1]
    corners1 = acc.corners1[-1]; ids1 = acc.ids1[-1]
    frame0_bgr = keyframes[-1][0]
    frame1_bgr = keyframes[-1][1]

    # Output directory
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    out_dir = os.path.join(repo_root, "data") if not args.out_dir else os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Save calibration JSON
    out_json = os.path.join(out_dir, f"stereo_calibration_{stamp}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image_size": [int(W), int(H)],
            "K0": K0.tolist(), "D0": D0.tolist(),
            "K1": K1.tolist(), "D1": D1.tolist(),
            "R": R.tolist(), "T": T.tolist(),
            "E": E.tolist(), "F": F.tolist(),
            "rms0": float(rms0), "rms1": float(rms1), "rms_stereo": float(rms_st),
            "board": "harvard" if args.board_source.lower()=="harvard" else "grid8x5"
        }, f, indent=2)
    print(f"[SAVE] Wrote {out_json}")

    # Plot 3D and keyframe image
    out_html = os.path.join(out_dir, f"stereo_calib_plot_{stamp}.html")
    plot_3d(board, K0, D0, K1, D1, corners0, ids0, corners1, ids1, acc, frame0_bgr, frame1_bgr, out_html, R_01=R, T_01=T)


if __name__ == "__main__":
    main()


