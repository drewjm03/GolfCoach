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

    # Harvard board: prefer local pickle, else env override, else attempt URL
    # Accepts structures: dict with 'ids' and 'objPoints'/'corners', list of dict objects with 'tag_id' and 'center'
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

    # Parse data
    def parse_from(data_obj):
        # If it's already a Board
        if hasattr(data_obj, "getObjPoints") and (hasattr(data_obj, "ids") or hasattr(data_obj, "getIds")):
            return data_obj
        # Try common keys
        for k in ("at_board_d", "at_coarseboard", "board", "at_board"):
            if isinstance(data_obj, dict) and k in data_obj:
                bd = data_obj[k]
                br = parse_from(bd)
                if br is not None:
                    return br
        # Try to build from dict/list formats
        obj_points = []
        ids_list = []
        if isinstance(data_obj, dict) and "ids" in data_obj:
            ids_raw = list(data_obj.get("ids", []))
            corners_raw = data_obj.get("objPoints", data_obj.get("corners", []))
            for tid, pts in zip(ids_raw, corners_raw):
                P = np.array(pts, dtype=np.float32).reshape(-1, 3 if np.array(pts).shape[-1] == 3 else 2)
                if P.shape[1] == 2:
                    P = np.hstack([P, np.zeros((P.shape[0], 1), dtype=np.float32)])
                if P.shape[0] == 4:
                    obj_points.append(P.reshape(4, 3))
                    ids_list.append(int(tid))
        elif isinstance(data_obj, (list, tuple)) and len(data_obj) > 0:
            # Pre-collect centers to infer unit->meter scale from tag_size + spacing if provided
            centers_units = []
            if harvard_tag_size_m is not None and harvard_tag_spacing_m is not None:
                for item in data_obj:
                    if isinstance(item, dict) and "tag_id" in item and "center" in item:
                        c = np.array(item["center"], dtype=np.float32).reshape(-1)
                        centers_units.append(c[:2].astype(np.float32))
            center_scale = None
            if centers_units:
                C = np.vstack(centers_units)
                dists = []
                for i in range(C.shape[0]):
                    di = np.hypot(C[i,0]-C[:,0], C[i,1]-C[:,1])
                    di = di[di > 1e-6]
                    if di.size > 0:
                        dists.append(np.min(di))
                if dists:
                    nn_units = float(np.median(np.array(dists, dtype=np.float32)))
                    desired_cc_m = float(harvard_tag_size_m) + float(harvard_tag_spacing_m)
                    if nn_units > 0:
                        center_scale = desired_cc_m / nn_units
                        print(f"[BOARD] Harvard center scale inferred: {center_scale:.6f} m/unit (nn={nn_units:.6f} units, cc={desired_cc_m:.6f} m)")
            for item in data_obj:
                if isinstance(item, dict):
                    # dict with id/corners
                    if "id" in item and ("corners" in item or "objPoints" in item):
                        tid = int(item["id"])
                        pts = item.get("objPoints", item.get("corners"))
                        P = np.array(pts, dtype=np.float32).reshape(-1, 3 if np.array(pts).shape[-1] == 3 else 2)
                        if P.shape[1] == 2:
                            P = np.hstack([P, np.zeros((P.shape[0], 1), dtype=np.float32)])
                        if P.shape[0] == 4:
                            obj_points.append(P.reshape(4, 3))
                            ids_list.append(tid)
                        continue
                    # Harvard dict with tag_id + center
                    if "tag_id" in item and "center" in item:
                        # Prefer explicit CLI/env size; else try from pickle metadata
                        size_m = None
                        if harvard_tag_size_m is not None:
                            size_m = float(harvard_tag_size_m)
                        else:
                            # Look for metadata in the outer scope (data) for a size
                            if isinstance(data, dict):
                                for k in ("tag_size", "tag_side", "tag_width", "april_tag_side", "tag_length", "marker_length"):
                                    if k in data and isinstance(data[k], (int, float)):
                                        size_m = float(data[k]); break
                                if size_m is None:
                                    for pk in ("params", "metadata", "board_params"):
                                        sub = data.get(pk)
                                        if isinstance(sub, dict):
                                            for k in ("tag_size", "tag_side", "tag_width", "april_tag_side", "tag_length", "marker_length"):
                                                if k in sub and isinstance(sub[k], (int, float)):
                                                    size_m = float(sub[k]); break
                                        if size_m is not None:
                                            break
                        if size_m is None:
                            raise RuntimeError("[BOARD] Harvard centers found but tag size unknown. Provide --harvard-tag-size-m or HARVARD_TAG_SIZE_M.")
                        tid = int(item["tag_id"])
                        c = np.array(item["center"], dtype=np.float32).reshape(-1)
                        if center_scale is not None:
                            c = c * float(center_scale)
                        if c.size == 2:
                            cx, cy, cz = float(c[0]), float(c[1]), 0.0
                        else:
                            cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
                        half = float(size_m) * 0.5
                        pts3 = np.array([
                            [cx - half, cy - half, cz],
                            [cx + half, cy - half, cz],
                            [cx + half, cy + half, cz],
                            [cx - half, cy + half, cz],
                        ], dtype=np.float32)
                        obj_points.append(pts3.reshape(4, 3)); ids_list.append(tid)
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

    board = parse_from(data)
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
    acc = CalibrationAccumulator(board, image_size)

    # Collect keyframes
    keyframes = []
    last_accept = 0.0
    print(f"[MAIN] Target keyframes: {args.target_keyframes}")
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
        obj_pts_list, img_pts_list = [], []
        view_sizes = []
        for corners_img, ids_img in zip(corners, ids):
            O, I = [], []
            for c, iv in zip(corners_img, ids_img):
                tid = int(iv[0])
                if tid not in acc.id_to_obj: continue
                O.append(acc.id_to_obj[tid]); I.append(c.reshape(-1, 2))
            if O:
                obj_cat = np.concatenate(O, axis=0).astype(np.float32)
                img_cat = np.concatenate(I, axis=0).astype(np.float32)
                obj_pts_list.append(obj_cat); img_pts_list.append(img_cat)
                view_sizes.append(int(len(obj_cat)))
        # Downselect to speed up if too many views
        if len(obj_pts_list) > int(args.mono_max_views):
            idxs = np.argsort(view_sizes)[::-1][:int(args.mono_max_views)]
            obj_pts_list = [obj_pts_list[i] for i in idxs]
            img_pts_list = [img_pts_list[i] for i in idxs]
            print(f"[CAL] Mono{which}: using top {len(obj_pts_list)} views by tag coverage")
        total_pts = sum(len(o) for o in obj_pts_list)
        print(f"[CAL] Mono{which}: views={len(obj_pts_list)} points={total_pts}")
        K = seed_K_pinhole(W, H, f_scale=1.0)
        D = np.zeros((8,1), dtype=np.float64)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, int(args.mono_max_iters), 1e-7)
        t0 = time.perf_counter()
        try:
            rms, K, D, _, _ = cv2.calibrateCamera(
                obj_pts_list, img_pts_list, image_size, K, D,
                flags=(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL),
                criteria=crit
            )
        except Exception as e:
            print(f"[CAL] Mono{which} with guess failed ({e}); retrying without guess")
            rms, K, D, _, _ = cv2.calibrateCamera(
                obj_pts_list, img_pts_list, image_size, None, None,
                flags=cv2.CALIB_RATIONAL_MODEL,
                criteria=crit
            )
        dt = time.perf_counter() - t0
        print(f"[CAL] Mono{which} done: RMS={rms:.3f} in {dt:.2f}s")
        return rms, K, D

    print("[CAL] Calibrating mono intrinsics…")
    rms0, K0, D0 = mono_calibrate(0)
    rms1, K1, D1 = mono_calibrate(1)
    print(f"[CAL] RMS0={rms0:.3f}  RMS1={rms1:.3f}")

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


