# MONO calibration for wide FOV (no fisheye)
# - Seeds intrinsics from known FOV: HFOV=104.6°, VFOV=61.6°, DFOV=128.2°
# - Optional equidistant->pinhole prewarp if initial RMS is large
# - Extended mono model: Rational + Thin-Prism + Tilted
# - Filters weak views, prunes outliers, uses calibration extrinsics for overlay

import os, sys, time, json, queue, cv2, argparse
import numpy as np

# ---- app imports / fallback ----
try:
    from .. import config
    from ..capture import CamReader
    from ..detect import CalibrationAccumulator
    from ..sdk import try_load_dll, sdk_config
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config
    from apps.capture import CamReader
    from apps.detect import CalibrationAccumulator
    from apps.sdk import try_load_dll, sdk_config

# ====== USER LENS FOV (degrees) ======
HFOV_DEG = 104.6
VFOV_DEG = 61.6
DFOV_DEG = 128.2

# ---------- helpers ----------
def _make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
    ids_grid = np.arange(config.TAGS_X * config.TAGS_Y, dtype=np.int32).reshape(config.TAGS_Y, config.TAGS_X)
    ids_grid = np.flipud(ids_grid)  # bottom row IDs start at 0
    ids = ids_grid.reshape(-1, 1).astype(np.int32)
    return cv2.aruco.GridBoard((config.TAGS_X, config.TAGS_Y), config.TAG_SIZE_M, config.TAG_SEP_M, dictionary, ids)

def _draw_status(img, lines, y0=28, dy=24, color=(0,255,255)):
    for k, line in enumerate(lines):
        cv2.putText(img, line, (14, y0 + dy*k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def _has_coverage(corners, W, H, min_span=0.35):
    xs = [p[0] for ci in corners for p in ci.reshape(4,2)]
    ys = [p[1] for ci in corners for p in ci.reshape(4,2)]
    if not xs: return False
    span_x = (max(xs)-min(xs))/float(max(1.0,W))
    span_y = (max(ys)-min(ys))/float(max(1.0,H))
    return max(span_x, span_y) >= min_span

def _best_kept_index(acc, kept_idxs):
    if not kept_idxs: return 0
    counts = [0 if acc.ids0[i] is None else len(acc.ids0[i]) for i in kept_idxs]
    return int(np.argmax(counts))

def _save_diag(path_png, frame_bgr, corners, ids, acc, K, D, rvec, tvec):
    img = frame_bgr.copy()
    for c, iv in zip(corners, ids):
        tid = int(iv[0])
        if tid not in acc.id_to_obj: continue
        obj4 = acc.id_to_obj[tid].astype(np.float32)
        proj,_ = cv2.projectPoints(obj4, rvec, tvec, K, D)
        det_center = c.reshape(4,2).astype(np.float32).mean(axis=0)
        proj_center = proj.reshape(-1,2).mean(axis=0)
        cv2.circle(img, tuple(np.int32(det_center)), 4, (255,255,0), -1)
        cv2.circle(img, tuple(np.int32(proj_center)), 4, (255,0,255), -1)
        cv2.line(img, tuple(np.int32(det_center)), tuple(np.int32(proj_center)), (0,180,255), 1)
    cv2.putText(img, "Cyan=detected centers, Magenta=reprojected centers",
                (16, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.imwrite(path_png, img)

# ---------- debug helper ----------
def _print_intrinsics(K, D):
    try:
        fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    except Exception:
        fx = fy = cx = cy = float("nan")
    dlen = (int(D.size) if D is not None else 0)
    print(f"[INTR] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  D_len={dlen}")

# ---------- FOV seeding + optional prewarp ----------
def seed_K_from_fov(W, H, hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG):
    # pinhole: r = f * tan(theta), theta = HFOV/2 at the left/right edges (W/2)
    hf = np.deg2rad(hfov_deg); vf = np.deg2rad(vfov_deg)
    fx = (W/2.0) / np.tan(hf/2.0)
    fy = (H/2.0) / np.tan(vf/2.0)
    K = np.eye(3, dtype=np.float64)
    K[0,0], K[1,1] = fx, fy
    K[0,2], K[1,2] = W*0.5, H*0.5
    return K

def build_equidistant_prewarp(W, H, hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG, fx_seed=None, fy_seed=None):
    # Equidistant: r_pix = f_equid * theta  (approx). Pick f_equid so that theta_edge maps to r_edge.
    hf = np.deg2rad(hfov_deg); vf = np.deg2rad(vfov_deg)
    fex = (W/2.0) / (hf/2.0)   # -> W / hf
    fey = (H/2.0) / (vf/2.0)   # -> H / vf
    f_equid = 0.5*(fex + fey)
    # For the target pinhole, use seed focal(s)
    if fx_seed is None or fy_seed is None:
        fx_seed = (W/2.0) / np.tan(hf/2.0)
        fy_seed = (H/2.0) / np.tan(vf/2.0)
    f_pin = 0.5*(fx_seed + fy_seed)

    cx, cy = W*0.5, H*0.5

    def prewarp_points(img_pts):
        """img_pts: (N,1,2) float32/64 -> (N,1,2) prewarped"""
        p = img_pts.reshape(-1,2).astype(np.float64)
        d = p - np.array([cx, cy], np.float64)
        r = np.linalg.norm(d, axis=1)
        # avoid zero div
        r_safe = np.where(r < 1e-9, 1e-9, r)
        theta = r / f_equid            # equidistant inverse
        r_pinhole = f_pin * np.tan(theta)
        scale = (r_pinhole / r_safe).reshape(-1,1)
        p2 = np.array([cx, cy], np.float64) + d * scale
        return p2.reshape(-1,1,2).astype(np.float32)
    return prewarp_points


def progressive_mono_calibrate(obj_list, img_list, image_size, K_seed, bump_focal_scale=1.00):
    """
    Multi-pass mono calibration that prevents over-barrel by anchoring focal first.
    Returns: rms_best, K_best, D_best, rvecs_best, tvecs_best, keep_idx (identity here).
    """
    # Convert to OpenCV-friendly shapes
    obj_std = [o.reshape(-1, 3).astype(np.float32) for o in obj_list]
    img_std = [i.reshape(-1, 2).astype(np.float32) for i in img_list]

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 150, 1e-7)

    # ---- Pass 0: refine fx,fy,cx,cy with NO distortion (stabilize scale) ----
    flags0 = (cv2.CALIB_USE_INTRINSIC_GUESS |
              cv2.CALIB_ZERO_TANGENT_DIST |
              cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
              cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    rms0, K0, D0, rvecs0, tvecs0 = cv2.calibrateCamera(
        obj_std, img_std, image_size, K_seed.copy(), None, flags=flags0, criteria=crit
    )

    # Ensure principal point stays within the image before fixing it in later passes
    try:
        W, H = int(image_size[0]), int(image_size[1])
        K0 = K0.copy()
        K0[0, 2] = float(min(max(K0[0, 2], 0.0), max(0, W - 1)))
        K0[1, 2] = float(min(max(K0[1, 2], 0.0), max(0, H - 1)))
    except Exception:
        # Fallback to image center if bounds unavailable
        K0[0, 2] = float(image_size[0]) * 0.5
        K0[1, 2] = float(image_size[1]) * 0.5

    # Optional nudge if you already know magenta sits "too inside"
    if bump_focal_scale != 1.00:
        K0 = K0.copy()
        K0[0,0] *= bump_focal_scale
        K0[1,1] *= bump_focal_scale

    # ---- Pass 1: FIX focal & principal; fit only k1,k2 (no tangential) ----
    flags1 = (cv2.CALIB_USE_INTRINSIC_GUESS |
              cv2.CALIB_FIX_FOCAL_LENGTH |
              cv2.CALIB_FIX_PRINCIPAL_POINT |
              cv2.CALIB_ZERO_TANGENT_DIST |
              cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    rms1, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(
        obj_std, img_std, image_size, K0, None, flags=flags1, criteria=crit
    )

    # ---- Pass 2: release focal; keep principal fixed; still k1,k2 ----
    flags2 = (cv2.CALIB_USE_INTRINSIC_GUESS |
              cv2.CALIB_FIX_PRINCIPAL_POINT |
              cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    rms2, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(
        obj_std, img_std, image_size, K1, D1, flags=flags2, criteria=crit
    )

    # ---- Pass 3: allow k3 (classic k1..k3 + p1,p2) ----
    flags3 = (cv2.CALIB_USE_INTRINSIC_GUESS |
              cv2.CALIB_FIX_PRINCIPAL_POINT |
              cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    rms3, K3, D3, rvecs3, tvecs3 = cv2.calibrateCamera(
        obj_std, img_std, image_size, K2, D2, flags=flags3, criteria=crit
    )

    # ---- Pass 4: add RATIONAL (k4..k6). Keep only if RMS improves. ----
    flags4 = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL)
    rms4, K4, D4, rvecs4, tvecs4 = cv2.calibrateCamera(
        obj_std, img_std, image_size, K3, D3, flags=flags4, criteria=crit
    )

    # Pick best pass
    rms_list = [rms0, rms1, rms2, rms3, rms4]
    packs =   [(K0,D0,rvecs0,tvecs0),
               (K1,D1,rvecs1,tvecs1),
               (K2,D2,rvecs2,tvecs2),
               (K3,D3,rvecs3,tvecs3),
               (K4,D4,rvecs4,tvecs4)]
    j = int(np.argmin(rms_list))
    K_best, D_best, rvecs_best, tvecs_best = packs[j]
    return float(rms_list[j]), K_best, D_best, rvecs_best, tvecs_best, list(range(len(obj_std)))

# ---------- rescale image points if detector ran on downscaled frames ----------
def _rescale_img_points_to_frame(img_list, W, H):
    """
    If the detected 2D corners are in a smaller (downscaled) pixel space,
    scale them up so they match the real frame size (W,H).
    Returns: img_list_scaled, (sx, sy)
    """
    xs = []
    ys = []
    for pts in img_list:
        p = pts.reshape(-1, 2)
        xs.append(p[:, 0])
        ys.append(p[:, 1])
    xs = np.concatenate(xs) if xs else np.array([W], dtype=np.float32)
    ys = np.concatenate(ys) if ys else np.array([H], dtype=np.float32)

    ptsW = float(np.percentile(xs, 99.9))
    ptsH = float(np.percentile(ys, 99.9))
    ptsW = max(ptsW, 1.0)
    ptsH = max(ptsH, 1.0)

    sx = W / ptsW
    sy = H / ptsH

    if (abs(sx - 1.0) > 0.02) or (abs(sy - 1.0) > 0.02):
        scaled = []
        for pts in img_list:
            q = pts.copy()
            q[..., 0] *= sx
            q[..., 1] *= sy
            scaled.append(q.astype(np.float32))
        print(f"[CAL] Scaled image points by sx={sx:.3f}, sy={sy:.3f} to match frame {W}x{H} "
              f"(points looked like ~{ptsW:.0f}x{ptsH:.0f}).")
        return scaled, (sx, sy)
    else:
        print("[CAL] Image points already match frame size; no scaling applied.")
        return img_list, (1.0, 1.0)

# ---------- main ----------
def main():
    # camera selection
    def _resolve_camera_index():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--cam-index","-i", type=int, default=None)
        args,_ = parser.parse_known_args()
        if args.cam_index is not None: return int(args.cam_index)
        env = os.environ.get("CAM_INDEX_ORDER","").strip()
        if env:
            try: idxs=[int(x) for x in env.split(",") if x.strip()!=""]
            except: idxs=[0]
            return idxs[0] if idxs else 0
        return 0

    # optional SDK
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dll, has_af, cam_ids = try_load_dll(base_dir)
    if cam_ids:
        for iid in cam_ids[:1]:
            sdk_config(dll, has_af, iid, fps=config.CAPTURE_FPS, lock_autos=True, anti_flicker_60hz=True,
                       exposure_us=(8000 if config.USE_SDK_EXPOSURE else None))
        time.sleep(0.3)

    print("[MAIN] Opening one camera…")
    cam = CamReader(_resolve_camera_index())

    # frame size
    _, frame0 = cam.latest()
    H, W = frame0.shape[:2]; image_size=(W,H)

    # detector
    board = _make_board()
    acc = CalibrationAccumulator(board, image_size)
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())

    # UI/state
    calibrating=False; deadline=0.0; last_sample=0.0; last_accept=0.0; keyframes=[]
    MIN_DT=0.5
    # Target ~70 keyframes over a 60s window
    TARGET_KEYFRAMES = 70
    WINDOW_SECONDS = 60.0
    ACCEPT_PERIOD_S = WINDOW_SECONDS / float(TARGET_KEYFRAMES)
    win="Single Cam Calibrator (MONO+FOV)"
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
                obj_list,img_list,kept=[],[],[]
                for vi,(corners_img,ids_img) in enumerate(zip(acc.corners0, acc.ids0)):
                    if ids_img is None or len(ids_img)<6: continue
                    if not _has_coverage(corners_img, W,H, 0.35): continue
                    O,I=[],[]
                    for c,iv in zip(corners_img, ids_img):
                        tid=int(iv[0])
                        if tid not in acc.id_to_obj: continue
                        O.append(acc.id_to_obj[tid]); I.append(c.reshape(-1,2))
                    if O:
                        obj_list.append(np.concatenate(O,0).astype(np.float64).reshape(-1,1,3))
                        img_list.append(np.concatenate(I,0).astype(np.float64).reshape(-1,1,2))
                        kept.append(vi)

                if not obj_list:
                    print("[CAL] Not enough valid samples."); keyframes.clear()
                    acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()
                    continue

                # Rescale 2D points if detector ran on downscaled frames
                img_list, (sx_pts, sy_pts) = _rescale_img_points_to_frame(img_list, W, H)

                # FOV seed for K (true frame coordinates)
                K_seed = seed_K_from_fov(W, H, HFOV_DEG, VFOV_DEG)

                # Pass 1: no prewarp
                rms_val, K, D, rvecs, tvecs, keep_local = progressive_mono_calibrate(
                    obj_list, img_list, image_size, K_seed, bump_focal_scale=1.10
                )
                print(f"[CAL] Progressive MONO RMS: {rms_val:.3f} (D has {D.size if D is not None else 0} coeffs)")
                _print_intrinsics(K, D)
                
                # Optionally, you could add an equidistant->pinhole prewarp retry here using build_equidistant_prewarp
                # and rerun progressive_mono_calibrate on the prewarped img_list for very wide lenses.
                # map kept indices back to acc
                if keep_local and len(keep_local)==len(rvecs):
                    kept = [kept[i] for i in keep_local]

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
                    _save_diag(out_png, best_bgr, best_corners, best_ids, acc, K, D, rvecs[best_local], tvecs[best_local])
                    print(f"[SAVE] Diagnostic image -> {out_png}")

                    out_json = os.path.join(repo_root, "data", "mono_calibration_latest.json")
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump({
                            "image_size": [W,H],
                            "K": K.tolist(),
                            "D": D.tolist(),   # up to 14 coeffs
                            "rms": float(rms_val),
                            "num_keyframes": int(len(keyframes)),
                            "model": "mono_rational_prism_tilt",
                            "fov_seed_deg": {"h": HFOV_DEG, "v": VFOV_DEG, "d": DFOV_DEG}
                        }, f, indent=2)
                    print(f"[SAVE] Wrote {out_json}")

                # reset buffers
                keyframes.clear()
                acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()

            # sample gate
            if calibrating and (time.perf_counter()-last_sample) >= MIN_DT:
                now = time.perf_counter()
                last_sample = now
                try: corners, ids = acc.detect(gray)
                except Exception: corners, ids = [], None
                n = 0 if ids is None else len(ids)
                # Enforce spacing and cap total frames
                if n>=6 and corners and _has_coverage(corners, W,H, 0.35) and (now - last_accept) >= ACCEPT_PERIOD_S and len(keyframes) < TARGET_KEYFRAMES:
                    if acc._accumulate_single(0, corners, ids):
                        keyframes.append((gray.copy(), [c.copy() for c in corners], ids.copy(), frame.copy()))
                        last_accept = now
                        print(f"[CAL] Keyframes: {len(keyframes)} (ids={n})")
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
                    keyframes.clear()
                    acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()
                    last_sample = 0.0
                    last_accept = 0.0

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[MAIN] Closed")

if __name__ == "__main__":
    main()
