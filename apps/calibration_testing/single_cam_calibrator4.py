# Single-camera calibration (robust fisheye-first, outlier-pruned)
import os, sys, time, json, queue, cv2, argparse
import numpy as np

# Module fallback
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


# ---------------- helpers ----------------
def _make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
    ids_grid = np.arange(config.TAGS_X * config.TAGS_Y, dtype=np.int32).reshape(config.TAGS_Y, config.TAGS_X)
    ids_grid = np.flipud(ids_grid)  # bottom row 0..(X-1)
    ids = ids_grid.reshape(-1, 1).astype(np.int32)
    return cv2.aruco.GridBoard((config.TAGS_X, config.TAGS_Y), config.TAG_SIZE_M, config.TAG_SEP_M, dictionary, ids)

def _draw_status(img, lines, y0=30, dy=26, color=(0,255,255)):
    for k, line in enumerate(lines):
        cv2.putText(img, line, (16, y0 + dy*k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def _has_coverage(corners, W, H, min_span=0.35):
    xs = [p[0] for ci in corners for p in ci.reshape(4,2)]
    ys = [p[1] for ci in corners for p in ci.reshape(4,2)]
    if not xs: return False
    span_x = (max(xs)-min(xs))/float(max(1.0,W))
    span_y = (max(ys)-min(ys))/float(max(1.0,H))
    return max(span_x, span_y) >= min_span

def _is_reasonable_K(K, W, H):
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    ok_f = (0.2*W <= fx <= 6*W) and (0.2*H <= fy <= 6*H)
    ok_c = (-0.25*W <= cx <= 1.25*W) and (-0.25*H <= cy <= 1.25*H)
    return ok_f and ok_c

def _best_kept_index(acc, kept_idxs):
    if not kept_idxs: return 0
    counts = [0 if acc.ids0[i] is None else len(acc.ids0[i]) for i in kept_idxs]
    return int(np.argmax(counts))

def _fisheye_view_errors(obj_list, img_list, rvecs, tvecs, K, D):
    errs = []
    for v,(obj,img) in enumerate(zip(obj_list, img_list)):
        proj,_ = cv2.fisheye.projectPoints(obj.astype(np.float64), rvecs[v], tvecs[v], K, D)
        e = np.mean(np.linalg.norm(proj.reshape(-1,2) - img.reshape(-1,2), axis=1))
        errs.append(float(e))
    return np.asarray(errs, dtype=float)

def _save_diag(path_png, frame_bgr, corners, ids, acc, K, D, rvec, tvec, use_fisheye):
    img = frame_bgr.copy()
    for c, iv in zip(corners, ids):
        tid = int(iv[0])
        if tid not in acc.id_to_obj: continue
        obj4 = acc.id_to_obj[tid].astype(np.float32).reshape(1,-1,3)
        if use_fisheye:
            proj,_ = cv2.fisheye.projectPoints(obj4.astype(np.float64), rvec, tvec, K, D)
        else:
            proj,_ = cv2.projectPoints(obj4.reshape(-1,3), rvec, tvec, K, D)
        proj = proj.reshape(-1,2)
        det_center = c.reshape(4,2).astype(np.float32).mean(axis=0)
        proj_center = proj.mean(axis=0)
        cv2.circle(img, (int(det_center[0]), int(det_center[1])), 4, (255,255,0), -1)
        cv2.circle(img, (int(proj_center[0]), int(proj_center[1])), 4, (255,0,255), -1)
        cv2.line(img, (int(det_center[0]), int(det_center[1])),
                      (int(proj_center[0]), int(proj_center[1])), (0,180,255), 1)
    cv2.putText(img, "Cyan=detected centers, Magenta=reprojected centers",
                (16, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.imwrite(path_png, img)


# ---------------- main ----------------
def main():
    # camera index
    def _resolve_camera_index():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--cam-index","-i", type=int, default=None)
        args,_ = parser.parse_known_args()
        if args.cam_index is not None: return int(args.cam_index)
        env = os.environ.get("CAM_INDEX_ORDER","").strip()
        if env:
            try: idxs=[int(x.strip()) for x in env.split(",") if x.strip()!=""]
            except: idxs=[0]
            if not idxs: idxs=[0]
            return idxs[0]
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
    index = _resolve_camera_index()
    print(f"[MAIN] Using camera index {index}")
    cam = CamReader(index)

    # frame size
    _, frame0 = cam.latest()
    H, W = frame0.shape[:2]
    image_size = (W, H)

    # detector
    board = _make_board()
    acc = CalibrationAccumulator(board, image_size)
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())

    # state
    calibrating = False
    deadline = 0.0
    last_sample_t = 0.0
    last_accept_t = 0.0
    keyframes = []
    MIN_SAMPLE_PERIOD_S = 0.5
    # Target ~50 keyframes over the window
    TARGET_KEYFRAMES = 70
    WINDOW_SECONDS = 60.0
    ACCEPT_PERIOD_S = WINDOW_SECONDS / float(TARGET_KEYFRAMES)

    win = "Single Cam Calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try: cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except: pass

    try:
        while True:
            try:
                ts, frame = cam.latest()
            except queue.Empty:
                time.sleep(0.01); continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            annotated = frame.copy()

            # ---- run calibration when window ends ----
            if calibrating and time.perf_counter() >= deadline:
                calibrating = False
                print(f"[CAL] Window complete. Collected {len(keyframes)} keyframes.")

                # build per-view with coverage filter
                obj_list, img_list, kept = [], [], []
                for vi,(corners_img, ids_img) in enumerate(zip(acc.corners0, acc.ids0)):
                    if ids_img is None or len(ids_img) < 6: continue
                    if not _has_coverage(corners_img, W, H, 0.35): continue
                    obj_pts, img_pts = [], []
                    for c,idv in zip(corners_img, ids_img):
                        tid = int(idv[0])
                        if tid not in acc.id_to_obj: continue
                        obj_pts.append(acc.id_to_obj[tid])
                        img_pts.append(c.reshape(-1,2))
                    if obj_pts:
                        obj = np.concatenate(obj_pts, 0).astype(np.float64).reshape(-1,1,3)
                        img = np.concatenate(img_pts, 0).astype(np.float64).reshape(-1,1,2)
                        obj_list.append(obj); img_list.append(img); kept.append(vi)

                if not obj_list:
                    print("[CAL] Not enough valid samples to calibrate.")
                    keyframes.clear(); acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()
                    continue

                # ---------- mono seed (+ prune gross outliers) ----------
                K_seed = np.eye(3, dtype=np.float64); K_seed[0,2]=W*0.5; K_seed[1,2]=H*0.5
                rms_value = None
                try:
                    obj_std = [o.reshape(-1,3).astype(np.float32) for o in obj_list]
                    img_std = [i.reshape(-1,2).astype(np.float32) for i in img_list]
                    Km = np.eye(3, dtype=np.float64); Dm = np.zeros((5,1), np.float64)
                    rms_mono, Km, Dm, _, _ = cv2.calibrateCamera(obj_std, img_std, image_size, Km, Dm)
                    # prune by mono PnP error
                    keep = []
                    for i,(objv,imgv) in enumerate(zip(obj_std,img_std)):
                        ok, rvec, tvec = cv2.solvePnP(objv, imgv, Km, Dm, flags=cv2.SOLVEPNP_EPNP)
                        if not ok: continue
                        proj,_ = cv2.projectPoints(objv, rvec, tvec, Km, Dm)
                        err = float(np.mean(np.linalg.norm(proj.reshape(-1,2)-imgv, axis=1)))
                        if err < 5.0: keep.append(i)
                    if keep:
                        obj_list = [obj_list[i] for i in keep]
                        img_list = [img_list[i] for i in keep]
                        kept     = [kept[i]     for i in keep]
                    K_seed = Km.copy()
                except Exception:
                    # fallback focal guess
                    K_seed[0,0] = K_seed[1,1] = 0.8*max(W,H)

                # ---------- fisheye progressive (with outlier pruning) ----------
                used_fisheye = False
                K = None; D = None; rvecs = []; tvecs = []
                try:
                    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
                    D0 = np.zeros((4,1), np.float64)

                    # pass1: lock focal+principal, fit k1,k2 + extrinsics
                    flags1 = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
                              cv2.fisheye.CALIB_FIX_SKEW |
                              cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                              cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT |
                              cv2.fisheye.CALIB_FIX_FOCAL_LENGTH |
                              cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4)
                    rms1, K1, D1, rvecs1, tvecs1 = cv2.fisheye.calibrate(obj_list, img_list, image_size,
                                                                         K_seed, D0, None, None, flags1, crit)
                    if not _is_reasonable_K(K1, W, H): raise RuntimeError("bad K pass1")

                    # pass2: release focal, allow k3
                    flags2 = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
                              cv2.fisheye.CALIB_FIX_SKEW |
                              cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                              cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT |
                              cv2.fisheye.CALIB_FIX_K4)
                    rms2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(obj_list, img_list, image_size,
                                                                         K1, D1, None, None, flags2, crit)
                    if not _is_reasonable_K(K2, W, H): raise RuntimeError("bad K pass2")

                    # ---- prune fisheye outliers by per-view error (robust) ----
                    errs = _fisheye_view_errors(obj_list, img_list, rvecs2, tvecs2, K2, D2)
                    med = float(np.median(errs))
                    thr = max(8.0, 3.0*med)  # keep within 3×median or <8 px
                    keep2 = [i for i,e in enumerate(errs) if e <= thr]
                    if keep2 and len(keep2) < len(obj_list):
                        obj_list = [obj_list[i] for i in keep2]
                        img_list = [img_list[i] for i in keep2]
                        kept     = [kept[i]     for i in keep2]
                        # re-run pass2 on the reduced set
                        rms2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(obj_list, img_list, image_size,
                                                                             K2, D2, None, None, flags2, crit)

                    # pass3: release principal & k4; try strict check; if it fails, keep pass2
                    flags3 = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
                              cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                              cv2.fisheye.CALIB_CHECK_COND)
                    try:
                        rms3, K3, D3, rvecs3, tvecs3 = cv2.fisheye.calibrate(obj_list, img_list, image_size,
                                                                             K2, D2, None, None, flags3, crit)
                        used_fisheye, rms_value, K, D, rvecs, tvecs = True, float(rms3), K3, D3, rvecs3, tvecs3
                    except Exception:
                        # keep pass2 result
                        used_fisheye, rms_value, K, D, rvecs, tvecs = True, float(rms2), K2, D2, rvecs2, tvecs2

                    print(f"[CAL] Fisheye RMS: {rms_value:.4f}")
                except Exception as e:
                    print("[CAL] Fisheye calibration failed:", e)
                    # ---- mono fallback ----
                    try:
                        obj_std = [o.reshape(-1,3).astype(np.float32) for o in obj_list]
                        img_std = [i.reshape(-1,2).astype(np.float32) for i in img_list]
                        Km = np.eye(3, dtype=np.float64); Dm = np.zeros((5,1), np.float64)
                        rms_mono, Km, Dm, rvecs_m, tvecs_m = cv2.calibrateCamera(obj_std, img_std, image_size, Km, Dm)
                        used_fisheye, rms_value, K, D, rvecs, tvecs = False, float(rms_mono), Km, Dm, rvecs_m, tvecs_m
                        print(f"[CAL] Mono RMS: {rms_value:.4f}")
                    except Exception as ee:
                        print("[CAL] Calibration failed completely:", ee)
                        K = D = None
                        rms_value = None

                # ---------- diagnostic & save ----------
                if K is not None:
                    best_local = _best_kept_index(acc, kept)
                    kf_idx = kept[best_local]
                    _, best_corners, best_ids, best_bgr = keyframes[kf_idx]

                    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
                    out_dir = os.path.join(repo_root, "data"); os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    out_png = os.path.join(out_dir, f"{'fisheye' if used_fisheye else 'mono'}_calib_diag_{stamp}.png")

                    _save_diag(out_png, best_bgr, best_corners, best_ids, acc, K, D,
                               rvec=rvecs[best_local], tvec=tvecs[best_local], use_fisheye=used_fisheye)
                    print(f"[SAVE] Diagnostic image -> {out_png}")

                    try:
                        out_json = os.path.join(repo_root, "data", "mono_calibration_latest.json")
                        payload = {
                            "image_size": list(image_size),
                            "K": K.tolist(),
                            "D": D.tolist(),
                            "rms": (float(rms_value) if rms_value is not None else None),
                            "num_keyframes": int(len(keyframes)),
                            "model": ("fisheye" if used_fisheye else "standard"),
                        }
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2)
                        print(f"[SAVE] Wrote {out_json}")
                    except Exception as e:
                        print("[SAVE] JSON write failed:", e)

                # reset buffers
                keyframes.clear()
                acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()

            # ---- sampling gate ----
            if calibrating and (time.perf_counter() - last_sample_t) >= MIN_SAMPLE_PERIOD_S:
                now = time.perf_counter()
                last_sample_t = now
                try:
                    corners, ids = acc.detect(gray)
                except Exception:
                    corners, ids = [], None
                n_ids = 0 if ids is None else len(ids)
                accept = (n_ids >= 6 and corners and _has_coverage(corners, W, H, 0.35))
                # Enforce spacing to target ~50 over the window and cap max count
                if accept and (now - last_accept_t) >= ACCEPT_PERIOD_S and len(keyframes) < TARGET_KEYFRAMES and acc._accumulate_single(0, corners, ids):
                    keyframes.append((gray.copy(), [c.copy() for c in corners], ids.copy(), frame.copy()))
                    last_accept_t = now
                    print(f"[CAL] Keyframes: {len(keyframes)} (ids={n_ids})")

                try:
                    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
                except Exception:
                    pass

            # ---- UI ----
            status = []
            if calibrating:
                remain = max(0.0, deadline - time.perf_counter())
                status.append(f"Calibrating… keyframes={len(keyframes)}  remain={int(remain)}s")
            status.append(f"Detector: {acc.get_backend_name()}")
            _draw_status(annotated, status, y0=30)

            cv2.imshow("Single Cam Calibrator", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
            if key == ord('c'):
                calibrating = not calibrating
                print(f"[KEY] Calibrating -> {calibrating}")
                if calibrating:
                    deadline = time.perf_counter() + WINDOW_SECONDS
                    keyframes.clear()
                    acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()
                    last_sample_t = 0.0
                    last_accept_t = 0.0

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[MAIN] Closed")


if __name__ == "__main__":
    main()
