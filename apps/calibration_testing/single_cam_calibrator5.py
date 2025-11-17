# Single-camera MONO calibration with extended distortion
# - No fisheye at all
# - Uses Rational (k1..k6), Thin-Prism (s1..s4), and Tilted-Sensor (tauX,tauY)
# - Filters weak views (few tags / poor image coverage)
# - Prunes outlier views by per-view reprojection error
# - Uses calibration rvec/tvec for the overlay (no PnP mismatch)
# - Saves K, D (length up to 14), RMS, and a diagnostic PNG

import os, sys, time, json, queue, cv2, argparse
import numpy as np

# Allow running as a module: python -m apps.calibration_testing.single_cam_calibrator_mono
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


# ---------- helpers ----------
def _make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
    ids_grid = np.arange(config.TAGS_X * config.TAGS_Y, dtype=np.int32).reshape(config.TAGS_Y, config.TAGS_X)
    ids_grid = np.flipud(ids_grid)  # bottom row IDs start at 0
    ids = ids_grid.reshape(-1, 1).astype(np.int32)
    return cv2.aruco.GridBoard((config.TAGS_X, config.TAGS_Y), config.TAG_SIZE_M, config.TAG_SEP_M, dictionary, ids)

def _draw_status(img, lines, y0=30, dy=26, color=(0, 255, 255)):
    for k, line in enumerate(lines):
        cv2.putText(img, line, (16, y0 + dy * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def _has_coverage(corners, W, H, min_span=0.35):
    xs = [p[0] for ci in corners for p in ci.reshape(4, 2)]
    ys = [p[1] for ci in corners for p in ci.reshape(4, 2)]
    if not xs: return False
    span_x = (max(xs) - min(xs)) / float(max(1.0, W))
    span_y = (max(ys) - min(ys)) / float(max(1.0, H))
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
        obj4 = acc.id_to_obj[tid].astype(np.float32).reshape(-1, 3)
        proj, _ = cv2.projectPoints(obj4, rvec, tvec, K, D)  # mono model supports extended D
        proj = proj.reshape(-1, 2)

        det_center = c.reshape(4, 2).astype(np.float32).mean(axis=0)
        proj_center = proj.mean(axis=0)

        cv2.circle(img, (int(det_center[0]), int(det_center[1])), 4, (255, 255, 0), -1)  # cyan
        cv2.circle(img, (int(proj_center[0]), int(proj_center[1])), 4, (255, 0, 255), -1)  # magenta
        cv2.line(img, (int(det_center[0]), int(det_center[1])),
                      (int(proj_center[0]), int(proj_center[1])), (0, 180, 255), 1)
    cv2.putText(img, "Cyan=detected centers, Magenta=reprojected centers",
                (16, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.imwrite(path_png, img)

def _robust_mono_calibrate(obj_list, img_list, image_size, K_seed=None):
    """
    Robust mono calibration:
      Pass A: RATIONAL model (k1..k6) to get a good seed
      Prune outlier views by per-view RMS
      Pass B: RATIONAL + THIN_PRISM + TILTED with USE_INTRINSIC_GUESS
    Returns: rms, K, D, rvecs, tvecs, keep_idx (indices into the input lists)
    """
    # Convert shapes/types for cv2.calibrateCamera
    obj_std = [o.reshape(-1, 3).astype(np.float32) for o in obj_list]
    img_std = [i.reshape(-1, 2).astype(np.float32) for i in img_list]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-7)

    # --- Pass A: RATIONAL only ---
    flagsA = cv2.CALIB_RATIONAL_MODEL  # enables k4,k5,k6 (plus usual k1,k2,k3,p1,p2)
    K0 = (K_seed.copy() if K_seed is not None else np.eye(3, dtype=np.float64))
    D0 = None  # let OpenCV size it as needed
    rmsA, KA, DA, rvecsA, tvecsA = cv2.calibrateCamera(
        obj_std, img_std, image_size, K0, D0, flags=flagsA, criteria=criteria
    )

    # Per-view errors -> robust pruning
    errs = []
    for i, (objv, imgv) in enumerate(zip(obj_std, img_std)):
        proj, _ = cv2.projectPoints(objv, rvecsA[i], tvecsA[i], KA, DA)
        e = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - imgv, axis=1)))
        errs.append(e)
    errs = np.asarray(errs, dtype=float)
    med = float(np.median(errs))
    thr = max(8.0, 3.0 * med)  # ≥ 3×median or ≥8 px
    keep_idx = [i for i, e in enumerate(errs) if e <= thr]

    if keep_idx and len(keep_idx) < len(obj_std):
        obj_std = [obj_std[i] for i in keep_idx]
        img_std = [img_std[i] for i in keep_idx]

    # --- Pass B: Add THIN_PRISM + TILTED with USE_INTRINSIC_GUESS ---
    flagsB = (cv2.CALIB_RATIONAL_MODEL |
              cv2.CALIB_THIN_PRISM_MODEL |
              cv2.CALIB_TILTED_MODEL |
              cv2.CALIB_USE_INTRINSIC_GUESS)
    try:
        rmsB, KB, DB, rvecsB, tvecsB = cv2.calibrateCamera(
            obj_std, img_std, image_size, KA, DA, flags=flagsB, criteria=criteria
        )
        # Choose better pass
        if rmsB <= rmsA:
            return rmsB, KB, DB, rvecsB, tvecsB, keep_idx
        else:
            return rmsA, KA, DA, rvecsA, tvecsA, keep_idx
    except Exception:
        # If extended pass fails, keep the Rational result
        return rmsA, KA, DA, rvecsA, tvecsA, keep_idx


# ---------- main ----------
def main():
    # Camera index resolution
    def _resolve_camera_index():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--cam-index", "-i", type=int, default=None)
        args, _ = parser.parse_known_args()
        if args.cam_index is not None:
            return int(args.cam_index)
        env = os.environ.get("CAM_INDEX_ORDER", "").strip()
        if env:
            try:
                idxs = [int(x.strip()) for x in env.split(",") if x.strip() != ""]
            except Exception:
                idxs = [0]
            if not idxs:
                idxs = [0]
            return idxs[0]
        return 0

    # Optional SDK pre-config
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

    # One frame for size
    _, frame0 = cam.latest()
    H, W = frame0.shape[:2]
    image_size = (W, H)

    # Board + detector
    board = _make_board()
    acc = CalibrationAccumulator(board, image_size)
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())

    # UI & state
    calibrating = False
    deadline = 0.0
    last_sample_t = 0.0
    last_accept_t = 0.0
    keyframes = []
    MIN_SAMPLE_PERIOD_S = 0.5
    # Targeted sampling cadence
    TARGET_KEYFRAMES = 70
    WINDOW_SECONDS = 60.0
    ACCEPT_PERIOD_S = WINDOW_SECONDS / float(TARGET_KEYFRAMES)

    win = "Single Cam Calibrator (MONO extended)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try: cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except Exception: pass

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

                # Build per-view lists with basic filtering
                obj_list, img_list, kept = [], [], []
                for vi, (corners_img, ids_img) in enumerate(zip(acc.corners0, acc.ids0)):
                    if ids_img is None or len(ids_img) < 6:  # need several tags
                        continue
                    if not _has_coverage(corners_img, W, H, 0.35):
                        continue
                    obj_pts, img_pts = [], []
                    for c, idv in zip(corners_img, ids_img):
                        tid = int(idv[0])
                        if tid not in acc.id_to_obj: continue
                        obj_pts.append(acc.id_to_obj[tid])           # (4,3)
                        img_pts.append(c.reshape(-1, 2))              # (4,2)
                    if obj_pts:
                        obj = np.concatenate(obj_pts, 0).astype(np.float64).reshape(-1, 1, 3)
                        img = np.concatenate(img_pts, 0).astype(np.float64).reshape(-1, 1, 2)
                        obj_list.append(obj); img_list.append(img); kept.append(vi)

                if not obj_list:
                    print("[CAL] Not enough valid samples to calibrate.")
                    keyframes.clear(); acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()
                    continue

                # Seed K roughly at center if nothing better is known
                K_seed = np.eye(3, dtype=np.float64)
                K_seed[0, 0] = K_seed[1, 1] = 0.8 * max(W, H)
                K_seed[0, 2] = W * 0.5
                K_seed[1, 2] = H * 0.5

                # ---- Robust MONO calibration (extended distortion) ----
                rms_val, K, D, rvecs, tvecs, keep_within_kept = _robust_mono_calibrate(
                    obj_list, img_list, image_size, K_seed=K_seed
                )
                print(f"[CAL] Robust MONO RMS: {rms_val:.4f}  (D has {D.size} coeffs)")

                # ---- Diagnostic & save ----
                if K is not None:
                    # If we pruned inside mono, map back to original accumulator index
                    if keep_within_kept and len(keep_within_kept) == len(rvecs):
                        kept = [kept[i] for i in keep_within_kept]

                    best_local = _best_kept_index(acc, kept)
                    kf_idx = kept[best_local]
                    # Use the detections that fed calibration for that same view
                    best_corners = acc.corners0[kf_idx]
                    best_ids     = acc.ids0[kf_idx]
                    _, _, _, best_bgr = keyframes[kf_idx]

                    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
                    out_dir = os.path.join(repo_root, "data"); os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    out_png = os.path.join(out_dir, f"mono_calib_diag_{stamp}.png")

                    _save_diag(out_png, best_bgr, best_corners, best_ids, acc, K, D,
                               rvec=rvecs[best_local], tvec=tvecs[best_local])
                    print(f"[SAVE] Diagnostic image -> {out_png}")

                    try:
                        out_json = os.path.join(repo_root, "data", "mono_calibration_latest.json")
                        payload = {
                            "image_size": list(image_size),
                            "K": K.tolist(),
                            "D": D.tolist(),  # may have up to 14 coeffs: [k1,k2,p1,p2,k3,k4,k5,k6,s1..s4,tauX,tauY]
                            "rms": float(rms_val),
                            "num_keyframes": int(len(keyframes)),
                            "model": "mono_rational_prism_tilt"
                        }
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2)
                        print(f"[SAVE] Wrote {out_json}")
                    except Exception as e:
                        print("[SAVE] JSON write failed:", e)

                # reset buffers for next run
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
                # Enforce spacing for ~TARGET_KEYFRAMES over WINDOW_SECONDS and cap total
                if accept and (now - last_accept_t) >= ACCEPT_PERIOD_S and len(keyframes) < TARGET_KEYFRAMES and acc._accumulate_single(0, corners, ids):
                    keyframes.append((gray.copy(), [c.copy() for c in corners], ids.copy(), frame.copy()))
                    last_accept_t = now
                    print(f"[CAL] Keyframes: {len(keyframes)} (ids={n_ids})")

                # quick preview overlay
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

            cv2.imshow(win, annotated)
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
