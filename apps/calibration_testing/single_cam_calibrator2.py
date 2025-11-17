# Single-camera calibration (cleaned & robust fisheye-first)
# - Filters weak views (few tags / poor coverage)
# - Prefilters outlier views via quick pinhole PnP
# - Progressive fisheye fit (k1,k2) -> (+k3) -> (+k4)
# - Uses rvec/tvec returned by the calibration for the diagnostic overlay
# - Avoids JSON 'rms' unbound error and reduces cluttered logging

import os, sys, time, json, queue, cv2, argparse
import numpy as np

# Allow running as a module: python -m apps.calibration_testing.single_cam_calibrator
try:
    from .. import config
    from ..capture import CamReader, set_manual_exposure_uvc, set_auto_exposure_uvc, set_uvc_gain
    from ..detect import CalibrationAccumulator, board_ids_safe
    from ..sdk import try_load_dll, sdk_config
except Exception:
    # Fallback for direct execution if relative import fails
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config
    from apps.capture import CamReader, set_manual_exposure_uvc, set_auto_exposure_uvc, set_uvc_gain
    from apps.detect import CalibrationAccumulator, board_ids_safe
    from apps.sdk import try_load_dll, sdk_config


def _make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
    ids_grid = np.arange(config.TAGS_X * config.TAGS_Y, dtype=np.int32).reshape(config.TAGS_Y, config.TAGS_X)
    ids_grid = np.flipud(ids_grid)  # bottom row IDs start at 0
    ids = ids_grid.reshape(-1, 1).astype(np.int32)
    board = cv2.aruco.GridBoard((config.TAGS_X, config.TAGS_Y), config.TAG_SIZE_M, config.TAG_SEP_M, dictionary, ids)
    return board


def _draw_status(img, lines, y0=30, dy=26, color=(0, 255, 255)):
    for k, line in enumerate(lines):
        cv2.putText(img, line, (16, y0 + dy * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def _compute_detection_metrics(corners):
    xs = [p[0] for ci in corners for p in ci.reshape(4, 2)]
    ys = [p[1] for ci in corners for p in ci.reshape(4, 2)]
    if not xs:
        return None
    center = np.array([np.mean(xs), np.mean(ys)], dtype=np.float64)
    mean_side = float(np.mean([np.mean([
        np.linalg.norm(ci.reshape(4, 2)[(i + 1) % 4] - ci.reshape(4, 2)[i]) for i in range(4)
    ]) for ci in corners]))
    return center, mean_side


def _has_coverage(corners, W, H, min_span=0.35):
    xs = [p[0] for ci in corners for p in ci.reshape(4, 2)]
    ys = [p[1] for ci in corners for p in ci.reshape(4, 2)]
    if not xs:
        return False
    span_x = (max(xs) - min(xs)) / float(max(1.0, W))
    span_y = (max(ys) - min(ys)) / float(max(1.0, H))
    return max(span_x, span_y) >= min_span


def _best_view_index(ids_list):
    best_i, best_n = -1, -1
    for i, ids_img in enumerate(ids_list):
        n = 0 if ids_img is None else len(ids_img)
        if n > best_n:
            best_n, best_i = n, i
    return best_i


def _save_diagnostic_image(path_png, frame_bgr, corners, ids, acc, K, D, rvec, tvec, use_fisheye):
    """Project using the *same* extrinsics from calibration."""
    obj_pts = []
    img_pts = []
    for c, iv in zip(corners, ids):
        tag_id = int(iv[0])
        if tag_id not in acc.id_to_obj:
            continue
        obj_pts.append(acc.id_to_obj[tag_id].astype(np.float32))      # (4,3)
        img_pts.append(c.reshape(-1, 2).astype(np.float32))           # (4,2)
    if not obj_pts:
        cv2.imwrite(path_png, frame_bgr)
        return
    img = frame_bgr.copy()
    for c, iv in zip(corners, ids):
        tag_id = int(iv[0])
        if tag_id not in acc.id_to_obj:
            continue
        obj4 = acc.id_to_obj[tag_id].astype(np.float32).reshape(1, -1, 3)
        if use_fisheye:
            proj, _ = cv2.fisheye.projectPoints(obj4.astype(np.float64), rvec, tvec, K, D)  # (1,4,2)
            proj = proj.reshape(-1, 2)
        else:
            proj, _ = cv2.projectPoints(obj4.reshape(-1, 3), rvec, tvec, K, D)
            proj = proj.reshape(-1, 2)

        det_center = c.reshape(4, 2).astype(np.float32).mean(axis=0)
        proj_center = proj.mean(axis=0)

        cv2.circle(img, (int(round(det_center[0])), int(round(det_center[1]))), 4, (255, 255, 0), -1)  # cyan
        cv2.circle(img, (int(round(proj_center[0])), int(round(proj_center[1]))), 4, (255, 0, 255), -1)  # magenta
        cv2.line(img,
                 (int(round(det_center[0])), int(round(det_center[1]))),
                 (int(round(proj_center[0])), int(round(proj_center[1]))),
                 (0, 180, 255), 1)
    cv2.putText(img, "Cyan=detected centers, Magenta=reprojected centers",
                (16, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imwrite(path_png, img)


def main():
    # --- CLI cam index --------------------------------------------------------
    def _resolve_camera_index():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--cam-index", "-i", type=int, default=None, help="Camera index for single-camera calibration")
        args, _ = parser.parse_known_args()
        if args.cam_index is not None:
            return int(args.cam_index)
        env_order = os.environ.get("CAM_INDEX_ORDER", "").strip()
        if env_order:
            try:
                indices = [int(x.strip()) for x in env_order.split(",") if x.strip() != ""]
            except Exception:
                indices = [0]
            if not indices:
                indices = [0]
            return indices[0]
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
    try:
        g = cam.cap.get(cv2.CAP_PROP_GAIN)
        current_gain = float(g) if g is not None else None
    except Exception:
        current_gain = None

    # One frame for size
    _, frame0 = cam.latest()
    H, W = frame0.shape[0], frame0.shape[1]
    image_size = (W, H)

    # Board and accumulator
    board = _make_board()
    acc = CalibrationAccumulator(board, image_size)
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())

    # Calibration state
    calibrating = False
    window_deadline = 0.0
    last_sample_t = 0.0
    keyframes = []  # list of (gray, corners, ids, frame_bgr)

    MIN_SAMPLE_PERIOD_S = 0.5

    win = "Single Cam Calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except Exception:
        pass

    try:
        while True:
            try:
                ts, frame = cam.latest()
            except queue.Empty:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.perf_counter()
            annotated = frame.copy()

            # ---------------------------- Run calibration window ----------------
            if calibrating and now >= window_deadline:
                calibrating = False
                print(f"[CAL] Window complete. Collected {len(keyframes)} keyframes.")

                # Build per-view lists (order matches acc.corners0 / acc.ids0)
                obj_list, img_list = [], []
                keep_indices = []
                for vi, (corners_img, ids_img) in enumerate(zip(acc.corners0, acc.ids0)):
                    if ids_img is None or len(ids_img) == 0:
                        continue
                    # Require minimum tags and coverage (robustness for fisheye)
                    if len(ids_img) < 6 or not _has_coverage(corners_img, W, H, min_span=0.35):
                        continue
                    obj_pts = []
                    img_pts = []
                    for c, idv in zip(corners_img, ids_img):
                        tag_id = int(idv[0])
                        if tag_id not in acc.id_to_obj:
                            continue
                        obj_pts.append(acc.id_to_obj[tag_id])                 # (4,3)
                        img_pts.append(c.reshape(-1, 2))                      # (4,2)
                    if obj_pts:
                        obj = np.concatenate(obj_pts, axis=0).astype(np.float64).reshape(-1, 1, 3)
                        img = np.concatenate(img_pts, axis=0).astype(np.float64).reshape(-1, 1, 2)
                        obj_list.append(obj)
                        img_list.append(img)
                        keep_indices.append(vi)

                if not obj_list:
                    print("[CAL] Not enough valid samples to calibrate.")
                else:
                    used_fisheye = False
                    K, D = None, None
                    rvecs, tvecs = [], []
                    rms_value = None  # ensure defined for JSON

                    # ------------------- Quick mono calibration (seed + screen) ---
                    try:
                        obj_std = [o.reshape(-1, 3).astype(np.float32) for o in obj_list]
                        img_std = [i.reshape(-1, 2).astype(np.float32) for i in img_list]
                        Ks = np.eye(3, dtype=np.float64)
                        Ds = np.zeros((5, 1), dtype=np.float64)
                        rms_mono, Ks, Ds, rvecs_mono, tvecs_mono = cv2.calibrateCamera(obj_std, img_std, image_size, Ks, Ds)
                        # Prefilter views by per-view mono PnP error
                        keep = []
                        for i, (objv, imgv) in enumerate(zip(obj_std, img_std)):
                            ok, rvec, tvec = cv2.solvePnP(objv, imgv, Ks, Ds, flags=cv2.SOLVEPNP_EPNP)
                            if not ok:
                                continue
                            proj, _ = cv2.projectPoints(objv, rvec, tvec, Ks, Ds)
                            err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - imgv, axis=1))
                            if err < 5.0:  # keep reasonably good views
                                keep.append(i)
                        if not keep:
                            # fall back to all if screen rejected everything
                            keep = list(range(len(obj_list)))
                        obj_list = [obj_list[i] for i in keep]
                        img_list = [img_list[i] for i in keep]
                        keep_indices = [keep_indices[i] for i in keep]
                        K_seed = Ks.copy()
                    except Exception as e:
                        # If mono seed fails, use identity-ish seed
                        K_seed = np.eye(3, dtype=np.float64)
                        K_seed[0, 0] = K_seed[1, 1] = max(W, H)
                        K_seed[0, 2] = W * 0.5
                        K_seed[1, 2] = H * 0.5

                    # -------------------- Progressive fisheye calibration ---------
                    try:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
                        D0 = np.zeros((4, 1), np.float64)

                        # Pass 1: fit k1,k2 only; no CHECK_COND on first pass
                        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                                 cv2.fisheye.CALIB_FIX_SKEW |
                                 cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
                                 cv2.fisheye.CALIB_FIX_K3 |
                                 cv2.fisheye.CALIB_FIX_K4)
                        rms1, K1, D1, rvecs1, tvecs1 = cv2.fisheye.calibrate(
                            obj_list, img_list, image_size, K_seed, D0, None, None, flags, criteria)

                        # Pass 2: release k3
                        flags &= ~cv2.fisheye.CALIB_FIX_K3
                        rms2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(
                            obj_list, img_list, image_size, K1, D1, None, None, flags, criteria)

                        # Pass 3: release k4
                        flags &= ~cv2.fisheye.CALIB_FIX_K4
                        rms3, K3, D3, rvecs3, tvecs3 = cv2.fisheye.calibrate(
                            obj_list, img_list, image_size, K2, D2, None, None, flags, criteria)

                        used_fisheye = True
                        rms_value, K, D, rvecs, tvecs = float(rms3), K3, D3, rvecs3, tvecs3
                        print(f"[CAL] Fisheye RMS: {rms_value:.4f} (k1..k4)")
                    except Exception as e:
                        print("[CAL] Fisheye calibration failed:", e)
                        # ----------------------- Mono fallback --------------------
                        try:
                            obj_std = [o.reshape(-1, 3).astype(np.float32) for o in obj_list]
                            img_std = [i.reshape(-1, 2).astype(np.float32) for i in img_list]
                            Ks = np.eye(3, dtype=np.float64)
                            Ds = np.zeros((5, 1), dtype=np.float64)
                            rms_mono, Ks, Ds, rvecs_mono, tvecs_mono = cv2.calibrateCamera(
                                obj_std, img_std, image_size, Ks, Ds)
                            rms_value, K, D, rvecs, tvecs = float(rms_mono), Ks, Ds, rvecs_mono, tvecs_mono
                            print(f"[CAL] Mono RMS: {rms_value:.4f}")
                        except Exception as ee:
                            print("[CAL] Calibration failed:", ee)
                            K = D = None
                            rms_value = None

                    # ---------------------- Diagnostic + Persist -------------------
                    if K is not None:
                        # Choose the best kept view (by ids count) so indices align with rvecs/tvecs
                        # keep_indices maps back to acc order; we choose the best among kept
                        best_global = _best_view_index([acc.ids0[i] for i in keep_indices])
                        # Translate to index within kept subset
                        if best_global != -1:
                            best_local = keep_indices.index(best_global)
                        else:
                            best_local = 0

                        # Pull corresponding keyframe
                        kf_idx = keep_indices[best_local]
                        _, best_corners, best_ids, best_bgr = keyframes[kf_idx]

                        repo_root = os.path.normpath(os.path.join(base_dir, ".."))
                        out_dir = os.path.join(repo_root, "data")
                        os.makedirs(out_dir, exist_ok=True)
                        stamp = time.strftime("%Y%m%d_%H%M%S")
                        out_png = os.path.join(out_dir, f"{'fisheye' if used_fisheye else 'mono'}_calib_diag_{stamp}.png")

                        # Use the very same extrinsics from the calibration
                        rvec_best, tvec_best = rvecs[best_local], tvecs[best_local]
                        _save_diagnostic_image(out_png, best_bgr, best_corners, best_ids, acc, K, D,
                                               rvec=rvec_best, tvec=tvec_best, use_fisheye=used_fisheye)
                        print(f"[SAVE] Diagnostic image -> {out_png}")

                        # Persist intrinsics
                        try:
                            out_json = os.path.join(repo_root, "data", "mono_calibration_latest.json")
                            os.makedirs(os.path.dirname(out_json), exist_ok=True)
                            payload = {
                                "image_size": list(image_size),
                                "K": (K.tolist() if K is not None else None),
                                "D": (D.tolist() if D is not None else None),
                                "rms": (float(rms_value) if rms_value is not None else None),
                                "num_keyframes": int(len(keyframes)),
                                "model": ("fisheye" if used_fisheye else "standard"),
                            }
                            with open(out_json, "w", encoding="utf-8") as f:
                                json.dump(payload, f, indent=2)
                            print(f"[SAVE] Wrote {out_json}")
                        except Exception as e:
                            print("[SAVE] JSON write failed:", e)

                # Reset for next run
                keyframes.clear()
                acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()

            # ------------------------------ Sampling gate ------------------------
            if calibrating and (time.perf_counter() - last_sample_t) >= MIN_SAMPLE_PERIOD_S:
                last_sample_t = time.perf_counter()
                try:
                    corners, ids = acc.detect(gray)
                except Exception:
                    corners, ids = [], None
                n_ids = 0 if ids is None else len(ids)

                accept = False
                if n_ids >= 6 and corners and _has_coverage(corners, W, H, min_span=0.35):
                    accept = True

                if accept:
                    ok = acc._accumulate_single(0, corners, ids)
                    if ok:
                        keyframes.append((gray.copy(), [c.copy() for c in corners], ids.copy(), frame.copy()))
                        print(f"[CAL] Keyframes: {len(keyframes)} (ids={n_ids})")

                # Draw detections for preview
                try:
                    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
                    for c, iv in zip(corners, ids if ids is not None else []):
                        p = c.reshape(4, 2).astype(int).mean(axis=0).astype(int)
                        cv2.putText(annotated, str(int(iv[0])), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except Exception:
                    pass

            # ------------------------------- UI / Keys ---------------------------
            status = []
            if calibrating:
                remain = max(0.0, window_deadline - time.perf_counter())
                status.append(f"Calibrating… keyframes={len(keyframes)}  remain={int(remain)}s")
            status.append(f"Detector: {acc.get_backend_name()}")
            _draw_status(annotated, status, y0=30)

            cv2.imshow(win, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c'):
                calibrating = not calibrating
                print(f"[KEY] Calibrating -> {calibrating}")
                if calibrating:
                    window_deadline = time.perf_counter() + 60.0
                    keyframes.clear()
                    acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()
                    last_sample_t = 0.0
            elif key == ord('e'):
                # optional UVC toggle for exposure if your camera supports it
                # kept for convenience; no extra prints
                pass

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[MAIN] Closed")


if __name__ == "__main__":
    main()
