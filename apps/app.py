import os, sys, time, json, queue, cv2
import numpy as np
from . import config
from .capture import CamReader, set_manual_exposure_uvc, set_auto_exposure_uvc, set_uvc_gain
from .pose import PoseEstimator, HAVE_MP, mp_pose, mp_drawing, mp_styles
from .calib import CalibrationResults
from .detect import CalibrationAccumulator, board_ids_safe, smoke_test_tag36h11, probe_aruco_6x6
from .ui import Button, draw_ids
from .sdk import try_load_dll, sdk_config

try:
    import winsound
    def beep_ok():
        try:
            winsound.Beep(1000, 180)
        except Exception:
            pass
except Exception:
    def beep_ok():
        pass

def _tag_h_error(corners_1x4x2, obj4x3):
    img = corners_1x4x2.reshape(4,2).astype(np.float32)
    obj = obj4x3[:, :2].astype(np.float32)  # Z=0
    H, _ = cv2.findHomography(obj, img, 0)
    if H is None:
        return 1e9
    proj = cv2.perspectiveTransform(obj.reshape(-1,1,2), H).reshape(-1,2)
    return float(np.mean(np.linalg.norm(proj - img, axis=1)))

def main():
    auto_exposure_on = False
    current_exposure_step = config.DEFAULT_EXPOSURE_STEP
    current_gain = None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dll, has_af, cam_ids = try_load_dll(base_dir)
    if cam_ids:
        active_paths = cam_ids[:2]
        for iid in active_paths:
            sdk_config(dll, has_af, iid, fps=config.CAPTURE_FPS, lock_autos=True, anti_flicker_60hz=True,
                       exposure_us=(8000 if config.USE_SDK_EXPOSURE else None))
        time.sleep(0.5)

    print("[MAIN] Opening two cameras…")
    cams = []
    target_count = 2
    env_order = os.environ.get("CAM_INDEX_ORDER", "").strip()
    if env_order:
        try:
            indices = [int(x.strip()) for x in env_order.split(",") if x.strip() != ""]
        except Exception:
            indices = list(range(target_count))
        if len(indices) == 0:
            indices = list(range(target_count))
        print(f"[INFO] Using camera indices from CAM_INDEX_ORDER: {indices}")
    else:
        indices = list(range(target_count))
        print(f"[INFO] Using default camera indices: {indices}. Set CAM_INDEX_ORDER (e.g., '1,2') to override.")
    for i in indices:
        cams.append(CamReader(i))
    if not cams:
        print("[ERR] no cameras opened"); return
    print(f"[MAIN] Using {len(cams)} cam(s)")

    try:
        g = cams[0].cap.get(cv2.CAP_PROP_GAIN)
        current_gain = float(g) if g is not None else None
        if current_gain is not None:
            print(f"[CV] Initial gain read-back: {current_gain}")
    except Exception:
        current_gain = None

    ts0, f0 = cams[0].latest()
    _, f1 = cams[1].latest()
    H, W = f0.shape[0], f0.shape[1]
    image_size = (W, H)

    dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
    ids_grid = np.arange(config.TAGS_X*config.TAGS_Y, dtype=np.int32).reshape(config.TAGS_Y, config.TAGS_X)
    ids_grid = np.flipud(ids_grid)  # bottom row becomes [0..TAGS_X-1]
    ids = ids_grid.reshape(-1, 1).astype(np.int32)
    board = cv2.aruco.GridBoard((config.TAGS_X, config.TAGS_Y), config.TAG_SIZE_M, config.TAG_SEP_M, dictionary, ids)

    acc = CalibrationAccumulator(board, image_size)

    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())

    results = CalibrationResults()
    best_stereo_rms = float("inf")
    last_recalc = 0.0
    calibrated = False
    last_sample_t = 0.0

    pose_on = False
    estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                  for _ in cams]

    btn_cal = Button("Start Calibration", 20, 20, 240, 50)
    btn_pose = Button("Toggle Pose", 280, 20, 200, 50)

    state = {"calibrating": False}

    def on_mouse(event, x, y, flags, param):
        nonlocal pose_on, estimators, last_sample_t
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_cal.hit(x, y):
                state["calibrating"] = not state["calibrating"]
                btn_cal.active = state["calibrating"]
                print(f"[UI] Calibrating -> {state['calibrating']}")
                if state["calibrating"]:
                    last_sample_t = 0.0
            elif btn_pose.hit(x, y):
                pose_on = not pose_on
                print(f"[UI] Pose toggle -> {pose_on}")
                for i in range(len(estimators)):
                    estimators[i].stop()
                estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                              for _ in cams]

    win = "Stereo Calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except Exception:
        pass
    cv2.setMouseCallback(win, on_mouse)

    next_probe_t = 0.0
    last_hdbg = 0.0

    try:
        while True:
            try:
                ts0, f0 = cams[0].latest()
                ts1, f1 = cams[1].latest()
            except queue.Empty:
                time.sleep(0.01)
                continue

            frames = [f0, f1]
            g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

            # Debug per-frame IDs paused (can be re-enabled if needed)

            now = time.perf_counter()

            if HAVE_MP and pose_on:
                estimators[0].submit(ts0, f0)
                estimators[1].submit(ts1, f1)

            if state["calibrating"] and (now - last_sample_t) >= config.CALIB_SAMPLE_PERIOD_S:
                added = acc.accumulate_pair(g0, g1)
                last_sample_t = now
                if added:
                    print(f"[CAL] samples: mono0={len(acc.corners0)} mono1={len(acc.corners1)} stereo={len(acc.stereo_samples)}")

            if config.DEBUG_PROBES and state["calibrating"] and now >= next_probe_t:
                ids0 = smoke_test_tag36h11(g0)
                ids1 = smoke_test_tag36h11(g1)
                aru0 = probe_aruco_6x6(g0)
                aru1 = probe_aruco_6x6(g1)
                if aru0 or aru1:
                    print("[PROBE ArUco 6x6] cam0:", aru0, " cam1:", aru1)
                print("[SMOKE 36h11] cam0:", ids0, " cam1:", ids1)
                next_probe_t = now + 1.0

            if state["calibrating"] and (now - last_recalc) >= config.RECALC_INTERVAL_S and acc.enough_samples():
                last_recalc = now
                changed = acc.calibrate_if_possible(results)
                if results.rms_stereo is not None and results.rms_stereo < best_stereo_rms:
                    best_stereo_rms = results.rms_stereo
                    print(f"[CAL] New best stereo RMS: {best_stereo_rms:.3f}")
                if results.is_complete() and results.rms_stereo is not None and results.rms_stereo <= config.TARGET_RMS_PX:
                    calibrated = True
                    state["calibrating"] = False
                    btn_cal.active = False
                    beep_ok()
                    print("[CAL] Calibration converged ✔")

            annotated = [fr.copy() for fr in frames]

            if HAVE_MP and pose_on:
                for i in range(2):
                    latest = estimators[i].latest_result()
                    if latest and latest[1] is not None:
                        if mp_styles is not None:
                            mp_drawing.draw_landmarks(
                                annotated[i], latest[1], mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
                        else:
                            mp_drawing.draw_landmarks(annotated[i], latest[1], mp_pose.POSE_CONNECTIONS)

            det_counts = (0, 0)
            if state["calibrating"]:
                # Detect separately; do not swallow all errors for the whole block
                try:
                    c0, i0 = acc.detect(g0)
                except Exception:
                    c0, i0 = [], None
                try:
                    c1, i1 = acc.detect(g1)
                except Exception:
                    c1, i1 = [], None

                # Draw overlays (best-effort)
                try:
                    if c0:
                        cv2.aruco.drawDetectedMarkers(annotated[0], c0, i0)
                        draw_ids(annotated[0], c0, i0, (0,255,255))
                    if c1:
                        cv2.aruco.drawDetectedMarkers(annotated[1], c1, i1)
                        draw_ids(annotated[1], c1, i1, (0,255,255))
                except Exception:
                    pass

                det_counts = (len(i0) if i0 is not None else 0, len(i1) if i1 is not None else 0)

                # Homography diagnostics (throttled; independent of draw exceptions)
                if time.perf_counter() - last_hdbg > 1.0:
                    errs0 = []
                    errs1 = []
                    if i0 is not None:
                        for c, iv in zip(c0, i0):
                            tid = int(iv[0])
                            if tid in acc.id_to_obj:
                                errs0.append(_tag_h_error(c, acc.id_to_obj[tid]))
                    if i1 is not None:
                        for c, iv in zip(c1, i1):
                            tid = int(iv[0])
                            if tid in acc.id_to_obj:
                                errs1.append(_tag_h_error(c, acc.id_to_obj[tid]))
                    dt_ms = (ts1 - ts0) * 1000.0
                    set0 = set(int(iv[0]) for iv in (i0 if i0 is not None else []))
                    set1 = set(int(iv[0]) for iv in (i1 if i1 is not None else []))
                    common_n = len(set0 & set1)
                    def _stats(v):
                        return (min(v) if v else -1.0, float(np.mean(v)) if v else -1.0, max(v) if v else -1.0)
                    s0 = _stats(errs0)
                    s1 = _stats(errs1)
                    print(f"[HDBG] dt={dt_ms:.1f}ms cam0(hmin,mean,hmax)={s0} cam1={s1} common={common_n}")
                    last_hdbg = time.perf_counter()

            for img in annotated:
                btn_cal.draw(img)
                btn_pose.draw(img)
            status_lines = []
            if state["calibrating"]:
                status_lines.append(f"Calibrating… n0={len(acc.corners0)} n1={len(acc.corners1)} ns={len(acc.stereo_samples)}")
                status_lines.append(f"Detected tags: cam0={det_counts[0]} cam1={det_counts[1]}")
            status_lines.append(f"Detector: {acc.get_backend_name()}")
            if auto_exposure_on:
                status_lines.append("Exposure: Auto (UVC)")
            else:
                status_lines.append(f"Exposure: Manual step {int(current_exposure_step)}")
            if current_gain is not None:
                status_lines.append(f"Gain: {current_gain:.1f}")
            if results.rms0 is not None:
                status_lines.append(f"RMS0={results.rms0:.3f}")
            if results.rms1 is not None:
                status_lines.append(f"RMS1={results.rms1:.3f}")
            if results.rms_stereo is not None:
                status_lines.append(f"StereoRMS={results.rms_stereo:.3f}")
            if calibrated:
                for img in annotated:
                    cv2.rectangle(img, (0,0), (img.shape[1], 60), (0,180,0), -1)
                    cv2.putText(img, f"CALIBRATED ✓ Stereo RMS {results.rms_stereo:.3f}", (16,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
            for i, img in enumerate(annotated):
                y0 = 90
                for k, line in enumerate(status_lines):
                    cv2.putText(img, line, (16, y0 + 28*k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            h1, w1 = annotated[0].shape[:2]
            h2, w2 = annotated[1].shape[:2]
            target_w = max(w1, w2)
            if w1 != target_w:
                annotated[0] = cv2.resize(annotated[0], (target_w, int(round(h1 * target_w / float(w1)))))
            if w2 != target_w:
                annotated[1] = cv2.resize(annotated[1], (target_w, int(round(h2 * target_w / float(w2)))))
            PREVIEW_MIRROR = False
            display_frames = [cv2.flip(img, 1) if PREVIEW_MIRROR else img for img in annotated]
            combined = cv2.vconcat(display_frames)
            if not config.PRESERVE_NATIVE_RES:
                h, w = combined.shape[:2]
                if w > config.MAX_COMBINED_WIDTH:
                    scale = config.MAX_COMBINED_WIDTH / float(w)
                    combined = cv2.resize(combined, (config.MAX_COMBINED_WIDTH, int(round(h * scale))))
            cv2.imshow(win, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c'):
                state["calibrating"] = not state["calibrating"]
                btn_cal.active = state["calibrating"]
                print(f"[KEY] Calibrating -> {state['calibrating']}")
                if state["calibrating"]:
                    last_sample_t = 0.0
            elif key == ord('p'):
                pose_on = not pose_on
                print(f"[KEY] Pose toggle -> {pose_on}")
                for i in range(len(estimators)):
                    estimators[i].stop()
                estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                              for _ in cams]
            elif key == ord('e'):
                auto_exposure_on = not auto_exposure_on
                print(f"[KEY] Auto exposure -> {auto_exposure_on}")
                if auto_exposure_on:
                    for c in cams:
                        set_auto_exposure_uvc(c.cap)
                else:
                    for c in cams:
                        set_manual_exposure_uvc(c.cap, step=current_exposure_step)
            elif key == ord(',') or key == 44:
                if not auto_exposure_on:
                    current_exposure_step = max(config.MIN_EXPOSURE_STEP, int(current_exposure_step) - 1)
                    for c in cams:
                        set_manual_exposure_uvc(c.cap, step=current_exposure_step)
                    print(f"[KEY] Exposure step -> {current_exposure_step}")
            elif key == ord('.') or key == 46:
                if not auto_exposure_on:
                    current_exposure_step = min(config.MAX_EXPOSURE_STEP, int(current_exposure_step) + 1)
                    for c in cams:
                        set_manual_exposure_uvc(c.cap, step=current_exposure_step)
                    print(f"[KEY] Exposure step -> {current_exposure_step}")
            elif key == ord(';') or key == 59:
                try:
                    if current_gain is None:
                        current_gain = float(cams[0].cap.get(cv2.CAP_PROP_GAIN))
                    current_gain = max(config.MIN_GAIN, current_gain - config.GAIN_DELTA)
                    for c in cams:
                        set_uvc_gain(c.cap, current_gain)
                    print(f"[KEY] Gain -> {current_gain}")
                except Exception as e:
                    print("[KEY] Gain decrease failed:", e)
            elif key == ord("'") or key == 39:
                try:
                    if current_gain is None:
                        current_gain = float(cams[0].cap.get(cv2.CAP_PROP_GAIN))
                    current_gain = min(config.MAX_GAIN, current_gain + config.GAIN_DELTA)
                    for c in cams:
                        set_uvc_gain(c.cap, current_gain)
                    print(f"[KEY] Gain -> {current_gain}")
                except Exception as e:
                    print("[KEY] Gain increase failed:", e)
            elif key == ord('s'):
                if results.is_complete():
                    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
                    out_dir = os.path.join(repo_root, "data")
                    os.makedirs(out_dir, exist_ok=True)
                    path = os.path.join(out_dir, "calibration_latest.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(results.to_json_dict(), f, indent=2)
                    print(f"[SAVE] Wrote {path}")
                else:
                    print("[SAVE] Calibration incomplete; nothing written")
    finally:
        for est in estimators:
            est.stop()
        for c in cams:
            c.release()
        cv2.destroyAllWindows()
        print("[MAIN] Closed")

if __name__ == "__main__":
    main()


