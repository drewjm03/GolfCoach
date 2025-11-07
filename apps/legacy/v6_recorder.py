import ctypes, cv2, time, threading, queue, os
import json, csv, datetime

# ---------- Tunable constants ----------
DEFAULT_EXPOSURE_STEP = -7            # −8 ≈ 3.9ms, −7 ≈ 7.8ms
DEFAULT_WB_KELVIN = 3950              # target white balance temperature in Kelvin
SYNC_THRESHOLD_MS = 15                # max allowed time skew for SYNC label
MAX_COMBINED_WIDTH = 1920             # downscale combined window to this width
FIRST_FRAME_RETRY_COUNT = 5           # attempts to read first frame
WB_TOGGLE_DELAY_S = 0.075             # delay for WB toggle dance
POSE_STOP_JOIN_TIMEOUT_S = 0.2        # join timeout for pose thread on stop
EMA_ALPHA = 0.03                      # EMA coefficient for offset
USE_SDK_EXPOSURE = False              # if True, set exposure via SDK (microseconds)
CAPTURE_FPS = 120                     # target camera FPS
MIN_EXPOSURE_US = 50                  # absolute minimum exposure
SAFETY_EXPOSURE_HEADROOM_US = 100     # keep under frame period by this much
AUTO_EXPOSURE_COMP_DELTA_US = 50      # delta for SDK exposure compensation when auto
MIN_EXPOSURE_STEP = -14               # UVC step min
MAX_EXPOSURE_STEP = 0                 # UVC step max
GAIN_DELTA = 1.0                      # UVC gain change amount per key press
MIN_GAIN = 0.0                        # min UVC gain
MAX_GAIN = 255.0                      # max UVC gain (driver dependent)

# Runtime exposure mode/state
auto_exposure_on = False
current_exposure_step = -7
current_gain = None                   # track last known UVC gain value

# ---------- Recording state ----------
recording_on = False
record_dir = None
video_writers = []                    # cv2.VideoWriter per cam
keypoint_files = []                   # open file handles per cam
keypoint_csv_writers = []             # csv.writer per cam
frame_indices = []                    # simple counters per cam
session_meta_written = False

# ---------- Optional: MediaPipe (BlazePose) import ----------
HAVE_MP, MP_IMPORT_ERR = False, ""
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    try:
        mp_styles = mp.solutions.drawing_styles
    except Exception:
        mp_styles = None
    HAVE_MP = True
except Exception as _mp_err:
    MP_IMPORT_ERR = str(_mp_err)

# ---------- DLL load (same as you had) ----------
base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.normpath(os.path.join(
    base_dir, "..", "sdk",
    "See3CAM_24CUG_Extension_Unit_SDK_1.0.65.81_Windows_20220620",
    "Win32", "Binary", "64Bit", "HIDLibraries", "eCAMFwSw.dll"))
print(f"[INFO] DLL path: {dll_path}")
dll = ctypes.WinDLL(dll_path)

WSTR  = ctypes.c_wchar_p
BOOL  = ctypes.c_bool
UINT8 = ctypes.c_ubyte
UINT32= ctypes.c_uint
INT32 = ctypes.c_int

dll.GetDevicesCount.argtypes = [ctypes.POINTER(UINT32)]
dll.GetDevicesCount.restype  = BOOL
dll.GetDevicePaths.argtypes  = [ctypes.POINTER(WSTR)]
dll.GetDevicePaths.restype   = BOOL
dll.GetExposureCompensation24CUG.argtypes = [ctypes.POINTER(UINT32)]
dll.GetExposureCompensation24CUG.restype  = BOOL

dll.InitExtensionUnit.argtypes   = [WSTR]
dll.InitExtensionUnit.restype    = BOOL
dll.DeinitExtensionUnit.argtypes = []
dll.DeinitExtensionUnit.restype  = BOOL

dll.SetStreamMode24CUG.argtypes = [UINT8, UINT8]        # 0x00=Master, 0x01=Trigger
dll.SetStreamMode24CUG.restype  = BOOL
dll.SetFrameRateValue24CUG.argtypes = [UINT8]
dll.SetFrameRateValue24CUG.restype  = BOOL
dll.SetExposureCompensation24CUG.argtypes = [UINT32]   # microseconds
dll.SetExposureCompensation24CUG.restype  = BOOL

# Optional anti-flicker (0=Auto, 1=50Hz, 2=60Hz, 3=Off)
try:
    dll.SetAntiFlickerMode24CUG.argtypes = [UINT8]
    dll.SetAntiFlickerMode24CUG.restype  = BOOL
    has_af = True
except AttributeError:
    has_af = False

# ---------- Enumerate devices (safe allocation) ----------
MAX_PATH = 260
cnt = UINT32(0)
assert dll.GetDevicesCount(ctypes.byref(cnt)), "GetDevicesCount failed"
print(f"[INFO] SDK sees {cnt.value} See3CAM_24CUG device(s)")
assert cnt.value > 0, "No cameras found by SDK"

wbufs = [ctypes.create_unicode_buffer(MAX_PATH) for _ in range(cnt.value)]
Paths = (WSTR * cnt.value)()
for i, b in enumerate(wbufs): Paths[i] = ctypes.cast(b, WSTR)
assert dll.GetDevicePaths(Paths), "GetDevicePaths failed"
cam_ids = [b.value for b in wbufs]
for i, pid in enumerate(cam_ids): print(f"[INFO] SDK device {i} instance path: {pid}")

# ---------- Helper function to set exposure compensation ----------
def set_sdk_exposure_us_for_device(instance_path, target_us):
    """Set sensor exposure time on a specific device via Extension Unit (absolute µs).
    Returns read-back exposure in µs on success, else None.
    """
    inited = False
    try:
        if not dll.InitExtensionUnit(instance_path):
            print(f"[SDK] ERROR: InitExtensionUnit failed for {instance_path}")
            return None
        inited = True
        ok = dll.SetExposureCompensation24CUG(UINT32(int(target_us)))
        if not ok:
            print("[SDK] WARNING: SetExposureCompensation24CUG failed")
            return None
        cur = UINT32(0)
        if dll.GetExposureCompensation24CUG(ctypes.byref(cur)):
            print(f"[SDK] Exposure now ~{cur.value} µs for device")
            return cur.value
        return int(target_us)
    finally:
        if inited:
            dll.DeinitExtensionUnit()

def max_exposure_us_for_fps(fps, safety_us=300):
    """Keep exposure under the frame period with a little headroom."""
    period_us = int(round(1_000_000 / max(1, int(fps))))
    return max(50, period_us - int(safety_us))

# ---------- SDK: Lock/unlock automatics (exposure, etc.) ----------
def set_sdk_lock_autos(instance_path, lock_autos=True):
    inited = False
    try:
        if not dll.InitExtensionUnit(instance_path):
            print(f"[SDK] ERROR: InitExtensionUnit failed for {instance_path}")
            return False
        inited = True
        ok = dll.SetStreamMode24CUG(UINT8(0x00), UINT8(1 if lock_autos else 0))
        if not ok:
            print(f"[SDK] WARNING: SetStreamMode24CUG lock_autos={lock_autos} failed")
        return ok
    finally:
        if inited:
            dll.DeinitExtensionUnit()

# ---------- SDK: Master mode + FPS + Anti-flicker ----------
def sdk_config(instance_path, fps=120, lock_autos=True, anti_flicker_60hz=True, exposure_us=None):
    print(f"[SDK] Init: {instance_path}")
    inited = False
    try:
        if not dll.InitExtensionUnit(instance_path):
            print("[SDK] ERROR: InitExtensionUnit failed"); return False
        inited = True

        assert dll.SetStreamMode24CUG(UINT8(0x00), UINT8(1 if lock_autos else 0))
        assert dll.SetFrameRateValue24CUG(UINT8(120 if fps >= 120 else 60))
        if has_af and anti_flicker_60hz:
            if not dll.SetAntiFlickerMode24CUG(UINT8(0x02)):
                print("[SDK] WARNING: SetAntiFlickerMode24CUG(60Hz) failed")

        # >>> set exposure here while the extension unit session is open <<<
        if exposure_us is not None:
            limit = max_exposure_us_for_fps(fps, safety_us=300)
            target_us = min(int(exposure_us), limit)
            if not dll.SetExposureCompensation24CUG(UINT32(target_us)):
                print("[SDK] WARNING: SetExposureCompensation24CUG failed")
            else:
                cur = UINT32(0)
                if dll.GetExposureCompensation24CUG(ctypes.byref(cur)):
                    print(f"[SDK] Exposure now ~{cur.value} µs")
                    # Keep overlay truthful if starting in auto
                    try:
                        global current_exposure_us
                        current_exposure_us = cur.value
                    except Exception:
                        pass

        print(f"[SDK] Master+AFL, FPS={fps} requested")
        return True
    finally:
        if inited:
            dll.DeinitExtensionUnit()
            print("[SDK] DeinitExtensionUnit")


ACTIVE_SDK_DEVICE_PATHS = cam_ids[:2]
for iid in ACTIVE_SDK_DEVICE_PATHS:
    sdk_config(
        iid,
        fps=120,
        lock_autos=True,
        anti_flicker_60hz=True,
        exposure_us=(8000 if USE_SDK_EXPOSURE else None),
    )

current_exposure_us = (8000 if USE_SDK_EXPOSURE else None)
if USE_SDK_EXPOSURE and current_exposure_us is not None:
    # Clamp to allowed limits
    current_exposure_us = max(MIN_EXPOSURE_US, min(current_exposure_us, max_exposure_us_for_fps(CAPTURE_FPS, SAFETY_EXPOSURE_HEADROOM_US)))

time.sleep(0.5)  # let driver settle

# ---------- Helpers to set UVC exposure & WB correctly ----------
def set_manual_exposure_uvc(cap, step=None):
    """Force manual exposure mode; optionally set UVC step if provided."""
    ok = False
    for v in (0.25, 1.0, 0.0):  # DShow, MSMF, fallback
        if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v):
            ok = True
            break
    if step is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(step))
        got = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"[CV] Exposure target step {step}, read-back {got}")
    else:
        print("[CV] Exposure set to MANUAL via UVC; step unchanged (SDK controls exposure)")
    return ok

def set_auto_exposure_uvc(cap):
    """Enable auto exposure via UVC common patterns."""
    for v in (0.75, 0.0):  # DShow auto ~0.75, MSMF sometimes 0
        if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v):
            print(f"[CV] Auto exposure enabled (val={v})")
            return True
    print("[CV] WARNING: failed to enable auto exposure via UVC")
    return False

def set_white_balance_uvc(cap, kelvin=4500):
    # Turn off auto WB (0=manual for DShow/MSMF); some stacks need a short delay
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    time.sleep(WB_TOGGLE_DELAY_S)
    if not cap.set(cv2.CAP_PROP_WB_TEMPERATURE, kelvin):
        cap.set(cv2.CAP_PROP_TEMPERATURE, kelvin)  # other alias
    got = cap.get(cv2.CAP_PROP_WB_TEMPERATURE) or cap.get(cv2.CAP_PROP_TEMPERATURE)
    # Fallback dance if write didn't stick
    if not got or (isinstance(got, (int, float)) and abs(float(got) - float(kelvin)) > 50):
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        time.sleep(WB_TOGGLE_DELAY_S)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        time.sleep(WB_TOGGLE_DELAY_S)
        if not cap.set(cv2.CAP_PROP_WB_TEMPERATURE, kelvin):
            cap.set(cv2.CAP_PROP_TEMPERATURE, kelvin)
        got = cap.get(cv2.CAP_PROP_WB_TEMPERATURE) or cap.get(cv2.CAP_PROP_TEMPERATURE)
    print(f"[CV] WB target {kelvin}K, read-back {got}")

def set_uvc_gain(cap, gain):
    ok = cap.set(cv2.CAP_PROP_GAIN, float(gain))
    print("[CV] Gain set->", gain, "read-back", cap.get(cv2.CAP_PROP_GAIN), "ok=", ok)

def open_cam(index, w=1280, h=720, fps=120, fourcc="MJPG"):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            continue
        print(f"[CV] cam{index} opened with backend {backend}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # --- Exposure control: SDK vs UVC ---
        if USE_SDK_EXPOSURE:
            set_auto_exposure_uvc(cap)
        else:
            set_manual_exposure_uvc(cap, step=current_exposure_step)

        # Set initial gain (driver permitting)
        try:
            set_uvc_gain(cap, 6.0)
        except Exception:
            pass

        # --- NEW: manual WB temperature ---
        set_white_balance_uvc(cap, kelvin=DEFAULT_WB_KELVIN)

        # Negotiate info
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_f = cap.get(cv2.CAP_PROP_FPS)
        fc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_readable = "".join([chr((fc_val >> (8 * i)) & 0xFF) for i in range(4)])
        print(f"[CV] cam{index} negotiated: {got_w}x{got_h} @ {got_f:.2f} (FOURCC={fourcc_readable})")

        # If negotiated FPS is far below target, try alternate FOURCC and retry
        try:
            if (isinstance(got_f, (int, float)) and got_f > 0 and got_f < (0.75 * fps)):
                print(f"[CV] cam{index} FPS {got_f:.2f} < target {fps}. Trying FOURCC=YUY2…")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
                cap.set(cv2.CAP_PROP_FPS, fps)
                time.sleep(0.05)
                got_f2 = cap.get(cv2.CAP_PROP_FPS)
                fc_val2 = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_readable2 = "".join([chr((fc_val2 >> (8 * i)) & 0xFF) for i in range(4)])
                print(f"[CV] cam{index} retry negotiated: {got_w}x{got_h} @ {got_f2:.2f} (FOURCC={fourcc_readable2})")
                if isinstance(got_f2, (int, float)) and got_f2 > 0 and got_f2 >= (0.75 * fps):
                    got_f = got_f2
                else:
                    print(f"[CV] cam{index} still below target FPS. Releasing and trying next backend…")
                    cap.release()
                    continue
        except Exception as _fps_retry_err:
            print("[CV] FPS retry check failed:", _fps_retry_err)

        ok = False
        for _ in range(FIRST_FRAME_RETRY_COUNT):
            ok, _ = cap.read()
            if ok:
                break
            time.sleep(0.02)
        if not ok:
            print(f"[WARN] cam{index} failed to deliver first frame; retrying next backend")
            cap.release()
            continue
        return cap
    raise RuntimeError(f"Could not open cam index {index}")

class CamReader:
    def __init__(self, index):
        self.cap = open_cam(index)
        self.q = queue.Queue(maxsize=1)
        self.ok = True
        self.fps = 0.0
        self._times = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.002); continue
            # Apply vertical flip for reliability across preview, pose, and recording
            try:
                f = cv2.flip(f, 0)
            except Exception:
                pass
            ts = time.perf_counter()
            if self.q.full():
                try: self.q.get_nowait()
                except: pass
            self.q.put((ts, f))
            self._times.append(ts)
            if len(self._times) > 30: self._times.pop(0)
            if len(self._times) >= 2:
                span = self._times[-1] - self._times[0]
                if span > 0: self.fps = (len(self._times)-1)/span

    def latest(self, timeout=2.0):
        ts, f = self.q.get(timeout=timeout)
        while not self.q.empty():
            ts, f = self.q.get_nowait()
        return ts, f

    def release(self):
        self.ok = False
        time.sleep(0.05)
        self.cap.release()

# ---------- Pose Estimator worker (per camera) ----------
class PoseEstimator:
    def __init__(self, enable=True, model_complexity=1, inference_width=640, inference_fps=30):
        self.enabled = enable and HAVE_MP
        self.model_complexity = model_complexity
        self.inference_width = int(inference_width)
        self.target_period = 1.0 / max(1, int(inference_fps))
        self._q = queue.Queue(maxsize=1)
        self._latest = None  # (ts, landmarks, world_landmarks)
        self._stop = False
        self._pose = None
        self._thread = None
        if self.enabled:
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                enable_segmentation=False,
                smooth_landmarks=True)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def submit(self, ts, frame_bgr):
        if not self.enabled:
            return False
        try:
            if self._q.full():
                self._q.get_nowait()
            self._q.put_nowait((ts, frame_bgr))
            return True
        except queue.Full:
            return False

    def latest_result(self):
        return self._latest

    def _loop(self):
        last_time = 0.0
        while not self._stop:
            try:
                ts, frame_bgr = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            now = time.perf_counter()
            if now - last_time < self.target_period:
                # Skip to meet target inference FPS
                # Drain queue to keep only the most recent submission
                try:
                    while self._q.qsize() > 1:
                        self._q.get_nowait()
                except queue.Empty:
                    pass
                continue
            last_time = now

            # Prepare image
            h, w = frame_bgr.shape[:2]
            if self.inference_width and w > 0:
                scale = self.inference_width / float(w)
                if scale > 0 and abs(scale - 1.0) > 1e-3:
                    nh = int(round(h * scale))
                    frame_bgr = cv2.resize(frame_bgr, (self.inference_width, nh))
            # Always operate on the most recent available frame
            try:
                while True:
                    ts_latest, frame_latest = self._q.get_nowait()
                    ts, frame_bgr = ts_latest, frame_latest
            except queue.Empty:
                pass
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._pose.process(image_rgb)
            self._latest = (ts, result.pose_landmarks, result.pose_world_landmarks)

    def stop(self):
        self._stop = True
        if self._pose is not None:
            self._pose.close()
        if self._thread is not None:
            try:
                self._thread.join(POSE_STOP_JOIN_TIMEOUT_S)
            except RuntimeError:
                pass

# ---------- Keypoints serialization helpers ----------
def landmarks_to_row(ts_frame, ts_lm, frame_w, frame_h, landmarks):
    """Convert MediaPipe normalized landmarks to a flat CSV row.
    Includes: ts_frame, ts_landmarks, frame_w, frame_h, then for each of 33 points: x_px, y_px, z_rel, visibility.
    If landmarks is None, writes NaNs for all keypoint fields.
    """
    row = [f"{ts_frame:.9f}", f"{(ts_lm if ts_lm is not None else float('nan')):.9f}", frame_w, frame_h]
    num_points = 33
    if landmarks is None:
        for _ in range(num_points):
            row.extend([float('nan'), float('nan'), float('nan'), float('nan')])
        return row
    lm_list = landmarks.landmark if hasattr(landmarks, 'landmark') else []
    for idx in range(num_points):
        if idx < len(lm_list):
            lm = lm_list[idx]
            x_px = float(lm.x) * float(frame_w)
            y_px = float(lm.y) * float(frame_h)
            z_rel = float(getattr(lm, 'z', 0.0))
            vis = float(getattr(lm, 'visibility', 0.0))
            row.extend([x_px, y_px, z_rel, vis])
        else:
            row.extend([float('nan'), float('nan'), float('nan'), float('nan')])
    return row

def keypoints_csv_header():
    base = ["ts_frame", "ts_landmarks", "frame_w", "frame_h"]
    cols = []
    for i in range(33):
        cols.extend([f"l{i}_x_px", f"l{i}_y_px", f"l{i}_z_rel", f"l{i}_vis"])
    return base + cols

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def create_session(cams):
    global record_dir, video_writers, keypoint_files, keypoint_csv_writers, frame_indices, session_meta_written
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    record_dir = os.path.join(repo_root, "data", ts)
    record_dir = os.path.normpath(record_dir)
    ensure_dir(record_dir)

    video_writers = []
    keypoint_files = []
    keypoint_csv_writers = []
    frame_indices = [0 for _ in cams]
    session_meta_written = False

    # Create per-cam writers
    for i, cam in enumerate(cams):
        w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vpath = os.path.join(record_dir, f"cam{i}.mp4")
        vw = cv2.VideoWriter(vpath, fourcc, float(CAPTURE_FPS), (w, h))
        video_writers.append(vw)

        kpath = os.path.join(record_dir, f"keypoints_cam{i}.csv")
        kf = open(kpath, "w", newline="", encoding="utf-8")
        kw = csv.writer(kf)
        kw.writerow(keypoints_csv_header())
        keypoint_files.append(kf)
        keypoint_csv_writers.append(kw)

    # Write meta.json
    meta = {
        "capture_fps": CAPTURE_FPS,
        "cams": [
            {
                "index": i,
                "negotiated_width": int(c.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "negotiated_height": int(c.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "sdk_instance_path": (ACTIVE_SDK_DEVICE_PATHS[i] if i < len(ACTIVE_SDK_DEVICE_PATHS) else None),
            } for i, c in enumerate(cams)
        ],
        "note": "Keypoints are 2D pixel coordinates per frame for 33 pose landmarks. Align by ts_frame across cams, with ts_landmarks from estimator.",
    }
    with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    session_meta_written = True

def close_session():
    global video_writers, keypoint_files, keypoint_csv_writers
    for vw in video_writers:
        try:
            vw.release()
        except Exception:
            pass
    for f in keypoint_files:
        try:
            f.flush()
            f.close()
        except Exception:
            pass
    video_writers = []
    keypoint_files = []
    keypoint_csv_writers = []

def main():
    global current_exposure_us, auto_exposure_on, current_exposure_step, recording_on
    print("[MAIN] Opening two cameras…")
    cams = []
    # Optional: allow explicit OpenCV index order via env var, e.g., "1,0"
    target_count = min(2, len(cam_ids))
    env_order = os.environ.get("CAM_INDEX_ORDER", "").strip()
    if env_order:
        try:
            indices = [int(x.strip()) for x in env_order.split(",") if x.strip() != ""]
        except Exception:
            indices = list(range(target_count))
        if len(indices) != target_count:
            print(f"[WARN] CAM_INDEX_ORDER specifies {len(indices)} index(es), SDK enumerated {target_count}; proceeding with provided indices")
    else:
        indices = list(range(target_count))
    if target_count != len(cam_ids):
        print(f"[WARN] SDK enumerated {len(cam_ids)} device(s) but only opening {target_count}. OpenCV index order may not match SDK order.")
    for i in indices:
        cams.append(CamReader(i))
    if not cams:
        print("[ERR] no cameras opened"); return
    print(f"[MAIN] Using {len(cams)} cam(s)")
    # Initialize current_gain from first camera if available
    try:
        g = cams[0].cap.get(cv2.CAP_PROP_GAIN)
        current_gain = float(g) if g is not None else None
        if current_gain is not None:
            print(f"[CV] Initial gain read-back: {current_gain}")
    except Exception:
        current_gain = None

    # Pose estimators per cam
    pose_on = True
    estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                  for _ in cams]

    alpha, offset = EMA_ALPHA, 0.0
    try:
        while True:
            try:
                ts0, f0 = cams[0].latest()
            except queue.Empty:
                print("[WARN] cam0 stalled: no frame within timeout")
                time.sleep(0.01)
                continue
            frames = [f0]
            status = ""
            if len(cams) > 1:
                try:
                    ts1, f1 = cams[1].latest()
                except queue.Empty:
                    print("[WARN] cam1 stalled: no frame within timeout")
                    time.sleep(0.01)
                    continue
                obs = ts1 - ts0
                offset = (1-alpha)*offset + alpha*obs
                dt_ms = abs((ts1 - (ts0 + offset))*1000.0)
                status = f"{'SYNC' if dt_ms<=SYNC_THRESHOLD_MS else 'UNSYNC'} ({dt_ms:.1f} ms)"
                frames.append(f1)

            # Submit frames to pose estimators (non-blocking)
            if HAVE_MP and pose_on:
                for i, fr in enumerate(frames):
                    ts = ts0 if i == 0 else ts1
                    estimators[i].submit(ts, fr)

            # Draw overlays if available
            annotated_frames = list(frames)
            if HAVE_MP and pose_on:
                for i in range(len(frames)):
                    latest = estimators[i].latest_result()
                    if latest and latest[1] is not None:
                        # Draw on a copy to avoid affecting input frame timing
                        annotated = annotated_frames[i].copy()
                        if mp_styles is not None:
                            mp_drawing.draw_landmarks(
                                annotated,
                                latest[1],
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
                        else:
                            mp_drawing.draw_landmarks(annotated, latest[1], mp_pose.POSE_CONNECTIONS)
                        annotated_frames[i] = annotated

            # Create side-by-side display
            if len(annotated_frames) == 2:
                # Resize frames to same height if needed
                h1, w1 = annotated_frames[0].shape[:2]
                h2, w2 = annotated_frames[1].shape[:2]
                target_h = max(h1, h2)
                
                # Resize frames to target height
                if h1 != target_h:
                    annotated_frames[0] = cv2.resize(annotated_frames[0], (int(w1 * target_h / h1), target_h))
                if h2 != target_h:
                    annotated_frames[1] = cv2.resize(annotated_frames[1], (int(w2 * target_h / h2), target_h))
                
                # Add text overlays
                cv2.putText(annotated_frames[0], f"cam0 FPS:{cams[0].fps:.1f} target:{int(CAPTURE_FPS)}", (16,36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(annotated_frames[1], f"cam1 FPS:{cams[1].fps:.1f} target:{int(CAPTURE_FPS)}", (16,36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if status:
                    cv2.putText(annotated_frames[0], status, (16,72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    cv2.putText(annotated_frames[1], status, (16,72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                
                # Concatenate horizontally
                combined_frame = cv2.hconcat(annotated_frames)
                # Overlay current exposure readout
                if auto_exposure_on and current_exposure_us is not None:
                    cv2.putText(combined_frame, f"Auto Exp: {int(current_exposure_us)} us",
                                (16, combined_frame.shape[0]-16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                elif not auto_exposure_on:
                    cv2.putText(combined_frame, f"Manual Step: {int(current_exposure_step)}",
                                (16, combined_frame.shape[0]-16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                # Overlay gain if available
                if current_gain is not None:
                    cv2.putText(combined_frame, f"Gain: {current_gain:.1f}",
                                (16, combined_frame.shape[0]-48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                # Optionally downscale to avoid overly wide windows on small displays
                h, w = combined_frame.shape[:2]
                if w > MAX_COMBINED_WIDTH:
                    scale = MAX_COMBINED_WIDTH / float(w)
                    combined_frame = cv2.resize(combined_frame, (MAX_COMBINED_WIDTH, int(round(h * scale))))
                # REC indicator
                if recording_on:
                    cv2.putText(combined_frame, "REC", (combined_frame.shape[1]-100, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                cv2.imshow("Dual Camera View", combined_frame)
            else:
                # Single camera fallback
                cv2.putText(annotated_frames[0], f"cam0 FPS:{cams[0].fps:.1f} target:{int(CAPTURE_FPS)}", (16,36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if auto_exposure_on and current_exposure_us is not None:
                    cv2.putText(annotated_frames[0], f"Auto Exp: {int(current_exposure_us)} us",
                                (16, annotated_frames[0].shape[0]-16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                elif not auto_exposure_on:
                    cv2.putText(annotated_frames[0], f"Manual Step: {int(current_exposure_step)}",
                                (16, annotated_frames[0].shape[0]-16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                if current_gain is not None:
                    cv2.putText(annotated_frames[0], f"Gain: {current_gain:.1f}",
                                (16, annotated_frames[0].shape[0]-48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                if recording_on:
                    cv2.putText(annotated_frames[0], "REC", (annotated_frames[0].shape[1]-100, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                cv2.imshow("Dual Camera View", annotated_frames[0])
                
            # Write recording outputs (videos + keypoints)
            if recording_on and video_writers:
                # Write raw frames per cam
                for i, fr in enumerate(frames):
                    try:
                        video_writers[i].write(fr)
                    except Exception:
                        pass
                # Serialize keypoints per cam (if MediaPipe available)
                for i in range(len(frames)):
                    w_i = int(cams[i].cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h_i = int(cams[i].cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    latest = estimators[i].latest_result() if HAVE_MP else None
                    ts_lm = latest[0] if (latest and latest[1] is not None) else None
                    lm = latest[1] if (latest and latest[1] is not None) else None
                    ts_frame = (ts0 if i == 0 else ts1) if len(frames) > 1 else ts0
                    row = landmarks_to_row(ts_frame, ts_lm, w_i, h_i, lm)
                    try:
                        keypoint_csv_writers[i].writerow(row)
                    except Exception:
                        pass

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('p'):
                print(f"[KEY] 'p' pressed. HAVE_MP={HAVE_MP}. Current pose_on={pose_on}")
                pose_on = not pose_on
                print(f"[KEY] Toggling pose to {'ON' if pose_on else 'OFF'}")
                # Reconfigure estimators
                for i in range(len(estimators)):
                    estimators[i].stop()
                time.sleep(POSE_STOP_JOIN_TIMEOUT_S)
                estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                              for _ in cams]
                if pose_on and not HAVE_MP:
                    print(f"[KEY] Pose requested ON but MediaPipe not available: {MP_IMPORT_ERR}")
                elif pose_on and HAVE_MP:
                    print(f"[KEY] Pose enabled: created {len(estimators)} estimator(s)")
                else:
                    print("[KEY] Pose disabled")
            elif key == ord('e'):
                # Toggle auto/manual exposure
                auto_exposure_on = not auto_exposure_on
                print(f"[KEY] 'e' pressed. Auto exposure -> {auto_exposure_on}")
                if auto_exposure_on:
                    # Enable auto via UVC; unlock autos via SDK too
                    for c in cams:
                        set_auto_exposure_uvc(c.cap)
                    for iid in ACTIVE_SDK_DEVICE_PATHS:
                        set_sdk_lock_autos(iid, lock_autos=False)
                else:
                    # Disable auto via UVC and set a manual step
                    for c in cams:
                        set_manual_exposure_uvc(c.cap, step=current_exposure_step)
                    for iid in ACTIVE_SDK_DEVICE_PATHS:
                        set_sdk_lock_autos(iid, lock_autos=True)
            elif key == ord('r'):
                # Toggle recording
                recording_on = not recording_on
                if recording_on:
                    print("[KEY] 'r' pressed. Recording START")
                    create_session(cams)
                else:
                    print("[KEY] 'r' pressed. Recording STOP")
                    close_session()
            elif key == ord(',') or key == 44:  # decrease exposure
                if auto_exposure_on:
                    # Adjust SDK exposure compensation (us)
                    if current_exposure_us is None:
                        current_exposure_us = max_exposure_us_for_fps(CAPTURE_FPS, SAFETY_EXPOSURE_HEADROOM_US)
                    current_exposure_us = max(MIN_EXPOSURE_US, current_exposure_us - AUTO_EXPOSURE_COMP_DELTA_US)
                    for iid in ACTIVE_SDK_DEVICE_PATHS:
                        rb = set_sdk_exposure_us_for_device(iid, current_exposure_us)
                        if rb is not None:
                            current_exposure_us = rb
                    print(f"[KEY] ',' pressed (auto). Exposure compensation -> {current_exposure_us} us")
                else:
                    # Adjust manual UVC step
                    current_exposure_step = max(MIN_EXPOSURE_STEP, int(current_exposure_step) - 1)
                    for c in cams:
                        set_manual_exposure_uvc(c.cap, step=current_exposure_step)
                    print(f"[KEY] ',' pressed (manual). Exposure step -> {current_exposure_step}")
            elif key == ord('.') or key == 46:  # increase exposure
                if auto_exposure_on:
                    if current_exposure_us is None:
                        current_exposure_us = max_exposure_us_for_fps(CAPTURE_FPS, SAFETY_EXPOSURE_HEADROOM_US)
                    limit = max_exposure_us_for_fps(CAPTURE_FPS, SAFETY_EXPOSURE_HEADROOM_US)
                    current_exposure_us = min(limit, current_exposure_us + AUTO_EXPOSURE_COMP_DELTA_US)
                    for iid in ACTIVE_SDK_DEVICE_PATHS:
                        rb = set_sdk_exposure_us_for_device(iid, current_exposure_us)
                        if rb is not None:
                            current_exposure_us = rb
                    print(f"[KEY] '.' pressed (auto). Exposure compensation -> {current_exposure_us} us")
                else:
                    current_exposure_step = min(MAX_EXPOSURE_STEP, int(current_exposure_step) + 1)
                    for c in cams:
                        set_manual_exposure_uvc(c.cap, step=current_exposure_step)
                    print(f"[KEY] '.' pressed (manual). Exposure step -> {current_exposure_step}")
            elif key == ord(';') or key == 59:  # decrease gain
                try:
                    if current_gain is None:
                        current_gain = float(cams[0].cap.get(cv2.CAP_PROP_GAIN))
                    current_gain = max(MIN_GAIN, current_gain - GAIN_DELTA)
                    for c in cams:
                        set_uvc_gain(c.cap, current_gain)
                    print(f"[KEY] ';' pressed. Gain -> {current_gain}")
                except Exception as e:
                    print("[KEY] Gain decrease failed:", e)
            elif key == ord("'") or key == 39:  # increase gain
                try:
                    if current_gain is None:
                        current_gain = float(cams[0].cap.get(cv2.CAP_PROP_GAIN))
                    current_gain = min(MAX_GAIN, current_gain + GAIN_DELTA)
                    for c in cams:
                        set_uvc_gain(c.cap, current_gain)
                    print(f"[KEY] '\'' pressed. Gain -> {current_gain}")
                except Exception as e:
                    print("[KEY] Gain increase failed:", e)
    finally:
        if recording_on:
            try:
                close_session()
            except Exception:
                pass
        for est in estimators:
            est.stop()
        for c in cams: c.release()
        cv2.destroyAllWindows()
        print("[MAIN] Closed")

if __name__ == "__main__":
    main()
