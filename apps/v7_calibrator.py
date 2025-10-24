import ctypes, cv2, time, threading, queue, os, sys, json
import numpy as np

# Optional sound on Windows
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

# ---------- Tunable constants ----------
CAPTURE_FPS = 120
MAX_COMBINED_WIDTH = 1920
FIRST_FRAME_RETRY_COUNT = 5
WB_TOGGLE_DELAY_S = 0.075

# Exposure/gain defaults
USE_SDK_EXPOSURE = False
DEFAULT_EXPOSURE_STEP = -7
MIN_EXPOSURE_STEP = -14
MAX_EXPOSURE_STEP = 0
MIN_EXPOSURE_US = 50
SAFETY_EXPOSURE_HEADROOM_US = 100
AUTO_EXPOSURE_COMP_DELTA_US = 50
DEFAULT_WB_KELVIN = 3950
MIN_GAIN = 0.0
MAX_GAIN = 255.0
GAIN_DELTA = 1.0

# AprilTag grid board configuration 
APRIL_DICT = cv2.aruco.DICT_APRILTAG_36h11
TAGS_X = 8                # number of tags horizontally
TAGS_Y = 5                # number of tags vertically
TAG_SIZE_M = 0.075         # tag black square size in meters
TAG_SEP_M = 0.01875          # white gap between tags in meters

# Calibration parameters/criteria
MIN_MARKERS_PER_VIEW = 8
MIN_SAMPLES = 20
TARGET_RMS_PX = 0.6
RECALC_INTERVAL_S = 1.0

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

# ---------- DLL load (See3CAM Extension Unit) ----------
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

# ---------- Enumerate devices ----------
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

def max_exposure_us_for_fps(fps, safety_us=300):
    period_us = int(round(1_000_000 / max(1, int(fps))))
    return max(50, period_us - int(safety_us))

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
        if exposure_us is not None:
            limit = max_exposure_us_for_fps(fps, safety_us=300)
            target_us = min(int(exposure_us), limit)
            if not dll.SetExposureCompensation24CUG(UINT32(target_us)):
                print("[SDK] WARNING: SetExposureCompensation24CUG failed")
        return True
    finally:
        if inited:
            dll.DeinitExtensionUnit()
            print("[SDK] DeinitExtensionUnit")

ACTIVE_SDK_DEVICE_PATHS = cam_ids[:2]
for iid in ACTIVE_SDK_DEVICE_PATHS:
    sdk_config(iid, fps=CAPTURE_FPS, lock_autos=True, anti_flicker_60hz=True,
               exposure_us=(8000 if USE_SDK_EXPOSURE else None))

time.sleep(0.5)

# ---------- UVC helpers ----------
def set_manual_exposure_uvc(cap, step=None):
    ok = False
    for v in (0.25, 1.0, 0.0):
        if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v):
            ok = True
            break
    if step is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(step))
    return ok

def set_auto_exposure_uvc(cap):
    for v in (0.75, 0.0):
        if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v):
            return True
    return False

def set_white_balance_uvc(cap, kelvin=4500):
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    time.sleep(WB_TOGGLE_DELAY_S)
    if not cap.set(cv2.CAP_PROP_WB_TEMPERATURE, kelvin):
        cap.set(cv2.CAP_PROP_TEMPERATURE, kelvin)
    got = cap.get(cv2.CAP_PROP_WB_TEMPERATURE) or cap.get(cv2.CAP_PROP_TEMPERATURE)
    if not got or (isinstance(got, (int, float)) and abs(float(got) - float(kelvin)) > 50):
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        time.sleep(WB_TOGGLE_DELAY_S)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        time.sleep(WB_TOGGLE_DELAY_S)
        if not cap.set(cv2.CAP_PROP_WB_TEMPERATURE, kelvin):
            cap.set(cv2.CAP_PROP_TEMPERATURE, kelvin)

def set_uvc_gain(cap, gain):
    try:
        cap.set(cv2.CAP_PROP_GAIN, float(gain))
    except Exception:
        pass

def open_cam(index, w=1280, h=720, fps=CAPTURE_FPS, fourcc="MJPG"):
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
        if USE_SDK_EXPOSURE:
            set_auto_exposure_uvc(cap)
        else:
            set_manual_exposure_uvc(cap, step=DEFAULT_EXPOSURE_STEP)
        try:
            set_uvc_gain(cap, 6.0)
        except Exception:
            pass
        set_white_balance_uvc(cap, kelvin=DEFAULT_WB_KELVIN)

        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_f = cap.get(cv2.CAP_PROP_FPS)
        fc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_readable = "".join([chr((fc_val >> (8 * i)) & 0xFF) for i in range(4)])
        print(f"[CV] cam{index} negotiated: {got_w}x{got_h} @ {got_f:.2f} (FOURCC={fourcc_readable})")

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

# ---------- Pose Estimator worker (optional) ----------
class PoseEstimator:
    def __init__(self, enable=True, model_complexity=1, inference_width=640, inference_fps=30):
        self.enabled = enable and HAVE_MP
        self.model_complexity = model_complexity
        self.inference_width = int(inference_width)
        self.target_period = 1.0 / max(1, int(inference_fps))
        self._q = queue.Queue(maxsize=1)
        self._latest = None
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
                try:
                    while self._q.qsize() > 1:
                        self._q.get_nowait()
                except queue.Empty:
                    pass
                continue
            last_time = now
            h, w = frame_bgr.shape[:2]
            if self.inference_width and w > 0:
                scale = self.inference_width / float(w)
                if scale > 0 and abs(scale - 1.0) > 1e-3:
                    nh = int(round(h * scale))
                    frame_bgr = cv2.resize(frame_bgr, (self.inference_width, nh))
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
                self._thread.join(0.2)
            except RuntimeError:
                pass

# ---------- Calibration data containers ----------
class CalibrationResults:
    def __init__(self):
        self.K0 = None
        self.D0 = None
        self.K1 = None
        self.D1 = None
        self.image_size = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.rms0 = None
        self.rms1 = None
        self.rms_stereo = None

    def is_complete(self):
        return all([x is not None for x in (self.K0, self.D0, self.K1, self.D1, self.R, self.T)])

    def to_json_dict(self):
        def as_list(x):
            return (x.tolist() if isinstance(x, np.ndarray) else x)
        return {
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "K0": as_list(self.K0),
            "D0": as_list(self.D0),
            "K1": as_list(self.K1),
            "D1": as_list(self.D1),
            "R": as_list(self.R),
            "T": as_list(self.T),
            "E": as_list(self.E),
            "F": as_list(self.F),
            "rms0": self.rms0,
            "rms1": self.rms1,
            "rms_stereo": self.rms_stereo,
        }

class StereoSample:
    def __init__(self, obj_pts, img_pts0, img_pts1):
        self.obj_pts = obj_pts  # (N,3)
        self.img_pts0 = img_pts0  # (N,2)
        self.img_pts1 = img_pts1  # (N,2)

class CalibrationAccumulator:
    def __init__(self, board, image_size):
        self.board = board
        self.image_size = image_size
        self.detector = self._make_detector()
        # Per-cam accumulators for mono calibration
        self.corners0, self.ids0, self.counter0 = [], [], []
        self.corners1, self.ids1, self.counter1 = [], [], []
        # Stereo samples (matched markers in both cams)
        self.stereo_samples = []
        # Intrinsics cache
        self.K0 = None; self.D0 = None; self.rms0 = None
        self.K1 = None; self.D1 = None; self.rms1 = None
        # id -> objCorners lookup
        self.id_to_obj = self._build_id_to_object()

    def _make_detector(self):
        dictionary = cv2.aruco.getPredefinedDictionary(APRIL_DICT)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.ArucoDetector(dictionary, params)

    def _build_id_to_object(self):
        # GridBoard has .ids (Nx1) and .objPoints list length N with 4x3 points
        id_to_obj = {}
        try:
            ids = self.board.ids.flatten().astype(int)
            for idx, tag_id in enumerate(ids):
                obj = np.array(self.board.objPoints[idx], dtype=np.float32).reshape(-1, 3)
                id_to_obj[int(tag_id)] = obj
        except Exception:
            # Fallback: assume sequential ids 0..N-1
            N = TAGS_X * TAGS_Y
            for idx in range(N):
                obj = np.array(self.board.objPoints[idx], dtype=np.float32).reshape(-1, 3)
                id_to_obj[idx] = obj
        return id_to_obj

    def detect(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return [], None
        # Optionally subpixel refine
        for c in corners:
            cv2.cornerSubPix(gray, c.squeeze(1), (3,3), (-1,-1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.01))
        return corners, ids.astype(int)

    def _accumulate_single(self, cam_idx, corners, ids):
        if corners is None or ids is None or len(corners) < MIN_MARKERS_PER_VIEW:
            return False
        if cam_idx == 0:
            self.corners0.append(corners)
            self.ids0.append(ids)
            self.counter0.append(len(ids))
        else:
            self.corners1.append(corners)
            self.ids1.append(ids)
            self.counter1.append(len(ids))
        return True

    def _match_stereo(self, corners0, ids0, corners1, ids1):
        # Build maps id -> 4x2 corners (order consistent with obj points)
        map0 = {int(i[0]): c.reshape(-1,2) for c, i in zip(corners0, ids0)}
        map1 = {int(i[0]): c.reshape(-1,2) for c, i in zip(corners1, ids1)}
        common = sorted(set(map0.keys()) & set(map1.keys()))
        if len(common) == 0:
            return None
        obj_pts = []
        img0 = []
        img1 = []
        for tag_id in common:
            if tag_id not in self.id_to_obj:
                continue
            obj = self.id_to_obj[tag_id]
            obj_pts.append(obj)
            img0.append(map0[tag_id])
            img1.append(map1[tag_id])
        if len(obj_pts) == 0:
            return None
        obj_pts = np.concatenate(obj_pts, axis=0).astype(np.float32)
        img0 = np.concatenate(img0, axis=0).astype(np.float32)
        img1 = np.concatenate(img1, axis=0).astype(np.float32)
        return StereoSample(obj_pts, img0, img1)

    def accumulate_pair(self, gray0, gray1):
        c0, i0 = self.detect(gray0)
        c1, i1 = self.detect(gray1)
        ok0 = self._accumulate_single(0, c0, i0)
        ok1 = self._accumulate_single(1, c1, i1)
        if ok0 and ok1:
            sample = self._match_stereo(c0, i0, c1, i1)
            if sample is not None and sample.obj_pts.shape[0] >= MIN_MARKERS_PER_VIEW*4:
                self.stereo_samples.append(sample)
                return True
        return False

    def enough_samples(self):
        return (len(self.corners0) >= MIN_SAMPLES and
                len(self.corners1) >= MIN_SAMPLES and
                len(self.stereo_samples) >= max(8, MIN_SAMPLES//2))

    def _mono_calibrate(self, which):
        if which == 0:
            corners, ids, counter = self.corners0, self.ids0, self.counter0
        else:
            corners, ids, counter = self.corners1, self.ids1, self.counter1
        if len(corners) == 0:
            return None, None, None
        dictionary = cv2.aruco.getPredefinedDictionary(APRIL_DICT)
        # Flatten per-image lists into the shape calibrateCameraAruco expects
        all_corners = [c for corners_img in corners for c in corners_img]
        all_ids = [i for ids_img in ids for i in ids_img]
        all_counter = [len(ids_img) for ids_img in ids]
        try:
            rms, K, D, _, _ = cv2.aruco.calibrateCameraAruco(
                all_corners, np.concatenate(all_ids, axis=0), np.array(all_counter, dtype=np.int32),
                self.board, self.image_size, None, None)
        except Exception:
            # Fallback to standard calibrateCamera using concatenated marker corners as points
            obj_pts_list = []
            img_pts_list = []
            for corners_img, ids_img in zip(corners, ids):
                # use board mapping for only those seen
                obj_pts = []
                img_pts = []
                for c, idv in zip(corners_img, ids_img):
                    tag_id = int(idv[0])
                    if tag_id not in self.id_to_obj:
                        continue
                    obj_pts.append(self.id_to_obj[tag_id])
                    img_pts.append(c.reshape(-1,2))
                if len(obj_pts) == 0:
                    continue
                obj_pts_list.append(np.concatenate(obj_pts, axis=0).astype(np.float32))
                img_pts_list.append(np.concatenate(img_pts, axis=0).astype(np.float32))
            if len(obj_pts_list) == 0:
                return None, None, None
            K = np.eye(3, dtype=np.float64)
            D = np.zeros((5,1), dtype=np.float64)
            rms, K, D, _, _ = cv2.calibrateCamera(obj_pts_list, img_pts_list, self.image_size, K, D)
        return rms, K, D

    def calibrate_if_possible(self, results: CalibrationResults):
        changed = False
        # Mono intrinsics
        if self.K0 is None or self.D0 is None:
            rms0, K0, D0 = self._mono_calibrate(0)
            if K0 is not None:
                self.K0, self.D0, self.rms0 = K0, D0, rms0
                results.K0, results.D0, results.rms0 = K0, D0, rms0
                changed = True
        if self.K1 is None or self.D1 is None:
            rms1, K1, D1 = self._mono_calibrate(1)
            if K1 is not None:
                self.K1, self.D1, self.rms1 = K1, D1, rms1
                results.K1, results.D1, results.rms1 = K1, D1, rms1
                changed = True

        # Stereo extrinsics if we have intrinsics and stereo samples
        if self.K0 is not None and self.K1 is not None and len(self.stereo_samples) >= 5:
            obj_list = [s.obj_pts for s in self.stereo_samples]
            img0_list = [s.img_pts0 for s in self.stereo_samples]
            img1_list = [s.img_pts1 for s in self.stereo_samples]
            flags = (cv2.CALIB_FIX_INTRINSIC)
            try:
                rms_st, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
                    obj_list, img0_list, img1_list,
                    self.K0.copy(), self.D0.copy(),
                    self.K1.copy(), self.D1.copy(),
                    self.image_size,
                    flags=flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6))
                results.K0, results.D0 = K0, D0
                results.K1, results.D1 = K1, D1
                results.R, results.T, results.E, results.F = R, T, E, F
                results.rms_stereo = float(rms_st)
                changed = True
            except Exception as e:
                print("[CAL] stereoCalibrate failed:", e)
        if results.image_size is None:
            results.image_size = self.image_size
        return changed

# ---------- UI helpers ----------
class Button:
    def __init__(self, label, x, y, w, h, color_idle=(50,50,50), color_active=(0,160,0)):
        self.label = label
        self.rect = (x, y, w, h)
        self.color_idle = color_idle
        self.color_active = color_active
        self.active = False

    def draw(self, img):
        x,y,w,h = self.rect
        color = self.color_active if self.active else self.color_idle
        cv2.rectangle(img, (x,y), (x+w,y+h), color, -1)
        cv2.putText(img, self.label, (x+10, y+h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    def hit(self, px, py):
        x,y,w,h = self.rect
        return (px>=x and px<=x+w and py>=y and py<=y+h)


def main():
    print("[MAIN] Opening two cameras…")
    cams = []
    target_count = min(2, len(cam_ids))
    indices = list(range(target_count))
    if target_count != len(cam_ids):
        print(f"[WARN] SDK enumerated {len(cam_ids)} device(s) but only opening {target_count}.")
    for i in indices:
        cams.append(CamReader(i))
    if not cams:
        print("[ERR] no cameras opened"); return
    print(f"[MAIN] Using {len(cams)} cam(s)")

    # Get one frame to determine image size
    ts0, f0 = cams[0].latest()
    if len(cams) > 1:
        _, f1 = cams[1].latest()
    else:
        f1 = f0
    H, W = f0.shape[0], f0.shape[1]
    image_size = (W, H)

    # AprilTag GridBoard
    dictionary = cv2.aruco.getPredefinedDictionary(APRIL_DICT)
    board = cv2.aruco.GridBoard_create(TAGS_X, TAGS_Y, TAG_SIZE_M, TAG_SEP_M, dictionary)
    acc = CalibrationAccumulator(board, image_size)
    results = CalibrationResults()
    best_stereo_rms = float("inf")
    last_recalc = 0.0
    calibrated = False

    # Pose
    pose_on = False
    estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                  for _ in cams]

    # UI buttons
    btn_cal = Button("Start Calibration", 20, 20, 240, 50)
    btn_pose = Button("Toggle Pose", 280, 20, 200, 50)

    state = {"calibrating": False}

    # Mouse callback
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_cal.hit(x, y):
                state["calibrating"] = not state["calibrating"]
                btn_cal.active = state["calibrating"]
                print(f"[UI] Calibrating -> {state['calibrating']}")
            elif btn_pose.hit(x, y):
                nonlocal pose_on, estimators
                pose_on = not pose_on
                print(f"[UI] Pose toggle -> {pose_on}")
                for i in range(len(estimators)):
                    estimators[i].stop()
                estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                              for _ in cams]

    win = "Stereo Calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            try:
                ts0, f0 = cams[0].latest()
                ts1, f1 = cams[1].latest()
            except queue.Empty:
                time.sleep(0.01)
                continue

            frames = [f0, f1]

            # Pose submission
            if HAVE_MP and pose_on:
                estimators[0].submit(ts0, f0)
                estimators[1].submit(ts1, f1)

            # Calibration accumulation
            if state["calibrating"] and (time.perf_counter() - last_recalc) > 0.05:
                g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
                g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                added = acc.accumulate_pair(g0, g1)
                if added:
                    print(f"[CAL] samples: mono0={len(acc.corners0)} mono1={len(acc.corners1)} stereo={len(acc.stereo_samples)}")

            now = time.perf_counter()
            if state["calibrating"] and (now - last_recalc) >= RECALC_INTERVAL_S and acc.enough_samples():
                last_recalc = now
                changed = acc.calibrate_if_possible(results)
                if results.rms_stereo is not None and results.rms_stereo < best_stereo_rms:
                    best_stereo_rms = results.rms_stereo
                    print(f"[CAL] New best stereo RMS: {best_stereo_rms:.3f}")
                if results.is_complete() and results.rms_stereo is not None and results.rms_stereo <= TARGET_RMS_PX:
                    calibrated = True
                    state["calibrating"] = False
                    btn_cal.active = False
                    beep_ok()
                    print("[CAL] Calibration converged ✔")

            # Draw overlays
            annotated = [fr.copy() for fr in frames]
            # Pose overlay
            if HAVE_MP and pose_on:
                for i in range(2):
                    latest = estimators[i].latest_result()
                    if latest and latest[1] is not None:
                        if 'mp_styles' in globals() and mp_styles is not None:
                            mp_drawing.draw_landmarks(
                                annotated[i], latest[1], mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
                        else:
                            mp_drawing.draw_landmarks(annotated[i], latest[1], mp_pose.POSE_CONNECTIONS)

            # UI buttons and status
            for img in annotated:
                btn_cal.draw(img)
                btn_pose.draw(img)
            status_lines = []
            if state["calibrating"]:
                status_lines.append(f"Calibrating… n0={len(acc.corners0)} n1={len(acc.corners1)} ns={len(acc.stereo_samples)}")
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
            # Put status text
            for i, img in enumerate(annotated):
                y0 = 90
                for k, line in enumerate(status_lines):
                    cv2.putText(img, line, (16, y0 + 28*k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # Side-by-side
            h1, w1 = annotated[0].shape[:2]
            h2, w2 = annotated[1].shape[:2]
            target_h = max(h1, h2)
            if h1 != target_h:
                annotated[0] = cv2.resize(annotated[0], (int(w1 * target_h / h1), target_h))
            if h2 != target_h:
                annotated[1] = cv2.resize(annotated[1], (int(w2 * target_h / h2), target_h))
            combined = cv2.hconcat(annotated)
            h,w = combined.shape[:2]
            if w > MAX_COMBINED_WIDTH:
                scale = MAX_COMBINED_WIDTH / float(w)
                combined = cv2.resize(combined, (MAX_COMBINED_WIDTH, int(round(h*scale))))

            cv2.imshow(win, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c'):
                state["calibrating"] = not state["calibrating"]
                btn_cal.active = state["calibrating"]
                print(f"[KEY] Calibrating -> {state['calibrating']}")
            elif key == ord('p'):
                pose_on = not pose_on
                print(f"[KEY] Pose toggle -> {pose_on}")
                for i in range(len(estimators)):
                    estimators[i].stop()
                estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                              for _ in cams]
            elif key == ord('s'):
                # Save calibration JSON stub and counts; easy to extend for CSV export later
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


