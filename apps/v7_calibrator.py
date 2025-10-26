import ctypes, cv2, time, threading, queue, os, sys, json
import numpy as np
try:
    from pupil_apriltags import Detector as PupilDetector
    HAVE_PUPIL = True
except Exception:
    PupilDetector = None
    HAVE_PUPIL = False

print(cv2.__version__)
print("ArucoDetector?", hasattr(cv2.aruco, "ArucoDetector"))
print("AprilTag in build?", "AprilTag" in cv2.getBuildInformation())

print("pupil_apriltags available?", HAVE_PUPIL)
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
CAPTURE_FPS = 60
MAX_COMBINED_WIDTH = 1920
FIRST_FRAME_RETRY_COUNT = 5
WB_TOGGLE_DELAY_S = 0.075
PRESERVE_NATIVE_RES = True  # if True, do not downscale combined preview; keep exact pixel size

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

# Runtime exposure/gain state
auto_exposure_on = False
current_exposure_step = DEFAULT_EXPOSURE_STEP
current_gain = None

# AprilTag grid board configuration 
APRIL_DICT = cv2.aruco.DICT_APRILTAG_36h11
TAGS_X = 8                # number of tags horizontally
TAGS_Y = 5                # number of tags vertically
TAG_SIZE_M = 0.075         # tag black square size in meters
TAG_SEP_M = 0.01875          # white gap between tags in meters

# AprilTag quality / gating
MAX_HAMMING = 0            # only perfect decodes
MIN_DECISION_MARGIN = 30   # 35–45 is good for 16h5; raise if you still see clutter
MIN_SIDE_PX = 32         # ignore tiny tags; adjust for your resolution
USE_ID_GATING = False 

# Calibration parameters/criteria
MIN_MARKERS_PER_VIEW = 8
MIN_SAMPLES = 20
TARGET_RMS_PX = 0.6
RECALC_INTERVAL_S = 1.0
CALIB_SAMPLE_PERIOD_S = 3.0  # only accumulate a new sample every N seconds

SENSOR_ROTATE_180 = False

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

def probe_tag36h11(gray):
    """Return sorted list of tag36h11 IDs seen with quality gates applied."""
    try:
        from pupil_apriltags import Detector as _PD
    except Exception:
        return []
    det = _PD(families="tag36h11", nthreads=2, quad_decimate=1.0,
              quad_sigma=0.0, refine_edges=True, decode_sharpening=0.25)
    dets = det.detect(gray, estimate_tag_pose=False) or det.detect(255 - gray, estimate_tag_pose=False)
    ids = []
    for d in (dets or []):
        if getattr(d, "hamming", 0) > MAX_HAMMING: 
            continue
        if getattr(d, "decision_margin", 0.0) < MIN_DECISION_MARGIN:
            continue
        # size gate
        c = np.array(d.corners, dtype=np.float32).reshape(4, 2)
        side = float(sum(np.linalg.norm(c[(i+1) % 4] - c[i]) for i in range(4)) / 4.0)
        if side < MIN_SIDE_PX:
            continue
        ids.append(int(d.tag_id))
    return sorted(set(ids))

def smoke_test_tag36h11(gray):
    """Very permissive single-frame test for 36h11; returns sorted IDs seen."""
    try:
        from pupil_apriltags import Detector as _PD
    except Exception:
        return []
    det = _PD(families="tag36h11", nthreads=2,
              quad_decimate=1.0, quad_sigma=0.0,
              refine_edges=True, decode_sharpening=0.25)
    dets = det.detect(gray, estimate_tag_pose=False) or det.detect(255 - gray, estimate_tag_pose=False)
    ids = []
    for d in (dets or []):
        # very lax gates: allow a little error, low margin, small tags
        if getattr(d, "hamming", 0) > 2:      # allow a bit of error
            continue
        if getattr(d, "decision_margin", 0.0) < 15:
            continue
        c = np.array(d.corners, dtype=np.float32).reshape(4, 2)
        side = float(sum(np.linalg.norm(c[(i+1)%4] - c[i]) for i in range(4)) / 4.0)
        if side < 16:                         # allow smaller tags
            continue
        ids.append(int(d.tag_id))
    return sorted(set(ids))

def probe_aruco_6x6(gray):
    """Return {'DICT_6X6_50': n, 'DICT_6X6_100': n, ...} that hit on this frame."""
    D = cv2.aruco
    results = {}
    for name, code in [("DICT_6X6_50", D.DICT_6X6_50),
                       ("DICT_6X6_100", D.DICT_6X6_100),
                       ("DICT_6X6_250", D.DICT_6X6_250),
                       ("DICT_6X6_1000", D.DICT_6X6_1000)]:
        dic = D.getPredefinedDictionary(code)
        params = D.DetectorParameters()
        try: params.detectInvertedMarker = True
        except: pass
        det = D.ArucoDetector(dic, params)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            results[name] = int(len(ids))
    return results

def draw_ids(img, corners, ids, color=(0,255,255)):
    if ids is None or len(ids) == 0: 
        return
    for c, i in zip(corners, ids):
        c4 = c.reshape(4,2).astype(int)
        p = c4.mean(axis=0).astype(int)
        cv2.putText(img, str(int(i[0])), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

def board_ids_safe(board):
    """Return Nx1 int32 array of board IDs in a version-agnostic way."""
    ids = getattr(board, "ids", None)
    if ids is None:
        # Some builds provide a getter
        try:
            ids = board.getIds()
        except Exception:
            ids = None
    if ids is None:
        # Fall back to sequential 0..N-1 based on number of markers
        try:
            N = len(board.getObjPoints())
        except Exception:
            N = 0
        ids = np.arange(N, dtype=np.int32).reshape(-1, 1)
    return np.asarray(ids, dtype=np.int32).reshape(-1, 1)


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
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.002)
                continue

            # NO mirroring here; optional safe rotation only
            if SENSOR_ROTATE_180:
                f = cv2.rotate(f, cv2.ROTATE_180)  # or: f = cv2.flip(f, -1)

            ts = time.perf_counter()

            # single-element queue: keep only latest frame
            try:
                while self.q.qsize() >= 1:
                    self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put((ts, f))

            # fps estimate over last ~30 timestamps
            self._times.append(ts)
            if len(self._times) > 30:
                self._times.pop(0)
            if len(self._times) >= 2:
                span = self._times[-1] - self._times[0]
                if span > 0:
                    self.fps = (len(self._times) - 1) / span

    def latest(self, timeout=2.0):
        ts, f = self.q.get(timeout=timeout)
        # drain any backlog so caller gets the freshest frame
        while True:
            try:
                ts, f = self.q.get_nowait()
            except queue.Empty:
                break
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
        # Prefer pupil-apriltags when available or when OpenCV build lacks AprilTag
        self._pupil = None
        self.backend_name = "OpenCV ArUco"
        if HAVE_PUPIL:
            try:
                self._pupil = PupilDetector(
                    families=self._apriltag_family_string(),
                    nthreads=2,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=True,
                    decode_sharpening=0.25,
                )
                self.backend_name = "pupil-apriltags"
            except Exception as e:
                print("[APRIL] pupil-apriltags init failed:", e)
        print(f"[APRIL] Detector backend: {self.backend_name}")

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

    def get_backend_name(self):
        return getattr(self, "backend_name", "OpenCV ArUco")

    def _make_detector(self):
        dictionary = cv2.aruco.getPredefinedDictionary(APRIL_DICT)
        params = cv2.aruco.DetectorParameters()
        # Make detection more tolerant for smaller/blurrier tags
        try:
            params.minMarkerPerimeterRate = 0.02
            params.maxMarkerPerimeterRate = 4.0
        except Exception:
            pass
        try:
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 4
        except Exception:
            pass
        try:
            params.adaptiveThreshConstant = 7
        except Exception:
            pass
        try:
            params.perspectiveRemoveIgnoredMarginPerCell = 0.2
            params.perspectiveRemovePixelPerCell = 8
        except Exception:
            pass
        try:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMinAccuracy = 0.01
        except Exception:
            pass
        try:
            params.detectInvertedMarker = True
        except Exception:
            pass
        return cv2.aruco.ArucoDetector(dictionary, params)

    def _apriltag_family_string(self):
        # Map OpenCV AprilTag dict to pupil-apriltags family string
        try:
            if APRIL_DICT == cv2.aruco.DICT_APRILTAG_36h11:
                return "tag36h11"
            if hasattr(cv2.aruco, 'DICT_APRILTAG_25h9') and APRIL_DICT == cv2.aruco.DICT_APRILTAG_25h9:
                return "tag25h9"
            if hasattr(cv2.aruco, 'DICT_APRILTAG_16h5') and APRIL_DICT == cv2.aruco.DICT_APRILTAG_16h5:
                return "tag16h5"
            if hasattr(cv2.aruco, 'DICT_APRILTAG_36h10') and APRIL_DICT == cv2.aruco.DICT_APRILTAG_36h10:
                return "tag36h10"
        except Exception:
            pass
        return "tag36h11"

    def _build_id_to_object(self):
        # GridBoard has .ids (Nx1) and .objPoints list length N with 4x3 points
        id_to_obj = {}
        try:
            ids = board_ids_safe(self.board).flatten().astype(int)
            obj_points = self.board.getObjPoints()
            for idx, tag_id in enumerate(ids):
                obj = np.array(obj_points[idx], dtype=np.float32).reshape(-1, 3)
                id_to_obj[int(tag_id)] = obj
        except Exception as e:
            # Fallback: assume sequential ids 0..N-1
            N = TAGS_X * TAGS_Y
            try:
                obj_points = self.board.getObjPoints()
                for idx in range(N):
                    obj = np.array(obj_points[idx], dtype=np.float32).reshape(-1, 3)
                    id_to_obj[idx] = obj
            except Exception as inner_e:
                print(f"[ERROR] Could not get object points: {inner_e}")
                id_to_obj = {}
        return id_to_obj

    @staticmethod
    def _avg_side_px(corners_1x4x2):
        c = corners_1x4x2.reshape(4, 2).astype(np.float32)
        return float(sum(np.linalg.norm(c[(i+1) % 4] - c[i]) for i in range(4)) / 4.0)

    def detect(self, gray):
        # Gate to only board IDs if (and only if) you constructed the board with real printed IDs
        allowed_ids = set(self.id_to_obj.keys()) if (self.id_to_obj and USE_ID_GATING) else None

        # 1) pupil-apriltags first (normal then inverted)
        if self._pupil is not None:
            for img in (gray, 255 - gray):
                try:
                    dets = self._pupil.detect(img, estimate_tag_pose=False)
                except Exception:
                    dets = []
                corners, ids = [], []
                for d in (dets or []):
                    tid = int(d.tag_id)
                    if getattr(d, "hamming", 0) > MAX_HAMMING: 
                        continue
                    if getattr(d, "decision_margin", 0.0) < MIN_DECISION_MARGIN:
                        continue
                    c = np.array(d.corners, dtype=np.float32).reshape(1, 4, 2)
                    if _avg_side_px(c) < MIN_SIDE_PX:
                        continue
                    if allowed_ids is not None and tid not in allowed_ids:
                        continue
                    corners.append(c); ids.append([tid])
                if ids:
                    ids = np.array(ids, dtype=np.int32)
                    try:
                        for c in corners:
                            cv2.cornerSubPix(gray, c.squeeze(0), (3,3), (-1,-1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.01))
                    except Exception:
                        pass
                    return corners, ids
            # fall through if none survived

        # 2) OpenCV fallback (works if your build has AprilTag or for ArUco)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return [], None

        filt_c, filt_i = [], []
        for c, i in zip(corners, ids):
            tid = int(i[0])
            if self._avg_side_px(c) < MIN_SIDE_PX:
                continue
            if allowed_ids is not None and tid not in allowed_ids:
                continue
            filt_c.append(c); filt_i.append([tid])

        if not filt_i:
            return [], None

        try:
            for c in filt_c:
                cv2.cornerSubPix(gray, c.squeeze(1), (3,3), (-1,-1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.01))
        except Exception:
            pass

        return filt_c, np.array(filt_i, dtype=np.int32)



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
    global auto_exposure_on, current_exposure_step, current_gain
    print("[MAIN] Opening two cameras…")
    cams = []
    target_count = min(2, len(cam_ids))
    # Allow overriding camera selection to avoid built-in webcam
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
        if target_count != len(cam_ids):
            print(f"[WARN] SDK enumerated {len(cam_ids)} device(s) but only opening {target_count}.")
        print(f"[INFO] Using default camera indices: {indices}. Set CAM_INDEX_ORDER (e.g., '1,2') to override.")
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
    # OpenCV >=4.7: use GridBoard class constructor with (markersX, markersY) tuple
    board = cv2.aruco.GridBoard((TAGS_X, TAGS_Y), TAG_SIZE_M, TAG_SEP_M, dictionary)
    
    print("[DBG] APRIL_DICT code:", APRIL_DICT)
    print("[DBG] Grid size:", TAGS_X, "x", TAGS_Y, " -> markers:", len(board.getObjPoints()))
    ids_dbg = board_ids_safe(board)
    print("[DBG] First 20 board IDs:", ids_dbg.flatten()[:20].tolist())
    print("[DBG] Tag size (m):", TAG_SIZE_M, "  Sep (m):", TAG_SEP_M, "  Sep/Size ratio:", TAG_SEP_M / TAG_SIZE_M)

    
    acc = CalibrationAccumulator(board, image_size)
    print("[APRIL] Backend:", acc.get_backend_name())
    print("[APRIL] Families:", acc._apriltag_family_string())

    results = CalibrationResults()
    best_stereo_rms = float("inf")
    last_recalc = 0.0
    calibrated = False
    last_sample_t = 0.0

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
                if state["calibrating"]:
                    # reset sampling timer so we capture a sample immediately
                    nonlocal last_sample_t
                    last_sample_t = 0.0
            elif btn_pose.hit(x, y):
                nonlocal pose_on, estimators
                pose_on = not pose_on
                print(f"[UI] Pose toggle -> {pose_on}")
                for i in range(len(estimators)):
                    estimators[i].stop()
                estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                              for _ in cams]

    win = "Stereo Calibrator"
    # Make window resizable while preserving aspect ratio
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except Exception:
        pass
    cv2.setMouseCallback(win, on_mouse)

    next_probe_t = 0.0

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

            now = time.perf_counter()

            # Pose submission
            if HAVE_MP and pose_on:
                estimators[0].submit(ts0, f0)
                estimators[1].submit(ts1, f1)

            # Calibration accumulation (sample once every CALIB_SAMPLE_PERIOD_S)
            if state["calibrating"] and (now - last_sample_t) >= CALIB_SAMPLE_PERIOD_S:
                added = acc.accumulate_pair(g0, g1)
                last_sample_t = now
                if added:
                    print(f"[CAL] samples: mono0={len(acc.corners0)} mono1={len(acc.corners1)} stereo={len(acc.stereo_samples)}")

            
            if state["calibrating"] and now >= next_probe_t:
                ids0 = smoke_test_tag36h11(g0)
                ids1 = smoke_test_tag36h11(g1)
                aru0 = probe_aruco_6x6(g0)
                aru1 = probe_aruco_6x6(g1)
                if aru0 or aru1:
                    print("[PROBE ArUco 6x6] cam0:", aru0, " cam1:", aru1)

                print("[SMOKE 36h11] cam0:", ids0, " cam1:", ids1)
                next_probe_t = now + 1.0


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

            # Detection overlay when calibrating: draw detected markers per view
            det_counts = (0, 0)
            if state["calibrating"]:
                try:
                    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                    g1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
                    c0, i0 = acc.detect(g0)
                    c1, i1 = acc.detect(g1)
                    if c0:
                        cv2.aruco.drawDetectedMarkers(annotated[0], c0, i0)
                        draw_ids(annotated[0], c0, i0, (0,255,255))
                    if c1:
                        cv2.aruco.drawDetectedMarkers(annotated[1], c1, i1)
                        draw_ids(annotated[1], c1, i1, (0,255,255))
                    det_counts = (len(i0) if i0 is not None else 0, len(i1) if i1 is not None else 0)
                except Exception:
                    pass

            # UI buttons and status
            for img in annotated:
                btn_cal.draw(img)
                btn_pose.draw(img)
            status_lines = []
            if state["calibrating"]:
                status_lines.append(f"Calibrating… n0={len(acc.corners0)} n1={len(acc.corners1)} ns={len(acc.stereo_samples)}")
                status_lines.append(f"Detected tags: cam0={det_counts[0]} cam1={det_counts[1]}")
            
            status_lines.append(f"Detector: {acc.get_backend_name()}")
            # Exposure/Gain readout
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
            # Put status text
            for i, img in enumerate(annotated):
                y0 = 90
                for k, line in enumerate(status_lines):
                    cv2.putText(img, line, (16, y0 + 28*k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # Stack vertically with preserved aspect ratio (align widths)
            h1, w1 = annotated[0].shape[:2]
            h2, w2 = annotated[1].shape[:2]
            target_w = max(w1, w2)
            if w1 != target_w:
                annotated[0] = cv2.resize(annotated[0], (target_w, int(round(h1 * target_w / float(w1)))))
            if w2 != target_w:
                annotated[1] = cv2.resize(annotated[1], (target_w, int(round(h2 * target_w / float(w2)))))
            combined = cv2.vconcat(annotated)
            h,w = combined.shape[:2]
            # Do not downscale when preserving native resolution; allow user to resize window
            if (not PRESERVE_NATIVE_RES) and w > MAX_COMBINED_WIDTH:
                scale = MAX_COMBINED_WIDTH / float(w)
                combined = cv2.resize(combined, (MAX_COMBINED_WIDTH, int(round(h*scale))))

            PREVIEW_MIRROR = False
            display_frames = [cv2.flip(img, 1) if PREVIEW_MIRROR else img for img in annotated]
            combined = cv2.vconcat(display_frames)
            cv2.imshow(win, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c'):
                state["calibrating"] = not state["calibrating"]
                btn_cal.active = state["calibrating"]
                print(f"[KEY] Calibrating -> {state['calibrating']}")
                if state["calibrating"]:
                    # reset sampling timer for immediate sample
                    last_sample_t = 0.0
            elif key == ord('p'):
                pose_on = not pose_on
                print(f"[KEY] Pose toggle -> {pose_on}")
                for i in range(len(estimators)):
                    estimators[i].stop()
                estimators = [PoseEstimator(enable=pose_on, model_complexity=1, inference_width=640, inference_fps=30)
                              for _ in cams]
            elif key == ord('e'):
                # Toggle auto/manual exposure via UVC only (SDK locked unless USE_SDK_EXPOSURE)
                auto_exposure_on = not auto_exposure_on
                print(f"[KEY] Auto exposure -> {auto_exposure_on}")
                if auto_exposure_on:
                    for c in cams:
                        set_auto_exposure_uvc(c.cap)
                else:
                    for c in cams:
                        set_manual_exposure_uvc(c.cap, step=current_exposure_step)
            elif key == ord(',') or key == 44:  # decrease exposure
                if not auto_exposure_on:
                    current_exposure_step = max(MIN_EXPOSURE_STEP, int(current_exposure_step) - 1)
                    for c in cams:
                        set_manual_exposure_uvc(c.cap, step=current_exposure_step)
                    print(f"[KEY] Exposure step -> {current_exposure_step}")
            elif key == ord('.') or key == 46:  # increase exposure
                if not auto_exposure_on:
                    current_exposure_step = min(MAX_EXPOSURE_STEP, int(current_exposure_step) + 1)
                    for c in cams:
                        set_manual_exposure_uvc(c.cap, step=current_exposure_step)
                    print(f"[KEY] Exposure step -> {current_exposure_step}")
            elif key == ord(';') or key == 59:  # decrease gain
                try:
                    if current_gain is None:
                        current_gain = float(cams[0].cap.get(cv2.CAP_PROP_GAIN))
                    current_gain = max(MIN_GAIN, current_gain - GAIN_DELTA)
                    for c in cams:
                        set_uvc_gain(c.cap, current_gain)
                    print(f"[KEY] Gain -> {current_gain}")
                except Exception as e:
                    print("[KEY] Gain decrease failed:", e)
            elif key == ord("'") or key == 39:  # increase gain
                try:
                    if current_gain is None:
                        current_gain = float(cams[0].cap.get(cv2.CAP_PROP_GAIN))
                    current_gain = min(MAX_GAIN, current_gain + GAIN_DELTA)
                    for c in cams:
                        set_uvc_gain(c.cap, current_gain)
                    print(f"[KEY] Gain -> {current_gain}")
                except Exception as e:
                    print("[KEY] Gain increase failed:", e)
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


