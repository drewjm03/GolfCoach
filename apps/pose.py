import time, queue, cv2

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
    mp = None
    mp_pose = None
    mp_drawing = None
    mp_styles = None

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
            import threading
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
            self._latest = (ts, result.pose_landmarks, getattr(result, 'pose_world_landmarks', None))

    def stop(self):
        self._stop = True
        if self._pose is not None:
            self._pose.close()
        if self._thread is not None:
            try:
                self._thread.join(0.2)
            except RuntimeError:
                pass


