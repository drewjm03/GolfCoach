import time, queue, cv2
import numpy as np

HAVE_RTM, RTM_IMPORT_ERR = False, ""
try:
    from rtmlib import Wholebody
    HAVE_RTM = True
except Exception as _rtm_err:
    RTM_IMPORT_ERR = str(_rtm_err)
    Wholebody = None

# RTMPose COCO format keypoint connections (17 keypoints)
# Keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
# 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
# 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
# 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
RTM_POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

class RTMLandmarks:
    """Wrapper to make RTMPose results compatible with MediaPipe landmarks interface."""
    def __init__(self, keypoints, scores=None):
        # keypoints: list of (x, y) tuples or numpy array of shape (N, 2)
        self.keypoints = np.asarray(keypoints, dtype=np.float32) if keypoints is not None else None
        self.scores = np.asarray(scores, dtype=np.float32) if scores is not None else None
        # Create landmark objects for compatibility
        self.landmark = []
        if self.keypoints is not None:
            for i, (x, y) in enumerate(self.keypoints):
                lm = type('Landmark', (), {
                    'x': float(x),
                    'y': float(y),
                    'z': 0.0,
                    'visibility': float(self.scores[i]) if self.scores is not None and i < len(self.scores) else 1.0
                })()
                self.landmark.append(lm)

class PoseEstimator:
    def __init__(self, enable=True, model_complexity=1, inference_width=480, inference_fps=30):
        self.enabled = enable and HAVE_RTM
        self.model_complexity = model_complexity
        self.inference_width = int(inference_width)
        self.target_period = 1.0 / max(1, int(inference_fps))
        self._q = queue.Queue(maxsize=1)
        self._latest = None
        self._stop = False
        self._model = None
        self._thread = None
        if self.enabled:
            try:
                # Initialize RTM model according to rtmlib README
                # backend can be 'onnxruntime' or 'openvino'
                # mode can be 'balanced', 'performance', or 'lightweight'
                self._model = Wholebody(
                    to_openpose=False,
                    mode="lightweight",
                    backend="onnxruntime",
                    device="cpu"  # Can be changed to "cuda" if GPU available
                )
            except Exception as e:
                print(f"[POSE] Failed to initialize RTM model: {e}")
                print(f"[POSE] Error type: {type(e).__name__}")
                self.enabled = False
            if self.enabled:
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
            original_shape = (h, w)
            if self.inference_width and w > 0:
                scale = self.inference_width / float(w)
                if scale > 0 and abs(scale - 1.0) > 1e-3:
                    nh = int(round(h * scale))
                    frame_bgr = cv2.resize(frame_bgr, (self.inference_width, nh))
            try:
                while True:
                    ts_latest, frame_latest = self._q.get_nowait()
                    ts, frame_bgr = ts_latest, frame_latest
                    original_shape = frame_bgr.shape[:2]
            except queue.Empty:
                pass
            
            try:
                # ---- rtmlib call: Wholebody(img) -> (keypoints, scores) ----
                # keypoints: (num_person, num_kpts, 2 or 3)
                # scores:    (num_person, num_kpts) or similar
                keypoints, scores = self._model(frame_bgr)

                pose_landmarks = None
                pose_world_landmarks = None

                # If nothing detected
                if keypoints is None:
                    self._latest = (ts, None, None)
                    continue

                keypoints = np.asarray(keypoints, dtype=np.float32)
                scores = np.asarray(scores, dtype=np.float32) if scores is not None else None

                # Handle multi-person: just keep the first person for now
                if keypoints.ndim == 3:
                    # (num_person, num_kpts, D)
                    keypoints = keypoints[0]          # (num_kpts, D)
                    if scores is not None and scores.ndim == 2:
                        scores = scores[0]            # (num_kpts,)

                # Now we expect keypoints shape ~ (num_kpts, 2 or 3)
                if keypoints.ndim != 2 or keypoints.shape[0] == 0:
                    self._latest = (ts, None, None)
                    continue

                # Separate xy and optional per-keypoint scores
                if keypoints.shape[1] >= 2:
                    kp_xy = keypoints[:, :2]          # pixel coords
                else:
                    kp_xy = None

                # Prefer explicit scores array if present, otherwise use 3rd column if exists
                if scores is not None and scores.shape[0] == kp_xy.shape[0]:
                    kp_scores = scores
                elif keypoints.shape[1] >= 3:
                    kp_scores = keypoints[:, 2]
                else:
                    kp_scores = None

                if kp_xy is None:
                    self._latest = (ts, None, None)
                    continue

                h_inf, w_inf = frame_bgr.shape[:2]

                # Normalized coordinates [0,1] for compatibility with your drawing utils
                keypoints_norm = kp_xy.copy()
                keypoints_norm[:, 0] /= float(w_inf)
                keypoints_norm[:, 1] /= float(h_inf)

                pose_landmarks = RTMLandmarks(keypoints_norm, kp_scores)
                # "world" landmarks: here just pixel coords; true 3D will come later
                pose_world_landmarks = RTMLandmarks(kp_xy, kp_scores)

                self._latest = (ts, pose_landmarks, pose_world_landmarks)

            except Exception as e:
                # Log once per error type to help debugging
                print(f"[POSE] Inference error: {type(e).__name__}: {e}")
                self._latest = (ts, None, None)

    def stop(self):
        self._stop = True
        # RTM model cleanup if needed
        if self._model is not None:
            # juxtapose RTM doesn't have explicit close, but we can set to None
            self._model = None
        if self._thread is not None:
            try:
                self._thread.join(0.2)
            except RuntimeError:
                pass


def draw_rtm_landmarks(image, landmarks, connections=None, landmark_color=(0, 255, 0), connection_color=(0, 255, 0)):
    """Draw RTM pose landmarks on an image."""
    if landmarks is None or landmarks.keypoints is None:
        return image
    
    h, w = image.shape[:2]
    keypoints = landmarks.keypoints
    
    # Denormalize coordinates if they're in [0, 1] range
    if keypoints.shape[0] > 0 and np.all(keypoints[:, 0] <= 1.0) and np.all(keypoints[:, 1] <= 1.0):
        keypoints_px = keypoints.copy()
        keypoints_px[:, 0] = keypoints_px[:, 0] * w
        keypoints_px[:, 1] = keypoints_px[:, 1] * h
    else:
        keypoints_px = keypoints
    
    # Draw connections
    if connections is None:
        connections = RTM_POSE_CONNECTIONS
    
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints_px) and end_idx < len(keypoints_px):
            pt1 = tuple(keypoints_px[start_idx].astype(int))
            pt2 = tuple(keypoints_px[end_idx].astype(int))
            cv2.line(image, pt1, pt2, connection_color, 2)
    
    # Draw landmarks
    for i, kp in enumerate(keypoints_px):
        pt = tuple(kp.astype(int))
        cv2.circle(image, pt, 5, landmark_color, -1)
        cv2.circle(image, pt, 5, (0, 0, 0), 1)
    
    return image


