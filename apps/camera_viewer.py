import cv2, threading, queue, time

class CamReader:
    def __init__(self, index, width=1280, height=720, fps=60, fourcc="MJPG", backend=cv2.CAP_DSHOW):
        self.cap = cv2.VideoCapture(index, backend)
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {index}")
            
        # Set format (best-effort; some drivers ignore FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Reduce buffering (some backends ignore this on Windows, but ask anyway)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.q = queue.Queue(maxsize=1)   # drop-old behavior
        self.tsq = queue.Queue(maxsize=1)
        self.ok = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.ok:
            ok, frame = self.cap.read()  # will block until the next trigger delivers a frame
            ts = time.perf_counter()
            if not ok:
                continue
            # drop old frame if consumer is behind
            if self.q.full():
                try: self.q.get_nowait()
                except: pass
            if self.tsq.full():
                try: self.tsq.get_nowait()
                except: pass
            self.q.put(frame)
            self.tsq.put(ts)

    def get(self):
        # get most recent frame (blocks until first triggered frame arrives)
        frame = self.q.get()
        ts = self.tsq.get()
        return frame, ts

    def release(self):
        self.ok = False
        self.t.join(timeout=0.5)
        self.cap.release()

# --- Open cameras (pick the correct indices for your system) ---
cameras = []
camera_names = []

# Try to open camera 0
try:
    cam0 = CamReader(0)
    cameras.append(cam0)
    camera_names.append("cam0")
    print("Camera 0 opened successfully")
except RuntimeError as e:
    print(f"Camera 0 not available: {e}")
    cam0 = None

# Try to open camera 1
try:
    cam1 = CamReader(1)
    cameras.append(cam1)
    camera_names.append("cam1")
    print("Camera 1 opened successfully")
except RuntimeError as e:
    print(f"Camera 1 not available: {e}")
    cam1 = None

if not cameras:
    print("No cameras available. Exiting.")
    exit()

print(f"Running with {len(cameras)} camera(s)")

try:
    while True:
        frames = []
        timestamps = []
        
        # Get frames from all available cameras
        for i, cam in enumerate(cameras):
            frame, ts = cam.get()
            frames.append(frame)
            timestamps.append(ts)

        # Optional: check inter-camera arrival skew (if multiple cameras)
        if len(cameras) > 1:
            skew_ms = (timestamps[1] - timestamps[0]) * 1000.0
            # print(f"Skew: {skew_ms:.2f} ms")

        # ---- run your keypoint model here on frames ----
        # keypoints = [model(frame) for frame in frames]

        # Display all available cameras
        for i, (frame, name) in enumerate(zip(frames, camera_names)):
            cv2.imshow(name, frame)
            
        if cv2.waitKey(1) == 27:
            break
finally:
    # Release all cameras
    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()
