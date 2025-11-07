import time, queue, cv2
from . import config

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
    time.sleep(config.WB_TOGGLE_DELAY_S)
    if not cap.set(cv2.CAP_PROP_WB_TEMPERATURE, kelvin):
        cap.set(cv2.CAP_PROP_TEMPERATURE, kelvin)
    got = cap.get(cv2.CAP_PROP_WB_TEMPERATURE) or cap.get(cv2.CAP_PROP_TEMPERATURE)
    if not got or (isinstance(got, (int, float)) and abs(float(got) - float(kelvin)) > 50):
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        time.sleep(config.WB_TOGGLE_DELAY_S)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        time.sleep(config.WB_TOGGLE_DELAY_S)
        if not cap.set(cv2.CAP_PROP_WB_TEMPERATURE, kelvin):
            cap.set(cv2.CAP_PROP_TEMPERATURE, kelvin)

def set_uvc_gain(cap, gain):
    try:
        cap.set(cv2.CAP_PROP_GAIN, float(gain))
    except Exception:
        pass

def open_cam(index, w=1280, h=720, fps=config.CAPTURE_FPS, fourcc="MJPG"):
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
        if config.USE_SDK_EXPOSURE:
            set_auto_exposure_uvc(cap)
        else:
            set_manual_exposure_uvc(cap, step=config.DEFAULT_EXPOSURE_STEP)
        try:
            set_uvc_gain(cap, 6.0)
        except Exception:
            pass
        set_white_balance_uvc(cap, kelvin=config.DEFAULT_WB_KELVIN)

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
        for _ in range(config.FIRST_FRAME_RETRY_COUNT):
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
        self._thread = None
        import threading
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.002)
                continue

            if config.SENSOR_ROTATE_180:
                f = cv2.rotate(f, cv2.ROTATE_180)

            ts = time.perf_counter()

            try:
                while self.q.qsize() >= 1:
                    self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put((ts, f))

            self._times.append(ts)
            if len(self._times) > 30:
                self._times.pop(0)
            if len(self._times) >= 2:
                span = self._times[-1] - self._times[0]
                if span > 0:
                    self.fps = (len(self._times) - 1) / span

    def latest(self, timeout=2.0):
        ts, f = self.q.get(timeout=timeout)
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


