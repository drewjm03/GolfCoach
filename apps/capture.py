import time, queue, cv2
import threading
from . import config
import os

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
        self.ok = True
        self.fps = 0.0
        self._times = []
        self.frames_captured = 0
        # Latest frame slot protected by a lock (no Queue overhead)
        self._latest = None  # type: ignore[var-annotated]
        self._lock = threading.Lock()
        # High-FPS recording state (per-camera)
        self.recording = False
        self.record_queue = queue.Queue(maxsize=300)  # ~2.5s at 120fps
        self.record_writer = None
        self.record_thread = None
        self.record_written = 0
        self.record_dropped = 0
        self.record_start_captured = 0
        self.record_first_ts = None
        self.record_last_ts = None
        self.record_path = None
        self.record_fps = None
        self.record_gate = None      # optional threading.Event for synchronized start
        self.record_armed = False    # armed but waiting for gate to open
        self.record_ts_log = []      # list of written-frame timestamps

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

            # Count captured frames and update latest slot atomically
            self.frames_captured += 1
            with self._lock:
                self._latest = (ts, f)
                # Enqueue for recording if enabled and gate (if any) is open.
                # Use f.copy() so the backing buffer is not reused unexpectedly
                # by OpenCV/driver at high FPS.
                gate_ok = (self.record_gate is None) or (
                    getattr(self.record_gate, "is_set", lambda: True)()
                )
                if self.recording and self.record_writer is not None and gate_ok:
                    try:
                        self.record_queue.put_nowait((ts, f.copy()))
                    except queue.Full:
                        # Drop the newest frame if queue is full
                        self.record_dropped += 1

            self._times.append(ts)
            if len(self._times) > 30:
                self._times.pop(0)
            if len(self._times) >= 2:
                span = self._times[-1] - self._times[0]
                if span > 0:
                    self.fps = (len(self._times) - 1) / span

        # Ensure recording thread is joined on exit
        if self.record_thread is not None:
            self.recording = False
            self.record_thread.join()
        if self.record_writer is not None:
            try:
                self.record_writer.release()
            except Exception:
                pass

    def _record_loop(self):
        """Background thread that pulls frames from record_queue and writes them."""
        while self.recording or not self.record_queue.empty():
            try:
                ts, frame = self.record_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            writer = self.record_writer
            if writer is None:
                continue
            try:
                writer.write(frame)
                self.record_written += 1
                if self.record_first_ts is None:
                    self.record_first_ts = ts
                self.record_last_ts = ts
                # Log timestamps for offline sync analysis
                self.record_ts_log.append(float(ts))
            except Exception as e:
                print(f"[CV][REC] write failed: {e}")

    def latest(self, timeout=2.0):
        """
        Return the most recent (timestamp, frame) without queue draining.

        Blocks until a frame is available or until timeout seconds have elapsed.
        Raises queue.Empty on timeout for compatibility with existing callers.
        """
        deadline = time.perf_counter() + float(timeout)
        last = None
        while True:
            with self._lock:
                last = self._latest
            if last is not None:
                return last
            if time.perf_counter() > deadline:
                raise queue.Empty()
            time.sleep(0.005)

    def start_recording_mp4(self, path: str, fps: float, fourcc: str = "mp4v") -> bool:
        """Start recording frames to an MP4 file at the given FPS.

        Recording happens on a background thread and pulls frames from record_queue.
        """
        # Stop any existing recording first
        if self.recording:
            self.stop_recording()

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
        if not writer.isOpened():
            print(f"[CV][REC] Failed to open VideoWriter for {path}")
            return False

        self.record_queue = queue.Queue(maxsize=300)
        self.record_writer = writer
        self.record_written = 0
        self.record_dropped = 0
        # Snapshot frames captured so far to measure recording-window capture
        self.record_start_captured = int(self.frames_captured)
        self.record_first_ts = None
        self.record_last_ts = None
        self.record_ts_log = []
        self.record_path = path
        self.record_fps = float(fps)
        self.record_gate = None
        self.record_armed = False
        self.recording = True
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        print(f"[CV][REC] Started recording -> {path} @ {fps} fps")
        return True

    def arm_recording_mp4(self, path: str, fps: float, gate, fourcc: str = "mp4v") -> bool:
        """Arm recording to an MP4 file, gated by an external Event.

        Frames will begin being written once gate.is_set() is True, but
        recording starts immediately so we don't miss early frames.
        """
        # Stop any existing recording first
        if self.recording:
            self.stop_recording()

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
        if not writer.isOpened():
            print(f"[CV][REC] Failed to open VideoWriter for {path}")
            return False

        self.record_queue = queue.Queue(maxsize=300)
        self.record_writer = writer
        self.record_written = 0
        self.record_dropped = 0
        self.record_start_captured = int(self.frames_captured)
        self.record_first_ts = None
        self.record_last_ts = None
        self.record_ts_log = []
        self.record_path = path
        self.record_fps = float(fps)
        self.record_gate = gate
        self.record_armed = True
        self.recording = True
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        # Intentionally do not treat this as "GO time"; gate controls that.
        print(f"[CV][REC] Armed recording -> {path} @ {fps} fps")
        return True

    def stop_recording(self):
        """Stop recording and finalize the MP4 file, writing stats sidecar JSON."""
        if not self.recording and self.record_writer is None:
            return
        self.recording = False
        if self.record_thread is not None:
            self.record_thread.join()
            self.record_thread = None
        if self.record_writer is not None:
            try:
                self.record_writer.release()
            except Exception:
                pass
            self.record_writer = None

        # Compute stats
        captured_total = int(self.frames_captured)
        captured_window = int(self.frames_captured) - int(self.record_start_captured or 0)
        written = int(self.record_written)
        dropped = int(self.record_dropped)
        first_ts = self.record_first_ts
        last_ts = self.record_last_ts
        if written >= 2 and first_ts is not None and last_ts is not None and last_ts > first_ts:
            duration_written = float(last_ts - first_ts)
            written_fps = float((written - 1) / duration_written)
        else:
            duration_written = 0.0
            written_fps = 0.0

        print(
            f"[CV][REC] Stats: captured_total={captured_total}, "
            f"captured_during_recording={captured_window}, "
            f"written={written}, dropped={dropped}, "
            f"duration_written={duration_written:.3f}s, written_fps={written_fps:.2f}"
        )

        # Sidecar JSON
        if self.record_path:
            sidecar_path = os.path.splitext(self.record_path)[0] + "_stats.json"
            payload = {
                "captured_total": captured_total,
                "captured_during_recording": captured_window,
                "written": written,
                "dropped": dropped,
                "duration_written": duration_written,
                "written_fps": written_fps,
                "ts": list(self.record_ts_log),
            }
            try:
                import json
                with open(sidecar_path, "w", encoding="utf-8") as jf:
                    json.dump(payload, jf, indent=2)
                print(f"[CV][REC] Wrote {sidecar_path}")
            except Exception as e:
                print(f"[CV][REC] Failed to write stats JSON: {e}")

    def release(self):
        self.ok = False
        # Stop any ongoing recording cleanly
        self.stop_recording()
        time.sleep(0.05)
        self.cap.release()


