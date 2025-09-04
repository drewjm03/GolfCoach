import ctypes, cv2, time, threading, queue, numpy as np

# ---------- SDK (Extension Unit) ----------
dll = ctypes.WinDLL(r"..\sdk\See3CAM_24CUG_Extension_Unit_SDK_1.0.65.81_Windows_20220620\Win32\Binary\64Bit\HIDLibraries\eCAMFwSw.dll")

WSTR   = ctypes.c_wchar_p
BOOL   = ctypes.c_bool
UINT8  = ctypes.c_ubyte
UINT32 = ctypes.c_uint
INT32  = ctypes.c_int

dll.GetDevicesCount.argtypes = [ctypes.POINTER(UINT32)]
dll.GetDevicesCount.restype  = BOOL
dll.GetDevicePaths.argtypes  = [ctypes.POINTER(WSTR)]
dll.GetDevicePaths.restype   = BOOL

dll.InitExtensionUnit.argtypes   = [WSTR]
dll.InitExtensionUnit.restype    = BOOL
dll.DeinitExtensionUnit.argtypes = []
dll.DeinitExtensionUnit.restype  = BOOL

# Key extension-unit controls we'll use
dll.SetStreamMode24CUG.argtypes = [UINT8, UINT8]      # (mode, AFL); 0x00=Master, 0x01=Trigger
dll.SetStreamMode24CUG.restype  = BOOL
dll.SetExposureCompensation24CUG.argtypes = [INT32]   # microseconds
dll.SetExposureCompensation24CUG.restype  = BOOL
dll.SetFrameRateValue24CUG.argtypes = [UINT8]         # 60 or 120 etc (see docs)
dll.SetFrameRateValue24CUG.restype  = BOOL
dll.SetAntiFlickerMode24CUG = getattr(dll, "SetAntiFlickerMode24CUG", None)

# Enumerate
cnt = UINT32(0)
assert dll.GetDevicesCount(ctypes.byref(cnt)) and cnt.value > 0, "No See3CAM_24CUG devices"
Paths = (WSTR * cnt.value)()
assert dll.GetDevicePaths(Paths), "GetDevicePaths failed"
cam_ids = [Paths[i] for i in range(min(2, cnt.value))]

# Configure each camera once via SDK (Master + lock autos + exposure + fps)
def sdk_config(instance_id, exposure_us=3000, fps=120, lock_autos=True, anti_flicker_60hz=True):
    assert dll.InitExtensionUnit(instance_id), f"InitExtensionUnit failed: {instance_id}"
    try:
        assert dll.SetStreamMode24CUG(UINT8(0x00), UINT8(1 if lock_autos else 0)), "SetStreamMode (Master) failed"
        assert dll.SetExposureCompensation24CUG(INT32(exposure_us)), "SetExposure (us) failed"
        assert dll.SetFrameRateValue24CUG(UINT8(120 if fps >= 120 else 60)), "SetFrameRate failed"
        if dll.SetAntiFlickerMode24CUG:
            # 0x01=50Hz, 0x02=60Hz, 0x00=Auto (per SDK)
            dll.SetAntiFlickerMode24CUG(UINT8(0x02 if anti_flicker_60hz else 0x01))
    finally:
        dll.DeinitExtensionUnit()

for iid in cam_ids:
    sdk_config(iid, exposure_us=3000, fps=120, lock_autos=True, anti_flicker_60hz=True)

# ---------- OpenCV capture ----------
def open_cam(index, w=1280, h=720, fps=120, fourcc="MJPG"):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Try to ensure manual WB (OpenCV backend differences!)
    # 1) Turn off auto-WB if exposed
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    # 2) Try WB temperature first (Kelvin)
    desired_k = 4500
    ok_temp = cap.set(cv2.CAP_PROP_WB_TEMPERATURE, desired_k)
    # Some backends expose CAP_PROP_TEMPERATURE instead:
    if not ok_temp:
        cap.set(cv2.CAP_PROP_TEMPERATURE, desired_k)

    # Verify negotiated mode
    got_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    got_fps =    cap.get(cv2.CAP_PROP_FPS)
    print(f"Cam{index} negotiated: {got_w}x{got_h} @ {got_fps:.2f} ({fourcc})")

    # Light sanity check read
    ok, frm = cap.read()
    if not ok:
        print(f"WARNING: Cam{index} opened but first read failed")
    return cap

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
                time.sleep(0.001)
                continue
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

    def latest(self):
        ts, f = self.q.get()
        while not self.q.empty():
            ts, f = self.q.get_nowait()
        return ts, f

    def release(self):
        self.ok = False
        time.sleep(0.05)
        self.cap.release()

def main():
    print("Dual See3CAM_24CUG @ 720p120 (Master mode, locked autos)")
    cams = []
    try:
        for idx in range(len(cam_ids)):
            cams.append(CamReader(idx))
        if not cams:
            print("No cameras opened"); return

        # simple pairing status
        alpha, offset = 0.002, 0.0
        while True:
            ts0, f0 = cams[0].latest()
            f_show = [f0]
            status = ""
            if len(cams) > 1:
                ts1, f1 = cams[1].latest()
                obs = ts1 - ts0
                offset = (1-alpha)*offset + alpha*obs
                dt_ms = abs((ts1 - (ts0 + offset))*1000.0)
                status = f"{'SYNC' if dt_ms<=15 else 'UNSYNC'} ({dt_ms:.1f} ms)"
                f_show.append(f1)

            for i, frm in enumerate(f_show):
                cv2.putText(frm, f"Cam{i} FPS: {cams[i].fps:.1f}", (16,36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if status: cv2.putText(frm, status, (16,72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.imshow(f"cam{i}", frm)

            if cv2.waitKey(1) == 27: break
    finally:
        for c in cams: c.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
