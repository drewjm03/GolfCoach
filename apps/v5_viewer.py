import ctypes, cv2, time, threading, queue, os

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

dll.InitExtensionUnit.argtypes   = [WSTR]
dll.InitExtensionUnit.restype    = BOOL
dll.DeinitExtensionUnit.argtypes = []
dll.DeinitExtensionUnit.restype  = BOOL

dll.SetStreamMode24CUG.argtypes = [UINT8, UINT8]        # 0x00=Master, 0x01=Trigger
dll.SetStreamMode24CUG.restype  = BOOL
dll.SetFrameRateValue24CUG.argtypes = [UINT8]
dll.SetFrameRateValue24CUG.restype  = BOOL

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

# ---------- SDK: Master mode + FPS + Anti-flicker ----------
def sdk_config(instance_path, fps=120, lock_autos=True, anti_flicker_60hz=True):
    print(f"[SDK] Init: {instance_path}")
    if not dll.InitExtensionUnit(instance_path):
        raise RuntimeError("InitExtensionUnit failed")
    try:
        # Stream mode only (exposure will be set via UVC below)
        assert dll.SetStreamMode24CUG(UINT8(0x00), UINT8(1 if lock_autos else 0)), "SetStreamMode Master failed"
        assert dll.SetFrameRateValue24CUG(UINT8(120 if fps >= 120 else 60)), "SetFrameRate failed"
        if has_af and anti_flicker_60hz:
            dll.SetAntiFlickerMode24CUG(UINT8(0x02))  # 60 Hz
        print(f"[SDK] Master+AFL, FPS={fps} requested")
    finally:
        dll.DeinitExtensionUnit()
        print("[SDK] DeinitExtensionUnit")

for iid in cam_ids[:2]:
    sdk_config(iid, fps=120, lock_autos=True, anti_flicker_60hz=True)

time.sleep(0.5)  # let driver settle

# ---------- Helpers to set UVC exposure & WB correctly ----------
def set_manual_exposure_uvc(cap, step=-8):
    """
    For Windows backends: put exposure into MANUAL and set the discrete step.
    DShow quirk: 0.25=manual, 0.75=auto.
    MSMF sometimes uses 1=manual/0=auto. Try both patterns.
    e-con mapping: -8 ≈ 3.9 ms, -7 ≈ 7.8 ms, -6 ≈ 15.6 ms.  (−14..0) 50us..1s
    """
    ok = False
    # Try common patterns to force MANUAL
    for v in (0.25, 1.0, 0.0):  # DShow, MSMF, fallback
        if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v):
            ok = True
            break
    # Now set the discrete step (negative number)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(step))
    # Verify
    got = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"[CV] Exposure target step {step}, read-back {got}")
    return ok

def set_white_balance_uvc(cap, kelvin=4500):
    # Turn off auto WB (0=manual for DShow/MSMF; if ignored, try 0 then 1 then 0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    if not cap.set(cv2.CAP_PROP_WB_TEMPERATURE, kelvin):
        cap.set(cv2.CAP_PROP_TEMPERATURE, kelvin)  # other alias
    got = cap.get(cv2.CAP_PROP_WB_TEMPERATURE) or cap.get(cv2.CAP_PROP_TEMPERATURE)
    print(f"[CV] WB target {kelvin}K, read-back {got}")

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

        # --- NEW: force manual exposure to a short time (−8 ≈ 3.9ms) ---
        set_manual_exposure_uvc(cap, step=-7)   # try −7 if you need brighter

        # --- NEW: manual WB temperature ---
        set_white_balance_uvc(cap, kelvin=3950)

        # Negotiate info
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_f = cap.get(cv2.CAP_PROP_FPS)
        print(f"[CV] cam{index} negotiated: {got_w}x{got_h} @ {got_f:.2f} ({fourcc})")

        ok, _ = cap.read()
        if not ok: print(f"[WARN] cam{index} first read failed")
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

def main():
    print("[MAIN] Opening two cameras…")
    cams = []
    for i in range(min(2, len(cam_ids))):
        cams.append(CamReader(i))
    if not cams:
        print("[ERR] no cameras opened"); return
    print(f"[MAIN] Using {len(cams)} cam(s)")

    alpha, offset = 0.002, 0.0
    try:
        while True:
            ts0, f0 = cams[0].latest()
            frames = [f0]
            status = ""
            if len(cams) > 1:
                ts1, f1 = cams[1].latest()
                obs = ts1 - ts0
                offset = (1-alpha)*offset + alpha*obs
                dt_ms = abs((ts1 - (ts0 + offset))*1000.0)
                status = f"{'SYNC' if dt_ms<=15 else 'UNSYNC'} ({dt_ms:.1f} ms)"
                frames.append(f1)

            for i, fr in enumerate(frames):
                cv2.putText(fr, f"cam{i} FPS: {cams[i].fps:.1f}", (16,36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if status:
                    cv2.putText(fr, status, (16,72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.imshow(f"cam{i}", fr)
            if cv2.waitKey(1) == 27: break
    finally:
        for c in cams: c.release()
        cv2.destroyAllWindows()
        print("[MAIN] Closed")

if __name__ == "__main__":
    main()
