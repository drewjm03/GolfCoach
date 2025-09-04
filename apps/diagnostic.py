import ctypes, cv2, time, threading, queue, os

# --------- Resolve DLL absolutely (adjust base_dir if needed) ---------
base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.normpath(os.path.join(
    base_dir,
    "..",
    "sdk",
    "See3CAM_24CUG_Extension_Unit_SDK_1.0.65.81_Windows_20220620",
    "Win32",
    "Binary",
    "64Bit",
    "HIDLibraries",
    "eCAMFwSw.dll",
))
print(f"[INFO] DLL path: {dll_path}")
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"SDK DLL not found at: {dll_path}")

# --------- SDK setup (Unicode signatures) ---------
dll   = ctypes.WinDLL(dll_path)
WSTR  = ctypes.c_wchar_p
BOOL  = ctypes.c_bool
UINT8 = ctypes.c_ubyte
UINT32= ctypes.c_uint
INT32 = ctypes.c_int

dll.GetDevicesCount.argtypes = [ctypes.POINTER(UINT32)]
dll.GetDevicesCount.restype  = BOOL
dll.GetDevicePaths.argtypes  = [ctypes.POINTER(WSTR)]   # array of WCHAR*
dll.GetDevicePaths.restype   = BOOL

dll.InitExtensionUnit.argtypes   = [WSTR]               # WCHAR*
dll.InitExtensionUnit.restype    = BOOL
dll.DeinitExtensionUnit.argtypes = []
dll.DeinitExtensionUnit.restype  = BOOL

dll.SetStreamMode24CUG.argtypes = [UINT8, UINT8]        # (mode, AFL)
dll.SetStreamMode24CUG.restype  = BOOL
dll.SetExposureCompensation24CUG.argtypes = [INT32]     # microseconds
dll.SetExposureCompensation24CUG.restype  = BOOL
dll.SetFrameRateValue24CUG.argtypes = [UINT8]           # 60 or 120
dll.SetFrameRateValue24CUG.restype  = BOOL

# Optional anti-flicker (SDK may or may not export it)
try:
    _SetAF = dll.SetAntiFlickerMode24CUG
    _SetAF.argtypes = [UINT8]                           # 0=Auto, 1=50Hz, 2=60Hz
    _SetAF.restype  = BOOL
except AttributeError:
    _SetAF = None

# --------- Enumerate devices (SAFE allocation) ---------
MAX_PATH = 260

cnt = UINT32(0)
assert dll.GetDevicesCount(ctypes.byref(cnt)), "GetDevicesCount failed"
print(f"[INFO] SDK sees {cnt.value} See3CAM_24CUG device(s)")
if cnt.value == 0:
    raise SystemExit("No cameras found by SDK")

# Allocate one wchar buffer per device, and pass an array of pointers to those buffers
wbufs = [ctypes.create_unicode_buffer(MAX_PATH) for _ in range(cnt.value)]
Paths = (WSTR * cnt.value)()
for i, b in enumerate(wbufs):
    Paths[i] = ctypes.cast(b, WSTR)   # pointer to wchar buffer

ok = dll.GetDevicePaths(Paths)
assert ok, "GetDevicePaths failed"

cam_ids = [wbufs[i].value for i in range(cnt.value)]
for i, pid in enumerate(cam_ids):
    print(f"[INFO] SDK device {i} instance path: {pid}")


# --------- Configure each cam via SDK (Master, exposure µs, fps) ---------
def sdk_config(instance_path, exposure_us=3000, fps=120, lock_autos=True, anti_flicker_60hz=True):
    print(f"[SDK] Init: {instance_path}")
    if not dll.InitExtensionUnit(instance_path):
        raise RuntimeError("InitExtensionUnit failed")
    try:
        if not dll.SetStreamMode24CUG(UINT8(0x00), UINT8(1 if lock_autos else 0)):  # 0x00 Master
            raise RuntimeError("SetStreamMode24CUG(Master) failed")
        if not dll.SetExposureCompensation24CUG(INT32(exposure_us)):
            raise RuntimeError("SetExposureCompensation24CUG failed")
        if not dll.SetFrameRateValue24CUG(UINT8(120 if fps >= 120 else 60)):
            raise RuntimeError("SetFrameRateValue24CUG failed")
        if _SetAF:
            _SetAF(UINT8(0x02 if anti_flicker_60hz else 0x01))
        print(f"[SDK] Master+AFL, Exposure={exposure_us}us, FPS={fps} (requested)")
    finally:
        dll.DeinitExtensionUnit()
        print("[SDK] DeinitExtensionUnit")

# Configure up to 2 cams
for iid in cam_ids[:2]:
    sdk_config(iid, exposure_us=3000, fps=120, lock_autos=True, anti_flicker_60hz=True)

# Small settle time for the driver after mode change
time.sleep(0.5)

# --------- OpenCV capture helpers ---------
def open_cv_cam(index, w=1280, h=720, fps=120, fourcc="MJPG"):
    # Try DShow then MSMF
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"[CV] cam{index} opened with backend {backend}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Turn off auto WB, try to set WB temperature (Kelvin)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            desired_k = 4500
            ok_wb = cap.set(cv2.CAP_PROP_WB_TEMPERATURE, desired_k)
            if not ok_wb:
                cap.set(cv2.CAP_PROP_TEMPERATURE, desired_k)

            # Verify negotiated
            got_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            got_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            got_fps =     cap.get(cv2.CAP_PROP_FPS)
            print(f"[CV] cam{index} negotiated: {got_w}x{got_h} @ {got_fps:.2f} ({fourcc})")

            # Sanity read
            ok, frm = cap.read()
            if not ok:
                print(f"[WARN] cam{index} opened but first read failed")
            return cap
        else:
            print(f"[CV] cam{index} failed with backend {backend}")
    raise RuntimeError(f"Could not open cam index {index} with DShow/MSMF")

class CamReader:
    def __init__(self, index):
        self.cap = open_cv_cam(index)
        self.q = queue.Queue(maxsize=1)
        self.ok = True
        self.fps = 0.0
        self._times = []
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.ok:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.002)
                continue
            ts = time.perf_counter()
            if self.q.full():
                try: self.q.get_nowait()
                except: pass
            self.q.put((ts, frame))
            self._times.append(ts)
            if len(self._times) > 30: self._times.pop(0)
            if len(self._times) >= 2:
                span = self._times[-1] - self._times[0]
                if span > 0: self.fps = (len(self._times)-1)/span

    def latest(self, timeout=2.0):
        item = self.q.get(timeout=timeout)
        ts, frame = item
        while not self.q.empty():
            ts, frame = self.q.get_nowait()
        return ts, frame

    def release(self):
        self.ok = False
        self.t.join(timeout=0.5)
        self.cap.release()

# Probe indices 0..5 to find up to 2 working cameras
def find_two_cams(max_index=5):
    found = []
    for idx in range(max_index+1):
        try:
            cr = CamReader(idx)
            # wait briefly for first frame
            try:
                _ = cr.latest(timeout=1.5)
                found.append(cr)
                print(f"[OK] cam{idx} streaming")
            except Exception:
                print(f"[WARN] cam{idx} opened but no frames yet")
                found.append(cr)
            if len(found) == 2: break
        except Exception as e:
            print(f"[SKIP] cam{idx}: {e}")
    return found

def main():
    print("[MAIN] Opening up to two cameras...")
    cams = find_two_cams()
    if not cams:
        print("[ERR] No streaming cameras found")
        return
    print(f"[MAIN] Using {len(cams)} camera(s)")

    # pairing (optional)
    alpha, offset = 0.002, 0.0
    try:
        while True:
            ts0, f0 = cams[0].latest(timeout=3.0)
            frames = [f0]
            status = ""
            if len(cams) > 1:
                ts1, f1 = cams[1].latest(timeout=3.0)
                # refine offset and compute skew
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
