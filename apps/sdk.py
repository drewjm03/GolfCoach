import os, ctypes, time

def try_load_dll(base_dir):
    dll_path = os.path.normpath(os.path.join(
        base_dir, "..", "sdk",
        "See3CAM_24CUG_Extension_Unit_SDK_1.0.65.81_Windows_20220620",
        "Win32", "Binary", "64Bit", "HIDLibraries", "eCAMFwSw.dll"))
    print(f"[INFO] DLL path: {dll_path}")
    if not os.path.exists(dll_path):
        print("[INFO] eCAMFwSw.dll not found; running in UVC-only mode")
        return None, False, []
    try:
        dll = ctypes.WinDLL(dll_path)

        WSTR  = ctypes.c_wchar_p
        BOOL  = ctypes.c_bool
        UINT8 = ctypes.c_ubyte
        UINT32= ctypes.c_uint

        dll.GetDevicesCount.argtypes = [ctypes.POINTER(UINT32)]
        dll.GetDevicesCount.restype  = BOOL
        dll.GetDevicePaths.argtypes  = [ctypes.POINTER(WSTR)]
        dll.GetDevicePaths.restype   = BOOL

        dll.InitExtensionUnit.argtypes   = [WSTR]
        dll.InitExtensionUnit.restype    = BOOL
        dll.DeinitExtensionUnit.argtypes = []
        dll.DeinitExtensionUnit.restype  = BOOL

        dll.SetStreamMode24CUG.argtypes = [UINT8, UINT8]
        dll.SetStreamMode24CUG.restype  = BOOL
        dll.SetFrameRateValue24CUG.argtypes = [UINT8]
        dll.SetFrameRateValue24CUG.restype  = BOOL
        dll.SetExposureCompensation24CUG.argtypes = [ctypes.c_uint]
        dll.SetExposureCompensation24CUG.restype  = BOOL

        has_af = False
        try:
            dll.SetAntiFlickerMode24CUG.argtypes = [UINT8]
            dll.SetAntiFlickerMode24CUG.restype  = BOOL
            has_af = True
        except AttributeError:
            has_af = False

        MAX_PATH = 260
        cnt = UINT32(0)
        if not dll.GetDevicesCount(ctypes.byref(cnt)):
            print("[WARN] GetDevicesCount failed")
            return dll, has_af, []
        print(f"[INFO] SDK sees {cnt.value} See3CAM_24CUG device(s)")
        cam_ids = []
        if cnt.value > 0:
            wbufs = [ctypes.create_unicode_buffer(MAX_PATH) for _ in range(cnt.value)]
            Paths = (WSTR * cnt.value)()
            for i, b in enumerate(wbufs):
                Paths[i] = ctypes.cast(b, WSTR)
            if dll.GetDevicePaths(Paths):
                cam_ids = [b.value for b in wbufs]
                for i, pid in enumerate(cam_ids):
                    print(f"[INFO] SDK device {i} instance path: {pid}")
        return dll, has_af, cam_ids
    except Exception as e:
        print(f"[WARN] SDK initialization failed: {e}. Falling back to UVC-only mode.")
        return None, False, []

def max_exposure_us_for_fps(fps, safety_us=300):
    period_us = int(round(1_000_000 / max(1, int(fps))))
    return max(50, period_us - int(safety_us))

def sdk_config(dll, has_af, instance_path, fps=120, lock_autos=True, anti_flicker_60hz=True, exposure_us=None):
    if dll is None:
        return False
    print(f"[SDK] Init: {instance_path}")
    inited = False
    try:
        if not dll.InitExtensionUnit(instance_path):
            print("[SDK] ERROR: InitExtensionUnit failed"); return False
        inited = True
        assert dll.SetStreamMode24CUG(ctypes.c_ubyte(0x00), ctypes.c_ubyte(1 if lock_autos else 0))
        assert dll.SetFrameRateValue24CUG(ctypes.c_ubyte(120 if fps >= 120 else 60))
        if has_af and anti_flicker_60hz:
            try:
                if not dll.SetAntiFlickerMode24CUG(ctypes.c_ubyte(0x02)):
                    print("[SDK] WARNING: SetAntiFlickerMode24CUG(60Hz) failed")
            except Exception:
                pass
        if exposure_us is not None:
            limit = max_exposure_us_for_fps(fps, safety_us=300)
            target_us = min(int(exposure_us), limit)
            if not dll.SetExposureCompensation24CUG(ctypes.c_uint(target_us)):
                print("[SDK] WARNING: SetExposureCompensation24CUG failed")
        return True
    finally:
        if inited:
            dll.DeinitExtensionUnit()
            print("[SDK] DeinitExtensionUnit")


