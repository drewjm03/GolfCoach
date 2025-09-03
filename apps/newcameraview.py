import cv2, time, threading, queue
import numpy as np
import ctypes
import os

# load DLL
dll = ctypes.WinDLL(r"../sdk/See3CAM_24CUG_Extension_Unit_SDK_1.0.65.81_Windows_20220620/Win32/Binary/64Bit/HIDLibraries/eCAMFwSw.dll")

# Define types
UINT8 = ctypes.c_ubyte
UINT32 = ctypes.c_uint32
BOOL = ctypes.c_bool

# Set up function signatures with proper calling convention
dll.GetDevicesCount.argtypes = [ctypes.POINTER(UINT32)]
dll.GetDevicesCount.restype = BOOL

dll.GetDevicePaths.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
dll.GetDevicePaths.restype = BOOL

dll.InitExtensionUnit.argtypes = [ctypes.POINTER(ctypes.c_char)]
dll.InitExtensionUnit.restype = BOOL

dll.SetStreamMode24CUG.argtypes = [UINT8, UINT8]
dll.SetStreamMode24CUG.restype = BOOL

# Initialize the extension unit
print("Initializing camera extension unit...")

# Get device count
device_count = UINT32(0)
if not dll.GetDevicesCount(ctypes.byref(device_count)):
    print("ERROR: Failed to get device count!")
    exit(1)

print(f"Found {device_count.value} device(s)")

if device_count.value == 0:
    print("ERROR: No devices found!")
    exit(1)

# Get device paths - allocate memory properly
device_paths = (ctypes.POINTER(ctypes.c_char) * device_count.value)()
for i in range(device_count.value):
    # Allocate MAX_PATH (260) bytes for each device path
    device_paths[i] = ctypes.cast(ctypes.create_string_buffer(260), ctypes.POINTER(ctypes.c_char))

if not dll.GetDevicePaths(device_paths):
    print("ERROR: Failed to get device paths!")
    exit(1)

print("Device paths retrieved successfully")

# Initialize with the first device
device_path = device_paths[0]
device_path_str = ctypes.string_at(device_path)
print(f"Initializing with device path: {device_path_str}")
if not dll.InitExtensionUnit(device_path):
    print("ERROR: Failed to initialize extension unit!")
    exit(1)

print("Extension unit initialized successfully")

# Set stream mode to Master with AFL lock
# 0x00 = Master, 0x01 = Trigger
# AFL = 1 locks auto functions
result = dll.SetStreamMode24CUG(UINT8(0x00), UINT8(0x01))
print(f"SetStreamMode24CUG result: {result}")
if not result:
    print("WARNING: Failed to set stream mode to Master!")
else:
    print("Successfully set to Master mode with AFL lock")

# Give cameras time to start streaming after mode change
print("Waiting for cameras to start streaming...")
time.sleep(2)

class CamReader:
    def __init__(self, index, w=1280, h=720, fps=120, fourcc="MJPG", backend=cv2.CAP_DSHOW):
        # FPS tracking
        self.frame_times = []
        self.last_fps_update = time.perf_counter()
        self.current_fps = 0.0
        # Try to open camera with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            print(f"Attempting to open camera {index} (attempt {attempt + 1}/{max_retries})...")
            
            self.cap = cv2.VideoCapture(index, backend)
            if not self.cap.isOpened():
                print(f"  DirectShow failed, trying Media Foundation...")
                self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            
            if self.cap.isOpened():
                print(f"  Camera {index} opened successfully")
                break
            else:
                print(f"  Attempt {attempt + 1} failed")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    raise RuntimeError(f"Could not open camera {index} after {max_retries} attempts")

        # Configure camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set manual camera controls
        print(f"Setting manual controls for camera {index}...")
        
        # Disable auto exposure (0 = manual, 1 = auto)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        print(f"  Auto exposure disabled for camera {index}")
        
        # Disable auto white balance
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        print(f"  Auto white balance disabled for camera {index}")
        
        # Set specific manual values
        exposure_time = 5000  # 8.34ms in microseconds
        gain_value = 5
        wb_temperature = 2950  # 2950K
        
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_time)
        self.cap.set(cv2.CAP_PROP_GAIN, gain_value)
        self.cap.set(cv2.CAP_PROP_TEMPERATURE, wb_temperature)
        
        print(f"  Camera {index} settings: Exp={exposure_time}μs, Gain={gain_value}, WB={wb_temperature}K")

        # verify negotiated
        print(f"Cam{index}: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
              f"{self.cap.get(cv2.CAP_PROP_FPS):.2f}")

        # Test frame reading
        ret, test_frame = self.cap.read()
        if not ret:
            print(f"WARNING: Camera {index} opened but cannot read frames!")
        else:
            print(f"Camera {index} test frame: {test_frame.shape}")

        self.q = queue.Queue(maxsize=1)  # (t, frame)
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        frame_count = 0
        failed_count = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                failed_count += 1
                if failed_count % 300 == 0:  # Print every 300 failures (much less frequent)
                    print(f"Camera {self.cap.get(cv2.CAP_PROP_POS_FRAMES)}: Failed to read frame {failed_count} times")
                continue
            frame_count += 1
            if frame_count % 300 == 0:  # Print every 300 frames (much less frequent)
                print(f"Camera {self.cap.get(cv2.CAP_PROP_POS_FRAMES)}: Frame {frame_count}, Frame shape: {frame.shape if frame is not None else 'None'}")
            
            # Update FPS calculation
            current_time = time.perf_counter()
            self.frame_times.append(current_time)
            
            # Keep only last 30 frame times for FPS calculation
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            # Calculate FPS every second
            if current_time - self.last_fps_update >= 1.0:
                if len(self.frame_times) >= 2:
                    time_span = self.frame_times[-1] - self.frame_times[0]
                    if time_span > 0:
                        self.current_fps = (len(self.frame_times) - 1) / time_span
                self.last_fps_update = current_time
            
            ts = time.perf_counter()
            if self.q.full():
                try: self.q.get_nowait()
                except: pass
            self.q.put((ts, frame))

    def latest(self):
        # block until one is available, then drain to latest
        ts, frame = self.q.get()
        while not self.q.empty():
            ts, frame = self.q.get_nowait()
        return ts, frame

    def release(self):
        self.running = False
        self.t.join(timeout=0.5)
        self.cap.release()

def pair_frames(ts0, f0, ts1, f1, tol_ms=15.0):
    dt_ms = abs((ts1 - ts0) * 1000.0)
    return (dt_ms <= tol_ms), dt_ms

def main():
    print("Initializing cameras...")
    try:
        cam0 = CamReader(0)   # pick the right indices
        print("Camera 0 initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize camera 0: {e}")
        return
        
    try:
        cam1 = CamReader(1)
        print("Camera 1 initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize camera 1: {e}")
        cam0.release()
        return
    try:
        alpha = 0.001  # EMA smoothing for offset (optional)
        offset = 0.0   # ts1 ≈ ts0 + offset (+ drift if you add it)

        frame_count = 0
        while True:
            t0, f0 = cam0.latest()
            t1, f1 = cam1.latest()
            
            frame_count += 1
            if frame_count % 300 == 0:  # Print every 300 frames (much less frequent)
                print(f"Main loop: Frame {frame_count}")
                print(f"  f0 shape: {f0.shape if f0 is not None else 'None'}")
                print(f"  f1 shape: {f1.shape if f1 is not None else 'None'}")
                print(f"  f0 mean: {np.mean(f0) if f0 is not None else 'None'}")
                print(f"  f1 mean: {np.mean(f1) if f1 is not None else 'None'}")

            # simple offset refine (optional): move offset toward observed delta
            obs = (t1 - t0)
            offset = (1 - alpha)*offset + alpha*obs

            ok, dt_ms = pair_frames(t0, f0, t1 - offset, f1, tol_ms=15.0)
            if not ok:
                # not close enough—display but mark unsynced
                status = f"UNSYNC ({dt_ms:.1f} ms)"
            else:
                status = f"SYNC ({dt_ms:.1f} ms)"

            # Get current camera settings for display
            exp0 = cam0.cap.get(cv2.CAP_PROP_EXPOSURE)
            gain0 = cam0.cap.get(cv2.CAP_PROP_GAIN)
            wb0 = cam0.cap.get(cv2.CAP_PROP_TEMPERATURE)
            
            exp1 = cam1.cap.get(cv2.CAP_PROP_EXPOSURE)
            gain1 = cam1.cap.get(cv2.CAP_PROP_GAIN)
            wb1 = cam1.cap.get(cv2.CAP_PROP_TEMPERATURE)
            
            # draw + show
            cv2.putText(f0, status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if ok else (0,255,255), 2)
            cv2.putText(f0, f"Exp: {exp0:.0f}us", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f0, f"Gain: {gain0:.0f}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f0, f"WB: {wb0:.0f}K", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f0, f"FPS: {cam0.current_fps:.1f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.putText(f1, status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if ok else (0,255,255), 2)
            cv2.putText(f1, f"Exp: {exp1:.0f}us", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f1, f"Gain: {gain1:.0f}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f1, f"WB: {wb1:.0f}K", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f1, f"FPS: {cam1.current_fps:.1f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow("cam0", f0); cv2.imshow("cam1", f1)
            if cv2.waitKey(1) == 27: break
    finally:
        cam0.release(); cam1.release()
        cv2.destroyAllWindows()
        # Clean up the extension unit
        dll.DeinitExtensionUnit()
        print("Extension unit deinitialized")

if __name__ == "__main__":
    main()
