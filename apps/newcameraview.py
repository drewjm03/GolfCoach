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

def check_camera_properties(cap, camera_name):
    """Check what camera properties are available and their current values"""
    print(f"\n=== {camera_name} Properties ===")
    
    # List of relevant properties to check
    properties = {
        'CAP_PROP_POS_MSEC': cv2.CAP_PROP_POS_MSEC,
        'CAP_PROP_FRAME_WIDTH': cv2.CAP_PROP_FRAME_WIDTH,
        'CAP_PROP_FRAME_HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,
        'CAP_PROP_FPS': cv2.CAP_PROP_FPS,
        'CAP_PROP_FOURCC': cv2.CAP_PROP_FOURCC,
        'CAP_PROP_FRAME_COUNT': cv2.CAP_PROP_FRAME_COUNT,
        'CAP_PROP_FORMAT': cv2.CAP_PROP_FORMAT,
        'CAP_PROP_MODE': cv2.CAP_PROP_MODE,
        'CAP_PROP_BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
        'CAP_PROP_CONTRAST': cv2.CAP_PROP_CONTRAST,
        'CAP_PROP_SATURATION': cv2.CAP_PROP_SATURATION,
        'CAP_PROP_HUE': cv2.CAP_PROP_HUE,
        'CAP_PROP_GAIN': cv2.CAP_PROP_GAIN,
        'CAP_PROP_EXPOSURE': cv2.CAP_PROP_EXPOSURE,
        'CAP_PROP_CONVERT_RGB': cv2.CAP_PROP_CONVERT_RGB,
        'CAP_PROP_WHITE_BALANCE_BLUE_U': cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
        'CAP_PROP_RECTIFICATION': cv2.CAP_PROP_RECTIFICATION,
        'CAP_PROP_MONOCHROME': cv2.CAP_PROP_MONOCHROME,
        'CAP_PROP_SHARPNESS': cv2.CAP_PROP_SHARPNESS,
        'CAP_PROP_AUTO_EXPOSURE': cv2.CAP_PROP_AUTO_EXPOSURE,
        'CAP_PROP_GAMMA': cv2.CAP_PROP_GAMMA,
        'CAP_PROP_TEMPERATURE': cv2.CAP_PROP_TEMPERATURE,
        'CAP_PROP_TRIGGER': cv2.CAP_PROP_TRIGGER,
        'CAP_PROP_TRIGGER_DELAY': cv2.CAP_PROP_TRIGGER_DELAY,
        'CAP_PROP_WHITE_BALANCE_RED_V': cv2.CAP_PROP_WHITE_BALANCE_RED_V,
        'CAP_PROP_ZOOM': cv2.CAP_PROP_ZOOM,
        'CAP_PROP_FOCUS': cv2.CAP_PROP_FOCUS,
        'CAP_PROP_GUID': cv2.CAP_PROP_GUID,
        'CAP_PROP_ISO_SPEED': cv2.CAP_PROP_ISO_SPEED,
        'CAP_PROP_BACKLIGHT': cv2.CAP_PROP_BACKLIGHT,
        'CAP_PROP_PAN': cv2.CAP_PROP_PAN,
        'CAP_PROP_TILT': cv2.CAP_PROP_TILT,
        'CAP_PROP_ROLL': cv2.CAP_PROP_ROLL,
        'CAP_PROP_IRIS': cv2.CAP_PROP_IRIS,
        'CAP_PROP_SETTINGS': cv2.CAP_PROP_SETTINGS,
        'CAP_PROP_BUFFERSIZE': cv2.CAP_PROP_BUFFERSIZE,
        'CAP_PROP_AUTOFOCUS': cv2.CAP_PROP_AUTOFOCUS,
        'CAP_PROP_SAR_NUM': cv2.CAP_PROP_SAR_NUM,
        'CAP_PROP_SAR_DEN': cv2.CAP_PROP_SAR_DEN,
        'CAP_PROP_BACKEND': cv2.CAP_PROP_BACKEND,
        'CAP_PROP_CHANNEL': cv2.CAP_PROP_CHANNEL,
        'CAP_PROP_AUTO_WB': cv2.CAP_PROP_AUTO_WB,
        'CAP_PROP_WB_TEMPERATURE': cv2.CAP_PROP_WB_TEMPERATURE,
    }
    
    for name, prop_id in properties.items():
        try:
            value = cap.get(prop_id)
            if value != -1:  # -1 usually means property not supported
                print(f"{name}: {value}")
        except Exception as e:
            print(f"{name}: Error - {e}")

class CameraControls:
    def __init__(self):
        # Create control window
        cv2.namedWindow('Camera Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Controls', 400, 600)
        
        # Initialize control values with more conservative defaults
        self.camera0_exposure = 5000
        self.camera0_gain = 5
        self.camera0_wb = 2950
        self.camera0_fps = 120
        
        self.camera1_exposure = 5000
        self.camera1_gain = 5
        self.camera1_wb = 2950
        self.camera1_fps = 120
        
        # Track if settings have changed
        self.settings_changed = False
        
        # Create trackbars with more conservative ranges
        cv2.createTrackbar('Cam0 Exposure (μs)', 'Camera Controls', 50, 100, self.on_cam0_exposure_change)
        cv2.createTrackbar('Cam0 Gain', 'Camera Controls', 5, 20, self.on_cam0_gain_change)
        cv2.createTrackbar('Cam0 WB (K)', 'Camera Controls', 295, 100, self.on_cam0_wb_change)
        cv2.createTrackbar('Cam0 FPS', 'Camera Controls', 120, 240, self.on_cam0_fps_change)
        
        cv2.createTrackbar('Cam1 Exposure (μs)', 'Camera Controls', 50, 100, self.on_cam1_exposure_change)
        cv2.createTrackbar('Cam1 Gain', 'Camera Controls', 5, 20, self.on_cam1_gain_change)
        cv2.createTrackbar('Cam1 WB (K)', 'Camera Controls', 295, 100, self.on_cam1_wb_change)
        cv2.createTrackbar('Cam1 FPS', 'Camera Controls', 120, 240, self.on_cam1_fps_change)
        
        # Set initial values
        cv2.setTrackbarPos('Cam0 Exposure (μs)', 'Camera Controls', 50)  # 5000μs / 100
        cv2.setTrackbarPos('Cam0 Gain', 'Camera Controls', 5)
        cv2.setTrackbarPos('Cam0 WB (K)', 'Camera Controls', 295)  # 2950K / 10
        cv2.setTrackbarPos('Cam0 FPS', 'Camera Controls', 120)
        
        cv2.setTrackbarPos('Cam1 Exposure (μs)', 'Camera Controls', 50)
        cv2.setTrackbarPos('Cam1 Gain', 'Camera Controls', 5)
        cv2.setTrackbarPos('Cam1 WB (K)', 'Camera Controls', 295)
        cv2.setTrackbarPos('Cam1 FPS', 'Camera Controls', 120)
    
    def on_cam0_exposure_change(self, val):
        self.camera0_exposure = val * 100  # Convert back to microseconds
        self.settings_changed = True
        print(f"Camera 0 Exposure: {self.camera0_exposure}μs")
    
    def on_cam0_gain_change(self, val):
        self.camera0_gain = val
        self.settings_changed = True
        print(f"Camera 0 Gain: {self.camera0_gain}")
    
    def on_cam0_wb_change(self, val):
        self.camera0_wb = val * 10  # Convert back to Kelvin
        self.settings_changed = True
        print(f"Camera 0 WB: {self.camera0_wb}K")
    
    def on_cam0_fps_change(self, val):
        self.camera0_fps = val
        self.settings_changed = True
        print(f"Camera 0 FPS: {self.camera0_fps}")
    
    def on_cam1_exposure_change(self, val):
        self.camera1_exposure = val * 100
        self.settings_changed = True
        print(f"Camera 1 Exposure: {self.camera1_exposure}μs")
    
    def on_cam1_gain_change(self, val):
        self.camera1_gain = val
        self.settings_changed = True
        print(f"Camera 1 Gain: {self.camera1_gain}")
    
    def on_cam1_wb_change(self, val):
        self.camera1_wb = val * 10
        self.settings_changed = True
        print(f"Camera 1 WB: {self.camera1_wb}K")
    
    def on_cam1_fps_change(self, val):
        self.camera1_fps = val
        self.settings_changed = True
        print(f"Camera 1 FPS: {self.camera1_fps}")
    
    def try_set_white_balance(self, cap, wb_value, camera_name):
        """Try different methods to set white balance"""
        success = False
        
        # Method 1: Try CAP_PROP_TEMPERATURE
        if cap.set(cv2.CAP_PROP_TEMPERATURE, wb_value):
            success = True
            print(f"{camera_name}: Set WB using CAP_PROP_TEMPERATURE to {wb_value}")
        else:
            print(f"{camera_name}: CAP_PROP_TEMPERATURE failed")
        
        # Method 2: Try CAP_PROP_WHITE_BALANCE_BLUE_U
        if not success:
            # Convert Kelvin to blue component (approximate)
            blue_u = int((wb_value - 2000) / 10)  # Rough conversion
            if cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, blue_u):
                success = True
                print(f"{camera_name}: Set WB using CAP_PROP_WHITE_BALANCE_BLUE_U to {blue_u}")
            else:
                print(f"{camera_name}: CAP_PROP_WHITE_BALANCE_BLUE_U failed")
        
        # Method 3: Try CAP_PROP_WHITE_BALANCE_RED_V
        if not success:
            # Convert Kelvin to red component (approximate)
            red_v = int((wb_value - 2000) / 10)  # Rough conversion
            if cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, red_v):
                success = True
                print(f"{camera_name}: Set WB using CAP_PROP_WHITE_BALANCE_RED_V to {red_v}")
            else:
                print(f"{camera_name}: CAP_PROP_WHITE_BALANCE_RED_V failed")
        
        # Method 4: Try setting both blue and red components
        if not success:
            blue_u = int((wb_value - 2000) / 10)
            red_v = int((wb_value - 2000) / 10)
            if cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, blue_u) and cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, red_v):
                success = True
                print(f"{camera_name}: Set WB using both blue and red components")
            else:
                print(f"{camera_name}: Combined WB setting failed")
        
        return success
    
    def apply_settings(self, cam0, cam1):
        """Apply current settings to cameras only if changed"""
        if not self.settings_changed:
            return
            
        try:
            if cam0:
                # Apply settings one by one with validation
                if not cam0.cap.set(cv2.CAP_PROP_EXPOSURE, self.camera0_exposure):
                    print(f"Warning: Failed to set Camera 0 exposure to {self.camera0_exposure}")
                if not cam0.cap.set(cv2.CAP_PROP_GAIN, self.camera0_gain):
                    print(f"Warning: Failed to set Camera 0 gain to {self.camera0_gain}")
                
                # Try to set white balance with multiple methods
                if not self.try_set_white_balance(cam0.cap, self.camera0_wb, "Camera 0"):
                    print(f"Warning: All white balance methods failed for Camera 0")
                
                if not cam0.cap.set(cv2.CAP_PROP_FPS, self.camera0_fps):
                    print(f"Warning: Failed to set Camera 0 FPS to {self.camera0_fps}")
            
            if cam1:
                if not cam1.cap.set(cv2.CAP_PROP_EXPOSURE, self.camera1_exposure):
                    print(f"Warning: Failed to set Camera 1 exposure to {self.camera1_exposure}")
                if not cam1.cap.set(cv2.CAP_PROP_GAIN, self.camera1_gain):
                    print(f"Warning: Failed to set Camera 1 gain to {self.camera1_gain}")
                
                # Try to set white balance with multiple methods
                if not self.try_set_white_balance(cam1.cap, self.camera1_wb, "Camera 1"):
                    print(f"Warning: All white balance methods failed for Camera 1")
                
                if not cam1.cap.set(cv2.CAP_PROP_FPS, self.camera1_fps):
                    print(f"Warning: Failed to set Camera 1 FPS to {self.camera1_fps}")
                    
            self.settings_changed = False
            print("Settings applied successfully")
            
        except Exception as e:
            print(f"Error applying settings: {e}")
            self.settings_changed = False

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

        # Check available camera properties
        check_camera_properties(self.cap, f"Camera {index}")

        self.q = queue.Queue(maxsize=1)  # (t, frame)
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        frame_count = 0
        failed_count = 0
        while self.running:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    failed_count += 1
                    if failed_count % 300 == 0:  # Print every 300 failures (much less frequent)
                        print(f"Camera {self.cap.get(cv2.CAP_PROP_POS_FRAMES)}: Failed to read frame {failed_count} times")
                    time.sleep(0.001)  # Brief pause to prevent busy waiting
                    continue
                
                # Reset failure counter on successful read
                failed_count = 0
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
                
            except Exception as e:
                failed_count += 1
                if failed_count % 100 == 0:  # Print every 100 exceptions
                    print(f"Camera {self.cap.get(cv2.CAP_PROP_POS_FRAMES)}: Exception in frame reading: {e}")
                time.sleep(0.001)  # Brief pause to prevent busy waiting
                continue

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
    
        # Initialize camera controls
    controls = CameraControls()
    
    try:
        alpha = 0.001  # EMA smoothing for offset (optional)
        offset = 0.0   # ts1 ≈ ts0 + offset (+ drift if you add it)

        frame_count = 0
        last_settings_update = time.perf_counter()
        consecutive_failures = 0
        
        while True:
            try:
                t0, f0 = cam0.latest()
                t1, f1 = cam1.latest()
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                
                frame_count += 1
                if frame_count % 300 == 0:  # Print every 300 frames (much less frequent)
                    print(f"Main loop: Frame {frame_count}")
                    print(f"  f0 shape: {f0.shape if f0 is not None else 'None'}")
                    print(f"  f1 shape: {f1.shape if f1 is not None else 'None'}")
                    print(f"  f0 mean: {np.mean(f0) if f0 is not None else 'None'}")
                    print(f"  f1 mean: {np.mean(f1) if f1 is not None else 'None'}")

                # Apply camera settings less frequently (every 500ms) and only if changed
                current_time = time.perf_counter()
                if current_time - last_settings_update >= 0.5:
                    controls.apply_settings(cam0, cam1)
                    last_settings_update = current_time

                                # simple offset refine (optional): move offset toward observed delta
                obs = (t1 - t0)
                offset = (1 - alpha)*offset + alpha*obs

                ok, dt_ms = pair_frames(t0, f0, t1 - offset, f1, tol_ms=15.0)
                
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures % 10 == 0:  # Print every 10 failures
                    print(f"Frame reading error: {e}")
                    print(f"Consecutive failures: {consecutive_failures}")
                time.sleep(0.01)  # Brief pause before retry
                continue
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
            cv2.putText(f0, f"Exp: {exp0:.0f}μs", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f0, f"Gain: {gain0:.0f}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f0, f"WB: {wb0:.0f}K", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f0, f"FPS: {cam0.current_fps:.1f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.putText(f1, status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if ok else (0,255,255), 2)
            cv2.putText(f1, f"Exp: {exp1:.0f}μs", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f1, f"Gain: {gain1:.0f}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f1, f"WB: {wb1:.0f}K", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(f1, f"FPS: {cam1.current_fps:.1f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow("cam0", f0); cv2.imshow("cam1", f1)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
    finally:
        cam0.release(); cam1.release()
        cv2.destroyAllWindows()
        # Clean up the extension unit
        dll.DeinitExtensionUnit()
        print("Extension unit deinitialized")

if __name__ == "__main__":
    main()
