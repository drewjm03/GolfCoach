import cv2
import time
import numpy as np

def list_available_properties(cap):
    """List all available camera properties"""
    properties = [
        (cv2.CAP_PROP_POS_MSEC, "Position in milliseconds"),
        (cv2.CAP_PROP_FRAME_WIDTH, "Frame width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "Frame height"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_FOURCC, "FourCC code"),
        (cv2.CAP_PROP_FRAME_COUNT, "Frame count"),
        (cv2.CAP_PROP_FORMAT, "Format"),
        (cv2.CAP_PROP_MODE, "Mode"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_HUE, "Hue"),
        (cv2.CAP_PROP_GAIN, "Gain"),
        (cv2.CAP_PROP_EXPOSURE, "Exposure"),
        (cv2.CAP_PROP_CONVERT_RGB, "Convert RGB"),
        (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, "White balance blue U"),
        (cv2.CAP_PROP_RECTIFICATION, "Rectification"),
        (cv2.CAP_PROP_MONOCHROME, "Monochrome"),
        (cv2.CAP_PROP_SHARPNESS, "Sharpness"),
        (cv2.CAP_PROP_AUTO_EXPOSURE, "Auto exposure"),
        (cv2.CAP_PROP_GAMMA, "Gamma"),
        (cv2.CAP_PROP_TEMPERATURE, "Temperature"),
        (cv2.CAP_PROP_TRIGGER, "Trigger"),
        (cv2.CAP_PROP_TRIGGER_DELAY, "Trigger delay"),
        (cv2.CAP_PROP_WHITE_BALANCE_RED_V, "White balance red V"),
        (cv2.CAP_PROP_ZOOM, "Zoom"),
        (cv2.CAP_PROP_FOCUS, "Focus"),
        (cv2.CAP_PROP_GUID, "GUID"),
        (cv2.CAP_PROP_ISO_SPEED, "ISO speed"),
        (cv2.CAP_PROP_BACKLIGHT, "Backlight"),
        (cv2.CAP_PROP_PAN, "Pan"),
        (cv2.CAP_PROP_TILT, "Tilt"),
        (cv2.CAP_PROP_ROLL, "Roll"),
        (cv2.CAP_PROP_IRIS, "Iris"),
        (cv2.CAP_PROP_SETTINGS, "Settings"),
        (cv2.CAP_PROP_BUFFERSIZE, "Buffer size"),
        (cv2.CAP_PROP_AUTOFOCUS, "Auto focus"),
    ]
    
    print("Available camera properties:")
    for prop_id, prop_name in properties:
        try:
            value = cap.get(prop_id)
            if value != -1:  # -1 usually means not supported
                print(f"  {prop_name} (ID: {prop_id}): {value}")
        except:
            pass

def set_camera_controls(cap, camera_name):
    """Set various camera controls"""
    print(f"\nSetting controls for {camera_name}...")
    
    # Disable auto exposure (0 = manual, 1 = auto)
    if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0):
        print("  ✓ Auto exposure disabled (manual mode)")
    else:
        print("  ✗ Failed to disable auto exposure")
    
    # Set exposure time (in microseconds)
    exposure_time = 10000  # 10ms
    if cap.set(cv2.CAP_PROP_EXPOSURE, exposure_time):
        print(f"  ✓ Exposure time set to {exposure_time} microseconds")
    else:
        print("  ✗ Failed to set exposure time")
    
    # Set gain
    gain = 100  # Adjust as needed
    if cap.set(cv2.CAP_PROP_GAIN, gain):
        print(f"  ✓ Gain set to {gain}")
    else:
        print("  ✗ Failed to set gain")
    
    # Set white balance (disable auto white balance first)
    if cap.set(cv2.CAP_PROP_AUTO_WB, 0):
        print("  ✓ Auto white balance disabled")
    else:
        print("  ✗ Failed to disable auto white balance")
    
    # Set white balance temperature
    temp = 5500  # Kelvin (5500K = daylight)
    if cap.set(cv2.CAP_PROP_TEMPERATURE, temp):
        print(f"  ✓ White balance temperature set to {temp}K")
    else:
        print("  ✗ Failed to set white balance temperature")
    
    # Set brightness
    brightness = 128  # 0-255
    if cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness):
        print(f"  ✓ Brightness set to {brightness}")
    else:
        print("  ✗ Failed to set brightness")
    
    # Set contrast
    contrast = 128  # 0-255
    if cap.set(cv2.CAP_PROP_CONTRAST, contrast):
        print(f"  ✓ Contrast set to {contrast}")
    else:
        print("  ✗ Failed to set contrast")

def read_camera_settings(cap, camera_name):
    """Read current camera settings"""
    print(f"\nCurrent settings for {camera_name}:")
    
    settings = [
        (cv2.CAP_PROP_AUTO_EXPOSURE, "Auto Exposure"),
        (cv2.CAP_PROP_EXPOSURE, "Exposure Time"),
        (cv2.CAP_PROP_GAIN, "Gain"),
        (cv2.CAP_PROP_AUTO_WB, "Auto White Balance"),
        (cv2.CAP_PROP_TEMPERATURE, "White Balance Temperature"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_HUE, "Hue"),
        (cv2.CAP_PROP_SHARPNESS, "Sharpness"),
        (cv2.CAP_PROP_GAMMA, "Gamma"),
    ]
    
    for prop_id, prop_name in settings:
        try:
            value = cap.get(prop_id)
            if value != -1:
                print(f"  {prop_name}: {value}")
        except:
            pass

def main():
    print("UVC Camera Controls Demo")
    print("=" * 50)
    
    # Open cameras
    cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap0.isOpened():
        cap0 = cv2.VideoCapture(0, cv2.CAP_MSMF)
    
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap1.isOpened():
        cap1 = cv2.VideoCapture(1, cv2.CAP_MSMF)
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("ERROR: Could not open cameras")
        return
    
    # Set resolution
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # List available properties for camera 0
    print("\nCamera 0 properties:")
    list_available_properties(cap0)
    
    # Set controls for both cameras
    set_camera_controls(cap0, "Camera 0")
    set_camera_controls(cap1, "Camera 1")
    
    # Read current settings
    read_camera_settings(cap0, "Camera 0")
    read_camera_settings(cap1, "Camera 1")
    
    print("\nPress 'q' to quit, 'e' to adjust exposure, 'g' to adjust gain, 'w' to adjust white balance")
    print("Camera controls: '1'/'2' for exposure, '3'/'4' for gain, '5'/'6' for white balance")
    
    # Main loop
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if ret0 and ret1:
            # Add text overlay with current settings
            exp0 = cap0.get(cv2.CAP_PROP_EXPOSURE)
            gain0 = cap0.get(cv2.CAP_PROP_GAIN)
            temp0 = cap0.get(cv2.CAP_PROP_TEMPERATURE)
            
            exp1 = cap1.get(cv2.CAP_PROP_EXPOSURE)
            gain1 = cap1.get(cv2.CAP_PROP_GAIN)
            temp1 = cap1.get(cv2.CAP_PROP_TEMPERATURE)
            
            # Draw settings on frames
            cv2.putText(frame0, f"Exp: {exp0:.0f}, Gain: {gain0:.0f}, WB: {temp0:.0f}K", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame1, f"Exp: {exp1:.0f}, Gain: {gain1:.0f}, WB: {temp1:.0f}K", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera 0", frame0)
            cv2.imshow("Camera 1", frame1)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):  # Decrease exposure
            exp = cap0.get(cv2.CAP_PROP_EXPOSURE)
            cap0.set(cv2.CAP_PROP_EXPOSURE, max(1, exp * 0.8))
            cap1.set(cv2.CAP_PROP_EXPOSURE, max(1, exp * 0.8))
            print(f"Exposure decreased to {cap0.get(cv2.CAP_PROP_EXPOSURE):.0f}")
        elif key == ord('2'):  # Increase exposure
            exp = cap0.get(cv2.CAP_PROP_EXPOSURE)
            cap0.set(cv2.CAP_PROP_EXPOSURE, exp * 1.2)
            cap1.set(cv2.CAP_PROP_EXPOSURE, exp * 1.2)
            print(f"Exposure increased to {cap0.get(cv2.CAP_PROP_EXPOSURE):.0f}")
        elif key == ord('3'):  # Decrease gain
            gain = cap0.get(cv2.CAP_PROP_GAIN)
            cap0.set(cv2.CAP_PROP_GAIN, max(0, gain - 10))
            cap1.set(cv2.CAP_PROP_GAIN, max(0, gain - 10))
            print(f"Gain decreased to {cap0.get(cv2.CAP_PROP_GAIN):.0f}")
        elif key == ord('4'):  # Increase gain
            gain = cap0.get(cv2.CAP_PROP_GAIN)
            cap0.set(cv2.CAP_PROP_GAIN, gain + 10)
            cap1.set(cv2.CAP_PROP_GAIN, gain + 10)
            print(f"Gain increased to {cap0.get(cv2.CAP_PROP_GAIN):.0f}")
        elif key == ord('5'):  # Decrease white balance temperature
            temp = cap0.get(cv2.CAP_PROP_TEMPERATURE)
            cap0.set(cv2.CAP_PROP_TEMPERATURE, max(2000, temp - 500))
            cap1.set(cv2.CAP_PROP_TEMPERATURE, max(2000, temp - 500))
            print(f"WB temperature decreased to {cap0.get(cv2.CAP_PROP_TEMPERATURE):.0f}K")
        elif key == ord('6'):  # Increase white balance temperature
            temp = cap0.get(cv2.CAP_PROP_TEMPERATURE)
            cap0.set(cv2.CAP_PROP_TEMPERATURE, min(10000, temp + 500))
            cap1.set(cv2.CAP_PROP_TEMPERATURE, min(10000, temp + 500))
            print(f"WB temperature increased to {cap0.get(cv2.CAP_PROP_TEMPERATURE):.0f}K")
    
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
