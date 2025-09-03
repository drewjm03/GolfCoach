import cv2
import time

def setup_camera_controls(cap, camera_name):
    """Setup manual camera controls"""
    print(f"Setting up {camera_name}...")
    
    # Disable auto exposure (0 = manual, 1 = auto)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    print(f"  Auto exposure disabled for {camera_name}")
    
    # Disable auto white balance
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    print(f"  Auto white balance disabled for {camera_name}")
    
    # Set initial values
    cap.set(cv2.CAP_PROP_EXPOSURE, 10000)  # 10ms exposure
    cap.set(cv2.CAP_PROP_GAIN, 100)        # Gain value
    cap.set(cv2.CAP_PROP_TEMPERATURE, 5500)  # 5500K white balance
    
    print(f"  Initial settings: Exp=10000μs, Gain=100, WB=5500K")

def main():
    print("Simple UVC Camera Controls")
    print("Controls:")
    print("  '1'/'2' - Decrease/Increase exposure")
    print("  '3'/'4' - Decrease/Increase gain") 
    print("  '5'/'6' - Decrease/Increase white balance temperature")
    print("  'q' - Quit")
    print("=" * 40)
    
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
    
    # Setup manual controls
    setup_camera_controls(cap0, "Camera 0")
    setup_camera_controls(cap1, "Camera 1")
    
    print("\nStarting camera stream...")
    
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if ret0 and ret1:
            # Get current settings
            exp0 = cap0.get(cv2.CAP_PROP_EXPOSURE)
            gain0 = cap0.get(cv2.CAP_PROP_GAIN)
            wb0 = cap0.get(cv2.CAP_PROP_TEMPERATURE)
            
            exp1 = cap1.get(cv2.CAP_PROP_EXPOSURE)
            gain1 = cap1.get(cv2.CAP_PROP_GAIN)
            wb1 = cap1.get(cv2.CAP_PROP_TEMPERATURE)
            
            # Add text overlay
            cv2.putText(frame0, f"Exp: {exp0:.0f}us", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame0, f"Gain: {gain0:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame0, f"WB: {wb0:.0f}K", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame1, f"Exp: {exp1:.0f}us", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame1, f"Gain: {gain1:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame1, f"WB: {wb1:.0f}K", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera 0", frame0)
            cv2.imshow("Camera 1", frame1)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):  # Decrease exposure
            exp = cap0.get(cv2.CAP_PROP_EXPOSURE)
            new_exp = max(1, exp * 0.8)
            cap0.set(cv2.CAP_PROP_EXPOSURE, new_exp)
            cap1.set(cv2.CAP_PROP_EXPOSURE, new_exp)
            print(f"Exposure: {exp:.0f} -> {new_exp:.0f} μs")
        elif key == ord('2'):  # Increase exposure
            exp = cap0.get(cv2.CAP_PROP_EXPOSURE)
            new_exp = exp * 1.2
            cap0.set(cv2.CAP_PROP_EXPOSURE, new_exp)
            cap1.set(cv2.CAP_PROP_EXPOSURE, new_exp)
            print(f"Exposure: {exp:.0f} -> {new_exp:.0f} μs")
        elif key == ord('3'):  # Decrease gain
            gain = cap0.get(cv2.CAP_PROP_GAIN)
            new_gain = max(0, gain - 10)
            cap0.set(cv2.CAP_PROP_GAIN, new_gain)
            cap1.set(cv2.CAP_PROP_GAIN, new_gain)
            print(f"Gain: {gain:.0f} -> {new_gain:.0f}")
        elif key == ord('4'):  # Increase gain
            gain = cap0.get(cv2.CAP_PROP_GAIN)
            new_gain = gain + 10
            cap0.set(cv2.CAP_PROP_GAIN, new_gain)
            cap1.set(cv2.CAP_PROP_GAIN, new_gain)
            print(f"Gain: {gain:.0f} -> {new_gain:.0f}")
        elif key == ord('5'):  # Decrease white balance temperature
            wb = cap0.get(cv2.CAP_PROP_TEMPERATURE)
            new_wb = max(2000, wb - 500)
            cap0.set(cv2.CAP_PROP_TEMPERATURE, new_wb)
            cap1.set(cv2.CAP_PROP_TEMPERATURE, new_wb)
            print(f"WB Temperature: {wb:.0f} -> {new_wb:.0f} K")
        elif key == ord('6'):  # Increase white balance temperature
            wb = cap0.get(cv2.CAP_PROP_TEMPERATURE)
            new_wb = min(10000, wb + 500)
            cap0.set(cv2.CAP_PROP_TEMPERATURE, new_wb)
            cap1.set(cv2.CAP_PROP_TEMPERATURE, new_wb)
            print(f"WB Temperature: {wb:.0f} -> {new_wb:.0f} K")
    
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    print("Cameras released")

if __name__ == "__main__":
    main()
