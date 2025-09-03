import cv2
import time
import ctypes

def reset_cameras():
    print("Attempting to reset cameras...")
    
    # First, try to close any existing camera connections
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Closing camera {i}")
            cap.release()
    
    time.sleep(2)
    
    # Now try to open cameras again
    print("\nTrying to open cameras after reset...")
    
    for i in range(2):
        print(f"\nTesting camera {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"  DirectShow failed for camera {i}")
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        
        if cap.isOpened():
            print(f"  Camera {i} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"  Camera {i} can read frames: {frame.shape}")
                cv2.imshow(f"Camera {i}", frame)
            else:
                print(f"  Camera {i} opened but cannot read frames")
            
            cap.release()
        else:
            print(f"  Camera {i} failed to open")
    
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reset_cameras()
