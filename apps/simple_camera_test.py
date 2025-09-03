import cv2
import time

def test_cameras():
    print("Testing cameras without DLL initialization...")
    
    # Test camera 0
    print("\nTesting camera 0...")
    cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap0.isOpened():
        cap0 = cv2.VideoCapture(0, cv2.CAP_MSMF)
    
    if cap0.isOpened():
        print("Camera 0 opened successfully")
        ret, frame = cap0.read()
        if ret:
            print(f"Camera 0 frame: {frame.shape}")
            cv2.imshow("Camera 0", frame)
        else:
            print("Camera 0 cannot read frames")
    else:
        print("Camera 0 failed to open")
    
    # Test camera 1
    print("\nTesting camera 1...")
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap1.isOpened():
        cap1 = cv2.VideoCapture(1, cv2.CAP_MSMF)
    
    if cap1.isOpened():
        print("Camera 1 opened successfully")
        ret, frame = cap1.read()
        if ret:
            print(f"Camera 1 frame: {frame.shape}")
            cv2.imshow("Camera 1", frame)
        else:
            print("Camera 1 cannot read frames")
    else:
        print("Camera 1 failed to open")
    
    # Wait for key press
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    
    # Cleanup
    if cap0.isOpened():
        cap0.release()
    if cap1.isOpened():
        cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_cameras()
