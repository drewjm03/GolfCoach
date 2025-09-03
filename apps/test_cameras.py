import cv2
import time

def test_camera(index):
    print(f"\nTesting camera index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"  Camera {index} not available with DirectShow")
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        
    if not cap.isOpened():
        print(f"  Camera {index} not available with Media Foundation")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print(f"  Camera {index} SUCCESS - Frame shape: {frame.shape}")
        print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")
        cap.release()
        return True
    else:
        print(f"  Camera {index} FAILED - Could not read frame")
        cap.release()
        return False

def main():
    print("Testing available cameras...")
    
    # Test first 10 camera indices
    available_cameras = []
    for i in range(10):
        if test_camera(i):
            available_cameras.append(i)
    
    print(f"\nAvailable cameras: {available_cameras}")
    
    if len(available_cameras) >= 2:
        print(f"Use cameras {available_cameras[0]} and {available_cameras[1]} in your main script")
    else:
        print("Not enough cameras found!")

if __name__ == "__main__":
    main()
