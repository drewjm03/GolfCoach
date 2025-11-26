"""
Stage 1: Load rig config and display dual camera feeds with pose overlay.

This script loads a rig_config JSON file, opens two cameras, and displays
pose estimation overlays on both camera feeds in separate OpenCV windows.
"""

import os
import sys
import json
import argparse
import cv2
import queue

try:
    from . import config
    from .capture import CamReader
    from .pose import PoseEstimator, draw_rtm_landmarks, HAVE_RTM
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config
    from apps.capture import CamReader
    from apps.pose import PoseEstimator, draw_rtm_landmarks, HAVE_RTM


def load_rig_config(config_path):
    """Load rig configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Load rig config and display dual camera feeds with pose overlay"
    )
    parser.add_argument(
        "--rig-config",
        type=str,
        required=True,
        help="Path to rig_config JSON file"
    )
    parser.add_argument(
        "--cam0",
        type=int,
        default=0,
        help="Camera index for camera 0 (default: 0)"
    )
    parser.add_argument(
        "--cam1",
        type=int,
        default=1,
        help="Camera index for camera 1 (default: 1)"
    )
    parser.add_argument(
        "--inference-width",
        type=int,
        default=640,
        help="Width for pose inference (default: 640)"
    )
    parser.add_argument(
        "--inference-fps",
        type=int,
        default=30,
        help="Target FPS for pose inference (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Load rig config
    if not os.path.exists(args.rig_config):
        print(f"[ERROR] Rig config file not found: {args.rig_config}")
        sys.exit(1)
    
    print(f"[MAIN] Loading rig config from: {args.rig_config}")
    rig_config = load_rig_config(args.rig_config)
    
    # Extract image size from config
    image_size = rig_config.get("image_size", [1280, 720])
    print(f"[MAIN] Image size from config: {image_size}")
    
    # Check if RTM Pose is available
    if not HAVE_RTM:
        print("[WARN] RTM Pose not available. Pose estimation will be disabled.")
        try:
            from apps.pose import RTM_IMPORT_ERR
            print(f"[WARN] Import error: {RTM_IMPORT_ERR}")
        except:
            print("[WARN] Could not determine import error")
    
    # Open cameras
    print(f"[MAIN] Opening cameras: cam0={args.cam0}, cam1={args.cam1}")
    try:
        cam0 = CamReader(args.cam0)
        cam1 = CamReader(args.cam1)
    except Exception as e:
        print(f"[ERROR] Failed to open cameras: {e}")
        sys.exit(1)
    
    # Initialize pose estimators
    pose_estimator0 = PoseEstimator(
        enable=HAVE_RTM,
        model_complexity=1,
        inference_width=args.inference_width,
        inference_fps=args.inference_fps
    )
    pose_estimator1 = PoseEstimator(
        enable=HAVE_RTM,
        model_complexity=1,
        inference_width=args.inference_width,
        inference_fps=args.inference_fps
    )
    
    # Create OpenCV windows
    cv2.namedWindow("cam 0", cv2.WINDOW_NORMAL)
    cv2.namedWindow("cam 1", cv2.WINDOW_NORMAL)
    
    print("[MAIN] Starting main loop. Press 'q' to quit.")
    
    try:
        while True:
            # Read frames from both cameras
            try:
                ts0, frame0 = cam0.latest(timeout=0.1)
                ts1, frame1 = cam1.latest(timeout=0.1)
            except queue.Empty:
                continue
            
            # Submit frames to pose estimators
            pose_estimator0.submit(ts0, frame0.copy())
            pose_estimator1.submit(ts1, frame1.copy())
            
            # Get pose results and draw overlays
            annotated0 = frame0.copy()
            annotated1 = frame1.copy()
            
            if HAVE_RTM:
                result0 = pose_estimator0.latest_result()
                result1 = pose_estimator1.latest_result()
                
                if result0 and result0[1] is not None:
                    annotated0 = draw_rtm_landmarks(annotated0, result0[1])
                
                if result1 and result1[1] is not None:
                    annotated1 = draw_rtm_landmarks(annotated1, result1[1])
            
            # Display frames
            cv2.imshow("cam 0", annotated0)
            cv2.imshow("cam 1", annotated1)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Check if windows are closed
            if cv2.getWindowProperty("cam 0", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty("cam 1", cv2.WND_PROP_VISIBLE) < 1:
                break
    
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        # Cleanup
        print("[MAIN] Cleaning up...")
        pose_estimator0.stop()
        pose_estimator1.stop()
        cam0.release()
        cam1.release()
        cv2.destroyAllWindows()
        print("[MAIN] Done")


if __name__ == "__main__":
    main()

