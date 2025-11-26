"""
Stage 2: Real-time 3D pose tracking with triangulation, filtering, and visualization.

This script loads a rig_config JSON file, performs pose estimation on both cameras,
triangulates to 3D, applies One-Euro filtering, and visualizes the result in 3D.
"""

import os
import sys
import json
import argparse
import time
import queue
import numpy as np

try:
    from . import config
    from .capture import CamReader
    from .pose import PoseEstimator, HAVE_RTM, RTM_POSE_CONNECTIONS
    from .triangulation import StereoTriangulator
    from .filtering import OneEuroFilter3D
    from .viewer3d import Viewer3D, HAVE_OPEN3D
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config
    from apps.capture import CamReader
    from apps.pose import PoseEstimator, HAVE_RTM, RTM_POSE_CONNECTIONS
    from apps.triangulation import StereoTriangulator
    from apps.filtering import OneEuroFilter3D
    from apps.viewer3d import Viewer3D, HAVE_OPEN3D


def load_rig_config(config_path):
    """Load rig configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Real-time 3D pose tracking with triangulation and filtering"
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
        default=480,
        help="Width for pose inference (default: 480)"
    )
    parser.add_argument(
        "--inference-fps",
        type=int,
        default=30,
        help="Target FPS for pose inference (default: 30)"
    )
    parser.add_argument(
        "--filter-min-cutoff",
        type=float,
        default=1.0,
        help="One-Euro filter min_cutoff (default: 1.0)"
    )
    parser.add_argument(
        "--filter-beta",
        type=float,
        default=0.0,
        help="One-Euro filter beta (default: 0.0)"
    )
    parser.add_argument(
        "--filter-dcutoff",
        type=float,
        default=1.0,
        help="One-Euro filter dcutoff (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAVE_RTM:
        print("[ERROR] RTM Pose not available. Install rtmlib.")
        sys.exit(1)
    
    if not HAVE_OPEN3D:
        print("[ERROR] Open3D not available. Install with: pip install open3d")
        sys.exit(1)
    
    # Load rig config
    if not os.path.exists(args.rig_config):
        print(f"[ERROR] Rig config file not found: {args.rig_config}")
        sys.exit(1)
    
    print(f"[MAIN] Loading rig config from: {args.rig_config}")
    rig_config = load_rig_config(args.rig_config)
    
    # Extract calibration data
    stereo_calib = rig_config.get("stereo_calib", {})
    cam0_calib = stereo_calib.get("camera_0", {})
    cam1_calib = stereo_calib.get("camera_1", {})
    
    K0 = np.asarray(cam0_calib["K"], dtype=np.float64)
    D0 = np.asarray(cam0_calib["dist_coeffs"], dtype=np.float64)
    K1 = np.asarray(cam1_calib["K"], dtype=np.float64)
    D1 = np.asarray(cam1_calib["dist_coeffs"], dtype=np.float64)
    
    # R and T from cam0 to cam1
    # In rig_config, camera_1 has R and t relative to camera_0
    R = np.asarray(cam1_calib["R"], dtype=np.float64)
    T = np.asarray(cam1_calib["t"], dtype=np.float64)
    
    image_size = tuple(rig_config.get("image_size", [1280, 720]))
    ground_plane = rig_config.get("ground_plane", None)
    
    print(f"[MAIN] Image size: {image_size}")
    print(f"[MAIN] Ground plane: {'present' if ground_plane else 'not found'}")
    
    # Open cameras
    print(f"[MAIN] Opening cameras: cam0={args.cam0}, cam1={args.cam1}")
    try:
        cam0 = CamReader(args.cam0)
        cam1 = CamReader(args.cam1)
    except Exception as e:
        print(f"[ERROR] Failed to open cameras: {e}")
        sys.exit(1)
    
    # Initialize pose estimators
    print("[MAIN] Initializing pose estimators...")
    pose_estimator0 = PoseEstimator(
        enable=True,
        model_complexity=1,
        inference_width=args.inference_width,
        inference_fps=args.inference_fps
    )
    pose_estimator1 = PoseEstimator(
        enable=True,
        model_complexity=1,
        inference_width=args.inference_width,
        inference_fps=args.inference_fps
    )
    
    # Initialize triangulator
    print("[MAIN] Initializing triangulator...")
    triangulator = StereoTriangulator(K0, D0, K1, D1, R, T, image_size)
    
    # Initialize filter (we'll determine num_keypoints from first detection)
    print("[MAIN] Initializing One-Euro filter...")
    filter_3d = None  # Will be initialized after first detection
    
    # Initialize 3D viewer
    print("[MAIN] Initializing 3D viewer...")
    viewer = Viewer3D(
        ground_plane=ground_plane,
        connections=RTM_POSE_CONNECTIONS,
        window_name="3D Pose Viewer"
    )
    
    print("[MAIN] Starting main loop. Close 3D viewer window to exit.")
    
    last_time = time.perf_counter()
    
    try:
        while True:
            # Check if viewer is closed
            if viewer.is_closed():
                print("[MAIN] Viewer window closed")
                break
            
            # Read frames from both cameras
            try:
                ts0, frame0 = cam0.latest(timeout=0.1)
                ts1, frame1 = cam1.latest(timeout=0.1)
            except queue.Empty:
                # Update viewer even if no new frames
                viewer.update()
                continue
            
            # Submit frames to pose estimators
            pose_estimator0.submit(ts0, frame0.copy())
            pose_estimator1.submit(ts1, frame1.copy())
            
            # Get pose results
            result0 = pose_estimator0.latest_result()
            result1 = pose_estimator1.latest_result()
            
            if result0 and result0[1] is not None and result1 and result1[1] is not None:
                landmarks0 = result0[1]
                landmarks1 = result1[1]
                
                # Triangulate to 3D
                keypoints_3d = triangulator.triangulate_keypoints(landmarks0, landmarks1)
                
                if keypoints_3d is not None and len(keypoints_3d) > 0:
                    # Initialize filter on first valid detection
                    if filter_3d is None:
                        num_kpts = len(keypoints_3d)
                        filter_3d = OneEuroFilter3D(
                            num_kpts,
                            min_cutoff=args.filter_min_cutoff,
                            beta=args.filter_beta,
                            dcutoff=args.filter_dcutoff
                        )
                        print(f"[MAIN] Initialized filter for {num_kpts} keypoints")
                    
                    # Apply filter
                    current_time = time.perf_counter()
                    keypoints_3d_filtered = filter_3d(keypoints_3d, current_time)
                    
                    # Update viewer
                    viewer.update_skeleton(keypoints_3d_filtered)
            
            # Update viewer
            viewer.update()
            
            # Small sleep to avoid busy waiting
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        # Cleanup
        print("[MAIN] Cleaning up...")
        pose_estimator0.stop()
        pose_estimator1.stop()
        cam0.release()
        cam1.release()
        viewer.close()
        print("[MAIN] Done")


if __name__ == "__main__":
    main()

