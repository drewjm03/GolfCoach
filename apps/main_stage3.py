"""
Stage 3: Real-time 3D pose tracking with triangulation, filtering,
and SMPL mesh visualization.

This script extends Stage 2 by adding a (stub) SMPL mesh estimator and
rendering the mesh in the same Open3D world as the skeleton and ground
plane.
"""

import os
import sys
import json
import argparse
import time
import queue
from typing import Optional
import numpy as np

try:
    from . import config
    from .capture import CamReader
    from .pose import PoseEstimator, HAVE_RTM, RTM_POSE_CONNECTIONS
    from .triangulation import StereoTriangulator
    from .filtering import OneEuroFilter3D
    from .viewer3d import Viewer3D, HAVE_OPEN3D
    from .mesh_estimator import MeshEstimator
except Exception:  # pragma: no cover - fallback when run as script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps import config  # type: ignore
    from apps.capture import CamReader  # type: ignore
    from apps.pose import PoseEstimator, HAVE_RTM, RTM_POSE_CONNECTIONS  # type: ignore
    from apps.triangulation import StereoTriangulator  # type: ignore
    from apps.filtering import OneEuroFilter3D  # type: ignore
    from apps.viewer3d import Viewer3D, HAVE_OPEN3D  # type: ignore
    from apps.mesh_estimator import MeshEstimator  # type: ignore


def load_rig_config(config_path: str):
    """Load rig configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Real-time 3D pose tracking with triangulation, filtering, "
            "and SMPL mesh visualization (stub)."
        )
    )
    parser.add_argument(
        "--rig-config",
        type=str,
        required=True,
        help="Path to rig_config JSON file",
    )
    parser.add_argument(
        "--cam0",
        type=int,
        default=0,
        help="Camera index for camera 0 (default: 0)",
    )
    parser.add_argument(
        "--cam1",
        type=int,
        default=1,
        help="Camera index for camera 1 (default: 1)",
    )
    parser.add_argument(
        "--inference-width",
        type=int,
        default=480,
        help="Width for pose inference (default: 480)",
    )
    parser.add_argument(
        "--inference-fps",
        type=int,
        default=30,
        help="Target FPS for pose inference (default: 30)",
    )
    parser.add_argument(
        "--filter-min-cutoff",
        type=float,
        default=1.0,
        help="One-Euro filter min_cutoff (default: 1.0)",
    )
    parser.add_argument(
        "--filter-beta",
        type=float,
        default=0.0,
        help="One-Euro filter beta (default: 0.0)",
    )
    parser.add_argument(
        "--filter-dcutoff",
        type=float,
        default=1.0,
        help="One-Euro filter dcutoff (default: 1.0)",
    )
    parser.add_argument(
        "--smpl-model",
        type=str,
        default=None,
        help=(
            "Path to SMPL model file. If not provided, mesh estimation "
            "is skipped."
        ),
    )

    args = parser.parse_args()

    # Dependency checks
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

    # Pose estimators
    print("[MAIN] Initializing pose estimators…")
    pose_estimator0 = PoseEstimator(
        enable=True,
        model_complexity=1,
        inference_width=args.inference_width,
        inference_fps=args.inference_fps,
    )
    pose_estimator1 = PoseEstimator(
        enable=True,
        model_complexity=1,
        inference_width=args.inference_width,
        inference_fps=args.inference_fps,
    )

    # Triangulator
    print("[MAIN] Initializing triangulator…")
    triangulator = StereoTriangulator(K0, D0, K1, D1, R, T, image_size)

    # 3D filter (initialized lazily)
    filter_3d: Optional[OneEuroFilter3D] = None  # type: ignore[type-arg]

    # Mesh estimator (optional, stub)
    mesh_estimator: Optional[MeshEstimator] = None  # type: ignore[type-arg]
    if args.smpl_model is not None:
        try:
            mesh_estimator = MeshEstimator(args.smpl_model, device="cpu")
            print("[MAIN] MeshEstimator initialized (stub).")
        except Exception as e:
            print(f"[WARN] Failed to initialize MeshEstimator: {e}")
            mesh_estimator = None

    # 3D viewer
    print("[MAIN] Initializing 3D viewer…")
    viewer = Viewer3D(
        ground_plane=ground_plane,
        connections=RTM_POSE_CONNECTIONS,
        window_name="3D Pose + Mesh Viewer",
    )

    print("[MAIN] Starting main loop. Close 3D viewer window to exit.")

    try:
        while True:
            # Check if viewer is closed
            if viewer.is_closed():
                print("[MAIN] Viewer window closed")
                break

            # Get latest frames
            try:
                ts0, frame0 = cam0.latest(timeout=0.1)
                ts1, frame1 = cam1.latest(timeout=0.1)
            except queue.Empty:
                viewer.update()
                continue

            # Submit to pose estimators
            pose_estimator0.submit(ts0, frame0.copy())
            pose_estimator1.submit(ts1, frame1.copy())

            # Get latest pose results
            result0 = pose_estimator0.latest_result()
            result1 = pose_estimator1.latest_result()

            if result0 and result0[1] is not None and result1 and result1[1] is not None:
                landmarks0 = result0[1]
                landmarks1 = result1[1]

                # Triangulate to 3D (cam0 frame)
                keypoints_3d = triangulator.triangulate_keypoints(landmarks0, landmarks1)

                if keypoints_3d is not None and len(keypoints_3d) > 0:
                    # Initialize filter on first detection
                    if filter_3d is None:
                        num_kpts = len(keypoints_3d)
                        filter_3d = OneEuroFilter3D(
                            num_kpts,
                            min_cutoff=args.filter_min_cutoff,
                            beta=args.filter_beta,
                            dcutoff=args.filter_dcutoff,
                        )
                        print(f"[MAIN] Initialized 3D filter for {num_kpts} keypoints")

                    # Filter
                    current_time = time.perf_counter()
                    keypoints_3d_filtered = filter_3d(keypoints_3d, current_time)

                    # Update skeleton
                    viewer.update_skeleton(keypoints_3d_filtered)

                    # Mesh estimation (stub) and visualization
                    if mesh_estimator is not None:
                        try:
                            verts_world, faces = mesh_estimator.estimate_mesh(
                                keypoints_3d_filtered
                            )
                            viewer.update_mesh(verts_world, faces)
                        except NotImplementedError:
                            # SMPL fitting not implemented yet
                            pass
                        except Exception as e:
                            print(f"[WARN] Mesh estimation failed: {e}")

            # Update viewer
            viewer.update()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        print("[MAIN] Cleaning up…")
        pose_estimator0.stop()
        pose_estimator1.stop()
        cam0.release()
        cam1.release()
        viewer.close()
        print("[MAIN] Done")


if __name__ == "__main__":
    main()

