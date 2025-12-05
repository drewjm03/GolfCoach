"""
Stage 4: Stereo HybrIK fusion with ground-plane visualization.

This script:
  - Loads rig_config (stereo calibration + ground plane).
  - Opens two cameras.
  - Runs HybrIK per camera to get 3D joints in each camera frame.
  - Transforms cam1 joints into cam0 frame and fuses with cam0 joints
    using confidence-weighted averaging.
  - Optionally runs a One-Euro filter on the fused 3D joints.
  - Visualizes the fused 3D skeleton in Open3D together with the
    ground plane (in cam0 frame).

Note: HybrIKRunner is currently a stub; you still need to plug in your
actual HybrIK implementation and checkpoint.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
import queue
from typing import Optional

import numpy as np

try:
    from .capture import CamReader
    from .viewer3d import Viewer3D, HAVE_OPEN3D
    from .hybrik_runner import HybrIKRunner
    from .fusion import fuse_two_views_joints
    from .filtering import OneEuroFilter3D
    from .smpl_model import SMPLModel
except Exception:  # pragma: no cover - fallback when run as script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from apps.capture import CamReader  # type: ignore
    from apps.viewer3d import Viewer3D, HAVE_OPEN3D  # type: ignore
    from apps.hybrik_runner import HybrIKRunner  # type: ignore
    from apps.fusion import fuse_two_views_joints  # type: ignore
    from apps.filtering import OneEuroFilter3D  # type: ignore
    from apps.smpl_model import SMPLModel  # type: ignore


def load_rig_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stereo HybrIK fusion with ground-plane visualization"
    )
    parser.add_argument("--rig-config", required=True, type=str, help="Path to rig_config JSON")
    parser.add_argument("--cam0", type=int, default=0, help="Camera index for cam0")
    parser.add_argument("--cam1", type=int, default=1, help="Camera index for cam1")
    parser.add_argument("--hybrik-cfg", type=str, required=True, help="HybrIK config path")
    parser.add_argument("--hybrik-ckpt", type=str, required=True, help="HybrIK checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--smpl-model",
        type=str,
        required=True,
        help="Path to SMPL-X model directory/file (for mesh generation)",
    )
    parser.add_argument(
        "--inference-fps",
        type=int,
        default=30,
        help="Target FPS for HybrIK inference (default: 30)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Enable One-Euro filter on fused 3D joints",
    )
    args = parser.parse_args()

    if not HAVE_OPEN3D:
        print("[ERROR] Open3D not installed. Install with: pip install open3d")
        sys.exit(1)

    if not os.path.exists(args.rig_config):
        print(f"[ERROR] Rig config file not found: {args.rig_config}")
        sys.exit(1)

    print(f"[MAIN] Loading rig config from: {args.rig_config}")
    rig = load_rig_config(args.rig_config)

    stereo_calib = rig.get("stereo_calib", {})
    cam0_calib = stereo_calib.get("camera_0", {})
    cam1_calib = stereo_calib.get("camera_1", {})

    if not cam0_calib or not cam1_calib:
        print("[ERROR] Missing camera_0 or camera_1 calibration in rig_config")
        sys.exit(1)

    K0 = np.asarray(cam0_calib["K"], dtype=np.float32)
    K1 = np.asarray(cam1_calib["K"], dtype=np.float32)
    R_10 = np.asarray(cam1_calib["R"], dtype=np.float32)  # cam1 -> cam0
    T_10 = np.asarray(cam1_calib["t"], dtype=np.float32)

    ground_plane = rig.get("ground_plane", None)
    image_size = tuple(rig.get("image_size", [1280, 720]))
    print(f"[MAIN] image_size={image_size}")
    print(f"[MAIN] Ground plane: {'present' if ground_plane else 'not found'}")

    # Initialize SMPL-X model (for mesh)
    print("[MAIN] Loading SMPL-X model…")
    smpl_model = SMPLModel(args.smpl_model, device=args.device)
    smpl_faces = smpl_model.faces

    print(f"[MAIN] Opening cameras: cam0={args.cam0}, cam1={args.cam1}")
    try:
        cam0 = CamReader(args.cam0)
        cam1 = CamReader(args.cam1)
    except Exception as e:
        print(f"[ERROR] Failed to open cameras: {e}")
        sys.exit(1)

    print("[MAIN] Loading HybrIK models…")
    hybrik0 = HybrIKRunner(args.hybrik_cfg, args.hybrik_ckpt, device=args.device)
    hybrik1 = HybrIKRunner(args.hybrik_cfg, args.hybrik_ckpt, device=args.device)

    print("[MAIN] Initializing viewer…")
    viewer = Viewer3D(
        ground_plane=ground_plane,
        connections=None,  # You can later pass a SMPL body graph here
        window_name="Stereo HybrIK Fusion",
    )

    filter_3d: Optional[OneEuroFilter3D] = None
    last_time = time.perf_counter()
    target_period = 1.0 / max(1, args.inference_fps)

    print("[MAIN] Starting main loop. Close 3D viewer window to exit.")

    try:
        while True:
            if viewer.is_closed():
                print("[MAIN] Viewer window closed")
                break

            # Get latest frames from both cameras
            try:
                ts0, frame0 = cam0.latest(timeout=0.1)
                ts1, frame1 = cam1.latest(timeout=0.1)
            except queue.Empty:
                viewer.update()
                continue

            now = time.perf_counter()
            if now - last_time < target_period:
                viewer.update()
                continue
            last_time = now

            # Run HybrIK per view
            try:
                out0 = hybrik0.infer(frame0, K0)
                out1 = hybrik1.infer(frame1, K1)
            except NotImplementedError as e:
                # Until HybrIKRunner is fully wired, fail fast with a clear message.
                print(f"[ERROR] HybrIKRunner.infer not implemented: {e}")
                break
            except Exception as e:
                print(f"[WARN] HybrIK inference failed: {e}")
                viewer.update()
                continue

            if out0 is None or out1 is None:
                viewer.update()
                continue

            joints0 = np.asarray(out0["joints_3d"], dtype=np.float32)  # (J, 3)
            conf0 = np.asarray(out0["confs"], dtype=np.float32)        # (J,)
            joints1 = np.asarray(out1["joints_3d"], dtype=np.float32)
            conf1 = np.asarray(out1["confs"], dtype=np.float32)

            # Fuse into cam0 frame
            fused_joints, fused_conf = fuse_two_views_joints(
                joints0, conf0,
                joints1, conf1,
                R_10, T_10,
            )

            # Optional temporal filtering
            if args.filter:
                if filter_3d is None:
                    J = fused_joints.shape[0]
                    filter_3d = OneEuroFilter3D(
                        num_keypoints=J,
                        min_cutoff=1.0,
                        beta=0.0,
                        dcutoff=1.0,
                    )
                    print(f"[MAIN] Initialized 3D One-Euro filter for {J} keypoints")
                fused_joints = filter_3d(fused_joints, now)

            # Visualize fused skeleton (as points; connections=None)
            viewer.update_skeleton(fused_joints)

            # ---- SMPL-X mesh from HybrIK pose + betas ----
            try:
                # Choose pose + betas from cam0 for now
                pose = np.asarray(out0["pose"], dtype=np.float32).reshape(1, -1)
                betas = np.asarray(out0["betas"], dtype=np.float32).reshape(1, -1)

                # Root index 0 assumed to be pelvis in SMPL-X joint layout
                root_idx = 0
                if fused_joints.shape[0] <= root_idx or not np.isfinite(
                    fused_joints[root_idx]
                ).all():
                    # Fallback: average all valid joints as root
                    valid = np.isfinite(fused_joints).all(axis=1)
                    if not np.any(valid):
                        viewer.update()
                        continue
                    fused_root_cam0 = fused_joints[valid].mean(axis=0)
                else:
                    fused_root_cam0 = fused_joints[root_idx]

                transl = fused_root_cam0.reshape(1, 3).astype(np.float32)

                verts, joints_smpl = smpl_model(betas, pose, transl)
                verts_world = verts[0]

                viewer.update_mesh(verts_world, smpl_faces)
            except KeyError:
                # HybrIK output missing pose/betas; skip mesh but keep skeleton
                pass
            except Exception as e:
                print(f"[WARN] SMPL-X mesh generation failed: {e}")

            viewer.update()

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        print("[MAIN] Cleaning up…")
        cam0.release()
        cam1.release()
        viewer.close()
        print("[MAIN] Done")


if __name__ == "__main__":
    main()



