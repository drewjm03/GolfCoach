from __future__ import annotations

"""
SMPL A vs B viewer for PHALP/4DHumans outputs (rotation-matrix SMPL params).

- Loads two PHALP PKLs (joblib-compressed pickle is supported)
- Extracts per-frame SMPL parameters:
    global_orient: (1,3,3) or (3,3) rotation matrix
    body_pose:     (23,3,3) rotation matrices (SMPL has 23 body joints)
    betas:         (10,)
- Generates vertices via smplx (model_type="smpl") with pose2rot=False
  (since inputs are rotation matrices, not axis-angles)
- Visualizes both sequences side-by-side in Open3D

Usage:
  python apps/smpl_view_compare.py \
    --pkl_a runs/my_swing_stageA_120front/less_clutter120cam1.pkl \
    --pkl_b runs/my_swing_stageA_120front/less_clutter120cam2.pkl \
    --model_dir body_models/smpl \
    --device cuda \
    --offset 1.5
"""

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import smplx

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


def load_pkl_any(path: str):
    p = Path(path)
    if joblib is not None:
        try:
            return joblib.load(p)
        except Exception:
            pass
    # Fallback to pickle
    import pickle

    with p.open("rb") as f:
        return pickle.load(f)


def resolve_smpl_root(model_dir: Path) -> Path:
    """
    smplx.create expects a ROOT that contains a subfolder named after the model type,
    e.g., 'smplx/' with 'SMPLX_NEUTRAL.npz'.

    If the provided path is '.../body_models/smplx', return its parent.
    If it's already ROOT, return as-is.
    """
    if model_dir.is_file():
        # user provided a specific file; use its parent
        model_dir = model_dir.parent
    if model_dir.name.lower() in {"smplx", "smpl"}:
        return model_dir.parent
    return model_dir


def matrix_to_axis_angle_torch(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices (N,3,3) to axis-angle (N,3) in radians.
    Assumes proper rotations. Handles small-angle stability.
    """
    if R.ndim != 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (N,3,3), got {tuple(R.shape)}")
    # Trace-based angle
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    # v from skew-symmetric part
    vx = R[:, 2, 1] - R[:, 1, 2]
    vy = R[:, 0, 2] - R[:, 2, 0]
    vz = R[:, 1, 0] - R[:, 0, 1]
    v = torch.stack([vx, vy, vz], dim=-1)  # (N,3)

    # 2*sin(theta)
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, 0.0, 1.0))
    denom = 2.0 * sin_theta

    # For small angles, use first-order approximation: axis ~ v / 2 (since sinθ ~ θ)
    eps = 1e-7
    small = denom.abs() < eps
    axis = torch.zeros_like(v)
    # Regular case
    axis[~small] = v[~small] / denom[~small].unsqueeze(-1)
    # Small-angle: fall back to normalized v (if nonzero), else zero
    if small.any():
        vv = v[small]
        norm_vv = torch.norm(vv, dim=-1, keepdim=True) + 1e-12
        axis_small = vv / norm_vv
        axis[small] = axis_small

    rotvec = axis * theta.unsqueeze(-1)
    # Clean NaNs if any (from degenerate inputs)
    rotvec = torch.nan_to_num(rotvec, nan=0.0, posinf=0.0, neginf=0.0)
    return rotvec


def extract_smpl_per_frame(frame_dict: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a PHALP per-frame dict, extract SMPL dict and return:
      betas: (10,)
      body_pose: (23, 3, 3)
      global_orient: (3, 3)
    Assumes single person per frame: frame['smpl'][0]
    """
    s0 = frame_dict["smpl"][0]

    betas = np.asarray(s0["betas"], dtype=np.float32).reshape(-1)  # (10,)

    body_pose = np.asarray(s0["body_pose"], dtype=np.float32)  # (23, 3, 3) for SMPL
    if body_pose.ndim != 3 or body_pose.shape[-2:] != (3, 3):
        raise ValueError(f"Unexpected body_pose shape {body_pose.shape}, expected (*,3,3)")

    global_orient = np.asarray(s0["global_orient"], dtype=np.float32)  # (1,3,3) or (3,3)
    if global_orient.ndim == 3 and global_orient.shape == (1, 3, 3):
        global_orient = global_orient[0]
    if global_orient.shape != (3, 3):
        raise ValueError(f"Unexpected global_orient shape {global_orient.shape}, expected (3,3)")

    return betas, body_pose, global_orient


def smpl_verts_from_phalp(pkl_dict: dict, smpl_model, device: torch.device) -> Tuple[List[int], np.ndarray]:
    # Build robust int-indexed frame map from arbitrary keys (full paths, filenames, or ints)
    frame_map: dict[int, dict] = {}
    for key, val in pkl_dict.items():
        idx = None
        if isinstance(key, str):
            try:
                idx = int(Path(key).stem)
            except Exception:
                idx = None
        elif isinstance(key, int):
            idx = key
        if idx is not None:
            frame_map[idx] = val

    keys = sorted(frame_map.keys())
    T = len(keys)
    betas_list: List[np.ndarray] = []
    body_list: List[np.ndarray] = []
    glob_list: List[np.ndarray] = []

    for k in keys:
        frame = frame_map[k]
        b, body_R, glob_R = extract_smpl_per_frame(frame)
        betas_list.append(b)         # (10,)
        body_list.append(body_R)     # (23,3,3)
        glob_list.append(glob_R)     # (3,3)

    betas = torch.as_tensor(np.stack(betas_list, axis=0), dtype=torch.float32, device=device)        # (T,10)
    body_R = torch.as_tensor(np.stack(body_list, axis=0), dtype=torch.float32, device=device)        # (T,23,3,3)
    glob_R = torch.as_tensor(np.stack(glob_list, axis=0), dtype=torch.float32, device=device)        # (T,3,3)

    # Convert rotation matrices to axis-angle for SMPL-X (pose2rot=True)
    T = glob_R.shape[0]
    glob_aa = matrix_to_axis_angle_torch(glob_R.view(-1, 3, 3)).view(T, 3)                            # (T,3)

    # Map SMPL (23) to SMPL-X body joints count
    J_model = int(getattr(smpl_model, "NUM_BODY_JOINTS", 21))
    body_aa = torch.zeros((T, J_model, 3), dtype=torch.float32, device=device)                       # (T,Jm,3)
    J_copy = min(body_R.shape[1], J_model)
    body_aa[:, :J_copy, :] = matrix_to_axis_angle_torch(body_R[:, :J_copy].reshape(-1, 3, 3)).view(T, J_copy, 3)

    # Flatten body pose
    body_flat = body_aa.view(T, -1)  # (T, 3*J_model)

    transl = torch.zeros((T, 3), dtype=torch.float32, device=device)

    # Zeros for hands/face/expression
    zeros_45 = torch.zeros((T, 45), dtype=torch.float32, device=device)
    zeros_3 = torch.zeros((T, 3), dtype=torch.float32, device=device)
    zeros_expr = torch.zeros((T, 10), dtype=torch.float32, device=device)

    with torch.no_grad():
        out = smpl_model(
            betas=betas,
            global_orient=glob_aa,
            body_pose=body_flat,
            transl=transl,
            left_hand_pose=zeros_45,
            right_hand_pose=zeros_45,
            jaw_pose=zeros_3,
            leye_pose=zeros_3,
            reye_pose=zeros_3,
            expression=zeros_expr,
            pose2rot=True,
        )
    verts = out.vertices.detach().cpu().numpy()  # (T, V, 3)
    return keys, verts


def view_two_sequences(vertsA: np.ndarray, faces: np.ndarray, vertsB: np.ndarray, offset: float = 1.5) -> None:
    try:
        import open3d as o3d
    except Exception as exc:
        print("Open3D is required for visualization: pip install open3d")
        raise

    T = min(len(vertsA), len(vertsB))
    if T == 0:
        print("No frames to visualize.")
        return

    meshA = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertsA[0]),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    meshB = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertsB[0] + np.array([offset, 0.0, 0.0])),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    meshA.compute_vertex_normals()
    meshB.compute_vertex_normals()

    state = {"t": 0, "playing": True}

    def set_frame(t: int, vis) -> None:
        t = int(np.clip(t, 0, T - 1))
        state["t"] = t
        meshA.vertices = o3d.utility.Vector3dVector(vertsA[t])
        meshB.vertices = o3d.utility.Vector3dVector(vertsB[t] + np.array([offset, 0.0, 0.0]))
        meshA.compute_vertex_normals()
        meshB.compute_vertex_normals()
        vis.update_geometry(meshA)
        vis.update_geometry(meshB)

    def anim(vis):
        if state["playing"]:
            nxt = state["t"] + 1
            if nxt >= T:
                nxt = 0
            set_frame(nxt, vis)
        return False

    def key_space(vis):
        state["playing"] = not state["playing"]
        return False

    def key_left(vis):
        set_frame(state["t"] - 1, vis)
        return False

    def key_right(vis):
        set_frame(state["t"] + 1, vis)
        return False

    def key_r(vis):
        set_frame(0, vis)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("PHALP SMPL A vs B", 1280, 720)
    vis.add_geometry(meshA)
    vis.add_geometry(meshB)

    vis.register_animation_callback(anim)
    vis.register_key_callback(ord(" "), key_space)
    vis.register_key_callback(263, key_left)   # left arrow
    vis.register_key_callback(262, key_right)  # right arrow
    vis.register_key_callback(ord("R"), key_r)

    vis.run()
    vis.destroy_window()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize PHALP sequences A vs B using SMPL-X (SMPL RotMat init).")
    ap.add_argument("--pkl_a", required=True, help="PHALP/4DHumans PKL for sequence A (joblib pickle).")
    ap.add_argument("--pkl_b", required=True, help="PHALP/4DHumans PKL for sequence B (joblib pickle).")
    ap.add_argument("--model_dir", default="body_models/smplx", help="Directory containing 'smplx' subfolder with SMPL-X model.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Torch device.")
    ap.add_argument("--offset", type=float, default=1.5, help="X offset to separate B from A in the viewer.")
    ap.add_argument("--rig_json", default=None, help="Stereo rig JSON to align cameras (optional).")
    ap.add_argument("--a_cam", type=int, choices=[0, 1], default=None, help="Camera index for sequence A (0 or 1).")
    ap.add_argument("--b_cam", type=int, choices=[0, 1], default=None, help="Camera index for sequence B (0 or 1).")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load PHALP PKLs
    A = load_pkl_any(args.pkl_a)
    B = load_pkl_any(args.pkl_b)

    # Resolve SMPL root for smplx.create
    model_root = resolve_smpl_root(Path(args.model_dir))

    smpl_model = smplx.create(
        model_path=str(model_root),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=1,
    ).to(device)
    faces = smpl_model.faces.astype(np.int32)

    keysA, vertsA = smpl_verts_from_phalp(A, smpl_model, device=device)
    keysB, vertsB = smpl_verts_from_phalp(B, smpl_model, device=device)
    print(f"Loaded A: T={len(keysA)} frames, B: T={len(keysB)} frames; verts shapes: {vertsA.shape}, {vertsB.shape}")

    # Optional: align both sequences to cam0 using rig extrinsics
    if args.rig_json is not None and (args.a_cam is not None or args.b_cam is not None):
        try:
            from golfcoach.io.rig_config import load_rig_config  # lazy import
            image_size, K0, D0, K1, D1, R, T = load_rig_config(args.rig_json)
            R = np.asarray(R, dtype=np.float32)
            T = np.asarray(T, dtype=np.float32).reshape(3)

            def to_cam0(verts: np.ndarray, cam_idx: int | None) -> np.ndarray:
                if cam_idx is None:
                    return verts
                if cam_idx == 0:
                    return verts
                # cam1 -> cam0: X0 = R^T (X1 - T)
                VT = verts - T.reshape(1, 1, 3)
                return VT @ R.T

            vertsA = to_cam0(vertsA, args.a_cam)
            vertsB = to_cam0(vertsB, args.b_cam)
            print("[align] Applied rig extrinsics to align sequences to cam0 frame.")
        except Exception as exc:
            print(f"[align] Failed to load/apply rig config: {exc}")

    view_two_sequences(vertsA, faces, vertsB, offset=float(args.offset))


if __name__ == "__main__":
    main()


