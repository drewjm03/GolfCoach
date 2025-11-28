"""
Mesh estimator using SMPL (via smplx).

First version: neutral SMPL body (betas=0, pose=0), scaled and translated
to roughly match the triangulated 3D skeleton.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from .smpl_model import SMPLModel


class MeshEstimator:
    """Estimate a human mesh (SMPL) from 3D joints."""

    def __init__(self, smpl_model_path: str | None = None, device: str = "cpu") -> None:
        """Initialize mesh estimator.

        Args:
            smpl_model_path: Path to SMPL model directory/file.
            device:          Device string, e.g. "cpu" or "cuda".
        """
        if smpl_model_path is None:
            raise ValueError("SMPL model path must be provided for real SMPL")

        self.smpl_model_path = str(smpl_model_path)
        self.device = str(device)

        # Instantiate SMPL wrapper
        self.model = SMPLModel(self.smpl_model_path, device=self.device)
        self.faces: Optional[np.ndarray] = self.model.faces
        if self.faces is None:
            raise RuntimeError("SMPLModel did not provide faces")

    def estimate_mesh(
        self,
        joints_3d_world: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple version:
        - neutral shape (betas=0)
        - neutral pose (all zeros)
        - scale to match skeleton height
        - rotate torso to match skeleton torso
        - translate so pelvis matches skeleton pelvis
        """
        # ----- 0) basic checks -----
        joints = np.asarray(joints_3d_world, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError("joints_3d_world must be shape (J, 3)")

        J = joints.shape[0]

        # Optional validity mask (for height, not indexing)
        if valid_mask is not None and len(valid_mask) == J:
            vm = np.asarray(valid_mask, dtype=bool)
            joints_for_height = joints[vm & np.isfinite(joints).all(axis=1)]
        else:
            joints_for_height = joints[np.isfinite(joints).all(axis=1)]

        if joints_for_height.shape[0] == 0:
            raise ValueError("No valid joints for mesh estimation")

        # ----- 1) pick pelvis & head & hips from *full* joints -----
        # RTM: 0 = nose, 11 = left_hip, 12 = right_hip
        def safe_get(idx: int, fallback: np.ndarray) -> np.ndarray:
            if idx < J and np.all(np.isfinite(joints[idx])):
                return joints[idx]
            return fallback

        # Fallback pelvis = centroid if hips are junk
        centroid = joints_for_height.mean(axis=0)
        left_hip_world  = safe_get(11, centroid)
        right_hip_world = safe_get(12, centroid)
        pelvis_world    = 0.5 * (left_hip_world + right_hip_world)
        head_world      = safe_get(0, pelvis_world + np.array([0.0, 1.7, 0.0]))

        # Skeleton "up" = pelvis → head
        up_world = head_world - pelvis_world
        up_norm = np.linalg.norm(up_world)
        if up_norm < 1e-6:
            up_world = np.array([0.0, 1.0, 0.0])
        else:
            up_world /= up_norm

        # Hips axis = left → right
        hips_vec = right_hip_world - left_hip_world
        hips_norm = np.linalg.norm(hips_vec)
        if hips_norm < 1e-6:
            hips_vec = np.array([1.0, 0.0, 0.0])
        else:
            hips_vec /= hips_norm

        # Build orthonormal target frame (x = hips, y = up, z = forward)
        y_tgt = up_world
        x_tgt = hips_vec
        z_tgt = np.cross(y_tgt, x_tgt)
        z_norm = np.linalg.norm(z_tgt)
        if z_norm < 1e-6:
            z_tgt = np.array([0.0, 0.0, 1.0])
        else:
            z_tgt /= z_norm
        # re-orthogonalize x
        x_tgt = np.cross(z_tgt, y_tgt)
        x_tgt /= np.linalg.norm(x_tgt) + 1e-8

        R_tgt = np.stack([x_tgt, y_tgt, z_tgt], axis=1)  # 3x3

        # ----- 2) estimate skeleton height (pelvis→head) -----
        skel_height = float(np.linalg.norm(head_world - pelvis_world))
        if skel_height < 0.3:
            skel_height = 1.7  # meters fallback

        # ----- 3) run SMPL-X in canonical pose -----
        B = 1
        betas = np.zeros((B, 10), dtype=np.float32)

        num_body = getattr(self.model, "num_body_joints", 21)
        pose_dim = 3 + 3 * int(num_body)
        pose = np.zeros((B, pose_dim), dtype=np.float32)  # T-pose / neutral
        transl = np.zeros((B, 3), dtype=np.float32)

        verts_smpl, joints_smpl = self.model(betas, pose, transl)
        verts_smpl  = verts_smpl[0]   # (N,3)
        joints_smpl = joints_smpl[0]  # (J_smpl,3)

        # Indices here assume: 0 = pelvis, 1/2 = left/right hip; adjust if needed
        pelvis_smpl = joints_smpl[0]
        left_hip_smpl  = joints_smpl[1]
        right_hip_smpl = joints_smpl[2]
        head_index = joints_smpl.shape[0] - 1  # last joint as rough "head"
        head_smpl = joints_smpl[head_index]

        up_rest = head_smpl - pelvis_smpl
        up_rest /= np.linalg.norm(up_rest) + 1e-8
        hips_rest = right_hip_smpl - left_hip_smpl
        hips_rest /= np.linalg.norm(hips_rest) + 1e-8

        y_rest = up_rest
        x_rest = hips_rest
        z_rest = np.cross(y_rest, x_rest)
        z_rest /= np.linalg.norm(z_rest) + 1e-8
        x_rest = np.cross(z_rest, y_rest)
        x_rest /= np.linalg.norm(x_rest) + 1e-8

        R_rest = np.stack([x_rest, y_rest, z_rest], axis=1)  # 3x3

        # ----- 4) compute scale and rotation -----
        smpl_height = float(np.linalg.norm(head_smpl - pelvis_smpl))
        if smpl_height < 1e-3:
            smpl_height = 1.7

        scale = skel_height / smpl_height

        # Rotation that takes SMPL rest torso frame into target torso frame
        R_world = R_tgt @ R_rest.T  # 3x3

        # ----- 5) apply transform to verts -----
        verts_centered = (verts_smpl - pelvis_smpl) * scale
        verts_rotated  = (R_world @ verts_centered.T).T
        verts_world    = verts_rotated + pelvis_world

        faces = self.faces
        if faces is None:
            raise RuntimeError("SMPL faces are not available")

        return verts_world.astype(np.float64), faces.astype(np.int32)


