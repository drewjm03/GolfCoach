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
        - translate so pelvis matches skeleton pelvis
        """
        pts = np.asarray(joints_3d_world, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("joints_3d_world must be shape (J, 3)")

        # Apply optional validity mask
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape[0] == pts.shape[0]:
                pts = pts[mask]

        # Drop NaN / non-finite joints
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]

        if pts.shape[0] == 0:
            raise ValueError("No valid joints for mesh estimation")

        # 1) pick pelvis & head indices from RTM layout
        # RTM_POSE: 0=nose, 11=left_hip, 12=right_hip, 15/16=ankles
        try:
            left_hip = pts[11]
            right_hip = pts[12]
            pelvis_world = 0.5 * (left_hip + right_hip)
            head_world = pts[0]  # nose as proxy for head
        except IndexError:
            # fallback to centroid + rough up vector
            pelvis_world = pts.mean(axis=0)
            head_world = pelvis_world + np.array([0.0, 1.7, 0.0], dtype=np.float64)

        # 2) estimate body height from skeleton (pelvis->head)
        skel_height = float(np.linalg.norm(head_world - pelvis_world))
        if skel_height < 0.3:  # degenerate
            skel_height = 1.7  # meters (fallback)

        # 3) build neutral SMPL-X params
        B = 1
        betas = np.zeros((B, 10), dtype=np.float32)
        # For SMPL-X: 3 (global) + 3 * num_body_joints (e.g. 21*3 = 63) = 66
        num_body = getattr(self.model, "num_body_joints", 21)
        pose_dim = 3 + 3 * int(num_body)
        pose = np.zeros((B, pose_dim), dtype=np.float32)  # all joints in rest pose
        transl = np.zeros((B, 3), dtype=np.float32)  # we'll place in world later

        # 4) run SMPL in its own canonical frame
        verts_smpl, joints_smpl = self.model(betas, pose, transl)  # (B, N, 3), (B, J, 3)
        verts_smpl = verts_smpl[0]
        joints_smpl = joints_smpl[0]

        # SMPL pelvis is usually joint 0; head approx joint 15 or last joint
        pelvis_smpl = joints_smpl[0]
        head_index = 15 if joints_smpl.shape[0] > 15 else (joints_smpl.shape[0] - 1)
        head_smpl = joints_smpl[head_index]

        smpl_height = float(np.linalg.norm(head_smpl - pelvis_smpl))
        if smpl_height < 1e-3:
            smpl_height = 1.7

        scale = skel_height / smpl_height

        # 5) apply scale, then translate to pelvis_world
        verts_world = (verts_smpl - pelvis_smpl) * scale + pelvis_world

        faces = self.faces
        if faces is None:
            raise RuntimeError("SMPL faces are not available")

        return verts_world.astype(np.float64), faces.astype(np.int32)


