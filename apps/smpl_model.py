"""
SMPL model wrapper using smplx.

This module wraps an SMPL model (via smplx) and exposes:
  - self.faces: triangle indices
  - forward(betas, pose, transl) -> (vertices, joints)
"""

from __future__ import annotations

from typing import Tuple, Optional
import os

import numpy as np
import torch
from torch import nn
import smplx


class SMPLModel(nn.Module):
    """SMPL model wrapper using smplx."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        super().__init__()
        self.model_path = str(model_path)
        self.device = torch.device(device)

        # Determine directory containing SMPL files
        model_dir = (
            os.path.dirname(self.model_path)
            if os.path.isfile(self.model_path)
            else self.model_path
        )

        # Create SMPL-X model
        self.smpl = smplx.create(
            model_path=model_dir,
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            batch_size=1,
        ).to(self.device)

        # Number of body joints (e.g. 21 for SMPL-X)
        self.num_body_joints: int = int(getattr(self.smpl, "NUM_BODY_JOINTS", 21))

        # Faces as numpy array
        self.faces: np.ndarray = self.smpl.faces.astype(np.int32)

    @torch.no_grad()
    def forward(
        self,
        betas: np.ndarray,
        pose: np.ndarray,
        transl: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            betas:  (B, 10)
            pose:   (B, 3 + 3 * num_body_joints) axis-angle
            transl: (B, 3) optional global translation

        Returns:
            vertices: (B, N, 3)
            joints:   (B, J, 3) SMPL-X joints
        """
        betas_t = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
        pose_t = torch.as_tensor(pose, dtype=torch.float32, device=self.device)

        # Validate pose dimensionality for SMPL-X
        expected_pose_dim = 3 + 3 * self.num_body_joints
        if pose_t.shape[1] != expected_pose_dim:
            raise ValueError(
                f"Expected pose dim {expected_pose_dim} for smplx "
                f"but got {pose_t.shape[1]}"
            )

        if transl is not None:
            transl_t = torch.as_tensor(transl, dtype=torch.float32, device=self.device)
        else:
            transl_t = torch.zeros(
                (betas_t.shape[0], 3), dtype=torch.float32, device=self.device
            )

        # SMPL-X expects:
        #   global_orient: (B, 3)
        #   body_pose:     (B, 3 * num_body_joints)
        global_orient = pose_t[:, :3]
        body_pose = pose_t[:, 3:]

        out = self.smpl(
            betas=betas_t,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl_t,
            pose2rot=True,
        )

        verts = out.vertices.detach().cpu().numpy()  # (B, N, 3)
        joints = out.joints.detach().cpu().numpy()  # (B, J, 3)
        return verts, joints



