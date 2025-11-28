"""
Lightweight SMPL-X fitting loop on 3D joints.

This module defines a small SMPLFitter that optimizes SMPL-X global
orientation, body pose (axis-angle), and optionally translation to make
SMPL-X joints match a set of target 3D joints (e.g. triangulated RTM
keypoints in the cam0/world frame).

It is intentionally simple: a few steps of Adam per frame, with an L2
loss on joints plus a small pose regularizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn, optim

from .smpl_model import SMPLModel


# Mapping from RTM 3D joints to SMPL-X joint indices.
# This follows the layout you provided.
RTM_TO_SMPLX: Dict[int, int] = {
    # Torso / head
    0: 15,  # nose        -> head

    # Shoulders / arms
    5: 16,  # L shoulder  -> L shoulder
    6: 17,  # R shoulder  -> R shoulder
    7: 18,  # L elbow     -> L elbow
    8: 19,  # R elbow     -> R elbow
    9: 20,  # L wrist     -> L wrist
    10: 21,  # R wrist    -> R wrist

    # Hips / legs
    11: 1,  # L hip       -> L hip
    12: 2,  # R hip       -> R hip
    13: 4,  # L knee      -> L knee
    14: 5,  # R knee      -> R knee
    15: 7,  # L ankle     -> L ankle
    16: 8,  # R ankle     -> R ankle
}


@dataclass
class FitterConfig:
    lr: float = 5e-2
    num_iters: int = 10
    pose_reg: float = 1e-3
    betas_reg: float = 1e-4
    optimize_betas: bool = False  # set True if you want to adjust shape as well


class SMPLFitter(nn.Module):
    """
    Minimal SMPL-X fitting helper.

    Usage:
        smpl_model = SMPLModel(model_path, device="cuda" or "cpu")
        joint_map = { rt_idx: smpl_idx, ... }
        fitter = SMPLFitter(smpl_model, joint_map, device="cuda")

        verts_world, joints_world = fitter.fit(joints_3d_world)
    """

    def __init__(
        self,
        smpl_model: SMPLModel,
        joint_map: Dict[int, int],
        device: str = "cpu",
        config: Optional[FitterConfig] = None,
    ) -> None:
        super().__init__()
        self.model = smpl_model
        self.device = torch.device(device)
        self.config = config or FitterConfig()

        # Underlying smplx layer
        self.smpl_layer = self.model.smpl
        self.num_body = int(getattr(self.model, "num_body_joints", 21))

        # Parameters to optimize
        self.betas = nn.Parameter(torch.zeros(1, 10, device=self.device))
        self.pose = nn.Parameter(torch.zeros(1, 3 + 3 * self.num_body, device=self.device))
        self.transl = nn.Parameter(torch.zeros(1, 3, device=self.device))

        # Joint index mapping: RTM index -> SMPL-X index
        # Only a subset is needed; you can extend as desired.
        # Example: pelvis (0), left_hip (11->1), right_hip (12->2), head (0->last)
        self.joint_map = dict(joint_map)

    def _build_optimizer(self) -> optim.Optimizer:
        params = [self.pose, self.transl]
        if self.config.optimize_betas:
            params.append(self.betas)
        return optim.Adam(params, lr=self.config.lr)

    def _select_joints(
        self, joints_world: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select corresponding SMPL-X and target joints according to joint_map.

        Args:
            joints_world: (1, J_rt, 3) tensor of target joints in world frame.
        Returns:
            joints_smpl_sel: (1, K, 3)  - subset of SMPL-X joints
            joints_tgt_sel:  (1, K, 3)  - corresponding target RTM joints
            joints_smpl_all: (1, J_smpl, 3) - all SMPL-X joints (for pelvis)
        """
        if not self.joint_map:
            raise ValueError("joint_map is empty; cannot build joint loss")

        rt_indices = list(self.joint_map.keys())
        smpl_indices = list(self.joint_map.values())

        rt_idx = torch.tensor(rt_indices, dtype=torch.long, device=self.device)
        smpl_idx = torch.tensor(smpl_indices, dtype=torch.long, device=self.device)

        # Target joints
        joints_tgt = joints_world[:, rt_idx, :]  # (1, K, 3)

        # Current SMPL-X joints (all)
        out = self.smpl_layer(
            betas=self.betas,
            global_orient=self.pose[:, :3],
            body_pose=self.pose[:, 3:],
            transl=self.transl,
            pose2rot=True,
        )
        joints_smpl_all = out.joints  # (1, J_smpl, 3)
        joints_smpl_sel = joints_smpl_all[:, smpl_idx, :]  # (1, K, 3)
        return joints_smpl_sel, joints_tgt, joints_smpl_all

    def fit(
        self,
        joints_3d_world: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit SMPL-X to target 3D joints for a few iterations.

        Args:
            joints_3d_world: (J_rt, 3) array of target joints in world frame.
            valid_mask: Optional (J_rt,) boolean mask of valid joints.
        Returns:
            verts_world: (N, 3)
            joints_world: (J_smpl, 3)
        """
        joints_np = np.asarray(joints_3d_world, dtype=np.float32)
        if joints_np.ndim != 2 or joints_np.shape[1] != 3:
            raise ValueError("joints_3d_world must be shape (J, 3)")

        # Build torch tensor (batch size 1)
        joints_world = torch.from_numpy(joints_np).to(self.device)[None, ...]  # (1, J, 3)

        # Optionally zero-out invalid joints by setting mask weight to 0
        if valid_mask is not None and len(valid_mask) == joints_np.shape[0]:
            vm = torch.as_tensor(valid_mask, dtype=torch.float32, device=self.device)[None, :, None]
        else:
            vm = torch.ones_like(joints_world[..., :1])

        opt = self._build_optimizer()

        for _ in range(self.config.num_iters):
            opt.zero_grad()

            # Forward through SMPL-X and select joints
            joints_smpl_sel, joints_tgt_sel, joints_smpl_all = self._select_joints(joints_world)

            # Apply validity mask on target side (broadcast along xyz)
            w = vm[:, list(self.joint_map.keys()), :]
            diff = (joints_smpl_sel - joints_tgt_sel) * w
            loss_j = (diff ** 2).sum(-1).mean()

            # Pelvis term: midpoint of RTM hips (11, 12) ↔ SMPL-X pelvis (0)
            pelvis_loss = torch.tensor(0.0, device=self.device)
            if joints_world.shape[1] > 12:
                lhip = joints_world[:, 11, :]  # (1, 3)
                rhip = joints_world[:, 12, :]  # (1, 3)
                valid_l = torch.isfinite(lhip).all(dim=-1)
                valid_r = torch.isfinite(rhip).all(dim=-1)
                valid_pelvis = (valid_l & valid_r).float().view(-1, 1)
                if valid_pelvis.sum() > 0:
                    pelvis_rtm = 0.5 * (lhip + rhip)
                    pelvis_smpl = joints_smpl_all[:, 0, :]  # SMPL-X pelvis joint
                    diff_p = (pelvis_smpl - pelvis_rtm) * valid_pelvis
                    pelvis_loss = (diff_p ** 2).sum(-1).mean()

            total_joint_loss = loss_j + pelvis_loss

            # Regularizers
            pose_reg = self.config.pose_reg * (self.pose[:, 3:] ** 2).mean()
            if self.config.optimize_betas:
                betas_reg = self.config.betas_reg * (self.betas ** 2).mean()
            else:
                betas_reg = torch.tensor(0.0, device=self.device)

            loss = total_joint_loss + pose_reg + betas_reg
            loss.backward()
            opt.step()

        # Final forward to get full verts/joints in world frame
        out = self.smpl_layer(
            betas=self.betas,
            global_orient=self.pose[:, :3],
            body_pose=self.pose[:, 3:],
            transl=self.transl,
            pose2rot=True,
        )
        verts = out.vertices[0].detach().cpu().numpy()
        joints_out = out.joints[0].detach().cpu().numpy()
        return verts, joints_out


