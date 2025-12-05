"""
HybrIK (or similar SMPL regressor) wrapper.

This module provides a thin, project-specific wrapper that hides all
HybrIK internals and exposes a clean interface for per-frame inference.

NOTE: This is a stub – you still need to hook it up to your actual
HybrIK implementation and checkpoint.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import torch


class HybrIKRunner:
    """
    Wrapper for a single-camera HybrIK model.

    The intent is:
      - Construct one instance per camera.
      - Call infer(frame_bgr, K) each frame.
      - Get back 3D joints and SMPL params in that camera's frame.
    """

    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str,
        device: str = "cuda",
        transform: Optional[object] = None,
        img_size: Tuple[int, int] = (256, 192),
    ) -> None:
        """
        Load HybrIK model (or similar SMPL regressor).

        Args:
            cfg_path:  Path to HybrIK config file.
            ckpt_path: Path to HybrIK checkpoint file.
            device:    "cuda" or "cpu".
        """
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)
        self.transform = transform  # should handle (img, bbox) -> (tensor, meta)
        # (W, H) expected input size; adjust to match your HybrIK config
        self.img_size = img_size

        # TODO: import HybrIK, build model, load weights, etc.
        # Example (pseudo-code):
        #   from hybrik import build_model
        #   self.model = build_model(cfg_path, ckpt_path, device=self.device)
        # For now we keep this as a stub; caller should construct and assign.
        self.model: Optional[torch.nn.Module] = None

    def _default_bbox(self, img_h: int, img_w: int) -> np.ndarray:
        """
        Fallback bbox: whole image, in (cx, cy, w, h, rot) format
        matching HybrIK's expected meta.
        """
        cx = img_w * 0.5
        cy = img_h * 0.5
        w = float(img_w)
        h = float(img_h)
        rot = 0.0
        return np.array([cx, cy, w, h, rot], dtype=np.float32)

    @torch.no_grad()
    def infer(
        self,
        frame_bgr: np.ndarray,
        K: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run HybrIK on one RGB frame.

        Args:
            frame_bgr: (H, W, 3) uint8 BGR image from OpenCV.
            K:         Optional (3, 3) camera intrinsics for this view (unused by default).
            bbox:      Optional bbox in the format expected by self.transform.
                        If None, the whole image is used as a bbox.

        Returns a dict in this camera's frame:
            {
              "joints_3d": (J, 3) float32  # 3D joints (cam coords)
              "confs":     (J,)   float32  # per-joint confidence 0–1
              "betas":     (10,)  float32  # SMPL/SMPL-X shape
              "pose":      (P,)   float32  # axis-angle pose (global + body)
              "transl":    (3,)   float32  # root translation in this cam frame, if available
            }
        or raises if model/output are not usable.
        """
        if self.model is None:
            raise NotImplementedError(
                "HybrIKRunner.model is None. "
                "Construct and assign a HybrIK model before calling infer()."
            )

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame passed to HybrIKRunner.infer")

        img = frame_bgr.copy()
        img_h, img_w = img.shape[:2]

        # Convert BGR -> RGB because most HybrIK pipelines expect RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ----- 1. Prepare bbox & transform -----
        if bbox is None:
            bbox = self._default_bbox(img_h, img_w)  # (cx, cy, w, h, rot)

        if self.transform is None:
            # Simple resize + normalization fallback.
            in_w, in_h = self.img_size
            resized = cv2.resize(img_rgb, (in_w, in_h))
            inp = resized.astype(np.float32) / 255.0
            inp = (inp - 0.5) / 0.5  # [-1, 1] normalization
            inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
            meta: Dict[str, Any] = {
                "bbox": bbox,
                "img_h": img_h,
                "img_w": img_w,
            }
            if K is not None:
                meta["K"] = np.asarray(K, dtype=np.float32)
        else:
            # Official HybrIK transform typically returns (tensor, meta)
            # e.g. inp, meta = self.transform(img_rgb, bbox)
            inp, meta = self.transform(img_rgb, bbox)

        inp = inp.to(self.device, non_blocking=True)

        # ----- 2. Forward pass -----
        output = self.model(inp, meta=meta)

        # ----- 3. Extract 3D joints -----
        joints_3d: Optional[np.ndarray] = None
        jts_tensor = None

        # Common key names across HybrIK forks
        if "pred_xyz_24_struct" in output:
            jts_tensor = output["pred_xyz_24_struct"]
        elif "pred_xyz_24" in output:
            jts_tensor = output["pred_xyz_24"]
        elif "pred_xyz_jts_24" in output:
            jts_tensor = output["pred_xyz_jts_24"]
        else:
            for k, v in output.items():
                if "pred_xyz" in k and isinstance(v, torch.Tensor):
                    jts_tensor = v
                    break

        if jts_tensor is None:
            raise ValueError(
                f"Could not find 3D joints in HybrIK output keys: {list(output.keys())}"
            )

        jts = jts_tensor.detach().cpu().numpy()  # (B, J, C)
        if jts.ndim != 3 or jts.shape[0] < 1:
            raise ValueError(f"Unexpected joints tensor shape from HybrIK: {jts.shape}")

        jts = jts[0]  # single batch → (J, C)
        if jts.shape[1] < 3:
            raise ValueError(f"Unexpected joints channel dimension from HybrIK: {jts.shape}")

        joints_3d = jts[:, :3].astype(np.float32)

        # ----- 4. Confidence scores -----
        # If 4th channel exists, interpret as confidence
        if jts.shape[1] >= 4:
            confs = jts[:, 3].astype(np.float32)
        else:
            if "joint_conf" in output:
                jc = output["joint_conf"].detach().cpu().numpy()[0]
                confs = jc.reshape(-1).astype(np.float32)
            else:
                confs = np.ones(joints_3d.shape[0], dtype=np.float32)

        # ----- 5. SMPL pose & betas -----
        if "smpl_pose" not in output or "smpl_beta" not in output:
            raise ValueError("HybrIK output missing 'smpl_pose' or 'smpl_beta'")

        pose = output["smpl_pose"].detach().cpu().numpy()[0].astype(np.float32)
        betas = output["smpl_beta"].detach().cpu().numpy()[0].astype(np.float32)

        # ----- 6. Root translation in camera coords (optional) -----
        root_trans_cam: Optional[np.ndarray] = None
        for key in ("cam_root", "root_cam", "pred_camera"):
            if key in output:
                root_trans_cam = (
                    output[key].detach().cpu().numpy()[0].astype(np.float32)
                )
                break

        return {
            "joints_3d": joints_3d,
            "confs": confs,
            "pose": pose,
            "betas": betas,
            "transl": root_trans_cam,
        }



