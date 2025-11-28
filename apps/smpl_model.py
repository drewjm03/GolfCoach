"""
SMPL model wrapper (stub).

This module defines a thin wrapper class for an SMPL model. The actual
SMPL loading and inference will be hooked up later. For now, this
provides a stable interface for downstream code.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


class SMPLModel:
    """Stub SMPL model wrapper.

    The real implementation should load an official SMPL model and expose
    faces / forward() that returns vertices and joints.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """Initialize SMPL model.

        Args:
            model_path: Path to SMPL model file(s).
            device:     Device string, e.g. "cpu" or "cuda".
        """
        self.model_path = str(model_path)
        self.device = str(device)
        # Placeholder for faces (F, 3) int array when real model is loaded
        self.faces: Optional[np.ndarray] = None

    def forward(
        self,
        betas: np.ndarray,
        pose: np.ndarray,
        transl: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SMPL forward pass (stub).

        Args:
            betas:  (B, 10) shape parameters
            pose:   (B, 72) or (B, 24, 3) pose parameters
            transl: (B, 3) optional global translation

        Returns:
            vertices: (B, N, 3)
            joints:   (B, J, 3)
        """
        raise NotImplementedError("Hook up SMPL here")


