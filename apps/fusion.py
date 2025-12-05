"""
Joint fusion utilities for multi-view 3D pose.

This module provides a simple confidence-weighted fusion of joints
from two cameras into a common frame (cam0), using the stereo extrinsics.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fuse_two_views_joints(
    joints0: np.ndarray,
    conf0: np.ndarray,
    joints1_cam1: np.ndarray,
    conf1: np.ndarray,
    R_10: np.ndarray,
    T_10: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse joints from cam0 and cam1 into cam0/world frame.

    Args:
        joints0:      (J, 3) joints in cam0 frame
        conf0:        (J,)   confidences for cam0 joints
        joints1_cam1: (J, 3) joints in cam1 frame
        conf1:        (J,)   confidences for cam1 joints
        R_10:         (3, 3) rotation from cam1 -> cam0
        T_10:         (3,)   translation from cam1 -> cam0

    Returns:
        joints_fused: (J, 3) fused 3D joints in cam0 frame
        conf_fused:   (J,)   fused confidences
    """
    joints0 = np.asarray(joints0, dtype=np.float32)
    conf0 = np.asarray(conf0, dtype=np.float32)
    joints1_cam1 = np.asarray(joints1_cam1, dtype=np.float32)
    conf1 = np.asarray(conf1, dtype=np.float32)

    if joints0.shape != joints1_cam1.shape:
        raise ValueError("joints0 and joints1_cam1 must have the same shape")
    if conf0.shape[0] != joints0.shape[0] or conf1.shape[0] != joints0.shape[0]:
        raise ValueError("conf0/conf1 must match number of joints")

    J = joints0.shape[0]

    # Transform cam1 joints into cam0 frame
    R_10 = np.asarray(R_10, dtype=np.float32).reshape(3, 3)
    T_10 = np.asarray(T_10, dtype=np.float32).reshape(3)
    joints1 = (R_10 @ joints1_cam1.T + T_10.reshape(3, 1)).T  # (J, 3)

    joints_fused = np.zeros_like(joints0, dtype=np.float32)
    conf_fused = np.zeros_like(conf0, dtype=np.float32)

    for j in range(J):
        p0 = joints0[j]
        c0 = float(conf0[j])
        p1 = joints1[j]
        c1 = float(conf1[j])

        # If either set has NaN, treat conf as 0
        if not np.isfinite(p0).all():
            c0 = 0.0
        if not np.isfinite(p1).all():
            c1 = 0.0

        if c0 <= 0.0 and c1 <= 0.0:
            joints_fused[j] = np.nan
            conf_fused[j] = 0.0
            continue

        w0 = max(c0, 0.0)
        w1 = max(c1, 0.0)
        wsum = w0 + w1
        if wsum <= 0.0:
            joints_fused[j] = np.nan
            conf_fused[j] = 0.0
            continue

        joints_fused[j] = (w0 * p0 + w1 * p1) / wsum
        # Simple fusion of confidence; you can switch to max(c0, c1) if preferred
        conf_fused[j] = wsum / 2.0

    return joints_fused, conf_fused



