from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from golfcoach.io.npz_io import load_npz
from golfcoach.rig.rig_loader import load_rig_json, projection_matrix


# You MUST ensure these indices match the 17-body keypoint ordering that GolfPose outputs.
# After Stage A works, print joint_names from the NPZ to confirm and update these.
# Here is a common H36M-style 17 ordering placeholder:
BODY_JOINT_NAMES_17 = [
    "pelvis",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]

# Bones as (name, i, j) indices in the 17-body subset
BONES_17 = [
    ("pelvis_rhip", 0, 1),
    ("rhip_rknee", 1, 2),
    ("rknee_rankle", 2, 3),
    ("pelvis_lhip", 0, 4),
    ("lhip_lknee", 4, 5),
    ("lknee_lankle", 5, 6),
    ("pelvis_spine", 0, 7),
    ("spine_thorax", 7, 8),
    ("thorax_neck", 8, 9),
    ("neck_head", 9, 10),
    ("thorax_lshoulder", 8, 11),
    ("lshoulder_lelbow", 11, 12),
    ("lelbow_lwrist", 12, 13),
    ("thorax_rshoulder", 8, 14),
    ("rshoulder_relbow", 14, 15),
    ("relbow_rwrist", 15, 16),
]


def triangulate_point(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Linear triangulation using DLT.
    x1, x2 are (2,) pixel coordinates.
    """
    x1h = np.array([x1[0], x1[1], 1.0], dtype=np.float64)
    x2h = np.array([x2[0], x2[1], 1.0], dtype=np.float64)

    A = np.stack([
        x1h[0] * P1[2] - P1[0],
        x1h[1] * P1[2] - P1[1],
        x2h[0] * P2[2] - P2[0],
        x2h[1] * P2[2] - P2[1],
    ], axis=0)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / (X[3] + 1e-12)
    return X.astype(np.float64)


def estimate_bone_lengths_from_pose2d_npz(
    left_npz: str,
    right_npz: str,
    rig_json: str,
    conf_thresh: float = 0.4,
    sample_stride: int = 2,
    max_samples: int = 2000,
) -> Dict[str, float]:
    """
    Uses only the 17 body joints (first 17 of the 22 keypoints).
    Triangulates joint centers and returns robust median lengths per bone.
    """
    L = load_npz(left_npz)
    R = load_npz(right_npz)
    rig = load_rig_json(rig_json)

    P1 = projection_matrix(rig.left)
    P2 = projection_matrix(rig.right)

    kL = L["kpts"][:, :17, :]   # (T,17,2)
    cL = L["conf"][:, :17]      # (T,17)
    kR = R["kpts"][:, :17, :]
    cR = R["conf"][:, :17]

    T = min(kL.shape[0], kR.shape[0])
    kL = kL[:T]
    cL = cL[:T]
    kR = kR[:T]
    cR = cR[:T]

    lengths: Dict[str, List[float]] = {name: [] for (name, _, _) in BONES_17}

    n = 0
    for t in range(0, T, sample_stride):
        if n >= max_samples:
            break

        # triangulate all joints for this frame where confident in both
        X = np.full((17, 3), np.nan, dtype=np.float64)
        valid = (cL[t] >= conf_thresh) & (cR[t] >= conf_thresh)

        for j in range(17):
            if valid[j]:
                X[j] = triangulate_point(P1, P2, kL[t, j], kR[t, j])

        # compute bone lengths where both endpoints valid
        for bone_name, i, j in BONES_17:
            if np.isfinite(X[i, 0]) and np.isfinite(X[j, 0]):
                lengths[bone_name].append(float(np.linalg.norm(X[i] - X[j])))

        n += 1

    out: Dict[str, float] = {}
    for bone_name, vals in lengths.items():
        if len(vals) < 10:
            continue
        v = np.array(vals, dtype=np.float64)

        # Robust trimming using MAD
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-12
        z = np.abs(v - med) / (1.4826 * mad)
        v2 = v[z < 3.5]  # keep inliers
        out[bone_name] = float(np.median(v2)) if v2.size > 0 else float(med)

    return out







