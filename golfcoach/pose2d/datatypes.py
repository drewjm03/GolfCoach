from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Pose2DSequence:
    """
    22-keypoint output: 17 body + 5 club
    """
    t: np.ndarray          # (T,)
    frame_idx: np.ndarray  # (T,)
    kpts: np.ndarray       # (T, 22, 2) in pixel coords
    conf: np.ndarray       # (T, 22) confidence [0..1] (or model score)
    bbox: np.ndarray       # (T, 4) [x1,y1,x2,y2] used for pose crop
    image_size: tuple[int, int]  # (W,H)
    joint_names: list[str]



