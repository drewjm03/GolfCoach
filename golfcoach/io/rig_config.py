from __future__ import annotations

"""
Helpers for loading stereo rig calibration from JSON.
"""

from pathlib import Path
import json

import numpy as np


def load_rig_config(path: str):
    """
    Reads rig JSON and returns:

        image_size (w, h),
        K0, D0, K1, D1, R, T

    The JSON is expected to follow the structure:

        rig['stereo_calib']['camera_0']['K']
        rig['stereo_calib']['camera_0']['dist_coeffs']
        rig['stereo_calib']['camera_1']['K']
        rig['stereo_calib']['camera_1']['dist_coeffs']
        rig['stereo_calib']['camera_1']['R']
        rig['stereo_calib']['camera_1']['t']

    where camera_1's R and t describe the transform from cam0 to cam1:

        X_cam1 = R @ X_cam0 + T

    Parameters
    ----------
    path : str
        Path to the rig JSON file.

    Returns
    -------
    image_size : tuple[int, int]
        (width, height)
    K0, D0, K1, D1, R, T : np.ndarray
        Camera intrinsics, distortion coefficients, stereo rotation and translation.
        T is returned as shape (3,), in cam0 frame.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        rig = json.load(f)

    image_size_raw = rig["image_size"]
    if len(image_size_raw) != 2:
        raise ValueError(f"Expected image_size with 2 elements, got {image_size_raw}")
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))

    stereo = rig["stereo_calib"]
    cam0 = stereo["camera_0"]
    cam1 = stereo["camera_1"]

    K0 = np.asarray(cam0["K"], dtype=np.float64)
    D0 = np.asarray(cam0["dist_coeffs"], dtype=np.float64)

    K1 = np.asarray(cam1["K"], dtype=np.float64)
    D1 = np.asarray(cam1["dist_coeffs"], dtype=np.float64)

    R = np.asarray(cam1["R"], dtype=np.float64)
    T = np.asarray(cam1["t"], dtype=np.float64).reshape(3)

    return image_size, K0, D0, K1, D1, R, T








