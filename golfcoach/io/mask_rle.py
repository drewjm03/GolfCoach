from __future__ import annotations

"""
Helpers for working with PHALP / 4DHumans COCO-style RLE masks.

This module provides:

    - decode_phalp_mask:   RLE dict -> (H, W) uint8 mask
    - undistort_mask:      apply OpenCV undistortion to a binary mask
"""

from typing import Dict, Any

import numpy as np
import cv2
from pycocotools import mask as mask_utils


def decode_phalp_mask(m: Dict[str, Any]) -> np.ndarray:
    """
    Decode a PHALP / COCO RLE mask to a 2D uint8 array.

    Parameters
    ----------
    m : dict
        Dictionary with keys:
          - 'size': [H, W]
          - 'counts': bytes or str (RLE)

    Returns
    -------
    np.ndarray
        Binary mask of shape (H, W), dtype uint8 in {0, 1}.
    """
    if "size" not in m or "counts" not in m:
        raise KeyError("RLE mask dict must contain 'size' and 'counts' keys")

    size = m["size"]
    counts = m["counts"]

    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        raise ValueError(f"Expected 'size' as [H, W], got {size!r}")

    if isinstance(counts, str):
        counts_bytes = counts.encode("utf-8")
    else:
        counts_bytes = counts

    rle = {"size": list(size), "counts": counts_bytes}
    arr = mask_utils.decode(rle)  # (H, W, 1) uint8 in {0,1}
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def undistort_mask(mask: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Undistort a binary mask using OpenCV's initUndistortRectifyMap + remap.

    This creates a "virtual pinhole" view compatible with rendering that
    ignores lens distortion.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask of shape (H, W), uint8 or bool.
    K : np.ndarray
        3x3 camera intrinsic matrix.
    D : np.ndarray
        Distortion coefficients for OpenCV (k1, k2, p1, p2, [k3, ...]).

    Returns
    -------
    np.ndarray
        Undistorted binary mask of shape (H, W), uint8 in {0, 1}.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    H, W = mask.shape
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).reshape(-1)

    # Use the same intrinsic matrix for the virtual pinhole camera.
    new_K = K.copy()

    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K,
        distCoeffs=D,
        R=np.eye(3, dtype=np.float64),
        newCameraMatrix=new_K,
        size=(W, H),
        m1type=cv2.CV_32FC1,
    )

    mask_u = cv2.remap(
        src=mask.astype(np.uint8),
        map1=map1,
        map2=map2,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Ensure binary {0,1}
    mask_u = (mask_u > 0).astype(np.uint8)
    return mask_u




