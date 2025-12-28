from __future__ import annotations

"""
Utilities for loading PHALP / 4DHumans per-frame pickle outputs.

These files are typically written via joblib and contain a dict:

    frame_path (str) -> frame_data (dict)

where `frame_path` looks like:

    'outputs//_DEMO/clip_name/img/000104.jpg'

We convert this to:

    frame_idx (int) -> frame_data

with `frame_idx` parsed from the basename ('000104' -> 104).
"""

from pathlib import Path
from collections.abc import Mapping
from typing import Dict, Any

import joblib
import numpy as np


def load_phalp_tracks(pkl_path: str) -> dict[int, dict]:
    """
    Load a PHALP / 4DHumans result pickle.

    Returns
    -------
    dict[int, dict]
        Mapping from frame index (e.g. 104) to the original frame_data dict.

    Notes
    -----
    - The underlying file is usually a joblib-compressed pickle.
    - Keys in the raw dict are frame-path strings; we parse the integer
      frame index from Path(key).stem, e.g. '000104.jpg' -> 104.
    - If a frame has multiple people, the frame dict will contain lists
      of length > 1 for fields like '2d_joints', '3d_joints', etc.
      By default we will use person index 0 when extracting joints,
      but this function keeps the full frame dict so that a different
      person or tid can be selected later.
    """
    p = Path(pkl_path)
    raw = joblib.load(p)

    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected mapping at top level in {p}, got {type(raw)}")

    tracks: dict[int, dict] = {}
    for key, frame_data in raw.items():
        stem = Path(key).stem  # e.g. '000104'
        try:
            frame_idx = int(stem)
        except ValueError as exc:
            raise ValueError(f"Could not parse frame index from key '{key}'") from exc

        # Keep the entire frame_data dict; we will select person index later
        if not isinstance(frame_data, Mapping):
            raise TypeError(
                f"Expected frame_data to be mapping for key '{key}', "
                f"got {type(frame_data)}"
            )
        tracks[frame_idx] = dict(frame_data)

    return tracks


def extract_2d_joints(frame_data: dict, person_i: int = 0) -> np.ndarray:
    """
    Extract 2D joints for a given person from a frame dict.

    Parameters
    ----------
    frame_data : dict
        The per-frame dict from PHALP, as stored in the loaded tracks.
    person_i : int, optional
        Index into the per-frame person list (default: 0).

    Returns
    -------
    np.ndarray
        Array of shape (J, 2), dtype float32.

    Notes
    -----
    PHALP stores `frame_data['2d_joints'][person_i]` either as:

    - shape (2*J,) : flattened (x0, y0, x1, y1, ..., x_{J-1}, y_{J-1})
    - shape (J, 2) : already in (x, y) form
    """
    if "2d_joints" not in frame_data:
        raise KeyError("Expected '2d_joints' key in frame_data")

    joints_list = frame_data["2d_joints"]
    if not isinstance(joints_list, (list, tuple)):
        raise TypeError(
            f"Expected '2d_joints' to be list/tuple, got {type(joints_list)}"
        )
    if not (0 <= person_i < len(joints_list)):
        raise IndexError(
            f"person_i={person_i} out of range for 2d_joints list of length {len(joints_list)}"
        )

    arr = np.asarray(joints_list[person_i], dtype=np.float32)

    if arr.ndim == 1:
        # Flattened (2*J,) -> (J, 2)
        if arr.size % 2 != 0:
            raise ValueError(
                f"Expected even length for flattened 2d_joints, got {arr.size}"
            )
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[1] == 2:
            # Already (J, 2)
            pass
        elif arr.shape[0] == 2:
            # Sometimes stored as (2, J) -> transpose
            arr = arr.T
        else:
            raise ValueError(
                f"Unexpected 2D shape for 2d_joints: {arr.shape}, "
                "expected (J,2) or (2,J)"
            )
    else:
        raise ValueError(
            f"Unexpected ndim for 2d_joints: {arr.ndim}, expected 1 or 2"
        )

    return arr.astype(np.float32, copy=False)


def extract_3d_joints(frame_data: dict, person_i: int = 0) -> np.ndarray:
    """
    Extract 3D joints for a given person from a frame dict.

    Parameters
    ----------
    frame_data : dict
        The per-frame dict from PHALP, as stored in the loaded tracks.
    person_i : int, optional
        Index into the per-frame person list (default: 0).

    Returns
    -------
    np.ndarray
        Array of shape (J, 3), dtype float32.

    Notes
    -----
    PHALP stores `frame_data['3d_joints'][person_i]` typically as (J, 3).
    We simply coerce to that shape.
    """
    if "3d_joints" not in frame_data:
        raise KeyError("Expected '3d_joints' key in frame_data")

    joints_list = frame_data["3d_joints"]
    if not isinstance(joints_list, (list, tuple)):
        raise TypeError(
            f"Expected '3d_joints' to be list/tuple, got {type(joints_list)}"
        )
    if not (0 <= person_i < len(joints_list)):
        raise IndexError(
            f"person_i={person_i} out of range for 3d_joints list of length {len(joints_list)}"
        )

    arr = np.asarray(joints_list[person_i], dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Unexpected shape for 3d_joints: {arr.shape}, expected (J, 3)"
        )

    return arr.astype(np.float32, copy=False)


def extract_smpl(frame_data: Dict[str, Any], person_i: int = 0) -> Dict[str, Any]:
    """
    Extract SMPL / SMPL-X parameters for a given person from a frame dict.

    Parameters
    ----------
    frame_data : dict
        Per-frame dict from PHALP / 4DHumans output.
    person_i : int, optional
        Person index within this frame (default: 0).

    Returns
    -------
    dict
        SMPL parameter dict, typically containing:

            - 'global_orient': (1, 3, 3) or (3, 3) rotation matrix
            - 'body_pose':    (J, 3, 3) rotation matrices
            - 'betas':        (10,) shape coefficients
            - possibly other fields like 'pose2rot', etc.

    Notes
    -----
    This function simply selects the appropriate entry from
    frame_data['smpl'][person_i] without modifying shapes, so that
    downstream code can adapt to the exact structure.
    """
    if "smpl" not in frame_data:
        raise KeyError("Expected 'smpl' key in frame_data")

    smpl_list = frame_data["smpl"]
    if not isinstance(smpl_list, (list, tuple)):
        raise TypeError(f"Expected 'smpl' to be list/tuple, got {type(smpl_list)}")

    if not (0 <= person_i < len(smpl_list)):
        raise IndexError(
            f"person_i={person_i} out of range for 'smpl' list of length {len(smpl_list)}"
        )

    smpl_entry = smpl_list[person_i]
    if not isinstance(smpl_entry, Mapping):
        raise TypeError(
            f"Expected smpl[{person_i}] to be a mapping, got {type(smpl_entry)}"
        )

    return dict(smpl_entry)
