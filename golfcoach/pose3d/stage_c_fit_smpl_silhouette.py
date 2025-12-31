from __future__ import annotations

"""
Stage C: Fit a single SMPL(-X) body per frame in cam0 using stereo silhouettes.

Inputs
------
- Left/right PHALP / 4DHumans PKLs containing SMPL rotations + masks per frame.
- Stereo rig JSON (our calibrated K/D + extrinsics).

Outputs
-------
Directory `out_dir` containing:

- frames.npy              (T,)
- betas.npy               (10,)
- global_orient_aa.npy    (T, 3)
- body_pose_aa.npy        (T, J_body, 3)
- trans_cam0.npy          (T, 3)
- loss_history.json       (with per-iteration losses)

This does NOT overwrite Stage A/B; it uses PHALP's SMPL as initialization
and fits a proper translation in cam0 under our calibrated cameras.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import json

import cv2
import numpy as np
import torch
from torch import nn

from pytorch3d.transforms import matrix_to_axis_angle

from apps.smpl_model import SMPLModel
from golfcoach.io.rig_config import load_rig_config
from golfcoach.io.phalp_pkl import load_phalp_tracks, extract_smpl
from golfcoach.io.mask_rle import decode_phalp_mask, undistort_mask
from golfcoach.pose3d.render_silhouette_pytorch3d import render_silhouette


def _default_smplx_model_path() -> Path:
    """
    Heuristic to find the default SMPL model shipped with this repo.

    We expect:
        body_models/smpl/SMPL_NEUTRAL.pkl
    at the project root.
    """
    here = Path(__file__).resolve()
    root = here.parents[2]
    candidates = [
        root / "body_models" / "smpl" / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Return the primary expected path so the error message is informative
    return candidates[0]


def _estimate_initial_translation_cam0(
    bbox: np.ndarray,
    K0: np.ndarray,
    person_height_m: float = 1.75,
) -> np.ndarray:
    """
    Estimate a plausible initial translation in cam0 from the 2D bbox.

    We use a simple pinhole depth heuristic:

        z ≈ fy * H_person / h_pixels

    and set x/y so that the pelvis (at origin in SMPL) projects near the
    bbox center.
    """
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if bbox.size != 4:
        raise ValueError(f"Expected bbox as [x, y, w, h], got shape {bbox.shape}")

    x, y, w, h = bbox
    fy = float(K0[1, 1])
    fx = float(K0[0, 0])
    cx = float(K0[0, 2])
    cy = float(K0[1, 2])

    h_pixels = max(float(h), 1.0)
    z = max(fy * float(person_height_m) / h_pixels, 0.5)

    # Center of bbox in pixels
    u = x + 0.5 * w
    v = y + 0.5 * h

    # Back-project assuming pelvis at origin in SMPL coords
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy

    return np.array([X, Y, z], dtype=np.float32)


def _to_axis_angle_from_rot_mats(
    glob_R_list: List[np.ndarray],
    body_R_list: List[np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert lists of rotation matrices from PHALP to axis-angle form.

    Returns
    -------
    glob_aa : (T, 3) torch.Tensor
    body_aa : (T, J_body, 3) torch.Tensor
    """
    # Global orientations
    glob_R = np.stack(glob_R_list, axis=0)  # (T, 3, 3) or (T, 1, 3, 3)
    if glob_R.ndim == 4:
        glob_R = glob_R[:, 0]
    T = glob_R.shape[0]

    glob_R_t = torch.as_tensor(glob_R, dtype=torch.float32, device=device)  # (T, 3, 3)
    glob_aa = matrix_to_axis_angle(glob_R_t.reshape(-1, 3, 3)).reshape(T, 3)

    # Body pose rotations: expect list of (J, 3, 3) or (1, J, 3, 3)
    body_R = np.stack(body_R_list, axis=0)
    if body_R.ndim == 4 and body_R.shape[1] == 1:
        body_R = body_R[:, 0]  # (T, J, 3, 3)
    if body_R.ndim != 4:
        raise ValueError(f"Expected body_R as (T, J, 3, 3), got shape {body_R.shape}")

    T_body, J_body, _, _ = body_R.shape
    assert T_body == T

    body_R_t = torch.as_tensor(body_R, dtype=torch.float32, device=device)
    body_aa = matrix_to_axis_angle(body_R_t.view(-1, 3, 3)).view(T, J_body, 3)

    return glob_aa, body_aa


def _compute_temporal_second_order(x: torch.Tensor) -> torch.Tensor:
    """
    Second-order temporal smoothness:

        sum_t ||x[t+1] - 2 x[t] + x[t-1]||^2

    returned as mean over all valid t and dimensions.
    """
    if x.shape[0] < 3:
        return torch.zeros([], dtype=x.dtype, device=x.device)
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return (d2 ** 2).mean()


def _soft_iou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft IoU loss between predicted and target masks.

    Both inputs are expected in [0, 1] with shape (T, H, W).
    Returns:
        loss = 1 - mean IoU over all frames.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in IoU: pred {pred.shape}, target {target.shape}")

    T = pred.shape[0]
    if T == 0:
        return torch.zeros([], dtype=pred.dtype, device=pred.device)

    pred_flat = pred.view(T, -1)
    target_flat = target.view(T, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = intersection / (union + eps)
    return 1.0 - iou.mean()


def fit_smpl_silhouette_stereo(
    pkl_left: str,
    pkl_right: str,
    rig_json: str,
    out_dir: str,
    person_i_left: int = 0,
    person_i_right: int = 0,
    max_frames: Optional[int] = None,
    device: str = "cuda",
) -> None:
    """
    Fit a single SMPL(-X) body across time using stereo silhouettes.

    Parameters
    ----------
    pkl_left, pkl_right : str
        Paths to left/right PHALP / 4DHumans result PKLs.
    rig_json : str
        Path to rig configuration JSON (see golfcoach.io.rig_config).
    out_dir : str
        Output directory for Stage C results.
    person_i_left, person_i_right : int, optional
        Person indices in left/right PKLs (default: 0).
    max_frames : int or None, optional
        If provided, limit optimization to the first N common frames.
    device : str, optional
        Torch device string, e.g. "cuda" or "cpu".
    """
    dev = torch.device(device)

    # ------------------------------------------------------------------
    # 1) Load calibration and PHALP tracks
    # ------------------------------------------------------------------
    image_size, K0, D0, K1, D1, R, T = load_rig_config(rig_json)
    w, h = int(image_size[0]), int(image_size[1])

    tracks_left = load_phalp_tracks(pkl_left)
    tracks_right = load_phalp_tracks(pkl_right)

    frame_idxs_left = set(tracks_left.keys())
    frame_idxs_right = set(tracks_right.keys())
    common_frames = sorted(frame_idxs_left & frame_idxs_right)

    if max_frames is not None:
        common_frames = common_frames[: max_frames]

    if not common_frames:
        raise RuntimeError("No overlapping frame indices between left and right PKLs.")

    Tlen = len(common_frames)

    # ------------------------------------------------------------------
    # 2) Extract SMPL params, masks, and build initializations
    # ------------------------------------------------------------------
    betas_list: List[np.ndarray] = []
    glob_R_list: List[np.ndarray] = []
    body_R_list: List[np.ndarray] = []
    trans_init_list: List[np.ndarray] = []

    # Target masks (undistorted, in the rig's "virtual pinhole" space)
    masks_left = np.zeros((Tlen, h, w), dtype=np.uint8)
    masks_right = np.zeros((Tlen, h, w), dtype=np.uint8)

    for ti, frame_idx in enumerate(common_frames):
        frame_left = tracks_left[frame_idx]
        frame_right = tracks_right[frame_idx]

        # --- SMPL from left view as initialization ---
        smpl_left = extract_smpl(frame_left, person_i_left)

        glob_R = np.asarray(smpl_left["global_orient"])
        if glob_R.shape == (1, 3, 3):
            glob_R = glob_R[0]
        glob_R_list.append(glob_R.astype(np.float32))

        body_R = np.asarray(smpl_left["body_pose"])
        if body_R.ndim == 4 and body_R.shape[0] == 1:
            body_R = body_R[0]
        body_R_list.append(body_R.astype(np.float32))

        betas = np.asarray(smpl_left["betas"], dtype=np.float32).reshape(-1)
        betas_list.append(betas)

        # --- bbox-based translation heuristic from left PKL ---
        if "bbox" not in frame_left:
            raise KeyError("Expected 'bbox' in PHALP frame_data for translation init")
        bbox_list = frame_left["bbox"]
        if not isinstance(bbox_list, (list, tuple)):
            raise TypeError(f"Expected 'bbox' to be list/tuple, got {type(bbox_list)}")
        if not (0 <= person_i_left < len(bbox_list)):
            raise IndexError(
                f"person_i_left={person_i_left} out of range for bbox list of length {len(bbox_list)}"
            )
        bbox = np.asarray(bbox_list[person_i_left], dtype=np.float32)
        trans_init = _estimate_initial_translation_cam0(bbox, K0)
        trans_init_list.append(trans_init)

        # --- Masks: decode + undistort for both views ---
        def _get_mask(frame_data: Dict[str, Any], person_i: int) -> Dict[str, Any]:
            # Try a few common keys
            if "masks" in frame_data:
                masks_field = frame_data["masks"]
            elif "mask" in frame_data:
                masks_field = frame_data["mask"]
            else:
                raise KeyError("Expected 'masks' or 'mask' in PHALP frame_data")

            if not isinstance(masks_field, (list, tuple)):
                raise TypeError(
                    f"Expected masks to be list/tuple, got {type(masks_field)}"
                )
            if not (0 <= person_i < len(masks_field)):
                raise IndexError(
                    f"person_i={person_i} out of range for masks list of length {len(masks_field)}"
                )
            return masks_field[person_i]

        mL_rle = _get_mask(frame_left, person_i_left)
        mR_rle = _get_mask(frame_right, person_i_right)

        # Debug print to inspect left mask structure before decoding
        if isinstance(mL_rle, dict):
            print("mL_rle keys:", list(mL_rle.keys()))
            for k in ("segmentation", "mask", "rle", "rle_mask"):
                if k in mL_rle and isinstance(mL_rle[k], dict):
                    print(k, "keys:", list(mL_rle[k].keys()))
        else:
            print("mL_rle type:", type(mL_rle))


        print("mL_rle is list, len =", len(mL_rle))
        print("first elem type =", type(mL_rle[0]))
        if isinstance(mL_rle[0], (list, tuple)) and len(mL_rle[0]) > 0:
            print("first elem[0] type =", type(mL_rle[0][0]))

        mL = decode_phalp_mask(mL_rle)
        mR = decode_phalp_mask(mR_rle)

        mL_u = undistort_mask(mL, K0, D0)
        mR_u = undistort_mask(mR, K1, D1)

        # Resize to rig image size if needed
        if mL_u.shape != (h, w):
            mL_u = cv2.resize(mL_u, (w, h), interpolation=cv2.INTER_NEAREST)
        if mR_u.shape != (h, w):
            mR_u = cv2.resize(mR_u, (w, h), interpolation=cv2.INTER_NEAREST)

        masks_left[ti] = mL_u
        masks_right[ti] = mR_u

    # Fixed betas across the clip: median over frames (left view)
    betas_np = np.median(np.stack(betas_list, axis=0), axis=0).astype(np.float32)

    # Convert rotations to axis-angle
    glob_aa_init, body_aa_init_phalp = _to_axis_angle_from_rot_mats(
        glob_R_list, body_R_list, dev
    )

    # ------------------------------------------------------------------
    # 3) Instantiate SMPL(-X) model and prepare optimization variables
    # ------------------------------------------------------------------
    smpl_model_path = _default_smplx_model_path()
    if not smpl_model_path.exists():
        raise FileNotFoundError(
            f"Default SMPL model not found at {smpl_model_path}. "
            "Place SMPL_NEUTRAL.pkl under body_models/smpl (or model_models/smpl)."
        )

    smpl_wrapper = SMPLModel(str(smpl_model_path), device=device)
    smpl_layer = smpl_wrapper.smpl
    faces_np = smpl_wrapper.faces.astype(np.int64)
    faces_t = torch.as_tensor(faces_np, dtype=torch.int64, device=dev)

    T_frames = Tlen
    J_body_phalp = body_aa_init_phalp.shape[1]
    J_body_model = int(getattr(smpl_layer, "NUM_BODY_JOINTS", J_body_phalp))

    # Body pose variable has shape (T, J_body_model, 3); initialize from PHALP
    body_aa_init = torch.zeros(
        (T_frames, J_body_model, 3), dtype=torch.float32, device=dev
    )
    J_copy = min(J_body_phalp, J_body_model)
    body_aa_init[:, :J_copy, :] = body_aa_init_phalp[:, :J_copy, :]

    # Optimization variables
    glob_aa = nn.Parameter(glob_aa_init.clone())  # (T, 3)
    body_aa = nn.Parameter(body_aa_init.clone())  # (T, J_body_model, 3)
    trans_cam0 = nn.Parameter(
        torch.as_tensor(
            np.stack(trans_init_list, axis=0), dtype=torch.float32, device=dev
        )
    )  # (T, 3)

    # Fixed betas shared across time
    betas_t = torch.as_tensor(betas_np[None, :], dtype=torch.float32, device=dev)  # (1, 10)

    # Target masks on device in [0, 1]
    target_left = torch.as_tensor(masks_left, dtype=torch.float32, device=dev)
    target_right = torch.as_tensor(masks_right, dtype=torch.float32, device=dev)

    # ------------------------------------------------------------------
    # 4) Loss weights
    # ------------------------------------------------------------------
    lambda_sil = 1.0
    lambda_trans = 50.0
    lambda_pose = 5.0
    lambda_prior = 1e-3

    R_t = torch.as_tensor(R, dtype=torch.float32, device=dev)  # (3, 3)
    T_t = torch.as_tensor(T, dtype=torch.float32, device=dev).view(1, 1, 3)  # (1,1,3) for broadcast

    def forward_and_loss() -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Run SMPL(-X), render silhouettes in both cameras, and compute losses.
        """
        Tcur = glob_aa.shape[0]

        betas_batch = betas_t.expand(Tcur, -1)  # (T, 10)
        global_orient = glob_aa  # (T, 3)
        body_pose_flat = body_aa.view(Tcur, -1)  # (T, 3 * J_body_model)
        
        print("global_orient", global_orient.shape)
        print("body_pose", body_pose_flat.shape)
        print("betas", betas_batch.shape)
        print("transl", None if trans_cam0 is None else trans_cam0.shape)
        
        out = smpl_layer(
            betas=betas_batch,
            global_orient=global_orient,
            body_pose=body_pose_flat,
            transl=trans_cam0,
            pose2rot=True,
        )
        verts_cam0 = out.vertices  # (T, V, 3)

        # Transform to cam1
        verts_cam1 = (verts_cam0 @ R_t.T) + T_t  # broadcast over T,V

        # Render silhouettes per frame
        sil_left_list: List[torch.Tensor] = []
        sil_right_list: List[torch.Tensor] = []

        for t_idx in range(Tcur):
            v0 = verts_cam0[t_idx]
            v1 = verts_cam1[t_idx]

            sil0 = render_silhouette(v0, faces_t, K0, image_size, device=dev)
            sil1 = render_silhouette(v1, faces_t, K1, image_size, device=dev)

            sil_left_list.append(sil0)
            sil_right_list.append(sil1)

        sil_left = torch.stack(sil_left_list, dim=0)  # (T, H, W)
        sil_right = torch.stack(sil_right_list, dim=0)

        # Silhouette losses
        L_sil_left = _soft_iou_loss(sil_left, target_left)
        L_sil_right = _soft_iou_loss(sil_right, target_right)
        L_sil = L_sil_left + L_sil_right

        # Temporal smoothness
        L_trans = _compute_temporal_second_order(trans_cam0)
        L_glob = _compute_temporal_second_order(glob_aa)
        L_pose_temp = _compute_temporal_second_order(body_aa.view(Tcur, -1))

        # Pose magnitude prior (exclude global)
        L_prior = (body_aa ** 2).mean()

        loss = (
            lambda_sil * L_sil
            + lambda_trans * L_trans
            + lambda_pose * (L_glob + L_pose_temp)
            + lambda_prior * L_prior
        )

        return loss, {
            "L_total": float(loss.detach().cpu()),
            "L_sil": float(L_sil.detach().cpu()),
            "L_sil_left": float(L_sil_left.detach().cpu()),
            "L_sil_right": float(L_sil_right.detach().cpu()),
            "L_trans": float(L_trans.detach().cpu()),
            "L_glob": float(L_glob.detach().cpu()),
            "L_pose_temp": float(L_pose_temp.detach().cpu()),
            "L_prior": float(L_prior.detach().cpu()),
        }

    # ------------------------------------------------------------------
    # 5) Two-stage optimization (Adam)
    # ------------------------------------------------------------------
    loss_history: Dict[str, Any] = {"stage1": [], "stage2": []}

    # Stage 1: coarse, optimize only translation + global orientation
    optimizer1 = torch.optim.Adam(
        [trans_cam0, glob_aa],
        lr=5e-3,
    )

    num_iters_stage1 = 400
    for it in range(num_iters_stage1):
        optimizer1.zero_grad(set_to_none=True)
        loss, components = forward_and_loss()
        loss.backward()
        optimizer1.step()
        loss_history["stage1"].append(components)

    # Stage 2: full optimization (translation + global + body pose)
    optimizer2 = torch.optim.Adam(
        [trans_cam0, glob_aa, body_aa],
        lr=2e-3,
    )
    num_iters_stage2 = 1200
    for it in range(num_iters_stage2):
        optimizer2.zero_grad(set_to_none=True)
        loss, components = forward_and_loss()
        loss.backward()
        optimizer2.step()
        loss_history["stage2"].append(components)

    # ------------------------------------------------------------------
    # 6) Save outputs
    # ------------------------------------------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frames_arr = np.asarray(common_frames, dtype=np.int32)
    np.save(out_path / "frames.npy", frames_arr)
    np.save(out_path / "betas.npy", betas_np.astype(np.float32))

    glob_aa_np = glob_aa.detach().cpu().numpy().astype(np.float32)  # (T, 3)
    body_aa_np = body_aa.detach().cpu().numpy().astype(np.float32)  # (T, J_body, 3)
    trans_np = trans_cam0.detach().cpu().numpy().astype(np.float32)  # (T, 3)

    np.save(out_path / "global_orient_aa.npy", glob_aa_np)
    np.save(out_path / "body_pose_aa.npy", body_aa_np)
    np.save(out_path / "trans_cam0.npy", trans_np)

    with (out_path / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)



