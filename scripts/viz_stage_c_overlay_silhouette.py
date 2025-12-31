from __future__ import annotations

"""
Stage C visualizer: overlay fitted SMPL(-X) silhouettes onto the original video.

This script:
  - Loads Stage C outputs (frames.npy, betas.npy, global/body pose, trans_cam0.npy)
  - Reconstructs SMPL(-X) vertices in cam0 for each frame
  - Transforms to cam1 if requested
  - Renders soft silhouettes in an undistorted "virtual pinhole" space
  - Undistorts each video frame into the same space and overlays the silhouette

By default it uses frames.npy as absolute video frame indices:
  for each ti, seek video to frames[ti] (+ optional frame_offset) before drawing.
"""

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from apps.smpl_model import SMPLModel
from golfcoach.io.rig_config import load_rig_config
from golfcoach.pose3d.render_silhouette_pytorch3d import render_silhouette


def _default_smplx_model_path() -> Path:
    here = Path(__file__).resolve()
    root = here.parents[1]
    return root / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"


def _load_stage_c_outputs(stage_c_dir: Path):
    frames = np.load(stage_c_dir / "frames.npy").astype(np.int64)
    betas = np.load(stage_c_dir / "betas.npy").astype(np.float32)
    glob_aa = np.load(stage_c_dir / "global_orient_aa.npy").astype(np.float32)
    body_aa = np.load(stage_c_dir / "body_pose_aa.npy").astype(np.float32)
    trans_cam0 = np.load(stage_c_dir / "trans_cam0.npy").astype(np.float32)

    if frames.ndim != 1:
        raise ValueError(f"frames.npy must be 1D, got shape {frames.shape}")
    T = frames.shape[0]
    if glob_aa.shape != (T, 3):
        raise ValueError(f"global_orient_aa.npy must be (T,3), got {glob_aa.shape}")
    if trans_cam0.shape != (T, 3):
        raise ValueError(f"trans_cam0.npy must be (T,3), got {trans_cam0.shape}")
    if body_aa.shape[0] != T:
        raise ValueError(f"body_pose_aa.npy first dim must be T={T}, got {body_aa.shape}")

    return frames, betas, glob_aa, body_aa, trans_cam0


def _build_smpl_sequence(
    betas: np.ndarray,
    glob_aa: np.ndarray,
    body_aa: np.ndarray,
    trans_cam0: np.ndarray,
    device: torch.device,
):
    """
    Run SMPL-X once over the whole sequence to obtain per-frame vertices in cam0.

    Returns
    -------
    verts_cam0 : torch.Tensor
        (T, V, 3) vertices in cam0 coordinates.
    faces : np.ndarray
        (F, 3) triangle indices as int32.
    """
    smpl_path = _default_smplx_model_path()
    if not smpl_path.exists():
        raise FileNotFoundError(
            f"Default SMPL-X model not found at {smpl_path}. "
            "Place SMPLX_NEUTRAL.npz there or adjust the path."
        )

    smpl_wrapper = SMPLModel(str(smpl_path), device=str(device))
    smpl_layer = smpl_wrapper.smpl
    faces = smpl_wrapper.faces.astype(np.int32)

    T = glob_aa.shape[0]
    betas_t = torch.as_tensor(
        betas.reshape(1, -1).repeat(T, axis=0), dtype=torch.float32, device=device
    )  # (T,10)
    glob_t = torch.as_tensor(glob_aa, dtype=torch.float32, device=device)  # (T,3)

    T_body, J_body, _ = body_aa.shape
    if T_body != T:
        raise ValueError(f"body_aa first dim {T_body} != T={T}")
    body_t = torch.as_tensor(body_aa, dtype=torch.float32, device=device)  # (T,J,3)

    trans_t = torch.as_tensor(trans_cam0, dtype=torch.float32, device=device)  # (T,3)

    body_flat = body_t.view(T, -1)  # (T, 3*J_body)

    out = smpl_layer(
        betas=betas_t,
        global_orient=glob_t,
        body_pose=body_flat,
        transl=trans_t,
        left_hand_pose=torch.zeros((T, 45), dtype=torch.float32, device=device),
        right_hand_pose=torch.zeros((T, 45), dtype=torch.float32, device=device),
        jaw_pose=torch.zeros((T, 3), dtype=torch.float32, device=device),
        leye_pose=torch.zeros((T, 3), dtype=torch.float32, device=device),
        reye_pose=torch.zeros((T, 3), dtype=torch.float32, device=device),
        expression=torch.zeros((T, 10), dtype=torch.float32, device=device),
        pose2rot=True,
    )
    verts_cam0 = out.vertices  # (T,V,3)

    return verts_cam0, faces


def _undistort_frame(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Undistort a video frame into the same virtual pinhole space as the masks.
    """
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    # Use newCameraMatrix=K to keep same intrinsics as Stage C
    undistorted = cv2.undistort(frame_bgr, K, D, None, K)
    return undistorted


def _overlay_silhouette_on_frame(
    frame_bgr: np.ndarray,
    sil: torch.Tensor,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
    edge_only: bool = True,
) -> np.ndarray:
    """
    Overlay a rendered silhouette onto a BGR frame.

    If edge_only is True, draws only the contour; otherwise alpha-blends fill.
    """
    h, w = frame_bgr.shape[:2]
    sil_np = sil.detach().cpu().numpy()

    if sil_np.shape != (h, w):
        sil_np = cv2.resize(sil_np, (w, h), interpolation=cv2.INTER_LINEAR)

    sil_norm = np.clip(sil_np, 0.0, 1.0)
    sil_mask = (sil_norm > 0.5).astype(np.uint8) * 255

    if edge_only:
        edges = cv2.Canny(sil_mask, 50, 150)
        overlay = frame_bgr.copy()
        overlay[edges > 0] = color
        return overlay

    # Filled alpha-blend
    colored = np.zeros_like(frame_bgr, dtype=np.float32)
    colored[..., 0] = color[0] / 255.0
    colored[..., 1] = color[1] / 255.0
    colored[..., 2] = color[2] / 255.0

    mask_f = (sil_norm[..., None] > 0.5).astype(np.float32)
    frame_f = frame_bgr.astype(np.float32) / 255.0

    blended = frame_f * (1.0 - alpha * mask_f) + colored * (alpha * mask_f)
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return blended


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay Stage C SMPL(-X) silhouettes onto a video for visual verification."
    )
    ap.add_argument("--video", required=True, help="Input video path (left or right).")
    ap.add_argument(
        "--stage_c_dir",
        required=True,
        help="Directory containing Stage C outputs (frames.npy, betas.npy, pose, trans_cam0).",
    )
    ap.add_argument(
        "--rig_json",
        required=True,
        help="Stereo rig calibration JSON (same as Stage C).",
    )
    ap.add_argument("--out", required=True, help="Output overlay video path.")
    ap.add_argument(
        "--camera_idx",
        type=int,
        choices=[0, 1],
        default=0,
        help="Camera index: 0 (cam0/left) or 1 (cam1/right).",
    )
    ap.add_argument(
        "--frame_offset",
        type=int,
        default=0,
        help=(
            "Optional offset added to frames.npy when seeking into the video. "
            "Default 0 (assume frames.npy contains absolute video frame indices)."
        ),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Torch device for SMPL and rendering, e.g. "cuda" or "cpu" (default: "cuda").',
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha for filled silhouette overlay (if edge_only is disabled).",
    )
    ap.add_argument(
        "--edge_only",
        action="store_true",
        help="If set, draw only the silhouette contour instead of a filled region.",
    )
    args = ap.parse_args()

    stage_c_dir = Path(args.stage_c_dir)
    frames, betas, glob_aa, body_aa, trans_cam0 = _load_stage_c_outputs(stage_c_dir)

    # Load rig calibration
    image_size, K0, D0, K1, D1, R, T_vec = load_rig_config(args.rig_json)
    w_calib, h_calib = int(image_size[0]), int(image_size[1])

    device = torch.device(args.device)

    # Build SMPL sequence in cam0
    verts_cam0, faces = _build_smpl_sequence(
        betas=betas,
        glob_aa=glob_aa,
        body_aa=body_aa,
        trans_cam0=trans_cam0,
        device=device,
    )
    T_seq, V, _ = verts_cam0.shape
    if T_seq != frames.shape[0]:
        raise RuntimeError(f"SMPL sequence length {T_seq} != frames length {frames.shape[0]}")

    # Transform to selected camera
    faces_t = torch.as_tensor(faces, dtype=torch.int64, device=device)
    R_t = torch.as_tensor(R, dtype=torch.float32, device=device)
    T_t = torch.as_tensor(T_vec, dtype=torch.float32, device=device).view(1, 1, 3)

    if args.camera_idx == 0:
        K = K0
        D = D0
        verts_cam = verts_cam0
    else:
        K = K1
        D = D1
        verts_cam = (verts_cam0 @ R_t.T) + T_t  # (T,V,3)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    current_video_idx = 0

    # Main loop over Stage C frames
    for ti in range(T_seq):
        target_idx = int(frames[ti] + args.frame_offset)

        # Seek to desired frame index
        while current_video_idx < target_idx:
            ret = cap.grab()
            if not ret:
                cap.release()
                writer.release()
                raise RuntimeError(
                    f"Video ended while seeking to frame {target_idx}. "
                    f"Reached {current_video_idx}."
                )
            current_video_idx += 1

        # Decode the target frame
        ret, frame = cap.read()
        if not ret:
            break
        current_video_idx += 1

        # Undistort into virtual pinhole space
        frame_u = _undistort_frame(frame, K, D)

        # Render silhouette in the same undistorted space
        v_cam = verts_cam[ti]  # (V,3) on device
        sil = render_silhouette(
            verts_cam=v_cam,
            faces=faces_t,
            K=K,
            image_size=image_size,
            device=device,
        )

        overlay = _overlay_silhouette_on_frame(
            frame_u,
            sil,
            color=(0, 255, 0),
            alpha=args.alpha,
            edge_only=args.edge_only,
        )

        # If calibration size differs from video size, resize overlay back
        if overlay.shape[1] != width or overlay.shape[0] != height:
            overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_LINEAR)

        writer.write(overlay)

    cap.release()
    writer.release()
    print(f"Wrote Stage C silhouette overlay video to: {args.out}")


if __name__ == "__main__":
    main()




