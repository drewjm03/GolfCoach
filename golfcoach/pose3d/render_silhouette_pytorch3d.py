from __future__ import annotations

"""
Differentiable silhouette rendering using PyTorch3D.

This module exposes a small helper:

    render_silhouette(verts_cam, faces, K, image_size, device)

Inputs
------
verts_cam : (V, 3) torch.Tensor
    Vertices in camera coordinates.
faces : (F, 3) np.ndarray or torch.Tensor
    Triangle indices.
K : (3, 3) np.ndarray or torch.Tensor
    Camera intrinsic matrix with focal lengths / principal point in pixels.
image_size : tuple[int, int]
    (width, height) of the target image.
device : str | torch.device
    Device on which to run the renderer.

Output
------
silhouette : (H, W) torch.Tensor
    Soft silhouette values in [0, 1].
"""

from typing import Tuple, Union

import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    BlendParams,
)


def _build_camera(
    K: Union[np.ndarray, torch.Tensor],
    image_size: Tuple[int, int],
    device: torch.device,
) -> PerspectiveCameras:
    """
    Build a PyTorch3D PerspectiveCameras from OpenCV intrinsics.

    We keep everything in screen/pixel space (in_ndc=False) so that
    (fx, fy, cx, cy) are interpreted as pixels.
    """
    if not torch.is_tensor(K):
        K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
    else:
        K_t = K.to(device=device, dtype=torch.float32)

    fx = K_t[0, 0]
    fy = K_t[1, 1]
    cx = K_t[0, 2]
    cy = K_t[1, 2]

    w, h = int(image_size[0]), int(image_size[1])

    cameras = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        image_size=((h, w),),  # PyTorch3D expects (H, W)
        in_ndc=False,
        device=device,
    )
    return cameras


def render_silhouette(
    verts_cam: torch.Tensor,
    faces: Union[np.ndarray, torch.Tensor],
    K: Union[np.ndarray, torch.Tensor],
    image_size: Tuple[int, int],
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """
    Render a soft silhouette for a single mesh in a single view.

    Parameters
    ----------
    verts_cam : torch.Tensor
        (V, 3) vertices in camera coordinates.
    faces : np.ndarray or torch.Tensor
        (F, 3) triangle indices.
    K : np.ndarray or torch.Tensor
        (3, 3) camera intrinsics.
    image_size : tuple[int, int]
        (width, height) of the output silhouette.
    device : str or torch.device, optional
        Device for computation (default: "cuda").

    Returns
    -------
    torch.Tensor
        (H, W) soft silhouette in [0, 1].
    """
    if verts_cam.ndim != 2 or verts_cam.shape[1] != 3:
        raise ValueError(f"verts_cam must be (V, 3), got {verts_cam.shape}")

    dev = torch.device(device)
    verts = verts_cam.to(dev, dtype=torch.float32)

    if torch.is_tensor(faces):
        faces_t = faces.to(dev, dtype=torch.int64)
    else:
        faces_t = torch.as_tensor(faces, dtype=torch.int64, device=dev)

    cameras = _build_camera(K, image_size, dev)

    w, h = int(image_size[0]), int(image_size[1])
    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=0.0,
        faces_per_pixel=50,
    )

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftSilhouetteShader(blend_params=blend_params)

    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    meshes = Meshes(verts=[verts], faces=[faces_t])
    # renderer returns (1, H, W, 4); alpha channel is silhouette
    images = renderer(meshes)
    alpha = images[0, ..., 3]
    return alpha




