import torch

def matrix_to_axis_angle(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert rotation matrices to axis-angle.
    R: (..., 3, 3)
    returns: (..., 3) axis-angle vectors
    """
    # Clamp trace for numerical stability
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # (...,)

    # Axis from skew-symmetric part
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)  # (..., 3)

    sin_theta = torch.sin(theta)
    denom = 2.0 * sin_theta[..., None]
    axis = axis / torch.clamp(denom, min=eps)

    # For very small angles, fall back to zeros
    small = theta < 1e-4
    axis = torch.where(small[..., None], torch.zeros_like(axis), axis)

    return axis * theta[..., None]
