"""
Mesh estimator (SMPL fitting stub).

This module defines a MeshEstimator that will eventually fit an SMPL
mesh to 3D joints. For now, it only provides the interface and raises
NotImplementedError when called.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import open3d as o3d


class MeshEstimator:
    """Estimate a human mesh from 3D joints (currently convex-hull stub)."""

    def __init__(self, smpl_model_path: str | None = None, device: str = "cpu") -> None:
        """Initialize mesh estimator (stub).

        Args:
            smpl_model_path: Optional path to SMPL model file. Currently unused.
            device:          Device string, e.g. "cpu" or "cuda". Currently unused.
        """
        self.smpl_model_path = None if smpl_model_path is None else str(smpl_model_path)
        self.device = str(device)

        # For now, do NOT instantiate SMPL at all.
        self.model = None
        self.faces: Optional[np.ndarray] = None

    def estimate_mesh(
        self,
        joints_3d_world: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate a mesh from 3D joints.

        Args:
            joints_3d_world: (J, 3) array of joints in world/cam0 frame.
            valid_mask:      Optional (J,) boolean mask of valid joints.

        Returns:
            vertices_world: (N, 3) mesh vertices in world frame.
            faces:          (F, 3) triangle indices.
        """
        pts = np.asarray(joints_3d_world, dtype=np.float64)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("joints_3d_world must be shape (J, 3)")

        # Apply optional validity mask
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape[0] == pts.shape[0]:
                pts = pts[mask]

        # Drop NaN / non-finite joints
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]

        # Need at least 4 points to form a 3D hull
        if pts.shape[0] < 4:
            raise ValueError("Not enough valid joints to build a mesh hull (need >= 4)")

        # Build convex hull from joints (Open3D API: PointCloud.compute_convex_hull)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        hull_mesh, _ = pcd.compute_convex_hull()
        hull_mesh.compute_vertex_normals()

        verts = np.asarray(hull_mesh.vertices, dtype=np.float64)
        faces = np.asarray(hull_mesh.triangles, dtype=np.int32)
        return verts, faces

