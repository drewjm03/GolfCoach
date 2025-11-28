"""
Open3D visualization for 3D skeleton and ground plane.
"""

import numpy as np

try:
    import open3d as o3d
    HAVE_OPEN3D = True
except ImportError:
    HAVE_OPEN3D = False
    o3d = None


class Viewer3D:
    """3D viewer for skeleton and ground plane visualization."""
    
    def __init__(self, ground_plane=None, connections=None, window_name="3D Pose Viewer"):
        """
        Initialize 3D viewer.
        
        Args:
            ground_plane: Dict with 'normal_cam' and 'd' (plane: n·X + d = 0)
            connections: List of (start_idx, end_idx) tuples for skeleton connections
            window_name: Window title
        """
        if not HAVE_OPEN3D:
            raise RuntimeError("Open3D is required but not installed. Install with: pip install open3d")
        
        self.ground_plane = ground_plane
        self.connections = connections or []
        self.window_name = window_name
        
        # Create visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=1280, height=720)
        
        # Geometry objects
        self.skeleton_lineset = None
        self.skeleton_points = None
        self.ground_plane_mesh = None
        self.axes = None
        self.mesh = None  # optional SMPL mesh
        
        # Initialize scene
        self._setup_scene()
    
    def _setup_scene(self):
        """Set up the 3D scene with ground plane and axes."""
        # Add coordinate axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(axes)
        self.axes = axes
        
        # Add ground plane if provided
        if self.ground_plane is not None:
            self._add_ground_plane()
        
        # Set up camera
        ctr = self.vis.get_view_control()
        ctr.set_front([0.0, -0.5, -1.0])
        ctr.set_lookat([0.0, 0.0, 2.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(0.7)
    
    def _add_ground_plane(self):
        """Add ground plane mesh to the scene."""
        if self.ground_plane is None:
            return
        
        normal = np.asarray(self.ground_plane['normal_cam'], dtype=np.float64)
        d = float(self.ground_plane['d'])
        
        # Normalize normal
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 0:
            normal = normal / normal_norm
        
        # Create a large plane mesh
        # Find a point on the plane (closest to origin)
        if abs(normal[2]) > 1e-6:
            # Use z=0 as a starting point
            z0 = -d / normal[2] if abs(normal[2]) > 1e-6 else 0.0
            point_on_plane = np.array([0.0, 0.0, z0])
        else:
            point_on_plane = np.array([0.0, -d / normal[1] if abs(normal[1]) > 1e-6 else 0.0, 0.0])
        
        # Create two orthogonal vectors in the plane
        if abs(normal[0]) < 0.9:
            v1 = np.cross(normal, [1, 0, 0])
        else:
            v1 = np.cross(normal, [0, 1, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Create a 10m x 10m plane
        size = 10.0
        vertices = [
            point_on_plane + size * (-v1 - v2),
            point_on_plane + size * (v1 - v2),
            point_on_plane + size * (v1 + v2),
            point_on_plane + size * (-v1 + v2),
        ]
        triangles = [[0, 1, 2], [0, 2, 3]]
        
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Add triangles for the back side (reverse winding order)
        triangles_back = [[0, 2, 1], [0, 3, 2]]
        all_triangles = triangles + triangles_back
        plane_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        
        plane_mesh.compute_vertex_normals()
        plane_mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
        
        # Make the mesh double-sided by ensuring normals are computed correctly
        # and the material renders both sides
        plane_mesh.compute_triangle_normals()
        plane_mesh.compute_vertex_normals()
        
        self.vis.add_geometry(plane_mesh)
        self.ground_plane_mesh = plane_mesh
    
    def update_skeleton(self, keypoints_3d):
        """
        Update skeleton visualization with new 3D keypoints.
        
        Args:
            keypoints_3d: (N, 3) array of 3D keypoints, may contain NaN
        """
        if keypoints_3d is None:
            return
        
        keypoints_3d = np.asarray(keypoints_3d, dtype=np.float32)
        
        # Remove old skeleton if it exists
        if self.skeleton_lineset is not None:
            self.vis.remove_geometry(self.skeleton_lineset, reset_bounding_box=False)
        if self.skeleton_points is not None:
            self.vis.remove_geometry(self.skeleton_points, reset_bounding_box=False)
        
        # Filter out NaN points
        valid_mask = np.isfinite(keypoints_3d).all(axis=1)
        valid_kpts = keypoints_3d[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_kpts) == 0:
            self.skeleton_lineset = None
            self.skeleton_points = None
            return
        
        # Create point cloud for keypoints
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(valid_kpts)
        points.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        
        # Create line set for connections
        lines = []
        line_colors = []
        
        # Create mapping from original index to valid index
        idx_map = {orig: new for new, orig in enumerate(valid_indices)}
        
        for start_idx, end_idx in self.connections:
            if start_idx in idx_map and end_idx in idx_map:
                lines.append([idx_map[start_idx], idx_map[end_idx]])
                line_colors.append([0.0, 1.0, 0.0])  # Green
        
        if len(lines) > 0:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(valid_kpts)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            self.skeleton_lineset = line_set
            self.vis.add_geometry(line_set, reset_bounding_box=False)
        
        self.skeleton_points = points
        self.vis.add_geometry(points, reset_bounding_box=False)
    
    def update_mesh(self, vertices_world, faces):
        """Update/add SMPL mesh in world coordinates.

        Args:
            vertices_world: (N, 3) numpy array in same frame as 3D skeleton
            faces:          (F, 3) int triangle indices
        """
        if vertices_world is None or faces is None:
            return

        vertices_world = np.asarray(vertices_world, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)

        # Remove old mesh if it exists
        if self.mesh is not None:
            self.vis.remove_geometry(self.mesh, reset_bounding_box=False)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.9])  # light bluish

        self.mesh = mesh
        self.vis.add_geometry(mesh, reset_bounding_box=False)
    
    def update_mesh(self, vertices_world, faces):
        """Update/add SMPL mesh in world coordinates.

        Args:
            vertices_world: (N, 3) numpy array in same frame as 3D skeleton
            faces:          (F, 3) int triangle indices
        """
        if vertices_world is None or faces is None:
            return

        vertices_world = np.asarray(vertices_world, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)

        # Remove old mesh if it exists
        if self.mesh is not None:
            self.vis.remove_geometry(self.mesh, reset_bounding_box=False)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.9])  # light bluish

        self.mesh = mesh
        self.vis.add_geometry(mesh, reset_bounding_box=False)
    
    def update(self):
        """Update the visualization (call in main loop)."""
        if not self.vis.poll_events():
            return False
        self.vis.update_renderer()
        return True
    
    def close(self):
        """Close the visualization window."""
        self.vis.destroy_window()
    
    def is_closed(self):
        """Check if window is closed."""
        return not self.vis.poll_events()

