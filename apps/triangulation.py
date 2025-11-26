"""
Stereo triangulation for 3D pose reconstruction.

Converts 2D keypoints from both cameras into 3D coordinates using
stereo calibration data.
"""

import numpy as np
import cv2


class StereoTriangulator:
    """Triangulates 2D keypoints from stereo cameras to 3D."""
    
    def __init__(self, K0, D0, K1, D1, R, T, image_size):
        """
        Initialize triangulator with stereo calibration.
        
        Args:
            K0: Camera 0 intrinsic matrix (3x3)
            D0: Camera 0 distortion coefficients
            K1: Camera 1 intrinsic matrix (3x3)
            D1: Camera 1 distortion coefficients
            R: Rotation matrix from cam0 to cam1 (3x3)
            T: Translation vector from cam0 to cam1 (3,)
            image_size: (width, height) tuple
        """
        self.K0 = np.asarray(K0, dtype=np.float64)
        self.D0 = np.asarray(D0, dtype=np.float64).reshape(-1)
        self.K1 = np.asarray(K1, dtype=np.float64)
        self.D1 = np.asarray(D1, dtype=np.float64).reshape(-1)
        self.R = np.asarray(R, dtype=np.float64)
        self.T = np.asarray(T, dtype=np.float64).reshape(3, 1)
        self.image_size = tuple(image_size)
        
        # Compute projection matrices
        # P0: cam0 projection (identity rotation/translation)
        self.P0 = self.K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # P1: cam1 projection (R, T relative to cam0)
        self.P1 = self.K1 @ np.hstack([self.R, self.T])
        
        # Pre-compute rectification maps if needed (for undistortion)
        # For now, we'll undistort on-the-fly
    
    def undistort_points(self, points_2d, camera_idx):
        """
        Undistort 2D points.
        
        Args:
            points_2d: (N, 2) array of normalized [0,1] coordinates
            camera_idx: 0 or 1
        
        Returns:
            (N, 2) array of undistorted pixel coordinates
        """
        if points_2d is None or len(points_2d) == 0:
            return None
        
        points_2d = np.asarray(points_2d, dtype=np.float32)
        
        # Convert normalized [0,1] to pixel coordinates
        w, h = self.image_size
        points_px = points_2d.copy()
        points_px[:, 0] *= w
        points_px[:, 1] *= h
        
        # Reshape for OpenCV (N, 1, 2)
        points_cv = points_px.reshape(-1, 1, 2)
        
        # Undistort
        K = self.K0 if camera_idx == 0 else self.K1
        D = self.D0 if camera_idx == 0 else self.D1
        
        points_undist = cv2.undistortPoints(
            points_cv, K, D, P=K, R=None
        )
        
        # Reshape back to (N, 2)
        return points_undist.reshape(-1, 2)
    
    def triangulate(self, points_2d_0, points_2d_1):
        """
        Triangulate 2D keypoints from both cameras to 3D.
        
        Args:
            points_2d_0: (N, 2) array of normalized [0,1] keypoints from cam0
            points_2d_1: (N, 2) array of normalized [0,1] keypoints from cam1
        
        Returns:
            points_3d: (N, 3) array of 3D coordinates in cam0 frame
                       Invalid points are set to NaN
        """
        if points_2d_0 is None or points_2d_1 is None:
            return None
        
        points_2d_0 = np.asarray(points_2d_0, dtype=np.float32)
        points_2d_1 = np.asarray(points_2d_1, dtype=np.float32)
        
        if len(points_2d_0) != len(points_2d_1):
            raise ValueError("Number of keypoints must match between cameras")
        
        if len(points_2d_0) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        # Undistort points
        pts0_undist = self.undistort_points(points_2d_0, 0)
        pts1_undist = self.undistort_points(points_2d_1, 1)
        
        if pts0_undist is None or pts1_undist is None:
            return None
        
        # Triangulate
        # cv2.triangulatePoints expects (2, N) shape
        pts0_t = pts0_undist.T  # (2, N)
        pts1_t = pts1_undist.T  # (2, N)
        
        points_4d = cv2.triangulatePoints(self.P0, self.P1, pts0_t, pts1_t)
        
        # Convert from homogeneous to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T  # (N, 3)
        
        # Check for invalid points (behind cameras, too far, etc.)
        # Mark points with negative Z or very large Z as invalid
        valid_mask = (
            (points_3d[:, 2] > 0) &  # In front of camera
            (points_3d[:, 2] < 50.0) &  # Not too far (reasonable range)
            np.isfinite(points_3d).all(axis=1)  # All finite
        )
        
        # Set invalid points to NaN
        points_3d[~valid_mask] = np.nan
        
        return points_3d.astype(np.float32)
    
    def triangulate_keypoints(self, landmarks_0, landmarks_1):
        """
        Triangulate pose landmarks from both cameras.
        
        Args:
            landmarks_0: RTMLandmarks from cam0 (or None)
            landmarks_1: RTMLandmarks from cam1 (or None)
        
        Returns:
            points_3d: (N, 3) array of 3D coordinates, or None if either input is None
        """
        if landmarks_0 is None or landmarks_1 is None:
            return None
        
        if landmarks_0.keypoints is None or landmarks_1.keypoints is None:
            return None
        
        # Extract normalized keypoints [0, 1]
        kp0 = landmarks_0.keypoints
        kp1 = landmarks_1.keypoints
        
        # Ensure same number of keypoints
        n_kpts = min(len(kp0), len(kp1))
        if n_kpts == 0:
            return None
        
        kp0 = kp0[:n_kpts]
        kp1 = kp1[:n_kpts]
        
        return self.triangulate(kp0, kp1)

