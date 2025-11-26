"""
One-Euro filter for temporal smoothing of 3D keypoints.

The One-Euro filter is a simple, adaptive low-pass filter that provides
smooth motion while maintaining responsiveness to quick movements.
"""

import numpy as np
import math


class OneEuroFilter:
    """One-Euro filter for a single value."""
    
    def __init__(self, min_cutoff=1.0, beta=0.0, dcutoff=1.0):
        """
        Initialize One-Euro filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz) - controls responsiveness
            beta: Speed coefficient - higher values reduce lag for fast movements
            dcutoff: Derivative cutoff frequency (Hz) - controls smoothing of derivative
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    
    def __call__(self, x, t):
        """
        Filter a value.
        
        Args:
            x: Current value
            t: Current timestamp (seconds)
        
        Returns:
            Filtered value
        """
        if not np.isfinite(x):
            return x
        
        if self.x_prev is None:
            # First value - no filtering
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        
        # Compute time delta
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        
        # Compute derivative (smoothed)
        dx = (x - self.x_prev) / dt
        dx_hat = self._exponential_smoothing(dx, self.dx_prev, dt, self.dcutoff)
        
        # Adaptive cutoff based on derivative
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter the value
        x_hat = self._exponential_smoothing(x, self.x_prev, dt, cutoff)
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def _exponential_smoothing(self, x, x_prev, dt, cutoff):
        """Exponential smoothing (low-pass filter)."""
        if cutoff <= 0:
            return x
        
        alpha = 1.0 / (1.0 + 2.0 * math.pi * dt * cutoff)
        return alpha * x + (1.0 - alpha) * x_prev
    
    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class OneEuroFilter3D:
    """One-Euro filter for 3D keypoints (N keypoints, each with x, y, z)."""
    
    def __init__(self, num_keypoints, min_cutoff=1.0, beta=0.0, dcutoff=1.0):
        """
        Initialize filter for 3D keypoints.
        
        Args:
            num_keypoints: Number of keypoints to filter
            min_cutoff: Minimum cutoff frequency (Hz)
            beta: Speed coefficient
            dcutoff: Derivative cutoff frequency (Hz)
        """
        self.num_keypoints = int(num_keypoints)
        # Create a filter for each coordinate of each keypoint
        self.filters = [
            [
                OneEuroFilter(min_cutoff, beta, dcutoff)
                for _ in range(3)  # x, y, z
            ]
            for _ in range(self.num_keypoints)
        ]
    
    def __call__(self, keypoints_3d, t):
        """
        Filter 3D keypoints.
        
        Args:
            keypoints_3d: (N, 3) array of 3D keypoints, may contain NaN
            t: Current timestamp (seconds)
        
        Returns:
            (N, 3) array of filtered 3D keypoints
        """
        if keypoints_3d is None:
            return None
        
        keypoints_3d = np.asarray(keypoints_3d, dtype=np.float32)
        
        if keypoints_3d.shape[0] != self.num_keypoints:
            raise ValueError(
                f"Expected {self.num_keypoints} keypoints, got {keypoints_3d.shape[0]}"
            )
        
        filtered = np.zeros_like(keypoints_3d)
        
        for i in range(self.num_keypoints):
            for j in range(3):  # x, y, z
                val = keypoints_3d[i, j]
                if np.isfinite(val):
                    filtered[i, j] = self.filters[i][j](val, t)
                else:
                    # NaN/invalid - pass through and reset filter
                    filtered[i, j] = val
                    self.filters[i][j].reset()
        
        return filtered
    
    def reset(self):
        """Reset all filters."""
        for filters_kpt in self.filters:
            for filt in filters_kpt:
                filt.reset()

