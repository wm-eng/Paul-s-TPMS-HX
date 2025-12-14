"""Flow path definitions for heat exchanger geometry."""

from enum import Enum
from typing import Tuple, List
import numpy as np


class FlowPathType(Enum):
    """Types of flow paths."""
    STRAIGHT = "straight"  # 1D straight flow
    U_SHAPED = "u_shaped"  # U-shaped path (planar)
    L_SHAPED = "l_shaped"  # L-shaped path (planar)


class FlowPath:
    """Defines a flow path through the heat exchanger."""
    
    def __init__(
        self,
        path_type: FlowPathType,
        inlet: Tuple[float, float],
        outlet: Tuple[float, float],
        n_segments: int,
    ):
        """
        Initialize flow path.
        
        Parameters
        ----------
        path_type : FlowPathType
            Type of flow path
        inlet : Tuple[float, float]
            Inlet position (x, y) in meters
        outlet : Tuple[float, float]
            Outlet position (x, y) in meters
        n_segments : int
            Number of segments along the path
        """
        self.path_type = path_type
        self.inlet = inlet
        self.outlet = outlet
        self.n_segments = n_segments
        self._path_coords = None
        self._path_lengths = None
        self._generate_path()
    
    def _generate_path(self):
        """Generate path coordinates and segment lengths."""
        if self.path_type == FlowPathType.STRAIGHT:
            # Straight line from inlet to outlet
            x_coords = np.linspace(self.inlet[0], self.outlet[0], self.n_segments + 1)
            y_coords = np.linspace(self.inlet[1], self.outlet[1], self.n_segments + 1)
            self._path_coords = np.column_stack([x_coords, y_coords])
            
        elif self.path_type == FlowPathType.U_SHAPED:
            # U-shaped path: go right, then down, then left
            # Inlet at (x_in, y_in), outlet at (x_out, y_out)
            x_in, y_in = self.inlet
            x_out, y_out = self.outlet
            
            # For U-shape: typically inlet and outlet are on same x, different y
            # Path: horizontal right, vertical, horizontal left
            # Find the turning point (rightmost x)
            x_max = max(x_in, x_out) + abs(y_out - y_in) * 0.5  # Extend right by half height
            
            # Split into 3 segments
            n1 = self.n_segments // 3
            n2 = self.n_segments // 3
            n3 = self.n_segments - n1 - n2
            
            # Segment 1: horizontal right (inlet to rightmost)
            x1 = np.linspace(x_in, x_max, n1 + 1)
            y1 = np.full(n1 + 1, y_in)
            
            # Segment 2: vertical (rightmost top to rightmost bottom)
            x2 = np.full(n2 + 1, x_max)
            y2 = np.linspace(y_in, y_out, n2 + 1)
            
            # Segment 3: horizontal left (rightmost to outlet)
            x3 = np.linspace(x_max, x_out, n3 + 1)
            y3 = np.full(n3 + 1, y_out)
            
            # Combine (avoid duplicate points)
            x_coords = np.concatenate([x1[:-1], x2[:-1], x3])
            y_coords = np.concatenate([y1[:-1], y2[:-1], y3])
            
            self._path_coords = np.column_stack([x_coords, y_coords])
            
        elif self.path_type == FlowPathType.L_SHAPED:
            # L-shaped path: go right, then down
            x_in, y_in = self.inlet
            x_out, y_out = self.outlet
            
            # Split at corner
            n1 = self.n_segments // 2
            n2 = self.n_segments - n1
            
            # Segment 1: horizontal
            x1 = np.linspace(x_in, x_out, n1 + 1)
            y1 = np.full(n1 + 1, y_in)
            
            # Segment 2: vertical
            x2 = np.full(n2 + 1, x_out)
            y2 = np.linspace(y_in, y_out, n2 + 1)
            
            x_coords = np.concatenate([x1[:-1], x2])
            y_coords = np.concatenate([y1[:-1], y2])
            
            self._path_coords = np.column_stack([x_coords, y_coords])
        
        # Compute segment lengths
        dx = np.diff(self._path_coords[:, 0])
        dy = np.diff(self._path_coords[:, 1])
        self._path_lengths = np.sqrt(dx**2 + dy**2)
    
    @property
    def coordinates(self) -> np.ndarray:
        """Get path coordinates (n_points, 2) array of (x, y)."""
        return self._path_coords
    
    @property
    def segment_lengths(self) -> np.ndarray:
        """Get segment lengths (n_segments,) array."""
        return self._path_lengths
    
    @property
    def total_length(self) -> float:
        """Total path length."""
        return np.sum(self._path_lengths)
    
    def get_direction(self, segment_idx: int) -> Tuple[float, float]:
        """
        Get flow direction vector for a segment.
        
        Parameters
        ----------
        segment_idx : int
            Segment index (0 to n_segments-1)
            
        Returns
        -------
        direction : Tuple[float, float]
            Normalized direction vector (dx, dy)
        """
        if segment_idx >= len(self._path_lengths):
            segment_idx = len(self._path_lengths) - 1
        
        dx = self._path_coords[segment_idx + 1, 0] - self._path_coords[segment_idx, 0]
        dy = self._path_coords[segment_idx + 1, 1] - self._path_coords[segment_idx, 1]
        length = self._path_lengths[segment_idx]
        
        if length > 1e-10:
            return (dx / length, dy / length)
        return (0.0, 0.0)

