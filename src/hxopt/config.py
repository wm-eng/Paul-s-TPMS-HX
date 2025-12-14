"""Configuration dataclasses for geometry, fluids, solver, and optimization."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

try:
    from .flow_paths import FlowPathType
except ImportError:
    # Fallback for circular import
    from enum import Enum
    class FlowPathType(Enum):
        STRAIGHT = "straight"
        U_SHAPED = "u_shaped"
        L_SHAPED = "l_shaped"


@dataclass
class GeometryConfig:
    """Heat exchanger geometry parameters."""
    length: float  # m, flow direction (x)
    width: float  # m, cross-flow direction (y)
    height: float  # m, thickness direction (z)
    n_segments: int = 50  # Discretization along flow path
    
    # 2D/Planar geometry support
    use_2d: bool = False  # If True, use 2D planar geometry
    nx: Optional[int] = None  # Grid points in x direction (for 2D)
    ny: Optional[int] = None  # Grid points in y direction (for 2D)
    
    # Flow path configuration
    hot_path_type: FlowPathType = FlowPathType.STRAIGHT
    cold_path_type: FlowPathType = FlowPathType.STRAIGHT
    hot_inlet: Optional[Tuple[float, float]] = None  # (x, y) for 2D
    hot_outlet: Optional[Tuple[float, float]] = None
    cold_inlet: Optional[Tuple[float, float]] = None
    cold_outlet: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        """Validate geometry."""
        assert self.length > 0, "Length must be positive"
        assert self.width > 0, "Width must be positive"
        assert self.height > 0, "Height must be positive"
        assert self.n_segments > 0, "n_segments must be positive"
        
        if self.use_2d:
            if self.nx is None:
                self.nx = int(np.sqrt(self.n_segments * self.length / self.width))
            if self.ny is None:
                self.ny = int(np.sqrt(self.n_segments * self.width / self.length))
            
            # Set default flow paths for U-shaped if not specified
            if self.hot_inlet is None:
                self.hot_inlet = (0.0, 0.0)
            if self.hot_outlet is None:
                if self.hot_path_type == FlowPathType.U_SHAPED:
                    self.hot_outlet = (0.0, self.width)
                else:
                    self.hot_outlet = (self.length, 0.0)
            
            if self.cold_inlet is None:
                if self.cold_path_type == FlowPathType.U_SHAPED:
                    self.cold_inlet = (self.length, self.width)
                else:
                    self.cold_inlet = (self.length, 0.0)
            if self.cold_outlet is None:
                self.cold_outlet = (0.0, 0.0)


@dataclass
class FluidConfig:
    """Fluid properties configuration."""
    # Hot side (helium)
    rho_hot: Optional[float] = None  # kg/m³ (for constant properties)
    mu_hot: Optional[float] = None  # Pa·s (for constant properties)
    cp_hot: Optional[float] = None  # J/(kg·K) (for constant properties)
    k_hot: Optional[float] = None  # W/(m·K) (for constant properties)
    
    # Cold side (LH2)
    rho_cold: Optional[float] = None  # kg/m³ (for constant properties)
    mu_cold: Optional[float] = None  # Pa·s (for constant properties)
    cp_cold: Optional[float] = None  # J/(kg·K) (for constant properties)
    k_cold: Optional[float] = None  # W/(m·K) (for constant properties)
    
    # Real-fluid properties (optional)
    use_real_properties: bool = False  # If True, use REFPROP/COOLProp
    hot_fluid_name: str = 'helium'  # Fluid name for hot side
    cold_fluid_name: str = 'hydrogen'  # Fluid name for cold side
    property_backend: Optional[str] = None  # 'REFPROP', 'COOLProp', or None (auto)
    
    # Inlet conditions
    T_hot_in: float = 300.0  # K
    T_cold_in: float = 20.0  # K
    P_hot_in: float = 2e5  # Pa
    P_cold_in: float = 1e5  # Pa
    m_dot_hot: float = 0.01  # kg/s
    m_dot_cold: float = 0.05  # kg/s
    
    def __post_init__(self):
        """Validate fluid configuration."""
        if self.use_real_properties:
            # Real properties: no need for constant values
            pass
        else:
            # Constant properties: all must be provided
            if any(v is None for v in [
                self.rho_hot, self.mu_hot, self.cp_hot, self.k_hot,
                self.rho_cold, self.mu_cold, self.cp_cold, self.k_cold
            ]):
                raise ValueError(
                    "Constant properties require all rho, mu, cp, k values "
                    "for both hot and cold sides"
                )


@dataclass
class SolverConfig:
    """Solver parameters."""
    max_iter: int = 100
    tol: float = 1e-6
    relax: float = 0.3  # Relaxation factor for iterative solver (reduced for stability)


@dataclass
class OptimizationConfig:
    """Optimization parameters."""
    max_iter: int = 50
    d_min: float = 0.1  # Minimum channel-bias value
    d_max: float = 0.9  # Maximum channel-bias value
    d_init: Optional[float] = None  # Initial d value (constant if None)
    
    # Constraints
    delta_P_max_hot: Optional[float] = None  # Pa, max pressure drop hot side
    delta_P_max_cold: Optional[float] = None  # Pa, max pressure drop cold side
    T_sat_margin: float = 5.0  # K, minimum margin below saturation for LH2
    
    # Optimization algorithm
    step_size: float = 0.01  # Initial step size for gradient ascent
    line_search_alpha: float = 0.3  # Backtracking parameter
    line_search_beta: float = 0.5  # Backtracking parameter
    smoothness_penalty: float = 0.0  # Penalty for d(x) smoothness (v1: disabled)
    
    def __post_init__(self):
        """Validate optimization config."""
        assert 0 < self.d_min < self.d_max <= 1.0, "d_min < d_max <= 1.0"
        if self.d_init is None:
            self.d_init = (self.d_min + self.d_max) / 2.0


@dataclass
class Config:
    """Main configuration container."""
    geometry: GeometryConfig
    fluid: FluidConfig
    solver: SolverConfig = field(default_factory=SolverConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Paths
    rve_table_path: str = "data/rve_tables/primitive_default.csv"
    output_dir: str = "runs"
    
    # Material properties
    metal_name: Optional[str] = None  # Metal for solid phase (e.g., 'Aluminum (6061)', 'Copper (OFHC)')

