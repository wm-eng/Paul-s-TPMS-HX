"""1D/2D counterflow macromodel solver (porous-equivalent)."""

import numpy as np
from typing import NamedTuple, Optional
from .config import Config
from .materials import ConstantProperties
from .rve_db import RVEDatabase
from .flow_paths import FlowPath, FlowPathType


class MacroModelResult(NamedTuple):
    """Results from macromodel solve."""
    Q: float  # W, total heat transfer
    delta_P_hot: float  # Pa, pressure drop hot side
    delta_P_cold: float  # Pa, pressure drop cold side
    h_hot_out: float  # J/kg, outlet enthalpy hot side
    h_cold_out: float  # J/kg, outlet enthalpy cold side
    x: np.ndarray  # m, position along flow
    T_hot: np.ndarray  # K, hot side temperature profile
    T_cold: np.ndarray  # K, cold side temperature profile
    T_solid: np.ndarray  # K, solid temperature profile
    P_hot: np.ndarray  # Pa, hot side pressure profile
    P_cold: np.ndarray  # Pa, cold side pressure profile


class MacroModel:
    """
    1D/2D steady counterflow porous-equivalent model.
    
    Supports:
    - 1D straight counterflow
    - 2D planar geometry with U-shaped/L-shaped flow paths
    
    Solves:
    - Energy balance: enthalpy-based with volumetric heat transfer
    - Momentum balance: Darcy-Forchheimer pressure drop
    - Three-temperature model: T_hot, T_cold, T_solid
    """
    
    def __init__(self, config: Config, rve_db: RVEDatabase):
        """
        Initialize macromodel.
        
        Parameters
        ----------
        config : Config
            Configuration object
        rve_db : RVEDatabase
            RVE property database
        """
        self.config = config
        self.rve_db = rve_db
        self.use_2d = config.geometry.use_2d
        
        # Create material property objects
        if config.fluid.use_real_properties:
            from .materials import RealFluidProperties
            backend = config.fluid.property_backend or 'auto'
            self.props_hot = RealFluidProperties(
                fluid_name=config.fluid.hot_fluid_name,
                backend=backend,
            )
            self.props_cold = RealFluidProperties(
                fluid_name=config.fluid.cold_fluid_name,
                backend=backend,
            )
        else:
            # Use constant properties
            self.props_hot = ConstantProperties(
                rho=config.fluid.rho_hot,
                mu=config.fluid.mu_hot,
                cp=config.fluid.cp_hot,
                k=config.fluid.k_hot,
            )
            self.props_cold = ConstantProperties(
                rho=config.fluid.rho_cold,
                mu=config.fluid.mu_cold,
                cp=config.fluid.cp_cold,
                k=config.fluid.k_cold,
                T_sat_ref=20.0,  # K, approximate LH2 saturation at 1 bar
                dT_sat_dP=1e-5,  # K/Pa, rough estimate
            )
        
        # Initialize flow paths
        if self.use_2d:
            # 2D planar geometry with flow paths
            self.hot_path = FlowPath(
                path_type=config.geometry.hot_path_type,
                inlet=config.geometry.hot_inlet,
                outlet=config.geometry.hot_outlet,
                n_segments=config.geometry.n_segments,
            )
            self.cold_path = FlowPath(
                path_type=config.geometry.cold_path_type,
                inlet=config.geometry.cold_inlet,
                outlet=config.geometry.cold_outlet,
                n_segments=config.geometry.n_segments,
            )
            self.n = config.geometry.n_segments
            # Use path coordinates for 2D
            self.x = np.array([0.0])  # Placeholder, will use path coordinates
        else:
            # 1D straight counterflow
            self.n = config.geometry.n_segments
            self.dx = config.geometry.length / self.n
            self.x = np.linspace(0, config.geometry.length, self.n + 1)
            self.hot_path = None
            self.cold_path = None
        
        # Cross-sectional areas (assume rectangular channels)
        self.A_hot = config.geometry.width * config.geometry.height * 0.5  # m²
        self.A_cold = config.geometry.width * config.geometry.height * 0.5  # m²
    
    def solve(self, d_field: np.ndarray) -> MacroModelResult:
        """
        Solve macromodel for given channel-bias field d(x) or d(x,y).
        
        Parameters
        ----------
        d_field : np.ndarray
            Channel-bias field values:
            - 1D: array of length n (cell centers along flow)
            - 2D: array of shape (n,) mapped to flow paths
            
        Returns
        -------
        result : MacroModelResult
            Solution including profiles and integrated quantities
        """
        if len(d_field) != self.n:
            raise ValueError(f"d_field length {len(d_field)} != n_segments {self.n}")
        
        # Clamp d to valid range
        d_field = np.clip(d_field, self.config.optimization.d_min, 
                         self.config.optimization.d_max)
        
        # Branch to 1D or 2D solver
        if self.use_2d:
            return self._solve_2d(d_field)
        else:
            return self._solve_1d(d_field)
    
    def _solve_1d(self, d_field: np.ndarray) -> MacroModelResult:
        """
        Solve 1D counterflow model.
        
        Parameters
        ----------
        d_field : np.ndarray
            Channel-bias field values at cell centers (length n)
            
        Returns
        -------
        result : MacroModelResult
            Solution including profiles and integrated quantities
        """
        # Get RVE properties
        kappa_hot = self.rve_db.kappa_hot(d_field)
        beta_hot = self.rve_db.beta_hot(d_field)
        eps_hot = self.rve_db.eps_hot(d_field)
        lambda_solid = self.rve_db.lambda_solid(d_field)
        
        # Use same properties for cold side (v1 assumption)
        kappa_cold = kappa_hot
        beta_cold = beta_hot
        eps_cold = eps_hot
        
        # Initialize temperature and pressure profiles
        T_hot = np.full(self.n + 1, self.config.fluid.T_hot_in)
        T_cold = np.full(self.n + 1, self.config.fluid.T_cold_in)
        T_solid = (T_hot + T_cold) / 2.0
        P_hot = np.full(self.n + 1, self.config.fluid.P_hot_in)
        P_cold = np.full(self.n + 1, self.config.fluid.P_cold_in)
        
        # Initialize enthalpies (h = cp * T for constant properties)
        h_hot = self.props_hot.specific_heat(T_hot[0], P_hot[0]) * T_hot
        h_cold = self.props_cold.specific_heat(T_cold[0], P_cold[0]) * T_cold
        
        # Iterative solve for energy balance
        for it in range(self.config.solver.max_iter):
            T_hot_old = T_hot.copy()
            T_cold_old = T_cold.copy()
            T_solid_old = T_solid.copy()
            
            # Compute velocities from mass flow rates
            # u = m_dot / (rho * A * eps)
            u_hot = self.config.fluid.m_dot_hot / (
                self.props_hot.density(T_hot[:-1], P_hot[:-1]) * 
                self.A_hot * eps_hot
            )
            u_cold = self.config.fluid.m_dot_cold / (
                self.props_cold.density(T_cold[:-1], P_cold[:-1]) * 
                self.A_cold * eps_cold
            )
            
            # Heat transfer coefficients
            h_htc_hot = self.rve_db.h_hot(u_hot, d_field)
            h_htc_cold = self.rve_db.h_cold(u_cold, d_field)
            
            # Volumetric heat transfer: Q_vol = h * A_surf/V * (T_solid - T_fluid)
            # A_surf/V is the volumetric heat transfer coefficient parameter
            # from RVE properties (function of d(x))
            A_surf_V = self.rve_db.A_surf_V(d_field)  # 1/m, surface area per unit volume
            Q_vol_hot = h_htc_hot * A_surf_V * (T_solid[:-1] - T_hot[:-1])
            Q_vol_cold = h_htc_cold * A_surf_V * (T_solid[:-1] - T_cold[:-1])
            
            # Energy balance: d(m_dot * h)/dx = Q_vol * A_cross
            # For constant m_dot: m_dot * dh/dx = Q_vol * A_cross
            cp_hot = self.props_hot.specific_heat(T_hot[:-1], P_hot[:-1])
            cp_cold = self.props_cold.specific_heat(T_cold[:-1], P_cold[:-1])
            
            # Update enthalpies (upwind differencing)
            # Hot side flows +x, cold side flows -x (counterflow)
            for i in range(1, self.n + 1):
                # Hot side: forward difference
                dh_dx_hot = Q_vol_hot[i-1] * self.A_hot / self.config.fluid.m_dot_hot
                h_hot[i] = h_hot[i-1] + dh_dx_hot * self.dx
                T_hot[i] = h_hot[i] / cp_hot[i-1] if i < self.n else h_hot[i] / cp_hot[-1]
            
            # Cold side: backward difference (flows opposite, inlet at x=L)
            # Set inlet condition at index -1
            h_cold[-1] = self.props_cold.specific_heat(
                self.config.fluid.T_cold_in, P_cold[-1]
            ) * self.config.fluid.T_cold_in
            T_cold[-1] = self.config.fluid.T_cold_in
            
            for j in range(self.n - 1, -1, -1):
                dh_dx_cold = Q_vol_cold[j] * self.A_cold / self.config.fluid.m_dot_cold
                h_cold[j] = h_cold[j+1] - dh_dx_cold * self.dx
                T_cold[j] = h_cold[j] / cp_cold[j]
            
            # Solid energy balance: conduction + heat transfer
            # Simplified: T_solid balances heat from both sides
            # k_solid * d²T/dx² + Q_vol_hot + Q_vol_cold = 0
            # For v1, use simple averaging with relaxation
            T_solid_new = (h_htc_hot * T_hot[:-1] + h_htc_cold * T_cold[:-1]) / (
                h_htc_hot + h_htc_cold + 1e-10
            )
            T_solid[:-1] = (1 - self.config.solver.relax) * T_solid[:-1] + \
                          self.config.solver.relax * T_solid_new
            T_solid[-1] = T_solid[-2]  # Extrapolate
            
            # Pressure drop: Darcy-Forchheimer
            # dP/dx = -(mu/kappa) * u - beta * rho * u * |u|
            rho_hot = self.props_hot.density(T_hot[:-1], P_hot[:-1])
            rho_cold = self.props_cold.density(T_cold[:-1], P_cold[:-1])
            mu_hot = self.props_hot.viscosity(T_hot[:-1], P_hot[:-1])
            mu_cold = self.props_cold.viscosity(T_cold[:-1], P_cold[:-1])
            
            dP_dx_hot = -(mu_hot / kappa_hot) * u_hot - beta_hot * rho_hot * u_hot * np.abs(u_hot)
            dP_dx_cold = -(mu_cold / kappa_cold) * u_cold - beta_cold * rho_cold * u_cold * np.abs(u_cold)
            
            # Integrate pressure
            for i in range(1, self.n + 1):
                P_hot[i] = P_hot[i-1] + dP_dx_hot[i-1] * self.dx
            
            # Cold side: integrate from outlet (x=0) to inlet (x=L)
            for j in range(self.n - 1, -1, -1):
                P_cold[j] = P_cold[j+1] - dP_dx_cold[j] * self.dx
            
            # Check convergence
            err = max(
                np.max(np.abs(T_hot - T_hot_old)),
                np.max(np.abs(T_cold - T_cold_old)),
                np.max(np.abs(T_solid - T_solid_old))
            )
            if err < self.config.solver.tol:
                break
        
        # Compute total heat transfer
        # Hot side: Q = m_dot * (h_in - h_out) = m_dot * (h[0] - h[-1])
        Q = self.config.fluid.m_dot_hot * (h_hot[0] - h_hot[-1])
        
        # Pressure drops
        delta_P_hot = P_hot[0] - P_hot[-1]
        # Cold side: inlet at x=L (index -1), outlet at x=0 (index 0)
        delta_P_cold = P_cold[-1] - P_cold[0]
        
        return MacroModelResult(
            Q=Q,
            delta_P_hot=delta_P_hot,
            delta_P_cold=delta_P_cold,
            h_hot_out=h_hot[-1],
            h_cold_out=h_cold[0],  # Cold outlet at x=0
            x=self.x,
            T_hot=T_hot,
            T_cold=T_cold,
            T_solid=T_solid,
            P_hot=P_hot,
            P_cold=P_cold,
        )
    
    def _solve_2d(self, d_field: np.ndarray) -> MacroModelResult:
        """
        Solve 2D planar model with flow paths (U-shaped, L-shaped, etc.).
        
        Parameters
        ----------
        d_field : np.ndarray
            Channel-bias field values along flow paths (length n)
            
        Returns
        -------
        result : MacroModelResult
            Solution including profiles and integrated quantities
        """
        # Get RVE properties
        kappa_hot = self.rve_db.kappa_hot(d_field)
        beta_hot = self.rve_db.beta_hot(d_field)
        eps_hot = self.rve_db.eps_hot(d_field)
        lambda_solid = self.rve_db.lambda_solid(d_field)
        
        # Use same properties for cold side (v1 assumption)
        kappa_cold = kappa_hot
        beta_cold = beta_hot
        eps_cold = eps_hot
        
        # Get path segment lengths
        ds_hot = self.hot_path.segment_lengths
        ds_cold = self.cold_path.segment_lengths
        
        # Initialize temperature and pressure profiles along paths
        T_hot = np.full(self.n + 1, self.config.fluid.T_hot_in)
        T_cold = np.full(self.n + 1, self.config.fluid.T_cold_in)
        T_solid = (T_hot + T_cold) / 2.0
        P_hot = np.full(self.n + 1, self.config.fluid.P_hot_in)
        P_cold = np.full(self.n + 1, self.config.fluid.P_cold_in)
        
        # Initialize enthalpies
        h_hot = self.props_hot.specific_heat(T_hot[0], P_hot[0]) * T_hot
        h_cold = self.props_cold.specific_heat(T_cold[0], P_cold[0]) * T_cold
        
        # Cumulative path distance for x coordinate
        s_hot = np.concatenate([[0], np.cumsum(ds_hot)])
        s_cold = np.concatenate([[0], np.cumsum(ds_cold)])
        
        # Iterative solve for energy balance
        for it in range(self.config.solver.max_iter):
            T_hot_old = T_hot.copy()
            T_cold_old = T_cold.copy()
            T_solid_old = T_solid.copy()
            
            # Compute velocities from mass flow rates
            u_hot = self.config.fluid.m_dot_hot / (
                self.props_hot.density(T_hot[:-1], P_hot[:-1]) * 
                self.A_hot * eps_hot
            )
            u_cold = self.config.fluid.m_dot_cold / (
                self.props_cold.density(T_cold[:-1], P_cold[:-1]) * 
                self.A_cold * eps_cold
            )
            
            # Heat transfer coefficients
            h_htc_hot = self.rve_db.h_hot(u_hot, d_field)
            h_htc_cold = self.rve_db.h_cold(u_cold, d_field)
            
            # Volumetric heat transfer
            A_surf_V = self.rve_db.A_surf_V(d_field)
            Q_vol_hot = h_htc_hot * A_surf_V * (T_solid[:-1] - T_hot[:-1])
            Q_vol_cold = h_htc_cold * A_surf_V * (T_solid[:-1] - T_cold[:-1])
            
            # Energy balance along flow paths
            cp_hot = self.props_hot.specific_heat(T_hot[:-1], P_hot[:-1])
            cp_cold = self.props_cold.specific_heat(T_cold[:-1], P_cold[:-1])
            
            # Update enthalpies along hot path (forward)
            for i in range(1, self.n + 1):
                dh_ds_hot = Q_vol_hot[i-1] * self.A_hot / self.config.fluid.m_dot_hot
                h_hot[i] = h_hot[i-1] + dh_ds_hot * ds_hot[i-1]
                T_hot[i] = h_hot[i] / cp_hot[i-1] if i < self.n else h_hot[i] / cp_hot[-1]
            
            # Update enthalpies along cold path (forward along its path)
            # Set inlet condition
            h_cold[0] = self.props_cold.specific_heat(
                self.config.fluid.T_cold_in, P_cold[0]
            ) * self.config.fluid.T_cold_in
            T_cold[0] = self.config.fluid.T_cold_in
            
            for j in range(1, self.n + 1):
                dh_ds_cold = Q_vol_cold[j-1] * self.A_cold / self.config.fluid.m_dot_cold
                h_cold[j] = h_cold[j-1] + dh_ds_cold * ds_cold[j-1]
                T_cold[j] = h_cold[j] / cp_cold[j-1] if j < self.n else h_cold[j] / cp_cold[-1]
            
            # Solid energy balance (simplified)
            T_solid_new = (h_htc_hot * T_hot[:-1] + h_htc_cold * T_cold[:-1]) / (
                h_htc_hot + h_htc_cold + 1e-10
            )
            T_solid[:-1] = (1 - self.config.solver.relax) * T_solid[:-1] + \
                          self.config.solver.relax * T_solid_new
            T_solid[-1] = T_solid[-2]
            
            # Pressure drop: Darcy-Forchheimer along paths
            rho_hot = self.props_hot.density(T_hot[:-1], P_hot[:-1])
            rho_cold = self.props_cold.density(T_cold[:-1], P_cold[:-1])
            mu_hot = self.props_hot.viscosity(T_hot[:-1], P_hot[:-1])
            mu_cold = self.props_cold.viscosity(T_cold[:-1], P_cold[:-1])
            
            dP_ds_hot = -(mu_hot / kappa_hot) * u_hot - beta_hot * rho_hot * u_hot * np.abs(u_hot)
            dP_ds_cold = -(mu_cold / kappa_cold) * u_cold - beta_cold * rho_cold * u_cold * np.abs(u_cold)
            
            # Integrate pressure along paths
            for i in range(1, self.n + 1):
                P_hot[i] = P_hot[i-1] + dP_ds_hot[i-1] * ds_hot[i-1]
            
            for j in range(1, self.n + 1):
                P_cold[j] = P_cold[j-1] + dP_ds_cold[j-1] * ds_cold[j-1]
            
            # Check convergence
            err = max(
                np.max(np.abs(T_hot - T_hot_old)),
                np.max(np.abs(T_cold - T_cold_old)),
                np.max(np.abs(T_solid - T_solid_old))
            )
            if err < self.config.solver.tol:
                break
        
        # Compute total heat transfer
        Q = self.config.fluid.m_dot_hot * (h_hot[0] - h_hot[-1])
        
        # Pressure drops
        delta_P_hot = P_hot[0] - P_hot[-1]
        delta_P_cold = P_cold[0] - P_cold[-1]
        
        # Use hot path distance for x coordinate
        x_coords = s_hot
        
        return MacroModelResult(
            Q=Q,
            delta_P_hot=delta_P_hot,
            delta_P_cold=delta_P_cold,
            h_hot_out=h_hot[-1],
            h_cold_out=h_cold[-1],
            x=x_coords,
            T_hot=T_hot,
            T_cold=T_cold,
            T_solid=T_solid,
            P_hot=P_hot,
            P_cold=P_cold,
        )

