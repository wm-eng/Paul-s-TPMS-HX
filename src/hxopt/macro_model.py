"""1D/2D counterflow macromodel solver (porous-equivalent)."""

import numpy as np
from typing import NamedTuple, Optional, Tuple
from .config import Config
from .materials import ConstantProperties
from .rve_db import RVEDatabase
from .flow_paths import FlowPath, FlowPathType


def _get_fluid_limits(fluid_name: Optional[str]) -> Tuple[float, float, float, float]:
    """
    Get temperature and pressure limits for a fluid.
    
    Parameters
    ----------
    fluid_name : str or None
        Fluid name (e.g., 'helium', 'hydrogen')
        
    Returns
    -------
    (T_min, T_max, P_min, P_max) : tuple
        Temperature limits in K, pressure limits in Pa
    """
    if fluid_name is None:
        # Default conservative limits
        return 2.0, 1000.0, 1e3, 1e8
    
    fluid_lower = fluid_name.lower()
    
    # Fluid-specific limits based on REFPROP/CoolProp valid ranges
    if fluid_lower in ['helium', 'he']:
        # Helium: triple point at 2.1768 K, valid up to ~1000 K
        return 2.2, 1000.0, 1e3, 1e8  # T_min slightly above triple point
    elif fluid_lower in ['hydrogen', 'h2', 'lh2']:
        # Hydrogen: triple point at 13.957 K, valid up to 1000 K (REFPROP max)
        return 14.0, 1000.0, 1e3, 1e8  # T_min slightly above triple point
    else:
        # Unknown fluid: use conservative defaults
        return 2.0, 1000.0, 1e3, 1e8


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
        # Store cell size for reference
        self.cell_size = getattr(rve_db, 'cell_size', 5e-3)  # Default 5mm
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
            # Get fluid-specific limits for validation
            self.hot_T_min, self.hot_T_max, self.hot_P_min, self.hot_P_max = _get_fluid_limits(
                config.fluid.hot_fluid_name
            )
            self.cold_T_min, self.cold_T_max, self.cold_P_min, self.cold_P_max = _get_fluid_limits(
                config.fluid.cold_fluid_name
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
            # For constant properties, use conservative limits
            # Pressure limits: allow outlet to go much lower to capture property effects
            # Previous P_min=1e3 (1 kPa) was too high and clamped all outlet pressures identically
            self.hot_T_min, self.hot_T_max, self.hot_P_min, self.hot_P_max = 2.0, 1000.0, 1e2, 1e8  # P_min: 1e3→1e2 (1→0.1 kPa)
            self.cold_T_min, self.cold_T_max, self.cold_P_min, self.cold_P_max = 2.0, 1000.0, 1e2, 1e8  # P_min: 1e3→1e2 (1→0.1 kPa)
        
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
        
        # Initialize temperature and pressure profiles with better initial guess
        # Linear interpolation between inlets for better initial condition
        # Ensure initial temperatures are within valid ranges
        T_hot_out_guess = max(
            self.config.fluid.T_hot_in - 10.0,
            self.hot_T_min
        )
        T_cold_out_guess = min(
            self.config.fluid.T_cold_in + 10.0,
            self.cold_T_max
        )
        
        T_hot = np.linspace(
            self.config.fluid.T_hot_in,
            T_hot_out_guess,
            self.n + 1
        )
        T_cold = np.linspace(
            T_cold_out_guess,
            self.config.fluid.T_cold_in,
            self.n + 1
        )
        T_solid = (T_hot + T_cold) / 2.0
        P_hot = np.full(self.n + 1, self.config.fluid.P_hot_in)
        P_cold = np.full(self.n + 1, self.config.fluid.P_cold_in)
        
        # Ensure initial values are within valid ranges
        T_hot = np.clip(T_hot, self.hot_T_min, self.hot_T_max)
        T_cold = np.clip(T_cold, self.cold_T_min, self.cold_T_max)
        P_hot = np.clip(P_hot, self.hot_P_min, self.hot_P_max)
        P_cold = np.clip(P_cold, self.cold_P_min, self.cold_P_max)
        
        # Initialize enthalpies using reference temperature
        # Use inlet conditions for reference enthalpy
        T_ref_hot = self.config.fluid.T_hot_in
        T_ref_cold = self.config.fluid.T_cold_in
        P_ref_hot = self.config.fluid.P_hot_in
        P_ref_cold = self.config.fluid.P_cold_in
        
        cp_ref_hot = self.props_hot.specific_heat(T_ref_hot, P_ref_hot)
        cp_ref_cold = self.props_cold.specific_heat(T_ref_cold, P_ref_cold)
        
        # Initialize enthalpies: h = cp_ref * T (for constant properties)
        # For variable properties, we'll update this iteratively
        h_hot = cp_ref_hot * T_hot
        h_cold = cp_ref_cold * T_cold
        
        # Physical bounds for temperatures
        # Hot side should cool down: T_hot_in >= T_hot >= T_cold_in
        # Cold side should heat up: T_cold_in <= T_cold <= T_hot_in
        T_min_global = min(self.config.fluid.T_cold_in, self.config.fluid.T_hot_in) - 10.0
        T_max_global = max(self.config.fluid.T_cold_in, self.config.fluid.T_hot_in) + 10.0
        T_hot_min = self.config.fluid.T_cold_in  # Hot can't cool below cold inlet
        T_hot_max = self.config.fluid.T_hot_in  # Hot can't exceed its inlet
        T_cold_min = self.config.fluid.T_cold_in  # Cold can't cool below its inlet
        T_cold_max = self.config.fluid.T_hot_in  # Cold can't exceed hot inlet
        
        # Iterative solve for energy balance with improved stability
        for it in range(self.config.solver.max_iter):
            T_hot_old = T_hot.copy()
            T_cold_old = T_cold.copy()
            T_solid_old = T_solid.copy()
            
            # Compute densities and properties at current temperatures
            # Ensure valid inputs (no NaN, no negative pressure) before property lookup
            # Use fluid-specific limits
            # IMPORTANT: Use all points including boundaries for property evaluation
            # We'll evaluate properties at segment midpoints (for Q_vol calculation)
            # but also need properties at boundaries for proper enthalpy calculation
            T_hot_safe = np.clip(T_hot[:-1], self.hot_T_min, self.hot_T_max)
            T_cold_safe = np.clip(T_cold[:-1], self.cold_T_min, self.cold_T_max)
            P_hot_safe = np.clip(P_hot[:-1], self.hot_P_min, self.hot_P_max)
            P_cold_safe = np.clip(P_cold[:-1], self.cold_P_min, self.cold_P_max)
            
            # Ensure boundary temperatures are also valid (for property lookups at boundaries)
            T_hot[0] = np.clip(T_hot[0], self.hot_T_min, self.hot_T_max)
            T_hot[-1] = np.clip(T_hot[-1], self.hot_T_min, self.hot_T_max)
            T_cold[0] = np.clip(T_cold[0], self.cold_T_min, self.cold_T_max)
            T_cold[-1] = np.clip(T_cold[-1], self.cold_T_min, self.cold_T_max)
            
            # Replace NaN/inf with safe values (use inlet conditions as defaults)
            T_hot_safe = np.nan_to_num(
                T_hot_safe, 
                nan=self.config.fluid.T_hot_in, 
                posinf=self.config.fluid.T_hot_in, 
                neginf=self.config.fluid.T_hot_in
            )
            T_cold_safe = np.nan_to_num(
                T_cold_safe, 
                nan=self.config.fluid.T_cold_in, 
                posinf=self.config.fluid.T_cold_in, 
                neginf=self.config.fluid.T_cold_in
            )
            P_hot_safe = np.nan_to_num(
                P_hot_safe, 
                nan=self.config.fluid.P_hot_in, 
                posinf=self.config.fluid.P_hot_in, 
                neginf=self.config.fluid.P_hot_in
            )
            P_cold_safe = np.nan_to_num(
                P_cold_safe, 
                nan=self.config.fluid.P_cold_in, 
                posinf=self.config.fluid.P_cold_in, 
                neginf=self.config.fluid.P_cold_in
            )
            
            # Final clamp after NaN replacement (in case NaN was replaced with out-of-range value)
            T_hot_safe = np.clip(T_hot_safe, self.hot_T_min, self.hot_T_max)
            T_cold_safe = np.clip(T_cold_safe, self.cold_T_min, self.cold_T_max)
            P_hot_safe = np.clip(P_hot_safe, self.hot_P_min, self.hot_P_max)
            P_cold_safe = np.clip(P_cold_safe, self.cold_P_min, self.cold_P_max)
            
            rho_hot = self.props_hot.density(T_hot_safe, P_hot_safe)
            rho_cold = self.props_cold.density(T_cold_safe, P_cold_safe)
            
            # Clamp densities to prevent division by zero
            rho_hot = np.clip(rho_hot, 1e-6, 1e6)
            rho_cold = np.clip(rho_cold, 1e-6, 1e6)
            
            # Compute velocities from mass flow rates
            # u = m_dot / (rho * A * eps)
            # Clamp porosity to prevent division by zero
            eps_hot_safe = np.clip(eps_hot, 1e-3, 1.0)
            eps_cold_safe = np.clip(eps_cold, 1e-3, 1.0)
            
            u_hot = self.config.fluid.m_dot_hot / (rho_hot * self.A_hot * eps_hot_safe)
            u_cold = self.config.fluid.m_dot_cold / (rho_cold * self.A_cold * eps_cold_safe)
            
            # Clamp velocities to reasonable range
            u_hot = np.clip(u_hot, 1e-6, 100.0)  # m/s
            u_cold = np.clip(u_cold, 1e-6, 100.0)  # m/s
            
            # Heat transfer coefficients
            h_htc_hot = self.rve_db.h_hot(u_hot, d_field)
            h_htc_cold = self.rve_db.h_cold(u_cold, d_field)
            
            # Clamp heat transfer coefficients
            h_htc_hot = np.clip(h_htc_hot, 1.0, 1e6)  # W/(m²·K)
            h_htc_cold = np.clip(h_htc_cold, 1.0, 1e6)  # W/(m²·K)
            
            # Volumetric heat transfer: Q_vol = h * A_surf/V * (T_solid - T_fluid)
            A_surf_V = self.rve_db.A_surf_V(d_field)  # 1/m
            A_surf_V = np.clip(A_surf_V, 1.0, 1e6)  # Clamp to reasonable range
            
            Q_vol_hot = h_htc_hot * A_surf_V * (T_solid[:-1] - T_hot[:-1])
            Q_vol_cold = h_htc_cold * A_surf_V * (T_solid[:-1] - T_cold[:-1])
            
            # Clamp volumetric heat transfer to prevent instability
            Q_max = 1e6  # W/m³, maximum reasonable heat transfer rate
            Q_vol_hot = np.clip(Q_vol_hot, -Q_max, Q_max)
            Q_vol_cold = np.clip(Q_vol_cold, -Q_max, Q_max)
            
            # Energy balance: d(m_dot * h)/dx = Q_vol * A_cross
            # For constant m_dot: m_dot * dh/dx = Q_vol * A_cross
            cp_hot = self.props_hot.specific_heat(T_hot_safe, P_hot_safe)
            cp_cold = self.props_cold.specific_heat(T_cold_safe, P_cold_safe)
            
            # Clamp specific heats
            cp_hot = np.clip(cp_hot, 100.0, 1e6)  # J/(kg·K)
            cp_cold = np.clip(cp_cold, 100.0, 1e6)  # J/(kg·K)
            
            # Update enthalpies with under-relaxation
            relax_h = self.config.solver.relax * 0.5  # More conservative for enthalpy
            
            # CRITICAL FIX: Reset hot inlet boundary condition each iteration
            # Hot side: inlet at x=0 (index 0)
            T_hot[0] = self.config.fluid.T_hot_in
            # Compute inlet enthalpy using inlet properties for consistency
            cp_hot_inlet = self.props_hot.specific_heat(
                self.config.fluid.T_hot_in, 
                self.config.fluid.P_hot_in
            )
            cp_hot_inlet = np.clip(cp_hot_inlet, 100.0, 1e6)
            h_hot[0] = cp_hot_inlet * self.config.fluid.T_hot_in
            
            # Hot side: forward difference (flows +x)
            for i in range(1, self.n + 1):
                # Hot side: forward difference
                dh_dx_hot = Q_vol_hot[i-1] * self.A_hot / self.config.fluid.m_dot_hot
                h_hot_new = h_hot[i-1] + dh_dx_hot * self.dx
                
                # Under-relaxation
                h_hot[i] = float((1 - relax_h) * h_hot[i] + relax_h * h_hot_new)
                
                # Convert enthalpy to temperature using current cp
                # Use property at the segment midpoint (i-1) for segment i
                cp_use = cp_hot[i-1] if i <= len(cp_hot) else cp_hot[-1]
                if cp_use <= 0 or not np.isfinite(cp_use):
                    cp_use = cp_hot_inlet  # Fallback to inlet cp
                T_hot_new = h_hot[i] / cp_use
                
                # Clamp temperature to fluid-specific limits
                T_hot_new = float(np.clip(T_hot_new, self.hot_T_min, self.hot_T_max))
                
                # Under-relaxation for temperature
                T_hot[i] = float((1 - self.config.solver.relax) * T_hot[i] + \
                           self.config.solver.relax * T_hot_new)
                
                # Update enthalpy to be consistent with clamped temperature
                h_hot[i] = cp_use * T_hot[i]
            
            # Cold side: backward difference (flows -x, inlet at x=L)
            # CRITICAL FIX: Reset inlet condition each iteration with proper property evaluation
            T_cold[-1] = self.config.fluid.T_cold_in
            # Compute inlet enthalpy using inlet properties for consistency
            cp_cold_inlet = self.props_cold.specific_heat(
                self.config.fluid.T_cold_in,
                self.config.fluid.P_cold_in
            )
            cp_cold_inlet = np.clip(cp_cold_inlet, 100.0, 1e6)
            h_cold[-1] = cp_cold_inlet * self.config.fluid.T_cold_in
            
            # Cold side flows in -x direction (from x=L to x=0)
            # Energy balance: m_dot * dh/dx = Q_vol * A_cross
            # For flow in -x: dh/dx = -Q_vol * A_cross / m_dot (negative because dx is negative)
            # So: h[j] = h[j+1] + (Q_vol * A_cross / m_dot) * dx
            # But since we're going backwards, dx is negative, so we add
            for j in range(self.n - 1, -1, -1):
                # Cold side receives heat, so enthalpy increases as we go from inlet to outlet
                # Q_vol_cold is positive when T_solid > T_cold (cold side heats up)
                dh_dx_cold = Q_vol_cold[j] * self.A_cold / self.config.fluid.m_dot_cold
                # Since cold flows -x, and we're integrating backwards, we add
                h_cold_new = h_cold[j+1] + dh_dx_cold * self.dx
                
                # Under-relaxation
                h_cold[j] = float((1 - relax_h) * h_cold[j] + relax_h * h_cold_new)
                
                # Convert enthalpy to temperature
                # Use property at segment j (which corresponds to segment midpoint)
                cp_use_cold = cp_cold[j] if j < len(cp_cold) else cp_cold[-1]
                if cp_use_cold <= 0 or not np.isfinite(cp_use_cold):
                    cp_use_cold = cp_cold_inlet  # Fallback to inlet cp
                T_cold_new = h_cold[j] / cp_use_cold
                
                # Clamp temperature to fluid-specific limits
                T_cold_new = float(np.clip(T_cold_new, self.cold_T_min, self.cold_T_max))
                
                # Under-relaxation for temperature
                T_cold[j] = float((1 - self.config.solver.relax) * T_cold[j] + \
                           self.config.solver.relax * T_cold_new)
                
                # Update enthalpy to be consistent with clamped temperature
                h_cold[j] = float(cp_use_cold * T_cold[j])
            
            # Solid energy balance with improved stability
            # Heat balance: h_htc_hot * (T_solid - T_hot) = h_htc_cold * (T_cold - T_solid)
            # Solving: T_solid = (h_htc_hot * T_hot + h_htc_cold * T_cold) / (h_htc_hot + h_htc_cold)
            denominator = h_htc_hot + h_htc_cold + 1e-10
            T_solid_new = (h_htc_hot * T_hot[:-1] + h_htc_cold * T_cold[:-1]) / denominator
            
            # Clamp solid temperature to valid range (between hot and cold limits)
            T_solid_min = min(self.hot_T_min, self.cold_T_min)
            T_solid_max = max(self.hot_T_max, self.cold_T_max)
            T_solid_new = np.clip(T_solid_new, T_solid_min, T_solid_max)
            
            # Under-relaxation for solid temperature
            T_solid_new_float = T_solid_new.astype(float) if hasattr(T_solid_new, 'astype') else np.array(T_solid_new, dtype=float)
            T_solid[:-1] = (1 - self.config.solver.relax) * T_solid[:-1] + \
                          self.config.solver.relax * T_solid_new_float
            
            # IMPROVED: Better boundary extrapolation for solid temperature
            # Use weighted average of last interior point and boundary fluid temperatures
            if len(T_solid) > 1:
                # At outlet (x=L): average of last interior point and outlet fluid temperatures
                T_solid_outlet = 0.5 * T_solid[-2] + 0.25 * T_hot[-1] + 0.25 * T_cold[0]
                T_solid[-1] = float(np.clip(T_solid_outlet, T_solid_min, T_solid_max))
            else:
                T_solid[-1] = float(T_solid[0]) if len(T_solid) > 0 else float((T_hot[0] + T_cold[0]) / 2.0)
            
            # Pressure drop: Darcy-Forchheimer
            # dP/dx = -(mu/kappa) * u - beta * rho * u * |u|
            # This gives pressure drop per unit length (always negative for flow)
            # Use safe temperature and pressure values (defined earlier in iteration)
            rho_hot = self.props_hot.density(T_hot_safe, P_hot_safe)
            rho_cold = self.props_cold.density(T_cold_safe, P_cold_safe)
            mu_hot = self.props_hot.viscosity(T_hot_safe, P_hot_safe)
            mu_cold = self.props_cold.viscosity(T_cold_safe, P_cold_safe)
            
            # Pressure gradient in +x direction (always negative for flow)
            dP_dx_hot = -(mu_hot / kappa_hot) * u_hot - beta_hot * rho_hot * u_hot * np.abs(u_hot)
            dP_dx_cold = -(mu_cold / kappa_cold) * u_cold - beta_cold * rho_cold * u_cold * np.abs(u_cold)
            
            # Clamp pressure gradients to prevent instability
            # Allow much larger gradients (up to -1e8 Pa/m = -100 MPa/m) to capture property effects
            # Previous limit of -1e6 Pa/m was too restrictive and masked property variations
            dP_dx_hot = np.clip(dP_dx_hot, -1e8, 0.0)  # Negative (pressure drop), increased from -1e6
            dP_dx_cold = np.clip(dP_dx_cold, -1e8, 0.0)  # Negative (pressure drop), increased from -1e6
            
            # Integrate pressure
            # Hot side: flows +x, pressure decreases (dP/dx < 0)
            for i in range(1, self.n + 1):
                P_hot[i] = float(P_hot[i-1] + dP_dx_hot[i-1] * self.dx)
                # Clamp pressure to valid range
                P_hot[i] = float(np.clip(P_hot[i], self.hot_P_min, self.hot_P_max))
            
            # Cold side: flows -x (from x=L to x=0)
            # Pressure decreases in the flow direction
            # Since cold flows -x, pressure at x=0 (outlet) < pressure at x=L (inlet)
            # dP_dx_cold is pressure gradient in +x direction (negative for flow)
            # When integrating backwards (from L to 0), we integrate in -x direction
            # For flow in -x direction: dP/d(-x) = -dP/dx
            # Since dP/dx is negative, dP/d(-x) is positive (pressure increases in -x)
            # But we want pressure to decrease in flow direction, so:
            # P[j] = P[j+1] + dP_dx_cold[j] * dx
            # Since dP_dx_cold is negative and dx is positive, this gives pressure drop
            for j in range(self.n - 1, -1, -1):
                # dP_dx_cold is negative (pressure drop in +x)
                # For cold flowing -x, pressure still drops, so we add the negative gradient
                P_cold[j] = float(P_cold[j+1] + dP_dx_cold[j] * self.dx)
                # Clamp pressure to valid range
                P_cold[j] = float(np.clip(P_cold[j], self.cold_P_min, self.cold_P_max))
            
            # Check for unphysical values (NaN, Inf, or extremely large)
            if (not np.all(np.isfinite(T_hot)) or 
                not np.all(np.isfinite(T_cold)) or 
                not np.all(np.isfinite(T_solid)) or
                np.any(np.abs(T_hot) > 1e6) or 
                np.any(np.abs(T_cold) > 1e6) or
                np.any(np.abs(T_solid) > 1e6)):
                # Reset to safe values and warn
                import warnings
                warnings.warn(
                    f"Solver iteration {it}: Unphysical temperatures detected. "
                    f"Resetting to inlet conditions.",
                    RuntimeWarning
                )
                # Reset to inlet conditions
                T_hot = np.linspace(
                    self.config.fluid.T_hot_in,
                    max(self.config.fluid.T_hot_in - 10.0, self.hot_T_min),
                    self.n + 1
                )
                T_cold = np.linspace(
                    min(self.config.fluid.T_cold_in + 10.0, self.cold_T_max),
                    self.config.fluid.T_cold_in,
                    self.n + 1
                )
                T_solid = (T_hot + T_cold) / 2.0
                # Re-initialize enthalpies
                h_hot = cp_ref_hot * T_hot
                h_cold = cp_ref_cold * T_cold
                continue  # Skip convergence check and try again
            
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
        # Hot side: inlet at x=0 (index 0), outlet at x=L (index -1)
        # Pressure drop is positive (inlet - outlet)
        delta_P_hot = P_hot[0] - P_hot[-1]
        
        # Cold side: inlet at x=L (index -1), outlet at x=0 (index 0)
        # Pressure drop is positive (inlet - outlet)
        delta_P_cold = P_cold[-1] - P_cold[0]
        
        # Ensure pressure drops are positive (pressure always decreases in flow direction)
        delta_P_hot = max(0.0, delta_P_hot)
        delta_P_cold = max(0.0, delta_P_cold)
        
        # Final validation: check for unphysical results
        if (not np.isfinite(Q) or abs(Q) > 1e15 or
            not np.all(np.isfinite(T_hot)) or not np.all(np.isfinite(T_cold)) or
            not np.all(np.isfinite(P_hot)) or not np.all(np.isfinite(P_cold))):
            import warnings
            warnings.warn(
                "Solver produced unphysical results. Results may be invalid.",
                RuntimeWarning
            )
        
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
        # Ensure initial values are within valid ranges
        T_hot_in = np.clip(self.config.fluid.T_hot_in, self.hot_T_min, self.hot_T_max)
        T_cold_in = np.clip(self.config.fluid.T_cold_in, self.cold_T_min, self.cold_T_max)
        P_hot_in = np.clip(self.config.fluid.P_hot_in, self.hot_P_min, self.hot_P_max)
        P_cold_in = np.clip(self.config.fluid.P_cold_in, self.cold_P_min, self.cold_P_max)
        
        T_hot = np.full(self.n + 1, T_hot_in)
        T_cold = np.full(self.n + 1, T_cold_in)
        T_solid = (T_hot + T_cold) / 2.0
        P_hot = np.full(self.n + 1, P_hot_in)
        P_cold = np.full(self.n + 1, P_cold_in)
        
        # Initialize enthalpies (use safe inlet values)
        cp_ref_hot = self.props_hot.specific_heat(T_hot_in, P_hot_in)
        cp_ref_cold = self.props_cold.specific_heat(T_cold_in, P_cold_in)
        h_hot = cp_ref_hot * T_hot
        h_cold = cp_ref_cold * T_cold
        
        # Cumulative path distance for x coordinate
        s_hot = np.concatenate([[0], np.cumsum(ds_hot)])
        s_cold = np.concatenate([[0], np.cumsum(ds_cold)])
        
        # Iterative solve for energy balance
        for it in range(self.config.solver.max_iter):
            T_hot_old = T_hot.copy()
            T_cold_old = T_cold.copy()
            T_solid_old = T_solid.copy()
            
            # Compute velocities from mass flow rates
            # Use safe temperature and pressure values for property lookups
            T_hot_safe_2d = np.clip(T_hot[:-1], self.hot_T_min, self.hot_T_max)
            T_cold_safe_2d = np.clip(T_cold[:-1], self.cold_T_min, self.cold_T_max)
            P_hot_safe_2d = np.clip(P_hot[:-1], self.hot_P_min, self.hot_P_max)
            P_cold_safe_2d = np.clip(P_cold[:-1], self.cold_P_min, self.cold_P_max)
            T_hot_safe_2d = np.nan_to_num(
                T_hot_safe_2d, 
                nan=T_hot_in, 
                posinf=T_hot_in, 
                neginf=T_hot_in
            )
            T_cold_safe_2d = np.nan_to_num(
                T_cold_safe_2d, 
                nan=T_cold_in, 
                posinf=T_cold_in, 
                neginf=T_cold_in
            )
            P_hot_safe_2d = np.nan_to_num(
                P_hot_safe_2d, 
                nan=P_hot_in, 
                posinf=P_hot_in, 
                neginf=P_hot_in
            )
            P_cold_safe_2d = np.nan_to_num(
                P_cold_safe_2d, 
                nan=P_cold_in, 
                posinf=P_cold_in, 
                neginf=P_cold_in
            )
            # Final clamp after NaN replacement
            T_hot_safe_2d = np.clip(T_hot_safe_2d, self.hot_T_min, self.hot_T_max)
            T_cold_safe_2d = np.clip(T_cold_safe_2d, self.cold_T_min, self.cold_T_max)
            P_hot_safe_2d = np.clip(P_hot_safe_2d, self.hot_P_min, self.hot_P_max)
            P_cold_safe_2d = np.clip(P_cold_safe_2d, self.cold_P_min, self.cold_P_max)
            
            u_hot = self.config.fluid.m_dot_hot / (
                self.props_hot.density(T_hot_safe_2d, P_hot_safe_2d) * 
                self.A_hot * eps_hot
            )
            u_cold = self.config.fluid.m_dot_cold / (
                self.props_cold.density(T_cold_safe_2d, P_cold_safe_2d) * 
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
            cp_hot = self.props_hot.specific_heat(T_hot_safe_2d, P_hot_safe_2d)
            cp_cold = self.props_cold.specific_heat(T_cold_safe_2d, P_cold_safe_2d)
            
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
            # Use safe temperature and pressure values for property lookups (fluid-specific limits)
            T_hot_safe_2d = np.clip(T_hot[:-1], self.hot_T_min, self.hot_T_max)
            T_cold_safe_2d = np.clip(T_cold[:-1], self.cold_T_min, self.cold_T_max)
            P_hot_safe_2d = np.clip(P_hot[:-1], self.hot_P_min, self.hot_P_max)
            P_cold_safe_2d = np.clip(P_cold[:-1], self.cold_P_min, self.cold_P_max)
            T_hot_safe_2d = np.nan_to_num(
                T_hot_safe_2d, 
                nan=self.config.fluid.T_hot_in, 
                posinf=self.config.fluid.T_hot_in, 
                neginf=self.config.fluid.T_hot_in
            )
            T_cold_safe_2d = np.nan_to_num(
                T_cold_safe_2d, 
                nan=self.config.fluid.T_cold_in, 
                posinf=self.config.fluid.T_cold_in, 
                neginf=self.config.fluid.T_cold_in
            )
            P_hot_safe_2d = np.nan_to_num(
                P_hot_safe_2d, 
                nan=self.config.fluid.P_hot_in, 
                posinf=self.config.fluid.P_hot_in, 
                neginf=self.config.fluid.P_hot_in
            )
            P_cold_safe_2d = np.nan_to_num(
                P_cold_safe_2d, 
                nan=self.config.fluid.P_cold_in, 
                posinf=self.config.fluid.P_cold_in, 
                neginf=self.config.fluid.P_cold_in
            )
            # Final clamp after NaN replacement
            T_hot_safe_2d = np.clip(T_hot_safe_2d, self.hot_T_min, self.hot_T_max)
            T_cold_safe_2d = np.clip(T_cold_safe_2d, self.cold_T_min, self.cold_T_max)
            P_hot_safe_2d = np.clip(P_hot_safe_2d, self.hot_P_min, self.hot_P_max)
            P_cold_safe_2d = np.clip(P_cold_safe_2d, self.cold_P_min, self.cold_P_max)
            
            rho_hot = self.props_hot.density(T_hot_safe_2d, P_hot_safe_2d)
            rho_cold = self.props_cold.density(T_cold_safe_2d, P_cold_safe_2d)
            mu_hot = self.props_hot.viscosity(T_hot_safe_2d, P_hot_safe_2d)
            mu_cold = self.props_cold.viscosity(T_cold_safe_2d, P_cold_safe_2d)
            
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

