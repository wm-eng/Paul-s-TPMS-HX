"""RVE property database with interpolation functions."""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from typing import Callable, Optional
import os
import warnings


class RVEDatabase:
    """
    RVE property database with interpolation.
    
    Based on Catchpole-Smith et al. (2019), thermal conductivity is primarily
    a function of volume fraction (porosity ε), not just the design variable d.
    
    Provides functions:
    - kappa_hot(d): Permeability (m²)
    - beta_hot(d): Forchheimer coefficient (1/m)
    - eps_hot(d): Porosity/volume fraction (dimensionless) - PRIMARY PARAMETER
    - lambda_solid(d, eps=None): Solid thermal conductivity (W/(m·K))
      Accounts for volume fraction effects per paper findings
    - A_surf_V(d): Surface area per unit volume (1/m)
    - h_hot(u): Heat transfer coefficient (W/(m²·K)) as function of velocity
    - h_cold(u): Heat transfer coefficient for cold side
    
    Attributes
    ----------
    cell_size : float, optional
        Unit cell size (m). Larger cells typically have higher conductivity
        due to intra-cell convection and better interface coupling.
    """
    
    def __init__(self, csv_path: str, cell_size: Optional[float] = None,
                 metal_name: Optional[str] = None, T_ref: float = 300.0):
        """
        Load RVE table from CSV.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with columns:
            d, kappa_hot, beta_hot, eps_hot, lambda_solid,
            h_a_hot, h_b_hot, h_a_cold, h_b_cold, A_surf_V
            Optional: cell_size (unit cell size in m)
        cell_size : float, optional
            Unit cell size (m). If not provided, will try to read from CSV
            or use default. Larger cells typically have higher conductivity.
        metal_name : str, optional
            Metal name for solid phase. If provided, will override
            lambda_solid from CSV with metal's thermal conductivity.
            See MetalProperties.list_metals() for available options.
        T_ref : float, optional
            Reference temperature (K) for metal conductivity lookup.
            Default: 300K (room temperature).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"RVE table not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required = ['d', 'kappa_hot', 'beta_hot', 'eps_hot', 'lambda_solid']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in RVE table: {missing}")
        
        # Get cell size (unit cell dimension in m)
        # Per Catchpole-Smith et al. (2019), larger cells have higher conductivity
        if cell_size is not None:
            self.cell_size = float(cell_size)
        elif 'cell_size' in df.columns:
            # Use cell size from CSV (should be constant for a given table)
            self.cell_size = float(df['cell_size'].iloc[0])
        else:
            # Default: assume 5mm unit cell (typical for TPMS lattices)
            self.cell_size = 5e-3  # m
            warnings.warn(f"cell_size not specified in CSV or parameter. Using default: {self.cell_size*1000:.1f} mm")
        
        # Sort by d for interpolation
        df = df.sort_values('d')
        d = df['d'].values
        
        # Clamp d to valid range
        self.d_min = float(d.min())
        self.d_max = float(d.max())
        
        # Store porosity (volume fraction) as PRIMARY PARAMETER
        # Per Catchpole-Smith et al. (2019), thermal conductivity is primarily
        # a function of volume fraction, not unit cell type
        self._eps_hot = PchipInterpolator(d, df['eps_hot'].values)
        eps_values = df['eps_hot'].values
        
        # Create interpolators (Pchip preserves monotonicity)
        self._kappa_hot = PchipInterpolator(d, df['kappa_hot'].values)
        self._beta_hot = PchipInterpolator(d, df['beta_hot'].values)
        
        # Lambda_solid: Store base values, but will account for volume fraction
        # The paper shows lambda ~ f(volume_fraction, cell_size, unit_cell_type)
        # If metal_name is provided, use metal's thermal conductivity as base
        if metal_name:
            try:
                from .metal_properties import MetalProperties
                metal = MetalProperties(metal_name)
                k_metal = metal.thermal_conductivity(T_ref)
                # Use metal conductivity as base, scaled by volume fraction
                # The CSV lambda_solid values will be ignored in favor of metal properties
                self.metal = metal
                self.use_metal_properties = True
                self.k_metal_ref = k_metal
                warnings.warn(
                    f"Using metal properties for {metal_name}: "
                    f"k = {k_metal:.1f} W/(m·K) at {T_ref}K. "
                    f"CSV lambda_solid values will be scaled by metal conductivity."
                )
            except ImportError:
                warnings.warn("MetalProperties not available, using CSV values")
                self.use_metal_properties = False
                self.metal = None
        else:
            self.use_metal_properties = False
            self.metal = None
        
        # Store base lambda_solid from CSV (may be overridden by metal)
        self._lambda_solid_base = PchipInterpolator(d, df['lambda_solid'].values)
        
        # Store volume fraction (1 - porosity) for thermal conductivity calculation
        # Volume fraction = 1 - eps (fraction of solid material)
        self._vol_frac = PchipInterpolator(d, 1.0 - eps_values)
        
        # Reference temperature for metal properties
        self.T_ref = T_ref
        
        # Surface area per unit volume (A_surf/V)
        if 'A_surf_V' in df.columns:
            self._A_surf_V = PchipInterpolator(d, df['A_surf_V'].values)
        else:
            # Fallback: estimate from porosity
            # A_surf/V ~ (1 - eps) / L_char, where L_char ~ sqrt(kappa)
            # For v1, use simple correlation based on porosity
            A_surf_V_est = 500.0 + 1000.0 * df['eps_hot'].values  # Rough estimate
            self._A_surf_V = PchipInterpolator(d, A_surf_V_est)
        
        # Heat transfer correlation parameters
        # Default: h = a * u^b
        if 'h_a_hot' in df.columns and 'h_b_hot' in df.columns:
            # Interpolate correlation parameters
            self._h_a_hot = PchipInterpolator(d, df['h_a_hot'].values)
            self._h_b_hot = PchipInterpolator(d, df['h_b_hot'].values)
        else:
            # Fallback to constant defaults
            self._h_a_hot = lambda d: 150.0
            self._h_b_hot = lambda d: 0.85
        
        if 'h_a_cold' in df.columns and 'h_b_cold' in df.columns:
            self._h_a_cold = PchipInterpolator(d, df['h_a_cold'].values)
            self._h_b_cold = PchipInterpolator(d, df['h_b_cold'].values)
        else:
            self._h_a_cold = lambda d: 120.0
            self._h_b_cold = lambda d: 0.80
    
    def _clamp_d(self, d: np.ndarray) -> np.ndarray:
        """Clamp d to valid range."""
        return np.clip(d, self.d_min, self.d_max)
    
    def kappa_hot(self, d: np.ndarray) -> np.ndarray:
        """
        Permeability for hot side. Units: m².
        
        Parameters
        ----------
        d : np.ndarray
            Channel-bias field values [0, 1]
            
        Returns
        -------
        kappa : np.ndarray
            Permeability values
        """
        d = self._clamp_d(np.asarray(d))
        return self._kappa_hot(d)
    
    def beta_hot(self, d: np.ndarray) -> np.ndarray:
        """
        Forchheimer coefficient for hot side. Units: 1/m.
        
        Parameters
        ----------
        d : np.ndarray
            Channel-bias field values
            
        Returns
        -------
        beta : np.ndarray
            Forchheimer coefficient values
        """
        d = self._clamp_d(np.asarray(d))
        return self._beta_hot(d)
    
    def eps_hot(self, d: np.ndarray) -> np.ndarray:
        """
        Porosity (volume fraction of void space) for hot side.
        Dimensionless. This is a PRIMARY PARAMETER per Catchpole-Smith et al. (2019).
        
        Volume fraction of solid = 1 - eps
        Thermal conductivity is primarily a function of this volume fraction.
        
        Parameters
        ----------
        d : np.ndarray
            Channel-bias field values
            
        Returns
        -------
        eps : np.ndarray
            Porosity values (0 = fully solid, 1 = fully void)
        """
        d = self._clamp_d(np.asarray(d))
        return self._eps_hot(d)
    
    def volume_fraction(self, d: np.ndarray) -> np.ndarray:
        """
        Volume fraction of solid material (1 - porosity).
        This is the PRIMARY PARAMETER for thermal conductivity per the paper.
        
        Parameters
        ----------
        d : np.ndarray
            Channel-bias field values
            
        Returns
        -------
        vol_frac : np.ndarray
            Volume fraction of solid material (0 = fully void, 1 = fully solid)
        """
        d = self._clamp_d(np.asarray(d))
        return self._vol_frac(d)
    
    def lambda_solid(self, d: np.ndarray, eps: Optional[np.ndarray] = None,
                     T: Optional[float] = None) -> np.ndarray:
        """
        Solid thermal conductivity accounting for volume fraction effects.
        Units: W/(m·K).
        
        Per Catchpole-Smith et al. (2019), thermal conductivity is primarily
        a function of volume fraction (1 - porosity), not just the design variable.
        Larger cell sizes also typically have higher conductivity.
        
        If metal_name was specified during initialization, uses metal's
        temperature-dependent thermal conductivity as the base value.
        
        Parameters
        ----------
        d : np.ndarray
            Channel-bias field values
        eps : np.ndarray, optional
            Porosity values. If provided, will use these directly instead of
            interpolating from d. This allows explicit volume fraction control.
        T : float, optional
            Temperature (K) for metal property lookup. If None, uses T_ref.
            Only used if metal properties are enabled.
            
        Returns
        -------
        lambda_solid : np.ndarray
            Thermal conductivity values accounting for volume fraction effects
        """
        d = self._clamp_d(np.asarray(d))
        
        # Get volume fraction (1 - porosity)
        if eps is not None:
            # Use provided porosity directly
            vol_frac = 1.0 - np.asarray(eps)
        else:
            # Interpolate from d
            vol_frac = self._vol_frac(d)
        
        # Base thermal conductivity
        if self.use_metal_properties and self.metal is not None:
            # Use metal's temperature-dependent thermal conductivity
            T_use = T if T is not None else self.T_ref
            k_metal = self.metal.thermal_conductivity(T_use)
            
            # Scale metal conductivity by relative magnitude from CSV
            # This preserves the d-dependence pattern while using real metal properties
            lambda_base_csv = self._lambda_solid_base(d)
            lambda_base_ref = self._lambda_solid_base(np.array([0.5]))[0]  # Reference at d=0.5
            
            # Scale factor: preserve relative variation from CSV
            if lambda_base_ref > 0:
                scale_factor = k_metal / lambda_base_ref
                lambda_base = lambda_base_csv * scale_factor
            else:
                lambda_base = np.full_like(d, k_metal)
        else:
            # Use CSV values directly
            lambda_base = self._lambda_solid_base(d)
        
        # Account for volume fraction effects per paper findings
        # The paper shows: lambda ~ f(volume_fraction)
        # For a given material, higher volume fraction -> higher conductivity
        # Use a simple scaling: lambda_eff = lambda_base * (vol_frac / vol_frac_ref)^n
        # where n ~ 1-2 based on effective medium theory
        
        # Reference volume fraction (use mean from data)
        vol_frac_ref = 0.5  # Typical value
        
        # Scaling exponent (n=1.5 is typical for porous media)
        # This accounts for the primary dependence on volume fraction
        n = 1.5
        
        # Cell size effect: larger cells have higher conductivity
        # Per paper: larger cells allow intra-cell convection and better coupling
        cell_size_ref = 5e-3  # Reference 5mm cell
        cell_size_factor = (self.cell_size / cell_size_ref) ** 0.2  # Weak dependence
        
        # Effective thermal conductivity accounting for volume fraction
        vol_frac_safe = np.clip(vol_frac, 0.1, 0.9)  # Clamp to reasonable range
        lambda_eff = lambda_base * (vol_frac_safe / vol_frac_ref) ** n * cell_size_factor
        
        return lambda_eff
    
    def A_surf_V(self, d: np.ndarray) -> np.ndarray:
        """
        Surface area per unit volume. Units: 1/m.
        
        This is the volumetric heat transfer coefficient parameter that
        characterizes the unit-volume heat transfer capability.
        
        Parameters
        ----------
        d : np.ndarray
            Channel-bias field values
            
        Returns
        -------
        A_surf_V : np.ndarray
            Surface area per unit volume values
        """
        d = self._clamp_d(np.asarray(d))
        return self._A_surf_V(d)
    
    def h_hot(self, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Heat transfer coefficient for hot side. Units: W/(m²·K).
        
        Correlation: h = a(d) * u^b(d)
        
        Parameters
        ----------
        u : np.ndarray
            Velocity (m/s)
        d : np.ndarray
            Channel-bias field values
            
        Returns
        -------
        h : np.ndarray
            Heat transfer coefficient values
        """
        d = self._clamp_d(np.asarray(d))
        u = np.asarray(u)
        a = self._h_a_hot(d)
        b = self._h_b_hot(d)
        # Handle scalar vs array
        if np.isscalar(a):
            return a * (u ** b)
        return a * (u ** b)
    
    def h_cold(self, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Heat transfer coefficient for cold side. Units: W/(m²·K).
        
        Correlation: h = a(d) * u^b(d)
        
        Parameters
        ----------
        u : np.ndarray
            Velocity (m/s)
        d : np.ndarray
            Channel-bias field values
            
        Returns
        -------
        h : np.ndarray
            Heat transfer coefficient values
        """
        d = self._clamp_d(np.asarray(d))
        u = np.asarray(u)
        a = self._h_a_cold(d)
        b = self._h_b_cold(d)
        if np.isscalar(a):
            return a * (u ** b)
        return a * (u ** b)

