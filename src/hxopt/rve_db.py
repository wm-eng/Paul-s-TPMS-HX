"""RVE property database with interpolation functions."""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from typing import Callable, Optional
import os


class RVEDatabase:
    """
    RVE property database with interpolation.
    
    Provides functions:
    - kappa_hot(d): Permeability (m²)
    - beta_hot(d): Forchheimer coefficient (1/m)
    - eps_hot(d): Porosity (dimensionless)
    - lambda_solid(d): Solid thermal conductivity (W/(m·K))
    - A_surf_V(d): Surface area per unit volume (1/m)
    - h_hot(u): Heat transfer coefficient (W/(m²·K)) as function of velocity
    - h_cold(u): Heat transfer coefficient for cold side
    """
    
    def __init__(self, csv_path: str):
        """
        Load RVE table from CSV.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with columns:
            d, kappa_hot, beta_hot, eps_hot, lambda_solid,
            h_a_hot, h_b_hot, h_a_cold, h_b_cold, A_surf_V
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"RVE table not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required = ['d', 'kappa_hot', 'beta_hot', 'eps_hot', 'lambda_solid']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in RVE table: {missing}")
        
        # Sort by d for interpolation
        df = df.sort_values('d')
        d = df['d'].values
        
        # Clamp d to valid range
        self.d_min = float(d.min())
        self.d_max = float(d.max())
        
        # Create interpolators (Pchip preserves monotonicity)
        self._kappa_hot = PchipInterpolator(d, df['kappa_hot'].values)
        self._beta_hot = PchipInterpolator(d, df['beta_hot'].values)
        self._eps_hot = PchipInterpolator(d, df['eps_hot'].values)
        self._lambda_solid = PchipInterpolator(d, df['lambda_solid'].values)
        
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
        Porosity for hot side. Dimensionless.
        
        Parameters
        ----------
        d : np.ndarray
            Channel-bias field values
            
        Returns
        -------
        eps : np.ndarray
            Porosity values
        """
        d = self._clamp_d(np.asarray(d))
        return self._eps_hot(d)
    
    def lambda_solid(self, d: np.ndarray) -> np.ndarray:
        """
        Solid thermal conductivity. Units: W/(m·K).
        
        Parameters
        ----------
        d : np.ndarray
            Channel-bias field values
            
        Returns
        -------
        lambda_solid : np.ndarray
            Thermal conductivity values
        """
        d = self._clamp_d(np.asarray(d))
        return self._lambda_solid(d)
    
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

