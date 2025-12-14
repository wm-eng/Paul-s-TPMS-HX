"""RVE property calibration from experimental data."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import PchipInterpolator
from typing import Dict, Tuple, Optional
import os


class ExperimentalData:
    """
    Container for experimental measurements.
    
    Expected measurements:
    - Pressure drop vs flow rate at different d values
    - Heat transfer coefficient vs velocity at different d values
    - Porosity measurements
    - Surface area measurements
    """
    
    def __init__(self):
        self.d_values = []  # Design variable values
        self.flow_rates = []  # m_dot (kg/s) or u (m/s)
        self.pressure_drops = []  # ΔP (Pa)
        self.heat_transfer_coeffs = []  # h (W/(m²·K))
        self.porosities = []  # ε (dimensionless)
        self.surface_areas = []  # A_surf/V (1/m)
        self.temperatures = []  # T (K) for property evaluation
        self.fluid_properties = {}  # rho, mu, etc.
    
    def add_measurement(
        self,
        d: float,
        flow_rate: float,
        pressure_drop: float,
        heat_transfer_coeff: Optional[float] = None,
        porosity: Optional[float] = None,
        surface_area: Optional[float] = None,
        temperature: float = 300.0,
    ):
        """Add a single experimental measurement."""
        self.d_values.append(d)
        self.flow_rates.append(flow_rate)
        self.pressure_drops.append(pressure_drop)
        self.heat_transfer_coeffs.append(heat_transfer_coeff)
        self.porosities.append(porosity)
        self.surface_areas.append(surface_area)
        self.temperatures.append(temperature)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'd': self.d_values,
            'flow_rate': self.flow_rates,
            'pressure_drop': self.pressure_drops,
            'heat_transfer_coeff': self.heat_transfer_coeffs,
            'porosity': self.porosities,
            'surface_area': self.surface_areas,
            'temperature': self.temperatures,
        })


def fit_darcy_forchheimer(
    flow_rates: np.ndarray,
    pressure_drops: np.ndarray,
    lengths: np.ndarray,
    rho: float,
    mu: float,
) -> Tuple[float, float]:
    """
    Fit Darcy-Forchheimer coefficients from pressure drop data.
    
    Model: ΔP/L = (μ/κ)u + βρu²
    
    Parameters
    ----------
    flow_rates : np.ndarray
        Flow rates (m/s velocity or m³/s volumetric)
    pressure_drops : np.ndarray
        Pressure drops (Pa)
    lengths : np.ndarray
        Flow path lengths (m)
    rho : float
        Fluid density (kg/m³)
    mu : float
        Fluid viscosity (Pa·s)
        
    Returns
    -------
    kappa : float
        Permeability (m²)
    beta : float
        Forchheimer coefficient (1/m)
    """
    # Convert to velocity if needed (assume flow_rate is velocity for now)
    u = np.asarray(flow_rates)
    dP_dL = np.asarray(pressure_drops) / np.asarray(lengths)
    
    # Linear regression: dP/dL = a*u + b*u²
    # where a = μ/κ, b = βρ
    A = np.column_stack([u, rho * u * np.abs(u)])
    coeffs, _ = np.linalg.lstsq(A, dP_dL, rcond=None)[:2]
    
    a, b = coeffs
    kappa = mu / a if a > 0 else 1e-10
    beta = b / rho if rho > 0 else 1e6
    
    return kappa, beta


def fit_heat_transfer_correlation(
    velocities: np.ndarray,
    h_values: np.ndarray,
) -> Tuple[float, float]:
    """
    Fit heat transfer correlation: h = a * u^b
    
    Parameters
    ----------
    velocities : np.ndarray
        Flow velocities (m/s)
    h_values : np.ndarray
        Heat transfer coefficients (W/(m²·K))
        
    Returns
    -------
    a : float
        Correlation coefficient
    b : float
        Correlation exponent
    """
    # Log-linear fit: log(h) = log(a) + b*log(u)
    u = np.asarray(velocities)
    h = np.asarray(h_values)
    
    # Filter out zero/negative values
    mask = (u > 0) & (h > 0)
    if not np.any(mask):
        return 150.0, 0.85  # Default values
    
    log_u = np.log(u[mask])
    log_h = np.log(h[mask])
    
    # Linear regression
    coeffs = np.polyfit(log_u, log_h, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])
    
    return a, b


def calibrate_from_experiments(
    exp_data: ExperimentalData,
    d_range: Tuple[float, float] = (0.1, 0.9),
    n_points: int = 20,
    fluid_properties: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Calibrate RVE properties from experimental data.
    
    Parameters
    ----------
    exp_data : ExperimentalData
        Experimental measurements
    d_range : Tuple[float, float]
        Range of d values to generate calibrated table
    n_points : int
        Number of points in calibrated table
    fluid_properties : dict, optional
        Fluid properties (rho, mu) for Darcy-Forchheimer fitting
        
    Returns
    -------
    calibrated_df : pd.DataFrame
        Calibrated RVE table with columns:
        d, kappa_hot, beta_hot, eps_hot, lambda_solid,
        h_a_hot, h_b_hot, h_a_cold, h_b_cold, A_surf_V
    """
    df_exp = exp_data.to_dataframe()
    
    # Default fluid properties (helium at 300K)
    if fluid_properties is None:
        fluid_properties = {
            'rho': 0.1786,  # kg/m³
            'mu': 2.0e-5,  # Pa·s
        }
    
    rho = fluid_properties.get('rho', 0.1786)
    mu = fluid_properties.get('mu', 2.0e-5)
    
    # Group by d value
    d_unique = np.sort(df_exp['d'].unique())
    
    # Storage for calibrated properties
    calibrated_props = {
        'd': [],
        'kappa_hot': [],
        'beta_hot': [],
        'eps_hot': [],
        'lambda_solid': [],
        'h_a_hot': [],
        'h_b_hot': [],
        'h_a_cold': [],
        'h_b_cold': [],
        'A_surf_V': [],
    }
    
    # Calibrate for each d value
    for d in d_unique:
        df_d = df_exp[df_exp['d'] == d]
        
        if len(df_d) < 2:
            continue
        
        # Fit permeability and Forchheimer coefficient
        if 'pressure_drop' in df_d.columns and 'flow_rate' in df_d.columns:
            # Assume unit length for now (can be improved)
            lengths = np.ones(len(df_d))
            try:
                kappa, beta = fit_darcy_forchheimer(
                    df_d['flow_rate'].values,
                    df_d['pressure_drop'].values,
                    lengths,
                    rho,
                    mu,
                )
            except:
                # Fallback to default
                kappa = 1e-10 * (1 + 9 * d)
                beta = 1e6 * (1 - 0.8 * d)
        else:
            kappa = 1e-10 * (1 + 9 * d)
            beta = 1e6 * (1 - 0.8 * d)
        
        # Get porosity (use measured or estimate)
        if 'porosity' in df_d.columns and not df_d['porosity'].isna().all():
            eps = df_d['porosity'].mean()
        else:
            # Estimate from d: higher d -> higher porosity
            eps = 0.3 + 0.4 * d
        
        # Fit heat transfer correlation
        if 'heat_transfer_coeff' in df_d.columns and 'flow_rate' in df_d.columns:
            try:
                h_a, h_b = fit_heat_transfer_correlation(
                    df_d['flow_rate'].values,
                    df_d['heat_transfer_coeff'].values,
                )
            except:
                h_a = 100.0 + 160.0 * d
                h_b = 0.8 + 0.16 * d
        else:
            h_a = 100.0 + 160.0 * d
            h_b = 0.8 + 0.16 * d
        
        # Get surface area
        if 'surface_area' in df_d.columns and not df_d['surface_area'].isna().all():
            A_surf_V = df_d['surface_area'].mean()
        else:
            # Estimate: higher porosity -> more surface area
            A_surf_V = 500.0 + 900.0 * d
        
        # Solid thermal conductivity (assume constant or function of d)
        lambda_solid = 50.0 - 40.0 * d  # Decreases with porosity
        
        calibrated_props['d'].append(d)
        calibrated_props['kappa_hot'].append(kappa)
        calibrated_props['beta_hot'].append(beta)
        calibrated_props['eps_hot'].append(eps)
        calibrated_props['lambda_solid'].append(lambda_solid)
        calibrated_props['h_a_hot'].append(h_a)
        calibrated_props['h_b_hot'].append(h_b)
        calibrated_props['h_a_cold'].append(h_a * 0.8)  # Cold side typically lower
        calibrated_props['h_b_cold'].append(h_b * 0.95)
        calibrated_props['A_surf_V'].append(A_surf_V)
    
    # Create DataFrame and interpolate to desired d range
    df_cal = pd.DataFrame(calibrated_props)
    df_cal = df_cal.sort_values('d')
    
    # Interpolate to uniform grid
    d_target = np.linspace(d_range[0], d_range[1], n_points)
    
    interpolated = {}
    for prop in ['kappa_hot', 'beta_hot', 'eps_hot', 'lambda_solid',
                 'h_a_hot', 'h_b_hot', 'h_a_cold', 'h_b_cold', 'A_surf_V']:
        interp = PchipInterpolator(df_cal['d'].values, df_cal[prop].values)
        interpolated[prop] = interp(d_target)
    
    interpolated['d'] = d_target
    
    return pd.DataFrame(interpolated)


def load_experimental_data(csv_path: str) -> ExperimentalData:
    """
    Load experimental data from CSV file.
    
    Expected CSV columns:
    - d: Design variable value
    - flow_rate: Flow rate (m/s velocity or kg/s mass flow)
    - pressure_drop: Pressure drop (Pa)
    - heat_transfer_coeff: Heat transfer coefficient (W/(m²·K)) [optional]
    - porosity: Porosity (dimensionless) [optional]
    - surface_area: Surface area per unit volume (1/m) [optional]
    - temperature: Temperature (K) [optional]
    """
    # Read CSV, skipping comment lines
    df = pd.read_csv(csv_path, comment='#')
    
    exp_data = ExperimentalData()
    
    for _, row in df.iterrows():
        exp_data.add_measurement(
            d=row['d'],
            flow_rate=row['flow_rate'],
            pressure_drop=row['pressure_drop'],
            heat_transfer_coeff=row.get('heat_transfer_coeff', None),
            porosity=row.get('porosity', None),
            surface_area=row.get('surface_area', None),
            temperature=row.get('temperature', 300.0),
        )
    
    return exp_data

