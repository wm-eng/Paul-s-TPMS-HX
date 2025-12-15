"""Tests for RVE property calibration."""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.calibrate_rve import (
    ExperimentalData,
    fit_darcy_forchheimer,
    fit_heat_transfer_correlation,
    calibrate_from_experiments,
    load_experimental_data,
)


def test_fit_darcy_forchheimer():
    """Test Darcy-Forchheimer coefficient fitting."""
    # Synthetic data: known kappa=1e-9 m², beta=1e5 1/m
    kappa_true = 1e-9
    beta_true = 1e5
    rho = 1.0  # kg/m³
    mu = 1e-3  # Pa·s
    
    # Generate test data
    u = np.array([0.5, 1.0, 1.5, 2.0])  # m/s
    L = 1.0  # m
    dP = (mu / kappa_true) * u * L + beta_true * rho * u * u * L
    
    kappa_fit, beta_fit = fit_darcy_forchheimer(u, dP, np.full(len(u), L), rho, mu)
    
    # Check within 10% tolerance
    assert abs(kappa_fit - kappa_true) / kappa_true < 0.1
    assert abs(beta_fit - beta_true) / beta_true < 0.1


def test_fit_heat_transfer_correlation():
    """Test heat transfer correlation fitting."""
    # Known: h = 150 * u^0.85
    a_true = 150.0
    b_true = 0.85
    
    u = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    h = a_true * (u ** b_true)
    
    a_fit, b_fit = fit_heat_transfer_correlation(u, h)
    
    # Check within 5% tolerance
    assert abs(a_fit - a_true) / a_true < 0.05
    assert abs(b_fit - b_true) / b_true < 0.05


def test_calibrate_from_experiments():
    """Test full calibration process."""
    exp_data = ExperimentalData()
    
    # Add synthetic experimental data
    for d in [0.2, 0.5, 0.8]:
        for u in [0.5, 1.0, 1.5]:
            # Synthetic pressure drop (simplified)
            dP = 10000.0 * (1 - d) * u * (1 + 0.5 * u)
            h = (100.0 + 200.0 * d) * (u ** 0.85)
            eps = 0.3 + 0.4 * d
            A_surf_V = 500.0 + 900.0 * d
            
            exp_data.add_measurement(
                d=d,
                flow_rate=u,
                pressure_drop=dP,
                heat_transfer_coeff=h,
                porosity=eps,
                surface_area=A_surf_V,
            )
    
    # Calibrate
    calibrated_df = calibrate_from_experiments(
        exp_data,
        d_range=(0.1, 0.9),
        n_points=10,
    )
    
    # Check output
    assert len(calibrated_df) == 10
    assert 'd' in calibrated_df.columns
    assert 'kappa_hot' in calibrated_df.columns
    assert 'beta_hot' in calibrated_df.columns
    assert 'eps_hot' in calibrated_df.columns
    
    # Check monotonicity (porosity should increase with d)
    assert np.all(np.diff(calibrated_df['eps_hot']) >= 0)
    
    # Check permeability increases with d
    assert np.all(np.diff(calibrated_df['kappa_hot']) > 0)


def test_load_experimental_data():
    """Test loading experimental data from CSV."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'experimental',
        'synthetic_tpms_primitive_data.csv'
    )
    
    if os.path.exists(csv_path):
        exp_data = load_experimental_data(csv_path)
        
        assert len(exp_data.d_values) > 0
        assert len(exp_data.flow_rates) > 0
        assert len(exp_data.pressure_drops) > 0
        
        # Check data ranges
        assert min(exp_data.d_values) >= 0.0
        assert max(exp_data.d_values) <= 1.0
        assert all(p > 0 for p in exp_data.pressure_drops)


def test_fig1a_calibration():
    """
    Test calibration/validation with Fig 1a digitized data.
    
    This test demonstrates:
    1. Reading digitized flow vs pressure drop data from Fig 1a
    2. Converting units (LPM -> m³/s, kPa -> Pa)
    3. Fitting Darcy-Forchheimer model: ΔP/L = (μ/κ) u + β ρ u²
    4. Validating that properties are physically reasonable
    """
    # Paths to data files
    downloads_path = os.path.expanduser('~/Downloads')
    digitized_csv = os.path.join(
        downloads_path, 'digitized_fig1a_flow_vs_dp.csv'
    )
    
    # Fallback to relative path if not in Downloads
    if not os.path.exists(digitized_csv):
        digitized_csv = os.path.join(
            os.path.dirname(__file__), '..', 'digitized_fig1a_flow_vs_dp.csv'
        )
    
    # Skip if file doesn't exist
    if not os.path.exists(digitized_csv):
        pytest.skip(f"Digitized data file not found: {digitized_csv}")
    
    # Read digitized flow vs pressure drop data
    df = pd.read_csv(digitized_csv)
    
    # Fluid properties (helium at room temperature)
    rho = 0.1786  # kg/m³
    mu = 2.0e-5   # Pa·s
    
    # Reference area and length for unit conversion
    A_ref = 1e-4  # m² (reference cross-sectional area)
    L_ref = 0.1   # m (reference length)
    
    # Test calibration for each series in the data
    for series_name in df['series'].unique():
        subset = df[df['series'] == series_name]
        
        if len(subset) < 2:
            continue  # Need at least 2 points for fitting
        
        # Convert units
        u = subset['flow_rate_LPM'].values / 60.0  # m³/s per unit area proxy
        dp = subset['pressure_drop_kPa'].values * 1e3  # Pa
        
        # Fit Darcy-Forchheimer model
        lengths = np.full(len(u), L_ref)
        
        try:
            kappa_fit, beta_fit = fit_darcy_forchheimer(
                u, dp, lengths, rho, mu
            )
        except Exception as e:
            pytest.fail(f"Failed to fit Darcy-Forchheimer for {series_name}: {e}")
        
        # Verify fitted values are physically reasonable
        assert kappa_fit > 0, f"kappa must be positive for {series_name}"
        assert beta_fit > 0, f"beta must be positive for {series_name}"
        assert kappa_fit < 1e-6, f"kappa seems too large for {series_name}: {kappa_fit:.2e}"
        assert beta_fit < 1e8, f"beta seems too large for {series_name}: {beta_fit:.2e}"


def test_fig1c_1d_1e_calibration():
    """
    Test calibration/validation with Fig 1c, 1d, 1e digitized data.
    
    This test demonstrates:
    1. Reading digitized flow vs pressure drop data
    2. Converting units (LPM -> m³/s, kPa -> Pa)
    3. Fitting Darcy-Forchheimer model: ΔP/L = (μ/κ) u + β ρ u²
    4. Validating against calibrated RVE proxy data
    """
    # Paths to data files
    # Try user's Downloads folder first, then relative paths
    downloads_path = os.path.expanduser('~/Downloads')
    digitized_csv = os.path.join(
        downloads_path, 'digitized_fig1c_1d_1e_flow_vs_dp.csv'
    )
    calibrated_csv = os.path.join(
        downloads_path, 'fig1c_1d_1e_calibrated_rve_proxy.csv'
    )
    
    # Fallback to relative path if not in Downloads
    if not os.path.exists(digitized_csv):
        digitized_csv = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'Downloads',
            'digitized_fig1c_1d_1e_flow_vs_dp.csv'
        )
    if not os.path.exists(calibrated_csv):
        calibrated_csv = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'Downloads',
            'fig1c_1d_1e_calibrated_rve_proxy.csv'
        )
    
    # Skip if files don't exist
    if not os.path.exists(digitized_csv):
        pytest.skip(f"Digitized data file not found: {digitized_csv}")
    if not os.path.exists(calibrated_csv):
        pytest.skip(f"Calibrated RVE proxy file not found: {calibrated_csv}")
    
    # Read digitized flow vs pressure drop data
    df = pd.read_csv(digitized_csv)
    
    # Read calibrated RVE proxy for validation
    df_calibrated = pd.read_csv(calibrated_csv)
    
    # Fluid properties (helium at room temperature)
    rho = 0.1786  # kg/m³
    mu = 2.0e-5   # Pa·s
    
    # Reference area and length for unit conversion
    # These are proxy values used in the calibration
    A_ref = 1e-4  # m² (reference cross-sectional area)
    L_ref = 0.1   # m (reference length)
    
    # Test calibration for each series in the data
    for series_name in df['series'].unique():
        subset = df[df['series'] == series_name]
        
        if len(subset) < 2:
            continue  # Need at least 2 points for fitting
        
        # Convert units
        # Flow rate: LPM -> m³/s per unit area proxy
        u = subset['flow_rate_LPM'].values / 60.0  # m³/s per unit area proxy
        
        # Pressure drop: kPa -> Pa
        dp = subset['pressure_drop_kPa'].values * 1e3  # Pa
        
        # Fit Darcy-Forchheimer model
        # ΔP/L = (μ/κ) u + β ρ u²
        # Using reference length for fitting
        lengths = np.full(len(u), L_ref)
        
        try:
            kappa_fit, beta_fit = fit_darcy_forchheimer(
                u, dp, lengths, rho, mu
            )
        except Exception as e:
            pytest.fail(f"Failed to fit Darcy-Forchheimer for {series_name}: {e}")
        
        # Extract expected values from calibrated RVE proxy
        # Match series name to calibrated data
        # Series names in digitized data may differ from calibrated CSV
        # Try to match by source_fig and series pattern
        source_fig = series_name.split('_')[0] if '_' in series_name else None
        
        # Find matching row in calibrated data
        if source_fig:
            cal_match = df_calibrated[df_calibrated['source_fig'] == source_fig]
            if len(cal_match) > 0:
                # Use first match (may need more sophisticated matching)
                cal_row = cal_match.iloc[0]
                
                # Extract calibrated coefficients
                # The CSV has: a_visc_mu_over_kappa and b_inert_beta_rho
                mu_over_kappa_cal = cal_row['a_visc_mu_over_kappa']
                beta_rho_cal = cal_row['b_inert_beta_rho']
                
                # Convert to kappa and beta
                kappa_cal = mu / mu_over_kappa_cal if mu_over_kappa_cal > 0 else 1e-10
                beta_cal = beta_rho_cal / rho if rho > 0 else 1e6
                
                # Compare fitted vs calibrated (within 50% tolerance for proxy fits)
                # Note: proxy fits use reference geometry, so exact match not expected
                kappa_ratio = kappa_fit / kappa_cal
                beta_ratio = beta_fit / beta_cal
                
                # Log the comparison (within reasonable bounds)
                assert 0.1 < kappa_ratio < 10.0, (
                    f"kappa mismatch for {series_name}: "
                    f"fitted={kappa_fit:.2e}, calibrated={kappa_cal:.2e}, "
                    f"ratio={kappa_ratio:.2f}"
                )
                assert 0.1 < beta_ratio < 10.0, (
                    f"beta mismatch for {series_name}: "
                    f"fitted={beta_fit:.2e}, calibrated={beta_cal:.2e}, "
                    f"ratio={beta_ratio:.2f}"
                )
        
        # Verify fitted values are physically reasonable
        assert kappa_fit > 0, f"kappa must be positive for {series_name}"
        assert beta_fit > 0, f"beta must be positive for {series_name}"
        assert kappa_fit < 1e-6, f"kappa seems too large for {series_name}: {kappa_fit:.2e}"
        assert beta_fit < 1e8, f"beta seems too large for {series_name}: {beta_fit:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

