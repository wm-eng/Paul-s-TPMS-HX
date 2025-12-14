"""Tests for RVE property calibration."""

import sys
import os
import numpy as np
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

