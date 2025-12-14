"""Tests for RVE interpolation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.rve_db import RVEDatabase


def test_rve_load():
    """Test RVE database loading."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_default.csv'
    )
    rve_db = RVEDatabase(csv_path)
    
    assert rve_db.d_min == 0.1
    assert rve_db.d_max == 0.9


def test_rve_interpolation():
    """Test RVE property interpolation."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_default.csv'
    )
    rve_db = RVEDatabase(csv_path)
    
    # Test at known values
    d = np.array([0.1, 0.5, 0.9])
    
    kappa = rve_db.kappa_hot(d)
    beta = rve_db.beta_hot(d)
    eps = rve_db.eps_hot(d)
    lambda_solid = rve_db.lambda_solid(d)
    
    # Check finite and positive
    assert np.all(np.isfinite(kappa))
    assert np.all(kappa > 0)
    assert np.all(np.isfinite(beta))
    assert np.all(beta > 0)
    assert np.all(np.isfinite(eps))
    assert np.all((eps > 0) & (eps < 1))
    assert np.all(np.isfinite(lambda_solid))
    assert np.all(lambda_solid > 0)


def test_rve_clamping():
    """Test that d values are clamped to valid range."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_default.csv'
    )
    rve_db = RVEDatabase(csv_path)
    
    # Test out-of-bounds
    d_low = np.array([0.0, -1.0])
    d_high = np.array([1.0, 2.0])
    
    kappa_low = rve_db.kappa_hot(d_low)
    kappa_high = rve_db.kappa_hot(d_high)
    
    # Should be clamped, not NaN
    assert np.all(np.isfinite(kappa_low))
    assert np.all(np.isfinite(kappa_high))


def test_heat_transfer_correlation():
    """Test heat transfer coefficient correlation."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_default.csv'
    )
    rve_db = RVEDatabase(csv_path)
    
    d = np.array([0.3, 0.5, 0.7])
    u = np.array([1.0, 2.0, 3.0])  # m/s
    
    h_hot = rve_db.h_hot(u, d)
    h_cold = rve_db.h_cold(u, d)
    
    # Check finite and positive
    assert np.all(np.isfinite(h_hot))
    assert np.all(h_hot > 0)
    assert np.all(np.isfinite(h_cold))
    assert np.all(h_cold > 0)
    
    # Check monotonicity w.r.t. velocity
    u2 = 2.0 * u
    h_hot2 = rve_db.h_hot(u2, d)
    assert np.all(h_hot2 > h_hot)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

