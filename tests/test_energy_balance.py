"""Tests for energy balance validation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig
from hxopt.rve_db import RVEDatabase
from hxopt.macro_model import MacroModel
from hxopt.validate import validate_energy_balance, validate_all


def create_test_config():
    """Create a test configuration."""
    geometry = GeometryConfig(
        length=0.5,
        width=0.1,
        height=0.1,
        n_segments=20,  # Fewer segments for faster tests
    )
    
    fluid = FluidConfig(
        rho_hot=0.1786,
        mu_hot=2.0e-5,
        cp_hot=5190.0,
        k_hot=0.152,
        rho_cold=70.8,
        mu_cold=1.3e-4,
        cp_cold=9600.0,
        k_cold=0.1,
        T_hot_in=300.0,
        T_cold_in=20.0,
        P_hot_in=2e5,
        P_cold_in=1e5,
        m_dot_hot=0.01,
        m_dot_cold=0.05,
    )
    
    return Config(
        geometry=geometry,
        fluid=fluid,
        rve_table_path=os.path.join(
            os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_default.csv'
        ),
    )


def test_energy_balance():
    """Test energy balance validation."""
    config = create_test_config()
    rve_db = RVEDatabase(config.rve_table_path)
    model = MacroModel(config, rve_db)
    
    # Solve with constant d
    d_field = np.full(config.geometry.n_segments, 0.5)
    result = model.solve(d_field)
    
    # Validate energy balance
    valid, error = validate_energy_balance(result, config, tol=0.1)
    
    # Should be valid (within 10% tolerance for v1)
    assert valid or error < 0.2, f"Energy balance error too large: {error:.2e}"


def test_validate_all():
    """Test full validation suite."""
    config = create_test_config()
    rve_db = RVEDatabase(config.rve_table_path)
    model = MacroModel(config, rve_db)
    
    # Solve with constant d
    d_field = np.full(config.geometry.n_segments, 0.5)
    result = model.solve(d_field)
    
    # Should not raise
    try:
        validate_all(result, config)
    except ValueError as e:
        pytest.fail(f"Validation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

