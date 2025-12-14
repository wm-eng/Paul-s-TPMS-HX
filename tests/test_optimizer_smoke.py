"""Smoke tests for optimizer (runs without crashing)."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig
from hxopt.rve_db import RVEDatabase
from hxopt.optimize_mma import optimize


def test_optimizer_smoke():
    """Test that optimizer runs 5-10 iterations without crashing."""
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
    
    optimization = OptimizationConfig(
        max_iter=5,  # Just 5 iterations for smoke test
        d_min=0.1,
        d_max=0.9,
        d_init=0.5,
    )
    
    config = Config(
        geometry=geometry,
        fluid=fluid,
        optimization=optimization,
        rve_table_path=os.path.join(
            os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_default.csv'
        ),
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'test'),
    )
    
    rve_db = RVEDatabase(config.rve_table_path)
    
    # Should not raise
    opt_result = optimize(config, rve_db, log_file="smoke_test_log.csv")
    
    # Check that we got some iterations
    assert len(opt_result.iterations) > 0
    assert len(opt_result.Q_values) > 0
    
    # Check that Q values are finite
    assert np.all(np.isfinite(opt_result.Q_values))
    
    print(f"Smoke test passed: {len(opt_result.iterations)} iterations completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

