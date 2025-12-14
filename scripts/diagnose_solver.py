"""
Detailed diagnostic to understand why solver produces identical results for different d values.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig, SolverConfig
from hxopt.rve_db import RVEDatabase
from hxopt.macro_model import MacroModel


def diagnose_solve(model, d_field, label="", verbose=True):
    """Solve and return detailed diagnostics."""
    print(f"\n{'='*60}")
    print(f"Diagnosing: {label}")
    print(f"{'='*60}")
    print(f"d_field: min={d_field.min():.3f}, max={d_field.max():.3f}, mean={d_field.mean():.3f}")
    
    # Get RVE properties
    kappa_hot = model.rve_db.kappa_hot(d_field)
    beta_hot = model.rve_db.beta_hot(d_field)
    eps_hot = model.rve_db.eps_hot(d_field)
    A_surf_V = model.rve_db.A_surf_V(d_field)
    
    print(f"\nRVE Properties (first 3 cells):")
    print(f"  κ: {kappa_hot[:3]}")
    print(f"  β: {beta_hot[:3]}")
    print(f"  ε: {eps_hot[:3]}")
    print(f"  A_surf/V: {A_surf_V[:3]}")
    
    # Solve
    result = model.solve(d_field)
    
    print(f"\nSolution Results:")
    print(f"  Q = {result.Q/1e6:.6f} MW")
    print(f"  ΔP_hot = {result.delta_P_hot/1e3:.2f} kPa")
    print(f"  ΔP_cold = {result.delta_P_cold/1e3:.2f} kPa")
    print(f"  T_hot_out = {result.T_hot[-1]:.2f} K")
    print(f"  T_cold_out = {result.T_cold[0]:.2f} K")
    
    # Check pressure profile
    if hasattr(result, 'P_hot') and result.P_hot is not None:
        print(f"\nPressure Profile:")
        print(f"  P_hot[0] (inlet) = {result.P_hot[0]/1e3:.2f} kPa")
        print(f"  P_hot[-1] (outlet) = {result.P_hot[-1]/1e3:.2f} kPa")
        print(f"  P_hot range = {result.P_hot.max() - result.P_hot.min():.2f} Pa")
    
    # Check intermediate values that should vary with properties
    # We need to access internal solver state - this is a limitation
    # But we can check if properties are being used by testing manually
    
    return result


def test_property_effects():
    """Test if properties actually affect calculations."""
    print("\n" + "="*60)
    print("TESTING PROPERTY EFFECTS ON CALCULATIONS")
    print("="*60)
    
    # Create minimal config
    geometry = GeometryConfig(
        length=0.5,
        width=0.1,
        height=0.1,
        n_segments=5,  # Very small for testing
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
        # Increased inlet pressures to avoid hitting minimum limit
        # Calibrated RVE properties still produce large pressure drops
        P_hot_in=10e6,  # 10 MPa (increased from 200 kPa)
        P_cold_in=5e6,  # 5 MPa (increased from 100 kPa)
        m_dot_hot=0.01,
        m_dot_cold=0.05,
    )
    
    # Use more aggressive solver settings
    solver = SolverConfig(
        max_iter=500,  # Many iterations
        tol=1e-9,      # Very tight tolerance
        relax=0.05,    # Very low relaxation
    )
    
    optimization = OptimizationConfig(
        max_iter=10,
        d_min=0.1,
        d_max=0.9,
        d_init=0.5,
        step_size=0.1,
    )
    
    config = Config(
        geometry=geometry,
        fluid=fluid,
        optimization=optimization,
        solver=solver,
        rve_table_path=os.path.join(
            os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_calibrated.csv'
        ),
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'diagnose'),
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load RVE database
    print("\nLoading RVE database...")
    rve_db = RVEDatabase(config.rve_table_path)
    
    # Create model
    model = MacroModel(config, rve_db)
    
    # Test with extreme d values
    print("\n" + "="*60)
    print("Testing with extreme d values")
    print("="*60)
    
    d_low = np.full(config.geometry.n_segments, 0.1)
    d_high = np.full(config.geometry.n_segments, 0.9)
    
    result_low = diagnose_solve(model, d_low, "d=0.1 (low)")
    result_high = diagnose_solve(model, d_high, "d=0.9 (high)")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Q difference: {abs(result_high.Q - result_low.Q)/1e6:.6f} MW")
    print(f"Q relative difference: {abs(result_high.Q - result_low.Q)/max(result_low.Q, 1.0)*100:.2f}%")
    print(f"ΔP_hot difference: {abs(result_high.delta_P_hot - result_low.delta_P_hot)/1e3:.2f} kPa")
    print(f"T_hot_out difference: {abs(result_high.T_hot[-1] - result_low.T_hot[-1]):.2f} K")
    print(f"T_cold_out difference: {abs(result_high.T_cold[0] - result_low.T_cold[0]):.2f} K")
    
    if abs(result_high.Q - result_low.Q) < 1.0:
        print("\n⚠️  CRITICAL: Q is essentially identical despite 8x d difference!")
        print("   This confirms the solver is not sensitive to property changes.")
        print("   Root cause is likely in solver algorithm or property usage.")
    else:
        print("\n✅ Q varies with d - model is sensitive!")
    
    # Test pressure gradient calculation
    print("\n" + "="*60)
    print("PRESSURE GRADIENT CALCULATION TEST")
    print("="*60)
    
    # Test what pressure gradient should be for different d values
    mu = 2.0e-5  # Pa·s
    rho = 0.1786  # kg/m³
    m_dot = config.fluid.m_dot_hot
    A = config.geometry.width * config.geometry.height * 0.5
    P_hot_in = config.fluid.P_hot_in
    
    for d_test, label in [(0.1, "d=0.1"), (0.9, "d=0.9")]:
        kappa = rve_db.kappa_hot(np.array([d_test]))[0]
        beta = rve_db.beta_hot(np.array([d_test]))[0]
        eps = rve_db.eps_hot(np.array([d_test]))[0]
        
        # Calculate velocity
        u = m_dot / (rho * A * eps)
        
        # Calculate pressure gradient: dP/dx = -(mu/kappa)*u - beta*rho*u^2
        darcy_term = (mu / kappa) * u
        forch_term = beta * rho * u**2
        dP_dx = -(darcy_term + forch_term)
        
        # Over length L=0.5m, what would pressure drop be?
        L = config.geometry.length  # m
        delta_P_expected = -dP_dx * L
        
        print(f"\n{label}:")
        print(f"  κ={kappa:.2e} m², β={beta:.2e} 1/m, ε={eps:.3f}")
        print(f"  u={u:.3f} m/s")
        print(f"  dP/dx={dP_dx/1e3:.2f} kPa/m")
        print(f"  Expected ΔP over {L}m: {delta_P_expected/1e3:.2f} kPa")
        print(f"  Expected P_outlet: {P_hot_in - delta_P_expected:.2f} Pa = {(P_hot_in - delta_P_expected)/1e3:.2f} kPa")
        
        if (P_hot_in - delta_P_expected) < config.optimization.delta_P_max_hot if config.optimization.delta_P_max_hot else 1e10:
            print(f"  ⚠️  Expected outlet pressure would be below minimum limit!")
        if delta_P_expected > 200e3:  # More than 200 kPa
            print(f"  ⚠️  Expected pressure drop ({delta_P_expected/1e3:.1f} kPa) > inlet pressure ({P_hot_in/1e3:.1f} kPa)!")
            print(f"     This would cause outlet to hit minimum limit regardless of properties.")
    
    # Test manual property calculations
    print("\n" + "="*60)
    print("MANUAL PROPERTY CALCULATION TEST")
    print("="*60)
    
    # Test if velocity should vary with porosity
    rho = 0.1786  # kg/m³
    A = config.geometry.width * config.geometry.height  # m²
    m_dot = config.fluid.m_dot_hot
    
    eps_low = rve_db.eps_hot(np.array([0.1]))[0]
    eps_high = rve_db.eps_hot(np.array([0.9]))[0]
    
    u_low = m_dot / (rho * A * eps_low)
    u_high = m_dot / (rho * A * eps_high)
    
    print(f"Porosity: ε_low={eps_low:.3f}, ε_high={eps_high:.3f}")
    print(f"Velocity (manual): u_low={u_low:.3f} m/s, u_high={u_high:.3f} m/s")
    print(f"Velocity ratio: {u_high/u_low:.2f}x")
    
    if abs(u_high - u_low) < 0.01:
        print("⚠️  Velocity doesn't vary much with porosity - may be issue")
    else:
        print("✅ Velocity varies significantly with porosity")
    
    # Test pressure drop calculation
    mu = 2.0e-5  # Pa·s
    kappa_low = rve_db.kappa_hot(np.array([0.1]))[0]
    kappa_high = rve_db.kappa_hot(np.array([0.9]))[0]
    beta_low = rve_db.beta_hot(np.array([0.1]))[0]
    beta_high = rve_db.beta_hot(np.array([0.9]))[0]
    
    # Darcy term: (mu/kappa) * u
    darcy_low = (mu / kappa_low) * u_low
    darcy_high = (mu / kappa_high) * u_high
    
    # Forchheimer term: beta * rho * u^2
    forch_low = beta_low * rho * u_low**2
    forch_high = beta_high * rho * u_high**2
    
    dP_dx_low = darcy_low + forch_low
    dP_dx_high = darcy_high + forch_high
    
    print(f"\nPressure drop per unit length (manual):")
    print(f"  dP/dx (d=0.1): {dP_dx_low/1e3:.2f} kPa/m")
    print(f"  dP/dx (d=0.9): {dP_dx_high/1e3:.2f} kPa/m")
    print(f"  Ratio: {dP_dx_low/dP_dx_high:.2f}x")
    
    if abs(dP_dx_low - dP_dx_high) < 100.0:
        print("⚠️  Pressure drop doesn't vary much - may indicate clamping or calculation issue")
    else:
        print("✅ Pressure drop varies significantly with properties")


if __name__ == "__main__":
    test_property_effects()
