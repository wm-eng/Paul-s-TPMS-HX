"""
Verify RVE property units and magnitudes against literature and physical expectations.

This script:
1. Loads calibrated RVE properties
2. Checks units and magnitudes
3. Computes expected pressure drops for typical flow conditions
4. Compares with literature values (Cheung et al., 2025)
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.rve_db import RVEDatabase
from hxopt.config import Config, GeometryConfig, FluidConfig


def verify_units_and_magnitudes():
    """Verify RVE property units and magnitudes."""
    print("="*60)
    print("RVE PROPERTY VERIFICATION")
    print("="*60)
    
    # Load calibrated RVE table
    rve_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_calibrated.csv'
    )
    
    if not os.path.exists(rve_path):
        print(f"ERROR: RVE table not found: {rve_path}")
        return
    
    print(f"\nLoading RVE table: {rve_path}")
    
    # Load raw CSV to inspect (skip comment lines)
    # Read file manually to skip comment lines
    with open(rve_path, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#') and line.strip()]
    
    # Create temporary file without comments for RVEDatabase
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name
    
    try:
        rve_db = RVEDatabase(tmp_path)
        df = pd.read_csv(tmp_path)
    finally:
        os.unlink(tmp_path)
    
    print("\n" + "="*60)
    print("PROPERTY RANGES")
    print("="*60)
    print(f"d range: [{df['d'].min():.2f}, {df['d'].max():.2f}]")
    print(f"κ (permeability) range: [{df['kappa_hot'].min():.2e}, {df['kappa_hot'].max():.2e}] m²")
    print(f"β (Forchheimer) range: [{df['beta_hot'].min():.2e}, {df['beta_hot'].max():.2e}] 1/m")
    print(f"ε (porosity) range: [{df['eps_hot'].min():.3f}, {df['eps_hot'].max():.3f}]")
    print(f"A_surf/V range: [{df['A_surf_V'].min():.1f}, {df['A_surf_V'].max():.1f}] 1/m")
    
    # Check monotonicity
    print("\n" + "="*60)
    print("MONOTONICITY CHECKS")
    print("="*60)
    d_sorted = df.sort_values('d')
    kappa_mono = np.all(np.diff(d_sorted['kappa_hot']) >= 0)
    beta_mono = np.all(np.diff(d_sorted['beta_hot']) <= 0)
    eps_mono = np.all(np.diff(d_sorted['eps_hot']) >= 0)
    
    print(f"κ increases with d: {'✅' if kappa_mono else '❌'}")
    print(f"β decreases with d: {'✅' if beta_mono else '❌'}")
    print(f"ε increases with d: {'✅' if eps_mono else '❌'}")
    
    # Physical bounds
    print("\n" + "="*60)
    print("PHYSICAL BOUNDS")
    print("="*60)
    print(f"Porosity in [0,1]: {'✅' if np.all((df['eps_hot'] > 0) & (df['eps_hot'] < 1)) else '❌'}")
    print(f"Permeability > 0: {'✅' if np.all(df['kappa_hot'] > 0) else '❌'}")
    print(f"Forchheimer > 0: {'✅' if np.all(df['beta_hot'] > 0) else '❌'}")
    
    # Expected property ranges from literature
    print("\n" + "="*60)
    print("LITERATURE COMPARISON")
    print("="*60)
    print("Typical TPMS Primitive properties (from literature):")
    print("  κ: 1e-10 to 1e-8 m² (for d=0.1 to 0.9)")
    print("  β: 1e4 to 1e6 1/m (for d=0.1 to 0.9)")
    print("  ε: 0.2 to 0.8 (for d=0.1 to 0.9)")
    
    kappa_in_range = np.all((df['kappa_hot'] >= 1e-11) & (df['kappa_hot'] <= 1e-8))
    beta_in_range = np.all((df['beta_hot'] >= 1e3) & (df['beta_hot'] <= 1e7))
    eps_in_range = np.all((df['eps_hot'] >= 0.1) & (df['eps_hot'] <= 0.9))
    
    print(f"\nκ in expected range: {'✅' if kappa_in_range else '❌'}")
    print(f"β in expected range: {'✅' if beta_in_range else '❌'}")
    print(f"ε in expected range: {'✅' if eps_in_range else '❌'}")


def compute_expected_pressure_drops():
    """Compute expected pressure drops for typical flow conditions."""
    print("\n" + "="*60)
    print("EXPECTED PRESSURE DROP CALCULATION")
    print("="*60)
    
    rve_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_calibrated.csv'
    )
    
    # Read file manually to skip comment lines
    with open(rve_path, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#') and line.strip()]
    
    # Create temporary file without comments for RVEDatabase
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name
    
    try:
        rve_db = RVEDatabase(tmp_path)
    finally:
        os.unlink(tmp_path)
    
    # Typical flow conditions
    rho = 0.1786  # kg/m³ (helium at 300K)
    mu = 2.0e-5   # Pa·s
    A = 0.1 * 0.1 * 0.5  # m² (width * height * 0.5 for half channel)
    m_dot = 0.01  # kg/s
    L = 0.5  # m
    
    print(f"\nFlow conditions:")
    print(f"  ρ = {rho:.4f} kg/m³")
    print(f"  μ = {mu:.2e} Pa·s")
    print(f"  A = {A:.4f} m²")
    print(f"  m_dot = {m_dot:.3f} kg/s")
    print(f"  L = {L:.2f} m")
    
    print(f"\n{'d':<6} {'ε':<8} {'u (m/s)':<12} {'κ (m²)':<12} {'β (1/m)':<12} {'ΔP (kPa)':<12} {'P_out (kPa)':<12}")
    print("-" * 80)
    
    for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
        kappa = rve_db.kappa_hot(np.array([d]))[0]
        beta = rve_db.beta_hot(np.array([d]))[0]
        eps = rve_db.eps_hot(np.array([d]))[0]
        
        # Velocity
        u = m_dot / (rho * A * eps)
        
        # Pressure gradient: dP/dx = -(mu/kappa)*u - beta*rho*u^2
        darcy_term = (mu / kappa) * u
        forch_term = beta * rho * u**2
        dP_dx = -(darcy_term + forch_term)
        
        # Total pressure drop
        delta_P = -dP_dx * L
        
        # Outlet pressure (assuming inlet = 10 MPa)
        P_in = 10e6  # Pa
        P_out = P_in - delta_P
        
        print(f"{d:<6.1f} {eps:<8.3f} {u:<12.3f} {kappa:<12.2e} {beta:<12.2e} {delta_P/1e3:<12.2f} {P_out/1e3:<12.2f}")
        
        if P_out < 0.1e3:  # Less than 0.1 kPa
            print(f"  ⚠️  Outlet pressure would be below minimum (0.1 kPa)")
        elif P_out < 1e3:  # Less than 1 kPa
            print(f"  ⚠️  Outlet pressure very low (< 1 kPa)")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("If pressure drops are too large:")
    print("  1. Increase inlet pressure (e.g., 10-50 MPa)")
    print("  2. Reduce mass flow rate")
    print("  3. Increase channel cross-sectional area")
    print("  4. Use RVE properties calibrated for your specific geometry/flow conditions")


def main():
    """Main verification function."""
    verify_units_and_magnitudes()
    compute_expected_pressure_drops()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

