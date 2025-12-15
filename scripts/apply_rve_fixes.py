#!/usr/bin/env python3
"""
Apply RVE Property Fixes - Implements recommendations from RVE_PROPERTY_FIXES.md

This script demonstrates the recommended fixes for addressing very large pressure drops:
1. Reduce mass flow rate (e.g., 0.001 kg/s instead of 0.01 kg/s)
2. Increase channel cross-sectional area
3. Recalibrate RVE properties for actual test geometry/flow conditions
4. Use properties from Cheung et al. (2025) if available
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig
from hxopt.rve_db import RVEDatabase
from hxopt.macro_model import MacroModel

def test_with_recommended_fixes():
    """Test with all recommended fixes applied."""
    
    print("=" * 70)
    print("Applying RVE Property Fixes (from RVE_PROPERTY_FIXES.md)")
    print("=" * 70)
    print()
    
    # Fix 1: Reduce mass flow rate (10x reduction)
    print("Fix 1: Reduced mass flow rates")
    print("  - m_dot_hot: 0.01 → 0.001 kg/s (10x reduction)")
    print("  - m_dot_cold: 0.05 → 0.005 kg/s (10x reduction)")
    print()
    
    # Fix 2: Increase channel cross-sectional area
    print("Fix 2: Increased channel cross-sectional area")
    print("  - Width: 0.1 → 0.2 m (2x)")
    print("  - Height: 0.1 → 0.2 m (2x)")
    print("  - Cross-sectional area: 0.005 → 0.02 m² (4x increase)")
    print()
    
    # Fix 3 & 4: Use Cheung et al. (2025) calibrated properties
    print("Fix 3 & 4: Using Cheung et al. (2025) calibrated RVE properties")
    rve_path = "data/rve_tables/cheung_2025_calibrated.csv"
    if not os.path.exists(rve_path):
        # Fall back to calibrated primitive
        rve_path = "data/rve_tables/primitive_calibrated.csv"
        print(f"  - Using: {rve_path} (Cheung not available)")
    else:
        print(f"  - Using: {rve_path}")
    print()
    
    # Create configuration with fixes
    geometry = GeometryConfig(
        length=0.5,      # m
        width=0.2,       # m (increased from 0.1)
        height=0.2,      # m (increased from 0.1)
        n_segments=50,
    )
    
    fluid = FluidConfig(
        T_hot_in=300.0,   # K
        T_cold_in=20.0,   # K
        P_hot_in=10e6,    # Pa (10 MPa - already increased)
        P_cold_in=5e6,    # Pa (5 MPa - already increased)
        m_dot_hot=0.001,  # kg/s (reduced from 0.01)
        m_dot_cold=0.005, # kg/s (reduced from 0.05)
        use_real_properties=True,
    )
    
    optimization = OptimizationConfig()
    
    config = Config(
        geometry=geometry,
        fluid=fluid,
        optimization=optimization,
        rve_table_path=rve_path,
    )
    
    # Load RVE database
    if not os.path.exists(rve_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        rve_path = os.path.join(project_root, rve_path)
    
    rve_db = RVEDatabase(rve_path)
    
    # Test with different d values
    print("Testing with different d values:")
    print("-" * 70)
    
    d_values = [0.1, 0.5, 0.9]
    results = []
    
    for d in d_values:
        d_field = np.full(config.geometry.n_segments, d)
        model = MacroModel(config, rve_db)
        result = model.solve(d_field)
        results.append((d, result))
        
        # Calculate cross-sectional area
        A = config.geometry.width * config.geometry.height * 0.5
        
        # Estimate velocity
        rho = 0.18  # kg/m³ (helium at ~40K)
        eps = rve_db.eps_hot(np.array([d]))[0]  # Get porosity for this d value
        u = fluid.m_dot_hot / (rho * A * eps)
        
        print(f"\nd = {d:.1f}:")
        print(f"  Porosity (ε): {eps:.3f}")
        print(f"  Estimated velocity: {u:.3f} m/s")
        print(f"  Pressure drop (hot): {result.delta_P_hot/1e3:.2f} kPa")
        print(f"  Pressure drop (cold): {result.delta_P_cold/1e3:.2f} kPa")
        print(f"  Heat transfer (Q): {result.Q/1e6:.6f} MW")
        print(f"  T_hot_out: {result.T_hot[-1]:.2f} K")
        print(f"  T_cold_out: {result.T_cold[0]:.2f} K")
    
    print()
    print("=" * 70)
    print("Comparison:")
    print("=" * 70)
    
    d1, r1 = results[0]
    d9, r9 = results[-1]
    
    print(f"\nPressure drop variation:")
    print(f"  d={d1:.1f}: {r1.delta_P_hot/1e3:.2f} kPa")
    print(f"  d={d9:.1f}: {r9.delta_P_hot/1e3:.2f} kPa")
    print(f"  Variation: {abs(r9.delta_P_hot - r1.delta_P_hot)/1e3:.2f} kPa")
    
    print(f"\nHeat transfer variation:")
    print(f"  d={d1:.1f}: {r1.Q/1e6:.6f} MW")
    print(f"  d={d9:.1f}: {r9.Q/1e6:.6f} MW")
    print(f"  Variation: {abs(r9.Q - r1.Q)/1e6:.6f} MW")
    print(f"  Relative variation: {abs(r9.Q - r1.Q)/r1.Q*100:.2f}%")
    
    # Check if fixes helped
    print()
    print("=" * 70)
    print("Fix Effectiveness:")
    print("=" * 70)
    
    # Check pressure drops are reasonable
    max_dP = max(r.delta_P_hot for _, r in results) / 1e6  # MPa
    if max_dP < 1.0:  # Less than 1 MPa
        print("✅ Pressure drops are reasonable (< 1 MPa)")
    elif max_dP < 10.0:  # Less than 10 MPa
        print(f"⚠️  Pressure drops still high ({max_dP:.1f} MPa) but manageable")
    else:
        print(f"❌ Pressure drops still very high ({max_dP:.1f} MPa)")
    
    # Check Q variation
    q_variation = abs(r9.Q - r1.Q) / r1.Q * 100
    if q_variation > 1.0:
        print(f"✅ Heat transfer varies significantly ({q_variation:.2f}%)")
    elif q_variation > 0.1:
        print(f"⚠️  Heat transfer varies slightly ({q_variation:.2f}%)")
    else:
        print(f"❌ Heat transfer still insensitive ({q_variation:.4f}%)")
    
    print()
    print("=" * 70)
    print("Summary of Applied Fixes:")
    print("=" * 70)
    print("1. ✅ Reduced mass flow rate: 0.01 → 0.001 kg/s (10x)")
    print("2. ✅ Increased cross-sectional area: 0.005 → 0.02 m² (4x)")
    print("3. ✅ Using calibrated RVE properties")
    print("4. ✅ Using Cheung et al. (2025) properties (if available)")
    print()
    
    return results

if __name__ == "__main__":
    try:
        results = test_with_recommended_fixes()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
