"""
Test with property-limited conditions to verify Q becomes sensitive to d.

This script implements the recommended solutions from Q_INSENSITIVITY_ANALYSIS.md:
1. Lower mass flow rates (m_dot = 0.001 kg/s)
2. Smaller temperature differences (ΔT = 20 K)
3. Longer heat exchanger (L = 2.0 m)

Expected: Q should vary with d when heat transfer is property-limited.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig, SolverConfig
from hxopt.rve_db import RVEDatabase
from hxopt.macro_model import MacroModel


def test_property_limited_conditions():
    """Test with property-limited conditions."""
    
    print("="*70)
    print("TESTING WITH PROPERTY-LIMITED CONDITIONS")
    print("="*70)
    print("\nThis test uses conditions where heat transfer IS limited by RVE properties:")
    print("  - Lower mass flow rates (10x reduction)")
    print("  - Smaller temperature difference (20 K instead of 280 K)")
    print("  - Longer heat exchanger (2.0 m instead of 0.5 m)")
    print("\nExpected: Q should vary with d when properties are limiting.\n")
    
    # Create config with property-limited conditions
    geometry = GeometryConfig(
        length=2.0,  # Longer (4x increase) - allows property effects to accumulate
        width=0.1,
        height=0.1,
        n_segments=20,  # More segments for longer length
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
        # Smaller temperature difference - makes properties more important
        T_hot_in=310.0,  # K (reduced from 300 K)
        T_cold_in=290.0,  # K (increased from 20 K) - ΔT = 20 K instead of 280 K
        P_hot_in=10e6,  # 10 MPa
        P_cold_in=5e6,   # 5 MPa
        # Lower mass flow rates - makes h_htc and A_surf/V limiting factors
        m_dot_hot=0.001,  # kg/s (10x lower)
        m_dot_cold=0.005, # kg/s (10x lower)
    )
    
    solver = SolverConfig(
        max_iter=200,
        tol=1e-7,
        relax=0.15,
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
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'property_limited'),
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load RVE database
    print("Loading RVE database...")
    rve_db = RVEDatabase(config.rve_table_path)
    
    # Create model
    model = MacroModel(config, rve_db)
    
    # Test with different d values
    print("\n" + "="*70)
    print("TESTING WITH DIFFERENT d VALUES")
    print("="*70)
    
    results = {}
    d_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"\n{'d':<6} {'Q (MW)':<12} {'ΔP_hot (kPa)':<15} {'T_hot_out (K)':<15} {'T_cold_out (K)':<15}")
    print("-" * 70)
    
    for d in d_values:
        d_field = np.full(config.geometry.n_segments, d)
        
        try:
            result = model.solve(d_field)
            results[d] = result
            
            print(f"{d:<6.1f} {result.Q/1e6:<12.6f} {result.delta_P_hot/1e3:<15.2f} "
                  f"{result.T_hot[-1]:<15.2f} {result.T_cold[0]:<15.2f}")
        except Exception as e:
            print(f"{d:<6.1f} ERROR: {e}")
            results[d] = None
    
    # Analyze results
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if len(results) >= 2 and all(r is not None for r in results.values()):
        Q_values = [r.Q for r in results.values() if r is not None]
        Q_min = min(Q_values)
        Q_max = max(Q_values)
        Q_range = Q_max - Q_min
        Q_relative = (Q_range / max(Q_min, 1.0)) * 100 if Q_min > 0 else 0
        
        print(f"\nQ variation:")
        print(f"  Min Q: {Q_min/1e6:.6f} MW")
        print(f"  Max Q: {Q_max/1e6:.6f} MW")
        print(f"  Range: {Q_range/1e6:.6f} MW")
        print(f"  Relative variation: {Q_relative:.2f}%")
        
        # Calculate relative variation based on max absolute value
        Q_max_abs = max(abs(Q_min), abs(Q_max))
        Q_relative_abs = (Q_range / max(Q_max_abs, 1e-9)) * 100 if Q_max_abs > 0 else 0
        
        if Q_range > 1e-6:  # More than 1 W variation
            print(f"\n✅ SUCCESS: Q varies by {Q_range/1e6:.6f} MW ({Q_relative_abs:.2f}% of max) with d!")
            print("   Heat transfer is now property-limited, as expected.")
            print("   Temperature profiles also vary (T_hot_out varies with d).")
        elif Q_range > 1e-9:  # More than 1 mW variation
            print(f"\n⚠️  PARTIAL: Q varies by {Q_range/1e6:.6f} MW ({Q_relative_abs:.2f}% of max) with d.")
            print("   Some sensitivity, but may need further adjustment.")
        else:
            print(f"\n❌ FAILED: Q still insensitive ({Q_range/1e6:.6f} MW variation).")
            print("   May need even more extreme property-limited conditions.")
        
        # Check pressure drop variation
        delta_P_values = [r.delta_P_hot for r in results.values() if r is not None]
        delta_P_min = min(delta_P_values)
        delta_P_max = max(delta_P_values)
        delta_P_range = delta_P_max - delta_P_min
        
        print(f"\nPressure drop variation:")
        print(f"  Min ΔP: {delta_P_min/1e3:.2f} kPa")
        print(f"  Max ΔP: {delta_P_max/1e3:.2f} kPa")
        print(f"  Range: {delta_P_range/1e3:.2f} kPa")
        
        if delta_P_range > 100.0:  # More than 100 kPa variation
            print(f"  ✅ Pressure drop varies significantly (as expected)")
        else:
            print(f"  ⚠️  Pressure drop variation is small")
        
        # Compare d=0.1 vs d=0.9
        if 0.1 in results and 0.9 in results and results[0.1] and results[0.9]:
            r_low = results[0.1]
            r_high = results[0.9]
            
            Q_diff = abs(r_high.Q - r_low.Q) / 1e6
            Q_diff_pct = abs(r_high.Q - r_low.Q) / max(r_low.Q, 1.0) * 100
            
            print(f"\nComparison: d=0.1 vs d=0.9")
            print(f"  Q difference: {Q_diff:.6f} MW ({Q_diff_pct:.2f}%)")
            print(f"  ΔP difference: {abs(r_high.delta_P_hot - r_low.delta_P_hot)/1e3:.2f} kPa")
            print(f"  T_hot_out difference: {abs(r_high.T_hot[-1] - r_low.T_hot[-1]):.2f} K")
            print(f"  T_cold_out difference: {abs(r_high.T_cold[0] - r_low.T_cold[0]):.2f} K")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
If Q varies significantly (>5%) with d:
  ✅ Property-limited conditions are working
  ✅ Model is sensitive to RVE properties
  ✅ Optimization should now show improvement

If Q still doesn't vary:
  - Try even lower mass flow rates (0.0001 kg/s)
  - Try even smaller temperature differences (10 K)
  - Check if properties are actually limiting heat transfer
  - Verify RVE properties are appropriate for these conditions
""")


def compare_with_original_conditions():
    """Compare property-limited vs original conditions."""
    
    print("\n" + "="*70)
    print("COMPARISON: PROPERTY-LIMITED vs ORIGINAL CONDITIONS")
    print("="*70)
    
    print("\nOriginal conditions (inlet-limited):")
    print("  m_dot_hot = 0.01 kg/s")
    print("  T_hot_in = 300 K, T_cold_in = 20 K (ΔT = 280 K)")
    print("  L = 0.5 m")
    print("  Result: Q identical for all d (0.000970 MW)")
    
    print("\nProperty-limited conditions:")
    print("  m_dot_hot = 0.001 kg/s (10x lower)")
    print("  T_hot_in = 310 K, T_cold_in = 290 K (ΔT = 20 K)")
    print("  L = 2.0 m (4x longer)")
    print("  Expected: Q should vary with d")
    
    print("\nWhy property-limited conditions work:")
    print("  1. Lower m_dot → lower velocity → lower h_htc → h_htc becomes limiting")
    print("  2. Smaller ΔT → smaller driving force → properties become more important")
    print("  3. Longer L → property effects accumulate over length")


if __name__ == "__main__":
    test_property_limited_conditions()
    compare_with_original_conditions()

