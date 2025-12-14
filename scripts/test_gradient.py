"""
Test script to diagnose gradient computation issues.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig
from hxopt.rve_db import RVEDatabase
from hxopt.macro_model import MacroModel
from hxopt.objective import compute_objective_gradient, compute_objective


def main():
    """Test gradient computation."""
    
    print("="*60)
    print("GRADIENT COMPUTATION DIAGNOSTIC")
    print("="*60)
    
    # Create minimal config
    geometry = GeometryConfig(
        length=0.5,
        width=0.1,
        height=0.1,
        n_segments=10,  # Small for testing
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
        rve_table_path=os.path.join(
            os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_calibrated.csv'
        ),
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'test'),
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load RVE database
    print("\nLoading RVE database...")
    rve_db = RVEDatabase(config.rve_table_path)
    print(f"RVE database loaded: d range [{rve_db.d_min:.2f}, {rve_db.d_max:.2f}]")
    
    # Test RVE properties at different d values
    print("\nTesting RVE properties vs d:")
    for d_test in [0.1, 0.3, 0.5, 0.7, 0.9]:
        kappa = rve_db.kappa_hot(np.array([d_test]))[0]
        beta = rve_db.beta_hot(np.array([d_test]))[0]
        eps = rve_db.eps_hot(np.array([d_test]))[0]
        print(f"  d={d_test:.1f}: κ={kappa:.2e} m², β={beta:.2e} 1/m, ε={eps:.3f}")
    
    # Create model
    model = MacroModel(config, rve_db)
    
    # Test with uniform d fields
    print("\n" + "="*60)
    print("Testing Q vs d (uniform fields)")
    print("="*60)
    
    d_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    Q_values = []
    
    for d_val in d_values:
        d_field = np.full(config.geometry.n_segments, d_val)
        try:
            # Get RVE properties for this d
            kappa = rve_db.kappa_hot(np.array([d_val]))[0]
            beta = rve_db.beta_hot(np.array([d_val]))[0]
            eps = rve_db.eps_hot(np.array([d_val]))[0]
            A_surf_V = rve_db.A_surf_V(np.array([d_val]))[0]
            
            result = model.solve(d_field)
            Q = result.Q
            Q_values.append(Q)
            
            # Check enthalpy difference
            h_diff = result.h_hot_in - result.h_hot_out if hasattr(result, 'h_hot_in') else 0.0
            
            print(f"  d={d_val:.1f} (uniform):")
            print(f"    Q={Q/1e6:.6f} MW, ΔP_hot={result.delta_P_hot/1e3:.2f} kPa")
            print(f"    κ={kappa:.2e} m², β={beta:.2e} 1/m, ε={eps:.3f}, A_surf/V={A_surf_V:.1f} 1/m")
            print(f"    T_hot_out={result.T_hot[-1]:.2f} K, T_cold_out={result.T_cold[0]:.2f} K")
        except Exception as e:
            print(f"  d={d_val:.1f} (uniform): Solve failed: {e}")
            Q_values.append(None)
    
    # Check if Q varies with d
    valid_Q = [q for q in Q_values if q is not None]
    if len(valid_Q) > 1:
        Q_range = max(valid_Q) - min(valid_Q)
        Q_rel_change = Q_range / max(valid_Q) if max(valid_Q) > 0 else 0
        print(f"\nQ variation: range={Q_range:.2e} W, relative={Q_rel_change*100:.2f}%")
        if Q_rel_change < 0.01:
            print("  ⚠️  WARNING: Q varies less than 1% with d. Model may be insensitive to d.")
        else:
            print("  ✅ Q varies significantly with d. Gradient should be computable.")
    
    # Test gradient computation
    print("\n" + "="*60)
    print("Testing gradient computation")
    print("="*60)
    
    d_field = np.full(config.geometry.n_segments, 0.5)
    result = model.solve(d_field)
    Q_base = result.Q
    print(f"Base case: d=0.5 (uniform), Q={Q_base/1e6:.6f} MW")
    
    try:
        grad = compute_objective_gradient(model, d_field, result)
        grad_norm = np.linalg.norm(grad)
        grad_max = np.max(np.abs(grad))
        grad_mean = np.mean(np.abs(grad))
        
        print(f"\nGradient computed:")
        print(f"  norm={grad_norm:.2e}")
        print(f"  max={grad_max:.2e}")
        print(f"  mean={grad_mean:.2e}")
        print(f"  First 5 components: {grad[:5]}")
        
        if grad_norm < 1e-10:
            print("\n  ⚠️  Gradient is essentially zero!")
            print("  This suggests the model is not sensitive to d changes.")
        else:
            print("\n  ✅ Gradient is non-zero. Optimization should work.")
    except Exception as e:
        print(f"\n  ❌ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Diagnostic complete")
    print("="*60)


if __name__ == "__main__":
    main()
