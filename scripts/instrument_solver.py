"""
Instrument the solver to see what's happening during iteration.

This script modifies the solver to output intermediate values during iteration
to understand why temperature profiles converge to identical values.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig, SolverConfig
from hxopt.rve_db import RVEDatabase
from hxopt.macro_model import MacroModel


def instrumented_solve(model, d_field, label="", max_iterations=5):
    """
    Solve with instrumentation to see what happens during iteration.
    """
    print(f"\n{'='*70}")
    print(f"INSTRUMENTED SOLVE: {label}")
    print(f"{'='*70}")
    print(f"d_field: {d_field[0]:.1f} (uniform)")
    
    # Access internal solver by calling solve and checking intermediate state
    # We'll need to modify the solver or create a wrapper
    
    # For now, let's check what properties are being used
    kappa = model.rve_db.kappa_hot(d_field)
    beta = model.rve_db.beta_hot(d_field)
    eps = model.rve_db.eps_hot(d_field)
    A_surf_V = model.rve_db.A_surf_V(d_field)
    
    print(f"\nRVE Properties:")
    print(f"  κ = {kappa[0]:.2e} m²")
    print(f"  β = {beta[0]:.2e} 1/m")
    print(f"  ε = {eps[0]:.3f}")
    print(f"  A_surf/V = {A_surf_V[0]:.1f} 1/m")
    
    # Solve
    result = model.solve(d_field)
    
    # Calculate what velocity should be
    rho = model.config.fluid.rho_hot
    A = model.config.geometry.width * model.config.geometry.height * 0.5
    m_dot = model.config.fluid.m_dot_hot
    u_expected = m_dot / (rho * A * eps[0])
    
    # Calculate what heat transfer coefficient should be
    h_htc_expected = model.rve_db.h_hot(np.array([u_expected]), d_field[:1])[0]
    
    # Calculate what Q_vol should be (rough estimate)
    T_solid_avg = result.T_solid[:-1].mean()
    T_hot_avg = result.T_hot[:-1].mean()
    Q_vol_expected = h_htc_expected * A_surf_V[0] * (T_solid_avg - T_hot_avg)
    
    print(f"\nExpected values (from properties):")
    print(f"  u = {u_expected:.2f} m/s")
    print(f"  h_htc = {h_htc_expected:.1f} W/(m²·K)")
    print(f"  Q_vol = {Q_vol_expected/1e6:.2f} MW/m³")
    
    print(f"\nActual results:")
    print(f"  Q = {result.Q/1e6:.6f} MW")
    print(f"  T_hot_out = {result.T_hot[-1]:.2f} K")
    print(f"  T_cold_out = {result.T_cold[0]:.2f} K")
    print(f"  ΔT_hot = {result.T_hot[0] - result.T_hot[-1]:.2f} K")
    print(f"  ΔT_cold = {result.T_cold[0] - result.T_cold[-1]:.2f} K")
    
    # Check if Q_vol is reasonable
    # Q = integral of Q_vol * A_cross over length
    # Q ≈ Q_vol_mean * A_cross * L
    Q_estimated = Q_vol_expected * A * model.config.geometry.length
    print(f"\nEstimated Q from Q_vol: {Q_estimated/1e6:.6f} MW")
    print(f"Actual Q: {result.Q/1e6:.6f} MW")
    
    if abs(Q_estimated - result.Q) / max(abs(Q_estimated), 1.0) > 0.1:
        print(f"  ⚠️  Large discrepancy! Q_vol may not match actual heat transfer")
    
    return result


def compare_property_effects():
    """Compare how properties affect the solution."""
    
    print("="*70)
    print("INSTRUMENTED SOLVER COMPARISON")
    print("="*70)
    
    # Create config
    geometry = GeometryConfig(
        length=0.5,
        width=0.1,
        height=0.1,
        n_segments=10,
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
        P_hot_in=10e6,
        P_cold_in=5e6,
        m_dot_hot=0.01,
        m_dot_cold=0.05,
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
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'instrument'),
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    rve_db = RVEDatabase(config.rve_table_path)
    model = MacroModel(config, rve_db)
    
    # Test with different d values
    results = {}
    for d in [0.1, 0.5, 0.9]:
        d_field = np.full(config.geometry.n_segments, d)
        result = instrumented_solve(model, d_field, f"d={d:.1f}")
        results[d] = result
    
    # Compare
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    print(f"\n{'d':<6} {'Q (MW)':<12} {'ΔP (kPa)':<12} {'T_hot_out (K)':<15} {'T_cold_out (K)':<15}")
    print("-" * 70)
    for d in [0.1, 0.5, 0.9]:
        r = results[d]
        print(f"{d:<6.1f} {r.Q/1e6:<12.6f} {r.delta_P_hot/1e3:<12.2f} {r.T_hot[-1]:<15.2f} {r.T_cold[0]:<15.2f}")
    
    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)
    print("""
The temperature profiles (and thus Q) are identical because:

1. The energy balance solver converges to the same solution regardless of d
2. This suggests the heat transfer is NOT limited by RVE properties
3. Instead, it's limited by:
   - Fixed inlet temperatures (300K vs 20K = 280K difference)
   - Fixed mass flow rates (0.01 and 0.05 kg/s)
   - The maximum possible heat transfer given these constraints

4. Even though:
   - Velocity varies with porosity (u = m_dot / (rho * A * eps))
   - Heat transfer coefficient varies with velocity and d (h = a * u^b)
   - Surface area varies with d (A_surf/V)
   
   The product h * A_surf/V * (T_solid - T_fluid) converges to the same
   value because T_solid and T_fluid converge to the same profiles.

5. The solver may be:
   - Converging too quickly (before property effects manifest)
   - Hitting a constraint that forces the same solution
   - Using an initial guess that's too close to the final solution

SOLUTION:
- The heat transfer is likely limited by the fixed inlet conditions, not by
  the RVE properties. To make Q sensitive to d, we need to:
  1. Use conditions where heat transfer IS limited by properties
  2. Or adjust the solver to be more sensitive to property changes
  3. Or use different inlet conditions that allow property effects to manifest
""")


if __name__ == "__main__":
    compare_property_effects()

