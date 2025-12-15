"""
Investigate why Q (heat transfer) remains insensitive despite pressure drop variation.

This script examines:
1. Temperature profiles for different d values
2. Velocity profiles (should vary with porosity)
3. Heat transfer coefficient profiles (should vary with velocity and d)
4. Volumetric heat transfer Q_vol profiles
5. Enthalpy profiles
6. Why Q = m_dot * (h_in - h_out) doesn't vary
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig, SolverConfig
from hxopt.rve_db import RVEDatabase
from hxopt.macro_model import MacroModel


def investigate_q_insensitivity():
    """Investigate why Q doesn't vary with d."""
    
    print("="*70)
    print("INVESTIGATING Q INSENSITIVITY")
    print("="*70)
    
    # Create config with calibrated RVE and high pressures
    geometry = GeometryConfig(
        length=0.5,
        width=0.1,
        height=0.1,
        n_segments=10,  # Small for detailed inspection
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
        P_hot_in=10e6,  # 10 MPa
        P_cold_in=5e6,  # 5 MPa
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
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'investigate'),
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load RVE database
    print("\nLoading RVE database...")
    rve_db = RVEDatabase(config.rve_table_path)
    
    # Create model
    model = MacroModel(config, rve_db)
    
    # Test with extreme d values
    print("\n" + "="*70)
    print("TESTING WITH EXTREME d VALUES")
    print("="*70)
    
    results = {}
    
    for d_test in [0.1, 0.9]:
        d_field = np.full(config.geometry.n_segments, d_test)
        
        print(f"\n{'='*70}")
        print(f"d = {d_test:.1f}")
        print(f"{'='*70}")
        
        # Get RVE properties
        kappa = rve_db.kappa_hot(d_field)
        beta = rve_db.beta_hot(d_field)
        eps = rve_db.eps_hot(d_field)
        A_surf_V = rve_db.A_surf_V(d_field)
        
        print(f"\nRVE Properties (mean):")
        print(f"  κ = {kappa.mean():.2e} m²")
        print(f"  β = {beta.mean():.2e} 1/m")
        print(f"  ε = {eps.mean():.3f}")
        print(f"  A_surf/V = {A_surf_V.mean():.1f} 1/m")
        
        # Solve
        result = model.solve(d_field)
        results[d_test] = result
        
        print(f"\nSolution:")
        print(f"  Q = {result.Q/1e6:.6f} MW")
        print(f"  ΔP_hot = {result.delta_P_hot/1e3:.2f} kPa")
        print(f"  T_hot_out = {result.T_hot[-1]:.2f} K")
        print(f"  T_cold_out = {result.T_cold[0]:.2f} K")
        print(f"  h_hot_out = {result.h_hot_out/1e3:.2f} kJ/kg")
        print(f"  h_cold_out = {result.h_cold_out/1e3:.2f} kJ/kg")
        
        # Calculate Q from enthalpy difference
        cp_hot = fluid.cp_hot
        cp_cold = fluid.cp_cold
        h_hot_in = cp_hot * fluid.T_hot_in
        h_cold_in = cp_cold * fluid.T_cold_in
        
        Q_from_enthalpy = fluid.m_dot_hot * (h_hot_in - result.h_hot_out)
        Q_from_temp = fluid.m_dot_hot * cp_hot * (fluid.T_hot_in - result.T_hot[-1])
        
        print(f"\nQ calculation:")
        print(f"  Q = m_dot * (h_in - h_out) = {Q_from_enthalpy/1e6:.6f} MW")
        print(f"  Q = m_dot * cp * (T_in - T_out) = {Q_from_temp/1e6:.6f} MW")
        print(f"  Q (from result) = {result.Q/1e6:.6f} MW")
        
        # Check temperature profiles
        print(f"\nTemperature Profiles:")
        print(f"  T_hot: [{result.T_hot[0]:.2f}, ..., {result.T_hot[-1]:.2f}] K")
        print(f"  T_cold: [{result.T_cold[0]:.2f}, ..., {result.T_cold[-1]:.2f}] K")
        print(f"  T_solid: [{result.T_solid[0]:.2f}, ..., {result.T_solid[-1]:.2f}] K")
        print(f"  ΔT_hot = {result.T_hot[0] - result.T_hot[-1]:.2f} K")
        print(f"  ΔT_cold = {result.T_cold[0] - result.T_cold[-1]:.2f} K")
        
        # Estimate velocity from properties
        rho = fluid.rho_hot
        A = config.geometry.width * config.geometry.height * 0.5
        u_est = fluid.m_dot_hot / (rho * A * eps.mean())
        print(f"\nEstimated velocity: u = {u_est:.2f} m/s (from m_dot, rho, A, eps)")
        
        # Estimate heat transfer coefficient
        h_htc_est = rve_db.h_hot(np.array([u_est]), np.array([d_test]))[0]
        print(f"Estimated h_htc: h = {h_htc_est:.1f} W/(m²·K)")
        
        # Estimate volumetric heat transfer (rough)
        T_solid_avg = result.T_solid[:-1].mean()
        T_hot_avg = result.T_hot[:-1].mean()
        Q_vol_est = h_htc_est * A_surf_V.mean() * (T_solid_avg - T_hot_avg)
        print(f"Estimated Q_vol: {Q_vol_est/1e6:.2f} MW/m³")
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON: d=0.1 vs d=0.9")
    print("="*70)
    
    r1 = results[0.1]
    r9 = results[0.9]
    
    print(f"\nQ difference: {abs(r9.Q - r1.Q)/1e6:.6f} MW")
    print(f"Q relative difference: {abs(r9.Q - r1.Q)/max(r1.Q, 1.0)*100:.2f}%")
    print(f"ΔP_hot difference: {abs(r9.delta_P_hot - r1.delta_P_hot)/1e3:.2f} kPa")
    print(f"T_hot_out difference: {abs(r9.T_hot[-1] - r1.T_hot[-1]):.2f} K")
    print(f"T_cold_out difference: {abs(r9.T_cold[0] - r1.T_cold[0]):.2f} K")
    print(f"h_hot_out difference: {abs(r9.h_hot_out - r1.h_hot_out)/1e3:.2f} kJ/kg")
    
    # Analyze why Q doesn't vary
    print("\n" + "="*70)
    print("ANALYSIS: WHY Q DOESN'T VARY")
    print("="*70)
    
    print("\nQ = m_dot * (h_in - h_out)")
    print(f"  For d=0.1: Q = {fluid.m_dot_hot:.3f} * ({cp_hot*fluid.T_hot_in/1e3:.2f} - {r1.h_hot_out/1e3:.2f}) = {r1.Q/1e6:.6f} MW")
    print(f"  For d=0.9: Q = {fluid.m_dot_hot:.3f} * ({cp_hot*fluid.T_hot_in/1e3:.2f} - {r9.h_hot_out/1e3:.2f}) = {r9.Q/1e6:.6f} MW")
    print(f"\n  h_out difference: {abs(r9.h_hot_out - r1.h_hot_out)/1e3:.4f} kJ/kg")
    print(f"  This would cause Q difference: {abs(r9.h_hot_out - r1.h_hot_out) * fluid.m_dot_hot / 1e6:.6f} MW")
    
    if abs(r9.h_hot_out - r1.h_hot_out) < 1.0:  # Less than 1 J/kg
        print("\n  ⚠️  CRITICAL: h_out is essentially identical!")
        print("     This means the enthalpy change (h_in - h_out) is the same,")
        print("     which means Q = m_dot * (h_in - h_out) is the same.")
        print("\n  Root cause: Temperature profiles are identical,")
        print("     so enthalpy profiles are identical,")
        print("     so Q is identical.")
    
    print("\n" + "="*70)
    print("HYPOTHESIS")
    print("="*70)
    print("""
The energy balance solver is converging to the same temperature profiles
regardless of d because:

1. The heat transfer is dominated by:
   - Fixed inlet temperatures (300K hot, 20K cold)
   - Fixed mass flow rates (0.01 and 0.05 kg/s)
   - Not by RVE properties (h_htc, A_surf/V)

2. The volumetric heat transfer Q_vol = h * A_surf/V * (T_solid - T_fluid)
   may be converging to the same value because:
   - T_solid and T_fluid converge to the same profiles
   - Even though h and A_surf/V vary with d, the product converges to same Q_vol

3. The solver may be:
   - Converging too quickly before property effects manifest
   - Hitting constraints that force same solution
   - Using initial guess that's too close to final solution

NEXT STEPS:
1. Check if Q_vol profiles vary with d (need to instrument solver)
2. Check if temperature profiles actually vary (they appear identical)
3. Test with different inlet conditions to see if Q becomes sensitive
4. Check if heat transfer is limited by mass flow rate, not properties
""")


if __name__ == "__main__":
    investigate_q_insensitivity()

