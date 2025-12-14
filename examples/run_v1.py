"""Example v1 optimization run."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig
from hxopt.rve_db import RVEDatabase
from hxopt.optimize_mma import optimize
from hxopt.export_geometry import export_field_csv, export_vtk


def main():
    """Run v1 optimization example."""
    
    # Geometry
    geometry = GeometryConfig(
        length=0.5,  # m
        width=0.1,  # m
        height=0.1,  # m
        n_segments=50,
    )
    
    # Fluid properties (constant for v1)
    # Hot side: Helium
    fluid = FluidConfig(
        rho_hot=0.1786,  # kg/m³ at 300K, 1 bar
        mu_hot=2.0e-5,  # Pa·s
        cp_hot=5190.0,  # J/(kg·K)
        k_hot=0.152,  # W/(m·K)
        
        # Cold side: LH2
        rho_cold=70.8,  # kg/m³ at 20K, 1 bar
        mu_cold=1.3e-4,  # Pa·s
        cp_cold=9600.0,  # J/(kg·K)
        k_cold=0.1,  # W/(m·K)
        
        # Inlet conditions
        T_hot_in=300.0,  # K
        T_cold_in=20.0,  # K
        P_hot_in=2e5,  # Pa (2 bar)
        P_cold_in=1e5,  # Pa (1 bar)
        m_dot_hot=0.01,  # kg/s
        m_dot_cold=0.05,  # kg/s
    )
    
    # Optimization settings
    optimization = OptimizationConfig(
        max_iter=20,
        d_min=0.1,
        d_max=0.9,
        d_init=0.5,
        delta_P_max_hot=10e3,  # 10 kPa
        delta_P_max_cold=5e3,  # 5 kPa
        T_sat_margin=5.0,  # K
        step_size=0.01,
    )
    
    # Create config
    config = Config(
        geometry=geometry,
        fluid=fluid,
        optimization=optimization,
        rve_table_path=os.path.join(
            os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_default.csv'
        ),
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'runs'),
    )
    
    # Load RVE database
    print("Loading RVE database...")
    rve_db = RVEDatabase(config.rve_table_path)
    print(f"RVE database loaded: d range [{rve_db.d_min:.2f}, {rve_db.d_max:.2f}]")
    
    # Run optimization
    print("\n" + "="*60)
    print("Starting optimization")
    print("="*60)
    opt_result = optimize(config, rve_db, log_file="optimization_log.csv")
    
    # Export results
    if opt_result.d_fields:
        final_d = opt_result.d_fields[-1]
        # Get final solution
        from hxopt.macro_model import MacroModel
        model = MacroModel(config, rve_db)
        final_result = model.solve(final_d)
        
        print("\n" + "="*60)
        print("Final Results")
        print("="*60)
        print(f"Q = {final_result.Q/1e6:.3f} MW")
        print(f"ΔP_hot = {final_result.delta_P_hot/1e3:.2f} kPa")
        print(f"ΔP_cold = {final_result.delta_P_cold/1e3:.2f} kPa")
        print(f"T_hot_out = {final_result.T_hot[-1]:.2f} K")
        print(f"T_cold_out = {final_result.T_cold[0]:.2f} K")
        
        # Export
        export_field_csv(final_result, final_d, config)
        export_vtk(final_result, final_d, config)
        
        print(f"\nResults exported to {config.output_dir}/")
    else:
        print("No optimization results to export.")


if __name__ == "__main__":
    main()

