"""Example v1.2 optimization run with real-fluid properties (REFPROP/COOLProp)."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig
from hxopt.rve_db import RVEDatabase
from hxopt.optimize_mma import optimize
from hxopt.export_geometry import export_field_csv, export_vtk


def main():
    """Run v1.2 optimization example with real-fluid properties."""
    
    # Geometry
    geometry = GeometryConfig(
        length=0.5,  # m
        width=0.1,  # m
        height=0.1,  # m
        n_segments=50,
    )
    
    # Fluid properties - using real-fluid properties
    fluid = FluidConfig(
        # Real-fluid properties enabled
        use_real_properties=True,
        hot_fluid_name='helium',  # Cryogenic helium gas
        cold_fluid_name='hydrogen',  # Liquid hydrogen
        property_backend='auto',  # Try REFPROP first, fallback to COOLProp
        
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
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'real_props'),
    )
    
    # Load RVE database
    print("Loading RVE database...")
    rve_db = RVEDatabase(config.rve_table_path)
    print(f"RVE database loaded: d range [{rve_db.d_min:.2f}, {rve_db.d_max:.2f}]")
    
    # Check property backend availability
    from hxopt.materials import HAS_REFPROP, HAS_COOLPROP
    print(f"\nProperty backends:")
    print(f"  REFPROP: {'✓ Available' if HAS_REFPROP else '✗ Not available'}")
    print(f"  COOLProp: {'✓ Available' if HAS_COOLPROP else '✗ Not available'}")
    
    if not HAS_REFPROP and not HAS_COOLPROP:
        print("\n⚠ Warning: Neither REFPROP nor COOLProp available!")
        print("  Install COOLProp: pip install CoolProp")
        print("  Falling back to constant properties...")
        # Fallback to constant properties
        fluid.use_real_properties = False
        fluid.rho_hot = 0.1786
        fluid.mu_hot = 2.0e-5
        fluid.cp_hot = 5190.0
        fluid.k_hot = 0.152
        fluid.rho_cold = 70.8
        fluid.mu_cold = 1.3e-4
        fluid.cp_cold = 9600.0
        fluid.k_cold = 0.1
    
    # Run optimization
    print("\n" + "="*60)
    print("Starting optimization with real-fluid properties")
    print("="*60)
    opt_result = optimize(config, rve_db, log_file="optimization_log_real_props.csv")
    
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
        export_field_csv(final_result, final_d, config, filename="field_data_real_props.csv")
        export_vtk(final_result, final_d, config, filename="field_data_real_props.vtk")
        
        print(f"\nResults exported to {config.output_dir}/")
    else:
        print("No optimization results to export.")


if __name__ == "__main__":
    main()

