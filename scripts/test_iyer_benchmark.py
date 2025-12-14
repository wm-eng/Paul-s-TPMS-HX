#!/usr/bin/env python3
"""
Quick test script for Iyer et al. (2022) benchmark conditions.

This script tests the code with room-temperature air conditions
typical of Iyer et al.'s study.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig
from hxopt.rve_db import RVEDatabase
from hxopt.macro_model import MacroModel

def main():
    print('='*70)
    print('Iyer et al. (2022) Benchmark Test')
    print('Room-Temperature TPMS Heat Exchanger Performance')
    print('='*70)
    print()
    
    # Test Case: Ambient air conditions (room temperature benchmark)
    geometry = GeometryConfig(
        length=0.1,  # 10 cm
        width=0.1,
        height=0.1,
        n_segments=20,
    )
    
    # Air properties at room temperature
    fluid = FluidConfig(
        T_hot_in=320.0,  # K (47°C)
        T_cold_in=300.0,  # K (27°C)
        P_hot_in=101325.0,  # Pa (1 atm)
        P_cold_in=101325.0,  # Pa
        m_dot_hot=0.001,  # kg/s (1 g/s)
        m_dot_cold=0.001,  # kg/s
        use_real_properties=False,
        # Air at 320K
        rho_hot=1.177,  # kg/m³
        mu_hot=1.95e-5,  # Pa·s
        cp_hot=1007.0,  # J/(kg·K)
        k_hot=0.0278,  # W/(m·K)
        # Air at 300K
        rho_cold=1.161,  # kg/m³
        mu_cold=1.85e-5,  # Pa·s
        cp_cold=1007.0,  # J/(kg·K)
        k_cold=0.0262,  # W/(m·K)
    )
    
    optimization = OptimizationConfig()
    
    config = Config(
        geometry=geometry,
        fluid=fluid,
        optimization=optimization,
        rve_table_path=os.path.join('data', 'rve_tables', 'primitive_default.csv'),
    )
    
    print('Test Conditions:')
    print(f'  Geometry: {config.geometry.length*100:.1f} cm × {config.geometry.width*100:.1f} cm × {config.geometry.height*100:.1f} cm')
    print(f'  T_hot_in: {config.fluid.T_hot_in:.1f} K ({config.fluid.T_hot_in-273.15:.1f}°C)')
    print(f'  T_cold_in: {config.fluid.T_cold_in:.1f} K ({config.fluid.T_cold_in-273.15:.1f}°C)')
    print(f'  ΔT: {config.fluid.T_hot_in - config.fluid.T_cold_in:.1f} K')
    print(f'  P: {config.fluid.P_hot_in/1000:.1f} kPa (atmospheric)')
    print(f'  m_dot: {config.fluid.m_dot_hot*1000:.2f} g/s')
    print()
    
    # Load RVE database
    rve_db = RVEDatabase(config.rve_table_path)
    print(f'RVE Database: {os.path.basename(config.rve_table_path)}')
    print(f'  d range: [{rve_db.d_min:.2f}, {rve_db.d_max:.2f}]')
    print()
    
    # Test with different d values
    d_values = [0.3, 0.5, 0.7]
    print('Testing different design variables (d):')
    print()
    
    for d in d_values:
        model = MacroModel(config, rve_db)
        d_field = np.full(config.geometry.n_segments, d)
        
        try:
            result = model.solve(d_field)
            
            # Check for numerical issues
            if not np.all(np.isfinite(result.T_hot)) or not np.all(np.isfinite(result.T_cold)):
                print(f'  d = {d:.1f}: ✗ Numerical instability (non-finite values)')
                continue
            
            if np.any(result.T_hot < 0) or np.any(result.T_cold < 0):
                print(f'  d = {d:.1f}: ✗ Numerical instability (negative temperatures)')
                continue
            
            print(f'  d = {d:.1f}:')
            print(f'    Q: {result.Q:.2f} W ({result.Q/1000:.3f} kW)')
            print(f'    ΔP_hot: {result.delta_P_hot:.1f} Pa ({result.delta_P_hot/1000:.3f} kPa)')
            print(f'    ΔP_cold: {result.delta_P_cold:.1f} Pa ({result.delta_P_cold/1000:.3f} kPa)')
            print(f'    T_hot_out: {result.T_hot[-1]:.2f} K ({result.T_hot[-1]-273.15:.2f}°C)')
            print(f'    T_cold_out: {result.T_cold[0]:.2f} K ({result.T_cold[0]-273.15:.2f}°C)')
            
            # Energy balance check
            Q_hot = config.fluid.m_dot_hot * config.fluid.cp_hot * (result.T_hot[0] - result.T_hot[-1])
            energy_error = abs(Q_hot - result.Q) / max(abs(Q_hot), abs(result.Q), 1.0) * 100
            print(f'    Energy balance error: {energy_error:.2f}%')
            print()
            
        except Exception as e:
            print(f'  d = {d:.1f}: ✗ Error: {e}')
            print()
    
    print('='*70)
    print('Expected Values (Iyer et al., 2022):')
    print('  - Heat transfer: 10-100 W (for these conditions)')
    print('  - Pressure drop: 100-1000 Pa')
    print('  - Nusselt number: 10-50')
    print('  - Heat transfer coefficient: 100-1000 W/(m²·K)')
    print()
    print('Note: Numerical stability issues may cause unrealistic results.')
    print('The solver needs improvement for robust operation.')
    print('='*70)


if __name__ == "__main__":
    main()

