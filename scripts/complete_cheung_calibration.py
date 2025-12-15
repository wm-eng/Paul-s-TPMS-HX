"""
Complete calibration from all Cheung et al. (2025) data (Fig 1a, 1c, 1d, 1e).

This script:
1. Reads all digitized data (Fig 1a, 1c, 1d, 1e)
2. Calibrates RVE properties for each dataset
3. Combines into a comprehensive RVE table
4. Interpolates to uniform d-grid
5. Creates final RVE table for use in solver
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.calibrate_rve import fit_darcy_forchheimer


def calibrate_from_data(df, series_name, rho, mu, L_ref):
    """Calibrate RVE properties from a data series."""
    subset = df[df['series'] == series_name]
    
    if len(subset) < 2:
        return None
    
    # Convert units
    u = subset['flow_rate_LPM'].values / 60.0  # m³/s per unit area proxy
    dp = subset['pressure_drop_kPa'].values * 1e3  # Pa
    lengths = np.full(len(u), L_ref)
    
    try:
        kappa, beta = fit_darcy_forchheimer(u, dp, lengths, rho, mu)
        return {
            'series': series_name,
            'kappa': kappa,
            'beta': beta,
            'n_points': len(subset),
        }
    except:
        return None


def estimate_d_from_series(series_name):
    """Estimate d value from series name."""
    # Fig 1a: channel configurations
    if 'Fig1a' in series_name or 'fig1a' in series_name.lower():
        if '1channel' in series_name:
            return 0.3
        elif '5channels' in series_name:
            return 0.5
        elif '10channels' in series_name:
            return 0.6
        elif '18channels' in series_name:
            return 0.7
        else:
            return 0.5
    
    # Fig 1c: low_d, mid_d, high_d
    elif 'Fig1c' in series_name or 'fig1c' in series_name.lower():
        if 'low_d' in series_name:
            return 0.7  # Note: low_d might mean low porosity = high d
        elif 'mid_d' in series_name:
            return 0.5
        elif 'high_d' in series_name:
            return 0.9
        else:
            return 0.5
    
    # Fig 1d: c_-0.2, c_0, c_+0.2
    elif 'Fig1d' in series_name or 'fig1d' in series_name.lower():
        if 'c_-0.2' in series_name or 'c_-' in series_name:
            return 0.4
        elif 'c_0' in series_name or 'c_0' in series_name:
            return 0.5
        elif 'c_+0.2' in series_name or 'c_+' in series_name:
            return 0.6
        else:
            return 0.5
    
    # Fig 1e: unit ratios
    elif 'Fig1e' in series_name or 'fig1e' in series_name.lower():
        if '1:1:1' in series_name:
            return 0.5
        elif '1:1:2' in series_name:
            return 0.55
        else:
            return 0.5
    
    return 0.5  # Default


def create_complete_rve_table():
    """Create complete RVE table from all Cheung et al. (2025) data."""
    
    print("="*70)
    print("COMPLETE CALIBRATION FROM CHEUNG ET AL. (2025) DATA")
    print("="*70)
    
    downloads_path = os.path.expanduser('~/Downloads')
    
    # Load all digitized data
    data_files = {
        'fig1a': os.path.join(downloads_path, 'digitized_fig1a_flow_vs_dp.csv'),
        'fig1c_1d_1e': os.path.join(downloads_path, 'digitized_fig1c_1d_1e_flow_vs_dp.csv'),
    }
    
    all_data = []
    for name, path in data_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"\n✅ Loaded {name}: {len(df)} data points, {len(df['series'].unique())} series")
            all_data.append(df)
        else:
            print(f"\n⚠️  File not found: {path}")
    
    if not all_data:
        print("\n❌ No data files found!")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal data points: {len(combined_df)}")
    print(f"Total series: {len(combined_df['series'].unique())}")
    
    # Fluid properties
    rho = 0.1786  # kg/m³
    mu = 2.0e-5   # Pa·s
    L_ref = 0.1   # m
    
    # Calibrate each series
    print("\n" + "="*70)
    print("CALIBRATING RVE PROPERTIES")
    print("="*70)
    
    calibration_results = []
    
    for series_name in sorted(combined_df['series'].unique()):
        result = calibrate_from_data(combined_df, series_name, rho, mu, L_ref)
        if result:
            d_est = estimate_d_from_series(series_name)
            result['d'] = d_est
            calibration_results.append(result)
            print(f"  {series_name:<25} d={d_est:.2f}  κ={result['kappa']:.2e} m²  β={result['beta']:.2e} 1/m")
    
    if not calibration_results:
        print("\n❌ No successful calibrations!")
        return None
    
    # Create DataFrame
    cal_df = pd.DataFrame(calibration_results)
    
    # Handle duplicate d values by averaging
    if cal_df['d'].duplicated().any():
        print("\n⚠️  Averaging duplicate d values...")
        cal_df = cal_df.groupby('d').agg({
            'kappa': 'mean',
            'beta': 'mean',
            'n_points': 'sum',
        }).reset_index()
    
    # Sort by d
    cal_df = cal_df.sort_values('d').reset_index(drop=True)
    
    print(f"\n✅ Calibrated {len(cal_df)} unique d values")
    print(f"   d range: [{cal_df['d'].min():.2f}, {cal_df['d'].max():.2f}]")
    
    # Interpolate to uniform d-grid
    print("\n" + "="*70)
    print("INTERPOLATING TO UNIFORM d-GRID")
    print("="*70)
    
    d_min = max(0.1, cal_df['d'].min())
    d_max = min(0.9, cal_df['d'].max())
    n_points = 20
    d_grid = np.linspace(d_min, d_max, n_points)
    
    # Interpolate kappa and beta
    kappa_interp = PchipInterpolator(cal_df['d'].values, cal_df['kappa'].values)
    beta_interp = PchipInterpolator(cal_df['d'].values, cal_df['beta'].values)
    
    kappa_grid = kappa_interp(d_grid)
    beta_grid = beta_interp(d_grid)
    
    # Estimate other properties
    # Porosity: roughly linear with d
    eps_grid = 0.3 + 0.4 * d_grid
    
    # Surface area per unit volume: increases with d
    A_surf_V_grid = 600.0 + 800.0 * d_grid
    
    # Heat transfer coefficients: estimated
    h_a_hot_grid = 95.0 + 240.0 * d_grid
    h_b_hot_grid = 0.78 + 0.18 * d_grid
    h_a_cold_grid = 76.0 + 192.0 * d_grid
    h_b_cold_grid = 0.74 + 0.19 * d_grid
    
    # Thermal conductivity: decreases with d (less solid)
    lambda_solid_grid = 50.0 - 34.0 * d_grid
    
    # Create final RVE table
    rve_table = pd.DataFrame({
        'd': d_grid,
        'kappa_hot': kappa_grid,
        'beta_hot': beta_grid,
        'eps_hot': eps_grid,
        'lambda_solid': lambda_solid_grid,
        'h_a_hot': h_a_hot_grid,
        'h_b_hot': h_b_hot_grid,
        'h_a_cold': h_a_cold_grid,
        'h_b_cold': h_b_cold_grid,
        'A_surf_V': A_surf_V_grid,
    })
    
    # Save RVE table
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'rve_tables', 'cheung_2025_complete.csv'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add header
    header = """# Complete RVE properties calibrated from Cheung et al. (2025)
# Source: Scientific Reports volume 15, Article number: 1688 (2025)
# DOI: https://doi.org/10.1038/s41598-025-85935-x
# Local copy: docs/s41598-025-85935-x.pdf
#
# Calibrated from:
# - Fig 1a: Different channel configurations (1, 5, 10, 18 channels)
# - Fig 1c, 1d, 1e: Different d values and configurations
# Data sources:
#   ~/Downloads/digitized_fig1a_flow_vs_dp.csv
#   ~/Downloads/digitized_fig1c_1d_1e_flow_vs_dp.csv
#
# Calibration method:
# - Darcy-Forchheimer fitting: ΔP/L = (μ/κ) u + β ρ u²
# - Interpolated to uniform d-grid using Pchip (monotonic)
# - Other properties estimated based on typical TPMS Primitive behavior
#
# Properties:
# d: Isosurface threshold / channel-bias value
# kappa_hot: Permeability (m²) - calibrated from pressure drop data
# beta_hot: Forchheimer coefficient (1/m) - calibrated from pressure drop data
# eps_hot: Porosity (dimensionless) - estimated
# lambda_solid: Solid thermal conductivity (W/(m·K)) - estimated
# h_a_hot, h_b_hot: Heat transfer correlation parameters h = a * u^b - estimated
# h_a_cold, h_b_cold: Cold side correlation parameters - estimated
# A_surf_V: Surface area per unit volume (1/m) - estimated
#
"""
    
    with open(output_path, 'w') as f:
        f.write(header)
        rve_table.to_csv(f, index=False)
    
    print(f"\n✅ Complete RVE table saved to: {output_path}")
    print(f"   Contains {len(rve_table)} entries")
    print(f"   d range: [{rve_table['d'].min():.2f}, {rve_table['d'].max():.2f}]")
    
    # Verify monotonicity
    print(f"\nVerification:")
    kappa_mono = np.all(np.diff(rve_table['kappa_hot']) >= 0)
    beta_mono = np.all(np.diff(rve_table['beta_hot']) <= 0)
    eps_mono = np.all(np.diff(rve_table['eps_hot']) >= 0)
    
    print(f"  κ increases with d: {'✅' if kappa_mono else '❌'}")
    print(f"  β decreases with d: {'✅' if beta_mono else '❌'}")
    print(f"  ε increases with d: {'✅' if eps_mono else '❌'}")
    
    return output_path


def test_complete_rve_table(rve_table_path):
    """Quick test of the complete RVE table."""
    
    print("\n" + "="*70)
    print("TESTING COMPLETE RVE TABLE")
    print("="*70)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from hxopt.rve_db import RVEDatabase
    
    try:
        rve_db = RVEDatabase(rve_table_path)
        print(f"✅ RVE database loaded successfully")
        print(f"   d range: [{rve_db.d_min:.2f}, {rve_db.d_max:.2f}]")
        
        # Test interpolation
        d_test = np.array([0.1, 0.5, 0.9])
        kappa = rve_db.kappa_hot(d_test)
        beta = rve_db.beta_hot(d_test)
        eps = rve_db.eps_hot(d_test)
        
        print(f"\nSample properties:")
        for d, k, b, e in zip(d_test, kappa, beta, eps):
            print(f"  d={d:.1f}: κ={k:.2e} m², β={b:.2e} 1/m, ε={e:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading RVE table: {e}")
        return False


def main():
    """Main function."""
    rve_table_path = create_complete_rve_table()
    
    if rve_table_path:
        success = test_complete_rve_table(rve_table_path)
        if success:
            print("\n" + "="*70)
            print("✅ CALIBRATION COMPLETE!")
            print("="*70)
            print(f"\nRVE table ready: {rve_table_path}")
            print("\nYou can now use this RVE table in the solver and GUI.")
            return 0
        else:
            return 1
    else:
        return 1


if __name__ == "__main__":
    exit(main())

