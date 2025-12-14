"""
Test script for calibrating RVE properties from Fig 1c, 1d, 1e digitized data.

This script demonstrates:
1. Reading digitized flow vs pressure drop data
2. Converting units (LPM -> m³/s, kPa -> Pa)
3. Fitting Darcy-Forchheimer model: ΔP/L = (μ/κ) u + β ρ u²
4. Validating against calibrated RVE proxy data
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.calibrate_rve import fit_darcy_forchheimer


def main():
    """Main calibration test function."""
    
    # Paths to data files
    # Update these paths to match your file locations
    digitized_csv = os.path.join(
        os.path.expanduser('~'), 'Downloads',
        'digitized_fig1c_1d_1e_flow_vs_dp.csv'
    )
    calibrated_csv = os.path.join(
        os.path.expanduser('~'), 'Downloads',
        'fig1c_1d_1e_calibrated_rve_proxy.csv'
    )
    
    # Check if files exist
    if not os.path.exists(digitized_csv):
        print(f"ERROR: Digitized data file not found: {digitized_csv}")
        print("Please ensure the file exists or update the path in the script.")
        return 1
    
    if not os.path.exists(calibrated_csv):
        print(f"ERROR: Calibrated RVE proxy file not found: {calibrated_csv}")
        print("Please ensure the file exists or update the path in the script.")
        return 1
    
    # Read digitized flow vs pressure drop data
    print(f"Reading digitized data from: {digitized_csv}")
    df = pd.read_csv(digitized_csv)
    print(f"Found {len(df)} data points")
    print(f"Series available: {df['series'].unique()}")
    print()
    
    # Read calibrated RVE proxy for validation
    print(f"Reading calibrated RVE proxy from: {calibrated_csv}")
    df_calibrated = pd.read_csv(calibrated_csv)
    print(f"Found {len(df_calibrated)} calibrated entries")
    print()
    
    # Fluid properties (helium at room temperature)
    rho = 0.1786  # kg/m³
    mu = 2.0e-5   # Pa·s
    
    # Reference area and length for unit conversion
    # These are proxy values used in the calibration
    A_ref = 1e-4  # m² (reference cross-sectional area)
    L_ref = 0.1   # m (reference length)
    
    print("Fluid properties:")
    print(f"  Density (ρ): {rho:.4f} kg/m³")
    print(f"  Viscosity (μ): {mu:.2e} Pa·s")
    print(f"  Reference area (A_ref): {A_ref:.2e} m²")
    print(f"  Reference length (L_ref): {L_ref:.3f} m")
    print()
    
    # Process each series
    results = []
    
    for series_name in sorted(df['series'].unique()):
        subset = df[df['series'] == series_name]
        
        if len(subset) < 2:
            print(f"Skipping {series_name}: insufficient data points ({len(subset)})")
            continue
        
        print(f"Processing series: {series_name}")
        print(f"  Data points: {len(subset)}")
        
        # Convert units
        # Flow rate: LPM -> m³/s per unit area proxy
        u = subset['flow_rate_LPM'].values / 60.0  # m³/s per unit area proxy
        
        # Pressure drop: kPa -> Pa
        dp = subset['pressure_drop_kPa'].values * 1e3  # Pa
        
        print(f"  Flow rates: {u.min():.4e} to {u.max():.4e} m³/s")
        print(f"  Pressure drops: {dp.min():.1f} to {dp.max():.1f} Pa")
        
        # Fit Darcy-Forchheimer model
        # ΔP/L = (μ/κ) u + β ρ u²
        # Using reference length for fitting
        lengths = np.full(len(u), L_ref)
        
        try:
            kappa_fit, beta_fit = fit_darcy_forchheimer(
                u, dp, lengths, rho, mu
            )
            
            # Calculate fitted coefficients in the form used in CSV
            mu_over_kappa_fit = mu / kappa_fit
            beta_rho_fit = beta_fit * rho
            
            print(f"  Fitted permeability (κ): {kappa_fit:.4e} m²")
            print(f"  Fitted Forchheimer coeff (β): {beta_fit:.4e} 1/m")
            print(f"  Fitted μ/κ: {mu_over_kappa_fit:.2f} Pa·s/m²")
            print(f"  Fitted βρ: {beta_rho_fit:.2f} kg/(m³·m)")
            
            # Try to match with calibrated data
            # Extract source figure from series name
            source_fig = None
            if 'Fig1c' in series_name or 'fig1c' in series_name.lower():
                source_fig = 'Fig1c'
            elif 'Fig1d' in series_name or 'fig1d' in series_name.lower():
                source_fig = 'Fig1d'
            elif 'Fig1e' in series_name or 'fig1e' in series_name.lower():
                source_fig = 'Fig1e'
            
            if source_fig:
                cal_match = df_calibrated[df_calibrated['source_fig'] == source_fig]
                if len(cal_match) > 0:
                    print(f"  Found {len(cal_match)} matching calibrated entries")
                    for idx, cal_row in cal_match.iterrows():
                        mu_over_kappa_cal = cal_row['a_visc_mu_over_kappa']
                        beta_rho_cal = cal_row['b_inert_beta_rho']
                        d_cal = cal_row['d']
                        
                        # Convert to kappa and beta for comparison
                        kappa_cal = mu / mu_over_kappa_cal if mu_over_kappa_cal > 0 else 1e-10
                        beta_cal = beta_rho_cal / rho if rho > 0 else 1e6
                        
                        kappa_ratio = kappa_fit / kappa_cal
                        beta_ratio = beta_fit / beta_cal
                        
                        print(f"    Calibrated entry (d={d_cal:.2f}):")
                        print(f"      κ: {kappa_cal:.4e} m² (ratio: {kappa_ratio:.2f}x)")
                        print(f"      β: {beta_cal:.4e} 1/m (ratio: {beta_ratio:.2f}x)")
                        print(f"      μ/κ: {mu_over_kappa_cal:.2f} Pa·s/m²")
                        print(f"      βρ: {beta_rho_cal:.2f} kg/(m³·m)")
            
            results.append({
                'series': series_name,
                'kappa': kappa_fit,
                'beta': beta_fit,
                'mu_over_kappa': mu_over_kappa_fit,
                'beta_rho': beta_rho_fit,
                'n_points': len(subset),
            })
            
        except Exception as e:
            print(f"  ERROR: Failed to fit Darcy-Forchheimer: {e}")
            continue
        
        print()
    
    # Summary
    if results:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        print()
        print("Note: Fitted values use proxy geometry (A_ref, L_ref).")
        print("For physical κ and β, use actual geometry/length values.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
