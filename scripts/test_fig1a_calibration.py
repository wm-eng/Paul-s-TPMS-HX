"""
Test script for calibrating RVE properties from Fig 1a digitized data (Cheung et al., 2025).

This script demonstrates:
1. Reading digitized flow vs pressure drop data from Fig 1a
2. Converting units (LPM -> m³/s, kPa -> Pa)
3. Fitting Darcy-Forchheimer model: ΔP/L = (μ/κ) u + β ρ u²
4. Comparing results across different channel configurations (1, 5, 10, 18 channels)

Fig 1a shows pressure drop vs flow rate for different channel configurations,
which can be used to calibrate RVE properties for different geometries.
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
    
    print("="*70)
    print("FIG 1A CALIBRATION TEST (Cheung et al., 2025)")
    print("="*70)
    print("\nThis script calibrates RVE properties from Fig 1a data")
    print("which shows pressure drop vs flow rate for different channel configurations.\n")
    
    # Paths to data files
    downloads_path = os.path.expanduser('~/Downloads')
    digitized_csv = os.path.join(
        downloads_path, 'digitized_fig1a_flow_vs_dp.csv'
    )
    
    # Fallback to project root if not in Downloads
    if not os.path.exists(digitized_csv):
        digitized_csv = os.path.join(
            os.path.dirname(__file__), '..', 'digitized_fig1a_flow_vs_dp.csv'
        )
    
    # Check if file exists
    if not os.path.exists(digitized_csv):
        print(f"ERROR: Digitized data file not found: {digitized_csv}")
        print("Please ensure the file exists or update the path in the script.")
        return 1
    
    # Read digitized flow vs pressure drop data
    print(f"Reading digitized data from: {digitized_csv}")
    df = pd.read_csv(digitized_csv)
    print(f"Found {len(df)} data points")
    print(f"Series available: {sorted(df['series'].unique())}")
    print()
    
    # Fluid properties (helium at room temperature, typical for Cheung et al.)
    rho = 0.1786  # kg/m³
    mu = 2.0e-5   # Pa·s
    
    print("Fluid properties:")
    print(f"  ρ = {rho:.4f} kg/m³")
    print(f"  μ = {mu:.2e} Pa·s")
    print()
    
    # Reference area and length for unit conversion
    # Note: Fig 1a shows different channel configurations, so we need to estimate
    # the effective geometry. The number of channels affects the total cross-sectional area.
    # For now, we'll use a reference area that scales with number of channels
    A_ref_base = 1e-4  # m² (base reference cross-sectional area)
    L_ref = 0.1        # m (reference length)
    
    print("Reference geometry:")
    print(f"  A_ref_base = {A_ref_base:.4f} m²")
    print(f"  L_ref = {L_ref:.2f} m")
    print()
    
    # Results storage
    calibration_results = []
    
    # Test calibration for each series in the data
    print("="*70)
    print("CALIBRATION RESULTS")
    print("="*70)
    print(f"\n{'Series':<20} {'d (est)':<10} {'κ (m²)':<15} {'β (1/m)':<15} {'Points':<8}")
    print("-" * 70)
    
    for series_name in sorted(df['series'].unique()):
        subset = df[df['series'] == series_name]
        
        if len(subset) < 2:
            print(f"{series_name:<20} {'SKIP':<10} {'(need ≥2 points)':<15} {'':<15} {len(subset):<8}")
            continue
        
        # Extract number of channels from series name
        # Format: Fig1a_Nchannels or Fig1a_1channel
        if '1channel' in series_name:
            n_channels = 1
        else:
            # Extract number from name like "5channels" -> 5
            try:
                n_channels = int(series_name.split('_')[1].replace('channels', ''))
            except:
                n_channels = 1  # Default
        
        # Estimate d value based on channel configuration
        # More channels typically correspond to higher d (more open structure)
        # This is a rough estimate - actual d would need to be determined from geometry
        if n_channels == 1:
            d_est = 0.3  # Low d for single channel (more restrictive)
        elif n_channels == 5:
            d_est = 0.5  # Medium d
        elif n_channels == 10:
            d_est = 0.6  # Medium-high d
        elif n_channels == 18:
            d_est = 0.7  # High d (more open)
        else:
            d_est = 0.5  # Default
        
        # Convert units
        # Flow rate: LPM -> m³/s per unit area proxy
        # Note: The actual flow rate per channel may vary, but we use total flow
        u = subset['flow_rate_LPM'].values / 60.0  # m³/s per unit area proxy
        
        # Pressure drop: kPa -> Pa
        dp = subset['pressure_drop_kPa'].values * 1e3  # Pa
        
        # Fit Darcy-Forchheimer model
        # ΔP/L = (μ/κ) u + β ρ u²
        # Using reference length for fitting
        lengths = np.full(len(u), L_ref)
        
        try:
            kappa_fit, beta_fit = fit_darcy_forchheimer(
                u, dp, lengths, rho, mu
            )
            
            # Store results
            calibration_results.append({
                'series': series_name,
                'n_channels': n_channels,
                'd_est': d_est,
                'kappa': kappa_fit,
                'beta': beta_fit,
                'n_points': len(subset),
            })
            
            print(f"{series_name:<20} {d_est:<10.2f} {kappa_fit:<15.2e} {beta_fit:<15.2e} {len(subset):<8}")
            
        except Exception as e:
            print(f"{series_name:<20} {'ERROR':<10} {str(e):<15} {'':<15} {len(subset):<8}")
    
    # Summary analysis
    if calibration_results:
        print("\n" + "="*70)
        print("SUMMARY ANALYSIS")
        print("="*70)
        
        results_df = pd.DataFrame(calibration_results)
        
        print(f"\nTotal series calibrated: {len(results_df)}")
        print(f"\nProperty ranges:")
        print(f"  κ: {results_df['kappa'].min():.2e} to {results_df['kappa'].max():.2e} m²")
        print(f"  β: {results_df['beta'].min():.2e} to {results_df['beta'].max():.2e} 1/m")
        
        print(f"\nTrend analysis:")
        print(f"  More channels → higher d (estimated)")
        print(f"  Expected: κ should increase with d, β should decrease with d")
        
        # Check if trends match expectations
        if len(results_df) >= 2:
            sorted_df = results_df.sort_values('d_est')
            kappa_trend = np.all(np.diff(sorted_df['kappa']) >= 0)
            beta_trend = np.all(np.diff(sorted_df['beta']) <= 0)
            
            print(f"\n  κ increases with d: {'✅' if kappa_trend else '❌'}")
            print(f"  β decreases with d: {'✅' if beta_trend else '❌'}")
        
        print(f"\n" + "="*70)
        print("NOTES")
        print("="*70)
        print("""
1. The d values are ESTIMATES based on channel configuration.
   Actual d values should be determined from geometry measurements.

2. The calibration uses proxy reference geometry (A_ref, L_ref).
   For physical κ and β values, use actual geometry dimensions.

3. Fig 1a shows different channel configurations, which may represent
   different TPMS lattice parameters or manufacturing variations.

4. To create a full RVE table, you would need:
   - Actual d values from geometry
   - Porosity measurements
   - Heat transfer coefficient data
   - Surface area per unit volume measurements

5. This data can be combined with Fig 1c, 1d, 1e data for a more
   comprehensive RVE property database.
""")
    
    print("\n✅ Calibration test complete!")
    return 0


if __name__ == "__main__":
    exit(main())
