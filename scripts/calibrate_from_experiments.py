"""Script to calibrate RVE properties from experimental data."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from hxopt.calibrate_rve import (
    load_experimental_data,
    calibrate_from_experiments,
    ExperimentalData,
)


def main():
    """Calibrate RVE properties from experimental data."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calibrate RVE properties from experimental data'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to experimental data CSV file'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output calibrated RVE table CSV'
    )
    parser.add_argument(
        '--d-min',
        type=float,
        default=0.1,
        help='Minimum d value for output table'
    )
    parser.add_argument(
        '--d-max',
        type=float,
        default=0.9,
        help='Maximum d value for output table'
    )
    parser.add_argument(
        '--n-points',
        type=int,
        default=20,
        help='Number of points in output table'
    )
    parser.add_argument(
        '--rho',
        type=float,
        default=0.1786,
        help='Fluid density (kg/m³) for Darcy-Forchheimer fitting'
    )
    parser.add_argument(
        '--mu',
        type=float,
        default=2.0e-5,
        help='Fluid viscosity (Pa·s) for Darcy-Forchheimer fitting'
    )
    
    args = parser.parse_args()
    
    print(f"Loading experimental data from {args.input_file}...")
    exp_data = load_experimental_data(args.input_file)
    
    print(f"Loaded {len(exp_data.d_values)} experimental measurements")
    print(f"  d range: [{min(exp_data.d_values):.2f}, {max(exp_data.d_values):.2f}]")
    
    print("\nCalibrating RVE properties...")
    fluid_props = {'rho': args.rho, 'mu': args.mu}
    
    calibrated_df = calibrate_from_experiments(
        exp_data,
        d_range=(args.d_min, args.d_max),
        n_points=args.n_points,
        fluid_properties=fluid_props,
    )
    
    print(f"\nCalibrated properties:")
    print(calibrated_df.to_string(index=False))
    
    # Save to CSV
    output_path = args.output_file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    calibrated_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Calibrated RVE table saved to {output_path}")
    print(f"\nYou can now use this table in your optimization:")
    print(f"  config.rve_table_path = '{output_path}'")


if __name__ == "__main__":
    main()

