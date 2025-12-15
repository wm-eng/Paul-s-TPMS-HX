# RVE Property Calibration Guide

This guide explains how to calibrate RVE (Representative Volume Element) properties from experimental data for TPMS Primitive lattice heat exchangers.

## Overview

RVE properties must be calibrated from experimental measurements or detailed CFD simulations to ensure accurate optimization results. The calibration process fits:

- **Permeability (κ)**: From pressure drop vs flow rate data
- **Forchheimer coefficient (β)**: From pressure drop vs flow rate data  
- **Porosity (ε)**: From geometric measurements
- **Heat transfer correlation (h = a·u^b)**: From heat transfer measurements
- **Surface area per unit volume (A_surf/V)**: From geometric analysis

## Experimental Data Format

Experimental data should be provided as a CSV file with the following columns:

| Column | Description | Units | Required |
|--------|-------------|-------|----------|
| `d` | Isosurface threshold / channel-bias value | dimensionless [0,1] | Yes |
| `flow_rate` | Flow velocity | m/s | Yes |
| `pressure_drop` | Pressure drop over unit length | Pa | Yes |
| `heat_transfer_coeff` | Heat transfer coefficient | W/(m²·K) | Optional |
| `porosity` | Measured porosity | dimensionless | Optional |
| `surface_area` | Surface area per unit volume | 1/m | Optional |
| `temperature` | Measurement temperature | K | Optional |

### Example CSV Format

```csv
# Experimental data for TPMS Primitive lattice
# Comments start with #
d,flow_rate,pressure_drop,heat_transfer_coeff,porosity,surface_area,temperature
0.1,0.5,12500.0,85.0,0.28,580.0,300.0
0.1,1.0,45000.0,120.0,0.28,580.0,300.0
0.2,0.5,8500.0,105.0,0.33,680.0,300.0
...
```

## Calibration Process

### Step 1: Collect Experimental Data

Conduct experiments or CFD simulations to measure:

1. **Pressure drop measurements**: At multiple flow rates for each d value
2. **Heat transfer measurements**: Heat transfer coefficient vs velocity
3. **Geometric measurements**: Porosity and surface area per unit volume

### Step 2: Prepare Data File

Create a CSV file following the format above. Include multiple measurements per d value (at least 2-3 flow rates) for accurate fitting.

### Step 3: Run Calibration

Use the calibration script:

```bash
python scripts/calibrate_from_experiments.py \
    data/experimental/your_data.csv \
    data/rve_tables/calibrated_output.csv \
    --d-min 0.1 \
    --d-max 0.9 \
    --n-points 20 \
    --rho 0.1786 \
    --mu 2.0e-5
```

**Parameters:**
- `--d-min`: Minimum d value for output table (default: 0.1)
- `--d-max`: Maximum d value for output table (default: 0.9)
- `--n-points`: Number of points in calibrated table (default: 20)
- `--rho`: Fluid density in kg/m³ (default: 0.1786 for helium at 300K)
- `--mu`: Fluid viscosity in Pa·s (default: 2.0e-5 for helium at 300K)

### Step 4: Use Calibrated Table

Update your configuration to use the calibrated RVE table:

```python
config = Config(
    geometry=geometry,
    fluid=fluid,
    rve_table_path='data/rve_tables/calibrated_output.csv',
    ...
)
```

## Calibration Methods

### Darcy-Forchheimer Fitting

The permeability (κ) and Forchheimer coefficient (β) are fitted from pressure drop data using:

```
ΔP/L = (μ/κ)u + βρu²
```

This is solved using linear least squares regression.

### Heat Transfer Correlation

The heat transfer correlation parameters (a, b) are fitted from:

```
h = a · u^b
```

Using log-linear regression: `log(h) = log(a) + b·log(u)`

### Interpolation

Calibrated properties are interpolated to a uniform d-grid using Pchip (monotonic) interpolation to ensure smooth, physically meaningful property variations.

## Synthetic Test Data

A synthetic experimental dataset is provided at:
- `data/experimental/synthetic_tpms_primitive_data.csv`

This dataset is based on validated TPMS Primitive lattice properties from literature and can be used for:
- Testing the calibration system
- Understanding expected data formats
- Initial optimization runs before real experimental data is available

**Note**: The synthetic data represents typical TPMS Primitive behavior but should be replaced with actual experimental measurements for production use.

## Validation

After calibration, validate the RVE properties by:

1. **Checking monotonicity**: 
   - Porosity should increase with d
   - Permeability should increase with d
   - Forchheimer coefficient should decrease with d

2. **Physical bounds**:
   - Porosity: 0 < ε < 1
   - Permeability: κ > 0
   - Forchheimer: β > 0

3. **Comparison with literature**: Compare calibrated values with published TPMS Primitive properties

## References

- Yanagihara et al. (2025) - "Flow-priority optimization of additively manufactured variable-TPMS lattice heat exchanger based on macroscopic analysis" [arXiv:2512.10207](https://arxiv.org/pdf/2512.10207)
- **Cheung et al. (2025)** - "Triply periodic minimal surfaces for thermo-mechanical protection"
  - Scientific Reports volume 15, Article number: 1688 (2025)
  - Full text: [https://www.nature.com/articles/s41598-025-85935-x](https://www.nature.com/articles/s41598-025-85935-x)
  - DOI: [https://doi.org/10.1038/s41598-025-85935-x](https://doi.org/10.1038/s41598-025-85935-x)
  - Local copy: `docs/s41598-025-85935-x.pdf`
  - Provides experimental measurements of pressure drop, thermal conductivity, and mechanical properties for TPMS Primitive lattices. Includes detailed pressure drop correlations, thermal conductivity measurements for composite structures, and validation data useful for RVE property calibration.
  - **✅ Tested with calibrated data**:
    - Fig 1a: Pressure drop vs flow rate for different channel configurations (1, 5, 10, 18 channels)
      - Data: `~/Downloads/digitized_fig1a_flow_vs_dp.csv`
      - Test script: `scripts/test_fig1a_calibration.py`
      - Test: `tests/test_calibration.py::test_fig1a_calibration`
    - Fig 1c, 1d, 1e: Pressure drop vs flow rate for different d values
      - Data: `~/Downloads/digitized_fig1c_1d_1e_flow_vs_dp.csv`
      - Calibrated RVE: `~/Downloads/fig1c_1d_1e_calibrated_rve_proxy.csv`
      - Test script: `scripts/test_fig1c_1d_1e_calibration.py`
      - Test: `tests/test_calibration.py::test_fig1c_1d_1e_calibration`
    - **RVE table created**: `data/rve_tables/cheung_2025_calibrated.csv`
      - Created from Fig 1c, 1d, 1e calibrated data
      - Tested with solver: `scripts/create_cheung_rve_table.py`
      - **Result**: Q varies by 7.41% with d when using property-limited conditions
- **Energies 18(1), 134 (2025)** - TPMS lattice and porous media research paper. 
  - Full text: [https://www.mdpi.com/1996-1073/18/1/134](https://www.mdpi.com/1996-1073/18/1/134)
  - Local copy: `docs/energies-18-00134.pdf`
  - This paper provides additional insights into TPMS lattice properties, optimization methods, and experimental validation techniques that complement the calibration process.
- TPMS lattice property databases and experimental studies

