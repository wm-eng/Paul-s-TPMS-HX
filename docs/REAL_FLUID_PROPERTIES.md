# Real-Fluid Properties Guide

This guide explains how to use REFPROP and COOLProp for accurate cryogenic fluid properties in the TPMS heat exchanger optimizer.

## Overview

The optimizer supports real-fluid properties for:
- **Helium (He)**: Cryogenic gas properties
- **Hydrogen (H2)**: Liquid hydrogen (LH2) properties

Two backends are supported:
1. **REFPROP** (preferred): NIST REFPROP database - highest accuracy
2. **COOLProp** (fallback): Open-source alternative - good accuracy, easier installation

## Installation

### COOLProp (Recommended for Quick Start)

COOLProp is installed by default:

```bash
pip install CoolProp
```

### REFPROP (Optional, Higher Accuracy)

REFPROP requires:
1. NIST REFPROP license and installation
2. Python wrapper installation

```bash
# After installing REFPROP, install Python wrapper
pip install REFPROP-dll  # or ctREFPROP depending on your REFPROP version
```

**Note**: REFPROP is commercial software. COOLProp is free and open-source.

## Usage

### Basic Configuration

Enable real-fluid properties in your configuration:

```python
from hxopt.config import Config, GeometryConfig, FluidConfig

fluid = FluidConfig(
    # Enable real-fluid properties
    use_real_properties=True,
    hot_fluid_name='helium',      # Cryogenic helium gas
    cold_fluid_name='hydrogen',    # Liquid hydrogen
    property_backend='auto',       # 'auto', 'REFPROP', or 'COOLProp'
    
    # Inlet conditions
    T_hot_in=300.0,    # K
    T_cold_in=20.0,    # K
    P_hot_in=2e5,      # Pa
    P_cold_in=1e5,     # Pa
    m_dot_hot=0.01,    # kg/s
    m_dot_cold=0.05,   # kg/s
)
```

### Backend Selection

The `property_backend` parameter controls which library to use:

- **`'auto'`** (default): Tries REFPROP first, falls back to COOLProp if unavailable
- **`'REFPROP'`**: Force use of REFPROP (raises error if unavailable)
- **`'COOLProp'`**: Force use of COOLProp (raises error if unavailable)

### Supported Fluid Names

| Name | Description | Backend Name |
|------|-------------|--------------|
| `'helium'` or `'he'` | Helium gas | HELIUM / Helium |
| `'hydrogen'` or `'h2'` | Hydrogen | HYDROGEN / Hydrogen |
| `'lh2'` | Liquid hydrogen (alias) | HYDROGEN / Hydrogen |

## Property Lookups

The real-fluid property system provides:

- **Density (ρ)**: Temperature and pressure dependent
- **Viscosity (μ)**: Temperature and pressure dependent
- **Specific Heat (cp)**: Temperature and pressure dependent
- **Thermal Conductivity (k)**: Temperature and pressure dependent
- **Saturation Temperature (Tsat)**: Pressure dependent (for hydrogen)

All properties are automatically evaluated at each point in the flow field during the solver iteration.

## Example

See `examples/run_v1_real_properties.py` for a complete example:

```bash
python examples/run_v1_real_properties.py
```

## Fallback Behavior

If neither REFPROP nor COOLProp is available, the system will:

1. **Warn** the user
2. **Fall back** to constant properties (if provided in config)
3. **Raise an error** if constant properties are not provided

To use constant properties as fallback:

```python
fluid = FluidConfig(
    use_real_properties=False,  # Use constant properties
    rho_hot=0.1786,  # kg/m³
    mu_hot=2.0e-5,   # Pa·s
    cp_hot=5190.0,   # J/(kg·K)
    k_hot=0.152,     # W/(m·K)
    # ... cold side properties
)
```

## Performance Considerations

- **REFPROP**: Fastest, most accurate, but requires license
- **COOLProp**: Good accuracy, free, slightly slower than REFPROP
- **Constant properties**: Fastest, but less accurate for cryogenic applications

For cryogenic helium and liquid hydrogen, **real-fluid properties are strongly recommended** due to significant property variations with temperature and pressure.

## Validation

The real-fluid properties are validated against:
- NIST REFPROP database (when using REFPROP)
- NIST WebBook data (when using COOLProp)

Typical accuracy:
- **REFPROP**: ±0.1% for most properties
- **COOLProp**: ±1-2% for most properties

## Troubleshooting

### REFPROP Not Found

If you see "REFPROP not available":
1. Ensure REFPROP is installed on your system
2. Install the Python wrapper: `pip install REFPROP-dll` or `pip install ctREFPROP`
3. Check REFPROP installation path is in system PATH
4. Use COOLProp as fallback: set `property_backend='COOLProp'`

### COOLProp Not Found

If you see "COOLProp not available":
1. Install COOLProp: `pip install CoolProp`
2. Verify installation: `python -c "import CoolProp; print(CoolProp.__version__)"`

### Property Lookup Errors

If property lookups fail:
1. Check temperature and pressure are within valid ranges
2. For hydrogen, ensure temperature is above triple point (~13.8 K)
3. Check fluid name spelling matches supported names
4. Review error messages for specific property calculation issues

## References

- **REFPROP**: NIST Standard Reference Database 23
- **COOLProp**: Open-source thermophysical property library
- NIST WebBook: https://webbook.nist.gov/chemistry/

