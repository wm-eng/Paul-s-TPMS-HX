# REFPROP Setup Guide

This guide explains how to set up REFPROP from the HyFlux_Hx repository for use with the TPMS heat exchanger optimizer.

## REFPROP Source

The optimizer supports REFPROP from:
- **HyFlux_Hx Repository**: https://github.com/psperera/HyFlux_Hx/tree/main/Hyflux/REFPROP_9
  - Python wrapper files (`refprop_core.py`, `refprop_integration.py`) are included in `src/hxopt/refprop_wrapper/`
  - REFPROP_9 data directory should be symlinked or copied to project root
- Standard REFPROP installations (ctREFPROP, REFPROP-dll)
- COOLProp (fallback, always available)

## Installation Options

### Option 1: Clone HyFlux_Hx Repository (Recommended)

1. Clone the repository:
```bash
cd ~
git clone https://github.com/psperera/HyFlux_Hx.git
```

2. The REFPROP_9 module should be at:
   - `~/HyFlux_Hx/Hyflux/REFPROP_9/`

3. The optimizer will automatically detect it if:
   - The repository is cloned to `~/HyFlux_Hx/`
   - Or placed in the project root as `REFPROP_9/`
   - Or placed as `Hyflux/REFPROP_9/` relative to project root

### Option 2: Symlink REFPROP_9 (Recommended)

If you have the REFPROP_9 directory elsewhere:

```bash
# Create symlink in project root (already done if following setup)
cd /Users/paulperera/Coding/TPMS-HX
ln -s /Users/paulperera/Coding/HyFlux_Hx/Hyflux/REFPROP_9 REFPROP_9

# Verify symlink
ls -la REFPROP_9
```

**Note**: The Python wrapper files (`refprop_core.py`, `refprop_integration.py`) are already copied to `src/hxopt/refprop_wrapper/` and do not need to be symlinked.

### Option 3: Add to Python Path

Add the REFPROP_9 directory to your Python path:

```python
import sys
import os
sys.path.insert(0, '/path/to/HyFlux_Hx/Hyflux/REFPROP_9')
```

## Verification

Test that REFPROP_9 is detected:

```python
from hxopt.materials import HAS_REFPROP, RealFluidProperties

print(f"REFPROP available: {HAS_REFPROP}")

if HAS_REFPROP:
    props = RealFluidProperties('helium', backend='REFPROP')
    rho = props.density(300.0, 2e5)
    print(f"Helium density: {rho:.4f} kg/m³")
```

## REFPROP_9 API Compatibility

The optimizer automatically detects REFPROP_9 and tries multiple API patterns:

1. **REFPROP_9 specific methods**:
   - `get_prop(prop, 'T', T, 'P', P, fluid)`
   - `PropsSI(prop, 'T', T, 'P', P, fluid)`
   - Direct methods: `rho()`, `mu()`, `cp()`, `k()`
   - `Tsat_P(P, fluid)`

2. **Standard REFPROP API**:
   - `REFPROPdll(fluid, inputs, outputs, ...)`
   - `REFPROP(fluid, inputs, outputs, ...)`

3. **Fallback**: COOLProp if REFPROP_9 methods not found

## Troubleshooting

### REFPROP_9 Not Found

If REFPROP_9 is not detected:

1. **Check path**: Verify REFPROP_9 directory exists and is accessible
2. **Check Python path**: Ensure REFPROP_9 directory is in sys.path
3. **Check module**: Try importing directly:
   ```python
   import sys
   sys.path.insert(0, '/path/to/REFPROP_9')
   import REFPROP_9
   ```

### Import Errors

If you get import errors:

1. Check that REFPROP_9 has all required dependencies
2. Verify Python version compatibility
3. Check for missing DLLs or shared libraries (if REFPROP_9 requires them)

### API Mismatch

If property lookups fail:

1. Check REFPROP_9 API documentation
2. The optimizer will automatically fall back to COOLProp
3. You can force COOLProp: `property_backend='COOLProp'`

## Usage

Once REFPROP_9 is set up, use it in your configuration:

```python
from hxopt.config import FluidConfig

fluid = FluidConfig(
    use_real_properties=True,
    hot_fluid_name='helium',
    cold_fluid_name='hydrogen',
    property_backend='REFPROP',  # Force REFPROP (or 'auto' to try REFPROP first)
    # ... other settings
)
```

The optimizer will automatically:
1. Try REFPROP_9 from HyFlux_Hx repository
2. Fall back to standard REFPROP if REFPROP_9 not found
3. Fall back to COOLProp if REFPROP not available
4. Use constant properties as last resort

## Notes

- REFPROP_9 from HyFlux_Hx may have a different API than standard REFPROP
- The optimizer handles multiple API patterns automatically
- COOLProp is always available as a reliable fallback
- Real-fluid properties are essential for accurate cryogenic calculations

