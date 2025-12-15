# TPMS STL Generation Guide

This guide explains how to generate STL files from optimized TPMS heat exchanger designs using the Python code stack.

## Overview

The STL generation workflow converts the optimized 1D channel-bias field `d(x)` into a 3D TPMS lattice mesh suitable for 3D printing or CAD import.

## Key Components

### 1. TPMS Library (`tpms_library.py`)
- Implicit functions for 8 TPMS types (Gyroid, Primitive, Diamond, IWP, etc.)
- Variant modes: single (`f ≤ t`) and double (`|f| ≤ t`)
- Unit-consistent and typed functions

### 2. STL Generator (`tpms_stl_generator.py`)
- Maps `d ∈ [0,1]` to isosurface threshold `t`
- Generates 3D mesh via marching cubes
- Exports to STL format

### 3. Export Module (`export_geometry.py`)
- Convenience functions for integration
- Works with optimization results

## Workflow

### Step 1: Run Optimization

First, optimize your heat exchanger to get the `d(x)` field:

```python
from hxopt import Config, MacroModel, RVEDatabase, optimize

# Setup configuration
config = Config(...)
rve_db = RVEDatabase(config.rve_table_path)

# Run optimization
opt_result = optimize(config, rve_db)

# Get optimized d field
d_field = opt_result.d_fields[-1]  # Final optimized field
result = opt_result.results[-1]    # Final solution
```

### Step 2: Generate STL

Export the optimized geometry to STL:

```python
from hxopt.export_geometry import export_tpms_stl_from_optimization
from hxopt.tpms_library import TPMSType, VariantMode

# Export to STL
stl_path = export_tpms_stl_from_optimization(
    result=result,
    d_field=d_field,
    config=config,
    filename="optimized_tpms.stl",
    tpms_type=TPMSType.PRIMITIVE,  # or GYROID, DIAMOND, etc.
    variant=VariantMode.SINGLE,     # or DOUBLE
    cell_size=0.001,                # 1mm unit cell
    resolution=50                   # Grid points per cell
)
```

## Parameters

### TPMS Type
- `TPMSType.PRIMITIVE` - Schwarz Primitive (default, good for HX)
- `TPMSType.GYROID` - Gyroid (high surface area)
- `TPMSType.DIAMOND` - Schwarz Diamond
- `TPMSType.IWP` - Schoen I-WP
- `TPMSType.LIDINOID`, `TPMSType.NEOVIUS`, `TPMSType.OCTO`, `TPMSType.SPLIT_P`

### Variant Mode
- `VariantMode.SINGLE` - Solid is `f(x,y,z) ≤ t` (default)
- `VariantMode.DOUBLE` - Solid is `|f(x,y,z)| ≤ t` (symmetric)

### Cell Size
- Unit cell size in meters (default: 0.001 m = 1 mm)
- Smaller = finer lattice structure
- Typical range: 0.0005 - 0.005 m

### Resolution
- Grid points per unit cell (default: 30, reduced from 50 for memory efficiency)
- Higher = smoother mesh but slower and more memory-intensive
- Typical range: 30 - 100
- **Memory management**: Automatic resolution scaling prevents memory exhaustion
- **Maximum grid points**: Limited to ~50 million points to prevent crashes

## d → t Mapping

The system automatically maps the optimizer variable `d ∈ [0,1]` to the TPMS isosurface threshold `t`:

1. **Porosity Sampling**: Samples TPMS function at various `t` values to build porosity lookup table
2. **Interpolation**: For each `d` value, finds corresponding `t` that gives similar porosity
3. **3D Mapping**: Maps `d(x)` field to 3D space and applies corresponding `t` values

The mapping assumes `d ≈ porosity`, which is reasonable for most TPMS structures.

## Advanced Usage

### Custom Mesh Generation

For more control, use the lower-level functions:

```python
from hxopt.tpms_stl_generator import (
    generate_tpms_mesh_from_d_field,
    export_tpms_stl
)

# Generate mesh
vertices, faces = generate_tpms_mesh_from_d_field(
    d_field=d_field,
    config=config,
    tpms_type=TPMSType.GYROID,
    variant=VariantMode.DOUBLE,
    cell_size=0.002,
    resolution=60,
    n_repeats=(2, 2, 2)  # 2x2x2 unit cells
)

# Export
stl_path = export_tpms_stl(
    vertices=vertices,
    faces=faces,
    filename="custom_tpms.stl",
    output_dir=config.output_dir
)
```

### Custom d → t Mapping

If you need custom porosity mapping:

```python
from hxopt.tpms_stl_generator import d_to_porosity_mapping

# Create custom mapping
mapping = d_to_porosity_mapping(
    d=np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
    tpms_type=TPMSType.PRIMITIVE,
    variant=VariantMode.SINGLE,
    n_samples=200  # More samples = more accurate
)

# mapping['d'] - d values
# mapping['t'] - corresponding t values
# mapping['porosity'] - actual porosities
```

## Comparison with MATLAB

### MATLAB Workflow (Typical)
1. Define TPMS implicit function
2. Create 3D grid
3. Evaluate function at grid points
4. Apply threshold `t` to create binary mask
5. Use `isosurface()` or `marching_cubes()` to generate mesh
6. Export to STL using `stlwrite()`

### Python Workflow (This Implementation)
1. ✅ TPMS functions in `tpms_library.py` (equivalent to MATLAB functions)
2. ✅ 3D grid creation in `generate_tpms_mesh_from_d_field()`
3. ✅ Function evaluation via `evaluate_tpms()`
4. ✅ Threshold application via `get_solid_mask()` or isosurface
5. ✅ Marching cubes via `skimage.measure.marching_cubes()`
6. ✅ STL export via `numpy-stl`

**Key Advantages:**
- Integrated with optimization workflow
- Automatic `d → t` mapping
- Type-safe and unit-consistent
- No MATLAB license required
- Works with existing Python stack

## Performance Tips

1. **Resolution**: Start with `resolution=30` for testing, increase to 50-100 for final export
2. **Cell Size**: Match to your 3D printer resolution (typically 0.001-0.002 m)
3. **Memory**: High resolution + large domains = high memory usage
   - GUI automatically scales down resolution if geometry would exceed memory limits
   - Maximum grid points capped at ~50 million to prevent crashes
4. **Parallelization**: Mesh generation is CPU-bound; consider multiprocessing for batch exports
5. **Background Processing**: STL export in GUI runs in background thread to keep UI responsive

## Troubleshooting

### Import Errors
```bash
pip install numpy-stl scikit-image
```

### Mesh Quality Issues
- Increase `resolution` for smoother surfaces
- Check `cell_size` matches your design intent
- Verify `d_field` values are in valid range [0, 1]

### File Size
- STL files can be large (10-100 MB for high-res meshes)
- Consider reducing `resolution` or using binary STL format

### Memory Issues
- **Automatic scaling**: Resolution automatically reduced if geometry would cause memory issues
- **Background processing**: STL export runs in background thread to prevent UI blocking
- **Cleanup**: Large arrays are cleaned up immediately after use
- **If crashes persist**: Try reducing geometry size or manually setting lower resolution

## Example: Complete Workflow

```python
from hxopt import Config, GeometryConfig, FluidConfig, OptimizationConfig
from hxopt import MacroModel, RVEDatabase, optimize
from hxopt.export_geometry import export_tpms_stl_from_optimization
from hxopt.tpms_library import TPMSType, VariantMode

# 1. Setup
config = Config(
    geometry=GeometryConfig(length=0.5, width=0.1, height=0.1, n_segments=50),
    fluid=FluidConfig(...),
    optimization=OptimizationConfig(...),
    rve_table_path="data/rve_tables/primitive_default.csv"
)

# 2. Optimize
rve_db = RVEDatabase(config.rve_table_path)
opt_result = optimize(config, rve_db)

# 3. Export STL
d_field = opt_result.d_fields[-1]
result = opt_result.results[-1]

stl_path = export_tpms_stl_from_optimization(
    result=result,
    d_field=d_field,
    config=config,
    filename="optimized_primitive.stl",
    tpms_type=TPMSType.PRIMITIVE,
    variant=VariantMode.SINGLE,
    cell_size=0.001,
    resolution=50
)

print(f"STL exported to: {stl_path}")
```

## References

- TPMS implicit functions based on standard formulations (see `tpms_library.py`)
- Marching cubes: `scikit-image.measure.marching_cubes`
- STL format: `numpy-stl` library
- MATLAB equivalent: `isosurface()`, `stlwrite()`

