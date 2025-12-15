"""Export geometry and field data."""

import numpy as np
import os
import csv
from .macro_model import MacroModelResult
from .config import Config

# Optional imports for STL generation
try:
    from .tpms_stl_generator import (
        generate_and_export_stl,
        generate_tpms_mesh_from_d_field,
        export_tpms_stl
    )
    from .tpms_library import TPMSType, VariantMode
    STL_EXPORT_AVAILABLE = True
except ImportError:
    STL_EXPORT_AVAILABLE = False


def export_field_csv(
    result: MacroModelResult,
    d_field: np.ndarray,
    config: Config,
    filename: str = "field_data.csv",
):
    """
    Export field data to CSV.
    
    Parameters
    ----------
    result : MacroModelResult
        Solution from macromodel
    d_field : np.ndarray
        Channel-bias field
    config : Config
        Configuration object
    filename : str
        Output filename
    """
    path = os.path.join(config.output_dir, filename)
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'x', 'd', 'T_hot', 'T_cold', 'T_solid',
            'P_hot', 'P_cold'
        ])
        
        # Write node data (temperatures, pressures)
        for i in range(len(result.x)):
            d_val = d_field[min(i, len(d_field)-1)] if i < len(d_field) else d_field[-1]
            writer.writerow([
                result.x[i],
                d_val,
                result.T_hot[i],
                result.T_cold[i],
                result.T_solid[i],
                result.P_hot[i],
                result.P_cold[i],
            ])
    
    print(f"Exported field data to {path}")


def export_vtk(
    result: MacroModelResult,
    d_field: np.ndarray,
    config: Config,
    filename: str = "field_data.vtk",
):
    """
    Export field data to VTK format (simple 1D line).
    
    Parameters
    ----------
    result : MacroModelResult
        Solution from macromodel
    d_field : np.ndarray
        Channel-bias field
    config : Config
        Configuration object
    filename : str
        Output filename
    """
    path = os.path.join(config.output_dir, filename)
    
    with open(path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("TPMS HX Field Data\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(result.x)} float\n")
        
        # Write points (1D line in 3D space)
        for x in result.x:
            f.write(f"{x} 0.0 0.0\n")
        
        f.write(f"LINES 1 {len(result.x) + 1}\n")
        f.write(f"{len(result.x)} ")
        for i in range(len(result.x)):
            f.write(f"{i} ")
        f.write("\n")
        
        # Point data
        f.write(f"POINT_DATA {len(result.x)}\n")
        
        # d field
        f.write("SCALARS d float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(len(result.x)):
            d_val = d_field[min(i, len(d_field)-1)] if i < len(d_field) else d_field[-1]
            f.write(f"{d_val}\n")
        
        # Temperatures
        for name, data in [
            ("T_hot", result.T_hot),
            ("T_cold", result.T_cold),
            ("T_solid", result.T_solid),
        ]:
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in data:
                f.write(f"{val}\n")
        
        # Pressures
        for name, data in [
            ("P_hot", result.P_hot),
            ("P_cold", result.P_cold),
        ]:
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in data:
                f.write(f"{val}\n")
    
    print(f"Exported VTK data to {path}")


def export_tpms_stl_from_optimization(
    result: MacroModelResult,
    d_field: np.ndarray,
    config: Config,
    filename: str = "optimized_tpms.stl",
    tpms_type=None,  # Will use TPMSType.PRIMITIVE if available
    variant=None,  # Will use VariantMode.SINGLE if available
    cell_size: float = 0.001,
    resolution: int = 50,
) -> str:
    """
    Export optimized TPMS geometry to STL file.
    
    This function takes the optimized d(x) field from the optimizer
    and generates a 3D TPMS lattice mesh, then exports it to STL.
    
    Parameters:
    -----------
    result : MacroModelResult
        Solution from macromodel
    d_field : np.ndarray
        Optimized channel-bias field
    config : Config
        Configuration object
    filename : str
        Output STL filename
    tpms_type : TPMSType, optional
        TPMS lattice type (default: Primitive)
    variant : VariantMode, optional
        Single or double mode (default: single)
    cell_size : float
        Unit cell size in meters (default: 1mm)
    resolution : int
        Grid resolution per unit cell (default: 50)
        
    Returns:
    --------
    str : Path to exported STL file
    """
    if not STL_EXPORT_AVAILABLE:
        raise ImportError(
            "STL export requires numpy-stl and scikit-image. "
            "Install with: pip install numpy-stl scikit-image"
        )
    
    if tpms_type is None:
        tpms_type = TPMSType.PRIMITIVE
    if variant is None:
        variant = VariantMode.SINGLE
    
    return generate_and_export_stl(
        d_field=d_field,
        result=result,
        config=config,
        filename=filename,
        tpms_type=tpms_type,
        variant=variant,
        cell_size=cell_size,
        resolution=resolution,
        n_repeats=(1, 1, 1)  # Can be made configurable
    )

