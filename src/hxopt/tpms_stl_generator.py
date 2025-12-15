"""
TPMS STL Generator

Generates STL files from optimized d(x) field using TPMS implicit functions.
Maps d ∈ [0,1] to isosurface threshold t and generates 3D mesh via marching cubes.

This replaces typical MATLAB workflows with a pure Python implementation.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from skimage import measure
from stl import mesh
import os
import gc

from .tpms_library import (
    TPMSType,
    VariantMode,
    evaluate_tpms,
    get_solid_mask,
    RECOMMENDED_T_RANGES
)
from .macro_model import MacroModelResult
from .config import Config


def d_to_porosity_mapping(
    d: np.ndarray,
    tpms_type: TPMSType = TPMSType.PRIMITIVE,
    variant: VariantMode = VariantMode.SINGLE,
    n_samples: int = 100
) -> Dict[str, np.ndarray]:
    """
    Create mapping from d ∈ [0,1] to porosity ε and threshold t.
    
    This function samples the TPMS function at various t values to build
    a lookup table: d → ε(t) → t.
    
    Parameters:
    -----------
    d : np.ndarray
        Channel-bias values (will be used as target porosities)
    tpms_type : TPMSType
        TPMS lattice type
    variant : VariantMode
        Single or double mode
    n_samples : int
        Number of t values to sample for lookup table
        
    Returns:
    --------
    dict : Mapping dictionary with keys:
        - 'd': d values (input)
        - 't': corresponding threshold values
        - 'porosity': actual porosities at those t values
    """
    # Get recommended t range
    t_min, t_max = RECOMMENDED_T_RANGES[tpms_type]
    t_samples = np.linspace(t_min, t_max, n_samples)
    
    # Sample porosity at each t
    # Use a small unit cell for fast sampling
    n_grid = 50
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    z = np.linspace(0, 1, n_grid)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    porosities = []
    for t in t_samples:
        solid_mask = get_solid_mask(tpms_type, X, Y, Z, variant, t)
        porosity = 1.0 - np.sum(solid_mask) / solid_mask.size
        porosities.append(porosity)
    
    porosities = np.array(porosities)
    
    # Interpolate d → t (assuming d ≈ porosity)
    # For each d value, find corresponding t
    t_values = np.interp(d, porosities[::-1], t_samples[::-1])
    
    # Handle edge cases (d outside porosity range)
    t_values = np.clip(t_values, t_min, t_max)
    
    return {
        'd': d,
        't': t_values,
        'porosity': porosities
    }


def generate_tpms_mesh_from_d_field(
    d_field: np.ndarray,
    config: Config,
    tpms_type: TPMSType = TPMSType.PRIMITIVE,
    variant: VariantMode = VariantMode.SINGLE,
    cell_size: float = 0.001,  # m, size of unit cell
    resolution: int = 50,  # grid points per unit cell
    n_repeats: Tuple[int, int, int] = (1, 1, 1),  # repeats in x, y, z
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D TPMS mesh from optimized d(x) field.
    
    This is the main function that converts the 1D optimized d field
    into a 3D TPMS lattice structure.
    
    Parameters:
    -----------
    d_field : np.ndarray
        Optimized channel-bias field (1D along flow direction)
    config : Config
        Configuration object (contains geometry info)
    tpms_type : TPMSType
        TPMS lattice type
    variant : VariantMode
        Single or double mode
    cell_size : float
        Size of TPMS unit cell in meters
    resolution : int
        Grid resolution per unit cell (higher = smoother)
    n_repeats : tuple
        Number of unit cell repeats in (x, y, z) directions
        
    Returns:
    --------
    vertices : np.ndarray
        Mesh vertices (N, 3)
    faces : np.ndarray
        Mesh faces (M, 3) - triangle indices
    """
    # Get geometry dimensions
    L = config.geometry.length  # Flow direction length
    W = config.geometry.width   # Width
    H = config.geometry.height  # Height
    
    # Create mapping from d to t
    d_unique = np.unique(d_field)
    mapping = d_to_porosity_mapping(d_unique, tpms_type, variant)
    d_to_t = dict(zip(mapping['d'], mapping['t']))
    
    # Create 3D grid
    # X is flow direction (along length)
    # Y is width direction
    # Z is height direction
    n_x = int(L / cell_size * resolution)
    n_y = int(W / cell_size * resolution)
    n_z = int(H / cell_size * resolution)
    
    # Limit grid size to prevent memory exhaustion
    max_points = 50_000_000  # 50M points max
    total_points = n_x * n_y * n_z
    if total_points > max_points:
        # Scale down resolution proportionally
        scale = (max_points / total_points) ** (1/3)
        n_x = max(10, int(n_x * scale))
        n_y = max(10, int(n_y * scale))
        n_z = max(10, int(n_z * scale))
    
    x = np.linspace(0, L, n_x)
    y = np.linspace(0, W, n_y)
    z = np.linspace(0, H, n_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Map d field to 3D space
    # Interpolate d_field to grid x positions
    d_3d = np.interp(X.flatten(), np.linspace(0, L, len(d_field)), d_field)
    d_3d = d_3d.reshape(X.shape)
    
    # Convert d to t for each point using vectorized lookup
    # Build sorted arrays for efficient lookup
    d_keys = np.array(sorted(d_to_t.keys()))
    t_values = np.array([d_to_t[d] for d in d_keys])
    
    # Vectorized nearest neighbor lookup (much faster than triple loop)
    # Find indices of closest d values
    indices = np.searchsorted(d_keys, d_3d, side='left')
    # Handle edge cases
    indices = np.clip(indices, 0, len(d_keys) - 1)
    # Check if left or right neighbor is closer
    left_diff = np.abs(d_3d - d_keys[np.clip(indices - 1, 0, len(d_keys) - 1)])
    right_diff = np.abs(d_3d - d_keys[indices])
    indices = np.where((indices > 0) & (left_diff < right_diff), indices - 1, indices)
    # Map to t values
    t_3d = t_values[indices]
    
    # Clean up intermediate arrays
    del d_keys, t_values, indices, left_diff, right_diff
    
    # Normalize coordinates for TPMS functions (unit period)
    X_norm = X / cell_size
    Y_norm = Y / cell_size
    Z_norm = Z / cell_size
    
    # Clean up original meshgrids before creating new arrays
    del X, Y, Z
    import gc
    gc.collect()
    
    # Evaluate TPMS function
    f = evaluate_tpms(tpms_type, X_norm, Y_norm, Z_norm, variant, 0.0)
    
    # Clean up normalized coordinates
    del X_norm, Y_norm, Z_norm
    gc.collect()
    
    # Create isosurface field: f - t
    # Solid is where f <= t (single) or |f| <= t (double)
    if variant == VariantMode.SINGLE:
        iso_field = f - t_3d
        level = 0.0
    else:  # DOUBLE
        iso_field = np.abs(f) - t_3d
        level = 0.0
    
    # Clean up before marching cubes
    del f, t_3d, d_3d
    gc.collect()
    
    # Marching cubes
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    result = measure.marching_cubes(iso_field, level=level, spacing=spacing)
    
    # Clean up iso_field after marching cubes
    del iso_field
    gc.collect()
    
    # Handle different return formats
    if isinstance(result, tuple):
        vertices, faces = result[0], result[1]
    else:
        vertices = result.vertices
        faces = result.faces
    
    return vertices, faces


def export_tpms_stl(
    vertices: np.ndarray,
    faces: np.ndarray,
    filename: str,
    output_dir: Optional[str] = None
) -> str:
    """
    Export TPMS mesh to STL file.
    
    Parameters:
    -----------
    vertices : np.ndarray
        Mesh vertices (N, 3)
    faces : np.ndarray
        Mesh faces (M, 3) - triangle vertex indices
    filename : str
        Output filename (should end in .stl)
    output_dir : str, optional
        Output directory (default: current directory)
        
    Returns:
    --------
    str : Full path to exported STL file
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    # Create STL mesh
    num_faces = len(faces)
    stl_data = np.zeros(num_faces, dtype=mesh.Mesh.dtype)
    
    for i, face in enumerate(faces):
        # Get triangle vertices
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Compute normal
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        # Store in STL format
        stl_data['vectors'][i][0] = v0
        stl_data['vectors'][i][1] = v1
        stl_data['vectors'][i][2] = v2
        stl_data['normals'][i] = normal
    
    stl_mesh = mesh.Mesh(stl_data)
    stl_mesh.save(filepath)
    
    print(f"Exported STL to {filepath}")
    return filepath


def generate_and_export_stl(
    d_field: np.ndarray,
    result: MacroModelResult,
    config: Config,
    filename: str = "tpms_lattice.stl",
    tpms_type: TPMSType = TPMSType.PRIMITIVE,
    variant: VariantMode = VariantMode.SINGLE,
    cell_size: float = 0.001,
    resolution: int = 50,
    n_repeats: Tuple[int, int, int] = (1, 1, 1),
) -> str:
    """
    Complete workflow: generate TPMS mesh from d field and export to STL.
    
    This is the main convenience function that combines mesh generation
    and STL export in one call.
    
    Parameters:
    -----------
    d_field : np.ndarray
        Optimized channel-bias field
    result : MacroModelResult
        Solution result (for geometry info)
    config : Config
        Configuration object
    filename : str
        Output STL filename
    tpms_type : TPMSType
        TPMS lattice type
    variant : VariantMode
        Single or double mode
    cell_size : float
        Unit cell size in meters
    resolution : int
        Grid resolution per unit cell
    n_repeats : tuple
        Repeats in (x, y, z)
        
    Returns:
    --------
    str : Path to exported STL file
    """
    # Generate mesh
    vertices, faces = generate_tpms_mesh_from_d_field(
        d_field=d_field,
        config=config,
        tpms_type=tpms_type,
        variant=variant,
        cell_size=cell_size,
        resolution=resolution,
        n_repeats=n_repeats
    )
    
    # Export to STL
    filepath = export_tpms_stl(
        vertices=vertices,
        faces=faces,
        filename=filename,
        output_dir=config.output_dir
    )
    
    # Clean up mesh data after export
    del vertices, faces
    gc.collect()
    
    return filepath

