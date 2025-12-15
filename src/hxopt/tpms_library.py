"""
TPMS (Triply Periodic Minimal Surfaces) Library

Implicit function definitions for various TPMS structures.
All functions are unit-consistent and typed.

For heat exchangers, the parameter t controls the isosurface threshold:
- single mode: solid is f(x,y,z) <= t
- double mode: solid is abs(f(x,y,z)) <= t

The optimizer variable d ∈ [0,1] maps to porosity ε(t) via:
- UI controls t
- backend computes porosity ε(t)
- d = ε(t) (or invert ε→t for target porosity control)
"""

import numpy as np
from typing import Literal, Tuple
from enum import Enum


class TPMSType(str, Enum):
    """Supported TPMS lattice types"""
    GYROID = "G"  # Gyroid
    PRIMITIVE = "P"  # Schwarz Primitive
    DIAMOND = "D"  # Schwarz Diamond
    IWP = "W"  # Schoen iWP
    LIDINOID = "L"  # Lidinoid
    NEOVIUS = "N"  # Neovius
    OCTO = "O"  # Octo
    SPLIT_P = "SP"  # Split-P


class VariantMode(str, Enum):
    """TPMS variant modes"""
    SINGLE = "single"  # solid is f(x,y,z) <= t
    DOUBLE = "double"  # solid is abs(f(x,y,z)) <= t


def gyroid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Gyroid (G) implicit function.
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
        
    Returns:
    --------
    np.ndarray : Implicit function values
    """
    return (np.sin(2*np.pi*x) * np.cos(2*np.pi*y) +
            np.sin(2*np.pi*y) * np.cos(2*np.pi*z) +
            np.sin(2*np.pi*z) * np.cos(2*np.pi*x))


def primitive(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Schwarz Primitive (P) implicit function.
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
        
    Returns:
    --------
    np.ndarray : Implicit function values
    """
    return (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z))


def diamond(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Schwarz Diamond (D) implicit function.
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
        
    Returns:
    --------
    np.ndarray : Implicit function values
    """
    return (np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2*np.pi*z) +
            np.sin(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) +
            np.cos(2*np.pi*x) * np.sin(2*np.pi*y) * np.cos(2*np.pi*z) +
            np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(2*np.pi*z))


def iwp(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Schoen I-WP (W) implicit function.
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
        
    Returns:
    --------
    np.ndarray : Implicit function values
    """
    return (2 * (np.cos(2*np.pi*x) * np.cos(2*np.pi*y) +
                 np.cos(2*np.pi*y) * np.cos(2*np.pi*z) +
                 np.cos(2*np.pi*z) * np.cos(2*np.pi*x)) -
            np.cos(4*np.pi*x) - np.cos(4*np.pi*y) - np.cos(4*np.pi*z))


def lidinoid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Lidinoid implicit function.
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
        
    Returns:
    --------
    np.ndarray : Implicit function values
    """
    return (np.sin(2*np.pi*x) * np.cos(2*np.pi*y) +
            np.sin(2*np.pi*y) * np.cos(2*np.pi*z) +
            np.sin(2*np.pi*z) * np.cos(2*np.pi*x) +
            0.5 * (np.cos(4*np.pi*x) * np.cos(4*np.pi*y) +
                   np.cos(4*np.pi*y) * np.cos(4*np.pi*z) +
                   np.cos(4*np.pi*z) * np.cos(4*np.pi*x)))


def neovius(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Neovius implicit function.
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
        
    Returns:
    --------
    np.ndarray : Implicit function values
    """
    return (3 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z)) +
            4 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z))


def octo(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Octo implicit function.
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
        
    Returns:
    --------
    np.ndarray : Implicit function values
    """
    return (np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) +
            np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2*np.pi*z))


def split_p(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Split-P implicit function.
    
    Parameters:
    -----------
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
        
    Returns:
    --------
    np.ndarray : Implicit function values
    """
    return (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z) +
            0.5 * (np.cos(2*np.pi*x) * np.cos(2*np.pi*y) +
                   np.cos(2*np.pi*y) * np.cos(2*np.pi*z) +
                   np.cos(2*np.pi*z) * np.cos(2*np.pi*x)))


# Mapping from TPMS type to function
TPMS_FUNCTIONS = {
    TPMSType.GYROID: gyroid,
    TPMSType.PRIMITIVE: primitive,
    TPMSType.DIAMOND: diamond,
    TPMSType.IWP: iwp,
    TPMSType.LIDINOID: lidinoid,
    TPMSType.NEOVIUS: neovius,
    TPMSType.OCTO: octo,
    TPMSType.SPLIT_P: split_p,
}


def get_tpms_function(tpms_type: TPMSType):
    """Get the implicit function for a TPMS type."""
    return TPMS_FUNCTIONS[tpms_type]


def evaluate_tpms(
    tpms_type: TPMSType,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    variant: VariantMode = VariantMode.SINGLE,
    t: float = 0.0
) -> np.ndarray:
    """
    Evaluate TPMS implicit function with variant mode.
    
    Parameters:
    -----------
    tpms_type : TPMSType
        Type of TPMS lattice
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
    variant : VariantMode
        Single or double mode
    t : float
        Isosurface threshold
        
    Returns:
    --------
    np.ndarray : Implicit function values (or abs values for double mode)
    """
    func = get_tpms_function(tpms_type)
    f = func(x, y, z)
    
    if variant == VariantMode.DOUBLE:
        f = np.abs(f)
    
    return f


def get_solid_mask(
    tpms_type: TPMSType,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    variant: VariantMode,
    t: float
) -> np.ndarray:
    """
    Get boolean mask for solid regions.
    
    Parameters:
    -----------
    tpms_type : TPMSType
        Type of TPMS lattice
    x, y, z : np.ndarray
        Coordinate arrays (unit: m)
    variant : VariantMode
        Single or double mode
    t : float
        Isosurface threshold
        
    Returns:
    --------
    np.ndarray : Boolean mask (True = solid, False = void)
    """
    f = evaluate_tpms(tpms_type, x, y, z, variant, t)
    
    if variant == VariantMode.SINGLE:
        return f <= t
    else:  # DOUBLE
        return np.abs(f) <= t


# Recommended t ranges for each TPMS type (for single mode)
# These are approximate ranges that give reasonable porosity values
RECOMMENDED_T_RANGES = {
    TPMSType.GYROID: (-1.5, 1.5),
    TPMSType.PRIMITIVE: (-2.0, 2.0),
    TPMSType.DIAMOND: (-1.5, 1.5),
    TPMSType.IWP: (-3.0, 3.0),
    TPMSType.LIDINOID: (-2.0, 2.0),
    TPMSType.NEOVIUS: (-5.0, 5.0),
    TPMSType.OCTO: (-1.5, 1.5),
    TPMSType.SPLIT_P: (-2.5, 2.5),
}

