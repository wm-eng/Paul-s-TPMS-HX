"""Validation functions for energy balance and bounds checking."""

import numpy as np
from .macro_model import MacroModelResult
from .config import Config


def validate_energy_balance(
    result: MacroModelResult,
    config: Config,
    tol: float = 1e-3,
) -> tuple[bool, float]:
    """
    Validate energy conservation.
    
    Check: Q_hot = m_dot_hot * cp_hot * (T_hot_in - T_hot_out)
           Q_cold = m_dot_cold * cp_cold * (T_cold_out - T_cold_in)
           Q_hot ≈ Q_cold (within tolerance)
    
    Parameters
    ----------
    result : MacroModelResult
        Solution from macromodel
    config : Config
        Configuration object
    tol : float
        Relative tolerance for energy balance
        
    Returns
    -------
    valid : bool
        True if energy balance satisfied
    error : float
        Relative error in energy balance
    """
    # Hot side heat transfer
    cp_hot = config.fluid.cp_hot
    Q_hot = config.fluid.m_dot_hot * cp_hot * (
        config.fluid.T_hot_in - result.T_hot[-1]
    )
    
    # Cold side heat transfer
    # Cold inlet is at x=L (index -1), outlet at x=0 (index 0)
    cp_cold = config.fluid.cp_cold
    Q_cold = config.fluid.m_dot_cold * cp_cold * (
        result.T_cold[0] - result.T_cold[-1]
    )
    
    # Check balance
    Q_avg = (abs(Q_hot) + abs(Q_cold)) / 2.0
    if Q_avg < 1e-6:
        return True, 0.0
    
    error = abs(Q_hot - Q_cold) / Q_avg
    valid = error < tol
    
    return valid, error


def validate_bounds(
    result: MacroModelResult,
    config: Config,
) -> tuple[bool, list[str]]:
    """
    Validate solution bounds and physical constraints.
    
    Parameters
    ----------
    result : MacroModelResult
        Solution from macromodel
    config : Config
        Configuration object
        
    Returns
    -------
    valid : bool
        True if all bounds satisfied
    errors : list[str]
        List of error messages
    """
    errors = []
    
    # Temperature bounds
    if np.any(result.T_hot < 0) or np.any(result.T_hot > 1000):
        errors.append("Hot side temperature out of bounds")
    if np.any(result.T_cold < 0) or np.any(result.T_cold > 100):
        errors.append("Cold side temperature out of bounds")
    if np.any(result.T_solid < 0) or np.any(result.T_solid > 1000):
        errors.append("Solid temperature out of bounds")
    
    # Pressure bounds
    if np.any(result.P_hot < 0):
        errors.append("Hot side pressure negative")
    if np.any(result.P_cold < 0):
        errors.append("Cold side pressure negative")
    
    # Monotonicity checks
    if not np.all(np.diff(result.T_hot) <= 0):
        errors.append("Hot side temperature not monotonically decreasing")
    if not np.all(np.diff(result.T_cold) >= 0):
        errors.append("Cold side temperature not monotonically increasing")
    
    return len(errors) == 0, errors


def validate_all(
    result: MacroModelResult,
    config: Config,
) -> bool:
    """
    Run all validation checks and fail fast if any fail.
    
    Parameters
    ----------
    result : MacroModelResult
        Solution from macromodel
    config : Config
        Configuration object
        
    Returns
    -------
    valid : bool
        True if all checks pass
        
    Raises
    ------
    ValueError
        If validation fails
    """
    # Energy balance
    energy_valid, energy_error = validate_energy_balance(result, config)
    if not energy_valid:
        raise ValueError(f"Energy balance violated: relative error = {energy_error:.2e}")
    
    # Bounds
    bounds_valid, errors = validate_bounds(result, config)
    if not bounds_valid:
        raise ValueError(f"Bounds violated: {', '.join(errors)}")
    
    return True

