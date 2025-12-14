"""Constraint functions for optimization."""

from typing import NamedTuple
import numpy as np
from .macro_model import MacroModelResult
from .materials import ConstantProperties
from .config import Config


class ConstraintViolation(NamedTuple):
    """Constraint violation information."""
    pressure_drop_hot: float  # Violation amount (positive = violated)
    pressure_drop_cold: float
    min_subcooling_margin: float  # Negative = violated
    smoothness: float  # Smoothness penalty value


def check_constraints(
    result: MacroModelResult,
    config: Config,
    props_cold: ConstantProperties,
    d_field: np.ndarray,
) -> ConstraintViolation:
    """
    Check all constraints and return violation amounts.
    
    Parameters
    ----------
    result : MacroModelResult
        Solution from macromodel
    config : Config
        Configuration object
    props_cold : ConstantProperties
        Cold side material properties
    d_field : np.ndarray
        Channel-bias field
        
    Returns
    -------
    violations : ConstraintViolation
        Constraint violation amounts
    """
    # Pressure drop constraints
    delta_P_hot_violation = 0.0
    if config.optimization.delta_P_max_hot is not None:
        excess = result.delta_P_hot - config.optimization.delta_P_max_hot
        delta_P_hot_violation = max(0.0, excess)
    
    delta_P_cold_violation = 0.0
    if config.optimization.delta_P_max_cold is not None:
        excess = result.delta_P_cold - config.optimization.delta_P_max_cold
        delta_P_cold_violation = max(0.0, excess)
    
    # Subcooling margin constraint
    # T_LH2(x) <= T_sat(P_LH2(x)) - margin
    T_sat = props_cold.saturation_temperature(result.P_cold)
    margin = result.T_cold - (T_sat - config.optimization.T_sat_margin)
    min_subcooling_margin = np.min(margin)  # Negative = violated
    
    # Smoothness penalty (v1: simple gradient penalty)
    smoothness = 0.0
    if config.optimization.smoothness_penalty > 0:
        grad_d = np.diff(d_field)
        smoothness = config.optimization.smoothness_penalty * np.sum(grad_d ** 2)
    
    return ConstraintViolation(
        pressure_drop_hot=delta_P_hot_violation,
        pressure_drop_cold=delta_P_cold_violation,
        min_subcooling_margin=min_subcooling_margin,
        smoothness=smoothness,
    )


def is_feasible(violations: ConstraintViolation, tol: float = 1e-6) -> bool:
    """
    Check if solution is feasible.
    
    Parameters
    ----------
    violations : ConstraintViolation
        Constraint violations
    tol : float
        Tolerance for constraint satisfaction
        
    Returns
    -------
    feasible : bool
        True if all constraints satisfied
    """
    return (
        violations.pressure_drop_hot <= tol and
        violations.pressure_drop_cold <= tol and
        violations.min_subcooling_margin >= -tol
    )

