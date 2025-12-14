"""Optimization objective function."""

import numpy as np
from .macro_model import MacroModel, MacroModelResult


def compute_objective(result: MacroModelResult) -> float:
    """
    Compute objective value (maximize Q = heat transfer rate).
    
    For minimization, return -Q.
    
    Parameters
    ----------
    result : MacroModelResult
        Solution from macromodel
        
    Returns
    -------
    objective : float
        Objective value (negative heat transfer for minimization)
    """
    # Maximize Q, so minimize -Q
    return -result.Q


def compute_objective_gradient(
    model: MacroModel,
    d_field: np.ndarray,
    result: MacroModelResult,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute gradient of objective w.r.t. d_field using finite differences.
    
    Parameters
    ----------
    model : MacroModel
        Macromodel instance
    d_field : np.ndarray
        Current channel-bias field
    result : MacroModelResult
        Current solution
    eps : float
        Perturbation size for finite differences
        
    Returns
    -------
    grad : np.ndarray
        Gradient of objective w.r.t. d_field
    """
    n = len(d_field)
    grad = np.zeros(n)
    obj_base = compute_objective(result)
    
    for i in range(n):
        d_pert = d_field.copy()
        d_pert[i] += eps
        # Clamp to valid range
        d_pert[i] = np.clip(d_pert[i], model.config.optimization.d_min,
                           model.config.optimization.d_max)
        
        result_pert = model.solve(d_pert)
        obj_pert = compute_objective(result_pert)
        
        grad[i] = (obj_pert - obj_base) / eps
    
    return grad

