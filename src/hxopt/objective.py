"""Optimization objective function."""

import numpy as np
from typing import Optional
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
    eps: Optional[float] = None,
) -> np.ndarray:
    """
    Compute gradient of objective w.r.t. d_field using finite differences.
    
    Uses adaptive perturbation size based on d field range to ensure
    meaningful perturbations that don't get clipped.
    
    Parameters
    ----------
    model : MacroModel
        Macromodel instance
    d_field : np.ndarray
        Current channel-bias field
    result : MacroModelResult
        Current solution
    eps : float, optional
        Perturbation size for finite differences. If None, uses adaptive size.
        
    Returns
    -------
    grad : np.ndarray
        Gradient of objective w.r.t. d_field
    """
    n = len(d_field)
    grad = np.zeros(n)
    obj_base = compute_objective(result)
    
    # Adaptive perturbation size: use 1% of d range, but at least 1e-4
    if eps is None:
        d_range = model.config.optimization.d_max - model.config.optimization.d_min
        eps = max(0.01 * d_range, 1e-4)
    
    # Use forward differences (simpler and more reliable)
    # Sample only a subset of points to speed up computation
    # For large n, compute gradient for every k-th point
    sample_stride = max(1, n // min(20, n))  # Sample at most 20 points
    
    for i in range(0, n, sample_stride):
        d_pert = d_field.copy()
        # Try forward perturbation first
        d_pert[i] = min(d_field[i] + eps, model.config.optimization.d_max)
        
        # If forward perturbation doesn't change d, try backward
        if d_pert[i] == d_field[i]:
            d_pert[i] = max(d_field[i] - eps, model.config.optimization.d_min)
        
        # Only compute if perturbation actually changed d
        if d_pert[i] != d_field[i]:
            try:
                result_pert = model.solve(d_pert)
                obj_pert = compute_objective(result_pert)
                delta_d = d_pert[i] - d_field[i]
                grad[i] = (obj_pert - obj_base) / delta_d
            except Exception as e:
                # If solve fails, try a smaller perturbation
                try:
                    small_eps = eps * 0.1
                    d_pert[i] = d_field[i] + small_eps
                    d_pert[i] = np.clip(d_pert[i], model.config.optimization.d_min,
                                       model.config.optimization.d_max)
                    if d_pert[i] != d_field[i]:
                        result_pert = model.solve(d_pert)
                        obj_pert = compute_objective(result_pert)
                        delta_d = d_pert[i] - d_field[i]
                        grad[i] = (obj_pert - obj_base) / delta_d
                    else:
                        grad[i] = 0.0
                except Exception:
                    grad[i] = 0.0
        else:
            grad[i] = 0.0
    
    # Interpolate gradient for skipped points
    if sample_stride > 1:
        # Use linear interpolation for non-sampled points
        sampled_indices = np.arange(0, n, sample_stride)
        sampled_grad = grad[sampled_indices]
        all_indices = np.arange(n)
        # Interpolate
        for i in range(n):
            if i not in sampled_indices:
                # Find nearest sampled points
                left_idx = (i // sample_stride) * sample_stride
                right_idx = min(left_idx + sample_stride, n - 1)
                if right_idx >= n:
                    right_idx = left_idx
                
                if left_idx == right_idx:
                    grad[i] = grad[left_idx]
                else:
                    # Linear interpolation
                    alpha = (i - left_idx) / (right_idx - left_idx) if right_idx > left_idx else 0.0
                    grad[i] = (1 - alpha) * grad[left_idx] + alpha * grad[right_idx]
    
    return grad

