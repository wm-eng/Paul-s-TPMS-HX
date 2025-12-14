"""Optimization loop using projected gradient ascent with line search."""

import numpy as np
import os
import csv
from typing import Optional
from .config import Config
from .rve_db import RVEDatabase
from .macro_model import MacroModel
from .objective import compute_objective, compute_objective_gradient
from .constraints import check_constraints, is_feasible
from .materials import ConstantProperties


class OptimizationResult:
    """Results from optimization."""
    
    def __init__(self):
        self.iterations = []
        self.d_fields = []
        self.objectives = []
        self.Q_values = []
        self.delta_P_hot = []
        self.delta_P_cold = []
        self.min_subcooling_margin = []
        self.feasible = []
    
    def add_iteration(
        self,
        iter: int,
        d_field: np.ndarray,
        objective: float,
        Q: float,
        delta_P_hot: float,
        delta_P_cold: float,
        min_subcooling_margin: float,
        feasible: bool,
    ):
        """Add iteration data."""
        self.iterations.append(iter)
        self.d_fields.append(d_field.copy())
        self.objectives.append(objective)
        self.Q_values.append(Q)
        self.delta_P_hot.append(delta_P_hot)
        self.delta_P_cold.append(delta_P_cold)
        self.min_subcooling_margin.append(min_subcooling_margin)
        self.feasible.append(feasible)


def optimize(
    config: Config,
    rve_db: RVEDatabase,
    d_init: Optional[np.ndarray] = None,
    log_file: Optional[str] = None,
) -> OptimizationResult:
    """
    Optimize channel-bias field d(x) to maximize heat transfer.
    
    Uses projected gradient ascent with backtracking line search.
    
    Parameters
    ----------
    config : Config
        Configuration object
    rve_db : RVEDatabase
        RVE property database
    d_init : np.ndarray, optional
        Initial d field. If None, uses constant from config.
    log_file : str, optional
        Path to CSV log file
        
    Returns
    -------
    result : OptimizationResult
        Optimization history and final solution
    """
    # Initialize
    model = MacroModel(config, rve_db)
    n = config.geometry.n_segments
    
    if d_init is None:
        d_field = np.full(n, config.optimization.d_init)
    else:
        d_field = np.clip(d_init, config.optimization.d_min, config.optimization.d_max)
    
    # Create material properties for constraint checking
    props_cold = ConstantProperties(
        rho=config.fluid.rho_cold,
        mu=config.fluid.mu_cold,
        cp=config.fluid.cp_cold,
        k=config.fluid.k_cold,
        T_sat_ref=20.0,
        dT_sat_dP=1e-5,
    )
    
    # Initialize result tracking
    opt_result = OptimizationResult()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Open log file
    log_fp = None
    if log_file:
        log_path = os.path.join(config.output_dir, log_file)
        log_fp = open(log_path, 'w', newline='')
        writer = csv.writer(log_fp)
        writer.writerow([
            'iter', 'Q', 'objective', 'delta_P_hot', 'delta_P_cold',
            'min_subcooling_margin', 'feasible'
        ])
    
    print("Starting optimization...")
    print(f"Initial d: {d_field[0]:.3f} (constant)")
    
    # Main optimization loop
    for it in range(config.optimization.max_iter):
        # Solve macromodel
        try:
            result = model.solve(d_field)
        except Exception as e:
            print(f"Iteration {it}: Solve failed: {e}")
            break
        
        # Compute objective
        objective = compute_objective(result)
        Q = result.Q
        
        # Check constraints
        violations = check_constraints(result, config, props_cold, d_field)
        feasible = is_feasible(violations)
        
        # Log iteration
        opt_result.add_iteration(
            iter=it,
            d_field=d_field,
            objective=objective,
            Q=Q,
            delta_P_hot=result.delta_P_hot,
            delta_P_cold=result.delta_P_cold,
            min_subcooling_margin=violations.min_subcooling_margin,
            feasible=feasible,
        )
        
        # Print progress
        print(f"Iter {it:3d}: Q={Q/1e6:.3f} MW, "
              f"ΔP_hot={result.delta_P_hot/1e3:.2f} kPa, "
              f"ΔP_cold={result.delta_P_cold/1e3:.2f} kPa, "
              f"min_margin={violations.min_subcooling_margin:.2f} K, "
              f"feasible={feasible}")
        
        # Write to log
        if log_fp:
            writer.writerow([
                it, Q, objective, result.delta_P_hot, result.delta_P_cold,
                violations.min_subcooling_margin, feasible
            ])
            log_fp.flush()
        
        # Check convergence (simple: no improvement)
        if it > 0 and abs(opt_result.Q_values[-1] - opt_result.Q_values[-2]) < 1e3:
            print(f"Converged at iteration {it}")
            break
        
        # Compute gradient
        if it < config.optimization.max_iter - 1:
            grad = compute_objective_gradient(model, d_field, result)
            
            # Projected gradient step with line search
            step_size = config.optimization.step_size
            alpha = config.optimization.line_search_alpha
            beta = config.optimization.line_search_beta
            
            # Backtracking line search
            for ls_it in range(20):
                d_new = d_field + step_size * grad
                d_new = np.clip(d_new, config.optimization.d_min, config.optimization.d_max)
                
                # Test step
                try:
                    result_new = model.solve(d_new)
                    objective_new = compute_objective(result_new)
                    
                    # Armijo condition: f(x + αp) <= f(x) + c1 * α * grad^T * p
                    # For maximization (minimize -Q), we want: -Q_new <= -Q_old + c1 * α * grad^T * grad
                    # Or: Q_new >= Q_old - c1 * α * ||grad||²
                    improvement = objective_new - objective
                    expected_improvement = alpha * step_size * np.dot(grad, grad)
                    
                    if improvement <= expected_improvement:
                        d_field = d_new
                        break
                except Exception:
                    pass
                
                step_size *= beta
            
            # If line search failed, use small step anyway
            if ls_it == 19:
                d_field = np.clip(
                    d_field + 0.1 * step_size * grad,
                    config.optimization.d_min,
                    config.optimization.d_max
                )
    
    if log_fp:
        log_fp.close()
    
    print("Optimization complete.")
    return opt_result

