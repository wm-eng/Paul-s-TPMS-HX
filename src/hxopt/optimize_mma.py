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
        # Use relative tolerance: 0.1% change or absolute 100 W
        if it > 0:
            Q_prev = opt_result.Q_values[-2]
            Q_curr = opt_result.Q_values[-1]
            rel_change = abs(Q_curr - Q_prev) / max(abs(Q_prev), 1.0)
            if rel_change < 1e-3 or abs(Q_curr - Q_prev) < 100.0:
                print(f"Converged at iteration {it} (Q change: {rel_change*100:.3f}%)")
                break
        
        # Compute gradient
        if it < config.optimization.max_iter - 1:
            print(f"  Computing gradient...")
            grad = compute_objective_gradient(model, d_field, result)
            
            # Check gradient magnitude
            grad_norm = np.linalg.norm(grad)
            grad_max = np.max(np.abs(grad))
            grad_mean = np.mean(np.abs(grad))
            
            print(f"  Gradient: norm={grad_norm:.2e}, max={grad_max:.2e}, mean={grad_mean:.2e}")
            
            if grad_norm < 1e-10:
                print(f"  Warning: Gradient is essentially zero (norm={grad_norm:.2e})")
                # Try a larger perturbation to test sensitivity
                print(f"  Testing sensitivity with larger perturbation...")
                test_d = d_field.copy()
                test_d[0] = min(test_d[0] + 0.1, config.optimization.d_max)
                try:
                    test_result = model.solve(test_d)
                    test_Q = test_result.Q
                    Q_diff = abs(test_Q - Q)
                    print(f"  Q change for d[0] += 0.1: {Q_diff:.2e} W (Q_base={Q:.2e} W, Q_test={test_Q:.2e} W)")
                    if Q_diff < 1.0:  # Less than 1 W change
                        print(f"  Model appears insensitive to d changes. This may indicate:")
                        print(f"    - RVE properties are constant or nearly constant")
                        print(f"    - Constraints are dominating the solution")
                        print(f"    - Solver convergence issues")
                except Exception as e:
                    print(f"  Test solve failed: {e}")
                
                # Try a random perturbation to escape flat region
                if it == 0:
                    print(f"  Trying random perturbation to escape flat region...")
                    d_field = d_field + 0.05 * (np.random.rand(n) - 0.5)
                    d_field = np.clip(d_field, config.optimization.d_min, config.optimization.d_max)
                    continue
                else:
                    print(f"  Stopping optimization due to zero gradient")
                    break
            
            # Since objective = -Q, grad is gradient of -Q
            # To maximize Q, we step in direction of -grad (which is gradient of Q)
            # So: d_new = d_old - step_size * grad (for maximization)
            # But we can also think: step in direction opposite to objective gradient
            search_dir = -grad  # Step in direction to maximize Q
            
            # Normalize search direction to prevent very large steps
            if grad_norm > 0:
                search_dir = search_dir / grad_norm  # Unit direction
                # Scale by typical d range to get reasonable step sizes
                d_range = config.optimization.d_max - config.optimization.d_min
                search_dir = search_dir * d_range  # Scale to d range
            
            # Projected gradient step with line search
            step_size = config.optimization.step_size
            alpha = config.optimization.line_search_alpha
            beta = config.optimization.line_search_beta
            
            # Backtracking line search
            ls_success = False
            for ls_it in range(20):
                d_new = d_field + step_size * search_dir
                d_new = np.clip(d_new, config.optimization.d_min, config.optimization.d_max)
                
                # Test step
                try:
                    result_new = model.solve(d_new)
                    Q_new = result_new.Q
                    Q_old = Q
                    
                    # For maximization: accept if Q increases
                    # Armijo condition for maximization: Q_new >= Q_old + c1 * step_size * grad_Q^T * search_dir
                    # Since search_dir = -grad_objective = -grad(-Q) = grad(Q)
                    # We want: Q_new >= Q_old + c1 * step_size * ||grad_Q||²
                    # Or equivalently: Q_new - Q_old >= c1 * step_size * ||grad_Q||²
                    improvement = Q_new - Q_old
                    grad_Q_norm_sq = np.dot(grad, grad)  # ||grad(-Q)||² = ||grad(Q)||²
                    expected_improvement = alpha * step_size * grad_Q_norm_sq
                    
                    # Accept step if we get sufficient improvement
                    if improvement >= expected_improvement or improvement > 0:
                        d_field = d_new
                        ls_success = True
                        break
                except Exception as e:
                    # If solve fails, reduce step size
                    pass
                
                step_size *= beta
            
            # If line search failed, try a very small step
            if not ls_success:
                small_step = 0.01 * config.optimization.step_size
                d_field = np.clip(
                    d_field + small_step * search_dir,
                    config.optimization.d_min,
                    config.optimization.d_max
                )
                print(f"  Line search failed, using small step: {small_step:.6f}")
    
    if log_fp:
        log_fp.close()
    
    print("Optimization complete.")
    return opt_result

