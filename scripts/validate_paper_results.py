"""
Validate optimization results against Yanagihara et al. (2025) paper.

The paper reports 28.7% average enhancement over uniform lattice.
This script:
1. Runs optimization with uniform d (baseline)
2. Runs optimization with variable d (optimized)
3. Compares improvement percentage

Usage:
    python scripts/validate_paper_results.py

Note: Requires the package to be installed:
    pip install -e .
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.config import Config, GeometryConfig, FluidConfig, OptimizationConfig
from hxopt.rve_db import RVEDatabase
from hxopt.macro_model import MacroModel
from hxopt.optimize_mma import optimize


def run_baseline_uniform(config: Config, rve_db: RVEDatabase, d_value: float = 0.5):
    """
    Run baseline case with uniform d field.
    
    Parameters
    ----------
    config : Config
        Configuration object
    rve_db : RVEDatabase
        RVE property database
    d_value : float
        Uniform d value (default: 0.5)
        
    Returns
    -------
    result : MacroModelResult
        Solution result
    """
    print(f"\n{'='*60}")
    print(f"BASELINE: Uniform d = {d_value:.2f}")
    print(f"{'='*60}")
    
    model = MacroModel(config, rve_db)
    d_uniform = np.full(config.geometry.n_segments, d_value)
    
    try:
        result = model.solve(d_uniform)
        print(f"Q = {result.Q/1e6:.3f} MW")
        print(f"ΔP_hot = {result.delta_P_hot/1e3:.2f} kPa")
        print(f"ΔP_cold = {result.delta_P_cold/1e3:.2f} kPa")
        return result
    except Exception as e:
        print(f"ERROR: Baseline solve failed: {e}")
        return None


def run_optimized(config: Config, rve_db: RVEDatabase, max_iter: int = 20):
    """
    Run optimization with variable d field.
    
    Parameters
    ----------
    config : Config
        Configuration object
    rve_db : RVEDatabase
        RVE property database
    max_iter : int
        Maximum optimization iterations
        
    Returns
    -------
    opt_result : OptimizationResult
        Optimization result
    """
    print(f"\n{'='*60}")
    print(f"OPTIMIZED: Variable d field (max_iter={max_iter})")
    print(f"{'='*60}")
    
    # Temporarily update max_iter
    original_max_iter = config.optimization.max_iter
    config.optimization.max_iter = max_iter
    
    try:
        opt_result = optimize(config, rve_db, log_file=None)
        return opt_result
    except Exception as e:
        print(f"ERROR: Optimization failed: {e}")
        return None
    finally:
        config.optimization.max_iter = original_max_iter


def compare_results(baseline_result, optimized_result, paper_improvement: float = 0.287):
    """
    Compare baseline vs optimized results and compare with paper.
    
    Parameters
    ----------
    baseline_result : MacroModelResult
        Baseline (uniform d) result
    optimized_result : OptimizationResult or MacroModelResult
        Optimized result (either OptimizationResult with Q_values or MacroModelResult with Q)
    paper_improvement : float
        Paper-reported improvement (28.7% = 0.287)
    """
    if baseline_result is None or optimized_result is None:
        print("\nERROR: Cannot compare - one or both results are None")
        return
    
    Q_baseline = baseline_result.Q
    
    # Handle both OptimizationResult and MacroModelResult
    if hasattr(optimized_result, 'Q_values'):
        # It's an OptimizationResult
        if not optimized_result.Q_values:
            print("\nERROR: No optimization iterations completed")
            return
        Q_optimized = optimized_result.Q_values[-1]
    elif hasattr(optimized_result, 'Q'):
        # It's a MacroModelResult
        Q_optimized = optimized_result.Q
    else:
        print("\nERROR: optimized_result must be OptimizationResult or MacroModelResult")
        return
    
    improvement = (Q_optimized - Q_baseline) / Q_baseline * 100.0
    
    print(f"\n{'='*60}")
    print("COMPARISON WITH PAPER")
    print(f"{'='*60}")
    print(f"Baseline (uniform d):     Q = {Q_baseline/1e6:.3f} MW")
    print(f"Optimized (variable d):   Q = {Q_optimized/1e6:.3f} MW")
    print(f"Improvement:              {improvement:.2f}%")
    print(f"Paper reported:          {paper_improvement*100:.1f}%")
    
    if improvement > 0:
        print(f"\n✅ Optimization shows {improvement:.2f}% improvement")
        if improvement >= paper_improvement * 100 * 0.5:  # At least 50% of paper's improvement
            print(f"   (Within reasonable range of paper's {paper_improvement*100:.1f}%)")
        else:
            print(f"   (Lower than paper's {paper_improvement*100:.1f}% - may need tuning)")
    else:
        print(f"\n⚠️  Optimization did not improve over baseline")
        print(f"   This may indicate:")
        print(f"   - Optimization needs more iterations")
        print(f"   - Constraints are too restrictive")
        print(f"   - Initial guess needs adjustment")


def main():
    """Main validation function."""
    
    print("="*60)
    print("VALIDATION AGAINST YANAGIHARA ET AL. (2025)")
    print("="*60)
    print("\nPaper reports: 28.7% average enhancement over uniform lattice")
    print("This script validates our implementation against this claim.\n")
    
    # Geometry (similar to paper's planar heat exchanger)
    geometry = GeometryConfig(
        length=0.5,  # m
        width=0.1,   # m
        height=0.1,  # m
        n_segments=50,
    )
    
    # Fluid properties (helium/hydrogen, similar to paper)
    fluid = FluidConfig(
        rho_hot=0.1786,  # kg/m³ (helium at 300K)
        mu_hot=2.0e-5,   # Pa·s
        cp_hot=5190.0,   # J/(kg·K)
        k_hot=0.152,     # W/(m·K)
        
        rho_cold=70.8,   # kg/m³ (LH2 at 20K)
        mu_cold=1.3e-4,  # Pa·s
        cp_cold=9600.0,  # J/(kg·K)
        k_cold=0.1,      # W/(m·K)
        
        T_hot_in=300.0,  # K
        T_cold_in=20.0,  # K
        P_hot_in=2e5,    # Pa
        P_cold_in=1e5,   # Pa
        m_dot_hot=0.01,  # kg/s
        m_dot_cold=0.05, # kg/s
    )
    
    # Optimization settings
    # Use more iterations and larger step size for better convergence
    optimization = OptimizationConfig(
        max_iter=50,  # Increased from 20
        d_min=0.1,
        d_max=0.9,
        d_init=0.5,
        delta_P_max_hot=10e3,   # 10 kPa
        delta_P_max_cold=5e3,   # 5 kPa
        T_sat_margin=5.0,        # K
        step_size=0.1,  # Increased from 0.01 for faster convergence
    )
    
    # Create config
    config = Config(
        geometry=geometry,
        fluid=fluid,
        optimization=optimization,
        rve_table_path=os.path.join(
            os.path.dirname(__file__), '..', 'data', 'rve_tables', 'primitive_default.csv'
        ),
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'paper_validation'),
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load RVE database
    print("Loading RVE database...")
    if not os.path.exists(config.rve_table_path):
        print(f"ERROR: RVE table not found: {config.rve_table_path}")
        print("Please ensure the RVE table exists or update the path.")
        return 1
    
    rve_db = RVEDatabase(config.rve_table_path)
    print(f"RVE database loaded: d range [{rve_db.d_min:.2f}, {rve_db.d_max:.2f}]")
    
    # Run baseline (uniform d)
    baseline_result = run_baseline_uniform(config, rve_db, d_value=0.5)
    
    # Run optimization
    optimized_result = run_optimized(config, rve_db, max_iter=50)
    
    # Compare results
    if optimized_result and optimized_result.d_fields:
        # Get final optimized solution
        model = MacroModel(config, rve_db)
        final_d = optimized_result.d_fields[-1]
        try:
            final_result = model.solve(final_d)
            compare_results(baseline_result, final_result, paper_improvement=0.287)
        except Exception as e:
            print(f"\nERROR: Failed to solve final optimized case: {e}")
            # Still compare using optimization Q values if available
            if hasattr(optimized_result, 'Q_values') and optimized_result.Q_values:
                print("\nUsing optimization Q values for comparison...")
                # Create a dummy result for comparison
                class DummyResult:
                    def __init__(self, Q):
                        self.Q = Q
                dummy_opt = DummyResult(optimized_result.Q_values[-1])
                compare_results(baseline_result, dummy_opt, paper_improvement=0.287)
            else:
                # Try comparing directly with OptimizationResult
                compare_results(baseline_result, optimized_result, paper_improvement=0.287)
    else:
        print("\n⚠️  Optimization did not complete successfully")
        print("   Cannot compare with paper results")
    
    print(f"\n{'='*60}")
    print("Validation complete")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
