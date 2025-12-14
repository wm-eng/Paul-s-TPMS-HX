# Optimization Algorithm Improvements

## Issues Identified and Fixed

### 1. **Wrong Search Direction** ✅ FIXED
**Problem**: The optimization was stepping in the direction of the gradient of the objective (-Q), but to maximize Q, we need to step in the opposite direction.

**Fix**: Changed from `d_new = d_field + step_size * grad` to `d_new = d_field + step_size * (-grad)`, where `search_dir = -grad` is the gradient of Q (not -Q).

### 2. **Incorrect Armijo Condition** ✅ FIXED
**Problem**: The line search was checking `improvement <= expected_improvement`, which is backwards for maximization. This would reject good steps and accept bad ones.

**Fix**: Changed to accept steps where `improvement >= expected_improvement` OR `improvement > 0` (any improvement is acceptable).

### 3. **Convergence Check Too Strict** ✅ FIXED
**Problem**: The convergence check used absolute tolerance of 1000 W (1e3), which is too large for small Q values. This caused premature convergence.

**Fix**: Changed to use relative tolerance (0.1% change) OR absolute tolerance of 100 W, whichever is more appropriate.

### 4. **Variable Scope Bug** ✅ FIXED
**Problem**: `ls_it` was used outside the loop scope, causing potential errors.

**Fix**: Introduced `ls_success` flag to track line search success.

### 5. **Step Size Too Small** ✅ FIXED
**Problem**: Default step size of 0.01 was too conservative, leading to slow convergence.

**Fix**: Increased default step size to 0.1 in validation script, and improved fallback step size when line search fails.

## Expected Improvements

With these fixes, the optimization should:
- Actually improve Q over iterations (not converge immediately)
- Show measurable improvement percentage
- Converge more reliably to better solutions
- Better match the paper's 28.7% improvement claim

## Testing

Run the validation script to see improvements:
```bash
source venv/bin/activate
python scripts/validate_paper_results.py
```

The optimization should now show:
- Multiple iterations before convergence
- Increasing Q values over iterations
- Measurable improvement over baseline

## Additional Recommendations

1. **Gradient Scaling**: Consider normalizing the gradient to prevent very large steps
2. **Adaptive Step Size**: Implement adaptive step size based on gradient magnitude
3. **Constraint Handling**: Consider penalty methods or barrier methods for better constraint satisfaction
4. **Gradient Accuracy**: The finite difference gradient computation is expensive (N+1 solves per iteration). Consider:
   - Adjoint method for faster gradients
   - Parallel gradient computation
   - Reduced-order models for gradient estimation

## Additional Improvements Made

### 6. **Adaptive Gradient Computation** ✅ IMPLEMENTED
**Problem**: Fixed perturbation size (1e-6) was too small relative to d values (0.1-0.9), causing gradients to be essentially zero or inaccurate.

**Fix**: 
- Implemented adaptive perturbation size: uses 1% of d range (minimum 1e-4)
- Uses central differences where possible for better accuracy
- Falls back to forward/backward differences at boundaries
- Handles clipping correctly

### 7. **Gradient Normalization** ✅ IMPLEMENTED
**Problem**: Unnormalized gradients could lead to very large or very small steps.

**Fix**: 
- Normalize search direction to unit vector
- Scale by d range to get reasonable step sizes
- Prevents numerical issues with very small or very large gradients

### 8. **Zero Gradient Detection** ✅ IMPLEMENTED
**Problem**: If gradient is zero, optimization would stall without warning.

**Fix**: 
- Detect zero gradients and warn user
- Try small random perturbation on first iteration to escape flat regions
- Exit gracefully if gradient remains zero

## References

- Yanagihara et al. (2025) - Paper reports 28.7% improvement
- **Cheung et al. (2025)** - "Triply periodic minimal surfaces for thermo-mechanical protection"
  - Scientific Reports 15, 1688 (2025)
  - [https://www.nature.com/articles/s41598-025-85935-x](https://www.nature.com/articles/s41598-025-85935-x)
  - DOI: [https://doi.org/10.1038/s41598-025-85935-x](https://doi.org/10.1038/s41598-025-85935-x)
  - Local copy: `docs/s41598-025-85935-x.pdf`
  - Experimental validation data for TPMS Primitive lattices including pressure drop and thermal conductivity measurements
- **Energies 18(1), 134 (2025)** - TPMS optimization and property characterization methods
  - [https://www.mdpi.com/1996-1073/18/1/134](https://www.mdpi.com/1996-1073/18/1/134)
  - Local copy: `docs/energies-18-00134.pdf`
- Standard optimization texts on gradient-based methods and line search
