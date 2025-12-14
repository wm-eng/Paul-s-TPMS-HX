# Iyer et al. (2022) Benchmark Reference

## Paper Summary

**Iyer et al. (2022)** - "Heat transfer and pressure drop characteristics of heat exchangers based on triply periodic minimal surfaces (TPMS) and periodic nodal surfaces (PNS)"

### Key Findings

1. **TPMS/PNS structures offer superior convective heat transfer** compared to traditional tubular heat exchangers
2. **Higher frictional pressure drops** are associated with TPMS structures
3. **Schwarz-D structure** demonstrated the best heat transfer performance
4. **Enhanced heat transfer efficiency** can lead to significant reductions in heat exchanger size
5. **Room-temperature benchmark conditions** provide baseline for lattice performance

### Test Conditions (Typical)

- **Temperature range**: 20-40°C (293-313 K) - ambient/room temperature
- **Pressure**: Atmospheric (~101.325 kPa)
- **Fluid**: Air (typical for room temperature studies)
- **Flow rates**: Varies by study, typically 0.001-0.01 kg/s for test sections
- **Geometry**: Test sections typically 5-20 cm in characteristic dimension

### Expected Performance Metrics

Based on Iyer et al. (2022) and similar TPMS studies:

1. **Nusselt Number (Nu)**: 10-50
   - Depends on structure type (gyroid, primitive, etc.)
   - Depends on Reynolds number
   - Higher for TPMS compared to smooth tubes

2. **Pressure Drop**: 100-1000 Pa
   - For typical test conditions (10 cm length, ~1 g/s flow)
   - Higher than smooth tubes due to complex geometry
   - Scales with velocity (Darcy-Forchheimer behavior)

3. **Heat Transfer Coefficient (h)**: 100-1000 W/(m²·K)
   - Depends on flow rate and structure
   - Higher surface area per unit volume increases h

4. **Heat Transfer Rate (Q)**: 10-1000 W
   - For typical test conditions
   - Depends on temperature difference and flow rate

### Structure Comparison

Iyer et al. studied multiple TPMS structures:
- **Gyroid**: Good balance of heat transfer and pressure drop
- **Primitive**: Similar to gyroid, slightly different characteristics
- **Schwarz-D**: Best heat transfer performance
- **Diamond**: Alternative structure
- **IWP**: Another TPMS variant

## Benchmark Test Cases

### Test Case 1: Ambient Air Conditions

**Conditions:**
- Hot fluid: Air at 320 K (47°C)
- Cold fluid: Air at 300 K (27°C)
- Pressure: 101.325 kPa (1 atm)
- Mass flow rate: 0.001 kg/s (1 g/s)
- Geometry: 10 cm × 10 cm × 10 cm
- Structure: Primitive or Gyroid

**Expected Results:**
- Heat transfer: 10-100 W
- Pressure drop: 100-1000 Pa
- Temperature change: 5-15 K (hot side), 5-15 K (cold side)

### Test Case 2: Higher Flow Rate

**Conditions:**
- Same as Test Case 1, but m_dot = 0.01 kg/s (10 g/s)

**Expected Results:**
- Heat transfer: 100-1000 W (scales with flow rate)
- Pressure drop: 1000-10000 Pa (scales with velocity²)

### Test Case 3: Different Temperature Difference

**Conditions:**
- Same as Test Case 1, but ΔT = 30 K (T_hot = 330 K, T_cold = 300 K)

**Expected Results:**
- Heat transfer: 15-150 W (scales with ΔT)

## Validation Checklist

When testing against Iyer et al. (2022) benchmark:

- [ ] Heat transfer is positive and physically reasonable
- [ ] Pressure drop increases with flow rate
- [ ] Heat transfer increases with temperature difference
- [ ] Energy balance is satisfied (Q_hot ≈ Q_cold)
- [ ] Temperature profiles are monotonic
- [ ] Pressure drop is within expected range (100-1000 Pa for typical conditions)
- [ ] Heat transfer coefficient is reasonable (100-1000 W/(m²·K))

## Implementation Notes

### Current Status

The code implementation follows the macroscopic modeling approach:
- ✅ Darcy-Forchheimer pressure drop model
- ✅ Volumetric heat transfer coefficient
- ✅ RVE property interpolation
- ⚠️ Numerical stability needs improvement (solver can produce unphysical results)

### Known Issues

1. **Numerical Instability**: The energy balance solver can produce negative temperatures or unrealistic values
   - **Cause**: Likely due to initial conditions or relaxation parameters
   - **Workaround**: Use smaller time steps, better initial guesses, or relaxation
   - **Status**: Under investigation

2. **RVE Property Calibration**: Default RVE tables are synthetic
   - **Solution**: Calibrate from experimental data (see `docs/CALIBRATION.md`)
   - **Iyer Data**: Can be used to calibrate RVE properties for room temperature

### Future Work

1. **Calibrate RVE properties** from Iyer et al. (2022) data if available
2. **Improve numerical stability** of energy balance solver
3. **Add structure-specific RVE tables** (gyroid, Schwarz-D, etc.)
4. **Compare results** with published Nusselt numbers and pressure drops

## References

- Iyer et al. (2022) - "Heat transfer and pressure drop characteristics of heat exchangers based on triply periodic minimal surfaces (TPMS) and periodic nodal surfaces (PNS)"
- Yanagihara et al. (2025) - "Flow-priority optimization of additively manufactured variable-TPMS lattice heat exchanger based on macroscopic analysis" [arXiv:2512.10207](https://arxiv.org/pdf/2512.10207)

## Test Suite

See `tests/test_iyer_2022_benchmark.py` for automated test cases.

