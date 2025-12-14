# TPMS Heat Exchanger Optimizer

Optimization loop for a He/LH₂ TPMS (Triply Periodic Minimal Surface) heat exchanger using:
# - RVE property lookup (κ, β, ε, λ, h)
# - Macromodel solver (porous-flow + 3-temperature energy)
# - MMA optimization over d(x) (channel-bias field)

## Installation

Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m pip install -e .
```

Or install directly (may require `--break-system-packages` on some systems):
```bash
python3 -m pip install -e .
```

## Quick Start

Run the v1 optimization example (1D counterflow):

```bash
python examples/run_v1.py
```

Run the v1.1 optimization example (2D U-shaped flow paths):

```bash
python examples/run_v1_2d.py
```

Run the v1.2 optimization example (real-fluid properties):

```bash
python examples/run_v1_real_properties.py
```

**Note**: Real-fluid properties require COOLProp (installed by default) or REFPROP (optional, higher accuracy).

### GUI Application

Launch the graphical user interface:

```bash
python scripts/run_gui.py
```

The GUI provides:
- TPMS structure selection (Primitive, Gyroid, Diamond, etc.)
- 2D flow path visualization
- Interactive parameter configuration
- Real-time optimization results
- Temperature and design variable plots

This will:
1. Load RVE property tables from `data/rve_tables/`
2. Solve the 1D counterflow macromodel
3. Optimize the channel-bias field d(x) to maximize heat transfer
4. Export results to `runs/`

## Project Structure

```
hx-optimizer/
├── src/hxopt/          # Main package
│   ├── config.py       # Configuration dataclasses
│   ├── materials.py    # Material properties
│   ├── rve_db.py       # RVE property interpolation
│   ├── macro_model.py  # 1D/2D/3D solver
│   ├── objective.py    # Optimization objective
│   ├── constraints.py  # Constraints
│   ├── optimize_mma.py # MMA optimizer
│   └── ...
├── data/               # RVE property tables
├── examples/           # Example scripts
└── tests/              # Unit tests
```

## Architecture

- **v1.0**: 1D counterflow model (fast, debuggable)
- **v1.1**: 2D planar geometry with U-shaped/L-shaped flow paths ✅
- **v1.2+**: Extend to 3D FEM (FEniCSx)

## Development

Install in development mode:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest tests/
```

## Architecture Notes

### v1.0 Features
- 1D counterflow porous-equivalent model
- Constant fluid properties
- RVE property interpolation from CSV tables
- Projected gradient optimization with line search
- Energy balance validation
- Constraint checking (pressure drop, subcooling margin)

### Future Enhancements (v1.2+)
- 2D/3D FEM solver (FEniCSx)
- True MMA optimizer
- TPMS mesh generation from d(x) field
- Parallel gradient computation

### v1.2 Features ✅
- Real-fluid properties via REFPROP/COOLProp
- Cryogenic helium gas and liquid hydrogen support
- Automatic fallback: REFPROP → COOLProp → constant properties

## Calibration

RVE properties can be calibrated from experimental data. See `docs/CALIBRATION.md` for details.

**Quick start:**
```bash
# Calibrate from experimental data
python scripts/calibrate_from_experiments.py \
    data/experimental/your_data.csv \
    data/rve_tables/calibrated.csv
```

Synthetic test data is available at `data/experimental/synthetic_tpms_primitive_data.csv` for testing.

## Methodology

This implementation follows the macroscopic modeling approach described in:

**Yanagihara et al. (2025)** - "Flow-priority optimization of additively manufactured variable-TPMS lattice heat exchanger based on macroscopic analysis"  
[arXiv:2512.10207](https://arxiv.org/pdf/2512.10207)

Key features aligned with the paper:
- ✅ Darcy-Forchheimer theory for macroscopic flow analysis
- ✅ Volumetric heat-transfer coefficient model
- ✅ RVE property lookup (κ, β, ε, λ) with interpolation
- ✅ Isosurface threshold d(x) as design variable
- ✅ Gradient-based optimization framework

See `PAPER_VERIFICATION.md` for detailed verification against the paper's methodology.

Experimental Data for RVE Calibration at Cryogenic Conditions

Calibrating TPMS lattice RVE properties for cryogenic fluids (helium gas at ~40 K and liquid hydrogen at ~21 K) requires carefully obtained experimental (or high-fidelity simulation) data. These extreme conditions significantly affect fluid properties, so using relevant data ensures the Darcy–Forchheimer and heat transfer correlations are accurate ￼. Below we outline what data is needed and where it might be sourced for each fluid, as well as available surrogate datasets.

Helium Gas at ~40 K

Fluid Properties: Helium’s density increases dramatically at cryogenic temperatures. At 1 atm and 40 K, helium’s density is about 1.2 kg/m³ (versus 0.18 kg/m³ at 300 K) ￼. Its dynamic viscosity drops slightly with temperature (on the order of 10^–5 Pa·s, similar to or a bit lower than at 300 K). The much higher density (and resulting lower kinematic viscosity) means flow Reynolds numbers will be much higher for a given velocity, often pushing the regime toward turbulence and significant inertial losses. This must be accounted for in the Darcy–Forchheimer fit.

Required Data: To calibrate permeability (κ) and Forchheimer coefficient (β) for the TPMS lattice at 40 K, measure pressure drop vs. flow rate through a sample lattice over a range of flow velocities ￼. Ideally, obtain at least 2–3 flow points for each lattice isosurface parameter d (as the guide suggests) so that a linear regression can fit the Darcy–Forchheimer equation ΔP/L = (μ/κ)·u + β·ρ·u². In practice, this means conducting cryogenic flow experiments where helium gas at ~40 K is passed through a representative lattice section and the pressure drop per unit length is recorded at different flow speeds. Published studies on helium flow through porous media (e.g. fusion reactor pebble beds) confirm that pressure drop increases with velocity and follows a combined linear & quadratic trend, as expected ￼ ￼. These trends need to be confirmed at 40 K. If direct experiments at 40 K are challenging, one can use CFD simulations at these conditions to generate surrogate data, but experimental validation is preferable.

Heat Transfer Data: If the lattice will handle heat exchange, measure the convective heat transfer coefficient vs. flow velocity with helium at 40 K. For example, one could send cold helium through a heated lattice and measure the heat transfer rate to determine h. Fitting a power-law correlation h = a·u^b requires multiple data points spanning the velocity range ￼. Note that helium at 40 K has a very low Prandtl number and high thermal diffusivity, so the exponent b may differ from room-temperature correlations. In absence of dedicated cryogenic data, one might rely on dimensionless correlations from literature (e.g. Nusselt–Reynolds relations) and adjust for helium’s properties.

Sources & References: There is limited publicly available experimental data specifically for TPMS lattices at 40 K, due to the difficulty of cryogenic tests. However, analogous research provides guidance. For instance, Yanagihara et al. (2025) calibrated a TPMS Primitive lattice using helium (though at ambient conditions) and validated the Darcy–Forchheimer approach with experiments ￼. Their methodology can be extended to cryogenic helium by using helium’s 40 K viscosity and density in the calibration script (--rho and --mu flags). Additionally, studies of cryogenic helium loops (e.g. at the European Spallation Source) underscore the need for such data – in the ESS hydrogen moderator, heat is removed via a helium circuit operating at 15–20 K ￼. This indicates helium-based cryogenic heat exchangers are an active area, and any experimental data from those projects (even if using different structures) could serve as a starting point. In summary, for helium at 40 K one should gather pressure-drop and heat-transfer measurements on a TPMS sample if possible; otherwise, use high-fidelity CFD or extrapolate from higher-temperature data with proper fluid property adjustments.

Liquid Hydrogen (LH₂) at ~21 K

Fluid Properties: Liquid hydrogen near 21 K (its normal boiling point at 1 atm) has very different characteristics from helium. It is much denser (about 70–71 kg/m³ at 1 atm ￼) and yet has an extremely low viscosity (0.009 cP, i.e. 9×10^–6 Pa·s at 20 K ￼). This combination of high density and ultra-low viscosity means LH₂ flows can achieve enormous Reynolds numbers even at modest velocities. In other words, flow through the lattice will almost certainly be turbulent or in the inertial regime for practically any useful flow rate. The Darcy term may become less significant compared to the Forchheimer (inertial) term under these conditions. Nevertheless, the calibration should capture both, since at lower speeds or higher d (more open lattice), the linear viscous component can still matter.

Required Data: To calibrate the lattice’s permeability and β for liquid hydrogen, one needs pressure drop measurements in single-phase liquid H₂ across the lattice. This is very challenging experimentally (due to hydrogen’s flammability and cryogenic temperature), but it’s crucial. Multiple flow rates (at least 2–3 per geometry setting d) should be tested. Expect that ΔP vs u will show a strong quadratic behavior (ρ·u² term) because of the high Re. Ensure the data covers the expected operating range of velocities in the actual heat exchanger. If direct LH₂ testing is infeasible, proxy experiments using a similar cryogenic fluid can be considered – e.g. liquid nitrogen (77 K) or liquid argon through the same lattice, coupled with appropriate scaling. Those fluids have higher viscosity than LH₂, but data from them could be extrapolated. Another option is to use a pressurized cold helium gas or neon as a surrogate to mimic some properties (though not density) in a safer way.

For heat transfer correlation, measuring h in liquid hydrogen is also complex. Ideally, one would flow LH₂ through a heated lattice and measure heat input vs temperature rise to deduce convective coefficients. Given the difficulties, it may be acceptable to rely on theoretical correlations (e.g. Dittus–Boelter for turbulent flow) to estimate the exponents. The calibration script’s log-log fitting (for h = a·u^b) can be applied if at least a few data points of heat transfer are obtainable. Keep in mind LH₂ has a very low Prandtl number (~Pr ≈ 0.7 at 20 K), so heat transfer behavior might resemble that of other low-Pr fluids.

Sources & References: There is virtually no published public dataset of TPMS lattice testing in liquid hydrogen (owing to safety and rarity of such tests). However, cryogenic heat exchanger literature can provide bounds. For example, studies on printed-circuit heat exchangers (PCHEs) for LH₂ vaporizers show the scale of pressure drops and heat transfer coefficients to expect. One such experiment noted that freezing of residual moisture was a concern in LH₂ PCHE testing ￼, highlighting the need for pure hydrogen in experiments. Additionally, hydrogen storage reactors with TPMS structures (for metal hydride systems) have been numerically studied ￼ – those aren’t measuring flow in liquid hydrogen, but indicate interest in using TPMS for hydrogen systems.

In practice, to get experimental data one might collaborate with a cryogenic laboratory or a facility like NASA or CERN that has hydrogen cryogenic loops. For example, the Spallation Source moderators use liquid H₂ circulators and report a ~30 kW heat exchanger with helium cooling ￼ – while not giving explicit numbers, this confirms that moderate pressure drops (on the order of tens of kPa) are acceptable in such systems. Any available data from such systems (even if not TPMS lattices) can serve as a validation check: e.g. ensure the calibrated permeability yields pressure drops in the same order of magnitude for given flow rates and hydraulic diameters.

Important: When handling LH₂ data, ensure it’s single-phase. If the hydrogen is near saturation, any boiling will complicate pressure drop greatly. For RVE calibration, assume liquid only. Also, validate that the calibrated porosity and surface area per volume align with expectations – e.g. porosity should remain between 0 and 1 and generally increase with d, and the fitted κ should increase with d while β decreases, even at these cryogenic conditions (physical trends must hold) ￼.

Using Synthetic and Literature Data

In absence of immediate experimental data, the guide provides a synthetic dataset (synthetic_tpms_primitive_data.csv) based on validated TPMS Primitive properties from literature. This synthetic data spans multiple d values and flow rates, reflecting typical behavior of the lattice ￼. It can be used to test the calibration procedure and ensure the script works correctly. For example, using the synthetic file with the calibration script should produce a calibrated RVE table that smoothly varies properties from d = 0.1 to 0.9. This is a good initial sanity check before incorporating real cryogenic data.

When real data is obtained, one should compare the calibrated results with known references. For instance, Iyer et al. (2022) measured the pressure drop and heat transfer performance of TPMS-based heat exchangers ￼. Their results (albeit at room temperature with air/water) can serve as a baseline – TPMS lattices showed significantly enhanced heat transfer and reasonable pressure losses in those studies. Our calibrated cryogenic properties should be in the same ballpark when non-dimensionalized. Likewise, Yan et al. (2023) experimentally tested TPMS and hybrid structures and could be consulted for trends ￼. While these references don’t cover 40 K helium or 21 K hydrogen specifically, they add confidence that the RVE calibration method yields realistic outcomes.

Finally, remember to update the fluid properties in the calibration script for cryogenic conditions. Use --rho and --mu corresponding to helium at 40 K and LH₂ at 21 K. For example, helium’s density around 40 K (1 atm) is ~1.2 kg/m³ ￼, and LH₂’s density is ~71 kg/m³ ￼. Viscosities should be set to ~1×10^–5 Pa·s (helium) and ~9×10^–6 Pa·s (hydrogen) accordingly. By inputting these, the calibration fitting will properly account for the fluid differences. After running the calibration, validate the outputs: check that porosity, κ, β vs d are monotonic and physically reasonable, and if possible, run a small CFD simulation or hand-calculation to verify that the calibrated model predicts pressure drops and heat transfer in line with any known cryogenic exchanger performance.

In summary, obtaining experimental data for helium at 40 K and liquid hydrogen at 21 K is non-trivial but essential. Leverage any available cryogenic flow test data (from literature or in-house experiments) for pressure drop and convective heat transfer. Use the synthetic dataset as a placeholder to fine-tune the calibration process. Once real data are collected and fed into the calibration script, the resulting RVE properties will enable accurate optimization and design of the TPMS lattice heat exchanger under these extreme conditions ￼ ￼. Always cross-check the calibrated values against fundamental expectations and published results to ensure the calibration is trustworthy before relying on it in design optimization.

References
	•	Yanagihara et al., 2025 – “Flow-priority optimization of additively manufactured variable-TPMS lattice heat exchanger based on macroscopic analysis.” Demonstrated Darcy–Forchheimer modeling of TPMS lattices and provided baseline data (using helium as working fluid) ￼.
	•	Iyer et al., 2022 – Studied heat transfer and pressure drop in TPMS heat exchangers (gyroid, etc.) under ambient conditions ￼. Their findings serve as a room-temperature benchmark for lattice performance.
	•	Engineering Toolbox: Helium gas density at cryogenic temperatures – at 40 K and 1 bar, ≈1.20 kg/m³ ￼. Liquid hydrogen at 1 atm: density ≈71 kg/m³, viscosity ~0.009 cP ￼.
	•	ESS (European Spallation Source) Cryogenics – Hydrogen moderators cooled by 15–20 K helium loops ￼, illustrating a practical helium–hydrogen heat exchanger scenario (though not providing detailed data, it underscores the operating conditions of interest).
	•	Fusion Reactor Blanket studies – Helium flow through pebble-bed porous media shows linear vs quadratic pressure drop behavior and the effect of temperature on pressure loss ￼ ￼, supporting our approach to use Darcy–Forchheimer at 40 K.