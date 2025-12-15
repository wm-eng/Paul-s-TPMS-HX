# TPMS Heat Exchanger Optimizer GUI Guide

## Overview

The GUI provides an interactive interface for:
- Selecting TPMS structure types
- Configuring geometry and flow parameters
- Visualizing 2D flow paths
- Running optimization
- Viewing results and plots

## Launching the GUI

**On macOS (recommended):**
```bash
# Option 1: Use the launcher script (opens in Terminal.app)
./scripts/launch_gui.sh

# Option 2: Run directly from Terminal.app (not Cursor's integrated terminal)
cd /path/to/TPMS-HX
source venv/bin/activate
python scripts/run_gui.py

# Option 3: Use pythonw (if available)
pythonw scripts/run_gui.py
```

**Note:** If running from Cursor's integrated terminal, the GUI may crash due to display access limitations. Use Terminal.app instead or the launcher script above.

Or from Python:

```python
from hxopt.gui import main
main()
```

## GUI Components

### Left Panel: Configuration

#### TPMS Structure Selection
- **TPMS Structure**: Dropdown to select structure type (Primitive, Gyroid, Diamond, IWP, Neovius)
- **RVE Table**: Path to RVE property table CSV file
  - Browse button to select file
  - Automatically updates when TPMS type changes
  - **Fallback behavior**: If TPMS-specific RVE table is missing, automatically falls back to `primitive_default.csv` with a notification

#### Geometry Parameters
- **Length (m)**: Flow direction length
- **Width (m)**: Cross-flow direction width
- **Height (m)**: Thickness direction height
- **Segments**: Number of discretization segments

#### Flow Path Configuration (2D)
- **Enable 2D Flow Path**: Checkbox to enable 2D planar geometry
- **Hot Path**: Flow path type for hot fluid (Straight, U-Shaped, L-Shaped)
- **Cold Path**: Flow path type for cold fluid (Straight, U-Shaped, L-Shaped)

#### Fluid Properties
- **T_hot_in (K)**: Hot fluid inlet temperature
- **T_cold_in (K)**: Cold fluid inlet temperature
- **P_hot_in (Pa)**: Hot fluid inlet pressure
- **P_cold_in (Pa)**: Cold fluid inlet pressure
- **m_dot_hot (kg/s)**: Hot fluid mass flow rate
- **m_dot_cold (kg/s)**: Cold fluid mass flow rate
- **Use Real-Fluid Properties**: Checkbox to enable REFPROP/COOLProp

#### Optimization Parameters
- **Max Iterations**: Maximum optimization iterations
- **d_min**: Minimum design variable value
- **d_max**: Maximum design variable value
- **d_init**: Initial design variable value
- **ΔP_max_hot (Pa)**: Maximum allowed hot side pressure drop
- **ΔP_max_cold (Pa)**: Maximum allowed cold side pressure drop

#### Action Buttons
- **Load Config**: Load configuration from file (JSON, not yet implemented)
- **Solve**: Solve the model with current configuration
- **Optimize**: Run optimization loop
- **Export Results**: Export results to CSV or VTK file

### Right Panel: Visualizations

#### Tab 1: 2D Model
- Visualizes flow paths for 2D geometry
- Shows hot and cold fluid paths
- For 1D models, shows flow direction

#### Tab 2: Temperature Profiles
- Hot fluid temperature vs. position
- Cold fluid temperature vs. position
- Solid temperature (if available)

#### Tab 3: Design Variable (d)
- Design variable d(x) vs. position
- Shows d_min and d_max bounds

#### Tab 4: Results
- Text summary of results:
  - Heat transfer rate (Q)
  - Pressure drops (ΔP_hot, ΔP_cold)
  - Inlet/outlet temperatures
  - Inlet/outlet pressures
  - Optimization statistics (if optimized)

## Usage Workflow

### Basic Solve

1. Select TPMS structure type
2. Configure geometry parameters
3. Set fluid properties
4. Click **Solve**
5. View results in visualization tabs

### Optimization

1. Configure all parameters as above
2. Set optimization parameters (max iterations, bounds, constraints)
3. Click **Optimize**
4. Wait for optimization to complete (runs in background thread)
5. View optimized results and improvement percentage

### 2D Flow Path Visualization

1. Check **Enable 2D Flow Path**
2. Select flow path types for hot and cold fluids
3. Click **Solve** or **Optimize**
4. View 2D flow paths in the **2D Model** tab

## Features

### Real-Time Updates
- All visualizations update automatically after solve/optimize
- Status bar shows current operation status

### Export Capabilities
- Export results to CSV (field data)
- Export results to VTK (for ParaView visualization)
- **Export STL**: Generate 3D TPMS mesh from optimized d(x) field
  - Runs in background thread to prevent UI blocking
  - Automatic memory management and resolution scaling
  - Prevents memory exhaustion for large geometries

### Threading
- Optimization runs in background thread
- GUI remains responsive during optimization
- STL export runs in background thread with memory limits

## Tips

1. **Start Simple**: Begin with 1D straight flow paths
2. **Check RVE Table**: Ensure RVE table path is correct for selected TPMS type
3. **Monitor Status**: Watch status bar for operation progress
4. **Use Real-Fluid Properties**: Enable for accurate cryogenic calculations
5. **Adjust Segments**: More segments = higher accuracy but slower computation

## Troubleshooting

### GUI Won't Launch
- Ensure matplotlib is installed: `pip install matplotlib`
- Check Python version (requires 3.9+)
- **macOS**: If running from Cursor's integrated terminal, use `./scripts/launch_gui.sh` or run from Terminal.app
- **Display errors**: GUI requires proper display access; use Terminal.app on macOS

### Solve Fails
- Check RVE table path is valid
- Verify all parameters are within valid ranges
- Check console for error messages
- **Missing RVE tables**: GUI automatically falls back to Primitive properties if TPMS-specific table is missing

### Optimization Takes Too Long
- Reduce max iterations
- Reduce number of segments
- Check constraints are reasonable

### No Visualization
- Ensure solve/optimize completed successfully
- Check that results are available
- Try clicking solve again

## Future Enhancements

- Config file save/load (JSON)
- Real-time parameter validation
- 3D visualization
- TPMS structure preview
- Comparison mode (multiple configurations)
- Export optimization history

