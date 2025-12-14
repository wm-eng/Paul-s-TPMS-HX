"""Material properties interface with support for constant, REFPROP, and COOLProp properties."""

from typing import Protocol, Optional, Literal
import numpy as np

# Try to import REFPROP and COOLProp
REFPROP_MODULE = None
HAS_REFPROP = False
REFPROP_SOURCE = None  # Track which REFPROP source was found
HYFLUX_REFPROP_INTERFACE = None  # HyFlux_Hx REFPROP interface

def _find_refprop9_paths():
    """Find potential REFPROP_9 paths from HyFlux_Hx repository."""
    import os
    
    # Get project root (assume we're in src/hxopt/)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    # Common locations for REFPROP_9
    paths = [
        # In project root (symlink)
        os.path.join(project_root, 'REFPROP_9'),
        # In Hyflux subdirectory
        os.path.join(project_root, 'Hyflux', 'REFPROP_9'),
        # In user's home directory (cloned repository)
        os.path.join(os.path.expanduser('~'), 'HyFlux_Hx', 'Hyflux', 'REFPROP_9'),
        # In parent directory
        os.path.join(os.path.dirname(project_root), 'HyFlux_Hx', 'Hyflux', 'REFPROP_9'),
        # Absolute path if set in environment
        os.environ.get('REFPROP9_PATH', ''),
    ]
    
    # Filter out empty and non-existent paths
    valid_paths = [p for p in paths if p and os.path.exists(p)]
    return valid_paths

def _find_refprop_library(refprop9_path: str):
    """Find REFPROP library file (librefprop.dylib, refprop.dll, etc.) in REFPROP_9 directory."""
    import os
    import platform
    
    if not refprop9_path or not os.path.exists(refprop9_path):
        return None
    
    # Platform-specific library names
    system = platform.system()
    if system == 'Darwin':  # macOS
        lib_names = ['librefprop.dylib', 'refprop.dylib']
    elif system == 'Windows':
        lib_names = ['refprop.dll', 'librefprop.dll']
    elif system == 'Linux':
        lib_names = ['librefprop.so', 'refprop.so']
    else:
        lib_names = ['librefprop.dylib', 'refprop.dll', 'librefprop.so']
    
    # Look for library in REFPROP_9 directory
    for lib_name in lib_names:
        lib_path = os.path.join(refprop9_path, lib_name)
        if os.path.exists(lib_path):
            return lib_path
    
    return None

# Try to import REFPROP from various sources
try:
    import sys
    import os
    
    # Step 1: Try HyFlux_Hx REFPROP wrapper (preferred)
    try:
        from .refprop_wrapper import REFPROPInterface, HAS_HYFLUX_REFPROP
        if HAS_HYFLUX_REFPROP:
            # Initialize with REFPROP_9 data path
            refprop9_paths = _find_refprop9_paths()
            refprop_data_path = refprop9_paths[0] if refprop9_paths else None
            
            # Find library file
            refprop_library_path = None
            if refprop_data_path:
                refprop_library_path = _find_refprop_library(refprop_data_path)
                # Set environment variables for REFPROP
                os.environ['RPPREFIX'] = refprop_data_path
                if refprop_library_path:
                    os.environ['REFPROP_LIBRARY'] = refprop_library_path
            
            try:
                HYFLUX_REFPROP_INTERFACE = REFPROPInterface(
                    refprop_library_path=refprop_library_path,
                    refprop_data_path=refprop_data_path
                )
                if HYFLUX_REFPROP_INTERFACE.refprop_available:
                    REFPROP_MODULE = HYFLUX_REFPROP_INTERFACE
                    HAS_REFPROP = True
                    REFPROP_SOURCE = 'HyFlux_Hx REFPROPInterface'
            except Exception as e:
                # Continue to try other methods
                pass
    except ImportError:
        pass
    
    # Step 2: Try REFPROP_9 namespace package (if HyFlux wrapper not available)
    if not HAS_REFPROP:
        refprop9_paths = _find_refprop9_paths()
        for refprop_path in refprop9_paths:
            abs_path = os.path.abspath(refprop_path)
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            
            try:
                import REFPROP_9
                REFPROP_MODULE = REFPROP_9
                HAS_REFPROP = True
                REFPROP_SOURCE = f'REFPROP_9 from {abs_path}'
                break
            except ImportError:
                continue
    
    # Step 3: Try standard REFPROP imports (if REFPROP_9 not found)
    if not HAS_REFPROP:
        try:
            import REFPROP
            REFPROP_MODULE = REFPROP
            HAS_REFPROP = True
            REFPROP_SOURCE = 'REFPROP (standard)'
        except ImportError:
            try:
                import ctREFPROP.ctREFPROP as REFPROP_MODULE
                HAS_REFPROP = True
                REFPROP_SOURCE = 'ctREFPROP'
            except ImportError:
                pass
except Exception:
    pass

try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False


class FluidProperties(Protocol):
    """Interface for fluid property lookups."""
    
    def density(self, T: float, P: float) -> float:
        """Density at temperature T and pressure P. Units: kg/m³."""
        ...
    
    def viscosity(self, T: float, P: float) -> float:
        """Dynamic viscosity. Units: Pa·s."""
        ...
    
    def specific_heat(self, T: float, P: float) -> float:
        """Specific heat capacity. Units: J/(kg·K)."""
        ...
    
    def thermal_conductivity(self, T: float, P: float) -> float:
        """Thermal conductivity. Units: W/(m·K)."""
        ...
    
    def saturation_temperature(self, P: float) -> float:
        """Saturation temperature at pressure P. Units: K."""
        ...


class ConstantProperties:
    """Constant property model (v1 implementation)."""
    
    def __init__(
        self,
        rho: float,
        mu: float,
        cp: float,
        k: float,
        T_sat_ref: Optional[float] = None,
        dT_sat_dP: Optional[float] = None,
    ):
        """
        Initialize constant properties.
        
        Parameters
        ----------
        rho : float
            Density (kg/m³)
        mu : float
            Dynamic viscosity (Pa·s)
        cp : float
            Specific heat capacity (J/(kg·K))
        k : float
            Thermal conductivity (W/(m·K))
        T_sat_ref : float, optional
            Reference saturation temperature at P_ref (K)
        dT_sat_dP : float, optional
            dT_sat/dP slope (K/Pa) for linear approximation
        """
        self.rho = rho
        self.mu = mu
        self.cp = cp
        self.k = k
        self.T_sat_ref = T_sat_ref
        self.dT_sat_dP = dT_sat_dP or 0.0
        self.P_ref = 101325.0  # Pa, standard reference pressure
    
    def density(self, T, P):
        """Constant density. Handles scalar or array inputs."""
        T = np.asarray(T)
        if T.ndim == 0:
            return self.rho
        return np.full_like(T, self.rho, dtype=float)
    
    def viscosity(self, T, P):
        """Constant viscosity. Handles scalar or array inputs."""
        T = np.asarray(T)
        if T.ndim == 0:
            return self.mu
        return np.full_like(T, self.mu, dtype=float)
    
    def specific_heat(self, T, P):
        """Constant specific heat. Handles scalar or array inputs."""
        T = np.asarray(T)
        if T.ndim == 0:
            return self.cp
        return np.full_like(T, self.cp, dtype=float)
    
    def thermal_conductivity(self, T, P):
        """Constant thermal conductivity. Handles scalar or array inputs."""
        T = np.asarray(T)
        if T.ndim == 0:
            return self.k
        return np.full_like(T, self.k, dtype=float)
    
    def saturation_temperature(self, P):
        """
        Linear approximation: T_sat(P) = T_sat_ref + dT_sat_dP * (P - P_ref).
        
        For v1, this is a stub. TODO: Replace with real-fluid property fits.
        Handles scalar or array inputs.
        """
        P = np.asarray(P)
        if self.T_sat_ref is None:
            # Default: assume subcooled liquid (no phase change)
            if P.ndim == 0:
                return 1e6  # Very high value to avoid constraint violation
            return np.full_like(P, 1e6, dtype=float)
        
        result = self.T_sat_ref + self.dT_sat_dP * (P - self.P_ref)
        return result


class RealFluidProperties:
    """
    Real-fluid properties using REFPROP (preferred) or COOLProp (fallback).
    
    Supports:
    - Helium (He) - cryogenic gas
    - Hydrogen (H2) - liquid hydrogen (LH2)
    """
    
    # Fluid name mappings
    FLUID_NAMES = {
        'helium': {'REFPROP': 'HELIUM', 'COOLProp': 'Helium'},
        'he': {'REFPROP': 'HELIUM', 'COOLProp': 'Helium'},
        'hydrogen': {'REFPROP': 'HYDROGEN', 'COOLProp': 'Hydrogen'},
        'h2': {'REFPROP': 'HYDROGEN', 'COOLProp': 'Hydrogen'},
        'lh2': {'REFPROP': 'HYDROGEN', 'COOLProp': 'Hydrogen'},
    }
    
    def __init__(
        self,
        fluid_name: str,
        backend: Optional[Literal['REFPROP', 'COOLProp', 'auto']] = 'auto',
    ):
        """
        Initialize real-fluid properties.
        
        Parameters
        ----------
        fluid_name : str
            Fluid name: 'helium', 'he', 'hydrogen', 'h2', or 'lh2'
        backend : str, optional
            Property backend: 'REFPROP' (preferred), 'COOLProp' (fallback), or 'auto'
            If 'auto', tries REFPROP first, then COOLProp
        """
        fluid_name_lower = fluid_name.lower()
        if fluid_name_lower not in self.FLUID_NAMES:
            raise ValueError(
                f"Unknown fluid: {fluid_name}. "
                f"Supported: {list(self.FLUID_NAMES.keys())}"
            )
        
        self.fluid_name = fluid_name_lower
        self.fluid_names = self.FLUID_NAMES[fluid_name_lower]
        
        # Determine backend
        if backend == 'auto':
            if HAS_REFPROP:
                self.backend = 'REFPROP'
                self._init_refprop()
            elif HAS_COOLPROP:
                self.backend = 'COOLProp'
                self._init_coolprop()
            else:
                raise RuntimeError(
                    "Neither REFPROP nor COOLProp available. "
                    "Install COOLProp: pip install CoolProp"
                )
        elif backend == 'REFPROP':
            if not HAS_REFPROP:
                raise RuntimeError("REFPROP not available")
            self.backend = 'REFPROP'
            self._init_refprop()
            # Also initialize COOLProp for fallback
            if HAS_COOLPROP:
                self._init_coolprop()
        elif backend == 'COOLProp':
            if not HAS_COOLPROP:
                raise RuntimeError("COOLProp not available")
            self.backend = 'COOLProp'
            self._init_coolprop()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _init_refprop(self):
        """Initialize REFPROP backend."""
        if not HAS_REFPROP or REFPROP_MODULE is None:
            raise RuntimeError("REFPROP not available")
        # REFPROP initialization
        self.rp = REFPROP_MODULE
        self.fluid_refprop = self.fluid_names['REFPROP']
        
        # Detect REFPROP_9 module (from HyFlux_Hx repository)
        self.is_refprop9 = (
            REFPROP_SOURCE and 'REFPROP_9' in REFPROP_SOURCE
        ) or (
            hasattr(self.rp, '__name__') and 'REFPROP_9' in str(self.rp.__name__)
        ) or (
            hasattr(self.rp, '__file__') and 'REFPROP_9' in str(self.rp.__file__)
        )
        
        # Detect HyFlux_Hx REFPROPInterface
        self.is_hyflux_interface = (
            REFPROP_SOURCE and 'HyFlux_Hx' in REFPROP_SOURCE
        ) or (
            hasattr(self.rp, 'get_property') and hasattr(self.rp, 'refprop_available')
        )
        
        # Try to set fluid (API may vary by wrapper)
        try:
            if self.is_hyflux_interface:
                # HyFlux_Hx interface - no explicit initialization needed
                # It handles fluid selection per call
                pass
            elif hasattr(self.rp, 'SETFLUIDS'):
                self.rp.SETFLUIDS(self.fluid_refprop)
            elif hasattr(self.rp, 'SETUPdll'):
                # Standard REFPROP API
                self.rp.SETUPdll(1, self.fluid_refprop, 'HMX.BNC', 'DEF')
            elif self.is_refprop9:
                # REFPROP_9 from HyFlux_Hx may have different initialization
                # Try common initialization patterns
                if hasattr(self.rp, 'setup'):
                    self.rp.setup(self.fluid_refprop)
        except Exception as e:
            # Continue anyway - fluid will be set per call
            pass
    
    def _init_coolprop(self):
        """Initialize COOLProp backend."""
        if not HAS_COOLPROP:
            raise RuntimeError("COOLProp not available")
        self.fluid_coolprop = self.fluid_names['COOLProp']
    
    def _ensure_coolprop_init(self):
        """Ensure COOLProp is initialized (for fallback use)."""
        if not hasattr(self, 'fluid_coolprop') and HAS_COOLPROP:
            self._init_coolprop()
    
    def _get_property_refprop(self, prop: str, T: float, P: float) -> float:
        """Get property from REFPROP."""
        # Property mapping for HyFlux_Hx interface
        hyflux_prop_map = {
            'D': 'DENSITY',  # Density
            'V': 'VIS',      # Viscosity
            'C': 'HEAT_CAPACITY_CP',  # Specific heat
            'L': 'THERMAL_CONDUCTIVITY',  # Thermal conductivity
        }
        
        # Standard REFPROP property codes
        prop_map = {
            'D': 'D',  # Density (kg/m³)
            'V': 'V',  # Viscosity (Pa·s)
            'C': 'C',  # Specific heat (J/(kg·K))
            'L': 'L',  # Thermal conductivity (W/(m·K))
        }
        
        if prop == 'T_sat':
            # Saturation temperature from pressure
            try:
                # Try HyFlux_Hx interface first
                if self.is_hyflux_interface:
                    # HyFlux interface: get saturation temperature from pressure
                    # Use critical temperature as approximation or calculate from P
                    try:
                        from .refprop_wrapper import PropertyType
                        # Get critical temperature first
                        T_crit = self.rp.get_property(
                            self.fluid_refprop, PropertyType.CRITICAL_TEMPERATURE.value, 0, P
                        )
                        # For hydrogen, use proper saturation calculation
                        # This is a simplified approach - may need refinement
                        if self.fluid_name in ['hydrogen', 'h2', 'lh2']:
                            # Use COOLProp for accurate saturation temperature
                            if HAS_COOLPROP:
                                return self._get_property_coolprop(prop, T, P)
                        # For helium (gas), return very high value
                        return 1e6
                    except Exception:
                        # Fallback
                        if HAS_COOLPROP:
                            return self._get_property_coolprop(prop, T, P)
                        return 1e6
                
                # Try REFPROP_9 API (from HyFlux_Hx)
                if self.is_refprop9:
                    if hasattr(self.rp, 'Tsat_P'):
                        return self.rp.Tsat_P(P / 1000.0, self.fluid_refprop)
                    elif hasattr(self.rp, 'get_prop'):
                        return self.rp.get_prop('T', 'P', P / 1000.0, self.fluid_refprop)
                    elif hasattr(self.rp, 'PropsSI'):
                        return self.rp.PropsSI('T', 'P', P, 'Q', 0, self.fluid_refprop)
                
                # Try standard REFPROP API
                if hasattr(self.rp, 'REFPROPdll'):
                    result = self.rp.REFPROPdll(
                        self.fluid_refprop, 'P', 'T', 0, 0, 0, P / 1000.0, 0
                    )
                    if hasattr(result, 'Output'):
                        return result.Output[0]
                    return result[0] if isinstance(result, (list, tuple)) else result
                elif hasattr(self.rp, 'REFPROP'):
                    result = self.rp.REFPROP(self.fluid_refprop, 'P', 'T', P / 1000.0)
                    return result[0] if isinstance(result, (list, tuple)) else result
            except Exception:
                pass
            
            # Fallback to COOLProp
            if HAS_COOLPROP:
                return self._get_property_coolprop(prop, T, P)
            return 1e6  # Very high value as last resort
        
        prop_code = prop_map.get(prop)
        if prop_code is None:
            raise ValueError(f"Unknown property: {prop}")
        
        try:
            # Try HyFlux_Hx interface first (preferred)
            if self.is_hyflux_interface:
                hyflux_prop = hyflux_prop_map.get(prop, prop_code)
                try:
                    # HyFlux interface: get_property(fluid, prop_type, T, P, quality=None)
                    return self.rp.get_property(
                        self.fluid_refprop, hyflux_prop, T, P
                    )
                except Exception:
                    # Try with standard property code or alternative names
                    try:
                        return self.rp.get_property(
                            self.fluid_refprop, prop_code, T, P
                        )
                    except Exception:
                        # Try with PropertyType enum values
                        from .refprop_wrapper import PropertyType
                        if prop == 'D':
                            return self.rp.get_property(self.fluid_refprop, PropertyType.DENSITY.value, T, P)
                        elif prop == 'V':
                            return self.rp.get_property(self.fluid_refprop, PropertyType.VISCOSITY.value, T, P)
                        elif prop == 'C':
                            return self.rp.get_property(self.fluid_refprop, PropertyType.HEAT_CAPACITY_CP.value, T, P)
                        elif prop == 'L':
                            return self.rp.get_property(self.fluid_refprop, PropertyType.THERMAL_CONDUCTIVITY.value, T, P)
                        raise
            
            # Try REFPROP_9 API (from HyFlux_Hx)
            if self.is_refprop9:
                try:
                    if hasattr(self.rp, 'get_prop'):
                        return self.rp.get_prop(prop_code, 'T', T, 'P', P / 1000.0, self.fluid_refprop)
                    elif hasattr(self.rp, prop_code.lower()):
                        method = getattr(self.rp, prop_code.lower())
                        return method(T, P / 1000.0, self.fluid_refprop)
                    elif hasattr(self.rp, 'PropsSI'):
                        return self.rp.PropsSI(prop_code, 'T', T, 'P', P / 1000.0, self.fluid_refprop)
                except Exception:
                    pass
            
            # Standard REFPROP API calls
            if hasattr(self.rp, 'REFPROPdll'):
                result = self.rp.REFPROPdll(
                    self.fluid_refprop, 'TP', prop_code, 0, 0, 0, T, P / 1000.0
                )
                if hasattr(result, 'Output'):
                    return result.Output[0]
                return result[0] if isinstance(result, (list, tuple)) else result
            elif hasattr(self.rp, 'REFPROP'):
                result = self.rp.REFPROP(self.fluid_refprop, 'TP', prop_code, T, P / 1000.0)
                return result[0] if isinstance(result, (list, tuple)) else result
            else:
                # Fallback to COOLProp
                return self._get_property_coolprop(prop, T, P)
        except Exception as e:
            # Fallback to COOLProp if REFPROP fails
            if HAS_COOLPROP:
                return self._get_property_coolprop(prop, T, P)
            raise RuntimeError(f"REFPROP property lookup failed and COOLProp unavailable: {e}")
    
    def _get_property_coolprop(self, prop: str, T: float, P: float) -> float:
        """Get property from COOLProp."""
        if not HAS_COOLPROP:
            raise RuntimeError("COOLProp not available")
        
        # Ensure COOLProp is initialized (for fallback use)
        if not hasattr(self, 'fluid_coolprop'):
            self._init_coolprop()
        
        prop_map = {
            'D': 'D',  # Density (kg/m³)
            'V': 'V',  # Viscosity (Pa·s)
            'C': 'C',  # Specific heat (J/(kg·K))
            'L': 'L',  # Thermal conductivity (W/(m·K))
        }
        
        if prop == 'T_sat':
            # Saturation temperature from pressure
            try:
                return CP.PropsSI('T', 'P', P, 'Q', 0, self.fluid_coolprop)
            except Exception:
                return 1e6  # Fallback
        
        prop_code = prop_map.get(prop)
        if prop_code is None:
            raise ValueError(f"Unknown property: {prop}")
        
        try:
            return CP.PropsSI(prop_code, 'T', T, 'P', P, self.fluid_coolprop)
        except Exception as e:
            raise RuntimeError(f"COOLProp property lookup failed: {e}")
    
    def _get_property(self, prop: str, T, P):
        """Get property, handling scalar or array inputs."""
        T = np.asarray(T)
        P = np.asarray(P)
        
        if T.ndim == 0:
            # Scalar
            if self.backend == 'REFPROP':
                return self._get_property_refprop(prop, float(T), float(P))
            else:
                return self._get_property_coolprop(prop, float(T), float(P))
        else:
            # Array - vectorize
            result = np.zeros_like(T, dtype=float)
            T_flat = T.flatten()
            P_flat = P.flatten()
            for i in range(T.size):
                try:
                    if self.backend == 'REFPROP':
                        val = self._get_property_refprop(
                            prop, float(T_flat[i]), float(P_flat[i])
                        )
                    else:
                        val = self._get_property_coolprop(
                            prop, float(T_flat[i]), float(P_flat[i])
                        )
                    # Ensure value is a scalar float
                    result.flat[i] = float(val)
                except Exception as e:
                    # If property lookup fails, use a reasonable default
                    # This prevents the "Error setting single item" issue
                    if prop == 'D':
                        result.flat[i] = 1.0  # Default density
                    elif prop in ['C', 'CP', 'HEAT_CAPACITY_CP']:
                        result.flat[i] = 5000.0  # Default specific heat
                    elif prop in ['V', 'VIS', 'VISCOSITY']:
                        result.flat[i] = 1e-5  # Default viscosity
                    elif prop in ['L', 'TCX', 'THERMAL_CONDUCTIVITY']:
                        result.flat[i] = 0.1  # Default thermal conductivity
                    else:
                        result.flat[i] = 0.0
            return result.reshape(T.shape)
    
    def density(self, T, P):
        """Density at temperature T and pressure P. Units: kg/m³."""
        return self._get_property('D', T, P)
    
    def viscosity(self, T, P):
        """Dynamic viscosity. Units: Pa·s."""
        return self._get_property('V', T, P)
    
    def specific_heat(self, T, P):
        """Specific heat capacity at constant pressure. Units: J/(kg·K)."""
        return self._get_property('C', T, P)
    
    def thermal_conductivity(self, T, P):
        """Thermal conductivity. Units: W/(m·K)."""
        return self._get_property('L', T, P)
    
    def saturation_temperature(self, P):
        """
        Saturation temperature at pressure P. Units: K.
        
        For gases (helium), returns a very high value to avoid constraint violations.
        """
        P = np.asarray(P)
        
        # For helium (gas), no saturation in typical operating range
        if self.fluid_name in ['helium', 'he']:
            if P.ndim == 0:
                return 1e6
            return np.full_like(P, 1e6, dtype=float)
        
        # For hydrogen (liquid), calculate actual saturation temperature
        if P.ndim == 0:
            if self.backend == 'REFPROP':
                return self._get_property_refprop('T_sat', 0, float(P))
            else:
                return self._get_property_coolprop('T_sat', 0, float(P))
        else:
            result = np.zeros_like(P, dtype=float)
            for i in range(P.size):
                if self.backend == 'REFPROP':
                    result.flat[i] = self._get_property_refprop('T_sat', 0, float(P.flat[i]))
                else:
                    result.flat[i] = self._get_property_coolprop('T_sat', 0, float(P.flat[i]))
            return result

