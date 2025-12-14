#!/usr/bin/env python3
"""
REFPROP Integration Module v1.1.0
Enhanced fluid properties using NIST REFPROP with fallback to CoolProp
Provides high-accuracy thermophysical properties for cryogenic and industrial fluids

Based on NIST REFPROP-cmake: https://github.com/usnistgov/REFPROP-cmake
"""

import numpy as np
import warnings
import os
import sys
from typing import Dict, Optional, Union, Tuple, List
from dataclasses import dataclass

# Try to import ctREFPROP (the correct Python wrapper for REFPROP)
try:
    from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
    REFPROP_AVAILABLE = True
    print("✅ ctREFPROP integration loaded successfully")
except ImportError:
    REFPROP_AVAILABLE = False
    print("⚠️  ctREFPROP not available - will use CoolProp fallback")

# Import our compatibility layer for better status reporting
try:
    from refprop_compatibility import get_refprop_status, is_refprop_available
    REFPROP_STATUS = get_refprop_status()
    REFPROP_FULLY_AVAILABLE = is_refprop_available()
    # Use the compatibility layer status instead of the basic message
    if not REFPROP_AVAILABLE:
        print(REFPROP_STATUS)
except ImportError:
    REFPROP_STATUS = "⚠️  ctREFPROP not available - will use CoolProp fallback"
    REFPROP_FULLY_AVAILABLE = False

# Try to import CoolProp as fallback
try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_AVAILABLE = True
    print("✅ CoolProp fallback available")
except ImportError:
    COOLPROP_AVAILABLE = False
    print("⚠️  CoolProp not available - will use built-in correlations")

# Constants
MINIMUM_DENOMINATOR = 1e-12
MINIMUM_HELIUM_TEMPERATURE = 4.0  # K
MINIMUM_HYDROGEN_TEMPERATURE = 13.8  # K
MINIMUM_NITROGEN_TEMPERATURE = 63.0  # K

@dataclass
class FluidSpec:
    """Fluid specification with REFPROP/CoolProp backend selection"""
    name: str
    backend: str = 'REFPROP'  # 'REFPROP', 'CoolProp', or 'BuiltIn'
    P_in: float = 101325.0  # Pa
    roughness: float = 1e-6  # m
    composition: Optional[Dict[str, float]] = None  # For mixtures
    
    def __post_init__(self):
        if self.backend == 'REFPROP' and not REFPROP_AVAILABLE:
            print(f"⚠️  REFPROP not available for {self.name}, falling back to CoolProp")
            self.backend = 'CoolProp'
        if self.backend == 'CoolProp' and not COOLPROP_AVAILABLE:
            print(f"⚠️  CoolProp not available for {self.name}, using built-in correlations")
            self.backend = 'BuiltIn'

class REFPROPInterface:
    """Interface to NIST REFPROP for high-accuracy fluid properties using ctREFPROP"""
    
    def __init__(self, refprop_library_path: Optional[str] = None):
        """
        Initialize REFPROP interface using ctREFPROP
        
        Args:
            refprop_library_path: Path to REFPROP shared library (e.g., librefprop.dylib)
        """
        self.refprop_library_path = refprop_library_path or self._find_refprop_library()
        self.rp = None
        self._initialize_refprop()
    
    def _find_refprop_library(self) -> Optional[str]:
        """Find REFPROP shared library path with smart auto-discovery"""
        # Check environment variables first (highest priority)
        env_vars = ["REFPROP_LIBRARY", "REFPROP_PATH"]
        for env_var in env_vars:
            if env_var in os.environ:
                path = os.environ[env_var]
                if os.path.exists(path):
                    return path
        
        # Common REFPROP library paths
        common_paths = [
            "/usr/local/lib/librefprop.dylib",  # macOS Homebrew
            "/Applications/REFPROP/librefprop.dylib",  # macOS official
            "/usr/local/lib/librefprop.so",  # Linux
            "C:/Program Files/REFPROP/REFPROP.dll",  # Windows
            "C:/Program Files (x86)/REFPROP/REFPROP.dll"  # Windows 32-bit
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Smart auto-discovery for macOS build folders
        if sys.platform == "darwin":  # macOS
            # Look for common build locations
            build_paths = [
                os.path.expanduser("~/REFPROP-cmake/build/librefprop.dylib"),
                os.path.expanduser("~/GitHub/REFPROP-cmake/build/librefprop.dylib"),
                "/Volumes/Lexar4TB/GitHub/Hyflux/REFPROP-cmake/build/librefprop.dylib",  # Your current setup
            ]
            
            for path in build_paths:
                if os.path.exists(path):
                    print(f"🔍 Auto-discovered REFPROP library: {path}")
                    return path
        
        return None
    
    def _initialize_refprop(self):
        """Initialize REFPROP using ctREFPROP"""
        global REFPROP_AVAILABLE
        if not REFPROP_AVAILABLE:
            return
        
        try:
            if self.refprop_library_path:
                # Initialize ctREFPROP with the library path
                self.rp = REFPROPFunctionLibrary(self.refprop_library_path)
                print(f"✅ REFPROP library loaded: {self.refprop_library_path}")
                
                # Check if REFPROP data is available
                self._check_refprop_data()
                
                # Set up REFPROP path and fluids
                self._setup_refprop()
                
                # Test REFPROP functionality
                self._test_refprop()
            else:
                print("⚠️  REFPROP library not found - check REFPROP_LIBRARY environment variable")
                REFPROP_AVAILABLE = False
                
        except Exception as e:
            print(f"⚠️  REFPROP initialization failed: {e}")
            REFPROP_AVAILABLE = False
    
    def _check_refprop_data(self):
        """Check if REFPROP data files are available"""
        # Check environment variable for data path
        rpprefix = os.environ.get('RPPREFIX', '')
        
        if rpprefix:
            fluids_path = os.path.join(rpprefix, 'FLUIDS')
            mixtures_path = os.path.join(rpprefix, 'MIXTURES')
            
            if os.path.exists(fluids_path) and os.path.exists(mixtures_path):
                print(f"✅ REFPROP data found: {rpprefix}")
            else:
                print(f"⚠️  REFPROP data not found at {rpprefix}")
                print("   Note: REFPROP will work with limited fluids until data is provided")
        else:
            print("⚠️  RPPREFIX not set - REFPROP data location unknown")
            print("   Set RPPREFIX to the folder containing FLUIDS/ and MIXTURES/")
    
    def _setup_refprop(self):
        """Set up REFPROP path and initialize fluids"""
        try:
            rpprefix = os.environ.get('RPPREFIX', '')
            if not rpprefix:
                print("⚠️  RPPREFIX not set - skipping REFPROP setup")
                return
            
            # Set the REFPROP path using the library's method
            print(f"🔧 Setting REFPROP path to: {rpprefix}")
            self.rp.SETPATHdll(rpprefix)
            
            # Try to set units to mass-based SI using SETUNITS method
            print("🔧 Setting REFPROP units to mass-based SI...")
            try:
                self.rp.SETUNITS(1)  # 0: molar SI, 1: mass SI
                print("✅ Units set to mass-based SI using SETUNITS")
            except AttributeError:
                print("⚠️  SETUNITS not available, will use MKS constant per call")
                self.use_mks_constant = True
            except Exception as e:
                print(f"⚠️  SETUNITS failed: {e}, will use MKS constant per call")
                self.use_mks_constant = True
            
            # Try to set up fluids (this loads the fluid database)
            print("🔧 Loading REFPROP fluid database...")
            # Note: We'll test this in the _test_refprop method
            
        except Exception as e:
            print(f"⚠️  REFPROP setup failed: {e}")
            print("   Continuing with fallback methods...")
    
    def _test_refprop(self):
        """Test REFPROP functionality with water properties"""
        global REFPROP_AVAILABLE
        try:
            # Test basic water properties at standard conditions
            # Using the correct ctREFPROP API
            z = [1.0] + [0.0]*19  # pure water composition
            T = 300.0             # K
            P = 101.325           # kPa
            
            # Test: Get water density at 300K, 101.325 kPa
            # Note: MASS_BASE_SI expects Pa input, MKS expects kPa input
            if hasattr(self, 'use_mks_constant') and self.use_mks_constant:
                # Use MKS constant if SETUNITS failed (expects kPa)
                result = self.rp.REFPROPdll("WATER", "TP", "D", self.rp.MKS, 0, 0, T, P, z)
            else:
                # Use mass-based SI units (expects Pa)
                P_pa = P * 1000.0  # Convert kPa to Pa for MASS_BASE_SI
                result = self.rp.REFPROPdll("WATER", "TP", "D", self.rp.MASS_BASE_SI, 0, 0, T, P_pa, z)
            
            # Check if we got a valid result (not error code -9999990.0)
            if hasattr(result, 'Output') and len(result.Output) > 0:
                density = result.Output[0]
                if density > 0 and density != -9999990.0:
                    print("✅ REFPROP functionality verified")
                    print(f"   Version: {self.rp.RPVersion()}")
                    print(f"   Water density at 300K, 101.325 kPa: {density:.3f} kg/m³")
                    print(f"   Expected range: 996-997 kg/m³")
                else:
                    raise RuntimeError(f"REFPROP returned error value: {density}")
            else:
                raise RuntimeError("REFPROP returned invalid result structure")
                
        except Exception as e:
            print(f"⚠️  REFPROP test failed: {e}")
            REFPROP_AVAILABLE = False
    
    def get_fluid_properties(self, fluid_name: str, T: Union[float, np.ndarray], 
                           P: Union[float, np.ndarray], 
                           properties: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Get fluid properties using REFPROP via ctREFPROP
        
        Args:
            fluid_name: Name of fluid (REFPROP format)
            T: Temperature(s) [K]
            P: Pressure(s) [Pa]
            properties: List of properties to retrieve
            
        Returns:
            Dictionary with property arrays
        """
        if not REFPROP_AVAILABLE or self.rp is None:
            raise RuntimeError("REFPROP not available")
        
        # Default properties
        if properties is None:
            properties = ['D', 'H', 'S', 'C', 'L', 'V', 'PRANDTL']
        
        # Ensure arrays
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)
        
        # Validate inputs
        self._validate_temperature_pressure(fluid_name, T, P)
        
        # Get properties
        try:
            result = self._call_refprop(fluid_name, T, P, properties)
            return self._process_refprop_result(result, properties, T, P)
            
        except Exception as e:
            print(f"⚠️  REFPROP failed for {fluid_name}: {e}")
            return self._get_fallback_properties(fluid_name, T, P, properties)
    
    def _validate_temperature_pressure(self, fluid_name: str, T: np.ndarray, P: np.ndarray):
        """Validate temperature and pressure ranges for fluid"""
        # Temperature bounds
        if fluid_name.lower() in ['helium', 'he']:
            T_min = MINIMUM_HELIUM_TEMPERATURE
        elif fluid_name.lower() in ['hydrogen', 'h2']:
            T_min = MINIMUM_HYDROGEN_TEMPERATURE
        elif fluid_name.lower() in ['nitrogen', 'n2']:
            T_min = MINIMUM_NITROGEN_TEMPERATURE
        else:
            T_min = 200.0  # General minimum
        
        if np.any(T < T_min):
            warnings.warn(f"Temperature below minimum for {fluid_name}: {T_min}K")
            T[T < T_min] = T_min
        
        # Pressure bounds
        if np.any(P < 100.0):  # 1 mbar minimum
            warnings.warn(f"Pressure below minimum: 100 Pa")
            P[P < 100.0] = 100.0
    
    def _call_refprop(self, fluid_name: str, T: np.ndarray, P: np.ndarray, 
                     properties: List[str]) -> List:
        """Call REFPROP for property calculation using ctREFPROP with vectorization"""
        # Convert to REFPROP units (K, kPa)
        T_refprop = T
        P_refprop = P / 1000.0  # REFPROP uses kPa
        
        # Map standard property names to REFPROP property codes
        prop_map = {
            'rho': 'D',         # Density
            'cp': 'CP',         # Specific heat (constant pressure)
            'k': 'TCX',         # Thermal conductivity
            'mu': 'VIS',        # Viscosity
            'Pr': 'PRANDTL',    # Prandtl number (will be computed)
            # Also support REFPROP native codes
            'D': 'D',           # Density
            'H': 'H',           # Enthalpy
            'S': 'S',           # Entropy
            'C': 'CP',          # Specific heat
            'L': 'TCX',         # Thermal conductivity
            'V': 'VIS',         # Viscosity
            'PRANDTL': 'PRANDTL' # Prandtl number
        }
        
        # ---- REFPROP vectorized helpers ----
        PROP_MAP = {
            "rho": "D",      # kg/m^3
            "cp":  "CP",     # kJ/kg-K (MKS)
            "k":   "TCX",    # W/m-K
            "mu":  "VIS",    # Pa·s
        }

        def _rp_scalar(rp, fluid, code, T, P_kpa, z=None, units=None):
            if z is None: z = [1.0] + [0.0]*19
            if units is None: units = rp.MKS      # kPa-friendly
            
            # Temperature bounds checking
            T_min = MINIMUM_HELIUM_TEMPERATURE if fluid.upper() in ['HELIUM', 'HE'] else MINIMUM_HYDROGEN_TEMPERATURE
            if T < T_min:
                raise RuntimeError(f"Temperature {T}K below minimum {T_min}K for {fluid}")
            
            out = rp.REFPROPdll(fluid, "TP", code, units, 0, 0, float(T), float(P_kpa), z)
            if out.ierr != 0:
                raise RuntimeError(out.herr)
            return out.Output[0]

        def rp_vector(rp, fluid, code, T_arr, P_arr_kpa, z=None, units=None):
            T_arr = np.atleast_1d(T_arr).astype(float)
            P_arr = np.atleast_1d(P_arr_kpa).astype(float)
            if P_arr.size == 1 and T_arr.size > 1:
                P_arr = np.full_like(T_arr, P_arr.item())
            assert T_arr.size == P_arr.size, "T and P lengths must match"
            out = np.empty_like(T_arr)
            for i, (t, p) in enumerate(zip(T_arr, P_arr)):
                out[i] = _rp_scalar(rp, fluid, code, t, p, z=z, units=units)
            return out

        def get_props_vectorized(rp, fluid, T, P_kpa, prefer_refprop=True, coolprop=None):
            """
            Returns dict with rho, cp, k, mu, Pr.
            - T, P_kpa can be scalars or arrays (same length).
            - Uses REFPROP first, falls back per-point to CoolProp if needed.
            """
            T_arr = np.atleast_1d(T).astype(float)
            P_arr = np.atleast_1d(P_kpa).astype(float)
            if P_arr.size == 1 and T_arr.size > 1:
                P_arr = np.full_like(T_arr, P_arr.item())

            results = {}
            failures = []

            def _try(code, key):
                try:
                    vals = rp_vector(rp, fluid, code, T_arr, P_arr, units=rp.MKS)
                    return vals, None
                except Exception as e:
                    return None, str(e)

            # REFPROP attempt
            if prefer_refprop and rp is not None:
                for key, code in PROP_MAP.items():
                    vals, err = _try(code, key)
                    if vals is None:
                        failures.append((key, err))
                    else:
                        results[key] = vals

            # Per-property fallback with CoolProp if any missing
            if coolprop is not None:
                import CoolProp.CoolProp as CP
                for key in PROP_MAP:
                    if key not in results:
                        vals = []
                        for t, p in zip(T_arr, P_arr):
                            try:
                                if key == "rho":
                                    vals.append(CP.PropsSI("D", "T", t, "P", p*1e3, fluid))
                                elif key == "cp":
                                    vals.append(CP.PropsSI("CPMASS", "T", t, "P", p*1e3, fluid)/1000.0)  # to kJ/kg-K
                                elif key == "k":
                                    vals.append(CP.PropsSI("L", "T", t, "P", p*1e3, fluid))
                                elif key == "mu":
                                    vals.append(CP.PropsSI("V", "T", t, "P", p*1e3, fluid))
                            except Exception as e:
                                vals.append(np.nan)
                        results[key] = np.array(vals, dtype=float)

            # Prandtl (dimensionless): cp[kJ/kg-K] -> J/kg-K
            cp_J = np.array(results["cp"])*1000.0
            mu   = np.array(results["mu"])
            k    = np.array(results["k"])
            results["Pr"] = cp_J * mu / k

            # If inputs were scalars, return scalars
            if np.isscalar(T) and np.isscalar(P_kpa):
                for k in results:
                    arr = np.atleast_1d(results[k])
                    results[k] = float(arr[0])

            return results
        
        # Use the new vectorized property retrieval
        try:
            # Get all properties using the vectorized helper
            all_props = get_props_vectorized(self.rp, fluid_name, T_refprop, P_refprop, 
                                           prefer_refprop=True, coolprop=COOLPROP_AVAILABLE)
            
            # Convert to the expected result format
            results = []
            for prop in properties:
                # Map property names
                if prop == 'PRANDTL':
                    prop_key = 'Pr'
                elif prop == 'D':
                    prop_key = 'rho'
                elif prop == 'CP':
                    prop_key = 'cp'
                elif prop == 'TCX':
                    prop_key = 'k'
                elif prop == 'VIS':
                    prop_key = 'mu'
                else:
                    prop_key = prop.lower()
                
                # Get the property values
                if prop_key in all_props:
                    prop_values = all_props[prop_key]
                else:
                    # Fallback for missing properties
                    prop_values = self._get_fallback_property(fluid_name, T, P, prop)
                
                # Create mock result object for consistency
                class MockResult:
                    def __init__(self, values):
                        self.Output = values
                
                results.append(MockResult(prop_values))
            
            return results
            
        except Exception as e:
            print(f"⚠️  Vectorized REFPROP failed: {e}, using fallback")
            # Fallback to individual property retrieval
            results = []
            for prop in properties:
                results.append(self._get_fallback_property(fluid_name, T, P, prop))
            return results
    
    def _process_refprop_result(self, results: List, properties: List[str], 
                              T: np.ndarray, P: np.ndarray) -> Dict[str, np.ndarray]:
        """Process REFPROP results into property dictionary"""
        processed = {}
        
        # Map REFPROP property names to standard names
        prop_map = {
            'D': 'rho',      # Density
            'H': 'h',        # Enthalpy
            'S': 's',        # Entropy
            'C': 'cp',       # Specific heat
            'CP': 'cp',      # Specific heat (REFPROP code)
            'L': 'k',        # Thermal conductivity
            'TCX': 'k',      # Thermal conductivity (REFPROP code)
            'V': 'mu',       # Viscosity
            'VIS': 'mu',     # Viscosity (REFPROP code)
            'PRANDTL': 'Pr'  # Prandtl number
        }
        
        # Process results
        for i, (prop, result) in enumerate(zip(properties, results)):
            prop_name = prop_map.get(prop, prop)
            
            if hasattr(result, 'Output') and len(result.Output) > 0:
                # Extract the property values from ctREFPROP output structure
                values = np.array(result.Output)
                # Check for error values (-9999990.0)
                if np.any(values == -9999990.0):
                    print(f"⚠️  REFPROP error for {prop}, using fallback")
                    values = self._get_fallback_property(prop_name, T, P, prop)
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                # Fallback to list/tuple handling
                if isinstance(result[0], (list, tuple)):
                    values = np.array(result[0])
                else:
                    values = np.array(result)
            else:
                # Fallback value
                values = self._get_fallback_property(prop_name, T, P, prop)
            
            processed[prop_name] = values
        
        # Validate properties
        for prop_name, values in processed.items():
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                warnings.warn(f"Invalid {prop_name} values detected")
                values[np.isnan(values)] = MINIMUM_DENOMINATOR
                values[np.isinf(values)] = MINIMUM_DENOMINATOR
        
        return processed
    
    def _get_fallback_property(self, fluid_name: str, T: np.ndarray, P: np.ndarray, prop: str) -> np.ndarray:
        """Get fallback property value for a single property"""
        # Simple fallback correlations - handle both property name formats
        if prop in ['D', 'rho']:
            return self._correlation_density(fluid_name, T, P)
        elif prop in ['C', 'CP', 'cp']:
            return self._correlation_cp(fluid_name, T)
        elif prop in ['L', 'TCX', 'k']:
            return self._correlation_thermal_conductivity(fluid_name, T)
        elif prop in ['V', 'VIS', 'mu']:
            return self._correlation_viscosity(fluid_name, T)
        elif prop in ['PRANDTL', 'Pr']:
            return self._correlation_prandtl(fluid_name, T)
        else:
            return np.full_like(T, 1.0)
    
    def _get_fallback_properties(self, fluid_name: str, T: np.ndarray, P: np.ndarray,
                               properties: List[str]) -> Dict[str, np.ndarray]:
        """Get fallback properties using CoolProp or built-in correlations"""
        if COOLPROP_AVAILABLE:
            return self._get_coolprop_properties(fluid_name, T, P, properties)
        else:
            return self._get_builtin_properties(fluid_name, T, P, properties)
    
    def _get_coolprop_properties(self, fluid_name: str, T: np.ndarray, P: np.ndarray,
                               properties: List[str]) -> Dict[str, np.ndarray]:
        """Get properties using CoolProp fallback"""
        # Map property names
        prop_map = {
            'rho': 'D', 'h': 'H', 's': 'S', 'cp': 'C',
            'k': 'L', 'mu': 'V', 'Pr': 'PRANDTL'
        }
        
        result = {}
        for prop in properties:
            coolprop_prop = prop_map.get(prop, prop)
            try:
                values = np.array([PropsSI(coolprop_prop, 'T', t, 'P', p, fluid_name) 
                                 for t, p in zip(T, P)])
                result[prop] = values
            except Exception as e:
                print(f"⚠️  CoolProp failed for {prop}: {e}")
                # Use default values
                if prop == 'rho':
                    result[prop] = np.full_like(T, 1000.0)
                elif prop == 'cp':
                    result[prop] = np.full_like(T, 1000.0)
                elif prop == 'k':
                    result[prop] = np.full_like(T, 0.1)
                elif prop == 'mu':
                    result[prop] = np.full_like(T, 1e-3)
                elif prop == 'Pr':
                    result[prop] = np.full_like(T, 1.0)
        
        return result
    
    def _get_builtin_properties(self, fluid_name: str, T: np.ndarray, P: np.ndarray,
                              properties: List[str]) -> Dict[str, np.ndarray]:
        """Get properties using built-in correlations"""
        # Simple temperature-dependent correlations for common fluids
        result = {}
        
        for prop in properties:
            if prop == 'rho':
                result[prop] = self._correlation_density(fluid_name, T, P)
            elif prop == 'cp':
                result[prop] = self._correlation_cp(fluid_name, T)
            elif prop == 'k':
                result[prop] = self._correlation_thermal_conductivity(fluid_name, T)
            elif prop == 'mu':
                result[prop] = self._correlation_viscosity(fluid_name, T)
            elif prop == 'Pr':
                result[prop] = self._correlation_prandtl(fluid_name, T)
        
        return result
    
    def _correlation_density(self, fluid_name: str, T: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Simple density correlation"""
        if fluid_name.lower() in ['water', 'h2o']:
            # Water density approximation
            rho_0 = 1000.0  # kg/m³ at 273.15K
            beta = 0.0002   # Thermal expansion coefficient
            return rho_0 * (1 - beta * (T - 273.15))
        elif fluid_name.lower() in ['helium', 'he']:
            # Helium density approximation
            return 0.1786 * P / (8.314 * T)  # Ideal gas law
        elif fluid_name.lower() in ['hydrogen', 'h2']:
            # Hydrogen density approximation
            return 2.016 * P / (8.314 * T)   # Ideal gas law
        else:
            # Generic ideal gas
            return 28.97 * P / (8.314 * T)   # Air-like
    
    def _correlation_cp(self, fluid_name: str, T: np.ndarray) -> np.ndarray:
        """Simple specific heat correlation"""
        if fluid_name.lower() in ['water', 'h2o']:
            # Water cp approximation
            return 4200.0 + 0.5 * (T - 273.15)  # J/kg·K
        elif fluid_name.lower() in ['helium', 'he']:
            # Helium cp approximation
            return 5193.0 + 0.1 * T  # J/kg·K
        elif fluid_name.lower() in ['hydrogen', 'h2']:
            # Hydrogen cp approximation
            return 14300.0 + 2.0 * T  # J/kg·K
        else:
            # Generic
            return 1000.0 + 0.5 * T  # J/kg·K
    
    def _correlation_thermal_conductivity(self, fluid_name: str, T: np.ndarray) -> np.ndarray:
        """Simple thermal conductivity correlation"""
        if fluid_name.lower() in ['water', 'h2o']:
            # Water k approximation
            return 0.6 + 0.001 * (T - 273.15)  # W/m·K
        elif fluid_name.lower() in ['helium', 'he']:
            # Helium k approximation
            return 0.15 + 0.0001 * T  # W/m·K
        elif fluid_name.lower() in ['hydrogen', 'h2']:
            # Hydrogen k approximation
            return 0.1 + 0.0002 * T  # W/m·K
        else:
            # Generic
            return 0.025 + 0.0001 * T  # W/m·K
    
    def _correlation_viscosity(self, fluid_name: str, T: np.ndarray) -> np.ndarray:
        """Simple viscosity correlation"""
        if fluid_name.lower() in ['water', 'h2o']:
            # Water viscosity approximation
            return 0.001 * np.exp(-0.02 * (T - 273.15))  # Pa·s
        elif fluid_name.lower() in ['helium', 'he']:
            # Helium viscosity approximation
            return 2e-5 * (T / 273.15)**0.7  # Pa·s
        elif fluid_name.lower() in ['hydrogen', 'h2']:
            # Hydrogen viscosity approximation
            return 9e-6 * (T / 273.15)**0.7  # Pa·s
        else:
            # Generic
            return 1.8e-5 * (T / 273.15)**0.7  # Pa·s
    
    def _correlation_prandtl(self, fluid_name: str, T: np.ndarray) -> np.ndarray:
        """Simple Prandtl number correlation"""
        if fluid_name.lower() in ['water', 'h2o']:
            # Water Pr approximation
            return 7.0 - 0.01 * (T - 273.15)
        elif fluid_name.lower() in ['helium', 'he']:
            # Helium Pr approximation
            return 0.7 + 0.0001 * T
        elif fluid_name.lower() in ['hydrogen', 'h2']:
            # Hydrogen Pr approximation
            return 0.7 + 0.0002 * T
        else:
            # Generic
            return 0.7 + 0.0001 * T

class FluidPropertyManager:
    """Unified fluid property manager with REFPROP/CoolProp/BuiltIn backends"""
    
    def __init__(self, preferred_backend: str = 'REFPROP'):
        """
        Initialize fluid property manager
        
        Args:
            preferred_backend: Preferred backend ('REFPROP', 'CoolProp', 'BuiltIn')
        """
        self.preferred_backend = preferred_backend
        self.refprop_interface = REFPROPInterface() if REFPROP_AVAILABLE else None
        self.backend_priority = self._setup_backend_priority()
        
        print(f"✅ Fluid Property Manager initialized with priority: {self.backend_priority}")
    
    def _setup_backend_priority(self) -> List[str]:
        """Setup backend priority based on availability"""
        priority = []
        
        if self.preferred_backend == 'REFPROP' and REFPROP_AVAILABLE:
            priority.extend(['REFPROP', 'CoolProp', 'BuiltIn'])
        elif self.preferred_backend == 'CoolProp' and COOLPROP_AVAILABLE:
            priority.extend(['CoolProp', 'BuiltIn'])
        else:
            priority.append('BuiltIn')
        
        return priority
    
    def get_fluid_properties(self, fluid_name: str, T: Union[float, np.ndarray], 
                           P: Union[float, np.ndarray] = 101325.0,
                           properties: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Get fluid properties using best available backend
        
        Args:
            fluid_name: Name of fluid
            T: Temperature(s) [K]
            P: Pressure(s) [Pa]
            properties: List of properties to retrieve
            
        Returns:
            Dictionary with property arrays
        """
        # Default properties
        if properties is None:
            properties = ['rho', 'cp', 'k', 'mu', 'Pr']
        
        # Try backends in priority order
        for backend in self.backend_priority:
            try:
                if backend == 'REFPROP' and self.refprop_interface:
                    return self.refprop_interface.get_fluid_properties(
                        fluid_name, T, P, properties)
                elif backend == 'CoolProp' and COOLPROP_AVAILABLE:
                    return self._get_coolprop_properties(fluid_name, T, P, properties)
                elif backend == 'BuiltIn':
                    return self._get_builtin_properties(fluid_name, T, P, properties)
            except Exception as e:
                print(f"⚠️  {backend} failed for {fluid_name}: {e}")
                continue
        
        # If all backends fail, return default values
        print(f"⚠️  All backends failed for {fluid_name}, using defaults")
        return self._get_default_properties(fluid_name, T, P, properties)
    
    def _get_coolprop_properties(self, fluid_name: str, T: np.ndarray, P: np.ndarray,
                               properties: List[str]) -> Dict[str, np.ndarray]:
        """Get properties using CoolProp"""
        # Map property names
        prop_map = {
            'rho': 'D', 'cp': 'C', 'k': 'L', 'mu': 'V', 'Pr': 'PRANDTL'
        }
        
        result = {}
        for prop in properties:
            coolprop_prop = prop_map.get(prop, prop)
            try:
                values = np.array([PropsSI(coolprop_prop, 'T', t, 'P', p, fluid_name) 
                                 for t, p in zip(T, P)])
                result[prop] = values
            except Exception as e:
                print(f"⚠️  CoolProp failed for {prop}: {e}")
                result[prop] = self._get_default_property(prop, T)
        
        return result
    
    def _get_builtin_properties(self, fluid_name: str, T: np.ndarray, P: np.ndarray,
                              properties: List[str]) -> Dict[str, np.ndarray]:
        """Get properties using built-in correlations"""
        if self.refprop_interface:
            return self.refprop_interface._get_builtin_properties(
                fluid_name, T, P, properties)
        else:
            # Simple fallback
            result = {}
            for prop in properties:
                result[prop] = self._get_default_property(prop, T)
            return result
    
    def _get_default_properties(self, fluid_name: str, T: np.ndarray, P: np.ndarray,
                              properties: List[str]) -> Dict[str, np.ndarray]:
        """Get default property values"""
        result = {}
        for prop in properties:
            result[prop] = self._get_default_property(prop, T)
        return result
    
    def _get_default_property(self, prop: str, T: np.ndarray) -> np.ndarray:
        """Get default property value"""
        if prop == 'rho':
            return np.full_like(T, 1000.0)  # kg/m³
        elif prop == 'cp':
            return np.full_like(T, 1000.0)  # J/kg·K
        elif prop == 'k':
            return np.full_like(T, 0.1)     # W/m·K
        elif prop == 'mu':
            return np.full_like(T, 1e-3)    # Pa·s
        elif prop == 'Pr':
            return np.full_like(T, 1.0)     # Dimensionless
        else:
            return np.full_like(T, 1.0)
    
    def get_backend_status(self) -> Dict[str, bool]:
        """Get status of available backends"""
        return {
            'REFPROP': REFPROP_AVAILABLE,
            'CoolProp': COOLPROP_AVAILABLE,
            'BuiltIn': True
        }
    
    def get_available_fluids(self) -> List[str]:
        """Get list of available fluids"""
        # This would need to be implemented based on actual REFPROP/CoolProp availability
        return ['Water', 'Helium', 'Hydrogen', 'Nitrogen', 'Air', 'CO2']

# Global instance
fluid_manager = FluidPropertyManager()

def get_fluid_properties(fluid_name: str, T: Union[float, np.ndarray], 
                        P: Union[float, np.ndarray] = 101325.0,
                        properties: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Convenience function to get fluid properties
    
    Args:
        fluid_name: Name of fluid
        T: Temperature(s) [K]
        P: Pressure(s) [Pa]
        properties: List of properties to retrieve
        
    Returns:
        Dictionary with property arrays
    """
    return fluid_manager.get_fluid_properties(fluid_name, T, P, properties)

def get_backend_status() -> Dict[str, bool]:
    """Get status of available backends"""
    return fluid_manager.get_backend_status()

def get_available_fluids() -> List[str]:
    """Get list of available fluids"""
    return fluid_manager.get_available_fluids()

# Test function
def test_refprop_integration():
    """Test REFPROP integration functionality"""
    print("🧪 Testing REFPROP Integration...")
    
    # Test backend status
    status = get_backend_status()
    print(f"Backend Status: {status}")
    
    # Test fluid properties
    T_test = np.array([273.15, 293.15, 313.15])  # 0°C, 20°C, 40°C
    P_test = 101325.0  # 1 atm
    
    test_fluids = ['Water', 'Helium', 'Hydrogen']
    
    for fluid in test_fluids:
        try:
            props = get_fluid_properties(fluid, T_test, P_test)
            print(f"✅ {fluid}: {list(props.keys())}")
        except Exception as e:
            print(f"❌ {fluid}: {e}")
    
    print("🧪 REFPROP Integration Test Complete")

if __name__ == "__main__":
    test_refprop_integration()
