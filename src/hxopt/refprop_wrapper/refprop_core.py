#!/usr/bin/env python3
"""
REFPROP Core Interface for HyFlux Heat Exchanger Engines
Provides unified fluid property access across all engine files
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import logging
import math

# Import numpy safely
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a simple replacement for basic functionality
    np = type('numpy', (), {
        'log': math.log,
        'sqrt': math.sqrt
    })

# Configure logging - use WARNING level to reduce noise
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# Set specific loggers to reduce verbosity
logging.getLogger('hxopt.refprop_wrapper.refprop_core').setLevel(logging.WARNING)

class FluidType(Enum):
    """Supported fluid types"""
    HELIUM = "HELIUM"
    HYDROGEN = "HYDROGEN"
    NITROGEN = "NITROGEN"
    OXYGEN = "OXYGEN"
    ARGON = "ARGON"
    NEON = "NEON"
    METHANE = "METHANE"
    ETHANE = "ETHANE"
    PROPANE = "PROPANE"
    BUTANE = "BUTANE"
    WATER = "WATER"
    AIR = "AIR"
    CO2 = "CO2"
    AMMONIA = "AMMONIA"
    R134A = "R134A"
    R410A = "R410A"
    R404A = "R404A"
    R507A = "R507A"
    R22 = "R22"
    R12 = "R12"

class PropertyType(Enum):
    """Available fluid properties"""
    DENSITY = "D"
    ENTHALPY = "H"
    ENTROPY = "S"
    INTERNAL_ENERGY = "U"
    GIBBS_ENERGY = "G"
    HELMHOLTZ_ENERGY = "A"
    THERMAL_CONDUCTIVITY = "TCX"
    VISCOSITY = "VIS"
    SURFACE_TENSION = "ST"
    HEAT_CAPACITY_CP = "CP"
    HEAT_CAPACITY_CV = "CV"
    SPEED_OF_SOUND = "W"
    ISENTROPIC_EXPONENT = "G"
    ISOTHERMAL_COMPRESSIBILITY = "KT"
    VOLUME_EXPANSION_COEFFICIENT = "ALPHA"
    JOULE_THOMSON_COEFFICIENT = "JT"
    CRITICAL_PRESSURE = "PCRIT"
    CRITICAL_TEMPERATURE = "TCRIT"
    CRITICAL_DENSITY = "DCRIT"
    TRIPLE_PRESSURE = "PTRIP"
    TRIPLE_TEMPERATURE = "TTRIP"
    TRIPLE_DENSITY = "DTRIP"
    MOLECULAR_WEIGHT = "M"
    ACENTRIC_FACTOR = "ACENTRIC"
    DIPOLE_MOMENT = "DIPOLE"

class REFPROPInterface:
    """
    Core REFPROP interface providing unified fluid property access
    Handles fallbacks to CoolProp and basic correlations when REFPROP unavailable
    """
    
    def __init__(self, refprop_library_path: Optional[str] = None, 
                 refprop_data_path: Optional[str] = None):
        """
        Initialize REFPROP interface
        
        Parameters:
        -----------
        refprop_library_path : str, optional
            Path to REFPROP library (librefprop.dylib, refprop.dll, etc.)
        refprop_data_path : str, optional
            Path to REFPROP data directory
        """
        self.refprop_available = False
        self.coolprop_available = False
        self.basic_correlations_available = True
        self.refprop_library_path = refprop_library_path
        self.refprop_data_path = refprop_data_path
        self.rp_instance = None  # Store REFPROP instance
        
        # Set environment variables if provided
        if refprop_library_path:
            os.environ['REFPROP_LIBRARY'] = refprop_library_path
        if refprop_data_path:
            os.environ['RPPREFIX'] = refprop_data_path
            
        # Initialize property providers
        self._init_refprop()
        self._init_coolprop()
        self._init_basic_correlations()
        
        logger.info(f"REFPROP Interface initialized - REFPROP: {self.refprop_available}, "
                   f"CoolProp: {self.coolprop_available}, Basic: {self.basic_correlations_available}")
    
    def _init_refprop(self):
        """Initialize REFPROP library"""
        try:
            import ctREFPROP
            from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
            self.ctREFPROP = ctREFPROP
            self.REFPROPFunctionLibrary = REFPROPFunctionLibrary
            
            # Find library path if not provided
            if not self.refprop_library_path:
                self.refprop_library_path = self._find_refprop_library()
            
            # Create REFPROP instance if library path is available
            if self.refprop_library_path and os.path.exists(self.refprop_library_path):
                try:
                    self.rp_instance = REFPROPFunctionLibrary(self.refprop_library_path)
                    
                    # Set data path if available
                    if self.refprop_data_path:
                        self.rp_instance.SETPATHdll(self.refprop_data_path)
                    elif 'RPPREFIX' in os.environ:
                        self.rp_instance.SETPATHdll(os.environ['RPPREFIX'])
                    
                    # Test REFPROP functionality
                    if self._test_refprop_basic():
                        self.refprop_available = True
                        logger.info("✅ REFPROP library initialized successfully")
                    else:
                        logger.warning("⚠️ REFPROP library found but basic functionality test failed")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize REFPROP instance: {e}")
            else:
                logger.warning("⚠️ REFPROP library path not found")
                
        except ImportError:
            logger.warning("⚠️ ctREFPROP not available - REFPROP functionality disabled")
        except Exception as e:
            logger.error(f"❌ REFPROP initialization error: {e}")
    
    def _find_refprop_library(self) -> Optional[str]:
        """Find REFPROP library path."""
        import platform
        
        # Check environment variable first
        if 'REFPROP_LIBRARY' in os.environ:
            path = os.environ['REFPROP_LIBRARY']
            if os.path.exists(path):
                return path
        
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
        
        # Find REFPROP_9 directory and look for library there
        def _find_refprop9_paths():
            """Find potential REFPROP_9 paths."""
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            
            paths = [
                os.path.join(project_root, 'REFPROP_9'),
                os.path.join(project_root, 'Hyflux', 'REFPROP_9'),
                os.path.join(os.path.expanduser('~'), 'HyFlux_Hx', 'Hyflux', 'REFPROP_9'),
                os.path.join(os.path.dirname(project_root), 'HyFlux_Hx', 'Hyflux', 'REFPROP_9'),
                os.environ.get('REFPROP9_PATH', ''),
            ]
            return [p for p in paths if p and os.path.exists(p)]
        
        # Check REFPROP_9 directories first (most likely location)
        refprop9_paths = _find_refprop9_paths()
        for refprop9_path in refprop9_paths:
            for lib_name in lib_names:
                lib_path = os.path.join(refprop9_path, lib_name)
                if os.path.exists(lib_path):
                    return lib_path
        
        # Common system locations
        common_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'REFPROP_9', lib_names[0]),
            os.path.join(os.path.expanduser('~'), 'REFPROP_9', lib_names[0]),
            '/usr/local/lib/' + lib_names[0],
            '/Applications/REFPROP/' + lib_names[0],
        ]
        
        for path in common_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        return None
    
    def _init_coolprop(self):
        """Initialize CoolProp as fallback"""
        try:
            from CoolProp.CoolProp import PropsSI
            self.PropsSI = PropsSI
            self.coolprop_available = True
            logger.info("✅ CoolProp fallback available")
        except ImportError:
            logger.warning("⚠️ CoolProp not available - will use basic correlations")
    
    def _init_basic_correlations(self):
        """Initialize basic fluid property correlations"""
        self.basic_fluids = {
            'HELIUM': self._helium_properties,
            'HYDROGEN': self._hydrogen_properties,
            'NITROGEN': self._nitrogen_properties,
            'OXYGEN': self._oxygen_properties,
            'ARGON': self._argon_properties,
            'NEON': self._neon_properties,
            'METHANE': self._methane_properties,
            'WATER': self._water_properties,
            'AIR': self._air_properties
        }
        logger.info(f"✅ Basic correlations available for {len(self.basic_fluids)} fluids")
    
    def _test_refprop_basic(self) -> bool:
        """Test basic REFPROP functionality"""
        if self.rp_instance is None:
            return False
        
        try:
            # Test basic property call using the instance
            # T=77K, P=101325Pa (1 atm), pure fluid z=[1.0]
            result = self.rp_instance.REFPROPdll("HELIUM", "TP", "D", 0, 0, 0, 77.0, 101.325, [1.0])
            if hasattr(result, 'ierr'):
                return result.ierr == 0
            else:
                # If result is a tuple/list, assume success if no error
                return True
        except Exception as e:
            logger.error(f"REFPROP test failed: {e}")
            return False
    
    def get_property(self, fluid: str, prop_type: str, T: float, P: float, 
                    quality: Optional[float] = None, **kwargs) -> float:
        """
        Get fluid property using best available method
        
        Parameters:
        -----------
        fluid : str
            Fluid name (e.g., 'HELIUM', 'HYDROGEN')
        prop_type : str
            Property type (e.g., 'D', 'H', 'TCX')
        T : float
            Temperature (K)
        P : float
            Pressure (Pa)
        quality : float, optional
            Vapor quality (0-1) for two-phase calculations
        **kwargs : additional parameters
            
        Returns:
        --------
        float : Property value in SI units
        """
        # Try REFPROP first
        if self.refprop_available:
            try:
                return self._get_refprop_property(fluid, prop_type, T, P, quality, **kwargs)
            except (RuntimeError, ValueError) as e:
                # RuntimeError for data file issues, ValueError for other errors
                # Only log if it's not a known recoverable error
                if "data file error" not in str(e).lower():
                    logger.warning(f"REFPROP failed for {fluid} {prop_type}: {e}")
            except Exception as e:
                # Other unexpected errors
                logger.warning(f"REFPROP failed for {fluid} {prop_type}: {e}")
        
        # Try CoolProp as fallback
        if self.coolprop_available:
            try:
                return self._get_coolprop_property(fluid, prop_type, T, P, quality, **kwargs)
            except Exception as e:
                # Suppress CoolProp warnings for common issues
                if "Output parameter parsing" not in str(e):
                    logger.warning(f"CoolProp failed for {fluid} {prop_type}: {e}")
        
        # Use basic correlations as final fallback
        if fluid.upper() in self.basic_fluids:
            try:
                return self._get_basic_property(fluid, prop_type, T, P, quality, **kwargs)
            except Exception as e:
                # Try to provide a reasonable default
                logger.debug(f"Basic correlation failed for {fluid} {prop_type}: {e}")
                # Will raise if no default available
                raise ValueError(f"Unable to calculate {prop_type} for {fluid} at T={T}K, P={P}Pa")
        else:
            # Try to provide defaults for unsupported fluids
            try:
                return self._get_basic_property(fluid, prop_type, T, P, quality, **kwargs)
            except Exception:
                raise ValueError(f"Fluid {fluid} not supported and no fallback available")
    
    def _get_refprop_property(self, fluid: str, prop_type: str, T: float, P: float,
                             quality: Optional[float] = None, **kwargs) -> float:
        """Get property using REFPROP"""
        if self.rp_instance is None:
            raise RuntimeError("REFPROP instance not initialized")
        
        try:
            # Use the stored REFPROP instance
            rp = self.rp_instance
            
            # Property type mapping
            prop_map = {
                'D': 'D',  # Density
                'DENSITY': 'D',
                'V': 'V',  # Viscosity
                'VIS': 'V',
                'VISCOSITY': 'V',
                'C': 'CP',  # Specific heat
                'CP': 'CP',
                'HEAT_CAPACITY_CP': 'CP',
                'L': 'TCX',  # Thermal conductivity
                'TCX': 'TCX',
                'THERMAL_CONDUCTIVITY': 'TCX',
            }
            
            refprop_prop = prop_map.get(prop_type, prop_type)
            
            # Call REFPROP with correct API
            # REFPROPdll(fluid, inputs, outputs, iMass, iFlag, iUnits, ...)
            # For TP (temperature, pressure) input: T in K, P in kPa
            # z is composition array (pure fluid = [1.0])
            z = [1.0]  # Pure fluid
            
            result = rp.REFPROPdll(fluid, "TP", refprop_prop, 0, 0, 0, T, P / 1000.0, z)
                
            # Check for errors
            if hasattr(result, 'ierr') and result.ierr != 0:
                # Error 102 and -29 are data file issues - should fall back gracefully
                # Error 119 is convergence failure - also should fall back
                if result.ierr in [102, -29, 119]:
                    # These are recoverable - raise to trigger fallback
                    raise RuntimeError(f"REFPROP data file error {result.ierr}: {getattr(result, 'herr', 'Unknown error')}")
                else:
                    raise ValueError(f"REFPROP error {result.ierr}: {getattr(result, 'herr', 'Unknown error')}")
                
            # Extract output value
            if hasattr(result, 'Output'):
                return result.Output[0]
            elif isinstance(result, (list, tuple)):
                return result[0]
            else:
                return float(result)
                
        except Exception as e:
            raise RuntimeError(f"REFPROP calculation failed: {e}")
    
    def _get_coolprop_property(self, fluid: str, prop_type: str, T: float, P: float,
                               quality: Optional[float] = None, **kwargs) -> float:
        """Get property using CoolProp"""
        try:
            # Map REFPROP property types to CoolProp
            prop_map = {
                'D': 'D',
                'H': 'H',
                'S': 'S',
                'U': 'U',
                'TCX': 'L',
                'VIS': 'V',
                'CP': 'C',
                'CV': 'O',
                'W': 'A'
            }
            
            coolprop_prop = prop_map.get(prop_type, prop_type)
            
            if quality is not None:
                # Two-phase calculation
                return self.PropsSI(coolprop_prop, 'T', T, 'P', P, 'Q', quality, fluid)
            else:
                # Single-phase calculation
                return self.PropsSI(coolprop_prop, 'T', T, 'P', P, fluid)
                
        except Exception as e:
            raise RuntimeError(f"CoolProp calculation failed: {e}")
    
    def _get_basic_property(self, fluid: str, prop_type: str, T: float, P: float,
                           quality: Optional[float] = None, **kwargs) -> float:
        """Get property using basic correlations"""
        fluid_func = self.basic_fluids.get(fluid.upper())
        if not fluid_func:
            # Try to provide a reasonable default instead of failing
            if prop_type in ['D', 'DENSITY']:
                # Ideal gas approximation
                R = 2077.1 if 'HELIUM' in fluid.upper() else 4124.3  # Helium or Hydrogen
                return P / (R * T)
            elif prop_type in ['CP', 'C', 'HEAT_CAPACITY_CP']:
                # Default specific heat
                return 5000.0 if 'HELIUM' in fluid.upper() else 10000.0
            elif prop_type in ['V', 'VIS', 'VISCOSITY']:
                # Default viscosity
                return 2.0e-5 if 'HELIUM' in fluid.upper() else 9.0e-6
            elif prop_type in ['TCX', 'L', 'THERMAL_CONDUCTIVITY']:
                # Default thermal conductivity
                return 0.15 if 'HELIUM' in fluid.upper() else 0.1
            raise ValueError(f"Fluid {fluid} not supported in basic correlations")
        
        return fluid_func(prop_type, T, P, quality, **kwargs)
    
    # Basic correlation implementations
    def _helium_properties(self, prop_type: str, T: float, P: float, 
                           quality: Optional[float] = None, **kwargs) -> float:
        """Basic helium property correlations"""
        if prop_type == 'D':  # Density
            # Ideal gas law approximation for helium
            R = 2077.1  # J/(kg·K) for helium
            return P / (R * T)
        elif prop_type == 'H':  # Enthalpy
            # Simplified enthalpy (reference at 0K)
            cp = 5193.2  # J/(kg·K) for helium
            return cp * T
        elif prop_type == 'S':  # Entropy
            # Simplified entropy (reference at 0K)
            cp = 5193.2  # J/(kg·K) for helium
            R = 2077.1  # J/(kg·K) for helium
            return cp * np.log(T / 273.15) - R * np.log(P / 101325.0)
        elif prop_type == 'U':  # Internal energy
            # Internal energy = enthalpy - pressure*volume
            h = 5193.2 * T  # J/kg
            v = 1.0 / (P / (2077.1 * T))  # m³/kg
            return h - P * v
        elif prop_type == 'TCX':  # Thermal conductivity
            # Basic thermal conductivity correlation
            return 0.144 * (T / 273.15) ** 0.7
        elif prop_type == 'VIS':  # Viscosity
            # Sutherland's formula for helium
            T0 = 273.15
            mu0 = 1.87e-5  # Pa·s at T0
            S = 79.4  # Sutherland constant
            return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        elif prop_type == 'CP':  # Heat capacity at constant pressure
            return 5193.2  # J/(kg·K) for helium (nearly constant)
        elif prop_type == 'C':  # Also accept 'C' for specific heat
            return 5193.2
        elif prop_type == 'HEAT_CAPACITY_CP':  # Full name
            return 5193.2
        elif prop_type == 'CV':  # Heat capacity at constant volume
            return 3116.1  # J/(kg·K) for helium (CP - R)
        elif prop_type == 'W':  # Speed of sound
            # Speed of sound = sqrt(gamma * R * T)
            gamma = 1.67  # Specific heat ratio for helium
            R = 2077.1  # J/(kg·K) for helium
            return (gamma * R * T) ** 0.5
        else:
            raise ValueError(f"Property {prop_type} not implemented for helium in basic correlations")
    
    def _hydrogen_properties(self, prop_type: str, T: float, P: float,
                            quality: Optional[float] = None, **kwargs) -> float:
        """Basic hydrogen property correlations"""
        if prop_type == 'D':
            R = 4124.3  # J/(kg·K) for hydrogen
            return P / (R * T)
        elif prop_type == 'H':
            cp = 14300.0  # J/(kg·K) for hydrogen
            return cp * T
        elif prop_type == 'CP' or prop_type == 'C' or prop_type == 'HEAT_CAPACITY_CP':
            # Specific heat capacity for hydrogen (liquid at cryogenic temps)
            # Use temperature-dependent value
            if T < 30.0:  # Liquid hydrogen range
                return 9600.0  # J/(kg·K) for liquid hydrogen
            else:
                return 14300.0  # J/(kg·K) for gas
        elif prop_type == 'TCX':
            return 0.167 * (T / 273.15) ** 0.8
        elif prop_type == 'VIS':
            T0 = 273.15
            mu0 = 8.4e-6  # Pa·s at T0
            S = 47.0  # Sutherland constant
            return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        else:
            raise ValueError(f"Property {prop_type} not implemented for hydrogen in basic correlations")
    
    def _nitrogen_properties(self, prop_type: str, T: float, P: float,
                            quality: Optional[float] = None, **kwargs) -> float:
        """Basic nitrogen property correlations"""
        if prop_type == 'D':
            R = 296.8  # J/(kg·K) for nitrogen
            return P / (R * T)
        elif prop_type == 'H':
            cp = 1040.0  # J/(kg·K) for nitrogen
            return cp * T
        elif prop_type == 'TCX':
            return 0.0242 * (T / 273.15) ** 0.8
        elif prop_type == 'VIS':
            T0 = 273.15
            mu0 = 1.66e-5  # Pa·s at T0
            S = 111.0  # Sutherland constant
            return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        else:
            raise ValueError(f"Property {prop_type} not implemented for nitrogen in basic correlations")
    
    def _oxygen_properties(self, prop_type: str, T: float, P: float,
                          quality: Optional[float] = None, **kwargs) -> float:
        """Basic oxygen property correlations"""
        if prop_type == 'D':
            R = 259.8  # J/(kg·K) for oxygen
            return P / (R * T)
        elif prop_type == 'H':
            cp = 919.0  # J/(kg·K) for oxygen
            return cp * T
        elif prop_type == 'TCX':
            return 0.0246 * (T / 273.15) ** 0.8
        elif prop_type == 'VIS':
            T0 = 273.15
            mu0 = 1.92e-5  # Pa·s at T0
            S = 125.0  # Sutherland constant
            return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        else:
            raise ValueError(f"Property {prop_type} not implemented for oxygen in basic correlations")
    
    def _argon_properties(self, prop_type: str, T: float, P: float,
                         quality: Optional[float] = None, **kwargs) -> float:
        """Basic argon property correlations"""
        if prop_type == 'D':
            R = 208.1  # J/(kg·K) for argon
            return P / (R * T)
        elif prop_type == 'H':
            cp = 520.3  # J/(kg·K) for argon
            return cp * T
        elif prop_type == 'TCX':
            return 0.0177 * (T / 273.15) ** 0.8
        elif prop_type == 'VIS':
            T0 = 273.15
            mu0 = 2.10e-5  # Pa·s at T0
            S = 144.0  # Sutherland constant
            return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        else:
            raise ValueError(f"Property {prop_type} not implemented for argon in basic correlations")
    
    def _neon_properties(self, prop_type: str, T: float, P: float,
                        quality: Optional[float] = None, **kwargs) -> float:
        """Basic neon property correlations"""
        if prop_type == 'D':
            R = 412.0  # J/(kg·K) for neon
            return P / (R * T)
        elif prop_type == 'H':
            cp = 1030.0  # J/(kg·K) for neon
            return cp * T
        elif prop_type == 'TCX':
            return 0.0491 * (T / 273.15) ** 0.8
        elif prop_type == 'VIS':
            T0 = 273.15
            mu0 = 3.12e-5  # Pa·s at T0
            S = 56.0  # Sutherland constant
            return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        else:
            raise ValueError(f"Property {prop_type} not implemented for neon in basic correlations")
    
    def _methane_properties(self, prop_type: str, T: float, P: float,
                           quality: Optional[float] = None, **kwargs) -> float:
        """Basic methane property correlations"""
        if prop_type == 'D':
            R = 518.3  # J/(kg·K) for methane
            return P / (R * T)
        elif prop_type == 'H':
            cp = 2220.0  # J/(kg·K) for methane
            return cp * T
        elif prop_type == 'TCX':
            return 0.0302 * (T / 273.15) ** 0.8
        elif prop_type == 'VIS':
            T0 = 273.15
            mu0 = 1.10e-5  # Pa·s at T0
            S = 198.0  # Sutherland constant
            return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        else:
            raise ValueError(f"Property {prop_type} not implemented for methane in basic correlations")
    
    def _water_properties(self, prop_type: str, T: float, P: float,
                         quality: Optional[float] = None, **kwargs) -> float:
        """Basic water property correlations"""
        if prop_type == 'D':
            # Simplified water density (liquid phase approximation)
            if T < 373.15:  # Below boiling point
                return 1000.0 * (1.0 - 0.0002 * (T - 273.15))
            else:
                # Vapor phase - ideal gas approximation
                R = 461.5  # J/(kg·K) for water vapor
                return P / (R * T)
        elif prop_type == 'H':
            # Simplified enthalpy
            cp_liquid = 4186.0  # J/(kg·K) for liquid water
            h_vap = 2257000.0  # J/kg latent heat of vaporization
            if T < 373.15:
                return cp_liquid * (T - 273.15)
            else:
                return cp_liquid * 100.0 + h_vap + 2000.0 * (T - 373.15)  # Vapor
        elif prop_type == 'TCX':
            return 0.6 * (T / 273.15) ** 0.5
        elif prop_type == 'VIS':
            T0 = 273.15
            mu0 = 1.79e-3  # Pa·s at T0
            S = 650.0  # Sutherland constant
            return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        else:
            raise ValueError(f"Property {prop_type} not implemented for water in basic correlations")
    
    def _air_properties(self, prop_type: str, T: float, P: float,
                       quality: Optional[float] = None, **kwargs) -> float:
        """Basic air property correlations"""
        if prop_type == 'D':
            R = 287.1  # J/(kg·K) for air
            return P / (R * T)
        elif prop_type == 'H':
            cp = 1005.0  # J/(kg·K) for air
            return cp * T
        elif prop_type == 'TCX':
            return 0.0242 * (T / 273.15) ** 0.8
        elif prop_type == 'VIS':
            T0 = 273.15
            mu0 = 1.73e-5  # Pa·s at T0
            S = 111.0  # Sutherland constant
            return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        else:
            raise ValueError(f"Property {prop_type} not implemented for air in basic correlations")
    
    def get_fluid_summary(self, fluid: str, T: float, P: float) -> Dict[str, float]:
        """
        Get comprehensive fluid properties summary
        
        Parameters:
        -----------
        fluid : str
            Fluid name
        T : float
            Temperature (K)
        P : float
            Pressure (Pa)
            
        Returns:
        --------
        Dict[str, float] : Dictionary of available properties
        """
        properties = {}
        available_props = [
            'D', 'H', 'S', 'U', 'TCX', 'VIS', 'CP', 'CV', 'W'
        ]
        
        for prop in available_props:
            try:
                properties[prop] = self.get_property(fluid, prop, T, P)
            except Exception as e:
                logger.warning(f"Failed to get {prop} for {fluid}: {e}")
                properties[prop] = None
        
        return properties
    
    def validate_fluid_support(self, fluid: str) -> Dict[str, bool]:
        """
        Validate which property calculation methods are available for a fluid
        
        Parameters:
        -----------
        fluid : str
            Fluid name
            
        Returns:
        --------
        Dict[str, bool] : Availability status for each method
        """
        T_test = 300.0  # K
        P_test = 101325.0  # Pa
        
        validation = {
            'refprop': False,
            'coolprop': False,
            'basic_correlations': False
        }
        
        # Test REFPROP
        if self.refprop_available:
            try:
                self._get_refprop_property(fluid, 'D', T_test, P_test)
                validation['refprop'] = True
            except:
                pass
        
        # Test CoolProp
        if self.coolprop_available:
            try:
                self._get_coolprop_property(fluid, 'D', T_test, P_test)
                validation['coolprop'] = True
            except:
                pass
        
        # Test basic correlations
        if fluid.upper() in self.basic_fluids:
            try:
                self._get_basic_property(fluid, 'D', T_test, P_test)
                validation['basic_correlations'] = True
            except:
                pass
        
        return validation
