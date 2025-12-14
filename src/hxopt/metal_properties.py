"""Metal thermal conductivity properties database.

Common metals used in heat exchangers with temperature-dependent
thermal conductivity data.
"""

from typing import Dict, Optional
import numpy as np
from scipy.interpolate import interp1d


class MetalProperties:
    """
    Metal thermal conductivity properties.
    
    Provides temperature-dependent thermal conductivity for common
    metals used in heat exchangers.
    """
    
    # Metal thermal conductivity data (W/(m·K)) at various temperatures
    # Data from NIST, ASM Handbook, and engineering references
    METAL_DATA: Dict[str, Dict] = {
        'Aluminum (6061)': {
            'k_300K': 167.0,  # W/(m·K) at 300K
            'k_77K': 200.0,   # W/(m·K) at 77K (cryogenic)
            'k_20K': 250.0,   # W/(m·K) at 20K (cryogenic)
            'temp_range': (20.0, 500.0),  # K
            'density': 2700.0,  # kg/m³
            'cp_300K': 896.0,   # J/(kg·K)
        },
        'Aluminum (7075)': {
            'k_300K': 130.0,
            'k_77K': 180.0,
            'k_20K': 220.0,
            'temp_range': (20.0, 500.0),
            'density': 2810.0,
            'cp_300K': 960.0,
        },
        'Copper (OFHC)': {
            'k_300K': 401.0,  # Very high conductivity
            'k_77K': 450.0,
            'k_20K': 500.0,
            'temp_range': (20.0, 600.0),
            'density': 8960.0,
            'cp_300K': 385.0,
        },
        'Stainless Steel (304)': {
            'k_300K': 16.2,
            'k_77K': 8.0,
            'k_20K': 2.0,
            'temp_range': (20.0, 1000.0),
            'density': 7900.0,
            'cp_300K': 500.0,
        },
        'Stainless Steel (316)': {
            'k_300K': 16.3,
            'k_77K': 8.5,
            'k_20K': 2.5,
            'temp_range': (20.0, 1000.0),
            'density': 8000.0,
            'cp_300K': 500.0,
        },
        'Inconel 625': {
            'k_300K': 9.8,
            'k_77K': 5.0,
            'k_20K': 1.5,
            'temp_range': (20.0, 1200.0),
            'density': 8440.0,
            'cp_300K': 410.0,
        },
        'Hastelloy X': {
            'k_300K': 11.7,
            'k_77K': 6.0,
            'k_20K': 2.0,
            'temp_range': (20.0, 1200.0),
            'density': 8220.0,
            'cp_300K': 435.0,
        },
        'Titanium (Ti-6Al-4V)': {
            'k_300K': 6.7,
            'k_77K': 4.0,
            'k_20K': 1.5,
            'temp_range': (20.0, 800.0),
            'density': 4430.0,
            'cp_300K': 526.0,
        },
        'Nickel': {
            'k_300K': 90.7,
            'k_77K': 120.0,
            'k_20K': 150.0,
            'temp_range': (20.0, 600.0),
            'density': 8900.0,
            'cp_300K': 444.0,
        },
        'Brass (C36000)': {
            'k_300K': 115.0,
            'k_77K': 140.0,
            'k_20K': 160.0,
            'temp_range': (20.0, 500.0),
            'density': 8520.0,
            'cp_300K': 380.0,
        },
    }
    
    def __init__(self, metal_name: str):
        """
        Initialize metal properties.
        
        Parameters
        ----------
        metal_name : str
            Metal name (must be in METAL_DATA)
        """
        if metal_name not in self.METAL_DATA:
            available = ', '.join(self.METAL_DATA.keys())
            raise ValueError(
                f"Unknown metal: {metal_name}. "
                f"Available: {available}"
            )
        
        self.metal_name = metal_name
        self.data = self.METAL_DATA[metal_name]
        
        # Create temperature-dependent conductivity interpolator
        # Linear interpolation between key points
        T_points = np.array([
            self.data['temp_range'][0],  # T_min
            20.0,  # Cryogenic
            77.0,  # Liquid nitrogen
            300.0,  # Room temperature
            self.data['temp_range'][1],  # T_max
        ])
        
        k_points = np.array([
            self.data['k_20K'],
            self.data['k_20K'],
            self.data['k_77K'],
            self.data['k_300K'],
            self.data['k_300K'],  # Extrapolate at high T
        ])
        
        # Remove duplicates and sort
        unique_indices = np.unique(T_points, return_index=True)[1]
        T_points = T_points[unique_indices]
        k_points = k_points[unique_indices]
        
        self._k_interp = interp1d(
            T_points, k_points,
            kind='linear',
            bounds_error=False,
            fill_value=(k_points[0], k_points[-1])
        )
    
    def thermal_conductivity(self, T: float) -> float:
        """
        Get thermal conductivity at temperature T.
        
        Parameters
        ----------
        T : float
            Temperature (K)
            
        Returns
        -------
        k : float
            Thermal conductivity (W/(m·K))
        """
        T = np.asarray(T)
        if T.ndim == 0:
            return float(self._k_interp(T))
        return self._k_interp(T).astype(float)
    
    def density(self) -> float:
        """Get density (kg/m³)."""
        return self.data['density']
    
    def specific_heat(self, T: float = 300.0) -> float:
        """
        Get specific heat capacity (approximate, temperature-independent for v1).
        
        Parameters
        ----------
        T : float, optional
            Temperature (K), default 300K
            
        Returns
        -------
        cp : float
            Specific heat capacity (J/(kg·K))
        """
        return self.data['cp_300K']  # Use room temp value for v1
    
    @classmethod
    def list_metals(cls) -> list:
        """List all available metal names."""
        return list(cls.METAL_DATA.keys())
    
    @classmethod
    def get_metal_info(cls, metal_name: str) -> Dict:
        """Get metal information dictionary."""
        if metal_name not in cls.METAL_DATA:
            raise ValueError(f"Unknown metal: {metal_name}")
        return cls.METAL_DATA[metal_name].copy()

