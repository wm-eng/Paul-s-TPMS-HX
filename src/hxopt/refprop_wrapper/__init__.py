"""REFPROP wrapper from HyFlux_Hx repository."""

# Import the REFPROP interfaces from HyFlux_Hx
try:
    from .refprop_core import REFPROPInterface, FluidType, PropertyType
    from .refprop_integration import REFPROPInterface as REFPROPIntegrationInterface
    HAS_HYFLUX_REFPROP = True
except ImportError:
    HAS_HYFLUX_REFPROP = False
    REFPROPInterface = None
    REFPROPIntegrationInterface = None

__all__ = [
    'REFPROPInterface',
    'REFPROPIntegrationInterface',
    'FluidType',
    'PropertyType',
    'HAS_HYFLUX_REFPROP',
]

