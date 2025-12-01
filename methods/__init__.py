"""
OpenMaterial Methods Module

This module provides a unified interface for different 3D reconstruction methods.
"""

from .wrappers.neus2_wrapper import NeuS2Method
from .wrappers.twodgs_wrapper import TwoDGSMethod
from .wrappers.pgsr_wrapper import PGSRMethod
from .wrappers.instant_nsr_wrapper import InstantNSRMethod

__all__ = [
    'NeuS2Method',
    'TwoDGSMethod',
    'PGSRMethod',
    'InstantNSRMethod',
]

# Method registry
METHODS = {
    'neus2': NeuS2Method,
    '2dgs': TwoDGSMethod,
    'pgsr': PGSRMethod,
    'instant-nsr-pl': InstantNSRMethod,
}

def get_method(name: str):
    """Get method class by name"""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHODS.keys())}")
    return METHODS[name]
