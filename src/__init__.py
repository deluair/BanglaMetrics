"""BanglaMetrics: Bangladesh GDP Calculation Reality Simulation

A comprehensive economic simulation platform that models the complex realities
of GDP calculation in Bangladesh from 2005-2025.
"""

__version__ = "1.0.0"
__author__ = "BanglaMetrics Development Team"
__email__ = "contact@banglametrics.org"

# Core modules
from . import core
from . import sectors
from . import data
from . import climate

__all__ = [
    "core",
    "sectors", 
    "data",
    "climate"
]