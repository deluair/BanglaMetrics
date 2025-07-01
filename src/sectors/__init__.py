"""Economic Sectors Module for Bangladesh GDP Simulation.

This module contains detailed models for Bangladesh's major economic sectors,
including agriculture, manufacturing, services, and their sub-sectors.
Each sector model incorporates Bangladesh-specific characteristics,
production functions, and external factors.
"""

from .agriculture import AgricultureSector
from .manufacturing import ManufacturingSector
from .services import ServicesSector
from .construction import ConstructionSector
from .informal_economy import InformalEconomySector

__all__ = [
    'AgricultureSector',
    'ManufacturingSector', 
    'ServicesSector',
    'ConstructionSector',
    'InformalEconomySector'
]

# Sector metadata
SECTOR_INFO = {
    'agriculture': {
        'name': 'Agriculture, Forestry and Fishing',
        'bbs_code': 'A',
        'gdp_share_2024': 15.2,  # Percentage of GDP
        'employment_share': 38.1,  # Percentage of total employment
        'key_subsectors': ['crops', 'livestock', 'forestry', 'fishing'],
        'climate_sensitive': True,
        'export_oriented': True
    },
    'manufacturing': {
        'name': 'Manufacturing',
        'bbs_code': 'C',
        'gdp_share_2024': 20.8,
        'employment_share': 16.2,
        'key_subsectors': ['rmg', 'textiles', 'food_processing', 'pharmaceuticals', 'leather'],
        'climate_sensitive': False,
        'export_oriented': True
    },
    'services': {
        'name': 'Services',
        'bbs_code': 'G-U',
        'gdp_share_2024': 52.7,
        'employment_share': 39.8,
        'key_subsectors': ['trade', 'transport', 'finance', 'telecom', 'education', 'health'],
        'climate_sensitive': False,
        'export_oriented': False
    },
    'construction': {
        'name': 'Construction',
        'bbs_code': 'F',
        'gdp_share_2024': 8.9,
        'employment_share': 4.8,
        'key_subsectors': ['residential', 'commercial', 'infrastructure'],
        'climate_sensitive': True,
        'export_oriented': False
    },
    'informal_economy': {
        'name': 'Informal Economy',
        'bbs_code': 'INF',
        'gdp_share_2024': 35.6,  # Cross-cutting across all sectors
        'employment_share': 85.1,
        'key_subsectors': ['street_vendors', 'home_workers', 'domestic_workers', 'rickshaw_pullers'],
        'climate_sensitive': True,
        'export_oriented': False
    }
}

# Version information
__version__ = '1.0.0'
__author__ = 'BanglaMetrics Development Team'
__description__ = 'Economic sector models for Bangladesh GDP simulation'