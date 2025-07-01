"""Climate Module for Bangladesh GDP Simulation.

This module provides comprehensive climate data management and impact assessment
for Bangladesh's economic sectors, incorporating monsoon patterns, extreme weather
events, and climate change projections.
"""

from .climate_data import ClimateDataManager
from .climate_impact import ClimateImpactAssessment
from .weather_patterns import WeatherPatternsAnalyzer
from .adaptation_measures import ClimateAdaptationManager

__version__ = "1.0.0"
__author__ = "BanglaMetrics Team"

# Climate module metadata
CLIMATE_METADATA = {
    "description": "Climate data and impact assessment for Bangladesh GDP simulation",
    "key_features": [
        "Monsoon pattern analysis",
        "Extreme weather event tracking",
        "Climate impact on agriculture",
        "Sea level rise projections",
        "Temperature and precipitation trends",
        "Cyclone frequency and intensity",
        "Drought and flood risk assessment",
        "Climate adaptation strategies"
    ],
    "data_sources": [
        "Bangladesh Meteorological Department (BMD)",
        "Climate Change Cell (CCC)",
        "World Bank Climate Data",
        "NASA Climate Data",
        "NOAA Climate Data",
        "IPCC Climate Projections"
    ],
    "climate_variables": {
        "temperature": {
            "description": "Daily temperature data (min, max, average)",
            "unit": "Celsius",
            "frequency": "daily",
            "spatial_resolution": "district_level"
        },
        "precipitation": {
            "description": "Daily rainfall data",
            "unit": "millimeters",
            "frequency": "daily",
            "spatial_resolution": "district_level"
        },
        "humidity": {
            "description": "Relative humidity",
            "unit": "percentage",
            "frequency": "daily",
            "spatial_resolution": "regional"
        },
        "wind_speed": {
            "description": "Wind speed and direction",
            "unit": "km/h",
            "frequency": "daily",
            "spatial_resolution": "regional"
        },
        "sea_level": {
            "description": "Sea level measurements",
            "unit": "meters",
            "frequency": "monthly",
            "spatial_resolution": "coastal_stations"
        }
    },
    "extreme_events": {
        "cyclones": {
            "description": "Tropical cyclone tracking and intensity",
            "frequency": "event_based",
            "impact_sectors": ["agriculture", "infrastructure", "manufacturing"]
        },
        "floods": {
            "description": "Flood extent and duration",
            "frequency": "event_based",
            "impact_sectors": ["agriculture", "transport", "urban_services"]
        },
        "droughts": {
            "description": "Drought severity and duration",
            "frequency": "seasonal",
            "impact_sectors": ["agriculture", "water_supply", "energy"]
        },
        "heat_waves": {
            "description": "Extreme temperature events",
            "frequency": "seasonal",
            "impact_sectors": ["health", "energy", "labor_productivity"]
        }
    },
    "climate_zones": {
        "northern_region": {
            "districts": ["Rangpur", "Kurigram", "Lalmonirhat", "Nilphamari", "Thakurgaon"],
            "characteristics": "Dry winter, moderate rainfall",
            "main_risks": ["drought", "river_erosion"]
        },
        "central_region": {
            "districts": ["Dhaka", "Gazipur", "Manikganj", "Tangail", "Mymensingh"],
            "characteristics": "Moderate climate, urban heat island",
            "main_risks": ["flooding", "heat_waves"]
        },
        "southern_region": {
            "districts": ["Barisal", "Patuakhali", "Bhola", "Pirojpur", "Jhalokati"],
            "characteristics": "High humidity, cyclone prone",
            "main_risks": ["cyclones", "sea_level_rise", "salinity"]
        },
        "southeastern_region": {
            "districts": ["Chittagong", "Cox's Bazar", "Bandarban", "Rangamati", "Khagrachhari"],
            "characteristics": "High rainfall, hilly terrain",
            "main_risks": ["landslides", "cyclones", "heavy_rainfall"]
        },
        "southwestern_region": {
            "districts": ["Khulna", "Satkhira", "Bagerhat", "Jessore", "Narail"],
            "characteristics": "Coastal, mangrove forests",
            "main_risks": ["sea_level_rise", "salinity", "cyclones"]
        }
    },
    "seasonal_patterns": {
        "winter": {
            "months": ["December", "January", "February"],
            "characteristics": "Dry, cool, low humidity",
            "economic_impact": "Favorable for agriculture and construction"
        },
        "pre_monsoon": {
            "months": ["March", "April", "May"],
            "characteristics": "Hot, dry, occasional storms",
            "economic_impact": "Heat stress on labor, energy demand"
        },
        "monsoon": {
            "months": ["June", "July", "August", "September"],
            "characteristics": "Heavy rainfall, high humidity, floods",
            "economic_impact": "Critical for agriculture, transport disruption"
        },
        "post_monsoon": {
            "months": ["October", "November"],
            "characteristics": "Moderate temperature, decreasing rainfall",
            "economic_impact": "Harvest season, recovery from floods"
        }
    },
    "climate_change_projections": {
        "temperature_increase": {
            "2030": "1.0-1.5°C",
            "2050": "1.5-2.5°C",
            "2100": "2.5-4.0°C"
        },
        "precipitation_change": {
            "monsoon_increase": "10-20%",
            "winter_decrease": "5-15%",
            "variability_increase": "High"
        },
        "sea_level_rise": {
            "2030": "10-20 cm",
            "2050": "20-40 cm",
            "2100": "50-100 cm"
        },
        "extreme_events": {
            "cyclone_intensity": "Increase by 10-20%",
            "flood_frequency": "Increase by 20-30%",
            "drought_severity": "Increase in northern regions"
        }
    }
}

# Export all components
__all__ = [
    'ClimateDataManager',
    'ClimateImpactAssessment', 
    'WeatherPatternsAnalyzer',
    'ClimateAdaptationManager',
    'CLIMATE_METADATA'
]