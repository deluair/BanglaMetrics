"""Climate Data Manager for Bangladesh GDP Simulation.

This module handles collection, processing, and management of climate data
from various sources including meteorological stations, satellite data,
and climate projections for Bangladesh.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import requests
import json
from pathlib import Path
import warnings
from dataclasses import dataclass
from enum import Enum
import random
import math

logger = logging.getLogger(__name__)


class ClimateVariable(Enum):
    """Climate variable types."""
    TEMPERATURE = "temperature"
    PRECIPITATION = "precipitation"
    HUMIDITY = "humidity"
    WIND_SPEED = "wind_speed"
    SEA_LEVEL = "sea_level"
    PRESSURE = "pressure"
    SUNSHINE = "sunshine_hours"


class ExtremeEvent(Enum):
    """Extreme weather event types."""
    CYCLONE = "cyclone"
    FLOOD = "flood"
    DROUGHT = "drought"
    HEAT_WAVE = "heat_wave"
    COLD_WAVE = "cold_wave"
    HEAVY_RAINFALL = "heavy_rainfall"
    STORM_SURGE = "storm_surge"


@dataclass
class ClimateStation:
    """Climate monitoring station information."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float
    district: str
    division: str
    station_type: str  # 'meteorological', 'hydrological', 'coastal'
    operational_since: datetime
    variables_measured: List[ClimateVariable]
    data_quality: str = "good"  # 'excellent', 'good', 'fair', 'poor'


@dataclass
class ExtremeEventRecord:
    """Record of extreme weather events."""
    event_id: str
    event_type: ExtremeEvent
    start_date: datetime
    end_date: datetime
    affected_districts: List[str]
    intensity: str  # 'low', 'moderate', 'high', 'severe', 'extreme'
    economic_impact: float  # Estimated impact in million BDT
    casualties: int
    description: str


class ClimateDataManager:
    """Comprehensive climate data management for Bangladesh.
    
    This class handles collection, processing, validation, and analysis
    of climate data from multiple sources for economic impact assessment.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the climate data manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Data storage paths
        self.data_dir = Path(self.config.get('data_dir', './data/climate'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize climate stations
        self.stations = self._initialize_stations()
        
        # Climate data cache
        self._data_cache = {}
        
        # API configurations
        self.api_configs = {
            'bmd': {
                'base_url': 'http://bmd.gov.bd/api',  # Simulated
                'api_key': self.config.get('bmd_api_key'),
                'timeout': 30
            },
            'world_bank': {
                'base_url': 'https://climateknowledgeportal.worldbank.org/api',
                'timeout': 30
            },
            'nasa': {
                'base_url': 'https://power.larc.nasa.gov/api',
                'timeout': 30
            }
        }
        
        logger.info(f"Climate data manager initialized with {len(self.stations)} stations")
    
    def _initialize_stations(self) -> Dict[str, ClimateStation]:
        """Initialize climate monitoring stations across Bangladesh."""
        
        stations = {
            'dhaka': ClimateStation(
                station_id='BMD_DHK_001',
                name='Dhaka Meteorological Station',
                latitude=23.8103,
                longitude=90.4125,
                elevation=8.0,
                district='Dhaka',
                division='Dhaka',
                station_type='meteorological',
                operational_since=datetime(1950, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.HUMIDITY,
                    ClimateVariable.WIND_SPEED,
                    ClimateVariable.PRESSURE,
                    ClimateVariable.SUNSHINE
                ]
            ),
            'chittagong': ClimateStation(
                station_id='BMD_CTG_001',
                name='Chittagong Meteorological Station',
                latitude=22.3569,
                longitude=91.7832,
                elevation=34.0,
                district='Chittagong',
                division='Chittagong',
                station_type='meteorological',
                operational_since=datetime(1948, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.HUMIDITY,
                    ClimateVariable.WIND_SPEED,
                    ClimateVariable.SEA_LEVEL
                ]
            ),
            'sylhet': ClimateStation(
                station_id='BMD_SYL_001',
                name='Sylhet Meteorological Station',
                latitude=24.8949,
                longitude=91.8687,
                elevation=34.0,
                district='Sylhet',
                division='Sylhet',
                station_type='meteorological',
                operational_since=datetime(1955, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.HUMIDITY
                ]
            ),
            'rajshahi': ClimateStation(
                station_id='BMD_RAJ_001',
                name='Rajshahi Meteorological Station',
                latitude=24.3636,
                longitude=88.6241,
                elevation=18.0,
                district='Rajshahi',
                division='Rajshahi',
                station_type='meteorological',
                operational_since=datetime(1952, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.HUMIDITY,
                    ClimateVariable.WIND_SPEED
                ]
            ),
            'khulna': ClimateStation(
                station_id='BMD_KHU_001',
                name='Khulna Meteorological Station',
                latitude=22.8456,
                longitude=89.5403,
                elevation=4.0,
                district='Khulna',
                division='Khulna',
                station_type='meteorological',
                operational_since=datetime(1953, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.HUMIDITY,
                    ClimateVariable.SEA_LEVEL
                ]
            ),
            'barisal': ClimateStation(
                station_id='BMD_BAR_001',
                name='Barisal Meteorological Station',
                latitude=22.7010,
                longitude=90.3535,
                elevation=3.0,
                district='Barisal',
                division='Barisal',
                station_type='meteorological',
                operational_since=datetime(1954, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.HUMIDITY,
                    ClimateVariable.SEA_LEVEL
                ]
            ),
            'rangpur': ClimateStation(
                station_id='BMD_RAN_001',
                name='Rangpur Meteorological Station',
                latitude=25.7439,
                longitude=89.2752,
                elevation=34.0,
                district='Rangpur',
                division='Rangpur',
                station_type='meteorological',
                operational_since=datetime(1956, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.HUMIDITY
                ]
            ),
            'mymensingh': ClimateStation(
                station_id='BMD_MYM_001',
                name='Mymensingh Meteorological Station',
                latitude=24.7471,
                longitude=90.4203,
                elevation=18.0,
                district='Mymensingh',
                division='Mymensingh',
                station_type='meteorological',
                operational_since=datetime(1957, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.HUMIDITY
                ]
            ),
            'coxs_bazar': ClimateStation(
                station_id='BMD_COX_001',
                name="Cox's Bazar Meteorological Station",
                latitude=21.4272,
                longitude=92.0058,
                elevation=3.0,
                district="Cox's Bazar",
                division='Chittagong',
                station_type='coastal',
                operational_since=datetime(1960, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.HUMIDITY,
                    ClimateVariable.WIND_SPEED,
                    ClimateVariable.SEA_LEVEL
                ]
            ),
            'teknaf': ClimateStation(
                station_id='BMD_TEK_001',
                name='Teknaf Meteorological Station',
                latitude=20.8644,
                longitude=92.2985,
                elevation=2.0,
                district='Teknaf',
                division='Chittagong',
                station_type='coastal',
                operational_since=datetime(1965, 1, 1),
                variables_measured=[
                    ClimateVariable.TEMPERATURE,
                    ClimateVariable.PRECIPITATION,
                    ClimateVariable.WIND_SPEED,
                    ClimateVariable.SEA_LEVEL
                ]
            )
        }
        
        return stations
    
    def collect_climate_data(self, 
                           start_date: str,
                           end_date: str,
                           variables: List[ClimateVariable] = None,
                           stations: List[str] = None,
                           source: str = 'all') -> Dict[str, pd.DataFrame]:
        """Collect climate data from various sources.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of climate variables to collect
            stations: List of station IDs to collect from
            source: Data source ('bmd', 'satellite', 'all')
            
        Returns:
            Dictionary of DataFrames with climate data
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if variables is None:
                variables = list(ClimateVariable)
            
            if stations is None:
                stations = list(self.stations.keys())
            
            collected_data = {}
            
            # Collect from each source
            if source in ['bmd', 'all']:
                bmd_data = self._collect_bmd_data(start_dt, end_dt, variables, stations)
                collected_data.update(bmd_data)
            
            if source in ['satellite', 'all']:
                satellite_data = self._collect_satellite_data(start_dt, end_dt, variables, stations)
                collected_data.update(satellite_data)
            
            # Generate synthetic data for demonstration
            if not collected_data or source == 'synthetic':
                synthetic_data = self._generate_synthetic_data(start_dt, end_dt, variables, stations)
                collected_data.update(synthetic_data)
            
            logger.info(f"Collected climate data for {len(collected_data)} datasets")
            return collected_data
            
        except Exception as e:
            logger.error(f"Error collecting climate data: {str(e)}")
            return {}
    
    def _collect_bmd_data(self, 
                         start_date: datetime,
                         end_date: datetime,
                         variables: List[ClimateVariable],
                         stations: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect data from Bangladesh Meteorological Department."""
        
        # Since BMD doesn't have a public API, we'll simulate the data collection
        logger.info("Simulating BMD data collection (no public API available)")
        
        return self._generate_synthetic_data(start_date, end_date, variables, stations, source='bmd')
    
    def _collect_satellite_data(self, 
                              start_date: datetime,
                              end_date: datetime,
                              variables: List[ClimateVariable],
                              stations: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect satellite-based climate data."""
        
        # Simulate satellite data collection
        logger.info("Simulating satellite data collection")
        
        return self._generate_synthetic_data(start_date, end_date, variables, stations, source='satellite')
    
    def _generate_synthetic_data(self, 
                               start_date: datetime,
                               end_date: datetime,
                               variables: List[ClimateVariable],
                               stations: List[str],
                               source: str = 'synthetic') -> Dict[str, pd.DataFrame]:
        """Generate realistic synthetic climate data for Bangladesh."""
        
        data_dict = {}
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for station_id in stations:
            if station_id not in self.stations:
                continue
                
            station = self.stations[station_id]
            
            # Initialize data dictionary for this station
            station_data = {
                'date': date_range,
                'station_id': station_id,
                'station_name': station.name,
                'latitude': station.latitude,
                'longitude': station.longitude,
                'source': source
            }
            
            # Generate data for each variable
            for variable in variables:
                if variable in station.variables_measured:
                    station_data.update(self._generate_variable_data(
                        variable, date_range, station
                    ))
            
            # Create DataFrame
            df = pd.DataFrame(station_data)
            data_dict[f"{station_id}_{source}"] = df
        
        return data_dict
    
    def _generate_variable_data(self, 
                              variable: ClimateVariable,
                              date_range: pd.DatetimeIndex,
                              station: ClimateStation) -> Dict[str, List]:
        """Generate realistic data for a specific climate variable."""
        
        data = {}
        
        if variable == ClimateVariable.TEMPERATURE:
            # Generate temperature data with seasonal patterns
            temp_data = self._generate_temperature_data(date_range, station)
            data.update(temp_data)
            
        elif variable == ClimateVariable.PRECIPITATION:
            # Generate precipitation data with monsoon patterns
            precip_data = self._generate_precipitation_data(date_range, station)
            data.update(precip_data)
            
        elif variable == ClimateVariable.HUMIDITY:
            # Generate humidity data
            humidity_data = self._generate_humidity_data(date_range, station)
            data.update(humidity_data)
            
        elif variable == ClimateVariable.WIND_SPEED:
            # Generate wind speed data
            wind_data = self._generate_wind_data(date_range, station)
            data.update(wind_data)
            
        elif variable == ClimateVariable.SEA_LEVEL:
            # Generate sea level data (for coastal stations)
            if station.station_type == 'coastal':
                sea_level_data = self._generate_sea_level_data(date_range, station)
                data.update(sea_level_data)
        
        return data
    
    def _generate_temperature_data(self, 
                                 date_range: pd.DatetimeIndex,
                                 station: ClimateStation) -> Dict[str, List]:
        """Generate realistic temperature data for Bangladesh."""
        
        # Base temperature parameters by region
        temp_params = {
            'Dhaka': {'base': 26, 'amplitude': 8, 'winter_offset': -2},
            'Chittagong': {'base': 25, 'amplitude': 6, 'winter_offset': -1},
            'Sylhet': {'base': 24, 'amplitude': 7, 'winter_offset': -3},
            'Rajshahi': {'base': 27, 'amplitude': 9, 'winter_offset': -4},
            'Khulna': {'base': 26, 'amplitude': 7, 'winter_offset': -2},
            'Barisal': {'base': 25, 'amplitude': 6, 'winter_offset': -1},
            'Rangpur': {'base': 25, 'amplitude': 10, 'winter_offset': -5},
            'Mymensingh': {'base': 25, 'amplitude': 8, 'winter_offset': -3}
        }
        
        params = temp_params.get(station.district, temp_params['Dhaka'])
        
        min_temps = []
        max_temps = []
        avg_temps = []
        
        for date in date_range:
            # Seasonal variation
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = math.sin(2 * math.pi * (day_of_year - 80) / 365)
            
            # Base temperature with seasonal variation
            base_temp = params['base'] + params['amplitude'] * seasonal_factor
            
            # Winter adjustment
            if date.month in [12, 1, 2]:
                base_temp += params['winter_offset']
            
            # Daily variation
            daily_range = random.uniform(6, 12)
            noise = random.gauss(0, 1)
            
            avg_temp = base_temp + noise
            min_temp = avg_temp - daily_range/2 + random.gauss(0, 0.5)
            max_temp = avg_temp + daily_range/2 + random.gauss(0, 0.5)
            
            min_temps.append(round(min_temp, 1))
            max_temps.append(round(max_temp, 1))
            avg_temps.append(round(avg_temp, 1))
        
        return {
            'temperature_min_c': min_temps,
            'temperature_max_c': max_temps,
            'temperature_avg_c': avg_temps
        }
    
    def _generate_precipitation_data(self, 
                                   date_range: pd.DatetimeIndex,
                                   station: ClimateStation) -> Dict[str, List]:
        """Generate realistic precipitation data with monsoon patterns."""
        
        # Monsoon parameters by region
        precip_params = {
            'Dhaka': {'annual': 2000, 'monsoon_share': 0.75},
            'Chittagong': {'annual': 2500, 'monsoon_share': 0.80},
            'Sylhet': {'annual': 3500, 'monsoon_share': 0.85},
            'Rajshahi': {'annual': 1400, 'monsoon_share': 0.70},
            'Khulna': {'annual': 1800, 'monsoon_share': 0.75},
            'Barisal': {'annual': 2200, 'monsoon_share': 0.80},
            'Rangpur': {'annual': 1600, 'monsoon_share': 0.70},
            'Mymensingh': {'annual': 2200, 'monsoon_share': 0.75}
        }
        
        params = precip_params.get(station.district, precip_params['Dhaka'])
        
        precipitation = []
        
        for date in date_range:
            month = date.month
            
            # Monsoon season (June-September)
            if month in [6, 7, 8, 9]:
                # High probability of rain during monsoon
                rain_prob = 0.6 if month in [7, 8] else 0.4
                base_intensity = params['annual'] * params['monsoon_share'] / (4 * 30)
            # Pre-monsoon (March-May)
            elif month in [3, 4, 5]:
                rain_prob = 0.2
                base_intensity = params['annual'] * 0.1 / (3 * 30)
            # Post-monsoon (October-November)
            elif month in [10, 11]:
                rain_prob = 0.3
                base_intensity = params['annual'] * 0.1 / (2 * 30)
            # Winter (December-February)
            else:
                rain_prob = 0.1
                base_intensity = params['annual'] * 0.05 / (3 * 30)
            
            # Determine if it rains
            if random.random() < rain_prob:
                # Generate rainfall amount
                if month in [7, 8]:  # Peak monsoon
                    rainfall = random.expovariate(1/base_intensity) * random.uniform(1, 3)
                else:
                    rainfall = random.expovariate(1/base_intensity)
                
                # Add extreme events occasionally
                if random.random() < 0.02:  # 2% chance of extreme rainfall
                    rainfall *= random.uniform(3, 8)
                
                precipitation.append(round(rainfall, 1))
            else:
                precipitation.append(0.0)
        
        return {'precipitation_mm': precipitation}
    
    def _generate_humidity_data(self, 
                              date_range: pd.DatetimeIndex,
                              station: ClimateStation) -> Dict[str, List]:
        """Generate realistic humidity data."""
        
        # Base humidity by region and season
        humidity_base = {
            'coastal': 80,  # Higher humidity for coastal areas
            'inland': 70,   # Lower humidity for inland areas
            'northern': 65  # Lowest humidity for northern regions
        }
        
        # Determine station type
        if station.station_type == 'coastal':
            base_humidity = humidity_base['coastal']
        elif station.district in ['Rangpur', 'Rajshahi']:
            base_humidity = humidity_base['northern']
        else:
            base_humidity = humidity_base['inland']
        
        humidity = []
        
        for date in date_range:
            month = date.month
            
            # Seasonal adjustment
            if month in [6, 7, 8, 9]:  # Monsoon - higher humidity
                seasonal_adj = 10
            elif month in [12, 1, 2]:  # Winter - lower humidity
                seasonal_adj = -10
            else:
                seasonal_adj = 0
            
            # Daily variation
            daily_humidity = base_humidity + seasonal_adj + random.gauss(0, 5)
            
            # Ensure realistic range
            daily_humidity = max(30, min(95, daily_humidity))
            
            humidity.append(round(daily_humidity, 1))
        
        return {'relative_humidity_percent': humidity}
    
    def _generate_wind_data(self, 
                          date_range: pd.DatetimeIndex,
                          station: ClimateStation) -> Dict[str, List]:
        """Generate realistic wind speed data."""
        
        wind_speeds = []
        wind_directions = []
        
        for date in date_range:
            month = date.month
            
            # Base wind speed varies by season
            if month in [3, 4, 5]:  # Pre-monsoon - higher winds
                base_speed = 15
            elif month in [6, 7, 8, 9]:  # Monsoon - variable winds
                base_speed = 12
            else:  # Winter/post-monsoon - calmer
                base_speed = 8
            
            # Add variability
            wind_speed = max(0, base_speed + random.gauss(0, 3))
            
            # Occasional strong winds (storms/cyclones)
            if random.random() < 0.01:  # 1% chance
                wind_speed *= random.uniform(2, 4)
            
            # Wind direction (degrees from north)
            if month in [6, 7, 8, 9]:  # Monsoon - predominantly south/southwest
                direction = random.gauss(225, 45) % 360
            else:  # Other seasons - more variable
                direction = random.uniform(0, 360)
            
            wind_speeds.append(round(wind_speed, 1))
            wind_directions.append(round(direction, 0))
        
        return {
            'wind_speed_kmh': wind_speeds,
            'wind_direction_degrees': wind_directions
        }
    
    def _generate_sea_level_data(self, 
                               date_range: pd.DatetimeIndex,
                               station: ClimateStation) -> Dict[str, List]:
        """Generate realistic sea level data for coastal stations."""
        
        sea_levels = []
        
        # Base sea level (relative to mean sea level)
        base_level = 0.0
        
        for i, date in enumerate(date_range):
            # Tidal variation (simplified)
            tidal_cycle = math.sin(2 * math.pi * i / 14.5)  # ~14.5 day cycle
            tidal_variation = tidal_cycle * 2.0  # ±2 meters tidal range
            
            # Seasonal variation
            day_of_year = date.timetuple().tm_yday
            seasonal_variation = math.sin(2 * math.pi * day_of_year / 365) * 0.3
            
            # Storm surge (occasional)
            storm_surge = 0
            if random.random() < 0.005:  # 0.5% chance of storm surge
                storm_surge = random.uniform(1, 4)
            
            # Climate change trend (gradual increase)
            years_since_2000 = (date.year - 2000)
            climate_trend = years_since_2000 * 0.003  # 3mm per year
            
            # Random noise
            noise = random.gauss(0, 0.1)
            
            sea_level = (base_level + tidal_variation + seasonal_variation + 
                        storm_surge + climate_trend + noise)
            
            sea_levels.append(round(sea_level, 3))
        
        return {'sea_level_m': sea_levels}
    
    def get_extreme_events(self, 
                          start_date: str,
                          end_date: str,
                          event_types: List[ExtremeEvent] = None) -> List[ExtremeEventRecord]:
        """Get extreme weather events for the specified period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            event_types: List of event types to include
            
        Returns:
            List of extreme event records
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if event_types is None:
                event_types = list(ExtremeEvent)
            
            # Generate synthetic extreme events
            events = self._generate_extreme_events(start_dt, end_dt, event_types)
            
            logger.info(f"Retrieved {len(events)} extreme events")
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving extreme events: {str(e)}")
            return []
    
    def _generate_extreme_events(self, 
                               start_date: datetime,
                               end_date: datetime,
                               event_types: List[ExtremeEvent]) -> List[ExtremeEventRecord]:
        """Generate realistic extreme weather events."""
        
        events = []
        current_date = start_date
        
        # Event probabilities by month
        event_probabilities = {
            ExtremeEvent.CYCLONE: {
                4: 0.1, 5: 0.2, 6: 0.1, 9: 0.1, 10: 0.3, 11: 0.2
            },
            ExtremeEvent.FLOOD: {
                6: 0.2, 7: 0.4, 8: 0.4, 9: 0.3, 10: 0.1
            },
            ExtremeEvent.DROUGHT: {
                1: 0.1, 2: 0.2, 3: 0.3, 4: 0.2, 11: 0.1, 12: 0.1
            },
            ExtremeEvent.HEAT_WAVE: {
                3: 0.1, 4: 0.3, 5: 0.4, 6: 0.2
            }
        }
        
        event_id_counter = 1
        
        while current_date <= end_date:
            month = current_date.month
            
            for event_type in event_types:
                if event_type in event_probabilities:
                    prob = event_probabilities[event_type].get(month, 0)
                    
                    # Check if event occurs (monthly probability)
                    if random.random() < prob / 30:  # Daily probability
                        event = self._create_extreme_event(
                            event_id_counter, event_type, current_date
                        )
                        events.append(event)
                        event_id_counter += 1
            
            current_date += timedelta(days=1)
        
        return events
    
    def _create_extreme_event(self, 
                            event_id: int,
                            event_type: ExtremeEvent,
                            start_date: datetime) -> ExtremeEventRecord:
        """Create a realistic extreme event record."""
        
        # Event duration by type
        duration_ranges = {
            ExtremeEvent.CYCLONE: (1, 3),
            ExtremeEvent.FLOOD: (3, 14),
            ExtremeEvent.DROUGHT: (30, 120),
            ExtremeEvent.HEAT_WAVE: (3, 10),
            ExtremeEvent.HEAVY_RAINFALL: (1, 2)
        }
        
        duration_days = random.randint(*duration_ranges.get(event_type, (1, 5)))
        end_date = start_date + timedelta(days=duration_days)
        
        # Affected districts by event type
        all_districts = list(self.stations.keys())
        
        if event_type == ExtremeEvent.CYCLONE:
            # Cyclones mainly affect coastal areas
            coastal_districts = ['chittagong', 'coxs_bazar', 'barisal', 'khulna']
            affected = random.sample(coastal_districts, random.randint(2, 4))
            intensity = random.choice(['moderate', 'high', 'severe'])
            economic_impact = random.uniform(500, 5000)  # Million BDT
            casualties = random.randint(0, 100)
            
        elif event_type == ExtremeEvent.FLOOD:
            # Floods can affect multiple regions
            affected = random.sample(all_districts, random.randint(3, 6))
            intensity = random.choice(['moderate', 'high', 'severe'])
            economic_impact = random.uniform(200, 2000)
            casualties = random.randint(0, 50)
            
        elif event_type == ExtremeEvent.DROUGHT:
            # Droughts mainly affect northern and western regions
            drought_prone = ['rajshahi', 'rangpur', 'khulna']
            affected = random.sample(drought_prone, random.randint(1, 3))
            intensity = random.choice(['moderate', 'high'])
            economic_impact = random.uniform(100, 1000)
            casualties = random.randint(0, 10)
            
        elif event_type == ExtremeEvent.HEAT_WAVE:
            # Heat waves affect inland areas more
            inland_districts = ['dhaka', 'rajshahi', 'rangpur', 'mymensingh']
            affected = random.sample(inland_districts, random.randint(2, 4))
            intensity = random.choice(['moderate', 'high'])
            economic_impact = random.uniform(50, 500)
            casualties = random.randint(0, 20)
            
        else:
            affected = random.sample(all_districts, random.randint(1, 3))
            intensity = 'moderate'
            economic_impact = random.uniform(10, 100)
            casualties = 0
        
        return ExtremeEventRecord(
            event_id=f"EXT_{event_type.value.upper()}_{event_id:04d}",
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            affected_districts=affected,
            intensity=intensity,
            economic_impact=economic_impact,
            casualties=casualties,
            description=f"{intensity.title()} {event_type.value.replace('_', ' ')} affecting {', '.join(affected)}"
        )
    
    def calculate_climate_indices(self, 
                                climate_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate climate indices from raw climate data.
        
        Args:
            climate_data: Dictionary of climate DataFrames
            
        Returns:
            Dictionary of climate indices
        """
        try:
            indices = {}
            
            for station_id, df in climate_data.items():
                if df.empty:
                    continue
                
                station_indices = {}
                
                # Temperature indices
                if 'temperature_max_c' in df.columns:
                    station_indices['hot_days'] = (df['temperature_max_c'] > 35).sum()
                    station_indices['very_hot_days'] = (df['temperature_max_c'] > 40).sum()
                
                if 'temperature_min_c' in df.columns:
                    station_indices['cold_days'] = (df['temperature_min_c'] < 10).sum()
                
                # Precipitation indices
                if 'precipitation_mm' in df.columns:
                    station_indices['rainy_days'] = (df['precipitation_mm'] > 1).sum()
                    station_indices['heavy_rain_days'] = (df['precipitation_mm'] > 50).sum()
                    station_indices['extreme_rain_days'] = (df['precipitation_mm'] > 100).sum()
                    station_indices['total_precipitation'] = df['precipitation_mm'].sum()
                    station_indices['max_daily_precipitation'] = df['precipitation_mm'].max()
                
                # Wind indices
                if 'wind_speed_kmh' in df.columns:
                    station_indices['windy_days'] = (df['wind_speed_kmh'] > 30).sum()
                    station_indices['very_windy_days'] = (df['wind_speed_kmh'] > 50).sum()
                    station_indices['max_wind_speed'] = df['wind_speed_kmh'].max()
                
                # Create indices DataFrame
                if station_indices:
                    indices_df = pd.DataFrame([station_indices])
                    indices_df['station_id'] = station_id
                    indices_df['period_start'] = df['date'].min()
                    indices_df['period_end'] = df['date'].max()
                    indices_df['total_days'] = len(df)
                    
                    indices[f"{station_id}_indices"] = indices_df
            
            logger.info(f"Calculated climate indices for {len(indices)} stations")
            return indices
            
        except Exception as e:
            logger.error(f"Error calculating climate indices: {str(e)}")
            return {}
    
    def get_climate_summary(self, 
                          start_date: str,
                          end_date: str,
                          region: str = None) -> Dict[str, Any]:
        """Get climate summary for the specified period and region.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            region: Specific region/division
            
        Returns:
            Climate summary dictionary
        """
        try:
            # Collect climate data
            climate_data = self.collect_climate_data(start_date, end_date)
            
            # Filter by region if specified
            if region:
                filtered_data = {}
                for station_id, df in climate_data.items():
                    station_name = station_id.split('_')[0]
                    if station_name in self.stations:
                        station = self.stations[station_name]
                        if station.division.lower() == region.lower():
                            filtered_data[station_id] = df
                climate_data = filtered_data
            
            # Calculate summary statistics
            summary = {
                'period': {'start': start_date, 'end': end_date},
                'region': region or 'All Bangladesh',
                'stations_included': len(climate_data),
                'temperature': {},
                'precipitation': {},
                'extreme_events': {},
                'climate_risks': []
            }
            
            # Aggregate temperature data
            all_temps = []
            for df in climate_data.values():
                if 'temperature_avg_c' in df.columns:
                    all_temps.extend(df['temperature_avg_c'].tolist())
            
            if all_temps:
                summary['temperature'] = {
                    'mean': round(np.mean(all_temps), 1),
                    'min': round(np.min(all_temps), 1),
                    'max': round(np.max(all_temps), 1),
                    'std': round(np.std(all_temps), 1)
                }
            
            # Aggregate precipitation data
            all_precip = []
            for df in climate_data.values():
                if 'precipitation_mm' in df.columns:
                    all_precip.extend(df['precipitation_mm'].tolist())
            
            if all_precip:
                summary['precipitation'] = {
                    'total': round(np.sum(all_precip), 1),
                    'mean_daily': round(np.mean(all_precip), 1),
                    'max_daily': round(np.max(all_precip), 1),
                    'rainy_days': len([p for p in all_precip if p > 1])
                }
            
            # Get extreme events
            events = self.get_extreme_events(start_date, end_date)
            summary['extreme_events'] = {
                'total_events': len(events),
                'by_type': {}
            }
            
            for event in events:
                event_type = event.event_type.value
                if event_type not in summary['extreme_events']['by_type']:
                    summary['extreme_events']['by_type'][event_type] = 0
                summary['extreme_events']['by_type'][event_type] += 1
            
            # Assess climate risks
            summary['climate_risks'] = self._assess_climate_risks(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating climate summary: {str(e)}")
            return {}
    
    def _assess_climate_risks(self, summary: Dict[str, Any]) -> List[str]:
        """Assess climate risks based on summary data."""
        
        risks = []
        
        # Temperature risks
        if summary.get('temperature', {}).get('max', 0) > 40:
            risks.append("Extreme heat risk - temperatures above 40°C recorded")
        
        if summary.get('temperature', {}).get('mean', 0) > 30:
            risks.append("High temperature stress on agriculture and labor productivity")
        
        # Precipitation risks
        precip_data = summary.get('precipitation', {})
        if precip_data.get('max_daily', 0) > 100:
            risks.append("Flash flood risk - extreme daily rainfall recorded")
        
        if precip_data.get('total', 0) < 1000:
            risks.append("Drought risk - below normal precipitation")
        
        # Extreme event risks
        events = summary.get('extreme_events', {}).get('by_type', {})
        if events.get('cyclone', 0) > 0:
            risks.append("Cyclone damage to coastal infrastructure and agriculture")
        
        if events.get('flood', 0) > 2:
            risks.append("Repeated flooding affecting multiple sectors")
        
        if not risks:
            risks.append("Normal climate conditions - low risk")
        
        return risks
    
    def export_climate_data(self, 
                          data: Dict[str, pd.DataFrame],
                          output_dir: str,
                          format: str = 'csv') -> bool:
        """Export climate data to files.
        
        Args:
            data: Climate data dictionary
            output_dir: Output directory path
            format: Export format ('csv', 'xlsx', 'json')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for dataset_name, df in data.items():
                if df.empty:
                    continue
                
                filename = f"{dataset_name}.{format}"
                filepath = output_path / filename
                
                if format == 'csv':
                    df.to_csv(filepath, index=False)
                elif format == 'xlsx':
                    df.to_excel(filepath, index=False)
                elif format == 'json':
                    df.to_json(filepath, orient='records', indent=2, date_format='iso')
                else:
                    logger.error(f"Unsupported format: {format}")
                    return False
            
            logger.info(f"Exported {len(data)} climate datasets to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting climate data: {str(e)}")
            return False