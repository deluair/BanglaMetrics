"""Weather Patterns Analysis for Bangladesh GDP Simulation.

This module provides comprehensive analysis of weather patterns, seasonal trends,
and climate variability for Bangladesh's economic modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import math
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class Season(Enum):
    """Bangladesh seasons."""
    WINTER = "winter"  # December-February
    PRE_MONSOON = "pre_monsoon"  # March-May
    MONSOON = "monsoon"  # June-September
    POST_MONSOON = "post_monsoon"  # October-November


class WeatherPattern(Enum):
    """Weather pattern types."""
    NORMAL = "normal"
    EL_NINO = "el_nino"
    LA_NINA = "la_nina"
    INDIAN_OCEAN_DIPOLE = "indian_ocean_dipole"
    MONSOON_BREAK = "monsoon_break"
    HEAT_WAVE = "heat_wave"
    COLD_WAVE = "cold_wave"


@dataclass
class SeasonalStatistics:
    """Seasonal weather statistics."""
    season: Season
    temperature_mean: float
    temperature_std: float
    temperature_min: float
    temperature_max: float
    precipitation_total: float
    precipitation_std: float
    rainy_days: int
    humidity_mean: float
    wind_speed_mean: float
    sunshine_hours: float


@dataclass
class WeatherAnomaly:
    """Weather anomaly detection result."""
    date: datetime
    variable: str
    observed_value: float
    expected_value: float
    anomaly_score: float
    severity: str  # 'mild', 'moderate', 'severe', 'extreme'
    description: str


@dataclass
class ClimateIndex:
    """Climate index calculation result."""
    name: str
    value: float
    category: str
    description: str
    impact_sectors: List[str]


class WeatherPatternsAnalyzer:
    """Comprehensive weather patterns analysis for Bangladesh.
    
    This class analyzes seasonal patterns, climate variability, weather anomalies,
    and provides insights for economic impact assessment.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the weather patterns analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize seasonal parameters
        self.seasonal_parameters = self._initialize_seasonal_parameters()
        
        # Climate indices thresholds
        self.climate_indices = self._initialize_climate_indices()
        
        # Regional characteristics
        self.regional_characteristics = self._initialize_regional_characteristics()
        
        logger.info("Weather patterns analyzer initialized")
    
    def _initialize_seasonal_parameters(self) -> Dict[Season, Dict]:
        """Initialize seasonal weather parameters for Bangladesh."""
        
        return {
            Season.WINTER: {
                'months': [12, 1, 2],
                'temperature_range': (10, 25),  # Celsius
                'precipitation_range': (5, 50),  # mm/month
                'humidity_range': (55, 75),  # %
                'wind_speed_range': (5, 15),  # km/h
                'characteristics': ['dry', 'cool', 'clear_skies', 'low_humidity']
            },
            Season.PRE_MONSOON: {
                'months': [3, 4, 5],
                'temperature_range': (20, 35),
                'precipitation_range': (50, 200),
                'humidity_range': (65, 85),
                'wind_speed_range': (10, 25),
                'characteristics': ['hot', 'humid', 'thunderstorms', 'nor_westers']
            },
            Season.MONSOON: {
                'months': [6, 7, 8, 9],
                'temperature_range': (25, 32),
                'precipitation_range': (200, 600),
                'humidity_range': (80, 95),
                'wind_speed_range': (15, 35),
                'characteristics': ['wet', 'humid', 'heavy_rainfall', 'flooding']
            },
            Season.POST_MONSOON: {
                'months': [10, 11],
                'temperature_range': (20, 30),
                'precipitation_range': (20, 150),
                'humidity_range': (70, 85),
                'wind_speed_range': (8, 20),
                'characteristics': ['mild', 'retreating_monsoon', 'cyclones', 'clear_weather']
            }
        }
    
    def _initialize_climate_indices(self) -> Dict[str, Dict]:
        """Initialize climate indices and their thresholds."""
        
        return {
            'heat_index': {
                'thresholds': {
                    'normal': (0, 27),
                    'caution': (27, 32),
                    'extreme_caution': (32, 41),
                    'danger': (41, 54),
                    'extreme_danger': (54, 100)
                },
                'formula': 'apparent_temperature'
            },
            'drought_index': {
                'thresholds': {
                    'wet': (1.5, 10),
                    'normal': (0.5, 1.5),
                    'mild_drought': (-0.5, 0.5),
                    'moderate_drought': (-1.0, -0.5),
                    'severe_drought': (-1.5, -1.0),
                    'extreme_drought': (-10, -1.5)
                },
                'formula': 'standardized_precipitation_index'
            },
            'comfort_index': {
                'thresholds': {
                    'comfortable': (18, 24),
                    'slightly_warm': (24, 27),
                    'warm': (27, 30),
                    'hot': (30, 35),
                    'very_hot': (35, 100)
                },
                'formula': 'temperature_humidity_index'
            },
            'monsoon_index': {
                'thresholds': {
                    'weak': (0, 0.5),
                    'normal': (0.5, 1.5),
                    'strong': (1.5, 2.5),
                    'very_strong': (2.5, 10)
                },
                'formula': 'monsoon_strength_ratio'
            }
        }
    
    def _initialize_regional_characteristics(self) -> Dict[str, Dict]:
        """Initialize regional weather characteristics."""
        
        return {
            'coastal': {
                'temperature_moderation': 0.8,  # Ocean moderates temperature
                'humidity_increase': 1.2,
                'cyclone_risk': 0.9,
                'sea_breeze_effect': True,
                'characteristics': ['moderate_temperature', 'high_humidity', 'cyclone_prone']
            },
            'northern': {
                'temperature_extremes': 1.3,  # More extreme temperatures
                'winter_cooling': 1.5,
                'drought_risk': 0.8,
                'fog_frequency': 1.4,
                'characteristics': ['temperature_extremes', 'winter_fog', 'drought_prone']
            },
            'northeastern': {
                'rainfall_enhancement': 1.4,  # Orographic effect
                'temperature_moderation': 0.9,
                'flash_flood_risk': 1.2,
                'characteristics': ['high_rainfall', 'flash_floods', 'moderate_temperature']
            },
            'central': {
                'average_conditions': 1.0,
                'river_influence': 1.1,
                'flood_risk': 1.0,
                'characteristics': ['typical_bangladesh_weather', 'river_influenced']
            },
            'southwestern': {
                'salinity_effect': 1.1,
                'tidal_influence': 1.2,
                'storm_surge_risk': 0.8,
                'characteristics': ['saline_environment', 'tidal_effects']
            }
        }
    
    def analyze_seasonal_patterns(self, 
                                climate_data: Dict[str, pd.DataFrame],
                                years: int = 5) -> Dict[Season, SeasonalStatistics]:
        """Analyze seasonal weather patterns.
        
        Args:
            climate_data: Climate data by station
            years: Number of years to analyze
            
        Returns:
            Dictionary of seasonal statistics
        """
        try:
            # Aggregate data across all stations
            aggregated_data = self._aggregate_station_data(climate_data)
            
            if aggregated_data.empty:
                logger.warning("No climate data available for seasonal analysis")
                return {}
            
            # Ensure date column is datetime
            if 'date' in aggregated_data.columns:
                aggregated_data['date'] = pd.to_datetime(aggregated_data['date'])
                aggregated_data['month'] = aggregated_data['date'].dt.month
            else:
                logger.error("Date column not found in climate data")
                return {}
            
            seasonal_stats = {}
            
            for season in Season:
                season_months = self.seasonal_parameters[season]['months']
                season_data = aggregated_data[aggregated_data['month'].isin(season_months)]
                
                if season_data.empty:
                    continue
                
                # Calculate seasonal statistics
                stats = self._calculate_seasonal_statistics(season_data, season)
                seasonal_stats[season] = stats
            
            logger.info(f"Analyzed seasonal patterns for {len(seasonal_stats)} seasons")
            return seasonal_stats
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {str(e)}")
            return {}
    
    def _aggregate_station_data(self, climate_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate climate data across all stations."""
        
        all_data = []
        
        for station_id, df in climate_data.items():
            if df.empty:
                continue
            
            # Add station identifier
            df_copy = df.copy()
            df_copy['station_id'] = station_id
            all_data.append(df_copy)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Aggregate by date (average across stations)
        numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'station_id']
        
        if 'date' in combined_df.columns:
            aggregated = combined_df.groupby('date')[numeric_columns].mean().reset_index()
        else:
            aggregated = combined_df[numeric_columns].mean().to_frame().T
        
        return aggregated
    
    def _calculate_seasonal_statistics(self, 
                                     season_data: pd.DataFrame,
                                     season: Season) -> SeasonalStatistics:
        """Calculate statistics for a specific season."""
        
        # Temperature statistics
        temp_col = 'temperature_avg_c'
        if temp_col in season_data.columns:
            temp_mean = season_data[temp_col].mean()
            temp_std = season_data[temp_col].std()
            temp_min = season_data[temp_col].min()
            temp_max = season_data[temp_col].max()
        else:
            temp_mean = temp_std = temp_min = temp_max = np.nan
        
        # Precipitation statistics
        precip_col = 'precipitation_mm'
        if precip_col in season_data.columns:
            precip_total = season_data[precip_col].sum()
            precip_std = season_data[precip_col].std()
            rainy_days = (season_data[precip_col] > 1.0).sum()  # Days with >1mm rain
        else:
            precip_total = precip_std = rainy_days = np.nan
        
        # Humidity statistics
        humidity_col = 'relative_humidity_percent'
        if humidity_col in season_data.columns:
            humidity_mean = season_data[humidity_col].mean()
        else:
            humidity_mean = np.nan
        
        # Wind speed statistics
        wind_col = 'wind_speed_kmh'
        if wind_col in season_data.columns:
            wind_speed_mean = season_data[wind_col].mean()
        else:
            wind_speed_mean = np.nan
        
        # Sunshine hours (estimated from other variables if not available)
        sunshine_col = 'sunshine_hours'
        if sunshine_col in season_data.columns:
            sunshine_hours = season_data[sunshine_col].mean()
        else:
            # Estimate based on season and cloud cover
            sunshine_hours = self._estimate_sunshine_hours(season, season_data)
        
        return SeasonalStatistics(
            season=season,
            temperature_mean=temp_mean,
            temperature_std=temp_std,
            temperature_min=temp_min,
            temperature_max=temp_max,
            precipitation_total=precip_total,
            precipitation_std=precip_std,
            rainy_days=int(rainy_days) if not np.isnan(rainy_days) else 0,
            humidity_mean=humidity_mean,
            wind_speed_mean=wind_speed_mean,
            sunshine_hours=sunshine_hours
        )
    
    def _estimate_sunshine_hours(self, season: Season, season_data: pd.DataFrame) -> float:
        """Estimate sunshine hours based on season and weather conditions."""
        
        # Base sunshine hours by season (daily average)
        base_sunshine = {
            Season.WINTER: 8.5,
            Season.PRE_MONSOON: 7.5,
            Season.MONSOON: 4.5,
            Season.POST_MONSOON: 7.0
        }
        
        base_hours = base_sunshine.get(season, 6.0)
        
        # Adjust based on precipitation (more rain = less sunshine)
        if 'precipitation_mm' in season_data.columns:
            avg_precip = season_data['precipitation_mm'].mean()
            if avg_precip > 10:  # Significant rainfall
                reduction_factor = min(0.5, avg_precip / 50)  # Max 50% reduction
                base_hours *= (1 - reduction_factor)
        
        return base_hours
    
    def detect_weather_anomalies(self, 
                               climate_data: Dict[str, pd.DataFrame],
                               threshold: float = 2.0) -> List[WeatherAnomaly]:
        """Detect weather anomalies using statistical methods.
        
        Args:
            climate_data: Climate data by station
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of detected weather anomalies
        """
        try:
            anomalies = []
            
            # Aggregate data
            aggregated_data = self._aggregate_station_data(climate_data)
            
            if aggregated_data.empty:
                return anomalies
            
            # Variables to check for anomalies
            variables_to_check = {
                'temperature_avg_c': 'Temperature',
                'precipitation_mm': 'Precipitation',
                'relative_humidity_percent': 'Humidity',
                'wind_speed_kmh': 'Wind Speed'
            }
            
            for var_col, var_name in variables_to_check.items():
                if var_col not in aggregated_data.columns:
                    continue
                
                var_anomalies = self._detect_variable_anomalies(
                    aggregated_data, var_col, var_name, threshold
                )
                anomalies.extend(var_anomalies)
            
            # Sort anomalies by severity
            anomalies.sort(key=lambda x: abs(x.anomaly_score), reverse=True)
            
            logger.info(f"Detected {len(anomalies)} weather anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting weather anomalies: {str(e)}")
            return []
    
    def _detect_variable_anomalies(self, 
                                 data: pd.DataFrame,
                                 variable: str,
                                 variable_name: str,
                                 threshold: float) -> List[WeatherAnomaly]:
        """Detect anomalies for a specific variable."""
        
        anomalies = []
        
        if variable not in data.columns or data[variable].empty:
            return anomalies
        
        # Calculate rolling statistics for seasonal adjustment
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Calculate 30-day rolling mean and std
        data['rolling_mean'] = data[variable].rolling(window=30, center=True).mean()
        data['rolling_std'] = data[variable].rolling(window=30, center=True).std()
        
        # Calculate z-scores
        data['z_score'] = (data[variable] - data['rolling_mean']) / data['rolling_std']
        
        # Identify anomalies
        anomaly_mask = abs(data['z_score']) > threshold
        anomaly_data = data[anomaly_mask]
        
        for _, row in anomaly_data.iterrows():
            if pd.isna(row['z_score']):
                continue
            
            # Determine severity
            abs_z_score = abs(row['z_score'])
            if abs_z_score > 4:
                severity = 'extreme'
            elif abs_z_score > 3:
                severity = 'severe'
            elif abs_z_score > 2.5:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            # Create description
            direction = 'above' if row['z_score'] > 0 else 'below'
            description = f"{variable_name} {direction} normal: {row[variable]:.1f} (expected: {row['rolling_mean']:.1f})"
            
            anomaly = WeatherAnomaly(
                date=row['date'],
                variable=variable_name,
                observed_value=row[variable],
                expected_value=row['rolling_mean'],
                anomaly_score=row['z_score'],
                severity=severity,
                description=description
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def calculate_climate_indices(self, 
                                climate_data: Dict[str, pd.DataFrame]) -> Dict[str, ClimateIndex]:
        """Calculate various climate indices.
        
        Args:
            climate_data: Climate data by station
            
        Returns:
            Dictionary of calculated climate indices
        """
        try:
            indices = {}
            
            # Aggregate data
            aggregated_data = self._aggregate_station_data(climate_data)
            
            if aggregated_data.empty:
                return indices
            
            # Heat Index
            heat_index = self._calculate_heat_index(aggregated_data)
            if heat_index:
                indices['heat_index'] = heat_index
            
            # Drought Index (Standardized Precipitation Index)
            drought_index = self._calculate_drought_index(aggregated_data)
            if drought_index:
                indices['drought_index'] = drought_index
            
            # Comfort Index
            comfort_index = self._calculate_comfort_index(aggregated_data)
            if comfort_index:
                indices['comfort_index'] = comfort_index
            
            # Monsoon Index
            monsoon_index = self._calculate_monsoon_index(aggregated_data)
            if monsoon_index:
                indices['monsoon_index'] = monsoon_index
            
            # Additional indices
            cooling_degree_days = self._calculate_cooling_degree_days(aggregated_data)
            if cooling_degree_days:
                indices['cooling_degree_days'] = cooling_degree_days
            
            heating_degree_days = self._calculate_heating_degree_days(aggregated_data)
            if heating_degree_days:
                indices['heating_degree_days'] = heating_degree_days
            
            logger.info(f"Calculated {len(indices)} climate indices")
            return indices
            
        except Exception as e:
            logger.error(f"Error calculating climate indices: {str(e)}")
            return {}
    
    def _calculate_heat_index(self, data: pd.DataFrame) -> Optional[ClimateIndex]:
        """Calculate heat index (apparent temperature)."""
        
        if 'temperature_avg_c' not in data.columns or 'relative_humidity_percent' not in data.columns:
            return None
        
        # Convert to Fahrenheit for heat index calculation
        temp_f = data['temperature_avg_c'] * 9/5 + 32
        humidity = data['relative_humidity_percent']
        
        # Heat index formula (Rothfusz equation)
        heat_index_f = (
            -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
            - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2
            - 5.481717e-2 * humidity**2 + 1.22874e-3 * temp_f**2 * humidity
            + 8.5282e-4 * temp_f * humidity**2 - 1.99e-6 * temp_f**2 * humidity**2
        )
        
        # Convert back to Celsius
        heat_index_c = (heat_index_f - 32) * 5/9
        avg_heat_index = heat_index_c.mean()
        
        # Categorize
        thresholds = self.climate_indices['heat_index']['thresholds']
        category = 'normal'
        for cat, (min_val, max_val) in thresholds.items():
            if min_val <= avg_heat_index < max_val:
                category = cat
                break
        
        return ClimateIndex(
            name='Heat Index',
            value=avg_heat_index,
            category=category,
            description=f"Average apparent temperature: {avg_heat_index:.1f}°C ({category})",
            impact_sectors=['agriculture', 'construction', 'services']
        )
    
    def _calculate_drought_index(self, data: pd.DataFrame) -> Optional[ClimateIndex]:
        """Calculate Standardized Precipitation Index (SPI)."""
        
        if 'precipitation_mm' not in data.columns:
            return None
        
        # Calculate monthly precipitation totals
        data['date'] = pd.to_datetime(data['date'])
        monthly_precip = data.groupby(data['date'].dt.to_period('M'))['precipitation_mm'].sum()
        
        if len(monthly_precip) < 12:  # Need at least 1 year of data
            return None
        
        # Calculate SPI (simplified version)
        precip_mean = monthly_precip.mean()
        precip_std = monthly_precip.std()
        
        if precip_std == 0:
            spi = 0
        else:
            # Use recent 3-month average
            recent_precip = monthly_precip.tail(3).mean()
            spi = (recent_precip - precip_mean) / precip_std
        
        # Categorize
        thresholds = self.climate_indices['drought_index']['thresholds']
        category = 'normal'
        for cat, (min_val, max_val) in thresholds.items():
            if min_val <= spi < max_val:
                category = cat
                break
        
        return ClimateIndex(
            name='Drought Index (SPI)',
            value=spi,
            category=category,
            description=f"Standardized Precipitation Index: {spi:.2f} ({category})",
            impact_sectors=['agriculture', 'water_resources']
        )
    
    def _calculate_comfort_index(self, data: pd.DataFrame) -> Optional[ClimateIndex]:
        """Calculate Temperature-Humidity Index (THI)."""
        
        if 'temperature_avg_c' not in data.columns or 'relative_humidity_percent' not in data.columns:
            return None
        
        # Temperature-Humidity Index
        temp = data['temperature_avg_c']
        humidity = data['relative_humidity_percent']
        
        # THI formula
        thi = temp - (0.55 - 0.0055 * humidity) * (temp - 14.5)
        avg_thi = thi.mean()
        
        # Categorize
        thresholds = self.climate_indices['comfort_index']['thresholds']
        category = 'comfortable'
        for cat, (min_val, max_val) in thresholds.items():
            if min_val <= avg_thi < max_val:
                category = cat
                break
        
        return ClimateIndex(
            name='Comfort Index (THI)',
            value=avg_thi,
            category=category,
            description=f"Temperature-Humidity Index: {avg_thi:.1f}°C ({category})",
            impact_sectors=['services', 'construction', 'manufacturing']
        )
    
    def _calculate_monsoon_index(self, data: pd.DataFrame) -> Optional[ClimateIndex]:
        """Calculate monsoon strength index."""
        
        if 'precipitation_mm' not in data.columns:
            return None
        
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        
        # Monsoon months (June-September)
        monsoon_months = [6, 7, 8, 9]
        monsoon_data = data[data['month'].isin(monsoon_months)]
        
        if monsoon_data.empty:
            return None
        
        # Calculate monsoon precipitation
        monsoon_precip = monsoon_data['precipitation_mm'].sum()
        
        # Normal monsoon precipitation (approximate)
        normal_monsoon_precip = 1200  # mm
        
        # Monsoon index as ratio to normal
        monsoon_index = monsoon_precip / normal_monsoon_precip
        
        # Categorize
        thresholds = self.climate_indices['monsoon_index']['thresholds']
        category = 'normal'
        for cat, (min_val, max_val) in thresholds.items():
            if min_val <= monsoon_index < max_val:
                category = cat
                break
        
        return ClimateIndex(
            name='Monsoon Index',
            value=monsoon_index,
            category=category,
            description=f"Monsoon strength: {monsoon_index:.2f} ({category})",
            impact_sectors=['agriculture', 'water_resources', 'transport']
        )
    
    def _calculate_cooling_degree_days(self, data: pd.DataFrame) -> Optional[ClimateIndex]:
        """Calculate cooling degree days."""
        
        if 'temperature_avg_c' not in data.columns:
            return None
        
        base_temp = 18  # Base temperature for cooling (Celsius)
        cooling_dd = (data['temperature_avg_c'] - base_temp).clip(lower=0).sum()
        
        # Categorize based on typical values for Bangladesh
        if cooling_dd < 1000:
            category = 'low'
        elif cooling_dd < 2000:
            category = 'moderate'
        elif cooling_dd < 3000:
            category = 'high'
        else:
            category = 'very_high'
        
        return ClimateIndex(
            name='Cooling Degree Days',
            value=cooling_dd,
            category=category,
            description=f"Cooling degree days: {cooling_dd:.0f} ({category})",
            impact_sectors=['services', 'manufacturing', 'residential']
        )
    
    def _calculate_heating_degree_days(self, data: pd.DataFrame) -> Optional[ClimateIndex]:
        """Calculate heating degree days."""
        
        if 'temperature_avg_c' not in data.columns:
            return None
        
        base_temp = 18  # Base temperature for heating (Celsius)
        heating_dd = (base_temp - data['temperature_avg_c']).clip(lower=0).sum()
        
        # Categorize based on typical values for Bangladesh
        if heating_dd < 100:
            category = 'very_low'
        elif heating_dd < 300:
            category = 'low'
        elif heating_dd < 600:
            category = 'moderate'
        else:
            category = 'high'
        
        return ClimateIndex(
            name='Heating Degree Days',
            value=heating_dd,
            category=category,
            description=f"Heating degree days: {heating_dd:.0f} ({category})",
            impact_sectors=['residential', 'commercial']
        )
    
    def identify_weather_patterns(self, 
                                climate_data: Dict[str, pd.DataFrame],
                                climate_indices: Dict[str, ClimateIndex] = None) -> List[WeatherPattern]:
        """Identify dominant weather patterns.
        
        Args:
            climate_data: Climate data by station
            climate_indices: Pre-calculated climate indices
            
        Returns:
            List of identified weather patterns
        """
        try:
            patterns = []
            
            # Calculate indices if not provided
            if climate_indices is None:
                climate_indices = self.calculate_climate_indices(climate_data)
            
            # Aggregate data
            aggregated_data = self._aggregate_station_data(climate_data)
            
            if aggregated_data.empty:
                return patterns
            
            # Check for El Niño/La Niña patterns (simplified)
            if 'temperature_avg_c' in aggregated_data.columns:
                temp_anomaly = self._calculate_temperature_anomaly(aggregated_data)
                if temp_anomaly > 1.0:
                    patterns.append(WeatherPattern.EL_NINO)
                elif temp_anomaly < -1.0:
                    patterns.append(WeatherPattern.LA_NINA)
            
            # Check for heat wave patterns
            if 'heat_index' in climate_indices:
                heat_index = climate_indices['heat_index']
                if heat_index.category in ['danger', 'extreme_danger']:
                    patterns.append(WeatherPattern.HEAT_WAVE)
            
            # Check for drought patterns
            if 'drought_index' in climate_indices:
                drought_index = climate_indices['drought_index']
                if 'drought' in drought_index.category:
                    patterns.append(WeatherPattern.MONSOON_BREAK)
            
            # Check for monsoon patterns
            if 'monsoon_index' in climate_indices:
                monsoon_index = climate_indices['monsoon_index']
                if monsoon_index.category == 'weak':
                    patterns.append(WeatherPattern.MONSOON_BREAK)
            
            # Check for cold wave (winter months)
            if 'temperature_avg_c' in aggregated_data.columns:
                aggregated_data['date'] = pd.to_datetime(aggregated_data['date'])
                winter_data = aggregated_data[aggregated_data['date'].dt.month.isin([12, 1, 2])]
                if not winter_data.empty:
                    min_temp = winter_data['temperature_avg_c'].min()
                    if min_temp < 8:  # Cold wave threshold for Bangladesh
                        patterns.append(WeatherPattern.COLD_WAVE)
            
            # Default to normal if no specific patterns identified
            if not patterns:
                patterns.append(WeatherPattern.NORMAL)
            
            logger.info(f"Identified {len(patterns)} weather patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying weather patterns: {str(e)}")
            return [WeatherPattern.NORMAL]
    
    def _calculate_temperature_anomaly(self, data: pd.DataFrame) -> float:
        """Calculate temperature anomaly from long-term average."""
        
        if 'temperature_avg_c' not in data.columns:
            return 0.0
        
        # Calculate anomaly from seasonal average
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        
        monthly_means = data.groupby('month')['temperature_avg_c'].mean()
        
        # Calculate recent anomaly
        recent_data = data.tail(30)  # Last 30 days
        if recent_data.empty:
            return 0.0
        
        recent_temp = recent_data['temperature_avg_c'].mean()
        recent_month = recent_data['month'].iloc[-1]
        
        expected_temp = monthly_means.get(recent_month, recent_temp)
        
        return recent_temp - expected_temp
    
    def generate_weather_summary(self, 
                               seasonal_stats: Dict[Season, SeasonalStatistics],
                               climate_indices: Dict[str, ClimateIndex],
                               weather_patterns: List[WeatherPattern],
                               anomalies: List[WeatherAnomaly]) -> Dict[str, Any]:
        """Generate comprehensive weather summary.
        
        Args:
            seasonal_stats: Seasonal statistics
            climate_indices: Climate indices
            weather_patterns: Identified weather patterns
            anomalies: Detected anomalies
            
        Returns:
            Comprehensive weather summary
        """
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'seasonal_overview': {},
            'climate_indices_summary': {},
            'dominant_patterns': [pattern.value for pattern in weather_patterns],
            'anomalies_summary': {},
            'risk_assessment': {},
            'economic_implications': {},
            'recommendations': []
        }
        
        # Seasonal overview
        for season, stats in seasonal_stats.items():
            summary['seasonal_overview'][season.value] = {
                'temperature_avg': round(stats.temperature_mean, 1),
                'temperature_range': f"{stats.temperature_min:.1f} to {stats.temperature_max:.1f}°C",
                'precipitation_total': round(stats.precipitation_total, 1),
                'rainy_days': stats.rainy_days,
                'humidity_avg': round(stats.humidity_mean, 1),
                'wind_speed_avg': round(stats.wind_speed_mean, 1),
                'sunshine_hours': round(stats.sunshine_hours, 1)
            }
        
        # Climate indices summary
        for index_name, index_data in climate_indices.items():
            summary['climate_indices_summary'][index_name] = {
                'value': round(index_data.value, 2),
                'category': index_data.category,
                'description': index_data.description,
                'impact_sectors': index_data.impact_sectors
            }
        
        # Anomalies summary
        if anomalies:
            summary['anomalies_summary'] = {
                'total_count': len(anomalies),
                'by_severity': {},
                'by_variable': {},
                'most_significant': []
            }
            
            # Group by severity
            for anomaly in anomalies:
                severity = anomaly.severity
                if severity not in summary['anomalies_summary']['by_severity']:
                    summary['anomalies_summary']['by_severity'][severity] = 0
                summary['anomalies_summary']['by_severity'][severity] += 1
                
                # Group by variable
                variable = anomaly.variable
                if variable not in summary['anomalies_summary']['by_variable']:
                    summary['anomalies_summary']['by_variable'][variable] = 0
                summary['anomalies_summary']['by_variable'][variable] += 1
            
            # Most significant anomalies
            significant_anomalies = sorted(anomalies, key=lambda x: abs(x.anomaly_score), reverse=True)[:5]
            for anomaly in significant_anomalies:
                summary['anomalies_summary']['most_significant'].append({
                    'date': anomaly.date.strftime('%Y-%m-%d'),
                    'variable': anomaly.variable,
                    'severity': anomaly.severity,
                    'description': anomaly.description
                })
        
        # Risk assessment
        summary['risk_assessment'] = self._assess_weather_risks(
            seasonal_stats, climate_indices, weather_patterns, anomalies
        )
        
        # Economic implications
        summary['economic_implications'] = self._assess_economic_implications(
            seasonal_stats, climate_indices, weather_patterns
        )
        
        # Recommendations
        summary['recommendations'] = self._generate_weather_recommendations(
            seasonal_stats, climate_indices, weather_patterns, anomalies
        )
        
        return summary
    
    def _assess_weather_risks(self, 
                            seasonal_stats: Dict[Season, SeasonalStatistics],
                            climate_indices: Dict[str, ClimateIndex],
                            weather_patterns: List[WeatherPattern],
                            anomalies: List[WeatherAnomaly]) -> Dict[str, str]:
        """Assess weather-related risks."""
        
        risks = {
            'drought_risk': 'low',
            'flood_risk': 'low',
            'heat_stress_risk': 'low',
            'cyclone_risk': 'low',
            'agricultural_risk': 'low',
            'overall_risk': 'low'
        }
        
        # Assess drought risk
        if 'drought_index' in climate_indices:
            drought_index = climate_indices['drought_index']
            if 'drought' in drought_index.category:
                risks['drought_risk'] = 'high' if 'severe' in drought_index.category else 'moderate'
        
        # Assess heat stress risk
        if 'heat_index' in climate_indices:
            heat_index = climate_indices['heat_index']
            if heat_index.category in ['danger', 'extreme_danger']:
                risks['heat_stress_risk'] = 'high'
            elif heat_index.category == 'extreme_caution':
                risks['heat_stress_risk'] = 'moderate'
        
        # Assess flood risk based on precipitation patterns
        if Season.MONSOON in seasonal_stats:
            monsoon_stats = seasonal_stats[Season.MONSOON]
            if monsoon_stats.precipitation_total > 2000:  # High precipitation
                risks['flood_risk'] = 'high'
            elif monsoon_stats.precipitation_total > 1500:
                risks['flood_risk'] = 'moderate'
        
        # Assess cyclone risk (seasonal)
        if Season.POST_MONSOON in seasonal_stats:
            post_monsoon_stats = seasonal_stats[Season.POST_MONSOON]
            if post_monsoon_stats.wind_speed_mean > 25:  # High wind speeds
                risks['cyclone_risk'] = 'moderate'
        
        # Assess agricultural risk
        agricultural_risk_factors = [
            risks['drought_risk'],
            risks['flood_risk'],
            risks['heat_stress_risk']
        ]
        
        if 'high' in agricultural_risk_factors:
            risks['agricultural_risk'] = 'high'
        elif 'moderate' in agricultural_risk_factors:
            risks['agricultural_risk'] = 'moderate'
        
        # Overall risk assessment
        all_risks = [risk for risk in risks.values() if risk != 'low']
        if any(risk == 'high' for risk in all_risks):
            risks['overall_risk'] = 'high'
        elif any(risk == 'moderate' for risk in all_risks):
            risks['overall_risk'] = 'moderate'
        
        return risks
    
    def _assess_economic_implications(self, 
                                    seasonal_stats: Dict[Season, SeasonalStatistics],
                                    climate_indices: Dict[str, ClimateIndex],
                                    weather_patterns: List[WeatherPattern]) -> Dict[str, str]:
        """Assess economic implications of weather patterns."""
        
        implications = {
            'agriculture': 'neutral',
            'manufacturing': 'neutral',
            'services': 'neutral',
            'construction': 'neutral',
            'transport': 'neutral',
            'energy': 'neutral'
        }
        
        # Agriculture implications
        if WeatherPattern.MONSOON_BREAK in weather_patterns or WeatherPattern.EL_NINO in weather_patterns:
            implications['agriculture'] = 'negative'
        elif WeatherPattern.LA_NINA in weather_patterns:
            implications['agriculture'] = 'positive'
        
        # Manufacturing implications
        if WeatherPattern.HEAT_WAVE in weather_patterns:
            implications['manufacturing'] = 'negative'
            implications['energy'] = 'negative'  # Increased cooling demand
        
        # Construction implications
        if WeatherPattern.HEAT_WAVE in weather_patterns:
            implications['construction'] = 'negative'
        elif WeatherPattern.COLD_WAVE in weather_patterns:
            implications['construction'] = 'positive'  # Better working conditions
        
        # Services implications
        if 'comfort_index' in climate_indices:
            comfort_index = climate_indices['comfort_index']
            if comfort_index.category in ['hot', 'very_hot']:
                implications['services'] = 'negative'
        
        # Transport implications
        if Season.MONSOON in seasonal_stats:
            monsoon_stats = seasonal_stats[Season.MONSOON]
            if monsoon_stats.precipitation_total > 2000:
                implications['transport'] = 'negative'
        
        return implications
    
    def _generate_weather_recommendations(self, 
                                        seasonal_stats: Dict[Season, SeasonalStatistics],
                                        climate_indices: Dict[str, ClimateIndex],
                                        weather_patterns: List[WeatherPattern],
                                        anomalies: List[WeatherAnomaly]) -> List[str]:
        """Generate weather-based recommendations."""
        
        recommendations = []
        
        # Heat-related recommendations
        if WeatherPattern.HEAT_WAVE in weather_patterns:
            recommendations.extend([
                "Implement heat action plans for outdoor workers",
                "Increase cooling capacity in manufacturing facilities",
                "Adjust working hours to avoid peak heat periods"
            ])
        
        # Drought-related recommendations
        if WeatherPattern.MONSOON_BREAK in weather_patterns:
            recommendations.extend([
                "Implement water conservation measures",
                "Promote drought-resistant crop varieties",
                "Develop alternative water sources"
            ])
        
        # Monsoon-related recommendations
        if Season.MONSOON in seasonal_stats:
            monsoon_stats = seasonal_stats[Season.MONSOON]
            if monsoon_stats.precipitation_total > 2000:
                recommendations.extend([
                    "Strengthen flood preparedness measures",
                    "Improve drainage infrastructure",
                    "Develop flood-resistant agricultural practices"
                ])
        
        # Cold wave recommendations
        if WeatherPattern.COLD_WAVE in weather_patterns:
            recommendations.extend([
                "Protect vulnerable crops from cold damage",
                "Ensure adequate heating for livestock",
                "Monitor health impacts on elderly population"
            ])
        
        # General recommendations based on anomalies
        if anomalies:
            severe_anomalies = [a for a in anomalies if a.severity in ['severe', 'extreme']]
            if severe_anomalies:
                recommendations.append("Enhance weather monitoring and early warning systems")
        
        # Climate adaptation recommendations
        recommendations.extend([
            "Develop climate-resilient infrastructure",
            "Promote climate-smart agricultural practices",
            "Strengthen disaster risk reduction measures"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def export_weather_analysis(self, 
                              weather_summary: Dict[str, Any],
                              output_path: str,
                              format: str = 'json') -> bool:
        """Export weather analysis results.
        
        Args:
            weather_summary: Weather analysis summary
            output_path: Output file path
            format: Export format ('json', 'csv', 'xlsx')
            
        Returns:
            Success status
        """
        try:
            from pathlib import Path
            import json
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(weather_summary, f, indent=2, default=str)
            
            elif format == 'csv':
                # Convert to flat structure for CSV
                flat_data = self._flatten_weather_summary(weather_summary)
                df = pd.DataFrame([flat_data])
                df.to_csv(output_path, index=False)
            
            elif format == 'xlsx':
                # Create multiple sheets for different sections
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Seasonal overview
                    if 'seasonal_overview' in weather_summary:
                        seasonal_df = pd.DataFrame(weather_summary['seasonal_overview']).T
                        seasonal_df.to_excel(writer, sheet_name='Seasonal_Overview')
                    
                    # Climate indices
                    if 'climate_indices_summary' in weather_summary:
                        indices_df = pd.DataFrame(weather_summary['climate_indices_summary']).T
                        indices_df.to_excel(writer, sheet_name='Climate_Indices')
                    
                    # Anomalies
                    if 'anomalies_summary' in weather_summary and 'most_significant' in weather_summary['anomalies_summary']:
                        anomalies_df = pd.DataFrame(weather_summary['anomalies_summary']['most_significant'])
                        anomalies_df.to_excel(writer, sheet_name='Anomalies', index=False)
            
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Exported weather analysis to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting weather analysis: {str(e)}")
            return False
    
    def _flatten_weather_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested weather summary for CSV export."""
        
        flat_data = {
            'analysis_date': summary.get('analysis_date', ''),
            'dominant_patterns': ', '.join(summary.get('dominant_patterns', [])),
            'overall_risk': summary.get('risk_assessment', {}).get('overall_risk', 'unknown')
        }
        
        # Add seasonal data
        seasonal_overview = summary.get('seasonal_overview', {})
        for season, data in seasonal_overview.items():
            for key, value in data.items():
                flat_data[f"{season}_{key}"] = value
        
        # Add climate indices
        climate_indices = summary.get('climate_indices_summary', {})
        for index_name, data in climate_indices.items():
            flat_data[f"{index_name}_value"] = data.get('value', '')
            flat_data[f"{index_name}_category"] = data.get('category', '')
        
        # Add risk assessment
        risk_assessment = summary.get('risk_assessment', {})
        for risk_type, level in risk_assessment.items():
            flat_data[f"risk_{risk_type}"] = level
        
        return flat_data