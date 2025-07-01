"""Climate Impact Assessment for Bangladesh GDP Simulation.

This module provides comprehensive assessment of climate impacts on Bangladesh's
economic sectors, including agriculture, manufacturing, services, and infrastructure.
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

logger = logging.getLogger(__name__)


class ImpactSeverity(Enum):
    """Climate impact severity levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"


class AdaptationLevel(Enum):
    """Adaptation capacity levels."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    ADVANCED = "advanced"


@dataclass
class ClimateImpactResult:
    """Result of climate impact assessment."""
    sector: str
    subsector: str
    impact_type: str
    severity: ImpactSeverity
    impact_factor: float  # Multiplier for economic output (0.0 to 2.0)
    confidence: float  # Confidence level (0.0 to 1.0)
    description: str
    adaptation_potential: AdaptationLevel
    cost_estimate: float  # Cost in million BDT
    timeframe: str  # 'immediate', 'short_term', 'medium_term', 'long_term'


class ClimateImpactAssessment:
    """Comprehensive climate impact assessment for Bangladesh economy.
    
    This class evaluates the impacts of climate variables and extreme events
    on different economic sectors and provides quantitative impact factors
    for GDP calculations.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the climate impact assessment system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize impact parameters
        self.impact_parameters = self._initialize_impact_parameters()
        
        # Vulnerability matrices
        self.vulnerability_matrix = self._initialize_vulnerability_matrix()
        
        # Adaptation measures database
        self.adaptation_measures = self._initialize_adaptation_measures()
        
        logger.info("Climate impact assessment system initialized")
    
    def _initialize_impact_parameters(self) -> Dict[str, Dict]:
        """Initialize climate impact parameters for different sectors."""
        
        return {
            'agriculture': {
                'temperature': {
                    'optimal_range': (20, 30),  # Celsius
                    'stress_threshold': 35,
                    'damage_threshold': 40,
                    'sensitivity': 0.8  # High sensitivity
                },
                'precipitation': {
                    'optimal_range': (1200, 2500),  # mm/year
                    'drought_threshold': 800,
                    'flood_threshold': 3500,
                    'sensitivity': 0.9  # Very high sensitivity
                },
                'humidity': {
                    'optimal_range': (60, 80),  # %
                    'stress_threshold': 90,
                    'sensitivity': 0.4  # Moderate sensitivity
                },
                'wind_speed': {
                    'damage_threshold': 60,  # km/h
                    'severe_damage_threshold': 100,
                    'sensitivity': 0.6
                }
            },
            'manufacturing': {
                'temperature': {
                    'optimal_range': (18, 28),
                    'stress_threshold': 32,
                    'damage_threshold': 38,
                    'sensitivity': 0.5  # Moderate sensitivity
                },
                'precipitation': {
                    'flood_threshold': 200,  # mm/day
                    'sensitivity': 0.3  # Low sensitivity (indoor activity)
                },
                'humidity': {
                    'optimal_range': (40, 70),
                    'stress_threshold': 85,
                    'sensitivity': 0.3
                },
                'wind_speed': {
                    'damage_threshold': 80,
                    'severe_damage_threshold': 120,
                    'sensitivity': 0.4
                }
            },
            'services': {
                'temperature': {
                    'optimal_range': (20, 28),
                    'stress_threshold': 35,
                    'sensitivity': 0.3  # Low sensitivity (mostly indoor)
                },
                'precipitation': {
                    'disruption_threshold': 100,  # mm/day
                    'sensitivity': 0.4
                },
                'wind_speed': {
                    'disruption_threshold': 50,
                    'damage_threshold': 80,
                    'sensitivity': 0.3
                }
            },
            'construction': {
                'temperature': {
                    'optimal_range': (15, 30),
                    'stress_threshold': 35,
                    'danger_threshold': 40,
                    'sensitivity': 0.7  # High sensitivity (outdoor work)
                },
                'precipitation': {
                    'disruption_threshold': 20,  # mm/day
                    'halt_threshold': 50,
                    'sensitivity': 0.8
                },
                'wind_speed': {
                    'disruption_threshold': 30,
                    'halt_threshold': 60,
                    'sensitivity': 0.7
                }
            },
            'transport': {
                'precipitation': {
                    'disruption_threshold': 50,
                    'severe_disruption_threshold': 100,
                    'sensitivity': 0.6
                },
                'wind_speed': {
                    'disruption_threshold': 40,
                    'halt_threshold': 70,
                    'sensitivity': 0.5
                },
                'visibility': {
                    'disruption_threshold': 1000,  # meters
                    'sensitivity': 0.4
                }
            }
        }
    
    def _initialize_vulnerability_matrix(self) -> Dict[str, Dict]:
        """Initialize vulnerability matrix for different regions and sectors."""
        
        return {
            'coastal_regions': {
                'sea_level_rise': {
                    'agriculture': 0.8,  # High vulnerability
                    'manufacturing': 0.6,
                    'services': 0.4,
                    'infrastructure': 0.9
                },
                'cyclones': {
                    'agriculture': 0.9,
                    'manufacturing': 0.7,
                    'services': 0.5,
                    'infrastructure': 0.8
                },
                'storm_surge': {
                    'agriculture': 0.9,
                    'manufacturing': 0.8,
                    'services': 0.6,
                    'infrastructure': 0.9
                }
            },
            'river_delta': {
                'flooding': {
                    'agriculture': 0.8,
                    'manufacturing': 0.6,
                    'services': 0.4,
                    'infrastructure': 0.7
                },
                'riverbank_erosion': {
                    'agriculture': 0.7,
                    'manufacturing': 0.3,
                    'services': 0.2,
                    'infrastructure': 0.8
                }
            },
            'northern_regions': {
                'drought': {
                    'agriculture': 0.9,
                    'manufacturing': 0.2,
                    'services': 0.1,
                    'infrastructure': 0.3
                },
                'cold_waves': {
                    'agriculture': 0.6,
                    'manufacturing': 0.3,
                    'services': 0.2,
                    'infrastructure': 0.4
                }
            },
            'urban_areas': {
                'heat_waves': {
                    'agriculture': 0.3,
                    'manufacturing': 0.5,
                    'services': 0.4,
                    'infrastructure': 0.6
                },
                'urban_flooding': {
                    'agriculture': 0.1,
                    'manufacturing': 0.7,
                    'services': 0.8,
                    'infrastructure': 0.9
                }
            }
        }
    
    def _initialize_adaptation_measures(self) -> Dict[str, Dict]:
        """Initialize adaptation measures database."""
        
        return {
            'agriculture': {
                'drought_resistant_crops': {
                    'effectiveness': 0.6,
                    'cost_per_hectare': 5000,  # BDT
                    'implementation_time': 'medium_term'
                },
                'improved_irrigation': {
                    'effectiveness': 0.7,
                    'cost_per_hectare': 15000,
                    'implementation_time': 'short_term'
                },
                'flood_resistant_varieties': {
                    'effectiveness': 0.5,
                    'cost_per_hectare': 3000,
                    'implementation_time': 'medium_term'
                },
                'climate_smart_practices': {
                    'effectiveness': 0.4,
                    'cost_per_hectare': 2000,
                    'implementation_time': 'short_term'
                }
            },
            'manufacturing': {
                'climate_controlled_facilities': {
                    'effectiveness': 0.8,
                    'cost_per_facility': 5000000,  # BDT
                    'implementation_time': 'medium_term'
                },
                'backup_power_systems': {
                    'effectiveness': 0.6,
                    'cost_per_facility': 2000000,
                    'implementation_time': 'short_term'
                },
                'flood_protection': {
                    'effectiveness': 0.7,
                    'cost_per_facility': 3000000,
                    'implementation_time': 'medium_term'
                }
            },
            'infrastructure': {
                'elevated_roads': {
                    'effectiveness': 0.8,
                    'cost_per_km': 50000000,  # BDT
                    'implementation_time': 'long_term'
                },
                'improved_drainage': {
                    'effectiveness': 0.6,
                    'cost_per_km': 10000000,
                    'implementation_time': 'medium_term'
                },
                'cyclone_shelters': {
                    'effectiveness': 0.9,
                    'cost_per_shelter': 5000000,
                    'implementation_time': 'medium_term'
                }
            }
        }
    
    def assess_climate_impact(self, 
                            climate_data: Dict[str, pd.DataFrame],
                            extreme_events: List = None,
                            sector: str = None,
                            region: str = None) -> Dict[str, ClimateImpactResult]:
        """Assess climate impacts on economic sectors.
        
        Args:
            climate_data: Climate data by station
            extreme_events: List of extreme weather events
            sector: Specific sector to assess (optional)
            region: Specific region to assess (optional)
            
        Returns:
            Dictionary of climate impact results
        """
        try:
            impact_results = {}
            
            # Assess gradual climate impacts
            gradual_impacts = self._assess_gradual_impacts(climate_data, sector, region)
            impact_results.update(gradual_impacts)
            
            # Assess extreme event impacts
            if extreme_events:
                extreme_impacts = self._assess_extreme_event_impacts(extreme_events, sector, region)
                impact_results.update(extreme_impacts)
            
            # Assess compound impacts
            compound_impacts = self._assess_compound_impacts(impact_results)
            impact_results.update(compound_impacts)
            
            logger.info(f"Assessed {len(impact_results)} climate impacts")
            return impact_results
            
        except Exception as e:
            logger.error(f"Error assessing climate impacts: {str(e)}")
            return {}
    
    def _assess_gradual_impacts(self, 
                              climate_data: Dict[str, pd.DataFrame],
                              sector: str = None,
                              region: str = None) -> Dict[str, ClimateImpactResult]:
        """Assess gradual climate change impacts."""
        
        impacts = {}
        
        # Sectors to assess
        sectors_to_assess = [sector] if sector else list(self.impact_parameters.keys())
        
        for sector_name in sectors_to_assess:
            if sector_name not in self.impact_parameters:
                continue
            
            sector_params = self.impact_parameters[sector_name]
            
            # Aggregate climate data across stations
            aggregated_data = self._aggregate_climate_data(climate_data, region)
            
            if aggregated_data.empty:
                continue
            
            # Assess temperature impacts
            if 'temperature_avg_c' in aggregated_data.columns:
                temp_impact = self._assess_temperature_impact(
                    aggregated_data['temperature_avg_c'], sector_name, sector_params
                )
                if temp_impact:
                    impacts[f"{sector_name}_temperature"] = temp_impact
            
            # Assess precipitation impacts
            if 'precipitation_mm' in aggregated_data.columns:
                precip_impact = self._assess_precipitation_impact(
                    aggregated_data['precipitation_mm'], sector_name, sector_params
                )
                if precip_impact:
                    impacts[f"{sector_name}_precipitation"] = precip_impact
            
            # Assess humidity impacts
            if 'relative_humidity_percent' in aggregated_data.columns:
                humidity_impact = self._assess_humidity_impact(
                    aggregated_data['relative_humidity_percent'], sector_name, sector_params
                )
                if humidity_impact:
                    impacts[f"{sector_name}_humidity"] = humidity_impact
            
            # Assess wind impacts
            if 'wind_speed_kmh' in aggregated_data.columns:
                wind_impact = self._assess_wind_impact(
                    aggregated_data['wind_speed_kmh'], sector_name, sector_params
                )
                if wind_impact:
                    impacts[f"{sector_name}_wind"] = wind_impact
        
        return impacts
    
    def _aggregate_climate_data(self, 
                              climate_data: Dict[str, pd.DataFrame],
                              region: str = None) -> pd.DataFrame:
        """Aggregate climate data across stations."""
        
        all_data = []
        
        for station_id, df in climate_data.items():
            if df.empty:
                continue
            
            # Filter by region if specified
            if region:
                station_name = station_id.split('_')[0]
                # This is a simplified region filter - in practice, you'd have a proper mapping
                if region.lower() not in station_name.lower():
                    continue
            
            all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Aggregate by date
        if 'date' in combined_df.columns:
            aggregated = combined_df.groupby('date').agg({
                col: 'mean' for col in combined_df.columns 
                if col not in ['date', 'station_id', 'station_name', 'source']
            }).reset_index()
        else:
            aggregated = combined_df
        
        return aggregated
    
    def _assess_temperature_impact(self, 
                                 temperature_data: pd.Series,
                                 sector: str,
                                 sector_params: Dict) -> Optional[ClimateImpactResult]:
        """Assess temperature impact on a sector."""
        
        if 'temperature' not in sector_params:
            return None
        
        temp_params = sector_params['temperature']
        
        # Calculate statistics
        mean_temp = temperature_data.mean()
        max_temp = temperature_data.max()
        min_temp = temperature_data.min()
        
        # Determine impact severity and factor
        optimal_min, optimal_max = temp_params['optimal_range']
        stress_threshold = temp_params.get('stress_threshold', optimal_max + 5)
        damage_threshold = temp_params.get('damage_threshold', optimal_max + 10)
        sensitivity = temp_params['sensitivity']
        
        # Calculate impact factor
        if optimal_min <= mean_temp <= optimal_max:
            impact_factor = 1.0  # No impact
            severity = ImpactSeverity.MINIMAL
            description = f"Temperature within optimal range for {sector}"
        elif mean_temp > optimal_max:
            if mean_temp > damage_threshold:
                impact_factor = 1.0 - (0.3 * sensitivity)  # 30% reduction at high sensitivity
                severity = ImpactSeverity.SEVERE
                description = f"Severe heat stress affecting {sector} productivity"
            elif mean_temp > stress_threshold:
                impact_factor = 1.0 - (0.15 * sensitivity)  # 15% reduction
                severity = ImpactSeverity.MODERATE
                description = f"Heat stress reducing {sector} efficiency"
            else:
                impact_factor = 1.0 - (0.05 * sensitivity)  # 5% reduction
                severity = ImpactSeverity.LOW
                description = f"Mild heat stress on {sector}"
        else:  # Below optimal
            temp_deficit = optimal_min - mean_temp
            impact_factor = 1.0 - (temp_deficit * 0.02 * sensitivity)
            severity = ImpactSeverity.LOW if temp_deficit < 5 else ImpactSeverity.MODERATE
            description = f"Below optimal temperature affecting {sector}"
        
        # Ensure impact factor is within reasonable bounds
        impact_factor = max(0.3, min(1.2, impact_factor))
        
        # Determine adaptation potential
        adaptation_potential = self._determine_adaptation_potential(sector, 'temperature')
        
        # Estimate cost
        cost_estimate = self._estimate_impact_cost(sector, severity, 'temperature')
        
        return ClimateImpactResult(
            sector=sector,
            subsector='all',
            impact_type='temperature',
            severity=severity,
            impact_factor=impact_factor,
            confidence=0.8,
            description=description,
            adaptation_potential=adaptation_potential,
            cost_estimate=cost_estimate,
            timeframe='immediate'
        )
    
    def _assess_precipitation_impact(self, 
                                   precipitation_data: pd.Series,
                                   sector: str,
                                   sector_params: Dict) -> Optional[ClimateImpactResult]:
        """Assess precipitation impact on a sector."""
        
        if 'precipitation' not in sector_params:
            return None
        
        precip_params = sector_params['precipitation']
        
        # Calculate statistics
        total_precip = precipitation_data.sum()
        max_daily_precip = precipitation_data.max()
        heavy_rain_days = (precipitation_data > 50).sum()
        
        # Determine impact
        sensitivity = precip_params['sensitivity']
        
        # Check for drought conditions
        if 'drought_threshold' in precip_params:
            drought_threshold = precip_params['drought_threshold']
            if total_precip < drought_threshold:
                deficit_ratio = (drought_threshold - total_precip) / drought_threshold
                impact_factor = 1.0 - (deficit_ratio * sensitivity)
                severity = ImpactSeverity.HIGH if deficit_ratio > 0.3 else ImpactSeverity.MODERATE
                description = f"Drought conditions affecting {sector}"
                timeframe = 'medium_term'
            else:
                impact_factor = 1.0
                severity = ImpactSeverity.MINIMAL
                description = f"Adequate precipitation for {sector}"
                timeframe = 'immediate'
        
        # Check for flood conditions
        elif 'flood_threshold' in precip_params:
            flood_threshold = precip_params['flood_threshold']
            if max_daily_precip > flood_threshold:
                excess_ratio = (max_daily_precip - flood_threshold) / flood_threshold
                impact_factor = 1.0 - (min(excess_ratio, 1.0) * sensitivity)
                severity = ImpactSeverity.HIGH if excess_ratio > 0.5 else ImpactSeverity.MODERATE
                description = f"Flooding affecting {sector} operations"
                timeframe = 'immediate'
            else:
                impact_factor = 1.0
                severity = ImpactSeverity.MINIMAL
                description = f"Normal precipitation levels for {sector}"
                timeframe = 'immediate'
        
        # Check for disruption threshold
        elif 'disruption_threshold' in precip_params:
            disruption_threshold = precip_params['disruption_threshold']
            disruption_days = (precipitation_data > disruption_threshold).sum()
            
            if disruption_days > 0:
                disruption_ratio = disruption_days / len(precipitation_data)
                impact_factor = 1.0 - (disruption_ratio * sensitivity)
                severity = ImpactSeverity.MODERATE if disruption_ratio > 0.1 else ImpactSeverity.LOW
                description = f"Heavy rainfall disrupting {sector} activities"
                timeframe = 'immediate'
            else:
                impact_factor = 1.0
                severity = ImpactSeverity.MINIMAL
                description = f"Normal precipitation for {sector}"
                timeframe = 'immediate'
        else:
            return None
        
        # Ensure impact factor is within bounds
        impact_factor = max(0.2, min(1.1, impact_factor))
        
        adaptation_potential = self._determine_adaptation_potential(sector, 'precipitation')
        cost_estimate = self._estimate_impact_cost(sector, severity, 'precipitation')
        
        return ClimateImpactResult(
            sector=sector,
            subsector='all',
            impact_type='precipitation',
            severity=severity,
            impact_factor=impact_factor,
            confidence=0.7,
            description=description,
            adaptation_potential=adaptation_potential,
            cost_estimate=cost_estimate,
            timeframe=timeframe
        )
    
    def _assess_humidity_impact(self, 
                              humidity_data: pd.Series,
                              sector: str,
                              sector_params: Dict) -> Optional[ClimateImpactResult]:
        """Assess humidity impact on a sector."""
        
        if 'humidity' not in sector_params:
            return None
        
        humidity_params = sector_params['humidity']
        
        mean_humidity = humidity_data.mean()
        optimal_min, optimal_max = humidity_params['optimal_range']
        stress_threshold = humidity_params.get('stress_threshold', optimal_max + 10)
        sensitivity = humidity_params['sensitivity']
        
        if optimal_min <= mean_humidity <= optimal_max:
            impact_factor = 1.0
            severity = ImpactSeverity.MINIMAL
            description = f"Optimal humidity levels for {sector}"
        elif mean_humidity > stress_threshold:
            excess = (mean_humidity - stress_threshold) / 20  # Normalize
            impact_factor = 1.0 - (excess * sensitivity * 0.1)
            severity = ImpactSeverity.MODERATE if excess > 0.5 else ImpactSeverity.LOW
            description = f"High humidity affecting {sector} comfort and efficiency"
        else:
            impact_factor = 1.0 - (0.05 * sensitivity)  # Minor impact for low humidity
            severity = ImpactSeverity.LOW
            description = f"Low humidity with minor impact on {sector}"
        
        impact_factor = max(0.8, min(1.1, impact_factor))
        
        adaptation_potential = self._determine_adaptation_potential(sector, 'humidity')
        cost_estimate = self._estimate_impact_cost(sector, severity, 'humidity')
        
        return ClimateImpactResult(
            sector=sector,
            subsector='all',
            impact_type='humidity',
            severity=severity,
            impact_factor=impact_factor,
            confidence=0.6,
            description=description,
            adaptation_potential=adaptation_potential,
            cost_estimate=cost_estimate,
            timeframe='immediate'
        )
    
    def _assess_wind_impact(self, 
                          wind_data: pd.Series,
                          sector: str,
                          sector_params: Dict) -> Optional[ClimateImpactResult]:
        """Assess wind impact on a sector."""
        
        if 'wind_speed' not in sector_params:
            return None
        
        wind_params = sector_params['wind_speed']
        
        max_wind = wind_data.max()
        mean_wind = wind_data.mean()
        
        damage_threshold = wind_params.get('damage_threshold', 60)
        severe_damage_threshold = wind_params.get('severe_damage_threshold', 100)
        disruption_threshold = wind_params.get('disruption_threshold', 40)
        sensitivity = wind_params['sensitivity']
        
        if max_wind > severe_damage_threshold:
            impact_factor = 1.0 - (0.4 * sensitivity)
            severity = ImpactSeverity.SEVERE
            description = f"Severe wind damage to {sector} infrastructure"
            timeframe = 'immediate'
        elif max_wind > damage_threshold:
            impact_factor = 1.0 - (0.2 * sensitivity)
            severity = ImpactSeverity.HIGH
            description = f"Wind damage affecting {sector} operations"
            timeframe = 'immediate'
        elif max_wind > disruption_threshold:
            impact_factor = 1.0 - (0.1 * sensitivity)
            severity = ImpactSeverity.MODERATE
            description = f"Strong winds disrupting {sector} activities"
            timeframe = 'immediate'
        else:
            impact_factor = 1.0
            severity = ImpactSeverity.MINIMAL
            description = f"Normal wind conditions for {sector}"
            timeframe = 'immediate'
        
        impact_factor = max(0.4, min(1.0, impact_factor))
        
        adaptation_potential = self._determine_adaptation_potential(sector, 'wind')
        cost_estimate = self._estimate_impact_cost(sector, severity, 'wind')
        
        return ClimateImpactResult(
            sector=sector,
            subsector='all',
            impact_type='wind',
            severity=severity,
            impact_factor=impact_factor,
            confidence=0.7,
            description=description,
            adaptation_potential=adaptation_potential,
            cost_estimate=cost_estimate,
            timeframe=timeframe
        )
    
    def _assess_extreme_event_impacts(self, 
                                    extreme_events: List,
                                    sector: str = None,
                                    region: str = None) -> Dict[str, ClimateImpactResult]:
        """Assess impacts of extreme weather events."""
        
        impacts = {}
        
        for event in extreme_events:
            event_impacts = self._assess_single_extreme_event(event, sector, region)
            impacts.update(event_impacts)
        
        return impacts
    
    def _assess_single_extreme_event(self, 
                                   event,
                                   sector: str = None,
                                   region: str = None) -> Dict[str, ClimateImpactResult]:
        """Assess impact of a single extreme event."""
        
        impacts = {}
        
        # Get event characteristics
        event_type = event.event_type.value
        intensity = event.intensity
        affected_districts = event.affected_districts
        duration_days = (event.end_date - event.start_date).days
        
        # Determine sectors to assess
        sectors_to_assess = [sector] if sector else ['agriculture', 'manufacturing', 'services', 'construction']
        
        for sector_name in sectors_to_assess:
            # Get vulnerability for this event type and sector
            vulnerability = self._get_vulnerability(event_type, sector_name, affected_districts)
            
            if vulnerability == 0:
                continue
            
            # Calculate impact factor based on event characteristics
            base_impact = self._calculate_base_event_impact(event_type, intensity)
            duration_factor = min(1.0 + (duration_days - 1) * 0.1, 2.0)  # Duration amplifies impact
            
            impact_factor = 1.0 - (base_impact * vulnerability * duration_factor)
            impact_factor = max(0.1, min(1.0, impact_factor))
            
            # Determine severity
            if impact_factor < 0.5:
                severity = ImpactSeverity.SEVERE
            elif impact_factor < 0.7:
                severity = ImpactSeverity.HIGH
            elif impact_factor < 0.9:
                severity = ImpactSeverity.MODERATE
            else:
                severity = ImpactSeverity.LOW
            
            # Create impact result
            impact_key = f"{sector_name}_{event_type}_{event.event_id}"
            
            impacts[impact_key] = ClimateImpactResult(
                sector=sector_name,
                subsector='all',
                impact_type=f"extreme_event_{event_type}",
                severity=severity,
                impact_factor=impact_factor,
                confidence=0.8,
                description=f"{intensity.title()} {event_type} affecting {sector_name} in {', '.join(affected_districts)}",
                adaptation_potential=self._determine_adaptation_potential(sector_name, event_type),
                cost_estimate=event.economic_impact / len(sectors_to_assess),  # Distribute cost across sectors
                timeframe='immediate'
            )
        
        return impacts
    
    def _get_vulnerability(self, 
                         event_type: str,
                         sector: str,
                         affected_districts: List[str]) -> float:
        """Get vulnerability score for event type and sector."""
        
        # Determine region type based on affected districts
        coastal_districts = ['chittagong', 'coxs_bazar', 'barisal', 'khulna']
        northern_districts = ['rangpur', 'rajshahi']
        urban_districts = ['dhaka']
        
        region_type = 'river_delta'  # Default
        
        if any(district in coastal_districts for district in affected_districts):
            region_type = 'coastal_regions'
        elif any(district in northern_districts for district in affected_districts):
            region_type = 'northern_regions'
        elif any(district in urban_districts for district in affected_districts):
            region_type = 'urban_areas'
        
        # Get vulnerability from matrix
        if region_type in self.vulnerability_matrix:
            region_vulnerabilities = self.vulnerability_matrix[region_type]
            
            # Map event types to vulnerability categories
            event_mapping = {
                'cyclone': 'cyclones',
                'flood': 'flooding',
                'drought': 'drought',
                'heat_wave': 'heat_waves',
                'storm_surge': 'storm_surge'
            }
            
            vulnerability_category = event_mapping.get(event_type, event_type)
            
            if vulnerability_category in region_vulnerabilities:
                return region_vulnerabilities[vulnerability_category].get(sector, 0.3)
        
        return 0.3  # Default moderate vulnerability
    
    def _calculate_base_event_impact(self, event_type: str, intensity: str) -> float:
        """Calculate base impact factor for an extreme event."""
        
        # Base impact by event type
        base_impacts = {
            'cyclone': {'low': 0.1, 'moderate': 0.3, 'high': 0.5, 'severe': 0.7, 'extreme': 0.9},
            'flood': {'low': 0.05, 'moderate': 0.2, 'high': 0.4, 'severe': 0.6, 'extreme': 0.8},
            'drought': {'low': 0.1, 'moderate': 0.25, 'high': 0.4, 'severe': 0.6, 'extreme': 0.8},
            'heat_wave': {'low': 0.05, 'moderate': 0.15, 'high': 0.3, 'severe': 0.5, 'extreme': 0.7},
            'heavy_rainfall': {'low': 0.05, 'moderate': 0.15, 'high': 0.3, 'severe': 0.5, 'extreme': 0.7}
        }
        
        return base_impacts.get(event_type, {}).get(intensity, 0.2)
    
    def _assess_compound_impacts(self, 
                               existing_impacts: Dict[str, ClimateImpactResult]) -> Dict[str, ClimateImpactResult]:
        """Assess compound climate impacts (interactions between multiple stressors)."""
        
        compound_impacts = {}
        
        # Group impacts by sector
        sector_impacts = {}
        for impact_key, impact in existing_impacts.items():
            sector = impact.sector
            if sector not in sector_impacts:
                sector_impacts[sector] = []
            sector_impacts[sector].append(impact)
        
        # Assess compound effects for each sector
        for sector, impacts in sector_impacts.items():
            if len(impacts) > 1:
                compound_impact = self._calculate_compound_effect(sector, impacts)
                if compound_impact:
                    compound_impacts[f"{sector}_compound"] = compound_impact
        
        return compound_impacts
    
    def _calculate_compound_effect(self, 
                                 sector: str,
                                 impacts: List[ClimateImpactResult]) -> Optional[ClimateImpactResult]:
        """Calculate compound effect of multiple climate impacts."""
        
        if len(impacts) < 2:
            return None
        
        # Calculate combined impact factor (not simply multiplicative)
        impact_factors = [impact.impact_factor for impact in impacts]
        
        # Use a compound formula that accounts for interaction effects
        combined_factor = 1.0
        for factor in impact_factors:
            combined_factor *= factor
        
        # Add interaction penalty (compound effects are often worse than individual effects)
        interaction_penalty = 0.05 * (len(impacts) - 1)
        combined_factor -= interaction_penalty
        
        combined_factor = max(0.1, min(1.0, combined_factor))
        
        # Determine overall severity
        severities = [impact.severity for impact in impacts]
        severity_values = {
            ImpactSeverity.MINIMAL: 0,
            ImpactSeverity.LOW: 1,
            ImpactSeverity.MODERATE: 2,
            ImpactSeverity.HIGH: 3,
            ImpactSeverity.SEVERE: 4,
            ImpactSeverity.EXTREME: 5
        }
        
        max_severity_value = max(severity_values[s] for s in severities)
        avg_severity_value = sum(severity_values[s] for s in severities) / len(severities)
        
        # Compound severity is between max and average, leaning toward max
        compound_severity_value = int(0.7 * max_severity_value + 0.3 * avg_severity_value)
        
        severity_map = {v: k for k, v in severity_values.items()}
        compound_severity = severity_map[compound_severity_value]
        
        # Combine descriptions
        impact_types = [impact.impact_type for impact in impacts]
        description = f"Compound climate impacts on {sector}: {', '.join(impact_types)}"
        
        # Sum cost estimates
        total_cost = sum(impact.cost_estimate for impact in impacts)
        
        return ClimateImpactResult(
            sector=sector,
            subsector='all',
            impact_type='compound',
            severity=compound_severity,
            impact_factor=combined_factor,
            confidence=0.6,  # Lower confidence for compound effects
            description=description,
            adaptation_potential=AdaptationLevel.MODERATE,
            cost_estimate=total_cost,
            timeframe='medium_term'
        )
    
    def _determine_adaptation_potential(self, sector: str, impact_type: str) -> AdaptationLevel:
        """Determine adaptation potential for a sector and impact type."""
        
        # Adaptation potential matrix
        adaptation_matrix = {
            'agriculture': {
                'temperature': AdaptationLevel.MODERATE,
                'precipitation': AdaptationLevel.HIGH,
                'drought': AdaptationLevel.HIGH,
                'flood': AdaptationLevel.MODERATE
            },
            'manufacturing': {
                'temperature': AdaptationLevel.HIGH,
                'precipitation': AdaptationLevel.MODERATE,
                'cyclone': AdaptationLevel.MODERATE
            },
            'services': {
                'temperature': AdaptationLevel.HIGH,
                'precipitation': AdaptationLevel.MODERATE,
                'heat_wave': AdaptationLevel.HIGH
            },
            'construction': {
                'temperature': AdaptationLevel.LOW,
                'precipitation': AdaptationLevel.LOW,
                'wind': AdaptationLevel.MODERATE
            }
        }
        
        return adaptation_matrix.get(sector, {}).get(impact_type, AdaptationLevel.MODERATE)
    
    def _estimate_impact_cost(self, sector: str, severity: ImpactSeverity, impact_type: str) -> float:
        """Estimate economic cost of climate impact."""
        
        # Base cost estimates by sector (million BDT)
        base_costs = {
            'agriculture': 100,
            'manufacturing': 200,
            'services': 150,
            'construction': 80,
            'transport': 120
        }
        
        # Severity multipliers
        severity_multipliers = {
            ImpactSeverity.MINIMAL: 0.1,
            ImpactSeverity.LOW: 0.3,
            ImpactSeverity.MODERATE: 1.0,
            ImpactSeverity.HIGH: 2.5,
            ImpactSeverity.SEVERE: 5.0,
            ImpactSeverity.EXTREME: 10.0
        }
        
        base_cost = base_costs.get(sector, 100)
        multiplier = severity_multipliers.get(severity, 1.0)
        
        return base_cost * multiplier
    
    def generate_impact_summary(self, 
                              impact_results: Dict[str, ClimateImpactResult]) -> Dict[str, Any]:
        """Generate summary of climate impact assessment."""
        
        if not impact_results:
            return {}
        
        summary = {
            'total_impacts': len(impact_results),
            'by_sector': {},
            'by_severity': {},
            'by_impact_type': {},
            'overall_impact_factor': 1.0,
            'total_cost_estimate': 0.0,
            'adaptation_recommendations': [],
            'priority_actions': []
        }
        
        # Aggregate by sector
        for impact in impact_results.values():
            sector = impact.sector
            if sector not in summary['by_sector']:
                summary['by_sector'][sector] = {
                    'count': 0,
                    'avg_impact_factor': 0.0,
                    'total_cost': 0.0,
                    'severities': []
                }
            
            summary['by_sector'][sector]['count'] += 1
            summary['by_sector'][sector]['avg_impact_factor'] += impact.impact_factor
            summary['by_sector'][sector]['total_cost'] += impact.cost_estimate
            summary['by_sector'][sector]['severities'].append(impact.severity.value)
        
        # Calculate averages
        for sector_data in summary['by_sector'].values():
            if sector_data['count'] > 0:
                sector_data['avg_impact_factor'] /= sector_data['count']
        
        # Aggregate by severity
        for impact in impact_results.values():
            severity = impact.severity.value
            if severity not in summary['by_severity']:
                summary['by_severity'][severity] = 0
            summary['by_severity'][severity] += 1
        
        # Aggregate by impact type
        for impact in impact_results.values():
            impact_type = impact.impact_type
            if impact_type not in summary['by_impact_type']:
                summary['by_impact_type'][impact_type] = 0
            summary['by_impact_type'][impact_type] += 1
        
        # Calculate overall impact factor (weighted average)
        total_weight = 0
        weighted_sum = 0
        
        for impact in impact_results.values():
            weight = 1.0  # Could be adjusted based on sector importance
            weighted_sum += impact.impact_factor * weight
            total_weight += weight
        
        if total_weight > 0:
            summary['overall_impact_factor'] = weighted_sum / total_weight
        
        # Total cost estimate
        summary['total_cost_estimate'] = sum(impact.cost_estimate for impact in impact_results.values())
        
        # Generate recommendations
        summary['adaptation_recommendations'] = self._generate_adaptation_recommendations(impact_results)
        summary['priority_actions'] = self._generate_priority_actions(impact_results)
        
        return summary
    
    def _generate_adaptation_recommendations(self, 
                                           impact_results: Dict[str, ClimateImpactResult]) -> List[str]:
        """Generate adaptation recommendations based on impact assessment."""
        
        recommendations = []
        
        # Analyze impacts by sector and type
        sector_impacts = {}
        for impact in impact_results.values():
            sector = impact.sector
            if sector not in sector_impacts:
                sector_impacts[sector] = []
            sector_impacts[sector].append(impact)
        
        for sector, impacts in sector_impacts.items():
            # Find most severe impacts
            severe_impacts = [i for i in impacts if i.severity in [ImpactSeverity.HIGH, ImpactSeverity.SEVERE]]
            
            if severe_impacts:
                # Get adaptation measures for this sector
                if sector in self.adaptation_measures:
                    measures = self.adaptation_measures[sector]
                    for measure_name, measure_data in measures.items():
                        effectiveness = measure_data['effectiveness']
                        if effectiveness > 0.5:  # Only recommend effective measures
                            recommendations.append(
                                f"Implement {measure_name.replace('_', ' ')} for {sector} "
                                f"(effectiveness: {effectiveness:.0%})"
                            )
        
        # Add general recommendations
        if any(i.impact_type == 'temperature' for i in impact_results.values()):
            recommendations.append("Develop heat action plans for vulnerable sectors")
        
        if any(i.impact_type == 'precipitation' for i in impact_results.values()):
            recommendations.append("Improve early warning systems for extreme weather")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_priority_actions(self, 
                                 impact_results: Dict[str, ClimateImpactResult]) -> List[str]:
        """Generate priority actions based on impact severity and cost."""
        
        actions = []
        
        # Sort impacts by severity and cost
        sorted_impacts = sorted(
            impact_results.values(),
            key=lambda x: (x.severity.value, x.cost_estimate),
            reverse=True
        )
        
        # Generate actions for top impacts
        for impact in sorted_impacts[:5]:
            if impact.severity in [ImpactSeverity.HIGH, ImpactSeverity.SEVERE]:
                actions.append(
                    f"Address {impact.impact_type} impacts on {impact.sector} "
                    f"(estimated cost: {impact.cost_estimate:.0f} million BDT)"
                )
        
        # Add cross-cutting actions
        if len(impact_results) > 5:
            actions.append("Develop integrated climate adaptation strategy")
        
        actions.append("Strengthen climate monitoring and early warning systems")
        actions.append("Build climate resilience into infrastructure planning")
        
        return actions
    
    def export_impact_assessment(self, 
                               impact_results: Dict[str, ClimateImpactResult],
                               output_path: str,
                               format: str = 'csv') -> bool:
        """Export climate impact assessment results."""
        
        try:
            from pathlib import Path
            
            # Convert results to DataFrame
            data = []
            for impact_key, impact in impact_results.items():
                data.append({
                    'impact_id': impact_key,
                    'sector': impact.sector,
                    'subsector': impact.subsector,
                    'impact_type': impact.impact_type,
                    'severity': impact.severity.value,
                    'impact_factor': impact.impact_factor,
                    'confidence': impact.confidence,
                    'description': impact.description,
                    'adaptation_potential': impact.adaptation_potential.value,
                    'cost_estimate_million_bdt': impact.cost_estimate,
                    'timeframe': impact.timeframe
                })
            
            df = pd.DataFrame(data)
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'xlsx':
                df.to_excel(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', indent=2)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Exported climate impact assessment to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting impact assessment: {str(e)}")
            return False