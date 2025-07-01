"""Agriculture Sector Model for Bangladesh GDP Simulation.

This module models Bangladesh's agriculture sector including crops, livestock,
forestry, and fishing. It incorporates climate sensitivity, seasonal patterns,
and the impact of monsoons, floods, and droughts on agricultural production.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AgricultureSector:
    """Model for Bangladesh's agriculture, forestry and fishing sector.
    
    This class simulates agricultural production considering:
    - Seasonal crop cycles (Aman, Aus, Boro rice)
    - Climate impacts (monsoon, floods, droughts)
    - Livestock and poultry production
    - Fisheries (inland and marine)
    - Forestry production
    """
    
    def __init__(self):
        """Initialize agriculture sector model."""
        
        # Subsector shares in agriculture GDP (2024 estimates)
        self.subsector_shares = {
            'crops': 0.58,          # Rice, wheat, jute, tea, etc.
            'livestock': 0.17,      # Cattle, buffalo, goats, poultry
            'forestry': 0.09,       # Timber, bamboo, fuelwood
            'fishing': 0.16         # Inland and marine fisheries
        }
        
        # Crop production parameters
        self.crop_parameters = {
            'rice': {
                'share_of_crops': 0.72,     # Rice dominance
                'seasons': {
                    'aman': {'share': 0.52, 'harvest_month': 11},      # Nov-Dec
                    'boro': {'share': 0.38, 'harvest_month': 4},       # Apr-May
                    'aus': {'share': 0.10, 'harvest_month': 7}         # Jul-Aug
                },
                'yield_per_hectare': 4.2,   # Tons per hectare
                'climate_sensitivity': 0.8   # High sensitivity
            },
            'wheat': {
                'share_of_crops': 0.08,
                'harvest_month': 3,          # March
                'yield_per_hectare': 3.1,
                'climate_sensitivity': 0.6
            },
            'jute': {
                'share_of_crops': 0.04,
                'harvest_month': 8,          # August
                'yield_per_hectare': 2.5,
                'climate_sensitivity': 0.7
            },
            'tea': {
                'share_of_crops': 0.03,
                'harvest_continuous': True,   # Year-round harvesting
                'yield_per_hectare': 1.8,
                'climate_sensitivity': 0.9
            },
            'vegetables': {
                'share_of_crops': 0.08,
                'harvest_continuous': True,
                'yield_per_hectare': 15.2,
                'climate_sensitivity': 0.5
            },
            'other_crops': {
                'share_of_crops': 0.05,
                'climate_sensitivity': 0.6
            }
        }
        
        # Livestock parameters
        self.livestock_parameters = {
            'cattle': {'share': 0.35, 'growth_rate': 0.02},
            'buffalo': {'share': 0.08, 'growth_rate': 0.01},
            'goats': {'share': 0.15, 'growth_rate': 0.04},
            'sheep': {'share': 0.02, 'growth_rate': 0.03},
            'poultry': {'share': 0.35, 'growth_rate': 0.08},
            'ducks': {'share': 0.05, 'growth_rate': 0.05}
        }
        
        # Fisheries parameters
        self.fisheries_parameters = {
            'inland_capture': {'share': 0.28, 'sustainability_factor': 0.95},
            'inland_culture': {'share': 0.56, 'growth_potential': 0.06},
            'marine_capture': {'share': 0.16, 'sustainability_factor': 0.90}
        }
        
        # Climate impact factors
        self.climate_factors = {
            'normal_monsoon': 1.0,
            'good_monsoon': 1.08,
            'poor_monsoon': 0.85,
            'flood_impact': 0.75,
            'drought_impact': 0.70,
            'cyclone_impact': 0.65
        }
        
        logger.info("Agriculture sector model initialized")
    
    def calculate_production(self, 
                           base_year_value: float,
                           quarter: int,
                           year: int,
                           climate_conditions: Dict,
                           policy_factors: Dict = None) -> Dict:
        """Calculate quarterly agricultural production.
        
        Args:
            base_year_value: Base year agriculture GDP (billion BDT)
            quarter: Quarter (1-4)
            year: Year
            climate_conditions: Climate impact factors
            policy_factors: Agricultural policy impacts
            
        Returns:
            Dictionary with production estimates by subsector
        """
        if policy_factors is None:
            policy_factors = {}
        
        # Calculate subsector production
        subsector_production = {}
        
        # Crops production
        crops_production = self._calculate_crop_production(
            base_year_value * self.subsector_shares['crops'],
            quarter, year, climate_conditions, policy_factors
        )
        subsector_production['crops'] = crops_production
        
        # Livestock production
        livestock_production = self._calculate_livestock_production(
            base_year_value * self.subsector_shares['livestock'],
            quarter, year, climate_conditions, policy_factors
        )
        subsector_production['livestock'] = livestock_production
        
        # Forestry production
        forestry_production = self._calculate_forestry_production(
            base_year_value * self.subsector_shares['forestry'],
            quarter, year, climate_conditions, policy_factors
        )
        subsector_production['forestry'] = forestry_production
        
        # Fishing production
        fishing_production = self._calculate_fishing_production(
            base_year_value * self.subsector_shares['fishing'],
            quarter, year, climate_conditions, policy_factors
        )
        subsector_production['fishing'] = fishing_production
        
        # Total agriculture production
        total_production = sum(subsector_production.values())
        
        return {
            'total_agriculture_gdp': total_production,
            'subsector_breakdown': subsector_production,
            'growth_factors': self._calculate_growth_factors(climate_conditions, policy_factors),
            'seasonal_adjustment': self._get_seasonal_adjustment(quarter),
            'climate_impact': self._assess_climate_impact(climate_conditions)
        }
    
    def _calculate_crop_production(self, 
                                 base_value: float,
                                 quarter: int,
                                 year: int,
                                 climate_conditions: Dict,
                                 policy_factors: Dict) -> float:
        """Calculate crop production with seasonal and climate adjustments."""
        
        # Base growth rate
        base_growth = 0.03  # 3% annual growth
        
        # Seasonal adjustment
        seasonal_factor = self._get_crop_seasonal_factor(quarter)
        
        # Climate impact
        climate_impact = self._calculate_climate_impact(climate_conditions)
        
        # Policy impact (subsidies, irrigation, seeds)
        policy_impact = 1.0
        if 'fertilizer_subsidy' in policy_factors:
            policy_impact *= (1 + policy_factors['fertilizer_subsidy'] * 0.15)
        if 'irrigation_investment' in policy_factors:
            policy_impact *= (1 + policy_factors['irrigation_investment'] * 0.12)
        if 'seed_technology' in policy_factors:
            policy_impact *= (1 + policy_factors['seed_technology'] * 0.08)
        
        # Technology adoption factor
        tech_factor = 1 + (year - 2024) * 0.015  # 1.5% annual tech improvement
        
        # Calculate production
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            climate_impact *
            policy_impact *
            tech_factor
        )
        
        return production
    
    def _calculate_livestock_production(self, 
                                      base_value: float,
                                      quarter: int,
                                      year: int,
                                      climate_conditions: Dict,
                                      policy_factors: Dict) -> float:
        """Calculate livestock production."""
        
        # Base growth rate (higher than crops)
        base_growth = 0.045  # 4.5% annual growth
        
        # Seasonal factor (less seasonal than crops)
        seasonal_factor = 1.0 + 0.05 * np.sin(2 * np.pi * quarter / 4)
        
        # Climate impact (moderate)
        climate_impact = 1.0
        if climate_conditions.get('temperature_anomaly', 0) > 2:
            climate_impact *= 0.95  # Heat stress
        if climate_conditions.get('flood_severity', 0) > 0.5:
            climate_impact *= 0.90  # Flood impact on feed
        
        # Policy impact (veterinary services, feed subsidies)
        policy_impact = 1.0
        if 'veterinary_services' in policy_factors:
            policy_impact *= (1 + policy_factors['veterinary_services'] * 0.10)
        if 'feed_subsidy' in policy_factors:
            policy_impact *= (1 + policy_factors['feed_subsidy'] * 0.08)
        
        # Poultry growth factor (faster growth)
        poultry_boost = 1 + (year - 2024) * 0.02  # 2% additional growth
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            climate_impact *
            policy_impact *
            poultry_boost
        )
        
        return production
    
    def _calculate_forestry_production(self, 
                                     base_value: float,
                                     quarter: int,
                                     year: int,
                                     climate_conditions: Dict,
                                     policy_factors: Dict) -> float:
        """Calculate forestry production."""
        
        # Slow growth due to deforestation concerns
        base_growth = 0.015  # 1.5% annual growth
        
        # Seasonal factor (harvesting patterns)
        seasonal_factor = 1.0 + 0.08 * np.cos(2 * np.pi * (quarter - 1) / 4)
        
        # Climate impact
        climate_impact = 1.0
        if climate_conditions.get('cyclone_frequency', 0) > 0.3:
            climate_impact *= 0.85  # Cyclone damage
        
        # Policy impact (reforestation, conservation)
        policy_impact = 1.0
        if 'reforestation_program' in policy_factors:
            policy_impact *= (1 + policy_factors['reforestation_program'] * 0.20)
        if 'forest_conservation' in policy_factors:
            policy_impact *= (1 + policy_factors['forest_conservation'] * 0.05)
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            climate_impact *
            policy_impact
        )
        
        return production
    
    def _calculate_fishing_production(self, 
                                    base_value: float,
                                    quarter: int,
                                    year: int,
                                    climate_conditions: Dict,
                                    policy_factors: Dict) -> float:
        """Calculate fishing production."""
        
        # Moderate growth with sustainability constraints
        base_growth = 0.035  # 3.5% annual growth
        
        # Seasonal factor (monsoon affects fishing)
        seasonal_factor = 1.0
        if quarter == 2:  # Monsoon season
            seasonal_factor = 0.85
        elif quarter == 4:  # Post-monsoon peak
            seasonal_factor = 1.15
        
        # Climate impact
        climate_impact = 1.0
        if climate_conditions.get('sea_temperature_anomaly', 0) > 1:
            climate_impact *= 0.92  # Fish migration
        if climate_conditions.get('river_flow_anomaly', 0) < -0.5:
            climate_impact *= 0.88  # Low river flow
        
        # Policy impact (aquaculture development, marine protection)
        policy_impact = 1.0
        if 'aquaculture_development' in policy_factors:
            policy_impact *= (1 + policy_factors['aquaculture_development'] * 0.25)
        if 'marine_sanctuary' in policy_factors:
            policy_impact *= (1 + policy_factors['marine_sanctuary'] * 0.08)
        
        # Aquaculture growth factor
        aquaculture_boost = 1 + (year - 2024) * 0.025  # 2.5% additional growth
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            climate_impact *
            policy_impact *
            aquaculture_boost
        )
        
        return production
    
    def _get_crop_seasonal_factor(self, quarter: int) -> float:
        """Get seasonal adjustment factor for crops."""
        # Bangladesh crop calendar
        seasonal_factors = {
            1: 0.85,  # Q1: Winter crops harvesting
            2: 0.95,  # Q2: Pre-monsoon planting
            3: 1.10,  # Q3: Monsoon growing season
            4: 1.10   # Q4: Post-monsoon harvest (Aman)
        }
        return seasonal_factors.get(quarter, 1.0)
    
    def _calculate_climate_impact(self, climate_conditions: Dict) -> float:
        """Calculate overall climate impact on agriculture."""
        impact = 1.0
        
        # Monsoon impact
        monsoon_quality = climate_conditions.get('monsoon_quality', 'normal')
        if monsoon_quality in self.climate_factors:
            impact *= self.climate_factors[monsoon_quality]
        
        # Extreme weather events
        if climate_conditions.get('flood_severity', 0) > 0.3:
            impact *= self.climate_factors['flood_impact']
        
        if climate_conditions.get('drought_severity', 0) > 0.3:
            impact *= self.climate_factors['drought_impact']
        
        if climate_conditions.get('cyclone_impact', 0) > 0.2:
            impact *= self.climate_factors['cyclone_impact']
        
        # Temperature stress
        temp_anomaly = climate_conditions.get('temperature_anomaly', 0)
        if abs(temp_anomaly) > 1.5:
            impact *= (1 - abs(temp_anomaly) * 0.05)  # 5% impact per degree
        
        return max(impact, 0.5)  # Minimum 50% of normal production
    
    def _calculate_growth_factors(self, climate_conditions: Dict, policy_factors: Dict) -> Dict:
        """Calculate various growth factors affecting agriculture."""
        return {
            'climate_factor': self._calculate_climate_impact(climate_conditions),
            'technology_factor': 1.015,  # Annual tech improvement
            'policy_factor': 1.0 + sum(policy_factors.values()) * 0.1,
            'market_factor': 1.02,  # Market development
            'infrastructure_factor': 1.01  # Rural infrastructure
        }
    
    def _get_seasonal_adjustment(self, quarter: int) -> Dict:
        """Get seasonal adjustment details."""
        adjustments = {
            1: {'factor': 0.85, 'reason': 'Winter crop harvest, lower activity'},
            2: {'factor': 0.95, 'reason': 'Pre-monsoon preparation'},
            3: {'factor': 1.10, 'reason': 'Monsoon planting and growing'},
            4: {'factor': 1.10, 'reason': 'Major harvest season (Aman rice)'}
        }
        return adjustments.get(quarter, {'factor': 1.0, 'reason': 'Normal season'})
    
    def _assess_climate_impact(self, climate_conditions: Dict) -> Dict:
        """Assess detailed climate impact on agriculture."""
        return {
            'overall_impact': self._calculate_climate_impact(climate_conditions),
            'monsoon_impact': climate_conditions.get('monsoon_quality', 'normal'),
            'flood_risk': climate_conditions.get('flood_severity', 0),
            'drought_risk': climate_conditions.get('drought_severity', 0),
            'temperature_stress': climate_conditions.get('temperature_anomaly', 0),
            'adaptation_measures': self._suggest_adaptation_measures(climate_conditions)
        }
    
    def _suggest_adaptation_measures(self, climate_conditions: Dict) -> List[str]:
        """Suggest climate adaptation measures."""
        measures = []
        
        if climate_conditions.get('flood_severity', 0) > 0.3:
            measures.extend([
                'Flood-resistant crop varieties',
                'Improved drainage systems',
                'Early warning systems'
            ])
        
        if climate_conditions.get('drought_severity', 0) > 0.3:
            measures.extend([
                'Drought-tolerant varieties',
                'Efficient irrigation systems',
                'Water conservation techniques'
            ])
        
        if climate_conditions.get('temperature_anomaly', 0) > 1.5:
            measures.extend([
                'Heat-resistant crop varieties',
                'Shade management',
                'Adjusted planting schedules'
            ])
        
        return measures
    
    def get_sector_indicators(self, production_data: Dict) -> Dict:
        """Get key agriculture sector indicators."""
        return {
            'total_agriculture_gdp': production_data['total_agriculture_gdp'],
            'crop_production_index': production_data['subsector_breakdown']['crops'] / (production_data['total_agriculture_gdp'] * self.subsector_shares['crops']),
            'livestock_production_index': production_data['subsector_breakdown']['livestock'] / (production_data['total_agriculture_gdp'] * self.subsector_shares['livestock']),
            'fisheries_production_index': production_data['subsector_breakdown']['fishing'] / (production_data['total_agriculture_gdp'] * self.subsector_shares['fishing']),
            'climate_resilience_score': min(1.0, production_data['climate_impact']['overall_impact'] + 0.2),
            'food_security_indicator': self._calculate_food_security_indicator(production_data),
            'export_potential': self._calculate_export_potential(production_data)
        }
    
    def _calculate_food_security_indicator(self, production_data: Dict) -> float:
        """Calculate food security indicator based on production."""
        # Simplified food security calculation
        rice_adequacy = min(1.0, production_data['subsector_breakdown']['crops'] / (production_data['total_agriculture_gdp'] * 0.42))
        protein_adequacy = min(1.0, (production_data['subsector_breakdown']['livestock'] + production_data['subsector_breakdown']['fishing']) / (production_data['total_agriculture_gdp'] * 0.33))
        
        return (rice_adequacy * 0.6 + protein_adequacy * 0.4)
    
    def _calculate_export_potential(self, production_data: Dict) -> float:
        """Calculate agricultural export potential."""
        # Based on surplus production capacity
        total_production = production_data['total_agriculture_gdp']
        domestic_consumption_ratio = 0.85  # 85% for domestic consumption
        
        export_potential = max(0, (total_production - total_production * domestic_consumption_ratio) / total_production)
        return min(export_potential, 0.25)  # Maximum 25% export potential