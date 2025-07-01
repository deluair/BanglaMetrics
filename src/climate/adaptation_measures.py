"""Climate Adaptation Measures for Bangladesh GDP Simulation.

This module provides comprehensive climate adaptation strategies, measures,
and recommendations for Bangladesh's economic sectors.
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


class AdaptationCategory(Enum):
    """Categories of adaptation measures."""
    INFRASTRUCTURE = "infrastructure"
    TECHNOLOGY = "technology"
    POLICY = "policy"
    CAPACITY_BUILDING = "capacity_building"
    ECOSYSTEM = "ecosystem"
    FINANCIAL = "financial"
    SOCIAL = "social"


class ImplementationTimeframe(Enum):
    """Implementation timeframes for adaptation measures."""
    IMMEDIATE = "immediate"  # 0-1 years
    SHORT_TERM = "short_term"  # 1-3 years
    MEDIUM_TERM = "medium_term"  # 3-7 years
    LONG_TERM = "long_term"  # 7+ years


class Priority(Enum):
    """Priority levels for adaptation measures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AdaptationMeasure:
    """Climate adaptation measure definition."""
    id: str
    name: str
    description: str
    category: AdaptationCategory
    target_sectors: List[str]
    climate_risks_addressed: List[str]
    effectiveness: float  # 0.0 to 1.0
    cost_estimate: float  # Million BDT
    implementation_timeframe: ImplementationTimeframe
    priority: Priority
    co_benefits: List[str]
    barriers: List[str]
    success_indicators: List[str]
    regional_applicability: List[str]
    stakeholders: List[str]


@dataclass
class AdaptationPlan:
    """Comprehensive adaptation plan."""
    sector: str
    climate_risks: List[str]
    adaptation_measures: List[AdaptationMeasure]
    total_cost: float
    implementation_timeline: Dict[str, List[str]]
    expected_benefits: Dict[str, float]
    monitoring_indicators: List[str]
    review_schedule: str


@dataclass
class CostBenefitAnalysis:
    """Cost-benefit analysis for adaptation measures."""
    measure_id: str
    implementation_cost: float
    annual_maintenance_cost: float
    avoided_damages: float
    co_benefits_value: float
    net_present_value: float
    benefit_cost_ratio: float
    payback_period: float
    confidence_level: float


class ClimateAdaptationManager:
    """Comprehensive climate adaptation management system.
    
    This class provides adaptation strategies, measures, and planning
    tools for Bangladesh's economic sectors.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the climate adaptation manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize adaptation measures database
        self.adaptation_measures = self._initialize_adaptation_measures()
        
        # Sector-specific adaptation strategies
        self.sector_strategies = self._initialize_sector_strategies()
        
        # Regional adaptation priorities
        self.regional_priorities = self._initialize_regional_priorities()
        
        # Cost-benefit parameters
        self.cost_benefit_parameters = self._initialize_cost_benefit_parameters()
        
        logger.info("Climate adaptation manager initialized")
    
    def _initialize_adaptation_measures(self) -> Dict[str, AdaptationMeasure]:
        """Initialize comprehensive adaptation measures database."""
        
        measures = {}
        
        # Agriculture sector measures
        measures['agri_drought_resistant_crops'] = AdaptationMeasure(
            id='agri_drought_resistant_crops',
            name='Drought-Resistant Crop Varieties',
            description='Development and promotion of drought-tolerant rice, wheat, and other crop varieties',
            category=AdaptationCategory.TECHNOLOGY,
            target_sectors=['agriculture'],
            climate_risks_addressed=['drought', 'temperature_increase', 'irregular_rainfall'],
            effectiveness=0.7,
            cost_estimate=500,  # Million BDT
            implementation_timeframe=ImplementationTimeframe.MEDIUM_TERM,
            priority=Priority.HIGH,
            co_benefits=['food_security', 'farmer_income', 'export_potential'],
            barriers=['farmer_acceptance', 'seed_availability', 'technical_knowledge'],
            success_indicators=['adoption_rate', 'yield_stability', 'water_use_efficiency'],
            regional_applicability=['northern', 'northwestern', 'central'],
            stakeholders=['farmers', 'research_institutes', 'government', 'seed_companies']
        )
        
        measures['agri_improved_irrigation'] = AdaptationMeasure(
            id='agri_improved_irrigation',
            name='Improved Irrigation Systems',
            description='Installation of efficient drip and sprinkler irrigation systems',
            category=AdaptationCategory.INFRASTRUCTURE,
            target_sectors=['agriculture'],
            climate_risks_addressed=['drought', 'irregular_rainfall', 'water_scarcity'],
            effectiveness=0.8,
            cost_estimate=2000,
            implementation_timeframe=ImplementationTimeframe.SHORT_TERM,
            priority=Priority.HIGH,
            co_benefits=['water_conservation', 'increased_productivity', 'reduced_labor'],
            barriers=['high_initial_cost', 'technical_expertise', 'maintenance_requirements'],
            success_indicators=['water_use_efficiency', 'crop_yield', 'area_coverage'],
            regional_applicability=['all_regions'],
            stakeholders=['farmers', 'water_management_boards', 'government', 'private_sector']
        )
        
        measures['agri_flood_resistant_varieties'] = AdaptationMeasure(
            id='agri_flood_resistant_varieties',
            name='Flood-Resistant Crop Varieties',
            description='Development of submergence-tolerant rice and other flood-resistant crops',
            category=AdaptationCategory.TECHNOLOGY,
            target_sectors=['agriculture'],
            climate_risks_addressed=['flooding', 'waterlogging', 'extreme_precipitation'],
            effectiveness=0.6,
            cost_estimate=300,
            implementation_timeframe=ImplementationTimeframe.MEDIUM_TERM,
            priority=Priority.HIGH,
            co_benefits=['food_security', 'reduced_crop_loss', 'farmer_resilience'],
            barriers=['limited_varieties', 'farmer_awareness', 'seed_distribution'],
            success_indicators=['flood_survival_rate', 'yield_recovery', 'adoption_rate'],
            regional_applicability=['coastal', 'river_delta', 'flood_prone_areas'],
            stakeholders=['farmers', 'research_institutes', 'extension_services']
        )
        
        measures['agri_climate_smart_practices'] = AdaptationMeasure(
            id='agri_climate_smart_practices',
            name='Climate-Smart Agricultural Practices',
            description='Promotion of conservation agriculture, agroforestry, and integrated farming',
            category=AdaptationCategory.TECHNOLOGY,
            target_sectors=['agriculture'],
            climate_risks_addressed=['soil_degradation', 'temperature_increase', 'irregular_rainfall'],
            effectiveness=0.5,
            cost_estimate=800,
            implementation_timeframe=ImplementationTimeframe.SHORT_TERM,
            priority=Priority.MEDIUM,
            co_benefits=['soil_health', 'carbon_sequestration', 'biodiversity'],
            barriers=['knowledge_gap', 'initial_investment', 'market_access'],
            success_indicators=['soil_organic_matter', 'water_retention', 'yield_stability'],
            regional_applicability=['all_regions'],
            stakeholders=['farmers', 'extension_services', 'ngos', 'research_institutes']
        )
        
        # Manufacturing sector measures
        measures['manu_climate_controlled_facilities'] = AdaptationMeasure(
            id='manu_climate_controlled_facilities',
            name='Climate-Controlled Manufacturing Facilities',
            description='Installation of advanced HVAC systems and climate control in factories',
            category=AdaptationCategory.INFRASTRUCTURE,
            target_sectors=['manufacturing'],
            climate_risks_addressed=['heat_waves', 'humidity', 'temperature_extremes'],
            effectiveness=0.9,
            cost_estimate=1500,
            implementation_timeframe=ImplementationTimeframe.SHORT_TERM,
            priority=Priority.MEDIUM,
            co_benefits=['worker_productivity', 'product_quality', 'equipment_longevity'],
            barriers=['high_energy_cost', 'initial_investment', 'technical_expertise'],
            success_indicators=['temperature_stability', 'productivity_index', 'defect_rate'],
            regional_applicability=['all_regions'],
            stakeholders=['manufacturers', 'workers', 'energy_providers']
        )
        
        measures['manu_backup_power_systems'] = AdaptationMeasure(
            id='manu_backup_power_systems',
            name='Backup Power and Energy Storage',
            description='Installation of backup generators and battery storage systems',
            category=AdaptationCategory.INFRASTRUCTURE,
            target_sectors=['manufacturing'],
            climate_risks_addressed=['power_outages', 'extreme_weather', 'grid_instability'],
            effectiveness=0.8,
            cost_estimate=1000,
            implementation_timeframe=ImplementationTimeframe.SHORT_TERM,
            priority=Priority.HIGH,
            co_benefits=['production_continuity', 'reduced_downtime', 'energy_security'],
            barriers=['high_cost', 'maintenance_requirements', 'fuel_availability'],
            success_indicators=['uptime_percentage', 'production_continuity', 'energy_reliability'],
            regional_applicability=['all_regions'],
            stakeholders=['manufacturers', 'energy_providers', 'equipment_suppliers']
        )
        
        measures['manu_flood_protection'] = AdaptationMeasure(
            id='manu_flood_protection',
            name='Industrial Flood Protection',
            description='Construction of flood barriers, elevated platforms, and drainage systems',
            category=AdaptationCategory.INFRASTRUCTURE,
            target_sectors=['manufacturing'],
            climate_risks_addressed=['flooding', 'storm_surge', 'extreme_precipitation'],
            effectiveness=0.8,
            cost_estimate=2500,
            implementation_timeframe=ImplementationTimeframe.MEDIUM_TERM,
            priority=Priority.HIGH,
            co_benefits=['asset_protection', 'business_continuity', 'worker_safety'],
            barriers=['high_cost', 'land_availability', 'regulatory_approval'],
            success_indicators=['flood_protection_level', 'damage_reduction', 'business_continuity'],
            regional_applicability=['coastal', 'river_delta', 'flood_prone_areas'],
            stakeholders=['manufacturers', 'government', 'insurance_companies']
        )
        
        # Infrastructure measures
        measures['infra_elevated_roads'] = AdaptationMeasure(
            id='infra_elevated_roads',
            name='Elevated Road Infrastructure',
            description='Construction of elevated highways and bridges in flood-prone areas',
            category=AdaptationCategory.INFRASTRUCTURE,
            target_sectors=['transport', 'all_sectors'],
            climate_risks_addressed=['flooding', 'sea_level_rise', 'storm_surge'],
            effectiveness=0.9,
            cost_estimate=10000,
            implementation_timeframe=ImplementationTimeframe.LONG_TERM,
            priority=Priority.HIGH,
            co_benefits=['improved_connectivity', 'economic_development', 'emergency_access'],
            barriers=['very_high_cost', 'land_acquisition', 'environmental_impact'],
            success_indicators=['flood_resilience', 'traffic_flow', 'connectivity_index'],
            regional_applicability=['coastal', 'river_delta', 'flood_prone_areas'],
            stakeholders=['government', 'transport_authorities', 'communities']
        )
        
        measures['infra_improved_drainage'] = AdaptationMeasure(
            id='infra_improved_drainage',
            name='Enhanced Urban Drainage Systems',
            description='Upgrade of storm water drainage and sewerage systems',
            category=AdaptationCategory.INFRASTRUCTURE,
            target_sectors=['urban_services', 'all_sectors'],
            climate_risks_addressed=['urban_flooding', 'extreme_precipitation', 'waterlogging'],
            effectiveness=0.7,
            cost_estimate=3000,
            implementation_timeframe=ImplementationTimeframe.MEDIUM_TERM,
            priority=Priority.HIGH,
            co_benefits=['public_health', 'urban_livability', 'property_values'],
            barriers=['high_cost', 'urban_planning', 'maintenance_requirements'],
            success_indicators=['flood_frequency', 'drainage_capacity', 'water_quality'],
            regional_applicability=['urban_areas', 'coastal_cities'],
            stakeholders=['city_corporations', 'residents', 'businesses']
        )
        
        measures['infra_cyclone_shelters'] = AdaptationMeasure(
            id='infra_cyclone_shelters',
            name='Cyclone Shelter Network',
            description='Construction of multipurpose cyclone shelters and safe buildings',
            category=AdaptationCategory.INFRASTRUCTURE,
            target_sectors=['all_sectors'],
            climate_risks_addressed=['cyclones', 'storm_surge', 'extreme_winds'],
            effectiveness=0.95,
            cost_estimate=1500,
            implementation_timeframe=ImplementationTimeframe.MEDIUM_TERM,
            priority=Priority.CRITICAL,
            co_benefits=['life_safety', 'community_resilience', 'emergency_response'],
            barriers=['site_selection', 'community_acceptance', 'maintenance_funding'],
            success_indicators=['shelter_capacity', 'evacuation_time', 'lives_saved'],
            regional_applicability=['coastal', 'cyclone_prone_areas'],
            stakeholders=['government', 'communities', 'disaster_management']
        )
        
        # Water resources measures
        measures['water_rainwater_harvesting'] = AdaptationMeasure(
            id='water_rainwater_harvesting',
            name='Rainwater Harvesting Systems',
            description='Installation of rainwater collection and storage systems',
            category=AdaptationCategory.INFRASTRUCTURE,
            target_sectors=['agriculture', 'urban_services'],
            climate_risks_addressed=['water_scarcity', 'irregular_rainfall', 'drought'],
            effectiveness=0.6,
            cost_estimate=400,
            implementation_timeframe=ImplementationTimeframe.SHORT_TERM,
            priority=Priority.MEDIUM,
            co_benefits=['water_security', 'reduced_groundwater_use', 'flood_mitigation'],
            barriers=['storage_capacity', 'water_quality', 'maintenance_requirements'],
            success_indicators=['water_storage_capacity', 'usage_rate', 'water_quality'],
            regional_applicability=['all_regions'],
            stakeholders=['communities', 'government', 'water_authorities']
        )
        
        measures['water_groundwater_management'] = AdaptationMeasure(
            id='water_groundwater_management',
            name='Sustainable Groundwater Management',
            description='Implementation of groundwater monitoring and sustainable extraction practices',
            category=AdaptationCategory.POLICY,
            target_sectors=['agriculture', 'urban_services'],
            climate_risks_addressed=['groundwater_depletion', 'saltwater_intrusion', 'drought'],
            effectiveness=0.7,
            cost_estimate=600,
            implementation_timeframe=ImplementationTimeframe.MEDIUM_TERM,
            priority=Priority.HIGH,
            co_benefits=['water_sustainability', 'aquifer_protection', 'water_quality'],
            barriers=['regulatory_framework', 'monitoring_capacity', 'user_compliance'],
            success_indicators=['groundwater_levels', 'extraction_rates', 'water_quality'],
            regional_applicability=['all_regions'],
            stakeholders=['water_authorities', 'users', 'government', 'communities']
        )
        
        # Coastal protection measures
        measures['coastal_mangrove_restoration'] = AdaptationMeasure(
            id='coastal_mangrove_restoration',
            name='Mangrove Forest Restoration',
            description='Restoration and expansion of mangrove forests along the coast',
            category=AdaptationCategory.ECOSYSTEM,
            target_sectors=['fisheries', 'coastal_communities'],
            climate_risks_addressed=['storm_surge', 'coastal_erosion', 'sea_level_rise'],
            effectiveness=0.8,
            cost_estimate=300,
            implementation_timeframe=ImplementationTimeframe.MEDIUM_TERM,
            priority=Priority.HIGH,
            co_benefits=['biodiversity', 'fisheries', 'carbon_sequestration', 'tourism'],
            barriers=['land_availability', 'community_cooperation', 'long_term_commitment'],
            success_indicators=['mangrove_coverage', 'coastal_protection', 'biodiversity_index'],
            regional_applicability=['coastal'],
            stakeholders=['communities', 'government', 'environmental_groups']
        )
        
        measures['coastal_sea_walls'] = AdaptationMeasure(
            id='coastal_sea_walls',
            name='Coastal Protection Structures',
            description='Construction of sea walls, embankments, and coastal barriers',
            category=AdaptationCategory.INFRASTRUCTURE,
            target_sectors=['coastal_communities', 'agriculture'],
            climate_risks_addressed=['sea_level_rise', 'storm_surge', 'coastal_erosion'],
            effectiveness=0.9,
            cost_estimate=5000,
            implementation_timeframe=ImplementationTimeframe.LONG_TERM,
            priority=Priority.HIGH,
            co_benefits=['land_protection', 'agricultural_productivity', 'settlement_security'],
            barriers=['very_high_cost', 'environmental_impact', 'maintenance_requirements'],
            success_indicators=['protection_level', 'erosion_rate', 'land_preservation'],
            regional_applicability=['coastal'],
            stakeholders=['government', 'coastal_communities', 'engineers']
        )
        
        # Early warning systems
        measures['early_warning_systems'] = AdaptationMeasure(
            id='early_warning_systems',
            name='Enhanced Early Warning Systems',
            description='Upgrade of weather monitoring and early warning communication systems',
            category=AdaptationCategory.TECHNOLOGY,
            target_sectors=['all_sectors'],
            climate_risks_addressed=['all_extreme_events'],
            effectiveness=0.8,
            cost_estimate=200,
            implementation_timeframe=ImplementationTimeframe.SHORT_TERM,
            priority=Priority.CRITICAL,
            co_benefits=['disaster_preparedness', 'life_safety', 'economic_protection'],
            barriers=['technology_requirements', 'communication_infrastructure', 'user_training'],
            success_indicators=['warning_accuracy', 'response_time', 'coverage_area'],
            regional_applicability=['all_regions'],
            stakeholders=['meteorological_department', 'disaster_management', 'communities']
        )
        
        # Capacity building measures
        measures['capacity_climate_training'] = AdaptationMeasure(
            id='capacity_climate_training',
            name='Climate Adaptation Training Programs',
            description='Training programs for farmers, workers, and communities on climate adaptation',
            category=AdaptationCategory.CAPACITY_BUILDING,
            target_sectors=['all_sectors'],
            climate_risks_addressed=['knowledge_gaps', 'adaptive_capacity'],
            effectiveness=0.6,
            cost_estimate=150,
            implementation_timeframe=ImplementationTimeframe.SHORT_TERM,
            priority=Priority.MEDIUM,
            co_benefits=['human_capital', 'innovation', 'community_resilience'],
            barriers=['training_capacity', 'participant_availability', 'follow_up_support'],
            success_indicators=['training_coverage', 'knowledge_improvement', 'practice_adoption'],
            regional_applicability=['all_regions'],
            stakeholders=['training_institutes', 'communities', 'government', 'ngos']
        )
        
        return measures
    
    def _initialize_sector_strategies(self) -> Dict[str, Dict]:
        """Initialize sector-specific adaptation strategies."""
        
        return {
            'agriculture': {
                'primary_risks': ['drought', 'flooding', 'temperature_increase', 'irregular_rainfall'],
                'priority_measures': [
                    'agri_drought_resistant_crops',
                    'agri_improved_irrigation',
                    'agri_flood_resistant_varieties',
                    'water_rainwater_harvesting'
                ],
                'adaptation_goals': [
                    'maintain_food_security',
                    'increase_climate_resilience',
                    'improve_water_use_efficiency',
                    'reduce_crop_losses'
                ],
                'success_metrics': [
                    'crop_yield_stability',
                    'water_productivity',
                    'farmer_income',
                    'food_security_index'
                ]
            },
            'manufacturing': {
                'primary_risks': ['heat_waves', 'power_outages', 'flooding', 'supply_chain_disruption'],
                'priority_measures': [
                    'manu_climate_controlled_facilities',
                    'manu_backup_power_systems',
                    'manu_flood_protection',
                    'early_warning_systems'
                ],
                'adaptation_goals': [
                    'maintain_production_continuity',
                    'protect_assets_and_equipment',
                    'ensure_worker_safety',
                    'reduce_climate_risks'
                ],
                'success_metrics': [
                    'production_uptime',
                    'asset_protection_level',
                    'worker_productivity',
                    'climate_risk_reduction'
                ]
            },
            'services': {
                'primary_risks': ['extreme_weather', 'power_outages', 'transport_disruption'],
                'priority_measures': [
                    'manu_backup_power_systems',
                    'infra_improved_drainage',
                    'early_warning_systems',
                    'capacity_climate_training'
                ],
                'adaptation_goals': [
                    'maintain_service_delivery',
                    'ensure_business_continuity',
                    'protect_infrastructure',
                    'build_adaptive_capacity'
                ],
                'success_metrics': [
                    'service_availability',
                    'business_continuity_index',
                    'infrastructure_resilience',
                    'adaptive_capacity_score'
                ]
            },
            'construction': {
                'primary_risks': ['heat_waves', 'extreme_precipitation', 'strong_winds'],
                'priority_measures': [
                    'capacity_climate_training',
                    'early_warning_systems',
                    'infra_improved_drainage'
                ],
                'adaptation_goals': [
                    'ensure_worker_safety',
                    'maintain_construction_schedules',
                    'build_climate_resilient_infrastructure'
                ],
                'success_metrics': [
                    'worker_safety_index',
                    'project_completion_rate',
                    'infrastructure_resilience_rating'
                ]
            },
            'transport': {
                'primary_risks': ['flooding', 'extreme_weather', 'sea_level_rise'],
                'priority_measures': [
                    'infra_elevated_roads',
                    'infra_improved_drainage',
                    'early_warning_systems'
                ],
                'adaptation_goals': [
                    'maintain_connectivity',
                    'ensure_transport_reliability',
                    'protect_transport_infrastructure'
                ],
                'success_metrics': [
                    'connectivity_index',
                    'transport_reliability',
                    'infrastructure_resilience'
                ]
            },
            'coastal_communities': {
                'primary_risks': ['sea_level_rise', 'storm_surge', 'cyclones', 'coastal_erosion'],
                'priority_measures': [
                    'infra_cyclone_shelters',
                    'coastal_mangrove_restoration',
                    'coastal_sea_walls',
                    'early_warning_systems'
                ],
                'adaptation_goals': [
                    'protect_lives_and_livelihoods',
                    'preserve_coastal_ecosystems',
                    'maintain_coastal_settlements'
                ],
                'success_metrics': [
                    'population_safety',
                    'ecosystem_health',
                    'settlement_sustainability'
                ]
            }
        }
    
    def _initialize_regional_priorities(self) -> Dict[str, Dict]:
        """Initialize regional adaptation priorities."""
        
        return {
            'coastal': {
                'priority_risks': ['sea_level_rise', 'cyclones', 'storm_surge', 'saltwater_intrusion'],
                'priority_measures': [
                    'infra_cyclone_shelters',
                    'coastal_mangrove_restoration',
                    'coastal_sea_walls',
                    'early_warning_systems'
                ],
                'investment_priority': 'critical',
                'timeframe': 'immediate_to_medium_term'
            },
            'northern': {
                'priority_risks': ['drought', 'temperature_extremes', 'irregular_rainfall'],
                'priority_measures': [
                    'agri_drought_resistant_crops',
                    'agri_improved_irrigation',
                    'water_rainwater_harvesting',
                    'water_groundwater_management'
                ],
                'investment_priority': 'high',
                'timeframe': 'short_to_medium_term'
            },
            'northeastern': {
                'priority_risks': ['flash_floods', 'landslides', 'extreme_precipitation'],
                'priority_measures': [
                    'infra_improved_drainage',
                    'agri_flood_resistant_varieties',
                    'early_warning_systems'
                ],
                'investment_priority': 'high',
                'timeframe': 'short_to_medium_term'
            },
            'central': {
                'priority_risks': ['flooding', 'drought', 'temperature_increase'],
                'priority_measures': [
                    'agri_improved_irrigation',
                    'infra_improved_drainage',
                    'water_rainwater_harvesting'
                ],
                'investment_priority': 'medium',
                'timeframe': 'medium_term'
            },
            'southwestern': {
                'priority_risks': ['saltwater_intrusion', 'cyclones', 'drought'],
                'priority_measures': [
                    'coastal_mangrove_restoration',
                    'water_groundwater_management',
                    'agri_drought_resistant_crops'
                ],
                'investment_priority': 'high',
                'timeframe': 'short_to_medium_term'
            },
            'urban_areas': {
                'priority_risks': ['urban_flooding', 'heat_islands', 'power_outages'],
                'priority_measures': [
                    'infra_improved_drainage',
                    'manu_backup_power_systems',
                    'early_warning_systems'
                ],
                'investment_priority': 'high',
                'timeframe': 'immediate_to_short_term'
            }
        }
    
    def _initialize_cost_benefit_parameters(self) -> Dict[str, Dict]:
        """Initialize cost-benefit analysis parameters."""
        
        return {
            'discount_rate': 0.08,  # 8% annual discount rate
            'analysis_period': 20,  # 20 years
            'damage_cost_factors': {
                'agriculture': {
                    'drought': 1000,  # Million BDT per severe event
                    'flooding': 800,
                    'temperature_increase': 500
                },
                'manufacturing': {
                    'heat_waves': 300,
                    'flooding': 1500,
                    'power_outages': 200
                },
                'infrastructure': {
                    'flooding': 2000,
                    'cyclones': 3000,
                    'sea_level_rise': 1000
                }
            },
            'co_benefit_values': {
                'biodiversity': 50,  # Million BDT per unit
                'carbon_sequestration': 30,
                'water_conservation': 40,
                'job_creation': 20,
                'health_benefits': 100
            }
        }
    
    def recommend_adaptation_measures(self, 
                                    sector: str,
                                    climate_risks: List[str],
                                    region: str = None,
                                    budget_constraint: float = None) -> List[AdaptationMeasure]:
        """Recommend adaptation measures for specific sector and risks.
        
        Args:
            sector: Target sector
            climate_risks: List of climate risks to address
            region: Specific region (optional)
            budget_constraint: Budget limit in million BDT (optional)
            
        Returns:
            List of recommended adaptation measures
        """
        try:
            recommendations = []
            
            # Get sector strategy
            sector_strategy = self.sector_strategies.get(sector, {})
            priority_measures = sector_strategy.get('priority_measures', [])
            
            # Get regional priorities if specified
            regional_measures = []
            if region and region in self.regional_priorities:
                regional_measures = self.regional_priorities[region]['priority_measures']
            
            # Filter measures based on criteria
            for measure_id, measure in self.adaptation_measures.items():
                # Check if measure targets the sector
                if sector not in measure.target_sectors and 'all_sectors' not in measure.target_sectors:
                    continue
                
                # Check if measure addresses the climate risks
                if not any(risk in measure.climate_risks_addressed for risk in climate_risks):
                    continue
                
                # Check regional applicability
                if region and region not in measure.regional_applicability and 'all_regions' not in measure.regional_applicability:
                    continue
                
                # Add priority score
                priority_score = 0
                if measure_id in priority_measures:
                    priority_score += 3
                if measure_id in regional_measures:
                    priority_score += 2
                if measure.priority == Priority.CRITICAL:
                    priority_score += 4
                elif measure.priority == Priority.HIGH:
                    priority_score += 3
                elif measure.priority == Priority.MEDIUM:
                    priority_score += 2
                else:
                    priority_score += 1
                
                # Add effectiveness score
                effectiveness_score = measure.effectiveness * 5
                
                # Calculate total score
                total_score = priority_score + effectiveness_score
                
                recommendations.append((measure, total_score))
            
            # Sort by score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # Apply budget constraint if specified
            if budget_constraint:
                filtered_recommendations = []
                total_cost = 0
                
                for measure, score in recommendations:
                    if total_cost + measure.cost_estimate <= budget_constraint:
                        filtered_recommendations.append(measure)
                        total_cost += measure.cost_estimate
                    elif not filtered_recommendations:  # Include at least one measure
                        filtered_recommendations.append(measure)
                        break
                
                return filtered_recommendations
            
            # Return top recommendations
            return [measure for measure, score in recommendations[:10]]
            
        except Exception as e:
            logger.error(f"Error recommending adaptation measures: {str(e)}")
            return []
    
    def create_adaptation_plan(self, 
                             sector: str,
                             climate_risks: List[str],
                             region: str = None,
                             budget: float = None,
                             timeframe: str = 'medium_term') -> AdaptationPlan:
        """Create comprehensive adaptation plan for a sector.
        
        Args:
            sector: Target sector
            climate_risks: List of climate risks
            region: Specific region (optional)
            budget: Available budget in million BDT (optional)
            timeframe: Planning timeframe
            
        Returns:
            Comprehensive adaptation plan
        """
        try:
            # Get recommended measures
            recommended_measures = self.recommend_adaptation_measures(
                sector, climate_risks, region, budget
            )
            
            # Calculate total cost
            total_cost = sum(measure.cost_estimate for measure in recommended_measures)
            
            # Create implementation timeline
            timeline = {
                'immediate': [],
                'short_term': [],
                'medium_term': [],
                'long_term': []
            }
            
            for measure in recommended_measures:
                timeframe_key = measure.implementation_timeframe.value
                timeline[timeframe_key].append(measure.id)
            
            # Calculate expected benefits
            expected_benefits = self._calculate_expected_benefits(recommended_measures, climate_risks)
            
            # Define monitoring indicators
            monitoring_indicators = self._define_monitoring_indicators(recommended_measures, sector)
            
            # Create adaptation plan
            plan = AdaptationPlan(
                sector=sector,
                climate_risks=climate_risks,
                adaptation_measures=recommended_measures,
                total_cost=total_cost,
                implementation_timeline=timeline,
                expected_benefits=expected_benefits,
                monitoring_indicators=monitoring_indicators,
                review_schedule='annual'
            )
            
            logger.info(f"Created adaptation plan for {sector} with {len(recommended_measures)} measures")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating adaptation plan: {str(e)}")
            return AdaptationPlan(
                sector=sector,
                climate_risks=climate_risks,
                adaptation_measures=[],
                total_cost=0,
                implementation_timeline={},
                expected_benefits={},
                monitoring_indicators=[],
                review_schedule='annual'
            )
    
    def _calculate_expected_benefits(self, 
                                   measures: List[AdaptationMeasure],
                                   climate_risks: List[str]) -> Dict[str, float]:
        """Calculate expected benefits from adaptation measures."""
        
        benefits = {
            'risk_reduction': 0.0,
            'economic_benefits': 0.0,
            'co_benefits_value': 0.0,
            'total_benefits': 0.0
        }
        
        for measure in measures:
            # Risk reduction benefit
            risk_reduction = measure.effectiveness * len(set(measure.climate_risks_addressed) & set(climate_risks))
            benefits['risk_reduction'] += risk_reduction
            
            # Economic benefits (simplified)
            economic_benefit = measure.cost_estimate * measure.effectiveness * 0.5  # 50% of cost as benefit
            benefits['economic_benefits'] += economic_benefit
            
            # Co-benefits value
            co_benefit_value = len(measure.co_benefits) * 50  # 50 million BDT per co-benefit
            benefits['co_benefits_value'] += co_benefit_value
        
        benefits['total_benefits'] = (
            benefits['economic_benefits'] + benefits['co_benefits_value']
        )
        
        return benefits
    
    def _define_monitoring_indicators(self, 
                                    measures: List[AdaptationMeasure],
                                    sector: str) -> List[str]:
        """Define monitoring indicators for adaptation plan."""
        
        indicators = set()
        
        # Add measure-specific indicators
        for measure in measures:
            indicators.update(measure.success_indicators)
        
        # Add sector-specific indicators
        sector_strategy = self.sector_strategies.get(sector, {})
        sector_metrics = sector_strategy.get('success_metrics', [])
        indicators.update(sector_metrics)
        
        # Add general adaptation indicators
        general_indicators = [
            'climate_risk_index',
            'adaptive_capacity_score',
            'resilience_rating',
            'implementation_progress'
        ]
        indicators.update(general_indicators)
        
        return list(indicators)
    
    def conduct_cost_benefit_analysis(self, 
                                    measure: AdaptationMeasure,
                                    climate_risks: List[str],
                                    sector: str) -> CostBenefitAnalysis:
        """Conduct cost-benefit analysis for an adaptation measure.
        
        Args:
            measure: Adaptation measure to analyze
            climate_risks: Climate risks being addressed
            sector: Target sector
            
        Returns:
            Cost-benefit analysis results
        """
        try:
            # Get parameters
            params = self.cost_benefit_parameters
            discount_rate = params['discount_rate']
            analysis_period = params['analysis_period']
            
            # Calculate costs
            implementation_cost = measure.cost_estimate
            annual_maintenance_cost = implementation_cost * 0.05  # 5% of implementation cost
            
            # Calculate avoided damages
            avoided_damages = self._calculate_avoided_damages(measure, climate_risks, sector)
            
            # Calculate co-benefits value
            co_benefits_value = self._calculate_co_benefits_value(measure)
            
            # Calculate present values
            total_costs = implementation_cost
            total_benefits = 0
            
            for year in range(1, analysis_period + 1):
                # Annual costs (maintenance)
                annual_cost = annual_maintenance_cost / ((1 + discount_rate) ** year)
                total_costs += annual_cost
                
                # Annual benefits (avoided damages + co-benefits)
                annual_benefit = (avoided_damages + co_benefits_value) / ((1 + discount_rate) ** year)
                total_benefits += annual_benefit
            
            # Calculate metrics
            net_present_value = total_benefits - total_costs
            benefit_cost_ratio = total_benefits / total_costs if total_costs > 0 else 0
            
            # Calculate payback period
            payback_period = self._calculate_payback_period(
                implementation_cost, avoided_damages + co_benefits_value, discount_rate
            )
            
            # Confidence level (based on data quality and uncertainty)
            confidence_level = self._assess_confidence_level(measure, sector)
            
            return CostBenefitAnalysis(
                measure_id=measure.id,
                implementation_cost=implementation_cost,
                annual_maintenance_cost=annual_maintenance_cost,
                avoided_damages=avoided_damages,
                co_benefits_value=co_benefits_value,
                net_present_value=net_present_value,
                benefit_cost_ratio=benefit_cost_ratio,
                payback_period=payback_period,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Error conducting cost-benefit analysis: {str(e)}")
            return CostBenefitAnalysis(
                measure_id=measure.id,
                implementation_cost=0,
                annual_maintenance_cost=0,
                avoided_damages=0,
                co_benefits_value=0,
                net_present_value=0,
                benefit_cost_ratio=0,
                payback_period=float('inf'),
                confidence_level=0.5
            )
    
    def _calculate_avoided_damages(self, 
                                 measure: AdaptationMeasure,
                                 climate_risks: List[str],
                                 sector: str) -> float:
        """Calculate avoided damages from adaptation measure."""
        
        damage_factors = self.cost_benefit_parameters['damage_cost_factors']
        sector_damages = damage_factors.get(sector, {})
        
        total_avoided_damages = 0
        
        for risk in climate_risks:
            if risk in measure.climate_risks_addressed:
                base_damage = sector_damages.get(risk, 100)  # Default 100 million BDT
                avoided_damage = base_damage * measure.effectiveness
                total_avoided_damages += avoided_damage
        
        return total_avoided_damages
    
    def _calculate_co_benefits_value(self, measure: AdaptationMeasure) -> float:
        """Calculate monetary value of co-benefits."""
        
        co_benefit_values = self.cost_benefit_parameters['co_benefit_values']
        
        total_value = 0
        for co_benefit in measure.co_benefits:
            # Map co-benefits to value categories
            if 'biodiversity' in co_benefit or 'ecosystem' in co_benefit:
                total_value += co_benefit_values['biodiversity']
            elif 'carbon' in co_benefit:
                total_value += co_benefit_values['carbon_sequestration']
            elif 'water' in co_benefit:
                total_value += co_benefit_values['water_conservation']
            elif 'job' in co_benefit or 'employment' in co_benefit:
                total_value += co_benefit_values['job_creation']
            elif 'health' in co_benefit:
                total_value += co_benefit_values['health_benefits']
            else:
                total_value += 25  # Default value
        
        return total_value
    
    def _calculate_payback_period(self, 
                                initial_cost: float,
                                annual_benefits: float,
                                discount_rate: float) -> float:
        """Calculate payback period for adaptation measure."""
        
        if annual_benefits <= 0:
            return float('inf')
        
        # Simple payback period (without discounting)
        simple_payback = initial_cost / annual_benefits
        
        # Discounted payback period
        cumulative_benefits = 0
        year = 0
        
        while cumulative_benefits < initial_cost and year < 50:  # Max 50 years
            year += 1
            annual_discounted_benefit = annual_benefits / ((1 + discount_rate) ** year)
            cumulative_benefits += annual_discounted_benefit
        
        return min(simple_payback, year)
    
    def _assess_confidence_level(self, measure: AdaptationMeasure, sector: str) -> float:
        """Assess confidence level in cost-benefit analysis."""
        
        confidence = 0.7  # Base confidence
        
        # Adjust based on measure characteristics
        if measure.category == AdaptationCategory.INFRASTRUCTURE:
            confidence += 0.1  # Infrastructure measures are well understood
        elif measure.category == AdaptationCategory.ECOSYSTEM:
            confidence -= 0.1  # Ecosystem measures have more uncertainty
        
        # Adjust based on effectiveness
        if measure.effectiveness > 0.8:
            confidence += 0.1
        elif measure.effectiveness < 0.5:
            confidence -= 0.1
        
        # Adjust based on implementation timeframe
        if measure.implementation_timeframe == ImplementationTimeframe.IMMEDIATE:
            confidence += 0.1
        elif measure.implementation_timeframe == ImplementationTimeframe.LONG_TERM:
            confidence -= 0.1
        
        return max(0.3, min(0.9, confidence))
    
    def prioritize_adaptation_investments(self, 
                                        adaptation_plans: List[AdaptationPlan],
                                        total_budget: float) -> Dict[str, Any]:
        """Prioritize adaptation investments across multiple plans.
        
        Args:
            adaptation_plans: List of adaptation plans
            total_budget: Total available budget
            
        Returns:
            Investment prioritization results
        """
        try:
            # Collect all measures with their cost-benefit analysis
            all_measures = []
            
            for plan in adaptation_plans:
                for measure in plan.adaptation_measures:
                    cba = self.conduct_cost_benefit_analysis(
                        measure, plan.climate_risks, plan.sector
                    )
                    
                    all_measures.append({
                        'measure': measure,
                        'sector': plan.sector,
                        'cba': cba,
                        'priority_score': self._calculate_priority_score(measure, cba)
                    })
            
            # Sort by priority score
            all_measures.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Select measures within budget
            selected_measures = []
            remaining_budget = total_budget
            
            for measure_data in all_measures:
                measure = measure_data['measure']
                if measure.cost_estimate <= remaining_budget:
                    selected_measures.append(measure_data)
                    remaining_budget -= measure.cost_estimate
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(selected_measures)
            
            return {
                'selected_measures': selected_measures,
                'total_investment': total_budget - remaining_budget,
                'remaining_budget': remaining_budget,
                'portfolio_metrics': portfolio_metrics,
                'investment_summary': self._create_investment_summary(selected_measures)
            }
            
        except Exception as e:
            logger.error(f"Error prioritizing adaptation investments: {str(e)}")
            return {}
    
    def _calculate_priority_score(self, 
                                measure: AdaptationMeasure,
                                cba: CostBenefitAnalysis) -> float:
        """Calculate priority score for adaptation measure."""
        
        score = 0
        
        # Benefit-cost ratio (weight: 30%)
        bcr_score = min(cba.benefit_cost_ratio, 5) * 0.3  # Cap at 5
        score += bcr_score
        
        # Effectiveness (weight: 25%)
        effectiveness_score = measure.effectiveness * 0.25
        score += effectiveness_score
        
        # Priority level (weight: 20%)
        priority_scores = {
            Priority.CRITICAL: 1.0,
            Priority.HIGH: 0.8,
            Priority.MEDIUM: 0.6,
            Priority.LOW: 0.4
        }
        priority_score = priority_scores.get(measure.priority, 0.5) * 0.2
        score += priority_score
        
        # Implementation timeframe (weight: 15%)
        timeframe_scores = {
            ImplementationTimeframe.IMMEDIATE: 1.0,
            ImplementationTimeframe.SHORT_TERM: 0.8,
            ImplementationTimeframe.MEDIUM_TERM: 0.6,
            ImplementationTimeframe.LONG_TERM: 0.4
        }
        timeframe_score = timeframe_scores.get(measure.implementation_timeframe, 0.5) * 0.15
        score += timeframe_score
        
        # Confidence level (weight: 10%)
        confidence_score = cba.confidence_level * 0.1
        score += confidence_score
        
        return score
    
    def _calculate_portfolio_metrics(self, selected_measures: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics for the adaptation investment portfolio."""
        
        if not selected_measures:
            return {}
        
        total_cost = sum(m['measure'].cost_estimate for m in selected_measures)
        total_benefits = sum(m['cba'].net_present_value for m in selected_measures)
        
        # Sector distribution
        sector_distribution = {}
        for measure_data in selected_measures:
            sector = measure_data['sector']
            cost = measure_data['measure'].cost_estimate
            if sector not in sector_distribution:
                sector_distribution[sector] = 0
            sector_distribution[sector] += cost
        
        # Category distribution
        category_distribution = {}
        for measure_data in selected_measures:
            category = measure_data['measure'].category.value
            cost = measure_data['measure'].cost_estimate
            if category not in category_distribution:
                category_distribution[category] = 0
            category_distribution[category] += cost
        
        # Risk coverage
        covered_risks = set()
        for measure_data in selected_measures:
            covered_risks.update(measure_data['measure'].climate_risks_addressed)
        
        return {
            'total_measures': len(selected_measures),
            'total_cost': total_cost,
            'total_net_benefits': total_benefits,
            'portfolio_bcr': total_benefits / total_cost if total_cost > 0 else 0,
            'sector_distribution': sector_distribution,
            'category_distribution': category_distribution,
            'risk_coverage': list(covered_risks),
            'average_effectiveness': np.mean([m['measure'].effectiveness for m in selected_measures])
        }
    
    def _create_investment_summary(self, selected_measures: List[Dict]) -> Dict[str, Any]:
        """Create investment summary."""
        
        summary = {
            'by_timeframe': {},
            'by_priority': {},
            'top_measures': [],
            'key_benefits': []
        }
        
        # Group by timeframe
        for measure_data in selected_measures:
            timeframe = measure_data['measure'].implementation_timeframe.value
            if timeframe not in summary['by_timeframe']:
                summary['by_timeframe'][timeframe] = {'count': 0, 'cost': 0}
            summary['by_timeframe'][timeframe]['count'] += 1
            summary['by_timeframe'][timeframe]['cost'] += measure_data['measure'].cost_estimate
        
        # Group by priority
        for measure_data in selected_measures:
            priority = measure_data['measure'].priority.value
            if priority not in summary['by_priority']:
                summary['by_priority'][priority] = {'count': 0, 'cost': 0}
            summary['by_priority'][priority]['count'] += 1
            summary['by_priority'][priority]['cost'] += measure_data['measure'].cost_estimate
        
        # Top measures by priority score
        top_measures = sorted(selected_measures, key=lambda x: x['priority_score'], reverse=True)[:5]
        summary['top_measures'] = [
            {
                'name': m['measure'].name,
                'sector': m['sector'],
                'cost': m['measure'].cost_estimate,
                'bcr': m['cba'].benefit_cost_ratio,
                'priority_score': m['priority_score']
            }
            for m in top_measures
        ]
        
        # Key benefits
        all_co_benefits = []
        for measure_data in selected_measures:
            all_co_benefits.extend(measure_data['measure'].co_benefits)
        
        from collections import Counter
        benefit_counts = Counter(all_co_benefits)
        summary['key_benefits'] = [benefit for benefit, count in benefit_counts.most_common(10)]
        
        return summary
    
    def generate_adaptation_report(self, 
                                 adaptation_plans: List[AdaptationPlan],
                                 investment_prioritization: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive adaptation report.
        
        Args:
            adaptation_plans: List of adaptation plans
            investment_prioritization: Investment prioritization results
            
        Returns:
            Comprehensive adaptation report
        """
        
        report = {
            'executive_summary': {},
            'sector_analysis': {},
            'investment_analysis': {},
            'implementation_roadmap': {},
            'monitoring_framework': {},
            'recommendations': []
        }
        
        # Executive summary
        total_measures = sum(len(plan.adaptation_measures) for plan in adaptation_plans)
        total_cost = sum(plan.total_cost for plan in adaptation_plans)
        covered_sectors = [plan.sector for plan in adaptation_plans]
        
        report['executive_summary'] = {
            'total_adaptation_plans': len(adaptation_plans),
            'total_measures': total_measures,
            'total_investment_required': total_cost,
            'covered_sectors': covered_sectors,
            'priority_timeframes': self._analyze_priority_timeframes(adaptation_plans)
        }
        
        # Sector analysis
        for plan in adaptation_plans:
            report['sector_analysis'][plan.sector] = {
                'climate_risks': plan.climate_risks,
                'adaptation_measures': len(plan.adaptation_measures),
                'investment_required': plan.total_cost,
                'expected_benefits': plan.expected_benefits,
                'implementation_timeline': plan.implementation_timeline,
                'key_measures': [m.name for m in plan.adaptation_measures[:3]]  # Top 3
            }
        
        # Investment analysis
        if investment_prioritization:
            report['investment_analysis'] = investment_prioritization.get('portfolio_metrics', {})
            report['investment_analysis']['investment_summary'] = investment_prioritization.get('investment_summary', {})
        
        # Implementation roadmap
        report['implementation_roadmap'] = self._create_implementation_roadmap(adaptation_plans)
        
        # Monitoring framework
        report['monitoring_framework'] = self._create_monitoring_framework(adaptation_plans)
        
        # Recommendations
        report['recommendations'] = self._generate_adaptation_recommendations(adaptation_plans)
        
        return report
    
    def _analyze_priority_timeframes(self, adaptation_plans: List[AdaptationPlan]) -> Dict[str, int]:
        """Analyze priority timeframes across adaptation plans."""
        
        timeframe_counts = {
            'immediate': 0,
            'short_term': 0,
            'medium_term': 0,
            'long_term': 0
        }
        
        for plan in adaptation_plans:
            for timeframe, measures in plan.implementation_timeline.items():
                timeframe_counts[timeframe] += len(measures)
        
        return timeframe_counts
    
    def _create_implementation_roadmap(self, adaptation_plans: List[AdaptationPlan]) -> Dict[str, Any]:
        """Create implementation roadmap."""
        
        roadmap = {
            'phase_1_immediate': {'measures': [], 'cost': 0, 'duration': '0-1 years'},
            'phase_2_short_term': {'measures': [], 'cost': 0, 'duration': '1-3 years'},
            'phase_3_medium_term': {'measures': [], 'cost': 0, 'duration': '3-7 years'},
            'phase_4_long_term': {'measures': [], 'cost': 0, 'duration': '7+ years'}
        }
        
        phase_mapping = {
            'immediate': 'phase_1_immediate',
            'short_term': 'phase_2_short_term',
            'medium_term': 'phase_3_medium_term',
            'long_term': 'phase_4_long_term'
        }
        
        for plan in adaptation_plans:
            for measure in plan.adaptation_measures:
                timeframe = measure.implementation_timeframe.value
                phase = phase_mapping.get(timeframe, 'phase_3_medium_term')
                
                roadmap[phase]['measures'].append({
                    'name': measure.name,
                    'sector': plan.sector,
                    'cost': measure.cost_estimate,
                    'priority': measure.priority.value
                })
                roadmap[phase]['cost'] += measure.cost_estimate
        
        return roadmap
    
    def _create_monitoring_framework(self, adaptation_plans: List[AdaptationPlan]) -> Dict[str, Any]:
        """Create monitoring framework."""
        
        all_indicators = set()
        for plan in adaptation_plans:
            all_indicators.update(plan.monitoring_indicators)
        
        framework = {
            'key_indicators': list(all_indicators),
            'monitoring_frequency': {
                'quarterly': ['implementation_progress', 'budget_utilization'],
                'annually': ['effectiveness_assessment', 'impact_evaluation'],
                'biannually': ['stakeholder_satisfaction', 'adaptive_management']
            },
            'reporting_structure': {
                'implementation_reports': 'quarterly',
                'progress_assessments': 'annually',
                'comprehensive_reviews': 'every_3_years'
            },
            'evaluation_criteria': [
                'measure_implementation_status',
                'target_achievement',
                'cost_effectiveness',
                'co_benefits_realization',
                'stakeholder_engagement'
            ]
        }
        
        return framework
    
    def _generate_adaptation_recommendations(self, adaptation_plans: List[AdaptationPlan]) -> List[str]:
        """Generate adaptation recommendations."""
        
        recommendations = [
            "Prioritize immediate and short-term measures to build early resilience",
            "Establish strong institutional coordination mechanisms",
            "Develop sustainable financing mechanisms for long-term implementation",
            "Build local capacity for adaptation measure implementation",
            "Integrate adaptation measures with development planning",
            "Strengthen monitoring and evaluation systems",
            "Promote community participation in adaptation planning",
            "Develop climate-resilient infrastructure standards",
            "Enhance early warning and disaster preparedness systems",
            "Foster innovation and technology transfer for adaptation"
        ]
        
        # Add sector-specific recommendations
        sectors = [plan.sector for plan in adaptation_plans]
        if 'agriculture' in sectors:
            recommendations.append("Accelerate development and deployment of climate-resilient crop varieties")
        if 'manufacturing' in sectors:
            recommendations.append("Implement energy-efficient climate control systems in industrial facilities")
        if 'coastal_communities' in sectors:
            recommendations.append("Strengthen coastal protection through ecosystem-based approaches")
        
        return recommendations
    
    def export_adaptation_analysis(self, 
                                 adaptation_report: Dict[str, Any],
                                 output_path: str,
                                 format: str = 'json') -> bool:
        """Export adaptation analysis results.
        
        Args:
            adaptation_report: Adaptation analysis report
            output_path: Output file path
            format: Export format ('json', 'xlsx')
            
        Returns:
            Success status
        """
        try:
            from pathlib import Path
            import json
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(adaptation_report, f, indent=2, default=str)
            
            elif format.lower() == 'xlsx':
                # Export to Excel with multiple sheets
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Executive summary
                    exec_summary_df = pd.DataFrame([adaptation_report['executive_summary']])
                    exec_summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
                    
                    # Sector analysis
                    if adaptation_report.get('sector_analysis'):
                        sector_data = []
                        for sector, data in adaptation_report['sector_analysis'].items():
                            row = {'sector': sector}
                            row.update(data)
                            sector_data.append(row)
                        
                        sector_df = pd.DataFrame(sector_data)
                        sector_df.to_excel(writer, sheet_name='Sector_Analysis', index=False)
                    
                    # Implementation roadmap
                    if adaptation_report.get('implementation_roadmap'):
                        roadmap_data = []
                        for phase, data in adaptation_report['implementation_roadmap'].items():
                            for measure in data.get('measures', []):
                                row = {'phase': phase, 'duration': data.get('duration', '')}
                                row.update(measure)
                                roadmap_data.append(row)
                        
                        if roadmap_data:
                            roadmap_df = pd.DataFrame(roadmap_data)
                            roadmap_df.to_excel(writer, sheet_name='Implementation_Roadmap', index=False)
            
            logger.info(f"Adaptation analysis exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting adaptation analysis: {str(e)}")
            return False


def create_sample_adaptation_scenario():
    """Create sample adaptation scenario for demonstration."""
    
    # Initialize adaptation manager
    manager = ClimateAdaptationManager()
    
    # Create adaptation plans for different sectors
    agriculture_plan = manager.create_adaptation_plan(
        sector='agriculture',
        climate_risks=['drought', 'flooding', 'temperature_increase'],
        region='northern',
        budget=3000  # 3 billion BDT
    )
    
    manufacturing_plan = manager.create_adaptation_plan(
        sector='manufacturing',
        climate_risks=['heat_waves', 'flooding', 'power_outages'],
        region='urban_areas',
        budget=2000  # 2 billion BDT
    )
    
    coastal_plan = manager.create_adaptation_plan(
        sector='coastal_communities',
        climate_risks=['sea_level_rise', 'cyclones', 'storm_surge'],
        region='coastal',
        budget=4000  # 4 billion BDT
    )
    
    # Prioritize investments
    all_plans = [agriculture_plan, manufacturing_plan, coastal_plan]
    investment_prioritization = manager.prioritize_adaptation_investments(
        all_plans, total_budget=8000  # 8 billion BDT
    )
    
    # Generate comprehensive report
    adaptation_report = manager.generate_adaptation_report(
        all_plans, investment_prioritization
    )
    
    return {
        'adaptation_plans': all_plans,
        'investment_prioritization': investment_prioritization,
        'adaptation_report': adaptation_report
    }


if __name__ == "__main__":
    # Create sample scenario
    scenario = create_sample_adaptation_scenario()
    
    print("\n=== Bangladesh Climate Adaptation Analysis ===")
    print(f"Total adaptation plans: {len(scenario['adaptation_plans'])}")
    print(f"Total investment required: {sum(plan.total_cost for plan in scenario['adaptation_plans']):.0f} million BDT")
    
    # Print sector summaries
    for plan in scenario['adaptation_plans']:
        print(f"\n{plan.sector.upper()} SECTOR:")
        print(f"  Climate risks: {', '.join(plan.climate_risks)}")
        print(f"  Adaptation measures: {len(plan.adaptation_measures)}")
        print(f"  Investment required: {plan.total_cost:.0f} million BDT")
        print(f"  Expected benefits: {plan.expected_benefits.get('total_benefits', 0):.0f} million BDT")
    
    # Print investment prioritization
    if scenario['investment_prioritization']:
        print("\n=== INVESTMENT PRIORITIZATION ===")
        selected = scenario['investment_prioritization']['selected_measures']
        print(f"Selected measures: {len(selected)}")
        print(f"Total investment: {scenario['investment_prioritization']['total_investment']:.0f} million BDT")
        
        print("\nTop priority measures:")
        for i, measure_data in enumerate(selected[:5], 1):
            measure = measure_data['measure']
            print(f"  {i}. {measure.name} ({measure_data['sector']}) - {measure.cost_estimate:.0f} million BDT")