"""Informal Economy Sector Model for Bangladesh GDP Simulation.

This module models Bangladesh's informal economy, which represents a significant
portion of economic activity. It includes street vendors, small-scale manufacturing,
domestic workers, rickshaw pullers, and other unregistered economic activities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class InformalEconomySector:
    """Model for Bangladesh's informal economy sector.
    
    This class simulates informal economic activity considering:
    - Street vendors and small traders
    - Informal manufacturing and handicrafts
    - Domestic and care workers
    - Transport services (rickshaw, CNG)
    - Construction day laborers
    - Agricultural informal workers
    """
    
    def __init__(self):
        """Initialize informal economy sector model."""
        
        # Informal economy size (estimated 35-40% of total economy)
        self.informal_economy_share = 0.38  # 38% of total GDP
        
        # Subsector shares within informal economy
        self.subsector_shares = {
            'trade_services': 0.35,         # Street vendors, small shops
            'manufacturing_handicrafts': 0.20, # Small-scale manufacturing
            'transport_services': 0.15,     # Rickshaw, CNG, small transport
            'domestic_care_services': 0.12, # Domestic workers, care services
            'construction_labor': 0.10,     # Day laborers, informal construction
            'agriculture_informal': 0.08    # Informal agricultural activities
        }
        
        # Trade and services parameters
        self.trade_services_parameters = {
            'urban_concentration': 0.75,     # 75% in urban areas
            'income_elasticity': 0.8,        # Response to income changes
            'competition_from_formal': -0.3, # Negative impact from formal sector growth
            'mobile_payment_adoption': 0.25, # Mobile payment usage
            'seasonal_variation': 0.15,      # Seasonal business variation
            'covid_recovery_factor': 0.85    # Still recovering from COVID impact
        }
        
        # Manufacturing and handicrafts parameters
        self.manufacturing_parameters = {
            'export_linkage': 0.40,          # 40% linked to export markets
            'skill_intensity': 0.60,         # Moderate skill requirements
            'technology_adoption': 0.20,     # Low technology adoption
            'raw_material_dependency': 0.70, # High dependency on raw materials
            'market_access_constraint': 0.45, # Limited market access
            'quality_standards_compliance': 0.30 # Low compliance with standards
        }
        
        # Transport services parameters
        self.transport_parameters = {
            'urbanization_elasticity': 1.2,  # High response to urbanization
            'fuel_price_elasticity': -0.6,   # Negative response to fuel prices
            'formal_transport_competition': -0.4, # Competition from formal transport
            'ride_sharing_impact': -0.2,     # Negative impact from ride-sharing
            'infrastructure_improvement_impact': 0.3, # Positive from better roads
            'electric_vehicle_transition': 0.05 # Slow EV adoption
        }
        
        # Domestic and care services parameters
        self.domestic_services_parameters = {
            'middle_class_growth_elasticity': 1.5, # High response to middle class growth
            'female_labor_participation': 0.85,    # High female participation
            'wage_growth_rate': 0.06,              # 6% annual wage growth
            'formalization_pressure': -0.1,        # Pressure to formalize
            'urban_demand_concentration': 0.80,     # 80% demand in urban areas
            'skill_development_impact': 0.15       # Impact of skill development
        }
        
        # Construction labor parameters
        self.construction_labor_parameters = {
            'construction_sector_linkage': 0.9,  # High linkage to formal construction
            'seasonal_migration': 0.40,          # 40% seasonal migrants
            'skill_level': 0.35,                 # Low to moderate skill level
            'wage_volatility': 0.25,             # High wage volatility
            'safety_standards': 0.20,            # Low safety standards
            'social_protection': 0.15            # Limited social protection
        }
        
        # Agriculture informal parameters
        self.agriculture_informal_parameters = {
            'subsistence_farming_share': 0.60,   # 60% subsistence farming
            'seasonal_labor_share': 0.30,       # 30% seasonal agricultural labor
            'climate_vulnerability': 0.80,       # High climate vulnerability
            'market_integration': 0.40,          # Limited market integration
            'technology_access': 0.25,           # Limited technology access
            'credit_access': 0.30               # Limited credit access
        }
        
        # Digitalization impact
        self.digitalization_parameters = {
            'mobile_phone_penetration': 0.95,    # High mobile penetration
            'internet_access': 0.65,             # Moderate internet access
            'digital_payment_adoption': 0.35,    # Growing digital payment adoption
            'e_commerce_participation': 0.20,    # Limited e-commerce participation
            'digital_skills': 0.25,              # Limited digital skills
            'platform_economy_growth': 0.15     # Growing platform economy
        }
        
        # Formalization pressures
        self.formalization_factors = {
            'tax_compliance_pressure': 0.15,     # Low tax compliance pressure
            'regulatory_enforcement': 0.25,      # Limited regulatory enforcement
            'access_to_credit_incentive': 0.40,  # Credit access incentive
            'social_protection_incentive': 0.35, # Social protection incentive
            'market_access_incentive': 0.50,     # Market access incentive
            'cost_of_formalization': 0.60       # High cost of formalization
        }
        
        # Seasonal patterns
        self.seasonal_patterns = {
            'trade_services': {1: 1.05, 2: 0.95, 3: 1.10, 4: 1.15},  # Peak in Q3-Q4
            'manufacturing_handicrafts': {1: 1.00, 2: 0.90, 3: 1.05, 4: 1.20}, # Peak in Q4
            'transport_services': {1: 1.02, 2: 0.85, 3: 1.08, 4: 1.10}, # Monsoon impact
            'domestic_care_services': {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.05}, # Stable
            'construction_labor': {1: 1.20, 2: 0.75, 3: 1.10, 4: 1.05}, # Winter peak
            'agriculture_informal': {1: 0.80, 2: 1.30, 3: 1.10, 4: 0.90} # Harvest seasons
        }
        
        logger.info("Informal economy sector model initialized")
    
    def calculate_production(self, 
                           base_year_value: float,
                           quarter: int,
                           year: int,
                           economic_conditions: Dict,
                           policy_factors: Dict = None) -> Dict:
        """Calculate quarterly informal economy production.
        
        Args:
            base_year_value: Base year informal economy GDP (billion BDT)
            quarter: Quarter (1-4)
            year: Year
            economic_conditions: Economic conditions affecting informal economy
            policy_factors: Policy impacts on informal economy
            
        Returns:
            Dictionary with production estimates by subsector
        """
        if policy_factors is None:
            policy_factors = {}
        
        # Calculate subsector production
        subsector_production = {}
        
        # Trade and services
        trade_production = self._calculate_trade_services(
            base_year_value * self.subsector_shares['trade_services'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['trade_services'] = trade_production
        
        # Manufacturing and handicrafts
        manufacturing_production = self._calculate_manufacturing_handicrafts(
            base_year_value * self.subsector_shares['manufacturing_handicrafts'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['manufacturing_handicrafts'] = manufacturing_production
        
        # Transport services
        transport_production = self._calculate_transport_services(
            base_year_value * self.subsector_shares['transport_services'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['transport_services'] = transport_production
        
        # Domestic and care services
        domestic_production = self._calculate_domestic_care_services(
            base_year_value * self.subsector_shares['domestic_care_services'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['domestic_care_services'] = domestic_production
        
        # Construction labor
        construction_labor_production = self._calculate_construction_labor(
            base_year_value * self.subsector_shares['construction_labor'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['construction_labor'] = construction_labor_production
        
        # Agriculture informal
        agriculture_production = self._calculate_agriculture_informal(
            base_year_value * self.subsector_shares['agriculture_informal'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['agriculture_informal'] = agriculture_production
        
        # Total informal economy production
        total_production = sum(subsector_production.values())
        
        return {
            'total_informal_gdp': total_production,
            'subsector_breakdown': subsector_production,
            'formalization_impact': self._calculate_formalization_impact(subsector_production, policy_factors),
            'digitalization_impact': self._calculate_digitalization_impact(subsector_production, economic_conditions),
            'vulnerability_assessment': self._assess_vulnerability(subsector_production, economic_conditions),
            'policy_recommendations': self._generate_policy_recommendations(economic_conditions, policy_factors)
        }
    
    def _calculate_trade_services(self, 
                                base_value: float,
                                quarter: int,
                                year: int,
                                economic_conditions: Dict,
                                policy_factors: Dict) -> float:
        """Calculate informal trade and services activity."""
        
        # Moderate growth rate
        base_growth = 0.045  # 4.5% annual growth
        
        # Strong seasonal pattern
        seasonal_factor = self.seasonal_patterns['trade_services'].get(quarter, 1.0)
        
        # Income elasticity impact
        per_capita_income_growth = economic_conditions.get('per_capita_income_growth', 0.04)
        income_factor = 1 + per_capita_income_growth * self.trade_services_parameters['income_elasticity']
        
        # Urban population growth impact
        urban_population_growth = economic_conditions.get('urban_population_growth', 0.03)
        urban_factor = 1 + urban_population_growth * 1.5  # High urban elasticity
        
        # Competition from formal sector
        formal_retail_growth = economic_conditions.get('formal_retail_growth', 0.08)
        competition_factor = 1 + formal_retail_growth * self.trade_services_parameters['competition_from_formal']
        
        # Mobile payment adoption impact
        mobile_payment_growth = economic_conditions.get('mobile_payment_growth', 0.20)
        digital_factor = 1 + mobile_payment_growth * 0.3  # Positive impact
        
        # COVID recovery factor
        covid_recovery = self.trade_services_parameters['covid_recovery_factor']
        if year <= 2025:
            covid_factor = covid_recovery + (year - 2024) * 0.05  # Gradual recovery
        else:
            covid_factor = 1.0
        
        # Policy impact
        policy_impact = 1.0
        if 'informal_sector_support' in policy_factors:
            policy_impact *= (1 + policy_factors['informal_sector_support'] * 0.12)
        if 'digital_inclusion_program' in policy_factors:
            policy_impact *= (1 + policy_factors['digital_inclusion_program'] * 0.08)
        if 'microfinance_expansion' in policy_factors:
            policy_impact *= (1 + policy_factors['microfinance_expansion'] * 0.10)
        
        # Inflation impact (negative for low-income consumers)
        inflation_rate = economic_conditions.get('inflation_rate', 0.06)
        inflation_factor = 1 - (inflation_rate - 0.05) * 0.5 if inflation_rate > 0.05 else 1.0
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            income_factor *
            urban_factor *
            competition_factor *
            digital_factor *
            covid_factor *
            policy_impact *
            inflation_factor
        )
        
        return production
    
    def _calculate_manufacturing_handicrafts(self, 
                                           base_value: float,
                                           quarter: int,
                                           year: int,
                                           economic_conditions: Dict,
                                           policy_factors: Dict) -> float:
        """Calculate informal manufacturing and handicrafts activity."""
        
        # Moderate growth rate
        base_growth = 0.055  # 5.5% annual growth
        
        # Seasonal pattern (peak in Q4 for exports)
        seasonal_factor = self.seasonal_patterns['manufacturing_handicrafts'].get(quarter, 1.0)
        
        # Export linkage impact
        export_growth = economic_conditions.get('handicraft_export_growth', 0.06)
        export_factor = 1 + export_growth * self.manufacturing_parameters['export_linkage']
        
        # Raw material cost impact
        raw_material_price_change = economic_conditions.get('raw_material_price_change', 0.05)
        material_factor = 1 - raw_material_price_change * 0.4  # Negative impact
        
        # Skill development impact
        skill_development = economic_conditions.get('skill_development_programs', 0.02)
        skill_factor = 1 + skill_development * 2.0  # High impact on productivity
        
        # Market access improvement
        market_access_improvement = economic_conditions.get('market_access_improvement', 0.03)
        market_factor = 1 + market_access_improvement * 1.5
        
        # Technology adoption (limited but growing)
        technology_factor = 1 + (year - 2024) * 0.02  # 2% annual technology improvement
        
        # Policy impact
        policy_impact = 1.0
        if 'handicraft_promotion' in policy_factors:
            policy_impact *= (1 + policy_factors['handicraft_promotion'] * 0.15)
        if 'skill_development_informal' in policy_factors:
            policy_impact *= (1 + policy_factors['skill_development_informal'] * 0.12)
        if 'market_linkage_programs' in policy_factors:
            policy_impact *= (1 + policy_factors['market_linkage_programs'] * 0.10)
        
        # Quality standards pressure
        quality_standards_factor = 1.0
        if economic_conditions.get('quality_standards_enforcement', 0) > 0:
            quality_standards_factor *= (1 - economic_conditions['quality_standards_enforcement'] * 0.1)
        
        # E-commerce participation
        ecommerce_growth = economic_conditions.get('ecommerce_growth', 0.15)
        ecommerce_factor = 1 + ecommerce_growth * 0.2  # Limited but growing participation
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            export_factor *
            material_factor *
            skill_factor *
            market_factor *
            technology_factor *
            policy_impact *
            quality_standards_factor *
            ecommerce_factor
        )
        
        return production
    
    def _calculate_transport_services(self, 
                                    base_value: float,
                                    quarter: int,
                                    year: int,
                                    economic_conditions: Dict,
                                    policy_factors: Dict) -> float:
        """Calculate informal transport services activity."""
        
        # Slow growth rate (facing competition)
        base_growth = 0.025  # 2.5% annual growth
        
        # Seasonal pattern (monsoon impact)
        seasonal_factor = self.seasonal_patterns['transport_services'].get(quarter, 1.0)
        
        # Urbanization impact (positive)
        urbanization_rate = economic_conditions.get('urbanization_rate', 0.38)
        urban_growth = economic_conditions.get('urban_population_growth', 0.03)
        urbanization_factor = 1 + urban_growth * self.transport_parameters['urbanization_elasticity']
        
        # Fuel price impact (negative)
        fuel_price_change = economic_conditions.get('fuel_price_change', 0.06)
        fuel_factor = 1 + fuel_price_change * self.transport_parameters['fuel_price_elasticity']
        
        # Competition from formal transport
        formal_transport_growth = economic_conditions.get('formal_transport_growth', 0.10)
        competition_factor = 1 + formal_transport_growth * self.transport_parameters['formal_transport_competition']
        
        # Ride-sharing impact
        ride_sharing_growth = economic_conditions.get('ride_sharing_growth', 0.25)
        ride_sharing_factor = 1 + ride_sharing_growth * self.transport_parameters['ride_sharing_impact']
        
        # Infrastructure improvement impact
        infrastructure_improvement = economic_conditions.get('transport_infrastructure_improvement', 0.05)
        infrastructure_factor = 1 + infrastructure_improvement * self.transport_parameters['infrastructure_improvement_impact']
        
        # Electric vehicle transition (slow)
        ev_adoption = economic_conditions.get('ev_adoption_informal_transport', 0.02)
        ev_factor = 1 + ev_adoption * 0.5  # Slight positive impact
        
        # Policy impact
        policy_impact = 1.0
        if 'informal_transport_regulation' in policy_factors:
            policy_impact *= (1 - policy_factors['informal_transport_regulation'] * 0.08)  # Negative impact
        if 'transport_worker_support' in policy_factors:
            policy_impact *= (1 + policy_factors['transport_worker_support'] * 0.06)
        if 'fuel_subsidy_informal_transport' in policy_factors:
            policy_impact *= (1 + policy_factors['fuel_subsidy_informal_transport'] * 0.10)
        
        # Economic activity linkage
        economic_activity_growth = economic_conditions.get('economic_activity_growth', 0.06)
        activity_factor = 1 + economic_activity_growth * 0.8  # Transport follows economic activity
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            urbanization_factor *
            fuel_factor *
            competition_factor *
            ride_sharing_factor *
            infrastructure_factor *
            ev_factor *
            policy_impact *
            activity_factor
        )
        
        return production
    
    def _calculate_domestic_care_services(self, 
                                        base_value: float,
                                        quarter: int,
                                        year: int,
                                        economic_conditions: Dict,
                                        policy_factors: Dict) -> float:
        """Calculate domestic and care services activity."""
        
        # High growth rate (growing middle class demand)
        base_growth = 0.08  # 8% annual growth
        
        # Stable seasonal pattern
        seasonal_factor = self.seasonal_patterns['domestic_care_services'].get(quarter, 1.0)
        
        # Middle class growth impact (major driver)
        middle_class_growth = economic_conditions.get('middle_class_growth', 0.05)
        middle_class_factor = 1 + middle_class_growth * self.domestic_services_parameters['middle_class_growth_elasticity']
        
        # Female labor force participation impact
        female_lfp_growth = economic_conditions.get('female_labor_participation_growth', 0.02)
        female_lfp_factor = 1 + female_lfp_growth * 2.0  # High elasticity
        
        # Urban household income growth
        urban_income_growth = economic_conditions.get('urban_household_income_growth', 0.05)
        urban_income_factor = 1 + urban_income_growth * 1.2
        
        # Aging population impact
        aging_population_factor = 1 + (year - 2024) * 0.015  # 1.5% annual increase due to aging
        
        # Formalization pressure (negative)
        formalization_pressure = economic_conditions.get('domestic_worker_formalization_pressure', 0.02)
        formalization_factor = 1 - formalization_pressure * 0.5
        
        # Policy impact
        policy_impact = 1.0
        if 'domestic_worker_protection' in policy_factors:
            policy_impact *= (1 + policy_factors['domestic_worker_protection'] * 0.08)
        if 'care_services_support' in policy_factors:
            policy_impact *= (1 + policy_factors['care_services_support'] * 0.10)
        if 'skill_training_domestic_workers' in policy_factors:
            policy_impact *= (1 + policy_factors['skill_training_domestic_workers'] * 0.12)
        
        # COVID impact (increased demand for cleaning services)
        covid_impact_factor = 1.05 if year <= 2026 else 1.0  # Sustained higher demand
        
        # Wage growth impact
        wage_growth = self.domestic_services_parameters['wage_growth_rate']
        wage_factor = 1 + wage_growth * 0.3  # Positive supply response
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            middle_class_factor *
            female_lfp_factor *
            urban_income_factor *
            aging_population_factor *
            formalization_factor *
            policy_impact *
            covid_impact_factor *
            wage_factor
        )
        
        return production
    
    def _calculate_construction_labor(self, 
                                    base_value: float,
                                    quarter: int,
                                    year: int,
                                    economic_conditions: Dict,
                                    policy_factors: Dict) -> float:
        """Calculate informal construction labor activity."""
        
        # High growth rate (linked to construction boom)
        base_growth = 0.09  # 9% annual growth
        
        # Strong seasonal pattern
        seasonal_factor = self.seasonal_patterns['construction_labor'].get(quarter, 1.0)
        
        # Construction sector linkage (major driver)
        construction_growth = economic_conditions.get('construction_growth', 0.10)
        construction_factor = 1 + construction_growth * self.construction_labor_parameters['construction_sector_linkage']
        
        # Infrastructure investment impact
        infrastructure_investment_growth = economic_conditions.get('infrastructure_investment_growth', 0.12)
        infrastructure_factor = 1 + infrastructure_investment_growth * 0.8
        
        # Rural-urban migration impact
        migration_rate = economic_conditions.get('rural_urban_migration_rate', 0.025)
        migration_factor = 1 + migration_rate * 2.0  # High elasticity
        
        # Wage level impact
        construction_wage_growth = economic_conditions.get('construction_wage_growth', 0.08)
        wage_factor = 1 + construction_wage_growth * 0.6  # Positive supply response
        
        # Mechanization impact (negative)
        mechanization_rate = economic_conditions.get('construction_mechanization_rate', 0.03)
        mechanization_factor = 1 - mechanization_rate * 0.4
        
        # Policy impact
        policy_impact = 1.0
        if 'construction_worker_protection' in policy_factors:
            policy_impact *= (1 + policy_factors['construction_worker_protection'] * 0.06)
        if 'skill_development_construction' in policy_factors:
            policy_impact *= (1 + policy_factors['skill_development_construction'] * 0.10)
        if 'safety_standards_enforcement' in policy_factors:
            policy_impact *= (1 - policy_factors['safety_standards_enforcement'] * 0.05)  # Short-term negative
        
        # Seasonal migration factor
        seasonal_migration_factor = 1.0
        if quarter in [1, 4]:  # Peak construction seasons
            seasonal_migration_factor *= 1.15
        elif quarter == 2:  # Monsoon - return to agriculture
            seasonal_migration_factor *= 0.80
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            construction_factor *
            infrastructure_factor *
            migration_factor *
            wage_factor *
            mechanization_factor *
            policy_impact *
            seasonal_migration_factor
        )
        
        return production
    
    def _calculate_agriculture_informal(self, 
                                      base_value: float,
                                      quarter: int,
                                      year: int,
                                      economic_conditions: Dict,
                                      policy_factors: Dict) -> float:
        """Calculate informal agriculture activity."""
        
        # Low growth rate (declining share)
        base_growth = 0.015  # 1.5% annual growth
        
        # Strong seasonal pattern
        seasonal_factor = self.seasonal_patterns['agriculture_informal'].get(quarter, 1.0)
        
        # Climate impact (major factor)
        climate_impact = economic_conditions.get('climate_impact_agriculture', 0.0)
        climate_factor = 1 + climate_impact  # Can be positive or negative
        
        # Subsistence farming pressure
        food_price_inflation = economic_conditions.get('food_price_inflation', 0.06)
        subsistence_factor = 1 + food_price_inflation * 0.3  # Higher food prices increase subsistence farming
        
        # Rural income opportunities
        rural_income_opportunities = economic_conditions.get('rural_income_opportunities', 0.02)
        opportunity_factor = 1 - rural_income_opportunities * 0.5  # Alternative opportunities reduce informal agriculture
        
        # Agricultural productivity improvement
        productivity_improvement = economic_conditions.get('agricultural_productivity_improvement', 0.03)
        productivity_factor = 1 + productivity_improvement * 0.8
        
        # Market integration improvement
        market_integration_improvement = economic_conditions.get('market_integration_improvement', 0.02)
        market_factor = 1 + market_integration_improvement * 1.0
        
        # Policy impact
        policy_impact = 1.0
        if 'smallholder_farmer_support' in policy_factors:
            policy_impact *= (1 + policy_factors['smallholder_farmer_support'] * 0.12)
        if 'agricultural_extension_services' in policy_factors:
            policy_impact *= (1 + policy_factors['agricultural_extension_services'] * 0.08)
        if 'rural_credit_access' in policy_factors:
            policy_impact *= (1 + policy_factors['rural_credit_access'] * 0.10)
        
        # Technology access (limited)
        technology_access_improvement = economic_conditions.get('rural_technology_access', 0.01)
        technology_factor = 1 + technology_access_improvement * 1.5
        
        # Climate adaptation measures
        climate_adaptation_factor = 1.0
        if 'climate_adaptation_agriculture' in policy_factors:
            climate_adaptation_factor *= (1 + policy_factors['climate_adaptation_agriculture'] * 0.08)
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            climate_factor *
            subsistence_factor *
            opportunity_factor *
            productivity_factor *
            market_factor *
            policy_impact *
            technology_factor *
            climate_adaptation_factor
        )
        
        return production
    
    def _calculate_formalization_impact(self, subsector_production: Dict, policy_factors: Dict) -> Dict:
        """Calculate impact of formalization on informal economy."""
        
        total_informal_production = sum(subsector_production.values())
        
        # Base formalization rate (annual)
        base_formalization_rate = 0.02  # 2% annual formalization
        
        # Policy-driven formalization
        policy_formalization_boost = 0
        if 'formalization_incentives' in policy_factors:
            policy_formalization_boost += policy_factors['formalization_incentives'] * 0.03
        if 'simplified_business_registration' in policy_factors:
            policy_formalization_boost += policy_factors['simplified_business_registration'] * 0.02
        if 'tax_incentives_small_business' in policy_factors:
            policy_formalization_boost += policy_factors['tax_incentives_small_business'] * 0.025
        
        total_formalization_rate = base_formalization_rate + policy_formalization_boost
        
        # Formalization by subsector (different rates)
        subsector_formalization_rates = {
            'trade_services': total_formalization_rate * 1.2,
            'manufacturing_handicrafts': total_formalization_rate * 1.5,
            'transport_services': total_formalization_rate * 0.8,
            'domestic_care_services': total_formalization_rate * 0.5,
            'construction_labor': total_formalization_rate * 0.6,
            'agriculture_informal': total_formalization_rate * 0.3
        }
        
        formalization_impact = {}
        total_formalized_value = 0
        
        for subsector, production in subsector_production.items():
            formalization_rate = subsector_formalization_rates.get(subsector, total_formalization_rate)
            formalized_value = production * formalization_rate
            formalization_impact[subsector] = {
                'formalization_rate': formalization_rate,
                'formalized_value': formalized_value,
                'remaining_informal': production - formalized_value
            }
            total_formalized_value += formalized_value
        
        return {
            'total_formalization_rate': total_formalization_rate,
            'total_formalized_value': total_formalized_value,
            'subsector_formalization': formalization_impact,
            'formalization_benefits': self._calculate_formalization_benefits(total_formalized_value),
            'formalization_challenges': self._identify_formalization_challenges()
        }
    
    def _calculate_digitalization_impact(self, subsector_production: Dict, economic_conditions: Dict) -> Dict:
        """Calculate impact of digitalization on informal economy."""
        
        # Digital adoption rates by subsector
        digital_adoption_rates = {
            'trade_services': 0.35,
            'manufacturing_handicrafts': 0.20,
            'transport_services': 0.45,
            'domestic_care_services': 0.15,
            'construction_labor': 0.10,
            'agriculture_informal': 0.08
        }
        
        # Digital impact factors
        mobile_payment_growth = economic_conditions.get('mobile_payment_growth', 0.20)
        internet_penetration_growth = economic_conditions.get('internet_penetration_growth', 0.08)
        smartphone_adoption_growth = economic_conditions.get('smartphone_adoption_growth', 0.12)
        
        digitalization_impact = {}
        total_digital_value = 0
        
        for subsector, production in subsector_production.items():
            adoption_rate = digital_adoption_rates.get(subsector, 0.20)
            
            # Calculate digital impact
            digital_boost = (
                mobile_payment_growth * 0.3 +
                internet_penetration_growth * 0.4 +
                smartphone_adoption_growth * 0.3
            )
            
            digital_value = production * adoption_rate * digital_boost
            digitalization_impact[subsector] = {
                'adoption_rate': adoption_rate,
                'digital_boost': digital_boost,
                'digital_value_added': digital_value
            }
            total_digital_value += digital_value
        
        return {
            'total_digital_impact': total_digital_value,
            'subsector_digital_impact': digitalization_impact,
            'digital_inclusion_score': self._calculate_digital_inclusion_score(economic_conditions),
            'digital_barriers': self._identify_digital_barriers(),
            'digital_opportunities': self._identify_digital_opportunities()
        }
    
    def _assess_vulnerability(self, subsector_production: Dict, economic_conditions: Dict) -> Dict:
        """Assess vulnerability of informal economy sectors."""
        
        # Vulnerability factors
        vulnerability_factors = {
            'trade_services': {
                'income_volatility': 0.7,
                'competition_vulnerability': 0.6,
                'economic_shock_sensitivity': 0.8,
                'social_protection_access': 0.2
            },
            'manufacturing_handicrafts': {
                'income_volatility': 0.6,
                'competition_vulnerability': 0.5,
                'economic_shock_sensitivity': 0.7,
                'social_protection_access': 0.3
            },
            'transport_services': {
                'income_volatility': 0.8,
                'competition_vulnerability': 0.7,
                'economic_shock_sensitivity': 0.6,
                'social_protection_access': 0.2
            },
            'domestic_care_services': {
                'income_volatility': 0.5,
                'competition_vulnerability': 0.3,
                'economic_shock_sensitivity': 0.6,
                'social_protection_access': 0.1
            },
            'construction_labor': {
                'income_volatility': 0.9,
                'competition_vulnerability': 0.4,
                'economic_shock_sensitivity': 0.8,
                'social_protection_access': 0.1
            },
            'agriculture_informal': {
                'income_volatility': 0.8,
                'competition_vulnerability': 0.3,
                'economic_shock_sensitivity': 0.9,
                'social_protection_access': 0.2
            }
        }
        
        # Calculate overall vulnerability scores
        vulnerability_assessment = {}
        
        for subsector, factors in vulnerability_factors.items():
            overall_vulnerability = sum(factors.values()) / len(factors)
            
            # Adjust for current economic conditions
            inflation_rate = economic_conditions.get('inflation_rate', 0.06)
            unemployment_rate = economic_conditions.get('unemployment_rate', 0.04)
            
            economic_stress_factor = (inflation_rate - 0.05) * 2 + (unemployment_rate - 0.03) * 3
            adjusted_vulnerability = min(1.0, overall_vulnerability + economic_stress_factor)
            
            vulnerability_assessment[subsector] = {
                'base_vulnerability': overall_vulnerability,
                'adjusted_vulnerability': adjusted_vulnerability,
                'vulnerability_factors': factors,
                'risk_level': self._categorize_risk_level(adjusted_vulnerability)
            }
        
        return {
            'subsector_vulnerability': vulnerability_assessment,
            'overall_vulnerability_score': sum(v['adjusted_vulnerability'] for v in vulnerability_assessment.values()) / len(vulnerability_assessment),
            'high_risk_sectors': [k for k, v in vulnerability_assessment.items() if v['risk_level'] == 'High'],
            'vulnerability_mitigation_strategies': self._suggest_vulnerability_mitigation()
        }
    
    def _generate_policy_recommendations(self, economic_conditions: Dict, policy_factors: Dict) -> List[str]:
        """Generate policy recommendations for informal economy."""
        recommendations = []
        
        # Based on economic conditions
        inflation_rate = economic_conditions.get('inflation_rate', 0.06)
        if inflation_rate > 0.08:
            recommendations.extend([
                'Implement targeted food subsidies for informal workers',
                'Provide inflation-indexed minimum wage for informal sectors',
                'Expand social safety net programs'
            ])
        
        unemployment_rate = economic_conditions.get('unemployment_rate', 0.04)
        if unemployment_rate > 0.05:
            recommendations.extend([
                'Create employment guarantee programs',
                'Expand skill development programs for informal workers',
                'Support micro-enterprise development'
            ])
        
        # Digital inclusion recommendations
        if economic_conditions.get('digital_divide', 0.4) > 0.3:
            recommendations.extend([
                'Expand digital literacy programs',
                'Provide affordable internet access',
                'Support digital payment adoption in informal sectors'
            ])
        
        # Formalization recommendations
        if 'formalization_incentives' not in policy_factors:
            recommendations.extend([
                'Simplify business registration processes',
                'Provide tax incentives for formalization',
                'Improve access to formal credit for small businesses'
            ])
        
        # Climate adaptation recommendations
        recommendations.extend([
            'Develop climate-resilient livelihood programs',
            'Provide weather-indexed insurance for informal workers',
            'Support adaptation of informal businesses to climate change'
        ])
        
        # Social protection recommendations
        recommendations.extend([
            'Extend social protection coverage to informal workers',
            'Implement portable benefits systems',
            'Strengthen occupational safety and health in informal sectors'
        ])
        
        return recommendations
    
    def _calculate_formalization_benefits(self, formalized_value: float) -> Dict:
        """Calculate benefits from formalization."""
        return {
            'increased_tax_revenue': formalized_value * 0.08,
            'improved_worker_protection': formalized_value * 0.05,
            'enhanced_productivity': formalized_value * 0.12,
            'better_access_to_credit': formalized_value * 0.15,
            'improved_market_access': formalized_value * 0.10
        }
    
    def _identify_formalization_challenges(self) -> List[str]:
        """Identify challenges to formalization."""
        return [
            'High cost of compliance with regulations',
            'Complex bureaucratic procedures',
            'Limited access to formal credit',
            'Lack of awareness about formalization benefits',
            'Fear of increased tax burden',
            'Inadequate social protection systems',
            'Limited market access for formal businesses'
        ]
    
    def _calculate_digital_inclusion_score(self, economic_conditions: Dict) -> float:
        """Calculate digital inclusion score for informal economy."""
        mobile_penetration = economic_conditions.get('mobile_penetration', 0.95)
        internet_access = economic_conditions.get('internet_access', 0.65)
        digital_payment_adoption = economic_conditions.get('digital_payment_adoption', 0.35)
        digital_skills = economic_conditions.get('digital_skills_informal', 0.25)
        
        return (mobile_penetration * 0.3 + internet_access * 0.3 + 
                digital_payment_adoption * 0.25 + digital_skills * 0.15)
    
    def _identify_digital_barriers(self) -> List[str]:
        """Identify barriers to digital adoption."""
        return [
            'Limited digital literacy',
            'High cost of internet access',
            'Lack of appropriate technology',
            'Language barriers in digital platforms',
            'Trust issues with digital payments',
            'Inadequate digital infrastructure',
            'Limited technical support'
        ]
    
    def _identify_digital_opportunities(self) -> List[str]:
        """Identify digital opportunities for informal economy."""
        return [
            'Mobile payment adoption for small transactions',
            'E-commerce platforms for handicrafts and products',
            'Digital platforms for service providers',
            'Online skill development and training',
            'Digital financial services access',
            'GPS-based transport services',
            'Digital marketing for small businesses'
        ]
    
    def _categorize_risk_level(self, vulnerability_score: float) -> str:
        """Categorize risk level based on vulnerability score."""
        if vulnerability_score >= 0.7:
            return 'High'
        elif vulnerability_score >= 0.5:
            return 'Medium'
        else:
            return 'Low'
    
    def _suggest_vulnerability_mitigation(self) -> List[str]:
        """Suggest strategies to mitigate vulnerability."""
        return [
            'Diversify income sources for informal workers',
            'Strengthen social safety nets',
            'Improve access to affordable credit',
            'Provide skills training and capacity building',
            'Develop weather-indexed insurance products',
            'Create cooperative structures for informal workers',
            'Improve market linkages and value chains',
            'Strengthen occupational safety and health measures'
        ]