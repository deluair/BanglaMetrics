"""Construction Sector Model for Bangladesh GDP Simulation.

This module models Bangladesh's construction sector including residential,
commercial, and infrastructure construction. It incorporates urbanization trends,
government infrastructure investment, and climate resilience considerations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConstructionSector:
    """Model for Bangladesh's construction sector.
    
    This class simulates construction activity considering:
    - Residential construction (urban and rural)
    - Commercial and industrial construction
    - Infrastructure development (roads, bridges, ports)
    - Climate-resilient construction
    - Government mega projects
    """
    
    def __init__(self):
        """Initialize construction sector model."""
        
        # Subsector shares in construction GDP (2024 estimates)
        self.subsector_shares = {
            'residential': 0.45,            # Housing construction
            'infrastructure': 0.30,         # Roads, bridges, utilities
            'commercial_industrial': 0.15,  # Commercial and industrial buildings
            'specialized': 0.10             # Specialized construction (ports, airports)
        }
        
        # Residential construction parameters
        self.residential_parameters = {
            'urbanization_elasticity': 1.8,  # High response to urbanization
            'income_elasticity': 1.5,        # Response to income growth
            'urban_rural_ratio': 2.5,        # Urban construction premium
            'housing_deficit': 8.5,          # Million units housing deficit
            'annual_demand_growth': 0.08,    # 8% annual housing demand growth
            'apartment_share_urban': 0.65,   # Apartment share in urban areas
            'single_family_share_rural': 0.90 # Single family share in rural areas
        }
        
        # Infrastructure parameters
        self.infrastructure_parameters = {
            'government_share': 0.75,        # Government infrastructure share
            'private_share': 0.25,           # Private infrastructure share
            'mega_project_impact': 0.20,     # Mega projects impact
            'maintenance_ratio': 0.15,       # Maintenance vs new construction
            'climate_resilience_factor': 0.12, # Climate-resilient infrastructure
            'connectivity_elasticity': 0.8   # Response to connectivity needs
        }
        
        # Commercial and industrial parameters
        self.commercial_parameters = {
            'manufacturing_linkage': 0.6,    # Linkage to manufacturing growth
            'services_linkage': 0.4,         # Linkage to services growth
            'fdi_elasticity': 0.8,           # Response to FDI
            'export_zone_factor': 0.15,      # Export processing zones
            'green_building_adoption': 0.08  # Green building growth
        }
        
        # Construction material parameters
        self.material_parameters = {
            'cement_share': 0.25,            # Cement cost share
            'steel_share': 0.20,             # Steel cost share
            'brick_share': 0.15,             # Brick cost share
            'labor_share': 0.30,             # Labor cost share
            'other_materials_share': 0.10,   # Other materials
            'import_dependency': {
                'cement': 0.05,              # Low cement import
                'steel': 0.40,               # Moderate steel import
                'specialized_materials': 0.70 # High specialized material import
            }
        }
        
        # Seasonal factors
        self.seasonal_factors = {
            'monsoon_impact': 0.75,          # Q2 monsoon impact
            'winter_peak': 1.20,             # Q1 winter peak
            'post_monsoon_recovery': 1.10,   # Q3 recovery
            'year_end_completion': 1.05      # Q4 project completion
        }
        
        # Climate considerations
        self.climate_factors = {
            'flood_resilience_premium': 0.15, # Cost premium for flood resilience
            'cyclone_resistance_premium': 0.12, # Cost premium for cyclone resistance
            'green_building_premium': 0.08,   # Green building cost premium
            'climate_adaptation_factor': 0.10 # General climate adaptation
        }
        
        logger.info("Construction sector model initialized")
    
    def calculate_production(self, 
                           base_year_value: float,
                           quarter: int,
                           year: int,
                           economic_conditions: Dict,
                           policy_factors: Dict = None) -> Dict:
        """Calculate quarterly construction production.
        
        Args:
            base_year_value: Base year construction GDP (billion BDT)
            quarter: Quarter (1-4)
            year: Year
            economic_conditions: Economic conditions affecting construction
            policy_factors: Construction policy impacts
            
        Returns:
            Dictionary with production estimates by subsector
        """
        if policy_factors is None:
            policy_factors = {}
        
        # Calculate subsector production
        subsector_production = {}
        
        # Residential construction
        residential_production = self._calculate_residential_construction(
            base_year_value * self.subsector_shares['residential'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['residential'] = residential_production
        
        # Infrastructure construction
        infrastructure_production = self._calculate_infrastructure_construction(
            base_year_value * self.subsector_shares['infrastructure'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['infrastructure'] = infrastructure_production
        
        # Commercial and industrial construction
        commercial_production = self._calculate_commercial_construction(
            base_year_value * self.subsector_shares['commercial_industrial'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['commercial_industrial'] = commercial_production
        
        # Specialized construction
        specialized_production = self._calculate_specialized_construction(
            base_year_value * self.subsector_shares['specialized'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['specialized'] = specialized_production
        
        # Total construction production
        total_production = sum(subsector_production.values())
        
        return {
            'total_construction_gdp': total_production,
            'subsector_breakdown': subsector_production,
            'material_cost_impact': self._calculate_material_cost_impact(economic_conditions),
            'climate_resilience_investment': self._calculate_climate_investment(subsector_production, policy_factors),
            'employment_impact': self._calculate_employment_impact(subsector_production),
            'sustainability_metrics': self._calculate_sustainability_metrics(policy_factors)
        }
    
    def _calculate_residential_construction(self, 
                                          base_value: float,
                                          quarter: int,
                                          year: int,
                                          economic_conditions: Dict,
                                          policy_factors: Dict) -> float:
        """Calculate residential construction activity."""
        
        # High growth rate due to housing demand
        base_growth = 0.085  # 8.5% annual growth
        
        # Strong seasonal pattern
        seasonal_factors = {
            1: self.seasonal_factors['winter_peak'],           # Peak construction season
            2: self.seasonal_factors['monsoon_impact'],        # Monsoon slowdown
            3: self.seasonal_factors['post_monsoon_recovery'],  # Recovery
            4: self.seasonal_factors['year_end_completion']     # Year-end push
        }
        seasonal_factor = seasonal_factors.get(quarter, 1.0)
        
        # Urbanization impact (major driver)
        urbanization_rate = economic_conditions.get('urbanization_rate', 0.38)
        urban_growth = economic_conditions.get('urban_population_growth', 0.03)
        urbanization_factor = 1 + urban_growth * self.residential_parameters['urbanization_elasticity']
        
        # Income growth impact
        per_capita_income_growth = economic_conditions.get('per_capita_income_growth', 0.04)
        income_factor = 1 + per_capita_income_growth * self.residential_parameters['income_elasticity']
        
        # Housing demand pressure
        housing_demand_factor = 1 + self.residential_parameters['annual_demand_growth']
        
        # Real estate price impact
        real_estate_price_growth = economic_conditions.get('real_estate_price_growth', 0.06)
        price_factor = 1 + real_estate_price_growth * 0.5  # Positive supply response
        
        # Credit availability impact
        housing_credit_growth = economic_conditions.get('housing_credit_growth', 0.12)
        credit_factor = 1 + housing_credit_growth * 0.6
        
        # Policy impact (housing programs)
        policy_impact = 1.0
        if 'affordable_housing_program' in policy_factors:
            policy_impact *= (1 + policy_factors['affordable_housing_program'] * 0.15)
        if 'housing_finance_support' in policy_factors:
            policy_impact *= (1 + policy_factors['housing_finance_support'] * 0.12)
        if 'urban_planning_improvement' in policy_factors:
            policy_impact *= (1 + policy_factors['urban_planning_improvement'] * 0.08)
        
        # Remittance impact (rural housing)
        remittance_growth = economic_conditions.get('remittance_growth', 0.08)
        remittance_factor = 1 + remittance_growth * 0.4  # Significant rural housing impact
        
        # Climate resilience factor
        climate_resilience_factor = 1.0
        if 'climate_resilient_housing' in policy_factors:
            climate_resilience_factor *= (1 + policy_factors['climate_resilient_housing'] * 0.10)
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            urbanization_factor *
            income_factor *
            housing_demand_factor *
            price_factor *
            credit_factor *
            policy_impact *
            remittance_factor *
            climate_resilience_factor
        )
        
        return production
    
    def _calculate_infrastructure_construction(self, 
                                             base_value: float,
                                             quarter: int,
                                             year: int,
                                             economic_conditions: Dict,
                                             policy_factors: Dict) -> float:
        """Calculate infrastructure construction activity."""
        
        # Very high growth rate (government priority)
        base_growth = 0.12  # 12% annual growth
        
        # Moderate seasonal impact (government projects continue)
        seasonal_factors = {
            1: 1.05,  # Slight winter advantage
            2: 0.85,  # Monsoon impact
            3: 1.10,  # Post-monsoon acceleration
            4: 1.00   # Normal pace
        }
        seasonal_factor = seasonal_factors.get(quarter, 1.0)
        
        # Government budget allocation
        infrastructure_budget_growth = economic_conditions.get('infrastructure_budget_growth', 0.15)
        budget_factor = 1 + infrastructure_budget_growth * 0.8
        
        # Mega project impact
        mega_project_factor = 1.0
        if economic_conditions.get('mega_project_implementation', 0) > 0:
            mega_project_factor *= (1 + economic_conditions['mega_project_implementation'] * 0.25)
        
        # Economic growth linkage (infrastructure follows growth)
        gdp_growth = economic_conditions.get('gdp_growth', 0.06)
        growth_linkage_factor = 1 + gdp_growth * 1.5  # Infrastructure grows faster than GDP
        
        # Foreign financing impact
        foreign_financing = economic_conditions.get('infrastructure_foreign_financing', 0.08)
        financing_factor = 1 + foreign_financing * 0.6
        
        # Policy impact (infrastructure development)
        policy_impact = 1.0
        if 'infrastructure_development_program' in policy_factors:
            policy_impact *= (1 + policy_factors['infrastructure_development_program'] * 0.20)
        if 'transport_infrastructure' in policy_factors:
            policy_impact *= (1 + policy_factors['transport_infrastructure'] * 0.15)
        if 'digital_infrastructure' in policy_factors:
            policy_impact *= (1 + policy_factors['digital_infrastructure'] * 0.10)
        
        # Climate adaptation infrastructure
        climate_adaptation_factor = 1.0
        if 'climate_adaptation_infrastructure' in policy_factors:
            climate_adaptation_factor *= (1 + policy_factors['climate_adaptation_infrastructure'] * 0.18)
        
        # Connectivity needs (trade and manufacturing linkage)
        trade_growth = economic_conditions.get('trade_growth', 0.08)
        connectivity_factor = 1 + trade_growth * self.infrastructure_parameters['connectivity_elasticity']
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            budget_factor *
            mega_project_factor *
            growth_linkage_factor *
            financing_factor *
            policy_impact *
            climate_adaptation_factor *
            connectivity_factor
        )
        
        return production
    
    def _calculate_commercial_construction(self, 
                                         base_value: float,
                                         quarter: int,
                                         year: int,
                                         economic_conditions: Dict,
                                         policy_factors: Dict) -> float:
        """Calculate commercial and industrial construction."""
        
        # High growth rate (business expansion)
        base_growth = 0.095  # 9.5% annual growth
        
        # Moderate seasonal variation
        seasonal_factors = {1: 1.08, 2: 0.90, 3: 1.05, 4: 0.97}
        seasonal_factor = seasonal_factors.get(quarter, 1.0)
        
        # Manufacturing sector linkage
        manufacturing_growth = economic_conditions.get('manufacturing_growth', 0.08)
        manufacturing_factor = 1 + manufacturing_growth * self.commercial_parameters['manufacturing_linkage']
        
        # Services sector linkage
        services_growth = economic_conditions.get('services_growth', 0.06)
        services_factor = 1 + services_growth * self.commercial_parameters['services_linkage']
        
        # Foreign Direct Investment impact
        fdi_growth = economic_conditions.get('fdi_growth', 0.10)
        fdi_factor = 1 + fdi_growth * self.commercial_parameters['fdi_elasticity']
        
        # Export processing zones development
        export_zone_factor = 1.0
        if economic_conditions.get('export_zone_expansion', 0) > 0:
            export_zone_factor *= (1 + economic_conditions['export_zone_expansion'] * 0.20)
        
        # Commercial real estate demand
        commercial_real_estate_growth = economic_conditions.get('commercial_real_estate_growth', 0.08)
        real_estate_factor = 1 + commercial_real_estate_growth * 0.7
        
        # Policy impact (industrial development)
        policy_impact = 1.0
        if 'industrial_park_development' in policy_factors:
            policy_impact *= (1 + policy_factors['industrial_park_development'] * 0.18)
        if 'commercial_zone_development' in policy_factors:
            policy_impact *= (1 + policy_factors['commercial_zone_development'] * 0.12)
        if 'green_building_incentives' in policy_factors:
            policy_impact *= (1 + policy_factors['green_building_incentives'] * 0.08)
        
        # Technology and automation factor
        tech_factor = 1 + (year - 2024) * 0.04  # 4% annual tech-driven construction
        
        # Energy efficiency requirements
        energy_efficiency_factor = 1 + (year - 2024) * 0.03  # Growing energy efficiency focus
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            manufacturing_factor *
            services_factor *
            fdi_factor *
            export_zone_factor *
            real_estate_factor *
            policy_impact *
            tech_factor *
            energy_efficiency_factor
        )
        
        return production
    
    def _calculate_specialized_construction(self, 
                                          base_value: float,
                                          quarter: int,
                                          year: int,
                                          economic_conditions: Dict,
                                          policy_factors: Dict) -> float:
        """Calculate specialized construction (ports, airports, etc.)."""
        
        # Very high growth rate (strategic projects)
        base_growth = 0.15  # 15% annual growth
        
        # Lower seasonal impact (strategic projects continue)
        seasonal_factors = {1: 1.02, 2: 0.92, 3: 1.06, 4: 1.00}
        seasonal_factor = seasonal_factors.get(quarter, 1.0)
        
        # Trade growth linkage (ports, logistics)
        trade_growth = economic_conditions.get('trade_growth', 0.08)
        trade_factor = 1 + trade_growth * 1.2  # High elasticity
        
        # Tourism infrastructure linkage
        tourism_growth = economic_conditions.get('tourism_growth', 0.05)
        tourism_factor = 1 + tourism_growth * 0.8
        
        # Strategic project implementation
        strategic_project_factor = 1.0
        if economic_conditions.get('strategic_project_implementation', 0) > 0:
            strategic_project_factor *= (1 + economic_conditions['strategic_project_implementation'] * 0.30)
        
        # International connectivity needs
        connectivity_demand = economic_conditions.get('international_connectivity_demand', 0.06)
        connectivity_factor = 1 + connectivity_demand * 1.0
        
        # Policy impact (strategic infrastructure)
        policy_impact = 1.0
        if 'port_development' in policy_factors:
            policy_impact *= (1 + policy_factors['port_development'] * 0.25)
        if 'airport_expansion' in policy_factors:
            policy_impact *= (1 + policy_factors['airport_expansion'] * 0.20)
        if 'special_economic_zones' in policy_factors:
            policy_impact *= (1 + policy_factors['special_economic_zones'] * 0.22)
        
        # Technology and modernization factor
        modernization_factor = 1 + (year - 2024) * 0.08  # 8% annual modernization
        
        # Climate resilience for critical infrastructure
        climate_resilience_factor = 1.0
        if 'critical_infrastructure_resilience' in policy_factors:
            climate_resilience_factor *= (1 + policy_factors['critical_infrastructure_resilience'] * 0.15)
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            trade_factor *
            tourism_factor *
            strategic_project_factor *
            connectivity_factor *
            policy_impact *
            modernization_factor *
            climate_resilience_factor
        )
        
        return production
    
    def _calculate_material_cost_impact(self, economic_conditions: Dict) -> Dict:
        """Calculate impact of material costs on construction."""
        
        # Material price changes
        cement_price_change = economic_conditions.get('cement_price_change', 0.05)
        steel_price_change = economic_conditions.get('steel_price_change', 0.08)
        fuel_price_change = economic_conditions.get('fuel_price_change', 0.06)
        
        # Calculate weighted cost impact
        material_cost_impact = (
            cement_price_change * self.material_parameters['cement_share'] +
            steel_price_change * self.material_parameters['steel_share'] +
            fuel_price_change * 0.1  # Fuel affects transport and machinery
        )
        
        # Labor cost impact
        wage_growth = economic_conditions.get('construction_wage_growth', 0.08)
        labor_cost_impact = wage_growth * self.material_parameters['labor_share']
        
        # Total cost impact
        total_cost_impact = material_cost_impact + labor_cost_impact
        
        return {
            'total_cost_impact': total_cost_impact,
            'material_cost_impact': material_cost_impact,
            'labor_cost_impact': labor_cost_impact,
            'cost_breakdown': {
                'cement': cement_price_change * self.material_parameters['cement_share'],
                'steel': steel_price_change * self.material_parameters['steel_share'],
                'labor': labor_cost_impact,
                'fuel_transport': fuel_price_change * 0.1
            },
            'cost_mitigation_strategies': self._suggest_cost_mitigation(economic_conditions)
        }
    
    def _calculate_climate_investment(self, subsector_production: Dict, policy_factors: Dict) -> Dict:
        """Calculate climate resilience investment in construction."""
        
        total_production = sum(subsector_production.values())
        
        # Base climate investment (percentage of total construction)
        base_climate_investment_rate = 0.08  # 8% of construction for climate resilience
        
        # Policy-driven increases
        policy_boost = 0
        if 'climate_resilient_construction' in policy_factors:
            policy_boost += policy_factors['climate_resilient_construction'] * 0.05
        if 'flood_protection_infrastructure' in policy_factors:
            policy_boost += policy_factors['flood_protection_infrastructure'] * 0.04
        if 'cyclone_resistant_building_codes' in policy_factors:
            policy_boost += policy_factors['cyclone_resistant_building_codes'] * 0.03
        
        climate_investment_rate = base_climate_investment_rate + policy_boost
        climate_investment = total_production * climate_investment_rate
        
        # Investment breakdown
        investment_breakdown = {
            'flood_resilience': climate_investment * 0.40,
            'cyclone_resistance': climate_investment * 0.25,
            'earthquake_resistance': climate_investment * 0.15,
            'energy_efficiency': climate_investment * 0.12,
            'sustainable_materials': climate_investment * 0.08
        }
        
        return {
            'total_climate_investment': climate_investment,
            'investment_rate': climate_investment_rate,
            'investment_breakdown': investment_breakdown,
            'climate_benefits': self._calculate_climate_benefits(climate_investment)
        }
    
    def _calculate_employment_impact(self, subsector_production: Dict) -> Dict:
        """Calculate employment impact of construction sector."""
        
        # Employment intensities by subsector (jobs per billion BDT)
        employment_intensities = {
            'residential': 12000,           # High labor intensity
            'infrastructure': 10000,        # Moderate labor intensity
            'commercial_industrial': 9000,  # Lower labor intensity (more mechanized)
            'specialized': 8000             # Lowest labor intensity (high-tech)
        }
        
        # Calculate employment by subsector
        subsector_employment = {}
        total_employment = 0
        
        for subsector, production in subsector_production.items():
            intensity = employment_intensities.get(subsector, 10000)
            employment = production * intensity / 1000  # Convert to thousands of jobs
            subsector_employment[subsector] = employment
            total_employment += employment
        
        # Skill composition
        skill_composition = {
            'unskilled': 0.45,      # 45% unskilled workers
            'semi_skilled': 0.35,   # 35% semi-skilled workers
            'skilled': 0.15,        # 15% skilled workers
            'professional': 0.05    # 5% professional/technical
        }
        
        return {
            'total_employment': total_employment,
            'subsector_employment': subsector_employment,
            'skill_composition': skill_composition,
            'average_wage_index': 1.08,     # 8% above national average
            'employment_growth_rate': 0.10, # 10% annual employment growth
            'safety_standards_compliance': 0.65,  # 65% compliance with safety standards
            'gender_participation': {
                'male': 0.92,
                'female': 0.08              # Low female participation
            }
        }
    
    def _calculate_sustainability_metrics(self, policy_factors: Dict) -> Dict:
        """Calculate sustainability metrics for construction sector."""
        
        # Base sustainability score
        base_sustainability = 0.45  # Current sustainability level
        
        # Policy-driven improvements
        sustainability_improvements = 0
        
        if 'green_building_standards' in policy_factors:
            sustainability_improvements += policy_factors['green_building_standards'] * 0.15
        if 'sustainable_materials_promotion' in policy_factors:
            sustainability_improvements += policy_factors['sustainable_materials_promotion'] * 0.12
        if 'energy_efficient_construction' in policy_factors:
            sustainability_improvements += policy_factors['energy_efficient_construction'] * 0.10
        if 'waste_reduction_construction' in policy_factors:
            sustainability_improvements += policy_factors['waste_reduction_construction'] * 0.08
        
        overall_sustainability = min(1.0, base_sustainability + sustainability_improvements)
        
        return {
            'overall_sustainability_score': overall_sustainability,
            'green_building_adoption_rate': min(0.30, 0.08 + sustainability_improvements * 2),
            'sustainable_materials_usage': min(0.40, 0.15 + sustainability_improvements * 1.5),
            'energy_efficiency_rating': min(1.0, 0.55 + sustainability_improvements * 1.2),
            'waste_reduction_rate': min(0.50, 0.20 + sustainability_improvements * 1.8),
            'carbon_footprint_reduction': min(0.30, sustainability_improvements * 2),
            'certification_compliance': {
                'leed_certified': min(0.15, sustainability_improvements * 3),
                'local_green_standards': min(0.25, 0.05 + sustainability_improvements * 2),
                'energy_star_rating': min(0.20, sustainability_improvements * 2.5)
            }
        }
    
    def _suggest_cost_mitigation(self, economic_conditions: Dict) -> List[str]:
        """Suggest cost mitigation strategies."""
        strategies = []
        
        cement_price_change = economic_conditions.get('cement_price_change', 0.05)
        steel_price_change = economic_conditions.get('steel_price_change', 0.08)
        
        if cement_price_change > 0.08:
            strategies.extend([
                'Promote alternative building materials',
                'Develop local cement production capacity',
                'Implement bulk purchasing programs'
            ])
        
        if steel_price_change > 0.10:
            strategies.extend([
                'Increase domestic steel production',
                'Explore steel recycling programs',
                'Develop alternative structural materials'
            ])
        
        strategies.extend([
            'Improve construction productivity through technology',
            'Develop skilled workforce to reduce labor costs',
            'Implement efficient project management systems',
            'Promote modular and prefabricated construction'
        ])
        
        return strategies
    
    def _calculate_climate_benefits(self, climate_investment: float) -> Dict:
        """Calculate benefits from climate resilience investment."""
        
        # Benefits per unit of climate investment
        benefit_multipliers = {
            'avoided_damage_costs': 3.5,    # 3.5x return on investment
            'reduced_maintenance_costs': 1.8, # 1.8x savings in maintenance
            'improved_property_values': 1.2,  # 1.2x property value increase
            'insurance_premium_reduction': 0.8 # 0.8x insurance savings
        }
        
        climate_benefits = {}
        total_benefits = 0
        
        for benefit_type, multiplier in benefit_multipliers.items():
            benefit_value = climate_investment * multiplier
            climate_benefits[benefit_type] = benefit_value
            total_benefits += benefit_value
        
        return {
            'total_climate_benefits': total_benefits,
            'benefit_breakdown': climate_benefits,
            'benefit_cost_ratio': total_benefits / climate_investment if climate_investment > 0 else 0,
            'resilience_improvement': min(1.0, climate_investment / 100)  # Resilience score improvement
        }