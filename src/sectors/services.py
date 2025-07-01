"""Services Sector Model for Bangladesh GDP Simulation.

This module models Bangladesh's services sector including trade, transport,
financial services, telecommunications, education, health, and other services.
It incorporates digitalization trends, urbanization impacts, and service quality improvements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ServicesSector:
    """Model for Bangladesh's services sector.
    
    This class simulates services production considering:
    - Trade and commerce (wholesale/retail)
    - Transportation and logistics
    - Financial services and mobile banking
    - Telecommunications and ICT
    - Education and health services
    - Government services
    - Tourism and hospitality
    """
    
    def __init__(self):
        """Initialize services sector model."""
        
        # Subsector shares in services GDP (2024 estimates)
        self.subsector_shares = {
            'wholesale_retail_trade': 0.28,     # Largest services subsector
            'transport_storage': 0.15,          # Transport and logistics
            'financial_services': 0.12,         # Banking, insurance, MFS
            'real_estate': 0.10,                # Real estate activities
            'public_administration': 0.08,      # Government services
            'education': 0.08,                  # Education services
            'health_social_work': 0.06,         # Health and social services
            'information_communication': 0.05,   # ICT and telecom
            'accommodation_food': 0.04,         # Hotels and restaurants
            'professional_services': 0.04       # Professional and business services
        }
        
        # Trade sector parameters
        self.trade_parameters = {
            'urbanization_elasticity': 0.8,     # Response to urbanization
            'income_elasticity': 1.2,           # Response to income growth
            'digitalization_impact': 0.15,      # E-commerce growth impact
            'informal_share': 0.65,             # Informal trade share
            'seasonal_variation': 0.12          # Seasonal demand variation
        }
        
        # Transport sector parameters
        self.transport_parameters = {
            'infrastructure_elasticity': 0.6,   # Response to infrastructure
            'trade_elasticity': 0.9,            # Response to trade growth
            'fuel_price_elasticity': -0.4,      # Response to fuel prices
            'modal_shares': {
                'road': 0.75,                   # Road transport dominance
                'rail': 0.08,                   # Railway transport
                'water': 0.12,                  # Water transport
                'air': 0.05                     # Air transport
            }
        }
        
        # Financial services parameters
        self.financial_parameters = {
            'banking_share': 0.65,              # Traditional banking
            'mfs_share': 0.20,                  # Mobile Financial Services
            'insurance_share': 0.10,            # Insurance services
            'capital_market_share': 0.05,       # Capital market services
            'financial_inclusion_rate': 0.58,   # Current inclusion rate
            'mfs_growth_rate': 0.25,            # Annual MFS growth
            'digital_banking_adoption': 0.35    # Digital banking adoption
        }
        
        # ICT sector parameters
        self.ict_parameters = {
            'mobile_penetration': 1.08,         # Mobile subscriptions per capita
            'internet_penetration': 0.39,       # Internet users percentage
            'broadband_penetration': 0.12,      # Broadband penetration
            'digital_services_growth': 0.18,    # Annual digital services growth
            'it_export_growth': 0.22,           # IT services export growth
            'e_governance_adoption': 0.25       # E-governance service adoption
        }
        
        # Education sector parameters
        self.education_parameters = {
            'public_share': 0.75,               # Public education share
            'private_share': 0.25,              # Private education share
            'enrollment_growth': 0.03,          # Annual enrollment growth
            'quality_improvement': 0.02,        # Annual quality improvement
            'digitalization_rate': 0.15,        # Education digitalization
            'skill_development_focus': 0.20     # Skill development emphasis
        }
        
        # Health sector parameters
        self.health_parameters = {
            'public_share': 0.60,               # Public health share
            'private_share': 0.40,              # Private health share
            'demographic_growth_factor': 0.025, # Aging population impact
            'health_infrastructure_growth': 0.08, # Infrastructure expansion
            'telemedicine_adoption': 0.12,      # Telemedicine growth
            'medical_tourism_potential': 0.05   # Medical tourism growth
        }
        
        logger.info("Services sector model initialized")
    
    def calculate_production(self, 
                           base_year_value: float,
                           quarter: int,
                           year: int,
                           economic_conditions: Dict,
                           policy_factors: Dict = None) -> Dict:
        """Calculate quarterly services production.
        
        Args:
            base_year_value: Base year services GDP (billion BDT)
            quarter: Quarter (1-4)
            year: Year
            economic_conditions: Economic conditions affecting services
            policy_factors: Service sector policy impacts
            
        Returns:
            Dictionary with production estimates by subsector
        """
        if policy_factors is None:
            policy_factors = {}
        
        # Calculate subsector production
        subsector_production = {}
        
        # Trade services
        trade_production = self._calculate_trade_services(
            base_year_value * self.subsector_shares['wholesale_retail_trade'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['wholesale_retail_trade'] = trade_production
        
        # Transport services
        transport_production = self._calculate_transport_services(
            base_year_value * self.subsector_shares['transport_storage'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['transport_storage'] = transport_production
        
        # Financial services
        financial_production = self._calculate_financial_services(
            base_year_value * self.subsector_shares['financial_services'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['financial_services'] = financial_production
        
        # ICT services
        ict_production = self._calculate_ict_services(
            base_year_value * self.subsector_shares['information_communication'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['information_communication'] = ict_production
        
        # Education services
        education_production = self._calculate_education_services(
            base_year_value * self.subsector_shares['education'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['education'] = education_production
        
        # Health services
        health_production = self._calculate_health_services(
            base_year_value * self.subsector_shares['health_social_work'],
            quarter, year, economic_conditions, policy_factors
        )
        subsector_production['health_social_work'] = health_production
        
        # Other services
        for subsector in ['real_estate', 'public_administration', 'accommodation_food', 'professional_services']:
            production = self._calculate_other_services(
                base_year_value * self.subsector_shares[subsector],
                subsector, quarter, year, economic_conditions, policy_factors
            )
            subsector_production[subsector] = production
        
        # Total services production
        total_production = sum(subsector_production.values())
        
        return {
            'total_services_gdp': total_production,
            'subsector_breakdown': subsector_production,
            'digitalization_impact': self._calculate_digitalization_impact(economic_conditions, policy_factors),
            'urbanization_impact': self._calculate_urbanization_impact(economic_conditions),
            'service_quality_index': self._calculate_service_quality_index(policy_factors),
            'employment_characteristics': self._analyze_employment_characteristics(subsector_production)
        }
    
    def _calculate_trade_services(self, 
                                base_value: float,
                                quarter: int,
                                year: int,
                                economic_conditions: Dict,
                                policy_factors: Dict) -> float:
        """Calculate wholesale and retail trade services."""
        
        # Base growth rate
        base_growth = 0.055  # 5.5% annual growth
        
        # Seasonal factor (higher in Q4 due to festivals)
        seasonal_factors = {1: 0.95, 2: 0.98, 3: 1.02, 4: 1.05}
        seasonal_factor = seasonal_factors.get(quarter, 1.0)
        
        # Income and consumption growth impact
        income_growth = economic_conditions.get('per_capita_income_growth', 0.04)
        consumption_factor = 1 + income_growth * self.trade_parameters['income_elasticity']
        
        # Urbanization impact
        urbanization_rate = economic_conditions.get('urbanization_rate', 0.38)
        urbanization_factor = 1 + (urbanization_rate - 0.35) * self.trade_parameters['urbanization_elasticity']
        
        # Digitalization impact (e-commerce growth)
        digital_factor = 1.0
        if 'digital_commerce_development' in policy_factors:
            digital_factor *= (1 + policy_factors['digital_commerce_development'] * 0.12)
        
        # E-commerce penetration growth
        ecommerce_growth = (year - 2024) * 0.08  # 8% annual e-commerce growth
        digital_factor *= (1 + ecommerce_growth * self.trade_parameters['digitalization_impact'])
        
        # Infrastructure impact
        infrastructure_factor = 1.0
        if 'market_infrastructure' in policy_factors:
            infrastructure_factor *= (1 + policy_factors['market_infrastructure'] * 0.08)
        
        # Financial inclusion impact
        financial_inclusion = economic_conditions.get('financial_inclusion_rate', 0.58)
        inclusion_factor = 1 + (financial_inclusion - 0.55) * 0.5
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            consumption_factor *
            urbanization_factor *
            digital_factor *
            infrastructure_factor *
            inclusion_factor
        )
        
        return production
    
    def _calculate_transport_services(self, 
                                    base_value: float,
                                    quarter: int,
                                    year: int,
                                    economic_conditions: Dict,
                                    policy_factors: Dict) -> float:
        """Calculate transport and storage services."""
        
        # Base growth rate
        base_growth = 0.065  # 6.5% annual growth
        
        # Seasonal factor (lower in Q2 due to monsoon)
        seasonal_factors = {1: 1.02, 2: 0.90, 3: 1.03, 4: 1.05}
        seasonal_factor = seasonal_factors.get(quarter, 1.0)
        
        # Trade growth impact
        trade_growth = economic_conditions.get('trade_growth', 0.08)
        trade_factor = 1 + trade_growth * self.transport_parameters['trade_elasticity']
        
        # Infrastructure development impact
        infrastructure_investment = economic_conditions.get('transport_infrastructure_investment', 0.12)
        infrastructure_factor = 1 + infrastructure_investment * self.transport_parameters['infrastructure_elasticity']
        
        # Fuel price impact
        fuel_price_change = economic_conditions.get('fuel_price_change', 0.05)
        fuel_factor = 1 + fuel_price_change * self.transport_parameters['fuel_price_elasticity']
        
        # Policy impact (transport development)
        policy_impact = 1.0
        if 'transport_modernization' in policy_factors:
            policy_impact *= (1 + policy_factors['transport_modernization'] * 0.10)
        if 'logistics_development' in policy_factors:
            policy_impact *= (1 + policy_factors['logistics_development'] * 0.08)
        
        # Digitalization impact (ride-sharing, logistics apps)
        digital_transport_factor = 1 + (year - 2024) * 0.06  # 6% annual digital growth
        
        # Manufacturing and export growth impact
        manufacturing_growth = economic_conditions.get('manufacturing_growth', 0.08)
        export_factor = 1 + manufacturing_growth * 0.6
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            trade_factor *
            infrastructure_factor *
            fuel_factor *
            policy_impact *
            digital_transport_factor *
            export_factor
        )
        
        return production
    
    def _calculate_financial_services(self, 
                                    base_value: float,
                                    quarter: int,
                                    year: int,
                                    economic_conditions: Dict,
                                    policy_factors: Dict) -> float:
        """Calculate financial and insurance services."""
        
        # High growth rate (financial deepening)
        base_growth = 0.085  # 8.5% annual growth
        
        # Minimal seasonal variation
        seasonal_factor = 1.0 + 0.02 * np.sin(2 * np.pi * quarter / 4)
        
        # Mobile Financial Services (MFS) growth
        mfs_growth_factor = (1 + self.financial_parameters['mfs_growth_rate']) ** (year - 2024)
        mfs_impact = self.financial_parameters['mfs_share'] * (mfs_growth_factor - 1) + 1
        
        # Financial inclusion expansion
        inclusion_growth = (year - 2024) * 0.03  # 3% annual inclusion growth
        inclusion_factor = 1 + inclusion_growth * 0.8
        
        # Digital banking adoption
        digital_banking_factor = 1 + (year - 2024) * 0.08  # 8% annual digital adoption
        
        # Economic growth impact (credit demand)
        gdp_growth = economic_conditions.get('gdp_growth', 0.06)
        credit_demand_factor = 1 + gdp_growth * 1.2  # Financial services grow faster than GDP
        
        # Policy impact (financial sector development)
        policy_impact = 1.0
        if 'financial_inclusion_program' in policy_factors:
            policy_impact *= (1 + policy_factors['financial_inclusion_program'] * 0.12)
        if 'digital_banking_promotion' in policy_factors:
            policy_impact *= (1 + policy_factors['digital_banking_promotion'] * 0.10)
        if 'capital_market_development' in policy_factors:
            policy_impact *= (1 + policy_factors['capital_market_development'] * 0.08)
        
        # Remittance flow impact
        remittance_growth = economic_conditions.get('remittance_growth', 0.08)
        remittance_factor = 1 + remittance_growth * 0.3  # MFS benefits from remittances
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            mfs_impact *
            inclusion_factor *
            digital_banking_factor *
            credit_demand_factor *
            policy_impact *
            remittance_factor
        )
        
        return production
    
    def _calculate_ict_services(self, 
                              base_value: float,
                              quarter: int,
                              year: int,
                              economic_conditions: Dict,
                              policy_factors: Dict) -> float:
        """Calculate ICT and telecommunications services."""
        
        # Very high growth rate (digital transformation)
        base_growth = 0.12  # 12% annual growth
        
        # Minimal seasonal variation
        seasonal_factor = 1.0 + 0.01 * np.cos(2 * np.pi * quarter / 4)
        
        # Internet penetration growth
        internet_growth = (year - 2024) * 0.05  # 5% annual penetration growth
        internet_factor = 1 + internet_growth * 2.0  # High elasticity
        
        # Mobile data consumption growth
        mobile_data_factor = (1.15) ** (year - 2024)  # 15% annual growth
        
        # Digital services expansion
        digital_services_factor = (1 + self.ict_parameters['digital_services_growth']) ** (year - 2024)
        
        # IT export growth
        it_export_factor = 1 + (year - 2024) * self.ict_parameters['it_export_growth'] * 0.1
        
        # Policy impact (Digital Bangladesh initiative)
        policy_impact = 1.0
        if 'digital_infrastructure' in policy_factors:
            policy_impact *= (1 + policy_factors['digital_infrastructure'] * 0.15)
        if 'it_sector_development' in policy_factors:
            policy_impact *= (1 + policy_factors['it_sector_development'] * 0.12)
        if 'e_governance' in policy_factors:
            policy_impact *= (1 + policy_factors['e_governance'] * 0.08)
        
        # 5G and broadband expansion
        broadband_factor = 1 + (year - 2024) * 0.10  # 10% annual broadband impact
        
        # COVID-19 digitalization acceleration (permanent effect)
        covid_digital_boost = 1.25  # 25% permanent boost from pandemic digitalization
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            internet_factor *
            mobile_data_factor *
            digital_services_factor *
            it_export_factor *
            policy_impact *
            broadband_factor *
            covid_digital_boost
        )
        
        return production
    
    def _calculate_education_services(self, 
                                    base_value: float,
                                    quarter: int,
                                    year: int,
                                    economic_conditions: Dict,
                                    policy_factors: Dict) -> float:
        """Calculate education services."""
        
        # Moderate growth rate
        base_growth = 0.045  # 4.5% annual growth
        
        # Academic calendar seasonality
        seasonal_factors = {1: 1.05, 2: 0.95, 3: 0.95, 4: 1.05}
        seasonal_factor = seasonal_factors.get(quarter, 1.0)
        
        # Demographic factor (school-age population)
        demographic_factor = 1 + (year - 2024) * 0.015  # 1.5% annual demographic growth
        
        # Income growth impact (private education demand)
        income_growth = economic_conditions.get('per_capita_income_growth', 0.04)
        private_education_factor = 1 + income_growth * 1.5 * self.education_parameters['private_share']
        
        # Digitalization impact (online education)
        digital_education_factor = 1 + (year - 2024) * self.education_parameters['digitalization_rate']
        
        # Policy impact (education development)
        policy_impact = 1.0
        if 'education_infrastructure' in policy_factors:
            policy_impact *= (1 + policy_factors['education_infrastructure'] * 0.10)
        if 'teacher_training' in policy_factors:
            policy_impact *= (1 + policy_factors['teacher_training'] * 0.08)
        if 'skill_development' in policy_factors:
            policy_impact *= (1 + policy_factors['skill_development'] * 0.12)
        
        # Quality improvement factor
        quality_factor = 1 + (year - 2024) * self.education_parameters['quality_improvement']
        
        # Higher education expansion
        higher_education_factor = 1 + (year - 2024) * 0.08  # 8% annual higher education growth
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            demographic_factor *
            private_education_factor *
            digital_education_factor *
            policy_impact *
            quality_factor *
            higher_education_factor
        )
        
        return production
    
    def _calculate_health_services(self, 
                                 base_value: float,
                                 quarter: int,
                                 year: int,
                                 economic_conditions: Dict,
                                 policy_factors: Dict) -> float:
        """Calculate health and social work services."""
        
        # High growth rate (healthcare expansion)
        base_growth = 0.075  # 7.5% annual growth
        
        # Seasonal factor (higher in winter months)
        seasonal_factors = {1: 1.08, 2: 0.95, 3: 0.95, 4: 1.02}
        seasonal_factor = seasonal_factors.get(quarter, 1.0)
        
        # Demographic factor (aging population, health awareness)
        demographic_factor = 1 + (year - 2024) * self.health_parameters['demographic_growth_factor']
        
        # Income growth impact (private healthcare demand)
        income_growth = economic_conditions.get('per_capita_income_growth', 0.04)
        private_health_factor = 1 + income_growth * 1.8 * self.health_parameters['private_share']
        
        # Health infrastructure expansion
        infrastructure_factor = 1 + (year - 2024) * self.health_parameters['health_infrastructure_growth']
        
        # Telemedicine and digital health
        digital_health_factor = 1 + (year - 2024) * self.health_parameters['telemedicine_adoption']
        
        # Policy impact (health sector development)
        policy_impact = 1.0
        if 'health_infrastructure' in policy_factors:
            policy_impact *= (1 + policy_factors['health_infrastructure'] * 0.12)
        if 'universal_health_coverage' in policy_factors:
            policy_impact *= (1 + policy_factors['universal_health_coverage'] * 0.10)
        if 'medical_education' in policy_factors:
            policy_impact *= (1 + policy_factors['medical_education'] * 0.08)
        
        # Medical tourism potential
        medical_tourism_factor = 1 + (year - 2024) * self.health_parameters['medical_tourism_potential']
        
        # Pharmaceutical sector linkage
        pharma_growth = economic_conditions.get('pharmaceutical_growth', 0.08)
        pharma_linkage_factor = 1 + pharma_growth * 0.3
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            demographic_factor *
            private_health_factor *
            infrastructure_factor *
            digital_health_factor *
            policy_impact *
            medical_tourism_factor *
            pharma_linkage_factor
        )
        
        return production
    
    def _calculate_other_services(self, 
                                base_value: float,
                                subsector: str,
                                quarter: int,
                                year: int,
                                economic_conditions: Dict,
                                policy_factors: Dict) -> float:
        """Calculate other services subsectors."""
        
        # Subsector-specific growth rates
        growth_rates = {
            'real_estate': 0.08,           # High growth due to urbanization
            'public_administration': 0.03,  # Steady government growth
            'accommodation_food': 0.06,     # Tourism and urbanization
            'professional_services': 0.09   # Business services growth
        }
        
        base_growth = growth_rates.get(subsector, 0.05)
        
        # Seasonal factors
        seasonal_factor = 1.0
        if subsector == 'accommodation_food':
            seasonal_factors = {1: 1.05, 2: 0.85, 3: 0.95, 4: 1.15}  # Tourism seasonality
            seasonal_factor = seasonal_factors.get(quarter, 1.0)
        elif subsector == 'real_estate':
            seasonal_factors = {1: 1.02, 2: 0.98, 3: 1.00, 4: 1.00}  # Mild seasonality
            seasonal_factor = seasonal_factors.get(quarter, 1.0)
        
        # Specific factors by subsector
        specific_factor = 1.0
        
        if subsector == 'real_estate':
            urbanization_rate = economic_conditions.get('urbanization_rate', 0.38)
            specific_factor *= (1 + (urbanization_rate - 0.35) * 2.0)  # High urbanization elasticity
            
        elif subsector == 'accommodation_food':
            tourism_growth = economic_conditions.get('tourism_growth', 0.05)
            specific_factor *= (1 + tourism_growth * 1.5)
            
        elif subsector == 'professional_services':
            business_growth = economic_conditions.get('business_services_demand', 0.08)
            specific_factor *= (1 + business_growth * 1.2)
        
        # Policy impact
        policy_impact = 1.0
        if f'{subsector}_development' in policy_factors:
            policy_impact *= (1 + policy_factors[f'{subsector}_development'] * 0.08)
        
        # Technology factor
        tech_factor = 1 + (year - 2024) * 0.03  # 3% annual tech improvement
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            specific_factor *
            policy_impact *
            tech_factor
        )
        
        return production
    
    def _calculate_digitalization_impact(self, economic_conditions: Dict, policy_factors: Dict) -> Dict:
        """Calculate digitalization impact on services."""
        
        # Base digitalization score
        base_score = 0.35  # Current digitalization level
        
        # Policy-driven improvements
        policy_boost = 0
        if 'digital_transformation' in policy_factors:
            policy_boost += policy_factors['digital_transformation'] * 0.15
        if 'e_governance' in policy_factors:
            policy_boost += policy_factors['e_governance'] * 0.10
        if 'digital_skills' in policy_factors:
            policy_boost += policy_factors['digital_skills'] * 0.08
        
        # Infrastructure impact
        infrastructure_score = economic_conditions.get('digital_infrastructure_score', 0.6)
        infrastructure_boost = (infrastructure_score - 0.5) * 0.2
        
        # Overall digitalization impact
        digitalization_score = min(1.0, base_score + policy_boost + infrastructure_boost)
        
        return {
            'overall_digitalization_score': digitalization_score,
            'e_commerce_penetration': min(0.25, 0.08 + policy_boost * 2),
            'digital_payment_adoption': min(0.70, 0.45 + policy_boost * 1.5),
            'online_service_delivery': min(0.60, 0.30 + policy_boost * 2),
            'digital_skills_index': min(1.0, 0.55 + policy_boost * 1.2)
        }
    
    def _calculate_urbanization_impact(self, economic_conditions: Dict) -> Dict:
        """Calculate urbanization impact on services."""
        
        urbanization_rate = economic_conditions.get('urbanization_rate', 0.38)
        urban_growth_rate = economic_conditions.get('urban_population_growth', 0.03)
        
        # Services most affected by urbanization
        urbanization_elasticities = {
            'wholesale_retail_trade': 1.2,
            'transport_storage': 1.0,
            'financial_services': 1.5,
            'real_estate': 2.0,
            'accommodation_food': 1.8,
            'professional_services': 1.6
        }
        
        impact_by_service = {}
        for service, elasticity in urbanization_elasticities.items():
            impact = 1 + urban_growth_rate * elasticity
            impact_by_service[service] = impact
        
        return {
            'current_urbanization_rate': urbanization_rate,
            'urban_growth_rate': urban_growth_rate,
            'service_impacts': impact_by_service,
            'urban_service_premium': 1.25  # Urban services 25% more productive
        }
    
    def _calculate_service_quality_index(self, policy_factors: Dict) -> float:
        """Calculate overall service quality index."""
        
        base_quality = 0.65  # Current service quality
        
        # Quality improvements from policies
        quality_improvements = 0
        
        if 'service_quality_standards' in policy_factors:
            quality_improvements += policy_factors['service_quality_standards'] * 0.12
        if 'customer_protection' in policy_factors:
            quality_improvements += policy_factors['customer_protection'] * 0.08
        if 'professional_certification' in policy_factors:
            quality_improvements += policy_factors['professional_certification'] * 0.10
        if 'service_innovation' in policy_factors:
            quality_improvements += policy_factors['service_innovation'] * 0.15
        
        service_quality_index = min(1.0, base_quality + quality_improvements)
        
        return service_quality_index
    
    def _analyze_employment_characteristics(self, subsector_production: Dict) -> Dict:
        """Analyze employment characteristics of services sector."""
        
        # Employment intensities by subsector
        employment_intensities = {
            'wholesale_retail_trade': 0.85,     # High employment intensity
            'transport_storage': 0.70,
            'financial_services': 0.60,
            'information_communication': 0.55,
            'education': 0.90,                  # Very high employment intensity
            'health_social_work': 0.88,
            'accommodation_food': 0.82,
            'professional_services': 0.65,
            'real_estate': 0.45,
            'public_administration': 0.75
        }
        
        # Calculate employment impact
        total_employment_impact = 0
        subsector_employment = {}
        
        for subsector, production in subsector_production.items():
            intensity = employment_intensities.get(subsector, 0.70)
            employment_impact = production * intensity * 0.001  # Convert to employment units
            subsector_employment[subsector] = employment_impact
            total_employment_impact += employment_impact
        
        return {
            'total_employment_impact': total_employment_impact,
            'subsector_employment': subsector_employment,
            'average_skill_level': 0.72,        # Services require higher skills
            'formal_employment_share': 0.45,    # Lower formality than manufacturing
            'gender_participation': {
                'male': 0.58,
                'female': 0.42                  # Better gender balance in services
            },
            'productivity_growth': 0.035,       # Annual productivity growth
            'wage_premium': 1.15               # 15% wage premium over agriculture
        }