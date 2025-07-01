"""Manufacturing Sector Model for Bangladesh GDP Simulation.

This module models Bangladesh's manufacturing sector with special focus on
Ready-Made Garments (RMG), textiles, pharmaceuticals, and other key industries.
It incorporates global demand patterns, competitiveness factors, and supply chain dynamics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ManufacturingSector:
    """Model for Bangladesh's manufacturing sector.
    
    This class simulates manufacturing production considering:
    - RMG industry dynamics and global demand
    - Textile production and backward linkages
    - Pharmaceutical industry growth
    - Food processing and other manufacturing
    - Global supply chain impacts
    """
    
    def __init__(self):
        """Initialize manufacturing sector model."""
        
        # Subsector shares in manufacturing GDP (2024 estimates)
        self.subsector_shares = {
            'rmg': 0.42,                    # Ready-Made Garments (dominant)
            'textiles': 0.18,               # Yarn, fabric, dyeing
            'food_processing': 0.12,        # Food and beverages
            'pharmaceuticals': 0.08,        # Medicines and chemicals
            'leather_footwear': 0.05,       # Leather goods and shoes
            'jute_products': 0.03,          # Jute and jute goods
            'cement': 0.04,                 # Cement and construction materials
            'steel': 0.03,                  # Steel and metal products
            'electronics': 0.02,            # Electronics assembly
            'other_manufacturing': 0.03     # Other manufacturing
        }
        
        # RMG sector parameters (most important)
        self.rmg_parameters = {
            'export_dependency': 0.95,      # 95% export-oriented
            'major_markets': {
                'usa': 0.18,                # 18% of exports
                'germany': 0.13,            # 13% of exports
                'uk': 0.10,                 # 10% of exports
                'spain': 0.07,              # 7% of exports
                'france': 0.06,             # 6% of exports
                'other_eu': 0.25,           # 25% other EU
                'other_markets': 0.21       # 21% other markets
            },
            'product_mix': {
                'knitwear': 0.58,           # Knitwear dominance
                'woven': 0.42               # Woven garments
            },
            'employment': 4.2,              # Million workers
            'factories': 4500,              # Number of factories
            'compliance_score': 0.75,       # Compliance rating
            'productivity_growth': 0.035    # Annual productivity growth
        }
        
        # Global demand elasticities
        self.demand_elasticities = {
            'rmg': {
                'income_elasticity': 1.2,   # Elastic to income
                'price_elasticity': -1.8,   # Highly price sensitive
                'fashion_cycle_impact': 0.15 # Fashion trend impact
            },
            'pharmaceuticals': {
                'income_elasticity': 0.8,
                'price_elasticity': -0.6,
                'demographic_factor': 0.12  # Aging population
            },
            'food_processing': {
                'income_elasticity': 0.6,
                'price_elasticity': -0.9,
                'urbanization_factor': 0.08
            }
        }
        
        # Competitiveness factors
        self.competitiveness_factors = {
            'labor_cost_advantage': 0.85,   # Relative to competitors
            'infrastructure_quality': 0.65, # Infrastructure score
            'trade_facilitation': 0.70,     # Trade ease score
            'energy_reliability': 0.75,     # Power supply reliability
            'skill_level': 0.68,            # Worker skill level
            'technology_adoption': 0.60     # Technology uptake
        }
        
        # Supply chain parameters
        self.supply_chain = {
            'backward_linkage': 0.35,       # Local input sourcing
            'import_dependency': {
                'cotton': 0.95,             # High cotton import
                'machinery': 0.90,          # Machinery import
                'chemicals': 0.75,          # Chemical import
                'accessories': 0.60         # Accessories import
            },
            'lead_times': {
                'rmg': 45,                  # Days
                'textiles': 30,             # Days
                'pharmaceuticals': 60       # Days
            }
        }
        
        logger.info("Manufacturing sector model initialized")
    
    def calculate_production(self, 
                           base_year_value: float,
                           quarter: int,
                           year: int,
                           global_conditions: Dict,
                           policy_factors: Dict = None) -> Dict:
        """Calculate quarterly manufacturing production.
        
        Args:
            base_year_value: Base year manufacturing GDP (billion BDT)
            quarter: Quarter (1-4)
            year: Year
            global_conditions: Global economic conditions
            policy_factors: Manufacturing policy impacts
            
        Returns:
            Dictionary with production estimates by subsector
        """
        if policy_factors is None:
            policy_factors = {}
        
        # Calculate subsector production
        subsector_production = {}
        
        # RMG production (most important)
        rmg_production = self._calculate_rmg_production(
            base_year_value * self.subsector_shares['rmg'],
            quarter, year, global_conditions, policy_factors
        )
        subsector_production['rmg'] = rmg_production
        
        # Textiles production
        textiles_production = self._calculate_textiles_production(
            base_year_value * self.subsector_shares['textiles'],
            quarter, year, global_conditions, policy_factors
        )
        subsector_production['textiles'] = textiles_production
        
        # Pharmaceuticals production
        pharma_production = self._calculate_pharmaceuticals_production(
            base_year_value * self.subsector_shares['pharmaceuticals'],
            quarter, year, global_conditions, policy_factors
        )
        subsector_production['pharmaceuticals'] = pharma_production
        
        # Food processing production
        food_production = self._calculate_food_processing_production(
            base_year_value * self.subsector_shares['food_processing'],
            quarter, year, global_conditions, policy_factors
        )
        subsector_production['food_processing'] = food_production
        
        # Other manufacturing subsectors
        for subsector in ['leather_footwear', 'jute_products', 'cement', 'steel', 'electronics', 'other_manufacturing']:
            production = self._calculate_other_manufacturing_production(
                base_year_value * self.subsector_shares[subsector],
                subsector, quarter, year, global_conditions, policy_factors
            )
            subsector_production[subsector] = production
        
        # Total manufacturing production
        total_production = sum(subsector_production.values())
        
        return {
            'total_manufacturing_gdp': total_production,
            'subsector_breakdown': subsector_production,
            'export_performance': self._calculate_export_performance(subsector_production, global_conditions),
            'competitiveness_index': self._calculate_competitiveness_index(global_conditions, policy_factors),
            'supply_chain_resilience': self._assess_supply_chain_resilience(global_conditions),
            'employment_impact': self._calculate_employment_impact(subsector_production)
        }
    
    def _calculate_rmg_production(self, 
                                base_value: float,
                                quarter: int,
                                year: int,
                                global_conditions: Dict,
                                policy_factors: Dict) -> float:
        """Calculate RMG production with global demand and competitiveness factors."""
        
        # Base growth rate
        base_growth = 0.055  # 5.5% annual growth
        
        # Seasonal factor (fashion cycles)
        seasonal_factor = self._get_rmg_seasonal_factor(quarter)
        
        # Global demand impact
        global_demand_factor = self._calculate_global_rmg_demand(global_conditions)
        
        # Competitiveness factor
        competitiveness_factor = self._calculate_rmg_competitiveness(global_conditions, policy_factors)
        
        # Trade policy impact
        trade_policy_factor = 1.0
        if 'trade_agreements' in policy_factors:
            trade_policy_factor *= (1 + policy_factors['trade_agreements'] * 0.08)
        if 'export_incentives' in policy_factors:
            trade_policy_factor *= (1 + policy_factors['export_incentives'] * 0.05)
        
        # Compliance and sustainability factor
        compliance_factor = 1.0
        if 'workplace_safety' in policy_factors:
            compliance_factor *= (1 + policy_factors['workplace_safety'] * 0.03)
        if 'green_manufacturing' in policy_factors:
            compliance_factor *= (1 + policy_factors['green_manufacturing'] * 0.04)
        
        # Technology and productivity factor
        tech_factor = 1 + (year - 2024) * self.rmg_parameters['productivity_growth']
        
        # Supply chain disruption factor
        supply_chain_factor = 1.0
        if global_conditions.get('supply_chain_disruption', 0) > 0.3:
            supply_chain_factor *= 0.92
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            global_demand_factor *
            competitiveness_factor *
            trade_policy_factor *
            compliance_factor *
            tech_factor *
            supply_chain_factor
        )
        
        return production
    
    def _calculate_textiles_production(self, 
                                     base_value: float,
                                     quarter: int,
                                     year: int,
                                     global_conditions: Dict,
                                     policy_factors: Dict) -> float:
        """Calculate textiles production (backward linkage to RMG)."""
        
        # Base growth rate (linked to RMG growth)
        base_growth = 0.048  # 4.8% annual growth
        
        # Seasonal factor (follows RMG patterns)
        seasonal_factor = self._get_rmg_seasonal_factor(quarter) * 0.8 + 0.2
        
        # Backward linkage factor (benefits from RMG growth)
        backward_linkage_factor = 1.0
        if global_conditions.get('rmg_demand_growth', 0) > 0:
            backward_linkage_factor *= (1 + global_conditions['rmg_demand_growth'] * 0.6)
        
        # Cotton price impact
        cotton_price_factor = 1.0
        cotton_price_change = global_conditions.get('cotton_price_change', 0)
        if cotton_price_change != 0:
            cotton_price_factor *= (1 - cotton_price_change * 0.3)  # Negative impact of higher prices
        
        # Policy impact (textile development)
        policy_impact = 1.0
        if 'textile_park_development' in policy_factors:
            policy_impact *= (1 + policy_factors['textile_park_development'] * 0.12)
        if 'backward_linkage_incentive' in policy_factors:
            policy_impact *= (1 + policy_factors['backward_linkage_incentive'] * 0.08)
        
        # Technology factor (modern machinery)
        tech_factor = 1 + (year - 2024) * 0.025  # 2.5% annual tech improvement
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            backward_linkage_factor *
            cotton_price_factor *
            policy_impact *
            tech_factor
        )
        
        return production
    
    def _calculate_pharmaceuticals_production(self, 
                                            base_value: float,
                                            quarter: int,
                                            year: int,
                                            global_conditions: Dict,
                                            policy_factors: Dict) -> float:
        """Calculate pharmaceuticals production."""
        
        # High growth rate (expanding sector)
        base_growth = 0.085  # 8.5% annual growth
        
        # Minimal seasonal variation
        seasonal_factor = 1.0 + 0.02 * np.sin(2 * np.pi * quarter / 4)
        
        # Domestic demand factor (healthcare expansion)
        domestic_demand_factor = 1.0 + (year - 2024) * 0.015  # Growing healthcare needs
        
        # Export potential factor
        export_factor = 1.0
        if global_conditions.get('global_health_demand', 0) > 0:
            export_factor *= (1 + global_conditions['global_health_demand'] * 0.15)
        
        # Policy impact (pharma development)
        policy_impact = 1.0
        if 'pharma_park_development' in policy_factors:
            policy_impact *= (1 + policy_factors['pharma_park_development'] * 0.15)
        if 'research_development_incentive' in policy_factors:
            policy_impact *= (1 + policy_factors['research_development_incentive'] * 0.10)
        if 'generic_drug_policy' in policy_factors:
            policy_impact *= (1 + policy_factors['generic_drug_policy'] * 0.08)
        
        # Technology and R&D factor
        tech_factor = 1 + (year - 2024) * 0.03  # 3% annual tech improvement
        
        # Raw material availability
        raw_material_factor = 1.0
        if global_conditions.get('api_supply_disruption', 0) > 0.2:
            raw_material_factor *= 0.95
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            domestic_demand_factor *
            export_factor *
            policy_impact *
            tech_factor *
            raw_material_factor
        )
        
        return production
    
    def _calculate_food_processing_production(self, 
                                            base_value: float,
                                            quarter: int,
                                            year: int,
                                            global_conditions: Dict,
                                            policy_factors: Dict) -> float:
        """Calculate food processing production."""
        
        # Moderate growth rate
        base_growth = 0.042  # 4.2% annual growth
        
        # Seasonal factor (harvest seasons)
        seasonal_factor = 1.0 + 0.06 * np.cos(2 * np.pi * (quarter - 1) / 4)
        
        # Urbanization factor (growing processed food demand)
        urbanization_factor = 1 + (year - 2024) * 0.012  # 1.2% annual urbanization impact
        
        # Agricultural input availability
        agri_input_factor = 1.0
        if global_conditions.get('agricultural_production_index', 1.0) != 1.0:
            agri_input_factor *= global_conditions['agricultural_production_index']
        
        # Policy impact (food safety, processing zones)
        policy_impact = 1.0
        if 'food_processing_zone' in policy_factors:
            policy_impact *= (1 + policy_factors['food_processing_zone'] * 0.10)
        if 'food_safety_standards' in policy_factors:
            policy_impact *= (1 + policy_factors['food_safety_standards'] * 0.05)
        
        # Technology factor (modern processing)
        tech_factor = 1 + (year - 2024) * 0.02  # 2% annual tech improvement
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            urbanization_factor *
            agri_input_factor *
            policy_impact *
            tech_factor
        )
        
        return production
    
    def _calculate_other_manufacturing_production(self, 
                                                base_value: float,
                                                subsector: str,
                                                quarter: int,
                                                year: int,
                                                global_conditions: Dict,
                                                policy_factors: Dict) -> float:
        """Calculate production for other manufacturing subsectors."""
        
        # Subsector-specific growth rates
        growth_rates = {
            'leather_footwear': 0.045,
            'jute_products': 0.025,
            'cement': 0.065,
            'steel': 0.055,
            'electronics': 0.075,
            'other_manufacturing': 0.040
        }
        
        base_growth = growth_rates.get(subsector, 0.040)
        
        # Seasonal factors
        seasonal_factor = 1.0
        if subsector == 'cement':
            seasonal_factor = 1.0 + 0.08 * np.cos(2 * np.pi * (quarter - 2) / 4)  # Peak in dry season
        elif subsector == 'jute_products':
            seasonal_factor = 1.0 + 0.05 * np.sin(2 * np.pi * quarter / 4)  # Harvest-linked
        
        # Global demand factor
        global_factor = 1.0
        if subsector in ['leather_footwear', 'jute_products']:
            global_factor *= (1 + global_conditions.get('global_demand_growth', 0) * 0.5)
        
        # Infrastructure factor (especially for cement, steel)
        infra_factor = 1.0
        if subsector in ['cement', 'steel']:
            infra_factor *= (1 + global_conditions.get('infrastructure_investment', 0) * 0.8)
        
        # Policy impact
        policy_impact = 1.0
        if f'{subsector}_development' in policy_factors:
            policy_impact *= (1 + policy_factors[f'{subsector}_development'] * 0.08)
        
        # Technology factor
        tech_factor = 1 + (year - 2024) * 0.02
        
        production = (
            base_value * 
            (1 + base_growth) ** (year - 2024) *
            seasonal_factor *
            global_factor *
            infra_factor *
            policy_impact *
            tech_factor
        )
        
        return production
    
    def _get_rmg_seasonal_factor(self, quarter: int) -> float:
        """Get seasonal factor for RMG production (fashion cycles)."""
        # RMG has distinct seasonal patterns
        seasonal_factors = {
            1: 1.05,  # Q1: Spring/Summer production
            2: 0.95,  # Q2: Lower production (transition)
            3: 1.10,  # Q3: Fall/Winter production peak
            4: 0.90   # Q4: Year-end slowdown
        }
        return seasonal_factors.get(quarter, 1.0)
    
    def _calculate_global_rmg_demand(self, global_conditions: Dict) -> float:
        """Calculate global demand factor for RMG."""
        demand_factor = 1.0
        
        # Major market economic conditions
        for market, share in self.rmg_parameters['major_markets'].items():
            market_growth = global_conditions.get(f'{market}_economic_growth', 0.02)
            demand_factor += market_growth * share * self.demand_elasticities['rmg']['income_elasticity']
        
        # Global fashion trends
        fashion_trend = global_conditions.get('fashion_trend_factor', 0)
        demand_factor *= (1 + fashion_trend * self.demand_elasticities['rmg']['fashion_cycle_impact'])
        
        # Competitor pricing
        competitor_price_change = global_conditions.get('competitor_price_change', 0)
        demand_factor *= (1 - competitor_price_change * 0.5)  # Bangladesh benefits from competitor price increases
        
        return max(demand_factor, 0.7)  # Minimum 70% of normal demand
    
    def _calculate_rmg_competitiveness(self, global_conditions: Dict, policy_factors: Dict) -> float:
        """Calculate RMG competitiveness factor."""
        competitiveness = 1.0
        
        # Labor cost competitiveness
        wage_growth = global_conditions.get('wage_growth_rate', 0.08)
        competitor_wage_growth = global_conditions.get('competitor_wage_growth', 0.06)
        competitiveness *= (1 + (competitor_wage_growth - wage_growth) * 0.3)
        
        # Infrastructure improvements
        if 'infrastructure_development' in policy_factors:
            competitiveness *= (1 + policy_factors['infrastructure_development'] * 0.06)
        
        # Skill development
        if 'skill_development' in policy_factors:
            competitiveness *= (1 + policy_factors['skill_development'] * 0.04)
        
        # Energy reliability
        energy_reliability = global_conditions.get('energy_reliability_score', 0.75)
        competitiveness *= (0.8 + 0.2 * energy_reliability)
        
        return competitiveness
    
    def _calculate_export_performance(self, subsector_production: Dict, global_conditions: Dict) -> Dict:
        """Calculate export performance metrics."""
        total_exports = 0
        export_breakdown = {}
        
        # RMG exports (dominant)
        rmg_exports = subsector_production['rmg'] * self.rmg_parameters['export_dependency']
        export_breakdown['rmg'] = rmg_exports
        total_exports += rmg_exports
        
        # Other export-oriented sectors
        export_ratios = {
            'textiles': 0.25,
            'leather_footwear': 0.80,
            'jute_products': 0.60,
            'pharmaceuticals': 0.15,
            'food_processing': 0.10
        }
        
        for sector, ratio in export_ratios.items():
            if sector in subsector_production:
                exports = subsector_production[sector] * ratio
                export_breakdown[sector] = exports
                total_exports += exports
        
        return {
            'total_manufacturing_exports': total_exports,
            'export_breakdown': export_breakdown,
            'export_growth_rate': global_conditions.get('export_growth_rate', 0.08),
            'market_diversification_index': self._calculate_market_diversification(),
            'export_competitiveness_index': self._calculate_export_competitiveness(global_conditions)
        }
    
    def _calculate_competitiveness_index(self, global_conditions: Dict, policy_factors: Dict) -> float:
        """Calculate overall manufacturing competitiveness index."""
        base_score = 0.68  # Current competitiveness score
        
        # Factor improvements
        improvements = 0
        
        if 'infrastructure_development' in policy_factors:
            improvements += policy_factors['infrastructure_development'] * 0.15
        
        if 'skill_development' in policy_factors:
            improvements += policy_factors['skill_development'] * 0.12
        
        if 'technology_adoption' in policy_factors:
            improvements += policy_factors['technology_adoption'] * 0.10
        
        if 'trade_facilitation' in policy_factors:
            improvements += policy_factors['trade_facilitation'] * 0.08
        
        # Global competitiveness trends
        global_trend = global_conditions.get('global_competitiveness_trend', 0)
        
        competitiveness_index = min(1.0, base_score + improvements + global_trend)
        
        return competitiveness_index
    
    def _assess_supply_chain_resilience(self, global_conditions: Dict) -> Dict:
        """Assess supply chain resilience."""
        resilience_score = 0.7  # Base resilience
        
        # Disruption factors
        disruptions = {
            'shipping_delays': global_conditions.get('shipping_disruption', 0),
            'raw_material_shortage': global_conditions.get('raw_material_disruption', 0),
            'energy_supply_issues': global_conditions.get('energy_disruption', 0),
            'trade_restrictions': global_conditions.get('trade_restriction_level', 0)
        }
        
        for disruption, level in disruptions.items():
            resilience_score *= (1 - level * 0.2)
        
        return {
            'overall_resilience_score': max(resilience_score, 0.3),
            'disruption_factors': disruptions,
            'mitigation_strategies': self._suggest_supply_chain_mitigation(disruptions),
            'diversification_needs': self._assess_diversification_needs()
        }
    
    def _calculate_employment_impact(self, subsector_production: Dict) -> Dict:
        """Calculate employment impact of manufacturing production."""
        # Employment elasticities by subsector
        employment_elasticities = {
            'rmg': 0.8,                    # Labor-intensive
            'textiles': 0.7,
            'food_processing': 0.6,
            'pharmaceuticals': 0.5,
            'leather_footwear': 0.75,
            'other': 0.6
        }
        
        total_employment_change = 0
        subsector_employment = {}
        
        for subsector, production in subsector_production.items():
            elasticity = employment_elasticities.get(subsector, 0.6)
            # Simplified employment calculation
            employment_change = production * elasticity * 0.001  # Convert to employment units
            subsector_employment[subsector] = employment_change
            total_employment_change += employment_change
        
        return {
            'total_employment_impact': total_employment_change,
            'subsector_employment': subsector_employment,
            'job_quality_index': 0.72,  # Manufacturing job quality
            'skill_requirements': self._assess_skill_requirements()
        }
    
    def _calculate_market_diversification(self) -> float:
        """Calculate market diversification index."""
        # Based on RMG market distribution
        market_shares = list(self.rmg_parameters['major_markets'].values())
        # Herfindahl-Hirschman Index (lower is more diversified)
        hhi = sum(share**2 for share in market_shares)
        diversification_index = 1 - hhi  # Convert to diversification measure
        return diversification_index
    
    def _calculate_export_competitiveness(self, global_conditions: Dict) -> float:
        """Calculate export competitiveness index."""
        base_competitiveness = 0.75
        
        # Adjust for global factors
        adjustments = [
            global_conditions.get('exchange_rate_competitiveness', 0),
            global_conditions.get('trade_policy_favorability', 0),
            global_conditions.get('logistics_performance', 0) * 0.5
        ]
        
        competitiveness = base_competitiveness + sum(adjustments)
        return min(max(competitiveness, 0.3), 1.0)
    
    def _suggest_supply_chain_mitigation(self, disruptions: Dict) -> List[str]:
        """Suggest supply chain mitigation strategies."""
        strategies = []
        
        if disruptions.get('shipping_delays', 0) > 0.3:
            strategies.extend([
                'Diversify shipping routes and ports',
                'Increase inventory buffers',
                'Develop regional sourcing options'
            ])
        
        if disruptions.get('raw_material_shortage', 0) > 0.3:
            strategies.extend([
                'Develop alternative suppliers',
                'Invest in backward linkages',
                'Create strategic material reserves'
            ])
        
        if disruptions.get('energy_supply_issues', 0) > 0.3:
            strategies.extend([
                'Invest in renewable energy',
                'Improve energy efficiency',
                'Develop backup power systems'
            ])
        
        return strategies
    
    def _assess_diversification_needs(self) -> Dict:
        """Assess product and market diversification needs."""
        return {
            'product_diversification': {
                'current_concentration': 0.42,  # RMG share
                'recommended_target': 0.35,
                'priority_sectors': ['pharmaceuticals', 'electronics', 'food_processing']
            },
            'market_diversification': {
                'current_concentration': 0.46,  # EU+USA share
                'recommended_target': 0.40,
                'priority_markets': ['asia_pacific', 'middle_east', 'africa']
            }
        }
    
    def _assess_skill_requirements(self) -> Dict:
        """Assess skill requirements for manufacturing sectors."""
        return {
            'current_skill_level': 0.68,
            'required_skill_level': 0.75,
            'priority_skills': [
                'technical_skills',
                'quality_control',
                'digital_literacy',
                'lean_manufacturing',
                'compliance_knowledge'
            ],
            'training_needs': {
                'rmg': 'quality_and_compliance',
                'pharmaceuticals': 'technical_and_regulatory',
                'electronics': 'advanced_technical',
                'textiles': 'modern_machinery_operation'
            }
        }