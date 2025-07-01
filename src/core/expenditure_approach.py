"""Expenditure Approach GDP Calculation for Bangladesh.

This module implements the expenditure approach to GDP calculation, measuring
final demand components: consumption, investment, government expenditure,
and net exports with Bangladesh-specific characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExpenditureApproach:
    """Expenditure approach GDP calculation following BBS methodology.
    
    Calculates GDP as C + I + G + (X - M) with detailed modeling of
    Bangladesh's consumption patterns, investment flows, and trade structure.
    """
    
    def __init__(self, base_year: int = 2015):
        """Initialize Expenditure Approach calculator.
        
        Args:
            base_year: Base year for constant price calculations
        """
        self.base_year = base_year
        
        # Bangladesh expenditure structure (% of GDP)
        self.expenditure_shares = {
            'private_consumption': 0.68,    # Household consumption
            'government_consumption': 0.06, # Government current expenditure
            'gross_investment': 0.31,       # Fixed investment + inventory changes
            'exports': 0.15,                # Goods and services exports
            'imports': 0.20                 # Goods and services imports
        }
        
        # Investment breakdown
        self.investment_components = {
            'private_fixed': 0.65,          # Private sector fixed investment
            'government_fixed': 0.30,       # Government infrastructure
            'inventory_changes': 0.05       # Stock changes
        }
        
        # Export structure (heavily concentrated in RMG)
        self.export_structure = {
            'rmg_textiles': 0.82,           # Ready-made garments
            'agricultural_products': 0.05,  # Rice, tea, fish
            'leather_goods': 0.03,          # Leather and footwear
            'pharmaceuticals': 0.02,        # Growing pharma exports
            'services': 0.05,               # IT services, remittances
            'other_goods': 0.03             # Other manufactured goods
        }
        
        # Import structure
        self.import_structure = {
            'capital_goods': 0.25,          # Machinery, equipment
            'intermediate_goods': 0.35,     # Raw materials, components
            'consumer_goods': 0.15,         # Final consumption goods
            'fuel_energy': 0.20,            # Oil, gas, coal
            'food_items': 0.05              # Food imports
        }
        
        logger.info(f"Expenditure Approach initialized with base year {base_year}")
    
    def calculate_gdp(self, 
                     year: int, 
                     quarter: Optional[int] = None,
                     provisional: bool = True) -> Dict:
        """Calculate GDP using expenditure approach.
        
        Args:
            year: Year for calculation
            quarter: Quarter (1-4) for quarterly estimates
            provisional: Whether this is provisional estimate
            
        Returns:
            Dictionary with GDP calculation results
        """
        logger.info(f"Calculating expenditure approach GDP for {year}{'Q' + str(quarter) if quarter else ''}")
        
        # Calculate expenditure components
        consumption = self._calculate_consumption(year, quarter)
        investment = self._calculate_investment(year, quarter)
        government = self._calculate_government_expenditure(year, quarter)
        exports = self._calculate_exports(year, quarter)
        imports = self._calculate_imports(year, quarter)
        
        # Calculate net exports
        net_exports = exports['total_nominal'] - imports['total_nominal']
        net_exports_real = exports['total_real'] - imports['total_real']
        
        # Total GDP calculation: C + I + G + (X - M)
        nominal_gdp = (consumption['total_nominal'] + 
                      investment['total_nominal'] + 
                      government['total_nominal'] + 
                      net_exports)
        
        real_gdp = (consumption['total_real'] + 
                   investment['total_real'] + 
                   government['total_real'] + 
                   net_exports_real)
        
        return {
            'nominal_gdp': nominal_gdp,
            'real_gdp': real_gdp,
            'gdp_deflator': (nominal_gdp / real_gdp) * 100,
            'components': {
                'private_consumption': consumption,
                'gross_investment': investment,
                'government_expenditure': government,
                'exports': exports,
                'imports': imports,
                'net_exports': {
                    'nominal': net_exports,
                    'real': net_exports_real
                }
            },
            'expenditure_shares': {
                'consumption_share': consumption['total_nominal'] / nominal_gdp,
                'investment_share': investment['total_nominal'] / nominal_gdp,
                'government_share': government['total_nominal'] / nominal_gdp,
                'net_exports_share': net_exports / nominal_gdp
            },
            'calculation_metadata': {
                'year': year,
                'quarter': quarter,
                'provisional': provisional,
                'base_year': self.base_year
            }
        }
    
    def _calculate_consumption(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate private consumption expenditure.
        
        Models household consumption patterns including urban-rural differences,
        remittance impact, and MFS-enabled spending.
        """
        # Base consumption (billion BDT)
        base_consumption = 11800  # Approximate 2024 private consumption
        
        # Growth factors
        years_from_base = year - 2024
        base_growth = 0.055  # 5.5% annual growth
        
        # Remittance impact on consumption
        remittance_factor = self._get_remittance_impact_factor(year)
        
        # MFS impact on consumption patterns
        mfs_factor = self._get_mfs_consumption_factor(year)
        
        # Seasonal consumption patterns
        seasonal_factor = 1.0
        if quarter:
            # Q1: Post-winter, Q2: Pre-monsoon, Q3: Monsoon/Eid, Q4: Post-harvest
            seasonal_factors = {1: 0.95, 2: 1.0, 3: 1.15, 4: 0.90}  # Eid boost in Q3
            seasonal_factor = seasonal_factors[quarter]
        
        # Calculate nominal consumption
        nominal_consumption = base_consumption * (1 + base_growth) ** years_from_base
        nominal_consumption *= remittance_factor * mfs_factor * seasonal_factor
        
        # Urban-rural breakdown
        urban_share = 0.38 + (year - 2024) * 0.01  # Urbanization trend
        rural_share = 1 - urban_share
        
        urban_consumption = nominal_consumption * urban_share * 1.8  # Higher urban consumption per capita
        rural_consumption = nominal_consumption * rural_share * 0.6  # Lower rural consumption per capita
        
        # Normalize to total
        total_weighted = urban_consumption + rural_consumption
        urban_consumption = (urban_consumption / total_weighted) * nominal_consumption
        rural_consumption = (rural_consumption / total_weighted) * nominal_consumption
        
        # Real consumption (deflated)
        price_index = self._get_price_index(year, 'consumption')
        real_consumption = nominal_consumption / price_index
        
        return {
            'total_nominal': nominal_consumption,
            'total_real': real_consumption,
            'urban_consumption': urban_consumption,
            'rural_consumption': rural_consumption,
            'remittance_impact_factor': remittance_factor,
            'mfs_impact_factor': mfs_factor,
            'consumption_categories': {
                'food_beverages': nominal_consumption * 0.45,      # Food dominant in Bangladesh
                'housing_utilities': nominal_consumption * 0.20,   # Housing costs
                'transport_communication': nominal_consumption * 0.12, # Transport, mobile
                'clothing_footwear': nominal_consumption * 0.08,   # Clothing
                'health_education': nominal_consumption * 0.10,    # Health, education
                'other_goods_services': nominal_consumption * 0.05 # Other items
            }
        }
    
    def _calculate_investment(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate gross fixed capital formation and inventory changes.
        
        Models private and government investment including infrastructure development,
        RMG factory expansion, and climate adaptation investments.
        """
        # Base investment (billion BDT)
        base_investment = 5400  # Approximate 2024 gross investment
        
        # Investment growth (higher than GDP growth)
        years_from_base = year - 2024
        investment_growth = 0.08  # 8% annual growth
        
        # Calculate total investment
        nominal_investment = base_investment * (1 + investment_growth) ** years_from_base
        
        # Climate adaptation investment factor
        climate_investment_factor = self._get_climate_investment_factor(year)
        nominal_investment *= climate_investment_factor
        
        # Break down by components
        private_fixed = nominal_investment * self.investment_components['private_fixed']
        government_fixed = nominal_investment * self.investment_components['government_fixed']
        inventory_changes = nominal_investment * self.investment_components['inventory_changes']
        
        # Seasonal patterns for investment
        if quarter:
            # Q1: Planning, Q2: Implementation, Q3: Monsoon slowdown, Q4: Completion
            seasonal_factors = {1: 0.85, 2: 1.20, 3: 0.80, 4: 1.15}
            seasonal_factor = seasonal_factors[quarter]
            private_fixed *= seasonal_factor
            government_fixed *= seasonal_factor
        
        # Sector-specific investment breakdown
        private_investment_sectors = {
            'rmg_manufacturing': private_fixed * 0.35,      # RMG factory expansion
            'real_estate_construction': private_fixed * 0.25, # Housing, commercial
            'transport_logistics': private_fixed * 0.15,    # Transport infrastructure
            'telecommunications': private_fixed * 0.10,     # Digital infrastructure
            'other_manufacturing': private_fixed * 0.10,    # Other industries
            'services': private_fixed * 0.05                # Service sector investment
        }
        
        government_investment_sectors = {
            'transport_infrastructure': government_fixed * 0.40,  # Roads, bridges, ports
            'power_energy': government_fixed * 0.25,             # Power plants, grid
            'education_health': government_fixed * 0.15,         # Schools, hospitals
            'climate_adaptation': government_fixed * 0.10,       # Flood protection, etc.
            'digital_infrastructure': government_fixed * 0.05,   # Digital Bangladesh
            'other_infrastructure': government_fixed * 0.05      # Other public investment
        }
        
        # Real investment (deflated)
        price_index = self._get_price_index(year, 'investment')
        real_investment = nominal_investment / price_index
        
        return {
            'total_nominal': nominal_investment,
            'total_real': real_investment,
            'private_fixed_investment': {
                'total': private_fixed,
                'sectors': private_investment_sectors
            },
            'government_fixed_investment': {
                'total': government_fixed,
                'sectors': government_investment_sectors
            },
            'inventory_changes': inventory_changes,
            'climate_investment_factor': climate_investment_factor
        }
    
    def _calculate_government_expenditure(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate government consumption expenditure.
        
        Models government current spending on salaries, operations, and services.
        """
        # Base government expenditure (billion BDT)
        base_gov_expenditure = 1050  # Approximate 2024 government consumption
        
        # Government expenditure growth
        years_from_base = year - 2024
        gov_growth = 0.06  # 6% annual growth
        
        nominal_gov_expenditure = base_gov_expenditure * (1 + gov_growth) ** years_from_base
        
        # Climate response expenditure factor
        climate_response_factor = self._get_climate_response_factor(year, quarter)
        nominal_gov_expenditure *= climate_response_factor
        
        # Government expenditure breakdown
        expenditure_breakdown = {
            'personnel_costs': nominal_gov_expenditure * 0.45,      # Salaries, benefits
            'goods_services': nominal_gov_expenditure * 0.25,       # Operations
            'defense_security': nominal_gov_expenditure * 0.15,     # Defense spending
            'social_protection': nominal_gov_expenditure * 0.10,    # Social programs
            'climate_emergency': nominal_gov_expenditure * 0.05     # Climate response
        }
        
        # Real government expenditure
        price_index = self._get_price_index(year, 'government')
        real_gov_expenditure = nominal_gov_expenditure / price_index
        
        return {
            'total_nominal': nominal_gov_expenditure,
            'total_real': real_gov_expenditure,
            'expenditure_breakdown': expenditure_breakdown,
            'climate_response_factor': climate_response_factor
        }
    
    def _calculate_exports(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate exports of goods and services.
        
        Dominated by RMG exports with growing services exports.
        """
        # Base exports (billion BDT)
        base_exports = 2600  # Approximate 2024 total exports
        
        # Export growth factors
        years_from_base = year - 2024
        
        # RMG export calculation (dominant component)
        rmg_exports = self._calculate_rmg_exports(year, quarter)
        
        # Other goods exports
        other_goods_base = base_exports * (1 - self.export_structure['rmg_textiles'] - self.export_structure['services'])
        other_goods_growth = 0.06  # 6% annual growth
        other_goods_exports = other_goods_base * (1 + other_goods_growth) ** years_from_base
        
        # Services exports (IT, remittances counted as transfers)
        services_base = base_exports * self.export_structure['services']
        services_growth = 0.15  # 15% annual growth (IT services boom)
        services_exports = services_base * (1 + services_growth) ** years_from_base
        
        # Total exports
        total_nominal_exports = rmg_exports['nominal'] + other_goods_exports + services_exports
        total_real_exports = rmg_exports['real'] + other_goods_exports / self._get_price_index(year, 'exports') + services_exports / self._get_price_index(year, 'services')
        
        # Export breakdown by product
        export_breakdown = {
            'rmg_textiles': rmg_exports['nominal'],
            'agricultural_products': other_goods_exports * 0.50,
            'leather_goods': other_goods_exports * 0.30,
            'pharmaceuticals': other_goods_exports * 0.20,
            'services': services_exports
        }
        
        return {
            'total_nominal': total_nominal_exports,
            'total_real': total_real_exports,
            'export_breakdown': export_breakdown,
            'rmg_details': rmg_exports
        }
    
    def _calculate_rmg_exports(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate RMG exports specifically."""
        # Base RMG exports (billion BDT)
        base_rmg_exports = 2132  # 82% of total exports
        
        # Global demand and competitiveness factors
        global_demand = self._get_global_rmg_demand_factor(year, quarter)
        competitiveness = self._get_rmg_competitiveness_factor(year)
        
        # Seasonal patterns (Western buying cycles)
        seasonal_factor = 1.0
        if quarter:
            seasonal_factors = {1: 1.1, 2: 0.9, 3: 1.2, 4: 0.8}  # Peak in Q1 and Q3
            seasonal_factor = seasonal_factors[quarter]
        
        years_from_base = year - 2024
        base_growth = 0.04  # 4% base growth
        
        nominal_rmg = base_rmg_exports * (1 + base_growth) ** years_from_base
        nominal_rmg *= global_demand * competitiveness * seasonal_factor
        
        real_rmg = nominal_rmg / self._get_price_index(year, 'exports')
        
        return {
            'nominal': nominal_rmg,
            'real': real_rmg,
            'global_demand_factor': global_demand,
            'competitiveness_factor': competitiveness
        }
    
    def _calculate_imports(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate imports of goods and services.
        
        Heavy dependence on imported inputs for RMG and energy imports.
        """
        # Base imports (billion BDT)
        base_imports = 3500  # Approximate 2024 total imports
        
        # Import growth (linked to economic growth and RMG production)
        years_from_base = year - 2024
        import_growth = 0.055  # 5.5% annual growth
        
        nominal_imports = base_imports * (1 + import_growth) ** years_from_base
        
        # Oil price impact factor
        oil_price_factor = self._get_oil_price_factor(year)
        
        # Import breakdown by category
        import_breakdown = {}
        for category, share in self.import_structure.items():
            category_imports = nominal_imports * share
            
            if category == 'fuel_energy':
                category_imports *= oil_price_factor
            elif category == 'intermediate_goods':
                # Linked to RMG production
                rmg_factor = self._get_global_rmg_demand_factor(year, quarter)
                category_imports *= rmg_factor
            
            import_breakdown[category] = category_imports
        
        # Recalculate total after adjustments
        total_nominal_imports = sum(import_breakdown.values())
        
        # Real imports
        price_index = self._get_price_index(year, 'imports')
        total_real_imports = total_nominal_imports / price_index
        
        return {
            'total_nominal': total_nominal_imports,
            'total_real': total_real_imports,
            'import_breakdown': import_breakdown,
            'oil_price_factor': oil_price_factor
        }
    
    # Helper methods for various factors
    def _get_remittance_impact_factor(self, year: int) -> float:
        """Calculate remittance impact on consumption."""
        # Remittance growth affects consumption
        remittance_factors = {
            2020: 0.95,  # COVID-19 impact
            2021: 1.10,  # Recovery
            2022: 1.05,  # Normalization
            2023: 1.03,  # Steady growth
            2024: 1.02,  # Moderate growth
            2025: 1.02   # Continued growth
        }
        return remittance_factors.get(year, 1.0)
    
    def _get_mfs_consumption_factor(self, year: int) -> float:
        """Calculate MFS impact on consumption patterns."""
        # MFS enables more formal consumption tracking
        mfs_factors = {
            2020: 1.05,
            2021: 1.08,
            2022: 1.10,
            2023: 1.12,
            2024: 1.13,
            2025: 1.14
        }
        return mfs_factors.get(year, 1.0)
    
    def _get_climate_investment_factor(self, year: int) -> float:
        """Calculate climate adaptation investment factor."""
        # Increasing climate adaptation investment
        base_factor = 1.0
        annual_increase = 0.05  # 5% annual increase in climate investment
        years_from_base = year - 2024
        return base_factor * (1 + annual_increase) ** years_from_base
    
    def _get_climate_response_factor(self, year: int, quarter: Optional[int]) -> float:
        """Calculate government climate response expenditure factor."""
        base_factor = 1.0
        
        # Major climate events requiring government response
        climate_events = {
            2020: 1.15,  # Cyclone Amphan
            2022: 1.10   # Severe flooding
        }
        
        return climate_events.get(year, base_factor)
    
    def _get_global_rmg_demand_factor(self, year: int, quarter: Optional[int]) -> float:
        """Get global RMG demand factor."""
        demand_factors = {
            2020: 0.75,  # COVID-19
            2021: 1.10,  # Recovery
            2022: 0.95,  # Supply chain issues
            2023: 1.05,  # Normalization
            2024: 1.02,  # Moderate growth
            2025: 1.03   # Continued growth
        }
        return demand_factors.get(year, 1.0)
    
    def _get_rmg_competitiveness_factor(self, year: int) -> float:
        """Get RMG competitiveness factor."""
        base_competitiveness = 1.0
        annual_wage_increase = 0.08
        productivity_improvement = 0.03
        
        years_from_base = year - 2024
        wage_impact = (1 + annual_wage_increase) ** years_from_base
        productivity_impact = (1 + productivity_improvement) ** years_from_base
        
        return base_competitiveness * (productivity_impact / wage_impact ** 0.7)
    
    def _get_oil_price_factor(self, year: int) -> float:
        """Get oil price impact factor on imports."""
        # Simplified oil price volatility
        oil_factors = {
            2020: 0.70,  # Oil price crash
            2021: 1.20,  # Recovery
            2022: 1.40,  # Ukraine war impact
            2023: 1.10,  # Normalization
            2024: 1.05,  # Moderate prices
            2025: 1.03   # Stable prices
        }
        return oil_factors.get(year, 1.0)
    
    def _get_price_index(self, year: int, component: str = 'overall') -> float:
        """Get price index for deflation."""
        base_inflation = 0.055
        years_from_base = year - self.base_year
        
        component_adjustments = {
            'consumption': 1.0,
            'investment': 0.95,
            'government': 1.05,
            'exports': 0.90,
            'imports': 1.10,
            'services': 1.05,
            'overall': 1.0
        }
        
        adjustment = component_adjustments.get(component, 1.0)
        return (1 + base_inflation * adjustment) ** years_from_base