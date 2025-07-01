"""Production Approach GDP Calculation for Bangladesh.

This module implements the production approach to GDP calculation, measuring
value-added across Bangladesh's key economic sectors including RMG manufacturing,
agriculture, services, and construction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProductionApproach:
    """Production approach GDP calculation following BBS methodology.
    
    Calculates GDP as sum of value-added across all economic sectors,
    with special attention to Bangladesh's unique economic structure.
    """
    
    def __init__(self, base_year: int = 2015):
        """Initialize Production Approach calculator.
        
        Args:
            base_year: Base year for constant price calculations
        """
        self.base_year = base_year
        
        # Bangladesh sector structure (% of GDP in base year)
        self.sector_weights = {
            'agriculture': 0.15,  # Agriculture, forestry, fishing
            'industry': 0.32,     # Manufacturing, construction, utilities
            'services': 0.53      # Services, government, other
        }
        
        # Industry sub-sectors (within 32% industry share)
        self.industry_subsectors = {
            'rmg_manufacturing': 0.57,    # RMG dominance in manufacturing
            'other_manufacturing': 0.23,  # Food, pharmaceuticals, etc.
            'construction': 0.15,         # Infrastructure, housing
            'utilities': 0.05            # Power, gas, water
        }
        
        # Services sub-sectors (within 53% services share)
        self.services_subsectors = {
            'wholesale_retail': 0.25,    # Trade
            'transport': 0.15,           # Transport, storage
            'financial_services': 0.12,  # Banking, insurance, MFS
            'real_estate': 0.10,         # Housing services
            'government': 0.18,          # Public administration
            'other_services': 0.20       # Education, health, personal services
        }
        
        logger.info(f"Production Approach initialized with base year {base_year}")
    
    def calculate_gdp(self, 
                     year: int, 
                     quarter: Optional[int] = None,
                     provisional: bool = True) -> Dict:
        """Calculate GDP using production approach.
        
        Args:
            year: Year for calculation
            quarter: Quarter (1-4) for quarterly estimates
            provisional: Whether this is provisional estimate
            
        Returns:
            Dictionary with GDP calculation results
        """
        logger.info(f"Calculating production approach GDP for {year}{'Q' + str(quarter) if quarter else ''}")
        
        # Calculate sectoral value-added
        agriculture_va = self._calculate_agriculture_va(year, quarter)
        industry_va = self._calculate_industry_va(year, quarter)
        services_va = self._calculate_services_va(year, quarter)
        
        # Sum to get total value-added
        total_va_nominal = agriculture_va['nominal'] + industry_va['nominal'] + services_va['nominal']
        total_va_real = agriculture_va['real'] + industry_va['real'] + services_va['real']
        
        # Add taxes on products minus subsidies
        taxes_on_products = self._calculate_taxes_on_products(total_va_nominal, year)
        subsidies = self._calculate_subsidies(total_va_nominal, year)
        
        # Final GDP calculation
        nominal_gdp = total_va_nominal + taxes_on_products - subsidies
        real_gdp = total_va_real + (taxes_on_products - subsidies) / self._get_price_index(year)
        
        return {
            'nominal_gdp': nominal_gdp,
            'real_gdp': real_gdp,
            'gdp_deflator': (nominal_gdp / real_gdp) * 100,
            'sectoral_breakdown': {
                'agriculture': agriculture_va,
                'industry': industry_va,
                'services': services_va
            },
            'taxes_on_products': taxes_on_products,
            'subsidies': subsidies,
            'total_value_added': {
                'nominal': total_va_nominal,
                'real': total_va_real
            },
            'calculation_metadata': {
                'year': year,
                'quarter': quarter,
                'provisional': provisional,
                'base_year': self.base_year
            }
        }
    
    def _calculate_agriculture_va(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate agriculture sector value-added.
        
        Includes rice production, other crops, livestock, forestry, and fishing.
        Accounts for monsoon seasonality and climate impacts.
        """
        # Base agriculture GDP (billion BDT)
        base_ag_gdp = 2500  # Approximate 2024 agriculture GDP
        
        # Growth factors
        trend_growth = 0.025  # 2.5% annual trend growth
        years_from_base = year - 2024
        
        # Climate impact factor (varies by year and season)
        climate_factor = self._get_climate_impact_factor(year, quarter, 'agriculture')
        
        # Seasonal adjustment for quarterly data
        seasonal_factor = 1.0
        if quarter:
            # Agriculture seasonality in Bangladesh
            seasonal_factors = {1: 0.8, 2: 1.3, 3: 1.1, 4: 0.8}  # Boro harvest in Q2
            seasonal_factor = seasonal_factors[quarter]
        
        # Calculate nominal value-added
        nominal_va = base_ag_gdp * (1 + trend_growth) ** years_from_base
        nominal_va *= climate_factor * seasonal_factor
        
        # Real value-added (deflated)
        price_index = self._get_price_index(year, 'agriculture')
        real_va = nominal_va / price_index
        
        return {
            'nominal': nominal_va,
            'real': real_va,
            'subsectors': {
                'rice': nominal_va * 0.35,
                'other_crops': nominal_va * 0.25,
                'livestock': nominal_va * 0.20,
                'forestry': nominal_va * 0.10,
                'fishing': nominal_va * 0.10
            },
            'climate_impact_factor': climate_factor
        }
    
    def _calculate_industry_va(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate industry sector value-added.
        
        Dominated by RMG manufacturing, also includes other manufacturing,
        construction, and utilities.
        """
        # Base industry GDP (billion BDT)
        base_industry_gdp = 5500  # Approximate 2024 industry GDP
        
        # Calculate RMG manufacturing separately due to its importance
        rmg_va = self._calculate_rmg_manufacturing(year, quarter)
        
        # Other manufacturing
        other_mfg_base = base_industry_gdp * self.industry_subsectors['other_manufacturing']
        other_mfg_growth = 0.06  # 6% annual growth
        years_from_base = year - 2024
        other_mfg_va = other_mfg_base * (1 + other_mfg_growth) ** years_from_base
        
        # Construction (infrastructure development)
        construction_base = base_industry_gdp * self.industry_subsectors['construction']
        construction_growth = 0.08  # 8% annual growth (infrastructure boom)
        construction_va = construction_base * (1 + construction_growth) ** years_from_base
        
        # Climate impact on construction (cyclone damage/reconstruction)
        climate_factor = self._get_climate_impact_factor(year, quarter, 'construction')
        construction_va *= climate_factor
        
        # Utilities
        utilities_base = base_industry_gdp * self.industry_subsectors['utilities']
        utilities_growth = 0.07  # 7% annual growth (power sector expansion)
        utilities_va = utilities_base * (1 + utilities_growth) ** years_from_base
        
        # Total industry value-added
        total_nominal = rmg_va['nominal'] + other_mfg_va + construction_va + utilities_va
        total_real = rmg_va['real'] + (other_mfg_va + construction_va + utilities_va) / self._get_price_index(year, 'industry')
        
        return {
            'nominal': total_nominal,
            'real': total_real,
            'subsectors': {
                'rmg_manufacturing': rmg_va,
                'other_manufacturing': other_mfg_va,
                'construction': construction_va,
                'utilities': utilities_va
            }
        }
    
    def _calculate_rmg_manufacturing(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate RMG manufacturing value-added.
        
        Models the complex dynamics of Bangladesh's dominant export industry.
        """
        # Base RMG value-added (billion BDT)
        base_rmg_va = 3135  # 57% of industry sector
        
        # Global demand factors
        global_demand_factor = self._get_global_rmg_demand_factor(year, quarter)
        
        # Competitiveness factor (vs Vietnam, Cambodia)
        competitiveness_factor = self._get_rmg_competitiveness_factor(year)
        
        # Seasonal patterns (Western buying seasons)
        seasonal_factor = 1.0
        if quarter:
            # Q1: Spring orders, Q2: Summer prep, Q3: Fall/Winter orders, Q4: Holiday season
            seasonal_factors = {1: 1.1, 2: 0.9, 3: 1.2, 4: 0.8}
            seasonal_factor = seasonal_factors[quarter]
        
        # Calculate nominal value-added
        years_from_base = year - 2024
        base_growth = 0.04  # 4% base annual growth
        
        nominal_va = base_rmg_va * (1 + base_growth) ** years_from_base
        nominal_va *= global_demand_factor * competitiveness_factor * seasonal_factor
        
        # Real value-added
        real_va = nominal_va / self._get_price_index(year, 'manufacturing')
        
        return {
            'nominal': nominal_va,
            'real': real_va,
            'global_demand_factor': global_demand_factor,
            'competitiveness_factor': competitiveness_factor,
            'employment_millions': 4.0 * (nominal_va / base_rmg_va),  # Employment scales with output
            'export_share': 0.82  # 82% of merchandise exports
        }
    
    def _calculate_services_va(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate services sector value-added.
        
        Includes traditional services plus rapidly growing digital financial services.
        """
        # Base services GDP (billion BDT)
        base_services_gdp = 9200  # Approximate 2024 services GDP
        
        # Calculate subsectors
        subsector_vas = {}
        
        for subsector, weight in self.services_subsectors.items():
            base_va = base_services_gdp * weight
            
            if subsector == 'financial_services':
                # Special treatment for financial services (MFS growth)
                growth_rate = 0.12  # 12% annual growth due to MFS expansion
                mfs_factor = self._get_mfs_growth_factor(year)
            elif subsector == 'transport':
                growth_rate = 0.05  # 5% annual growth
                mfs_factor = 1.0
            elif subsector == 'wholesale_retail':
                growth_rate = 0.06  # 6% annual growth (e-commerce boost)
                mfs_factor = 1.0
            else:
                growth_rate = 0.04  # 4% default growth
                mfs_factor = 1.0
            
            years_from_base = year - 2024
            subsector_va = base_va * (1 + growth_rate) ** years_from_base * mfs_factor
            subsector_vas[subsector] = subsector_va
        
        # Total services value-added
        total_nominal = sum(subsector_vas.values())
        total_real = total_nominal / self._get_price_index(year, 'services')
        
        return {
            'nominal': total_nominal,
            'real': total_real,
            'subsectors': subsector_vas,
            'mfs_penetration': self._get_mfs_penetration_rate(year)
        }
    
    def _get_climate_impact_factor(self, year: int, quarter: Optional[int], sector: str) -> float:
        """Calculate climate impact factor for given sector and period."""
        # Simplified climate impact model
        base_factor = 1.0
        
        # Historical major climate events
        climate_events = {
            2007: 0.85,  # Cyclone Sidr
            2009: 0.90,  # Severe flooding
            2020: 0.88,  # Cyclone Amphan
            2022: 0.92   # Flooding
        }
        
        if year in climate_events:
            base_factor = climate_events[year]
        
        # Sector-specific vulnerability
        if sector == 'agriculture':
            vulnerability = 1.2  # Higher vulnerability
        elif sector == 'construction':
            vulnerability = 1.1  # Moderate vulnerability (damage + reconstruction)
        else:
            vulnerability = 1.0
        
        return base_factor ** vulnerability
    
    def _get_global_rmg_demand_factor(self, year: int, quarter: Optional[int]) -> float:
        """Calculate global demand factor for RMG exports."""
        # Global economic conditions affecting RMG demand
        demand_factors = {
            2020: 0.75,  # COVID-19 impact
            2021: 1.10,  # Recovery
            2022: 0.95,  # Supply chain issues
            2023: 1.05,  # Normalization
            2024: 1.02,  # Moderate growth
            2025: 1.03   # Continued growth
        }
        
        return demand_factors.get(year, 1.0)
    
    def _get_rmg_competitiveness_factor(self, year: int) -> float:
        """Calculate RMG competitiveness factor vs other countries."""
        # Bangladesh competitiveness trends
        base_competitiveness = 1.0
        
        # Factors: wage increases, productivity improvements, compliance costs
        annual_wage_increase = 0.08  # 8% annual wage increase
        productivity_improvement = 0.03  # 3% annual productivity gain
        
        years_from_base = year - 2024
        wage_impact = (1 + annual_wage_increase) ** years_from_base
        productivity_impact = (1 + productivity_improvement) ** years_from_base
        
        # Net competitiveness (productivity gains offset some wage increases)
        competitiveness = base_competitiveness * (productivity_impact / wage_impact ** 0.7)
        
        return competitiveness
    
    def _get_mfs_growth_factor(self, year: int) -> float:
        """Calculate Mobile Financial Services growth factor."""
        # MFS rapid expansion in Bangladesh
        mfs_factors = {
            2020: 1.25,  # COVID-19 acceleration
            2021: 1.20,  # Continued growth
            2022: 1.15,  # Maturation
            2023: 1.12,  # Steady growth
            2024: 1.10,  # Moderate growth
            2025: 1.08   # Slower growth as market matures
        }
        
        return mfs_factors.get(year, 1.0)
    
    def _get_mfs_penetration_rate(self, year: int) -> float:
        """Get MFS penetration rate for given year."""
        # MFS account penetration (% of adult population)
        penetration_rates = {
            2020: 0.45,
            2021: 0.52,
            2022: 0.58,
            2023: 0.63,
            2024: 0.67,
            2025: 0.70
        }
        
        return penetration_rates.get(year, 0.70)
    
    def _calculate_taxes_on_products(self, total_va: float, year: int) -> float:
        """Calculate taxes on products (VAT, customs duties, etc.)."""
        # Tax-to-GDP ratio in Bangladesh (approximately 9%)
        tax_rate = 0.09
        return total_va * tax_rate
    
    def _calculate_subsidies(self, total_va: float, year: int) -> float:
        """Calculate government subsidies."""
        # Subsidy rate (fuel, fertilizer, electricity subsidies)
        subsidy_rate = 0.02  # 2% of GDP
        return total_va * subsidy_rate
    
    def _get_price_index(self, year: int, sector: str = 'overall') -> float:
        """Get price index for deflation to constant prices."""
        # Simplified price index (base year = 1.0)
        base_inflation = 0.055  # 5.5% average inflation
        years_from_base = year - self.base_year
        
        # Sector-specific inflation adjustments
        sector_adjustments = {
            'agriculture': 1.1,    # Higher food inflation
            'manufacturing': 0.9,  # Lower manufacturing inflation
            'industry': 0.95,      # Moderate industrial inflation
            'services': 1.05,      # Moderate services inflation
            'overall': 1.0
        }
        
        adjustment = sector_adjustments.get(sector, 1.0)
        return (1 + base_inflation * adjustment) ** years_from_base