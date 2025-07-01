"""Income Approach GDP Calculation for Bangladesh.

This module implements the income approach to GDP calculation, measuring
factor payments including labor income, operating surplus, mixed income,
and remittances with Bangladesh-specific income distribution patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class IncomeApproach:
    """Income approach GDP calculation following BBS methodology.
    
    Calculates GDP as sum of factor payments: compensation of employees,
    operating surplus, mixed income, taxes on production, and remittances.
    """
    
    def __init__(self, base_year: int = 2015):
        """Initialize Income Approach calculator.
        
        Args:
            base_year: Base year for constant price calculations
        """
        self.base_year = base_year
        
        # Bangladesh income structure (% of GDP)
        self.income_shares = {
            'compensation_of_employees': 0.35,  # Wages, salaries, benefits
            'operating_surplus': 0.25,          # Corporate profits, rent
            'mixed_income': 0.32,               # Self-employed, informal sector
            'taxes_on_production': 0.08         # Production taxes minus subsidies
        }
        
        # Labor force structure
        self.labor_structure = {
            'formal_sector': 0.15,              # Formal employment
            'informal_sector': 0.85             # Informal employment (dominant)
        }
        
        # Sectoral employment distribution
        self.employment_by_sector = {
            'agriculture': 0.40,                # Still largest employer
            'industry': 0.20,                   # Including RMG
            'services': 0.40                    # Growing service sector
        }
        
        # RMG sector employment details
        self.rmg_employment = {
            'total_workers': 4.0,               # 4 million workers
            'female_share': 0.85,               # 85% female workers
            'rural_origin': 0.90                # 90% from rural areas
        }
        
        logger.info(f"Income Approach initialized with base year {base_year}")
    
    def calculate_gdp(self, 
                     year: int, 
                     quarter: Optional[int] = None,
                     provisional: bool = True) -> Dict:
        """Calculate GDP using income approach.
        
        Args:
            year: Year for calculation
            quarter: Quarter (1-4) for quarterly estimates
            provisional: Whether this is provisional estimate
            
        Returns:
            Dictionary with GDP calculation results
        """
        logger.info(f"Calculating income approach GDP for {year}{'Q' + str(quarter) if quarter else ''}")
        
        # Calculate income components
        compensation = self._calculate_compensation_of_employees(year, quarter)
        operating_surplus = self._calculate_operating_surplus(year, quarter)
        mixed_income = self._calculate_mixed_income(year, quarter)
        taxes_production = self._calculate_taxes_on_production(year, quarter)
        
        # Add remittances (important for Bangladesh)
        remittances = self._calculate_remittance_income(year, quarter)
        
        # Total GDP calculation
        nominal_gdp = (compensation['total_nominal'] + 
                      operating_surplus['total_nominal'] + 
                      mixed_income['total_nominal'] + 
                      taxes_production['total_nominal'] + 
                      remittances['total_nominal'])
        
        real_gdp = (compensation['total_real'] + 
                   operating_surplus['total_real'] + 
                   mixed_income['total_real'] + 
                   taxes_production['total_real'] + 
                   remittances['total_real'])
        
        return {
            'nominal_gdp': nominal_gdp,
            'real_gdp': real_gdp,
            'gdp_deflator': (nominal_gdp / real_gdp) * 100,
            'income_components': {
                'compensation_of_employees': compensation,
                'operating_surplus': operating_surplus,
                'mixed_income': mixed_income,
                'taxes_on_production': taxes_production,
                'remittances': remittances
            },
            'income_distribution': {
                'labor_share': (compensation['total_nominal'] + mixed_income['total_nominal']) / nominal_gdp,
                'capital_share': operating_surplus['total_nominal'] / nominal_gdp,
                'government_share': taxes_production['total_nominal'] / nominal_gdp,
                'remittance_share': remittances['total_nominal'] / nominal_gdp
            },
            'calculation_metadata': {
                'year': year,
                'quarter': quarter,
                'provisional': provisional,
                'base_year': self.base_year
            }
        }
    
    def _calculate_compensation_of_employees(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate compensation of employees (wages, salaries, benefits).
        
        Includes formal sector wages with special attention to RMG sector.
        """
        # Base compensation (billion BDT)
        base_compensation = 6100  # Approximate 2024 total compensation
        
        # Wage growth factors
        years_from_base = year - 2024
        nominal_wage_growth = 0.08  # 8% annual nominal wage growth
        
        # Calculate total compensation
        total_compensation = base_compensation * (1 + nominal_wage_growth) ** years_from_base
        
        # Sectoral breakdown
        sectoral_compensation = {}
        
        # RMG sector compensation (detailed due to importance)
        rmg_compensation = self._calculate_rmg_worker_compensation(year, quarter)
        
        # Other manufacturing
        other_mfg_compensation = total_compensation * 0.15
        
        # Services sector (formal)
        services_compensation = total_compensation * 0.40
        
        # Government sector
        government_compensation = total_compensation * 0.25
        
        # Agriculture (formal workers only)
        agriculture_compensation = total_compensation * 0.05
        
        # Adjust RMG share
        rmg_share = rmg_compensation['total_compensation'] / total_compensation
        adjustment_factor = 0.15 / rmg_share  # Target 15% share for RMG
        
        sectoral_compensation = {
            'rmg_manufacturing': rmg_compensation['total_compensation'] * adjustment_factor,
            'other_manufacturing': other_mfg_compensation,
            'services': services_compensation,
            'government': government_compensation,
            'agriculture': agriculture_compensation
        }
        
        # Recalculate total
        total_nominal = sum(sectoral_compensation.values())
        
        # Real compensation
        price_index = self._get_price_index(year, 'wages')
        total_real = total_nominal / price_index
        
        # Employment statistics
        total_formal_employment = self._calculate_formal_employment(year)
        average_wage = total_nominal / total_formal_employment * 1000  # BDT per worker
        
        return {
            'total_nominal': total_nominal,
            'total_real': total_real,
            'sectoral_breakdown': sectoral_compensation,
            'employment_statistics': {
                'total_formal_employment_millions': total_formal_employment,
                'average_annual_wage_bdt': average_wage,
                'rmg_worker_details': rmg_compensation
            }
        }
    
    def _calculate_rmg_worker_compensation(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate RMG sector worker compensation in detail."""
        # RMG employment and wages
        total_rmg_workers = 4.0  # 4 million workers
        
        # Wage structure
        base_monthly_wage_2024 = 12500  # BDT per month (2024 minimum wage)
        years_from_base = year - 2024
        wage_growth = 0.08  # 8% annual wage increase
        
        current_monthly_wage = base_monthly_wage_2024 * (1 + wage_growth) ** years_from_base
        annual_wage_per_worker = current_monthly_wage * 12
        
        # Total RMG compensation
        total_rmg_compensation = (total_rmg_workers * annual_wage_per_worker) / 1e9  # Convert to billion BDT
        
        # Gender and skill breakdown
        female_workers = total_rmg_workers * self.rmg_employment['female_share']
        male_workers = total_rmg_workers - female_workers
        
        # Skill categories
        skilled_workers = total_rmg_workers * 0.20
        semi_skilled_workers = total_rmg_workers * 0.50
        unskilled_workers = total_rmg_workers * 0.30
        
        return {
            'total_compensation': total_rmg_compensation,
            'total_workers_millions': total_rmg_workers,
            'average_monthly_wage_bdt': current_monthly_wage,
            'worker_breakdown': {
                'female_workers_millions': female_workers,
                'male_workers_millions': male_workers,
                'skilled_workers_millions': skilled_workers,
                'semi_skilled_workers_millions': semi_skilled_workers,
                'unskilled_workers_millions': unskilled_workers
            }
        }
    
    def _calculate_operating_surplus(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate operating surplus (corporate profits, rent, interest).
        
        Includes profits from formal businesses, rent, and capital income.
        """
        # Base operating surplus (billion BDT)
        base_surplus = 4350  # Approximate 2024 operating surplus
        
        # Profit growth (linked to economic growth)
        years_from_base = year - 2024
        profit_growth = 0.06  # 6% annual growth
        
        total_surplus = base_surplus * (1 + profit_growth) ** years_from_base
        
        # Sectoral breakdown of operating surplus
        sectoral_surplus = {
            'rmg_manufacturing': total_surplus * 0.25,      # RMG factory profits
            'financial_services': total_surplus * 0.20,     # Banking, MFS profits
            'real_estate': total_surplus * 0.15,            # Property rent
            'telecommunications': total_surplus * 0.10,     # Telecom profits
            'other_manufacturing': total_surplus * 0.10,    # Other industry profits
            'trade_commerce': total_surplus * 0.10,         # Trading profits
            'transport_logistics': total_surplus * 0.05,    # Transport profits
            'other_services': total_surplus * 0.05          # Other service profits
        }
        
        # Adjust for economic cycles
        cycle_factor = self._get_business_cycle_factor(year, quarter)
        total_nominal = total_surplus * cycle_factor
        
        # Apply cycle factor to sectoral breakdown
        for sector in sectoral_surplus:
            sectoral_surplus[sector] *= cycle_factor
        
        # Real operating surplus
        price_index = self._get_price_index(year, 'profits')
        total_real = total_nominal / price_index
        
        return {
            'total_nominal': total_nominal,
            'total_real': total_real,
            'sectoral_breakdown': sectoral_surplus,
            'business_cycle_factor': cycle_factor
        }
    
    def _calculate_mixed_income(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate mixed income (self-employed, informal sector income).
        
        This is particularly important for Bangladesh given the large informal economy.
        """
        # Base mixed income (billion BDT)
        base_mixed_income = 5570  # Approximate 2024 mixed income
        
        # Growth in informal sector
        years_from_base = year - 2024
        informal_growth = 0.05  # 5% annual growth (slower than formal sector)
        
        total_mixed_income = base_mixed_income * (1 + informal_growth) ** years_from_base
        
        # Remittance impact on informal sector
        remittance_factor = self._get_remittance_impact_on_informal_sector(year)
        total_mixed_income *= remittance_factor
        
        # Sectoral breakdown of mixed income
        mixed_income_sectors = {
            'agriculture_smallholder': total_mixed_income * 0.45,    # Small farmers
            'retail_trade': total_mixed_income * 0.20,              # Small shops, vendors
            'transport_services': total_mixed_income * 0.15,        # Rickshaw, CNG drivers
            'construction_informal': total_mixed_income * 0.10,     # Informal construction
            'manufacturing_cottage': total_mixed_income * 0.05,     # Cottage industries
            'personal_services': total_mixed_income * 0.05          # Domestic workers, etc.
        }
        
        # Urban-rural breakdown
        urban_mixed_income = total_mixed_income * 0.35  # Urban informal sector
        rural_mixed_income = total_mixed_income * 0.65  # Rural informal sector
        
        # Real mixed income
        price_index = self._get_price_index(year, 'informal')
        total_real = total_mixed_income / price_index
        
        # Estimate informal employment
        informal_employment = self._calculate_informal_employment(year)
        average_informal_income = total_mixed_income / informal_employment * 1000  # BDT per worker
        
        return {
            'total_nominal': total_mixed_income,
            'total_real': total_real,
            'sectoral_breakdown': mixed_income_sectors,
            'geographic_breakdown': {
                'urban_mixed_income': urban_mixed_income,
                'rural_mixed_income': rural_mixed_income
            },
            'employment_statistics': {
                'informal_employment_millions': informal_employment,
                'average_annual_income_bdt': average_informal_income
            },
            'remittance_impact_factor': remittance_factor
        }
    
    def _calculate_taxes_on_production(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate taxes on production and imports minus subsidies."""
        # Base taxes on production (billion BDT)
        base_taxes = 1390  # Approximate 2024 production taxes
        
        # Tax collection growth
        years_from_base = year - 2024
        tax_growth = 0.07  # 7% annual growth (tax administration improvement)
        
        total_taxes = base_taxes * (1 + tax_growth) ** years_from_base
        
        # Tax breakdown
        tax_breakdown = {
            'vat_on_domestic_goods': total_taxes * 0.40,        # VAT
            'import_duties': total_taxes * 0.25,                # Customs duties
            'excise_taxes': total_taxes * 0.15,                 # Excise on specific goods
            'other_production_taxes': total_taxes * 0.20        # Other taxes
        }
        
        # Subsidies (reduce total)
        subsidies = self._calculate_subsidies(year)
        net_taxes = total_taxes - subsidies
        
        # Real taxes
        price_index = self._get_price_index(year, 'overall')
        total_real = net_taxes / price_index
        
        return {
            'total_nominal': net_taxes,
            'total_real': total_real,
            'gross_taxes': total_taxes,
            'subsidies': subsidies,
            'tax_breakdown': tax_breakdown
        }
    
    def _calculate_remittance_income(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Calculate remittance income impact on GDP.
        
        Bangladesh receives significant remittances that boost domestic income.
        """
        # Base remittances (billion USD)
        base_remittances_usd = 22.1  # 2024 remittances in USD
        
        # Remittance growth
        years_from_base = year - 2024
        remittance_growth = 0.04  # 4% annual growth
        
        remittances_usd = base_remittances_usd * (1 + remittance_growth) ** years_from_base
        
        # Convert to BDT
        exchange_rate = self._get_exchange_rate(year)
        remittances_bdt = remittances_usd * exchange_rate
        
        # Seasonal patterns
        seasonal_factor = 1.0
        if quarter:
            # Q1: Post-Eid, Q2: Pre-Eid savings, Q3: Eid peak, Q4: Post-harvest
            seasonal_factors = {1: 0.90, 2: 1.05, 3: 1.20, 4: 0.85}
            seasonal_factor = seasonal_factors[quarter]
        
        total_remittances = remittances_bdt * seasonal_factor
        
        # Channel breakdown (formal vs informal)
        formal_channels = total_remittances * 0.70  # Banks, MFS
        informal_channels = total_remittances * 0.30  # Hundi system
        
        # Geographic distribution
        urban_remittances = total_remittances * 0.30
        rural_remittances = total_remittances * 0.70
        
        # Real remittances
        price_index = self._get_price_index(year, 'overall')
        total_real = total_remittances / price_index
        
        return {
            'total_nominal': total_remittances,
            'total_real': total_real,
            'remittances_usd_billion': remittances_usd,
            'exchange_rate_bdt_usd': exchange_rate,
            'channel_breakdown': {
                'formal_channels': formal_channels,
                'informal_channels': informal_channels
            },
            'geographic_distribution': {
                'urban_remittances': urban_remittances,
                'rural_remittances': rural_remittances
            }
        }
    
    def _calculate_subsidies(self, year: int) -> float:
        """Calculate government subsidies."""
        # Major subsidies in Bangladesh
        base_subsidies = 350  # Billion BDT
        years_from_base = year - 2024
        subsidy_growth = 0.05  # 5% annual growth
        
        return base_subsidies * (1 + subsidy_growth) ** years_from_base
    
    def _calculate_formal_employment(self, year: int) -> float:
        """Calculate formal sector employment in millions."""
        base_formal_employment = 10.0  # 10 million formal workers in 2024
        years_from_base = year - 2024
        employment_growth = 0.03  # 3% annual growth
        
        return base_formal_employment * (1 + employment_growth) ** years_from_base
    
    def _calculate_informal_employment(self, year: int) -> float:
        """Calculate informal sector employment in millions."""
        base_informal_employment = 56.7  # 56.7 million informal workers in 2024
        years_from_base = year - 2024
        employment_growth = 0.02  # 2% annual growth (slower than formal)
        
        return base_informal_employment * (1 + employment_growth) ** years_from_base
    
    def _get_remittance_impact_on_informal_sector(self, year: int) -> float:
        """Calculate how remittances impact informal sector income."""
        # Remittances boost informal sector through consumption linkages
        remittance_factors = {
            2020: 0.95,  # COVID-19 impact
            2021: 1.08,  # Recovery boost
            2022: 1.05,  # Normalization
            2023: 1.03,  # Steady impact
            2024: 1.02,  # Moderate impact
            2025: 1.02   # Continued impact
        }
        return remittance_factors.get(year, 1.0)
    
    def _get_business_cycle_factor(self, year: int, quarter: Optional[int] = None) -> float:
        """Calculate business cycle impact on profits."""
        # Economic cycle effects on operating surplus
        cycle_factors = {
            2020: 0.85,  # COVID-19 recession
            2021: 1.15,  # Recovery boom
            2022: 0.95,  # Normalization
            2023: 1.05,  # Growth
            2024: 1.02,  # Moderate growth
            2025: 1.03   # Continued growth
        }
        return cycle_factors.get(year, 1.0)
    
    def _get_exchange_rate(self, year: int) -> float:
        """Get BDT/USD exchange rate."""
        base_rate_2024 = 110.0  # BDT per USD
        annual_depreciation = 0.03  # 3% annual depreciation
        years_from_base = year - 2024
        
        return base_rate_2024 * (1 + annual_depreciation) ** years_from_base
    
    def _get_price_index(self, year: int, component: str = 'overall') -> float:
        """Get price index for deflation."""
        base_inflation = 0.055  # 5.5% average inflation
        years_from_base = year - self.base_year
        
        component_adjustments = {
            'wages': 1.0,
            'profits': 0.95,
            'informal': 1.1,    # Higher inflation for informal sector
            'overall': 1.0
        }
        
        adjustment = component_adjustments.get(component, 1.0)
        return (1 + base_inflation * adjustment) ** years_from_base