"""Main GDP Calculator for Bangladesh Economic Simulation.

This module implements the comprehensive GDP calculation system that coordinates
all three approaches (Production, Expenditure, Income) following Bangladesh
Bureau of Statistics (BBS) methodology.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import logging

from .production_approach import ProductionApproach
from .expenditure_approach import ExpenditureApproach
from .income_approach import IncomeApproach
from .bbs_methodology import BBSMethodology

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDPCalculator:
    """Main GDP Calculator implementing BBS methodology for Bangladesh.
    
    This class coordinates the three GDP calculation approaches and provides
    comprehensive economic output measurement following Bangladesh-specific
    methodological frameworks.
    """
    
    def __init__(self, base_year: int = 2015, current_year: int = 2024):
        """Initialize GDP Calculator.
        
        Args:
            base_year: Base year for constant price calculations (BBS uses 2015-16)
            current_year: Current year for calculations
        """
        self.base_year = base_year
        self.current_year = current_year
        
        # Initialize calculation approaches
        self.production = ProductionApproach(base_year)
        self.expenditure = ExpenditureApproach(base_year)
        self.income = IncomeApproach(base_year)
        self.bbs_methodology = BBSMethodology()
        
        # GDP calculation results storage
        self.results = {
            'production': {},
            'expenditure': {},
            'income': {},
            'reconciled': {},
            'metadata': {}
        }
        
        logger.info(f"GDP Calculator initialized for base year {base_year}, current year {current_year}")
    
    def calculate_gdp_all_approaches(self, 
                                   year: int,
                                   quarter: Optional[int] = None,
                                   provisional: bool = True) -> Dict:
        """Calculate GDP using all three approaches.
        
        Args:
            year: Year for calculation
            quarter: Quarter (1-4) for quarterly estimates, None for annual
            provisional: Whether this is provisional or final estimate
            
        Returns:
            Dictionary containing GDP estimates from all approaches
        """
        logger.info(f"Calculating GDP for {year}{'Q' + str(quarter) if quarter else ''} ({'Provisional' if provisional else 'Final'})")
        
        # Production Approach
        production_gdp = self.production.calculate_gdp(
            year=year, 
            quarter=quarter,
            provisional=provisional
        )
        
        # Expenditure Approach  
        expenditure_gdp = self.expenditure.calculate_gdp(
            year=year,
            quarter=quarter, 
            provisional=provisional
        )
        
        # Income Approach
        income_gdp = self.income.calculate_gdp(
            year=year,
            quarter=quarter,
            provisional=provisional
        )
        
        # Statistical discrepancy and reconciliation
        reconciled_gdp = self._reconcile_approaches(
            production_gdp, expenditure_gdp, income_gdp
        )
        
        # Store results
        period_key = f"{year}{'Q' + str(quarter) if quarter else ''}"
        self.results['production'][period_key] = production_gdp
        self.results['expenditure'][period_key] = expenditure_gdp
        self.results['income'][period_key] = income_gdp
        self.results['reconciled'][period_key] = reconciled_gdp
        
        return {
            'period': period_key,
            'production_approach': production_gdp,
            'expenditure_approach': expenditure_gdp,
            'income_approach': income_gdp,
            'reconciled': reconciled_gdp,
            'metadata': {
                'calculation_date': datetime.now(),
                'provisional': provisional,
                'base_year': self.base_year
            }
        }
    
    def _reconcile_approaches(self, 
                            production: Dict, 
                            expenditure: Dict, 
                            income: Dict) -> Dict:
        """Reconcile GDP estimates from three approaches.
        
        Following BBS practice, production approach is typically used as benchmark
        with statistical discrepancy calculated.
        
        Args:
            production: Production approach results
            expenditure: Expenditure approach results  
            income: Income approach results
            
        Returns:
            Reconciled GDP estimate with statistical discrepancy
        """
        # Extract nominal GDP values
        prod_gdp = production['nominal_gdp']
        exp_gdp = expenditure['nominal_gdp']
        inc_gdp = income['nominal_gdp']
        
        # BBS typically uses production approach as benchmark
        benchmark_gdp = prod_gdp
        
        # Calculate statistical discrepancies
        exp_discrepancy = exp_gdp - benchmark_gdp
        inc_discrepancy = inc_gdp - benchmark_gdp
        
        # Calculate average discrepancy
        avg_discrepancy = (exp_discrepancy + inc_discrepancy) / 2
        
        # Reconciled estimate (weighted average with production approach priority)
        reconciled_nominal = (0.5 * prod_gdp + 0.25 * exp_gdp + 0.25 * inc_gdp)
        
        return {
            'nominal_gdp': reconciled_nominal,
            'real_gdp': production['real_gdp'],  # Use production approach for real GDP
            'gdp_deflator': reconciled_nominal / production['real_gdp'] * 100,
            'statistical_discrepancy': {
                'expenditure_vs_production': exp_discrepancy,
                'income_vs_production': inc_discrepancy,
                'average_discrepancy': avg_discrepancy,
                'discrepancy_percent': (avg_discrepancy / benchmark_gdp) * 100
            },
            'approach_weights': {
                'production': 0.5,
                'expenditure': 0.25,
                'income': 0.25
            }
        }
    
    def calculate_growth_rates(self, periods: List[str]) -> Dict:
        """Calculate GDP growth rates for specified periods.
        
        Args:
            periods: List of period keys (e.g., ['2023', '2024Q1', '2024Q2'])
            
        Returns:
            Dictionary with growth rate calculations
        """
        growth_rates = {}
        
        for i in range(1, len(periods)):
            current_period = periods[i]
            previous_period = periods[i-1]
            
            if (current_period in self.results['reconciled'] and 
                previous_period in self.results['reconciled']):
                
                current_gdp = self.results['reconciled'][current_period]['real_gdp']
                previous_gdp = self.results['reconciled'][previous_period]['real_gdp']
                
                growth_rate = ((current_gdp - previous_gdp) / previous_gdp) * 100
                
                growth_rates[f"{previous_period}_to_{current_period}"] = {
                    'growth_rate_percent': growth_rate,
                    'current_gdp': current_gdp,
                    'previous_gdp': previous_gdp
                }
        
        return growth_rates
    
    def generate_bbs_report(self, year: int, quarter: Optional[int] = None) -> Dict:
        """Generate BBS-style GDP report.
        
        Args:
            year: Year for report
            quarter: Quarter for quarterly report
            
        Returns:
            Formatted report following BBS release format
        """
        period_key = f"{year}{'Q' + str(quarter) if quarter else ''}"
        
        if period_key not in self.results['reconciled']:
            raise ValueError(f"No GDP calculation found for period {period_key}")
        
        result = self.results['reconciled'][period_key]
        
        # Calculate per capita GDP (assuming population data available)
        population = self._get_population_estimate(year)
        per_capita_gdp = (result['nominal_gdp'] * 1e9) / population  # Convert to BDT per person
        per_capita_usd = per_capita_gdp / self._get_exchange_rate(year)  # Convert to USD
        
        report = {
            'title': f"Bangladesh GDP Estimates {year}{'Q' + str(quarter) if quarter else ''}",
            'release_date': datetime.now().strftime('%Y-%m-%d'),
            'period': period_key,
            'gdp_estimates': {
                'nominal_gdp_billion_bdt': round(result['nominal_gdp'], 2),
                'real_gdp_billion_bdt': round(result['real_gdp'], 2),
                'gdp_deflator': round(result['gdp_deflator'], 2),
                'per_capita_gdp_bdt': round(per_capita_gdp, 0),
                'per_capita_gdp_usd': round(per_capita_usd, 0)
            },
            'sectoral_breakdown': self.results['production'][period_key]['sectoral_breakdown'],
            'expenditure_components': self.results['expenditure'][period_key]['components'],
            'statistical_quality': {
                'statistical_discrepancy_percent': round(
                    result['statistical_discrepancy']['discrepancy_percent'], 3
                ),
                'data_quality_score': self._calculate_data_quality_score(period_key)
            },
            'methodology_notes': self.bbs_methodology.get_methodology_notes()
        }
        
        return report
    
    def _get_population_estimate(self, year: int) -> float:
        """Get population estimate for given year.
        
        Args:
            year: Year for population estimate
            
        Returns:
            Population estimate in millions
        """
        # Bangladesh population estimates (in millions)
        base_population_2020 = 164.7
        annual_growth_rate = 0.011  # 1.1% annual growth
        
        population = base_population_2020 * ((1 + annual_growth_rate) ** (year - 2020))
        return population * 1e6  # Convert to actual population
    
    def _get_exchange_rate(self, year: int) -> float:
        """Get BDT/USD exchange rate for given year.
        
        Args:
            year: Year for exchange rate
            
        Returns:
            BDT per USD exchange rate
        """
        # Simplified exchange rate model (actual implementation would use real data)
        base_rate_2020 = 84.8
        annual_depreciation = 0.03  # 3% annual depreciation
        
        return base_rate_2020 * ((1 + annual_depreciation) ** (year - 2020))
    
    def _calculate_data_quality_score(self, period_key: str) -> float:
        """Calculate data quality score for the period.
        
        Args:
            period_key: Period identifier
            
        Returns:
            Data quality score (0-100)
        """
        # Simplified data quality assessment
        discrepancy = abs(self.results['reconciled'][period_key]['statistical_discrepancy']['discrepancy_percent'])
        
        # Lower discrepancy = higher quality
        if discrepancy < 1.0:
            return 95.0
        elif discrepancy < 2.0:
            return 85.0
        elif discrepancy < 3.0:
            return 75.0
        else:
            return 65.0
    
    def export_results(self, filepath: str, format: str = 'excel') -> None:
        """Export calculation results to file.
        
        Args:
            filepath: Output file path
            format: Export format ('excel', 'csv', 'json')
        """
        if format == 'excel':
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Export each approach results
                for approach in ['production', 'expenditure', 'income', 'reconciled']:
                    df = pd.DataFrame(self.results[approach]).T
                    df.to_excel(writer, sheet_name=approach.capitalize())
        
        elif format == 'json':
            import json
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath} in {format} format")