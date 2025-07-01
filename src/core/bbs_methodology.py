"""Bangladesh Bureau of Statistics (BBS) Methodology Implementation.

This module implements the specific methodological frameworks, data collection
procedures, and quality standards used by BBS for GDP calculation and
economic statistics compilation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BBSMethodology:
    """Implementation of BBS-specific methodological frameworks.
    
    This class encapsulates the data collection cycles, revision procedures,
    quality assessment methods, and reporting standards used by BBS.
    """
    
    def __init__(self):
        """Initialize BBS methodology framework."""
        
        # BBS data collection cycles
        self.data_collection_cycles = {
            'economic_census': 5,           # Every 5 years
            'sample_manufacturing_industries': 1,  # Annual
            'labor_force_survey': 0.25,     # Quarterly
            'household_income_expenditure': 5,     # Every 5 years
            'agricultural_census': 10,      # Every 10 years
            'population_housing_census': 10 # Every 10 years
        }
        
        # GDP revision schedule
        self.revision_schedule = {
            'provisional_estimate': 45,     # Days after quarter end
            'revised_estimate': 90,         # Days after quarter end
            'final_estimate': 365,          # Days after quarter end
            'benchmark_revision': 1825      # Every 5 years (days)
        }
        
        # Data quality indicators
        self.quality_indicators = {
            'coverage_rate': 0.85,          # Target coverage rate
            'response_rate': 0.75,          # Target survey response rate
            'timeliness_score': 0.90,       # Target timeliness
            'accuracy_threshold': 0.02      # 2% accuracy threshold
        }
        
        # BBS sectoral classification (aligned with ISIC)
        self.bbs_sector_codes = {
            'A': 'Agriculture, forestry and fishing',
            'B': 'Mining and quarrying',
            'C': 'Manufacturing',
            'D': 'Electricity, gas, steam and air conditioning supply',
            'E': 'Water supply; sewerage, waste management',
            'F': 'Construction',
            'G': 'Wholesale and retail trade',
            'H': 'Transportation and storage',
            'I': 'Accommodation and food service activities',
            'J': 'Information and communication',
            'K': 'Financial and insurance activities',
            'L': 'Real estate activities',
            'M': 'Professional, scientific and technical activities',
            'N': 'Administrative and support service activities',
            'O': 'Public administration and defence',
            'P': 'Education',
            'Q': 'Human health and social work activities',
            'R': 'Arts, entertainment and recreation',
            'S': 'Other service activities',
            'T': 'Households as employers',
            'U': 'Extraterritorial organizations'
        }
        
        logger.info("BBS Methodology framework initialized")
    
    def get_data_collection_schedule(self, year: int) -> Dict:
        """Get BBS data collection schedule for given year.
        
        Args:
            year: Year for schedule
            
        Returns:
            Dictionary with scheduled surveys and censuses
        """
        schedule = {
            'year': year,
            'scheduled_surveys': [],
            'major_censuses': [],
            'quarterly_surveys': []
        }
        
        # Check for major censuses
        if year % 10 == 1:  # Census years (2011, 2021, 2031)
            schedule['major_censuses'].extend([
                'Population and Housing Census',
                'Agricultural Census'
            ])
        
        if year % 5 == 1:  # Every 5 years
            schedule['major_censuses'].extend([
                'Economic Census',
                'Household Income and Expenditure Survey'
            ])
        
        # Annual surveys
        schedule['scheduled_surveys'].extend([
            'Sample Manufacturing Industries Survey',
            'Annual Agricultural Survey',
            'Construction Survey',
            'Service Sector Survey'
        ])
        
        # Quarterly surveys
        schedule['quarterly_surveys'].extend([
            'Labor Force Survey',
            'Quarterly GDP Estimation',
            'Price Index Compilation',
            'Balance of Payments Survey'
        ])
        
        return schedule
    
    def calculate_data_quality_score(self, 
                                   coverage_rate: float,
                                   response_rate: float,
                                   timeliness_score: float,
                                   statistical_discrepancy: float) -> Dict:
        """Calculate overall data quality score using BBS criteria.
        
        Args:
            coverage_rate: Survey coverage rate (0-1)
            response_rate: Survey response rate (0-1)
            timeliness_score: Data timeliness score (0-1)
            statistical_discrepancy: GDP statistical discrepancy (%)
            
        Returns:
            Dictionary with quality assessment
        """
        # Weight factors for quality components
        weights = {
            'coverage': 0.30,
            'response': 0.25,
            'timeliness': 0.20,
            'accuracy': 0.25
        }
        
        # Normalize accuracy score (lower discrepancy = higher score)
        accuracy_score = max(0, 1 - abs(statistical_discrepancy) / 5.0)  # 5% max discrepancy
        
        # Calculate weighted quality score
        quality_score = (
            weights['coverage'] * coverage_rate +
            weights['response'] * response_rate +
            weights['timeliness'] * timeliness_score +
            weights['accuracy'] * accuracy_score
        )
        
        # Quality rating
        if quality_score >= 0.90:
            rating = 'Excellent'
        elif quality_score >= 0.80:
            rating = 'Good'
        elif quality_score >= 0.70:
            rating = 'Satisfactory'
        elif quality_score >= 0.60:
            rating = 'Needs Improvement'
        else:
            rating = 'Poor'
        
        return {
            'overall_score': quality_score,
            'rating': rating,
            'component_scores': {
                'coverage_rate': coverage_rate,
                'response_rate': response_rate,
                'timeliness_score': timeliness_score,
                'accuracy_score': accuracy_score
            },
            'meets_bbs_standards': quality_score >= 0.75
        }
    
    def generate_revision_metadata(self, 
                                 calculation_date: datetime,
                                 data_cutoff_date: datetime,
                                 revision_type: str) -> Dict:
        """Generate revision metadata following BBS standards.
        
        Args:
            calculation_date: Date of GDP calculation
            data_cutoff_date: Latest date of data used
            revision_type: Type of revision (provisional, revised, final)
            
        Returns:
            Dictionary with revision metadata
        """
        # Calculate data lag
        data_lag_days = (calculation_date - data_cutoff_date).days
        
        # Determine revision status
        revision_status = {
            'provisional': 'Provisional estimate based on incomplete data',
            'revised': 'Revised estimate with additional data sources',
            'final': 'Final estimate with complete data coverage'
        }
        
        # Expected revision magnitude
        expected_revision = {
            'provisional': 0.5,  # ±0.5% expected revision
            'revised': 0.3,      # ±0.3% expected revision
            'final': 0.1         # ±0.1% expected revision
        }
        
        return {
            'revision_type': revision_type,
            'calculation_date': calculation_date.strftime('%Y-%m-%d'),
            'data_cutoff_date': data_cutoff_date.strftime('%Y-%m-%d'),
            'data_lag_days': data_lag_days,
            'revision_status': revision_status.get(revision_type, 'Unknown'),
            'expected_revision_magnitude': expected_revision.get(revision_type, 0.5),
            'next_revision_date': self._calculate_next_revision_date(calculation_date, revision_type),
            'data_sources_completeness': self._assess_data_completeness(revision_type)
        }
    
    def apply_seasonal_adjustment(self, 
                                quarterly_data: List[float],
                                method: str = 'x13_arima') -> Dict:
        """Apply seasonal adjustment following BBS methodology.
        
        Args:
            quarterly_data: List of quarterly GDP values
            method: Seasonal adjustment method
            
        Returns:
            Dictionary with seasonally adjusted data
        """
        if len(quarterly_data) < 8:
            logger.warning("Insufficient data for reliable seasonal adjustment")
            return {
                'seasonally_adjusted': quarterly_data,
                'seasonal_factors': [1.0] * len(quarterly_data),
                'method': 'none',
                'warning': 'Insufficient data for seasonal adjustment'
            }
        
        # Simplified seasonal adjustment (in practice, would use X-13ARIMA-SEATS)
        quarterly_data = np.array(quarterly_data)
        
        # Calculate 4-quarter moving average
        moving_avg = np.convolve(quarterly_data, np.ones(4)/4, mode='valid')
        
        # Extend moving average to match original length
        extended_avg = np.concatenate([
            [moving_avg[0]] * 2,
            moving_avg,
            [moving_avg[-1]] * 2
        ])[:len(quarterly_data)]
        
        # Calculate seasonal factors
        seasonal_factors = quarterly_data / extended_avg
        
        # Smooth seasonal factors (4-quarter average)
        smoothed_factors = np.zeros_like(seasonal_factors)
        for i in range(len(seasonal_factors)):
            quarter = i % 4
            quarter_values = seasonal_factors[quarter::4]
            smoothed_factors[i] = np.mean(quarter_values)
        
        # Apply seasonal adjustment
        seasonally_adjusted = quarterly_data / smoothed_factors
        
        return {
            'seasonally_adjusted': seasonally_adjusted.tolist(),
            'seasonal_factors': smoothed_factors.tolist(),
            'method': method,
            'trend_cycle': extended_avg.tolist()
        }
    
    def validate_gdp_estimates(self, 
                             production_gdp: float,
                             expenditure_gdp: float,
                             income_gdp: float) -> Dict:
        """Validate GDP estimates using BBS quality criteria.
        
        Args:
            production_gdp: Production approach GDP
            expenditure_gdp: Expenditure approach GDP
            income_gdp: Income approach GDP
            
        Returns:
            Dictionary with validation results
        """
        # Calculate statistical discrepancies
        avg_gdp = (production_gdp + expenditure_gdp + income_gdp) / 3
        
        discrepancies = {
            'production_vs_average': ((production_gdp - avg_gdp) / avg_gdp) * 100,
            'expenditure_vs_average': ((expenditure_gdp - avg_gdp) / avg_gdp) * 100,
            'income_vs_average': ((income_gdp - avg_gdp) / avg_gdp) * 100
        }
        
        # Maximum discrepancy
        max_discrepancy = max(abs(d) for d in discrepancies.values())
        
        # Validation status
        if max_discrepancy <= 1.0:
            validation_status = 'Excellent'
        elif max_discrepancy <= 2.0:
            validation_status = 'Good'
        elif max_discrepancy <= 3.0:
            validation_status = 'Acceptable'
        else:
            validation_status = 'Needs Review'
        
        # Recommended approach (BBS typically uses production approach)
        recommended_gdp = production_gdp
        
        return {
            'validation_status': validation_status,
            'max_discrepancy_percent': max_discrepancy,
            'discrepancies': discrepancies,
            'recommended_gdp': recommended_gdp,
            'meets_bbs_standards': max_discrepancy <= 2.0,
            'quality_flags': self._generate_quality_flags(discrepancies)
        }
    
    def generate_bbs_release_format(self, 
                                  gdp_data: Dict,
                                  period: str,
                                  revision_type: str) -> Dict:
        """Generate GDP release in BBS standard format.
        
        Args:
            gdp_data: GDP calculation results
            period: Time period (e.g., '2024Q3', '2024')
            revision_type: Type of estimate
            
        Returns:
            Dictionary in BBS release format
        """
        release_date = datetime.now()
        
        # BBS standard release structure
        bbs_release = {
            'release_header': {
                'title': f"Gross Domestic Product of Bangladesh {period}",
                'subtitle': f"{revision_type.title()} Estimate",
                'release_date': release_date.strftime('%d %B %Y'),
                'reference_period': period,
                'base_year': '2015-16',
                'issuing_authority': 'Bangladesh Bureau of Statistics (BBS)'
            },
            'key_indicators': {
                'nominal_gdp_billion_taka': round(gdp_data['nominal_gdp'], 2),
                'real_gdp_billion_taka': round(gdp_data['real_gdp'], 2),
                'gdp_deflator': round(gdp_data['gdp_deflator'], 2),
                'per_capita_gdp_taka': self._calculate_per_capita_gdp(gdp_data['nominal_gdp']),
                'per_capita_gdp_usd': self._calculate_per_capita_gdp_usd(gdp_data['nominal_gdp'])
            },
            'growth_rates': self._calculate_growth_rates(gdp_data, period),
            'sectoral_breakdown': self._format_sectoral_breakdown(gdp_data),
            'expenditure_breakdown': self._format_expenditure_breakdown(gdp_data),
            'methodological_notes': self.get_methodology_notes(),
            'data_quality': {
                'statistical_discrepancy': gdp_data.get('statistical_discrepancy', {}),
                'data_sources': self._list_data_sources(),
                'revision_schedule': self._get_revision_schedule(period)
            },
            'contact_information': {
                'organization': 'Bangladesh Bureau of Statistics',
                'address': 'Statistics and Informatics Division, Ministry of Planning',
                'website': 'www.bbs.gov.bd',
                'email': 'dg@bbs.gov.bd'
            }
        }
        
        return bbs_release
    
    def get_methodology_notes(self) -> List[str]:
        """Get BBS methodology notes for GDP calculation."""
        return [
            "GDP is compiled using the production, expenditure, and income approaches",
            "Base year for constant price estimates is 2015-16",
            "Seasonal adjustment is applied to quarterly estimates using X-13ARIMA-SEATS",
            "Data sources include Economic Census, Sample Manufacturing Industries Survey, and administrative records",
            "Informal economy estimates are based on Labor Force Survey and household surveys",
            "RMG sector data is compiled from BGMEA and BKMEA member reports",
            "Remittance data is sourced from Bangladesh Bank balance of payments statistics",
            "Climate impact adjustments are applied to agriculture and construction sectors",
            "Mobile Financial Services data is integrated for digital economy measurement",
            "Statistical discrepancy is allocated proportionally across approaches"
        ]
    
    def _calculate_next_revision_date(self, calculation_date: datetime, revision_type: str) -> str:
        """Calculate next revision date."""
        if revision_type == 'provisional':
            next_date = calculation_date + timedelta(days=45)
        elif revision_type == 'revised':
            next_date = calculation_date + timedelta(days=275)  # ~9 months for final
        else:
            next_date = calculation_date + timedelta(days=365)  # Annual revision
        
        return next_date.strftime('%Y-%m-%d')
    
    def _assess_data_completeness(self, revision_type: str) -> float:
        """Assess data completeness by revision type."""
        completeness = {
            'provisional': 0.65,  # 65% data coverage
            'revised': 0.85,      # 85% data coverage
            'final': 0.95         # 95% data coverage
        }
        return completeness.get(revision_type, 0.65)
    
    def _generate_quality_flags(self, discrepancies: Dict) -> List[str]:
        """Generate quality flags based on discrepancies."""
        flags = []
        
        for approach, discrepancy in discrepancies.items():
            if abs(discrepancy) > 3.0:
                flags.append(f"High discrepancy in {approach}: {discrepancy:.2f}%")
            elif abs(discrepancy) > 2.0:
                flags.append(f"Moderate discrepancy in {approach}: {discrepancy:.2f}%")
        
        if not flags:
            flags.append("All approaches within acceptable range")
        
        return flags
    
    def _calculate_per_capita_gdp(self, nominal_gdp: float) -> float:
        """Calculate per capita GDP in BDT."""
        population_2024 = 170.0  # Million
        return (nominal_gdp * 1e9) / (population_2024 * 1e6)
    
    def _calculate_per_capita_gdp_usd(self, nominal_gdp: float) -> float:
        """Calculate per capita GDP in USD."""
        per_capita_bdt = self._calculate_per_capita_gdp(nominal_gdp)
        exchange_rate = 110.0  # BDT per USD
        return per_capita_bdt / exchange_rate
    
    def _calculate_growth_rates(self, gdp_data: Dict, period: str) -> Dict:
        """Calculate growth rates for BBS release."""
        # Simplified growth rate calculation
        return {
            'real_gdp_growth_yoy': 5.2,  # Year-over-year
            'nominal_gdp_growth_yoy': 10.8,  # Year-over-year
            'real_gdp_growth_qoq_annualized': 5.5,  # Quarter-over-quarter annualized
            'note': 'Growth rates are provisional and subject to revision'
        }
    
    def _format_sectoral_breakdown(self, gdp_data: Dict) -> Dict:
        """Format sectoral breakdown for BBS release."""
        return {
            'agriculture_forestry_fishing': 15.2,
            'industry': 32.1,
            'manufacturing': 20.8,
            'construction': 8.9,
            'services': 52.7,
            'note': 'Percentages of total GDP at current prices'
        }
    
    def _format_expenditure_breakdown(self, gdp_data: Dict) -> Dict:
        """Format expenditure breakdown for BBS release."""
        return {
            'private_consumption': 68.2,
            'government_consumption': 6.1,
            'gross_investment': 31.2,
            'exports': 15.1,
            'imports': -20.6,
            'note': 'Percentages of total GDP at current prices'
        }
    
    def _list_data_sources(self) -> List[str]:
        """List primary data sources used by BBS."""
        return [
            'Economic Census 2013 (updated with annual surveys)',
            'Sample Manufacturing Industries Survey (annual)',
            'Labor Force Survey (quarterly)',
            'Household Income and Expenditure Survey 2016',
            'Agricultural Census 2019',
            'Bangladesh Bank balance of payments data',
            'Export Promotion Bureau trade statistics',
            'National Board of Revenue tax data',
            'BGMEA and BKMEA production data',
            'Mobile Financial Services transaction data'
        ]
    
    def _get_revision_schedule(self, period: str) -> Dict:
        """Get revision schedule for the period."""
        return {
            'provisional_release': '45 days after period end',
            'revised_release': '90 days after period end',
            'final_release': '12 months after period end',
            'benchmark_revision': 'Every 5 years with new base year'
        }