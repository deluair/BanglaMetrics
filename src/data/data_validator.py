"""Data Validator Module for Bangladesh GDP Simulation.

This module provides comprehensive data validation capabilities to ensure
data quality, consistency, and reliability for the GDP simulation system.
It includes validation rules specific to Bangladesh's economic context.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    field: str
    severity: ValidationSeverity
    message: str
    value: Any = None
    expected_range: Tuple = None
    suggestion: str = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'field': self.field,
            'severity': self.severity.value,
            'message': self.message,
            'value': self.value,
            'expected_range': self.expected_range,
            'suggestion': self.suggestion
        }


class DataValidator:
    """Comprehensive data validation system for Bangladesh economic data.
    
    This class provides validation rules and methods to ensure data quality
    and consistency across all economic indicators and datasets used in the
    GDP simulation.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the data validator.
        
        Args:
            config: Configuration dictionary with validation settings
        """
        self.config = config or {}
        self.validation_results = []
        self.validation_rules = {}
        
        # Initialize validation rules
        self._initialize_validation_rules()
        
        # Set validation thresholds
        self.thresholds = self.config.get('thresholds', {
            'gdp_growth_min': -10.0,
            'gdp_growth_max': 15.0,
            'inflation_min': -5.0,
            'inflation_max': 25.0,
            'unemployment_min': 0.0,
            'unemployment_max': 30.0,
            'exchange_rate_min': 50.0,
            'exchange_rate_max': 150.0,
            'remittance_growth_min': -20.0,
            'remittance_growth_max': 50.0
        })
        
        logger.info("Data validator initialized")
    
    def _initialize_validation_rules(self):
        """Initialize validation rules for different data types."""
        
        self.validation_rules = {
            'gdp_data': {
                'required_fields': ['date', 'gdp_total', 'gdp_agriculture', 'gdp_industry', 'gdp_services'],
                'numeric_fields': ['gdp_total', 'gdp_agriculture', 'gdp_industry', 'gdp_services', 'gdp_growth_rate'],
                'date_fields': ['date'],
                'positive_fields': ['gdp_total', 'gdp_agriculture', 'gdp_industry', 'gdp_services'],
                'percentage_fields': ['gdp_growth_rate'],
                'sum_validation': {
                    'total_field': 'gdp_total',
                    'component_fields': ['gdp_agriculture', 'gdp_industry', 'gdp_services'],
                    'tolerance': 0.01
                }
            },
            'population_data': {
                'required_fields': ['date', 'total_population', 'urban_population', 'rural_population'],
                'numeric_fields': ['total_population', 'urban_population', 'rural_population', 'urbanization_rate', 'population_growth_rate'],
                'date_fields': ['date'],
                'positive_fields': ['total_population', 'urban_population', 'rural_population'],
                'percentage_fields': ['urbanization_rate', 'population_growth_rate'],
                'sum_validation': {
                    'total_field': 'total_population',
                    'component_fields': ['urban_population', 'rural_population'],
                    'tolerance': 0.001
                }
            },
            'employment_data': {
                'required_fields': ['date', 'unemployment_rate', 'labor_force_participation'],
                'numeric_fields': ['unemployment_rate', 'labor_force_participation', 'agriculture_employment_share', 'industry_employment_share', 'services_employment_share'],
                'date_fields': ['date'],
                'percentage_fields': ['unemployment_rate', 'labor_force_participation', 'agriculture_employment_share', 'industry_employment_share', 'services_employment_share', 'informal_employment_rate'],
                'sum_validation': {
                    'total_value': 100.0,
                    'component_fields': ['agriculture_employment_share', 'industry_employment_share', 'services_employment_share'],
                    'tolerance': 1.0
                }
            },
            'price_data': {
                'required_fields': ['date', 'general_inflation_rate', 'food_inflation_rate', 'non_food_inflation_rate'],
                'numeric_fields': ['general_inflation_rate', 'food_inflation_rate', 'non_food_inflation_rate', 'general_price_index', 'food_price_index', 'non_food_price_index'],
                'date_fields': ['date'],
                'percentage_fields': ['general_inflation_rate', 'food_inflation_rate', 'non_food_inflation_rate'],
                'positive_fields': ['general_price_index', 'food_price_index', 'non_food_price_index']
            },
            'agriculture_data': {
                'required_fields': ['date', 'rice_production_tons', 'wheat_production_tons'],
                'numeric_fields': ['rice_production_tons', 'wheat_production_tons', 'jute_production_tons', 'agricultural_gdp_growth'],
                'date_fields': ['date'],
                'positive_fields': ['rice_production_tons', 'wheat_production_tons', 'jute_production_tons'],
                'percentage_fields': ['agricultural_gdp_growth']
            },
            'monetary_data': {
                'required_fields': ['date', 'money_supply_million_bdt', 'repo_rate'],
                'numeric_fields': ['money_supply_million_bdt', 'money_supply_growth_rate', 'repo_rate', 'lending_rate', 'deposit_rate'],
                'date_fields': ['date'],
                'positive_fields': ['money_supply_million_bdt', 'repo_rate', 'lending_rate', 'deposit_rate'],
                'percentage_fields': ['money_supply_growth_rate', 'repo_rate', 'lending_rate', 'deposit_rate']
            },
            'exchange_rate_data': {
                'required_fields': ['date', 'usd_bdt_rate'],
                'numeric_fields': ['usd_bdt_rate', 'daily_change_percent'],
                'date_fields': ['date'],
                'positive_fields': ['usd_bdt_rate'],
                'percentage_fields': ['daily_change_percent']
            },
            'remittance_data': {
                'required_fields': ['date', 'remittance_million_usd'],
                'numeric_fields': ['remittance_million_usd', 'remittance_growth_rate', 'seasonal_factor'],
                'date_fields': ['date'],
                'positive_fields': ['remittance_million_usd', 'seasonal_factor'],
                'percentage_fields': ['remittance_growth_rate']
            }
        }
    
    def validate_dataset(self, data: Union[pd.DataFrame, List[Dict]], data_type: str) -> Dict:
        """Validate a complete dataset.
        
        Args:
            data: Dataset to validate (DataFrame or list of dictionaries)
            data_type: Type of data (e.g., 'gdp_data', 'population_data')
            
        Returns:
            Dictionary containing validation results and summary
        """
        logger.info(f"Starting validation for {data_type} dataset")
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Clear previous results
        self.validation_results = []
        
        # Get validation rules for this data type
        rules = self.validation_rules.get(data_type, {})
        
        if not rules:
            self.validation_results.append(
                ValidationResult(
                    field="data_type",
                    severity=ValidationSeverity.WARNING,
                    message=f"No validation rules defined for data type: {data_type}",
                    suggestion="Define validation rules for this data type"
                )
            )
            return self._compile_validation_summary()
        
        # Perform validation checks
        self._validate_basic_structure(df, rules)
        self._validate_data_types(df, rules)
        self._validate_required_fields(df, rules)
        self._validate_numeric_ranges(df, data_type)
        self._validate_date_fields(df, rules)
        self._validate_percentage_fields(df, rules)
        self._validate_positive_fields(df, rules)
        self._validate_sum_constraints(df, rules)
        self._validate_data_consistency(df, data_type)
        self._validate_temporal_consistency(df, data_type)
        self._validate_outliers(df, data_type)
        self._validate_completeness(df, rules)
        
        # Compile and return results
        return self._compile_validation_summary()
    
    def _validate_basic_structure(self, df: pd.DataFrame, rules: Dict):
        """Validate basic dataset structure."""
        
        # Check if dataset is empty
        if df.empty:
            self.validation_results.append(
                ValidationResult(
                    field="dataset",
                    severity=ValidationSeverity.CRITICAL,
                    message="Dataset is empty",
                    suggestion="Ensure data collection is working properly"
                )
            )
            return
        
        # Check minimum number of records
        min_records = self.config.get('min_records', 10)
        if len(df) < min_records:
            self.validation_results.append(
                ValidationResult(
                    field="dataset",
                    severity=ValidationSeverity.WARNING,
                    message=f"Dataset has only {len(df)} records, minimum recommended: {min_records}",
                    value=len(df),
                    suggestion="Collect more data for reliable analysis"
                )
            )
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            self.validation_results.append(
                ValidationResult(
                    field="columns",
                    severity=ValidationSeverity.ERROR,
                    message=f"Duplicate columns found: {duplicate_cols}",
                    value=duplicate_cols,
                    suggestion="Remove or rename duplicate columns"
                )
            )
    
    def _validate_data_types(self, df: pd.DataFrame, rules: Dict):
        """Validate data types of fields."""
        
        numeric_fields = rules.get('numeric_fields', [])
        date_fields = rules.get('date_fields', [])
        
        for field in numeric_fields:
            if field in df.columns:
                if not pd.api.types.is_numeric_dtype(df[field]):
                    # Try to convert to numeric
                    try:
                        df[field] = pd.to_numeric(df[field], errors='coerce')
                        null_count = df[field].isnull().sum()
                        if null_count > 0:
                            self.validation_results.append(
                                ValidationResult(
                                    field=field,
                                    severity=ValidationSeverity.WARNING,
                                    message=f"Field {field} has {null_count} non-numeric values converted to NaN",
                                    value=null_count,
                                    suggestion="Check data source for non-numeric values"
                                )
                            )
                    except Exception as e:
                        self.validation_results.append(
                            ValidationResult(
                                field=field,
                                severity=ValidationSeverity.ERROR,
                                message=f"Cannot convert field {field} to numeric: {str(e)}",
                                suggestion="Check data format and clean non-numeric values"
                            )
                        )
        
        for field in date_fields:
            if field in df.columns:
                try:
                    df[field] = pd.to_datetime(df[field])
                except Exception as e:
                    self.validation_results.append(
                        ValidationResult(
                            field=field,
                            severity=ValidationSeverity.ERROR,
                            message=f"Cannot convert field {field} to datetime: {str(e)}",
                            suggestion="Check date format (expected: YYYY-MM-DD)"
                        )
                    )
    
    def _validate_required_fields(self, df: pd.DataFrame, rules: Dict):
        """Validate presence of required fields."""
        
        required_fields = rules.get('required_fields', [])
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            self.validation_results.append(
                ValidationResult(
                    field="required_fields",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Missing required fields: {missing_fields}",
                    value=missing_fields,
                    suggestion="Ensure all required fields are included in the dataset"
                )
            )
    
    def _validate_numeric_ranges(self, df: pd.DataFrame, data_type: str):
        """Validate numeric values are within expected ranges."""
        
        # Define Bangladesh-specific validation ranges
        validation_ranges = {
            'gdp_data': {
                'gdp_growth_rate': (-10.0, 15.0),
                'gdp_total': (30000, 80000),  # Billion BDT
            },
            'population_data': {
                'population_growth_rate': (0.5, 3.0),
                'urbanization_rate': (20.0, 60.0),
                'total_population': (160000000, 200000000)
            },
            'employment_data': {
                'unemployment_rate': (2.0, 15.0),
                'labor_force_participation': (40.0, 70.0),
                'informal_employment_rate': (70.0, 95.0)
            },
            'price_data': {
                'general_inflation_rate': (-2.0, 15.0),
                'food_inflation_rate': (-5.0, 25.0),
                'general_price_index': (80.0, 200.0)
            },
            'agriculture_data': {
                'rice_production_tons': (30000000, 40000000),
                'wheat_production_tons': (800000, 1500000),
                'agricultural_gdp_growth': (-5.0, 10.0)
            },
            'monetary_data': {
                'repo_rate': (3.0, 10.0),
                'lending_rate': (6.0, 15.0),
                'money_supply_growth_rate': (5.0, 20.0)
            },
            'exchange_rate_data': {
                'usd_bdt_rate': (70.0, 120.0),
                'daily_change_percent': (-2.0, 2.0)
            },
            'remittance_data': {
                'remittance_million_usd': (1000, 3000),
                'remittance_growth_rate': (-10.0, 30.0)
            }
        }
        
        ranges = validation_ranges.get(data_type, {})
        
        for field, (min_val, max_val) in ranges.items():
            if field in df.columns:
                out_of_range = df[(df[field] < min_val) | (df[field] > max_val)]
                
                if not out_of_range.empty:
                    self.validation_results.append(
                        ValidationResult(
                            field=field,
                            severity=ValidationSeverity.WARNING,
                            message=f"Field {field} has {len(out_of_range)} values outside expected range",
                            value=len(out_of_range),
                            expected_range=(min_val, max_val),
                            suggestion=f"Review values outside range [{min_val}, {max_val}]"
                        )
                    )
    
    def _validate_date_fields(self, df: pd.DataFrame, rules: Dict):
        """Validate date fields."""
        
        date_fields = rules.get('date_fields', [])
        
        for field in date_fields:
            if field in df.columns:
                # Check for null dates
                null_dates = df[field].isnull().sum()
                if null_dates > 0:
                    self.validation_results.append(
                        ValidationResult(
                            field=field,
                            severity=ValidationSeverity.ERROR,
                            message=f"Field {field} has {null_dates} null date values",
                            value=null_dates,
                            suggestion="Fill missing dates or remove incomplete records"
                        )
                    )
                
                # Check date range
                if not df[field].isnull().all():
                    min_date = df[field].min()
                    max_date = df[field].max()
                    
                    # Check if dates are reasonable (not too far in past or future)
                    current_date = datetime.now()
                    if min_date < datetime(1970, 1, 1):
                        self.validation_results.append(
                            ValidationResult(
                                field=field,
                                severity=ValidationSeverity.WARNING,
                                message=f"Field {field} has dates before 1970: {min_date}",
                                value=min_date,
                                suggestion="Check if historical dates are correct"
                            )
                        )
                    
                    if max_date > current_date + timedelta(days=365):
                        self.validation_results.append(
                            ValidationResult(
                                field=field,
                                severity=ValidationSeverity.WARNING,
                                message=f"Field {field} has future dates beyond 1 year: {max_date}",
                                value=max_date,
                                suggestion="Check if future dates are intentional"
                            )
                        )
    
    def _validate_percentage_fields(self, df: pd.DataFrame, rules: Dict):
        """Validate percentage fields are within 0-100 range."""
        
        percentage_fields = rules.get('percentage_fields', [])
        
        for field in percentage_fields:
            if field in df.columns:
                # Check for values outside reasonable percentage range
                out_of_range = df[(df[field] < -50) | (df[field] > 200)]
                
                if not out_of_range.empty:
                    self.validation_results.append(
                        ValidationResult(
                            field=field,
                            severity=ValidationSeverity.WARNING,
                            message=f"Field {field} has {len(out_of_range)} values outside reasonable percentage range (-50%, 200%)",
                            value=len(out_of_range),
                            expected_range=(-50, 200),
                            suggestion="Review extreme percentage values"
                        )
                    )
    
    def _validate_positive_fields(self, df: pd.DataFrame, rules: Dict):
        """Validate fields that should be positive."""
        
        positive_fields = rules.get('positive_fields', [])
        
        for field in positive_fields:
            if field in df.columns:
                negative_values = df[df[field] < 0]
                
                if not negative_values.empty:
                    self.validation_results.append(
                        ValidationResult(
                            field=field,
                            severity=ValidationSeverity.ERROR,
                            message=f"Field {field} has {len(negative_values)} negative values",
                            value=len(negative_values),
                            suggestion="Check data source - field should contain only positive values"
                        )
                    )
    
    def _validate_sum_constraints(self, df: pd.DataFrame, rules: Dict):
        """Validate sum constraints (e.g., components should sum to total)."""
        
        sum_validation = rules.get('sum_validation', {})
        
        if sum_validation:
            if 'total_field' in sum_validation:
                # Components should sum to total
                total_field = sum_validation['total_field']
                component_fields = sum_validation['component_fields']
                tolerance = sum_validation.get('tolerance', 0.01)
                
                if total_field in df.columns and all(field in df.columns for field in component_fields):
                    df['calculated_total'] = df[component_fields].sum(axis=1)
                    df['difference'] = abs(df[total_field] - df['calculated_total'])
                    df['relative_difference'] = df['difference'] / df[total_field]
                    
                    violations = df[df['relative_difference'] > tolerance]
                    
                    if not violations.empty:
                        self.validation_results.append(
                            ValidationResult(
                                field="sum_constraint",
                                severity=ValidationSeverity.WARNING,
                                message=f"{len(violations)} records where {component_fields} don't sum to {total_field} within {tolerance*100}% tolerance",
                                value=len(violations),
                                suggestion="Check calculation methodology for component fields"
                            )
                        )
            
            elif 'total_value' in sum_validation:
                # Components should sum to a specific value
                total_value = sum_validation['total_value']
                component_fields = sum_validation['component_fields']
                tolerance = sum_validation.get('tolerance', 1.0)
                
                if all(field in df.columns for field in component_fields):
                    df['calculated_sum'] = df[component_fields].sum(axis=1)
                    df['difference'] = abs(df['calculated_sum'] - total_value)
                    
                    violations = df[df['difference'] > tolerance]
                    
                    if not violations.empty:
                        self.validation_results.append(
                            ValidationResult(
                                field="sum_constraint",
                                severity=ValidationSeverity.WARNING,
                                message=f"{len(violations)} records where {component_fields} don't sum to {total_value} within {tolerance} tolerance",
                                value=len(violations),
                                suggestion="Check that percentage fields sum to 100%"
                            )
                        )
    
    def _validate_data_consistency(self, df: pd.DataFrame, data_type: str):
        """Validate data consistency and logical relationships."""
        
        # GDP data consistency checks
        if data_type == 'gdp_data':
            # Check if sectoral shares are reasonable
            if all(field in df.columns for field in ['gdp_agriculture', 'gdp_industry', 'gdp_services', 'gdp_total']):
                df['agriculture_share'] = df['gdp_agriculture'] / df['gdp_total'] * 100
                df['industry_share'] = df['gdp_industry'] / df['gdp_total'] * 100
                df['services_share'] = df['gdp_services'] / df['gdp_total'] * 100
                
                # Check for unrealistic sectoral shares
                if (df['agriculture_share'] > 25).any():
                    self.validation_results.append(
                        ValidationResult(
                            field="agriculture_share",
                            severity=ValidationSeverity.WARNING,
                            message="Agriculture share exceeds 25% - unusually high for Bangladesh",
                            suggestion="Verify agriculture GDP calculations"
                        )
                    )
                
                if (df['services_share'] < 45).any():
                    self.validation_results.append(
                        ValidationResult(
                            field="services_share",
                            severity=ValidationSeverity.WARNING,
                            message="Services share below 45% - unusually low for Bangladesh",
                            suggestion="Verify services GDP calculations"
                        )
                    )
        
        # Population data consistency checks
        elif data_type == 'population_data':
            if all(field in df.columns for field in ['urban_population', 'rural_population', 'total_population']):
                # Check urbanization trend
                if 'urbanization_rate' in df.columns:
                    calculated_urban_rate = df['urban_population'] / df['total_population'] * 100
                    rate_difference = abs(calculated_urban_rate - df['urbanization_rate'])
                    
                    if (rate_difference > 1.0).any():
                        self.validation_results.append(
                            ValidationResult(
                                field="urbanization_rate",
                                severity=ValidationSeverity.WARNING,
                                message="Urbanization rate inconsistent with urban/total population ratio",
                                suggestion="Check urbanization rate calculation"
                            )
                        )
    
    def _validate_temporal_consistency(self, df: pd.DataFrame, data_type: str):
        """Validate temporal consistency and trends."""
        
        if 'date' not in df.columns:
            return
        
        # Sort by date
        df_sorted = df.sort_values('date')
        
        # Check for reasonable growth rates
        if data_type == 'gdp_data' and 'gdp_total' in df.columns:
            df_sorted['gdp_growth_calculated'] = df_sorted['gdp_total'].pct_change() * 100
            
            # Check for extreme quarter-to-quarter changes
            extreme_changes = df_sorted[abs(df_sorted['gdp_growth_calculated']) > 20]
            
            if not extreme_changes.empty:
                self.validation_results.append(
                    ValidationResult(
                        field="gdp_growth",
                        severity=ValidationSeverity.WARNING,
                        message=f"{len(extreme_changes)} periods with extreme GDP growth (>20%)",
                        value=len(extreme_changes),
                        suggestion="Review periods with extreme growth rates"
                    )
                )
        
        # Check for data gaps
        if len(df_sorted) > 1:
            date_diffs = df_sorted['date'].diff().dt.days
            expected_interval = self._get_expected_interval(data_type)
            
            large_gaps = date_diffs[date_diffs > expected_interval * 1.5]
            
            if not large_gaps.empty:
                self.validation_results.append(
                    ValidationResult(
                        field="date",
                        severity=ValidationSeverity.WARNING,
                        message=f"{len(large_gaps)} large gaps in time series data",
                        value=len(large_gaps),
                        suggestion="Check for missing data periods"
                    )
                )
    
    def _validate_outliers(self, df: pd.DataFrame, data_type: str):
        """Validate and identify statistical outliers."""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['year', 'month', 'day', 'quarter']:  # Skip time components
                continue
            
            # Use IQR method to detect outliers
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
            if not outliers.empty and len(outliers) > len(df) * 0.05:  # More than 5% outliers
                self.validation_results.append(
                    ValidationResult(
                        field=column,
                        severity=ValidationSeverity.INFO,
                        message=f"Field {column} has {len(outliers)} statistical outliers ({len(outliers)/len(df)*100:.1f}%)",
                        value=len(outliers),
                        suggestion="Review outlier values for data quality issues"
                    )
                )
    
    def _validate_completeness(self, df: pd.DataFrame, rules: Dict):
        """Validate data completeness."""
        
        required_fields = rules.get('required_fields', [])
        
        for field in required_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                null_percentage = null_count / len(df) * 100
                
                if null_percentage > 10:  # More than 10% missing
                    self.validation_results.append(
                        ValidationResult(
                            field=field,
                            severity=ValidationSeverity.WARNING,
                            message=f"Field {field} has {null_percentage:.1f}% missing values",
                            value=null_percentage,
                            suggestion="Investigate data collection issues or implement imputation"
                        )
                    )
                elif null_percentage > 0:
                    self.validation_results.append(
                        ValidationResult(
                            field=field,
                            severity=ValidationSeverity.INFO,
                            message=f"Field {field} has {null_percentage:.1f}% missing values",
                            value=null_percentage,
                            suggestion="Consider data imputation for missing values"
                        )
                    )
    
    def _get_expected_interval(self, data_type: str) -> int:
        """Get expected interval between data points in days."""
        
        intervals = {
            'gdp_data': 90,        # Quarterly
            'population_data': 365, # Annual
            'employment_data': 90,  # Quarterly
            'price_data': 30,      # Monthly
            'agriculture_data': 90, # Quarterly
            'monetary_data': 30,   # Monthly
            'exchange_rate_data': 1, # Daily
            'remittance_data': 30  # Monthly
        }
        
        return intervals.get(data_type, 30)
    
    def _compile_validation_summary(self) -> Dict:
        """Compile validation results into a summary."""
        
        summary = {
            'total_checks': len(self.validation_results),
            'severity_counts': {
                'critical': 0,
                'error': 0,
                'warning': 0,
                'info': 0
            },
            'validation_passed': True,
            'results': [result.to_dict() for result in self.validation_results],
            'timestamp': datetime.now().isoformat()
        }
        
        # Count by severity
        for result in self.validation_results:
            summary['severity_counts'][result.severity.value] += 1
        
        # Determine if validation passed
        if summary['severity_counts']['critical'] > 0 or summary['severity_counts']['error'] > 0:
            summary['validation_passed'] = False
        
        # Add recommendations
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        # Count issues by type
        critical_count = sum(1 for r in self.validation_results if r.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for r in self.validation_results if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in self.validation_results if r.severity == ValidationSeverity.WARNING)
        
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical issues before using data for analysis")
        
        if error_count > 0:
            recommendations.append(f"Fix {error_count} data errors to improve data quality")
        
        if warning_count > 5:
            recommendations.append(f"Review {warning_count} warnings - some may indicate data quality issues")
        
        # Specific recommendations based on common issues
        field_issues = {}
        for result in self.validation_results:
            if result.field not in field_issues:
                field_issues[result.field] = []
            field_issues[result.field].append(result.severity)
        
        for field, severities in field_issues.items():
            if ValidationSeverity.CRITICAL in severities or ValidationSeverity.ERROR in severities:
                recommendations.append(f"Priority: Fix data issues in field '{field}'")
        
        if not recommendations:
            recommendations.append("Data validation passed - dataset appears to be of good quality")
        
        return recommendations
    
    def export_validation_report(self, output_path: str, format: str = 'json') -> bool:
        """Export validation report to file."""
        
        try:
            output_path = Path(output_path)
            
            if not self.validation_results:
                logger.warning("No validation results to export")
                return False
            
            summary = self._compile_validation_summary()
            
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
            
            elif format == 'csv':
                df = pd.DataFrame([result.to_dict() for result in self.validation_results])
                df.to_csv(output_path, index=False)
            
            elif format == 'xlsx':
                df = pd.DataFrame([result.to_dict() for result in self.validation_results])
                with pd.ExcelWriter(output_path) as writer:
                    df.to_excel(writer, sheet_name='Validation Results', index=False)
                    
                    # Add summary sheet
                    summary_df = pd.DataFrame([
                        {'Metric': 'Total Checks', 'Value': summary['total_checks']},
                        {'Metric': 'Critical Issues', 'Value': summary['severity_counts']['critical']},
                        {'Metric': 'Errors', 'Value': summary['severity_counts']['error']},
                        {'Metric': 'Warnings', 'Value': summary['severity_counts']['warning']},
                        {'Metric': 'Info', 'Value': summary['severity_counts']['info']},
                        {'Metric': 'Validation Passed', 'Value': summary['validation_passed']}
                    ])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            logger.info(f"Validation report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting validation report: {str(e)}")
            return False
    
    def get_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)."""
        
        if not self.validation_results:
            return 100.0
        
        # Weight different severity levels
        weights = {
            ValidationSeverity.CRITICAL: -25,
            ValidationSeverity.ERROR: -10,
            ValidationSeverity.WARNING: -3,
            ValidationSeverity.INFO: -1
        }
        
        total_deduction = 0
        for result in self.validation_results:
            total_deduction += weights.get(result.severity, 0)
        
        # Calculate score (minimum 0, maximum 100)
        score = max(0, min(100, 100 + total_deduction))
        
        return score
    
    def validate_multiple_datasets(self, datasets: Dict[str, Union[pd.DataFrame, List[Dict]]]) -> Dict:
        """Validate multiple datasets and provide consolidated report."""
        
        logger.info(f"Validating {len(datasets)} datasets")
        
        all_results = {}
        overall_summary = {
            'datasets_validated': len(datasets),
            'datasets_passed': 0,
            'total_issues': 0,
            'average_quality_score': 0,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        quality_scores = []
        
        for dataset_name, dataset in datasets.items():
            logger.info(f"Validating dataset: {dataset_name}")
            
            # Validate individual dataset
            validation_result = self.validate_dataset(dataset, dataset_name)
            all_results[dataset_name] = validation_result
            
            # Update overall summary
            if validation_result['validation_passed']:
                overall_summary['datasets_passed'] += 1
            
            overall_summary['total_issues'] += validation_result['total_checks']
            
            # Calculate quality score for this dataset
            quality_score = self.get_data_quality_score()
            quality_scores.append(quality_score)
        
        # Calculate average quality score
        if quality_scores:
            overall_summary['average_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        return {
            'overall_summary': overall_summary,
            'dataset_results': all_results,
            'quality_scores': dict(zip(datasets.keys(), quality_scores))
        }