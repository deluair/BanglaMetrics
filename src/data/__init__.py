"""Data Management Module for Bangladesh GDP Simulation.

This module provides comprehensive data management capabilities for the BanglaMetrics
GDP simulation platform, including data collection, validation, storage, and retrieval
functionalities specifically designed for Bangladesh's economic data requirements.

Key Components:
- DataCollector: Automated data collection from multiple sources
- DataValidator: Data quality assurance and validation
- DataStorage: Efficient data storage and retrieval systems
- EconomicIndicators: Key economic indicators calculation and tracking
- BBSDataInterface: Interface with Bangladesh Bureau of Statistics data
- ClimateDataInterface: Climate and weather data integration

Data Sources:
- Bangladesh Bureau of Statistics (BBS)
- Bangladesh Bank (Central Bank)
- Export Promotion Bureau (EPB)
- National Board of Revenue (NBR)
- Ministry of Finance
- World Bank and IMF datasets
- Climate data from meteorological services
- International trade databases

Features:
- Real-time data collection and updates
- Automated data quality checks
- Historical data management
- Data interpolation and estimation
- Multi-source data reconciliation
- API integration capabilities
- Data export in multiple formats
"""

from .data_collector import DataCollector
from .data_validator import DataValidator
from .data_storage import DataStorage

# Module metadata
__version__ = "1.0.0"
__author__ = "BanglaMetrics Development Team"
__description__ = "Data Management Module for Bangladesh GDP Simulation"

# Data source configurations
DATA_SOURCES = {
    'bbs': {
        'name': 'Bangladesh Bureau of Statistics',
        'url': 'http://www.bbs.gov.bd',
        'data_types': ['gdp', 'population', 'employment', 'prices', 'agriculture'],
        'update_frequency': 'quarterly',
        'reliability_score': 0.95
    },
    'bangladesh_bank': {
        'name': 'Bangladesh Bank',
        'url': 'https://www.bb.org.bd',
        'data_types': ['monetary', 'exchange_rate', 'inflation', 'credit', 'remittances'],
        'update_frequency': 'monthly',
        'reliability_score': 0.98
    },
    'epb': {
        'name': 'Export Promotion Bureau',
        'url': 'http://www.epb.gov.bd',
        'data_types': ['exports', 'imports', 'trade_balance', 'rmg_exports'],
        'update_frequency': 'monthly',
        'reliability_score': 0.92
    },
    'nbr': {
        'name': 'National Board of Revenue',
        'url': 'https://nbr.gov.bd',
        'data_types': ['tax_revenue', 'customs_revenue', 'vat_collection'],
        'update_frequency': 'monthly',
        'reliability_score': 0.90
    },
    'mof': {
        'name': 'Ministry of Finance',
        'url': 'https://mof.gov.bd',
        'data_types': ['budget', 'government_expenditure', 'debt', 'fiscal_policy'],
        'update_frequency': 'quarterly',
        'reliability_score': 0.93
    },
    'world_bank': {
        'name': 'World Bank',
        'url': 'https://data.worldbank.org',
        'data_types': ['development_indicators', 'poverty', 'education', 'health'],
        'update_frequency': 'annual',
        'reliability_score': 0.96
    },
    'imf': {
        'name': 'International Monetary Fund',
        'url': 'https://data.imf.org',
        'data_types': ['balance_of_payments', 'international_reserves', 'exchange_rates'],
        'update_frequency': 'quarterly',
        'reliability_score': 0.97
    },
    'climate_data': {
        'name': 'Bangladesh Meteorological Department',
        'url': 'http://www.bmd.gov.bd',
        'data_types': ['temperature', 'rainfall', 'humidity', 'cyclones', 'floods'],
        'update_frequency': 'daily',
        'reliability_score': 0.88
    }
}

# Economic indicator categories
ECONOMIC_INDICATORS = {
    'growth_indicators': {
        'gdp_growth_rate': 'Real GDP growth rate (annual %)',
        'per_capita_gdp_growth': 'Per capita GDP growth rate (annual %)',
        'sectoral_growth_rates': 'Growth rates by economic sector',
        'productivity_growth': 'Labor productivity growth rate'
    },
    'price_indicators': {
        'inflation_rate': 'Consumer price inflation (annual %)',
        'food_inflation': 'Food price inflation (annual %)',
        'core_inflation': 'Core inflation excluding food and energy',
        'gdp_deflator': 'GDP deflator (annual % change)'
    },
    'employment_indicators': {
        'unemployment_rate': 'Unemployment rate (%)',
        'labor_force_participation': 'Labor force participation rate (%)',
        'employment_by_sector': 'Employment distribution by sector',
        'informal_employment_share': 'Share of informal employment (%)'
    },
    'external_indicators': {
        'export_growth': 'Export growth rate (annual %)',
        'import_growth': 'Import growth rate (annual %)',
        'trade_balance': 'Trade balance (billion USD)',
        'current_account_balance': 'Current account balance (% of GDP)',
        'remittance_inflows': 'Worker remittances (billion USD)',
        'fdi_inflows': 'Foreign direct investment inflows (billion USD)'
    },
    'fiscal_indicators': {
        'budget_balance': 'Budget balance (% of GDP)',
        'government_debt': 'Government debt (% of GDP)',
        'tax_revenue': 'Tax revenue (% of GDP)',
        'government_expenditure': 'Government expenditure (% of GDP)'
    },
    'monetary_indicators': {
        'money_supply_growth': 'Money supply growth (annual %)',
        'credit_growth': 'Private sector credit growth (annual %)',
        'interest_rates': 'Key interest rates (%)',
        'exchange_rate': 'Exchange rate (BDT per USD)'
    },
    'development_indicators': {
        'poverty_rate': 'Poverty headcount ratio (%)',
        'human_development_index': 'Human Development Index',
        'literacy_rate': 'Adult literacy rate (%)',
        'life_expectancy': 'Life expectancy at birth (years)'
    }
}

# Data quality standards
DATA_QUALITY_STANDARDS = {
    'completeness': {
        'minimum_threshold': 0.85,  # 85% data completeness required
        'preferred_threshold': 0.95  # 95% preferred
    },
    'accuracy': {
        'maximum_error_rate': 0.02,  # 2% maximum error rate
        'validation_rules': [
            'range_checks',
            'consistency_checks',
            'trend_analysis',
            'cross_validation'
        ]
    },
    'timeliness': {
        'maximum_delay_days': 30,  # Maximum 30 days delay
        'preferred_delay_days': 7   # Preferred within 7 days
    },
    'consistency': {
        'cross_source_tolerance': 0.05,  # 5% tolerance between sources
        'temporal_consistency_check': True,
        'logical_consistency_check': True
    }
}

# Data collection frequencies
COLLECTION_FREQUENCIES = {
    'real_time': 'Continuous data collection',
    'daily': 'Daily data updates',
    'weekly': 'Weekly data collection',
    'monthly': 'Monthly data collection',
    'quarterly': 'Quarterly data collection',
    'annual': 'Annual data collection'
}

# Export formats supported
EXPORT_FORMATS = {
    'csv': 'Comma-separated values',
    'xlsx': 'Microsoft Excel format',
    'json': 'JavaScript Object Notation',
    'xml': 'Extensible Markup Language',
    'parquet': 'Apache Parquet format',
    'hdf5': 'Hierarchical Data Format'
}

# API endpoints configuration
API_ENDPOINTS = {
    'data_collection': '/api/v1/data/collect',
    'data_validation': '/api/v1/data/validate',
    'data_retrieval': '/api/v1/data/retrieve',
    'indicators': '/api/v1/indicators',
    'export': '/api/v1/export',
    'status': '/api/v1/status'
}

# Default configuration
DEFAULT_CONFIG = {
    'data_storage_path': './data',
    'cache_size_mb': 512,
    'auto_validation': True,
    'backup_enabled': True,
    'compression_enabled': True,
    'encryption_enabled': False,
    'log_level': 'INFO'
}

# Module initialization
def initialize_data_module(config=None):
    """Initialize the data management module.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized data module components
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Initialize components
    data_collector = DataCollector(config)
    data_validator = DataValidator(config)
    data_storage = DataStorage(config)
    economic_indicators = EconomicIndicators(config)
    bbs_interface = BBSDataInterface(config)
    climate_interface = ClimateDataInterface(config)
    
    return {
        'collector': data_collector,
        'validator': data_validator,
        'storage': data_storage,
        'indicators': economic_indicators,
        'bbs_interface': bbs_interface,
        'climate_interface': climate_interface,
        'config': config
    }

# Utility functions
def get_data_source_info(source_name):
    """Get information about a specific data source."""
    return DATA_SOURCES.get(source_name, None)

def list_available_indicators():
    """List all available economic indicators."""
    indicators = []
    for category, indicator_dict in ECONOMIC_INDICATORS.items():
        for indicator_code, description in indicator_dict.items():
            indicators.append({
                'category': category,
                'code': indicator_code,
                'description': description
            })
    return indicators

def validate_data_quality(data, standards=None):
    """Quick data quality validation."""
    if standards is None:
        standards = DATA_QUALITY_STANDARDS
    
    # Basic validation logic
    completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
    
    return {
        'completeness': completeness,
        'meets_standards': completeness >= standards['completeness']['minimum_threshold']
    }

# Export module components
__all__ = [
    'DataCollector',
    'DataValidator', 
    'DataStorage',
    'EconomicIndicators',
    'BBSDataInterface',
    'ClimateDataInterface',
    'DATA_SOURCES',
    'ECONOMIC_INDICATORS',
    'DATA_QUALITY_STANDARDS',
    'initialize_data_module',
    'get_data_source_info',
    'list_available_indicators',
    'validate_data_quality'
]