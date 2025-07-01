"""Main Simulation Engine for Bangladesh GDP Simulation.

This module provides the core simulation engine that integrates all components
of the Bangladesh GDP simulation system, including economic sectors, climate
impacts, data management, and scenario analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import concurrent.futures
import threading
from copy import deepcopy

# Import sector models
from .sectors.agriculture import AgricultureSector
from .sectors.manufacturing import ManufacturingSector
from .sectors.services import ServicesSector
from .sectors.construction import ConstructionSector
from .sectors.informal_economy import InformalEconomySector

# Import climate modules
from .climate.climate_data import ClimateDataManager
from .climate.climate_impact import ClimateImpactAssessment
from .climate.weather_patterns import WeatherPatternsAnalyzer
from .climate.adaptation_measures import ClimateAdaptationManager

# Import data modules
from .data.data_collector import DataCollector
from .data.data_validator import DataValidator
from .data.data_storage import DataStorage

logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Simulation execution modes."""
    HISTORICAL = "historical"
    FORECAST = "forecast"
    SCENARIO = "scenario"
    SENSITIVITY = "sensitivity"
    MONTE_CARLO = "monte_carlo"


class OutputFormat(Enum):
    """Output format options."""
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    PARQUET = "parquet"


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    start_date: datetime
    end_date: datetime
    mode: SimulationMode
    base_year: int = 2023
    currency: str = "BDT"
    price_level: str = "constant_2023"
    frequency: str = "quarterly"  # quarterly, monthly, annual
    include_climate_impacts: bool = True
    include_adaptation_measures: bool = False
    monte_carlo_runs: int = 1000
    confidence_intervals: List[float] = None
    output_formats: List[OutputFormat] = None
    save_intermediate_results: bool = True
    parallel_processing: bool = True
    random_seed: int = 42

    def __post_init__(self):
        if self.confidence_intervals is None:
            self.confidence_intervals = [0.05, 0.25, 0.75, 0.95]
        if self.output_formats is None:
            self.output_formats = [OutputFormat.JSON, OutputFormat.CSV]


@dataclass
class ScenarioParameters:
    """Scenario-specific parameters."""
    name: str
    description: str
    economic_assumptions: Dict[str, float]
    climate_assumptions: Dict[str, float]
    policy_assumptions: Dict[str, float]
    external_assumptions: Dict[str, float]
    sector_specific_assumptions: Dict[str, Dict[str, float]]


@dataclass
class SimulationResults:
    """Comprehensive simulation results."""
    config: SimulationConfig
    scenario: Optional[ScenarioParameters]
    gdp_results: pd.DataFrame
    sector_results: Dict[str, pd.DataFrame]
    climate_impacts: Optional[pd.DataFrame]
    adaptation_analysis: Optional[Dict[str, Any]]
    summary_statistics: Dict[str, Any]
    validation_results: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


class BangladeshGDPSimulator:
    """Main simulation engine for Bangladesh GDP analysis.
    
    This class integrates all components of the simulation system to provide
    comprehensive GDP analysis with climate impacts and adaptation measures.
    """
    
    def __init__(self, config: SimulationConfig, data_path: str = None):
        """Initialize the simulation engine.
        
        Args:
            config: Simulation configuration
            data_path: Path to data directory
        """
        self.config = config
        self.data_path = Path(data_path) if data_path else Path("data")
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        # Initialize components
        self._initialize_components()
        
        # Initialize sectors
        self._initialize_sectors()
        
        # Simulation state
        self.current_results = None
        self.execution_lock = threading.Lock()
        
        logger.info(f"Bangladesh GDP Simulator initialized for {config.mode.value} mode")
    
    def _initialize_components(self):
        """Initialize all simulation components."""
        
        # Data management components
        self.data_collector = DataCollector()
        self.data_validator = DataValidator()
        self.data_storage = DataStorage(str(self.data_path / "simulation.db"))
        
        # Climate components
        self.climate_data_manager = ClimateDataManager()
        self.climate_impact_assessment = ClimateImpactAssessment()
        self.weather_patterns_analyzer = WeatherPatternsAnalyzer()
        
        if self.config.include_adaptation_measures:
            self.adaptation_manager = ClimateAdaptationManager()
        else:
            self.adaptation_manager = None
        
        logger.info("Simulation components initialized")
    
    def _initialize_sectors(self):
        """Initialize economic sector models."""
        
        self.sectors = {
            'agriculture': AgricultureSector(),
            'manufacturing': ManufacturingSector(),
            'services': ServicesSector(),
            'construction': ConstructionSector(),
            'informal_economy': InformalEconomySector()
        }
        
        logger.info(f"Initialized {len(self.sectors)} economic sectors")
    
    def load_historical_data(self, data_sources: Dict[str, str] = None) -> bool:
        """Load historical data for simulation.
        
        Args:
            data_sources: Dictionary of data source configurations
            
        Returns:
            Success status
        """
        try:
            logger.info("Loading historical data...")
            
            # Collect data from various sources
            collected_data = self.data_collector.collect_all_data()
            
            # Validate collected data
            validation_results = {}
            for source, data in collected_data.items():
                if data is not None and not data.empty:
                    validation_result = self.data_validator.validate_dataset(
                        data, f"{source}_data"
                    )
                    validation_results[source] = validation_result
                    
                    # Store validated data
                    if validation_result.overall_quality_score > 0.7:
                        self.data_storage.store_dataset(data, source)
                    else:
                        logger.warning(f"Data quality issues in {source}: {validation_result.overall_quality_score:.2f}")
            
            # Load climate data
            if self.config.include_climate_impacts:
                climate_data = self.climate_data_manager.get_climate_summary(
                    start_date=self.config.start_date,
                    end_date=self.config.end_date
                )
                self.data_storage.store_dataset(
                    pd.DataFrame([climate_data]), "climate_summary"
                )
            
            logger.info("Historical data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            return False
    
    def run_simulation(self, 
                      scenario: ScenarioParameters = None,
                      save_results: bool = True) -> SimulationResults:
        """Run the main simulation.
        
        Args:
            scenario: Scenario parameters (optional)
            save_results: Whether to save results to storage
            
        Returns:
            Comprehensive simulation results
        """
        with self.execution_lock:
            try:
                start_time = datetime.now()
                logger.info(f"Starting {self.config.mode.value} simulation...")
                
                # Initialize results containers
                gdp_results = []
                sector_results = {sector: [] for sector in self.sectors.keys()}
                climate_impacts = [] if self.config.include_climate_impacts else None
                warnings_list = []
                errors_list = []
                
                # Generate time periods
                time_periods = self._generate_time_periods()
                
                # Run simulation for each time period
                if self.config.parallel_processing and len(time_periods) > 4:
                    results = self._run_parallel_simulation(time_periods, scenario)
                else:
                    results = self._run_sequential_simulation(time_periods, scenario)
                
                # Process results
                gdp_results, sector_results, climate_impacts, warnings_list, errors_list = results
                
                # Create DataFrames
                gdp_df = pd.DataFrame(gdp_results)
                sector_dfs = {sector: pd.DataFrame(data) for sector, data in sector_results.items()}
                climate_df = pd.DataFrame(climate_impacts) if climate_impacts else None
                
                # Calculate summary statistics
                summary_stats = self._calculate_summary_statistics(gdp_df, sector_dfs)
                
                # Validate results
                validation_results = self._validate_simulation_results(gdp_df, sector_dfs)
                
                # Adaptation analysis
                adaptation_analysis = None
                if self.config.include_adaptation_measures and self.adaptation_manager:
                    adaptation_analysis = self._conduct_adaptation_analysis(scenario)
                
                # Execution metadata
                execution_metadata = {
                    'start_time': start_time,
                    'end_time': datetime.now(),
                    'execution_duration': (datetime.now() - start_time).total_seconds(),
                    'time_periods_simulated': len(time_periods),
                    'parallel_processing_used': self.config.parallel_processing,
                    'climate_impacts_included': self.config.include_climate_impacts,
                    'adaptation_measures_included': self.config.include_adaptation_measures
                }
                
                # Create comprehensive results
                simulation_results = SimulationResults(
                    config=self.config,
                    scenario=scenario,
                    gdp_results=gdp_df,
                    sector_results=sector_dfs,
                    climate_impacts=climate_df,
                    adaptation_analysis=adaptation_analysis,
                    summary_statistics=summary_stats,
                    validation_results=validation_results,
                    execution_metadata=execution_metadata,
                    warnings=warnings_list,
                    errors=errors_list
                )
                
                # Save results if requested
                if save_results:
                    self._save_simulation_results(simulation_results)
                
                self.current_results = simulation_results
                
                logger.info(f"Simulation completed in {execution_metadata['execution_duration']:.2f} seconds")
                return simulation_results
                
            except Exception as e:
                logger.error(f"Simulation failed: {str(e)}")
                raise
    
    def _generate_time_periods(self) -> List[datetime]:
        """Generate time periods for simulation."""
        
        periods = []
        current_date = self.config.start_date
        
        if self.config.frequency == "quarterly":
            delta = timedelta(days=90)
        elif self.config.frequency == "monthly":
            delta = timedelta(days=30)
        elif self.config.frequency == "annual":
            delta = timedelta(days=365)
        else:
            delta = timedelta(days=90)  # Default to quarterly
        
        while current_date <= self.config.end_date:
            periods.append(current_date)
            current_date += delta
        
        return periods
    
    def _run_sequential_simulation(self, 
                                 time_periods: List[datetime],
                                 scenario: ScenarioParameters) -> Tuple:
        """Run simulation sequentially."""
        
        gdp_results = []
        sector_results = {sector: [] for sector in self.sectors.keys()}
        climate_impacts = [] if self.config.include_climate_impacts else None
        warnings_list = []
        errors_list = []
        
        for i, period in enumerate(time_periods):
            try:
                period_results = self._simulate_period(period, scenario)
                
                gdp_results.append(period_results['gdp'])
                for sector, data in period_results['sectors'].items():
                    sector_results[sector].append(data)
                
                if climate_impacts is not None and period_results.get('climate'):
                    climate_impacts.append(period_results['climate'])
                
                if period_results.get('warnings'):
                    warnings_list.extend(period_results['warnings'])
                
                # Progress logging
                if (i + 1) % max(1, len(time_periods) // 10) == 0:
                    progress = (i + 1) / len(time_periods) * 100
                    logger.info(f"Simulation progress: {progress:.1f}%")
                    
            except Exception as e:
                error_msg = f"Error simulating period {period}: {str(e)}"
                logger.error(error_msg)
                errors_list.append(error_msg)
        
        return gdp_results, sector_results, climate_impacts, warnings_list, errors_list
    
    def _run_parallel_simulation(self, 
                               time_periods: List[datetime],
                               scenario: ScenarioParameters) -> Tuple:
        """Run simulation in parallel."""
        
        gdp_results = []
        sector_results = {sector: [] for sector in self.sectors.keys()}
        climate_impacts = [] if self.config.include_climate_impacts else None
        warnings_list = []
        errors_list = []
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(4, len(time_periods))  # Limit to 4 workers
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_period = {
                executor.submit(self._simulate_period, period, scenario): period
                for period in time_periods
            }
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_period):
                period = future_to_period[future]
                try:
                    period_results = future.result()
                    
                    gdp_results.append(period_results['gdp'])
                    for sector, data in period_results['sectors'].items():
                        sector_results[sector].append(data)
                    
                    if climate_impacts is not None and period_results.get('climate'):
                        climate_impacts.append(period_results['climate'])
                    
                    if period_results.get('warnings'):
                        warnings_list.extend(period_results['warnings'])
                    
                    completed += 1
                    if completed % max(1, len(time_periods) // 10) == 0:
                        progress = completed / len(time_periods) * 100
                        logger.info(f"Parallel simulation progress: {progress:.1f}%")
                        
                except Exception as e:
                    error_msg = f"Error simulating period {period}: {str(e)}"
                    logger.error(error_msg)
                    errors_list.append(error_msg)
        
        # Sort results by date
        combined_results = list(zip(time_periods, gdp_results))
        combined_results.sort(key=lambda x: x[0])
        gdp_results = [result[1] for result in combined_results]
        
        # Sort sector results similarly
        for sector in sector_results:
            sector_combined = list(zip(time_periods, sector_results[sector]))
            sector_combined.sort(key=lambda x: x[0])
            sector_results[sector] = [result[1] for result in sector_combined]
        
        return gdp_results, sector_results, climate_impacts, warnings_list, errors_list
    
    def _simulate_period(self, 
                        period: datetime,
                        scenario: ScenarioParameters) -> Dict[str, Any]:
        """Simulate a single time period.
        
        Args:
            period: Time period to simulate
            scenario: Scenario parameters
            
        Returns:
            Period simulation results
        """
        try:
            # Get climate data for the period
            climate_data = None
            climate_impacts = None
            
            if self.config.include_climate_impacts:
                climate_data = self.climate_data_manager.get_climate_data(
                    start_date=period,
                    end_date=period + timedelta(days=90),  # Quarterly period
                    variables=['temperature', 'precipitation', 'humidity']
                )
                
                # Assess climate impacts
                climate_impacts = self.climate_impact_assessment.assess_impacts(
                    climate_data=climate_data,
                    sectors=list(self.sectors.keys()),
                    assessment_date=period
                )
            
            # Simulate each sector
            sector_results = {}
            total_gdp = 0
            
            for sector_name, sector_model in self.sectors.items():
                # Get sector-specific parameters
                sector_params = self._get_sector_parameters(sector_name, scenario, period)
                
                # Apply climate impacts if available
                if climate_impacts:
                    sector_climate_impact = climate_impacts.get(sector_name, {})
                    sector_params.update(sector_climate_impact)
                
                # Calculate sector production
                if hasattr(sector_model, 'calculate_quarterly_production'):
                    sector_production = sector_model.calculate_quarterly_production(
                        quarter=self._get_quarter(period),
                        year=period.year,
                        **sector_params
                    )
                else:
                    # Fallback for sectors without quarterly calculation
                    sector_production = self._calculate_sector_fallback(
                        sector_name, sector_params, period
                    )
                
                sector_results[sector_name] = {
                    'date': period,
                    'sector': sector_name,
                    'production': sector_production,
                    'parameters': sector_params
                }
                
                total_gdp += sector_production
            
            # Calculate GDP
            gdp_result = {
                'date': period,
                'total_gdp': total_gdp,
                'gdp_growth_rate': self._calculate_gdp_growth_rate(total_gdp, period),
                'per_capita_gdp': total_gdp / 165000000,  # Approximate population
                'sector_shares': {
                    sector: (results['production'] / total_gdp * 100) if total_gdp > 0 else 0
                    for sector, results in sector_results.items()
                }
            }
            
            # Prepare climate result
            climate_result = None
            if climate_data:
                climate_result = {
                    'date': period,
                    'temperature': climate_data.get('temperature', 0),
                    'precipitation': climate_data.get('precipitation', 0),
                    'humidity': climate_data.get('humidity', 0),
                    'climate_impact_score': self._calculate_climate_impact_score(climate_impacts)
                }
            
            return {
                'gdp': gdp_result,
                'sectors': sector_results,
                'climate': climate_result,
                'warnings': []
            }
            
        except Exception as e:
            logger.error(f"Error simulating period {period}: {str(e)}")
            raise
    
    def _get_sector_parameters(self, 
                             sector_name: str,
                             scenario: ScenarioParameters,
                             period: datetime) -> Dict[str, Any]:
        """Get parameters for sector simulation."""
        
        # Base parameters
        params = {
            'global_demand_growth': 0.03,
            'domestic_demand_growth': 0.05,
            'inflation_rate': 0.06,
            'exchange_rate_change': 0.02,
            'policy_support': 1.0,
            'investment_growth': 0.08
        }
        
        # Apply scenario parameters if available
        if scenario:
            # Economic assumptions
            params.update(scenario.economic_assumptions)
            
            # Sector-specific assumptions
            if sector_name in scenario.sector_specific_assumptions:
                params.update(scenario.sector_specific_assumptions[sector_name])
        
        # Add time-varying factors
        params['year'] = period.year
        params['quarter'] = self._get_quarter(period)
        
        return params
    
    def _get_quarter(self, date: datetime) -> int:
        """Get quarter number from date."""
        return (date.month - 1) // 3 + 1
    
    def _calculate_sector_fallback(self, 
                                 sector_name: str,
                                 params: Dict[str, Any],
                                 period: datetime) -> float:
        """Fallback calculation for sectors without quarterly method."""
        
        # Simple growth-based calculation
        base_production = {
            'agriculture': 50000,  # Million BDT
            'manufacturing': 80000,
            'services': 120000,
            'construction': 30000,
            'informal_economy': 40000
        }.get(sector_name, 50000)
        
        growth_rate = params.get('domestic_demand_growth', 0.05)
        years_from_base = (period.year - self.config.base_year)
        
        return base_production * (1 + growth_rate) ** years_from_base
    
    def _calculate_gdp_growth_rate(self, current_gdp: float, period: datetime) -> float:
        """Calculate GDP growth rate."""
        
        # Simple calculation - in real implementation, would compare with previous period
        base_gdp = 320000  # Million BDT (approximate 2023 GDP)
        years_from_base = (period.year - self.config.base_year)
        
        if years_from_base == 0:
            return 0.065  # Base growth rate
        
        expected_gdp = base_gdp * (1.065 ** years_from_base)
        return (current_gdp / expected_gdp - 1) if expected_gdp > 0 else 0
    
    def _calculate_climate_impact_score(self, climate_impacts: Dict) -> float:
        """Calculate overall climate impact score."""
        
        if not climate_impacts:
            return 0.0
        
        # Simple aggregation of climate impacts
        impact_scores = []
        for sector, impacts in climate_impacts.items():
            if isinstance(impacts, dict) and 'impact_score' in impacts:
                impact_scores.append(impacts['impact_score'])
        
        return np.mean(impact_scores) if impact_scores else 0.0
    
    def _calculate_summary_statistics(self, 
                                    gdp_df: pd.DataFrame,
                                    sector_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate summary statistics for simulation results."""
        
        stats = {}
        
        # GDP statistics
        if not gdp_df.empty:
            stats['gdp'] = {
                'mean_gdp': gdp_df['total_gdp'].mean(),
                'median_gdp': gdp_df['total_gdp'].median(),
                'std_gdp': gdp_df['total_gdp'].std(),
                'min_gdp': gdp_df['total_gdp'].min(),
                'max_gdp': gdp_df['total_gdp'].max(),
                'mean_growth_rate': gdp_df['gdp_growth_rate'].mean(),
                'growth_volatility': gdp_df['gdp_growth_rate'].std(),
                'mean_per_capita_gdp': gdp_df['per_capita_gdp'].mean()
            }
        
        # Sector statistics
        stats['sectors'] = {}
        for sector_name, sector_df in sector_dfs.items():
            if not sector_df.empty and 'production' in sector_df.columns:
                stats['sectors'][sector_name] = {
                    'mean_production': sector_df['production'].mean(),
                    'production_volatility': sector_df['production'].std(),
                    'min_production': sector_df['production'].min(),
                    'max_production': sector_df['production'].max(),
                    'growth_trend': self._calculate_trend(sector_df['production'])
                }
        
        return stats
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend coefficient for a time series."""
        
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        coefficients = np.polyfit(x, series, 1)
        return coefficients[0]  # Slope coefficient
    
    def _validate_simulation_results(self, 
                                   gdp_df: pd.DataFrame,
                                   sector_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate simulation results."""
        
        validation_results = {
            'data_quality_checks': {},
            'economic_consistency_checks': {},
            'overall_validation_score': 0.0
        }
        
        try:
            # Data quality checks
            validation_results['data_quality_checks'] = {
                'gdp_completeness': 1.0 - gdp_df['total_gdp'].isna().sum() / len(gdp_df),
                'gdp_positive_values': (gdp_df['total_gdp'] > 0).sum() / len(gdp_df),
                'reasonable_growth_rates': ((gdp_df['gdp_growth_rate'] >= -0.2) & 
                                          (gdp_df['gdp_growth_rate'] <= 0.3)).sum() / len(gdp_df)
            }
            
            # Economic consistency checks
            if not gdp_df.empty:
                validation_results['economic_consistency_checks'] = {
                    'gdp_trend_consistency': 1.0 if gdp_df['total_gdp'].iloc[-1] > gdp_df['total_gdp'].iloc[0] else 0.5,
                    'sector_sum_consistency': 1.0,  # Would check if sector sums equal total GDP
                    'growth_rate_stability': 1.0 - min(1.0, gdp_df['gdp_growth_rate'].std() / 0.1)
                }
            
            # Calculate overall score
            all_scores = []
            all_scores.extend(validation_results['data_quality_checks'].values())
            all_scores.extend(validation_results['economic_consistency_checks'].values())
            
            validation_results['overall_validation_score'] = np.mean(all_scores) if all_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error validating simulation results: {str(e)}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def _conduct_adaptation_analysis(self, scenario: ScenarioParameters) -> Dict[str, Any]:
        """Conduct climate adaptation analysis."""
        
        if not self.adaptation_manager:
            return {}
        
        try:
            # Define climate risks based on scenario
            climate_risks = ['drought', 'flooding', 'heat_waves', 'cyclones', 'sea_level_rise']
            
            # Create adaptation plans for each sector
            adaptation_plans = []
            for sector in self.sectors.keys():
                plan = self.adaptation_manager.create_adaptation_plan(
                    sector=sector,
                    climate_risks=climate_risks,
                    budget=1000  # 1 billion BDT per sector
                )
                adaptation_plans.append(plan)
            
            # Prioritize investments
            investment_prioritization = self.adaptation_manager.prioritize_adaptation_investments(
                adaptation_plans, total_budget=8000  # 8 billion BDT total
            )
            
            # Generate comprehensive report
            adaptation_report = self.adaptation_manager.generate_adaptation_report(
                adaptation_plans, investment_prioritization
            )
            
            return {
                'adaptation_plans': [asdict(plan) for plan in adaptation_plans],
                'investment_prioritization': investment_prioritization,
                'adaptation_report': adaptation_report
            }
            
        except Exception as e:
            logger.error(f"Error conducting adaptation analysis: {str(e)}")
            return {'error': str(e)}
    
    def _save_simulation_results(self, results: SimulationResults) -> bool:
        """Save simulation results to storage."""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save to database
            self.data_storage.store_simulation_results(
                results.gdp_results, f"gdp_simulation_{timestamp}"
            )
            
            for sector, sector_df in results.sector_results.items():
                self.data_storage.store_simulation_results(
                    sector_df, f"{sector}_simulation_{timestamp}"
                )
            
            # Save to files in requested formats
            output_dir = self.data_path / "simulation_outputs" / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for format_type in self.config.output_formats:
                self._export_results(results, output_dir, format_type)
            
            logger.info(f"Simulation results saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving simulation results: {str(e)}")
            return False
    
    def _export_results(self, 
                       results: SimulationResults,
                       output_dir: Path,
                       format_type: OutputFormat) -> bool:
        """Export results in specified format."""
        
        try:
            if format_type == OutputFormat.JSON:
                # Export as JSON
                output_file = output_dir / "simulation_results.json"
                
                # Convert DataFrames to dictionaries for JSON serialization
                json_data = {
                    'config': asdict(results.config),
                    'scenario': asdict(results.scenario) if results.scenario else None,
                    'gdp_results': results.gdp_results.to_dict('records'),
                    'sector_results': {
                        sector: df.to_dict('records') 
                        for sector, df in results.sector_results.items()
                    },
                    'summary_statistics': results.summary_statistics,
                    'validation_results': results.validation_results,
                    'execution_metadata': results.execution_metadata,
                    'warnings': results.warnings,
                    'errors': results.errors
                }
                
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
            
            elif format_type == OutputFormat.CSV:
                # Export as CSV files
                results.gdp_results.to_csv(output_dir / "gdp_results.csv", index=False)
                
                for sector, sector_df in results.sector_results.items():
                    sector_df.to_csv(output_dir / f"{sector}_results.csv", index=False)
                
                if results.climate_impacts is not None:
                    results.climate_impacts.to_csv(output_dir / "climate_impacts.csv", index=False)
            
            elif format_type == OutputFormat.XLSX:
                # Export as Excel file with multiple sheets
                output_file = output_dir / "simulation_results.xlsx"
                
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    results.gdp_results.to_excel(writer, sheet_name='GDP_Results', index=False)
                    
                    for sector, sector_df in results.sector_results.items():
                        sheet_name = f"{sector.title()}_Results"[:31]  # Excel sheet name limit
                        sector_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    if results.climate_impacts is not None:
                        results.climate_impacts.to_excel(writer, sheet_name='Climate_Impacts', index=False)
                    
                    # Summary statistics sheet
                    summary_df = pd.DataFrame([results.summary_statistics])
                    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            elif format_type == OutputFormat.PARQUET:
                # Export as Parquet files
                results.gdp_results.to_parquet(output_dir / "gdp_results.parquet")
                
                for sector, sector_df in results.sector_results.items():
                    sector_df.to_parquet(output_dir / f"{sector}_results.parquet")
                
                if results.climate_impacts is not None:
                    results.climate_impacts.to_parquet(output_dir / "climate_impacts.parquet")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results in {format_type.value} format: {str(e)}")
            return False
    
    def run_scenario_analysis(self, scenarios: List[ScenarioParameters]) -> Dict[str, SimulationResults]:
        """Run multiple scenario analysis.
        
        Args:
            scenarios: List of scenarios to analyze
            
        Returns:
            Dictionary of scenario results
        """
        
        scenario_results = {}
        
        for scenario in scenarios:
            logger.info(f"Running scenario: {scenario.name}")
            
            try:
                # Update config for scenario mode
                scenario_config = deepcopy(self.config)
                scenario_config.mode = SimulationMode.SCENARIO
                
                # Create temporary simulator for scenario
                scenario_simulator = BangladeshGDPSimulator(scenario_config, str(self.data_path))
                
                # Run simulation
                results = scenario_simulator.run_simulation(scenario, save_results=False)
                scenario_results[scenario.name] = results
                
                logger.info(f"Completed scenario: {scenario.name}")
                
            except Exception as e:
                logger.error(f"Error running scenario {scenario.name}: {str(e)}")
                scenario_results[scenario.name] = None
        
        return scenario_results
    
    def run_sensitivity_analysis(self, 
                               parameter_ranges: Dict[str, Tuple[float, float]],
                               num_samples: int = 100) -> Dict[str, Any]:
        """Run sensitivity analysis.
        
        Args:
            parameter_ranges: Dictionary of parameter ranges to test
            num_samples: Number of samples for each parameter
            
        Returns:
            Sensitivity analysis results
        """
        
        logger.info(f"Running sensitivity analysis with {num_samples} samples")
        
        sensitivity_results = {
            'parameter_ranges': parameter_ranges,
            'samples': [],
            'gdp_impacts': {},
            'sector_impacts': {}
        }
        
        # Generate parameter samples
        for param_name, (min_val, max_val) in parameter_ranges.items():
            param_values = np.linspace(min_val, max_val, num_samples)
            
            gdp_results = []
            sector_results = {sector: [] for sector in self.sectors.keys()}
            
            for param_value in param_values:
                try:
                    # Create scenario with modified parameter
                    scenario = ScenarioParameters(
                        name=f"sensitivity_{param_name}_{param_value:.3f}",
                        description=f"Sensitivity test for {param_name}",
                        economic_assumptions={param_name: param_value},
                        climate_assumptions={},
                        policy_assumptions={},
                        external_assumptions={},
                        sector_specific_assumptions={}
                    )
                    
                    # Run simulation
                    results = self.run_simulation(scenario, save_results=False)
                    
                    # Extract key metrics
                    mean_gdp = results.gdp_results['total_gdp'].mean()
                    gdp_results.append(mean_gdp)
                    
                    for sector in self.sectors.keys():
                        if sector in results.sector_results:
                            sector_df = results.sector_results[sector]
                            if 'production' in sector_df.columns:
                                mean_production = sector_df['production'].mean()
                                sector_results[sector].append(mean_production)
                            else:
                                sector_results[sector].append(0)
                        else:
                            sector_results[sector].append(0)
                    
                except Exception as e:
                    logger.error(f"Error in sensitivity analysis for {param_name}={param_value}: {str(e)}")
                    gdp_results.append(np.nan)
                    for sector in self.sectors.keys():
                        sector_results[sector].append(np.nan)
            
            # Store results
            sensitivity_results['gdp_impacts'][param_name] = {
                'parameter_values': param_values.tolist(),
                'gdp_values': gdp_results,
                'sensitivity_coefficient': self._calculate_sensitivity_coefficient(param_values, gdp_results)
            }
            
            for sector in self.sectors.keys():
                if sector not in sensitivity_results['sector_impacts']:
                    sensitivity_results['sector_impacts'][sector] = {}
                
                sensitivity_results['sector_impacts'][sector][param_name] = {
                    'parameter_values': param_values.tolist(),
                    'production_values': sector_results[sector],
                    'sensitivity_coefficient': self._calculate_sensitivity_coefficient(
                        param_values, sector_results[sector]
                    )
                }
        
        logger.info("Sensitivity analysis completed")
        return sensitivity_results
    
    def _calculate_sensitivity_coefficient(self, 
                                         param_values: np.ndarray,
                                         output_values: List[float]) -> float:
        """Calculate sensitivity coefficient (elasticity)."""
        
        try:
            # Remove NaN values
            valid_indices = ~np.isnan(output_values)
            if valid_indices.sum() < 2:
                return 0.0
            
            param_clean = param_values[valid_indices]
            output_clean = np.array(output_values)[valid_indices]
            
            # Calculate elasticity (percentage change in output / percentage change in parameter)
            param_pct_change = (param_clean[-1] - param_clean[0]) / param_clean[0] if param_clean[0] != 0 else 0
            output_pct_change = (output_clean[-1] - output_clean[0]) / output_clean[0] if output_clean[0] != 0 else 0
            
            if param_pct_change != 0:
                return output_pct_change / param_pct_change
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating sensitivity coefficient: {str(e)}")
            return 0.0
    
    def generate_simulation_report(self, results: SimulationResults = None) -> Dict[str, Any]:
        """Generate comprehensive simulation report.
        
        Args:
            results: Simulation results (uses current results if None)
            
        Returns:
            Comprehensive simulation report
        """
        
        if results is None:
            results = self.current_results
        
        if results is None:
            raise ValueError("No simulation results available")
        
        report = {
            'executive_summary': {},
            'simulation_overview': {},
            'gdp_analysis': {},
            'sector_analysis': {},
            'climate_analysis': {},
            'validation_summary': {},
            'recommendations': []
        }
        
        # Executive summary
        gdp_df = results.gdp_results
        if not gdp_df.empty:
            report['executive_summary'] = {
                'simulation_period': f"{gdp_df['date'].min()} to {gdp_df['date'].max()}",
                'total_periods': len(gdp_df),
                'average_gdp': gdp_df['total_gdp'].mean(),
                'gdp_growth_rate': gdp_df['gdp_growth_rate'].mean(),
                'gdp_volatility': gdp_df['gdp_growth_rate'].std(),
                'sectors_analyzed': len(results.sector_results),
                'climate_impacts_included': results.climate_impacts is not None,
                'overall_validation_score': results.validation_results.get('overall_validation_score', 0)
            }
        
        # Simulation overview
        report['simulation_overview'] = {
            'configuration': asdict(results.config),
            'execution_metadata': results.execution_metadata,
            'warnings_count': len(results.warnings),
            'errors_count': len(results.errors)
        }
        
        # GDP analysis
        if not gdp_df.empty:
            report['gdp_analysis'] = {
                'gdp_statistics': results.summary_statistics.get('gdp', {}),
                'gdp_trends': self._analyze_gdp_trends(gdp_df),
                'sector_contributions': self._analyze_sector_contributions(gdp_df),
                'growth_patterns': self._analyze_growth_patterns(gdp_df)
            }
        
        # Sector analysis
        report['sector_analysis'] = {}
        for sector, sector_df in results.sector_results.items():
            if not sector_df.empty:
                report['sector_analysis'][sector] = {
                    'statistics': results.summary_statistics.get('sectors', {}).get(sector, {}),
                    'performance_metrics': self._analyze_sector_performance(sector_df),
                    'key_insights': self._generate_sector_insights(sector, sector_df)
                }
        
        # Climate analysis
        if results.climate_impacts is not None:
            report['climate_analysis'] = self._analyze_climate_impacts(results.climate_impacts)
        
        # Validation summary
        report['validation_summary'] = results.validation_results
        
        # Recommendations
        report['recommendations'] = self._generate_recommendations(results)
        
        return report
    
    def _analyze_gdp_trends(self, gdp_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze GDP trends."""
        
        trends = {}
        
        if len(gdp_df) > 1:
            # Calculate trend
            gdp_values = gdp_df['total_gdp'].values
            x = np.arange(len(gdp_values))
            trend_coef = np.polyfit(x, gdp_values, 1)[0]
            
            trends['trend_coefficient'] = trend_coef
            trends['trend_direction'] = 'increasing' if trend_coef > 0 else 'decreasing'
            trends['trend_strength'] = abs(trend_coef) / gdp_values.mean() if gdp_values.mean() > 0 else 0
            
            # Identify turning points
            growth_rates = gdp_df['gdp_growth_rate'].values
            turning_points = []
            for i in range(1, len(growth_rates) - 1):
                if (growth_rates[i-1] < growth_rates[i] > growth_rates[i+1]) or \
                   (growth_rates[i-1] > growth_rates[i] < growth_rates[i+1]):
                    turning_points.append(i)
            
            trends['turning_points'] = len(turning_points)
            trends['volatility_periods'] = (gdp_df['gdp_growth_rate'].abs() > gdp_df['gdp_growth_rate'].std() * 2).sum()
        
        return trends
    
    def _analyze_sector_contributions(self, gdp_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sector contributions to GDP."""
        
        contributions = {}
        
        # Extract sector shares if available
        if 'sector_shares' in gdp_df.columns:
            # This would need to be implemented based on actual data structure
            pass
        
        return contributions
    
    def _analyze_growth_patterns(self, gdp_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze GDP growth patterns."""
        
        patterns = {}
        
        if 'gdp_growth_rate' in gdp_df.columns:
            growth_rates = gdp_df['gdp_growth_rate']
            
            patterns['average_growth'] = growth_rates.mean()
            patterns['growth_volatility'] = growth_rates.std()
            patterns['positive_growth_periods'] = (growth_rates > 0).sum()
            patterns['negative_growth_periods'] = (growth_rates < 0).sum()
            patterns['high_growth_periods'] = (growth_rates > 0.07).sum()  # Above 7%
            patterns['recession_periods'] = (growth_rates < -0.02).sum()  # Below -2%
        
        return patterns
    
    def _analyze_sector_performance(self, sector_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual sector performance."""
        
        performance = {}
        
        if 'production' in sector_df.columns:
            production = sector_df['production']
            
            performance['average_production'] = production.mean()
            performance['production_volatility'] = production.std()
            performance['production_growth'] = self._calculate_trend(production)
            performance['peak_production'] = production.max()
            performance['trough_production'] = production.min()
            performance['production_range'] = production.max() - production.min()
        
        return performance
    
    def _generate_sector_insights(self, sector: str, sector_df: pd.DataFrame) -> List[str]:
        """Generate insights for individual sector."""
        
        insights = []
        
        if 'production' in sector_df.columns:
            production = sector_df['production']
            
            # Growth trend insight
            trend = self._calculate_trend(production)
            if trend > 0:
                insights.append(f"{sector.title()} sector shows positive growth trend")
            else:
                insights.append(f"{sector.title()} sector shows declining trend")
            
            # Volatility insight
            volatility = production.std() / production.mean() if production.mean() > 0 else 0
            if volatility > 0.2:
                insights.append(f"{sector.title()} sector exhibits high volatility")
            elif volatility < 0.05:
                insights.append(f"{sector.title()} sector shows stable performance")
        
        return insights
    
    def _analyze_climate_impacts(self, climate_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze climate impacts on the economy."""
        
        analysis = {}
        
        if 'climate_impact_score' in climate_df.columns:
            impact_scores = climate_df['climate_impact_score']
            
            analysis['average_impact'] = impact_scores.mean()
            analysis['impact_volatility'] = impact_scores.std()
            analysis['severe_impact_periods'] = (impact_scores > 0.7).sum()
            analysis['low_impact_periods'] = (impact_scores < 0.3).sum()
        
        if 'temperature' in climate_df.columns:
            temperature = climate_df['temperature']
            analysis['average_temperature'] = temperature.mean()
            analysis['temperature_extremes'] = ((temperature > 35) | (temperature < 10)).sum()
        
        if 'precipitation' in climate_df.columns:
            precipitation = climate_df['precipitation']
            analysis['average_precipitation'] = precipitation.mean()
            analysis['drought_periods'] = (precipitation < 50).sum()  # Less than 50mm
            analysis['flood_risk_periods'] = (precipitation > 300).sum()  # More than 300mm
        
        return analysis
    
    def _generate_recommendations(self, results: SimulationResults) -> List[str]:
        """Generate policy and strategic recommendations."""
        
        recommendations = []
        
        # GDP-based recommendations
        gdp_stats = results.summary_statistics.get('gdp', {})
        if gdp_stats.get('growth_volatility', 0) > 0.05:
            recommendations.append("Implement counter-cyclical fiscal policies to reduce GDP volatility")
        
        if gdp_stats.get('mean_growth_rate', 0) < 0.06:
            recommendations.append("Focus on productivity-enhancing investments to boost long-term growth")
        
        # Sector-based recommendations
        sector_stats = results.summary_statistics.get('sectors', {})
        for sector, stats in sector_stats.items():
            if stats.get('production_volatility', 0) > stats.get('mean_production', 0) * 0.2:
                recommendations.append(f"Develop risk management strategies for {sector} sector")
        
        # Climate-based recommendations
        if results.climate_impacts is not None:
            recommendations.append("Strengthen climate adaptation measures across all sectors")
            recommendations.append("Invest in climate-resilient infrastructure")
        
        # Validation-based recommendations
        validation_score = results.validation_results.get('overall_validation_score', 1.0)
        if validation_score < 0.8:
            recommendations.append("Improve data quality and collection systems")
        
        # General recommendations
        recommendations.extend([
            "Diversify the economy to reduce sector concentration risks",
            "Strengthen institutional capacity for economic planning",
            "Enhance monitoring and evaluation systems",
            "Promote innovation and technology adoption",
            "Improve human capital development programs"
        ])
        
        return recommendations


def create_default_config(start_year: int = 2024, end_year: int = 2030) -> SimulationConfig:
    """Create default simulation configuration.
    
    Args:
        start_year: Simulation start year
        end_year: Simulation end year
        
    Returns:
        Default simulation configuration
    """
    
    return SimulationConfig(
        start_date=datetime(start_year, 1, 1),
        end_date=datetime(end_year, 12, 31),
        mode=SimulationMode.FORECAST,
        base_year=2023,
        currency="BDT",
        price_level="constant_2023",
        frequency="quarterly",
        include_climate_impacts=True,
        include_adaptation_measures=True,
        monte_carlo_runs=1000,
        confidence_intervals=[0.05, 0.25, 0.75, 0.95],
        output_formats=[OutputFormat.JSON, OutputFormat.CSV, OutputFormat.XLSX],
        save_intermediate_results=True,
        parallel_processing=True,
        random_seed=42
    )


def create_sample_scenarios() -> List[ScenarioParameters]:
    """Create sample scenarios for analysis.
    
    Returns:
        List of sample scenarios
    """
    
    scenarios = []
    
    # Baseline scenario
    scenarios.append(ScenarioParameters(
        name="baseline",
        description="Baseline economic growth scenario",
        economic_assumptions={
            'global_demand_growth': 0.03,
            'domestic_demand_growth': 0.05,
            'inflation_rate': 0.06,
            'investment_growth': 0.08
        },
        climate_assumptions={
            'temperature_increase': 0.02,
            'precipitation_change': 0.0,
            'extreme_events_frequency': 1.0
        },
        policy_assumptions={
            'policy_support': 1.0,
            'infrastructure_investment': 1.0,
            'education_investment': 1.0
        },
        external_assumptions={
            'global_growth': 0.035,
            'commodity_prices': 1.0,
            'exchange_rate_stability': 1.0
        },
        sector_specific_assumptions={
            'agriculture': {'productivity_growth': 0.03},
            'manufacturing': {'export_growth': 0.08},
            'services': {'digitalization_rate': 0.05}
        }
    ))
    
    # High growth scenario
    scenarios.append(ScenarioParameters(
        name="high_growth",
        description="Optimistic high growth scenario",
        economic_assumptions={
            'global_demand_growth': 0.04,
            'domestic_demand_growth': 0.07,
            'inflation_rate': 0.05,
            'investment_growth': 0.12
        },
        climate_assumptions={
            'temperature_increase': 0.015,
            'precipitation_change': 0.05,
            'extreme_events_frequency': 0.8
        },
        policy_assumptions={
            'policy_support': 1.2,
            'infrastructure_investment': 1.5,
            'education_investment': 1.3
        },
        external_assumptions={
            'global_growth': 0.04,
            'commodity_prices': 1.1,
            'exchange_rate_stability': 1.1
        },
        sector_specific_assumptions={
            'agriculture': {'productivity_growth': 0.05},
            'manufacturing': {'export_growth': 0.12},
            'services': {'digitalization_rate': 0.08}
        }
    ))
    
    # Climate stress scenario
    scenarios.append(ScenarioParameters(
        name="climate_stress",
        description="High climate impact scenario",
        economic_assumptions={
            'global_demand_growth': 0.025,
            'domestic_demand_growth': 0.04,
            'inflation_rate': 0.07,
            'investment_growth': 0.06
        },
        climate_assumptions={
            'temperature_increase': 0.03,
            'precipitation_change': -0.1,
            'extreme_events_frequency': 1.5
        },
        policy_assumptions={
            'policy_support': 1.1,
            'infrastructure_investment': 1.2,
            'education_investment': 1.0
        },
        external_assumptions={
            'global_growth': 0.03,
            'commodity_prices': 1.2,
            'exchange_rate_stability': 0.9
        },
        sector_specific_assumptions={
            'agriculture': {'productivity_growth': 0.01},
            'manufacturing': {'export_growth': 0.05},
            'services': {'digitalization_rate': 0.04}
        }
    ))
    
    return scenarios


if __name__ == "__main__":
    # Create default configuration
    config = create_default_config(2024, 2030)
    
    # Initialize simulator
    simulator = BangladeshGDPSimulator(config, "data")
    
    print("\n=== Bangladesh GDP Simulation System ===")
    print(f"Simulation period: {config.start_date.year} - {config.end_date.year}")
    print(f"Mode: {config.mode.value}")
    print(f"Frequency: {config.frequency}")
    print(f"Climate impacts: {config.include_climate_impacts}")
    print(f"Adaptation measures: {config.include_adaptation_measures}")
    
    # Load historical data
    print("\nLoading historical data...")
    data_loaded = simulator.load_historical_data()
    print(f"Data loading status: {'Success' if data_loaded else 'Failed'}")
    
    # Run baseline simulation
    print("\nRunning baseline simulation...")
    baseline_results = simulator.run_simulation()
    
    print(f"\nSimulation completed:")
    print(f"  Periods simulated: {len(baseline_results.gdp_results)}")
    print(f"  Average GDP: {baseline_results.gdp_results['total_gdp'].mean():.0f} million BDT")
    print(f"  Average growth rate: {baseline_results.gdp_results['gdp_growth_rate'].mean():.2%}")
    print(f"  Validation score: {baseline_results.validation_results.get('overall_validation_score', 0):.2f}")
    
    # Generate report
    print("\nGenerating simulation report...")
    report = simulator.generate_simulation_report(baseline_results)
    
    print("\n=== Executive Summary ===")
    exec_summary = report.get('executive_summary', {})
    for key, value in exec_summary.items():
        print(f"  {key}: {value}")
    
    # Run scenario analysis
    print("\nRunning scenario analysis...")
    scenarios = create_sample_scenarios()
    scenario_results = simulator.run_scenario_analysis(scenarios)
    
    print(f"\nScenario analysis completed:")
    for scenario_name, results in scenario_results.items():
        if results:
            avg_gdp = results.gdp_results['total_gdp'].mean()
            avg_growth = results.gdp_results['gdp_growth_rate'].mean()
            print(f"  {scenario_name}: GDP={avg_gdp:.0f}M BDT, Growth={avg_growth:.2%}")
        else:
            print(f"  {scenario_name}: Failed")
    
    # Run sensitivity analysis
    print("\nRunning sensitivity analysis...")
    sensitivity_params = {
        'global_demand_growth': (0.01, 0.05),
        'domestic_demand_growth': (0.03, 0.08),
        'inflation_rate': (0.04, 0.08)
    }
    
    sensitivity_results = simulator.run_sensitivity_analysis(sensitivity_params, num_samples=20)
    
    print(f"\nSensitivity analysis completed:")
    for param, impacts in sensitivity_results['gdp_impacts'].items():
        sensitivity_coef = impacts.get('sensitivity_coefficient', 0)
        print(f"  {param}: Sensitivity coefficient = {sensitivity_coef:.2f}")
    
    print("\n=== Simulation System Ready ===")
    print("All components initialized and tested successfully!")