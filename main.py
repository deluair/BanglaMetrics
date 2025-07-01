#!/usr/bin/env python3
"""
Bangladesh GDP Simulation System - Main Application

This is the main entry point for the Bangladesh GDP simulation system.
It demonstrates the complete functionality including data collection,
sector modeling, climate impact assessment, and comprehensive analysis.

Usage:
    python main.py [--mode MODE] [--years START_YEAR END_YEAR] [--config CONFIG_FILE]

Example:
    python main.py --mode forecast --years 2024 2030
    python main.py --mode scenario --config scenarios.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import simulation components
from src.simulation_engine import (
    BangladeshGDPSimulator,
    SimulationConfig,
    SimulationMode,
    ScenarioParameters,
    OutputFormat,
    create_default_config,
    create_sample_scenarios
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bangladesh_gdp_simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories for the simulation."""
    
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/simulation_outputs',
        'reports',
        'config',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created {len(directories)} directories")


def load_config_file(config_path: str) -> Dict:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config_data
    except Exception as e:
        logger.error(f"Error loading configuration file {config_path}: {str(e)}")
        return {}


def create_config_from_args(args) -> SimulationConfig:
    """Create simulation configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Simulation configuration
    """
    
    # Load configuration file if provided
    config_data = {}
    if args.config:
        config_data = load_config_file(args.config)
    
    # Determine simulation mode
    mode_mapping = {
        'historical': SimulationMode.HISTORICAL,
        'forecast': SimulationMode.FORECAST,
        'scenario': SimulationMode.SCENARIO,
        'sensitivity': SimulationMode.SENSITIVITY,
        'monte_carlo': SimulationMode.MONTE_CARLO
    }
    
    mode = mode_mapping.get(args.mode, SimulationMode.FORECAST)
    
    # Create configuration
    start_year, end_year = args.years
    config = SimulationConfig(
        start_date=datetime(start_year, 1, 1),
        end_date=datetime(end_year, 12, 31),
        mode=mode,
        base_year=config_data.get('base_year', 2023),
        currency=config_data.get('currency', 'BDT'),
        price_level=config_data.get('price_level', 'constant_2023'),
        frequency=config_data.get('frequency', 'quarterly'),
        include_climate_impacts=config_data.get('include_climate_impacts', True),
        include_adaptation_measures=config_data.get('include_adaptation_measures', True),
        monte_carlo_runs=config_data.get('monte_carlo_runs', 1000),
        confidence_intervals=config_data.get('confidence_intervals', [0.05, 0.25, 0.75, 0.95]),
        output_formats=[
            OutputFormat(fmt) for fmt in config_data.get('output_formats', ['json', 'csv', 'xlsx'])
        ],
        save_intermediate_results=config_data.get('save_intermediate_results', True),
        parallel_processing=config_data.get('parallel_processing', True),
        random_seed=config_data.get('random_seed', 42)
    )
    
    return config


def load_scenarios_from_file(scenarios_path: str) -> List[ScenarioParameters]:
    """Load scenarios from JSON file.
    
    Args:
        scenarios_path: Path to scenarios file
        
    Returns:
        List of scenario parameters
    """
    
    try:
        with open(scenarios_path, 'r') as f:
            scenarios_data = json.load(f)
        
        scenarios = []
        for scenario_data in scenarios_data:
            scenario = ScenarioParameters(
                name=scenario_data['name'],
                description=scenario_data['description'],
                economic_assumptions=scenario_data.get('economic_assumptions', {}),
                climate_assumptions=scenario_data.get('climate_assumptions', {}),
                policy_assumptions=scenario_data.get('policy_assumptions', {}),
                external_assumptions=scenario_data.get('external_assumptions', {}),
                sector_specific_assumptions=scenario_data.get('sector_specific_assumptions', {})
            )
            scenarios.append(scenario)
        
        logger.info(f"Loaded {len(scenarios)} scenarios from {scenarios_path}")
        return scenarios
        
    except Exception as e:
        logger.error(f"Error loading scenarios file {scenarios_path}: {str(e)}")
        return create_sample_scenarios()


def run_historical_analysis(simulator: BangladeshGDPSimulator):
    """Run historical analysis mode.
    
    Args:
        simulator: GDP simulator instance
    """
    
    logger.info("Running historical analysis...")
    
    # Load historical data
    data_loaded = simulator.load_historical_data()
    if not data_loaded:
        logger.warning("Historical data loading failed, using simulated data")
    
    # Run simulation
    results = simulator.run_simulation()
    
    # Generate report
    report = simulator.generate_simulation_report(results)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"reports/historical_analysis_{timestamp}.json")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Historical analysis completed. Report saved to {report_path}")
    
    # Print summary
    print("\n=== Historical Analysis Summary ===")
    exec_summary = report.get('executive_summary', {})
    for key, value in exec_summary.items():
        print(f"  {key}: {value}")


def run_forecast_analysis(simulator: BangladeshGDPSimulator):
    """Run forecast analysis mode.
    
    Args:
        simulator: GDP simulator instance
    """
    
    logger.info("Running forecast analysis...")
    
    # Load historical data for calibration
    data_loaded = simulator.load_historical_data()
    if not data_loaded:
        logger.warning("Historical data loading failed, using default parameters")
    
    # Run baseline forecast
    results = simulator.run_simulation()
    
    # Generate report
    report = simulator.generate_simulation_report(results)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"reports/forecast_analysis_{timestamp}.json")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Forecast analysis completed. Report saved to {report_path}")
    
    # Print summary
    print("\n=== Forecast Analysis Summary ===")
    exec_summary = report.get('executive_summary', {})
    for key, value in exec_summary.items():
        print(f"  {key}: {value}")
    
    # Print key forecasts
    gdp_analysis = report.get('gdp_analysis', {})
    if gdp_analysis:
        print("\n=== Key Forecasts ===")
        gdp_stats = gdp_analysis.get('gdp_statistics', {})
        print(f"  Average GDP: {gdp_stats.get('mean_gdp', 0):.0f} million BDT")
        print(f"  Average Growth Rate: {gdp_stats.get('mean_growth_rate', 0):.2%}")
        print(f"  GDP Volatility: {gdp_stats.get('growth_volatility', 0):.2%}")
        print(f"  Per Capita GDP: {gdp_stats.get('mean_per_capita_gdp', 0):.0f} BDT")


def run_scenario_analysis(simulator: BangladeshGDPSimulator, scenarios_file: str = None):
    """Run scenario analysis mode.
    
    Args:
        simulator: GDP simulator instance
        scenarios_file: Path to scenarios file (optional)
    """
    
    logger.info("Running scenario analysis...")
    
    # Load scenarios
    if scenarios_file and Path(scenarios_file).exists():
        scenarios = load_scenarios_from_file(scenarios_file)
    else:
        scenarios = create_sample_scenarios()
        logger.info("Using default sample scenarios")
    
    # Load historical data
    data_loaded = simulator.load_historical_data()
    if not data_loaded:
        logger.warning("Historical data loading failed, using default parameters")
    
    # Run scenario analysis
    scenario_results = simulator.run_scenario_analysis(scenarios)
    
    # Generate comparative report
    comparative_report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'scenarios_analyzed': len(scenarios),
        'scenario_results': {},
        'comparative_analysis': {}
    }
    
    # Process each scenario result
    gdp_comparisons = {}
    growth_comparisons = {}
    
    for scenario_name, results in scenario_results.items():
        if results:
            # Generate individual report
            scenario_report = simulator.generate_simulation_report(results)
            comparative_report['scenario_results'][scenario_name] = scenario_report
            
            # Extract key metrics for comparison
            gdp_stats = scenario_report.get('gdp_analysis', {}).get('gdp_statistics', {})
            gdp_comparisons[scenario_name] = gdp_stats.get('mean_gdp', 0)
            growth_comparisons[scenario_name] = gdp_stats.get('mean_growth_rate', 0)
        else:
            comparative_report['scenario_results'][scenario_name] = {'error': 'Simulation failed'}
    
    # Comparative analysis
    if gdp_comparisons:
        baseline_gdp = gdp_comparisons.get('baseline', 0)
        baseline_growth = growth_comparisons.get('baseline', 0)
        
        comparative_analysis = {
            'gdp_comparison': gdp_comparisons,
            'growth_comparison': growth_comparisons,
            'gdp_differences_from_baseline': {
                scenario: (gdp - baseline_gdp) / baseline_gdp * 100 if baseline_gdp > 0 else 0
                for scenario, gdp in gdp_comparisons.items()
            },
            'growth_differences_from_baseline': {
                scenario: (growth - baseline_growth) * 100
                for scenario, growth in growth_comparisons.items()
            }
        }
        
        comparative_report['comparative_analysis'] = comparative_analysis
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"reports/scenario_analysis_{timestamp}.json")
    
    with open(report_path, 'w') as f:
        json.dump(comparative_report, f, indent=2, default=str)
    
    logger.info(f"Scenario analysis completed. Report saved to {report_path}")
    
    # Print summary
    print("\n=== Scenario Analysis Summary ===")
    print(f"Scenarios analyzed: {len(scenarios)}")
    
    if gdp_comparisons:
        print("\n=== GDP Comparison (Million BDT) ===")
        for scenario, gdp in gdp_comparisons.items():
            print(f"  {scenario}: {gdp:.0f}")
        
        print("\n=== Growth Rate Comparison ===")
        for scenario, growth in growth_comparisons.items():
            print(f"  {scenario}: {growth:.2%}")
        
        if 'gdp_differences_from_baseline' in comparative_report['comparative_analysis']:
            print("\n=== Differences from Baseline ===")
            gdp_diffs = comparative_report['comparative_analysis']['gdp_differences_from_baseline']
            for scenario, diff in gdp_diffs.items():
                if scenario != 'baseline':
                    print(f"  {scenario}: {diff:+.1f}% GDP difference")


def run_sensitivity_analysis(simulator: BangladeshGDPSimulator):
    """Run sensitivity analysis mode.
    
    Args:
        simulator: GDP simulator instance
    """
    
    logger.info("Running sensitivity analysis...")
    
    # Load historical data
    data_loaded = simulator.load_historical_data()
    if not data_loaded:
        logger.warning("Historical data loading failed, using default parameters")
    
    # Define parameters for sensitivity analysis
    sensitivity_parameters = {
        'global_demand_growth': (0.01, 0.06),
        'domestic_demand_growth': (0.02, 0.10),
        'inflation_rate': (0.03, 0.10),
        'investment_growth': (0.04, 0.15),
        'exchange_rate_change': (-0.05, 0.05),
        'policy_support': (0.8, 1.3)
    }
    
    # Run sensitivity analysis
    sensitivity_results = simulator.run_sensitivity_analysis(
        sensitivity_parameters, num_samples=50
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"reports/sensitivity_analysis_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(sensitivity_results, f, indent=2, default=str)
    
    logger.info(f"Sensitivity analysis completed. Results saved to {results_path}")
    
    # Print summary
    print("\n=== Sensitivity Analysis Summary ===")
    print(f"Parameters analyzed: {len(sensitivity_parameters)}")
    
    print("\n=== GDP Sensitivity Coefficients ===")
    gdp_impacts = sensitivity_results.get('gdp_impacts', {})
    for param, impacts in gdp_impacts.items():
        sensitivity_coef = impacts.get('sensitivity_coefficient', 0)
        print(f"  {param}: {sensitivity_coef:.3f}")
    
    print("\n=== Sector Sensitivity Summary ===")
    sector_impacts = sensitivity_results.get('sector_impacts', {})
    for sector, sector_data in sector_impacts.items():
        print(f"\n  {sector.title()} Sector:")
        for param, impacts in sector_data.items():
            sensitivity_coef = impacts.get('sensitivity_coefficient', 0)
            print(f"    {param}: {sensitivity_coef:.3f}")


def run_monte_carlo_analysis(simulator: BangladeshGDPSimulator):
    """Run Monte Carlo analysis mode.
    
    Args:
        simulator: GDP simulator instance
    """
    
    logger.info("Running Monte Carlo analysis...")
    
    # Load historical data
    data_loaded = simulator.load_historical_data()
    if not data_loaded:
        logger.warning("Historical data loading failed, using default parameters")
    
    # Create scenarios with uncertainty
    base_scenario = create_sample_scenarios()[0]  # Use baseline scenario
    
    # Run multiple simulations with parameter uncertainty
    monte_carlo_results = []
    num_runs = simulator.config.monte_carlo_runs
    
    print(f"\nRunning {num_runs} Monte Carlo simulations...")
    
    for i in range(num_runs):
        try:
            # Add random variations to scenario parameters
            varied_scenario = ScenarioParameters(
                name=f"monte_carlo_run_{i+1}",
                description=f"Monte Carlo simulation run {i+1}",
                economic_assumptions={
                    key: value * (1 + np.random.normal(0, 0.1))  # 10% standard deviation
                    for key, value in base_scenario.economic_assumptions.items()
                },
                climate_assumptions=base_scenario.climate_assumptions,
                policy_assumptions=base_scenario.policy_assumptions,
                external_assumptions=base_scenario.external_assumptions,
                sector_specific_assumptions=base_scenario.sector_specific_assumptions
            )
            
            # Run simulation
            results = simulator.run_simulation(varied_scenario, save_results=False)
            
            # Extract key metrics
            avg_gdp = results.gdp_results['total_gdp'].mean()
            avg_growth = results.gdp_results['gdp_growth_rate'].mean()
            
            monte_carlo_results.append({
                'run': i + 1,
                'avg_gdp': avg_gdp,
                'avg_growth': avg_growth
            })
            
            # Progress update
            if (i + 1) % max(1, num_runs // 10) == 0:
                progress = (i + 1) / num_runs * 100
                print(f"  Progress: {progress:.1f}%")
                
        except Exception as e:
            logger.error(f"Error in Monte Carlo run {i+1}: {str(e)}")
    
    # Analyze results
    if monte_carlo_results:
        import numpy as np
        
        gdp_values = [result['avg_gdp'] for result in monte_carlo_results]
        growth_values = [result['avg_growth'] for result in monte_carlo_results]
        
        # Calculate statistics
        monte_carlo_analysis = {
            'total_runs': len(monte_carlo_results),
            'successful_runs': len(monte_carlo_results),
            'gdp_statistics': {
                'mean': np.mean(gdp_values),
                'median': np.median(gdp_values),
                'std': np.std(gdp_values),
                'min': np.min(gdp_values),
                'max': np.max(gdp_values),
                'percentiles': {
                    '5th': np.percentile(gdp_values, 5),
                    '25th': np.percentile(gdp_values, 25),
                    '75th': np.percentile(gdp_values, 75),
                    '95th': np.percentile(gdp_values, 95)
                }
            },
            'growth_statistics': {
                'mean': np.mean(growth_values),
                'median': np.median(growth_values),
                'std': np.std(growth_values),
                'min': np.min(growth_values),
                'max': np.max(growth_values),
                'percentiles': {
                    '5th': np.percentile(growth_values, 5),
                    '25th': np.percentile(growth_values, 25),
                    '75th': np.percentile(growth_values, 75),
                    '95th': np.percentile(growth_values, 95)
                }
            },
            'confidence_intervals': {
                'gdp_90_percent': (np.percentile(gdp_values, 5), np.percentile(gdp_values, 95)),
                'gdp_50_percent': (np.percentile(gdp_values, 25), np.percentile(gdp_values, 75)),
                'growth_90_percent': (np.percentile(growth_values, 5), np.percentile(growth_values, 95)),
                'growth_50_percent': (np.percentile(growth_values, 25), np.percentile(growth_values, 75))
            }
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(f"reports/monte_carlo_analysis_{timestamp}.json")
        
        full_results = {
            'analysis': monte_carlo_analysis,
            'raw_results': monte_carlo_results
        }
        
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        logger.info(f"Monte Carlo analysis completed. Results saved to {results_path}")
        
        # Print summary
        print("\n=== Monte Carlo Analysis Summary ===")
        print(f"Successful runs: {monte_carlo_analysis['successful_runs']}/{num_runs}")
        
        gdp_stats = monte_carlo_analysis['gdp_statistics']
        print(f"\n=== GDP Statistics (Million BDT) ===")
        print(f"  Mean: {gdp_stats['mean']:.0f}")
        print(f"  Median: {gdp_stats['median']:.0f}")
        print(f"  Standard Deviation: {gdp_stats['std']:.0f}")
        print(f"  Range: {gdp_stats['min']:.0f} - {gdp_stats['max']:.0f}")
        
        growth_stats = monte_carlo_analysis['growth_statistics']
        print(f"\n=== Growth Rate Statistics ===")
        print(f"  Mean: {growth_stats['mean']:.2%}")
        print(f"  Median: {growth_stats['median']:.2%}")
        print(f"  Standard Deviation: {growth_stats['std']:.2%}")
        print(f"  Range: {growth_stats['min']:.2%} - {growth_stats['max']:.2%}")
        
        confidence_intervals = monte_carlo_analysis['confidence_intervals']
        print(f"\n=== Confidence Intervals ===")
        gdp_90 = confidence_intervals['gdp_90_percent']
        gdp_50 = confidence_intervals['gdp_50_percent']
        print(f"  GDP 90% CI: {gdp_90[0]:.0f} - {gdp_90[1]:.0f} million BDT")
        print(f"  GDP 50% CI: {gdp_50[0]:.0f} - {gdp_50[1]:.0f} million BDT")
        
        growth_90 = confidence_intervals['growth_90_percent']
        growth_50 = confidence_intervals['growth_50_percent']
        print(f"  Growth 90% CI: {growth_90[0]:.2%} - {growth_90[1]:.2%}")
        print(f"  Growth 50% CI: {growth_50[0]:.2%} - {growth_50[1]:.2%}")
    
    else:
        print("\nMonte Carlo analysis failed - no successful runs")


def create_sample_config_files():
    """Create sample configuration files for users."""
    
    # Sample configuration file
    sample_config = {
        "base_year": 2023,
        "currency": "BDT",
        "price_level": "constant_2023",
        "frequency": "quarterly",
        "include_climate_impacts": True,
        "include_adaptation_measures": True,
        "monte_carlo_runs": 1000,
        "confidence_intervals": [0.05, 0.25, 0.75, 0.95],
        "output_formats": ["json", "csv", "xlsx"],
        "save_intermediate_results": True,
        "parallel_processing": True,
        "random_seed": 42
    }
    
    config_path = Path("config/sample_config.json")
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    # Sample scenarios file
    sample_scenarios = [
        {
            "name": "baseline",
            "description": "Baseline economic growth scenario",
            "economic_assumptions": {
                "global_demand_growth": 0.03,
                "domestic_demand_growth": 0.05,
                "inflation_rate": 0.06,
                "investment_growth": 0.08
            },
            "climate_assumptions": {
                "temperature_increase": 0.02,
                "precipitation_change": 0.0,
                "extreme_events_frequency": 1.0
            },
            "policy_assumptions": {
                "policy_support": 1.0,
                "infrastructure_investment": 1.0,
                "education_investment": 1.0
            },
            "external_assumptions": {
                "global_growth": 0.035,
                "commodity_prices": 1.0,
                "exchange_rate_stability": 1.0
            },
            "sector_specific_assumptions": {
                "agriculture": {"productivity_growth": 0.03},
                "manufacturing": {"export_growth": 0.08},
                "services": {"digitalization_rate": 0.05}
            }
        },
        {
            "name": "high_growth",
            "description": "Optimistic high growth scenario",
            "economic_assumptions": {
                "global_demand_growth": 0.04,
                "domestic_demand_growth": 0.07,
                "inflation_rate": 0.05,
                "investment_growth": 0.12
            },
            "climate_assumptions": {
                "temperature_increase": 0.015,
                "precipitation_change": 0.05,
                "extreme_events_frequency": 0.8
            },
            "policy_assumptions": {
                "policy_support": 1.2,
                "infrastructure_investment": 1.5,
                "education_investment": 1.3
            },
            "external_assumptions": {
                "global_growth": 0.04,
                "commodity_prices": 1.1,
                "exchange_rate_stability": 1.1
            },
            "sector_specific_assumptions": {
                "agriculture": {"productivity_growth": 0.05},
                "manufacturing": {"export_growth": 0.12},
                "services": {"digitalization_rate": 0.08}
            }
        }
    ]
    
    scenarios_path = Path("config/sample_scenarios.json")
    with open(scenarios_path, 'w') as f:
        json.dump(sample_scenarios, f, indent=2)
    
    logger.info(f"Created sample configuration files:")
    logger.info(f"  - {config_path}")
    logger.info(f"  - {scenarios_path}")


def main():
    """Main application entry point."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Bangladesh GDP Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode forecast --years 2024 2030
  python main.py --mode scenario --config config/sample_config.json --scenarios config/sample_scenarios.json
  python main.py --mode sensitivity --years 2024 2028
  python main.py --mode monte_carlo --years 2024 2026
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['historical', 'forecast', 'scenario', 'sensitivity', 'monte_carlo'],
        default='forecast',
        help='Simulation mode (default: forecast)'
    )
    
    parser.add_argument(
        '--years',
        nargs=2,
        type=int,
        default=[2024, 2030],
        metavar=('START_YEAR', 'END_YEAR'),
        help='Simulation period (default: 2024 2030)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON)'
    )
    
    parser.add_argument(
        '--scenarios',
        type=str,
        help='Path to scenarios file (JSON)'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup directories and create sample configuration files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup mode
    if args.setup:
        print("Setting up Bangladesh GDP Simulation System...")
        setup_directories()
        create_sample_config_files()
        print("Setup completed successfully!")
        print("\nYou can now run simulations using:")
        print("  python main.py --mode forecast --years 2024 2030")
        return
    
    # Validate years
    start_year, end_year = args.years
    if start_year >= end_year:
        print("Error: Start year must be before end year")
        sys.exit(1)
    
    if end_year - start_year > 20:
        print("Warning: Simulation period is very long (>20 years). This may take significant time.")
    
    # Setup directories
    setup_directories()
    
    # Create configuration
    config = create_config_from_args(args)
    
    print("\n" + "="*60)
    print("    BANGLADESH GDP SIMULATION SYSTEM")
    print("="*60)
    print(f"Mode: {config.mode.value.title()}")
    print(f"Period: {start_year} - {end_year}")
    print(f"Frequency: {config.frequency.title()}")
    print(f"Climate Impacts: {'Yes' if config.include_climate_impacts else 'No'}")
    print(f"Adaptation Measures: {'Yes' if config.include_adaptation_measures else 'No'}")
    print("="*60)
    
    try:
        # Initialize simulator
        simulator = BangladeshGDPSimulator(config, "data")
        
        # Run analysis based on mode
        if config.mode == SimulationMode.HISTORICAL:
            run_historical_analysis(simulator)
        
        elif config.mode == SimulationMode.FORECAST:
            run_forecast_analysis(simulator)
        
        elif config.mode == SimulationMode.SCENARIO:
            run_scenario_analysis(simulator, args.scenarios)
        
        elif config.mode == SimulationMode.SENSITIVITY:
            run_sensitivity_analysis(simulator)
        
        elif config.mode == SimulationMode.MONTE_CARLO:
            run_monte_carlo_analysis(simulator)
        
        print("\n" + "="*60)
        print("    SIMULATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved in: reports/")
        print(f"Logs saved in: bangladesh_gdp_simulation.log")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Check the log file for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()