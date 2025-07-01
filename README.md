# BanglaMetrics: Bangladesh GDP Simulation System

## 🌟 Overview

BanglaMetrics is a comprehensive economic simulation platform designed to model Bangladesh's GDP dynamics, incorporating climate impacts, sector-specific analysis, and policy scenario evaluation. The system provides sophisticated tools for economic forecasting, policy analysis, and climate adaptation planning.

## 🚀 Key Features

### Economic Modeling
- **Multi-sector GDP simulation** with detailed subsector analysis
- **Quarterly, monthly, and annual** simulation frequencies
- **Historical analysis and forecasting** capabilities
- **Policy impact assessment** with customizable parameters
- **Five major economic sectors**: Agriculture, Manufacturing, Services, Construction, and Informal Economy

### Climate Integration
- **Climate data integration** from multiple sources
- **Climate impact assessment** on economic sectors
- **Weather pattern analysis** and extreme event modeling
- **Adaptation strategy evaluation** and cost-benefit analysis
- **Monsoon patterns, cyclones, floods, and drought modeling**

### Advanced Analytics
- **Scenario comparison** with multiple economic pathways
- **Sensitivity analysis** for parameter uncertainty
- **Monte Carlo simulation** for risk assessment
- **Comprehensive reporting** with actionable insights

### Data Management
- **Automated data collection** from various sources
- **Data quality validation** with comprehensive checks
- **Flexible data storage** with SQLite backend
- **Multiple export formats** (JSON, CSV, Excel, Parquet)

## 🏗️ System Architecture

```
BanglaMetrics/
├── src/
│   ├── sectors/              # Economic sector models
│   │   ├── agriculture.py    # Agriculture sector simulation
│   │   ├── manufacturing.py  # Manufacturing sector simulation
│   │   ├── services.py       # Services sector simulation
│   │   ├── construction.py   # Construction sector simulation
│   │   └── informal_economy.py # Informal economy modeling
│   ├── climate/              # Climate impact modeling
│   │   ├── climate_data.py   # Climate data management
│   │   ├── climate_impact.py # Climate impact assessment
│   │   ├── weather_patterns.py # Weather pattern analysis
│   │   └── adaptation_measures.py # Climate adaptation strategies
│   ├── data/                 # Data management
│   │   ├── data_collector.py # Automated data collection
│   │   ├── data_validator.py # Data quality validation
│   │   └── data_storage.py   # Data storage and retrieval
│   └── simulation_engine.py  # Core simulation engine
├── data/                     # Data storage
│   ├── raw/                  # Raw economic data
│   ├── processed/            # Processed datasets
│   └── simulation_outputs/   # Simulation results
├── config/                   # Configuration files
├── reports/                  # Generated reports
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for data collection

### Python Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
requests>=2.25.0
sqlalchemy>=1.4.0
openpyxl>=3.0.0
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/deluair/BanglaMetrics.git
cd BanglaMetrics
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup the System
```bash
python main.py --setup
```

This will create necessary directories and sample configuration files.

## 🎯 Quick Start

### Basic Forecast Simulation
```bash
python main.py --mode forecast --years 2024 2030
```

### Scenario Analysis
```bash
python main.py --mode scenario --config config/sample_config.json --scenarios config/sample_scenarios.json
```

### Sensitivity Analysis
```bash
python main.py --mode sensitivity --years 2024 2028
```

### Monte Carlo Simulation
```bash
python main.py --mode monte_carlo --years 2024 2026
```

## 📊 Usage Examples

### 1. Running a Basic GDP Forecast

```python
from src.simulation_engine import BangladeshGDPSimulator, create_default_config
from datetime import datetime

# Create configuration
config = create_default_config(2024, 2030)

# Initialize simulator
simulator = BangladeshGDPSimulator(config, "data")

# Load historical data
simulator.load_historical_data()

# Run simulation
results = simulator.run_simulation()

# Generate report
report = simulator.generate_simulation_report(results)
print(f"Average GDP: {report['executive_summary']['average_gdp']:.0f} million BDT")
```

### 2. Custom Scenario Analysis

```python
from src.simulation_engine import ScenarioParameters

# Define custom scenario
scenario = ScenarioParameters(
    name="high_investment",
    description="High infrastructure investment scenario",
    economic_assumptions={
        'investment_growth': 0.15,  # 15% investment growth
        'infrastructure_spending': 1.5  # 50% increase in infrastructure
    },
    policy_assumptions={
        'policy_support': 1.2  # 20% increase in policy support
    },
    sector_specific_assumptions={
        'construction': {'government_investment_growth': 0.20}
    }
)

# Run scenario
results = simulator.run_simulation(scenario)
```

### 3. Climate Impact Assessment

```python
from src.climate.climate_impact import ClimateImpactAssessment
from src.climate.climate_data import ClimateDataManager

# Initialize climate components
climate_data_manager = ClimateDataManager()
climate_impact_assessment = ClimateImpactAssessment()

# Get climate data
climate_data = climate_data_manager.get_climate_data(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    variables=['temperature', 'precipitation']
)

# Assess impacts
impacts = climate_impact_assessment.assess_impacts(
    climate_data=climate_data,
    sectors=['agriculture', 'manufacturing'],
    assessment_date=datetime(2024, 6, 1)
)
```

## Economic Context

### Bangladesh Economy (2024-25)
- **GDP**: $55.5 trillion BDT
- **Per Capita Income**: $2,820 USD
- **Growth Rate**: 3.97%
- **RMG Exports**: $50 billion (80%+ of total exports)
- **Remittances**: $22.1 billion (7.74% of GDP)

### Key Sectors Modeled
- **Ready-Made Garments (RMG)**: 4M workers, 82% of exports
- **Agriculture**: Climate-vulnerable rice production
- **Services**: Traditional banking + Mobile Financial Services
- **Informal Economy**: 27.2% of GDP

## Methodology

### GDP Calculation Approaches
1. **Production Approach**: Value-added by sector with input-output tracking
2. **Expenditure Approach**: Consumption, investment, government spending, net exports
3. **Income Approach**: Labor income, operating surplus, mixed income, remittances

### Climate Integration
- Historical cyclone patterns (Bhola 1970, Sidr 2007, Amphan 2020)
- Monsoon flood modeling
- Agricultural productivity impact
- Infrastructure damage and recovery costs

### Digital Economy Features
- MFS transaction volume correlation (bKash, Nagad, Rocket)
- Financial inclusion progression tracking
- E-commerce growth modeling
- Government digital service impact

## Data Sources

- Bangladesh Bureau of Statistics (BBS)
- Bangladesh Bank balance of payments
- Export Promotion Bureau (EPB)
- Bangladesh Garment Manufacturers and Exporters Association (BGMEA)
- Bangladesh Meteorological Department
- World Bank, IMF, ADB estimates

## Research Applications

- Policy simulation and scenario analysis
- Climate adaptation investment optimization
- Export diversification strategy testing
- Financial inclusion impact assessment
- Academic research in development economics

## Contributing

Please read our contributing guidelines and submit pull requests for improvements.

## License

MIT License - see LICENSE file for details.

## Contact

For questions and collaboration opportunities, please contact the development team.