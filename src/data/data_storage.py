"""Data Storage Module for Bangladesh GDP Simulation.

This module provides comprehensive data storage and retrieval capabilities
using SQLite database with support for time series data, metadata management,
and efficient querying for the GDP simulation system.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import pickle
from pathlib import Path
import hashlib
import threading
from contextlib import contextmanager
import warnings
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DataFrequency(Enum):
    """Data frequency types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class DataSource(Enum):
    """Data source types."""
    BBS = "bbs"
    BANGLADESH_BANK = "bangladesh_bank"
    WORLD_BANK = "world_bank"
    IMF = "imf"
    CLIMATE = "climate"
    SIMULATION = "simulation"
    MANUAL = "manual"


@dataclass
class DatasetMetadata:
    """Metadata for stored datasets."""
    dataset_id: str
    name: str
    description: str
    source: DataSource
    frequency: DataFrequency
    start_date: datetime
    end_date: datetime
    record_count: int
    columns: List[str]
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    tags: List[str] = None
    quality_score: float = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class DataStorage:
    """Comprehensive data storage system for Bangladesh economic data.
    
    This class provides database operations, time series storage,
    metadata management, and efficient data retrieval capabilities.
    """
    
    def __init__(self, db_path: str = None, config: Dict = None):
        """Initialize the data storage system.
        
        Args:
            db_path: Path to SQLite database file
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Set database path
        if db_path is None:
            db_path = self.config.get('db_path', './data/bangladesh_gdp.db')
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize database
        self._initialize_database()
        
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_size_limit = self.config.get('cache_size_limit', 100)
        
        logger.info(f"Data storage initialized with database: {self.db_path}")
    
    def _initialize_database(self):
        """Initialize database schema."""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    source TEXT NOT NULL,
                    frequency TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    columns TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version TEXT DEFAULT '1.0',
                    tags TEXT,
                    quality_score REAL
                )
            """)
            
            # Create time series data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS time_series_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES dataset_metadata (dataset_id),
                    UNIQUE(dataset_id, date)
                )
            """)
            
            # Create GDP data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gdp_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    quarter INTEGER,
                    gdp_total REAL NOT NULL,
                    gdp_agriculture REAL NOT NULL,
                    gdp_industry REAL NOT NULL,
                    gdp_services REAL NOT NULL,
                    gdp_growth_rate REAL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(date, source)
                )
            """)
            
            # Create population data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS population_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    total_population INTEGER NOT NULL,
                    urban_population INTEGER NOT NULL,
                    rural_population INTEGER NOT NULL,
                    urbanization_rate REAL,
                    population_growth_rate REAL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(date, source)
                )
            """)
            
            # Create employment data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    quarter INTEGER,
                    unemployment_rate REAL,
                    labor_force_participation REAL,
                    agriculture_employment_share REAL,
                    industry_employment_share REAL,
                    services_employment_share REAL,
                    informal_employment_rate REAL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(date, source)
                )
            """)
            
            # Create price data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    general_inflation_rate REAL,
                    food_inflation_rate REAL,
                    non_food_inflation_rate REAL,
                    general_price_index REAL,
                    food_price_index REAL,
                    non_food_price_index REAL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(date, source)
                )
            """)
            
            # Create agriculture data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agriculture_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    quarter INTEGER,
                    rice_production_tons INTEGER,
                    wheat_production_tons INTEGER,
                    jute_production_tons INTEGER,
                    agricultural_gdp_growth REAL,
                    climate_impact_factor REAL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(date, source)
                )
            """)
            
            # Create monetary data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monetary_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    money_supply_million_bdt REAL,
                    money_supply_growth_rate REAL,
                    repo_rate REAL,
                    lending_rate REAL,
                    deposit_rate REAL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(date, source)
                )
            """)
            
            # Create exchange rate data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exchange_rate_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    day INTEGER NOT NULL,
                    usd_bdt_rate REAL NOT NULL,
                    daily_change_percent REAL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(date, source)
                )
            """)
            
            # Create remittance data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS remittance_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    remittance_million_usd REAL NOT NULL,
                    remittance_growth_rate REAL,
                    seasonal_factor REAL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(date, source)
                )
            """)
            
            # Create simulation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    gdp_total REAL NOT NULL,
                    gdp_growth_rate REAL,
                    parameters_json TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(simulation_id, date, scenario_name)
                )
            """)
            
            # Create indices for better performance
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_gdp_date ON gdp_data(date)",
                "CREATE INDEX IF NOT EXISTS idx_gdp_year ON gdp_data(year)",
                "CREATE INDEX IF NOT EXISTS idx_population_date ON population_data(date)",
                "CREATE INDEX IF NOT EXISTS idx_employment_date ON employment_data(date)",
                "CREATE INDEX IF NOT EXISTS idx_price_date ON price_data(date)",
                "CREATE INDEX IF NOT EXISTS idx_agriculture_date ON agriculture_data(date)",
                "CREATE INDEX IF NOT EXISTS idx_monetary_date ON monetary_data(date)",
                "CREATE INDEX IF NOT EXISTS idx_exchange_date ON exchange_rate_data(date)",
                "CREATE INDEX IF NOT EXISTS idx_remittance_date ON remittance_data(date)",
                "CREATE INDEX IF NOT EXISTS idx_simulation_id ON simulation_results(simulation_id)",
                "CREATE INDEX IF NOT EXISTS idx_time_series_dataset ON time_series_data(dataset_id)",
                "CREATE INDEX IF NOT EXISTS idx_time_series_date ON time_series_data(date)"
            ]
            
            for index_sql in indices:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def store_dataset(self, 
                     data: Union[pd.DataFrame, List[Dict]], 
                     dataset_id: str,
                     metadata: DatasetMetadata = None,
                     table_name: str = None) -> bool:
        """Store a dataset in the database.
        
        Args:
            data: Data to store (DataFrame or list of dictionaries)
            dataset_id: Unique identifier for the dataset
            metadata: Dataset metadata
            table_name: Specific table name (if not using generic time_series_data)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                # Convert to DataFrame if needed
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = data.copy()
                
                if df.empty:
                    logger.warning(f"Empty dataset provided for {dataset_id}")
                    return False
                
                # Generate metadata if not provided
                if metadata is None:
                    metadata = self._generate_metadata(df, dataset_id)
                
                # Store metadata
                self._store_metadata(metadata)
                
                # Store data
                if table_name and self._table_exists(table_name):
                    success = self._store_to_specific_table(df, table_name)
                else:
                    success = self._store_to_time_series_table(df, dataset_id)
                
                if success:
                    logger.info(f"Successfully stored dataset {dataset_id} with {len(df)} records")
                    # Clear cache for this dataset
                    self._clear_cache(dataset_id)
                
                return success
                
        except Exception as e:
            logger.error(f"Error storing dataset {dataset_id}: {str(e)}")
            return False
    
    def _generate_metadata(self, df: pd.DataFrame, dataset_id: str) -> DatasetMetadata:
        """Generate metadata for a dataset."""
        
        # Determine data frequency and date range
        date_col = None
        for col in ['date', 'Date', 'DATE']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            start_date = df[date_col].min()
            end_date = df[date_col].max()
            
            # Determine frequency
            if len(df) > 1:
                date_diff = (df[date_col].max() - df[date_col].min()).days
                avg_interval = date_diff / (len(df) - 1)
                
                if avg_interval <= 2:
                    frequency = DataFrequency.DAILY
                elif avg_interval <= 10:
                    frequency = DataFrequency.WEEKLY
                elif avg_interval <= 35:
                    frequency = DataFrequency.MONTHLY
                elif avg_interval <= 100:
                    frequency = DataFrequency.QUARTERLY
                else:
                    frequency = DataFrequency.ANNUAL
            else:
                frequency = DataFrequency.MONTHLY
        else:
            start_date = datetime.now()
            end_date = datetime.now()
            frequency = DataFrequency.MONTHLY
        
        # Determine source
        source = DataSource.MANUAL
        if 'source' in df.columns and not df['source'].empty:
            source_value = df['source'].iloc[0].lower()
            if 'bbs' in source_value:
                source = DataSource.BBS
            elif 'bangladesh bank' in source_value or 'bb' in source_value:
                source = DataSource.BANGLADESH_BANK
            elif 'world bank' in source_value:
                source = DataSource.WORLD_BANK
            elif 'climate' in source_value or 'weather' in source_value:
                source = DataSource.CLIMATE
        
        return DatasetMetadata(
            dataset_id=dataset_id,
            name=dataset_id.replace('_', ' ').title(),
            description=f"Dataset containing {len(df)} records",
            source=source,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            record_count=len(df),
            columns=df.columns.tolist(),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _store_metadata(self, metadata: DatasetMetadata):
        """Store dataset metadata."""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Convert metadata to database format
            metadata_dict = asdict(metadata)
            metadata_dict['source'] = metadata.source.value
            metadata_dict['frequency'] = metadata.frequency.value
            metadata_dict['start_date'] = metadata.start_date.isoformat()
            metadata_dict['end_date'] = metadata.end_date.isoformat()
            metadata_dict['columns'] = json.dumps(metadata.columns)
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            metadata_dict['updated_at'] = metadata.updated_at.isoformat()
            metadata_dict['tags'] = json.dumps(metadata.tags or [])
            
            # Insert or update metadata
            cursor.execute("""
                INSERT OR REPLACE INTO dataset_metadata 
                (dataset_id, name, description, source, frequency, start_date, end_date,
                 record_count, columns, created_at, updated_at, version, tags, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata_dict['dataset_id'],
                metadata_dict['name'],
                metadata_dict['description'],
                metadata_dict['source'],
                metadata_dict['frequency'],
                metadata_dict['start_date'],
                metadata_dict['end_date'],
                metadata_dict['record_count'],
                metadata_dict['columns'],
                metadata_dict['created_at'],
                metadata_dict['updated_at'],
                metadata_dict['version'],
                metadata_dict['tags'],
                metadata_dict['quality_score']
            ))
            
            conn.commit()
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            return cursor.fetchone() is not None
    
    def _store_to_specific_table(self, df: pd.DataFrame, table_name: str) -> bool:
        """Store data to a specific table."""
        
        try:
            with self._get_connection() as conn:
                # Add created_at timestamp
                df['created_at'] = datetime.now().isoformat()
                
                # Store data
                df.to_sql(table_name, conn, if_exists='append', index=False)
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error storing to table {table_name}: {str(e)}")
            return False
    
    def _store_to_time_series_table(self, df: pd.DataFrame, dataset_id: str) -> bool:
        """Store data to generic time series table."""
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Find date column
                date_col = None
                for col in ['date', 'Date', 'DATE']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if not date_col:
                    logger.error(f"No date column found in dataset {dataset_id}")
                    return False
                
                # Store each row
                for _, row in df.iterrows():
                    row_dict = row.to_dict()
                    date_value = row_dict.pop(date_col)
                    
                    # Convert date to string if needed
                    if isinstance(date_value, pd.Timestamp):
                        date_value = date_value.isoformat()
                    elif isinstance(date_value, datetime):
                        date_value = date_value.isoformat()
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO time_series_data
                        (dataset_id, date, data_json, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (
                        dataset_id,
                        date_value,
                        json.dumps(row_dict, default=str),
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error storing to time series table: {str(e)}")
            return False
    
    def retrieve_dataset(self, 
                        dataset_id: str = None,
                        table_name: str = None,
                        start_date: str = None,
                        end_date: str = None,
                        columns: List[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve a dataset from the database.
        
        Args:
            dataset_id: Dataset identifier
            table_name: Specific table name
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            columns: Specific columns to retrieve
            
        Returns:
            DataFrame with retrieved data or None if not found
        """
        try:
            # Check cache first
            cache_key = f"{dataset_id or table_name}_{start_date}_{end_date}_{columns}"
            if cache_key in self._cache:
                logger.debug(f"Retrieved {dataset_id or table_name} from cache")
                return self._cache[cache_key].copy()
            
            with self._get_connection() as conn:
                if table_name and self._table_exists(table_name):
                    df = self._retrieve_from_specific_table(conn, table_name, start_date, end_date, columns)
                elif dataset_id:
                    df = self._retrieve_from_time_series_table(conn, dataset_id, start_date, end_date, columns)
                else:
                    logger.error("Either dataset_id or table_name must be provided")
                    return None
                
                # Cache the result
                if df is not None and not df.empty:
                    self._add_to_cache(cache_key, df)
                
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving dataset: {str(e)}")
            return None
    
    def _retrieve_from_specific_table(self, 
                                    conn: sqlite3.Connection,
                                    table_name: str,
                                    start_date: str = None,
                                    end_date: str = None,
                                    columns: List[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve data from a specific table."""
        
        # Build query
        if columns:
            column_str = ', '.join(columns)
        else:
            column_str = '*'
        
        query = f"SELECT {column_str} FROM {table_name}"
        params = []
        
        # Add date filters
        where_conditions = []
        if start_date:
            where_conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            where_conditions.append("date <= ?")
            params.append(end_date)
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        query += " ORDER BY date"
        
        # Execute query
        df = pd.read_sql_query(query, conn, params=params)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df if not df.empty else None
    
    def _retrieve_from_time_series_table(self, 
                                       conn: sqlite3.Connection,
                                       dataset_id: str,
                                       start_date: str = None,
                                       end_date: str = None,
                                       columns: List[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve data from time series table."""
        
        # Build query
        query = "SELECT date, data_json FROM time_series_data WHERE dataset_id = ?"
        params = [dataset_id]
        
        # Add date filters
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        # Execute query
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return None
        
        # Parse JSON data
        data_list = []
        for row in rows:
            date_value = row['date']
            data_dict = json.loads(row['data_json'])
            data_dict['date'] = date_value
            data_list.append(data_dict)
        
        df = pd.DataFrame(data_list)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter columns if specified
        if columns:
            available_columns = [col for col in columns if col in df.columns]
            if available_columns:
                df = df[available_columns]
        
        return df if not df.empty else None
    
    def get_dataset_metadata(self, dataset_id: str = None) -> Union[DatasetMetadata, List[DatasetMetadata]]:
        """Get metadata for one or all datasets.
        
        Args:
            dataset_id: Specific dataset ID, or None for all datasets
            
        Returns:
            DatasetMetadata object or list of DatasetMetadata objects
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if dataset_id:
                    cursor.execute(
                        "SELECT * FROM dataset_metadata WHERE dataset_id = ?",
                        (dataset_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_metadata(row)
                    else:
                        return None
                else:
                    cursor.execute("SELECT * FROM dataset_metadata ORDER BY created_at DESC")
                    rows = cursor.fetchall()
                    
                    return [self._row_to_metadata(row) for row in rows]
                    
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            return None if dataset_id else []
    
    def _row_to_metadata(self, row: sqlite3.Row) -> DatasetMetadata:
        """Convert database row to DatasetMetadata object."""
        
        return DatasetMetadata(
            dataset_id=row['dataset_id'],
            name=row['name'],
            description=row['description'],
            source=DataSource(row['source']),
            frequency=DataFrequency(row['frequency']),
            start_date=datetime.fromisoformat(row['start_date']),
            end_date=datetime.fromisoformat(row['end_date']),
            record_count=row['record_count'],
            columns=json.loads(row['columns']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            version=row['version'],
            tags=json.loads(row['tags'] or '[]'),
            quality_score=row['quality_score']
        )
    
    def delete_dataset(self, dataset_id: str, table_name: str = None) -> bool:
        """Delete a dataset from the database.
        
        Args:
            dataset_id: Dataset identifier
            table_name: Specific table name (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Delete from metadata
                    cursor.execute(
                        "DELETE FROM dataset_metadata WHERE dataset_id = ?",
                        (dataset_id,)
                    )
                    
                    # Delete from time series data
                    cursor.execute(
                        "DELETE FROM time_series_data WHERE dataset_id = ?",
                        (dataset_id,)
                    )
                    
                    # Delete from specific table if provided
                    if table_name and self._table_exists(table_name):
                        # This is more complex as we need to identify records by dataset_id
                        # For now, we'll skip this unless the table has a dataset_id column
                        pass
                    
                    conn.commit()
                    
                    # Clear cache
                    self._clear_cache(dataset_id)
                    
                    logger.info(f"Successfully deleted dataset {dataset_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
            return False
    
    def store_gdp_data(self, data: Union[pd.DataFrame, List[Dict]]) -> bool:
        """Store GDP data in the dedicated GDP table."""
        return self.store_dataset(data, 'gdp_data', table_name='gdp_data')
    
    def retrieve_gdp_data(self, 
                         start_date: str = None, 
                         end_date: str = None,
                         source: str = None) -> Optional[pd.DataFrame]:
        """Retrieve GDP data with optional filters."""
        
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM gdp_data"
                params = []
                where_conditions = []
                
                if start_date:
                    where_conditions.append("date >= ?")
                    params.append(start_date)
                if end_date:
                    where_conditions.append("date <= ?")
                    params.append(end_date)
                if source:
                    where_conditions.append("source = ?")
                    params.append(source)
                
                if where_conditions:
                    query += " WHERE " + " AND ".join(where_conditions)
                
                query += " ORDER BY date"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving GDP data: {str(e)}")
            return None
    
    def store_simulation_results(self, 
                               simulation_id: str,
                               results: Union[pd.DataFrame, List[Dict], str, Dict],
                               scenario_name: str = "default",
                               parameters: Dict = None) -> bool:
        """Store simulation results."""
        
        try:
            with self._lock:
                # Convert to DataFrame if needed
                if isinstance(results, str):
                    # Handle string results (e.g., JSON string)
                    try:
                        import json
                        results_data = json.loads(results)
                        if isinstance(results_data, list):
                            df = pd.DataFrame(results_data)
                        else:
                            df = pd.DataFrame([results_data])
                    except:
                        # If not JSON, create a simple record
                        df = pd.DataFrame([{'result': results}])
                elif isinstance(results, dict):
                    df = pd.DataFrame([results])
                elif isinstance(results, list):
                    df = pd.DataFrame(results)
                elif isinstance(results, pd.DataFrame):
                    df = results.copy()
                else:
                    # Fallback for other types
                    df = pd.DataFrame([{'result': str(results)}])
                
                # Add simulation metadata
                if not df.empty:
                    df['simulation_id'] = simulation_id
                    df['scenario_name'] = scenario_name
                    df['parameters_json'] = json.dumps(parameters or {})
                    df['created_at'] = datetime.now().isoformat()
                else:
                    # Create a minimal record for empty results
                    df = pd.DataFrame([{
                        'simulation_id': simulation_id,
                        'scenario_name': scenario_name,
                        'parameters_json': json.dumps(parameters or {}),
                        'created_at': datetime.now().isoformat(),
                        'status': 'empty_results'
                    }])
                
                with self._get_connection() as conn:
                    df.to_sql('simulation_results', conn, if_exists='append', index=False)
                    conn.commit()
                
                logger.info(f"Stored simulation results for {simulation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing simulation results: {str(e)}")
            return False
    
    def retrieve_simulation_results(self, 
                                  simulation_id: str = None,
                                  scenario_name: str = None,
                                  start_date: str = None,
                                  end_date: str = None) -> Optional[pd.DataFrame]:
        """Retrieve simulation results with optional filters."""
        
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM simulation_results"
                params = []
                where_conditions = []
                
                if simulation_id:
                    where_conditions.append("simulation_id = ?")
                    params.append(simulation_id)
                if scenario_name:
                    where_conditions.append("scenario_name = ?")
                    params.append(scenario_name)
                if start_date:
                    where_conditions.append("date >= ?")
                    params.append(start_date)
                if end_date:
                    where_conditions.append("date <= ?")
                    params.append(end_date)
                
                if where_conditions:
                    query += " WHERE " + " AND ".join(where_conditions)
                
                query += " ORDER BY date"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving simulation results: {str(e)}")
            return None
    
    def _add_to_cache(self, key: str, data: pd.DataFrame):
        """Add data to cache with size limit."""
        
        if len(self._cache) >= self._cache_size_limit:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = data.copy()
    
    def _clear_cache(self, dataset_id: str = None):
        """Clear cache for specific dataset or all cache."""
        
        if dataset_id:
            keys_to_remove = [key for key in self._cache.keys() if dataset_id in key]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {
                    'database_size_mb': self.db_path.stat().st_size / (1024 * 1024),
                    'total_datasets': 0,
                    'total_records': 0,
                    'table_stats': {},
                    'cache_size': len(self._cache)
                }
                
                # Get dataset count
                cursor.execute("SELECT COUNT(*) FROM dataset_metadata")
                stats['total_datasets'] = cursor.fetchone()[0]
                
                # Get table statistics
                tables = [
                    'gdp_data', 'population_data', 'employment_data', 'price_data',
                    'agriculture_data', 'monetary_data', 'exchange_rate_data',
                    'remittance_data', 'simulation_results', 'time_series_data'
                ]
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats['table_stats'][table] = count
                    stats['total_records'] += count
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    def export_data(self, 
                   output_path: str,
                   dataset_id: str = None,
                   table_name: str = None,
                   format: str = 'csv') -> bool:
        """Export data to file."""
        
        try:
            df = self.retrieve_dataset(dataset_id=dataset_id, table_name=table_name)
            
            if df is None or df.empty:
                logger.warning(f"No data found for export")
                return False
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'xlsx':
                df.to_excel(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', indent=2, date_format='iso')
            elif format == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        
        try:
            import shutil
            
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp to backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_path.parent / f"{backup_path.stem}_{timestamp}{backup_path.suffix}"
            
            shutil.copy2(self.db_path, backup_file)
            
            logger.info(f"Database backed up to {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up database: {str(e)}")
            return False
    
    def optimize_database(self) -> bool:
        """Optimize database performance."""
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Vacuum database
                cursor.execute("VACUUM")
                
                # Analyze tables for query optimization
                cursor.execute("ANALYZE")
                
                conn.commit()
                
            logger.info("Database optimized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing database: {str(e)}")
            return False
    
    def close(self):
        """Close database connections and cleanup."""
        
        self._clear_cache()
        logger.info("Data storage closed")