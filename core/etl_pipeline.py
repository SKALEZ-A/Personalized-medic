"""
ETL Pipeline System for AI Personalized Medicine Platform
Comprehensive data extraction, transformation, and loading pipelines
"""

import json
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time
import asyncio
import random
import logging
import re
import hashlib
from pathlib import Path
import sqlite3

class ETLDataSource:
    """Data source abstraction for ETL operations"""

    def __init__(self, source_type: str, connection_params: Dict[str, Any]):
        self.source_type = source_type
        self.connection_params = connection_params
        self.is_connected = False

    def connect(self) -> bool:
        """Establish connection to data source"""
        try:
            if self.source_type == "database":
                # Simulate database connection
                self.connection = sqlite3.connect(self.connection_params.get("database", ":memory:"))
            elif self.source_type == "api":
                # Simulate API connection
                self.base_url = self.connection_params.get("base_url")
            elif self.source_type == "file":
                # File system connection
                self.base_path = Path(self.connection_params.get("base_path", "."))
            elif self.source_type == "stream":
                # Streaming data connection
                self.stream_url = self.connection_params.get("stream_url")

            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to {self.source_type}: {e}")
            return False

    def disconnect(self):
        """Close connection to data source"""
        if hasattr(self, 'connection'):
            self.connection.close()
        self.is_connected = False

    def extract_data(self, query_params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract data from source"""
        if not self.is_connected:
            raise ConnectionError("Not connected to data source")

        if self.source_type == "database":
            return self._extract_from_database(query_params)
        elif self.source_type == "api":
            return self._extract_from_api(query_params)
        elif self.source_type == "file":
            return self._extract_from_file(query_params)
        elif self.source_type == "stream":
            return self._extract_from_stream(query_params)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")

    def _extract_from_database(self, query_params: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from database"""
        query = query_params.get("query", "SELECT * FROM data_table")
        # Simulate database query
        data = []
        for i in range(random.randint(100, 1000)):
            record = {
                "id": i,
                "patient_id": f"PAT_{random.randint(1000, 9999)}",
                "timestamp": datetime.now() - timedelta(days=random.randint(0, 365)),
                "value": random.uniform(0, 100),
                "category": random.choice(["A", "B", "C", "D"])
            }
            data.append(record)

        return pd.DataFrame(data)

    def _extract_from_api(self, query_params: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from API"""
        endpoint = query_params.get("endpoint", "/data")
        # Simulate API call
        data = []
        for i in range(random.randint(50, 500)):
            record = {
                "api_id": f"API_{i}",
                "source": "external_api",
                "data_type": query_params.get("data_type", "generic"),
                "value": random.uniform(0, 100),
                "quality_score": random.uniform(0.7, 1.0)
            }
            data.append(record)

        return pd.DataFrame(data)

    def _extract_from_file(self, query_params: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from files"""
        file_pattern = query_params.get("file_pattern", "*.csv")
        # Simulate file reading
        data = []
        for i in range(random.randint(200, 2000)):
            record = {
                "file_id": f"FILE_{i}",
                "filename": f"data_{random.randint(1, 100)}.csv",
                "row_number": i,
                "field1": f"value_{random.randint(1, 100)}",
                "field2": random.uniform(0, 100),
                "field3": random.choice(["type_a", "type_b", "type_c"])
            }
            data.append(record)

        return pd.DataFrame(data)

    def _extract_from_stream(self, query_params: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from streaming source"""
        # Simulate streaming data extraction
        data = []
        for i in range(random.randint(10, 100)):
            record = {
                "stream_id": f"STREAM_{i}",
                "timestamp": datetime.now(),
                "sensor_id": f"sensor_{random.randint(1, 50)}",
                "measurement": random.uniform(0, 100),
                "unit": random.choice(["mg/dL", "mmHg", "bpm", "°C"])
            }
            data.append(record)

        return pd.DataFrame(data)

class ETLTransformer:
    """Data transformation engine for ETL pipelines"""

    def __init__(self):
        self.transformation_rules = {}
        self.data_quality_rules = {}
        self.initialize_transformations()

    def initialize_transformations(self):
        """Initialize transformation rules"""
        self.transformation_rules = {
            "normalize_numeric": self._normalize_numeric,
            "encode_categorical": self._encode_categorical,
            "handle_missing": self._handle_missing_values,
            "remove_outliers": self._remove_outliers,
            "feature_engineering": self._feature_engineering,
            "data_validation": self._data_validation,
            "standardize_units": self._standardize_units,
            "temporal_alignment": self._temporal_alignment
        }

        self.data_quality_rules = {
            "completeness": self._check_completeness,
            "accuracy": self._check_accuracy,
            "consistency": self._check_consistency,
            "timeliness": self._check_timeliness,
            "validity": self._check_validity,
            "uniqueness": self._check_uniqueness
        }

    def transform_data(self, data: pd.DataFrame, transformations: List[str],
                      quality_checks: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply transformations to data"""
        transformed_data = data.copy()
        quality_report = {}

        # Apply transformations
        for transformation in transformations:
            if transformation in self.transformation_rules:
                transformed_data, transform_report = self.transformation_rules[transformation](transformed_data)
                quality_report[f"transform_{transformation}"] = transform_report
            else:
                print(f"Unknown transformation: {transformation}")

        # Apply quality checks
        if quality_checks:
            for check in quality_checks:
                if check in self.data_quality_rules:
                    check_result = self.data_quality_rules[check](transformed_data)
                    quality_report[f"quality_{check}"] = check_result
                else:
                    print(f"Unknown quality check: {check}")

        return transformed_data, quality_report

    def _normalize_numeric(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize numeric columns"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if data[col].std() > 0:  # Avoid division by zero
                data[col] = (data[col] - data[col].mean()) / data[col].std()

        return data, {"normalized_columns": list(numeric_columns), "method": "zscore"}

    def _encode_categorical(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical columns"""
        categorical_columns = data.select_dtypes(include=['object']).columns
        encoded_columns = []

        for col in categorical_columns:
            unique_values = data[col].unique()
            if len(unique_values) <= 10:  # One-hot encode for low cardinality
                dummies = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data, dummies], axis=1)
                data = data.drop(col, axis=1)
                encoded_columns.extend(dummies.columns.tolist())

        return data, {"encoded_columns": encoded_columns, "method": "one_hot"}

    def _handle_missing_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values"""
        missing_report = {}

        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                missing_report[col] = {
                    "missing_count": int(missing_count),
                    "missing_percentage": float(missing_count / len(data))
                }

                # Fill missing values
                if data[col].dtype in ['int64', 'float64']:
                    data[col] = data[col].fillna(data[col].median())
                    missing_report[col]["fill_method"] = "median"
                else:
                    data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else "unknown")
                    missing_report[col]["fill_method"] = "mode"

        return data, missing_report

    def _remove_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove outliers using IQR method"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outliers_removed = {}

        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound))
            outliers_count = outliers.sum()

            if outliers_count > 0:
                data = data[~outliers]
                outliers_removed[col] = int(outliers_count)

        return data, {"outliers_removed": outliers_removed, "method": "iqr"}

    def _feature_engineering(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform feature engineering"""
        new_features = []

        # Create interaction features for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    interaction_name = f"{col1}_{col2}_interaction"
                    data[interaction_name] = data[col1] * data[col2]
                    new_features.append(interaction_name)

        # Create temporal features if timestamp exists
        timestamp_cols = [col for col in data.columns if 'timestamp' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            for col in timestamp_cols:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(data[col]):
                        data[col] = pd.to_datetime(data[col])

                    data[f"{col}_hour"] = data[col].dt.hour
                    data[f"{col}_day_of_week"] = data[col].dt.dayofweek
                    data[f"{col}_month"] = data[col].dt.month
                    new_features.extend([f"{col}_hour", f"{col}_day_of_week", f"{col}_month"])
                except:
                    pass  # Skip if conversion fails

        return data, {"new_features_created": new_features, "method": "automated"}

    def _data_validation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate data integrity"""
        validation_rules = {
            "age": lambda x: 0 <= x <= 150,
            "bmi": lambda x: 10 <= x <= 70,
            "blood_pressure": lambda x: 60 <= x <= 250,
            "glucose": lambda x: 20 <= x <= 600,
            "cholesterol": lambda x: 50 <= x <= 400
        }

        validation_report = {}

        for col in data.columns:
            if col.lower() in validation_rules:
                rule = validation_rules[col.lower()]
                valid_mask = data[col].apply(rule)
                invalid_count = (~valid_mask).sum()

                if invalid_count > 0:
                    validation_report[col] = {
                        "invalid_count": int(invalid_count),
                        "invalid_percentage": float(invalid_count / len(data))
                    }

        return data, validation_report

    def _standardize_units(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Standardize measurement units"""
        unit_conversions = {
            "weight": {
                "lbs": lambda x: x * 0.453592,  # lbs to kg
                "oz": lambda x: x * 0.0283495   # oz to kg
            },
            "height": {
                "inches": lambda x: x * 2.54,    # inches to cm
                "feet": lambda x: x * 30.48      # feet to cm
            },
            "temperature": {
                "fahrenheit": lambda x: (x - 32) * 5/9  # F to C
            }
        }

        standardization_report = {}

        # This would require column metadata to know which units are used
        # For simulation, we'll assume some standard conversions
        standardization_report["conversions_applied"] = []
        standardization_report["standard_unit"] = "SI_units"

        return data, standardization_report

    def _temporal_alignment(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Align temporal data"""
        temporal_report = {}

        # Find timestamp columns
        timestamp_cols = [col for col in data.columns if 'timestamp' in col.lower() or 'date' in col.lower()]

        if timestamp_cols:
            for col in timestamp_cols:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(data[col]):
                        data[col] = pd.to_datetime(data[col])

                    # Sort by timestamp
                    data = data.sort_values(col)

                    temporal_report[col] = {
                        "aligned": True,
                        "date_range": {
                            "start": data[col].min().isoformat(),
                            "end": data[col].max().isoformat()
                        },
                        "frequency": "irregular"  # Would analyze actual frequency
                    }
                except Exception as e:
                    temporal_report[col] = {"error": str(e)}

        return data, temporal_report

    def _check_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness"""
        completeness_report = {}

        for col in data.columns:
            missing_count = data[col].isnull().sum()
            completeness = 1 - (missing_count / len(data))
            completeness_report[col] = {
                "completeness_score": float(completeness),
                "missing_count": int(missing_count)
            }

        overall_completeness = sum(r["completeness_score"] for r in completeness_report.values()) / len(completeness_report)

        return {
            "overall_completeness": overall_completeness,
            "column_completeness": completeness_report
        }

    def _check_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data accuracy"""
        # This would involve domain-specific validation rules
        accuracy_report = {}

        # Check for impossible values
        accuracy_rules = {
            "age": lambda x: x > 0 and x < 150,
            "bmi": lambda x: x > 5 and x < 100,
            "blood_pressure_systolic": lambda x: x > 50 and x < 300,
            "blood_pressure_diastolic": lambda x: x > 20 and x < 200,
            "heart_rate": lambda x: x > 30 and x < 250
        }

        for col in data.columns:
            rule = None
            for rule_name, rule_func in accuracy_rules.items():
                if rule_name.lower() in col.lower():
                    rule = rule_func
                    break

            if rule:
                accurate_count = data[col].apply(rule).sum()
                accuracy_score = accurate_count / len(data) if len(data) > 0 else 0
                accuracy_report[col] = {
                    "accuracy_score": float(accuracy_score),
                    "accurate_count": int(accurate_count)
                }

        return {
            "overall_accuracy": sum(r["accuracy_score"] for r in accuracy_report.values()) / len(accuracy_report) if accuracy_report else 1.0,
            "column_accuracy": accuracy_report
        }

    def _check_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency"""
        consistency_report = {}

        # Check for logical inconsistencies
        if "blood_pressure_systolic" in data.columns and "blood_pressure_diastolic" in data.columns:
            inconsistent_bp = (data["blood_pressure_systolic"] <= data["blood_pressure_diastolic"]).sum()
            consistency_report["blood_pressure"] = {
                "consistent_count": int(len(data) - inconsistent_bp),
                "consistency_score": float((len(data) - inconsistent_bp) / len(data))
            }

        # Check date consistency
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        if len(date_cols) >= 2:
            for i, col1 in enumerate(date_cols):
                for col2 in date_cols[i+1:]:
                    try:
                        future_dates = (pd.to_datetime(data[col1]) > pd.to_datetime(data[col2])).sum()
                        consistency_report[f"{col1}_vs_{col2}"] = {
                            "consistent_count": int(len(data) - future_dates),
                            "consistency_score": float((len(data) - future_dates) / len(data))
                        }
                    except:
                        pass

        overall_consistency = sum(r["consistency_score"] for r in consistency_report.values()) / len(consistency_report) if consistency_report else 1.0

        return {
            "overall_consistency": overall_consistency,
            "consistency_checks": consistency_report
        }

    def _check_timeliness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data timeliness"""
        timeliness_report = {}

        # Find timestamp columns
        timestamp_cols = [col for col in data.columns if 'timestamp' in col.lower()]

        if timestamp_cols:
            for col in timestamp_cols:
                try:
                    timestamps = pd.to_datetime(data[col])
                    now = pd.Timestamp.now()

                    # Calculate age of data
                    data_age = (now - timestamps).dt.total_seconds()

                    # Check timeliness (data should not be too old)
                    max_age_days = 365  # 1 year
                    timely_count = (data_age < max_age_days * 24 * 3600).sum()

                    timeliness_report[col] = {
                        "timeliness_score": float(timely_count / len(data)),
                        "average_age_days": float(data_age.mean() / (24 * 3600)),
                        "max_age_days": float(data_age.max() / (24 * 3600))
                    }
                except Exception as e:
                    timeliness_report[col] = {"error": str(e)}

        return {
            "overall_timeliness": sum(r["timeliness_score"] for r in timeliness_report.values()) / len(timeliness_report) if timeliness_report else 1.0,
            "column_timeliness": timeliness_report
        }

    def _check_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity"""
        validity_report = {}

        # Define validity rules
        validity_rules = {
            "email": lambda x: bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(x))),
            "phone": lambda x: bool(re.match(r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$', str(x))),
            "zipcode": lambda x: bool(re.match(r'^\d{5}(-\d{4})?$', str(x))),
            "patient_id": lambda x: len(str(x)) >= 5,
            "date": lambda x: pd.to_datetime(x, errors='coerce') is not pd.NaT
        }

        for col in data.columns:
            rule = None
            for rule_name, rule_func in validity_rules.items():
                if rule_name.lower() in col.lower():
                    rule = rule_func
                    break

            if rule:
                valid_count = data[col].apply(rule).sum()
                validity_score = valid_count / len(data) if len(data) > 0 else 0
                validity_report[col] = {
                    "validity_score": float(validity_score),
                    "valid_count": int(valid_count)
                }

        return {
            "overall_validity": sum(r["validity_score"] for r in validity_report.values()) / len(validity_report) if validity_report else 1.0,
            "column_validity": validity_report
        }

    def _check_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data uniqueness"""
        uniqueness_report = {}

        for col in data.columns:
            unique_count = data[col].nunique()
            uniqueness_score = unique_count / len(data) if len(data) > 0 else 0

            # Check for duplicate values
            duplicate_count = data[col].duplicated().sum()

            uniqueness_report[col] = {
                "uniqueness_score": float(uniqueness_score),
                "unique_count": int(unique_count),
                "duplicate_count": int(duplicate_count),
                "duplicate_percentage": float(duplicate_count / len(data))
            }

        # Check composite uniqueness (e.g., patient_id + timestamp)
        key_columns = ["patient_id", "id", "record_id"]
        existing_key_cols = [col for col in key_columns if col in data.columns]

        if len(existing_key_cols) >= 2:
            composite_duplicates = data.duplicated(subset=existing_key_cols).sum()
            uniqueness_report["composite_key"] = {
                "uniqueness_score": float(1 - composite_duplicates / len(data)),
                "duplicate_count": int(composite_duplicates),
                "key_columns": existing_key_cols
            }

        return {
            "overall_uniqueness": sum(r["uniqueness_score"] for r in uniqueness_report.values()) / len(uniqueness_report),
            "column_uniqueness": uniqueness_report
        }

class ETLLoader:
    """Data loading engine for ETL pipelines"""

    def __init__(self):
        self.destinations = {}
        self.loading_strategies = {}
        self.initialize_loader()

    def initialize_loader(self):
        """Initialize loading capabilities"""
        self.loading_strategies = {
            "upsert": self._load_upsert,
            "append": self._load_append,
            "replace": self._load_replace,
            "merge": self._load_merge,
            "incremental": self._load_incremental
        }

    def load_data(self, data: pd.DataFrame, destination: str,
                 strategy: str = "upsert", options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load data to destination"""
        if options is None:
            options = {}

        if strategy not in self.loading_strategies:
            raise ValueError(f"Unsupported loading strategy: {strategy}")

        # Get destination configuration
        dest_config = self.destinations.get(destination)
        if not dest_config:
            # Create default destination
            dest_config = {"type": "database", "table": destination}

        # Execute loading
        result = self.loading_strategies[strategy](data, dest_config, options)

        return result

    def _load_upsert(self, data: pd.DataFrame, dest_config: Dict[str, Any],
                    options: Dict[str, Any]) -> Dict[str, Any]:
        """Load data with upsert strategy (insert or update)"""
        # Simulate upsert operation
        key_columns = options.get("key_columns", ["id"])

        # Check for existing records (simplified)
        existing_count = random.randint(0, len(data) // 2)
        new_records = len(data) - existing_count

        return {
            "strategy": "upsert",
            "total_records": len(data),
            "inserted_records": new_records,
            "updated_records": existing_count,
            "key_columns": key_columns,
            "destination": dest_config.get("table", "unknown")
        }

    def _load_append(self, data: pd.DataFrame, dest_config: Dict[str, Any],
                    options: Dict[str, Any]) -> Dict[str, Any]:
        """Load data by appending to existing data"""
        return {
            "strategy": "append",
            "total_records": len(data),
            "appended_records": len(data),
            "destination": dest_config.get("table", "unknown")
        }

    def _load_replace(self, data: pd.DataFrame, dest_config: Dict[str, Any],
                     options: Dict[str, Any]) -> Dict[str, Any]:
        """Load data by replacing existing data"""
        # Simulate backup of existing data
        backup_records = random.randint(1000, 10000)

        return {
            "strategy": "replace",
            "total_records": len(data),
            "replaced_records": len(data),
            "backup_records": backup_records,
            "destination": dest_config.get("table", "unknown")
        }

    def _load_merge(self, data: pd.DataFrame, dest_config: Dict[str, Any],
                   options: Dict[str, Any]) -> Dict[str, Any]:
        """Load data with merge strategy"""
        merge_key = options.get("merge_key", "id")

        return {
            "strategy": "merge",
            "total_records": len(data),
            "merge_key": merge_key,
            "destination": dest_config.get("table", "unknown")
        }

    def _load_incremental(self, data: pd.DataFrame, dest_config: Dict[str, Any],
                         options: Dict[str, Any]) -> Dict[str, Any]:
        """Load data incrementally based on timestamp"""
        timestamp_column = options.get("timestamp_column", "timestamp")

        # Filter for recent data
        if timestamp_column in data.columns:
            cutoff_date = datetime.now() - timedelta(days=options.get("days_back", 1))
            recent_data = data[pd.to_datetime(data[timestamp_column]) > cutoff_date]
        else:
            recent_data = data

        return {
            "strategy": "incremental",
            "total_records": len(data),
            "incremental_records": len(recent_data),
            "timestamp_column": timestamp_column,
            "days_back": options.get("days_back", 1),
            "destination": dest_config.get("table", "unknown")
        }

class ETLPipeline:
    """Complete ETL pipeline orchestration"""

    def __init__(self):
        self.pipelines = {}
        self.pipeline_runs = defaultdict(list)
        self.is_running = False
        self.pipeline_workers = []
        self.initialize_etl_system()

    def initialize_etl_system(self):
        """Initialize ETL system components"""
        self.extractor = ETLDataSource
        self.transformer = ETLTransformer()
        self.loader = ETLLoader()

        print("🔄 ETL Pipeline System initialized")

    def start_etl_system(self):
        """Start ETL system"""
        self.is_running = True

        # Start pipeline workers
        for i in range(3):  # 3 concurrent pipeline workers
            worker = threading.Thread(target=self._pipeline_worker, daemon=True)
            worker.start()
            self.pipeline_workers.append(worker)

        # Start monitoring worker
        monitor_worker = threading.Thread(target=self._pipeline_monitor, daemon=True)
        monitor_worker.start()
        self.pipeline_workers.append(monitor_worker)

        print("⚡ ETL Pipeline System started")

    def stop_etl_system(self):
        """Stop ETL system"""
        self.is_running = False
        print("🛑 ETL Pipeline System stopped")

    def create_pipeline(self, pipeline_config: Dict[str, Any]) -> str:
        """Create a new ETL pipeline"""
        pipeline_id = f"pipeline_{int(time.time())}_{random.randint(1000, 9999)}"

        pipeline = {
            "pipeline_id": pipeline_id,
            "name": pipeline_config.get("name", f"Pipeline {pipeline_id}"),
            "description": pipeline_config.get("description", ""),
            "source": pipeline_config["source"],
            "transformations": pipeline_config.get("transformations", []),
            "quality_checks": pipeline_config.get("quality_checks", []),
            "destination": pipeline_config["destination"],
            "loading_strategy": pipeline_config.get("loading_strategy", "upsert"),
            "schedule": pipeline_config.get("schedule"),  # cron-like schedule
            "dependencies": pipeline_config.get("dependencies", []),
            "error_handling": pipeline_config.get("error_handling", "fail_fast"),
            "monitoring": pipeline_config.get("monitoring", True),
            "created_at": datetime.now(),
            "status": "created"
        }

        self.pipelines[pipeline_id] = pipeline

        print(f"📋 Created ETL pipeline: {pipeline_id}")
        return pipeline_id

    def run_pipeline(self, pipeline_id: str, run_params: Dict[str, Any] = None) -> str:
        """Execute an ETL pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        if run_params is None:
            run_params = {}

        pipeline = self.pipelines[pipeline_id]
        run_id = f"run_{pipeline_id}_{int(time.time())}"

        run_record = {
            "run_id": run_id,
            "pipeline_id": pipeline_id,
            "status": "queued",
            "started_at": None,
            "completed_at": None,
            "run_params": run_params,
            "stages": {
                "extraction": {"status": "pending"},
                "transformation": {"status": "pending"},
                "loading": {"status": "pending"}
            },
            "metrics": {},
            "errors": [],
            "logs": []
        }

        self.pipeline_runs[pipeline_id].append(run_record)

        # Queue pipeline execution
        asyncio.run_coroutine_threadsafe(
            self._pipeline_queue.put(run_record),
            asyncio.get_event_loop()
        )

        print(f"▶️ Pipeline execution queued: {run_id}")
        return run_id

    async def initialize_pipeline_queue(self):
        """Initialize the pipeline queue"""
        self._pipeline_queue = asyncio.Queue()

    def _pipeline_worker(self):
        """Background pipeline execution worker"""
        while self.is_running:
            try:
                # Get pipeline run from queue
                run_record = asyncio.run(self._pipeline_queue.get())

                # Execute pipeline
                self._execute_pipeline(run_record)

                self._pipeline_queue.task_done()

            except asyncio.QueueEmpty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Pipeline worker error: {e}")

    def _execute_pipeline(self, run_record: Dict[str, Any]):
        """Execute a complete ETL pipeline"""
        try:
            run_record["status"] = "running"
            run_record["started_at"] = datetime.now()

            pipeline_id = run_record["pipeline_id"]
            pipeline = self.pipelines[pipeline_id]

            # Stage 1: Extraction
            run_record["stages"]["extraction"]["status"] = "running"
            run_record["stages"]["extraction"]["started_at"] = datetime.now()

            source_config = pipeline["source"]
            source = self.extractor(source_config["type"], source_config.get("config", {}))

            if source.connect():
                extracted_data = source.extract_data(run_record["run_params"])
                source.disconnect()

                run_record["stages"]["extraction"]["status"] = "completed"
                run_record["stages"]["extraction"]["completed_at"] = datetime.now()
                run_record["stages"]["extraction"]["records_extracted"] = len(extracted_data)
                run_record["logs"].append("Extraction completed successfully")
            else:
                raise Exception("Failed to connect to data source")

            # Stage 2: Transformation
            run_record["stages"]["transformation"]["status"] = "running"
            run_record["stages"]["transformation"]["started_at"] = datetime.now()

            transformed_data, quality_report = self.transformer.transform_data(
                extracted_data,
                pipeline["transformations"],
                pipeline["quality_checks"]
            )

            run_record["stages"]["transformation"]["status"] = "completed"
            run_record["stages"]["transformation"]["completed_at"] = datetime.now()
            run_record["stages"]["transformation"]["records_transformed"] = len(transformed_data)
            run_record["stages"]["transformation"]["quality_report"] = quality_report
            run_record["logs"].append("Transformation completed successfully")

            # Stage 3: Loading
            run_record["stages"]["loading"]["status"] = "running"
            run_record["stages"]["loading"]["started_at"] = datetime.now()

            loading_options = run_record["run_params"].get("loading_options", {})
            load_result = self.loader.load_data(
                transformed_data,
                pipeline["destination"],
                pipeline["loading_strategy"],
                loading_options
            )

            run_record["stages"]["loading"]["status"] = "completed"
            run_record["stages"]["loading"]["completed_at"] = datetime.now()
            run_record["stages"]["loading"]["load_result"] = load_result
            run_record["logs"].append("Loading completed successfully")

            # Finalize run
            run_record["status"] = "completed"
            run_record["completed_at"] = datetime.now()

            # Calculate metrics
            total_time = (run_record["completed_at"] - run_record["started_at"]).total_seconds()
            run_record["metrics"] = {
                "total_runtime_seconds": total_time,
                "records_processed": len(transformed_data),
                "throughput_records_per_second": len(transformed_data) / total_time if total_time > 0 else 0,
                "data_quality_score": self._calculate_data_quality_score(quality_report),
                "success_rate": 1.0
            }

            print(f"✅ Pipeline execution completed: {run_record['run_id']}")

        except Exception as e:
            run_record["status"] = "failed"
            run_record["error"] = str(e)
            run_record["completed_at"] = datetime.now()
            run_record["logs"].append(f"Pipeline failed: {str(e)}")

            # Update failed stages
            for stage_name, stage_info in run_record["stages"].items():
                if stage_info["status"] == "running":
                    stage_info["status"] = "failed"
                    stage_info["error"] = str(e)

            print(f"❌ Pipeline execution failed: {run_record['run_id']} - {e}")

    def _calculate_data_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        quality_scores = []

        for report_type, report_data in quality_report.items():
            if "overall" in report_data:
                quality_scores.append(report_data["overall"])
            elif isinstance(report_data, dict):
                # Average of individual scores
                scores = [v for v in report_data.values() if isinstance(v, (int, float)) and "score" in k.lower()]
                if scores:
                    quality_scores.append(sum(scores) / len(scores))

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.8

    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline status and recent runs"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        pipeline = self.pipelines[pipeline_id]
        runs = self.pipeline_runs[pipeline_id][-5:]  # Last 5 runs

        # Calculate success rate
        completed_runs = [r for r in runs if r["status"] in ["completed", "failed"]]
        success_rate = len([r for r in completed_runs if r["status"] == "completed"]) / len(completed_runs) if completed_runs else 0

        return {
            "pipeline_id": pipeline_id,
            "pipeline_info": {
                "name": pipeline["name"],
                "status": pipeline["status"],
                "last_run": runs[-1] if runs else None,
                "total_runs": len(runs),
                "success_rate": success_rate
            },
            "recent_runs": runs
        }

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get overall ETL pipeline metrics"""
        total_pipelines = len(self.pipelines)
        active_pipelines = len([p for p in self.pipelines.values() if p["status"] == "active"])

        all_runs = []
        for runs in self.pipeline_runs.values():
            all_runs.extend(runs)

        if all_runs:
            completed_runs = [r for r in all_runs if r["status"] == "completed"]
            failed_runs = [r for r in all_runs if r["status"] == "failed"]

            success_rate = len(completed_runs) / len(all_runs) if all_runs else 0
            avg_runtime = sum(r["metrics"].get("total_runtime_seconds", 0) for r in completed_runs) / len(completed_runs) if completed_runs else 0
            total_records_processed = sum(r["metrics"].get("records_processed", 0) for r in completed_runs)
        else:
            success_rate = 0
            avg_runtime = 0
            total_records_processed = 0

        return {
            "total_pipelines": total_pipelines,
            "active_pipelines": active_pipelines,
            "total_runs": len(all_runs),
            "success_rate": success_rate,
            "average_runtime_seconds": avg_runtime,
            "total_records_processed": total_records_processed,
            "data_quality_avg": sum(r["metrics"].get("data_quality_score", 0.8) for r in all_runs if r.get("metrics")) / len([r for r in all_runs if r.get("metrics")]) if all_runs else 0
        }

    def _pipeline_monitor(self):
        """Background pipeline monitoring"""
        while self.is_running:
            try:
                # Monitor pipeline health
                for pipeline_id, pipeline in self.pipelines.items():
                    runs = self.pipeline_runs[pipeline_id][-10:]  # Last 10 runs

                    if runs:
                        recent_failures = len([r for r in runs if r["status"] == "failed"])
                        if recent_failures >= 3:  # 3 or more recent failures
                            print(f"⚠️ Pipeline {pipeline_id} has {recent_failures} recent failures")

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                print(f"Pipeline monitor error: {e}")

    def export_pipeline_report(self, pipeline_id: str, format: str = "json") -> str:
        """Export comprehensive pipeline report"""
        pipeline = self.pipelines[pipeline_id]
        runs = self.pipeline_runs[pipeline_id]

        report = {
            "pipeline_id": pipeline_id,
            "pipeline_config": pipeline,
            "execution_history": runs,
            "performance_metrics": self._calculate_pipeline_performance(pipeline_id),
            "data_quality_trends": self._analyze_data_quality_trends(runs),
            "generated_at": datetime.now()
        }

        # Export in requested format
        if format == "json":
            return json.dumps(report, indent=2, default=str)
        else:
            return str(report)

    def _calculate_pipeline_performance(self, pipeline_id: str) -> Dict[str, Any]:
        """Calculate pipeline performance metrics"""
        runs = [r for r in self.pipeline_runs[pipeline_id] if r["status"] == "completed"]

        if not runs:
            return {"error": "No completed runs"}

        runtimes = [r["metrics"]["total_runtime_seconds"] for r in runs]
        throughputs = [r["metrics"]["throughput_records_per_second"] for r in runs]
        quality_scores = [r["metrics"]["data_quality_score"] for r in runs]

        return {
            "average_runtime": sum(runtimes) / len(runtimes),
            "runtime_std": statistics.stdev(runtimes) if len(runtimes) > 1 else 0,
            "average_throughput": sum(throughputs) / len(throughputs),
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "total_records_processed": sum(r["metrics"]["records_processed"] for r in runs),
            "best_runtime": min(runtimes),
            "worst_runtime": max(runtimes)
        }

    def _analyze_data_quality_trends(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data quality trends over time"""
        completed_runs = [r for r in runs if r["status"] == "completed"]

        if not completed_runs:
            return {"error": "No completed runs"}

        quality_scores = [r["metrics"]["data_quality_score"] for r in completed_runs]
        timestamps = [r["completed_at"] for r in completed_runs]

        # Calculate trend (simplified)
        if len(quality_scores) >= 2:
            trend = "improving" if quality_scores[-1] > quality_scores[0] else "declining"
        else:
            trend = "stable"

        return {
            "quality_trend": trend,
            "average_quality": sum(quality_scores) / len(quality_scores),
            "quality_volatility": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            "best_quality_score": max(quality_scores),
            "worst_quality_score": min(quality_scores)
        }
