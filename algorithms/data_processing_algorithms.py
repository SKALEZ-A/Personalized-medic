"""
Advanced Data Processing Algorithms for AI Personalized Medicine Platform
Comprehensive ETL pipelines, data validation, and preprocessing algorithms
"""

import math
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
import json
import hashlib
import re
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import queue


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality assessment metrics"""
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    timeliness_score: float = 0.0
    validity_score: float = 0.0
    uniqueness_score: float = 0.0
    overall_quality_score: float = 0.0
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    duplicate_records: int = 0
    outlier_records: int = 0
    data_types: Dict[str, str] = field(default_factory=dict)
    range_violations: Dict[str, int] = field(default_factory=dict)
    format_violations: Dict[str, int] = field(default_factory=dict)
    assessment_timestamp: datetime = None

    def __post_init__(self):
        if self.assessment_timestamp is None:
            self.assessment_timestamp = datetime.now()


@dataclass
class ETLJob:
    """ETL job configuration and tracking"""
    job_id: str
    source_type: str
    target_type: str
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    quality_metrics: Optional[DataQualityMetrics] = None
    transformation_rules: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class BaseDataProcessor(ABC):
    """Abstract base class for data processors"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False
        self.processing_stats = {
            'records_processed': 0,
            'records_failed': 0,
            'processing_time': 0.0,
            'memory_usage': 0.0
        }

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data"""
        pass

    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate data"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()


class DataValidationProcessor(BaseDataProcessor):
    """Advanced data validation processor with healthcare-specific rules"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.validation_rules = self.config.get('validation_rules', {})
        self.custom_validators = self.config.get('custom_validators', {})
        self.is_initialized = True

    def process(self, data: Any) -> Any:
        """Validate and clean data"""
        start_time = time.time()

        if isinstance(data, dict):
            validated_data, errors = self._validate_record(data)
            self.processing_stats['records_processed'] += 1
            if errors:
                self.processing_stats['records_failed'] += 1
        elif isinstance(data, list):
            validated_data = []
            for record in data:
                if isinstance(record, dict):
                    valid_record, errors = self._validate_record(record)
                    self.processing_stats['records_processed'] += 1
                    if errors:
                        self.processing_stats['records_failed'] += 1
                    else:
                        validated_data.append(valid_record)
        else:
            raise ValueError("Data must be a dictionary or list of dictionaries")

        self.processing_stats['processing_time'] += time.time() - start_time
        return validated_data

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate data and return validation results"""
        if isinstance(data, dict):
            _, errors = self._validate_record(data)
            return len(errors) == 0, errors
        elif isinstance(data, list):
            all_errors = []
            for record in data:
                if isinstance(record, dict):
                    _, errors = self._validate_record(record)
                    all_errors.extend(errors)
            return len(all_errors) == 0, all_errors
        return False, ["Invalid data format"]

    def _validate_record(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validate a single record"""
        errors = []
        validated_record = record.copy()

        # Apply validation rules
        for field_name, field_value in record.items():
            field_rules = self.validation_rules.get(field_name, {})

            # Required field validation
            if field_rules.get('required', False) and (field_value is None or field_value == ""):
                errors.append(f"Field '{field_name}' is required")
                continue

            # Skip validation if field is None and not required
            if field_value is None:
                continue

            # Data type validation
            expected_type = field_rules.get('type')
            if expected_type:
                if not self._validate_type(field_value, expected_type):
                    errors.append(f"Field '{field_name}' must be of type {expected_type}")
                    continue

            # Range validation
            min_val = field_rules.get('min')
            max_val = field_rules.get('max')
            if min_val is not None and isinstance(field_value, (int, float)) and field_value < min_val:
                errors.append(f"Field '{field_name}' must be >= {min_val}")
            if max_val is not None and isinstance(field_value, (int, float)) and field_value > max_val:
                errors.append(f"Field '{field_name}' must be <= {max_val}")

            # Pattern validation
            pattern = field_rules.get('pattern')
            if pattern and isinstance(field_value, str):
                if not re.match(pattern, field_value):
                    errors.append(f"Field '{field_name}' does not match required pattern")

            # Enum validation
            allowed_values = field_rules.get('enum')
            if allowed_values and field_value not in allowed_values:
                errors.append(f"Field '{field_name}' must be one of: {allowed_values}")

            # Custom validation
            custom_validator = field_rules.get('custom_validator')
            if custom_validator and custom_validator in self.custom_validators:
                validator_func = self.custom_validators[custom_validator]
                if not validator_func(field_value):
                    errors.append(f"Field '{field_name}' failed custom validation")

            # Healthcare-specific validations
            if field_name.lower().startswith(('patient_id', 'medical_id')):
                if not self._validate_patient_id(field_value):
                    errors.append(f"Invalid patient ID format: {field_name}")

            if 'date' in field_name.lower() or field_name.endswith('_date'):
                if not self._validate_date(field_value):
                    errors.append(f"Invalid date format: {field_name}")

            if field_name.lower().endswith(('_value', '_level', '_score')):
                if not self._validate_numeric_range(field_value, field_name):
                    errors.append(f"Invalid numeric value: {field_name}")

        return validated_record, errors

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate data type"""
        type_map = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict
        }

        expected_python_type = type_map.get(expected_type.lower())
        if expected_python_type:
            return isinstance(value, expected_python_type)

        return True  # Unknown type, assume valid

    def _validate_patient_id(self, patient_id: str) -> bool:
        """Validate patient ID format (healthcare-specific)"""
        if not isinstance(patient_id, str):
            return False

        # Common patient ID patterns
        patterns = [
            r'^[A-Z]{2}\d{6,10}$',  # AA12345678
            r'^\d{8,12}$',          # 12345678
            r'^[A-Z]{3}\d{5,8}$',   # AAA12345
            r'^P\d{7,10}$',         # P12345678
        ]

        return any(re.match(pattern, patient_id) for pattern in patterns)

    def _validate_date(self, date_value: Any) -> bool:
        """Validate date format"""
        if isinstance(date_value, datetime):
            return True

        if isinstance(date_value, str):
            try:
                # Try common date formats
                formats = [
                    '%Y-%m-%d',
                    '%Y/%m/%d',
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%SZ'
                ]
                for fmt in formats:
                    try:
                        datetime.strptime(date_value, fmt)
                        return True
                    except ValueError:
                        continue
            except:
                pass

        return False

    def _validate_numeric_range(self, value: Any, field_name: str) -> bool:
        """Validate numeric ranges for healthcare data"""
        if not isinstance(value, (int, float)):
            return False

        # Healthcare-specific range validations
        range_rules = {
            'blood_pressure_systolic': (70, 250),
            'blood_pressure_diastolic': (40, 150),
            'heart_rate': (30, 200),
            'temperature': (30.0, 45.0),
            'glucose': (20, 600),
            'cholesterol': (50, 400),
            'bmi': (10.0, 70.0),
            'age': (0, 150),
            'weight': (1.0, 500.0),
            'height': (30.0, 250.0)
        }

        # Check if field name contains any of the keys
        for key, (min_val, max_val) in range_rules.items():
            if key in field_name.lower():
                return min_val <= value <= max_val

        # Default numeric validation
        return -1e10 <= value <= 1e10


class DataTransformationProcessor(BaseDataProcessor):
    """Advanced data transformation processor with ETL capabilities"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.transformation_rules = self.config.get('transformation_rules', {})
        self.custom_transformers = self.config.get('custom_transformers', {})
        self.is_initialized = True

    def process(self, data: Any) -> Any:
        """Transform data according to rules"""
        start_time = time.time()

        if isinstance(data, dict):
            transformed_data = self._transform_record(data)
            self.processing_stats['records_processed'] += 1
        elif isinstance(data, list):
            transformed_data = []
            for record in data:
                if isinstance(record, dict):
                    transformed_record = self._transform_record(record)
                    transformed_data.append(transformed_record)
                    self.processing_stats['records_processed'] += 1
        else:
            raise ValueError("Data must be a dictionary or list of dictionaries")

        self.processing_stats['processing_time'] += time.time() - start_time
        return transformed_data

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate transformation rules"""
        errors = []

        if not isinstance(self.transformation_rules, dict):
            errors.append("Transformation rules must be a dictionary")
            return False, errors

        # Validate rule structure
        for field_name, rules in self.transformation_rules.items():
            if not isinstance(rules, dict):
                errors.append(f"Rules for field '{field_name}' must be a dictionary")
                continue

            for rule_type, rule_config in rules.items():
                if rule_type not in ['map', 'calculate', 'normalize', 'encode', 'custom']:
                    errors.append(f"Unknown transformation rule type: {rule_type}")

        return len(errors) == 0, errors

    def _transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single record"""
        transformed_record = record.copy()

        # Apply transformation rules
        for field_name, field_rules in self.transformation_rules.items():
            if field_name not in record:
                continue

            field_value = record[field_name]

            for rule_type, rule_config in field_rules.items():
                try:
                    if rule_type == 'map':
                        transformed_record[field_name] = self._apply_mapping(field_value, rule_config)
                    elif rule_type == 'calculate':
                        transformed_record[field_name] = self._apply_calculation(field_value, rule_config, record)
                    elif rule_type == 'normalize':
                        transformed_record[field_name] = self._apply_normalization(field_value, rule_config)
                    elif rule_type == 'encode':
                        transformed_record[field_name] = self._apply_encoding(field_value, rule_config)
                    elif rule_type == 'custom':
                        transformed_record[field_name] = self._apply_custom_transform(field_value, rule_config, record)
                except Exception as e:
                    # Log error but continue processing
                    print(f"Transformation error for field '{field_name}': {e}")
                    continue

        return transformed_record

    def _apply_mapping(self, value: Any, mapping_config: Dict[str, Any]) -> Any:
        """Apply value mapping transformation"""
        mapping_dict = mapping_config.get('mapping', {})

        if isinstance(value, str):
            return mapping_dict.get(value, value)
        elif isinstance(value, (int, float)):
            return mapping_dict.get(value, value)

        return value

    def _apply_calculation(self, value: Any, calc_config: Dict[str, Any], record: Dict[str, Any]) -> Any:
        """Apply calculation transformation"""
        operation = calc_config.get('operation', 'add')
        operands = calc_config.get('operands', [])

        if not isinstance(value, (int, float)):
            return value

        result = value

        for operand in operands:
            if isinstance(operand, str) and operand in record:
                operand_value = record[operand]
            elif isinstance(operand, (int, float)):
                operand_value = operand
            else:
                continue

            if operation == 'add':
                result += operand_value
            elif operation == 'subtract':
                result -= operand_value
            elif operation == 'multiply':
                result *= operand_value
            elif operation == 'divide' and operand_value != 0:
                result /= operand_value
            elif operation == 'power':
                result = result ** operand_value

        return result

    def _apply_normalization(self, value: Any, norm_config: Dict[str, Any]) -> Any:
        """Apply normalization transformation"""
        method = norm_config.get('method', 'minmax')

        if not isinstance(value, (int, float)):
            return value

        if method == 'minmax':
            min_val = norm_config.get('min', 0)
            max_val = norm_config.get('max', 1)
            data_min = norm_config.get('data_min', 0)
            data_max = norm_config.get('data_max', 100)

            if data_max != data_min:
                return min_val + (value - data_min) * (max_val - min_val) / (data_max - data_min)

        elif method == 'zscore':
            mean_val = norm_config.get('mean', 0)
            std_val = norm_config.get('std', 1)
            return (value - mean_val) / std_val if std_val != 0 else value

        elif method == 'robust':
            median_val = norm_config.get('median', 0)
            mad_val = norm_config.get('mad', 1)  # Median Absolute Deviation
            return (value - median_val) / mad_val if mad_val != 0 else value

        return value

    def _apply_encoding(self, value: Any, encode_config: Dict[str, Any]) -> Any:
        """Apply encoding transformation"""
        method = encode_config.get('method', 'label')

        if method == 'label':
            categories = encode_config.get('categories', [])
            if value in categories:
                return categories.index(value)
            else:
                return -1  # Unknown category

        elif method == 'onehot':
            categories = encode_config.get('categories', [])
            encoding = [0] * len(categories)
            if value in categories:
                encoding[categories.index(value)] = 1
            return encoding

        elif method == 'binary':
            # Convert categorical to binary encoding
            if isinstance(value, str):
                # Simple hash-based encoding
                hash_val = int(hashlib.md5(value.encode()).hexdigest(), 16)
                return hash_val % 2
            return 0

        return value

    def _apply_custom_transform(self, value: Any, custom_config: Dict[str, Any], record: Dict[str, Any]) -> Any:
        """Apply custom transformation"""
        transformer_name = custom_config.get('transformer')

        if transformer_name in self.custom_transformers:
            transformer_func = self.custom_transformers[transformer_name]
            return transformer_func(value, record)

        return value


class DataQualityAssessmentProcessor(BaseDataProcessor):
    """Comprehensive data quality assessment processor"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.quality_rules = self.config.get('quality_rules', {})
        self.is_initialized = True

    def process(self, data: Any) -> DataQualityMetrics:
        """Assess data quality"""
        start_time = time.time()

        metrics = DataQualityMetrics()

        if isinstance(data, dict):
            self._assess_record_quality(data, metrics)
        elif isinstance(data, list):
            for record in data:
                if isinstance(record, dict):
                    self._assess_record_quality(record, metrics)
        else:
            raise ValueError("Data must be a dictionary or list of dictionaries")

        # Calculate overall scores
        self._calculate_overall_scores(metrics)

        self.processing_stats['processing_time'] += time.time() - start_time
        self.processing_stats['records_processed'] = metrics.total_records

        return metrics

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate data for quality assessment"""
        errors = []

        if not isinstance(data, (dict, list)):
            errors.append("Data must be a dictionary or list of dictionaries")

        if isinstance(data, list) and not all(isinstance(item, dict) for item in data):
            errors.append("All items in data list must be dictionaries")

        return len(errors) == 0, errors

    def _assess_record_quality(self, record: Dict[str, Any], metrics: DataQualityMetrics):
        """Assess quality of a single record"""
        metrics.total_records += 1

        # Check for missing values
        for field_name, field_value in record.items():
            if field_value is None or (isinstance(field_value, str) and field_value.strip() == ""):
                metrics.missing_values[field_name] = metrics.missing_values.get(field_name, 0) + 1

        # Data type inference
        for field_name, field_value in record.items():
            if field_name not in metrics.data_types:
                metrics.data_types[field_name] = self._infer_data_type(field_value)

        # Range and format validation
        self._validate_ranges_and_formats(record, metrics)

        # Uniqueness check (simplified - would need full dataset)
        # Outlier detection (simplified)
        self._detect_outliers(record, metrics)

    def _infer_data_type(self, value: Any) -> str:
        """Infer data type from value"""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            # Check if it's a date string
            if self._is_date_string(value):
                return "date"
            return "string"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"

    def _is_date_string(self, value: str) -> bool:
        """Check if string represents a date"""
        try:
            # Try to parse as date
            datetime.fromisoformat(value.replace('Z', '+00:00'))
            return True
        except:
            return False

    def _validate_ranges_and_formats(self, record: Dict[str, Any], metrics: DataQualityMetrics):
        """Validate ranges and formats"""
        range_rules = self.quality_rules.get('range_rules', {})

        for field_name, field_value in record.items():
            if field_name in range_rules and field_value is not None:
                rule = range_rules[field_name]
                min_val = rule.get('min')
                max_val = rule.get('max')

                if isinstance(field_value, (int, float)):
                    if min_val is not None and field_value < min_val:
                        metrics.range_violations[field_name] = metrics.range_violations.get(field_name, 0) + 1
                    if max_val is not None and field_value > max_val:
                        metrics.range_violations[field_name] = metrics.range_violations.get(field_name, 0) + 1

            # Format validation for specific field types
            if 'email' in field_name.lower() and isinstance(field_value, str):
                if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', field_value):
                    metrics.format_violations[field_name] = metrics.format_violations.get(field_name, 0) + 1

            if 'phone' in field_name.lower() and isinstance(field_value, str):
                if not re.match(r'^\+?[\d\s\-\(\)]+$', field_value):
                    metrics.format_violations[field_name] = metrics.format_violations.get(field_name, 0) + 1

    def _detect_outliers(self, record: Dict[str, Any], metrics: DataQualityMetrics):
        """Detect outliers in record (simplified implementation)"""
        # This is a simplified outlier detection
        # In practice, would use statistical methods like IQR, Z-score, etc.
        outlier_rules = self.quality_rules.get('outlier_rules', {})

        for field_name, field_value in record.items():
            if field_name in outlier_rules and isinstance(field_value, (int, float)):
                rule = outlier_rules[field_name]
                threshold = rule.get('threshold', 3)  # Standard deviations
                mean_val = rule.get('mean', 0)
                std_val = rule.get('std', 1)

                z_score = abs(field_value - mean_val) / std_val if std_val > 0 else 0
                if z_score > threshold:
                    metrics.outlier_records += 1

    def _calculate_overall_scores(self, metrics: DataQualityMetrics):
        """Calculate overall quality scores"""
        if metrics.total_records == 0:
            return

        # Completeness score
        total_missing = sum(metrics.missing_values.values())
        total_expected_values = metrics.total_records * len(metrics.data_types)
        metrics.completeness_score = 1.0 - (total_missing / total_expected_values) if total_expected_values > 0 else 1.0

        # Accuracy score (simplified - based on format/range violations)
        total_violations = sum(metrics.range_violations.values()) + sum(metrics.format_violations.values())
        metrics.accuracy_score = 1.0 - (total_violations / (metrics.total_records * len(metrics.data_types)))

        # Consistency score (simplified - based on data type consistency)
        # For this implementation, assume high consistency
        metrics.consistency_score = 0.9

        # Timeliness score (simplified - based on assessment being recent)
        metrics.timeliness_score = 1.0

        # Validity score (based on range and format compliance)
        invalid_records = sum(metrics.range_violations.values()) + sum(metrics.format_violations.values())
        metrics.validity_score = 1.0 - (invalid_records / metrics.total_records) if metrics.total_records > 0 else 1.0

        # Uniqueness score (simplified - assume high uniqueness)
        metrics.uniqueness_score = 0.95

        # Overall quality score (weighted average)
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.15,
            'timeliness': 0.1,
            'validity': 0.15,
            'uniqueness': 0.1
        }

        metrics.overall_quality_score = (
            metrics.completeness_score * weights['completeness'] +
            metrics.accuracy_score * weights['accuracy'] +
            metrics.consistency_score * weights['consistency'] +
            metrics.timeliness_score * weights['timeliness'] +
            metrics.validity_score * weights['validity'] +
            metrics.uniqueness_score * weights['uniqueness']
        )

        metrics.valid_records = metrics.total_records - invalid_records
        metrics.invalid_records = invalid_records


class ETLProcessor(BaseDataProcessor):
    """Complete ETL (Extract, Transform, Load) processor"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.extract_processor = None
        self.transform_processor = DataTransformationProcessor(self.config.get('transform_config', {}))
        self.load_processor = None
        self.quality_processor = DataQualityAssessmentProcessor(self.config.get('quality_config', {}))
        self.jobs = {}
        self.is_initialized = True

    def process(self, data: Any) -> Dict[str, Any]:
        """Execute complete ETL pipeline"""
        job_id = f"etl_{int(time.time())}"
        job = ETLJob(
            job_id=job_id,
            source_type=self.config.get('source_type', 'unknown'),
            target_type=self.config.get('target_type', 'unknown'),
            status="running",
            start_time=datetime.now()
        )
        self.jobs[job_id] = job

        try:
            # Extract
            extracted_data = self._extract(data)
            job.records_processed = len(extracted_data) if isinstance(extracted_data, list) else 1

            # Transform
            transformed_data = self.transform_processor.process(extracted_data)

            # Quality Assessment
            quality_metrics = self.quality_processor.process(transformed_data)
            job.quality_metrics = quality_metrics

            # Load
            loaded_data = self._load(transformed_data)

            job.status = "completed"
            job.end_time = datetime.now()

            return {
                'job_id': job_id,
                'status': 'success',
                'records_processed': job.records_processed,
                'quality_metrics': quality_metrics,
                'data': loaded_data
            }

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now()
            raise e

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate ETL pipeline configuration"""
        errors = []

        # Validate transformation rules
        transform_valid, transform_errors = self.transform_processor.validate(data)
        if not transform_valid:
            errors.extend(transform_errors)

        # Validate quality rules
        quality_valid, quality_errors = self.quality_processor.validate(data)
        if not quality_valid:
            errors.extend(quality_errors)

        return len(errors) == 0, errors

    def _extract(self, data: Any) -> Any:
        """Extract data from source"""
        # In a real implementation, this would handle different data sources
        # (files, databases, APIs, etc.)
        return data

    def _load(self, data: Any) -> Any:
        """Load data to target destination"""
        # In a real implementation, this would handle different destinations
        # (databases, files, APIs, etc.)
        return data

    def get_job_status(self, job_id: str) -> Optional[ETLJob]:
        """Get ETL job status"""
        return self.jobs.get(job_id)


class HealthcareDataProcessingAlgorithms:
    """Collection of data processing algorithms for healthcare"""

    def __init__(self):
        self.processors = {
            'validation': DataValidationProcessor,
            'transformation': DataTransformationProcessor,
            'quality_assessment': DataQualityAssessmentProcessor,
            'etl': ETLProcessor
        }
        self.active_processors = {}

    def create_processor(self, processor_type: str, config: Dict[str, Any] = None) -> str:
        """Create a data processing pipeline"""

        if processor_type not in self.processors:
            raise ValueError(f"Unsupported processor type: {processor_type}")

        processor_id = f"{processor_type}_{int(time.time())}"
        processor = self.processors[processor_type](config)
        self.active_processors[processor_id] = processor

        return processor_id

    def process_data(self, processor_id: str, data: Any) -> Any:
        """Process data using specified processor"""
        if processor_id not in self.active_processors:
            raise ValueError(f"Processor {processor_id} not found")

        processor = self.active_processors[processor_id]
        return processor.process(data)

    def validate_data(self, processor_id: str, data: Any) -> Tuple[bool, List[str]]:
        """Validate data using specified processor"""
        if processor_id not in self.active_processors:
            raise ValueError(f"Processor {processor_id} not found")

        processor = self.active_processors[processor_id]
        return processor.validate(data)

    def get_processor_stats(self, processor_id: str) -> Dict[str, Any]:
        """Get processor statistics"""
        if processor_id not in self.active_processors:
            raise ValueError(f"Processor {processor_id} not found")

        processor = self.active_processors[processor_id]
        return processor.get_stats()

    def get_available_processors(self) -> List[str]:
        """Get list of available processor types"""
        return list(self.processors.keys())

    def create_healthcare_etl_pipeline(self, config: Dict[str, Any] = None) -> str:
        """Create a healthcare-specific ETL pipeline"""

        default_config = {
            'source_type': 'healthcare_records',
            'target_type': 'processed_healthcare_data',
            'transform_config': {
                'transformation_rules': {
                    'patient_age': {
                        'calculate': {
                            'operation': 'subtract',
                            'operands': ['current_year', 'birth_year']
                        }
                    },
                    'bmi_category': {
                        'custom': {
                            'transformer': 'bmi_categorizer'
                        }
                    }
                },
                'custom_transformers': {
                    'bmi_categorizer': self._categorize_bmi
                }
            },
            'quality_config': {
                'quality_rules': {
                    'range_rules': {
                        'age': {'min': 0, 'max': 150},
                        'weight_kg': {'min': 1, 'max': 500},
                        'height_cm': {'min': 30, 'max': 250},
                        'bmi': {'min': 10, 'max': 70}
                    },
                    'outlier_rules': {
                        'glucose_mg_dl': {'mean': 100, 'std': 30, 'threshold': 3},
                        'cholesterol_mg_dl': {'mean': 200, 'std': 50, 'threshold': 3}
                    }
                }
            }
        }

        if config:
            self._deep_merge(default_config, config)

        return self.create_processor('etl', default_config)

    def _categorize_bmi(self, bmi_value: float, record: Dict[str, Any]) -> str:
        """Categorize BMI value"""
        if not isinstance(bmi_value, (int, float)):
            return "unknown"

        if bmi_value < 18.5:
            return "underweight"
        elif bmi_value < 25:
            return "normal"
        elif bmi_value < 30:
            return "overweight"
        else:
            return "obese"

    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep merge two dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
