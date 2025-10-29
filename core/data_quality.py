"""
Data Quality Validation System for AI Personalized Medicine Platform
Comprehensive data quality assessment, validation, and improvement
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import statistics
import threading
import time
import hashlib
from pathlib import Path

class DataQualityValidator:
    """Advanced data quality validation system"""

    def __init__(self):
        self.quality_rules = {}
        self.validation_profiles = {}
        self.quality_metrics_history = defaultdict(list)
        self.data_quality_alerts = []
        self.is_running = False
        self.quality_workers = []
        self.initialize_quality_system()

    def initialize_quality_system(self):
        """Initialize data quality validation system"""
        # Define quality dimensions
        self.quality_dimensions = {
            "completeness": self._assess_completeness,
            "accuracy": self._assess_accuracy,
            "consistency": self._assess_consistency,
            "validity": self._assess_validity,
            "timeliness": self._assess_timeliness,
            "uniqueness": self._assess_uniqueness,
            "integrity": self._assess_integrity
        }

        # Initialize domain-specific validation rules
        self._initialize_domain_rules()

        print("🔍 Data Quality Validation System initialized")

    def _initialize_domain_rules(self):
        """Initialize domain-specific validation rules"""
        self.domain_rules = {
            "patient_demographics": {
                "age": {"type": "range", "min": 0, "max": 150},
                "gender": {"type": "categorical", "values": ["M", "F", "O", "U"]},
                "date_of_birth": {"type": "date", "not_future": True},
                "email": {"type": "regex", "pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
                "phone": {"type": "regex", "pattern": r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'}
            },
            "vital_signs": {
                "blood_pressure_systolic": {"type": "range", "min": 60, "max": 250},
                "blood_pressure_diastolic": {"type": "range", "min": 30, "max": 150},
                "heart_rate": {"type": "range", "min": 30, "max": 250},
                "temperature": {"type": "range", "min": 95, "max": 108},  # Fahrenheit
                "respiratory_rate": {"type": "range", "min": 8, "max": 60},
                "oxygen_saturation": {"type": "range", "min": 70, "max": 100}
            },
            "laboratory_results": {
                "glucose": {"type": "range", "min": 20, "max": 600},
                "hemoglobin_a1c": {"type": "range", "min": 3, "max": 20},
                "cholesterol_total": {"type": "range", "min": 50, "max": 400},
                "hdl_cholesterol": {"type": "range", "min": 10, "max": 150},
                "ldl_cholesterol": {"type": "range", "min": 20, "max": 300},
                "triglycerides": {"type": "range", "min": 20, "max": 1500}
            },
            "medications": {
                "dosage": {"type": "positive_number"},
                "frequency": {"type": "categorical", "values": ["daily", "twice_daily", "three_times_daily", "weekly", "as_needed"]},
                "route": {"type": "categorical", "values": ["oral", "intravenous", "subcutaneous", "topical", "inhaled"]},
                "start_date": {"type": "date", "not_future": True},
                "end_date": {"type": "date", "after_start_date": True}
            }
        }

    def start_quality_system(self):
        """Start data quality validation system"""
        self.is_running = True

        # Start quality workers
        for i in range(3):  # 3 concurrent quality workers
            worker = threading.Thread(target=self._quality_worker, daemon=True)
            worker.start()
            self.quality_workers.append(worker)

        # Start monitoring worker
        monitor_worker = threading.Thread(target=self._quality_monitor, daemon=True)
        monitor_worker.start()
        self.quality_workers.append(monitor_worker)

        print("⚡ Data Quality Validation System started")

    def stop_quality_system(self):
        """Stop data quality validation system"""
        self.is_running = False
        print("🛑 Data Quality Validation System stopped")

    def validate_dataset(self, data: pd.DataFrame, domain: str = "general",
                        validation_profile: str = "comprehensive") -> Dict[str, Any]:
        """Perform comprehensive data quality validation"""
        validation_start = datetime.now()

        # Select validation rules based on profile
        if validation_profile == "comprehensive":
            dimensions_to_check = list(self.quality_dimensions.keys())
        elif validation_profile == "basic":
            dimensions_to_check = ["completeness", "validity", "uniqueness"]
        else:
            dimensions_to_check = validation_profile.split(",") if isinstance(validation_profile, str) else [validation_profile]

        quality_report = {
            "dataset_info": {
                "rows": len(data),
                "columns": len(data.columns),
                "domain": domain,
                "validation_profile": validation_profile
            },
            "overall_quality_score": 0,
            "quality_dimensions": {},
            "column_quality": {},
            "anomalies_detected": [],
            "recommendations": [],
            "validation_timestamp": validation_start,
            "processing_time_seconds": 0
        }

        # Assess each quality dimension
        dimension_scores = {}
        for dimension in dimensions_to_check:
            if dimension in self.quality_dimensions:
                dimension_result = self.quality_dimensions[dimension](data, domain)
                quality_report["quality_dimensions"][dimension] = dimension_result
                dimension_scores[dimension] = dimension_result.get("score", 0)

        # Calculate overall quality score
        if dimension_scores:
            quality_report["overall_quality_score"] = sum(dimension_scores.values()) / len(dimension_scores)

        # Assess column-level quality
        quality_report["column_quality"] = self._assess_column_quality(data, domain)

        # Detect anomalies
        quality_report["anomalies_detected"] = self._detect_data_anomalies(data, domain)

        # Generate recommendations
        quality_report["recommendations"] = self._generate_quality_recommendations(quality_report)

        # Calculate processing time
        processing_time = (datetime.now() - validation_start).total_seconds()
        quality_report["processing_time_seconds"] = processing_time

        # Store quality metrics history
        self.quality_metrics_history[domain].append({
            "timestamp": datetime.now(),
            "quality_score": quality_report["overall_quality_score"],
            "dimensions": dimension_scores,
            "anomalies_count": len(quality_report["anomalies_detected"])
        })

        return quality_report

    def _assess_completeness(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Assess data completeness"""
        completeness_scores = {}

        for col in data.columns:
            missing_count = data[col].isnull().sum()
            completeness_score = 1 - (missing_count / len(data)) if len(data) > 0 else 0
            completeness_scores[col] = {
                "score": completeness_score,
                "missing_count": missing_count,
                "missing_percentage": (missing_count / len(data)) * 100 if len(data) > 0 else 0
            }

        overall_score = sum(s["score"] for s in completeness_scores.values()) / len(completeness_scores) if completeness_scores else 0

        # Identify critical missing data
        critical_missing = []
        for col, stats in completeness_scores.items():
            if stats["missing_percentage"] > 20:  # More than 20% missing
                critical_missing.append({
                    "column": col,
                    "missing_percentage": stats["missing_percentage"],
                    "severity": "high" if stats["missing_percentage"] > 50 else "medium"
                })

        return {
            "score": overall_score,
            "column_completeness": completeness_scores,
            "critical_missing_data": critical_missing,
            "assessment": self._interpret_quality_score(overall_score, "completeness")
        }

    def _assess_accuracy(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Assess data accuracy"""
        accuracy_scores = {}

        # Apply domain-specific validation rules
        domain_rules = self.domain_rules.get(domain, {})

        for col in data.columns:
            if col in domain_rules:
                rule = domain_rules[col]
                accuracy_score = self._validate_column_accuracy(data[col], rule)
                accuracy_scores[col] = accuracy_score
            else:
                # Default accuracy check - look for extreme outliers
                accuracy_score = self._check_outlier_accuracy(data[col])
                accuracy_scores[col] = {"score": accuracy_score, "rule": "outlier_check"}

        overall_score = sum(s["score"] for s in accuracy_scores.values()) / len(accuracy_scores) if accuracy_scores else 1.0

        return {
            "score": overall_score,
            "column_accuracy": accuracy_scores,
            "assessment": self._interpret_quality_score(overall_score, "accuracy")
        }

    def _validate_column_accuracy(self, series: pd.Series, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate accuracy of a column based on rules"""
        rule_type = rule["type"]

        if rule_type == "range":
            min_val, max_val = rule["min"], rule["max"]
            valid_count = ((series >= min_val) & (series <= max_val)).sum()
            accuracy_score = valid_count / len(series) if len(series) > 0 else 0

            return {
                "score": accuracy_score,
                "valid_count": valid_count,
                "invalid_count": len(series) - valid_count,
                "rule_type": "range",
                "range": [min_val, max_val]
            }

        elif rule_type == "categorical":
            valid_values = set(rule["values"])
            valid_count = series.astype(str).str.lower().isin([v.lower() for v in valid_values]).sum()
            accuracy_score = valid_count / len(series) if len(series) > 0 else 0

            return {
                "score": accuracy_score,
                "valid_count": valid_count,
                "invalid_count": len(series) - valid_count,
                "rule_type": "categorical",
                "valid_values": rule["values"]
            }

        elif rule_type == "regex":
            pattern = re.compile(rule["pattern"])
            valid_count = series.astype(str).apply(lambda x: bool(pattern.match(str(x)))).sum()
            accuracy_score = valid_count / len(series) if len(series) > 0 else 0

            return {
                "score": accuracy_score,
                "valid_count": valid_count,
                "invalid_count": len(series) - valid_count,
                "rule_type": "regex",
                "pattern": rule["pattern"]
            }

        elif rule_type == "positive_number":
            valid_count = (pd.to_numeric(series, errors='coerce') > 0).sum()
            accuracy_score = valid_count / len(series) if len(series) > 0 else 0

            return {
                "score": accuracy_score,
                "valid_count": valid_count,
                "invalid_count": len(series) - valid_count,
                "rule_type": "positive_number"
            }

        return {"score": 1.0, "rule_type": "not_validated"}

    def _check_outlier_accuracy(self, series: pd.Series) -> float:
        """Check for outliers as accuracy indicator"""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) < 4:
                return 1.0

            # Use IQR method to detect outliers
            Q1 = numeric_series.quantile(0.25)
            Q3 = numeric_series.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((numeric_series < lower_bound) | (numeric_series > upper_bound)).sum()
            outlier_percentage = outliers / len(numeric_series)

            # Accuracy decreases with more outliers (but not too harshly)
            accuracy_score = max(0.5, 1.0 - outlier_percentage * 2)

            return accuracy_score

        except:
            return 1.0  # Default to perfect accuracy if check fails

    def _assess_consistency(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Assess data consistency"""
        consistency_checks = {}

        # Check logical consistency rules
        if domain == "vital_signs":
            # Blood pressure consistency
            if "blood_pressure_systolic" in data.columns and "blood_pressure_diastolic" in data.columns:
                systolic = pd.to_numeric(data["blood_pressure_systolic"], errors='coerce')
                diastolic = pd.to_numeric(data["blood_pressure_diastolic"], errors='coerce')

                valid_bp = (systolic > diastolic) & (systolic > 0) & (diastolic > 0)
                consistency_score = valid_bp.sum() / len(valid_bp) if len(valid_bp) > 0 else 0

                consistency_checks["blood_pressure"] = {
                    "score": consistency_score,
                    "inconsistent_count": (~valid_bp).sum(),
                    "check": "systolic > diastolic"
                }

        # Date consistency
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        if len(date_columns) >= 2:
            for i, col1 in enumerate(date_columns):
                for col2 in date_columns[i+1:]:
                    try:
                        date1 = pd.to_datetime(data[col1], errors='coerce')
                        date2 = pd.to_datetime(data[col2], errors='coerce')

                        # Check if dates are in logical order
                        if col1.lower().startswith("start") and col2.lower().startswith("end"):
                            consistent = (date1 <= date2) | (date1.isnull() | date2.isnull())
                        else:
                            consistent = pd.Series([True] * len(data))  # No specific order expected

                        consistency_score = consistent.sum() / len(consistent)
                        consistency_checks[f"{col1}_vs_{col2}"] = {
                            "score": consistency_score,
                            "inconsistent_count": (~consistent).sum()
                        }
                    except:
                        consistency_checks[f"{col1}_vs_{col2}"] = {"score": 1.0, "error": "date_parsing_failed"}

        # Calculate overall consistency
        if consistency_checks:
            overall_score = sum(check["score"] for check in consistency_checks.values()) / len(consistency_checks)
        else:
            overall_score = 1.0

        return {
            "score": overall_score,
            "consistency_checks": consistency_checks,
            "assessment": self._interpret_quality_score(overall_score, "consistency")
        }

    def _assess_validity(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Assess data validity"""
        validity_scores = {}

        for col in data.columns:
            validity_score = self._validate_column_values(data[col])
            validity_scores[col] = validity_score

        overall_score = sum(s["score"] for s in validity_scores.values()) / len(validity_scores) if validity_scores else 1.0

        return {
            "score": overall_score,
            "column_validity": validity_scores,
            "assessment": self._interpret_quality_score(overall_score, "validity")
        }

    def _validate_column_values(self, series: pd.Series) -> Dict[str, Any]:
        """Validate individual column values"""
        # Check for obviously invalid values
        invalid_patterns = [
            r'^null$', r'^n/a$', r'^none$', r'^unknown$', r'^\s*$',
            r'^9999.*', r'^0000.*', r'^test.*', r'.*dummy.*'
        ]

        invalid_count = 0
        for pattern in invalid_patterns:
            matches = series.astype(str).str.contains(pattern, case=False, regex=True, na=False)
            invalid_count += matches.sum()

        # Check for non-printable characters
        non_printable = series.astype(str).apply(lambda x: any(ord(c) < 32 and c not in '\t\n\r' for c in x))
        invalid_count += non_printable.sum()

        validity_score = 1 - (invalid_count / len(series)) if len(series) > 0 else 0

        return {
            "score": validity_score,
            "invalid_count": invalid_count,
            "invalid_percentage": (invalid_count / len(series)) * 100 if len(series) > 0 else 0
        }

    def _assess_timeliness(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Assess data timeliness"""
        timeliness_scores = {}

        # Find timestamp/date columns
        time_columns = [col for col in data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp'])]

        for col in time_columns:
            try:
                parsed_dates = pd.to_datetime(data[col], errors='coerce')
                valid_dates = parsed_dates.dropna()

                if len(valid_dates) == 0:
                    timeliness_scores[col] = {"score": 0, "error": "no_valid_dates"}
                    continue

                # Check how recent the data is
                now = pd.Timestamp.now()
                age_days = (now - valid_dates).dt.total_seconds() / (24 * 3600)

                # Define timeliness thresholds based on domain
                if domain == "vital_signs":
                    max_age_days = 7  # Vital signs should be recent
                elif domain == "laboratory_results":
                    max_age_days = 365  # Lab results can be older
                else:
                    max_age_days = 90  # General data freshness

                timely_count = (age_days <= max_age_days).sum()
                timeliness_score = timely_count / len(valid_dates)

                timeliness_scores[col] = {
                    "score": timeliness_score,
                    "timely_count": timely_count,
                    "outdated_count": len(valid_dates) - timely_count,
                    "average_age_days": age_days.mean(),
                    "max_age_days": age_days.max(),
                    "threshold_days": max_age_days
                }

            except Exception as e:
                timeliness_scores[col] = {"score": 0.5, "error": str(e)}

        # Overall timeliness
        if timeliness_scores:
            overall_score = sum(ts["score"] for ts in timeliness_scores.values() if "score" in ts) / len(timeliness_scores)
        else:
            overall_score = 1.0  # No time columns = perfectly timely

        return {
            "score": overall_score,
            "column_timeliness": timeliness_scores,
            "assessment": self._interpret_quality_score(overall_score, "timeliness")
        }

    def _assess_uniqueness(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Assess data uniqueness"""
        uniqueness_scores = {}

        for col in data.columns:
            unique_count = data[col].nunique(dropna=True)
            total_count = len(data[col].dropna())

            if total_count > 0:
                uniqueness_score = unique_count / total_count
                duplicate_count = total_count - unique_count

                uniqueness_scores[col] = {
                    "score": uniqueness_score,
                    "unique_count": unique_count,
                    "duplicate_count": duplicate_count,
                    "duplicate_percentage": (duplicate_count / total_count) * 100
                }
            else:
                uniqueness_scores[col] = {"score": 1.0, "error": "no_data"}

        # Check composite uniqueness (key fields)
        key_candidates = self._identify_key_candidates(data, domain)
        if key_candidates:
            composite_duplicates = data.duplicated(subset=key_candidates).sum()
            composite_uniqueness = 1 - (composite_duplicates / len(data)) if len(data) > 0 else 0

            uniqueness_scores["composite_key"] = {
                "score": composite_uniqueness,
                "duplicate_count": composite_duplicates,
                "key_columns": key_candidates
            }

        overall_score = sum(us["score"] for us in uniqueness_scores.values()) / len(uniqueness_scores) if uniqueness_scores else 1.0

        return {
            "score": overall_score,
            "column_uniqueness": uniqueness_scores,
            "assessment": self._interpret_quality_score(overall_score, "uniqueness")
        }

    def _identify_key_candidates(self, data: pd.DataFrame, domain: str) -> List[str]:
        """Identify potential key columns for uniqueness checking"""
        key_patterns = {
            "general": ["id", "patient_id", "record_id", "unique_id"],
            "patient_demographics": ["patient_id", "medical_record_number", "social_security_number"],
            "vital_signs": ["patient_id", "timestamp", "encounter_id"],
            "laboratory_results": ["patient_id", "test_code", "timestamp", "accession_number"]
        }

        candidates = key_patterns.get(domain, key_patterns["general"])

        # Find matching columns
        existing_candidates = [col for col in candidates if col in data.columns]

        # If no standard keys found, use columns with high uniqueness
        if not existing_candidates:
            uniqueness_scores = {}
            for col in data.columns:
                if data[col].dtype in ['object', 'string']:
                    unique_ratio = data[col].nunique() / len(data[col].dropna())
                    if unique_ratio > 0.8:  # High uniqueness
                        uniqueness_scores[col] = unique_ratio

            if uniqueness_scores:
                existing_candidates = [max(uniqueness_scores, key=uniqueness_scores.get)]

        return existing_candidates

    def _assess_integrity(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Assess data integrity (referential integrity, constraints)"""
        integrity_checks = {}

        # Check referential integrity
        if "patient_id" in data.columns:
            # Check if patient_ids are consistent format
            patient_ids = data["patient_id"].dropna()
            if len(patient_ids) > 0:
                # Simple format check (should be numeric or specific pattern)
                valid_format = patient_ids.astype(str).str.match(r'^[A-Za-z0-9_-]+$')
                format_score = valid_format.sum() / len(valid_format)

                integrity_checks["patient_id_format"] = {
                    "score": format_score,
                    "valid_count": valid_format.sum(),
                    "invalid_count": len(valid_format) - valid_format.sum()
                }

        # Check for constraint violations
        if domain == "medications":
            if "start_date" in data.columns and "end_date" in data.columns:
                start_dates = pd.to_datetime(data["start_date"], errors='coerce')
                end_dates = pd.to_datetime(data["end_date"], errors='coerce')

                valid_order = (start_dates <= end_dates) | (start_dates.isnull() | end_dates.isnull())
                order_score = valid_order.sum() / len(valid_order) if len(valid_order) > 0 else 1

                integrity_checks["medication_date_order"] = {
                    "score": order_score,
                    "valid_count": valid_order.sum(),
                    "constraint": "start_date <= end_date"
                }

        # Overall integrity score
        if integrity_checks:
            overall_score = sum(check["score"] for check in integrity_checks.values()) / len(integrity_checks)
        else:
            overall_score = 1.0  # No specific integrity checks = assume good

        return {
            "score": overall_score,
            "integrity_checks": integrity_checks,
            "assessment": self._interpret_quality_score(overall_score, "integrity")
        }

    def _assess_column_quality(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Assess quality at column level"""
        column_quality = {}

        for col in data.columns:
            col_data = data[col]

            # Basic statistics
            stats = {
                "data_type": str(col_data.dtype),
                "total_count": len(col_data),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100 if len(col_data) > 0 else 0
            }

            # Type-specific statistics
            if col_data.dtype in ['int64', 'float64']:
                numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(numeric_data) > 0:
                    stats.update({
                        "mean": numeric_data.mean(),
                        "median": numeric_data.median(),
                        "std": numeric_data.std(),
                        "min": numeric_data.min(),
                        "max": numeric_data.max(),
                        "zeros_count": (numeric_data == 0).sum(),
                        "negative_count": (numeric_data < 0).sum()
                    })
            elif col_data.dtype == 'object':
                string_data = col_data.dropna().astype(str)
                if len(string_data) > 0:
                    stats.update({
                        "unique_count": string_data.nunique(),
                        "avg_length": string_data.str.len().mean(),
                        "empty_strings": (string_data == "").sum(),
                        "whitespace_only": string_data.str.match(r'^\s*$').sum()
                    })

            # Domain-specific quality checks
            domain_checks = self._apply_domain_quality_checks(col, col_data, domain)
            stats.update(domain_checks)

            # Overall column quality score
            quality_score = self._calculate_column_quality_score(stats, domain)
            stats["quality_score"] = quality_score

            column_quality[col] = stats

        return column_quality

    def _apply_domain_quality_checks(self, column: str, data: pd.Series, domain: str) -> Dict[str, Any]:
        """Apply domain-specific quality checks"""
        checks = {}

        if domain == "patient_demographics":
            if column == "age":
                numeric_age = pd.to_numeric(data, errors='coerce')
                checks["implausible_ages"] = ((numeric_age < 0) | (numeric_age > 150)).sum()
            elif column == "email":
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                checks["invalid_emails"] = (~data.astype(str).str.match(email_pattern, na=False)).sum()

        elif domain == "vital_signs":
            if "blood_pressure" in column:
                numeric_bp = pd.to_numeric(data, errors='coerce')
                checks["implausible_bp"] = ((numeric_bp < 50) | (numeric_bp > 300)).sum()

        return checks

    def _calculate_column_quality_score(self, stats: Dict[str, Any], domain: str) -> float:
        """Calculate overall quality score for a column"""
        score = 1.0

        # Penalize for nulls
        null_penalty = stats.get("null_percentage", 0) / 100 * 0.5
        score -= null_penalty

        # Penalize for invalid values
        invalid_penalty = 0
        for key, value in stats.items():
            if "invalid" in key.lower() or "implausible" in key.lower():
                invalid_penalty += value * 0.1  # Assume each invalid value costs 0.1 score

        score -= min(0.3, invalid_penalty)  # Cap invalid penalty

        # Bonus for good statistical properties
        if "std" in stats and stats["std"] > 0:
            # Reasonable variance (not too low, not too high)
            cv = stats["std"] / abs(stats["mean"]) if stats["mean"] != 0 else 0
            if 0.1 <= cv <= 2.0:
                score += 0.05

        return max(0, min(1, score))

    def _detect_data_anomalies(self, data: pd.DataFrame, domain: str) -> List[Dict[str, Any]]:
        """Detect data anomalies"""
        anomalies = []

        # Statistical anomalies
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            col_data = data[col].dropna()
            if len(col_data) < 10:
                continue

            # Z-score based anomaly detection
            mean_val = col_data.mean()
            std_val = col_data.std()

            if std_val > 0:
                z_scores = abs((col_data - mean_val) / std_val)
                outlier_count = (z_scores > 3).sum()  # 3 sigma rule

                if outlier_count > 0:
                    anomalies.append({
                        "type": "statistical_outlier",
                        "column": col,
                        "severity": "high" if outlier_count > len(col_data) * 0.05 else "medium",
                        "description": f"Found {outlier_count} statistical outliers in {col}",
                        "affected_rows": outlier_count
                    })

        # Pattern-based anomalies
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for unusual patterns
                value_counts = data[col].value_counts()
                most_common_pct = value_counts.iloc[0] / len(data[col]) if len(value_counts) > 0 else 0

                if most_common_pct > 0.8:  # One value makes up >80% of data
                    anomalies.append({
                        "type": "low_diversity",
                        "column": col,
                        "severity": "medium",
                        "description": f"Low diversity in {col}: {value_counts.index[0]} appears in {most_common_pct:.1%} of rows",
                        "affected_rows": int(most_common_pct * len(data[col]))
                    })

        # Domain-specific anomaly detection
        domain_anomalies = self._detect_domain_anomalies(data, domain)
        anomalies.extend(domain_anomalies)

        return anomalies

    def _detect_domain_anomalies(self, data: pd.DataFrame, domain: str) -> List[Dict[str, Any]]:
        """Detect domain-specific anomalies"""
        anomalies = []

        if domain == "vital_signs":
            # Check for impossible vital sign combinations
            if "blood_pressure_systolic" in data.columns and "blood_pressure_diastolic" in data.columns:
                systolic = pd.to_numeric(data["blood_pressure_systolic"], errors='coerce')
                diastolic = pd.to_numeric(data["blood_pressure_diastolic"], errors='coerce')

                # Pulse pressure check (systolic - diastolic should be reasonable)
                pulse_pressure = systolic - diastolic
                abnormal_pp = ((pulse_pressure < 20) | (pulse_pressure > 100)).sum()

                if abnormal_pp > 0:
                    anomalies.append({
                        "type": "abnormal_pulse_pressure",
                        "columns": ["blood_pressure_systolic", "blood_pressure_diastolic"],
                        "severity": "medium",
                        "description": f"Abnormal pulse pressure in {abnormal_pp} records",
                        "affected_rows": abnormal_pp
                    })

        return anomalies

    def _generate_quality_recommendations(self, quality_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate quality improvement recommendations"""
        recommendations = []

        overall_score = quality_report["overall_quality_score"]

        # Overall quality recommendations
        if overall_score < 0.7:
            recommendations.append({
                "priority": "critical",
                "category": "overall_quality",
                "recommendation": "Comprehensive data quality improvement program required",
                "actions": ["Implement data validation rules", "Set up automated quality monitoring", "Staff training on data entry"]
            })

        # Dimension-specific recommendations
        dimensions = quality_report["quality_dimensions"]

        for dimension, result in dimensions.items():
            score = result.get("score", 1.0)

            if dimension == "completeness" and score < 0.8:
                critical_missing = result.get("critical_missing_data", [])
                if critical_missing:
                    recommendations.append({
                        "priority": "high",
                        "category": "completeness",
                        "recommendation": f"Address missing data in critical columns: {', '.join([cm['column'] for cm in critical_missing])}",
                        "actions": ["Implement data collection validation", "Set up automated missing data alerts"]
                    })

            elif dimension == "accuracy" and score < 0.85:
                recommendations.append({
                    "priority": "high",
                    "category": "accuracy",
                    "recommendation": "Implement automated data accuracy validation",
                    "actions": ["Define validation rules", "Set up automated validation checks", "Implement data correction workflows"]
                })

            elif dimension == "consistency" and score < 0.9:
                recommendations.append({
                    "priority": "medium",
                    "category": "consistency",
                    "recommendation": "Improve data consistency through standardization",
                    "actions": ["Define data standards", "Implement consistency checks", "Create data normalization procedures"]
                })

        # Column-specific recommendations
        column_quality = quality_report["column_quality"]
        for col, stats in column_quality.items():
            quality_score = stats.get("quality_score", 1.0)

            if quality_score < 0.6:
                recommendations.append({
                    "priority": "high",
                    "category": "column_quality",
                    "column": col,
                    "recommendation": f"Critical quality issues in column '{col}' require immediate attention",
                    "actions": ["Review data collection process", "Implement validation rules", "Consider data cleansing"]
                })

        return recommendations

    def _interpret_quality_score(self, score: float, dimension: str) -> str:
        """Interpret quality score into human-readable assessment"""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.75:
            return "acceptable"
        elif score >= 0.65:
            return "needs_improvement"
        else:
            return "critical_attention_required"

    def create_quality_profile(self, domain: str, rules: Dict[str, Any]) -> str:
        """Create a custom quality validation profile"""
        profile_id = f"profile_{domain}_{int(time.time())}"

        self.validation_profiles[profile_id] = {
            "domain": domain,
            "rules": rules,
            "created_at": datetime.now(),
            "version": "1.0"
        }

        print(f"📋 Created quality validation profile: {profile_id}")
        return profile_id

    def get_quality_trends(self, domain: str = None, days: int = 30) -> Dict[str, Any]:
        """Get quality trends over time"""
        if domain:
            history = self.quality_metrics_history.get(domain, [])
        else:
            # Aggregate all domains
            all_history = []
            for domain_history in self.quality_metrics_history.values():
                all_history.extend(domain_history)
            history = sorted(all_history, key=lambda x: x["timestamp"])

        # Filter by time range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [h for h in history if h["timestamp"] > cutoff_date]

        if not recent_history:
            return {"error": "No quality history available"}

        # Calculate trends
        scores = [h["quality_score"] for h in recent_history]
        trend = "improving" if len(scores) >= 2 and scores[-1] > scores[0] else "declining" if len(scores) >= 2 and scores[-1] < scores[0] else "stable"

        return {
            "period_days": days,
            "data_points": len(recent_history),
            "average_quality": sum(scores) / len(scores) if scores else 0,
            "quality_trend": trend,
            "best_score": max(scores) if scores else 0,
            "worst_score": min(scores) if scores else 0,
            "volatility": statistics.stdev(scores) if len(scores) > 1 else 0
        }

    def _quality_worker(self):
        """Background quality validation worker"""
        while self.is_running:
            try:
                # Process queued quality validation tasks (simplified)
                time.sleep(0.1)
            except Exception as e:
                print(f"Quality worker error: {e}")

    def _quality_monitor(self):
        """Background quality monitoring"""
        while self.is_running:
            try:
                # Monitor quality trends and generate alerts
                for domain, history in self.quality_metrics_history.items():
                    if len(history) >= 5:
                        recent_scores = [h["quality_score"] for h in history[-5:]]

                        # Check for quality degradation
                        if recent_scores[-1] < 0.7 and recent_scores[-1] < min(recent_scores[:-1]):
                            alert = {
                                "type": "quality_degradation",
                                "domain": domain,
                                "severity": "high",
                                "message": f"Data quality degrading in {domain}: {recent_scores[-1]:.2f}",
                                "timestamp": datetime.now()
                            }
                            self.data_quality_alerts.append(alert)
                            print(f"🚨 QUALITY ALERT: {alert['message']}")

                time.sleep(3600)  # Check every hour

            except Exception as e:
                print(f"Quality monitor error: {e}")

    def export_quality_report(self, quality_report: Dict[str, Any], format: str = "json") -> str:
        """Export quality report"""
        if format == "json":
            return json.dumps(quality_report, indent=2, default=str)
        else:
            # Simple text format
            report_lines = [
                f"Data Quality Report - {quality_report['dataset_info']['domain']}",
                f"Generated: {quality_report['validation_timestamp']}",
                f"Overall Quality Score: {quality_report['overall_quality_score']:.3f}",
                "",
                "Quality Dimensions:"
            ]

            for dimension, result in quality_report["quality_dimensions"].items():
                report_lines.append(f"  {dimension}: {result['score']:.3f} ({result.get('assessment', 'unknown')})")

            report_lines.extend([
                "",
                f"Anomalies Detected: {len(quality_report['anomalies_detected'])}",
                "",
                "Recommendations:"
            ])

            for rec in quality_report["recommendations"][:5]:  # Top 5 recommendations
                report_lines.append(f"  • {rec['recommendation']}")

            return "\n".join(report_lines)
