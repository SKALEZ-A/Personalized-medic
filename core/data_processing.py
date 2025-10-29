"""
Data Processing Pipelines and ETL Processes for Healthcare Data
Advanced data ingestion, transformation, and processing workflows
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable, Iterator
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import csv
import uuid
from io import StringIO, BytesIO
import re

class DataPipeline:
    """Generic data processing pipeline"""

    def __init__(self, pipeline_id: str, name: str):
        self.pipeline_id = pipeline_id
        self.name = name
        self.stages = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "status": "inactive"
        }

    def add_stage(self, stage: 'PipelineStage') -> None:
        """Add processing stage to pipeline"""
        self.stages.append(stage)

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute pipeline with input data"""
        self.metadata["status"] = "running"
        self.metadata["started_at"] = datetime.now().isoformat()

        current_data = input_data
        execution_log = []
        stage_results = {}

        try:
            for i, stage in enumerate(self.stages):
                stage_start = datetime.now()

                # Execute stage
                result = await stage.execute(current_data)

                stage_end = datetime.now()
                execution_time = (stage_end - stage_start).total_seconds()

                # Log stage execution
                stage_log = {
                    "stage_index": i,
                    "stage_name": stage.name,
                    "execution_time_seconds": execution_time,
                    "status": "completed",
                    "input_records": self._count_records(current_data),
                    "output_records": self._count_records(result["data"])
                }
                execution_log.append(stage_log)

                # Store result for next stage
                current_data = result["data"]
                stage_results[stage.name] = result

                # Check for errors
                if result.get("errors"):
                    stage_log["status"] = "completed_with_errors"
                    stage_log["error_count"] = len(result["errors"])

            self.metadata["status"] = "completed"
            self.metadata["completed_at"] = datetime.now().isoformat()

            return {
                "pipeline_id": self.pipeline_id,
                "status": "success",
                "final_data": current_data,
                "stage_results": stage_results,
                "execution_log": execution_log,
                "metadata": self.metadata
            }

        except Exception as e:
            self.metadata["status"] = "failed"
            self.metadata["error"] = str(e)
            self.metadata["failed_at"] = datetime.now().isoformat()

            return {
                "pipeline_id": self.pipeline_id,
                "status": "failed",
                "error": str(e),
                "execution_log": execution_log,
                "metadata": self.metadata
            }

    def _count_records(self, data: Any) -> int:
        """Count records in data"""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return 1
        elif hasattr(data, '__len__'):
            return len(data)
        else:
            return 1

class PipelineStage:
    """Base class for pipeline stages"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute stage processing"""
        raise NotImplementedError("Subclasses must implement execute method")

class DataIngestionStage(PipelineStage):
    """Data ingestion stage for various sources"""

    def __init__(self, name: str, source_config: Dict[str, Any]):
        super().__init__(name, source_config)

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """Ingest data from configured source"""
        source_type = self.config.get("source_type")

        if source_type == "api":
            data = await self._ingest_from_api()
        elif source_type == "database":
            data = await self._ingest_from_database()
        elif source_type == "file":
            data = await self._ingest_from_file()
        elif source_type == "stream":
            data = await self._ingest_from_stream()
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        return {
            "data": data,
            "metadata": {
                "source_type": source_type,
                "ingested_at": datetime.now().isoformat(),
                "record_count": len(data) if isinstance(data, list) else 1
            },
            "errors": []
        }

    async def _ingest_from_api(self) -> List[Dict[str, Any]]:
        """Ingest data from API endpoint"""
        # Simulate API call
        await asyncio.sleep(0.1)
        return [
            {"patient_id": "P001", "data": "api_data_1"},
            {"patient_id": "P002", "data": "api_data_2"}
        ]

    async def _ingest_from_database(self) -> List[Dict[str, Any]]:
        """Ingest data from database"""
        # Simulate database query
        await asyncio.sleep(0.2)
        return [
            {"patient_id": "P001", "vitals": {"bp": "120/80", "hr": 72}},
            {"patient_id": "P002", "vitals": {"bp": "130/85", "hr": 75}}
        ]

    async def _ingest_from_file(self) -> List[Dict[str, Any]]:
        """Ingest data from file"""
        # Simulate file reading
        await asyncio.sleep(0.05)
        return [
            {"patient_id": "P001", "file_data": "content_1"},
            {"patient_id": "P002", "file_data": "content_2"}
        ]

    async def _ingest_from_stream(self) -> List[Dict[str, Any]]:
        """Ingest data from streaming source"""
        # Simulate streaming data
        data = []
        for i in range(10):
            data.append({"stream_id": i, "value": f"stream_data_{i}"})
            await asyncio.sleep(0.01)  # Simulate streaming delay
        return data

class DataTransformationStage(PipelineStage):
    """Data transformation and cleansing stage"""

    def __init__(self, name: str, transformation_config: Dict[str, Any]):
        super().__init__(name, transformation_config)

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """Transform input data"""
        transformations = self.config.get("transformations", [])
        transformed_data = input_data
        errors = []

        for transformation in transformations:
            try:
                transformed_data = await self._apply_transformation(
                    transformed_data, transformation
                )
            except Exception as e:
                errors.append({
                    "transformation": transformation,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # Data quality checks
        quality_report = self._perform_quality_checks(transformed_data)

        return {
            "data": transformed_data,
            "metadata": {
                "transformations_applied": len(transformations),
                "quality_score": quality_report["overall_score"],
                "transformed_at": datetime.now().isoformat()
            },
            "quality_report": quality_report,
            "errors": errors
        }

    async def _apply_transformation(self, data: Any, transformation: Dict[str, Any]) -> Any:
        """Apply specific transformation"""
        transform_type = transformation.get("type")

        if transform_type == "map_fields":
            return self._map_fields(data, transformation)
        elif transform_type == "filter_records":
            return self._filter_records(data, transformation)
        elif transform_type == "normalize_data":
            return self._normalize_data(data, transformation)
        elif transform_type == "aggregate_data":
            return self._aggregate_data(data, transformation)
        elif transform_type == "enrich_data":
            return await self._enrich_data(data, transformation)
        else:
            raise ValueError(f"Unsupported transformation type: {transform_type}")

    def _map_fields(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map fields from one format to another"""
        field_mapping = config.get("field_mapping", {})

        transformed = []
        for record in data:
            new_record = {}
            for new_field, old_field in field_mapping.items():
                if old_field in record:
                    new_record[new_field] = record[old_field]
                else:
                    new_record[new_field] = None
            transformed.append(new_record)

        return transformed

    def _filter_records(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter records based on criteria"""
        filter_condition = config.get("condition", "")

        filtered = []
        for record in data:
            if self._evaluate_condition(record, filter_condition):
                filtered.append(record)

        return filtered

    def _normalize_data(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize data values"""
        normalization_rules = config.get("rules", {})

        normalized = []
        for record in data:
            new_record = record.copy()
            for field, rule in normalization_rules.items():
                if field in new_record:
                    new_record[field] = self._apply_normalization_rule(new_record[field], rule)
            normalized.append(new_record)

        return normalized

    def _aggregate_data(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data by specified dimensions"""
        group_by = config.get("group_by", [])
        aggregations = config.get("aggregations", {})

        aggregated = defaultdict(dict)

        for record in data:
            # Create group key
            group_key = tuple(record.get(field) for field in group_by)

            if group_key not in aggregated:
                aggregated[group_key] = {field: record.get(field) for field in group_by}

            # Apply aggregations
            for agg_field, agg_func in aggregations.items():
                if agg_field in record:
                    value = record[agg_field]
                    if agg_func == "sum":
                        aggregated[group_key][agg_field] = aggregated[group_key].get(agg_field, 0) + value
                    elif agg_func == "count":
                        aggregated[group_key][agg_field] = aggregated[group_key].get(agg_field, 0) + 1
                    elif agg_func == "avg":
                        if f"{agg_field}_count" not in aggregated[group_key]:
                            aggregated[group_key][f"{agg_field}_count"] = 0
                            aggregated[group_key][f"{agg_field}_sum"] = 0
                        aggregated[group_key][f"{agg_field}_count"] += 1
                        aggregated[group_key][f"{agg_field}_sum"] += value
                        aggregated[group_key][agg_field] = (
                            aggregated[group_key][f"{agg_field}_sum"] /
                            aggregated[group_key][f"{agg_field}_count"]
                        )

        return dict(aggregated)

    async def _enrich_data(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enrich data with external sources"""
        enrichment_source = config.get("source", "")

        if enrichment_source == "geocoding":
            return await self._enrich_with_geocoding(data)
        elif enrichment_source == "demographics":
            return await self._enrich_with_demographics(data)
        elif enrichment_source == "medical_codes":
            return await self._enrich_with_medical_codes(data)

        return data

    async def _enrich_with_geocoding(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich with geocoding data"""
        # Simulate geocoding API call
        await asyncio.sleep(0.1)
        for record in data:
            if "address" in record:
                record["coordinates"] = {"lat": 40.7128, "lng": -74.0060}  # NYC coords
        return data

    async def _enrich_with_demographics(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich with demographic data"""
        await asyncio.sleep(0.1)
        for record in data:
            if "zip_code" in record:
                record["demographics"] = {
                    "median_income": 75000,
                    "population_density": 15000,
                    "health_index": 78.5
                }
        return data

    async def _enrich_with_medical_codes(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich with medical coding"""
        await asyncio.sleep(0.1)
        for record in data:
            if "diagnosis_text" in record:
                record["icd_codes"] = ["E11.9", "I10"]
                record["snomed_codes"] = ["44054006", "38341003"]
        return data

    def _evaluate_condition(self, record: Dict[str, Any], condition: str) -> bool:
        """Evaluate filter condition"""
        # Simple condition evaluation (would use proper expression parser)
        if "age > 18" in condition:
            return record.get("age", 0) > 18
        elif "status == 'active'" in condition:
            return record.get("status") == "active"
        return True

    def _apply_normalization_rule(self, value: Any, rule: str) -> Any:
        """Apply normalization rule to value"""
        if rule == "lowercase":
            return str(value).lower() if value else value
        elif rule == "uppercase":
            return str(value).upper() if value else value
        elif rule == "strip":
            return str(value).strip() if value else value
        elif rule == "numeric":
            try:
                return float(value) if value else None
            except:
                return None
        return value

    def _perform_quality_checks(self, data: Any) -> Dict[str, Any]:
        """Perform data quality checks"""
        if not isinstance(data, list):
            return {"overall_score": 100, "checks": []}

        total_records = len(data)
        checks = []

        # Completeness check
        complete_records = sum(1 for record in data if all(v is not None for v in record.values()))
        completeness_score = (complete_records / total_records) * 100 if total_records > 0 else 100
        checks.append({
            "check": "completeness",
            "score": completeness_score,
            "details": f"{complete_records}/{total_records} records complete"
        })

        # Uniqueness check
        unique_records = len(set(str(record) for record in data))
        uniqueness_score = (unique_records / total_records) * 100 if total_records > 0 else 100
        checks.append({
            "check": "uniqueness",
            "score": uniqueness_score,
            "details": f"{unique_records}/{total_records} records unique"
        })

        # Validity check (basic)
        valid_records = sum(1 for record in data if self._is_record_valid(record))
        validity_score = (valid_records / total_records) * 100 if total_records > 0 else 100
        checks.append({
            "check": "validity",
            "score": validity_score,
            "details": f"{valid_records}/{total_records} records valid"
        })

        # Overall score (weighted average)
        weights = {"completeness": 0.4, "uniqueness": 0.3, "validity": 0.3}
        overall_score = sum(check["score"] * weights[check["check"]] for check in checks)

        return {
            "overall_score": round(overall_score, 1),
            "checks": checks,
            "total_records": total_records
        }

    def _is_record_valid(self, record: Dict[str, Any]) -> bool:
        """Check if record is valid"""
        # Basic validation - check for required fields and data types
        required_fields = ["patient_id"]  # Example

        for field in required_fields:
            if field not in record or not record[field]:
                return False

        return True

class DataValidationStage(PipelineStage):
    """Data validation and quality assurance stage"""

    def __init__(self, name: str, validation_config: Dict[str, Any]):
        super().__init__(name, validation_config)

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """Validate input data"""
        validation_rules = self.config.get("rules", [])
        validation_results = []

        for rule in validation_rules:
            rule_result = await self._validate_rule(input_data, rule)
            validation_results.append(rule_result)

        # Overall validation status
        passed_rules = sum(1 for result in validation_results if result["passed"])
        total_rules = len(validation_results)

        validation_status = "passed" if passed_rules == total_rules else "failed"

        return {
            "data": input_data,  # Data passes through unchanged
            "validation_status": validation_status,
            "passed_rules": passed_rules,
            "total_rules": total_rules,
            "validation_results": validation_results,
            "errors": [r for r in validation_results if not r["passed"]],
            "metadata": {
                "validated_at": datetime.now().isoformat(),
                "validation_score": (passed_rules / total_rules) * 100 if total_rules > 0 else 100
            }
        }

    async def _validate_rule(self, data: Any, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against specific rule"""
        rule_type = rule.get("type")
        field = rule.get("field")
        condition = rule.get("condition")

        try:
            if rule_type == "required_field":
                passed = self._validate_required_field(data, field)
            elif rule_type == "data_type":
                passed = self._validate_data_type(data, field, condition)
            elif rule_type == "range_check":
                passed = self._validate_range(data, field, condition)
            elif rule_type == "format_check":
                passed = self._validate_format(data, field, condition)
            elif rule_type == "cross_field":
                passed = self._validate_cross_field(data, rule)
            else:
                passed = False

            return {
                "rule": rule,
                "passed": passed,
                "field": field,
                "error_message": None if passed else f"Validation failed for {rule_type} on field {field}"
            }

        except Exception as e:
            return {
                "rule": rule,
                "passed": False,
                "field": field,
                "error_message": f"Validation error: {str(e)}"
            }

    def _validate_required_field(self, data: List[Dict[str, Any]], field: str) -> bool:
        """Validate required field presence"""
        return all(field in record and record[field] is not None for record in data)

    def _validate_data_type(self, data: List[Dict[str, Any]], field: str, expected_type: str) -> bool:
        """Validate data type"""
        type_checks = {
            "string": lambda x: isinstance(x, str),
            "integer": lambda x: isinstance(x, int),
            "float": lambda x: isinstance(x, (int, float)),
            "boolean": lambda x: isinstance(x, bool)
        }

        if expected_type not in type_checks:
            return False

        check_func = type_checks[expected_type]
        return all(check_func(record.get(field)) for record in data if field in record)

    def _validate_range(self, data: List[Dict[str, Any]], field: str, range_config: Dict[str, Any]) -> bool:
        """Validate numeric range"""
        min_val = range_config.get("min")
        max_val = range_config.get("max")

        for record in data:
            if field in record:
                value = record[field]
                if not isinstance(value, (int, float)):
                    continue

                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False

        return True

    def _validate_format(self, data: List[Dict[str, Any]], field: str, format_pattern: str) -> bool:
        """Validate data format using regex"""
        try:
            pattern = re.compile(format_pattern)
            return all(
                pattern.match(str(record.get(field, "")))
                for record in data
                if field in record
            )
        except re.error:
            return False

    def _validate_cross_field(self, data: List[Dict[str, Any]], rule: Dict[str, Any]) -> bool:
        """Validate relationships between fields"""
        field1 = rule.get("field1")
        field2 = rule.get("field2")
        relationship = rule.get("relationship")

        for record in data:
            val1 = record.get(field1)
            val2 = record.get(field2)

            if relationship == "field1_greater_than_field2":
                if not (isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and val1 > val2):
                    return False
            elif relationship == "field1_equals_field2":
                if val1 != val2:
                    return False

        return True

class DataStorageStage(PipelineStage):
    """Data storage and persistence stage"""

    def __init__(self, name: str, storage_config: Dict[str, Any]):
        super().__init__(name, storage_config)

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """Store processed data"""
        storage_type = self.config.get("storage_type", "database")
        storage_location = self.config.get("location", "default")

        if storage_type == "database":
            result = await self._store_to_database(input_data, storage_location)
        elif storage_type == "data_warehouse":
            result = await self._store_to_data_warehouse(input_data, storage_location)
        elif storage_type == "file_system":
            result = await self._store_to_file_system(input_data, storage_location)
        elif storage_type == "object_storage":
            result = await self._store_to_object_storage(input_data, storage_location)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        return result

    async def _store_to_database(self, data: Any, location: str) -> Dict[str, Any]:
        """Store data to database"""
        # Simulate database storage
        await asyncio.sleep(0.2)

        stored_records = len(data) if isinstance(data, list) else 1

        return {
            "data": data,  # Pass through
            "storage_result": {
                "storage_type": "database",
                "location": location,
                "records_stored": stored_records,
                "storage_id": f"db_{int(datetime.now().timestamp())}",
                "stored_at": datetime.now().isoformat()
            },
            "errors": []
        }

    async def _store_to_data_warehouse(self, data: Any, location: str) -> Dict[str, Any]:
        """Store data to data warehouse"""
        await asyncio.sleep(0.3)

        # Simulate data partitioning
        partitions = self._create_partitions(data)

        return {
            "data": data,
            "storage_result": {
                "storage_type": "data_warehouse",
                "location": location,
                "partitions_created": len(partitions),
                "total_records": sum(len(p["data"]) for p in partitions),
                "stored_at": datetime.now().isoformat()
            },
            "partitions": partitions,
            "errors": []
        }

    async def _store_to_file_system(self, data: Any, location: str) -> Dict[str, Any]:
        """Store data to file system"""
        await asyncio.sleep(0.1)

        file_path = f"{location}/data_{int(datetime.now().timestamp())}.json"

        return {
            "data": data,
            "storage_result": {
                "storage_type": "file_system",
                "file_path": file_path,
                "file_size_bytes": len(json.dumps(data)),
                "stored_at": datetime.now().isoformat()
            },
            "errors": []
        }

    async def _store_to_object_storage(self, data: Any, location: str) -> Dict[str, Any]:
        """Store data to object storage"""
        await asyncio.sleep(0.15)

        object_key = f"data_{int(datetime.now().timestamp())}.json"

        return {
            "data": data,
            "storage_result": {
                "storage_type": "object_storage",
                "bucket": location,
                "object_key": object_key,
                "stored_at": datetime.now().isoformat()
            },
            "errors": []
        }

    def _create_partitions(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create data partitions for warehouse storage"""
        partitions = defaultdict(list)

        # Partition by date (example)
        for record in data:
            date_key = record.get("date", datetime.now().date().isoformat())[:7]  # YYYY-MM
            partitions[date_key].append(record)

        return [
            {"partition_key": key, "data": records, "record_count": len(records)}
            for key, records in partitions.items()
        ]

class ETLOrchestrator:
    """ETL process orchestrator for healthcare data"""

    def __init__(self):
        self.pipelines = {}
        self.etl_jobs = {}
        self.schedules = {}

    def create_etl_pipeline(self, pipeline_config: Dict[str, Any]) -> str:
        """Create ETL pipeline"""
        pipeline_id = f"etl_{int(datetime.now().timestamp())}"

        pipeline = DataPipeline(pipeline_id, pipeline_config.get("name", f"ETL Pipeline {pipeline_id}"))

        # Add stages based on configuration
        stages_config = pipeline_config.get("stages", [])

        for stage_config in stages_config:
            stage_type = stage_config.get("type")

            if stage_type == "ingestion":
                stage = DataIngestionStage(stage_config["name"], stage_config["config"])
            elif stage_type == "transformation":
                stage = DataTransformationStage(stage_config["name"], stage_config["config"])
            elif stage_type == "validation":
                stage = DataValidationStage(stage_config["name"], stage_config["config"])
            elif stage_type == "storage":
                stage = DataStorageStage(stage_config["name"], stage_config["config"])
            else:
                continue

            pipeline.add_stage(stage)

        self.pipelines[pipeline_id] = pipeline

        return pipeline_id

    async def execute_etl_job(self, pipeline_id: str, input_data: Any = None) -> Dict[str, Any]:
        """Execute ETL job"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        pipeline = self.pipelines[pipeline_id]

        job_id = f"job_{int(datetime.now().timestamp())}"

        job_record = {
            "job_id": job_id,
            "pipeline_id": pipeline_id,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "input_data": input_data
        }

        self.etl_jobs[job_id] = job_record

        try:
            result = await pipeline.execute(input_data)

            job_record["status"] = "completed" if result["status"] == "success" else "failed"
            job_record["completed_at"] = datetime.now().isoformat()
            job_record["result"] = result

            return {
                "job_id": job_id,
                "status": job_record["status"],
                "pipeline_result": result
            }

        except Exception as e:
            job_record["status"] = "failed"
            job_record["error"] = str(e)
            job_record["failed_at"] = datetime.now().isoformat()

            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }

    def schedule_etl_job(self, pipeline_id: str, schedule_config: Dict[str, Any]) -> str:
        """Schedule recurring ETL job"""
        schedule_id = f"schedule_{int(datetime.now().timestamp())}"

        schedule = {
            "schedule_id": schedule_id,
            "pipeline_id": pipeline_id,
            "frequency": schedule_config.get("frequency", "daily"),
            "next_run": self._calculate_next_run(schedule_config),
            "active": True,
            "created_at": datetime.now().isoformat(),
            "config": schedule_config
        }

        self.schedules[schedule_id] = schedule

        return schedule_id

    def _calculate_next_run(self, schedule_config: Dict[str, Any]) -> str:
        """Calculate next scheduled run"""
        frequency = schedule_config.get("frequency", "daily")
        now = datetime.now()

        if frequency == "hourly":
            next_run = now + timedelta(hours=1)
        elif frequency == "daily":
            next_run = now + timedelta(days=1)
        elif frequency == "weekly":
            next_run = now + timedelta(weeks=1)
        elif frequency == "monthly":
            next_run = now + timedelta(days=30)
        else:
            next_run = now + timedelta(days=1)

        return next_run.isoformat()

    def get_etl_status(self) -> Dict[str, Any]:
        """Get ETL system status"""
        active_jobs = [job for job in self.etl_jobs.values() if job["status"] == "running"]
        completed_jobs = [job for job in self.etl_jobs.values() if job["status"] == "completed"]
        failed_jobs = [job for job in self.etl_jobs.values() if job["status"] == "failed"]

        return {
            "total_pipelines": len(self.pipelines),
            "active_jobs": len(active_jobs),
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len(failed_jobs),
            "active_schedules": len([s for s in self.schedules.values() if s["active"]]),
            "system_health": "healthy" if len(failed_jobs) == 0 else "warning"
        }

class DataQualityMonitor:
    """Data quality monitoring and alerting"""

    def __init__(self):
        self.quality_metrics = {}
        self.quality_thresholds = {
            "completeness": 95.0,
            "accuracy": 98.0,
            "timeliness": 95.0,
            "consistency": 90.0
        }
        self.alerts = []

    def monitor_data_quality(self, data_source: str, data: Any) -> Dict[str, Any]:
        """Monitor data quality for a source"""
        quality_report = self._assess_data_quality(data)

        # Store metrics
        self.quality_metrics[data_source] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": quality_report,
            "data_sample_size": self._get_data_size(data)
        }

        # Check thresholds and generate alerts
        alerts = self._check_quality_thresholds(data_source, quality_report)

        if alerts:
            self.alerts.extend(alerts)

        return {
            "data_source": data_source,
            "quality_report": quality_report,
            "alerts_generated": len(alerts),
            "alerts": alerts
        }

    def _assess_data_quality(self, data: Any) -> Dict[str, float]:
        """Assess overall data quality"""
        if not isinstance(data, list) or not data:
            return {"completeness": 0, "accuracy": 0, "timeliness": 0, "consistency": 0}

        total_records = len(data)

        # Completeness: percentage of non-null values
        total_fields = sum(len(record) for record in data)
        non_null_fields = sum(1 for record in data for value in record.values() if value is not None)
        completeness = (non_null_fields / total_fields) * 100 if total_fields > 0 else 0

        # Accuracy: basic validation (would use more sophisticated checks)
        valid_records = sum(1 for record in data if self._is_record_accurate(record))
        accuracy = (valid_records / total_records) * 100

        # Timeliness: check if data is recent (simplified)
        timely_records = sum(1 for record in data if self._is_record_timely(record))
        timeliness = (timely_records / total_records) * 100

        # Consistency: check for logical consistency
        consistent_records = sum(1 for record in data if self._is_record_consistent(record))
        consistency = (consistent_records / total_records) * 100

        return {
            "completeness": round(completeness, 2),
            "accuracy": round(accuracy, 2),
            "timeliness": round(timeliness, 2),
            "consistency": round(consistency, 2)
        }

    def _is_record_accurate(self, record: Dict[str, Any]) -> bool:
        """Check record accuracy"""
        # Basic accuracy checks
        if "age" in record:
            age = record["age"]
            if not isinstance(age, (int, float)) or age < 0 or age > 150:
                return False

        if "email" in record:
            email = record["email"]
            if email and "@" not in email:
                return False

        return True

    def _is_record_timely(self, record: Dict[str, Any]) -> bool:
        """Check if record is timely"""
        if "timestamp" in record:
            try:
                record_date = datetime.fromisoformat(record["timestamp"])
                days_old = (datetime.now() - record_date).days
                return days_old <= 30  # Consider timely if less than 30 days old
            except:
                pass

        return True  # Default to timely if no timestamp

    def _is_record_consistent(self, record: Dict[str, Any]) -> bool:
        """Check record consistency"""
        # Example consistency checks
        if "birth_date" in record and "age" in record:
            try:
                birth_date = datetime.fromisoformat(record["birth_date"])
                calculated_age = (datetime.now() - birth_date).days // 365
                if abs(calculated_age - record["age"]) > 2:  # Allow 2 year tolerance
                    return False
            except:
                pass

        return True

    def _get_data_size(self, data: Any) -> int:
        """Get data size"""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return 1
        else:
            return 0

    def _check_quality_thresholds(self, data_source: str, quality_report: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check quality metrics against thresholds"""
        alerts = []

        for metric, value in quality_report.items():
            threshold = self.quality_thresholds.get(metric, 100)

            if value < threshold:
                alerts.append({
                    "alert_id": f"quality_{int(datetime.now().timestamp())}_{len(self.alerts)}",
                    "data_source": data_source,
                    "metric": metric,
                    "current_value": value,
                    "threshold": threshold,
                    "severity": "high" if value < threshold * 0.8 else "medium",
                    "message": f"{metric.title()} quality below threshold: {value:.1f}% < {threshold}%",
                    "timestamp": datetime.now().isoformat()
                })

        return alerts

    def get_quality_dashboard(self) -> Dict[str, Any]:
        """Get quality monitoring dashboard"""
        recent_alerts = self.alerts[-10:] if self.alerts else []

        dashboard = {
            "overall_quality_score": self._calculate_overall_quality_score(),
            "data_sources_monitored": len(self.quality_metrics),
            "active_alerts": len([a for a in self.alerts if not a.get("resolved", False)]),
            "recent_alerts": recent_alerts,
            "quality_trends": self._calculate_quality_trends(),
            "generated_at": datetime.now().isoformat()
        }

        return dashboard

    def _calculate_overall_quality_score(self) -> float:
        """Calculate overall quality score across all sources"""
        if not self.quality_metrics:
            return 100.0

        total_score = 0
        count = 0

        for source_metrics in self.quality_metrics.values():
            metrics = source_metrics["metrics"]
            source_score = sum(metrics.values()) / len(metrics)
            total_score += source_score
            count += 1

        return round(total_score / count, 2) if count > 0 else 100.0

    def _calculate_quality_trends(self) -> Dict[str, Any]:
        """Calculate quality trends over time"""
        trends = {}

        for metric in self.quality_thresholds.keys():
            values = []
            timestamps = []

            for source_data in self.quality_metrics.values():
                if metric in source_data["metrics"]:
                    values.append(source_data["metrics"][metric])
                    timestamps.append(source_data["timestamp"])

            if len(values) >= 2:
                trend = "stable"
                if values[-1] > values[0]:
                    trend = "improving"
                elif values[-1] < values[0]:
                    trend = "declining"

                trends[metric] = {
                    "current": values[-1],
                    "previous": values[0],
                    "change": values[-1] - values[0],
                    "trend": trend
                }

        return trends
