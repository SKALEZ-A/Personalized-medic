"""
Comprehensive Logging System for AI Personalized Medicine Platform
Advanced logging with structured data, log analysis, and intelligent filtering
"""

import json
import logging
import logging.handlers
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re
import gzip
import os
from pathlib import Path
import hashlib
import statistics

class StructuredLogger:
    """Advanced structured logging system"""

    def __init__(self, service_name: str = "healthcare_platform"):
        self.service_name = service_name
        self.log_buffer = deque(maxlen=50000)
        self.log_index = defaultdict(lambda: defaultdict(list))
        self.log_handlers = []
        self.filters = {}
        self.formatters = {}
        self.is_running = False
        self.logging_workers = []

        self._setup_base_logging()
        self._initialize_formatters()

    def _setup_base_logging(self):
        """Setup base logging configuration"""
        # Create logger
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.log_handlers.append(console_handler)

        # File handler with rotation
        log_file = f"logs/{self.service_name}.log"
        os.makedirs("logs", exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.log_handlers.append(file_handler)

    def _initialize_formatters(self):
        """Initialize log formatters"""
        self.formatters = {
            "json": self._json_formatter,
            "structured": self._structured_formatter,
            "audit": self._audit_formatter,
            "performance": self._performance_formatter,
            "security": self._security_formatter
        }

    def _json_formatter(self, record: logging.LogRecord) -> str:
        """JSON log formatter"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "service": self.service_name,
            "thread": record.thread,
            "thread_name": record.threadName
        }

        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'patient_id'):
            log_entry["patient_id"] = record.patient_id
        if hasattr(record, 'trace_id'):
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, 'component'):
            log_entry["component"] = record.component
        if hasattr(record, 'operation'):
            log_entry["operation"] = record.operation
        if hasattr(record, 'duration'):
            log_entry["duration"] = record.duration
        if hasattr(record, 'status_code'):
            log_entry["status_code"] = record.status_code
        if hasattr(record, 'error_details'):
            log_entry["error_details"] = record.error_details
        if hasattr(record, 'additional_data'):
            log_entry["additional_data"] = record.additional_data

        return json.dumps(log_entry)

    def _structured_formatter(self, record: logging.LogRecord) -> str:
        """Structured text formatter"""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        level = record.levelname
        component = getattr(record, 'component', 'unknown')
        operation = getattr(record, 'operation', 'unknown')

        return f"[{timestamp}] {level} [{component}:{operation}] {record.getMessage()}"

    def _audit_formatter(self, record: logging.LogRecord) -> str:
        """Audit log formatter"""
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        user_id = getattr(record, 'user_id', 'system')
        action = getattr(record, 'operation', 'unknown')
        resource = getattr(record, 'resource', 'unknown')
        result = getattr(record, 'result', 'unknown')

        return f"AUDIT|{timestamp}|{user_id}|{action}|{resource}|{result}|{record.getMessage()}"

    def _performance_formatter(self, record: logging.LogRecord) -> str:
        """Performance log formatter"""
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        operation = getattr(record, 'operation', 'unknown')
        duration = getattr(record, 'duration', 0)
        status = getattr(record, 'status', 'unknown')

        return f"PERF|{timestamp}|{operation}|{duration:.3f}s|{status}|{record.getMessage()}"

    def _security_formatter(self, record: logging.LogRecord) -> str:
        """Security log formatter"""
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        event_type = getattr(record, 'event_type', 'unknown')
        user_id = getattr(record, 'user_id', 'unknown')
        ip_address = getattr(record, 'ip_address', 'unknown')
        severity = getattr(record, 'severity', 'info')

        return f"SEC|{timestamp}|{severity}|{event_type}|{user_id}|{ip_address}|{record.getMessage()}"

    def add_filter(self, filter_name: str, filter_func: Callable[[logging.LogRecord], bool]):
        """Add a custom log filter"""
        self.filters[filter_name] = filter_func

    def log(self, level: str, message: str, **kwargs):
        """Log a message with structured data"""
        # Create log record with extra fields
        extra = {}
        for key, value in kwargs.items():
            if key in ['user_id', 'patient_id', 'trace_id', 'component', 'operation',
                      'duration', 'status_code', 'error_details', 'additional_data',
                      'resource', 'result', 'event_type', 'ip_address', 'severity', 'status']:
                extra[key] = value

        # Log using standard logging
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, extra=extra)

        # Store in buffer for analysis
        log_entry = {
            "timestamp": datetime.now(),
            "level": level.upper(),
            "message": message,
            **kwargs
        }
        self.log_buffer.append(log_entry)

        # Index the log entry
        self._index_log_entry(log_entry)

    def _index_log_entry(self, log_entry: Dict[str, Any]):
        """Index log entry for efficient querying"""
        # Index by level
        level = log_entry["level"]
        self.log_index["level"][level].append(log_entry)

        # Index by component
        component = log_entry.get("component", "unknown")
        self.log_index["component"][component].append(log_entry)

        # Index by user
        user_id = log_entry.get("user_id", "unknown")
        self.log_index["user"][user_id].append(log_entry)

        # Index by trace
        trace_id = log_entry.get("trace_id", "unknown")
        self.log_index["trace"][trace_id].append(log_entry)

        # Index by time (hourly buckets)
        hour_bucket = log_entry["timestamp"].strftime("%Y-%m-%d-%H")
        self.log_index["time"][hour_bucket].append(log_entry)

        # Limit index size
        for index_type in self.log_index:
            for key in self.log_index[index_type]:
                if len(self.log_index[index_type][key]) > 1000:
                    self.log_index[index_type][key] = self.log_index[index_type][key][-1000:]

    def query_logs(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Query logs with advanced filtering"""
        if not filters:
            return list(self.log_buffer)[-limit:]

        # Start with all logs
        matching_logs = list(self.log_buffer)

        # Apply filters
        for filter_key, filter_value in filters.items():
            if filter_key == "level":
                matching_logs = [log for log in matching_logs if log.get("level") == filter_value.upper()]
            elif filter_key == "component":
                matching_logs = [log for log in matching_logs if log.get("component") == filter_value]
            elif filter_key == "user_id":
                matching_logs = [log for log in matching_logs if log.get("user_id") == filter_value]
            elif filter_key == "trace_id":
                matching_logs = [log for log in matching_logs if log.get("trace_id") == filter_value]
            elif filter_key == "time_range":
                start_time, end_time = filter_value
                matching_logs = [log for log in matching_logs
                               if start_time <= log["timestamp"] <= end_time]
            elif filter_key == "contains":
                matching_logs = [log for log in matching_logs
                               if filter_value.lower() in log.get("message", "").lower()]
            elif filter_key == "regex":
                pattern = re.compile(filter_value, re.IGNORECASE)
                matching_logs = [log for log in matching_logs
                               if pattern.search(log.get("message", ""))]

        return matching_logs[-limit:]

    def start_log_processing(self):
        """Start background log processing"""
        self.is_running = True

        # Start log analysis worker
        analysis_worker = threading.Thread(target=self._process_logs_background, daemon=True)
        analysis_worker.start()
        self.logging_workers.append(analysis_worker)

        # Start log archiving worker
        archive_worker = threading.Thread(target=self._archive_logs_background, daemon=True)
        archive_worker.start()
        self.logging_workers.append(archive_worker)

    def stop_log_processing(self):
        """Stop background log processing"""
        self.is_running = False

    def _process_logs_background(self):
        """Background log processing and analysis"""
        while self.is_running:
            try:
                if len(self.log_buffer) > 100:
                    # Process batch of logs
                    batch_size = min(500, len(self.log_buffer))
                    log_batch = []
                    for _ in range(batch_size):
                        log_batch.append(self.log_buffer.popleft())

                    # Analyze batch
                    self._analyze_log_batch(log_batch)

                time.sleep(30)  # Process every 30 seconds

            except Exception as e:
                print(f"Log processing error: {e}")

    def _analyze_log_batch(self, log_batch: List[Dict[str, Any]]):
        """Analyze a batch of logs"""
        # Count errors and warnings
        error_count = sum(1 for log in log_batch if log["level"] == "ERROR")
        warning_count = sum(1 for log in log_batch if log["level"] == "WARNING")

        # Detect patterns
        if error_count > len(log_batch) * 0.1:  # More than 10% errors
            print(f"🚨 HIGH ERROR RATE DETECTED: {error_count}/{len(log_batch)} logs are errors")

        # Analyze performance logs
        performance_logs = [log for log in log_batch if log.get("component") == "performance"]
        if performance_logs:
            avg_duration = statistics.mean([log.get("duration", 0) for log in performance_logs])
            if avg_duration > 2.0:  # Average > 2 seconds
                print(f"🐌 SLOW PERFORMANCE DETECTED: Average operation time {avg_duration:.2f}s")

    def _archive_logs_background(self):
        """Background log archiving"""
        while self.is_running:
            try:
                # Archive logs older than 24 hours
                self._archive_old_logs()
                time.sleep(3600)  # Archive every hour

            except Exception as e:
                print(f"Log archiving error: {e}")

    def _archive_old_logs(self):
        """Archive old log entries"""
        cutoff_time = datetime.now() - timedelta(hours=24)

        # Find logs to archive
        logs_to_archive = []
        remaining_logs = deque()

        while self.log_buffer:
            log_entry = self.log_buffer.popleft()
            if log_entry["timestamp"] < cutoff_time:
                logs_to_archive.append(log_entry)
            else:
                remaining_logs.append(log_entry)

        self.log_buffer = remaining_logs

        if logs_to_archive:
            # Archive to compressed file
            archive_filename = f"logs/archive_{int(time.time())}.json.gz"
            with gzip.open(archive_filename, 'wt', encoding='utf-8') as f:
                json.dump(logs_to_archive, f, default=str, indent=2)

            print(f"📦 Archived {len(logs_to_archive)} log entries to {archive_filename}")

class LogAnalyzer:
    """Advanced log analysis and insights system"""

    def __init__(self, structured_logger: StructuredLogger):
        self.logger = structured_logger
        self.analysis_results = {}
        self.patterns = {}
        self.anomalies = []
        self.insights = []

    def analyze_logs(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Perform comprehensive log analysis"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_logs = [log for log in self.logger.log_buffer if log["timestamp"] > cutoff_time]

        analysis = {
            "time_window_hours": time_window_hours,
            "total_logs": len(recent_logs),
            "analysis_timestamp": datetime.now(),
            "summary": self._generate_summary(recent_logs),
            "patterns": self._detect_patterns(recent_logs),
            "anomalies": self._detect_anomalies(recent_logs),
            "performance_insights": self._analyze_performance(recent_logs),
            "security_insights": self._analyze_security(recent_logs),
            "recommendations": self._generate_recommendations(recent_logs)
        }

        self.analysis_results = analysis
        return analysis

    def _generate_summary(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate log summary statistics"""
        if not logs:
            return {"empty": True}

        # Level distribution
        level_counts = defaultdict(int)
        for log in logs:
            level_counts[log["level"]] += 1

        # Component distribution
        component_counts = defaultdict(int)
        for log in logs:
            component = log.get("component", "unknown")
            component_counts[component] += 1

        # Time distribution (by hour)
        hourly_distribution = defaultdict(int)
        for log in logs:
            hour = log["timestamp"].hour
            hourly_distribution[hour] += 1

        # User activity
        user_activity = defaultdict(int)
        for log in logs:
            user_id = log.get("user_id", "unknown")
            user_activity[user_id] += 1

        return {
            "level_distribution": dict(level_counts),
            "component_distribution": dict(component_counts),
            "hourly_distribution": dict(hourly_distribution),
            "user_activity": dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            "logs_per_hour": len(logs) / 24,
            "error_rate": level_counts.get("ERROR", 0) / len(logs) if logs else 0,
            "warning_rate": level_counts.get("WARNING", 0) / len(logs) if logs else 0
        }

    def _detect_patterns(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect log patterns and trends"""
        patterns = {}

        if len(logs) < 10:
            return {"insufficient_data": True}

        # Error pattern analysis
        error_logs = [log for log in logs if log["level"] == "ERROR"]
        if error_logs:
            # Group errors by message similarity
            error_groups = defaultdict(list)
            for error in error_logs:
                # Simple similarity based on first 50 characters
                key = error["message"][:50]
                error_groups[key].append(error)

            patterns["error_clusters"] = [
                {
                    "pattern": key,
                    "count": len(group),
                    "percentage": len(group) / len(error_logs) * 100,
                    "sample_message": group[0]["message"]
                }
                for key, group in error_groups.items()
                if len(group) > 1
            ]

        # Performance pattern analysis
        performance_logs = [log for log in logs if log.get("component") == "performance"]
        if performance_logs and len(performance_logs) > 5:
            durations = [log.get("duration", 0) for log in performance_logs]
            patterns["performance_trends"] = {
                "avg_duration": statistics.mean(durations),
                "p95_duration": self._percentile(durations, 95),
                "slow_operations": len([d for d in durations if d > 2.0]),
                "trend": self._calculate_trend(durations)
            }

        # User behavior patterns
        user_patterns = defaultdict(lambda: defaultdict(int))
        for log in logs:
            user_id = log.get("user_id", "unknown")
            operation = log.get("operation", "unknown")
            user_patterns[user_id][operation] += 1

        patterns["user_behavior"] = {
            "active_users": len(user_patterns),
            "most_common_operations": self._get_most_common_operations(user_patterns)
        }

        return patterns

    def _detect_anomalies(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalous log patterns"""
        anomalies = []

        if len(logs) < 20:
            return anomalies

        # Statistical anomaly detection
        # Check for unusual error spikes
        recent_logs = logs[-100:]  # Last 100 logs
        error_rate = sum(1 for log in recent_logs if log["level"] == "ERROR") / len(recent_logs)

        if error_rate > 0.2:  # More than 20% errors in recent logs
            anomalies.append({
                "type": "error_spike",
                "severity": "high",
                "description": f"Error rate of {error_rate:.1%} detected in recent logs",
                "timestamp": datetime.now(),
                "affected_logs": len([log for log in recent_logs if log["level"] == "ERROR"])
            })

        # Check for unusual user activity
        user_activity = defaultdict(int)
        for log in logs[-200:]:  # Last 200 logs
            user_id = log.get("user_id", "unknown")
            user_activity[user_id] += 1

        if user_activity:
            avg_activity = statistics.mean(user_activity.values())
            std_activity = statistics.stdev(user_activity.values()) if len(user_activity) > 1 else 0

            for user_id, activity in user_activity.items():
                if std_activity > 0 and abs(activity - avg_activity) > 2 * std_activity:
                    anomalies.append({
                        "type": "unusual_user_activity",
                        "severity": "medium",
                        "description": f"User {user_id} has unusual activity level: {activity} logs",
                        "timestamp": datetime.now(),
                        "deviation": abs(activity - avg_activity) / std_activity
                    })

        # Check for security anomalies
        security_logs = [log for log in logs if log.get("component") == "security"]
        failed_auth_attempts = sum(1 for log in security_logs[-50:]
                                 if "failed" in log.get("message", "").lower())

        if failed_auth_attempts > 10:
            anomalies.append({
                "type": "security_threat",
                "severity": "critical",
                "description": f"High number of failed authentication attempts: {failed_auth_attempts}",
                "timestamp": datetime.now(),
                "recommendation": "Investigate potential security breach"
            })

        return anomalies

    def _analyze_performance(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance-related logs"""
        performance_logs = [log for log in logs if log.get("component") == "performance"]

        if not performance_logs:
            return {"no_performance_logs": True}

        # Duration analysis
        durations = [log.get("duration", 0) for log in performance_logs]
        operations = [log.get("operation", "unknown") for log in performance_logs]

        # Group by operation
        operation_stats = defaultdict(list)
        for i, operation in enumerate(operations):
            operation_stats[operation].append(durations[i])

        operation_summary = {}
        for operation, op_durations in operation_stats.items():
            operation_summary[operation] = {
                "count": len(op_durations),
                "avg_duration": statistics.mean(op_durations),
                "p95_duration": self._percentile(op_durations, 95),
                "max_duration": max(op_durations)
            }

        return {
            "total_performance_logs": len(performance_logs),
            "overall_stats": {
                "avg_duration": statistics.mean(durations),
                "p95_duration": self._percentile(durations, 95),
                "p99_duration": self._percentile(durations, 99),
                "slow_operations": len([d for d in durations if d > 1.0])
            },
            "operation_breakdown": operation_summary,
            "performance_trend": self._calculate_trend(durations[-50:])  # Last 50 measurements
        }

    def _analyze_security(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze security-related logs"""
        security_logs = [log for log in logs if log.get("component") == "security"]

        insights = {
            "total_security_logs": len(security_logs),
            "threats_detected": 0,
            "auth_failures": 0,
            "suspicious_activity": 0
        }

        if not security_logs:
            return insights

        for log in security_logs:
            message = log.get("message", "").lower()

            if "threat" in message or "attack" in message:
                insights["threats_detected"] += 1
            if "failed" in message and "auth" in message:
                insights["auth_failures"] += 1
            if "suspicious" in message or "unusual" in message:
                insights["suspicious_activity"] += 1

        # Risk assessment
        risk_score = min(100, (insights["threats_detected"] * 20 +
                               insights["auth_failures"] * 5 +
                               insights["suspicious_activity"] * 10))

        insights["risk_score"] = risk_score
        insights["risk_level"] = "low" if risk_score < 30 else "medium" if risk_score < 70 else "high"

        return insights

    def _generate_recommendations(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate log analysis recommendations"""
        recommendations = []

        summary = self._generate_summary(logs)

        # Error rate recommendations
        error_rate = summary.get("error_rate", 0)
        if error_rate > 0.05:  # More than 5% errors
            recommendations.append({
                "category": "reliability",
                "priority": "high",
                "recommendation": f"High error rate detected: {error_rate:.1%}",
                "actions": ["Review error logs", "Implement error handling", "Fix root causes"]
            })

        # Performance recommendations
        performance = self._analyze_performance(logs)
        if not performance.get("no_performance_logs"):
            overall_stats = performance.get("overall_stats", {})
            slow_ops = overall_stats.get("slow_operations", 0)
            if slow_ops > len(logs) * 0.1:  # More than 10% slow operations
                recommendations.append({
                    "category": "performance",
                    "priority": "medium",
                    "recommendation": f"High number of slow operations: {slow_ops}",
                    "actions": ["Optimize slow endpoints", "Implement caching", "Database query optimization"]
                })

        # Security recommendations
        security = self._analyze_security(logs)
        if security.get("risk_level") == "high":
            recommendations.append({
                "category": "security",
                "priority": "critical",
                "recommendation": f"High security risk detected (score: {security.get('risk_score', 0)})",
                "actions": ["Review security logs", "Implement additional security measures", "Conduct security audit"]
            })

        return recommendations

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 5:
            return "insufficient_data"

        # Compare first half with second half
        midpoint = len(values) // 2
        first_half = values[:midpoint]
        second_half = values[midpoint:]

        if not first_half or not second_half:
            return "insufficient_data"

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _get_most_common_operations(self, user_patterns: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
        """Get most common operations across users"""
        operation_totals = defaultdict(int)

        for user_ops in user_patterns.values():
            for operation, count in user_ops.items():
                operation_totals[operation] += count

        return [
            {"operation": op, "total_count": count}
            for op, count in sorted(operation_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0
        values_sorted = sorted(values)
        index = int(len(values_sorted) * percentile / 100)
        return values_sorted[min(index, len(values_sorted) - 1)]

class LogArchiver:
    """Intelligent log archiving and retention system"""

    def __init__(self, structured_logger: StructuredLogger):
        self.logger = structured_logger
        self.archive_path = Path("logs/archive")
        self.archive_path.mkdir(exist_ok=True)
        self.retention_policies = {
            "debug": 7,      # 7 days
            "info": 30,      # 30 days
            "warning": 90,   # 90 days
            "error": 365,    # 1 year
            "critical": 365  # 1 year
        }
        self.compression_enabled = True

    def archive_logs(self, level: str = None, days_old: int = 1) -> Dict[str, Any]:
        """Archive logs based on criteria"""
        cutoff_time = datetime.now() - timedelta(days=days_old)
        logs_to_archive = []

        # Collect logs to archive
        for log_entry in list(self.logger.log_buffer):
            if log_entry["timestamp"] < cutoff_time:
                if level is None or log_entry["level"] == level.upper():
                    logs_to_archive.append(log_entry)

        if not logs_to_archive:
            return {"archived_count": 0, "message": "No logs to archive"}

        # Create archive file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        level_suffix = f"_{level.lower()}" if level else ""
        filename = f"archive_{timestamp}{level_suffix}.json"

        if self.compression_enabled:
            filename += ".gz"

        archive_file = self.archive_path / filename

        # Write archive
        if self.compression_enabled:
            with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
                json.dump(logs_to_archive, f, default=str, indent=2)
        else:
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(logs_to_archive, f, default=str, indent=2)

        # Remove archived logs from buffer
        remaining_logs = deque()
        archived_ids = {self._log_hash(log) for log in logs_to_archive}

        for log_entry in self.logger.log_buffer:
            if self._log_hash(log_entry) not in archived_ids:
                remaining_logs.append(log_entry)

        self.logger.log_buffer = remaining_logs

        return {
            "archived_count": len(logs_to_archive),
            "archive_file": str(archive_file),
            "compression": self.compression_enabled,
            "file_size_mb": archive_file.stat().st_size / 1024 / 1024
        }

    def cleanup_old_archives(self) -> Dict[str, Any]:
        """Clean up old archive files based on retention policies"""
        cleaned_files = []
        total_space_reclaimed = 0

        for archive_file in self.archive_path.glob("*.json*"):
            if self._should_cleanup_archive(archive_file):
                file_size = archive_file.stat().st_size
                archive_file.unlink()
                cleaned_files.append(str(archive_file))
                total_space_reclaimed += file_size

        return {
            "cleaned_files": len(cleaned_files),
            "files_list": cleaned_files,
            "space_reclaimed_mb": total_space_reclaimed / 1024 / 1024
        }

    def _should_cleanup_archive(self, archive_file: Path) -> bool:
        """Determine if archive file should be cleaned up"""
        # Extract date from filename
        filename = archive_file.name
        # archive_20231201_143022.json.gz -> 20231201
        date_match = re.search(r'archive_(\d{8})', filename)
        if not date_match:
            return False

        archive_date = datetime.strptime(date_match.group(1), "%Y%m%d")
        days_old = (datetime.now() - archive_date).days

        # Check retention policy (use error level as default)
        retention_days = self.retention_policies.get("error", 365)

        return days_old > retention_days

    def _log_hash(self, log_entry: Dict[str, Any]) -> str:
        """Generate hash for log entry deduplication"""
        key_data = f"{log_entry['timestamp']}{log_entry['level']}{log_entry['message']}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archive statistics"""
        archive_files = list(self.archive_path.glob("*.json*"))

        total_size = sum(f.stat().st_size for f in archive_files)
        file_count = len(archive_files)

        # Size by level
        size_by_level = defaultdict(int)
        for archive_file in archive_files:
            filename = archive_file.name
            level = "unknown"
            if "_debug." in filename:
                level = "debug"
            elif "_info." in filename:
                level = "info"
            elif "_warning." in filename:
                level = "warning"
            elif "_error." in filename:
                level = "error"
            elif "_critical." in filename:
                level = "critical"

            size_by_level[level] += archive_file.stat().st_size

        return {
            "total_files": file_count,
            "total_size_mb": total_size / 1024 / 1024,
            "size_by_level": dict(size_by_level),
            "oldest_archive": min((f.stat().st_mtime for f in archive_files), default=None),
            "newest_archive": max((f.stat().st_mtime for f in archive_files), default=None)
        }

class LogCorrelator:
    """Log correlation and trace reconstruction system"""

    def __init__(self, structured_logger: StructuredLogger):
        self.logger = structured_logger
        self.correlations = defaultdict(list)
        self.trace_patterns = {}

    def correlate_logs(self, trace_id: str) -> Dict[str, Any]:
        """Correlate logs by trace ID"""
        related_logs = self.logger.query_logs({"trace_id": trace_id})

        if not related_logs:
            return {"trace_id": trace_id, "logs_found": 0}

        # Sort by timestamp
        related_logs.sort(key=lambda x: x["timestamp"])

        # Group by component
        component_logs = defaultdict(list)
        for log in related_logs:
            component = log.get("component", "unknown")
            component_logs[component].append(log)

        # Analyze flow
        flow_analysis = self._analyze_log_flow(related_logs)

        return {
            "trace_id": trace_id,
            "total_logs": len(related_logs),
            "duration": self._calculate_trace_duration(related_logs),
            "components_involved": list(component_logs.keys()),
            "component_breakdown": {comp: len(logs) for comp, logs in component_logs.items()},
            "flow_analysis": flow_analysis,
            "error_sequence": self._extract_error_sequence(related_logs),
            "performance_timeline": self._build_performance_timeline(related_logs)
        }

    def find_related_traces(self, log_entry: Dict[str, Any]) -> List[str]:
        """Find traces related to a log entry"""
        related_traces = set()

        # Find by user
        user_id = log_entry.get("user_id")
        if user_id:
            user_logs = self.logger.query_logs({"user_id": user_id}, limit=100)
            related_traces.update(log.get("trace_id") for log in user_logs if log.get("trace_id"))

        # Find by component and time window
        component = log_entry.get("component")
        timestamp = log_entry["timestamp"]
        time_window = timedelta(minutes=5)

        if component:
            component_logs = self.logger.query_logs({
                "component": component,
                "time_range": (timestamp - time_window, timestamp + time_window)
            }, limit=50)

            related_traces.update(log.get("trace_id") for log in component_logs if log.get("trace_id"))

        return list(related_traces)

    def _analyze_log_flow(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the flow of logs in a trace"""
        if len(logs) < 2:
            return {"insufficient_logs": True}

        # Analyze component transitions
        transitions = []
        for i in range(len(logs) - 1):
            from_comp = logs[i].get("component", "unknown")
            to_comp = logs[i + 1].get("component", "unknown")
            time_diff = (logs[i + 1]["timestamp"] - logs[i]["timestamp"]).total_seconds()

            transitions.append({
                "from": from_comp,
                "to": to_comp,
                "time_diff": time_diff,
                "level": logs[i]["level"]
            })

        # Calculate average transition times
        avg_transition_times = defaultdict(list)
        for transition in transitions:
            key = f"{transition['from']}->{transition['to']}"
            avg_transition_times[key].append(transition['time_diff'])

        transition_stats = {}
        for key, times in avg_transition_times.items():
            transition_stats[key] = {
                "avg_time": statistics.mean(times),
                "count": len(times)
            }

        return {
            "total_transitions": len(transitions),
            "unique_transitions": len(transition_stats),
            "transition_stats": transition_stats,
            "bottlenecks": self._identify_flow_bottlenecks(transitions)
        }

    def _calculate_trace_duration(self, logs: List[Dict[str, Any]]) -> float:
        """Calculate total trace duration"""
        if not logs:
            return 0

        start_time = min(log["timestamp"] for log in logs)
        end_time = max(log["timestamp"] for log in logs)

        return (end_time - start_time).total_seconds()

    def _extract_error_sequence(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract error sequence from logs"""
        error_logs = [log for log in logs if log["level"] in ["ERROR", "CRITICAL"]]

        return [{
            "timestamp": log["timestamp"],
            "component": log.get("component", "unknown"),
            "message": log["message"],
            "level": log["level"]
        } for log in error_logs]

    def _build_performance_timeline(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build performance timeline for trace"""
        performance_logs = [log for log in logs if log.get("duration", 0) > 0]

        return [{
            "timestamp": log["timestamp"],
            "operation": log.get("operation", "unknown"),
            "duration": log["duration"],
            "component": log.get("component", "unknown")
        } for log in performance_logs]

    def _identify_flow_bottlenecks(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify bottlenecks in log flow"""
        bottlenecks = []

        # Find slow transitions
        slow_transitions = [t for t in transitions if t["time_diff"] > 1.0]  # > 1 second

        if slow_transitions:
            bottlenecks.append({
                "type": "slow_transitions",
                "count": len(slow_transitions),
                "avg_delay": statistics.mean(t["time_diff"] for t in slow_transitions),
                "description": f"{len(slow_transitions)} transitions taking >1 second"
            })

        # Find error cascades
        error_transitions = [t for t in transitions if t["level"] in ["ERROR", "CRITICAL"]]
        if len(error_transitions) > len(transitions) * 0.3:  # >30% error transitions
            bottlenecks.append({
                "type": "error_cascade",
                "error_rate": len(error_transitions) / len(transitions),
                "description": "High error rate in component transitions"
            })

        return bottlenecks
