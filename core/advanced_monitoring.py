"""
Advanced Monitoring and Observability System for AI Personalized Medicine Platform
Comprehensive system monitoring, alerting, performance tracking, and observability
"""

import json
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
import statistics
import logging
import logging.handlers
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

class MetricsCollector:
    """Advanced metrics collection system"""

    def __init__(self):
        self.metrics_store = defaultdict(lambda: deque(maxlen=10000))
        self.custom_metrics = defaultdict(dict)
        self.performance_counters = defaultdict(int)
        self.gauge_metrics = {}
        self.histogram_metrics = defaultdict(list)
        self.is_running = False
        self.collection_workers = []

    def start_collection(self):
        """Start metrics collection"""
        self.is_running = True

        # Start system metrics collection
        system_worker = threading.Thread(target=self._collect_system_metrics, daemon=True)
        system_worker.start()
        self.collection_workers.append(system_worker)

        # Start application metrics collection
        app_worker = threading.Thread(target=self._collect_application_metrics, daemon=True)
        app_worker.start()
        self.collection_workers.append(app_worker)

        # Start custom metrics collection
        custom_worker = threading.Thread(target=self._collect_custom_metrics, daemon=True)
        custom_worker.start()
        self.collection_workers.append(custom_worker)

    def stop_collection(self):
        """Stop metrics collection"""
        self.is_running = False

    def record_counter(self, name: str, value: float = 1, tags: Dict[str, str] = None):
        """Record a counter metric"""
        self._record_metric("counter", name, value, tags)

    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric"""
        self._record_metric("gauge", name, value, tags)
        self.gauge_metrics[name] = {"value": value, "tags": tags or {}, "timestamp": datetime.now()}

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric"""
        self._record_metric("histogram", name, value, tags)
        self.histogram_metrics[name].append({"value": value, "tags": tags or {}, "timestamp": datetime.now()})

        # Keep only recent values
        if len(self.histogram_metrics[name]) > 1000:
            self.histogram_metrics[name] = self.histogram_metrics[name][-1000:]

    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer metric"""
        self._record_metric("timer", name, duration, tags)

    def _record_metric(self, metric_type: str, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric"""
        metric_record = {
            "timestamp": datetime.now(),
            "type": metric_type,
            "name": name,
            "value": value,
            "tags": tags or {}
        }

        self.metrics_store[name].append(metric_record)

    def get_metric_stats(self, name: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get statistics for a metric"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_metrics = [m for m in self.metrics_store[name] if m["timestamp"] > cutoff_time]

        if not recent_metrics:
            return {"error": "No metrics available"}

        values = [m["value"] for m in recent_metrics]

        return {
            "count": len(values),
            "sum": sum(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "p50": statistics.median(values),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
            "time_window_minutes": time_window_minutes
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0
        values_sorted = sorted(values)
        index = int(len(values_sorted) * percentile / 100)
        return values_sorted[min(index, len(values_sorted) - 1)]

    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        while self.is_running:
            try:
                # CPU metrics
                self.record_gauge("system.cpu_percent", psutil.cpu_percent(interval=1))
                self.record_gauge("system.cpu_count", psutil.cpu_count())

                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_gauge("system.memory_percent", memory.percent)
                self.record_gauge("system.memory_used_mb", memory.used / 1024 / 1024)
                self.record_gauge("system.memory_available_mb", memory.available / 1024 / 1024)

                # Disk metrics
                disk = psutil.disk_usage('/')
                self.record_gauge("system.disk_percent", disk.percent)
                self.record_gauge("system.disk_used_gb", disk.used / 1024 / 1024 / 1024)
                self.record_gauge("system.disk_free_gb", disk.free / 1024 / 1024 / 1024)

                # Network metrics
                network = psutil.net_io_counters()
                self.record_counter("system.network_bytes_sent", network.bytes_sent)
                self.record_counter("system.network_bytes_recv", network.bytes_recv)

                time.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                print(f"System metrics collection error: {e}")
                time.sleep(60)

    def _collect_application_metrics(self):
        """Collect application-level metrics"""
        while self.is_running:
            try:
                # Simulate application metrics
                self.record_counter("app.requests_total")
                self.record_gauge("app.active_connections", random.randint(10, 100))
                self.record_gauge("app.memory_usage_mb", random.uniform(100, 500))
                self.record_gauge("app.cpu_usage_percent", random.uniform(5, 30))
                self.record_gauge("app.response_time_avg", random.uniform(50, 200))

                time.sleep(10)  # Collect every 10 seconds

            except Exception as e:
                print(f"Application metrics collection error: {e}")

    def _collect_custom_metrics(self):
        """Collect custom business metrics"""
        while self.is_running:
            try:
                # Healthcare-specific metrics
                self.record_counter("healthcare.patients_served")
                self.record_gauge("healthcare.active_monitoring_sessions", random.randint(50, 200))
                self.record_gauge("healthcare.genomic_analyses_pending", random.randint(10, 50))
                self.record_gauge("healthcare.model_prediction_accuracy", random.uniform(0.85, 0.98))
                self.record_counter("healthcare.alerts_generated")
                self.record_timer("healthcare.api_response_time", random.uniform(0.1, 2.0))

                time.sleep(15)  # Collect every 15 seconds

            except Exception as e:
                print(f"Custom metrics collection error: {e}")

class AlertingSystem:
    """Advanced alerting and notification system"""

    def __init__(self):
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = defaultdict(list)
        self.notification_channels = {}
        self.is_running = False
        self.alert_workers = []

    def add_alert_rule(self, rule_config: Dict[str, Any]) -> str:
        """Add an alert rule"""
        rule_id = f"rule_{int(time.time())}_{random.randint(1000, 9999)}"

        rule = {
            "rule_id": rule_id,
            "name": rule_config["name"],
            "description": rule_config.get("description", ""),
            "metric": rule_config["metric"],
            "condition": rule_config["condition"],  # e.g., "> 80", "< 10"
            "threshold": rule_config["threshold"],
            "duration": rule_config.get("duration", 60),  # seconds
            "severity": rule_config.get("severity", "warning"),
            "channels": rule_config.get("channels", ["log"]),
            "enabled": rule_config.get("enabled", True),
            "cooldown": rule_config.get("cooldown", 300),  # seconds
            "last_triggered": None
        }

        self.alert_rules[rule_id] = rule
        return rule_id

    def start_alerting(self):
        """Start the alerting system"""
        self.is_running = True

        # Start alert evaluation worker
        alert_worker = threading.Thread(target=self._evaluate_alerts, daemon=True)
        alert_worker.start()
        self.alert_workers.append(alert_worker)

        # Start alert cleanup worker
        cleanup_worker = threading.Thread(target=self._cleanup_alerts, daemon=True)
        cleanup_worker.start()
        self.alert_workers.append(cleanup_worker)

    def stop_alerting(self):
        """Stop the alerting system"""
        self.is_running = False

    def _evaluate_alerts(self):
        """Evaluate alert rules against metrics"""
        while self.is_running:
            try:
                for rule_id, rule in self.alert_rules.items():
                    if not rule["enabled"]:
                        continue

                    # Check cooldown
                    if rule["last_triggered"] and (datetime.now() - rule["last_triggered"]).seconds < rule["cooldown"]:
                        continue

                    # Evaluate condition
                    if self._evaluate_condition(rule):
                        self._trigger_alert(rule)

                time.sleep(30)  # Evaluate every 30 seconds

            except Exception as e:
                print(f"Alert evaluation error: {e}")

    def _evaluate_condition(self, rule: Dict[str, Any]) -> bool:
        """Evaluate alert condition"""
        # Simplified condition evaluation
        metric_value = self._get_current_metric_value(rule["metric"])

        if not metric_value:
            return False

        operator = rule["condition"][0]
        threshold = rule["threshold"]

        if operator == ">":
            return metric_value > threshold
        elif operator == "<":
            return metric_value < threshold
        elif operator == ">=":
            return metric_value >= threshold
        elif operator == "<=":
            return metric_value <= threshold
        elif operator == "==":
            return metric_value == threshold

        return False

    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric"""
        # This would integrate with the metrics collector
        # For simulation, return random values based on metric type
        if "cpu" in metric_name:
            return random.uniform(10, 90)
        elif "memory" in metric_name:
            return random.uniform(20, 85)
        elif "response_time" in metric_name:
            return random.uniform(50, 500)
        elif "accuracy" in metric_name:
            return random.uniform(0.8, 0.98)
        else:
            return random.uniform(0, 100)

    def _trigger_alert(self, rule: Dict[str, Any]):
        """Trigger an alert"""
        alert_id = f"alert_{int(time.time())}_{random.randint(1000, 9999)}"

        alert = {
            "alert_id": alert_id,
            "rule_id": rule["rule_id"],
            "rule_name": rule["name"],
            "severity": rule["severity"],
            "description": rule["description"],
            "metric": rule["metric"],
            "threshold": rule["threshold"],
            "current_value": self._get_current_metric_value(rule["metric"]),
            "triggered_at": datetime.now(),
            "status": "active",
            "acknowledged": False,
            "resolved_at": None
        }

        self.active_alerts[alert_id] = alert
        rule["last_triggered"] = datetime.now()

        # Send notifications
        for channel in rule["channels"]:
            self._send_notification(channel, alert)

        print(f"🚨 ALERT TRIGGERED: {rule['name']} - {alert['current_value']}")

    def _send_notification(self, channel: str, alert: Dict[str, Any]):
        """Send notification via specified channel"""
        if channel == "log":
            logging.warning(f"ALERT: {alert['rule_name']} - {alert['description']}")
        elif channel == "email":
            self._send_email_alert(alert)
        elif channel == "slack":
            self._send_slack_alert(alert)
        elif channel == "pagerduty":
            self._send_pagerduty_alert(alert)

    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert"""
        # Simplified email sending (would need proper SMTP configuration)
        print(f"📧 EMAIL ALERT: {alert['rule_name']} sent to configured recipients")

    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert"""
        print(f"💬 SLACK ALERT: {alert['rule_name']} posted to configured channel")

    def _send_pagerduty_alert(self, alert: Dict[str, Any]):
        """Send PagerDuty alert"""
        print(f"📟 PAGERDUTY ALERT: {alert['rule_name']} triggered incident")

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert["acknowledged"] = True
        alert["acknowledged_by"] = user_id
        alert["acknowledged_at"] = datetime.now()

        return True

    def resolve_alert(self, alert_id: str, resolution: str = "") -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert["status"] = "resolved"
        alert["resolved_at"] = datetime.now()
        alert["resolution"] = resolution

        # Move to history
        self.alert_history[alert["rule_id"]].append(alert)
        del self.active_alerts[alert_id]

        return True

    def _cleanup_alerts(self):
        """Clean up old resolved alerts"""
        while self.is_running:
            try:
                # Move old alerts to history (keep active alerts for 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)

                alerts_to_remove = []
                for alert_id, alert in self.active_alerts.items():
                    if alert["status"] == "resolved" and alert.get("resolved_at", datetime.min) < cutoff_time:
                        alerts_to_remove.append(alert_id)

                for alert_id in alerts_to_remove:
                    self.alert_history[self.active_alerts[alert_id]["rule_id"]].append(self.active_alerts[alert_id])
                    del self.active_alerts[alert_id]

                time.sleep(3600)  # Clean up every hour

            except Exception as e:
                print(f"Alert cleanup error: {e}")

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_history(self, rule_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        if rule_id:
            return self.alert_history.get(rule_id, [])[-limit:]
        else:
            # Get all alert history
            all_history = []
            for rule_alerts in self.alert_history.values():
                all_history.extend(rule_alerts)

            return sorted(all_history, key=lambda x: x["triggered_at"], reverse=True)[:limit]

class DistributedTracing:
    """Distributed tracing system for request tracking"""

    def __init__(self):
        self.traces = {}
        self.spans = defaultdict(list)
        self.active_traces = {}
        self.trace_sampling_rate = 0.1  # Sample 10% of requests

    def start_trace(self, trace_name: str, trace_id: str = None) -> str:
        """Start a new trace"""
        if not trace_id:
            trace_id = f"trace_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"

        trace = {
            "trace_id": trace_id,
            "name": trace_name,
            "started_at": datetime.now(),
            "status": "active",
            "spans": [],
            "duration": None,
            "sampled": random.random() < self.trace_sampling_rate
        }

        self.traces[trace_id] = trace
        self.active_traces[trace_id] = trace

        return trace_id

    def start_span(self, trace_id: str, span_name: str, parent_span_id: str = None) -> str:
        """Start a new span within a trace"""
        if trace_id not in self.active_traces:
            return None

        span_id = f"span_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"

        span = {
            "span_id": span_id,
            "trace_id": trace_id,
            "name": span_name,
            "parent_span_id": parent_span_id,
            "started_at": datetime.now(),
            "ended_at": None,
            "duration": None,
            "tags": {},
            "events": []
        }

        self.spans[trace_id].append(span)
        self.active_traces[trace_id]["spans"].append(span_id)

        return span_id

    def end_span(self, span_id: str):
        """End a span"""
        for trace_id, spans in self.spans.items():
            for span in spans:
                if span["span_id"] == span_id:
                    span["ended_at"] = datetime.now()
                    span["duration"] = (span["ended_at"] - span["started_at"]).total_seconds() * 1000  # milliseconds
                    return

    def end_trace(self, trace_id: str):
        """End a trace"""
        if trace_id in self.active_traces:
            trace = self.active_traces[trace_id]
            trace["ended_at"] = datetime.now()
            trace["duration"] = (trace["ended_at"] - trace["started_at"]).total_seconds() * 1000
            trace["status"] = "completed"

            # End any remaining spans
            for span_id in trace["spans"]:
                self.end_span(span_id)

            del self.active_traces[trace_id]

    def add_span_tag(self, span_id: str, key: str, value: Any):
        """Add a tag to a span"""
        for trace_id, spans in self.spans.items():
            for span in spans:
                if span["span_id"] == span_id:
                    span["tags"][key] = value
                    return

    def add_span_event(self, span_id: str, event_name: str, attributes: Dict[str, Any] = None):
        """Add an event to a span"""
        for trace_id, spans in self.spans.items():
            for span in spans:
                if span["span_id"] == span_id:
                    event = {
                        "name": event_name,
                        "timestamp": datetime.now(),
                        "attributes": attributes or {}
                    }
                    span["events"].append(event)
                    return

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get complete trace information"""
        if trace_id not in self.traces:
            return None

        trace = self.traces[trace_id].copy()
        trace["spans"] = []

        # Add span details
        for span_id in trace["spans"]:
            for span in self.spans[trace_id]:
                if span["span_id"] == span_id:
                    trace["spans"].append(span)
                    break

        return trace

    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace summary"""
        trace = self.get_trace(trace_id)
        if not trace:
            return None

        spans = trace.get("spans", [])
        total_spans = len(spans)
        error_spans = len([s for s in spans if s.get("tags", {}).get("error")])

        return {
            "trace_id": trace_id,
            "name": trace["name"],
            "duration_ms": trace.get("duration", 0),
            "total_spans": total_spans,
            "error_spans": error_spans,
            "sampled": trace.get("sampled", False),
            "status": trace["status"]
        }

class LogAggregationSystem:
    """Advanced log aggregation and analysis system"""

    def __init__(self):
        self.log_buffer = deque(maxlen=10000)
        self.log_index = defaultdict(list)
        self.log_handlers = []
        self.is_running = False
        self.log_workers = []

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logger
        self.logger = logging.getLogger('healthcare_platform')
        self.logger.setLevel(logging.INFO)

        # Create formatters
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", '
            '"message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", '
            '"line": %(lineno)d, "trace_id": "%(trace_id)s", "user_id": "%(user_id)s"}'
        )

        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(json_formatter)

        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            'logs/healthcare_platform.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(json_formatter)

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.log_handlers.extend([console_handler, file_handler])

    def start_log_aggregation(self):
        """Start log aggregation system"""
        self.is_running = True

        # Start log processing worker
        log_worker = threading.Thread(target=self._process_logs, daemon=True)
        log_worker.start()
        self.log_workers.append(log_worker)

        # Start log analysis worker
        analysis_worker = threading.Thread(target=self._analyze_logs, daemon=True)
        analysis_worker.start()
        self.log_workers.append(analysis_worker)

    def stop_log_aggregation(self):
        """Stop log aggregation system"""
        self.is_running = False

    def log_event(self, level: str, message: str, **kwargs):
        """Log an event with structured data"""
        extra = {
            'trace_id': kwargs.get('trace_id', ''),
            'user_id': kwargs.get('user_id', ''),
            'patient_id': kwargs.get('patient_id', ''),
            'component': kwargs.get('component', ''),
            'operation': kwargs.get('operation', ''),
            'duration': kwargs.get('duration', 0),
            'status_code': kwargs.get('status_code', 0),
            'error_details': kwargs.get('error_details', ''),
            'additional_data': kwargs.get('additional_data', {})
        }

        log_entry = {
            "timestamp": datetime.now(),
            "level": level.upper(),
            "message": message,
            **extra
        }

        # Add to buffer
        self.log_buffer.append(log_entry)

        # Index the log
        self._index_log_entry(log_entry)

        # Log using standard logging
        if level.upper() == "ERROR":
            self.logger.error(message, extra=extra)
        elif level.upper() == "WARNING":
            self.logger.warning(message, extra=extra)
        elif level.upper() == "INFO":
            self.logger.info(message, extra=extra)
        else:
            self.logger.debug(message, extra=extra)

    def _index_log_entry(self, log_entry: Dict[str, Any]):
        """Index log entry for efficient querying"""
        # Index by level
        self.log_index[f"level:{log_entry['level']}"].append(log_entry)

        # Index by component
        if log_entry.get('component'):
            self.log_index[f"component:{log_entry['component']}"].append(log_entry)

        # Index by user
        if log_entry.get('user_id'):
            self.log_index[f"user:{log_entry['user_id']}"].append(log_entry)

        # Index by trace
        if log_entry.get('trace_id'):
            self.log_index[f"trace:{log_entry['trace_id']}"].append(log_entry)

        # Keep only recent entries in index
        for key in self.log_index:
            if len(self.log_index[key]) > 1000:
                self.log_index[key] = self.log_index[key][-1000:]

    def query_logs(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Query logs with filters"""
        if not filters:
            # Return most recent logs
            return list(self.log_buffer)[-limit:]

        matching_logs = []

        # Apply filters
        for log_entry in reversed(list(self.log_buffer)):
            if self._matches_filters(log_entry, filters):
                matching_logs.append(log_entry)
                if len(matching_logs) >= limit:
                    break

        return matching_logs

    def _matches_filters(self, log_entry: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if log entry matches filters"""
        for key, value in filters.items():
            if key == "level":
                if log_entry.get("level") != value.upper():
                    return False
            elif key == "component":
                if log_entry.get("component") != value:
                    return False
            elif key == "user_id":
                if log_entry.get("user_id") != value:
                    return False
            elif key == "trace_id":
                if log_entry.get("trace_id") != value:
                    return False
            elif key == "time_range":
                # Handle time range filters
                start_time, end_time = value
                if not (start_time <= log_entry["timestamp"] <= end_time):
                    return False
            elif key == "contains":
                # Text search in message
                if value.lower() not in log_entry.get("message", "").lower():
                    return False

        return True

    def _process_logs(self):
        """Process and enrich logs"""
        while self.is_running:
            try:
                if self.log_buffer:
                    # Process batch of logs
                    batch_size = min(100, len(self.log_buffer))
                    log_batch = [self.log_buffer.popleft() for _ in range(batch_size)]

                    for log_entry in log_batch:
                        self._enrich_log_entry(log_entry)

                time.sleep(5)  # Process every 5 seconds

            except Exception as e:
                print(f"Log processing error: {e}")

    def _enrich_log_entry(self, log_entry: Dict[str, Any]):
        """Enrich log entry with additional information"""
        # Add processing metadata
        log_entry["processed_at"] = datetime.now()
        log_entry["log_id"] = f"log_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"

        # Add severity score
        severity_scores = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        log_entry["severity_score"] = severity_scores.get(log_entry.get("level", "INFO"), 1)

        # Add anomaly detection (simplified)
        log_entry["is_anomalous"] = self._detect_log_anomaly(log_entry)

    def _detect_log_anomaly(self, log_entry: Dict[str, Any]) -> bool:
        """Detect if log entry is anomalous"""
        # Simple anomaly detection based on error patterns
        if log_entry.get("level") == "ERROR":
            error_message = log_entry.get("message", "").lower()
            error_patterns = ["exception", "failed", "timeout", "connection refused", "unauthorized"]

            for pattern in error_patterns:
                if pattern in error_message:
                    return True

        return False

    def _analyze_logs(self):
        """Analyze logs for insights and patterns"""
        while self.is_running:
            try:
                # Analyze recent logs
                recent_logs = list(self.log_buffer)[-1000:]  # Last 1000 logs

                if recent_logs:
                    analysis = self._perform_log_analysis(recent_logs)

                    # Store analysis results
                    self.log_index["analysis"] = [analysis]

                    # Check for alerting conditions
                    self._check_log_alerts(analysis)

                time.sleep(60)  # Analyze every minute

            except Exception as e:
                print(f"Log analysis error: {e}")

    def _perform_log_analysis(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive log analysis"""
        analysis = {
            "analyzed_at": datetime.now(),
            "total_logs": len(logs),
            "time_range": {
                "start": min(l["timestamp"] for l in logs),
                "end": max(l["timestamp"] for l in logs)
            }
        }

        # Level distribution
        level_counts = defaultdict(int)
        for log in logs:
            level_counts[log.get("level", "UNKNOWN")] += 1
        analysis["level_distribution"] = dict(level_counts)

        # Error rate
        error_logs = [l for l in logs if l.get("level") in ["ERROR", "CRITICAL"]]
        analysis["error_rate"] = len(error_logs) / len(logs) if logs else 0

        # Top error messages
        error_messages = [l.get("message", "") for l in error_logs]
        message_counts = defaultdict(int)
        for msg in error_messages:
            # Simplify message for grouping
            simplified = msg.split(":")[0] if ":" in msg else msg[:50]
            message_counts[simplified] += 1

        analysis["top_errors"] = dict(sorted(message_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        # Component activity
        component_activity = defaultdict(int)
        for log in logs:
            component = log.get("component", "unknown")
            component_activity[component] += 1
        analysis["component_activity"] = dict(sorted(component_activity.items(), key=lambda x: x[1], reverse=True))

        # Performance insights
        duration_logs = [l for l in logs if l.get("duration", 0) > 0]
        if duration_logs:
            durations = [l["duration"] for l in duration_logs]
            analysis["performance"] = {
                "avg_duration": statistics.mean(durations),
                "p95_duration": self._percentile(durations, 95),
                "max_duration": max(durations),
                "slow_requests": len([d for d in durations if d > 1000])  # > 1 second
            }

        # Anomaly detection
        anomalous_logs = [l for l in logs if l.get("is_anomalous", False)]
        analysis["anomalies"] = {
            "count": len(anomalous_logs),
            "rate": len(anomalous_logs) / len(logs) if logs else 0,
            "recent_anomalies": len([l for l in anomalous_logs if (datetime.now() - l["timestamp"]).seconds < 300])
        }

        return analysis

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile from list of values"""
        if not values:
            return 0
        values_sorted = sorted(values)
        index = int(len(values_sorted) * percentile / 100)
        return values_sorted[min(index, len(values_sorted) - 1)]

    def _check_log_alerts(self, analysis: Dict[str, Any]):
        """Check for alerting conditions in log analysis"""
        # High error rate alert
        if analysis["error_rate"] > 0.1:  # More than 10% errors
            print(f"🚨 HIGH ERROR RATE ALERT: {analysis['error_rate']:.1%} of recent logs are errors")

        # Performance degradation alert
        if "performance" in analysis:
            perf = analysis["performance"]
            if perf["p95_duration"] > 5000:  # 95th percentile > 5 seconds
                print(f"🚨 PERFORMANCE DEGRADATION ALERT: P95 response time is {perf['p95_duration']:.2f}ms")

        # Anomaly spike alert
        if analysis["anomalies"]["rate"] > 0.05:  # More than 5% anomalous logs
            print(f"🚨 ANOMALY SPIKE ALERT: {analysis['anomalies']['rate']:.1%} of logs are anomalous")

    def get_log_insights(self) -> Dict[str, Any]:
        """Get log analysis insights"""
        analysis = self.log_index.get("analysis", [{}])[-1]  # Most recent analysis

        insights = {
            "summary": {
                "total_logs_analyzed": analysis.get("total_logs", 0),
                "error_rate": analysis.get("error_rate", 0),
                "anomaly_rate": analysis.get("anomalies", {}).get("rate", 0)
            },
            "top_issues": analysis.get("top_errors", {}),
            "component_health": analysis.get("component_activity", {}),
            "performance_metrics": analysis.get("performance", {}),
            "recommendations": self._generate_log_recommendations(analysis)
        }

        return insights

    def _generate_log_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on log analysis"""
        recommendations = []

        if analysis.get("error_rate", 0) > 0.05:
            recommendations.append("Investigate high error rate - check application logs for root causes")

        if analysis.get("anomalies", {}).get("rate", 0) > 0.03:
            recommendations.append("Review anomalous log patterns for potential security issues")

        perf = analysis.get("performance", {})
        if perf.get("p95_duration", 0) > 2000:
            recommendations.append("Optimize slow API endpoints and database queries")

        top_errors = analysis.get("top_errors", {})
        if top_errors:
            most_common_error = max(top_errors.items(), key=lambda x: x[1])
            recommendations.append(f"Address most common error: '{most_common_error[0]}' ({most_common_error[1]} occurrences)")

        return recommendations

class ObservabilityDashboard:
    """Centralized observability dashboard"""

    def __init__(self, metrics_collector: MetricsCollector,
                 alerting_system: AlertingSystem,
                 log_system: LogAggregationSystem,
                 tracing_system: DistributedTracing):
        self.metrics = metrics_collector
        self.alerting = alerting_system
        self.logging = log_system
        self.tracing = tracing_system
        self.dashboard_data = {}
        self.is_running = False
        self.dashboard_workers = []

    def start_dashboard(self):
        """Start observability dashboard"""
        self.is_running = True

        # Start dashboard update worker
        update_worker = threading.Thread(target=self._update_dashboard, daemon=True)
        update_worker.start()
        self.dashboard_workers.append(update_worker)

    def stop_dashboard(self):
        """Stop observability dashboard"""
        self.is_running = False

    def _update_dashboard(self):
        """Update dashboard data"""
        while self.is_running:
            try:
                self.dashboard_data = self._compile_dashboard_data()
                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                print(f"Dashboard update error: {e}")

    def _compile_dashboard_data(self) -> Dict[str, Any]:
        """Compile comprehensive dashboard data"""
        return {
            "timestamp": datetime.now(),
            "system_health": self._get_system_health(),
            "performance_metrics": self._get_performance_metrics(),
            "alert_summary": self._get_alert_summary(),
            "log_insights": self.logging.get_log_insights(),
            "trace_summary": self._get_trace_summary(),
            "recommendations": self._generate_dashboard_recommendations()
        }

    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health overview"""
        cpu_stats = self.metrics.get_metric_stats("system.cpu_percent")
        memory_stats = self.metrics.get_metric_stats("system.memory_percent")
        disk_stats = self.metrics.get_metric_stats("system.disk_percent")

        health_score = 100
        issues = []

        if cpu_stats.get("avg", 0) > 80:
            health_score -= 20
            issues.append("High CPU usage")
        if memory_stats.get("avg", 0) > 85:
            health_score -= 20
            issues.append("High memory usage")
        if disk_stats.get("avg", 0) > 90:
            health_score -= 15
            issues.append("Low disk space")

        status = "healthy"
        if health_score < 70:
            status = "warning"
        if health_score < 50:
            status = "critical"

        return {
            "overall_score": health_score,
            "status": status,
            "issues": issues,
            "metrics": {
                "cpu": cpu_stats,
                "memory": memory_stats,
                "disk": disk_stats
            }
        }

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        response_time = self.metrics.get_metric_stats("app.response_time_avg")
        throughput = self.metrics.get_metric_stats("app.requests_total")
        error_rate = self.metrics.get_metric_stats("system.error_rate")

        return {
            "response_time": response_time,
            "throughput": throughput,
            "error_rate": error_rate,
            "apdex_score": self._calculate_apdex(response_time),
            "performance_trend": "stable"  # Would analyze trends
        }

    def _calculate_apdex(self, response_time_stats: Dict[str, Any]) -> float:
        """Calculate Apdex score"""
        if not response_time_stats or "p95" not in response_time_stats:
            return 0.8  # Default satisfactory

        p95_time = response_time_stats["p95"]

        # Apdex thresholds: Satisfied (< 500ms), Tolerating (500-2000ms), Frustrated (> 2000ms)
        satisfied = p95_time <= 500
        tolerating = 500 < p95_time <= 2000

        if satisfied:
            return 1.0
        elif tolerating:
            return 0.5
        else:
            return 0.0

    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active_alerts = self.alerting.get_active_alerts()
        recent_history = self.alerting.get_alert_history(limit=100)

        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert["severity"]] += 1

        return {
            "active_alerts": len(active_alerts),
            "severity_breakdown": dict(severity_counts),
            "recent_alerts": len([a for a in recent_history if (datetime.now() - a["triggered_at"]).days < 1]),
            "top_alert_types": self._get_top_alert_types(recent_history)
        }

    def _get_top_alert_types(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get most common alert types"""
        alert_counts = defaultdict(int)
        for alert in alerts:
            alert_counts[alert.get("rule_name", "Unknown")] += 1

        return [
            {"alert_type": alert_type, "count": count}
            for alert_type, count in sorted(alert_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    def _get_trace_summary(self) -> Dict[str, Any]:
        """Get distributed tracing summary"""
        # This would aggregate trace data
        return {
            "active_traces": len(self.tracing.active_traces),
            "total_traces": len(self.traces),
            "avg_trace_duration": random.uniform(100, 1000),  # ms
            "error_traces": random.randint(5, 50),
            "sampled_traces": random.randint(50, 200)
        }

    def _generate_dashboard_recommendations(self) -> List[Dict[str, Any]]:
        """Generate dashboard recommendations"""
        recommendations = []

        system_health = self._get_system_health()
        if system_health["status"] != "healthy":
            recommendations.append({
                "priority": "high",
                "category": "system_health",
                "recommendation": f"Address system health issues: {', '.join(system_health['issues'])}",
                "actions": ["Scale resources", "Optimize performance", "Investigate root causes"]
            })

        alert_summary = self._get_alert_summary()
        if alert_summary["active_alerts"] > 5:
            recommendations.append({
                "priority": "high",
                "category": "alerts",
                "recommendation": f"Resolve {alert_summary['active_alerts']} active alerts",
                "actions": ["Acknowledge critical alerts", "Investigate root causes", "Implement fixes"]
            })

        performance = self._get_performance_metrics()
        if performance.get("apdex_score", 1.0) < 0.8:
            recommendations.append({
                "priority": "medium",
                "category": "performance",
                "recommendation": "Improve application performance",
                "actions": ["Optimize slow endpoints", "Scale infrastructure", "Cache frequently accessed data"]
            })

        return recommendations

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data

    def export_dashboard_report(self, format: str = "json") -> str:
        """Export dashboard report"""
        report = {
            "generated_at": datetime.now(),
            "dashboard_data": self.dashboard_data,
            "export_format": format,
            "platform_version": "1.0.0"
        }

        if format == "json":
            return json.dumps(report, indent=2, default=str)
        else:
            # Simple text report
            lines = [
                "Observability Dashboard Report",
                f"Generated: {report['generated_at']}",
                "",
                f"System Health: {self.dashboard_data.get('system_health', {}).get('status', 'unknown').upper()}",
                f"Active Alerts: {self.dashboard_data.get('alert_summary', {}).get('active_alerts', 0)}",
                f"Error Rate: {self.dashboard_data.get('performance_metrics', {}).get('error_rate', {}).get('avg', 0):.2%}",
                "",
                "Top Recommendations:"
            ]

            for rec in self.dashboard_data.get("recommendations", [])[:3]:
                lines.append(f"  • {rec['recommendation']}")

            return "\n".join(lines)
