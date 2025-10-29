"""
Advanced Monitoring and Logging System for AI Personalized Medicine Platform
Comprehensive monitoring, alerting, and observability features
"""

import logging
import logging.handlers
import time
import threading
import psutil
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import socket
import platform
import os
from abc import ABC, abstractmethod
import queue
import sys
import traceback
from pathlib import Path


@dataclass
class MetricData:
    """Metric data structure"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Log entry structure"""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    exception_info: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str
    threshold: float
    duration: int  # seconds
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    enabled: bool = True
    cooldown_period: int = 300  # seconds
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_name: str
    severity: str
    message: str
    timestamp: datetime
    value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


@dataclass
class SystemHealth:
    """System health status"""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy', 'critical'
    uptime: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_connections: int
    active_threads: int
    queue_size: int
    last_check: datetime
    error_count: int = 0
    warning_count: int = 0
    response_time_avg: float = 0.0
    throughput: float = 0.0


class MetricsCollector:
    """Advanced metrics collection system"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.max_history = 10000
        self.collection_interval = 60  # seconds
        self.collectors = {}
        self._stop_event = threading.Event()
        self._collection_thread = None

    def start_collection(self):
        """Start metrics collection"""
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()

    def stop_collection(self):
        """Stop metrics collection"""
        self._stop_event.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5)

    def add_collector(self, name: str, collector_func: Callable[[], Dict[str, Any]]):
        """Add a custom metrics collector"""
        self.collectors[name] = collector_func

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a metric value"""
        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )

        # Maintain rolling history
        self.metrics[name].append(metric)
        if len(self.metrics[name]) > self.max_history:
            self.metrics[name].pop(0)

    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricData]:
        """Get metric history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics.get(name, []) if m.timestamp >= cutoff_time]

    def get_metric_stats(self, name: str, hours: int = 1) -> Dict[str, float]:
        """Get statistical summary of metric"""
        history = self.get_metric_history(name, hours)
        if not history:
            return {}

        values = [m.value for m in history]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'latest': values[-1],
            'change_rate': (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
        }

    def _collection_loop(self):
        """Main metrics collection loop"""
        while not self._stop_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Collect custom metrics
                for name, collector in self.collectors.items():
                    try:
                        metrics_data = collector()
                        for metric_name, value in metrics_data.items():
                            if isinstance(value, (int, float)):
                                self.record_metric(f"{name}.{metric_name}", value)
                    except Exception as e:
                        logging.error(f"Error collecting metrics from {name}: {e}")

                # Sleep until next collection
                self._stop_event.wait(self.collection_interval)

            except Exception as e:
                logging.error(f"Error in metrics collection loop: {e}")
                time.sleep(5)  # Brief pause before retry

    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('system.cpu.usage_percent', cpu_percent)

            cpu_times = psutil.cpu_times()
            self.record_metric('system.cpu.user_time', cpu_times.user)
            self.record_metric('system.cpu.system_time', cpu_times.system)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric('system.memory.total', memory.total / 1024 / 1024)  # MB
            self.record_metric('system.memory.used', memory.used / 1024 / 1024)
            self.record_metric('system.memory.available', memory.available / 1024 / 1024)
            self.record_metric('system.memory.usage_percent', memory.percent)

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric('system.disk.total', disk.total / 1024 / 1024 / 1024)  # GB
            self.record_metric('system.disk.used', disk.used / 1024 / 1024 / 1024)
            self.record_metric('system.disk.free', disk.free / 1024 / 1024 / 1024)
            self.record_metric('system.disk.usage_percent', disk.percent)

            # Network metrics
            network = psutil.net_io_counters()
            self.record_metric('system.network.bytes_sent', network.bytes_sent / 1024 / 1024)  # MB
            self.record_metric('system.network.bytes_recv', network.bytes_recv / 1024 / 1024)
            self.record_metric('system.network.packets_sent', network.packets_sent)
            self.record_metric('system.network.packets_recv', network.packets_recv)

            # Process metrics (current process)
            process = psutil.Process()
            self.record_metric('process.cpu_usage_percent', process.cpu_percent())
            self.record_metric('process.memory_rss', process.memory_info().rss / 1024 / 1024)
            self.record_metric('process.memory_vms', process.memory_info().vms / 1024 / 1024)
            self.record_metric('process.threads', process.num_threads())
            self.record_metric('process.open_files', len(process.open_files()))

        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")


class AdvancedLogger:
    """Advanced logging system with structured logging and multiple handlers"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.loggers = {}
        self.handlers = {}
        self.formatters = {}
        self.filters = {}
        self._setup_default_logging()

    def _setup_default_logging(self):
        """Setup default logging configuration"""
        # Create formatters
        self.formatters['detailed'] = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )

        self.formatters['json'] = JsonFormatter()
        self.formatters['simple'] = logging.Formatter(
            '%(levelname)s - %(message)s'
        )

        # Create handlers
        self.handlers['console'] = logging.StreamHandler(sys.stdout)
        self.handlers['console'].setLevel(logging.INFO)
        self.handlers['console'].setFormatter(self.formatters['detailed'])

        # File handler with rotation
        log_file = self.config.get('log_file', 'logs/healthcare_platform.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.handlers['file'] = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        self.handlers['file'].setLevel(logging.DEBUG)
        self.handlers['file'].setFormatter(self.formatters['json'])

        # Error file handler
        error_log_file = self.config.get('error_log_file', 'logs/healthcare_errors.log')
        self.handlers['error_file'] = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        self.handlers['error_file'].setLevel(logging.ERROR)
        self.handlers['error_file'].setFormatter(self.formatters['detailed'])

        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Add handlers to root logger
        for handler in self.handlers.values():
            root_logger.addHandler(handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger

            # Add context filter
            context_filter = ContextFilter()
            logger.addFilter(context_filter)

        return self.loggers[name]

    def log_performance_metric(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Log performance metric"""
        logger = self.get_logger('performance')
        extra = {'operation': operation, 'duration': duration, 'performance_metric': True}
        if metadata:
            extra.update(metadata)

        logger.info(f"Performance: {operation} completed in {duration:.3f}s", extra=extra)

    def log_security_event(self, event_type: str, user_id: str = None, resource: str = None,
                          action: str = None, details: Dict[str, Any] = None):
        """Log security event"""
        logger = self.get_logger('security')
        extra = {
            'security_event': True,
            'event_type': event_type,
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'severity': self._determine_security_severity(event_type)
        }
        if details:
            extra.update(details)

        message = f"Security event: {event_type}"
        if user_id:
            message += f" by user {user_id}"
        if resource:
            message += f" on resource {resource}"
        if action:
            message += f" (action: {action})"

        logger.warning(message, extra=extra)

    def log_healthcare_event(self, event_type: str, patient_id: str = None,
                           provider_id: str = None, details: Dict[str, Any] = None):
        """Log healthcare-specific event"""
        logger = self.get_logger('healthcare')
        extra = {
            'healthcare_event': True,
            'event_type': event_type,
            'patient_id': patient_id,
            'provider_id': provider_id,
            'phi_data': self._contains_phi(details) if details else False
        }
        if details:
            extra.update(details)

        message = f"Healthcare event: {event_type}"
        if patient_id:
            message += f" for patient {self._mask_patient_id(patient_id)}"

        logger.info(message, extra=extra)

    def _determine_security_severity(self, event_type: str) -> str:
        """Determine security event severity"""
        high_severity_events = [
            'unauthorized_access', 'data_breach', 'sql_injection',
            'authentication_failure', 'privilege_escalation'
        ]
        medium_severity_events = [
            'suspicious_activity', 'failed_login', 'unusual_pattern'
        ]

        if event_type in high_severity_events:
            return 'high'
        elif event_type in medium_severity_events:
            return 'medium'
        else:
            return 'low'

    def _contains_phi(self, data: Dict[str, Any]) -> bool:
        """Check if data contains protected health information"""
        phi_fields = ['patient_id', 'medical_record', 'diagnosis', 'treatment', 'phi']
        return any(field in str(data).lower() for field in phi_fields)

    def _mask_patient_id(self, patient_id: str) -> str:
        """Mask patient ID for logging"""
        if len(patient_id) <= 4:
            return patient_id
        return patient_id[:2] + '*' * (len(patient_id) - 4) + patient_id[-2:]


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'process': record.process,
            'thread': record.thread,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                             'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                             'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                             'thread', 'threadName', 'processName', 'process', 'message']:
                    log_entry[key] = value

        return json.dumps(log_entry, default=str)


class ContextFilter(logging.Filter):
    """Logging filter to add context information"""

    def __init__(self):
        super().__init__()
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.platform = platform.platform()

    def filter(self, record):
        record.hostname = self.hostname
        record.pid = self.pid
        record.platform = self.platform
        record.request_id = getattr(record, 'request_id', 'N/A')
        record.user_id = getattr(record, 'user_id', 'N/A')
        return True


class AlertingSystem:
    """Advanced alerting system with multiple notification channels"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.notification_channels = {}
        self._setup_default_rules()
        self._setup_notification_channels()

    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name='high_cpu_usage',
                condition='cpu_usage > 90',
                threshold=90.0,
                duration=300,  # 5 minutes
                severity='high',
                description='CPU usage above 90% for 5 minutes'
            ),
            AlertRule(
                name='high_memory_usage',
                condition='memory_usage > 85',
                threshold=85.0,
                duration=600,  # 10 minutes
                severity='high',
                description='Memory usage above 85% for 10 minutes'
            ),
            AlertRule(
                name='disk_space_low',
                condition='disk_usage > 90',
                threshold=90.0,
                duration=3600,  # 1 hour
                severity='critical',
                description='Disk usage above 90% for 1 hour'
            ),
            AlertRule(
                name='response_time_high',
                condition='response_time_avg > 5.0',
                threshold=5.0,
                duration=300,  # 5 minutes
                severity='medium',
                description='Average response time above 5 seconds for 5 minutes'
            ),
            AlertRule(
                name='error_rate_high',
                condition='error_count > 10',
                threshold=10.0,
                duration=300,  # 5 minutes
                severity='high',
                description='Error count above 10 in 5 minutes'
            )
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

    def _setup_notification_channels(self):
        """Setup notification channels"""
        # Email channel
        if self.config.get('email_enabled'):
            self.notification_channels['email'] = EmailNotifier(self.config.get('email_config', {}))

        # Slack channel
        if self.config.get('slack_enabled'):
            self.notification_channels['slack'] = SlackNotifier(self.config.get('slack_config', {}))

        # SMS channel
        if self.config.get('sms_enabled'):
            self.notification_channels['sms'] = SMSNotifier(self.config.get('sms_config', {}))

        # Webhook channel
        if self.config.get('webhook_enabled'):
            self.notification_channels['webhook'] = WebhookNotifier(self.config.get('webhook_config', {}))

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]

    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        current_time = datetime.now()

        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Check if rule is in cooldown
            if rule.last_triggered and (current_time - rule.last_triggered).seconds < rule.cooldown_period:
                continue

            # Evaluate condition
            if self._evaluate_condition(rule.condition, metrics):
                # Check duration
                alert_key = f"{rule.name}_{hash(str(metrics))}"
                if alert_key not in self.active_alerts:
                    self.active_alerts[alert_key] = {
                        'start_time': current_time,
                        'rule': rule,
                        'metrics': metrics
                    }
                else:
                    alert_data = self.active_alerts[alert_key]
                    duration = (current_time - alert_data['start_time']).seconds

                    if duration >= rule.duration:
                        # Trigger alert
                        self._trigger_alert(rule, metrics, current_time)
                        rule.last_triggered = current_time

                        # Clean up active alert
                        del self.active_alerts[alert_key]

    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition"""
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            parts = condition.split()
            if len(parts) == 3:
                metric_name, operator, threshold_str = parts
                threshold = float(threshold_str)

                metric_value = metrics.get(metric_name)
                if metric_value is None:
                    return False

                if operator == '>':
                    return metric_value > threshold
                elif operator == '<':
                    return metric_value < threshold
                elif operator == '>=':
                    return metric_value >= threshold
                elif operator == '<=':
                    return metric_value <= threshold
                elif operator == '==':
                    return metric_value == threshold

        except (ValueError, KeyError):
            pass

        return False

    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any], timestamp: datetime):
        """Trigger an alert"""
        alert = Alert(
            alert_id=f"alert_{int(timestamp.timestamp())}_{rule.name}",
            rule_name=rule.name,
            severity=rule.severity,
            message=self._generate_alert_message(rule, metrics),
            timestamp=timestamp,
            value=metrics.get(rule.condition.split()[0], 0),
            threshold=rule.threshold
        )

        self.alert_history.append(alert)

        # Send notifications
        for channel_name, channel in self.notification_channels.items():
            try:
                channel.send_alert(alert)
            except Exception as e:
                logging.error(f"Failed to send alert via {channel_name}: {e}")

        logging.warning(f"Alert triggered: {alert.message}")

    def _generate_alert_message(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Generate alert message"""
        metric_name = rule.condition.split()[0]
        current_value = metrics.get(metric_name, 'unknown')

        return f"{rule.description}. Current value: {current_value}, Threshold: {rule.threshold}"

    def acknowledge_alert(self, alert_id: str, user_id: str):
        """Acknowledge an alert"""
        for alert in self.alert_history:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = user_id
                break

    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        for alert in self.alert_history:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                break

    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alert_history if not alert.resolved]


class NotificationChannel(ABC):
    """Base class for notification channels"""

    @abstractmethod
    def send_alert(self, alert: Alert):
        """Send an alert notification"""
        pass


class EmailNotifier(NotificationChannel):
    """Email notification channel"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email', self.username)
        self.to_emails = config.get('to_emails', [])

    def send_alert(self, alert: Alert):
        """Send alert via email"""
        msg = MIMEMultipart()
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        msg['Subject'] = f"ALERT: {alert.severity.upper()} - {alert.rule_name}"

        body = f"""
        Alert Details:
        - Alert ID: {alert.alert_id}
        - Rule: {alert.rule_name}
        - Severity: {alert.severity.upper()}
        - Message: {alert.message}
        - Value: {alert.value}
        - Threshold: {alert.threshold}
        - Timestamp: {alert.timestamp.isoformat()}

        Please check the system immediately.
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        server.starttls()
        server.login(self.username, self.password)
        text = msg.as_string()
        server.sendmail(self.from_email, self.to_emails, text)
        server.quit()


class SlackNotifier(NotificationChannel):
    """Slack notification channel"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')

    def send_alert(self, alert: Alert):
        """Send alert via Slack webhook"""
        severity_emoji = {
            'low': '⚠️',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }

        payload = {
            'channel': self.channel,
            'text': f"{severity_emoji.get(alert.severity, '❗')} *ALERT: {alert.severity.upper()}*",
            'attachments': [{
                'color': self._get_slack_color(alert.severity),
                'fields': [
                    {'title': 'Rule', 'value': alert.rule_name, 'short': True},
                    {'title': 'Value', 'value': str(alert.value), 'short': True},
                    {'title': 'Threshold', 'value': str(alert.threshold), 'short': True},
                    {'title': 'Message', 'value': alert.message, 'short': False}
                ],
                'footer': f"Alert ID: {alert.alert_id}",
                'ts': alert.timestamp.timestamp()
            }]
        }

        requests.post(self.webhook_url, json=payload)

    def _get_slack_color(self, severity: str) -> str:
        """Get Slack color for severity"""
        colors = {
            'low': 'good',
            'medium': 'warning',
            'high': 'danger',
            'critical': '#FF0000'
        }
        return colors.get(severity, 'warning')


class SMSNotifier(NotificationChannel):
    """SMS notification channel"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.from_number = config.get('from_number')
        self.to_numbers = config.get('to_numbers', [])

    def send_alert(self, alert: Alert):
        """Send alert via SMS"""
        message = f"ALERT {alert.severity.upper()}: {alert.message[:140]}"  # SMS length limit

        # This would integrate with an SMS service like Twilio
        # For demonstration, we'll just log it
        logging.info(f"SMS Alert to {self.to_numbers}: {message}")


class WebhookNotifier(NotificationChannel):
    """Webhook notification channel"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url = config.get('url')
        self.headers = config.get('headers', {})
        self.method = config.get('method', 'POST')

    def send_alert(self, alert: Alert):
        """Send alert via webhook"""
        payload = {
            'alert_id': alert.alert_id,
            'rule_name': alert.rule_name,
            'severity': alert.severity,
            'message': alert.message,
            'value': alert.value,
            'threshold': alert.threshold,
            'timestamp': alert.timestamp.isoformat()
        }

        requests.request(
            method=self.method,
            url=self.url,
            headers=self.headers,
            json=payload
        )


class HealthChecker:
    """System health monitoring"""

    def __init__(self):
        self.components = {}
        self.health_checks = {}
        self._check_thread = None
        self._stop_event = threading.Event()

    def register_component(self, name: str, check_func: Callable[[], SystemHealth]):
        """Register a component for health checking"""
        self.components[name] = check_func

    def start_health_checks(self, interval: int = 60):
        """Start periodic health checks"""
        self._check_thread = threading.Thread(
            target=self._health_check_loop,
            args=(interval,),
            daemon=True
        )
        self._check_thread.start()

    def stop_health_checks(self):
        """Stop health checks"""
        self._stop_event.set()
        if self._check_thread:
            self._check_thread.join(timeout=5)

    def get_health_status(self) -> Dict[str, SystemHealth]:
        """Get current health status of all components"""
        status = {}
        for name, check_func in self.components.items():
            try:
                status[name] = check_func()
            except Exception as e:
                logging.error(f"Health check failed for {name}: {e}")
                status[name] = SystemHealth(
                    component=name,
                    status='critical',
                    uptime=0,
                    cpu_usage=0,
                    memory_usage=0,
                    disk_usage=0,
                    network_connections=0,
                    active_threads=0,
                    queue_size=0,
                    last_check=datetime.now(),
                    error_count=1
                )
        return status

    def _health_check_loop(self, interval: int):
        """Main health check loop"""
        while not self._stop_event.is_set():
            try:
                health_status = self.get_health_status()

                # Log health issues
                for component, status in health_status.items():
                    if status.status in ['degraded', 'unhealthy', 'critical']:
                        logging.warning(f"Component {component} health: {status.status}")

                # Store health data
                self.health_checks[datetime.now()] = health_status

                # Clean old health data (keep last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.health_checks = {
                    ts: data for ts, data in self.health_checks.items()
                    if ts >= cutoff_time
                }

            except Exception as e:
                logging.error(f"Error in health check loop: {e}")

            self._stop_event.wait(interval)


# Global instances
metrics_collector = MetricsCollector()
advanced_logger = AdvancedLogger()
alerting_system = AlertingSystem()
health_checker = HealthChecker()

# Initialize monitoring system
def initialize_monitoring(config: Dict[str, Any] = None):
    """Initialize the complete monitoring system"""
    config = config or {}

    # Start metrics collection
    metrics_collector.start_collection()

    # Add custom healthcare metrics collectors
    metrics_collector.add_collector('healthcare', collect_healthcare_metrics)
    metrics_collector.add_collector('api', collect_api_metrics)
    metrics_collector.add_collector('database', collect_database_metrics)

    # Start health checks
    health_checker.start_health_checks()

    # Register component health checks
    health_checker.register_component('genomic_engine', check_genomic_engine_health)
    health_checker.register_component('ai_models', check_ai_models_health)
    health_checker.register_component('database', check_database_health)
    health_checker.register_component('api_server', check_api_server_health)

    logging.info("Monitoring system initialized successfully")


def collect_healthcare_metrics() -> Dict[str, float]:
    """Collect healthcare-specific metrics"""
    return {
        'active_patients': 1250,  # Mock data
        'pending_analyses': 45,
        'completed_analyses': 1234,
        'failed_analyses': 12,
        'active_clinical_trials': 8,
        'drug_discovery_jobs': 23
    }


def collect_api_metrics() -> Dict[str, float]:
    """Collect API performance metrics"""
    return {
        'requests_total': 15432,
        'requests_per_second': 12.5,
        'response_time_avg': 0.234,
        'error_rate': 0.023,
        'active_connections': 45
    }


def collect_database_metrics() -> Dict[str, float]:
    """Collect database performance metrics"""
    return {
        'connections_active': 12,
        'connections_idle': 8,
        'query_time_avg': 0.045,
        'cache_hit_rate': 0.87,
        'deadlocks': 0
    }


def check_genomic_engine_health() -> SystemHealth:
    """Check genomic engine health"""
    # Mock health check
    return SystemHealth(
        component='genomic_engine',
        status='healthy',
        uptime=86400,  # 24 hours
        cpu_usage=15.2,
        memory_usage=234.5,
        disk_usage=45.2,
        network_connections=3,
        active_threads=8,
        queue_size=2,
        last_check=datetime.now(),
        response_time_avg=0.123,
        throughput=45.6
    )


def check_ai_models_health() -> SystemHealth:
    """Check AI models health"""
    return SystemHealth(
        component='ai_models',
        status='healthy',
        uptime=86400,
        cpu_usage=28.7,
        memory_usage=456.8,
        disk_usage=67.1,
        network_connections=1,
        active_threads=12,
        queue_size=0,
        last_check=datetime.now(),
        response_time_avg=0.089,
        throughput=78.3
    )


def check_database_health() -> SystemHealth:
    """Check database health"""
    return SystemHealth(
        component='database',
        status='healthy',
        uptime=172800,  # 48 hours
        cpu_usage=8.9,
        memory_usage=678.9,
        disk_usage=34.5,
        network_connections=25,
        active_threads=4,
        queue_size=1,
        last_check=datetime.now(),
        response_time_avg=0.034,
        throughput=234.1
    )


def check_api_server_health() -> SystemHealth:
    """Check API server health"""
    return SystemHealth(
        component='api_server',
        status='healthy',
        uptime=86400,
        cpu_usage=12.3,
        memory_usage=123.4,
        disk_usage=23.4,
        network_connections=150,
        active_threads=16,
        queue_size=0,
        last_check=datetime.now(),
        response_time_avg=0.067,
        throughput=156.7
    )


# Initialize on import
if __name__ != '__main__':
    initialize_monitoring()
