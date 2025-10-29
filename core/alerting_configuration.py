"""
Advanced Alerting Configuration and Management System
Comprehensive alert rules, escalation policies, and notification management
"""

import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import requests

class AlertRuleEngine:
    """Advanced alert rule engine with complex conditions"""

    def __init__(self):
        self.rules = {}
        self.rule_groups = defaultdict(list)
        self.rule_templates = {}
        self.rule_evaluations = defaultdict(list)
        self.is_running = False
        self.evaluation_workers = []

    def create_rule(self, rule_config: Dict[str, Any]) -> str:
        """Create a new alert rule"""
        rule_id = f"rule_{int(time.time())}_{hash(str(rule_config)) % 10000}"

        rule = {
            "rule_id": rule_id,
            "name": rule_config["name"],
            "description": rule_config.get("description", ""),
            "group": rule_config.get("group", "default"),
            "severity": rule_config.get("severity", "warning"),
            "enabled": rule_config.get("enabled", True),
            "condition": rule_config["condition"],
            "threshold": rule_config["threshold"],
            "time_window": rule_config.get("time_window", 300),  # 5 minutes default
            "evaluation_interval": rule_config.get("evaluation_interval", 60),  # 1 minute
            "cooldown_period": rule_config.get("cooldown_period", 300),  # 5 minutes
            "escalation_policy": rule_config.get("escalation_policy", "default"),
            "notification_channels": rule_config.get("notification_channels", ["log"]),
            "tags": rule_config.get("tags", []),
            "metadata": rule_config.get("metadata", {}),
            "created_at": datetime.now(),
            "last_triggered": None,
            "trigger_count": 0
        }

        self.rules[rule_id] = rule
        self.rule_groups[rule["group"]].append(rule_id)

        return rule_id

    def create_rule_template(self, template_name: str, template_config: Dict[str, Any]):
        """Create a reusable alert rule template"""
        self.rule_templates[template_name] = {
            "template_name": template_name,
            "config": template_config,
            "created_at": datetime.now(),
            "usage_count": 0
        }

    def create_rule_from_template(self, template_name: str, customizations: Dict[str, Any] = None) -> Optional[str]:
        """Create a rule from a template"""
        if template_name not in self.rule_templates:
            return None

        template = self.rule_templates[template_name]
        rule_config = template["config"].copy()

        # Apply customizations
        if customizations:
            rule_config.update(customizations)

        rule_id = self.create_rule(rule_config)
        template["usage_count"] += 1

        return rule_id

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        if rule_id not in self.rules:
            return False

        rule = self.rules[rule_id]

        # Update allowed fields
        allowed_updates = ["name", "description", "enabled", "condition", "threshold",
                         "time_window", "evaluation_interval", "cooldown_period",
                         "escalation_policy", "notification_channels", "tags", "metadata"]

        for key, value in updates.items():
            if key in allowed_updates:
                rule[key] = value

        rule["updated_at"] = datetime.now()
        return True

    def delete_rule(self, rule_id: str) -> bool:
        """Delete an alert rule"""
        if rule_id not in self.rules:
            return False

        rule = self.rules[rule_id]
        group = rule["group"]

        # Remove from group
        if rule_id in self.rule_groups[group]:
            self.rule_groups[group].remove(rule_id)

        # Delete rule
        del self.rules[rule_id]
        return True

    def start_evaluation_engine(self):
        """Start the rule evaluation engine"""
        self.is_running = True

        # Group rules by evaluation interval for efficient processing
        interval_groups = defaultdict(list)
        for rule_id, rule in self.rules.items():
            if rule["enabled"]:
                interval_groups[rule["evaluation_interval"]].append(rule_id)

        # Start evaluation workers for each interval
        for interval, rule_ids in interval_groups.items():
            worker = threading.Thread(
                target=self._evaluate_rules_worker,
                args=(rule_ids, interval),
                daemon=True
            )
            worker.start()
            self.evaluation_workers.append(worker)

    def stop_evaluation_engine(self):
        """Stop the rule evaluation engine"""
        self.is_running = False

    def _evaluate_rules_worker(self, rule_ids: List[str], interval: int):
        """Worker to evaluate rules at specified interval"""
        while self.is_running:
            try:
                for rule_id in rule_ids:
                    if rule_id in self.rules:
                        self._evaluate_rule(self.rules[rule_id])

                time.sleep(interval)

            except Exception as e:
                print(f"Rule evaluation error for interval {interval}: {e}")

    def _evaluate_rule(self, rule: Dict[str, Any]):
        """Evaluate a single alert rule"""
        rule_id = rule["rule_id"]

        # Check cooldown period
        if rule["last_triggered"]:
            cooldown_remaining = (datetime.now() - rule["last_triggered"]).total_seconds()
            if cooldown_remaining < rule["cooldown_period"]:
                return  # Still in cooldown

        # Evaluate condition
        is_triggered = self._evaluate_condition(rule)

        if is_triggered:
            # Record evaluation
            evaluation = {
                "rule_id": rule_id,
                "timestamp": datetime.now(),
                "triggered": True,
                "threshold": rule["threshold"],
                "current_value": self._get_current_value(rule),
                "condition": rule["condition"]
            }
            self.rule_evaluations[rule_id].append(evaluation)

            # Keep only recent evaluations
            if len(self.rule_evaluations[rule_id]) > 100:
                self.rule_evaluations[rule_id] = self.rule_evaluations[rule_id][-100:]

            # Trigger alert
            self._trigger_alert(rule)

            # Update rule state
            rule["last_triggered"] = datetime.now()
            rule["trigger_count"] += 1

    def _evaluate_condition(self, rule: Dict[str, Any]) -> bool:
        """Evaluate alert condition"""
        current_value = self._get_current_value(rule)
        if current_value is None:
            return False

        condition = rule["condition"]
        threshold = rule["threshold"]

        # Simple condition evaluation
        if condition == "greater_than":
            return current_value > threshold
        elif condition == "less_than":
            return current_value < threshold
        elif condition == "equal":
            return current_value == threshold
        elif condition == "not_equal":
            return current_value != threshold
        elif condition == "greater_equal":
            return current_value >= threshold
        elif condition == "less_equal":
            return current_value <= threshold
        elif condition == "percentage_change":
            # Check for percentage change over time window
            return self._check_percentage_change(rule, current_value)

        return False

    def _get_current_value(self, rule: Dict[str, Any]) -> Optional[float]:
        """Get current value for rule evaluation"""
        # This would integrate with metrics collection system
        # For simulation, return mock values based on rule metadata
        metric_type = rule.get("metadata", {}).get("metric_type", "random")

        if metric_type == "cpu":
            return 85.0  # Mock CPU usage
        elif metric_type == "memory":
            return 78.0  # Mock memory usage
        elif metric_type == "response_time":
            return 2.1  # Mock response time in seconds
        elif metric_type == "error_rate":
            return 0.03  # Mock error rate (3%)
        else:
            import random
            return random.uniform(0, 100)

    def _check_percentage_change(self, rule: Dict[str, Any], current_value: float) -> bool:
        """Check for percentage change condition"""
        time_window = rule["time_window"]
        threshold = rule["threshold"]

        # Get historical values (simplified)
        # In real implementation, this would query time-series data
        historical_values = [current_value * (1 + (i-5)/10) for i in range(10)]  # Mock historical data

        if len(historical_values) < 2:
            return False

        baseline_value = statistics.mean(historical_values[:-1])
        if baseline_value == 0:
            return False

        percentage_change = ((current_value - baseline_value) / baseline_value) * 100
        return abs(percentage_change) > threshold

    def _trigger_alert(self, rule: Dict[str, Any]):
        """Trigger an alert for the rule"""
        alert_data = {
            "rule_id": rule["rule_id"],
            "rule_name": rule["name"],
            "severity": rule["severity"],
            "description": rule["description"],
            "current_value": self._get_current_value(rule),
            "threshold": rule["threshold"],
            "condition": rule["condition"],
            "tags": rule["tags"],
            "metadata": rule["metadata"],
            "triggered_at": datetime.now()
        }

        # Send to notification channels
        for channel in rule["notification_channels"]:
            self._send_notification(channel, alert_data)

        print(f"🚨 ALERT TRIGGERED: {rule['name']} - {alert_data['description']}")

    def _send_notification(self, channel: str, alert_data: Dict[str, Any]):
        """Send notification via specified channel"""
        if channel == "log":
            print(f"ALERT: {alert_data['rule_name']} - {alert_data['description']}")
        elif channel == "email":
            self._send_email_alert(alert_data)
        elif channel == "slack":
            self._send_slack_alert(alert_data)
        elif channel == "webhook":
            self._send_webhook_alert(alert_data)

    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send email alert (mock implementation)"""
        print(f"📧 EMAIL ALERT: {alert_data['rule_name']} sent to configured recipients")

    def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send Slack alert (mock implementation)"""
        print(f"💬 SLACK ALERT: {alert_data['rule_name']} posted to configured channel")

    def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Send webhook alert (mock implementation)"""
        print(f"🔗 WEBHOOK ALERT: {alert_data['rule_name']} sent to configured endpoint")

    def get_rule_status(self, rule_id: str = None) -> Dict[str, Any]:
        """Get status of alert rules"""
        if rule_id:
            if rule_id not in self.rules:
                return {"error": f"Rule {rule_id} not found"}

            rule = self.rules[rule_id]
            recent_evaluations = self.rule_evaluations[rule_id][-10:]  # Last 10 evaluations

            return {
                "rule": rule,
                "recent_evaluations": recent_evaluations,
                "evaluation_count": len(self.rule_evaluations[rule_id]),
                "status": "active" if rule["enabled"] else "disabled"
            }
        else:
            # Return summary of all rules
            return {
                "total_rules": len(self.rules),
                "enabled_rules": len([r for r in self.rules.values() if r["enabled"]]),
                "rules_by_group": dict(self.rule_groups),
                "rules_by_severity": self._get_rules_by_severity(),
                "recently_triggered": self._get_recently_triggered_rules()
            }

    def _get_rules_by_severity(self) -> Dict[str, int]:
        """Get count of rules by severity"""
        severity_counts = defaultdict(int)
        for rule in self.rules.values():
            severity_counts[rule["severity"]] += 1
        return dict(severity_counts)

    def _get_recently_triggered_rules(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get rules triggered in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recently_triggered = []

        for rule in self.rules.values():
            if rule["last_triggered"] and rule["last_triggered"] > cutoff_time:
                recently_triggered.append({
                    "rule_id": rule["rule_id"],
                    "name": rule["name"],
                    "severity": rule["severity"],
                    "last_triggered": rule["last_triggered"],
                    "trigger_count": rule["trigger_count"]
                })

        return sorted(recently_triggered, key=lambda x: x["last_triggered"], reverse=True)

class EscalationPolicyManager:
    """Advanced escalation policy management"""

    def __init__(self):
        self.policies = {}
        self.policy_templates = {}
        self.active_escalations = {}

    def create_escalation_policy(self, policy_config: Dict[str, Any]) -> str:
        """Create a new escalation policy"""
        policy_id = f"policy_{int(time.time())}_{hash(str(policy_config)) % 10000}"

        policy = {
            "policy_id": policy_id,
            "name": policy_config["name"],
            "description": policy_config.get("description", ""),
            "escalation_levels": policy_config["escalation_levels"],
            "max_escalation_level": len(policy_config["escalation_levels"]),
            "enabled": policy_config.get("enabled", True),
            "created_at": datetime.now()
        }

        self.policies[policy_id] = policy
        return policy_id

    def create_policy_template(self, template_name: str, template_config: Dict[str, Any]):
        """Create an escalation policy template"""
        self.policy_templates[template_name] = {
            "template_name": template_name,
            "config": template_config,
            "created_at": datetime.now()
        }

    def escalate_alert(self, alert_data: Dict[str, Any], policy_id: str):
        """Escalate an alert according to policy"""
        if policy_id not in self.policies:
            print(f"Escalation policy {policy_id} not found")
            return

        policy = self.policies[policy_id]
        if not policy["enabled"]:
            return

        alert_id = alert_data.get("rule_id", str(hash(str(alert_data))))

        if alert_id not in self.active_escalations:
            self.active_escalations[alert_id] = {
                "alert_data": alert_data,
                "policy_id": policy_id,
                "current_level": 0,
                "started_at": datetime.now(),
                "last_escalation": None,
                "escalation_history": []
            }

        escalation = self.active_escalations[alert_id]

        # Check if escalation is needed
        current_level = escalation["current_level"]
        if current_level >= policy["max_escalation_level"]:
            return  # Already at max escalation

        escalation_config = policy["escalation_levels"][current_level]

        # Check escalation delay
        if escalation["last_escalation"]:
            delay_passed = (datetime.now() - escalation["last_escalation"]).total_seconds()
            if delay_passed < escalation_config.get("delay_seconds", 0):
                return  # Delay not passed yet

        # Perform escalation
        self._perform_escalation(escalation, escalation_config)

        # Update escalation state
        escalation["current_level"] += 1
        escalation["last_escalation"] = datetime.now()
        escalation["escalation_history"].append({
            "level": current_level + 1,
            "timestamp": datetime.now(),
            "action": escalation_config["action"],
            "channels": escalation_config["channels"]
        })

    def _perform_escalation(self, escalation: Dict[str, Any], escalation_config: Dict[str, Any]):
        """Perform escalation action"""
        action = escalation_config["action"]
        channels = escalation_config["channels"]
        alert_data = escalation["alert_data"]

        if action == "notify":
            for channel in channels:
                self._send_escalation_notification(channel, alert_data, escalation["current_level"] + 1)
        elif action == "page":
            self._send_page_alert(alert_data, channels)
        elif action == "create_ticket":
            self._create_support_ticket(alert_data)

        print(f"🚀 ESCALATION: Level {escalation['current_level'] + 1} - {action} via {channels}")

    def _send_escalation_notification(self, channel: str, alert_data: Dict[str, Any], level: int):
        """Send escalation notification"""
        message = f"ESCALATION LEVEL {level}: {alert_data['rule_name']} - {alert_data['description']}"

        if channel == "email":
            print(f"📧 ESCALATION EMAIL: {message}")
        elif channel == "sms":
            print(f"📱 ESCALATION SMS: {message}")
        elif channel == "phone":
            print(f"📞 ESCALATION CALL: {message}")

    def _send_page_alert(self, alert_data: Dict[str, Any], channels: List[str]):
        """Send page alert (critical notification)"""
        print(f"🚨 PAGE ALERT: {alert_data['rule_name']} - CRITICAL ISSUE")

    def _create_support_ticket(self, alert_data: Dict[str, Any]):
        """Create support ticket for alert"""
        print(f"🎫 SUPPORT TICKET: Created for {alert_data['rule_name']}")

    def resolve_escalation(self, alert_id: str):
        """Resolve an active escalation"""
        if alert_id in self.active_escalations:
            escalation = self.active_escalations[alert_id]
            escalation["resolved_at"] = datetime.now()
            escalation["resolution_time"] = (escalation["resolved_at"] - escalation["started_at"]).total_seconds()

            print(f"✅ ESCALATION RESOLVED: {alert_id}")
            del self.active_escalations[alert_id]

    def get_escalation_status(self) -> Dict[str, Any]:
        """Get escalation status"""
        return {
            "active_escalations": len(self.active_escalations),
            "policies": list(self.policies.keys()),
            "escalation_summary": self._get_escalation_summary()
        }

    def _get_escalation_summary(self) -> Dict[str, Any]:
        """Get escalation summary statistics"""
        if not self.active_escalations:
            return {"total_escalations": 0}

        levels = [e["current_level"] for e in self.active_escalations.values()]
        durations = [(datetime.now() - e["started_at"]).total_seconds()
                    for e in self.active_escalations.values() if "resolved_at" not in e]

        return {
            "total_escalations": len(self.active_escalations),
            "avg_level": statistics.mean(levels) if levels else 0,
            "max_level": max(levels) if levels else 0,
            "avg_duration_active": statistics.mean(durations) if durations else 0
        }

class NotificationManager:
    """Advanced notification management system"""

    def __init__(self):
        self.channels = {}
        self.notification_history = defaultdict(list)
        self.notification_templates = {}
        self.rate_limits = {}
        self.is_running = False
        self.notification_workers = []

    def register_channel(self, channel_config: Dict[str, Any]) -> str:
        """Register a notification channel"""
        channel_id = f"channel_{int(time.time())}_{hash(str(channel_config)) % 10000}"

        channel = {
            "channel_id": channel_id,
            "type": channel_config["type"],
            "name": channel_config["name"],
            "config": channel_config["config"],
            "enabled": channel_config.get("enabled", True),
            "rate_limit": channel_config.get("rate_limit", {"max_per_minute": 60}),
            "retry_policy": channel_config.get("retry_policy", {"max_retries": 3, "backoff_seconds": 60}),
            "created_at": datetime.now()
        }

        self.channels[channel_id] = channel
        return channel_id

    def create_notification_template(self, template_name: str, template_config: Dict[str, Any]):
        """Create a notification template"""
        self.notification_templates[template_name] = {
            "template_name": template_name,
            "subject_template": template_config["subject"],
            "body_template": template_config["body"],
            "type": template_config.get("type", "alert"),
            "created_at": datetime.now()
        }

    def send_notification(self, channel_ids: List[str], message: Dict[str, Any],
                         template_name: str = None) -> Dict[str, Any]:
        """Send notification via specified channels"""
        results = {}

        for channel_id in channel_ids:
            if channel_id not in self.channels:
                results[channel_id] = {"status": "error", "error": "Channel not found"}
                continue

            channel = self.channels[channel_id]
            if not channel["enabled"]:
                results[channel_id] = {"status": "skipped", "reason": "Channel disabled"}
                continue

            # Check rate limit
            if not self._check_rate_limit(channel_id):
                results[channel_id] = {"status": "rate_limited"}
                continue

            # Apply template if specified
            if template_name and template_name in self.notification_templates:
                message = self._apply_template(template_name, message)

            # Send notification
            result = self._send_to_channel(channel, message)
            results[channel_id] = result

            # Record in history
            self.notification_history[channel_id].append({
                "timestamp": datetime.now(),
                "message": message,
                "result": result
            })

            # Keep history limited
            if len(self.notification_history[channel_id]) > 1000:
                self.notification_history[channel_id] = self.notification_history[channel_id][-1000:]

        return results

    def _check_rate_limit(self, channel_id: str) -> bool:
        """Check if channel is within rate limits"""
        channel = self.channels[channel_id]
        rate_limit = channel["rate_limit"]
        max_per_minute = rate_limit.get("max_per_minute", 60)

        # Get recent notifications
        recent_notifications = [
            n for n in self.notification_history[channel_id]
            if (datetime.now() - n["timestamp"]).total_seconds() < 60
        ]

        return len(recent_notifications) < max_per_minute

    def _apply_template(self, template_name: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Apply notification template"""
        template = self.notification_templates[template_name]

        # Simple template application (in real implementation, use proper templating)
        subject = template["subject_template"].format(**message)
        body = template["body_template"].format(**message)

        return {
            **message,
            "subject": subject,
            "body": body
        }

    def _send_to_channel(self, channel: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to specific channel"""
        channel_type = channel["type"]
        config = channel["config"]

        try:
            if channel_type == "email":
                return self._send_email(config, message)
            elif channel_type == "slack":
                return self._send_slack(config, message)
            elif channel_type == "sms":
                return self._send_sms(config, message)
            elif channel_type == "webhook":
                return self._send_webhook(config, message)
            else:
                return {"status": "error", "error": f"Unsupported channel type: {channel_type}"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _send_email(self, config: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification"""
        # Mock implementation
        print(f"📧 EMAIL: {message.get('subject', 'Alert')} sent to {config.get('recipients', [])}")
        return {"status": "sent", "channel": "email"}

    def _send_slack(self, config: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send Slack notification"""
        # Mock implementation
        print(f"💬 SLACK: Message posted to {config.get('channel', '#alerts')}")
        return {"status": "sent", "channel": "slack"}

    def _send_sms(self, config: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send SMS notification"""
        # Mock implementation
        print(f"📱 SMS: Alert sent to {config.get('phone_numbers', [])}")
        return {"status": "sent", "channel": "sms"}

    def _send_webhook(self, config: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook notification"""
        # Mock implementation
        print(f"🔗 WEBHOOK: Payload sent to {config.get('url', 'endpoint')}")
        return {"status": "sent", "channel": "webhook"}

    def get_notification_status(self) -> Dict[str, Any]:
        """Get notification system status"""
        return {
            "channels": len(self.channels),
            "enabled_channels": len([c for c in self.channels.values() if c["enabled"]]),
            "templates": len(self.notification_templates),
            "total_notifications": sum(len(history) for history in self.notification_history.values()),
            "recent_notifications": self._get_recent_notifications()
        }

    def _get_recent_notifications(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent notifications"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent = []

        for channel_history in self.notification_history.values():
            recent.extend([
                n for n in channel_history
                if n["timestamp"] > cutoff_time
            ])

        return sorted(recent, key=lambda x: x["timestamp"], reverse=True)[:50]

class AlertAnalytics:
    """Advanced alert analytics and reporting"""

    def __init__(self):
        self.alert_metrics = defaultdict(list)
        self.alert_patterns = {}
        self.predictive_alerts = {}
        self.analytics_data = {}

    def record_alert_event(self, alert_data: Dict[str, Any]):
        """Record alert event for analytics"""
        event = {
            "timestamp": datetime.now(),
            "alert_data": alert_data,
            "severity": alert_data.get("severity", "unknown"),
            "rule_id": alert_data.get("rule_id"),
            "response_time": None,  # Would be set when alert is acknowledged
            "resolution_time": None  # Would be set when alert is resolved
        }

        self.alert_metrics["events"].append(event)

        # Keep only recent events
        if len(self.alert_metrics["events"]) > 10000:
            self.alert_metrics["events"] = self.alert_metrics["events"][-10000:]

    def generate_alert_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive alert analytics report"""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        recent_events = [e for e in self.alert_metrics["events"] if e["timestamp"] > cutoff_time]

        return {
            "time_range_hours": time_range_hours,
            "total_alerts": len(recent_events),
            "alerts_by_severity": self._group_by_severity(recent_events),
            "alerts_by_rule": self._group_by_rule(recent_events),
            "alert_patterns": self._analyze_alert_patterns(recent_events),
            "performance_metrics": self._calculate_alert_performance(recent_events),
            "trends": self._analyze_alert_trends(recent_events),
            "recommendations": self._generate_alert_recommendations(recent_events)
        }

    def _group_by_severity(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group alerts by severity"""
        severity_counts = defaultdict(int)
        for event in events:
            severity_counts[event["severity"]] += 1
        return dict(severity_counts)

    def _group_by_rule(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group alerts by rule"""
        rule_counts = defaultdict(int)
        for event in events:
            rule_id = event.get("rule_id", "unknown")
            rule_counts[rule_id] += 1
        return dict(rule_counts)

    def _analyze_alert_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze alert patterns"""
        if len(events) < 10:
            return {"insufficient_data": True}

        # Time-based patterns
        hourly_distribution = defaultdict(int)
        for event in events:
            hour = event["timestamp"].hour
            hourly_distribution[hour] += 1

        # Severity progression
        severity_progression = []
        current_events = sorted(events, key=lambda x: x["timestamp"])
        for i in range(1, len(current_events)):
            prev_severity = current_events[i-1]["severity"]
            curr_severity = current_events[i]["severity"]
            severity_progression.append(f"{prev_severity}->{curr_severity}")

        return {
            "peak_hours": sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)[:3],
            "common_transitions": self._most_common(severity_progression),
            "alert_frequency": len(events) / 24  # alerts per hour
        }

    def _most_common(self, items: List[str], top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most common items"""
        from collections import Counter
        return Counter(items).most_common(top_n)

    def _calculate_alert_performance(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate alert handling performance"""
        response_times = [e["response_time"] for e in events if e["response_time"]]
        resolution_times = [e["resolution_time"] for e in events if e["resolution_time"]]

        return {
            "avg_response_time": statistics.mean(response_times) if response_times else None,
            "avg_resolution_time": statistics.mean(resolution_times) if resolution_times else None,
            "response_time_p95": self._percentile(response_times, 95) if response_times else None,
            "resolution_time_p95": self._percentile(resolution_times, 95) if resolution_times else None
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0
        values_sorted = sorted(values)
        index = int(len(values_sorted) * percentile / 100)
        return values_sorted[min(index, len(values_sorted) - 1)]

    def _analyze_alert_trends(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze alert trends over time"""
        if len(events) < 20:
            return {"insufficient_data": True}

        # Split into time periods
        midpoint = len(events) // 2
        first_half = events[:midpoint]
        second_half = events[midpoint:]

        first_half_rate = len(first_half) / 12  # alerts per hour in first half
        second_half_rate = len(second_half) / 12  # alerts per hour in second half

        trend = "stable"
        if second_half_rate > first_half_rate * 1.2:
            trend = "increasing"
        elif second_half_rate < first_half_rate * 0.8:
            trend = "decreasing"

        return {
            "alert_trend": trend,
            "first_half_rate": first_half_rate,
            "second_half_rate": second_half_rate,
            "change_percentage": ((second_half_rate - first_half_rate) / first_half_rate) * 100 if first_half_rate > 0 else 0
        }

    def _generate_alert_recommendations(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alert system recommendations"""
        recommendations = []

        if len(events) == 0:
            return recommendations

        # High frequency alerts
        rule_counts = self._group_by_rule(events)
        for rule_id, count in rule_counts.items():
            if count > len(events) * 0.3:  # More than 30% of alerts from one rule
                recommendations.append({
                    "type": "alert_frequency",
                    "priority": "high",
                    "description": f"Rule {rule_id} is generating {count} alerts ({count/len(events)*100:.1f}% of total)",
                    "recommendation": "Review and adjust alert thresholds or rule logic"
                })

        # Poor response times
        performance = self._calculate_alert_performance(events)
        if performance.get("avg_response_time", 0) > 300:  # > 5 minutes average response
            recommendations.append({
                "type": "response_time",
                "priority": "high",
                "description": f"Average alert response time is {performance['avg_response_time']:.1f} seconds",
                "recommendation": "Improve alert notification delivery and on-call response processes"
            })

        # Alert trend analysis
        trends = self._analyze_alert_trends(events)
        if trends.get("alert_trend") == "increasing":
            recommendations.append({
                "type": "alert_trend",
                "priority": "medium",
                "description": f"Alert frequency is increasing by {trends.get('change_percentage', 0):.1f}%",
                "recommendation": "Investigate root causes of increasing alert frequency"
            })

        return recommendations
