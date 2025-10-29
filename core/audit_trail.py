"""
Comprehensive Audit Trail System for AI Personalized Medicine Platform
Tracks all system activities, data access, and security events with immutable logging
"""

import hashlib
import json
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
import logging
import sqlite3
from pathlib import Path

class AuditTrailSystem:
    """Comprehensive audit trail and logging system"""

    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self.audit_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        self.is_running = False
        self.audit_workers = []
        self.event_subscribers = defaultdict(set)
        self.retention_policies = {
            "access_logs": 7 * 365,  # 7 years
            "security_events": 7 * 365,
            "data_modifications": 7 * 365,
            "system_events": 365,  # 1 year
            "performance_logs": 90  # 90 days
        }
        self.initialize_database()

    def initialize_database(self):
        """Initialize audit trail database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create audit events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                event_type VARCHAR(100) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                user_id VARCHAR(100),
                patient_id VARCHAR(100),
                session_id VARCHAR(100),
                ip_address VARCHAR(45),
                user_agent TEXT,
                action VARCHAR(100),
                resource VARCHAR(200),
                details TEXT,
                compliance_status VARCHAR(50),
                hash_chain VARCHAR(64),
                previous_hash VARCHAR(64)
            )
        ''')

        # Create indexes for efficient querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patient_id ON audit_events(patient_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)')

        # Create alert rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name VARCHAR(200) NOT NULL,
                event_pattern TEXT NOT NULL,
                severity_threshold VARCHAR(20),
                alert_action VARCHAR(100),
                enabled BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create compliance reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type VARCHAR(100) NOT NULL,
                period_start DATE NOT NULL,
                period_end DATE NOT NULL,
                findings TEXT,
                compliance_score DECIMAL(5,2),
                generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

        print("📊 Audit trail database initialized")

    def start_audit_system(self):
        """Start the audit trail system"""
        self.is_running = True

        # Start audit processing workers
        for i in range(3):  # 3 worker threads
            worker = threading.Thread(target=self._audit_worker, daemon=True)
            worker.start()
            self.audit_workers.append(worker)

        # Start alert monitoring worker
        alert_worker = threading.Thread(target=self._alert_monitor, daemon=True)
        alert_worker.start()
        self.audit_workers.append(alert_worker)

        # Start retention policy worker
        retention_worker = threading.Thread(target=self._retention_worker, daemon=True)
        retention_worker.start()
        self.audit_workers.append(retention_worker)

        print("🔍 Audit trail system started")

    def stop_audit_system(self):
        """Stop the audit trail system"""
        self.is_running = False
        print("🛑 Audit trail system stopped")

    def log_event(self, event_data: Dict[str, Any]) -> str:
        """Log an audit event"""
        # Enrich event data
        enriched_event = self._enrich_event_data(event_data)

        # Generate hash chain for immutability
        enriched_event["hash_chain"] = self._generate_event_hash(enriched_event)

        # Add to processing queue
        self.audit_queue.put(enriched_event)

        # Check for alerts
        self._check_alert_rules(enriched_event)

        return enriched_event.get("id", "unknown")

    def _enrich_event_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich event data with additional context"""
        enriched = dict(event_data)  # Copy original data

        # Add timestamp if not provided
        if "timestamp" not in enriched:
            enriched["timestamp"] = datetime.now()

        # Add session ID if not provided
        if "session_id" not in enriched:
            enriched["session_id"] = f"session_{int(time.time())}_{random.randint(1000, 9999)}"

        # Determine severity if not provided
        if "severity" not in enriched:
            enriched["severity"] = self._determine_event_severity(enriched)

        # Add compliance status
        enriched["compliance_status"] = self._assess_compliance_status(enriched)

        # Add additional context
        enriched["system_context"] = {
            "platform_version": "1.0.0",
            "environment": "production",
            "data_center": "us-east-1"
        }

        return enriched

    def _determine_event_severity(self, event: Dict[str, Any]) -> str:
        """Determine event severity level"""
        event_type = event.get("event_type", "").lower()

        # Critical events
        if any(keyword in event_type for keyword in ["breach", "unauthorized", "intrusion"]):
            return "critical"

        # High severity events
        if any(keyword in event_type for keyword in ["access_denied", "encryption_failure", "data_loss"]):
            return "high"

        # Medium severity events
        if any(keyword in event_type for keyword in ["policy_violation", "unusual_activity", "failed_login"]):
            return "medium"

        # Low severity events (default)
        return "low"

    def _assess_compliance_status(self, event: Dict[str, Any]) -> str:
        """Assess compliance status of the event"""
        event_type = event.get("event_type", "").lower()

        # Non-compliant events
        if "violation" in event_type or "breach" in event_type:
            return "non_compliant"

        # Requires review
        if event.get("severity") in ["critical", "high"]:
            return "requires_review"

        # Compliant events
        return "compliant"

    def _generate_event_hash(self, event: Dict[str, Any]) -> str:
        """Generate cryptographic hash for event immutability"""
        # Get previous hash from last event
        previous_hash = self._get_last_event_hash()

        # Create event string for hashing
        event_string = json.dumps({
            "timestamp": event["timestamp"].isoformat() if hasattr(event["timestamp"], 'isoformat') else str(event["timestamp"]),
            "event_type": event["event_type"],
            "user_id": event.get("user_id"),
            "patient_id": event.get("patient_id"),
            "action": event.get("action"),
            "details": event.get("details"),
            "previous_hash": previous_hash
        }, sort_keys=True)

        # Generate SHA-256 hash
        return hashlib.sha256(event_string.encode()).hexdigest()

    def _get_last_event_hash(self) -> str:
        """Get hash of the last audit event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT hash_chain FROM audit_events ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()

        conn.close()

        return result[0] if result else "genesis_hash"

    def _audit_worker(self):
        """Background worker for processing audit events"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        while self.is_running:
            try:
                # Get event from queue with timeout
                event = self.audit_queue.get(timeout=1)

                # Insert into database
                cursor.execute('''
                    INSERT INTO audit_events
                    (timestamp, event_type, severity, user_id, patient_id, session_id,
                     ip_address, user_agent, action, resource, details, compliance_status,
                     hash_chain, previous_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event["timestamp"].isoformat() if hasattr(event["timestamp"], 'isoformat') else str(event["timestamp"]),
                    event["event_type"],
                    event["severity"],
                    event.get("user_id"),
                    event.get("patient_id"),
                    event.get("session_id"),
                    event.get("ip_address"),
                    event.get("user_agent"),
                    event.get("action"),
                    event.get("resource"),
                    json.dumps(event.get("details", {})),
                    event.get("compliance_status"),
                    event["hash_chain"],
                    event.get("previous_hash", "genesis_hash")
                ))

                conn.commit()

                # Notify subscribers
                self._notify_event_subscribers(event)

                self.audit_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audit worker error: {e}")

        conn.close()

    def _check_alert_rules(self, event: Dict[str, Any]):
        """Check if event triggers any alert rules"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get enabled alert rules
        cursor.execute("SELECT * FROM alert_rules WHERE enabled = 1")
        rules = cursor.fetchall()

        for rule in rules:
            rule_data = {
                "id": rule[0],
                "rule_name": rule[1],
                "event_pattern": json.loads(rule[2]),
                "severity_threshold": rule[3],
                "alert_action": rule[4]
            }

            if self._matches_alert_rule(event, rule_data):
                alert = {
                    "rule_id": rule_data["id"],
                    "rule_name": rule_data["rule_name"],
                    "triggered_event": event,
                    "severity": event["severity"],
                    "action": rule_data["alert_action"],
                    "timestamp": datetime.now()
                }

                self.alert_queue.put(alert)

        conn.close()

    def _matches_alert_rule(self, event: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check if event matches alert rule pattern"""
        pattern = rule["event_pattern"]

        # Check event type match
        if pattern.get("event_type") and pattern["event_type"] != event.get("event_type"):
            return False

        # Check severity threshold
        if rule.get("severity_threshold"):
            severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            event_severity = severity_levels.get(event.get("severity", "low"), 1)
            threshold_severity = severity_levels.get(rule["severity_threshold"], 1)

            if event_severity < threshold_severity:
                return False

        # Check additional conditions
        for key, value in pattern.items():
            if key in ["event_type", "severity_threshold"]:
                continue

            if key not in event or event[key] != value:
                return False

        return True

    def _alert_monitor(self):
        """Monitor and process alerts"""
        while self.is_running:
            try:
                alert = self.alert_queue.get(timeout=1)

                # Process alert based on action
                action = alert.get("action", "log")

                if action == "log":
                    self._log_alert(alert)
                elif action == "email":
                    self._send_alert_email(alert)
                elif action == "page":
                    self._send_alert_page(alert)
                elif action == "escalate":
                    self._escalate_alert(alert)

                self.alert_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Alert monitor error: {e}")

    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to file"""
        alert_log = logging.getLogger('audit_alerts')
        alert_log.warning(f"ALERT: {alert['rule_name']} - {alert['triggered_event']['event_type']}")

    def _send_alert_email(self, alert: Dict[str, Any]):
        """Send alert via email (simplified)"""
        print(f"📧 ALERT EMAIL: {alert['rule_name']} triggered")

    def _send_alert_page(self, alert: Dict[str, Any]):
        """Send alert page (simplified)"""
        print(f"📟 ALERT PAGE: CRITICAL - {alert['rule_name']} triggered")

    def _escalate_alert(self, alert: Dict[str, Any]):
        """Escalate alert to security team"""
        print(f"🚨 ESCALATED ALERT: {alert['rule_name']} - Immediate security team notification")

    def _notify_event_subscribers(self, event: Dict[str, Any]):
        """Notify event subscribers"""
        event_type = event.get("event_type")

        for subscriber in self.event_subscribers[event_type]:
            try:
                subscriber(event)
            except Exception as e:
                print(f"Subscriber notification error: {e}")

    def subscribe_to_events(self, event_type: str, callback: Callable):
        """Subscribe to specific event types"""
        self.event_subscribers[event_type].add(callback)

    def unsubscribe_from_events(self, event_type: str, callback: Callable):
        """Unsubscribe from event types"""
        self.event_subscribers[event_type].discard(callback)

    def _retention_worker(self):
        """Background worker for data retention management"""
        while self.is_running:
            try:
                self._apply_retention_policies()
                time.sleep(86400)  # Run daily
            except Exception as e:
                print(f"Retention worker error: {e}")

    def _apply_retention_policies(self):
        """Apply data retention policies"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for event_type, retention_days in self.retention_policies.items():
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            # Delete old records
            if event_type == "access_logs":
                cursor.execute("DELETE FROM audit_events WHERE event_type LIKE '%access%' AND timestamp < ?",
                             (cutoff_date.isoformat(),))
            elif event_type == "security_events":
                cursor.execute("DELETE FROM audit_events WHERE severity IN ('high', 'critical') AND timestamp < ?",
                             (cutoff_date.isoformat(),))
            elif event_type == "data_modifications":
                cursor.execute("DELETE FROM audit_events WHERE action LIKE '%modify%' AND timestamp < ?",
                             (cutoff_date.isoformat(),))
            elif event_type == "system_events":
                cursor.execute("DELETE FROM audit_events WHERE event_type LIKE 'system_%' AND timestamp < ?",
                             (cutoff_date.isoformat(),))
            elif event_type == "performance_logs":
                cursor.execute("DELETE FROM audit_events WHERE event_type LIKE '%performance%' AND timestamp < ?",
                             (cutoff_date.isoformat(),))

        conn.commit()
        conn.close()

    def query_audit_events(self, filters: Dict[str, Any] = None,
                          start_date: datetime = None, end_date: datetime = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Query audit events with filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        if filters:
            for key, value in filters.items():
                if key == "event_type":
                    query += " AND event_type = ?"
                    params.append(value)
                elif key == "user_id":
                    query += " AND user_id = ?"
                    params.append(value)
                elif key == "patient_id":
                    query += " AND patient_id = ?"
                    params.append(value)
                elif key == "severity":
                    query += " AND severity = ?"
                    params.append(value)
                elif key == "compliance_status":
                    query += " AND compliance_status = ?"
                    params.append(value)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to dictionaries
        columns = [desc[0] for desc in cursor.description]
        events = []

        for row in rows:
            event = dict(zip(columns, row))
            # Parse JSON details
            if event.get("details"):
                try:
                    event["details"] = json.loads(event["details"])
                except:
                    pass
            events.append(event)

        conn.close()
        return events

    def get_audit_statistics(self, start_date: datetime = None,
                           end_date: datetime = None) -> Dict[str, Any]:
        """Get audit statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Default to last 30 days
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()

        # Event type distribution
        cursor.execute('''
            SELECT event_type, COUNT(*) as count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY event_type
            ORDER BY count DESC
        ''', (start_date.isoformat(), end_date.isoformat()))

        event_distribution = dict(cursor.fetchall())

        # Severity distribution
        cursor.execute('''
            SELECT severity, COUNT(*) as count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY severity
        ''', (start_date.isoformat(), end_date.isoformat()))

        severity_distribution = dict(cursor.fetchall())

        # Compliance status
        cursor.execute('''
            SELECT compliance_status, COUNT(*) as count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY compliance_status
        ''', (start_date.isoformat(), end_date.isoformat()))

        compliance_distribution = dict(cursor.fetchall())

        # Top users by activity
        cursor.execute('''
            SELECT user_id, COUNT(*) as activity_count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ? AND user_id IS NOT NULL
            GROUP BY user_id
            ORDER BY activity_count DESC
            LIMIT 10
        ''', (start_date.isoformat(), end_date.isoformat()))

        top_users = dict(cursor.fetchall())

        # Daily event counts
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (start_date.isoformat(), end_date.isoformat()))

        daily_counts = dict(cursor.fetchall())

        conn.close()

        return {
            "period": {"start": start_date, "end": end_date},
            "total_events": sum(event_distribution.values()),
            "event_distribution": event_distribution,
            "severity_distribution": severity_distribution,
            "compliance_distribution": compliance_distribution,
            "top_users": top_users,
            "daily_activity": daily_counts,
            "average_daily_events": sum(daily_counts.values()) / max(1, len(daily_counts)),
            "anomalies_detected": self._detect_audit_anomalies(daily_counts)
        }

    def _detect_audit_anomalies(self, daily_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """Detect anomalies in audit patterns"""
        if len(daily_counts) < 7:
            return []

        anomalies = []
        values = list(daily_counts.values())
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0

        if std == 0:
            return []

        dates = list(daily_counts.keys())
        for i, (date, count) in enumerate(daily_counts.items()):
            z_score = abs(count - mean) / std

            if z_score > 2.5:  # 2.5 standard deviations
                anomalies.append({
                    "date": date,
                    "count": count,
                    "expected_range": (mean - 2*std, mean + 2*std),
                    "z_score": z_score,
                    "severity": "high" if z_score > 3 else "medium",
                    "description": f"Unusual activity level: {count} events"
                })

        return anomalies

    def create_alert_rule(self, rule_data: Dict[str, Any]) -> int:
        """Create a new alert rule"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO alert_rules (rule_name, event_pattern, severity_threshold, alert_action, enabled)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            rule_data["rule_name"],
            json.dumps(rule_data["event_pattern"]),
            rule_data.get("severity_threshold"),
            rule_data.get("alert_action", "log"),
            rule_data.get("enabled", True)
        ))

        rule_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return rule_id

    def generate_compliance_report(self, report_type: str, start_date: date,
                                 end_date: date) -> Dict[str, Any]:
        """Generate compliance report"""
        # Gather audit data
        audit_data = self.get_audit_statistics(start_date, end_date)

        # Calculate compliance metrics
        compliance_metrics = self._calculate_compliance_metrics(audit_data)

        # Identify compliance issues
        compliance_issues = self._identify_compliance_issues(audit_data)

        # Generate recommendations
        recommendations = self._generate_audit_recommendations(compliance_issues)

        report = {
            "report_type": report_type,
            "period": {"start": start_date, "end": end_date},
            "generated_at": datetime.now(),
            "compliance_score": compliance_metrics["overall_score"],
            "compliance_grade": self._calculate_compliance_grade(compliance_metrics["overall_score"]),
            "audit_summary": audit_data,
            "compliance_metrics": compliance_metrics,
            "identified_issues": compliance_issues,
            "recommendations": recommendations,
            "next_audit_date": end_date + timedelta(days=90)
        }

        # Save report to database
        self._save_compliance_report(report)

        return report

    def _calculate_compliance_metrics(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compliance metrics from audit data"""
        metrics = {}

        # Access control compliance
        access_denied = audit_data["event_distribution"].get("ACCESS_DENIED", 0)
        total_access = sum(count for event, count in audit_data["event_distribution"].items()
                          if "access" in event.lower())
        metrics["access_control_compliance"] = 100 - (access_denied / max(1, total_access) * 100)

        # Data encryption compliance
        encryption_events = audit_data["event_distribution"].get("PHI_ENCRYPTION", 0)
        data_events = sum(count for event, count in audit_data["event_distribution"].items()
                         if "data" in event.lower())
        metrics["encryption_compliance"] = min(100, (encryption_events / max(1, data_events)) * 100)

        # Incident reporting compliance
        security_events = sum(count for event, count in audit_data["event_distribution"].items()
                             if "security" in event.lower() or "breach" in event.lower())
        reported_incidents = audit_data["event_distribution"].get("SECURITY_INCIDENT", 0)
        metrics["incident_reporting_compliance"] = min(100, (reported_incidents / max(1, security_events)) * 100)

        # User activity monitoring
        unique_users = len(audit_data.get("top_users", {}))
        total_events = audit_data["total_events"]
        metrics["activity_monitoring_compliance"] = min(100, (unique_users / max(1, total_events)) * 1000)

        # Overall compliance score
        metrics["overall_score"] = statistics.mean([
            metrics["access_control_compliance"],
            metrics["encryption_compliance"],
            metrics["incident_reporting_compliance"],
            metrics["activity_monitoring_compliance"]
        ])

        return metrics

    def _identify_compliance_issues(self, audit_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify compliance issues from audit data"""
        issues = []

        # Check for access control issues
        access_denied_rate = audit_data["event_distribution"].get("ACCESS_DENIED", 0) / max(1, audit_data["total_events"])
        if access_denied_rate > 0.05:  # More than 5% access denied
            issues.append({
                "category": "access_control",
                "severity": "high",
                "description": f"High access denial rate: {access_denied_rate:.1%}",
                "recommendation": "Review access control policies and user permissions"
            })

        # Check for unusual activity
        if audit_data.get("anomalies_detected"):
            issues.append({
                "category": "anomaly_detection",
                "severity": "medium",
                "description": f"Detected {len(audit_data['anomalies_detected'])} anomalous activity patterns",
                "recommendation": "Investigate unusual activity patterns and implement additional monitoring"
            })

        # Check compliance distribution
        non_compliant = audit_data["compliance_distribution"].get("non_compliant", 0)
        if non_compliant > 0:
            issues.append({
                "category": "compliance_violations",
                "severity": "high",
                "description": f"Identified {non_compliant} non-compliant events",
                "recommendation": "Review and address compliance violations immediately"
            })

        return issues

    def _generate_audit_recommendations(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate audit recommendations based on identified issues"""
        recommendations = []

        issue_categories = set(issue["category"] for issue in issues)

        if "access_control" in issue_categories:
            recommendations.append({
                "priority": "high",
                "area": "Access Control",
                "recommendation": "Implement multi-factor authentication and regular access reviews",
                "timeline": "30 days",
                "responsible_party": "IT Security"
            })

        if "anomaly_detection" in issue_categories:
            recommendations.append({
                "priority": "medium",
                "area": "Monitoring",
                "recommendation": "Enhance real-time monitoring and alerting systems",
                "timeline": "60 days",
                "responsible_party": "IT Operations"
            })

        if "compliance_violations" in issue_categories:
            recommendations.append({
                "priority": "critical",
                "area": "Compliance",
                "recommendation": "Conduct comprehensive compliance training and policy review",
                "timeline": "14 days",
                "responsible_party": "Compliance Officer"
            })

        return recommendations

    def _calculate_compliance_grade(self, score: float) -> str:
        """Calculate compliance grade from score"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _save_compliance_report(self, report: Dict[str, Any]):
        """Save compliance report to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO compliance_reports
            (report_type, period_start, period_end, findings, compliance_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            report["report_type"],
            report["period"]["start"].isoformat(),
            report["period"]["end"].isoformat(),
            json.dumps({
                "issues": report["identified_issues"],
                "metrics": report["compliance_metrics"]
            }),
            report["compliance_score"]
        ))

        conn.commit()
        conn.close()

    def verify_audit_integrity(self, start_date: datetime = None,
                             end_date: datetime = None) -> Dict[str, Any]:
        """Verify integrity of audit trail using hash chain"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get events in chronological order
        query = "SELECT id, hash_chain, previous_hash FROM audit_events"
        params = []

        if start_date:
            query += " WHERE timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            if "WHERE" in query:
                query += " AND timestamp <= ?"
            else:
                query += " WHERE timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY id"

        cursor.execute(query, params)
        events = cursor.fetchall()

        conn.close()

        # Verify hash chain
        integrity_issues = []
        previous_hash = "genesis_hash"

        for event_id, current_hash, stored_previous_hash in events:
            # Check if previous hash matches
            if stored_previous_hash != previous_hash:
                integrity_issues.append({
                    "event_id": event_id,
                    "issue": "hash_chain_broken",
                    "expected_previous": previous_hash,
                    "stored_previous": stored_previous_hash
                })

            # Update previous hash for next iteration
            previous_hash = current_hash

        return {
            "events_checked": len(events),
            "integrity_violations": len(integrity_issues),
            "integrity_score": 100 - (len(integrity_issues) / max(1, len(events)) * 100),
            "violations": integrity_issues,
            "verification_status": "compromised" if integrity_issues else "intact"
        }
