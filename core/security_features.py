"""
Advanced Security Features for AI Personalized Medicine Platform
Multi-factor authentication, role-based access control, and audit logging
"""

import hashlib
import hmac
import secrets
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import re

class SecurityLevel(Enum):
    PUBLIC = 0
    BASIC = 1
    SENSITIVE = 2
    CRITICAL = 3
    RESTRICTED = 4

class UserRole(Enum):
    PATIENT = "patient"
    PROVIDER = "provider"
    RESEARCHER = "researcher"
    ADMIN = "admin"
    SYSTEM = "system"

class MultiFactorAuthentication:
    """Multi-factor authentication system"""

    def __init__(self):
        self.active_sessions = {}
        self.mfa_methods = {
            "totp": self._handle_totp,
            "sms": self._handle_sms,
            "email": self._handle_email,
            "push": self._handle_push,
            "hardware": self._handle_hardware_key
        }
        self.mfa_secrets = {}
        self.pending_challenges = {}

    def setup_mfa(self, user_id: str, method: str = "totp") -> Dict[str, Any]:
        """Setup MFA for user"""
        if method not in self.mfa_methods:
            raise ValueError(f"Unsupported MFA method: {method}")

        setup_data = {
            "user_id": user_id,
            "method": method,
            "enabled": False,
            "setup_at": datetime.now().isoformat()
        }

        if method == "totp":
            setup_data["secret"] = self._generate_totp_secret()
            setup_data["qr_code_url"] = self._generate_qr_code_url(user_id, setup_data["secret"])
        elif method == "sms":
            setup_data["phone_number"] = None  # To be set by user
        elif method == "email":
            setup_data["email"] = None  # To be set by user

        self.mfa_secrets[user_id] = setup_data

        return setup_data

    def verify_mfa_setup(self, user_id: str, verification_code: str) -> bool:
        """Verify MFA setup with verification code"""
        if user_id not in self.mfa_secrets:
            return False

        setup_data = self.mfa_secrets[user_id]
        method = setup_data["method"]

        if method == "totp":
            return self._verify_totp_code(setup_data["secret"], verification_code)
        elif method == "sms":
            # Simulate SMS verification
            return verification_code == "123456"
        elif method == "email":
            # Simulate email verification
            return verification_code == "987654"

        return False

    def authenticate_with_mfa(self, user_id: str, session_id: str, method: str = None) -> Dict[str, Any]:
        """Initiate MFA authentication for login"""
        if user_id not in self.mfa_secrets:
            return {"error": "MFA not configured for user"}

        setup_data = self.mfa_secrets[user_id]
        method = method or setup_data["method"]

        challenge_id = f"challenge_{int(datetime.now().timestamp())}_{secrets.token_hex(4)}"

        challenge = {
            "challenge_id": challenge_id,
            "user_id": user_id,
            "session_id": session_id,
            "method": method,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=5)).isoformat()
        }

        # Send challenge based on method
        success = self.mfa_methods[method](user_id, "challenge", challenge)

        if success:
            self.pending_challenges[challenge_id] = challenge
            return {
                "challenge_id": challenge_id,
                "method": method,
                "message": f"MFA challenge sent via {method}"
            }
        else:
            return {"error": "Failed to send MFA challenge"}

    def verify_mfa_challenge(self, challenge_id: str, verification_code: str) -> bool:
        """Verify MFA challenge response"""
        if challenge_id not in self.pending_challenges:
            return False

        challenge = self.pending_challenges[challenge_id]

        # Check if challenge expired
        if datetime.now() > datetime.fromisoformat(challenge["expires_at"]):
            del self.pending_challenges[challenge_id]
            return False

        setup_data = self.mfa_secrets[challenge["user_id"]]
        method = challenge["method"]

        verified = False
        if method == "totp":
            verified = self._verify_totp_code(setup_data["secret"], verification_code)
        elif method == "sms":
            verified = verification_code == "123456"  # Simulate
        elif method == "email":
            verified = verification_code == "987654"  # Simulate

        if verified:
            challenge["status"] = "verified"
            challenge["verified_at"] = datetime.now().isoformat()

            # Mark MFA as enabled if not already
            setup_data["enabled"] = True
            setup_data["last_verified"] = datetime.now().isoformat()

        return verified

    def _generate_totp_secret(self) -> str:
        """Generate TOTP secret"""
        return secrets.token_hex(20).upper()

    def _generate_qr_code_url(self, user_id: str, secret: str) -> str:
        """Generate QR code URL for TOTP setup"""
        issuer = "AI+Medicine+Platform"
        account = f"{issuer}:{user_id}"
        parameters = f"secret={secret}&issuer={issuer}&algorithm=SHA1&digits=6&period=30"
        return f"otpauth://totp/{account}?{parameters}"

    def _verify_totp_code(self, secret: str, code: str) -> bool:
        """Verify TOTP code"""
        # Simplified TOTP verification (would use proper TOTP library)
        expected_code = str(int(time.time() // 30) % 1000000).zfill(6)
        return code == expected_code

    def _handle_totp(self, user_id: str, action: str, data: Dict[str, Any]) -> bool:
        """Handle TOTP MFA"""
        return True  # TOTP doesn't need to send anything

    def _handle_sms(self, user_id: str, action: str, data: Dict[str, Any]) -> bool:
        """Handle SMS MFA"""
        # Simulate SMS sending
        print(f"SMS sent to user {user_id}: Your verification code is 123456")
        return True

    def _handle_email(self, user_id: str, action: str, data: Dict[str, Any]) -> bool:
        """Handle email MFA"""
        # Simulate email sending
        print(f"Email sent to user {user_id}: Your verification code is 987654")
        return True

    def _handle_push(self, user_id: str, action: str, data: Dict[str, Any]) -> bool:
        """Handle push notification MFA"""
        # Simulate push notification
        print(f"Push notification sent to user {user_id} device")
        return True

    def _handle_hardware_key(self, user_id: str, action: str, data: Dict[str, Any]) -> bool:
        """Handle hardware security key MFA"""
        # Simulate hardware key challenge
        print(f"Hardware key challenge sent to user {user_id}")
        return True

class RoleBasedAccessControl:
    """Role-based access control system"""

    def __init__(self):
        self.roles = self._initialize_roles()
        self.permissions = self._initialize_permissions()
        self.user_roles = defaultdict(set)
        self.role_permissions = defaultdict(set)

    def _initialize_roles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize system roles"""
        return {
            "patient": {
                "name": "Patient",
                "description": "Basic patient access",
                "level": 1,
                "inherits_from": []
            },
            "provider": {
                "name": "Healthcare Provider",
                "description": "Medical professional access",
                "level": 2,
                "inherits_from": ["patient"]
            },
            "researcher": {
                "name": "Researcher",
                "description": "Research and analytics access",
                "level": 3,
                "inherits_from": ["patient"]
            },
            "admin": {
                "name": "Administrator",
                "description": "Full system administration access",
                "level": 4,
                "inherits_from": ["provider", "researcher"]
            },
            "system": {
                "name": "System",
                "description": "Automated system processes",
                "level": 5,
                "inherits_from": []
            }
        }

    def _initialize_permissions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize system permissions"""
        return {
            # Patient permissions
            "view_own_records": {
                "resource": "patient_records",
                "action": "read",
                "scope": "own",
                "security_level": SecurityLevel.SENSITIVE
            },
            "update_own_profile": {
                "resource": "patient_profile",
                "action": "update",
                "scope": "own",
                "security_level": SecurityLevel.BASIC
            },

            # Provider permissions
            "view_patient_records": {
                "resource": "patient_records",
                "action": "read",
                "scope": "assigned_patients",
                "security_level": SecurityLevel.SENSITIVE
            },
            "update_patient_records": {
                "resource": "patient_records",
                "action": "update",
                "scope": "assigned_patients",
                "security_level": SecurityLevel.CRITICAL
            },
            "prescribe_medications": {
                "resource": "medications",
                "action": "create",
                "scope": "assigned_patients",
                "security_level": SecurityLevel.CRITICAL
            },

            # Researcher permissions
            "view_aggregate_data": {
                "resource": "population_data",
                "action": "read",
                "scope": "aggregate",
                "security_level": SecurityLevel.SENSITIVE
            },
            "export_research_data": {
                "resource": "research_data",
                "action": "export",
                "scope": "approved_studies",
                "security_level": SecurityLevel.RESTRICTED
            },

            # Admin permissions
            "manage_users": {
                "resource": "user_accounts",
                "action": "manage",
                "scope": "all",
                "security_level": SecurityLevel.RESTRICTED
            },
            "system_configuration": {
                "resource": "system_settings",
                "action": "manage",
                "scope": "all",
                "security_level": SecurityLevel.RESTRICTED
            },
            "audit_logs": {
                "resource": "audit_logs",
                "action": "read",
                "scope": "all",
                "security_level": SecurityLevel.RESTRICTED
            }
        }

    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user"""
        if role not in self.roles:
            return False

        self.user_roles[user_id].add(role)

        # Update role permissions cache
        self._update_user_permissions(user_id)

        return True

    def revoke_role(self, user_id: str, role: str) -> bool:
        """Revoke role from user"""
        if role not in self.user_roles[user_id]:
            return False

        self.user_roles[user_id].remove(role)

        # Update role permissions cache
        self._update_user_permissions(user_id)

        return True

    def _update_user_permissions(self, user_id: str) -> None:
        """Update cached permissions for user"""
        user_permissions = set()

        # Get direct role permissions and inherited permissions
        for role in self.user_roles[user_id]:
            role_permissions = self._get_role_permissions(role)
            user_permissions.update(role_permissions)

        self.role_permissions[user_id] = user_permissions

    def _get_role_permissions(self, role: str) -> Set[str]:
        """Get all permissions for a role including inherited ones"""
        permissions = set()

        # Add direct permissions (simplified - would map roles to permissions)
        if role == "admin":
            permissions.update([
                "view_own_records", "update_own_profile", "view_patient_records",
                "update_patient_records", "prescribe_medications", "view_aggregate_data",
                "export_research_data", "manage_users", "system_configuration", "audit_logs"
            ])
        elif role == "provider":
            permissions.update([
                "view_own_records", "update_own_profile", "view_patient_records",
                "update_patient_records", "prescribe_medications"
            ])
        elif role == "researcher":
            permissions.update([
                "view_own_records", "update_own_profile", "view_aggregate_data",
                "export_research_data"
            ])
        elif role == "patient":
            permissions.update(["view_own_records", "update_own_profile"])

        # Add inherited permissions
        for inherited_role in self.roles[role]["inherits_from"]:
            permissions.update(self._get_role_permissions(inherited_role))

        return permissions

    def check_permission(self, user_id: str, permission: str, resource_context: Dict[str, Any] = None) -> bool:
        """Check if user has specific permission"""
        if user_id not in self.role_permissions:
            return False

        user_permissions = self.role_permissions[user_id]

        # Check direct permission
        if permission not in user_permissions:
            return False

        # Check resource context (scope validation)
        if resource_context:
            return self._validate_permission_scope(user_id, permission, resource_context)

        return True

    def _validate_permission_scope(self, user_id: str, permission: str, context: Dict[str, Any]) -> bool:
        """Validate permission scope against resource context"""
        perm_config = self.permissions.get(permission, {})

        scope = perm_config.get("scope", "all")

        if scope == "own":
            # Check if resource belongs to user
            resource_owner = context.get("owner_id")
            return resource_owner == user_id

        elif scope == "assigned_patients":
            # Check if patient is assigned to provider
            patient_id = context.get("patient_id")
            return self._is_patient_assigned_to_provider(user_id, patient_id)

        elif scope == "approved_studies":
            # Check if study is approved for researcher
            study_id = context.get("study_id")
            return self._is_study_approved_for_researcher(user_id, study_id)

        elif scope == "all":
            return True

        return False

    def _is_patient_assigned_to_provider(self, provider_id: str, patient_id: str) -> bool:
        """Check if patient is assigned to provider"""
        # Simplified check - would query database
        return True  # Assume assigned for demo

    def _is_study_approved_for_researcher(self, researcher_id: str, study_id: str) -> bool:
        """Check if study is approved for researcher"""
        # Simplified check - would query database
        return True  # Assume approved for demo

    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for user"""
        return list(self.role_permissions.get(user_id, set()))

    def get_user_roles(self, user_id: str) -> List[str]:
        """Get all roles for user"""
        return list(self.user_roles[user_id])

class AuditLoggingSystem:
    """Comprehensive audit logging system"""

    def __init__(self):
        self.audit_logs = deque(maxlen=100000)  # Keep last 100k entries
        self.log_levels = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        self.retention_policies = {
            "DEBUG": timedelta(days=7),
            "INFO": timedelta(days=30),
            "WARNING": timedelta(days=90),
            "ERROR": timedelta(days=365),
            "CRITICAL": timedelta(days=365*2)
        }

    def log_event(self, event_type: str, user_id: str, resource: str,
                  action: str, details: Dict[str, Any] = None,
                  level: str = "INFO", ip_address: str = None) -> str:
        """Log audit event"""
        event_id = f"audit_{int(datetime.now().timestamp())}_{secrets.token_hex(4)}"

        audit_entry = {
            "event_id": event_id,
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "details": details or {},
            "level": level,
            "ip_address": ip_address,
            "session_id": details.get("session_id") if details else None,
            "user_agent": details.get("user_agent") if details else None,
            "security_level": self._determine_security_level(event_type, action)
        }

        self.audit_logs.append(audit_entry)

        # Check for security alerts
        self._check_security_alerts(audit_entry)

        return event_id

    def _determine_security_level(self, event_type: str, action: str) -> SecurityLevel:
        """Determine security level of audit event"""
        security_events = {
            "authentication": SecurityLevel.SENSITIVE,
            "authorization_failure": SecurityLevel.CRITICAL,
            "data_access": SecurityLevel.SENSITIVE,
            "data_modification": SecurityLevel.CRITICAL,
            "system_configuration": SecurityLevel.RESTRICTED,
            "security_incident": SecurityLevel.RESTRICTED
        }

        if event_type in security_events:
            return security_events[event_type]

        if action in ["delete", "modify"]:
            return SecurityLevel.CRITICAL

        return SecurityLevel.BASIC

    def _check_security_alerts(self, audit_entry: Dict[str, Any]) -> None:
        """Check for security alerts based on audit entry"""
        alerts = []

        # Check for suspicious login patterns
        if audit_entry["event_type"] == "authentication":
            if audit_entry["details"].get("failed_attempts", 0) > 3:
                alerts.append({
                    "alert_type": "multiple_failed_logins",
                    "severity": "high",
                    "user_id": audit_entry["user_id"],
                    "message": f"Multiple failed login attempts for user {audit_entry['user_id']}"
                })

        # Check for unauthorized access attempts
        elif audit_entry["event_type"] == "authorization_failure":
            alerts.append({
                "alert_type": "unauthorized_access",
                "severity": "high",
                "user_id": audit_entry["user_id"],
                "resource": audit_entry["resource"],
                "message": f"Unauthorized access attempt to {audit_entry['resource']} by {audit_entry['user_id']}"
            })

        # Check for data export activities
        elif audit_entry["action"] == "export" and audit_entry["resource"] == "patient_data":
            alerts.append({
                "alert_type": "data_export",
                "severity": "medium",
                "user_id": audit_entry["user_id"],
                "message": f"Large data export initiated by {audit_entry['user_id']}"
            })

    def query_audit_logs(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit logs with filters"""
        logs = list(self.audit_logs)

        if filters:
            filtered_logs = []
            for log_entry in logs:
                match = True

                for key, value in filters.items():
                    if key not in log_entry or log_entry[key] != value:
                        match = False
                        break

                if match:
                    filtered_logs.append(log_entry)

            logs = filtered_logs

        return logs[-limit:]

    def generate_audit_report(self, start_date: str, end_date: str,
                            report_type: str = "summary") -> Dict[str, Any]:
        """Generate audit report for date range"""
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        relevant_logs = [
            log for log in self.audit_logs
            if start <= datetime.fromisoformat(log["timestamp"]) <= end
        ]

        if report_type == "summary":
            return self._generate_summary_report(relevant_logs, start, end)
        elif report_type == "security":
            return self._generate_security_report(relevant_logs, start, end)
        elif report_type == "compliance":
            return self._generate_compliance_report(relevant_logs, start, end)

        return {"error": "Unknown report type"}

    def _generate_summary_report(self, logs: List[Dict[str, Any]],
                               start: datetime, end: datetime) -> Dict[str, Any]:
        """Generate summary audit report"""
        total_events = len(logs)
        events_by_type = defaultdict(int)
        events_by_user = defaultdict(int)
        events_by_level = defaultdict(int)

        for log in logs:
            events_by_type[log["event_type"]] += 1
            events_by_user[log["user_id"]] += 1
            events_by_level[log["level"]] += 1

        return {
            "report_type": "audit_summary",
            "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "total_events": total_events,
            "events_by_type": dict(events_by_type),
            "events_by_user": dict(events_by_user),
            "events_by_level": dict(events_by_level),
            "most_active_user": max(events_by_user.keys(), key=lambda x: events_by_user[x]) if events_by_user else None,
            "generated_at": datetime.now().isoformat()
        }

    def _generate_security_report(self, logs: List[Dict[str, Any]],
                                start: datetime, end: datetime) -> Dict[str, Any]:
        """Generate security-focused audit report"""
        security_events = [
            log for log in logs
            if log["security_level"] in [SecurityLevel.CRITICAL, SecurityLevel.RESTRICTED]
        ]

        failed_authentications = [
            log for log in logs
            if log["event_type"] == "authentication" and not log["details"].get("success", False)
        ]

        unauthorized_access = [
            log for log in logs
            if log["event_type"] == "authorization_failure"
        ]

        return {
            "report_type": "security_audit",
            "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "security_events_count": len(security_events),
            "failed_authentications": len(failed_authentications),
            "unauthorized_access_attempts": len(unauthorized_access),
            "security_events": security_events[-50:],  # Last 50 security events
            "risk_assessment": self._assess_security_risk(security_events),
            "generated_at": datetime.now().isoformat()
        }

    def _generate_compliance_report(self, logs: List[Dict[str, Any]],
                                  start: datetime, end: datetime) -> Dict[str, Any]:
        """Generate compliance-focused audit report"""
        compliance_events = {
            "data_access": [log for log in logs if log["action"] in ["read", "export"]],
            "data_modification": [log for log in logs if log["action"] in ["create", "update", "delete"]],
            "user_management": [log for log in logs if log["resource"] == "user_accounts"],
            "system_changes": [log for log in logs if log["resource"] == "system_settings"]
        }

        return {
            "report_type": "compliance_audit",
            "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "compliance_events": {k: len(v) for k, v in compliance_events.items()},
            "data_access_patterns": self._analyze_data_access_patterns(compliance_events["data_access"]),
            "user_activity_summary": self._summarize_user_activity(logs),
            "generated_at": datetime.now().isoformat()
        }

    def _assess_security_risk(self, security_events: List[Dict[str, Any]]) -> str:
        """Assess overall security risk level"""
        if len(security_events) > 50:
            return "HIGH"
        elif len(security_events) > 20:
            return "MEDIUM"
        else:
            return "LOW"

    def _analyze_data_access_patterns(self, access_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data access patterns"""
        patterns = {
            "unique_users": len(set(log["user_id"] for log in access_events)),
            "most_accessed_resource": None,
            "peak_access_hour": None
        }

        if access_events:
            resource_counts = defaultdict(int)
            hour_counts = defaultdict(int)

            for event in access_events:
                resource_counts[event["resource"]] += 1
                event_hour = datetime.fromisoformat(event["timestamp"]).hour
                hour_counts[event_hour] += 1

            patterns["most_accessed_resource"] = max(resource_counts.keys(),
                                                    key=lambda x: resource_counts[x])
            patterns["peak_access_hour"] = max(hour_counts.keys(),
                                             key=lambda x: hour_counts[x])

        return patterns

    def _summarize_user_activity(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize user activity patterns"""
        user_activity = defaultdict(lambda: {"events": 0, "last_activity": None})

        for log in logs:
            user_id = log["user_id"]
            user_activity[user_id]["events"] += 1
            user_activity[user_id]["last_activity"] = log["timestamp"]

        return dict(user_activity)

class DataEncryptionEngine:
    """Data encryption and key management system"""

    def __init__(self):
        self.encryption_keys = {}
        self.key_versions = defaultdict(list)
        self.key_rotation_schedule = {}

    def generate_encryption_key(self, key_id: str, key_type: str = "AES256") -> str:
        """Generate new encryption key"""
        key = secrets.token_hex(32)  # 256-bit key

        key_record = {
            "key_id": key_id,
            "key_type": key_type,
            "key_material": key,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "version": len(self.key_versions[key_id]) + 1
        }

        self.encryption_keys[key_id] = key_record
        self.key_versions[key_id].append(key_record)

        return key_id

    def encrypt_data(self, data: str, key_id: str) -> Dict[str, Any]:
        """Encrypt data using specified key"""
        if key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key not found: {key_id}")

        key_record = self.encryption_keys[key_id]

        # Simplified encryption (would use proper encryption library)
        encrypted_data = self._simple_encrypt(data, key_record["key_material"])

        return {
            "encrypted_data": encrypted_data,
            "key_id": key_id,
            "key_version": key_record["version"],
            "encryption_method": "AES256-GCM",
            "encrypted_at": datetime.now().isoformat()
        }

    def decrypt_data(self, encrypted_data: Dict[str, Any]) -> str:
        """Decrypt data using appropriate key"""
        key_id = encrypted_data["key_id"]
        key_version = encrypted_data.get("key_version", 1)

        # Find the correct key version
        key_record = None
        for version in self.key_versions[key_id]:
            if version["version"] == key_version:
                key_record = version
                break

        if not key_record:
            raise ValueError(f"Key version not found: {key_id} v{key_version}")

        # Simplified decryption
        decrypted_data = self._simple_decrypt(
            encrypted_data["encrypted_data"],
            key_record["key_material"]
        )

        return decrypted_data

    def _simple_encrypt(self, data: str, key: str) -> str:
        """Simple encryption for demonstration (NOT SECURE)"""
        # This is just for demonstration - use proper encryption in production
        return data[::-1]  # Reverse string as simple "encryption"

    def _simple_decrypt(self, encrypted_data: str, key: str) -> str:
        """Simple decryption for demonstration (NOT SECURE)"""
        # This is just for demonstration - use proper decryption in production
        return encrypted_data[::-1]  # Reverse back

    def rotate_key(self, key_id: str) -> str:
        """Rotate encryption key"""
        new_key_id = f"{key_id}_v{len(self.key_versions[key_id]) + 1}"
        return self.generate_encryption_key(new_key_id)

class IntrusionDetectionSystem:
    """Intrusion detection and prevention system"""

    def __init__(self):
        self.suspicious_patterns = self._initialize_patterns()
        self.incident_log = []
        self.blocked_ips = set()
        self.threat_levels = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }

    def _initialize_patterns(self) -> List[Dict[str, Any]]:
        """Initialize intrusion detection patterns"""
        return [
            {
                "pattern_id": "brute_force",
                "description": "Multiple failed authentication attempts",
                "threshold": 5,
                "time_window": 300,  # 5 minutes
                "severity": "high"
            },
            {
                "pattern_id": "unusual_access",
                "description": "Access from unusual location/time",
                "threshold": 1,
                "severity": "medium"
            },
            {
                "pattern_id": "data_exfiltration",
                "description": "Large data export followed by account deletion",
                "threshold": 1,
                "severity": "critical"
            },
            {
                "pattern_id": "sql_injection_attempt",
                "description": "SQL injection patterns in input",
                "threshold": 1,
                "severity": "high"
            }
        ]

    def analyze_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze event for suspicious patterns"""
        detected_threats = []

        for pattern in self.suspicious_patterns:
            if self._matches_pattern(event, pattern):
                threat = {
                    "threat_id": f"threat_{int(datetime.now().timestamp())}_{secrets.token_hex(4)}",
                    "pattern_id": pattern["pattern_id"],
                    "severity": pattern["severity"],
                    "description": pattern["description"],
                    "event": event,
                    "detected_at": datetime.now().isoformat(),
                    "confidence": random.uniform(0.7, 0.95)
                }
                detected_threats.append(threat)

        if detected_threats:
            # Log the incident
            incident = {
                "incident_id": f"incident_{int(datetime.now().timestamp())}",
                "threats": detected_threats,
                "overall_severity": max(t["severity"] for t in detected_threats),
                "status": "detected",
                "response_actions": self._determine_response_actions(detected_threats)
            }

            self.incident_log.append(incident)

            return incident

        return None

    def _matches_pattern(self, event: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if event matches intrusion pattern"""
        pattern_id = pattern["pattern_id"]

        if pattern_id == "brute_force":
            return (event.get("event_type") == "authentication" and
                   event.get("details", {}).get("failed_attempts", 0) >= pattern["threshold"])

        elif pattern_id == "sql_injection_attempt":
            # Check for SQL injection patterns
            suspicious_input = str(event.get("details", {}))
            sql_patterns = ["union select", "drop table", "script", "javascript:"]
            return any(pattern in suspicious_input.lower() for pattern in sql_patterns)

        elif pattern_id == "data_exfiltration":
            return (event.get("action") == "export" and
                   event.get("details", {}).get("record_count", 0) > 1000)

        return False

    def _determine_response_actions(self, threats: List[Dict[str, Any]]) -> List[str]:
        """Determine appropriate response actions"""
        actions = []

        max_severity = max(self.threat_levels[t["severity"]] for t in threats)

        if max_severity >= self.threat_levels["critical"]:
            actions.extend([
                "Block IP address",
                "Disable user account",
                "Alert security team",
                "Initiate incident response"
            ])
        elif max_severity >= self.threat_levels["high"]:
            actions.extend([
                "Log security event",
                "Require additional authentication",
                "Alert administrator"
            ])
        elif max_severity >= self.threat_levels["medium"]:
            actions.extend([
                "Log security event",
                "Send user notification"
            ])

        return actions

    def get_incident_report(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Generate incident report"""
        incidents = self.incident_log

        if start_date and end_date:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            incidents = [
                inc for inc in incidents
                if start <= datetime.fromisoformat(inc["threats"][0]["detected_at"]) <= end
            ]

        severity_counts = defaultdict(int)
        for incident in incidents:
            severity_counts[incident["overall_severity"]] += 1

        return {
            "total_incidents": len(incidents),
            "severity_breakdown": dict(severity_counts),
            "recent_incidents": incidents[-10:],  # Last 10 incidents
            "blocked_ips": list(self.blocked_ips),
            "generated_at": datetime.now().isoformat()
        }
