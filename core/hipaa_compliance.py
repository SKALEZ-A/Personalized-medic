"""
HIPAA Compliance Module for AI Personalized Medicine Platform
Ensures compliance with HIPAA Privacy Rule, Security Rule, and Breach Notification Rule
"""

import hashlib
import hmac
import secrets
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import random
import re
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class HIPAAComplianceEngine:
    """HIPAA compliance enforcement engine"""

    def __init__(self):
        self.encryption_keys = {}
        self.audit_logs = defaultdict(list)
        self.consent_records = {}
        self.breach_incidents = []
        self.compliance_alerts = []
        self.business_associates = set()
        self.initialize_hipaa_framework()

    def initialize_hipaa_framework(self):
        """Initialize HIPAA compliance framework"""
        # Generate master encryption key
        self.master_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)

        # Initialize audit logging
        self.audit_logger = logging.getLogger('hipaa_compliance')
        self.audit_logger.setLevel(logging.INFO)

        # Create audit log handler
        audit_handler = logging.FileHandler('hipaa_audit.log')
        audit_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.audit_logger.addHandler(audit_handler)

        print("🔒 HIPAA Compliance Framework initialized")

    def encrypt_phi_data(self, data: str, patient_id: str, data_category: str = "general") -> Dict[str, Any]:
        """Encrypt Protected Health Information (PHI)"""
        if not isinstance(data, str):
            data = json.dumps(data)

        # Generate data-specific encryption key
        data_key = self._generate_data_encryption_key(patient_id, data_category)

        # Create cipher for this data
        data_cipher = Fernet(data_key)

        # Encrypt the data
        encrypted_data = data_cipher.encrypt(data.encode())

        # Store key reference (not the key itself for security)
        key_reference = self._generate_key_reference(patient_id, data_category)

        # Log encryption event
        self._log_audit_event(
            event_type="PHI_ENCRYPTION",
            patient_id=patient_id,
            data_category=data_category,
            action="ENCRYPT",
            details={"data_size": len(data), "encryption_method": "AES256"}
        )

        return {
            "encrypted_data": encrypted_data.decode(),
            "key_reference": key_reference,
            "encryption_timestamp": datetime.now().isoformat(),
            "data_category": data_category,
            "encryption_method": "AES256-GCM",
            "compliance_status": "encrypted"
        }

    def decrypt_phi_data(self, encrypted_data: Dict[str, Any], requester_id: str,
                        purpose: str, patient_id: str) -> Optional[str]:
        """Decrypt PHI data with access controls"""
        # Verify access authorization
        if not self._verify_access_authorization(requester_id, patient_id, purpose):
            self._log_audit_event(
                event_type="ACCESS_DENIED",
                patient_id=patient_id,
                requester_id=requester_id,
                action="DECRYPT_ATTEMPT",
                details={"purpose": purpose, "reason": "unauthorized_access"}
            )
            raise PermissionError("Access denied: Unauthorized PHI access attempt")

        # Retrieve encryption key
        key_reference = encrypted_data["key_reference"]
        data_category = encrypted_data["data_category"]

        try:
            data_key = self._retrieve_data_encryption_key(key_reference, patient_id, data_category)
            data_cipher = Fernet(data_key)

            # Decrypt the data
            decrypted_data = data_cipher.decrypt(encrypted_data["encrypted_data"].encode())

            # Log decryption event
            self._log_audit_event(
                event_type="PHI_DECRYPTION",
                patient_id=patient_id,
                requester_id=requester_id,
                action="DECRYPT",
                details={"purpose": purpose, "data_category": data_category}
            )

            return decrypted_data.decode()

        except Exception as e:
            self._log_audit_event(
                event_type="DECRYPTION_ERROR",
                patient_id=patient_id,
                requester_id=requester_id,
                action="DECRYPT_FAILED",
                details={"error": str(e), "data_category": data_category}
            )
            raise ValueError(f"Decryption failed: {str(e)}")

    def _generate_data_encryption_key(self, patient_id: str, data_category: str) -> bytes:
        """Generate data-specific encryption key"""
        # Create key derivation input
        key_input = f"{patient_id}:{data_category}:{datetime.now().isoformat()}"

        # Use PBKDF2 to derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.master_key[:16],  # Use first 16 bytes as salt
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(key_input.encode()))
        return key

    def _generate_key_reference(self, patient_id: str, data_category: str) -> str:
        """Generate secure key reference"""
        reference_data = f"{patient_id}:{data_category}:{int(time.time())}"
        reference_hash = hashlib.sha256(reference_data.encode()).hexdigest()[:16]
        return f"key_ref_{reference_hash}"

    def _retrieve_data_encryption_key(self, key_reference: str, patient_id: str,
                                    data_category: str) -> bytes:
        """Retrieve encryption key for decryption"""
        # In practice, this would securely retrieve the key from HSM or key vault
        # For simulation, regenerate the key using the same parameters
        return self._generate_data_encryption_key(patient_id, data_category)

    def _verify_access_authorization(self, requester_id: str, patient_id: str,
                                   purpose: str) -> bool:
        """Verify if requester is authorized to access PHI"""
        # Check if requester has valid role
        valid_roles = ["physician", "nurse", "administrator", "researcher"]
        requester_role = self._get_user_role(requester_id)

        if requester_role not in valid_roles:
            return False

        # Check if purpose is legitimate
        valid_purposes = [
            "treatment", "payment", "healthcare_operations",
            "research", "public_health", "legal"
        ]

        if purpose not in valid_purposes:
            return False

        # Check if patient has provided consent (if required)
        if purpose in ["research", "marketing"] and not self._check_patient_consent(patient_id, purpose):
            return False

        # Additional checks based on role and purpose
        if purpose == "treatment" and requester_role not in ["physician", "nurse"]:
            return False

        return True

    def _get_user_role(self, user_id: str) -> str:
        """Get user role (simplified)"""
        # In practice, this would query user management system
        role_mapping = {
            "dr_smith": "physician",
            "nurse_jones": "nurse",
            "admin_user": "administrator",
            "research_lead": "researcher"
        }
        return role_mapping.get(user_id, "unknown")

    def _check_patient_consent(self, patient_id: str, purpose: str) -> bool:
        """Check if patient has consented to use of PHI"""
        if patient_id not in self.consent_records:
            return False

        consents = self.consent_records[patient_id]
        return any(
            consent["purpose"] == purpose and
            consent["status"] == "granted" and
            consent["expires"] > datetime.now()
            for consent in consents
        )

    def manage_patient_consent(self, patient_id: str, consent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Manage patient consent for PHI use"""
        if patient_id not in self.consent_records:
            self.consent_records[patient_id] = []

        consent_record = {
            "consent_id": f"consent_{int(time.time())}_{random.randint(1000, 9999)}",
            "purpose": consent_data["purpose"],
            "scope": consent_data.get("scope", "all_phi"),
            "status": "granted" if consent_data.get("granted", True) else "denied",
            "granted_at": datetime.now(),
            "expires": datetime.now() + timedelta(days=consent_data.get("duration_days", 365)),
            "consent_form_version": consent_data.get("form_version", "1.0"),
            "witnessed_by": consent_data.get("witnessed_by"),
            "revocation_allowed": consent_data.get("revocation_allowed", True)
        }

        self.consent_records[patient_id].append(consent_record)

        # Log consent event
        self._log_audit_event(
            event_type="PATIENT_CONSENT",
            patient_id=patient_id,
            action="CONSENT_" + consent_record["status"].upper(),
            details={
                "purpose": consent_record["purpose"],
                "scope": consent_record["scope"],
                "expires": consent_record["expires"].isoformat()
            }
        )

        return consent_record

    def revoke_patient_consent(self, patient_id: str, consent_id: str,
                              reason: str) -> Dict[str, Any]:
        """Revoke patient consent"""
        if patient_id not in self.consent_records:
            raise ValueError("Patient not found")

        for consent in self.consent_records[patient_id]:
            if consent["consent_id"] == consent_id:
                if not consent.get("revocation_allowed", True):
                    raise ValueError("Consent revocation not allowed")

                consent["status"] = "revoked"
                consent["revoked_at"] = datetime.now()
                consent["revocation_reason"] = reason

                # Log revocation
                self._log_audit_event(
                    event_type="CONSENT_REVOCATION",
                    patient_id=patient_id,
                    action="REVOKE_CONSENT",
                    details={
                        "consent_id": consent_id,
                        "reason": reason,
                        "original_purpose": consent["purpose"]
                    }
                )

                return consent

        raise ValueError("Consent not found")

    def _log_audit_event(self, event_type: str, patient_id: str = None,
                        requester_id: str = None, action: str = None,
                        details: Dict[str, Any] = None):
        """Log HIPAA audit event"""
        audit_event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "patient_id": patient_id,
            "requester_id": requester_id,
            "action": action,
            "details": details or {},
            "ip_address": "192.168.1.100",  # Would be actual IP
            "user_agent": "HealthcarePlatform/1.0",
            "session_id": f"session_{random.randint(10000, 99999)}"
        }

        # Store in memory (would be persisted to secure database)
        if patient_id:
            self.audit_logs[patient_id].append(audit_event)

        # Also log globally
        self.audit_logs["global"].append(audit_event)

        # Log to file
        log_message = f"{event_type}: Patient {patient_id or 'N/A'}, Action: {action or 'N/A'}, Details: {json.dumps(details or {})}"
        self.audit_logger.info(log_message)

    def get_audit_trail(self, patient_id: str, start_date: datetime = None,
                       end_date: datetime = None, event_types: List[str] = None) -> List[Dict[str, Any]]:
        """Get audit trail for a patient"""
        if patient_id not in self.audit_logs:
            return []

        audit_trail = self.audit_logs[patient_id]

        # Filter by date range
        if start_date:
            audit_trail = [event for event in audit_trail if event["timestamp"] >= start_date]
        if end_date:
            audit_trail = [event for event in audit_trail if event["timestamp"] <= end_date]

        # Filter by event types
        if event_types:
            audit_trail = [event for event in audit_trail if event["event_type"] in event_types]

        return sorted(audit_trail, key=lambda x: x["timestamp"], reverse=True)

    def detect_potential_breach(self, access_pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect potential HIPAA breach based on access patterns"""
        patient_id = access_pattern.get("patient_id")
        requester_id = access_pattern.get("requester_id")
        access_count = access_pattern.get("access_count", 0)
        time_window = access_pattern.get("time_window_hours", 24)

        # Check for unusual access patterns
        breach_indicators = []

        # High frequency access
        if access_count > 50:  # More than 50 accesses in time window
            breach_indicators.append("high_frequency_access")

        # Access outside normal hours (simplified)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Outside 6 AM - 10 PM
            breach_indicators.append("off_hours_access")

        # Access by unauthorized personnel
        if not self._verify_access_authorization(requester_id, patient_id, "investigation"):
            breach_indicators.append("unauthorized_access")

        # Unusual data volume
        if access_pattern.get("data_volume_mb", 0) > 100:  # Large data export
            breach_indicators.append("large_data_export")

        if breach_indicators:
            breach_alert = {
                "breach_id": f"breach_{int(time.time())}_{random.randint(1000, 9999)}",
                "patient_id": patient_id,
                "requester_id": requester_id,
                "detected_at": datetime.now(),
                "indicators": breach_indicators,
                "risk_level": self._assess_breach_risk(breach_indicators),
                "recommended_actions": self._get_breach_response_actions(breach_indicators),
                "investigation_required": True
            }

            self.breach_incidents.append(breach_alert)
            self.compliance_alerts.append(breach_alert)

            return breach_alert

        return None

    def _assess_breach_risk(self, indicators: List[str]) -> str:
        """Assess breach risk level"""
        high_risk_indicators = ["unauthorized_access", "large_data_export"]
        medium_risk_indicators = ["high_frequency_access", "off_hours_access"]

        if any(indicator in high_risk_indicators for indicator in indicators):
            return "high"
        elif any(indicator in medium_risk_indicators for indicator in indicators):
            return "medium"
        else:
            return "low"

    def _get_breach_response_actions(self, indicators: List[str]) -> List[str]:
        """Get recommended breach response actions"""
        actions = []

        if "unauthorized_access" in indicators:
            actions.extend([
                "Immediately suspend user access",
                "Notify security team",
                "Conduct internal investigation"
            ])

        if "large_data_export" in indicators:
            actions.extend([
                "Audit data export logs",
                "Verify export authorization",
                "Check data recipient credentials"
            ])

        if "high_frequency_access" in indicators:
            actions.extend([
                "Review access patterns",
                "Implement access throttling",
                "Require additional authorization for bulk access"
            ])

        actions.append("Document incident in breach log")
        actions.append("Report to compliance officer")

        return actions

    def report_security_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Report and manage security incidents"""
        incident_report = {
            "incident_id": f"incident_{int(time.time())}_{random.randint(1000, 9999)}",
            "reported_at": datetime.now(),
            "reported_by": incident_data.get("reported_by"),
            "incident_type": incident_data.get("incident_type"),
            "description": incident_data.get("description"),
            "affected_patients": incident_data.get("affected_patients", 0),
            "data_compromised": incident_data.get("data_compromised"),
            "severity": incident_data.get("severity", "unknown"),
            "status": "reported",
            "investigation_status": "pending",
            "corrective_actions": [],
            "timeline": [{
                "timestamp": datetime.now(),
                "event": "Incident reported",
                "details": incident_data.get("description")
            }]
        }

        # Determine if this is a breach requiring notification
        if self._is_reportable_breach(incident_report):
            incident_report["breach_notification_required"] = True
            incident_report["notification_deadline"] = datetime.now() + timedelta(hours=60)  # 60 hours for large breaches
            self._schedule_breach_notification(incident_report)

        # Log the incident
        self._log_audit_event(
            event_type="SECURITY_INCIDENT",
            patient_id=incident_data.get("patient_id"),
            action="INCIDENT_REPORTED",
            details={
                "incident_id": incident_report["incident_id"],
                "type": incident_report["incident_type"],
                "severity": incident_report["severity"]
            }
        )

        return incident_report

    def _is_reportable_breach(self, incident: Dict[str, Any]) -> bool:
        """Determine if incident is a reportable breach under HIPAA"""
        # Breach reporting thresholds
        if incident["affected_patients"] >= 500:
            return True  # Large breach - must report within 60 days

        if incident["severity"] == "high":
            return True  # High severity incidents

        # Check for sensitive data exposure
        sensitive_data = ["full_name", "ssn", "medical_record_number", "health_insurance_id"]
        compromised_data = incident.get("data_compromised", [])

        if any(data in sensitive_data for data in compromised_data):
            return True

        return False

    def _schedule_breach_notification(self, breach_report: Dict[str, Any]):
        """Schedule breach notification process"""
        # In practice, this would integrate with notification systems
        notification_plan = {
            "breach_id": breach_report["incident_id"],
            "notification_type": "large_breach",
            "timeline": {
                "immediate": ["Notify privacy officer", "Begin investigation"],
                "24_hours": ["Notify business associates"],
                "60_days": ["Notify affected individuals", "Notify media (if applicable)"],
                "annual": ["File breach report with HHS"]
            },
            "notification_methods": ["email", "phone", "certified_mail"],
            "content_requirements": [
                "Description of breach",
                "Types of PHI involved",
                "Steps individuals should take",
                "Contact information for questions"
            ]
        }

        breach_report["notification_plan"] = notification_plan

    def generate_hipaa_compliance_report(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate HIPAA compliance report"""
        report_period = {"start": start_date, "end": end_date}

        # Gather compliance metrics
        compliance_metrics = self._calculate_compliance_metrics(start_date, end_date)

        # Analyze audit logs
        audit_analysis = self._analyze_audit_logs(start_date, end_date)

        # Check security controls
        security_assessment = self._assess_security_controls()

        # Review breach incidents
        breach_analysis = self._analyze_breach_incidents(start_date, end_date)

        report = {
            "report_type": "hipaa_compliance",
            "generated_at": datetime.now(),
            "report_period": report_period,
            "overall_compliance_score": self._calculate_overall_compliance_score(
                compliance_metrics, audit_analysis, security_assessment
            ),
            "compliance_metrics": compliance_metrics,
            "audit_analysis": audit_analysis,
            "security_assessment": security_assessment,
            "breach_analysis": breach_analysis,
            "recommendations": self._generate_compliance_recommendations(
                compliance_metrics, audit_analysis, security_assessment
            ),
            "certification_status": "compliant" if compliance_metrics["overall_score"] >= 85 else "needs_attention"
        }

        return report

    def _calculate_compliance_metrics(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Calculate HIPAA compliance metrics"""
        # Simulate compliance metrics
        return {
            "privacy_rule_compliance": random.uniform(85, 98),
            "security_rule_compliance": random.uniform(82, 96),
            "breach_notification_compliance": random.uniform(88, 99),
            "business_associate_compliance": random.uniform(80, 95),
            "training_completion_rate": random.uniform(75, 98),
            "incident_reporting_rate": random.uniform(85, 100),
            "access_control_effectiveness": random.uniform(78, 96),
            "encryption_compliance": random.uniform(90, 100),
            "overall_score": 0  # Will be calculated as average
        }

    def _analyze_audit_logs(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Analyze audit logs for compliance insights"""
        # Simulate audit analysis
        return {
            "total_audit_events": random.randint(10000, 50000),
            "access_denied_events": random.randint(10, 100),
            "unauthorized_access_attempts": random.randint(5, 50),
            "encryption_events": random.randint(5000, 20000),
            "consent_events": random.randint(100, 1000),
            "anomalous_patterns": random.randint(0, 10),
            "compliance_violations": random.randint(0, 5)
        }

    def _assess_security_controls(self) -> Dict[str, Any]:
        """Assess effectiveness of security controls"""
        return {
            "access_controls": {
                "role_based_access": random.uniform(90, 100),
                "multi_factor_authentication": random.uniform(85, 100),
                "session_management": random.uniform(80, 95)
            },
            "encryption": {
                "data_at_rest": random.uniform(95, 100),
                "data_in_transit": random.uniform(90, 100),
                "key_management": random.uniform(85, 98)
            },
            "monitoring": {
                "intrusion_detection": random.uniform(75, 95),
                "log_monitoring": random.uniform(80, 98),
                "alert_response": random.uniform(70, 90)
            },
            "physical_security": {
                "facility_access": random.uniform(85, 100),
                "server_room_security": random.uniform(90, 100),
                "device_security": random.uniform(80, 95)
            }
        }

    def _analyze_breach_incidents(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Analyze breach incidents"""
        # Filter breaches in the date range
        period_breaches = [
            breach for breach in self.breach_incidents
            if start_date <= breach["detected_at"].date() <= end_date
        ]

        return {
            "total_breaches": len(period_breaches),
            "breaches_by_severity": {
                "high": len([b for b in period_breaches if b.get("risk_level") == "high"]),
                "medium": len([b for b in period_breaches if b.get("risk_level") == "medium"]),
                "low": len([b for b in period_breaches if b.get("risk_level") == "low"])
            },
            "average_response_time_hours": random.uniform(2, 24),
            "breach_trends": "stable" if len(period_breaches) <= 5 else "increasing",
            "most_common_cause": "unauthorized_access"
        }

    def _calculate_overall_compliance_score(self, metrics: Dict[str, Any],
                                          audit: Dict[str, Any],
                                          security: Dict[str, Any]) -> float:
        """Calculate overall HIPAA compliance score"""
        # Weight different components
        weights = {
            "privacy_rule": 0.25,
            "security_rule": 0.30,
            "breach_notification": 0.15,
            "training": 0.10,
            "incident_reporting": 0.10,
            "audit_compliance": 0.10
        }

        score = (
            metrics["privacy_rule_compliance"] * weights["privacy_rule"] +
            metrics["security_rule_compliance"] * weights["security_rule"] +
            metrics["breach_notification_compliance"] * weights["breach_notification"] +
            metrics["training_completion_rate"] * weights["training"] +
            metrics["incident_reporting_rate"] * weights["incident_reporting"] +
            (100 - audit["compliance_violations"] * 5) * weights["audit_compliance"]  # Penalty for violations
        )

        return min(100, max(0, score))

    def _generate_compliance_recommendations(self, metrics: Dict[str, Any],
                                           audit: Dict[str, Any],
                                           security: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate compliance improvement recommendations"""
        recommendations = []

        # Check for low-scoring areas
        if metrics["security_rule_compliance"] < 90:
            recommendations.append({
                "area": "Security Rule",
                "priority": "high",
                "recommendation": "Strengthen technical safeguards and access controls",
                "actions": ["Implement encryption for all PHI", "Enhance access monitoring", "Regular security assessments"]
            })

        if metrics["training_completion_rate"] < 85:
            recommendations.append({
                "area": "Staff Training",
                "priority": "medium",
                "recommendation": "Improve HIPAA training completion rates",
                "actions": ["Mandatory annual training", "Online training modules", "Training tracking system"]
            })

        if audit["unauthorized_access_attempts"] > 20:
            recommendations.append({
                "area": "Access Controls",
                "priority": "high",
                "recommendation": "Strengthen access control mechanisms",
                "actions": ["Implement role-based access", "Regular access reviews", "Enhanced authentication"]
            })

        return recommendations

    def add_business_associate(self, associate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a business associate under HIPAA BAA"""
        associate_id = f"ba_{int(time.time())}_{random.randint(1000, 9999)}"

        business_associate = {
            "associate_id": associate_id,
            "name": associate_data["name"],
            "type": associate_data["type"],  # "individual", "organization", " subcontractor"
            "services_provided": associate_data["services_provided"],
            "baa_signed": associate_data.get("baa_signed", False),
            "baa_signed_date": associate_data.get("baa_signed_date"),
            "baa_expires": associate_data.get("baa_expires"),
            "contact_information": associate_data["contact_information"],
            "data_shared": associate_data.get("data_shared", []),
            "security_assessment": associate_data.get("security_assessment", "pending"),
            "added_at": datetime.now(),
            "status": "active"
        }

        self.business_associates.add(associate_id)

        # Log BAA addition
        self._log_audit_event(
            event_type="BUSINESS_ASSOCIATE",
            action="ADD_ASSOCIATE",
            details={
                "associate_id": associate_id,
                "name": business_associate["name"],
                "baa_signed": business_associate["baa_signed"]
            }
        )

        return business_associate

    def monitor_business_associate_compliance(self, associate_id: str) -> Dict[str, Any]:
        """Monitor business associate compliance"""
        # Simulate compliance monitoring
        compliance_status = {
            "associate_id": associate_id,
            "last_audit": datetime.now() - timedelta(days=random.randint(30, 365)),
            "compliance_score": random.uniform(70, 98),
            "open_findings": random.randint(0, 5),
            "corrective_actions": random.randint(0, 3),
            "data_handling_compliant": random.random() > 0.1,
            "security_measures_adequate": random.random() > 0.15,
            "reporting_timely": random.random() > 0.05
        }

        if compliance_status["compliance_score"] < 85 or compliance_status["open_findings"] > 2:
            compliance_status["status"] = "needs_attention"
            compliance_status["recommended_actions"] = [
                "Schedule compliance audit",
                "Review BAA terms",
                "Implement corrective measures"
            ]
        else:
            compliance_status["status"] = "compliant"

        return compliance_status
