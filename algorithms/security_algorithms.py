"""
Advanced Security Algorithms for AI Personalized Medicine Platform
Comprehensive security, authentication, encryption, and privacy algorithms
"""

import math
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
import json
import hashlib
import hmac
import secrets
import base64
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import queue
import re


@dataclass
class SecurityMetrics:
    """Security and authentication metrics"""
    authentication_attempts: int = 0
    successful_authentications: int = 0
    failed_authentications: int = 0
    suspicious_activities: int = 0
    blocked_attempts: int = 0
    encryption_operations: int = 0
    decryption_operations: int = 0
    key_rotations: int = 0
    audit_events: int = 0
    last_security_check: datetime = None
    security_score: float = 0.0

    def __post_init__(self):
        if self.last_security_check is None:
            self.last_security_check = datetime.now()


@dataclass
class UserCredentials:
    """User authentication credentials"""
    user_id: str
    username: str
    password_hash: str
    salt: str
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)


@dataclass
class SecurityEvent:
    """Security audit event"""
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: str
    action: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"


class BaseSecurityAlgorithm(ABC):
    """Abstract base class for security algorithms"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False
        self.metrics = SecurityMetrics()
        self.audit_log = deque(maxlen=10000)  # Rolling audit log

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the security algorithm"""
        pass

    @abstractmethod
    def validate_operation(self, operation: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a security operation"""
        pass

    def log_security_event(self, event: SecurityEvent):
        """Log a security event"""
        self.audit_log.append(event)
        self.metrics.audit_events += 1

    def get_metrics(self) -> SecurityMetrics:
        """Get security metrics"""
        return self.metrics


class AdvancedEncryptionAlgorithm(BaseSecurityAlgorithm):
    """Advanced encryption algorithms with key management"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.encryption_keys = {}
        self.key_versions = {}
        self.key_rotation_schedule = {}
        self.algorithm = self.config.get('algorithm', 'AES256')
        self.key_size = self.config.get('key_size', 256)

    def initialize(self) -> bool:
        """Initialize encryption system"""
        try:
            # Generate master key
            self.master_key = self._generate_key(self.key_size)
            self.encryption_keys['master'] = self.master_key
            self.key_versions['master'] = 1

            # Initialize key rotation schedule
            self._setup_key_rotation()

            self.is_initialized = True
            return True
        except Exception as e:
            self.log_security_event(SecurityEvent(
                event_id=f"init_{int(time.time())}",
                timestamp=datetime.now(),
                event_type="encryption_initialization",
                status="failed",
                details={"error": str(e)}
            ))
            return False

    def validate_operation(self, operation: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate encryption operation"""
        if operation not in ['encrypt', 'decrypt', 'key_rotate']:
            return False, "Unsupported encryption operation"

        if not self.is_initialized:
            return False, "Encryption system not initialized"

        return True, "Operation validated"

    def encrypt_data(self, data: bytes, key_id: str = 'master') -> Tuple[bytes, str]:
        """Encrypt data using specified key"""
        if not self.is_initialized:
            raise ValueError("Encryption system not initialized")

        key = self._get_current_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found")

        # Generate initialization vector
        iv = secrets.token_bytes(16)

        # Encrypt data (simplified - in practice would use proper AES)
        encrypted_data = self._aes_encrypt(data, key, iv)

        # Create encrypted package
        encrypted_package = {
            'algorithm': self.algorithm,
            'key_version': self.key_versions.get(key_id, 1),
            'iv': base64.b64encode(iv).decode(),
            'data': base64.b64encode(encrypted_data).decode(),
            'timestamp': datetime.now().isoformat()
        }

        self.metrics.encryption_operations += 1

        return json.dumps(encrypted_package).encode(), key_id

    def decrypt_data(self, encrypted_data: bytes, key_id: str = 'master') -> bytes:
        """Decrypt data using specified key"""
        if not self.is_initialized:
            raise ValueError("Encryption system not initialized")

        # Parse encrypted package
        try:
            package = json.loads(encrypted_data.decode())
        except:
            raise ValueError("Invalid encrypted data format")

        key_version = package.get('key_version', 1)
        key = self._get_key_version(key_id, key_version)
        if not key:
            raise ValueError(f"Key {key_id} version {key_version} not found")

        iv = base64.b64decode(package['iv'])
        encrypted_bytes = base64.b64decode(package['data'])

        # Decrypt data (simplified)
        decrypted_data = self._aes_decrypt(encrypted_bytes, key, iv)

        self.metrics.decryption_operations += 1

        return decrypted_data

    def rotate_key(self, key_id: str) -> bool:
        """Rotate encryption key"""
        if key_id not in self.encryption_keys:
            return False

        # Generate new key
        new_key = self._generate_key(self.key_size)

        # Update key version
        current_version = self.key_versions.get(key_id, 1)
        self.key_versions[key_id] = current_version + 1

        # Store old key for decryption of existing data
        old_key_id = f"{key_id}_v{current_version}"
        self.encryption_keys[old_key_id] = self.encryption_keys[key_id]

        # Set new key as current
        self.encryption_keys[key_id] = new_key

        # Schedule cleanup of old key
        cleanup_time = datetime.now() + timedelta(days=30)  # Keep old keys for 30 days
        self.key_rotation_schedule[old_key_id] = cleanup_time

        self.metrics.key_rotations += 1

        self.log_security_event(SecurityEvent(
            event_id=f"key_rotate_{int(time.time())}",
            timestamp=datetime.now(),
            event_type="key_rotation",
            resource=key_id,
            action="rotate",
            status="success",
            details={"new_version": current_version + 1}
        ))

        return True

    def _generate_key(self, size: int) -> bytes:
        """Generate cryptographic key"""
        return secrets.token_bytes(size // 8)

    def _get_current_key(self, key_id: str) -> Optional[bytes]:
        """Get current key for encryption"""
        return self.encryption_keys.get(key_id)

    def _get_key_version(self, key_id: str, version: int) -> Optional[bytes]:
        """Get specific version of key"""
        if version == self.key_versions.get(key_id, 1):
            return self.encryption_keys.get(key_id)
        else:
            return self.encryption_keys.get(f"{key_id}_v{version}")

    def _aes_encrypt(self, data: bytes, key: bytes, iv: bytes) -> bytes:
        """Simplified AES encryption (in practice, use cryptography library)"""
        # This is a placeholder - real implementation would use AES
        # For demonstration, we'll use a simple XOR cipher
        result = bytearray()
        key_len = len(key)
        iv_len = len(iv)

        for i, byte in enumerate(data):
            key_byte = key[i % key_len]
            iv_byte = iv[i % iv_len]
            result.append(byte ^ key_byte ^ iv_byte)

        return bytes(result)

    def _aes_decrypt(self, data: bytes, key: bytes, iv: bytes) -> bytes:
        """Simplified AES decryption"""
        # XOR is symmetric
        return self._aes_encrypt(data, key, iv)

    def _setup_key_rotation(self):
        """Setup automatic key rotation"""
        # In a real system, this would schedule periodic key rotation
        pass


class MultiFactorAuthenticationAlgorithm(BaseSecurityAlgorithm):
    """Multi-factor authentication implementation"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.totp_secrets = {}
        self.backup_codes = {}
        self.trusted_devices = defaultdict(set)
        self.session_tokens = {}
        self.max_attempts = self.config.get('max_attempts', 3)
        self.lockout_duration = self.config.get('lockout_duration', 900)  # 15 minutes

    def initialize(self) -> bool:
        """Initialize MFA system"""
        self.is_initialized = True
        return True

    def validate_operation(self, operation: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate MFA operation"""
        valid_operations = ['setup', 'verify', 'disable', 'backup_verify']
        if operation not in valid_operations:
            return False, f"Unsupported MFA operation: {operation}"

        return True, "Operation validated"

    def setup_mfa(self, user_id: str) -> Dict[str, Any]:
        """Setup MFA for user"""
        # Generate TOTP secret
        secret = self._generate_totp_secret()

        # Generate backup codes
        backup_codes = [self._generate_backup_code() for _ in range(10)]

        # Store secrets
        self.totp_secrets[user_id] = secret
        self.backup_codes[user_id] = backup_codes

        return {
            'secret': secret,
            'qr_code_url': self._generate_qr_code_url(user_id, secret),
            'backup_codes': backup_codes.copy()  # Return copy for user
        }

    def verify_mfa(self, user_id: str, code: str, method: str = 'totp') -> bool:
        """Verify MFA code"""
        if method == 'totp':
            return self._verify_totp(user_id, code)
        elif method == 'backup':
            return self._verify_backup_code(user_id, code)
        else:
            return False

    def disable_mfa(self, user_id: str) -> bool:
        """Disable MFA for user"""
        if user_id in self.totp_secrets:
            del self.totp_secrets[user_id]
        if user_id in self.backup_codes:
            del self.backup_codes[user_id]
        if user_id in self.trusted_devices:
            del self.trusted_devices[user_id]

        return True

    def _generate_totp_secret(self) -> str:
        """Generate TOTP secret"""
        # Generate 32-character base32 secret
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'
        return ''.join(random.choice(alphabet) for _ in range(32))

    def _generate_backup_code(self) -> str:
        """Generate backup code"""
        return ''.join(random.choice('0123456789') for _ in range(8))

    def _generate_qr_code_url(self, user_id: str, secret: str) -> str:
        """Generate QR code URL for TOTP setup"""
        # otpauth://totp/Issuer:Account?secret=Secret&issuer=Issuer
        issuer = "AI_Personalized_Medicine"
        account = user_id
        return f"otpauth://totp/{issuer}:{account}?secret={secret}&issuer={issuer}"

    def _verify_totp(self, user_id: str, code: str) -> bool:
        """Verify TOTP code"""
        if user_id not in self.totp_secrets:
            return False

        secret = self.totp_secrets[user_id]

        # Generate current and adjacent TOTP codes
        current_time = int(time.time() // 30)  # 30-second windows

        for time_window in [current_time - 1, current_time, current_time + 1]:
            expected_code = self._generate_totp_code(secret, time_window)
            if expected_code == code:
                return True

        return False

    def _generate_totp_code(self, secret: str, time_window: int) -> str:
        """Generate TOTP code from secret and time window"""
        # Decode base32 secret
        secret_bytes = base64.b32decode(secret)

        # Create HMAC-SHA1 hash
        time_bytes = time_window.to_bytes(8, 'big')
        hmac_hash = hmac.new(secret_bytes, time_bytes, hashlib.sha1).digest()

        # Get offset from last byte
        offset = hmac_hash[-1] & 0x0F

        # Get 4 bytes starting from offset
        code_bytes = hmac_hash[offset:offset + 4]

        # Convert to integer
        code_int = int.from_bytes(code_bytes, 'big') & 0x7FFFFFFF

        # Get 6-digit code
        code = str(code_int % 1000000).zfill(6)

        return code

    def _verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup code"""
        if user_id not in self.backup_codes:
            return False

        if code in self.backup_codes[user_id]:
            # Remove used backup code
            self.backup_codes[user_id].remove(code)
            return True

        return False


class RoleBasedAccessControlAlgorithm(BaseSecurityAlgorithm):
    """Role-Based Access Control (RBAC) implementation"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.roles = {}
        self.permissions = {}
        self.user_roles = defaultdict(set)
        self.role_permissions = defaultdict(set)
        self.permission_hierarchy = {}
        self.session_cache = {}
        self.cache_timeout = self.config.get('cache_timeout', 300)  # 5 minutes

    def initialize(self) -> bool:
        """Initialize RBAC system"""
        # Define default roles and permissions
        self._setup_default_roles()
        self.is_initialized = True
        return True

    def validate_operation(self, operation: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate RBAC operation"""
        valid_operations = ['check_access', 'assign_role', 'revoke_role', 'create_role', 'delete_role']
        if operation not in valid_operations:
            return False, f"Unsupported RBAC operation: {operation}"

        return True, "Operation validated"

    def check_access(self, user_id: str, permission: str, resource: str = None) -> bool:
        """Check if user has permission"""
        # Check cache first
        cache_key = f"{user_id}:{permission}:{resource}"
        if cache_key in self.session_cache:
            cached_result, cache_time = self.session_cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=self.cache_timeout):
                return cached_result

        # Get user roles
        user_roles = self.user_roles.get(user_id, set())

        # Check direct permissions and role permissions
        has_permission = self._has_permission(user_roles, permission)

        # Cache result
        self.session_cache[cache_key] = (has_permission, datetime.now())

        return has_permission

    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user"""
        if role not in self.roles:
            return False

        self.user_roles[user_id].add(role)

        # Clear user cache
        self._clear_user_cache(user_id)

        self.log_security_event(SecurityEvent(
            event_id=f"role_assign_{int(time.time())}",
            timestamp=datetime.now(),
            event_type="role_assignment",
            user_id=user_id,
            resource=role,
            action="assign",
            status="success"
        ))

        return True

    def revoke_role(self, user_id: str, role: str) -> bool:
        """Revoke role from user"""
        if role in self.user_roles.get(user_id, set()):
            self.user_roles[user_id].remove(role)

            # Clear user cache
            self._clear_user_cache(user_id)

            self.log_security_event(SecurityEvent(
                event_id=f"role_revoke_{int(time.time())}",
                timestamp=datetime.now(),
                event_type="role_assignment",
                user_id=user_id,
                resource=role,
                action="revoke",
                status="success"
            ))

            return True

        return False

    def create_role(self, role_name: str, permissions: List[str]) -> bool:
        """Create new role with permissions"""
        if role_name in self.roles:
            return False

        self.roles[role_name] = {
            'name': role_name,
            'permissions': permissions,
            'created_at': datetime.now()
        }

        # Set role permissions
        self.role_permissions[role_name] = set(permissions)

        return True

    def delete_role(self, role_name: str) -> bool:
        """Delete role"""
        if role_name not in self.roles:
            return False

        # Remove role from all users
        for user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)

        # Remove role data
        del self.roles[role_name]
        del self.role_permissions[role_name]

        return True

    def _setup_default_roles(self):
        """Setup default healthcare roles and permissions"""
        roles_config = {
            'patient': [
                'read_own_health_data',
                'update_own_profile',
                'schedule_appointments',
                'view_own_reports'
            ],
            'physician': [
                'read_patient_health_data',
                'write_patient_health_data',
                'prescribe_medications',
                'order_tests',
                'view_all_reports'
            ],
            'nurse': [
                'read_patient_health_data',
                'update_patient_vitals',
                'administer_medications',
                'view_basic_reports'
            ],
            'pharmacist': [
                'read_medication_data',
                'dispense_medications',
                'check_interactions',
                'view_pharmacy_reports'
            ],
            'admin': [
                'manage_users',
                'manage_roles',
                'view_all_data',
                'generate_reports',
                'system_configuration'
            ],
            'researcher': [
                'read_anonymized_data',
                'conduct_studies',
                'generate_analytics',
                'export_research_data'
            ]
        }

        for role_name, permissions in roles_config.items():
            self.create_role(role_name, permissions)

    def _has_permission(self, user_roles: Set[str], permission: str) -> bool:
        """Check if user roles have the required permission"""
        for role in user_roles:
            if permission in self.role_permissions.get(role, set()):
                return True

        # Check permission hierarchy
        for role in user_roles:
            if self._check_permission_hierarchy(role, permission):
                return True

        return False

    def _check_permission_hierarchy(self, role: str, permission: str) -> bool:
        """Check permission hierarchy (e.g., admin inherits all permissions)"""
        hierarchy = self.permission_hierarchy.get(role, [])

        if permission in hierarchy:
            return True

        # Check inherited permissions recursively
        for inherited_perm in hierarchy:
            if inherited_perm in self.permissions:
                if permission in self.permissions[inherited_perm].get('children', []):
                    return True

        return False

    def _clear_user_cache(self, user_id: str):
        """Clear cache entries for user"""
        keys_to_remove = [key for key in self.session_cache.keys() if key.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.session_cache[key]


class IntrusionDetectionAlgorithm(BaseSecurityAlgorithm):
    """Intrusion Detection System for healthcare platform"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 3.0)
        self.block_threshold = self.config.get('block_threshold', 5)
        self.monitoring_window = self.config.get('monitoring_window', 3600)  # 1 hour
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.activity_log = defaultdict(list)
        self.blocked_ips = set()
        self.rate_limits = defaultdict(lambda: defaultdict(int))

    def initialize(self) -> bool:
        """Initialize IDS"""
        self.is_initialized = True
        return True

    def validate_operation(self, operation: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate IDS operation"""
        valid_operations = ['analyze_request', 'check_intrusion', 'block_ip', 'unblock_ip']
        if operation not in valid_operations:
            return False, f"Unsupported IDS operation: {operation}"

        return True, "Operation validated"

    def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze incoming request for suspicious activity"""
        ip_address = request_data.get('ip_address')
        user_id = request_data.get('user_id')
        user_agent = request_data.get('user_agent', '')
        endpoint = request_data.get('endpoint', '')
        method = request_data.get('method', '')

        risk_score = 0.0
        risk_factors = []

        # Check rate limiting
        if self._check_rate_limit(ip_address, endpoint):
            risk_score += 2.0
            risk_factors.append("rate_limit_exceeded")

        # Check suspicious patterns
        pattern_risk, pattern_factors = self._check_suspicious_patterns(request_data)
        risk_score += pattern_risk
        risk_factors.extend(pattern_factors)

        # Check for anomalous behavior
        anomaly_risk, anomaly_factors = self._detect_anomalous_behavior(user_id, ip_address, endpoint)
        risk_score += anomaly_risk
        risk_factors.extend(anomaly_factors)

        # Log activity
        self._log_activity(ip_address, user_id, endpoint, risk_score)

        # Determine action
        action = "allow"
        if risk_score >= self.block_threshold:
            action = "block"
            self.blocked_ips.add(ip_address)
            self.metrics.blocked_attempts += 1
        elif risk_score >= self.anomaly_threshold:
            action = "flag"
            self.metrics.suspicious_activities += 1

        result = {
            'action': action,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'ip_blocked': ip_address in self.blocked_ips
        }

        # Log security event
        if action != "allow":
            self.log_security_event(SecurityEvent(
                event_id=f"ids_{int(time.time())}",
                timestamp=datetime.now(),
                event_type="intrusion_detected",
                user_id=user_id,
                ip_address=ip_address,
                resource=endpoint,
                action=action,
                status="alert",
                severity="high" if action == "block" else "medium",
                details=result
            ))

        return result

    def _check_rate_limit(self, ip_address: str, endpoint: str) -> bool:
        """Check if request exceeds rate limits"""
        current_time = time.time()
        key = f"{ip_address}:{endpoint}"

        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute window
        self.rate_limits[key] = {
            timestamp: count
            for timestamp, count in self.rate_limits[key].items()
            if timestamp > cutoff_time
        }

        # Count requests in current minute
        current_minute = int(current_time // 60)
        request_count = self.rate_limits[key].get(current_minute, 0) + 1
        self.rate_limits[key][current_minute] = request_count

        # Check rate limits (30 requests per minute per endpoint)
        return request_count > 30

    def _check_suspicious_patterns(self, request_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check for suspicious patterns in request"""
        risk_score = 0.0
        risk_factors = []

        user_agent = request_data.get('user_agent', '').lower()
        endpoint = request_data.get('endpoint', '').lower()

        # Check for automated tools
        automated_tools = ['curl', 'wget', 'python', 'scrapy', 'selenium']
        if any(tool in user_agent for tool in automated_tools):
            risk_score += 1.5
            risk_factors.append("automated_tool_detected")

        # Check for SQL injection attempts
        sql_patterns = [r'union\s+select', r';\s*drop', r'--', r'/\*.*\*/']
        request_body = str(request_data.get('body', '')).lower()
        if any(re.search(pattern, request_body) for pattern in sql_patterns):
            risk_score += 2.0
            risk_factors.append("sql_injection_attempt")

        # Check for suspicious endpoints
        suspicious_endpoints = ['/admin', '/config', '/backup', '/debug']
        if any(suspicious in endpoint for suspicious in suspicious_endpoints):
            risk_score += 1.0
            risk_factors.append("suspicious_endpoint")

        return risk_score, risk_factors

    def _detect_anomalous_behavior(self, user_id: str, ip_address: str, endpoint: str) -> Tuple[float, List[str]]:
        """Detect anomalous user behavior"""
        risk_score = 0.0
        risk_factors = []

        # Get recent activity
        recent_activity = self._get_recent_activity(user_id, ip_address)

        if len(recent_activity) < 5:
            return risk_score, risk_factors  # Not enough data

        # Check for unusual endpoint access patterns
        endpoints = [activity['endpoint'] for activity in recent_activity]
        if len(set(endpoints)) > len(endpoints) * 0.8:  # High endpoint diversity
            risk_score += 1.0
            risk_factors.append("unusual_endpoint_access")

        # Check for rapid successive requests
        timestamps = [activity['timestamp'] for activity in recent_activity]
        if len(timestamps) > 1:
            time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_time_diff = sum(time_diffs) / len(time_diffs)
            if avg_time_diff < 1.0:  # Less than 1 second between requests
                risk_score += 1.5
                risk_factors.append("rapid_requests")

        return risk_score, risk_factors

    def _get_recent_activity(self, user_id: str, ip_address: str) -> List[Dict[str, Any]]:
        """Get recent activity for user/IP"""
        cutoff_time = datetime.now() - timedelta(seconds=self.monitoring_window)

        recent_activity = []
        for activity in self.activity_log.get(user_id, []) + self.activity_log.get(f"ip_{ip_address}", []):
            if activity['timestamp'] > cutoff_time:
                recent_activity.append(activity)

        return recent_activity[-20:]  # Last 20 activities

    def _log_activity(self, ip_address: str, user_id: str, endpoint: str, risk_score: float):
        """Log user activity"""
        activity = {
            'timestamp': datetime.now(),
            'endpoint': endpoint,
            'risk_score': risk_score
        }

        if user_id:
            self.activity_log[user_id].append(activity)
        if ip_address:
            self.activity_log[f"ip_{ip_address}"].append(activity)

        # Clean old activities
        cutoff_time = datetime.now() - timedelta(seconds=self.monitoring_window)
        for key in self.activity_log:
            self.activity_log[key] = [
                act for act in self.activity_log[key]
                if act['timestamp'] > cutoff_time
            ]

    def _load_suspicious_patterns(self) -> Dict[str, Any]:
        """Load suspicious activity patterns"""
        return {
            'user_agents': [
                'sqlmap', 'nmap', 'metasploit', 'burpsuite',
                'owasp', 'acunetix', 'qualys'
            ],
            'endpoints': [
                '/phpmyadmin', '/adminer', '/wp-admin', '/administrator',
                '/config', '/backup', '/.env', '/.git'
            ],
            'payloads': [
                '<script>', 'javascript:', 'data:', 'vbscript:',
                'onload=', 'onerror=', 'eval(', 'exec('
            ]
        }


class HealthcareSecurityAlgorithms:
    """Collection of security algorithms for healthcare platform"""

    def __init__(self):
        self.algorithms = {
            'encryption': AdvancedEncryptionAlgorithm,
            'mfa': MultiFactorAuthenticationAlgorithm,
            'rbac': RoleBasedAccessControlAlgorithm,
            'ids': IntrusionDetectionAlgorithm
        }
        self.active_instances = {}

    def create_security_system(self, system_type: str, config: Dict[str, Any] = None) -> str:
        """Create a security system instance"""

        if system_type not in self.algorithms:
            raise ValueError(f"Unsupported security system: {system_type}")

        instance_id = f"{system_type}_{int(time.time())}"
        instance = self.algorithms[system_type](config)

        if not instance.initialize():
            raise RuntimeError(f"Failed to initialize {system_type} security system")

        self.active_instances[instance_id] = instance
        return instance_id

    def execute_security_operation(self, instance_id: str, operation: str, context: Dict[str, Any]) -> Any:
        """Execute security operation"""
        if instance_id not in self.active_instances:
            raise ValueError(f"Security instance {instance_id} not found")

        instance = self.active_instances[instance_id]

        # Validate operation
        valid, error_msg = instance.validate_operation(operation, context)
        if not valid:
            raise ValueError(error_msg)

        # Execute operation
        if hasattr(instance, operation):
            method = getattr(instance, operation)
            return method(**context)
        else:
            raise ValueError(f"Operation {operation} not supported by {instance_id}")

    def get_security_metrics(self, instance_id: str) -> SecurityMetrics:
        """Get security metrics for instance"""
        if instance_id not in self.active_instances:
            raise ValueError(f"Security instance {instance_id} not found")

        return self.active_instances[instance_id].get_metrics()

    def get_audit_log(self, instance_id: str, limit: int = 100) -> List[SecurityEvent]:
        """Get audit log for security instance"""
        if instance_id not in self.active_instances:
            raise ValueError(f"Security instance {instance_id} not found")

        audit_log = list(self.active_instances[instance_id].audit_log)
        return audit_log[-limit:] if limit > 0 else audit_log

    def create_comprehensive_security_suite(self) -> Dict[str, str]:
        """Create a comprehensive security suite for healthcare"""

        suite_config = {
            'encryption': {
                'algorithm': 'AES256',
                'key_size': 256,
                'auto_rotate': True,
                'rotation_interval_days': 90
            },
            'mfa': {
                'max_attempts': 3,
                'lockout_duration': 900,
                'backup_codes_count': 10
            },
            'rbac': {
                'cache_timeout': 300,
                'default_roles': ['patient', 'physician', 'admin']
            },
            'ids': {
                'anomaly_threshold': 3.0,
                'block_threshold': 5.0,
                'monitoring_window': 3600
            }
        }

        suite_instances = {}
        for system_type, config in suite_config.items():
            try:
                instance_id = self.create_security_system(system_type, config)
                suite_instances[system_type] = instance_id
            except Exception as e:
                print(f"Failed to create {system_type} security system: {e}")

        return suite_instances

    def perform_security_health_check(self, suite_instances: Dict[str, str]) -> Dict[str, Any]:
        """Perform comprehensive security health check"""

        health_check = {
            'timestamp': datetime.now(),
            'overall_status': 'healthy',
            'systems_status': {},
            'recommendations': []
        }

        for system_type, instance_id in suite_instances.items():
            try:
                metrics = self.get_security_metrics(instance_id)

                # Check system health
                system_status = 'healthy'
                issues = []

                if hasattr(metrics, 'failed_authentications') and metrics.failed_authentications > 100:
                    system_status = 'warning'
                    issues.append("High number of failed authentications")

                if hasattr(metrics, 'suspicious_activities') and metrics.suspicious_activities > 50:
                    system_status = 'warning'
                    issues.append("High number of suspicious activities")

                if hasattr(metrics, 'blocked_attempts') and metrics.blocked_attempts > 20:
                    system_status = 'critical'
                    issues.append("High number of blocked attempts")

                health_check['systems_status'][system_type] = {
                    'status': system_status,
                    'issues': issues,
                    'metrics': {
                        'audit_events': getattr(metrics, 'audit_events', 0),
                        'last_check': getattr(metrics, 'last_security_check', None)
                    }
                }

                if system_status in ['warning', 'critical']:
                    health_check['overall_status'] = 'warning' if health_check['overall_status'] == 'healthy' else 'critical'

            except Exception as e:
                health_check['systems_status'][system_type] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_check['overall_status'] = 'critical'

        # Generate recommendations
        if health_check['overall_status'] != 'healthy':
            health_check['recommendations'] = [
                "Review security logs for suspicious activities",
                "Consider updating security policies",
                "Check for potential security breaches",
                "Verify MFA and access control configurations"
            ]

        return health_check
