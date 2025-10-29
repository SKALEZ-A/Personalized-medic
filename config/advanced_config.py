"""
Advanced Configuration Management for AI Personalized Medicine Platform
Comprehensive configuration system with environment management, feature flags, and dynamic settings
"""

import os
import json
import yaml
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
import time
from abc import ABC, abstractmethod
import secrets
import base64


@dataclass
class ConfigurationMetadata:
    """Metadata for configuration settings"""
    key: str
    value: Any
    data_type: str
    description: str
    category: str
    environment: str
    is_secret: bool
    last_modified: datetime
    modified_by: str
    version: int
    dependencies: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    name: str
    enabled: bool
    rollout_percentage: float
    conditions: Dict[str, Any]
    description: str
    created_at: datetime
    updated_at: datetime
    owner: str
    tags: List[str] = field(default_factory=list)


@dataclass
class ConfigurationProfile:
    """Configuration profile for different environments"""
    name: str
    environment: str
    settings: Dict[str, Any]
    feature_flags: Dict[str, FeatureFlag]
    secrets: Dict[str, str]
    created_at: datetime
    updated_at: datetime
    version: str
    checksum: str


class ConfigurationSource(ABC):
    """Abstract base class for configuration sources"""

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration from source"""
        pass

    @abstractmethod
    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration to source"""
        pass

    @abstractmethod
    def watch(self, callback: Callable[[Dict[str, Any]], None]):
        """Watch for configuration changes"""
        pass


class FileConfigurationSource(ConfigurationSource):
    """File-based configuration source"""

    def __init__(self, file_path: str, format: str = 'json'):
        self.file_path = Path(file_path)
        self.format = format
        self._last_modified = None
        self._config_hash = None

    def load(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.file_path.exists():
            return {}

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                if self.format == 'json':
                    return json.load(f)
                elif self.format == 'yaml':
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported format: {self.format}")
        except Exception as e:
            logging.error(f"Failed to load configuration from {self.file_path}: {e}")
            return {}

    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                if self.format == 'json':
                    json.dump(config, f, indent=2, default=str)
                elif self.format == 'yaml':
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported format: {self.format}")

            # Update metadata
            self._last_modified = datetime.now()
            self._config_hash = self._calculate_hash(config)

            return True
        except Exception as e:
            logging.error(f"Failed to save configuration to {self.file_path}: {e}")
            return False

    def watch(self, callback: Callable[[Dict[str, Any]], None]):
        """Watch for file changes"""
        def watch_thread():
            while True:
                try:
                    if self.file_path.exists():
                        current_modified = self.file_path.stat().st_mtime
                        current_config = self.load()
                        current_hash = self._calculate_hash(current_config)

                        if (self._last_modified is None or
                            current_modified > self._last_modified.timestamp() or
                            current_hash != self._config_hash):

                            self._last_modified = datetime.fromtimestamp(current_modified)
                            self._config_hash = current_hash
                            callback(current_config)

                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logging.error(f"Error watching configuration file: {e}")
                    time.sleep(10)

        thread = threading.Thread(target=watch_thread, daemon=True)
        thread.start()

    def _calculate_hash(self, config: Dict[str, Any]) -> str:
        """Calculate configuration hash"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()


class EnvironmentConfigurationSource(ConfigurationSource):
    """Environment variable configuration source"""

    def __init__(self, prefix: str = ''):
        self.prefix = prefix.upper()

    def load(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}

        for key, value in os.environ.items():
            if self.prefix and key.startswith(self.prefix):
                config_key = key[len(self.prefix):].lower() if self.prefix else key.lower()
                config[config_key] = self._parse_env_value(value)

        return config

    def save(self, config: Dict[str, Any]) -> bool:
        """Environment variables cannot be saved"""
        logging.warning("Cannot save to environment variables")
        return False

    def watch(self, callback: Callable[[Dict[str, Any]], None]):
        """Environment variables don't support watching"""
        pass

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except:
            pass

        # Try to parse as boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except:
            pass

        return value


class DatabaseConfigurationSource(ConfigurationSource):
    """Database-backed configuration source"""

    def __init__(self, connection_string: str, table_name: str = 'configurations'):
        self.connection_string = connection_string
        self.table_name = table_name
        self._cache = {}
        self._last_sync = None

    def load(self) -> Dict[str, Any]:
        """Load configuration from database"""
        # In a real implementation, this would connect to a database
        # For now, return cached configuration
        if self._cache and self._last_sync:
            # Check if cache is still valid (e.g., within 5 minutes)
            if datetime.now() - self._last_sync < timedelta(minutes=5):
                return self._cache.copy()

        # Mock database load
        self._cache = {
            'database_host': 'localhost',
            'database_port': 5432,
            'database_name': 'healthcare_db',
            'cache_enabled': True,
            'cache_ttl': 3600,
        }
        self._last_sync = datetime.now()

        return self._cache.copy()

    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration to database"""
        # Mock database save
        try:
            self._cache.update(config)
            self._last_sync = datetime.now()
            return True
        except Exception as e:
            logging.error(f"Failed to save configuration to database: {e}")
            return False

    def watch(self, callback: Callable[[Dict[str, Any]], None]):
        """Watch for database changes"""
        def watch_thread():
            while True:
                try:
                    current_config = self.load()
                    current_hash = self._calculate_hash(current_config)

                    if not hasattr(self, '_db_hash') or current_hash != self._db_hash:
                        self._db_hash = current_hash
                        callback(current_config)

                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logging.error(f"Error watching database configuration: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=watch_thread, daemon=True)
        thread.start()

    def _calculate_hash(self, config: Dict[str, Any]) -> str:
        """Calculate configuration hash"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()


class ConfigurationManager:
    """Main configuration management system"""

    def __init__(self):
        self.sources: List[ConfigurationSource] = []
        self.configurations: Dict[str, Any] = {}
        self.metadata: Dict[str, ConfigurationMetadata] = {}
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.profiles: Dict[str, ConfigurationProfile] = {}
        self.secrets: Dict[str, str] = {}
        self._listeners: List[Callable] = []
        self._lock = threading.RLock()

    def add_source(self, source: ConfigurationSource, priority: int = 0):
        """Add configuration source with priority"""
        self.sources.append((source, priority))
        self.sources.sort(key=lambda x: x[1], reverse=True)  # Higher priority first

    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration from all sources"""
        with self._lock:
            merged_config = {}

            # Load from all sources (higher priority overrides lower)
            for source, _ in self.sources:
                source_config = source.load()
                merged_config.update(source_config)

            self.configurations = merged_config

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(merged_config)
                except Exception as e:
                    logging.error(f"Configuration listener error: {e}")

            return merged_config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.configurations.get(key, default)

    def set(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> bool:
        """Set configuration value"""
        with self._lock:
            self.configurations[key] = value

            # Update metadata
            if metadata:
                existing_meta = self.metadata.get(key)
                version = (existing_meta.version + 1) if existing_meta else 1

                self.metadata[key] = ConfigurationMetadata(
                    key=key,
                    value=value,
                    data_type=type(value).__name__,
                    description=metadata.get('description', ''),
                    category=metadata.get('category', 'general'),
                    environment=metadata.get('environment', 'default'),
                    is_secret=metadata.get('is_secret', False),
                    last_modified=datetime.now(),
                    modified_by=metadata.get('modified_by', 'system'),
                    version=version,
                    dependencies=metadata.get('dependencies', []),
                    validation_rules=metadata.get('validation_rules', {}),
                    tags=metadata.get('tags', [])
                )

            # Save to primary source
            if self.sources:
                primary_source = self.sources[0][0]
                return primary_source.save(self.configurations)

            return True

    def validate_configuration(self, config: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """Validate configuration against rules"""
        config_to_validate = config or self.configurations
        errors = {}

        for key, metadata in self.metadata.items():
            if key not in config_to_validate:
                continue

            value = config_to_validate[key]
            key_errors = []

            # Type validation
            if metadata.data_type and type(value).__name__ != metadata.data_type:
                key_errors.append(f"Expected type {metadata.data_type}, got {type(value).__name__}")

            # Range validation
            if 'min' in metadata.validation_rules and isinstance(value, (int, float)):
                if value < metadata.validation_rules['min']:
                    key_errors.append(f"Value must be >= {metadata.validation_rules['min']}")

            if 'max' in metadata.validation_rules and isinstance(value, (int, float)):
                if value > metadata.validation_rules['max']:
                    key_errors.append(f"Value must be <= {metadata.validation_rules['max']}")

            # Pattern validation
            if 'pattern' in metadata.validation_rules and isinstance(value, str):
                import re
                if not re.match(metadata.validation_rules['pattern'], value):
                    key_errors.append("Value does not match required pattern")

            # Enum validation
            if 'enum' in metadata.validation_rules:
                if value not in metadata.validation_rules['enum']:
                    key_errors.append(f"Value must be one of: {metadata.validation_rules['enum']}")

            if key_errors:
                errors[key] = key_errors

        return errors

    def create_feature_flag(self, name: str, config: Dict[str, Any]) -> bool:
        """Create a feature flag"""
        with self._lock:
            if name in self.feature_flags:
                # Update existing flag
                existing = self.feature_flags[name]
                updated_flag = FeatureFlag(
                    name=name,
                    enabled=config.get('enabled', existing.enabled),
                    rollout_percentage=config.get('rollout_percentage', existing.rollout_percentage),
                    conditions=config.get('conditions', existing.conditions),
                    description=config.get('description', existing.description),
                    created_at=existing.created_at,
                    updated_at=datetime.now(),
                    owner=config.get('owner', existing.owner),
                    tags=config.get('tags', existing.tags)
                )
            else:
                # Create new flag
                updated_flag = FeatureFlag(
                    name=name,
                    enabled=config.get('enabled', False),
                    rollout_percentage=config.get('rollout_percentage', 0.0),
                    conditions=config.get('conditions', {}),
                    description=config.get('description', ''),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    owner=config.get('owner', 'system'),
                    tags=config.get('tags', [])
                )

            self.feature_flags[name] = updated_flag
            return True

    def is_feature_enabled(self, feature_name: str, context: Dict[str, Any] = None) -> bool:
        """Check if feature is enabled for given context"""
        flag = self.feature_flags.get(feature_name)
        if not flag:
            return False

        if not flag.enabled:
            return False

        context = context or {}

        # Check rollout percentage
        if flag.rollout_percentage < 1.0:
            # Use user ID or session ID for consistent rollout
            user_id = context.get('user_id', 'anonymous')
            rollout_key = f"{feature_name}:{user_id}"
            rollout_value = int(hashlib.md5(rollout_key.encode()).hexdigest(), 16) % 100
            if rollout_value / 100.0 >= flag.rollout_percentage:
                return False

        # Check conditions
        for condition_key, condition_value in flag.conditions.items():
            context_value = context.get(condition_key)
            if context_value != condition_value:
                return False

        return True

    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret value"""
        # In a real implementation, use proper encryption
        # For now, use base64 encoding
        return base64.b64encode(secret.encode()).decode()

    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt a secret value"""
        # In a real implementation, use proper decryption
        # For now, use base64 decoding
        return base64.b64decode(encrypted_secret.encode()).decode()

    def create_configuration_profile(self, name: str, environment: str) -> ConfigurationProfile:
        """Create a configuration profile"""
        profile = ConfigurationProfile(
            name=name,
            environment=environment,
            settings=self.configurations.copy(),
            feature_flags=self.feature_flags.copy(),
            secrets={},  # Don't include secrets in profile
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="1.0.0",
            checksum=self._calculate_checksum()
        )

        self.profiles[name] = profile
        return profile

    def load_configuration_profile(self, name: str) -> bool:
        """Load a configuration profile"""
        profile = self.profiles.get(name)
        if not profile:
            return False

        self.configurations = profile.settings.copy()
        self.feature_flags = profile.feature_flags.copy()

        return True

    def export_configuration(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export complete configuration"""
        export_data = {
            'configurations': self.configurations,
            'metadata': {k: v.__dict__ for k, v in self.metadata.items()},
            'feature_flags': {k: v.__dict__ for k, v in self.feature_flags.items()},
            'profiles': {k: v.__dict__ for k, v in self.profiles.items()},
            'exported_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }

        if include_secrets:
            export_data['secrets'] = self.secrets

        return export_data

    def import_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Import configuration"""
        try:
            self.configurations = config_data.get('configurations', {})
            self.metadata = {
                k: ConfigurationMetadata(**v)
                for k, v in config_data.get('metadata', {}).items()
            }
            self.feature_flags = {
                k: FeatureFlag(**v)
                for k, v in config_data.get('feature_flags', {}).items()
            }
            self.profiles = {
                k: ConfigurationProfile(**v)
                for k, v in config_data.get('profiles', {}).items()
            }

            if 'secrets' in config_data:
                self.secrets = config_data['secrets']

            return True
        except Exception as e:
            logging.error(f"Failed to import configuration: {e}")
            return False

    def add_change_listener(self, listener: Callable[[Dict[str, Any]], None]):
        """Add configuration change listener"""
        self._listeners.append(listener)

    def remove_change_listener(self, listener: Callable[[Dict[str, Any]], None]):
        """Remove configuration change listener"""
        self._listeners.remove(listener)

    def _calculate_checksum(self) -> str:
        """Calculate configuration checksum"""
        config_str = json.dumps(self.configurations, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'total_settings': len(self.configurations),
            'total_metadata': len(self.metadata),
            'total_feature_flags': len(self.feature_flags),
            'total_profiles': len(self.profiles),
            'total_secrets': len(self.secrets),
            'sources_count': len(self.sources),
            'listeners_count': len(self._listeners),
            'last_updated': max(
                (meta.last_modified for meta in self.metadata.values()),
                default=None
            )
        }


# Global configuration manager instance
config_manager = ConfigurationManager()

# Initialize with default configuration
def initialize_default_configuration():
    """Initialize default configuration"""

    # Add configuration sources
    config_manager.add_source(FileConfigurationSource('config/settings.json'), priority=1)
    config_manager.add_source(FileConfigurationSource('config/secrets.json'), priority=2)
    config_manager.add_source(EnvironmentConfigurationSource('HEALTHCARE_'), priority=3)

    # Load initial configuration
    config_manager.load_configuration()

    # Set up default settings if not exists
    default_settings = {
        'app_name': 'AI Personalized Medicine Platform',
        'version': '1.0.0',
        'environment': 'development',
        'debug': True,
        'log_level': 'INFO',

        # Database settings
        'database_url': 'postgresql://localhost/healthcare_db',
        'database_pool_size': 10,
        'database_timeout': 30,

        # API settings
        'api_host': '0.0.0.0',
        'api_port': 8000,
        'api_timeout': 30,
        'api_rate_limit': 100,

        # Security settings
        'jwt_secret_key': secrets.token_hex(32),
        'jwt_expiration_hours': 24,
        'bcrypt_rounds': 12,
        'mfa_issuer': 'AI Personalized Medicine',

        # ML settings
        'ml_model_cache_size': 100,
        'ml_batch_size': 32,
        'ml_learning_rate': 0.001,
        'ml_epochs': 100,

        # Genomic settings
        'genomic_analysis_timeout': 3600,
        'genomic_max_file_size': 1073741824,  # 1GB
        'genomic_supported_formats': ['vcf', 'bam', 'fastq'],

        # Health monitoring settings
        'health_monitoring_interval': 300,  # 5 minutes
        'health_alert_threshold': 0.8,
        'health_data_retention_days': 365,

        # Cache settings
        'redis_url': 'redis://localhost:6379',
        'cache_ttl': 3600,
        'cache_max_memory': '1gb',

        # External API settings
        'openai_api_key': '',
        'pubmed_api_key': '',
        'drugbank_api_key': '',

        # Notification settings
        'email_smtp_server': 'smtp.gmail.com',
        'email_smtp_port': 587,
        'sms_provider': 'twilio',

        # Compliance settings
        'hipaa_compliance_enabled': True,
        'gdpr_compliance_enabled': True,
        'audit_log_retention_days': 2555,  # 7 years

        # Performance settings
        'max_concurrent_requests': 100,
        'request_timeout': 30,
        'worker_processes': 4,
    }

    # Set default settings with metadata
    for key, value in default_settings.items():
        if key not in config_manager.configurations:
            config_manager.set(key, value, {
                'description': f'Default {key.replace("_", " ")} setting',
                'category': 'system',
                'environment': 'all',
                'is_secret': 'secret' in key.lower() or 'key' in key.lower(),
                'modified_by': 'system'
            })

    # Create default feature flags
    default_feature_flags = [
        {
            'name': 'advanced_genomic_analysis',
            'enabled': True,
            'rollout_percentage': 1.0,
            'conditions': {},
            'description': 'Enable advanced genomic analysis features',
            'owner': 'system',
            'tags': ['genomics', 'analysis']
        },
        {
            'name': 'real_time_monitoring',
            'enabled': True,
            'rollout_percentage': 1.0,
            'conditions': {},
            'description': 'Enable real-time health monitoring',
            'owner': 'system',
            'tags': ['monitoring', 'realtime']
        },
        {
            'name': 'ai_drug_discovery',
            'enabled': False,
            'rollout_percentage': 0.1,
            'conditions': {'user_role': 'researcher'},
            'description': 'Enable AI-powered drug discovery (beta)',
            'owner': 'system',
            'tags': ['ai', 'drug-discovery', 'beta']
        },
        {
            'name': 'virtual_assistant',
            'enabled': True,
            'rollout_percentage': 0.5,
            'conditions': {},
            'description': 'Enable AI virtual health assistant',
            'owner': 'system',
            'tags': ['ai', 'assistant']
        }
    ]

    for flag_config in default_feature_flags:
        config_manager.create_feature_flag(flag_config['name'], flag_config)

    # Setup configuration watching
    for source, _ in config_manager.sources:
        if hasattr(source, 'watch'):
            source.watch(lambda config: config_manager.load_configuration())

    print("Configuration system initialized successfully")


# Initialize on import
if __name__ != '__main__':
    initialize_default_configuration()
