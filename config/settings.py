"""
Configuration settings for AI Personalized Medicine Platform
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Settings:
    """Application settings and configuration"""

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Database settings (simulated for size constraints)
    DATABASE_URL: str = "sqlite:///healthcare.db"
    REDIS_URL: str = "redis://localhost:6379"

    # AI/ML settings
    MODEL_CACHE_SIZE: int = 100
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 32

    # Genomic analysis settings
    REFERENCE_GENOME: str = "GRCh38"
    VARIANT_CALLING_QUALITY: float = 30.0
    MIN_ALLELE_FREQUENCY: float = 0.01

    # Drug discovery settings
    MOLECULAR_WEIGHT_RANGE: tuple = (100, 800)
    LOGP_RANGE: tuple = (-2, 6)
    TPSA_MAX: float = 140.0

    # Health monitoring settings
    VITAL_SIGNS_INTERVAL: int = 60  # seconds
    BIOMARKER_UPDATE_INTERVAL: int = 300  # seconds
    ALERT_THRESHOLDS: Dict[str, Dict[str, float]] = None

    # Security settings
    ENCRYPTION_KEY: str = "default_key_change_in_production"
    JWT_SECRET: str = "jwt_secret_key"
    SESSION_TIMEOUT: int = 3600

    # API settings
    API_VERSION: str = "v1"
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW: int = 3600

    # Blockchain settings
    BLOCKCHAIN_NETWORK: str = "local"
    BLOCK_TIME: int = 10  # seconds

    def __post_init__(self):
        # Initialize alert thresholds
        self.ALERT_THRESHOLDS = {
            "heart_rate": {"min": 40, "max": 180, "critical_min": 30, "critical_max": 200},
            "blood_pressure_systolic": {"min": 90, "max": 140, "critical_min": 80, "critical_max": 160},
            "blood_pressure_diastolic": {"min": 60, "max": 90, "critical_min": 50, "critical_max": 100},
            "temperature": {"min": 95.0, "max": 100.4, "critical_min": 93.0, "critical_max": 105.0},
            "oxygen_saturation": {"min": 95.0, "max": 100.0, "critical_min": 90.0, "critical_max": 100.0},
            "glucose": {"min": 70, "max": 140, "critical_min": 50, "critical_max": 200}
        }

class GenomicSettings:
    """Genomic analysis specific settings"""

    # Reference genomes
    SUPPORTED_GENOMES: List[str] = ["GRCh37", "GRCh38", "T2T-CHM13v2.0"]

    # Variant types
    VARIANT_TYPES: List[str] = ["SNP", "INDEL", "CNV", "SV"]

    # Annotation sources
    ANNOTATION_SOURCES: List[str] = [
        "ClinVar", "dbSNP", "COSMIC", "OMIM",
        "HGMD", "PharmGKB", "GWAS Catalog"
    ]

    # Disease categories
    DISEASE_CATEGORIES: Dict[str, List[str]] = {
        "cardiovascular": ["coronary_artery_disease", "hypertension", "heart_failure"],
        "oncology": ["breast_cancer", "lung_cancer", "colorectal_cancer", "prostate_cancer"],
        "neurological": ["alzheimer", "parkinson", "epilepsy", "multiple_sclerosis"],
        "metabolic": ["diabetes", "obesity", "metabolic_syndrome"],
        "autoimmune": ["rheumatoid_arthritis", "lupus", "crohn_disease"]
    }

    # Pharmacogenomic genes
    PHARMACOGENOMIC_GENES: List[str] = [
        "CYP2D6", "CYP2C19", "CYP2C9", "CYP3A4", "CYP3A5",
        "SLCO1B1", "VKORC1", "TPMT", "NUDT15", "HLA-B"
    ]

class DrugDiscoverySettings:
    """Drug discovery specific settings"""

    # Molecular properties
    MOLECULAR_PROPERTIES: Dict[str, Dict[str, Any]] = {
        "molecular_weight": {"min": 100, "max": 800, "optimal": 400},
        "logp": {"min": -2, "max": 6, "optimal": 3},
        "tpsa": {"min": 0, "max": 140, "optimal": 70},
        "hbd": {"min": 0, "max": 6, "optimal": 2},
        "hba": {"min": 0, "max": 10, "optimal": 4},
        "rotatable_bonds": {"min": 0, "max": 15, "optimal": 5}
    }

    # Target classes
    TARGET_CLASSES: List[str] = [
        "GPCR", "Kinase", "Nuclear Receptor", "Ion Channel",
        "Protease", "Phosphatase", "Transporter"
    ]

    # Screening libraries
    SCREENING_LIBRARIES: Dict[str, Any] = {
        "diversity_library": {"size": 1000000, "diversity": "high"},
        "focused_library": {"size": 500000, "focus": "kinase_inhibitors"},
        "natural_products": {"size": 100000, "source": "natural"},
        "fragments": {"size": 200000, "size_range": "100-300"}
    }

class HealthMonitoringSettings:
    """Health monitoring specific settings"""

    # Vital signs
    VITAL_SIGNS: Dict[str, Dict[str, Any]] = {
        "heart_rate": {
            "unit": "bpm",
            "normal_range": [60, 100],
            "collection_interval": 60
        },
        "blood_pressure": {
            "unit": "mmHg",
            "normal_range": [[90, 120], [60, 80]],
            "collection_interval": 300
        },
        "temperature": {
            "unit": "°F",
            "normal_range": [97.0, 99.0],
            "collection_interval": 3600
        },
        "oxygen_saturation": {
            "unit": "%",
            "normal_range": [95, 100],
            "collection_interval": 60
        },
        "respiratory_rate": {
            "unit": "breaths/min",
            "normal_range": [12, 20],
            "collection_interval": 60
        }
    }

    # Biomarkers
    BIOMARKERS: Dict[str, Dict[str, Any]] = {
        "glucose": {"unit": "mg/dL", "normal_range": [70, 140], "critical_range": [50, 200]},
        "cholesterol_total": {"unit": "mg/dL", "normal_range": [0, 200], "critical_range": [0, 300]},
        "hdl_cholesterol": {"unit": "mg/dL", "normal_range": [40, 100]},
        "ldl_cholesterol": {"unit": "mg/dL", "normal_range": [0, 100], "critical_range": [0, 160]},
        "triglycerides": {"unit": "mg/dL", "normal_range": [0, 150], "critical_range": [0, 200]},
        "creatinine": {"unit": "mg/dL", "normal_range": [0.6, 1.2]},
        "bun": {"unit": "mg/dL", "normal_range": [7, 20]},
        "alt": {"unit": "U/L", "normal_range": [7, 56]},
        "ast": {"unit": "U/L", "normal_range": [10, 40]},
        "crp": {"unit": "mg/L", "normal_range": [0, 3], "critical_range": [0, 10]}
    }

class SecuritySettings:
    """Security and privacy settings"""

    # Encryption
    ENCRYPTION_ALGORITHM: str = "AES-256-GCM"
    KEY_ROTATION_DAYS: int = 90

    # Authentication
    PASSWORD_MIN_LENGTH: int = 12
    MFA_REQUIRED: bool = True
    SESSION_TIMEOUT_MINUTES: int = 60

    # Privacy
    DATA_RETENTION_DAYS: int = 2555  # 7 years for medical data
    ANONYMIZATION_LEVEL: str = "strict"

    # Compliance
    HIPAA_COMPLIANT: bool = True
    GDPR_COMPLIANT: bool = True
    SOC2_COMPLIANT: bool = True

# Global instances
settings = Settings()
genomic_settings = GenomicSettings()
drug_settings = DrugDiscoverySettings()
health_settings = HealthMonitoringSettings()
security_settings = SecuritySettings()
