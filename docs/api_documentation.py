"""
Comprehensive API Documentation for AI Personalized Medicine Platform
Complete OpenAPI specifications and interactive documentation
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class APIEndpoint:
    """API endpoint specification"""
    path: str
    method: str
    summary: str
    description: str
    tags: List[str]
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security: List[Dict[str, str]] = field(default_factory=list)
    deprecated: bool = False


@dataclass
class APISchema:
    """OpenAPI schema definition"""
    title: str
    version: str
    description: str
    endpoints: List[APIEndpoint] = field(default_factory=list)
    schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security_schemes: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class HealthcareAPIDocumentation:
    """Comprehensive healthcare API documentation generator"""

    def __init__(self):
        self.api_spec = APISchema(
            title="AI Personalized Medicine Platform API",
            version="2.0.0",
            description="Comprehensive healthcare platform combining genomics, AI, and personalized medicine"
        )
        self._build_api_specification()

    def _build_api_specification(self):
        """Build complete API specification"""
        self._add_security_schemes()
        self._add_schemas()
        self._add_authentication_endpoints()
        self._add_genomic_endpoints()
        self._add_ai_endpoints()
        self._add_drug_discovery_endpoints()
        self._add_health_monitoring_endpoints()
        self._add_treatment_endpoints()
        self._add_clinical_support_endpoints()
        self._add_patient_endpoints()
        self._add_research_endpoints()
        self._add_blockchain_endpoints()
        self._add_admin_endpoints()

    def _add_security_schemes(self):
        """Add security schemes"""
        self.api_spec.security_schemes = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            },
            "apiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            },
            "oauth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "https://auth.healthcare-platform.com/oauth/authorize",
                        "tokenUrl": "https://auth.healthcare-platform.com/oauth/token",
                        "scopes": {
                            "read:health_data": "Read health data",
                            "write:health_data": "Write health data",
                            "admin:system": "System administration"
                        }
                    }
                }
            }
        }

    def _add_schemas(self):
        """Add comprehensive data schemas"""
        self.api_spec.schemas = {
            "PatientProfile": {
                "type": "object",
                "required": ["patient_id", "demographics"],
                "properties": {
                    "patient_id": {"type": "string", "description": "Unique patient identifier"},
                    "demographics": {
                        "type": "object",
                        "properties": {
                            "first_name": {"type": "string"},
                            "last_name": {"type": "string"},
                            "date_of_birth": {"type": "string", "format": "date"},
                            "gender": {"type": "string", "enum": ["M", "F", "O"]},
                            "ethnicity": {"type": "string"},
                            "address": {"type": "object"}
                        }
                    },
                    "medical_history": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/MedicalRecord"}
                    },
                    "genomic_data": {"$ref": "#/components/schemas/GenomicData"},
                    "lifestyle_data": {"$ref": "#/components/schemas/LifestyleData"}
                }
            },
            "MedicalRecord": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "date": {"type": "string", "format": "date-time"},
                    "type": {"type": "string", "enum": ["diagnosis", "treatment", "test", "procedure"]},
                    "description": {"type": "string"},
                    "provider": {"type": "string"},
                    "facility": {"type": "string"}
                }
            },
            "GenomicData": {
                "type": "object",
                "properties": {
                    "genome_sequence": {"type": "string", "description": "Complete genome sequence"},
                    "variants": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/GeneticVariant"}
                    },
                    "ancestry": {"type": "object"},
                    "pharmacogenomics": {"type": "object"}
                }
            },
            "GeneticVariant": {
                "type": "object",
                "properties": {
                    "chromosome": {"type": "string"},
                    "position": {"type": "integer"},
                    "reference": {"type": "string"},
                    "alternate": {"type": "string"},
                    "quality": {"type": "number"},
                    "depth": {"type": "integer"},
                    "genotype": {"type": "string"}
                }
            },
            "LifestyleData": {
                "type": "object",
                "properties": {
                    "diet": {"type": "object"},
                    "exercise": {"type": "object"},
                    "sleep": {"type": "object"},
                    "stress": {"type": "object"},
                    "substances": {"type": "object"}
                }
            },
            "HealthMetrics": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "vital_signs": {
                        "type": "object",
                        "properties": {
                            "heart_rate": {"type": "number", "minimum": 30, "maximum": 200},
                            "blood_pressure_systolic": {"type": "number", "minimum": 70, "maximum": 250},
                            "blood_pressure_diastolic": {"type": "number", "minimum": 40, "maximum": 150},
                            "temperature": {"type": "number", "minimum": 30, "maximum": 45},
                            "respiratory_rate": {"type": "number", "minimum": 8, "maximum": 60},
                            "oxygen_saturation": {"type": "number", "minimum": 70, "maximum": 100}
                        }
                    },
                    "biomarkers": {
                        "type": "object",
                        "properties": {
                            "glucose": {"type": "number", "minimum": 20, "maximum": 600},
                            "cholesterol_total": {"type": "number", "minimum": 50, "maximum": 400},
                            "hdl": {"type": "number", "minimum": 20, "maximum": 100},
                            "ldl": {"type": "number", "minimum": 0, "maximum": 300},
                            "triglycerides": {"type": "number", "minimum": 0, "maximum": 1000},
                            "creatinine": {"type": "number", "minimum": 0.1, "maximum": 20},
                            "bun": {"type": "number", "minimum": 2, "maximum": 100},
                            "hemoglobin": {"type": "number", "minimum": 5, "maximum": 20},
                            "wbc": {"type": "number", "minimum": 1000, "maximum": 50000},
                            "platelets": {"type": "number", "minimum": 10000, "maximum": 500000}
                        }
                    }
                }
            },
            "TreatmentPlan": {
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string"},
                    "patient_id": {"type": "string"},
                    "diagnosis": {"type": "string"},
                    "medications": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Medication"}
                    },
                    "lifestyle_recommendations": {"type": "array", "items": {"type": "string"}},
                    "monitoring_schedule": {"type": "object"},
                    "follow_up": {"type": "object"}
                }
            },
            "Medication": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dosage": {"type": "string"},
                    "frequency": {"type": "string"},
                    "duration": {"type": "string"},
                    "indications": {"type": "array", "items": {"type": "string"}},
                    "contraindications": {"type": "array", "items": {"type": "string"}},
                    "side_effects": {"type": "array", "items": {"type": "string"}}
                }
            },
            "ClinicalTrial": {
                "type": "object",
                "properties": {
                    "trial_id": {"type": "string"},
                    "title": {"type": "string"},
                    "phase": {"type": "string", "enum": ["I", "II", "III", "IV"]},
                    "condition": {"type": "string"},
                    "intervention": {"type": "string"},
                    "eligibility_criteria": {"type": "object"},
                    "recruitment_status": {"type": "string"},
                    "estimated_completion": {"type": "string", "format": "date"}
                }
            },
            "DrugDiscoveryResult": {
                "type": "object",
                "properties": {
                    "target_protein": {"type": "string"},
                    "compounds": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Compound"}
                    },
                    "binding_affinity": {"type": "number"},
                    "toxicity_score": {"type": "number"},
                    "efficacy_prediction": {"type": "number"}
                }
            },
            "Compound": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                    "molecular_weight": {"type": "number"},
                    "logp": {"type": "number"},
                    "binding_energy": {"type": "number"},
                    "toxicity_probability": {"type": "number"}
                }
            }
        }

    def _add_authentication_endpoints(self):
        """Add authentication endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/auth/login",
                method="POST",
                summary="User authentication",
                description="Authenticate user and return JWT token",
                tags=["Authentication"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["username", "password"],
                                "properties": {
                                    "username": {"type": "string"},
                                    "password": {"type": "string"},
                                    "mfa_code": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Authentication successful",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "access_token": {"type": "string"},
                                        "refresh_token": {"type": "string"},
                                        "token_type": {"type": "string", "enum": ["Bearer"]},
                                        "expires_in": {"type": "integer"},
                                        "user": {"$ref": "#/components/schemas/UserProfile"}
                                    }
                                }
                            }
                        }
                    },
                    "401": {"description": "Authentication failed"},
                    "429": {"description": "Too many failed attempts"}
                },
                security=[]
            ),
            APIEndpoint(
                path="/auth/refresh",
                method="POST",
                summary="Refresh access token",
                description="Refresh expired access token using refresh token",
                tags=["Authentication"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["refresh_token"],
                                "properties": {
                                    "refresh_token": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Token refreshed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "access_token": {"type": "string"},
                                        "token_type": {"type": "string", "enum": ["Bearer"]},
                                        "expires_in": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/auth/logout",
                method="POST",
                summary="User logout",
                description="Invalidate current session",
                tags=["Authentication"],
                responses={
                    "200": {"description": "Logged out successfully"}
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/auth/mfa/setup",
                method="POST",
                summary="Setup MFA",
                description="Setup multi-factor authentication for user",
                tags=["Authentication"],
                responses={
                    "200": {
                        "description": "MFA setup initiated",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "qr_code_url": {"type": "string"},
                                        "secret": {"type": "string"},
                                        "backup_codes": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/auth/mfa/verify",
                method="POST",
                summary="Verify MFA code",
                description="Verify multi-factor authentication code",
                tags=["Authentication"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["code"],
                                "properties": {
                                    "code": {"type": "string"},
                                    "method": {"type": "string", "enum": ["totp", "backup"]}
                                }
                            }
                        }
                    }
                },
                responses={
                    "200": {"description": "MFA verified successfully"},
                    "401": {"description": "Invalid MFA code"}
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_genomic_endpoints(self):
        """Add genomic analysis endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/genomics/analyze",
                method="POST",
                summary="Submit genome for analysis",
                description="Submit genomic data for comprehensive analysis",
                tags=["Genomics"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["patient_id"],
                                "properties": {
                                    "patient_id": {"type": "string"},
                                    "genome_sequence": {"type": "string"},
                                    "analysis_type": {
                                        "type": "string",
                                        "enum": ["comprehensive", "variants_only", "pharmacogenomics", "disease_risk"],
                                        "default": "comprehensive"
                                    },
                                    "reference_genome": {"type": "string", "default": "GRCh38"}
                                }
                            }
                        }
                    }
                },
                responses={
                    "202": {
                        "description": "Analysis queued successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "job_id": {"type": "string"},
                                        "status": {"type": "string", "enum": ["queued"]},
                                        "estimated_completion": {"type": "string"},
                                        "analysis_types": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "Invalid genomic data"}
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/genomics/results/{job_id}",
                method="GET",
                summary="Get genomic analysis results",
                description="Retrieve results of genomic analysis job",
                tags=["Genomics"],
                parameters=[
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Genomic analysis job ID"
                    }
                ],
                responses={
                    "200": {
                        "description": "Analysis results",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "job_id": {"type": "string"},
                                        "status": {"type": "string", "enum": ["completed", "processing", "failed"]},
                                        "results": {"$ref": "#/components/schemas/GenomicData"},
                                        "confidence_scores": {"type": "object"},
                                        "recommendations": {"type": "array", "items": {"type": "string"}},
                                        "warnings": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    },
                    "404": {"description": "Job not found"}
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/genomics/variants/{patient_id}",
                method="GET",
                summary="Get genetic variants",
                description="Retrieve genetic variants for patient",
                tags=["Genomics"],
                parameters=[
                    {
                        "name": "patient_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "chromosome",
                        "in": "query",
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "impact",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["high", "moderate", "low"]}
                    }
                ],
                responses={
                    "200": {
                        "description": "Genetic variants",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/GeneticVariant"}
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/genomics/pharmacogenomics/{patient_id}",
                method="GET",
                summary="Get pharmacogenomics profile",
                description="Retrieve pharmacogenomics analysis for patient",
                tags=["Genomics"],
                parameters=[
                    {
                        "name": "patient_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": "Pharmacogenomics profile",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "cyp2d6": {"type": "string", "enum": ["poor", "intermediate", "normal", "ultrarapid"]},
                                        "cyp2c19": {"type": "string", "enum": ["poor", "intermediate", "normal", "ultrarapid"]},
                                        "cyp2c9": {"type": "string", "enum": ["poor", "intermediate", "normal"]},
                                        "slco1b1": {"type": "string", "enum": ["reduced", "normal"]},
                                        "recommendations": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_ai_endpoints(self):
        """Add AI/ML prediction endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/ai/predict/disease-risk",
                method="POST",
                summary="Predict disease risk",
                description="AI-powered disease risk prediction based on patient data",
                tags=["AI", "Predictions"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["patient_id"],
                                "properties": {
                                    "patient_id": {"type": "string"},
                                    "demographics": {"type": "object"},
                                    "genomic_data": {"$ref": "#/components/schemas/GenomicData"},
                                    "biomarkers": {"type": "object"},
                                    "lifestyle_factors": {"type": "object"},
                                    "family_history": {"type": "array", "items": {"type": "string"}},
                                    "diseases": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Disease risk predictions",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "patient_id": {"type": "string"},
                                        "predictions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "disease": {"type": "string"},
                                                    "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
                                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                                    "timeframe": {"type": "string"},
                                                    "preventive_measures": {"type": "array", "items": {"type": "string"}}
                                                }
                                            }
                                        },
                                        "overall_risk_assessment": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/ai/predict/drug-response",
                method="POST",
                summary="Predict drug response",
                description="Predict patient response to specific medications",
                tags=["AI", "Predictions"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["patient_id", "medications"],
                                "properties": {
                                    "patient_id": {"type": "string"},
                                    "medications": {"type": "array", "items": {"type": "string"}},
                                    "genomic_data": {"$ref": "#/components/schemas/GenomicData"},
                                    "current_health": {"type": "object"}
                                }
                            }
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Drug response predictions",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "patient_id": {"type": "string"},
                                        "predictions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "medication": {"type": "string"},
                                                    "efficacy_score": {"type": "number", "minimum": 0, "maximum": 1},
                                                    "toxicity_risk": {"type": "number", "minimum": 0, "maximum": 1},
                                                    "recommended_dosage": {"type": "string"},
                                                    "monitoring_required": {"type": "boolean"},
                                                    "alternative_suggestions": {"type": "array", "items": {"type": "string"}}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/ai/predict/treatment-outcome",
                method="POST",
                summary="Predict treatment outcome",
                description="Predict outcomes of proposed treatment plans",
                tags=["AI", "Predictions"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["patient_id", "treatment_plan"],
                                "properties": {
                                    "patient_id": {"type": "string"},
                                    "treatment_plan": {"$ref": "#/components/schemas/TreatmentPlan"},
                                    "historical_data": {"type": "array", "items": {"type": "object"}}
                                }
                            }
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Treatment outcome predictions",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "patient_id": {"type": "string"},
                                        "outcome_predictions": {
                                            "type": "object",
                                            "properties": {
                                                "success_probability": {"type": "number", "minimum": 0, "maximum": 1},
                                                "expected_improvement": {"type": "string"},
                                                "time_to_response": {"type": "string"},
                                                "risk_factors": {"type": "array", "items": {"type": "string"}},
                                                "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_drug_discovery_endpoints(self):
        """Add drug discovery endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/drug-discovery/discover",
                method="POST",
                summary="Initiate drug discovery",
                description="AI-powered drug discovery for specific targets",
                tags=["Drug Discovery"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["target_protein", "disease_context"],
                                "properties": {
                                    "target_protein": {"type": "string"},
                                    "disease_context": {"type": "string"},
                                    "patient_profile": {"type": "object"},
                                    "search_parameters": {
                                        "type": "object",
                                        "properties": {
                                            "max_compounds": {"type": "integer", "default": 100},
                                            "toxicity_threshold": {"type": "number", "default": 0.8},
                                            "efficacy_threshold": {"type": "number", "default": 0.7}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                responses={
                    "202": {
                        "description": "Drug discovery initiated",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "job_id": {"type": "string"},
                                        "status": {"type": "string", "enum": ["queued"]},
                                        "estimated_completion": {"type": "string"},
                                        "target_protein": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/drug-discovery/results/{job_id}",
                method="GET",
                summary="Get drug discovery results",
                description="Retrieve results of drug discovery job",
                tags=["Drug Discovery"],
                parameters=[
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": "Drug discovery results",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "job_id": {"type": "string"},
                                        "status": {"type": "string", "enum": ["completed", "processing", "failed"]},
                                        "results": {"$ref": "#/components/schemas/DrugDiscoveryResult"},
                                        "compounds_analyzed": {"type": "integer"},
                                        "lead_compounds": {"type": "array", "items": {"$ref": "#/components/schemas/Compound"}}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_health_monitoring_endpoints(self):
        """Add health monitoring endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/monitoring/health-data",
                method="POST",
                summary="Submit health monitoring data",
                description="Submit real-time health monitoring data from devices",
                tags=["Health Monitoring"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthMetrics"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Health data processed",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "enum": ["processed"]},
                                        "alerts": {"type": "array", "items": {"type": "object"}},
                                        "recommendations": {"type": "array", "items": {"type": "string"}},
                                        "next_check_interval": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/monitoring/devices",
                method="GET",
                summary="Get connected devices",
                description="Retrieve list of connected health monitoring devices",
                tags=["Health Monitoring"],
                parameters=[
                    {
                        "name": "patient_id",
                        "in": "query",
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": "Connected devices",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "device_id": {"type": "string"},
                                            "type": {"type": "string"},
                                            "status": {"type": "string", "enum": ["connected", "disconnected"]},
                                            "last_sync": {"type": "string", "format": "date-time"},
                                            "battery_level": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/monitoring/alerts/{patient_id}",
                method="GET",
                summary="Get health alerts",
                description="Retrieve health alerts for patient",
                tags=["Health Monitoring"],
                parameters=[
                    {
                        "name": "patient_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "severity",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                    }
                ],
                responses={
                    "200": {
                        "description": "Health alerts",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "alert_id": {"type": "string"},
                                            "type": {"type": "string"},
                                            "severity": {"type": "string"},
                                            "message": {"type": "string"},
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "resolved": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_treatment_endpoints(self):
        """Add treatment planning endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/treatment/plan",
                method="POST",
                summary="Create treatment plan",
                description="Generate personalized treatment plan",
                tags=["Treatment"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["patient_id", "diagnosis"],
                                "properties": {
                                    "patient_id": {"type": "string"},
                                    "diagnosis": {"type": "string"},
                                    "current_medications": {"type": "array", "items": {"type": "string"}},
                                    "contraindications": {"type": "array", "items": {"type": "string"}},
                                    "preferences": {"type": "object"}
                                }
                            }
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Treatment plan created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TreatmentPlan"}
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/treatment/plans/{patient_id}",
                method="GET",
                summary="Get treatment plans",
                description="Retrieve treatment plans for patient",
                tags=["Treatment"],
                parameters=[
                    {
                        "name": "patient_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": "Treatment plans",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/TreatmentPlan"}
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_clinical_support_endpoints(self):
        """Add clinical decision support endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/clinical-support/query",
                method="POST",
                summary="Clinical decision support",
                description="Get AI-powered clinical recommendations",
                tags=["Clinical Support"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["patient_id", "query"],
                                "properties": {
                                    "patient_id": {"type": "string"},
                                    "query": {"type": "string"},
                                    "context": {"type": "object"}
                                }
                            }
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Clinical recommendations",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "recommendations": {"type": "array", "items": {"type": "object"}},
                                        "evidence_level": {"type": "string"},
                                        "confidence_score": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_patient_endpoints(self):
        """Add patient engagement endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/patient/dashboard/{patient_id}",
                method="GET",
                summary="Get patient dashboard",
                description="Retrieve comprehensive patient health dashboard",
                tags=["Patient"],
                parameters=[
                    {
                        "name": "patient_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": "Patient dashboard",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "patient_info": {"$ref": "#/components/schemas/PatientProfile"},
                                        "health_metrics": {"type": "object"},
                                        "recent_activity": {"type": "array", "items": {"type": "object"}},
                                        "recommendations": {"type": "array", "items": {"type": "string"}},
                                        "upcoming_appointments": {"type": "array", "items": {"type": "object"}}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/patient/virtual-assistant",
                method="POST",
                summary="Virtual health assistant",
                description="Interact with AI-powered virtual health assistant",
                tags=["Patient"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["patient_id", "message"],
                                "properties": {
                                    "patient_id": {"type": "string"},
                                    "message": {"type": "string"},
                                    "context": {"type": "object"}
                                }
                            }
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Assistant response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "response": {"type": "string"},
                                        "actions": {"type": "array", "items": {"type": "object"}},
                                        "follow_up_questions": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_research_endpoints(self):
        """Add research and clinical trials endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/research/trials",
                method="POST",
                summary="Create clinical trial",
                description="Create and configure clinical trial",
                tags=["Research"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ClinicalTrial"}
                        }
                    }
                },
                responses={
                    "201": {
                        "description": "Clinical trial created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ClinicalTrial"}
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/research/trials/{patient_id}/matches",
                method="GET",
                summary="Find matching clinical trials",
                description="Find suitable clinical trials for patient",
                tags=["Research"],
                parameters=[
                    {
                        "name": "patient_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": "Matching clinical trials",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/ClinicalTrial"}
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_blockchain_endpoints(self):
        """Add blockchain security endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/blockchain/record",
                method="POST",
                summary="Create secured health record",
                description="Create immutable health record on blockchain",
                tags=["Blockchain"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["patient_id", "record_type", "data"],
                                "properties": {
                                    "patient_id": {"type": "string"},
                                    "record_type": {"type": "string"},
                                    "data": {"type": "object"},
                                    "consent_given": {"type": "boolean"}
                                }
                            }
                        }
                    }
                },
                responses={
                    "201": {
                        "description": "Health record secured",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "record_id": {"type": "string"},
                                        "block_hash": {"type": "string"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/blockchain/verify/{record_id}",
                method="GET",
                summary="Verify record integrity",
                description="Verify health record integrity using blockchain",
                tags=["Blockchain"],
                parameters=[
                    {
                        "name": "record_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": "Record verification",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "record_id": {"type": "string"},
                                        "verified": {"type": "boolean"},
                                        "block_hash": {"type": "string"},
                                        "tamper_evidence": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def _add_admin_endpoints(self):
        """Add administrative endpoints"""
        self.api_spec.endpoints.extend([
            APIEndpoint(
                path="/admin/metrics",
                method="GET",
                summary="System metrics",
                description="Get comprehensive system performance metrics",
                tags=["Administration"],
                responses={
                    "200": {
                        "description": "System metrics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "system_health": {"type": "string"},
                                        "active_users": {"type": "integer"},
                                        "api_requests": {"type": "integer"},
                                        "genomic_analyses": {"type": "integer"},
                                        "drug_discoveries": {"type": "integer"},
                                        "error_rate": {"type": "number"},
                                        "response_time_avg": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/admin/audit-logs",
                method="GET",
                summary="Audit logs",
                description="Retrieve system audit logs",
                tags=["Administration"],
                parameters=[
                    {
                        "name": "start_date",
                        "in": "query",
                        "schema": {"type": "string", "format": "date"}
                    },
                    {
                        "name": "end_date",
                        "in": "query",
                        "schema": {"type": "string", "format": "date"}
                    },
                    {
                        "name": "event_type",
                        "in": "query",
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": "Audit logs",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "event_type": {"type": "string"},
                                            "user_id": {"type": "string"},
                                            "resource": {"type": "string"},
                                            "action": {"type": "string"},
                                            "status": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ])

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate complete OpenAPI 3.0 specification"""
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": self.api_spec.title,
                "version": self.api_spec.version,
                "description": self.api_spec.description,
                "contact": {
                    "name": "AI Personalized Medicine Platform Support",
                    "email": "support@healthcare-platform.com"
                },
                "license": {
                    "name": "Proprietary",
                    "url": "https://healthcare-platform.com/license"
                }
            },
            "servers": [
                {
                    "url": "https://api.healthcare-platform.com/v2",
                    "description": "Production server"
                },
                {
                    "url": "https://staging-api.healthcare-platform.com/v2",
                    "description": "Staging server"
                },
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                }
            ],
            "security": [
                {"bearerAuth": []}
            ],
            "paths": {},
            "components": {
                "schemas": self.api_spec.schemas,
                "securitySchemes": self.api_spec.security_schemes
            }
        }

        # Add paths
        for endpoint in self.api_spec.endpoints:
            path_spec = spec["paths"].get(endpoint.path, {})

            operation_spec = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "responses": endpoint.responses
            }

            if endpoint.parameters:
                operation_spec["parameters"] = endpoint.parameters

            if endpoint.request_body:
                operation_spec["requestBody"] = endpoint.request_body

            if endpoint.security:
                operation_spec["security"] = endpoint.security

            if endpoint.deprecated:
                operation_spec["deprecated"] = True

            path_spec[endpoint.method.lower()] = operation_spec
            spec["paths"][endpoint.path] = path_spec

        return spec

    def generate_html_documentation(self) -> str:
        """Generate HTML documentation from API spec"""
        spec = self.generate_openapi_spec()

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{spec['info']['title']} - API Documentation</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .endpoint {{
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .method {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 4px;
                    color: white;
                    font-weight: bold;
                    text-transform: uppercase;
                    font-size: 12px;
                }}
                .method.get {{ background-color: #61affe; }}
                .method.post {{ background-color: #49cc90; }}
                .method.put {{ background-color: #fca130; }}
                .method.delete {{ background-color: #f93e3e; }}
                .path {{ font-family: 'Courier New', monospace; font-size: 16px; margin-left: 10px; }}
                .schema {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px; }}
                .code {{ font-family: 'Courier New', monospace; background-color: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .tag {{ display: inline-block; background-color: #e2e8f0; color: #2d3748; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{spec['info']['title']}</h1>
                <p>{spec['info']['description']}</p>
                <p><strong>Version:</strong> {spec['info']['version']}</p>
            </div>

            <h2>API Endpoints</h2>
        """

        # Group endpoints by tags
        endpoints_by_tag = {}
        for endpoint in self.api_spec.endpoints:
            for tag in endpoint.tags:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)

        for tag, endpoints in endpoints_by_tag.items():
            html += f"<h3>{tag}</h3>"
            for endpoint in endpoints:
                method_class = endpoint.method.lower()
                html += f"""
                <div class="endpoint">
                    <h4>
                        <span class="method {method_class}">{endpoint.method}</span>
                        <span class="path">{endpoint.path}</span>
                    </h4>
                    <p>{endpoint.description}</p>
                    <div>
                        {" ".join(f'<span class="tag">{t}</span>' for t in endpoint.tags)}
                    </div>
                """

                if endpoint.parameters:
                    html += "<h5>Parameters:</h5><ul>"
                    for param in endpoint.parameters:
                        required = " (required)" if param.get('required', False) else ""
                        html += f"<li><strong>{param['name']}</strong>{required}: {param.get('description', '')}</li>"
                    html += "</ul>"

                if endpoint.request_body:
                    html += "<h5>Request Body:</h5><div class='schema'>"
                    html += f"<pre class='code'>{json.dumps(endpoint.request_body, indent=2)}</pre>"
                    html += "</div>"

                html += "<h5>Responses:</h5><div class='schema'>"
                html += f"<pre class='code'>{json.dumps(endpoint.responses, indent=2)}</pre>"
                html += "</div></div>"

        html += """
        </body>
        </html>
        """

        return html

    def export_openapi_json(self, filename: str = "openapi_spec.json"):
        """Export OpenAPI specification as JSON file"""
        spec = self.generate_openapi_spec()
        with open(filename, 'w') as f:
            json.dump(spec, f, indent=2)

    def export_html_documentation(self, filename: str = "api_documentation.html"):
        """Export HTML documentation"""
        html = self.generate_html_documentation()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
