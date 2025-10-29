# Comprehensive API Documentation

## AI Personalized Medicine Platform API

### Overview

The AI Personalized Medicine Platform provides a comprehensive REST API for healthcare applications, enabling personalized medicine through advanced analytics, genomic analysis, real-time health monitoring, and clinical decision support. The API is built with FastAPI and follows RESTful principles with comprehensive OpenAPI 3.0 documentation.

### API Base URL
```
https://api.healthcare-platform.com/v2
```

### Authentication

All API requests require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <jwt_token>
```

#### Authentication Endpoints

##### POST /auth/login
Authenticate a user and receive access/refresh tokens.

**Request Body:**
```json
{
  "username": "string",
  "password": "string",
  "remember_me": "boolean"
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "id": 1,
    "username": "string",
    "email": "string",
    "first_name": "string",
    "last_name": "string",
    "role": "patient|physician|nurse|pharmacist|researcher|admin",
    "profile_complete": true,
    "email_verified": true,
    "phone_verified": false,
    "mfa_enabled": false,
    "last_login": "2024-01-15T10:30:00Z",
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

##### POST /auth/refresh
Refresh an access token using a refresh token.

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

##### POST /auth/register
Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "first_name": "string",
  "last_name": "string",
  "password": "string",
  "role": "patient|physician|nurse|pharmacist|researcher|admin",
  "date_of_birth": "2000-01-01",
  "gender": "M|F|O|U",
  "phone": "+1234567890"
}
```

##### POST /auth/logout
Log out the current user and invalidate tokens.

##### POST /auth/forgot-password
Initiate password reset process.

**Request Body:**
```json
{
  "email": "string"
}
```

##### POST /auth/reset-password
Complete password reset with token.

**Request Body:**
```json
{
  "token": "string",
  "new_password": "string"
}
```

##### POST /auth/setup-mfa
Setup multi-factor authentication.

**Response (200):**
```json
{
  "secret": "string",
  "qr_code_url": "string",
  "backup_codes": ["string"]
}
```

##### POST /auth/verify-mfa
Verify MFA code.

**Request Body:**
```json
{
  "code": "string",
  "method": "totp|sms|email"
}
```

### Patient Management

#### Patient CRUD Operations

##### POST /patients
Create a new patient record.

**Request Body:**
```json
{
  "patient_id": "string",
  "first_name": "string",
  "last_name": "string",
  "date_of_birth": "2000-01-01",
  "gender": "M|F|O|U",
  "race": "string",
  "ethnicity": "string",
  "language": "string",
  "phone_primary": "+1234567890",
  "phone_secondary": "+1234567891",
  "email": "patient@example.com",
  "address_street": "123 Main St",
  "address_city": "Anytown",
  "address_state": "CA",
  "address_zip": "12345",
  "address_country": "US",
  "insurance_provider": "Blue Cross",
  "insurance_policy_number": "POL123456",
  "emergency_contact_name": "Jane Doe",
  "emergency_contact_relationship": "Spouse",
  "emergency_contact_phone": "+1234567892"
}
```

##### GET /patients/{patient_id}
Retrieve patient information.

**Response (200):**
```json
{
  "id": 1,
  "patient_id": "PAT001",
  "user_id": 1,
  "first_name": "John",
  "last_name": "Doe",
  "date_of_birth": "1980-01-01",
  "gender": "M",
  "race": "White",
  "ethnicity": "Not Hispanic",
  "language": "English",
  "phone_primary": "+1234567890",
  "phone_secondary": null,
  "email": "john.doe@example.com",
  "address_street": "123 Main St",
  "address_city": "Anytown",
  "address_state": "CA",
  "address_zip": "12345",
  "address_country": "US",
  "insurance_provider": "Blue Cross",
  "insurance_policy_number": "POL123456",
  "emergency_contact_name": "Jane Doe",
  "emergency_contact_relationship": "Spouse",
  "emergency_contact_phone": "+1234567892",
  "blood_type": "O+",
  "height_cm": 175.0,
  "weight_kg": 80.0,
  "bmi": 26.1,
  "smoking_status": "never",
  "alcohol_use": "moderate",
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

##### PUT /patients/{patient_id}
Update patient information.

##### GET /patients
List patients with filtering and pagination.

**Query Parameters:**
- `search`: Search term for name or patient ID
- `status`: active|inactive
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)

#### Vital Signs Management

##### POST /patients/{patient_id}/vitals
Add vital signs measurement.

**Request Body:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "heart_rate": 72,
  "blood_pressure_systolic": 128,
  "blood_pressure_diastolic": 82,
  "temperature": 98.6,
  "respiratory_rate": 16,
  "oxygen_saturation": 98.0,
  "weight": 80.0,
  "height": 175.0,
  "pain_scale": null,
  "blood_glucose": 95.0,
  "notes": "Patient reports feeling well",
  "device_type": "Manual",
  "device_id": null,
  "measurement_method": "Manual"
}
```

##### GET /patients/{patient_id}/vitals
Get vital signs history.

**Query Parameters:**
- `start_date`: Start date for measurements
- `end_date`: End date for measurements
- `limit`: Maximum number of records (default: 50)

**Response (200):**
```json
[
  {
    "id": 1,
    "patient_id": "PAT001",
    "timestamp": "2024-01-15T10:30:00Z",
    "heart_rate": 72,
    "blood_pressure_systolic": 128,
    "blood_pressure_diastolic": 82,
    "temperature": 98.6,
    "respiratory_rate": 16,
    "oxygen_saturation": 98.0,
    "weight": 80.0,
    "height": 175.0,
    "bmi": 26.1,
    "pain_scale": null,
    "blood_glucose": 95.0,
    "notes": "Patient reports feeling well",
    "device_type": "Manual",
    "device_id": null,
    "measurement_method": "Manual",
    "measurement_quality": "good",
    "recorded_by": "Dr. Smith",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
]
```

### Genomic Analysis

#### Analysis Management

##### POST /genomics/analyze
Submit genomic data for analysis.

**Request Body:**
```json
{
  "patient_id": "PAT001",
  "analysis_type": "comprehensive",
  "reference_genome": "GRCh38",
  "sequencing_platform": "Illumina NovaSeq",
  "coverage_depth": 30.0,
  "clinical_indication": "Family history of breast cancer"
}
```

**Response (202):**
```json
{
  "analysis_id": "analysis_1705312200_abc123",
  "status": "queued",
  "estimated_completion": "2-4 hours",
  "message": "Genomic analysis submitted successfully"
}
```

##### GET /genomics/analysis/{analysis_id}
Get analysis results.

**Response (200):**
```json
{
  "analysis_id": "analysis_1705312200_abc123",
  "patient_id": "PAT001",
  "analysis_type": "comprehensive",
  "reference_genome": "GRCh38",
  "sequencing_platform": "Illumina NovaSeq",
  "coverage_depth": 30.0,
  "status": "completed",
  "progress_percentage": 100.0,
  "started_at": "2024-01-15T08:30:00Z",
  "completed_at": "2024-01-15T10:30:00Z",
  "estimated_completion_time": "2024-01-15T10:30:00Z",
  "variants_called": 150000,
  "variants_filtered": 50000,
  "clinical_variants": 25,
  "clinical_report": "Comprehensive genomic analysis completed. 25 clinically relevant variants identified.",
  "recommendations": [
    "Schedule genetic counseling consultation",
    "Consider pharmacogenomic testing for medication optimization",
    "Annual cardiovascular screening recommended"
  ],
  "warnings": [
    "Several variants of uncertain significance identified",
    "Limited clinical evidence for some pharmacogenomic associations"
  ],
  "created_at": "2024-01-15T08:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "requested_by": "Dr. Smith",
  "reviewed_by": "Dr. Johnson"
}
```

##### GET /genomics/analysis/{analysis_id}/variants
Get genetic variants from analysis.

**Query Parameters:**
- `chromosome`: Filter by chromosome
- `impact`: Filter by variant impact (HIGH|MODERATE|LOW|MODIFIER)
- `gene`: Filter by gene name
- `limit`: Maximum number of variants (default: 100)

**Response (200):**
```json
[
  {
    "id": 1,
    "analysis_id": "analysis_1705312200_abc123",
    "chromosome": "17",
    "position": 43044295,
    "reference_allele": "G",
    "alternate_allele": "A",
    "variant_id": "rs80357508",
    "quality_score": 45.0,
    "depth": 30,
    "gene_name": "BRCA1",
    "consequence": "splice_donor_variant",
    "impact": "HIGH",
    "clinvar_significance": "Pathogenic",
    "clinvar_id": "VCV000038238",
    "sift_score": 0.0,
    "polyphen_score": 0.0,
    "cadd_score": 35.0,
    "allele_frequency_global": 0.0001,
    "disease_association": "Breast cancer susceptibility",
    "functional_studies": "Multiple studies confirm pathogenicity"
  }
]
```

### AI/ML Model Endpoints

#### Disease Risk Prediction

##### POST /ai/predict/disease-risk
Predict disease risk using AI models.

**Request Body:**
```json
{
  "patient_id": "PAT001",
  "demographics": {
    "age": 45,
    "gender": "M",
    "race": "White",
    "ethnicity": "Not Hispanic"
  },
  "clinical_data": {
    "bmi": 28.5,
    "blood_pressure": {"systolic": 140, "diastolic": 90},
    "glucose": 110,
    "cholesterol": {"total": 240, "hdl": 45, "ldl": 160}
  },
  "lifestyle_factors": {
    "smoking_status": "former",
    "alcohol_use": "moderate",
    "exercise_frequency": "occasional",
    "diet_quality": "fair"
  },
  "family_history": [
    "coronary_artery_disease",
    "type_2_diabetes",
    "breast_cancer"
  ],
  "genomic_data": {
    "risk_alleles": ["APOE4", "BRCA1_variant"],
    "polygenic_risk_score": 1.8
  }
}
```

**Response (200):**
```json
{
  "patient_id": "PAT001",
  "predictions": [
    {
      "disease": "Cardiovascular Disease",
      "risk_score": 0.15,
      "confidence": 0.85,
      "timeframe": "10 years",
      "risk_factors": [
        "Hypertension",
        "Elevated LDL cholesterol",
        "Family history",
        "APOE4 genotype"
      ],
      "preventive_measures": [
        "Blood pressure management",
        "Statin therapy",
        "Lifestyle modifications",
        "Regular cardiovascular screening"
      ]
    },
    {
      "disease": "Type 2 Diabetes",
      "risk_score": 0.22,
      "confidence": 0.78,
      "timeframe": "5 years",
      "risk_factors": [
        "Elevated BMI",
        "Prediabetic glucose levels",
        "Family history"
      ],
      "preventive_measures": [
        "Weight management",
        "Regular exercise",
        "Dietary modifications",
        "Metformin prophylaxis consideration"
      ]
    }
  ],
  "overall_risk_assessment": "moderate",
  "confidence_score": 0.8,
  "recommendations": [
    "Schedule annual comprehensive metabolic panel",
    "Consider genetic counseling for hereditary risk factors",
    "Implement lifestyle intervention program",
    "Regular monitoring of cardiovascular risk factors"
  ]
}
```

#### Drug Response Prediction

##### POST /ai/predict/drug-response
Predict drug response and efficacy.

**Request Body:**
```json
{
  "patient_id": "PAT001",
  "medications": ["simvastatin", "metformin", "lisinopril"],
  "genomic_data": {
    "cyp2d6_genotype": "*1/*4",
    "slco1b1_genotype": "*1/*5",
    "cyp2c19_genotype": "*1/*2"
  },
  "clinical_factors": {
    "age": 45,
    "weight": 80,
    "liver_function": "normal",
    "kidney_function": "normal",
    "comorbidities": ["hypertension", "dyslipidemia"]
  }
}
```

**Response (200):**
```json
[
  {
    "patient_id": "PAT001",
    "medication": "simvastatin",
    "efficacy_score": 0.75,
    "toxicity_risk": 0.25,
    "recommended_dosage": "20mg daily",
    "monitoring_required": true,
    "alternative_suggestions": ["pravastatin", "atorvastatin"],
    "pharmacogenomic_considerations": [
      "SLCO1B1*5 variant may increase myopathy risk",
      "Consider dose reduction or alternative statin"
    ],
    "monitoring_recommendations": [
      "Liver function tests at baseline and 3 months",
      "Muscle enzyme monitoring",
      "Lipid panel every 3-6 months"
    ]
  }
]
```

#### Treatment Outcome Prediction

##### POST /ai/predict/treatment-outcome
Predict treatment outcomes.

**Request Body:**
```json
{
  "patient_id": "PAT001",
  "treatment_plan": {
    "primary_diagnosis": "Type 2 Diabetes",
    "medications": [
      {"name": "metformin", "dosage": "500mg twice daily"},
      {"name": "glipizide", "dosage": "5mg daily"}
    ],
    "lifestyle_interventions": [
      "Medical nutrition therapy",
      "Regular exercise program",
      "Weight management"
    ],
    "monitoring_schedule": [
      {"test": "HbA1c", "frequency": "quarterly"},
      {"test": "lipid_panel", "frequency": "annually"}
    ]
  },
  "patient_factors": {
    "age": 55,
    "duration_of_disease": "2 years",
    "baseline_hba1c": 8.5,
    "comorbidities": ["hypertension", "obesity"],
    "adherence_history": 0.8
  }
}
```

**Response (200):**
```json
{
  "patient_id": "PAT001",
  "success_probability": 0.78,
  "expected_improvement_timeline": "3-6 months",
  "projected_outcomes": {
    "hba1c_reduction": "1.5-2.0%",
    "weight_loss": "5-10%",
    "complications_risk_reduction": "25%"
  },
  "potential_complications": [
    {
      "complication": "Hypoglycemia",
      "probability": 0.15,
      "preventive_measures": ["Patient education", "Regular glucose monitoring"]
    },
    {
      "complication": "Gastrointestinal side effects",
      "probability": 0.25,
      "preventive_measures": ["Start with low dose", "Take with meals"]
    }
  ],
  "monitoring_schedule": [
    "Weekly glucose monitoring for first month",
    "Monthly HbA1c checks",
    "Quarterly comprehensive diabetes assessment",
    "Annual eye and foot exams"
  ],
  "intervention_adjustments": [
    {
      "condition": "Poor glycemic control at 3 months",
      "action": "Increase metformin dose or add second agent"
    },
    {
      "condition": "Significant hypoglycemia",
      "action": "Reduce glipizide dose or discontinue"
    }
  ]
}
```

### Drug Discovery

#### Drug Discovery Analysis

##### POST /drug-discovery/analyze
Initiate drug discovery process.

**Request Body:**
```json
{
  "target_protein": "EGFR",
  "disease_context": "Non-small cell lung cancer",
  "patient_profile": {
    "genetic_mutations": ["EGFR_L858R", "TP53_R175H"],
    "resistance_mutations": ["T790M"],
    "biomarker_status": {
      "pd_l1_expression": "negative",
      "alk_rearrangement": "negative",
      "ros1_rearrangement": "negative"
    }
  },
  "search_parameters": {
    "chemical_space": "kinase_inhibitors",
    "binding_affinity_threshold": -8.0,
    "toxicity_filters": ["hepatotoxicity", "cardiotoxicity"],
    "pharmacokinetic_filters": ["oral_bioavailability", "half_life"],
    "structural_constraints": ["molecular_weight_300-500", "logp_-2_to_5"]
  }
}
```

**Response (202):**
```json
{
  "job_id": "drug_discovery_1705312200_def456",
  "status": "processing",
  "target_protein": "EGFR",
  "estimated_completion": "30-60 minutes",
  "message": "Drug discovery analysis initiated"
}
```

##### GET /drug-discovery/results/{job_id}
Get drug discovery results.

**Response (200):**
```json
{
  "job_id": "drug_discovery_1705312200_def456",
  "status": "completed",
  "target_protein": "EGFR",
  "compounds_identified": 1250,
  "compounds_screened": 50000,
  "lead_compounds": [
    {
      "smiles": "C1CCN(CC1)C2=CC=C(C=C2)NC3=NC=C(C(=N3)NC4=CC=CC(=C4)OC)C5=CC=CC=C5",
      "molecular_weight": 412.5,
      "logp": 4.2,
      "binding_energy": -9.8,
      "binding_affinity_score": 0.95,
      "toxicity_score": 0.15,
      "efficacy_score": 0.88,
      "selectivity_score": 0.92,
      "pk_properties": {
        "oral_bioavailability": 65,
        "half_life_hours": 8.5,
        "clearance_rate": 12.3,
        "volume_distribution": 45.2
      },
      "structural_features": {
        "scaffold_type": "quinazoline",
        "functional_groups": ["amine", "ether", "aromatic"],
        "chirality_centers": 1
      },
      "predicted_resistance_profile": {
        "t790m_resistance": "low",
        "c797s_resistance": "moderate",
        "brain_penetration": "good"
      }
    }
  ],
  "analysis_summary": {
    "virtual_screening_score": 0.85,
    "hit_rate": 0.025,
    "enrichment_factor": 12.5,
    "novelty_score": 0.75
  },
  "recommendations": [
    "Prioritize compounds with binding energy < -9.5 kcal/mol",
    "Focus on molecules with good brain penetration for CNS tumors",
    "Consider structural modifications to reduce toxicity",
    "Validate top hits with experimental assays"
  ]
}
```

### Health Monitoring

#### Real-time Health Data

##### POST /health-monitoring/data
Submit health monitoring data.

**Request Body:**
```json
{
  "patient_id": "PAT001",
  "timestamp": "2024-01-15T14:30:00Z",
  "vital_signs": {
    "heart_rate": 72,
    "blood_pressure_systolic": 128,
    "blood_pressure_diastolic": 82,
    "temperature": 98.6,
    "respiratory_rate": 16,
    "oxygen_saturation": 98.0,
    "weight": 80.2,
    "height": 175.0,
    "pain_scale": null,
    "blood_glucose": 95.0
  },
  "biomarkers": {
    "glucose": 95.0,
    "cholesterol_total": 185.0,
    "cholesterol_hdl": 50.0,
    "cholesterol_ldl": 115.0,
    "triglycerides": 120.0,
    "creatinine": 0.9,
    "bun": 15.0,
    "alt": 25.0,
    "ast": 22.0,
    "tsh": 2.1,
    "free_t4": 1.2,
    "vitamin_d": 32.0
  },
  "symptoms": [
    "mild_fatigue",
    "occasional_headache"
  ],
  "device_info": {
    "type": "smartwatch",
    "id": "SW001",
    "model": "Apple Watch Series 8",
    "firmware_version": "9.1.0",
    "battery_level": 85
  },
  "environmental_data": {
    "location": "home",
    "activity_level": "sedentary",
    "sleep_quality_last_night": "good"
  }
}
```

**Response (200):**
```json
{
  "status": "processed",
  "patient_id": "PAT001",
  "timestamp": "2024-01-15T14:30:00Z",
  "alerts": [
    {
      "alert_id": "alert_1705324200_001",
      "severity": "low",
      "title": "Slightly Elevated Glucose",
      "message": "Blood glucose (95 mg/dL) is at the upper end of normal range",
      "recommendation": "Monitor diet and exercise patterns",
      "timestamp": "2024-01-15T14:30:00Z",
      "acknowledged": false
    }
  ],
  "recommendations": [
    "Continue regular glucose monitoring",
    "Maintain current healthy diet and exercise routine",
    "Schedule routine follow-up in 3 months"
  ],
  "next_check_interval": "24 hours",
  "data_quality_score": 0.95,
  "processed_at": "2024-01-15T14:30:05Z"
}
```

##### GET /health-monitoring/alerts/{patient_id}
Get health alerts for patient.

**Query Parameters:**
- `severity`: critical|high|medium|low
- `status`: active|acknowledged|resolved
- `start_date`: Start date for alerts
- `end_date`: End date for alerts

**Response (200):**
```json
[
  {
    "alert_id": "alert_1705324200_001",
    "severity": "low",
    "title": "Slightly Elevated Glucose",
    "message": "Blood glucose (95 mg/dL) is at the upper end of normal range",
    "recommendation": "Monitor diet and exercise patterns",
    "timestamp": "2024-01-15T14:30:00Z",
    "acknowledged": false,
    "acknowledged_at": null,
    "resolved": false,
    "resolved_at": null,
    "follow_up_required": false
  }
]
```

##### GET /health-monitoring/devices/{patient_id}
Get connected health monitoring devices.

**Response (200):**
```json
[
  {
    "device_id": "SW001",
    "type": "smartwatch",
    "model": "Apple Watch Series 8",
    "status": "connected",
    "last_sync": "2024-01-15T14:30:00Z",
    "battery_level": 85,
    "capabilities": [
      "heart_rate",
      "blood_pressure",
      "oxygen_saturation",
      "steps",
      "sleep_tracking",
      "ecg"
    ],
    "firmware_version": "9.1.0",
    "data_quality_score": 0.92,
    "registered_at": "2024-01-01T00:00:00Z"
  }
]
```

##### GET /health-monitoring/dashboard/{patient_id}
Get comprehensive health monitoring dashboard.

**Response (200):**
```json
{
  "patient_id": "PAT001",
  "time_range": "30 days",
  "summary": {
    "total_measurements": 840,
    "data_completeness": 0.92,
    "alert_count": 3,
    "trend_direction": "stable"
  },
  "vital_signs_trends": {
    "heart_rate": {
      "current_value": 72,
      "average_30d": 74,
      "min_30d": 58,
      "max_30d": 95,
      "trend": "stable",
      "normal_range": [60, 100]
    },
    "blood_pressure": {
      "current_systolic": 128,
      "current_diastolic": 82,
      "average_systolic_30d": 132,
      "average_diastolic_30d": 84,
      "trend": "improving",
      "normal_range": [90, 120]
    }
  },
  "biomarker_trends": {
    "glucose": {
      "current_value": 95,
      "average_30d": 98,
      "trend": "stable",
      "normal_range": [70, 100]
    }
  },
  "recent_alerts": [
    {
      "alert_id": "alert_1705324200_001",
      "severity": "low",
      "title": "Slightly Elevated Glucose",
      "timestamp": "2024-01-15T14:30:00Z",
      "status": "active"
    }
  ],
  "recommendations": [
    "Continue current health monitoring schedule",
    "Maintain healthy diet and exercise routine",
    "Schedule routine checkup in 2 weeks"
  ],
  "next_actions": [
    {
      "action": "Blood pressure check",
      "due_date": "2024-01-16T08:00:00Z",
      "priority": "medium"
    },
    {
      "action": "Weekly health summary review",
      "due_date": "2024-01-21T09:00:00Z",
      "priority": "low"
    }
  ]
}
```

### Treatment Planning

#### Treatment Plan Management

##### POST /treatment-plans
Create a personalized treatment plan.

**Request Body:**
```json
{
  "patient_id": "PAT001",
  "specialty": "Cardiology",
  "primary_diagnosis": "Hypertension",
  "secondary_diagnoses": ["Dyslipidemia"],
  "treatment_goals": [
    "Reduce blood pressure to <130/80 mmHg",
    "Lower LDL cholesterol to <100 mg/dL",
    "Improve cardiovascular risk profile"
  ],
  "medications": [
    {
      "name": "lisinopril",
      "dosage": "10mg daily",
      "indication": "Blood pressure control",
      "start_date": "2024-01-15",
      "monitoring": ["blood_pressure", "renal_function"]
    },
    {
      "name": "atorvastatin",
      "dosage": "20mg daily",
      "indication": "Dyslipidemia management",
      "start_date": "2024-01-15",
      "monitoring": ["lipid_panel", "liver_function"]
    }
  ],
  "lifestyle_modifications": [
    "DASH diet with sodium restriction <2g/day",
    "Regular aerobic exercise 150 minutes/week",
    "Weight management to achieve BMI <25",
    "Smoking cessation if applicable"
  ],
  "procedures": [
    {
      "name": "Echocardiogram",
      "timing": "Within 1 month",
      "indication": "Assess cardiac structure and function"
    },
    {
      "name": "Stress test",
      "timing": "Within 3 months",
      "indication": "Evaluate exercise capacity and cardiac response"
    }
  ],
  "monitoring_schedule": [
    {
      "parameter": "Blood pressure",
      "frequency": "Weekly at home, monthly in clinic",
      "target_range": "<130/80 mmHg"
    },
    {
      "parameter": "Lipid panel",
      "frequency": "Every 3 months",
      "target_range": "LDL <100 mg/dL"
    },
    {
      "parameter": "Renal function",
      "frequency": "Every 6 months",
      "target_range": "eGFR >60 mL/min/1.73m²"
    }
  ],
  "follow_up_schedule": [
    {
      "timing": "2 weeks",
      "purpose": "Medication titration and side effect assessment"
    },
    {
      "timing": "1 month",
      "purpose": "Comprehensive evaluation of treatment response"
    },
    {
      "timing": "3 months",
      "purpose": "Long-term treatment efficacy assessment"
    }
  ],
  "contingency_plans": [
    {
      "condition": "Inadequate BP control on dual therapy",
      "action": "Add third antihypertensive agent or switch to combination pill"
    },
    {
      "condition": "Statin intolerance",
      "action": "Switch to alternative lipid-lowering agent (ezetimibe, PCSK9 inhibitor)"
    }
  ],
  "patient_education": [
    "Hypertension management and lifestyle modifications",
    "Medication adherence strategies",
    "Recognition of medication side effects",
    "Importance of regular follow-up"
  ]
}
```

**Response (201):**
```json
{
  "plan_id": "plan_1705312200_ghi789",
  "patient_id": "PAT001",
  "specialty": "Cardiology",
  "primary_diagnosis": "Hypertension",
  "status": "active",
  "created_by": "Dr. Smith",
  "created_at": "2024-01-15T10:00:00Z",
  "estimated_duration_months": 12,
  "success_probability": 0.82,
  "message": "Treatment plan created successfully"
}
```

##### GET /treatment-plans/{patient_id}
Get treatment plans for patient.

**Query Parameters:**
- `status`: active|completed|discontinued
- `specialty`: Filter by medical specialty

**Response (200):**
```json
[
  {
    "plan_id": "plan_1705312200_ghi789",
    "patient_id": "PAT001",
    "specialty": "Cardiology",
    "primary_diagnosis": "Hypertension",
    "secondary_diagnoses": ["Dyslipidemia"],
    "treatment_goals": [
      "Reduce blood pressure to <130/80 mmHg",
      "Lower LDL cholesterol to <100 mg/dL"
    ],
    "medications": [
      {
        "name": "lisinopril",
        "dosage": "10mg daily",
        "indication": "Blood pressure control",
        "start_date": "2024-01-15",
        "monitoring": ["blood_pressure", "renal_function"]
      }
    ],
    "lifestyle_modifications": [
      "DASH diet with sodium restriction <2g/day",
      "Regular aerobic exercise 150 minutes/week"
    ],
    "monitoring_schedule": [
      {
        "parameter": "Blood pressure",
        "frequency": "Weekly at home, monthly in clinic",
        "target_range": "<130/80 mmHg"
      }
    ],
    "follow_up_schedule": [
      {
        "timing": "2 weeks",
        "purpose": "Medication titration and side effect assessment"
      }
    ],
    "status": "active",
    "created_by": "Dr. Smith",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:00:00Z"
  }
]
```

### Clinical Decision Support

#### Decision Support Queries

##### POST /clinical-support/advise
Get clinical decision support recommendations.

**Request Body:**
```json
{
  "patient_id": "PAT001",
  "query": "Patient with new onset chest pain, what diagnostic workup is recommended?",
  "context": {
    "age": 55,
    "gender": "M",
    "risk_factors": ["hypertension", "smoking", "family_history"],
    "symptoms": ["retrosternal_chest_pain", "radiates_to_left_arm", "shortness_of_breath"],
    "symptom_duration": "2 hours",
    "pain_characteristics": {
      "severity": "7/10",
      "character": "pressure",
      "radiation": "left arm and jaw",
      "associated_symptoms": ["nausea", "diaphoresis"]
    },
    "vital_signs": {
      "blood_pressure": "160/95",
      "heart_rate": 95,
      "respiratory_rate": 20,
      "oxygen_saturation": 96
    },
    "physical_exam": {
      "general": "anxious_appearing",
      "cardiovascular": "regular_rhythm_no_murmurs",
      "respiratory": "clear_to_auscultation"
    },
    "ecg_findings": "normal_sinus_rhythm_no_st_changes",
    "lab_results": {
      "troponin_i": "0.02 ng/mL",
      "ck_mb": "3.2 ng/mL",
      "total_ck": "145 U/L"
    }
  },
  "urgency": "high",
  "differential_diagnosis": [
    "acute_coronary_syndrome",
    "pulmonary_embolism",
    "aortic_dissection",
    "musculoskeletal_chest_pain"
  ]
}
```

**Response (200):**
```json
{
  "query": "Patient with new onset chest pain, what diagnostic workup is recommended?",
  "urgency_assessment": "high",
  "recommendations": [
    {
      "type": "diagnostic",
      "priority": "immediate",
      "procedure": "Serial cardiac biomarkers (troponin)",
      "rationale": "Rule out myocardial injury in patient with concerning chest pain",
      "evidence_level": "A",
      "urgency": "STAT"
    },
    {
      "type": "diagnostic",
      "priority": "immediate",
      "procedure": "ECG",
      "rationale": "Essential first test in chest pain evaluation",
      "evidence_level": "A",
      "urgency": "immediate"
    },
    {
      "type": "diagnostic",
      "priority": "urgent",
      "procedure": "Chest X-ray",
      "rationale": "Evaluate for pulmonary pathology",
      "evidence_level": "B",
      "urgency": "within_1_hour"
    },
    {
      "type": "diagnostic",
      "priority": "urgent",
      "procedure": "CT pulmonary angiogram",
      "rationale": "Rule out pulmonary embolism given risk factors",
      "evidence_level": "A",
      "urgency": "within_4_hours"
    },
    {
      "type": "management",
      "priority": "immediate",
      "action": "Aspirin 325mg PO",
      "rationale": "Antiplatelet therapy for suspected ACS",
      "evidence_level": "A",
      "urgency": "immediate"
    },
    {
      "type": "management",
      "priority": "immediate",
      "action": "Nitroglycerin 0.4mg SL",
      "rationale": "Vasodilation for ischemic chest pain",
      "evidence_level": "B",
      "urgency": "immediate"
    }
  ],
  "differential_diagnosis_analysis": {
    "most_likely": "acute_coronary_syndrome",
    "probability": 0.65,
    "key_features": [
      "retrosternal location",
      "radiation to left arm",
      "diaphoresis",
      "hypertension"
    ]
  },
  "risk_stratification": {
    "timI_risk_score": 3,
    "timI_risk_category": "intermediate",
    "grace_score": 125,
    "grace_risk_category": "intermediate"
  },
  "alternative_options": [
    {
      "scenario": "Low suspicion for ACS",
      "alternative_workup": [
        "Outpatient stress testing",
        "Holter monitoring",
        "Gastroenterology evaluation for GERD"
      ],
      "rationale": "If cardiac enzymes and ECG normal, consider outpatient evaluation"
    }
  ],
  "follow_up_actions": [
    "Admit to telemetry unit for monitoring",
    "Consult cardiology for possible cardiac catheterization",
    "Start heparin if pulmonary embolism suspected",
    "Pain control and anxiolysis",
    "Notify family of admission"
  ],
  "evidence_summary": {
    "guidelines_cited": [
      "ACC/AHA 2021 Chest Pain Guidelines",
      "ESC 2020 NSTE-ACS Guidelines"
    ],
    "evidence_level": "A",
    "confidence_score": 0.92
  },
  "disclaimer": "This is decision support only and should not replace clinical judgment. Consider patient-specific factors and institutional protocols."
}
```

### Virtual Assistant

#### Patient Engagement

##### POST /assistant/query
Interact with virtual health assistant.

**Request Body:**
```json
{
  "patient_id": "PAT001",
  "message": "I'm experiencing chest pain and shortness of breath. What should I do?",
  "context": {
    "current_location": "home",
    "time_of_day": "evening",
    "recent_activities": ["exercise", "stressful_meeting"],
    "current_medications": ["lisinopril", "metoprolol"],
    "known_conditions": ["hypertension", "anxiety"]
  },
  "conversation_history": [
    {
      "role": "patient",
      "message": "I've been feeling more anxious lately",
      "timestamp": "2024-01-15T18:00:00Z"
    },
    {
      "role": "assistant",
      "message": "I understand anxiety can be challenging. Have you tried relaxation techniques?",
      "timestamp": "2024-01-15T18:02:00Z"
    }
  ]
}
```

**Response (200):**
```json
{
  "response": "I'm concerned about your symptoms of chest pain and shortness of breath. These could be serious and require immediate medical attention. Please call emergency services (911) right away or go to the nearest emergency room. While waiting for help, sit down and try to stay calm.",
  "actions": [
    {
      "type": "emergency_contact",
      "label": "Call Emergency Services",
      "action": "call_911",
      "urgent": true
    },
    {
      "type": "location_services",
      "label": "Find Nearest ER",
      "action": "find_emergency_room",
      "urgent": true
    },
    {
      "type": "symptom_tracker",
      "label": "Log Current Symptoms",
      "action": "log_symptoms",
      "data": {
        "symptoms": ["chest_pain", "shortness_of_breath"],
        "severity": "severe",
        "onset": "sudden"
      }
    }
  ],
  "follow_up_questions": [
    "Are you experiencing any other symptoms like dizziness or nausea?",
    "Do you have any chest pain or cardiac history?",
    "Are you currently taking any medications?"
  ],
  "confidence_score": 0.95,
  "escalation_level": "emergency",
  "sources": [
    "American Heart Association Guidelines",
    "Emergency Medical Protocols",
    "Cardiology Best Practices"
  ],
  "next_steps": [
    "Stay on the line with emergency services",
    "Do not drive yourself to the hospital",
    "Have someone stay with you until help arrives"
  ]
}
```

### Research Tools

#### Clinical Trials

##### POST /research/clinical-trials
Create a clinical trial.

**Request Body:**
```json
{
  "trial_id": "CARDIO_PREVENT_2024",
  "nct_id": "NCT04567890",
  "title": "Cardiovascular Prevention in High-Risk Patients",
  "phase": "III",
  "status": "recruiting",
  "condition": "Cardiovascular Disease",
  "intervention": "Drug: Atorvastatin 40mg daily + Lifestyle counseling",
  "description": "A randomized controlled trial evaluating intensive statin therapy combined with lifestyle intervention for primary prevention of cardiovascular events in high-risk patients.",
  "enrollment_target": 2000,
  "minimum_age": 40,
  "maximum_age": 75,
  "gender": "both",
  "inclusion_criteria": [
    "Age 40-75 years",
    "No known cardiovascular disease",
    "LDL cholesterol ≥130 mg/dL or calculated 10-year ASCVD risk ≥7.5%",
    "Willing to participate in lifestyle counseling"
  ],
  "exclusion_criteria": [
    "Known cardiovascular disease",
    "Current statin use",
    "Liver disease (ALT >3x ULN)",
    "Pregnancy or lactation"
  ],
  "study_design": {
    "allocation": "randomized",
    "intervention_model": "parallel_assignment",
    "masking": "none_open_label",
    "purpose": "prevention"
  },
  "locations": [
    {
      "facility": "General Hospital",
      "city": "New York",
      "state": "NY",
      "country": "US",
      "contact": "Dr. John Smith",
      "phone": "+1-212-555-0123"
    }
  ],
  "sponsor": "National Institutes of Health",
  "principal_investigator": "Dr. Sarah Johnson, MD",
  "start_date": "2024-02-01",
  "completion_date": "2027-01-31",
  "primary_outcome": "Time to first cardiovascular event (composite of myocardial infarction, stroke, cardiovascular death)",
  "secondary_outcomes": [
    "Change in LDL cholesterol from baseline",
    "Achievement of LDL <70 mg/dL",
    "Cardiovascular risk factor control",
    "Quality of life measures"
  ]
}
```

**Response (201):**
```json
{
  "trial_id": "CARDIO_PREVENT_2024",
  "status": "created",
  "message": "Clinical trial created successfully",
  "next_steps": [
    "Submit to ClinicalTrials.gov for NCT number assignment",
    "Obtain IRB approval",
    "Begin site activation and recruitment"
  ]
}
```

##### GET /research/clinical-trials/matching/{patient_id}
Find clinical trials matching patient criteria.

**Query Parameters:**
- `condition`: Specific condition to match
- `phase`: Trial phase preference
- `distance`: Maximum distance from patient location (miles)

**Response (200):**
```json
[
  {
    "trial_id": "CARDIO_PREVENT_2024",
    "nct_id": "NCT04567890",
    "title": "Cardiovascular Prevention in High-Risk Patients",
    "phase": "III",
    "condition": "Cardiovascular Disease",
    "eligibility_score": 0.85,
    "matching_criteria": [
      "Age within range (45-75)",
      "No known cardiovascular disease",
      "Elevated LDL cholesterol",
      "High ASCVD risk score"
    ],
    "exclusion_criteria_met": [],
    "distance_to_nearest_site": 5.2,
    "nearest_site": {
      "facility": "General Hospital",
      "address": "123 Medical Center Dr, New York, NY",
      "contact": "Dr. John Smith"
    },
    "compensation": "$500 for completed participation",
    "time_commitment": "12 months with 8 study visits",
    "potential_benefits": [
      "Free cardiovascular screening and treatment",
      "Close monitoring of heart health",
      "Potential access to new treatments"
    ],
    "risks": [
      "Possible side effects from study medication",
      "Additional blood draws and testing",
      "Time commitment for study visits"
    ]
  }
]
```

### Blockchain Security

#### Secure Health Records

##### POST /blockchain/records
Create a blockchain-secured health record.

**Request Body:**
```json
{
  "patient_id": "PAT001",
  "record_type": "medication_prescription",
  "data": {
    "prescription_id": "RX123456",
    "medication": "lisinopril",
    "dosage": "10mg daily",
    "prescribing_physician": "Dr. Smith",
    "date_prescribed": "2024-01-15",
    "indication": "Hypertension",
    "instructions": "Take one tablet by mouth daily"
  },
  "consent_given": true,
  "access_permissions": {
    "owner": "patient",
    "authorized_providers": ["Dr. Smith", "General Hospital"],
    "emergency_access": true,
    "research_use": false
  }
}
```

**Response (201):**
```json
{
  "record_id": "record_1705312200_jkl012",
  "block_hash": "a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef",
  "transaction_hash": "tx_abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
  "block_number": 15467890,
  "timestamp": "2024-01-15T10:00:00Z",
  "verification_status": "confirmed",
  "immutable_proof": {
    "previous_block_hash": "prev_block_hash_here",
    "merkle_root": "merkle_root_hash_here",
    "confirmations": 12
  }
}
```

##### GET /blockchain/records/verify/{record_id}
Verify record integrity on blockchain.

**Response (200):**
```json
{
  "record_id": "record_1705312200_jkl012",
  "verification_status": "valid",
  "blockchain_status": "confirmed",
  "tamper_evidence": false,
  "last_verification": "2024-01-15T10:05:00Z",
  "block_details": {
    "block_hash": "a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef",
    "block_number": 15467890,
    "confirmations": 12,
    "timestamp": "2024-01-15T10:00:00Z"
  },
  "audit_trail": [
    {
      "event": "record_created",
      "timestamp": "2024-01-15T10:00:00Z",
      "actor": "healthcare_platform",
      "action": "create"
    },
    {
      "event": "record_verified",
      "timestamp": "2024-01-15T10:00:15Z",
      "actor": "blockchain_network",
      "action": "verify"
    }
  ]
}
```

### Administration

#### System Management

##### GET /admin/system/metrics
Get system performance metrics.

**Response (200):**
```json
{
  "system_health": "healthy",
  "uptime_seconds": 2592000,
  "active_users": {
    "total": 1250,
    "physicians": 85,
    "patients": 1120,
    "nurses": 35,
    "administrators": 10
  },
  "api_performance": {
    "total_requests": 45230456,
    "average_response_time_ms": 245.7,
    "error_rate_percent": 0.023,
    "requests_per_second": 156.8,
    "endpoint_performance": {
      "/api/v1/genomics/analyze": {"avg_time": 1250.5, "requests": 1234},
      "/api/v1/health-monitoring/data": {"avg_time": 45.2, "requests": 45678},
      "/api/v1/patients/vitals": {"avg_time": 89.3, "requests": 23456}
    }
  },
  "genomic_analyses": {
    "total_completed": 15678,
    "average_completion_time_hours": 2.5,
    "success_rate_percent": 98.7,
    "queue_length": 23,
    "processing_capacity": 95
  },
  "drug_discovery": {
    "active_jobs": 12,
    "completed_analyses": 2341,
    "average_compounds_screened": 50000,
    "hit_rate_percent": 2.3
  },
  "database_performance": {
    "connections_active": 12,
    "connections_idle": 8,
    "query_cache_hit_rate": 87.5,
    "slow_queries_count": 23,
    "average_query_time_ms": 15.7
  },
  "cache_performance": {
    "redis_hit_rate": 89.2,
    "memory_usage_mb": 1024,
    "eviction_rate": 0.05
  },
  "storage_usage": {
    "genomic_data_tb": 45.7,
    "imaging_data_tb": 123.4,
    "patient_records_tb": 67.8,
    "total_storage_tb": 236.9,
    "growth_rate_tb_per_month": 2.1
  },
  "security_metrics": {
    "failed_login_attempts": 1456,
    "blocked_ips": 23,
    "encryption_operations": 1234567,
    "audit_logs_generated": 4567890
  },
  "external_integrations": {
    "fhir_endpoints": {"status": "healthy", "response_time_ms": 234},
    "laboratory_interfaces": {"status": "healthy", "pending_results": 456},
    "pharmacy_systems": {"status": "degraded", "error_rate": 0.15},
    "insurance_providers": {"status": "healthy", "claim_processing_time_days": 2.1}
  }
}
```

##### GET /admin/audit/logs
Get audit logs for compliance and security monitoring.

**Query Parameters:**
- `start_date`: Start date for logs
- `end_date`: End date for logs
- `event_type`: Filter by event type
- `user_id`: Filter by user
- `resource_type`: Filter by resource type
- `limit`: Maximum number of logs (default: 100)

**Response (200):**
```json
[
  {
    "id": 123456,
    "event_id": "audit_1705312200_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "event_type": "data_access",
    "event_category": "patient_data",
    "user_id": "user_789",
    "session_id": "session_abc123",
    "patient_id": "PAT001",
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "resource_type": "patient_record",
    "resource_id": "PAT001",
    "action": "view",
    "success": true,
    "old_values": null,
    "new_values": null,
    "metadata": {
      "access_reason": "patient_care",
      "duration_ms": 1250,
      "data_elements_accessed": ["vital_signs", "medications"]
    },
    "phi_access": true,
    "compliance_flags": ["hipaa_access", "audit_required"]
  }
]
```

### Error Handling

All API endpoints follow consistent error response formats:

#### HTTP Status Codes
- `200`: Success
- `201`: Created
- `202`: Accepted (asynchronous processing)
- `400`: Bad Request (validation errors)
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `409`: Conflict
- `422`: Unprocessable Entity
- `429`: Too Many Requests
- `500`: Internal Server Error
- `503`: Service Unavailable

#### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field_errors": {
        "email": ["Invalid email format"],
        "password": ["Password must be at least 8 characters"]
      }
    }
  },
  "request_id": "req_abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Rate Limiting

API requests are subject to rate limiting:

- **Authenticated users**: 1000 requests per hour
- **Genomic analysis**: 10 concurrent analyses per user
- **Health monitoring data**: 1000 data points per hour
- **Drug discovery**: 5 concurrent jobs per user

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1640995200
```

### Pagination

List endpoints support pagination:

**Query Parameters:**
- `page`: Page number (1-based, default: 1)
- `limit`: Items per page (default: 20, max: 100)
- `sort`: Sort field (e.g., "created_at")
- `order`: Sort order ("asc" or "desc")

**Response Format:**
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "total_pages": 8,
    "has_next": true,
    "has_previous": false
  }
}
```

### Versioning

API versioning follows URL path versioning:

- Current version: `v2` (recommended)
- Previous version: `v1` (deprecated, sunset date: 2025-12-31)

Breaking changes will result in new major versions. Minor updates are backward compatible.

### Webhooks

The platform supports webhooks for real-time notifications:

#### Supported Events
- `patient.created`
- `appointment.scheduled`
- `genomic_analysis.completed`
- `health_alert.triggered`
- `medication.prescribed`

#### Webhook Payload Format
```json
{
  "event": "patient.created",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "patient_id": "PAT001",
    "event_specific_data": {...}
  },
  "webhook_id": "wh_abc123"
}
```

### SDKs and Libraries

Official SDKs are available for:
- **Python**: `pip install healthcare-platform-sdk`
- **JavaScript/Node.js**: `npm install @healthcare-platform/sdk`
- **Java**: Available via Maven Central
- **C#**: Available via NuGet

### Support

- **Documentation**: https://docs.healthcare-platform.com
- **API Reference**: https://api.healthcare-platform.com/docs
- **Support Portal**: https://support.healthcare-platform.com
- **Status Page**: https://status.healthcare-platform.com
- **Developer Community**: https://community.healthcare-platform.com

### Changelog

#### Version 2.0.0 (Current)
- Complete API redesign with improved performance
- Added genomic analysis and drug discovery endpoints
- Enhanced security with blockchain integration
- Real-time health monitoring capabilities
- Advanced AI/ML model integration
- Comprehensive audit logging and compliance features

#### Version 1.5.0
- Added treatment planning endpoints
- Enhanced clinical decision support
- Virtual assistant integration
- Research tools for clinical trials

#### Version 1.0.0
- Initial release with core patient management
- Basic appointment scheduling
- Vital signs tracking
- Prescription management

This comprehensive API documentation provides detailed information about all endpoints, request/response formats, authentication, error handling, and integration guidelines for the AI Personalized Medicine Platform.
