# Comprehensive API Reference
## AI Personalized Medicine Platform

This document provides a complete reference for all APIs available in the AI Personalized Medicine Platform, including FastAPI endpoints, GraphQL schema, WebSocket protocols, and integration APIs.

## Table of Contents

1. [REST API Endpoints](#rest-api-endpoints)
2. [GraphQL API](#graphql-api)
3. [WebSocket API](#websocket-api)
4. [Authentication & Authorization](#authentication--authorization)
5. [Rate Limiting](#rate-limiting)
6. [API Versioning](#api-versioning)
7. [Error Handling](#error-handling)
8. [Data Models](#data-models)
9. [Integration APIs](#integration-apis)
10. [SDKs & Libraries](#sdks--libraries)

## REST API Endpoints

### Base URL
```
https://api.ai-personalized-medicine.com/v1
```

### Authentication
All API requests require authentication via JWT tokens:
```
Authorization: Bearer <jwt_token>
```

### Core Endpoints

#### Patient Management

##### Create Patient Profile
```http
POST /api/patients
```

**Request Body:**
```json
{
  "patient_id": "string",
  "demographics": {
    "first_name": "string",
    "last_name": "string",
    "date_of_birth": "2023-01-01",
    "gender": "male|female|other",
    "ethnicity": "string",
    "contact": {
      "email": "string",
      "phone": "string",
      "address": {
        "street": "string",
        "city": "string",
        "state": "string",
        "zip_code": "string",
        "country": "string"
      }
    }
  },
  "medical_history": {
    "allergies": ["string"],
    "chronic_conditions": ["string"],
    "medications": ["string"],
    "surgeries": ["string"],
    "family_history": {
      "diabetes": "boolean",
      "cancer": "boolean",
      "heart_disease": "boolean",
      "other": ["string"]
    }
  },
  "consent": {
    "data_processing": "boolean",
    "research_participation": "boolean",
    "emergency_contact": "boolean"
  }
}
```

**Response:**
```json
{
  "patient_id": "string",
  "created_at": "2023-01-01T00:00:00Z",
  "status": "active",
  "profile_completion": 85
}
```

##### Get Patient Profile
```http
GET /api/patients/{patient_id}
```

**Response:**
```json
{
  "patient_id": "string",
  "demographics": { ... },
  "medical_history": { ... },
  "current_health_status": {
    "vital_signs": {
      "heart_rate": 72,
      "blood_pressure": "120/80",
      "temperature": 98.6,
      "oxygen_saturation": 98,
      "respiratory_rate": 16
    },
    "biometrics": {
      "weight": 70.5,
      "height": 175,
      "bmi": 23.0,
      "body_fat_percentage": 15.2
    },
    "last_updated": "2023-01-01T12:00:00Z"
  },
  "risk_assessment": {
    "overall_risk": "low",
    "risk_factors": ["family_history", "lifestyle"],
    "preventive_measures": ["regular_checkups", "dietary_changes"]
  }
}
```

##### Update Patient Profile
```http
PUT /api/patients/{patient_id}
```

**Request Body:** Same as create patient profile

##### Delete Patient Profile
```http
DELETE /api/patients/{patient_id}
```

#### Genomic Analysis

##### Initiate Genomic Analysis
```http
POST /api/genomic-analysis
```

**Request Body:**
```json
{
  "patient_id": "string",
  "sample_type": "blood|saliva|tissue",
  "analysis_type": "comprehensive|targeted|cancer_risk|drug_response",
  "genetic_data": {
    "format": "vcf|bam|fastq",
    "file_urls": ["string"],
    "reference_genome": "GRCh38|GRCh37",
    "sequencing_type": "wgs|wes|panel"
  },
  "consent_for_research": "boolean",
  "priority": "routine|urgent|stat"
}
```

**Response:**
```json
{
  "analysis_id": "string",
  "patient_id": "string",
  "status": "queued",
  "estimated_completion": "2023-01-02T10:00:00Z",
  "queue_position": 5
}
```

##### Get Genomic Analysis Results
```http
GET /api/genomic-results/{patient_id}
```

**Query Parameters:**
- `analysis_type`: Filter by analysis type
- `date_from`: Filter results after date
- `date_to`: Filter results before date

**Response:**
```json
{
  "patient_id": "string",
  "analyses": [
    {
      "analysis_id": "string",
      "analysis_type": "comprehensive",
      "status": "completed",
      "completed_at": "2023-01-01T15:30:00Z",
      "results": {
        "genetic_variants": {
          "pathogenic": [
            {
              "gene": "BRCA1",
              "variant": "c.68_69delAG",
              "zygosity": "heterozygous",
              "clinical_significance": "pathogenic",
              "disease_association": "breast_cancer",
              "risk_increase": "high"
            }
          ],
          "benign": [...],
          "vus": [...]
        },
        "pharmacogenomics": {
          "drug_responses": [
            {
              "drug": "warfarin",
              "gene": "CYP2C9",
              "metabolizer_status": "poor_metabolizer",
              "recommended_dose_adjustment": "reduce_by_50%",
              "monitoring_required": "INR_monitoring"
            }
          ]
        },
        "carrier_status": {
          "recessive_conditions": [
            {
              "condition": "cystic_fibrosis",
              "gene": "CFTR",
              "carrier_status": "carrier",
              "partner_testing_recommended": "boolean"
            }
          ]
        },
        "polygenic_risk_scores": {
          "disease_risks": [
            {
              "condition": "coronary_artery_disease",
              "risk_percentile": 75,
              "lifetime_risk": 0.15,
              "confidence_interval": "0.12-0.18"
            }
          ]
        }
      },
      "recommendations": [
        "Genetic counseling recommended for BRCA1 variant",
        "Annual breast cancer screening starting at age 25",
        "Consider prophylactic mastectomy consultation"
      ],
      "clinical_action_items": [
        {
          "action": "referral",
          "specialty": "genetics",
          "priority": "high",
          "due_date": "2023-02-01"
        }
      ]
    }
  ]
}
```

##### Get Genomic Analysis Status
```http
GET /api/genomic-analysis/{analysis_id}/status
```

**Response:**
```json
{
  "analysis_id": "string",
  "status": "processing",
  "progress": {
    "current_step": "variant_calling",
    "completed_steps": ["data_ingestion", "quality_control", "alignment"],
    "remaining_steps": ["variant_calling", "annotation", "interpretation"],
    "percent_complete": 60
  },
  "estimated_completion": "2023-01-02T08:00:00Z",
  "issues": []
}
```

#### Drug Discovery

##### Initiate Drug Discovery
```http
POST /api/drug-discovery
```

**Request Body:**
```json
{
  "project_name": "string",
  "target_disease": "string",
  "target_protein": "string",
  "approach": "virtual_screening|de_novo_design|repurposing",
  "constraints": {
    "molecular_weight_range": [100, 500],
    "logp_range": [-2, 5],
    "tpsa_range": [20, 140],
    "excluded_functional_groups": ["string"],
    "required_properties": {
      "blood_brain_barrier_permeability": "high",
      "oral_bioavailability": "high"
    }
  },
  "datasets": {
    "training_data": "string",
    "validation_data": "string",
    "external_databases": ["chembl", "pubchem", "zinc"]
  }
}
```

**Response:**
```json
{
  "project_id": "string",
  "status": "initialized",
  "estimated_candidates": 1000,
  "timeline": {
    "virtual_screening": "2_days",
    "molecular_docking": "3_days",
    "adme_prediction": "1_day",
    "toxicity_prediction": "1_day"
  }
}
```

##### Get Drug Discovery Results
```http
GET /api/drug-discovery/{project_id}/results
```

**Query Parameters:**
- `limit`: Number of results to return (default: 50)
- `sort_by`: Sort criteria (score, novelty, safety)
- `filter_by`: Filter criteria (adme_pass, toxicity_pass)

**Response:**
```json
{
  "project_id": "string",
  "total_candidates": 50000,
  "filtered_candidates": 1500,
  "top_candidates": [
    {
      "candidate_id": "string",
      "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
      "molecular_properties": {
        "molecular_weight": 456.5,
        "logp": 3.2,
        "tpsa": 95.8,
        "hbd": 2,
        "hba": 8,
        "rotatable_bonds": 6
      },
      "predicted_activity": {
        "binding_affinity": -8.5,
        "ic50": 0.023,
        "selectivity_ratio": 1250
      },
      "adme_properties": {
        "solubility": "high",
        "permeability": "high",
        "oral_bioavailability": 0.85,
        "half_life": 8.5
      },
      "toxicity_profile": {
        "hepatotoxicity_risk": "low",
        "cardiotoxicity_risk": "low",
        "mutagenicity_risk": "low",
        "overall_safety_score": 0.92
      },
      "synthetic_feasibility": {
        "score": 0.78,
        "estimated_cost": "medium",
        "route_complexity": "moderate"
      },
      "novelty_score": 0.85,
      "overall_score": 0.91
    }
  ],
  "statistics": {
    "average_binding_affinity": -7.2,
    "adme_pass_rate": 0.65,
    "toxicity_pass_rate": 0.78,
    "novelty_distribution": {
      "high_novelty": 0.25,
      "medium_novelty": 0.45,
      "low_novelty": 0.30
    }
  }
}
```

#### Health Monitoring

##### Submit Health Monitoring Data
```http
POST /api/health-monitoring
```

**Request Body:**
```json
{
  "patient_id": "string",
  "device_type": "wearable|smartphone|medical_device",
  "data_type": "vitals|activity|symptoms|environmental",
  "timestamp": "2023-01-01T12:00:00Z",
  "data": {
    "vital_signs": {
      "heart_rate": 72,
      "heart_rate_variability": 45,
      "blood_pressure": {
        "systolic": 120,
        "diastolic": 80
      },
      "temperature": 98.6,
      "oxygen_saturation": 98,
      "respiratory_rate": 16,
      "blood_glucose": 95
    },
    "activity": {
      "steps": 8432,
      "calories_burned": 320,
      "active_minutes": 45,
      "sleep_duration": 7.5,
      "sleep_quality": 0.85,
      "exercise_type": "walking",
      "exercise_duration": 30
    },
    "symptoms": [
      {
        "symptom": "headache",
        "severity": "mild",
        "duration": "2_hours",
        "triggers": ["stress", "dehydration"],
        "relieving_factors": ["rest", "hydration"]
      }
    ],
    "environmental": {
      "location": "home",
      "air_quality": "good",
      "temperature": 72,
      "humidity": 45,
      "noise_level": "moderate"
    }
  },
  "metadata": {
    "device_id": "string",
    "app_version": "1.2.3",
    "data_quality_score": 0.95
  }
}
```

**Response:**
```json
{
  "data_id": "string",
  "status": "processed",
  "insights": {
    "anomalies_detected": [
      {
        "type": "heart_rate_elevation",
        "severity": "moderate",
        "timestamp": "2023-01-01T11:45:00Z",
        "recommendation": "Monitor for next 2 hours"
      }
    ],
    "trends": {
      "sleep_quality": "improving",
      "activity_level": "consistent",
      "stress_indicators": "elevated"
    },
    "alerts": [
      {
        "alert_type": "preventive",
        "message": "Consider stress management techniques",
        "priority": "medium"
      }
    ]
  },
  "processed_at": "2023-01-01T12:01:00Z"
}
```

##### Get Health Monitoring Dashboard
```http
GET /api/health-monitoring/{patient_id}/dashboard
```

**Query Parameters:**
- `time_range`: Time period (1d, 7d, 30d, 90d)
- `metrics`: Comma-separated list of metrics to include

**Response:**
```json
{
  "patient_id": "string",
  "time_range": "7d",
  "dashboard_data": {
    "vital_signs_trends": {
      "heart_rate": {
        "current": 72,
        "average": 74,
        "min": 58,
        "max": 95,
        "trend": "stable",
        "chart_data": [...]
      },
      "blood_pressure": {
        "current": "120/80",
        "average": "122/82",
        "trend": "stable",
        "hypertension_risk": "low"
      }
    },
    "activity_summary": {
      "total_steps": 52341,
      "average_daily_steps": 7477,
      "active_days": 7,
      "calories_burned": 2240,
      "goal_achievement": 0.95
    },
    "sleep_analysis": {
      "average_duration": 7.3,
      "sleep_efficiency": 0.88,
      "deep_sleep_percentage": 0.22,
      "rem_sleep_percentage": 0.25,
      "sleep_score": 85
    },
    "health_insights": [
      {
        "category": "cardiovascular",
        "insight": "Heart rate variability indicates good autonomic function",
        "confidence": 0.92
      },
      {
        "category": "metabolic",
        "insight": "Blood glucose levels are well controlled",
        "confidence": 0.88
      }
    ],
    "risk_assessment": {
      "overall_risk_score": 0.15,
      "risk_factors": {
        "sedentary_lifestyle": 0.3,
        "poor_sleep": 0.2,
        "stress": 0.4
      },
      "preventive_actions": [
        "Increase daily physical activity",
        "Improve sleep hygiene",
        "Practice stress management"
      ]
    }
  }
}
```

#### Treatment Planning

##### Create Treatment Plan
```http
POST /api/treatment-planning
```

**Request Body:**
```json
{
  "patient_id": "string",
  "condition": "string",
  "severity": "mild|moderate|severe",
  "treatment_goals": [
    "pain_management",
    "functional_improvement",
    "disease_modification",
    "symptom_control"
  ],
  "constraints": {
    "patient_preferences": ["oral_medications", "minimal_side_effects"],
    "contraindications": ["penicillin_allergy", "renal_impairment"],
    "cost_limit": 500,
    "time_commitment": "low"
  },
  "clinical_data": {
    "current_medications": ["string"],
    "lab_results": {...},
    "imaging_results": {...},
    "genomic_data": {...}
  },
  "ai_enhancement": "boolean"
}
```

**Response:**
```json
{
  "plan_id": "string",
  "patient_id": "string",
  "status": "generated",
  "treatment_plan": {
    "primary_treatment": {
      "medications": [
        {
          "name": "ibuprofen",
          "dosage": "400mg",
          "frequency": "every_8_hours",
          "duration": "7_days",
          "rationale": "NSAID for pain and inflammation"
        }
      ],
      "lifestyle_modifications": [
        {
          "modification": "rest",
          "details": "Avoid strenuous activities for 3 days",
          "rationale": "Allow healing process"
        }
      ],
      "physical_therapy": {
        "recommended": "boolean",
        "frequency": "3_times_week",
        "duration": "4_weeks",
        "focus_areas": ["range_of_motion", "strengthening"]
      }
    },
    "alternative_options": [...],
    "monitoring_plan": {
      "follow_up_schedule": "weekly",
      "outcome_measures": ["pain_scale", "functional_assessment"],
      "adjustment_triggers": ["worsening_symptoms", "side_effects"]
    },
    "estimated_outcomes": {
      "success_probability": 0.85,
      "expected_recovery_time": "2_weeks",
      "potential_complications": ["gastric_irritation"],
      "cost_estimate": 125
    }
  },
  "generated_at": "2023-01-01T10:00:00Z",
  "ai_confidence_score": 0.92
}
```

##### Get Treatment Plan
```http
GET /api/treatment-planning/{patient_id}/{plan_id}
```

**Response:** Returns the treatment plan object as created above.

##### Update Treatment Plan
```http
PUT /api/treatment-planning/{plan_id}
```

**Request Body:** Updated treatment plan data

##### Execute Treatment Plan Action
```http
POST /api/treatment-planning/{plan_id}/execute
```

**Request Body:**
```json
{
  "action_type": "start_medication|schedule_appointment|order_test",
  "action_details": {...},
  "reason": "string"
}
```

#### Clinical Decision Support

##### Get Clinical Support
```http
GET /api/clinical-support/{patient_id}
```

**Query Parameters:**
- `query`: Clinical question or scenario
- `context`: Additional context (current_condition, medications, etc.)
- `evidence_level`: Desired evidence level (A, B, C, D)

**Response:**
```json
{
  "query": "string",
  "response": {
    "recommendations": [
      {
        "recommendation": "Start low-dose aspirin therapy",
        "evidence_level": "A",
        "strength": "strong",
        "rationale": "Based on multiple RCTs showing cardiovascular benefit",
        "references": ["NEJM 2020;382(13):1234-1243"],
        "alternative_options": ["clopidogrel"],
        "monitoring": "annual_bleeding_risk_assessment"
      }
    ],
    "differential_diagnosis": [
      {
        "condition": "acute_coronary_syndrome",
        "probability": 0.75,
        "key_features": ["chest_pain", "ekg_changes"],
        "next_steps": ["troponin_levels", "cardiology_consultation"]
      }
    ],
    "risk_assessment": {
      "cardiac_risk_score": 15,
      "risk_category": "intermediate",
      "lifestyle_modifications": ["smoking_cessation", "exercise"],
      "preventive_measures": ["statin_therapy", "beta_blocker"]
    }
  },
  "confidence_score": 0.88,
  "disclaimer": "This is AI-assisted clinical decision support and should not replace clinical judgment"
}
```

#### Patient Engagement Platform

##### Get Patient Dashboard
```http
GET /api/patient-dashboard/{patient_id}
```

**Response:**
```json
{
  "patient_id": "string",
  "dashboard": {
    "health_overview": {
      "current_status": "stable",
      "next_appointment": "2023-01-15T10:00:00Z",
      "active_medications": 3,
      "pending_tasks": 2
    },
    "recent_activity": [
      {
        "activity_type": "medication_adherence",
        "timestamp": "2023-01-01T08:00:00Z",
        "details": "Took morning medications on time",
        "points_earned": 10
      }
    ],
    "health_goals": [
      {
        "goal": "lose_10_pounds",
        "progress": 0.6,
        "target_date": "2023-03-01",
        "milestones": [...]
      }
    ],
    "educational_content": [
      {
        "title": "Understanding Your Blood Pressure",
        "type": "article",
        "read_time": 5,
        "relevance_score": 0.95
      }
    ],
    "appointments": [...],
    "messages": [...]
  },
  "gamification": {
    "current_level": 5,
    "points_to_next_level": 250,
    "total_points": 1250,
    "achievements": [...]
  }
}
```

##### Submit Patient Feedback
```http
POST /api/patient-feedback/{patient_id}
```

**Request Body:**
```json
{
  "feedback_type": "app_usability|clinical_outcome|educational_content",
  "rating": 4,
  "comments": "string",
  "suggestions": "string"
}
```

#### Research Tools

##### Create Clinical Trial
```http
POST /api/research/clinical-trial
```

**Request Body:**
```json
{
  "trial_name": "string",
  "principal_investigator": "string",
  "study_design": {
    "type": "randomized_controlled_trial|cohort_study|case_control",
    "phase": "I|II|III|IV",
    "blinding": "single|double|open_label",
    "sample_size": 500,
    "duration_months": 24
  },
  "eligibility_criteria": {
    "inclusion": ["age_18_plus", "diagnosis_confirmed"],
    "exclusion": ["pregnancy", "renal_failure"]
  },
  "interventions": [
    {
      "name": "experimental_drug",
      "dosage": "10mg_daily",
      "duration": "12_weeks"
    }
  ],
  "outcome_measures": {
    "primary": "reduction_in_symptoms",
    "secondary": ["quality_of_life", "adverse_events"]
  },
  "data_collection": {
    "visit_schedule": "baseline,week4,week8,week12",
    "assessments": ["vital_signs", "lab_tests", "questionnaires"]
  }
}
```

**Response:**
```json
{
  "trial_id": "string",
  "status": "protocol_review",
  "estimated_start_date": "2023-03-01",
  "recruitment_target": 500,
  "timeline": {
    "protocol_finalization": "4_weeks",
    "regulatory_approval": "8_weeks",
    "recruitment": "16_weeks",
    "follow_up": "48_weeks"
  }
}
```

##### Get Research Data
```http
GET /api/research/data
```

**Query Parameters:**
- `study_id`: Specific study identifier
- `data_type`: Type of data requested
- `date_range`: Time period for data
- `aggregation_level`: Individual or aggregated data

**Response:**
```json
{
  "study_id": "string",
  "data_type": "patient_outcomes",
  "aggregation_level": "summary",
  "data": {
    "total_participants": 450,
    "completion_rate": 0.85,
    "primary_outcome": {
      "intervention_group": {
        "mean_improvement": 25.3,
        "standard_deviation": 8.7,
        "confidence_interval": "22.1-28.5"
      },
      "control_group": {
        "mean_improvement": 15.8,
        "standard_deviation": 9.2,
        "confidence_interval": "12.3-19.3"
      }
    },
    "adverse_events": {
      "total_events": 45,
      "serious_events": 3,
      "common_events": ["headache", "nausea", "fatigue"]
    }
  },
  "metadata": {
    "data_quality_score": 0.95,
    "last_updated": "2023-01-01T00:00:00Z",
    "compliance_status": "HIPAA_compliant"
  }
}
```

#### Blockchain Security

##### Verify Medical Record
```http
GET /api/blockchain/verify/{record_id}
```

**Response:**
```json
{
  "record_id": "string",
  "verification_status": "valid",
  "blockchain_info": {
    "block_hash": "string",
    "transaction_hash": "string",
    "block_number": 12345,
    "timestamp": "2023-01-01T10:00:00Z",
    "confirmations": 12
  },
  "record_integrity": {
    "hash_matches": "boolean",
    "chain_verified": "boolean",
    "tamper_evidence": "none"
  },
  "audit_trail": [
    {
      "action": "record_created",
      "actor": "dr_smith",
      "timestamp": "2023-01-01T09:00:00Z",
      "location": "hospital_main"
    }
  ]
}
```

##### Submit Medical Record to Blockchain
```http
POST /api/blockchain/record
```

**Request Body:**
```json
{
  "record_type": "patient_data|genomic_result|treatment_plan",
  "record_data": {...},
  "metadata": {
    "patient_consent": "boolean",
    "data_sensitivity": "high|medium|low",
    "retention_period": "perpetual|10_years|treatment_period"
  }
}
```

## GraphQL API

### Endpoint
```
POST https://api.ai-personalized-medicine.com/graphql
```

### Schema

```graphql
type Query {
  # Patient queries
  patient(id: ID!): Patient
  patients(
    filter: PatientFilter
    pagination: PaginationInput
    sort: PatientSortInput
  ): PatientConnection!

  # Health monitoring queries
  healthData(
    patientId: ID!
    timeRange: TimeRangeInput
    dataTypes: [HealthDataType!]
  ): HealthDataConnection!

  # Genomic analysis queries
  genomicAnalysis(
    patientId: ID!
    analysisId: ID
    filter: GenomicFilterInput
  ): GenomicAnalysis

  # Clinical support queries
  clinicalSupport(
    patientId: ID!
    query: String!
    context: ClinicalContextInput
  ): ClinicalRecommendation

  # Research queries
  clinicalTrials(
    filter: TrialFilterInput
    pagination: PaginationInput
  ): ClinicalTrialConnection!

  researchData(
    studyId: ID!
    dataType: ResearchDataType
    aggregationLevel: AggregationLevel
  ): ResearchData
}

type Mutation {
  # Patient mutations
  createPatient(input: CreatePatientInput!): Patient!
  updatePatient(id: ID!, input: UpdatePatientInput!): Patient!
  deletePatient(id: ID!): DeleteResult!

  # Health monitoring mutations
  submitHealthData(input: HealthDataInput!): HealthDataSubmissionResult!

  # Treatment planning mutations
  createTreatmentPlan(input: TreatmentPlanInput!): TreatmentPlan!
  updateTreatmentPlan(id: ID!, input: UpdateTreatmentPlanInput!): TreatmentPlan!
  executeTreatmentAction(planId: ID!, action: TreatmentActionInput!): TreatmentActionResult!

  # Research mutations
  createClinicalTrial(input: ClinicalTrialInput!): ClinicalTrial!
  updateTrialStatus(trialId: ID!, status: TrialStatus!): ClinicalTrial!
  submitResearchData(studyId: ID!, data: ResearchDataInput!): ResearchDataSubmissionResult!

  # Blockchain mutations
  submitToBlockchain(recordType: RecordType!, data: JSON!, metadata: BlockchainMetadataInput): BlockchainSubmissionResult!
}

type Subscription {
  # Real-time subscriptions
  healthDataUpdates(patientId: ID!): HealthDataUpdate!
  clinicalAlerts(patientId: ID!): ClinicalAlert!
  treatmentPlanUpdates(planId: ID!): TreatmentPlanUpdate!
  researchNotifications(userId: ID!): ResearchNotification!
}

# Core types
type Patient {
  id: ID!
  demographics: Demographics!
  medicalHistory: MedicalHistory!
  currentHealthStatus: HealthStatus
  riskAssessment: RiskAssessment
  treatmentPlans: [TreatmentPlan!]!
  genomicAnalyses: [GenomicAnalysis!]!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Demographics {
  firstName: String!
  lastName: String!
  dateOfBirth: Date!
  gender: Gender!
  ethnicity: String
  contact: ContactInfo!
}

type ContactInfo {
  email: String!
  phone: String
  address: Address
}

type Address {
  street: String!
  city: String!
  state: String!
  zipCode: String!
  country: String!
}

type MedicalHistory {
  allergies: [String!]!
  chronicConditions: [String!]!
  medications: [String!]!
  surgeries: [String!]!
  familyHistory: FamilyHistory!
}

type FamilyHistory {
  diabetes: Boolean!
  cancer: Boolean!
  heartDisease: Boolean!
  other: [String!]!
}

type HealthStatus {
  vitalSigns: VitalSigns
  biometrics: Biometrics
  lastUpdated: DateTime!
}

type VitalSigns {
  heartRate: Int
  bloodPressure: BloodPressure
  temperature: Float
  oxygenSaturation: Int
  respiratoryRate: Int
}

type BloodPressure {
  systolic: Int!
  diastolic: Int!
}

type Biometrics {
  weight: Float
  height: Int
  bmi: Float
  bodyFatPercentage: Float
}

type RiskAssessment {
  overallRisk: RiskLevel!
  riskFactors: [String!]!
  preventiveMeasures: [String!]!
  lastAssessed: DateTime!
}

enum RiskLevel {
  LOW
  MODERATE
  HIGH
  CRITICAL
}

enum Gender {
  MALE
  FEMALE
  OTHER
  PREFER_NOT_TO_SAY
}

# Genomic Analysis Types
type GenomicAnalysis {
  id: ID!
  patientId: ID!
  analysisType: AnalysisType!
  status: AnalysisStatus!
  results: GenomicResults
  recommendations: [String!]!
  clinicalActionItems: [ClinicalAction!]!
  createdAt: DateTime!
  completedAt: DateTime
}

type GenomicResults {
  geneticVariants: GeneticVariants!
  pharmacogenomics: Pharmacogenomics!
  carrierStatus: CarrierStatus!
  polygenicRiskScores: PolygenicRiskScores!
}

type GeneticVariants {
  pathogenic: [PathogenicVariant!]!
  benign: [Variant!]!
  vus: [Variant!]!
}

type PathogenicVariant {
  gene: String!
  variant: String!
  zygosity: Zygosity!
  clinicalSignificance: ClinicalSignificance!
  diseaseAssociation: String!
  riskIncrease: RiskLevel!
}

type Pharmacogenomics {
  drugResponses: [DrugResponse!]!
}

type DrugResponse {
  drug: String!
  gene: String!
  metabolizerStatus: MetabolizerStatus!
  recommendedDoseAdjustment: String!
  monitoringRequired: String!
}

enum MetabolizerStatus {
  ULTRA_RAPID
  RAPID
  NORMAL
  INTERMEDIATE
  POOR
}

enum Zygosity {
  HOMOZYGOUS
  HETEROZYGOUS
}

enum ClinicalSignificance {
  PATHOGENIC
  LIKELY_PATHOGENIC
  UNCERTAIN_SIGNIFICANCE
  LIKELY_BENIGN
  BENIGN
}

enum AnalysisType {
  COMPREHENSIVE
  TARGETED
  CANCER_RISK
  DRUG_RESPONSE
}

enum AnalysisStatus {
  QUEUED
  PROCESSING
  COMPLETED
  FAILED
}

# Treatment Planning Types
type TreatmentPlan {
  id: ID!
  patientId: ID!
  condition: String!
  severity: Severity!
  treatmentGoals: [TreatmentGoal!]!
  primaryTreatment: Treatment!
  alternativeOptions: [Treatment!]!
  monitoringPlan: MonitoringPlan!
  estimatedOutcomes: TreatmentOutcomes!
  status: TreatmentStatus!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Treatment {
  medications: [Medication!]!
  lifestyleModifications: [LifestyleModification!]!
  physicalTherapy: PhysicalTherapy
  procedures: [Procedure!]!
}

type Medication {
  name: String!
  dosage: String!
  frequency: String!
  duration: String!
  rationale: String!
}

type LifestyleModification {
  modification: String!
  details: String!
  rationale: String!
}

type PhysicalTherapy {
  recommended: Boolean!
  frequency: String
  duration: String
  focusAreas: [String!]!
}

type MonitoringPlan {
  followUpSchedule: String!
  outcomeMeasures: [String!]!
  adjustmentTriggers: [String!]!
}

type TreatmentOutcomes {
  successProbability: Float!
  expectedRecoveryTime: String!
  potentialComplications: [String!]!
  costEstimate: Float!
}

enum Severity {
  MILD
  MODERATE
  SEVERE
  CRITICAL
}

enum TreatmentStatus {
  DRAFT
  ACTIVE
  COMPLETED
  DISCONTINUED
}

# Health Monitoring Types
type HealthData {
  id: ID!
  patientId: ID!
  deviceType: DeviceType!
  dataType: HealthDataType!
  timestamp: DateTime!
  data: HealthDataPayload!
  insights: HealthInsights!
  processedAt: DateTime!
}

type HealthDataPayload {
  vitalSigns: VitalSigns
  activity: ActivityData
  symptoms: [Symptom!]!
  environmental: EnvironmentalData
}

type ActivityData {
  steps: Int
  caloriesBurned: Int
  activeMinutes: Int
  sleepDuration: Float
  sleepQuality: Float
  exerciseType: String
  exerciseDuration: Int
}

type Symptom {
  symptom: String!
  severity: Severity!
  duration: String!
  triggers: [String!]!
  relievingFactors: [String!]!
}

type EnvironmentalData {
  location: String
  airQuality: AirQuality
  temperature: Float
  humidity: Float
  noiseLevel: String
}

type HealthInsights {
  anomaliesDetected: [Anomaly!]!
  trends: HealthTrends!
  alerts: [HealthAlert!]!
}

type Anomaly {
  type: String!
  severity: Severity!
  timestamp: DateTime!
  recommendation: String!
}

type HealthTrends {
  sleepQuality: TrendDirection!
  activityLevel: TrendDirection!
  stressIndicators: TrendDirection!
}

type HealthAlert {
  alertType: AlertType!
  message: String!
  priority: Priority!
}

enum DeviceType {
  WEARABLE
  SMARTPHONE
  MEDICAL_DEVICE
}

enum HealthDataType {
  VITALS
  ACTIVITY
  SYMPTOMS
  ENVIRONMENTAL
}

enum AirQuality {
  EXCELLENT
  GOOD
  MODERATE
  POOR
  HAZARDOUS
}

enum TrendDirection {
  IMPROVING
  STABLE
  DECLINING
}

enum AlertType {
  PREVENTIVE
  MONITORING
  URGENT
  EMERGENCY
}

enum Priority {
  LOW
  MEDIUM
  HIGH
  CRITICAL
}

# Clinical Decision Support Types
type ClinicalRecommendation {
  query: String!
  recommendations: [Recommendation!]!
  differentialDiagnosis: [Diagnosis!]!
  riskAssessment: RiskAssessment!
  confidenceScore: Float!
  disclaimer: String!
}

type Recommendation {
  recommendation: String!
  evidenceLevel: EvidenceLevel!
  strength: RecommendationStrength!
  rationale: String!
  references: [String!]!
  alternativeOptions: [String!]!
  monitoring: String!
}

type Diagnosis {
  condition: String!
  probability: Float!
  keyFeatures: [String!]!
  nextSteps: [String!]!
}

enum EvidenceLevel {
  A
  B
  C
  D
}

enum RecommendationStrength {
  STRONG
  MODERATE
  WEAK
}

# Research Types
type ClinicalTrial {
  id: ID!
  trialName: String!
  principalInvestigator: String!
  studyDesign: StudyDesign!
  eligibilityCriteria: EligibilityCriteria!
  interventions: [Intervention!]!
  outcomeMeasures: OutcomeMeasures!
  status: TrialStatus!
  participants: Int!
  targetParticipants: Int!
  startDate: Date
  endDate: Date
  createdAt: DateTime!
}

type StudyDesign {
  type: StudyType!
  phase: Phase!
  blinding: BlindingType!
  sampleSize: Int!
  durationMonths: Int!
}

type EligibilityCriteria {
  inclusion: [String!]!
  exclusion: [String!]!
}

type Intervention {
  name: String!
  dosage: String!
  duration: String!
}

type OutcomeMeasures {
  primary: String!
  secondary: [String!]!
}

enum StudyType {
  RANDOMIZED_CONTROLLED_TRIAL
  COHORT_STUDY
  CASE_CONTROL_STUDY
  CROSS_SECTIONAL
}

enum Phase {
  PHASE_I
  PHASE_II
  PHASE_III
  PHASE_IV
}

enum BlindingType {
  SINGLE_BLIND
  DOUBLE_BLIND
  OPEN_LABEL
}

enum TrialStatus {
  PROTOCOL_REVIEW
  RECRUITING
  ACTIVE
  COMPLETED
  TERMINATED
}

# Blockchain Types
type BlockchainRecord {
  id: ID!
  recordType: RecordType!
  recordData: JSON!
  blockchainInfo: BlockchainInfo!
  auditTrail: [AuditEntry!]!
  createdAt: DateTime!
}

type BlockchainInfo {
  blockHash: String!
  transactionHash: String!
  blockNumber: Int!
  timestamp: DateTime!
  confirmations: Int!
}

type AuditEntry {
  action: String!
  actor: String!
  timestamp: DateTime!
  location: String!
}

enum RecordType {
  PATIENT_DATA
  GENOMIC_RESULT
  TREATMENT_PLAN
  CLINICAL_TRIAL_DATA
  RESEARCH_DATA
}

# Input Types
input CreatePatientInput {
  demographics: DemographicsInput!
  medicalHistory: MedicalHistoryInput!
  consent: ConsentInput!
}

input DemographicsInput {
  firstName: String!
  lastName: String!
  dateOfBirth: Date!
  gender: Gender!
  ethnicity: String
  contact: ContactInfoInput!
}

input MedicalHistoryInput {
  allergies: [String!]!
  chronicConditions: [String!]!
  medications: [String!]!
  surgeries: [String!]!
  familyHistory: FamilyHistoryInput!
}

input ConsentInput {
  dataProcessing: Boolean!
  researchParticipation: Boolean!
  emergencyContact: Boolean!
}

input PatientFilter {
  ageRange: AgeRange
  gender: [Gender!]
  conditions: [String!]
  riskLevel: [RiskLevel!]
}

input AgeRange {
  min: Int
  max: Int
}

input PaginationInput {
  first: Int
  after: String
  last: Int
  before: String
}

input PatientSortInput {
  field: PatientSortField!
  direction: SortDirection!
}

enum PatientSortField {
  NAME
  AGE
  RISK_LEVEL
  CREATED_AT
}

enum SortDirection {
  ASC
  DESC
}

# Connection Types for Pagination
type PatientConnection {
  edges: [PatientEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type PatientEdge {
  node: Patient!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

# Scalars
scalar Date
scalar DateTime
scalar JSON
```

### Example Queries

#### Get Patient with Health Data
```graphql
query GetPatientDashboard($patientId: ID!) {
  patient(id: $patientId) {
    demographics {
      firstName
      lastName
      dateOfBirth
    }
    currentHealthStatus {
      vitalSigns {
        heartRate
        bloodPressure {
          systolic
          diastolic
        }
      }
    }
    riskAssessment {
      overallRisk
      riskFactors
    }
  }

  healthData(
    patientId: $patientId
    timeRange: { days: 7 }
    dataTypes: [VITALS, ACTIVITY]
  ) {
    edges {
      node {
        timestamp
        data {
          vitalSigns {
            heartRate
            oxygenSaturation
          }
          activity {
            steps
            sleepDuration
          }
        }
        insights {
          anomaliesDetected {
            type
            severity
            recommendation
          }
          trends {
            sleepQuality
            activityLevel
          }
        }
      }
    }
  }
}
```

#### Create Treatment Plan
```graphql
mutation CreateTreatmentPlan($input: TreatmentPlanInput!) {
  createTreatmentPlan(input: $input) {
    id
    status
    primaryTreatment {
      medications {
        name
        dosage
        frequency
        rationale
      }
      lifestyleModifications {
        modification
        details
        rationale
      }
    }
    estimatedOutcomes {
      successProbability
      expectedRecoveryTime
      potentialComplications
      costEstimate
    }
  }
}
```

## WebSocket API

### Endpoint
```
wss://api.ai-personalized-medicine.com/ws
```

### Connection Establishment
```javascript
const ws = new WebSocket('wss://api.ai-personalized-medicine.com/ws');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'authenticate',
    token: 'jwt_token_here'
  }));
};
```

### Message Types

#### Authentication
```json
{
  "type": "authenticate",
  "token": "jwt_token"
}
```

#### Subscribe to Real-time Data
```json
{
  "type": "subscribe",
  "channels": [
    {
      "channel": "health_monitoring",
      "patient_id": "patient_123"
    },
    {
      "channel": "clinical_alerts",
      "patient_id": "patient_123"
    },
    {
      "channel": "treatment_updates",
      "plan_id": "plan_456"
    }
  ]
}
```

#### Unsubscribe from Channels
```json
{
  "type": "unsubscribe",
  "channels": ["health_monitoring:patient_123"]
}
```

#### Heartbeat
```json
{
  "type": "heartbeat",
  "timestamp": 1640995200000
}
```

### Incoming Messages

#### Health Monitoring Update
```json
{
  "type": "health_data_update",
  "channel": "health_monitoring:patient_123",
  "data": {
    "patient_id": "patient_123",
    "timestamp": "2023-01-01T12:00:00Z",
    "vital_signs": {
      "heart_rate": 75,
      "blood_pressure": "125/82",
      "oxygen_saturation": 97
    },
    "anomalies": [
      {
        "type": "blood_pressure_elevation",
        "severity": "moderate",
        "recommendation": "Monitor blood pressure for next hour"
      }
    ]
  }
}
```

#### Clinical Alert
```json
{
  "type": "clinical_alert",
  "channel": "clinical_alerts:patient_123",
  "data": {
    "alert_id": "alert_789",
    "patient_id": "patient_123",
    "alert_type": "critical",
    "message": "Acute hypertension detected",
    "timestamp": "2023-01-01T12:05:00Z",
    "recommended_actions": [
      "Contact patient immediately",
      "Schedule urgent appointment",
      "Adjust medication if necessary"
    ],
    "clinical_context": {
      "current_medications": ["lisinopril_10mg"],
      "recent_vitals": {...},
      "risk_factors": ["hypertension", "diabetes"]
    }
  }
}
```

#### Treatment Plan Update
```json
{
  "type": "treatment_update",
  "channel": "treatment_updates:plan_456",
  "data": {
    "plan_id": "plan_456",
    "update_type": "medication_adjustment",
    "changes": {
      "medication": "metformin",
      "old_dosage": "500mg_bid",
      "new_dosage": "1000mg_bid",
      "reason": "inadequate_glycemic_control",
      "effective_date": "2023-01-02"
    },
    "rationale": "HbA1c remains above target despite current therapy",
    "monitoring_instructions": "Check blood glucose daily for 1 week"
  }
}
```

#### Genomic Analysis Completion
```json
{
  "type": "genomic_analysis_complete",
  "channel": "genomic_updates:patient_123",
  "data": {
    "analysis_id": "analysis_101",
    "patient_id": "patient_123",
    "status": "completed",
    "results_summary": {
      "pathogenic_variants": 2,
      "pharmacogenomic_findings": 3,
      "risk_score_changes": {
        "breast_cancer_risk": "increased",
        "cardiovascular_risk": "decreased"
      }
    },
    "urgent_findings": [
      {
        "finding": "BRCA1_pathogenic_variant",
        "clinical_significance": "high",
        "recommended_action": "urgent_genetics_consultation"
      }
    ],
    "next_steps": [
      "Schedule genetic counseling appointment",
      "Discuss prophylactic options",
      "Update family history"
    ]
  }
}
```

### Error Handling
```json
{
  "type": "error",
  "code": "AUTHENTICATION_FAILED",
  "message": "Invalid or expired JWT token",
  "timestamp": "2023-01-01T12:00:00Z"
}
```

### Connection Management

#### Ping/Pong
```javascript
// Send ping every 30 seconds
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'ping' }));
  }
}, 30000);

// Handle pong response
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'pong') {
    console.log('Connection alive');
  }
};
```

#### Reconnection Logic
```javascript
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectInterval = 1000; // Start with 1 second

function connect() {
  const ws = new WebSocket('wss://api.ai-personalized-medicine.com/ws');

  ws.onopen = () => {
    console.log('Connected to WebSocket');
    reconnectAttempts = 0;
    // Send authentication and subscriptions
  };

  ws.onclose = () => {
    if (reconnectAttempts < maxReconnectAttempts) {
      reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnecting... Attempt ${reconnectAttempts}`);
        connect();
      }, reconnectInterval * reconnectAttempts); // Exponential backoff
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  return ws;
}
```

## Authentication & Authorization

### JWT Token Structure
```json
{
  "iss": "ai-personalized-medicine-platform",
  "sub": "user_123",
  "aud": "api.ai-personalized-medicine.com",
  "exp": 1640995200,
  "iat": 1640991600,
  "roles": ["clinician", "researcher"],
  "permissions": ["read_patient_data", "write_treatment_plans"],
  "patient_access": ["patient_456", "patient_789"]
}
```

### Role-Based Access Control (RBAC)

#### Roles
- **patient**: Access to own health data and dashboard
- **clinician**: Full access to assigned patients' data
- **researcher**: Access to anonymized research data
- **administrator**: System-wide access and configuration
- **api_consumer**: Limited API access for integrations

#### Permissions Matrix

| Permission | Patient | Clinician | Researcher | Administrator | API Consumer |
|------------|---------|-----------|------------|---------------|--------------|
| read_own_data | ✓ | ✗ | ✗ | ✗ | ✗ |
| read_patient_data | ✗ | ✓ | ✗ | ✓ | ✓ |
| write_patient_data | ✗ | ✓ | ✗ | ✓ | ✗ |
| read_research_data | ✗ | ✗ | ✓ | ✓ | ✗ |
| write_research_data | ✗ | ✗ | ✓ | ✓ | ✗ |
| manage_users | ✗ | ✗ | ✗ | ✓ | ✗ |
| system_config | ✗ | ✗ | ✗ | ✓ | ✗ |
| api_access | ✓ | ✓ | ✓ | ✓ | ✓ |

### OAuth 2.0 Integration

#### Authorization Code Flow
```http
GET /oauth/authorize?response_type=code&client_id=client_123&redirect_uri=https://app.example.com/callback&scope=read_patient_data write_treatment_plans&state=random_state
```

#### Token Exchange
```http
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code=auth_code_here&redirect_uri=https://app.example.com/callback&client_id=client_123&client_secret=client_secret_here
```

#### Token Response
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here",
  "scope": "read_patient_data write_treatment_plans"
}
```

### Multi-Factor Authentication (MFA)

#### TOTP Setup
```http
POST /auth/mfa/setup
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "secret": "JBSWY3DPEHPK3PXP",
  "qr_code_url": "otpauth://totp/AI%20Med%20Platform:user@example.com?secret=JBSWY3DPEHPK3PXP&issuer=AI%20Med%20Platform",
  "backup_codes": ["12345678", "87654321", "11223344", "44332211"]
}
```

#### MFA Verification
```http
POST /auth/mfa/verify
Content-Type: application/json

{
  "code": "123456",
  "remember_device": true
}
```

## Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 60
```

### Rate Limit Policies

#### By Endpoint Category
- **Patient Data**: 1000 requests/hour per patient
- **Clinical Support**: 100 requests/hour per user
- **Genomic Analysis**: 10 requests/hour per user
- **Drug Discovery**: 50 requests/hour per user
- **Research Data**: 500 requests/hour per user

#### By User Role
- **Patients**: 100 requests/hour
- **Clinicians**: 1000 requests/hour
- **Researchers**: 2000 requests/hour
- **Administrators**: 5000 requests/hour

### Rate Limit Exceeded Response
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please try again later.",
  "retry_after": 3600,
  "limit": 1000,
  "remaining": 0,
  "reset_time": "2023-01-01T13:00:00Z"
}
```

## API Versioning

### Version Header
```http
Accept-Version: v1
API-Version: v1
```

### Version Endpoints
```http
GET /api/v1/patients
GET /api/v2/patients
```

### Version Compatibility
```json
{
  "version": "v1.2.3",
  "deprecated": false,
  "sunset_date": null,
  "supported_versions": ["v1.0.0", "v1.1.0", "v1.2.0"],
  "latest_version": "v1.2.3",
  "changelog": "https://api.ai-personalized-medicine.com/changelog"
}
```

## Error Handling

### Standard Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data provided",
    "details": {
      "field": "email",
      "issue": "invalid_format",
      "provided_value": "invalid-email"
    },
    "timestamp": "2023-01-01T12:00:00Z",
    "request_id": "req_123456789",
    "path": "/api/patients",
    "method": "POST"
  }
}
```

### Error Codes

#### Authentication Errors (4xx)
- `INVALID_TOKEN`: JWT token is malformed or invalid
- `EXPIRED_TOKEN`: JWT token has expired
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `MFA_REQUIRED`: Multi-factor authentication is required

#### Validation Errors (4xx)
- `VALIDATION_ERROR`: Input data failed validation
- `MISSING_REQUIRED_FIELD`: Required field is missing
- `INVALID_FORMAT`: Field value has invalid format
- `OUT_OF_RANGE`: Numeric value is outside acceptable range

#### Business Logic Errors (4xx)
- `PATIENT_NOT_FOUND`: Specified patient does not exist
- `TREATMENT_PLAN_INACTIVE`: Treatment plan is no longer active
- `GENOMIC_ANALYSIS_IN_PROGRESS`: Another analysis is already in progress
- `CLINICAL_TRIAL_FULL`: Clinical trial has reached maximum enrollment

#### System Errors (5xx)
- `INTERNAL_SERVER_ERROR`: Unexpected server error
- `DATABASE_ERROR`: Database operation failed
- `EXTERNAL_SERVICE_ERROR`: Third-party service is unavailable
- `RATE_LIMIT_EXCEEDED`: Too many requests (should be 429, but included here for completeness)

### Error Response Examples

#### Validation Error
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Patient date of birth cannot be in the future",
    "details": {
      "field": "date_of_birth",
      "issue": "future_date",
      "provided_value": "2025-01-01",
      "constraint": "must_be_past_date"
    },
    "suggestions": [
      "Please provide the patient's actual date of birth",
      "Format should be YYYY-MM-DD"
    ]
  }
}
```

#### Authentication Error
```json
{
  "error": {
    "code": "INSUFFICIENT_PERMISSIONS",
    "message": "You do not have permission to access this patient's genomic data",
    "details": {
      "required_permission": "read_genomic_data",
      "user_permissions": ["read_basic_patient_data"],
      "patient_id": "patient_123"
    },
    "resolution_steps": [
      "Request access from the patient's primary care physician",
      "Contact system administrator for permission escalation",
      "Use your assigned patient access list"
    ]
  }
}
```

#### System Error
```json
{
  "error": {
    "code": "EXTERNAL_SERVICE_ERROR",
    "message": "Genomic analysis service is temporarily unavailable",
    "details": {
      "service": "genomic_analysis_pipeline",
      "status": "degraded",
      "estimated_resolution": "2023-01-01T14:00:00Z"
    },
    "retry": {
      "recommended": true,
      "backoff_strategy": "exponential",
      "max_attempts": 3
    }
  }
}
```

## Data Models

### Core Data Models

#### Patient Model
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class Address(BaseModel):
    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1)
    zip_code: str = Field(..., pattern=r'^\d{5}(-\d{4})?$')
    country: str = Field(..., min_length=1)

class ContactInfo(BaseModel):
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    phone: Optional[str] = Field(None, regex=r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$')
    address: Optional[Address] = None

class Demographics(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    date_of_birth: date
    gender: Gender
    ethnicity: Optional[str] = None
    contact: ContactInfo

    class Config:
        json_encoders = {
            date: lambda v: v.isoformat()
        }

class FamilyHistory(BaseModel):
    diabetes: bool = False
    cancer: bool = False
    heart_disease: bool = False
    other: List[str] = Field(default_factory=list)

class MedicalHistory(BaseModel):
    allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    surgeries: List[str] = Field(default_factory=list)
    family_history: FamilyHistory

class ConsentInfo(BaseModel):
    data_processing: bool = Field(..., description="Consent for data processing and analysis")
    research_participation: bool = Field(..., description="Consent for research participation")
    emergency_contact: bool = Field(..., description="Consent for emergency contact")

class VitalSigns(BaseModel):
    heart_rate: Optional[int] = Field(None, ge=30, le=250)
    blood_pressure_systolic: Optional[int] = Field(None, ge=70, le=250)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=40, le=150)
    temperature: Optional[float] = Field(None, ge=95.0, le=110.0)
    oxygen_saturation: Optional[int] = Field(None, ge=70, le=100)
    respiratory_rate: Optional[int] = Field(None, ge=8, le=60)

class Biometrics(BaseModel):
    weight: Optional[float] = Field(None, gt=0, le=500)
    height: Optional[int] = Field(None, gt=0, le=250)  # cm
    bmi: Optional[float] = Field(None, gt=0, le=100)
    body_fat_percentage: Optional[float] = Field(None, ge=0, le=50)

class HealthStatus(BaseModel):
    vital_signs: Optional[VitalSigns] = None
    biometrics: Optional[Biometrics] = None
    last_updated: datetime

class RiskAssessment(BaseModel):
    overall_risk: RiskLevel
    risk_factors: List[str] = Field(default_factory=list)
    preventive_measures: List[str] = Field(default_factory=list)
    last_assessed: datetime

class PatientProfile(BaseModel):
    patient_id: str = Field(..., pattern=r'^P\d{6}$')
    demographics: Demographics
    medical_history: MedicalHistory
    consent: ConsentInfo
    current_health_status: Optional[HealthStatus] = None
    risk_assessment: Optional[RiskAssessment] = None
    created_at: datetime
    updated_at: datetime
    profile_completion_percentage: float = Field(0.0, ge=0.0, le=100.0)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }
```

#### Genomic Analysis Models
```python
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class AnalysisType(str, Enum):
    COMPREHENSIVE = "comprehensive"
    TARGETED = "targeted"
    CANCER_RISK = "cancer_risk"
    DRUG_RESPONSE = "drug_response"

class AnalysisStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Zygosity(str, Enum):
    HOMOZYGOUS = "homozygous"
    HETEROZYGOUS = "heterozygous"

class ClinicalSignificance(str, Enum):
    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    UNCERTAIN_SIGNIFICANCE = "uncertain_significance"
    LIKELY_BENIGN = "likely_benign"
    BENIGN = "benign"

class MetabolizerStatus(str, Enum):
    ULTRA_RAPID = "ultra_rapid"
    RAPID = "rapid"
    NORMAL = "normal"
    INTERMEDIATE = "intermediate"
    POOR = "poor"

class PathogenicVariant(BaseModel):
    gene: str = Field(..., min_length=1)
    variant: str = Field(..., min_length=1)
    zygosity: Zygosity
    clinical_significance: ClinicalSignificance
    disease_association: str = Field(..., min_length=1)
    risk_increase: str = Field(..., min_length=1)

class Variant(BaseModel):
    gene: str = Field(..., min_length=1)
    variant: str = Field(..., min_length=1)
    zygosity: Zygosity
    frequency: Optional[float] = Field(None, ge=0.0, le=1.0)

class GeneticVariants(BaseModel):
    pathogenic: List[PathogenicVariant] = Field(default_factory=list)
    benign: List[Variant] = Field(default_factory=list)
    vus: List[Variant] = Field(default_factory=list)

class DrugResponse(BaseModel):
    drug: str = Field(..., min_length=1)
    gene: str = Field(..., min_length=1)
    metabolizer_status: MetabolizerStatus
    recommended_dose_adjustment: str = Field(..., min_length=1)
    monitoring_required: str = Field(..., min_length=1)

class Pharmacogenomics(BaseModel):
    drug_responses: List[DrugResponse] = Field(default_factory=list)

class CarrierCondition(BaseModel):
    condition: str = Field(..., min_length=1)
    gene: str = Field(..., min_length=1)
    carrier_status: bool
    partner_testing_recommended: bool = False

class CarrierStatus(BaseModel):
    recessive_conditions: List[CarrierCondition] = Field(default_factory=list)

class PolygenicRiskScore(BaseModel):
    condition: str = Field(..., min_length=1)
    risk_percentile: float = Field(..., ge=0.0, le=100.0)
    lifetime_risk: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: str = Field(..., min_length=1)

class PolygenicRiskScores(BaseModel):
    disease_risks: List[PolygenicRiskScore] = Field(default_factory=list)

class GenomicResults(BaseModel):
    genetic_variants: GeneticVariants
    pharmacogenomics: Pharmacogenomics
    carrier_status: CarrierStatus
    polygenic_risk_scores: PolygenicRiskScores

class ClinicalAction(BaseModel):
    action: str = Field(..., min_length=1)
    specialty: Optional[str] = None
    priority: str = Field(..., min_length=1)
    due_date: Optional[datetime] = None

class GenomicAnalysis(BaseModel):
    analysis_id: str = Field(..., pattern=r'^GA\d{8}$')
    patient_id: str = Field(..., pattern=r'^P\d{6}$')
    analysis_type: AnalysisType
    status: AnalysisStatus
    results: Optional[GenomicResults] = None
    recommendations: List[str] = Field(default_factory=list)
    clinical_action_items: List[ClinicalAction] = Field(default_factory=list)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    issues: List[str] = Field(default_factory=list)
```

## Integration APIs

### EHR Integration

#### FHIR Integration
```http
GET /api/integration/fhir/Patient/{patient_id}
Accept: application/fhir+json
```

**Response:**
```json
{
  "resourceType": "Patient",
  "id": "patient_123",
  "identifier": [
    {
      "system": "https://api.ai-personalized-medicine.com",
      "value": "P000123"
    }
  ],
  "name": [
    {
      "family": "Doe",
      "given": ["John"]
    }
  ],
  "gender": "male",
  "birthDate": "1980-01-01",
  "address": [
    {
      "line": ["123 Main St"],
      "city": "Anytown",
      "state": "CA",
      "postalCode": "12345"
    }
  ]
}
```

#### HL7 Integration
```http
POST /api/integration/hl7
Content-Type: application/hl7-v2

MSH|^~\&|SENDING_APP|SENDING_FACILITY|RECEIVING_APP|RECEIVING_FACILITY|20230101120000||ADT^A01|MSG00001|P|2.5
EVN|A01|20230101120000
PID|1||P000123^^^AI_MED^MR||DOE^JOHN^^^^^L|MALE|19800101|W|||123 MAIN ST^ANYTOWN^CA^12345
```

### API Gateway Integration

#### Webhook Registration
```http
POST /api/webhooks/register
```

**Request Body:**
```json
{
  "url": "https://external-system.com/webhook",
  "events": [
    "patient.created",
    "genomic_analysis.completed",
    "treatment_plan.updated"
  ],
  "secret": "webhook_secret_key",
  "headers": {
    "Authorization": "Bearer external_token",
    "X-API-Key": "api_key_here"
  }
}
```

#### Webhook Payload
```json
{
  "event": "genomic_analysis.completed",
  "timestamp": "2023-01-01T15:30:00Z",
  "data": {
    "analysis_id": "GA20230001",
    "patient_id": "P000123",
    "results": {...}
  },
  "signature": "sha256_signature_here"
}
```

### Third-party Integrations

#### Laboratory Information System (LIS)
```http
POST /api/integration/lis/order
```

**Request Body:**
```json
{
  "patient_id": "P000123",
  "tests": [
    {
      "test_code": "CBC",
      "test_name": "Complete Blood Count",
      "priority": "routine",
      "clinical_indication": "Annual physical"
    }
  ],
  "ordering_provider": "Dr. Smith",
  "callback_url": "https://api.ai-personalized-medicine.com/callback/lis"
}
```

#### Pharmacy Integration
```http
GET /api/integration/pharmacy/medications/{patient_id}
```

**Response:**
```json
{
  "patient_id": "P000123",
  "medications": [
    {
      "medication_id": "MED001",
      "name": "Lisinopril",
      "dosage": "10mg",
      "frequency": "daily",
      "prescribed_date": "2023-01-01",
      "days_supply": 30,
      "refills_remaining": 5,
      "pharmacy": "CVS Pharmacy #123"
    }
  ]
}
```

## SDKs & Libraries

### Python SDK

#### Installation
```bash
pip install ai-personalized-medicine-sdk
```

#### Basic Usage
```python
from ai_med_sdk import AIHealthcarePlatform

# Initialize client
client = AIHealthcarePlatform(
    api_key="your_api_key",
    base_url="https://api.ai-personalized-medicine.com"
)

# Create patient
patient = client.patients.create({
    "patient_id": "P000123",
    "demographics": {
        "first_name": "John",
        "last_name": "Doe",
        "date_of_birth": "1980-01-01",
        "gender": "male"
    },
    "medical_history": {
        "chronic_conditions": ["hypertension"],
        "medications": ["lisinopril_10mg"]
    }
})

# Get genomic analysis
analysis = client.genomic_analysis.get_results("P000123")
print(f"Pathogenic variants found: {len(analysis.genetic_variants.pathogenic)}")

# Submit health monitoring data
health_data = client.health_monitoring.submit({
    "patient_id": "P000123",
    "vital_signs": {
        "heart_rate": 72,
        "blood_pressure": "120/80",
        "oxygen_saturation": 98
    }
})
```

### JavaScript SDK

#### Installation
```bash
npm install ai-personalized-medicine-sdk
```

#### Basic Usage
```javascript
import { AIHealthcarePlatform } from 'ai-personalized-medicine-sdk';

// Initialize client
const client = new AIHealthcarePlatform({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.ai-personalized-medicine.com'
});

// Real-time health monitoring
const ws = client.websocket.connect();

ws.subscribe('health_monitoring', 'P000123', (data) => {
  console.log('Health update:', data);
  if (data.anomalies.length > 0) {
    alert('Health anomaly detected!');
  }
});

// Get patient dashboard
const dashboard = await client.patients.getDashboard('P000123');
console.log('Patient risk level:', dashboard.risk_assessment.overall_risk);
```

### Mobile SDKs

#### iOS SDK (Swift)
```swift
import AIHealthcarePlatform

let client = AIHealthcarePlatformClient(apiKey: "your_api_key")

// Background health monitoring
client.healthMonitoring.startMonitoring(patientId: "P000123") { result in
    switch result {
    case .success(let data):
        // Process health data
        self.processHealthData(data)
    case .failure(let error):
        print("Health monitoring error: \(error)")
    }
}

// Offline data sync
client.syncManager.syncPendingData { result in
    // Handle sync completion
}
```

#### Android SDK (Kotlin)
```kotlin
import com.aihealthcare.platform.AIHealthcarePlatformClient

val client = AIHealthcarePlatformClient(apiKey = "your_api_key")

// Continuous health monitoring
client.healthMonitoring.startMonitoring("P000123")
    .observeOn(AndroidSchedulers.mainThread())
    .subscribe({ data ->
        // Process health data
        processHealthData(data)
    }, { error ->
        Log.e("HealthMonitor", "Error: $error")
    })

// Emergency alert
client.emergency.sendAlert(
    patientId = "P000123",
    alertType = EmergencyAlertType.CARDIAC_ARREST,
    location = currentLocation
)
```

---

*This comprehensive API reference covers all major endpoints, data models, authentication methods, and integration patterns for the AI Personalized Medicine Platform. For specific implementation details and the latest updates, please refer to the official documentation at https://docs.ai-personalized-medicine.com*
