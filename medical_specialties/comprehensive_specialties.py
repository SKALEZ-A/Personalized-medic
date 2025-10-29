"""
Comprehensive Medical Specialties Module for AI Personalized Medicine Platform
Specialized modules for different medical specialties with AI-powered diagnostics and treatment
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json


@dataclass
class SpecialtyConfig:
    """Configuration for medical specialty"""
    name: str
    description: str
    key_symptoms: List[str]
    common_conditions: List[str]
    diagnostic_tests: List[str]
    treatment_modalities: List[str]
    ai_models: List[str]
    risk_factors: List[str]
    preventive_measures: List[str]


@dataclass
class PatientAssessment:
    """Patient assessment data"""
    patient_id: str
    specialty: str
    symptoms: List[str]
    medical_history: List[Dict[str, Any]]
    physical_exam: Dict[str, Any]
    diagnostic_results: Dict[str, Any]
    risk_factors: List[str]
    assessment_date: datetime
    provider_id: str


@dataclass
class SpecialtyDiagnosis:
    """Specialty-specific diagnosis"""
    condition: str
    confidence_score: float
    supporting_evidence: List[str]
    differential_diagnoses: List[str]
    recommended_tests: List[str]
    urgency_level: str  # 'routine', 'urgent', 'emergency'
    specialty: str


@dataclass
class TreatmentPlan:
    """Specialty-specific treatment plan"""
    patient_id: str
    specialty: str
    primary_diagnosis: str
    medications: List[Dict[str, Any]]
    procedures: List[Dict[str, Any]]
    lifestyle_modifications: List[str]
    follow_up_schedule: List[Dict[str, Any]]
    monitoring_parameters: List[str]
    expected_outcomes: List[str]
    plan_date: datetime


class BaseMedicalSpecialty(ABC):
    """Base class for medical specialties"""

    def __init__(self, config: SpecialtyConfig):
        self.config = config
        self.diagnostic_rules = {}
        self.treatment_protocols = {}
        self.ai_models = {}

    @abstractmethod
    def assess_patient(self, patient_data: PatientAssessment) -> SpecialtyDiagnosis:
        """Assess patient and provide diagnosis"""
        pass

    @abstractmethod
    def create_treatment_plan(self, diagnosis: SpecialtyDiagnosis, patient_data: PatientAssessment) -> TreatmentPlan:
        """Create treatment plan based on diagnosis"""
        pass

    @abstractmethod
    def predict_outcomes(self, treatment_plan: TreatmentPlan, patient_data: PatientAssessment) -> Dict[str, Any]:
        """Predict treatment outcomes"""
        pass

    def validate_symptoms(self, symptoms: List[str]) -> List[str]:
        """Validate symptoms against specialty"""
        valid_symptoms = []
        for symptom in symptoms:
            if symptom.lower() in [s.lower() for s in self.config.key_symptoms]:
                valid_symptoms.append(symptom)
        return valid_symptoms

    def calculate_risk_score(self, patient_data: PatientAssessment) -> float:
        """Calculate risk score for patient"""
        risk_score = 0.0
        risk_factors = set(patient_data.risk_factors)

        for factor in self.config.risk_factors:
            if factor.lower() in [rf.lower() for rf in risk_factors]:
                risk_score += 0.2

        # Age-based risk (simplified)
        if patient_data.physical_exam.get('age', 0) > 65:
            risk_score += 0.3

        return min(risk_score, 1.0)


class CardiologySpecialty(BaseMedicalSpecialty):
    """Cardiology specialty with cardiovascular disease focus"""

    def __init__(self):
        config = SpecialtyConfig(
            name="Cardiology",
            description="Cardiovascular disease diagnosis and treatment",
            key_symptoms=[
                "chest pain", "shortness of breath", "palpitations", "fatigue",
                "dizziness", "swelling", "syncope", "irregular heartbeat"
            ],
            common_conditions=[
                "coronary artery disease", "heart failure", "arrhythmia",
                "valvular heart disease", "hypertension", "myocardial infarction",
                "peripheral artery disease", "cardiomyopathy"
            ],
            diagnostic_tests=[
                "electrocardiogram", "echocardiogram", "stress test",
                "cardiac catheterization", "coronary angiography",
                "cardiac MRI", "holter monitor", "blood pressure monitoring"
            ],
            treatment_modalities=[
                "medications", "angioplasty", "coronary bypass surgery",
                "pacemaker implantation", "cardiac ablation", "valve replacement",
                "lifestyle modifications", "cardiac rehabilitation"
            ],
            ai_models=[
                "cardiac_risk_predictor", "ecg_analyzer", "echo_quantifier",
                "coronary_calcium_scorer", "heart_failure_predictor"
            ],
            risk_factors=[
                "hypertension", "diabetes", "smoking", "high cholesterol",
                "family history", "obesity", "sedentary lifestyle", "stress"
            ],
            preventive_measures=[
                "regular exercise", "heart-healthy diet", "smoking cessation",
                "blood pressure control", "cholesterol management", "stress reduction",
                "regular check-ups", "medication adherence"
            ]
        )
        super().__init__(config)
        self._initialize_cardiac_rules()

    def _initialize_cardiac_rules(self):
        """Initialize cardiac diagnostic rules"""
        self.diagnostic_rules = {
            "acute_coronary_syndrome": {
                "symptoms": ["chest_pain", "shortness_of_breath", "sweating"],
                "risk_factors": ["hypertension", "smoking", "diabetes"],
                "urgency": "emergency"
            },
            "heart_failure": {
                "symptoms": ["dyspnea", "fatigue", "edema"],
                "findings": ["elevated_jvp", "pulmonary_crackles"],
                "urgency": "urgent"
            },
            "arrhythmia": {
                "symptoms": ["palpitations", "dizziness", "syncope"],
                "findings": ["irregular_pulse", "murmur"],
                "urgency": "urgent"
            }
        }

    def assess_patient(self, patient_data: PatientAssessment) -> SpecialtyDiagnosis:
        """Assess cardiac patient"""
        symptoms = [s.lower().replace(' ', '_') for s in patient_data.symptoms]
        risk_factors = [rf.lower().replace(' ', '_') for rf in patient_data.risk_factors]

        # Check for acute coronary syndrome
        acs_symptoms = self.diagnostic_rules["acute_coronary_syndrome"]["symptoms"]
        acs_risks = self.diagnostic_rules["acute_coronary_syndrome"]["risk_factors"]

        if any(s in symptoms for s in acs_symptoms) and any(r in risk_factors for r in acs_risks):
            return SpecialtyDiagnosis(
                condition="Acute Coronary Syndrome",
                confidence_score=0.85,
                supporting_evidence=["Typical chest pain", "Cardiac risk factors present"],
                differential_diagnoses=["Pulmonary embolism", "Aortic dissection", "Pericarditis"],
                recommended_tests=["ECG", "Troponin", "Chest X-ray"],
                urgency_level="emergency",
                specialty="Cardiology"
            )

        # Check for heart failure
        hf_symptoms = self.diagnostic_rules["heart_failure"]["symptoms"]
        if any(s in symptoms for s in hf_symptoms):
            return SpecialtyDiagnosis(
                condition="Heart Failure",
                confidence_score=0.75,
                supporting_evidence=["Dyspnea", "Fatigue", "Edema"],
                differential_diagnoses=["COPD", "Anemia", "Thyroid disease"],
                recommended_tests=["BNP", "Echocardiogram", "Chest X-ray"],
                urgency_level="urgent",
                specialty="Cardiology"
            )

        # Default assessment
        return SpecialtyDiagnosis(
            condition="Cardiovascular Evaluation Needed",
            confidence_score=0.5,
            supporting_evidence=["Cardiac symptoms present"],
            differential_diagnoses=self.config.common_conditions[:5],
            recommended_tests=["ECG", "Echocardiogram", "Cardiac enzymes"],
            urgency_level="routine",
            specialty="Cardiology"
        )

    def create_treatment_plan(self, diagnosis: SpecialtyDiagnosis, patient_data: PatientAssessment) -> TreatmentPlan:
        """Create cardiac treatment plan"""
        medications = []
        procedures = []
        lifestyle_mods = []
        monitoring = []

        if "coronary" in diagnosis.condition.lower():
            medications.extend([
                {"name": "Aspirin", "dosage": "81mg daily", "indication": "Antiplatelet therapy"},
                {"name": "Atorvastatin", "dosage": "40mg daily", "indication": "Statin therapy"},
                {"name": "Metoprolol", "dosage": "25mg twice daily", "indication": "Beta blocker"}
            ])
            procedures.append({
                "name": "Coronary Angiography",
                "timing": "Urgent",
                "indication": "Coronary artery evaluation"
            })
            monitoring.extend(["ECG", "Cardiac enzymes", "Blood pressure"])

        elif "failure" in diagnosis.condition.lower():
            medications.extend([
                {"name": "Lisinopril", "dosage": "5mg daily", "indication": "ACE inhibitor"},
                {"name": "Furosemide", "dosage": "20mg daily", "indication": "Diuretic"},
                {"name": "Carvedilol", "dosage": "3.125mg twice daily", "indication": "Beta blocker"}
            ])
            lifestyle_mods.extend([
                "Sodium restriction (<2g/day)",
                "Fluid restriction (1.5-2L/day)",
                "Daily weight monitoring",
                "Regular exercise as tolerated"
            ])
            monitoring.extend(["Daily weights", "Blood pressure", "BNP levels"])

        return TreatmentPlan(
            patient_id=patient_data.patient_id,
            specialty="Cardiology",
            primary_diagnosis=diagnosis.condition,
            medications=medications,
            procedures=procedures,
            lifestyle_modifications=lifestyle_mods,
            follow_up_schedule=[
                {"timing": "1 week", "purpose": "Treatment response assessment"},
                {"timing": "1 month", "purpose": "Medication titration"},
                {"timing": "3 months", "purpose": "Comprehensive evaluation"}
            ],
            monitoring_parameters=monitoring,
            expected_outcomes=[
                "Symptom improvement",
                "Improved cardiac function",
                "Reduced hospitalizations"
            ],
            plan_date=datetime.now()
        )

    def predict_outcomes(self, treatment_plan: TreatmentPlan, patient_data: PatientAssessment) -> Dict[str, Any]:
        """Predict cardiac treatment outcomes"""
        base_success_rate = 0.75

        # Adjust based on risk factors
        risk_adjustment = len(patient_data.risk_factors) * -0.05
        adjusted_success_rate = max(0.4, min(0.95, base_success_rate + risk_adjustment))

        return {
            "treatment_success_probability": adjusted_success_rate,
            "expected_improvement_timeline": "4-8 weeks",
            "potential_complications": ["Medication side effects", "Procedure complications"],
            "long_term_prognosis": "Good with adherence to treatment plan",
            "monitoring_schedule": ["Weekly for first month", "Monthly thereafter"],
            "follow_up_recommendations": [
                "Regular cardiology visits",
                "Home monitoring devices",
                "Lifestyle counseling"
            ]
        }


class OncologySpecialty(BaseMedicalSpecialty):
    """Oncology specialty with cancer diagnosis and treatment focus"""

    def __init__(self):
        config = SpecialtyConfig(
            name="Oncology",
            description="Cancer diagnosis, staging, and treatment",
            key_symptoms=[
                "unexplained weight loss", "fatigue", "pain", "lump or mass",
                "changes in skin", "persistent cough", "difficulty swallowing",
                "unusual bleeding", "fever", "night sweats"
            ],
            common_conditions=[
                "breast cancer", "lung cancer", "colorectal cancer",
                "prostate cancer", "skin cancer", "leukemia", "lymphoma",
                "pancreatic cancer", "ovarian cancer", "liver cancer"
            ],
            diagnostic_tests=[
                "biopsy", "CT scan", "MRI", "PET scan", "mammography",
                "colonoscopy", "PSA test", "tumor markers", "bone marrow biopsy",
                "genetic testing", "liquid biopsy"
            ],
            treatment_modalities=[
                "surgery", "chemotherapy", "radiation therapy", "immunotherapy",
                "targeted therapy", "hormone therapy", "stem cell transplant",
                "clinical trials", "palliative care"
            ],
            ai_models=[
                "cancer_detection_ai", "tumor_segmentation", "risk_prediction",
                "treatment_response_predictor", "survival_analyzer"
            ],
            risk_factors=[
                "age", "family history", "genetic mutations", "smoking",
                "alcohol use", "obesity", "radiation exposure", "chemical exposure",
                "chronic infections", "hormonal factors"
            ],
            preventive_measures=[
                "regular screening", "healthy lifestyle", "vaccinations",
                "genetic counseling", "risk factor modification", "early detection"
            ]
        )
        super().__init__(config)
        self._initialize_oncology_rules()

    def _initialize_oncology_rules(self):
        """Initialize oncology diagnostic rules"""
        self.diagnostic_rules = {
            "breast_cancer": {
                "symptoms": ["breast_lump", "nipple_changes", "skin_changes"],
                "risk_factors": ["family_history", "brca_mutation", "age"],
                "screening_tests": ["mammography", "ultrasound", "mri"],
                "staging": ["tumor_size", "lymph_node_involvement", "metastasis"]
            },
            "lung_cancer": {
                "symptoms": ["persistent_cough", "hemoptysis", "weight_loss"],
                "risk_factors": ["smoking", "asbestos_exposure", "radon_exposure"],
                "screening_tests": ["chest_ct", "sputum_cytology", "biopsy"],
                "staging": ["tumor_size", "regional_lymph_nodes", "distant_metastasis"]
            },
            "colorectal_cancer": {
                "symptoms": ["rectal_bleeding", "change_in_bowel_habits", "abdominal_pain"],
                "risk_factors": ["age", "family_history", "inflammatory_bowel_disease"],
                "screening_tests": ["colonoscopy", "fecal_immunochemical_test", "ct_colonography"],
                "staging": ["tumor_penetration", "regional_lymph_nodes", "distant_metastasis"]
            }
        }

    def assess_patient(self, patient_data: PatientAssessment) -> SpecialtyDiagnosis:
        """Assess oncology patient"""
        symptoms = [s.lower().replace(' ', '_') for s in patient_data.symptoms]
        risk_factors = [rf.lower().replace(' ', '_') for rf in patient_data.risk_factors]

        # Comprehensive cancer screening assessment
        cancer_indicators = {
            "breast_cancer": any(s in symptoms for s in ["breast_lump", "nipple_discharge", "skin_dimpling"]),
            "lung_cancer": any(s in symptoms for s in ["persistent_cough", "hemoptysis", "chest_pain"]),
            "colorectal_cancer": any(s in symptoms for s in ["rectal_bleeding", "change_in_bowel_habits"]),
            "prostate_cancer": any(s in symptoms for s in ["urinary_difficulty", "bone_pain"]),
            "skin_cancer": any(s in symptoms for s in ["skin_lesion", "changing_mole"])
        }

        # Check for high-risk indicators
        if cancer_indicators["breast_cancer"] and ("family_history" in risk_factors or "brca_mutation" in risk_factors):
            return SpecialtyDiagnosis(
                condition="Suspected Breast Cancer",
                confidence_score=0.82,
                supporting_evidence=["Breast abnormality", "Strong family history"],
                differential_diagnoses=["Fibroadenoma", "Cyst", "Mastitis"],
                recommended_tests=["Diagnostic mammogram", "Ultrasound", "Core biopsy"],
                urgency_level="urgent",
                specialty="Oncology"
            )

        # Age and symptom-based assessment
        age = patient_data.physical_exam.get('age', 50)
        if age > 50 and any(cancer_indicators.values()):
            return SpecialtyDiagnosis(
                condition="Cancer Screening Recommended",
                confidence_score=0.65,
                supporting_evidence=["Age-appropriate screening indicated", "Symptoms present"],
                differential_diagnoses=["Benign conditions", "Infectious processes"],
                recommended_tests=["Age-appropriate cancer screening", "Diagnostic workup"],
                urgency_level="routine",
                specialty="Oncology"
            )

        return SpecialtyDiagnosis(
            condition="Oncology Consultation Recommended",
            confidence_score=0.4,
            supporting_evidence=["Further evaluation needed"],
            differential_diagnoses=self.config.common_conditions[:3],
            recommended_tests=["Comprehensive physical exam", "Initial screening tests"],
            urgency_level="routine",
            specialty="Oncology"
        )

    def create_treatment_plan(self, diagnosis: SpecialtyDiagnosis, patient_data: PatientAssessment) -> TreatmentPlan:
        """Create oncology treatment plan"""
        medications = []
        procedures = []
        lifestyle_mods = []
        monitoring = []

        if "breast" in diagnosis.condition.lower():
            procedures.extend([
                {"name": "Breast Biopsy", "timing": "Urgent", "indication": "Tissue diagnosis"},
                {"name": "Breast MRI", "timing": "Within 2 weeks", "indication": "Extent of disease"}
            ])
            medications.append({
                "name": "Supportive Care Medications",
                "dosage": "As needed",
                "indication": "Symptom management"
            })
            monitoring.extend(["Tumor markers", "Imaging studies", "Clinical follow-up"])

        elif "lung" in diagnosis.condition.lower():
            procedures.extend([
                {"name": "Bronchoscopy", "timing": "Urgent", "indication": "Diagnosis and staging"},
                {"name": "PET-CT Scan", "timing": "Within 1 week", "indication": "Staging evaluation"}
            ])
            medications.append({
                "name": "Smoking Cessation Support",
                "dosage": "As indicated",
                "indication": "Risk factor modification"
            })

        # General oncology recommendations
        lifestyle_mods.extend([
            "Nutritional support and counseling",
            "Exercise as tolerated",
            "Psychological support",
            "Sleep hygiene optimization"
        ])

        return TreatmentPlan(
            patient_id=patient_data.patient_id,
            specialty="Oncology",
            primary_diagnosis=diagnosis.condition,
            medications=medications,
            procedures=procedures,
            lifestyle_modifications=lifestyle_mods,
            follow_up_schedule=[
                {"timing": "1 week", "purpose": "Initial evaluation"},
                {"timing": "2-4 weeks", "purpose": "Multidisciplinary tumor board review"},
                {"timing": "Regular intervals", "purpose": "Treatment monitoring and surveillance"}
            ],
            monitoring_parameters=monitoring,
            expected_outcomes=[
                "Accurate diagnosis and staging",
                "Appropriate treatment initiation",
                "Symptom control and quality of life maintenance"
            ],
            plan_date=datetime.now()
        )

    def predict_outcomes(self, treatment_plan: TreatmentPlan, patient_data: PatientAssessment) -> Dict[str, Any]:
        """Predict oncology treatment outcomes"""
        # Simplified outcome prediction based on diagnosis type and patient factors
        base_survival_rate = 0.7
        age = patient_data.physical_exam.get('age', 60)

        # Age adjustment
        if age > 70:
            age_adjustment = -0.15
        elif age < 50:
            age_adjustment = 0.1
        else:
            age_adjustment = 0

        # Risk factor adjustment
        risk_adjustment = len(patient_data.risk_factors) * -0.05
        adjusted_survival_rate = max(0.3, min(0.95, base_survival_rate + age_adjustment + risk_adjustment))

        return {
            "estimated_survival_rate": adjusted_survival_rate,
            "treatment_response_probability": 0.75,
            "quality_of_life_impact": "Variable depending on treatment type and stage",
            "potential_complications": [
                "Treatment-related side effects",
                "Disease progression",
                "Secondary malignancies",
                "Psychological impact"
            ],
            "supportive_care_needs": [
                "Pain management",
                "Nutritional support",
                "Psychological counseling",
                "Palliative care as needed"
            ],
            "long_term_monitoring": [
                "Regular imaging and tumor markers",
                "Survivorship care planning",
                "Secondary cancer screening"
            ]
        }


class NeurologySpecialty(BaseMedicalSpecialty):
    """Neurology specialty with neurological disorders focus"""

    def __init__(self):
        config = SpecialtyConfig(
            name="Neurology",
            description="Neurological disorders diagnosis and treatment",
            key_symptoms=[
                "headache", "dizziness", "seizures", "weakness", "numbness",
                "tremor", "memory loss", "confusion", "speech difficulty",
                "vision changes", "balance problems", "sleep disturbances"
            ],
            common_conditions=[
                "stroke", "epilepsy", "multiple sclerosis", "parkinson's disease",
                "alzheimer's disease", "migraine", "neuropathy", "brain tumor",
                "aneurysm", "hydrocephalus"
            ],
            diagnostic_tests=[
                "MRI brain", "CT brain", "EEG", "EMG", "nerve conduction studies",
                "lumbar puncture", "cognitive assessment", "neuropsychological testing",
                "evoked potentials", "polysomnography", "carotid ultrasound"
            ],
            treatment_modalities=[
                "medications", "physical therapy", "occupational therapy",
                "speech therapy", "cognitive rehabilitation", "surgical interventions",
                "deep brain stimulation", "botulinum toxin injections", "plasmapheresis"
            ],
            ai_models=[
                "stroke_detection_ai", "epilepsy_prediction", "neuroimaging_analyzer",
                "cognitive_assessment_ai", "movement_disorder_classifier"
            ],
            risk_factors=[
                "hypertension", "diabetes", "smoking", "high cholesterol",
                "atrial fibrillation", "family history", "age", "obesity",
                "sleep apnea", "head trauma"
            ],
            preventive_measures=[
                "blood pressure control", "healthy lifestyle", "regular exercise",
                "cognitive stimulation", "fall prevention", "stroke prevention",
                "migraine trigger avoidance", "sleep hygiene"
            ]
        )
        super().__init__(config)
        self._initialize_neurology_rules()

    def _initialize_neurology_rules(self):
        """Initialize neurology diagnostic rules"""
        self.diagnostic_rules = {
            "acute_stroke": {
                "symptoms": ["sudden_weakness", "speech_difficulty", "vision_changes", "severe_headache"],
                "time_window": "4.5 hours",
                "urgency": "emergency"
            },
            "epilepsy": {
                "symptoms": ["seizures", "post_ictal_confusion", "tongue_biting"],
                "patterns": ["recurrent_seizures", "trigger_identification"],
                "urgency": "urgent"
            },
            "multiple_sclerosis": {
                "symptoms": ["visual_changes", "weakness", "sensory_changes", "balance_problems"],
                "progression": ["relapsing_remitting", "progressive"],
                "urgency": "urgent"
            }
        }

    def assess_patient(self, patient_data: PatientAssessment) -> SpecialtyDiagnosis:
        """Assess neurology patient"""
        symptoms = [s.lower().replace(' ', '_') for s in patient_data.symptoms]

        # Check for acute stroke (FAST assessment)
        stroke_symptoms = ["sudden_weakness", "speech_difficulty", "facial_droop", "severe_headache"]
        if any(s in symptoms for s in stroke_symptoms):
            return SpecialtyDiagnosis(
                condition="Suspected Acute Stroke",
                confidence_score=0.88,
                supporting_evidence=["Acute onset neurological symptoms", "FAST criteria positive"],
                differential_diagnoses=["Migraine", "Seizure", "Hypoglycemia", "Bell's palsy"],
                recommended_tests=["Immediate CT brain", "Neurological examination", "NIH Stroke Scale"],
                urgency_level="emergency",
                specialty="Neurology"
            )

        # Check for seizure activity
        seizure_indicators = ["seizure", "convulsion", "post_ictal", "tongue_biting"]
        if any(indicator in ' '.join(symptoms) for indicator in seizure_indicators):
            return SpecialtyDiagnosis(
                condition="Seizure Disorder",
                confidence_score=0.78,
                supporting_evidence=["Seizure activity reported", "Post-ictal symptoms"],
                differential_diagnoses=["Syncope", "Psychogenic episodes", "Sleep disorders"],
                recommended_tests=["EEG", "MRI brain", "Blood work", "Sleep study"],
                urgency_level="urgent",
                specialty="Neurology"
            )

        # Check for movement disorders
        movement_symptoms = ["tremor", "rigidity", "bradykinesia", "postural_instability"]
        if any(s in symptoms for s in movement_symptoms):
            return SpecialtyDiagnosis(
                condition="Movement Disorder",
                confidence_score=0.72,
                supporting_evidence=["Movement abnormalities", "Progressive symptoms"],
                differential_diagnoses=["Parkinson's disease", "Essential tremor", "Drug-induced"],
                recommended_tests=["Neurological examination", "DaTscan", "Genetic testing"],
                urgency_level="urgent",
                specialty="Neurology"
            )

        return SpecialtyDiagnosis(
            condition="Neurological Evaluation Needed",
            confidence_score=0.5,
            supporting_evidence=["Neurological symptoms present"],
            differential_diagnoses=self.config.common_conditions[:4],
            recommended_tests=["Comprehensive neurological exam", "MRI brain", "EEG"],
            urgency_level="routine",
            specialty="Neurology"
        )

    def create_treatment_plan(self, diagnosis: SpecialtyDiagnosis, patient_data: PatientAssessment) -> TreatmentPlan:
        """Create neurology treatment plan"""
        medications = []
        procedures = []
        lifestyle_mods = []
        monitoring = []

        if "stroke" in diagnosis.condition.lower():
            medications.extend([
                {"name": "tPA", "dosage": "IV protocol", "indication": "Thrombolytic therapy (if within window)"},
                {"name": "Aspirin", "dosage": "325mg loading dose", "indication": "Antiplatelet therapy"},
                {"name": "Atorvastatin", "dosage": "80mg daily", "indication": "Statin therapy"}
            ])
            procedures.append({
                "name": "Mechanical Thrombectomy",
                "timing": "Emergency",
                "indication": "Large vessel occlusion"
            })
            monitoring.extend(["Neurological status", "Vital signs", "NIH Stroke Scale"])

        elif "seizure" in diagnosis.condition.lower():
            medications.extend([
                {"name": "Levetiracetam", "dosage": "500mg twice daily", "indication": "Antiepileptic therapy"},
                {"name": "Lorazepam", "dosage": "2mg IV", "indication": "Acute seizure control"}
            ])
            lifestyle_mods.extend([
                "Sleep hygiene optimization",
                "Stress reduction techniques",
                "Seizure diary maintenance",
                "Driving restrictions assessment"
            ])
            monitoring.extend(["Seizure frequency", "Medication levels", "EEG monitoring"])

        elif "movement" in diagnosis.condition.lower():
            medications.append({
                "name": "Carbidopa/Levodopa",
                "dosage": "25/100mg three times daily",
                "indication": "Dopamine replacement therapy"
            })
            procedures.append({
                "name": "Physical Therapy Evaluation",
                "timing": "Within 2 weeks",
                "indication": "Functional assessment and rehabilitation planning"
            })
            lifestyle_mods.extend([
                "Regular exercise program",
                "Balance training",
                "Speech therapy evaluation",
                "Swallowing evaluation"
            ])

        return TreatmentPlan(
            patient_id=patient_data.patient_id,
            specialty="Neurology",
            primary_diagnosis=diagnosis.condition,
            medications=medications,
            procedures=procedures,
            lifestyle_modifications=lifestyle_mods,
            follow_up_schedule=[
                {"timing": "24-48 hours", "purpose": "Acute management assessment"},
                {"timing": "1 week", "purpose": "Treatment response evaluation"},
                {"timing": "1 month", "purpose": "Comprehensive neurological assessment"}
            ],
            monitoring_parameters=monitoring,
            expected_outcomes=[
                "Neurological function optimization",
                "Symptom control and management",
                "Prevention of complications",
                "Improved quality of life"
            ],
            plan_date=datetime.now()
        )

    def predict_outcomes(self, treatment_plan: TreatmentPlan, patient_data: PatientAssessment) -> Dict[str, Any]:
        """Predict neurology treatment outcomes"""
        base_recovery_rate = 0.65

        # Time from symptom onset (critical for stroke outcomes)
        time_factor = 1.0
        if "stroke" in patient_data.medical_history:
            # Simplified time-based adjustment
            time_factor = 0.8  # Reduced recovery expectation

        # Age adjustment
        age = patient_data.physical_exam.get('age', 50)
        if age > 80:
            age_adjustment = -0.2
        elif age < 30:
            age_adjustment = 0.1
        else:
            age_adjustment = 0

        adjusted_recovery_rate = max(0.2, min(0.9, base_recovery_rate * time_factor + age_adjustment))

        return {
            "neurological_recovery_probability": adjusted_recovery_rate,
            "functional_independence_probability": adjusted_recovery_rate * 0.9,
            "complication_risk": 1 - adjusted_recovery_rate,
            "rehabilitation_needs": [
                "Physical therapy",
                "Occupational therapy",
                "Speech therapy",
                "Cognitive rehabilitation"
            ],
            "long_term_monitoring": [
                "Regular neurological follow-up",
                "Functional assessments",
                "Comorbidity management",
                "Medication monitoring"
            ],
            "prognostic_factors": [
                "Age at onset",
                "Time to treatment initiation",
                "Severity of neurological deficits",
                "Presence of comorbidities"
            ]
        }


class EndocrinologySpecialty(BaseMedicalSpecialty):
    """Endocrinology specialty with hormonal disorders focus"""

    def __init__(self):
        config = SpecialtyConfig(
            name="Endocrinology",
            description="Hormonal and metabolic disorders diagnosis and treatment",
            key_symptoms=[
                "fatigue", "weight changes", "thirst", "frequent urination",
                "heat intolerance", "cold intolerance", "hair loss", "skin changes",
                "mood changes", "sleep disturbances", "muscle weakness", "bone pain"
            ],
            common_conditions=[
                "diabetes mellitus", "thyroid disorders", "adrenal disorders",
                "pituitary disorders", "parathyroid disorders", "gonadal disorders",
                "metabolic syndrome", "osteoporosis", "obesity", "polycystic ovary syndrome"
            ],
            diagnostic_tests=[
                "hormone levels", "thyroid function tests", "glucose tolerance test",
                "cortisol levels", "ACTH stimulation test", "insulin levels",
                "bone density scan", "24-hour urine collection", "oral glucose tolerance test",
                "thyroid ultrasound", "fine needle aspiration"
            ],
            treatment_modalities=[
                "medications", "lifestyle modifications", "insulin therapy",
                "thyroid hormone replacement", "radioactive iodine", "surgery",
                "radiation therapy", "chemotherapy", "hormone therapy"
            ],
            ai_models=[
                "diabetes_prediction_ai", "thyroid_nodule_classifier", "hormone_analyzer",
                "metabolic_syndrome_predictor", "bone_density_analyzer"
            ],
            risk_factors=[
                "family history", "obesity", "sedentary lifestyle", "poor diet",
                "age", "ethnicity", "autoimmune disorders", "radiation exposure",
                "certain medications", "stress"
            ],
            preventive_measures=[
                "healthy weight maintenance", "regular exercise", "balanced diet",
                "regular health screenings", "stress management", "adequate sleep",
                "avoidance of endocrine disruptors"
            ]
        )
        super().__init__(config)
        self._initialize_endocrinology_rules()

    def _initialize_endocrinology_rules(self):
        """Initialize endocrinology diagnostic rules"""
        self.diagnostic_rules = {
            "diabetic_ketoacidosis": {
                "symptoms": ["polyuria", "polydipsia", "fatigue", "nausea"],
                "lab_findings": ["hyperglycemia", "ketonuria", "metabolic_acidosis"],
                "urgency": "emergency"
            },
            "hyperthyroidism": {
                "symptoms": ["heat_intolerance", "weight_loss", "tachycardia", "tremor"],
                "findings": ["goiter", "exophthalmos", "pretibial_myxedema"],
                "urgency": "urgent"
            },
            "hypothyroidism": {
                "symptoms": ["cold_intolerance", "weight_gain", "fatigue", "constipation"],
                "findings": ["bradycardia", "delayed_relaxation_phase"],
                "urgency": "routine"
            }
        }

    def assess_patient(self, patient_data: PatientAssessment) -> SpecialtyDiagnosis:
        """Assess endocrinology patient"""
        symptoms = [s.lower().replace(' ', '_') for s in patient_data.symptoms]

        # Check for diabetic ketoacidosis
        dka_symptoms = ["polyuria", "polydipsia", "fatigue", "nausea", "vomiting"]
        if (any(s in symptoms for s in dka_symptoms) and
            patient_data.diagnostic_results.get('glucose', 0) > 300):
            return SpecialtyDiagnosis(
                condition="Diabetic Ketoacidosis",
                confidence_score=0.92,
                supporting_evidence=["Classic symptoms present", "Severe hyperglycemia"],
                differential_diagnoses=["Other causes of acidosis", "Infection", "Medication effects"],
                recommended_tests=["Blood glucose", "Arterial blood gas", "Urine ketones", "Serum ketones"],
                urgency_level="emergency",
                specialty="Endocrinology"
            )

        # Check for thyroid disorders
        hyper_symptoms = ["heat_intolerance", "weight_loss", "tachycardia", "tremor"]
        hypo_symptoms = ["cold_intolerance", "weight_gain", "fatigue", "constipation"]

        if any(s in symptoms for s in hyper_symptoms):
            return SpecialtyDiagnosis(
                condition="Hyperthyroidism",
                confidence_score=0.78,
                supporting_evidence=["Hyperthyroid symptoms", "Physical examination findings"],
                differential_diagnoses=["Anxiety", "Pheochromocytoma", "Medication effects"],
                recommended_tests=["TSH", "Free T4", "Free T3", "Thyroid antibodies", "Thyroid ultrasound"],
                urgency_level="urgent",
                specialty="Endocrinology"
            )

        elif any(s in symptoms for s in hypo_symptoms):
            return SpecialtyDiagnosis(
                condition="Hypothyroidism",
                confidence_score=0.75,
                supporting_evidence=["Hypothyroid symptoms", "Physical examination findings"],
                differential_diagnoses=["Depression", "Anemia", "Chronic fatigue syndrome"],
                recommended_tests=["TSH", "Free T4", "Thyroid antibodies"],
                urgency_level="routine",
                specialty="Endocrinology"
            )

        # Check for diabetes
        diabetes_symptoms = ["polyuria", "polydipsia", "unexplained_weight_loss"]
        if (any(s in symptoms for s in diabetes_symptoms) and
            patient_data.diagnostic_results.get('glucose', 0) > 140):
            return SpecialtyDiagnosis(
                condition="Diabetes Mellitus",
                confidence_score=0.82,
                supporting_evidence=["Classic symptoms", "Elevated blood glucose"],
                differential_diagnoses=["Other causes of hyperglycemia", "Steroid use", "Pancreatic disease"],
                recommended_tests=["Fasting glucose", "HbA1c", "Oral glucose tolerance test", "C-peptide"],
                urgency_level="urgent",
                specialty="Endocrinology"
            )

        return SpecialtyDiagnosis(
            condition="Endocrine Evaluation Needed",
            confidence_score=0.4,
            supporting_evidence=["Endocrine symptoms present"],
            differential_diagnoses=self.config.common_conditions[:4],
            recommended_tests=["Comprehensive metabolic panel", "Hormone panel", "Thyroid function tests"],
            urgency_level="routine",
            specialty="Endocrinology"
        )

    def create_treatment_plan(self, diagnosis: SpecialtyDiagnosis, patient_data: PatientAssessment) -> TreatmentPlan:
        """Create endocrinology treatment plan"""
        medications = []
        procedures = []
        lifestyle_mods = []
        monitoring = []

        if "diabetic_ketoacidosis" in diagnosis.condition.lower():
            medications.extend([
                {"name": "Regular Insulin", "dosage": "IV bolus followed by infusion", "indication": "Acute hyperglycemia management"},
                {"name": "IV Fluids", "dosage": "Normal saline", "indication": "Rehydration and electrolyte correction"},
                {"name": "Potassium", "dosage": "As indicated by levels", "indication": "Electrolyte replacement"}
            ])
            monitoring.extend(["Blood glucose hourly", "Electrolytes", "Acid-base status", "Mental status"])

        elif "hyperthyroidism" in diagnosis.condition.lower():
            medications.extend([
                {"name": "Methimazole", "dosage": "10-20mg daily", "indication": "Thyroid hormone synthesis inhibition"},
                {"name": "Propranolol", "dosage": "20-40mg every 6 hours", "indication": "Symptom control"}
            ])
            procedures.append({
                "name": "Thyroid Function Tests",
                "timing": "Weekly until stable",
                "indication": "Treatment monitoring"
            })

        elif "hypothyroidism" in diagnosis.condition.lower():
            medications.append({
                "name": "Levothyroxine", "dosage": "25-50mcg daily", "indication": "Thyroid hormone replacement"},
            )
            monitoring.extend(["TSH levels", "Free T4 levels", "Thyroid function symptoms"])

        elif "diabetes" in diagnosis.condition.lower():
            medications.extend([
                {"name": "Metformin", "dosage": "500mg twice daily", "indication": "First-line therapy for type 2 diabetes"},
                {"name": "Glipizide", "dosage": "5mg daily", "indication": "Additional glycemic control"}
            ])
            lifestyle_mods.extend([
                "Carbohydrate counting and meal planning",
                "Regular exercise (150 minutes/week)",
                "Weight management",
                "Blood glucose monitoring",
                "Foot care and eye exams"
            ])
            monitoring.extend(["HbA1c every 3 months", "Fasting glucose", "Blood pressure", "Lipid profile"])

        return TreatmentPlan(
            patient_id=patient_data.patient_id,
            specialty="Endocrinology",
            primary_diagnosis=diagnosis.condition,
            medications=medications,
            procedures=procedures,
            lifestyle_modifications=lifestyle_mods,
            follow_up_schedule=[
                {"timing": "1 week", "purpose": "Treatment initiation and monitoring"},
                {"timing": "1 month", "purpose": "Dose adjustment and response assessment"},
                {"timing": "3 months", "purpose": "Comprehensive evaluation and long-term planning"}
            ],
            monitoring_parameters=monitoring,
            expected_outcomes=[
                "Hormonal/metabolic stabilization",
                "Symptom resolution",
                "Prevention of complications",
                "Improved quality of life"
            ],
            plan_date=datetime.now()
        )

    def predict_outcomes(self, treatment_plan: TreatmentPlan, patient_data: PatientAssessment) -> Dict[str, Any]:
        """Predict endocrinology treatment outcomes"""
        base_success_rate = 0.8

        # Condition-specific adjustments
        if "diabetic_ketoacidosis" in patient_data.medical_history:
            condition_adjustment = 0.1  # Good response to acute treatment
        elif "diabetes" in patient_data.medical_history:
            condition_adjustment = -0.1  # Chronic condition management
        else:
            condition_adjustment = 0

        # Age adjustment
        age = patient_data.physical_exam.get('age', 50)
        age_adjustment = max(-0.15, min(0.05, (50 - age) * 0.005))

        # Compliance factor (simplified)
        compliance_factor = 0.9

        adjusted_success_rate = max(0.5, min(0.95, base_success_rate + condition_adjustment + age_adjustment)) * compliance_factor

        return {
            "treatment_success_probability": adjusted_success_rate,
            "metabolic_control_probability": adjusted_success_rate * 0.9,
            "complication_prevention_rate": adjusted_success_rate * 0.85,
            "quality_of_life_improvement": "Expected improvement with treatment adherence",
            "monitoring_intensity": "Regular endocrine follow-up required",
            "long_term_prognosis": "Good with appropriate management and lifestyle modifications",
            "potential_challenges": [
                "Treatment adherence",
                "Lifestyle modification",
                "Comorbid condition management",
                "Medication side effects"
            ]
        }


class MedicalSpecialtiesManager:
    """Manager for all medical specialties"""

    def __init__(self):
        self.specialties = {
            'cardiology': CardiologySpecialty(),
            'oncology': OncologySpecialty(),
            'neurology': NeurologySpecialty(),
            'endocrinology': EndocrinologySpecialty()
        }

    def get_specialty(self, name: str) -> Optional[BaseMedicalSpecialty]:
        """Get a medical specialty by name"""
        return self.specialties.get(name.lower())

    def assess_patient_specialty(self, patient_data: PatientAssessment,
                               specialty_name: str) -> SpecialtyDiagnosis:
        """Assess patient in specific specialty"""
        specialty = self.get_specialty(specialty_name)
        if not specialty:
            raise ValueError(f"Unknown specialty: {specialty_name}")

        return specialty.assess_patient(patient_data)

    def create_treatment_plan(self, diagnosis: SpecialtyDiagnosis,
                            patient_data: PatientAssessment) -> TreatmentPlan:
        """Create treatment plan for diagnosis"""
        specialty = self.get_specialty(diagnosis.specialty)
        if not specialty:
            raise ValueError(f"Unknown specialty: {diagnosis.specialty}")

        return specialty.create_treatment_plan(diagnosis, patient_data)

    def predict_treatment_outcomes(self, treatment_plan: TreatmentPlan,
                                patient_data: PatientAssessment) -> Dict[str, Any]:
        """Predict treatment outcomes"""
        specialty = self.get_specialty(treatment_plan.specialty)
        if not specialty:
            raise ValueError(f"Unknown specialty: {treatment_plan.specialty}")

        return specialty.predict_outcomes(treatment_plan, patient_data)

    def get_available_specialties(self) -> List[str]:
        """Get list of available medical specialties"""
        return list(self.specialties.keys())

    def get_specialty_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specialty"""
        specialty = self.get_specialty(name)
        if not specialty:
            return None

        return {
            'name': specialty.config.name,
            'description': specialty.config.description,
            'key_symptoms': specialty.config.key_symptoms,
            'common_conditions': specialty.config.common_conditions,
            'diagnostic_tests': specialty.config.diagnostic_tests,
            'treatment_modalities': specialty.config.treatment_modalities,
            'risk_factors': specialty.config.risk_factors,
            'preventive_measures': specialty.config.preventive_measures
        }


# Global specialty manager instance
specialty_manager = MedicalSpecialtiesManager()
