"""
Comprehensive data structures for AI Personalized Medicine Platform
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
import uuid

@dataclass
class PatientProfile:
    """Comprehensive patient profile data structure"""
    patient_id: str
    demographics: Dict[str, Any]
    medical_history: List[Dict[str, Any]]
    family_history: List[Dict[str, Any]] = field(default_factory=list)
    lifestyle_factors: Dict[str, Any] = field(default_factory=dict)
    genomic_data: Optional[Dict[str, Any]] = None
    proteomic_data: Optional[Dict[str, Any]] = None
    metabolomic_data: Optional[Dict[str, Any]] = None
    microbiomic_data: Optional[Dict[str, Any]] = None
    imaging_data: List[Dict[str, Any]] = field(default_factory=list)
    wearable_data: Dict[str, Any] = field(default_factory=dict)
    social_determinants: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "patient_id": self.patient_id,
            "demographics": self.demographics,
            "medical_history": self.medical_history,
            "family_history": self.family_history,
            "lifestyle_factors": self.lifestyle_factors,
            "genomic_data": self.genomic_data,
            "proteomic_data": self.proteomic_data,
            "metabolomic_data": self.metabolomic_data,
            "microbiomic_data": self.microbiomic_data,
            "imaging_data": self.imaging_data,
            "wearable_data": self.wearable_data,
            "social_determinants": self.social_determinants,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

@dataclass
class GenomicVariant:
    """Genomic variant data structure"""
    chromosome: str
    position: int
    reference: str
    alternate: str
    variant_id: str
    variant_type: str
    quality: float
    depth: int
    allele_frequency: float
    genotype: str
    annotations: Dict[str, Any] = field(default_factory=dict)
    clinical_significance: str = "unknown"
    disease_associations: List[str] = field(default_factory=list)
    drug_responses: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenomicAnalysis:
    """Comprehensive genomic analysis results"""
    patient_id: str
    genome_build: str
    variants: List[GenomicVariant]
    pharmacogenomic_profile: Dict[str, Any]
    disease_risk_scores: Dict[str, float]
    ancestry_composition: Dict[str, float]
    carrier_status: Dict[str, bool]
    polygenic_risk_scores: Dict[str, float]
    mitochondrial_analysis: Dict[str, Any]
    copy_number_variants: List[Dict[str, Any]]
    structural_variants: List[Dict[str, Any]]
    analysis_date: datetime = field(default_factory=datetime.now)

@dataclass
class DrugCompound:
    """Drug compound data structure"""
    compound_id: str
    smiles: str
    molecular_weight: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    molecular_formula: str
    structure_properties: Dict[str, Any]
    predicted_targets: List[str]
    predicted_activities: Dict[str, float]
    toxicity_predictions: Dict[str, float]
    pharmacokinetic_properties: Dict[str, Any]
    synthesis_routes: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class DrugDiscoveryResult:
    """Drug discovery analysis results"""
    target_protein: str
    disease_context: str
    candidate_compounds: List[DrugCompound]
    binding_affinities: Dict[str, float]
    selectivity_profiles: Dict[str, Any]
    optimization_suggestions: List[str]
    clinical_trial_readiness: Dict[str, Any]
    patent_landscape: Dict[str, Any]

@dataclass
class VitalSigns:
    """Vital signs data structure"""
    heart_rate: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    temperature: float
    oxygen_saturation: float
    respiratory_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Biomarker:
    """Biomarker data structure"""
    name: str
    value: float
    unit: str
    reference_range: Tuple[float, float]
    category: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class HealthMonitoringData:
    """Comprehensive health monitoring data"""
    patient_id: str
    vital_signs: VitalSigns
    biomarkers: List[Biomarker]
    symptoms: List[str]
    medications_taken: List[Dict[str, Any]]
    physical_activity: Dict[str, Any]
    sleep_data: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TreatmentPlan:
    """Personalized treatment plan"""
    patient_id: str
    diagnosis: str
    treatment_goals: List[str]
    primary_medications: List[Dict[str, Any]]
    alternative_medications: List[Dict[str, Any]]
    lifestyle_modifications: List[str]
    monitoring_schedule: List[Dict[str, Any]]
    follow_up_schedule: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    contraindications: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ClinicalTrial:
    """Clinical trial data structure"""
    trial_id: str
    title: str
    phase: int
    disease_area: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    treatment_arms: List[Dict[str, Any]]
    primary_outcome: str
    secondary_outcomes: List[str]
    target_sample_size: int
    current_enrollment: int
    estimated_completion: datetime
    sponsors: List[str]
    investigators: List[str]

@dataclass
class BlockchainRecord:
    """Blockchain-based health record"""
    record_id: str
    patient_id: str
    record_type: str
    data_hash: str
    previous_hash: str
    timestamp: datetime
    block_hash: str
    signature: str
    verified: bool = True

class HealthDataStructures:
    """Utility class for health data structure operations"""

    @staticmethod
    def validate_patient_data(patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate patient data structure"""
        required_fields = ["patient_id", "demographics", "medical_history"]
        missing_fields = []

        for field in required_fields:
            if field not in patient_data:
                missing_fields.append(field)

        if missing_fields:
            return {
                "valid": False,
                "errors": f"Missing required fields: {missing_fields}"
            }

        # Validate demographics
        demographics = patient_data.get("demographics", {})
        required_demo = ["age", "gender", "ethnicity"]
        missing_demo = [f for f in required_demo if f not in demographics]

        if missing_demo:
            return {
                "valid": False,
                "errors": f"Missing demographic fields: {missing_demo}"
            }

        return {"valid": True, "errors": []}

    @staticmethod
    def calculate_patient_risk_score(patient_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive risk scores for patient"""
        risk_scores = {}

        # Age-based risk
        age = patient_data.get("demographics", {}).get("age", 0)
        if age > 65:
            risk_scores["age_risk"] = min(age / 100, 1.0)
        else:
            risk_scores["age_risk"] = age / 100

        # Family history risk
        family_history = patient_data.get("family_history", [])
        family_risk = len(family_history) * 0.1
        risk_scores["family_history_risk"] = min(family_risk, 1.0)

        # Lifestyle risk
        lifestyle = patient_data.get("lifestyle_factors", {})
        lifestyle_risk = 0
        if lifestyle.get("smoking", False):
            lifestyle_risk += 0.3
        if lifestyle.get("alcohol_abuse", False):
            lifestyle_risk += 0.2
        if lifestyle.get("sedentary", False):
            lifestyle_risk += 0.1
        risk_scores["lifestyle_risk"] = lifestyle_risk

        # Overall risk score (weighted average)
        weights = {"age_risk": 0.3, "family_history_risk": 0.4, "lifestyle_risk": 0.3}
        overall_risk = sum(risk_scores[key] * weights[key] for key in risk_scores)
        risk_scores["overall_risk"] = overall_risk

        return risk_scores

    @staticmethod
    def create_health_timeline(patient_id: str, health_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chronological health timeline"""
        timeline = []

        for record in health_records:
            timeline_entry = {
                "timestamp": record.get("timestamp"),
                "type": record.get("type", "health_data"),
                "summary": HealthDataStructures._summarize_health_record(record),
                "severity": HealthDataStructures._calculate_record_severity(record),
                "recommendations": HealthDataStructures._generate_record_recommendations(record)
            }
            timeline.append(timeline_entry)

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    @staticmethod
    def _summarize_health_record(record: Dict[str, Any]) -> str:
        """Generate summary for health record"""
        record_type = record.get("type", "unknown")

        if record_type == "vital_signs":
            hr = record.get("heart_rate", 0)
            return f"Heart rate: {hr} bpm"
        elif record_type == "biomarker":
            name = record.get("name", "unknown")
            value = record.get("value", 0)
            unit = record.get("unit", "")
            return f"{name}: {value} {unit}"
        elif record_type == "symptom":
            symptoms = record.get("symptoms", [])
            return f"Symptoms: {', '.join(symptoms)}"

        return "Health data recorded"

    @staticmethod
    def _calculate_record_severity(record: Dict[str, Any]) -> str:
        """Calculate severity level for health record"""
        # Simple severity calculation based on thresholds
        severity = "normal"

        if record.get("type") == "vital_signs":
            hr = record.get("heart_rate", 80)
            if hr < 50 or hr > 150:
                severity = "high"
            elif hr < 60 or hr > 100:
                severity = "moderate"

        return severity

    @staticmethod
    def _generate_record_recommendations(record: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health record"""
        recommendations = []

        if record.get("type") == "vital_signs":
            hr = record.get("heart_rate", 80)
            if hr > 100:
                recommendations.append("Consider relaxation techniques to lower heart rate")
            elif hr < 60:
                recommendations.append("Monitor for fatigue or dizziness")

        return recommendations

class DataValidation:
    """Data validation utilities"""

    @staticmethod
    def validate_genomic_data(genome_data: str) -> Dict[str, Any]:
        """Validate genomic sequence data"""
        errors = []

        # Check for valid nucleotides
        valid_nucleotides = set("ATCGNatcgn")
        invalid_chars = set(genome_data.upper()) - valid_nucleotides

        if invalid_chars:
            errors.append(f"Invalid nucleotides found: {invalid_chars}")

        # Check sequence length
        if len(genome_data) < 1000:
            errors.append("Genome sequence too short for analysis")

        # Check for N content (unknown bases)
        n_content = genome_data.upper().count('N') / len(genome_data)
        if n_content > 0.1:
            errors.append(f"High unknown base content: {n_content:.2%}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "sequence_length": len(genome_data),
            "gc_content": (genome_data.upper().count('G') + genome_data.upper().count('C')) / len(genome_data),
            "n_content": n_content
        }

    @staticmethod
    def validate_drug_structure(smiles: str) -> Dict[str, Any]:
        """Validate drug molecular structure (SMILES format)"""
        errors = []

        # Basic SMILES validation
        if not smiles or len(smiles.strip()) == 0:
            errors.append("Empty SMILES string")

        # Check for balanced parentheses
        open_parens = smiles.count('(')
        close_parens = smiles.count(')')
        if open_parens != close_parens:
            errors.append("Unbalanced parentheses in SMILES")

        # Check for valid characters (simplified)
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}@+-=#.:/")
        invalid_chars = set(smiles) - valid_chars

        if invalid_chars:
            errors.append(f"Invalid characters in SMILES: {invalid_chars}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "length": len(smiles),
            "atoms": smiles.count('C') + smiles.count('N') + smiles.count('O') + smiles.count('S')
        }

class DataSerialization:
    """Data serialization utilities"""

    @staticmethod
    def serialize_patient_data(patient: PatientProfile) -> str:
        """Serialize patient data to JSON"""
        return json.dumps(patient.to_dict(), indent=2, default=str)

    @staticmethod
    def deserialize_patient_data(data: str) -> PatientProfile:
        """Deserialize patient data from JSON"""
        parsed = json.loads(data)
        return PatientProfile(**parsed)

    @staticmethod
    def create_data_hash(data: Dict[str, Any]) -> str:
        """Create SHA-256 hash of data for integrity checking"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @staticmethod
    def generate_unique_id(prefix: str = "") -> str:
        """Generate unique identifier"""
        return f"{prefix}{uuid.uuid4().hex[:16]}"
