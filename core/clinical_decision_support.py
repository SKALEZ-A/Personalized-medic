"""
Comprehensive Clinical Decision Support System for AI Personalized Medicine Platform
"""

import asyncio
import json
import random
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

from utils.data_structures import DataValidation
from utils.ml_algorithms import MachineLearningAlgorithms

class ClinicalDecisionSupport:
    """AI-powered clinical decision support system"""

    def __init__(self):
        self.guideline_database = self._initialize_guideline_database()
        self.evidence_database = self._initialize_evidence_database()
        self.decision_algorithms = DecisionAlgorithms()
        self.risk_stratification = RiskStratification()
        self.differential_diagnosis = DifferentialDiagnosis()
        self.treatment_recommendation = TreatmentRecommendation()
        self.monitoring_queue = queue.Queue()
        self.active_decisions = {}
        self.completed_decisions = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._start_decision_workers()

    def _start_decision_workers(self):
        """Start background decision support workers"""
        for i in range(4):
            worker_thread = threading.Thread(
                target=self._decision_worker,
                daemon=True,
                name=f"DecisionSupport-{i+1}"
            )
            worker_thread.start()

    def _decision_worker(self):
        """Background worker for clinical decision support"""
        while True:
            try:
                job = self.decision_queue.get(timeout=1)
                if job:
                    self._process_decision_job(job)
                    self.decision_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Clinical decision support worker error: {e}")

    def _process_decision_job(self, job: Dict[str, Any]):
        """Process clinical decision support job"""
        try:
            job["status"] = "running"
            job["started_at"] = datetime.now()
            self.active_decisions[job["decision_id"]] = job

            # Generate clinical recommendations
            recommendations = self.generate_recommendations(
                query=job["query"],
                patient_data=job["patient_data"],
                context=job.get("context", {})
            )

            # Complete job
            job["status"] = "completed"
            job["completed_at"] = datetime.now()
            job["recommendations"] = recommendations

            # Move to completed
            self.completed_decisions[job["decision_id"]] = job
            del self.active_decisions[job["decision_id"]]

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = datetime.now()
            self.completed_decisions[job["decision_id"]] = job
            if job["decision_id"] in self.active_decisions:
                del self.active_decisions[job["decision_id"]]

    def _initialize_guideline_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize clinical guideline database"""
        return {
            "hypertension_management": {
                "organization": "ACC/AHA",
                "year": 2017,
                "target_population": "adults_18_plus",
                "key_recommendations": [
                    "BP target <130/80 mmHg for high-risk patients",
                    "First-line therapy: ACEI/ARB, CCB, or thiazide diuretic",
                    "Combination therapy for BP >20/10 mmHg above target"
                ],
                "evidence_level": "A",
                "last_updated": "2017-11-13"
            },
            "diabetes_management": {
                "organization": "ADA",
                "year": 2023,
                "target_population": "adults_with_diabetes",
                "key_recommendations": [
                    "HbA1c target <7.0% for most patients",
                    "Metformin as first-line therapy unless contraindicated",
                    "Comprehensive cardiovascular risk management"
                ],
                "evidence_level": "A",
                "last_updated": "2023-01-01"
            },
            "lipid_management": {
                "organization": "ACC/AHA",
                "year": 2018,
                "target_population": "adults_with_atherosclerotic_cv_disease",
                "key_recommendations": [
                    "High-intensity statin therapy for ASCVD patients",
                    "LDL-C target <70 mg/dL for very high-risk patients",
                    "Consider ezetimibe or PCSK9 inhibitors if needed"
                ],
                "evidence_level": "A",
                "last_updated": "2018-11-10"
            }
        }

    def _initialize_evidence_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize clinical evidence database"""
        return {
            "metformin_cardiovascular_benefits": {
                "study_type": "meta_analysis",
                "n_patients": 50000,
                "outcome": "reduced_cv_events",
                "effect_size": 0.85,
                "confidence_interval": [0.78, 0.93],
                "evidence_level": "1A",
                "publication": "BMJ 2019"
            },
            "sglt2_inhibitors_heart_failure": {
                "study_type": "randomized_trial",
                "n_patients": 15000,
                "outcome": "reduced_hf_hospitalization",
                "effect_size": 0.75,
                "confidence_interval": [0.68, 0.82],
                "evidence_level": "1A",
                "publication": "NEJM 2019"
            },
            "statin_primary_prevention": {
                "study_type": "meta_analysis",
                "n_patients": 65000,
                "outcome": "reduced_cv_events",
                "effect_size": 0.88,
                "confidence_interval": [0.82, 0.95],
                "evidence_level": "1A",
                "publication": "Lancet 2019"
            }
        }

    def generate_recommendations(self, query: str, patient_data: Dict[str, Any],
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate clinical decision support recommendations"""
        if context is None:
            context = {}

        decision_id = f"decision_{int(random.random() * 10000)}"

        # Analyze query and patient data
        query_analysis = self._analyze_query(query)
        patient_analysis = self._analyze_patient_data(patient_data)

        # Generate recommendations based on query type
        if query_analysis["type"] == "diagnosis":
            recommendations = self.differential_diagnosis.generate_differential_diagnosis(
                patient_data, query_analysis
            )
        elif query_analysis["type"] == "treatment":
            recommendations = self.treatment_recommendation.generate_treatment_recommendations(
                patient_data, query_analysis, context
            )
        elif query_analysis["type"] == "monitoring":
            recommendations = self._generate_monitoring_recommendations(
                patient_data, query_analysis
            )
        elif query_analysis["type"] == "risk_assessment":
            recommendations = self.risk_stratification.assess_risks(
                patient_data, query_analysis
            )
        else:
            recommendations = self._generate_general_recommendations(
                patient_data, query_analysis
            )

        # Add evidence-based support
        evidence_support = self._add_evidence_support(recommendations)

        # Calculate confidence and uncertainty
        confidence_metrics = self._calculate_confidence_metrics(recommendations, patient_data)

        return {
            "decision_id": decision_id,
            "query": query,
            "query_analysis": query_analysis,
            "patient_analysis": patient_analysis,
            "recommendations": recommendations,
            "evidence_support": evidence_support,
            "confidence_metrics": confidence_metrics,
            "alternative_options": self._generate_alternatives(recommendations),
            "follow_up_questions": self._generate_follow_up_questions(query_analysis),
            "disclaimer": "These recommendations are AI-generated suggestions and should be reviewed by qualified healthcare providers",
            "generated_at": datetime.now().isoformat()
        }

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze clinical query to determine intent and context"""
        query_lower = query.lower()

        # Determine query type
        if any(word in query_lower for word in ["diagnos", "differential", "what is wrong"]):
            query_type = "diagnosis"
        elif any(word in query_lower for word in ["treat", "medication", "therapy", "management"]):
            query_type = "treatment"
        elif any(word in query_lower for word in ["monitor", "follow", "check", "test"]):
            query_type = "monitoring"
        elif any(word in query_lower for word in ["risk", "probability", "chance", "likely"]):
            query_type = "risk_assessment"
        else:
            query_type = "general"

        # Extract key medical concepts
        medical_terms = self._extract_medical_terms(query)

        # Determine urgency
        urgency_keywords = ["emergency", "urgent", "critical", "severe", "immediately"]
        urgency = "high" if any(word in query_lower for word in urgency_keywords) else "normal"

        return {
            "type": query_type,
            "original_query": query,
            "medical_terms": medical_terms,
            "urgency": urgency,
            "estimated_complexity": len(medical_terms) * 0.2 + (0.5 if urgency == "high" else 0)
        }

    def _extract_medical_terms(self, query: str) -> List[str]:
        """Extract medical terms from query"""
        # Simplified medical term extraction
        medical_terms = []

        # Common medical terms and abbreviations
        term_patterns = [
            r'\b(diabetes|hypertension|heart|cardiac|stroke|cancer|tumor)\b',
            r'\b(pain|chest|abdominal|headache|fever|cough)\b',
            r'\b(bp|blood pressure|glucose|hba1c|cholesterol|creatinine)\b',
            r'\b(aspirin|metformin|atorvastatin|lisinopril|amlodipine)\b'
        ]

        import re
        for pattern in term_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            medical_terms.extend(matches)

        return list(set(medical_terms))

    def _analyze_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patient data for clinical decision support"""
        analysis = {
            "demographics_summary": {},
            "clinical_summary": {},
            "risk_factors": [],
            "key_findings": [],
            "data_completeness": 0.0
        }

        # Demographics summary
        demographics = patient_data.get("demographics", {})
        age = demographics.get("age")
        gender = demographics.get("gender")

        if age:
            age_group = "elderly" if age >= 65 else "adult" if age >= 18 else "pediatric"
            analysis["demographics_summary"]["age_group"] = age_group

        analysis["demographics_summary"]["gender"] = gender

        # Clinical summary
        medical_history = patient_data.get("medical_history", [])
        current_medications = patient_data.get("current_medications", [])
        symptoms = patient_data.get("symptoms", [])

        analysis["clinical_summary"] = {
            "active_conditions": len(medical_history),
            "current_medications": len(current_medications),
            "active_symptoms": len(symptoms),
            "comorbidity_index": len(medical_history) * 0.1
        }

        # Risk factors
        risk_factors = []

        if age and age >= 65:
            risk_factors.append("advanced_age")
        if len(medical_history) >= 3:
            risk_factors.append("multiple_comorbidities")
        if len(current_medications) >= 5:
            risk_factors.append("polypharmacy")

        # Biomarker-based risk factors
        biomarkers = patient_data.get("biomarkers", [])
        for biomarker in biomarkers:
            name = biomarker.get("name")
            value = biomarker.get("value")

            if name == "glucose" and value and value > 140:
                risk_factors.append("hyperglycemia")
            elif name == "cholesterol_total" and value and value > 240:
                risk_factors.append("hypercholesterolemia")

        analysis["risk_factors"] = risk_factors

        # Key findings
        key_findings = []
        if symptoms:
            key_findings.append(f"Reports {len(symptoms)} active symptoms")
        if medical_history:
            key_findings.append(f"History of {len(medical_history)} medical conditions")
        if risk_factors:
            key_findings.append(f"Identified {len(risk_factors)} risk factors")

        analysis["key_findings"] = key_findings

        # Data completeness
        required_fields = ["demographics", "medical_history", "symptoms"]
        completed_fields = sum(1 for field in required_fields if patient_data.get(field))
        analysis["data_completeness"] = completed_fields / len(required_fields)

        return analysis

    def _generate_monitoring_recommendations(self, patient_data: Dict[str, Any],
                                           query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate monitoring recommendations"""
        recommendations = []

        # Base monitoring frequency
        base_frequency = "monthly"

        # Adjust based on risk factors
        risk_factors = patient_data.get("analysis", {}).get("risk_factors", [])
        if len(risk_factors) > 2:
            base_frequency = "weekly"
        elif len(risk_factors) > 0:
            base_frequency = "biweekly"

        # Specific monitoring recommendations
        recommendations.append({
            "type": "vital_signs",
            "frequency": base_frequency,
            "parameters": ["blood_pressure", "heart_rate", "weight"],
            "rationale": "Regular monitoring of cardiovascular parameters",
            "evidence_level": "A"
        })

        # Biomarker monitoring
        biomarkers = patient_data.get("biomarkers", [])
        if biomarkers:
            recommendations.append({
                "type": "laboratory",
                "frequency": "quarterly",
                "parameters": ["glucose", "lipid_profile", "renal_function"],
                "rationale": "Monitoring metabolic and renal parameters",
                "evidence_level": "B"
            })

        return recommendations

    def _generate_general_recommendations(self, patient_data: Dict[str, Any],
                                        query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate general clinical recommendations"""
        recommendations = []

        # Preventive care recommendations
        recommendations.append({
            "type": "preventive_care",
            "priority": "high",
            "recommendation": "Annual comprehensive health assessment",
            "rationale": "Early detection and prevention of health issues",
            "evidence_level": "A",
            "implementation": "Schedule with primary care provider"
        })

        # Lifestyle recommendations
        recommendations.append({
            "type": "lifestyle",
            "priority": "medium",
            "recommendation": "Regular physical activity and balanced nutrition",
            "rationale": "Fundamental to maintaining health and preventing disease",
            "evidence_level": "A",
            "implementation": "30 minutes moderate exercise most days, Mediterranean diet"
        })

        return recommendations

    def _add_evidence_support(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add evidence-based support to recommendations"""
        supported_recommendations = []

        for rec in recommendations:
            # Find relevant evidence
            evidence_key = self._find_relevant_evidence(rec)
            evidence = self.evidence_database.get(evidence_key, {})

            if evidence:
                rec["evidence_support"] = {
                    "study_type": evidence.get("study_type"),
                    "n_patients": evidence.get("n_patients"),
                    "effect_size": evidence.get("effect_size"),
                    "confidence_interval": evidence.get("confidence_interval"),
                    "evidence_level": evidence.get("evidence_level"),
                    "publication": evidence.get("publication")
                }
            else:
                rec["evidence_support"] = {
                    "evidence_level": "C",
                    "note": "Based on clinical guidelines and expert consensus"
                }

            supported_recommendations.append(rec)

        return supported_recommendations

    def _find_relevant_evidence(self, recommendation: Dict[str, Any]) -> Optional[str]:
        """Find relevant evidence for recommendation"""
        rec_type = recommendation.get("type", "")
        rec_text = str(recommendation).lower()

        if "metformin" in rec_text or "diabetes" in rec_text:
            return "metformin_cardiovascular_benefits"
        elif "sglt2" in rec_text or "heart failure" in rec_text:
            return "sglt2_inhibitors_heart_failure"
        elif "statin" in rec_text or "cholesterol" in rec_text:
            return "statin_primary_prevention"

        return None

    def _calculate_confidence_metrics(self, recommendations: List[Dict[str, Any]],
                                   patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for recommendations"""
        base_confidence = 0.8

        # Adjust based on data completeness
        data_completeness = patient_data.get("analysis", {}).get("data_completeness", 0.5)
        confidence = base_confidence * data_completeness

        # Adjust based on evidence level
        evidence_levels = {"A": 1.0, "B": 0.9, "C": 0.7, "D": 0.5}
        avg_evidence_quality = sum(
            evidence_levels.get(rec.get("evidence_support", {}).get("evidence_level", "C")[0], 0.7)
            for rec in recommendations
        ) / len(recommendations) if recommendations else 0.7

        confidence *= avg_evidence_quality

        # Uncertainty factors
        uncertainty_factors = []
        if data_completeness < 0.7:
            uncertainty_factors.append("incomplete_patient_data")
        if len(recommendations) > 5:
            uncertainty_factors.append("multiple_competing_recommendations")

        return {
            "overall_confidence": round(confidence, 3),
            "confidence_range": [round(confidence * 0.8, 3), round(min(confidence * 1.2, 1.0), 3)],
            "uncertainty_factors": uncertainty_factors,
            "recommendation_count": len(recommendations)
        }

    def _generate_alternatives(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alternative options"""
        alternatives = []

        for rec in recommendations:
            rec_type = rec.get("type", "")

            if rec_type == "treatment":
                alternatives.append({
                    "original_recommendation": rec.get("recommendation"),
                    "alternative": "Lifestyle modification before medication",
                    "rationale": "Minimize medication burden when possible",
                    "evidence_level": "B"
                })
            elif rec_type == "monitoring":
                alternatives.append({
                    "original_recommendation": rec.get("recommendation"),
                    "alternative": "Home monitoring with periodic clinic visits",
                    "rationale": "Cost-effective and convenient for stable patients",
                    "evidence_level": "B"
                })

        return alternatives

    def _generate_follow_up_questions(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Generate follow-up questions to clarify clinical decision"""
        questions = []

        query_type = query_analysis.get("type")

        if query_type == "diagnosis":
            questions.extend([
                "How long have these symptoms been present?",
                "Are there any associated symptoms not mentioned?",
                "What medications are currently being taken?"
            ])
        elif query_type == "treatment":
            questions.extend([
                "What treatments have been tried previously?",
                "Are there any contraindications or allergies?",
                "What is the patient's response to current medications?"
            ])
        elif query_type == "monitoring":
            questions.extend([
                "What monitoring is currently in place?",
                "Are there any barriers to recommended monitoring?",
                "How often are vital signs being checked?"
            ])

        # General questions
        questions.extend([
            "Is there any additional clinical context?",
            "Are there any recent changes in health status?"
        ])

        return questions[:5]  # Limit to 5 questions

class DecisionAlgorithms:
    """Advanced decision algorithms for clinical decision support"""

    def __init__(self):
        self.ml_algorithms = MachineLearningAlgorithms()

    def calculate_decision_confidence(self, patient_data: Dict[str, Any],
                                    recommendation: Dict[str, Any]) -> float:
        """Calculate confidence score for clinical recommendation"""
        confidence = 0.8  # Base confidence

        # Data quality factor
        data_completeness = self._assess_data_quality(patient_data)
        confidence *= data_completeness

        # Evidence strength factor
        evidence_level = recommendation.get("evidence_level", "C")
        evidence_multipliers = {"A": 1.0, "B": 0.9, "C": 0.7, "D": 0.5}
        confidence *= evidence_multipliers.get(evidence_level[0], 0.7)

        # Patient complexity factor
        complexity = self._assess_patient_complexity(patient_data)
        confidence *= (1 - complexity * 0.1)  # Reduce confidence with complexity

        return round(confidence, 3)

    def _assess_data_quality(self, patient_data: Dict[str, Any]) -> float:
        """Assess quality of patient data"""
        quality_score = 0.5
        quality_checks = 0

        # Demographics completeness
        if patient_data.get("demographics"):
            quality_score += 0.2
        quality_checks += 1

        # Medical history completeness
        if patient_data.get("medical_history"):
            quality_score += 0.2
        quality_checks += 1

        # Current symptoms
        if patient_data.get("symptoms"):
            quality_score += 0.2
        quality_checks += 1

        # Biomarker data
        if patient_data.get("biomarkers"):
            quality_score += 0.2
        quality_checks += 1

        return quality_score / quality_checks if quality_checks > 0 else 0.5

    def _assess_patient_complexity(self, patient_data: Dict[str, Any]) -> float:
        """Assess patient clinical complexity"""
        complexity = 0.0

        # Comorbidity count
        comorbidities = len(patient_data.get("medical_history", []))
        complexity += min(comorbidities * 0.1, 0.5)

        # Medication count
        medications = len(patient_data.get("current_medications", []))
        complexity += min(medications * 0.05, 0.3)

        # Age factor
        age = patient_data.get("demographics", {}).get("age", 50)
        if age > 65:
            complexity += 0.2

        return min(complexity, 1.0)

    def rank_recommendations(self, recommendations: List[Dict[str, Any]],
                           patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank recommendations by clinical priority and benefit"""
        for rec in recommendations:
            # Calculate priority score
            priority_scores = {"high": 3, "medium": 2, "low": 1}
            priority_score = priority_scores.get(rec.get("priority", "medium"), 2)

            # Evidence strength score
            evidence_scores = {"A": 3, "B": 2, "C": 1, "D": 0}
            evidence_level = rec.get("evidence_level", "C")
            evidence_score = evidence_scores.get(evidence_level[0], 1)

            # Patient-specific benefit score
            benefit_score = self._calculate_patient_benefit(rec, patient_data)

            # Combined ranking score
            rec["ranking_score"] = priority_score * 0.4 + evidence_score * 0.3 + benefit_score * 0.3

        # Sort by ranking score (descending)
        ranked = sorted(recommendations, key=lambda x: x["ranking_score"], reverse=True)

        return ranked

    def _calculate_patient_benefit(self, recommendation: Dict[str, Any],
                                 patient_data: Dict[str, Any]) -> float:
        """Calculate patient-specific benefit score"""
        benefit = 1.0  # Base benefit

        rec_type = recommendation.get("type", "")

        # Age-specific benefits
        age = patient_data.get("demographics", {}).get("age", 50)
        if rec_type == "preventive_care" and age > 65:
            benefit += 0.5  # Higher benefit for elderly

        # Comorbidity-specific benefits
        comorbidities = patient_data.get("medical_history", [])
        if rec_type == "treatment" and len(comorbidities) > 2:
            benefit += 0.3  # Higher benefit for complex patients

        return min(benefit, 3.0)

class RiskStratification:
    """Advanced risk stratification algorithms"""

    def assess_risks(self, patient_data: Dict[str, Any],
                    query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        risks = {
            "cardiovascular_risk": self._assess_cardiovascular_risk(patient_data),
            "metabolic_risk": self._assess_metabolic_risk(patient_data),
            "cancer_risk": self._assess_cancer_risk(patient_data),
            "overall_risk_profile": "moderate",
            "risk_trends": [],
            "interventions": []
        }

        # Determine overall risk profile
        high_risk_count = sum(1 for risk in [risks["cardiovascular_risk"],
                                           risks["metabolic_risk"],
                                           risks["cancer_risk"]]
                             if risk.get("level") == "high")

        if high_risk_count >= 2:
            risks["overall_risk_profile"] = "high"
        elif high_risk_count == 1:
            risks["overall_risk_profile"] = "moderate"
        else:
            risks["overall_risk_profile"] = "low"

        # Generate interventions
        risks["interventions"] = self._generate_risk_interventions(risks)

        return risks

    def _assess_cardiovascular_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cardiovascular disease risk"""
        risk_score = 0.1  # Base population risk

        # Age factor
        age = patient_data.get("demographics", {}).get("age", 50)
        if age > 65:
            risk_score += 0.3
        elif age > 45:
            risk_score += 0.15

        # Blood pressure factor
        bp_systolic = None
        for biomarker in patient_data.get("biomarkers", []):
            if biomarker.get("name") == "blood_pressure_systolic":
                bp_systolic = biomarker.get("value")
                break

        if bp_systolic and bp_systolic > 140:
            risk_score += 0.2

        # Cholesterol factor
        cholesterol = None
        for biomarker in patient_data.get("biomarkers", []):
            if biomarker.get("name") == "cholesterol_total":
                cholesterol = biomarker.get("value")
                break

        if cholesterol and cholesterol > 240:
            risk_score += 0.15

        # Medical history
        cv_conditions = ["hypertension", "coronary_artery_disease", "heart_failure"]
        history = patient_data.get("medical_history", [])
        cv_history = sum(1 for condition in history if any(cv in str(condition).lower() for cv in cv_conditions))

        risk_score += cv_history * 0.25

        return {
            "risk_score": min(risk_score, 1.0),
            "level": "high" if risk_score > 0.3 else "moderate" if risk_score > 0.15 else "low",
            "contributing_factors": ["age", "blood_pressure", "cholesterol", "medical_history"],
            "timeframe": "10_year_risk"
        }

    def _assess_metabolic_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess metabolic disease risk"""
        risk_score = 0.08  # Base diabetes risk

        # BMI factor (estimated)
        weight = patient_data.get("demographics", {}).get("weight", 70)
        height = patient_data.get("demographics", {}).get("height", 170)
        if height > 0:
            bmi = weight / ((height / 100) ** 2)
            if bmi > 30:
                risk_score += 0.3
            elif bmi > 25:
                risk_score += 0.15

        # Glucose factor
        glucose = None
        for biomarker in patient_data.get("biomarkers", []):
            if biomarker.get("name") == "glucose":
                glucose = biomarker.get("value")
                break

        if glucose and glucose > 140:
            risk_score += 0.25

        # Family history
        family_history = patient_data.get("family_history", [])
        diabetes_family = sum(1 for condition in family_history
                            if "diabetes" in str(condition).lower())
        risk_score += diabetes_family * 0.15

        return {
            "risk_score": min(risk_score, 1.0),
            "level": "high" if risk_score > 0.25 else "moderate" if risk_score > 0.15 else "low",
            "contributing_factors": ["bmi", "glucose", "family_history"],
            "timeframe": "5_year_risk"
        }

    def _assess_cancer_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cancer risk"""
        risk_score = 0.05  # Base cancer risk

        # Age factor
        age = patient_data.get("demographics", {}).get("age", 50)
        if age > 65:
            risk_score += 0.2
        elif age > 45:
            risk_score += 0.1

        # Family history
        family_history = patient_data.get("family_history", [])
        cancer_family = sum(1 for condition in family_history
                          if any(cancer_type in str(condition).lower()
                                for cancer_type in ["cancer", "carcinoma", "tumor"]))
        risk_score += cancer_family * 0.15

        # Lifestyle factors
        lifestyle = patient_data.get("lifestyle_factors", {})
        if lifestyle.get("smoking", False):
            risk_score += 0.2

        return {
            "risk_score": min(risk_score, 1.0),
            "level": "high" if risk_score > 0.2 else "moderate" if risk_score > 0.1 else "low",
            "contributing_factors": ["age", "family_history", "lifestyle"],
            "timeframe": "lifetime_risk"
        }

    def _generate_risk_interventions(self, risks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk-specific interventions"""
        interventions = []

        if risks["cardiovascular_risk"]["level"] in ["moderate", "high"]:
            interventions.append({
                "target_risk": "cardiovascular",
                "intervention": "Intensive lifestyle modification",
                "components": ["Dietary changes", "Exercise program", "Weight management"],
                "expected_risk_reduction": 0.3,
                "timeline": "3-6 months"
            })

        if risks["metabolic_risk"]["level"] in ["moderate", "high"]:
            interventions.append({
                "target_risk": "metabolic",
                "intervention": "Diabetes prevention program",
                "components": ["Carbohydrate counting", "Regular physical activity", "Weight loss"],
                "expected_risk_reduction": 0.4,
                "timeline": "6-12 months"
            })

        if risks["overall_risk_profile"] == "high":
            interventions.append({
                "target_risk": "overall",
                "intervention": "Comprehensive risk management",
                "components": ["Multidisciplinary care team", "Frequent monitoring", "Patient education"],
                "expected_risk_reduction": 0.5,
                "timeline": "ongoing"
            })

        return interventions

class DifferentialDiagnosis:
    """AI-powered differential diagnosis generation"""

    def generate_differential_diagnosis(self, patient_data: Dict[str, Any],
                                      query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate differential diagnosis"""
        symptoms = patient_data.get("symptoms", [])
        medical_history = patient_data.get("medical_history", [])
        demographics = patient_data.get("demographics", {})

        # Identify chief complaint
        chief_complaint = self._identify_chief_complaint(symptoms)

        # Generate differential diagnosis
        differential = self._create_differential_list(chief_complaint, patient_data)

        # Rank by likelihood
        ranked_differential = self._rank_differential_diagnosis(differential, patient_data)

        return {
            "chief_complaint": chief_complaint,
            "differential_diagnosis": ranked_differential[:5],  # Top 5
            "most_likely_diagnosis": ranked_differential[0] if ranked_differential else None,
            "workup_recommendations": self._generate_workup_recommendations(ranked_differential[:3]),
            "red_flags": self._identify_red_flags(symptoms, medical_history)
        }

    def _identify_chief_complaint(self, symptoms: List[str]) -> str:
        """Identify the chief complaint from symptoms"""
        if not symptoms:
            return "asymptomatic"

        # Map symptoms to chief complaints
        symptom_mapping = {
            "chest_pain": "chest pain",
            "shortness_of_breath": "dyspnea",
            "abdominal_pain": "abdominal pain",
            "headache": "headache",
            "fatigue": "fatigue",
            "nausea": "gastrointestinal symptoms"
        }

        for symptom in symptoms:
            if symptom in symptom_mapping:
                return symptom_mapping[symptom]

        return symptoms[0] if symptoms else "general_symptoms"

    def _create_differential_list(self, chief_complaint: str,
                                patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create initial differential diagnosis list"""
        differential = []

        if chief_complaint == "chest_pain":
            differential.extend([
                {
                    "diagnosis": "acute_coronary_syndrome",
                    "category": "cardiac",
                    "urgency": "high",
                    "key_features": ["retrosternal_pain", "radiation_to_jaw", "diaphoresis"]
                },
                {
                    "diagnosis": "pulmonary_embolism",
                    "category": "pulmonary",
                    "urgency": "high",
                    "key_features": ["sudden_onset", "tachypnea", "risk_factors"]
                },
                {
                    "diagnosis": "gastroesophageal_reflux",
                    "category": "gastrointestinal",
                    "urgency": "low",
                    "key_features": ["burning_pain", "postprandial", "relieved_by_antacids"]
                }
            ])
        elif chief_complaint == "dyspnea":
            differential.extend([
                {
                    "diagnosis": "congestive_heart_failure",
                    "category": "cardiac",
                    "urgency": "high",
                    "key_features": ["orthopnea", "peripheral_edema", "cardiac_history"]
                },
                {
                    "diagnosis": "chronic_obstructive_pulmonary_disease",
                    "category": "pulmonary",
                    "urgency": "medium",
                    "key_features": ["chronic_cough", "smoking_history", "wheezing"]
                }
            ])

        return differential

    def _rank_differential_diagnosis(self, differential: List[Dict[str, Any]],
                                   patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank differential diagnosis by likelihood"""
        for diagnosis in differential:
            # Calculate likelihood score
            likelihood_score = self._calculate_diagnosis_likelihood(diagnosis, patient_data)
            diagnosis["likelihood_score"] = likelihood_score

            # Determine likelihood category
            if likelihood_score > 0.7:
                diagnosis["likelihood"] = "high"
            elif likelihood_score > 0.4:
                diagnosis["likelihood"] = "moderate"
            else:
                diagnosis["likelihood"] = "low"

        # Sort by likelihood score (descending)
        ranked = sorted(differential, key=lambda x: x["likelihood_score"], reverse=True)

        return ranked

    def _calculate_diagnosis_likelihood(self, diagnosis: Dict[str, Any],
                                      patient_data: Dict[str, Any]) -> float:
        """Calculate likelihood score for diagnosis"""
        score = 0.5  # Base score

        # Age factor
        age = patient_data.get("demographics", {}).get("age", 50)
        diagnosis_name = diagnosis.get("diagnosis", "")

        if diagnosis_name == "acute_coronary_syndrome" and age > 45:
            score += 0.2
        elif diagnosis_name == "congestive_heart_failure" and age > 65:
            score += 0.3

        # Symptom matching
        patient_symptoms = set(patient_data.get("symptoms", []))
        diagnosis_features = set(diagnosis.get("key_features", []))

        symptom_match = len(patient_symptoms & diagnosis_features) / len(diagnosis_features) if diagnosis_features else 0
        score += symptom_match * 0.3

        # Medical history factor
        medical_history = patient_data.get("medical_history", [])
        history_relevant = any(keyword in str(medical_history).lower()
                             for keyword in diagnosis_name.split("_"))

        if history_relevant:
            score += 0.2

        return min(score, 1.0)

    def _generate_workup_recommendations(self, top_diagnoses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate diagnostic workup recommendations"""
        workup = []

        for diagnosis in top_diagnoses:
            diagnosis_name = diagnosis.get("diagnosis")

            if diagnosis_name == "acute_coronary_syndrome":
                workup.append({
                    "diagnosis": diagnosis_name,
                    "tests": ["ECG", "troponin", "chest_xray"],
                    "urgency": "immediate",
                    "rationale": "Rule out myocardial infarction"
                })
            elif diagnosis_name == "pulmonary_embolism":
                workup.append({
                    "diagnosis": diagnosis_name,
                    "tests": ["CT_pulmonary_angiogram", "D_dimer"],
                    "urgency": "urgent",
                    "rationale": "High-risk diagnosis requiring prompt evaluation"
                })

        return workup

    def _identify_red_flags(self, symptoms: List[str], medical_history: List[str]) -> List[str]:
        """Identify red flag symptoms requiring urgent attention"""
        red_flags = []

        urgent_symptoms = [
            "chest_pain", "severe_shortness_of_breath", "sudden_weakness",
            "severe_headache", "unconsciousness", "severe_bleeding"
        ]

        for symptom in symptoms:
            if symptom in urgent_symptoms:
                red_flags.append(f"URGENT: {symptom.replace('_', ' ')} requires immediate evaluation")

        # History-based red flags
        if any("cancer" in str(condition).lower() for condition in medical_history):
            red_flags.append("History of malignancy - consider recurrence")

        return red_flags

class TreatmentRecommendation:
    """AI-powered treatment recommendation system"""

    def generate_treatment_recommendations(self, patient_data: Dict[str, Any],
                                         query_analysis: Dict[str, Any],
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate treatment recommendations"""
        recommendations = []

        # Identify target condition
        target_condition = self._identify_target_condition(query_analysis, context)

        # Get treatment guidelines
        guidelines = self._get_treatment_guidelines(target_condition)

        # Generate specific recommendations
        if target_condition == "hypertension":
            recommendations.extend(self._hypertension_treatment(patient_data, guidelines))
        elif target_condition == "diabetes":
            recommendations.extend(self._diabetes_treatment(patient_data, guidelines))
        elif target_condition == "hyperlipidemia":
            recommendations.extend(self._lipid_treatment(patient_data, guidelines))

        return recommendations

    def _identify_target_condition(self, query_analysis: Dict[str, Any],
                                 context: Dict[str, Any]) -> str:
        """Identify the target condition for treatment"""
        medical_terms = query_analysis.get("medical_terms", [])

        condition_mapping = {
            "hypertension": ["hypertension", "high_blood_pressure", "bp"],
            "diabetes": ["diabetes", "glucose", "hba1c"],
            "hyperlipidemia": ["cholesterol", "lipid", "statin"]
        }

        for condition, terms in condition_mapping.items():
            if any(term in " ".join(medical_terms).lower() for term in terms):
                return condition

        return "general"

    def _get_treatment_guidelines(self, condition: str) -> Dict[str, Any]:
        """Get treatment guidelines for condition"""
        guidelines = {
            "hypertension": {
                "first_line": ["ACEI", "ARB", "CCB", "thiazide"],
                "target_bp": "<130/80",
                "lifestyle": ["diet", "exercise", "weight_loss"],
                "combination_therapy": True
            },
            "diabetes": {
                "first_line": ["metformin"],
                "target_hba1c": "<7.0",
                "lifestyle": ["medical_nutrition_therapy", "exercise"],
                "combination_therapy": True
            }
        }

        return guidelines.get(condition, {})

    def _hypertension_treatment(self, patient_data: Dict[str, Any],
                              guidelines: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hypertension treatment recommendations"""
        recommendations = []

        # Lifestyle modifications
        recommendations.append({
            "type": "lifestyle",
            "priority": "high",
            "recommendation": "DASH diet and regular exercise",
            "evidence_level": "A",
            "expected_benefit": "5-10 mmHg reduction"
        })

        # Medication recommendations
        bp_systolic = None
        for biomarker in patient_data.get("biomarkers", []):
            if biomarker.get("name") == "blood_pressure_systolic":
                bp_systolic = biomarker.get("value")
                break

        if bp_systolic and bp_systolic > 140:
            recommendations.append({
                "type": "medication",
                "priority": "high",
                "recommendation": "Initiate monotherapy with ACE inhibitor or ARB",
                "evidence_level": "A",
                "expected_benefit": "10-15 mmHg reduction"
            })

        return recommendations

    def _diabetes_treatment(self, patient_data: Dict[str, Any],
                          guidelines: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate diabetes treatment recommendations"""
        recommendations = []

        # Lifestyle first
        recommendations.append({
            "type": "lifestyle",
            "priority": "high",
            "recommendation": "Medical nutrition therapy and exercise",
            "evidence_level": "A",
            "expected_benefit": "1-2% HbA1c reduction"
        })

        # Medication
        glucose = None
        for biomarker in patient_data.get("biomarkers", []):
            if biomarker.get("name") == "glucose":
                glucose = biomarker.get("value")
                break

        if glucose and glucose > 200:
            recommendations.append({
                "type": "medication",
                "priority": "high",
                "recommendation": "Initiate metformin therapy",
                "evidence_level": "A",
                "expected_benefit": "1-2% HbA1c reduction"
            })

        return recommendations

    def _lipid_treatment(self, patient_data: Dict[str, Any],
                       guidelines: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate lipid treatment recommendations"""
        recommendations = []

        cholesterol = None
        for biomarker in patient_data.get("biomarkers", []):
            if biomarker.get("name") == "cholesterol_total":
                cholesterol = biomarker.get("value")
                break

        if cholesterol and cholesterol > 240:
            recommendations.append({
                "type": "medication",
                "priority": "high",
                "recommendation": "High-intensity statin therapy",
                "evidence_level": "A",
                "expected_benefit": "30-50% LDL reduction"
            })

        return recommendations
