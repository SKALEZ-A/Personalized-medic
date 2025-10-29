"""
Comprehensive Treatment Planning Engine for AI Personalized Medicine Platform
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

from utils.data_structures import TreatmentPlan
from utils.ml_algorithms import MachineLearningAlgorithms

class TreatmentPlanningEngine:
    """AI-powered personalized treatment planning"""

    def __init__(self):
        self.treatment_database = self._initialize_treatment_database()
        self.drug_interaction_checker = DrugInteractionChecker()
        self.dosage_optimizer = DosageOptimizer()
        self.side_effect_predictor = SideEffectPredictor()
        self.treatment_outcome_predictor = TreatmentOutcomePredictor()
        self.planning_queue = queue.Queue()
        self.active_plans = {}
        self.completed_plans = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._start_planning_workers()

    def _start_planning_workers(self):
        """Start background treatment planning workers"""
        for i in range(4):
            worker_thread = threading.Thread(
                target=self._planning_worker,
                daemon=True,
                name=f"TreatmentPlanner-{i+1}"
            )
            worker_thread.start()

    def _planning_worker(self):
        """Background worker for treatment planning"""
        while True:
            try:
                job = self.planning_queue.get(timeout=1)
                if job:
                    self._process_planning_job(job)
                    self.planning_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Treatment planning worker error: {e}")

    def _process_planning_job(self, job: Dict[str, Any]):
        """Process treatment planning job"""
        try:
            job["status"] = "running"
            job["started_at"] = datetime.now()
            self.active_plans[job["plan_id"]] = job

            # Generate treatment plan
            treatment_plan = self.create_treatment_plan(
                diagnosis=job["diagnosis"],
                patient_data=job["patient_data"],
                current_medications=job.get("current_medications", []),
                contraindications=job.get("contraindications", [])
            )

            # Complete job
            job["status"] = "completed"
            job["completed_at"] = datetime.now()
            job["treatment_plan"] = treatment_plan

            # Move to completed
            self.completed_plans[job["plan_id"]] = job
            del self.active_plans[job["plan_id"]]

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = datetime.now()
            self.completed_plans[job["plan_id"]] = job
            if job["plan_id"] in self.active_plans:
                del self.active_plans[job["plan_id"]]

    def _initialize_treatment_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive treatment database"""
        return {
            "diabetes_type_2": {
                "primary_medications": [
                    {
                        "drug": "metformin",
                        "class": "biguanide",
                        "starting_dose": "500mg daily",
                        "max_dose": "2000mg daily",
                        "indications": ["first_line_therapy"],
                        "contraindications": ["renal_impairment", "lactic_acidosis"]
                    },
                    {
                        "drug": "sitagliptin",
                        "class": "dpp4_inhibitor",
                        "starting_dose": "100mg daily",
                        "max_dose": "100mg daily",
                        "indications": ["second_line_therapy"],
                        "contraindications": ["pancreatitis"]
                    },
                    {
                        "drug": "empagliflozin",
                        "class": "sglt2_inhibitor",
                        "starting_dose": "10mg daily",
                        "max_dose": "25mg daily",
                        "indications": ["cardiovascular_benefits"],
                        "contraindications": ["genitourinary_infections"]
                    }
                ],
                "lifestyle_modifications": [
                    "Medical nutrition therapy with carbohydrate counting",
                    "Regular aerobic exercise (150 minutes/week)",
                    "Weight loss target: 5-10% of body weight",
                    "Smoking cessation if applicable"
                ],
                "monitoring_schedule": [
                    {"frequency": "quarterly", "tests": ["HbA1c", "blood_pressure", "weight"]},
                    {"frequency": "annually", "tests": ["lipid_profile", "renal_function", "eye_exam"]}
                ],
                "treatment_goals": [
                    "HbA1c < 7.0%",
                    "Blood pressure < 130/80 mmHg",
                    "LDL cholesterol < 100 mg/dL"
                ]
            },
            "hypertension": {
                "primary_medications": [
                    {
                        "drug": "amlodipine",
                        "class": "calcium_channel_blocker",
                        "starting_dose": "5mg daily",
                        "max_dose": "10mg daily",
                        "indications": ["first_line_therapy"],
                        "contraindications": ["heart_failure"]
                    },
                    {
                        "drug": "lisinopril",
                        "class": "ace_inhibitor",
                        "starting_dose": "10mg daily",
                        "max_dose": "40mg daily",
                        "indications": ["heart_failure", "diabetes"],
                        "contraindications": ["angioedema", "hyperkalemia"]
                    },
                    {
                        "drug": "hydrochlorothiazide",
                        "class": "diuretic",
                        "starting_dose": "12.5mg daily",
                        "max_dose": "50mg daily",
                        "indications": ["elderly_patients"],
                        "contraindications": ["gout", "severe_renal_impairment"]
                    }
                ],
                "lifestyle_modifications": [
                    "DASH diet (Dietary Approaches to Stop Hypertension)",
                    "Sodium restriction (<2300mg daily)",
                    "Regular aerobic exercise",
                    "Weight reduction if overweight",
                    "Limit alcohol consumption"
                ],
                "monitoring_schedule": [
                    {"frequency": "monthly", "tests": ["blood_pressure"]},
                    {"frequency": "quarterly", "tests": ["renal_function", "electrolytes"]},
                    {"frequency": "annually", "tests": ["lipid_profile", "glucose"]}
                ],
                "treatment_goals": [
                    "Blood pressure < 130/80 mmHg",
                    "Maintain renal function",
                    "No medication side effects"
                ]
            },
            "coronary_artery_disease": {
                "primary_medications": [
                    {
                        "drug": "aspirin",
                        "class": "antiplatelet",
                        "starting_dose": "81mg daily",
                        "max_dose": "325mg daily",
                        "indications": ["secondary_prevention"],
                        "contraindications": ["active_bleeding", "aspirin_allergy"]
                    },
                    {
                        "drug": "atorvastatin",
                        "class": "statin",
                        "starting_dose": "20mg daily",
                        "max_dose": "80mg daily",
                        "indications": ["dyslipidemia"],
                        "contraindications": ["liver_disease", "myopathy"]
                    },
                    {
                        "drug": "metoprolol",
                        "class": "beta_blocker",
                        "starting_dose": "25mg twice daily",
                        "max_dose": "100mg twice daily",
                        "indications": ["post_mi", "heart_failure"],
                        "contraindications": ["asthma", "bradycardia"]
                    }
                ],
                "lifestyle_modifications": [
                    "Heart-healthy Mediterranean diet",
                    "Regular aerobic exercise program",
                    "Complete smoking cessation",
                    "Stress management techniques",
                    "Weight management"
                ],
                "monitoring_schedule": [
                    {"frequency": "monthly", "tests": ["blood_pressure", "heart_rate"]},
                    {"frequency": "quarterly", "tests": ["lipid_profile", "liver_function"]},
                    {"frequency": "annually", "tests": ["stress_test", "carotid_ultrasound"]}
                ],
                "treatment_goals": [
                    "LDL cholesterol < 70 mg/dL",
                    "Blood pressure < 130/80 mmHg",
                    "No angina episodes",
                    "Improved exercise tolerance"
                ]
            }
        }

    def create_treatment_plan(self, diagnosis: str, patient_data: Dict[str, Any],
                            current_medications: List[str] = None,
                            contraindications: List[str] = None) -> Dict[str, Any]:
        """Create comprehensive personalized treatment plan"""
        if current_medications is None:
            current_medications = []
        if contraindications is None:
            contraindications = []

        plan_id = f"plan_{diagnosis}_{int(random.random() * 10000)}"

        # Get treatment guidelines
        treatment_info = self.treatment_database.get(diagnosis, {})

        if not treatment_info:
            return {
                "error": f"No treatment guidelines found for diagnosis: {diagnosis}",
                "plan_id": plan_id,
                "status": "incomplete"
            }

        # Select appropriate medications
        selected_medications = self._select_medications(
            treatment_info.get("primary_medications", []),
            patient_data,
            current_medications,
            contraindications
        )

        # Optimize dosages
        optimized_medications = self._optimize_medications(
            selected_medications,
            patient_data
        )

        # Generate lifestyle recommendations
        lifestyle_recommendations = self._personalize_lifestyle_recommendations(
            treatment_info.get("lifestyle_modifications", []),
            patient_data
        )

        # Create monitoring schedule
        monitoring_schedule = self._create_monitoring_schedule(
            treatment_info.get("monitoring_schedule", []),
            patient_data
        )

        # Assess risk factors
        risk_assessment = self._assess_treatment_risks(
            optimized_medications,
            patient_data,
            contraindications
        )

        # Predict treatment outcomes
        outcome_predictions = self.treatment_outcome_predictor.predict_outcomes(
            diagnosis,
            optimized_medications,
            patient_data
        )

        # Generate follow-up plan
        follow_up_plan = self._create_follow_up_plan(
            diagnosis,
            optimized_medications,
            monitoring_schedule
        )

        treatment_plan = {
            "plan_id": plan_id,
            "diagnosis": diagnosis,
            "treatment_goals": treatment_info.get("treatment_goals", []),
            "primary_medications": optimized_medications,
            "alternative_medications": self._suggest_alternatives(
                optimized_medications,
                patient_data,
                contraindications
            ),
            "lifestyle_modifications": lifestyle_recommendations,
            "monitoring_schedule": monitoring_schedule,
            "follow_up_schedule": follow_up_plan,
            "risk_assessment": risk_assessment,
            "outcome_predictions": outcome_predictions,
            "contraindications": contraindications,
            "created_at": datetime.now().isoformat(),
            "estimated_duration_months": self._estimate_treatment_duration(diagnosis),
            "success_probability": outcome_predictions.get("success_probability", 0.7)
        }

        return treatment_plan

    def _select_medications(self, available_medications: List[Dict[str, Any]],
                          patient_data: Dict[str, Any],
                          current_medications: List[str],
                          contraindications: List[str]) -> List[Dict[str, Any]]:
        """Select appropriate medications based on patient profile"""
        selected = []

        # Patient characteristics
        age = patient_data.get("demographics", {}).get("age", 50)
        comorbidities = patient_data.get("medical_history", [])
        genomics = patient_data.get("genomic_data", {})

        for med in available_medications:
            # Check contraindications
            med_contraindications = med.get("contraindications", [])
            if any contra in med_contraindications for contra in contraindications:
                continue

            # Check current medications (avoid duplicates)
            if med["drug"] in current_medications:
                continue

            # Age-based selection
            if age > 65 and "elderly" not in med.get("indications", []):
                # Prefer certain medications for elderly
                if med["drug"] in ["hydrochlorothiazide", "amlodipine"]:
                    selected.append(med)
                    break

            # Genomic considerations
            if genomics and self._check_genomic_compatibility(med, genomics):
                selected.append(med)

            # Default selection
            if len(selected) < 2:  # Select up to 2 primary medications
                selected.append(med)

        return selected[:2]  # Limit to 2 primary medications

    def _check_genomic_compatibility(self, medication: Dict[str, Any],
                                   genomic_data: Dict[str, Any]) -> bool:
        """Check if medication is genomically compatible"""
        drug = medication["drug"]

        # Check for known pharmacogenomic interactions
        pgx_variants = genomic_data.get("pharmacogenomic_variants", [])

        if drug == "metformin":
            # Check for OCT1 variants that affect metformin transport
            return "OCT1" not in [v.get("gene") for v in pgx_variants]
        elif drug == "atorvastatin":
            # Check for SLCO1B1 variants that affect statin metabolism
            return "SLCO1B1" not in [v.get("gene") for v in pgx_variants]
        elif drug == "clopidogrel":
            # Check for CYP2C19 variants that affect clopidogrel activation
            cyp2c19_variants = [v for v in pgx_variants if v.get("gene") == "CYP2C19"]
            return len(cyp2c19_variants) == 0  # Prefer no variants for clopidogrel

        return True  # Default compatible

    def _optimize_medications(self, medications: List[Dict[str, Any]],
                            patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize medication dosages based on patient characteristics"""
        optimized = []

        for med in medications:
            optimized_med = med.copy()

            # Get optimized dosage
            optimized_dosage = self.dosage_optimizer.optimize_dosage(
                med,
                patient_data
            )

            optimized_med["optimized_dosage"] = optimized_dosage
            optimized_med["titration_schedule"] = self._create_titration_schedule(med, optimized_dosage)

            # Predict side effects
            side_effects = self.side_effect_predictor.predict_side_effects(
                med,
                patient_data
            )
            optimized_med["predicted_side_effects"] = side_effects

            optimized.append(optimized_med)

        return optimized

    def _create_titration_schedule(self, medication: Dict[str, Any],
                                 optimized_dosage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create medication titration schedule"""
        starting_dose = medication.get("starting_dose", "50mg daily")
        target_dose = optimized_dosage.get("recommended_dosage", starting_dose)

        # Simple titration schedule
        schedule = []

        if starting_dose != target_dose:
            schedule.append({
                "week": 1,
                "dosage": starting_dose,
                "monitoring": ["side_effects", "tolerability"]
            })
            schedule.append({
                "week": 2,
                "dosage": f"Increased to {target_dose}",
                "monitoring": ["side_effects", "efficacy", "vital_signs"]
            })
        else:
            schedule.append({
                "week": 1,
                "dosage": target_dose,
                "monitoring": ["side_effects", "efficacy"]
            })

        return schedule

    def _personalize_lifestyle_recommendations(self, general_recommendations: List[str],
                                             patient_data: Dict[str, Any]) -> List[str]:
        """Personalize lifestyle recommendations based on patient data"""
        personalized = []

        # Patient characteristics
        demographics = patient_data.get("demographics", {})
        age = demographics.get("age", 50)
        weight = demographics.get("weight", 70)
        height = demographics.get("height", 170)

        # Calculate BMI if available
        if weight and height:
            bmi = weight / ((height / 100) ** 2)
            if bmi > 25:
                personalized.append("Weight management program with gradual weight loss (0.5-1 kg/week)")
            elif bmi < 18.5:
                personalized.append("Nutritional support to achieve healthy weight")

        # Age-specific recommendations
        if age > 65:
            personalized.append("Balance and fall prevention exercises")
            personalized.append("Adequate calcium and vitamin D intake for bone health")
        elif age < 30:
            personalized.append("Long-term cardiovascular health promotion")

        # Add general recommendations
        personalized.extend(general_recommendations)

        # Limit to most relevant recommendations
        return personalized[:6]

    def _create_monitoring_schedule(self, base_schedule: List[Dict[str, Any]],
                                  patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create personalized monitoring schedule"""
        personalized_schedule = []

        # Patient risk factors
        age = patient_data.get("demographics", {}).get("age", 50)
        comorbidities = len(patient_data.get("medical_history", []))

        for schedule_item in base_schedule:
            item = schedule_item.copy()

            # Adjust frequency based on risk
            if age > 65 or comorbidities > 2:
                if item["frequency"] == "annually":
                    item["frequency"] = "biannually"
                elif item["frequency"] == "quarterly":
                    item["frequency"] = "monthly"

            personalized_schedule.append(item)

        return personalized_schedule

    def _assess_treatment_risks(self, medications: List[Dict[str, Any]],
                              patient_data: Dict[str, Any],
                              contraindications: List[str]) -> Dict[str, Any]:
        """Assess risks associated with treatment plan"""
        risks = {
            "drug_interactions": [],
            "side_effects": [],
            "contraindications": [],
            "monitoring_requirements": [],
            "overall_risk_level": "low"
        }

        # Check drug interactions
        if len(medications) > 1:
            interactions = self.drug_interaction_checker.check_interactions(
                [med["drug"] for med in medications]
            )
            risks["drug_interactions"] = interactions

        # Assess side effect risks
        for med in medications:
            side_effects = med.get("predicted_side_effects", [])
            risks["side_effects"].extend(side_effects)

        # Check contraindications
        for med in medications:
            med_contraindications = med.get("contraindications", [])
            conflicts = [contra for contra in med_contraindications if contra in contraindications]
            if conflicts:
                risks["contraindications"].extend(conflicts)

        # Determine overall risk level
        risk_factors = (len(risks["drug_interactions"]) +
                       len(risks["side_effects"]) +
                       len(risks["contraindications"]))

        if risk_factors > 5:
            risks["overall_risk_level"] = "high"
        elif risk_factors > 2:
            risks["overall_risk_level"] = "moderate"
        else:
            risks["overall_risk_level"] = "low"

        # Monitoring requirements
        if risks["overall_risk_level"] in ["moderate", "high"]:
            risks["monitoring_requirements"].extend([
                "Frequent vital sign monitoring",
                "Regular laboratory testing",
                "Close follow-up appointments"
            ])

        return risks

    def _suggest_alternatives(self, current_medications: List[Dict[str, Any]],
                            patient_data: Dict[str, Any],
                            contraindications: List[str]) -> List[Dict[str, Any]]:
        """Suggest alternative medications"""
        alternatives = []

        # For each current medication, suggest alternatives
        for med in current_medications:
            med_class = med.get("class")

            # Find alternatives in same class
            treatment_info = self.treatment_database.get("diabetes_type_2", {})  # Example
            all_meds = treatment_info.get("primary_medications", [])

            class_alternatives = [
                alt for alt in all_meds
                if alt.get("class") == med_class and alt["drug"] != med["drug"]
            ]

            for alt in class_alternatives[:2]:  # Up to 2 alternatives per medication
                # Check compatibility
                if not any contra in alt.get("contraindications", []) for contra in contraindications:
                    alternatives.append({
                        "alternative_for": med["drug"],
                        "drug": alt["drug"],
                        "class": alt["class"],
                        "reason": f"Alternative in {med_class} class",
                        "starting_dose": alt.get("starting_dose")
                    })

        return alternatives

    def _create_follow_up_plan(self, diagnosis: str, medications: List[Dict[str, Any]],
                             monitoring_schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create follow-up plan"""
        follow_up = []

        # Initial follow-up
        follow_up.append({
            "timeframe": "1 week",
            "purpose": "Assess medication tolerance and initial response",
            "activities": ["Review side effects", "Check vital signs", "Medication counseling"]
        })

        # Short-term follow-up
        follow_up.append({
            "timeframe": "1 month",
            "purpose": "Evaluate treatment efficacy and adjust as needed",
            "activities": ["Review symptoms", "Check laboratory results", "Dose adjustment if needed"]
        })

        # Regular follow-up based on monitoring schedule
        for monitor_item in monitoring_schedule:
            frequency = monitor_item["frequency"]
            if frequency == "monthly":
                follow_up.append({
                    "timeframe": "Monthly",
                    "purpose": "Routine monitoring and treatment optimization",
                    "activities": ["Vital signs", "Symptom review", f"Tests: {', '.join(monitor_item['tests'])}"]
                })
            elif frequency in ["quarterly", "biannually"]:
                follow_up.append({
                    "timeframe": frequency.title(),
                    "purpose": "Comprehensive evaluation",
                    "activities": ["Complete physical examination", f"Laboratory tests: {', '.join(monitor_item['tests'])}"]
                })

        return follow_up

    def _estimate_treatment_duration(self, diagnosis: str) -> int:
        """Estimate treatment duration in months"""
        duration_estimates = {
            "diabetes_type_2": 120,  # 10 years
            "hypertension": 240,     # 20 years (chronic)
            "coronary_artery_disease": 240,  # 20 years
            "acute_infection": 1,    # 1 month
            "mental_health": 12      # 1 year
        }

        return duration_estimates.get(diagnosis, 12)

    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get status of treatment plan"""
        if plan_id in self.active_plans:
            job = self.active_plans[plan_id]
            return {
                "plan_id": plan_id,
                "status": job["status"],
                "created_at": job["created_at"].isoformat(),
                "started_at": job.get("started_at", "").isoformat() if job.get("started_at") else None
            }
        elif plan_id in self.completed_plans:
            job = self.completed_plans[plan_id]
            return {
                "plan_id": plan_id,
                "status": job["status"],
                "completed_at": job["completed_at"].isoformat(),
                "available": "treatment_plan" in job,
                "error": job.get("error")
            }
        else:
            return {"error": "Plan not found"}

    def get_plan_results(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get treatment plan results"""
        if plan_id in self.completed_plans:
            job = self.completed_plans[plan_id]
            if job["status"] == "completed" and "treatment_plan" in job:
                return job["treatment_plan"]
        return None

class DrugInteractionChecker:
    """Comprehensive drug interaction checking"""

    def __init__(self):
        self.interaction_database = self._initialize_interaction_database()

    def _initialize_interaction_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize drug interaction database"""
        return {
            "aspirin_statin": {
                "severity": "moderate",
                "description": "Increased risk of muscle-related side effects",
                "recommendation": "Monitor for myopathy symptoms"
            },
            "ace_inhibitor_diuretic": {
                "severity": "moderate",
                "description": "Enhanced blood pressure lowering effect",
                "recommendation": "Monitor blood pressure closely, watch for hypotension"
            },
            "metformin_insulin": {
                "severity": "moderate",
                "description": "Increased risk of hypoglycemia",
                "recommendation": "Monitor blood glucose frequently"
            },
            "warfarin_aspirin": {
                "severity": "major",
                "description": "Increased bleeding risk",
                "recommendation": "Use with extreme caution, frequent INR monitoring"
            }
        }

    def check_interactions(self, drug_list: List[str]) -> List[Dict[str, Any]]:
        """Check for interactions between drugs"""
        interactions = []

        # Check all pairs
        for i, drug1 in enumerate(drug_list):
            for j, drug2 in enumerate(drug_list):
                if i < j:  # Avoid duplicate checks
                    interaction_key = f"{drug1}_{drug2}"
                    alt_key = f"{drug2}_{drug1}"

                    interaction = (self.interaction_database.get(interaction_key) or
                                 self.interaction_database.get(alt_key))

                    if interaction:
                        interactions.append({
                            "drugs": [drug1, drug2],
                            "severity": interaction["severity"],
                            "description": interaction["description"],
                            "recommendation": interaction["recommendation"]
                        })

        return interactions

class DosageOptimizer:
    """AI-powered dosage optimization"""

    def __init__(self):
        self.ml_algorithms = MachineLearningAlgorithms()

    def optimize_dosage(self, medication: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize medication dosage for patient"""
        base_dosage = medication.get("starting_dose", "50mg daily")
        max_dosage = medication.get("max_dose", "100mg daily")

        # Extract patient factors
        age = patient_data.get("demographics", {}).get("age", 50)
        weight = patient_data.get("demographics", {}).get("weight", 70)
        renal_function = self._assess_renal_function(patient_data)
        liver_function = self._assess_liver_function(patient_data)

        # Genomic factors
        genomic_adjustment = self._calculate_genomic_adjustment(medication, patient_data)

        # Calculate optimal dosage
        optimal_dosage = self.ml_algorithms.optimize_treatment_dosage(patient_data, medication["drug"])

        return {
            "recommended_dosage": optimal_dosage["recommended_dosage"],
            "base_dosage": base_dosage,
            "max_dosage": max_dosage,
            "adjustment_factors": {
                "age": f"{age} years",
                "weight": f"{weight} kg",
                "renal_function": renal_function,
                "liver_function": liver_function,
                "genomic_factors": genomic_adjustment
            },
            "monitoring_required": True,
            "titration_needed": optimal_dosage["recommended_dosage"] != base_dosage
        }

    def _assess_renal_function(self, patient_data: Dict[str, Any]) -> str:
        """Assess renal function"""
        creatinine = None
        for biomarker in patient_data.get("biomarkers", []):
            if biomarker.get("name") == "creatinine":
                creatinine = biomarker.get("value")
                break

        if creatinine:
            if creatinine < 0.8:
                return "normal"
            elif creatinine < 1.5:
                return "mild_impairment"
            elif creatinine < 3.0:
                return "moderate_impairment"
            else:
                return "severe_impairment"

        return "unknown"

    def _assess_liver_function(self, patient_data: Dict[str, Any]) -> str:
        """Assess liver function"""
        alt = None
        for biomarker in patient_data.get("biomarkers", []):
            if biomarker.get("name") == "alt":
                alt = biomarker.get("value")
                break

        if alt:
            if alt < 40:
                return "normal"
            elif alt < 80:
                return "mild_elevation"
            elif alt < 200:
                return "moderate_elevation"
            else:
                return "severe_elevation"

        return "unknown"

    def _calculate_genomic_adjustment(self, medication: Dict[str, Any], patient_data: Dict[str, Any]) -> str:
        """Calculate genomic adjustment factor"""
        genomics = patient_data.get("genomic_data", {})
        drug = medication["drug"]

        # Check for known pharmacogenomic effects
        if drug == "warfarin":
            if "VKORC1" in str(genomics):
                return "reduced_dosage"
        elif drug == "clopidogrel":
            if "CYP2C19" in str(genomics):
                return "alternative_therapy"
        elif drug in ["metformin", "atorvastatin"]:
            if "SLCO1B1" in str(genomics):
                return "reduced_dosage"

        return "standard_dosage"

class SideEffectPredictor:
    """AI-powered side effect prediction"""

    def __init__(self):
        self.side_effect_database = self._initialize_side_effect_database()

    def _initialize_side_effect_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize side effect database"""
        return {
            "metformin": [
                {"effect": "gastrointestinal_distress", "probability": 0.3, "severity": "mild"},
                {"effect": "lactic_acidosis", "probability": 0.01, "severity": "severe"},
                {"effect": "vitamin_b12_deficiency", "probability": 0.1, "severity": "moderate"}
            ],
            "atorvastatin": [
                {"effect": "muscle_pain", "probability": 0.15, "severity": "moderate"},
                {"effect": "liver_enzyme_elevation", "probability": 0.1, "severity": "mild"},
                {"effect": "diabetes_risk", "probability": 0.05, "severity": "moderate"}
            ],
            "amlodipine": [
                {"effect": "peripheral_edema", "probability": 0.2, "severity": "mild"},
                {"effect": "dizziness", "probability": 0.1, "severity": "mild"},
                {"effect": "flushing", "probability": 0.05, "severity": "mild"}
            ]
        }

    def predict_side_effects(self, medication: Dict[str, Any], patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict side effects for medication"""
        drug = medication["drug"]
        side_effects = self.side_effect_database.get(drug, [])

        predicted_effects = []

        for effect in side_effects:
            # Adjust probability based on patient factors
            adjusted_probability = self._adjust_probability(effect, patient_data)

            if adjusted_probability > 0.05:  # Only include significant risks
                predicted_effects.append({
                    "side_effect": effect["effect"],
                    "probability": adjusted_probability,
                    "severity": effect["severity"],
                    "management": self._get_management_strategy(effect["effect"]),
                    "monitoring": self._get_monitoring_recommendation(effect["effect"])
                })

        return sorted(predicted_effects, key=lambda x: x["probability"], reverse=True)

    def _adjust_probability(self, effect: Dict[str, Any], patient_data: Dict[str, Any]) -> float:
        """Adjust side effect probability based on patient factors"""
        probability = effect["probability"]

        # Age adjustment
        age = patient_data.get("demographics", {}).get("age", 50)
        if age > 65:
            probability *= 1.5  # Increased risk in elderly

        # Renal function adjustment
        renal_function = patient_data.get("biomarkers", [])
        creatinine = None
        for biomarker in renal_function:
            if biomarker.get("name") == "creatinine":
                creatinine = biomarker.get("value")
                break

        if creatinine and creatinine > 1.5:
            if effect["effect"] in ["lactic_acidosis", "peripheral_edema"]:
                probability *= 2.0  # Significantly increased risk

        # Genomic adjustment
        genomics = patient_data.get("genomic_data", {})
        if "poor_metabolizer" in str(genomics):
            probability *= 1.3  # Increased risk for poor metabolizers

        return min(probability, 0.9)  # Cap at 90%

    def _get_management_strategy(self, side_effect: str) -> str:
        """Get management strategy for side effect"""
        strategies = {
            "gastrointestinal_distress": "Take with food, start with low dose",
            "muscle_pain": "Monitor CK levels, consider dose reduction",
            "peripheral_edema": "Monitor weight, consider dose reduction",
            "dizziness": "Change positions slowly, monitor blood pressure",
            "lactic_acidosis": "Monitor symptoms, discontinue if suspected"
        }

        return strategies.get(side_effect, "Monitor closely and consult healthcare provider")

    def _get_monitoring_recommendation(self, side_effect: str) -> str:
        """Get monitoring recommendation for side effect"""
        monitoring = {
            "gastrointestinal_distress": "Monitor symptoms weekly",
            "muscle_pain": "Monitor CK levels monthly",
            "peripheral_edema": "Monitor weight weekly",
            "dizziness": "Monitor blood pressure with position changes",
            "lactic_acidosis": "Monitor lactate levels if symptoms present"
        }

        return monitoring.get(side_effect, "Regular symptom monitoring")

class TreatmentOutcomePredictor:
    """AI-powered treatment outcome prediction"""

    def __init__(self):
        self.ml_algorithms = MachineLearningAlgorithms()

    def predict_outcomes(self, diagnosis: str, medications: List[Dict[str, Any]],
                        patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict treatment outcomes"""
        # Use ML algorithms for prediction
        outcome_prediction = self.ml_algorithms.predict_treatment_outcome(
            diagnosis, medications, patient_data
        )

        # Calculate success probability
        success_probability = self._calculate_success_probability(
            diagnosis, medications, patient_data
        )

        # Predict timeline
        timeline = self._predict_treatment_timeline(diagnosis, medications)

        # Predict potential complications
        complications = self._predict_complications(medications, patient_data)

        return {
            "success_probability": success_probability,
            "predicted_outcomes": outcome_prediction,
            "treatment_timeline": timeline,
            "potential_complications": complications,
            "quality_of_life_impact": self._predict_quality_of_life_impact(medications),
            "adherence_prediction": self._predict_adherence_probability(patient_data),
            "cost_effectiveness": self._assess_cost_effectiveness(medications, success_probability)
        }

    def _calculate_success_probability(self, diagnosis: str, medications: List[Dict[str, Any]],
                                    patient_data: Dict[str, Any]) -> float:
        """Calculate treatment success probability"""
        base_probability = 0.7  # Base success rate

        # Diagnosis-specific adjustments
        diagnosis_multipliers = {
            "diabetes_type_2": 0.8,
            "hypertension": 0.85,
            "coronary_artery_disease": 0.75,
            "depression": 0.7,
            "asthma": 0.8
        }

        base_probability *= diagnosis_multipliers.get(diagnosis, 1.0)

        # Medication factors
        if len(medications) > 2:
            base_probability *= 0.9  # Slightly reduced with polypharmacy

        # Patient factors
        age = patient_data.get("demographics", {}).get("age", 50)
        if age > 65:
            base_probability *= 0.95  # Slightly reduced in elderly

        comorbidities = len(patient_data.get("medical_history", []))
        base_probability *= max(0.7, 1 - comorbidities * 0.05)  # Reduced with comorbidities

        return round(base_probability, 3)

    def _predict_treatment_timeline(self, diagnosis: str, medications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict treatment timeline"""
        timelines = {
            "diabetes_type_2": {
                "initial_response": "2-4 weeks",
                "peak_effect": "3-6 months",
                "maintenance": "ongoing"
            },
            "hypertension": {
                "initial_response": "1-2 weeks",
                "peak_effect": "4-6 weeks",
                "maintenance": "ongoing"
            },
            "coronary_artery_disease": {
                "initial_response": "1-4 weeks",
                "peak_effect": "6-12 weeks",
                "maintenance": "ongoing"
            }
        }

        return timelines.get(diagnosis, {
            "initial_response": "2-4 weeks",
            "peak_effect": "1-3 months",
            "maintenance": "ongoing"
        })

    def _predict_complications(self, medications: List[Dict[str, Any]],
                            patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict potential complications"""
        complications = []

        # Check for drug-specific complications
        for med in medications:
            drug = med["drug"]

            if drug == "metformin":
                complications.append({
                    "complication": "vitamin_b12_deficiency",
                    "probability": 0.1,
                    "prevention": "Monitor B12 levels annually"
                })
            elif drug == "atorvastatin":
                complications.append({
                    "complication": "myopathy",
                    "probability": 0.05,
                    "prevention": "Monitor CK levels, report muscle symptoms"
                })

        # Age-related complications
        age = patient_data.get("demographics", {}).get("age", 50)
        if age > 65:
            complications.append({
                "complication": "falls_due_to_dizziness",
                "probability": 0.08,
                "prevention": "Monitor for orthostatic hypotension"
            })

        return complications

    def _predict_quality_of_life_impact(self, medications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict quality of life impact"""
        # Simplified prediction
        side_effect_burden = sum(len(med.get("predicted_side_effects", [])) for med in medications)

        if side_effect_burden > 3:
            impact = "moderate_negative"
            description = "Some quality of life impact from side effects"
        elif side_effect_burden > 1:
            impact = "mild_negative"
            description = "Minimal quality of life impact"
        else:
            impact = "neutral_to_positive"
            description = "Generally well tolerated"

        return {
            "overall_impact": impact,
            "description": description,
            "factors": ["medication_side_effects", "dosing_frequency", "monitoring_requirements"]
        }

    def _predict_adherence_probability(self, patient_data: Dict[str, Any]) -> float:
        """Predict medication adherence probability"""
        adherence_probability = 0.75  # Base adherence

        # Age factor
        age = patient_data.get("demographics", {}).get("age", 50)
        if age > 65:
            adherence_probability += 0.1  # Better adherence in elderly
        elif age < 30:
            adherence_probability -= 0.1  # Lower adherence in young adults

        # Comorbidity factor
        comorbidities = len(patient_data.get("medical_history", []))
        adherence_probability -= comorbidities * 0.02  # Reduced with more conditions

        # Socioeconomic factors (simplified)
        social_factors = patient_data.get("social_determinants", {})
        if social_factors.get("insurance_coverage"):
            adherence_probability += 0.05

        return round(max(0.4, min(0.95, adherence_probability)), 3)

    def _assess_cost_effectiveness(self, medications: List[Dict[str, Any]], success_probability: float) -> Dict[str, Any]:
        """Assess cost-effectiveness of treatment"""
        # Simplified cost calculation
        monthly_cost = sum(50 for _ in medications)  # $50 per medication estimate

        # Quality-adjusted life years (simplified)
        qaly_gain = success_probability * 2  # Assume 2 QALY gain with successful treatment

        # Cost per QALY
        cost_per_qaly = (monthly_cost * 12) / qaly_gain if qaly_gain > 0 else float('inf')

        # Cost-effectiveness threshold typically $50,000-$100,000 per QALY
        is_cost_effective = cost_per_qaly < 100000

        return {
            "monthly_cost_estimate": monthly_cost,
            "annual_cost_estimate": monthly_cost * 12,
            "qaly_gain": qaly_gain,
            "cost_per_qaly": cost_per_qaly,
            "is_cost_effective": is_cost_effective,
            "cost_effectiveness_rating": "high" if cost_per_qaly < 50000 else "moderate" if cost_per_qaly < 100000 else "low"
        }
