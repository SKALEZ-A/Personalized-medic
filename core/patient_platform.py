"""
Comprehensive Patient Engagement Platform for AI Personalized Medicine
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

class PatientEngagementPlatform:
    """Comprehensive patient engagement and education platform"""

    def __init__(self):
        self.patient_profiles = {}
        self.health_dashboards = {}
        self.education_content = self._initialize_education_content()
        self.virtual_assistant = VirtualHealthAssistant()
        self.mobile_companion = MobileHealthCompanion()
        self.gamification_engine = HealthGamificationEngine()
        self.notification_system = NotificationSystem()

    def generate_initial_insights(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial health insights for new patients"""
        insights = {
            "welcome_message": "Welcome to your personalized health journey!",
            "initial_assessment": self._perform_initial_assessment(patient_data),
            "personalized_goals": self._generate_personalized_goals(patient_data),
            "education_recommendations": self._recommend_initial_education(patient_data),
            "next_steps": [
                "Complete your health profile",
                "Set up health monitoring devices",
                "Schedule initial consultation",
                "Start daily health tracking"
            ]
        }

        return insights

    def _perform_initial_assessment(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform initial health assessment"""
        assessment = {
            "health_status": "unknown",
            "risk_level": "unknown",
            "key_focus_areas": [],
            "immediate_actions": []
        }

        # Analyze demographics
        demographics = patient_data.get("demographics", {})
        age = demographics.get("age", 50)

        if age > 65:
            assessment["key_focus_areas"].append("geriatric_care")
            assessment["immediate_actions"].append("Schedule comprehensive geriatric assessment")
        elif age < 30:
            assessment["key_focus_areas"].append("preventive_care")
            assessment["immediate_actions"].append("Establish primary care relationship")

        # Analyze medical history
        medical_history = patient_data.get("medical_history", [])
        if len(medical_history) > 2:
            assessment["key_focus_areas"].append("chronic_disease_management")
            assessment["risk_level"] = "moderate"
        elif len(medical_history) > 0:
            assessment["key_focus_areas"].append("health_maintenance")
        else:
            assessment["health_status"] = "good"
            assessment["key_focus_areas"].append("health_optimization")

        return assessment

    def _generate_personalized_goals(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized health goals"""
        goals = []

        # Weight management goals
        demographics = patient_data.get("demographics", {})
        weight = demographics.get("weight")
        height = demographics.get("height")

        if weight and height:
            bmi = weight / ((height / 100) ** 2)
            if bmi > 25:
                goals.append({
                    "category": "weight_management",
                    "goal": "Achieve healthy BMI",
                    "target": "BMI < 25",
                    "timeline": "6 months",
                    "priority": "high"
                })

        # Activity goals
        lifestyle = patient_data.get("lifestyle_factors", {})
        exercise_freq = lifestyle.get("exercise_frequency", 0)

        if exercise_freq < 3:
            goals.append({
                "category": "physical_activity",
                "goal": "Regular exercise routine",
                "target": "150 minutes moderate exercise per week",
                "timeline": "3 months",
                "priority": "medium"
            })

        # Monitoring goals
        goals.append({
            "category": "health_monitoring",
            "goal": "Consistent vital signs tracking",
            "target": "Daily blood pressure and weight monitoring",
            "timeline": "1 month",
            "priority": "high"
        })

        return goals

    def _recommend_initial_education(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend initial educational content"""
        recommendations = []

        # Basic health literacy
        recommendations.append({
            "topic": "understanding_your_health_data",
            "format": "interactive_tutorial",
            "duration": "15 minutes",
            "priority": "high"
        })

        # Condition-specific education
        medical_history = patient_data.get("medical_history", [])
        if any("diabetes" in str(condition).lower() for condition in medical_history):
            recommendations.append({
                "topic": "diabetes_self_management",
                "format": "video_series",
                "duration": "30 minutes",
                "priority": "high"
            })

        if any("hypertension" in str(condition).lower() for condition in medical_history):
            recommendations.append({
                "topic": "blood_pressure_management",
                "format": "infographic",
                "duration": "10 minutes",
                "priority": "high"
            })

        return recommendations

    def generate_dashboard(self, patient_id: str, patient_data: Dict[str, Any],
                         health_records: List[Dict[str, Any]],
                         genomic_data: Dict[str, Any] = None,
                         treatment_plans: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive patient health dashboard"""

        dashboard = {
            "patient_id": patient_id,
            "last_updated": datetime.now().isoformat(),
            "health_overview": self._generate_health_overview(patient_data, health_records),
            "vital_signs_summary": self._summarize_vital_signs(health_records),
            "biomarker_trends": self._analyze_biomarker_trends(health_records),
            "medication_adherence": self._calculate_medication_adherence(patient_data),
            "goal_progress": self._track_goal_progress(patient_id),
            "educational_progress": self._track_educational_progress(patient_id),
            "risk_assessment": self._generate_risk_assessment(genomic_data, health_records),
            "treatment_summary": self._summarize_treatment_plans(treatment_plans),
            "notifications": self.notification_system.get_patient_notifications(patient_id),
            "gamification_stats": self.gamification_engine.get_patient_stats(patient_id),
            "next_appointments": self._get_upcoming_appointments(patient_id)
        }

        return dashboard

    def _generate_health_overview(self, patient_data: Dict[str, Any],
                                health_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate health overview section"""
        overview = {
            "overall_health_score": 75,  # Placeholder calculation
            "health_trend": "stable",
            "active_conditions": len(patient_data.get("medical_history", [])),
            "current_medications": len(patient_data.get("current_medications", [])),
            "recent_activity": "normal",
            "alerts_count": 2
        }

        # Calculate health score based on recent data
        if health_records:
            recent_records = health_records[-7:]  # Last 7 days
            overview["overall_health_score"] = self._calculate_health_score(recent_records)

        return overview

    def _calculate_health_score(self, health_records: List[Dict[str, Any]]) -> int:
        """Calculate overall health score from recent records"""
        score = 70  # Base score

        # Analyze vital signs
        vital_scores = []
        for record in health_records:
            vital_signs = record.get("vital_signs", {})
            hr = vital_signs.get("heart_rate", {}).get("value")
            bp = vital_signs.get("blood_pressure", {})

            if hr and 60 <= hr <= 100:
                vital_scores.append(1)
            else:
                vital_scores.append(0)

        if vital_scores:
            vital_avg = sum(vital_scores) / len(vital_scores)
            score += int(vital_avg * 20)

        return min(100, max(0, score))

    def _summarize_vital_signs(self, health_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize vital signs from recent records"""
        if not health_records:
            return {"status": "no_data"}

        recent_record = health_records[-1]
        vital_signs = recent_record.get("vital_signs", {})

        summary = {
            "last_reading": recent_record.get("timestamp"),
            "heart_rate": vital_signs.get("heart_rate", {}),
            "blood_pressure": vital_signs.get("blood_pressure", {}),
            "temperature": vital_signs.get("temperature", {}),
            "oxygen_saturation": vital_signs.get("oxygen_saturation", {}),
            "trend": "stable"
        }

        return summary

    def _analyze_biomarker_trends(self, health_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze biomarker trends"""
        trends = {}

        # Group biomarkers by type
        biomarker_data = {}
        for record in health_records[-30:]:  # Last 30 records
            biomarkers = record.get("biomarkers", [])
            for biomarker in biomarkers:
                name = biomarker.get("name")
                value = biomarker.get("value")

                if name not in biomarker_data:
                    biomarker_data[name] = []
                biomarker_data[name].append({
                    "value": value,
                    "timestamp": record.get("timestamp")
                })

        # Analyze trends for each biomarker
        for name, data_points in biomarker_data.items():
            if len(data_points) >= 2:
                values = [point["value"] for point in data_points]
                trend = self._calculate_trend(values)
                current_value = values[-1]
                normal_range = self._get_biomarker_normal_range(name)

                trends[name] = {
                    "current_value": current_value,
                    "normal_range": normal_range,
                    "trend": trend,
                    "status": "normal" if normal_range[0] <= current_value <= normal_range[1] else "abnormal"
                }

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"

        recent_avg = sum(values[-3:]) / min(3, len(values))
        earlier_avg = sum(values[:-3]) / max(1, len(values) - 3) if len(values) > 3 else recent_avg

        if recent_avg > earlier_avg * 1.05:
            return "increasing"
        elif recent_avg < earlier_avg * 0.95:
            return "decreasing"
        else:
            return "stable"

    def _get_biomarker_normal_range(self, biomarker: str) -> Tuple[float, float]:
        """Get normal range for biomarker"""
        ranges = {
            "glucose": (70, 140),
            "cholesterol_total": (0, 200),
            "creatinine": (0.6, 1.2),
            "alt": (7, 56)
        }
        return ranges.get(biomarker, (0, 100))

    def _calculate_medication_adherence(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate medication adherence metrics"""
        medications = patient_data.get("current_medications", [])

        adherence = {
            "overall_adherence": 0.85,  # Placeholder
            "medications": [],
            "last_refill": "2024-01-15",
            "next_refill_due": "2024-02-15"
        }

        for med in medications:
            adherence["medications"].append({
                "name": med,
                "adherence_rate": random.uniform(0.7, 0.95),
                "doses_taken": random.randint(20, 30),
                "doses_prescribed": 30
            })

        return adherence

    def _track_goal_progress(self, patient_id: str) -> List[Dict[str, Any]]:
        """Track progress on health goals"""
        # Placeholder goal progress
        goals = [
            {
                "goal": "Weight management",
                "target": "Lose 10 lbs",
                "current_progress": 6,
                "target_value": 10,
                "percentage_complete": 60,
                "status": "on_track"
            },
            {
                "goal": "Exercise routine",
                "target": "150 min/week",
                "current_progress": 120,
                "target_value": 150,
                "percentage_complete": 80,
                "status": "on_track"
            }
        ]

        return goals

    def _track_educational_progress(self, patient_id: str) -> Dict[str, Any]:
        """Track educational content progress"""
        progress = {
            "completed_modules": 5,
            "total_modules": 12,
            "current_streak": 7,  # Days
            "favorite_topics": ["diabetes_management", "nutrition"],
            "next_recommended": "exercise_safety"
        }

        return progress

    def _generate_risk_assessment(self, genomic_data: Dict[str, Any] = None,
                                health_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate personalized risk assessment"""
        assessment = {
            "overall_risk_level": "moderate",
            "key_risks": [],
            "preventive_actions": [],
            "last_assessment": datetime.now().isoformat()
        }

        if genomic_data:
            # Add genomic risks
            variants = genomic_data.get("variants", [])
            if len(variants) > 5:
                assessment["key_risks"].append("Genetic predisposition to multiple conditions")

        if health_records:
            # Analyze recent health data for risks
            recent_bp = []
            for record in health_records[-7:]:
                bp = record.get("vital_signs", {}).get("blood_pressure", {})
                if bp.get("systolic"):
                    recent_bp.append(bp["systolic"])

            if recent_bp and sum(recent_bp) / len(recent_bp) > 140:
                assessment["key_risks"].append("Hypertension")
                assessment["preventive_actions"].append("Regular blood pressure monitoring")

        return assessment

    def _summarize_treatment_plans(self, treatment_plans: Dict[str, Any] = None) -> Dict[str, Any]:
        """Summarize active treatment plans"""
        if not treatment_plans:
            return {"status": "no_active_plans"}

        summary = {
            "active_plans": 1,
            "primary_diagnosis": treatment_plans.get("diagnosis", "Unknown"),
            "medications_count": len(treatment_plans.get("primary_medications", [])),
            "next_follow_up": treatment_plans.get("follow_up_schedule", [{}])[0].get("timeframe", "Not scheduled"),
            "adherence_score": 0.82
        }

        return summary

    def _get_upcoming_appointments(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get upcoming appointments"""
        appointments = [
            {
                "date": "2024-02-01",
                "time": "10:00 AM",
                "type": "Primary Care Visit",
                "provider": "Dr. Smith",
                "purpose": "Routine checkup"
            },
            {
                "date": "2024-02-15",
                "time": "2:00 PM",
                "type": "Cardiology Consultation",
                "provider": "Dr. Johnson",
                "purpose": "Blood pressure management"
            }
        ]

        return appointments

class VirtualHealthAssistant:
    """AI-powered virtual health assistant"""

    def __init__(self):
        self.conversation_history = {}
        self.health_knowledge_base = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> Dict[str, Dict[str, Any]]:
        """Initialize health knowledge base"""
        return {
            "medication_questions": {
                "patterns": ["when to take", "how to take", "side effects", "interactions"],
                "responses": {
                    "timing": "Take medications as prescribed, usually with meals unless specified otherwise",
                    "administration": "Follow the specific instructions for each medication",
                    "side_effects": "Common side effects are usually mild and temporary",
                    "interactions": "Always inform your doctor about all medications and supplements"
                }
            },
            "symptom_assessment": {
                "urgent_symptoms": ["chest_pain", "shortness_of_breath", "severe_headache"],
                "response": "This symptom requires immediate medical attention. Please seek emergency care."
            },
            "lifestyle_advice": {
                "exercise": "Aim for 150 minutes of moderate aerobic activity per week",
                "nutrition": "Focus on whole foods, vegetables, fruits, and lean proteins",
                "stress_management": "Practice deep breathing, meditation, or yoga regularly"
            }
        }

    def process_query(self, patient_id: str, query: str) -> Dict[str, Any]:
        """Process user query and generate response"""
        # Analyze query intent
        intent = self._analyze_intent(query)

        # Generate response based on intent
        if intent == "medication_question":
            response = self._handle_medication_question(query)
        elif intent == "symptom_check":
            response = self._handle_symptom_check(query)
        elif intent == "lifestyle_advice":
            response = self._handle_lifestyle_advice(query)
        else:
            response = self._generate_general_response(query)

        # Store conversation
        if patient_id not in self.conversation_history:
            self.conversation_history[patient_id] = []

        self.conversation_history[patient_id].append({
            "timestamp": datetime.now(),
            "query": query,
            "response": response,
            "intent": intent
        })

        return {
            "response": response,
            "intent": intent,
            "follow_up_questions": self._generate_follow_up_questions(intent),
            "resources": self._suggest_resources(intent)
        }

    def _analyze_intent(self, query: str) -> str:
        """Analyze query intent"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["medication", "drug", "pill", "prescription"]):
            return "medication_question"
        elif any(word in query_lower for word in ["symptom", "pain", "feel", "sick"]):
            return "symptom_check"
        elif any(word in query_lower for word in ["exercise", "diet", "lifestyle", "eat", "weight"]):
            return "lifestyle_advice"
        else:
            return "general"

    def _handle_medication_question(self, query: str) -> str:
        """Handle medication-related questions"""
        knowledge = self.health_knowledge_base["medication_questions"]

        for pattern, response in knowledge["responses"].items():
            if pattern in query.lower():
                return response

        return "For medication questions, please consult your healthcare provider or pharmacist for personalized advice."

    def _handle_symptom_check(self, query: str) -> str:
        """Handle symptom checking"""
        urgent_symptoms = self.health_knowledge_base["symptom_assessment"]["urgent_symptoms"]

        for symptom in urgent_symptoms:
            if symptom in query.lower():
                return self.health_knowledge_base["symptom_assessment"]["response"]

        return "Monitor your symptoms and contact your healthcare provider if they persist or worsen."

    def _handle_lifestyle_advice(self, query: str) -> str:
        """Handle lifestyle advice requests"""
        advice = self.health_knowledge_base["lifestyle_advice"]

        for category, response in advice.items():
            if category in query.lower():
                return response

        return "Focus on maintaining a healthy lifestyle with balanced nutrition, regular exercise, and adequate rest."

    def _generate_general_response(self, query: str) -> str:
        """Generate general response"""
        responses = [
            "I'm here to help with your health questions. Could you provide more details?",
            "For personalized health advice, please consult with your healthcare provider.",
            "I can help answer questions about medications, symptoms, and healthy living."
        ]

        return random.choice(responses)

    def _generate_follow_up_questions(self, intent: str) -> List[str]:
        """Generate follow-up questions"""
        questions = {
            "medication_question": [
                "Are you experiencing any side effects?",
                "Have you been taking the medication as prescribed?"
            ],
            "symptom_check": [
                "How long have you had these symptoms?",
                "Have you noticed any triggers or patterns?"
            ],
            "lifestyle_advice": [
                "What are your current exercise habits?",
                "Are there any barriers to healthy eating?"
            ]
        }

        return questions.get(intent, ["Can you tell me more about your concern?"])

    def _suggest_resources(self, intent: str) -> List[str]:
        """Suggest relevant resources"""
        resources = {
            "medication_question": ["Medication guide", "Drug interaction checker"],
            "symptom_check": ["Symptom checker tool", "When to see a doctor guide"],
            "lifestyle_advice": ["Nutrition guidelines", "Exercise recommendations"]
        }

        return resources.get(intent, ["General health resources"])

class MobileHealthCompanion:
    """Mobile health companion features"""

    def __init__(self):
        self.device_integrations = {}
        self.health_tracking = {}

    def sync_health_data(self, patient_id: str, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync health data from mobile devices"""
        if patient_id not in self.health_tracking:
            self.health_tracking[patient_id] = []

        # Process and store device data
        processed_data = {
            "timestamp": datetime.now(),
            "device_type": device_data.get("device"),
            "data_type": device_data.get("type"),
            "values": device_data.get("values", {}),
            "quality_score": self._assess_data_quality(device_data)
        }

        self.health_tracking[patient_id].append(processed_data)

        return {
            "status": "synced",
            "records_processed": len(device_data.get("values", {})),
            "data_quality": processed_data["quality_score"]
        }

    def _assess_data_quality(self, device_data: Dict[str, Any]) -> float:
        """Assess quality of device data"""
        quality = 0.8  # Base quality

        # Check for completeness
        if device_data.get("values"):
            quality += 0.1

        # Check for reasonable ranges
        values = device_data.get("values", {})
        if values.get("heart_rate") and 50 <= values["heart_rate"] <= 150:
            quality += 0.1

        return min(1.0, quality)

    def generate_daily_report(self, patient_id: str) -> Dict[str, Any]:
        """Generate daily health report"""
        today_data = [record for record in self.health_tracking.get(patient_id, [])
                     if record["timestamp"].date() == datetime.now().date()]

        report = {
            "date": datetime.now().date().isoformat(),
            "steps": self._calculate_daily_steps(today_data),
            "active_minutes": self._calculate_active_minutes(today_data),
            "heart_rate_trend": self._analyze_heart_rate_trend(today_data),
            "sleep_quality": self._assess_sleep_quality(today_data),
            "insights": self._generate_daily_insights(today_data)
        }

        return report

    def _calculate_daily_steps(self, daily_data: List[Dict[str, Any]]) -> int:
        """Calculate total daily steps"""
        total_steps = 0
        for record in daily_data:
            if record.get("data_type") == "activity":
                total_steps += record.get("values", {}).get("steps", 0)

        return total_steps

    def _calculate_active_minutes(self, daily_data: List[Dict[str, Any]]) -> int:
        """Calculate active minutes"""
        active_minutes = 0
        for record in daily_data:
            if record.get("data_type") == "activity":
                active_minutes += record.get("values", {}).get("active_minutes", 0)

        return active_minutes

    def _analyze_heart_rate_trend(self, daily_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze heart rate trend"""
        hr_values = []
        for record in daily_data:
            if record.get("data_type") == "heart_rate":
                hr_values.extend(record.get("values", {}).get("readings", []))

        if not hr_values:
            return {"status": "no_data"}

        avg_hr = sum(hr_values) / len(hr_values)
        min_hr = min(hr_values)
        max_hr = max(hr_values)

        return {
            "average": round(avg_hr, 1),
            "minimum": min_hr,
            "maximum": max_hr,
            "status": "normal" if 60 <= avg_hr <= 100 else "abnormal"
        }

    def _assess_sleep_quality(self, daily_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess sleep quality"""
        sleep_data = None
        for record in daily_data:
            if record.get("data_type") == "sleep":
                sleep_data = record.get("values", {})
                break

        if not sleep_data:
            return {"status": "no_data"}

        total_sleep = sleep_data.get("total_minutes", 0)
        deep_sleep = sleep_data.get("deep_minutes", 0)

        quality_score = min(100, (total_sleep / 480) * 100)  # 8 hours = 100%

        return {
            "total_hours": round(total_sleep / 60, 1),
            "deep_sleep_percentage": round((deep_sleep / total_sleep) * 100, 1) if total_sleep > 0 else 0,
            "quality_score": round(quality_score, 1),
            "rating": "excellent" if quality_score > 85 else "good" if quality_score > 70 else "fair"
        }

    def _generate_daily_insights(self, daily_data: List[Dict[str, Any]]) -> List[str]:
        """Generate daily health insights"""
        insights = []

        steps = self._calculate_daily_steps(daily_data)
        if steps >= 10000:
            insights.append("Great job meeting your daily step goal!")
        elif steps < 5000:
            insights.append("Try to increase your daily activity level.")

        active_minutes = self._calculate_active_minutes(daily_data)
        if active_minutes >= 30:
            insights.append("You've achieved your daily activity target.")
        else:
            insights.append("Consider adding more physical activity today.")

        return insights

class HealthGamificationEngine:
    """Health gamification and motivation system"""

    def __init__(self):
        self.patient_achievements = {}
        self.challenges = self._initialize_challenges()

    def _initialize_challenges(self) -> List[Dict[str, Any]]:
        """Initialize available health challenges"""
        return [
            {
                "id": "step_challenge",
                "name": "10,000 Steps Daily",
                "description": "Walk 10,000 steps every day for a week",
                "duration_days": 7,
                "reward_points": 100,
                "criteria": {"steps": 10000}
            },
            {
                "id": "medication_adherence",
                "name": "Perfect Adherence Week",
                "description": "Take all medications on time for 7 days",
                "duration_days": 7,
                "reward_points": 150,
                "criteria": {"adherence_rate": 1.0}
            },
            {
                "id": "healthy_eating",
                "name": "Nutrition Champion",
                "description": "Log meals for 5 consecutive days",
                "duration_days": 5,
                "reward_points": 75,
                "criteria": {"meals_logged": 5}
            }
        ]

    def get_patient_stats(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's gamification statistics"""
        if patient_id not in self.patient_achievements:
            self.patient_achievements[patient_id] = {
                "total_points": 0,
                "current_streak": 0,
                "completed_challenges": 0,
                "active_challenges": [],
                "badges": []
            }

        stats = self.patient_achievements[patient_id]

        # Calculate level based on points
        level = min(10, stats["total_points"] // 500 + 1)

        return {
            "level": level,
            "total_points": stats["total_points"],
            "points_to_next_level": (level * 500) - stats["total_points"],
            "current_streak": stats["current_streak"],
            "completed_challenges": stats["completed_challenges"],
            "active_challenges": stats["active_challenges"],
            "badges": stats["badges"]
        }

    def check_challenge_completion(self, patient_id: str, health_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for completed challenges"""
        completed = []

        if patient_id not in self.patient_achievements:
            return completed

        active_challenges = self.patient_achievements[patient_id]["active_challenges"]

        for challenge in active_challenges:
            if self._evaluate_challenge(challenge, health_data):
                completed.append(challenge)
                # Award points
                self.patient_achievements[patient_id]["total_points"] += challenge["reward_points"]
                self.patient_achievements[patient_id]["completed_challenges"] += 1

                # Remove from active challenges
                active_challenges.remove(challenge)

        return completed

    def _evaluate_challenge(self, challenge: Dict[str, Any], health_data: Dict[str, Any]) -> bool:
        """Evaluate if challenge criteria are met"""
        criteria = challenge.get("criteria", {})

        for metric, target in criteria.items():
            if metric == "steps":
                daily_steps = health_data.get("daily_steps", 0)
                if daily_steps < target:
                    return False
            elif metric == "adherence_rate":
                adherence = health_data.get("medication_adherence", 0)
                if adherence < target:
                    return False
            elif metric == "meals_logged":
                meals = health_data.get("meals_logged", 0)
                if meals < target:
                    return False

        return True

    def get_available_challenges(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get challenges available to patient"""
        # Return all challenges for simplicity
        return self.challenges

class NotificationSystem:
    """Smart notification system for patient engagement"""

    def __init__(self):
        self.patient_notifications = {}

    def get_patient_notifications(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get notifications for patient"""
        if patient_id not in self.patient_notifications:
            self.patient_notifications[patient_id] = []

        # Add some sample notifications
        notifications = [
            {
                "id": "medication_reminder",
                "type": "reminder",
                "title": "Medication Reminder",
                "message": "Time to take your evening medications",
                "priority": "high",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "appointment_reminder",
                "type": "appointment",
                "title": "Upcoming Appointment",
                "message": "You have a cardiology appointment tomorrow at 2:00 PM",
                "priority": "medium",
                "timestamp": (datetime.now() + timedelta(hours=2)).isoformat()
            },
            {
                "id": "goal_celebration",
                "type": "achievement",
                "title": "Goal Achievement!",
                "message": "Congratulations! You've reached your weekly step goal.",
                "priority": "low",
                "timestamp": datetime.now().isoformat()
            }
        ]

        return notifications

    def schedule_notification(self, patient_id: str, notification: Dict[str, Any]) -> str:
        """Schedule a new notification"""
        if patient_id not in self.patient_notifications:
            self.patient_notifications[patient_id] = []

        notification_id = f"notif_{int(time.time())}_{random.randint(1000, 9999)}"
        notification["id"] = notification_id
        notification["scheduled_at"] = datetime.now().isoformat()

        self.patient_notifications[patient_id].append(notification)

        return notification_id
