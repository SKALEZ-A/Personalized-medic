"""
Telemedicine and Remote Monitoring System
Video consultations, remote patient monitoring, and virtual care delivery
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
import random
import statistics

class ConsultationStatus(Enum):
    SCHEDULED = "scheduled"
    WAITING_ROOM = "waiting_room"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"

class DeviceType(Enum):
    BLOOD_PRESSURE_MONITOR = "blood_pressure_monitor"
    GLUCOSE_METER = "glucose_meter"
    PULSE_OXIMETER = "pulse_oximeter"
    WEIGHING_SCALE = "weighing_scale"
    THERMOMETER = "thermometer"
    ECG_MONITOR = "ecg_monitor"
    PEAK_FLOW_METER = "peak_flow_meter"

class TelemedicinePlatform:
    """Main telemedicine platform"""

    def __init__(self):
        self.active_consultations = {}
        self.consultation_history = []
        self.waiting_rooms = {}
        self.providers = {}
        self.patients = {}
        self.devices = {}

    def schedule_consultation(self, consultation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a telemedicine consultation"""
        consultation_id = f"telemed_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"

        consultation = {
            "consultation_id": consultation_id,
            "patient_id": consultation_data["patient_id"],
            "provider_id": consultation_data["provider_id"],
            "scheduled_time": consultation_data["scheduled_time"],
            "duration_minutes": consultation_data.get("duration_minutes", 30),
            "consultation_type": consultation_data.get("consultation_type", "general"),
            "status": ConsultationStatus.SCHEDULED.value,
            "urgency": consultation_data.get("urgency", "routine"),
            "specialty": consultation_data.get("specialty", "general_medicine"),
            "symptoms": consultation_data.get("symptoms", []),
            "notes": consultation_data.get("notes", ""),
            "video_platform": consultation_data.get("video_platform", "integrated"),
            "meeting_link": self._generate_meeting_link(),
            "created_at": datetime.now().isoformat(),
            "reminders_sent": [],
            "follow_up_required": False,
            "follow_up_date": None
        }

        self.active_consultations[consultation_id] = consultation

        # Schedule reminders
        self._schedule_reminders(consultation_id, consultation)

        return consultation

    def _generate_meeting_link(self) -> str:
        """Generate secure meeting link"""
        meeting_id = uuid.uuid4().hex[:12].upper()
        return f"https://telemed.platform.com/meet/{meeting_id}"

    def _schedule_reminders(self, consultation_id: str, consultation: Dict[str, Any]):
        """Schedule consultation reminders"""
        scheduled_time = datetime.fromisoformat(consultation["scheduled_time"])

        # 24-hour reminder
        reminder_24h = scheduled_time - timedelta(hours=24)
        if reminder_24h > datetime.now():
            self._schedule_reminder(consultation_id, reminder_24h, "24_hour_reminder")

        # 1-hour reminder
        reminder_1h = scheduled_time - timedelta(hours=1)
        if reminder_1h > datetime.now():
            self._schedule_reminder(consultation_id, reminder_1h, "1_hour_reminder")

        # 15-minute reminder
        reminder_15m = scheduled_time - timedelta(minutes=15)
        if reminder_15m > datetime.now():
            self._schedule_reminder(consultation_id, reminder_15m, "15_minute_reminder")

    def _schedule_reminder(self, consultation_id: str, reminder_time: datetime, reminder_type: str):
        """Schedule individual reminder"""
        reminder = {
            "consultation_id": consultation_id,
            "reminder_type": reminder_type,
            "scheduled_time": reminder_time.isoformat(),
            "sent": False,
            "channel": "both"  # email and SMS
        }

        # In real implementation, would schedule with task scheduler
        consultation = self.active_consultations[consultation_id]
        consultation["reminders_sent"].append(reminder)

    def start_consultation(self, consultation_id: str) -> Dict[str, Any]:
        """Start a telemedicine consultation"""
        if consultation_id not in self.active_consultations:
            raise ValueError(f"Consultation not found: {consultation_id}")

        consultation = self.active_consultations[consultation_id]

        if consultation["status"] != ConsultationStatus.SCHEDULED.value:
            raise ValueError(f"Consultation is not in scheduled state: {consultation['status']}")

        consultation["status"] = ConsultationStatus.IN_PROGRESS.value
        consultation["started_at"] = datetime.now().isoformat()

        # Initialize consultation session
        session = {
            "session_id": f"session_{int(datetime.now().timestamp())}",
            "consultation_id": consultation_id,
            "participants": {
                "provider": consultation["provider_id"],
                "patient": consultation["patient_id"]
            },
            "video_quality": "HD",
            "audio_quality": "clear",
            "connection_status": "connected",
            "shared_documents": [],
            "chat_messages": [],
            "screen_sharing": False,
            "recording": False
        }

        consultation["session"] = session

        return consultation

    def join_waiting_room(self, consultation_id: str, participant_type: str,
                         participant_id: str) -> Dict[str, Any]:
        """Join consultation waiting room"""
        if consultation_id not in self.active_consultations:
            raise ValueError(f"Consultation not found: {consultation_id}")

        consultation = self.active_consultations[consultation_id]

        if consultation_id not in self.waiting_rooms:
            self.waiting_rooms[consultation_id] = {
                "consultation_id": consultation_id,
                "participants": [],
                "waiting_since": datetime.now().isoformat()
            }

        waiting_room = self.waiting_rooms[consultation_id]

        # Check if participant already in waiting room
        existing_participant = next(
            (p for p in waiting_room["participants"]
             if p["participant_id"] == participant_id), None
        )

        if existing_participant:
            existing_participant["joined_at"] = datetime.now().isoformat()
            return existing_participant

        # Add new participant
        participant = {
            "participant_id": participant_id,
            "participant_type": participant_type,
            "joined_at": datetime.now().isoformat(),
            "ready": False,
            "device_check": {
                "camera": False,
                "microphone": False,
                "internet": "checking"
            }
        }

        waiting_room["participants"].append(participant)

        return participant

    def complete_consultation(self, consultation_id: str, completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete a telemedicine consultation"""
        if consultation_id not in self.active_consultations:
            raise ValueError(f"Consultation not found: {consultation_id}")

        consultation = self.active_consultations[consultation_id]

        consultation["status"] = ConsultationStatus.COMPLETED.value
        consultation["completed_at"] = datetime.now().isoformat()
        consultation["duration_actual"] = (
            datetime.fromisoformat(consultation["completed_at"]) -
            datetime.fromisoformat(consultation["started_at"])
        ).total_seconds() / 60

        # Add completion data
        consultation.update({
            "diagnosis": completion_data.get("diagnosis"),
            "treatment_plan": completion_data.get("treatment_plan"),
            "prescriptions": completion_data.get("prescriptions", []),
            "follow_up_required": completion_data.get("follow_up_required", False),
            "follow_up_instructions": completion_data.get("follow_up_instructions"),
            "patient_satisfaction": completion_data.get("patient_satisfaction"),
            "provider_notes": completion_data.get("provider_notes"),
            "recording_url": completion_data.get("recording_url")
        })

        # Move to history
        self.consultation_history.append(consultation)
        del self.active_consultations[consultation_id]

        # Schedule follow-up if required
        if consultation["follow_up_required"]:
            self._schedule_follow_up(consultation)

        return consultation

    def _schedule_follow_up(self, consultation: Dict[str, Any]):
        """Schedule follow-up consultation"""
        follow_up_date = datetime.now() + timedelta(days=7)  # Default 1 week

        follow_up_consultation = {
            "consultation_id": f"followup_{consultation['consultation_id']}",
            "patient_id": consultation["patient_id"],
            "provider_id": consultation["provider_id"],
            "scheduled_time": follow_up_date.isoformat(),
            "consultation_type": "follow_up",
            "reason": "Follow-up from previous consultation",
            "parent_consultation": consultation["consultation_id"]
        }

        self.schedule_consultation(follow_up_consultation)

    def get_consultation_status(self, consultation_id: str) -> Optional[Dict[str, Any]]:
        """Get consultation status"""
        if consultation_id in self.active_consultations:
            return self.active_consultations[consultation_id]

        # Check history
        historical = next(
            (c for c in self.consultation_history if c["consultation_id"] == consultation_id),
            None
        )

        return historical

class RemotePatientMonitoring:
    """Remote patient monitoring system"""

    def __init__(self):
        self.monitored_patients = {}
        self.device_registrations = {}
        self.monitoring_alerts = []
        self.vital_signs_history = defaultdict(list)

    def register_device(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a medical device for monitoring"""
        device_id = device_data.get("device_id", f"device_{uuid.uuid4().hex[:8]}")

        device = {
            "device_id": device_id,
            "patient_id": device_data["patient_id"],
            "device_type": device_data["device_type"],
            "model": device_data.get("model", "unknown"),
            "serial_number": device_data.get("serial_number"),
            "registered_at": datetime.now().isoformat(),
            "last_sync": None,
            "battery_level": 100,
            "firmware_version": device_data.get("firmware_version", "1.0"),
            "status": "active",
            "alert_thresholds": self._get_default_thresholds(device_data["device_type"])
        }

        self.device_registrations[device_id] = device

        # Add to patient's monitored devices
        if device["patient_id"] not in self.monitored_patients:
            self.monitored_patients[device["patient_id"]] = {
                "patient_id": device["patient_id"],
                "devices": [],
                "monitoring_plan": {},
                "alert_preferences": {}
            }

        self.monitored_patients[device["patient_id"]]["devices"].append(device_id)

        return device

    def _get_default_thresholds(self, device_type: str) -> Dict[str, Any]:
        """Get default alert thresholds for device type"""
        thresholds = {
            DeviceType.BLOOD_PRESSURE_MONITOR.value: {
                "systolic_high": 140,
                "systolic_low": 90,
                "diastolic_high": 90,
                "diastolic_low": 60
            },
            DeviceType.GLUCOSE_METER.value: {
                "glucose_high": 180,
                "glucose_low": 70
            },
            DeviceType.PULSE_OXIMETER.value: {
                "spo2_low": 95,
                "heart_rate_high": 100,
                "heart_rate_low": 60
            },
            DeviceType.WEIGHING_SCALE.value: {
                "weight_change_threshold": 2.0  # kg
            }
        }

        return thresholds.get(device_type, {})

    def receive_device_data(self, device_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Receive data from medical device"""
        if device_id not in self.device_registrations:
            return {"error": "Device not registered"}

        device = self.device_registrations[device_id]
        patient_id = device["patient_id"]

        # Process the data
        processed_data = self._process_device_data(device, data)

        # Store in history
        timestamp = datetime.now().isoformat()
        history_entry = {
            "timestamp": timestamp,
            "device_id": device_id,
            "device_type": device["device_type"],
            "data": processed_data,
            "raw_data": data
        }

        self.vital_signs_history[patient_id].append(history_entry)

        # Update device sync time
        device["last_sync"] = timestamp

        # Check for alerts
        alerts = self._check_alerts(device, processed_data)

        if alerts:
            self.monitoring_alerts.extend(alerts)

        return {
            "device_id": device_id,
            "patient_id": patient_id,
            "data_processed": True,
            "alerts_generated": len(alerts),
            "alerts": alerts
        }

    def _process_device_data(self, device: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw device data"""
        device_type = device["device_type"]
        processed = {}

        if device_type == DeviceType.BLOOD_PRESSURE_MONITOR.value:
            processed = {
                "systolic": data.get("systolic"),
                "diastolic": data.get("diastolic"),
                "heart_rate": data.get("heart_rate"),
                "measurement_time": data.get("timestamp")
            }
        elif device_type == DeviceType.GLUCOSE_METER.value:
            processed = {
                "glucose_level": data.get("glucose"),
                "measurement_type": data.get("type", "fasting"),
                "measurement_time": data.get("timestamp")
            }
        elif device_type == DeviceType.PULSE_OXIMETER.value:
            processed = {
                "spo2": data.get("spo2"),
                "heart_rate": data.get("heart_rate"),
                "perfusion_index": data.get("pi"),
                "measurement_time": data.get("timestamp")
            }
        elif device_type == DeviceType.WEIGHING_SCALE.value:
            processed = {
                "weight": data.get("weight"),
                "body_fat": data.get("body_fat"),
                "muscle_mass": data.get("muscle_mass"),
                "measurement_time": data.get("timestamp")
            }

        return processed

    def _check_alerts(self, device: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check data against alert thresholds"""
        alerts = []
        thresholds = device.get("alert_thresholds", {})

        for metric, value in data.items():
            if metric in thresholds:
                threshold_config = thresholds[metric]

                if isinstance(threshold_config, dict):
                    # Range thresholds
                    if "high" in threshold_config and value > threshold_config["high"]:
                        alerts.append({
                            "alert_id": f"alert_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}",
                            "patient_id": device["patient_id"],
                            "device_id": device["device_id"],
                            "alert_type": "threshold_high",
                            "metric": metric,
                            "value": value,
                            "threshold": threshold_config["high"],
                            "severity": "high",
                            "message": f"{metric} is above normal threshold",
                            "timestamp": datetime.now().isoformat()
                        })
                    elif "low" in threshold_config and value < threshold_config["low"]:
                        alerts.append({
                            "alert_id": f"alert_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}",
                            "patient_id": device["patient_id"],
                            "device_id": device["device_id"],
                            "alert_type": "threshold_low",
                            "metric": metric,
                            "value": value,
                            "threshold": threshold_config["low"],
                            "severity": "high",
                            "message": f"{metric} is below normal threshold",
                            "timestamp": datetime.now().isoformat()
                        })

        # Check for trends
        trend_alerts = self._check_trends(device["patient_id"], device["device_type"], data)
        alerts.extend(trend_alerts)

        return alerts

    def _check_trends(self, patient_id: str, device_type: str, current_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for concerning trends in patient data"""
        alerts = []
        history = self.vital_signs_history[patient_id][-10:]  # Last 10 readings

        if len(history) < 3:
            return alerts

        # Check blood pressure trends
        if device_type == DeviceType.BLOOD_PRESSURE_MONITOR.value:
            recent_bp = [
                (h["data"].get("systolic"), h["data"].get("diastolic"))
                for h in history[-3:]
                if h["data"].get("systolic") and h["data"].get("diastolic")
            ]

            if len(recent_bp) >= 3:
                systolic_trend = all(bp[0] > 140 for bp in recent_bp)
                if systolic_trend:
                    alerts.append({
                        "alert_id": f"trend_{int(datetime.now().timestamp())}",
                        "patient_id": patient_id,
                        "alert_type": "trend_alert",
                        "metric": "blood_pressure",
                        "severity": "medium",
                        "message": "Consistently elevated blood pressure readings",
                        "timestamp": datetime.now().isoformat()
                    })

        return alerts

    def get_patient_monitoring_status(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient monitoring status"""
        if patient_id not in self.monitored_patients:
            return {"error": "Patient not monitored"}

        patient_monitoring = self.monitored_patients[patient_id]

        # Get latest readings for each device
        latest_readings = {}
        for device_id in patient_monitoring["devices"]:
            if device_id in self.device_registrations:
                device = self.device_registrations[device_id]
                history = self.vital_signs_history[patient_id]

                if history:
                    latest_reading = max(history, key=lambda x: x["timestamp"])
                    latest_readings[device["device_type"]] = {
                        "device_id": device_id,
                        "latest_reading": latest_reading,
                        "last_sync": device.get("last_sync")
                    }

        # Get active alerts
        active_alerts = [
            alert for alert in self.monitoring_alerts
            if alert["patient_id"] == patient_id and not alert.get("resolved", False)
        ]

        # Calculate compliance
        compliance = self._calculate_monitoring_compliance(patient_id)

        return {
            "patient_id": patient_id,
            "devices": patient_monitoring["devices"],
            "latest_readings": latest_readings,
            "active_alerts": active_alerts,
            "compliance_score": compliance,
            "monitoring_summary": self._generate_monitoring_summary(latest_readings, active_alerts)
        }

    def _calculate_monitoring_compliance(self, patient_id: str) -> float:
        """Calculate patient monitoring compliance"""
        devices = self.monitored_patients.get(patient_id, {}).get("devices", [])
        if not devices:
            return 100.0

        total_expected_readings = len(devices) * 7  # Expected daily readings for 7 days
        actual_readings = len(self.vital_signs_history[patient_id])

        compliance = min(100.0, (actual_readings / total_expected_readings) * 100) if total_expected_readings > 0 else 100.0

        return round(compliance, 1)

    def _generate_monitoring_summary(self, latest_readings: Dict[str, Any],
                                   active_alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate monitoring summary"""
        summary = {
            "total_devices": len(latest_readings),
            "active_alerts_count": len(active_alerts),
            "overall_status": "normal",
            "last_update": None
        }

        if latest_readings:
            timestamps = [r["latest_reading"]["timestamp"] for r in latest_readings.values()]
            summary["last_update"] = max(timestamps) if timestamps else None

        if active_alerts:
            high_priority_alerts = [a for a in active_alerts if a["severity"] == "high"]
            summary["overall_status"] = "critical" if high_priority_alerts else "warning"

        return summary

class VirtualCareCoordinator:
    """Virtual care coordination system"""

    def __init__(self):
        self.care_plans = {}
        self.patient_engagement = {}
        self.intervention_triggers = {}

    def create_virtual_care_plan(self, patient_id: str, care_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create virtual care plan"""
        plan_id = f"vcare_{patient_id}_{int(datetime.now().timestamp())}"

        care_plan = {
            "plan_id": plan_id,
            "patient_id": patient_id,
            "conditions": care_plan_data.get("conditions", []),
            "monitoring_schedule": care_plan_data.get("monitoring_schedule", {}),
            "intervention_triggers": care_plan_data.get("intervention_triggers", []),
            "educational_content": care_plan_data.get("educational_content", []),
            "communication_preferences": care_plan_data.get("communication_preferences", {}),
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

        self.care_plans[plan_id] = care_plan

        # Setup intervention triggers
        self._setup_intervention_triggers(plan_id, care_plan["intervention_triggers"])

        return care_plan

    def _setup_intervention_triggers(self, plan_id: str, triggers: List[Dict[str, Any]]):
        """Setup automated intervention triggers"""
        for trigger in triggers:
            trigger_id = f"trigger_{plan_id}_{int(datetime.now().timestamp())}"

            self.intervention_triggers[trigger_id] = {
                "trigger_id": trigger_id,
                "plan_id": plan_id,
                "condition": trigger["condition"],
                "action": trigger["action"],
                "threshold": trigger.get("threshold"),
                "cooldown_period": trigger.get("cooldown_period", 3600),  # 1 hour default
                "last_triggered": None,
                "active": True
            }

    def check_intervention_triggers(self, patient_id: str, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any intervention triggers should be activated"""
        interventions = []

        # Find active care plans for patient
        patient_plans = [
            plan for plan in self.care_plans.values()
            if plan["patient_id"] == patient_id and plan["status"] == "active"
        ]

        for plan in patient_plans:
            plan_triggers = [
                trigger for trigger in self.intervention_triggers.values()
                if trigger["plan_id"] == plan["plan_id"] and trigger["active"]
            ]

            for trigger in plan_triggers:
                if self._evaluate_trigger_condition(trigger, patient_data):
                    interventions.append({
                        "intervention_id": f"intervention_{int(datetime.now().timestamp())}",
                        "trigger_id": trigger["trigger_id"],
                        "patient_id": patient_id,
                        "plan_id": plan["plan_id"],
                        "action": trigger["action"],
                        "reason": trigger["condition"],
                        "timestamp": datetime.now().isoformat()
                    })

                    trigger["last_triggered"] = datetime.now().isoformat()

        return interventions

    def _evaluate_trigger_condition(self, trigger: Dict[str, Any], patient_data: Dict[str, Any]) -> bool:
        """Evaluate if trigger condition is met"""
        condition = trigger["condition"]
        threshold = trigger.get("threshold")

        # Check cooldown period
        if trigger["last_triggered"]:
            last_triggered = datetime.fromisoformat(trigger["last_triggered"])
            cooldown_end = last_triggered + timedelta(seconds=trigger["cooldown_period"])
            if datetime.now() < cooldown_end:
                return False

        # Evaluate condition
        if condition == "high_blood_pressure":
            bp = patient_data.get("blood_pressure", {})
            systolic = bp.get("systolic", 0)
            return systolic > threshold if threshold else systolic > 140

        elif condition == "low_oxygen_saturation":
            spo2 = patient_data.get("spo2", 100)
            return spo2 < threshold if threshold else spo2 < 95

        elif condition == "missed_monitoring":
            last_reading = patient_data.get("last_reading_time")
            if last_reading:
                hours_since_reading = (datetime.now() - datetime.fromisoformat(last_reading)).total_seconds() / 3600
                return hours_since_reading > threshold if threshold else hours_since_reading > 24

        return False

    def update_patient_engagement(self, patient_id: str, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update patient engagement metrics"""
        if patient_id not in self.patient_engagement:
            self.patient_engagement[patient_id] = {
                "patient_id": patient_id,
                "engagement_score": 0,
                "last_interaction": None,
                "completed_modules": [],
                "survey_responses": [],
                "communication_history": []
            }

        engagement = self.patient_engagement[patient_id]

        # Update engagement score based on activities
        score_changes = {
            "completed_education": 10,
            "responded_to_survey": 5,
            "attended_consultation": 15,
            "regular_monitoring": 8,
            "shared_data": 3
        }

        activity_type = engagement_data.get("activity_type")
        if activity_type in score_changes:
            engagement["engagement_score"] = min(100, engagement["engagement_score"] + score_changes[activity_type])

        engagement["last_interaction"] = datetime.now().isoformat()

        # Record the activity
        engagement["communication_history"].append({
            "timestamp": datetime.now().isoformat(),
            "activity_type": activity_type,
            "details": engagement_data.get("details", {})
        })

        return engagement

    def get_patient_engagement_report(self, patient_id: str) -> Dict[str, Any]:
        """Get patient engagement report"""
        if patient_id not in self.patient_engagement:
            return {"error": "No engagement data found"}

        engagement = self.patient_engagement[patient_id]

        # Calculate engagement level
        score = engagement["engagement_score"]
        if score >= 80:
            level = "highly_engaged"
        elif score >= 60:
            level = "moderately_engaged"
        elif score >= 40:
            level = "low_engagement"
        else:
            level = "disengaged"

        # Analyze trends
        recent_activities = engagement["communication_history"][-10:]
        activity_trend = "stable"

        if len(recent_activities) >= 5:
            recent_dates = [datetime.fromisoformat(a["timestamp"]) for a in recent_activities[-5:]]
            days_span = (recent_dates[-1] - recent_dates[0]).days
            avg_days_between_activities = days_span / (len(recent_dates) - 1) if len(recent_dates) > 1 else 0

            if avg_days_between_activities < 2:
                activity_trend = "increasing"
            elif avg_days_between_activities > 7:
                activity_trend = "decreasing"

        return {
            "patient_id": patient_id,
            "engagement_score": score,
            "engagement_level": level,
            "activity_trend": activity_trend,
            "total_activities": len(engagement["communication_history"]),
            "last_interaction": engagement["last_interaction"],
            "completed_modules": len(engagement["completed_modules"]),
            "recommendations": self._generate_engagement_recommendations(level, activity_trend)
        }

    def _generate_engagement_recommendations(self, level: str, trend: str) -> List[str]:
        """Generate engagement improvement recommendations"""
        recommendations = []

        if level == "disengaged":
            recommendations.extend([
                "Send personalized welcome message",
                "Schedule introductory consultation",
                "Provide simple educational materials",
                "Set up automated check-in reminders"
            ])
        elif level == "low_engagement":
            recommendations.extend([
                "Send targeted educational content",
                "Offer virtual support group invitation",
                "Provide progress tracking tools",
                "Schedule follow-up engagement call"
            ])

        if trend == "decreasing":
            recommendations.extend([
                "Investigate barriers to engagement",
                "Send re-engagement survey",
                "Offer alternative communication methods",
                "Consider care plan adjustment"
            ])

        return recommendations

class TelemedicineAnalytics:
    """Analytics for telemedicine services"""

    def __init__(self):
        self.consultation_metrics = []
        self.patient_outcomes = []
        self.system_performance = []

    def record_consultation_metrics(self, consultation_data: Dict[str, Any]):
        """Record consultation performance metrics"""
        metrics = {
            "consultation_id": consultation_data["consultation_id"],
            "duration_minutes": consultation_data.get("duration_actual", 0),
            "wait_time_minutes": consultation_data.get("wait_time", 0),
            "technical_issues": consultation_data.get("technical_issues", False),
            "patient_satisfaction": consultation_data.get("patient_satisfaction"),
            "completion_status": consultation_data["status"],
            "consultation_type": consultation_data.get("consultation_type"),
            "timestamp": datetime.now().isoformat()
        }

        self.consultation_metrics.append(metrics)

    def analyze_telemedicine_effectiveness(self, time_period: str = "30_days") -> Dict[str, Any]:
        """Analyze telemedicine effectiveness"""
        # Filter recent metrics
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_metrics = [
            m for m in self.consultation_metrics
            if datetime.fromisoformat(m["timestamp"]) > cutoff_date
        ]

        if not recent_metrics:
            return {"error": "No recent consultation data"}

        analysis = {
            "time_period": time_period,
            "total_consultations": len(recent_metrics),
            "completed_consultations": len([m for m in recent_metrics if m["completion_status"] == "completed"]),
            "average_duration": statistics.mean([m["duration_minutes"] for m in recent_metrics if m["duration_minutes"] > 0]),
            "average_wait_time": statistics.mean([m["wait_time_minutes"] for m in recent_metrics]),
            "technical_issue_rate": len([m for m in recent_metrics if m["technical_issues"]]) / len(recent_metrics),
            "patient_satisfaction_avg": statistics.mean([m["patient_satisfaction"] for m in recent_metrics if m["patient_satisfaction"]]),
            "consultation_types": self._analyze_consultation_types(recent_metrics),
            "efficiency_metrics": self._calculate_efficiency_metrics(recent_metrics)
        }

        return analysis

    def _analyze_consultation_types(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consultation types distribution"""
        type_counts = {}
        for metric in metrics:
            ctype = metric.get("consultation_type", "unknown")
            type_counts[ctype] = type_counts.get(ctype, 0) + 1

        return {
            "distribution": type_counts,
            "most_common": max(type_counts.keys(), key=lambda x: type_counts[x]) if type_counts else None
        }

    def _calculate_efficiency_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate telemedicine efficiency metrics"""
        completed = [m for m in metrics if m["completion_status"] == "completed"]
        if not completed:
            return {"efficiency_score": 0}

        # Efficiency based on completion rate and wait times
        completion_rate = len(completed) / len(metrics)
        avg_wait_time = statistics.mean([m["wait_time_minutes"] for m in completed])
        technical_issues = len([m for m in completed if m["technical_issues"]])

        # Normalize to 0-100 scale
        efficiency_score = (
            (completion_rate * 40) +  # 40% weight for completion
            (max(0, 40 - avg_wait_time) * 0.4) +  # 40% weight for wait time (lower is better)
            ((1 - technical_issues/len(completed)) * 20)  # 20% weight for technical reliability
        )

        return {
            "efficiency_score": round(efficiency_score, 1),
            "completion_rate": round(completion_rate * 100, 1),
            "avg_wait_time": round(avg_wait_time, 1),
            "technical_reliability": round((1 - technical_issues/len(completed)) * 100, 1)
        }

    def track_patient_outcomes(self, outcome_data: Dict[str, Any]):
        """Track telemedicine patient outcomes"""
        outcome = {
            "patient_id": outcome_data["patient_id"],
            "consultation_id": outcome_data.get("consultation_id"),
            "outcome_type": outcome_data["outcome_type"],
            "baseline_value": outcome_data.get("baseline_value"),
            "current_value": outcome_data.get("current_value"),
            "improvement": outcome_data.get("improvement"),
            "timestamp": datetime.now().isoformat()
        }

        self.patient_outcomes.append(outcome)

    def generate_telemedicine_report(self) -> Dict[str, Any]:
        """Generate comprehensive telemedicine report"""
        effectiveness = self.analyze_telemedicine_effectiveness()

        # Patient outcome analysis
        outcomes_by_type = {}
        for outcome in self.patient_outcomes[-100:]:  # Last 100 outcomes
            otype = outcome["outcome_type"]
            if otype not in outcomes_by_type:
                outcomes_by_type[otype] = []
            outcomes_by_type[otype].append(outcome)

        outcome_summary = {}
        for otype, outcomes in outcomes_by_type.items():
            improvements = [o.get("improvement", 0) for o in outcomes if o.get("improvement") is not None]
            if improvements:
                outcome_summary[otype] = {
                    "average_improvement": statistics.mean(improvements),
                    "positive_outcomes": len([i for i in improvements if i > 0]),
                    "total_measurements": len(improvements)
                }

        return {
            "report_generated": datetime.now().isoformat(),
            "effectiveness_analysis": effectiveness,
            "patient_outcomes": outcome_summary,
            "system_recommendations": self._generate_system_recommendations(effectiveness),
            "telemedicine_roi": self._calculate_telemedicine_roi()
        }

    def _generate_system_recommendations(self, effectiveness: Dict[str, Any]) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []

        if effectiveness.get("technical_issue_rate", 0) > 0.1:
            recommendations.append("Improve technical infrastructure to reduce connection issues")

        if effectiveness.get("average_wait_time", 0) > 15:
            recommendations.append("Optimize scheduling system to reduce patient wait times")

        if effectiveness.get("patient_satisfaction_avg", 0) < 4.0:
            recommendations.append("Enhance provider training and patient communication protocols")

        efficiency_score = effectiveness.get("efficiency_metrics", {}).get("efficiency_score", 0)
        if efficiency_score < 70:
            recommendations.append("Implement process improvements to increase overall efficiency")

        return recommendations

    def _calculate_telemedicine_roi(self) -> Dict[str, Any]:
        """Calculate telemedicine return on investment"""
        # Simplified ROI calculation
        virtual_visits = len([m for m in self.consultation_metrics
                            if m.get("consultation_type") == "virtual"])

        # Assume cost savings per virtual visit
        cost_savings_per_visit = 150  # USD
        total_savings = virtual_visits * cost_savings_per_visit

        # Assume implementation costs
        implementation_costs = 50000  # USD
        monthly_operational_costs = 2000  # USD

        months_active = 6  # Assume 6 months of operation
        operational_costs = months_active * monthly_operational_costs

        total_costs = implementation_costs + operational_costs
        roi = ((total_savings - total_costs) / total_costs) * 100 if total_costs > 0 else 0

        return {
            "virtual_visits": virtual_visits,
            "total_savings": total_savings,
            "total_costs": total_costs,
            "roi_percentage": round(roi, 1),
            "break_even_months": round(implementation_costs / (cost_savings_per_visit * virtual_visits / months_active), 1) if virtual_visits > 0 else 0
        }
