"""
Workflow Engines for Healthcare Process Management
Care coordination, appointment scheduling, and clinical workflows
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
import uuid
import heapq
from collections import defaultdict, deque

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"

class WorkflowPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class WorkflowEngine:
    """Core workflow engine for healthcare processes"""

    def __init__(self):
        self.workflows = {}
        self.workflow_definitions = {}
        self.task_queue = []
        self.running_tasks = {}
        self.completed_workflows = []

    def define_workflow(self, workflow_id: str, definition: Dict[str, Any]) -> None:
        """Define a workflow template"""
        self.workflow_definitions[workflow_id] = {
            "id": workflow_id,
            "name": definition.get("name", workflow_id),
            "description": definition.get("description", ""),
            "steps": definition.get("steps", []),
            "triggers": definition.get("triggers", []),
            "variables": definition.get("variables", {}),
            "created_at": datetime.now().isoformat()
        }

    def start_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> str:
        """Start a new workflow instance"""
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow definition not found: {workflow_id}")

        if context is None:
            context = {}

        instance_id = f"{workflow_id}_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"

        workflow_instance = {
            "instance_id": instance_id,
            "workflow_id": workflow_id,
            "status": WorkflowStatus.PENDING,
            "priority": context.get("priority", WorkflowPriority.NORMAL),
            "context": context,
            "current_step": 0,
            "step_history": [],
            "variables": self.workflow_definitions[workflow_id]["variables"].copy(),
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error_message": None
        }

        self.workflows[instance_id] = workflow_instance

        # Add to priority queue
        heapq.heappush(self.task_queue, (
            workflow_instance["priority"].value,
            workflow_instance["created_at"],
            instance_id
        ))

        return instance_id

    async def execute_workflow(self, instance_id: str) -> Dict[str, Any]:
        """Execute a workflow instance"""
        if instance_id not in self.workflows:
            raise ValueError(f"Workflow instance not found: {instance_id}")

        workflow = self.workflows[instance_id]
        definition = self.workflow_definitions[workflow["workflow_id"]]

        workflow["status"] = WorkflowStatus.RUNNING
        workflow["started_at"] = datetime.now().isoformat()
        self.running_tasks[instance_id] = workflow

        try:
            for step_index, step in enumerate(definition["steps"]):
                workflow["current_step"] = step_index

                step_result = await self._execute_step(workflow, step)

                workflow["step_history"].append({
                    "step": step_index,
                    "step_name": step.get("name", f"Step {step_index}"),
                    "executed_at": datetime.now().isoformat(),
                    "result": step_result,
                    "status": "completed"
                })

                # Check for conditional branching
                if "condition" in step:
                    condition_result = self._evaluate_condition(workflow, step["condition"])
                    if not condition_result:
                        if "alternative_step" in step:
                            workflow["current_step"] = step["alternative_step"]
                            continue
                        else:
                            break

            workflow["status"] = WorkflowStatus.COMPLETED
            workflow["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            workflow["status"] = WorkflowStatus.FAILED
            workflow["error_message"] = str(e)
            workflow["completed_at"] = datetime.now().isoformat()

        finally:
            if instance_id in self.running_tasks:
                del self.running_tasks[instance_id]

            if workflow["status"] == WorkflowStatus.COMPLETED:
                self.completed_workflows.append(workflow)

        return workflow

    async def _execute_step(self, workflow: Dict[str, Any], step: Dict[str, Any]) -> Any:
        """Execute a single workflow step"""
        step_type = step.get("type", "task")

        if step_type == "task":
            return await self._execute_task_step(workflow, step)
        elif step_type == "decision":
            return self._execute_decision_step(workflow, step)
        elif step_type == "parallel":
            return await self._execute_parallel_step(workflow, step)
        elif step_type == "subworkflow":
            return await self._execute_subworkflow_step(workflow, step)
        else:
            raise ValueError(f"Unknown step type: {step_type}")

    async def _execute_task_step(self, workflow: Dict[str, Any], step: Dict[str, Any]) -> Any:
        """Execute a task step"""
        task_name = step.get("task", "")
        parameters = step.get("parameters", {})

        # Resolve parameters with workflow context
        resolved_params = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                # Variable reference
                var_name = value[1:]
                resolved_params[key] = workflow["variables"].get(var_name, value)
            else:
                resolved_params[key] = value

        # Simulate task execution (would integrate with actual task handlers)
        if task_name == "send_notification":
            return await self._send_notification(resolved_params)
        elif task_name == "update_patient_record":
            return await self._update_patient_record(resolved_params)
        elif task_name == "schedule_appointment":
            return await self._schedule_appointment(resolved_params)
        else:
            # Generic task execution
            await asyncio.sleep(0.1)  # Simulate async operation
            return {"status": "completed", "task": task_name, "parameters": resolved_params}

    def _execute_decision_step(self, workflow: Dict[str, Any], step: Dict[str, Any]) -> Any:
        """Execute a decision step"""
        condition = step.get("condition", "")
        return self._evaluate_condition(workflow, condition)

    async def _execute_parallel_step(self, workflow: Dict[str, Any], step: Dict[str, Any]) -> Any:
        """Execute parallel steps"""
        parallel_tasks = step.get("tasks", [])

        # Execute all tasks concurrently
        tasks = [self._execute_step(workflow, task) for task in parallel_tasks]
        results = await asyncio.gather(*tasks)

        return {"parallel_results": results}

    async def _execute_subworkflow_step(self, workflow: Dict[str, Any], step: Dict[str, Any]) -> Any:
        """Execute a subworkflow"""
        subworkflow_id = step.get("subworkflow_id", "")
        subworkflow_context = step.get("context", {})

        # Start and execute subworkflow
        subworkflow_instance_id = self.start_workflow(subworkflow_id, subworkflow_context)
        subworkflow_result = await self.execute_workflow(subworkflow_instance_id)

        return {"subworkflow_result": subworkflow_result}

    def _evaluate_condition(self, workflow: Dict[str, Any], condition: str) -> bool:
        """Evaluate a workflow condition"""
        # Simple condition evaluation (would use more sophisticated expression evaluator)
        if condition.startswith("variable."):
            var_path = condition[9:]  # Remove "variable."
            var_parts = var_path.split(".")
            value = workflow["variables"]

            for part in var_parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return False

            return bool(value)
        elif condition == "always_true":
            return True
        elif condition == "always_false":
            return False

        return False

    async def _send_notification(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification task"""
        await asyncio.sleep(0.05)  # Simulate async operation
        return {
            "notification_sent": True,
            "recipient": params.get("recipient"),
            "message": params.get("message"),
            "channel": params.get("channel", "email")
        }

    async def _update_patient_record(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update patient record task"""
        await asyncio.sleep(0.1)  # Simulate database operation
        return {
            "record_updated": True,
            "patient_id": params.get("patient_id"),
            "updates": params.get("updates", {})
        }

    async def _schedule_appointment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule appointment task"""
        await asyncio.sleep(0.08)  # Simulate scheduling operation
        return {
            "appointment_scheduled": True,
            "patient_id": params.get("patient_id"),
            "provider_id": params.get("provider_id"),
            "appointment_time": params.get("appointment_time")
        }

    def get_workflow_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        return self.workflows.get(instance_id)

    def list_workflows(self, status_filter: Optional[WorkflowStatus] = None) -> List[Dict[str, Any]]:
        """List workflows with optional status filter"""
        workflows = list(self.workflows.values())

        if status_filter:
            workflows = [w for w in workflows if w["status"] == status_filter]

        return workflows

class CareCoordinationEngine:
    """Care coordination and patient journey management"""

    def __init__(self):
        self.care_plans = {}
        self.patient_journeys = {}
        self.care_team_assignments = defaultdict(list)
        self.workflow_engine = WorkflowEngine()

    def create_care_plan(self, patient_id: str, care_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive care plan"""
        care_plan_id = f"care_{patient_id}_{int(datetime.now().timestamp())}"

        care_plan = {
            "care_plan_id": care_plan_id,
            "patient_id": patient_id,
            "diagnosis": care_plan_data.get("diagnosis", []),
            "goals": care_plan_data.get("goals", []),
            "interventions": care_plan_data.get("interventions", []),
            "care_team": care_plan_data.get("care_team", []),
            "timeline": care_plan_data.get("timeline", {}),
            "milestones": care_plan_data.get("milestones", []),
            "monitoring_schedule": care_plan_data.get("monitoring_schedule", {}),
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

        self.care_plans[care_plan_id] = care_plan

        # Assign care team members
        for team_member in care_plan["care_team"]:
            self.care_team_assignments[team_member["provider_id"]].append(care_plan_id)

        return care_plan

    def update_care_plan(self, care_plan_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update care plan with new information"""
        if care_plan_id not in self.care_plans:
            raise ValueError(f"Care plan not found: {care_plan_id}")

        care_plan = self.care_plans[care_plan_id]

        # Apply updates
        for key, value in updates.items():
            if key in care_plan:
                care_plan[key] = value

        care_plan["last_updated"] = datetime.now().isoformat()

        return care_plan

    def track_patient_journey(self, patient_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        """Track patient journey events"""
        if patient_id not in self.patient_journeys:
            self.patient_journeys[patient_id] = {
                "patient_id": patient_id,
                "events": [],
                "current_phase": "initial_assessment",
                "journey_start": datetime.now().isoformat()
            }

        journey = self.patient_journeys[patient_id]

        journey_event = {
            "event_id": f"event_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}",
            "event_type": event.get("type", "unknown"),
            "description": event.get("description", ""),
            "timestamp": datetime.now().isoformat(),
            "data": event.get("data", {}),
            "phase": event.get("phase", journey["current_phase"])
        }

        journey["events"].append(journey_event)
        journey["current_phase"] = journey_event["phase"]
        journey["last_updated"] = datetime.now().isoformat()

        # Update care plan milestones if applicable
        self._update_care_plan_milestones(patient_id, journey_event)

        return journey_event

    def _update_care_plan_milestones(self, patient_id: str, event: Dict[str, Any]) -> None:
        """Update care plan milestones based on patient journey events"""
        # Find active care plans for patient
        patient_care_plans = [
            cp for cp in self.care_plans.values()
            if cp["patient_id"] == patient_id and cp["status"] == "active"
        ]

        for care_plan in patient_care_plans:
            for milestone in care_plan.get("milestones", []):
                if milestone.get("event_type") == event["event_type"]:
                    milestone["status"] = "completed"
                    milestone["completed_at"] = event["timestamp"]
                    care_plan["last_updated"] = datetime.now().isoformat()

    def coordinate_care_team(self, patient_id: str, coordination_request: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate care team activities"""
        coordination_id = f"coord_{int(datetime.now().timestamp())}"

        coordination = {
            "coordination_id": coordination_id,
            "patient_id": patient_id,
            "request_type": coordination_request.get("type", "consultation"),
            "requester": coordination_request.get("requester"),
            "recipients": coordination_request.get("recipients", []),
            "subject": coordination_request.get("subject", ""),
            "message": coordination_request.get("message", ""),
            "priority": coordination_request.get("priority", "normal"),
            "status": "pending",
            "responses": [],
            "created_at": datetime.now().isoformat()
        }

        # Start workflow for care coordination
        workflow_context = {
            "coordination_id": coordination_id,
            "patient_id": patient_id,
            "coordination_type": coordination["request_type"]
        }

        workflow_instance_id = self.workflow_engine.start_workflow("care_coordination", workflow_context)

        coordination["workflow_instance_id"] = workflow_instance_id

        return coordination

class AppointmentSchedulingEngine:
    """Intelligent appointment scheduling system"""

    def __init__(self):
        self.appointments = {}
        self.provider_schedules = defaultdict(dict)
        self.patient_appointments = defaultdict(list)
        self.waiting_list = deque()

    def schedule_appointment(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a new appointment with intelligent slot selection"""
        patient_id = appointment_data["patient_id"]
        provider_id = appointment_data["provider_id"]
        appointment_type = appointment_data.get("appointment_type", "consultation")
        duration_minutes = appointment_data.get("duration_minutes", 30)

        # Find available slot
        preferred_date = appointment_data.get("preferred_date")
        available_slot = self._find_available_slot(
            provider_id, preferred_date, duration_minutes, appointment_type
        )

        if not available_slot:
            # Add to waiting list
            waiting_entry = {
                "waiting_id": f"wait_{int(datetime.now().timestamp())}",
                "patient_id": patient_id,
                "provider_id": provider_id,
                "appointment_type": appointment_type,
                "duration_minutes": duration_minutes,
                "requested_at": datetime.now().isoformat(),
                "priority": appointment_data.get("priority", "normal")
            }
            self.waiting_list.append(waiting_entry)

            return {
                "status": "waiting_list",
                "waiting_id": waiting_entry["waiting_id"],
                "estimated_wait_days": self._estimate_wait_time(provider_id, appointment_type)
            }

        # Create appointment
        appointment_id = f"appt_{int(datetime.now().timestamp())}"

        appointment = {
            "appointment_id": appointment_id,
            "patient_id": patient_id,
            "provider_id": provider_id,
            "appointment_type": appointment_type,
            "scheduled_time": available_slot["start_time"],
            "duration_minutes": duration_minutes,
            "end_time": available_slot["end_time"],
            "status": "scheduled",
            "notes": appointment_data.get("notes", ""),
            "created_at": datetime.now().isoformat(),
            "reminders_sent": [],
            "check_in_time": None,
            "check_out_time": None
        }

        self.appointments[appointment_id] = appointment
        self.patient_appointments[patient_id].append(appointment_id)

        # Block the time slot
        self._block_time_slot(provider_id, available_slot["start_time"], available_slot["end_time"])

        return appointment

    def _find_available_slot(self, provider_id: str, preferred_date: Optional[str],
                           duration_minutes: int, appointment_type: str) -> Optional[Dict[str, Any]]:
        """Find available time slot for appointment"""
        # Get provider schedule
        provider_schedule = self.provider_schedules.get(provider_id, self._get_default_schedule())

        # Determine search dates
        if preferred_date:
            search_dates = [datetime.fromisoformat(preferred_date).date()]
        else:
            # Search next 7 days
            today = datetime.now().date()
            search_dates = [today + timedelta(days=i) for i in range(7)]

        for search_date in search_dates:
            if search_date.weekday() >= 5:  # Skip weekends
                continue

            # Check provider availability for this date
            day_schedule = provider_schedule.get(search_date.weekday(), [])

            for time_slot in day_schedule:
                start_time = datetime.combine(search_date, time_slot["start"])
                end_time = datetime.combine(search_date, time_slot["end"])

                # Check if slot is available
                if self._is_slot_available(provider_id, start_time, end_time, duration_minutes):
                    return {
                        "start_time": start_time.isoformat(),
                        "end_time": (start_time + timedelta(minutes=duration_minutes)).isoformat()
                    }

        return None

    def _get_default_schedule(self) -> Dict[int, List[Dict[str, Any]]]:
        """Get default provider schedule"""
        # Monday to Friday, 9 AM to 5 PM
        default_times = [
            {"start": datetime.strptime("09:00", "%H:%M").time(),
             "end": datetime.strptime("17:00", "%H:%M").time()}
        ]

        return {i: default_times for i in range(5)}  # Monday to Friday

    def _is_slot_available(self, provider_id: str, start_time: datetime,
                          end_time: datetime, duration: int) -> bool:
        """Check if time slot is available"""
        # Check existing appointments
        for appointment in self.appointments.values():
            if (appointment["provider_id"] == provider_id and
                appointment["status"] in ["scheduled", "confirmed"]):

                appt_start = datetime.fromisoformat(appointment["scheduled_time"])
                appt_end = datetime.fromisoformat(appointment["end_time"])

                # Check for overlap
                if (start_time < appt_end and end_time > appt_start):
                    return False

        return True

    def _block_time_slot(self, provider_id: str, start_time: str, end_time: str) -> None:
        """Block time slot in provider schedule"""
        # This would update the provider's calendar system
        pass

    def _estimate_wait_time(self, provider_id: str, appointment_type: str) -> int:
        """Estimate wait time for appointment type"""
        # Simple estimation based on waiting list length
        waiting_count = len([w for w in self.waiting_list if w["provider_id"] == provider_id])
        return min(waiting_count * 2, 14)  # Max 2 weeks

    def reschedule_appointment(self, appointment_id: str, new_time: str) -> Dict[str, Any]:
        """Reschedule existing appointment"""
        if appointment_id not in self.appointments:
            raise ValueError(f"Appointment not found: {appointment_id}")

        appointment = self.appointments[appointment_id]

        # Check if new time is available
        new_start = datetime.fromisoformat(new_time)
        new_end = new_start + timedelta(minutes=appointment["duration_minutes"])

        if not self._is_slot_available(appointment["provider_id"], new_start, new_end,
                                    appointment["duration_minutes"]):
            return {"error": "New time slot not available"}

        # Update appointment
        old_start = appointment["scheduled_time"]
        old_end = appointment["end_time"]

        appointment["scheduled_time"] = new_time
        appointment["end_time"] = new_end.isoformat()
        appointment["status"] = "rescheduled"
        appointment["last_updated"] = datetime.now().isoformat()

        # Free old slot and block new slot
        # (In real implementation, would update calendar system)

        return appointment

    def cancel_appointment(self, appointment_id: str, reason: str = "") -> Dict[str, Any]:
        """Cancel appointment"""
        if appointment_id not in self.appointments:
            raise ValueError(f"Appointment not found: {appointment_id}")

        appointment = self.appointments[appointment_id]
        appointment["status"] = "cancelled"
        appointment["cancelled_at"] = datetime.now().isoformat()
        appointment["cancellation_reason"] = reason

        # Free up the time slot
        # (In real implementation, would update calendar system)

        # Check waiting list for potential fill
        self._check_waiting_list(appointment["provider_id"], appointment["appointment_type"])

        return appointment

    def _check_waiting_list(self, provider_id: str, appointment_type: str) -> None:
        """Check waiting list for appointment fill opportunities"""
        # Find waiting patients for this provider and appointment type
        waiting_patients = [
            w for w in self.waiting_list
            if w["provider_id"] == provider_id and w["appointment_type"] == appointment_type
        ]

        if waiting_patients:
            # Offer appointment to first waiting patient
            waiting_patient = waiting_patients[0]
            self.waiting_list.remove(waiting_patient)

            # Schedule appointment for waiting patient
            appointment_data = {
                "patient_id": waiting_patient["patient_id"],
                "provider_id": provider_id,
                "appointment_type": appointment_type,
                "duration_minutes": waiting_patient["duration_minutes"],
                "priority": "high"  # From waiting list
            }

            self.schedule_appointment(appointment_data)

    def get_provider_schedule(self, provider_id: str, date: str) -> Dict[str, Any]:
        """Get provider schedule for specific date"""
        target_date = datetime.fromisoformat(date).date()

        # Get appointments for this date
        day_appointments = [
            appt for appt in self.appointments.values()
            if (appt["provider_id"] == provider_id and
                datetime.fromisoformat(appt["scheduled_time"]).date() == target_date and
                appt["status"] in ["scheduled", "confirmed"])
        ]

        # Get provider's working hours
        provider_schedule = self.provider_schedules.get(provider_id, self._get_default_schedule())
        day_schedule = provider_schedule.get(target_date.weekday(), [])

        return {
            "provider_id": provider_id,
            "date": date,
            "working_hours": day_schedule,
            "appointments": day_appointments,
            "available_slots": self._calculate_available_slots(day_schedule, day_appointments)
        }

    def _calculate_available_slots(self, working_hours: List[Dict[str, Any]],
                                 appointments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate available time slots"""
        available_slots = []

        for work_period in working_hours:
            current_time = work_period["start"]

            while current_time < work_period["end"]:
                slot_end = (datetime.combine(datetime.now().date(), current_time) +
                           timedelta(minutes=30)).time()

                # Check if this slot conflicts with any appointment
                conflict = False
                for appt in appointments:
                    appt_start = datetime.fromisoformat(appt["scheduled_time"]).time()
                    appt_end = datetime.fromisoformat(appt["end_time"]).time()

                    if (current_time < appt_end and slot_end > appt_start):
                        conflict = True
                        break

                if not conflict:
                    available_slots.append({
                        "start_time": current_time.strftime("%H:%M"),
                        "end_time": slot_end.strftime("%H:%M"),
                        "duration_minutes": 30
                    })

                current_time = slot_end

        return available_slots

class ClinicalWorkflowEngine:
    """Clinical workflow management for specific medical processes"""

    def __init__(self):
        self.workflow_templates = self._initialize_workflow_templates()
        self.workflow_engine = WorkflowEngine()

    def _initialize_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize clinical workflow templates"""
        return {
            "diabetes_management": {
                "name": "Diabetes Management Workflow",
                "description": "Comprehensive diabetes care coordination",
                "steps": [
                    {
                        "name": "Initial Assessment",
                        "type": "task",
                        "task": "schedule_appointment",
                        "parameters": {
                            "appointment_type": "diabetes_consultation",
                            "duration_minutes": 45
                        }
                    },
                    {
                        "name": "Diagnostic Testing",
                        "type": "task",
                        "task": "order_laboratory_test",
                        "parameters": {
                            "test_code": "cmp",
                            "priority": "routine"
                        }
                    },
                    {
                        "name": "Treatment Planning",
                        "type": "decision",
                        "condition": "variable.diagnosis_confirmed",
                        "alternative_step": 4
                    },
                    {
                        "name": "Medication Management",
                        "type": "task",
                        "task": "create_prescription",
                        "parameters": {
                            "medication": "metformin",
                            "dosage": "500mg twice daily"
                        }
                    },
                    {
                        "name": "Patient Education",
                        "type": "task",
                        "task": "send_notification",
                        "parameters": {
                            "channel": "patient_portal",
                            "message": "Diabetes management education materials"
                        }
                    }
                ]
            },
            "cardiac_care": {
                "name": "Cardiac Care Workflow",
                "description": "Cardiovascular disease management",
                "steps": [
                    {
                        "name": "Emergency Assessment",
                        "type": "task",
                        "task": "schedule_appointment",
                        "parameters": {
                            "appointment_type": "cardiology_consultation",
                            "duration_minutes": 60,
                            "priority": "urgent"
                        }
                    },
                    {
                        "name": "Cardiac Testing",
                        "type": "parallel",
                        "tasks": [
                            {
                                "name": "ECG",
                                "type": "task",
                                "task": "order_cardiac_test",
                                "parameters": {"test_type": "ecg"}
                            },
                            {
                                "name": "Echocardiogram",
                                "type": "task",
                                "task": "order_cardiac_test",
                                "parameters": {"test_type": "echo"}
                            }
                        ]
                    },
                    {
                        "name": "Risk Stratification",
                        "type": "task",
                        "task": "calculate_cardiac_risk",
                        "parameters": {"use_advanced_model": True}
                    }
                ]
            }
        }

    def start_clinical_workflow(self, workflow_type: str, patient_data: Dict[str, Any]) -> str:
        """Start a clinical workflow for a patient"""
        if workflow_type not in self.workflow_templates:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        template = self.workflow_templates[workflow_type]

        # Register workflow definition
        self.workflow_engine.define_workflow(workflow_type, template)

        # Start workflow instance
        context = {
            "patient_id": patient_data.get("patient_id"),
            "diagnosis": patient_data.get("diagnosis", []),
            "priority": patient_data.get("priority", WorkflowPriority.NORMAL),
            "clinical_data": patient_data.get("clinical_data", {})
        }

        return self.workflow_engine.start_workflow(workflow_type, context)

    async def execute_clinical_workflow(self, instance_id: str) -> Dict[str, Any]:
        """Execute clinical workflow"""
        return await self.workflow_engine.execute_workflow(instance_id)

    def get_clinical_workflow_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get clinical workflow status"""
        return self.workflow_engine.get_workflow_status(instance_id)

class NotificationEngine:
    """Intelligent notification system for healthcare workflows"""

    def __init__(self):
        self.notification_templates = self._initialize_templates()
        self.notification_queue = deque()
        self.sent_notifications = []

    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize notification templates"""
        return {
            "appointment_reminder": {
                "channels": ["email", "sms", "push"],
                "template": {
                    "subject": "Appointment Reminder",
                    "message": "You have an appointment with {provider_name} on {appointment_date} at {appointment_time}"
                },
                "timing": ["24_hours_before", "1_hour_before"]
            },
            "test_results_available": {
                "channels": ["email", "patient_portal"],
                "template": {
                    "subject": "Test Results Available",
                    "message": "Your {test_name} results are now available. Please log in to view them."
                },
                "priority": "high"
            },
            "medication_reminder": {
                "channels": ["sms", "push"],
                "template": {
                    "subject": "Medication Reminder",
                    "message": "Time to take your {medication_name} ({dosage})"
                },
                "timing": ["scheduled_time"]
            },
            "care_coordination": {
                "channels": ["email", "secure_message"],
                "template": {
                    "subject": "Care Coordination Update",
                    "message": "Update regarding {patient_name}'s care: {message}"
                },
                "priority": "high"
            }
        }

    def schedule_notification(self, notification_type: str, recipient: str,
                            context: Dict[str, Any], timing: str = "immediate") -> str:
        """Schedule a notification"""
        if notification_type not in self.notification_templates:
            raise ValueError(f"Unknown notification type: {notification_type}")

        template = self.notification_templates[notification_type]

        notification_id = f"notif_{int(datetime.now().timestamp())}"

        notification = {
            "notification_id": notification_id,
            "type": notification_type,
            "recipient": recipient,
            "context": context,
            "template": template,
            "timing": timing,
            "status": "scheduled",
            "created_at": datetime.now().isoformat(),
            "scheduled_for": self._calculate_schedule_time(timing),
            "channels": template["channels"],
            "priority": template.get("priority", "normal")
        }

        self.notification_queue.append(notification)

        return notification_id

    def _calculate_schedule_time(self, timing: str) -> Optional[str]:
        """Calculate when notification should be sent"""
        now = datetime.now()

        if timing == "immediate":
            return now.isoformat()
        elif timing == "24_hours_before":
            return (now + timedelta(hours=24)).isoformat()
        elif timing == "1_hour_before":
            return (now + timedelta(hours=1)).isoformat()
        elif timing == "scheduled_time":
            return now.isoformat()  # Would be calculated based on medication schedule

        return now.isoformat()

    async def process_notifications(self) -> List[Dict[str, Any]]:
        """Process pending notifications"""
        sent_notifications = []

        # Get notifications ready to send
        ready_notifications = []
        current_time = datetime.now()

        for notification in self.notification_queue:
            scheduled_time = datetime.fromisoformat(notification["scheduled_for"])
            if current_time >= scheduled_time and notification["status"] == "scheduled":
                ready_notifications.append(notification)

        # Send notifications
        for notification in ready_notifications:
            await self._send_notification(notification)
            notification["status"] = "sent"
            notification["sent_at"] = datetime.now().isoformat()
            sent_notifications.append(notification)
            self.sent_notifications.append(notification)

        # Remove sent notifications from queue
        for sent in sent_notifications:
            try:
                self.notification_queue.remove(sent)
            except ValueError:
                pass  # Already removed

        return sent_notifications

    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """Send notification through available channels"""
        template = notification["template"]
        context = notification["context"]

        # Format message
        subject = template["template"]["subject"].format(**context)
        message = template["template"]["message"].format(**context)

        # Send through each channel (simplified)
        for channel in notification["channels"]:
            await self._send_channel_notification(channel, notification["recipient"], subject, message)

    async def _send_channel_notification(self, channel: str, recipient: str,
                                       subject: str, message: str) -> None:
        """Send notification through specific channel"""
        # Simulate sending (would integrate with actual services)
        await asyncio.sleep(0.01)

        print(f"Sent {channel} notification to {recipient}: {subject}")

    def get_notification_history(self, recipient: str = None,
                               notification_type: str = None) -> List[Dict[str, Any]]:
        """Get notification history with optional filters"""
        notifications = self.sent_notifications

        if recipient:
            notifications = [n for n in notifications if n["recipient"] == recipient]

        if notification_type:
            notifications = [n for n in notifications if n["type"] == notification_type]

        return notifications
