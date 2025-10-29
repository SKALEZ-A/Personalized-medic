"""
GraphQL API for AI Personalized Medicine Platform
Advanced GraphQL schema with real-time subscriptions and complex queries
"""

import graphene
from graphene import ObjectType, String, Int, Float, Boolean, List, Field, ID, DateTime, JSONString
from graphene.relay import Node, Connection, ConnectionField
from graphene_mongo import MongoengineObjectType
import graphql_jwt
from graphql_jwt.decorators import login_required, superuser_required
from graphql_jwt.refresh_token.decorators import refresh_token_required
import asyncio
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
import threading
import time

# Mock data models (would be replaced with actual database models)
class Patient(graphene.ObjectType):
    id = ID(required=True)
    patient_id = String(required=True)
    name = String(required=True)
    age = Int()
    gender = String()
    medical_history = List(String)
    current_medications = List(String)
    allergies = List(String)
    last_visit = DateTime()
    risk_score = Float()
    genomic_data = JSONString()
    created_at = DateTime()
    updated_at = DateTime()

class HealthMetric(graphene.ObjectType):
    id = ID(required=True)
    patient_id = String(required=True)
    metric_type = String(required=True)
    value = Float(required=True)
    unit = String()
    timestamp = DateTime(required=True)
    device_id = String()
    quality_score = Float()

class Appointment(graphene.ObjectType):
    id = ID(required=True)
    patient_id = String(required=True)
    doctor_id = String(required=True)
    appointment_type = String(required=True)
    scheduled_time = DateTime(required=True)
    duration_minutes = Int()
    status = String(required=True)
    notes = String()
    location = String()

class Medication(graphene.ObjectType):
    id = ID(required=True)
    patient_id = String(required=True)
    name = String(required=True)
    dosage = String(required=True)
    frequency = String(required=True)
    start_date = DateTime(required=True)
    end_date = DateTime()
    prescribed_by = String(required=True)
    side_effects = List(String)

class GenomicAnalysis(graphene.ObjectType):
    id = ID(required=True)
    patient_id = String(required=True)
    analysis_type = String(required=True)
    results = JSONString(required=True)
    confidence_score = Float()
    risk_assessment = JSONString()
    recommendations = List(String)
    analyzed_at = DateTime(required=True)
    analyst_id = String()

# Query Resolvers
class Query(ObjectType):
    """Root Query for GraphQL API"""

    # Patient queries
    patient = Field(Patient, patient_id=String(required=True))
    patients = List(Patient,
                   limit=Int(default_value=50),
                   offset=Int(default_value=0),
                   search=String(),
                   risk_level=String())

    # Health metrics queries
    health_metrics = List(HealthMetric,
                         patient_id=String(required=True),
                         metric_type=String(),
                         start_date=DateTime(),
                         end_date=DateTime(),
                         limit=Int(default_value=100))

    # Appointment queries
    appointments = List(Appointment,
                       patient_id=String(),
                       doctor_id=String(),
                       status=String(),
                       start_date=DateTime(),
                       end_date=DateTime())

    # Medication queries
    medications = List(Medication,
                      patient_id=String(required=True),
                      active_only=Boolean(default_value=True))

    # Genomic analysis queries
    genomic_analyses = List(GenomicAnalysis,
                           patient_id=String(required=True),
                           analysis_type=String())

    # Dashboard queries
    patient_dashboard = Field(lambda: PatientDashboard,
                             patient_id=String(required=True))

    # Analytics queries
    health_analytics = Field(lambda: HealthAnalytics,
                            patient_id=String(required=True),
                            time_range=String(default_value="30d"))

    @staticmethod
    def resolve_patient(root, info, patient_id):
        """Resolve single patient query"""
        # Mock patient data - would query actual database
        return Patient(
            id=f"patient_{patient_id}",
            patient_id=patient_id,
            name="John Doe",
            age=45,
            gender="Male",
            medical_history=["Hypertension", "Type 2 Diabetes"],
            current_medications=["Lisinopril 10mg", "Metformin 500mg"],
            allergies=["Penicillin"],
            last_visit=datetime.now() - timedelta(days=7),
            risk_score=0.65,
            genomic_data=json.dumps({"variants": ["BRCA1", "APOE4"]}),
            created_at=datetime.now() - timedelta(days=365),
            updated_at=datetime.now() - timedelta(days=1)
        )

    @staticmethod
    def resolve_patients(root, info, limit=50, offset=0, search=None, risk_level=None):
        """Resolve patients list query"""
        # Mock patients data - would query actual database with filtering
        patients = []

        # Generate mock patients
        for i in range(min(limit, 100)):  # Max 100 for demo
            patient_id = f"PAT{i+offset+1:04d}"
            patients.append(Patient(
                id=f"patient_{patient_id}",
                patient_id=patient_id,
                name=f"Patient {i+offset+1}",
                age=30 + (i % 50),
                gender="Male" if i % 2 == 0 else "Female",
                medical_history=["Condition A", "Condition B"][:i % 3],
                current_medications=["Med A", "Med B"][:i % 2],
                allergies=[],
                last_visit=datetime.now() - timedelta(days=i % 30),
                risk_score=0.1 + (i % 9) * 0.1,
                genomic_data=json.dumps({"status": "analyzed"}),
                created_at=datetime.now() - timedelta(days=365 - i),
                updated_at=datetime.now() - timedelta(days=i % 10)
            ))

        # Apply filters
        if search:
            patients = [p for p in patients if search.lower() in p.name.lower() or search in p.patient_id]

        if risk_level:
            if risk_level == "high":
                patients = [p for p in patients if p.risk_score > 0.7]
            elif risk_level == "medium":
                patients = [p for p in patients if 0.4 <= p.risk_score <= 0.7]
            elif risk_level == "low":
                patients = [p for p in patients if p.risk_score < 0.4]

        return patients[offset:offset + limit]

    @staticmethod
    def resolve_health_metrics(root, info, patient_id, metric_type=None,
                             start_date=None, end_date=None, limit=100):
        """Resolve health metrics query"""
        # Mock health metrics data
        metrics = []
        base_time = datetime.now()

        for i in range(min(limit, 1000)):
            timestamp = base_time - timedelta(hours=i)
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue

            metric_types = ["heart_rate", "blood_pressure", "blood_glucose", "weight", "temperature"]
            selected_type = metric_type or metric_types[i % len(metric_types)]

            value = {
                "heart_rate": 60 + (i % 40),
                "blood_pressure": 120 + (i % 20),
                "blood_glucose": 80 + (i % 40),
                "weight": 70 + (i % 10),
                "temperature": 36.5 + (i % 2)
            }[selected_type]

            unit = {
                "heart_rate": "bpm",
                "blood_pressure": "mmHg",
                "blood_glucose": "mg/dL",
                "weight": "kg",
                "temperature": "°C"
            }[selected_type]

            metrics.append(HealthMetric(
                id=f"metric_{patient_id}_{i}",
                patient_id=patient_id,
                metric_type=selected_type,
                value=float(value),
                unit=unit,
                timestamp=timestamp,
                device_id=f"device_{i % 5}",
                quality_score=0.8 + (i % 3) * 0.1
            ))

        return metrics

    @staticmethod
    def resolve_appointments(root, info, patient_id=None, doctor_id=None,
                           status=None, start_date=None, end_date=None):
        """Resolve appointments query"""
        # Mock appointments data
        appointments = []
        base_time = datetime.now()

        statuses = ["scheduled", "completed", "cancelled", "no_show"]
        types = ["Consultation", "Follow-up", "Check-up", "Specialist Visit"]

        for i in range(20):
            apt_time = base_time + timedelta(days=i % 14)
            if start_date and apt_time < start_date:
                continue
            if end_date and apt_time > end_date:
                continue

            apt_patient_id = patient_id or f"PAT{(i % 10)+1:04d}"
            apt_doctor_id = doctor_id or f"DR{(i % 5)+1:03d}"
            apt_status = status or statuses[i % len(statuses)]

            appointments.append(Appointment(
                id=f"apt_{i}",
                patient_id=apt_patient_id,
                doctor_id=apt_doctor_id,
                appointment_type=types[i % len(types)],
                scheduled_time=apt_time,
                duration_minutes=30 + (i % 3) * 15,
                status=apt_status,
                notes=f"Follow-up appointment {i+1}" if i % 3 == 0 else None,
                location="Main Clinic"
            ))

        return appointments

    @staticmethod
    def resolve_medications(root, info, patient_id, active_only=True):
        """Resolve medications query"""
        # Mock medications data
        medications = [
            Medication(
                id=f"med_{patient_id}_1",
                patient_id=patient_id,
                name="Lisinopril",
                dosage="10mg",
                frequency="Once daily",
                start_date=datetime.now() - timedelta(days=30),
                end_date=None,
                prescribed_by="Dr. Smith",
                side_effects=[]
            ),
            Medication(
                id=f"med_{patient_id}_2",
                patient_id=patient_id,
                name="Metformin",
                dosage="500mg",
                frequency="Twice daily",
                start_date=datetime.now() - timedelta(days=60),
                end_date=datetime.now() + timedelta(days=180),
                prescribed_by="Dr. Johnson",
                side_effects=["Nausea", "Diarrhea"]
            )
        ]

        if active_only:
            medications = [m for m in medications if m.end_date is None or m.end_date > datetime.now()]

        return medications

    @staticmethod
    def resolve_genomic_analyses(root, info, patient_id, analysis_type=None):
        """Resolve genomic analyses query"""
        # Mock genomic analysis data
        analyses = [
            GenomicAnalysis(
                id=f"gen_{patient_id}_1",
                patient_id=patient_id,
                analysis_type="comprehensive_panel",
                results=json.dumps({
                    "variants": ["BRCA1", "APOE4"],
                    "risk_factors": ["Cardiovascular disease", "Alzheimer's"],
                    "confidence": 0.92
                }),
                confidence_score=0.92,
                risk_assessment=json.dumps({
                    "overall_risk": "moderate",
                    "lifetime_risks": {
                        "cancer": 0.15,
                        "cardiovascular": 0.25,
                        "diabetes": 0.35
                    }
                }),
                recommendations=[
                    "Regular cardiovascular screening",
                    "Lifestyle modifications",
                    "Genetic counseling"
                ],
                analyzed_at=datetime.now() - timedelta(days=7),
                analyst_id="GEN001"
            )
        ]

        if analysis_type:
            analyses = [a for a in analyses if a.analysis_type == analysis_type]

        return analyses

    @staticmethod
    def resolve_patient_dashboard(root, info, patient_id):
        """Resolve patient dashboard query"""
        return PatientDashboard(patient_id=patient_id)

    @staticmethod
    def resolve_health_analytics(root, info, patient_id, time_range="30d"):
        """Resolve health analytics query"""
        return HealthAnalytics(patient_id=patient_id, time_range=time_range)

# Mutations
class CreatePatient(graphene.Mutation):
    """Create new patient mutation"""
    class Arguments:
        patient_id = String(required=True)
        name = String(required=True)
        age = Int()
        gender = String()
        medical_history = List(String)
        allergies = List(String)

    patient = Field(Patient)
    success = Boolean()
    message = String()

    @staticmethod
    def mutate(root, info, patient_id, name, age=None, gender=None,
              medical_history=None, allergies=None):
        # Mock patient creation - would save to database
        patient = Patient(
            id=f"patient_{patient_id}",
            patient_id=patient_id,
            name=name,
            age=age,
            gender=gender,
            medical_history=medical_history or [],
            current_medications=[],
            allergies=allergies or [],
            last_visit=None,
            risk_score=0.1,
            genomic_data=json.dumps({}),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        return CreatePatient(patient=patient, success=True, message="Patient created successfully")

class RecordHealthMetric(graphene.Mutation):
    """Record health metric mutation"""
    class Arguments:
        patient_id = String(required=True)
        metric_type = String(required=True)
        value = Float(required=True)
        unit = String()
        device_id = String()

    metric = Field(HealthMetric)
    success = Boolean()
    message = String()

    @staticmethod
    def mutate(root, info, patient_id, metric_type, value, unit=None, device_id=None):
        # Mock health metric recording - would save to database
        metric = HealthMetric(
            id=f"metric_{patient_id}_{int(time.time())}",
            patient_id=patient_id,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            device_id=device_id,
            quality_score=0.95
        )

        return RecordHealthMetric(metric=metric, success=True, message="Health metric recorded")

class ScheduleAppointment(graphene.Mutation):
    """Schedule appointment mutation"""
    class Arguments:
        patient_id = String(required=True)
        doctor_id = String(required=True)
        appointment_type = String(required=True)
        scheduled_time = DateTime(required=True)
        duration_minutes = Int(default_value=30)
        notes = String()

    appointment = Field(Appointment)
    success = Boolean()
    message = String()

    @staticmethod
    def mutate(root, info, patient_id, doctor_id, appointment_type,
              scheduled_time, duration_minutes=30, notes=None):
        # Mock appointment scheduling - would save to database
        appointment = Appointment(
            id=f"apt_{int(time.time())}",
            patient_id=patient_id,
            doctor_id=doctor_id,
            appointment_type=appointment_type,
            scheduled_time=scheduled_time,
            duration_minutes=duration_minutes,
            status="scheduled",
            notes=notes,
            location="Main Clinic"
        )

        return ScheduleAppointment(appointment=appointment, success=True, message="Appointment scheduled")

class Mutation(ObjectType):
    """Root Mutation for GraphQL API"""

    # Authentication mutations
    token_auth = graphql_jwt.ObtainJSONWebToken.Field()
    verify_token = graphql_jwt.Verify.Field()
    refresh_token = graphql_jwt.Refresh.Field()

    # Patient mutations
    create_patient = CreatePatient.Field()
    record_health_metric = RecordHealthMetric.Field()
    schedule_appointment = ScheduleAppointment.Field()

# Subscriptions (WebSocket support)
class Subscription(ObjectType):
    """Root Subscription for GraphQL API"""

    # Real-time health metrics
    health_metric_updates = Field(HealthMetric, patient_id=String(required=True))

    # Appointment updates
    appointment_updates = Field(Appointment, patient_id=String())

    # Alert notifications
    alert_notifications = Field(lambda: AlertNotification, patient_id=String())

    def resolve_health_metric_updates(root, info, patient_id):
        """Subscribe to real-time health metric updates"""
        # This would be implemented with a pub/sub system
        # For now, return mock data
        return HealthMetric(
            id=f"metric_{patient_id}_realtime",
            patient_id=patient_id,
            metric_type="heart_rate",
            value=72.0,
            unit="bpm",
            timestamp=datetime.now(),
            device_id="wearable_001",
            quality_score=0.98
        )

    def resolve_appointment_updates(root, info, patient_id=None):
        """Subscribe to appointment updates"""
        # Mock appointment update
        return Appointment(
            id="apt_realtime",
            patient_id=patient_id or "PAT0001",
            doctor_id="DR001",
            appointment_type="Follow-up",
            scheduled_time=datetime.now() + timedelta(hours=1),
            duration_minutes=30,
            status="confirmed",
            notes="Routine check-up",
            location="Main Clinic"
        )

    def resolve_alert_notifications(root, info, patient_id=None):
        """Subscribe to alert notifications"""
        return AlertNotification(
            id="alert_realtime",
            patient_id=patient_id or "PAT0001",
            alert_type="medication_reminder",
            message="Time to take your medication",
            severity="info",
            timestamp=datetime.now()
        )

# Additional GraphQL Types
class PatientDashboard(ObjectType):
    """Patient dashboard data"""
    patient_id = String(required=True)
    patient = Field(Patient)
    recent_metrics = List(HealthMetric)
    upcoming_appointments = List(Appointment)
    current_medications = List(Medication)
    health_score = Float()
    risk_assessment = JSONString()
    last_updated = DateTime()

    def resolve_patient(self, info):
        return Query.resolve_patient(None, info, self.patient_id)

    def resolve_recent_metrics(self, info):
        return Query.resolve_health_metrics(None, info, self.patient_id, limit=10)

    def resolve_upcoming_appointments(self, info):
        appointments = Query.resolve_appointments(None, info, patient_id=self.patient_id)
        return [a for a in appointments if a.status == "scheduled" and a.scheduled_time > datetime.now()][:5]

    def resolve_current_medications(self, info):
        return Query.resolve_medications(None, info, self.patient_id)

    def resolve_health_score(self, info):
        # Mock health score calculation
        return 0.75

    def resolve_risk_assessment(self, info):
        return json.dumps({
            "overall_risk": "moderate",
            "categories": {
                "cardiovascular": "low",
                "diabetes": "moderate",
                "cancer": "low"
            }
        })

    def resolve_last_updated(self, info):
        return datetime.now()

class HealthAnalytics(ObjectType):
    """Health analytics data"""
    patient_id = String(required=True)
    time_range = String()
    metric_summaries = List(lambda: MetricSummary)
    trends = JSONString()
    correlations = JSONString()
    insights = List(String)
    recommendations = List(String)

    def resolve_metric_summaries(self, info):
        metrics = Query.resolve_health_metrics(None, info, self.patient_id)
        summaries = {}

        for metric in metrics:
            if metric.metric_type not in summaries:
                values = [m.value for m in metrics if m.metric_type == metric.metric_type]
                summaries[metric.metric_type] = MetricSummary(
                    metric_type=metric.metric_type,
                    count=len(values),
                    average=sum(values) / len(values),
                    min_value=min(values),
                    max_value=max(values),
                    latest_value=values[0],
                    unit=metric.unit
                )

        return list(summaries.values())

    def resolve_trends(self, info):
        return json.dumps({
            "heart_rate": {"trend": "stable", "change": "+2%"},
            "blood_pressure": {"trend": "decreasing", "change": "-5%"},
            "weight": {"trend": "stable", "change": "0%"}
        })

    def resolve_correlations(self, info):
        return json.dumps({
            "heart_rate_blood_pressure": 0.65,
            "weight_blood_glucose": 0.45,
            "exercise_heart_rate": -0.78
        })

    def resolve_insights(self, info):
        return [
            "Your heart rate shows good recovery after exercise",
            "Blood pressure has improved by 5% this month",
            "Consistent medication adherence noted"
        ]

    def resolve_recommendations(self, info):
        return [
            "Continue current exercise regimen",
            "Monitor blood pressure weekly",
            "Schedule follow-up appointment in 2 weeks"
        ]

class MetricSummary(ObjectType):
    """Health metric summary"""
    metric_type = String(required=True)
    count = Int(required=True)
    average = Float(required=True)
    min_value = Float(required=True)
    max_value = Float(required=True)
    latest_value = Float(required=True)
    unit = String()

class AlertNotification(ObjectType):
    """Alert notification"""
    id = String(required=True)
    patient_id = String(required=True)
    alert_type = String(required=True)
    message = String(required=True)
    severity = String(required=True)
    timestamp = DateTime(required=True)
    metadata = JSONString()

# Create executable schema
schema = graphene.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    types=[PatientDashboard, HealthAnalytics, AlertNotification]
)

# GraphQL API class for integration
class GraphQLAPI:
    """GraphQL API handler"""

    def __init__(self):
        self.schema = schema
        self.subscriptions = {}
        self.query_cache = {}
        self.rate_limits = defaultdict(lambda: {"count": 0, "reset_time": datetime.now()})

    def execute_query(self, query: str, variables: Dict[str, Any] = None,
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute GraphQL query"""
        try:
            result = self.schema.execute(
                query,
                variables=variables,
                context=context
            )

            if result.errors:
                return {
                    "errors": [str(error) for error in result.errors],
                    "data": None
                }

            return {
                "data": result.data,
                "errors": None
            }

        except Exception as e:
            return {
                "errors": [str(e)],
                "data": None
            }

    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate GraphQL query"""
        try:
            # Parse and validate query
            document = graphene.parse(query)
            errors = graphene.validate(self.schema, document)

            return {
                "valid": len(errors) == 0,
                "errors": [str(error) for error in errors] if errors else []
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)]
            }

    def get_schema_introspection(self) -> Dict[str, Any]:
        """Get GraphQL schema introspection"""
        introspection_query = """
        query IntrospectionQuery {
          __schema {
            queryType { name }
            mutationType { name }
            subscriptionType { name }
            types {
              ...FullType
            }
            directives {
              name
              description
              locations
              args {
                ...InputValue
              }
            }
          }
        }

        fragment FullType on __Type {
          kind
          name
          description
          fields(includeDeprecated: true) {
            name
            description
            args {
              ...InputValue
            }
            type {
              ...TypeRef
            }
            isDeprecated
            deprecationReason
          }
          inputFields {
            ...InputValue
          }
          interfaces {
            ...TypeRef
          }
          enumValues(includeDeprecated: true) {
            name
            description
            isDeprecated
            deprecationReason
          }
          possibleTypes {
            ...TypeRef
          }
        }

        fragment InputValue on __InputValue {
          name
          description
          type { ...TypeRef }
          defaultValue
        }

        fragment TypeRef on __Type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                      ofType {
                        kind
                        name
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """

        return self.execute_query(introspection_query)

    def cache_query_result(self, query_hash: str, result: Dict[str, Any], ttl_seconds: int = 300):
        """Cache query result"""
        self.query_cache[query_hash] = {
            "result": result,
            "expires_at": datetime.now() + timedelta(seconds=ttl_seconds)
        }

    def get_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        if query_hash in self.query_cache:
            cached = self.query_cache[query_hash]
            if cached["expires_at"] > datetime.now():
                return cached["result"]
            else:
                del self.query_cache[query_hash]

        return None

    def check_rate_limit(self, client_id: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
        """Check rate limit for client"""
        now = datetime.now()
        client_limit = self.rate_limits[client_id]

        # Reset counter if window has passed
        if now > client_limit["reset_time"]:
            client_limit["count"] = 0
            client_limit["reset_time"] = now + timedelta(seconds=window_seconds)

        # Check if under limit
        if client_limit["count"] >= max_requests:
            return False

        client_limit["count"] += 1
        return True

# Export the GraphQL API instance
graphql_api = GraphQLAPI()
