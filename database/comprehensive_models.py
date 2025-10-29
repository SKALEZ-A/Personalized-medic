"""
Comprehensive Database Models for AI Personalized Medicine Platform
Complete SQLAlchemy models with relationships, constraints, and indexes
"""

import uuid
from datetime import datetime, date, time
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, Date, Time,
    ForeignKey, Table, Enum as SQLEnum, JSON, UUID, Numeric, Index,
    CheckConstraint, UniqueConstraint, func, text, event
)
from sqlalchemy.orm import relationship, backref, deferred, validates, Mapped
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID as PGUUID, TIMESTAMP, INTERVAL
import enum

Base = declarative_base()

# Enums
class UserRoleEnum(str, enum.Enum):
    PATIENT = "patient"
    PHYSICIAN = "physician"
    NURSE = "nurse"
    PHARMACIST = "pharmacist"
    RESEARCHER = "researcher"
    ADMIN = "admin"

class GenderEnum(str, enum.Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"

class BloodTypeEnum(str, enum.Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

class AppointmentStatusEnum(str, enum.Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"

class GenomicAnalysisStatusEnum(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class MedicationStatusEnum(str, enum.Enum):
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"

class LabResultStatusEnum(str, enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ImagingModalityEnum(str, enum.Enum):
    XRAY = "X-Ray"
    CT = "CT"
    MRI = "MRI"
    ULTRASOUND = "Ultrasound"
    MAMMOGRAPHY = "Mammography"
    PET = "PET"
    NUCLEAR = "Nuclear Medicine"

class ClinicalTrialPhaseEnum(str, enum.Enum):
    PHASE_0 = "0"
    PHASE_1 = "I"
    PHASE_2 = "II"
    PHASE_3 = "III"
    PHASE_4 = "IV"

class ClinicalTrialStatusEnum(str, enum.Enum):
    RECRUITING = "recruiting"
    NOT_RECRUITING = "not_recruiting"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

class AuditEventTypeEnum(str, enum.Enum):
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    MEDICATION_PRESCRIBED = "medication_prescribed"
    APPOINTMENT_SCHEDULED = "appointment_scheduled"
    GENOMIC_ANALYSIS_REQUESTED = "genomic_analysis_requested"
    SECURITY_EVENT = "security_event"
    SYSTEM_MAINTENANCE = "system_maintenance"

# Association Tables
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)

patient_allergies = Table(
    'patient_allergies',
    Base.metadata,
    Column('patient_id', Integer, ForeignKey('patients.id'), primary_key=True),
    Column('allergen_id', Integer, ForeignKey('allergens.id'), primary_key=True),
    Column('severity', String(20)),
    Column('reaction', Text),
    Column('notes', Text),
    Column('onset_date', Date),
    Column('created_at', DateTime, default=datetime.utcnow)
)

medication_allergies = Table(
    'medication_allergies',
    Base.metadata,
    Column('patient_id', Integer, ForeignKey('patients.id'), primary_key=True),
    Column('medication_id', Integer, ForeignKey('medications.id'), primary_key=True),
    Column('reaction', Text),
    Column('severity', String(20)),
    Column('created_at', DateTime, default=datetime.utcnow)
)

treatment_plan_medications = Table(
    'treatment_plan_medications',
    Base.metadata,
    Column('treatment_plan_id', Integer, ForeignKey('treatment_plans.id'), primary_key=True),
    Column('medication_id', Integer, ForeignKey('medications.id'), primary_key=True),
    Column('dosage', String(100)),
    Column('frequency', String(100)),
    Column('duration', String(100)),
    Column('instructions', Text),
    Column('start_date', Date),
    Column('end_date', Date, nullable=True),
    Column('created_at', DateTime, default=datetime.utcnow)
)

clinical_trial_eligibility = Table(
    'clinical_trial_eligibility',
    Base.metadata,
    Column('trial_id', Integer, ForeignKey('clinical_trials.id'), primary_key=True),
    Column('patient_id', Integer, ForeignKey('patients.id'), primary_key=True),
    Column('eligibility_score', Float),
    Column('matching_criteria', JSONB),
    Column('exclusion_criteria', JSONB),
    Column('assessed_at', DateTime, default=datetime.utcnow),
    Column('assessed_by', Integer, ForeignKey('users.id'))
)

# Main Models
class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    permissions = Column(JSONB, default=dict)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = relationship('User', secondary=user_roles, back_populates='roles')

    __table_args__ = (
        Index('idx_role_name', 'name'),
        Index('idx_role_active', 'is_active'),
    )

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(64), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date)
    gender = Column(SQLEnum(GenderEnum), default=GenderEnum.UNKNOWN)
    phone_primary = Column(String(20))
    phone_secondary = Column(String(20))
    profile_complete = Column(Boolean, default=False)
    email_verified = Column(Boolean, default=False)
    phone_verified = Column(Boolean, default=False)
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(32))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    last_login = Column(DateTime)
    last_password_change = Column(DateTime)
    is_active = Column(Boolean, default=True)
    deactivation_reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'))
    updated_by = Column(Integer, ForeignKey('users.id'))

    # Relationships
    roles = relationship('Role', secondary=user_roles, back_populates='users')
    patient_profile = relationship('Patient', back_populates='user', uselist=False)
    appointments = relationship('Appointment', back_populates='provider', foreign_keys='Appointment.provider_id')
    created_appointments = relationship('Appointment', back_populates='created_by_user', foreign_keys='Appointment.created_by')
    updated_appointments = relationship('Appointment', back_populates='updated_by_user', foreign_keys='Appointment.updated_by')
    prescriptions = relationship('Prescription', back_populates='prescribing_physician')
    lab_orders = relationship('LabOrder', back_populates='ordering_physician')
    imaging_orders = relationship('ImagingOrder', back_populates='ordering_physician')
    treatment_plans = relationship('TreatmentPlan', back_populates='created_by_user')
    genomic_analyses = relationship('GenomicAnalysis', back_populates='requested_by_user')
    audit_logs = relationship('AuditLog', back_populates='user')

    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_created_at', 'created_at'),
        Index('idx_user_last_login', 'last_login'),
        CheckConstraint('failed_login_attempts >= 0', name='check_failed_attempts_positive'),
        CheckConstraint('length(username) >= 3', name='check_username_length'),
        CheckConstraint('length(password_hash) >= 8', name='check_password_length'),
    )

    @validates('email')
    def validate_email(self, key, email):
        if email and '@' not in email:
            raise ValueError('Invalid email format')
        return email

class Patient(Base):
    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(SQLEnum(GenderEnum), nullable=False)
    race = Column(String(50))
    ethnicity = Column(String(50))
    language = Column(String(50), default='English')
    phone_primary = Column(String(20))
    phone_secondary = Column(String(20))
    email = Column(String(255))
    address_street = Column(String(255))
    address_city = Column(String(100))
    address_state = Column(String(50))
    address_zip = Column(String(20))
    address_country = Column(String(50), default='US')
    insurance_provider = Column(String(100))
    insurance_policy_number = Column(String(100))
    insurance_group_number = Column(String(50))
    insurance_effective_date = Column(Date)
    insurance_expiration_date = Column(Date)
    emergency_contact_name = Column(String(100))
    emergency_contact_relationship = Column(String(50))
    emergency_contact_phone = Column(String(20))
    emergency_contact_email = Column(String(255))
    blood_type = Column(SQLEnum(BloodTypeEnum))
    height_cm = Column(Numeric(5,1))
    weight_kg = Column(Numeric(5,1))
    bmi = Column(Numeric(4,1))
    smoking_status = Column(String(20))
    alcohol_use = Column(String(20))
    exercise_frequency = Column(String(50))
    occupation = Column(String(100))
    marital_status = Column(String(20))
    primary_care_physician_id = Column(Integer, ForeignKey('users.id'))
    is_active = Column(Boolean, default=True)
    deactivation_reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'))
    updated_by = Column(Integer, ForeignKey('users.id'))

    # Relationships
    user = relationship('User', back_populates='patient_profile', foreign_keys=[user_id])
    primary_care_physician = relationship('User', foreign_keys=[primary_care_physician_id])
    vital_signs = relationship('VitalSigns', back_populates='patient', cascade='all, delete-orphan')
    appointments = relationship('Appointment', back_populates='patient', cascade='all, delete-orphan')
    prescriptions = relationship('Prescription', back_populates='patient', cascade='all, delete-orphan')
    lab_results = relationship('LabResult', back_populates='patient', cascade='all, delete-orphan')
    imaging_studies = relationship('ImagingStudy', back_populates='patient', cascade='all, delete-orphan')
    genomic_analyses = relationship('GenomicAnalysis', back_populates='patient', cascade='all, delete-orphan')
    treatment_plans = relationship('TreatmentPlan', back_populates='patient', cascade='all, delete-orphan')
    clinical_trial_participation = relationship('ClinicalTrialParticipation', back_populates='patient')
    health_monitoring_sessions = relationship('HealthMonitoringSession', back_populates='patient', cascade='all, delete-orphan')
    blockchain_records = relationship('BlockchainRecord', back_populates='patient', cascade='all, delete-orphan')
    audit_logs = relationship('AuditLog', back_populates='patient')

    __table_args__ = (
        Index('idx_patient_user_id', 'user_id'),
        Index('idx_patient_name', 'last_name', 'first_name'),
        Index('idx_patient_dob', 'date_of_birth'),
        Index('idx_patient_active', 'is_active'),
        Index('idx_patient_created_at', 'created_at'),
        CheckConstraint('date_of_birth <= CURRENT_DATE', name='check_dob_not_future'),
        CheckConstraint('height_cm > 0 AND height_cm < 300', name='check_height_range'),
        CheckConstraint('weight_kg > 0 AND weight_kg < 500', name='check_weight_range'),
        CheckConstraint('bmi > 0 AND bmi < 100', name='check_bmi_range'),
    )

class Allergen(Base):
    __tablename__ = 'allergens'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    category = Column(String(50))  # drug, food, environmental, etc.
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patients = relationship('Patient', secondary=patient_allergies, backref='allergies')

    __table_args__ = (
        Index('idx_allergen_category', 'category'),
        Index('idx_allergen_active', 'is_active'),
    )

class VitalSigns(Base):
    __tablename__ = 'vital_signs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    heart_rate = Column(Integer)
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    temperature = Column(Numeric(4,1))
    respiratory_rate = Column(Integer)
    oxygen_saturation = Column(Numeric(4,1))
    weight = Column(Numeric(5,1))
    height = Column(Numeric(5,1))
    bmi = Column(Numeric(4,1))
    pain_scale = Column(Integer)
    blood_glucose = Column(Numeric(5,1))
    notes = Column(Text)
    device_type = Column(String(50))
    device_id = Column(String(100))
    measurement_method = Column(String(50))
    measurement_quality = Column(String(20), default='good')
    recorded_by = Column(String(100))
    recorded_by_user_id = Column(Integer, ForeignKey('users.id'))
    is_manual_entry = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship('Patient', back_populates='vital_signs')

    __table_args__ = (
        Index('idx_vital_signs_patient_timestamp', 'patient_id', 'timestamp'),
        Index('idx_vital_signs_timestamp', 'timestamp'),
        Index('idx_vital_signs_device', 'device_type', 'device_id'),
        CheckConstraint('heart_rate IS NULL OR (heart_rate >= 30 AND heart_rate <= 250)', name='check_heart_rate_range'),
        CheckConstraint('blood_pressure_systolic IS NULL OR (blood_pressure_systolic >= 70 AND blood_pressure_systolic <= 300)', name='check_bp_systolic_range'),
        CheckConstraint('blood_pressure_diastolic IS NULL OR (blood_pressure_diastolic >= 40 AND blood_pressure_diastolic <= 150)', name='check_bp_diastolic_range'),
        CheckConstraint('temperature IS NULL OR (temperature >= 30.0 AND temperature <= 45.0)', name='check_temperature_range'),
        CheckConstraint('respiratory_rate IS NULL OR (respiratory_rate >= 8 AND respiratory_rate <= 60)', name='check_respiratory_rate_range'),
        CheckConstraint('oxygen_saturation IS NULL OR (oxygen_saturation >= 70.0 AND oxygen_saturation <= 100.0)', name='check_o2_sat_range'),
        CheckConstraint('pain_scale IS NULL OR (pain_scale >= 0 AND pain_scale <= 10)', name='check_pain_scale_range'),
        CheckConstraint('blood_glucose IS NULL OR (blood_glucose >= 20.0 AND blood_glucose <= 600.0)', name='check_glucose_range'),
    )

class Appointment(Base):
    __tablename__ = 'appointments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    appointment_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    provider_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    appointment_type = Column(String(50), nullable=False)
    specialty = Column(String(50))
    urgency = Column(String(20), default='routine')
    scheduled_date = Column(DateTime, nullable=False, index=True)
    duration_minutes = Column(Integer, nullable=False)
    status = Column(SQLEnum(AppointmentStatusEnum), default=AppointmentStatusEnum.SCHEDULED, index=True)
    location_type = Column(String(20), default='clinic')  # clinic, telehealth, home_visit
    facility_name = Column(String(100))
    room_number = Column(String(20))
    virtual_meeting_link = Column(String(500))
    reason_for_visit = Column(Text)
    chief_complaint = Column(Text)
    notes = Column(Text)
    diagnosis_codes = Column(ARRAY(String(10)))
    procedure_codes = Column(ARRAY(String(10)))
    follow_up_instructions = Column(Text)
    next_appointment_date = Column(DateTime)
    cancellation_reason = Column(Text)
    no_show_reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    updated_by = Column(Integer, ForeignKey('users.id'))
    cancelled_by = Column(Integer, ForeignKey('users.id'))

    # Relationships
    patient = relationship('Patient', back_populates='appointments')
    provider = relationship('User', back_populates='appointments', foreign_keys=[provider_id])
    created_by_user = relationship('User', foreign_keys=[created_by])
    updated_by_user = relationship('User', foreign_keys=[updated_by])

    __table_args__ = (
        Index('idx_appointment_patient_date', 'patient_id', 'scheduled_date'),
        Index('idx_appointment_provider_date', 'provider_id', 'scheduled_date'),
        Index('idx_appointment_status_date', 'status', 'scheduled_date'),
        Index('idx_appointment_created_at', 'created_at'),
        CheckConstraint('duration_minutes >= 15 AND duration_minutes <= 480', name='check_duration_range'),
        CheckConstraint('scheduled_date >= CURRENT_TIMESTAMP', name='check_future_appointment'),
    )

class Medication(Base):
    __tablename__ = 'medications'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    generic_name = Column(String(255))
    brand_name = Column(String(255))
    drug_class = Column(String(100))
    strength = Column(String(50))
    form = Column(String(50))  # tablet, capsule, injection, etc.
    route = Column(String(50))  # oral, IV, topical, etc.
    controlled_substance = Column(Boolean, default=False)
    dea_schedule = Column(String(10))
    indications = Column(ARRAY(String(255)))
    contraindications = Column(ARRAY(String(255)))
    side_effects = Column(ARRAY(String(255)))
    interactions = Column(JSONB)
    dosage_forms = Column(JSONB)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    prescriptions = relationship('Prescription', back_populates='medication')
    allergic_patients = relationship('Patient', secondary=medication_allergies, backref='allergic_medications')

    __table_args__ = (
        Index('idx_medication_class', 'drug_class'),
        Index('idx_medication_active', 'is_active'),
        Index('idx_medication_name_generic', 'name', 'generic_name'),
    )

class Prescription(Base):
    __tablename__ = 'prescriptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    prescription_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    prescribing_physician_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    medication_id = Column(Integer, ForeignKey('medications.id'), nullable=False, index=True)
    dosage = Column(String(100), nullable=False)
    frequency = Column(String(100), nullable=False)
    duration = Column(String(100))
    quantity = Column(Integer)
    refills_allowed = Column(Integer, default=0)
    refills_remaining = Column(Integer, default=0)
    instructions = Column(Text)
    indications = Column(Text)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date)
    status = Column(SQLEnum(MedicationStatusEnum), default=MedicationStatusEnum.ACTIVE, index=True)
    is_prn = Column(Boolean, default=False)  # as needed
    pharmacy_name = Column(String(255))
    pharmacy_phone = Column(String(20))
    pharmacy_address = Column(Text)
    filled_date = Column(Date)
    filled_by = Column(String(100))
    cost = Column(Numeric(10,2))
    insurance_coverage = Column(Numeric(10,2))
    patient_pay = Column(Numeric(10,2))
    notes = Column(Text)
    discontinuation_reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship('Patient', back_populates='prescriptions')
    prescribing_physician = relationship('User', back_populates='prescriptions')
    medication = relationship('Medication', back_populates='prescriptions')

    __table_args__ = (
        Index('idx_prescription_patient_status', 'patient_id', 'status'),
        Index('idx_prescription_physician_date', 'prescribing_physician_id', 'created_at'),
        Index('idx_prescription_medication', 'medication_id'),
        Index('idx_prescription_dates', 'start_date', 'end_date'),
        CheckConstraint('quantity > 0', name='check_quantity_positive'),
        CheckConstraint('refills_allowed >= 0', name='check_refills_allowed_positive'),
        CheckConstraint('refills_remaining >= 0', name='check_refills_remaining_positive'),
        CheckConstraint('end_date IS NULL OR end_date >= start_date', name='check_end_date_after_start'),
    )

class LabTest(Base):
    __tablename__ = 'lab_tests'

    id = Column(Integer, primary_key=True, autoincrement=True)
    test_code = Column(String(20), unique=True, nullable=False, index=True)
    test_name = Column(String(255), nullable=False)
    category = Column(String(50), index=True)
    specimen_type = Column(String(50))
    reference_range_male = Column(String(100))
    reference_range_female = Column(String(100))
    reference_range_pediatric = Column(String(100))
    units = Column(String(20))
    method = Column(String(100))
    turnaround_time_hours = Column(Integer)
    critical_high = Column(String(50))
    critical_low = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    results = relationship('LabResult', back_populates='test')

    __table_args__ = (
        Index('idx_lab_test_category_active', 'category', 'is_active'),
        Index('idx_lab_test_name', 'test_name'),
    )

class LabOrder(Base):
    __tablename__ = 'lab_orders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    ordering_physician_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    appointment_id = Column(Integer, ForeignKey('appointments.id'))
    diagnosis_codes = Column(ARRAY(String(10)))
    clinical_indication = Column(Text)
    priority = Column(String(20), default='routine')
    collection_date = Column(DateTime)
    collection_method = Column(String(50))
    specimen_collected = Column(Boolean, default=False)
    specimen_type = Column(String(50))
    specimen_volume = Column(String(20))
    collection_notes = Column(Text)
    performing_lab = Column(String(255))
    lab_account_number = Column(String(50))
    cost_estimate = Column(Numeric(8,2))
    insurance_approval_required = Column(Boolean, default=False)
    insurance_approved = Column(Boolean, default=False)
    status = Column(String(20), default='ordered', index=True)
    cancellation_reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship('Patient', backref='lab_orders')
    ordering_physician = relationship('User', back_populates='lab_orders')
    results = relationship('LabResult', back_populates='order', cascade='all, delete-orphan')

    __table_args__ = (
        Index('idx_lab_order_patient_date', 'patient_id', 'created_at'),
        Index('idx_lab_order_status_date', 'status', 'created_at'),
        Index('idx_lab_order_physician', 'ordering_physician_id'),
    )

class LabResult(Base):
    __tablename__ = 'lab_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(String(50), unique=True, nullable=False, index=True)
    order_id = Column(Integer, ForeignKey('lab_orders.id'), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    test_id = Column(Integer, ForeignKey('lab_tests.id'), nullable=False, index=True)
    test_name = Column(String(255), nullable=False)
    test_code = Column(String(20))
    category = Column(String(50))
    collection_date = Column(DateTime)
    received_date = Column(DateTime)
    reported_date = Column(DateTime, nullable=False)
    result_value = Column(String(255))
    result_numeric = Column(Numeric(10,3))
    units = Column(String(20))
    reference_range = Column(String(100))
    interpretation = Column(String(50))
    flag = Column(String(20))  # normal, high, low, critical, abnormal
    notes = Column(Text)
    performing_lab = Column(String(255))
    lab_technician = Column(String(100))
    instrument_used = Column(String(100))
    quality_control_passed = Column(Boolean, default=True)
    status = Column(SQLEnum(LabResultStatusEnum), default=LabResultStatusEnum.COMPLETED, index=True)
    reviewed_by = Column(Integer, ForeignKey('users.id'))
    reviewed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    order = relationship('LabOrder', back_populates='results')
    patient = relationship('Patient', back_populates='lab_results')
    test = relationship('LabTest', back_populates='results')

    __table_args__ = (
        Index('idx_lab_result_patient_date', 'patient_id', 'reported_date'),
        Index('idx_lab_result_test_patient', 'test_id', 'patient_id'),
        Index('idx_lab_result_status_date', 'status', 'reported_date'),
        Index('idx_lab_result_flag', 'flag'),
    )

class ImagingOrder(Base):
    __tablename__ = 'imaging_orders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    ordering_physician_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    appointment_id = Column(Integer, ForeignKey('appointments.id'))
    modality = Column(SQLEnum(ImagingModalityEnum), nullable=False)
    body_part = Column(String(100))
    clinical_indication = Column(Text)
    priority = Column(String(20), default='routine')
    scheduled_date = Column(DateTime)
    performing_facility = Column(String(255))
    contrast_required = Column(Boolean, default=False)
    contrast_type = Column(String(50))
    preparation_instructions = Column(Text)
    cost_estimate = Column(Numeric(8,2))
    insurance_approval_required = Column(Boolean, default=False)
    insurance_approved = Column(Boolean, default=False)
    status = Column(String(20), default='ordered', index=True)
    cancellation_reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship('Patient', backref='imaging_orders')
    ordering_physician = relationship('User', back_populates='imaging_orders')
    studies = relationship('ImagingStudy', back_populates='order', cascade='all, delete-orphan')

    __table_args__ = (
        Index('idx_imaging_order_patient_date', 'patient_id', 'created_at'),
        Index('idx_imaging_order_modality', 'modality'),
        Index('idx_imaging_order_status_date', 'status', 'created_at'),
    )

class ImagingStudy(Base):
    __tablename__ = 'imaging_studies'

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(String(50), unique=True, nullable=False, index=True)
    order_id = Column(Integer, ForeignKey('imaging_orders.id'), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    accession_number = Column(String(50), unique=True, index=True)
    study_type = Column(String(100), nullable=False)
    modality = Column(SQLEnum(ImagingModalityEnum), nullable=False)
    body_part = Column(String(100))
    study_date = Column(DateTime, nullable=False, index=True)
    study_time = Column(Time)
    performing_physician = Column(String(100))
    performing_facility = Column(String(255))
    equipment_model = Column(String(100))
    protocol_used = Column(String(100))
    findings = Column(Text)
    impression = Column(Text)
    recommendations = Column(Text)
    comparison_studies = Column(Text)
    technique = Column(Text)
    contrast_used = Column(Boolean, default=False)
    contrast_details = Column(String(100))
    radiation_dose = Column(Numeric(8,3))
    image_quality = Column(String(20))
    critical_findings = Column(Boolean, default=False)
    preliminary_report = Column(Text)
    final_report = Column(Text)
    reported_by = Column(Integer, ForeignKey('users.id'))
    reported_at = Column(DateTime)
    verified_by = Column(Integer, ForeignKey('users.id'))
    verified_at = Column(DateTime)
    status = Column(String(20), default='final', index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    order = relationship('ImagingOrder', back_populates='studies')
    patient = relationship('Patient', back_populates='imaging_studies')

    __table_args__ = (
        Index('idx_imaging_study_patient_date', 'patient_id', 'study_date'),
        Index('idx_imaging_study_modality_date', 'modality', 'study_date'),
        Index('idx_imaging_study_status_date', 'status', 'study_date'),
        Index('idx_imaging_study_accession', 'accession_number'),
    )

class GenomicAnalysis(Base):
    __tablename__ = 'genomic_analyses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False)
    reference_genome = Column(String(20), default='GRCh38')
    sequencing_platform = Column(String(50))
    coverage_depth = Column(Numeric(6,1))
    clinical_indication = Column(Text)
    status = Column(SQLEnum(GenomicAnalysisStatusEnum), default=GenomicAnalysisStatusEnum.PENDING, index=True)
    progress_percentage = Column(Numeric(5,2), default=0.0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    estimated_completion_time = Column(DateTime)
    variants_called = Column(Integer)
    variants_filtered = Column(Integer)
    variants_annotated = Column(Integer)
    clinical_variants = Column(Integer)
    clinical_report = Column(Text)
    recommendations = Column(JSONB)
    warnings = Column(JSONB)
    raw_data_path = Column(String(500))
    processed_data_path = Column(String(500))
    report_path = Column(String(500))
    cost = Column(Numeric(8,2))
    requested_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    reviewed_by = Column(Integer, ForeignKey('users.id'))
    reviewed_at = Column(DateTime)
    quality_score = Column(Numeric(3,2))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship('Patient', back_populates='genomic_analyses')
    requested_by_user = relationship('User', foreign_keys=[requested_by])
    variants = relationship('GeneticVariant', back_populates='analysis', cascade='all, delete-orphan')

    __table_args__ = (
        Index('idx_genomic_analysis_patient_status', 'patient_id', 'status'),
        Index('idx_genomic_analysis_status_date', 'status', 'created_at'),
        Index('idx_genomic_analysis_type', 'analysis_type'),
        Index('idx_genomic_analysis_requested_by', 'requested_by'),
    )

class GeneticVariant(Base):
    __tablename__ = 'genetic_variants'

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey('genomic_analyses.id'), nullable=False, index=True)
    chromosome = Column(String(5), nullable=False, index=True)
    position = Column(Integer, nullable=False, index=True)
    reference_allele = Column(String(1000), nullable=False)
    alternate_allele = Column(String(1000), nullable=False)
    variant_id = Column(String(50), index=True)
    quality_score = Column(Numeric(8,2))
    depth = Column(Integer)
    genotype_quality = Column(Numeric(8,2))
    allele_frequency_global = Column(Numeric(8,6))
    allele_frequency_african = Column(Numeric(8,6))
    allele_frequency_european = Column(Numeric(8,6))
    allele_frequency_asian = Column(Numeric(8,6))
    allele_frequency_latino = Column(Numeric(8,6))
    gene_name = Column(String(50), index=True)
    transcript_id = Column(String(50))
    consequence = Column(String(100))
    impact = Column(String(20), index=True)
    exon_number = Column(String(20))
    intron_number = Column(String(20))
    amino_acid_change = Column(String(100))
    codon_change = Column(String(50))
    clinvar_significance = Column(String(50))
    clinvar_id = Column(String(20))
    sift_score = Column(Numeric(3,2))
    polyphen_score = Column(Numeric(3,2))
    cadd_score = Column(Numeric(5,2))
    gnomad_af = Column(Numeric(8,6))
    exac_af = Column(Numeric(8,6))
    thousand_genomes_af = Column(Numeric(8,6))
    esp6500_af = Column(Numeric(8,6))
    pathogenicity_score = Column(Numeric(5,2))
    conservation_score = Column(Numeric(5,2))
    inheritance_pattern = Column(String(50))
    disease_association = Column(Text)
    drug_response = Column(JSONB)
    functional_studies = Column(Text)
    literature_references = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    analysis = relationship('GenomicAnalysis', back_populates='variants')

    __table_args__ = (
        Index('idx_genetic_variant_chromosome_position', 'chromosome', 'position'),
        Index('idx_genetic_variant_gene_impact', 'gene_name', 'impact'),
        Index('idx_genetic_variant_clinvar', 'clinvar_significance'),
        Index('idx_genetic_variant_analysis', 'analysis_id'),
        CheckConstraint('position > 0', name='check_position_positive'),
        CheckConstraint('depth IS NULL OR depth > 0', name='check_depth_positive'),
    )

class TreatmentPlan(Base):
    __tablename__ = 'treatment_plans'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    specialty = Column(String(50), nullable=False)
    primary_diagnosis = Column(String(255), nullable=False)
    secondary_diagnoses = Column(ARRAY(String(255)))
    plan_date = Column(Date, nullable=False)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    plan_objectives = Column(ARRAY(String(500)))
    medications = Column(JSONB)
    procedures = Column(JSONB)
    lifestyle_modifications = Column(JSONB)
    follow_up_schedule = Column(JSONB)
    monitoring_parameters = Column(JSONB)
    expected_outcomes = Column(JSONB)
    success_criteria = Column(JSONB)
    contingency_plans = Column(JSONB)
    patient_goals = Column(JSONB)
    caregiver_involvement = Column(JSONB)
    status = Column(String(20), default='active', index=True)
    review_date = Column(Date)
    end_date = Column(Date)
    discontinuation_reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship('Patient', back_populates='treatment_plans')
    created_by_user = relationship('User', back_populates='treatment_plans')

    __table_args__ = (
        Index('idx_treatment_plan_patient_status', 'patient_id', 'status'),
        Index('idx_treatment_plan_specialty_date', 'specialty', 'plan_date'),
        Index('idx_treatment_plan_created_by', 'created_by'),
        CheckConstraint('end_date IS NULL OR end_date >= plan_date', name='check_end_date_after_plan_date'),
    )

class ClinicalTrial(Base):
    __tablename__ = 'clinical_trials'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_id = Column(String(50), unique=True, nullable=False, index=True)
    nct_id = Column(String(20), unique=True, index=True)
    title = Column(String(500), nullable=False)
    acronym = Column(String(50))
    phase = Column(SQLEnum(ClinicalTrialPhaseEnum))
    status = Column(SQLEnum(ClinicalTrialStatusEnum), default=ClinicalTrialStatusEnum.RECRUITING, index=True)
    study_type = Column(String(50))
    allocation = Column(String(50))
    intervention_model = Column(String(50))
    masking = Column(String(50))
    purpose = Column(String(50))
    condition = Column(String(255), index=True)
    intervention = Column(Text)
    description = Column(Text)
    enrollment_target = Column(Integer)
    enrollment_actual = Column(Integer, default=0)
    minimum_age = Column(String(20))
    maximum_age = Column(String(20))
    gender = Column(String(20))
    healthy_volunteers = Column(Boolean)
    criteria = Column(Text)
    locations = Column(JSONB)
    sponsor = Column(String(255))
    collaborators = Column(ARRAY(String(255)))
    start_date = Column(Date)
    completion_date = Column(Date)
    primary_completion_date = Column(Date)
    results_date = Column(Date)
    url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    participants = relationship('ClinicalTrialParticipation', back_populates='trial', cascade='all, delete-orphan')

    __table_args__ = (
        Index('idx_clinical_trial_status_phase', 'status', 'phase'),
        Index('idx_clinical_trial_condition', 'condition'),
        Index('idx_clinical_trial_sponsor', 'sponsor'),
        Index('idx_clinical_trial_dates', 'start_date', 'completion_date'),
    )

class ClinicalTrialParticipation(Base):
    __tablename__ = 'clinical_trial_participation'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_id = Column(Integer, ForeignKey('clinical_trials.id'), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    enrollment_date = Column(Date, nullable=False)
    randomization_date = Column(Date)
    randomization_arm = Column(String(100))
    status = Column(String(20), default='active', index=True)
    withdrawal_date = Column(Date)
    withdrawal_reason = Column(Text)
    adverse_events = Column(JSONB)
    protocol_deviations = Column(JSONB)
    data_collection_compliance = Column(Numeric(5,2))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trial = relationship('ClinicalTrial', back_populates='participants')
    patient = relationship('Patient', back_populates='clinical_trial_participation')

    __table_args__ = (
        Index('idx_trial_participation_patient_status', 'patient_id', 'status'),
        Index('idx_trial_participation_trial_date', 'trial_id', 'enrollment_date'),
        CheckConstraint('withdrawal_date IS NULL OR withdrawal_date >= enrollment_date', name='check_withdrawal_after_enrollment'),
    )

class HealthMonitoringSession(Base):
    __tablename__ = 'health_monitoring_sessions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    device_type = Column(String(50), nullable=False)
    device_id = Column(String(100), nullable=False, index=True)
    device_model = Column(String(100))
    device_firmware_version = Column(String(20))
    session_start = Column(DateTime, nullable=False, index=True)
    session_end = Column(DateTime)
    duration_seconds = Column(Integer)
    data_points_collected = Column(Integer)
    alerts_generated = Column(Integer)
    critical_alerts = Column(Integer)
    data_quality_score = Column(Numeric(3,2))
    battery_level_start = Column(Numeric(5,2))
    battery_level_end = Column(Numeric(5,2))
    signal_strength_avg = Column(Numeric(5,2))
    connection_drops = Column(Integer, default=0)
    firmware_updates_available = Column(Boolean, default=False)
    calibration_required = Column(Boolean, default=False)
    status = Column(String(20), default='completed', index=True)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship('Patient', back_populates='health_monitoring_sessions')
    vital_signs_data = relationship('VitalSigns', backref='monitoring_session', foreign_keys='VitalSigns.patient_id')

    __table_args__ = (
        Index('idx_monitoring_session_patient_device', 'patient_id', 'device_id'),
        Index('idx_monitoring_session_device_type', 'device_type'),
        Index('idx_monitoring_session_start_end', 'session_start', 'session_end'),
        Index('idx_monitoring_session_status', 'status'),
        CheckConstraint('duration_seconds IS NULL OR duration_seconds > 0', name='check_duration_positive'),
        CheckConstraint('data_points_collected IS NULL OR data_points_collected >= 0', name='check_data_points_positive'),
    )

class BlockchainRecord(Base):
    __tablename__ = 'blockchain_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String(100), unique=True, nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    record_type = Column(String(50), nullable=False, index=True)
    data_hash = Column(String(128), nullable=False, index=True)
    block_hash = Column(String(128), nullable=False, index=True)
    transaction_hash = Column(String(128), unique=True, index=True)
    block_number = Column(Integer)
    timestamp = Column(DateTime, nullable=False, index=True)
    data_size_bytes = Column(Integer)
    encryption_algorithm = Column(String(20), default='AES-256')
    key_id = Column(String(100))
    verified = Column(Boolean, default=True, index=True)
    verification_timestamp = Column(DateTime)
    access_log = Column(JSONB)
    metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship('Patient', back_populates='blockchain_records')

    __table_args__ = (
        Index('idx_blockchain_record_patient_type', 'patient_id', 'record_type'),
        Index('idx_blockchain_record_timestamp', 'timestamp'),
        Index('idx_blockchain_record_block_hash', 'block_hash'),
        Index('idx_blockchain_record_transaction', 'transaction_hash'),
        CheckConstraint('data_size_bytes IS NULL OR data_size_bytes > 0', name='check_data_size_positive'),
    )

class AuditLog(Base):
    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    event_type = Column(SQLEnum(AuditEventTypeEnum), nullable=False, index=True)
    event_category = Column(String(50), index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    session_id = Column(String(100), index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), index=True)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    resource_type = Column(String(50))
    resource_id = Column(String(100))
    action = Column(String(50))
    success = Column(Boolean, default=True, index=True)
    error_code = Column(String(20))
    error_message = Column(Text)
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    metadata = Column(JSONB)
    phi_access = Column(Boolean, default=False, index=True)
    compliance_flags = Column(ARRAY(String(50)))
    retention_period_days = Column(Integer, default=2555)  # 7 years for HIPAA
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship('User', back_populates='audit_logs')
    patient = relationship('Patient', back_populates='audit_logs')

    __table_args__ = (
        Index('idx_audit_log_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_log_patient_timestamp', 'patient_id', 'timestamp'),
        Index('idx_audit_log_event_type_timestamp', 'event_type', 'timestamp'),
        Index('idx_audit_log_success_timestamp', 'success', 'timestamp'),
        Index('idx_audit_log_phi_access', 'phi_access'),
        Index('idx_audit_log_composite', 'event_type', 'resource_type', 'success'),
        CheckConstraint('retention_period_days > 0', name='check_retention_positive'),
    )

class SystemConfiguration(Base):
    __tablename__ = 'system_configuration'

    id = Column(Integer, primary_key=True, autoincrement=True)
    config_key = Column(String(100), unique=True, nullable=False, index=True)
    config_value = Column(JSONB)
    config_type = Column(String(20), default='string')
    description = Column(Text)
    is_encrypted = Column(Boolean, default=False)
    environment = Column(String(20), default='production')
    last_modified_by = Column(Integer, ForeignKey('users.id'))
    last_modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    modifier = relationship('User', foreign_keys=[last_modified_by])

    __table_args__ = (
        Index('idx_system_config_environment', 'environment'),
        Index('idx_system_config_type', 'config_type'),
        CheckConstraint('version > 0', name='check_version_positive'),
    )

class APIKey(Base):
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(50), unique=True, nullable=False, index=True)
    key_secret_hash = Column(String(128), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    expires_at = Column(DateTime)
    last_used_at = Column(DateTime)
    usage_count = Column(Integer, default=0)
    rate_limit_requests = Column(Integer, default=1000)
    rate_limit_window_seconds = Column(Integer, default=3600)
    permissions = Column(JSONB)
    ip_whitelist = Column(ARRAY(String(45)))
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    creator = relationship('User', foreign_keys=[created_by])

    __table_args__ = (
        Index('idx_api_key_active_expires', 'is_active', 'expires_at'),
        Index('idx_api_key_created_by', 'created_by'),
        CheckConstraint('rate_limit_requests > 0', name='check_rate_limit_positive'),
        CheckConstraint('rate_limit_window_seconds > 0', name='check_window_positive'),
        CheckConstraint('usage_count >= 0', name='check_usage_positive'),
    )

class NotificationTemplate(Base):
    __tablename__ = 'notification_templates'

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False, index=True)  # email, sms, push
    subject = Column(String(255))
    body_template = Column(Text, nullable=False)
    variables = Column(JSONB)
    language = Column(String(10), default='en')
    is_active = Column(Boolean, default=True, index=True)
    created_by = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    creator = relationship('User', foreign_keys=[created_by])

    __table_args__ = (
        Index('idx_notification_template_type_active', 'type', 'is_active'),
        Index('idx_notification_template_language', 'language'),
    )

class NotificationLog(Base):
    __tablename__ = 'notification_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    notification_id = Column(String(50), unique=True, nullable=False, index=True)
    template_id = Column(String(50), ForeignKey('notification_templates.template_id'))
    recipient_user_id = Column(Integer, ForeignKey('users.id'))
    recipient_email = Column(String(255))
    recipient_phone = Column(String(20))
    type = Column(String(50), nullable=False, index=True)
    subject = Column(String(255))
    body = Column(Text)
    status = Column(String(20), default='sent', index=True)
    sent_at = Column(DateTime)
    delivered_at = Column(DateTime)
    read_at = Column(DateTime)
    failed_at = Column(DateTime)
    failure_reason = Column(Text)
    retry_count = Column(Integer, default=0)
    metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    recipient = relationship('User', foreign_keys=[recipient_user_id])

    __table_args__ = (
        Index('idx_notification_log_recipient_status', 'recipient_user_id', 'status'),
        Index('idx_notification_log_type_status', 'type', 'status'),
        Index('idx_notification_log_sent_at', 'sent_at'),
        CheckConstraint('retry_count >= 0', name='check_retry_count_positive'),
    )

class MLModel(Base):
    __tablename__ = 'ml_models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    type = Column(String(50), nullable=False, index=True)
    framework = Column(String(50))
    algorithm = Column(String(100))
    purpose = Column(String(255))
    input_schema = Column(JSONB)
    output_schema = Column(JSONB)
    hyperparameters = Column(JSONB)
    training_data_info = Column(JSONB)
    performance_metrics = Column(JSONB)
    accuracy_score = Column(Numeric(5,4))
    precision_score = Column(Numeric(5,4))
    recall_score = Column(Numeric(5,4))
    f1_score = Column(Numeric(5,4))
    auc_roc = Column(Numeric(5,4))
    model_path = Column(String(500))
    model_size_bytes = Column(Integer)
    training_start_time = Column(DateTime)
    training_end_time = Column(DateTime)
    training_duration_seconds = Column(Integer)
    status = Column(String(20), default='trained', index=True)
    is_active = Column(Boolean, default=True, index=True)
    created_by = Column(Integer, ForeignKey('users.id'))
    approved_by = Column(Integer, ForeignKey('users.id'))
    approved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    creator = relationship('User', foreign_keys=[created_by])
    approver = relationship('User', foreign_keys=[approved_by])

    __table_args__ = (
        Index('idx_ml_model_type_status', 'type', 'status'),
        Index('idx_ml_model_active_version', 'is_active', 'version'),
        CheckConstraint('model_size_bytes IS NULL OR model_size_bytes > 0', name='check_model_size_positive'),
        CheckConstraint('training_duration_seconds IS NULL OR training_duration_seconds > 0', name='check_training_duration_positive'),
    )

# Event listeners for audit logging
@event.listens_for(User, 'after_insert')
def audit_user_insert(mapper, connection, target):
    # Automatic audit logging for user creation
    pass

@event.listens_for(User, 'after_update')
def audit_user_update(mapper, connection, target):
    # Automatic audit logging for user updates
    pass

@event.listens_for(Patient, 'after_insert')
def audit_patient_insert(mapper, connection, target):
    # Automatic audit logging for patient creation
    pass

@event.listens_for(Patient, 'after_update')
def audit_patient_update(mapper, connection, target):
    # Automatic audit logging for patient updates
    pass

# Utility functions
def get_patient_age(patient: Patient) -> int:
    """Calculate patient age from date of birth"""
    today = date.today()
    return today.year - patient.date_of_birth.year - ((today.month, today.day) < (patient.date_of_birth.month, patient.date_of_birth.day))

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI from weight and height"""
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

def get_bmi_category(bmi: float) -> str:
    """Get BMI category"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    elif bmi < 35.0:
        return "Obese Class I"
    elif bmi < 40.0:
        return "Obese Class II"
    else:
        return "Obese Class III"

# Database initialization functions
def create_all_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def drop_all_tables(engine):
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)

def get_table_names():
    """Get list of all table names"""
    return [table.name for table in Base.metadata.sorted_tables]

def get_model_relationships():
    """Get model relationships information"""
    relationships = {}
    for table in Base.metadata.sorted_tables:
        table_name = table.name
        relationships[table_name] = {
            'columns': [col.name for col in table.columns],
            'relationships': []
        }

        for relationship in table.relationships:
            rel_info = {
                'name': relationship.key,
                'target_table': relationship.target.name,
                'direction': 'one-to-many' if relationship.uselist else 'many-to-one',
                'foreign_keys': [fk for fk in relationship._calculated_mappers[0].relationships[relationship.key]._calculated_mappers[0].relationships[relationship.key].local_columns]
            }
            relationships[table_name]['relationships'].append(rel_info)

    return relationships

# Export all models
__all__ = [
    'Base',
    'User', 'Role', 'Patient', 'Allergen',
    'VitalSigns', 'Appointment',
    'Medication', 'Prescription',
    'LabTest', 'LabOrder', 'LabResult',
    'ImagingOrder', 'ImagingStudy',
    'GenomicAnalysis', 'GeneticVariant',
    'TreatmentPlan', 'ClinicalTrial', 'ClinicalTrialParticipation',
    'HealthMonitoringSession', 'BlockchainRecord', 'AuditLog',
    'SystemConfiguration', 'APIKey',
    'NotificationTemplate', 'NotificationLog', 'MLModel',
    'UserRoleEnum', 'GenderEnum', 'BloodTypeEnum',
    'AppointmentStatusEnum', 'GenomicAnalysisStatusEnum',
    'MedicationStatusEnum', 'LabResultStatusEnum',
    'ImagingModalityEnum', 'ClinicalTrialPhaseEnum',
    'ClinicalTrialStatusEnum', 'AuditEventTypeEnum',
    'create_all_tables', 'drop_all_tables', 'get_table_names', 'get_model_relationships'
]
