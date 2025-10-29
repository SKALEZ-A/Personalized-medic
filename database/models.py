"""
Comprehensive Database Models for AI Personalized Medicine Platform
SQLAlchemy ORM models with relationships and constraints
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Table, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, validates
from sqlalchemy.sql import func
import enum
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass, field
import hashlib
import secrets


Base = declarative_base()


# Association tables for many-to-many relationships
patient_medication = Table('patient_medication', Base.metadata,
    Column('patient_id', String(50), ForeignKey('patients.patient_id'), primary_key=True),
    Column('medication_id', Integer, ForeignKey('medications.id'), primary_key=True),
    Column('prescribed_date', DateTime, default=datetime.utcnow),
    Column('dosage', String(100)),
    Column('frequency', String(100)),
    Column('prescribing_provider', String(100)),
    Column('active', Boolean, default=True)
)

patient_condition = Table('patient_condition', Base.metadata,
    Column('patient_id', String(50), ForeignKey('patients.patient_id'), primary_key=True),
    Column('condition_id', Integer, ForeignKey('medical_conditions.id'), primary_key=True),
    Column('diagnosis_date', DateTime, default=datetime.utcnow),
    Column('severity', Enum('mild', 'moderate', 'severe', 'critical', name='severity_enum')),
    Column('status', Enum('active', 'resolved', 'chronic', name='condition_status_enum'), default='active'),
    Column('diagnosing_provider', String(100))
)

patient_allergy = Table('patient_allergy', Base.metadata,
    Column('patient_id', String(50), ForeignKey('patients.patient_id'), primary_key=True),
    Column('allergy_id', Integer, ForeignKey('allergies.id'), primary_key=True),
    Column('severity', Enum('mild', 'moderate', 'severe', 'life_threatening', name='allergy_severity_enum')),
    Column('reaction', Text),
    Column('reported_date', DateTime, default=datetime.utcnow)
)


class UserRole(enum.Enum):
    PATIENT = "patient"
    PHYSICIAN = "physician"
    NURSE = "nurse"
    PHARMACIST = "pharmacist"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class Gender(enum.Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class BloodType(enum.Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"


class AppointmentStatus(enum.Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class GenomicAnalysisStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    """User authentication and profile information"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(64), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.PATIENT)
    first_name = Column(String(100))
    last_name = Column(String(100))
    date_of_birth = Column(DateTime)
    gender = Column(Enum(Gender), default=Gender.UNKNOWN)
    phone = Column(String(20))
    address_street = Column(String(255))
    address_city = Column(String(100))
    address_state = Column(String(50))
    address_zip = Column(String(20))
    address_country = Column(String(50), default='US')

    # Security fields
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(32))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    last_login = Column(DateTime)
    password_changed_at = Column(DateTime, default=datetime.utcnow)

    # Profile completion
    profile_complete = Column(Boolean, default=False)
    email_verified = Column(Boolean, default=False)
    phone_verified = Column(Boolean, default=False)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by = Column(String(100))
    updated_by = Column(String(100))

    # Relationships
    patients = relationship("Patient", back_populates="user", uselist=False)
    provider_appointments = relationship("Appointment", foreign_keys="Appointment.provider_id", back_populates="provider")

    @validates('email')
    def validate_email(self, key, email):
        """Validate email format"""
        if email and '@' not in email:
            raise ValueError("Invalid email format")
        return email

    @validates('phone')
    def validate_phone(self, key, phone):
        """Validate phone format"""
        if phone:
            # Remove all non-digit characters
            digits = ''.join(filter(str.isdigit, phone))
            if len(digits) < 10:
                raise ValueError("Phone number must have at least 10 digits")
        return phone

    def check_password(self, password: str) -> bool:
        """Check password against stored hash"""
        hashed_input = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            self.salt.encode(),
            100000
        ).hex()
        return hashed_input == self.password_hash

    def set_password(self, password: str):
        """Set new password with salt"""
        salt = secrets.token_hex(32)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        ).hex()
        self.password_hash = hashed
        self.salt = salt
        self.password_changed_at = datetime.utcnow()

    def is_locked(self) -> bool:
        """Check if account is locked"""
        if self.locked_until and datetime.utcnow() < self.locked_until:
            return True
        return False

    def record_failed_login(self):
        """Record failed login attempt"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= 5:
            # Lock account for 15 minutes
            self.locked_until = datetime.utcnow() + timedelta(minutes=15)

    def record_successful_login(self):
        """Record successful login"""
        self.failed_login_attempts = 0
        self.locked_until = None
        self.last_login = datetime.utcnow()


class Patient(Base):
    """Patient demographic and clinical information"""
    __tablename__ = 'patients'

    patient_id = Column(String(50), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True)

    # Demographics
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(DateTime, nullable=False)
    gender = Column(Enum(Gender), nullable=False)
    race = Column(String(50))
    ethnicity = Column(String(50))
    language = Column(String(50), default='English')

    # Contact Information
    phone_primary = Column(String(20))
    phone_secondary = Column(String(20))
    email = Column(String(255))
    address_street = Column(String(255))
    address_city = Column(String(100))
    address_state = Column(String(50))
    address_zip = Column(String(20))
    address_country = Column(String(50), default='US')

    # Emergency Contact
    emergency_contact_name = Column(String(100))
    emergency_contact_relationship = Column(String(50))
    emergency_contact_phone = Column(String(20))

    # Insurance Information
    insurance_provider = Column(String(100))
    insurance_policy_number = Column(String(50))
    insurance_group_number = Column(String(50))

    # Clinical Information
    blood_type = Column(Enum(BloodType))
    height_cm = Column(Float)
    weight_kg = Column(Float)
    bmi = Column(Float)

    # Medical History
    smoking_status = Column(Enum('never', 'former', 'current', name='smoking_status_enum'))
    alcohol_use = Column(Enum('none', 'occasional', 'moderate', 'heavy', name='alcohol_use_enum'))
    drug_use = Column(Text)  # JSON string of drug use history

    # Family History
    family_history = Column(Text)  # JSON string of family medical history

    # Preferences
    communication_preferences = Column(JSON)  # Email, SMS, phone preferences
    appointment_reminders = Column(Boolean, default=True)
    test_result_notifications = Column(Boolean, default=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by = Column(String(100))
    updated_by = Column(String(100))

    # Relationships
    user = relationship("User", back_populates="patients")
    appointments = relationship("Appointment", back_populates="patient")
    vital_signs = relationship("VitalSigns", back_populates="patient", order_by="VitalSigns.timestamp.desc()")
    medications = relationship("Medication", secondary=patient_medication, back_populates="patients")
    conditions = relationship("MedicalCondition", secondary=patient_condition, back_populates="patients")
    allergies = relationship("Allergy", secondary=patient_allergy, back_populates="patients")
    genomic_analyses = relationship("GenomicAnalysis", back_populates="patient", order_by="GenomicAnalysis.created_at.desc()")
    lab_results = relationship("LabResult", back_populates="patient", order_by="LabResult.collection_date.desc()")
    imaging_studies = relationship("ImagingStudy", back_populates="patient", order_by="ImagingStudy.study_date.desc()")

    @property
    def age(self) -> int:
        """Calculate current age"""
        today = date.today()
        return today.year - self.date_of_birth.year - ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))

    @property
    def full_name(self) -> str:
        """Get full name"""
        return f"{self.first_name} {self.last_name}"

    @validates('date_of_birth')
    def validate_date_of_birth(self, key, date_of_birth):
        """Validate date of birth is not in future"""
        if date_of_birth and date_of_birth > datetime.utcnow():
            raise ValueError("Date of birth cannot be in the future")
        return date_of_birth


class MedicalCondition(Base):
    """Medical conditions and diagnoses"""
    __tablename__ = 'medical_conditions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    icd10_code = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    chronic = Column(Boolean, default=False)
    contagious = Column(Boolean, default=False)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    patients = relationship("Patient", secondary=patient_condition, back_populates="conditions")


class Medication(Base):
    """Medication information"""
    __tablename__ = 'medications'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    generic_name = Column(String(255))
    brand_name = Column(String(255))
    drug_class = Column(String(100))
    indication = Column(Text)
    contraindications = Column(Text)
    side_effects = Column(Text)
    dosage_forms = Column(JSON)  # Available dosage forms and strengths
    controlled_substance = Column(Boolean, default=False)
    schedule = Column(Enum('I', 'II', 'III', 'IV', 'V', name='drug_schedule_enum'))

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    patients = relationship("Patient", secondary=patient_medication, back_populates="medications")


class Allergy(Base):
    """Allergy information"""
    __tablename__ = 'allergies'

    id = Column(Integer, primary_key=True, autoincrement=True)
    allergen = Column(String(255), nullable=False, unique=True)
    category = Column(Enum('drug', 'food', 'environmental', 'other', name='allergen_category_enum'))
    description = Column(Text)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    patients = relationship("Patient", secondary=patient_allergy, back_populates="allergies")


class VitalSigns(Base):
    """Vital signs measurements"""
    __tablename__ = 'vital_signs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Core vital signs
    heart_rate = Column(Integer)
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    temperature = Column(Float)  # Celsius
    respiratory_rate = Column(Integer)
    oxygen_saturation = Column(Float)
    weight = Column(Float)
    height = Column(Float)
    bmi = Column(Float)

    # Additional measurements
    pain_scale = Column(Integer)  # 0-10 pain scale
    blood_glucose = Column(Float)
    peak_flow = Column(Float)  # For asthma monitoring

    # Device information
    device_type = Column(String(100))
    device_id = Column(String(100))
    measurement_method = Column(String(50))

    # Quality indicators
    measurement_quality = Column(Enum('excellent', 'good', 'fair', 'poor', name='measurement_quality_enum'))
    notes = Column(Text)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    recorded_by = Column(String(100))

    # Relationships
    patient = relationship("Patient", back_populates="vital_signs")

    @validates('heart_rate')
    def validate_heart_rate(self, key, heart_rate):
        """Validate heart rate range"""
        if heart_rate and (heart_rate < 30 or heart_rate > 250):
            raise ValueError("Heart rate must be between 30 and 250 bpm")
        return heart_rate

    @validates('blood_pressure_systolic')
    def validate_bp_systolic(self, key, systolic):
        """Validate systolic blood pressure"""
        if systolic and (systolic < 70 or systolic > 300):
            raise ValueError("Systolic blood pressure must be between 70 and 300 mmHg")
        return systolic

    @validates('temperature')
    def validate_temperature(self, key, temperature):
        """Validate temperature range"""
        if temperature and (temperature < 30.0 or temperature > 45.0):
            raise ValueError("Temperature must be between 30°C and 45°C")
        return temperature


class Appointment(Base):
    """Medical appointments"""
    __tablename__ = 'appointments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    appointment_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    provider_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)

    # Appointment details
    title = Column(String(255), nullable=False)
    description = Column(Text)
    appointment_type = Column(String(100))
    specialty = Column(String(100))
    urgency = Column(Enum('routine', 'urgent', 'emergency', name='appointment_urgency_enum'), default='routine')

    # Scheduling
    scheduled_date = Column(DateTime, nullable=False, index=True)
    duration_minutes = Column(Integer, default=30)
    status = Column(Enum(AppointmentStatus), default=AppointmentStatus.SCHEDULED)

    # Location
    location_type = Column(Enum('clinic', 'telemedicine', 'home_visit', 'hospital', name='location_type_enum'), default='clinic')
    facility_name = Column(String(255))
    room_number = Column(String(50))
    virtual_meeting_link = Column(String(500))

    # Follow-up information
    reason_for_visit = Column(Text)
    chief_complaint = Column(Text)
    previous_appointment_id = Column(String(50))

    # Outcomes
    notes = Column(Text)
    diagnosis_codes = Column(JSON)  # List of ICD-10 codes
    procedure_codes = Column(JSON)  # List of CPT codes
    follow_up_instructions = Column(Text)
    next_appointment_date = Column(DateTime)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by = Column(String(100))
    updated_by = Column(String(100))
    cancelled_at = Column(DateTime)
    cancellation_reason = Column(String(255))

    # Relationships
    patient = relationship("Patient", back_populates="appointments")
    provider = relationship("User", foreign_keys=[provider_id], back_populates="provider_appointments")

    @property
    def end_time(self) -> datetime:
        """Calculate appointment end time"""
        return self.scheduled_date + timedelta(minutes=self.duration_minutes)

    @property
    def is_past(self) -> bool:
        """Check if appointment is in the past"""
        return self.end_time < datetime.utcnow()

    @property
    def is_today(self) -> bool:
        """Check if appointment is today"""
        today = date.today()
        return self.scheduled_date.date() == today

    @validates('scheduled_date')
    def validate_scheduled_date(self, key, scheduled_date):
        """Validate appointment date is not in the past"""
        if scheduled_date and scheduled_date < datetime.utcnow():
            raise ValueError("Cannot schedule appointments in the past")
        return scheduled_date


class GenomicAnalysis(Base):
    """Genomic analysis results"""
    __tablename__ = 'genomic_analyses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)

    # Analysis details
    analysis_type = Column(String(50), default='comprehensive')
    reference_genome = Column(String(20), default='GRCh38')
    sequencing_platform = Column(String(100))
    coverage_depth = Column(Float)

    # Status and progress
    status = Column(Enum(GenomicAnalysisStatus), default=GenomicAnalysisStatus.PENDING)
    progress_percentage = Column(Float, default=0.0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    estimated_completion_time = Column(DateTime)

    # Results
    variants_called = Column(Integer)
    variants_filtered = Column(Integer)
    clinical_variants = Column(Integer)

    # Storage paths
    raw_data_path = Column(String(500))
    processed_data_path = Column(String(500))
    results_path = Column(String(500))

    # Quality metrics
    quality_metrics = Column(JSON)

    # Reports and interpretations
    clinical_report = Column(Text)
    research_findings = Column(Text)
    recommendations = Column(JSON)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    requested_by = Column(String(100))
    reviewed_by = Column(String(100))
    approved_by = Column(String(100))

    # Relationships
    patient = relationship("Patient", back_populates="genomic_analyses")
    variants = relationship("GeneticVariant", back_populates="analysis", cascade="all, delete-orphan")

    @property
    def processing_time(self) -> Optional[timedelta]:
        """Calculate total processing time"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def is_overdue(self) -> bool:
        """Check if analysis is overdue"""
        if self.status in [GenomicAnalysisStatus.PENDING, GenomicAnalysisStatus.PROCESSING]:
            return datetime.utcnow() > self.estimated_completion_time
        return False


class GeneticVariant(Base):
    """Genetic variants identified in genomic analysis"""
    __tablename__ = 'genetic_variants'

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey('genomic_analyses.id'), nullable=False, index=True)

    # Variant information
    chromosome = Column(String(5), nullable=False)
    position = Column(Integer, nullable=False)
    reference_allele = Column(String(1000), nullable=False)
    alternate_allele = Column(String(1000), nullable=False)
    variant_id = Column(String(50))

    # Quality metrics
    quality_score = Column(Float)
    depth = Column(Integer)
    genotype_quality = Column(Float)

    # Annotations
    gene_name = Column(String(50), index=True)
    transcript_id = Column(String(50))
    consequence = Column(String(100))
    impact = Column(Enum('high', 'moderate', 'low', 'modifier', name='variant_impact_enum'))

    # Population frequencies
    allele_frequency_global = Column(Float)
    allele_frequency_african = Column(Float)
    allele_frequency_european = Column(Float)
    allele_frequency_asian = Column(Float)

    # Clinical significance
    clinvar_significance = Column(String(50))
    clinvar_id = Column(String(20))
    cosmic_id = Column(String(20))

    # Functional predictions
    sift_score = Column(Float)
    polyphen_score = Column(Float)
    cadd_score = Column(Float)

    # Additional annotations
    exac_frequency = Column(Float)
    gnomad_frequency = Column(Float)
    dbnsfp_predictions = Column(JSON)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    analysis = relationship("GenomicAnalysis", back_populates="variants")

    @property
    def is_pathogenic(self) -> bool:
        """Check if variant is likely pathogenic"""
        return self.clinvar_significance in ['Pathogenic', 'Likely pathogenic']

    @property
    def is_benign(self) -> bool:
        """Check if variant is likely benign"""
        return self.clinvar_significance in ['Benign', 'Likely benign']


class LabResult(Base):
    """Laboratory test results"""
    __tablename__ = 'lab_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)

    # Test information
    test_name = Column(String(255), nullable=False)
    test_code = Column(String(20), index=True)  # LOINC code
    category = Column(String(100))
    specimen_type = Column(String(50))

    # Timing
    ordered_date = Column(DateTime, nullable=False)
    collection_date = Column(DateTime)
    received_date = Column(DateTime)
    reported_date = Column(DateTime)

    # Results
    result_value = Column(String(255))
    result_numeric = Column(Float)
    units = Column(String(50))
    reference_range = Column(String(100))
    interpretation = Column(Enum('normal', 'abnormal', 'critical', 'pending', name='result_interpretation_enum'))

    # Quality and validation
    performing_lab = Column(String(255))
    lab_director = Column(String(100))
    validation_status = Column(Enum('preliminary', 'final', 'amended', 'corrected', name='validation_status_enum'), default='preliminary')

    # Flags and notes
    abnormal_flags = Column(JSON)  # List of abnormal flags
    critical_values = Column(JSON)  # Critical value notifications
    notes = Column(Text)

    # Methodology
    method = Column(String(100))
    instrument = Column(String(100))
    reagent_lot = Column(String(50))

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    ordered_by = Column(String(100))
    collected_by = Column(String(100))
    verified_by = Column(String(100))

    # Relationships
    patient = relationship("Patient", back_populates="lab_results")

    @property
    def is_abnormal(self) -> bool:
        """Check if result is abnormal"""
        return self.interpretation in ['abnormal', 'critical']

    @property
    def is_critical(self) -> bool:
        """Check if result is critical"""
        return self.interpretation == 'critical'

    @property
    def turnaround_time(self) -> Optional[timedelta]:
        """Calculate lab turnaround time"""
        if self.collection_date and self.reported_date:
            return self.reported_date - self.collection_date
        return None


class ImagingStudy(Base):
    """Medical imaging studies"""
    __tablename__ = 'imaging_studies'

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)

    # Study information
    study_type = Column(String(100), nullable=False)
    modality = Column(Enum('CT', 'MRI', 'XRAY', 'US', 'NM', 'PET', name='imaging_modality_enum'), nullable=False)
    body_part = Column(String(100))
    contrast_used = Column(Boolean, default=False)
    contrast_type = Column(String(50))

    # Timing
    study_date = Column(DateTime, nullable=False, index=True)
    scheduled_date = Column(DateTime)
    completed_date = Column(DateTime)

    # Clinical information
    clinical_indication = Column(Text)
    comparison_studies = Column(JSON)  # List of previous study IDs
    technique = Column(Text)

    # Results and interpretation
    findings = Column(Text)
    impression = Column(Text)
    recommendations = Column(Text)
    report_status = Column(Enum('preliminary', 'final', 'amended', name='report_status_enum'), default='preliminary')

    # Quality and technical details
    image_quality = Column(Enum('excellent', 'good', 'adequate', 'poor', name='image_quality_enum'))
    radiation_dose = Column(Float)  # For radiation-based modalities
    slice_thickness = Column(Float)  # For CT/MRI

    # Personnel
    ordering_provider = Column(String(100))
    performing_technologist = Column(String(100))
    interpreting_radiologist = Column(String(100))

    # Storage and access
    dicom_study_uid = Column(String(100), unique=True)
    storage_path = Column(String(500))
    web_access_link = Column(String(500))

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    patient = relationship("Patient", back_populates="imaging_studies")
    series = relationship("ImagingSeries", back_populates="study", cascade="all, delete-orphan")

    @property
    def study_age_days(self) -> int:
        """Calculate age of study in days"""
        return (datetime.utcnow() - self.study_date).days


class ImagingSeries(Base):
    """Individual series within an imaging study"""
    __tablename__ = 'imaging_series'

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(Integer, ForeignKey('imaging_studies.id'), nullable=False, index=True)
    series_number = Column(Integer, nullable=False)

    # Series details
    series_description = Column(String(255))
    modality = Column(String(10))
    protocol_name = Column(String(255))

    # Image information
    image_count = Column(Integer)
    slice_count = Column(Integer)
    dicom_series_uid = Column(String(100), unique=True)

    # Technical parameters
    slice_thickness = Column(Float)
    spacing_between_slices = Column(Float)
    pixel_spacing = Column(String(50))
    field_of_view = Column(String(50))

    # Storage
    storage_path = Column(String(500))

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    study = relationship("ImagingStudy", back_populates="series")


class ClinicalTrial(Base):
    """Clinical trial information and enrollment"""
    __tablename__ = 'clinical_trials'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_id = Column(String(50), unique=True, nullable=False, index=True)

    # Trial information
    nct_id = Column(String(20), unique=True)  # ClinicalTrials.gov ID
    title = Column(String(500), nullable=False)
    phase = Column(Enum('I', 'II', 'III', 'IV', name='trial_phase_enum'))
    status = Column(Enum('recruiting', 'active', 'completed', 'terminated', 'withdrawn', name='trial_status_enum'))

    # Study details
    condition = Column(String(255))
    intervention = Column(Text)
    study_type = Column(Enum('interventional', 'observational', name='study_type_enum'))
    allocation = Column(Enum('randomized', 'nonrandomized', name='allocation_enum'))
    masking = Column(String(100))

    # Eligibility criteria
    age_min = Column(Integer)
    age_max = Column(Integer)
    gender = Column(Enum(Gender))
    accepts_healthy_volunteers = Column(Boolean, default=False)
    inclusion_criteria = Column(Text)
    exclusion_criteria = Column(Text)

    # Trial logistics
    sponsor = Column(String(255))
    collaborators = Column(JSON)
    start_date = Column(DateTime)
    completion_date = Column(DateTime)
    enrollment_target = Column(Integer)
    enrollment_actual = Column(Integer, default=0)

    # Locations
    locations = Column(JSON)  # List of trial locations

    # Results and outcomes
    primary_outcome = Column(Text)
    secondary_outcomes = Column(JSON)
    results_summary = Column(Text)
    publications = Column(JSON)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    participants = relationship("TrialParticipant", back_populates="trial", cascade="all, delete-orphan")


class TrialParticipant(Base):
    """Clinical trial participant information"""
    __tablename__ = 'trial_participants'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_id = Column(Integer, ForeignKey('clinical_trials.id'), nullable=False, index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)

    # Enrollment information
    enrollment_date = Column(DateTime, default=datetime.utcnow)
    randomization_group = Column(String(100))
    participant_id = Column(String(50), unique=True)

    # Study progress
    visit_schedule = Column(JSON)
    completed_visits = Column(Integer, default=0)
    adverse_events = Column(JSON)

    # Outcomes
    primary_outcome_value = Column(Float)
    secondary_outcomes = Column(JSON)
    study_completion_status = Column(Enum('ongoing', 'completed', 'withdrawn', 'lost_to_followup', name='completion_status_enum'), default='ongoing')

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    trial = relationship("ClinicalTrial", back_populates="participants")
    patient = relationship("Patient", foreign_keys=[patient_id])


class AuditLog(Base):
    """Comprehensive audit logging"""
    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Event information
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(Enum('authentication', 'authorization', 'data_access', 'data_modification', 'system', 'security', name='audit_category_enum'), nullable=False)

    # User information
    user_id = Column(String(100), index=True)
    session_id = Column(String(100))
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(String(500))

    # Resource information
    resource_type = Column(String(100))
    resource_id = Column(String(100))
    action = Column(String(50))
    method = Column(String(20))  # HTTP method for API calls

    # Event details
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    old_values = Column(JSON)  # For data modification events
    new_values = Column(JSON)  # For data modification events
    metadata = Column(JSON)    # Additional event-specific data

    # Security context
    risk_level = Column(Enum('low', 'medium', 'high', 'critical', name='risk_level_enum'), default='low')
    compliance_flags = Column(JSON)  # HIPAA, GDPR compliance flags

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    @classmethod
    def log_event(cls, session, event_type: str, event_category: str, **kwargs):
        """Helper method to log audit events"""
        audit_entry = cls(
            event_type=event_type,
            event_category=event_category,
            **kwargs
        )
        session.add(audit_entry)
        session.commit()


# Database connection and session management
def create_database_engine(database_url: str):
    """Create SQLAlchemy engine"""
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False  # Set to True for SQL debugging
    )


def create_session_factory(engine):
    """Create session factory"""
    return sessionmaker(bind=engine)


def initialize_database(engine):
    """Initialize database schema"""
    Base.metadata.create_all(engine)


# Utility functions for data management
def get_patient_summary(session, patient_id: str) -> Dict[str, Any]:
    """Get comprehensive patient summary"""
    from sqlalchemy.orm import joinedload

    patient = session.query(Patient).options(
        joinedload(Patient.vital_signs),
        joinedload(Patient.conditions),
        joinedload(Patient.medications),
        joinedload(Patient.appointments)
    ).filter_by(patient_id=patient_id).first()

    if not patient:
        return None

    return {
        'patient_id': patient.patient_id,
        'full_name': patient.full_name,
        'age': patient.age,
        'gender': patient.gender.value if patient.gender else None,
        'blood_type': patient.blood_type.value if patient.blood_type else None,
        'active_conditions': len([c for c in patient.conditions if c.status == 'active']),
        'current_medications': len([m for m in patient.medications if m.active]),
        'upcoming_appointments': len([a for a in patient.appointments if a.status == 'scheduled' and not a.is_past]),
        'latest_vitals': patient.vital_signs[0].__dict__ if patient.vital_signs else None,
        'bmi': patient.bmi,
        'last_updated': patient.updated_at
    }


def get_recent_activity(session, patient_id: str, days: int = 30) -> List[Dict[str, Any]]:
    """Get recent patient activity"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    # Get recent appointments
    appointments = session.query(Appointment).filter(
        Appointment.patient_id == patient_id,
        Appointment.scheduled_date >= cutoff_date
    ).order_by(Appointment.scheduled_date.desc()).limit(10).all()

    # Get recent lab results
    lab_results = session.query(LabResult).filter(
        LabResult.patient_id == patient_id,
        LabResult.reported_date >= cutoff_date
    ).order_by(LabResult.reported_date.desc()).limit(10).all()

    # Get recent vital signs
    vitals = session.query(VitalSigns).filter(
        VitalSigns.patient_id == patient_id,
        VitalSigns.timestamp >= cutoff_date
    ).order_by(VitalSigns.timestamp.desc()).limit(10).all()

    activity = []

    # Add appointments
    for apt in appointments:
        activity.append({
            'type': 'appointment',
            'date': apt.scheduled_date,
            'description': f"Appointment: {apt.title}",
            'provider': apt.provider.first_name + ' ' + apt.provider.last_name if apt.provider else 'Unknown'
        })

    # Add lab results
    for lab in lab_results:
        activity.append({
            'type': 'lab_result',
            'date': lab.reported_date,
            'description': f"Lab Result: {lab.test_name}",
            'value': lab.result_value,
            'interpretation': lab.interpretation.value if lab.interpretation else 'pending'
        })

    # Add vital signs
    for vital in vitals:
        activity.append({
            'type': 'vital_signs',
            'date': vital.timestamp,
            'description': f"Vital Signs Recorded",
            'details': f"BP: {vital.blood_pressure_systolic}/{vital.blood_pressure_diastolic}, HR: {vital.heart_rate}"
        })

    # Sort by date and return recent activity
    activity.sort(key=lambda x: x['date'], reverse=True)
    return activity[:20]


def get_population_health_metrics(session, condition_filter: str = None) -> Dict[str, Any]:
    """Get population health metrics"""
    # Total patients
    total_patients = session.query(Patient).count()

    # Active conditions
    active_conditions = session.query(patient_condition).filter_by(status='active').count()

    # Current medications
    current_medications = session.query(patient_medication).filter_by(active=True).count()

    # Recent appointments
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_appointments = session.query(Appointment).filter(
        Appointment.scheduled_date >= thirty_days_ago
    ).count()

    # Average age
    avg_age_result = session.query(func.avg(
        func.extract('year', func.age(Patient.date_of_birth))
    )).scalar()
    avg_age = float(avg_age_result) if avg_age_result else 0

    # BMI distribution
    bmi_stats = session.query(
        func.avg(Patient.bmi),
        func.min(Patient.bmi),
        func.max(Patient.bmi)
    ).filter(Patient.bmi.isnot(None)).first()

    return {
        'total_patients': total_patients,
        'active_conditions_per_patient': active_conditions / total_patients if total_patients > 0 else 0,
        'medications_per_patient': current_medications / total_patients if total_patients > 0 else 0,
        'appointments_per_month': recent_appointments / 30 * 30,  # Normalized to per month
        'average_age': round(avg_age, 1),
        'bmi_average': round(bmi_stats[0], 1) if bmi_stats[0] else None,
        'bmi_range': {
            'min': round(bmi_stats[1], 1) if bmi_stats[1] else None,
            'max': round(bmi_stats[2], 1) if bmi_stats[2] else None
        }
    }


# Export all models and utilities
__all__ = [
    'Base',
    'User', 'Patient', 'MedicalCondition', 'Medication', 'Allergy',
    'VitalSigns', 'Appointment', 'GenomicAnalysis', 'GeneticVariant',
    'LabResult', 'ImagingStudy', 'ImagingSeries', 'ClinicalTrial',
    'TrialParticipant', 'AuditLog',
    'create_database_engine', 'create_session_factory', 'initialize_database',
    'get_patient_summary', 'get_recent_activity', 'get_population_health_metrics',
    'UserRole', 'Gender', 'BloodType', 'AppointmentStatus', 'GenomicAnalysisStatus'
]
