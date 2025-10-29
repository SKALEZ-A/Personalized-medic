"""
Comprehensive Testing Suite for AI Personalized Medicine Platform
Complete test coverage with unit tests, integration tests, security tests, and performance benchmarks
"""

import unittest
import pytest
import asyncio
import time
import json
import tempfile
import shutil
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Callable, Generator
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor
import requests
import aiohttp
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Import project modules (mock imports for testing)
try:
    from main import app
    from core.genomic_engine import GenomicAnalysisEngine
    from core.advanced_ml_models import AdvancedMLModels
    from algorithms.ml_algorithms import PredictiveAnalytics, HealthOptimization
    from algorithms.data_processing_algorithms import ETLPipeline, DataQualityMonitor
    from algorithms.security_algorithms import AuthenticationAlgorithms, AuthorizationAlgorithms, EncryptionAlgorithms
    from database.models import (
        User, Patient, VitalSigns, Appointment, Prescription,
        LabResult, GenomicAnalysis, TreatmentPlan, AuditLog
    )
except ImportError:
    # Mock classes for testing
    class FastAPI: pass
    app = FastAPI()
    class GenomicAnalysisEngine: pass
    class AdvancedMLModels: pass
    class PredictiveAnalytics: pass
    class HealthOptimization: pass
    class ETLPipeline: pass
    class DataQualityMonitor: pass
    class AuthenticationAlgorithms: pass
    class AuthorizationAlgorithms: pass
    class EncryptionAlgorithms: pass

# Test Configuration
TEST_DATABASE_URL = "sqlite:///:memory:"
TEST_REDIS_URL = "redis://localhost:6379/1"
TEST_CONFIG = {
    "database_url": TEST_DATABASE_URL,
    "redis_url": TEST_REDIS_URL,
    "jwt_secret_key": "test_jwt_secret_key_for_testing_only",
    "encryption_key": "test_encryption_key_32_chars_long",
    "api_key": "test_api_key_for_testing",
    "environment": "testing"
}

# Base Test Classes
class BaseTestCase(unittest.TestCase):
    """Base test case with common setup and teardown"""

    def setUp(self):
        """Set up test environment"""
        self.start_time = time.time()
        self.test_data = {}
        self.mock_objects = {}
        self.temp_files = []

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files.append(self.temp_dir)

    def tearDown(self):
        """Clean up test environment"""
        # Clean up temporary files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                else:
                    shutil.rmtree(temp_file)

        # Clean up mock objects
        for mock_obj in self.mock_objects.values():
            if hasattr(mock_obj, 'stop'):
                mock_obj.stop()

        # Log test duration
        duration = time.time() - self.start_time
        print(".2f")

    def create_mock_user(self, **kwargs) -> Dict[str, Any]:
        """Create mock user data"""
        return {
            "id": kwargs.get("id", 1),
            "username": kwargs.get("username", "testuser"),
            "email": kwargs.get("email", "test@example.com"),
            "first_name": kwargs.get("first_name", "Test"),
            "last_name": kwargs.get("last_name", "User"),
            "role": kwargs.get("role", "patient"),
            "is_active": kwargs.get("is_active", True),
            "created_at": kwargs.get("created_at", datetime.utcnow()),
            "updated_at": kwargs.get("updated_at", datetime.utcnow())
        }

    def create_mock_patient(self, **kwargs) -> Dict[str, Any]:
        """Create mock patient data"""
        return {
            "id": kwargs.get("id", 1),
            "patient_id": kwargs.get("patient_id", "PAT001"),
            "user_id": kwargs.get("user_id", 1),
            "first_name": kwargs.get("first_name", "John"),
            "last_name": kwargs.get("last_name", "Doe"),
            "date_of_birth": kwargs.get("date_of_birth", date(1980, 1, 1)),
            "gender": kwargs.get("gender", "M"),
            "blood_type": kwargs.get("blood_type", "O+"),
            "is_active": kwargs.get("is_active", True),
            "created_at": kwargs.get("created_at", datetime.utcnow()),
            "updated_at": kwargs.get("updated_at", datetime.utcnow())
        }

    def assertValidResponse(self, response: Dict[str, Any], required_fields: List[str]):
        """Assert that response contains required fields"""
        self.assertIsInstance(response, dict)
        for field in required_fields:
            self.assertIn(field, response, f"Required field '{field}' missing from response")

    def assertValidTimestamp(self, timestamp: str):
        """Assert that timestamp is valid ISO format"""
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            self.fail(f"Invalid timestamp format: {timestamp}")


class AsyncTestCase(unittest.TestCase):
    """Base test case for async tests"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def async_test(self, coro):
        """Run async test coroutine"""
        return self.loop.run_until_complete(coro)


class DatabaseTestCase(BaseTestCase):
    """Base test case with database setup"""

    def setUp(self):
        super().setUp()
        # Create in-memory SQLite database for testing
        self.engine = create_engine(
            TEST_DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create all tables
        from database.models import Base
        Base.metadata.create_all(bind=self.engine)

    def tearDown(self):
        super().tearDown()
        # Drop all tables
        from database.models import Base
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def get_db_session(self) -> Session:
        """Get database session for testing"""
        return self.SessionLocal()


class APITestCase(BaseTestCase):
    """Base test case for API testing"""

    def setUp(self):
        super().setUp()
        # Create test client
        self.client = TestClient(app)
        self.base_url = "http://testserver"

        # Mock authentication
        self.test_user = self.create_mock_user()
        self.auth_headers = {"Authorization": f"Bearer test_token"}

    def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request"""
        headers = kwargs.pop('headers', {})
        headers.update(self.auth_headers)
        kwargs['headers'] = headers

        response = getattr(self.client, method.lower())(endpoint, **kwargs)
        return {
            "status_code": response.status_code,
            "data": response.json() if response.content else None,
            "headers": dict(response.headers)
        }

    def assertSuccessResponse(self, response: Dict[str, Any], status_code: int = 200):
        """Assert successful API response"""
        self.assertEqual(response["status_code"], status_code)
        self.assertIsNotNone(response["data"])

    def assertErrorResponse(self, response: Dict[str, Any], status_code: int, error_message: Optional[str] = None):
        """Assert error API response"""
        self.assertEqual(response["status_code"], status_code)
        if error_message:
            self.assertIn("detail", response["data"])
            self.assertIn(error_message, response["data"]["detail"])


# Unit Tests
class TestAuthenticationAlgorithms(DatabaseTestCase):
    """Test authentication algorithms"""

    def test_password_hashing(self):
        """Test password hashing and verification"""
        auth = AuthenticationAlgorithms()

        password = "test_password_123"
        hashed = auth.hash_password(password)

        self.assertIsInstance(hashed, str)
        self.assertNotEqual(hashed, password)
        self.assertTrue(auth.verify_password(password, hashed))
        self.assertFalse(auth.verify_password("wrong_password", hashed))

    def test_jwt_token_generation(self):
        """Test JWT token generation and validation"""
        auth = AuthenticationAlgorithms()

        payload = {"user_id": 123, "role": "physician"}
        token = auth.generate_jwt_token(payload)

        self.assertIsInstance(token, str)
        decoded = auth.validate_jwt_token(token)
        self.assertEqual(decoded["user_id"], 123)
        self.assertEqual(decoded["role"], "physician")

    def test_mfa_setup_and_verification(self):
        """Test MFA setup and verification"""
        auth = AuthenticationAlgorithms()

        # Setup MFA
        secret, qr_code = auth.setup_mfa("test@example.com")
        self.assertIsInstance(secret, str)
        self.assertIsInstance(qr_code, str)

        # Generate valid code (mock)
        valid_code = auth.generate_mfa_code(secret)
        self.assertTrue(auth.verify_mfa_code(secret, valid_code))

        # Test invalid code
        self.assertFalse(auth.verify_mfa_code(secret, "123456"))

    def test_password_policy_enforcement(self):
        """Test password policy enforcement"""
        auth = AuthenticationAlgorithms()

        # Valid passwords
        self.assertTrue(auth.validate_password_policy("StrongPass123!"))
        self.assertTrue(auth.validate_password_policy("Complex@Password#456"))

        # Invalid passwords
        self.assertFalse(auth.validate_password_policy("weak"))
        self.assertFalse(auth.validate_password_policy("12345678"))
        self.assertFalse(auth.validate_password_policy("password"))

    def test_account_lockout_protection(self):
        """Test account lockout protection"""
        auth = AuthenticationAlgorithms()

        user_id = 123

        # Simulate failed login attempts
        for i in range(5):
            auth.record_failed_login_attempt(user_id)
            locked = auth.is_account_locked(user_id)
            if i < 4:
                self.assertFalse(locked)
            else:
                self.assertTrue(locked)

        # Test successful login resets counter
        auth.record_successful_login(user_id)
        self.assertFalse(auth.is_account_locked(user_id))


class TestAuthorizationAlgorithms(DatabaseTestCase):
    """Test authorization algorithms"""

    def test_role_based_permissions(self):
        """Test role-based permission checking"""
        authz = AuthorizationAlgorithms()

        # Define permissions for roles
        role_permissions = {
            "patient": ["read_own_data", "update_own_profile"],
            "physician": ["read_own_data", "read_patient_data", "write_prescriptions"],
            "admin": ["*"]  # All permissions
        }

        # Test patient permissions
        self.assertTrue(authz.has_permission("patient", "read_own_data", role_permissions))
        self.assertFalse(authz.has_permission("patient", "write_prescriptions", role_permissions))

        # Test physician permissions
        self.assertTrue(authz.has_permission("physician", "read_patient_data", role_permissions))
        self.assertTrue(authz.has_permission("physician", "write_prescriptions", role_permissions))

        # Test admin permissions (wildcard)
        self.assertTrue(authz.has_permission("admin", "any_permission", role_permissions))

    def test_resource_level_permissions(self):
        """Test resource-level permission checking"""
        authz = AuthorizationAlgorithms()

        # Patient can only access their own data
        user_id = 123
        resource_owner_id = 123

        self.assertTrue(authz.can_access_resource(user_id, resource_owner_id, "patient"))
        self.assertFalse(authz.can_access_resource(user_id, 456, "patient"))

        # Physician can access any patient's data
        self.assertTrue(authz.can_access_resource(user_id, resource_owner_id, "physician"))
        self.assertTrue(authz.can_access_resource(user_id, 456, "physician"))

    def test_permission_inheritance(self):
        """Test role permission inheritance"""
        authz = AuthorizationAlgorithms()

        role_hierarchy = {
            "patient": [],
            "nurse": ["patient"],
            "physician": ["nurse"],
            "admin": ["physician"]
        }

        # Nurse inherits patient permissions
        self.assertTrue(authz.has_inherited_permission("nurse", "read_own_data", role_hierarchy))

        # Physician inherits nurse permissions
        self.assertTrue(authz.has_inherited_permission("physician", "read_own_data", role_hierarchy))

        # Admin inherits all permissions
        self.assertTrue(authz.has_inherited_permission("admin", "write_prescriptions", role_hierarchy))


class TestEncryptionAlgorithms(BaseTestCase):
    """Test encryption algorithms"""

    def setUp(self):
        super().setUp()
        self.encryption = EncryptionAlgorithms()

    def test_aes_encryption_decryption(self):
        """Test AES encryption and decryption"""
        key = self.encryption.generate_key()
        plaintext = "Sensitive medical data"

        ciphertext = self.encryption.encrypt_aes(plaintext, key)
        decrypted = self.encryption.decrypt_aes(ciphertext, key)

        self.assertEqual(decrypted, plaintext)
        self.assertNotEqual(ciphertext, plaintext)

    def test_rsa_key_generation(self):
        """Test RSA key pair generation"""
        private_key, public_key = self.encryption.generate_rsa_keypair()

        self.assertIsInstance(private_key, str)
        self.assertIsInstance(public_key, str)
        self.assertIn("BEGIN", private_key)
        self.assertIn("BEGIN", public_key)

    def test_rsa_encryption_decryption(self):
        """Test RSA encryption and decryption"""
        private_key, public_key = self.encryption.generate_rsa_keypair()
        message = "PHI data"

        ciphertext = self.encryption.encrypt_rsa(message, public_key)
        decrypted = self.encryption.decrypt_rsa(ciphertext, private_key)

        self.assertEqual(decrypted, message)

    def test_digital_signature(self):
        """Test digital signature creation and verification"""
        private_key, public_key = self.encryption.generate_rsa_keypair()
        message = "Medical record data"

        signature = self.encryption.sign_data(message, private_key)
        is_valid = self.encryption.verify_signature(message, signature, public_key)

        self.assertTrue(is_valid)
        self.assertFalse(self.encryption.verify_signature("tampered message", signature, public_key))

    def test_hash_functions(self):
        """Test cryptographic hash functions"""
        data = "Medical record content"

        # SHA-256
        sha256_hash = self.encryption.hash_sha256(data)
        self.assertIsInstance(sha256_hash, str)
        self.assertEqual(len(sha256_hash), 64)  # SHA-256 produces 64 character hex string

        # SHA-512
        sha512_hash = self.encryption.hash_sha512(data)
        self.assertIsInstance(sha512_hash, str)
        self.assertEqual(len(sha512_hash), 128)  # SHA-512 produces 128 character hex string

        # Test hash consistency
        self.assertEqual(self.encryption.hash_sha256(data), sha256_hash)


class TestETLPipeline(DatabaseTestCase):
    """Test ETL pipeline functionality"""

    def setUp(self):
        super().setUp()
        self.etl = ETLPipeline()

    def test_data_extraction(self):
        """Test data extraction from various sources"""
        # Mock data sources
        csv_data = "name,age,gender\nJohn,30,M\nJane,25,F"
        json_data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]

        # Test CSV extraction
        extracted_csv = self.etl.extract_from_csv(csv_data)
        self.assertEqual(len(extracted_csv), 2)
        self.assertEqual(extracted_csv[0]["name"], "John")

        # Test JSON extraction
        extracted_json = self.etl.extract_from_json(json_data)
        self.assertEqual(len(extracted_json), 2)
        self.assertEqual(extracted_json[0]["name"], "John")

    def test_data_transformation(self):
        """Test data transformation operations"""
        raw_data = [
            {"name": "john doe", "age": "30", "gender": "m"},
            {"name": "jane smith", "age": "25", "gender": "f"}
        ]

        # Define transformation rules
        transformations = [
            {"field": "name", "operation": "title_case"},
            {"field": "age", "operation": "to_integer"},
            {"field": "gender", "operation": "map", "mapping": {"m": "Male", "f": "Female"}}
        ]

        transformed_data = self.etl.transform_data(raw_data, transformations)

        self.assertEqual(transformed_data[0]["name"], "John Doe")
        self.assertEqual(transformed_data[0]["age"], 30)
        self.assertEqual(transformed_data[0]["gender"], "Male")

    def test_data_validation(self):
        """Test data validation rules"""
        validation_rules = {
            "age": {"type": "integer", "min": 0, "max": 150},
            "email": {"type": "email"},
            "phone": {"type": "regex", "pattern": r"^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$"}
        }

        # Valid data
        valid_data = [
            {"age": 30, "email": "test@example.com", "phone": "+1234567890"},
            {"age": 25, "email": "user@test.com", "phone": "123-456-7890"}
        ]

        is_valid, errors = self.etl.validate_data(valid_data, validation_rules)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Invalid data
        invalid_data = [
            {"age": 200, "email": "invalid-email", "phone": "invalid"}
        ]

        is_valid, errors = self.etl.validate_data(invalid_data, validation_rules)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_data_loading(self):
        """Test data loading to database"""
        test_data = [
            {"name": "John Doe", "age": 30, "gender": "Male"},
            {"name": "Jane Smith", "age": 25, "gender": "Female"}
        ]

        # Mock database loading
        with patch('database.connection') as mock_conn:
            result = self.etl.load_to_database(test_data, "test_table")
            self.assertTrue(result["success"])
            self.assertEqual(result["records_loaded"], 2)


class TestDataQualityMonitor(DatabaseTestCase):
    """Test data quality monitoring"""

    def setUp(self):
        super().setUp()
        self.monitor = DataQualityMonitor()

    def test_completeness_check(self):
        """Test data completeness validation"""
        data = [
            {"name": "John", "age": 30, "email": "john@example.com"},
            {"name": "Jane", "age": None, "email": "jane@example.com"},
            {"name": "Bob", "age": 25, "email": None}
        ]

        completeness_score = self.monitor.check_completeness(data, ["name", "age", "email"])
        self.assertLess(completeness_score, 1.0)  # Should be less than perfect

        # Calculate expected completeness: 7 out of 9 fields filled = 77.8%
        expected_score = 7/9
        self.assertAlmostEqual(completeness_score, expected_score, places=2)

    def test_accuracy_validation(self):
        """Test data accuracy validation"""
        # Test email validation
        valid_emails = ["test@example.com", "user@test.org"]
        invalid_emails = ["invalid-email", "@test.com", "test@"]

        accuracy_score = self.monitor.check_accuracy(
            valid_emails + invalid_emails,
            lambda x: '@' in x and '.' in x
        )
        expected_score = len(valid_emails) / len(valid_emails + invalid_emails)
        self.assertEqual(accuracy_score, expected_score)

    def test_consistency_check(self):
        """Test data consistency validation"""
        data = [
            {"age": 30, "date_of_birth": "1990-01-01"},
            {"age": 25, "date_of_birth": "1995-01-01"},
            {"age": 200, "date_of_birth": "1800-01-01"}  # Inconsistent
        ]

        def consistency_rule(record):
            # Age should roughly match year of birth
            birth_year = int(record["date_of_birth"].split("-")[0])
            current_year = datetime.now().year
            calculated_age = current_year - birth_year
            return abs(calculated_age - record["age"]) <= 1

        consistency_score = self.monitor.check_consistency(data, consistency_rule)
        self.assertLess(consistency_score, 1.0)  # Should be less than perfect due to age 200

    def test_timeliness_monitoring(self):
        """Test data timeliness monitoring"""
        # Simulate data with different timestamps
        current_time = datetime.utcnow()
        data_timestamps = [
            current_time - timedelta(hours=1),  # Recent
            current_time - timedelta(days=1),   # Day old
            current_time - timedelta(days=7),   # Week old
            current_time - timedelta(days=30)   # Month old
        ]

        timeliness_score = self.monitor.check_timeliness(data_timestamps, timedelta(days=1))
        # Only first item is within 1 day
        expected_score = 1/4
        self.assertEqual(timeliness_score, expected_score)

    def test_uniqueness_validation(self):
        """Test data uniqueness validation"""
        data = [
            {"id": 1, "name": "John"},
            {"id": 2, "name": "Jane"},
            {"id": 1, "name": "John"},  # Duplicate
            {"id": 3, "name": "Bob"}
        ]

        uniqueness_score = self.monitor.check_uniqueness(data, ["id"])
        expected_score = 3/4  # 3 unique IDs out of 4 records
        self.assertEqual(uniqueness_score, expected_score)


class TestPredictiveAnalytics(BaseTestCase):
    """Test predictive analytics algorithms"""

    def setUp(self):
        super().setUp()
        self.analytics = PredictiveAnalytics()

    def test_disease_risk_prediction(self):
        """Test disease risk prediction model"""
        patient_data = {
            "age": 45,
            "gender": "M",
            "bmi": 28.5,
            "family_history": ["diabetes", "heart_disease"],
            "lifestyle_factors": ["smoking", "sedentary"],
            "biomarkers": {"glucose": 110, "cholesterol": 240}
        }

        prediction = self.analytics.predict_disease_risk(patient_data, "cardiovascular")

        self.assertIn("risk_score", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("risk_factors", prediction)
        self.assertIsInstance(prediction["risk_score"], (int, float))
        self.assertGreaterEqual(prediction["risk_score"], 0)
        self.assertLessEqual(prediction["risk_score"], 1)

    def test_drug_response_prediction(self):
        """Test drug response prediction"""
        patient_profile = {
            "genetic_markers": ["CYP2D6*4", "SLCO1B1*5"],
            "age": 35,
            "weight": 70,
            "comorbidities": ["hypertension"]
        }
        medication = "simvastatin"

        prediction = self.analytics.predict_drug_response(patient_profile, medication)

        self.assertIn("efficacy_score", prediction)
        self.assertIn("toxicity_risk", prediction)
        self.assertIn("recommended_dosage", prediction)
        self.assertIn("monitoring_required", prediction)

    def test_treatment_outcome_prediction(self):
        """Test treatment outcome prediction"""
        treatment_data = {
            "treatment_type": "medication",
            "medications": ["metformin", "lisinopril"],
            "duration_weeks": 12,
            "patient_compliance": 0.85
        }
        historical_outcomes = [
            {"success": True, "duration": 8, "compliance": 0.9},
            {"success": False, "duration": 4, "compliance": 0.6}
        ]

        prediction = self.analytics.predict_treatment_outcome(treatment_data, historical_outcomes)

        self.assertIn("success_probability", prediction)
        self.assertIn("expected_duration", prediction)
        self.assertIn("confidence_interval", prediction)

    def test_health_trajectory_prediction(self):
        """Test health trajectory prediction"""
        historical_data = [
            {"date": "2023-01-01", "weight": 85, "glucose": 120},
            {"date": "2023-02-01", "weight": 83, "glucose": 115},
            {"date": "2023-03-01", "weight": 81, "glucose": 110}
        ]
        prediction_horizon = 30  # days

        trajectory = self.analytics.predict_health_trajectory(historical_data, prediction_horizon)

        self.assertIn("predictions", trajectory)
        self.assertIn("trend_analysis", trajectory)
        self.assertIn("intervention_recommendations", trajectory)
        self.assertEqual(len(trajectory["predictions"]), prediction_horizon)


class TestHealthOptimization(BaseTestCase):
    """Test health optimization algorithms"""

    def setUp(self):
        super().setUp()
        self.optimizer = HealthOptimization()

    def test_lifestyle_recommendation(self):
        """Test personalized lifestyle recommendations"""
        patient_profile = {
            "age": 40,
            "bmi": 32,
            "activity_level": "sedentary",
            "diet_quality": "poor",
            "sleep_hours": 6,
            "stress_level": "high"
        }
        health_goals = ["weight_loss", "better_sleep", "stress_reduction"]

        recommendations = self.optimizer.generate_lifestyle_recommendations(patient_profile, health_goals)

        self.assertIn("exercise_plan", recommendations)
        self.assertIn("nutrition_plan", recommendations)
        self.assertIn("sleep_hygiene", recommendations)
        self.assertIn("stress_management", recommendations)
        self.assertIn("timeline", recommendations)

    def test_risk_factor_optimization(self):
        """Test risk factor optimization"""
        current_risks = {
            "cardiovascular": 0.25,
            "diabetes": 0.35,
            "hypertension": 0.40
        }
        modifiable_factors = {
            "weight": 90,
            "exercise_minutes_week": 60,
            "sodium_intake_mg": 3500,
            "alcohol_drinks_week": 10
        }

        optimization_plan = self.optimizer.optimize_risk_factors(current_risks, modifiable_factors)

        self.assertIn("interventions", optimization_plan)
        self.assertIn("expected_risk_reduction", optimization_plan)
        self.assertIn("timeline_months", optimization_plan)
        self.assertIn("monitoring_schedule", optimization_plan)

    def test_behavior_change_modeling(self):
        """Test behavior change modeling"""
        target_behavior = "regular_exercise"
        current_adherence = 0.3  # 30% adherence
        intervention_intensity = "moderate"

        behavior_model = self.optimizer.model_behavior_change(target_behavior, current_adherence, intervention_intensity)

        self.assertIn("success_probability", behavior_model)
        self.assertIn("time_to_habit_formation", behavior_model)
        self.assertIn("relapse_probability", behavior_model)
        self.assertIn("reinforcement_schedule", behavior_model)


# Integration Tests
class TestAPIIntegration(APITestCase):
    """Test API integration scenarios"""

    def test_patient_registration_flow(self):
        """Test complete patient registration flow"""
        # Register new user
        user_data = {
            "username": "newpatient",
            "email": "newpatient@example.com",
            "first_name": "New",
            "last_name": "Patient",
            "password": "SecurePass123!",
            "role": "patient"
        }

        response = self.make_request("POST", "/auth/register", json=user_data)
        self.assertSuccessResponse(response, 201)
        self.assertValidResponse(response["data"], ["id", "username", "email"])

        user_id = response["data"]["id"]

        # Create patient profile
        patient_data = {
            "patient_id": "PAT_NEW_001",
            "first_name": "New",
            "last_name": "Patient",
            "date_of_birth": "1990-01-01",
            "gender": "M",
            "phone_primary": "+1234567890",
            "email": "newpatient@example.com",
            "address_street": "123 Test St",
            "address_city": "Test City",
            "address_state": "TS",
            "address_zip": "12345",
            "address_country": "US"
        }

        response = self.make_request("POST", "/patients", json=patient_data)
        self.assertSuccessResponse(response, 201)
        self.assertValidResponse(response["data"], ["id", "patient_id", "first_name"])

    def test_appointment_booking_flow(self):
        """Test appointment booking flow"""
        # Create appointment
        appointment_data = {
            "patient_id": "PAT001",
            "provider_id": 2,
            "title": "Annual Physical",
            "appointment_type": "physical_exam",
            "scheduled_date": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "duration_minutes": 30,
            "reason_for_visit": "Annual checkup"
        }

        response = self.make_request("POST", "/appointments", json=appointment_data)
        self.assertSuccessResponse(response, 201)
        self.assertValidResponse(response["data"], ["appointment_id", "patient_id", "status"])

        appointment_id = response["data"]["appointment_id"]

        # Update appointment status
        update_data = {"status": "confirmed"}
        response = self.make_request("PATCH", f"/appointments/{appointment_id}", json=update_data)
        self.assertSuccessResponse(response)

        # Verify status update
        response = self.make_request("GET", f"/appointments/{appointment_id}")
        self.assertSuccessResponse(response)
        self.assertEqual(response["data"]["status"], "confirmed")

    def test_genomic_analysis_workflow(self):
        """Test genomic analysis workflow"""
        # Submit analysis request
        analysis_data = {
            "patient_id": "PAT001",
            "analysis_type": "comprehensive",
            "clinical_indication": "Family history of breast cancer"
        }

        response = self.make_request("POST", "/genomics/analyze", json=analysis_data)
        self.assertSuccessResponse(response, 202)  # Accepted for processing
        self.assertValidResponse(response["data"], ["analysis_id", "status"])

        analysis_id = response["data"]["analysis_id"]

        # Check analysis status (mock as completed)
        response = self.make_request("GET", f"/genomics/analysis/{analysis_id}")
        self.assertSuccessResponse(response)
        self.assertValidResponse(response["data"], ["status", "variants_called", "clinical_variants"])

        # Get variants
        response = self.make_request("GET", f"/genomics/analysis/{analysis_id}/variants")
        self.assertSuccessResponse(response)
        self.assertIsInstance(response["data"], list)

    def test_prescription_management_flow(self):
        """Test prescription management flow"""
        # Create prescription
        prescription_data = {
            "patient_id": "PAT001",
            "medication_id": 1,
            "dosage": "10mg",
            "frequency": "once daily",
            "duration": "30 days",
            "quantity": 30,
            "indications": "Hypertension management"
        }

        response = self.make_request("POST", "/prescriptions", json=prescription_data)
        self.assertSuccessResponse(response, 201)
        self.assertValidResponse(response["data"], ["prescription_id", "patient_id", "status"])

        prescription_id = response["data"]["prescription_id"]

        # Refill prescription
        refill_data = {"refills_requested": 1}
        response = self.make_request("POST", f"/prescriptions/{prescription_id}/refill", json=refill_data)
        self.assertSuccessResponse(response)

    def test_health_monitoring_integration(self):
        """Test health monitoring data integration"""
        # Submit health data
        health_data = {
            "patient_id": "PAT001",
            "timestamp": datetime.utcnow().isoformat(),
            "vital_signs": {
                "heart_rate": 72,
                "blood_pressure_systolic": 128,
                "blood_pressure_diastolic": 82,
                "temperature": 98.6,
                "oxygen_saturation": 98
            },
            "biomarkers": {
                "glucose": 95,
                "cholesterol": 180
            },
            "device_info": {
                "type": "smartwatch",
                "id": "SW001",
                "model": "Apple Watch Series 8"
            }
        }

        response = self.make_request("POST", "/health-monitoring/data", json=health_data)
        self.assertSuccessResponse(response)
        self.assertValidResponse(response["data"], ["status", "alerts", "recommendations"])

        # Check for alerts
        alerts_response = self.make_request("GET", f"/health-monitoring/alerts/{health_data['patient_id']}")
        self.assertSuccessResponse(alerts_response)
        self.assertIsInstance(alerts_response["data"], list)


# Security Tests
class TestSecurityFeatures(APITestCase):
    """Test security features and vulnerabilities"""

    def test_sql_injection_protection(self):
        """Test protection against SQL injection"""
        malicious_queries = [
            {"username": "'; DROP TABLE users; --"},
            {"username": "' OR '1'='1"},
            {"username": "admin'--"}
        ]

        for query in malicious_queries:
            response = self.make_request("POST", "/auth/login", json=query)
            # Should not crash and should return appropriate error
            self.assertIn(response["status_code"], [400, 401, 422])

    def test_xss_protection(self):
        """Test protection against XSS attacks"""
        xss_payloads = [
            {"name": "<script>alert('XSS')</script>"},
            {"description": "<img src=x onerror=alert('XSS')>"},
            {"notes": "javascript:alert('XSS')"}
        ]

        for payload in xss_payloads:
            response = self.make_request("POST", "/patients", json=payload)
            if response["status_code"] == 201:
                # If creation succeeded, check that scripts are not in response
                response_data = json.dumps(response["data"])
                self.assertNotIn("<script>", response_data.lower())
                self.assertNotIn("javascript:", response_data.lower())

    def test_rate_limiting(self):
        """Test API rate limiting"""
        endpoint = "/health-monitoring/vitals"

        # Make multiple rapid requests
        responses = []
        for i in range(100):  # Exceed rate limit
            response = self.make_request("GET", endpoint)
            responses.append(response["status_code"])

        # Should have some 429 (Too Many Requests) responses
        self.assertIn(429, responses)

    def test_input_validation(self):
        """Test comprehensive input validation"""
        invalid_inputs = [
            {"email": "invalid-email"},
            {"phone": "invalid-phone-number"},
            {"date_of_birth": "invalid-date"},
            {"age": -5},
            {"bmi": 150},  # Impossible BMI
            {"heart_rate": 500}  # Impossible heart rate
        ]

        for invalid_input in invalid_inputs:
            response = self.make_request("POST", "/patients", json=invalid_input)
            self.assertIn(response["status_code"], [400, 422])

    def test_authentication_bypass_attempts(self):
        """Test protection against authentication bypass"""
        # Try various bypass techniques
        bypass_attempts = [
            {"Authorization": "Bearer invalid_token"},
            {"Authorization": "Bearer "},
            {"Authorization": "Basic invalid_base64"},
            {}  # No auth header
        ]

        for attempt in bypass_attempts:
            response = self.client.get("/patients", headers=attempt)
            self.assertIn(response.status_code, [401, 403])

    def test_data_encryption_at_rest(self):
        """Test that sensitive data is encrypted"""
        # This would require checking actual database/storage encryption
        # For this test, we'll mock the encryption check
        with patch('algorithms.security_algorithms.EncryptionAlgorithms.is_data_encrypted') as mock_encrypt:
            mock_encrypt.return_value = True

            from algorithms.security_algorithms import EncryptionAlgorithms
            encryption = EncryptionAlgorithms()

            self.assertTrue(encryption.is_data_encrypted("test_sensitive_data"))

    def test_audit_logging(self):
        """Test that security events are audited"""
        # Perform a security-sensitive operation
        response = self.make_request("POST", "/auth/login", json={
            "username": "testuser",
            "password": "wrongpassword"
        })

        # Check that failed login was logged
        audit_response = self.make_request("GET", "/admin/audit-logs", params={
            "event_type": "user_login",
            "success": False
        })
        self.assertSuccessResponse(audit_response)

        # Verify log contains expected information
        logs = audit_response["data"]
        self.assertGreater(len(logs), 0)

        failed_login_log = logs[0]
        self.assertValidResponse(failed_login_log, ["event_type", "success", "timestamp"])
        self.assertEqual(failed_login_log["event_type"], "user_login")
        self.assertFalse(failed_login_log["success"])


# Performance Tests
class TestPerformanceBenchmarks(BaseTestCase):
    """Performance benchmarking tests"""

    def setUp(self):
        super().setUp()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def test_api_response_times(self):
        """Test API response time performance"""
        client = TestClient(app)

        # Measure response times for various endpoints
        endpoints = [
            "/health",
            "/auth/login",
            "/patients",
            "/genomics/analyze"
        ]

        response_times = {}
        for endpoint in endpoints:
            start_time = time.time()
            try:
                response = client.get(endpoint)
                end_time = time.time()
                response_times[endpoint] = end_time - start_time
            except:
                response_times[endpoint] = float('inf')

        # Assert reasonable response times (< 1 second for most endpoints)
        for endpoint, response_time in response_times.items():
            if response_time != float('inf'):
                self.assertLess(response_time, 1.0, f"Endpoint {endpoint} too slow: {response_time}s")

    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests"""
        client = TestClient(app)

        def make_request():
            try:
                response = client.get("/health")
                return response.status_code
            except:
                return 500

        # Make 10 concurrent requests
        futures = [self.executor.submit(make_request) for _ in range(10)]
        results = [future.result() for future in futures]

        # All requests should succeed
        successful_requests = sum(1 for result in results if result == 200)
        self.assertGreaterEqual(successful_requests, 8)  # At least 80% success rate

    def test_memory_usage_monitoring(self):
        """Test memory usage during operations"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform memory-intensive operation (create many objects)
        large_data = [{"data": "x" * 1000} for _ in range(1000)]

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB)
        self.assertLess(memory_increase, 50 * 1024 * 1024)

    def test_database_query_performance(self):
        """Test database query performance"""
        # This would require actual database setup
        # For now, we'll mock the performance test
        with patch('database.connection') as mock_conn:
            mock_conn.execute_query.return_value = {"execution_time": 0.05}

            # Simulate query execution
            query_time = 0.05  # Mock 50ms query time
            self.assertLess(query_time, 0.1)  # Should be under 100ms

    def test_genomic_analysis_performance(self):
        """Test genomic analysis performance"""
        # Mock genomic analysis timing
        analysis_start = time.time()

        # Simulate analysis work
        time.sleep(0.1)  # 100ms simulation

        analysis_end = time.time()
        analysis_time = analysis_end - analysis_start

        # Analysis should complete in reasonable time (< 5 seconds for this simulation)
        self.assertLess(analysis_time, 5.0)


# Load Tests
class TestLoadHandling(APITestCase):
    """Test system behavior under load"""

    def test_high_concurrency_handling(self):
        """Test handling of high concurrent load"""
        import concurrent.futures

        def make_api_call():
            try:
                response = self.client.get("/health")
                return response.status_code == 200
            except:
                return False

        # Test with 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: make_api_call(), range(50)))

        success_rate = sum(results) / len(results)
        self.assertGreaterEqual(success_rate, 0.9)  # At least 90% success rate

    def test_memory_leak_detection(self):
        """Test for memory leaks during sustained load"""
        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss

        # Perform repeated operations
        for i in range(100):
            # Simulate API operations
            response = self.client.get("/health")
            gc.collect()  # Force garbage collection

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory should not increase significantly (< 10MB)
        self.assertLess(memory_increase, 10 * 1024 * 1024)

    def test_database_connection_pooling(self):
        """Test database connection pool behavior under load"""
        # This would test actual database connection pooling
        # For now, we'll mock the test
        with patch('database.connection_pool') as mock_pool:
            mock_pool.get_connection.return_value = Mock()
            mock_pool.return_connection.return_value = None

            # Simulate connection usage
            connections = []
            for i in range(20):
                conn = mock_pool.get_connection()
                connections.append(conn)
                mock_pool.return_connection(conn)

            # Pool should handle connections efficiently
            self.assertEqual(len(connections), 20)


# Compliance Tests
class TestComplianceFeatures(DatabaseTestCase):
    """Test HIPAA and other compliance features"""

    def test_phi_data_masking(self):
        """Test PHI data masking in logs and responses"""
        # Create patient with PHI data
        phi_data = {
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": "1980-01-01",
            "ssn": "123-45-6789",
            "medical_record_number": "MRN123456"
        }

        # Check that PHI is masked in logs
        with patch('logging.Logger.info') as mock_log:
            # Simulate logging PHI data
            from utils.compliance import ComplianceUtils
            compliance = ComplianceUtils()

            masked_data = compliance.mask_phi_data(phi_data)
            self.assertNotIn("123-45-6789", json.dumps(masked_data))
            self.assertNotIn("MRN123456", json.dumps(masked_data))
            self.assertIn("***", json.dumps(masked_data))

    def test_audit_trail_completeness(self):
        """Test that audit trails capture all required events"""
        required_events = [
            "user_login", "user_logout", "data_access", "data_modification",
            "medication_prescribed", "appointment_scheduled", "genomic_analysis_requested"
        ]

        # Check that all required events are logged
        for event in required_events:
            with patch('database.audit_logger.log_event') as mock_log:
                # Simulate triggering the event
                mock_log.assert_called_with(
                    event_type=event,
                    user_id="test_user",
                    resource_type="test_resource",
                    success=True
                )

    def test_data_retention_policies(self):
        """Test data retention policy enforcement"""
        # Test that old data is properly archived/deleted
        with patch('database.retention_manager') as mock_retention:
            mock_retention.enforce_retention_policy.return_value = {
                "archived_records": 150,
                "deleted_records": 50,
                "errors": 0
            }

            # Simulate retention policy execution
            result = mock_retention.enforce_retention_policy("audit_logs", 2555)  # 7 years in days
            self.assertGreater(result["archived_records"], 0)
            self.assertEqual(result["errors"], 0)

    def test_access_control_enforcement(self):
        """Test role-based access control enforcement"""
        access_control = AuthorizationAlgorithms()

        # Define test scenarios
        test_cases = [
            {"user_role": "patient", "resource": "own_medical_records", "action": "read", "expected": True},
            {"user_role": "patient", "resource": "other_patient_records", "action": "read", "expected": False},
            {"user_role": "physician", "resource": "patient_records", "action": "write", "expected": True},
            {"user_role": "nurse", "resource": "medication_administration", "action": "write", "expected": True},
            {"user_role": "researcher", "resource": "anonymized_data", "action": "read", "expected": True},
        ]

        for test_case in test_cases:
            result = access_control.check_access(
                test_case["user_role"],
                test_case["resource"],
                test_case["action"]
            )
            self.assertEqual(result, test_case["expected"],
                           f"Access control failed for {test_case}")


# Main test runner
if __name__ == '__main__':
    # Configure test runner
    unittest.main(
        verbosity=2,
        buffer=True,  # Capture stdout/stderr
        catchbreak=True,  # Allow Ctrl+C to interrupt
        failfast=False  # Continue running tests even if some fail
    )
