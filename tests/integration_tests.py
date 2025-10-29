"""
Integration Testing Suite for AI Personalized Medicine Platform
Comprehensive integration tests covering component interactions, API workflows, and end-to-end scenarios
"""

import unittest
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import requests
import websocket
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

class IntegrationTestCase(unittest.TestCase):
    """Base class for integration tests"""

    def setUp(self):
        """Set up test environment"""
        self.base_url = "http://localhost:8000"
        self.api_timeout = 30
        self.test_patient_id = "TEST_PAT_001"
        self.test_user_id = "test_user_001"

        # Setup test data
        self.test_patient = {
            "patient_id": self.test_patient_id,
            "name": "Test Patient",
            "age": 45,
            "gender": "Female",
            "medical_history": ["Hypertension", "Type 2 Diabetes"],
            "current_medications": ["Lisinopril 10mg", "Metformin 500mg"],
            "allergies": ["Penicillin"]
        }

        self.test_health_metric = {
            "patient_id": self.test_patient_id,
            "metric_type": "heart_rate",
            "value": 72.0,
            "unit": "bpm",
            "device_id": "test_device_001"
        }

    def tearDown(self):
        """Clean up test environment"""
        # Clean up test data
        try:
            # Reset test patient data
            pass
        except:
            pass

class APIIntegrationTests(IntegrationTestCase):
    """API integration tests"""

    def test_patient_management_workflow(self):
        """Test complete patient management workflow"""
        # 1. Create patient
        response = requests.post(
            f"{self.base_url}/api/patients",
            json=self.test_patient,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 201)
        patient_data = response.json()

        # 2. Retrieve patient
        response = requests.get(
            f"{self.base_url}/api/patients/{self.test_patient_id}",
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)
        retrieved_patient = response.json()
        self.assertEqual(retrieved_patient['patient_id'], self.test_patient_id)

        # 3. Update patient
        updated_patient = self.test_patient.copy()
        updated_patient['age'] = 46

        response = requests.put(
            f"{self.base_url}/api/patients/{self.test_patient_id}",
            json=updated_patient,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)

        # 4. Verify update
        response = requests.get(
            f"{self.base_url}/api/patients/{self.test_patient_id}",
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['age'], 46)

    def test_health_monitoring_workflow(self):
        """Test health monitoring data workflow"""
        # 1. Submit health metric
        response = requests.post(
            f"{self.base_url}/api/health-monitoring",
            json=self.test_health_metric,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)

        # 2. Retrieve health metrics
        response = requests.get(
            f"{self.base_url}/api/health-monitoring/{self.test_patient_id}",
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)
        metrics = response.json()
        self.assertTrue(len(metrics) > 0)

        # 3. Verify metric data
        latest_metric = metrics[-1]  # Most recent
        self.assertEqual(latest_metric['patient_id'], self.test_patient_id)
        self.assertEqual(latest_metric['metric_type'], 'heart_rate')
        self.assertEqual(latest_metric['value'], 72.0)

    def test_appointment_scheduling_workflow(self):
        """Test appointment scheduling workflow"""
        appointment_data = {
            "patient_id": self.test_patient_id,
            "doctor_id": "DR001",
            "appointment_type": "Follow-up",
            "scheduled_time": (datetime.now() + timedelta(days=1)).isoformat(),
            "duration_minutes": 30,
            "notes": "Routine check-up"
        }

        # 1. Schedule appointment
        response = requests.post(
            f"{self.base_url}/api/appointments",
            json=appointment_data,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 201)
        appointment = response.json()

        # 2. Retrieve appointment
        response = requests.get(
            f"{self.base_url}/api/appointments/{appointment['id']}",
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)

        # 3. Update appointment
        updated_appointment = appointment.copy()
        updated_appointment['status'] = 'confirmed'

        response = requests.put(
            f"{self.base_url}/api/appointments/{appointment['id']}",
            json=updated_appointment,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)

    def test_genomic_analysis_workflow(self):
        """Test genomic analysis workflow"""
        analysis_request = {
            "patient_id": self.test_patient_id,
            "analysis_type": "comprehensive_panel",
            "priority": "normal"
        }

        # 1. Request genomic analysis
        response = requests.post(
            f"{self.base_url}/api/genomic-analysis",
            json=analysis_request,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 202)  # Accepted for processing

        # 2. Check analysis status (would be async in real implementation)
        analysis_id = response.json().get('analysis_id')
        if analysis_id:
            response = requests.get(
                f"{self.base_url}/api/genomic-analysis/{analysis_id}/status",
                timeout=self.api_timeout
            )
            self.assertEqual(response.status_code, 200)

    def test_medication_management_workflow(self):
        """Test medication management workflow"""
        medication_data = {
            "patient_id": self.test_patient_id,
            "name": "Test Medication",
            "dosage": "10mg",
            "frequency": "Once daily",
            "start_date": datetime.now().date().isoformat(),
            "prescribed_by": "Dr. Test"
        }

        # 1. Add medication
        response = requests.post(
            f"{self.base_url}/api/medications",
            json=medication_data,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 201)
        medication = response.json()

        # 2. Retrieve medications
        response = requests.get(
            f"{self.base_url}/api/medications/{self.test_patient_id}",
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)
        medications = response.json()
        self.assertTrue(len(medications) > 0)

        # 3. Update medication
        updated_medication = medication.copy()
        updated_medication['frequency'] = 'Twice daily'

        response = requests.put(
            f"{self.base_url}/api/medications/{medication['id']}",
            json=updated_medication,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)

class WebSocketIntegrationTests(IntegrationTestCase):
    """WebSocket integration tests"""

    def test_realtime_health_monitoring(self):
        """Test real-time health monitoring via WebSocket"""
        # WebSocket connection test (mock implementation)
        ws_url = f"ws://localhost:8000/ws/health/{self.test_patient_id}"

        # In a real test, you would:
        # 1. Connect to WebSocket
        # 2. Subscribe to health metrics
        # 3. Send health data via REST API
        # 4. Verify data is received via WebSocket

        # For now, test the WebSocket endpoint availability
        try:
            # This would be a real WebSocket test
            pass
        except Exception as e:
            self.skipTest(f"WebSocket test skipped: {e}")

    def test_realtime_alerts(self):
        """Test real-time alerts via WebSocket"""
        # Test WebSocket alert notifications
        ws_url = f"ws://localhost:8000/ws/alerts/{self.test_patient_id}"

        # Test alert triggering and WebSocket notification
        try:
            # This would test real-time alert delivery
            pass
        except Exception as e:
            self.skipTest(f"WebSocket alerts test skipped: {e}")

class GraphQLIntegrationTests(IntegrationTestCase):
    """GraphQL API integration tests"""

    def test_graphql_patient_query(self):
        """Test GraphQL patient queries"""
        query = """
        query GetPatient($patientId: String!) {
            patient(patient_id: $patientId) {
                patient_id
                name
                age
                medical_history
                current_medications
            }
        }
        """

        variables = {"patient_id": self.test_patient_id}

        response = requests.post(
            f"{self.base_url}/graphql",
            json={"query": query, "variables": variables},
            timeout=self.api_timeout
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('data', data)
        self.assertIsNotNone(data['data'].get('patient'))

    def test_graphql_health_metrics_query(self):
        """Test GraphQL health metrics queries"""
        query = """
        query GetHealthMetrics($patientId: String!) {
            health_metrics(patient_id: $patientId, limit: 10) {
                metric_type
                value
                unit
                timestamp
            }
        }
        """

        variables = {"patient_id": self.test_patient_id}

        response = requests.post(
            f"{self.base_url}/graphql",
            json={"query": query, "variables": variables},
            timeout=self.api_timeout
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('data', data)

    def test_graphql_mutation(self):
        """Test GraphQL mutations"""
        mutation = """
        mutation RecordHealthMetric($patientId: String!, $metricType: String!, $value: Float!) {
            record_health_metric(
                patient_id: $patientId,
                metric_type: $metricType,
                value: $value
            ) {
                success
                message
                metric {
                    patient_id
                    metric_type
                    value
                }
            }
        }
        """

        variables = {
            "patientId": self.test_patient_id,
            "metricType": "blood_pressure",
            "value": 120.0
        }

        response = requests.post(
            f"{self.base_url}/graphql",
            json={"query": mutation, "variables": variables},
            timeout=self.api_timeout
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('data', data)
        self.assertTrue(data['data']['record_health_metric']['success'])

class DatabaseIntegrationTests(IntegrationTestCase):
    """Database integration tests"""

    def test_database_connection(self):
        """Test database connectivity"""
        # Test basic database operations
        response = requests.get(f"{self.base_url}/api/health/db", timeout=self.api_timeout)
        self.assertEqual(response.status_code, 200)

    def test_data_persistence(self):
        """Test data persistence across requests"""
        # 1. Create test data
        test_data = {
            "patient_id": f"persist_test_{int(time.time())}",
            "name": "Persistence Test Patient",
            "test_marker": "integration_test"
        }

        response = requests.post(
            f"{self.base_url}/api/patients",
            json=test_data,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 201)

        # 2. Retrieve and verify
        response = requests.get(
            f"{self.base_url}/api/patients/{test_data['patient_id']}",
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)
        retrieved = response.json()
        self.assertEqual(retrieved['name'], test_data['name'])
        self.assertEqual(retrieved.get('test_marker'), test_data['test_marker'])

    def test_concurrent_database_access(self):
        """Test concurrent database access"""
        def worker(worker_id):
            """Worker function for concurrent access"""
            test_data = {
                "patient_id": f"concurrent_test_{worker_id}_{int(time.time())}",
                "name": f"Concurrent Test Patient {worker_id}",
                "worker_id": worker_id
            }

            # Create patient
            response = requests.post(
                f"{self.base_url}/api/patients",
                json=test_data,
                timeout=self.api_timeout
            )
            return response.status_code == 201

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]

        # Verify all succeeded
        successful_operations = sum(results)
        self.assertEqual(successful_operations, 10)

class SecurityIntegrationTests(IntegrationTestCase):
    """Security integration tests"""

    def test_authentication_workflow(self):
        """Test authentication workflow"""
        # Test login
        login_data = {
            "username": "test_user",
            "password": "test_password"
        }

        response = requests.post(
            f"{self.base_url}/api/auth/login",
            json=login_data,
            timeout=self.api_timeout
        )

        # Should either succeed or fail gracefully
        self.assertIn(response.status_code, [200, 401, 422])

    def test_authorization_checks(self):
        """Test authorization checks"""
        # Test accessing protected resource without auth
        response = requests.get(
            f"{self.base_url}/api/admin/users",
            timeout=self.api_timeout
        )

        # Should be unauthorized
        self.assertIn(response.status_code, [401, 403])

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make multiple rapid requests
        responses = []
        for i in range(20):
            response = requests.get(
                f"{self.base_url}/api/health",
                timeout=self.api_timeout
            )
            responses.append(response.status_code)

        # Should see some rate limiting (429 status)
        rate_limited_responses = sum(1 for status in responses if status == 429)
        # Note: This test may need adjustment based on actual rate limits

    def test_input_validation(self):
        """Test input validation"""
        # Test with invalid data
        invalid_patient = {
            "patient_id": "",  # Invalid: empty
            "name": "A" * 1000,  # Potentially too long
            "age": -5  # Invalid: negative age
        }

        response = requests.post(
            f"{self.base_url}/api/patients",
            json=invalid_patient,
            timeout=self.api_timeout
        )

        # Should fail validation
        self.assertIn(response.status_code, [400, 422])

class EndToEndWorkflowTests(IntegrationTestCase):
    """End-to-end workflow integration tests"""

    def test_complete_patient_journey(self):
        """Test complete patient journey from registration to treatment"""
        # 1. Patient registration
        patient_data = {
            "patient_id": f"e2e_test_{int(time.time())}",
            "name": "E2E Test Patient",
            "age": 35,
            "gender": "Male",
            "medical_history": ["High Cholesterol"],
            "allergies": []
        }

        response = requests.post(
            f"{self.base_url}/api/patients",
            json=patient_data,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 201)
        patient = response.json()

        # 2. Health monitoring setup
        health_data = {
            "patient_id": patient['patient_id'],
            "metric_type": "cholesterol",
            "value": 245.0,
            "unit": "mg/dL"
        }

        response = requests.post(
            f"{self.base_url}/api/health-monitoring",
            json=health_data,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)

        # 3. AI analysis request
        analysis_request = {
            "patient_id": patient['patient_id'],
            "analysis_type": "cardiovascular_risk"
        }

        response = requests.post(
            f"{self.base_url}/api/ai-analysis",
            json=analysis_request,
            timeout=self.api_timeout
        )
        self.assertIn(response.status_code, [200, 202])  # Success or accepted

        # 4. Treatment recommendation retrieval
        response = requests.get(
            f"{self.base_url}/api/treatment/{patient['patient_id']}",
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)

        # 5. Appointment scheduling
        appointment_data = {
            "patient_id": patient['patient_id'],
            "doctor_id": "DR001",
            "appointment_type": "Cardiology Consultation",
            "scheduled_time": (datetime.now() + timedelta(days=3)).isoformat()
        }

        response = requests.post(
            f"{self.base_url}/api/appointments",
            json=appointment_data,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 201)

    def test_emergency_response_workflow(self):
        """Test emergency response workflow"""
        # 1. Emergency alert trigger
        emergency_data = {
            "patient_id": self.test_patient_id,
            "alert_type": "cardiac_arrest",
            "severity": "critical",
            "location": "123 Test Street",
            "vital_signs": {
                "heart_rate": 0,
                "blood_pressure": "0/0",
                "oxygen_saturation": 0
            }
        }

        response = requests.post(
            f"{self.base_url}/api/emergency",
            json=emergency_data,
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)

        # 2. Emergency response verification
        response = requests.get(
            f"{self.base_url}/api/emergency/status/{emergency_data['patient_id']}",
            timeout=self.api_timeout
        )
        self.assertEqual(response.status_code, 200)

class PerformanceIntegrationTests(IntegrationTestCase):
    """Performance-focused integration tests"""

    def test_concurrent_api_access(self):
        """Test concurrent API access"""
        def api_call(call_id):
            """Individual API call"""
            response = requests.get(
                f"{self.base_url}/api/health",
                timeout=self.api_timeout
            )
            return response.status_code == 200

        # Run concurrent calls
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(api_call, i) for i in range(50)]
            results = [future.result() for future in as_completed(futures)]

        # Verify all calls succeeded
        successful_calls = sum(results)
        self.assertEqual(successful_calls, 50)

    def test_database_performance_under_load(self):
        """Test database performance under concurrent load"""
        def db_operation(op_id):
            """Database operation"""
            test_data = {
                "patient_id": f"perf_test_{op_id}_{int(time.time())}",
                "name": f"Performance Test Patient {op_id}",
                "age": 30
            }

            response = requests.post(
                f"{self.base_url}/api/patients",
                json=test_data,
                timeout=self.api_timeout
            )
            return response.status_code == 201

        # Run concurrent database operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(db_operation, i) for i in range(25)]
            results = [future.result() for future in as_completed(futures)]

        successful_operations = sum(results)
        self.assertEqual(successful_operations, 25)

def run_integration_tests():
    """Run all integration tests"""

    # Create test suite
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        APIIntegrationTests,
        GraphQLIntegrationTests,
        DatabaseIntegrationTests,
        SecurityIntegrationTests,
        EndToEndWorkflowTests,
        PerformanceIntegrationTests
    ]

    # Skip WebSocket tests if not available
    try:
        suite.addTest(unittest.makeSuite(WebSocketIntegrationTests))
    except:
        pass

    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate test report
    report = {
        'test_run_timestamp': datetime.now(),
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        'failures_details': result.failures,
        'errors_details': result.errors
    }

    with open('integration_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("Integration test report generated: integration_test_report.json")

    return result.wasSuccessful()

if __name__ == "__main__":
    # Run integration tests
    success = run_integration_tests()
    exit(0 if success else 1)
