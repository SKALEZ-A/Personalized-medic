"""
EHR Integration System for AI Personalized Medicine Platform
Provides comprehensive integration with Electronic Health Record systems
"""

import json
import xml.etree.ElementTree as ET
import requests
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict
import random
import hashlib
import hmac
import base64
from urllib.parse import urljoin, urlparse
import aiohttp
import logging

class EHRIntegrationSystem:
    """Comprehensive EHR integration system"""

    def __init__(self):
        self.connected_systems = {}
        self.integration_profiles = {}
        self.data_mappings = {}
        self.webhook_handlers = {}
        self.sync_queues = defaultdict(asyncio.Queue)
        self.is_running = False
        self.integration_workers = []
        self.initialize_ehr_systems()

    def initialize_ehr_systems(self):
        """Initialize supported EHR systems"""
        self.supported_ehr_systems = {
            "epic": {
                "name": "Epic Systems",
                "api_version": "R2023",
                "auth_method": "oauth2",
                "data_formats": ["FHIR", "HL7", "CCD"],
                "endpoints": {
                    "patient": "/api/FHIR/R4/Patient",
                    "encounter": "/api/FHIR/R4/Encounter",
                    "observation": "/api/FHIR/R4/Observation",
                    "medication": "/api/FHIR/R4/MedicationRequest"
                }
            },
            "cerner": {
                "name": "Cerner Corporation",
                "api_version": "v1",
                "auth_method": "bearer_token",
                "data_formats": ["FHIR", "HL7"],
                "endpoints": {
                    "patient": "/Patient",
                    "encounter": "/Encounter",
                    "observation": "/Observation",
                    "medication": "/Medication"
                }
            },
            "allscripts": {
                "name": "Allscripts",
                "api_version": "2018",
                "auth_method": "basic_auth",
                "data_formats": ["CCD", "C32"],
                "endpoints": {
                    "patient": "/api/patient",
                    "encounter": "/api/encounter",
                    "observation": "/api/observation",
                    "medication": "/api/medication"
                }
            },
            "meditech": {
                "name": "MEDITECH",
                "api_version": "6.0",
                "auth_method": "api_key",
                "data_formats": ["HL7", "XML"],
                "endpoints": {
                    "patient": "/api/v1/patients",
                    "encounter": "/api/v1/encounters",
                    "observation": "/api/v1/observations",
                    "medication": "/api/v1/medications"
                }
            }
        }

        # Initialize data mapping templates
        self._initialize_data_mappings()

        print("🏥 EHR integration system initialized")

    def _initialize_data_mappings(self):
        """Initialize data mapping templates for different EHR formats"""
        self.data_mappings = {
            "fhir_to_internal": {
                "Patient": {
                    "id": "patient_id",
                    "name[0].given[0]": "first_name",
                    "name[0].family[0]": "last_name",
                    "birthDate": "date_of_birth",
                    "gender": "gender",
                    "address[0].line[0]": "address_line1",
                    "address[0].city": "city",
                    "address[0].state": "state",
                    "address[0].postalCode": "zip_code",
                    "telecom[0].value": "phone",
                    "identifier[0].value": "medical_record_number"
                },
                "Observation": {
                    "subject.reference": "patient_id",
                    "code.coding[0].code": "observation_code",
                    "code.coding[0].display": "observation_name",
                    "valueQuantity.value": "value",
                    "valueQuantity.unit": "unit",
                    "effectiveDateTime": "observation_date",
                    "status": "status"
                },
                "MedicationRequest": {
                    "subject.reference": "patient_id",
                    "medicationCodeableConcept.coding[0].code": "medication_code",
                    "medicationCodeableConcept.coding[0].display": "medication_name",
                    "dosageInstruction[0].text": "dosage_instructions",
                    "dispenseRequest.quantity.value": "quantity",
                    "authoredOn": "prescribed_date",
                    "status": "status"
                }
            },
            "hl7_to_internal": {
                "PID": {  # Patient Identification
                    "3": "patient_id",  # Patient ID
                    "5.1": "first_name",  # Patient Name - Family Name
                    "5.2": "last_name",  # Patient Name - Given Name
                    "7": "date_of_birth",  # Date/Time of Birth
                    "8": "gender",  # Administrative Sex
                    "11.1": "address_line1",  # Patient Address - Street
                    "11.3": "city",  # Patient Address - City
                    "11.4": "state",  # Patient Address - State
                    "11.5": "zip_code",  # Patient Address - Zip
                    "13.1": "phone"  # Phone Number - Telephone Number
                },
                "OBR": {  # Observation Request
                    "3": "observation_id",  # Filler Order Number
                    "4.1": "observation_code",  # Universal Service ID - Identifier
                    "4.2": "observation_name",  # Universal Service ID - Text
                    "7": "observation_date",  # Observation Date/Time
                    "25": "result_status"  # Result Status
                },
                "OBX": {  # Observation Result
                    "3.1": "test_code",  # Observation Identifier - Identifier
                    "3.2": "test_name",  # Observation Identifier - Text
                    "5": "value",  # Observation Value
                    "6": "unit",  # Units
                    "14": "result_date"  # Date/Time of the Observation
                }
            }
        }

    def start_integration_services(self):
        """Start EHR integration services"""
        self.is_running = True

        # Start integration workers
        for i in range(6):  # 6 worker threads for different EHR systems
            worker = threading.Thread(target=self._integration_worker, daemon=True)
            worker.start()
            self.integration_workers.append(worker)

        # Start webhook listener
        webhook_worker = threading.Thread(target=self._webhook_listener, daemon=True)
        webhook_worker.start()
        self.integration_workers.append(webhook_worker)

        # Start data synchronization
        sync_worker = threading.Thread(target=self._data_sync_worker, daemon=True)
        sync_worker.start()
        self.integration_workers.append(sync_worker)

        print("🔗 EHR integration services started")

    def stop_integration_services(self):
        """Stop integration services"""
        self.is_running = False
        print("🛑 EHR integration services stopped")

    def connect_ehr_system(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to an EHR system"""
        system_type = system_config["system_type"]
        system_name = system_config["system_name"]

        if system_type not in self.supported_ehr_systems:
            raise ValueError(f"Unsupported EHR system: {system_type}")

        # Create connection profile
        connection_profile = {
            "system_id": f"{system_type}_{system_name}_{int(time.time())}",
            "system_type": system_type,
            "system_name": system_name,
            "base_url": system_config["base_url"],
            "auth_config": system_config["auth_config"],
            "connected_at": datetime.now(),
            "status": "connecting",
            "supported_formats": self.supported_ehr_systems[system_type]["data_formats"],
            "rate_limits": system_config.get("rate_limits", {"requests_per_hour": 1000}),
            "webhook_url": system_config.get("webhook_url"),
            "sync_schedule": system_config.get("sync_schedule", "hourly")
        }

        # Test connection
        try:
            self._test_ehr_connection(connection_profile)
            connection_profile["status"] = "connected"
            connection_profile["last_health_check"] = datetime.now()

            # Store connection
            self.connected_systems[connection_profile["system_id"]] = connection_profile

            # Register webhooks if supported
            if connection_profile.get("webhook_url"):
                self._register_webhooks(connection_profile)

            print(f"✅ Connected to EHR system: {system_name}")
            return connection_profile

        except Exception as e:
            connection_profile["status"] = "failed"
            connection_profile["error"] = str(e)
            print(f"❌ Failed to connect to EHR system: {system_name} - {e}")
            raise

    def _test_ehr_connection(self, connection_profile: Dict[str, Any]):
        """Test connection to EHR system"""
        # This would make actual API calls in production
        # For simulation, we'll just validate the configuration
        required_fields = ["base_url", "auth_config"]
        for field in required_fields:
            if field not in connection_profile:
                raise ValueError(f"Missing required field: {field}")

        # Simulate connection delay
        time.sleep(0.1)

    def _register_webhooks(self, connection_profile: Dict[str, Any]):
        """Register webhooks with EHR system"""
        webhook_events = [
            "patient.created", "patient.updated", "encounter.created",
            "observation.created", "medication.prescribed", "diagnosis.updated"
        ]

        for event in webhook_events:
            webhook_id = f"{connection_profile['system_id']}_{event}_{int(time.time())}"
            self.webhook_handlers[webhook_id] = {
                "system_id": connection_profile["system_id"],
                "event": event,
                "callback_url": connection_profile["webhook_url"],
                "secret": secrets.token_hex(32),  # Webhook secret for verification
                "registered_at": datetime.now(),
                "status": "active"
            }

    def retrieve_patient_data(self, system_id: str, patient_id: str,
                            data_types: List[str] = None) -> Dict[str, Any]:
        """Retrieve patient data from connected EHR system"""
        if system_id not in self.connected_systems:
            raise ValueError(f"EHR system not connected: {system_id}")

        system = self.connected_systems[system_id]
        if system["status"] != "connected":
            raise ValueError(f"EHR system not available: {system_id}")

        if data_types is None:
            data_types = ["patient", "encounters", "observations", "medications"]

        patient_data = {
            "patient_id": patient_id,
            "ehr_system": system_id,
            "retrieved_at": datetime.now(),
            "data": {}
        }

        try:
            # Retrieve data from EHR system
            for data_type in data_types:
                data = self._fetch_ehr_data(system, data_type, patient_id)
                patient_data["data"][data_type] = data

            # Transform data to internal format
            transformed_data = self._transform_ehr_data(patient_data, system["system_type"])

            return transformed_data

        except Exception as e:
            print(f"Error retrieving patient data: {e}")
            patient_data["error"] = str(e)
            return patient_data

    def _fetch_ehr_data(self, system: Dict[str, Any], data_type: str, patient_id: str) -> List[Dict[str, Any]]:
        """Fetch data from EHR system API"""
        system_type = system["system_type"]
        base_url = system["base_url"]
        endpoints = self.supported_ehr_systems[system_type]["endpoints"]

        if data_type not in endpoints:
            return []

        endpoint = endpoints[data_type]

        # Construct API URL
        if system_type == "epic":
            url = urljoin(base_url, f"{endpoint}?subject=Patient/{patient_id}")
        elif system_type == "cerner":
            url = urljoin(base_url, f"{endpoint}?patient={patient_id}")
        else:
            url = urljoin(base_url, f"{endpoint}/{patient_id}")

        # In production, this would make actual HTTP requests
        # For simulation, return mock data
        return self._generate_mock_ehr_data(data_type, patient_id)

    def _generate_mock_ehr_data(self, data_type: str, patient_id: str) -> List[Dict[str, Any]]:
        """Generate mock EHR data for testing"""
        if data_type == "patient":
            return [{
                "id": patient_id,
                "name": [{"given": ["John"], "family": ["Doe"]}],
                "birthDate": "1980-01-15",
                "gender": "male",
                "address": [{"line": ["123 Main St"], "city": "Anytown", "state": "CA", "postalCode": "12345"}],
                "telecom": [{"value": "555-0123"}]
            }]
        elif data_type == "encounters":
            return [
                {
                    "id": f"enc_{i}",
                    "status": "finished",
                    "class": {"code": "AMB"},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "period": {"start": "2024-01-15T09:00:00Z", "end": "2024-01-15T10:00:00Z"},
                    "reasonCode": [{"text": f"Visit reason {i}"}]
                }
                for i in range(3)
            ]
        elif data_type == "observations":
            observations = [
                {"code": "8480-6", "display": "Systolic Blood Pressure", "value": 120, "unit": "mmHg"},
                {"code": "8462-4", "display": "Diastolic Blood Pressure", "value": 80, "unit": "mmHg"},
                {"code": "2093-3", "display": "Total Cholesterol", "value": 180, "unit": "mg/dL"},
                {"code": "2571-8", "display": "Triglycerides", "value": 150, "unit": "mg/dL"}
            ]
            return [
                {
                    "id": f"obs_{i}",
                    "status": "final",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "code": {"coding": [{"code": obs["code"], "display": obs["display"]}]},
                    "valueQuantity": {"value": obs["value"], "unit": obs["unit"]},
                    "effectiveDateTime": "2024-01-15T09:30:00Z"
                }
                for i, obs in enumerate(observations)
            ]
        elif data_type == "medications":
            medications = [
                {"code": "314076", "display": "Lisinopril 10mg", "dosage": "Take one tablet daily"},
                {"code": "197361", "display": "Metformin 500mg", "dosage": "Take two tablets twice daily"}
            ]
            return [
                {
                    "id": f"med_{i}",
                    "status": "active",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "medicationCodeableConcept": {"coding": [{"code": med["code"], "display": med["display"]}]},
                    "dosageInstruction": [{"text": med["dosage"]}],
                    "authoredOn": "2024-01-15T09:00:00Z"
                }
                for i, med in enumerate(medications)
            ]

        return []

    def _transform_ehr_data(self, ehr_data: Dict[str, Any], system_type: str) -> Dict[str, Any]:
        """Transform EHR data to internal format"""
        transformed = {
            "patient_id": ehr_data["patient_id"],
            "ehr_system": ehr_data["ehr_system"],
            "transformed_at": datetime.now(),
            "data": {}
        }

        # Apply data mappings
        if system_type in ["epic", "cerner"]:  # FHIR-based systems
            mapping = self.data_mappings["fhir_to_internal"]
        else:  # HL7 or other formats
            mapping = self.data_mappings["hl7_to_internal"]

        for data_type, records in ehr_data["data"].items():
            transformed["data"][data_type] = []

            for record in records:
                transformed_record = self._apply_data_mapping(record, mapping.get(data_type.upper(), {}))
                transformed["data"][data_type].append(transformed_record)

        return transformed

    def _apply_data_mapping(self, source_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """Apply data mapping to transform source data"""
        transformed = {}

        for source_path, target_field in mapping.items():
            value = self._extract_nested_value(source_data, source_path)
            if value is not None:
                transformed[target_field] = value

        return transformed

    def _extract_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Extract nested value from data using dot notation"""
        try:
            keys = path.replace('[', '.').replace(']', '').split('.')
            value = data

            for key in keys:
                if '[' in key and ']' in key:
                    # Handle array access like name[0].given[0]
                    base_key, index = key.split('[')
                    index = int(index.rstrip(']'))
                    value = value[base_key][index]
                else:
                    value = value[key]

            return value
        except (KeyError, IndexError, ValueError, TypeError):
            return None

    def sync_patient_data(self, system_id: str, patient_id: str,
                         sync_mode: str = "incremental") -> Dict[str, Any]:
        """Synchronize patient data with EHR system"""
        sync_job = {
            "job_id": f"sync_{system_id}_{patient_id}_{int(time.time())}",
            "system_id": system_id,
            "patient_id": patient_id,
            "sync_mode": sync_mode,
            "started_at": datetime.now(),
            "status": "queued"
        }

        # Add to sync queue
        asyncio.run_coroutine_threadsafe(
            self.sync_queues[system_id].put(sync_job),
            asyncio.get_event_loop()
        )

        return sync_job

    def _data_sync_worker(self):
        """Background worker for data synchronization"""
        while self.is_running:
            try:
                # Check all sync queues
                for system_id, queue in self.sync_queues.items():
                    try:
                        # Non-blocking get with timeout
                        sync_job = queue.get_nowait()
                        self._process_sync_job(sync_job)
                        queue.task_done()
                    except asyncio.QueueEmpty:
                        continue

                time.sleep(1)  # Check every second

            except Exception as e:
                print(f"Data sync worker error: {e}")

    def _process_sync_job(self, sync_job: Dict[str, Any]):
        """Process a data synchronization job"""
        try:
            system_id = sync_job["system_id"]
            patient_id = sync_job["patient_id"]
            sync_mode = sync_job["sync_mode"]

            sync_job["status"] = "processing"

            # Retrieve latest data from EHR
            ehr_data = self.retrieve_patient_data(system_id, patient_id)

            if "error" not in ehr_data:
                # Compare with existing data and update as needed
                self._update_local_data(ehr_data, sync_mode)

                sync_job["status"] = "completed"
                sync_job["completed_at"] = datetime.now()
                sync_job["records_processed"] = len(ehr_data.get("data", {}))
            else:
                sync_job["status"] = "failed"
                sync_job["error"] = ehr_data["error"]

        except Exception as e:
            sync_job["status"] = "failed"
            sync_job["error"] = str(e)

    def _update_local_data(self, ehr_data: Dict[str, Any], sync_mode: str):
        """Update local data with EHR data"""
        # In production, this would update the local database
        # For simulation, we'll just log the update
        print(f"📊 Updated local data for patient {ehr_data['patient_id']} with {len(ehr_data.get('data', {}))} record types")

    def handle_webhook_event(self, webhook_data: Dict[str, Any],
                           webhook_headers: Dict[str, str]) -> Dict[str, Any]:
        """Handle incoming webhook event from EHR system"""
        try:
            # Verify webhook authenticity
            if not self._verify_webhook_signature(webhook_data, webhook_headers):
                return {"status": "rejected", "reason": "invalid_signature"}

            event_type = webhook_data.get("event_type", "unknown")
            system_id = webhook_data.get("system_id", "unknown")
            patient_id = webhook_data.get("patient_id")

            # Log webhook event
            webhook_event = {
                "event_id": f"webhook_{int(time.time())}_{random.randint(1000, 9999)}",
                "system_id": system_id,
                "event_type": event_type,
                "patient_id": patient_id,
                "received_at": datetime.now(),
                "data": webhook_data,
                "processed": False
            }

            # Process webhook based on event type
            if event_type == "patient.updated":
                self._handle_patient_update_webhook(webhook_event)
            elif event_type == "observation.created":
                self._handle_observation_webhook(webhook_event)
            elif event_type == "medication.prescribed":
                self._handle_medication_webhook(webhook_event)
            else:
                webhook_event["processing_result"] = "event_type_not_supported"

            webhook_event["processed"] = True
            webhook_event["processed_at"] = datetime.now()

            return {"status": "processed", "event_id": webhook_event["event_id"]}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _verify_webhook_signature(self, webhook_data: Dict[str, Any],
                                headers: Dict[str, str]) -> bool:
        """Verify webhook signature for authenticity"""
        signature_header = headers.get("X-Webhook-Signature")
        if not signature_header:
            return False

        # Find matching webhook handler
        system_id = webhook_data.get("system_id")
        if not system_id:
            return False

        # Look for webhook handler with matching system
        matching_handlers = [
            handler for handler in self.webhook_handlers.values()
            if handler["system_id"] == system_id
        ]

        if not matching_handlers:
            return False

        # Try to verify with each handler's secret
        payload = json.dumps(webhook_data, sort_keys=True).encode()

        for handler in matching_handlers:
            secret = handler["secret"]
            expected_signature = hmac.new(
                secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()

            if hmac.compare_digest(signature_header, f"sha256={expected_signature}"):
                return True

        return False

    def _handle_patient_update_webhook(self, webhook_event: Dict[str, Any]):
        """Handle patient update webhook"""
        patient_id = webhook_event["patient_id"]
        # Trigger patient data sync
        self.sync_patient_data(webhook_event["system_id"], patient_id, "incremental")

    def _handle_observation_webhook(self, webhook_event: Dict[str, Any]):
        """Handle observation webhook"""
        # Process new observation data
        observation_data = webhook_event["data"].get("observation", {})
        # In production, this would update local observation records
        print(f"📊 New observation received for patient {webhook_event['patient_id']}")

    def _handle_medication_webhook(self, webhook_event: Dict[str, Any]):
        """Handle medication webhook"""
        # Process new medication data
        medication_data = webhook_event["data"].get("medication", {})
        # In production, this would update local medication records
        print(f"💊 New medication prescribed for patient {webhook_event['patient_id']}")

    def _webhook_listener(self):
        """Background webhook listener (simplified)"""
        # In production, this would be a proper HTTP server
        # For simulation, we'll just process any queued webhooks
        while self.is_running:
            try:
                # Check for queued webhooks (simplified)
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"Webhook listener error: {e}")

    def _integration_worker(self):
        """Background worker for EHR integration tasks"""
        while self.is_running:
            try:
                # Process queued integration tasks
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                print(f"Integration worker error: {e}")

    def generate_integration_report(self, system_id: str = None,
                                  start_date: date = None,
                                  end_date: date = None) -> Dict[str, Any]:
        """Generate EHR integration report"""
        if start_date is None:
            start_date = datetime.now().date() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now().date()

        report = {
            "report_type": "ehr_integration",
            "generated_at": datetime.now(),
            "period": {"start": start_date, "end": end_date},
            "systems": {},
            "summary": {}
        }

        # Generate report for specific system or all systems
        systems_to_report = [system_id] if system_id else list(self.connected_systems.keys())

        total_syncs = 0
        total_records = 0
        total_errors = 0

        for sys_id in systems_to_report:
            if sys_id in self.connected_systems:
                system_report = self._generate_system_report(sys_id, start_date, end_date)
                report["systems"][sys_id] = system_report

                total_syncs += system_report.get("sync_operations", 0)
                total_records += system_report.get("records_processed", 0)
                total_errors += system_report.get("errors", 0)

        report["summary"] = {
            "total_systems": len(systems_to_report),
            "total_sync_operations": total_syncs,
            "total_records_processed": total_records,
            "total_errors": total_errors,
            "overall_success_rate": (total_syncs - total_errors) / total_syncs if total_syncs > 0 else 0
        }

        return report

    def _generate_system_report(self, system_id: str, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate report for a specific EHR system"""
        system = self.connected_systems[system_id]

        return {
            "system_name": system["system_name"],
            "system_type": system["system_type"],
            "status": system["status"],
            "sync_operations": random.randint(100, 1000),  # Mock data
            "records_processed": random.randint(5000, 50000),
            "errors": random.randint(0, 10),
            "average_response_time": random.uniform(0.5, 3.0),
            "webhook_events_received": random.randint(50, 500),
            "last_sync": datetime.now() - timedelta(minutes=random.randint(0, 60))
        }

    def export_ehr_data(self, system_id: str, patient_id: str,
                       export_format: str = "fhir") -> Dict[str, Any]:
        """Export patient data in EHR-compatible format"""
        # Retrieve patient data
        patient_data = self.retrieve_patient_data(system_id, patient_id)

        if "error" in patient_data:
            raise ValueError(f"Failed to retrieve patient data: {patient_data['error']}")

        # Convert to requested format
        if export_format.lower() == "fhir":
            exported_data = self._convert_to_fhir(patient_data)
        elif export_format.lower() == "ccd":
            exported_data = self._convert_to_ccd(patient_data)
        elif export_format.lower() == "hl7":
            exported_data = self._convert_to_hl7(patient_data)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        return {
            "patient_id": patient_id,
            "export_format": export_format,
            "exported_at": datetime.now(),
            "data": exported_data
        }

    def _convert_to_fhir(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal data to FHIR format"""
        # Simplified FHIR conversion
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                # Patient resource would go here
                # Observations would go here
                # Medications would go here
            ]
        }

    def _convert_to_ccd(self, patient_data: Dict[str, Any]) -> str:
        """Convert internal data to CCD format"""
        # Simplified CCD conversion (XML)
        root = ET.Element("ClinicalDocument")
        # Add CCD structure here
        return ET.tostring(root, encoding='unicode')

    def _convert_to_hl7(self, patient_data: Dict[str, Any]) -> str:
        """Convert internal data to HL7 format"""
        # Simplified HL7 conversion
        return "MSH|^~\\&|...HL7 message content..."
