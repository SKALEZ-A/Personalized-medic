"""
Integration Modules for Medical Data Exchange Standards
HL7, FHIR, DICOM, and other healthcare interoperability standards
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
import hashlib

class FHIRIntegration:
    """Fast Healthcare Interoperability Resources (FHIR) integration"""

    def __init__(self):
        self.base_url = "http://localhost:8080/fhir"
        self.resources = {
            "Patient": self._create_patient_resource,
            "Observation": self._create_observation_resource,
            "Condition": self._create_condition_resource,
            "MedicationRequest": self._create_medication_request_resource,
            "DiagnosticReport": self._create_diagnostic_report_resource
        }

    def create_fhir_bundle(self, patient_data: Dict[str, Any], resources: List[str] = None) -> Dict[str, Any]:
        """Create a FHIR bundle with patient data"""
        if resources is None:
            resources = ["Patient", "Observation", "Condition"]

        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": []
        }

        for resource_type in resources:
            if resource_type in self.resources:
                resource = self.resources[resource_type](patient_data)
                if resource:
                    bundle["entry"].append({
                        "resource": resource
                    })

        return bundle

    def _create_patient_resource(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create FHIR Patient resource"""
        demographics = patient_data.get("demographics", {})

        patient = {
            "resourceType": "Patient",
            "id": patient_data.get("patient_id", str(uuid.uuid4())),
            "identifier": [{
                "system": "http://hospital.smarthealthit.org",
                "value": patient_data.get("patient_id")
            }],
            "name": [{
                "family": demographics.get("last_name", "Unknown"),
                "given": [demographics.get("first_name", "Unknown")]
            }],
            "gender": demographics.get("gender", "unknown").lower(),
            "birthDate": demographics.get("birth_date", "1900-01-01"),
            "address": [{
                "line": [demographics.get("address", "Unknown")],
                "city": demographics.get("city", "Unknown"),
                "state": demographics.get("state", "Unknown"),
                "postalCode": demographics.get("zip_code", "00000")
            }]
        }

        return patient

    def _create_observation_resource(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create FHIR Observation resources"""
        observations = []
        patient_id = patient_data.get("patient_id")

        # Vital signs
        vital_signs = patient_data.get("vital_signs", {})
        if vital_signs:
            for vital_type, value in vital_signs.items():
                if isinstance(value, dict) and "value" in value:
                    observation = {
                        "resourceType": "Observation",
                        "id": str(uuid.uuid4()),
                        "status": "final",
                        "subject": {
                            "reference": f"Patient/{patient_id}"
                        },
                        "code": {
                            "coding": [{
                                "system": "http://loinc.org",
                                "code": self._get_loinc_code(vital_type),
                                "display": vital_type.replace("_", " ").title()
                            }]
                        },
                        "valueQuantity": {
                            "value": value["value"],
                            "unit": value.get("unit", "unknown"),
                            "system": "http://unitsofmeasure.org"
                        },
                        "effectiveDateTime": datetime.now().isoformat()
                    }
                    observations.append(observation)

        # Biomarkers
        biomarkers = patient_data.get("biomarkers", [])
        for biomarker in biomarkers:
            observation = {
                "resourceType": "Observation",
                "id": str(uuid.uuid4()),
                "status": "final",
                "subject": {
                    "reference": f"Patient/{patient_id}"
                },
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": biomarker.get("loinc_code", "unknown"),
                        "display": biomarker.get("name", "unknown")
                    }]
                },
                "valueQuantity": {
                    "value": biomarker.get("value"),
                    "unit": biomarker.get("unit", "unknown"),
                    "system": "http://unitsofmeasure.org"
                },
                "effectiveDateTime": biomarker.get("timestamp", datetime.now().isoformat())
            }
            observations.append(observation)

        return observations[0] if observations else None

    def _get_loinc_code(self, vital_type: str) -> str:
        """Get LOINC code for vital sign"""
        loinc_codes = {
            "heart_rate": "8867-4",
            "blood_pressure_systolic": "8480-6",
            "blood_pressure_diastolic": "8462-4",
            "temperature": "8310-5",
            "oxygen_saturation": "59408-5",
            "respiratory_rate": "9279-1"
        }
        return loinc_codes.get(vital_type, "unknown")

    def _create_condition_resource(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create FHIR Condition resource"""
        medical_history = patient_data.get("medical_history", [])
        patient_id = patient_data.get("patient_id")

        if not medical_history:
            return None

        condition = {
            "resourceType": "Condition",
            "id": str(uuid.uuid4()),
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "73211009",  # Diabetes mellitus (example)
                    "display": medical_history[0] if medical_history else "Unknown"
                }]
            },
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active"
                }]
            },
            "recordedDate": datetime.now().isoformat()
        }

        return condition

    def _create_medication_request_resource(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create FHIR MedicationRequest resource"""
        medications = patient_data.get("current_medications", [])
        patient_id = patient_data.get("patient_id")

        if not medications:
            return None

        medication_request = {
            "resourceType": "MedicationRequest",
            "id": str(uuid.uuid4()),
            "status": "active",
            "intent": "order",
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "medicationCodeableConcept": {
                "coding": [{
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": "123456",  # Example RxNorm code
                    "display": medications[0]
                }]
            },
            "authoredOn": datetime.now().isoformat()
        }

        return medication_request

    def _create_diagnostic_report_resource(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create FHIR DiagnosticReport resource"""
        genomic_data = patient_data.get("genomic_data", {})
        patient_id = patient_data.get("patient_id")

        if not genomic_data:
            return None

        diagnostic_report = {
            "resourceType": "DiagnosticReport",
            "id": str(uuid.uuid4()),
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "81247-9",
                    "display": "Genetic analysis summary panel"
                }]
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "effectiveDateTime": datetime.now().isoformat(),
            "result": [
                {
                    "reference": "Observation/genetic-variant-1"
                }
            ],
            "conclusion": "Genomic analysis completed. See detailed results for interpretation."
        }

        return diagnostic_report

class HL7Integration:
    """Health Level 7 (HL7) integration for legacy systems"""

    def __init__(self):
        self.version = "2.5.1"
        self.message_types = {
            "ADT": self._create_adt_message,
            "ORU": self._create_oru_message,
            "ORM": self._create_orm_message
        }

    def create_hl7_message(self, message_type: str, patient_data: Dict[str, Any]) -> str:
        """Create HL7 message"""
        if message_type not in self.message_types:
            raise ValueError(f"Unsupported message type: {message_type}")

        return self.message_types[message_type](patient_data)

    def _create_adt_message(self, patient_data: Dict[str, Any]) -> str:
        """Create ADT (Admit/Discharge/Transfer) message"""
        demographics = patient_data.get("demographics", {})

        # MSH segment
        msh = f"MSH|^~\\&|AI_MED_PLATFORM|SENDER|RECEIVER|DEST|{datetime.now().strftime('%Y%m%d%H%M%S')}||ADT^A01|MSG{int(datetime.now().timestamp())}|P|2.5.1"

        # PID segment
        pid = f"PID|1||{patient_data.get('patient_id', 'UNKNOWN')}||{demographics.get('last_name', 'UNKNOWN')}^{demographics.get('first_name', 'UNKNOWN')}||{demographics.get('birth_date', '19000101')}|{demographics.get('gender', 'U')}"

        # PV1 segment (Patient Visit)
        pv1 = f"PV1|1|I|WARD^ROOM^B^^^^^^^^^^^|ROUTINE|||DOC^DOCTOR^PHYSICIAN^^^^^^^^^^^|REF^REFERRING^DOCTOR^^^^^^^^^^^|||||||ADM||{patient_data.get('patient_id', 'UNKNOWN')}||||||||||||||||||||||||{datetime.now().strftime('%Y%m%d%H%M%S')}"

        message = f"{msh}\r{pid}\r{pv1}\r"

        return message

    def _create_oru_message(self, patient_data: Dict[str, Any]) -> str:
        """Create ORU (Observation Result Unsolicited) message"""
        # MSH segment
        msh = f"MSH|^~\\&|AI_MED_PLATFORM|LAB|RECEIVER|DEST|{datetime.now().strftime('%Y%m%d%H%M%S')}||ORU^R01|MSG{int(datetime.now().timestamp())}|P|2.5.1"

        # PID segment
        demographics = patient_data.get("demographics", {})
        pid = f"PID|1||{patient_data.get('patient_id', 'UNKNOWN')}||{demographics.get('last_name', 'UNKNOWN')}^{demographics.get('first_name', 'UNKNOWN')}"

        # OBR segment (Observation Request)
        obr = f"OBR|1||ORDER123|GLUCOSE^Glucose^L|||202401011200|||{datetime.now().strftime('%Y%m%d%H%M%S')}||||||||||F"

        # OBX segments (Observation Result)
        obx_segments = []
        biomarkers = patient_data.get("biomarkers", [])

        for i, biomarker in enumerate(biomarkers[:5]):  # Limit to 5 observations
            value = biomarker.get("value", "")
            unit = biomarker.get("unit", "")
            name = biomarker.get("name", "").upper()
            obx = f"OBX|{i+1}|NM|{name}^{name}^L||{value}|{unit}|||||F|||202401011200"
            obx_segments.append(obx)

        message = f"{msh}\r{pid}\r{obr}\r" + "\r".join(obx_segments) + "\r"

        return message

    def _create_orm_message(self, patient_data: Dict[str, Any]) -> str:
        """Create ORM (Order Message) message"""
        # MSH segment
        msh = f"MSH|^~\\&|AI_MED_PLATFORM|CLINIC|LAB|DEST|{datetime.now().strftime('%Y%m%d%H%M%S')}||ORM^O01|MSG{int(datetime.now().timestamp())}|P|2.5.1"

        # PID segment
        demographics = patient_data.get("demographics", {})
        pid = f"PID|1||{patient_data.get('patient_id', 'UNKNOWN')}||{demographics.get('last_name', 'UNKNOWN')}^{demographics.get('first_name', 'UNKNOWN')}"

        # ORC segment (Order Control)
        orc = "ORC|NW|ORDER123"

        # OBR segment
        obr = f"OBR|1|ORDER123||GENETIC^Genetic Analysis^L|||202401011200||||||||||1"

        message = f"{msh}\r{pid}\r{orc}\r{obr}\r"

        return message

    def parse_hl7_message(self, hl7_message: str) -> Dict[str, Any]:
        """Parse HL7 message into structured data"""
        segments = hl7_message.strip().split('\r')

        parsed_data = {}

        for segment in segments:
            if not segment:
                continue

            fields = segment.split('|')
            segment_type = fields[0]

            if segment_type == 'MSH':
                parsed_data['message_header'] = {
                    'sending_application': fields[2],
                    'sending_facility': fields[3],
                    'receiving_application': fields[4],
                    'receiving_facility': fields[5],
                    'message_type': fields[8],
                    'message_control_id': fields[9],
                    'version': fields[11]
                }
            elif segment_type == 'PID':
                parsed_data['patient'] = {
                    'id': fields[3],
                    'name': fields[5],
                    'birth_date': fields[7],
                    'gender': fields[8]
                }
            elif segment_type == 'OBX':
                if 'observations' not in parsed_data:
                    parsed_data['observations'] = []

                observation = {
                    'set_id': fields[1],
                    'value_type': fields[2],
                    'observation_id': fields[3],
                    'value': fields[5],
                    'units': fields[6],
                    'reference_range': fields[7]
                }
                parsed_data['observations'].append(observation)

        return parsed_data

class DICOMIntegration:
    """Digital Imaging and Communications in Medicine (DICOM) integration"""

    def __init__(self):
        self.supported_modalities = [
            "CT", "MR", "US", "CR", "DX", "MG", "XA", "RF"
        ]

    def process_dicom_image(self, dicom_data: bytes) -> Dict[str, Any]:
        """Process DICOM image data (simplified)"""
        # In real implementation, would use pydicom library
        processed_data = {
            "modality": "CT",  # Would be extracted from DICOM header
            "study_instance_uid": str(uuid.uuid4()),
            "series_instance_uid": str(uuid.uuid4()),
            "sop_instance_uid": str(uuid.uuid4()),
            "patient_id": "PATIENT_001",
            "study_date": datetime.now().strftime("%Y%m%d"),
            "image_dimensions": [512, 512],
            "pixel_spacing": [0.5, 0.5],
            "slice_thickness": 5.0,
            "window_center": 40,
            "window_width": 400,
            "image_data_hash": hashlib.sha256(dicom_data).hexdigest()
        }

        return processed_data

    def create_dicom_sr(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Create DICOM Structured Report"""
        sr_document = {
            "sop_class_uid": "1.2.840.10008.5.1.4.1.1.88.11",  # Basic Text SR
            "instance_uid": str(uuid.uuid4()),
            "patient_id": findings.get("patient_id"),
            "study_instance_uid": findings.get("study_uid"),
            "content": {
                "findings": findings.get("findings", []),
                "impressions": findings.get("impressions", []),
                "recommendations": findings.get("recommendations", [])
            },
            "created_date": datetime.now().isoformat()
        }

        return sr_document

class MedicalDeviceIntegration:
    """Integration with medical devices and wearables"""

    def __init__(self):
        self.device_protocols = {
            "bluetooth_le": self._handle_bluetooth_le,
            "wifi_direct": self._handle_wifi_direct,
            " Continua Health Alliance": self._handle_continua
        }

    def connect_device(self, device_type: str, connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to medical device"""
        protocol = connection_params.get("protocol", "bluetooth_le")

        if protocol not in self.device_protocols:
            raise ValueError(f"Unsupported protocol: {protocol}")

        return self.device_protocols[protocol](device_type, connection_params)

    def _handle_bluetooth_le(self, device_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Bluetooth LE device connection"""
        return {
            "connection_id": str(uuid.uuid4()),
            "device_type": device_type,
            "protocol": "bluetooth_le",
            "status": "connected",
            "mac_address": params.get("mac_address"),
            "services": ["heart_rate", "battery_level"],
            "connection_strength": "good"
        }

    def _handle_wifi_direct(self, device_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WiFi Direct device connection"""
        return {
            "connection_id": str(uuid.uuid4()),
            "device_type": device_type,
            "protocol": "wifi_direct",
            "status": "connected",
            "ip_address": params.get("ip_address"),
            "services": ["data_sync", "firmware_update"],
            "bandwidth": "high"
        }

    def _handle_continua(self, device_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Continua Health Alliance device connection"""
        return {
            "connection_id": str(uuid.uuid4()),
            "device_type": device_type,
            "protocol": "continua",
            "status": "connected",
            "device_specialization": params.get(" Continua _type"),
            "certified": True,
            "supported_measures": ["weight", "blood_pressure", "glucose"]
        }

    def sync_device_data(self, connection_id: str, sync_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sync data from connected device"""
        # Simulate device data sync
        device_data = {
            "connection_id": connection_id,
            "sync_timestamp": datetime.now().isoformat(),
            "data_points": random.randint(10, 100),
            "data_types": ["heart_rate", "steps", "calories"],
            "sync_duration_seconds": random.uniform(5, 30),
            "data_quality_score": random.uniform(0.8, 0.98)
        }

        return device_data

class TelemedicineIntegration:
    """Telemedicine and remote consultation integration"""

    def __init__(self):
        self.video_providers = ["zoom", "webex", "teams", "custom"]
        self.active_sessions = {}

    def create_telemedicine_session(self, session_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create telemedicine consultation session"""
        session_id = f"telemed_{int(datetime.now().timestamp())}"

        session = {
            "session_id": session_id,
            "patient_id": session_params.get("patient_id"),
            "provider_id": session_params.get("provider_id"),
            "consultation_type": session_params.get("consultation_type", "general"),
            "scheduled_time": session_params.get("scheduled_time", datetime.now().isoformat()),
            "duration_minutes": session_params.get("duration_minutes", 30),
            "video_provider": session_params.get("video_provider", "zoom"),
            "status": "scheduled",
            "meeting_link": self._generate_meeting_link(session_params.get("video_provider", "zoom")),
            "security_enabled": True,
            "recording_enabled": session_params.get("recording_enabled", False)
        }

        self.active_sessions[session_id] = session

        return session

    def _generate_meeting_link(self, provider: str) -> str:
        """Generate meeting link for video consultation"""
        base_urls = {
            "zoom": "https://zoom.us/j/",
            "webex": "https://webex.com/meet/",
            "teams": "https://teams.microsoft.com/l/meetup-join/",
            "custom": "https://telemedicine.platform.com/session/"
        }

        meeting_id = str(uuid.uuid4())[:8].upper()
        return f"{base_urls.get(provider, base_urls['custom'])}{meeting_id}"

    def join_session(self, session_id: str, participant_type: str) -> Dict[str, Any]:
        """Join telemedicine session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]

        # Update session status
        if session["status"] == "scheduled":
            session["status"] = "active"
            session["started_at"] = datetime.now().isoformat()

        participant_info = {
            "session_id": session_id,
            "participant_type": participant_type,
            "joined_at": datetime.now().isoformat(),
            "connection_quality": "good",
            "permissions": self._get_participant_permissions(participant_type)
        }

        return participant_info

    def _get_participant_permissions(self, participant_type: str) -> List[str]:
        """Get participant permissions based on type"""
        permissions = {
            "patient": ["view_consultation", "share_screen", "send_messages"],
            "provider": ["view_consultation", "share_screen", "send_messages", "control_recording", "end_session"],
            "consultant": ["view_consultation", "send_messages"],
            "observer": ["view_consultation"]
        }

        return permissions.get(participant_type, ["view_consultation"])

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End telemedicine session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]

        session["status"] = "completed"
        session["ended_at"] = datetime.now().isoformat()

        if "started_at" in session:
            duration = (datetime.fromisoformat(session["ended_at"]) -
                       datetime.fromisoformat(session["started_at"])).total_seconds() / 60
            session["actual_duration_minutes"] = round(duration, 1)

        return {
            "session_id": session_id,
            "status": "completed",
            "summary": {
                "duration_minutes": session.get("actual_duration_minutes"),
                "participants": [session["patient_id"], session["provider_id"]],
                "recording_available": session.get("recording_enabled", False)
            }
        }

class PharmacyIntegration:
    """Integration with pharmacy systems and medication databases"""

    def __init__(self):
        self.drug_databases = {
            "rxnorm": self._initialize_rxnorm,
            "fdb": self._initialize_fdb,
            "micromedex": self._initialize_micromedex
        }
        self.active_prescriptions = {}

    def _initialize_rxnorm(self) -> Dict[str, Any]:
        return {"name": "RxNorm", "version": "2023AA", "drug_count": 50000}

    def _initialize_fdb(self) -> Dict[str, Any]:
        return {"name": "First Databank", "version": "2023.1", "drug_count": 75000}

    def _initialize_micromedex(self) -> Dict[str, Any]:
        return {"name": "Micromedex", "version": "2.0", "drug_count": 30000}

    def search_medication(self, search_term: str, database: str = "rxnorm") -> List[Dict[str, Any]]:
        """Search for medication in specified database"""
        if database not in self.drug_databases:
            raise ValueError(f"Unsupported database: {database}")

        # Simulate medication search
        medications = [
            {
                "rxcui": "123456",
                "name": f"{search_term.title()} 10mg",
                "brand_names": [f"{search_term.title()}Tab"],
                "strength": "10mg",
                "form": "tablet",
                "route": "oral",
                "generic": True
            },
            {
                "rxcui": "789012",
                "name": f"{search_term.title()} 20mg",
                "brand_names": [f"Brand{search_term.title()}Tab"],
                "strength": "20mg",
                "form": "tablet",
                "route": "oral",
                "generic": False
            }
        ]

        return medications

    def check_drug_interactions(self, drug_list: List[str]) -> List[Dict[str, Any]]:
        """Check for drug interactions"""
        interactions = []

        # Define some common interactions
        interaction_rules = {
            ("warfarin", "aspirin"): {
                "severity": "major",
                "description": "Increased bleeding risk",
                "recommendation": "Monitor INR closely"
            },
            ("lisinopril", "potassium"): {
                "severity": "moderate",
                "description": "Hyperkalemia risk",
                "recommendation": "Monitor potassium levels"
            }
        }

        drug_set = set(d.lower() for d in drug_list)

        for (drug1, drug2), interaction in interaction_rules.items():
            if drug1 in drug_set and drug2 in drug_set:
                interactions.append({
                    "drugs": [drug1, drug2],
                    **interaction
                })

        return interactions

    def create_prescription(self, prescription_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create electronic prescription"""
        prescription_id = f"rx_{int(datetime.now().timestamp())}"

        prescription = {
            "prescription_id": prescription_id,
            "patient_id": prescription_data["patient_id"],
            "provider_id": prescription_data["provider_id"],
            "medication": prescription_data["medication"],
            "dosage": prescription_data["dosage"],
            "quantity": prescription_data["quantity"],
            "directions": prescription_data["directions"],
            "refills": prescription_data.get("refills", 0),
            "prescribed_date": datetime.now().isoformat(),
            "expiration_date": (datetime.now() + timedelta(days=365)).isoformat(),
            "status": "active",
            "pharmacy_id": prescription_data.get("pharmacy_id")
        }

        self.active_prescriptions[prescription_id] = prescription

        return prescription

    def transmit_prescription(self, prescription_id: str, pharmacy_system: str) -> Dict[str, Any]:
        """Transmit prescription to pharmacy system"""
        if prescription_id not in self.active_prescriptions:
            return {"error": "Prescription not found"}

        prescription = self.active_prescriptions[prescription_id]

        # Simulate transmission
        transmission_result = {
            "prescription_id": prescription_id,
            "pharmacy_system": pharmacy_system,
            "transmission_status": "successful",
            "confirmation_number": f"CONF_{int(datetime.now().timestamp())}",
            "estimated_ready_time": "2 hours",
            "transmission_timestamp": datetime.now().isoformat()
        }

        return transmission_result

class LaboratoryIntegration:
    """Integration with laboratory information systems"""

    def __init__(self):
        self.lis_protocols = ["HL7", "ASTM", "POCT1-A"]
        self.test_catalog = self._initialize_test_catalog()

    def _initialize_test_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Initialize laboratory test catalog"""
        return {
            "cbc": {
                "loinc_code": "58410-2",
                "name": "Complete Blood Count",
                "components": ["WBC", "RBC", "HGB", "HCT", "PLT"],
                "turnaround_time_hours": 4,
                "reference_ranges": {
                    "WBC": {"min": 4.0, "max": 11.0, "unit": "K/uL"},
                    "RBC": {"min": 4.2, "max": 5.4, "unit": "M/uL"},
                    "HGB": {"min": 12.0, "max": 16.0, "unit": "g/dL"}
                }
            },
            "cmp": {
                "loinc_code": "24323-8",
                "name": "Comprehensive Metabolic Panel",
                "components": ["GLU", "BUN", "CRE", "ALT", "AST"],
                "turnaround_time_hours": 6,
                "reference_ranges": {
                    "GLU": {"min": 70, "max": 140, "unit": "mg/dL"},
                    "BUN": {"min": 7, "max": 20, "unit": "mg/dL"},
                    "CRE": {"min": 0.6, "max": 1.2, "unit": "mg/dL"}
                }
            }
        }

    def order_laboratory_test(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Order laboratory test"""
        test_code = order_data["test_code"]
        patient_id = order_data["patient_id"]

        if test_code not in self.test_catalog:
            return {"error": f"Unknown test code: {test_code}"}

        test_info = self.test_catalog[test_code]

        order = {
            "order_id": f"lab_{int(datetime.now().timestamp())}",
            "patient_id": patient_id,
            "test_code": test_code,
            "test_name": test_info["name"],
            "ordered_by": order_data["provider_id"],
            "order_date": datetime.now().isoformat(),
            "priority": order_data.get("priority", "routine"),
            "status": "ordered",
            "estimated_completion": (datetime.now() + timedelta(hours=test_info["turnaround_time_hours"])).isoformat()
        }

        return order

    def receive_lab_results(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Receive and process laboratory results"""
        test_code = result_data["test_code"]
        patient_id = result_data["patient_id"]

        test_info = self.test_catalog.get(test_code, {})

        processed_results = {
            "result_id": f"result_{int(datetime.now().timestamp())}",
            "patient_id": patient_id,
            "test_code": test_code,
            "test_name": test_info.get("name", "Unknown Test"),
            "results": {},
            "abnormal_flags": [],
            "interpretation": "",
            "received_date": datetime.now().isoformat()
        }

        # Process individual test components
        for component, value in result_data.get("results", {}).items():
            reference_range = test_info.get("reference_ranges", {}).get(component, {})

            if reference_range:
                min_val = reference_range.get("min", 0)
                max_val = reference_range.get("max", 100)

                status = "normal"
                if value < min_val:
                    status = "low"
                    processed_results["abnormal_flags"].append(f"{component} below normal")
                elif value > max_val:
                    status = "high"
                    processed_results["abnormal_flags"].append(f"{component} above normal")

                processed_results["results"][component] = {
                    "value": value,
                    "unit": reference_range.get("unit", ""),
                    "reference_range": f"{min_val}-{max_val}",
                    "status": status
                }
            else:
                processed_results["results"][component] = {
                    "value": value,
                    "status": "unknown"
                }

        # Generate interpretation
        if processed_results["abnormal_flags"]:
            processed_results["interpretation"] = f"Abnormal findings: {', '.join(processed_results['abnormal_flags'])}. Clinical correlation recommended."
        else:
            processed_results["interpretation"] = "All results within normal limits."

        return processed_results
