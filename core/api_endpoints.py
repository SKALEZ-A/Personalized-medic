"""
Comprehensive API Endpoints for AI Personalized Medicine Platform
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Dict, Any, Optional
from datetime import datetime
import jwt
import json

from core.genomic_engine import GenomicAnalysisEngine
from core.ai_models import AIModels
from core.drug_discovery import DrugDiscoveryEngine
from core.health_monitoring import HealthMonitoringSystem
from core.treatment_engine import TreatmentPlanningEngine
from core.clinical_decision_support import ClinicalDecisionSupport
from core.patient_platform import PatientEngagementPlatform
from core.research_tools import ResearchTools
from core.blockchain_security import BlockchainSecurity

class APIEndpoints:
    """Centralized API endpoint management"""

    def __init__(self):
        self.router = APIRouter()
        self.security = HTTPBearer()
        self.secret_key = "your-secret-key-change-in-production"

        # Initialize all core systems
        self.genomic_engine = GenomicAnalysisEngine()
        self.ai_models = AIModels()
        self.drug_discovery = DrugDiscoveryEngine()
        self.health_monitoring = HealthMonitoringSystem()
        self.treatment_engine = TreatmentPlanningEngine()
        self.clinical_support = ClinicalDecisionSupport()
        self.patient_platform = PatientEngagementPlatform()
        self.research_tools = ResearchTools()
        self.blockchain_security = BlockchainSecurity()

        self._setup_routes()

    def _setup_routes(self):
        """Setup all API routes"""

        @self.router.get("/health")
        async def health_check():
            """Platform health check"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "services": {
                    "genomic_engine": "operational",
                    "ai_models": "operational",
                    "drug_discovery": "operational",
                    "health_monitoring": "operational",
                    "treatment_engine": "operational",
                    "clinical_support": "operational",
                    "patient_platform": "operational",
                    "research_tools": "operational",
                    "blockchain_security": "operational"
                }
            }

        @self.router.post("/auth/login")
        async def login(credentials: Dict[str, Any]):
            """User authentication"""
            # Simplified authentication - in production use proper auth
            user_id = credentials.get("user_id")
            if not user_id:
                raise HTTPException(status_code=400, detail="User ID required")

            token = jwt.encode(
                {"user_id": user_id, "exp": datetime.now().timestamp() + 3600},
                self.secret_key,
                algorithm="HS256"
            )

            return {"access_token": token, "token_type": "bearer"}

        @self.router.get("/patients/{patient_id}")
        async def get_patient(patient_id: str, credentials=Depends(self.security)):
            """Get patient information"""
            self._verify_token(credentials)
            # In real implementation, fetch from database
            return {"patient_id": patient_id, "status": "active"}

        @self.router.post("/genomics/analyze")
        async def analyze_genome(request: Dict[str, Any], background_tasks: BackgroundTasks,
                               credentials=Depends(self.security)):
            """Submit genome for analysis"""
            self._verify_token(credentials)

            result = self.genomic_engine.analyze_genome(request)

            # Add background processing for comprehensive analysis
            if request.get("analysis_type") == "comprehensive":
                background_tasks.add_task(
                    self.genomic_engine.process_genome_async,
                    request.get("patient_id"),
                    request.get("genome_sequence"),
                    "comprehensive"
                )

            return result

        @self.router.get("/genomics/results/{job_id}")
        async def get_genomic_results(job_id: str, credentials=Depends(self.security)):
            """Get genomic analysis results"""
            self._verify_token(credentials)

            results = self.genomic_engine.get_analysis_results(job_id)
            if not results:
                raise HTTPException(status_code=404, detail="Analysis not complete")

            return results

        @self.router.post("/ai/predict/disease-risk")
        async def predict_disease_risk(request: Dict[str, Any], credentials=Depends(self.security)):
            """AI-powered disease risk prediction"""
            self._verify_token(credentials)

            try:
                result = await self.ai_models.predict_disease_risk(request)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/ai/predict/drug-response")
        async def predict_drug_response(request: Dict[str, Any], credentials=Depends(self.security)):
            """AI-powered drug response prediction"""
            self._verify_token(credentials)

            try:
                result = await self.ai_models.predict_drug_response(
                    request["patient_profile"],
                    request["drug_info"]
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/drug-discovery/discover")
        async def discover_drugs(request: Dict[str, Any], credentials=Depends(self.security)):
            """AI-powered drug discovery"""
            self._verify_token(credentials)

            result = self.drug_discovery.discover_compounds(
                request["target_protein"],
                request["disease_context"],
                request["patient_profile"]
            )

            return result

        @self.router.post("/monitoring/health-data")
        async def submit_health_data(data: Dict[str, Any], background_tasks: BackgroundTasks,
                                   credentials=Depends(self.security)):
            """Submit health monitoring data"""
            self._verify_token(credentials)

            result = self.health_monitoring.process_health_data(data)

            # Background processing for comprehensive analysis
            background_tasks.add_task(
                self._process_health_insights,
                data
            )

            return result

        @self.router.post("/treatment/plan")
        async def create_treatment_plan(request: Dict[str, Any], credentials=Depends(self.security)):
            """Create personalized treatment plan"""
            self._verify_token(credentials)

            plan = self.treatment_engine.create_treatment_plan(
                diagnosis=request["diagnosis"],
                patient_data=request["patient_data"],
                current_medications=request.get("current_medications", []),
                contraindications=request.get("contraindications", [])
            )

            return plan

        @self.router.post("/clinical-support/query")
        async def clinical_decision_support(query: Dict[str, Any], credentials=Depends(self.security)):
            """Clinical decision support"""
            self._verify_token(credentials)

            result = self.clinical_support.generate_recommendations(
                query=query["query"],
                patient_data=query["patient_data"],
                context=query.get("context")
            )

            return result

        @self.router.get("/patient/dashboard/{patient_id}")
        async def get_patient_dashboard(patient_id: str, credentials=Depends(self.security)):
            """Get patient health dashboard"""
            self._verify_token(credentials)

            # Get patient data (simplified)
            patient_data = {"patient_id": patient_id}  # Would fetch from database

            # Get health records (simplified)
            health_records = []  # Would fetch from database

            dashboard = self.patient_platform.generate_dashboard(
                patient_id=patient_id,
                patient_data=patient_data,
                health_records=health_records
            )

            return dashboard

        @self.router.post("/research/trial")
        async def create_clinical_trial(trial_data: Dict[str, Any], credentials=Depends(self.security)):
            """Create clinical trial"""
            self._verify_token(credentials)

            # Validate trial design
            validation = self.research_tools.validate_trial_design(trial_data)
            if not validation["valid"]:
                raise HTTPException(status_code=400, detail=validation["errors"])

            # Create trial
            result = self.research_tools.create_clinical_trial(trial_data)
            return result

        @self.router.get("/research/trials/{patient_id}/matches")
        async def get_trial_matches(patient_id: str, credentials=Depends(self.security)):
            """Get suitable clinical trials for patient"""
            self._verify_token(credentials)

            # Get patient data (simplified)
            patient_data = {"patient_id": patient_id}  # Would fetch from database

            matches = self.research_tools.match_patients_to_trials(patient_data)
            return {"matches": matches}

        @self.router.post("/blockchain/record")
        async def create_blockchain_record(record: Dict[str, Any], credentials=Depends(self.security)):
            """Create blockchain-secured health record"""
            self._verify_token(credentials)

            tx_id = self.blockchain_security.create_health_record_transaction(
                patient_id=record["patient_id"],
                provider_id=record["provider_id"],
                record_data=record["data"]
            )

            # Mine block if enough transactions
            if len(self.blockchain_security.pending_transactions) >= 5:
                self.blockchain_security.mine_block("healthcare_validator")

            return {"transaction_id": tx_id, "status": "recorded"}

        @self.router.get("/blockchain/verify/{record_id}")
        async def verify_blockchain_record(record_id: str, credentials=Depends(self.security)):
            """Verify record integrity using blockchain"""
            self._verify_token(credentials)

            verification = self.blockchain_security.verify_record_integrity(record_id)
            return verification

        @self.router.post("/consent/create")
        async def create_consent(consent_data: Dict[str, Any], credentials=Depends(self.security)):
            """Create patient consent record"""
            self._verify_token(credentials)

            consent_id = self.blockchain_security.consent_management.create_consent_record(
                patient_id=consent_data["patient_id"],
                consent_type=consent_data["consent_type"],
                parameters=consent_data.get("parameters", {})
            )

            # Record on blockchain
            self.blockchain_security.create_consent_transaction(
                patient_id=consent_data["patient_id"],
                consent_data={"consent_id": consent_id, **consent_data}
            )

            return {"consent_id": consent_id, "status": "created"}

        @self.router.get("/consent/check/{patient_id}")
        async def check_consent(patient_id: str, data_type: str, credentials=Depends(self.security)):
            """Check patient consent for data access"""
            self._verify_token(credentials)

            result = self.blockchain_security.consent_management.check_consent(
                patient_id=patient_id,
                data_type=data_type,
                requester="api_caller"
            )

            return result

        @self.router.get("/analytics/overview")
        async def get_platform_analytics(credentials=Depends(self.security)):
            """Get platform analytics overview"""
            self._verify_token(credentials)

            analytics = {
                "genomic_analyses": self.genomic_engine.get_analysis_statistics(),
                "drug_discoveries": self.drug_discovery.get_discovery_statistics(),
                "health_monitoring": self.health_monitoring.get_monitoring_statistics(),
                "blockchain": self.blockchain_security.get_chain_info(),
                "ai_models": await self.ai_models.get_model_performance_metrics()
            }

            return analytics

        @self.router.post("/iot/connect")
        async def connect_iot_device(device_data: Dict[str, Any], credentials=Depends(self.security)):
            """Connect IoT health device"""
            self._verify_token(credentials)

            result = self.health_monitoring.iot_integrations.connect_device(
                device_type=device_data["device_type"],
                patient_id=device_data["patient_id"],
                credentials=device_data.get("credentials", {})
            )

            return result

        @self.router.get("/iot/data/{connection_id}")
        async def get_iot_data(connection_id: str, credentials=Depends(self.security)):
            """Get data from connected IoT device"""
            self._verify_token(credentials)

            result = self.health_monitoring.iot_integrations.get_device_data(
                connection_id=connection_id
            )

            return result

        @self.router.post("/virtual-assistant/query")
        async def virtual_assistant_query(query: Dict[str, Any], credentials=Depends(self.security)):
            """Interact with virtual health assistant"""
            self._verify_token(credentials)

            response = self.patient_platform.virtual_assistant.process_query(
                patient_id=query["patient_id"],
                query=query["message"]
            )

            return response

        @self.router.get("/mobile/report/{patient_id}")
        async def get_mobile_health_report(patient_id: str, credentials=Depends(self.security)):
            """Get mobile health companion report"""
            self._verify_token(credentials)

            report = self.patient_platform.mobile_companion.generate_daily_report(patient_id)
            return report

    def _verify_token(self, credentials: HTTPAuthorizationCredentials):
        """Verify JWT token"""
        try:
            payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=["HS256"])
            # In production, validate user permissions, etc.
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def _process_health_insights(self, health_data: Dict[str, Any]):
        """Process health insights in background"""
        # Generate AI-powered insights
        await self.ai_models.predict_disease_risk(health_data)

        # Update patient dashboard
        patient_id = health_data.get("patient_id")
        if patient_id:
            # Trigger dashboard update (simplified)
            pass

# Global API endpoints instance
api_endpoints = APIEndpoints()
