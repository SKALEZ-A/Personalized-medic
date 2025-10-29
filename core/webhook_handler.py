"""
Webhook Handler System for AI Personalized Medicine Platform
Manages incoming webhooks from external services and EHR systems
"""

import json
import hashlib
import hmac
import secrets
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
import logging
import re
from urllib.parse import urlparse, parse_qs
import aiohttp
import requests

class WebhookHandlerSystem:
    """Comprehensive webhook handling system"""

    def __init__(self):
        self.webhook_endpoints = {}
        self.webhook_handlers = {}
        self.webhook_logs = defaultdict(list)
        self.retry_queues = defaultdict(asyncio.Queue)
        self.failed_webhooks = defaultdict(list)
        self.webhook_metrics = defaultdict(lambda: {
            "total_received": 0,
            "successful_processed": 0,
            "failed_processing": 0,
            "retries_attempted": 0,
            "average_processing_time": 0,
            "processing_times": deque(maxlen=1000)
        })
        self.is_running = False
        self.webhook_workers = []
        self.initialize_webhook_system()

    def initialize_webhook_system(self):
        """Initialize webhook handling system"""
        # Register default webhook handlers
        self._register_default_handlers()

        print("🪝 Webhook Handler System initialized")

    def _register_default_handlers(self):
        """Register default webhook handlers"""
        default_handlers = {
            "ehr.patient_update": self._handle_ehr_patient_update,
            "ehr.encounter_created": self._handle_ehr_encounter_created,
            "ehr.observation_added": self._handle_ehr_observation_added,
            "ehr.medication_prescribed": self._handle_ehr_medication_prescribed,
            "lab_results_ready": self._handle_lab_results_ready,
            "imaging_study_completed": self._handle_imaging_study_completed,
            "telemedicine_session_ended": self._handle_telemedicine_session_ended,
            "device_data_received": self._handle_device_data_received,
            "clinical_trial_update": self._handle_clinical_trial_update,
            "pharmacy_dispensation": self._handle_pharmacy_dispensation
        }

        for event_type, handler in default_handlers.items():
            self.register_webhook_handler(event_type, handler)

    def start_webhook_system(self):
        """Start webhook processing system"""
        self.is_running = True

        # Start webhook processing workers
        for i in range(5):  # 5 worker threads
            worker = threading.Thread(target=self._webhook_worker, daemon=True)
            worker.start()
            self.webhook_workers.append(worker)

        # Start retry worker
        retry_worker = threading.Thread(target=self._retry_worker, daemon=True)
        retry_worker.start()
        self.webhook_workers.append(retry_worker)

        # Start cleanup worker
        cleanup_worker = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_worker.start()
        self.webhook_workers.append(cleanup_worker)

        print("⚡ Webhook Handler System started")

    def stop_webhook_system(self):
        """Stop webhook processing system"""
        self.is_running = False
        print("🛑 Webhook Handler System stopped")

    def register_webhook_endpoint(self, endpoint_path: str, supported_events: List[str],
                                auth_required: bool = True, rate_limit: int = 100) -> str:
        """Register a webhook endpoint"""
        endpoint_id = f"endpoint_{hashlib.md5(endpoint_path.encode()).hexdigest()[:8]}"

        endpoint_config = {
            "endpoint_id": endpoint_id,
            "endpoint_path": endpoint_path,
            "supported_events": supported_events,
            "auth_required": auth_required,
            "rate_limit": rate_limit,
            "webhook_secret": secrets.token_hex(32),
            "registered_at": datetime.now(),
            "status": "active",
            "metrics": {
                "total_webhooks": 0,
                "successful_webhooks": 0,
                "failed_webhooks": 0,
                "rate_limited": 0,
                "invalid_signatures": 0
            }
        }

        self.webhook_endpoints[endpoint_id] = endpoint_config

        print(f"📍 Registered webhook endpoint: {endpoint_path} -> {endpoint_id}")
        return endpoint_id

    def register_webhook_handler(self, event_type: str, handler: Callable) -> str:
        """Register a webhook handler for specific event type"""
        handler_id = f"handler_{event_type}_{int(time.time())}"

        handler_config = {
            "handler_id": handler_id,
            "event_type": event_type,
            "handler": handler,
            "registered_at": datetime.now(),
            "status": "active",
            "metrics": {
                "total_processed": 0,
                "successful": 0,
                "failed": 0,
                "average_processing_time": 0
            }
        }

        self.webhook_handlers[event_type] = handler_config

        print(f"🎯 Registered webhook handler: {event_type} -> {handler_id}")
        return handler_id

    def process_webhook(self, endpoint_id: str, webhook_data: Dict[str, Any],
                       headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Process an incoming webhook"""
        start_time = time.time()

        if headers is None:
            headers = {}

        # Validate endpoint
        if endpoint_id not in self.webhook_endpoints:
            return self._create_webhook_response("error", "Endpoint not found")

        endpoint_config = self.webhook_endpoints[endpoint_id]

        # Check if endpoint is active
        if endpoint_config["status"] != "active":
            return self._create_webhook_response("error", "Endpoint inactive")

        # Validate authentication if required
        if endpoint_config["auth_required"]:
            auth_result = self._validate_webhook_auth(endpoint_config, headers)
            if not auth_result["valid"]:
                endpoint_config["metrics"]["invalid_signatures"] += 1
                return self._create_webhook_response("error", auth_result["message"])

        # Check rate limiting
        if self._is_rate_limited(endpoint_config):
            endpoint_config["metrics"]["rate_limited"] += 1
            return self._create_webhook_response("error", "Rate limit exceeded", 429)

        # Extract event data
        event_type = webhook_data.get("event_type", "unknown")
        event_data = webhook_data.get("data", {})

        # Update endpoint metrics
        endpoint_config["metrics"]["total_webhooks"] += 1

        # Queue webhook for processing
        webhook_job = {
            "webhook_id": f"webhook_{int(time.time())}_{random.randint(1000, 9999)}",
            "endpoint_id": endpoint_id,
            "event_type": event_type,
            "event_data": event_data,
            "headers": headers,
            "received_at": datetime.now(),
            "processing_started_at": None,
            "processing_completed_at": None,
            "status": "queued",
            "retry_count": 0,
            "max_retries": 3
        }

        # Add to processing queue
        asyncio.run_coroutine_threadsafe(
            self.retry_queues["processing"].put(webhook_job),
            asyncio.get_event_loop()
        )

        # Log webhook receipt
        self._log_webhook_event("received", webhook_job)

        processing_time = time.time() - start_time
        return self._create_webhook_response("accepted", "Webhook queued for processing", 202)

    def _validate_webhook_auth(self, endpoint_config: Dict[str, Any],
                             headers: Dict[str, str]) -> Dict[str, Any]:
        """Validate webhook authentication"""
        # Check for signature
        signature = headers.get("X-Webhook-Signature") or headers.get("X-Hub-Signature-256")

        if not signature:
            return {"valid": False, "message": "Missing signature"}

        # Verify signature
        secret = endpoint_config["webhook_secret"]
        payload = json.dumps(endpoint_config, sort_keys=True).encode()  # Simplified

        try:
            if signature.startswith("sha256="):
                expected_signature = hmac.new(
                    secret.encode(),
                    payload,
                    hashlib.sha256
                ).hexdigest()

                if hmac.compare_digest(signature[7:], expected_signature):
                    return {"valid": True}
            else:
                # Try direct comparison for simple signatures
                expected_signature = hmac.new(
                    secret.encode(),
                    payload,
                    hashlib.sha256
                ).hexdigest()

                if hmac.compare_digest(signature, expected_signature):
                    return {"valid": True}

        except Exception as e:
            return {"valid": False, "message": f"Signature verification error: {str(e)}"}

        return {"valid": False, "message": "Invalid signature"}

    def _is_rate_limited(self, endpoint_config: Dict[str, Any]) -> bool:
        """Check if endpoint is rate limited"""
        # Simple rate limiting (could be enhanced with Redis)
        current_time = time.time()
        rate_limit_window = 60  # 1 minute window

        # Track requests in sliding window (simplified)
        if not hasattr(endpoint_config, 'request_times'):
            endpoint_config['request_times'] = deque(maxlen=endpoint_config["rate_limit"])

        request_times = endpoint_config['request_times']

        # Remove old requests
        while request_times and request_times[0] < current_time - rate_limit_window:
            request_times.popleft()

        # Check if under limit
        if len(request_times) >= endpoint_config["rate_limit"]:
            return True

        request_times.append(current_time)
        return False

    def _create_webhook_response(self, status: str, message: str,
                               status_code: int = 200) -> Dict[str, Any]:
        """Create standardized webhook response"""
        return {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code
        }

    def _webhook_worker(self):
        """Background webhook processing worker"""
        while self.is_running:
            try:
                # Get webhook from processing queue
                webhook_job = asyncio.run(self.retry_queues["processing"].get())

                # Process webhook
                self._process_webhook_job(webhook_job)

                self.retry_queues["processing"].task_done()

            except asyncio.QueueEmpty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Webhook worker error: {e}")

    def _process_webhook_job(self, webhook_job: Dict[str, Any]):
        """Process a webhook job"""
        webhook_job["processing_started_at"] = datetime.now()
        webhook_job["status"] = "processing"

        try:
            event_type = webhook_job["event_type"]
            event_data = webhook_job["event_data"]

            # Get handler for event type
            handler_config = self.webhook_handlers.get(event_type)
            if not handler_config:
                raise ValueError(f"No handler registered for event type: {event_type}")

            # Call handler
            handler = handler_config["handler"]
            if asyncio.iscoroutinefunction(handler):
                result = asyncio.run(handler(event_data, webhook_job))
            else:
                result = handler(event_data, webhook_job)

            # Update metrics
            webhook_job["status"] = "completed"
            webhook_job["result"] = result
            handler_config["metrics"]["total_processed"] += 1
            handler_config["metrics"]["successful"] += 1

            # Update endpoint metrics
            endpoint_config = self.webhook_endpoints[webhook_job["endpoint_id"]]
            endpoint_config["metrics"]["successful_webhooks"] += 1

        except Exception as e:
            webhook_job["status"] = "failed"
            webhook_job["error"] = str(e)

            # Update metrics
            handler_config = self.webhook_handlers.get(webhook_job["event_type"])
            if handler_config:
                handler_config["metrics"]["failed"] += 1

            endpoint_config = self.webhook_endpoints[webhook_job["endpoint_id"]]
            endpoint_config["metrics"]["failed_webhooks"] += 1

            # Queue for retry if retries remaining
            if webhook_job["retry_count"] < webhook_job["max_retries"]:
                webhook_job["retry_count"] += 1
                asyncio.run_coroutine_threadsafe(
                    self.retry_queues["retry"].put(webhook_job),
                    asyncio.get_event_loop()
                )

        finally:
            processing_time = time.time() - webhook_job["processing_started_at"].timestamp()
            webhook_job["processing_completed_at"] = datetime.now()

            # Update processing time metrics
            handler_config = self.webhook_handlers.get(webhook_job["event_type"])
            if handler_config:
                handler_config["metrics"]["processing_times"] = handler_config["metrics"].get("processing_times", deque(maxlen=100))
                handler_config["metrics"]["processing_times"].append(processing_time)

                times = list(handler_config["metrics"]["processing_times"])
                handler_config["metrics"]["average_processing_time"] = sum(times) / len(times) if times else 0

            # Log completion
            self._log_webhook_event("processed", webhook_job)

    def _retry_worker(self):
        """Background retry worker"""
        while self.is_running:
            try:
                # Get failed webhook for retry
                webhook_job = asyncio.run(self.retry_queues["retry"].get())

                # Add exponential backoff delay
                delay = 2 ** webhook_job["retry_count"]  # Exponential backoff
                time.sleep(min(delay, 300))  # Max 5 minutes delay

                # Re-queue for processing
                asyncio.run_coroutine_threadsafe(
                    self.retry_queues["processing"].put(webhook_job),
                    asyncio.get_event_loop()
                )

                self.retry_queues["retry"].task_done()

            except asyncio.QueueEmpty:
                time.sleep(1)
            except Exception as e:
                print(f"Retry worker error: {e}")

    def _cleanup_worker(self):
        """Background cleanup worker"""
        while self.is_running:
            try:
                # Clean up old webhook logs (keep last 7 days)
                cutoff_date = datetime.now() - timedelta(days=7)

                for endpoint_id in list(self.webhook_logs.keys()):
                    self.webhook_logs[endpoint_id] = [
                        log for log in self.webhook_logs[endpoint_id]
                        if log.get("timestamp", datetime.min) > cutoff_date
                    ]

                    # Remove empty log lists
                    if not self.webhook_logs[endpoint_id]:
                        del self.webhook_logs[endpoint_id]

                # Clean up old failed webhooks (keep last 30 days)
                cutoff_date = datetime.now() - timedelta(days=30)

                for event_type in list(self.failed_webhooks.keys()):
                    self.failed_webhooks[event_type] = [
                        failure for failure in self.failed_webhooks[event_type]
                        if failure.get("timestamp", datetime.min) > cutoff_date
                    ]

                    if not self.failed_webhooks[event_type]:
                        del self.failed_webhooks[event_type]

                time.sleep(3600)  # Clean up every hour

            except Exception as e:
                print(f"Cleanup worker error: {e}")

    def _log_webhook_event(self, event: str, webhook_job: Dict[str, Any]):
        """Log webhook event"""
        log_entry = {
            "timestamp": datetime.now(),
            "event": event,
            "webhook_id": webhook_job["webhook_id"],
            "endpoint_id": webhook_job["endpoint_id"],
            "event_type": webhook_job["event_type"],
            "status": webhook_job["status"],
            "retry_count": webhook_job.get("retry_count", 0),
            "processing_time": (
                webhook_job["processing_completed_at"] - webhook_job["processing_started_at"]
            ).total_seconds() if webhook_job.get("processing_completed_at") else None,
            "error": webhook_job.get("error")
        }

        self.webhook_logs[webhook_job["endpoint_id"]].append(log_entry)

        # Log failed webhooks separately
        if webhook_job["status"] == "failed":
            self.failed_webhooks[webhook_job["event_type"]].append({
                "timestamp": datetime.now(),
                "webhook_job": webhook_job,
                "error": webhook_job.get("error"),
                "retry_count": webhook_job.get("retry_count", 0)
            })

    # Default webhook handlers
    async def _handle_ehr_patient_update(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle EHR patient update webhook"""
        patient_id = event_data.get("patient_id")
        updates = event_data.get("updates", {})

        # In production, this would update local patient records
        print(f"📊 Processing patient update for {patient_id}")

        # Simulate processing
        await asyncio.sleep(0.1)

        return {
            "action": "patient_updated",
            "patient_id": patient_id,
            "fields_updated": list(updates.keys()),
            "sync_required": True
        }

    async def _handle_ehr_encounter_created(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle EHR encounter creation webhook"""
        encounter_id = event_data.get("encounter_id")
        patient_id = event_data.get("patient_id")

        print(f"🏥 Processing new encounter {encounter_id} for patient {patient_id}")

        # Simulate processing
        await asyncio.sleep(0.05)

        return {
            "action": "encounter_created",
            "encounter_id": encounter_id,
            "patient_id": patient_id,
            "requires_review": True
        }

    async def _handle_ehr_observation_added(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle EHR observation added webhook"""
        patient_id = event_data.get("patient_id")
        observation_type = event_data.get("observation_type")

        print(f"📈 Processing new observation ({observation_type}) for patient {patient_id}")

        # Simulate processing
        await asyncio.sleep(0.08)

        return {
            "action": "observation_added",
            "patient_id": patient_id,
            "observation_type": observation_type,
            "requires_analysis": True
        }

    async def _handle_ehr_medication_prescribed(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle EHR medication prescription webhook"""
        patient_id = event_data.get("patient_id")
        medication = event_data.get("medication")

        print(f"💊 Processing medication prescription for patient {patient_id}: {medication}")

        # Simulate processing
        await asyncio.sleep(0.06)

        return {
            "action": "medication_prescribed",
            "patient_id": patient_id,
            "medication": medication,
            "requires_review": True,
            "drug_interaction_check": True
        }

    async def _handle_lab_results_ready(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle lab results ready webhook"""
        patient_id = event_data.get("patient_id")
        test_type = event_data.get("test_type")

        print(f"🧪 Processing lab results ({test_type}) for patient {patient_id}")

        # Simulate processing
        await asyncio.sleep(0.1)

        return {
            "action": "lab_results_processed",
            "patient_id": patient_id,
            "test_type": test_type,
            "requires_clinical_review": True,
            "critical_values_check": True
        }

    async def _handle_imaging_study_completed(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle imaging study completed webhook"""
        patient_id = event_data.get("patient_id")
        study_type = event_data.get("study_type")

        print(f"🖼️ Processing imaging study ({study_type}) for patient {patient_id}")

        # Simulate AI analysis
        await asyncio.sleep(0.2)

        return {
            "action": "imaging_study_analyzed",
            "patient_id": patient_id,
            "study_type": study_type,
            "ai_analysis_completed": True,
            "requires_radiologist_review": True
        }

    async def _handle_telemedicine_session_ended(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle telemedicine session ended webhook"""
        session_id = event_data.get("session_id")
        patient_id = event_data.get("patient_id")
        duration = event_data.get("duration_minutes")

        print(f"📹 Processing telemedicine session {session_id} for patient {patient_id}")

        # Simulate processing
        await asyncio.sleep(0.05)

        return {
            "action": "telemedicine_session_processed",
            "session_id": session_id,
            "patient_id": patient_id,
            "duration": duration,
            "transcription_completed": True,
            "summary_generated": True
        }

    async def _handle_device_data_received(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle device data received webhook"""
        patient_id = event_data.get("patient_id")
        device_type = event_data.get("device_type")
        data_points = event_data.get("data_points", 0)

        print(f"📱 Processing device data ({device_type}, {data_points} points) for patient {patient_id}")

        # Simulate processing
        await asyncio.sleep(0.03)

        return {
            "action": "device_data_processed",
            "patient_id": patient_id,
            "device_type": device_type,
            "data_points_processed": data_points,
            "anomalies_detected": random.randint(0, 3),
            "requires_alert": random.random() > 0.8
        }

    async def _handle_clinical_trial_update(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clinical trial update webhook"""
        trial_id = event_data.get("trial_id")
        update_type = event_data.get("update_type")

        print(f"🧫 Processing clinical trial update ({update_type}) for trial {trial_id}")

        # Simulate processing
        await asyncio.sleep(0.07)

        return {
            "action": "clinical_trial_updated",
            "trial_id": trial_id,
            "update_type": update_type,
            "requires_data_refresh": True,
            "compliance_check_required": True
        }

    async def _handle_pharmacy_dispensation(self, event_data: Dict[str, Any], webhook_job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pharmacy dispensation webhook"""
        patient_id = event_data.get("patient_id")
        medication = event_data.get("medication")
        dispensed_quantity = event_data.get("quantity")

        print(f"🏥 Processing pharmacy dispensation for patient {patient_id}: {medication} x{dispensed_quantity}")

        # Simulate processing
        await asyncio.sleep(0.04)

        return {
            "action": "medication_dispensed",
            "patient_id": patient_id,
            "medication": medication,
            "quantity": dispensed_quantity,
            "adherence_check": True,
            "interaction_alert": random.random() > 0.9
        }

    def get_webhook_metrics(self, endpoint_id: str = None) -> Dict[str, Any]:
        """Get webhook processing metrics"""
        if endpoint_id:
            return dict(self.webhook_metrics.get(endpoint_id, {}))

        # Aggregate all endpoint metrics
        total_metrics = {
            "total_endpoints": len(self.webhook_endpoints),
            "total_handlers": len(self.webhook_handlers),
            "total_received": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_retries": 0,
            "average_processing_time": 0
        }

        processing_times = []

        for endpoint_id, metrics in self.webhook_metrics.items():
            total_metrics["total_received"] += metrics["total_received"]
            total_metrics["total_successful"] += metrics["successful_processed"]
            total_metrics["total_failed"] += metrics["failed_processing"]
            total_metrics["total_retries"] += metrics["retries_attempted"]

            if metrics["processing_times"]:
                processing_times.extend(list(metrics["processing_times"]))

        if processing_times:
            total_metrics["average_processing_time"] = sum(processing_times) / len(processing_times)

        return total_metrics

    def get_failed_webhooks(self, event_type: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get failed webhook records"""
        if event_type:
            return self.failed_webhooks.get(event_type, [])[-limit:]

        # Get all failed webhooks
        all_failed = []
        for failures in self.failed_webhooks.values():
            all_failed.extend(failures)

        # Sort by timestamp and return most recent
        all_failed.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_failed[:limit]

    def retry_failed_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Retry a failed webhook"""
        # Find the failed webhook
        for event_type, failures in self.failed_webhooks.items():
            for failure in failures:
                if failure["webhook_job"]["webhook_id"] == webhook_id:
                    # Reset retry count and re-queue
                    failure["webhook_job"]["retry_count"] = 0
                    failure["webhook_job"]["status"] = "retry_queued"

                    asyncio.run_coroutine_threadsafe(
                        self.retry_queues["processing"].put(failure["webhook_job"]),
                        asyncio.get_event_loop()
                    )

                    return {"status": "retry_queued", "webhook_id": webhook_id}

        return {"status": "not_found", "webhook_id": webhook_id}

    def get_webhook_logs(self, endpoint_id: str, start_date: datetime = None,
                        end_date: datetime = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get webhook processing logs"""
        if endpoint_id not in self.webhook_logs:
            return []

        logs = self.webhook_logs[endpoint_id]

        # Filter by date range
        if start_date:
            logs = [log for log in logs if log["timestamp"] >= start_date]
        if end_date:
            logs = [log for log in logs if log["timestamp"] <= end_date]

        # Sort by timestamp (most recent first) and limit
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        return logs[:limit]
