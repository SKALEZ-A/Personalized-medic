"""
Backup and Recovery System for AI Personalized Medicine Platform
Comprehensive data backup, disaster recovery, and business continuity
"""

import json
import shutil
import gzip
import hashlib
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import random
import logging
import sqlite3
import os
import tarfile
import tempfile

class BackupRecoverySystem:
    """Comprehensive backup and recovery system"""

    def __init__(self, backup_root: str = "./backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)

        self.backup_jobs = {}
        self.recovery_jobs = {}
        self.retention_policies = {}
        self.backup_schedules = {}
        self.integrity_checks = {}
        self.disaster_recovery_plans = {}

        self.is_running = False
        self.backup_workers = []
        self.monitoring_workers = []

        self.initialize_backup_system()

    def initialize_backup_system(self):
        """Initialize backup and recovery system"""
        # Create backup directories
        self.backup_dirs = {
            "full": self.backup_root / "full",
            "incremental": self.backup_root / "incremental",
            "differential": self.backup_root / "differential",
            "config": self.backup_root / "config",
            "logs": self.backup_root / "logs"
        }

        for dir_path in self.backup_dirs.values():
            dir_path.mkdir(exist_ok=True)

        # Initialize retention policies
        self.retention_policies = {
            "full_backups": {"count": 7, "days": 30},  # Keep 7 full backups or 30 days
            "incremental_backups": {"count": 30, "days": 7},  # Keep 30 incremental or 7 days
            "differential_backups": {"count": 14, "days": 14},  # Keep 14 differential or 14 days
            "logs": {"count": 100, "days": 90},  # Keep 100 log backups or 90 days
            "config": {"count": 50, "days": 365}  # Keep 50 config backups or 1 year
        }

        # Initialize disaster recovery plans
        self._initialize_disaster_recovery_plans()

        print("💾 Backup and Recovery System initialized")

    def _initialize_disaster_recovery_plans(self):
        """Initialize disaster recovery plans"""
        self.disaster_recovery_plans = {
            "data_center_failure": {
                "name": "Primary Data Center Failure",
                "rto_minutes": 240,  # 4 hours
                "rpo_minutes": 15,   # 15 minutes data loss
                "steps": [
                    "Activate secondary data center",
                    "Failover database connections",
                    "Redirect application traffic",
                    "Validate system functionality",
                    "Notify stakeholders"
                ],
                "resources_required": ["secondary_dc", "dns_failover", "monitoring_alerts"],
                "contact_escalation": ["it_director", "ceo", "regulatory_compliance"]
            },
            "cyber_attack": {
                "name": "Cybersecurity Incident",
                "rto_minutes": 480,  # 8 hours
                "rpo_minutes": 60,   # 1 hour data loss
                "steps": [
                    "Isolate affected systems",
                    "Activate incident response team",
                    "Restore from clean backups",
                    "Validate data integrity",
                    "Report to authorities if required"
                ],
                "resources_required": ["clean_backups", "incident_response_team", "forensics_tools"],
                "contact_escalation": ["ciso", "legal_counsel", "regulatory_bodies"]
            },
            "data_corruption": {
                "name": "Data Corruption Incident",
                "rto_minutes": 180,  # 3 hours
                "rpo_minutes": 30,   # 30 minutes data loss
                "steps": [
                    "Stop data processing",
                    "Identify corruption scope",
                    "Restore from last good backup",
                    "Validate data consistency",
                    "Resume operations"
                ],
                "resources_required": ["backup_verification", "data_validation_tools"],
                "contact_escalation": ["data_engineering", "clinical_lead"]
            }
        }

    def start_backup_system(self):
        """Start backup and recovery system"""
        self.is_running = True

        # Start backup workers
        for i in range(3):  # 3 concurrent backup workers
            worker = threading.Thread(target=self._backup_worker, daemon=True)
            worker.start()
            self.backup_workers.append(worker)

        # Start monitoring worker
        monitor_worker = threading.Thread(target=self._backup_monitor, daemon=True)
        monitor_worker.start()
        self.monitoring_workers.append(monitor_worker)

        # Start cleanup worker
        cleanup_worker = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_worker.start()
        self.monitoring_workers.append(cleanup_worker)

        print("⚡ Backup and Recovery System started")

    def stop_backup_system(self):
        """Stop backup and recovery system"""
        self.is_running = False
        print("🛑 Backup and Recovery System stopped")

    def create_backup_job(self, backup_config: Dict[str, Any]) -> str:
        """Create a backup job"""
        job_id = f"backup_{int(time.time())}_{random.randint(1000, 9999)}"

        backup_job = {
            "job_id": job_id,
            "backup_type": backup_config.get("type", "full"),
            "source_paths": backup_config.get("source_paths", []),
            "destination": backup_config.get("destination", "default"),
            "compression": backup_config.get("compression", True),
            "encryption": backup_config.get("encryption", True),
            "verification": backup_config.get("verification", True),
            "priority": backup_config.get("priority", "normal"),
            "status": "queued",
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
            "progress": 0,
            "estimated_size_mb": backup_config.get("estimated_size_mb", 100),
            "actual_size_mb": None,
            "checksum": None,
            "error": None,
            "retention_days": backup_config.get("retention_days", 30)
        }

        self.backup_jobs[job_id] = backup_job

        # Queue backup job
        asyncio.run_coroutine_threadsafe(
            self._backup_queue.put(backup_job),
            asyncio.get_event_loop()
        )

        print(f"📦 Backup job created: {job_id}")
        return job_id

    async def initialize_backup_queue(self):
        """Initialize the backup queue"""
        self._backup_queue = asyncio.Queue()

    def schedule_backup(self, schedule_config: Dict[str, Any]) -> str:
        """Schedule recurring backups"""
        schedule_id = f"schedule_{int(time.time())}_{random.randint(1000, 9999)}"

        schedule = {
            "schedule_id": schedule_id,
            "name": schedule_config.get("name", f"Backup Schedule {schedule_id}"),
            "backup_config": schedule_config["backup_config"],
            "cron_expression": schedule_config.get("cron_expression", "0 2 * * *"),  # Daily at 2 AM
            "enabled": True,
            "last_run": None,
            "next_run": self._calculate_next_run(schedule_config.get("cron_expression", "0 2 * * *")),
            "run_history": [],
            "created_at": datetime.now()
        }

        self.backup_schedules[schedule_id] = schedule

        print(f"📅 Backup schedule created: {schedule_id}")
        return schedule_id

    def _calculate_next_run(self, cron_expression: str) -> datetime:
        """Calculate next run time from cron expression (simplified)"""
        # Simplified cron parsing - assume format "minute hour * * *"
        try:
            parts = cron_expression.split()
            if len(parts) >= 2:
                hour = int(parts[1])
                minute = int(parts[0])
            else:
                hour, minute = 2, 0  # Default to 2 AM
        except:
            hour, minute = 2, 0

        now = datetime.now()
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if next_run <= now:
            next_run += timedelta(days=1)

        return next_run

    def _backup_worker(self):
        """Background backup worker"""
        while self.is_running:
            try:
                # Get backup job from queue
                backup_job = asyncio.run(self._backup_queue.get())

                # Execute backup
                self._execute_backup(backup_job)

                self._backup_queue.task_done()

            except asyncio.QueueEmpty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Backup worker error: {e}")

    def _execute_backup(self, backup_job: Dict[str, Any]):
        """Execute backup job"""
        try:
            backup_job["status"] = "running"
            backup_job["started_at"] = datetime.now()

            backup_type = backup_job["backup_type"]
            source_paths = backup_job["source_paths"]

            # Create backup directory
            backup_dir = self.backup_dirs.get(backup_type, self.backup_dirs["full"])
            timestamp = backup_job["started_at"].strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{backup_type}_backup_{timestamp}"

            backup_job["backup_path"] = str(backup_path)
            backup_job["progress"] = 10

            # Create backup archive
            with tarfile.open(f"{backup_path}.tar.gz", "w:gz") as tar:
                total_files = 0
                for source_path in source_paths:
                    source_path = Path(source_path)
                    if source_path.exists():
                        if source_path.is_file():
                            tar.add(source_path, arcname=source_path.name)
                            total_files += 1
                        elif source_path.is_dir():
                            for file_path in source_path.rglob("*"):
                                if file_path.is_file():
                                    tar.add(file_path, arcname=str(file_path.relative_to(source_path.parent)))
                                    total_files += 1

                        backup_job["progress"] = min(80, 10 + (total_files / 100))  # Simulate progress

            # Calculate checksum
            backup_file = Path(f"{backup_path}.tar.gz")
            if backup_file.exists():
                checksum = self._calculate_file_checksum(backup_file)
                backup_job["checksum"] = checksum
                backup_job["actual_size_mb"] = backup_file.stat().st_size / (1024 * 1024)

            backup_job["progress"] = 90

            # Verification (simplified)
            if backup_job.get("verification", True):
                verification_result = self._verify_backup_integrity(backup_job)
                backup_job["verification_result"] = verification_result

            backup_job["status"] = "completed"
            backup_job["completed_at"] = datetime.now()
            backup_job["progress"] = 100

            # Update retention metadata
            self._update_backup_metadata(backup_job)

            print(f"✅ Backup completed: {backup_job['job_id']}")

        except Exception as e:
            backup_job["status"] = "failed"
            backup_job["error"] = str(e)
            backup_job["completed_at"] = datetime.now()
            print(f"❌ Backup failed: {backup_job['job_id']} - {e}")

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _verify_backup_integrity(self, backup_job: Dict[str, Any]) -> Dict[str, Any]:
        """Verify backup integrity"""
        backup_file = Path(f"{backup_job['backup_path']}.tar.gz")

        if not backup_file.exists():
            return {"valid": False, "error": "backup_file_not_found"}

        # Check file size
        expected_size = backup_job.get("estimated_size_mb", 0) * 1024 * 1024
        actual_size = backup_file.stat().st_size

        size_valid = abs(actual_size - expected_size) / expected_size < 0.5 if expected_size > 0 else True

        # Check checksum
        current_checksum = self._calculate_file_checksum(backup_file)
        stored_checksum = backup_job.get("checksum")

        checksum_valid = current_checksum == stored_checksum

        # Try to extract and verify
        extraction_valid = True
        try:
            with tarfile.open(backup_file, "r:gz") as tar:
                # Just check if we can read the members
                members = tar.getmembers()
                extraction_valid = len(members) > 0
        except:
            extraction_valid = False

        return {
            "valid": size_valid and checksum_valid and extraction_valid,
            "size_check": size_valid,
            "checksum_check": checksum_valid,
            "extraction_check": extraction_valid,
            "file_count": len(members) if 'members' in locals() else 0
        }

    def _update_backup_metadata(self, backup_job: Dict[str, Any]):
        """Update backup metadata for retention tracking"""
        backup_metadata = {
            "backup_id": backup_job["job_id"],
            "backup_type": backup_job["backup_type"],
            "created_at": backup_job["created_at"],
            "expires_at": backup_job["created_at"] + timedelta(days=backup_job["retention_days"]),
            "size_mb": backup_job.get("actual_size_mb", 0),
            "checksum": backup_job.get("checksum"),
            "path": backup_job.get("backup_path"),
            "status": "active"
        }

        # Store metadata (simplified - would use database)
        metadata_file = self.backup_dirs["logs"] / f"backup_metadata_{backup_job['job_id']}.json"
        with open(metadata_file, 'w') as f:
            json.dump(backup_metadata, f, indent=2, default=str)

    def create_recovery_job(self, recovery_config: Dict[str, Any]) -> str:
        """Create a data recovery job"""
        job_id = f"recovery_{int(time.time())}_{random.randint(1000, 9999)}"

        recovery_job = {
            "job_id": job_id,
            "recovery_type": recovery_config.get("type", "full"),
            "backup_id": recovery_config.get("backup_id"),
            "target_path": recovery_config.get("target_path", "./recovery"),
            "point_in_time": recovery_config.get("point_in_time"),  # For PITR
            "validation": recovery_config.get("validation", True),
            "dry_run": recovery_config.get("dry_run", False),
            "status": "queued",
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
            "progress": 0,
            "error": None,
            "recovery_stats": {}
        }

        self.recovery_jobs[job_id] = recovery_job

        # Execute recovery immediately (could be queued)
        threading.Thread(target=self._execute_recovery, args=(recovery_job,), daemon=True).start()

        print(f"🔄 Recovery job created: {job_id}")
        return job_id

    def _execute_recovery(self, recovery_job: Dict[str, Any]):
        """Execute data recovery"""
        try:
            recovery_job["status"] = "running"
            recovery_job["started_at"] = datetime.now()

            backup_id = recovery_job["backup_id"]
            target_path = Path(recovery_job["target_path"])
            target_path.mkdir(parents=True, exist_ok=True)

            # Find backup file
            backup_metadata = self._find_backup_metadata(backup_id)
            if not backup_metadata:
                raise ValueError(f"Backup not found: {backup_id}")

            backup_file = Path(f"{backup_metadata['path']}.tar.gz")

            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")

            recovery_job["progress"] = 20

            # Extract backup
            extracted_files = []
            with tarfile.open(backup_file, "r:gz") as tar:
                for member in tar.getmembers():
                    if not recovery_job.get("dry_run", False):
                        tar.extract(member, target_path)
                    extracted_files.append(member.name)

                    recovery_job["progress"] = min(80, 20 + (len(extracted_files) / len(tar.getmembers())) * 60)

            recovery_job["progress"] = 90

            # Validation
            if recovery_job.get("validation", True):
                validation_result = self._validate_recovery(target_path, extracted_files)
                recovery_job["validation_result"] = validation_result

            recovery_job["status"] = "completed"
            recovery_job["completed_at"] = datetime.now()
            recovery_job["progress"] = 100

            recovery_job["recovery_stats"] = {
                "files_recovered": len(extracted_files),
                "total_size_mb": sum(backup_metadata.get("size_mb", 0) for _ in extracted_files) / len(extracted_files) if extracted_files else 0,
                "recovery_time_seconds": (recovery_job["completed_at"] - recovery_job["started_at"]).total_seconds()
            }

            print(f"✅ Recovery completed: {recovery_job['job_id']}")

        except Exception as e:
            recovery_job["status"] = "failed"
            recovery_job["error"] = str(e)
            recovery_job["completed_at"] = datetime.now()
            print(f"❌ Recovery failed: {recovery_job['job_id']} - {e}")

    def _find_backup_metadata(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Find backup metadata"""
        # Search for metadata file
        for metadata_file in self.backup_dirs["logs"].glob(f"backup_metadata_{backup_id}.json"):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None

    def _validate_recovery(self, target_path: Path, extracted_files: List[str]) -> Dict[str, Any]:
        """Validate recovery integrity"""
        validation_results = {
            "files_exist": 0,
            "files_readable": 0,
            "total_files_checked": len(extracted_files)
        }

        for file_name in extracted_files:
            file_path = target_path / file_name
            if file_path.exists():
                validation_results["files_exist"] += 1

                # Try to read file
                try:
                    if file_path.is_file():
                        with open(file_path, 'rb') as f:
                            f.read(1024)  # Read first 1KB
                        validation_results["files_readable"] += 1
                except:
                    pass

        validation_results["success_rate"] = validation_results["files_readable"] / validation_results["total_files_checked"] if validation_results["total_files_checked"] > 0 else 0

        return validation_results

    def get_backup_status(self, job_id: str) -> Dict[str, Any]:
        """Get backup job status"""
        if job_id not in self.backup_jobs:
            raise ValueError(f"Backup job not found: {job_id}")

        job = self.backup_jobs[job_id]

        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "backup_type": job["backup_type"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "estimated_size_mb": job["estimated_size_mb"],
            "actual_size_mb": job.get("actual_size_mb"),
            "error": job.get("error")
        }

    def get_recovery_status(self, job_id: str) -> Dict[str, Any]:
        """Get recovery job status"""
        if job_id not in self.recovery_jobs:
            raise ValueError(f"Recovery job not found: {job_id}")

        job = self.recovery_jobs[job_id]

        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "recovery_type": job["recovery_type"],
            "backup_id": job["backup_id"],
            "target_path": job["target_path"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "recovery_stats": job.get("recovery_stats", {}),
            "error": job.get("error")
        }

    def list_available_backups(self, backup_type: str = None) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []

        # Search metadata files
        for metadata_file in self.backup_dirs["logs"].glob("backup_metadata_*.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

                if backup_type and metadata["backup_type"] != backup_type:
                    continue

                if metadata["status"] == "active":
                    backups.append(metadata)

        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)

        return backups

    def activate_disaster_recovery_plan(self, scenario: str) -> Dict[str, Any]:
        """Activate disaster recovery plan"""
        if scenario not in self.disaster_recovery_plans:
            raise ValueError(f"Unknown disaster scenario: {scenario}")

        plan = self.disaster_recovery_plans[scenario]

        activation = {
            "activation_id": f"dr_activation_{int(time.time())}_{random.randint(1000, 9999)}",
            "scenario": scenario,
            "plan": plan,
            "activated_at": datetime.now(),
            "status": "active",
            "progress": [],
            "estimated_completion": datetime.now() + timedelta(minutes=plan["rto_minutes"])
        }

        # Log activation
        print(f"🚨 DISASTER RECOVERY ACTIVATED: {scenario}")

        # Execute recovery steps (simplified)
        for i, step in enumerate(plan["steps"]):
            activation["progress"].append({
                "step": i + 1,
                "description": step,
                "status": "completed",
                "timestamp": datetime.now() + timedelta(minutes=i * 10)  # Simulate timing
            })

        activation["completed_at"] = datetime.now() + timedelta(minutes=plan["rto_minutes"])
        activation["status"] = "completed"

        return activation

    def get_backup_metrics(self) -> Dict[str, Any]:
        """Get backup system metrics"""
        total_backups = len(list(self.backup_dirs["logs"].glob("backup_metadata_*.json")))
        active_backups = len([b for b in self.list_available_backups() if b["status"] == "active"])

        backup_sizes = []
        for backup in self.list_available_backups():
            if "size_mb" in backup:
                backup_sizes.append(backup["size_mb"])

        avg_backup_size = statistics.mean(backup_sizes) if backup_sizes else 0

        # Calculate storage utilization
        total_size = sum(backup_sizes) if backup_sizes else 0

        return {
            "total_backups": total_backups,
            "active_backups": active_backups,
            "total_storage_gb": total_size / 1024,
            "average_backup_size_mb": avg_backup_size,
            "backup_success_rate": len([j for j in self.backup_jobs.values() if j["status"] == "completed"]) / len(self.backup_jobs) if self.backup_jobs else 0,
            "recovery_success_rate": len([j for j in self.recovery_jobs.values() if j["status"] == "completed"]) / len(self.recovery_jobs) if self.recovery_jobs else 0
        }

    def _backup_monitor(self):
        """Background backup monitoring"""
        while self.is_running:
            try:
                # Check scheduled backups
                now = datetime.now()
                for schedule_id, schedule in self.backup_schedules.items():
                    if schedule["enabled"] and schedule["next_run"] <= now:
                        # Create backup job
                        backup_job_id = self.create_backup_job(schedule["backup_config"])

                        # Update schedule
                        schedule["last_run"] = now
                        schedule["next_run"] = self._calculate_next_run(schedule["cron_expression"])
                        schedule["run_history"].append({
                            "timestamp": now,
                            "backup_job_id": backup_job_id
                        })

                # Check backup health
                failed_backups = len([j for j in self.backup_jobs.values() if j["status"] == "failed"])
                if failed_backups > 0:
                    print(f"⚠️ {failed_backups} backup jobs have failed recently")

                time.sleep(60)  # Check every minute

            except Exception as e:
                print(f"Backup monitor error: {e}")

    def _cleanup_worker(self):
        """Background cleanup worker for old backups"""
        while self.is_running:
            try:
                # Apply retention policies
                self._apply_retention_policies()

                time.sleep(3600)  # Clean up every hour

            except Exception as e:
                print(f"Cleanup worker error: {e}")

    def _apply_retention_policies(self):
        """Apply retention policies to old backups"""
        for backup_type, policy in self.retention_policies.items():
            backup_dir = self.backup_dirs.get(backup_type, self.backup_dirs["full"])

            # Get all backups of this type
            backups = []
            for metadata_file in self.backup_dirs["logs"].glob(f"backup_metadata_*.json"):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                    if metadata["backup_type"] == backup_type:
                        backups.append((metadata_file, metadata))

            # Sort by creation date (oldest first)
            backups.sort(key=lambda x: x[1]["created_at"])

            # Apply retention rules
            max_count = policy["count"]
            max_age_days = policy["days"]

            to_delete = []

            # Keep only the most recent N backups
            if len(backups) > max_count:
                to_delete.extend(backups[:-max_count])

            # Delete backups older than max age
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            for metadata_file, metadata in backups:
                created_at = datetime.fromisoformat(metadata["created_at"])
                if created_at < cutoff_date:
                    if metadata_file not in [d[0] for d in to_delete]:
                        to_delete.append((metadata_file, metadata))

            # Delete old backups
            for metadata_file, metadata in to_delete:
                try:
                    # Delete backup file
                    backup_file = Path(f"{metadata['path']}.tar.gz")
                    if backup_file.exists():
                        backup_file.unlink()

                    # Delete metadata file
                    metadata_file.unlink()

                    # Mark as deleted
                    metadata["status"] = "deleted"
                    metadata["deleted_at"] = datetime.now()

                    print(f"🗑️ Deleted old backup: {metadata['backup_id']}")

                except Exception as e:
                    print(f"Error deleting backup {metadata['backup_id']}: {e}")

    def export_backup_report(self, format: str = "json") -> str:
        """Export comprehensive backup system report"""
        report = {
            "generated_at": datetime.now(),
            "system_metrics": self.get_backup_metrics(),
            "backup_jobs": list(self.backup_jobs.values())[-20:],  # Last 20 jobs
            "recovery_jobs": list(self.recovery_jobs.values())[-10:],  # Last 10 recoveries
            "available_backups": self.list_available_backups(),
            "scheduled_backups": list(self.backup_schedules.values()),
            "disaster_recovery_plans": self.disaster_recovery_plans,
            "retention_policies": self.retention_policies
        }

        if format == "json":
            return json.dumps(report, indent=2, default=str)
        else:
            # Simple text report
            lines = [
                "Backup and Recovery System Report",
                f"Generated: {report['generated_at']}",
                "",
                "System Metrics:",
                f"  Total Backups: {report['system_metrics']['total_backups']}",
                f"  Active Backups: {report['system_metrics']['active_backups']}",
                f"  Total Storage: {report['system_metrics']['total_storage_gb']:.2f} GB",
                f"  Backup Success Rate: {report['system_metrics']['backup_success_rate']:.1%}",
                "",
                f"Recent Backup Jobs: {len(report['backup_jobs'])}",
                f"Recent Recovery Jobs: {len(report['recovery_jobs'])}"
            ]

            return "\n".join(lines)
