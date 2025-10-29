"""
Comprehensive Reporting Systems for Medical Data Analytics
Clinical reports, compliance reporting, and business intelligence
"""

import json
import csv
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
from io import StringIO, BytesIO
import pandas as pd
import numpy as np

class ClinicalReportingEngine:
    """Clinical reporting and analytics engine"""

    def __init__(self):
        self.report_templates = self._initialize_report_templates()
        self.generated_reports = {}
        self.report_cache = {}

    def _initialize_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize clinical report templates"""
        return {
            "patient_summary": {
                "name": "Patient Summary Report",
                "description": "Comprehensive patient health summary",
                "sections": ["demographics", "medical_history", "current_medications",
                           "recent_vitals", "upcoming_appointments", "care_plan"],
                "data_sources": ["patient_records", "vital_signs", "medications", "appointments"],
                "format": "pdf"
            },
            "clinical_outcomes": {
                "name": "Clinical Outcomes Report",
                "description": "Analysis of treatment outcomes and effectiveness",
                "sections": ["outcome_metrics", "treatment_effectiveness",
                           "complication_rates", "patient_satisfaction"],
                "data_sources": ["treatment_records", "outcome_measures", "patient_feedback"],
                "format": "html"
            },
            "population_health": {
                "name": "Population Health Report",
                "description": "Community health trends and analytics",
                "sections": ["demographics", "prevalent_conditions", "health_trends",
                           "preventive_care_coverage", "health_disparities"],
                "data_sources": ["population_data", "claims_data", "public_health_records"],
                "format": "dashboard"
            },
            "quality_metrics": {
                "name": "Quality Metrics Report",
                "description": "Healthcare quality and performance metrics",
                "sections": ["process_measures", "outcome_measures", "patient_experience",
                           "efficiency_metrics", "safety_indicators"],
                "data_sources": ["quality_data", "patient_safety", "efficiency_logs"],
                "format": "pdf"
            }
        }

    def generate_patient_report(self, patient_id: str, report_type: str = "patient_summary",
                              date_range: Tuple[str, str] = None) -> Dict[str, Any]:
        """Generate patient-specific clinical report"""
        if report_type not in self.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")

        template = self.report_templates[report_type]

        # Set default date range if not provided
        if not date_range:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            date_range = (start_date.isoformat(), end_date.isoformat())

        report_id = f"{report_type}_{patient_id}_{int(datetime.now().timestamp())}"

        report_data = {
            "report_id": report_id,
            "patient_id": patient_id,
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "date_range": date_range,
            "sections": {}
        }

        # Generate each section
        for section in template["sections"]:
            report_data["sections"][section] = self._generate_report_section(
                section, patient_id, date_range
            )

        # Calculate summary metrics
        report_data["summary_metrics"] = self._calculate_summary_metrics(report_data)

        # Store report
        self.generated_reports[report_id] = report_data

        return report_data

    def _generate_report_section(self, section_name: str, patient_id: str,
                               date_range: Tuple[str, str]) -> Dict[str, Any]:
        """Generate individual report section"""
        if section_name == "demographics":
            return self._get_patient_demographics(patient_id)
        elif section_name == "medical_history":
            return self._get_medical_history(patient_id, date_range)
        elif section_name == "current_medications":
            return self._get_current_medications(patient_id)
        elif section_name == "recent_vitals":
            return self._get_recent_vitals(patient_id, date_range)
        elif section_name == "upcoming_appointments":
            return self._get_upcoming_appointments(patient_id)
        elif section_name == "care_plan":
            return self._get_care_plan(patient_id)
        elif section_name == "outcome_metrics":
            return self._calculate_outcome_metrics(patient_id, date_range)
        elif section_name == "treatment_effectiveness":
            return self._analyze_treatment_effectiveness(patient_id, date_range)
        else:
            return {"data": [], "summary": "Section not implemented"}

    def _get_patient_demographics(self, patient_id: str) -> Dict[str, Any]:
        """Get patient demographics section"""
        # Simulate patient demographics data
        demographics = {
            "patient_id": patient_id,
            "name": f"Patient {patient_id[-4:]}",
            "age": 45,
            "gender": "Female",
            "ethnicity": "Caucasian",
            "address": "123 Main St, Anytown, USA",
            "phone": "(555) 123-4567",
            "email": f"patient{patient_id[-4:]}@example.com",
            "emergency_contact": {
                "name": "John Doe",
                "relationship": "Spouse",
                "phone": "(555) 987-6543"
            }
        }

        return {
            "data": demographics,
            "summary": f"Patient is a {demographics['age']}-year-old {demographics['gender'].lower()}"
        }

    def _get_medical_history(self, patient_id: str, date_range: Tuple[str, str]) -> Dict[str, Any]:
        """Get medical history section"""
        # Simulate medical history
        conditions = [
            {
                "condition": "Type 2 Diabetes Mellitus",
                "diagnosed_date": "2022-03-15",
                "status": "Active",
                "severity": "Moderate"
            },
            {
                "condition": "Hypertension",
                "diagnosed_date": "2021-08-22",
                "status": "Active",
                "severity": "Mild"
            }
        ]

        procedures = [
            {
                "procedure": "Colonoscopy",
                "date": "2023-06-10",
                "outcome": "Normal"
            }
        ]

        return {
            "conditions": conditions,
            "procedures": procedures,
            "allergies": ["Penicillin", "Sulfa drugs"],
            "summary": f"Patient has {len(conditions)} active conditions and {len(procedures)} recent procedures"
        }

    def _get_current_medications(self, patient_id: str) -> Dict[str, Any]:
        """Get current medications section"""
        medications = [
            {
                "name": "Metformin",
                "dosage": "500mg",
                "frequency": "twice daily",
                "prescribed_date": "2023-01-15",
                "prescribed_by": "Dr. Smith"
            },
            {
                "name": "Lisinopril",
                "dosage": "10mg",
                "frequency": "once daily",
                "prescribed_date": "2022-09-01",
                "prescribed_by": "Dr. Johnson"
            }
        ]

        return {
            "medications": medications,
            "active_count": len(medications),
            "summary": f"Patient is currently taking {len(medications)} medications"
        }

    def _get_recent_vitals(self, patient_id: str, date_range: Tuple[str, str]) -> Dict[str, Any]:
        """Get recent vital signs section"""
        vitals = [
            {
                "date": "2024-01-15",
                "blood_pressure": "128/82",
                "heart_rate": 72,
                "temperature": 98.6,
                "weight": 165.5,
                "bmi": 26.8
            },
            {
                "date": "2024-01-08",
                "blood_pressure": "132/85",
                "heart_rate": 75,
                "temperature": 98.4,
                "weight": 166.0,
                "bmi": 27.0
            }
        ]

        return {
            "vitals": vitals,
            "trends": self._calculate_vital_trends(vitals),
            "summary": f"Latest vitals from {vitals[0]['date']}: BP {vitals[0]['blood_pressure']}, HR {vitals[0]['heart_rate']} bpm"
        }

    def _calculate_vital_trends(self, vitals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trends in vital signs"""
        if len(vitals) < 2:
            return {"insufficient_data": True}

        trends = {}
        metrics = ["heart_rate", "weight", "bmi"]

        for metric in metrics:
            values = [v[metric] for v in vitals if metric in v]
            if len(values) >= 2:
                change = values[0] - values[-1]
                trend = "stable"
                if abs(change) > 5:
                    trend = "increasing" if change > 0 else "decreasing"
                trends[metric] = {
                    "current": values[0],
                    "previous": values[-1],
                    "change": change,
                    "trend": trend
                }

        return trends

    def _get_upcoming_appointments(self, patient_id: str) -> Dict[str, Any]:
        """Get upcoming appointments section"""
        appointments = [
            {
                "date": "2024-01-25",
                "time": "10:00 AM",
                "provider": "Dr. Smith",
                "type": "Diabetes Follow-up",
                "location": "Clinic A"
            },
            {
                "date": "2024-02-10",
                "time": "2:30 PM",
                "provider": "Dr. Johnson",
                "type": "Cardiology Consultation",
                "location": "Clinic B"
            }
        ]

        return {
            "appointments": appointments,
            "next_appointment": appointments[0] if appointments else None,
            "summary": f"Next appointment: {appointments[0]['date']} at {appointments[0]['time']} with {appointments[0]['provider']}"
        }

    def _get_care_plan(self, patient_id: str) -> Dict[str, Any]:
        """Get care plan section"""
        care_plan = {
            "goals": [
                "Achieve HbA1c < 7.0%",
                "Maintain blood pressure < 130/80",
                "Lose 10 lbs over 6 months"
            ],
            "interventions": [
                "Medication management",
                "Dietary counseling",
                "Exercise program",
                "Regular monitoring"
            ],
            "care_team": [
                "Dr. Smith (Primary Care)",
                "Dr. Johnson (Cardiology)",
                "Nutritionist Sarah"
            ],
            "monitoring_schedule": "Monthly visits, weekly weight checks"
        }

        return {
            "care_plan": care_plan,
            "status": "Active",
            "last_updated": "2024-01-10",
            "summary": f"Active care plan with {len(care_plan['goals'])} goals and {len(care_plan['interventions'])} interventions"
        }

    def _calculate_summary_metrics(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics for the report"""
        metrics = {
            "total_conditions": 0,
            "active_medications": 0,
            "upcoming_appointments": 0,
            "care_plan_status": "Unknown"
        }

        sections = report_data.get("sections", {})

        if "medical_history" in sections:
            metrics["total_conditions"] = len(sections["medical_history"].get("conditions", []))

        if "current_medications" in sections:
            metrics["active_medications"] = sections["current_medications"].get("active_count", 0)

        if "upcoming_appointments" in sections:
            metrics["upcoming_appointments"] = len(sections["upcoming_appointments"].get("appointments", []))

        if "care_plan" in sections:
            metrics["care_plan_status"] = sections["care_plan"].get("status", "Unknown")

        return metrics

    def export_report(self, report_id: str, format_type: str = "pdf") -> bytes:
        """Export report in specified format"""
        if report_id not in self.generated_reports:
            raise ValueError(f"Report not found: {report_id}")

        report_data = self.generated_reports[report_id]

        if format_type == "pdf":
            return self._export_pdf(report_data)
        elif format_type == "html":
            return self._export_html(report_data)
        elif format_type == "json":
            return json.dumps(report_data, indent=2).encode('utf-8')
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_pdf(self, report_data: Dict[str, Any]) -> bytes:
        """Export report as PDF (simplified)"""
        # In real implementation, would use reportlab or similar
        pdf_content = f"""
        AI Personalized Medicine Platform - Clinical Report
        Report ID: {report_data['report_id']}
        Patient ID: {report_data['patient_id']}
        Generated: {report_data['generated_at']}

        Summary Metrics:
        {json.dumps(report_data.get('summary_metrics', {}), indent=2)}
        """
        return pdf_content.encode('utf-8')

    def _export_html(self, report_data: Dict[str, Any]) -> bytes:
        """Export report as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clinical Report - {report_data['patient_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Personalized Medicine Platform</h1>
                <h2>Clinical Report</h2>
                <p>Report ID: {report_data['report_id']}</p>
                <p>Patient ID: {report_data['patient_id']}</p>
                <p>Generated: {report_data['generated_at']}</p>
            </div>

            <div class="section">
                <h3>Summary Metrics</h3>
                {self._generate_metrics_html(report_data.get('summary_metrics', {}))}
            </div>
        </body>
        </html>
        """
        return html_content.encode('utf-8')

    def _generate_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Generate HTML for metrics display"""
        html = ""
        for key, value in metrics.items():
            html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        return html

class PopulationHealthAnalytics:
    """Population health analytics and reporting"""

    def __init__(self):
        self.population_data = {}
        self.health_indicators = {}
        self.demographic_data = {}

    def analyze_population_health(self, population_segment: str = "all",
                                time_period: str = "1_year") -> Dict[str, Any]:
        """Analyze population health metrics"""
        analysis = {
            "population_segment": population_segment,
            "time_period": time_period,
            "analyzed_at": datetime.now().isoformat(),
            "metrics": {}
        }

        # Calculate health prevalence
        analysis["metrics"]["condition_prevalence"] = self._calculate_condition_prevalence(population_segment)

        # Calculate health outcomes
        analysis["metrics"]["health_outcomes"] = self._calculate_health_outcomes(population_segment, time_period)

        # Calculate preventive care coverage
        analysis["metrics"]["preventive_care"] = self._calculate_preventive_care_coverage(population_segment)

        # Calculate health disparities
        analysis["metrics"]["health_disparities"] = self._calculate_health_disparities(population_segment)

        # Calculate cost analysis
        analysis["metrics"]["cost_analysis"] = self._calculate_cost_analysis(population_segment, time_period)

        return analysis

    def _calculate_condition_prevalence(self, population_segment: str) -> Dict[str, Any]:
        """Calculate prevalence of health conditions"""
        # Simulate condition prevalence data
        conditions = {
            "diabetes": {"prevalence": 0.12, "trend": "increasing", "confidence": 0.95},
            "hypertension": {"prevalence": 0.28, "trend": "stable", "confidence": 0.97},
            "obesity": {"prevalence": 0.35, "trend": "increasing", "confidence": 0.93},
            "depression": {"prevalence": 0.15, "trend": "increasing", "confidence": 0.89},
            "asthma": {"prevalence": 0.08, "trend": "stable", "confidence": 0.91}
        }

        return {
            "conditions": conditions,
            "total_population_analyzed": 10000,
            "most_prevalent": max(conditions.keys(), key=lambda x: conditions[x]["prevalence"]),
            "summary": f"Analysis of {len(conditions)} major conditions in population segment '{population_segment}'"
        }

    def _calculate_health_outcomes(self, population_segment: str, time_period: str) -> Dict[str, Any]:
        """Calculate health outcomes metrics"""
        outcomes = {
            "hospitalization_rate": {
                "rate": 0.045,
                "change_from_previous": -0.005,
                "benchmark_comparison": "below_average"
            },
            "emergency_visits": {
                "rate": 0.125,
                "change_from_previous": 0.012,
                "benchmark_comparison": "average"
            },
            "preventable_admissions": {
                "rate": 0.018,
                "change_from_previous": -0.008,
                "benchmark_comparison": "below_average"
            },
            "mortality_rate": {
                "rate": 0.0032,
                "change_from_previous": -0.0003,
                "benchmark_comparison": "below_average"
            }
        }

        return {
            "outcomes": outcomes,
            "overall_health_score": 78.5,  # Out of 100
            "improvement_trend": "positive",
            "summary": f"Health outcomes analysis for {time_period} period"
        }

    def _calculate_preventive_care_coverage(self, population_segment: str) -> Dict[str, Any]:
        """Calculate preventive care coverage rates"""
        preventive_measures = {
            "annual_physical": {"coverage": 0.68, "target": 0.80},
            "flu_vaccination": {"coverage": 0.45, "target": 0.70},
            "cancer_screening": {"coverage": 0.52, "target": 0.75},
            "blood_pressure_check": {"coverage": 0.71, "target": 0.85},
            "cholesterol_screening": {"coverage": 0.58, "target": 0.70}
        }

        overall_coverage = statistics.mean([m["coverage"] for m in preventive_measures.values()])

        return {
            "preventive_measures": preventive_measures,
            "overall_coverage": overall_coverage,
            "coverage_gaps": [
                measure for measure, data in preventive_measures.items()
                if data["coverage"] < data["target"]
            ],
            "recommendations": [
                "Increase flu vaccination campaigns",
                "Improve cancer screening outreach",
                "Enhance cholesterol screening programs"
            ]
        }

    def _calculate_health_disparities(self, population_segment: str) -> Dict[str, Any]:
        """Calculate health disparities across demographic groups"""
        disparities = {
            "by_age": {
                "18-34": {"diabetes_rate": 0.05, "hypertension_rate": 0.12},
                "35-54": {"diabetes_rate": 0.15, "hypertension_rate": 0.25},
                "55-74": {"diabetes_rate": 0.22, "hypertension_rate": 0.38},
                "75+": {"diabetes_rate": 0.28, "hypertension_rate": 0.45}
            },
            "by_ethnicity": {
                "White": {"diabetes_rate": 0.12, "hypertension_rate": 0.28},
                "Black": {"diabetes_rate": 0.18, "hypertension_rate": 0.35},
                "Hispanic": {"diabetes_rate": 0.16, "hypertension_rate": 0.32},
                "Asian": {"diabetes_rate": 0.10, "hypertension_rate": 0.25}
            },
            "by_income": {
                "Low": {"diabetes_rate": 0.20, "hypertension_rate": 0.40},
                "Middle": {"diabetes_rate": 0.14, "hypertension_rate": 0.30},
                "High": {"diabetes_rate": 0.08, "hypertension_rate": 0.22}
            }
        }

        return {
            "disparities": disparities,
            "largest_gap": "Income-based disparities in diabetes rates",
            "priority_interventions": [
                "Target low-income communities for diabetes prevention",
                "Address hypertension disparities in Black communities",
                "Focus on elderly population health management"
            ],
            "equity_score": 65.2  # Out of 100
        }

    def _calculate_cost_analysis(self, population_segment: str, time_period: str) -> Dict[str, Any]:
        """Calculate healthcare cost analysis"""
        cost_analysis = {
            "total_cost_per_capita": 8500,
            "preventive_vs_reactive_ratio": 0.35,  # 35% preventive, 65% reactive
            "high_cost_conditions": [
                {"condition": "Diabetes", "cost_per_patient": 12000, "prevalence": 0.12},
                {"condition": "Heart Disease", "cost_per_patient": 18000, "prevalence": 0.08},
                {"condition": "Cancer", "cost_per_patient": 45000, "prevalence": 0.03}
            ],
            "cost_trends": {
                "emergency_care": {"change_percent": 8.5},
                "preventive_care": {"change_percent": 12.3},
                "pharmacy": {"change_percent": 5.2}
            },
            "roi_analysis": {
                "preventive_programs": {"roi": 2.8, "payback_period_months": 18},
                "chronic_disease_management": {"roi": 3.2, "payback_period_months": 15}
            }
        }

        return cost_analysis

class ComplianceReportingEngine:
    """Regulatory compliance and audit reporting"""

    def __init__(self):
        self.compliance_frameworks = {
            "hipaa": self._initialize_hipaa_compliance(),
            "gdpr": self._initialize_gdpr_compliance(),
            "hitech": self._initialize_hitech_compliance(),
            "meaningful_use": self._initialize_meaningful_use_compliance()
        }
        self.audit_logs = []
        self.compliance_alerts = []

    def _initialize_hipaa_compliance(self) -> Dict[str, Any]:
        return {
            "name": "HIPAA",
            "requirements": ["Privacy Rule", "Security Rule", "Breach Notification"],
            "check_frequency": "annual",
            "last_audit": "2023-10-15",
            "compliance_score": 92
        }

    def _initialize_gdpr_compliance(self) -> Dict[str, Any]:
        return {
            "name": "GDPR",
            "requirements": ["Data Protection", "Consent Management", "Right to Access"],
            "check_frequency": "continuous",
            "last_audit": "2023-11-20",
            "compliance_score": 88
        }

    def _initialize_hitech_compliance(self) -> Dict[str, Any]:
        return {
            "name": "HITECH",
            "requirements": ["Electronic Health Records", "Data Security", "Audit Controls"],
            "check_frequency": "quarterly",
            "last_audit": "2023-12-01",
            "compliance_score": 95
        }

    def _initialize_meaningful_use_compliance(self) -> Dict[str, Any]:
        return {
            "name": "Meaningful Use",
            "requirements": ["EHR Adoption", "Clinical Quality Measures", "Patient Engagement"],
            "check_frequency": "quarterly",
            "last_audit": "2023-12-15",
            "compliance_score": 87
        }

    def generate_compliance_report(self, framework: str = "all",
                                 date_range: Tuple[str, str] = None) -> Dict[str, Any]:
        """Generate compliance report"""
        if not date_range:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            date_range = (start_date.isoformat(), end_date.isoformat())

        report = {
            "report_type": "compliance",
            "generated_at": datetime.now().isoformat(),
            "date_range": date_range,
            "frameworks": {}
        }

        frameworks_to_check = [framework] if framework != "all" else list(self.compliance_frameworks.keys())

        for fw in frameworks_to_check:
            if fw in self.compliance_frameworks:
                report["frameworks"][fw] = self._assess_framework_compliance(fw, date_range)

        # Overall compliance assessment
        report["overall_assessment"] = self._calculate_overall_compliance(report["frameworks"])

        return report

    def _assess_framework_compliance(self, framework: str, date_range: Tuple[str, str]) -> Dict[str, Any]:
        """Assess compliance for specific framework"""
        framework_config = self.compliance_frameworks[framework]

        assessment = {
            "framework": framework_config["name"],
            "compliance_score": framework_config["compliance_score"],
            "requirements_check": {},
            "audit_findings": [],
            "recommendations": []
        }

        # Check each requirement
        for requirement in framework_config["requirements"]:
            assessment["requirements_check"][requirement] = self._check_requirement_compliance(
                framework, requirement, date_range
            )

        # Generate audit findings
        assessment["audit_findings"] = self._generate_audit_findings(framework, date_range)

        # Generate recommendations
        assessment["recommendations"] = self._generate_compliance_recommendations(framework, assessment)

        return assessment

    def _check_requirement_compliance(self, framework: str, requirement: str,
                                    date_range: Tuple[str, str]) -> Dict[str, Any]:
        """Check compliance for specific requirement"""
        # Simulate compliance check
        compliance_checks = {
            "hipaa_privacy": {"status": "compliant", "last_check": "2023-12-01", "issues": 0},
            "hipaa_security": {"status": "compliant", "last_check": "2023-12-01", "issues": 1},
            "gdpr_consent": {"status": "needs_attention", "last_check": "2023-11-15", "issues": 3},
            "ehr_adoption": {"status": "compliant", "last_check": "2023-12-15", "issues": 0}
        }

        check_key = f"{framework}_{requirement.lower().replace(' ', '_')}"
        return compliance_checks.get(check_key, {"status": "unknown", "issues": 0})

    def _generate_audit_findings(self, framework: str, date_range: Tuple[str, str]) -> List[Dict[str, Any]]:
        """Generate audit findings for framework"""
        findings = []

        # Simulate findings based on framework
        if framework == "hipaa":
            findings = [
                {
                    "finding_id": "HIPAA-001",
                    "severity": "low",
                    "description": "Minor delay in access request response",
                    "status": "resolved",
                    "resolution_date": "2023-11-20"
                }
            ]
        elif framework == "gdpr":
            findings = [
                {
                    "finding_id": "GDPR-001",
                    "severity": "medium",
                    "description": "Data subject access request process needs optimization",
                    "status": "in_progress",
                    "due_date": "2024-02-01"
                }
            ]

        return findings

    def _generate_compliance_recommendations(self, framework: str, assessment: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []

        if assessment["compliance_score"] < 90:
            recommendations.append("Conduct comprehensive compliance training for staff")
            recommendations.append("Implement automated compliance monitoring systems")

        if any(check["status"] != "compliant" for check in assessment["requirements_check"].values()):
            recommendations.append("Address outstanding compliance issues promptly")

        if framework == "gdpr":
            recommendations.append("Enhance data subject consent management processes")
            recommendations.append("Implement data minimization practices")

        return recommendations

    def _calculate_overall_compliance(self, framework_assessments: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall compliance assessment"""
        scores = [fw["compliance_score"] for fw in framework_assessments.values()]
        overall_score = statistics.mean(scores) if scores else 0

        assessment = {
            "overall_compliance_score": round(overall_score, 1),
            "grade": self._calculate_compliance_grade(overall_score),
            "critical_findings": sum(len(fw["audit_findings"]) for fw in framework_assessments.values()),
            "status": "compliant" if overall_score >= 85 else "needs_attention"
        }

        return assessment

    def _calculate_compliance_grade(self, score: float) -> str:
        """Calculate compliance grade"""
        if score >= 95:
            return "A"
        elif score >= 90:
            return "B"
        elif score >= 85:
            return "C"
        elif score >= 80:
            return "D"
        else:
            return "F"

class DataExportEngine:
    """Data export and analytics engine"""

    def __init__(self):
        self.export_formats = ["csv", "json", "xml", "parquet", "excel"]
        self.export_history = []

    def export_data(self, data_source: str, filters: Dict[str, Any] = None,
                   format_type: str = "csv", compression: str = None) -> Dict[str, Any]:
        """Export data in specified format"""
        if format_type not in self.export_formats:
            raise ValueError(f"Unsupported export format: {format_type}")

        export_id = f"export_{int(datetime.now().timestamp())}"

        # Get data (simplified - would query actual data sources)
        data = self._get_data_for_export(data_source, filters)

        # Export in specified format
        if format_type == "csv":
            exported_data = self._export_csv(data)
        elif format_type == "json":
            exported_data = self._export_json(data)
        elif format_type == "xml":
            exported_data = self._export_xml(data)
        else:
            exported_data = self._export_json(data)  # Default fallback

        # Apply compression if requested
        if compression:
            exported_data = self._compress_data(exported_data, compression)

        export_record = {
            "export_id": export_id,
            "data_source": data_source,
            "format": format_type,
            "compression": compression,
            "record_count": len(data) if isinstance(data, list) else 1,
            "exported_at": datetime.now().isoformat(),
            "file_size_bytes": len(exported_data),
            "filters_applied": filters or {}
        }

        self.export_history.append(export_record)

        return {
            "export_id": export_id,
            "data": exported_data,
            "metadata": export_record
        }

    def _get_data_for_export(self, data_source: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get data for export based on source and filters"""
        # Simulate data retrieval
        if data_source == "patients":
            return [
                {"patient_id": "P001", "name": "John Doe", "age": 45, "diagnosis": "Diabetes"},
                {"patient_id": "P002", "name": "Jane Smith", "age": 52, "diagnosis": "Hypertension"}
            ]
        elif data_source == "appointments":
            return [
                {"appointment_id": "A001", "patient_id": "P001", "date": "2024-01-15", "type": "Follow-up"},
                {"appointment_id": "A002", "patient_id": "P002", "date": "2024-01-16", "type": "Consultation"}
            ]
        else:
            return []

    def _export_csv(self, data: List[Dict[str, Any]]) -> bytes:
        """Export data as CSV"""
        if not data:
            return b""

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

        return output.getvalue().encode('utf-8')

    def _export_json(self, data: Any) -> bytes:
        """Export data as JSON"""
        return json.dumps(data, indent=2, default=str).encode('utf-8')

    def _export_xml(self, data: Any) -> bytes:
        """Export data as XML"""
        # Simplified XML export
        xml_content = "<?xml version='1.0' encoding='UTF-8'?>\n<data>\n"

        if isinstance(data, list):
            for item in data:
                xml_content += "  <record>\n"
                for key, value in item.items():
                    xml_content += f"    <{key}>{value}</{key}>\n"
                xml_content += "  </record>\n"
        elif isinstance(data, dict):
            xml_content += "  <record>\n"
            for key, value in data.items():
                xml_content += f"    <{key}>{value}</{key}>\n"
            xml_content += "  </record>\n"

        xml_content += "</data>"

        return xml_content.encode('utf-8')

    def _compress_data(self, data: bytes, compression_type: str) -> bytes:
        """Compress exported data"""
        # Simplified - in real implementation would use gzip, zip, etc.
        return data  # Return uncompressed for now

    def get_export_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get export history"""
        return self.export_history[-limit:]

    def schedule_recurring_export(self, export_config: Dict[str, Any]) -> str:
        """Schedule recurring data export"""
        schedule_id = f"schedule_{int(datetime.now().timestamp())}"

        schedule = {
            "schedule_id": schedule_id,
            "config": export_config,
            "frequency": export_config.get("frequency", "weekly"),
            "next_run": self._calculate_next_run(export_config.get("frequency", "weekly")),
            "status": "active",
            "created_at": datetime.now().isoformat()
        }

        return schedule_id

    def _calculate_next_run(self, frequency: str) -> str:
        """Calculate next run date based on frequency"""
        now = datetime.now()

        if frequency == "daily":
            next_run = now + timedelta(days=1)
        elif frequency == "weekly":
            next_run = now + timedelta(weeks=1)
        elif frequency == "monthly":
            next_run = now + timedelta(days=30)
        else:
            next_run = now + timedelta(weeks=1)  # Default to weekly

        return next_run.isoformat()
