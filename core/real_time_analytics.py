"""
Real-Time Analytics Engine for AI Personalized Medicine Platform
Provides real-time dashboards, metrics calculation, and streaming analytics
"""

import asyncio
import threading
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
import math

class RealTimeAnalyticsEngine:
    """Real-time analytics processing engine"""

    def __init__(self):
        self.metrics_store = defaultdict(lambda: deque(maxlen=1000))
        self.alerts_queue = asyncio.Queue()
        self.dashboard_subscribers = set()
        self.analytics_workers = {}
        self.is_running = False

    def start_analytics_engine(self):
        """Start the real-time analytics engine"""
        self.is_running = True

        # Start background analytics workers
        self._start_metric_collector()
        self._start_anomaly_detector()
        self._start_trend_analyzer()
        self._start_performance_monitor()

        print("🎯 Real-time analytics engine started")

    def stop_analytics_engine(self):
        """Stop the analytics engine"""
        self.is_running = False
        print("🛑 Real-time analytics engine stopped")

    def _start_metric_collector(self):
        """Start metric collection worker"""
        def collect_metrics():
            while self.is_running:
                try:
                    self._collect_system_metrics()
                    self._collect_health_metrics()
                    self._collect_performance_metrics()
                    time.sleep(5)  # Collect every 5 seconds
                except Exception as e:
                    print(f"Metrics collection error: {e}")

        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()
        self.analytics_workers["metric_collector"] = thread

    def _start_anomaly_detector(self):
        """Start anomaly detection worker"""
        def detect_anomalies():
            while self.is_running:
                try:
                    self._detect_metric_anomalies()
                    self._detect_health_anomalies()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    print(f"Anomaly detection error: {e}")

        thread = threading.Thread(target=detect_anomalies, daemon=True)
        thread.start()
        self.analytics_workers["anomaly_detector"] = thread

    def _start_trend_analyzer(self):
        """Start trend analysis worker"""
        def analyze_trends():
            while self.is_running:
                try:
                    self._analyze_metric_trends()
                    self._generate_predictive_insights()
                    time.sleep(300)  # Analyze every 5 minutes
                except Exception as e:
                    print(f"Trend analysis error: {e}")

        thread = threading.Thread(target=analyze_trends, daemon=True)
        thread.start()
        self.analytics_workers["trend_analyzer"] = thread

    def _start_performance_monitor(self):
        """Start performance monitoring worker"""
        def monitor_performance():
            while self.is_running:
                try:
                    self._monitor_api_performance()
                    self._monitor_model_performance()
                    self._generate_performance_reports()
                    time.sleep(60)  # Monitor every minute
                except Exception as e:
                    print(f"Performance monitoring error: {e}")

        thread = threading.Thread(target=monitor_performance, daemon=True)
        thread.start()
        self.analytics_workers["performance_monitor"] = thread

    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = datetime.now()

        # Simulate system metrics
        metrics = {
            "cpu_usage": random.uniform(10, 90),
            "memory_usage": random.uniform(20, 85),
            "disk_usage": random.uniform(15, 75),
            "network_io": random.uniform(100, 1000),
            "active_connections": random.randint(50, 200),
            "response_time": random.uniform(50, 500),
            "error_rate": random.uniform(0.001, 0.05),
            "throughput": random.uniform(1000, 5000)
        }

        for metric_name, value in metrics.items():
            self.metrics_store[f"system_{metric_name}"].append({
                "timestamp": timestamp,
                "value": value
            })

    def _collect_health_metrics(self):
        """Collect health-related metrics"""
        timestamp = datetime.now()

        # Simulate health metrics
        health_metrics = {
            "active_patients": random.randint(1000, 5000),
            "genomic_analyses_today": random.randint(50, 200),
            "drug_discovery_sessions": random.randint(10, 50),
            "clinical_trials_active": random.randint(20, 100),
            "alerts_generated": random.randint(5, 50),
            "data_processed_gb": random.uniform(10, 100),
            "api_calls_per_minute": random.randint(500, 2000),
            "model_predictions": random.randint(1000, 5000)
        }

        for metric_name, value in health_metrics.items():
            self.metrics_store[f"health_{metric_name}"].append({
                "timestamp": timestamp,
                "value": value
            })

    def _collect_performance_metrics(self):
        """Collect performance metrics"""
        timestamp = datetime.now()

        performance_metrics = {
            "model_accuracy": random.uniform(0.85, 0.98),
            "prediction_latency": random.uniform(10, 100),
            "data_processing_time": random.uniform(5, 50),
            "memory_efficiency": random.uniform(0.7, 0.95),
            "cache_hit_rate": random.uniform(0.8, 0.98),
            "uptime_percentage": random.uniform(0.99, 0.9999)
        }

        for metric_name, value in performance_metrics.items():
            self.metrics_store[f"performance_{metric_name}"].append({
                "timestamp": timestamp,
                "value": value
            })

    def _detect_metric_anomalies(self):
        """Detect anomalies in metrics"""
        for metric_name, data_points in self.metrics_store.items():
            if len(data_points) < 10:
                continue

            # Calculate recent statistics
            recent_values = [point["value"] for point in list(data_points)[-10:]]
            mean = statistics.mean(recent_values)
            std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0

            # Check for anomalies (3 sigma rule)
            latest_value = recent_values[-1]
            if std > 0 and abs(latest_value - mean) > 3 * std:
                anomaly = {
                    "metric": metric_name,
                    "timestamp": datetime.now(),
                    "value": latest_value,
                    "expected_range": (mean - 2*std, mean + 2*std),
                    "severity": "high" if abs(latest_value - mean) > 4 * std else "medium",
                    "description": f"Anomalous value detected in {metric_name}"
                }

                # Add to alerts queue
                asyncio.run_coroutine_threadsafe(
                    self.alerts_queue.put(anomaly),
                    asyncio.get_event_loop()
                )

    def _detect_health_anomalies(self):
        """Detect health-related anomalies"""
        # Check for unusual patient activity
        active_patients = [point["value"] for point in self.metrics_store["health_active_patients"]]
        if active_patients:
            latest = active_patients[-1]
            avg_24h = statistics.mean(active_patients[-288:]) if len(active_patients) >= 288 else latest

            if abs(latest - avg_24h) > avg_24h * 0.5:  # 50% deviation
                anomaly = {
                    "type": "patient_activity_spike",
                    "timestamp": datetime.now(),
                    "current_value": latest,
                    "baseline": avg_24h,
                    "severity": "high",
                    "description": "Unusual patient activity detected"
                }
                asyncio.run_coroutine_threadsafe(
                    self.alerts_queue.put(anomaly),
                    asyncio.get_event_loop()
                )

    def _analyze_metric_trends(self):
        """Analyze trends in metrics"""
        trend_analysis = {}

        for metric_name, data_points in self.metrics_store.items():
            if len(data_points) < 20:
                continue

            # Calculate trend over last hour (12 data points)
            recent = [point["value"] for point in list(data_points)[-12:]]
            if len(recent) >= 2:
                slope = self._calculate_trend_slope(recent)
                trend_analysis[metric_name] = {
                    "slope": slope,
                    "direction": "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable",
                    "magnitude": abs(slope),
                    "confidence": min(1.0, len(recent) / 12.0)
                }

        self.metrics_store["trend_analysis"] = trend_analysis

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate slope of linear trend"""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))
        y = values

        # Simple linear regression slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0

        return slope

    def _generate_predictive_insights(self):
        """Generate predictive insights"""
        # Predict system performance for next hour
        if len(self.metrics_store["system_cpu_usage"]) >= 60:  # 5 minutes of data
            cpu_values = [point["value"] for point in list(self.metrics_store["system_cpu_usage"])[-60:]]

            # Simple exponential smoothing prediction
            alpha = 0.3
            prediction = self._exponential_smoothing_prediction(cpu_values, alpha, 12)  # Next hour

            insight = {
                "type": "cpu_usage_prediction",
                "prediction": prediction,
                "timeframe": "next_hour",
                "confidence": 0.75,
                "recommendations": self._generate_cpu_recommendations(prediction)
            }

            self.metrics_store["predictive_insights"].append(insight)

    def _exponential_smoothing_prediction(self, values: List[float], alpha: float, steps: int) -> List[float]:
        """Generate exponential smoothing predictions"""
        if not values:
            return []

        predictions = []
        current = values[0]

        # Smooth existing data
        for value in values:
            current = alpha * value + (1 - alpha) * current

        # Generate predictions
        for _ in range(steps):
            predictions.append(current)
            current = current  # No trend assumption

        return predictions

    def _generate_cpu_recommendations(self, predictions: List[float]) -> List[str]:
        """Generate recommendations based on CPU predictions"""
        recommendations = []
        max_predicted = max(predictions) if predictions else 0

        if max_predicted > 80:
            recommendations.append("Scale up compute resources")
            recommendations.append("Optimize background processes")
        elif max_predicted > 60:
            recommendations.append("Monitor resource usage closely")
        else:
            recommendations.append("Resources are within normal range")

        return recommendations

    def _monitor_api_performance(self):
        """Monitor API performance"""
        response_times = [point["value"] for point in self.metrics_store["system_response_time"]]
        if response_times:
            avg_response_time = statistics.mean(response_times[-10:])  # Last 10 measurements

            if avg_response_time > 300:  # 300ms threshold
                alert = {
                    "type": "api_performance_degraded",
                    "metric": "response_time",
                    "value": avg_response_time,
                    "threshold": 300,
                    "severity": "medium"
                }
                asyncio.run_coroutine_threadsafe(
                    self.alerts_queue.put(alert),
                    asyncio.get_event_loop()
                )

    def _monitor_model_performance(self):
        """Monitor ML model performance"""
        accuracies = [point["value"] for point in self.metrics_store["performance_model_accuracy"]]
        if accuracies and len(accuracies) >= 10:
            current_accuracy = accuracies[-1]
            baseline_accuracy = statistics.mean(accuracies[-10:-1])

            if current_accuracy < baseline_accuracy * 0.95:  # 5% degradation
                alert = {
                    "type": "model_performance_degraded",
                    "current_accuracy": current_accuracy,
                    "baseline_accuracy": baseline_accuracy,
                    "degradation": (baseline_accuracy - current_accuracy) / baseline_accuracy,
                    "severity": "high"
                }
                asyncio.run_coroutine_threadsafe(
                    self.alerts_queue.put(alert),
                    asyncio.get_event_loop()
                )

    def _generate_performance_reports(self):
        """Generate comprehensive performance reports"""
        report = {
            "timestamp": datetime.now(),
            "period": "last_hour",
            "system_metrics": self._summarize_metrics("system_"),
            "health_metrics": self._summarize_metrics("health_"),
            "performance_metrics": self._summarize_metrics("performance_"),
            "alerts_summary": self._get_alerts_summary(),
            "recommendations": self._generate_system_recommendations()
        }

        self.metrics_store["performance_reports"].append(report)

    def _summarize_metrics(self, prefix: str) -> Dict[str, Any]:
        """Summarize metrics for a category"""
        summary = {}

        for metric_name, data_points in self.metrics_store.items():
            if not metric_name.startswith(prefix):
                continue

            if len(data_points) == 0:
                continue

            values = [point["value"] for point in data_points]
            summary[metric_name[len(prefix):]] = {
                "current": values[-1] if values else None,
                "average": statistics.mean(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "trend": self._calculate_trend_slope(values[-20:]) if len(values) >= 20 else 0
            }

        return summary

    def _get_alerts_summary(self) -> Dict[str, int]:
        """Get summary of alerts"""
        # This would normally query the alerts database
        return {
            "total_today": random.randint(10, 50),
            "critical": random.randint(0, 5),
            "high": random.randint(5, 15),
            "medium": random.randint(10, 25),
            "low": random.randint(20, 40)
        }

    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []

        # Check system health
        cpu_usage = self._get_latest_metric("system_cpu_usage")
        memory_usage = self._get_latest_metric("system_memory_usage")

        if cpu_usage and cpu_usage > 80:
            recommendations.append("Consider scaling compute resources")
        if memory_usage and memory_usage > 85:
            recommendations.append("Optimize memory usage or increase RAM")

        # Check model performance
        model_accuracy = self._get_latest_metric("performance_model_accuracy")
        if model_accuracy and model_accuracy < 0.90:
            recommendations.append("Retraining models may be beneficial")

        return recommendations

    def _get_latest_metric(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric"""
        data_points = self.metrics_store[metric_name]
        return data_points[-1]["value"] if data_points else None

    def get_realtime_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        return {
            "timestamp": datetime.now(),
            "system_health": self._get_system_health_status(),
            "key_metrics": self._get_key_metrics(),
            "active_alerts": self._get_active_alerts(),
            "performance_summary": self._get_performance_summary(),
            "predictive_insights": self._get_predictive_insights()
        }

    def _get_system_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        cpu = self._get_latest_metric("system_cpu_usage") or 0
        memory = self._get_latest_metric("system_memory_usage") or 0
        error_rate = self._get_latest_metric("system_error_rate") or 0

        # Calculate health score (0-100)
        health_score = 100 - (cpu * 0.4 + memory * 0.3 + error_rate * 1000 * 0.3)
        health_score = max(0, min(100, health_score))

        status = "healthy"
        if health_score < 70:
            status = "warning"
        if health_score < 50:
            status = "critical"

        return {
            "score": health_score,
            "status": status,
            "indicators": {
                "cpu": cpu,
                "memory": memory,
                "error_rate": error_rate
            }
        }

    def _get_key_metrics(self) -> Dict[str, Any]:
        """Get key business metrics"""
        return {
            "active_patients": self._get_latest_metric("health_active_patients") or 0,
            "genomic_analyses_today": self._get_latest_metric("health_genomic_analyses_today") or 0,
            "api_calls_per_minute": self._get_latest_metric("health_api_calls_per_minute") or 0,
            "model_accuracy": self._get_latest_metric("performance_model_accuracy") or 0,
            "data_processed_gb": self._get_latest_metric("health_data_processed_gb") or 0
        }

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts (simplified)"""
        # In a real implementation, this would query the alerts database
        return [
            {
                "id": f"alert_{i}",
                "type": random.choice(["system", "health", "performance"]),
                "severity": random.choice(["low", "medium", "high"]),
                "message": f"Sample alert {i}",
                "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 60))
            }
            for i in range(random.randint(0, 5))
        ]

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "uptime": self._get_latest_metric("performance_uptime_percentage") or 0,
            "response_time_avg": self._get_latest_metric("system_response_time") or 0,
            "throughput": self._get_latest_metric("system_throughput") or 0,
            "error_rate": self._get_latest_metric("system_error_rate") or 0
        }

    def _get_predictive_insights(self) -> List[Dict[str, Any]]:
        """Get predictive insights"""
        insights = list(self.metrics_store["predictive_insights"])[-5:]  # Last 5 insights
        return insights if insights else []

    async def subscribe_to_dashboard(self, callback: Callable):
        """Subscribe to real-time dashboard updates"""
        self.dashboard_subscribers.add(callback)

        # Send initial data
        await callback(self.get_realtime_dashboard_data())

    def unsubscribe_from_dashboard(self, callback: Callable):
        """Unsubscribe from dashboard updates"""
        self.dashboard_subscribers.discard(callback)


class PredictiveAnalyticsEngine:
    """Advanced predictive analytics for healthcare"""

    def __init__(self):
        self.models = {}
        self.prediction_cache = {}
        self.confidence_thresholds = {
            "disease_risk": 0.7,
            "treatment_response": 0.75,
            "adverse_events": 0.65,
            "hospitalization": 0.8
        }

    def initialize_predictive_models(self):
        """Initialize predictive models"""
        self.models = {
            "disease_risk": self._create_disease_risk_model(),
            "treatment_response": self._create_treatment_response_model(),
            "adverse_events": self._create_adverse_events_model(),
            "hospitalization": self._create_hospitalization_model(),
            "readmission": self._create_readmission_model(),
            "chronic_disease": self._create_chronic_disease_model()
        }

    def _create_disease_risk_model(self) -> Dict[str, Any]:
        """Create disease risk prediction model"""
        return {
            "algorithm": "XGBoost",
            "features": ["age", "bmi", "family_history", "lifestyle_score", "biomarkers"],
            "target_diseases": ["diabetes", "cardiovascular", "cancer", "alzheimer"],
            "accuracy": 0.87,
            "auc_roc": 0.89,
            "calibration_score": 0.85
        }

    def _create_treatment_response_model(self) -> Dict[str, Any]:
        """Create treatment response prediction model"""
        return {
            "algorithm": "NeuralNetwork",
            "features": ["genetics", "demographics", "medication_history", "biomarkers"],
            "response_classes": ["excellent", "good", "moderate", "poor", "adverse"],
            "accuracy": 0.82,
            "precision": 0.79,
            "recall": 0.81
        }

    def _create_adverse_events_model(self) -> Dict[str, Any]:
        """Create adverse events prediction model"""
        return {
            "algorithm": "RandomForest",
            "features": ["medications", "allergies", "age", "organ_function", "comorbidities"],
            "adverse_events": ["rash", "nausea", "neuropathy", "cardiotoxicity", "hepatotoxicity"],
            "accuracy": 0.91,
            "sensitivity": 0.88,
            "specificity": 0.93
        }

    def _create_hospitalization_model(self) -> Dict[str, Any]:
        """Create hospitalization risk prediction model"""
        return {
            "algorithm": "LogisticRegression",
            "features": ["vital_signs", "symptoms", "lab_results", "medication_adherence"],
            "prediction_window": "30_days",
            "accuracy": 0.85,
            "auc_roc": 0.87,
            "positive_predictive_value": 0.76
        }

    def _create_readmission_model(self) -> Dict[str, Any]:
        """Create hospital readmission prediction model"""
        return {
            "algorithm": "GradientBoosting",
            "features": ["discharge_diagnosis", "length_of_stay", "comorbidities", "social_support"],
            "prediction_window": "30_days",
            "accuracy": 0.83,
            "auc_roc": 0.86,
            "recall": 0.78
        }

    def _create_chronic_disease_model(self) -> Dict[str, Any]:
        """Create chronic disease progression model"""
        return {
            "algorithm": "LSTM",
            "features": ["longitudinal_vitals", "lab_trends", "symptoms", "treatments"],
            "diseases": ["diabetes", "hypertension", "copd", "heart_failure"],
            "accuracy": 0.79,
            "mean_absolute_error": 0.12,
            "r_squared": 0.81
        }

    def predict_disease_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict disease risk for a patient"""
        model = self.models.get("disease_risk")
        if not model:
            return {"error": "Disease risk model not initialized"}

        # Extract features
        features = self._extract_disease_risk_features(patient_data)

        # Simulate prediction (would use actual model)
        predictions = {}
        for disease in model["target_diseases"]:
            risk_score = self._calculate_disease_risk_score(features, disease)
            predictions[disease] = {
                "risk_score": risk_score,
                "risk_category": self._categorize_risk(risk_score),
                "confidence": random.uniform(0.7, 0.95),
                "time_horizon": "5_years",
                "preventive_measures": self._get_preventive_measures(disease, risk_score)
            }

        return {
            "patient_id": patient_data.get("patient_id"),
            "predictions": predictions,
            "overall_risk_score": statistics.mean([p["risk_score"] for p in predictions.values()]),
            "risk_factors": self._identify_risk_factors(features),
            "recommendations": self._generate_risk_recommendations(predictions),
            "model_info": model
        }

    def _extract_disease_risk_features(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for disease risk prediction"""
        return {
            "age": patient_data.get("age", 50),
            "bmi": patient_data.get("bmi", 25),
            "family_history_score": len(patient_data.get("family_history", [])) * 0.1,
            "lifestyle_score": patient_data.get("lifestyle_score", 0.5),
            "biomarker_score": self._calculate_biomarker_score(patient_data.get("biomarkers", {}))
        }

    def _calculate_biomarker_score(self, biomarkers: Dict[str, Any]) -> float:
        """Calculate biomarker risk score"""
        # Simplified biomarker scoring
        risk_markers = ["cholesterol", "glucose", "blood_pressure", "inflammation"]
        score = 0

        for marker in risk_markers:
            if marker in biomarkers:
                value = biomarkers[marker]
                # Normalize based on typical ranges
                if marker == "cholesterol" and value > 200:
                    score += 0.3
                elif marker == "glucose" and value > 100:
                    score += 0.25
                elif marker == "blood_pressure" and value > 140:
                    score += 0.25
                elif marker == "inflammation" and value > 5:
                    score += 0.2

        return min(1.0, score)

    def _calculate_disease_risk_score(self, features: Dict[str, float], disease: str) -> float:
        """Calculate risk score for a specific disease"""
        base_risk = {
            "diabetes": 0.15,
            "cardiovascular": 0.12,
            "cancer": 0.08,
            "alzheimer": 0.05
        }.get(disease, 0.1)

        # Adjust based on features
        risk_multiplier = 1.0
        risk_multiplier += features["age"] * 0.01  # Age factor
        risk_multiplier += (features["bmi"] - 20) * 0.02  # BMI factor
        risk_multiplier += features["family_history_score"]
        risk_multiplier += features["lifestyle_score"] * 0.5
        risk_multiplier += features["biomarker_score"]

        return min(1.0, base_risk * risk_multiplier)

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score < 0.2:
            return "low"
        elif risk_score < 0.4:
            return "moderate"
        elif risk_score < 0.7:
            return "high"
        else:
            return "very_high"

    def _get_preventive_measures(self, disease: str, risk_score: float) -> List[str]:
        """Get preventive measures for a disease"""
        base_measures = {
            "diabetes": ["Regular exercise", "Healthy diet", "Weight management"],
            "cardiovascular": ["Blood pressure monitoring", "Cholesterol management", "Quit smoking"],
            "cancer": ["Regular screenings", "Healthy lifestyle", "Genetic counseling"],
            "alzheimer": ["Cognitive exercises", "Social engagement", "Heart health"]
        }.get(disease, [])

        if risk_score > 0.5:
            base_measures.insert(0, "Consult healthcare provider immediately")

        return base_measures

    def _identify_risk_factors(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify key risk factors"""
        risk_factors = []

        if features["age"] > 60:
            risk_factors.append({"factor": "age", "impact": "high", "description": "Age > 60"})
        if features["bmi"] > 30:
            risk_factors.append({"factor": "bmi", "impact": "high", "description": "Obesity (BMI > 30)"})
        if features["family_history_score"] > 0.2:
            risk_factors.append({"factor": "family_history", "impact": "medium", "description": "Family history"})
        if features["biomarker_score"] > 0.5:
            risk_factors.append({"factor": "biomarkers", "impact": "high", "description": "Elevated biomarkers"})

        return risk_factors

    def _generate_risk_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate overall risk recommendations"""
        recommendations = []

        high_risk_diseases = [disease for disease, pred in predictions.items() if pred["risk_category"] in ["high", "very_high"]]

        if high_risk_diseases:
            recommendations.append(f"Immediate attention needed for: {', '.join(high_risk_diseases)}")
            recommendations.append("Schedule consultation with healthcare provider")
            recommendations.append("Consider advanced screening and monitoring")

        recommendations.append("Maintain healthy lifestyle and regular check-ups")
        recommendations.append("Keep track of family medical history")

        return recommendations

    def predict_treatment_response(self, patient_data: Dict[str, Any], treatment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Predict treatment response"""
        model = self.models.get("treatment_response")
        if not model:
            return {"error": "Treatment response model not initialized"}

        # Extract features
        features = self._extract_treatment_features(patient_data, treatment_plan)

        # Simulate prediction
        response_probabilities = self._calculate_response_probabilities(features)
        predicted_response = max(response_probabilities, key=response_probabilities.get)

        return {
            "patient_id": patient_data.get("patient_id"),
            "treatment": treatment_plan.get("treatment_name"),
            "predicted_response": predicted_response,
            "response_probabilities": response_probabilities,
            "confidence": response_probabilities[predicted_response],
            "expected_outcomes": self._get_expected_outcomes(predicted_response),
            "monitoring_recommendations": self._get_monitoring_recommendations(predicted_response),
            "alternative_options": self._suggest_alternatives(response_probabilities)
        }

    def _extract_treatment_features(self, patient_data: Dict[str, Any], treatment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for treatment response prediction"""
        return {
            "genetic_factors": self._analyze_genetic_factors(patient_data.get("genomic_data", {})),
            "age": patient_data.get("age", 50),
            "comorbidities": len(patient_data.get("comorbidities", [])),
            "medication_history": len(patient_data.get("medication_history", [])),
            "biomarker_profile": patient_data.get("biomarkers", {}),
            "treatment_type": treatment_plan.get("treatment_type", "unknown"),
            "dosage": treatment_plan.get("dosage", 1.0)
        }

    def _analyze_genetic_factors(self, genomic_data: Dict[str, Any]) -> float:
        """Analyze genetic factors for treatment response"""
        # Simplified genetic analysis
        genetic_score = 0.5  # Baseline

        if "variants" in genomic_data:
            # Look for known pharmacogenomic variants
            favorable_variants = ["CYP2D6_normal", "TPMT_normal"]
            unfavorable_variants = ["CYP2D6_poor", "TPMT_deficient"]

            variants = genomic_data["variants"]
            for variant in favorable_variants:
                if variant in variants:
                    genetic_score += 0.2

            for variant in unfavorable_variants:
                if variant in variants:
                    genetic_score -= 0.3

        return max(0, min(1, genetic_score))

    def _calculate_response_probabilities(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate response probabilities"""
        # Simplified probability calculation
        base_probabilities = {
            "excellent": 0.2,
            "good": 0.3,
            "moderate": 0.3,
            "poor": 0.15,
            "adverse": 0.05
        }

        # Adjust based on features
        genetic_factor = features["genetic_factors"]
        age_factor = max(0, 1 - (features["age"] - 30) * 0.01)  # Younger better
        comorbidity_penalty = features["comorbidities"] * 0.05

        adjusted_probabilities = {}
        for response, prob in base_probabilities.items():
            adjustment = genetic_factor * 0.3 + age_factor * 0.2 - comorbidity_penalty

            if response in ["excellent", "good"]:
                adjusted_probabilities[response] = prob * (1 + adjustment)
            elif response in ["poor", "adverse"]:
                adjusted_probabilities[response] = prob * (1 - adjustment)
            else:
                adjusted_probabilities[response] = prob

        # Normalize
        total = sum(adjusted_probabilities.values())
        return {k: v/total for k, v in adjusted_probabilities.items()}

    def _get_expected_outcomes(self, predicted_response: str) -> Dict[str, Any]:
        """Get expected outcomes for predicted response"""
        outcomes = {
            "excellent": {
                "efficacy": "90-100%",
                "side_effects": "minimal",
                "duration": "as_expected",
                "quality_of_life": "improved"
            },
            "good": {
                "efficacy": "75-89%",
                "side_effects": "mild",
                "duration": "as_expected",
                "quality_of_life": "stable"
            },
            "moderate": {
                "efficacy": "50-74%",
                "side_effects": "moderate",
                "duration": "may_extend",
                "quality_of_life": "variable"
            },
            "poor": {
                "efficacy": "<50%",
                "side_effects": "significant",
                "duration": "likely_extend",
                "quality_of_life": "declining"
            },
            "adverse": {
                "efficacy": "minimal",
                "side_effects": "severe",
                "duration": "discontinue",
                "quality_of_life": "severely_impacted"
            }
        }

        return outcomes.get(predicted_response, {})

    def _get_monitoring_recommendations(self, predicted_response: str) -> List[str]:
        """Get monitoring recommendations based on predicted response"""
        recommendations = {
            "excellent": ["Standard monitoring schedule", "Regular efficacy assessments"],
            "good": ["Standard monitoring", "Watch for side effects"],
            "moderate": ["Increased monitoring frequency", "Early intervention planning"],
            "poor": ["Intensive monitoring", "Consider treatment modification"],
            "adverse": ["Immediate discontinuation", "Emergency monitoring", "Supportive care"]
        }

        return recommendations.get(predicted_response, ["Regular monitoring"])

    def _suggest_alternatives(self, response_probabilities: Dict[str, float]) -> List[str]:
        """Suggest alternative treatments if response is poor"""
        poor_responses = ["poor", "adverse"]
        poor_probability = sum(response_probabilities.get(r, 0) for r in poor_responses)

        if poor_probability > 0.3:
            return [
                "Consider alternative medication classes",
                "Evaluate combination therapies",
                "Assess for drug interactions",
                "Consult pharmacogenomics specialist"
            ]

        return []

    def predict_adverse_events(self, patient_data: Dict[str, Any], medications: List[str]) -> Dict[str, Any]:
        """Predict adverse events for medication regimen"""
        model = self.models.get("adverse_events")
        if not model:
            return {"error": "Adverse events model not initialized"}

        # Analyze each medication
        adverse_predictions = {}
        for medication in medications:
            risk_score = self._calculate_adverse_risk(medication, patient_data)
            adverse_predictions[medication] = {
                "overall_risk": risk_score,
                "specific_events": self._predict_specific_events(medication, patient_data),
                "risk_factors": self._identify_adverse_risk_factors(medication, patient_data),
                "preventive_measures": self._get_adverse_prevention(medication, risk_score)
            }

        # Calculate regimen-level risks
        regimen_risk = self._calculate_regimen_risk(adverse_predictions)

        return {
            "patient_id": patient_data.get("patient_id"),
            "medications_analyzed": medications,
            "adverse_predictions": adverse_predictions,
            "regimen_risk_score": regimen_risk,
            "regimen_risk_level": self._categorize_adverse_risk(regimen_risk),
            "monitoring_recommendations": self._get_adverse_monitoring(regimen_risk),
            "interactions": self._analyze_drug_interactions(medications, patient_data)
        }

    def _calculate_adverse_risk(self, medication: str, patient_data: Dict[str, Any]) -> float:
        """Calculate adverse event risk for a medication"""
        # Base risk by medication type
        base_risks = {
            "statin": 0.15,
            "antibiotic": 0.12,
            "chemotherapy": 0.35,
            "antidepressant": 0.18,
            "anticoagulant": 0.25
        }

        # Determine medication category
        med_category = "unknown"
        for category in base_risks.keys():
            if category.lower() in medication.lower():
                med_category = category
                break

        base_risk = base_risks.get(med_category, 0.1)

        # Adjust for patient factors
        age_factor = (patient_data.get("age", 50) - 50) * 0.002
        comorbidity_factor = len(patient_data.get("comorbidities", [])) * 0.05
        allergy_factor = 0.2 if self._has_relevant_allergy(medication, patient_data) else 0

        return min(1.0, base_risk + age_factor + comorbidity_factor + allergy_factor)

    def _has_relevant_allergy(self, medication: str, patient_data: Dict[str, Any]) -> bool:
        """Check if patient has relevant allergies"""
        allergies = patient_data.get("allergies", [])
        medication_lower = medication.lower()

        for allergy in allergies:
            allergy_lower = allergy.lower()
            if allergy_lower in medication_lower or medication_lower in allergy_lower:
                return True

        return False

    def _predict_specific_events(self, medication: str, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict specific adverse events"""
        # Medication-specific adverse events
        event_mappings = {
            "statin": ["muscle_pain", "liver_enzyme_elevation", "diabetes_risk"],
            "antibiotic": ["nausea", "diarrhea", "rash", "c_difficile"],
            "chemotherapy": ["nausea", "fatigue", "neutropenia", "cardiotoxicity"],
            "antidepressant": ["nausea", "insomnia", "sexual_dysfunction", "serotonin_syndrome"],
            "anticoagulant": ["bleeding", "bruising", "thrombosis"]
        }

        med_category = "unknown"
        for category in event_mappings.keys():
            if category.lower() in medication.lower():
                med_category = category
                break

        events = event_mappings.get(med_category, ["general_side_effects"])

        predicted_events = []
        for event in events:
            risk_score = random.uniform(0.05, 0.8)
            predicted_events.append({
                "event": event,
                "risk_score": risk_score,
                "severity": "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low",
                "onset_time": f"{random.randint(1, 30)} days"
            })

        return sorted(predicted_events, key=lambda x: x["risk_score"], reverse=True)

    def _identify_adverse_risk_factors(self, medication: str, patient_data: Dict[str, Any]) -> List[str]:
        """Identify risk factors for adverse events"""
        risk_factors = []

        if patient_data.get("age", 50) > 65:
            risk_factors.append("Advanced age")
        if len(patient_data.get("comorbidities", [])) > 2:
            risk_factors.append("Multiple comorbidities")
        if self._has_relevant_allergy(medication, patient_data):
            risk_factors.append("Known allergy")
        if len(patient_data.get("current_medications", [])) > 3:
            risk_factors.append("Polypharmacy")

        return risk_factors

    def _get_adverse_prevention(self, medication: str, risk_score: float) -> List[str]:
        """Get preventive measures for adverse events"""
        prevention = ["Regular monitoring", "Report unusual symptoms promptly"]

        if risk_score > 0.5:
            prevention.insert(0, "Close monitoring required")
        if risk_score > 0.7:
            prevention.insert(0, "Consider alternative medication")

        return prevention

    def _calculate_regimen_risk(self, adverse_predictions: Dict[str, Any]) -> float:
        """Calculate overall regimen risk"""
        individual_risks = [pred["overall_risk"] for pred in adverse_predictions.values()]
        if not individual_risks:
            return 0

        # Combined risk (not simply additive)
        max_risk = max(individual_risks)
        avg_risk = statistics.mean(individual_risks)
        regimen_risk = (max_risk * 0.6) + (avg_risk * 0.4)

        return min(1.0, regimen_risk)

    def _categorize_adverse_risk(self, risk_score: float) -> str:
        """Categorize adverse event risk"""
        if risk_score < 0.2:
            return "low"
        elif risk_score < 0.4:
            return "moderate"
        elif risk_score < 0.7:
            return "high"
        else:
            return "very_high"

    def _get_adverse_monitoring(self, regimen_risk: float) -> List[str]:
        """Get monitoring recommendations for adverse events"""
        if regimen_risk < 0.2:
            return ["Standard monitoring"]
        elif regimen_risk < 0.4:
            return ["Regular lab monitoring", "Symptom assessment"]
        elif regimen_risk < 0.7:
            return ["Frequent monitoring", "Baseline lab tests", "Patient education"]
        else:
            return ["Intensive monitoring", "Daily assessment", "Emergency planning"]

    def _analyze_drug_interactions(self, medications: List[str], patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential drug interactions"""
        interactions = []

        # Simple interaction detection (would use actual drug interaction database)
        known_interactions = [
            (["warfarin", "aspirin"], "Increased bleeding risk"),
            (["statin", "fibrate"], "Muscle toxicity risk"),
            (["ace_inhibitor", "potassium"], "Hyperkalemia risk")
        ]

        med_lower = [med.lower() for med in medications]

        for interaction_pair, risk in known_interactions:
            if all(any(interact in med for med in med_lower) for interact in interaction_pair):
                interactions.append({
                    "medications": interaction_pair,
                    "risk": risk,
                    "severity": "high",
                    "recommendation": "Monitor closely, consider dose adjustment"
                })

        return interactions


class RiskStratificationEngine:
    """Advanced risk stratification for healthcare populations"""

    def __init__(self):
        self.stratification_models = {}
        self.risk_thresholds = {
            "preventive_care": 0.3,
            "chronic_disease": 0.4,
            "acute_care": 0.6,
            "intensive_care": 0.8
        }

    def initialize_stratification_models(self):
        """Initialize risk stratification models"""
        self.stratification_models = {
            "population_health": self._create_population_model(),
            "individual_risk": self._create_individual_model(),
            "cohort_analysis": self._create_cohort_model(),
            "temporal_trends": self._create_temporal_model()
        }

    def _create_population_model(self) -> Dict[str, Any]:
        """Create population-level risk stratification model"""
        return {
            "algorithm": "Clustering+KMeans",
            "features": ["demographics", "health_indicators", "social_determinants", "utilization"],
            "risk_levels": ["low", "moderate", "high", "very_high"],
            "population_segments": ["healthy", "at_risk", "chronic", "complex_care"],
            "accuracy": 0.85,
            "coverage": 0.92
        }

    def _create_individual_model(self) -> Dict[str, Any]:
        """Create individual risk stratification model"""
        return {
            "algorithm": "XGBoost",
            "features": ["clinical_history", "biomarkers", "genetics", "lifestyle", "social_factors"],
            "prediction_horizon": ["1_month", "6_months", "1_year", "5_years"],
            "calibration_score": 0.87,
            "discrimination_auc": 0.89
        }

    def _create_cohort_model(self) -> Dict[str, Any]:
        """Create cohort-based stratification model"""
        return {
            "algorithm": "SurvivalAnalysis",
            "cohort_types": ["age_group", "condition", "intervention", "geographic"],
            "outcomes": ["mortality", "hospitalization", "readmission", "complications"],
            "adjustment_variables": ["comorbidities", "severity", "treatment"],
            "c_index": 0.82
        }

    def _create_temporal_model(self) -> Dict[str, Any]:
        """Create temporal trend analysis model"""
        return {
            "algorithm": "TimeSeries+ARIMA",
            "trend_types": ["improving", "stable", "declining", "volatile"],
            "prediction_intervals": ["short_term", "medium_term", "long_term"],
            "seasonal_adjustment": True,
            "accuracy": 0.78
        }

    def stratify_population_risk(self, population_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stratify risk across a population"""
        if not population_data:
            return {"error": "No population data provided"}

        # Analyze population characteristics
        population_stats = self._analyze_population_characteristics(population_data)

        # Perform risk stratification
        risk_strata = self._perform_risk_stratification(population_data)

        # Generate insights and recommendations
        insights = self._generate_population_insights(risk_strata, population_stats)

        return {
            "population_size": len(population_data),
            "analysis_timestamp": datetime.now(),
            "population_characteristics": population_stats,
            "risk_stratification": risk_strata,
            "insights": insights,
            "resource_allocation": self._calculate_resource_needs(risk_strata),
            "intervention_priorities": self._prioritize_interventions(risk_strata)
        }

    def _analyze_population_characteristics(self, population_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze population characteristics"""
        ages = [p.get("age", 50) for p in population_data]
        conditions = [p.get("conditions", []) for p in population_data]
        flattened_conditions = [cond for sublist in conditions for cond in sublist]

        return {
            "age_distribution": {
                "mean": statistics.mean(ages),
                "median": statistics.median(ages),
                "range": (min(ages), max(ages))
            },
            "condition_prevalence": dict(Counter(flattened_conditions)),
            "demographics": self._analyze_demographics(population_data),
            "health_utilization": self._analyze_utilization(population_data)
        }

    def _analyze_demographics(self, population_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze demographic characteristics"""
        genders = [p.get("gender", "unknown") for p in population_data]
        ethnicities = [p.get("ethnicity", "unknown") for p in population_data]

        return {
            "gender_distribution": dict(Counter(genders)),
            "ethnicity_distribution": dict(Counter(ethnicities)),
            "geographic_distribution": self._analyze_geographic_distribution(population_data)
        }

    def _analyze_geographic_distribution(self, population_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze geographic distribution"""
        locations = [p.get("location", "unknown") for p in population_data]
        return dict(Counter(locations))

    def _analyze_utilization(self, population_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze healthcare utilization patterns"""
        utilization = [p.get("utilization_score", 0.5) for p in population_data]

        return {
            "mean_utilization": statistics.mean(utilization),
            "high_utilizers": len([u for u in utilization if u > 0.8]),
            "low_utilizers": len([u for u in utilization if u < 0.2])
        }

    def _perform_risk_stratification(self, population_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform risk stratification on population"""
        strata = {"low": [], "moderate": [], "high": [], "very_high": []}

        for patient in population_data:
            risk_score = self._calculate_individual_risk_score(patient)
            risk_level = self._assign_risk_level(risk_score)

            strata[risk_level].append({
                "patient_id": patient.get("patient_id"),
                "risk_score": risk_score,
                "risk_factors": self._identify_key_risk_factors(patient)
            })

        # Calculate stratum statistics
        stratum_stats = {}
        for level, patients in strata.items():
            if patients:
                scores = [p["risk_score"] for p in patients]
                stratum_stats[level] = {
                    "count": len(patients),
                    "percentage": len(patients) / len(population_data) * 100,
                    "avg_risk_score": statistics.mean(scores),
                    "risk_range": (min(scores), max(scores))
                }
            else:
                stratum_stats[level] = {"count": 0, "percentage": 0, "avg_risk_score": 0, "risk_range": (0, 0)}

        return {
            "strata": strata,
            "statistics": stratum_stats,
            "stratification_method": "multidimensional_risk_scoring"
        }

    def _calculate_individual_risk_score(self, patient: Dict[str, Any]) -> float:
        """Calculate individual risk score"""
        # Multi-dimensional risk scoring
        age_risk = min(1.0, patient.get("age", 50) / 100)
        condition_risk = min(1.0, len(patient.get("conditions", [])) * 0.2)
        utilization_risk = patient.get("utilization_score", 0.5)
        biomarker_risk = self._calculate_biomarker_risk(patient.get("biomarkers", {}))

        # Weighted combination
        risk_score = (
            age_risk * 0.2 +
            condition_risk * 0.3 +
            utilization_risk * 0.25 +
            biomarker_risk * 0.25
        )

        return min(1.0, risk_score)

    def _calculate_biomarker_risk(self, biomarkers: Dict[str, Any]) -> float:
        """Calculate biomarker-based risk"""
        risk_score = 0
        risk_indicators = {
            "glucose": lambda x: min(1.0, (x - 70) / 130) if x > 70 else 0,
            "cholesterol": lambda x: min(1.0, (x - 150) / 150) if x > 150 else 0,
            "blood_pressure": lambda x: min(1.0, (x - 90) / 90) if x > 90 else 0,
            "bmi": lambda x: min(1.0, (x - 18.5) / 20) if x > 18.5 else 0
        }

        for biomarker, risk_func in risk_indicators.items():
            if biomarker in biomarkers:
                risk_score += risk_func(biomarkers[biomarker]) * 0.25

        return risk_score

    def _assign_risk_level(self, risk_score: float) -> str:
        """Assign risk level based on score"""
        if risk_score < 0.25:
            return "low"
        elif risk_score < 0.5:
            return "moderate"
        elif risk_score < 0.75:
            return "high"
        else:
            return "very_high"

    def _identify_key_risk_factors(self, patient: Dict[str, Any]) -> List[str]:
        """Identify key risk factors for a patient"""
        risk_factors = []

        if patient.get("age", 50) > 65:
            risk_factors.append("advanced_age")
        if len(patient.get("conditions", [])) > 2:
            risk_factors.append("multiple_conditions")
        if patient.get("utilization_score", 0.5) > 0.8:
            risk_factors.append("high_utilization")
        if self._calculate_biomarker_risk(patient.get("biomarkers", {})) > 0.6:
            risk_factors.append("elevated_biomarkers")

        return risk_factors

    def _generate_population_insights(self, risk_strata: Dict[str, Any], population_stats: Dict[str, Any]) -> List[str]:
        """Generate population-level insights"""
        insights = []

        stats = risk_strata["statistics"]

        # Risk distribution insights
        high_risk_pct = stats["high"]["percentage"] + stats["very_high"]["percentage"]
        if high_risk_pct > 30:
            insights.append(f"High-risk population segment is {high_risk_pct:.1f}% - requires intensive intervention")
        elif high_risk_pct < 10:
            insights.append(f"Low high-risk population ({high_risk_pct:.1f}%) - focus on prevention")

        # Age-related insights
        age_stats = population_stats["age_distribution"]
        if age_stats["mean"] > 60:
            insights.append("Aging population - prioritize geriatric care and prevention")
        elif age_stats["mean"] < 40:
            insights.append("Younger population - focus on lifestyle interventions and education")

        # Condition prevalence insights
        condition_prev = population_stats["condition_prevalence"]
        if condition_prev:
            top_condition = max(condition_prev, key=condition_prev.get)
            if condition_prev[top_condition] > len(population_stats) * 0.2:
                insights.append(f"High prevalence of {top_condition} - consider population-level interventions")

        return insights

    def _calculate_resource_needs(self, risk_strata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource allocation needs"""
        stats = risk_strata["statistics"]

        # Resource allocation based on risk levels
        resource_weights = {
            "low": 0.5,
            "moderate": 1.0,
            "high": 2.0,
            "very_high": 4.0
        }

        total_weighted_risk = sum(
            stats[level]["count"] * resource_weights[level]
            for level in resource_weights.keys()
        )

        resource_allocation = {}
        for level, weight in resource_weights.items():
            resource_allocation[level] = {
                "patients": stats[level]["count"],
                "resource_multiplier": weight,
                "allocated_resources": (stats[level]["count"] * weight) / total_weighted_risk * 100
            }

        return resource_allocation

    def _prioritize_interventions(self, risk_strata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize interventions based on risk stratification"""
        priorities = []

        stats = risk_strata["statistics"]

        # High-risk interventions
        if stats["very_high"]["count"] > 0:
            priorities.append({
                "priority": "critical",
                "target_group": "very_high_risk",
                "intervention_type": "intensive_case_management",
                "rationale": f"{stats['very_high']['count']} patients at very high risk",
                "timeline": "immediate"
            })

        # High-risk interventions
        if stats["high"]["count"] > 10:
            priorities.append({
                "priority": "high",
                "target_group": "high_risk",
                "intervention_type": "coordinated_care",
                "rationale": f"{stats['high']['count']} patients need coordinated care",
                "timeline": "within_1_week"
            })

        # Moderate-risk prevention
        if stats["moderate"]["percentage"] > 40:
            priorities.append({
                "priority": "medium",
                "target_group": "moderate_risk",
                "intervention_type": "preventive_programs",
                "rationale": f"{stats['moderate']['percentage']:.1f}% of population needs prevention focus",
                "timeline": "within_1_month"
            })

        return priorities
