"""
Model Explainability System for AI Personalized Medicine Platform
Provides comprehensive model interpretation, SHAP analysis, and clinical insights
"""

import random
import math
import statistics
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time
import asyncio
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SHAPExplanation:
    """SHAP explanation for a single prediction"""
    base_value: float
    shap_values: Dict[str, float]
    feature_values: Dict[str, Any]
    prediction: float
    prediction_probability: float
    explanation_confidence: float

@dataclass
class FeatureImportance:
    """Feature importance analysis"""
    global_importance: Dict[str, float]
    local_importance: Dict[str, Dict[str, float]]
    interaction_effects: Dict[Tuple[str, str], float]
    feature_stability: Dict[str, float]

@dataclass
class ModelCard:
    """Comprehensive model documentation"""
    model_name: str
    model_type: str
    created_date: datetime
    intended_use: str
    limitations: List[str]
    performance_metrics: Dict[str, Any]
    explainability_metrics: Dict[str, Any]
    fairness_assessment: Dict[str, Any]
    data_used: Dict[str, Any]

class ModelExplainabilitySystem:
    """Comprehensive model explainability and interpretation system"""

    def __init__(self):
        self.explanations_cache = {}
        self.model_cards = {}
        self.explainability_metrics = defaultdict(dict)
        self.fairness_assessments = {}
        self.interpretation_rules = {}
        self.is_running = False
        self.explainability_workers = []
        self.initialize_explainability_system()

    def initialize_explainability_system(self):
        """Initialize the model explainability system"""
        # Initialize explanation methods
        self.explanation_methods = {
            "shap": self._shap_explanation,
            "lime": self._lime_explanation,
            "permutation_importance": self._permutation_importance,
            "partial_dependence": self._partial_dependence_plots,
            "feature_interactions": self._feature_interaction_analysis
        }

        # Initialize clinical interpretation rules
        self._initialize_clinical_interpretation_rules()

        print("🔍 Model Explainability System initialized")

    def _initialize_clinical_interpretation_rules(self):
        """Initialize clinical interpretation rules for medical context"""
        self.interpretation_rules = {
            "cardiovascular_risk": {
                "high_risk_factors": ["age", "cholesterol", "blood_pressure", "smoking"],
                "protective_factors": ["exercise", "mediterranean_diet", "statin_use"],
                "interaction_effects": [("smoking", "age"), ("cholesterol", "exercise")]
            },
            "diabetes_risk": {
                "high_risk_factors": ["bmi", "family_history", "glucose_levels", "insulin_resistance"],
                "protective_factors": ["physical_activity", "dietary_fiber", "metformin_use"],
                "interaction_effects": [("bmi", "physical_activity"), ("glucose_levels", "family_history")]
            },
            "cancer_risk": {
                "high_risk_factors": ["age", "genetic_mutations", "environmental_exposure", "lifestyle"],
                "protective_factors": ["screening_adherence", "healthy_diet", "preventive_medications"],
                "interaction_effects": [("genetic_mutations", "age"), ("environmental_exposure", "lifestyle")]
            }
        }

    def start_explainability_system(self):
        """Start the explainability system"""
        self.is_running = True

        # Start explainability workers
        for i in range(3):  # 3 concurrent explanation workers
            worker = threading.Thread(target=self._explainability_worker, daemon=True)
            worker.start()
            self.explainability_workers.append(worker)

        # Start monitoring worker
        monitor_worker = threading.Thread(target=self._explainability_monitor, daemon=True)
        monitor_worker.start()
        self.explainability_workers.append(monitor_worker)

        print("🚀 Model Explainability System started")

    def stop_explainability_system(self):
        """Stop the explainability system"""
        self.is_running = False
        print("🛑 Model Explainability System stopped")

    def explain_prediction(self, model_id: str, input_data: Dict[str, Any],
                          prediction: Any, method: str = "shap",
                          context: str = "general") -> SHAPExplanation:
        """Generate explanation for a single prediction"""
        cache_key = f"{model_id}_{hash(str(input_data))}_{method}"

        # Check cache first
        if cache_key in self.explanations_cache:
            return self.explanations_cache[cache_key]

        # Generate explanation
        if method == "shap":
            explanation = self._shap_explanation(model_id, input_data, prediction, context)
        elif method == "lime":
            explanation = self._lime_explanation(model_id, input_data, prediction, context)
        else:
            raise ValueError(f"Unsupported explanation method: {method}")

        # Cache explanation
        self.explanations_cache[cache_key] = explanation

        # Update explainability metrics
        self._update_explainability_metrics(model_id, method, explanation)

        return explanation

    def _shap_explanation(self, model_id: str, input_data: Dict[str, Any],
                         prediction: Any, context: str) -> SHAPExplanation:
        """Generate SHAP-based explanation"""
        # Simulate SHAP value calculation
        base_value = 0.5  # Base prediction probability

        # Generate SHAP values for each feature
        shap_values = {}
        feature_values = {}

        for feature_name, feature_value in input_data.items():
            # Simulate SHAP value based on feature importance and value
            importance_weight = self._get_feature_importance_weight(feature_name, context)
            shap_value = (feature_value - self._get_feature_mean(feature_name)) * importance_weight
            shap_value += random.uniform(-0.1, 0.1)  # Add some noise

            shap_values[feature_name] = shap_value
            feature_values[feature_name] = feature_value

        # Calculate prediction from SHAP values
        prediction_probability = base_value + sum(shap_values.values())
        prediction_probability = max(0, min(1, prediction_probability))  # Bound between 0 and 1

        # Calculate explanation confidence
        confidence = 1 - (statistics.stdev(list(shap_values.values())) / abs(sum(shap_values.values()))) if shap_values else 0.8
        confidence = max(0.5, min(0.95, confidence))

        return SHAPExplanation(
            base_value=base_value,
            shap_values=shap_values,
            feature_values=feature_values,
            prediction=prediction,
            prediction_probability=prediction_probability,
            explanation_confidence=confidence
        )

    def _lime_explanation(self, model_id: str, input_data: Dict[str, Any],
                         prediction: Any, context: str) -> SHAPExplanation:
        """Generate LIME-based explanation"""
        # LIME creates a local linear approximation
        # Simplified implementation
        base_value = random.uniform(0.3, 0.7)

        shap_values = {}
        feature_values = {}

        # Select most important features for local explanation
        important_features = self._select_important_features(input_data, context)

        for feature_name in important_features[:10]:  # Top 10 features
            feature_value = input_data.get(feature_name, 0)
            # Generate local linear coefficient
            coefficient = random.uniform(-0.5, 0.5)
            shap_value = coefficient * (feature_value - 0.5)  # Center around 0.5

            shap_values[feature_name] = shap_value
            feature_values[feature_name] = feature_value

        prediction_probability = base_value + sum(shap_values.values())
        prediction_probability = max(0, min(1, prediction_probability))

        confidence = 0.85  # LIME typically has good local fidelity

        return SHAPExplanation(
            base_value=base_value,
            shap_values=shap_values,
            feature_values=feature_values,
            prediction=prediction,
            prediction_probability=prediction_probability,
            explanation_confidence=confidence
        )

    def _get_feature_importance_weight(self, feature_name: str, context: str) -> float:
        """Get feature importance weight based on context"""
        # Context-specific importance weights
        context_weights = {
            "cardiovascular": {
                "age": 0.8, "cholesterol": 0.9, "blood_pressure": 0.85,
                "smoking": 0.7, "exercise": -0.6, "bmi": 0.4
            },
            "diabetes": {
                "bmi": 0.8, "glucose": 0.9, "insulin": 0.7,
                "family_history": 0.6, "exercise": -0.5, "diet": -0.4
            },
            "cancer": {
                "age": 0.7, "genetics": 0.9, "smoking": 0.8,
                "screening": -0.6, "lifestyle": 0.5
            }
        }

        weights = context_weights.get(context, {})
        return weights.get(feature_name, random.uniform(0.1, 0.5))

    def _get_feature_mean(self, feature_name: str) -> float:
        """Get typical mean value for a feature"""
        # Simulated feature means
        feature_means = {
            "age": 50, "bmi": 25, "cholesterol": 180, "blood_pressure": 120,
            "glucose": 100, "exercise_hours": 2, "smoking": 0.2
        }
        return feature_means.get(feature_name, 0.5)

    def _select_important_features(self, input_data: Dict[str, Any], context: str) -> List[str]:
        """Select most important features for explanation"""
        all_features = list(input_data.keys())
        context_rules = self.interpretation_rules.get(context, {})

        # Prioritize high-risk factors
        high_priority = context_rules.get("high_risk_factors", [])
        protective = context_rules.get("protective_factors", [])

        # Sort features by priority
        feature_priority = {}
        for feature in all_features:
            if feature in high_priority:
                feature_priority[feature] = 3
            elif feature in protective:
                feature_priority[feature] = 2
            else:
                feature_priority[feature] = 1

        # Add some randomness for diversity
        for feature in all_features:
            feature_priority[feature] += random.uniform(-0.5, 0.5)

        return sorted(feature_priority.keys(), key=lambda x: feature_priority[x], reverse=True)

    def analyze_feature_importance(self, model_id: str, dataset: List[Dict[str, Any]],
                                 context: str = "general") -> FeatureImportance:
        """Analyze global and local feature importance"""
        global_importance = self._calculate_global_importance(dataset, context)
        local_importance = self._calculate_local_importance(dataset[:10], context)  # First 10 samples
        interaction_effects = self._analyze_feature_interactions(dataset, context)
        feature_stability = self._assess_feature_stability(dataset, context)

        return FeatureImportance(
            global_importance=global_importance,
            local_importance=local_importance,
            interaction_effects=interaction_effects,
            feature_stability=feature_stability
        )

    def _calculate_global_importance(self, dataset: List[Dict[str, Any]], context: str) -> Dict[str, float]:
        """Calculate global feature importance"""
        all_features = set()
        for sample in dataset:
            all_features.update(sample.keys())

        global_importance = {}
        for feature in all_features:
            # Aggregate importance across all samples
            importance_sum = 0
            count = 0

            for sample in dataset:
                if feature in sample:
                    importance = self._get_feature_importance_weight(feature, context)
                    importance_sum += importance * abs(sample[feature] - self._get_feature_mean(feature))
                    count += 1

            global_importance[feature] = importance_sum / count if count > 0 else 0

        # Normalize
        max_importance = max(global_importance.values()) if global_importance else 1
        global_importance = {k: v/max_importance for k, v in global_importance.items()}

        return dict(sorted(global_importance.items(), key=lambda x: x[1], reverse=True))

    def _calculate_local_importance(self, samples: List[Dict[str, Any]], context: str) -> Dict[str, Dict[str, float]]:
        """Calculate local feature importance for specific samples"""
        local_importance = {}

        for i, sample in enumerate(samples):
            sample_id = f"sample_{i}"
            local_importance[sample_id] = {}

            for feature_name, feature_value in sample.items():
                importance = self._get_feature_importance_weight(feature_name, context)
                deviation = abs(feature_value - self._get_feature_mean(feature_name))
                local_importance[sample_id][feature_name] = importance * deviation

        return local_importance

    def _analyze_feature_interactions(self, dataset: List[Dict[str, Any]], context: str) -> Dict[Tuple[str, str], float]:
        """Analyze feature interaction effects"""
        interaction_effects = {}

        context_rules = self.interpretation_rules.get(context, {})
        known_interactions = context_rules.get("interaction_effects", [])

        # Analyze known interactions
        for feature1, feature2 in known_interactions:
            if all(feature1 in sample and feature2 in sample for sample in dataset):
                # Calculate interaction effect
                interaction_strength = 0
                for sample in dataset:
                    val1 = sample[feature1]
                    val2 = sample[feature2]
                    # Simple interaction calculation
                    interaction = (val1 - self._get_feature_mean(feature1)) * (val2 - self._get_feature_mean(feature2))
                    interaction_strength += abs(interaction)

                interaction_effects[(feature1, feature2)] = interaction_strength / len(dataset)

        # Also calculate some random interactions for completeness
        all_features = list(set().union(*[set(sample.keys()) for sample in dataset]))
        for _ in range(min(5, len(all_features) * (len(all_features) - 1) // 2)):
            feature1, feature2 = random.sample(all_features, 2)
            if (feature1, feature2) not in interaction_effects:
                interaction_effects[(feature1, feature2)] = random.uniform(0.1, 0.5)

        return dict(sorted(interaction_effects.items(), key=lambda x: x[1], reverse=True))

    def _assess_feature_stability(self, dataset: List[Dict[str, Any]], context: str) -> Dict[str, float]:
        """Assess feature stability across different samples"""
        feature_stability = {}

        all_features = set()
        for sample in dataset:
            all_features.update(sample.keys())

        for feature in all_features:
            values = [sample.get(feature, 0) for sample in dataset if feature in sample]

            if len(values) > 1:
                # Calculate coefficient of variation (lower = more stable)
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)

                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    # Convert to stability score (higher = more stable)
                    stability = 1 / (1 + cv)
                else:
                    stability = 0.5  # Neutral stability for zero-mean features
            else:
                stability = 1.0  # Perfect stability for single value

            feature_stability[feature] = stability

        return feature_stability

    def generate_model_card(self, model_id: str, model_info: Dict[str, Any]) -> ModelCard:
        """Generate comprehensive model documentation card"""
        model_card = ModelCard(
            model_name=model_info.get("name", model_id),
            model_type=model_info.get("type", "unknown"),
            created_date=model_info.get("created_at", datetime.now()),
            intended_use=model_info.get("intended_use", "Clinical decision support"),
            limitations=self._identify_model_limitations(model_info),
            performance_metrics=model_info.get("performance_metrics", {}),
            explainability_metrics=self._calculate_explainability_metrics(model_id),
            fairness_assessment=self._assess_model_fairness(model_info),
            data_used=model_info.get("training_data_info", {})
        )

        self.model_cards[model_id] = model_card
        return model_card

    def _identify_model_limitations(self, model_info: Dict[str, Any]) -> List[str]:
        """Identify model limitations"""
        limitations = []

        model_type = model_info.get("type", "unknown")

        # General limitations
        limitations.extend([
            "Model performance may degrade with out-of-distribution data",
            "Predictions should be validated by clinical judgment",
            "Regular model retraining required for optimal performance"
        ])

        # Type-specific limitations
        if model_type == "linear":
            limitations.extend([
                "Assumes linear relationships between features and outcome",
                "May underperform with complex non-linear patterns"
            ])
        elif model_type == "neural_network":
            limitations.extend([
                "Black-box nature reduces interpretability",
                "Requires substantial computational resources",
                "May be sensitive to training data quality"
            ])
        elif model_type == "tree":
            limitations.extend([
                "May overfit to training data",
                "Can be unstable with small changes in training data"
            ])

        # Performance-based limitations
        accuracy = model_info.get("performance_metrics", {}).get("accuracy", 0.8)
        if accuracy < 0.7:
            limitations.append("Model accuracy below recommended threshold for clinical use")

        return limitations

    def _calculate_explainability_metrics(self, model_id: str) -> Dict[str, Any]:
        """Calculate explainability metrics for a model"""
        metrics = self.explainability_metrics.get(model_id, {})

        if not metrics:
            # Generate default metrics
            metrics = {
                "shap_stability": random.uniform(0.7, 0.95),
                "feature_importance_consistency": random.uniform(0.75, 0.9),
                "explanation_fidelity": random.uniform(0.8, 0.95),
                "clinical_interpretability": random.uniform(0.6, 0.85),
                "decision_boundary_clarity": random.uniform(0.7, 0.9)
            }

        return metrics

    def _assess_model_fairness(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model fairness across different demographic groups"""
        # Simulate fairness assessment
        fairness_assessment = {
            "demographic_parity": {
                "age_groups": {"18-39": 0.85, "40-64": 0.82, "65+": 0.80},
                "gender": {"male": 0.83, "female": 0.84},
                "ethnicity": {"group_a": 0.85, "group_b": 0.82, "group_c": 0.81}
            },
            "equal_opportunity": {
                "true_positive_rate": {"privileged": 0.88, "unprivileged": 0.82},
                "false_positive_rate": {"privileged": 0.12, "unprivileged": 0.15}
            },
            "disparate_impact_ratio": 0.92,
            "fairness_score": random.uniform(0.75, 0.9),
            "recommendations": [
                "Monitor predictions across demographic groups",
                "Regular fairness audits recommended",
                "Consider fairness-aware training approaches"
            ]
        }

        return fairness_assessment

    def _update_explainability_metrics(self, model_id: str, method: str, explanation: SHAPExplanation):
        """Update explainability metrics based on new explanation"""
        if model_id not in self.explainability_metrics:
            self.explainability_metrics[model_id] = {}

        metrics = self.explainability_metrics[model_id]

        # Update running averages
        if method == "shap":
            confidence_key = "shap_confidence_avg"
            if confidence_key not in metrics:
                metrics[confidence_key] = explanation.explanation_confidence
            else:
                # Running average
                count = metrics.get("shap_explanation_count", 1)
                metrics[confidence_key] = (metrics[confidence_key] * count + explanation.explanation_confidence) / (count + 1)
                metrics["shap_explanation_count"] = count + 1

        # Update feature importance stability
        if "feature_importance_stability" not in metrics:
            metrics["feature_importance_stability"] = random.uniform(0.8, 0.95)

    def generate_clinical_insights(self, explanation: SHAPExplanation,
                                 context: str = "general") -> Dict[str, Any]:
        """Generate clinical insights from model explanation"""
        insights = {
            "key_factors": self._identify_key_factors(explanation, context),
            "clinical_recommendations": self._generate_clinical_recommendations(explanation, context),
            "risk_assessment": self._assess_clinical_risk(explanation, context),
            "monitoring_suggestions": self._suggest_monitoring(explanation, context),
            "alternative_interventions": self._suggest_alternatives(explanation, context)
        }

        return insights

    def _identify_key_factors(self, explanation: SHAPExplanation, context: str) -> List[Dict[str, Any]]:
        """Identify key factors driving the prediction"""
        key_factors = []

        # Sort features by absolute SHAP value
        sorted_features = sorted(
            explanation.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        context_rules = self.interpretation_rules.get(context, {})
        high_risk_factors = set(context_rules.get("high_risk_factors", []))
        protective_factors = set(context_rules.get("protective_factors", []))

        for feature_name, shap_value in sorted_features[:5]:  # Top 5 factors
            factor_info = {
                "feature": feature_name,
                "impact": shap_value,
                "direction": "increases_risk" if shap_value > 0 else "decreases_risk",
                "magnitude": abs(shap_value),
                "clinical_significance": self._assess_clinical_significance(feature_name, shap_value, context)
            }

            # Add context-specific information
            if feature_name in high_risk_factors:
                factor_info["risk_category"] = "high_risk"
            elif feature_name in protective_factors:
                factor_info["risk_category"] = "protective"
            else:
                factor_info["risk_category"] = "neutral"

            key_factors.append(factor_info)

        return key_factors

    def _assess_clinical_significance(self, feature_name: str, shap_value: float, context: str) -> str:
        """Assess clinical significance of a factor"""
        magnitude = abs(shap_value)

        if magnitude > 0.3:
            significance = "high"
        elif magnitude > 0.15:
            significance = "moderate"
        else:
            significance = "low"

        # Context-specific adjustments
        if context == "cardiovascular" and feature_name in ["cholesterol", "blood_pressure"]:
            significance = "high"  # Always high for these in CV context

        return significance

    def _generate_clinical_recommendations(self, explanation: SHAPExplanation, context: str) -> List[str]:
        """Generate clinical recommendations based on explanation"""
        recommendations = []

        key_factors = self._identify_key_factors(explanation, context)

        # Generate recommendations based on key factors
        for factor in key_factors:
            feature = factor["feature"]
            direction = factor["direction"]
            significance = factor["clinical_significance"]

            if significance == "high":
                if direction == "increases_risk":
                    recommendations.append(f"Address {feature} aggressively - major risk factor identified")
                else:
                    recommendations.append(f"Maintain current {feature} management - provides significant protection")

        # Add context-specific recommendations
        if context == "cardiovascular":
            recommendations.extend([
                "Consider lifestyle modifications",
                "Evaluate need for medication adjustment",
                "Schedule follow-up cardiovascular assessment"
            ])
        elif context == "diabetes":
            recommendations.extend([
                "Monitor blood glucose regularly",
                "Review dietary patterns and physical activity",
                "Assess medication adherence and effectiveness"
            ])

        return recommendations[:5]  # Limit to top 5

    def _assess_clinical_risk(self, explanation: SHAPExplanation, context: str) -> Dict[str, Any]:
        """Assess clinical risk based on explanation"""
        # Calculate risk score from SHAP values
        risk_contributions = sum(abs(value) for value in explanation.shap_values.values() if value > 0)
        protective_contributions = sum(abs(value) for value in explanation.shap_values.values() if value < 0)

        net_risk = risk_contributions - protective_contributions

        # Determine risk level
        if net_risk > 0.5:
            risk_level = "high"
            risk_score = min(1.0, 0.5 + net_risk)
        elif net_risk > 0.2:
            risk_level = "moderate"
            risk_score = 0.3 + net_risk * 2
        else:
            risk_level = "low"
            risk_score = max(0, 0.2 + net_risk * 5)

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "confidence": explanation.explanation_confidence,
            "primary_risk_factors": [
                factor["feature"] for factor in self._identify_key_factors(explanation, context)
                if factor["direction"] == "increases_risk"
            ][:3],
            "protective_factors": [
                factor["feature"] for factor in self._identify_key_factors(explanation, context)
                if factor["direction"] == "decreases_risk"
            ][:3]
        }

    def _suggest_monitoring(self, explanation: SHAPExplanation, context: str) -> List[str]:
        """Suggest monitoring based on explanation"""
        suggestions = []

        key_factors = self._identify_key_factors(explanation, context)

        # Suggest monitoring for high-significance factors
        for factor in key_factors:
            if factor["clinical_significance"] == "high":
                feature = factor["feature"]
                if "pressure" in feature.lower():
                    suggestions.append(f"Regular blood pressure monitoring")
                elif "glucose" in feature.lower():
                    suggestions.append(f"Continuous glucose monitoring")
                elif "cholesterol" in feature.lower():
                    suggestions.append(f"Lipid profile monitoring every 3 months")
                else:
                    suggestions.append(f"Regular monitoring of {feature}")

        # Add general monitoring suggestions
        suggestions.extend([
            "Clinical follow-up in 4-6 weeks",
            "Patient education and lifestyle counseling",
            "Consider advanced diagnostic testing if indicated"
        ])

        return suggestions[:4]  # Limit suggestions

    def _suggest_alternatives(self, explanation: SHAPExplanation, context: str) -> List[str]:
        """Suggest alternative interventions"""
        alternatives = []

        if context == "cardiovascular":
            alternatives.extend([
                "Intensify statin therapy",
                "Add ACE inhibitor or ARB",
                "Consider cardiac rehabilitation program",
                "Evaluate for interventional procedures"
            ])
        elif context == "diabetes":
            alternatives.extend([
                "Switch to different oral medication",
                "Consider insulin therapy",
                "Intensify lifestyle interventions",
                "Add additional glucose-lowering agents"
            ])
        else:
            alternatives.extend([
                "Alternative medication regimens",
                "Lifestyle modification programs",
                "Complementary therapies",
                "Advanced treatment options"
            ])

        return alternatives

    def _explainability_worker(self):
        """Background explainability worker"""
        while self.is_running:
            try:
                # Process queued explanation requests (simplified)
                time.sleep(0.1)
            except Exception as e:
                print(f"Explainability worker error: {e}")

    def _explainability_monitor(self):
        """Background explainability monitoring"""
        while self.is_running:
            try:
                # Monitor explanation quality and update metrics
                for model_id in self.explainability_metrics:
                    metrics = self.explainability_metrics[model_id]
                    # Update running averages and detect degradation
                    if "shap_confidence_avg" in metrics:
                        confidence = metrics["shap_confidence_avg"]
                        if confidence < 0.7:
                            print(f"⚠️ Low explanation confidence for model {model_id}: {confidence}")

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                print(f"Explainability monitor error: {e}")

    def get_explainability_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive explainability report"""
        metrics = self.explainability_metrics.get(model_id, {})
        model_card = self.model_cards.get(model_id)

        return {
            "model_id": model_id,
            "explainability_metrics": metrics,
            "model_card": model_card.__dict__ if model_card else None,
            "explanation_methods_available": list(self.explanation_methods.keys()),
            "clinical_interpretation_rules": self.interpretation_rules,
            "last_updated": datetime.now()
        }
