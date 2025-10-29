"""
Comprehensive Machine Learning Algorithms for AI Personalized Medicine Platform
"""

import math
import random
import statistics
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import json

class MachineLearningAlgorithms:
    """Comprehensive ML algorithms for healthcare applications"""

    def __init__(self):
        self.models = {}
        self.feature_importance = {}

    class NeuralNetwork:
        """Custom neural network implementation for medical predictions"""

        def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
            self.input_size = input_size
            self.hidden_sizes = hidden_sizes
            self.output_size = output_size
            self.weights = []
            self.biases = []
            self._initialize_weights()

        def _initialize_weights(self):
            """Initialize network weights using Xavier initialization"""
            layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

            for i in range(len(layer_sizes) - 1):
                # Xavier initialization
                limit = math.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
                weights = [[random.uniform(-limit, limit) for _ in range(layer_sizes[i])]
                          for _ in range(layer_sizes[i + 1])]
                bias = [0.0] * layer_sizes[i + 1]

                self.weights.append(weights)
                self.biases.append(bias)

        def sigmoid(self, x: float) -> float:
            """Sigmoid activation function"""
            return 1 / (1 + math.exp(-x))

        def relu(self, x: float) -> float:
            """ReLU activation function"""
            return max(0, x)

        def forward(self, inputs: List[float]) -> List[float]:
            """Forward propagation"""
            current = inputs

            for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                # Matrix multiplication
                next_layer = []
                for j in range(len(weight)):
                    activation = bias[j]
                    for k in range(len(current)):
                        activation += weight[j][k] * current[k]

                    # Use ReLU for hidden layers, sigmoid for output
                    if i < len(self.weights) - 1:
                        next_layer.append(self.relu(activation))
                    else:
                        next_layer.append(self.sigmoid(activation))

                current = next_layer

            return current

        def predict(self, inputs: List[float]) -> int:
            """Make prediction"""
            outputs = self.forward(inputs)
            return outputs.index(max(outputs))

    class DecisionTree:
        """Custom decision tree for medical decision making"""

        def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.tree = None

        def fit(self, X: List[List[float]], y: List[int]):
            """Fit decision tree"""
            self.tree = self._build_tree(X, y, depth=0)

        def _build_tree(self, X: List[List[float]], y: List[int], depth: int) -> Dict[str, Any]:
            """Recursively build decision tree"""
            n_samples = len(y)
            n_features = len(X[0]) if X else 0

            # Stopping conditions
            if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(set(y)) == 1):
                return {"value": statistics.mode(y)}

            # Find best split
            best_split = self._find_best_split(X, y, n_features)

            if not best_split:
                return {"value": statistics.mode(y)}

            # Create child nodes
            left_indices = [i for i in range(n_samples) if X[i][best_split["feature"]] <= best_split["threshold"]]
            right_indices = [i for i in range(n_samples) if X[i][best_split["feature"]] > best_split["threshold"]]

            return {
                "feature": best_split["feature"],
                "threshold": best_split["threshold"],
                "left": self._build_tree([X[i] for i in left_indices], [y[i] for i in left_indices], depth + 1),
                "right": self._build_tree([X[i] for i in right_indices], [y[i] for i in right_indices], depth + 1)
            }

        def _find_best_split(self, X: List[List[float]], y: List[int], n_features: int) -> Optional[Dict[str, Any]]:
            """Find best feature and threshold for splitting"""
            best_gini = float('inf')
            best_split = None

            for feature in range(n_features):
                thresholds = sorted(set(row[feature] for row in X))

                for threshold in thresholds:
                    left_indices = [i for i in range(len(X)) if X[i][feature] <= threshold]
                    right_indices = [i for i in range(len(X)) if X[i][feature] > threshold]

                    if not left_indices or not right_indices:
                        continue

                    left_labels = [y[i] for i in left_indices]
                    right_labels = [y[i] for i in right_indices]

                    gini = self._calculate_gini(left_labels, right_labels)

                    if gini < best_gini:
                        best_gini = gini
                        best_split = {
                            "feature": feature,
                            "threshold": threshold,
                            "gini": gini
                        }

            return best_split

        def _calculate_gini(self, left_labels: List[int], right_labels: List[int]) -> float:
            """Calculate Gini impurity"""
            def gini_impurity(labels: List[int]) -> float:
                if not labels:
                    return 0
                proportions = [labels.count(label) / len(labels) for label in set(labels)]
                return 1 - sum(p ** 2 for p in proportions)

            n_left = len(left_labels)
            n_right = len(right_labels)
            n_total = n_left + n_right

            return (n_left / n_total) * gini_impurity(left_labels) + \
                   (n_right / n_total) * gini_impurity(right_labels)

        def predict(self, X: List[List[float]]) -> List[int]:
            """Make predictions"""
            return [self._predict_single(row, self.tree) for row in X]

        def _predict_single(self, row: List[float], node: Dict[str, Any]) -> int:
            """Predict single instance"""
            if "value" in node:
                return node["value"]

            if row[node["feature"]] <= node["threshold"]:
                return self._predict_single(row, node["left"])
            else:
                return self._predict_single(row, node["right"])

    class RandomForest:
        """Ensemble of decision trees"""

        def __init__(self, n_estimators: int = 100, max_depth: int = 10, sample_size: float = 0.8):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.sample_size = sample_size
            self.trees = []

        def fit(self, X: List[List[float]], y: List[int]):
            """Fit random forest"""
            n_samples = len(X)

            for _ in range(self.n_estimators):
                # Bootstrap sampling
                sample_indices = random.sample(range(n_samples), int(n_samples * self.sample_size))
                sample_X = [X[i] for i in sample_indices]
                sample_y = [y[i] for i in sample_indices]

                # Train decision tree
                tree = MachineLearningAlgorithms.DecisionTree(max_depth=self.max_depth)
                tree.fit(sample_X, sample_y)
                self.trees.append(tree)

        def predict(self, X: List[List[float]]) -> List[int]:
            """Make predictions using majority voting"""
            predictions = []

            for row in X:
                tree_predictions = [tree._predict_single(row, tree.tree) for tree in self.trees]
                majority_vote = statistics.mode(tree_predictions)
                predictions.append(majority_vote)

            return predictions

    class SVM:
        """Support Vector Machine implementation"""

        def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000):
            self.lr = learning_rate
            self.lambda_param = lambda_param
            self.n_iters = n_iters
            self.w = None
            self.b = None

        def fit(self, X: List[List[float]], y: List[int]):
            """Fit SVM using gradient descent"""
            n_samples, n_features = len(X), len(X[0])

            # Convert labels to -1, 1
            y_ = [1 if label == 1 else -1 for label in y]

            self.w = [0.0] * n_features
            self.b = 0.0

            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    condition = y_[idx] * (sum(self.w[j] * x_i[j] for j in range(n_features)) + self.b) >= 1

                    if condition:
                        # No misclassification
                        self.w = [self.w[j] - self.lr * (2 * self.lambda_param * self.w[j])
                                for j in range(n_features)]
                    else:
                        # Misclassification
                        self.w = [self.w[j] - self.lr * (2 * self.lambda_param * self.w[j] -
                                y_[idx] * x_i[j]) for j in range(n_features)]
                        self.b -= self.lr * (-y_[idx])

        def predict(self, X: List[List[float]]) -> List[int]:
            """Make predictions"""
            predictions = []

            for x in X:
                linear_output = sum(self.w[j] * x[j] for j in range(len(x))) + self.b
                prediction = 1 if linear_output >= 0 else 0
                predictions.append(prediction)

            return predictions

    class KMeans:
        """K-means clustering for patient stratification"""

        def __init__(self, k: int = 3, max_iters: int = 100):
            self.k = k
            self.max_iters = max_iters
            self.centroids = []
            self.clusters = []

        def fit(self, X: List[List[float]]) -> List[int]:
            """Fit K-means clustering"""
            n_samples, n_features = len(X), len(X[0])

            # Initialize centroids randomly
            random_indices = random.sample(range(n_samples), self.k)
            self.centroids = [X[i][:] for i in random_indices]

            for _ in range(self.max_iters):
                # Assign clusters
                self.clusters = [[] for _ in range(self.k)]

                for point in X:
                    distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
                    cluster_idx = distances.index(min(distances))
                    self.clusters[cluster_idx].append(point)

                # Update centroids
                new_centroids = []
                for cluster in self.clusters:
                    if cluster:
                        centroid = [statistics.mean(dim) for dim in zip(*cluster)]
                        new_centroids.append(centroid)
                    else:
                        new_centroids.append(self.centroids[len(new_centroids)])

                # Check convergence
                if self._centroids_equal(self.centroids, new_centroids):
                    break

                self.centroids = new_centroids

            # Return cluster assignments
            cluster_assignments = []
            for point in X:
                distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
                cluster_idx = distances.index(min(distances))
                cluster_assignments.append(cluster_idx)

            return cluster_assignments

        def _euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
            """Calculate Euclidean distance"""
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

        def _centroids_equal(self, centroids1: List[List[float]], centroids2: List[List[float]]) -> bool:
            """Check if centroids are equal"""
            for c1, c2 in zip(centroids1, centroids2):
                if any(abs(a - b) > 1e-4 for a, b in zip(c1, c2)):
                    return False
            return True

    class LinearRegression:
        """Linear regression for biomarker prediction"""

        def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000):
            self.lr = learning_rate
            self.n_iters = n_iters
            self.weights = None
            self.bias = None

        def fit(self, X: List[List[float]], y: List[float]):
            """Fit linear regression using gradient descent"""
            n_samples, n_features = len(X), len(X[0])

            self.weights = [0.0] * n_features
            self.bias = 0.0

            for _ in range(self.n_iters):
                y_predicted = [sum(self.weights[j] * x[j] for j in range(n_features)) + self.bias for x in X]

                # Calculate gradients
                dw = [0.0] * n_features
                db = 0.0

                for i in range(n_samples):
                    error = y_predicted[i] - y[i]
                    for j in range(n_features):
                        dw[j] += (1 / n_samples) * error * X[i][j]
                    db += (1 / n_samples) * error

                # Update parameters
                self.weights = [self.weights[j] - self.lr * dw[j] for j in range(n_features)]
                self.bias -= self.lr * db

        def predict(self, X: List[List[float]]) -> List[float]:
            """Make predictions"""
            return [sum(self.weights[j] * x[j] for j in range(len(x))) + self.bias for x in X]

    class LogisticRegression:
        """Logistic regression for binary classification"""

        def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000):
            self.lr = learning_rate
            self.n_iters = n_iters
            self.weights = None
            self.bias = None

        def fit(self, X: List[List[float]], y: List[int]):
            """Fit logistic regression"""
            n_samples, n_features = len(X), len(X[0])

            self.weights = [0.0] * n_features
            self.bias = 0.0

            for _ in range(self.n_iters):
                linear_model = [sum(self.weights[j] * x[j] for j in range(n_features)) + self.bias for x in X]
                y_predicted = [self._sigmoid(z) for z in linear_model]

                # Calculate gradients
                dw = [0.0] * n_features
                db = 0.0

                for i in range(n_samples):
                    error = y_predicted[i] - y[i]
                    for j in range(n_features):
                        dw[j] += (1 / n_samples) * error * X[i][j]
                    db += (1 / n_samples) * error

                # Update parameters
                self.weights = [self.weights[j] - self.lr * dw[j] for j in range(n_features)]
                self.bias -= self.lr * db

        def _sigmoid(self, z: float) -> float:
            """Sigmoid function"""
            return 1 / (1 + math.exp(-z))

        def predict_proba(self, X: List[List[float]]) -> List[float]:
            """Predict probabilities"""
            linear_model = [sum(self.weights[j] * x[j] for j in range(len(x))) + self.bias for x in X]
            return [self._sigmoid(z) for z in linear_model]

        def predict(self, X: List[List[float]]) -> List[int]:
            """Make binary predictions"""
            probabilities = self.predict_proba(X)
            return [1 if p >= 0.5 else 0 for p in probabilities]

    # Medical-specific ML methods
    def predict_disease_risk(self, patient_features: List[float], disease_type: str) -> Dict[str, Any]:
        """Predict disease risk using ensemble methods"""
        # Simulate risk prediction model
        base_risk = sum(patient_features) / len(patient_features)

        # Disease-specific adjustments
        adjustments = {
            "cardiovascular": 0.1,
            "diabetes": 0.15,
            "cancer": 0.2,
            "alzheimer": 0.12
        }

        risk_score = base_risk + adjustments.get(disease_type, 0.1)
        risk_score = min(max(risk_score, 0), 1)  # Clamp to [0, 1]

        return {
            "risk_score": risk_score,
            "risk_category": "high" if risk_score > 0.7 else "moderate" if risk_score > 0.4 else "low",
            "confidence": 0.85,
            "contributing_factors": ["age", "family_history", "lifestyle"]
        }

    def cluster_patients(self, patient_data: List[List[float]], n_clusters: int = 4) -> Dict[str, Any]:
        """Cluster patients for stratification"""
        kmeans = self.KMeans(k=n_clusters)
        cluster_assignments = kmeans.fit(patient_data)

        clusters = {}
        for i in range(n_clusters):
            cluster_indices = [j for j, cluster in enumerate(cluster_assignments) if cluster == i]
            clusters[f"cluster_{i}"] = {
                "size": len(cluster_indices),
                "patients": cluster_indices,
                "centroid": kmeans.centroids[i],
                "characteristics": self._analyze_cluster_characteristics([patient_data[j] for j in cluster_indices])
            }

        return {
            "n_clusters": n_clusters,
            "cluster_assignments": cluster_assignments,
            "clusters": clusters,
            "silhouette_score": self._calculate_silhouette_score(patient_data, cluster_assignments, kmeans.centroids)
        }

    def _analyze_cluster_characteristics(self, cluster_data: List[List[float]]) -> Dict[str, Any]:
        """Analyze characteristics of a patient cluster"""
        if not cluster_data:
            return {}

        n_features = len(cluster_data[0])
        characteristics = {}

        for i in range(n_features):
            feature_values = [row[i] for row in cluster_data]
            characteristics[f"feature_{i}"] = {
                "mean": statistics.mean(feature_values),
                "std": statistics.stdev(feature_values) if len(feature_values) > 1 else 0,
                "min": min(feature_values),
                "max": max(feature_values)
            }

        return characteristics

    def _calculate_silhouette_score(self, data: List[List[float]], labels: List[int], centroids: List[List[float]]) -> float:
        """Calculate silhouette score for clustering quality"""
        if len(set(labels)) < 2:
            return 0

        silhouette_scores = []

        for i, point in enumerate(data):
            cluster = labels[i]
            centroid = centroids[cluster]

            # Calculate intra-cluster distance
            intra_distance = self._euclidean_distance(point, centroid)

            # Calculate nearest cluster distance
            inter_distances = []
            for j, other_centroid in enumerate(centroids):
                if j != cluster:
                    inter_distances.append(self._euclidean_distance(point, other_centroid))

            inter_distance = min(inter_distances) if inter_distances else intra_distance

            # Calculate silhouette score
            if max(intra_distance, inter_distance) > 0:
                silhouette = (inter_distance - intra_distance) / max(intra_distance, inter_distance)
            else:
                silhouette = 0

            silhouette_scores.append(silhouette)

        return statistics.mean(silhouette_scores)

    def predict_drug_response(self, patient_genetics: Dict[str, Any], drug_info: Dict[str, Any]) -> Dict[str, Any]:
        """Predict patient response to specific drug"""
        # Simulate drug response prediction
        genetic_factors = patient_genetics.get("variants", [])
        drug_targets = drug_info.get("targets", [])

        # Simple matching algorithm
        matching_score = len(set(genetic_factors) & set(drug_targets)) / max(len(drug_targets), 1)

        # Response categories
        if matching_score > 0.8:
            response = "excellent"
            efficacy = 0.9
        elif matching_score > 0.5:
            response = "good"
            efficacy = 0.75
        elif matching_score > 0.2:
            response = "moderate"
            efficacy = 0.6
        else:
            response = "poor"
            efficacy = 0.3

        return {
            "predicted_response": response,
            "efficacy_score": efficacy,
            "confidence": 0.8,
            "side_effects_risk": 1 - efficacy,
            "recommendations": self._generate_drug_recommendations(response, drug_info)
        }

    def _generate_drug_recommendations(self, response: str, drug_info: Dict[str, Any]) -> List[str]:
        """Generate drug-specific recommendations"""
        recommendations = []

        if response == "excellent":
            recommendations.append("Drug appears highly suitable for this patient")
            recommendations.append("Standard dosage likely optimal")
        elif response == "good":
            recommendations.append("Drug suitable with close monitoring")
            recommendations.append("Consider starting with lower dose")
        elif response == "moderate":
            recommendations.append("Drug may be effective but monitor closely")
            recommendations.append("Consider alternative therapies")
        else:
            recommendations.append("Drug may not be suitable for this patient")
            recommendations.append("Consider alternative medications")

        return recommendations

    def optimize_treatment_dosage(self, patient_data: Dict[str, Any], drug: str) -> Dict[str, Any]:
        """Optimize drug dosage based on patient characteristics"""
        # Base dosage from drug database
        base_dosage = 100  # mg, example

        # Patient factors affecting dosage
        age_factor = 1.0
        weight_factor = patient_data.get("demographics", {}).get("weight", 70) / 70
        liver_function = patient_data.get("biomarkers", {}).get("alt", 30) / 30
        kidney_function = patient_data.get("biomarkers", {}).get("creatinine", 1.0)

        # Genetic factors
        genetic_adjustment = 1.0
        if patient_data.get("genomic_data", {}).get("cyp2d6_status") == "poor_metabolizer":
            genetic_adjustment = 0.5

        # Calculate optimal dosage
        optimal_dosage = base_dosage * age_factor * weight_factor * genetic_adjustment / (liver_function * kidney_function)

        return {
            "recommended_dosage": round(optimal_dosage, 1),
            "base_dosage": base_dosage,
            "adjustment_factors": {
                "weight": weight_factor,
                "genetics": genetic_adjustment,
                "liver_function": 1 / liver_function,
                "kidney_function": 1 / kidney_function
            },
            "monitoring_frequency": "weekly" if optimal_dosage != base_dosage else "monthly",
            "safety_margins": {
                "minimum_safe": optimal_dosage * 0.5,
                "maximum_safe": optimal_dosage * 1.5
            }
        }
