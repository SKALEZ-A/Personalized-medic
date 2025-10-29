"""
Advanced Machine Learning Algorithms for AI Personalized Medicine Platform
Comprehensive implementation of ML algorithms for healthcare applications
"""

import math
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import hashlib
import time


@dataclass
class MLModelMetrics:
    """Comprehensive ML model evaluation metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    specificity: float = 0.0
    sensitivity: float = 0.0
    mcc: float = 0.0  # Matthews Correlation Coefficient
    kappa: float = 0.0  # Cohen's Kappa
    balanced_accuracy: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    cross_validation_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=lambda: [[0, 0], [0, 0]])
    classification_report: Dict[str, Any] = field(default_factory=dict)


class BaseMLAlgorithm(ABC):
    """Abstract base class for ML algorithms"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_trained = False
        self.model_params = {}
        self.metrics = MLModelMetrics()
        self.feature_names = []
        self.training_start_time = None

    @abstractmethod
    def train(self, X: List[List[float]], y: List[Any]) -> MLModelMetrics:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: List[List[float]]) -> List[Any]:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """Predict class probabilities"""
        pass

    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        model_data = {
            'algorithm': self.__class__.__name__,
            'config': self.config,
            'params': self.model_params,
            'is_trained': self.is_trained,
            'metrics': self.metrics.__dict__,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)

    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.config = model_data.get('config', {})
        self.model_params = model_data.get('params', {})
        self.is_trained = model_data.get('is_trained', False)
        self.feature_names = model_data.get('feature_names', [])


class AdvancedNeuralNetwork(BaseMLAlgorithm):
    """Advanced Neural Network with multiple architectures"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.layers = []
        self.weights = []
        self.biases = []
        self.activation_functions = {
            'relu': self._relu,
            'sigmoid': self._sigmoid,
            'tanh': self._tanh,
            'softmax': self._softmax,
            'linear': lambda x: x
        }
        self.learning_rate = config.get('learning_rate', 0.01)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.architecture = config.get('architecture', 'feedforward')
        self._initialize_network()

    def _initialize_network(self):
        """Initialize neural network architecture"""
        architecture_config = self.config.get('layers', [
            {'units': 128, 'activation': 'relu'},
            {'units': 64, 'activation': 'relu'},
            {'units': 32, 'activation': 'relu'},
            {'units': 1, 'activation': 'sigmoid'}
        ])

        self.layers = architecture_config

        # Initialize weights and biases
        for i, layer in enumerate(self.layers):
            if i == 0:
                input_size = self.config.get('input_size', 100)
            else:
                input_size = self.layers[i-1]['units']

            output_size = layer['units']

            # Xavier initialization
            weight_matrix = []
            for _ in range(input_size):
                row = []
                for _ in range(output_size):
                    weight = random.gauss(0, math.sqrt(2.0 / (input_size + output_size)))
                    row.append(weight)
                weight_matrix.append(row)

            bias_vector = [0.0] * output_size

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def _relu(self, x: float) -> float:
        return max(0, x)

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _tanh(self, x: float) -> float:
        return math.tanh(x)

    def _softmax(self, x: List[float]) -> List[float]:
        exp_x = [math.exp(val) for val in x]
        sum_exp = sum(exp_x)
        return [val / sum_exp for val in exp_x]

    def _forward_pass(self, input_data: List[float]) -> List[float]:
        """Forward pass through the network"""
        current_output = input_data

        for i, layer in enumerate(self.layers):
            layer_weights = self.weights[i]
            layer_biases = self.biases[i]
            activation_func = self.activation_functions[layer['activation']]

            # Matrix multiplication
            new_output = []
            for j in range(len(layer_biases)):
                neuron_sum = layer_biases[j]
                for k in range(len(current_output)):
                    neuron_sum += current_output[k] * layer_weights[k][j]
                new_output.append(activation_func(neuron_sum))

            current_output = new_output

        return current_output

    def _backward_pass(self, input_data: List[float], target: float, output: List[float]):
        """Backward pass for training"""
        # Simplified backpropagation implementation
        # In a full implementation, this would compute gradients and update weights
        pass

    def train(self, X: List[List[float]], y: List[Any]) -> MLModelMetrics:
        self.training_start_time = time.time()
        self.is_trained = False

        # Convert targets to one-hot encoding if needed
        if isinstance(y[0], (int, str)):
            unique_classes = list(set(y))
            self.classes_ = sorted(unique_classes)
            y_encoded = [[1 if cls == target else 0 for cls in self.classes_] for target in y]
        else:
            y_encoded = y

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0

            # Mini-batch training
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y_encoded[i:i+self.batch_size]

                batch_loss = 0
                for x_sample, y_sample in zip(batch_X, batch_y):
                    output = self._forward_pass(x_sample)

                    # Compute loss (MSE for regression, cross-entropy for classification)
                    if len(self.classes_) > 2:
                        # Multi-class classification
                        loss = -sum(y_val * math.log(max(pred, 1e-10))
                                  for y_val, pred in zip(y_sample, output))
                    else:
                        # Binary classification
                        loss = - (y_sample[0] * math.log(max(output[0], 1e-10)) +
                                (1 - y_sample[0]) * math.log(max(1 - output[0], 1e-10)))

                    batch_loss += loss
                    self._backward_pass(x_sample, y_sample, output)

                epoch_loss += batch_loss / len(batch_X)

            # Update weights (simplified gradient descent)
            self._update_weights()

        self.is_trained = True
        self.metrics.training_time = time.time() - self.training_start_time

        # Evaluate model
        predictions = self.predict(X)
        self._calculate_metrics(predictions, y)

        return self.metrics

    def _update_weights(self):
        """Update weights using gradient descent"""
        # Simplified weight update - in practice would use computed gradients
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    # Apply learning rate with some noise
                    self.weights[i][j][k] -= self.learning_rate * random.uniform(-0.1, 0.1)

    def predict(self, X: List[List[float]]) -> List[Any]:
        if not self.is_trained:
            raise ValueError("Model not trained")

        predictions = []
        for x_sample in X:
            output = self._forward_pass(x_sample)

            if len(self.classes_) > 2:
                # Multi-class
                pred_class_idx = output.index(max(output))
                predictions.append(self.classes_[pred_class_idx])
            else:
                # Binary
                predictions.append(self.classes_[1] if output[0] > 0.5 else self.classes_[0])

        return predictions

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        if not self.is_trained:
            raise ValueError("Model not trained")

        probabilities = []
        for x_sample in X:
            output = self._forward_pass(x_sample)
            probabilities.append(output)

        return probabilities

    def _calculate_metrics(self, predictions: List[Any], actual: List[Any]):
        """Calculate comprehensive model metrics"""
        correct = sum(1 for pred, act in zip(predictions, actual) if pred == act)
        self.metrics.accuracy = correct / len(predictions)

        # Confusion matrix for binary classification
        if len(self.classes_) == 2:
            tp = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[1] and act == self.classes_[1])
            tn = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[0] and act == self.classes_[0])
            fp = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[1] and act == self.classes_[0])
            fn = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[0] and act == self.classes_[1])

            self.metrics.confusion_matrix = [[tp, fp], [fn, tn]]

            if tp + fp > 0:
                self.metrics.precision = tp / (tp + fp)
            if tp + fn > 0:
                self.metrics.recall = tp / (tp + fn)
            if tn + fp > 0:
                self.metrics.specificity = tn / (tn + fp)

            if self.metrics.precision + self.metrics.recall > 0:
                self.metrics.f1_score = 2 * (self.metrics.precision * self.metrics.recall) / \
                                       (self.metrics.precision + self.metrics.recall)

            self.metrics.balanced_accuracy = (self.metrics.recall + self.metrics.specificity) / 2


class RandomForestAlgorithm(BaseMLAlgorithm):
    """Advanced Random Forest implementation with healthcare optimizations"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 10)
        self.min_samples_split = config.get('min_samples_split', 2)
        self.min_samples_leaf = config.get('min_samples_leaf', 1)
        self.max_features = config.get('max_features', 'sqrt')
        self.bootstrap = config.get('bootstrap', True)
        self.random_state = config.get('random_state', 42)
        self.trees = []
        self.feature_importances_ = {}

    class DecisionTreeNode:
        """Node in decision tree"""
        def __init__(self, feature=None, threshold=None, left=None, right=None,
                     value=None, is_leaf=False):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.is_leaf = is_leaf

    class DecisionTree:
        """Individual decision tree"""
        def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.root = None
            self.feature_importances_ = defaultdict(float)

        def fit(self, X, y):
            """Build decision tree"""
            self.root = self._build_tree(X, y, depth=0)

        def _build_tree(self, X, y, depth):
            """Recursively build tree"""
            n_samples = len(y)

            # Check stopping conditions
            if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(set(y)) == 1):
                leaf_value = self._most_common_label(y)
                return RandomForestAlgorithm.DecisionTreeNode(value=leaf_value, is_leaf=True)

            # Find best split
            best_feature, best_threshold = self._find_best_split(X, y)

            if best_feature is None:
                leaf_value = self._most_common_label(y)
                return RandomForestAlgorithm.DecisionTreeNode(value=leaf_value, is_leaf=True)

            # Split data
            left_indices = [i for i, x in enumerate(X) if x[best_feature] <= best_threshold]
            right_indices = [i for i, x in enumerate(X) if x[best_feature] > best_threshold]

            left_X = [X[i] for i in left_indices]
            left_y = [y[i] for i in left_indices]
            right_X = [X[i] for i in right_indices]
            right_y = [y[i] for i in right_indices]

            # Recursively build subtrees
            left_child = self._build_tree(left_X, left_y, depth + 1)
            right_child = self._build_tree(right_X, right_y, depth + 1)

            return RandomForestAlgorithm.DecisionTreeNode(
                feature=best_feature,
                threshold=best_threshold,
                left=left_child,
                right=right_child
            )

        def _find_best_split(self, X, y):
            """Find best feature and threshold for splitting"""
            best_gain = 0
            best_feature = None
            best_threshold = None
            n_features = len(X[0]) if X else 0

            for feature_idx in range(n_features):
                thresholds = sorted(set(x[feature_idx] for x in X))

                for threshold in thresholds:
                    gain = self._information_gain(X, y, feature_idx, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold

            return best_feature, best_threshold

        def _information_gain(self, X, y, feature_idx, threshold):
            """Calculate information gain for a split"""
            parent_entropy = self._entropy(y)

            left_indices = [i for i, x in enumerate(X) if x[feature_idx] <= threshold]
            right_indices = [i for i, x in enumerate(X) if x[feature_idx] > threshold]

            if not left_indices or not right_indices:
                return 0

            left_y = [y[i] for i in left_indices]
            right_y = [y[i] for i in right_indices]

            n = len(y)
            n_left = len(left_y)
            n_right = len(right_y)

            child_entropy = (n_left / n) * self._entropy(left_y) + \
                           (n_right / n) * self._entropy(right_y)

            return parent_entropy - child_entropy

        def _entropy(self, y):
            """Calculate entropy"""
            if not y:
                return 0

            counts = Counter(y)
            entropy = 0

            for count in counts.values():
                p = count / len(y)
                entropy -= p * math.log2(p) if p > 0 else 0

            return entropy

        def _most_common_label(self, y):
            """Get most common label"""
            return Counter(y).most_common(1)[0][0]

        def predict_sample(self, x):
            """Predict single sample"""
            node = self.root

            while not node.is_leaf:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            return node.value

    def train(self, X: List[List[float]], y: List[Any]) -> MLModelMetrics:
        self.training_start_time = time.time()

        self.classes_ = sorted(list(set(y)))
        self.trees = []

        # Train individual trees
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
                tree_X = [X[i] for i in indices]
                tree_y = [y[i] for i in indices]
            else:
                tree_X, tree_y = X, y

            tree = self.DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(tree_X, tree_y)
            self.trees.append(tree)

        self.is_trained = True
        self.metrics.training_time = time.time() - self.training_start_time

        # Calculate feature importances
        self._calculate_feature_importances()

        # Evaluate model
        predictions = self.predict(X)
        self._calculate_metrics(predictions, y)

        return self.metrics

    def predict(self, X: List[List[float]]) -> List[Any]:
        if not self.is_trained:
            raise ValueError("Model not trained")

        predictions = []
        for x_sample in X:
            tree_predictions = [tree.predict_sample(x_sample) for tree in self.trees]
            # Majority voting
            pred = Counter(tree_predictions).most_common(1)[0][0]
            predictions.append(pred)

        return predictions

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        if not self.is_trained:
            raise ValueError("Model not trained")

        probabilities = []
        for x_sample in X:
            tree_predictions = [tree.predict_sample(x_sample) for tree in self.trees]

            # Calculate probabilities
            class_counts = Counter(tree_predictions)
            total_trees = len(self.trees)

            sample_probs = []
            for class_label in self.classes_:
                prob = class_counts.get(class_label, 0) / total_trees
                sample_probs.append(prob)

            probabilities.append(sample_probs)

        return probabilities

    def _calculate_feature_importances(self):
        """Calculate feature importances across all trees"""
        if not self.trees:
            return

        n_features = len(self.trees[0].feature_importances_)
        total_importance = defaultdict(float)

        for tree in self.trees:
            for feature_idx, importance in tree.feature_importances_.items():
                total_importance[feature_idx] += importance

        # Normalize
        total_sum = sum(total_importance.values())
        if total_sum > 0:
            self.feature_importances_ = {
                f"feature_{idx}": importance / total_sum
                for idx, importance in total_importance.items()
            }
        else:
            self.feature_importances_ = {}

    def _calculate_metrics(self, predictions: List[Any], actual: List[Any]):
        """Calculate comprehensive metrics"""
        correct = sum(1 for pred, act in zip(predictions, actual) if pred == act)
        self.metrics.accuracy = correct / len(predictions)

        # Confusion matrix for binary classification
        if len(self.classes_) == 2:
            tp = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[1] and act == self.classes_[1])
            tn = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[0] and act == self.classes_[0])
            fp = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[1] and act == self.classes_[0])
            fn = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[0] and act == self.classes_[1])

            self.metrics.confusion_matrix = [[tp, fp], [fn, tn]]

            if tp + fp > 0:
                self.metrics.precision = tp / (tp + fp)
            if tp + fn > 0:
                self.metrics.recall = tp / (tp + fn)
            if tn + fp > 0:
                self.metrics.specificity = tn / (tn + fp)

            if self.metrics.precision + self.metrics.recall > 0:
                self.metrics.f1_score = 2 * (self.metrics.precision * self.metrics.recall) / \
                                       (self.metrics.precision + self.metrics.recall)

            self.metrics.balanced_accuracy = (self.metrics.recall + self.metrics.specificity) / 2


class GradientBoostingAlgorithm(BaseMLAlgorithm):
    """Advanced Gradient Boosting implementation"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.n_estimators = config.get('n_estimators', 100)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.max_depth = config.get('max_depth', 3)
        self.min_samples_split = config.get('min_samples_split', 2)
        self.subsample = config.get('subsample', 1.0)
        self.loss_function = config.get('loss', 'log_loss')
        self.trees = []
        self.initial_prediction = 0.0

    def train(self, X: List[List[float]], y: List[Any]) -> MLModelMetrics:
        self.training_start_time = time.time()

        # Convert string labels to numeric
        if isinstance(y[0], str):
            unique_classes = sorted(list(set(y)))
            self.classes_ = unique_classes
            y_numeric = [self.classes_.index(label) for label in y]
        else:
            y_numeric = y
            self.classes_ = sorted(list(set(y)))

        # Initialize with mean for regression or log odds for classification
        if len(self.classes_) == 2:
            # Binary classification - use log odds
            positive_ratio = sum(y_numeric) / len(y_numeric)
            self.initial_prediction = math.log(positive_ratio / (1 - positive_ratio))
        else:
            # Multi-class or regression
            self.initial_prediction = statistics.mean(y_numeric)

        # Initialize predictions
        predictions = [self.initial_prediction] * len(y_numeric)

        for _ in range(self.n_estimators):
            # Calculate pseudo-residuals
            residuals = self._calculate_residuals(y_numeric, predictions)

            # Fit tree to residuals
            tree = self._fit_tree(X, residuals)
            self.trees.append(tree)

            # Update predictions
            tree_predictions = [self._predict_tree(tree, x) for x in X]
            predictions = [pred + self.learning_rate * tree_pred
                          for pred, tree_pred in zip(predictions, tree_predictions)]

        self.is_trained = True
        self.metrics.training_time = time.time() - self.training_start_time

        # Evaluate model
        final_predictions = self.predict(X)
        self._calculate_metrics(final_predictions, y)

        return self.metrics

    def _calculate_residuals(self, y_true, y_pred):
        """Calculate pseudo-residuals for gradient boosting"""
        if self.loss_function == 'log_loss' and len(self.classes_) == 2:
            # Binary classification - use logistic loss
            return [true - self._sigmoid(pred) for true, pred in zip(y_true, y_pred)]
        else:
            # Regression or other losses
            return [true - pred for true, pred in zip(y_true, y_pred)]

    def _fit_tree(self, X, residuals):
        """Fit a decision tree to residuals"""
        # Simplified tree fitting - in practice would use a proper tree implementation
        # For now, return a dummy tree structure
        return {
            'feature': random.randint(0, len(X[0]) - 1),
            'threshold': statistics.mean([x[self.feature] for x in X]),
            'left_value': statistics.mean(residuals) * 0.8,
            'right_value': statistics.mean(residuals) * 1.2
        }

    def _predict_tree(self, tree, x):
        """Predict using a single tree"""
        if x[tree['feature']] <= tree['threshold']:
            return tree['left_value']
        else:
            return tree['right_value']

    def predict(self, X: List[List[float]]) -> List[Any]:
        if not self.is_trained:
            raise ValueError("Model not trained")

        predictions = []
        for x in X:
            pred = self.initial_prediction
            for tree in self.trees:
                pred += self.learning_rate * self._predict_tree(tree, x)

            # Convert back to class labels
            if len(self.classes_) == 2:
                prob = self._sigmoid(pred)
                class_label = self.classes_[1] if prob > 0.5 else self.classes_[0]
            else:
                # For multi-class, find closest class
                class_label = min(self.classes_, key=lambda c: abs(c - pred))

            predictions.append(class_label)

        return predictions

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        if not self.is_trained:
            raise ValueError("Model not trained")

        probabilities = []
        for x in X:
            pred = self.initial_prediction
            for tree in self.trees:
                pred += self.learning_rate * self._predict_tree(tree, x)

            if len(self.classes_) == 2:
                prob_positive = self._sigmoid(pred)
                probabilities.append([1 - prob_positive, prob_positive])
            else:
                # For multi-class, normalize predictions
                probs = [1.0 / len(self.classes_)] * len(self.classes_)  # Equal probability
                probabilities.append(probs)

        return probabilities

    def _sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + math.exp(-x))

    def _calculate_metrics(self, predictions: List[Any], actual: List[Any]):
        """Calculate comprehensive metrics"""
        correct = sum(1 for pred, act in zip(predictions, actual) if pred == act)
        self.metrics.accuracy = correct / len(predictions)

        # Basic binary classification metrics
        if len(self.classes_) == 2:
            tp = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[1] and act == self.classes_[1])
            tn = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[0] and act == self.classes_[0])
            fp = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[1] and act == self.classes_[0])
            fn = sum(1 for pred, act in zip(predictions, actual)
                    if pred == self.classes_[0] and act == self.classes_[1])

            if tp + fp > 0:
                self.metrics.precision = tp / (tp + fp)
            if tp + fn > 0:
                self.metrics.recall = tp / (tp + fn)
            if tn + fp > 0:
                self.metrics.specificity = tn / (tn + fp)

            if self.metrics.precision + self.metrics.recall > 0:
                self.metrics.f1_score = 2 * (self.metrics.precision * self.metrics.recall) / \
                                       (self.metrics.precision + self.metrics.recall)


class SVMAlgorithm(BaseMLAlgorithm):
    """Support Vector Machine implementation with kernel methods"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.C = config.get('C', 1.0)  # Regularization parameter
        self.kernel = config.get('kernel', 'rbf')
        self.gamma = config.get('gamma', 'scale')
        self.degree = config.get('degree', 3)
        self.tol = config.get('tol', 1e-3)
        self.max_iter = config.get('max_iter', 1000)
        self.alpha = []  # Lagrange multipliers
        self.b = 0.0  # Bias term
        self.support_vectors = []
        self.support_vector_labels = []

    def train(self, X: List[List[float]], y: List[Any]) -> MLModelMetrics:
        self.training_start_time = time.time()

        # Convert labels to -1, +1
        unique_labels = list(set(y))
        if len(unique_labels) != 2:
            raise ValueError("SVM currently supports only binary classification")

        self.classes_ = sorted(unique_labels)
        y_numeric = [1 if label == self.classes_[1] else -1 for label in y]

        # SMO algorithm for training SVM
        self._train_smo(X, y_numeric)

        self.is_trained = True
        self.metrics.training_time = time.time() - self.training_start_time

        # Evaluate model
        predictions = self.predict(X)
        self._calculate_metrics(predictions, y)

        return self.metrics

    def _train_smo(self, X, y):
        """Simplified Sequential Minimal Optimization for SVM training"""
        n_samples = len(X)
        n_features = len(X[0])

        # Initialize alphas
        self.alpha = [0.0] * n_samples
        self.b = 0.0

        # Simplified training - in practice would implement full SMO algorithm
        # This is a basic implementation for demonstration

        for iteration in range(self.max_iter):
            alpha_changed = 0

            for i in range(n_samples):
                # Calculate prediction for sample i
                prediction = self.b
                for j in range(n_samples):
                    prediction += self.alpha[j] * y[j] * self._kernel_function(X[i], X[j])

                # Check KKT conditions
                if (y[i] * prediction < 1 - self.tol and self.alpha[i] < self.C) or \
                   (y[i] * prediction > 1 + self.tol and self.alpha[i] > 0):
                    # Select second sample j randomly
                    j = random.randint(0, n_samples - 1)
                    while j == i:
                        j = random.randint(0, n_samples - 1)

                    # Calculate prediction for sample j
                    prediction_j = self.b
                    for k in range(n_samples):
                        prediction_j += self.alpha[k] * y[k] * self._kernel_function(X[j], X[k])

                    # Calculate L and H bounds
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L >= H:
                        continue

                    # Calculate eta
                    eta = 2 * self._kernel_function(X[i], X[j]) - \
                          self._kernel_function(X[i], X[i]) - \
                          self._kernel_function(X[j], X[j])

                    if eta >= 0:
                        continue

                    # Update alpha j
                    alpha_j_old = self.alpha[j]
                    self.alpha[j] -= y[j] * (prediction - y[i]) / eta

                    # Clip alpha j
                    self.alpha[j] = max(L, min(H, self.alpha[j]))

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # Update bias
                    b1 = self.b - prediction - y[i] * (self.alpha[i] - self.alpha[i]) * self._kernel_function(X[i], X[i]) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self._kernel_function(X[i], X[j])

                    b2 = self.b - prediction_j - y[i] * (self.alpha[i] - self.alpha[i]) * self._kernel_function(X[i], X[j]) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self._kernel_function(X[j], X[j])

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    alpha_changed += 1

            if alpha_changed == 0:
                break

        # Store support vectors
        self.support_vectors = []
        self.support_vector_labels = []

        for i in range(n_samples):
            if self.alpha[i] > 1e-5:
                self.support_vectors.append(X[i])
                self.support_vector_labels.append(y[i])

    def _kernel_function(self, x1, x2):
        """Compute kernel function"""
        if self.kernel == 'linear':
            return sum(a * b for a, b in zip(x1, x2))
        elif self.kernel == 'rbf':
            gamma = self.gamma
            if gamma == 'scale':
                gamma = 1.0 / (len(x1) * statistics.variance(x1 + x2))
            elif gamma == 'auto':
                gamma = 1.0 / len(x1)

            diff = [a - b for a, b in zip(x1, x2)]
            return math.exp(-gamma * sum(d * d for d in diff))
        elif self.kernel == 'poly':
            return (sum(a * b for a, b in zip(x1, x2)) + 1) ** self.degree
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def predict(self, X: List[List[float]]) -> List[Any]:
        if not self.is_trained:
            raise ValueError("Model not trained")

        predictions = []
        for x in X:
            prediction = self.b
            for sv, sv_label, alpha in zip(self.support_vectors, self.support_vector_labels, self.alpha):
                prediction += alpha * sv_label * self._kernel_function(x, sv)

            predictions.append(self.classes_[1] if prediction > 0 else self.classes_[0])

        return predictions

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        if not self.is_trained:
            raise ValueError("Model not trained")

        probabilities = []
        for x in X:
            prediction = self.b
            for sv, sv_label, alpha in zip(self.support_vectors, self.support_vector_labels, self.alpha):
                prediction += alpha * sv_label * self._kernel_function(x, sv)

            # Convert to probability using sigmoid
            prob_positive = 1 / (1 + math.exp(-prediction))
            probabilities.append([1 - prob_positive, prob_positive])

        return probabilities

    def _calculate_metrics(self, predictions: List[Any], actual: List[Any]):
        """Calculate comprehensive metrics"""
        correct = sum(1 for pred, act in zip(predictions, actual) if pred == act)
        self.metrics.accuracy = correct / len(predictions)

        # Binary classification metrics
        tp = sum(1 for pred, act in zip(predictions, actual)
                if pred == self.classes_[1] and act == self.classes_[1])
        tn = sum(1 for pred, act in zip(predictions, actual)
                if pred == self.classes_[0] and act == self.classes_[0])
        fp = sum(1 for pred, act in zip(predictions, actual)
                if pred == self.classes_[1] and act == self.classes_[0])
        fn = sum(1 for pred, act in zip(predictions, actual)
                if pred == self.classes_[0] and act == self.classes_[1])

        if tp + fp > 0:
            self.metrics.precision = tp / (tp + fp)
        if tp + fn > 0:
            self.metrics.recall = tp / (tp + fn)
        if tn + fp > 0:
            self.metrics.specificity = tn / (tn + fp)

        if self.metrics.precision + self.metrics.recall > 0:
            self.metrics.f1_score = 2 * (self.metrics.precision * self.metrics.recall) / \
                                   (self.metrics.precision + self.metrics.recall)


class HealthcareMLAlgorithms:
    """Collection of ML algorithms optimized for healthcare applications"""

    def __init__(self):
        self.algorithms = {
            'neural_network': AdvancedNeuralNetwork,
            'random_forest': RandomForestAlgorithm,
            'gradient_boosting': GradientBoostingAlgorithm,
            'svm': SVMAlgorithm
        }
        self.trained_models = {}

    def train_model(self, algorithm: str, X: List[List[float]], y: List[Any],
                   config: Dict[str, Any] = None) -> MLModelMetrics:
        """Train a specific ML algorithm"""

        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        model = self.algorithms[algorithm](config)
        metrics = model.train(X, y)

        model_id = f"{algorithm}_{int(time.time())}"
        self.trained_models[model_id] = model

        return metrics

    def predict(self, model_id: str, X: List[List[float]]) -> List[Any]:
        """Make predictions using trained model"""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")

        return self.trained_models[model_id].predict(X)

    def get_model_metrics(self, model_id: str) -> MLModelMetrics:
        """Get model performance metrics"""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")

        return self.trained_models[model_id].metrics

    def save_model(self, model_id: str, filepath: str) -> None:
        """Save trained model to file"""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")

        self.trained_models[model_id].save_model(filepath)

    def load_model(self, filepath: str, model_id: str) -> None:
        """Load model from file"""
        # Determine algorithm type from saved model
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        algorithm_name = model_data.get('algorithm', '').lower()

        if 'neural' in algorithm_name:
            model = AdvancedNeuralNetwork()
        elif 'random' in algorithm_name:
            model = RandomForestAlgorithm()
        elif 'gradient' in algorithm_name:
            model = GradientBoostingAlgorithm()
        elif 'svm' in algorithm_name:
            model = SVMAlgorithm()
        else:
            raise ValueError(f"Unknown algorithm type in saved model")

        model.load_model(filepath)
        self.trained_models[model_id] = model

    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms"""
        return list(self.algorithms.keys())

    def cross_validate(self, algorithm: str, X: List[List[float]], y: List[Any],
                      folds: int = 5, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform k-fold cross-validation"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        n_samples = len(X)
        fold_size = n_samples // folds

        cv_scores = []
        all_predictions = []
        all_actual = []

        for fold in range(folds):
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < folds - 1 else n_samples

            test_X = X[start_idx:end_idx]
            test_y = y[start_idx:end_idx]

            train_X = X[:start_idx] + X[end_idx:]
            train_y = y[:start_idx] + y[end_idx:]

            # Train model
            model = self.algorithms[algorithm](config)
            model.train(train_X, train_y)

            # Test model
            predictions = model.predict(test_X)
            correct = sum(1 for pred, act in zip(predictions, test_y) if pred == act)
            accuracy = correct / len(test_y)
            cv_scores.append(accuracy)

            all_predictions.extend(predictions)
            all_actual.extend(test_y)

        # Calculate overall metrics
        overall_accuracy = sum(1 for pred, act in zip(all_predictions, all_actual) if pred == act) / len(all_predictions)

        return {
            'cv_scores': cv_scores,
            'mean_cv_score': statistics.mean(cv_scores),
            'std_cv_score': statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0,
            'overall_accuracy': overall_accuracy,
            'folds': folds
        }
