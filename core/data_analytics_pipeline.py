"""
Data Analytics Pipeline for AI Personalized Medicine Platform
Comprehensive data processing, analytics, and visualization capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import warnings
warnings.filterwarnings('ignore')

# Data processing and analytics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
from scipy.signal import find_peaks
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Healthcare-specific analytics
from collections import defaultdict, Counter
import networkx as nx
from networkx.algorithms import community
import community as community_louvain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareDataAnalytics:
    """
    Comprehensive data analytics pipeline for healthcare data
    Includes statistical analysis, time series analysis, clustering, and visualization
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
        self.processed_data = None
        self.analytics_results = {}
        self.visualizations = {}

        # Initialize analytics components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config.get('pca_components', 0.95))

        logger.info("Healthcare Data Analytics initialized")

    def load_and_process_data(self, data_source: Union[str, pd.DataFrame],
                            data_type: str = 'healthcare') -> pd.DataFrame:
        """Load and perform initial data processing"""

        if isinstance(data_source, str):
            # Load from file
            if data_source.endswith('.csv'):
                self.data = pd.read_csv(data_source)
            elif data_source.endswith('.json'):
                self.data = pd.read_json(data_source)
            elif data_source.endswith('.parquet'):
                self.data = pd.read_parquet(data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        else:
            self.data = data_source.copy()

        logger.info(f"Loaded data with shape: {self.data.shape}")

        # Initial data cleaning and preprocessing
        self._clean_data()
        self._standardize_data_formats()

        if data_type == 'healthcare':
            self._process_healthcare_data()
        elif data_type == 'time_series':
            self._process_time_series_data()
        elif data_type == 'genomic':
            self._process_genomic_data()

        self.processed_data = self.data.copy()
        return self.processed_data

    def _clean_data(self):
        """Perform initial data cleaning"""
        # Remove duplicates
        initial_shape = self.data.shape
        self.data = self.data.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - self.data.shape[0]} duplicate rows")

        # Handle missing values
        missing_stats = self.data.isnull().sum()
        if missing_stats.sum() > 0:
            logger.info(f"Missing values found: {missing_stats[missing_stats > 0].to_dict()}")

            # Fill missing values based on data type
            for column in self.data.columns:
                if self.data[column].dtype in ['int64', 'float64']:
                    # Numeric columns: fill with median
                    self.data[column] = self.data[column].fillna(self.data[column].median())
                elif self.data[column].dtype == 'object':
                    # Categorical columns: fill with mode
                    self.data[column] = self.data[column].fillna(self.data[column].mode().iloc[0])

        # Remove columns with too many missing values
        missing_threshold = self.config.get('missing_threshold', 0.5)
        columns_to_drop = self.data.columns[self.data.isnull().mean() > missing_threshold]
        if len(columns_to_drop) > 0:
            self.data = self.data.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns with >{missing_threshold*100}% missing values: {list(columns_to_drop)}")

    def _standardize_data_formats(self):
        """Standardize data formats"""
        # Convert date columns
        date_columns = self._identify_date_columns()
        for col in date_columns:
            try:
                self.data[col] = pd.to_datetime(self.data[col])
            except:
                logger.warning(f"Could not convert {col} to datetime")

        # Standardize categorical values
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            # Convert to lowercase and strip whitespace
            self.data[col] = self.data[col].astype(str).str.lower().str.strip()

    def _identify_date_columns(self) -> List[str]:
        """Identify columns that contain dates"""
        date_columns = []
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                date_columns.append(col)
            elif self.data[col].dtype == 'object':
                # Try to parse first few values
                sample_values = self.data[col].dropna().head(5)
                try:
                    pd.to_datetime(sample_values)
                    date_columns.append(col)
                except:
                    continue
        return date_columns

    def _process_healthcare_data(self):
        """Process healthcare-specific data"""
        # Calculate BMI if height and weight are available
        if 'height' in self.data.columns and 'weight' in self.data.columns:
            # Convert height to meters if in cm
            height_m = self.data['height']
            if height_m.mean() > 3:  # Likely in cm
                height_m = height_m / 100

            self.data['bmi'] = self.data['weight'] / (height_m ** 2)
            self.data['bmi_category'] = pd.cut(
                self.data['bmi'],
                bins=[0, 18.5, 25, 30, np.inf],
                labels=['underweight', 'normal', 'overweight', 'obese']
            )

        # Calculate cardiovascular risk scores
        self._calculate_risk_scores()

        # Identify health patterns
        self._identify_health_patterns()

    def _calculate_risk_scores(self):
        """Calculate various health risk scores"""
        # Framingham Risk Score (simplified)
        if all(col in self.data.columns for col in ['age', 'gender', 'cholesterol', 'blood_pressure_systolic', 'smoking']):
            self.data['cardiovascular_risk'] = self._calculate_framingham_risk()

        # Diabetes risk score
        if all(col in self.data.columns for col in ['age', 'bmi', 'glucose', 'family_history_diabetes']):
            self.data['diabetes_risk'] = self._calculate_diabetes_risk()

    def _calculate_framingham_risk(self) -> pd.Series:
        """Calculate simplified Framingham risk score"""
        # Simplified implementation - in practice would use full Framingham equation
        risk_score = np.zeros(len(self.data))

        # Age factor
        risk_score += (self.data['age'] - 20) * 0.1

        # Gender factor
        risk_score += (self.data['gender'] == 'male').astype(int) * 2

        # Cholesterol factor
        risk_score += (self.data['cholesterol'] - 200) * 0.01

        # Blood pressure factor
        risk_score += (self.data['blood_pressure_systolic'] - 120) * 0.05

        # Smoking factor
        risk_score += self.data['smoking'].astype(int) * 3

        # Normalize to 0-100 scale
        risk_score = np.clip(risk_score, 0, 20) * 5

        return risk_score

    def _calculate_diabetes_risk(self) -> pd.Series:
        """Calculate diabetes risk score"""
        risk_score = np.zeros(len(self.data))

        # Age factor
        risk_score += np.where(self.data['age'] > 45, 2, 0)

        # BMI factor
        risk_score += np.where(self.data['bmi'] > 25, 2, 0)

        # Glucose factor
        risk_score += np.where(self.data['glucose'] > 100, 2, 0)

        # Family history
        risk_score += self.data['family_history_diabetes'].astype(int) * 3

        return np.clip(risk_score, 0, 10)

    def _identify_health_patterns(self):
        """Identify health patterns and correlations"""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns

        # Calculate correlations
        correlation_matrix = self.data[numeric_columns].corr()

        # Identify strong correlations (|r| > 0.7)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })

        self.analytics_results['strong_correlations'] = strong_correlations

    def _process_time_series_data(self):
        """Process time series healthcare data"""
        if 'timestamp' not in self.data.columns:
            logger.warning("No timestamp column found for time series processing")
            return

        # Set timestamp as index
        self.data = self.data.set_index('timestamp').sort_index()

        # Resample to regular intervals
        self.data = self.data.resample('1H').mean()

        # Fill missing values
        self.data = self.data.interpolate(method='time')

    def _process_genomic_data(self):
        """Process genomic data"""
        # This would handle genomic-specific processing
        # Placeholder for genomic data processing logic
        pass

    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        logger.info("Performing statistical analysis...")

        results = {
            'descriptive_stats': {},
            'inferential_stats': {},
            'distribution_analysis': {},
            'outlier_analysis': {}
        }

        numeric_columns = self.data.select_dtypes(include=[np.number]).columns

        # Descriptive statistics
        results['descriptive_stats'] = self.data[numeric_columns].describe().to_dict()

        # Distribution analysis
        results['distribution_analysis'] = {}
        for col in numeric_columns:
            distribution_stats = self._analyze_distribution(self.data[col])
            results['distribution_analysis'][col] = distribution_stats

        # Outlier analysis
        results['outlier_analysis'] = {}
        for col in numeric_columns:
            outlier_info = self._detect_outliers(self.data[col])
            results['outlier_analysis'][col] = outlier_info

        # Inferential statistics
        results['inferential_stats'] = self._perform_inferential_stats()

        self.analytics_results['statistical_analysis'] = results
        return results

    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze distribution of a numeric series"""
        # Normality tests
        _, shapiro_p = stats.shapiro(series.dropna().sample(min(5000, len(series))))

        # Skewness and kurtosis
        skewness = series.skew()
        kurtosis = series.kurtosis()

        # Distribution type classification
        if shapiro_p > 0.05:
            distribution_type = 'normal'
        elif abs(skewness) > 1:
            distribution_type = 'skewed'
        else:
            distribution_type = 'non-normal'

        return {
            'shapiro_p_value': shapiro_p,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'distribution_type': distribution_type,
            'is_normal': shapiro_p > 0.05
        }

    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        # IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()

        # Z-score method
        z_scores = np.abs(stats.zscore(series.dropna()))
        zscore_outliers = (z_scores > 3).sum()

        # Modified Z-score method
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        mod_zscore_outliers = (np.abs(modified_z_scores) > 3.5).sum()

        return {
            'iqr_outliers': int(iqr_outliers),
            'zscore_outliers': int(zscore_outliers),
            'modified_zscore_outliers': int(mod_zscore_outliers),
            'total_samples': len(series),
            'outlier_percentage': (iqr_outliers / len(series)) * 100
        }

    def _perform_inferential_stats(self) -> Dict[str, Any]:
        """Perform inferential statistical tests"""
        results = {}

        # Identify categorical and numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns

        # ANOVA for numeric vs categorical
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            results['anova_tests'] = {}
            for num_col in numeric_cols[:5]:  # Limit to first 5 for performance
                for cat_col in categorical_cols[:3]:
                    try:
                        groups = [group for name, group in self.data.groupby(cat_col)[num_col]]
                        if len(groups) > 1 and all(len(g) > 1 for g in groups):
                            f_stat, p_value = stats.f_oneway(*groups)
                            results['anova_tests'][f'{num_col}_vs_{cat_col}'] = {
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                    except:
                        continue

        # Chi-square tests for categorical variables
        if len(categorical_cols) > 1:
            results['chi_square_tests'] = {}
            for i, col1 in enumerate(categorical_cols[:3]):
                for col2 in categorical_cols[i+1:i+2]:
                    try:
                        contingency_table = pd.crosstab(self.data[col1], self.data[col2])
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                        results['chi_square_tests'][f'{col1}_vs_{col2}'] = {
                            'chi_square': chi2,
                            'p_value': p_value,
                            'degrees_of_freedom': dof,
                            'significant': p_value < 0.05
                        }
                    except:
                        continue

        return results

    def perform_clustering_analysis(self, n_clusters: Optional[int] = None,
                                  method: str = 'kmeans') -> Dict[str, Any]:
        """Perform clustering analysis"""
        logger.info(f"Performing {method} clustering analysis...")

        # Prepare data for clustering
        numeric_data = self.data.select_dtypes(include=[np.number])
        scaled_data = self.scaler.fit_transform(numeric_data)

        # Apply PCA for dimensionality reduction if needed
        if scaled_data.shape[1] > 10:
            scaled_data = self.pca.fit_transform(scaled_data)
            logger.info(f"Applied PCA: reduced to {scaled_data.shape[1]} components")

        results = {
            'method': method,
            'n_clusters': n_clusters,
            'cluster_labels': [],
            'cluster_centers': [],
            'evaluation_metrics': {},
            'cluster_characteristics': {}
        }

        if method == 'kmeans':
            if n_clusters is None:
                # Use elbow method to find optimal k
                n_clusters = self._find_optimal_clusters(scaled_data, max_k=10)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)

            results['cluster_labels'] = cluster_labels.tolist()
            results['cluster_centers'] = kmeans.cluster_centers_.tolist()
            results['inertia'] = kmeans.inertia_

            # Evaluate clustering
            if len(np.unique(cluster_labels)) > 1:
                results['evaluation_metrics'] = {
                    'silhouette_score': silhouette_score(scaled_data, cluster_labels),
                    'calinski_harabasz_score': calinski_harabasz_score(scaled_data, cluster_labels)
                }

        elif method == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(scaled_data)

            results['cluster_labels'] = cluster_labels.tolist()
            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            results['n_clusters_found'] = n_clusters_found
            results['noise_points'] = list(cluster_labels).count(-1)

        elif method == 'hierarchical':
            if n_clusters is None:
                n_clusters = 3

            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = hierarchical.fit_predict(scaled_data)

            results['cluster_labels'] = cluster_labels.tolist()

        # Analyze cluster characteristics
        results['cluster_characteristics'] = self._analyze_cluster_characteristics(
            cluster_labels, numeric_data
        )

        # Add cluster labels to original data
        self.data['cluster'] = cluster_labels

        self.analytics_results['clustering_analysis'] = results
        return results

    def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

            if k > 1:
                silhouette_scores.append(silhouette_score(data, kmeans.labels_))

        # Find elbow point (simplified)
        # In practice, you might use more sophisticated methods
        optimal_k = 3  # Default fallback

        # Simple elbow detection
        if len(inertias) > 2:
            # Calculate second derivatives
            accelerations = []
            for i in range(1, len(inertias) - 1):
                acceleration = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                accelerations.append(acceleration)

            if accelerations:
                elbow_idx = np.argmax(np.abs(accelerations))
                optimal_k = elbow_idx + 2  # +2 because we start from k=2

        return min(optimal_k, max_k)

    def _analyze_cluster_characteristics(self, cluster_labels: np.ndarray,
                                       original_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of each cluster"""
        characteristics = {}

        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue

            cluster_data = original_data[cluster_labels == cluster_id]

            characteristics[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(original_data)) * 100,
                'feature_means': cluster_data.mean().to_dict(),
                'feature_stds': cluster_data.std().to_dict(),
                'outliers': self._count_cluster_outliers(cluster_data)
            }

        return characteristics

    def _count_cluster_outliers(self, cluster_data: pd.DataFrame) -> int:
        """Count outliers within a cluster"""
        outliers = 0
        for col in cluster_data.select_dtypes(include=[np.number]).columns:
            # Simple outlier detection within cluster
            Q1 = cluster_data[col].quantile(0.25)
            Q3 = cluster_data[col].quantile(0.75)
            IQR = Q3 - Q1
            cluster_outliers = ((cluster_data[col] < (Q1 - 1.5 * IQR)) |
                               (cluster_data[col] > (Q3 + 1.5 * IQR))).sum()
            outliers += cluster_outliers

        return int(outliers)

    def perform_time_series_analysis(self, time_column: str,
                                   value_columns: List[str]) -> Dict[str, Any]:
        """Perform time series analysis"""
        logger.info("Performing time series analysis...")

        if time_column not in self.data.columns:
            logger.error(f"Time column '{time_column}' not found in data")
            return {}

        # Ensure data is sorted by time
        ts_data = self.data.sort_values(time_column).set_index(time_column)

        results = {}

        for col in value_columns:
            if col not in ts_data.columns:
                logger.warning(f"Column '{col}' not found, skipping...")
                continue

            series_results = self._analyze_time_series(ts_data[col], col)
            results[col] = series_results

        # Cross-series analysis if multiple columns
        if len(value_columns) > 1:
            results['cross_series_analysis'] = self._analyze_cross_series(ts_data[value_columns])

        self.analytics_results['time_series_analysis'] = results
        return results

    def _analyze_time_series(self, series: pd.Series, name: str) -> Dict[str, Any]:
        """Analyze individual time series"""
        results = {
            'stationarity': {},
            'decomposition': {},
            'forecasting': {},
            'anomaly_detection': {}
        }

        # Stationarity test
        try:
            adf_result = adfuller(series.dropna())
            results['stationarity'] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            }
        except:
            results['stationarity'] = {'error': 'Could not perform stationarity test'}

        # Seasonal decomposition
        try:
            if len(series.dropna()) > 24:  # Need sufficient data
                decomposition = seasonal_decompose(series.dropna(), model='additive', period=24)
                results['decomposition'] = {
                    'trend_strength': self._calculate_component_strength(decomposition.trend, series),
                    'seasonal_strength': self._calculate_component_strength(decomposition.seasonal, series),
                    'residual_strength': self._calculate_component_strength(decomposition.resid, series)
                }
        except:
            results['decomposition'] = {'error': 'Could not perform seasonal decomposition'}

        # Forecasting
        try:
            forecast_results = self._perform_time_series_forecasting(series, name)
            results['forecasting'] = forecast_results
        except Exception as e:
            results['forecasting'] = {'error': str(e)}

        # Anomaly detection
        try:
            anomalies = self._detect_time_series_anomalies(series)
            results['anomaly_detection'] = anomalies
        except Exception as e:
            results['anomaly_detection'] = {'error': str(e)}

        return results

    def _calculate_component_strength(self, component: pd.Series, original: pd.Series) -> float:
        """Calculate strength of a seasonal component"""
        if component.isnull().all():
            return 0.0

        component_var = component.var()
        original_var = original.var()

        if original_var == 0:
            return 0.0

        return max(0, 1 - component_var / original_var)

    def _perform_time_series_forecasting(self, series: pd.Series, name: str) -> Dict[str, Any]:
        """Perform time series forecasting"""
        # Simple ARIMA forecasting
        try:
            # Fit ARIMA model (simplified parameters)
            model = ARIMA(series.dropna(), order=(1, 1, 1))
            model_fit = model.fit()

            # Forecast next 10 periods
            forecast = model_fit.forecast(steps=10)

            return {
                'model': 'ARIMA(1,1,1)',
                'forecast_values': forecast.tolist(),
                'forecast_dates': [(series.index[-1] + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                                 for i in range(1, 11)],
                'model_aic': model_fit.aic,
                'model_bic': model_fit.bic
            }

        except Exception as e:
            # Fallback to exponential smoothing
            try:
                model = ExponentialSmoothing(series.dropna(), seasonal='add', seasonal_periods=7)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=10)

                return {
                    'model': 'ExponentialSmoothing',
                    'forecast_values': forecast.tolist(),
                    'forecast_dates': [(series.index[-1] + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                                     for i in range(1, 11)]
                }
            except:
                return {'error': f'Could not perform forecasting: {str(e)}'}

    def _detect_time_series_anomalies(self, series: pd.Series) -> Dict[str, Any]:
        """Detect anomalies in time series"""
        # Simple anomaly detection using statistical methods
        rolling_mean = series.rolling(window=7, center=True).mean()
        rolling_std = series.rolling(window=7, center=True).std()

        # Z-score based anomaly detection
        z_scores = np.abs((series - rolling_mean) / rolling_std)
        anomalies = z_scores > 3

        anomaly_indices = series[anomalies].index.tolist()
        anomaly_values = series[anomalies].tolist()

        return {
            'anomaly_count': len(anomaly_indices),
            'anomaly_percentage': (len(anomaly_indices) / len(series)) * 100,
            'anomaly_indices': [idx.strftime('%Y-%m-%d %H:%M:%S') for idx in anomaly_indices],
            'anomaly_values': anomaly_values,
            'method': 'z_score_rolling_window'
        }

    def _analyze_cross_series(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between multiple time series"""
        results = {
            'correlation_matrix': data.corr().to_dict(),
            'granger_causality': {},
            'coherence_analysis': {}
        }

        # Correlation analysis
        corr_matrix = data.corr()
        strong_correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'series1': corr_matrix.columns[i],
                        'series2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })

        results['strong_correlations'] = strong_correlations

        return results

    def create_visualizations(self, viz_types: List[str] = None) -> Dict[str, Any]:
        """Create comprehensive visualizations"""
        if viz_types is None:
            viz_types = ['distribution', 'correlation', 'clustering', 'time_series']

        visualizations = {}

        for viz_type in viz_types:
            try:
                if viz_type == 'distribution':
                    visualizations['distribution_plots'] = self._create_distribution_plots()
                elif viz_type == 'correlation':
                    visualizations['correlation_plot'] = self._create_correlation_plot()
                elif viz_type == 'clustering':
                    visualizations['cluster_plots'] = self._create_cluster_plots()
                elif viz_type == 'time_series':
                    visualizations['time_series_plots'] = self._create_time_series_plots()
                elif viz_type == 'healthcare_dashboard':
                    visualizations['healthcare_dashboard'] = self._create_healthcare_dashboard()
            except Exception as e:
                logger.error(f"Error creating {viz_type} visualization: {str(e)}")
                visualizations[viz_type] = {'error': str(e)}

        self.visualizations = visualizations
        self.analytics_results['visualizations'] = visualizations

        return visualizations

    def _create_distribution_plots(self) -> Dict[str, Any]:
        """Create distribution plots for numeric variables"""
        plots = {}

        numeric_columns = self.data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns[:5]:  # Limit to first 5 for performance
            # Histogram with KDE
            fig = px.histogram(
                self.data, x=col,
                title=f'Distribution of {col}',
                marginal='box',
                opacity=0.7
            )

            plots[f'{col}_distribution'] = fig.to_json()

            # Q-Q plot for normality check
            try:
                qq_fig = ff.create_qq_plot(self.data[col].dropna(), x=np.random.normal(
                    loc=self.data[col].mean(),
                    scale=self.data[col].std(),
                    size=len(self.data[col].dropna())
                ))
                qq_fig.update_layout(title=f'Q-Q Plot for {col}')
                plots[f'{col}_qq_plot'] = qq_fig.to_json()
            except:
                pass

        return plots

    def _create_correlation_plot(self) -> str:
        """Create correlation heatmap"""
        numeric_data = self.data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] < 2:
            return {'error': 'Need at least 2 numeric columns for correlation plot'}

        corr_matrix = numeric_data.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )

        return fig.to_json()

    def _create_cluster_plots(self) -> Dict[str, Any]:
        """Create clustering visualization plots"""
        plots = {}

        if 'cluster' not in self.data.columns:
            return {'error': 'No cluster column found. Run clustering analysis first.'}

        numeric_columns = self.data.select_dtypes(include=[np.number]).columns[:3]  # Use first 3 numeric columns

        if len(numeric_columns) >= 2:
            # 2D scatter plot
            fig = px.scatter(
                self.data,
                x=numeric_columns[0],
                y=numeric_columns[1],
                color='cluster',
                title=f'Clusters: {numeric_columns[0]} vs {numeric_columns[1]}',
                opacity=0.7
            )
            plots['cluster_scatter_2d'] = fig.to_json()

        if len(numeric_columns) >= 3:
            # 3D scatter plot
            fig = px.scatter_3d(
                self.data,
                x=numeric_columns[0],
                y=numeric_columns[1],
                z=numeric_columns[2],
                color='cluster',
                title=f'3D Clusters: {numeric_columns[0]}, {numeric_columns[1]}, {numeric_columns[2]}',
                opacity=0.7
            )
            plots['cluster_scatter_3d'] = fig.to_json()

        # Cluster sizes bar chart
        cluster_sizes = self.data['cluster'].value_counts().sort_index()

        fig = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            title='Cluster Sizes',
            labels={'x': 'Cluster', 'y': 'Number of Samples'}
        )
        plots['cluster_sizes'] = fig.to_json()

        return plots

    def _create_time_series_plots(self) -> Dict[str, Any]:
        """Create time series visualization plots"""
        plots = {}

        date_columns = self._identify_date_columns()

        if not date_columns:
            return {'error': 'No date columns found for time series plotting'}

        time_col = date_columns[0]
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns[:3]

        for col in numeric_columns:
            try:
                # Time series line plot
                fig = px.line(
                    self.data.sort_values(time_col),
                    x=time_col,
                    y=col,
                    title=f'Time Series: {col}',
                    markers=True
                )
                plots[f'{col}_time_series'] = fig.to_json()

                # Rolling statistics
                rolling_data = self.data.set_index(time_col)[col].rolling(window=7).agg(['mean', 'std'])

                fig = make_subplots(rows=2, cols=1,
                                  subplot_titles=[f'{col} - Rolling Mean (7-day)',
                                                f'{col} - Rolling Std (7-day)'])

                fig.add_trace(
                    go.Scatter(x=rolling_data.index, y=rolling_data['mean'],
                             mode='lines', name='Rolling Mean'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=rolling_data.index, y=rolling_data['std'],
                             mode='lines', name='Rolling Std'),
                    row=2, col=1
                )

                fig.update_layout(title=f'Rolling Statistics: {col}')
                plots[f'{col}_rolling_stats'] = fig.to_json()

            except Exception as e:
                plots[f'{col}_error'] = str(e)

        return plots

    def _create_healthcare_dashboard(self) -> Dict[str, Any]:
        """Create healthcare-specific dashboard visualizations"""
        dashboard = {}

        # Health metrics overview
        if 'bmi' in self.data.columns:
            bmi_stats = self.data['bmi'].describe()
            dashboard['bmi_distribution'] = {
                'mean': bmi_stats['mean'],
                'median': bmi_stats['50%'],
                'categories': self.data['bmi_category'].value_counts().to_dict() if 'bmi_category' in self.data.columns else {}
            }

        # Risk score distributions
        risk_columns = [col for col in self.data.columns if 'risk' in col.lower()]
        for risk_col in risk_columns[:3]:  # Limit to 3 risk columns
            fig = px.histogram(
                self.data, x=risk_col,
                title=f'Distribution of {risk_col}',
                opacity=0.7
            )
            dashboard[f'{risk_col}_distribution'] = fig.to_json()

        # Health correlations
        if 'strong_correlations' in self.analytics_results:
            correlations = self.analytics_results['strong_correlations']
            if correlations:
                corr_df = pd.DataFrame(correlations)
                fig = px.bar(
                    corr_df,
                    x='correlation',
                    y=['variable1', 'variable2'],
                    orientation='h',
                    title='Strong Health Variable Correlations'
                )
                dashboard['health_correlations'] = fig.to_json()

        return dashboard

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        report = {
            'summary': {
                'data_shape': self.data.shape if self.data is not None else None,
                'analysis_timestamp': datetime.now().isoformat(),
                'analytics_performed': list(self.analytics_results.keys())
            },
            'data_quality': {},
            'statistical_insights': {},
            'patterns_discovered': {},
            'recommendations': [],
            'visualizations_summary': {}
        }

        # Data quality summary
        if self.data is not None:
            report['data_quality'] = {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'missing_data_percentage': (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100,
                'duplicate_rows': self.data.duplicated().sum()
            }

        # Statistical insights
        if 'statistical_analysis' in self.analytics_results:
            stats = self.analytics_results['statistical_analysis']
            report['statistical_insights'] = {
                'normal_distributions': len([col for col, analysis in stats.get('distribution_analysis', {}).items()
                                           if analysis.get('is_normal', False)]),
                'outlier_percentage': np.mean([analysis.get('outlier_percentage', 0)
                                             for analysis in stats.get('outlier_analysis', {}).values()]),
                'significant_tests': len([test for test in stats.get('inferential_stats', {}).get('anova_tests', {}).values()
                                        if test.get('significant', False)])
            }

        # Patterns discovered
        patterns = {}
        if 'strong_correlations' in self.analytics_results:
            patterns['correlations'] = len(self.analytics_results['strong_correlations'])

        if 'clustering_analysis' in self.analytics_results:
            cluster_info = self.analytics_results['clustering_analysis']
            patterns['clusters'] = cluster_info.get('n_clusters', 0)
            patterns['clustering_quality'] = cluster_info.get('evaluation_metrics', {}).get('silhouette_score')

        if 'time_series_analysis' in self.analytics_results:
            ts_patterns = []
            for col, analysis in self.analytics_results['time_series_analysis'].items():
                if isinstance(analysis, dict) and 'anomaly_detection' in analysis:
                    ts_patterns.append({
                        'series': col,
                        'anomalies': analysis['anomaly_detection'].get('anomaly_count', 0)
                    })
            patterns['time_series_anomalies'] = ts_patterns

        report['patterns_discovered'] = patterns

        # Generate recommendations
        recommendations = []

        # Data quality recommendations
        if report['data_quality'].get('missing_data_percentage', 0) > 10:
            recommendations.append("High percentage of missing data detected. Consider data imputation or collection improvements.")

        if report['data_quality'].get('duplicate_rows', 0) > 0:
            recommendations.append(f"Found {report['data_quality']['duplicate_rows']} duplicate rows. Consider data deduplication.")

        # Statistical recommendations
        if report['statistical_insights'].get('normal_distributions', 0) == 0:
            recommendations.append("No normally distributed variables found. Consider data transformation for parametric tests.")

        if report['statistical_insights'].get('outlier_percentage', 0) > 5:
            recommendations.append("High percentage of outliers detected. Consider robust statistical methods or outlier treatment.")

        # Pattern-based recommendations
        if patterns.get('clusters', 0) > 0:
            recommendations.append(f"Identified {patterns['clusters']} distinct patient clusters. Consider personalized treatment approaches.")

        if patterns.get('correlations', 0) > 0:
            recommendations.append(f"Found {patterns['correlations']} strong variable correlations. These may indicate causal relationships worth investigating.")

        anomaly_series = [p for p in patterns.get('time_series_anomalies', []) if p.get('anomalies', 0) > 0]
        if anomaly_series:
            recommendations.append(f"Time series anomalies detected in {len(anomaly_series)} variables. Monitor for clinical significance.")

        report['recommendations'] = recommendations

        # Visualization summary
        if self.visualizations:
            report['visualizations_summary'] = {
                'total_plots': sum(len(plots) for plots in self.visualizations.values() if isinstance(plots, dict)),
                'plot_types': list(self.visualizations.keys())
            }

        return report

    def export_results(self, export_path: str, formats: List[str] = None):
        """Export analytics results in multiple formats"""
        if formats is None:
            formats = ['json', 'csv']

        export_data = {
            'analytics_results': self.analytics_results,
            'comprehensive_report': self.generate_comprehensive_report(),
            'export_timestamp': datetime.now().isoformat()
        }

        for fmt in formats:
            try:
                if fmt == 'json':
                    with open(f"{export_path}/analytics_results.json", 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)

                elif fmt == 'csv':
                    # Export processed data
                    if self.processed_data is not None:
                        self.processed_data.to_csv(f"{export_path}/processed_data.csv", index=False)

                    # Export key metrics as CSV
                    if 'statistical_analysis' in self.analytics_results:
                        stats_df = pd.DataFrame.from_dict(
                            self.analytics_results['statistical_analysis']['descriptive_stats']
                        )
                        stats_df.to_csv(f"{export_path}/statistical_summary.csv")

            except Exception as e:
                logger.error(f"Error exporting to {fmt}: {str(e)}")

        logger.info(f"Results exported to {export_path}")


# Utility functions for healthcare analytics
def create_healthcare_analytics_pipeline(config: Dict[str, Any]) -> HealthcareDataAnalytics:
    """Create a pre-configured healthcare analytics pipeline"""

    default_config = {
        'missing_threshold': 0.5,
        'pca_components': 0.95,
        'test_size': 0.2,
        'random_state': 42,
        'time_series_freq': '1H',
        'clustering_method': 'kmeans',
        'max_clusters': 10
    }

    default_config.update(config)

    return HealthcareDataAnalytics(default_config)


def analyze_patient_cohort(data: pd.DataFrame, cohort_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze specific patient cohorts"""

    # Apply cohort criteria
    cohort_mask = pd.Series([True] * len(data))

    for column, criteria in cohort_criteria.items():
        if column in data.columns:
            if isinstance(criteria, dict):
                if 'min' in criteria:
                    cohort_mask &= (data[column] >= criteria['min'])
                if 'max' in criteria:
                    cohort_mask &= (data[column] <= criteria['max'])
                if 'values' in criteria:
                    cohort_mask &= data[column].isin(criteria['values'])
            else:
                cohort_mask &= (data[column] == criteria)

    cohort_data = data[cohort_mask]

    analysis = {
        'cohort_size': len(cohort_data),
        'cohort_percentage': (len(cohort_data) / len(data)) * 100,
        'demographics': {},
        'health_metrics': {},
        'risk_profiles': {}
    }

    # Analyze cohort demographics
    if 'age' in cohort_data.columns:
        analysis['demographics']['age_stats'] = cohort_data['age'].describe().to_dict()

    if 'gender' in cohort_data.columns:
        analysis['demographics']['gender_distribution'] = cohort_data['gender'].value_counts().to_dict()

    # Analyze health metrics
    health_columns = ['bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                     'cholesterol', 'glucose', 'heart_rate']

    for col in health_columns:
        if col in cohort_data.columns:
            analysis['health_metrics'][col] = {
                'mean': cohort_data[col].mean(),
                'median': cohort_data[col].median(),
                'std': cohort_data[col].std(),
                'normal_range_percentage': calculate_normal_range_percentage(cohort_data[col], col)
            }

    # Analyze risk profiles
    risk_columns = [col for col in cohort_data.columns if 'risk' in col.lower()]
    for col in risk_columns:
        analysis['risk_profiles'][col] = {
            'mean_risk': cohort_data[col].mean(),
            'high_risk_percentage': (cohort_data[col] > 0.7).mean() * 100,
            'distribution': cohort_data[col].describe().to_dict()
        }

    return analysis


def calculate_normal_range_percentage(series: pd.Series, metric: str) -> float:
    """Calculate percentage of values in normal range for health metrics"""

    normal_ranges = {
        'bmi': (18.5, 25),
        'blood_pressure_systolic': (90, 140),
        'blood_pressure_diastolic': (60, 90),
        'cholesterol': (0, 200),
        'glucose': (70, 140),
        'heart_rate': (60, 100)
    }

    if metric in normal_ranges:
        min_val, max_val = normal_ranges[metric]
        in_range = ((series >= min_val) & (series <= max_val)).sum()
        return (in_range / len(series)) * 100

    return 0.0


def perform_comparative_analysis(data: pd.DataFrame, group_column: str,
                               metrics: List[str]) -> Dict[str, Any]:
    """Perform comparative analysis between groups"""

    if group_column not in data.columns:
        return {'error': f'Group column {group_column} not found'}

    groups = data.groupby(group_column)
    comparison_results = {}

    for metric in metrics:
        if metric not in data.columns:
            continue

        group_stats = groups[metric].agg(['mean', 'std', 'median', 'count'])

        # Perform statistical tests
        group_values = [group[metric].dropna() for name, group in groups]
        if len(group_values) > 1:
            try:
                # ANOVA test
                f_stat, p_value = stats.f_oneway(*group_values)

                comparison_results[metric] = {
                    'group_statistics': group_stats.to_dict(),
                    'anova_f_statistic': f_stat,
                    'anova_p_value': p_value,
                    'significant_difference': p_value < 0.05
                }

                # Post-hoc tests if significant
                if p_value < 0.05 and len(group_values) > 2:
                    # Tukey HSD test (simplified)
                    comparison_results[metric]['post_hoc'] = 'significant differences found between groups'

            except:
                comparison_results[metric] = {
                    'group_statistics': group_stats.to_dict(),
                    'statistical_test': 'failed'
                }

    return comparison_results


def create_predictive_healthcare_model(data: pd.DataFrame, target_column: str,
                                     feature_columns: List[str],
                                     model_type: str = 'classification') -> Dict[str, Any]:
    """Create predictive healthcare models"""

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import classification_report, mean_squared_error, r2_score

    # Prepare data
    X = data[feature_columns]
    y = data[target_column]

    # Handle missing values
    X = X.fillna(X.mean())
    if y.dtype == 'object':
        y = y.fillna(y.mode().iloc[0])
    else:
        y = y.fillna(y.mean())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_results = {
        'model_type': model_type,
        'target_column': target_column,
        'feature_columns': feature_columns,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    if model_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        model_results.update({
            'model': 'RandomForestClassifier',
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        })

    elif model_type == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        model_results.update({
            'model': 'RandomForestRegressor',
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred),
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        })

    return model_results


# Example usage and testing
def example_healthcare_analytics_workflow():
    """Example workflow for healthcare data analytics"""

    # Create sample healthcare data
    np.random.seed(42)
    n_patients = 1000

    healthcare_data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.normal(50, 15, n_patients),
        'gender': np.random.choice(['male', 'female'], n_patients),
        'bmi': np.random.normal(27, 5, n_patients),
        'blood_pressure_systolic': np.random.normal(130, 20, n_patients),
        'blood_pressure_diastolic': np.random.normal(85, 15, n_patients),
        'cholesterol': np.random.normal(200, 40, n_patients),
        'glucose': np.random.normal(100, 25, n_patients),
        'heart_rate': np.random.normal(75, 10, n_patients),
        'smoking_status': np.random.choice(['never', 'former', 'current'], n_patients),
        'diabetes_status': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
        'hypertension_status': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'cardiovascular_risk': np.random.normal(0.3, 0.2, n_patients)
    }

    df = pd.DataFrame(healthcare_data)

    # Initialize analytics pipeline
    config = {
        'missing_threshold': 0.3,
        'pca_components': 0.9
    }

    analytics = HealthcareDataAnalytics(config)

    # Load and process data
    processed_data = analytics.load_and_process_data(df, data_type='healthcare')

    # Perform statistical analysis
    statistical_results = analytics.perform_statistical_analysis()

    # Perform clustering analysis
    clustering_results = analytics.perform_clustering_analysis(n_clusters=4, method='kmeans')

    # Create visualizations
    visualizations = analytics.create_visualizations(['distribution', 'correlation', 'clustering'])

    # Generate comprehensive report
    report = analytics.generate_comprehensive_report()

    # Analyze patient cohorts
    diabetic_cohort = analyze_patient_cohort(df, {'diabetes_status': 1})
    high_risk_cohort = analyze_patient_cohort(df, {'cardiovascular_risk': {'min': 0.7}})

    # Comparative analysis
    gender_comparison = perform_comparative_analysis(df, 'gender', ['bmi', 'blood_pressure_systolic', 'cholesterol'])

    # Create predictive model
    prediction_model = create_predictive_healthcare_model(
        df, 'cardiovascular_risk',
        ['age', 'bmi', 'blood_pressure_systolic', 'cholesterol', 'glucose'],
        model_type='regression'
    )

    return {
        'processed_data_shape': processed_data.shape,
        'statistical_insights': len(statistical_results.get('distribution_analysis', {})),
        'clusters_found': clustering_results.get('n_clusters', 0),
        'visualizations_created': len(visualizations),
        'cohort_analyses': {
            'diabetic_patients': diabetic_cohort['cohort_size'],
            'high_risk_patients': high_risk_cohort['cohort_size']
        },
        'comparative_analysis': len(gender_comparison),
        'prediction_model_performance': prediction_model.get('r2_score', 'N/A')
    }


if __name__ == "__main__":
    # Run example workflow
    results = example_healthcare_analytics_workflow()

    print("Healthcare Analytics Pipeline Example Results:")
    print(f"Processed {results['processed_data_shape'][0]} patients with {results['processed_data_shape'][1]} features")
    print(f"Identified {results['clusters_found']} patient clusters")
    print(f"Created {results['visualizations_created']} visualization types")
    print(f"Diabetic cohort: {results['cohort_analyses']['diabetic_patients']} patients")
    print(f"High-risk cohort: {results['cohort_analyses']['high_risk_patients']} patients")
    print(f"Prediction model R² score: {results['prediction_model_performance']:.3f}")

    # Export results
    # analytics.export_results('./analytics_output')
