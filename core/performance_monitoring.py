"""
Performance Monitoring and Analytics System for AI Personalized Medicine Platform
Real-time performance tracking, bottleneck detection, and optimization recommendations
"""

import time
import threading
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import cProfile
import pstats
import io
import gc
import inspect
import functools

class PerformanceProfiler:
    """Advanced performance profiling system"""

    def __init__(self):
        self.profilers = {}
        self.performance_data = defaultdict(dict)
        self.baseline_metrics = {}
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.is_profiling = False
        self.profiling_workers = []

    def start_profiling(self):
        """Start comprehensive performance profiling"""
        self.is_profiling = True

        # Start memory profiling
        tracemalloc.start()

        # Start CPU profiling worker
        cpu_worker = threading.Thread(target=self._cpu_profiling_loop, daemon=True)
        cpu_worker.start()
        self.profiling_workers.append(cpu_worker)

        # Start memory profiling worker
        memory_worker = threading.Thread(target=self._memory_profiling_loop, daemon=True)
        memory_worker.start()
        self.profiling_workers.append(memory_worker)

        # Start I/O profiling worker
        io_worker = threading.Thread(target=self._io_profiling_loop, daemon=True)
        io_worker.start()
        self.profiling_workers.append(io_worker)

    def stop_profiling(self):
        """Stop performance profiling"""
        self.is_profiling = False
        tracemalloc.stop()

    def profile_function(self, func_name: str = None) -> Callable:
        """Decorator to profile function performance"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss

                    execution_time = end_time - start_time
                    memory_used = end_memory - start_memory

                    function_name = func_name or f"{func.__module__}.{func.__name__}"

                    self.record_function_performance(
                        function_name,
                        execution_time,
                        memory_used,
                        len(args),
                        len(kwargs)
                    )
            return wrapper
        return decorator

    def record_function_performance(self, function_name: str, execution_time: float,
                                  memory_used: int, args_count: int, kwargs_count: int):
        """Record function performance metrics"""
        performance_record = {
            "timestamp": datetime.now(),
            "function_name": function_name,
            "execution_time": execution_time,
            "memory_used": memory_used,
            "args_count": args_count,
            "kwargs_count": kwargs_count,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }

        self.performance_history[function_name].append(performance_record)

        # Update current performance data
        self.performance_data[function_name] = {
            "last_execution": performance_record,
            "avg_execution_time": self._calculate_average_execution_time(function_name),
            "avg_memory_usage": self._calculate_average_memory_usage(function_name),
            "call_count": len(self.performance_history[function_name])
        }

    def _calculate_average_execution_time(self, function_name: str) -> float:
        """Calculate average execution time for a function"""
        if not self.performance_history[function_name]:
            return 0

        execution_times = [record["execution_time"] for record in self.performance_history[function_name]]
        return statistics.mean(execution_times)

    def _calculate_average_memory_usage(self, function_name: str) -> float:
        """Calculate average memory usage for a function"""
        if not self.performance_history[function_name]:
            return 0

        memory_usage = [record["memory_used"] for record in self.performance_history[function_name]]
        return statistics.mean(memory_usage)

    def _cpu_profiling_loop(self):
        """Continuous CPU profiling"""
        while self.is_profiling:
            try:
                # Profile CPU usage patterns
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_freq = psutil.cpu_freq()

                cpu_data = {
                    "timestamp": datetime.now(),
                    "cpu_percent": cpu_percent,
                    "cpu_freq_current": cpu_freq.current if cpu_freq else 0,
                    "cpu_freq_min": cpu_freq.min if cpu_freq else 0,
                    "cpu_freq_max": cpu_freq.max if cpu_freq else 0
                }

                self.performance_history["system.cpu"].append(cpu_data)
                time.sleep(5)

            except Exception as e:
                print(f"CPU profiling error: {e}")

    def _memory_profiling_loop(self):
        """Continuous memory profiling"""
        while self.is_profiling:
            try:
                # Profile memory usage
                memory = psutil.virtual_memory()
                process_memory = psutil.Process().memory_info()

                memory_data = {
                    "timestamp": datetime.now(),
                    "system_memory_percent": memory.percent,
                    "system_memory_used": memory.used,
                    "system_memory_available": memory.available,
                    "process_memory_rss": process_memory.rss,
                    "process_memory_vms": process_memory.vms,
                    "process_memory_percent": psutil.Process().memory_percent()
                }

                # Add tracemalloc data if available
                if tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    memory_data["tracemalloc_current"] = current
                    memory_data["tracemalloc_peak"] = peak

                self.performance_history["system.memory"].append(memory_data)
                time.sleep(10)

            except Exception as e:
                print(f"Memory profiling error: {e}")

    def _io_profiling_loop(self):
        """Continuous I/O profiling"""
        while self.is_profiling:
            try:
                # Profile disk I/O
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()

                io_data = {
                    "timestamp": datetime.now(),
                    "disk_read_bytes": disk_io.read_bytes if disk_io else 0,
                    "disk_write_bytes": disk_io.write_bytes if disk_io else 0,
                    "disk_read_count": disk_io.read_count if disk_io else 0,
                    "disk_write_count": disk_io.write_count if disk_io else 0,
                    "network_bytes_sent": net_io.bytes_sent if net_io else 0,
                    "network_bytes_recv": net_io.bytes_recv if net_io else 0,
                    "network_packets_sent": net_io.packets_sent if net_io else 0,
                    "network_packets_recv": net_io.packets_recv if net_io else 0
                }

                self.performance_history["system.io"].append(io_data)
                time.sleep(15)

            except Exception as e:
                print(f"I/O profiling error: {e}")

    def get_function_performance_report(self, function_name: str = None) -> Dict[str, Any]:
        """Get performance report for functions"""
        if function_name:
            if function_name not in self.performance_data:
                return {"error": f"No performance data for function {function_name}"}

            return self.performance_data[function_name]
        else:
            # Return summary for all functions
            return {
                "function_count": len(self.performance_data),
                "functions": list(self.performance_data.keys()),
                "slowest_functions": self._get_slowest_functions(),
                "memory_intensive_functions": self._get_memory_intensive_functions(),
                "most_called_functions": self._get_most_called_functions()
            }

    def _get_slowest_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest functions by average execution time"""
        function_times = []
        for func_name, data in self.performance_data.items():
            if "avg_execution_time" in data:
                function_times.append({
                    "function": func_name,
                    "avg_time": data["avg_execution_time"],
                    "call_count": data["call_count"]
                })

        return sorted(function_times, key=lambda x: x["avg_time"], reverse=True)[:limit]

    def _get_memory_intensive_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most memory-intensive functions"""
        function_memory = []
        for func_name, data in self.performance_data.items():
            if "avg_memory_usage" in data:
                function_memory.append({
                    "function": func_name,
                    "avg_memory": data["avg_memory_usage"],
                    "call_count": data["call_count"]
                })

        return sorted(function_memory, key=lambda x: x["avg_memory"], reverse=True)[:limit]

    def _get_most_called_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently called functions"""
        function_calls = []
        for func_name, data in self.performance_data.items():
            function_calls.append({
                "function": func_name,
                "call_count": data["call_count"],
                "avg_time": data.get("avg_execution_time", 0)
            })

        return sorted(function_calls, key=lambda x: x["call_count"], reverse=True)[:limit]

    def get_system_performance_report(self) -> Dict[str, Any]:
        """Get system performance report"""
        return {
            "cpu_performance": self._analyze_cpu_performance(),
            "memory_performance": self._analyze_memory_performance(),
            "io_performance": self._analyze_io_performance(),
            "performance_trends": self._analyze_performance_trends(),
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_performance_recommendations()
        }

    def _analyze_cpu_performance(self) -> Dict[str, Any]:
        """Analyze CPU performance data"""
        cpu_data = list(self.performance_history.get("system.cpu", []))

        if not cpu_data:
            return {"error": "No CPU data available"}

        cpu_percents = [d["cpu_percent"] for d in cpu_data[-100:]]  # Last 100 readings

        return {
            "current_cpu_percent": cpu_percents[-1] if cpu_percents else 0,
            "avg_cpu_percent": statistics.mean(cpu_percents) if cpu_percents else 0,
            "max_cpu_percent": max(cpu_percents) if cpu_percents else 0,
            "cpu_utilization_trend": self._calculate_trend(cpu_percents),
            "cpu_frequency": cpu_data[-1].get("cpu_freq_current", 0) if cpu_data else 0
        }

    def _analyze_memory_performance(self) -> Dict[str, Any]:
        """Analyze memory performance data"""
        memory_data = list(self.performance_history.get("system.memory", []))

        if not memory_data:
            return {"error": "No memory data available"}

        recent_data = memory_data[-50:]  # Last 50 readings

        return {
            "current_memory_percent": recent_data[-1]["system_memory_percent"] if recent_data else 0,
            "avg_memory_percent": statistics.mean([d["system_memory_percent"] for d in recent_data]) if recent_data else 0,
            "process_memory_mb": recent_data[-1]["process_memory_rss"] / 1024 / 1024 if recent_data else 0,
            "memory_trend": self._calculate_trend([d["system_memory_percent"] for d in recent_data]),
            "memory_leaks_detected": self._detect_memory_leaks(recent_data)
        }

    def _analyze_io_performance(self) -> Dict[str, Any]:
        """Analyze I/O performance data"""
        io_data = list(self.performance_history.get("system.io", []))

        if not io_data or len(io_data) < 2:
            return {"error": "Insufficient I/O data"}

        # Calculate rates
        recent_data = io_data[-20:]  # Last 20 readings
        time_diffs = [(recent_data[i+1]["timestamp"] - recent_data[i]["timestamp"]).total_seconds()
                     for i in range(len(recent_data)-1)]

        disk_read_rates = []
        disk_write_rates = []
        network_send_rates = []
        network_recv_rates = []

        for i in range(len(recent_data)-1):
            time_diff = time_diffs[i]
            if time_diff > 0:
                disk_read_rates.append((recent_data[i+1]["disk_read_bytes"] - recent_data[i]["disk_read_bytes"]) / time_diff)
                disk_write_rates.append((recent_data[i+1]["disk_write_bytes"] - recent_data[i]["disk_write_bytes"]) / time_diff)
                network_send_rates.append((recent_data[i+1]["network_bytes_sent"] - recent_data[i]["network_bytes_sent"]) / time_diff)
                network_recv_rates.append((recent_data[i+1]["network_bytes_recv"] - recent_data[i]["network_bytes_recv"]) / time_diff)

        return {
            "disk_read_rate_avg": statistics.mean(disk_read_rates) if disk_read_rates else 0,
            "disk_write_rate_avg": statistics.mean(disk_write_rates) if disk_write_rates else 0,
            "network_send_rate_avg": statistics.mean(network_send_rates) if network_send_rates else 0,
            "network_recv_rate_avg": statistics.mean(network_recv_rates) if network_recv_rates else 0,
            "io_efficiency_score": self._calculate_io_efficiency(io_data)
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 10:
            return "insufficient_data"

        # Compare first half with second half
        midpoint = len(values) // 2
        first_half = values[:midpoint]
        second_half = values[midpoint:]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _detect_memory_leaks(self, memory_data: List[Dict[str, Any]]) -> bool:
        """Detect potential memory leaks"""
        if len(memory_data) < 20:
            return False

        # Check if memory usage is consistently increasing
        memory_values = [d["process_memory_rss"] for d in memory_data]
        trend = self._calculate_trend(memory_values)

        # Check for significant growth without corresponding drops
        recent_avg = statistics.mean(memory_values[-10:])
        earlier_avg = statistics.mean(memory_values[:10])

        growth_ratio = recent_avg / earlier_avg if earlier_avg > 0 else 1

        return trend == "increasing" and growth_ratio > 1.5

    def _calculate_io_efficiency(self, io_data: List[Dict[str, Any]]) -> float:
        """Calculate I/O efficiency score (0-100)"""
        if len(io_data) < 5:
            return 50.0  # Neutral score

        # Simple efficiency based on read/write balance and consistency
        read_counts = [d["disk_read_count"] for d in io_data[-10:]]
        write_counts = [d["disk_write_count"] for d in io_data[-10:]]

        # Prefer balanced read/write operations
        total_operations = sum(read_counts) + sum(write_counts)
        if total_operations == 0:
            return 50.0

        read_ratio = sum(read_counts) / total_operations
        balance_score = 100 - abs(read_ratio - 0.5) * 200  # Prefer 50/50 balance

        # Consistency score
        read_std = statistics.stdev(read_counts) if len(read_counts) > 1 else 0
        write_std = statistics.stdev(write_counts) if len(write_counts) > 1 else 0
        consistency_score = max(0, 100 - (read_std + write_std) / 10)

        return (balance_score + consistency_score) / 2

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        return {
            "cpu_trend": self._calculate_trend([d["cpu_percent"] for d in self.performance_history.get("system.cpu", [])]),
            "memory_trend": self._calculate_trend([d["system_memory_percent"] for d in self.performance_history.get("system.memory", [])]),
            "performance_stability": self._calculate_performance_stability(),
            "peak_usage_times": self._identify_peak_usage_times()
        }

    def _calculate_performance_stability(self) -> float:
        """Calculate performance stability score (0-100)"""
        # Measure consistency of performance metrics
        cpu_data = [d["cpu_percent"] for d in self.performance_history.get("system.cpu", [])]
        memory_data = [d["system_memory_percent"] for d in self.performance_history.get("system.memory", [])]

        stability_scores = []

        if cpu_data:
            cpu_std = statistics.stdev(cpu_data) if len(cpu_data) > 1 else 0
            cpu_stability = max(0, 100 - cpu_std * 2)  # Lower std dev = higher stability
            stability_scores.append(cpu_stability)

        if memory_data:
            memory_std = statistics.stdev(memory_data) if len(memory_data) > 1 else 0
            memory_stability = max(0, 100 - memory_std)
            stability_scores.append(memory_stability)

        return statistics.mean(stability_scores) if stability_scores else 50.0

    def _identify_peak_usage_times(self) -> List[Dict[str, Any]]:
        """Identify peak usage time periods"""
        cpu_data = self.performance_history.get("system.cpu", [])
        if not cpu_data:
            return []

        # Find periods with high CPU usage
        peak_periods = []
        for i, data_point in enumerate(cpu_data):
            if data_point["cpu_percent"] > 80:  # High usage threshold
                peak_periods.append({
                    "timestamp": data_point["timestamp"],
                    "cpu_percent": data_point["cpu_percent"],
                    "duration_minutes": 1  # Assuming 1-minute intervals
                })

        return peak_periods[-10:]  # Return last 10 peak periods

    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify system and application bottlenecks"""
        bottlenecks = []

        # CPU bottlenecks
        cpu_analysis = self._analyze_cpu_performance()
        if cpu_analysis.get("current_cpu_percent", 0) > 90:
            bottlenecks.append({
                "type": "cpu",
                "severity": "critical",
                "description": f"CPU usage at {cpu_analysis['current_cpu_percent']:.1f}%",
                "impact": "High",
                "recommendation": "Scale CPU resources or optimize compute-intensive operations"
            })

        # Memory bottlenecks
        memory_analysis = self._analyze_memory_performance()
        if memory_analysis.get("current_memory_percent", 0) > 95:
            bottlenecks.append({
                "type": "memory",
                "severity": "critical",
                "description": f"Memory usage at {memory_analysis['current_memory_percent']:.1f}%",
                "impact": "High",
                "recommendation": "Increase memory allocation or optimize memory usage"
            })

        # Function bottlenecks
        slow_functions = self._get_slowest_functions(5)
        for func in slow_functions:
            if func["avg_time"] > 1.0:  # More than 1 second
                bottlenecks.append({
                    "type": "function",
                    "severity": "medium",
                    "description": f"Function {func['function']} averaging {func['avg_time']:.2f}s",
                    "impact": "Medium",
                    "recommendation": "Optimize function performance or implement caching"
                })

        return bottlenecks

    def _generate_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations"""
        recommendations = []

        # Memory optimization
        memory_analysis = self._analyze_memory_performance()
        if memory_analysis.get("memory_leaks_detected"):
            recommendations.append({
                "category": "memory",
                "priority": "high",
                "recommendation": "Fix memory leaks detected in application",
                "actions": ["Run memory profiling", "Implement garbage collection optimizations", "Fix circular references"]
            })

        # CPU optimization
        cpu_analysis = self._analyze_cpu_performance()
        if cpu_analysis.get("cpu_utilization_trend") == "increasing":
            recommendations.append({
                "category": "cpu",
                "priority": "medium",
                "recommendation": "Monitor increasing CPU usage trend",
                "actions": ["Optimize compute-intensive algorithms", "Consider horizontal scaling", "Implement CPU profiling"]
            })

        # I/O optimization
        io_analysis = self._analyze_io_performance()
        if io_analysis.get("io_efficiency_score", 50) < 60:
            recommendations.append({
                "category": "io",
                "priority": "low",
                "recommendation": "Improve I/O efficiency",
                "actions": ["Implement caching", "Optimize database queries", "Use asynchronous I/O"]
            })

        # Function optimization
        slow_functions = self._get_slowest_functions(3)
        for func in slow_functions:
            if func["avg_time"] > 0.5:  # More than 500ms
                recommendations.append({
                    "category": "function",
                    "priority": "medium",
                    "recommendation": f"Optimize slow function: {func['function']}",
                    "actions": ["Profile function execution", "Implement caching", "Consider algorithm optimization"]
                })

        return recommendations

class APMSystem:
    """Application Performance Monitoring System"""

    def __init__(self):
        self.transactions = defaultdict(list)
        self.apm_data = defaultdict(dict)
        self.transaction_traces = {}
        self.is_monitoring = False
        self.monitoring_workers = []

    def start_apm_monitoring(self):
        """Start APM monitoring"""
        self.is_monitoring = True

        # Start transaction monitoring
        transaction_worker = threading.Thread(target=self._monitor_transactions, daemon=True)
        transaction_worker.start()
        self.monitoring_workers.append(transaction_worker)

        # Start performance analysis
        analysis_worker = threading.Thread(target=self._analyze_performance, daemon=True)
        analysis_worker.start()
        self.monitoring_workers.append(analysis_worker)

    def stop_apm_monitoring(self):
        """Stop APM monitoring"""
        self.is_monitoring = False

    def record_transaction(self, transaction_id: str, transaction_type: str,
                          start_time: datetime, duration: float, success: bool,
                          metadata: Dict[str, Any] = None):
        """Record a transaction"""
        transaction = {
            "transaction_id": transaction_id,
            "type": transaction_type,
            "start_time": start_time,
            "duration": duration,
            "success": success,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }

        self.transactions[transaction_type].append(transaction)
        self.transaction_traces[transaction_id] = transaction

        # Keep only recent transactions
        if len(self.transactions[transaction_type]) > 1000:
            self.transactions[transaction_type] = self.transactions[transaction_type][-1000:]

    def get_transaction_metrics(self, transaction_type: str = None) -> Dict[str, Any]:
        """Get transaction performance metrics"""
        if transaction_type:
            transactions = self.transactions.get(transaction_type, [])
            if not transactions:
                return {"error": f"No transactions found for type {transaction_type}"}

            return self._calculate_transaction_metrics(transactions, transaction_type)
        else:
            # Return summary for all transaction types
            all_metrics = {}
            for tx_type in self.transactions:
                all_metrics[tx_type] = self._calculate_transaction_metrics(
                    self.transactions[tx_type], tx_type
                )

            return {
                "transaction_types": list(self.transactions.keys()),
                "metrics_by_type": all_metrics,
                "overall_metrics": self._calculate_overall_metrics()
            }

    def _calculate_transaction_metrics(self, transactions: List[Dict[str, Any]], tx_type: str) -> Dict[str, Any]:
        """Calculate metrics for a set of transactions"""
        if not transactions:
            return {}

        durations = [tx["duration"] for tx in transactions]
        success_count = sum(1 for tx in transactions if tx["success"])

        return {
            "transaction_type": tx_type,
            "total_transactions": len(transactions),
            "success_rate": success_count / len(transactions),
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "p50_duration": statistics.median(durations),
            "p95_duration": self._percentile(durations, 95),
            "p99_duration": self._percentile(durations, 99),
            "throughput_per_minute": len(transactions) / max(1, (datetime.now() - transactions[0]["start_time"]).total_seconds() / 60)
        }

    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall APM metrics"""
        all_transactions = []
        for tx_list in self.transactions.values():
            all_transactions.extend(tx_list)

        if not all_transactions:
            return {}

        return self._calculate_transaction_metrics(all_transactions, "overall")

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0
        values_sorted = sorted(values)
        index = int(len(values_sorted) * percentile / 100)
        return values_sorted[min(index, len(values_sorted) - 1)]

    def _monitor_transactions(self):
        """Monitor transaction patterns"""
        while self.is_monitoring:
            try:
                # Analyze transaction patterns
                self._analyze_transaction_patterns()
                time.sleep(60)  # Analyze every minute

            except Exception as e:
                print(f"Transaction monitoring error: {e}")

    def _analyze_transaction_patterns(self):
        """Analyze transaction patterns for anomalies"""
        for tx_type, transactions in self.transactions.items():
            if len(transactions) < 10:
                continue

            # Check for performance degradation
            recent_transactions = transactions[-50:]  # Last 50 transactions
            older_transactions = transactions[-100:-50]  # Previous 50

            if recent_transactions and older_transactions:
                recent_avg = statistics.mean([tx["duration"] for tx in recent_transactions])
                older_avg = statistics.mean([tx["duration"] for tx in older_transactions])

                if recent_avg > older_avg * 1.5:  # 50% performance degradation
                    print(f"🚨 PERFORMANCE DEGRADATION ALERT: {tx_type} transactions slowed by {(recent_avg/older_avg - 1)*100:.1f}%")

            # Check for error rate spikes
            recent_success_rate = sum(1 for tx in recent_transactions if tx["success"]) / len(recent_transactions)
            if recent_success_rate < 0.95:  # Less than 95% success rate
                print(f"🚨 HIGH ERROR RATE ALERT: {tx_type} transactions at {(1-recent_success_rate)*100:.1f}% error rate")

    def _analyze_performance(self):
        """Analyze overall application performance"""
        while self.is_monitoring:
            try:
                # Update APM metrics
                self.apm_data["overall"] = {
                    "timestamp": datetime.now(),
                    "active_transactions": len(self.transaction_traces),
                    "transaction_types": len(self.transactions),
                    "total_transactions": sum(len(txs) for txs in self.transactions.values()),
                    "performance_score": self._calculate_performance_score(),
                    "bottlenecks": self._identify_transaction_bottlenecks()
                }

                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                print(f"Performance analysis error: {e}")

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        if not self.transactions:
            return 100.0  # No transactions = perfect score

        scores = []

        for tx_type, transactions in self.transactions.items():
            if not transactions:
                continue

            metrics = self._calculate_transaction_metrics(transactions, tx_type)

            # Success rate score (0-100)
            success_score = metrics["success_rate"] * 100

            # Response time score (0-100, better for lower times)
            avg_duration = metrics["avg_duration"]
            if avg_duration < 0.1:  # < 100ms = excellent
                response_score = 100
            elif avg_duration < 1.0:  # < 1s = good
                response_score = 80
            elif avg_duration < 5.0:  # < 5s = acceptable
                response_score = 60
            else:  # > 5s = poor
                response_score = 20

            transaction_score = (success_score + response_score) / 2
            scores.append(transaction_score)

        return statistics.mean(scores) if scores else 100.0

    def _identify_transaction_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify transaction bottlenecks"""
        bottlenecks = []

        for tx_type, transactions in self.transactions.items():
            if len(transactions) < 5:
                continue

            metrics = self._calculate_transaction_metrics(transactions, tx_type)

            # Check for slow transactions
            if metrics["p95_duration"] > 2.0:  # 95th percentile > 2 seconds
                bottlenecks.append({
                    "type": "slow_transactions",
                    "transaction_type": tx_type,
                    "severity": "medium",
                    "description": f"P95 duration: {metrics['p95_duration']:.2f}s",
                    "impact": f"Affects {metrics['total_transactions']} transactions"
                })

            # Check for high error rates
            if metrics["success_rate"] < 0.9:  # < 90% success rate
                bottlenecks.append({
                    "type": "high_error_rate",
                    "transaction_type": tx_type,
                    "severity": "high",
                    "description": f"Success rate: {metrics['success_rate']:.1%}",
                    "impact": f"{(1-metrics['success_rate'])*100:.1f}% of transactions failing"
                })

            # Check for low throughput
            if metrics["throughput_per_minute"] < 10:  # Less than 10 transactions per minute
                bottlenecks.append({
                    "type": "low_throughput",
                    "transaction_type": tx_type,
                    "severity": "low",
                    "description": f"Throughput: {metrics['throughput_per_minute']:.1f} tx/min",
                    "impact": "Low transaction volume"
                })

        return bottlenecks

    def get_apm_dashboard(self) -> Dict[str, Any]:
        """Get APM dashboard data"""
        return {
            "overall_metrics": self.apm_data.get("overall", {}),
            "transaction_metrics": self.get_transaction_metrics(),
            "performance_score": self._calculate_performance_score(),
            "bottlenecks": self._identify_transaction_bottlenecks(),
            "recommendations": self._generate_apm_recommendations()
        }

    def _generate_apm_recommendations(self) -> List[Dict[str, Any]]:
        """Generate APM recommendations"""
        recommendations = []

        performance_score = self._calculate_performance_score()
        if performance_score < 70:
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "recommendation": f"Overall performance score is {performance_score:.1f}/100 - needs improvement",
                "actions": ["Profile slow transactions", "Optimize database queries", "Implement caching"]
            })

        # Check for specific transaction issues
        for tx_type, transactions in self.transactions.items():
            if not transactions:
                continue

            metrics = self._calculate_transaction_metrics(transactions, tx_type)

            if metrics["success_rate"] < 0.95:
                recommendations.append({
                    "category": "reliability",
                    "priority": "high",
                    "recommendation": f"Improve {tx_type} transaction success rate ({metrics['success_rate']:.1%})",
                    "actions": ["Add error handling", "Implement retry logic", "Monitor error patterns"]
                })

            if metrics["p95_duration"] > 1.0:
                recommendations.append({
                    "category": "performance",
                    "priority": "medium",
                    "recommendation": f"Optimize {tx_type} transaction response time (P95: {metrics['p95_duration']:.2f}s)",
                    "actions": ["Add database indexes", "Implement caching", "Optimize algorithms"]
                })

        return recommendations
