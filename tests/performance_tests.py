"""
Performance Testing Suite for AI Personalized Medicine Platform
Comprehensive performance testing with load testing, stress testing, and benchmarking
"""

import time
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import psutil
import tracemalloc
import cProfile
import pstats
import io

class PerformanceTestResult:
    """Container for performance test results"""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = None
        self.requests_total = 0
        self.requests_successful = 0
        self.requests_failed = 0
        self.response_times = []
        self.error_types = defaultdict(int)
        self.throughput_data = []
        self.memory_usage = []
        self.cpu_usage = []
        self.custom_metrics = {}

    def record_request(self, response_time: float, success: bool, error_type: str = None):
        """Record a request result"""
        self.requests_total += 1
        if success:
            self.requests_successful += 1
        else:
            self.requests_failed += 1
            if error_type:
                self.error_types[error_type] += 1

        self.response_times.append(response_time)

    def record_throughput(self, requests_per_second: float):
        """Record throughput measurement"""
        self.throughput_data.append({
            'timestamp': datetime.now(),
            'rps': requests_per_second
        })

    def record_system_metrics(self, memory_mb: float, cpu_percent: float):
        """Record system resource usage"""
        self.memory_usage.append(memory_mb)
        self.cpu_usage.append(cpu_percent)

    def finalize(self):
        """Finalize test results"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        if not self.response_times:
            return {"error": "No requests recorded"}

        return {
            'test_name': self.test_name,
            'duration_seconds': self.duration,
            'total_requests': self.requests_total,
            'successful_requests': self.requests_successful,
            'failed_requests': self.requests_failed,
            'success_rate': self.requests_successful / self.requests_total if self.requests_total > 0 else 0,
            'avg_response_time': statistics.mean(self.response_times),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'p50_response_time': statistics.median(self.response_times),
            'p95_response_time': self._percentile(self.response_times, 95),
            'p99_response_time': self._percentile(self.response_times, 99),
            'requests_per_second': self.requests_total / self.duration if self.duration > 0 else 0,
            'error_breakdown': dict(self.error_types),
            'memory_peak_mb': max(self.memory_usage) if self.memory_usage else 0,
            'cpu_avg_percent': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            'throughput_avg_rps': statistics.mean([d['rps'] for d in self.throughput_data]) if self.throughput_data else 0
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile / 100)
        return data_sorted[min(index, len(data_sorted) - 1)]

    def export_to_json(self, filename: str):
        """Export results to JSON file"""
        summary = self.get_summary()
        summary['response_times'] = self.response_times[:1000]  # Limit for file size
        summary['throughput_data'] = self.throughput_data
        summary['memory_usage'] = self.memory_usage
        summary['cpu_usage'] = self.cpu_usage
        summary['custom_metrics'] = self.custom_metrics

        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def export_to_csv(self, filename: str):
        """Export response times to CSV"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['response_time_seconds'])
            for rt in self.response_times:
                writer.writerow([rt])

class LoadGenerator:
    """Load generator for performance testing"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.is_running = False

    def generate_request(self, endpoint: str, method: str = "GET",
                        data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Generate a single request"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                headers=headers,
                timeout=30
            )

            response_time = time.time() - start_time
            success = response.status_code < 400

            return {
                'success': success,
                'response_time': response_time,
                'status_code': response.status_code,
                'error_type': None if success else f'HTTP_{response.status_code}'
            }

        except requests.exceptions.Timeout:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'status_code': None,
                'error_type': 'timeout'
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'status_code': None,
                'error_type': 'connection_error'
            }
        except Exception as e:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'status_code': None,
                'error_type': str(type(e).__name__)
            }

class PerformanceTestSuite:
    """Comprehensive performance test suite"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.load_generator = LoadGenerator(base_url)
        self.results = []

    def run_load_test(self, endpoint: str, concurrent_users: int,
                     duration_seconds: int, ramp_up_seconds: int = 10) -> PerformanceTestResult:
        """Run load test with specified parameters"""

        test_result = PerformanceTestResult(f"load_test_{endpoint}_{concurrent_users}users")
        stop_event = threading.Event()

        def worker_thread(user_id: int):
            """Worker thread for each concurrent user"""
            requests_made = 0

            while not stop_event.is_set():
                result = self.load_generator.generate_request(endpoint)
                test_result.record_request(
                    result['response_time'],
                    result['success'],
                    result['error_type']
                )
                requests_made += 1
                time.sleep(0.1)  # Small delay between requests

        # System monitoring thread
        def monitor_system():
            while not stop_event.is_set():
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)
                test_result.record_system_metrics(memory.used / 1024 / 1024, cpu)
                time.sleep(5)

        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()

        # Ramp up users gradually
        threads = []
        users_started = 0

        for i in range(concurrent_users):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)

            if ramp_up_seconds > 0:
                # Stagger thread starts
                time.sleep(ramp_up_seconds / concurrent_users)
            else:
                thread.start()
                users_started += 1

        # All users started, run for duration
        time.sleep(duration_seconds)
        stop_event.set()

        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=5)

        test_result.finalize()
        self.results.append(test_result)

        return test_result

    def run_stress_test(self, endpoint: str, max_users: int,
                       increment_users: int = 10, duration_per_level: int = 30) -> List[PerformanceTestResult]:
        """Run stress test by gradually increasing load"""

        results = []
        current_users = increment_users

        while current_users <= max_users:
            print(f"Running stress test with {current_users} users...")
            result = self.run_load_test(endpoint, current_users, duration_per_level, 5)
            results.append(result)

            # Check if system is failing
            if result.requests_failed / result.requests_total > 0.5:  # >50% failure rate
                print(f"High failure rate detected at {current_users} users. Stopping stress test.")
                break

            current_users += increment_users

        return results

    def run_spike_test(self, endpoint: str, baseline_users: int,
                      spike_users: int, spike_duration: int = 60) -> PerformanceTestResult:
        """Run spike test with sudden load increase"""

        test_result = PerformanceTestResult(f"spike_test_{endpoint}")
        stop_event = threading.Event()

        def baseline_worker(user_id: int):
            """Baseline load worker"""
            while not stop_event.is_set():
                result = self.load_generator.generate_request(endpoint)
                test_result.record_request(
                    result['response_time'],
                    result['success'],
                    result['error_type']
                )
                time.sleep(0.2)

        def spike_worker(user_id: int):
            """Spike load worker"""
            while not stop_event.is_set():
                result = self.load_generator.generate_request(endpoint)
                test_result.record_request(
                    result['response_time'],
                    result['success'],
                    result['error_type']
                )
                time.sleep(0.05)  # Faster requests during spike

        # Start baseline load
        baseline_threads = []
        for i in range(baseline_users):
            thread = threading.Thread(target=baseline_worker, args=(i,))
            thread.start()
            baseline_threads.append(thread)

        # Run baseline for a while
        time.sleep(30)

        # Add spike load
        spike_threads = []
        for i in range(spike_users):
            thread = threading.Thread(target=spike_worker, args=(i + baseline_users,))
            thread.start()
            spike_threads.append(thread)

        # Run spike for specified duration
        time.sleep(spike_duration)

        stop_event.set()

        # Wait for all threads
        for thread in baseline_threads + spike_threads:
            thread.join(timeout=5)

        test_result.finalize()
        self.results.append(test_result)

        return test_result

    def run_endurance_test(self, endpoint: str, users: int,
                          duration_minutes: int) -> PerformanceTestResult:
        """Run endurance test for long-duration stability"""

        duration_seconds = duration_minutes * 60
        return self.run_load_test(endpoint, users, duration_seconds, 30)

    def run_api_endpoint_benchmark(self, endpoints: List[Dict[str, Any]],
                                 concurrent_users: int = 10,
                                 duration_seconds: int = 60) -> Dict[str, PerformanceTestResult]:
        """Benchmark multiple API endpoints"""

        results = {}

        for endpoint_config in endpoints:
            endpoint = endpoint_config['endpoint']
            method = endpoint_config.get('method', 'GET')
            data = endpoint_config.get('data')

            print(f"Benchmarking {method} {endpoint}...")

            # Temporarily modify load generator for this endpoint
            original_generate = self.load_generator.generate_request

            def custom_request_generator(ep, meth, dat):
                return lambda: self.load_generator.generate_request(ep, meth, dat)

            result = self.run_load_test(
                endpoint,
                concurrent_users,
                duration_seconds,
                5
            )

            results[endpoint] = result

        return results

class MemoryProfiler:
    """Memory profiling for performance tests"""

    def __init__(self):
        self.snapshots = []
        self.is_profiling = False

    def start_profiling(self):
        """Start memory profiling"""
        tracemalloc.start()
        self.is_profiling = True
        self.snapshots = []

    def take_snapshot(self, label: str = ""):
        """Take memory snapshot"""
        if not self.is_profiling:
            return

        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'timestamp': datetime.now(),
            'label': label,
            'snapshot': snapshot,
            'stats': snapshot.statistics('lineno')
        })

    def stop_profiling(self):
        """Stop memory profiling"""
        tracemalloc.stop()
        self.is_profiling = False

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate memory usage report"""
        if not self.snapshots:
            return {"error": "No snapshots taken"}

        report = {
            'total_snapshots': len(self.snapshots),
            'memory_growth': [],
            'top_allocations': []
        }

        if len(self.snapshots) > 1:
            # Calculate memory growth between snapshots
            for i in range(1, len(self.snapshots)):
                prev_stats = self.snapshots[i-1]['stats']
                curr_stats = self.snapshots[i]['stats']

                prev_total = sum(stat.size for stat in prev_stats)
                curr_total = sum(stat.size for stat in curr_stats)

                growth = curr_total - prev_total

                report['memory_growth'].append({
                    'from_snapshot': i-1,
                    'to_snapshot': i,
                    'growth_bytes': growth,
                    'growth_mb': growth / 1024 / 1024
                })

        # Get top memory allocations
        if self.snapshots:
            latest_stats = self.snapshots[-1]['stats'][:20]  # Top 20
            report['top_allocations'] = [
                {
                    'size': stat.size,
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count,
                    'traceback': str(stat.traceback)
                }
                for stat in latest_stats
            ]

        return report

class CPUProfiler:
    """CPU profiling for performance tests"""

    def __init__(self):
        self.profiles = []
        self.is_profiling = False

    def start_profiling(self, label: str = ""):
        """Start CPU profiling"""
        self.pr = cProfile.Profile()
        self.pr.enable()
        self.is_profiling = True
        self.current_label = label

    def stop_profiling(self):
        """Stop CPU profiling and save results"""
        if not self.is_profiling:
            return

        self.pr.disable()
        self.is_profiling = False

        # Get profile stats
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions

        self.profiles.append({
            'label': self.current_label,
            'timestamp': datetime.now(),
            'stats': s.getvalue()
        })

    def get_cpu_report(self) -> Dict[str, Any]:
        """Generate CPU usage report"""
        return {
            'total_profiles': len(self.profiles),
            'profiles': [
                {
                    'label': p['label'],
                    'timestamp': p['timestamp'],
                    'stats_summary': p['stats'][:500] + '...' if len(p['stats']) > 500 else p['stats']
                }
                for p in self.profiles
            ]
        }

class PerformanceBenchmarkSuite:
    """Performance benchmarking suite"""

    def __init__(self):
        self.benchmarks = []
        self.baseline_results = {}

    def add_benchmark(self, name: str, function: Callable, *args, **kwargs):
        """Add a benchmark function"""
        self.benchmarks.append({
            'name': name,
            'function': function,
            'args': args,
            'kwargs': kwargs
        })

    def run_benchmarks(self, iterations: int = 100) -> Dict[str, Any]:
        """Run all benchmarks"""
        results = {}

        for benchmark in self.benchmarks:
            print(f"Running benchmark: {benchmark['name']}")

            times = []
            for i in range(iterations):
                start_time = time.perf_counter()
                try:
                    benchmark['function'](*benchmark['args'], **benchmark['kwargs'])
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                except Exception as e:
                    print(f"Benchmark {benchmark['name']} iteration {i} failed: {e}")
                    continue

            if times:
                results[benchmark['name']] = {
                    'iterations': len(times),
                    'total_time': sum(times),
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'p50_time': statistics.median(times),
                    'p95_time': self._percentile(times, 95),
                    'p99_time': self._percentile(times, 99)
                }

        return results

    def set_baseline(self, results: Dict[str, Any]):
        """Set baseline results for comparison"""
        self.baseline_results = results

    def compare_with_baseline(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline"""
        comparison = {}

        for benchmark_name, current in current_results.items():
            if benchmark_name in self.baseline_results:
                baseline = self.baseline_results[benchmark_name]

                comparison[benchmark_name] = {
                    'current_avg': current['avg_time'],
                    'baseline_avg': baseline['avg_time'],
                    'change_percent': ((current['avg_time'] - baseline['avg_time']) / baseline['avg_time']) * 100,
                    'improvement': current['avg_time'] < baseline['avg_time']
                }

        return comparison

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile / 100)
        return data_sorted[min(index, len(data_sorted) - 1)]

class DatabasePerformanceTest:
    """Database performance testing"""

    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string
        self.test_data = []

    def generate_test_data(self, num_records: int = 10000):
        """Generate test data for database testing"""
        self.test_data = [
            {
                'patient_id': f'PAT{i:04d}',
                'metric_type': ['heart_rate', 'blood_pressure', 'weight'][i % 3],
                'value': 50 + (i % 100),
                'timestamp': datetime.now() - timedelta(hours=i % 24),
                'device_id': f'device_{i % 10}'
            }
            for i in range(num_records)
        ]

    def run_database_load_test(self, concurrent_writers: int = 10,
                             duration_seconds: int = 60) -> PerformanceTestResult:
        """Run database load test"""

        result = PerformanceTestResult("database_load_test")
        stop_event = threading.Event()

        def writer_thread(thread_id: int):
            """Database writer thread"""
            while not stop_event.is_set():
                # Simulate database write operation
                start_time = time.time()

                # Mock database operation
                time.sleep(0.01)  # Simulate I/O

                response_time = time.time() - start_time
                result.record_request(response_time, True)

        # Start writer threads
        threads = []
        for i in range(concurrent_writers):
            thread = threading.Thread(target=writer_thread, args=(i,))
            thread.start()
            threads.append(thread)

        # Run for duration
        time.sleep(duration_seconds)
        stop_event.set()

        # Wait for threads
        for thread in threads:
            thread.join(timeout=5)

        result.finalize()
        return result

# Example usage and test scenarios
def run_comprehensive_performance_test():
    """Run comprehensive performance test suite"""

    suite = PerformanceTestSuite("http://localhost:8000")

    print("Starting comprehensive performance test suite...")

    # Test scenarios
    scenarios = [
        {
            'name': 'API Health Check',
            'endpoint': '/api/health',
            'users': 50,
            'duration': 60
        },
        {
            'name': 'Patient Data Retrieval',
            'endpoint': '/api/patients',
            'users': 20,
            'duration': 120
        },
        {
            'name': 'Health Metrics Upload',
            'endpoint': '/api/health-monitoring',
            'users': 30,
            'duration': 90
        },
        {
            'name': 'Genomic Analysis',
            'endpoint': '/api/genomic-analysis',
            'users': 10,
            'duration': 180
        }
    ]

    all_results = {}

    for scenario in scenarios:
        print(f"\nRunning {scenario['name']}...")
        result = suite.run_load_test(
            scenario['endpoint'],
            scenario['users'],
            scenario['duration']
        )

        summary = result.get_summary()
        all_results[scenario['name']] = summary

        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(".3f")
        print(".1f")

        # Export results
        result.export_to_json(f"performance_results_{scenario['name'].lower().replace(' ', '_')}.json")

    # Generate comprehensive report
    generate_performance_report(all_results)

    return all_results

def generate_performance_report(results: Dict[str, Any]):
    """Generate comprehensive performance report"""

    report = {
        'generated_at': datetime.now(),
        'test_results': results,
        'summary': {
            'total_scenarios': len(results),
            'overall_success_rate': statistics.mean([r['success_rate'] for r in results.values()]),
            'average_response_time': statistics.mean([r['avg_response_time'] for r in results.values()]),
            'total_requests': sum([r['total_requests'] for r in results.values()])
        },
        'recommendations': []
    }

    # Generate recommendations based on results
    for scenario_name, result in results.items():
        if result['success_rate'] < 0.95:
            report['recommendations'].append(f"Improve reliability for {scenario_name} (success rate: {result['success_rate']:.1%})")

        if result['p95_response_time'] > 2.0:
            report['recommendations'].append(f"Optimize performance for {scenario_name} (P95: {result['p95_response_time']:.2f}s)")

    # Save report
    with open('performance_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("Performance test report generated: performance_test_report.json")

if __name__ == "__main__":
    # Run comprehensive performance tests
    results = run_comprehensive_performance_test()

    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE TEST SUMMARY")
    print("="*50)

    for scenario, result in results.items():
        print(f"\n{scenario}:")
        print(f"  Requests: {result['total_requests']}")
        print(f"  Success Rate: {result['success_rate']:.1%}")
        print(".3f")
        print(f"  P95 Response Time: {result['p95_response_time']:.3f}s")
