"""
Rate Limiting System for AI Personalized Medicine Platform
Advanced rate limiting with multiple algorithms, burst handling, and analytics
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import random
import math
import statistics
from enum import Enum

class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    GCRA = "gcra"

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, retry_after: int, limit: int, remaining: int):
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds")

class RateLimiter:
    """Base rate limiter class"""

    def __init__(self, algorithm: RateLimitAlgorithm, capacity: int, refill_rate: float = None):
        self.algorithm = algorithm
        self.capacity = capacity
        self.refill_rate = refill_rate or capacity  # tokens per second
        self.requests = defaultdict(list)
        self.last_refill = defaultdict(float)

    def check_limit(self, key: str, cost: int = 1) -> Dict[str, Any]:
        """Check if request is within rate limit"""
        raise NotImplementedError

    def reset(self, key: str):
        """Reset rate limit for a key"""
        if key in self.requests:
            del self.requests[key]
        if key in self.last_refill:
            del self.last_refill[key]

class FixedWindowLimiter(RateLimiter):
    """Fixed window rate limiter"""

    def __init__(self, capacity: int, window_seconds: int):
        super().__init__(RateLimitAlgorithm.FIXED_WINDOW, capacity)
        self.window_seconds = window_seconds

    def check_limit(self, key: str, cost: int = 1) -> Dict[str, Any]:
        current_time = time.time()
        window_start = current_time - (current_time % self.window_seconds)

        if key not in self.requests:
            self.requests[key] = []

        # Remove old requests outside current window
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time >= window_start
        ]

        current_count = len(self.requests[key])

        if current_count + cost <= self.capacity:
            # Allow request
            for _ in range(cost):
                self.requests[key].append(current_time)

            remaining = self.capacity - (current_count + cost)
            return {
                'allowed': True,
                'remaining': remaining,
                'reset_time': window_start + self.window_seconds,
                'retry_after': 0
            }
        else:
            # Deny request
            reset_time = window_start + self.window_seconds
            retry_after = int(reset_time - current_time)
            return {
                'allowed': False,
                'remaining': 0,
                'reset_time': reset_time,
                'retry_after': max(1, retry_after),
                'limit': self.capacity
            }

class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter"""

    def __init__(self, capacity: int, window_seconds: int):
        super().__init__(RateLimitAlgorithm.SLIDING_WINDOW, capacity)
        self.window_seconds = window_seconds

    def check_limit(self, key: str, cost: int = 1) -> Dict[str, Any]:
        current_time = time.time()

        if key not in self.requests:
            self.requests[key] = deque()

        # Remove old requests outside sliding window
        cutoff_time = current_time - self.window_seconds
        while self.requests[key] and self.requests[key][0] < cutoff_time:
            self.requests[key].popleft()

        current_count = len(self.requests[key])

        if current_count + cost <= self.capacity:
            # Allow request
            for _ in range(cost):
                self.requests[key].append(current_time)

            # Calculate reset time based on oldest request
            if self.requests[key]:
                reset_time = self.requests[key][0] + self.window_seconds
            else:
                reset_time = current_time + self.window_seconds

            remaining = self.capacity - (current_count + cost)
            return {
                'allowed': True,
                'remaining': remaining,
                'reset_time': reset_time,
                'retry_after': 0
            }
        else:
            # Deny request
            if self.requests[key]:
                reset_time = self.requests[key][0] + self.window_seconds
            else:
                reset_time = current_time + self.window_seconds

            retry_after = int(reset_time - current_time)
            return {
                'allowed': False,
                'remaining': 0,
                'reset_time': reset_time,
                'retry_after': max(1, retry_after),
                'limit': self.capacity
            }

class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter"""

    def __init__(self, capacity: int, refill_rate: float):
        super().__init__(RateLimitAlgorithm.TOKEN_BUCKET, capacity, refill_rate)
        self.tokens = defaultdict(float)

    def check_limit(self, key: str, cost: int = 1) -> Dict[str, Any]:
        current_time = time.time()

        if key not in self.tokens:
            self.tokens[key] = float(self.capacity)

        # Refill tokens
        if key in self.last_refill:
            time_passed = current_time - self.last_refill[key]
            tokens_to_add = time_passed * self.refill_rate
            self.tokens[key] = min(self.capacity, self.tokens[key] + tokens_to_add)

        self.last_refill[key] = current_time

        if self.tokens[key] >= cost:
            # Allow request
            self.tokens[key] -= cost
            reset_time = current_time + ((self.capacity - self.tokens[key]) / self.refill_rate)

            return {
                'allowed': True,
                'remaining': int(self.tokens[key]),
                'reset_time': reset_time,
                'retry_after': 0
            }
        else:
            # Deny request
            tokens_needed = cost - self.tokens[key]
            retry_after = math.ceil(tokens_needed / self.refill_rate)

            return {
                'allowed': False,
                'remaining': int(self.tokens[key]),
                'reset_time': current_time + retry_after,
                'retry_after': retry_after,
                'limit': self.capacity
            }

class LeakyBucketLimiter(RateLimiter):
    """Leaky bucket rate limiter"""

    def __init__(self, capacity: int, leak_rate: float):
        super().__init__(RateLimitAlgorithm.LEAKY_BUCKET, capacity)
        self.leak_rate = leak_rate  # requests per second
        self.last_leak = defaultdict(float)

    def check_limit(self, key: str, cost: int = 1) -> Dict[str, Any]:
        current_time = time.time()

        if key not in self.requests:
            self.requests[key] = 0

        # Leak tokens
        if key in self.last_leak:
            time_passed = current_time - self.last_leak[key]
            leaked = time_passed * self.leak_rate
            self.requests[key] = max(0, self.requests[key] - leaked)

        self.last_leak[key] = current_time

        if self.requests[key] + cost <= self.capacity:
            # Allow request
            self.requests[key] += cost

            # Calculate reset time
            excess_requests = self.requests[key] - (self.capacity - cost)
            if excess_requests > 0:
                reset_time = current_time + (excess_requests / self.leak_rate)
            else:
                reset_time = current_time

            remaining = self.capacity - int(self.requests[key])
            return {
                'allowed': True,
                'remaining': remaining,
                'reset_time': reset_time,
                'retry_after': 0
            }
        else:
            # Deny request
            excess_requests = self.requests[key] + cost - self.capacity
            retry_after = math.ceil(excess_requests / self.leak_rate)

            return {
                'allowed': False,
                'remaining': self.capacity - int(self.requests[key]),
                'reset_time': current_time + retry_after,
                'retry_after': retry_after,
                'limit': self.capacity
            }

class GCRALimiter(RateLimiter):
    """Generic Cell Rate Algorithm (GCRA) limiter"""

    def __init__(self, capacity: int, time_unit: float):
        super().__init__(RateLimitAlgorithm.GCRA, capacity)
        self.time_unit = time_unit  # seconds per request
        self.tat = defaultdict(float)  # Theoretical Arrival Time

    def check_limit(self, key: str, cost: int = 1) -> Dict[str, Any]:
        current_time = time.time()

        if key not in self.tat:
            self.tat[key] = current_time

        # Calculate new TAT
        new_tat = max(self.tat[key], current_time) + (cost * self.time_unit)

        if new_tat - current_time <= (self.capacity * self.time_unit):
            # Allow request
            self.tat[key] = new_tat

            reset_time = new_tat
            remaining = int((self.capacity * self.time_unit - (new_tat - current_time)) / self.time_unit)

            return {
                'allowed': True,
                'remaining': max(0, remaining),
                'reset_time': reset_time,
                'retry_after': 0
            }
        else:
            # Deny request
            retry_after = int(new_tat - current_time)

            return {
                'allowed': False,
                'remaining': 0,
                'reset_time': new_tat,
                'retry_after': retry_after,
                'limit': self.capacity
            }

class RateLimitRule:
    """Rate limit rule with conditions"""

    def __init__(self, name: str, limiter: RateLimiter,
                 conditions: Dict[str, Any] = None, priority: int = 0):
        self.name = name
        self.limiter = limiter
        self.conditions = conditions or {}
        self.priority = priority

    def matches(self, request_context: Dict[str, Any]) -> bool:
        """Check if rule matches request context"""
        for condition_key, condition_value in self.conditions.items():
            request_value = request_context.get(condition_key)

            if isinstance(condition_value, list):
                if request_value not in condition_value:
                    return False
            elif isinstance(condition_value, dict):
                # Range conditions
                if 'min' in condition_value and request_value < condition_value['min']:
                    return False
                if 'max' in condition_value and request_value > condition_value['max']:
                    return False
            else:
                if request_value != condition_value:
                    return False

        return True

class RateLimitManager:
    """Main rate limiting manager"""

    def __init__(self):
        self.rules = []
        self.exclusions = set()
        self.analytics = defaultdict(list)
        self.is_running = False
        self.cleanup_thread = None

    def add_rule(self, rule: RateLimitRule):
        """Add a rate limit rule"""
        self.rules.append(rule)
        # Sort by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def exclude_path(self, path: str):
        """Exclude a path from rate limiting"""
        self.exclusions.add(path)

    def check_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if request should be rate limited"""
        path = request_context.get('path', '')

        # Check exclusions
        if any(exclusion in path for exclusion in self.exclusions):
            return {'allowed': True, 'source': 'excluded'}

        # Get client identifier
        client_key = self._get_client_key(request_context)

        # Find matching rule
        for rule in self.rules:
            if rule.matches(request_context):
                result = rule.limiter.check_limit(client_key, request_context.get('cost', 1))

                # Add rule info
                result['rule'] = rule.name
                result['client_key'] = client_key

                # Record analytics
                self._record_analytics(client_key, rule.name, result)

                return result

        # No rule matched, allow request
        return {'allowed': True, 'source': 'no_rule'}

    def _get_client_key(self, request_context: Dict[str, Any]) -> str:
        """Generate client key for rate limiting"""
        # Try different identifiers in order of preference
        identifiers = []

        # IP address
        ip = request_context.get('ip_address')
        if ip:
            identifiers.append(f"ip:{ip}")

        # User ID (if authenticated)
        user_id = request_context.get('user_id')
        if user_id:
            identifiers.append(f"user:{user_id}")

        # API key
        api_key = request_context.get('api_key')
        if api_key:
            # Hash API key for privacy
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            identifiers.append(f"key:{key_hash}")

        # Client ID
        client_id = request_context.get('client_id')
        if client_id:
            identifiers.append(f"client:{client_id}")

        # Combine identifiers
        if identifiers:
            return "|".join(identifiers)
        else:
            # Fallback to IP only
            return f"unknown:{request_context.get('ip_address', '0.0.0.0')}"

    def _record_analytics(self, client_key: str, rule_name: str, result: Dict[str, Any]):
        """Record rate limiting analytics"""
        analytics_entry = {
            'timestamp': datetime.now(),
            'client_key': client_key,
            'rule': rule_name,
            'allowed': result['allowed'],
            'remaining': result.get('remaining', 0),
            'retry_after': result.get('retry_after', 0)
        }

        self.analytics[client_key].append(analytics_entry)

        # Keep only recent analytics (last 1000 entries per client)
        if len(self.analytics[client_key]) > 1000:
            self.analytics[client_key] = self.analytics[client_key][-1000:]

    def get_client_stats(self, client_key: str) -> Dict[str, Any]:
        """Get statistics for a client"""
        client_analytics = self.analytics.get(client_key, [])

        if not client_analytics:
            return {'requests': 0, 'blocked': 0, 'success_rate': 1.0}

        total_requests = len(client_analytics)
        blocked_requests = sum(1 for entry in client_analytics if not entry['allowed'])
        recent_requests = [entry for entry in client_analytics
                          if entry['timestamp'] > datetime.now() - timedelta(hours=1)]

        return {
            'total_requests': total_requests,
            'blocked_requests': blocked_requests,
            'success_rate': (total_requests - blocked_requests) / total_requests if total_requests > 0 else 1.0,
            'requests_per_hour': len(recent_requests),
            'last_request': client_analytics[-1]['timestamp'] if client_analytics else None
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics"""
        all_entries = []
        for client_entries in self.analytics.values():
            all_entries.extend(client_entries)

        if not all_entries:
            return {'total_requests': 0, 'total_blocked': 0, 'success_rate': 1.0}

        total_requests = len(all_entries)
        total_blocked = sum(1 for entry in all_entries if not entry['allowed'])

        # Requests per hour
        recent_entries = [entry for entry in all_entries
                         if entry['timestamp'] > datetime.now() - timedelta(hours=1)]

        return {
            'total_requests': total_requests,
            'total_blocked': total_blocked,
            'success_rate': (total_requests - total_blocked) / total_requests,
            'requests_per_hour': len(recent_entries),
            'unique_clients': len(self.analytics),
            'active_rules': len(self.rules)
        }

    def cleanup_old_data(self):
        """Clean up old analytics data"""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of data

        for client_key in list(self.analytics.keys()):
            self.analytics[client_key] = [
                entry for entry in self.analytics[client_key]
                if entry['timestamp'] > cutoff_time
            ]

            # Remove empty client entries
            if not self.analytics[client_key]:
                del self.analytics[client_key]

    def start_cleanup_scheduler(self):
        """Start periodic cleanup scheduler"""
        self.is_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_scheduler, daemon=True)
        self.cleanup_thread.start()

    def stop_cleanup_scheduler(self):
        """Stop cleanup scheduler"""
        self.is_running = False

    def _cleanup_scheduler(self):
        """Periodic cleanup scheduler"""
        while self.is_running:
            try:
                self.cleanup_old_data()
                time.sleep(3600)  # Clean up every hour
            except Exception as e:
                print(f"Rate limit cleanup error: {e}")

class BurstHandler:
    """Handles burst traffic patterns"""

    def __init__(self, burst_capacity: int, sustained_rate: float):
        self.burst_capacity = burst_capacity
        self.sustained_rate = sustained_rate
        self.burst_limiter = TokenBucketLimiter(burst_capacity, sustained_rate)
        self.sustained_limiter = TokenBucketLimiter(burst_capacity * 2, sustained_rate)

    def check_burst(self, key: str, cost: int = 1) -> Dict[str, Any]:
        """Check burst allowance"""
        # First check burst capacity
        burst_result = self.burst_limiter.check_limit(key, cost)

        if burst_result['allowed']:
            return burst_result

        # If burst is exceeded, check sustained rate
        sustained_result = self.sustained_limiter.check_limit(key, cost)

        if sustained_result['allowed']:
            # Allow but mark as burst
            return {
                **sustained_result,
                'burst_exceeded': True,
                'sustained_allowed': True
            }
        else:
            return {
                **sustained_result,
                'burst_exceeded': True,
                'sustained_allowed': False
            }

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""

    def __init__(self, base_capacity: int, algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET):
        self.base_capacity = base_capacity
        self.algorithm = algorithm
        self.current_capacity = base_capacity
        self.system_load = 0.5  # Default 50% load
        self.adjustment_factor = 0.1
        self.limiter = self._create_limiter()

    def _create_limiter(self) -> RateLimiter:
        """Create rate limiter based on algorithm"""
        if self.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return TokenBucketLimiter(self.current_capacity, self.current_capacity / 60)
        elif self.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return FixedWindowLimiter(self.current_capacity, 60)
        elif self.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return SlidingWindowLimiter(self.current_capacity, 60)
        else:
            return TokenBucketLimiter(self.current_capacity, self.current_capacity / 60)

    def update_system_load(self, load: float):
        """Update system load and adjust capacity"""
        self.system_load = max(0.1, min(1.0, load))  # Clamp between 0.1 and 1.0

        # Adjust capacity based on load
        if self.system_load > 0.8:  # High load
            self.current_capacity = int(self.base_capacity * 0.7)  # Reduce capacity
        elif self.system_load < 0.3:  # Low load
            self.current_capacity = int(self.base_capacity * 1.3)  # Increase capacity
        else:
            # Gradually move towards base capacity
            adjustment = (self.base_capacity - self.current_capacity) * self.adjustment_factor
            self.current_capacity = int(self.current_capacity + adjustment)

        # Recreate limiter with new capacity
        self.limiter = self._create_limiter()

    def check_limit(self, key: str, cost: int = 1) -> Dict[str, Any]:
        """Check rate limit with current adaptive capacity"""
        return self.limiter.check_limit(key, cost)

# Global rate limiting instances
rate_limit_manager = RateLimitManager()
burst_handler = BurstHandler(burst_capacity=100, sustained_rate=10)
adaptive_limiter = AdaptiveRateLimiter(base_capacity=1000)

# Default rate limit rules
default_rules = [
    RateLimitRule(
        "api_general",
        TokenBucketLimiter(100, 10),  # 100 requests, 10 per second
        {"path_prefix": "/api/"},
        priority=1
    ),
    RateLimitRule(
        "health_monitoring",
        TokenBucketLimiter(500, 50),  # Higher limit for health monitoring
        {"path_contains": "health"},
        priority=2
    ),
    RateLimitRule(
        "admin_endpoints",
        TokenBucketLimiter(50, 5),  # Stricter limits for admin
        {"path_prefix": "/api/admin/"},
        priority=3
    )
]

# Add default rules
for rule in default_rules:
    rate_limit_manager.add_rule(rule)

# Start cleanup scheduler
rate_limit_manager.start_cleanup_scheduler()
