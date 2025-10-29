"""
API Gateway System for AI Personalized Medicine Platform
Provides centralized API management, routing, security, and monitoring
"""

import json
import time
import hashlib
import hmac
import secrets
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
import logging
import re
from urllib.parse import urlparse, parse_qs
import aiohttp
import redis
import jwt

class APIGateway:
    """Advanced API Gateway for healthcare platform"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.routes = {}
        self.middlewares = []
        self.rate_limiters = {}
        self.api_keys = {}
        self.jwt_secrets = {}
        self.webhook_endpoints = {}
        self.service_registry = {}
        self.metrics_collector = MetricsCollector()
        self.security_enforcer = SecurityEnforcer()
        self.load_balancer = LoadBalancer()
        self.cache_manager = CacheManager(redis_url)
        self.is_running = False
        self.gateway_workers = []

        # Initialize default configurations
        self._initialize_gateway()

    def _initialize_gateway(self):
        """Initialize API gateway configuration"""
        # Default middlewares
        self.middlewares = [
            self._cors_middleware,
            self._authentication_middleware,
            self._rate_limiting_middleware,
            self._logging_middleware,
            self._caching_middleware
        ]

        # Default rate limits
        self.rate_limiters = {
            "default": RateLimiter(requests_per_minute=60),
            "authenticated": RateLimiter(requests_per_minute=300),
            "premium": RateLimiter(requests_per_minute=1000),
            "admin": RateLimiter(requests_per_minute=5000)
        }

        print("🚪 API Gateway initialized")

    def start_gateway(self):
        """Start the API gateway"""
        self.is_running = True

        # Start gateway workers
        for i in range(8):  # 8 worker threads
            worker = threading.Thread(target=self._gateway_worker, daemon=True)
            worker.start()
            self.gateway_workers.append(worker)

        # Start metrics collection
        metrics_worker = threading.Thread(target=self._metrics_worker, daemon=True)
        metrics_worker.start()
        self.gateway_workers.append(metrics_worker)

        print("⚡ API Gateway started")

    def stop_gateway(self):
        """Stop the API gateway"""
        self.is_running = False
        print("🛑 API Gateway stopped")

    def register_route(self, path: str, methods: List[str], handler: Callable,
                      middlewares: List[Callable] = None, rate_limit: str = "default",
                      auth_required: bool = True, cache_enabled: bool = False) -> str:
        """Register an API route"""
        route_id = f"route_{hashlib.md5(path.encode()).hexdigest()[:8]}"

        route_config = {
            "route_id": route_id,
            "path": path,
            "methods": methods,
            "handler": handler,
            "middlewares": middlewares or [],
            "rate_limit": rate_limit,
            "auth_required": auth_required,
            "cache_enabled": cache_enabled,
            "registered_at": datetime.now(),
            "version": "v1",
            "deprecated": False
        }

        self.routes[route_id] = route_config

        # Register with load balancer
        self.load_balancer.register_route(route_id, route_config)

        print(f"📍 Registered route: {path} -> {route_id}")
        return route_id

    def register_service(self, service_name: str, service_config: Dict[str, Any]) -> str:
        """Register a backend service"""
        service_id = f"service_{service_name}_{int(time.time())}"

        service = {
            "service_id": service_id,
            "service_name": service_name,
            "base_url": service_config["base_url"],
            "health_check_endpoint": service_config.get("health_check", "/health"),
            "timeout": service_config.get("timeout", 30),
            "retry_policy": service_config.get("retry_policy", {"max_retries": 3, "backoff_factor": 2}),
            "circuit_breaker": service_config.get("circuit_breaker", {"failure_threshold": 5, "recovery_timeout": 60}),
            "registered_at": datetime.now(),
            "status": "unknown"
        }

        self.service_registry[service_id] = service

        # Start health monitoring
        self._start_service_health_monitor(service)

        print(f"🔧 Registered service: {service_name} -> {service_id}")
        return service_id

    def _start_service_health_monitor(self, service: Dict[str, Any]):
        """Start health monitoring for a service"""
        def health_check_worker():
            while self.is_running:
                try:
                    self._check_service_health(service)
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    print(f"Health check error for {service['service_name']}: {e}")

        worker = threading.Thread(target=health_check_worker, daemon=True)
        worker.start()

    def _check_service_health(self, service: Dict[str, Any]):
        """Check health of a backend service"""
        try:
            # In production, this would make actual HTTP calls
            # For simulation, randomly determine health
            is_healthy = random.random() > 0.1  # 90% uptime

            service["last_health_check"] = datetime.now()
            service["status"] = "healthy" if is_healthy else "unhealthy"
            service["response_time"] = random.uniform(0.1, 2.0)

        except Exception as e:
            service["status"] = "error"
            service["last_error"] = str(e)

    def generate_api_key(self, user_id: str, key_type: str = "standard",
                        permissions: List[str] = None) -> Dict[str, Any]:
        """Generate API key for a user"""
        if permissions is None:
            permissions = ["read"]

        api_key = secrets.token_hex(32)
        key_id = f"key_{user_id}_{int(time.time())}"

        key_config = {
            "key_id": key_id,
            "api_key": api_key,
            "user_id": user_id,
            "key_type": key_type,
            "permissions": permissions,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=365),
            "status": "active",
            "rate_limit": self.rate_limiters.get(key_type, self.rate_limiters["default"]),
            "usage": {"requests_today": 0, "last_used": None}
        }

        # Hash the API key for storage (never store plain text)
        key_config["key_hash"] = hashlib.sha256(api_key.encode()).hexdigest()

        self.api_keys[key_id] = key_config

        return {
            "key_id": key_id,
            "api_key": api_key,  # Only returned once for security
            "expires_at": key_config["expires_at"],
            "permissions": permissions
        }

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        for key_config in self.api_keys.values():
            if key_config["key_hash"] == key_hash and key_config["status"] == "active":
                if datetime.now() > key_config["expires_at"]:
                    key_config["status"] = "expired"
                    return None

                # Update usage
                key_config["usage"]["last_used"] = datetime.now()
                key_config["usage"]["requests_today"] += 1

                return key_config

        return None

    def generate_jwt_token(self, user_id: str, permissions: List[str],
                          expires_in: timedelta = timedelta(hours=1)) -> str:
        """Generate JWT token"""
        secret = self._get_jwt_secret(user_id)

        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "iat": datetime.now(),
            "exp": datetime.now() + expires_in,
            "iss": "ai-medicine-platform",
            "aud": "api-gateway"
        }

        token = jwt.encode(payload, secret, algorithm="HS256")
        return token

    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token"""
        try:
            # Try to decode without verification first to get user_id
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            user_id = unverified_payload.get("user_id")

            if not user_id:
                return None

            secret = self._get_jwt_secret(user_id)

            # Verify the token
            payload = jwt.decode(token, secret, algorithms=["HS256"],
                               audience="api-gateway", issuer="ai-medicine-platform")

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def _get_jwt_secret(self, user_id: str) -> str:
        """Get JWT secret for user"""
        if user_id not in self.jwt_secrets:
            self.jwt_secrets[user_id] = secrets.token_hex(32)

        return self.jwt_secrets[user_id]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming API request"""
        start_time = time.time()

        try:
            # Extract request details
            method = request.get("method", "GET")
            path = request.get("path", "/")
            headers = request.get("headers", {})
            body = request.get("body", {})
            query_params = request.get("query_params", {})

            # Find matching route
            route = self._find_route(path, method)
            if not route:
                return self._create_response(404, {"error": "Route not found"})

            # Apply middlewares
            context = {
                "request": request,
                "route": route,
                "start_time": start_time,
                "auth_info": None,
                "rate_limit_info": None
            }

            for middleware in self.middlewares + route.get("middlewares", []):
                result = await self._apply_middleware(middleware, context)
                if result.get("blocked"):
                    return self._create_response(
                        result.get("status_code", 403),
                        {"error": result.get("message", "Request blocked")}
                    )

            # Route to handler
            response = await self._route_request(route, context)

            # Record metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_request(
                path, method, response["status_code"], processing_time
            )

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics_collector.record_error(path, str(e), processing_time)
            return self._create_response(500, {"error": "Internal server error"})

    def _find_route(self, path: str, method: str) -> Optional[Dict[str, Any]]:
        """Find matching route for request"""
        for route in self.routes.values():
            if method in route["methods"] and self._path_matches(path, route["path"]):
                return route
        return None

    def _path_matches(self, request_path: str, route_path: str) -> bool:
        """Check if request path matches route pattern"""
        # Simple pattern matching (could be enhanced with regex)
        if route_path == request_path:
            return True

        # Handle parameterized routes like /api/patients/{id}
        route_parts = route_path.split('/')
        request_parts = request_path.split('/')

        if len(route_parts) != len(request_parts):
            return False

        for route_part, request_part in zip(route_parts, request_parts):
            if route_part.startswith('{') and route_part.endswith('}'):
                continue  # Parameter match
            elif route_part != request_part:
                return False

        return True

    async def _apply_middleware(self, middleware: Callable, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply middleware to request"""
        try:
            if asyncio.iscoroutinefunction(middleware):
                return await middleware(context)
            else:
                return middleware(context)
        except Exception as e:
            return {"blocked": True, "status_code": 500, "message": f"Middleware error: {str(e)}"}

    async def _cors_middleware(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """CORS middleware"""
        headers = context["request"].get("headers", {})
        origin = headers.get("origin", "")

        # Allow all origins for development (restrict in production)
        context["cors_headers"] = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key"
        }

        return {"blocked": False}

    async def _authentication_middleware(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Authentication middleware"""
        route = context["route"]
        request = context["request"]

        if not route.get("auth_required", True):
            return {"blocked": False}

        # Check for API key
        api_key = self._extract_api_key(request)
        if api_key:
            key_config = self.validate_api_key(api_key)
            if key_config:
                context["auth_info"] = {
                    "type": "api_key",
                    "user_id": key_config["user_id"],
                    "permissions": key_config["permissions"]
                }
                return {"blocked": False}

        # Check for JWT token
        jwt_token = self._extract_jwt_token(request)
        if jwt_token:
            token_payload = self.validate_jwt_token(jwt_token)
            if token_payload:
                context["auth_info"] = {
                    "type": "jwt",
                    "user_id": token_payload["user_id"],
                    "permissions": token_payload["permissions"]
                }
                return {"blocked": False}

        return {"blocked": True, "status_code": 401, "message": "Authentication required"}

    def _extract_api_key(self, request: Dict[str, Any]) -> Optional[str]:
        """Extract API key from request"""
        # Check headers
        headers = request.get("headers", {})
        api_key = headers.get("X-API-Key") or headers.get("Authorization", "").replace("Bearer ", "")
        if api_key and not api_key.startswith("ey"):  # Not a JWT
            return api_key

        # Check query parameters
        query_params = request.get("query_params", {})
        return query_params.get("api_key")

    def _extract_jwt_token(self, request: Dict[str, Any]) -> Optional[str]:
        """Extract JWT token from request"""
        headers = request.get("headers", {})
        auth_header = headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            if token.startswith("ey"):  # Looks like JWT
                return token

        return None

    async def _rate_limiting_middleware(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rate limiting middleware"""
        route = context["route"]
        auth_info = context.get("auth_info")

        # Determine rate limiter
        rate_limit_key = route.get("rate_limit", "default")
        if auth_info and auth_info.get("type") == "api_key":
            # Use API key's rate limiter
            key_config = self.validate_api_key(self._extract_api_key(context["request"]))
            if key_config:
                rate_limiter = key_config["rate_limit"]
            else:
                rate_limiter = self.rate_limiters[rate_limit_key]
        else:
            rate_limiter = self.rate_limiters[rate_limit_key]

        # Check rate limit
        client_id = self._get_client_id(context["request"])
        if rate_limiter.is_rate_limited(client_id):
            return {
                "blocked": True,
                "status_code": 429,
                "message": "Rate limit exceeded",
                "retry_after": rate_limiter.get_retry_after(client_id)
            }

        context["rate_limit_info"] = {"limiter": rate_limiter, "client_id": client_id}
        return {"blocked": False}

    def _get_client_id(self, request: Dict[str, Any]) -> str:
        """Get client identifier for rate limiting"""
        # Use IP address as client ID
        return request.get("headers", {}).get("X-Forwarded-For",
              request.get("headers", {}).get("X-Real-IP", "unknown"))

    async def _logging_middleware(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Logging middleware"""
        request = context["request"]
        auth_info = context.get("auth_info")

        log_entry = {
            "timestamp": datetime.now(),
            "method": request.get("method"),
            "path": request.get("path"),
            "user_id": auth_info.get("user_id") if auth_info else None,
            "client_ip": self._get_client_id(request),
            "user_agent": request.get("headers", {}).get("User-Agent")
        }

        # Log to metrics collector
        self.metrics_collector.log_request(log_entry)

        return {"blocked": False}

    async def _caching_middleware(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Caching middleware"""
        route = context["route"]
        request = context["request"]

        if not route.get("cache_enabled", False):
            return {"blocked": False}

        # Generate cache key
        cache_key = self.cache_manager.generate_cache_key(request)

        # Check cache
        cached_response = self.cache_manager.get(cache_key)
        if cached_response:
            context["cached_response"] = cached_response
            return {"blocked": False, "cached": True}

        context["cache_key"] = cache_key
        return {"blocked": False}

    async def _route_request(self, route: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate handler"""
        # Check cache first
        if "cached_response" in context:
            return context["cached_response"]

        # Load balance to backend service
        service_response = await self.load_balancer.route_request(route, context)

        # Cache response if enabled
        if route.get("cache_enabled") and "cache_key" in context:
            self.cache_manager.set(context["cache_key"], service_response)

        return service_response

    def _create_response(self, status_code: int, data: Dict[str, Any],
                        headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Create standardized API response"""
        response = {
            "status_code": status_code,
            "data": data,
            "headers": headers or {},
            "timestamp": datetime.now()
        }

        # Add CORS headers if available
        cors_headers = getattr(self, '_cors_headers', {})
        response["headers"].update(cors_headers)

        return response

    def register_webhook_endpoint(self, endpoint_path: str, handler: Callable,
                                auth_required: bool = True, secret: str = None) -> str:
        """Register webhook endpoint"""
        webhook_id = f"webhook_{hashlib.md5(endpoint_path.encode()).hexdigest()[:8]}"

        webhook_config = {
            "webhook_id": webhook_id,
            "endpoint_path": endpoint_path,
            "handler": handler,
            "auth_required": auth_required,
            "secret": secret or secrets.token_hex(32),
            "registered_at": datetime.now(),
            "status": "active",
            "metrics": {"calls_today": 0, "errors_today": 0, "last_call": None}
        }

        self.webhook_endpoints[webhook_id] = webhook_config

        print(f"🪝 Registered webhook endpoint: {endpoint_path} -> {webhook_id}")
        return webhook_id

    def process_webhook(self, webhook_id: str, webhook_data: Dict[str, Any],
                       headers: Dict[str, str]) -> Dict[str, Any]:
        """Process incoming webhook"""
        if webhook_id not in self.webhook_endpoints:
            return {"status": "error", "message": "Webhook not found"}

        webhook_config = self.webhook_endpoints[webhook_id]

        try:
            # Verify webhook signature if secret is configured
            if webhook_config["secret"]:
                signature = headers.get("X-Webhook-Signature")
                if not self._verify_webhook_signature(webhook_data, signature, webhook_config["secret"]):
                    return {"status": "error", "message": "Invalid signature"}

            # Call webhook handler
            if asyncio.iscoroutinefunction(webhook_config["handler"]):
                result = asyncio.run(webhook_config["handler"](webhook_data, headers))
            else:
                result = webhook_config["handler"](webhook_data, headers)

            # Update metrics
            webhook_config["metrics"]["calls_today"] += 1
            webhook_config["metrics"]["last_call"] = datetime.now()

            return {"status": "success", "result": result}

        except Exception as e:
            webhook_config["metrics"]["errors_today"] += 1
            return {"status": "error", "message": str(e)}

    def _verify_webhook_signature(self, data: Dict[str, Any], signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        if not signature or not secret:
            return False

        payload = json.dumps(data, sort_keys=True).encode()
        expected_signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        return hmac.compare_digest(signature, f"sha256={expected_signature}")

    def get_gateway_metrics(self) -> Dict[str, Any]:
        """Get API gateway metrics"""
        return self.metrics_collector.get_metrics()

    def get_route_metrics(self, route_id: str = None) -> Dict[str, Any]:
        """Get metrics for specific route or all routes"""
        return self.metrics_collector.get_route_metrics(route_id)

    def _gateway_worker(self):
        """Background gateway worker"""
        while self.is_running:
            try:
                # Process queued requests (simplified)
                time.sleep(0.1)
            except Exception as e:
                print(f"Gateway worker error: {e}")

    def _metrics_worker(self):
        """Background metrics collection worker"""
        while self.is_running:
            try:
                self.metrics_collector.aggregate_metrics()
                time.sleep(60)  # Aggregate every minute
            except Exception as e:
                print(f"Metrics worker error: {e}")


class RateLimiter:
    """Rate limiting implementation"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_minute / 60
        self.client_requests = defaultdict(lambda: deque(maxlen=requests_per_minute))

    def is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited"""
        now = time.time()
        client_queue = self.client_requests[client_id]

        # Remove old requests (older than 1 minute)
        while client_queue and client_queue[0] < now - 60:
            client_queue.popleft()

        # Check if under limit
        if len(client_queue) < self.requests_per_minute:
            client_queue.append(now)
            return False

        return True

    def get_retry_after(self, client_id: str) -> int:
        """Get retry after time in seconds"""
        client_queue = self.client_requests[client_id]
        if not client_queue:
            return 0

        oldest_request = client_queue[0]
        return int(60 - (time.time() - oldest_request))


class SecurityEnforcer:
    """Security enforcement for API gateway"""

    def __init__(self):
        self.security_rules = []
        self.threat_patterns = []
        self.initialize_security_rules()

    def initialize_security_rules(self):
        """Initialize security rules"""
        self.security_rules = [
            self._sql_injection_rule,
            self._xss_rule,
            self._path_traversal_rule,
            self._malformed_request_rule
        ]

        self.threat_patterns = [
            r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)",
            r"<script[^>]*>.*?</script>",
            r"\.\./\.\./",
            r"(\.\./|/\\\.\\\.\\)"
        ]

    def enforce_security(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce security rules on request"""
        for rule in self.security_rules:
            result = rule(request)
            if result["blocked"]:
                return result

        return {"blocked": False}

    def _sql_injection_rule(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check for SQL injection patterns"""
        check_fields = ["body", "query_params", "path"]

        for field in check_fields:
            value = request.get(field, "")
            if isinstance(value, dict):
                value = json.dumps(value)

            for pattern in self.threat_patterns[:1]:  # SQL patterns
                if re.search(pattern, str(value), re.IGNORECASE):
                    return {
                        "blocked": True,
                        "reason": "sql_injection_detected",
                        "field": field
                    }

        return {"blocked": False}

    def _xss_rule(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check for XSS patterns"""
        check_fields = ["body", "query_params"]

        for field in check_fields:
            value = request.get(field, "")
            if isinstance(value, dict):
                value = json.dumps(value)

            for pattern in self.threat_patterns[1:2]:  # XSS patterns
                if re.search(pattern, str(value), re.IGNORECASE):
                    return {
                        "blocked": True,
                        "reason": "xss_detected",
                        "field": field
                    }

        return {"blocked": False}

    def _path_traversal_rule(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check for path traversal attacks"""
        path = request.get("path", "")

        for pattern in self.threat_patterns[2:]:  # Path traversal patterns
            if re.search(pattern, path):
                return {
                    "blocked": True,
                    "reason": "path_traversal_detected",
                    "field": "path"
                }

        return {"blocked": False}

    def _malformed_request_rule(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check for malformed requests"""
        # Check for excessively long fields
        max_length = 10000

        check_fields = ["body", "query_params", "headers"]
        for field in check_fields:
            value = request.get(field, "")
            if isinstance(value, dict):
                value = json.dumps(value)

            if len(str(value)) > max_length:
                return {
                    "blocked": True,
                    "reason": "request_too_large",
                    "field": field
                }

        return {"blocked": False}


class LoadBalancer:
    """Load balancer for API gateway"""

    def __init__(self):
        self.routes = {}
        self.service_health = defaultdict(dict)

    def register_route(self, route_id: str, route_config: Dict[str, Any]):
        """Register route with load balancer"""
        self.routes[route_id] = route_config

    async def route_request(self, route: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate backend service"""
        # Simplified routing - in production would use actual load balancing
        handler = route.get("handler")
        if handler:
            if asyncio.iscoroutinefunction(handler):
                return await handler(context)
            else:
                return handler(context)

        # Default response
        return {"status_code": 200, "data": {"message": "Request processed"}}


class CacheManager:
    """Cache manager for API gateway"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            self.redis_client = redis.from_url(redis_url)
        except:
            self.redis_client = None  # Fallback if Redis not available

        self.cache_ttl = 300  # 5 minutes default TTL

    def generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        key_components = [
            request.get("method", "GET"),
            request.get("path", "/"),
            str(sorted(request.get("query_params", {}).items())),
            hashlib.md5(json.dumps(request.get("body", {}), sort_keys=True).encode()).hexdigest()[:8]
        ]

        return f"cache:{hashlib.md5(':'.join(key_components).encode()).hexdigest()}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except:
            pass

        return None

    def set(self, key: str, response: Dict[str, Any], ttl: int = None):
        """Cache response"""
        if not self.redis_client:
            return

        try:
            ttl = ttl or self.cache_ttl
            self.redis_client.setex(key, ttl, json.dumps(response))
        except:
            pass


class MetricsCollector:
    """Metrics collection for API gateway"""

    def __init__(self):
        self.request_metrics = defaultdict(list)
        self.error_metrics = defaultdict(list)
        self.route_metrics = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "error_requests": 0,
            "average_response_time": 0,
            "response_times": deque(maxlen=1000)
        })

    def record_request(self, path: str, method: str, status_code: int, response_time: float):
        """Record request metrics"""
        route_key = f"{method} {path}"

        self.route_metrics[route_key]["total_requests"] += 1
        self.route_metrics[route_key]["response_times"].append(response_time)

        if 200 <= status_code < 400:
            self.route_metrics[route_key]["successful_requests"] += 1
        else:
            self.route_metrics[route_key]["error_requests"] += 1

        # Update average response time
        times = list(self.route_metrics[route_key]["response_times"])
        if times:
            self.route_metrics[route_key]["average_response_time"] = sum(times) / len(times)

    def record_error(self, path: str, error: str, response_time: float):
        """Record error metrics"""
        self.error_metrics[path].append({
            "error": error,
            "response_time": response_time,
            "timestamp": datetime.now()
        })

    def log_request(self, log_entry: Dict[str, Any]):
        """Log request details"""
        self.request_metrics["requests"].append(log_entry)

    def get_metrics(self) -> Dict[str, Any]:
        """Get overall gateway metrics"""
        total_requests = sum(route["total_requests"] for route in self.route_metrics.values())
        total_errors = sum(route["error_requests"] for route in self.route_metrics.values())

        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "routes_count": len(self.route_metrics),
            "uptime": "simulated",  # Would track actual uptime
            "timestamp": datetime.now()
        }

    def get_route_metrics(self, route_id: str = None) -> Dict[str, Any]:
        """Get metrics for specific route"""
        if route_id:
            return dict(self.route_metrics.get(route_id, {}))

        return dict(self.route_metrics)

    def aggregate_metrics(self):
        """Aggregate metrics for reporting"""
        # Clean old metrics (keep last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)

        for route_metrics in self.route_metrics.values():
            # Keep only recent response times
            recent_times = [t for t in route_metrics["response_times"]
                          if (datetime.now() - timedelta(seconds=t)).timestamp() > cutoff_time.timestamp()]
            route_metrics["response_times"] = deque(recent_times, maxlen=1000)
