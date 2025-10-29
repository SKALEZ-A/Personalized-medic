"""
API Versioning System for AI Personalized Medicine Platform
Comprehensive versioning, backward compatibility, and migration support
"""

import json
import re
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import inspect
from packaging import version
import warnings

class APIVersion:
    """Represents an API version"""

    def __init__(self, version_string: str, release_date: datetime = None,
                 deprecated: bool = False, sunset_date: datetime = None):
        self.version_string = version_string
        self.version = version.parse(version_string)
        self.release_date = release_date or datetime.now()
        self.deprecated = deprecated
        self.sunset_date = sunset_date
        self.endpoints = {}
        self.changelog = []
        self.compatibility = {}

    def add_endpoint(self, path: str, methods: List[str], handler: Callable,
                    schema: Dict[str, Any] = None):
        """Add an endpoint to this version"""
        self.endpoints[path] = {
            'methods': methods,
            'handler': handler,
            'schema': schema or {},
            'added_in': self.version_string
        }

    def add_changelog_entry(self, entry: Dict[str, Any]):
        """Add a changelog entry"""
        self.changelog.append({
            **entry,
            'version': self.version_string,
            'date': datetime.now()
        })

    def set_compatibility(self, other_version: str, compatibility: str):
        """Set compatibility with another version"""
        self.compatibility[other_version] = compatibility

    def is_compatible_with(self, client_version: str) -> bool:
        """Check if this version is compatible with a client version"""
        try:
            client_ver = version.parse(client_version)
            # Major version must match for compatibility
            return self.version.major == client_ver.major
        except:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary"""
        return {
            'version': self.version_string,
            'release_date': self.release_date.isoformat(),
            'deprecated': self.deprecated,
            'sunset_date': self.sunset_date.isoformat() if self.sunset_date else None,
            'endpoints_count': len(self.endpoints),
            'changelog_count': len(self.changelog),
            'compatibility': self.compatibility
        }

class APIVersionManager:
    """Manages multiple API versions"""

    def __init__(self):
        self.versions = {}
        self.current_version = None
        self.version_pattern = re.compile(r'^v(\d+)\.(\d+)\.(\d+)$')
        self.deprecation_warnings_sent = set()

    def add_version(self, version: APIVersion) -> bool:
        """Add a new API version"""
        if version.version_string in self.versions:
            return False

        self.versions[version.version_string] = version

        # Set as current if it's the highest version
        if not self.current_version or version.version > self.versions[self.current_version].version:
            self.current_version = version.version_string

        return True

    def get_version(self, version_string: str) -> Optional[APIVersion]:
        """Get a specific API version"""
        return self.versions.get(version_string)

    def get_current_version(self) -> Optional[APIVersion]:
        """Get the current (latest) API version"""
        return self.versions.get(self.current_version) if self.current_version else None

    def get_supported_versions(self) -> List[str]:
        """Get list of supported (non-sunset) versions"""
        now = datetime.now()
        supported = []

        for version_str, version_obj in self.versions.items():
            if not version_obj.sunset_date or version_obj.sunset_date > now:
                supported.append(version_str)

        return sorted(supported, key=lambda x: version.parse(x), reverse=True)

    def parse_version_from_request(self, request) -> str:
        """Parse API version from request"""
        # Check Accept header
        accept_header = getattr(request, 'headers', {}).get('Accept', '')
        if 'application/vnd.api.' in accept_header:
            version_match = re.search(r'application/vnd\.api\.v(\d+\.\d+\.\d+)', accept_header)
            if version_match:
                return f"v{version_match.group(1)}"

        # Check URL path
        path = getattr(request, 'url', {}).get('path', '') or getattr(request, 'path', '')
        version_match = re.search(r'/api/v(\d+\.\d+\.\d+)/', path)
        if version_match:
            return f"v{version_match.group(1)}"

        # Check query parameter
        query_params = getattr(request, 'query_params', {}) or getattr(request, 'args', {})
        api_version = query_params.get('api_version') or query_params.get('version')
        if api_version and self.version_pattern.match(api_version):
            return api_version

        # Check custom header
        version_header = getattr(request, 'headers', {}).get('X-API-Version', '')
        if version_header and self.version_pattern.match(version_header):
            return version_header

        return None

    def get_appropriate_version(self, requested_version: str = None,
                              client_accepts: str = None) -> APIVersion:
        """Get the appropriate API version for a request"""
        # If specific version requested
        if requested_version and requested_version in self.versions:
            version_obj = self.versions[requested_version]
            # Check if version is still supported
            if version_obj.sunset_date and version_obj.sunset_date <= datetime.now():
                raise ValueError(f"API version {requested_version} is no longer supported")
            return version_obj

        # Default to current version
        return self.get_current_version()

    def check_version_compatibility(self, client_version: str, requested_version: str) -> Dict[str, Any]:
        """Check compatibility between client and requested versions"""
        client_ver = self.versions.get(client_version)
        requested_ver = self.versions.get(requested_version)

        if not client_ver or not requested_ver:
            return {'compatible': False, 'reason': 'Version not found'}

        # Check major version compatibility
        if client_ver.version.major != requested_ver.version.major:
            return {
                'compatible': False,
                'reason': 'Major version mismatch',
                'breaking_changes': True
            }

        # Check deprecation status
        if requested_ver.deprecated:
            return {
                'compatible': True,
                'deprecated': True,
                'sunset_date': requested_ver.sunset_date.isoformat() if requested_ver.sunset_date else None
            }

        return {'compatible': True}

    def generate_deprecation_warning(self, version: str) -> Dict[str, Any]:
        """Generate deprecation warning for a version"""
        version_obj = self.versions.get(version)
        if not version_obj or not version_obj.deprecated:
            return None

        return {
            'type': 'deprecation_warning',
            'version': version,
            'sunset_date': version_obj.sunset_date.isoformat() if version_obj.sunset_date else None,
            'message': f'API version {version} is deprecated and will be sunset on {version_obj.sunset_date.strftime("%Y-%m-%d") if version_obj.sunset_date else "a future date"}. Please migrate to a newer version.',
            'migration_guide': f'/api/docs/migration/{version}-to-{self.current_version}'
        }

class VersionedEndpoint:
    """Represents a versioned API endpoint"""

    def __init__(self, path: str, methods: List[str]):
        self.path = path
        self.methods = methods
        self.versions = {}
        self.middlewares = []
        self.rate_limits = {}
        self.auth_requirements = {}

    def add_version(self, version: str, handler: Callable,
                   schema: Dict[str, Any] = None, middlewares: List[Callable] = None):
        """Add a version-specific handler"""
        self.versions[version] = {
            'handler': handler,
            'schema': schema or {},
            'middlewares': middlewares or [],
            'added_in': version
        }

    def get_handler_for_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Get the appropriate handler for a version"""
        # Try exact version match
        if version in self.versions:
            return self.versions[version]

        # Find the latest compatible version
        version_obj = version.parse(version) if version else None
        compatible_versions = []

        for ver_str, ver_data in self.versions.items():
            ver_parsed = version.parse(ver_str)
            if not version_obj or (ver_parsed.major == version_obj.major and
                                 ver_parsed <= version_obj):
                compatible_versions.append((ver_parsed, ver_data))

        if compatible_versions:
            # Return the latest compatible version
            compatible_versions.sort(key=lambda x: x[0], reverse=True)
            return compatible_versions[0][1]

        return None

    def add_middleware(self, middleware: Callable, versions: List[str] = None):
        """Add middleware to specific versions or all versions"""
        if versions:
            for ver in versions:
                if ver in self.versions:
                    self.versions[ver]['middlewares'].append(middleware)
        else:
            self.middlewares.append(middleware)

    def set_rate_limit(self, limits: Dict[str, Any], versions: List[str] = None):
        """Set rate limiting for specific versions"""
        if versions:
            for ver in versions:
                if ver in self.versions:
                    self.versions[ver]['rate_limits'] = limits
        else:
            self.rate_limits = limits

    def set_auth_requirement(self, requirements: Dict[str, Any], versions: List[str] = None):
        """Set authentication requirements for specific versions"""
        if requirements:
            for ver in versions:
                if ver in self.versions:
                    self.versions[ver]['auth_requirements'] = requirements
        else:
            self.auth_requirements = requirements

class APIRouter:
    """Versioned API router"""

    def __init__(self, version_manager: APIVersionManager):
        self.version_manager = version_manager
        self.endpoints = {}
        self.middlewares = []
        self.global_rate_limits = {}

    def add_endpoint(self, endpoint: VersionedEndpoint):
        """Add a versioned endpoint"""
        self.endpoints[endpoint.path] = endpoint

    def add_middleware(self, middleware: Callable, versions: List[str] = None):
        """Add global middleware"""
        if versions:
            for version in versions:
                if version in self.version_manager.versions:
                    self.version_manager.versions[version].middlewares.append(middleware)
        else:
            self.middlewares.append(middleware)

    def set_global_rate_limit(self, limits: Dict[str, Any], versions: List[str] = None):
        """Set global rate limits"""
        if versions:
            for version in versions:
                if version in self.version_manager.versions:
                    self.version_manager.versions[version].rate_limits = limits
        else:
            self.global_rate_limits = limits

    def route_request(self, request) -> Optional[Dict[str, Any]]:
        """Route a request to the appropriate handler"""
        path = getattr(request, 'url', {}).get('path', '') or getattr(request, 'path', '')
        method = getattr(request, 'method', 'GET')

        # Remove version prefix from path for routing
        clean_path = re.sub(r'/api/v\d+\.\d+\.\d+/', '/api/', path)

        # Find matching endpoint
        for endpoint_path, endpoint in self.endpoints.items():
            if self._path_matches(endpoint_path, clean_path) and method in endpoint.methods:
                # Parse requested version
                requested_version = self.version_manager.parse_version_from_request(request)

                # Get appropriate version
                api_version = self.version_manager.get_appropriate_version(requested_version)

                # Get handler for version
                handler_data = endpoint.get_handler_for_version(api_version.version_string)

                if handler_data:
                    return {
                        'handler': handler_data['handler'],
                        'version': api_version,
                        'endpoint': endpoint,
                        'middlewares': self._collect_middlewares(api_version, endpoint, handler_data),
                        'rate_limits': self._collect_rate_limits(api_version, endpoint, handler_data),
                        'auth_requirements': self._collect_auth_requirements(api_version, endpoint, handler_data)
                    }

        return None

    def _path_matches(self, pattern: str, path: str) -> bool:
        """Check if path matches pattern with parameter support"""
        # Convert pattern to regex
        # e.g., /api/patients/{id} -> /api/patients/([^/]+)
        pattern_regex = re.sub(r'\{[^}]+\}', r'([^/]+)', pattern)
        return bool(re.match(f'^{pattern_regex}$', path))

    def _collect_middlewares(self, api_version: APIVersion, endpoint: VersionedEndpoint,
                           handler_data: Dict[str, Any]) -> List[Callable]:
        """Collect all applicable middlewares"""
        middlewares = []

        # Global middlewares
        middlewares.extend(self.middlewares)

        # Version-specific middlewares
        if hasattr(api_version, 'middlewares'):
            middlewares.extend(api_version.middlewares)

        # Endpoint middlewares
        middlewares.extend(endpoint.middlewares)

        # Handler-specific middlewares
        middlewares.extend(handler_data.get('middlewares', []))

        return middlewares

    def _collect_rate_limits(self, api_version: APIVersion, endpoint: VersionedEndpoint,
                           handler_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect applicable rate limits"""
        # Handler-specific takes precedence, then endpoint, then version, then global
        rate_limits = {}

        rate_limits.update(self.global_rate_limits)
        rate_limits.update(api_version.rate_limits if hasattr(api_version, 'rate_limits') else {})
        rate_limits.update(endpoint.rate_limits)
        rate_limits.update(handler_data.get('rate_limits', {}))

        return rate_limits

    def _collect_auth_requirements(self, api_version: APIVersion, endpoint: VersionedEndpoint,
                                 handler_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect applicable auth requirements"""
        auth_reqs = {}

        auth_reqs.update(self.version_manager.auth_requirements if hasattr(self.version_manager, 'auth_requirements') else {})
        auth_reqs.update(endpoint.auth_requirements)
        auth_reqs.update(handler_data.get('auth_requirements', {}))

        return auth_reqs

class APIMigrationGuide:
    """API migration guide generator"""

    def __init__(self, version_manager: APIVersionManager):
        self.version_manager = version_manager
        self.migration_paths = defaultdict(dict)

    def add_migration_path(self, from_version: str, to_version: str,
                          changes: List[Dict[str, Any]], breaking: bool = False):
        """Add a migration path between versions"""
        self.migration_paths[from_version][to_version] = {
            'changes': changes,
            'breaking': breaking,
            'created_at': datetime.now()
        }

    def generate_migration_guide(self, from_version: str, to_version: str) -> Dict[str, Any]:
        """Generate a migration guide"""
        if from_version not in self.migration_paths or to_version not in self.migration_paths[from_version]:
            return {'error': 'Migration path not found'}

        migration = self.migration_paths[from_version][to_version]

        return {
            'from_version': from_version,
            'to_version': to_version,
            'breaking_changes': migration['breaking'],
            'changes': migration['changes'],
            'migration_steps': self._generate_migration_steps(migration['changes']),
            'testing_guide': self._generate_testing_guide(migration['changes']),
            'rollback_plan': self._generate_rollback_plan(from_version, to_version)
        }

    def _generate_migration_steps(self, changes: List[Dict[str, Any]]) -> List[str]:
        """Generate step-by-step migration instructions"""
        steps = []

        for change in changes:
            change_type = change.get('type')

            if change_type == 'endpoint_added':
                steps.append(f"Update client to use new endpoint: {change.get('endpoint')}")
            elif change_type == 'endpoint_removed':
                steps.append(f"Replace usage of removed endpoint: {change.get('endpoint')} with: {change.get('replacement', 'alternative endpoint')}")
            elif change_type == 'parameter_changed':
                steps.append(f"Update parameter '{change.get('parameter')}' in endpoint: {change.get('endpoint')}")
            elif change_type == 'response_format_changed':
                steps.append(f"Update response parsing for endpoint: {change.get('endpoint')} - {change.get('details')}")
            elif change_type == 'authentication_changed':
                steps.append(f"Update authentication method for endpoint: {change.get('endpoint')} - {change.get('details')}")

        return steps

    def _generate_testing_guide(self, changes: List[Dict[str, Any]]) -> List[str]:
        """Generate testing recommendations"""
        tests = []

        for change in changes:
            change_type = change.get('type')

            if change_type in ['endpoint_added', 'endpoint_removed', 'parameter_changed']:
                tests.append(f"Test all API calls to: {change.get('endpoint')}")
            elif change_type == 'response_format_changed':
                tests.append(f"Test response parsing for: {change.get('endpoint')}")
            elif change_type == 'authentication_changed':
                tests.append("Test authentication flows with new method")

        tests.extend([
            "Test error handling for all endpoints",
            "Test rate limiting behavior",
            "Test backward compatibility if applicable",
            "Run integration tests with real API",
            "Monitor for increased error rates after migration"
        ])

        return tests

    def _generate_rollback_plan(self, from_version: str, to_version: str) -> Dict[str, Any]:
        """Generate rollback plan"""
        return {
            'rollback_version': from_version,
            'steps': [
                f"Change API version header to: {from_version}",
                "Update client configuration to use previous version",
                "Restart client application",
                "Verify functionality with previous API version",
                "Monitor for any issues after rollback"
            ],
            'estimated_time': '5-15 minutes',
            'risk_level': 'low',
            'prerequisites': [
                f'Ensure {from_version} is still supported',
                'Have backup of current configuration',
                'Test rollback in staging environment first'
            ]
        }

class APIResponseFormatter:
    """API response formatting and content negotiation"""

    def __init__(self):
        self.formatters = {
            'application/json': self._format_json,
            'application/xml': self._format_xml,
            'application/vnd.api+json': self._format_json_api,
            'text/csv': self._format_csv,
            'application/octet-stream': self._format_binary
        }

    def format_response(self, data: Any, content_type: str = 'application/json',
                       version: str = None) -> tuple:
        """Format response based on content type"""
        formatter = self.formatters.get(content_type, self._format_json)

        try:
            formatted_data = formatter(data, version)
            return formatted_data, content_type
        except Exception as e:
            # Fallback to JSON
            return self._format_json({'error': f'Formatting error: {str(e)}'}), 'application/json'

    def _format_json(self, data: Any, version: str = None) -> str:
        """Format as JSON"""
        response = {
            'data': data,
            'meta': {
                'version': version or 'unknown',
                'timestamp': datetime.now().isoformat(),
                'format': 'json'
            }
        }
        return json.dumps(response, indent=2, default=str)

    def _format_xml(self, data: Any, version: str = None) -> str:
        """Format as XML"""
        # Simple XML formatter (would use proper XML library in production)
        def dict_to_xml(d, root_name='response'):
            xml = f'<{root_name}>'
            if isinstance(d, dict):
                for key, value in d.items():
                    xml += f'<{key}>{dict_to_xml(value, key)}</{key}>'
            elif isinstance(d, list):
                for item in d:
                    xml += dict_to_xml(item, 'item')
            else:
                xml += str(d)
            xml += f'</{root_name}>'
            return xml

        return f'<?xml version="1.0" encoding="UTF-8"?>{dict_to_xml(data)}'

    def _format_json_api(self, data: Any, version: str = None) -> str:
        """Format as JSON:API"""
        response = {
            'data': data,
            'jsonapi': {
                'version': '1.0',
                'meta': {
                    'api_version': version or 'unknown'
                }
            }
        }
        return json.dumps(response, indent=2, default=str)

    def _format_csv(self, data: Any, version: str = None) -> str:
        """Format as CSV"""
        if not isinstance(data, list) or not data:
            return 'No data available'

        # Get headers from first item
        headers = list(data[0].keys()) if isinstance(data[0], dict) else ['value']

        # Create CSV
        csv_lines = [','.join(headers)]

        for item in data:
            if isinstance(item, dict):
                row = [str(item.get(header, '')) for header in headers]
            else:
                row = [str(item)]
            csv_lines.append(','.join(row))

        return '\n'.join(csv_lines)

    def _format_binary(self, data: Any, version: str = None) -> bytes:
        """Format as binary"""
        if isinstance(data, bytes):
            return data
        elif isinstance(data, str):
            return data.encode('utf-8')
        else:
            return json.dumps(data, default=str).encode('utf-8')

# Create global instances
version_manager = APIVersionManager()
api_router = APIRouter(version_manager)
migration_guide = APIMigrationGuide(version_manager)
response_formatter = APIResponseFormatter()
