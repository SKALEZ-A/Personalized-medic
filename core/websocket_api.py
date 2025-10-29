"""
WebSocket API for AI Personalized Medicine Platform
Real-time communication for health monitoring, alerts, and live updates
"""

import asyncio
import json
import websockets
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from collections import defaultdict
import random
import hashlib
import jwt
from concurrent.futures import ThreadPoolExecutor
import queue

class WebSocketConnectionManager:
    """Manages WebSocket connections and real-time communication"""

    def __init__(self):
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.rooms: Dict[str, Set[str]] = defaultdict(set)
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 300  # 5 minutes
        self.max_connections_per_user = 5
        self.message_queue = queue.Queue()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=10)

    def start(self):
        """Start the WebSocket manager"""
        self.is_running = True

        # Start heartbeat monitoring
        heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        heartbeat_thread.start()

        # Start message processing
        message_thread = threading.Thread(target=self._process_messages, daemon=True)
        message_thread.start()

        # Start cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_connections, daemon=True)
        cleanup_thread.start()

        print("WebSocket Connection Manager started")

    def stop(self):
        """Stop the WebSocket manager"""
        self.is_running = False

        # Close all connections
        for connection_id, connection in self.connections.items():
            if connection.get('websocket'):
                asyncio.create_task(self._close_connection(connection_id))

        print("WebSocket Connection Manager stopped")

    async def register_connection(self, websocket, client_id: str, user_id: str = None,
                                auth_token: str = None) -> str:
        """Register a new WebSocket connection"""
        connection_id = f"conn_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"

        # Validate user limit
        user_connections = [cid for cid, conn in self.connections.items()
                          if conn.get('user_id') == user_id]
        if len(user_connections) >= self.max_connections_per_user:
            # Close oldest connection
            oldest_conn = min(user_connections,
                            key=lambda x: self.connections[x]['connected_at'])
            await self._close_connection(oldest_conn)

        # Validate authentication if required
        if auth_token and not self._validate_token(auth_token):
            raise websockets.exceptions.ConnectionClosedError(1008, "Invalid token")

        self.connections[connection_id] = {
            'websocket': websocket,
            'client_id': client_id,
            'user_id': user_id,
            'connected_at': datetime.now(),
            'last_heartbeat': datetime.now(),
            'subscribed_rooms': set(),
            'metadata': {}
        }

        print(f"Connection registered: {connection_id} for user {user_id}")

        # Send welcome message
        await self.send_to_connection(connection_id, {
            'type': 'connection_established',
            'connection_id': connection_id,
            'timestamp': datetime.now().isoformat()
        })

        return connection_id

    async def unregister_connection(self, connection_id: str):
        """Unregister a WebSocket connection"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]

            # Remove from rooms
            for room in connection['subscribed_rooms']:
                self.rooms[room].discard(connection_id)

            # Close websocket if still open
            if connection.get('websocket'):
                try:
                    await connection['websocket'].close()
                except:
                    pass

            del self.connections[connection_id]
            print(f"Connection unregistered: {connection_id}")

    async def subscribe_to_room(self, connection_id: str, room: str):
        """Subscribe connection to a room"""
        if connection_id not in self.connections:
            return False

        self.connections[connection_id]['subscribed_rooms'].add(room)
        self.rooms[room].add(connection_id)

        await self.send_to_connection(connection_id, {
            'type': 'subscribed',
            'room': room,
            'timestamp': datetime.now().isoformat()
        })

        print(f"Connection {connection_id} subscribed to room {room}")
        return True

    async def unsubscribe_from_room(self, connection_id: str, room: str):
        """Unsubscribe connection from a room"""
        if connection_id not in self.connections:
            return False

        self.connections[connection_id]['subscribed_rooms'].discard(room)
        self.rooms[room].discard(connection_id)

        await self.send_to_connection(connection_id, {
            'type': 'unsubscribed',
            'room': room,
            'timestamp': datetime.now().isoformat()
        })

        return True

    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection"""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        websocket = connection.get('websocket')

        if websocket and not websocket.closed:
            try:
                await websocket.send(json.dumps(message))
                return True
            except Exception as e:
                print(f"Failed to send message to {connection_id}: {e}")
                await self.unregister_connection(connection_id)
                return False

        return False

    async def send_to_room(self, room: str, message: Dict[str, Any], exclude_connection: str = None):
        """Send message to all connections in a room"""
        if room not in self.rooms:
            return 0

        sent_count = 0
        for connection_id in self.rooms[room]:
            if connection_id != exclude_connection:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1

        return sent_count

    async def broadcast(self, message: Dict[str, Any], exclude_connection: str = None):
        """Broadcast message to all connections"""
        sent_count = 0
        for connection_id in list(self.connections.keys()):
            if connection_id != exclude_connection:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1

        return sent_count

    def queue_message(self, message: Dict[str, Any]):
        """Queue message for processing"""
        self.message_queue.put(message)

    async def _close_connection(self, connection_id: str):
        """Close a WebSocket connection"""
        if connection_id in self.connections:
            websocket = self.connections[connection_id].get('websocket')
            if websocket and not websocket.closed:
                try:
                    await websocket.close()
                except:
                    pass

    def _validate_token(self, token: str) -> bool:
        """Validate JWT token"""
        try:
            # Mock token validation - would use proper JWT verification
            decoded = jwt.decode(token, 'secret_key', algorithms=['HS256'])
            return decoded.get('valid', False)
        except:
            return False

    def _heartbeat_monitor(self):
        """Monitor connection heartbeats"""
        while self.is_running:
            try:
                now = datetime.now()
                timeout_threshold = now - timedelta(seconds=self.connection_timeout)

                # Check for timed out connections
                timed_out = []
                for connection_id, connection in self.connections.items():
                    if connection['last_heartbeat'] < timeout_threshold:
                        timed_out.append(connection_id)

                # Close timed out connections
                for connection_id in timed_out:
                    print(f"Connection {connection_id} timed out")
                    asyncio.create_task(self.unregister_connection(connection_id))

                # Send heartbeat pings
                for connection in list(self.connections.values()):
                    if connection.get('websocket') and not connection['websocket'].closed:
                        asyncio.create_task(self._send_heartbeat(connection))

                time.sleep(self.heartbeat_interval)

            except Exception as e:
                print(f"Heartbeat monitor error: {e}")

    async def _send_heartbeat(self, connection):
        """Send heartbeat to connection"""
        websocket = connection.get('websocket')
        if websocket and not websocket.closed:
            try:
                await websocket.send(json.dumps({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat()
                }))
            except:
                pass

    def _process_messages(self):
        """Process queued messages"""
        while self.is_running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(timeout=1)

                    # Process message asynchronously
                    asyncio.create_task(self._handle_message(message))

                time.sleep(0.1)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Message processing error: {e}")

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle queued message"""
        message_type = message.get('type', 'unknown')

        if message_type == 'health_metric':
            await self._handle_health_metric(message)
        elif message_type == 'alert':
            await self._handle_alert(message)
        elif message_type == 'appointment_update':
            await self._handle_appointment_update(message)
        elif message_type == 'medication_reminder':
            await self._handle_medication_reminder(message)

    async def _handle_health_metric(self, message: Dict[str, Any]):
        """Handle health metric message"""
        patient_id = message.get('patient_id')
        if patient_id:
            room = f"patient_{patient_id}"
            await self.send_to_room(room, {
                'type': 'health_metric_update',
                'data': message,
                'timestamp': datetime.now().isoformat()
            })

    async def _handle_alert(self, message: Dict[str, Any]):
        """Handle alert message"""
        patient_id = message.get('patient_id')
        severity = message.get('severity', 'info')

        if patient_id:
            room = f"patient_{patient_id}"
            await self.send_to_room(room, {
                'type': 'alert',
                'severity': severity,
                'data': message,
                'timestamp': datetime.now().isoformat()
            })

    async def _handle_appointment_update(self, message: Dict[str, Any]):
        """Handle appointment update"""
        patient_id = message.get('patient_id')
        doctor_id = message.get('doctor_id')

        if patient_id:
            await self.send_to_room(f"patient_{patient_id}", {
                'type': 'appointment_update',
                'data': message,
                'timestamp': datetime.now().isoformat()
            })

        if doctor_id:
            await self.send_to_room(f"doctor_{doctor_id}", {
                'type': 'appointment_update',
                'data': message,
                'timestamp': datetime.now().isoformat()
            })

    async def _handle_medication_reminder(self, message: Dict[str, Any]):
        """Handle medication reminder"""
        patient_id = message.get('patient_id')
        if patient_id:
            await self.send_to_room(f"patient_{patient_id}", {
                'type': 'medication_reminder',
                'data': message,
                'timestamp': datetime.now().isoformat()
            })

    def _cleanup_connections(self):
        """Clean up stale connections"""
        while self.is_running:
            try:
                # Remove connections that have been disconnected for too long
                now = datetime.now()
                stale_threshold = now - timedelta(hours=1)

                stale_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.get('disconnected_at') and connection['disconnected_at'] < stale_threshold:
                        stale_connections.append(connection_id)

                for connection_id in stale_connections:
                    print(f"Cleaning up stale connection: {connection_id}")
                    del self.connections[connection_id]

                    # Clean up from rooms
                    for room, connections in self.rooms.items():
                        connections.discard(connection_id)
                        if not connections:
                            del self.rooms[room]

                time.sleep(300)  # Clean up every 5 minutes

            except Exception as e:
                print(f"Connection cleanup error: {e}")

class RealTimeHealthMonitor:
    """Real-time health monitoring via WebSocket"""

    def __init__(self, connection_manager: WebSocketConnectionManager):
        self.manager = connection_manager
        self.active_monitors = {}
        self.monitoring_data = defaultdict(list)
        self.alert_thresholds = {
            'heart_rate': {'min': 50, 'max': 150},
            'blood_pressure_systolic': {'min': 90, 'max': 180},
            'blood_pressure_diastolic': {'min': 60, 'max': 120},
            'blood_glucose': {'min': 70, 'max': 200},
            'oxygen_saturation': {'min': 95, 'max': 100}
        }

    def start_monitoring(self, patient_id: str, device_id: str):
        """Start real-time monitoring for a patient"""
        monitor_id = f"monitor_{patient_id}_{device_id}"

        if monitor_id in self.active_monitors:
            return False

        self.active_monitors[monitor_id] = {
            'patient_id': patient_id,
            'device_id': device_id,
            'started_at': datetime.now(),
            'last_update': datetime.now(),
            'metrics_count': 0
        }

        print(f"Started monitoring: {monitor_id}")
        return True

    def stop_monitoring(self, patient_id: str, device_id: str):
        """Stop monitoring for a patient"""
        monitor_id = f"monitor_{patient_id}_{device_id}"

        if monitor_id in self.active_monitors:
            del self.active_monitors[monitor_id]
            print(f"Stopped monitoring: {monitor_id}")
            return True

        return False

    async def process_health_data(self, patient_id: str, device_id: str,
                                metric_type: str, value: float, timestamp: datetime = None):
        """Process incoming health data"""
        if timestamp is None:
            timestamp = datetime.now()

        # Store monitoring data
        data_point = {
            'patient_id': patient_id,
            'device_id': device_id,
            'metric_type': metric_type,
            'value': value,
            'timestamp': timestamp,
            'quality_score': self._calculate_data_quality(value, metric_type)
        }

        self.monitoring_data[patient_id].append(data_point)

        # Keep only recent data (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.monitoring_data[patient_id] = [
            d for d in self.monitoring_data[patient_id]
            if d['timestamp'] > cutoff
        ]

        # Check for alerts
        alerts = self._check_alerts(data_point)

        # Update monitor stats
        monitor_id = f"monitor_{patient_id}_{device_id}"
        if monitor_id in self.active_monitors:
            self.active_monitors[monitor_id]['last_update'] = datetime.now()
            self.active_monitors[monitor_id]['metrics_count'] += 1

        # Send real-time update
        await self.manager.send_to_room(f"patient_{patient_id}", {
            'type': 'health_data',
            'data': data_point,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        })

        # Send alerts if any
        for alert in alerts:
            await self.manager.queue_message({
                'type': 'alert',
                'patient_id': patient_id,
                'severity': alert['severity'],
                'message': alert['message'],
                'metric_type': metric_type,
                'value': value,
                'threshold': alert.get('threshold')
            })

    def _calculate_data_quality(self, value: float, metric_type: str) -> float:
        """Calculate data quality score"""
        if metric_type not in self.alert_thresholds:
            return 0.8  # Default quality

        thresholds = self.alert_thresholds[metric_type]
        min_val = thresholds.get('min', value - 10)
        max_val = thresholds.get('max', value + 10)

        if min_val <= value <= max_val:
            return 1.0  # Perfect quality
        elif value < min_val * 0.5 or value > max_val * 1.5:
            return 0.3  # Poor quality - way out of range
        else:
            return 0.7  # Acceptable quality - slightly out of range

    def _check_alerts(self, data_point: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        metric_type = data_point['metric_type']
        value = data_point['value']

        if metric_type in self.alert_thresholds:
            thresholds = self.alert_thresholds[metric_type]

            if value < thresholds.get('min', float('-inf')):
                alerts.append({
                    'severity': 'warning',
                    'message': f'{metric_type.replace("_", " ").title()} is below normal range',
                    'threshold': thresholds['min'],
                    'direction': 'low'
                })

            if value > thresholds.get('max', float('inf')):
                severity = 'critical' if value > thresholds['max'] * 1.2 else 'warning'
                alerts.append({
                    'severity': severity,
                    'message': f'{metric_type.replace("_", " ").title()} is above normal range',
                    'threshold': thresholds['max'],
                    'direction': 'high'
                })

        # Check for rapid changes (trending alerts)
        recent_data = self.monitoring_data[data_point['patient_id']][-10:]  # Last 10 readings
        if len(recent_data) >= 5:
            recent_values = [d['value'] for d in recent_data if d['metric_type'] == metric_type]
            if len(recent_values) >= 5:
                # Check for significant upward trend
                if self._detect_trend(recent_values, direction='up', threshold=20):
                    alerts.append({
                        'severity': 'warning',
                        'message': f'{metric_type.replace("_", " ").title()} showing rapid increase',
                        'trend': 'increasing'
                    })

                # Check for significant downward trend
                if self._detect_trend(recent_values, direction='down', threshold=20):
                    alerts.append({
                        'severity': 'warning',
                        'message': f'{metric_type.replace("_", " ").title()} showing rapid decrease',
                        'trend': 'decreasing'
                    })

        return alerts

    def _detect_trend(self, values: List[float], direction: str, threshold: float) -> bool:
        """Detect trend in values"""
        if len(values) < 3:
            return False

        # Calculate percentage change
        first_avg = sum(values[:len(values)//3]) / (len(values)//3)
        last_avg = sum(values[-len(values)//3:]) / (len(values)//3)

        if first_avg == 0:
            return False

        change_percent = ((last_avg - first_avg) / first_avg) * 100

        if direction == 'up':
            return change_percent > threshold
        elif direction == 'down':
            return change_percent < -threshold

        return False

    def get_monitoring_stats(self, patient_id: str) -> Dict[str, Any]:
        """Get monitoring statistics for a patient"""
        data = self.monitoring_data.get(patient_id, [])

        if not data:
            return {'active': False, 'metrics_count': 0}

        # Calculate statistics
        metrics_by_type = defaultdict(list)
        for point in data:
            metrics_by_type[point['metric_type']].append(point['value'])

        stats = {
            'active': True,
            'total_metrics': len(data),
            'metrics_by_type': {},
            'time_range': {
                'start': min(d['timestamp'] for d in data),
                'end': max(d['timestamp'] for d in data)
            },
            'data_quality': {
                'avg_quality': sum(d['quality_score'] for d in data) / len(data),
                'high_quality_count': sum(1 for d in data if d['quality_score'] > 0.8)
            }
        }

        for metric_type, values in metrics_by_type.items():
            stats['metrics_by_type'][metric_type] = {
                'count': len(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1]
            }

        return stats

class WebSocketAPI:
    """Main WebSocket API class"""

    def __init__(self):
        self.connection_manager = WebSocketConnectionManager()
        self.health_monitor = RealTimeHealthMonitor(self.connection_manager)
        self.message_handlers = {
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
            'ping': self._handle_ping,
            'health_data': self._handle_health_data,
            'start_monitoring': self._handle_start_monitoring,
            'stop_monitoring': self._handle_stop_monitoring
        }

    def start(self):
        """Start the WebSocket API"""
        self.connection_manager.start()

    def stop(self):
        """Stop the WebSocket API"""
        self.connection_manager.stop()

    async def handle_connection(self, websocket, path):
        """Handle WebSocket connection"""
        # Parse query parameters
        query_params = {}
        if '?' in path:
            query_string = path.split('?', 1)[1]
            for param in query_string.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    query_params[key] = value

        client_id = query_params.get('client_id', f"client_{random.randint(1000, 9999)}")
        user_id = query_params.get('user_id')
        auth_token = query_params.get('token')

        try:
            # Register connection
            connection_id = await self.connection_manager.register_connection(
                websocket, client_id, user_id, auth_token
            )

            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(connection_id, data)
                except json.JSONDecodeError:
                    await self.connection_manager.send_to_connection(connection_id, {
                        'type': 'error',
                        'message': 'Invalid JSON message',
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    await self.connection_manager.send_to_connection(connection_id, {
                        'type': 'error',
                        'message': f'Message handling error: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    })

        except websockets.exceptions.ConnectionClosedError:
            pass
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            await self.connection_manager.unregister_connection(connection_id)

    async def _handle_message(self, connection_id: str, data: Dict[str, Any]):
        """Handle incoming message"""
        message_type = data.get('type', 'unknown')

        handler = self.message_handlers.get(message_type)
        if handler:
            try:
                await handler(connection_id, data)
            except Exception as e:
                await self.connection_manager.send_to_connection(connection_id, {
                    'type': 'error',
                    'message': f'Handler error: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                })
        else:
            await self.connection_manager.send_to_connection(connection_id, {
                'type': 'error',
                'message': f'Unknown message type: {message_type}',
                'timestamp': datetime.now().isoformat()
            })

    async def _handle_subscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscribe message"""
        room = data.get('room')
        if room:
            success = await self.connection_manager.subscribe_to_room(connection_id, room)
            await self.connection_manager.send_to_connection(connection_id, {
                'type': 'subscribe_response',
                'room': room,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })

    async def _handle_unsubscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle unsubscribe message"""
        room = data.get('room')
        if room:
            success = await self.connection_manager.unsubscribe_from_room(connection_id, room)
            await self.connection_manager.send_to_connection(connection_id, {
                'type': 'unsubscribe_response',
                'room': room,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })

    async def _handle_ping(self, connection_id: str, data: Dict[str, Any]):
        """Handle ping message"""
        await self.connection_manager.send_to_connection(connection_id, {
            'type': 'pong',
            'timestamp': datetime.now().isoformat()
        })

    async def _handle_health_data(self, connection_id: str, data: Dict[str, Any]):
        """Handle health data message"""
        health_data = data.get('data', {})

        await self.health_monitor.process_health_data(
            patient_id=health_data.get('patient_id'),
            device_id=health_data.get('device_id', 'unknown'),
            metric_type=health_data.get('metric_type'),
            value=health_data.get('value'),
            timestamp=health_data.get('timestamp')
        )

        await self.connection_manager.send_to_connection(connection_id, {
            'type': 'health_data_ack',
            'success': True,
            'timestamp': datetime.now().isoformat()
        })

    async def _handle_start_monitoring(self, connection_id: str, data: Dict[str, Any]):
        """Handle start monitoring message"""
        patient_id = data.get('patient_id')
        device_id = data.get('device_id')

        if patient_id and device_id:
            success = self.health_monitor.start_monitoring(patient_id, device_id)

            # Subscribe to patient's room
            if success:
                await self.connection_manager.subscribe_to_room(connection_id, f"patient_{patient_id}")

            await self.connection_manager.send_to_connection(connection_id, {
                'type': 'monitoring_started',
                'patient_id': patient_id,
                'device_id': device_id,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })

    async def _handle_stop_monitoring(self, connection_id: str, data: Dict[str, Any]):
        """Handle stop monitoring message"""
        patient_id = data.get('patient_id')
        device_id = data.get('device_id')

        if patient_id and device_id:
            success = self.health_monitor.stop_monitoring(patient_id, device_id)

            await self.connection_manager.send_to_connection(connection_id, {
                'type': 'monitoring_stopped',
                'patient_id': patient_id,
                'device_id': device_id,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })

    def broadcast_health_alert(self, patient_id: str, alert_data: Dict[str, Any]):
        """Broadcast health alert to patient's connections"""
        self.connection_manager.queue_message({
            'type': 'alert',
            'patient_id': patient_id,
            **alert_data
        })

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'total_connections': len(self.connection_manager.connections),
            'active_rooms': len(self.connection_manager.rooms),
            'active_monitors': len(self.health_monitor.active_monitors),
            'total_rooms': sum(len(connections) for connections in self.connection_manager.rooms.values())
        }

# Create WebSocket API instance
websocket_api = WebSocketAPI()
