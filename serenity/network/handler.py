# File: serenity/network/handler.py
# Created: 2024-12-24 15:41:52
# Updated: 2025-03-25 17:00:00
# Purpose: Network Communication Handler
# Version: 1.0.1

"""
Serenity Network Communication Handler
Manages HTTP and WebSocket communications with encryption
"""

import logging
import requests
import websockets
import asyncio
import threading
import queue
from typing import Dict, Optional
import json
from datetime import datetime

class NetworkHandler:
    def __init__(self):
        self.logger = logging.getLogger('Serenity.Network')
        self.config = None
        self.events = None
        self.request_queue = queue.Queue()
        self.running = False
        self.active_connections = set()
        self.loop = None
        self.server = None  # Store the server object

    def set_config(self, config):
        """Set configuration from ConfigManager."""
        self.config = config
        self.logger.info("Network configuration set")

    def set_events(self, events):
        """Set event handler for communication."""
        self.events = events
        self.events.subscribe("input_received", self.send_message)
        self.events.subscribe("system_shutdown", self.stop)
        self.logger.info("Network event handler set")

    def start(self):
        """Start the network handler."""
        try:
            if not self.config or not self.events:
                raise ValueError("Configuration or event handler not set")
            self.running = True

            # Start request processor and WebSocket server
            threading.Thread(target=self._process_requests, daemon=True).start()
            threading.Thread(target=self._start_websocket_server, daemon=True).start()
            self.logger.info("Network handler started")
        except Exception as e:
            self.logger.error(f"Network startup failed: {str(e)}")
            self.stop()

    def _process_requests(self):
        """Process queued HTTP requests."""
        while self.running:
            try:
                if not self.request_queue.empty():
                    request_data = self.request_queue.get_nowait()
                    self._handle_request(request_data)
                    self.request_queue.task_done()
            except queue.Empty:
                threading.Event().wait(0.1)
            except Exception as e:
                self.logger.error(f"Request processing error: {str(e)}")

    def _handle_request(self, request_data: Dict):
        """Handle an HTTP request."""
        try:
            url = request_data.get('url')
            method = request_data.get('method', 'GET').upper()
            payload = request_data.get('payload', {})

            if not url:
                self.logger.warning("No URL provided in request")
                return

            if method == 'GET':
                response = requests.get(url, timeout=5)
            elif method == 'POST':
                response = requests.post(url, json=payload, timeout=5)
            else:
                self.logger.warning(f"Unsupported method: {method}")
                return

            response.raise_for_status()
            self.events.emit("network_response", {
                "status": "success",
                "data": response.json() if response.text else {},
                "timestamp": self._get_current_time()
            }, priority=3)
            self.logger.debug(f"HTTP {method} to {url} succeeded")

        except requests.RequestException as e:
            self.logger.error(f"HTTP request failed: {str(e)}")
            self.events.emit("network_response", {
                "status": "error",
                "error": str(e),
                "timestamp": self._get_current_time()
            }, priority=3)

    async def _websocket_server(self, websocket, path):
        """Handle WebSocket connections."""
        try:
            conn_id = id(websocket)
            self.active_connections.add(conn_id)
            self.logger.info(f"WebSocket connection opened: {conn_id}")

            async for message in websocket:
                data = json.loads(message)
                self.events.emit("input_received", {
                    "text": data.get("text", ""),
                    "timestamp": self._get_current_time()
                }, priority=2)
                await websocket.send(json.dumps({"status": "received"}))

        except websockets.ConnectionClosed:
            self.logger.info(f"WebSocket connection closed: {conn_id}")
        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
        finally:
            self.active_connections.remove(conn_id)

    def _start_websocket_server(self):
        """Run the WebSocket server in its own event loop."""
        async def run_server():
            try:
                port = self.config.get('network', 'websocket_port', 8765)
                async with websockets.serve(self._websocket_server, "localhost", port) as server:
                    self.server = server
                    self.logger.info(f"WebSocket server running on port {port}")
                    await server.serve_forever()
            except Exception as e:
                self.logger.error(f"WebSocket server failed: {str(e)}")

        try:
            asyncio.run(run_server())
        except Exception as e:
            self.logger.error(f"WebSocket server startup failed: {str(e)}")

    def send_message(self, event_data: Dict):
        """Send a message via HTTP or WebSocket based on config."""
        try:
            if not self.running:
                self.logger.warning("Network handler not running")
                return

            text = event_data.get('text', '')
            if not text:
                self.logger.debug("Empty message to send")
                return

            # Use HTTP if configured, else queue for processing
            http_url = self.config.get('network', 'http_endpoint', None)
            if http_url:
                self.request_queue.put({
                    "url": http_url,
                    "method": "POST",
                    "payload": {"text": text}
                })
            self.logger.debug(f"Queued message for sending: {text}")

        except Exception as e:
            self.logger.error(f"Message send failed: {str(e)}")

    def get_network_stats(self) -> Dict:
        """Get network system statistics."""
        try:
            return {
                "active_connections": len(self.active_connections),
                "request_queue_size": self.request_queue.qsize(),
                "timestamp": self._get_current_time()
            }
        except Exception as e:
            self.logger.error(f"Network stats failed: {str(e)}")
            return {}

    def _get_current_time(self) -> str:
        """Helper to get current timestamp."""
        return datetime.now().isoformat()

    def stop(self, event_data=None):
        """Stop the network handler and clean up."""
        try:
            self.running = False
            if self.server:
                self.server.close()
            self.active_connections.clear()
            self.logger.info("Network handler stopped")
        except Exception as e:
            self.logger.error(f"Network shutdown failed: {str(e)}")

if __name__ == "__main__":
    from serenity.utils.config_manager import ConfigManager
    from serenity.utils.event_handler import EventHandler

    try:
        network = NetworkHandler()
        config = ConfigManager()
        events = EventHandler()
        network.set_config(config)
        network.set_events(events)
        network.start()

        events.emit("input_received", {"text": "Test message"})
        import time
        time.sleep(2)  # Let it process
        print("Network stats:", json.dumps(network.get_network_stats(), indent=2))
        network.stop()

    except Exception as e:
        logging.getLogger('Serenity.Network').error(f"Test failed: {str(e)}")