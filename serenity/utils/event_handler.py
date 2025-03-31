# File: serenity/utils/event_handler.py
# Created: 2024-12-24 15:34:12
# Updated: 2025-03-26 20:00:00
# Purpose: Event Handling System
# Version: 1.0.4

"""
Serenity Event Handling System
Manages event subscriptions and emissions with priority-based queuing.
Includes event batching, queue size monitoring, and diagnostic events.
"""

import logging
from typing import Callable, Dict, List, Any
import queue
import threading
import time

class EventHandler:
    def __init__(self):
        self.logger = logging.getLogger('Serenity.EventHandler')
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.event_queue = queue.PriorityQueue()
        self.running = False
        self.lock = threading.Lock()
        self.batch_size = 10  # Process up to 10 events at a time
        self.max_queue_size = 1000  # Maximum queue size before warning
        self.logger.debug("EventHandler initialized")

    def subscribe(self, event_name: str, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe a callback to an event."""
        try:
            with self.lock:
                if event_name not in self.subscribers:
                    self.subscribers[event_name] = []
                if callback not in self.subscribers[event_name]:
                    self.subscribers[event_name].append(callback)
                    self.logger.debug(f"Subscribed to event: {event_name}")
        except Exception as e:
            self.logger.error(f"Failed to subscribe to event {event_name}: {str(e)}")
            self.emit("event_diagnostic", {
                "message": f"Failed to subscribe to event {event_name}: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def unsubscribe(self, event_name: str, callback: Callable[[Dict[str, Any]], None]):
        """Unsubscribe a callback from an event."""
        try:
            with self.lock:
                if event_name in self.subscribers and callback in self.subscribers[event_name]:
                    self.subscribers[event_name].remove(callback)
                    self.logger.debug(f"Unsubscribed from event: {event_name}")
                    if not self.subscribers[event_name]:
                        del self.subscribers[event_name]
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from event {event_name}: {str(e)}")
            self.emit("event_diagnostic", {
                "message": f"Failed to unsubscribe from event {event_name}: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def emit(self, event_name: str, data: Dict[str, Any], priority: int = 1):
        """Emit an event with data and priority."""
        try:
            if not self.running:
                self.logger.warning(f"EventHandler not running, cannot emit event: {event_name}")
                return
            # Check queue size before adding
            queue_size = self.event_queue.qsize()
            if queue_size >= self.max_queue_size:
                self.logger.warning(f"Event queue size exceeded: {queue_size}/{self.max_queue_size}")
                self.emit("event_diagnostic", {
                    "message": f"Event queue size exceeded: {queue_size}/{self.max_queue_size}",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }, priority=1)
                return
            self.event_queue.put((priority, time.time(), event_name, data))
            self.logger.debug(f"Emitted event: {event_name} with priority {priority}, queue size: {self.event_queue.qsize()}")
        except Exception as e:
            self.logger.error(f"Failed to emit event {event_name}: {str(e)}")
            self.emit("event_diagnostic", {
                "message": f"Failed to emit event {event_name}: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def start(self):
        """Start the event processing loop."""
        try:
            if self.running:
                self.logger.warning("EventHandler already running")
                return
            self.running = True
            threading.Thread(target=self.process_events, daemon=True).start()
            self.logger.info("EventHandler started")
        except Exception as e:
            self.logger.error(f"Failed to start EventHandler: {str(e)}")
            self.emit("event_diagnostic", {
                "message": f"Failed to start EventHandler: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            self.stop()

    def process_events(self):
        """Process events from the queue in batches."""
        try:
            self.logger.info("Event processing loop started")
            while self.running:
                batch = []
                try:
                    # Collect up to batch_size events
                    for _ in range(self.batch_size):
                        try:
                            item = self.event_queue.get(timeout=0.1)
                            batch.append(item)
                        except queue.Empty:
                            break

                    if not batch:
                        continue

                    for priority, timestamp, event_name, data in batch:
                        with self.lock:
                            if event_name in self.subscribers:
                                for callback in self.subscribers[event_name]:
                                    try:
                                        callback(data)
                                        self.logger.debug(f"Processed event: {event_name} with data: {data}")
                                    except Exception as e:
                                        self.logger.error(f"Error in callback for event {event_name}: {str(e)}")
                                        self.emit("event_diagnostic", {
                                            "message": f"Error in callback for event {event_name}: {str(e)}",
                                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                                        }, priority=1)
                        self.event_queue.task_done()

                    # Log queue size after processing batch
                    queue_size = self.event_queue.qsize()
                    self.logger.debug(f"Processed batch of {len(batch)} events, queue size now: {queue_size}")
                    if queue_size > self.max_queue_size * 0.8:
                        self.logger.warning(f"Event queue size approaching limit: {queue_size}/{self.max_queue_size}")
                        self.emit("event_diagnostic", {
                            "message": f"Event queue size approaching limit: {queue_size}/{self.max_queue_size}",
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                        }, priority=1)

                except Exception as e:
                    self.logger.error(f"Error processing event batch: {str(e)}")
                    self.emit("event_diagnostic", {
                        "message": f"Error processing event batch: {str(e)}",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }, priority=1)
        except Exception as e:
            self.logger.error(f"Event processing loop failed: {str(e)}")
            self.emit("event_diagnostic", {
                "message": f"Event processing loop failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            self.stop()

    def stop(self):
        """Stop the event processing loop."""
        try:
            self.running = False
            # Clear the queue
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                    self.event_queue.task_done()
                except queue.Empty:
                    break
            self.logger.info("EventHandler stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop EventHandler: {str(e)}")
            self.emit("event_diagnostic", {
                "message": f"Failed to stop EventHandler: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    handler = EventHandler()

    # Test callback for regular events
    def test_callback(data):
        print(f"Received event with data: {data}")

    # Test callback for diagnostic events
    def diagnostic_callback(data):
        print(f"Diagnostic event: {data}")

    # Subscribe to events
    handler.subscribe("test_event", test_callback)
    handler.subscribe("event_diagnostic", diagnostic_callback)

    # Start the event handler
    handler.start()

    # Test 1: Basic event emission
    print("Test 1: Emitting a single event")
    handler.emit("test_event", {"message": "Hello, Serenity!"}, priority=1)
    time.sleep(1)

    # Test 2: Priority-based queuing
    print("Test 2: Emitting events with different priorities")
    handler.emit("test_event", {"message": "Low priority"}, priority=3)
    handler.emit("test_event", {"message": "High priority"}, priority=1)
    time.sleep(1)

    # Test 3: Batching (emit multiple events to trigger batch processing)
    print("Test 3: Emitting multiple events to test batching")
    for i in range(15):  # More than batch_size (10)
        handler.emit("test_event", {"message": f"Batch event {i}"}, priority=2)
    time.sleep(2)

    # Test 4: Queue size monitoring (emit many events to approach max_queue_size)
    print("Test 4: Testing queue size monitoring")
    handler.batch_size = 1  # Slow down processing to build up the queue
    for i in range(900):  # Close to max_queue_size (1000)
        handler.emit("test_event", {"message": f"Queue test {i}"}, priority=2)
    time.sleep(2)

    # Test 5: Stop and queue clearing
    print("Test 5: Stopping and clearing queue")
    handler.stop()
    print(f"Queue size after stop: {handler.event_queue.qsize()}")