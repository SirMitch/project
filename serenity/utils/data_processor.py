# File: serenity/utils/data_processor.py
# Created: 2025-03-25 15:47:21
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: Data Processing System
# Version: 1.0

"""
Serenity Data Processing System
Processes incoming data for AI consumption
"""

import logging
from typing import Dict, Any, Optional
import threading
from datetime import datetime

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger('Serenity.DataProcessor')
        self.config = None
        self.events = None
        self.running = False
        self.lock = threading.Lock()

    def set_config(self, config):
        """Set configuration from ConfigManager."""
        self.config = config
        self.logger.info("Data processor configuration set")

    def set_events(self, events):
        """Set event handler for communication."""
        self.events = events
        self.events.subscribe("input_received", self.process_input)
        self.events.subscribe("system_shutdown", self.stop)
        self.logger.info("Data processor event handler set")

    def start(self):
        """Start the data processor."""
        try:
            if not self.config or not self.events:
                raise ValueError("Configuration or event handler not set")
            self.running = True
            self.logger.info("Data processor started")
        except Exception as e:
            self.logger.error(f"Data processor startup failed: {str(e)}")
            self.stop()

    def process_input(self, event_data: Dict[str, Any]):
        """Process incoming input data."""
        try:
            if not self.running:
                self.logger.warning("Data processor not running")
                return

            text = event_data.get('text', '')
            timestamp = event_data.get('timestamp', datetime.now().isoformat())

            if not text:
                self.logger.debug("Empty input received")
                return

            with self.lock:
                processed_data = self._process_text(text)
                self.events.emit("data_processed", {
                    "type": "text",
                    "data": processed_data,
                    "timestamp": timestamp
                }, priority=3)
                self.logger.debug(f"Processed input: {text}")

        except Exception as e:
            self.logger.error(f"Input processing failed: {str(e)}")

    def _process_text(self, text: str) -> Dict[str, Any]:
        """Process text into a structured format."""
        try:
            words = text.split()
            return {
                "original": text,
                "word_count": len(words),
                "char_count": len(text),
                "sentiment": self._analyze_sentiment(text)
            }
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            return {"original": text, "word_count": 0, "char_count": 0, "sentiment": 0.0}

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (placeholder)."""
        try:
            positive_words = {"good", "great", "happy", "awesome"}
            negative_words = {"bad", "terrible", "sad", "fail"}
            words = set(text.lower().split())
            score = sum(1 for w in words if w in positive_words) - sum(1 for w in words if w in negative_words)
            return max(min(score / 5.0, 1.0), -1.0)  # Normalized [-1, 1]
        except Exception:
            return 0.0

    def stop(self, event_data=None):
        """Stop the data processor."""
        try:
            self.running = False
            self.logger.info("Data processor stopped")
        except Exception as e:
            self.logger.error(f"Data processor shutdown failed: {str(e)}")

if __name__ == "__main__":
    from serenity.utils.event_handler import EventHandler
    from serenity.utils.config_manager import ConfigManager

    logging.basicConfig(level=logging.INFO)
    processor = DataProcessor()
    config = ConfigManager()
    events = EventHandler()
    processor.set_config(config)
    processor.set_events(events)
    processor.start()

    events.emit("input_received", {"text": "Hello, great day!", "timestamp": datetime.now().isoformat()})
    import time
    time.sleep(1)
    processor.stop()