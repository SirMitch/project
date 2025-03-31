# File: serenity/learning/pattern_recognition.py
# Created: 2024-12-24 15:37:46
# Purpose: Pattern Recognition System
# Version: 1.0

"""
Serenity Pattern Recognition System
Analyzes and identifies patterns in various data types
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
from scipy.fft import fft2
import threading

class PatternRecognizer:
    def __init__(self):
        self.logger = logging.getLogger('Serenity.Patterns')
        self.data_path = Path('serenity/data/patterns')
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = None
        self.events = None
        self.model = None
        self.running = False
        self.lock = threading.Lock()

    def set_config(self, config):
        """Set configuration from ConfigManager."""
        self.config = config
        self.logger.info("Pattern recognizer configuration set")

    def set_events(self, events):
        """Set event handler for communication."""
        self.events = events
        self.events.subscribe("data_processed", self.analyze_data)
        self.events.subscribe("system_shutdown", self.stop)
        self.logger.info("Pattern recognizer event handler set")

    def initialize(self):
        """Initialize pattern recognition model."""
        try:
            if not self.config:
                raise ValueError("Configuration not set")

            self.logger.info("Initializing pattern recognizer...")
            # Simple CNN for pattern detection
            self.model = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten(),
                nn.Linear(32 * 192, 128),  # Adjust based on input size
                nn.ReLU(),
                nn.Linear(128, 10)  # 10 pattern classes
            ).to(self.device)

            if self.device.type == 'cuda':
                self.model.half()
                torch.cuda.empty_cache()

            self.logger.info("Pattern recognizer initialized")
        except Exception as e:
            self.logger.error(f"Pattern initialization failed: {str(e)}")
            raise

    def start(self):
        """Start the pattern recognizer."""
        try:
            if not self.events:
                raise ValueError("Event handler not set")
            self.running = True
            self.initialize()
            self.logger.info("Pattern recognizer started")
        except Exception as e:
            self.logger.error(f"Pattern startup failed: {str(e)}")
            self.stop()

    def analyze_data(self, event_data: Dict):
        """Analyze incoming data for patterns."""
        try:
            if not self.running:
                self.logger.warning("Pattern recognizer not running")
                return

            data = event_data.get('data')
            data_type = event_data.get('type')
            if not data or not data_type:
                self.logger.debug("Invalid data for pattern analysis")
                return

            self.logger.info(f"Analyzing {data_type} data for patterns...")
            with self.lock:
                patterns = self.detect_patterns(data, data_type)
                if patterns:
                    self.events.emit("patterns_detected", {
                        "type": data_type,
                        "patterns": patterns,
                        "timestamp": event_data.get('timestamp')
                    }, priority=3)

        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {str(e)}")

    def detect_patterns(self, data: any, data_type: str) -> Optional[Dict]:
        """Detect patterns in data based on type."""
        try:
            processed_data = self.preprocess_data(data, data_type)
            if processed_data is None:
                return None

            self.model.eval()
            with torch.no_grad():
                inputs = torch.tensor(processed_data, dtype=torch.float32).unsqueeze(0).to(self.device)
                outputs = self.model(inputs)
                pattern_scores = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            patterns = {
                "scores": pattern_scores.tolist(),
                "dominant_pattern": int(np.argmax(pattern_scores)),
                "confidence": float(np.max(pattern_scores))
            }
            self.logger.debug(f"Patterns detected: {patterns}")
            return patterns

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}")
            return None

    def preprocess_data(self, data: any, data_type: str) -> Optional[np.ndarray]:
        """Preprocess data for pattern recognition."""
        try:
            if data_type == 'text':
                return self._text_to_vector(data['original'])
            elif data_type == 'numeric':
                return self._numeric_to_vector(data)
            elif data_type == 'time_series':
                return self._timeseries_to_vector(data)
            else:
                self.logger.warning(f"Unsupported data type for patterns: {data_type}")
                return None
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            return None

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to a vector for pattern analysis."""
        try:
            # Simple char frequency vector, padded to 768
            freq = [text.count(chr(i)) for i in range(32, 128)]  # ASCII printable
            padded = np.pad(freq, (0, 768 - len(freq)), mode='constant')
            return padded.reshape(1, -1)  # 1 channel for CNN
        except Exception:
            return np.zeros((1, 768))

    def _numeric_to_vector(self, data: Dict) -> np.ndarray:
        """Convert numeric data to a vector."""
        try:
            values = [data[k] for k in ['mean', 'std', 'min', 'max']]
            padded = np.pad(values, (0, 768 - len(values)), mode='constant')
            return padded.reshape(1, -1)
        except Exception:
            return np.zeros((1, 768))

    def _timeseries_to_vector(self, data: List[Dict]) -> np.ndarray:
        """Convert time series to a vector with FFT."""
        try:
            values = [d.get('value', 0) for d in data[:768]]  # Cap at 768
            padded = np.pad(values, (0, 768 - len(values)), mode='constant')
            fft_result = np.abs(fft2(padded.reshape(1, -1)))
            return fft_result
        except Exception:
            return np.zeros((1, 768))

    def stop(self, event_data=None):
        """Stop the pattern recognizer and clean up."""
        try:
            self.running = False
            if self.model and self.device.type == 'cuda':
                del self.model
                torch.cuda.empty_cache()
            self.logger.info("Pattern recognizer stopped")
        except Exception as e:
            self.logger.error(f"Pattern shutdown failed: {str(e)}")

if __name__ == "__main__":
    from serenity.utils.config_manager import ConfigManager
    from serenity.utils.event_handler import EventHandler

    try:
        recognizer = PatternRecognizer()
        config = ConfigManager()
        events = EventHandler()
        recognizer.set_config(config)
        recognizer.set_events(events)
        recognizer.start()

        events.emit("data_processed", {
            "type": "text",
            "data": {"original": "Hello, Serenity!"},
            "timestamp": "now"
        })
        time.sleep(2)  # Let it process
        recognizer.stop()

    except Exception as e:
        logging.getLogger('Serenity.Patterns').error(f"Test failed: {str(e)}")