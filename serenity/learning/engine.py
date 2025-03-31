# File: serenity/learning/engine.py
# Created: 2024-12-24 15:36:21
# Purpose: Core Learning Engine
# Version: 1.0

"""
Serenity Core Learning Engine
Handles machine learning operations and model training
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional
import numpy as np

class LearningEngine:
    def __init__(self):
        self.logger = logging.getLogger('Serenity.Learning')
        self.model_path = Path('serenity/models')
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = None
        self.events = None
        self.model = None
        self.optimizer = None
        self.running = False

    def set_config(self, config):
        """Set configuration from ConfigManager."""
        self.config = config
        self.logger.info("Learning configuration set")

    def set_events(self, events):
        """Set event handler for communication."""
        self.events = events
        self.events.subscribe("data_processed", self.train_on_data)
        self.events.subscribe("system_shutdown", self.stop)
        self.logger.info("Learning event handler set")

    def initialize(self):
        """Initialize learning engine with a simple model."""
        try:
            if not self.config:
                raise ValueError("Configuration not set")

            self.logger.info("Initializing learning engine...")
            # Simple feedforward network as a starting point
            self.model = nn.Sequential(
                nn.Linear(768, 512),  # Match GPT-2 embedding size
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 768)  # Output embeddings
            ).to(self.device)

            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.get('ai_engine', 'learning_rate', 0.001)
            )
            self.criterion = nn.MSELoss()

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            self.logger.info("Learning engine initialized")
        except Exception as e:
            self.logger.error(f"Learning initialization failed: {str(e)}")
            raise

    def start(self):
        """Start the learning engine."""
        try:
            if not self.events:
                raise ValueError("Event handler not set")
            self.running = True
            self.initialize()
            self.logger.info("Learning engine started")
        except Exception as e:
            self.logger.error(f"Learning startup failed: {str(e)}")
            self.stop()

    def train_on_data(self, event_data: Dict):
        """Train model on processed data from DataProcessor."""
        try:
            if not self.running:
                self.logger.warning("Learning engine not running")
                return

            data = event_data.get('data')
            data_type = event_data.get('type')
            if not data or not data_type:
                self.logger.debug("Invalid training data received")
                return

            self.logger.info(f"Training on {data_type} data...")
            if data_type == 'text':
                embeddings = self._text_to_embedding(data['original'])
                self._train_step(embeddings)
            elif data_type == 'numeric':
                embeddings = self._numeric_to_embedding(data)
                self._train_step(embeddings)

            self.events.emit("model_updated", {"status": "trained"}, priority=4)
            self.save_model()

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")

    def _text_to_embedding(self, text: str) -> torch.Tensor:
        """Convert text to a simple embedding (placeholder)."""
        try:
            # For now, fake it with a random tensor—real impl would use tokenizer
            return torch.tensor(np.random.randn(1, 768), dtype=torch.float32).to(self.device)
        except Exception as e:
            self.logger.error(f"Text embedding failed: {str(e)}")
            return torch.zeros(1, 768).to(self.device)

    def _numeric_to_embedding(self, data: Dict) -> torch.Tensor:
        """Convert numeric data to embedding."""
        try:
            # Flatten stats into a vector, pad to 768
            values = [data[k] for k in ['mean', 'std', 'min', 'max']]
            padded = np.pad(values, (0, 768 - len(values)), mode='constant')
            return torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(self.device)
        except Exception as e:
            self.logger.error(f"Numeric embedding failed: {str(e)}")
            return torch.zeros(1, 768).to(self.device)

    def _train_step(self, embeddings: torch.Tensor):
        """Perform one training step."""
        try:
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(embeddings)
            loss = self.criterion(output, embeddings)  # Autoencoder-style
            loss.backward()
            self.optimizer.step()
            self.logger.debug(f"Training step complete, loss: {loss.item()}")
        except Exception as e:
            self.logger.error(f"Training step failed: {str(e)}")

    def save_model(self):
        """Save the trained model."""
        try:
            model_file = self.model_path / 'learning_model.pt'
            torch.save(self.model.state_dict(), model_file)
            self.logger.info(f"Model saved to {model_file}")
        except Exception as e:
            self.logger.error(f"Model save failed: {str(e)}")

    def stop(self, event_data=None):
        """Stop the learning engine and clean up."""
        try:
            self.running = False
            if self.model and self.device.type == 'cuda':
                del self.model
                torch.cuda.empty_cache()
            self.save_model()
            self.logger.info("Learning engine stopped")
        except Exception as e:
            self.logger.error(f"Learning shutdown failed: {str(e)}")

if __name__ == "__main__":
    from serenity.utils.config_manager import ConfigManager
    from serenity.utils.event_handler import EventHandler

    try:
        engine = LearningEngine()
        config = ConfigManager()
        events = EventHandler()
        engine.set_config(config)
        engine.set_events(events)
        engine.start()

        events.emit("data_processed", {
            "type": "text",
            "data": {"original": "Hello, Serenity!"}
        })
        time.sleep(2)  # Let it train
        engine.stop()

    except Exception as e:
        logging.getLogger('Serenity.Learning').error(f"Test failed: {str(e)}")