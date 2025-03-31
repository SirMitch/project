# File: serenity/utils/logger.py
# Created: 2025-03-25 06:58:00
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: Logging System
# Version: 1.0

"""
Serenity Logging System
Centralized logging for all system components
"""

import logging
import threading
from typing import Optional
import importlib
import subprocess
import sys

# Auto-install required packages (none yet, but ready for future)
def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing required package: {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} installed successfully!")

class SerenityLogger:
    def __init__(self):
        self.logger = logging.getLogger('Serenity')
        self.logger.setLevel(logging.DEBUG)
        self.handler = logging.StreamHandler()
        self.handler.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.config = None
        self.running = False

    def set_config(self, config):
        """Set configuration from ConfigManager."""
        self.config = config
        log_level = self.config.get("logging", "level", "INFO")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.logger.info("Logger configuration set")

    def start(self):
        """Start the logger."""
        self.running = True
        self.logger.info("Logger started")

    def stop(self):
        """Stop the logger."""
        self.running = False
        self.logger.info("Logger stopped")
        logging.shutdown()

if __name__ == "__main__":
    logger = SerenityLogger()
    logger.start()
    logger.logger.info("Test info message")
    logger.logger.error("Test error message")
    logger.stop()