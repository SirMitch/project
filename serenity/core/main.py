# File: serenity/core/main.py
# Created: 2024-12-24 15:32:47
# Updated: 2025-03-31 18:00:00
# Created by: Original author, refined by Grok 3, xAI with Mitch827
# Purpose: Main System Controller
# Version: 1.2.0

"""
Serenity Main System Controller
Coordinates all core systems with robust logging, threading, and Qt GUI integration.
Enhanced with self-repairing mechanisms, comprehensive debugging, and diagnostic event emissions.
Runs from any directory with dynamic path resolution and dependency management.
"""

import sys
import os
import logging
import threading
import time
import json
import traceback
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import QApplication
import importlib

# Dynamically set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from serenity.utils.config_manager import ConfigManager
from serenity.utils.event_handler import EventHandler
from serenity.utils.tools import SerenityTools
from serenity.utils.logger import SerenityLogger
from serenity.utils.data_processor import DataProcessor
from serenity.system_monitor import SystemMonitor
from serenity.ai_engine.core import AIProcessor
from serenity.memory.manager import MemoryManager
from serenity.network.handler import NetworkHandler
from serenity.security.guardian import SecurityGuardian
from serenity.voice.processor import VoiceProcessor
from serenity.gui.interface import SerenityGUI

# Check for pywin32 availability
try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

class SerenityController:
    def __init__(self):
        """Initialize controller with robust logging, config, events, and self-repair tools."""
        self.project_root = PROJECT_ROOT
        self.logger = None
        self.tools = None
        self.config = None
        self.events = None
        self.app = None
        self.running = False
        self.modules: Dict[str, Any] = {}
        self.hwnd = None
        self.debug_info: Dict[str, Any] = {
            "startup_time": time.time(),
            "system_state": "initializing",
            "errors": [],
            "module_status": {},
            "health_alerts": [],
            "recovery_attempts": []
        }
        self.lock = threading.Lock()
        self._initialize_core_components()
        self.logger.logger.debug("Controller initialized")
        self.events.emit("system_diagnostic", {
            "message": "SerenityController initialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }, priority=1)

    def _initialize_core_components(self):
        """Initialize core components with retry logic and fallbacks."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.lock:
                    if not self.logger:
                        self.logger = SerenityLogger()
                        self.setup_logging()
                    if not self.tools:
                        self.tools = SerenityTools("Serenity.Core", events=None)
                        self.tools.install_requirements()
                        self.tools.ensure_package("pywin32")
                        self.tools.ensure_package("PyQt5")
                    if not self.events:
                        self.events = EventHandler()
                        self.events.start()
                        self.tools.set_events(self.events)
                    if not self.config:
                        self.config = ConfigManager()
                        self.config.set_events(self.events)
                        self.config.load_config()
                        self.logger.set_config(self.config)
                        self.logger.start()
                break
            except Exception as e:
                self._log_recovery_attempt("core_components", str(e), attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    self._log_fatal_error("Failed to initialize core components", e)
                    sys.exit(1)
                time.sleep(1)

    def setup_logging(self):
        """Configure file logging with error handling and directory creation."""
        try:
            log_dir = os.path.join(self.project_root, "logs")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "serenity_controller.log"))
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            for handler in self.logger.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler) and handler.baseFilename == file_handler.baseFilename:
                    self.logger.logger.handlers.remove(handler)
            self.logger.logger.addHandler(file_handler)
            self.logger.logger.debug("File logging configured")
        except Exception as e:
            self._log_error(e, "setup_logging")
            self.events.emit("system_diagnostic", {
                "message": f"Logging setup failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def _log_error(self, error: Exception, context: str):
        """Log an error with detailed context and emit diagnostic event."""
        error_info = {
            "timestamp": time.time(),
            "context": context,
            "error": str(error),
            "stack_trace": traceback.format_exc()
        }
        self.debug_info["errors"].append(error_info)
        if self.logger:
            self.logger.logger.error(f"Error in {context}: {str(error)}\n{traceback.format_exc()}")
        if self.events:
            self.events.emit("system_diagnostic", {
                "message": f"Error in {context}: {str(error)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def _log_recovery_attempt(self, context: str, error: str, attempt: int, max_retries: int):
        """Log a recovery attempt for diagnostics."""
        recovery_info = {
            "context": context,
            "error": error,
            "attempt": attempt,
            "max_retries": max_retries,
            "timestamp": time.time()
        }
        self.debug_info["recovery_attempts"].append(recovery_info)
        if self.logger:
            self.logger.logger.warning(f"Recovery attempt {attempt}/{max_retries} in {context}: {error}")
        if self.events:
            self.events.emit("system_diagnostic", {
                "message": f"Recovery attempt {attempt}/{max_retries} in {context}: {error}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def _log_fatal_error(self, message: str, error: Exception):
        """Log a fatal error and save debug info before exiting."""
        self._log_error(error, "fatal")
        self.debug_info["system_state"] = "fatal_error"
        self.save_debug_info()
        if self.logger:
            self.logger.logger.critical(f"{message}: {str(error)}")
        if self.events:
            self.events.emit("system_diagnostic", {
                "message": f"Fatal error: {message}: {str(error)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def save_debug_info(self):
        """Save debug information to a file with error handling."""
        debug_file = os.path.join(self.project_root, "logs", "controller_debug_info.json")
        try:
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            with open(debug_file, "w") as f:
                json.dump(self.debug_info, f, indent=4)
            self.logger.logger.debug(f"Debug info saved to {debug_file}")
        except Exception as e:
            self._log_error(e, "save_debug_info")

    def initialize_system(self):
        """Initialize all core systems with retry logic and fallbacks."""
        try:
            self.logger.logger.info("Starting Serenity system initialization...")
            self.events.subscribe("system_shutdown", self.stop)
            self.events.subscribe("health_alert", self.handle_health_alert)

            self.modules = {
                "system_monitor": SystemMonitor(),
                "data_processor": DataProcessor(),
                "ai_processor": AIProcessor(),
                "memory": MemoryManager(),
                "network": NetworkHandler(),
                "security": SecurityGuardian(),
                "voice": VoiceProcessor(),
                "gui": None  # Initialized later
            }

            for name, module in self.modules.items():
                if name != "gui" and module:
                    self._configure_module_with_retry(name, module)
            self.debug_info["system_state"] = "modules_initialized"
            self.logger.logger.info("Core modules initialized")
        except Exception as e:
            self._log_fatal_error("System initialization failed", e)
            raise

    def _configure_module_with_retry(self, name: str, module: Any, max_retries: int = 3):
        """Configure a module with retry logic."""
        for attempt in range(max_retries):
            try:
                self.configure_module(name, module)
                break
            except Exception as e:
                self._log_recovery_attempt(f"configure_module: {name}", str(e), attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    self.debug_info["module_status"][name] = f"failed: {str(e)}"
                    self.save_debug_info()
                time.sleep(1)

    def configure_module(self, name: str, module: Any):
        """Configure a single module with config and events."""
        if not module:
            self.logger.logger.warning(f"Module {name} is None, skipping configuration")
            return
        try:
            if hasattr(module, "set_config"):
                module.set_config(self.config)
                self.logger.logger.debug(f"{name} config set")
            if hasattr(module, "set_events"):
                module.set_events(self.events)
                self.logger.logger.debug(f"{name} events set")
            self.logger.logger.info(f"{name} configured")
            self.debug_info["module_status"][name] = "configured"
        except Exception as e:
            self._log_error(e, f"configure_module: {name}")
            raise

    def start(self):
        """Start all systems with GUI and self-repair mechanisms."""
        try:
            if sys.version_info < (3, 8):
                self.logger.logger.warning(f"Python {sys.version} detected; 3.8+ recommended")
            if WIN32_AVAILABLE:
                self.hwnd = win32gui.GetForegroundWindow()
                win32gui.ShowWindow(self.hwnd, win32con.SW_MINIMIZE)
            else:
                self.tools.ensure_package("pywin32")
                if "win32gui" in sys.modules:
                    importlib.reload(sys.modules["win32gui"])
                    self.hwnd = win32gui.GetForegroundWindow()
                    win32gui.ShowWindow(self.hwnd, win32con.SW_MINIMIZE)

            self.running = True
            self.initialize_system()

            self.app = QApplication(sys.argv)
            self.modules["gui"] = SerenityGUI()
            if self.modules["gui"]:
                self._configure_module_with_retry("gui", self.modules["gui"])
                self.modules["gui"].show()
                self.debug_info["system_state"] = "gui_initialized"
            else:
                self.logger.logger.error("GUI failed to load, continuing without GUI")

            for name, module in self.modules.items():
                if module and hasattr(module, "start"):
                    self._start_module_with_retry(name, module)

            self.events.emit("system_started", {"status": "running"}, priority=1)
            self.logger.logger.info("System fully started")
            self.debug_info["system_state"] = "running"
            self.save_debug_info()
            sys.exit(self.app.exec_())
        except Exception as e:
            self._log_fatal_error("System startup failed", e)
            self.stop()
            sys.exit(1)

    def _start_module_with_retry(self, name: str, module: Any, max_retries: int = 3):
        """Start a module with retry logic."""
        for attempt in range(max_retries):
            try:
                thread = threading.Thread(target=module.start, daemon=True)
                thread.start()
                self.debug_info["module_status"][name] = "started"
                self.logger.logger.info(f"{name} started")
                break
            except Exception as e:
                self._log_recovery_attempt(f"start_module: {name}", str(e), attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    self.debug_info["module_status"][name] = f"failed_to_start: {str(e)}"
                    self.save_debug_info()
                time.sleep(1)

    def handle_health_alert(self, event_data: Dict):
        """Handle health alerts from SystemMonitor."""
        try:
            self.debug_info["health_alerts"].append(event_data)
            self.logger.logger.warning(f"Health alert: {event_data.get('message')}")
            self.save_debug_info()
        except Exception as e:
            self._log_error(e, "handle_health_alert")

    def stop(self, event_data=None):
        """Stop all systems gracefully."""
        try:
            self.running = False
            for name, module in self.modules.items():
                if module and hasattr(module, "stop"):
                    try:
                        module.stop()
                        self.logger.logger.info(f"{name} stopped")
                        self.debug_info["module_status"][name] = "stopped"
                    except Exception as e:
                        self._log_error(e, f"stop_module: {name}")
            if self.events:
                self.events.stop()
            self.debug_info["system_state"] = "stopped"
            self.save_debug_info()
            self.logger.logger.info("Serenity system stopped")
        except Exception as e:
            self._log_fatal_error("System shutdown failed", e)

if __name__ == "__main__":
    controller = SerenityController()
    controller.start()