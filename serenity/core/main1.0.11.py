# File: serenity/core/main.py
# Created: 2024-12-24 15:32:47
# Updated: 2025-03-31 10:00:00
# Created by: Original author, refined by Grok 3, xAI with Mitch827
# Purpose: Main System Controller
# Version: 1.0.11

"""
Serenity Main System Controller
Coordinates all core systems with robust logging, threading, and Qt GUI integration.
Enhanced with debugging, error-handling tools, and diagnostic event emissions.
"""

import logging
import threading
import time
import sys
import os
import json
import traceback
from typing import Dict, Any
from PyQt5.QtWidgets import QApplication
from serenity.utils.config_manager import ConfigManager
from serenity.utils.event_handler import EventHandler
from serenity.utils.tools import SerenityTools  # Import real SerenityTools
from serenity.system_monitor import SystemMonitor
from serenity.utils.logger import SerenityLogger
from serenity.utils.data_processor import DataProcessor
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
        """Initialize controller with logger, config, events, and debug tracking."""
        self.logger = SerenityLogger()
        self.config = ConfigManager()
        self.events = EventHandler()
        self.tools = SerenityTools("Serenity.Core", events=self.events)  # Initialize with events
        self.tools.install_requirements()  # Install dependencies at startup
        self.tools.ensure_package("pywin32")  # Ensure pywin32 is available
        if not WIN32_AVAILABLE:
            self.logger.logger.warning("pywin32 not available - terminal control features disabled")
        self.app = None  # Qt app
        self.running = False
        self.modules: Dict[str, Any] = {}
        self.hwnd = None  # Terminal window handle
        self.debug_info: Dict[str, Any] = {
            "startup_time": time.time(),
            "system_state": "initializing",
            "errors": [],
            "module_status": {},
            "health_alerts": []  # Added to track health alerts
        }
        self.setup_logging()
        self.logger.logger.debug("Controller initialized")
        # Emit diagnostic event for controller initialization
        self.events.emit("system_diagnostic", {
            "message": "SerenityController initialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }, priority=1)

    def setup_logging(self):
        """Configure additional file logging for debugging."""
        log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "serenity_controller.log"))
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # Check for existing file handlers to avoid duplicates
        for handler in self.logger.logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == file_handler.baseFilename:
                return  # Handler already exists, skip adding
        self.logger.logger.addHandler(file_handler)
        self.logger.logger.debug("File logging configured for controller")

    def log_error(self, error: Exception, context: str):
        """Log an error with detailed context for debugging and emit diagnostic event."""
        error_info = {
            "timestamp": time.time(),
            "context": context,
            "error": str(error),
            "stack_trace": traceback.format_exc()
        }
        self.debug_info["errors"].append(error_info)
        self.logger.logger.error(f"Error in {context}: {str(error)}\n{traceback.format_exc()}")
        self.events.emit("system_diagnostic", {
            "message": f"Error in {context}: {str(error)}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }, priority=1)

    def save_debug_info(self):
        """Save debug information to a file for auto-debugging."""
        debug_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "logs", "controller_debug_info.json")
        try:
            with open(debug_file, "w") as f:
                json.dump(self.debug_info, f, indent=4)
            self.logger.logger.debug(f"Debug info saved to {debug_file}")
        except Exception as e:
            self.logger.logger.error(f"Failed to save debug info: {str(e)}")
            self.events.emit("system_diagnostic", {
                "message": f"Failed to save debug info: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def initialize_system(self):
        """Initialize all core systems with config and events."""
        try:
            self.logger.logger.debug("Setting logger config...")
            self.logger.set_config(self.config)
            self.logger.start()
            self.logger.logger.info("Starting Serenity system initialization...")

            self.logger.logger.debug("Loading configuration...")
            self.config.set_events(self.events)  # Enable config diagnostics
            self.config.load_config()
            self.logger.logger.info("Config loaded")
            self.debug_info["system_state"] = "config_loaded"

            self.logger.logger.debug("Starting event handler...")
            self.events.start()
            self.events.subscribe("system_shutdown", self.stop)
            self.events.subscribe("health_alert", self.handle_health_alert)
            self.logger.logger.info("Events started")
            self.debug_info["system_state"] = "events_started"

            self.logger.logger.debug("Initializing modules...")
            self.modules = {
                "system_monitor": SystemMonitor(),
                "data_processor": DataProcessor(),
                "ai_processor": AIProcessor(),
                "memory": MemoryManager(),
                "network": NetworkHandler(),
                "security": SecurityGuardian(),
                "voice": VoiceProcessor(),
                "gui": None  # Set after QApplication
            }

            for name, module in self.modules.items():
                if name != "gui" and module:
                    self.configure_module(name, module)
                    self.debug_info["module_status"][name] = "initialized"
            self.logger.logger.info("Core modules initialized")
            self.debug_info["system_state"] = "modules_initialized"
        except Exception as e:
            self.log_error(e, "initialize_system")
            self.debug_info["system_state"] = "failed"
            self.save_debug_info()
            raise

    def configure_module(self, name: str, module: Any):
        """Configure a single module with config and events."""
        try:
            if hasattr(module, "set_config"):
                print(f"SerenityController: {name} set_config called")
                module.set_config(self.config)
                self.logger.logger.debug(f"{name} config set")
            if hasattr(module, "set_events"):
                print(f"SerenityController: {name} set_events called")
                module.set_events(self.events)
                self.logger.logger.debug(f"{name} events set")
            self.logger.logger.info(f"{name} initialized")
            self.debug_info["module_status"][name] = "configured"
        except Exception as e:
            self.log_error(e, f"configure_module: {name}")
            self.debug_info["module_status"][name] = f"failed: {str(e)}"
            self.save_debug_info()
            raise

    def start(self):
        """Start all systems with GUI."""
        try:
            # Minimize terminal window if pywin32 is available
            if WIN32_AVAILABLE:
                self.hwnd = win32gui.GetForegroundWindow()
                win32gui.ShowWindow(self.hwnd, win32con.SW_MINIMIZE)
            else:
                self.logger.logger.warning("Cannot minimize terminal - pywin32 not installed")

            self.logger.logger.debug("Starting system...")
            self.running = True
            self.initialize_system()

            self.logger.logger.debug("Initializing Qt application...")
            self.app = QApplication(sys.argv)
            try:
                self.modules["gui"] = SerenityGUI()
                print("SerenityController: SerenityGUI created")
                self.configure_module("gui", self.modules["gui"])
                self.modules["gui"].show()  # Ensure GUI is visible
                print("SerenityGUI shown")
                self.debug_info["system_state"] = "gui_initialized"
            except Exception as e:
                self.log_error(e, "GUI initialization")
                self.debug_info["system_state"] = "gui_failed"
                self.save_debug_info()
                raise

            self.logger.logger.debug("Launching module threads (excluding GUI)...")
            for name, module in self.modules.items():
                if name != "gui" and module and hasattr(module, "start"):
                    self.logger.logger.info(f"Starting {name}...")
                    thread = threading.Thread(target=module.start, daemon=True)
                    thread.start()
                    print(f"SerenityController: {name} started")
                    self.debug_info["module_status"][name] = "started"

            self.logger.logger.debug("Starting GUI in main thread...")
            if self.modules["gui"] and hasattr(self.modules["gui"], "start"):
                self.modules["gui"].start()
                print("SerenityGUI started")
                self.debug_info["module_status"]["gui"] = "started"

            self.events.emit("system_started", {"status": "running"}, priority=1)
            self.logger.logger.info("System started")
            self.debug_info["system_state"] = "running"
            self.save_debug_info()
            # Emit diagnostic event for successful startup
            self.events.emit("system_diagnostic", {
                "message": "System startup completed successfully",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

            sys.exit(self.app.exec_())  # Enter Qt event loop and exit with its return code
        except Exception as e:
            self.log_error(e, "start")
            self.debug_info["system_state"] = "failed"
            self.save_debug_info()
            self.stop()

    def stop(self, event_data=None):
        """Gracefully stop all systems."""
        try:
            self.logger.logger.debug("Stopping system...")
            self.running = False
            for name, module in self.modules.items():
                if module and hasattr(module, "stop"):
                    self.logger.logger.info(f"Stopping {name}...")
                    try:
                        module.stop()
                        print(f"SerenityController: {name} stopped")
                        self.debug_info["module_status"][name] = "stopped"
                    except Exception as e:
                        self.log_error(e, f"stop: {name}")
                        self.debug_info["module_status"][name] = f"stop_failed: {str(e)}"
            self.events.emit("system_stopped", {"status": "stopped"}, priority=1)
            try:
                self.events.stop()
            except Exception as e:
                self.log_error(e, "stop: events")
            try:
                self.logger.stop()
            except Exception as e:
                self.log_error(e, "stop: logger")
            if WIN32_AVAILABLE and self.hwnd:
                win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)  # Hide terminal on shutdown
            self.logger.logger.info("System shutdown complete")
            self.debug_info["system_state"] = "stopped"
            self.save_debug_info()
        except Exception as e:
            self.log_error(e, "stop")
            self.debug_info["system_state"] = "shutdown_failed"
            self.save_debug_info()

    def handle_health_alert(self, event_data):
        """Handle system health alerts and emit diagnostic event."""
        self.logger.logger.warning(f"Health alert: {event_data}")
        self.debug_info["health_alerts"].append(event_data)
        self.events.emit("system_diagnostic", {
            "message": f"Health alert: {event_data}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }, priority=1)

    def get_debug_info(self) -> Dict[str, Any]:
        """Return debug information for auto-debugging."""
        return self.debug_info

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    controller = SerenityController()

    # Subscribe to events to verify emissions
    def system_started_callback(data):
        print(f"System started event: {data}")

    def system_stopped_callback(data):
        print(f"System stopped event: {data}")

    def diagnostic_callback(data):
        print(f"Diagnostic event: {data}")

    controller.events.subscribe("system_started", system_started_callback)
    controller.events.subscribe("system_stopped", system_stopped_callback)
    controller.events.subscribe("system_diagnostic", diagnostic_callback)

    try:
        # Test 1: Start the system
        print("Test 1: Starting the system")
        controller.start()

    except KeyboardInterrupt:
        print("Test 2: Stopping the system via KeyboardInterrupt")
        controller.stop()

    except Exception as e:
        logging.getLogger('Serenity.Core').error(f"Main execution failed: {str(e)}\n{traceback.format_exc()}")
        controller.debug_info["system_state"] = "main_failed"
        controller.log_error(e, "main")
        controller.save_debug_info()

    # Test 3: Verify debug info
    print("Test 3: Checking debug info")
    debug_info = controller.get_debug_info()
    print(f"Debug info: {debug_info}")