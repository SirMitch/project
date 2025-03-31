# File: serenity/utils/tools.py
# Created: 2025-03-25 07:25:00
# Updated: 2025-03-28 07:00:00
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: Centralized Tools for Serenity
# Version: 1.6.1

"""
Centralized Tools for Serenity
Singleton utility class for package management, logging, and error handling, designed for GUI integration.
Enhanced with robust debugging, error-handling tools, diagnostic event emissions, and package update checks.
"""

import logging
import sys
import subprocess
import importlib
import importlib.util
import pkg_resources
import traceback
import os
import time
from typing import Any, Callable, Optional, List, Dict
import threading
import glob
import json

class SerenityTools:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, module_name: str = "Serenity", events=None):
        """Singleton pattern for tools instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(module_name, events)
        return cls._instance

    def _initialize(self, module_name: str, events=None):
        """Initialize tools with logger, environment diagnostics, and debug tracking."""
        self.logger = logging.getLogger(module_name)
        self.events = events
        self.log_buffer = []  # To store logs for GUI display
        self.setup_logging()
        self.debug_info: Dict[str, Any] = {
            "startup_time": time.time(),
            "errors": [],
            "install_attempts": [],
            "module_loads": []
        }
        self.log_environment()
        self.logger.debug(f"Tools initialized for {module_name}")
        self.emit_diagnostic("INFO", f"Tools initialized for {module_name}")

    def setup_logging(self):
        """Configure logging with GUI-accessible handler and file logging for debugging."""
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            # Custom console handler to buffer logs
            class BufferingHandler(logging.StreamHandler):
                def __init__(self):
                    super().__init__()
                    self.buffer = []

                def emit(self, record):
                    self.buffer.append(record)
                    super().emit(record)

            console_handler = BufferingHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            # File handler for persistent debug logs
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            log_dir = os.path.join(project_root, "logs")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "serenity_tools.log"))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.handler = console_handler  # For get_logs compatibility
        self.logger.debug("Logging configured with console and file handlers")
        self.emit_diagnostic("INFO", "Logging configured with console and file handlers")

    def set_events(self, events):
        """Set the event handler for diagnostic emissions."""
        self.events = events
        self.logger.debug("Event handler set for SerenityTools")
        self.emit_diagnostic("INFO", "Event handler set for SerenityTools")

    def log_error(self, error: Exception, context: str):
        """Log an error with detailed context for debugging."""
        error_info = {
            "timestamp": time.time(),
            "context": context,
            "error": str(error),
            "stack_trace": traceback.format_exc()
        }
        self.debug_info["errors"].append(error_info)
        self.logger.error(f"Error in {context}: {str(error)}\n{traceback.format_exc()}")
        self.emit_diagnostic("ERROR", f"Error in {context}: {str(error)}")

    def save_debug_info(self):
        """Save debug information to a file for auto-debugging."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        debug_file = os.path.join(project_root, "logs", "debug_info.json")
        try:
            with open(debug_file, "w") as f:
                json.dump(self.debug_info, f, indent=4)
            self.logger.debug(f"Debug info saved to {debug_file}")
            self.emit_diagnostic("INFO", f"Debug info saved to {debug_file}")
        except Exception as e:
            self.logger.error(f"Failed to save debug info: {str(e)}")
            self.emit_diagnostic("ERROR", f"Failed to save debug info: {str(e)}")

    def install_requirements(self):
        """Install all dependencies listed in requirements.txt with detailed logging."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        requirements_file = os.path.join(project_root, "requirements.txt")
        
        if not os.path.exists(requirements_file):
            error_msg = f"requirements.txt not found at {requirements_file}"
            self.logger.warning(error_msg)
            self.debug_info["errors"].append({
                "timestamp": time.time(),
                "context": "install_requirements",
                "error": "requirements.txt not found"
            })
            self.emit_diagnostic("WARNING", error_msg)
            return

        self.logger.info("Checking and installing dependencies from requirements.txt...")
        self.emit_diagnostic("INFO", "Checking and installing dependencies from requirements.txt")
        with open(requirements_file, "r") as f:
            for line in f:
                package = line.strip()
                if package and not package.startswith("#"):  # Ignore empty lines and comments
                    package_name = package.split("==")[0].split(" ")[0]
                    try:
                        importlib.import_module(package_name)
                        self.logger.info(f"{package_name} already installed")
                        self.debug_info["install_attempts"].append({
                            "package": package_name,
                            "status": "already_installed",
                            "timestamp": time.time()
                        })
                        self.emit_diagnostic("INFO", f"{package_name} already installed")
                    except ImportError:
                        self.logger.warning(f"{package_name} not found, attempting install...")
                        self.emit_diagnostic("WARNING", f"{package_name} not found, attempting install")
                        try:
                            start_time = time.time()
                            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-cache-dir"])
                            duration = time.time() - start_time
                            self.logger.info(f"{package} installed successfully in {duration:.2f} seconds")
                            self.debug_info["install_attempts"].append({
                                "package": package,
                                "status": "installed",
                                "duration": duration,
                                "timestamp": time.time()
                            })
                            self.emit_diagnostic("INFO", f"{package} installed successfully in {duration:.2f} seconds")
                        except subprocess.CalledProcessError as e:
                            self.log_error(e, f"install_requirements: {package}")
                            self.debug_info["install_attempts"].append({
                                "package": package,
                                "status": "failed",
                                "error": str(e),
                                "timestamp": time.time()
                            })

    def ensure_package(self, package_name: str, import_name: Optional[str] = None) -> Any:
        """Ensure a package is installed and importable, with recursive file search and debugging."""
        if import_name is None:
            import_name = package_name
        self.logger.debug(f"Ensuring package {import_name} (install as {package_name})...")
        self.emit_diagnostic("DEBUG", f"Ensuring package {import_name} (install as {package_name})")
        self.logger.debug(f"sys.executable: {sys.executable}")
        self.logger.debug(f"sys.path: {sys.path}")

        try:
            module = importlib.import_module(import_name)
            version = pkg_resources.get_distribution(package_name).version
            self.logger.info(f"{import_name} already available, version {version}")
            self.debug_info["module_loads"].append({
                "module": import_name,
                "status": "loaded",
                "version": version,
                "timestamp": time.time()
            })
            self.emit_diagnostic("INFO", f"{import_name} already available, version {version}")
            return module
        except ImportError:
            self.logger.warning(f"{import_name} not found, attempting install...")
            self.emit_diagnostic("WARNING", f"{import_name} not found, attempting install")

        site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
        if site_packages not in sys.path:
            sys.path.append(site_packages)
            self.logger.debug(f"Added {site_packages} to sys.path")
            self.emit_diagnostic("DEBUG", f"Added {site_packages} to sys.path")

        try:
            self.logger.debug(f"Running pip install {package_name}...")
            self.emit_diagnostic("DEBUG", f"Running pip install {package_name}")
            start_time = time.time()
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--no-cache-dir"])
            duration = time.time() - start_time
            self.logger.info(f"{package_name} installed successfully in {duration:.2f} seconds")
            version = pkg_resources.get_distribution(package_name).version
            self.logger.debug(f"Verified {package_name} installed, version {version}")
            self.debug_info["install_attempts"].append({
                "package": package_name,
                "status": "installed",
                "duration": duration,
                "timestamp": time.time()
            })
            self.emit_diagnostic("INFO", f"{package_name} installed successfully in {duration:.2f} seconds")

            importlib.invalidate_caches()
            try:
                module = importlib.import_module(import_name)
                self.logger.info(f"{import_name} imported successfully after install")
                self.debug_info["module_loads"].append({
                    "module": import_name,
                    "status": "loaded_after_install",
                    "version": version,
                    "timestamp": time.time()
                })
                self.emit_diagnostic("INFO", f"{import_name} imported successfully after install")
                return module
            except ImportError:
                self.logger.warning(f"Standard import failed, trying PyQt5.QtChart...")
                self.emit_diagnostic("WARNING", "Standard import failed, trying PyQt5.QtChart")
                try:
                    module = importlib.import_module("PyQt5.QtChart")
                    self.logger.info(f"Using PyQt5.QtChart instead of {import_name}")
                    self.debug_info["module_loads"].append({
                        "module": "PyQt5.QtChart",
                        "status": "loaded_fallback",
                        "timestamp": time.time()
                    })
                    self.emit_diagnostic("INFO", f"Using PyQt5.QtChart instead of {import_name}")
                    return module
                except ImportError:
                    self.logger.warning(f"PyQt5.QtChart failed, searching files...")
                    self.emit_diagnostic("WARNING", "PyQt5.QtChart failed, searching files")

            module_files = []
            for root, _, files in os.walk(site_packages):
                for file in files:
                    if file.startswith(import_name) and file.endswith((".py", ".pyd")):
                        module_files.append(os.path.join(root, file))
            self.logger.debug(f"Found files for {import_name}: {module_files}")
            self.emit_diagnostic("DEBUG", f"Found files for {import_name}: {module_files}")
            if not module_files:
                error_msg = f"No module files found for {import_name} in {site_packages}"
                self.log_error(ImportError(error_msg), "ensure_package")
                raise ImportError(error_msg)

            for file_path in module_files:
                self.logger.debug(f"Attempting to load {import_name} from {file_path}")
                self.emit_diagnostic("DEBUG", f"Attempting to load {import_name} from {file_path}")
                spec = importlib.util.spec_from_file_location(import_name, file_path)
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[import_name] = module
                    spec.loader.exec_module(module)
                    self.logger.info(f"{import_name} loaded from {file_path}")
                    self.debug_info["module_loads"].append({
                        "module": import_name,
                        "status": "loaded_from_file",
                        "file_path": file_path,
                        "timestamp": time.time()
                    })
                    self.emit_diagnostic("INFO", f"{import_name} loaded from {file_path}")
                    return module
            raise ImportError(f"{import_name} not loaded from any file")
        except Exception as e:
            self.log_error(e, f"ensure_package: {package_name}")
            raise ImportError(f"Cannot install or import {import_name}: {str(e)}")

    def safe_import(self, module_path: str, class_name: str, fallback_class: Any = None) -> Any:
        """Safely import a class, falling back to a stub if missing, with debugging."""
        try:
            module = importlib.import_module(module_path)
            clazz = getattr(module, class_name)
            self.logger.debug(f"Imported {module_path}.{class_name}")
            self.debug_info["module_loads"].append({
                "module": f"{module_path}.{class_name}",
                "status": "imported",
                "timestamp": time.time()
            })
            self.emit_diagnostic("DEBUG", f"Imported {module_path}.{class_name}")
            return clazz
        except (ImportError, AttributeError) as e:
            self.log_error(e, f"safe_import: {module_path}.{class_name}")
            if fallback_class:
                return fallback_class
            stub = type(f"Stub{class_name}", (), {
                "__init__": lambda self: setattr(self, "logger", logging.getLogger(f"Stub.{class_name}")),
                "set_config": lambda self, config: self.logger.debug(f"Stub {class_name}.set_config called"),
                "set_events": lambda self, events: self.logger.debug(f"Stub {class_name}.set_events called"),
                "start": lambda self: self.logger.debug(f"Stub {class_name}.start called"),
                "stop": lambda self: self.logger.debug(f"Stub {class_name}.stop called")
            })
            self.logger.info(f"Created stub for {class_name}")
            self.debug_info["module_loads"].append({
                "module": f"Stub.{class_name}",
                "status": "stub_created",
                "timestamp": time.time()
            })
            self.emit_diagnostic("INFO", f"Created stub for {class_name}")
            return stub

    def check_for_updates(self, package_name: str) -> bool:
        """Check for updates to a package and install if available."""
        try:
            self.logger.info(f"Checking for updates to {package_name}...")
            self.emit_diagnostic("INFO", f"Checking for updates to {package_name}")
            result = subprocess.check_output([sys.executable, "-m", "pip", "list", "--outdated"])
            result = result.decode("utf-8")
            for line in result.splitlines():
                if package_name in line:
                    self.logger.info(f"Update available for {package_name}, installing...")
                    self.emit_diagnostic("INFO", f"Update available for {package_name}, installing")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
                    self.logger.info(f"Package {package_name} updated successfully")
                    self.emit_diagnostic("INFO", f"Package {package_name} updated successfully")
                    return True
            self.logger.debug(f"No updates available for {package_name}")
            self.emit_diagnostic("DEBUG", f"No updates available for {package_name}")
            return False
        except subprocess.CalledProcessError as e:
            self.log_error(e, f"check_for_updates: {package_name}")
            return False
        except Exception as e:
            self.log_error(e, f"check_for_updates: {package_name}")
            return False

    def log_environment(self):
        """Log system environment details for debugging."""
        self.logger.debug(f"Python version: {sys.version}")
        self.logger.debug(f"OS: {os.name} {sys.platform}")
        self.logger.debug(f"Working directory: {os.getcwd()}")
        self.logger.debug(f"PATH: {os.environ.get('PATH', 'Not set')}")
        self.debug_info["environment"] = {
            "python_version": sys.version,
            "os": f"{os.name} {sys.platform}",
            "working_directory": os.getcwd(),
            "path": os.environ.get('PATH', 'Not set')
        }
        self.emit_diagnostic("DEBUG", "Logged system environment details")

    def get_logs(self, level: str = "ALL") -> List[str]:
        """Retrieve logs for GUI display, filtered by level."""
        logs = [record.getMessage() for record in self.handler.buffer]
        if level != "ALL":
            logs = [log for log in logs if level.upper() in log]
        return logs

    def get_debug_info(self) -> Dict[str, Any]:
        """Return debug information for auto-debugging."""
        return self.debug_info

    def emit_diagnostic(self, level: str, message: str):
        """Emit a diagnostic event for debugging."""
        if self.events:
            self.events.emit("tools_diagnostic", {
                "message": message,
                "level": level,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

if __name__ == "__main__":
    from serenity.utils.event_handler import EventHandler

    # Set up event handler for testing
    events = EventHandler()
    events.start()

    # Subscribe to diagnostic events for testing
    def tools_diagnostic_callback(data):
        print(f"Tools Diagnostic: {data}")

    events.subscribe("tools_diagnostic", tools_diagnostic_callback)

    # Initialize tools with events
    tools = SerenityTools("Serenity.Test", events)

    # Test 1: Install requirements
    print("Test 1: Installing requirements")
    tools.install_requirements()

    # Test 2: Ensure package
    print("Test 2: Ensuring package 'requests'")
    tools.ensure_package("requests")

    # Test 3: Check for updates
    print("Test 3: Checking for updates to 'requests'")
    tools.check_for_updates("requests")

    # Test 4: Safe import (mocked, will use stub)
    print("Test 4: Safe importing non-existent module")
    mock_class = tools.safe_import("non_existent_module", "NonExistentClass")
    print(f"Imported class: {mock_class}")

    # Test 5: Get logs
    print("Test 5: Retrieving logs")
    logs = tools.get_logs("INFO")
    print(f"INFO Logs: {logs}")

    # Test 6: Log environment and save debug info
    print("Test 6: Logging environment and saving debug info")
    tools.log_environment()
    tools.save_debug_info()