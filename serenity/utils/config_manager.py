# File: serenity/utils/config_manager.py
# Created: 2025-03-25 15:47:20
# Updated: 2025-03-28 02:00:00
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: Configuration Management System
# Version: 1.0.1

"""
Serenity Configuration Management System
Handles system-wide and user-specific configuration settings with diagnostic tools.
Enhanced with logging, event emissions, and user preferences support.
"""

import json
import os
import logging
from typing import Any, Dict
import time
import threading

# Define SerenityTools class (normally in serenity/utils/tools.py)
class SerenityTools:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.debug("SerenityTools initialized")

    def install_requirements(self):
        """Mock installation of dependencies for testing."""
        self.logger.debug("Installing requirements (mocked for testing)")

    def ensure_package(self, package: str):
        """Mock ensuring a package is installed."""
        self.logger.debug(f"Ensuring package {package} is installed (mocked for testing)")

    def safe_import(self, module_path: str, class_name: str):
        """Mock safe import with a fallback."""
        self.logger.debug(f"Safe importing {class_name} from {module_path}")
        class MockClass:
            def __init__(self):
                self.logger = logging.getLogger(f"Mock.{class_name}")
        return MockClass

# Initialize tools
tools = SerenityTools("Serenity.Config")
tools.install_requirements()
tools.ensure_package("json")

class ConfigManager:
    def __init__(self):
        self.logger = logging.getLogger("Serenity.Config")
        self.config: Dict[str, Any] = {}
        self.user_config: Dict[str, Any] = {}
        self.config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
        self.user_config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'user_config.json')
        self.events = None
        self.running = False
        self.config_lock = threading.Lock()
        self.last_modified = 0

    def set_events(self, events):
        """Set event handler for communication."""
        self.events = events
        self.logger.info("ConfigManager event handler set")
        self.start_watching()

    def load_config(self):
        """Load system configuration from file or set defaults."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                self.last_modified = os.path.getmtime(self.config_file)
            else:
                self.config = {
                    "logging": {"level": "INFO"},
                    "system_monitor": {
                        "update_interval": 5,
                        "cpu_temp_threshold": 85,
                        "gpu_temp_threshold": 90
                    },
                    "voice": {
                        "speech_rate": 150,
                        "volume": 1.0,
                        "enable_input": True,
                        "listen_timeout": 5
                    },
                    "ai": {
                        "model_path": "models/serenity_ai",
                        "max_tokens": 512
                    },
                    "gui": {
                        "theme": "dark",
                        "font_size": 12
                    }
                }
                self.save_config()
            self.validate_config()
            self.logger.info("System configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Config load failed: {str(e)}")
            self.config = {
                "logging": {"level": "INFO"},
                "system_monitor": {
                    "update_interval": 5,
                    "cpu_temp_threshold": 85,
                    "gpu_temp_threshold": 90
                },
                "voice": {
                    "speech_rate": 150,
                    "volume": 1.0,
                    "enable_input": True,
                    "listen_timeout": 5
                },
                "ai": {
                    "model_path": "models/serenity_ai",
                    "max_tokens": 512
                },
                "gui": {
                    "theme": "dark",
                    "font_size": 12
                }
            }
            self.emit_diagnostic(f"Config load failed: {str(e)}")

    def load_user_config(self):
        """Load user-specific configuration from file or set defaults."""
        try:
            if os.path.exists(self.user_config_file):
                with open(self.user_config_file, 'r') as f:
                    self.user_config = json.load(f)
            else:
                self.user_config = {
                    "user_preferences": {
                        "username": "default_user",
                        "preferred_voice": None,
                        "chat_history_enabled": True
                    }
                }
                self.save_user_config()
            self.logger.info("User configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"User config load failed: {str(e)}")
            self.user_config = {
                "user_preferences": {
                    "username": "default_user",
                    "preferred_voice": None,
                    "chat_history_enabled": True
                }
            }
            self.emit_diagnostic(f"User config load failed: {str(e)}")

    def save_config(self):
        """Save system configuration to file."""
        try:
            with self.config_lock:
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=4)
                self.last_modified = os.path.getmtime(self.config_file)
            self.logger.info("System configuration saved successfully")
            if self.events:
                self.events.emit("config_updated", {"config": self.config}, priority=1)
        except Exception as e:
            self.logger.error(f"Config save failed: {str(e)}")
            self.emit_diagnostic(f"Config save failed: {str(e)}")

    def save_user_config(self):
        """Save user-specific configuration to file."""
        try:
            with self.config_lock:
                with open(self.user_config_file, 'w') as f:
                    json.dump(self.user_config, f, indent=4)
            self.logger.info("User configuration saved successfully")
            if self.events:
                self.events.emit("user_config_updated", {"user_config": self.user_config}, priority=1)
        except Exception as e:
            self.logger.error(f"User config save failed: {str(e)}")
            self.emit_diagnostic(f"User config save failed: {str(e)}")

    def validate_config(self):
        """Validate configuration values and correct if necessary."""
        try:
            # System Monitor
            if self.config.get("system_monitor", {}).get("update_interval", 5) < 1:
                self.config["system_monitor"]["update_interval"] = 5
                self.logger.warning("Corrected system_monitor.update_interval to 5")
            if self.config.get("system_monitor", {}).get("cpu_temp_threshold", 85) > 100:
                self.config["system_monitor"]["cpu_temp_threshold"] = 85
                self.logger.warning("Corrected system_monitor.cpu_temp_threshold to 85")
            if self.config.get("system_monitor", {}).get("gpu_temp_threshold", 90) > 100:
                self.config["system_monitor"]["gpu_temp_threshold"] = 90
                self.logger.warning("Corrected system_monitor.gpu_temp_threshold to 90")

            # Voice
            if self.config.get("voice", {}).get("speech_rate", 150) < 50:
                self.config["voice"]["speech_rate"] = 150
                self.logger.warning("Corrected voice.speech_rate to 150")
            if self.config.get("voice", {}).get("volume", 1.0) > 1.0 or self.config.get("voice", {}).get("volume", 1.0) < 0.0:
                self.config["voice"]["volume"] = 1.0
                self.logger.warning("Corrected voice.volume to 1.0")
            if self.config.get("voice", {}).get("listen_timeout", 5) < 1:
                self.config["voice"]["listen_timeout"] = 5
                self.logger.warning("Corrected voice.listen_timeout to 5")

            # AI
            if self.config.get("ai", {}).get("max_tokens", 512) < 128:
                self.config["ai"]["max_tokens"] = 512
                self.logger.warning("Corrected ai.max_tokens to 512")

            # GUI
            if self.config.get("gui", {}).get("font_size", 12) < 8:
                self.config["gui"]["font_size"] = 12
                self.logger.warning("Corrected gui.font_size to 12")

            self.logger.info("Configuration validated successfully")
        except Exception as e:
            self.logger.error(f"Config validation failed: {str(e)}")
            self.emit_diagnostic(f"Config validation failed: {str(e)}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value, preferring user config if available."""
        user_value = self.user_config.get(section, {}).get(key, None)
        if user_value is not None:
            return user_value
        return self.config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any, user_specific: bool = False):
        """Set a configuration value."""
        try:
            target_config = self.user_config if user_specific else self.config
            with self.config_lock:
                if section not in target_config:
                    target_config[section] = {}
                target_config[section][key] = value
            if user_specific:
                self.save_user_config()
            else:
                self.save_config()
            self.logger.info(f"Set {section}.{key} to {value} (user_specific={user_specific})")
        except Exception as e:
            self.logger.error(f"Failed to set {section}.{key}: {str(e)}")
            self.emit_diagnostic(f"Failed to set {section}.{key}: {str(e)}")

    def start_watching(self):
        """Start watching for configuration file changes."""
        if not self.events:
            self.logger.warning("Cannot start watching config - event handler not set")
            return
        self.running = True
        threading.Thread(target=self.watch_config, daemon=True).start()
        self.logger.info("Started watching configuration file for changes")

    def watch_config(self):
        """Watch for changes in the configuration file and emit updates."""
        while self.running:
            try:
                if os.path.exists(self.config_file):
                    mtime = os.path.getmtime(self.config_file)
                    if mtime > self.last_modified:
                        self.logger.info("Configuration file changed, reloading...")
                        self.load_config()
                        self.events.emit("config_updated", {"config": self.config}, priority=1)
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Config watch failed: {str(e)}")
                self.emit_diagnostic(f"Config watch failed: {str(e)}")
                time.sleep(5)

    def stop(self):
        """Stop watching for configuration changes."""
        self.running = False
        self.logger.info("Stopped watching configuration file")

    def emit_diagnostic(self, message: str):
        """Emit a diagnostic event for errors."""
        if self.events:
            self.events.emit("config_diagnostic", {
                "message": message,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

if __name__ == "__main__":
    from serenity.utils.event_handler import EventHandler
    from serenity.utils.logger import SerenityLogger

    logging.basicConfig(level=logging.INFO)
    logger = SerenityLogger().logger
    events = EventHandler()

    # Subscribe to diagnostic events for testing
    def config_diagnostic_callback(data):
        print(f"Config Diagnostic: {data}")

    def config_updated_callback(data):
        print(f"Config Updated: {data}")

    events.subscribe("config_diagnostic", config_diagnostic_callback)
    events.subscribe("config_updated", config_updated_callback)
    events.subscribe("user_config_updated", lambda data: print(f"User Config Updated: {data}"))

    config_manager = ConfigManager()
    config_manager.logger = logger
    config_manager.set_events(events)

    # Test 1: Load system config
    print("Test 1: Loading system config")
    config_manager.load_config()
    print(f"System Config: {config_manager.config}")

    # Test 2: Load user config
    print("Test 2: Loading user config")
    config_manager.load_user_config()
    print(f"User Config: {config_manager.user_config}")

    # Test 3: Get a value
    print("Test 3: Getting a value")
    speech_rate = config_manager.get("voice", "speech_rate")
    print(f"Voice speech_rate: {speech_rate}")

    # Test 4: Set a system config value
    print("Test 4: Setting a system config value")
    config_manager.set("voice", "speech_rate", 160)
    print(f"Updated voice.speech_rate: {config_manager.get('voice', 'speech_rate')}")

    # Test 5: Set a user-specific config value
    print("Test 5: Setting a user-specific config value")
    config_manager.set("user_preferences", "username", "test_user", user_specific=True)
    print(f"Updated user_preferences.username: {config_manager.get('user_preferences', 'username')}")

    # Test 6: Simulate config file change (manual modification required in real test)
    print("Test 6: Please modify config.json and wait for update (manual test)")
    time.sleep(5)

    # Test 7: Stop watching
    print("Test 7: Stopping config manager")
    config_manager.stop()