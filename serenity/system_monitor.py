# File: serenity/system_monitor.py
# Created: 2025-03-25 02:46:00
# Updated: 2025-03-31 19:00:00
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: System Monitoring System
# Version: 1.1.0

"""
Serenity System Monitoring System
Monitors CPU, memory, swap, and GPU health with robust error handling and self-repair.
Emits diagnostic events and supports dynamic recovery from hardware access failures.
"""

import logging
import threading
import time
import traceback
from typing import Dict, Optional
import psutil
import platform
import json
import os

try:
    import wmi
except ImportError:
    wmi = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

class SystemMonitor:
    def __init__(self):
        """Initialize system monitor with robust logging and thread safety."""
        self.logger = logging.getLogger('Serenity.SystemMonitor')
        self.config = None
        self.events = None
        self.running = False
        self.lock = threading.Lock()
        self.wmi_conn = None
        self.temp_unavailable = False
        self.gpu_unavailable = False
        self.debug_info: Dict[str, Any] = {
            "startup_time": time.time(),
            "state": "initializing",
            "errors": [],
            "recovery_attempts": [],
            "health_checks": 0
        }
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self._initialize_hardware_access()

    def _initialize_hardware_access(self):
        """Initialize hardware access with fallbacks."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if wmi and not self.wmi_conn and platform.system() == "Windows":
                    self.wmi_conn = wmi.WMI()
                    self.logger.debug("WMI connection established")
                break
            except Exception as e:
                self._log_recovery_attempt("wmi_init", str(e), attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    self.logger.warning("WMI initialization failed permanently")
                    self.temp_unavailable = True
                time.sleep(1)

    def set_config(self, config):
        """Set configuration with error handling."""
        try:
            self.config = config
            self.logger.info("System monitor configuration set")
            self._log_diagnostic("Configuration set successfully")
        except Exception as e:
            self._log_error(e, "set_config")
            raise

    def set_events(self, events):
        """Set event handler with subscriptions."""
        try:
            self.events = events
            self.events.subscribe("request_system_health", self._send_health)
            self.events.subscribe("system_shutdown", self.stop)
            self.logger.info("System monitor event handler set")
            self._log_diagnostic("Event handler set with subscriptions")
        except Exception as e:
            self._log_error(e, "set_events")
            raise

    def start(self):
        """Start the system monitor with self-repair."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.config or not self.events:
                    raise ValueError("Configuration or event handler not set")
                with self.lock:
                    self.running = True
                threading.Thread(target=self._monitor_loop, daemon=True).start()
                self.logger.info("System monitor started")
                self.debug_info["state"] = "running"
                self._log_diagnostic("System monitor started")
                break
            except Exception as e:
                self._log_recovery_attempt("start", str(e), attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    self._log_error(e, "start_failed_permanently")
                    self.stop()
                time.sleep(1)

    def _monitor_loop(self):
        """Main monitoring loop with recovery mechanisms."""
        while self.running:
            try:
                interval = self.config.get("system_monitor", "update_interval", 5)
                cpu_temp_threshold = self.config.get("system_monitor", "cpu_temp_threshold", 85)
                gpu_temp_threshold = self.config.get("system_monitor", "gpu_temp_threshold", 90)

                health = self.get_system_health()
                self.debug_info["health_checks"] += 1
                self.events.emit("system_health", {"data": health}, priority=4)

                cpu_temp = health.get("cpu_health", {}).get("temperature", 0)
                gpu_temp = health.get("gpu_health", {}).get("temperature", 0)
                if cpu_temp > cpu_temp_threshold:
                    self.events.emit("health_alert", {
                        "message": f"CPU temp {cpu_temp}°C exceeds threshold {cpu_temp_threshold}°C",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }, priority=1)
                if gpu_temp > gpu_temp_threshold:
                    self.events.emit("health_alert", {
                        "message": f"GPU temp {gpu_temp}°C exceeds threshold {gpu_temp_threshold}°C",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }, priority=1)

                self._save_debug_info()
                time.sleep(interval)
            except Exception as e:
                self._log_error(e, "monitor_loop")
                self._attempt_recovery("monitor_loop")
                time.sleep(5)

    def get_system_health(self) -> Dict[str, Dict]:
        """Get comprehensive system health metrics."""
        try:
            with self.lock:
                health = {
                    "cpu_health": self._get_cpu_health(),
                    "memory_health": self._get_memory_health(),
                    "swap_health": self._get_swap_health(),
                    "gpu_health": self._get_gpu_health()
                }
                return health
        except Exception as e:
            self._log_error(e, "get_system_health")
            return {
                "cpu_health": {"percent": 0, "temperature": 0},
                "memory_health": {"percent": 0},
                "swap_health": {"percent": 0},
                "gpu_health": {"usage": 0, "temperature": 0, "memory_percent": 0}
            }

    def _get_cpu_health(self) -> Dict:
        """Get CPU usage and temperature with fallbacks."""
        try:
            percent = psutil.cpu_percent(interval=1)
            temp = 0
            if not self.temp_unavailable and self.wmi_conn and platform.system() == "Windows":
                try:
                    for cpu in self.wmi_conn.Win32_TemperatureProbe():
                        temp = cpu.CurrentReading / 10.0 if cpu.CurrentReading else 0
                        break
                    if not temp:
                        temps = psutil.sensors_temperatures().get("coretemp", [])
                        temp = temps[0].current if temps else 0
                except Exception as e:
                    self.logger.warning(f"CPU temp unavailable: {str(e)}")
                    self.temp_unavailable = True
            return {"percent": percent, "temperature": temp}
        except Exception as e:
            self._log_error(e, "get_cpu_health")
            return {"percent": 0, "temperature": 0}

    def _get_memory_health(self) -> Dict:
        """Get memory usage."""
        try:
            mem = psutil.virtual_memory()
            return {"percent": mem.percent}
        except Exception as e:
            self._log_error(e, "get_memory_health")
            return {"percent": 0}

    def _get_swap_health(self) -> Dict:
        """Get swap usage."""
        try:
            swap = psutil.swap_memory()
            return {"percent": swap.percent}
        except Exception as e:
            self._log_error(e, "get_swap_health")
            return {"percent": 0}

    def _get_gpu_health(self) -> Dict:
        """Get GPU metrics with recovery."""
        try:
            if self.gpu_unavailable or not GPUtil:
                return {"usage": 0, "temperature": 0, "memory_percent": 0}
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "usage": gpu.load * 100,
                    "temperature": gpu.temperature,
                    "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100
                }
            return {"usage": 0, "temperature": 0, "memory_percent": 0}
        except Exception as e:
            self._log_error(e, "get_gpu_health")
            self.gpu_unavailable = True
            return {"usage": 0, "temperature": 0, "memory_percent": 0}

    def _send_health(self, event_data: Dict):
        """Respond to health requests."""
        try:
            health = self.get_system_health()
            self.events.emit("system_health", {"data": health}, priority=4)
            self._log_diagnostic("Sent system health on request")
        except Exception as e:
            self._log_error(e, "send_health")

    def stop(self, event_data=None):
        """Stop the system monitor gracefully."""
        try:
            with self.lock:
                self.running = False
            self.debug_info["state"] = "stopped"
            self._save_debug_info()
            self.logger.info("System monitor stopped")
            self._log_diagnostic("System monitor stopped")
        except Exception as e:
            self._log_error(e, "stop")

    def _log_error(self, error: Exception, context: str):
        """Log an error with detailed context."""
        error_info = {
            "timestamp": time.time(),
            "context": context,
            "error": str(error),
            "stack_trace": traceback.format_exc()
        }
        self.debug_info["errors"].append(error_info)
        self.logger.error(f"Error in {context}: {str(error)}\n{traceback.format_exc()}")
        if self.events:
            self.events.emit("system_diagnostic", {
                "message": f"SystemMonitor error in {context}: {str(error)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def _log_recovery_attempt(self, context: str, error: str, attempt: int, max_retries: int):
        """Log a recovery attempt."""
        recovery_info = {
            "context": context,
            "error": error,
            "attempt": attempt,
            "max_retries": max_retries,
            "timestamp": time.time()
        }
        self.debug_info["recovery_attempts"].append(recovery_info)
        self.logger.warning(f"Recovery attempt {attempt}/{max_retries} in {context}: {error}")
        if self.events:
            self.events.emit("system_diagnostic", {
                "message": f"Recovery attempt {attempt}/{max_retries} in {context}: {error}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def _log_diagnostic(self, message: str):
        """Emit a diagnostic event."""
        if self.events:
            self.events.emit("system_diagnostic", {
                "message": f"SystemMonitor: {message}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
        self.logger.debug(message)

    def _attempt_recovery(self, context: str):
        """Attempt to recover from a failure."""
        try:
            if context == "monitor_loop":
                self.logger.info("Attempting recovery of monitor loop...")
                self._initialize_hardware_access()
                self.temp_unavailable = False
                self.gpu_unavailable = False
                self._log_diagnostic("Recovery attempted for monitor loop")
        except Exception as e:
            self._log_error(e, f"recovery_{context}")

    def _save_debug_info(self):
        """Save debug information to a file."""
        debug_file = os.path.join(self.project_root, "logs", "system_monitor_debug_info.json")
        try:
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            with open(debug_file, "w") as f:
                json.dump(self.debug_info, f, indent=4)
            self.logger.debug(f"Debug info saved to {debug_file}")
        except Exception as e:
            self._log_error(e, "save_debug_info")

if __name__ == "__main__":
    from serenity.utils.config_manager import ConfigManager
    from serenity.utils.event_handler import EventHandler

    logging.basicConfig(level=logging.DEBUG)
    monitor = SystemMonitor()
    config = ConfigManager()
    events = EventHandler()
    events.start()
    monitor.set_config(config)
    monitor.set_events(events)
    monitor.start()
    time.sleep(10)  # Run for 10 seconds
    monitor.stop()