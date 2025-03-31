# File: serenity/gui/sensor_suite.py
# Created: 2025-03-26 10:00:00
# Updated: 2025-03-31 14:00:00
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: System Monitor Widget for Serenity GUI with Enhanced Sensors and Diagnostics
# Version: 1.1.0

"""
SensorSuite Module
Handles the system monitor display for CPU, GPU, RAM, Swap, Disk, Network, and Temperature stats.
Includes robust error handling, self-diagnosis, and auto-fixing capabilities.
"""

import traceback
import time
from typing import Dict, Any
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class SensorSuite(QWidget):
    def __init__(self, logger, events):
        """Initialize the system monitor widget with diagnostic tools."""
        super().__init__()
        self.logger = logger
        self.events = events
        self.sensor_status = {}  # Track sensor health
        self.last_update = 0  # Timestamp of last successful update
        self.setup_ui()
        self.subscribe_to_events()
        self.initialize_sensors()
        self.logger.debug("SensorSuite initialized")

    def setup_ui(self):
        """Set up the system monitor layout with additional sensors."""
        try:
            self.logger.debug("Setting up SensorSuite layout...")
            monitor_layout = QGridLayout(self)
            
            # Initialize labels for all sensors
            self.cpu_label = QLabel("CPU: 0% | 0°C")
            self.gpu_label = QLabel("GPU: 0% | 0°C" if GPU_AVAILABLE else "GPU: N/A")
            self.ram_label = QLabel("RAM: 0%")
            self.swap_label = QLabel("Swap: 0%")
            self.disk_label = QLabel("Disk: 0%")
            self.network_label = QLabel("Net: 0 B/s | 0 B/s")
            self.temp_label = QLabel("Temp: N/A")  # System-wide temperature if available

            # Add widgets to layout
            monitor_layout.addWidget(self.cpu_label, 0, 0)
            monitor_layout.addWidget(self.gpu_label, 1, 0)
            monitor_layout.addWidget(self.ram_label, 2, 0)
            monitor_layout.addWidget(self.swap_label, 3, 0)
            monitor_layout.addWidget(self.disk_label, 4, 0)
            monitor_layout.addWidget(self.network_label, 5, 0)
            monitor_layout.addWidget(self.temp_label, 6, 0)
            
            self.logger.debug("SensorSuite layout setup complete")
        except Exception as e:
            self.logger.error(f"SensorSuite setup failed: {str(e)}\n{traceback.format_exc()}")
            self.emit_diagnostic("UI setup failure", str(e))

    def subscribe_to_events(self):
        """Subscribe to system health and diagnostic events."""
        try:
            self.logger.debug("Subscribing to events...")
            self.events.subscribe("system_health", self.update_system_monitor)
            self.events.subscribe("sensor_reset", self.reset_sensors)
            self.logger.debug("SensorSuite subscribed to events")
        except Exception as e:
            self.logger.error(f"Event subscription failed: {str(e)}\n{traceback.format_exc()}")
            self.emit_diagnostic("Event subscription failure", str(e))

    def initialize_sensors(self):
        """Initialize sensor status and perform initial checks."""
        self.sensor_status = {
            "cpu": {"active": False, "last_error": None},
            "gpu": {"active": GPU_AVAILABLE, "last_error": None},
            "ram": {"active": False, "last_error": None},
            "swap": {"active": False, "last_error": None},
            "disk": {"active": False, "last_error": None},
            "network": {"active": False, "last_error": None},
            "temperature": {"active": False, "last_error": None}
        }
        self.check_sensor_availability()
        self.logger.debug("Sensors initialized with status: %s", self.sensor_status)

    def check_sensor_availability(self):
        """Verify sensor availability and emit diagnostics."""
        try:
            # CPU
            psutil.cpu_percent()
            self.sensor_status["cpu"]["active"] = True
            
            # RAM
            psutil.virtual_memory()
            self.sensor_status["ram"]["active"] = True
            
            # Swap
            psutil.swap_memory()
            self.sensor_status["swap"]["active"] = True
            
            # Disk
            psutil.disk_usage('/')
            self.sensor_status["disk"]["active"] = True
            
            # Network
            psutil.net_io_counters()
            self.sensor_status["network"]["active"] = True
            
            # Temperature (optional, platform-dependent)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    self.sensor_status["temperature"]["active"] = True
            
            # GPU (if GPUtil is available)
            if GPU_AVAILABLE:
                try:
                    GPUtil.getGPUs()
                    self.sensor_status["gpu"]["active"] = True
                except Exception as e:
                    self.sensor_status["gpu"]["active"] = False
                    self.sensor_status["gpu"]["last_error"] = str(e)
                    self.emit_diagnostic("GPU sensor unavailable", str(e))
                    
            self.logger.debug("Sensor availability checked")
        except Exception as e:
            self.logger.error(f"Sensor availability check failed: {str(e)}\n{traceback.format_exc()}")
            self.emit_diagnostic("Sensor availability check failure", str(e))

    def update_system_monitor(self, event_data: Dict[str, Any]):
        """Update system monitor display with robust data handling."""
        try:
            current_time = time.time()
            if current_time - self.last_update < 1:  # Rate limit updates to 1 per second
                return
                
            self.logger.debug("Updating system monitor...")
            data = event_data.get("data", {})
            
            # CPU
            cpu = data.get("cpu_health", {})
            cpu_percent = self.validate_sensor_value(cpu.get("percent"), 0, 100, "cpu")
            cpu_temp = self.validate_sensor_value(cpu.get("temperature"), 0, 150, "cpu")
            self.cpu_label.setText(f"CPU: {cpu_percent:.1f}% | {cpu_temp:.1f}°C")
            self.sensor_status["cpu"]["active"] = True
            
            # GPU
            gpu = data.get("gpu_health", {})
            if GPU_AVAILABLE and self.sensor_status["gpu"]["active"]:
                gpu_usage = self.validate_sensor_value(gpu.get("usage"), 0, 100, "gpu")
                gpu_temp = self.validate_sensor_value(gpu.get("temperature"), 0, 150, "gpu")
                self.gpu_label.setText(f"GPU: {gpu_usage:.1f}% | {gpu_temp:.1f}°C")
            else:
                self.gpu_label.setText("GPU: N/A")
                
            # RAM
            ram = data.get("memory_health", {})
            ram_percent = self.validate_sensor_value(ram.get("percent"), 0, 100, "ram")
            self.ram_label.setText(f"RAM: {ram_percent:.1f}%")
            self.sensor_status["ram"]["active"] = True
            
            # Swap
            swap = data.get("swap_health", {})
            swap_percent = self.validate_sensor_value(swap.get("percent"), 0, 100, "swap")
            self.swap_label.setText(f"Swap: {swap_percent:.1f}%")
            self.sensor_status["swap"]["active"] = True
            
            # Disk
            disk = data.get("disk_health", {})
            disk_percent = self.validate_sensor_value(disk.get("percent"), 0, 100, "disk")
            self.disk_label.setText(f"Disk: {disk_percent:.1f}%")
            self.sensor_status["disk"]["active"] = True
            
            # Network
            network = data.get("network_health", {})
            net_sent = self.validate_sensor_value(network.get("sent_bytes"), 0, None, "network")
            net_recv = self.validate_sensor_value(network.get("recv_bytes"), 0, None, "network")
            self.network_label.setText(f"Net: {net_sent:.0f} B/s | {net_recv:.0f} B/s")
            self.sensor_status["network"]["active"] = True
            
            # Temperature (system-wide)
            temp = data.get("system_temp", {})
            sys_temp = self.validate_sensor_value(temp.get("temperature"), 0, 150, "temperature")
            if self.sensor_status["temperature"]["active"]:
                self.temp_label.setText(f"Temp: {sys_temp:.1f}°C")
            else:
                self.temp_label.setText("Temp: N/A")
                
            self.last_update = current_time
            self.logger.debug("System monitor updated")
            self.check_sensor_health()
        except Exception as e:
            self.logger.error(f"System monitor update failed: {str(e)}\n{traceback.format_exc()}")
            self.emit_diagnostic("Monitor update failure", str(e))
            self.attempt_sensor_recovery()

    def validate_sensor_value(self, value, min_val, max_val, sensor_name):
        """Validate sensor data and log anomalies."""
        try:
            if value is None:
                self.sensor_status[sensor_name]["last_error"] = "No data received"
                return 0
            value = float(value)
            if max_val is not None and (value < min_val or value > max_val):
                error_msg = f"{sensor_name} value {value} out of range ({min_val}-{max_val})"
                self.sensor_status[sensor_name]["last_error"] = error_msg
                self.emit_diagnostic(f"{sensor_name} anomaly", error_msg)
                return 0 if value < min_val else max_val
            self.sensor_status[sensor_name]["last_error"] = None
            return value
        except (ValueError, TypeError) as e:
            self.sensor_status[sensor_name]["last_error"] = f"Invalid {sensor_name} data: {str(e)}"
            self.emit_diagnostic(f"{sensor_name} validation failure", str(e))
            return 0

    def check_sensor_health(self):
        """Check sensor health and trigger diagnostics if needed."""
        for sensor, status in self.sensor_status.items():
            if not status["active"]:
                self.emit_diagnostic(f"{sensor} sensor offline", "Sensor not responding")
            elif status["last_error"]:
                self.emit_diagnostic(f"{sensor} sensor issue", status["last_error"])

    def attempt_sensor_recovery(self):
        """Attempt to recover failed sensors."""
        try:
            self.logger.debug("Attempting sensor recovery...")
            failed_sensors = [s for s, status in self.sensor_status.items() if not status["active"] or status["last_error"]]
            if not failed_sensors:
                return
                
            self.check_sensor_availability()
            for sensor in failed_sensors:
                if self.sensor_status[sensor]["active"]:
                    self.logger.info(f"{sensor} sensor recovered")
                    self.events.emit("sensor_recovered", {"sensor": sensor}, priority=1)
                else:
                    self.logger.warning(f"Failed to recover {sensor} sensor")
                    self.events.emit("sensor_failure", {"sensor": sensor, "action": "reset_needed"}, priority=1)
        except Exception as e:
            self.logger.error(f"Sensor recovery failed: {str(e)}\n{traceback.format_exc()}")
            self.emit_diagnostic("Sensor recovery failure", str(e))

    def reset_sensors(self, event_data: Dict):
        """Reset sensors based on event data."""
        try:
            sensor = event_data.get("sensor", "all")
            self.logger.debug(f"Resetting sensor(s): {sensor}")
            if sensor == "all":
                self.initialize_sensors()
            else:
                self.sensor_status[sensor]["active"] = False
                self.check_sensor_availability()
            self.logger.info(f"Sensor(s) {sensor} reset")
            self.events.emit("sensor_reset_complete", {"sensor": sensor}, priority=1)
        except Exception as e:
            self.logger.error(f"Sensor reset failed: {str(e)}\n{traceback.format_exc()}")
            self.emit_diagnostic("Sensor reset failure", str(e))

    def emit_diagnostic(self, issue: str, details: str):
        """Emit a diagnostic event for self-diagnosis."""
        self.events.emit("sensor_diagnostic", {
            "issue": issue,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }, priority=1)