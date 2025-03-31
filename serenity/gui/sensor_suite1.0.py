# File: serenity/gui/sensor_suite.py
# Created: 2025-03-26 10:00:00
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: System Monitor Widget for Serenity GUI
# Version: 1.0.0

"""
SensorSuite Module
Handles the system monitor display for CPU, GPU, RAM, and Swap stats.
"""

import traceback
from typing import Dict, Any
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel

class SensorSuite(QWidget):
    def __init__(self, logger, events):
        """Initialize the system monitor widget."""
        super().__init__()
        self.logger = logger
        self.events = events
        self.setup_ui()
        self.subscribe_to_events()
        self.logger.debug("SensorSuite initialized")

    def setup_ui(self):
        """Set up the system monitor layout."""
        try:
            self.logger.debug("Setting up SensorSuite layout...")
            monitor_layout = QGridLayout(self)
            self.cpu_label = QLabel("CPU: 0% | 0°C")
            self.gpu_label = QLabel("GPU: 0% | 0°C")
            self.ram_label = QLabel("RAM: 0%")
            self.swap_label = QLabel("Swap: 0%")
            monitor_layout.addWidget(self.cpu_label, 0, 0)
            monitor_layout.addWidget(self.gpu_label, 1, 0)
            monitor_layout.addWidget(self.ram_label, 2, 0)
            monitor_layout.addWidget(self.swap_label, 3, 0)
            self.logger.debug("SensorSuite layout setup complete")
        except Exception as e:
            self.logger.error(f"SensorSuite setup failed: {str(e)}\n{traceback.format_exc()}")

    def subscribe_to_events(self):
        """Subscribe to system health events."""
        try:
            self.events.subscribe("system_health", self.update_system_monitor)
            self.logger.debug("SensorSuite subscribed to system_health events")
        except Exception as e:
            self.logger.error(f"SensorSuite event subscription failed: {str(e)}\n{traceback.format_exc()}")

    def update_system_monitor(self, event_data: Dict[str, Any]):
        """Update system monitor display."""
        try:
            self.logger.debug("Updating system monitor...")
            data = event_data.get("data", {})
            cpu = data.get("cpu_health", {})
            gpu = data.get("gpu_health", {})
            ram = data.get("memory_health", {})
            swap = data.get("swap_health", {})
            self.cpu_label.setText(f"CPU: {cpu.get('percent', 0):.1f}% | {cpu.get('temperature', 0):.1f}°C")
            self.gpu_label.setText(f"GPU: {gpu.get('usage', 0):.1f}% | {gpu.get('temperature', 0):.1f}°C")
            self.ram_label.setText(f"RAM: {ram.get('percent', 0):.1f}%")
            self.swap_label.setText(f"Swap: {swap.get('percent', 0):.1f}%")
            self.logger.debug("System monitor updated")
        except Exception as e:
            self.logger.error(f"System monitor update failed: {str(e)}\n{traceback.format_exc()}")