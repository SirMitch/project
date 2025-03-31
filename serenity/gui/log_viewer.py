# File: serenity/gui/log_viewer.py
# Created: 2025-03-26 10:00:00
# Updated: 2025-04-01 10:00:00
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: Log Viewer Widget for Serenity GUI
# Version: 1.0.1

"""
LogViewer Module
Handles the log tabs for Interactions, Errors, Debug logs, and Process Steps.
"""

import time
import traceback
from typing import Dict, Any
from PyQt5.QtWidgets import QTabWidget, QTextEdit, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

class LogViewer(QWidget):
    def __init__(self, logger, events):
        """Initialize the log viewer widget."""
        super().__init__()
        self.logger = logger
        self.events = events
        self.setup_ui()
        self.subscribe_to_events()
        self.logger.debug("LogViewer initialized")

    def setup_ui(self):
        """Set up the log tabs layout."""
        try:
            self.logger.debug("Setting up LogViewer layout...")
            self.log_tabs = QTabWidget()
            self.interaction_log = QTextEdit()
            self.interaction_log.setReadOnly(True)
            self.error_log = QTextEdit()
            self.error_log.setReadOnly(True)
            self.debug_log = QTextEdit()
            self.debug_log.setReadOnly(True)
            self.process_log = QTextEdit()
            self.process_log.setReadOnly(True)
            self.log_tabs.addTab(self.interaction_log, "Interactions")
            self.log_tabs.addTab(self.error_log, "Errors")
            self.log_tabs.addTab(self.debug_log, "Debug")
            self.log_tabs.addTab(self.process_log, "Process Steps")

            # Set layout for the widget
            layout = QVBoxLayout(self)
            layout.addWidget(self.log_tabs)
            self.logger.debug("LogViewer layout setup complete")
        except Exception as e:
            self.logger.error(f"LogViewer setup failed: {str(e)}\n{traceback.format_exc()}")

    def subscribe_to_events(self):
        """Subscribe to log entry and process step events."""
        try:
            self.logger.debug("Subscribing to events in LogViewer...")
            self.events.subscribe("log_entry", self.update_logs)
            self.events.subscribe("process_step", self.update_process_steps)
            self.logger.debug("LogViewer subscribed to log_entry and process_step events")
        except Exception as e:
            self.logger.error(f"LogViewer event subscription failed: {str(e)}\n{traceback.format_exc()}")

    def update_logs(self, event_data: Dict[str, Any]):
        """Update log tabs for real-time display."""
        try:
            self.logger.debug("Updating logs...")
            level = event_data.get("level", "INFO")
            message = event_data.get("message", "")
            module = event_data.get("module", "Unknown")
            timestamp = event_data.get("timestamp", time.strftime("%H:%M:%S"))
            log_text = f"[{timestamp}] [{module}] {message}"
            
            if level == "ERROR":
                self.error_log.append(log_text)
                self.error_log.verticalScrollBar().setValue(self.error_log.verticalScrollBar().maximum())
            elif level == "DEBUG":
                self.debug_log.append(log_text)
                self.debug_log.verticalScrollBar().setValue(self.debug_log.verticalScrollBar().maximum())
            else:
                self.interaction_log.append(log_text)
                self.interaction_log.verticalScrollBar().setValue(self.interaction_log.verticalScrollBar().maximum())
            self.logger.debug("Logs updated")
        except Exception as e:
            self.logger.error(f"Log update failed: {str(e)}\n{traceback.format_exc()}")

    def update_process_steps(self, event_data: Dict[str, Any]):
        """Update the Process Steps tab with process step events."""
        try:
            self.logger.debug("Updating process steps...")
            step = event_data.get("step", "")
            category = event_data.get("category", "Action")
            timestamp = event_data.get("timestamp", time.strftime("%H:%M:%S"))
            formatted_step = f"[{timestamp}] {category}: {step}"
            self.process_log.append(formatted_step)
            self.process_log.verticalScrollBar().setValue(self.process_log.verticalScrollBar().maximum())
            self.logger.debug("Process steps updated")
        except Exception as e:
            self.logger.error(f"Process steps update failed: {str(e)}\n{traceback.format_exc()}")

    def is_initialized(self):
        """Return whether the LogViewer is initialized."""
        return True

    def close(self):
        """Clean up the LogViewer by unsubscribing from events."""
        try:
            self.logger.debug("Closing LogViewer...")
            self.events.unsubscribe("log_entry", self.update_logs)
            self.events.unsubscribe("process_step", self.update_process_steps)
            self.logger.debug("LogViewer closed")
        except Exception as e:
            self.logger.error(f"LogViewer close failed: {str(e)}\n{traceback.format_exc()}")