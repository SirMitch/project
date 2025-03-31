# File: serenity/gui/chat_box.py
# Created: 2025-03-26 10:00:00
# Updated: 2025-03-29 10:00:00
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: Chat Box Widget for Serenity GUI (Chat Window Section)
# Version: 1.0.6

"""
ChatBox Module
Handles the chat display, input, status lights, control buttons, and a debugging tab.
Designed as a modular chat window section for integration into the main GUI (interface.py).
Enhanced with a debug log tab for diagnostic events and developer tools.
"""

import sys
import time
import traceback
from typing import Dict, Any
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QTabWidget, QComboBox
from PyQt5.QtCore import Qt, QTimer
from serenity.utils.tools import SerenityTools  # Import the real SerenityTools

# Check for pywin32 availability
try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

class ChatBox(QWidget):
    def __init__(self, logger, events):
        """Initialize the chat box widget with diagnostic and debugging capabilities."""
        super().__init__()
        self.logger = logger
        self.events = events
        self.tools = SerenityTools("Serenity.ChatBox", events=events)  # Initialize with events
        self.tools.install_requirements()  # Ensure dependencies are installed
        self.tools.ensure_package("pywin32")  # Ensure pywin32 is available
        self.ai_status = "red"  # red (off), yellow (loading), green (ready)
        self.network_status = "red"
        self.terminal_visible = False
        self.hwnd = None
        self.event_subscriptions = []
        self.debug_messages = []  # Store debug messages for filtering
        if not WIN32_AVAILABLE:
            self.logger.warning("pywin32 not available - terminal control features disabled")
        self.setup_ui()
        self.subscribe_to_events()
        self.logger.debug("ChatBox initialized")
        self.log_diagnostic_state("ChatBox initialized")
        # Emit a diagnostic event for SerenityTools initialization
        self.events.emit("chatbox_diagnostic", {
            "message": "SerenityTools initialized for ChatBox",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }, priority=1)

    def setup_ui(self):
        """Set up the chat box layout with a debugging tab."""
        try:
            self.logger.debug("Setting up ChatBox layout...")
            layout = QVBoxLayout(self)

            # Status lights
            status_layout = QHBoxLayout()
            self.ai_label = QLabel("AI")
            self.ai_light = QLabel("  ")
            self.ai_light.setFixedSize(20, 20)
            self.ai_light.setStyleSheet("background-color: red; border: 1px solid black;")
            self.network_label = QLabel("Network")
            self.network_light = QLabel("  ")
            self.network_light.setFixedSize(20, 20)
            self.network_light.setStyleSheet("background-color: red; border: 1px solid black;")
            status_layout.addWidget(self.ai_label)
            status_layout.addWidget(self.ai_light)
            status_layout.addWidget(self.network_label)
            status_layout.addWidget(self.network_light)
            status_layout.addStretch()
            layout.addLayout(status_layout)

            # Tab widget for chat and debug log
            self.tabs = QTabWidget()
            layout.addWidget(self.tabs)

            # Chat tab
            chat_tab = QWidget()
            chat_layout = QVBoxLayout(chat_tab)

            # Chat display
            self.chat_display = QTextEdit()
            self.chat_display.setReadOnly(True)
            chat_layout.addWidget(self.chat_display)

            # Chat input
            chat_input_widget = QWidget()
            chat_input_layout = QHBoxLayout(chat_input_widget)
            self.chat_input = QLineEdit()
            self.chat_input.returnPressed.connect(self.send_chat)
            self.send_button = QPushButton("Send")
            self.send_button.setEnabled(False)
            self.send_button.clicked.connect(self.send_chat)
            chat_input_layout.addWidget(self.chat_input)
            chat_input_layout.addWidget(self.send_button)
            chat_layout.addWidget(chat_input_widget)

            self.tabs.addTab(chat_tab, "Chat")

            # Debug log tab
            debug_tab = QWidget()
            debug_layout = QVBoxLayout(debug_tab)

            # Debug log filter
            filter_layout = QHBoxLayout()
            self.filter_label = QLabel("Filter:")
            self.filter_combo = QComboBox()
            self.filter_combo.addItems(["All", "Errors", "Info", "Debug"])
            self.filter_combo.currentTextChanged.connect(self.update_debug_log)
            filter_layout.addWidget(self.filter_label)
            filter_layout.addWidget(self.filter_combo)
            filter_layout.addStretch()
            debug_layout.addLayout(filter_layout)

            # Debug log display
            self.debug_log = QTextEdit()
            self.debug_log.setReadOnly(True)
            debug_layout.addWidget(self.debug_log)

            # Debug controls
            debug_controls = QHBoxLayout()
            self.diagnose_button = QPushButton("Run Diagnostics")
            self.diagnose_button.clicked.connect(self.run_diagnosis)
            self.clear_debug_button = QPushButton("Clear Debug Log")
            self.clear_debug_button.clicked.connect(self.clear_debug_log)
            self.install_deps_button = QPushButton("Install Dependencies")
            self.install_deps_button.clicked.connect(self.install_dependencies)
            debug_controls.addWidget(self.diagnose_button)
            debug_controls.addWidget(self.clear_debug_button)
            debug_controls.addWidget(self.install_deps_button)
            debug_layout.addLayout(debug_controls)

            self.tabs.addTab(debug_tab, "Debug Log")

            # Control buttons
            button_layout = QHBoxLayout()
            self.terminal_button = QPushButton("Show Terminal")
            if not WIN32_AVAILABLE:
                self.terminal_button.setEnabled(False)
                self.terminal_button.setText("Show Terminal (pywin32 required)")
            self.terminal_button.clicked.connect(self.toggle_terminal)
            self.clear_button = QPushButton("Clear Chat")
            self.clear_button.clicked.connect(self.clear_chat)
            self.shutdown_button = QPushButton("Shutdown")
            self.shutdown_button.clicked.connect(self.confirm_shutdown)
            button_layout.addWidget(self.terminal_button)
            button_layout.addWidget(self.clear_button)
            button_layout.addWidget(self.shutdown_button)
            layout.addLayout(button_layout)

            # Status bar
            self.status_bar = QLabel("Status: Initializing...")
            self.status_bar.setAlignment(Qt.AlignLeft)
            layout.addWidget(self.status_bar)

            self.logger.debug("ChatBox layout setup complete")
            self.log_diagnostic_state("ChatBox UI setup complete")
        except Exception as e:
            self.logger.error(f"ChatBox setup failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to set up chat interface. Please check logs.")

    def subscribe_to_events(self):
        """Subscribe to chat-related and diagnostic events."""
        try:
            self.logger.debug("Subscribing to chat-related events...")
            self.event_subscriptions = [
                ("ai_response", self.update_chat_response),
                ("system_started", self.on_system_started),
                ("network_status", self.update_network_status),
                ("ai_initialized", self.on_ai_initialized),
                ("system_shutdown", self.shutdown),
                ("voice_diagnostic", self.handle_voice_diagnostic),
                ("system_diagnostic", self.handle_system_diagnostic),
                ("config_diagnostic", self.handle_config_diagnostic),
                ("chatbox_diagnostic", self.handle_chatbox_diagnostic),
                ("chatbox_diagnostic_state", self.handle_chatbox_diagnostic_state)
            ]
            for event_name, callback in self.event_subscriptions:
                self.events.subscribe(event_name, callback)
                self.logger.debug(f"Subscribed to event: {event_name}")
            self.logger.debug("ChatBox subscribed to events")
            self.log_diagnostic_state("ChatBox event subscriptions complete")
        except Exception as e:
            self.logger.error(f"ChatBox event subscription failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to subscribe to events. Chat functionality may be limited.")

    def update_chat_response(self, event_data: Dict[str, Any]):
        """Update chat display with AI response."""
        try:
            self.logger.debug(f"Received ai_response event with data: {event_data}")
            response = event_data.get("response", "AI: No response received")
            QTimer.singleShot(0, lambda: self.chat_display.append(f"AI: {response}"))
            self.ai_status = "green"
            QTimer.singleShot(0, lambda: self.ai_light.setStyleSheet("background-color: green; border: 1px solid black;"))
            self.logger.debug("Chat response updated")
            self.log_diagnostic_state("Chat response updated")
        except Exception as e:
            self.logger.error(f"Chat update failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to update chat with AI response.")

    def send_chat(self):
        """Send chat input to event system with robust error handling."""
        try:
            self.logger.debug("Sending chat input...")
            text = self.chat_input.text().strip()
            if not text:
                self.logger.debug("Empty message, ignoring send request")
                return
            QTimer.singleShot(0, lambda: self.chat_display.append(f"You: {text}"))
            event_data = {
                "text": text,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            self.events.emit("input_received", event_data, priority=2)
            self.logger.debug(f"Emitted input_received event with data: {event_data}")
            QTimer.singleShot(0, self.chat_input.clear)
            self.logger.debug("Chat input sent")
            self.log_diagnostic_state("Chat input sent")
        except Exception as e:
            self.logger.error(f"Chat send failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to send message to AI. Please try again.")

    def on_system_started(self, event_data):
        """Handle system startup with status updates."""
        try:
            self.logger.debug(f"Received system_started event with data: {event_data}")
            QTimer.singleShot(0, lambda: self.chat_display.append("System: Starting Serenity..."))
            QTimer.singleShot(0, lambda: self.chat_display.append("System: Loading AI model..."))
            self.ai_status = "yellow"
            QTimer.singleShot(0, lambda: self.ai_light.setStyleSheet("background-color: yellow; border: 1px solid black;"))
            self.network_status = "green"
            QTimer.singleShot(0, lambda: self.network_light.setStyleSheet("background-color: green; border: 1px solid black;"))
            QTimer.singleShot(0, lambda: self.status_bar.setText("Status: System started, AI loading..."))
            self.logger.debug("System started, AI status set to loading")
            self.log_diagnostic_state("System started, AI loading")
        except Exception as e:
            self.logger.error(f"System started handling failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to handle system startup event.")

    def on_ai_initialized(self, event_data):
        """Handle AI initialization completion."""
        try:
            self.logger.debug(f"Received ai_initialized event with data: {event_data}")
            QTimer.singleShot(0, lambda: self.chat_display.append("System: AI model loaded successfully!"))
            self.ai_status = "green"
            QTimer.singleShot(0, lambda: self.ai_light.setStyleSheet("background-color: green; border: 1px solid black;"))
            QTimer.singleShot(0, lambda: self.send_button.setEnabled(True))
            QTimer.singleShot(0, lambda: self.status_bar.setText("Status: AI ready"))
            self.logger.debug("AI initialized, Send button enabled")
            self.log_diagnostic_state("AI initialized, chat ready")
        except Exception as e:
            self.logger.error(f"AI initialization handling failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to handle AI initialization event.")

    def update_network_status(self, event_data):
        """Update network status light."""
        try:
            self.logger.debug(f"Received network_status event with data: {event_data}")
            status = event_data.get("status", "disconnected")
            color = {"connected": "green", "connecting": "yellow", "disconnected": "red"}.get(status, "red")
            self.network_status = color
            QTimer.singleShot(0, lambda: self.network_light.setStyleSheet(f"background-color: {color}; border: 1px solid black;"))
            QTimer.singleShot(0, lambda: self.status_bar.setText(f"Status: Network {status}"))
            self.logger.debug(f"Network status updated to {status}")
            self.log_diagnostic_state("Network status updated")
        except Exception as e:
            self.logger.error(f"Network status update failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to update network status.")

    def handle_voice_diagnostic(self, event_data):
        """Handle voice diagnostic events."""
        try:
            self.logger.debug(f"Received voice_diagnostic event with data: {event_data}")
            message = event_data.get("message", "Voice: Unknown issue")
            issues = event_data.get("issues", [])
            if issues:
                message = "Voice: " + "; ".join(issues)
            QTimer.singleShot(0, lambda: self.chat_display.append(message))
            QTimer.singleShot(0, lambda: self.status_bar.setText(f"Status: {message}"))
            self.add_debug_message("ERROR", message, event_data.get("timestamp"))
            self.logger.debug("Voice diagnostic message displayed")
            self.log_diagnostic_state("Voice diagnostic handled")
        except Exception as e:
            self.logger.error(f"Voice diagnostic handling failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to handle voice diagnostic event.")

    def handle_system_diagnostic(self, event_data):
        """Handle system diagnostic events."""
        try:
            self.logger.debug(f"Received system_diagnostic event with data: {event_data}")
            message = event_data.get("message", "System: Unknown issue")
            QTimer.singleShot(0, lambda: self.chat_display.append(message))
            QTimer.singleShot(0, lambda: self.status_bar.setText(f"Status: {message}"))
            self.add_debug_message("INFO", message, event_data.get("timestamp"))
            self.logger.debug("System diagnostic message displayed")
            self.log_diagnostic_state("System diagnostic handled")
        except Exception as e:
            self.logger.error(f"System diagnostic handling failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to handle system diagnostic event.")

    def handle_config_diagnostic(self, event_data):
        """Handle config diagnostic events."""
        try:
            self.logger.debug(f"Received config_diagnostic event with data: {event_data}")
            message = event_data.get("message", "Config: Unknown issue")
            QTimer.singleShot(0, lambda: self.chat_display.append(message))
            QTimer.singleShot(0, lambda: self.status_bar.setText(f"Status: {message}"))
            self.add_debug_message("ERROR", message, event_data.get("timestamp"))
            self.logger.debug("Config diagnostic message displayed")
            self.log_diagnostic_state("Config diagnostic handled")
        except Exception as e:
            self.logger.error(f"Config diagnostic handling failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to handle config diagnostic event.")

    def handle_chatbox_diagnostic(self, event_data):
        """Handle chatbox diagnostic events."""
        try:
            self.logger.debug(f"Received chatbox_diagnostic event with data: {event_data}")
            message = event_data.get("message", "ChatBox: Unknown issue")
            QTimer.singleShot(0, lambda: self.chat_display.append(message))
            QTimer.singleShot(0, lambda: self.status_bar.setText(f"Status: {message}"))
            self.add_debug_message("ERROR", message, event_data.get("timestamp"))
            self.logger.debug("ChatBox diagnostic message displayed")
            self.log_diagnostic_state("ChatBox diagnostic handled")
        except Exception as e:
            self.logger.error(f"ChatBox diagnostic handling failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to handle chatbox diagnostic event.")

    def handle_chatbox_diagnostic_state(self, event_data):
        """Handle chatbox diagnostic state events."""
        try:
            self.logger.debug(f"Received chatbox_diagnostic_state event with data: {event_data}")
            message = f"ChatBox State: {event_data.get('context', 'Unknown context')}"
            self.add_debug_message("DEBUG", message, time.strftime("%Y-%m-%dT%H:%M:%S"))
            self.logger.debug("ChatBox diagnostic state logged")
        except Exception as e:
            self.logger.error(f"ChatBox diagnostic state handling failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to handle chatbox diagnostic state event.")

    def add_debug_message(self, level: str, message: str, timestamp: str = None):
        """Add a message to the debug log with a timestamp and level."""
        if not timestamp:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        debug_entry = {"level": level, "message": message, "timestamp": timestamp}
        self.debug_messages.append(debug_entry)
        self.update_debug_log()

    def update_debug_log(self):
        """Update the debug log display based on the current filter."""
        try:
            filter_type = self.filter_combo.currentText()
            self.debug_log.clear()
            for entry in self.debug_messages:
                if filter_type == "All" or filter_type == entry["level"]:
                    QTimer.singleShot(0, lambda e=entry: self.debug_log.append(
                        f"[{e['timestamp']}][{e['level']}] {e['message']}"
                    ))
        except Exception as e:
            self.logger.error(f"Debug log update failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to update debug log.")

    def clear_debug_log(self):
        """Clear the debug log."""
        try:
            self.debug_messages = []
            QTimer.singleShot(0, self.debug_log.clear)
            self.logger.debug("Debug log cleared")
            self.log_diagnostic_state("Debug log cleared")
        except Exception as e:
            self.logger.error(f"Clear debug log failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to clear debug log.")

    def install_dependencies(self):
        """Trigger dependency installation via SerenityTools."""
        try:
            self.logger.debug("Installing dependencies...")
            self.tools.install_requirements()
            self.add_debug_message("INFO", "Dependencies installed successfully")
            self.logger.debug("Dependencies installed")
            self.log_diagnostic_state("Dependencies installed")
        except Exception as e:
            self.logger.error(f"Dependency installation failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to install dependencies.")

    def toggle_terminal(self):
        """Toggle terminal visibility with error handling."""
        try:
            if not WIN32_AVAILABLE:
                self.logger.warning("Cannot toggle terminal - pywin32 not installed")
                return
            if not self.hwnd:
                self.hwnd = win32gui.FindWindow("ConsoleWindowClass", None)
                if not self.hwnd:
                    self.logger.error("Terminal window not found")
                    self.report_issue("System: Terminal window not found.")
                    return
            if self.terminal_visible:
                win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)
                QTimer.singleShot(0, lambda: self.terminal_button.setText("Show Terminal"))
                self.terminal_visible = False
                self.logger.debug("Terminal hidden")
            else:
                win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
                QTimer.singleShot(0, lambda: self.terminal_button.setText("Hide Terminal"))
                self.terminal_visible = True
                self.logger.debug("Terminal shown")
            self.log_diagnostic_state("Terminal visibility toggled")
        except Exception as e:
            self.logger.error(f"Terminal toggle failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to toggle terminal visibility.")

    def clear_chat(self):
        """Clear the chat display."""
        try:
            QTimer.singleShot(0, self.chat_display.clear)
            self.logger.debug("Chat display cleared")
            self.log_diagnostic_state("Chat display cleared")
        except Exception as e:
            self.logger.error(f"Clear chat failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to clear chat display.")

    def run_diagnosis(self):
        """Trigger a full system diagnosis."""
        try:
            self.logger.debug("Running system diagnosis...")
            QTimer.singleShot(0, lambda: self.chat_display.append("System: Running diagnostics..."))
            self.events.emit("diagnose_voice", {}, priority=1)
            self.events.emit("system_diagnostic", {"message": "Manual diagnostic triggered"}, priority=1)
            self.log_diagnostic_state("System diagnosis triggered")
        except Exception as e:
            self.logger.error(f"Diagnosis failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to run diagnostics.")

    def confirm_shutdown(self):
        """Show a confirmation dialog before shutting down."""
        try:
            reply = QMessageBox.question(self, "Shutdown Confirmation",
                                       "Are you sure you want to shut down Serenity?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.logger.debug("Shutdown confirmed by user")
                self.events.emit("system_shutdown", {"reason": "User initiated"}, priority=1)
            else:
                self.logger.debug("Shutdown cancelled by user")
                self.log_diagnostic_state("Shutdown cancelled")
        except Exception as e:
            self.logger.error(f"Shutdown confirmation failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to show shutdown confirmation.")

    def shutdown(self, event_data: Dict[str, Any] = None):
        """Handle system shutdown with error handling and interruption support."""
        try:
            self.logger.debug(f"Received system_shutdown event with data: {event_data}")
            self.logger.info("ChatBox shutting down...")
            QTimer.singleShot(0, lambda: self.send_button.setEnabled(False))
            QTimer.singleShot(0, lambda: self.chat_input.setEnabled(False))
            QTimer.singleShot(0, lambda: self.terminal_button.setEnabled(False))
            QTimer.singleShot(0, lambda: self.diagnose_button.setEnabled(False))
            QTimer.singleShot(0, lambda: self.clear_button.setEnabled(False))
            QTimer.singleShot(0, lambda: self.shutdown_button.setEnabled(False))
            self.logger.debug("UI elements disabled during shutdown")
            QTimer.singleShot(0, self.chat_display.clear)
            self.logger.debug("Chat display cleared")
            for event_name, callback in self.event_subscriptions:
                try:
                    self.events.unsubscribe(event_name, callback)
                    self.logger.debug(f"Unsubscribed from event: {event_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to unsubscribe from {event_name}: {str(e)}")
            self.event_subscriptions = []
            self.logger.debug("Unsubscribed from all events")
            self.log_diagnostic_state("ChatBox shut down")
            QTimer.singleShot(0, lambda: sys.exit(0))
        except KeyboardInterrupt:
            self.logger.warning("Shutdown interrupted by KeyboardInterrupt, forcing cleanup...")
            self.event_subscriptions = []
            self.logger.debug("Event subscriptions cleared due to interrupt")
            self.log_diagnostic_state("ChatBox shutdown interrupted")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Shutdown failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to shut down properly.")
            sys.exit(1)

    def set_terminal_handle(self, hwnd):
        """Set the terminal window handle with validation."""
        try:
            self.hwnd = hwnd
            self.logger.debug(f"Terminal handle set to {hwnd}")
            self.log_diagnostic_state("Terminal handle set")
        except Exception as e:
            self.logger.error(f"Setting terminal handle failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("System: Failed to set terminal handle.")

    def report_issue(self, message: str):
        """Report an issue to the user and emit a diagnostic event."""
        try:
            QTimer.singleShot(0, lambda: self.chat_display.append(message))
            self.events.emit("chatbox_diagnostic", {
                "message": message,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            self.logger.debug(f"Reported issue: {message}")
        except Exception as e:
            self.logger.error(f"Failed to report issue: {str(e)}\n{traceback.format_exc()}")

    def log_diagnostic_state(self, context: str):
        """Log the current state of the ChatBox for diagnostic purposes."""
        try:
            state = {
                "context": context,
                "ai_status": self.ai_status,
                "network_status": self.network_status,
                "send_button_enabled": self.send_button.isEnabled(),
                "terminal_visible": self.terminal_visible,
                "event_subscriptions": [event_name for event_name, _ in self.event_subscriptions]
            }
            self.logger.debug(f"Diagnostic state: {state}")
            self.events.emit("chatbox_diagnostic_state", state, priority=1)
        except Exception as e:
            self.logger.error(f"Diagnostic state logging failed: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    import logging
    from PyQt5.QtWidgets import QApplication
    from serenity.utils.event_handler import EventHandler
    from serenity.utils.logger import SerenityLogger

    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    logger = SerenityLogger().logger
    events = EventHandler()

    # Subscribe to diagnostic events for testing
    def chatbox_diagnostic_callback(data):
        print(f"ChatBox Diagnostic: {data}")

    def chatbox_diagnostic_state_callback(data):
        print(f"ChatBox State: {data}")

    events.subscribe("chatbox_diagnostic", chatbox_diagnostic_callback)
    events.subscribe("chatbox_diagnostic_state", chatbox_diagnostic_state_callback)

    chat_box = ChatBox(logger, events)

    # Test 1: Initialize and show
    print("Test 1: Initializing and showing ChatBox")
    chat_box.show()

    # Test 2: Simulate system start
    print("Test 2: Simulating system start")
    events.emit("system_started", {"status": "running"}, priority=1)

    # Test 3: Simulate AI initialization
    print("Test 3: Simulating AI initialization")
    events.emit("ai_initialized", {"status": "ready"}, priority=1)

    # Test 4: Simulate network status update
    print("Test 4: Simulating network status update")
    events.emit("network_status", {"status": "connected"}, priority=1)

    # Test 5: Simulate AI response
    print("Test 5: Simulating AI response")
    events.emit("ai_response", {"response": "Hello, user!"}, priority=2)

    # Test 6: Simulate voice diagnostic
    print("Test 6: Simulating voice diagnostic")
    events.emit("voice_diagnostic", {"message": "Microphone not detected"}, priority=1)

    # Test 7: Simulate system diagnostic
    print("Test 7: Simulating system diagnostic")
    events.emit("system_diagnostic", {"message": "System check: All good"}, priority=1)

    # Test 8: Simulate config diagnostic
    print("Test 8: Simulating config diagnostic")
    events.emit("config_diagnostic", {"message": "Config load failed"}, priority=1)

    # Test 9: Simulate user input (manual interaction required in real test)
    print("Test 9: Please type a message and press Send (manual test)")

    # Test 10: Run diagnosis
    print("Test 10: Running diagnosis")
    chat_box.run_diagnosis()

    # Test 11: Clear chat
    print("Test 11: Clearing chat")
    chat_box.clear_chat()

    # Test 12: Clear debug log
    print("Test 12: Clearing debug log")
    chat_box.clear_debug_log()

    # Test 13: Install dependencies
    print("Test 13: Installing dependencies")
    chat_box.install_dependencies()

    # Test 14: Shutdown (manual confirmation required in real test)
    print("Test 14: Please click Shutdown and confirm (manual test)")

    app.exec_()