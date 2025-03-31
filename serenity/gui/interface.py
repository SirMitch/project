# File: serenity/gui/interface.py
# Created: 2024-12-24 15:45:28
# Updated: 2025-04-01 11:30:00
# Created by: Original author Mitch827, updated by Grok 3, xAI in collaboration with Mitch827
# Purpose: Main GUI Interface for Serenity
# Version: 1.2.12

"""
Serenity GUI Interface
Main GUI controller that assembles modular components: SensorSuite, LogViewer, and ChatBox.
Includes status bar, diagnostic tools, improved theme consistency, and AI model display.
"""

import sys
import traceback
import time
from serenity.utils.tools import SerenityTools
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QApplication, QStatusBar, QAction, QMenuBar, QMessageBox, QLabel
from PyQt5.QtCore import QSettings
from serenity.gui.sensor_suite import SensorSuite
from serenity.gui.log_viewer import LogViewer
from serenity.gui.chat_box import ChatBox

tools = SerenityTools("Serenity.GUI")
tools.ensure_package("PyQt5")
tools.ensure_package("PyQtChart")
tools.ensure_package("pywin32")

# Check for pywin32 availability
try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    tools.logger.warning("pywin32 not available - terminal control features disabled")

class SerenityGUI(QMainWindow):
    def __init__(self):
        """Initialize the GUI with centralized tools."""
        print("SerenityGUI.__init__ called")
        super().__init__()
        self.tools = tools
        self.logger = self.tools.logger
        print("SerenityGUI: Logger assigned")
        self.logger.debug("Initializing SerenityGUI...")
        self.config = None
        self.events = None
        self.running = False
        self.sensor_suite = None
        self.log_viewer = None
        self.chat_box = None
        self.terminal_handle = None
        self.ai_model_name = "Unknown"  # Placeholder for AI model name
        self.settings = QSettings("Serenity", "GUI")  # For saving window geometry and theme
        self.theme = self.settings.value("theme", "dark")  # Default to dark theme
        self.app = QApplication.instance() or QApplication(sys.argv)  # Ensure single QApplication instance
        self.setup_gui()
        self.setup_menu()
        self.logger.debug("SerenityGUI initialized")
        print("SerenityGUI.__init__ completed")

    def set_config(self, config):
        """Set configuration from ConfigManager."""
        print("SerenityGUI.set_config called")
        try:
            self.logger.debug("Setting GUI config...")
            self.config = config
            self.logger.info("GUI configuration set")
            print("SerenityGUI.set_config completed")
        except Exception as e:
            self.logger.error(f"GUI config setup failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.set_config failed: {str(e)}")
            raise

    def set_events(self, events):
        """Set event handler for communication."""
        print("SerenityGUI.set_events called")
        self.logger.debug("Entering set_events method...")
        self.events = events
        self.logger.debug("Events assigned successfully")
        self.events.emit("process_step", {
            "step": "Setting up GUI event handlers",
            "category": "Action",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }, priority=1)
        print("SerenityGUI: Events assigned")
        try:
            # Initialize modules with detailed logging
            self.logger.debug("Initializing SensorSuite...")
            print("SerenityGUI: Initializing SensorSuite...")
            self.sensor_suite = SensorSuite(self.logger, self.events)
            self.logger.debug("SensorSuite initialized successfully")
            print("SerenityGUI: SensorSuite initialized")
            
            self.logger.debug("Initializing LogViewer...")
            print("SerenityGUI: Initializing LogViewer...")
            self.log_viewer = LogViewer(self.logger, self.events)
            self.logger.debug("LogViewer initialized successfully")
            print("SerenityGUI: LogViewer initialized")
            
            self.logger.debug("Initializing ChatBox...")
            print("SerenityGUI: Initializing ChatBox...")
            self.chat_box = ChatBox(self.logger, self.events)
            self.logger.debug("ChatBox initialized successfully")
            print("SerenityGUI: ChatBox initialized")
            
            # Subscribe to events
            self.logger.debug("Subscribing to system_shutdown event...")
            print("SerenityGUI: Subscribing to system_shutdown...")
            self.events.subscribe("system_shutdown", self.stop)
            self.logger.debug("Subscribed to system_shutdown event")
            print("SerenityGUI: Subscribed to system_shutdown")
            
            self.logger.debug("Subscribing to ai_status event...")
            print("SerenityGUI: Subscribing to ai_status...")
            self.events.subscribe("ai_status", self.update_status_bar)
            self.logger.debug("Subscribed to ai_status event")
            print("SerenityGUI: Subscribed to ai_status")
            
            self.logger.debug("Subscribing to network_status event...")
            print("SerenityGUI: Subscribing to network_status...")
            self.events.subscribe("network_status", self.update_status_bar)
            self.logger.debug("Subscribed to network_status event")
            print("SerenityGUI: Subscribed to network_status")
            
            self.logger.debug("Subscribing to ai_diagnostic event...")
            print("SerenityGUI: Subscribing to ai_diagnostic...")
            self.events.subscribe("ai_diagnostic", self.show_diagnostic_message)
            self.logger.debug("Subscribed to ai_diagnostic event")
            print("SerenityGUI: Subscribed to ai_diagnostic")
            
            self.logger.debug("Subscribing to ai_initialized event...")
            print("SerenityGUI: Subscribing to ai_initialized...")
            self.events.subscribe("ai_initialized", self.update_ai_model)
            self.logger.debug("Subscribed to ai_initialized event")
            print("SerenityGUI: Subscribed to ai_initialized")
            
            self.logger.info("GUI event handler set")
            print("SerenityGUI: GUI event handler set")
            # Update GUI layout with actual widgets
            self.logger.debug("Calling update_gui_layout...")
            print("SerenityGUI: Calling update_gui_layout...")
            self.update_gui_layout()
            self.logger.debug("update_gui_layout completed")
            print("SerenityGUI: update_gui_layout completed")
        except Exception as e:
            self.logger.error(f"GUI event setup failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.set_events failed: {str(e)}")
            raise

    def setup_gui(self):
        """Initialize the GUI layout with placeholders for modular components."""
        print("SerenityGUI.setup_gui called")
        try:
            self.logger.debug("Setting up GUI layout...")
            self.setWindowTitle("Serenity AI Assistant")

            # Load saved geometry or set default
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            else:
                self.setGeometry(100, 100, 1000, 600)

            main_widget = QWidget(self)
            self.setCentralWidget(main_widget)
            self.main_layout = QHBoxLayout(main_widget)

            # Left panel: System stats and logs
            self.left_panel = QWidget()
            self.left_layout = QVBoxLayout(self.left_panel)
            self.main_layout.addWidget(self.left_panel, stretch=1)

            # Right panel: Chat and controls
            self.chat_box_container = QWidget()
            self.main_layout.addWidget(self.chat_box_container, stretch=2)

            # Status bar with AI model label
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.ai_model_label = QLabel(f"Model: {self.ai_model_name}")
            self.status_bar.addPermanentWidget(self.ai_model_label)
            self.status_bar.showMessage("Initializing...")

            # Apply theme
            self.apply_theme()

            self.logger.debug("GUI layout setup complete")
            print("SerenityGUI.setup_gui completed")
        except Exception as e:
            self.logger.error(f"GUI setup failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.setup_gui failed: {str(e)}")
            raise

    def update_gui_layout(self):
        """Update the GUI layout with actual widgets after events are set."""
        print("SerenityGUI.update_gui_layout called")
        try:
            self.logger.debug("Updating GUI layout with actual widgets...")
            # Add actual widgets to the layout
            if self.sensor_suite:
                self.left_layout.addWidget(self.sensor_suite)
                self.logger.debug("SensorSuite added to layout")
                print("SerenityGUI: SensorSuite added to layout")
            else:
                self.logger.warning("SensorSuite not initialized, skipping addition to layout")
                print("SerenityGUI: SensorSuite not initialized")
            if self.log_viewer:
                self.left_layout.addWidget(self.log_viewer)
                self.logger.debug("LogViewer added to layout")
                print("SerenityGUI: LogViewer added to layout")
            else:
                self.logger.warning("LogViewer not initialized, skipping addition to layout")
                print("SerenityGUI: LogViewer not initialized")
            if self.chat_box:
                self.main_layout.removeWidget(self.chat_box_container)
                self.chat_box_container.setParent(None)
                self.main_layout.addWidget(self.chat_box, stretch=2)
                self.logger.debug("ChatBox added to layout")
                print("SerenityGUI: ChatBox added to layout")
            else:
                self.logger.warning("ChatBox not initialized, skipping addition to layout")
                print("SerenityGUI: ChatBox not initialized")
            # Reapply theme to ensure child widgets are styled
            self.apply_theme()
            self.logger.debug("GUI layout updated with actual widgets")
            print("SerenityGUI.update_gui_layout completed")
        except Exception as e:
            self.logger.error(f"GUI layout update failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.update_gui_layout failed: {str(e)}")
            raise

    def setup_menu(self):
        """Set up the menu bar with diagnostic, theme, and AI model options."""
        print("SerenityGUI.setup_menu called")
        try:
            menu_bar = self.menuBar()
            tools_menu = menu_bar.addMenu("Tools")

            # Terminal toggle action
            terminal_action = QAction("Show Terminal", self)
            terminal_action.setCheckable(True)
            terminal_action.setChecked(False)
            terminal_action.triggered.connect(self.toggle_terminal)
            tools_menu.addAction(terminal_action)

            # Diagnostics action
            diagnostics_action = QAction("Diagnostics", self)
            diagnostics_action.triggered.connect(self.show_diagnostics)
            tools_menu.addAction(diagnostics_action)

            # Theme toggle action
            theme_action = QAction("Toggle Theme (Light/Dark)", self)
            theme_action.triggered.connect(self.toggle_theme)
            tools_menu.addAction(theme_action)

            # AI Model selection menu (placeholder for future feature)
            ai_model_menu = tools_menu.addMenu("Select AI Model")
            distilgpt2_action = QAction("distilgpt2 (Current)", self)
            distilgpt2_action.setEnabled(False)  # Placeholder, disabled for now
            ai_model_menu.addAction(distilgpt2_action)
            placeholder_action = QAction("More models coming soon...", self)
            placeholder_action.setEnabled(False)
            ai_model_menu.addAction(placeholder_action)

            # Persona selection menu (placeholder for future feature)
            persona_menu = tools_menu.addMenu("Select Persona")
            default_persona_action = QAction("Default (Current)", self)
            default_persona_action.setEnabled(False)  # Placeholder, disabled for now
            persona_menu.addAction(default_persona_action)
            placeholder_persona_action = QAction("More personas coming soon...", self)
            placeholder_persona_action.setEnabled(False)
            persona_menu.addAction(placeholder_persona_action)

            self.logger.debug("Menu bar setup complete")
            print("SerenityGUI.setup_menu completed")
        except Exception as e:
            self.logger.error(f"Menu setup failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.setup_menu failed: {str(e)}")
            raise

    def apply_theme(self):
        """Apply the selected theme to the GUI."""
        print("SerenityGUI.apply_theme called")
        try:
            if self.theme == "dark":
                stylesheet = """
                    QMainWindow, QWidget {
                        background-color: #1B5E20;  /* Darker green background */
                    }
                    QTabWidget::pane {
                        border: 1px solid #000000;  /* Black trim */
                        background-color: #2F5A87;  /* Darker blue tabs */
                    }
                    QTabBar::tab {
                        background: #2F5A87;
                        color: white;
                        padding: 5px;
                        border: 1px solid #000000;
                    }
                    QTabBar::tab:selected {
                        background: #1B5E20;
                        border: 1px solid #000000;
                        border-bottom: none;
                    }
                    QTextEdit, QLineEdit {
                        background-color: white;
                        border: 1px solid #000000;
                    }
                    QPushButton {
                        background-color: #2F5A87;
                        color: white;
                        border: 1px solid #000000;
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: #437BB5;
                    }
                    QPushButton:disabled {
                        background-color: #A9A9A9;
                    }
                    QLabel {
                        color: white;
                    }
                    QStatusBar {
                        background-color: #2F5A87;
                        color: white;
                    }
                """
            else:
                stylesheet = """
                    QMainWindow, QWidget {
                        background-color: #E0F7FA;  /* Light cyan background */
                    }
                    QTabWidget::pane {
                        border: 1px solid #000000;
                        background-color: #B2EBF2;
                    }
                    QTabBar::tab {
                        background: #B2EBF2;
                        color: black;
                        padding: 5px;
                        border: 1px solid #000000;
                    }
                    QTabBar::tab:selected {
                        background: #E0F7FA;
                        border: 1px solid #000000;
                        border-bottom: none;
                    }
                    QTextEdit, QLineEdit {
                        background-color: white;
                        border: 1px solid #000000;
                    }
                    QPushButton {
                        background-color: #4FC3F7;
                        color: black;
                        border: 1px solid #000000;
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: #81D4FA;
                    }
                    QPushButton:disabled {
                        background-color: #B0BEC5;
                    }
                    QLabel {
                        color: black;
                    }
                    QStatusBar {
                        background-color: #B2EBF2;
                        color: black;
                    }
                """
            self.setStyleSheet(stylesheet)
            # Reapply stylesheet to child widgets
            if self.sensor_suite:
                self.sensor_suite.setStyleSheet(stylesheet)
            if self.log_viewer:
                self.log_viewer.setStyleSheet(stylesheet)
            if self.chat_box:
                self.chat_box.setStyleSheet(stylesheet)
            if self.ai_model_label:
                self.ai_model_label.setStyleSheet(f"QLabel {{ color: {'white' if self.theme == 'dark' else 'black'} }}")
            self.logger.debug(f"Applied {self.theme} theme")
            print("SerenityGUI.apply_theme completed")
        except Exception as e:
            self.logger.error(f"Theme application failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.apply_theme failed: {str(e)}")
            raise

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        print("SerenityGUI.toggle_theme called")
        try:
            self.theme = "light" if self.theme == "dark" else "dark"
            self.settings.setValue("theme", self.theme)
            self.apply_theme()
            self.status_bar.showMessage(f"Switched to {self.theme} theme", 3000)
            print("SerenityGUI.toggle_theme completed")
        except Exception as e:
            self.logger.error(f"Theme toggle failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.toggle_theme failed: {str(e)}")
            raise

    def toggle_terminal(self, checked):
        """Toggle terminal visibility."""
        print("SerenityGUI.toggle_terminal called")
        try:
            if not WIN32_AVAILABLE:
                self.status_bar.showMessage("Terminal control not available (pywin32 missing)", 3000)
                print("SerenityGUI.toggle_terminal: pywin32 missing")
                return
            if self.terminal_handle:
                win32gui.ShowWindow(self.terminal_handle, win32con.SW_SHOW if checked else win32con.SW_HIDE)
                self.status_bar.showMessage("Terminal shown" if checked else "Terminal hidden", 3000)
                print("SerenityGUI.toggle_terminal: Terminal toggled")
            else:
                self.status_bar.showMessage("Terminal handle not found", 3000)
                print("SerenityGUI.toggle_terminal: Terminal handle not found")
        except Exception as e:
            self.logger.error(f"Terminal toggle failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.toggle_terminal failed: {str(e)}")
            raise

    def show_diagnostics(self):
        """Show diagnostic information in a dialog."""
        print("SerenityGUI.show_diagnostics called")
        try:
            self.events.emit("process_step", {
                "step": "Displaying diagnostic information",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            diagnostics = []
            diagnostics.append(f"GUI Running: {self.running}")
            diagnostics.append(f"Theme: {self.theme}")
            diagnostics.append(f"AI Model: {self.ai_model_name}")
            diagnostics.append(f"WIN32 Available: {WIN32_AVAILABLE}")
            if self.sensor_suite:
                diagnostics.append(f"SensorSuite Initialized: {self.sensor_suite.is_initialized()}")
            else:
                diagnostics.append("SensorSuite: Not initialized")
            if self.log_viewer:
                diagnostics.append(f"LogViewer Initialized: {self.log_viewer.is_initialized()}")
            else:
                diagnostics.append("LogViewer: Not initialized")
            if self.chat_box:
                diagnostics.append(f"ChatBox AI Status: {self.chat_box.ai_status}")
                diagnostics.append(f"ChatBox Network Status: {self.chat_box.network_status}")
            else:
                diagnostics.append("ChatBox: Not initialized")
            message = "\n".join(diagnostics)
            QMessageBox.information(self, "Diagnostics", message)
            print("SerenityGUI.show_diagnostics completed")
        except Exception as e:
            self.logger.error(f"Diagnostics display failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.show_diagnostics failed: {str(e)}")
            raise

    def update_status_bar(self, data: dict):
        """Update the status bar with system status and update ChatBox state."""
        print("SerenityGUI.update_status_bar called")
        try:
            self.events.emit("process_step", {
                "step": "Updating status bar with system status",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            # Update ChatBox state first
            if self.chat_box:
                ai_status = data.get("ai_status", self.chat_box.ai_status)
                network_status = data.get("status", self.chat_box.network_status)
                self.chat_box.ai_status = ai_status
                self.chat_box.network_status = network_status
                # Update Send button state based on AI and network status
                if ai_status == "green" and network_status == "green":
                    self.chat_box.send_button.setEnabled(True)
                    self.logger.debug("AI and Network green, Send button enabled")
                else:
                    self.chat_box.send_button.setEnabled(False)
                    self.logger.debug("AI or Network not green, Send button disabled")
            else:
                ai_status = data.get("ai_status", "red")
                network_status = data.get("status", "red")

            # Update status bar message
            message = f"AI: {ai_status.capitalize()} | Network: {network_status.capitalize()}"
            self.status_bar.showMessage(message)
            print("SerenityGUI.update_status_bar completed")
        except Exception as e:
            self.logger.error(f"Status bar update failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.update_status_bar failed: {str(e)}")
            raise

    def update_ai_model(self, data: dict):
        """Update the AI model name displayed in the GUI."""
        print("SerenityGUI.update_ai_model called")
        try:
            self.events.emit("process_step", {
                "step": f"Updating AI model name to {data.get('model_name', 'Unknown')}",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            self.ai_model_name = data.get("model_name", "Unknown")
            self.ai_model_label.setText(f"Model: {self.ai_model_name}")
            self.logger.debug(f"Updated AI model name to {self.ai_model_name}")
            print("SerenityGUI.update_ai_model completed")
        except Exception as e:
            self.logger.error(f"AI model update failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.update_ai_model failed: {str(e)}")
            raise

    def show_diagnostic_message(self, data: dict):
        """Show diagnostic messages in the status bar."""
        print("SerenityGUI.show_diagnostic_message called")
        try:
            self.events.emit("process_step", {
                "step": "Showing diagnostic message in status bar",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            message = data.get("message", "Unknown error")
            self.status_bar.showMessage(f"Error: {message}", 5000)
            print("SerenityGUI.show_diagnostic_message completed")
        except Exception as e:
            self.logger.error(f"Diagnostic message display failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.show_diagnostic_message failed: {str(e)}")
            raise

    def start(self):
        """Start the GUI with environment checks."""
        print("SerenityGUI.start called")
        try:
            self.logger.debug("Starting GUI...")
            self.events.emit("process_step", {
                "step": "Starting GUI and performing environment checks",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            if not self.config or not self.events:
                raise ValueError("Configuration or event handler not set")
            self.running = True
            if WIN32_AVAILABLE:
                try:
                    self.terminal_handle = win32gui.FindWindow("ConsoleWindowClass", None)
                    if self.terminal_handle:
                        win32gui.ShowWindow(self.terminal_handle, win32con.SW_HIDE)
                        if self.chat_box:
                            self.chat_box.set_terminal_handle(self.terminal_handle)
                    else:
                        self.logger.warning("Terminal window not found")
                        print("SerenityGUI.start: Terminal window not found")
                except Exception as e:
                    self.logger.error(f"Failed to hide terminal: {str(e)}\n{traceback.format_exc()}")
                    print(f"SerenityGUI.start: Failed to hide terminal: {str(e)}")
            self.show()
            self.logger.info("GUI started")
            print("SerenityGUI.start completed")
            self.app.exec_()  # Start the PyQt5 event loop
        except Exception as e:
            self.logger.error(f"GUI startup failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.start failed: {str(e)}")
            self.stop()
            raise

    def stop(self, event_data=None):
        """Stop the GUI and clean up child widgets."""
        print("SerenityGUI.stop called")
        try:
            self.logger.debug("Stopping GUI...")
            self.events.emit("process_step", {
                "step": "Stopping GUI and cleaning up resources",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            self.running = False
            # Clean up child widgets
            if self.sensor_suite:
                self.sensor_suite.close()
            if self.log_viewer:
                self.log_viewer.close()
            if self.chat_box:
                self.chat_box.close()
            # Unsubscribe from events
            if self.events:
                self.events.unsubscribe("system_shutdown", self.stop)
                self.events.unsubscribe("ai_status", self.update_status_bar)
                self.events.unsubscribe("network_status", self.update_status_bar)
                self.events.unsubscribe("ai_diagnostic", self.show_diagnostic_message)
                self.events.unsubscribe("ai_initialized", self.update_ai_model)
            # Save window geometry
            self.settings.setValue("geometry", self.saveGeometry())
            # Close the window and quit the application
            self.close()
            self.app.quit()
            self.logger.info("GUI stopped")
            print("SerenityGUI.stop completed")
        except Exception as e:
            self.logger.error(f"GUI shutdown failed: {str(e)}\n{traceback.format_exc()}")
            print(f"SerenityGUI.stop failed: {str(e)}")
            raise

    def closeEvent(self, event):
        """Handle window close event."""
        print("SerenityGUI.closeEvent called")
        self.stop()
        event.accept()

if __name__ == "__main__":
    from serenity.utils.config_manager import ConfigManager
    from serenity.utils.event_handler import EventHandler
    app = QApplication(sys.argv)
    gui = SerenityGUI()
    config = ConfigManager()
    config.load_config()
    events = EventHandler()
    events.start()
    gui.set_config(config)
    gui.set_events(events)
    gui.start()
    sys.exit(app.exec_())