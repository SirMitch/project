# File: serenity/voice/processor.py
# Created: 2025-03-25 15:47:23
# Updated: 2025-03-26 23:45:00
# Created by: Grok 3, xAI in collaboration with Mitch827
# Purpose: Voice Processing System
# Version: 1.0.3

"""
Serenity Voice Processing System
Handles voice input and output with diagnostic tools for microphone and speech issues.
Enhanced with microphone selection, speech engine configuration, and performance optimizations.
"""

import logging
import speech_recognition as sr
import pyttsx3
from typing import Dict, Any, Optional
import threading
import time
import pyaudio
import queue

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
        # For testing, return a mock class; in production, this would attempt a real import
        class MockClass:
            def __init__(self):
                self.logger = logging.getLogger(f"Mock.{class_name}")
        return MockClass

# Initialize tools
tools = SerenityTools("Serenity.Voice")
tools.install_requirements()
tools.ensure_package("speech_recognition")
tools.ensure_package("pyttsx3")
tools.ensure_package("pyaudio")

class VoiceProcessor:
    def __init__(self):
        self.logger = logging.getLogger('Serenity.Voice')
        self.config = None
        self.events = None
        self.running = False
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()
        self.mic_available = False
        self.enable_voice_input = True
        self.selected_mic_index = None  # Allow specific microphone selection
        self.listen_timeout = 5  # Configurable timeout for listening
        self.speech_queue = queue.Queue()  # Queue for speech responses
        self.speech_thread = None

    def set_config(self, config):
        """Set configuration from ConfigManager."""
        self.config = config
        self.engine.setProperty('rate', self.config.get("voice", "speech_rate", 150))
        self.engine.setProperty('volume', self.config.get("voice", "volume", 1.0))
        # Set voice if specified
        preferred_voice = self.config.get("voice", "voice_id", None)
        if preferred_voice:
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if voice.id == preferred_voice:
                    self.engine.setProperty('voice', voice.id)
                    self.logger.info(f"Set voice to {voice.name}")
                    break
            else:
                self.logger.warning(f"Voice {preferred_voice} not found")
        self.enable_voice_input = self.config.get("voice", "enable_input", True)
        self.listen_timeout = self.config.get("voice", "listen_timeout", 5)
        self.logger.info("Voice processor configuration set")

    def set_events(self, events):
        """Set event handler for communication."""
        self.events = events
        self.events.subscribe("data_processed", self.speak_response)
        self.events.subscribe("system_shutdown", self.stop)
        self.events.subscribe("diagnose_voice", self.diagnose)
        self.logger.info("Voice processor event handler set")

    def _check_mic(self):
        """Check if a default or selected input device is available."""
        try:
            p = pyaudio.PyAudio()
            if self.selected_mic_index is not None:
                device_info = p.get_device_info_by_index(self.selected_mic_index)
                if device_info.get("maxInputChannels", 0) <= 0:
                    raise ValueError(f"Device at index {self.selected_mic_index} is not an input device")
            else:
                device_info = p.get_default_input_device_info()
            p.terminate()
            self.mic_available = True
            self.logger.info(f"Microphone detected: {device_info.get('name', 'Unknown')}")
            return True
        except OSError:
            self.mic_available = False
            self.logger.warning("No default microphone available - voice input disabled")
            self.events.emit("voice_diagnostic", {
                "message": "No default microphone detected. Please ensure a microphone is connected and set as the default input device.",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            return False
        except Exception as e:
            self.mic_available = False
            self.logger.error(f"Microphone check failed: {str(e)}")
            self.events.emit("voice_diagnostic", {
                "message": f"Microphone check failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            return False

    def select_microphone(self, mic_index: int):
        """Select a specific microphone by index."""
        try:
            p = pyaudio.PyAudio()
            device_info = p.get_device_info_by_index(mic_index)
            p.terminate()
            if device_info.get("maxInputChannels", 0) <= 0:
                self.logger.error(f"Device at index {mic_index} is not an input device")
                return False
            self.selected_mic_index = mic_index
            self.mic_available = False  # Reset and recheck
            self._check_mic()
            self.logger.info(f"Selected microphone: {device_info.get('name', 'Unknown')}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to select microphone at index {mic_index}: {str(e)}")
            self.events.emit("voice_diagnostic", {
                "message": f"Failed to select microphone at index {mic_index}: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            return False

    def list_microphones(self):
        """List all available microphones for diagnostics."""
        try:
            p = pyaudio.PyAudio()
            devices = []
            default_index = p.get_default_input_device_info().get("index") if p.get_default_input_device_info() else None
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info.get("maxInputChannels", 0) > 0:
                    devices.append({
                        "index": i,
                        "name": device_info.get("name", "Unknown"),
                        "default": i == default_index
                    })
            p.terminate()
            return devices
        except Exception as e:
            self.logger.error(f"Failed to list microphones: {str(e)}")
            self.events.emit("voice_diagnostic", {
                "message": f"Failed to list microphones: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            return []

    def start(self):
        """Start the voice processor."""
        try:
            if not self.config or not self.events:
                raise ValueError("Configuration or event handler not set")
            self.running = True
            if self.enable_voice_input and self._check_mic():
                threading.Thread(target=self.listen_loop, daemon=True).start()
            else:
                self.logger.info("Voice processor started without listening (no mic or disabled)")
            # Start speech processing thread
            self.speech_thread = threading.Thread(target=self.speech_loop, daemon=True)
            self.speech_thread.start()
            self.logger.info("Voice processor started")
        except Exception as e:
            self.logger.error(f"Voice processor startup failed: {str(e)}")
            self.events.emit("voice_diagnostic", {
                "message": f"Voice processor startup failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            self.stop()

    def listen_loop(self):
        """Continuously listen for voice input if mic is available."""
        if not self.mic_available:
            self.logger.debug("Listen loop skipped - no microphone available")
            return
        retries = 3
        while self.running and retries > 0:
            try:
                mic = sr.Microphone(device_index=self.selected_mic_index)
                with mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    self.logger.info("Voice listening loop started")
                    while self.running:
                        try:
                            self.logger.debug("Listening for voice input...")
                            audio = self.recognizer.listen(source, timeout=self.listen_timeout)
                            text = self.recognizer.recognize_google(audio)
                            self.logger.info(f"Recognized voice input: {text}")
                            with self.lock:
                                self.events.emit("input_received", {
                                    "text": text,
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                                }, priority=2)
                            retries = 3  # Reset retries on success
                        except sr.UnknownValueError:
                            self.logger.debug("Could not understand audio")
                        except sr.RequestError as e:
                            self.logger.error(f"Speech recognition request failed: {str(e)}")
                        except Exception as e:
                            self.logger.error(f"Voice listening failed: {str(e)}")
            except OSError as e:
                self.logger.error(f"Microphone access failed: {str(e)}")
                self.mic_available = False
                retries -= 1
                if retries > 0:
                    self.logger.info(f"Retrying microphone access ({retries} attempts left)...")
                    time.sleep(1)
                    self._check_mic()
                else:
                    self.events.emit("voice_diagnostic", {
                        "message": f"Microphone access failed after retries: {str(e)}",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }, priority=1)
                    break

    def speech_loop(self):
        """Process speech responses from the queue."""
        while self.running:
            try:
                text = self.speech_queue.get(timeout=1)
                with self.lock:
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.logger.info(f"Spoke response: {text}")
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Speech output failed: {str(e)}")
                self.events.emit("voice_diagnostic", {
                    "message": f"Speech output failed: {str(e)}",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }, priority=1)

    def speak_response(self, event_data: Dict[str, Any]):
        """Queue a response to be spoken."""
        try:
            if not self.running:
                self.logger.warning("Voice processor not running")
                return
            data = event_data.get("data", {})
            text = data.get("original", "")
            if text:
                self.speech_queue.put(text)
        except Exception as e:
            self.logger.error(f"Failed to queue speech response: {str(e)}")
            self.events.emit("voice_diagnostic", {
                "message": f"Failed to queue speech response: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def stop(self, event_data=None):
        """Stop the voice processor."""
        try:
            self.running = False
            self.engine.stop()
            # Clear the speech queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                except queue.Empty:
                    break
            if self.speech_thread:
                self.speech_thread.join()
            self.logger.info("Voice processor stopped")
        except Exception as e:
            self.logger.error(f"Voice processor shutdown failed: {str(e)}")
            self.events.emit("voice_diagnostic", {
                "message": f"Voice processor shutdown failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

    def diagnose(self, event_data=None):
        """Diagnose the health of the voice processor."""
        try:
            issues = []
            status = {
                "running": self.running,
                "mic_available": self.mic_available,
                "enable_voice_input": self.enable_voice_input,
                "selected_mic_index": self.selected_mic_index,
                "microphones": self.list_microphones(),
                "speech_engine": "unknown",
                "speech_queue_size": self.speech_queue.qsize()
            }
            if not self.mic_available:
                issues.append("No default microphone available.")
            if not self.enable_voice_input:
                issues.append("Voice input is disabled in configuration.")
            if not status["microphones"]:
                issues.append("No microphones detected.")
            # Check speech engine
            try:
                voices = self.engine.getProperty('voices')
                status["speech_engine"] = "operational"
                status["available_voices"] = [voice.id for voice in voices]
            except Exception as e:
                status["speech_engine"] = f"failed: {str(e)}"
                issues.append(f"Speech engine error: {str(e)}")
            status["issues"] = issues
            self.logger.info(f"VoiceProcessor diagnostic: {status}")
            self.events.emit("voice_diagnostic", status, priority=1)
            return status
        except Exception as e:
            self.logger.error(f"VoiceProcessor diagnosis failed: {str(e)}")
            self.events.emit("voice_diagnostic", {
                "message": f"Diagnosis failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            return {"issues": [f"Diagnosis failed: {str(e)}"]}

if __name__ == "__main__":
    from serenity.utils.event_handler import EventHandler
    from serenity.utils.config_manager import ConfigManager

    logging.basicConfig(level=logging.INFO)

    # Mock ConfigManager for testing
    class MockConfigManager:
        def get(self, section, key, default=None):
            config = {
                "voice": {
                    "speech_rate": 150,
                    "volume": 1.0,
                    "enable_input": True,
                    "listen_timeout": 5
                }
            }
            return config.get(section, {}).get(key, default)

        def set_events(self, events):
            print("ConfigManager set_events called")

    # Test VoiceProcessor
    voice = VoiceProcessor()
    config = MockConfigManager()
    events = EventHandler()

    # Subscribe to diagnostic events
    def diagnostic_callback(data):
        print(f"Diagnostic event: {data}")

    events.subscribe("voice_diagnostic", diagnostic_callback)
    events.subscribe("input_received", lambda data: print(f"Input received: {data}"))

    voice.set_config(config)
    voice.set_events(events)

    # Test 1: Start the voice processor
    print("Test 1: Starting voice processor")
    voice.start()

    # Test 2: List microphones
    print("Test 2: Listing microphones")
    mics = voice.list_microphones()
    print(f"Available microphones: {mics}")

    # Test 3: Select a microphone (if available)
    if mics:
        print("Test 3: Selecting first available microphone")
        voice.select_microphone(mics[0]["index"])

    # Test 4: Speak a response
    print("Test 4: Speaking a test response")
    voice.speak_response({"data": {"original": "This is a test response"}})

    # Test 5: Diagnose
    print("Test 5: Running diagnostics")
    voice.diagnose()

    # Test 6: Simulate data processing
    print("Test 6: Simulating data processing")
    events.emit("data_processed", {"data": {"original": "Hello, Serenity!"}}, priority=2)

    time.sleep(5)  # Allow some time for processing

    # Test 7: Stop
    print("Test 7: Stopping voice processor")
    voice.stop()