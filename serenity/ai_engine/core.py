# File: serenity/ai_engine/core.py
# Created: 2024-12-24 15:34:12
# Updated: 2025-04-01 11:00:00
# Purpose: Core AI Processing Engine
# Version: 1.0.20

"""
Serenity Core AI Processing Engine
Handles primary AI operations and response generation.
Includes robust error handling, detailed logging, self-diagnostic tools, resource monitoring,
and performance optimizations for older hardware.
"""

import logging
import torch
import traceback
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Add the missing import for transformers
try:
    import transformers
except ImportError:
    transformers = None
from typing import Dict, Optional
import time
import os
import threading
import queue
import psutil  # For resource monitoring
import sys
from torch import autocast  # For FP16 inference

# Version check
EXPECTED_VERSION = "1.0.20"
CURRENT_VERSION = "1.0.20"
if CURRENT_VERSION != EXPECTED_VERSION:
    logging.getLogger('Serenity.AI').warning(
        f"Version mismatch in core.py: Expected {EXPECTED_VERSION}, but found {CURRENT_VERSION}. "
        "Please update the file to the correct version."
    )

class AIProcessor:
    def __init__(self):
        self.logger = logging.getLogger('Serenity.AI')
        self.device = None  # Set in initialize
        self.config = None
        self.events = None
        self.model = None
        self.tokenizer = None
        self.running = False
        self.initialized = False
        self.event_subscriptions = []
        self.process = psutil.Process()
        self.use_mock = False
        self.response_cache = {}  # Cache for recent responses
        self.current_task = None  # Track the current response generation task
        self.cancel_event = threading.Event()  # For canceling long-running tasks
        self.model_path = None  # Store model path for GUI display

    def set_config(self, config):
        """Set configuration from ConfigManager with error handling."""
        try:
            self.config = config
            self.logger.info("AI configuration set")
            self.log_diagnostic_state("AI configuration set")
        except Exception as e:
            self.logger.error(f"Failed to set config: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("AI: Failed to set configuration.")

    def set_events(self, events):
        """Set event handler for communication with error handling."""
        try:
            self.events = events
            self.event_subscriptions = [
                ("input_received", self.process_input),
                ("cancel_task", self.cancel_task)
            ]
            for event_name, callback in self.event_subscriptions:
                self.events.subscribe(event_name, callback)
                self.logger.debug(f"Subscribed to event: {event_name}")
            self.logger.info("AI event handler set")
            self.log_diagnostic_state("AI event handler set")
        except Exception as e:
            self.logger.error(f"Failed to set events: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("AI: Failed to set event handler.")

    def check_hardware_compatibility(self):
        """Check hardware compatibility and emit diagnostic events."""
        try:
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability(0)
                if compute_capability[0] < 6:
                    self.logger.warning("GPU compute capability is below 6.0. Performance may be poor.")
                    self.events.emit("ai_diagnostic", {
                        "message": f"GPU compute capability {compute_capability[0]}.{compute_capability[1]} is below recommended 6.0."
                    }, priority=1)
            else:
                self.logger.warning("No GPU detected. Performance may be slower on CPU.")
                self.events.emit("ai_diagnostic", {
                    "message": "No GPU detected. Using CPU, which may result in slower performance."
                }, priority=1)
        except Exception as e:
            self.logger.error(f"Hardware compatibility check failed: {str(e)}")
            self.report_issue(f"AI: Hardware compatibility check failed: {str(e)}")

    def initialize(self):
        """Initialize AI processor with model and tokenizer, optimized for performance."""
        try:
            if not self.config:
                raise ValueError("Configuration not set")

            self.events.emit("process_step", {
                "step": "Checking hardware compatibility",
                "category": "Thinking",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

            # Check if CPU is forced
            force_cpu = self.config.get('ai_engine', 'force_cpu', default=False)
            if force_cpu:
                self.device = torch.device('cpu')
                self.logger.info("Forcing CPU usage as per configuration")
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.logger.info(f"Device in use: {self.device}")
            self.check_hardware_compatibility()

            if self.device.type == 'cuda':
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.info("Using CPU for inference")

            self.events.emit("ai_status", {"ai_status": "yellow"}, priority=1)  # Loading

            if self.use_mock:
                self.logger.info("Using mock model for testing...")
                self.initialized = True
                self.events.emit("ai_initialized", {"using_mock": self.use_mock, "model_name": "mock"}, priority=1)
                self.events.emit("ai_status", {"ai_status": "green"}, priority=1)
                self.logger.debug("Emitted ai_initialized event")
                self.log_diagnostic_state("Mock AI processor initialized")
                return

            self.model_path = self.config.get('ai_engine', 'model_path', default='distilgpt2')  # Use smaller model
            self.logger.info(f"Loading AI model from {self.model_path}...")

            self.events.emit("process_step", {
                "step": f"Loading AI model from {self.model_path}",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
            self.logger.debug(f"Using cache directory: {cache_dir}")

            try:
                disk_usage = psutil.disk_usage(cache_dir)
                self.logger.debug(f"Cache directory disk usage: Total {disk_usage.total / (1024**3):.2f} GB, Used {disk_usage.used / (1024**3):.2f} GB, Free {disk_usage.free / (1024**3):.2f} GB")
            except Exception as e:
                self.logger.warning(f"Could not check disk usage for cache directory: {str(e)}")

            # Check if transformers is available
            if transformers is None:
                self.logger.error("Transformers library not found. Please ensure it is installed.")
                raise ImportError("Transformers library not found")
            self.logger.debug(f"Transformers version: {transformers.__version__}")
            self.logger.debug(f"PyTorch version: {torch.__version__}")
            self.logger.debug(f"Python version: {sys.version}")

            model_dir = os.path.join(cache_dir, self.model_path)
            self.logger.debug(f"Checking if model files exist at: {model_dir}")
            if os.path.exists(model_dir):
                self.logger.debug("Model files found in cache")
            else:
                self.logger.debug("Model files not found in cache, will attempt to download")

            try:
                self.logger.debug("Loading model...")
                start_time = time.time()
                self.model = GPT2LMHeadModel.from_pretrained(
                    self.model_path,
                    cache_dir=cache_dir,
                    local_files_only=os.path.exists(model_dir)
                )
                elapsed_time = time.time() - start_time
                self.logger.debug(f"Model loading took {elapsed_time:.2f} seconds")

                self.logger.debug("Loading tokenizer...")
                start_time = time.time()
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    self.model_path,
                    cache_dir=cache_dir,
                    local_files_only=os.path.exists(model_dir)
                )
                elapsed_time = time.time() - start_time
                self.logger.debug(f"Tokenizer loading took {elapsed_time:.2f} seconds")

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.logger.debug("Set pad_token to eos_token to resolve attention mask issue")

                # Apply optimizations
                self.model.eval()
                if self.device.type == 'cuda':
                    self.model.to(self.device)
                    self.model.half()  # FP16 for speed
                    # Attempt quantization if available
                    try:
                        self.model = torch.quantization.quantize_dynamic(
                            self.model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        self.logger.info("Applied dynamic quantization to model (8-bit integers)")
                    except Exception as e:
                        self.logger.warning(f"Quantization failed: {str(e)}. Proceeding without quantization.")
                    torch.cuda.empty_cache()
                else:
                    self.model.to(self.device)

                self.initialized = True
                self.logger.info("AI processor initialized with real model")
            except Exception as e:
                self.logger.error(f"Failed to load real model: {str(e)}\n{traceback.format_exc()}")
                self.logger.warning("Falling back to mock model due to failure in loading real model")
                self.use_mock = True
                self.initialized = True
                self.model = None
                self.tokenizer = None

            self.events.emit("ai_initialized", {"using_mock": self.use_mock, "model_name": self.model_path}, priority=1)
            self.events.emit("ai_status", {"ai_status": "green" if self.initialized else "red"}, priority=1)
            self.logger.debug("Emitted ai_initialized event")
            self.log_diagnostic_state("AI processor initialized")
        except Exception as e:
            self.logger.error(f"AI initialization failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("AI: Failed to initialize model.")
            self.initialized = False
            if self.events:
                self.events.emit("ai_initialized", {"using_mock": True, "failed": True, "model_name": "unknown"}, priority=1)
                self.events.emit("ai_status", {"ai_status": "red"}, priority=1)
                self.logger.debug("Emitted ai_initialized event with failure status")

    def start(self):
        """Start the AI processor with error handling."""
        try:
            self.events.emit("process_step", {
                "step": "Starting AI processor",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            if not self.events:
                raise ValueError("Event handler not set")
            self.running = True
            self.initialize()
            if not self.initialized:
                self.logger.warning("AI processor started without model - responses disabled")
            else:
                self.logger.info("AI processor started")
            self.log_diagnostic_state("AI processor started")
        except Exception as e:
            self.logger.error(f"AI startup failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("AI: Failed to start processor.")
            self.stop()

    def process_input(self, event_data: Dict):
        """Process incoming input from events and generate response in a separate thread."""
        try:
            self.logger.debug(f"Received input_received event with data: {event_data}")
            if not self.running:
                self.logger.warning("AI processor not running")
                return

            if not self.initialized:
                self.logger.warning("AI model not initialized - skipping input")
                self.events.emit("ai_response", {
                    "response": "AI not initialized yet.",
                    "timestamp": event_data.get('timestamp')
                }, priority=2)
                return

            input_text = event_data.get('text', '')
            if not input_text:
                self.logger.debug("Empty input received")
                return

            self.events.emit("process_step", {
                "step": f"Received input: {input_text}",
                "category": "Thinking",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

            # Check cache
            if input_text in self.response_cache:
                self.logger.debug(f"Cache hit for input: {input_text}")
                self.events.emit("process_step", {
                    "step": "Found response in cache",
                    "category": "Action",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }, priority=1)
                self.events.emit("ai_response", {
                    "response": self.response_cache[input_text],
                    "timestamp": event_data.get('timestamp')
                }, priority=2)
                return

            self.logger.info(f"Processing input: {input_text}")
            cpu_usage = self.process.cpu_percent(interval=0.1)
            memory_usage = self.process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Resource usage before response generation: CPU {cpu_usage}%, Memory {memory_usage:.2f} MB")

            self.events.emit("process_step", {
                "step": "Generating response",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)

            # Emit a "processing" event to inform the GUI
            self.events.emit("ai_processing", {
                "message": "Processing your input...",
                "timestamp": event_data.get('timestamp')
            }, priority=1)

            # Reset cancel event
            self.cancel_event.clear()
            response_queue = queue.Queue()
            self.current_task = threading.Thread(
                target=self._generate_response_thread,
                args=(input_text, response_queue, event_data.get('timestamp'))
            )
            self.current_task.daemon = True
            self.current_task.start()

            try:
                response = response_queue.get(timeout=15)  # Reduced to 15 seconds
                # Cache the response
                self.response_cache[input_text] = response
                # Limit cache size (configurable)
                max_cache_size = self.config.get('ai_engine', 'cache_size', default=100)
                if len(self.response_cache) > max_cache_size:
                    self.response_cache.pop(next(iter(self.response_cache)))
            except queue.Empty:
                if self.cancel_event.is_set():
                    self.logger.info("Response generation canceled by user")
                    response = "Response generation canceled."
                else:
                    self.logger.error("Response generation timed out after 15 seconds")
                    self.report_issue("AI: Response generation timed out.")
                    response = "Sorry, I couldn't respond in time. Try a shorter input or check your hardware."

            self.events.emit("ai_response", {
                "response": response,
                "timestamp": event_data.get('timestamp')
            }, priority=2)
            self.logger.debug(f"Emitted ai_response event with response: {response}")
            self.log_diagnostic_state("AI response generated and emitted")
        except Exception as e:
            self.logger.error(f"Input processing failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("AI: Failed to process input.")
            self.events.emit("ai_response", {
                "response": "An error occurred while processing your input.",
                "timestamp": event_data.get('timestamp')
            }, priority=2)

    def _generate_response_thread(self, text: str, response_queue: queue.Queue, timestamp: str):
        """Generate a response in a separate thread and put the result in the queue."""
        try:
            start_time = time.time()
            response = self.generate_response(text)
            if self.cancel_event.is_set():
                self.logger.info("Response generation canceled during processing")
                return
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Response generation took {elapsed_time:.2f} seconds")
            response_queue.put(response)

            cpu_usage = self.process.cpu_percent(interval=0.1)
            memory_usage = self.process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"Resource usage after generation: CPU {cpu_usage}%, Memory {memory_usage:.2f} MB")
        except Exception as e:
            self.logger.error(f"Response generation thread failed: {str(e)}\n{traceback.format_exc()}")
            response_queue.put("Sorry, I couldn't process that due to an error.")

    def generate_response(self, text: str) -> str:
        """Generate a response using the AI model with detailed logging and optimizations."""
        try:
            if self.use_mock:
                self.logger.debug("Generating mock response...")
                time.sleep(1)
                return f"Mock response to: {text}"

            self.logger.debug("Starting response generation...")
            start_time = time.time()
            self.logger.debug("Tokenizing input...")
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Tokenization took {elapsed_time:.2f} seconds")
            self.logger.debug(f"Tokenized input: {inputs}")

            start_time = time.time()
            self.logger.debug("Generating response with model...")
            with torch.no_grad():  # Disable gradient computation for inference
                if self.device.type == 'cuda':
                    with autocast("cuda"):  # FP16 inference
                        outputs = self.model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_length=10,
                            num_return_sequences=1,
                            do_sample=True,
                            top_k=40,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                else:
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=10,
                        num_return_sequences=1,
                        do_sample=True,
                        top_k=40,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Model generation took {elapsed_time:.2f} seconds")
            self.logger.debug("Response generated successfully")

            start_time = time.time()
            self.logger.debug("Decoding response...")
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Decoding took {elapsed_time:.2f} seconds")
            self.logger.debug(f"Decoded response: {response}")

            if response.startswith(text):
                response = response[len(text):].strip()
            final_response = response.strip() or "I understood your input, but I don’t have a meaningful response."
            self.logger.debug(f"Final response: {final_response}")

            # Clean up memory
            del inputs, outputs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return final_response
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("AI: Failed to generate response.")
            return "Sorry, I couldn't process that."

    def cancel_task(self, data: Dict):
        """Cancel the current response generation task."""
        try:
            self.logger.info("Received cancel_task event")
            self.events.emit("process_step", {
                "step": "Canceling current response generation task",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            self.cancel_event.set()
            if self.current_task and self.current_task.is_alive():
                self.logger.debug("Waiting for current task to cancel...")
                self.current_task.join(timeout=1)
            self.current_task = None
            self.events.emit("ai_processing", {
                "message": "Processing canceled.",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
        except Exception as e:
            self.logger.error(f"Failed to cancel task: {str(e)}\n{traceback.format_exc()}")

    def stop(self):
        """Stop the AI processor and clean up with error handling."""
        try:
            self.events.emit("process_step", {
                "step": "Stopping AI processor and cleaning up resources",
                "category": "Action",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, priority=1)
            self.running = False
            self.cancel_event.set()
            if self.current_task and self.current_task.is_alive():
                self.current_task.join(timeout=1)
            if self.model and self.device.type == 'cuda':
                del self.model
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.initialized = False
            self.response_cache.clear()
            self.events.emit("ai_status", {"ai_status": "red"}, priority=1)
            self.logger.info("AI processor stopped")
            self.log_diagnostic_state("AI processor stopped")
        except Exception as e:
            self.logger.error(f"AI shutdown failed: {str(e)}\n{traceback.format_exc()}")
            self.report_issue("AI: Failed to stop processor.")

    def report_issue(self, message: str):
        """Report an issue and emit a diagnostic event."""
        try:
            self.logger.error(message)
            if self.events:
                self.events.emit("ai_diagnostic", {
                    "message": message,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }, priority=1)
            self.logger.debug(f"Reported issue: {message}")
        except Exception as e:
            self.logger.error(f"Failed to report issue: {str(e)}\n{traceback.format_exc()}")

    def log_diagnostic_state(self, context: str):
        """Log the current state of the AIProcessor for diagnostic purposes."""
        try:
            state = {
                "context": context,
                "running": self.running,
                "initialized": self.initialized,
                "device": self.device.type if self.device else "not_set",
                "event_subscriptions": [event_name for event_name, _ in self.event_subscriptions],
                "using_mock": self.use_mock,
                "cache_size": len(self.response_cache)
            }
            self.logger.debug(f"Diagnostic state: {state}")
            if self.events:
                self.events.emit("ai_diagnostic_state", state, priority=1)
        except Exception as e:
            self.logger.error(f"Diagnostic state logging failed: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    from serenity.utils.config_manager import ConfigManager
    from serenity.utils.event_handler import EventHandler

    try:
        ai = AIProcessor()
        config = ConfigManager()
        events = EventHandler()
        ai.set_config(config)
        ai.set_events(events)
        ai.start()

        events.emit("input_received", {"text": "Hello, Serenity!", "timestamp": "now"})
        time.sleep(2)
        ai.stop()

    except Exception as e:
        logging.getLogger('Serenity.AI').error(f"Test failed: {str(e)}")