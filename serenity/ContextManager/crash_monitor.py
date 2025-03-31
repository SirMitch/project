# File: serenity\ContextManager\crash_monitor.py
# Version: 1.2.6  # Updated with new features and compatibility with context_manager.py Version 1.6.7
# Created: 2025-03-27
# Updated: 2025-10-15
# Description: Monitors context_manager.py for crashes and performs cleanup if necessary.
#              Includes detailed error handling, diagnostics, adaptive sleep, timeout enforcement,
#              console output, hang detection, and full compatibility with context_manager.py Version 1.6.7.
#              Operates from any root directory under \ContextManager (e.g., G:\project\serenity\ContextManager).
# Usage:
#   python crash_monitor.py <pid> [context_manager_dir] [--sleep-interval SEC] [--log-level LEVEL] [--retry-attempts N]
#   [--retry-delay SEC] [--debug-mode] [--timeout SEC] [--diagnostic-interval SEC] [--max-cpu-percent PCT]
#   [--hang-detection-threshold SEC] [--disable-hang-detection] [--no-timeout] [--skip-missing-backups]
#   [--version] [--status-interval SEC] [--exit-code-on-success N]
# Dependencies: psutil

import os
import json
import time
import sys
import shutil
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

class CrashMonitor:
    def __init__(self, pid: int, context_manager_dir: str, config: Dict[str, Any] = None):
        """
        Initialize the Crash Monitor with enhanced features.

        Args:
            pid (int): PID of the context_manager.py process to monitor.
            context_manager_dir (str): Directory where context_manager.py files reside.
            config (Dict[str, Any], optional): Configuration settings.
        """
        self.context_manager_dir = os.path.abspath(context_manager_dir)
        if not os.path.exists(self.context_manager_dir):
            os.makedirs(self.context_manager_dir, exist_ok=True)
        self.pid = int(pid)
        self.transaction_file = os.path.join(self.context_manager_dir, "transaction.json")
        self.locks_file = os.path.join(self.context_manager_dir, "locks.json")
        self.signal_file = os.path.join(self.context_manager_dir, "context_manager_done.signal")
        self.lock_file = os.path.join(self.context_manager_dir, "context_manager.lock")
        self.log_file = os.path.join(self.context_manager_dir, "context_manager.log")  # Unified logging
        self.temp_dir = os.path.join(self.context_manager_dir, "temp")

        # Configuration aligned with context_manager.py defaults, extended with new options
        self.config = {
            "sleep_interval": 5.0,
            "log_level": "INFO",
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "debug_mode": False,
            "timeout": 3600,  # 1 hour, adjustable for API mode
            "max_cpu_percent": 80.0,
            "diagnostic_interval": 60.0,
            "hang_detection_threshold": 60.0,  # Configurable hang detection wait time
            "hang_detection_enabled": True,    # Default to enabled
            "skip_missing_backups": False,     # Default to not skipping
            "no_timeout": False,               # Default to timeout enabled
            "status_interval": 300.0,          # New: Log status every 5 minutes by default
            "exit_code_on_success": 0          # New: Default success exit code
        }
        if config:
            self.config.update(config)

        self.setup_logging()
        self.running = True
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        self.logger.info(f"CrashMonitor initialized for PID {self.pid} in {self.context_manager_dir}")
        print(f"CrashMonitor initialized for PID {self.pid} in {self.context_manager_dir}")

    def setup_logging(self) -> None:
        """Configure logging to append to context_manager.log with rotation."""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                self.log_file, maxBytes=5*1024*1024, backupCount=3
            )
            handler.setFormatter(logging.Formatter("[CrashMonitor] %(asctime)s - %(levelname)s - %(message)s"))
            logging.basicConfig(level=getattr(logging, self.config["log_level"].upper()), handlers=[handler])
            self.logger = logging.getLogger()
            self.logger.info(f"Logging set up with level {self.config['log_level']} and rotation")
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            sys.exit(1)

    def is_process_alive(self) -> bool:
        """Check if the monitored process is still running."""
        try:
            return psutil.pid_exists(self.pid)
        except Exception as e:
            self.logger.error(f"Error checking PID {self.pid}: {e}")
            return False

    def detect_hang(self) -> bool:
        """Detect if the process is hung based on low CPU usage over a configurable period."""
        if not self.config["hang_detection_enabled"]:
            return False
        try:
            process = psutil.Process(self.pid)
            cpu_percent = process.cpu_percent(interval=1.0)
            memory_info = process.memory_info()
            if cpu_percent < 1.0 and memory_info.rss / (1024 * 1024) > 0:  # Low CPU, non-zero memory
                self.logger.debug(f"Low CPU usage detected: {cpu_percent}%")
                time.sleep(self.config["hang_detection_threshold"])
                if process.cpu_percent(interval=1.0) < 1.0:
                    self.logger.warning(f"Hang detected: CPU usage {cpu_percent}% for {self.config['hang_detection_threshold']}s")
                    return True
            return False
        except psutil.NoSuchProcess:
            return False
        except Exception as e:
            self.logger.error(f"Error in hang detection: {e}")
            return False

    def load_transaction(self) -> Dict[str, Any]:
        """Load the transaction state with retry logic."""
        if not os.path.exists(self.transaction_file):
            self.logger.warning(f"No transaction file found at {self.transaction_file}")
            return {"state": "idle", "backups": {}}
        for attempt in range(self.config["retry_attempts"]):
            try:
                with open(self.transaction_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{self.config['retry_attempts']}: Failed to load transaction: {e}")
                if attempt < self.config["retry_attempts"] - 1:
                    time.sleep(self.config["retry_delay"])
                else:
                    self.logger.error("Failed to load transaction.json after retries")
                    return {"state": "failed", "backups": {}}

    def validate_backup(self, backup_path: str) -> bool:
        """Validate that a backup file exists and is non-empty."""
        if not os.path.exists(backup_path):
            return False
        if os.path.getsize(backup_path) == 0:
            self.logger.warning(f"Backup file {backup_path} is empty")
            return False
        return True

    def cleanup(self) -> bool:
        """Perform cleanup operations on crash detection."""
        self.logger.info("Initiating cleanup due to crash detection")
        print("Performing cleanup...")
        success = True
        transaction = self.load_transaction()

        # Restore backups
        for filepath, backup_path in transaction.get("backups", {}).items():
            try:
                if self.validate_backup(backup_path):
                    shutil.copy2(backup_path, filepath)
                    self.logger.info(f"Restored {filepath} from {backup_path}")
                    print(f"Restored {filepath} from backup")
                else:
                    if self.config["skip_missing_backups"]:
                        self.logger.warning(f"Backup {backup_path} missing or invalid; skipping restoration.")
                        continue
                    else:
                        self.logger.error(f"Backup file {backup_path} not found or invalid for {filepath}")
                        success = False
            except Exception as e:
                self.logger.error(f"Failed to restore {filepath}: {e}")
                success = False

        # Reset locks
        for attempt in range(self.config["retry_attempts"]):
            try:
                with open(self.locks_file, "w") as f:
                    json.dump({}, f)
                self.logger.info("Reset locks.json")
                break
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{self.config['retry_attempts']}: Failed to reset locks: {e}")
                if attempt == self.config["retry_attempts"] - 1:
                    success = False
                time.sleep(self.config["retry_delay"])

        # Remove lock file
        if os.path.exists(self.lock_file):
            try:
                os.remove(self.lock_file)
                self.logger.info("Removed context_manager.lock")
            except Exception as e:
                self.logger.error(f"Failed to remove lock file: {e}")
                success = False

        # Clear temp directory
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info("Cleared temp directory")
            except Exception as e:
                self.logger.error(f"Failed to clear temp directory: {e}")
                success = False

        self.logger.info(f"Cleanup completed {'successfully' if success else 'with errors'}")
        print(f"Cleanup completed {'successfully' if success else 'with errors'}")
        return success

    def write_crash_status(self, reason: str) -> None:
        """Write crash status to a file for context_manager.py to check on restart."""
        status_file = os.path.join(self.context_manager_dir, "crash_status.json")
        status = {
            "message": f"Crash detected: {reason}",
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(status_file, "w") as f:
                json.dump(status, f, indent=4)
            self.logger.info(f"Wrote crash status: {reason}")
        except Exception as e:
            self.logger.error(f"Failed to write crash status: {e}")

    def check_signal(self) -> bool:
        """Check for completion signal with retry logic and timestamp verification."""
        for attempt in range(self.config["retry_attempts"]):
            try:
                if os.path.exists(self.signal_file):
                    with open(self.signal_file, "r") as f:
                        content = f.read().strip()
                    if content.startswith("done:"):
                        timestamp = content.split("done:")[1]
                        signal_time = datetime.fromisoformat(timestamp)
                        if (datetime.now() - signal_time).total_seconds() < 300:  # 5-minute freshness
                            self.logger.info("Received fresh completion signal from context_manager.py")
                            print("context_manager.py completed successfully. Crash monitor shutting down.")
                            os.remove(self.signal_file)
                            self.logger.debug("Removed signal file")
                            return True
                        else:
                            self.logger.warning(f"Stale signal detected: {timestamp}")
                    return False
                return False
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{self.config['retry_attempts']}: Error checking signal file {self.signal_file}: {e}")
                if attempt < self.config["retry_attempts"] - 1:
                    time.sleep(self.config["retry_delay"])
                else:
                    return False

    def release_locks(self) -> None:
        """Release any remaining locks on successful completion."""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
                self.logger.info("Released context_manager.lock on completion")
        except Exception as e:
            self.logger.error(f"Failed to release lock file: {e}")

    def log_diagnostics(self) -> None:
        """Log detailed diagnostics in debug mode."""
        try:
            process = psutil.Process(self.pid)
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            elapsed_time = time.time() - self.start_time
            self.logger.debug(
                f"Diagnostics: PID {self.pid}, CPU: {cpu_percent:.2f}%, "
                f"Memory: {memory_info.rss / (1024 * 1024):.2f} MB, Elapsed: {elapsed_time:.2f}s"
            )
        except psutil.NoSuchProcess:
            self.logger.debug(f"Process {self.pid} no longer exists during diagnostics")
        except Exception as e:
            self.logger.error(f"Error logging diagnostics: {e}")

    def log_process_status(self) -> None:
        """Log basic process status at regular intervals."""
        try:
            process = psutil.Process(self.pid)
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            self.logger.info(
                f"Process Status: PID {self.pid}, CPU: {cpu_percent:.2f}%, "
                f"Memory: {memory_info.rss / (1024 * 1024):.2f} MB"
            )
        except psutil.NoSuchProcess:
            self.logger.info(f"Process {self.pid} no longer exists during status check")
        except Exception as e:
            self.logger.error(f"Error logging process status: {e}")

    def adaptive_sleep(self) -> None:
        """Sleep with adaptive interval based on CPU usage."""
        try:
            process = psutil.Process(self.pid)
            cpu_percent = process.cpu_percent(interval=0.1)
            sleep_time = self.config["sleep_interval"] / 2 if cpu_percent > self.config["max_cpu_percent"] else self.config["sleep_interval"]
            time.sleep(sleep_time)
        except psutil.NoSuchProcess:
            time.sleep(self.config["sleep_interval"])
        except Exception as e:
            self.logger.error(f"Error in adaptive sleep: {e}")
            time.sleep(self.config["sleep_interval"])

    def run(self) -> None:
        """Main monitoring loop with timeout, crash, and hang detection."""
        self.logger.info(f"Starting monitoring for PID {self.pid} with timeout {self.config['timeout']}s")
        print(f"Monitoring context_manager.py (PID: {self.pid})...")
        last_diagnostic_time = time.time()
        last_status_time = time.time()

        while self.running:
            if not self.config["no_timeout"]:
                elapsed_time = time.time() - self.start_time
                if elapsed_time > self.config["timeout"]:
                    self.logger.error(f"Timeout of {self.config['timeout']}s exceeded. Assuming crash.")
                    print(f"Timeout exceeded. Performing cleanup...")
                    self.cleanup()
                    self.write_crash_status("timeout")
                    sys.exit(1)

            if not self.is_process_alive():
                self.logger.warning(f"PID {self.pid} no longer exists.")
                print(f"Process terminated. Initiating cleanup...")
                self.cleanup()
                self.write_crash_status("process_terminated")
                sys.exit(0 if self.cleanup() else 1)

            if self.config["hang_detection_enabled"] and self.detect_hang():
                self.logger.error("Hang detected in context_manager.py. Performing cleanup...")
                print("Hang detected. Performing cleanup...")
                self.cleanup()
                self.write_crash_status("hang_detected")
                sys.exit(1)

            if self.check_signal():
                self.logger.info("ContextManager completed successfully.")
                self.release_locks()
                sys.exit(self.config["exit_code_on_success"])

            if self.config["debug_mode"] and (time.time() - last_diagnostic_time) >= self.config["diagnostic_interval"]:
                self.log_diagnostics()
                last_diagnostic_time = time.time()

            if (time.time() - last_status_time) >= self.config["status_interval"]:
                self.log_process_status()
                last_status_time = time.time()

            self.adaptive_sleep()

def parse_arguments():
    """Parse command-line arguments for CrashMonitor."""
    parser = argparse.ArgumentParser(description="Crash Monitor for context_manager.py")
    parser.add_argument("pid", type=int, help="PID of context_manager.py to monitor")
    parser.add_argument("context_manager_dir", nargs="?", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory containing context_manager.py files (default: script dir)")
    parser.add_argument("--sleep-interval", type=float, default=5.0, help="Base sleep interval in seconds")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--retry-attempts", type=int, default=3, help="Number of retry attempts")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Delay between retries in seconds")
    parser.add_argument("--debug-mode", action="store_true", help="Enable debug diagnostics")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds before assuming crash")
    parser.add_argument("--diagnostic-interval", type=float, default=60.0,
                        help="Interval for diagnostic logging in debug mode")
    parser.add_argument("--max-cpu-percent", type=float, default=80.0,
                        help="CPU percent threshold for adaptive sleep")
    parser.add_argument("--hang-detection-threshold", type=float, default=60.0,
                        help="Time in seconds to wait before confirming a hang")
    parser.add_argument("--disable-hang-detection", action="store_true", help="Disable hang detection")
    parser.add_argument("--no-timeout", action="store_true", help="Disable timeout")
    parser.add_argument("--skip-missing-backups", action="store_true", help="Skip restoration if backups are missing")
    parser.add_argument("--version", action="version", version="crash_monitor.py 1.2.6")
    parser.add_argument("--status-interval", type=float, default=300.0,
                        help="Interval for logging process status in seconds")
    parser.add_argument("--exit-code-on-success", type=int, default=0,
                        help="Exit code on successful completion")
    return parser.parse_args()

def check_permissions(directory: str) -> None:
    """Check if the script has write permissions to the specified directory."""
    if not os.access(directory, os.W_OK):
        logging.error(f"Insufficient permissions to write to {directory}")
        print(f"Error: Insufficient permissions to write to {directory}")
        sys.exit(1)

if __name__ == "__main__":
    # Delayed import of psutil to allow installation if missing
    try:
        import psutil
    except ImportError:
        print("psutil not found. Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            import psutil
        except Exception as e:
            print(f"Failed to install psutil: {e}. Please install manually: pip install psutil")
            sys.exit(1)

    args = parse_arguments()
    check_permissions(args.context_manager_dir)
    config = {
        "sleep_interval": args.sleep_interval,
        "log_level": args.log_level,
        "retry_attempts": args.retry_attempts,
        "retry_delay": args.retry_delay,
        "debug_mode": args.debug_mode,
        "timeout": args.timeout,
        "diagnostic_interval": args.diagnostic_interval,
        "max_cpu_percent": args.max_cpu_percent,
        "hang_detection_threshold": args.hang_detection_threshold,
        "hang_detection_enabled": not args.disable_hang_detection,
        "skip_missing_backups": args.skip_missing_backups,
        "no_timeout": args.no_timeout,
        "status_interval": args.status_interval,
        "exit_code_on_success": args.exit_code_on_success
    }
    monitor = CrashMonitor(args.pid, args.context_manager_dir, config)
    try:
        monitor.run()
    except KeyboardInterrupt:
        monitor.logger.info("Received KeyboardInterrupt. Shutting down...")
        print("Shutting down CrashMonitor...")
        monitor.running = False
        sys.exit(0)
    except Exception as e:
        monitor.logger.error(f"Unexpected error in CrashMonitor: {e}")
        print(f"CrashMonitor failed: {e}")
        monitor.cleanup()
        monitor.write_crash_status("unexpected_error")
        sys.exit(1)