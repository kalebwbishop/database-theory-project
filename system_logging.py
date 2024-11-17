import psutil
import time
import threading
import os

class SystemLogging:
    def __init__(self, file_name: str):
        # Ensure the logs directory exists within /tmp
        self.logs_dir = "/tmp/local_logs"
        os.makedirs(self.logs_dir, exist_ok=True)

        self.file_name = file_name
        self.keep_logging = False
        self.thread = None

    def get_system_defaults(self):
        self.cpu_usage = 0
        self.memory_info = 0
        self.disk_info = 0

        count = 5
        for _ in range(count):
            self.cpu_usage += psutil.cpu_percent(interval=1)
            self.memory_info += psutil.virtual_memory().percent
            self.disk_info += psutil.disk_usage('/').percent

        self.cpu_usage /= count
        self.memory_info /= count
        self.disk_info /= count

    def _log_system_usage(self):
        """Internal method to log system usage periodically."""
        log_file_path = os.path.join(self.logs_dir, f"{self.file_name}.log")
        with open(log_file_path, "w") as f:
            f.write("System Usage Log\n")
        while self.keep_logging:
            # Calculate deltas
            cpu_usage = psutil.cpu_percent(interval=1) - self.cpu_usage
            memory_info = psutil.virtual_memory().percent - self.memory_info
            disk_info = psutil.disk_usage('/').percent - self.disk_info

            # Write to log file
            with open(log_file_path, "a") as f:
                f.write(f"System Usage - CPU: {cpu_usage:.2f}%, Memory: {memory_info:.2f}%, Disk: {disk_info:.2f}%\n")
            time.sleep(1)  # Log every second

    def start_logging(self):
        """Start the logging in a separate thread."""
        self.get_system_defaults()

        if not self.keep_logging:
            self.keep_logging = True
            self.thread = threading.Thread(target=self._log_system_usage, daemon=True)
            self.thread.start()

    def stop_logging(self):
        """Stop the logging and wait for the thread to finish."""
        self.keep_logging = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def write_output(self, output):
        """Manually write additional output to the log file."""
        log_file_path = os.path.join(self.logs_dir, f"{self.file_name}.log")
        with open(log_file_path, "a") as f:
            f.write(f"\n{output}\n")
