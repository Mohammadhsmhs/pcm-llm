"""
Thread-safe logging utilities for concurrent benchmark execution.
"""

import threading
from config import MAX_CONCURRENT_LOGGERS


class ThreadSafeLogger:
    """Thread-safe logger using semaphore."""

    def __init__(self, logger_instance):
        self.logger = logger_instance
        self.semaphore = threading.Semaphore(MAX_CONCURRENT_LOGGERS)

    def log_result(self, result_data):
        """Thread-safe logging."""
        with self.semaphore:
            self.logger.log_result(result_data)
