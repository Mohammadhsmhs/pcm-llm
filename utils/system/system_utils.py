"""
System utilities for memory management and system monitoring.
"""

import psutil
import torch

from core.config import settings


def clear_memory():
    """Utility function to clear GPU cache if available."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def calculate_adaptive_batch_size(base_batch_size=None):
    """Calculate adaptive batch size based on available memory."""
    if base_batch_size is None:
        base_batch_size = settings.performance.batch_size_base

    if not settings.performance.adaptive_batch_size:
        return base_batch_size

    available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
    memory_usage = get_memory_usage()

    # Reserve some memory for system operations
    safe_memory = available_memory - 1024  # Leave 1GB for system

    if safe_memory < 2048:  # Less than 2GB available
        return max(1, base_batch_size // 4)
    elif safe_memory < 4096:  # Less than 4GB available
        return max(1, base_batch_size // 2)
    else:
        return base_batch_size


def log_memory_usage(phase_name, run_info_logger=None):
    """Log memory usage summary."""
    memory_mb = get_memory_usage()
    print(f"📊 Memory at {phase_name}: {memory_mb:.1f}MB")

    # Also log to run info logger if available
    if run_info_logger:
        run_info_logger.log_memory_usage(memory_mb)
