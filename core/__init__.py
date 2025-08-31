"""
Core package for PCM-LLM benchmark execution.
"""

from .benchmark_executor import BenchmarkExecutor
from .cli import main

__all__ = ['BenchmarkExecutor', 'main']
