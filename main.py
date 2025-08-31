#!/usr/bin/env python3
"""
PCM-LLM: Prompt Compression Benchmark Tool

A comprehensive benchmarking tool for evaluating prompt compression methods
across different language models and tasks.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core functionality
from core.cli import main

if __name__ == "__main__":
    main()
