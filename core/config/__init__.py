"""
Configuration management system following SOLID principles.
Uses centralized settings with proper abstraction layers.
"""

from .config_manager import (
    BenchmarkConfig,
    CentralizedConfigProvider,
    IConfigProvider,
    LLMConfig,
    TaskConfig,
    config_provider,
)
from .settings import (
    CompressionSettings,
    EvaluationSettings,
    LLMSettings,
    PathSettings,
    PerformanceSettings,
    Settings,
    TaskSettings,
    settings,
)

__all__ = [
    "TaskConfig",
    "LLMConfig",
    "BenchmarkConfig",
    "IConfigProvider",
    "CentralizedConfigProvider",
    "config_provider",
    "Settings",
    "TaskSettings",
    "LLMSettings",
    "CompressionSettings",
    "EvaluationSettings",
    "PerformanceSettings",
    "PathSettings",
    "settings",
]
