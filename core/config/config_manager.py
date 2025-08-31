"""
Configuration management system following SOLID principles.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TaskConfig:
    """Configuration for a benchmark task."""
    name: str
    dataset: str
    config: str
    description: str


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    quantization: str = "none"
    temperature: float = 0.0
    max_tokens: int = 4096
    stream_tokens: bool = True
    unlimited_mode: bool = True


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    tasks: Dict[str, TaskConfig]
    default_task: str
    num_samples: int
    compression_methods: list
    target_ratio: float
    unlimited_mode: bool
    stream_tokens: bool


class IConfigProvider(ABC):
    """Interface for configuration providers."""

    @abstractmethod
    def get_llm_config(self, provider: str) -> LLMConfig:
        """Get LLM configuration for a provider."""
        pass

    @abstractmethod
    def get_benchmark_config(self) -> BenchmarkConfig:
        """Get benchmark configuration."""
        pass

    @abstractmethod
    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get configuration for a specific task."""
        pass


class EnvironmentConfigProvider(IConfigProvider):
    """Configuration provider that reads from environment and config files."""

    def __init__(self):
        self._task_configs = {
            "reasoning": TaskConfig(
                name="reasoning",
                dataset="gsm8k",
                config="main",
                description="Mathematical reasoning with GSM8K dataset"
            ),
            "summarization": TaskConfig(
                name="summarization",
                dataset="cnn_dailymail",
                config="3.0.0",
                description="News article summarization"
            ),
            "classification": TaskConfig(
                name="classification",
                dataset="imdb",
                config="plain_text",
                description="Sentiment classification on movie reviews"
            )
        }

    def get_llm_config(self, provider: str) -> LLMConfig:
        """Get LLM configuration for a provider."""
        configs = {
            "openai": LLMConfig(
                provider="openai",
                model_name="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                temperature=0.0,
                max_tokens=4096
            ),
            "huggingface": LLMConfig(
                provider="huggingface",
                model_name="microsoft/Phi-3.5-mini-instruct",
                quantization="none",
                temperature=0.0,
                max_tokens=4096
            ),
            "openrouter": LLMConfig(
                provider="openrouter",
                model_name="deepseek/deepseek-r1:free",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                temperature=0.0,
                max_tokens=4096
            )
        }

        if provider not in configs:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return configs[provider]

    def get_benchmark_config(self) -> BenchmarkConfig:
        """Get benchmark configuration."""
        return BenchmarkConfig(
            tasks=self._task_configs,
            default_task="reasoning",
            num_samples=int(os.getenv("NUM_SAMPLES", "1")),
            compression_methods=["llmlingua2"],
            target_ratio=0.8,
            unlimited_mode=os.getenv("UNLIMITED_MODE", "true").lower() == "true",
            stream_tokens=os.getenv("STREAM_TOKENS", "true").lower() == "true"
        )

    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get configuration for a specific task."""
        if task_name not in self._task_configs:
            raise ValueError(f"Unsupported task: {task_name}")
        return self._task_configs[task_name]


# Global configuration instance
config_provider = EnvironmentConfigProvider()