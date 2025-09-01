"""
Configuration management system following SOLID principles.
Uses centralized settings with proper abstraction layers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

from .settings import settings, TaskSettings, LLMSettings


@dataclass
class TaskConfig:
    """Task configuration data transfer object."""
    name: str
    dataset: str
    config: str
    description: str


@dataclass
class LLMConfig:
    """LLM configuration data transfer object."""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 300
    rate_limit_rpm: Optional[int] = None


@dataclass
class BenchmarkConfig:
    """Benchmark configuration data transfer object."""
    default_task: str
    tasks: Dict[str, TaskConfig]
    compression_methods: List[str]
    target_ratio: float
    num_samples: int
    unlimited_mode: bool
    enable_checkpointing: bool


class IConfigProvider(ABC):
    """Interface for configuration providers following Interface Segregation Principle."""

    @abstractmethod
    def get_benchmark_config(self) -> BenchmarkConfig:
        """Get benchmark configuration."""
        pass

    @abstractmethod
    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get configuration for a specific task."""
        pass

    @abstractmethod
    def get_llm_config(self, provider: str) -> LLMConfig:
        """Get configuration for a specific LLM provider."""
        pass

    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task names."""
        pass

    @abstractmethod
    def get_supported_llm_providers(self) -> List[str]:
        """Get list of supported LLM provider names."""
        pass


class CentralizedConfigProvider(IConfigProvider):
    """Configuration provider that uses centralized settings."""

    def __init__(self):
        self._settings = settings

    def get_benchmark_config(self) -> BenchmarkConfig:
        """Get benchmark configuration from centralized settings."""
        return BenchmarkConfig(
            default_task=self._settings.default_task,
            tasks={
                name: TaskConfig(
                    name=task.name,
                    dataset=task.dataset,
                    config=task.config,
                    description=task.description
                )
                for name, task in self._settings.tasks.items()
                if task.enabled
            },
            compression_methods=self._settings.get_compression_methods(),
            target_ratio=self._settings.get_target_ratio(),
            num_samples=self._settings.performance.num_samples,
            unlimited_mode=self._settings.evaluation.unlimited_mode,
            enable_checkpointing=self._settings.evaluation.enable_checkpointing
        )

    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get configuration for a specific task."""
        task_settings = self._settings.get_task_config(task_name)
        return TaskConfig(
            name=task_settings.name,
            dataset=task_settings.dataset,
            config=task_settings.config,
            description=task_settings.description
        )

    def get_llm_config(self, provider: str) -> LLMConfig:
        """Get configuration for a specific LLM provider."""
        llm_settings = self._settings.get_llm_config(provider)
        return LLMConfig(
            provider=llm_settings.provider,
            model_name=llm_settings.model_name,
            api_key=llm_settings.api_key,
            temperature=llm_settings.temperature,
            max_tokens=llm_settings.max_tokens,
            timeout=llm_settings.timeout,
            rate_limit_rpm=llm_settings.rate_limit_rpm
        )

    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task names."""
        return self._settings.get_supported_tasks()

    def get_supported_llm_providers(self) -> List[str]:
        """Get list of supported LLM provider names."""
        return self._settings.get_supported_llm_providers()


# Default configuration provider instance
config_provider: IConfigProvider = CentralizedConfigProvider()