"""
Centralized configuration settings for PCM-LLM.
Follows the 12-factor app methodology and environment-based configuration.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TaskSettings:
    """Configuration for individual tasks."""

    name: str
    dataset: str
    config: str
    description: str
    enabled: bool = True


@dataclass
class LLMSettings:
    """Configuration for LLM providers."""

    provider: str
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 300
    rate_limit_rpm: Optional[int] = None
    stream_tokens: bool = False
    quantization: Optional[str] = None
    top_k: int = 40
    repetition_penalty: float = 1.1


@dataclass
class CompressionSettings:
    """Configuration for compression methods."""

    methods: List[str] = field(
        default_factory=lambda: ["llmlingua2", "selective_context", "naive_truncation"]
    )
    target_ratio: float = 0.75
    naive_truncation_model: str = "bert-base-uncased"


@dataclass
class EvaluationSettings:
    """Configuration for evaluation features."""

    enable_qualitative_analysis: bool = True
    enable_style_aware_scoring: bool = True
    enable_checkpointing: bool = True
    unlimited_mode: bool = True


@dataclass
class PerformanceSettings:
    """Configuration for performance optimization."""

    num_samples: int = 3
    batch_size_base: int = 5
    adaptive_batch_size: bool = True
    max_concurrent_loggers: int = 1
    memory_checkpoint_interval: int = 50
    use_jsonl: bool = False
    compress_intermediate: bool = False


@dataclass
class PathSettings:
    """Configuration for file paths."""

    results_dir: str = "results"
    cache_dir: str = "compressed_cache"
    logs_dir: str = "logs"
    models_dir: str = "models"


class Settings:
    """Centralized settings manager."""

    def __init__(self):
        # Load all settings in one pass
        self._load_settings()

    def _load_settings(self):
        """Load all settings from environment variables and defaults."""
        # Environment variables with defaults
        self.default_task = os.getenv("PCM_DEFAULT_TASK", "reasoning")
        self.default_llm_provider = "ollama"  # Always use Ollama
        self.num_samples = int(os.getenv("PCM_NUM_SAMPLES", "5"))
        self.unlimited_mode = os.getenv("PCM_UNLIMITED_MODE", "true").lower() == "true"

        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.hf_token = os.getenv("HF_TOKEN", "")

        # Configure HuggingFace model cache
        self._configure_huggingface_cache()

        # Task configurations
        self.tasks = self._create_task_configs()

        # LLM configurations
        self.llm_providers = self._create_llm_configs()

        # Other settings
        self.compression = CompressionSettings(
            target_ratio=float(os.getenv("PCM_COMPRESSION_TARGET_RATIO", "0.75"))
        )
        self.evaluation = EvaluationSettings(unlimited_mode=self.unlimited_mode)
        self.performance = PerformanceSettings(num_samples=self.num_samples)
        self.paths = PathSettings()

    def _configure_huggingface_cache(self):
        """Configure HuggingFace cache directories."""
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.getenv("HF_HOME"):
            os.environ["HF_HOME"] = models_dir
        if not os.getenv("TRANSFORMERS_CACHE"):
            os.environ["TRANSFORMERS_CACHE"] = models_dir

    def _create_task_configs(self) -> Dict[str, TaskSettings]:
        """Create task configurations."""
        return {
            "reasoning": TaskSettings(
                name="reasoning",
                dataset="gsm8k",
                config="main",
                description="Mathematical reasoning with GSM8K dataset",
            ),
            "summarization": TaskSettings(
                name="summarization",
                dataset="cnn_dailymail",
                config="3.0.0",
                description="News article summarization",
            ),
            "classification": TaskSettings(
                name="classification",
                dataset="imdb",
                config="plain_text",
                description="Sentiment classification on movie reviews",
            ),
        }

    def _create_llm_configs(self) -> Dict[str, LLMSettings]:
        """Create LLM provider configurations."""
        return {
            "ollama": LLMSettings(
                provider="ollama",
                model_name=os.getenv(
                    "PCM_OLLAMA_MODEL", "hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q8_0"
                ),
                timeout=600,
                stream_tokens=os.getenv("PCM_STREAM_TOKENS", "true").lower() == "true",
            ),
            "manual": LLMSettings(provider="manual", model_name="manual", timeout=0),
            "mock": LLMSettings(provider="mock", model_name="mock-model", timeout=0),
            "openai": LLMSettings(
                provider="openai",
                model_name="gpt-4o",
                api_key=self.openai_api_key,
                rate_limit_rpm=20,
            ),
            "openrouter": LLMSettings(
                provider="openrouter",
                model_name="deepseek/deepseek-chat",
                api_key=self.openrouter_api_key,
                rate_limit_rpm=16,
            ),
            "huggingface": LLMSettings(
                provider="huggingface",
                model_name="microsoft/Phi-3-mini-4k-instruct",
                api_key=self.hf_token,
                quantization="4bit",
            ),
            "llamacpp": LLMSettings(
                provider="llamacpp",
                model_name="microsoft/Phi-3-mini-4k-instruct-gguf",
                api_key=self.hf_token,
            ),
        }

    # Property-based accessors to eliminate redundant methods
    @property
    def supported_tasks(self) -> List[str]:
        """Get list of supported task names."""
        return list(self.tasks.keys())

    @property
    def supported_llm_providers(self) -> List[str]:
        """Get list of supported LLM provider names."""
        return list(self.llm_providers.keys())

    @property
    def compression_methods(self) -> List[str]:
        """Get list of compression methods."""
        return self.compression.methods

    @property
    def target_ratio(self) -> float:
        """Get target compression ratio."""
        return self.compression.target_ratio

    def get_task_config(self, task_name: str) -> TaskSettings:
        """Get configuration for a specific task."""
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")
        return self.tasks[task_name]

    def get_llm_config(self, provider: str) -> LLMSettings:
        """Get configuration for a specific LLM provider."""
        if provider not in self.llm_providers:
            raise ValueError(f"Unknown LLM provider: {provider}")
        return self.llm_providers[provider]

    def is_task_enabled(self, task_name: str) -> bool:
        """Check if a task is enabled."""
        return task_name in self.tasks and self.tasks[task_name].enabled


# Global settings instance
settings = Settings()
