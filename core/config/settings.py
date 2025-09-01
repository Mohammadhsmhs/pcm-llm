"""
Centralized configuration settings for PCM-LLM.
Follows the 12-factor app methodology and environment-based configuration.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass
class CompressionSettings:
    """Configuration for compression methods."""
    methods: List[str] = field(default_factory=lambda: ["llmlingua2", "selective_context", "naive_truncation"])
    target_ratio: float = 0.9
    naive_truncation_model: str = "bert-base-uncased"


@dataclass
class EvaluationSettings:
    """Configuration for evaluation features."""
    enable_qualitative_analysis: bool = True
    enable_style_aware_scoring: bool = True
    enable_checkpointing: bool = True
    unlimited_mode: bool = False


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
        self._load_environment_settings()
        self._load_default_settings()
    
    def _load_environment_settings(self):
        """Load settings from environment variables."""
        # Task configuration
        self.default_task = os.getenv("PCM_DEFAULT_TASK", "reasoning")
        
        # LLM configuration
        self.default_llm_provider = os.getenv("PCM_DEFAULT_LLM_PROVIDER", "ollama")
        
        # Performance configuration
        self.num_samples = int(os.getenv("PCM_NUM_SAMPLES", "3"))
        self.unlimited_mode = os.getenv("PCM_UNLIMITED_MODE", "false").lower() == "true"
        
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.hf_token = os.getenv("HF_TOKEN", "")
    
    def _load_default_settings(self):
        """Load default configuration values."""
        # Task configurations
        self.tasks = {
            "reasoning": TaskSettings(
                name="reasoning",
                dataset="gsm8k",
                config="main",
                description="Mathematical reasoning with GSM8K dataset"
            ),
            "summarization": TaskSettings(
                name="summarization",
                dataset="cnn_dailymail",
                config="3.0.0",
                description="News article summarization"
            ),
            "classification": TaskSettings(
                name="classification",
                dataset="imdb",
                config="plain_text",
                description="Sentiment classification on movie reviews"
            )
        }
        
        # LLM configurations
        self.llm_providers = {
            "openai": LLMSettings(
                provider="openai",
                model_name=os.getenv("PCM_OPENAI_MODEL", "gpt-3.5-turbo"),
                api_key=self.openai_api_key,
                rate_limit_rpm=60
            ),
            "huggingface": LLMSettings(
                provider="huggingface",
                model_name=os.getenv("PCM_HUGGINGFACE_MODEL", "microsoft/Phi-3.5-mini-instruct"),
                timeout=600
            ),
            "openrouter": LLMSettings(
                provider="openrouter",
                model_name=os.getenv("PCM_OPENROUTER_MODEL", "deepseek/deepseek-r1:free"),
                api_key=self.openrouter_api_key,
                rate_limit_rpm=16
            ),
            "ollama": LLMSettings(
                provider="ollama",
                model_name=os.getenv("PCM_OLLAMA_MODEL", "hf.co/Qwen/Qwen3-14B-GGUF:Q8_0"),
                timeout=600
            ),
            "llamacpp": LLMSettings(
                provider="llamacpp",
                model_name=os.getenv("PCM_LLAMACPP_MODEL", "Qwen/Qwen3-14B-GGUF"),
                timeout=600
            ),
            "manual": LLMSettings(
                provider="manual",
                model_name="manual",
                timeout=0
            ),
            "mock": LLMSettings(
                provider="mock",
                model_name="mock-model",
                timeout=0
            )
        }
        
        # Compression settings
        self.compression = CompressionSettings()
        
        # Evaluation settings
        self.evaluation = EvaluationSettings(
            unlimited_mode=self.unlimited_mode
        )
        
        # Performance settings
        self.performance = PerformanceSettings(
            num_samples=self.num_samples
        )
        
        # Path settings
        self.paths = PathSettings()
    
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
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task names."""
        return list(self.tasks.keys())
    
    def get_supported_llm_providers(self) -> List[str]:
        """Get list of supported LLM provider names."""
        return list(self.llm_providers.keys())
    
    def is_task_enabled(self, task_name: str) -> bool:
        """Check if a task is enabled."""
        return task_name in self.tasks and self.tasks[task_name].enabled
    
    def get_compression_methods(self) -> List[str]:
        """Get list of compression methods to run."""
        return self.compression.methods.copy()
    
    def get_target_ratio(self) -> float:
        """Get target compression ratio."""
        return self.compression.target_ratio


# Global settings instance
settings = Settings()
