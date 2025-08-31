"""
Refactored LLM Factory following SOLID principles with dependency injection.
"""

from typing import Dict, Type
from abc import ABC, abstractmethod
from llms.base import BaseLLM
from core.config import IConfigProvider, LLMConfig


class ILLMFactory(ABC):
    """Interface for LLM factory implementations."""

    @abstractmethod
    def create_llm(self, provider: str) -> BaseLLM:
        """Create an LLM instance for the given provider."""
        pass

    @abstractmethod
    def get_supported_providers(self) -> list[str]:
        """Get list of supported LLM providers."""
        pass


class LLMFactory(ILLMFactory):
    """Factory for creating LLM instances with dependency injection."""

    def __init__(self, config_provider: IConfigProvider):
        self.config_provider = config_provider
        self._creators: Dict[str, Type[BaseLLM]] = {}

    def register_provider(self, provider: str, creator: Type[BaseLLM]) -> None:
        """Register a provider with its creator class."""
        self._creators[provider] = creator

    def create_llm(self, provider: str) -> BaseLLM:
        """Create an LLM instance for the given provider."""
        if provider not in self._creators:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        llm_config = self.config_provider.get_llm_config(provider)
        creator_class = self._creators[provider]

        # Create the LLM instance with configuration
        return creator_class(llm_config)

    def get_supported_providers(self) -> list[str]:
        """Get list of supported LLM providers."""
        return list(self._creators.keys())


def create_llm_factory(config_provider: IConfigProvider) -> LLMFactory:
    """Factory function to create and configure an LLM factory."""
    factory = LLMFactory(config_provider)

    # Import and register LLM implementations
    try:
        from llms.openai_llm import OpenAI_LLM
        factory.register_provider("openai", OpenAI_LLM)
    except ImportError:
        pass

    try:
        from llms.huggingface_llm import HuggingFace_LLM
        factory.register_provider("huggingface", HuggingFace_LLM)
    except ImportError:
        pass

    try:
        from llms.openrouter_llm import OpenRouter_LLM
        factory.register_provider("openrouter", OpenRouter_LLM)
    except ImportError:
        pass

    try:
        from llms.manual_llm import ManualLLM
        factory.register_provider("manual", ManualLLM)
    except ImportError:
        pass

    return factory
