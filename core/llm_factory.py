"""
LLM factory implementation following SOLID principles.
Uses dependency injection and proper abstraction layers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
from core.config import IConfigProvider, LLMConfig
from core.container import container
from llms.base import BaseLLM


class ILLMFactory(ABC):
    """Interface for LLM factory following Interface Segregation Principle."""

    @abstractmethod
    def create_llm(self, provider: str, config: LLMConfig) -> BaseLLM:
        """Create an LLM instance for the specified provider."""
        pass

    @abstractmethod
    def register_provider(self, provider: str, llm_class: Type[BaseLLM]) -> None:
        """Register an LLM implementation for a provider."""
        pass

    @abstractmethod
    def get_supported_providers(self) -> list[str]:
        """Get list of supported LLM providers."""
        pass

    @abstractmethod
    def is_provider_supported(self, provider: str) -> bool:
        """Check if a provider is supported."""
        pass


class LLMFactory(ILLMFactory):
    """LLM factory implementation using dependency injection."""

    def __init__(self, config_provider: IConfigProvider):
        self._config_provider = config_provider
        self._creators: Dict[str, Type[BaseLLM]] = {}
        self._register_default_providers()

    def create_llm(self, provider: str, config: LLMConfig) -> BaseLLM:
        """Create an LLM instance for the specified provider."""
        if not self.is_provider_supported(provider):
            raise ValueError(f"Unsupported LLM provider: {provider}")

        llm_class = self._creators[provider]
        
        try:
            # Try to create with config parameter
            return llm_class(config)
        except TypeError:
            # Fallback to creating with model_name parameter
            return llm_class(config.model_name)

    def register_provider(self, provider: str, llm_class: Type[BaseLLM]) -> None:
        """Register an LLM implementation for a provider."""
        if not issubclass(llm_class, BaseLLM):
            raise ValueError(f"LLM class must inherit from BaseLLM: {llm_class}")
        
        self._creators[provider] = llm_class

    def get_supported_providers(self) -> list[str]:
        """Get list of supported LLM providers."""
        return list(self._creators.keys())

    def is_provider_supported(self, provider: str) -> bool:
        """Check if a provider is supported."""
        return provider in self._creators

    def _register_default_providers(self) -> None:
        """Register default LLM providers with lazy loading."""
        # Register providers that are always available
        self._register_core_providers()
        
        # Register optional providers with lazy loading
        self._register_optional_providers()

    def _register_core_providers(self) -> None:
        """Register core LLM providers that are always available."""
        try:
            from llms.manual_llm import ManualLLM
            self.register_provider("manual", ManualLLM)
        except ImportError:
            pass

        try:
            from llms.mock_llm import MockLLM
            self.register_provider("mock", MockLLM)
        except ImportError:
            pass

    def _register_optional_providers(self) -> None:
        """Register optional LLM providers with lazy loading."""
        # OpenAI provider
        try:
            from llms.openai_llm import OpenAI_LLM
            self.register_provider("openai", OpenAI_LLM)
        except ImportError:
            pass

        # HuggingFace provider
        try:
            from llms.huggingface_llm import HuggingFaceLLM
            self.register_provider("huggingface", HuggingFaceLLM)
        except ImportError:
            pass

        # OpenRouter provider
        try:
            from llms.openrouter_llm import OpenRouter_LLM
            self.register_provider("openrouter", OpenRouter_LLM)
        except ImportError:
            pass

        # Llama.cpp provider
        try:
            from llms.llamacpp_llm import LlamaCPP_LLM
            self.register_provider("llamacpp", LlamaCPP_LLM)
        except ImportError:
            pass

        # Ollama provider
        try:
            from llms.ollama_llm import Ollama_LLM
            self.register_provider("ollama", Ollama_LLM)
        except ImportError:
            pass


def create_llm_factory(config_provider: IConfigProvider) -> LLMFactory:
    """Factory function to create and configure an LLM factory."""
    return LLMFactory(config_provider)
