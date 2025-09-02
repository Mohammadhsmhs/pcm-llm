"""
LLM factory implementation following SOLID principles.
Uses dependency injection and proper abstraction layers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
from core.config import IConfigProvider, LLMConfig
from core.container import container
from llms.base.base import BaseLLM


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

        # Special handling for HuggingFaceLLM which needs device_service
        if provider == "huggingface":
            from core.device_service import DeviceService
            device_service = DeviceService()
            return llm_class(config, device_service)
            
        try:
            # All other providers are expected to be initializable with just the config
            return llm_class(config)
        except Exception as e:
            print(f"Error creating LLM for provider {provider}: {e}")
            raise

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
        self._register_core_providers()
        self._register_optional_providers()

    def _register_core_providers(self) -> None:
        """Register core LLM providers that are always available."""
        core_providers = {
            "manual": "llms.providers.manual_llm.ManualLLM",
            "mock": "llms.providers.mock_llm.MockLLM",
        }
        for provider, path in core_providers.items():
            try:
                module_path, class_name = path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                llm_class = getattr(module, class_name)
                self.register_provider(provider, llm_class)
            except ImportError as e:
                print(f"Warning: Core provider {provider} could not be loaded: {e}")
                pass

    def _register_optional_providers(self) -> None:
        """Register optional LLM providers with lazy loading."""
        providers = {
            "ollama": "llms.providers.ollama_llm.OllamaLLM",
            "huggingface": "llms.providers.huggingface_llm.HuggingFaceLLM",
            "llamacpp": "llms.providers.llamacpp_llm.LlamaCPPLLM",
            "openai": "llms.providers.openai_llm.OpenAILLM",
            "openrouter": "llms.providers.openrouter_llm.OpenRouterLLM",
        }

        for provider, path in providers.items():
            try:
                module_path, class_name = path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                llm_class = getattr(module, class_name)
                self.register_provider(provider, llm_class)
            except ImportError as e:
                print(f"Warning: {provider} LLM provider could not be loaded: {e}")
                pass


def create_llm_factory(config_provider: IConfigProvider) -> LLMFactory:
    """Factory function to create and configure an LLM factory."""
    return LLMFactory(config_provider)
