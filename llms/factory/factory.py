from abc import ABC, abstractmethod

from core.config import LLMConfig, settings
from llms.base import BaseLLM
from llms.providers import (
    HuggingFaceLLM,
    LlamaCPPLLM,
    ManualLLM,
    OllamaLLM,
    OpenAILLM,
    OpenRouterLLM,
)


class ILLMFactory(ABC):
    @abstractmethod
    def create_llm(self, provider: str, config: LLMConfig) -> BaseLLM: ...


class LLMFactory(ILLMFactory):
    """
    Factory for creating LLM handler instances based on the provider.
    """

    def create_llm(self, provider: str, config: LLMConfig) -> BaseLLM:
        if provider == "openai":
            return OpenAILLM(config)
        elif provider == "huggingface":
            return HuggingFaceLLM(config)
        elif provider == "llamacpp":
            return LlamaCPPLLM(config)
        elif provider == "openrouter":
            return OpenRouterLLM(config)
        elif provider == "ollama":
            return OllamaLLM(config)
        elif provider == "manual":
            return ManualLLM(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
