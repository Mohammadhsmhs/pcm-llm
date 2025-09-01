from .base import BaseLLM
from .manual_llm import ManualLLM
from .openai_llm import OpenAI_LLM
from .huggingface_llm import HuggingFace_LLM
from .llamacpp_llm import LlamaCpp_LLM
from .openrouter_llm import OpenRouter_LLM
from .ollama_llm import Ollama_LLM
from core.config import settings


class LLMFactory:
    """
    Factory for creating LLM handler instances based on the provider.
    """

    @staticmethod
    def create(provider: str) -> BaseLLM:
        if provider != "ollama":
            raise ValueError(f"Only Ollama provider is supported. Got: '{provider}'")
            
        llm_config = settings.get_llm_config(provider)
        
        return Ollama_LLM(llm_config)


