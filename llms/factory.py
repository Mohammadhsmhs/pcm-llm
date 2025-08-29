from .base import BaseLLM
from .manual_llm import ManualLLM
from .openai_llm import OpenAI_LLM
from .huggingface_llm import HuggingFace_LLM
from config import OPENAI_API_KEY, OPENAI_MODEL, HUGGINGFACE_MODEL, HUGGINGFACE_QUANTIZATION


class LLMFactory:
    """
    Factory for creating LLM handler instances based on the provider.
    """

    @staticmethod
    def create(provider: str) -> BaseLLM:
        if provider == "manual":
            return ManualLLM()
        elif provider == "openai":
            return OpenAI_LLM(model_name=OPENAI_MODEL, api_key=OPENAI_API_KEY)
        elif provider == "huggingface":
            return HuggingFace_LLM(model_name=HUGGINGFACE_MODEL, quantization=HUGGINGFACE_QUANTIZATION)
        else:
            raise ValueError(f"Unknown LLM provider in config: '{provider}'")


