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
        llm_config = settings.get_llm_config(provider)
        
        if provider == "manual":
            return ManualLLM()
        elif provider == "openai":
            return OpenAI_LLM(model_name=llm_config.model_name, api_key=llm_config.api_key)
        elif provider == "huggingface":
            return HuggingFace_LLM(model_name=llm_config.model_name, quantization="none")  # Default quantization
        elif provider == "llamacpp":
            return LlamaCpp_LLM(
                model_path=None,  # Will be handled by the class
                repo_id=llm_config.model_name.split('/')[0] if '/' in llm_config.model_name else None,
                filename=llm_config.model_name.split('/')[-1] if '/' in llm_config.model_name else llm_config.model_name,
                n_ctx=llm_config.max_tokens,
                n_gpu_layers=1,  # Default
                n_threads=4,     # Default
            )
        elif provider == "openrouter":
            return OpenRouter_LLM(model_name=llm_config.model_name, api_key=llm_config.api_key)
        elif provider == "ollama":
            return Ollama_LLM(
                model_name=llm_config.model_name,
                temperature=llm_config.temperature,
                top_k=40,  # Default
                num_ctx=llm_config.max_tokens,
                repeat_penalty=1.1,  # Default repeat penalty
                repeat_last_n=64,    # Default repeat last n
            )
        else:
            raise ValueError(f"Unknown LLM provider in config: '{provider}'")


