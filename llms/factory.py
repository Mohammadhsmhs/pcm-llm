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
                n_gpu_layers=settings.performance.n_gpu_layers if hasattr(settings.performance, 'n_gpu_layers') else -1,
                n_threads=settings.performance.n_threads if hasattr(settings.performance, 'n_threads') else 4,
            )
        elif provider == "openrouter":
            return OpenRouter_LLM(model_name=llm_config.model_name, api_key=llm_config.api_key)
        elif provider == "ollama":
            return Ollama_LLM(
                model_name=llm_config.model_name,
                temperature=llm_config.temperature,
                top_k=settings.generation.top_k if hasattr(settings.generation, 'top_k') else 40,
                num_ctx=llm_config.max_tokens,
                repeat_penalty=settings.generation.repeat_penalty if hasattr(settings.generation, 'repeat_penalty') else 1.1,
                repeat_last_n=settings.generation.repeat_last_n if hasattr(settings.generation, 'repeat_last_n') else 64,
            )
        else:
            raise ValueError(f"Unknown LLM provider in config: '{provider}'")


