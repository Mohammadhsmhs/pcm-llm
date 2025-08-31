from .base import BaseLLM
from .manual_llm import ManualLLM
from .openai_llm import OpenAI_LLM
from .huggingface_llm import HuggingFace_LLM
from .llamacpp_llm import LlamaCpp_LLM
from .openrouter_llm import OpenRouter_LLM
from .ollama_llm import Ollama_LLM
from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    HUGGINGFACE_MODEL,
    HUGGINGFACE_QUANTIZATION,
    DEFAULT_LLM_PROVIDER,
    LLAMACPP_MODEL_PATH,
    LLAMACPP_N_CTX,
    LLAMACPP_N_GPU_LAYERS,
    LLAMACPP_N_THREADS,
    LLAMACPP_REPO_ID,
    LLAMACPP_FILENAME,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_THREAD,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_K,
    OLLAMA_TOP_P,
    OLLAMA_NUM_PREDICT,
)


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
        elif provider == "llamacpp":
            return LlamaCpp_LLM(
                model_path=LLAMACPP_MODEL_PATH if LLAMACPP_REPO_ID == "" else None,
                repo_id=LLAMACPP_REPO_ID or None,
                filename=LLAMACPP_FILENAME or None,
                n_ctx=LLAMACPP_N_CTX,
                n_gpu_layers=LLAMACPP_N_GPU_LAYERS,
                n_threads=LLAMACPP_N_THREADS,
            )
        elif provider == "openrouter":
            return OpenRouter_LLM(model_name=OPENROUTER_MODEL, api_key=OPENROUTER_API_KEY)
        elif provider == "ollama":
            return Ollama_LLM(
                model_name=OLLAMA_MODEL,
                temperature=OLLAMA_TEMPERATURE,
                top_k=OLLAMA_TOP_K,
                num_ctx=OLLAMA_NUM_CTX,
                repeat_penalty=1.1,  # Default repeat penalty
                repeat_last_n=64,    # Default repeat last n
            )
        else:
            raise ValueError(f"Unknown LLM provider in config: '{provider}'")


