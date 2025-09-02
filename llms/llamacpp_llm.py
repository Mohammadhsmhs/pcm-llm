import os
from typing import Optional

from llms.base.base import BaseLLM


class LlamaCpp_LLM(BaseLLM):
    """
    LLM backend using llama.cpp via llama-cpp-python.
    - Runs 4-bit (and other) GGUF models locally on macOS with Metal.
    - Streams tokens to stdout during generation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        n_ctx: int = 40960,  # Reduced from 40960 to prevent memory issues
        n_gpu_layers: int = -1,  # Disable GPU layers to avoid Metal issues
        n_threads: Optional[int] = None,
    ):
        super().__init__(model_path or (repo_id + ":" + filename if repo_id and filename else "llama-cpp"))

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise ImportError(
                "llama-cpp-python is not installed. Add it to requirements and pip install it."
            ) from e

        if n_threads is None:
            try:
                n_threads = max(1, (os.cpu_count() or 4) - 1)
            except Exception:
                n_threads = 4

        if model_path and os.path.isfile(model_path):
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                logits_all=False,
                verbose=False,
            )
            origin = f"local file: {model_path}"
        elif repo_id and filename:
            if not hasattr(Llama, "from_pretrained"):
                raise RuntimeError(
                    "Your llama-cpp-python version lacks from_pretrained(); upgrade to >=0.2.90 or provide a local model_path."
                )
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                logits_all=False,
                verbose=False,
            )
            origin = f"HF Hub: {repo_id}/{filename}"
        else:
            raise FileNotFoundError(
                "Provide either a valid local GGUF model_path or repo_id+filename for HF Hub."
            )

        print(
            f"Initialized llama.cpp model from {origin}\n"
            f"  Context: {n_ctx} | n_gpu_layers: {n_gpu_layers} | n_threads: {n_threads}"
        )

    def get_response(self, prompt: str) -> str:
        print(f"\n🤖 Generating response for prompt: {prompt[:100]}...")

        try:
            # Use greedy sampling and proper parameters to prevent repetition
            from core.config import settings
            
            temperature = getattr(settings.generation, 'temperature', 0.0) if hasattr(settings.generation, 'temperature') else 0.0
            max_tokens = getattr(settings.performance, 'max_tokens', 1024) if hasattr(settings.performance, 'max_tokens') else 1024
            
            response = self.llm.create_completion(
                prompt=prompt,
                temperature=temperature,  # Greedy sampling (equivalent to --top-k 1)
                top_p=1.0,
                top_k=1,          # Greedy sampling
                max_tokens=max_tokens,  # Reasonable limit to prevent infinite generation
                stream=False,     # Use non-streaming for reliability
                stop=None,        # No stop tokens for now
                repeat_penalty=1.0,  # No repetition penalty (can cause issues)
            )

            result = response["choices"][0]["text"]
            print(f"📝 Generated {len(result)} characters")
            print(f"📝 Response preview: {result[:200]}...")

            return result

        except Exception as e:
            return f"Error in generation: {e}"
