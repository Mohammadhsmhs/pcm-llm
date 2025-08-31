import os
from typing import Optional

from llms.base import BaseLLM


class LlamaCpp_Simple_LLM(BaseLLM):
    """
    Simple, unrestricted LLM backend using llama.cpp via llama-cpp-python.
    No token limits, no stop conditions - just raw model output.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        n_ctx: int = 2048,  # Reduced from 40960 to prevent memory issues
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
    ):
        super().__init__(model_path or (repo_id + ":" + filename if repo_id and filename else "llama-cpp-simple"))

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
                n_gpu_layers=0,  # Disable GPU layers to avoid Metal issues
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
                n_gpu_layers=0,  # Disable GPU layers to avoid Metal issues
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
            f"Initialized SIMPLE llama.cpp model from {origin}\n"
            f"  Context: {n_ctx} | n_gpu_layers: {n_gpu_layers} | n_threads: {n_threads}"
        )

    def get_response(self, prompt: str) -> str:
        print(f"\n🤖 SIMPLE: Generating response for prompt: {prompt[:100]}...")

        try:
            # FIXED: Use greedy sampling and proper parameters to prevent repetition
            # Based on llama.cpp issue #12251 fixes
            response = self.llm.create_completion(
                prompt=prompt,
                temperature=0.0,  # Greedy sampling (equivalent to --top-k 1)
                top_p=1.0,
                top_k=1,          # Greedy sampling
                max_tokens=1024,  # Reasonable limit to prevent infinite generation
                stream=False,
                stop=None,
                repeat_penalty=1.0,  # No repetition penalty (can cause issues)
            )

            result = response["choices"][0]["text"]
            print(f"📝 SIMPLE: Generated {len(result)} characters")
            print(f"📝 SIMPLE: Response preview: {result[:200]}...")

            return result

        except Exception as e:
            return f"Error in simple generation: {e}"
