import os
from typing import Optional

from core.config import LLMConfig
from llms.base.base import BaseLLM
from utils.prompt_utils import add_structured_instructions


class LlamaCPPLLM(BaseLLM):
    """
    LLM backend using llama.cpp via llama-cpp-python.
    - Runs 4-bit (and other) GGUF models locally on macOS with Metal.
    - Streams tokens to stdout during generation.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config.model_name or "llama-cpp")
        self.config = config

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. Add it to requirements and pip install it."
            ) from e

        n_threads = self.config.num_threads or self._get_default_threads()

        if self.config.model_path and os.path.isfile(self.config.model_path):
            self.llm = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.max_tokens,
                n_gpu_layers=self.config.gpu_layers,
                n_threads=n_threads,
                logits_all=False,
                verbose=False,
            )
            origin = f"local file: {self.config.model_path}"
        elif self.config.repo_id and self.config.repo_filename:
            if not hasattr(Llama, "from_pretrained"):
                raise RuntimeError(
                    "Your llama-cpp-python version lacks from_pretrained(); upgrade to >=0.2.90 or provide a local model_path."
                )
            self.llm = Llama.from_pretrained(
                repo_id=self.config.repo_id,
                filename=self.config.repo_filename,
                n_ctx=self.config.max_tokens,
                n_gpu_layers=self.config.gpu_layers,
                n_threads=n_threads,
                logits_all=False,
                verbose=False,
            )
            origin = f"HF Hub: {self.config.repo_id}/{self.config.repo_filename}"
        else:
            raise FileNotFoundError(
                "Provide either a valid local GGUF model_path or repo_id+filename for HF Hub."
            )

        print(
            f"Initialized llama.cpp model from {origin}\n"
            f"  Context: {self.config.max_tokens} | n_gpu_layers: {self.config.gpu_layers} | n_threads: {n_threads}"
        )

    def _get_default_threads(self) -> int:
        try:
            return max(1, (os.cpu_count() or 4) - 1)
        except Exception:
            return 4

    def get_response(self, prompt: str, task_type: str = "reasoning") -> str:
        print(f"\n🤖 Generating response for prompt: {prompt[:100]}...")

        # Add task-specific structured instructions
        structured_prompt = add_structured_instructions(prompt, task_type)

        try:
            # Use greedy sampling and proper parameters to prevent repetition
            response = self.llm.create_completion(
                prompt=structured_prompt,
                temperature=self.config.temperature,
                top_p=1.0,
                top_k=1,
                max_tokens=self.config.max_tokens,
                stream=False,
                stop=None,
                repeat_penalty=1.0,
            )

            result = response["choices"][0]["text"]
            print(f"📝 Generated {len(result)} characters")
            print(f"📝 Response preview: {result[:200]}...")

            return result

        except Exception as e:
            return f"Error in generation: {e}"
