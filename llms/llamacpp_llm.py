import os
import time
from typing import Optional

from llms.base import BaseLLM
from config import STREAM_TOKENS
from transformers import AutoTokenizer, AutoModelForCausalLM # For Selective Context


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
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
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

        # Note: When installed with Metal support, llama.cpp will use GPU offloading on macOS.
        # n_gpu_layers=-1 attempts to offload all layers to GPU.
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
        # Add structured output instruction
        structured_prompt = (
            prompt
            + "\n\nPlease provide your final answer in this exact format: #### [final_answer_number]"
        )

        print(
            "\n🤖 Generating response with llama.cpp (streaming)..."
            if STREAM_TOKENS
            else "\n🤖 Generating response with llama.cpp..."
        )

        # Stream tokens as they are generated and collect the final text
        full_text_parts = []
        
        # Adjust timeout and max_tokens based on prompt length
        prompt_length = len(structured_prompt.split())
        if prompt_length > 1000:  # Very long prompts
            timeout_seconds = 180  # 3 minutes for very long prompts
            max_tokens = 256  # Shorter responses for long inputs
            print(f"📝 Long prompt detected ({prompt_length} words) - using extended timeout")
        elif prompt_length > 500:  # Long prompts
            timeout_seconds = 120  # 2 minutes for long prompts
            max_tokens = 512
        else:  # Normal prompts
            timeout_seconds = 60   # 1 minute for normal prompts
            max_tokens = 1024
        
        try:
            # create_completion uses an OpenAI-like schema; stream=True yields chunks
            stream = self.llm.create_completion(
                prompt=structured_prompt,
                temperature=0.3,
                top_p=0.9,
                max_tokens=max_tokens,
                stream=True,
                stop=None,
            )
            
            start_time = time.time()
            tokens_generated = 0
            
            for chunk in stream:
                # Check for timeout
                if time.time() - start_time > timeout_seconds:
                    print(f"⚠️  Generation timeout after {timeout_seconds}s ({tokens_generated} tokens generated)")
                    break
                    
                token = chunk.get("choices", [{}])[0].get("text", "")
                if token:
                    tokens_generated += 1
                    if STREAM_TOKENS:
                        print(token, end="", flush=True)
                    full_text_parts.append(token)
                    
        except Exception as e:
            error_msg = str(e).lower()
            if "kv cache" in error_msg or "context" in error_msg or "overflow" in error_msg:
                print(f"⚠️  Context/KV cache issue detected: {e}")
                print("   This usually means the context window is too large for the model.")
                print("   Try reducing LLAMACPP_N_CTX in config.py (current: 2048)")
                return f"Error: Context overflow - reduce LLAMACPP_N_CTX or use a model with larger context window. Details: {e}"
            else:
                return f"Error during llama.cpp generation: {e}"

        if STREAM_TOKENS:
            print()  # newline after streaming
            
        response = "".join(full_text_parts).strip()
        
        # If response is empty or too short due to timeout, provide fallback
        if not response or len(response.split()) < 3:
            return f"Error: Generation failed or timed out. Prompt was {prompt_length} words long, generated {tokens_generated} tokens."
            
        return response
